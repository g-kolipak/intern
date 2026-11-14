"""Thin Drex client. One function: ask(state, questions).

Docs: https://console.nace.ai/docs  (POST /v1/systemone, Bearer key)
Key comes from NACE_API_KEY. Metrics (latency, input tokens) are recorded
on every call, per the project spec.
"""

import os
import random
import time
from dataclasses import dataclass

import httpx

API_URL = "https://drex.nace.ai/v1/systemone"
DEFAULT_MODEL = "drex-v2.0"
VISION_MODEL = "drex-max-v1.1"  # reads images, up to 8 per request
RETRYABLE_STATUS = {429, 500, 529}


@dataclass
class CallResult:
    body: dict
    latency_ms: float
    input_tokens: int
    request_id: str
    model: str
    cost_usd: float | None = None  # set only when the API reports it


def ask(state, questions, model: str = DEFAULT_MODEL, key: str | None = None,
        timeout: float = 90.0, max_retries: int = 8,
        images: list[str] | None = None) -> CallResult:
    """Send one decision request to Drex and return the parsed response.

    state: any JSON-serializable value (text, dict, list).
    questions: {id: {type, instructions, criteria?}}.
    images: data URLs for a Drex Max model; questions call them Image 1..N.
    Raises DrexError on auth/validation failure (never retried).
    """
    key = key or os.environ.get("NACE_API_KEY")
    if not key:
        raise RuntimeError("NACE_API_KEY is not set")
    payload = {"model": model, "state": state, "questions": questions}
    if images:
        payload["images"] = images
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        t0 = time.monotonic()
        try:
            resp = httpx.post(
                API_URL,
                headers={"Authorization": f"Bearer {key}"},
                json=payload,
                timeout=timeout,
            )
        except httpx.TransportError as e:
            last_err = e
            _sleep_backoff(attempt)
            continue
        latency_ms = (time.monotonic() - t0) * 1000
        if resp.status_code in RETRYABLE_STATUS and attempt < max_retries:
            last_err = DrexError(resp.status_code, resp.text[:200])
            _sleep_backoff(attempt, resp)
            continue
        if resp.status_code != 200:
            raise DrexError(resp.status_code, resp.text[:500])
        body = resp.json()
        return CallResult(
            body=body,
            latency_ms=latency_ms,
            input_tokens=body["usage"]["input_tokens"],
            request_id=body.get("request_id", ""),
            model=body.get("model", ""),
        )
    raise DrexError(0, f"exhausted retries: {last_err}")


class DrexError(RuntimeError):
    def __init__(self, status: int, detail: str):
        super().__init__(f"Drex HTTP {status}: {detail}")
        self.status = status


def _sleep_backoff(attempt: int, resp: httpx.Response | None = None) -> None:
    retry_after = None
    if resp is not None:
        try:
            retry_after = int(resp.headers["retry-after-ms"]) / 1000
        except (ValueError, KeyError):
            pass
    base = retry_after if retry_after else min(2 ** attempt, 20)
    time.sleep(base + random.random() * 0.5)
