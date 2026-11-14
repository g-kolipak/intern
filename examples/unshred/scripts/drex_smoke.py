"""Step 1 smoke test: one real Drex yes/no call, print the raw response.

Reads NACE_API_KEY from the environment. Usage: uv run scripts/drex_smoke.py
"""

import json
import os
import sys

import httpx

API_URL = "https://drex.nace.ai/v1/systemone"
MODEL = "drex-v2.0"


def main() -> int:
    key = os.environ.get("NACE_API_KEY")
    if not key:
        print("NACE_API_KEY is not set", file=sys.stderr)
        return 1

    payload = {
        "model": MODEL,
        "state": "The quick brown fox jumps over the lazy dog.",
        "questions": {
            "is_continuous": {
                "type": "noul",
                "instructions": (
                    "Is the text in state one continuous, natural English "
                    "passage with no broken or missing words?"
                ),
            }
        },
    }

    resp = httpx.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}"},
        json=payload,
        timeout=90,
    )
    print("HTTP", resp.status_code)
    raw = resp.text
    print("RAW RESPONSE:")
    print(raw)
    try:
        body = resp.json()
        print("PARSED answer:", body.get("answers", {}).get("is_continuous"))
    except json.JSONDecodeError:
        print("(response was not JSON)")
    return 0 if resp.status_code == 200 else 2


if __name__ == "__main__":
    raise SystemExit(main())
