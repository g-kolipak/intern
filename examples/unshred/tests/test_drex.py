"""Tests for unshred.drex (no network: httpx is monkeypatched)."""
import httpx
import pytest

from unshred.drex import API_URL as URL, DrexError, ask


def _fake_resp(status_code=200, body=None):
    body = body or {
        "answers": {"q": {"type": "noul", "noul": 0.9, "confidence": 0.9}},
        "usage": {"input_tokens": 10, "output_tokens": 1},
        "request_id": "r1",
        "model": "drex-v2.0",
    }
    return httpx.Response(status_code, json=body, request=httpx.Request("POST", URL))


def test_ask_returns_parsed_metrics(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "sk-test")
    calls = {}

    def fake_post(url, **kw):
        calls["url"] = url
        calls["headers"] = kw["headers"]
        calls["payload"] = kw["json"]
        return _fake_resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    res = ask("state", {"q": {"type": "noul", "instructions": "x?"}})
    assert calls["url"] == "https://drex.nace.ai/v1/systemone"
    assert calls["headers"]["Authorization"] == "Bearer sk-test"
    assert calls["payload"]["questions"] == {
        "q": {"type": "noul", "instructions": "x?"}}
    assert res.body["answers"]["q"]["noul"] == 0.9
    assert res.input_tokens == 10
    assert res.latency_ms >= 0
    assert res.model == "drex-v2.0"


def test_ask_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "sk-test")
    monkeypatch.setattr("time.sleep", lambda s: None)
    seq = [_fake_resp(429), _fake_resp(200)]

    def fake_post(url, **kw):
        return seq.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    res = ask("s", {"q": {"type": "noul", "instructions": "x?"}})
    assert res.input_tokens == 10
    assert seq == []


def test_ask_raises_on_401_without_retry(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "bad")
    seen = []

    def fake_post(url, **kw):
        seen.append(1)
        return httpx.Response(
            401, text="no", request=httpx.Request("POST", URL))

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(DrexError) as ei:
        ask("s", {"q": {"type": "noul", "instructions": "x?"}})
    assert ei.value.status == 401
    assert len(seen) == 1  # never retried


def test_ask_requires_key(monkeypatch):
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="NACE_API_KEY"):
        ask("s", {})


def test_ask_sends_images_only_when_given(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "sk-test")
    sent = []

    def fake_post(url, **kw):
        sent.append(kw["json"])
        return _fake_resp()

    monkeypatch.setattr(httpx, "post", fake_post)
    q = {"q": {"type": "noul", "instructions": "x?"}}
    ask("s", q)
    ask("s", q, model="drex-max-v1.1", images=["data:image/jpeg;base64,AA"])
    assert "images" not in sent[0]
    assert sent[1]["images"] == ["data:image/jpeg;base64,AA"]
    assert sent[1]["model"] == "drex-max-v1.1"
