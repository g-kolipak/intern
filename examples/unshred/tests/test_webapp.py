"""Tests for unshred.webapp (no network, no pipeline: subprocess is faked)."""
import time

from fastapi.testclient import TestClient

from unshred import webapp


class FakeProc:
    def __init__(self, cmd, **kw):
        src = webapp.Path(cmd[-1])
        out = webapp.ROOT / "runs" / "photo" / src.stem
        out.mkdir(parents=True, exist_ok=True)
        (out / "rebuilt.jpg").write_bytes(b"\xff\xd8fake")
        (out / "steps").mkdir(exist_ok=True)
        (out / "steps" / "step1.jpg").write_bytes(b"\xff\xd8step")
        (out / "steps.json").write_text(
            '{"steps": [{"step": 1, "label": "find", "seconds": 1.0, '
            '"drex_calls": 0, "image": "step1.jpg"}], "total_seconds": 1.0}')
        self.stdout = iter(["  1.00 s  find: 2 candidate regions\n",
                            "  total 1.0 s\n"])

    def wait(self):
        return 0


def test_upload_runs_the_pipeline_and_serves_the_result(monkeypatch, tmp_path):
    monkeypatch.setattr(webapp, "ROOT", tmp_path)
    monkeypatch.setattr(webapp, "UPLOADS", tmp_path / "runs" / "uploads")
    monkeypatch.setattr(webapp.subprocess, "Popen", FakeProc)
    client = TestClient(webapp.app)
    jid = client.post("/api/rebuild", content=b"jpegbytes").json()["id"]
    for _ in range(50):
        job = client.get(f"/api/rebuild/{jid}").json()
        if job["status"] not in ("running", "queued"):
            break
        time.sleep(0.02)
    assert job["status"] == "done"
    assert any("find" in line for line in job["log"])
    img = client.get(f"/api/rebuild/{jid}/image")
    assert img.status_code == 200 and img.content.startswith(b"\xff\xd8")
    assert job["steps"][0]["image"] == "step1.jpg"
    assert client.get(f"/api/rebuild/{jid}/step/step1.jpg").status_code == 200
    assert client.get(f"/api/rebuild/{jid}/step/../x.jpg").status_code == 404
    runs = client.get("/api/history").json()["runs"]
    assert runs[0]["id"] == jid and runs[0]["status"] == "done"
    webapp.JOBS.clear()  # after a restart the run is still in the history
    assert client.get(f"/api/rebuild/{jid}").json()["status"] == "done"


def test_bad_requests_are_refused():
    client = TestClient(webapp.app)
    assert client.post("/api/rebuild", content=b"").status_code == 400
    assert client.get("/api/rebuild/nothere123").status_code == 404
    assert client.get("/api/rebuild/../../etc").status_code == 404
