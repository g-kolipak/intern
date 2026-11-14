"""Website: upload a photo of torn pieces, get the page Drex rebuilt.

  uv run --env-file .env uvicorn unshred.webapp:app --port 8765
  then open http://127.0.0.1:8765

Each upload runs scripts/rebuild_photo.py (every decision by Drex Max) in
the background. Every step saves a picture and its timing (steps.json),
which the page shows as Step 1, Step 2, ... Uploads stay on disk, so the
history survives restarts. The key comes from the server's environment.
"""

import json
import re
import subprocess
import sys
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent.parent
UPLOADS = ROOT / "runs" / "uploads"
MAX_BYTES = 40_000_000
JOBS: dict = {}
RUN_ONE = threading.Lock()  # one photo at a time: Drex limits concurrency
app = FastAPI(title="Unshred")


def _out(jid: str) -> Path:
    return ROOT / "runs" / "photo" / f"up_{jid}"


def _job(jid: str) -> dict:
    if not re.fullmatch(r"[0-9a-f]{10}", jid):
        raise HTTPException(404, "no such job")
    if jid not in JOBS:  # an upload from before a restart
        if not (UPLOADS / f"up_{jid}.jpg").exists():
            raise HTTPException(404, "no such job")
        done = (_out(jid) / "rebuilt.jpg").exists()
        JOBS[jid] = {"status": "done" if done else "failed", "log": []}
    return JOBS[jid]


def _steps(jid: str) -> dict:
    f = _out(jid) / "steps.json"
    return json.loads(f.read_text()) if f.exists() else {"steps": []}


def _run(jid: str, src: Path) -> None:
    """Run the pipeline on one upload (waiting for any earlier upload to
    finish first), keeping its progress lines."""
    job = JOBS[jid]
    with RUN_ONE:
        job["status"] = "running"
        _pipeline(jid, src, job)


def _pipeline(jid: str, src: Path, job: dict) -> None:
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "rebuild_photo.py"), str(src)],
        cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        job["log"].append(line.rstrip())
    code = proc.wait()
    done = code == 0 and (_out(jid) / "rebuilt.jpg").exists()
    job["status"] = "done" if done else "failed"


@app.get("/")
def page():
    return FileResponse(ROOT / "web" / "rebuild.html")


@app.post("/api/rebuild")
async def start(request: Request):
    body = await request.body()
    if not body or len(body) > MAX_BYTES:
        raise HTTPException(400, "send one image file, under 40 MB")
    jid = uuid.uuid4().hex[:10]
    UPLOADS.mkdir(parents=True, exist_ok=True)
    src = UPLOADS / f"up_{jid}.jpg"
    src.write_bytes(body)
    JOBS[jid] = {"status": "queued", "log": []}
    threading.Thread(target=_run, args=(jid, src), daemon=True).start()
    return {"id": jid}


@app.get("/api/rebuild/{jid}")
def status(jid: str):
    job = _job(jid)
    return {"id": jid, "status": job["status"], "log": job["log"][-60:],
            **_steps(jid)}


@app.get("/api/rebuild/{jid}/image")
def image(jid: str):
    if _job(jid)["status"] != "done":
        raise HTTPException(409, "not finished yet")
    return FileResponse(_out(jid) / "rebuilt.jpg", media_type="image/jpeg")


@app.get("/api/rebuild/{jid}/source")
def source(jid: str):
    _job(jid)
    return FileResponse(UPLOADS / f"up_{jid}.jpg", media_type="image/jpeg")


@app.get("/api/rebuild/{jid}/step/{name}")
def step_image(jid: str, name: str):
    _job(jid)
    path = _out(jid) / "steps" / name
    if not re.fullmatch(r"step\d+\.jpg", name) or not path.exists():
        raise HTTPException(404, "no such step image")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/history")
def history():
    """Every upload, newest first, with its status and total time."""
    items = []
    if UPLOADS.exists():
        for f in sorted(UPLOADS.glob("up_*.jpg"),
                        key=lambda p: p.stat().st_mtime, reverse=True):
            jid = f.stem[3:]
            if not re.fullmatch(r"[0-9a-f]{10}", jid):
                continue
            st = _steps(jid)
            items.append({"id": jid, "status": _job(jid)["status"],
                          "steps": len(st["steps"]),
                          "seconds": st.get("total_seconds")})
    return {"runs": items[:50]}
