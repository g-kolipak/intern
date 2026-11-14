"""Pilot benchmark on ShredBench (arXiv 2604.23813), English news, 8 pieces.

  uv run --env-file .env scripts/shredbench.py news_008 news_031 news_041

Needs runs/shredbench/raw/NAME.jpg + NAME.md (from github.com/ythere-y/
ShredBench) and, for the 'ours' arm, runs/photo/NAME/rebuilt.jpg from
scripts/rebuild_photo.py. Two arms, both read by Nace Perception Parse:
  scene   the shredded scene as given (what the paper's models see)
  ours    the page Drex rebuilt
Scores follow the paper: NED (lower is better), BLEU, ROUGE-L (higher).
"""

import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
API = "https://console.nace.ai/v1"


def _small_jpeg(path: Path, maxdim: int = 3000) -> bytes:
    import cv2
    img = cv2.imread(str(path))
    s = maxdim / max(img.shape[:2])
    if s < 1:
        img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def parse(path: Path) -> str:
    """Markdown of an image file, via Nace Perception Parse."""
    h = {"Authorization": f"Bearer {os.environ['NACE_API_KEY']}"}
    body = _small_jpeg(path)
    import hashlib
    name = (f"{path.parent.name}_{path.stem}_"
            f"{hashlib.sha1(body).hexdigest()[:10]}.jpg")  # unique per image
    for attempt in range(3):
        grant = httpx.post(f"{API}/documents/upload-grants", headers=h,
                           json={"path": name}, timeout=60).json()
        up = httpx.post(grant["upload_url"],
                        headers={"X-Upload-Token": grant["token"]},
                        files={"file": (name, body)},
                        data={"metadata": json.dumps({"path": name})},
                        timeout=120).json()
        if "result" in up:
            break
        print(f"  upload retry {attempt + 1}: {str(up)[:200]}", flush=True)
        time.sleep(3)
    else:
        raise RuntimeError(f"upload failed: {str(up)[:300]}")
    src = {"type": "workspace_file", "workspace_id": grant["workspace_id"],
           "file_id": up["result"]["file"]["file_id"]}
    job = httpx.post(f"{API}/documents/parse?wait_seconds=60", headers=h,
                     json={"source": src, "output": {"formats": ["text"]}},
                     timeout=120).json()
    while job.get("status") not in ("succeeded", "failed", "cancelled"):
        time.sleep(2)
        job = httpx.get(f"{API}/documents/jobs/{job['id']}", headers=h,
                        timeout=60).json()
    if job["status"] != "succeeded":
        raise RuntimeError(f"parse {job['status']}: {job.get('error')}")
    return job["result"]["document"]["text"] or ""


def ned(a: str, b: str) -> float:
    if not a and not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1,
                           prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def rouge_l(ref: list, hyp: list) -> float:
    if not ref or not hyp:
        return 0.0
    prev = [0] * (len(hyp) + 1)
    for r in ref:
        cur = [0]
        for j, h in enumerate(hyp, 1):
            cur.append(prev[j - 1] + 1 if r == h else max(prev[j], cur[-1]))
        prev = cur
    lcs = prev[-1]
    if lcs == 0:
        return 0.0
    p, r = lcs / len(hyp), lcs / len(ref)
    return 2 * p * r / (p + r)


def bleu(ref: list, hyp: list, n: int = 4) -> float:
    if not hyp:
        return 0.0
    logs = []
    for k in range(1, n + 1):
        h = Counter(tuple(hyp[i:i + k]) for i in range(len(hyp) - k + 1))
        r = Counter(tuple(ref[i:i + k]) for i in range(len(ref) - k + 1))
        hit = sum(min(c, r[g]) for g, c in h.items())
        logs.append(math.log((hit + 1) / (sum(h.values()) + 1)))  # smoothed
    bp = 1.0 if len(hyp) > len(ref) else math.exp(1 - len(ref) / len(hyp))
    return bp * math.exp(sum(logs) / n)


def clean(md: str) -> str:
    return " ".join(md.replace("#", " ").replace("*", " ").split())


def main() -> int:
    names = sys.argv[1:] or ["news_008", "news_031", "news_041"]
    raw = ROOT / "runs" / "shredbench" / "raw"
    rows = {"scene": [], "ours": []}
    for name in names:
        truth = clean((raw / f"{name}.md").read_text())
        arms = {"scene": raw / f"{name}.jpg",
                "ours": ROOT / "runs" / "photo" / name / "rebuilt.jpg"}
        for arm, img in arms.items():
            if not img.exists():
                continue
            cached = ROOT / "runs" / "shredbench" / f"{name}.{arm}.txt"
            if arm == "scene" and cached.exists():
                text = cached.read_text()  # the scene never changes
            else:
                text = clean(parse(img))
                cached.write_text(text)
            s = (ned(truth, text), bleu(truth.split(), text.split()),
                 rouge_l(truth.split(), text.split()))
            rows[arm].append(s)
            print(f"{name:9s} {arm:5s} NED {s[0]:.2f}  BLEU {s[1]:.2f}  "
                  f"ROUGE-L {s[2]:.2f}  ({len(text)} chars read, "
                  f"{len(truth)} true)", flush=True)
    for arm, r in rows.items():
        if r:
            m = [sum(x[i] for x in r) / len(r) for i in range(3)]
            print(f"MEAN {arm:5s} NED {m[0]:.2f}  BLEU {m[1]:.2f}  "
                  f"ROUGE-L {m[2]:.2f}  over {len(r)} documents")
    print("Paper, 8 pieces, all domains: Gemini 3 Pro NED 0.33 / ROUGE 0.83;"
          " GPT-5.1 0.77 / 0.28; Qwen-VL-Plus 0.59 / 0.58")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
