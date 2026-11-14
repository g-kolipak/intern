"""Make simulated torn-page photos, or score a rebuild against the answer.

  uv run scripts/eval_torn.py make 4 6 pale     -> runs/synthetic/torn_4x6_pale.png
  uv run --env-file .env scripts/rebuild_photo.py runs/synthetic/torn_4x6_pale.png
  uv run scripts/eval_torn.py score torn_4x6_pale

Scores (the answer key is read only here, after the run):
  found      pieces Drex kept as paper / pieces in the photo
  links      neighbour pairs in the layout that are true neighbours
  recall     true neighbour pairs that the layout contains
  position   pieces in the right cell, after the best shift of the grid
"""

import json
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unshred import torn  # noqa: E402

STEP = {"right": (0, 1), "below": (1, 0)}


def make(rows: int, cols: int, kind: str, seed: int = 1) -> Path:
    text = (ROOT / "data" / "sample.txt").read_text().splitlines()
    lines = [t for t in text if t.strip()] * 2
    pieces, truth = torn.tear(torn.page(lines), rows, cols, seed=seed)
    photo, where = torn.scatter(pieces, kind, seed=seed)
    out = ROOT / "runs" / "synthetic" / f"torn_{rows}x{cols}_{kind}"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out.with_suffix(".png")), photo)
    out.with_suffix(".json").write_text(json.dumps(
        {"truth": truth, "where": where, "rows": rows, "cols": cols}))
    return out.with_suffix(".png")


def score(name: str) -> dict:
    gt = json.loads((ROOT / "runs" / "synthetic" / f"{name}.json").read_text())
    lay = json.loads((ROOT / "runs" / "photo" / name / "layout.json")
                     .read_text())
    centers = np.array([w["center"] for w in gt["where"]])
    cell_of = {}  # piece id -> true (row, col), by nearest true centre
    for p, (x, y) in lay["where"].items():
        cell_of[p] = tuple(gt["truth"][int(np.argmin(
            np.hypot(centers[:, 0] - x, centers[:, 1] - y)))])
    good = sum(1 for a, b, d, _ in lay["links"]
               if (cell_of[b][0] - cell_of[a][0],
                   cell_of[b][1] - cell_of[a][1]) == STEP[d])
    true_pairs = 2 * gt["rows"] * gt["cols"] - gt["rows"] - gt["cols"]
    shifts = Counter((cell_of[p][0] - r, cell_of[p][1] - c)
                     for p, (r, c) in lay["cells"].items())
    n = len(gt["truth"])
    return {"found": f"{len(lay['where'])}/{n}",
            "links": f"{good}/{len(lay['links'])}",
            "recall": f"{good}/{true_pairs}",
            "position": f"{shifts.most_common(1)[0][1] if shifts else 0}/{n}"}


if __name__ == "__main__":
    if sys.argv[1] == "make":
        print(make(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]))
    else:
        print(sys.argv[2], score(sys.argv[2]))
