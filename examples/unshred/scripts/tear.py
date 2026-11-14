"""Real paper: cut the torn pieces out of a photo, then let Drex rebuild it.

  uv run scripts/tear.py image.png            cut out only: no key needed
  uv run scripts/tear.py image.png --match    Drex Max turns every piece
                                              upright and judges the seams
  uv run scripts/tear.py image.png --reuse    lay out again from the saved
                                              scores, without new calls

Output goes to runs/photo/NAME/: outline.jpg (photo with every piece
bordered), sheet.jpg (the cut-outs), pieces/, and with --match also
upright/, scores.json and rebuilt.jpg.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unshred import cutouts, demo, scoring, tear  # noqa: E402
from unshred.assembler import CONFIDENCE_THRESHOLD, assemble  # noqa: E402
from unshred.photo import find_pieces  # noqa: E402


def rebuild(pieces: dict, out: Path, reuse: bool, threshold: float):
    """Turn upright, judge seams, lay out. Returns what the page needs."""
    saved = out / "scores.json"
    if reuse:
        data = json.loads(saved.read_text())
        turns, right, below = data["turns"], data["right"], data["below"]
        use = data.get("usage", {})
        upright = {p: tear.quarter_turn(pieces[p], turns[p]["degrees"])
                   for p in pieces}
    else:
        t0 = time.monotonic()
        upright, turns, calls = tear.orient_all(pieces)
        match = tear.match_pairs(upright)
        right, below = match.right, match.below
        use = scoring.usage(calls + match.calls,
                            scoring.price_per_mtok("DREX_MAX_USD_PER_MTOK"))
        use["wall_s"] = round(time.monotonic() - t0, 1)
        saved.write_text(json.dumps({"turns": turns, "right": right,
                                     "below": below, "usage": use}, indent=1))
        print(f"Drex Max: {use['calls']} calls, average "
              f"{use['avg_latency_ms']:.0f} ms, {use['input_tokens']} input "
              f"tokens, wall time {use['wall_s']} s")
    (out / "upright").mkdir(exist_ok=True)
    for p, img in upright.items():
        cv2.imwrite(str(out / "upright" / f"{p}.png"), img)
    print("Upright turn per piece (clockwise degrees, confidence):")
    print("  " + "  ".join(f"{p}:{t['degrees']}({t['confidence']:.2f})"
                           for p, t in turns.items()))
    n = len(pieces)
    asm = assemble(right, below, n, n, threshold)
    print(f"Links accepted at threshold {threshold}:")
    for link in asm.accepted:
        print(f"  {link.b} is {link.direction} of {link.a}  "
              f"score {link.score:.2f}")
    linked, loose = demo.groups(asm)
    cv2.imwrite(str(out / "rebuilt.jpg"),
                cutouts.compose(upright, linked, loose))
    cv2.imwrite(str(out / "upright.jpg"),
                cutouts.sheet(list(upright.values())))
    print(f"{n - len(loose)} of {n} pieces linked, {len(loose)} loose "
          f"(last row of rebuilt.jpg)")
    return upright, turns, asm, use


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("--match", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    args = ap.parse_args()
    src = Path(args.photo)
    image = cv2.imread(str(src))
    if image is None:
        print(f"cannot read {src}")
        return 1
    t0 = time.monotonic()
    masks = find_pieces(image)
    out = ROOT / "runs" / "photo" / src.stem
    (out / "pieces").mkdir(parents=True, exist_ok=True)
    pieces, meta = {}, {}
    for i, mask in enumerate(masks):
        raw = cutouts.cutout(image, mask)
        skew = cutouts.skew_angle(raw)
        m = cv2.moments(mask, binaryImage=True)
        pieces[f"p{i + 1}"] = cutouts.rotate(raw, -skew)
        meta[f"p{i + 1}"] = {"skew": skew,
                             "cx": m["m10"] / m["m00"] / image.shape[1],
                             "cy": m["m01"] / m["m00"] / image.shape[0]}
    for p, img in pieces.items():
        cv2.imwrite(str(out / "pieces" / f"{p}.png"), img)
    cv2.imwrite(str(out / "outline.jpg"), cutouts.outline(image, masks))
    cv2.imwrite(str(out / "sheet.jpg"), cutouts.sheet(list(pieces.values())))
    print(f"{len(pieces)} pieces cut out in {time.monotonic() - t0:.1f} s")
    if args.match or args.reuse:
        upright, turns, asm, use = rebuild(pieces, out, args.reuse,
                                           args.threshold)
        for p, img in upright.items():
            meta[p]["h"], meta[p]["w"] = img.shape[:2]
        demo.write(out, demo.build(src.stem, image.shape[1::-1], meta, turns,
                                   asm, use, args.threshold))
        print(f"Page: open 'web/tear.html?run={src.stem}' "
              f"(or: scripts/show.sh {src.stem})")
    print(f"Saved to {out.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
