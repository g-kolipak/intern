"""Rebuild a torn page from a photo. Every decision is Drex Max's.

  uv run --env-file .env scripts/rebuild_photo.py image.png
  uv run --env-file .env scripts/rebuild_photo.py image.png --fast
  uv run scripts/rebuild_photo.py image.png --reuse     re-lay out only

1. find      OpenCV proposes every region that might be a piece, traces
             its outline (in parallel threads) and wins back paper bent
             into shadow (shade.py).
2. check     Drex, one call per region, all in parallel: is it paper, and
             which quarter turn is upright.
3. rate      OpenCV pre-screens pairs (text-line rhythm, colour, torn
             edge fit) and draws joined images at 3 alignments; Drex rates
             every one yes/no, all in parallel. --fast skips this and
             trusts the pre-screen (about 9 s faster, less accurate).
4. compare   Drex, in parallel: for each piece and side, which of its top
             candidates is the true neighbour (pick one, or none).
5. page      the best complete layouts from Drex's picks are drawn as
             full pages with the tears closed; Drex picks the right page,
             then re-decides the least supported pieces.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unshred import (agents, arrange, build, compare, cutouts, fit,  # noqa: E402
                     holes, plain, review, scoring, screen, shade, stages)
from unshred.stages import Clock  # noqa: E402
from unshred.photo import find_pieces  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("photo")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--reuse", action="store_true")
    args = ap.parse_args()
    image = cv2.imread(args.photo)
    out = ROOT / "runs" / "photo" / Path(args.photo).stem
    out.mkdir(parents=True, exist_ok=True)
    clock = Clock(out)
    (out / "joins").mkdir(parents=True, exist_ok=True)
    saved = out / "scores.json"
    masks = (plain.find(image) if plain.is_plain(image) else
             plain.trim_sunlit_table(image, shade.grow(image, find_pieces(image))))
    clock(f"find: {len(masks)} candidate regions (OpenCV)",
          image=cutouts.outline(image, masks) if masks else image)
    if not args.reuse:  # Drex decides which touching regions are one piece
        n0 = len(masks)
        masks, cm = plain.merge_touching(image, masks)
        clock(f"merge: Drex joined touching regions that are one piece "
              f"({n0} -> {len(masks)})", cm, image=cutouts.outline(image, masks))
    pieces = {f"p{i + 1}": cutouts.straighten(cutouts.cutout(image, m))
              for i, m in enumerate(masks)}
    where = {f"p{i + 1}": [float(v) for v in np.argwhere(m > 0).mean(0)[::-1]]
             for i, m in enumerate(masks)}
    if args.reuse:
        data = json.loads(saved.read_text())
    else:
        data, calls = {}, []
        paper, turns, upright, c1 = screen.check(pieces, workers=args.workers)
        calls += c1
        kept = [upright[p] for p in pieces if paper[p] >= 0.5]
        clock("check: paper or not + which way up", c1,
              image=cutouts.sheet(kept) if kept else None)
        up = stages.variants(pieces, turns, paper)
        if not args.fast:
            best, c2 = stages.rate_pairs(up, args.workers)
            calls += c2
            clock("screen + Drex rates every shortlisted pair", c2)
        else:
            best = screen.shortlist(up, as_best=True)
            clock("screen: shortlist pairs and alignments (OpenCV)")
        right, below, c3 = compare.compare(up, best, workers=args.workers)
        calls += c3
        clock("compare: Drex picks each piece's true neighbours", c3)
        data = {"paper": paper, "turns": turns, "best": best,
                "rated": not args.fast,
                "pick": {"right": right, "below": below},
                "usage": scoring.usage(calls)}
        saved.write_text(json.dumps(data, indent=1))
    paper, turns, best = data["paper"], data["turns"], data["best"]
    up = stages.variants(pieces, turns, paper)
    right, below = data["pick"]["right"], data["pick"]["below"]
    area = {p: int((up[p][..., 3] > 128).sum()) for p in up}
    tiny = [p for p in up if area[p] < 0.2 * float(np.median(list(area.values())))]
    for t in (right, below):  # slivers can't be placed: leave them out
        for p in tiny:
            t.pop(p, None)
        for row in t.values():
            for p in tiny:
                row.pop(p, None)
    if not up:
        print("  Drex found no paper pieces; nothing to rebuild")
        return 1
    kept = {arrange.base(p) for p in up}
    turned = sorted(p for p in kept if f"{p}@0" in up)
    dropped = [f"{p} ({paper[p]:.2f})" for p in pieces if p not in kept]
    print(f"  Drex: {len(kept)} of {len(pieces)} regions are paper"
          + (f"; trying all 4 turns of {', '.join(turned)}" if turned else "")
          + (f"; not paper: {', '.join(dropped)}" if dropped else ""))
    # the best layouts two ways (pick-one only / both answers): Drex chooses
    layouts = stages.candidates(up, best, right, below, data.get("rated", True))
    with ThreadPoolExecutor(max_workers=8) as pool:
        places = list(pool.map(lambda lay: build.place(up, lay[0], best),
                               layouts))
    pick, probs, res = build.choose([fit.compose(up, pl) for pl in places])
    choice = pick if pick is not None else 0
    (right, below), base = layouts[choice][1], layouts[choice][2]
    clock(image=fit.compose(up, places[choice]), label=f"page: Drex picked layout {choice + 1} of {len(layouts)} ("
          f"{'pick-one' if base else 'both answers'})", calls=[res])
    cells, placed, c4 = agents.repair(
        up, layouts[choice][0], best, right, below, base, on_step=lambda e: print(
            "           " + (f"piece agents, fit: {e['happy']}" if "happy" in e
                             else f"{e['piece']} picked option {e['pick'] + 1} of "
                                  f"{e['options']} ({e['p']:.2f})")))
    clock("piece agents: each piece checks its spot, unhappy ones move", c4,
          image=fit.compose(up, placed))
    cells, placed, c5 = review.review(
        up, cells, placed, best, right, below, base, on_step=lambda e: print(
            "           " + (f"Drex review: page right {e['review_ok']}, "
                             f"flagged {e['flagged']}" if "review_ok" in e
                             else f"{e['piece']} picked option {e['pick'] + 1}"
                                  f" of {e['options']} ({e['p']:.2f})")))
    clock("final check: Drex reviews the whole page and fixes what it flags",
          c5, image=fit.compose(up, placed))
    placed, cells, c6 = holes.fill(up, cells, placed, on_step=lambda e: print(
        f"           {e['piece']} -> hole, as {e['to']} (Drex {e['pick']:.2f};"
        f" page {e['page'][0]:.2f} -> {e['page'][1]:.2f})"))
    clock("fill holes: loose pieces tried in the gaps of the page", c6,
          image=fit.compose(up, placed))
    cells = arrange._normal(cells)
    at = {v: k for k, v in cells.items()}
    for r in range(1 + max(r for r, _ in cells.values())):
        print("     " + " ".join(f"{at.get((r, c), '..'):>3}" for c in
                                 range(1 + max(c for _, c in cells.values()))))
    done = {arrange.base(p) for p in placed}
    left = list({arrange.base(p): p for p in up
                 if arrange.base(p) not in done}.values())
    page = fit.compose(up, placed)
    if left:  # leftover pieces in a tray under the page, for a human
        tray = fit.compose(up, fit.layout({p: up[p] for p in left}, [], {}))
        w = max(page.shape[1], tray.shape[1])
        page, tray = (cv2.copyMakeBorder(i, top, 0, 0, w - i.shape[1],
                                         cv2.BORDER_CONSTANT, value=fit.BACK)
                      for i, top in ((page, 0), (tray, 80)))
        page = np.vstack([page, tray])
        print(f"  needs human: {', '.join(left)}")
    cv2.imwrite(str(out / "rebuilt.jpg"), page)
    (out / "layout.json").write_text(json.dumps({
        "cells": cells, "placed": placed, "links": arrange._links(
            cells, right, below), "where": {p: where[arrange.base(p)] for p in placed}},
        indent=1))
    print(f"  total {time.monotonic() - clock.t0:.1f} s -> {out}/rebuilt.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
