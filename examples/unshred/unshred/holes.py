"""Fill holes: put loose pieces into the gaps inside the rebuilt page.

The layout works on a grid, so a gap between two pieces that the grid
calls neighbours is never offered as a spot. Here the gaps are found in
the picture itself: the page's outline (convex hull of the other pieces)
minus the pieces leaves holes. A piece hanging off the page (or left in
the tray) is tried in every hole that could hold it, in each of its
turns, snugged against its new neighbours. Drex picks from close-ups;
the move stands only if Drex then rates the whole page at least as well.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .agents import _pick, _view
from .arrange import base as stem
from .build import STEPS, snug
from .drex import VISION_MODEL, ask as default_ask
from .review import CONTEXT, REVIEW, _numbered
from .tear import data_url

S = 0.25  # holes are found at a quarter of the pieces' resolution


def _mask(piece):
    m = (piece[..., 3] > 128).astype(np.uint8)
    return cv2.resize(m, None, fx=S, fy=S, interpolation=cv2.INTER_NEAREST)


def holes(up: dict, placed: dict, skip: str, min_area: float):
    """Gaps inside the page made by every piece except `skip`:
    [(centre_x, centre_y, area, hole mask, frame origin)] in placed units."""
    others = {p: v for p, v in placed.items() if p != skip}
    if len(others) < 2:
        return []
    x0 = min(v[0] for v in others.values())
    y0 = min(v[1] for v in others.values())
    W = max(v[0] + up[p].shape[1] for p, v in others.items()) - x0
    H = max(v[1] + up[p].shape[0] for p, v in others.items()) - y0
    page = np.zeros((int(H * S) + 2, int(W * S) + 2), np.uint8)
    for p, (x, y) in others.items():
        m = _mask(up[p])
        px, py = int((x - x0) * S), int((y - y0) * S)
        page[py:py + m.shape[0], px:px + m.shape[1]] |= m
    cs, _ = cv2.findContours(page, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    hull = np.zeros_like(page)
    cv2.drawContours(hull, [cv2.convexHull(np.vstack(cs))], -1, 1, -1)
    gap = cv2.morphologyEx(hull & (1 - page), cv2.MORPH_OPEN,
                           np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(gap)
    return [(x0 + cen[i][0] / S, y0 + cen[i][1] / S, st[i][4] / S / S,
             lab == i, (x0, y0)) for i in range(1, n)
            if st[i][4] / S / S >= min_area]


def fill(up: dict, cells: dict, placed: dict, ask_fn=default_ask,
         model: str = VISION_MODEL, on_step=None):
    """Try dangling and tray pieces in the page's holes. Returns
    (placed, cells, calls); a moved piece leaves the grid (it now sits
    in a gap the grid has no cell for)."""
    calls = []
    at = {v: k for k, v in cells.items()}
    dangling = [p for p, (r, c) in cells.items()
                if sum((r + dr, c + dc) in at for dr, dc in STEPS) <= 1]
    used = {stem(p) for p in placed}
    tray = [p for p in up if stem(p) not in used
            and p.split("@")[-1] in (p, "0")]
    for w in (dangling + tray)[:3]:
        turns = [t for t in up if stem(t) == stem(w)]
        area = float((up[w][..., 3] > 128).sum())
        found = holes(up, placed, w, 0.3 * area)
        if not found:
            continue
        rest = {k: v for k, v in placed.items() if k != w}
        tries = []
        for cx, cy, h_area, hole, _ in found:
            if h_area > 3 * area:
                continue
            for t in turns:
                h, wd = up[t].shape[:2]
                pos = snug((int(cx - wd / 2), int(cy - h / 2)), up[t], rest, up)
                tries.append({**rest, t: pos})
        if not tries:
            continue
        options = ([placed] if w in placed else []) + tries[:7]
        focus = [next(t for t in turns if t in o) for o in options]
        with ThreadPoolExecutor(max_workers=8) as pool:
            images = list(pool.map(lambda a: _view(up, a[0], a[1]),
                                   zip(options, focus)))
        k, pr, res = _pick(images, ask_fn, model)
        calls.append(res)
        current = w in placed
        if (k == 0 and current) or pr[f"image_{k + 1}"] < (
                pr["image_1"] + 0.1 if current else 0.3):
            continue
        before, after = (_page_ok(up, pl, ask_fn, model, calls)
                         for pl in (placed, options[k]))
        if on_step:
            on_step({"piece": w, "to": focus[k], "pick": round(
                pr[f"image_{k + 1}"], 2), "page": (round(before, 2),
                                                   round(after, 2))})
        if after >= before - 0.05:
            placed = options[k]
            cells = {c: v for c, v in cells.items() if stem(c) != stem(w)}
    return placed, cells, calls


def _page_ok(up, placed, ask_fn, model, calls) -> float:
    img, _ = _numbered(up, placed)
    res = ask_fn(CONTEXT, {"ok": {"type": "noul", "instructions": REVIEW}},
                 model=model, images=[data_url(img, 1000)])
    calls.append(res)
    return float(res.body["answers"]["ok"]["noul"])
