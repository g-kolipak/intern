"""Draw candidate layouts as full pages and let Drex pick the right one.

arrange.py proposes the best few complete layouts, all built from Drex's
own pair and pick scores. Here each layout is drawn as a whole page,
every piece snugged against its neighbours so the torn edges close up,
and Drex Max sees all the pages in one request and picks the one that is
the original page put back together. Drex sees the whole page, so it can
check the reading order across the full width, not just one seam.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from . import fit
from .drex import VISION_MODEL, ask as default_ask
from .tear import data_url

CONTEXT = ("A torn printed page has been put back together in different "
           "ways. Each image is one attempt, shown on a dark background.")
PICK = ("Images 1 to {n} are attempts to put the same torn page back "
        "together. Which image is the original page correctly rebuilt: every "
        "fragment next to its true neighbours, text lines and printed shapes "
        "continuing across every torn edge, and the page reading in the "
        "right order from top to bottom? Answer none if no image is right.")
STEPS = {(0, -1): "right", (0, 1): "right", (-1, 0): "below", (1, 0): "below"}


_MASKS: dict = {}


def _mask(piece, scale: float):
    """The piece's paper mask at `scale`, computed once per piece."""
    key = (id(piece), piece.shape, scale)
    if key not in _MASKS:
        m = (piece[..., 3] > 128).astype(np.uint8)
        _MASKS[key] = cv2.resize(m, None, fx=scale, fy=scale,
                                 interpolation=cv2.INTER_NEAREST)
    return _MASKS[key]


def snug(pos, piece, placed: dict, up: dict, reach: int = 120,
         scale: float = 0.25):
    """Nudge `piece` around `pos` so its torn edges close up against the
    placed pieces without overlapping them (pure geometry)."""
    x0, y0 = pos
    near = [q for q, (qx, qy) in placed.items()
            if abs(qx - x0) < 2 * max(piece.shape) and
            abs(qy - y0) < 2 * max(piece.shape)]
    if not near:
        return pos
    ox = min([placed[q][0] for q in near] + [x0 - reach])
    oy = min([placed[q][1] for q in near] + [y0 - reach])
    W = max([placed[q][0] + up[q].shape[1] for q in near]
            + [x0 + reach + piece.shape[1]]) - ox
    H = max([placed[q][1] + up[q].shape[0] for q in near]
            + [y0 + reach + piece.shape[0]]) - oy
    board = np.zeros((int(H * scale) + 2, int(W * scale) + 2), np.uint8)
    for q in near:
        qm = _mask(up[q], scale)
        qx = int((placed[q][0] - ox) * scale)
        qy = int((placed[q][1] - oy) * scale)
        board[qy:qy + qm.shape[0], qx:qx + qm.shape[1]] |= qm
    halo = cv2.dilate(board, np.ones((9, 9), np.uint8))
    pm = _mask(piece, scale)
    def score(dx, dy):
        x, y = int((x0 + dx - ox) * scale), int((y0 + dy - oy) * scale)
        win_b = board[y:y + pm.shape[0], x:x + pm.shape[1]]
        if x < 0 or y < 0 or win_b.shape != pm.shape:
            return -1e9
        win_h = halo[y:y + pm.shape[0], x:x + pm.shape[1]]
        overlap = int((win_b & pm).sum())
        # close gaps; a little overlap of torn fibres is fine
        return int((win_h & pm).sum()) - 1.5 * overlap

    # coarse grid first, then a fine one around the best coarse spot
    grid = [(dx, dy) for dy in range(-reach, reach + 1, 20)
            for dx in range(-reach, reach + 1, 20)]
    cx, cy = max(grid, key=lambda d: score(*d))
    fine = [(cx + dx, cy + dy) for dy in range(-18, 19, 3)
            for dx in range(-18, 19, 3)]
    dx, dy = max(fine, key=lambda d: score(*d))
    return (x0 + dx, y0 + dy)


def _offset(a, b, d, up, best):
    """Where b goes relative to a (b on side d of a): Drex's alignment if
    it was asked about this pair, else the text-line fit."""
    v = best.get(f"{a}|{b}|{d}")
    return (v["dx"], v["dy"]) if v else fit.fit(up[a], up[b], d)[:2]


def place(up: dict, cells: dict, best: dict) -> dict:
    """Pixel places for a grid layout: walk outward from the top-left
    piece; each piece goes where its placed neighbours put it (averaged),
    then is snugged so the tears close."""
    at = {v: k for k, v in cells.items()}
    order = sorted(cells, key=lambda p: cells[p])
    placed = {order[0]: (0, 0)}
    todo = order[1:]
    while todo:
        for p in todo:
            r, c = cells[p]
            votes = []
            for (dr, dc), d in STEPS.items():
                q = at.get((r + dr, c + dc))
                if q not in placed:
                    continue
                if (dr, dc) in ((0, -1), (-1, 0)):  # q is left of / above p
                    ox, oy = _offset(q, p, d, up, best)
                    votes.append((placed[q][0] + ox, placed[q][1] + oy))
                else:                               # q is right of / below p
                    ox, oy = _offset(p, q, d, up, best)
                    votes.append((placed[q][0] - ox, placed[q][1] - oy))
            if votes:
                pos = (int(np.mean([v[0] for v in votes])),
                       int(np.mean([v[1] for v in votes])))
                placed[p] = snug(pos, up[p], placed, up)
                todo.remove(p)
                break
        else:  # not connected to the rest: put it to the side
            p = todo.pop(0)
            placed[p] = (max(x + up[q].shape[1] for q, (x, _) in placed.items())
                         + 60, 0)
    return placed


def choose(pages: list, ask_fn=default_ask, model: str = VISION_MODEL):
    """Drex picks the right page. Returns (index or None, probs, call)."""
    criteria = {f"image_{i + 1}": f"Image {i + 1} is right."
                for i in range(len(pages))}
    criteria["none"] = "No image is right."
    res = ask_fn(CONTEXT, {"page": {"type": "choice", "criteria": criteria,
                                    "instructions": PICK.format(
                                        n=len(pages))}},
                 model=model, images=[data_url(p, 900) for p in pages])
    probs = res.body["answers"]["page"]["probabilities"]
    pick = max(probs, key=probs.get)
    return (None if pick == "none" else int(pick[6:]) - 1), probs, res


def improve(up: dict, cells: dict, best: dict, right: dict, below: dict,
            base: float, rounds: int = 3, ask_fn=default_ask,
            model: str = VISION_MODEL, on_round=None):
    """Drex re-decides the least supported pieces, seeing whole pages.

    Each round: the piece whose neighbours support it least is drawn in
    its current spot and in up to 6 other free spots along the page;
    Drex picks the page that is right. Returns (cells, placed, calls).
    """
    from .arrange import _gain
    calls, tried = [], set()
    placed = place(up, cells, best)
    for _ in range(rounds):
        at = {v: k for k, v in cells.items()}
        support = {p: _gain(p, cells[p], cells, {k: v for k, v in at.items()
                                                  if v != p},
                            right, below, base)[0]
                   for p in cells if p not in tried}
        if not support:
            break
        weak = min(support, key=support.get)
        tried.add(weak)
        rest = {k: v for k, v in at.items() if v != weak}
        free = {(r + dr, c + dc) for r, c in rest
                for dr, dc in STEPS} - set(rest)
        spots = sorted(free, key=lambda cell: -_gain(
            weak, cell, cells, rest, right, below, base)[0])
        spots = [cells[weak]] + [s for s in spots if s != cells[weak]][:6]
        options = [{**cells, weak: s} for s in spots]
        with ThreadPoolExecutor(max_workers=8) as pool:
            places = list(pool.map(lambda o: place(up, o, best), options))
        pick, probs, res = choose([fit.compose(up, pl) for pl in places],
                                  ask_fn, model)
        calls.append(res)
        if on_round:
            on_round(weak, len(options), pick, probs)
        if pick:  # 0 means: the current spot was right
            cells, placed = options[pick], places[pick]
    return cells, placed, calls
