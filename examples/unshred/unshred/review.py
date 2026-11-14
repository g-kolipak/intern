"""Drex reviews the whole rebuilt page and fixes what it flags.

The page is drawn with a number on every piece; Drex answers whether the
page is right and which numbered piece is most clearly wrong. That piece
gets close-up options: its spot, free spots, its other turns, swaps, and
inserts between placed pieces (shifting a row or column to make room);
Drex picks. Pieces left in the tray get a chance to join. Repeat.
"""

from concurrent.futures import ThreadPoolExecutor

import cv2

from . import fit
from .agents import CONTEXT, _pick, _view
from .arrange import _gain
from .build import STEPS, place
from .drex import VISION_MODEL, ask as default_ask
from .tear import data_url


def _insert(cells: dict, w: str, t: str, cell, axis: int) -> dict:
    """Layout with turn t of piece w put at `cell`; the pieces from there
    on down (axis 0) or to the right (axis 1) shift by one."""
    r0, c0 = cell
    out = {}
    for p, (r, c) in cells.items():
        if p == w:
            continue
        if axis == 0 and c == c0 and r >= r0:
            out[p] = (r + 1, c)
        elif axis == 1 and r == r0 and c >= c0:
            out[p] = (r, c + 1)
        else:
            out[p] = (r, c)
    out[t] = cell
    return out


REVIEW = ("Image 1 is a torn page put back together; each fragment carries a "
          "red number. Is the page correctly rebuilt: every fragment in its "
          "right place and turned the right way, text reading across tears?")
WORST = ("In Image 1, which numbered fragment is most clearly in the wrong "
         "place or turned the wrong way? Answer none if all are right.")


def _numbered(up, placed):
    img = fit.compose(up, placed).copy()
    x0 = min(v[0] for v in placed.values())
    y0 = min(v[1] for v in placed.values())
    names = sorted(placed)
    size = max(2.0, img.shape[1] / 500)
    for n, p in enumerate(names, 1):
        h, w = up[p].shape[:2]
        c = (placed[p][0] - x0 + w // 2, placed[p][1] - y0 + h // 2)
        cv2.putText(img, str(n), c, cv2.FONT_HERSHEY_SIMPLEX, size,
                    (40, 40, 230), int(size * 3), cv2.LINE_AA)
    return img, names


def review(up: dict, cells: dict, placed: dict, best: dict, right: dict,
           below: dict, base, rounds: int = 3, ask_fn=default_ask,
           model: str = VISION_MODEL, on_step=None):
    """Drex reviews the whole page; the piece it flags is re-placed (spots,
    swaps, other turns) from close-ups, and tray pieces get a chance too.
    Returns (cells, placed, calls)."""
    from .arrange import base as stem
    calls, first, prev = [], True, None
    for _ in range(rounds):
        img, names = _numbered(up, placed)
        crit = {f"piece_{n}": f"Fragment {n}." for n in range(1, len(names) + 1)}
        crit["none"] = "None: every fragment is right."
        res = ask_fn(CONTEXT, {"ok": {"type": "noul", "instructions": REVIEW},
                               "worst": {"type": "choice", "instructions": WORST,
                                         "criteria": crit}},
                     model=model, images=[data_url(img, 1000)])
        calls.append(res)
        ok = float(res.body["answers"]["ok"]["noul"])
        probs = res.body["answers"]["worst"]["probabilities"]
        worst = max(probs, key=probs.get)
        if prev and ok < prev[0] - 0.1:  # Drex likes the page less: undo
            cells, placed = prev[1], prev[2]
            if on_step:
                on_step({"review_ok": round(ok, 2),
                         "flagged": "undo: last move made the page worse"})
            break
        prev = (ok, cells, placed)
        if on_step:
            on_step({"review_ok": round(ok, 2), "flagged": worst})
        used = {stem(p) for p in cells}
        movers = [] if worst == "none" or ok >= 0.6 else \
            [names[int(worst[6:]) - 1]]
        if first:  # a piece hanging off the page by one side is suspect
            at = {v: k for k, v in cells.items()}
            movers += [p for p, (r, c) in cells.items() if p not in movers
                       and sum((r + dr, c + dc) in at
                               for dr, dc in STEPS) <= 1][:2]
            first = False
        movers += [p for p in up if stem(p) not in used
                   and p.split("@")[-1] in (p, "0")][:2]  # tray pieces
        if not movers:
            break
        for w in movers:
            at = {v: k for k, v in cells.items()}
            rest = {k: v for k, v in at.items() if v != w}
            free = {(r + dr, c + dc) for r, c in rest
                    for dr, dc in STEPS} - set(rest)
            spots = sorted(free, key=lambda s: -_gain(
                w, s, cells, rest, right, below, base)[0])[:4]
            twins = [t for t in up if stem(t) == stem(w) and t != w]
            options = ([cells] if w in cells else []) + \
                [{**{k: v for k, v in cells.items() if k != w}, w: s}
                 for s in spots]
            if w in cells:
                options += [{**{k: v for k, v in cells.items() if k != w},
                             t: cells[w]} for t in twins]
                options += [{**cells, w: cells[u], u: cells[w]}
                            for u in sorted(cells, key=lambda u: -_gain(
                                w, cells[u], cells, rest, right, below,
                                base)[0]) if u != w][:2]
            if w in cells:  # also insert it between placed pieces
                ins = []
                for t in [w] + twins:
                    for cell in rest:
                        for axis in (0, 1):
                            o = _insert(cells, w, t, cell, axis)
                            ins.append((_gain(t, cell, o, {v: k for k, v in
                                        o.items() if k != t}, right, below,
                                        base)[0], o))
                ins.sort(key=lambda x: -x[0])
                options = options[:3] + [o for _, o in ins[:3]] + options[3:]
            options = options[:8]
            if len(options) < 2 and (w in cells or not options):
                continue  # nothing to choose between
            with ThreadPoolExecutor(max_workers=8) as pool:
                places = list(pool.map(lambda o: place(up, o, best), options))
            focus = [w if w in o else next(t for t in twins if t in o)
                     for o in options]
            k, pr, res2 = _pick([_view(up, pl, f) for pl, f in
                                 zip(places, focus)], ask_fn, model)
            calls.append(res2)
            keep_first = w in cells
            better = pr[f"image_{k + 1}"] >= (pr["image_1"] + 0.1
                                              if keep_first else 0.3)
            if (k or not keep_first) and better:
                cells, placed = options[k], places[k]
            if on_step:
                on_step({"piece": w, "options": len(options), "pick": k,
                         "p": round(pr[f"image_{k + 1}"], 2)})
    return cells, placed, calls
