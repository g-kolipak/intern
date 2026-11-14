"""Arrange pieces on a grid by checking them together, not one link at a time.

Drex's confidence p that B continues A (to the right, or below) becomes
evidence log(p / (1 - p)): positive when Drex believes the match, negative
when it does not. A layout's score is the sum of that evidence over EVERY
pair of side-by-side neighbours in it, so a piece that touches several
placed pieces must agree with all of them, and one wrong but confident
link cannot win against a block of pieces that support each other.

Beam search: start from the most confident links, add one piece at a time
to the free cell where its neighbours give it the most evidence, keep the
best `beam` partial layouts. A piece is only added if that evidence is
positive; the rest are left for a human, with their best candidates.
"""

import math

CLIP = 0.995  # keep log-odds finite


def base(p: str) -> str:
    """'p3@90' (piece p3 turned 90 degrees) -> 'p3'."""
    return p.split("@")[0]


_b = base


def evidence(p, base: float | None = None) -> float:
    """Yes/no confidence -> log-odds. With `base` (a pick-one question's
    chance level, e.g. 1/8), a pick share -> log(p / base) instead; a pair
    Drex was never asked about (p is None) counts as mildly against."""
    if base is not None:
        return -1.0 if p is None else max(-3.0, min(3.0, math.log(
            max(p, 1e-3) / base)))
    p = min(max(p or 0.0, 1 - CLIP), CLIP)
    return math.log(p / (1 - p))


def _gain(piece, cell, layout, at, right, below, base=None):
    """Evidence for `piece` at `cell` from every placed neighbour."""
    r, c = cell
    total, touching = 0.0, 0
    for (dr, dc), (table, first) in {(0, -1): (right, True),
                                     (0, 1): (right, False),
                                     (-1, 0): (below, True),
                                     (1, 0): (below, False)}.items():
        other = at.get((r + dr, c + dc))
        if other is None:
            continue
        a, b = (other, piece) if first else (piece, other)
        total += evidence(table.get(a, {}).get(b), base)
        touching += 1
    return total, touching


def arrange(right: dict, below: dict, beam: int = 60, starts: int = 12,
            base: float | None = None, top: int = 1):
    """Best layout found: (cells {piece: (row, col)}, score, links).

    links lists every neighbour pair in the layout as
    (a, b, direction, Drex confidence). With top > 1, a list of the best
    `top` distinct layouts instead, each as (cells, score, links).
    """
    pieces = sorted(set(right) | set(below))
    seeds = sorted(((p, a, b, d) for d, t in (("right", right),
                                              ("below", below))
                    for a in t for b, p in t[a].items()), reverse=True)
    states = []
    for p, a, b, d in [s for s in seeds if evidence(s[0], base) > 0
                       and _b(s[1]) != _b(s[2])][:starts]:
        cells = {a: (0, 0), b: (0, 1) if d == "right" else (1, 0)}
        states.append((evidence(p, base), cells))
    best = max(states, key=lambda s: s[0]) if states else (0.0, {})
    finals = list(states)
    while states:
        grown = []
        for score, cells in states:
            at = {v: k for k, v in cells.items()}
            free = {(r + dr, c + dc) for r, c in cells.values()
                    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0))} - set(at)
            used = {_b(c) for c in cells}
            for piece in pieces:
                if _b(piece) in used:  # one turn per piece
                    continue
                for cell in free:
                    gain, touching = _gain(piece, cell, cells, at, right,
                                           below, base)
                    if touching and gain > 0:
                        grown.append((score + gain, {**cells, piece: cell}))
        if not grown:
            break
        seen, states = set(), []
        for score, cells in sorted(grown, key=lambda s: -s[0]):
            key = _shape(cells)
            if key not in seen:
                seen.add(key)
                states.append((score, cells))
            if len(states) == beam:
                break
        if states[0][0] > best[0]:
            best = states[0]
        finals = states + [f for f in finals if len(f[1]) == len(states[0][1])]
    if top > 1:
        out, seen = [], set()
        for score, cells in sorted([best] + finals, key=lambda s: -s[0]):
            if _shape(cells) not in seen:
                seen.add(_shape(cells))
                out.append((_normal(cells), score, _links(cells, right, below)))
            if len(out) == top:
                break
        return out
    score, cells = best
    return _normal(cells), score, _links(cells, right, below)


def _shape(cells: dict) -> tuple:
    norm = _normal(cells)
    return tuple(sorted(norm.items()))


def _normal(cells: dict) -> dict:
    if not cells:
        return {}
    r0 = min(r for r, _ in cells.values())
    c0 = min(c for _, c in cells.values())
    return {p: (r - r0, c - c0) for p, (r, c) in cells.items()}


def _links(cells: dict, right: dict, below: dict) -> list:
    at = {v: k for k, v in cells.items()}
    out = []
    for a, (r, c) in cells.items():
        for (dr, dc), d, t in (((0, 1), "right", right),
                               ((1, 0), "below", below)):
            b = at.get((r + dr, c + dc))
            if b is not None:
                out.append((a, b, d, t.get(a, {}).get(b, 0.0)))
    return out


def unplaced(cells: dict, right: dict, below: dict, top: int = 3) -> list:
    """For each piece left out: its best candidates either way."""
    out, used = [], {_b(c) for c in cells}
    for p in sorted(set(right) | set(below)):
        if _b(p) in used:
            continue
        cands = sorted(((t[a][b], a, b, d) for d, t in (("right", right),
                                                         ("below", below))
                        for a in t for b in t[a] if p in (a, b)),
                       reverse=True)[:top]
        out.append({"piece": p, "candidates": [
            {"link": f"{b} {d} of {a}", "score": round(s, 4)}
            for s, a, b, d in cands]})
    return out
