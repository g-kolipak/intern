"""Assembler: turn neighbor scores into a rebuilt grid.

All candidate links (A, B, direction, score) are sorted high to low. A
link is accepted only if it is the best option for both pieces, it is at
or above CONFIDENCE_THRESHOLD, and it does not put two pieces in one grid
cell or grow a group past ROWS x COLS. Pieces whose best link is too weak
go to a "needs human" list with their top 3 candidates.
"""

from dataclasses import dataclass, field

CONFIDENCE_THRESHOLD = 0.6
STEP = {"right": (0, 1), "below": (1, 0)}


@dataclass(frozen=True)
class Link:
    a: str
    b: str
    direction: str
    score: float
    human: bool = False


@dataclass
class Assembly:
    grid: dict[str, tuple[int, int]] = field(default_factory=dict)
    accepted: list[Link] = field(default_factory=list)
    needs_human: list[dict] = field(default_factory=list)
    events: list[dict] = field(default_factory=list)
    clusters: list[dict[str, tuple[int, int]]] = field(default_factory=list)
    complete: bool = False


def collect_links(right: dict, below: dict) -> list[Link]:
    """Every candidate link from both tables, best score first."""
    links = [Link(a, b, d, s)
             for d, table in (("right", right), ("below", below))
             for a, row in table.items() for b, s in row.items()]
    return sorted(links, key=lambda link: link.score, reverse=True)


def _mutual_best(table: dict) -> set[tuple[str, str]]:
    """Pairs (A, B) where B is A's top pick and A is B's top source."""
    best_in: dict[str, tuple[float, str]] = {}
    for a, row in table.items():
        for b, s in row.items():
            if b not in best_in or s > best_in[b][0]:
                best_in[b] = (s, a)
    return {(a, max(row, key=row.get)) for a, row in table.items()
            if row and best_in[max(row, key=row.get)][1] == a}


class _Board:
    """Groups of pieces with positions relative to each other."""

    def __init__(self, ids, rows: int, cols: int):
        self.rows, self.cols = rows, cols
        self.of = {p: p for p in ids}
        self.cells = {p: {p: (0, 0)} for p in ids}

    def join(self, link: Link) -> bool:
        dr, dc = STEP[link.direction]
        ga, gb = self.of[link.a], self.of[link.b]
        ra, ca = self.cells[ga][link.a]
        want = (ra + dr, ca + dc)
        if ga == gb:
            return self.cells[ga][link.b] == want
        rb, cb = self.cells[gb][link.b]
        moved = {p: (r + want[0] - rb, c + want[1] - cb)
                 for p, (r, c) in self.cells[gb].items()}
        taken = set(self.cells[ga].values())
        if taken & set(moved.values()):
            return False  # two pieces in one cell
        both = taken | set(moved.values())
        rs, cs = [r for r, _ in both], [c for _, c in both]
        if max(rs) - min(rs) >= self.rows or max(cs) - min(cs) >= self.cols:
            return False  # would not fit the page
        self.cells[ga].update(moved)
        for p in moved:
            self.of[p] = ga
        del self.cells[gb]
        return True

    def far_side(self, p: str, direction: str) -> bool:
        """True when p already sits on the last column (or last row)."""
        group = self.cells[self.of[p]]
        axis = 1 if direction == "right" else 0
        vals = [pos[axis] for pos in group.values()]
        size = self.cols if direction == "right" else self.rows
        return (max(vals) - min(vals) + 1 == size
                and group[p][axis] == max(vals))

    def row_start(self, p: str) -> bool:
        group = self.cells[self.of[p]]
        r, c = group[p]
        return c == min(cc for rr, cc in group.values() if rr == r)


def assemble(right: dict, below: dict, rows: int, cols: int,
             threshold: float = CONFIDENCE_THRESHOLD, right_edge=None,
             bottom_edge=None, forced=(), fillers=(), wide=()) -> Assembly:
    """Build the grid. `forced` links come from a person and go first.

    `fillers` are blank pieces. They carry no text to match, so they take
    whatever cells are left once every other piece is in one group. Those
    also listed in `wide` only fit the last column.
    """
    ids = list(right)
    board, out = _Board(ids, rows, cols), Assembly()
    used_out: set[tuple[str, str]] = set()
    used_in: set[tuple[str, str]] = set()
    mutual = {"right": _mutual_best(right), "below": _mutual_best(below)}
    human = [Link(f.a, f.b, f.direction, 1.0, True) for f in forced]
    for link in human + collect_links(right, below):
        k_out, k_in = (link.direction, link.a), (link.direction, link.b)
        if k_out in used_out or k_in in used_in:
            continue
        if not link.human and (link.score < threshold or (link.a, link.b)
                               not in mutual[link.direction]):
            continue
        if board.join(link):
            used_out.add(k_out)
            used_in.add(k_in)
            out.accepted.append(link)
    _place(board, ids, rows, cols, out, list(fillers), set(wide))
    edges = {"right": right_edge or {}, "below": bottom_edge or {}}
    for d, table in (("right", right), ("below", below)):
        for a in ids:
            row = table.get(a) or {}
            if ((d, a) in used_out or not row or board.far_side(a, d)
                    or edges[d].get(a, 0.0) >= 0.5
                    or (d == "below" and not board.row_start(a))):
                continue
            top = sorted(((s, b) for b, s in row.items()
                          if (d, b) not in used_in), reverse=True)[:3]
            if top:
                out.needs_human.append({
                    "piece": a, "direction": d,
                    "candidates": [{"piece": b, "score": round(s, 4)}
                                   for s, b in top]})
    return out


def _place(board: _Board, ids, rows: int, cols: int, out: Assembly,
           fillers: list, wide: set) -> None:
    """Fit groups onto the page, biggest first, and write the events."""
    for group in sorted(board.cells.values(), key=len, reverse=True):
        r0 = min(r for r, _ in group.values())
        c0 = min(c for _, c in group.values())
        out.clusters.append({p: (r - r0, c - c0)
                             for p, (r, c) in group.items()})
    free = {(r, c) for r in range(rows) for c in range(cols)}
    for group in out.clusters:
        if len(group) < 2 and len(ids) > 1:
            continue  # a lone piece stays in the pile
        spot = next(((r, c) for r in range(rows) for c in range(cols)
                     if all((gr + r, gc + c) in free
                            for gr, gc in group.values())), None)
        if spot is None:
            continue
        for p, (gr, gc) in group.items():
            out.grid[p] = (gr + spot[0], gc + spot[1])
            free.discard(out.grid[p])
    one_group = bool(out.clusters) and len(out.clusters[0]) == len(ids)
    last = [cell for cell in sorted(free) if wide and cell[1] == cols - 1]
    rest = [cell for cell in sorted(free) if cell not in last]
    blanks = dict(list(zip([f for f in fillers if f in wide], last))
                  + list(zip([f for f in fillers if f not in wide], rest))
                  ) if one_group else {}
    out.grid.update(blanks)
    out.complete = one_group and len(out.grid) == rows * cols
    order = [(p, link.score, link.human) for link in out.accepted
             for p in (link.a, link.b)]
    order += [(p, 1.0, False) for p in out.grid]  # blanks, 1x1 pages
    seen: set[str] = set()
    for p, score, human in order:
        if p in out.grid and p not in seen:
            seen.add(p)
            out.events.append({
                "type": "place", "piece": p, "row": out.grid[p][0],
                "col": out.grid[p][1], "score": round(score, 4),
                "human": human, "blank": p in blanks,
                "provisional": not out.complete})
