"""Pipeline: one rebuild, start to finish, with live events.

Order of work (rows first, then the order of rows, as cross-cut shredding
solvers do):
  1. match_right    which piece sits right of which        (N requests)
  2. assemble       best-buddy links chain pieces into full-width rows
  3. match_below    which full row sits below which        (1 request)
  4. assemble       all links -> grid, placements, needs-human list

Blank pieces have no text to judge, so they are never sent to the model.
They fill the cells that are left when everything else is in place.

The answer key never enters this module.
"""

import time
from dataclasses import dataclass

from .assembler import CONFIDENCE_THRESHOLD, Assembly, assemble
from .drex import DEFAULT_MODEL, ask as default_ask
from .matcher import MatchResult, match_below, match_right
from .shredder import Piece


@dataclass
class RunResult:
    match: MatchResult
    assembly: Assembly
    elapsed_s: float


def run(pieces: list[Piece], rows: int, cols: int, ask_fn=default_ask,
        model: str = DEFAULT_MODEL, strategy: str = "noul",
        threshold: float = CONFIDENCE_THRESHOLD, emit=None,
        workers: int = 4, prior: MatchResult | None = None,
        forced=(), **ask_kwargs) -> RunResult:
    """Rebuild the page. `emit` receives one dict per event.

    Pass `prior` (an earlier MatchResult) and `forced` (links a person
    chose) to rebuild after human input without redoing step 1.
    """
    emit = emit or (lambda event: None)
    by_id = {p.piece_id: p for p in pieces}
    t0 = time.monotonic()

    def on_call(phase: str, piece_id: str, res) -> None:
        emit({"type": "call", "phase": phase, "piece": piece_id,
              "latency_ms": round(res.latency_ms, 1),
              "input_tokens": res.input_tokens})

    blank = [p.piece_id for p in pieces if not "".join(p.lines).strip()]
    narrow = min(len(p.lines[0]) for p in pieces)
    wide = [i for i in blank if len(by_id[i].lines[0]) > narrow]
    live = [p for p in pieces if p.piece_id not in blank]
    if prior is None:
        match = match_right(live, ask_fn, model, strategy, on_call,
                            workers, **ask_kwargs)
    else:
        match = prior
        emit({"type": "reset"})
    row_links = [f for f in forced if f.direction == "right"]
    rows_only = assemble(match.right, {}, rows, cols, threshold,
                         match.right_edge, forced=row_links)
    strips = [[by_id[i] for i in row] for row in full_rows(
        rows_only, cols, match.right_edge, spare=len(blank))]
    emit({"type": "rows", "full_rows": len(strips), "rows": rows})
    match_below(match, strips, ask_fn, model, strategy, on_call,
                **ask_kwargs)
    assembly = assemble(match.right, match.below, rows, cols, threshold,
                        match.right_edge, match.bottom_edge, forced, blank,
                        wide)
    for event in assembly.events:
        emit(event)
    for item in assembly.needs_human:
        emit({"type": "needs_human", **item})
    return RunResult(match, assembly, time.monotonic() - t0)


def full_rows(assembly: Assembly, cols: int, right_edge=None,
              spare: int = 0) -> list[list[str]]:
    """Piece ids of every rebuilt row that reaches the right of the page.

    A row counts when it spans all columns. A shorter row counts too when
    it ends in a right-edge piece and `spare` blank pieces can fill the
    gaps of all such rows.
    """
    edge = right_edge or {}
    full, short = [], []
    for group in assembly.clusters:
        by_row: dict[int, dict[int, str]] = {}
        for p, (r, c) in group.items():
            by_row.setdefault(r, {})[c] = p
        for r in sorted(by_row):
            row = [by_row[r][c] for c in sorted(by_row[r])]
            if len(row) == cols:
                full.append(row)
            elif edge.get(row[-1], 0.0) >= 0.5:
                short.append(row)
    if sum(cols - len(row) for row in short) <= spare:
        full += short
    return full
