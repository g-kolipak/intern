"""Matcher: ask Drex (or the baseline) which pieces are neighbors.

Phase 1, match_right: one request per piece A. It scores every other
piece as A's right neighbor and asks whether A is on the right edge.

Phase 2, match_below: pieces narrower than the page carry no top-to-bottom
continuity (line 2 of a piece does not continue line 1), so stacking two
single pieces has no signal. Rows that phase 1 rebuilt to full width do:
the last line of a row runs on into the first line of the next row. One
request stacks every full row on every other and scores the result, and
the row score is copied to the pieces that sit in the same column.

Output: right[A][B] and below[A][B], values 0 to 1, plus per-call metrics
(latency ms, input tokens). The answer key is never part of any request.
"""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from .drex import DEFAULT_MODEL, CallResult, ask as default_ask
from .questions import below_request, right_request
from .shredder import Piece

Table = dict[str, dict[str, float]]


@dataclass
class MatchResult:
    right: Table = field(default_factory=dict)
    below: Table = field(default_factory=dict)
    right_edge: dict[str, float] = field(default_factory=dict)
    bottom_edge: dict[str, float] = field(default_factory=dict)
    calls: list[CallResult] = field(default_factory=list)


def _noul(answers: dict, qid: str) -> float:
    try:
        return float(answers[qid]["noul"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"no noul answer came back for {qid!r}") from e


def _probs(answers: dict, qid: str) -> dict:
    try:
        return dict(answers[qid]["probabilities"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"no choice answer came back for {qid!r}") from e


def match_right(pieces: list[Piece], ask_fn=default_ask,
                model: str = DEFAULT_MODEL, strategy: str = "noul",
                on_call=None, workers: int = 4, **ask_kwargs) -> MatchResult:
    """Score every ordered pair (A, B) as 'B sits directly right of A'."""

    def one(a: Piece):
        others = [p for p in pieces if p.piece_id != a.piece_id]
        state, questions = right_request(a, others, strategy)
        res = ask_fn(state, questions, model=model, **ask_kwargs)
        answers = res.body["answers"]
        if strategy == "choice" and others:
            probs = _probs(answers, "right_pick")
            row = {o.piece_id: float(probs.get(o.piece_id, 0.0))
                   for o in others}
        else:
            row = {o.piece_id: _noul(answers, f"right_{o.piece_id}")
                   for o in others}
        edge = _noul(answers, "right_edge")
        if on_call:
            on_call("right", a.piece_id, res)
        return a.piece_id, row, edge, res

    result = MatchResult()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for pid, row, edge, res in pool.map(one, pieces):
            result.right[pid] = row
            result.right_edge[pid] = edge
            result.calls.append(res)
    return result


def strip_lines(row: list[Piece]) -> list[str]:
    """The full-width text lines of one rebuilt row of pieces."""
    height = max(len(p.lines) for p in row)
    return ["".join(p.lines[j] if j < len(p.lines) else "" for p in row)
            .rstrip() for j in range(height)]


def match_below(result: MatchResult, rows: list[list[Piece]],
                ask_fn=default_ask, model: str = DEFAULT_MODEL,
                strategy: str = "noul", on_call=None,
                **ask_kwargs) -> MatchResult:
    """Score every ordered pair of full rows as 'Y sits directly below X'.

    Fills result.below and result.bottom_edge for the pieces in `rows`.
    """
    result.below, result.bottom_edge = {}, {}
    strips = {f"r{i}": strip_lines(row) for i, row in enumerate(rows)}
    state, questions = below_request(strips, strategy)
    answers: dict = {}
    if questions:
        res = ask_fn(state, questions, model=model, **ask_kwargs)
        result.calls.append(res)
        answers = res.body["answers"]
        if on_call:
            on_call("below", rows[0][0].piece_id, res)
    for i, top in enumerate(rows):
        x = f"r{i}"
        ended = f"bottom_{x}" not in questions  # blank last line: page end
        bottom = 1.0 if ended else _noul(answers, f"bottom_{x}")
        picks = (_probs(answers, f"below_pick_{x}")
                 if f"below_pick_{x}" in questions else None)
        for j, low in enumerate(rows):
            if i == j:
                continue
            y = f"r{j}"
            if ended:
                score = 0.0
            elif picks is not None:
                score = float(picks.get(y, 0.0))
            else:
                score = _noul(answers, f"below_{x}__{y}")
            for pa, pb in zip(top, low):
                result.below.setdefault(pa.piece_id, {})[pb.piece_id] = score
        for p in top:
            result.bottom_edge[p.piece_id] = bottom
    return result
