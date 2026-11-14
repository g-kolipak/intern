"""Scoring: the only code that reads the answer key, and only after a run.

Two pieces with identical text (for example two blank ones) cannot be
told apart by anyone, so they count as interchangeable when `lines`
(piece id -> text lines) is given.
"""

import os

from .assembler import STEP


def _same(lines, p, q) -> bool:
    return p == q or (lines is not None and lines.get(p) == lines.get(q))


def _id_at(key: dict) -> dict:
    return {pos: pid for pid, pos in key.items()}


def neighbor_accuracy(accepted, key: dict, lines=None) -> tuple[int, int]:
    """(correct, total) over the links the assembler accepted."""
    id_at, good = _id_at(key), 0
    for link in accepted:
        r, c = key[link.a]
        dr, dc = STEP[link.direction]
        true = id_at.get((r + dr, c + dc))
        good += true is not None and _same(lines, true, link.b)
    return good, len(accepted)


def position_accuracy(grid: dict, key: dict, lines=None) -> tuple[int, int]:
    """(pieces in the right cell, all pieces)."""
    id_at = _id_at(key)
    good = sum(1 for p, pos in grid.items()
               if pos in id_at and _same(lines, id_at[pos], p))
    return good, len(key)


def pick_stats(table: dict, key: dict, direction: str,
               lines=None) -> tuple[int, int]:
    """Raw matcher quality: is the top-scored candidate the true neighbor?

    Counts only pieces that have a true neighbor in that direction and a
    scored row in the table.
    """
    id_at, good, total = _id_at(key), 0, 0
    dr, dc = STEP[direction]
    for a, row in table.items():
        r, c = key[a]
        true = id_at.get((r + dr, c + dc))
        if true is None or not row:
            continue
        total += 1
        good += _same(lines, true, max(row, key=row.get))
    return good, total


def edge_stats(edge: dict, key: dict, cols: int) -> tuple[int, int]:
    """(correct, total) for the 'is A on the right edge?' answers."""
    good = sum(1 for p, s in edge.items()
               if (s >= 0.5) == (key[p][1] == cols - 1))
    return good, len(edge)


def price_per_mtok(env_name: str) -> float | None:
    """USD per million input tokens from the environment, if set."""
    try:
        return float(os.environ[env_name])
    except (KeyError, ValueError):
        return None


def usage(calls, usd_per_mtok: float | None = None) -> dict:
    """Totals over a list of calls. cost_usd is None when unknown."""
    tokens = sum(c.input_tokens for c in calls)
    reported = [c.cost_usd for c in calls
                if getattr(c, "cost_usd", None) is not None]
    if reported:
        cost = sum(reported)
    elif usd_per_mtok is not None:
        cost = tokens / 1_000_000 * usd_per_mtok
    else:
        cost = None
    latency = sum(c.latency_ms for c in calls)
    return {
        "calls": len(calls),
        "input_tokens": tokens,
        "latency_ms": round(latency, 1),
        "avg_latency_ms": round(latency / len(calls), 1) if calls else 0.0,
        "cost_usd": None if cost is None else round(cost, 6),
    }


def summary(assembly, key: dict, calls, elapsed_s: float, lines=None,
            usd_per_mtok: float | None = None) -> dict:
    """The scoreboard numbers for one finished run."""
    n_ok, n_all = neighbor_accuracy(assembly.accepted, key, lines)
    p_ok, p_all = position_accuracy(assembly.grid, key, lines)
    return {
        "placed": len(assembly.grid), "total": p_all,
        "links": n_all, "links_correct": n_ok,
        "neighbor_accuracy": round(n_ok / n_all, 4) if n_all else 0.0,
        "position_accuracy": round(p_ok / p_all, 4) if p_all else 0.0,
        "needs_human": len(assembly.needs_human),
        "elapsed_s": round(elapsed_s, 2),
        **usage(calls, usd_per_mtok),
    }
