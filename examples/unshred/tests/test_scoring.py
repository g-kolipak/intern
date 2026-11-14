"""Tests for unshred.scoring (pure logic, no network)."""
from types import SimpleNamespace

from unshred import scoring
from unshred.assembler import Assembly, Link

KEY = {"a": (0, 0), "b": (0, 1), "c": (1, 0), "d": (1, 1)}


def call(tokens=100, ms=10.0, cost=None):
    return SimpleNamespace(input_tokens=tokens, latency_ms=ms, cost_usd=cost)


def test_neighbor_accuracy_counts_correct_accepted_links():
    links = [Link("a", "b", "right", 0.9), Link("a", "c", "below", 0.9),
             Link("c", "b", "right", 0.9), Link("b", "a", "right", 0.9)]
    assert scoring.neighbor_accuracy(links, KEY) == (2, 4)
    assert scoring.neighbor_accuracy([], KEY) == (0, 0)


def test_position_accuracy_counts_pieces_in_the_right_cell():
    grid = {"a": (0, 0), "b": (1, 1), "d": (0, 1)}
    assert scoring.position_accuracy(grid, KEY) == (1, 4)
    assert scoring.position_accuracy(KEY, KEY) == (4, 4)


def test_identical_pieces_are_interchangeable():
    lines = {"a": ("x",), "b": ("  ",), "c": ("y",), "d": ("  ",)}
    grid = {"a": (0, 0), "d": (0, 1), "c": (1, 0), "b": (1, 1)}
    assert scoring.position_accuracy(grid, KEY) == (2, 4)
    assert scoring.position_accuracy(grid, KEY, lines) == (4, 4)
    swapped = [Link("a", "d", "right", 0.9)]
    assert scoring.neighbor_accuracy(swapped, KEY) == (0, 1)
    assert scoring.neighbor_accuracy(swapped, KEY, lines) == (1, 1)


def test_pick_stats_looks_at_the_top_score_only():
    right = {"a": {"b": 0.9, "c": 0.2, "d": 0.1},
             "b": {"a": 0.9, "c": 0.1, "d": 0.1},   # b has no right neighbor
             "c": {"a": 0.8, "b": 0.1, "d": 0.3},   # wrong pick
             "d": {}}
    assert scoring.pick_stats(right, KEY, "right") == (1, 2)
    below = {"a": {"c": 0.7, "b": 0.1}, "b": {"d": 0.2, "a": 0.6}}
    assert scoring.pick_stats(below, KEY, "below") == (1, 2)


def test_edge_stats():
    edge = {"a": 0.1, "b": 0.9, "c": 0.8, "d": 0.4}
    assert scoring.edge_stats(edge, KEY, cols=2) == (2, 4)


def test_usage_totals_and_cost():
    calls = [call(1000, 10.0), call(3000, 30.0)]
    u = scoring.usage(calls)
    assert u == {"calls": 2, "input_tokens": 4000, "latency_ms": 40.0,
                 "avg_latency_ms": 20.0, "cost_usd": None}
    assert scoring.usage(calls, usd_per_mtok=0.5)["cost_usd"] == 0.002
    reported = [call(1000, 10.0, cost=0.01), call(1000, 10.0, cost=0.02)]
    assert scoring.usage(reported, usd_per_mtok=99)["cost_usd"] == 0.03
    assert scoring.usage([])["avg_latency_ms"] == 0.0


def test_price_comes_from_the_environment(monkeypatch):
    monkeypatch.delenv("DREX_USD_PER_MTOK", raising=False)
    assert scoring.price_per_mtok("DREX_USD_PER_MTOK") is None
    monkeypatch.setenv("DREX_USD_PER_MTOK", "0.25")
    assert scoring.price_per_mtok("DREX_USD_PER_MTOK") == 0.25
    monkeypatch.setenv("DREX_USD_PER_MTOK", "cheap")
    assert scoring.price_per_mtok("DREX_USD_PER_MTOK") is None


def test_summary_has_the_scoreboard_numbers():
    asm = Assembly(grid={"a": (0, 0), "b": (0, 1)},
                   accepted=[Link("a", "b", "right", 0.9)],
                   needs_human=[{"piece": "c"}])
    s = scoring.summary(asm, KEY, [call()], elapsed_s=1.234)
    assert s["placed"] == 2 and s["total"] == 4
    assert s["neighbor_accuracy"] == 1.0 and s["position_accuracy"] == 0.5
    assert s["needs_human"] == 1 and s["elapsed_s"] == 1.23
    assert s["calls"] == 1 and s["cost_usd"] is None
