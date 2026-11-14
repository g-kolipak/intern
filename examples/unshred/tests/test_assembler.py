"""Tests for unshred.assembler (pure logic, no network)."""
from unshred.assembler import (
    CONFIDENCE_THRESHOLD,
    Link,
    _Board,
    assemble,
    collect_links,
)
from unshred.pipeline import full_rows

ROWS, COLS = 2, 3
KEY = {f"p{r}{c}": (r, c) for r in range(ROWS) for c in range(COLS)}


def tables(hit=0.9, miss=0.1, key=KEY):
    """Score tables a perfect matcher would produce for `key`."""
    at = {pos: pid for pid, pos in key.items()}
    right = {a: {b: (hit if at.get((r, c + 1)) == b else miss)
                 for b in key if b != a} for a, (r, c) in key.items()}
    below = {a: {b: (hit if at.get((r + 1, c)) == b else miss)
                 for b in key if b != a} for a, (r, c) in key.items()}
    return right, below


def test_threshold_starts_at_point_six():
    assert CONFIDENCE_THRESHOLD == 0.6


def test_collect_links_sorts_high_to_low():
    links = collect_links({"a": {"b": 0.2, "c": 0.9}}, {"a": {"b": 0.5}})
    assert [(l.b, l.direction, l.score) for l in links] == [
        ("c", "right", 0.9), ("b", "below", 0.5), ("b", "right", 0.2)]


def test_perfect_scores_rebuild_the_whole_grid():
    right, below = tables()
    asm = assemble(right, below, ROWS, COLS)
    assert asm.grid == KEY
    assert asm.complete and not asm.needs_human
    assert len(asm.accepted) == ROWS * (COLS - 1) + (ROWS - 1) * COLS


def test_one_event_per_placement_with_position_and_score():
    right, below = tables()
    asm = assemble(right, below, ROWS, COLS)
    assert sorted(e["piece"] for e in asm.events) == sorted(KEY)
    for e in asm.events:
        assert e["type"] == "place"
        assert (e["row"], e["col"]) == KEY[e["piece"]]
        assert e["score"] == 0.9 and e["provisional"] is False


def test_weak_links_are_not_placed_and_go_to_needs_human():
    right, below = tables(hit=0.5, miss=0.1)
    right["p00"]["p11"] = 0.3
    asm = assemble(right, below, ROWS, COLS)
    assert asm.grid == {} and asm.accepted == [] and asm.events == []
    item = next(n for n in asm.needs_human
                if n["piece"] == "p00" and n["direction"] == "right")
    assert [c["piece"] for c in item["candidates"]][:2] == ["p01", "p11"]
    assert [c["score"] for c in item["candidates"]][:2] == [0.5, 0.3]
    assert len(item["candidates"]) == 3


def test_edge_pieces_do_not_need_a_human():
    right, below = tables(hit=0.5, miss=0.1)
    edge = {p: (0.9 if c == COLS - 1 else 0.1) for p, (r, c) in KEY.items()}
    asm = assemble(right, below, ROWS, COLS, right_edge=edge,
                   bottom_edge={p: 0.9 for p in KEY})
    asked = {(n["piece"], n["direction"]) for n in asm.needs_human}
    assert asked == {(p, "right") for p, (r, c) in KEY.items()
                     if c < COLS - 1}


def test_link_must_be_the_best_option_for_both_pieces():
    right = {"a": {"b": 0.9, "c": 0.1}, "c": {"b": 0.95, "a": 0.1},
             "b": {"a": 0.1, "c": 0.1}}
    asm = assemble(right, {}, 1, 3)
    assert [(l.a, l.b) for l in asm.accepted] == [("c", "b")]


def test_board_refuses_two_pieces_in_one_cell():
    board = _Board("abcd", 3, 3)
    assert board.join(Link("a", "b", "right", 0.9))
    assert board.join(Link("c", "d", "below", 0.9))
    # d below b would move c onto b's cell
    assert not board.join(Link("b", "d", "below", 0.9))
    assert board.of["c"] != board.of["a"]


def test_board_refuses_groups_bigger_than_the_page():
    board = _Board("abc", 1, 2)
    assert board.join(Link("a", "b", "right", 0.9))
    assert not board.join(Link("b", "c", "right", 0.9))
    assert not board.join(Link("a", "c", "below", 0.9))


def test_board_checks_links_inside_one_group():
    board = _Board("abc", 2, 2)
    assert board.join(Link("a", "b", "right", 0.9))
    assert board.join(Link("a", "c", "below", 0.9))
    assert not board.join(Link("c", "b", "right", 0.9))  # b is not there
    assert board.join(Link("a", "b", "right", 0.9))      # already true


def test_forced_links_go_first_and_are_marked_human():
    right, below = tables(hit=0.5, miss=0.1)
    asm = assemble(right, below, ROWS, COLS,
                   forced=[Link("p00", "p01", "right", 0.0)])
    assert [(l.a, l.b, l.human) for l in asm.accepted] == [
        ("p00", "p01", True)]
    assert {e["piece"] for e in asm.events} == {"p00", "p01"}
    assert all(e["human"] and e["provisional"] for e in asm.events)
    assert not any(n["piece"] == "p00" and n["direction"] == "right"
                   for n in asm.needs_human)
    taken = [c["piece"] for n in asm.needs_human
             if n["direction"] == "right" for c in n["candidates"]]
    assert "p01" not in taken  # a placed piece is no longer offered


def test_full_rows_lists_rows_that_span_the_page():
    right, _ = tables()
    asm = assemble(right, {}, ROWS, COLS)
    assert sorted(full_rows(asm, COLS)) == [
        ["p00", "p01", "p02"], ["p10", "p11", "p12"]]
    assert not asm.complete
    assert all(e["provisional"] for e in asm.events)
    right["p00"]["p01"] = 0.2  # break one row
    assert full_rows(assemble(right, {}, ROWS, COLS), COLS) == [
        ["p10", "p11", "p12"]]


def test_single_piece_page():
    asm = assemble({"only": {}}, {}, 1, 1)
    assert asm.grid == {"only": (0, 0)} and asm.complete
    assert [e["piece"] for e in asm.events] == ["only"]


def test_blank_fillers_take_the_free_cells_and_wide_ones_the_last_column():
    key = {p: pos for p, pos in KEY.items() if p not in ("p02", "p12")}
    right, below = tables(key=key)
    asm = assemble(right, below, ROWS, COLS, fillers=["p02", "p12"])
    assert asm.complete and len(asm.grid) == 6
    assert {asm.grid["p02"], asm.grid["p12"]} == {(0, 2), (1, 2)}
    assert [e["piece"] for e in asm.events if e["blank"]] == ["p02", "p12"]
    key = {p: pos for p, pos in KEY.items() if p not in ("p11", "p12")}
    right, below = tables(key=key)
    asm = assemble(right, below, ROWS, COLS, fillers=["p11", "p12"],
                   wide=["p12"])
    assert asm.grid["p12"] == (1, 2) and asm.grid["p11"] == (1, 1)
    alone = assemble(*tables(hit=0.5, key=key), ROWS, COLS, fillers=["p11"])
    assert "p11" not in alone.grid  # nothing is sure yet, so blanks wait
