"""Tests for unshred.arrange (pure logic, no network)."""
from unshred.arrange import arrange, evidence, unplaced

GRID = [["a", "b"], ["c", "d"]]


def tables(grid, hit=0.9, miss=0.1):
    pos = {p: (r, c) for r, row in enumerate(grid) for c, p in enumerate(row)}
    at = {v: k for k, v in pos.items()}
    right = {a: {b: hit if at.get((r, c + 1)) == b else miss
                 for b in pos if b != a} for a, (r, c) in pos.items()}
    below = {a: {b: hit if at.get((r + 1, c)) == b else miss
                 for b in pos if b != a} for a, (r, c) in pos.items()}
    return right, below, pos


def test_evidence_sign_and_clip():
    assert evidence(0.5) == 0
    assert evidence(0.9) > 0 > evidence(0.1)
    assert evidence(1.0) < 10 and evidence(0.0) > -10


def test_perfect_scores_give_the_true_grid():
    right, below, pos = tables(GRID)
    cells, score, links = arrange(right, below)
    assert cells == pos
    assert {(a, b, d) for a, b, d, _ in links} == {
        ("a", "b", "right"), ("c", "d", "right"),
        ("a", "c", "below"), ("b", "d", "below")}


def test_block_that_agrees_beats_one_confident_wrong_link():
    # a b    true layout; Drex is very sure (0.97) that c sits below b,
    # c d    but the true block a-b / c-d supports itself with 4 links.
    right, below, pos = tables(GRID, hit=0.9, miss=0.1)
    below["b"]["c"] = 0.97
    cells, _, _ = arrange(right, below)
    assert cells == pos


def test_piece_with_no_support_is_left_for_a_human():
    right, below, _ = tables([["a", "b"]])
    right["z"], below["z"] = {"a": 0.2, "b": 0.3}, {"a": 0.1, "b": 0.2}
    for t in (right, below):
        t["a"]["z"], t["b"]["z"] = 0.1, 0.2
    cells, _, _ = arrange(right, below)
    assert "z" not in cells
    (item,) = unplaced(cells, right, below)
    assert item["piece"] == "z" and len(item["candidates"]) == 3
    assert item["candidates"][0]["score"] == 0.3
