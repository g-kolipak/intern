"""Tests for unshred.matcher and unshred.questions (no network)."""
import json

import pytest

from tests.fakes import FakeAsk, words
from unshred.matcher import match_below, match_right, strip_lines
from unshred.questions import below_request, join_right, show, stack
from unshred.shredder import Piece, shred

ROWS, COLS = 2, 3


@pytest.fixture
def shr():
    return shred(words(42), rows=ROWS, cols=COLS, seed=1)  # 6 full lines


def _rows(shr):
    by_id = {p.piece_id: p for p in shr.pieces}
    id_at = {pos: pid for pid, pos in shr.answer_key.items()}
    return [[by_id[id_at[(r, c)]] for c in range(shr.cols)]
            for r in range(shr.rows)]


def test_text_helpers():
    a, b = Piece("a", ("ab", "cd")), Piece("b", ("12", "34"))
    assert join_right(a, b) == "ab12\ncd34"
    assert join_right(a, Piece("c", ("12",))) == "ab12\ncd"
    assert stack(a.lines, b.lines) == "ab\ncd\n12\n34"
    assert show(("x ", " y")) == "[x ]\n[ y]"


def test_right_one_call_per_piece_with_metrics(shr):
    fake = FakeAsk(shr)
    res = match_right(shr.pieces, ask_fn=fake)
    assert len(fake.requests) == len(res.calls) == ROWS * COLS
    assert all(c.latency_ms == 12.5 and c.input_tokens == 100
               for c in res.calls)


def test_right_tables_cover_every_other_piece(shr):
    res = match_right(shr.pieces, ask_fn=FakeAsk(shr))
    ids = set(shr.answer_key)
    assert set(res.right) == set(res.right_edge) == ids
    for a in ids:
        assert set(res.right[a]) == ids - {a}
        assert all(0.0 <= s <= 1.0 for s in res.right[a].values())


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_right_scores_come_from_the_answers(shr, strategy):
    res = match_right(shr.pieces, ask_fn=FakeAsk(shr), strategy=strategy)
    id_at = {pos: pid for pid, pos in shr.answer_key.items()}
    a, right_b, other = id_at[(0, 0)], id_at[(0, 1)], id_at[(1, 0)]
    assert res.right[a][right_b] == 0.95
    assert res.right[a][other] < 0.1
    assert res.right_edge[a] == 0.05
    assert res.right_edge[id_at[(0, COLS - 1)]] == 0.95


def test_right_request_quotes_each_candidate(shr):
    fake = FakeAsk(shr)
    match_right(shr.pieces, ask_fn=fake, workers=1)
    by_id = {p.piece_id: p for p in shr.pieces}
    for req in fake.requests:
        state, questions = req["state"], req["questions"]
        assert set(state) == {"context", "left_piece"}
        others = {q[6:] for q in questions if q != "right_edge"}
        (a_id,) = set(by_id) - others
        assert state["left_piece"] == show(by_id[a_id].lines)
        assert set(questions) == {"right_edge"} | {f"right_{o}"
                                                   for o in others}
        for o in others:
            q = questions[f"right_{o}"]
            assert q["type"] == "noul"
            assert join_right(by_id[a_id], by_id[o]) in q["instructions"]
        texts = [q["instructions"] for q in questions.values()]
        assert len(set(texts)) == len(texts)  # same text => same answer


def test_answer_key_is_never_sent(shr):
    fake = FakeAsk(shr)
    res = match_right(shr.pieces, ask_fn=fake)
    match_below(res, _rows(shr), ask_fn=fake)
    for req in fake.requests:
        blob = json.dumps({"s": req["state"], "q": req["questions"]})
        assert "answer_key" not in blob
        assert all(isinstance(v, str) for v in req["state"].values())


def test_model_and_kwargs_are_passed_through(shr):
    fake = FakeAsk(shr)
    res = match_right(shr.pieces, ask_fn=fake, model="drex-test", timeout=5.0)
    assert all(r["kwargs"] == {"model": "drex-test", "timeout": 5.0}
               for r in fake.requests)
    assert res.calls[0].model == "drex-test"


def test_missing_answer_is_an_error(shr):
    fake = FakeAsk(shr)

    def broken(state, questions, **kw):
        res = fake(state, questions, **kw)
        res.body["answers"].pop("right_edge")
        return res

    with pytest.raises(ValueError, match="right_edge"):
        match_right(shr.pieces, ask_fn=broken, workers=1)


@pytest.mark.parametrize("strategy", ["noul", "choice"])
def test_below_scores_rows_and_copies_to_columns(shr, strategy):
    fake = FakeAsk(shr)
    res = match_right(shr.pieces, ask_fn=fake)
    before = len(res.calls)
    rows = _rows(shr)
    match_below(res, rows, ask_fn=fake, strategy=strategy)
    assert len(res.calls) == before + 1  # one request for all rows
    for c in range(COLS):
        top, low = rows[0][c].piece_id, rows[1][c].piece_id
        assert res.below[top] == {low: 0.95}
        assert res.below[low][top] < 0.1
        assert res.bottom_edge[top] == 0.05
        assert res.bottom_edge[low] == 0.95


def test_below_strip_with_blank_last_line_is_the_page_end():
    shr = shred(words(40), rows=4, cols=3, seed=2)  # 6 lines padded to 8
    fake = FakeAsk(shr)
    res = match_right(shr.pieces, ask_fn=fake)
    rows = _rows(shr)
    match_below(res, rows, ask_fn=fake)
    last = rows[-1][0].piece_id
    assert strip_lines(rows[-1])[-1] == ""
    assert res.bottom_edge[last] == 1.0
    assert set(res.below[last].values()) == {0.0}
    asked = fake.requests[-1]["questions"]
    assert not any(q.startswith(("below_r3__", "bottom_r3")) for q in asked)


def test_below_request_needs_two_strips_to_compare():
    state, questions = below_request({"r0": ["one line", "two"]})
    assert set(questions) == {"bottom_r0"}
    assert "strips" in state["context"]
