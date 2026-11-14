"""Tests for unshred.pipeline: the whole rebuild with a fake Drex."""
import json

import pytest

from tests.fakes import FakeAsk, words
from unshred import scoring
from unshred.assembler import Link
from unshred.pipeline import run
from unshred.shredder import shred


@pytest.mark.parametrize("strategy", ["noul", "choice"])
@pytest.mark.parametrize("n_words,rows,cols", [(42, 2, 3), (40, 4, 3),
                                               (140, 5, 6)])
def test_good_answers_rebuild_the_page(strategy, n_words, rows, cols):
    shr = shred(words(n_words), rows=rows, cols=cols, seed=3)
    fake = FakeAsk(shr)
    res = run(shr.pieces, shr.rows, shr.cols, ask_fn=fake, strategy=strategy)
    lines = {p.piece_id: p.lines for p in shr.pieces}
    blank = sum(1 for text in lines.values() if not "".join(text).strip())
    n = len(shr.pieces)  # blank pieces are interchangeable
    assert scoring.position_accuracy(
        res.assembly.grid, shr.answer_key, lines) == (n, n)
    assert res.assembly.complete and not res.assembly.needs_human
    assert len(res.match.calls) == n - blank + 1  # blanks are never sent
    n = scoring.neighbor_accuracy(res.assembly.accepted, shr.answer_key)
    assert n[0] == n[1] > 0


@pytest.mark.parametrize("rows,cols", [(r, c) for r in (2, 5, 6, 8, 10)
                                       for c in (5, 6, 8, 10)])
def test_sample_invoice_rebuilds_at_every_slider_size(rows, cols):
    text = open("data/sample.txt").read()
    shr = shred(text, rows=rows, cols=cols, seed=7)
    res = run(shr.pieces, shr.rows, shr.cols, ask_fn=FakeAsk(shr))
    lines = {p.piece_id: p.lines for p in shr.pieces}
    n = len(shr.pieces)
    assert scoring.position_accuracy(
        res.assembly.grid, shr.answer_key, lines) == (n, n)


def test_events_are_emitted_in_order():
    shr = shred(words(42), rows=2, cols=3, seed=3)
    events = []
    run(shr.pieces, 2, 3, ask_fn=FakeAsk(shr), emit=events.append, workers=1)
    kinds = [e["type"] for e in events]
    assert kinds == ["call"] * 6 + ["rows", "call"] + ["place"] * 6
    assert events[6] == {"type": "rows", "full_rows": 2, "rows": 2}
    assert {e["phase"] for e in events if e["type"] == "call"} == {
        "right", "below"}
    assert all(e["latency_ms"] == 12.5 and e["input_tokens"] == 100
               for e in events if e["type"] == "call")
    placed = {e["piece"]: (e["row"], e["col"]) for e in events
              if e["type"] == "place"}
    assert placed == shr.answer_key


def test_answer_key_never_reaches_the_model():
    shr = shred(words(42), rows=2, cols=3, seed=3)
    fake = FakeAsk(shr)
    run(shr.pieces, 2, 3, ask_fn=fake)
    blob = json.dumps([[r["state"], r["questions"]] for r in fake.requests])
    assert "answer_key" not in blob


def test_unsure_answers_place_nothing_and_ask_a_human():
    shr = shred(words(42), rows=2, cols=3, seed=3)
    events = []
    res = run(shr.pieces, 2, 3, ask_fn=FakeAsk(shr, hit=0.5, miss=0.1),
              emit=events.append)
    assert res.assembly.grid == {}
    assert len(res.match.calls) == 6  # no full rows, so no stacking call
    asks = [e for e in events if e["type"] == "needs_human"]
    assert len(asks) == 4 and all(e["direction"] == "right" for e in asks)
    assert all(len(e["candidates"]) == 3 for e in asks)


def test_human_links_rebuild_without_asking_for_rows_again():
    shr = shred(words(42), rows=2, cols=3, seed=3)
    id_at = {pos: pid for pid, pos in shr.answer_key.items()}
    fake = FakeAsk(shr, hit=0.5, miss=0.1)
    first = run(shr.pieces, 2, 3, ask_fn=fake)
    forced = [Link(id_at[(r, c)], id_at[(r, c + 1)], "right", 1.0)
              for r in range(2) for c in range(2)]
    fake.hit = 0.95  # the stacking question now gets a clear answer
    events = []
    again = run(shr.pieces, 2, 3, ask_fn=fake, prior=first.match,
                forced=forced, emit=events.append)
    assert events[0] == {"type": "reset"}
    assert len(again.match.calls) == 7  # 6 from before + 1 stacking call
    assert again.assembly.grid == shr.answer_key
    assert sum(1 for l in again.assembly.accepted if l.human) == 4
