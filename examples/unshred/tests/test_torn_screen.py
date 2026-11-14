"""Tests for unshred.torn, screen, compare and masks (no network)."""
import numpy as np
import pytest

from unshred import compare, masks, screen, torn
from unshred.drex import CallResult

LINES = [f"line {i} of the test page, with some words" for i in range(20)]


@pytest.fixture(scope="module")
def pieces():
    ps, truth = torn.tear(torn.page(LINES, 600, 800), 2, 3, seed=2)
    return ps, truth


def test_tear_covers_the_page_once(pieces):
    ps, truth = pieces
    assert len(ps) == 6 and sorted(truth) == [(r, c) for r in range(2)
                                              for c in range(3)]
    assert sum(int((p[..., 3] > 0).sum()) for p in ps) == 600 * 800
    assert all(p.shape[2] == 4 for p in ps)


def test_scatter_keeps_pieces_apart(pieces):
    ps, _ = pieces
    photo, where = torn.scatter(ps, "pale", seed=3)
    assert photo.ndim == 3 and len(where) == 6
    centers = np.array([w["center"] for w in where])
    d = np.hypot(*(centers[:, None] - centers[None]).transpose(2, 0, 1))
    assert (d + np.eye(6) * 1e9).min() > 50


def fake(choice=None, noul=0.8):
    def ask(state, questions, **kw):
        ans = {}
        for q, spec in questions.items():
            if spec["type"] == "noul":
                ans[q] = {"type": "noul", "noul": noul}
            else:
                labels = list(spec["criteria"])
                pick = choice or labels[0]
                ans[q] = {"type": "choice", "choice": pick, "probabilities": {
                    lab: 0.9 if lab == pick else 0.1 / (len(labels) - 1)
                    for lab in labels}}
        return CallResult(body={"answers": ans}, latency_ms=1.0,
                          input_tokens=10, request_id="r", model="fake")
    return ask


def test_is_paper_asks_drex_about_every_candidate(pieces):
    ps, _ = pieces
    probs, calls = screen.is_paper({f"p{i}": p for i, p in enumerate(ps)},
                                   ask_fn=fake(noul=0.7))
    assert set(probs) == {f"p{i}" for i in range(6)}
    assert all(v == 0.7 for v in probs.values()) and len(calls) == 1


def test_shortlist_checks_all_pairs_when_few(pieces):
    ps, _ = pieces
    up = {f"p{i}": p for i, p in enumerate(ps)}
    jobs = screen.shortlist(up, k=2)
    pairs = {(a, b, d) for a, b, d, _ in jobs}
    assert len(pairs) == 6 * 5 * 2  # every ordered pair, both directions


def test_shortlist_keeps_a_few_per_side_when_many(pieces):
    ps, _ = pieces
    up = {f"p{i}": p for i, p in enumerate(ps)}
    jobs = screen.shortlist(up, k=1, keep=1, all_upto=2)
    pairs = {(a, b, d) for a, b, d, _ in jobs}
    assert len(pairs) <= 6 * 2 * 2  # one out + one in, per piece and side


def test_compare_turns_picks_into_tables(pieces):
    ps, _ = pieces
    up = {f"p{i}": p for i, p in enumerate(ps)}
    best = {f"{a}|{b}|{d}": {"p": 0.5, "dx": 10, "dy": 0}
            for d in ("right", "below") for a in up for b in up if a != b}
    right, below, calls = compare.compare(up, best, top=3, ask_fn=fake())
    assert set(right) == set(below) == set(up)
    assert all(0 <= p <= 1 for t in (right, below) for row in t.values()
               for p in row.values())
    assert len(calls) == 6 * 2 * 2  # each piece, each side, both ways


def test_masks_largest_and_split():
    m = np.zeros((60, 120), np.uint8)
    m[10:50, 5:50] = 1
    m[10:50, 70:115] = 1
    m[28:32, 50:70] = 1  # a thin neck joins two blobs
    parts = masks.split(m, radius=3, min_area=100)
    assert len(parts) == 2
    small = np.zeros_like(m)
    small[0:5, 0:5] = 1
    assert masks.largest(m | small).sum() < (m | small).sum()
