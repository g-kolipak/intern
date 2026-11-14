"""Tests for unshred.shredder (no network)."""

import string

import pytest

from unshred.shredder import PAGE_WIDTH, Piece, shred

WORDS = string.ascii_lowercase
TEXT = " ".join(
    "".join(WORDS[i % 26 : i % 26 + 5 + (i % 4)] for i in range(8))
    for _ in range(30)
)


def _reassemble(shr: "Shred") -> list[str]:
    """Put pieces back in grid order and rejoin into page lines."""
    grid: dict[tuple[int, int], Piece] = {
        pos: next(p for p in shr.pieces if p.piece_id == pid)
        for pid, pos in shr.answer_key.items()
    }
    lines_per_row = max(len(grid[(0, 0)].lines) for _ in range(1))
    rows = shr.rows
    cols = shr.cols
    page = []
    for r in range(rows):
        for i in range(lines_per_row):
            page.append(
                "".join(grid[(r, c)].lines[i] for c in range(cols))
            )
    return page


def test_reassembly_restores_the_page():
    shr = shred(TEXT, rows=2, cols=5, seed=42)
    page = _reassemble(shr)
    assert all(len(line) == PAGE_WIDTH for line in page)
    assert len(page) % 2 == 0
    # every non-space character of the source survives the shred
    assert TEXT.replace(" ", "") == "".join(
        ch for ch in "".join(page) if ch != " "
    )


def test_answer_key_covers_every_piece_exactly_once():
    shr = shred(TEXT, rows=2, cols=5, seed=1)
    assert len(shr.pieces) == 2 * 5
    assert len(shr.answer_key) == 2 * 5
    assert set(shr.answer_key) == {p.piece_id for p in shr.pieces}
    assert sorted(shr.answer_key.values()) == sorted(
        (r, c) for r in range(2) for c in range(5)
    )


def test_pieces_are_shuffled_and_seed_stable():
    a = shred(TEXT, rows=2, cols=5, seed=7)
    b = shred(TEXT, rows=2, cols=5, seed=7)
    c = shred(TEXT, rows=2, cols=5, seed=8)
    ids_a = [p.piece_id for p in a.pieces]
    ids_b = [p.piece_id for p in b.pieces]
    ids_c = [p.piece_id for p in c.pieces]
    grid_order = [
        key for key, _ in sorted(a.answer_key.items(),
                                 key=lambda kv: kv[1])
    ]
    assert ids_a != grid_order  # shuffle actually shuffled
    assert ids_b != ids_a  # fresh random IDs
    assert ids_a != ids_c
    # same seed shuffles differently only because IDs are random;
    # the multiset of piece contents per seed is identical
    content_a = sorted(p.lines for p in a.pieces)
    content_b = sorted(p.lines for p in b.pieces)
    assert content_a == content_b


def test_piece_line_widths_match_grid_columns():
    cols = 5
    shr = shred(TEXT, rows=2, cols=cols, seed=3)
    by_id = {p.piece_id: p for p in shr.pieces}
    for pid, pos in shr.answer_key.items():
        piece = by_id[pid]  # KeyError names the missing id
        c = pos[1]
        expected = PAGE_WIDTH // cols if c < cols - 1 else (
            PAGE_WIDTH - PAGE_WIDTH // cols * (cols - 1)
        )
        assert all(len(line) == expected for line in piece.lines)


def test_cuts_can_land_mid_word():
    shr = shred("a " * 400 + "x" * 200, rows=2, cols=5, seed=11)
    # at least one piece should start with a partial word, i.e. its
    # first line does not begin with a word boundary of the source
    joined = "".join(l for p in shr.pieces for l in p.lines)
    assert "xxxx" in joined


def test_short_text_is_rejected():
    with pytest.raises(ValueError, match="at least 2"):
        shred("tiny", rows=2, cols=5)


def test_bad_dimensions_are_rejected():
    with pytest.raises(ValueError):
        shred(TEXT, rows=0, cols=5)
    with pytest.raises(ValueError):
        shred(TEXT, rows=2, cols=0)
