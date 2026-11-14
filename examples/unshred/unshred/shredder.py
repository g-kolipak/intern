"""Simulated shredder: cut a text page into a shuffled grid of pieces.

The page is wrapped to a fixed width and padded into a character
rectangle, then cut into ROWS x COLS equal cells. Cuts may land in the
middle of words; that is intended. Each piece gets a random ID.

The answer key (ID -> (row, col)) is returned separately and must NEVER
be sent to Drex or the baseline model.
"""

import random
import textwrap
import uuid
from dataclasses import dataclass, field

PAGE_WIDTH = 60


@dataclass(frozen=True)
class Piece:
    """One shredded cell: a block of fixed-width text lines."""

    piece_id: str
    lines: tuple[str, ...]


@dataclass
class Shred:
    """Shuffled pieces plus the hidden answer key."""

    pieces: list[Piece] = field(default_factory=list)
    answer_key: dict[str, tuple[int, int]] = field(default_factory=dict)
    rows: int = 0
    cols: int = 0


def _make_page(text: str, width: int) -> list[str]:
    """Wrap text to `width` and pad lines into a full rectangle row."""
    wrapped = textwrap.wrap(text, width=width) or [""]
    return [line.ljust(width) for line in wrapped]


def _grid_dims(n_lines: int, rows: int, cols: int,
               width: int) -> tuple[int, int, int]:
    """Row height, base column width, last column width."""
    row_h = -(-n_lines // rows)  # ceil division
    col_w = width // cols
    last_col_w = width - col_w * (cols - 1)
    return row_h, col_w, last_col_w


def shred(text: str, rows: int, cols: int, width: int = PAGE_WIDTH,
          seed: int | None = None) -> Shred:
    """Shred `text` into rows x cols shuffled pieces.

    Returns pieces in shuffled order and a separate answer key mapping
    piece ID to (row, col). Raises ValueError on bad input.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be at least 1")
    page = _make_page(text, width)
    if len(page) < rows:
        raise ValueError(
            f"text wraps to {len(page)} lines; need at least {rows} "
            f"for a {rows}x{cols} grid"
        )
    # Pad with blank lines so the rectangle divides evenly into `rows`.
    while len(page) % rows != 0:
        page.append(" " * width)
    row_h, col_w, last_col_w = _grid_dims(len(page), rows, cols, width)
    n_rows = len(page) // row_h

    pieces: list[Piece] = []
    answer_key: dict[str, tuple[int, int]] = {}
    for r in range(n_rows):
        for c in range(cols):
            x0 = c * col_w
            x1 = width if c == cols - 1 else x0 + col_w
            lines = tuple(page[r * row_h + i][x0:x1] for i in range(row_h))
            piece_id = uuid.uuid4().hex[:8]
            pieces.append(Piece(piece_id=piece_id, lines=lines))
            answer_key[piece_id] = (r, c)

    ids = [p.piece_id for p in pieces]
    if len(ids) != len(set(ids)) or set(ids) != set(answer_key):
        raise RuntimeError(
            f"piece/answer-key divergence: pieces={len(ids)}, "
            f"unique={len(set(ids))}, key={len(answer_key)}"
        )
    rng = random.Random(seed)
    rng.shuffle(pieces)
    return Shred(pieces=pieces, answer_key=answer_key,
                 rows=n_rows, cols=cols)
