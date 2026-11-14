"""Tests for unshred.holes (no network)."""
import numpy as np

from unshred import holes
from unshred.drex import CallResult


def sheet(h, w):
    p = np.zeros((h, w, 4), np.uint8)
    p[..., :3] = 230
    p[..., 3] = 255
    return p


def test_finds_the_gap_inside_the_page():
    # a 2x2 page of 200x200 pieces with the bottom-right one missing,
    # plus a loose piece far away that must not count
    up = {"a": sheet(200, 200), "b": sheet(200, 200), "c": sheet(200, 200),
          "z": sheet(200, 200)}
    placed = {"a": (0, 0), "b": (200, 0), "c": (0, 200), "z": (900, 900)}
    found = holes.holes(up, placed, skip="z", min_area=10_000)
    assert len(found) == 1
    cx, cy, area, _, _ = found[0]
    assert 250 < cx < 350 and 250 < cy < 350   # the missing corner
    assert area > 15_000


def test_fill_moves_a_dangling_piece_into_the_hole_when_drex_agrees():
    up = {"a": sheet(200, 200), "b": sheet(200, 200), "c": sheet(200, 200),
          "z": sheet(200, 200)}
    cells = {"a": (0, 0), "b": (0, 1), "c": (1, 0), "z": (2, 0)}
    placed = {"a": (0, 0), "b": (200, 0), "c": (0, 200), "z": (0, 400)}

    def ask(state, questions, **kw):
        if "spot" in questions:   # prefer the second image (the hole)
            labels = list(questions["spot"]["criteria"])
            probs = {lab: (0.8 if lab == "image_2" else 0.05) for lab in labels}
            ans = {"spot": {"type": "choice", "probabilities": probs}}
        else:                     # page looks at least as good afterwards
            ans = {"ok": {"type": "noul", "noul": 0.7}}
        return CallResult(body={"answers": ans}, latency_ms=1,
                          input_tokens=1, request_id="r", model="m")

    new, grid, calls = holes.fill(up, cells, placed, ask_fn=ask)
    x, y = new["z"]
    assert 100 < x < 300 and 100 < y < 300   # moved into the gap
    assert len(calls) == 3                   # pick + page before/after
    assert "z" not in grid                   # it sits in a gap, not a cell
