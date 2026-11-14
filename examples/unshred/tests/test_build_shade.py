"""Tests for unshred.build and unshred.shade (no network)."""
import numpy as np

from unshred import build, shade, torn
from unshred.drex import CallResult

LINES = [f"line {i} of the test page, with some words" for i in range(20)]


def pieces():
    ps, truth = torn.tear(torn.page(LINES, 600, 800), 2, 2, seed=4)
    return {f"p{r}{c}": p for p, (r, c) in zip(ps, truth)}


def test_place_puts_neighbours_side_by_side_without_overlap():
    up = pieces()
    cells = {p: (int(p[1]), int(p[2])) for p in up}
    placed = build.place(up, cells, best={})
    assert set(placed) == set(up)
    assert placed["p01"][0] > placed["p00"][0]   # right of
    assert placed["p10"][1] > placed["p00"][1]   # below
    x0 = min(x for x, _ in placed.values())
    y0 = min(y for _, y in placed.values())
    W = max(x + up[p].shape[1] for p, (x, _) in placed.items()) - x0
    H = max(y + up[p].shape[0] for p, (_, y) in placed.items()) - y0
    count = np.zeros((H, W), np.int32)
    for p, (x, y) in placed.items():
        m = up[p][..., 3] > 0
        count[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] += m
    paper = sum(int((up[p][..., 3] > 0).sum()) for p in up)
    assert (count > 1).sum() < 0.02 * paper  # pieces touch, barely overlap


def test_choose_returns_drex_pick_or_none():
    def ask(state, questions, **kw):
        crit = list(questions["page"]["criteria"])
        probs = {c: (0.7 if c == "image_2" else 0.1) for c in crit}
        return CallResult(body={"answers": {"page": {
            "type": "choice", "probabilities": probs}}},
            latency_ms=1, input_tokens=1, request_id="r", model="m")
    page = np.zeros((40, 40, 3), np.uint8)
    pick, probs, _ = build.choose([page, page, page], ask_fn=ask)
    assert pick == 1 and probs["image_2"] == 0.7


def test_shade_grow_keeps_masks_when_nothing_is_shaded():
    photo, _ = torn.scatter(list(pieces().values()), "dark", seed=5)
    masks = [np.zeros(photo.shape[:2], np.uint8)]
    masks[0][100:200, 100:200] = 1
    out = shade.grow(photo, masks)
    assert len(out) == 1 and out[0].shape == masks[0].shape
    assert shade.grow(photo, []) == []
