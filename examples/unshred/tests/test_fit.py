"""Tests for unshred.fit (pure geometry, no network)."""
import numpy as np

from unshred import fit


def page(h=300, w=400):
    """A white page with dark text-like lines at irregular heights."""
    img = np.full((h, w, 4), 255, np.uint8)
    y = 12
    rng = np.random.default_rng(0)
    while y < h - 12:
        img[y:y + 6, 10:w - 10, :3] = 20
        y += int(rng.integers(14, 30))
    return img


def tear_lr(img, x):
    """Left and right halves of a page, torn along a wavy vertical line."""
    h = img.shape[0]
    cut = (x + 6 * np.sin(np.arange(h) / 9)).astype(int)
    left, right = img.copy(), img.copy()
    cols = np.arange(img.shape[1])[None, :]
    left[..., 3] = np.where(cols < cut[:, None], 255, 0)
    right[..., 3] = np.where(cols >= cut[:, None], 255, 0)
    return left, right


def test_text_rhythm_finds_the_true_vertical_offset():
    left, right = tear_lr(page(), 200)
    shift = 37  # the right half was cropped 37 rows lower in its own image
    b = right[shift:]
    dx, dy, score = fit.fit(left, b, "right")
    assert dy == shift and score > 0.9
    assert abs(dx) <= 2  # pushed in until the torn edges touch


def test_below_is_the_transposed_problem():
    img = np.ascontiguousarray(np.transpose(page(400, 300), (1, 0, 2)))
    top, bottom = (np.ascontiguousarray(np.transpose(p, (1, 0, 2)))
                   for p in tear_lr(np.transpose(img, (1, 0, 2)), 200))
    dx, dy, _ = fit.fit(top, bottom[:, 21:], "below")
    assert dx == 21


def test_several_alignments_are_kept_apart():
    left, right = tear_lr(page(), 200)
    picks = fit.fits(left, right, "right", k=3)
    assert len(picks) == 3
    dys = sorted(p[1] for p in picks)
    assert all(b - a >= 6 for a, b in zip(dys, dys[1:]))


def test_compose_and_joined_draw_both_pieces():
    left, right = tear_lr(page(), 200)
    img = fit.joined(left, right, (0, 0), maxdim=200)
    assert img.shape[2] == 3 and max(img.shape[:2]) == 200
    canvas = fit.compose({"a": left, "b": right}, {"a": (0, 0),
                                                   "b": (0, 300)})
    assert canvas.shape[:2] == (600, 400)
