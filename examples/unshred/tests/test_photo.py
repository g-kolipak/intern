"""Tests for unshred.photo and unshred.cutouts on a synthetic table photo."""
import cv2
import numpy as np
import pytest

from unshred import cutouts
from unshred.photo import find_pieces

H, W = 900, 1200


def _paper(kind, angle, center, size):
    """One piece: BGR texture, its mask and its true tilt, on the canvas."""
    w, h = size
    tile = np.full((h, w, 3), 236 if kind == "white" else 28, np.uint8)
    ink = 30 if kind == "white" else 235
    for y in range(24, h - 20, 26):  # lines of fake text
        for x in range(18, w - 30, 34):
            cv2.rectangle(tile, (x, y), (x + 22, y + 9), (ink, ink, ink), -1)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    m[0, 2] += center[0] - w / 2
    m[1, 2] += center[1] - h / 2
    img = cv2.warpAffine(tile, m, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpAffine(np.full((h, w), 255, np.uint8), m, (W, H))
    return img, mask > 127


@pytest.fixture(scope="module")
def scene():
    rng = np.random.default_rng(5)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 175 + 25 * xx / W - 20 * yy / H            # uneven lighting
    grain = cv2.GaussianBlur(rng.normal(0, 14, (H, W)).astype(np.float32),
                             (0, 0), 1.6)               # table texture
    table = np.stack([light + grain - 14, light + grain - 4, light + grain],
                     -1)                                # cream: less blue
    photo = np.clip(table, 0, 255).astype(np.uint8)
    truth = []
    for kind, angle, center, size in (("white", 20, (300, 260), (300, 220)),
                                      ("black", -35, (820, 300), (280, 240)),
                                      ("white", 70, (560, 640), (320, 200))):
        img, mask = _paper(kind, angle, center, size)
        photo[mask] = img[mask]
        truth.append(mask)
    return photo, truth


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_finds_every_piece_with_a_tight_border(scene):
    photo, truth = scene
    masks = find_pieces(photo)
    assert len(masks) == len(truth)
    for want in truth:
        best = max(_iou(m > 0, want) for m in masks)
        assert best > 0.93
    assert all(m.shape == photo.shape[:2] and m.dtype == np.uint8
               for m in masks)


def test_pieces_come_back_in_reading_order(scene):
    photo, _ = scene
    centres = [cv2.moments(m, binaryImage=True) for m in find_pieces(photo)]
    xs = [c["m10"] / c["m00"] for c in centres]
    ys = [c["m01"] / c["m00"] for c in centres]
    assert xs[0] < xs[1] and ys[2] > max(ys[:2])


def test_an_empty_table_has_no_pieces(scene):
    photo, truth = scene
    covered = np.zeros(photo.shape[:2], bool)
    for m in truth:
        covered |= m
    empty = photo.copy()
    empty[covered] = np.median(photo[~covered], axis=0)
    assert find_pieces(empty) == []


def test_cutout_is_transparent_outside_the_piece(scene):
    photo, truth = scene
    mask = truth[1].astype(np.uint8)
    piece = cutouts.cutout(photo, mask, pad=10)
    assert piece.shape[2] == 4
    assert piece[0, 0, 3] == 0 and piece[-1, -1, 3] == 0
    cy, cx = piece.shape[0] // 2, piece.shape[1] // 2
    assert piece[cy, cx, 3] == 255
    flat = cutouts.on_canvas(piece, color=(1, 2, 3))
    assert tuple(flat[0, 0]) == (1, 2, 3)


@pytest.mark.parametrize("index,angle", [(0, 20), (1, -35), (2, 70)])
def test_straighten_undoes_the_tilt(scene, index, angle):
    photo, truth = scene
    piece = cutouts.cutout(photo, truth[index].astype(np.uint8))
    tilt = cutouts.skew_angle(piece)
    # getRotationMatrix2D turns anticlockwise, so the tilt is -angle mod 90
    want = ((-angle + 45) % 90) - 45
    assert abs(tilt - want) < 2.0
    assert abs(cutouts.skew_angle(cutouts.straighten(piece))) < 2.0


def test_rotate_keeps_the_whole_piece():
    piece = np.zeros((40, 100, 4), np.uint8)
    piece[..., :3], piece[..., 3] = 200, 255
    turned = cutouts.rotate(piece, 90)
    assert abs(turned.shape[0] - 100) <= 3 and abs(turned.shape[1] - 40) <= 3
    assert (turned[..., 3] > 128).sum() == pytest.approx(4000, rel=0.06)


def test_outline_sheet_and_compose_make_pictures(scene):
    photo, truth = scene
    masks = [m.astype(np.uint8) for m in truth]
    assert cutouts.outline(photo, masks).shape == photo.shape
    pieces = {f"p{i}": cutouts.cutout(photo, m) for i, m in enumerate(masks)}
    assert cutouts.sheet(list(pieces.values()), cols=2,
                         cell=200).shape == (400, 400, 3)
    page = cutouts.compose(pieces, [{"p0": (0, 0), "p1": (0, 1)}], ["p2"])
    assert page.ndim == 3 and page.shape[2] == 3
    widths = pieces["p0"].shape[1] + pieces["p1"].shape[1]
    assert page.shape[1] >= widths


def test_layout_puts_groups_side_by_side_and_loose_pieces_below():
    sizes = {"a": (100, 200), "b": (120, 150), "c": (80, 90), "d": (50, 60)}
    pos, (width, height) = cutouts.layout(
        sizes, [{"a": (0, 0), "b": (0, 1)}, {"c": (0, 0)}], ["d"], gap=10)
    assert pos["a"][0] < pos["b"][0] < pos["c"][0]      # left to right
    assert pos["b"][0] >= pos["a"][0] + 200             # no overlap
    assert pos["d"][1] >= max(pos[p][1] + sizes[p][0] for p in "abc")
    assert width >= pos["c"][0] + 90 and height >= pos["d"][1] + 50
    assert cutouts.layout(sizes, [], ["a", "d"], gap=10)[0]["d"][0] == 220
