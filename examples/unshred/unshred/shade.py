"""Win back paper that is bent into shadow (pure OpenCV, no decisions).

On a pale, warm table, a piece bent into shadow is no brighter than the
table, so photo.py can leave that part out. It still differs in three
ways: paper is bluer than the warm table, a cast shadow would be darker
than the table but shaded paper is not, and paper has none of the
table's speckle. Each piece grows into neighbouring areas that are all
three; a second pass near the piece drops the speckle test so printed
text comes along. GrabCut then redraws the border.
"""

import cv2
import numpy as np

from . import photo
from .masks import largest

BLUER, DARKER, SPECKLE, BAND = 4.0, 8.0, 0.02, 20


def _el(r: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,
                                                         2 * r + 1))


def _level(chan, table, pct: float, size: int = 64):
    """Smooth map of the table's own value of `chan`."""
    h, w = chan.shape
    gh, gw = -(-h // size), -(-w // size)
    grid = np.full((gh, gw), np.nan, np.float32)
    for i in range(gh):
        for j in range(gw):
            box = np.s_[i * size:(i + 1) * size, j * size:(j + 1) * size]
            vals = chan[box][table[box]]
            if vals.size > 0.2 * size * size:
                grid[i, j] = np.percentile(vals, pct)
    known = ~np.isnan(grid)
    level = (cv2.GaussianBlur(np.where(known, grid, 0), (0, 0), 1.2)
             / np.maximum(cv2.GaussianBlur(known.astype(np.float32),
                                           (0, 0), 1.2), 1e-3))
    return cv2.resize(level, (gw * size, gh * size),
                      interpolation=cv2.INTER_CUBIC)[:h, :w]


def _maps(small):
    """(strict, relaxed): areas that look like shaded paper."""
    table, _ = photo._cues(small)
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    yb = cv2.GaussianBlur(lab[..., 2], (0, 0), 3)
    lum = cv2.GaussianBlur(lab[..., 0], (0, 0), 3)
    bluer = _level(yb, table, 70) - yb
    darker = _level(lum, table, 50) - lum
    n, lb, st, _ = cv2.connectedComponentsWithStats((~table).astype(np.uint8))
    dots = np.isin(lb, [i for i in range(1, n) if st[i][4] < 40])
    dense = cv2.blur(dots.astype(np.float32), (25, 25))
    ink = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5) < 70
    relaxed = (bluer > BLUER) & (darker < DARKER) & ~ink
    strict = cv2.morphologyEx((relaxed & (dense < SPECKLE)).astype(np.uint8),
                              cv2.MORPH_OPEN, _el(2))
    return strict, relaxed.astype(np.uint8), table


def _add(m, area, others, near: int, close: int):
    """m plus the parts of `area` that touch it and no other piece."""
    n, lb = cv2.connectedComponents(area & (1 - others))
    ids = np.unique(lb[(cv2.dilate(m, _el(near)) > 0) & (lb > 0)])
    grown = m | np.isin(lb, ids).astype(np.uint8)
    return largest(cv2.morphologyEx(grown, cv2.MORPH_CLOSE, _el(close)))


def grow(image, masks: list) -> list:
    """Full-size masks with each piece's shaded paper added back."""
    if not masks:
        return masks
    big_h, big_w = image.shape[:2]
    scale = 1280 / max(big_h, big_w)
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    h, w = small.shape[:2]
    strict, relaxed, table = _maps(small)
    work = [cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            for m in masks]
    taken = np.clip(sum(work), 0, 1).astype(np.uint8)
    out = []
    for i, m in enumerate(work):
        others = np.clip(taken - m, 0, 1).astype(np.uint8)
        g = _add(m, strict, others, 7, 3)
        cs, _ = cv2.findContours(g, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        hull = np.zeros_like(g)
        cv2.drawContours(hull, [cv2.convexHull(max(cs, key=cv2.contourArea))],
                         -1, 1, -1)
        zone = cv2.dilate(hull, _el(BAND)) & (1 - cv2.dilate(others, _el(3)))
        out.append(_add(g, relaxed & zone, others, 2, 4))
    half = cv2.resize(image, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
    color = np.median(half[cv2.resize(table.astype(np.uint8), (w * 2, h * 2),
                                      interpolation=cv2.INTER_NEAREST) > 0],
                      axis=0)
    up2 = [cv2.resize(g, (w * 2, h * 2), interpolation=cv2.INTER_NEAREST)
           for g in out]
    final = []
    for i, g in enumerate(up2):
        if int(out[i].sum()) < 1.05 * int(work[i].sum()):
            final.append(masks[i])  # under 5% added: keep it as it was
            continue
        others = np.clip(sum(o for j, o in enumerate(up2) if j != i), 0, 1)
        r = photo._refine(half, g, cv2.dilate(others.astype(np.uint8),
                                              _el(4)), color)
        r = largest(r | cv2.erode(g, _el(4)))
        final.append(cv2.resize(r, (big_w, big_h),
                                interpolation=cv2.INTER_NEAREST))
    return final
