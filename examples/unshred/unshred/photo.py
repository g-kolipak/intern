"""Find torn paper pieces in a photo and cut them out along their edges.

Plain image processing, no model: Drex should only ever see clean pieces
with no table around them. On a quarter-size copy of the photo:
  1. The table is the biggest smooth region. Whatever it cannot reach
     without crossing a sharp edge is a candidate piece.
  2. Smooth pockets in the table's colour (gaps, shadows) go back to the
     table. White paper the table "leaked" into is taken out again.
  3. GrabCut redraws each border on a half-size copy from local colours,
     which drops shadows. Pieces touching at a corner are split apart.
"""

import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .masks import largest as _largest, split as _split

# piece area as a share of the photo: only a noise floor and 'not the
# whole table'. What is really paper is Drex's call (scripts/rebuild_photo).
MIN_SHARE, MAX_SHARE, LEAK_SHARE = 0.0003, 0.8, 0.004  # leak: rescue size


def _el(k: int):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _cues(small):
    """Table mask and white-paper mask for a small BGR image."""
    h, w = small.shape[:2]
    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
    white = small.min(axis=2).astype(np.float32)  # paper high, table mid
    grad = np.zeros((h, w), np.float32)
    for chan, weight in ((lab[..., 0], 1.0), (lab[..., 1], 1.5),
                         (lab[..., 2], 1.5), (white, 1.0)):
        chan = cv2.GaussianBlur(chan, (0, 0), 1.5)
        grad = np.maximum(grad, weight * np.hypot(
            cv2.Sobel(chan, cv2.CV_32F, 1, 0, ksize=3),
            cv2.Sobel(chan, cv2.CV_32F, 0, 1, ksize=3)))
    smooth = (grad < 2.5 * float(np.median(grad))).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(smooth, 4)
    main = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    table = labels == main
    # local table level: 30th percentile of table pixels in each block
    blur, size = cv2.GaussianBlur(white, (0, 0), 3), 64
    gh, gw = -(-h // size), -(-w // size)
    grid = np.full((gh, gw), np.nan, np.float32)
    for i in range(gh):
        for j in range(gw):
            box = np.s_[i * size:(i + 1) * size, j * size:(j + 1) * size]
            vals = blur[box][table[box]]
            if vals.size > 0.2 * size * size:
                grid[i, j] = np.percentile(vals, 30)
    known = ~np.isnan(grid)
    level = (cv2.GaussianBlur(np.where(known, grid, 0), (0, 0), 1.2)
             / np.maximum(cv2.GaussianBlur(known.astype(np.float32),
                                           (0, 0), 1.2), 1e-3))
    level = cv2.resize(level, (gw * size, gh * size),
                       interpolation=cv2.INTER_CUBIC)[:h, :w]
    lum = cv2.GaussianBlur(lab[..., 0], (0, 0), 1.5)
    above = blur - level
    for k in range(1, n):  # table-coloured pockets belong to the table
        if k != main and stats[k, cv2.CC_STAT_AREA] >= 12:
            sel = labels == k
            if np.median(lum[sel]) >= 70 and np.median(above[sel]) < 10:
                table |= sel
    return table, above > 14


def _leaked_paper(table, bright):
    """White paper the table region leaked into: bright and edge-bound."""
    out = np.zeros(table.shape, bool)
    cand = cv2.morphologyEx((bright & table).astype(np.uint8),
                            cv2.MORPH_OPEN, _el(9))
    near_edge = cv2.dilate((~table).astype(np.uint8), _el(9)) > 0
    n, labels, stats, _ = cv2.connectedComponentsWithStats(cand, 4)
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < LEAK_SHARE * table.size:
            continue
        sel = (labels == k).astype(np.uint8)
        rim = (sel - cv2.erode(sel, _el(3))) > 0
        if near_edge[rim].mean() >= 0.5:
            out |= sel > 0
    return out


def _candidates(table, bright):
    """Solid blobs of 'not table', away from the photo's border."""
    h, w = table.shape
    leaked = _leaked_paper(table, bright)
    paper = ((~table) | leaked).astype(np.uint8)
    blobs = cv2.morphologyEx(paper, cv2.MORPH_OPEN, _el(7))
    found = []
    contours, _ = cv2.findContours(blobs, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    for c in contours:
        if not MIN_SHARE * h * w < cv2.contourArea(c) < MAX_SHARE * h * w:
            continue
        mask = np.zeros((h, w), np.uint8)
        cv2.drawContours(mask, [c], -1, 1, -1)
        if (leaked & (mask > 0)).any():
            mask = _close_outline(mask, paper)
        found.append(mask)
    return found


def _close_outline(mask, paper):
    """Join small breaks in the thin outline of a pale piece, then fill."""
    near = cv2.dilate(mask, _el(61)) & paper
    n, labels = cv2.connectedComponents(near, connectivity=8)
    own = np.isin(labels, np.unique(labels[mask > 0])) & (near > 0)
    closed = cv2.morphologyEx(own.astype(np.uint8), cv2.MORPH_CLOSE, _el(17))
    return _largest(cv2.morphologyEx(_largest(closed), cv2.MORPH_OPEN,
                                     _el(7)))


def _refine(half, cand, others, table_color):
    """GrabCut one candidate on the half-size photo. Masks are half-size."""
    cv2.setRNGSeed(7)  # GrabCut starts at random: same answer every run
    ys, xs = np.where(cv2.dilate(cand, _el(101)) > 0)
    box = np.s_[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    crop = half[box].copy()
    crop[others[box] > 0] = table_color  # hide the neighbours' colours
    gc = np.full(crop.shape[:2], cv2.GC_BGD, np.uint8)
    gc[cv2.dilate(cand, _el(53))[box] > 0] = cv2.GC_PR_BGD
    gc[cand[box] > 0] = cv2.GC_PR_FGD
    gc[cv2.erode(cand, _el(37))[box] > 0] = cv2.GC_FGD
    gc[others[box] > 0] = cv2.GC_BGD
    try:
        cv2.grabCut(crop, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 5,
                    cv2.GC_INIT_WITH_MASK)
    except cv2.error:  # no table around it (fills its crop): keep as found
        return cand
    fg = ((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)).astype(np.uint8)
    out = np.zeros(half.shape[:2], np.uint8)
    out[box] = cv2.morphologyEx(fg, cv2.MORPH_OPEN, _el(5))
    return out


def find_pieces(image, work: int | None = None) -> list:
    """Masks, one per candidate piece (see _find). The working size is
    picked from the photo: if the pieces come out small at 1280 px, look
    again at twice the resolution, so small scraps keep their detail."""
    if work:
        return _find(image, work)
    masks = _find(image, 1280)
    longest = max(image.shape[:2])
    if masks and longest > 1280:
        scale = 1280 / longest
        typical = np.median([m.sum() for m in masks]) * scale * scale
        if typical < 60 * 60:
            return _find(image, min(2560, longest))
    return masks


def _find(image, work: int) -> list:
    """Masks (uint8 0/1, same size as `image`), one per torn piece, in
    reading order of their centres: top to bottom, then left to right."""
    big_h, big_w = image.shape[:2]
    scale = work / max(big_h, big_w)  # the "quarter" working size
    small = cv2.resize(image, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_AREA)
    half = cv2.resize(image, (small.shape[1] * 2, small.shape[0] * 2),
                      interpolation=cv2.INTER_AREA)
    table, bright = _cues(small)
    size = (half.shape[1], half.shape[0])

    def up(m):
        return cv2.resize(m.astype(np.uint8), size,
                          interpolation=cv2.INTER_NEAREST)

    cands = [up(c) for c in _candidates(table, bright)]
    table_color = np.median(half[up(table) > 0], axis=0)
    pieces = []
    grown = [cv2.dilate(c, _el(9)) for c in cands]

    def refine(i):  # OpenCV releases the GIL, so threads run side by side
        others = np.zeros(half.shape[:2], np.uint8)
        for j, g in enumerate(grown):
            if j != i:
                others |= g
        return _refine(half, cands[i], others, table_color)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        results = list(pool.map(refine, range(len(cands))))
    for refined in results:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(refined, 4)
        for k in range(1, n):
            if stats[k, cv2.CC_STAT_AREA] >= MIN_SHARE * refined.size / 2:
                pieces += _split(_largest(labels == k), solid=0.85)
    pieces = [cv2.resize(p, (big_w, big_h), interpolation=cv2.INTER_NEAREST)
              for p in pieces if p.sum() >= MIN_SHARE * p.size / 2]

    def centre(mask):
        m = cv2.moments(mask, binaryImage=True)
        return (round(m["m01"] / m["m00"] / (big_h / 6)), m["m10"] / m["m00"])
    return sorted(pieces, key=centre)
