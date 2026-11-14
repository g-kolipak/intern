"""Pick the piece finder that fits the photo's background.

A plain, even background (a studio or rendered scene, like ShredBench's
grey) needs nothing clever: paper is brighter than it, so Otsu's
threshold separates them, and touching pieces stay apart because the
shadows between them are darker still. A real table with texture,
sunlight and objects at the edges goes to photo.find_pieces.
"""

import cv2
import numpy as np

from .photo import MIN_SHARE, find_pieces

PLAIN = 8.0  # lightness spread along the photo's edge below which it is plain


def is_plain(image) -> bool:
    s = 1280 / max(image.shape[:2])
    small = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    L = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    b = max(4, int(0.04 * min(L.shape)))
    ring = np.concatenate([L[:b].ravel(), L[-b:].ravel(),
                           L[:, :b].ravel(), L[:, -b:].ravel()])
    return float(ring.std()) < PLAIN


def _fill(mask):
    h, w = mask.shape
    ff = np.pad(mask, 1).copy()
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 1)
    return mask | (ff[1:-1, 1:-1] == 0).astype(np.uint8)


def bright_pieces(image) -> list:
    """Masks of the pieces that are brighter than a plain background."""
    L = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[..., 0]
    L = cv2.GaussianBlur(L, (0, 0), 2)
    _, m = cv2.threshold(L, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    m = _fill(cv2.morphologyEx(m, cv2.MORPH_OPEN, k))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    floor = max(MIN_SHARE, 0.003) * m.size
    return [(lab == i).astype(np.uint8) for i in range(1, n)
            if st[i][4] >= floor]


def find(image) -> list:
    """Candidate piece masks, full size, with the finder that fits."""
    return bright_pieces(image) if is_plain(image) else find_pieces(image)


def trim_sunlit_table(image, masks: list) -> list:
    """Cut sunlit table out of the candidate regions, then let regions
    that fall apart become separate candidates.

    Sunlight makes a pale table warm yellow (Lab b above 140) without
    making it as bright as paper in the same sun (paper is neutral, and
    brighter than about 205), so warm-but-not-bright pixels are table.
    """
    if not masks:
        return masks
    lab = cv2.cvtColor(cv2.GaussianBlur(image, (0, 0), 3), cv2.COLOR_BGR2LAB)
    warm = (lab[..., 2] > 140) & (lab[..., 0] < 205)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    out = []
    floor = max(MIN_SHARE, 0.003) * warm.size
    for m in masks:
        kept = cv2.morphologyEx((m.astype(bool) & ~warm).astype(np.uint8),
                                cv2.MORPH_OPEN, k)
        n, lab_, st, _ = cv2.connectedComponentsWithStats(kept)
        out += [_fill((lab_ == i).astype(np.uint8)) for i in range(1, n)
                if st[i][4] >= floor]
    return out


MERGE_Q = ("Image 1 is part of a photo of torn paper pieces on a table. The "
           "GREEN and the MAGENTA outlines mark two neighbouring regions. Are "
           "they one single piece of paper, only marked apart by a fold, a "
           "crease, a pen line or a shadow, rather than two separate pieces?")


def merge_touching(image, masks: list, ask_fn=None, sure: float = 0.7,
                   workers: int = 16):
    """Drex decides which touching regions are really one piece.

    Returns (masks, calls). Regions merge only on a confident yes.
    """
    from concurrent.futures import ThreadPoolExecutor

    from .drex import VISION_MODEL, ask as drex_ask
    from .tear import data_url
    ask_fn = ask_fn or drex_ask
    s = 1280 / max(image.shape[:2])
    small = [cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
             for m in masks]
    grow = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    pairs = [(i, j) for i in range(len(small)) for j in range(i + 1, len(small))
             if (cv2.dilate(small[i], grow) & small[j]).any()]
    img = cv2.resize(image, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)

    def one(pair):
        i, j = pair
        ys, xs = np.where((small[i] | small[j]) > 0)
        pad = 30
        y0, x0 = max(0, ys.min() - pad), max(0, xs.min() - pad)
        crop = img[y0:ys.max() + pad, x0:xs.max() + pad].copy()
        for m, color in ((small[i], (0, 220, 0)), (small[j], (255, 0, 255))):
            cs, _ = cv2.findContours(m[y0:ys.max() + pad, x0:xs.max() + pad]
                                     .copy(), cv2.RETR_EXTERNAL,
                                     cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(crop, cs, -1, color, 2)
        res = ask_fn("Regions of a photo of torn paper.", {"one": {
            "type": "noul", "instructions": MERGE_Q}}, model=VISION_MODEL,
            images=[data_url(crop, 720)])
        return pair, float(res.body["answers"]["one"]["noul"]), res

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(one, pairs))
    parent = list(range(len(masks)))

    def root(k):
        while parent[k] != k:
            k = parent[k]
        return k
    for (i, j), p, _ in results:
        if p >= sure:
            parent[root(j)] = root(i)
    groups: dict = {}
    for k, m in enumerate(masks):
        groups.setdefault(root(k), []).append(m)
    merged = [_fill(np.clip(sum(g), 0, 1).astype(np.uint8)) if len(g) > 1
              else g[0] for g in groups.values()]
    return merged, [r for _, _, r in results]
