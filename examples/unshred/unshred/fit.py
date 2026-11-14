"""Fit two torn pieces together at their edges, like puzzle pieces.

Pure geometry, no decisions: for piece B to the right of (or below) piece
A, slide B along A's torn edge and keep the offset where the two edges
leave the smallest gap. Then draw both pieces pushed together at that
offset as one image, which is what Drex is asked about.

Pieces are BGRA cut-outs (alpha > 128 is paper), both upright.
"""

import cv2
import numpy as np

BACK = (30, 28, 28)  # dark background, as in cutouts.on_canvas
MIN_OVERLAP = 0.35   # the edges must share this much of the shorter side


def _edges(piece):
    """Rightmost paper column of each row of A, leftmost of each row of B."""
    m = piece[..., 3] > 128
    w = m.shape[1]
    has = m.any(axis=1)
    right = np.where(has, w - 1 - np.argmax(m[:, ::-1], axis=1), -1)
    left = np.where(has, np.argmax(m, axis=1), -1)
    return right, left


def _rows(piece, side: str, share: float = 0.18):
    """Print activity per row in a strip along one side of the piece.

    Lines of text make a rhythm of busy and quiet rows; on the true
    neighbour the same rhythm continues at the same heights.
    """
    gray = cv2.cvtColor(piece[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32)
    mag = np.abs(cv2.Sobel(cv2.GaussianBlur(gray, (0, 0), 1.5), cv2.CV_32F,
                           0, 1, ksize=3))
    m = piece[..., 3] > 128
    w = m.shape[1]
    band = max(8, int(share * w))
    right, left = _edges(piece)
    cols = np.arange(w)[None, :]
    if side == "right":
        near = (cols <= right[:, None]) & (cols > right[:, None] - band)
    else:
        near = (cols >= left[:, None]) & (cols < left[:, None] + band)
    near &= cv2.erode(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
    cnt = near.sum(axis=1)
    prof = np.where(cnt > 0, (mag * near).sum(axis=1) / np.maximum(cnt, 1), 0)
    return prof, cnt > 0


def fits_right(a, b, k: int = 3):
    """Up to k (dx, dy, score), best first: put B's top-left at (dx, dy)
    in A's frame.

    dy lines up the print rhythm along A's right edge with the one along
    B's left edge (normalised correlation, higher is better); dx then
    pushes B left until the torn edges touch. Text lines repeat, so the
    rhythm can also match a line or two off: the k answers are kept at
    least a few percent of the height apart, and Drex picks among them.
    """
    pa, ha_ok = _rows(a, "right")
    pb, hb_ok = _rows(b, "left")
    ra, _ = _edges(a)
    _, lb = _edges(b)
    ha, hb = len(pa), len(pb)
    need = int(MIN_OVERLAP * min(ha, hb))
    found = []
    for dy in range(-hb + need, ha - need + 1, 2):
        y0, y1 = max(0, dy), min(ha, hb + dy)
        ok = ha_ok[y0:y1] & hb_ok[y0 - dy:y1 - dy]
        if ok.sum() < need:
            continue
        u, v = pa[y0:y1][ok], pb[y0 - dy:y1 - dy][ok]
        u, v = u - u.mean(), v - v.mean()
        corr = float((u * v).sum() / (np.sqrt((u * u).sum() * (v * v).sum())
                                      + 1e-6))
        A, B = ra[y0:y1][ok], lb[y0 - dy:y1 - dy][ok]
        found.append((int((A - B).max()) + 1, dy, corr))
    apart = max(6, int(0.05 * min(ha, hb)))
    picks = []
    for f in sorted(found, key=lambda f: -f[2]):
        if all(abs(f[1] - p[1]) >= apart for p in picks):
            picks.append(f)
        if len(picks) == k:
            break
    return picks or [(0, 0, -1.0)]


def fit_right(a, b):
    return fits_right(a, b, 1)[0]


def _t(piece):
    return np.ascontiguousarray(np.transpose(piece, (1, 0, 2)))


def fits(a, b, direction: str, k: int = 3):
    """fits_right, or the same with B under A (the transposed problem)."""
    if direction == "right":
        return fits_right(a, b, k)
    return [(dy, dx, sc) for dx, dy, sc in fits_right(_t(a), _t(b), k)]


def fit(a, b, direction: str):
    return fits(a, b, direction, 1)[0]


def compose(pieces: dict, places: dict, back=BACK):
    """Draw pieces at their (x, y) places on one dark canvas, BGR."""
    xs = [places[p][0] for p in places] + [places[p][0] + pieces[p].shape[1]
                                           for p in places]
    ys = [places[p][1] for p in places] + [places[p][1] + pieces[p].shape[0]
                                           for p in places]
    x0, y0 = min(xs), min(ys)
    canvas = np.full((max(ys) - y0, max(xs) - x0, 3), back, np.float32)
    for p, (x, y) in places.items():
        img = pieces[p]
        h, w = img.shape[:2]
        alpha = img[..., 3:4].astype(np.float32) / 255
        box = canvas[y - y0:y - y0 + h, x - x0:x - x0 + w]
        box[:] = img[..., :3] * alpha + box * (1 - alpha)
    return canvas.astype(np.uint8)


def joined(a, b, offset, maxdim: int = 720):
    """A and B pushed together, B's top-left at `offset`, one BGR image."""
    dx, dy = offset[:2]
    img = compose({"a": a, "b": b}, {"a": (0, 0), "b": (dx, dy)})
    scale = maxdim / max(img.shape[:2])
    if scale < 1:
        img = cv2.resize(img, None, fx=scale, fy=scale,
                         interpolation=cv2.INTER_AREA)
    return img


def layout(pieces: dict, links, offsets: dict, gap: int = 40) -> dict:
    """Places for every piece: linked pieces at the offsets Drex chose,
    the rest in a row underneath. links: (a, b, direction) in the order
    they were accepted; offsets[(a, b, direction)] = (dx, dy)."""
    places = {}
    for a, b, d in links:
        dx, dy = offsets[(a, b, d)][:2]
        if a in places and b not in places:
            places[b] = (places[a][0] + dx, places[a][1] + dy)
        elif b in places and a not in places:
            places[a] = (places[b][0] - dx, places[b][1] - dy)
        elif a not in places and b not in places:
            x = (max(places[p][0] + pieces[p].shape[1] for p in places) + gap
                 if places else 0)
            places[a], places[b] = (x, 0), (x + dx, dy)
    loose = [p for p in pieces if p not in places]
    if loose:
        y = max(places[p][1] + pieces[p].shape[0] for p in places) + 2 * gap \
            if places else 0
        x = min(v[0] for v in places.values()) if places else 0
        for p in loose:
            places[p] = (x, y)
            x += pieces[p].shape[1] + gap
    return places
