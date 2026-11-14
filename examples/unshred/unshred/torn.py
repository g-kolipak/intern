"""Simulated torn-paper photos with a known answer, for testing at any size.

  page()        a printed page: header band, boxes, lines of text
  tear()        cut it into rows x cols pieces along wavy, jagged tears
  scatter()     turn every piece to a random angle and drop it, apart from
                the others, on a dark or a pale speckled table with uneven
                light and soft shadows

The answer (which piece came from which cell) is returned separately and
is only ever used for scoring.
"""

import cv2
import numpy as np


def page(lines: list[str], width: int = 1700, height: int = 2200, seed=0):
    rng = np.random.default_rng(seed)
    img = np.full((height, width, 3), 250, np.uint8)
    cv2.rectangle(img, (0, 0), (width, 260), (25, 25, 25), -1)
    cv2.putText(img, "TORN PAGE TEST", (60, 170), cv2.FONT_HERSHEY_DUPLEX,
                3.2, (245, 245, 245), 6, cv2.LINE_AA)
    y, box_at = 360, int(rng.integers(6, 12))
    for i, text in enumerate(lines):
        if y > height - 80:
            break
        if i == box_at:
            cv2.rectangle(img, (60, y - 50), (width - 60, y + 170),
                          (30, 30, 30), 4)
        cv2.putText(img, text[:58], (80, y), cv2.FONT_HERSHEY_SIMPLEX, 1.55,
                    (20, 20, 20), 3, cv2.LINE_AA)
        y += 62
    return img


def _cut(n: int, length: int, span: int, rng, wobble: float):
    """n+1 tear lines across `length` pixels, spread over `span`."""
    out = []
    for k in range(n + 1):
        if k in (0, n):
            out.append(np.full(length, 0 if k == 0 else span, np.int32))
            continue
        walk = np.cumsum(rng.normal(0, 1, length))
        walk = cv2.GaussianBlur(walk.reshape(1, -1), (0, 0), 25).ravel()
        walk = walk / (np.abs(walk).max() + 1e-6) * wobble * span / n
        jag = rng.normal(0, 1.5, length)
        out.append((k * span / n + walk + jag).astype(np.int32))
    return out


def tear(img, rows: int, cols: int, seed=0, wobble: float = 0.12):
    """BGRA pieces and their true (row, col), torn along wavy lines."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    xs = _cut(cols, h, w, rng, wobble)   # x of each vertical tear, per row
    ys = _cut(rows, w, h, rng, wobble)   # y of each horizontal tear, per col
    yy, xx = np.mgrid[0:h, 0:w]
    pieces, truth = [], []
    for r in range(rows):
        for c in range(cols):
            m = ((xx >= xs[c][:, None]) & (xx < xs[c + 1][:, None])
                 & (yy >= ys[r][None, :]) & (yy < ys[r + 1][None, :]))
            ys_, xs_ = np.where(m)
            y0, y1, x0, x1 = ys_.min(), ys_.max() + 1, xs_.min(), xs_.max() + 1
            piece = np.dstack([img[y0:y1, x0:x1],
                               (m[y0:y1, x0:x1] * 255).astype(np.uint8)])
            pieces.append(piece)
            truth.append((r, c))
    return pieces, truth


def _rotate(piece, degrees: float):
    h, w = piece.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos) + 2, int(h * cos + w * sin) + 2
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(piece, m, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderValue=(0, 0, 0, 0))


def _table(h: int, w: int, kind: str, rng):
    if kind == "dark":
        base = np.full((h, w, 3), (38, 36, 34), np.float32)
        noise = rng.normal(0, 3, (h, w, 1))
    else:  # pale, warm, speckled, like a cheap table top
        base = np.full((h, w, 3), (178, 192, 200), np.float32)
        noise = rng.normal(0, 6, (h, w, 1))
        dots = rng.random((h, w)) < 0.004
        base[dots] -= 60
    yy, xx = np.mgrid[0:h, 0:w]
    light = 0.82 + 0.3 * (xx / w) * (1 - 0.5 * yy / h)  # uneven light
    return np.clip(base * light[..., None] + noise, 0, 255)


def scatter(pieces, kind: str = "pale", seed=0, room: float = 3.2,
            gap: int = 30):
    """One photo with every piece turned and placed apart, plus where
    each piece went: [{"center": (x, y), "angle": degrees}, ...]."""
    rng = np.random.default_rng(seed)
    turned = [_rotate(p, float(rng.uniform(0, 360))) for p in pieces]
    area = sum(t.shape[0] * t.shape[1] for t in turned) * room
    w = int(np.sqrt(area * 4 / 3))
    h = int(area / w)
    photo = _table(h, w, kind, rng)
    taken = np.zeros((h, w), np.uint8)
    where = []
    for t in turned:
        th, tw = t.shape[:2]
        alpha = t[..., 3] > 0
        for _ in range(4000):
            y = int(rng.integers(gap, max(gap + 1, h - th - gap)))
            x = int(rng.integers(gap, max(gap + 1, w - tw - gap)))
            if not taken[y - gap:y + th + gap, x - gap:x + tw + gap].any():
                break
        else:
            raise RuntimeError("no room left: raise `room`")
        if kind != "dark":  # a soft shadow down and to the right
            sh = cv2.GaussianBlur(alpha.astype(np.float32), (0, 0), 6)
            box = photo[y + 6:y + 6 + th, x + 6:x + 6 + tw]
            box *= 1 - 0.35 * sh[:box.shape[0], :box.shape[1], None]
        a = (t[..., 3:4] / 255.0)
        box = photo[y:y + th, x:x + tw]
        box[:] = t[..., :3] * a + box * (1 - a)
        taken[y:y + th, x:x + tw] |= alpha.astype(np.uint8)
        where.append({"center": (x + tw / 2, y + th / 2)})
    return np.clip(photo, 0, 255).astype(np.uint8), where
