"""Turn piece masks into clean, straightened cut-outs and preview images."""

import cv2
import numpy as np


def cutout(image, mask, pad: int = 12):
    """BGRA crop of one piece: transparent everywhere outside its border."""
    ys, xs = np.where(mask > 0)
    y0, y1 = max(ys.min() - pad, 0), min(ys.max() + pad + 1, mask.shape[0])
    x0, x1 = max(xs.min() - pad, 0), min(xs.max() + pad + 1, mask.shape[1])
    alpha = cv2.GaussianBlur((mask[y0:y1, x0:x1] > 0).astype(np.float32),
                             (0, 0), 1.2)
    alpha = np.clip((alpha - 0.35) / 0.3, 0, 1)  # soft, slightly tight edge
    return np.dstack([image[y0:y1, x0:x1],
                      (alpha * 255).astype(np.uint8)])


def skew_angle(piece) -> float:
    """Degrees (-45..45] by which the print on a BGRA piece is tilted.

    Printed text and boxes have edges in two directions 90 degrees apart,
    so edge directions are averaged modulo 90 degrees.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(piece[..., :3], cv2.COLOR_BGR2GRAY)
                            .astype(np.float32), (0, 0), 2)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    side = max(15, (min(piece.shape[:2]) // 12) | 1)
    inner = cv2.erode((piece[..., 3] > 128).astype(np.uint8),
                      np.ones((side, side), np.uint8)) > 0
    mag = np.hypot(gx, gy)
    if not inner.any() or mag[inner].max() <= 0:
        return 0.0
    sel = inner & (mag >= np.percentile(mag[inner], 85))
    ang = np.arctan2(gy[sel], gx[sel])
    c = float((mag[sel] * np.cos(4 * ang)).sum())
    s = float((mag[sel] * np.sin(4 * ang)).sum())
    return float(np.degrees(np.arctan2(s, c)) / 4)


def rotate(piece, degrees: float):
    """Rotate a BGRA piece by `degrees`: positive turns it clockwise on
    screen. The canvas grows so nothing is cut off."""
    h, w = piece.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), -degrees, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    new_w, new_h = int(h * sin + w * cos + 2), int(h * cos + w * sin + 2)
    m[0, 2] += new_w / 2 - w / 2
    m[1, 2] += new_h / 2 - h / 2
    out = cv2.warpAffine(piece, m, (new_w, new_h), flags=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    ys, xs = np.where(out[..., 3] > 8)
    return out[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def straighten(piece):
    """Undo the tilt, so the print runs along the image axes. Which of the
    four upright turns is right is a reading question, left to Drex."""
    return rotate(piece, -skew_angle(piece))


def on_canvas(piece, color=(30, 28, 28)):
    """Flatten a BGRA piece onto a plain BGR background."""
    alpha = piece[..., 3:4].astype(np.float32) / 255
    back = np.full(piece.shape[:2] + (3,), color, np.float32)
    return (piece[..., :3] * alpha + back * (1 - alpha)).astype(np.uint8)


def outline(image, masks, width: int | None = None):
    """The photo with the table dimmed and every piece bordered + numbered."""
    width = width or max(3, image.shape[1] // 500)
    covered = np.zeros(image.shape[:2], bool)
    for mask in masks:
        covered |= mask > 0
    out = image.copy()
    out[~covered] = (image[~covered] * 0.35).astype(np.uint8)
    for i, mask in enumerate(masks):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(out, contours, -1, (80, 255, 80), width)
        x, y, w, h = cv2.boundingRect(contours[0])
        cv2.putText(out, str(i + 1), (x - width * 6, y + width * 10),
                    cv2.FONT_HERSHEY_SIMPLEX, width / 2.5, (60, 60, 255),
                    width)
    return out


def sheet(pieces, cols: int = 3, cell: int = 520, color=(30, 28, 28)):
    """A grid of flattened pieces, each scaled to fit a square cell."""
    rows = -(-len(pieces) // cols)
    out = np.full((rows * cell, cols * cell, 3), color, np.uint8)
    for i, piece in enumerate(pieces):
        scale = (cell - 40) / max(piece.shape[:2])
        small = cv2.resize(piece, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_AREA)
        flat = on_canvas(small, color)
        y = (i // cols) * cell + (cell - flat.shape[0]) // 2
        x = (i % cols) * cell + (cell - flat.shape[1]) // 2
        out[y:y + flat.shape[0], x:x + flat.shape[1]] = flat
        cv2.putText(out, str(i + 1), ((i % cols) * cell + 12,
                    (i // cols) * cell + 36), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, (60, 60, 255), 2)
    return out


def layout(sizes: dict, groups: list, loose=(), gap: int = 24):
    """Top-left corner (x, y) of every piece, and the canvas (width, height).

    sizes maps id -> (height, width). groups is a list of {id: (row, col)}
    and they are drawn side by side. Loose pieces form one row underneath.
    Pieces keep their real size and sit in the middle of their grid cell.
    """
    pos, x0, bottom = {}, gap, gap
    for group in groups:
        rows = sorted({r for r, _ in group.values()})
        cols = sorted({c for _, c in group.values()})
        hs = {r: max(sizes[p][0] for p, (pr, _) in group.items() if pr == r)
              for r in rows}
        ws = {c: max(sizes[p][1] for p, (_, pc) in group.items() if pc == c)
              for c in cols}
        for p, (r, c) in group.items():
            x = x0 + sum(ws[q] + gap for q in cols if q < c)
            y = gap + sum(hs[q] + gap for q in rows if q < r)
            pos[p] = (x + (ws[c] - sizes[p][1]) // 2,
                      y + (hs[r] - sizes[p][0]) // 2)
        x0 += sum(ws.values()) + gap * len(cols) + 2 * gap
        bottom = max(bottom, gap + sum(hs.values()) + gap * len(rows))
    x, y = gap, bottom + (2 * gap if groups else 0)
    for p in loose:
        pos[p] = (x, y)
        x += sizes[p][1] + gap
    height = y + max((sizes[p][0] for p in loose), default=0) + gap
    return pos, (max(x0, x), height if loose else bottom)


def compose(pieces: dict, groups: list, loose=(), gap: int = 24,
            color=(30, 28, 28)):
    """The rebuilt page as one picture. pieces maps id -> BGRA cut-out."""
    pos, (width, height) = layout(
        {p: img.shape[:2] for p, img in pieces.items()}, groups, loose, gap)
    out = np.zeros((height, width, 4), np.uint8)
    for p, (x, y) in pos.items():
        img = pieces[p]
        region = out[y:y + img.shape[0], x:x + img.shape[1]]
        region[...] = np.where(img[..., 3:4] > region[..., 3:4], img, region)
    return on_canvas(out, color)
