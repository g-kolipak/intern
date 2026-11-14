"""Real paper: Drex Max turns each cut-out upright and judges every seam.

Piece agent, one request per piece: the piece is shown in its four
quarter turns and Drex picks the upright one.
Parent agent, one request per pair of pieces: four yes/no questions ask
whether B continues A on the right, A continues B on the right, B
continues A below, A continues B below.

Images go out as data URLs and questions call them Image 1..N. Only the
cut-outs are sent: no table, no neighbours, no positions in the photo.
"""

import base64
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations

import cv2
import numpy as np

from .cutouts import on_canvas
from .drex import VISION_MODEL, ask as default_ask
from .matcher import MatchResult

TURNS = (0, 90, 180, 270)
ONE = ("One torn fragment of a printed page, cut out and shown on a plain "
       "dark background. Images 1 to 4 show the same fragment, each one "
       "turned a further quarter turn clockwise.")
TWO = ("Two torn fragments of the same printed page, both upright, each "
       "cut out and shown on a plain dark background.")
SEAM = ("Image {b} is placed directly {side} Image {a}, both as shown. Do "
        "the text lines, letter shapes and design elements continue "
        "across the torn edge between them, as on the original page? "
        "Answer yes only if Image {b} is the true continuation of Image "
        "{a} on that side.")
PAIR = {  # question id -> (first image, second image, side)
    "b_right_of_a": (1, 2, "to the right of"),
    "a_right_of_b": (2, 1, "to the right of"),
    "b_below_a": (1, 2, "below"),
    "a_below_b": (2, 1, "below"),
}


def data_url(piece, maxdim: int = 720) -> str:
    """A BGR or BGRA image as a JPEG data URL, at most `maxdim` wide/high."""
    flat = on_canvas(piece) if piece.shape[2] == 4 else piece
    scale = maxdim / max(flat.shape[:2])
    if scale < 1:
        flat = cv2.resize(flat, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", flat, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError("could not encode the piece as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


def quarter_turn(piece, degrees: int):
    """Turn a piece clockwise by 0, 90, 180 or 270 degrees, losslessly."""
    return np.ascontiguousarray(np.rot90(piece, k=(-degrees // 90) % 4))


def orient_request(piece) -> tuple[str, list[str], dict]:
    images = [data_url(quarter_turn(piece, t)) for t in TURNS]
    questions = {"upright": {
        "type": "choice",
        "instructions": ("In which image is the printed text upright, so "
                         "that it reads normally from left to right?"),
        "criteria": {f"image_{i + 1}": f"The text is upright in Image {i + 1}."
                     for i in range(len(TURNS))},
    }}
    return ONE, images, questions


def seam_request(a, b) -> tuple[str, list[str], dict]:
    questions = {
        qid: {"type": "noul",
              "instructions": SEAM.format(a=first, b=second, side=side)}
        for qid, (first, second, side) in PAIR.items()}
    return TWO, [data_url(a), data_url(b)], questions


def orient_all(pieces: dict, ask_fn=default_ask, model: str = VISION_MODEL,
               on_call=None, workers: int = 4, **kw):
    """Piece agents. Returns (upright pieces, turns, calls).

    turns maps a piece id to {"degrees": clockwise turn applied,
    "confidence": Drex's probability for that turn}.
    """
    def one(item):
        pid, piece = item
        state, images, questions = orient_request(piece)
        res = ask_fn(state, questions, model=model, images=images, **kw)
        try:
            probs = res.body["answers"]["upright"]["probabilities"]
            best = max(range(len(TURNS)),
                       key=lambda i: probs[f"image_{i + 1}"])
            conf = float(probs[f"image_{best + 1}"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"no upright answer for piece {pid!r}") from e
        if on_call:
            on_call("orient", pid, res)
        return pid, TURNS[best], conf, res

    upright, turns, calls = {}, {}, []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for pid, degrees, conf, res in pool.map(one, pieces.items()):
            upright[pid] = quarter_turn(pieces[pid], degrees)
            turns[pid] = {"degrees": degrees, "confidence": round(conf, 4)}
            calls.append(res)
    return upright, turns, calls


def match_pairs(pieces: dict, ask_fn=default_ask, model: str = VISION_MODEL,
                on_call=None, workers: int = 4, **kw) -> MatchResult:
    """Parent agent: score every pair of upright pieces on all four sides.

    right[A][B] is Drex's probability that B continues A on the right,
    below[A][B] that B continues A below.
    """
    def one(pair):
        a, b = pair
        state, images, questions = seam_request(pieces[a], pieces[b])
        res = ask_fn(state, questions, model=model, images=images, **kw)
        try:
            got = {q: float(res.body["answers"][q]["noul"]) for q in PAIR}
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"no seam answer for {a!r}/{b!r}") from e
        if on_call:
            on_call("seam", a, res)
        return a, b, got, res

    result = MatchResult()
    for pid in pieces:
        result.right[pid], result.below[pid] = {}, {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for a, b, got, res in pool.map(one, combinations(pieces, 2)):
            result.right[a][b] = got["b_right_of_a"]
            result.right[b][a] = got["a_right_of_b"]
            result.below[a][b] = got["b_below_a"]
            result.below[b][a] = got["a_below_b"]
            result.calls.append(res)
    return result
