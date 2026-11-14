"""Tests for unshred.tear: Drex Max is replaced by a fake, no network."""
import base64

import cv2
import numpy as np
import pytest

from unshred import tear
from unshred.drex import CallResult, VISION_MODEL


def piece(value=200, h=60, w=90):
    img = np.zeros((h, w, 4), np.uint8)
    img[..., :3], img[..., 3] = value, 255
    img[:10, :20, :3] = 0  # a dark corner, so turns can be told apart
    return img


class FakeMax:
    """Answers orient and seam requests; records what it was sent."""

    def __init__(self, upright="image_2", seams=None):
        self.upright, self.seams, self.requests = upright, seams or {}, []

    def __call__(self, state, questions, **kw):
        self.requests.append({"state": state, "questions": questions, **kw})
        answers = {}
        for qid, q in questions.items():
            if q["type"] == "choice":
                answers[qid] = {"type": "choice", "choice": self.upright,
                                "probabilities": {
                                    lab: (0.7 if lab == self.upright else 0.1)
                                    for lab in q["criteria"]}}
            else:
                answers[qid] = {"type": "noul",
                                "noul": self.seams.get(qid, 0.05)}
        return CallResult(body={"answers": answers}, latency_ms=20.0,
                          input_tokens=1000, request_id="r", model=kw["model"])


def test_data_url_is_a_small_jpeg():
    url = tear.data_url(piece(h=1000, w=2000), maxdim=400)
    head, b64 = url.split(",", 1)
    assert head == "data:image/jpeg;base64"
    img = cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), 1)
    assert img.shape == (200, 400, 3)


def test_quarter_turn_is_clockwise():
    p = piece()
    assert tear.quarter_turn(p, 0).shape == (60, 90, 4)
    turned = tear.quarter_turn(p, 90)
    assert turned.shape == (90, 60, 4)
    assert turned[0, -1, 0] == 0   # the dark top-left corner is now top-right
    assert np.array_equal(tear.quarter_turn(turned, 270), p)


def test_orient_request_shows_four_turns():
    state, images, questions = tear.orient_request(piece())
    assert len(images) == 4 and len(set(images)) == 4
    assert set(questions["upright"]["criteria"]) == {
        "image_1", "image_2", "image_3", "image_4"}
    assert questions["upright"]["type"] == "choice" and "four" not in state


def test_orient_all_turns_each_piece_and_reports_confidence():
    fake = FakeMax(upright="image_2")
    calls_seen = []
    upright, turns, calls = tear.orient_all(
        {"p1": piece(), "p2": piece(120)}, ask_fn=fake,
        on_call=lambda *a: calls_seen.append(a[:2]))
    assert turns == {"p1": {"degrees": 90, "confidence": 0.7},
                     "p2": {"degrees": 90, "confidence": 0.7}}
    assert upright["p1"].shape == (90, 60, 4)
    assert len(calls) == 2 and sorted(calls_seen) == [("orient", "p1"),
                                                      ("orient", "p2")]
    assert all(r["model"] == VISION_MODEL and len(r["images"]) == 4
               for r in fake.requests)


def test_match_pairs_fills_both_directions_from_one_request_per_pair():
    fake = FakeMax(seams={"b_right_of_a": 0.9, "a_below_b": 0.8})
    pieces = {"p1": piece(), "p2": piece(120), "p3": piece(60)}
    res = tear.match_pairs(pieces, ask_fn=fake, workers=1)
    assert len(fake.requests) == len(res.calls) == 3   # 3 pairs
    assert res.right["p1"] == {"p2": 0.9, "p3": 0.9}
    assert res.right["p2"] == {"p1": 0.05, "p3": 0.9}
    assert res.below["p2"]["p1"] == 0.8 and res.below["p1"]["p2"] == 0.05
    req = fake.requests[0]
    assert len(req["images"]) == 2 and set(req["questions"]) == set(tear.PAIR)
    text = req["questions"]["a_right_of_b"]["instructions"]
    assert "Image 1 is placed directly to the right of Image 2" in text


def test_a_missing_answer_is_an_error():
    def broken(state, questions, **kw):
        return CallResult(body={"answers": {}}, latency_ms=1.0,
                          input_tokens=1, request_id="r", model="m")
    with pytest.raises(ValueError, match="p1"):
        tear.orient_all({"p1": piece()}, ask_fn=broken)
    with pytest.raises(ValueError, match="seam"):
        tear.match_pairs({"p1": piece(), "p2": piece()}, ask_fn=broken)
