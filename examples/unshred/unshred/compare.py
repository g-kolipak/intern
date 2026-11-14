"""Drex compares candidates side by side and picks the true neighbour.

A yes/no question about one joined pair is soft: on small pieces Drex
says yes to many wrong pairs. A pick-one question is sharp. For each
piece and side, Drex sees the piece joined with each of its top
candidates (one image each, same request) and picks the true
continuation, or none. It is asked both ways: which piece continues A,
and which piece leads into B. A pair's score is the mean of the ways it
was asked; arrange.py compares it with chance (1 in top + 1).
"""

from concurrent.futures import ThreadPoolExecutor

from . import fit
from .drex import VISION_MODEL, ask as default_ask
from .tear import data_url

CONTEXT = ("Torn fragments of one printed page. Each image shows two "
           "fragments pushed together at their torn edges, on a dark "
           "background.")
SIDE = {"right": ("to the right of", "on the left"),
        "below": ("directly below", "on top")}
OUT = ("Images 1 to {n} each show the same fragment {where} with a "
       "different candidate fragment placed {rel} it. In which image is the "
       "candidate the true continuation on the original page: text lines, "
       "letters and printed shapes continue correctly across the join? "
       "Answer none if no image is a true match.")


def _ask_one(job, up, best, ask_fn, model):
    piece, d, way, cands = job
    images, labels = [], {}
    for n, other in enumerate(cands):
        a, b = (piece, other) if way == "out" else (other, piece)
        v = best[f"{a}|{b}|{d}"]
        images.append(data_url(fit.joined(up[a], up[b], (v["dx"], v["dy"]),
                                          560)))
        labels[f"image_{n + 1}"] = other
    rel, where = SIDE[d]
    if way == "in":  # the fixed piece is the second one in each image
        where = {"on the left": "on the right", "on top": "at the bottom"}[
            where]
        rel = {"to the right of": "to the left of",
               "directly below": "directly above"}[rel]
    criteria = {lab: f"Image {lab[6:]} is the true match." for lab in labels}
    criteria["none"] = "No image is a true match."
    res = ask_fn(CONTEXT, {"pick": {
        "type": "choice", "criteria": criteria,
        "instructions": OUT.format(n=len(cands), where=where, rel=rel)}},
        model=model, images=images)
    probs = res.body["answers"]["pick"]["probabilities"]
    return [((piece, labels[lab], d, way), float(probs.get(lab, 0.0)))
            for lab in labels], res


def compare(up: dict, best: dict, top: int = 7, ask_fn=default_ask,
            model: str = VISION_MODEL, workers: int = 4):
    """{a: {b: p}} tables for right and below, from pick-one questions."""
    jobs = []
    for d in ("right", "below"):
        for p in up:
            out = sorted((k.split("|")[1] for k in best
                          if k.startswith(p + "|") and k.endswith("|" + d)),
                         key=lambda b: -best[f"{p}|{b}|{d}"]["p"])[:top]
            inn = sorted((k.split("|")[0] for k in best
                          if k.split("|")[1] == p and k.endswith("|" + d)),
                         key=lambda a: -best[f"{a}|{p}|{d}"]["p"])[:top]
            if out:
                jobs.append((p, d, "out", out))
            if inn:
                jobs.append((p, d, "in", inn))
    votes, calls = {}, []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for got, res in pool.map(
                lambda j: _ask_one(j, up, best, ask_fn, model), jobs):
            calls.append(res)
            for (piece, other, d, way), p in got:
                a, b = (piece, other) if way == "out" else (other, piece)
                votes.setdefault((a, b, d), {})[way] = p
    right = {p: {} for p in up}
    below = {p: {} for p in up}
    for (a, b, d), v in votes.items():
        score = sum(v.values()) / len(v)  # only the ways it was asked
        (right if d == "right" else below)[a][b] = score
    return right, below, calls
