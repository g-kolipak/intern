"""Helpers for scripts/rebuild_photo.py: Drex pair ratings, score tables,
candidate layouts and stage timing."""

import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

from . import arrange, fit, screen
from .drex import VISION_MODEL, ask as default_ask
from .tear import data_url

CONTEXT = ("Torn fragments of one printed page. Each image shows two "
           "fragments pushed together at their torn edges, on a dark "
           "background.")
ASK = {"right": "Image {i}: the second fragment sits directly to the right "
                "of the first. Does every line of text and every printed "
                "shape continue correctly across the join?",
       "below": "Image {i}: the second fragment sits directly below the "
                "first, in the same column. Do the margins, boxes and lines "
                "of print continue straight down across the join?"}


def rate_pairs(up: dict, workers: int, ask_fn=default_ask):
    """Drex rates every shortlisted pair and alignment, in parallel."""
    jobs = screen.shortlist(up, k=3)

    def one(start):
        batch = jobs[start:start + 8]  # Drex Max reads up to 8 images
        res = ask_fn(CONTEXT, {f"join_{n + 1}": {
            "type": "noul", "instructions": ASK[d].format(i=n + 1)}
            for n, (_, _, d, _) in enumerate(batch)}, model=VISION_MODEL,
            images=[data_url(fit.joined(up[a], up[b], f, 640))
                    for a, b, _, f in batch])
        return [(job, float(res.body["answers"][f"join_{n + 1}"]["noul"]))
                for n, job in enumerate(batch)], res

    best, calls = {}, []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pairs, res in pool.map(one, range(0, len(jobs), 8)):
            calls.append(res)
            for (a, b, d, f), p in pairs:
                key = f"{a}|{b}|{d}"
                if key not in best or p > best[key]["p"]:
                    best[key] = {"p": round(p, 4), "dx": f[0], "dy": f[1]}
    return best, calls


def variants(pieces: dict, turns: dict, paper: dict, sure: float = 0.9):
    """Pieces Drex kept, turned upright. A piece whose upright turn Drex is
    unsure of (often a blank one) enters as all 4 turns, 'p3@90' etc., so
    the comparisons decide its turn; the layout uses one turn per piece."""
    from .tear import TURNS, quarter_turn
    up = {}
    for p, img in pieces.items():
        if paper[p] < 0.5:
            continue
        if turns[p]["confidence"] >= sure:
            up[p] = quarter_turn(img, turns[p]["degrees"])
        else:
            for t in TURNS:
                up[f"{p}@{t}"] = quarter_turn(img, t)
    return up


def merge(pick: dict, d: str, best: dict, up: dict) -> dict:
    """One confidence per pair from both Drex answers: the pick-one share
    (against chance, 1 in 8) and the yes/no rating, averaged as evidence.
    A pair Drex was never asked about counts as mildly against."""
    out = {a: {} for a in up}
    for a in up:
        for b in up:
            if a == b:
                continue
            ev = []
            if pick.get(a, {}).get(b) is not None:
                ev.append(arrange.evidence(pick[a][b], 1 / 8))
            if f"{a}|{b}|{d}" in best:
                ev.append(arrange.evidence(best[f"{a}|{b}|{d}"]["p"]))
            e = sum(ev) / len(ev) if ev else -1.0
            out[a][b] = 1 / (1 + math.exp(-e))
    return out


def candidates(up, best, right, below, rated: bool, per: int = 4):
    """Best layouts two ways, for Drex to choose between: from the pick-one
    answers alone, and from both answers merged. Returns a list of
    (cells, tables, base) without duplicates."""
    ways = [((right, below), 1 / 8)]
    if rated:
        ways.append(((merge(right, "right", best, up),
                      merge(below, "below", best, up)), None))
    out, seen = [], set()
    for k in range(per):
        for (r, b), base in ways:
            lays = arrange.arrange(r, b, base=base, top=per)
            if k < len(lays):
                key = tuple(sorted(lays[k][0].items()))
                if key not in seen:
                    seen.add(key)
                    out.append((lays[k][0], (r, b), base))
    return out


class Clock:
    """Times each stage, prints it, and (with `out`) keeps a step-by-step
    record in out/steps.json with one picture per step in out/steps/."""

    def __init__(self, out=None):
        self.t0 = self.t = time.monotonic()
        self.out, self.steps = out, []

    def __call__(self, label: str, calls=None, image=None):
        now = time.monotonic()
        extra = f"  ({len(calls)} Drex calls)" if calls is not None else ""
        print(f"  {now - self.t:6.2f} s  {label}{extra}", flush=True)
        step = {"step": len(self.steps) + 1, "label": label,
                "seconds": round(now - self.t, 2),
                "drex_calls": len(calls) if calls is not None else 0,
                "image": None}
        if self.out is not None:
            if image is not None:
                import cv2
                (self.out / "steps").mkdir(parents=True, exist_ok=True)
                s = min(1.0, 1400 / max(image.shape[:2]))
                small = cv2.resize(image, None, fx=s, fy=s,
                                   interpolation=cv2.INTER_AREA)
                step["image"] = f"step{step['step']}.jpg"
                cv2.imwrite(str(self.out / "steps" / step["image"]), small)
            self.steps.append(step)
            (self.out / "steps.json").write_text(json.dumps(
                {"steps": self.steps,
                 "total_seconds": round(now - self.t0, 2)}, indent=1))
        self.t = now
