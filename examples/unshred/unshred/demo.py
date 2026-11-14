"""Demo data for web/tear.html: one finished real-paper run as a script.

The page is static. It loads runs/photo/NAME/demo.js, which sets
window.DEMO, so it also works when opened straight from disk and a saved
run can be replayed without any key or network.
"""

import json
from pathlib import Path

from .cutouts import layout


def groups(assembly) -> tuple[list, list]:
    """(linked groups, biggest first; ids of the loose pieces)."""
    linked = sorted((g for g in assembly.clusters if len(g) > 1),
                    key=len, reverse=True)
    loose = [p for g in assembly.clusters if len(g) == 1 for p in g]
    return linked, loose


def build(name: str, photo_size, meta: dict, turns: dict, assembly,
          usage: dict | None, threshold: float) -> dict:
    """Everything the page needs.

    meta maps id -> {"cx", "cy": centre in the photo as 0..1, "skew":
    degrees the print was tilted, "w", "h": size of the upright cut-out}.
    """
    linked, loose = groups(assembly)
    pos, canvas = layout({p: (m["h"], m["w"]) for p, m in meta.items()},
                         linked, loose)
    score = {}
    for link in assembly.accepted:
        for p in (link.a, link.b):
            score.setdefault(p, round(link.score, 4))
    pieces = []
    for p, m in meta.items():
        turn = turns[p]["degrees"]
        pieces.append({
            "id": p, "src": f"upright/{p}.png", "w": m["w"], "h": m["h"],
            "cx": round(m["cx"], 4), "cy": round(m["cy"], 4),
            # clockwise degrees from upright back to how it lay in the photo
            "start": round(m["skew"] - turn, 1),
            "turn": turn, "confidence": turns[p]["confidence"],
            "x": pos[p][0], "y": pos[p][1],
            "placed": p not in loose, "score": score.get(p),
        })
    return {
        "name": name, "photo": "outline.jpg", "threshold": threshold,
        "photo_size": [int(photo_size[0]), int(photo_size[1])],
        "canvas": [int(canvas[0]), int(canvas[1])],
        "pieces": pieces, "loose": loose,
        "links": [{"a": link.a, "b": link.b, "direction": link.direction,
                   "score": round(link.score, 4)}
                  for link in assembly.accepted],
        "usage": usage or {},
    }


def write(out_dir: Path, data: dict) -> Path:
    """Save the run as demo.js next to its images."""
    path = Path(out_dir) / "demo.js"
    path.write_text("window.DEMO = " + json.dumps(data, indent=1) + ";\n")
    return path
