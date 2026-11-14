"""Command-line accuracy test, no UI. Keys come from the environment.

  uv run scripts/experiment.py --rows 2 --cols 5
  uv run scripts/experiment.py --rows 2 --cols 5 --strategy choice -v
  uv run scripts/experiment.py --rows 5 --cols 6 --baseline

Every run also saves its raw scores to runs/exp_*.json, so thresholds
can be tuned afterwards without paying for the calls again.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from unshred import scoring  # noqa: E402
from unshred.assembler import CONFIDENCE_THRESHOLD  # noqa: E402
from unshred.pipeline import run  # noqa: E402
from unshred.shredder import shred  # noqa: E402


def _pct(pair) -> str:
    good, total = pair
    return f"{good}/{total}" + (f" = {good / total:.0%}" if total else "")


def _verbose(match, key, lines) -> None:
    """Per piece: the true right neighbor's score and rank, and the top 3."""
    id_at = {pos: pid for pid, pos in key.items()}
    for a in sorted(match.right, key=lambda p: key[p]):
        r, c = key[a]
        true = id_at.get((r, c + 1))
        row = sorted(match.right[a].items(), key=lambda kv: -kv[1])
        rank = next((i + 1 for i, (b, _) in enumerate(row) if b == true), 0)
        top = "  ".join(f"{s:.2f}{'*' if b == true else ''}"
                        for b, s in row[:3])
        want = f"true={match.right[a][true]:.2f} rank {rank}" if true \
            else "right edge"
        print(f"  ({r},{c}) {lines[a][0]!r:16} edge={match.right_edge[a]:.2f}"
              f"  {want:22} top3: {top}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--file", default="data/sample.txt")
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None)
    ap.add_argument("--strategy", choices=["noul", "choice"], default="noul")
    ap.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--baseline", action="store_true",
                    help="ask BASELINE_MODEL instead of Drex")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.baseline:
        from unshred import baseline as backend
        price = scoring.price_per_mtok("BASELINE_USD_PER_MTOK")
    else:
        from unshred import drex as backend
        price = scoring.price_per_mtok("DREX_USD_PER_MTOK")
    model = args.model or backend.DEFAULT_MODEL

    text = Path(args.file).read_text()
    shr = shred(text, rows=args.rows, cols=args.cols, seed=args.seed)
    key, lines = shr.answer_key, {p.piece_id: p.lines for p in shr.pieces}
    print(f"Shredded {len(text)} chars into {shr.rows}x{shr.cols} = "
          f"{len(shr.pieces)} pieces. Model {model}, strategy "
          f"{args.strategy}, threshold {args.threshold}")

    res = run(shr.pieces, shr.rows, shr.cols, ask_fn=backend.ask,
              model=model, strategy=args.strategy, threshold=args.threshold,
              workers=args.workers)
    match, asm = res.match, res.assembly
    stats = scoring.summary(asm, key, match.calls, res.elapsed_s, lines,
                            price)
    if args.verbose:
        _verbose(match, key, lines)
    print("Raw picks (top score is the true neighbor):")
    print(f"  right  {_pct(scoring.pick_stats(match.right, key, 'right', lines))}")
    print(f"  below  {_pct(scoring.pick_stats(match.below, key, 'below', lines))}"
          f"   (rows rebuilt to full width: "
          f"{len(match.bottom_edge) // shr.cols} of {shr.rows})")
    print(f"  edge   {_pct(scoring.edge_stats(match.right_edge, key, shr.cols))}")
    print(f"Neighbor accuracy: {stats['neighbor_accuracy']:.1%}  "
          f"({stats['links_correct']}/{stats['links']} accepted links correct)")
    print(f"Position accuracy: {stats['position_accuracy']:.1%}  "
          f"({stats['placed']} of {stats['total']} pieces placed, "
          f"{stats['needs_human']} need a human)")
    cost = "n/a (set the price env var)" if stats["cost_usd"] is None \
        else f"${stats['cost_usd']:.4f}"
    print(f"Calls: {stats['calls']}, average latency "
          f"{stats['avg_latency_ms']:.0f} ms, wall time "
          f"{stats['elapsed_s']:.1f} s, {stats['input_tokens']} input "
          f"tokens, cost {cost}")

    out = ROOT / "runs"
    out.mkdir(exist_ok=True)
    name = (f"exp_{'baseline' if args.baseline else 'drex'}_{args.strategy}"
            f"_{shr.rows}x{shr.cols}_seed{args.seed}.json")
    (out / name).write_text(json.dumps({
        "args": vars(args), "model": model, "stats": stats,
        "pieces": {k: list(v) for k, v in lines.items()},
        "answer_key": {k: list(v) for k, v in key.items()},
        "right": match.right, "below": match.below,
        "right_edge": match.right_edge, "bottom_edge": match.bottom_edge,
        "grid": {k: list(v) for k, v in asm.grid.items()},
        "needs_human": asm.needs_human,
    }, indent=1))
    print(f"Raw scores saved to runs/{name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
