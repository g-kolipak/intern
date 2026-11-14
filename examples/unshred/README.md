# Unshred: Drex rebuilds torn and shredded pages

Drexathon entry (Apex Predator track). Take a photo of torn or shredded paper
pieces; Unshred finds every piece and puts the page back together. Code only
cuts the photo into pieces and lays out the options; **Drex Max
(`drex-max-v1.1`) makes every decision**: what is paper, which way is up,
which pieces join, which page layout is right, and what to fix at the end.

**Live demo: https://unshred-production.up.railway.app.** Upload a photo and
watch each step with its picture, timing and Drex call count. Past runs are
kept in a history.

## How Drex rebuilds a page

| step | who decides | what happens |
|---|---|---|
| find | OpenCV | outline each piece (textured or plain background, shadows recovered) |
| merge | Drex | are two touching regions one piece? |
| check | Drex | paper, background, object or shadow, and which way is up |
| rate pairs | Drex | yes/no on two pieces joined side by side, at 3 alignments; a piece Drex is unsure is the right way up is tried in all 4 turns |
| compare | Drex | pick the best neighbour of each piece from its top 7, asked both ways |
| page | Drex | a beam search turns Drex's scores into whole-page layouts; Drex picks one |
| piece agents | Drex | each piece checks its spot on a close-up, and clear misfits move |
| final review | Drex | Drex rates the numbered page and flags the worst piece; moves that make the page worse are undone |
| fill holes | Drex | loose pieces are tried in the gaps inside the page, and Drex picks the spot |

Calls run in parallel (16 workers). A photo takes about 25 to 60 s; the 9-scrap
flyer takes about 125 Drex calls.

## Results

- **Menu photo, 7 torn pieces on black cloth:** all 7 pieces end up in the page. The last one fills a gap on the right edge, found by the fill-holes step.
- **Flyer photo, 9 scraps:** 8 of 9 pieces in place in the best runs.
- **ShredBench pilot** (arXiv 2604.23813), 3 news documents of 8 pieces, scored on text NED (lower is better): the scrambled scene scores 0.91 and the Drex rebuild 0.60, against GPT-5.1 at 0.77 in the paper.
- **Simulated tears** (with an answer key): 5 of 6 links right on 6 pieces, and 18 of 28 on 24 pieces.

Details: [docs/RESULTS.md](docs/RESULTS.md).

**Limits.**
- **Cluttered or sunlit backgrounds:** the piece finder can merge or split pieces.
- **Near-blank pieces:** these give Drex little to go on.
- **Large pages:** more pieces means more mistakes.
- **Simulated tears:** the tears in `scripts/eval_torn.py` are simulated; the photos are real.

## Run it

```
uv sync
cp .env.example .env          # then put your NACE_API_KEY in .env
uv run --env-file .env uvicorn unshred.webapp:app --port 8765    # website
uv run --env-file .env python scripts/rebuild_photo.py photo.jpg # one photo
uv run pytest -q              # tests, no network (Drex is mocked)
```

`unshred/` holds the pipeline (`photo.py`/`plain.py` find, `screen.py` check,
`stages.py` rate, `compare.py`, `arrange.py`, `build.py`, `agents.py`,
`review.py`, `holes.py`, `drex.py`). `scripts/` holds the command-line tools
and benchmarks, and `web/` holds the page.
