# Unshred: rebuilding shredded documents with Drex

## What we are building
A demo for the Drexathon hackathon (track: Apex Predator, deadline 3:00 PM).
We take a real document, shred it in a simulator into many scrambled pieces,
and use the Drex decision model to work out which piece goes next to which,
rebuilding the page live on screen. We compare Drex with a large chat model
on the same task: accuracy, time, and cost. NDI reads the document before
shredding and extracts key facts after rebuilding.

The shredding is SIMULATED. The UI must say so.

## Stack
- Python 3.12, uv, FastAPI, httpx, pytest.
- One static page: web/index.html with plain JavaScript. No frameworks.
- Keys come from environment variables: NACE_API_KEY, DEEPINFRA_API_KEY,
  NDI_API_KEY, BASELINE_MODEL. Never write a key into any file in this repo.
- .env must be in .gitignore. The repo is public.

## Rules for you, the coding agent
- Do ONE task at a time and stop when it is done.
- Keep each file under 200 lines.
- Write a pytest test for every module. Tests must not use the network;
  mock Drex and NDI.
- Do NOT guess API field names. Before writing Drex code, read the Drex docs
  (https://console.nace.ai/docs and https://github.com/nace-ai/drex-agent-skill)
  and make one real call that prints the raw response. Do the same for NDI
  (pip install ndi-sdk; see https://pypi.org/project/ndi-sdk/).
- Commit after every working step.

## Files
- unshred/shredder.py   cut text into pieces, shuffle, keep the answer key
- unshred/drex.py       thin Drex client (one function: ask(state, questions))
- unshred/matcher.py    ask Drex which pieces are neighbors
- unshred/assembler.py  turn neighbor scores into a rebuilt grid
- unshred/baseline.py   same questions to a large chat model
- unshred/ndi.py        parse a document, extract facts (results cached)
- unshred/server.py     FastAPI: start a run, stream events, serve the page
- web/index.html        the board, the scoreboard, the slider
- scripts/experiment.py command-line accuracy test, no UI
- data/                 sample documents and cached NDI results

## How the pieces work
Shredder:
- Input: plain text. Wrap it to a fixed width of 60 characters, pad lines
  with spaces, so the page is a rectangle of characters.
- Cut the rectangle into a grid of ROWS x COLS equal pieces. Each piece is
  a small block of text lines. Cuts will land in the middle of words; that
  is intended.
- Give each piece a random ID, shuffle the list, and return the pieces plus
  a separate answer key (ID -> row, col). The answer key must NEVER be sent
  to Drex or the baseline.

Matcher (the core):
- For each piece A, find its RIGHT neighbor and its BELOW neighbor.
- Right neighbor: for each candidate B, join A and B line by line (A's line
  followed directly by B's line). Ask Drex a yes/no question: "Is this
  joined text one continuous natural passage?" Put many candidates in one
  request as separate named questions. Also ask "Is A on the right edge of
  the page (nothing follows it)?"
- Below neighbor: stack A on top of B and ask the same kind of question.
- Output: two score tables, right[A][B] and below[A][B], values 0 to 1.
- Record for every call: latency in ms and input tokens.

Assembler:
- Collect all candidate links (A, B, direction, score). Sort by score, high
  to low.
- Accept a link only if it is the best option for both pieces and it does
  not put two pieces in the same grid cell.
- Links below CONFIDENCE_THRESHOLD (start at 0.6) are not placed; put them
  in a "needs human" list with the top 3 candidates.
- Emit an event for every placement: piece ID, grid position, score.

Baseline:
- Ask the model named in BASELINE_MODEL the same questions through an
  OpenAI-compatible chat endpoint, forcing a JSON answer. Record the same
  metrics.

Scoring (uses the answer key, after the run):
- neighbor accuracy: share of accepted links that are correct
- position accuracy: share of pieces in the right cell
- total time, number of calls, estimated cost

NDI:
- Before: parse a PDF from data/ into text. Cache it as data/parsed/NAME.md.
- After: extract a few fields from the rebuilt text (title, date, total
  amount, parties) with a JSON schema. Cache the result.
- If NDI is unavailable, read data/NAME.txt so work is never blocked.

## The page (web/index.html)
- Left: the shuffled pile of pieces (monospace text blocks).
- Center: the board. Pieces slide into place as events arrive. Border color
  shows confidence (green high, amber low).
- Right: scoreboard for Drex and for the baseline: pieces placed, accuracy,
  elapsed time, calls, cost.
- Controls: document picker, grid size slider (2x5 up to 10x10), Start.
- "Needs human" panel: shows piece A and its top candidates; clicking one
  places it.
- After the rebuild: a panel with the facts NDI extracted.
- A visible label: "Simulated shredding".
- The page polls GET /runs/{id}/events every 300 ms. Finished runs are saved
  to runs/ID.json and can be replayed, which is the demo backup.

## Build order
1. scripts: one real Drex call, print the raw response.
2. shredder.py with tests.
3. matcher.py + scripts/experiment.py on a 2x5 grid. Print accuracy.
   STOP and report the number. Do not continue if it is near chance.
4. assembler.py with tests, then experiment on 5x6.
5. server.py and web/index.html with live events.
6. baseline.py and the two-column scoreboard.
7. ndi.py before and after.
8. Needs-human panel, replay of saved runs, README.