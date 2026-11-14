# Results (Nov 14, 2026)

## Real photo (image.png, 9 torn scraps on a pale table)
- Morning, first Drex rebuild (separate image per pair): 3 links, 4 scraps loose.
- Now (joined checks, pick-one comparisons, whole-page picks): 8 of 9 scraps in place in the best run;
  the near-blank scrap is still misplaced and the bottom row varies between runs.
- Time per run (careful mode): find 14.8 s, check 0.9 s, rate 11.3 s, compare 4.8 s, page 3.4 s,
  re-decide 5.2 s; total 40.5 s, 125 Drex Max calls.

## Simulated torn pages (scripts/eval_torn.py)
| pieces | links right | pieces in place | time |
|---|---|---|---|
| 6  | 5/6   | 5/6   | 26 s |
| 24 | 18/28 | 10/24 | 54 s |
| 48 | found 48/48 (not rebuilt) | | |

## ShredBench pilot (arXiv 2604.23813), English news, 8 pieces, 3 documents
Text read by Nace Perception Parse; NED lower is better.
| document | scene NED | scene ROUGE-L | Drex rebuilt NED | BLEU | ROUGE-L |
|---|---|---|---|---|---|
| news_008 | 0.92 | 0.04 | 0.54 | 0.54 | 0.51 |
| news_031 | 0.94 | 0.03 | 0.68 | 0.45 | 0.41 |
| news_041 | 0.88 | 0.07 | 0.57 | 0.54 | 0.52 |
| mean     | 0.91 | 0.05 | 0.60 | 0.51 | 0.48 |
Paper, 8 pieces, all domains: Gemini 3 Pro 0.33, Gemini 3 Flash 0.34, Qwen-VL-Plus 0.59,
InternVL3.5-38B 0.74, GPT-5.1 0.77. Only 3 documents have public images and ground truth.
