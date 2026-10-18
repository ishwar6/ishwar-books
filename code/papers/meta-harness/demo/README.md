# The Meta-Harness demo (/papers/meta-harness/demo): how its files are made

The page is `src/demos/meta-harness/` (React, three tabs). No model runs in the page. Every number it shows comes from
the files in `public/demos/meta-harness/`, made by the scripts here.

| Script | What it makes |
|---|---|
| `paper_numbers.json` | The paper's numbers, typed by hand: Tables 1, 2, 3, 4, 6, 7, 8 and 9, the Figure 3 baselines, and the Appendix A.2 scores. Each block names its table and page. |
| `export_paper.py` | Checks `paper_numbers.json` against the text of the paper's PDF: every row of every table must appear on its page, in order (it also checks that each Table 6 improvement equals the difference to the no-retriever row). Writes `public/demos/meta-harness/paper.json`. |
| `export_run.py` | Exports the saved search runs of `../scratch/runs/<run>/` (Part 6): every candidate's code, parent, proposal, search scores, held-out test score, rejected candidates and the log, to `run/<run>.json`, and every candidate's full search-set traces to `run/<run>/<candidate>.json` (loaded only when the reader opens them). Checks each score against the run's own `scores.json`, `traces.jsonl`, `summary.json` and `heldout/`. |
| `make_thumb.mjs` | The card picture `thumb.jpg` (800 x 420), a screenshot of the run chart. |

```bash
~/.venvs/attn/bin/python export_paper.py          # needs pymupdf and ~/.cache/papers/2603.28052.pdf
~/.venvs/attn/bin/python export_run.py            # every finished run in ../scratch/runs (smoke* skipped)
```

## Notes

- The page never names the paper's proposer product. Table 7's row for Anthropic's own coding-agent harness is written
  "Anthropic's coding agent".
- Figure 3 draws accuracy against context in characters. Its stars are Table 9 (characters). Its baselines are Table 2,
  whose context is in tokens; the figure places them at about four characters per token, so their x positions in
  `paper_numbers.json` are read from the plot and marked as such. The few-shot 4 and 16 points and the non-Pareto
  candidates have no printed numbers and are left out.

## Results (this machine)

- `export_paper.py`: 235 table cells and labels found on the right PDF page, in order.
- `export_run.py`: runs `scripted-1` (13 candidates; selected `012_60_tfidf_examples_verify`, search 58.0%, test 50.0%)
  and `local-llm-1` (8 candidates, 5 rejected; selected `004_llm_t1_1`, search 48.0%, test 46.0%). 3.0 MB in all; the page
  loads about 100 to 150 KB per run, and one candidate's traces (60 to 360 KB) only when they are opened.
- `check_browser.mjs` (built site, `vite preview`): 67/67 paper values and 84/84 run values on the page equal the
  source files; 0 console errors (`results/browser_check.txt`).
