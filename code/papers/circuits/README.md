# A Mathematical Framework for Transformer Circuits, explained (code)

Code and results for the breakdown of *A Mathematical Framework for Transformer Circuits* (Elhage et al., Anthropic, 2021;
https://transformer-circuits.pub/2021/framework/index.html) at https://ishwarj.com/papers/transformer-circuits/

Every number in the articles that does not come from the paper comes from one of these scripts, and the exact output is saved in `results/`.

| File | What it does |
|---|---|
| `circuits_part1.py` | Part 1 on real models (attn-only-1l, attn-only-2l, gpt2): shapes, layer-norm folding, the residual stream split into parts, the direct path W_U W_E, virtual weights, bandwidth, the rotation test |
| `circuits_part1_toy.py` | Part 1 worked examples small enough to check by hand |
| `circuits_part2.py` | Part 2 on attn-only-1l and attn-only-2l: one head's matrices, heads add independently, a real attention pattern, the tensor form of a head, the pattern from W_QK in one step, ranks and singular values of W_OV and W_QK, reparametrising values and queries, and the OV and QK circuits of head 0.3 |
| `circuits_part2_toy.py` | Part 2 worked examples small enough to check by hand |
| `common.py` | a log that prints and saves to `results/<name>_stdout.txt`, and JSON results |
| `paper_shots.mjs`, `shots_part1.json`, `shots_part2.json` | open the paper's web page with Playwright, highlight phrases and crop excerpts into `public/img/papers/circuits/` |
| `figlib.py`, `cfig.py`, `alammar.py`, `figs_part1.py`, `figs_part2.py` | draw the SVG figures from the results (theme-aware CSS classes, no hard-coded colours) |
| `termshot.py` | renders a saved log as a terminal screenshot PNG |
| `assemble.py` | turns `partN.md` into `content/papers/transformer-circuits/<slug>.md`, inserting the figures; checks images, em dashes and banned words |
| `check_md.mjs` | renders a page's maths with KaTeX and reports errors (run from the repo root) |

```bash
pip install torch "transformer_lens<4"     # TransformerLens 4 removed HookedTransformer
python circuits_part1.py && python circuits_part1_toy.py
node paper_shots.mjs shots_part1.json        # needs Playwright
python figs_part1.py && python assemble.py 1
python circuits_part2.py && python circuits_part2_toy.py
node paper_shots.mjs shots_part2.json
python figs_part2.py && python assemble.py 2
```

## The side-by-side reader

`/papers/transformer-circuits/read` shows the paper on the left and the explanations on the right, scrolling in step.
The paper is a web page, not a PDF, so we print our own copy.

| File | What it does |
|---|---|
| `paper_pdf.mjs` | opens the paper's web page with Playwright, waits for the maths, fonts and figures, and prints it to `~/.cache/papers/circuits-framework.pdf` (US Letter, one column, figures fitted to the text width, the site's top bar hidden). It finds every excerpt of `shots_part*.json` with the same rules as `paper_shots.mjs`, paints its key phrases yellow, and tags the excerpt with invisible links, so that its exact place in the PDF is known |
| `reader.py` | for every unnamed `> [!PAPER] ·` box of the numbered parts, in article order: reads the excerpt's place from the tags, checks that its `find` and `upto` words are there, draws a numbered teal dot and bar in the left margin, removes the tags, down-samples the images, and writes `public/papers/pdf/transformer-circuits.pdf` and `transformer-circuits.json` (same format as `sam.json`, with `url` in place of `arxiv`) |

```bash
python reader.py            # re-run after every new part (uses the cached print)
python reader.py --fresh    # print the web page again first (needs network and Playwright)
```

A new part needs nothing else: its boxes use screenshots listed in `shots_partN.json`, which `paper_pdf.mjs` reads.
Run `npm run content` afterwards; the site turns the reader on because the JSON file exists.
