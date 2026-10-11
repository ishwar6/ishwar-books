# A Mathematical Framework for Transformer Circuits, explained (code)

Code and results for the breakdown of *A Mathematical Framework for Transformer Circuits* (Elhage et al., Anthropic, 2021;
https://transformer-circuits.pub/2021/framework/index.html) at https://ishwarj.com/papers/transformer-circuits/

Every number in the articles that does not come from the paper comes from one of these scripts, and the exact output is saved in `results/`.

| File | What it does |
|---|---|
| `circuits_part1.py` | Part 1 on real models (attn-only-1l, attn-only-2l, gpt2): shapes, layer-norm folding, the residual stream split into parts, the direct path W_U W_E, virtual weights, bandwidth, the rotation test |
| `circuits_part1_toy.py` | Part 1 worked examples small enough to check by hand |
| `common.py` | a log that prints and saves to `results/<name>_stdout.txt`, and JSON results |
| `paper_shots.mjs`, `shots_part1.json` | open the paper's web page with Playwright, highlight phrases and crop excerpts into `public/img/papers/circuits/` |
| `figlib.py`, `cfig.py`, `alammar.py`, `figs_part1.py` | draw the SVG figures from the results (theme-aware CSS classes, no hard-coded colours) |
| `termshot.py` | renders a saved log as a terminal screenshot PNG |
| `assemble.py` | turns `partN.md` into `content/papers/transformer-circuits/<slug>.md`, inserting the figures; checks images, em dashes and banned words |
| `check_md.mjs` | renders a page's maths with KaTeX and reports errors (run from the repo root) |

```bash
pip install torch "transformer_lens<4"     # TransformerLens 4 removed HookedTransformer
python circuits_part1.py && python circuits_part1_toy.py
node paper_shots.mjs shots_part1.json        # needs Playwright
python figs_part1.py && python assemble.py 1
```
