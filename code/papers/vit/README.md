# Vision Transformer, explained (code)

Code and results for the six-part breakdown of *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale*
(Dosovitskiy et al.; arXiv:2010.11929) at https://ishwarj.com/papers/vit/

Every number in the articles that does not come from the paper comes from one of these scripts, and the exact output is saved in `results/`.

| File | What it does |
|---|---|
| `vit_partN*.py` | the experiments and worked examples of part N (see each file's docstring) |
| `common.py` | a log that prints and saves to `results/<name>_stdout.txt`, and JSON results |
| `paper_shots.py`, `shots_partN.py` | download the paper, highlight phrases and crop excerpts into `public/img/papers/vit/` |
| `figlib.py`, `vitfig.py`, `alammar.py`, `figs_partN.py` | draw the SVG figures from the results (theme-aware CSS classes, no hard-coded colours) |
| `termshot.py` | renders a saved log as a terminal screenshot PNG |
| `assemble.py` | turns `partN.md` into `content/papers/vit/<slug>.md`, inserting the figures; checks images, em dashes and banned words |
| `check_md.mjs` | renders a page's maths with KaTeX and reports errors (run from the repo root) |
| `results/` | the JSON and text output every number in the articles comes from |

```bash
pip install torch transformers datasets pymupdf pillow
python vit_part1.py && python vit_part1_math.py      # and the other vit_part*.py scripts
python shots_part1.py                                 # paper excerpts (needs pymupdf)
python figs_part1.py && python assemble.py 1
```
