# Segment Anything, explained (code)

Code and results for the six-part breakdown of *Segment Anything*
(Kirillov et al.; arXiv:2304.02643) at https://ishwarj.com/papers/sam/

Every number in the articles that does not come from the paper comes from one of these scripts, and the exact output is saved in `results/`.

| File | What it does |
|---|---|
| `sam_partN*.py` | the experiments and worked examples of part N (see each file's docstring) |
| `sam_road.py`, `figs_road.py`, `partb.md` | the Background page (the road to SAM): YOLO11n-seg and SAM ViT-H on the same photo (`pip install ultralytics`) |
| `common.py` | a log that prints and saves to `results/<name>_stdout.txt`, and JSON results |
| `paper_shots.py`, `shots_partN.py` | download the paper, highlight phrases and crop excerpts into `public/img/papers/sam/` |
| `figlib.py`, `samfig.py`, `alammar.py`, `figs_partN.py` | draw the SVG figures from the results (theme-aware CSS classes, no hard-coded colours) |
| `termshot.py` | renders a saved log as a terminal screenshot PNG |
| `assemble.py` | turns `partN.md` into `content/papers/sam/<slug>.md`, inserting the figures; checks images, em dashes, banned words and the ASD-STE100 writing rules (sentence and paragraph length, unapproved words) |
| `check_md.mjs` | renders a page's maths with KaTeX and reports errors (run from the repo root) |
| `results/` | the JSON and text output every number in the articles comes from |

```bash
pip install torch transformers datasets pymupdf pillow
python sam_part1.py                  # and the other sam_part*.py scripts
python shots_part1.py                # paper excerpts (needs pymupdf)
python figs_part1.py && python termshot.py part1 && python assemble.py 1
node check_md.mjs ../../../content/papers/sam/part-1-the-big-idea.md
```
