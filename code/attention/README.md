# Attention, From the Ground Up (code)

Code and results for the series at https://ishwarj.com/writings/attention-1-self-attention/

| File | What it does |
|---|---|
| `part1_attention.py` | Part 1: self-attention by hand, the √d scaling experiment, multi-head attention checked against PyTorch, real attention maps from Qwen2.5-0.5B |
| `part1_coref_check.py` | Part 1: tests the pronoun head on six unseen sentences |
| `figlib.py`, `figs_part*.py` | draws the figures from the results |
| `results/` | the JSON every number in the articles comes from |

```bash
pip install torch transformers
python part1_attention.py
python part1_coref_check.py
```
