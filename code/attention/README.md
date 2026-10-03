# Attention, From the Ground Up (code)

Code and results for the series at https://ishwarj.com/writings/attention-1-self-attention/

| File | What it does |
|---|---|
| `part1_attention.py` | Part 1: self-attention by hand, the √d scaling experiment, multi-head attention checked against PyTorch, real attention maps from Qwen2.5-0.5B |
| `part1_coref_check.py` | Part 1: tests the pronoun head on six unseen sentences |
| `part2_kv_variants.py` | Part 2: GQA/MQA checked against PyTorch, Qwen2.5-0.5B layer 0 recomputed by hand, decode timing for 32/8/1 KV heads, MLA (full and absorbed paths), KV cache sizes of real models |
| `part3_sparse.py` | Part 3: sliding-window attention checks, receptive field via gradients, rolling buffer cache, KV memory for Gemma 3 / Mistral / gpt-oss, decode timing |
| `part3_qwen.py` | Part 3: Qwen2.5-0.5B with windows and attention sinks; a DeepSeek-style lightning indexer per layer, trained with the dense warm-up KL loss and tested with top-k sparse attention |
| `part4_linear.py` | Part 4: linear attention (parallel vs recurrent), overwrite and capacity tests for the delta rule, forget-gate fading, our gated delta rule vs the Qwen3-Next reference code, hybrid memory, decode timing |
| `part4_train.py` | Part 4: five small models (softmax, gated, linear, Gated DeltaNet, hybrid) trained on books and on an associative-recall task |
| `figlib.py`, `figs_part*.py` | draws the figures from the results |
| `results/` | the JSON every number in the articles comes from |

```bash
pip install torch transformers
python part1_attention.py
python part1_coref_check.py
python part2_kv_variants.py
python part3_sparse.py
python part3_qwen.py
python part4_linear.py
python part4_train.py
```
