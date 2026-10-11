# Serving LLMs in production: article experiments

Source for [LLM Inference, Part 6](https://ishwarj.com/writings/llm-inference-6-serving-in-production).
Edit `article.md`, then run `assemble.py`; the published Markdown is generated from this source plus saved SVGs.

## What was run

Runs on macOS arm64 (Apple M5 Pro), Python 3.14 for the simulations and Python 3.12 with PyTorch 2.14 on the Apple GPU (`mps`) for the measurements. Nothing here runs vLLM or SGLang; no NVIDIA GPU was available.

Other jobs shared the GPU while the measurements ran. Timings are therefore the **fastest** of several runs (the median is saved too), and they are slower than an idle machine.

- `measure.py`: **real timings**. Qwen2.5-0.5B, BF16: 18 decode steps (batch 1 to 128, contexts 256 to 4,096) and 11 prefills (16 to 4,096 tokens, some on a cached context), fastest of 30 interleaved rounds after 2 warm-up rounds (each configuration timed once per round). Fits the step-cost model `a + b*new + g*prefill_pairs + k*kv_read` used by every laptop simulation.
- `engine.py`: the discrete-event engine model (continuous batching, chunked prefill with a token budget, KV memory with newest-first preemption and recomputation, an LRU prefix cache, cancellation on timeout). `test_engine.py` checks it.
- `queueing.py`: M/M/1 and M/D/1 (DistServe eq. 1) against simulation; the engine under a load sweep (TTFT, TPOT, ITL, E2E percentiles, goodput); Little's law; closed versus open loop and coordinated omission with a 6-second stall; bursty arrivals.
- `capacity.py`: Llama 3.1 8B on one H100 from published specs: memory budget, Little's law, roofline prefill and decode costs, the engine driven by the roofline for BF16 and FP8, GPU count, and cost per million tokens (GPU prices are assumptions).
- `quant.py`: **real quality measurements**. Qwen2.5-0.5B perplexity on the WikiText-2 test split (299,078 tokens), KL divergence and top-1 agreement against BF16, for INT8, FP8, INT4 (per channel, g128, g128 with a simplified AWQ scale search), INT3, W8A8 (FP8 and INT8, per token and per tensor) and an FP8 KV cache. Weights and activations are rounded and turned back into float32 ("fake quantization"), so this measures quality, not speed.
- `route.py`: four replicas behind seven routing policies, including SGLang Model Gateway's cache-aware rule, on Zipf and hot-prefix traffic.
- `lora.py`: LoRA adapter memory for Llama 3.1 8B, and a **real** decode-step timing of 32 requests with 32 different rank-16 adapters (gathered batched products against a loop), checked against merged weights.
- `coldstart.py`: **real** cold-start phases in fresh processes, SSD read speed on a fresh file with `F_NOCACHE`, and an extrapolation to larger models at stated bandwidths.
- `autoscale.py`: autoscaling after a traffic jump with three cold-start times, scaling on in-flight load against scaling on GPU busy time.
- `failures.py`: a retry storm after a 10-second freeze (with and without cancellation, backoff and jitter), preemption with a small KV cache, head-of-line blocking with and without chunked prefill, and a noisy neighbour.
- `walkthrough.py`: the small worked examples in the text.

Outputs: `results/*.json` and `results/*_stdout.txt`. Figures: `figures.py` writes `results/figures.json`; `assemble.py` inlines them into the article.

Screenshots: `shots_paper.py` (highlighted crops of the papers, manifest with PDF checksums in `results/shots_paper.json`), `rowshot.mjs` (rows of docs tables), `termshots.py` (terminal pictures of the saved logs). The docs passages for `vllm bench serve` were taken with `code/sglang/docshot.mjs`.

## Reproduce from the repository root

```bash
python3 code/serving/walkthrough.py
~/.venvs/attn/bin/python code/serving/measure.py      # needs torch and a GPU; writes the cost model
python3 code/serving/queueing.py
python3 code/serving/capacity.py
python3 code/serving/route.py
python3 code/serving/autoscale.py
python3 code/serving/failures.py
~/.venvs/attn/bin/python code/serving/quant.py        # needs torch, transformers, pyarrow and WikiText-2 from the HF hub
~/.venvs/attn/bin/python code/serving/lora.py
~/.venvs/attn/bin/python code/serving/coldstart.py
~/.venvs/attn/bin/python code/serving/termshots.py
python3 code/serving/figures.py
python3 code/serving/assemble.py
python3 -m unittest discover -s code/serving -p 'test_*.py'
NODE_ENV=production npm run build
```

The simulations need only the standard library. Running `measure.py` on another GPU recalibrates every laptop simulation to that GPU.

## Sources

`results/sources.json` pins the commits read for flags, defaults and metric names: vLLM `187a0eb`, SGLang `f9cee8d`, vLLM Production Stack `4e076e8` (all 11 October 2026). Documentation links were checked on 11 October 2026. Paper PDFs are downloaded to `~/.cache/papers` and not committed.
