# Beyond one GPU: article experiments

Source for [LLM Inference, Part 7](https://ishwarj.com/writings/llm-inference-7-beyond-one-gpu).
Edit `article.md`, then run `assemble.py`; the published Markdown is generated from this source plus saved SVGs.

## What was run

Runs on macOS arm64 (Apple M5 Pro, 64 GB), Python 3.12, PyTorch 2.14.1, transformers 5.18.0. There is no NVIDIA GPU here: **no multi-GPU speed was measured**. The laptop was shared with other heavy jobs during the runs, so timings use the fastest of several repeats and are used only for shapes and ratios.

- `common.py`: model shapes (Llama 3 report Table 3, DeepSeek-V3 report Section 4.2) and NVIDIA spec-sheet bandwidths.
- `memory_math.py`: weights plus KV cache for Llama 3.1 8B, 70B, 405B and DeepSeek-V3 against 80 GB GPUs, and the decode floor (weight bytes over aggregate HBM bandwidth). Arithmetic.
- `tp_demo.py`: **real runs.** (A) Megatron column-then-row split of an MLP, checked against the unsplit result, and the wrong row split. (B) Qwen2.5-0.5B with tensor parallelism across 2 separate processes that only communicate through `torch.distributed.all_reduce` (gloo backend, CPU, float32): logits and 24 greedy tokens compared with the unsplit Hugging Face model; all-reduce calls and bytes counted. (C) gloo all-reduce time from 4 B to 64 MiB with 2 and 4 processes, and an alpha-beta fit. These timings describe this laptop's CPU processes, not any GPU link.
- `measure_mps.py`: **real timings** on the Apple GPU (`mps`): copy bandwidth, and one Llama 3.1 8B MLP matrix (14,336 x 4,096, BF16) whole and as TP=2 and TP=4 shards for 1 to 4,096 tokens. Fastest of 40 runs.
- `collectives.py`: ring all-reduce traced step by step on 4 simulated ranks, byte counts against 2(p-1)/p, all-gather, reduce-scatter and all-to-all; ring and two-step all-reduce cost tables for Llama 3.1 70B (alpha swept over 2 to 10 microseconds, an assumption); DeepSeek-V3 all-to-all bytes checked against DeepEP's published timings.
- `layout_model.py`: a roofline model of one decode step for TP scaling (70B), one 8-GPU node (TP=8 x 1, TP=4 x 2, TP=2 x 4), 405B on two nodes (TP=16 against TP=8 x PP=2), and a pipeline schedule. Peak numbers; alpha assumed (5 us NVLink, 10 us InfiniBand).
- `moe_routing.py`: **real routing decisions** of OLMoE-1B-7B-0924 (64 experts, top-8, 16 layers; BF16 on `mps`) over 12,288 tokens of WikiText-2 and 12,288 tokens of Python code (codeparrot-clean-valid), saved in `results/moe_picks.npz`. Then, in NumPy: per-expert load, busiest-GPU-over-average for EP 8, 16 and 64 against random routing, and load-based placement and redundant copies learned on the first half of the tokens and tested on the second. `--reuse` re-analyses the saved decisions without the model.
- `kv_transfer.py`: KV bytes per prompt and transfer time over NVLink, 8 or 1 400 Gb/s NICs and 100 Gb/s Ethernet, checked against DistServe's OPT-66B example; exposed time with layer-by-layer sending. Prefill time assumes 50% of H100 peak.
- `disagg_sim.py`: an iteration-level simulator of 4 H100s serving Llama 3.1 8B: colocated (prefill first, or 512-token chunked prefill) against disaggregated 1+3, 2+2 and 3+1, on a chat and a long-prompt workload, with goodput under three SLOs. Costs come from a roofline with assumed efficiencies (70% of HBM bandwidth, 50% of peak FLOP/s, 1 ms per iteration); they are not engine benchmarks.

Outputs: `results/*.json` and `results/*_stdout.txt`. Figures: `figures.py` writes `results/figures.json`; `assemble.py` inlines them into the article. `results/graph_nodes.json` holds this part's topic-map nodes and edges, in the shape of `src/data/graph.json`, to be merged.

Screenshots: `shots_paper.py` (highlighted crops of the papers, manifest with PDF checksums in `results/shots_paper.json`), `web_shots.sh` (vLLM and SGLang docs, DeepSeek's inference overview; uses `code/sglang/docshot.mjs`), `termshots.py` (terminal pictures of the saved logs).

## Reproduce from the repository root

```bash
python3 -m venv /tmp/multigpu-env
/tmp/multigpu-env/bin/pip install -r code/multigpu/requirements.txt
cd code/multigpu
/tmp/multigpu-env/bin/python memory_math.py
/tmp/multigpu-env/bin/python tp_demo.py          # downloads Qwen2.5-0.5B
/tmp/multigpu-env/bin/python measure_mps.py      # Apple GPU; set dev = 'cuda' elsewhere
/tmp/multigpu-env/bin/python collectives.py
/tmp/multigpu-env/bin/python layout_model.py
/tmp/multigpu-env/bin/python moe_routing.py      # downloads OLMoE-1B-7B (14 GB); or: moe_routing.py --reuse
/tmp/multigpu-env/bin/python kv_transfer.py
/tmp/multigpu-env/bin/python disagg_sim.py
/tmp/multigpu-env/bin/python shots_paper.py
/tmp/multigpu-env/bin/python termshots.py
/tmp/multigpu-env/bin/python figures.py
/tmp/multigpu-env/bin/python assemble.py
cd ../..
bash code/multigpu/web_shots.sh
node code/agents/check_md.mjs content/writings/llm-inference-7-beyond-one-gpu.md
NODE_ENV=production npm run build
```

On a machine with several NVIDIA GPUs, change `tp_demo.py`'s backend from `gloo` to `nccl` and place each rank's tensors on `cuda:<rank>` to run the same tensor-parallel code over NVLink or PCIe.

## Sources checked

Papers (arXiv IDs verified 11 October 2026): Megatron-LM 1909.08053, GPipe 1811.06965, Pope et al. 2211.05102, Llama 3 2407.21783, DeepSeek-V3 2412.19437, Splitwise 2311.18677, DistServe 2401.09670, Mooncake 2407.00079, OLMoE 2409.02060.

Documentation read on 11 October 2026: vLLM [Parallelism and Scaling](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/), [Expert Parallel Deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/), [Disaggregated Prefilling](https://docs.vllm.ai/en/latest/features/disagg_prefill/) (vLLM main `187a0eb98aa42341d703f83421d693fa7585581b`); SGLang [Server Arguments](https://docs.sglang.io/docs/advanced_features/server_arguments) and [PD Disaggregation](https://docs.sglang.io/docs/advanced_features/pd_disaggregation) (SGLang main `f9cee8d2b2d96a626db1968ba81e4d72bb31794a`); NVIDIA [H100](https://www.nvidia.com/en-us/data-center/h100/), [NVLink](https://www.nvidia.com/en-us/data-center/nvlink/), [ConnectX-7](https://www.nvidia.com/en-us/networking/infiniband-adapters/); [DeepEP v1.2.1 README](https://github.com/deepseek-ai/DeepEP/tree/v1.2.1); NVIDIA Dynamo v1.5.1 (7 October 2026); llm-d v0.10.0 (29 September 2026). Recheck flags before using another release.
