# MoE viewer: watch a mixture-of-experts model pick its experts, live

```bash
# DeepSeek-V2-Lite-Chat (15.7B weights, 2.4B per token, 26 MoE layers x 64 experts + 2 shared, top-6; 31 GB download)
~/.venvs/attn/bin/python code/deepseek/moe-viewer/server.py

# OLMoE-1B-7B (6.9B weights, 1.3B per token, 16 MoE layers x 64 experts, top-8; already cached on this Mac)
MODEL=allenai/OLMoE-1B-7B-0924 HF_HUB_OFFLINE=1 ~/.venvs/attn/bin/python code/deepseek/moe-viewer/server.py
```

Open http://localhost:8765.

## What is real and what is simulated

- **Real:** the model runs on this machine (Apple GPU through PyTorch `mps`, or CUDA, or CPU). Forward hooks on every
  router record the experts each token picks and their weights. Step times are measured per token.
- **Simulated:** the SSD / RAM / GPU tiers. An Apple Silicon Mac has one unified memory, so there is no separate GPU
  memory to fill. The page replays the real picks through a pretend machine: a GPU tier that holds the most-picked
  experts (re-pinned every 32 tokens and at each new answer), a least-recently-used RAM cache, and an SSD that holds
  everything. Sizes and SSD speed are sliders. Bytes per expert come from the model config (gate, up and down
  projections in bf16).

## Under the hood

- The **log panel** shows the server's lines (prefill size and time, each token with its step time and the experts of
  the first, middle and last MoE layer) and the cache's lines (where each token's picks came from, bytes from SSD).
  Tick "every layer's picks and weights" for all layers. Click any word of the answer to inspect its routing.
- The same server lines go to the terminal and to `logs/<date>.log`.
- `EXPERTS=batched_mm` (default) runs the picked experts as one batched matmul. On the M5 Pro it decoded OLMoE at
  about 240 ms per token, against about 2,000 ms for transformers' default loop over experts.
