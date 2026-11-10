# 04. Hardware and serving: what it takes to run DeepSeek, and how DeepSeek serves it

Research dossier for "DeepSeek, From the Ground Up". Compiled 2026-11-10.

Conventions used in this file:

- GB means 10^9 bytes. GiB means 2^30 bytes. Many community posts mix them; this matters (see the "131 GB" note in section 1.3).
- "Vendor claim" means the number comes from the people who built the thing (DeepSeek, a model or engine team, a hardware maker). "Independent" means a third party measured it. Most numbers in this space are vendor claims.
- "Derived" means I computed it from cited inputs. The arithmetic is shown so a writer can check it.
- "UNVERIFIED" means I could not confirm it from a primary source.
- All Hugging Face (HF) sizes below were summed from the HF API file listings (`https://huggingface.co/api/models/<repo>?blobs=true`) on 2026-11-10.

---

## 0. The model line-up as of November 2026

Writers should know that the DeepSeek family moved on a lot after V3/R1. The HF organisation lists these main checkpoints (source: https://huggingface.co/api/models?author=deepseek-ai&sort=createdAt&direction=-1&limit=40):

| Model | Released (HF) | Total params | Active params | Native weight format | Context |
|---|---|---|---|---|---|
| DeepSeek-V2-Lite | 2024-05 | 16B (15.7B) | 2.4B | BF16 | 32K |
| DeepSeek-V2 / V2.5 | 2024-05 / 2024-09 | 236B | 21B | BF16 | 128K |
| DeepSeek-V3, V3-0324, R1, R1-0528 | 2024-12 to 2025-05 | 671B (+14B MTP module) | 37B | FP8 (E4M3, 128x128 blocks) | 128K |
| DeepSeek-V3.1, V3.1-Terminus | 2025-08 / 2025-09 | 671B | 37B | FP8 with UE8M0 scales | 128K |
| DeepSeek-V3.2-Exp, V3.2, V3.2-Speciale | 2025-09 to 2025-12 | 671B | 37B | FP8 with UE8M0 scales | 128K |
| DeepSeek-V4-Flash (preview), V4-Flash-0731 | 2026-04 / 2026-07 | 284B | 13B | FP4 experts + FP8 rest | 1M |
| DeepSeek-V4-Pro (preview), V4-Pro-0813 | 2026-04 / 2026-08 | 1.6T | 49B | FP4 experts + FP8 rest | 1M |
| DeepSeek-V4.1-Flash, V4.1-Flash-1017 | 2026-09 / 2026-10 | 552B backbone + 196B Engram | 8B prefill / 16B decode | FP4 experts + FP8 rest | 1M |
| R1 distills (Qwen 1.5B/7B/14B/32B, Llama 8B/70B) | 2025-01 | 1.5B to 70B dense | all | BF16 | 128K |
| DeepSeek-R1-0528-Qwen3-8B | 2025-05 | 8.2B dense | all | BF16 | 128K |

Sources: model cards and configs at https://huggingface.co/deepseek-ai/DeepSeek-V3 , https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite , https://huggingface.co/deepseek-ai/DeepSeek-V3.1 , https://huggingface.co/deepseek-ai/DeepSeek-V3.2 , https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro , https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 , https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash , https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 . V4 release dates: DeepSeek API news list at https://api-docs.deepseek.com/news/news261017 (V4 Preview 2026/04/24, V4-Pro GA 2026/08/13, V4.1-Flash 2026/09/10, V4.1-Flash-1017 2026/10/17).

Important for the series: the DeepSeek API no longer serves V3/R1. As of 2026-11-10 the API serves only `deepseek-flash` (DeepSeek-V4.1-Flash-1017) and `deepseek-v4-pro` (DeepSeek-V4-Pro-0813) (https://api-docs.deepseek.com/quick_start/pricing). The V4.1-Flash announcement says V4-Pro requests are routed to V4.1-Flash "until V4.1-Pro launches" (https://api-docs.deepseek.com/news/news260910). The pricing page still lists V4-Pro-0813 as a model version, so the routing state is not fully clear from public pages (UNVERIFIED which model actually answers `deepseek-v4-pro` calls today).

---

## 1. Memory arithmetic

### 1.1 The basic rule

Weight memory = number of parameters x bytes per parameter.

| Format | Bytes per parameter | Notes |
|---|---|---|
| BF16 / FP16 | 2 | Training format of V2 and the distills. |
| FP8 (E4M3) | 1 | Native format of V3 and later, plus small per-block scale tensors. |
| INT4 / FP4 / Q4 | 0.5 | Plus scales; real GGUF Q4_K_M is about 4.8 bits per weight on V3 (see below). |
| 1.58-bit class (IQ1_S) | about 0.2 | Only for the large MoE expert tensors. Other tensors stay higher. |

Weights are not the whole bill. You also need KV cache (section 1.4), activations, CUDA graphs and engine buffers. Engines like vLLM and SGLang reserve a fixed fraction of GPU memory (for example `--gpu-memory-utilization 0.95` or `--mem-fraction-static 0.90`) and use what is left after weights for KV cache.

### 1.2 Weight sizes per model

Theoretical columns are derived (params x bytes). "Real" columns are actual file sizes from HF listings.

#### V3 / R1 / V3.1 / V3.2 (671B; 684.5B tensors on disk including the 14B MTP module)

| Format | Arithmetic | Size | Source |
|---|---|---|---|
| BF16 (theoretical) | 671e9 x 2 B | 1,342 GB | derived |
| BF16 GGUF (real) | sum of files | 1,342.3 GB | https://huggingface.co/unsloth/DeepSeek-R1-GGUF |
| FP8 official checkpoint (real) | 680.6B FP8 params x 1 B + 3.9B BF16 params x 2 B + scales | 688.6 GB (V3, R1, V3.1); 689.5 GB (V3.2) | https://huggingface.co/deepseek-ai/DeepSeek-V3 , https://huggingface.co/deepseek-ai/DeepSeek-V3.2 |
| INT4 (theoretical) | 671e9 x 0.5 B | 335.5 GB | derived |
| GGUF Q4_K_M (real) | | 404.4 GB (R1), 405.4 GB (V3.1, V3.2) | https://huggingface.co/unsloth/DeepSeek-R1-GGUF , https://huggingface.co/unsloth/DeepSeek-V3.2-GGUF |
| Q4_K_M effective bits | 404.4e9 x 8 / 671e9 | about 4.8 bits per weight | derived |
| GGUF Q8_0 (real) | | 713.3 GB | same repos |
| Ollama `deepseek-r1:671b` (Q4_K_M) | | 404 GB | https://ollama.com/library/deepseek-r1/tags |

The HF safetensors metadata for DeepSeek-V3 reports 680,571,043,840 F8_E4M3 parameters plus 3,918,786,560 BF16 parameters (https://huggingface.co/api/models/deepseek-ai/DeepSeek-V3). The BF16 parts are mostly embeddings, norms and other small tensors. DeepSeek ships only FP8 weights and gives a `fp8_cast_bf16.py` script for anyone who needs BF16 (https://github.com/deepseek-ai/DeepSeek-V3).

Unsloth dynamic low-bit GGUFs for the 671B family (all real sizes from HF listings):

| Quant | R1 (Jan 2025) | R1-0528 | V3.1 | V3.2 |
|---|---|---|---|---|
| UD-IQ1_S ("1.58-bit") | 140.2 GB | 185.5 GB | 192.3 GB | 184.1 GB |
| UD-IQ1_M ("1.73-bit") | 168.9 GB | 200.5 GB | 206.8 GB | 199.3 GB |
| UD-IQ2_XXS ("2.22-bit") | 196.2 GB | 216.8 GB | 225.9 GB | 217.1 GB |
| UD-Q2_K_XL ("2.51-bit") | 226.6 GB | 251.1 GB | 255.7 GB | 246.6 GB |
| UD-TQ1_0 (single file) | n/a | 161.6 GB | 170.5 GB | 161.3 GB |
| UD-Q4_K_XL | n/a | 384.0 GB | 386.9 GB | 407.8 GB |

Sources: https://huggingface.co/unsloth/DeepSeek-R1-GGUF , https://huggingface.co/unsloth/DeepSeek-R1-0528-GGUF , https://huggingface.co/unsloth/DeepSeek-V3.1-GGUF , https://huggingface.co/unsloth/DeepSeek-V3.2-GGUF .

#### V2 (236B) and V2-Lite (16B)

| Model | Format | Arithmetic | Size | Source |
|---|---|---|---|---|
| V2 / V2.5 | BF16 official (real) | 235.7B x 2 B | 471.5 GB | https://huggingface.co/deepseek-ai/DeepSeek-V2 |
| V2 | FP8 (theoretical) | 235.7B x 1 B | 236 GB | derived; DeepSeek did not ship FP8 V2 |
| V2 | INT4 (theoretical) | 235.7B x 0.5 B | 118 GB | derived |
| V2 | Ollama `deepseek-v2:236b` (4-bit) | | 133 GB | https://ollama.com/library/deepseek-v2/tags |
| V2-Lite | BF16 official (real) | 15.7B x 2 B | 31.4 GB | https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite |
| V2-Lite | FP8 / INT4 (theoretical) | 15.7B x 1 / x 0.5 | 15.7 GB / 7.9 GB | derived |
| V2-Lite | Ollama `deepseek-v2:16b` (4-bit) | | 8.9 GB | https://ollama.com/library/deepseek-v2/tags |

#### The R1 distills (dense)

BF16 sizes are the official HF checkpoints. Q4_K_M sizes are Unsloth GGUFs. Ollama sizes are the default tags (4-bit).

| Distill | Params (HF metadata) | BF16 (real) | FP8 (theoretical) | Q4_K_M GGUF (real) | Ollama tag size |
|---|---|---|---|---|---|
| R1-Distill-Qwen-1.5B | 1.78B | 3.6 GB | 1.8 GB | 1.1 GB | 1.1 GB |
| R1-Distill-Qwen-7B | 7.62B | 15.2 GB | 7.6 GB | 4.7 GB | 4.7 GB |
| R1-Distill-Llama-8B | 8.03B | 16.1 GB | 8.0 GB | 4.9 GB | n/a (see note) |
| R1-0528-Qwen3-8B | 8.19B | 16.4 GB | 8.2 GB | n/a here | 5.2 GB (`deepseek-r1:8b`, `latest`) |
| R1-Distill-Qwen-14B | 14.77B | 29.5 GB | 14.8 GB | 9.0 GB | 9.0 GB |
| R1-Distill-Qwen-32B | 32.76B | 65.5 GB | 32.8 GB | 19.9 GB | 20 GB |
| R1-Distill-Llama-70B | 70.55B | 141.1 GB | 70.6 GB | 42.5 GB | 43 GB |

Sources: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B (and the sibling repos), https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-32B-GGUF (and siblings), https://ollama.com/library/deepseek-r1/tags .

Note: on Ollama, `deepseek-r1:8b` and `deepseek-r1:latest` now point to DeepSeek-R1-0528-Qwen3-8B, not the original Llama-8B distill (https://ollama.com/library/deepseek-r1). A writer who says "ollama run deepseek-r1" should say which model that pulls.

Note: these distills are Qwen2.5 or Llama 3 models fine-tuned on R1 outputs. They do not use MLA or MoE. They are "DeepSeek" in training data only, not in architecture (https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B).

#### 2026 models: V4-Flash, V4-Pro, V4.1-Flash

The V4 family ships routed experts in FP4 ("FP4 + FP8 Mixed: MoE expert parameters use FP4 precision; most other parameters use FP8", https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro). So the official checkpoint is already about 4.3 bits per weight, and community 4-bit GGUFs barely shrink it.

| Model | Format | Arithmetic | Size | Source |
|---|---|---|---|---|
| V4-Pro (1.6T) | BF16 (theoretical) | 1.6e12 x 2 B | 3.2 TB | derived |
| V4-Pro | all-FP8 (theoretical) | 1.6e12 x 1 B | 1.6 TB | derived |
| V4-Pro preview | official FP4+FP8 (real) | 864.7e9 x 8 / 1.6e12 | 864.7 GB, about 4.3 bits/weight | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro |
| V4-Pro-0813 | official FP4+FP8 with DSpark module (real) | | 892.7 GB | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 |
| V4-Pro-0813 | Unsloth UD-Q4_K_XL / UD-Q8_K_XL | | 849.7 GB / 873.4 GB | https://huggingface.co/unsloth/DeepSeek-V4-Pro-0813-GGUF |
| V4-Flash (284B) | BF16 (theoretical) | 284e9 x 2 B | 568 GB | derived |
| V4-Flash preview | official FP4+FP8 (real) | | 159.6 GB | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash |
| V4-Flash | Unsloth UD-IQ1_S / UD-Q2_K_XL / UD-Q4_K_XL | | 82.5 / 96.8 / 155.1 GB | https://huggingface.co/unsloth/DeepSeek-V4-Flash-GGUF |
| V4.1-Flash-1017 | official FP4+FP8 (real) | | 510.3 GB | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 |
| V4.1-Flash-1017 | Unsloth UD-IQ1_S / UD-Q2_K_XL / UD-Q4_K_XL | | 233.0 / 334.8 / 500.7 GB | https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF |

V4.1-Flash has a twist: of its size, the Engram n-gram tables are large but sparsely read. Unsloth's quant table splits each file into "Engram GB" and "Rest GB". The Rest is the minimum RAM/VRAM; the Engram can stay on SSD via mmap. Examples: UD-IQ1_S is 233.0 GB total, 110.6 GB Engram, 122.4 GB Rest; UD-Q4_K_XL is 500.7 GB total, 202.8 GB Engram, 297.9 GB Rest (vendor claim, https://unsloth.ai/docs/models/deepseek-v4.1-flash). The antirez DwarfStar (ds4) GGUFs say each language GGUF includes "about 188.83 GiB of FP8 Engram tables" read from disk as needed (https://huggingface.co/antirez/deepseek-v4.1-flash-gguf).

### 1.3 The "131 GB" Unsloth 1.58-bit quant, checked

Unsloth's January 2025 blog said the 1.58-bit R1 is 131 GB and the dynamic set ranges "from 131GB to 212GB" (https://unsloth.ai/blog/deepseekr1-dynamic). The HF folder `DeepSeek-R1-UD-IQ1_S` sums to 140.2 GB (decimal). 140.2e9 / 2^30 = 130.6 GiB. So "131 GB" is really 131 GiB. Same for the top end: UD-Q2_K_XL is 226.6 GB = 211.0 GiB, the "212 GB". Effective average: 140.2e9 x 8 / 671e9 = 1.67 bits per weight. The "1.58-bit" label describes most MoE expert weights, not the file average. Unsloth's own description: the first 3 dense layers stay at higher precision (they are "only 0.5% of all weights") and 1.58-bit is used for "88%" of the weights (reported via https://wandb.ai/byyoung3/ml-news/reports/Unsloth-unveils-1-58-bit-DeepSeek-R1--VmlldzoxMTEyMjU3Mg ; primary blog is https://unsloth.ai/blog/deepseekr1-dynamic , which returns HTTP 403 to automated fetchers, so I could not re-read it directly).

Later R1-0528, V3.1 and V3.2 "UD-IQ1_S" files are larger (184 to 192 GB) than the January R1 one. A writer should not reuse "131 GB" for those models.

### 1.4 KV cache: MLA arithmetic

#### V3 / R1 / V3.1 (MLA, 61 layers)

From config.json (https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/config.json): `num_hidden_layers` 61, `kv_lora_rank` 512, `qk_rope_head_dim` 64, `num_attention_heads` 128, `qk_nope_head_dim` 128, `v_head_dim` 128.

MLA caches one compressed latent (512 values) plus one shared RoPE key (64 values) per layer, for all 128 heads together.

- Values per token per layer: 512 + 64 = 576
- Values per token: 576 x 61 = 35,136
- BF16 bytes per token: 35,136 x 2 = 70,272 B (about 68.6 KiB)
- 128K context (131,072 tokens): 70,272 x 131,072 = 9.21 GB (8.58 GiB) per sequence

Compare with a naive multi-head cache for the same model (derived): each head would store K (128 nope + 64 rope = 192) and V (128): 128 heads x 320 values x 61 layers x 2 B = 4,997,120 B, about 5.0 MB per token. MLA is about 71x smaller (4,997,120 / 70,272). At 128K, naive MHA would be 655 GB per sequence.

#### V3.2 (MLA + DeepSeek Sparse Attention, FP8 cache)

FlashMLA defines the V3.2 "FP8 with scale" cache row as 656 bytes per token per layer: 512 bytes of FP8 latent, 16 bytes of 4 FP32 scales, 128 bytes of BF16 RoPE (https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7 , README "FP8 with scale" section).

- Bytes per token: 656 x 61 = 40,016 B
- 128K context: 40,016 x 131,072 = 5.24 GB per sequence
- The lightning indexer also keeps its own small key cache per layer (`index_head_dim` 128 in config). Exact bytes in production engines: UNVERIFIED, not included above.

DSA does not shrink the cache. It cuts attention compute: each query attends to `index_topk` = 2048 selected tokens (https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json).

#### V2 and V2-Lite

- V2: 60 layers, same 512 + 64. 576 x 60 x 2 B = 69,120 B per token. At 128K: 9.06 GB.
- V2-Lite: 27 layers. 576 x 27 x 2 B = 31,104 B per token. At its 32K context: 1.02 GB. At 128K (not supported by the card, shown for comparison): 4.08 GB.

Configs: https://huggingface.co/deepseek-ai/DeepSeek-V2/blob/main/config.json , https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/blob/main/config.json . Context lengths from the V2-Lite card table (32k for Lite, 128k for V2).

#### R1 distills (GQA, BF16 cache), for contrast

Formula: 2 (K and V) x layers x KV heads x head_dim x 2 bytes.

| Distill | Layers | KV heads | head_dim | Bytes/token | 128K context |
|---|---|---|---|---|---|
| Qwen-1.5B | 28 | 2 | 128 | 28,672 | 3.76 GB |
| Qwen-7B | 28 | 4 | 128 | 57,344 | 7.52 GB |
| Llama-8B | 32 | 8 | 128 | 131,072 | 17.18 GB |
| Qwen-14B | 48 | 8 | 128 | 196,608 | 25.77 GB |
| Qwen-32B | 64 | 8 | 128 | 262,144 | 34.36 GB |
| Llama-70B | 80 | 8 | 128 | 327,680 | 42.95 GB |

Configs from each repo's config.json (for example https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B/blob/main/config.json). Teaching point: the 671B V3 needs less KV cache per token (70 KB) than the 8B Llama distill (131 KB).

#### V4 and V4.1 (compressed and sparse attention)

V4 uses a hybrid of Compressed Sparse Attention (CSA, compress ratio 4) and Heavily Compressed Attention (HCA, ratio 128), plus a 128-token sliding window (config `compress_ratios`, `sliding_window` 128, `head_dim` 512, `num_key_value_heads` 1; https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/config.json).

Primary claims:
- "In the 1M-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2." V4-Flash: "10% of the single-token FLOPs and 7% of the KV cache size" (V4 tech report, https://arxiv.org/abs/2606.19348 ; model card https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro).
- Against a BF16 GQA8 baseline with head_dim 128, V4's KV cache is "approximately 2%" at 1M context (same report).
- FlashMLA's V4 cache row is 584 bytes per compressed entry: 448 B FP8 NoPE + 128 B BF16 RoPE + 8 B scale row (https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7).
- V4.1-Flash: "global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash", and persistent (SSD or host) cache about 1/8 of V4-Flash. It claims a 437x reduction in per-token global KV cache versus DeepSeek V1 (V4.1 tech report, https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf). V4.1 cache rows are 528 B (FP8) or 288 B (FP4) per entry (https://github.com/deepseek-ai/FlashMLA).

Derived estimates (mine; they ignore the indexer cache and the bounded sliding-window cache):
- V4-Pro: 30 CSA layers x 584/4 + 31 HCA layers x 584/128 = 4,380 + 141 = about 4,521 B per token. At 1M tokens: about 4.7 GB.
- V4-Flash: 21 x 584/4 + 20 x 584/128 = about 3,157 B per token. At 1M tokens: about 3.3 GB.
- V3.2 at 1M (extrapolated, V3.2 only supports 128K): 40,016 x 1,048,576 = 42.0 GB. So V4-Pro is about 11% and V4-Flash about 8% of V3.2. This matches the report's 10% and 7%.
- V4.1-Flash: 890 B x 1,048,576 = 0.93 GB at 1M tokens; 117 MB at 128K.

Arc for writers, bytes per token of KV cache: naive MHA on V3 about 5 MB, MLA V3 BF16 70 KB, V3.2 FP8 40 KB, V4-Pro about 4.5 KB, V4.1-Flash 0.89 KB.

---

## 2. Serving hardware recipes (data-centre GPUs and NPUs)

### 2.1 Useful hardware numbers

| Accelerator | Memory | Bandwidth | Source |
|---|---|---|---|
| NVIDIA H200 | 141 GB HBM3e | 4.8 TB/s | https://www.nvidia.com/en-us/data-center/h200/ |
| AMD MI300X | 192 GB HBM3 | 5.3 TB/s | https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html |
| Huawei Ascend 910C | 128 GB (2 dies x 64 GB) | n/a | secondary summary of https://arxiv.org/abs/2506.12708 (UNVERIFIED from Huawei spec) |
| Apple M3 Ultra (Mac Studio) | up to 512 GB unified | "over 800GB/s" | https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/ |

Why 8x H200 is the canonical V3 box (derived): 8 x 141 = 1,128 GB HBM. The FP8 checkpoint is 688.6 GB. That leaves about 440 GB minus engine overhead for KV cache. 8x H100 (8 x 80 = 640 GB) cannot hold 688.6 GB, so H100 needs two nodes. 8x MI300X = 1,536 GB.

### 2.2 V3 / R1 / V3.1 (671B, FP8)

SGLang hardware table (vendor docs, https://docs.sglang.io/basic_usage/deepseek_v3.html):

| Weights | Hardware listed |
|---|---|
| FP8 (recommended) | 8x H200, 8x B200, 8x MI300X, 2x8 H100/H800/H20 |
| BF16 (upcast) | 2x8 H200, 2x8 MI300X, 4x8 H100/H800, 4x8 A100/A800 |
| INT8 | 16x A100/A800, 32x L40S, Xeon 6980P CPU, 4x Ascend A3 series |
| W4A8 / AWQ / MXFP4 / NVFP4 | 8x H20/H100, 4x H200; 8x H100/A100; 8/4x MI355X/MI350X; 8/4x B200 |

SGLang flags (same page):
- Single node: `python -m sglang.launch_server --model deepseek-ai/DeepSeek-V3 --tp 8` (add `--reasoning-parser deepseek-v3` for thinking output). The page says not to add `--quantization fp8` because the checkpoint is already FP8.
- DP attention: `--tp 8 --attn-dp-size 8`, recommended for high-throughput, large-batch use, not for low latency.
- MTP speculative decoding: `--speculative-algorithm EAGLE` with defaults `--speculative-num-steps 3 --speculative-eagle-topk 1 --speculative-num-draft-tokens 4`. Reported "up to 1.8x decoding speed at batch size 1 and 1.5x at batch size 32" on H200 TP8 (vendor claim).
- Reported numbers on 8x MI300X, TP8, SGLang 0.5.7, 1K in / 1K out: concurrency 1 with EAGLE: 34.26 output tok/s, TPOT 25.04 ms. Concurrency 100 with `--ep 8 --attn-dp-size 8`: 463.80 output tok/s total, TPOT 214.73 ms (vendor docs).

vLLM recipe (vendor docs, https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3.html):
- FP8 on 8x H200 or 8x MI300X; FP4 (`nvidia/DeepSeek-R1-FP4`) on 4x B200.
- `vllm serve deepseek-ai/DeepSeek-R1-0528 --trust-remote-code --tensor-parallel-size 8 --enable-expert-parallel`, or `--data-parallel-size 8 --enable-expert-parallel`.
- ROCm adds `VLLM_ROCM_USE_AITER=1` and `VLLM_ROCM_USE_AITER_MOE=1`.
- The only number on the page is a single-request sample (61.00 output tok/s, TPOT 15.85 ms) with unstated hardware. Do not cite it as a throughput result.

DeepSeek's own early list of engines (https://github.com/deepseek-ai/DeepSeek-V3 , section 6): SGLang, LMDeploy, TensorRT-LLM, vLLM, LightLLM, AMD GPUs via SGLang, and Huawei Ascend via MindIE (BF16 at the time; "INT8 and BF16" in the list).

Large-scale results (vendor or engine-team claims):
- SGLang on 96 H100 (12 nodes x 8, Atlas Cloud), PD disaggregation + EP via DeepEP + EPLB + two-batch overlap: 52.3k input tok/s and 22.3k output tok/s per node for 2,000-token inputs; prefill EP32 on 4 nodes, decode EP72 on 9 nodes (the decode measurement used 9 nodes). Cost claim: $0.20 per 1M output tokens. Up to 5x output throughput versus vanilla TP16 on the same hardware (https://lmsys.org/blog/2025-05-05-large-scale-ep/).
- vLLM wide-EP: "sustained throughput of 2.2k tokens/s per H200 GPU in production-like, multi-node deployments" on a CoreWeave H200 cluster with ConnectX-7 InfiniBand, up from about 1.5k (https://vllm.ai/blog/large-scale-serving , 2025-12-17). Flags: `--enable-expert-parallel`, `--enable-dbo` (dual batch overlap), EPLB built in.
- Huawei CloudMatrix384 (384 Ascend 910 NPUs + 192 Kunpeng CPUs), R1 with INT8: prefill 6,688 tok/s per NPU, decode 1,943 tok/s per NPU at under 50 ms TPOT, 538 tok/s per NPU under 15 ms TPOT; EP320 (https://arxiv.org/abs/2506.12708). Secondary summaries say the 6,688 figure assumes perfect EPLB (UNVERIFIED in this dossier).
- KTransformers + SGLang hybrid on 8x L20 + Xeon Gold 6454S, R1-0528 FP8: 227.85 tok/s total, 87.58 tok/s output at 8-way concurrency (vendor claim, https://github.com/kvcache-ai/ktransformers).

### 2.3 V3.2-Exp / V3.2

- DeepSeek's README: SGLang Docker images `lmsysorg/sglang:dsv32` (H200), `dsv32-rocm` (MI350), `dsv32-a2` and `dsv32-a3` (Ascend NPUs); launch `--tp 8 --dp 8 --enable-dp-attention`. "vLLM provides day-0 support" (https://github.com/deepseek-ai/DeepSeek-V3.2-Exp).
- vLLM recipe: 8x H200 or H20, or 8x B200; recommended `vllm serve deepseek-ai/DeepSeek-V3.2-Exp -dp 8 --enable-expert-parallel`; kernels "mainly optimized for TP=1 (DP=8, EP=8)"; default FP8 KV cache; set `VLLM_USE_DEEP_GEMM=0` to skip DeepGEMM MoE (some H20 users reported it faster). No throughput numbers (https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html).
- vLLM-Ascend: DeepSeek-V3.2 W8A8 fits one Atlas 800 A3 node (16 NPUs) or two Atlas 800 A2 nodes; multi-node A3 example uses DP 2, TP 16 and EP (https://docs.vllm.ai/projects/ascend/en/v0.13.0/tutorials/DeepSeek-V3.2.html).

### 2.4 V4-Pro, V4-Flash, V4.1-Flash (2026)

DeepSeek's own V4-Pro-0813 card gives two launch lines (vendor):
- vLLM on one 4x GB300 node: `vllm serve deepseek-ai/DeepSeek-V4-Pro-0813 --trust-remote-code --kv-cache-dtype fp8 --block-size 256 --data-parallel-size 4 --enable-expert-parallel --moe-backend deep_gemm_mega_moe --attention-config '{"use_fp4_indexer_cache": true}' --speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"greedy"}'`
- SGLang: `sglang serve --model-path deepseek-ai/DeepSeek-V4-Pro-0813 --tp 4 --moe-runner-backend flashinfer_mxfp4 --speculative-algorithm DSPARK --mem-fraction-static 0.90 --chunked-prefill-size 4096 --swa-full-tokens-ratio 0.1`
(https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813)

vLLM recipe for V4-Pro (vendor docs, https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro), as summarised from the page:
- Default 8x H200, FP8 KV cache, block size 256, `--max-model-len 800000` (capped below 1M for KV headroom), DSpark with 7 speculative tokens; DP+EP recommended on H200.
- Also 8x B300 (DP8+EP), 8x B200 presets, GB200 NVL4 across 2 trays (the checkpoint "does not fit on one 4-GPU tray"), 8x MI355X (ROCm + AITER).
- No tokens/s, TTFT or ITL numbers on the page; it cites a synthetic DSpark acceptance length of 3.77 for throughput sweeps only.

SGLang cookbook for V4 (vendor docs, https://docs.sglang.io/cookbook/autoregressive/DeepSeek/DeepSeek-V4), as summarised from the page prose:
- Flash: TP=4 on B200, B300, GB200, GB300 or H200; TP=2 on RTX PRO 6000 (96 GB); TP=8 on H100.
- Pro: TP=8 on B200/B300; TP=4 on GB300; H200 FP4 on one 8-GPU node; H200 FP8 needs 2 nodes (TP=16); H100 needs 2 nodes (TP=16). Pro "does not fit on 8x 96 GB RTX PRO 6000".
- DGX Spark: 2x GB10 over ConnectX-7 at TP=2; "a single 128 GB GB10 cannot hold any of these checkpoints" (refers to the official checkpoints, not low-bit GGUFs).
- Pro Official low-latency recipe reaches DSpark accept length 4.678. Throughput tables are rendered by a widget, so the numbers were not readable by my tools (gap).

V4.1-Flash: DeepSeek says it "will work closely with the open-source community on V4.1-Flash inference support" and invites anyone "planning a large-scale deployment with 2,000 GPUs + a storage cluster" to contact them (https://api-docs.deepseek.com/news/news260910). The reference code converts weights for `MP=8` tensor-parallel ranks (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/inference/README.md). Production recipes for V4.1 in vLLM/SGLang: not checked in detail (gap).

---

## 3. Running on CPUs and consumer machines

### 3.1 The physics (derived)

Decode is memory-bandwidth bound. Each new token must read the active weights once. For V3/R1: 37B active params. At Q4_K_M (about 4.8 bits) that is about 37e9 x 0.6 B = 22 GB read per token. On an M3 Ultra ("over 800GB/s") the ceiling is about 800 / 22 = 36 tok/s; real results are about half of that (below). On a dual-socket server with 8 channels of DDR5-4800 per socket, the theoretical bandwidth is about 2 x 8 x 38.4 GB/s = 614 GB/s; NUMA effects cut what one process can use. The total RAM need comes from total params (all experts must be resident or streamed), the speed from active params.

### 3.2 llama.cpp and GGUF

- MLA support for DeepSeek V2/V3 landed in llama.cpp PR #12801, "DeepSeek V2/V3 MLA implementation", by jukofyork, merged 2025-04-15 (https://github.com/ggml-org/llama.cpp/pull/12801). A follow-up to keep only the K cache (#13529, "47% saving on KV-cache size") was closed without merge (https://github.com/ggml-org/llama.cpp/pull/13529).
- Before MLA support, llama.cpp stored a full per-head cache. Independent measurement on M3 Ultra 512 GB with V3 Q4_K_M: KV cache about 40,260 MiB at 8K, 79,300 MiB at 16K, 157,380 MiB at 32K context (https://dev.to/someoddcodeguy/m3-ultra-mac-studio-512gb-prompt-and-write-speeds-for-deepseek-v3-0-671b-gguf-q4km-for-those-47hi). Compare MLA BF16: 70,272 B x 32,768 = 2.3 GB at 32K. This gap is a great story for the MLA chapter.
- DeepSeek V4 support: PR #24162 "DeepSeek V4" by am17an, merged 2026-06-29 (https://github.com/ggml-org/llama.cpp/pull/24162); fix for quantized KV cache #25202 merged 2026-07-07 (https://github.com/ggml-org/llama.cpp/pull/25202).
- V4.1-Flash: Unsloth's guide says to build from PR #30817 (https://unsloth.ai/docs/models/deepseek-v4.1-flash). That PR, "model : add DeepSeek-V4.1-Flash (deepseek41)" by danielhanchen, is closed and not merged as of 2026-11-10 (https://github.com/ggml-org/llama.cpp/pull/30817). Mainline status of V4.1 in llama.cpp: UNVERIFIED.

Unsloth's 1.58-bit R1 claims (vendor, January 2025, via https://unsloth.ai/blog/deepseekr1-dynamic as quoted in search results): fits in 160 GB VRAM on 2x H100 80GB for "around 140 tokens per second for throughput and 14 tokens/s for single user inference". Unsloth staff on Reddit said a single 24 GB RTX 4090 gives "at least 2-3 tokens/s" with offload (https://lr.psf.lt/r/selfhosted/comments/1ic8zil/yes_you_can_run_deepseekr1_locally_on_your_device/m9rjejl , UNVERIFIED as I read it through a mirror).

V4.1-Flash-1017 with llama.cpp (vendor, https://unsloth.ai/docs/models/deepseek-v4.1-flash): RAM/VRAM need with Engram on SSD is 128 to 145 GB for 1-bit, 176 to 205 GB for 2-bit, 256 to 290 GB for 3-bit, 305 GB for 4-bit; flags `--load-mode mmap --lazy-mode on` to keep Engram on SSD. The page gives no measured tokens/s.

### 3.3 KTransformers (CPU + GPU hybrid)

Idea: keep attention (MLA) and shared parts on one GPU, run the routed experts on CPU with AMX/AVX-512 kernels.

V3/R1 671B Q4_K_M (vendor claims, https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepseekR1_V3_tutorial.md):
- Hardware: Intel Xeon Gold 6454S, 32 cores per socket, 2 sockets; RTX 4090D 24 GB (later 4090); DDR5-4800, 8 channels per socket, 1 TB.
- Memory: single socket "382G DRAM, at least 14GB VRAM"; dual socket "1T DRAM" (the dual-socket mode copies weights to both NUMA nodes).
- V0.2, 500-token prompt: dual socket, 8 experts: 82.94 tok/s prefill, 12.21 tok/s decode; with 6 experts: 97.32 / 13.69. llama.cpp on the same 2x32 cores: 10.31 / 4.51. Claimed "3.03x" decode and "9.44x" prefill speedup.
- V0.3 preview with AMX: prefill up to 255.26 tok/s (8 experts) and 286.55 tok/s (6 experts), "up to 27.79x" versus llama.cpp.
- V0.2.1, 6 experts: decode 16.8 tok/s at a tiny prompt, 13.9 tok/s at 8K prompt.
- Caveat: "6 experts" means selectively using fewer experts than the model's 8, which changes the model's computation. Present it as an approximation trade-off.

V4-Flash (vendor claims, https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash.md): 1x RTX 5090 (32 GB), x86 CPU with AVX2/FMA, RAM 200 GB or more, about 340 GB disk; "Decode throughput: 20+ tok/s on a single RTX 5090"; MTP as EAGLE gives "~1.2x" single-request decode (26.5 to 32.74 tok/s on 8x RTX 5090).

V4-Flash on one Ascend die (vendor claims, https://github.com/kvcache-ai/ktransformers/blob/main/doc/en/DeepSeek-V4-Flash_tutorial_for_Ascend_NPU.md): Atlas A3 single die (61.3 GB HBM), 40 cores, 229 GB host RAM; decode 18.32 to 20.48 tok/s across 118 to 31,540-token prompts; prefill about 15.6 to 20.4 s; "Decode is host-memory-bandwidth bound".

### 3.4 Apple Silicon

R1 / V3 671B on M3 Ultra 512 GB:
- Apple's own claim: Mac Studio with M3 Ultra can run "LLMs with over 600 billion parameters entirely in memory" (https://www.apple.com/newsroom/2025/03/apple-unveils-new-mac-studio-the-most-powerful-mac-ever/).
- Dave Lee (Dave2D) review, R1 4-bit: "approximately 17-18 tokens per second" (independent reviewer, reported by MacRumors https://www.macrumors.com/2025/03/17/apples-m3-ultra-runs-deepseek-r1-efficiently/). TweakTown headline says 448 GB of RAM used and under 200 W (https://www.tweaktown.com/news/103891/apples-new-m3-ultra-runs-deepseek-r1-with-671b-parameters-uses-448gb-of-ram-only-200w-power/index.html ; page blocks automated fetches, so the 448 GB and 200 W figures are UNVERIFIED by me).
- Independent llama.cpp-family test, V3-0324 Q4_K_M with flash attention, M3 Ultra 512 GB: 4.8K prompt at 64.02 tok/s prompt processing, then 10.89 tok/s generation; 12.7K prompt at 58.82 tok/s processing, then 6.67 tok/s generation. Earlier (March 2025) build: prompt processing only about 9 tok/s, generation about 6.2 tok/s at 8K context (https://dev.to/someoddcodeguy/m3-ultra-mac-studio-512gb-prompt-and-write-speeds-for-deepseek-v3-0-671b-gguf-q4km-for-those-47hi). Lesson: short-prompt demos hide very slow prefill on Macs.

V4-Flash and V4.1-Flash on Macs (antirez DwarfStar `ds4`, Metal engine; numbers are the project's own measurements, https://github.com/antirez/ds4/blob/main/docs/PERFORMANCE.md):

| Machine | Context | Prefill | Generation |
|---|---|---|---|
| M5 Max, 128 GB, V4-Flash Q2 | 2,048 | 790.18 t/s | 39.35 t/s |
| M5 Max, 128 GB, V4-Flash Q2 | 65,536 | 398.50 t/s | 27.64 t/s |
| DGX Spark, 128 GB, V4-Flash Q2 | 2,048 | 825.76 t/s | 18.05 t/s |
| DGX Spark, 128 GB, V4-Flash Q2 | 65,536 | 822.98 t/s | 13.84 t/s |

- ds4 README: eight L40S cards reached "about 126 t/s aggregate generation with 16 sessions" on V4-Flash; two 128 GB Macs over RDMA can run 4-bit V4-Flash with tensor parallelism (https://github.com/antirez/ds4).
- V4.1-Flash on ds4: Q2 runs with SSD streaming on one 128 GB Mac or Spark; "Q4 needs SSD streaming or a 512 GB Mac" (https://github.com/antirez/ds4). Downloads: Q2 340.60 GiB total (151.77 GiB main weights), Q4 482.98 GiB (294.15 GiB main) (https://huggingface.co/antirez/deepseek-v4.1-flash-gguf). No tokens/s published for V4.1 there.
- A search summary attributed "36.86 t/s" generation on M3 Ultra 512 GB at q2 to the ds4 README. I could not find it in the current README: UNVERIFIED.

### 3.5 Ollama for the distills (and the rest)

- Distill tags and sizes: 1.5b 1.1 GB, 7b 4.7 GB, 8b 5.2 GB (now R1-0528-Qwen3-8B), 14b 9.0 GB, 32b 20 GB, 70b 43 GB, all 128K context; 671b 404 GB at 160K (https://ollama.com/library/deepseek-r1/tags).
- `deepseek-v3.1` 671b q4_K_M 404 GB, q8_0 713 GB, fp16 1.3 TB (https://ollama.com/library/deepseek-v3.1/tags).
- `deepseek-v4-pro` and `deepseek-v4.1-flash` exist on Ollama only as `:cloud` tags (hosted, not local) as of 2026-11-10 (https://ollama.com/library/deepseek-v4-pro/tags , https://ollama.com/library/deepseek-v4.1-flash/tags).
- Rule of thumb (derived): the download size plus KV cache must fit in RAM/VRAM. A 16 GB laptop runs 7B/8B at Q4 comfortably; a 32 GB machine fits 14B, and 32B at Q4 (20 GB) is tight; 70B needs 48 GB or more. I found no single authoritative tok/s table for the distills on laptops: give measured numbers only after the writer runs them.

---

## 4. DeepSeek's own inference system

### 4.1 Day 6 overview (Feb 2025, V3/R1 on H800)

Source: https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md (all vendor claims).

Design:
- Goals: "higher throughput and lower latency" via cross-node Expert Parallelism. 8 of 256 routed experts are active per layer, so a large total batch is needed for a decent per-expert batch.
- Prefill/decode disaggregation:
  - Prefill: routed experts EP32, MLA and shared expert DP32. One deployment unit = 4 nodes (32 GPUs), 32 redundant routed experts, each GPU holds 9 routed experts + 1 shared expert.
  - Decode: routed experts EP144, MLA and shared expert DP144. One unit = 18 nodes (144 GPUs), 32 redundant routed experts, each GPU holds 2 routed experts + 1 shared expert.
  - Check (derived): prefill 256 + 32 = 288 = 32 x 9; decode 288 = 144 x 2.
- Overlap: dual-batch overlap in prefill (two microbatches alternate, one's communication hides behind the other's compute). In decode the attention layer is split into two steps, giving a 5-stage pipeline.
- Load balancers: prefill balances core-attention compute and dispatch send load; decode balances KV cache usage and request counts; the EP balancer minimises the maximum dispatch receive load.
- Precision: matmuls and dispatch in FP8 (as in training); core MLA and combine in BF16.
- Internal engine: built on "an early fork of vLLM from over a year ago", which DeepSeek chose not to open-source; it contributes pieces to vLLM/SGLang instead (https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md).

Production statistics, 24 hours from 2025-02-27 12:00 to 2025-02-28 12:00 (UTC+8), V3 and R1 combined:
- H800 nodes of 8 GPUs. Peak 278 nodes. Average 226.75 nodes.
- Cost assumption $2 per H800 GPU-hour. Daily cost = 226.75 x 8 x 24 x $2 = $87,072.
- Input tokens 608B, of which 342B (56.3%) hit the on-disk KV cache. Output tokens 168B.
- Average output speed 20 to 22 tok/s per request. Average KV cache length per output token 4,989.
- Per H800 node: about 73.7k input tok/s in prefill (including cache hits); about 14.8k output tok/s in decode.
- Derived: 14.8k / 8 = 1,850 output tok/s per H800 in decode. At 20 to 22 tok/s per user, that is roughly 670 to 740 concurrent streams per decode node.

The 545% "theoretical" margin:
- The statistics "include all user requests from web, APP, and API".
- If every token were billed at R1 prices ($0.14 per 1M input on cache hit, $0.55 per 1M input on cache miss, $2.19 per 1M output), daily revenue would be $562,027. Margin = (562,027 - 87,072) / 87,072 = 5.45, i.e. 545%.
- DeepSeek's own caveats: V3 is priced well below R1; only part of the service is monetised (web and app are free); night-time off-peak discounts apply automatically. So "actual revenue is substantially lower".
- Additional caveats for writers (my reasoning, not from DeepSeek): the $2/GPU-hour is an assumed rental rate, not DeepSeek's real cost; it excludes training, research, staff, storage (3FS), networking and idle capacity planning; the 56.3% cache-hit share inflates the cheap input bucket.

Cross-check (independent re-implementation): SGLang on 96 H100 reported prefill within about 6% and decode 6.6% below DeepSeek's profile-data numbers under simulated perfect balance; its table lists "DeepSeek blog" 32,206 tok/s per node prefill at 4K input versus "DeepSeek profile" 62,713 (https://lmsys.org/blog/2025-05-05-large-scale-ep/).

### 4.2 V4 and V4.1 inference changes (vendor, tech reports)

- V4: "Our inference framework largely inherits from that of DeepSeek-V3, with some differences in KV Cache management"; heterogeneous KV cache layout; on-disk storage of compressed CSA/HCA entries for shared-prefix reuse, with three strategies for the larger sliding-window entries (V4 report, https://arxiv.org/abs/2606.19348).
- V4: fine-grained EP mega-kernel ("MegaMoE", open-sourced in DeepGEMM PR #304) "validated ... on both NVIDIA GPUs and HUAWEI Ascend NPUs", 1.50 to 1.73x speedup for general inference and up to 1.96x for latency-sensitive cases versus non-fused baselines (same report).
- V4.1: Encoder-Prefill-Decode (EPD) disaggregation; Causal Encoder-Decoder activates 8B params in prefill and 16B in decode; KV cache needs "1/4 the HBM" and "1/8 the SSD storage" of the previous generation (https://api-docs.deepseek.com/news/news260910 ; V4.1 report).

### 4.3 The open-sourced pieces

FlashMLA (https://github.com/deepseek-ai/FlashMLA):
- Day 1 (Feb 2025): "3000 GB/s memory-bound | BF16 580 TFLOPS compute-bound on H800" (https://github.com/deepseek-ai/open-infra-index).
- 2025-04-22 update: up to 660 TFLOPS on H800 SXM5 (5% to 15% faster compute-bound).
- 2025-09-29 (V3.2 sparse kernels): up to 640 TFLOPS prefill and 410 TFLOPS decode on H800 (FP8 KV cache, BF16 matmul); later README also lists 1,450 TFLOPS sparse prefill on B200.
- 2026-09-10: V4.1 kernels. Current README (B200, CUDA 13.3): sparse prefill up to 1,350 TFLOPS, sparse decode up to 1,024 TFLOPS, fused norm-RoPE-attn kernel up to 1,460 TFLOPS in prefill.
- 2026-09-30: Ascend 950 kernels: 410 TFLOPS prefill (95% of hardware peak), 360 TFLOPS decode (83%). Breaking change: the main branch dropped Hopper and older models (V3, V3.2, V4.0); use commit ba89a34 for those (https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7).
- Ports by other vendors are linked from the README: MetaX, Moore Threads, Hygon DCU, Iluvatar Corex, AMD Instinct.

DeepEP (https://github.com/deepseek-ai/DeepEP). Original V1 numbers (H800, NVLink about 160 GB/s, CX7 400 Gb/s InfiniBand about 50 GB/s; README at commit 9fe9021, https://github.com/deepseek-ai/DeepEP/tree/9fe9021f29c9083cd1808ab36b740208524d9f63):

Normal kernels (4,096 tokens per batch, hidden 7168, top-8, FP8 dispatch, BF16 combine):

| Type | EP | Dispatch bandwidth | Combine bandwidth |
|---|---|---|---|
| Intranode | 8 | 153 GB/s (NVLink) | 158 GB/s (NVLink) |
| Internode | 16 | 43 GB/s (RDMA) | 43 GB/s (RDMA) |
| Internode | 32 | 58 GB/s (RDMA) | 57 GB/s (RDMA) |
| Internode | 64 | 51 GB/s (RDMA) | 50 GB/s (RDMA) |

Low-latency kernels (128 tokens per batch, decode setting):

| EP | Dispatch latency | Dispatch RDMA BW | Combine latency | Combine RDMA BW |
|---|---|---|---|---|
| 8 | 163 us | 46 GB/s | 318 us | 46 GB/s |
| 16 | 173 us | 43 GB/s | 329 us | 44 GB/s |
| 32 | 182 us | 41 GB/s | 350 us | 41 GB/s |
| 64 | 186 us | 40 GB/s | 353 us | 41 GB/s |
| 128 | 192 us | 39 GB/s | 369 us | 39 GB/s |
| 256 | 194 us | 39 GB/s | 360 us | 40 GB/s |

Teaching point: latency barely grows from EP8 to EP256. That is why DeepSeek can afford EP144 in decode.

Later DeepEP status (current README, no dates given): V2 refactor with an NCCL "Gin" backend; V2.5 removes V1 and the NVSHMEM dependency; Ascend 950 version in DeepEP-Ascend; experimental branches for AMD (Mori-EP), hybrid-EP with NVFP4. The current README has no performance tables.

DeepGEMM (https://github.com/deepseek-ai/DeepGEMM):
- Day 3: "Up to 1350+ FP8 TFLOPS on Hopper GPUs", core logic about 300 lines, fully JIT (https://github.com/deepseek-ai/open-infra-index).
- Original table, H800 SXM5, NVCC 12.8, versus DeepSeek's internal CUTLASS 3.6 baseline: decode-shaped M=64 GEMMs 206 to 336 TFLOPS (1.4x to 2.7x speedup); prefill-shaped M=4096 GEMMs 590 to 1,358 TFLOPS (1.0x to 1.2x) (README at commit 4d4f234, https://github.com/deepseek-ai/DeepGEMM/tree/4d4f2342febb829e88f9cc7cbf4b955823b395fc).
- 2025-04-18: "up to 1550 TFLOPS on H800". 2025-07-20: SM90 and SM100 support. 2025-09-28: lightning-indexer scoring kernels for V3.2. 2026-04-16: Mega MoE, FP8xFP4 GEMM, FP4 indexer. 2026-09-30: DeepGEMM Ascend. Formats: FP8, FP4, BF16; SM100 uses packed UE8M0 scale factors (current README).

EPLB (https://github.com/deepseek-ai/EPLB): redundant-experts strategy; two policies. Hierarchical (pack expert groups onto nodes, then replicate within a node, used for prefill with smaller EP) and global (replicate globally, used for decode with larger EP). Load prediction is out of scope; the README suggests a moving average of history.

3FS (https://github.com/deepseek-ai/3FS): 6.6 TiB/s aggregate read throughput on a 180-node cluster; "40+ GiB/s peak throughput per client node for KVCache lookup"; used for the on-disk KV cache (https://github.com/deepseek-ai/open-infra-index).

---

## 5. Domestic (Chinese) hardware and formats

- V3 (Dec 2024): Huawei Ascend via MindIE, BF16 at first (https://github.com/deepseek-ai/DeepSeek-V3 , section 6.8).
- V3.1 (Aug 2025) and UE8M0: the model card says V3.1 "is trained using the UE8M0 FP8 scale data format on both model weights and activations to ensure compatibility with microscaling data formats" (https://huggingface.co/deepseek-ai/DeepSeek-V3.1). The config shows `"scale_fmt": "ue8m0"` (https://huggingface.co/deepseek-ai/DeepSeek-V3.1/blob/main/config.json). UE8M0 means an unsigned 8-bit exponent-only scale, so each scale is a power of two. DeepSeek's WeChat post said UE8M0 FP8 is designed for "soon-to-be-released next-generation domestic chips"; no chip was named (Reuters-based report: https://www.malaymail.com/news/tech-gadgets/2025/08/21/chinas-deepseek-unveils-faster-ai-model-with-domestic-chip-support/188478 ; the WeChat original was not checked: UNVERIFIED primary).
- V3.2-Exp (Sep 2025): DeepSeek released "Key GPU kernels in TileLang & CUDA (use TileLang for rapid research prototyping!)" (https://api-docs.deepseek.com/news/news250929) and pointed to TileLang examples (https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32). SGLang shipped Ascend A2/A3 images on day 0 (https://github.com/deepseek-ai/DeepSeek-V3.2-Exp). Huawei's day-0 support claims (TTFT under 2 s and TPOT under 30 ms at 128K) and Cambricon's vLLM-MLU update come from press reports (https://www.tomshardware.com/tech-industry/deepseek-new-model-supports-huawei-cann): UNVERIFIED from Huawei or Cambricon primary sources.
- V4 (Apr 2026): MegaMoE EP kernel validated on NVIDIA GPUs and Huawei Ascend NPUs (V4 report, https://arxiv.org/abs/2606.19348). The V4 report and card do not say V4 was trained on Ascend. Press claims that V4 was trained and deployed "entirely on Huawei Ascend 950PR" are UNVERIFIED (search summary of https://www.huaweicentral.com/deepseek-launches-new-v4-ai-models-running-on-huawei-chips/amp/). V4 uses TileLang for most fused kernels (V4 report section 3.2).
- V4.1 (Sep 2026): FlashMLA describes itself as "powering inference of the DeepSeek-V4.1 model on NVIDIA GPUs and Huawei Ascend NPUs", with Ascend 950 kernels at 410 / 360 TFLOPS; DeepEP and DeepGEMM both got Ascend versions (https://github.com/deepseek-ai/FlashMLA , https://github.com/deepseek-ai/DeepEP , https://github.com/deepseek-ai/DeepGEMM). This is the clearest primary evidence that DeepSeek itself now targets Ascend in production. Whether DeepSeek's public API runs on Ascend: UNVERIFIED.
- Community: KTransformers runs V4-Flash on a single Ascend die with CPU expert offload (section 3.3). Huawei CloudMatrix384 serves R1 at scale in INT8 (section 2.2).

---

## 6. Which setup for which goal

| Goal | Setup | Memory needed | Expected speed (sourced) | Cost (sourced only) |
|---|---|---|---|---|
| Just try current DeepSeek | DeepSeek API, `deepseek-flash` (V4.1-Flash-1017) | none | n/a | Per 1M tokens, peak: $0.006 cache-hit input, $0.30 cache-miss input, $1.20 output; off-peak half (https://api-docs.deepseek.com/quick_start/pricing) |
| Try the big model via API | `deepseek-v4-pro` | none | n/a | Peak: $0.044 / $1.32 / $3.96 per 1M (cache-hit in / miss in / out) (same page) |
| Learn reasoning on a laptop | R1 distill 7B/8B via Ollama | about 5 GB + KV | not sourced; measure it | free software |
| Better laptop reasoning | R1 distill 14B or 32B Q4 | 9 GB / 20 GB + KV | not sourced | free software |
| Run full V3/R1 671B at home, cheapest | Unsloth 1.58-bit GGUF, llama.cpp, one 24 GB GPU + lots of RAM | 140 GB file (R1); RAM+VRAM ideally at least that | "at least 2-3 tokens/s" on one 4090 (vendor staff) | n/a |
| Full V3/R1 at home, usable speed | KTransformers, 1 GPU 24 GB + dual Xeon with AMX | 382 GB DRAM (1 socket) or 1 TB (2 sockets), 14 GB VRAM | 13.7 to 16.8 tok/s decode, 97 to 286 tok/s prefill (vendor) | n/a |
| Full V3/R1 on a Mac | M3 Ultra 512 GB, 4-bit | about 404 GB file | about 17-18 tok/s short prompts (Dave2D); 6.7 to 10.9 tok/s at 5K to 13K prompts (independent llama.cpp test) | n/a |
| Modern model at home | V4-Flash Q2 with ds4 on a 128 GB Mac or DGX Spark | 96.8 GB (Unsloth UD-Q2_K_XL) | 39 tok/s (M5 Max) / 18 tok/s (DGX Spark) at 2K context (ds4 project) | n/a |
| Modern model, GPU + CPU | V4-Flash with KTransformers, 1x RTX 5090 | 200 GB RAM or more, 32 GB VRAM | "20+ tok/s" decode (vendor) | n/a |
| V4.1-Flash at home | Unsloth GGUF + Engram on SSD, or ds4 | 122 to 306 GB RAM/VRAM plus 233 to 508 GB disk | no tok/s published | n/a |
| Serve V3/R1 in production, one box | 8x H200 or 8x MI300X, FP8, SGLang or vLLM, DP-attention + EP, MTP | 688.6 GB weights | MI300X TP8: 34 tok/s at concurrency 1 (EAGLE), 464 tok/s total at concurrency 100 (SGLang docs) | n/a |
| Serve V3/R1 at scale | PD disaggregation + wide EP, tens of nodes | n/a | DeepSeek: 73.7k in / 14.8k out tok/s per H800 node; SGLang: 52.3k / 22.3k per H100 node; vLLM: 2.2k tok/s per H200 | DeepSeek's assumed $2/GPU-hour gives $87,072/day for 226.75 nodes; SGLang claims $0.20 per 1M output tokens |
| Serve V4-Pro | 8x H200 (FP4) / 8x B200 / 4x GB300 / 2 nodes H100 | 865 to 893 GB weights | no tok/s on vLLM recipe page | n/a |
| Serve on Chinese hardware | Ascend A3 (vLLM-Ascend, SGLang NPU images), CloudMatrix384 | V3.2 W8A8: 1 A3 node (16 NPUs) | CloudMatrix: 1,943 tok/s per NPU decode (Huawei paper) | n/a |

---

## 7. Gaps and things I could not verify

- Throughput tables for V4 in the SGLang cookbook are rendered by a widget; my tools could not read them.
- No published tok/s for V4.1-Flash on consumer hardware (Unsloth and antirez pages give none).
- Unsloth's January 2025 blog, TweakTown and hardware-corner return HTTP 403 to automated fetching; their numbers are quoted through search results or secondary pages.
- Exact V3.2 lightning-indexer cache bytes per token in production engines.
- Whether DeepSeek's production API runs on Ascend, H800, H20 or a mix in 2026.
- Hardware prices (Mac Studio, H200 nodes) are deliberately not given; no primary price source was checked.

URL check (2026-11-10, `curl -s -o /dev/null -w '%{http_code}' -L`): 87 of the 90 cited URLs returned 200. Three returned 403 to the automated check because the sites block bots: https://unsloth.ai/blog/deepseekr1-dynamic , https://www.tweaktown.com/news/103891/apples-new-m3-ultra-runs-deepseek-r1-with-671b-parameters-uses-448gb-of-ram-only-200w-power/index.html and the Malay Mail V3.1 report. They are real pages, but their content could not be read again for this dossier. Open them in a browser before quoting them. The `https://huggingface.co/api/models/<repo>?blobs=true` pattern in the header is a template, not a link.
