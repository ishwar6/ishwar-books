# Research dossier 03: How DeepSeek models are trained

Pre-training, training infrastructure, and reinforcement learning (RL). Compiled 2026-11-10 for "DeepSeek, From the Ground Up".

## How to read this dossier

- Every number has a source. Papers are cited as `arXiv id, p.N`, where N is the **1-based PDF page** (not the printed page number). All PDFs are in `~/.cache/papers/`.
- Quotes in "Screenshot passages" blocks were extracted with pymupdf (`~/.venvs/attn/bin/python`, `import pymupdf`). Line-break hyphens were joined. Math symbols are shown as pymupdf extracts them, so some spaces are missing inside formulas.
- **UNVERIFIED** = I could not confirm it in a primary source.
- **Derived** = my own arithmetic from primary numbers. Label it as such in any article.
- **Third-party** = a claim by someone outside DeepSeek. Never present it as a DeepSeek number.

### Source files

| Short name | Source | Local file | Pages |
|---|---|---|---|
| LLM | DeepSeek LLM (67B), arXiv 2401.02954 | `2401.02954.pdf` | 48 |
| V2 | DeepSeek-V2, arXiv 2405.04434 | `2405.04434.pdf` | 52 |
| V3 | DeepSeek-V3 Technical Report, arXiv 2412.19437 | `2412.19437.pdf` | 53 |
| Math | DeepSeekMath (GRPO origin), arXiv 2402.03300 | `2402.03300.pdf` | 30 |
| R1v1 | DeepSeek-R1 arXiv v1 (Jan 2025) | `2501.12948v1.pdf` | 22 |
| R1v2 | DeepSeek-R1 arXiv v2 (4 Jan 2026), main text + supplementary, matches Nature content | `2501.12948.pdf` | 86 |
| Nature | Guo et al., "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", Nature 645, 633-638 (2025), doi:10.1038/s41586-025-09422-z | `nature-r1.pdf` | 19 |
| NatureSI | Nature R1 Supplementary Information (41586_2025_9422_MOESM1_ESM.pdf) | `nature-r1-si.pdf` | ~80 |
| FireFlyer | Fire-Flyer AI-HPC, arXiv 2408.14158 (SC24) | `2408.14158.pdf` | 18 |
| ISCA | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware, arXiv 2505.09343 (ISCA 2025) | `2505.09343.pdf` | 15 |
| Prover2 | DeepSeek-Prover-V2, arXiv 2504.21801 (v2, 18 Jul 2025) | `2504.21801.pdf` | 39 |
| MathV2 | DeepSeekMath-V2, arXiv 2511.22570 | `2511.22570.pdf` | 19 |
| V3.2 | DeepSeek-V3.2, arXiv 2512.02556 | `2512.02556.pdf` | 23 |
| V4 | DeepSeek-V4 (preview), arXiv 2606.19348 (26 Apr 2026) | `2606.19348.pdf` | 58 |
| V4.1 | DeepSeek-V4.1-Flash, arXiv 2609.19969 (17 Sep 2026); identical to the PDF in the HF repo | `2609.19969.pdf` (also `deepseek-v4.1-flash-tech-report.pdf`) | 51 |

Nature URLs: article https://www.nature.com/articles/s41586-025-09422-z ; SI https://static-content.springer.com/esm/art%3A10.1038%2Fs41586-025-09422-z/MediaObjects/41586_2025_9422_MOESM1_ESM.pdf . Nature dates (Nature PDF p.1 and article info): Received 14 February 2025, Accepted 17 July 2025, Published online 17 September 2025, issue of 18 September 2025 (Vol 645, issue 8081). Crossref metadata confirms volume 645, pages 633-638.

Model cards (Hugging Face, `huggingface.co/deepseek-ai/<name>`): DeepSeek-V3.1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2(-Speciale), DeepSeek-R1-0528, DeepSeek-V4-Pro-0813, DeepSeek-V4-Flash-0731, DeepSeek-V4.1-Flash, DeepSeek-V4.1-Flash-1017. Release dates below are the HF repo `createdAt` timestamps from the HF API.

---

## 1. Pre-training, model by model

### 1.1 Summary table

| Model | Params (total / active) | Pre-training tokens | Tokenizer | Context and extension | Peak LR and schedule | Batch schedule | Hardware | GPU hours | Stated cost |
|---|---|---|---|---|---|---|---|---|---|
| DeepSeek LLM 67B (Jan 2024) | 67B dense | 2.0T (LLM p.5 Table 2; "2 trillion tokens", p.3) | BBPE, 100,000 conventional tokens + 15 special = 100,015; vocab set to 102,400 (p.5) | 4096 (p.5 Table 2) | 3.2e-4; multi-step: 2000 warmup, to 31.6% at 80% of tokens, to 10% at 90% (p.5) | 4608 sequences (p.5 Table 2) | Not stated in the paper (HAI-LLM framework, p.6). UNVERIFIED which cluster | Not stated. V2 paper says 67B needs 300.6K H800 GPU hours per trillion tokens on the H800 cluster (V2 p.16) | Not stated |
| DeepSeek-V2 (May 2024) | 236B / 21B (V2 p.12) | 8.1T; "Chinese tokens are approximately 12% more than English ones" (V2 p.12) | Same BBPE tokenizer as 67B, vocab 100K (V2 p.12) | 4K pre-train; YaRN to 128K: 1000 steps at 32K, batch 576 (V2 p.13) | 2.4e-4; warmup 2K steps, x0.316 at ~60% and again at ~90% of tokens (V2 p.12) | 2304 to 9216 over first 225B tokens (V2 p.12) | H800 cluster, 8 GPUs per node, NVLink + IB (V2 p.13). GPU count not stated | 172.8K H800 GPU hours per trillion tokens (V2 p.16). Derived total: 8.1 x 172.8K = ~1.40M | "saves 42.5% of training costs" vs 67B (V2 p.1, p.16). No dollar figure |
| DeepSeek-V3 (Dec 2024) | 671B / 37B (V3 p.22) | 14.8T (V3 p.22) | Byte-level BPE, 128K vocab, new pretokenizer (V3 p.22) | 4K pre-train; YaRN 4K to 32K to 128K, 2 phases x 1000 steps (V3 p.23) | 2.2e-4 constant to 10T; cosine to 2.2e-5 over 4.3T; last 500B: 2.2e-5 for 333B then 7.3e-6 for 167B (V3 p.22-23) | 3072 to 15360 over first 469B tokens (V3 p.23) | 2048 H800 (V3 p.11) | 2,664K pre-train + 119K context + 5K post = 2,788K (V3 p.5 Table 1) | $5.576M at $2/GPU-hour (V3 p.5) |
| DeepSeek-V3.1 (Aug 2025) | 671B / 37B (card) | Continued long-context training on the V3 base: 32K phase 630B tokens (10x), 128K phase 209B tokens (3.3x) (V3.1 card) | Same as V3 (structure same as V3, card) | 128K | Not stated | Not stated | Not stated | Not stated | Not stated |
| DeepSeek-V3.2-Exp / V3.2 (Sep / Dec 2025) | 671B / 37B (same as V3.1 architecture plus DSA) | Continued pre-training from V3.1-Terminus: 2.1B tokens indexer warm-up + 943.7B tokens sparse training (V3.2 p.4) | Same | 128K (DSA, top-2048 KV tokens per query) | Warm-up LR 1e-3 (indexer only); sparse stage LR 7.3e-6 (V3.2 p.4) | 16 seqs x 128K (warm-up); 480 seqs x 128K (sparse) (V3.2 p.4) | Not stated (inference cost measured on H800, p.5) | Not stated | "post-training computational budget exceeding 10% of the pre-training cost" (V3.2 p.2) |
| DeepSeek-V4-Flash (Apr 2026) | 284B / 13B (V4 p.24) | 32T (V4 p.5, p.25) | V3 tokenizer + a few special tokens, vocab still 128K (V4 p.24) | 4K, then 16K, 64K, 1M during pre-training (V4 p.25) | 2.7e-4, cosine to 2.7e-5 near the end; Muon + AdamW (V4 p.25) | Increased to 75.5M tokens, then constant (V4 p.25) | NOT DISCLOSED (the paper mentions kernels for NVIDIA GPUs and Huawei Ascend NPUs, p.16, but no training cluster) | NOT DISCLOSED | NOT DISCLOSED |
| DeepSeek-V4-Pro (Apr 2026) | 1.6T / 49B (V4 p.25) | 33T (V4 p.5, p.25) | Same as Flash | Same as Flash; longer dense-attention stage (V4 p.26) | 2.0e-4 peak, 2.0e-5 end (V4 p.25) | Max 94.4M tokens (V4 p.25) | NOT DISCLOSED | NOT DISCLOSED | NOT DISCLOSED |
| DeepSeek-V4.1-Flash (Sep 2026) | 552B backbone; 8B active in prefill, 16B in decode (V4.1 p.22) | 45T multimodal tokens, text-only to multimodal ratio 7:1 (V4.1 p.21-22) | Not restated | Trained from scratch with sparse attention at 64K; extended to 1M at 34T tokens (V4.1 p.22) | 2.6e-4 to 28T; cosine to 2.6e-5 between 28T and 40T; constant 40T-45T (V4.1 p.22) | Fixed 100.6M tokens (V4.1 p.22) | NOT DISCLOSED | NOT DISCLOSED | NOT DISCLOSED |

### 1.2 DeepSeek LLM 67B (arXiv 2401.02954)

- 2 trillion tokens, "primarily in Chinese and English" (p.3).
- Architecture follows LLaMA (Pre-Norm, RMSNorm, SwiGLU, RoPE); 67B uses GQA, 95 layers, d_model 8192 (p.5 Table 2).
- They replaced the cosine schedule with a multi-step schedule "maintaining performance while facilitating continual training" (p.3). The stage split is 80% / 10% / 10% (p.6).
- SFT on ~1.5 million instruction instances, then DPO (p.12, p.1).
- Section 3 studies scaling laws for batch size and learning rate (p.3, p.7).

Screenshot passage (2401.02954, p.5, Section 2.3 Hyperparameters):
> "Specifically, the learning rate of the model reaches its maximum value after 2000 warmup steps, and then decreases to 31.6% of the maximum value after processing 80% of the training tokens. It further reduces to 10% of the maximum value after 90% of the tokens."

### 1.3 DeepSeek-V2 (arXiv 2405.04434)

- 8.1T tokens; Chinese ~12% more than English (p.12).
- Training: HAI-LLM framework, 16-way zero-bubble pipeline parallelism, 8-way expert parallelism, ZeRO-1 DP, no tensor parallelism (p.12-13).
- Token-dropping during training for speed (p.12).
- YaRN: scale s=40, alpha=1, beta=32, target 160K; attention entropy factor sqrt(t) = 0.0707 ln s + 1; one 1000-step phase at 32K, batch 576 (p.13). Note: V2 used a single 32K phase; V3 used two phases (32K then 128K).
- Cost claim (p.16): "for training on each trillion tokens, DeepSeek 67B requires 300.6K GPU hours, while DeepSeek-V2 needs only 172.8K GPU hours, i.e., sparse DeepSeek-V2 can save 42.5% training costs compared with dense DeepSeek 67B."
- SFT: 1.5M instances (1.2M helpfulness, 0.3M safety) (p.16). RL with GRPO, two stages: reasoning alignment, then human preference alignment with helpful + safety + rule rewards (p.17).

### 1.4 DeepSeek-V3 (arXiv 2412.19437): the headline numbers

**Data.** Higher ratio of math and code; wider multilingual coverage; document packing without cross-sample attention masking; FIM (PSM format) at rate 0.1 (p.21-22).

Screenshot passages (2412.19437, Section 4.1):
> p.21-22: "Compared with DeepSeek-V2, we optimize the pre-training corpus by enhancing the ratio of mathematical and programming samples, while expanding multilingual coverage beyond English and Chinese. Also, our data processing pipeline is refined to minimize redundancy while maintaining corpus diversity."
>
> p.22: "Finally, the training corpus for DeepSeek-V3 consists of 14.8T high-quality and diverse tokens in our tokenizer."
>
> p.22: "The tokenizer for DeepSeek-V3 employs Byte-level BPE (Shibata et al., 1999) with an extended vocabulary of 128K tokens."

Tokenizer detail (p.22): the new pretokenizer adds tokens that combine punctuation and line breaks; to avoid token-boundary bias they "randomly split a certain proportion of such combined tokens during training".

R1v2 adds a data disclosure about the V3 base (2501.12948, p.13, Supplementary A.1):
> "For the training data of DeepSeek-V3-Base, we exclusively use plain web pages and e-books, without incorporating any synthetic data. However, we have observed that some web pages contain a significant number of OpenAI-model-generated answers, which may lead the base model to acquire knowledge from other powerful models indirectly. However, we did not intentionally include synthetic data generated by OpenAI during the pre-training cooldown phase; all data used in this phase were naturally occurring and collected through web crawling."

R1v2 also gives the V3 base knowledge cutoff (July 2024) and decontamination: 10-gram filtering; "in the mathematics domain alone, our decontamination process identified and removed approximately six million potential pre-training texts." (2501.12948, p.40)

**Hyper-parameters** (p.22-23): AdamW beta1 0.9, beta2 0.95, weight decay 0.1; init std 0.006; max sequence 4K; gradient clip 1.0; MTP depth 1, MTP loss weight 0.3 for first 10T tokens then 0.1; aux-loss-free bias update speed 0.001 for first 14.3T tokens then 0.0; sequence balance loss alpha 0.0001; each layer's routed experts spread over 64 GPUs on 8 nodes; each token to at most 4 nodes.

Screenshot passages (2412.19437, Section 4.2):
> p.22: "Then, we keep a constant learning rate of 2.2 × 10−4 until the model consumes 10T training tokens. Subsequently, we gradually decay the learning rate to 2.2 × 10−5 in 4.3T tokens,"
>
> p.23: "We employ a batch size scheduling strategy, where the batch size is gradually increased from 3072 to 15360 in the training of the first 469B tokens, and then keeps 15360 in the remaining training."

**Context extension** (p.23, Section 4.3): YaRN applied only to the decoupled shared key; s=40, alpha=1, beta=32, sqrt(t) = 0.1 ln s + 1. Phase 1: 32K, batch 1920. Phase 2: 128K, batch 480. LR 7.3e-6 in both.
> p.23: "After the pre-training stage, we apply YaRN (Peng et al., 2023a) for context extension and perform two additional training phases, each comprising 1000 steps, to progressively expand the context window from 4K to 32K and then to 128K."

Derived: phase 1 = 1000 x 1920 x 32,768 = ~62.9B tokens; phase 2 = 1000 x 480 x 131,072 = ~62.9B tokens. This matches the V3.1 card's "10-fold to 630B" and "3.3x to 209B" (see 1.5).

**Cost table (the passage to screenshot).** 2412.19437, p.5, Table 1 and the paragraph above it (Section 1, Introduction):

| Training Costs | Pre-Training | Context Extension | Post-Training | Total |
|---|---|---|---|---|
| in H800 GPU Hours | 2664K | 119K | 5K | 2788K |
| in USD | $5.328M | $0.238M | $0.01M | $5.576M |

> p.5: "Table 1 | Training costs of DeepSeek-V3, assuming the rental price of H800 is $2 per GPU hour."
>
> p.5: "During the pre-training stage, training DeepSeek-V3 on each trillion tokens requires only 180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs."
>
> p.5: "Consequently, our pretraining stage is completed in less than two months and costs 2664K GPU hours. Combined with 119K GPU hours for the context length extension and 5K GPU hours for post-training, DeepSeek-V3 costs only 2.788M GPU hours for its full training. Assuming the rental price of the H800 GPU is $2 per GPU hour, our total training costs amount to only $5.576M. Note that the aforementioned costs include only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data."

The same caveat-free figure is restated in the Conclusion: "It requires only 2.788M H800 GPU hours" (p.35).

Two things writers often miss:
1. The $5.576M is **rental-equivalent** at an assumed $2/GPU-hour, not money spent. DeepSeek owns its GPUs.
2. The "5K GPU hours for post-training" is V3's own SFT + RL. It does **not** include the internal R1 model used to generate V3's reasoning SFT data (V3 p.28-29), nor R1's own cost (see 4.6).

**Compute check (derived, label it as such).** 6 x N_active x D = 6 x 37e9 x 14.8e12 = ~3.3e24 FLOPs. The ISCA paper gives V3 at "approximately 250 GFLOPS per token" for training at sequence length 4096 (2505.09343, p.4, Table 2); 250e9 x 14.8e12 = ~3.7e24 FLOPs. Epoch AI's database entry for V3 was not checked: UNVERIFIED.

**V3 post-training** (p.28-30): 1.5M SFT instances; reasoning data generated by "an internal DeepSeek-R1 model" through domain expert models (SFT + RL) and rejection sampling; non-reasoning data from DeepSeek-V2.5 checked by human annotators; SFT 2 epochs, cosine LR 5e-6 to 1e-6; RL with rule-based and model-based reward models; GRPO (Section 5.2.2, p.30).

### 1.5 DeepSeek-V3.1 (HF card, Aug 2025)

Source: https://huggingface.co/deepseek-ai/DeepSeek-V3.1 (repo created 2025-08-21).
- "DeepSeek-V3.1 is post-trained on the top of DeepSeek-V3.1-Base, which is built upon the original V3 base checkpoint through a two-phase long context extension approach, following the methodology outlined in the original DeepSeek-V3 report."
- "The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens."
- "DeepSeek-V3.1 is trained using the **UE8M0 FP8 scale data format on both model weights and activations** to ensure compatibility with microscaling data formats."
- No technical report, no GPU hours, no cost. UNVERIFIED anything beyond the card.

### 1.6 DeepSeek-V3.2-Exp and V3.2 (arXiv 2512.02556)

> p.3: "Starting from a base checkpoint of DeepSeek-V3.1-Terminus, whose context length has been extended to 128K, we perform continued pre-training followed by post-training to create DeepSeek-V3.2."

Both stages use data "totally aligned with the 128K long context extension data used for DeepSeek-V3.1-Terminus" (p.3).
- Dense warm-up: only the lightning indexer trains, main model frozen, KL loss to the main attention distribution; LR 1e-3; 1000 steps x 16 sequences x 128K = 2.1B tokens (p.4).
- Sparse training: all parameters; top-2048 KV tokens per query; LR 7.3e-6.
> p.4: "We train both the main model and the indexer for 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens."

DeepSeekMath-V2 is built on "DeepSeek-V3.2-Exp-Base" (2511.22570, p.2).

### 1.7 DeepSeek-V4 (arXiv 2606.19348, preview, 26 Apr 2026)

HF repos created 2026-04-22. Official releases: V4-Flash-0731 (2026-07-31) and V4-Pro-0813 (2026-08-13), both with a DSpark speculative-decoding module (model cards).

> p.1 (abstract): "We pre-train both models on more than 32T diverse and high-quality tokens, followed by a comprehensive post-training pipeline"
>
> p.5: "During pre-training, we train DeepSeek-V4-Flash on 32T tokens and DeepSeek-V4-Pro on 33T tokens, respectively."

- Data (p.24): filters out "batched auto-generated and templated content" to avoid model collapse; agentic data added in mid-training; larger multilingual corpus; emphasis on long documents (scientific papers, technical reports). Sample-level attention masking is now used (V3 did not mask).
- Optimizer (p.25): Muon for most parameters; AdamW for embedding, prediction head and RMSNorm weights. Muon momentum 0.95, weight decay 0.1, update RMS rescaled to 0.18.
- Sequence length grows 4K, 16K, 64K, 1M during pre-training. Dense attention for the first 1T tokens (Flash), sparse attention introduced at 64K with a short indexer warm-up (p.25).
- MTP weight 0.3 then 0.1 at LR decay; bias update speed 0.001; balance loss 0.0001 (p.25-26).
- Stability (p.26, Section 4.2.3): "Anticipatory Routing" (routing indices computed with parameters from step t minus delta-t; triggered only after a loss spike; overhead bounded "to approximately 20%" when active) and "SwiGLU Clamping" (linear part to [-10, 10], gate capped at 10).
- Routed expert weights in FP4 (p.5); FP4 quantization-aware training in post-training (p.33).
- **Not disclosed:** cluster size, GPU type for training, GPU hours, cost. Do not invent them.

### 1.8 DeepSeek-V4.1-Flash (arXiv 2609.19969, 17 Sep 2026)

HF repo created 2026-09-10; update DeepSeek-V4.1-Flash-1017 created 2026-10-17 (card calls it "a minor update").
> p.22: "We train DeepSeek-V4.1-Flash on 45T tokens of multimodal data with no instability. We keep the batch size fixed at 100.6 million tokens throughout training."

- Causal Encoder-Decoder (20 + 20 layers), CSA2, Single-Pass mHC, Engram (196B parameters per the card), DSpark; natively multimodal with a from-scratch DeepSeek-ViT (p.1, p.21-23; card).
- Data (p.20-21): filters "model-generated content with limited information gain"; text-only to multimodal token ratio 7:1; packing padding rate at most 1e-4.
- Optimizer (p.22): Muon for linear layers, AdamW for RMSNorm and non-matrix params, "Sinkhorn-balanced update" for embeddings and prediction head; Engram LR scaled 5x.
- Vision encoder pre-trained separately: SigLIP loss on ~47B image-text pairs, then 236B tokens of autoregressive fine-tuning with a 4B MoE LLM (p.22-23).
- **Not disclosed:** hardware, GPU hours, cost.

---

## 2. Training infrastructure

### 2.1 The V3 cluster and parallelism (2412.19437, Section 3)

> p.11: "DeepSeek-V3 is trained on a cluster equipped with 2048 NVIDIA H800 GPUs. Each node in the H800 cluster contains 8 GPUs connected by NVLink and NVSwitch within nodes. Across different nodes, InfiniBand (IB) interconnects are utilized to facilitate communications."
>
> p.12: "On the whole, DeepSeek-V3 applies 16-way Pipeline Parallelism (PP) (Qi et al., 2023a), 64-way Expert Parallelism (EP) (Lepikhin et al., 2021) spanning 8 nodes, and ZeRO-1 Data Parallelism (DP) (Rajbhandari et al., 2020)."

Framework: HAI-LLM, built in-house (p.12). No tensor parallelism in training (p.4, p.12).

### 2.2 DualPipe (Section 3.2.1, p.12-13)

> p.12: "For DeepSeek-V3, the communication overhead introduced by cross-node expert parallelism results in an inefficient computation-to-communication ratio of approximately 1:1."
>
> p.12: "The key idea of DualPipe is to overlap the computation and communication within a pair of individual forward and backward chunks."
>
> p.12: "It employs a bidirectional pipeline scheduling, which feeds micro-batches from both ends of the pipeline simultaneously and a significant portion of communications can be fully overlapped."

- Each chunk is split into attention, all-to-all dispatch, MLP, all-to-all combine; backward split into "backward for input" and "backward for weights" as in ZeroBubble (p.12).
- Figure 4 (p.12): overlap of one forward and one backward chunk. Figure 5 (p.13): 8 PP ranks, 20 micro-batches, two directions.
- Table 2 (p.13), bubbles and memory:

| Method | Bubble | Parameter | Activation |
|---|---|---|---|
| 1F1B | (PP-1)(F+B) | 1x | PP |
| ZB1P | (PP-1)(F+B-2W) | 1x | PP |
| DualPipe | (PP/2-1)(F&B+B-3W) | 2x | PP+1 |

- DualPipe keeps two copies of parameters, but "this does not significantly increase the memory consumption since we use a large EP size" (p.13). Only needs stages and micro-batches divisible by 2 (p.13).
- Repo: https://github.com/deepseek-ai/DualPipe (released Day 4 of Open-Source Week, Feb 2025). The README adds **DualPipeV**, a V-shape "cut-in-half" schedule credited to Sea AI Lab, same bubble formula but PP/2 devices.

### 2.3 Cross-node all-to-all (Section 3.2.2, p.13-14)

> p.13: "NVLink offers a bandwidth of 160 GB/s, roughly 3.2 times that of IB (50 GB/s)."

- Each token goes to at most 4 nodes; tokens cross IB to the GPU with the same in-node index, then forward over NVLink (p.13).
- Each token can select an average of 3.2 experts per node, so up to 13 experts (4 nodes x 3.2) at the same communication cost (p.13-14).
> p.14: "Overall, under such a communication strategy, only 20 SMs are sufficient to fully utilize the bandwidths of IB and NVLink."
- Warp specialization, 20 SMs split into 10 channels, custom PTX instructions, auto-tuned chunk size to reduce L2 cache use (p.14).
- Repo: https://github.com/deepseek-ai/DeepEP (Day 2, Feb 2025): "the first open-source EP communication library for MoE model training and inference", with NVLink and RDMA, FP8 dispatch, low-latency decode kernels (open-infra-index README). As of Nov 2026 the README describes V2 / V2.5 (EPv2, NCCL Gin backend, Engram and PP buffers, Ascend port). Benchmark numbers in older README versions were not rechecked: UNVERIFIED.

### 2.4 Memory saving (Section 3.2.3, p.14)

1. Recompute all RMSNorm and MLA up-projections in backward.
2. EMA of parameters kept in CPU memory, updated asynchronously.
3. Shared embedding and output head between MTP module and main model, made physical by placing the shallowest and deepest layers on the same PP rank (possible because of DualPipe).

### 2.5 FP8 mixed precision (Section 3.3, p.14-18)

Figure to screenshot: **Figure 6** (p.15, overall FP8 framework, Linear operator) and **Figure 7** (p.16, (a) fine-grained quantization, (b) increased accumulation precision).

> p.15: "we introduce a fine-grained quantization strategy: tile-wise grouping with 1 × 𝑁𝑐elements or block-wise grouping with 𝑁𝑐× 𝑁𝑐elements."
>
> p.15: "Notably, compared with the BF16 baseline, the relative loss error of our FP8-training model remains consistently below 0.25%, a level well within the acceptable range of training randomness."
>
> p.16: "(1) for activations, we group and scale elements on a 1x128 tile basis (i.e., per token per 128 channels); and (2) for weights, we group and scale elements on a 128x128 block basis (i.e., per 128 input channels per 128 output channels)."
>
> p.17: "we observe that the accumulation precision of FP8 GEMM on NVIDIA H800 GPUs is limited to retaining around 14 bits, which is significantly lower than FP32 accumulation precision."
>
> p.17: "Taking GEMM operations of two random matrices with K = 4096 for example, in our preliminary test, the limited accumulation precision in Tensor Cores results in a maximum relative error of nearly 2%."
>
> p.17: "Based on our experiments, setting 𝑁𝐶= 128 elements, equivalent to 4 WGMMAs, represents the minimal accumulation interval that can significantly improve precision without introducing substantial overhead." (sentence continues past the extracted span)

Other FP8 facts:
- All three Linear GEMMs (Fprop, Dgrad, Wgrad) in FP8; kept in BF16/FP32: embedding, output head, MoE gating, normalization, attention (p.15).
- Validated on two scales (similar to V2-Lite and V2) for ~1T tokens (p.15, Appendix B.1).
- E4M3 on all tensors, unlike the usual E4M3 forward / E5M2 backward hybrid (p.17).
- Online max-abs scaling per tile/block, no delayed quantization (p.17).
- AdamW moments in BF16; master weights and gradients in FP32 (p.18).
- Activations after attention stored in a custom E5M6 format; SwiGLU inputs cached in FP8 and output recomputed (p.18).
- Dispatch in FP8; combine kept in BF16 (p.18).
- Section 3.5 (p.19-21) gives hardware suggestions (e.g. higher FP8 accumulation precision, tile/block scaling in Tensor Cores).
- Repo: https://github.com/deepseek-ai/DeepGEMM (Day 3, Feb 2025): FP8 GEMM library for dense and MoE GEMMs, "Up to 1350+ FP8 TFLOPS on Hopper GPUs" (open-infra-index), later "up to 1550 TFLOPS on H800" (DeepGEMM README news, 2025.04.18). By Sep 2026 it also covers FP4, Mega MoE, the V3.2 lightning indexer, mHC and an Ascend port (README news).
- V3.1 switched to UE8M0 FP8 scales (V3.1 card). V4 uses FP4 routed experts (2606.19348, p.5).

### 2.6 Fire-Flyer AI-HPC (arXiv 2408.14158, SC24)

> p.1: "For DL training, we deployed the Fire-Flyer 2 with 10,000 PCIe A100 GPUs, achieved performance approximating the DGX-A100 while reducing costs by half and energy consumption by 40%."

- Two-layer Fat-Tree with storage and compute on one network; two zones of 800-port Fat-Trees; 122 switches instead of a three-layer design (p.5).
- PCIe A100 gives ~83% of DGX-A100 TF32/FP16 GEMM performance at ~60% of GPU cost and energy (p.5, Table II).
- Software: HFReduce (allreduce), HaiScale (parallelism), 3FS (file system), HAI-Platform (p.1-2).
- Note: this is the A100 cluster. V3 was trained on H800 (V3 p.11). R1 preparation experiments used A100s (R1v2 p.35). Which cluster trained DeepSeek LLM 67B is not stated in that paper: UNVERIFIED.

### 2.7 3FS and the rest of Open-Source Week (Feb 24-28, 2025)

Source: https://github.com/deepseek-ai/open-infra-index
- Day 1 FlashMLA (inference), Day 2 DeepEP, Day 3 DeepGEMM, Day 4 DualPipe + EPLB + profile-data, Day 5 3FS + Smallpond, Day 6 inference-system overview.
- 3FS: "6.6 TiB/s aggregate read throughput in a 180-node cluster"; "3.66 TiB/min throughput on GraySort benchmark in a 25-node cluster"; used for "Training data preprocessing, dataset loading, checkpoint saving/reloading, embedding vector search & KVCache lookups" (open-infra-index). The 3FS README gives the test setup: 180 storage nodes, each with 2x200Gbps IB NICs and sixteen 14TiB NVMe SSDs, 500+ client nodes.

### 2.8 ISCA 2025 paper (arXiv 2505.09343)

- Table 2 (p.4): training cost per token at sequence length 4096: V2 155 GFLOPS, V3 250 GFLOPS, Qwen-72B dense 394 GFLOPS, LLaMA-405B dense 2448 GFLOPS.
- Table 1 (p.4): KV cache per token (BF16): V3 (MLA) 70.272 KB; Qwen-2.5 72B 327.680 KB (4.66x); LLaMA-3.1 405B 516.096 KB (7.28x).
- Also covers a multi-plane two-layer Fat-Tree network (p.1-2). Not extracted in detail here.

### 2.9 RL infrastructure (R1v2 Supplementary B.1, 2501.12948 p.17)

Figure 5 (p.17) shows four modules: Rollout (vLLM workers, expert parallelism, redundant hot experts, MTP for self-speculative decoding), Inference (reward model + reference forward pass), Rule-based Reward (code executor, answer matcher, format checker, run asynchronously), and Training. Good diagram for an article.

---

## 3. GRPO

### 3.1 Origin: DeepSeekMath (arXiv 2402.03300, Feb 2024)

- Context: DeepSeekMath 7B was continued from DeepSeek-Coder-Base-v1.5 7B on 120B math tokens from Common Crawl (p.1). MATH score 51.7% (p.1).
- GRPO is Section 4.1 (starts p.11). **Equation (3), the GRPO objective, is on p.13** (with Figure 4 at the top of p.13). The KL estimator, equation (4), is on p.14. Algorithm 1 (iterative GRPO) is on p.14.

Screenshot passages:
> p.13 (Figure 4 caption): "Figure 4 | Demonstration of PPO and our GRPO. GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."
>
> p.13: "As the value function employed in PPO is typically another model of comparable size as the policy model, it brings a substantial memory and computational burden."
>
> p.13: "To address this, as shown in Figure 4, we propose Group Relative Policy Optimization (GRPO), which obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline."
>
> p.14 (Section 4.1.2, outcome supervision): "Subsequently, these rewards are normalized by subtracting the group average and dividing by the group standard deviation."
>
> p.14: "we estimate the KL divergence with the following unbiased estimator (Schulman, 2020):"

Equation (3), written cleanly (DeepSeekMath p.13):

J_GRPO(theta) = E[q ~ P(Q), {o_i}_{i=1..G} ~ pi_theta_old(O|q)] (1/G) sum_i (1/|o_i|) sum_t { min[ r_{i,t} * A_{i,t}, clip(r_{i,t}, 1-eps, 1+eps) * A_{i,t} ] - beta * D_KL[pi_theta || pi_ref] },
with r_{i,t} = pi_theta(o_{i,t} | q, o_{i,<t}) / pi_theta_old(o_{i,t} | q, o_{i,<t}).

KL estimator, equation (4) (p.14): D_KL = pi_ref/pi_theta - log(pi_ref/pi_theta) - 1, "which is guaranteed to be positive."

Group-relative advantage (outcome supervision, p.14): A_{i,t} = (r_i - mean(r)) / std(r), the same value for every token in output i.

Why it drops the value model (two reasons, p.13): (1) the critic is about as large as the policy, so memory and compute cost; (2) in LLM RL usually only the last token gets a reward, which makes a per-token value function hard to train.

Two more points:
- KL goes directly in the loss, not into the reward as in PPO (p.13).
- DeepSeekMath-RL setup (p.15): ~144K GSM8K/MATH CoT questions; policy LR 1e-6; KL coefficient 0.04; 64 samples per question; max length 1024; batch 1024; one update per exploration stage. Results: GSM8K 82.9% to 88.2%, MATH 46.8% to 51.7% (p.2).
- A finding writers like (p.21, Figure 7): "RL enhances Maj@K but not Pass@K." "These findings indicate that RL enhances the model’s overall performance by rendering the output distribution more robust".

### 3.2 GRPO in later DeepSeek models

| Model | How GRPO is used | Source |
|---|---|---|
| DeepSeek-V2 Chat | Two-stage RL: reasoning alignment (reasoning RM), then human preference (helpful + safety + rule RMs) | 2405.04434 p.17 |
| DeepSeek-V3 | GRPO "Similar to DeepSeek-V2" | 2412.19437 p.30 |
| R1-Zero and R1 | Sequence-level form, rule-based rewards; see section 4 | 2501.12948 (R1v2) p.2-3 |
| V3.2 | Single merged RL stage, plus four stabilizers (below) | 2512.02556 p.6-8 |
| V4 specialists | GRPO "maintaining hyper-parameters closely aligned with our prior research" | 2606.19348 p.28 |
| V4.1 | "standard paradigm of supervised fine-tuning (SFT) followed by reinforcement learning (RL) and on-policy distillation (OPD)", "without algorithmic modifications" | 2609.19969 p.25 |

R1's GRPO hyper-parameters (2501.12948 R1v2, p.3, R1-Zero):
> "To train DeepSeek-R1-Zero, we set the learning rate to 3e-6, the KL coefficient to 0.001, and the sampling temperature to 1 for rollout. For each question, we sample 16 outputs with a maximum length of 32,768 tokens before the 8.2k step and 65,536 tokens afterward. As a result, both the performance and response length of DeepSeek-R1-Zero exhibit a significant jump at the 8.2k step, with training continuing for a total of 10,400 steps, corresponding to 1.6 training epochs. Each training step consists of 32 unique questions, resulting in a training batch size of 512."

Also p.3: reference model replaced by the latest policy every 400 steps; each rollout generates 8,192 outputs, split into 16 mini-batches, one inner epoch. R1 first RL stage (p.7): "In the first stage of RL, we set the learning rate to 3e-6, the KL coefficient to 0.001, the GRPO clip ratio 𝜀to 10, and the sampling temperature to 1 for rollout." p.8 notes the clip ratio "plays a crucial role in training". (The value 10 is what the PDF prints; it is unusually large compared with PPO's typical 0.2. Quote it, but do not over-interpret.)

V3.2 changes to GRPO (2512.02556, Section 3.1, p.7-8):
1. Advantage is **mean-centred only**: "ˆ𝐴𝑖,𝑡= 𝑅𝑖−mean(𝑹)." (p.7). No division by std (contrast DeepSeekMath p.14).
2. Unbiased KL estimate: the K3 estimator is corrected with the importance ratio pi_theta / pi_old; "For certain domains, such as mathematics, applying a relatively weak KL penalty or even omitting it entirely can yield improved performance." (p.7)
3. Off-Policy Sequence Masking: mask negative-advantage sequences whose KL(pi_old || pi_theta) exceeds a threshold delta (p.7-8).
4. Keep Routing: reuse the inference-time expert routes during training. "This Keep Routing operation was found crucial for RL training stability of MoE models, and has been adopted in our RL training pipeline since" (p.8; sentence continues to name the model).
5. Keep Sampling Mask: reuse top-p/top-k truncation masks from sampling during training (p.8).

---

## 4. DeepSeek-R1

### 4.1 Versions and what changed

- arXiv v1 (22 Jan 2025, 22 pages) = `2501.12948v1.pdf`.
- Nature article (Vol 645, pp. 633-638, published online 17 Sep 2025) + Supplementary Information + peer review file ("Peer reviewer reports are available", Nature PDF p.8; reviewers named include Edward Beeching, Yarin Gal, José Hernández-Orallo).
- arXiv v2 (4 Jan 2026, 86 pages) = `2501.12948.pdf`: the Nature main text plus the Supplementary material as appendices. Page numbers in this dossier marked R1v2 refer to this file.

**Discrepancy to flag:** R1-Zero AIME 2024 pass@1 is **71.0%** in v1 (v1 p.6-7, Table 2) and **77.9%** in Nature/v2 ("jumping from an initial 15.6% to 77.9%", R1v2 p.4; Nature p.2). The cons@16 / majority-vote number 86.7% is unchanged. v2 describes raising the max length from 32,768 to 65,536 tokens at step 8.2k with "a significant jump" (R1v2 p.3). That this explains the change is my inference: UNVERIFIED. Use 77.9% when citing Nature; say which version you cite.

### 4.2 R1-Zero: pure RL from the base model

- Base: DeepSeek-V3-Base. No SFT before RL (R1v2 p.2).
- Template (R1v2 p.3 Table 1; v1 p.6 Table 1): reasoning inside `<think>...</think>`, answer inside `<answer>...</answer>`.
- Rewards: accuracy (boxed answer match; compiler and test cases for code) + format (think tags), equal weight (R1v2 p.3-4).
- No neural reward models for reasoning. v1 p.6: "We do not apply the outcome or process neural reward model in developing DeepSeek-R1-Zero, because we find that the neural reward model may suffer from reward hacking in the large-scale reinforcement learning process". R1v2 p.4 repeats the reasoning and adds that retraining RMs "necessitates substantial computational resources".
- RL data (R1v2 p.19-20, Supplementary B.3.1, Table 4): math 26k questions (proofs excluded, binary reward); code 17k competition problems + 8k bug-fixing problems from GitHub issues; STEM 22k multiple-choice (15.5% physics, 30.7% biology, 46.5% chemistry, 7.3% other); logic 15k; general 66k helpfulness + 12,000 harmlessness questions. All in Chinese or English.

**Aha moment (screenshot: v1 p.9 Table 3, or R1v2 p.5 Table 2):**
> v1 p.8: "A particularly intriguing phenomenon observed during the training of DeepSeek-R1-Zero is the occurrence of an “aha moment”."
>
> v1 p.9 (inside the table): "Wait, wait. Wait. That’s an aha moment I can flag here."
>
> v1 p.9 caption: "Table 3 | An interesting “aha moment” of an intermediate version of DeepSeek-R1-Zero. The model learns to rethink using an anthropomorphic tone. This is also an aha moment for us, allowing us to witness the power and beauty of reinforcement learning."
>
> R1v2 p.5: "Notably, during training, DeepSeek-R1-Zero exhibits an “aha moment” (Table 2), characterized by a sudden increase in the use of the word “wait” during reflections (see Figure 9(b) in Supplementary C.2 for details)."

The Nature version makes the aha moment measurable (frequency of "wait"); R1v2 p.38 notes a reflection word that was absent early, "showed occasional usage between steps 4000-7000", then rose. Figure 9 is in Supplementary C.2 (R1v2 p.37-38).

Response length grows over training (v1 p.8 Figure 3; R1v2 p.4 Figure 1(b)).

**Language mixing:** R1-Zero mixes languages, "occasionally combining English and Chinese within a single chain-of-thought response" (R1v2 p.2). R1v2 p.13 suggests the cause: "The training data of DeepSeek-V3 base are mostly Chinese and English, which might be the cause for DeepSeek-R1-Zero language mixing when the language consistent reward is absent."

### 4.3 The four-stage R1 pipeline

Screenshot: **Figure 2 on R1v2 p.6** (multi-stage pipeline with checkpoints Dev1, Dev2, Dev3), or the v1 text on p.9-11.

> v1 p.9: "The pipeline consists of four stages, outlined as follows."

1. **Cold start** (v1 p.9; R1v2 p.6, B.3.2 p.20). "In this work, we collect thousands of cold-start data to fine-tune the DeepSeek-V3-Base as the starting point for RL." (v1 p.9). Format `|special_token|<reasoning_process>|special_token|<summary>` (v1 p.9). R1v2 p.20 says the motivation "is primarily product-driven".
2. **Reasoning-oriented RL** (v1 p.10; R1v2 p.7-8). Same RL as R1-Zero plus a language consistency reward:
   > v1 p.10: "To mitigate the issue of language mixing, we introduce a language consistency reward during RL training, which is calculated as the proportion of target language words in the CoT."
   Ablation shows "a slight degradation in the model’s performance" (v1 p.10; R1v2 B.6 p.35-37).
3. **Rejection sampling + SFT** (v1 p.10-11). "In total, we collect about 600k reasoning related training samples." (v1 p.10). "In the end, we collected a total of approximately 200k training samples that are unrelated to reasoning." (v1 p.11). 800k total; fine-tune DeepSeek-V3-Base for two epochs (v1 p.11). Some samples judged by a generative reward model (DeepSeek-V3) (v1 p.10).
4. **RL for all scenarios** (v1 p.11; R1v2 p.8). Rule rewards for reasoning; reward models for helpfulness (judged on the final summary only) and harmlessness (judged on the whole response). R1v2 p.8: temperature 0.7; "The stage comprises a total of 1,700 training steps, during which general instruction data and preference-based rewards are incorporated exclusively in the final 400 steps." More steps with model-based reward caused reward hacking (R1v2 p.8, B.5 p.35).

Reward models (R1v2 p.7): helpful RM trained on 66,000 preference pairs judged by DeepSeek-V3 four times each (batch 256, LR 6e-6, one epoch); safety RM on 106,000 prompts labelled safe/unsafe, point-wise.
> R1v2 p.7: "In total, we curated 66,000 data pairs for training the reward model. The prompts used in this dataset are all non-reasoning questions and are sourced either from publicly available open-source datasets or from users who have explicitly consented to share their data for the purpose of model improvement."

Stage-by-stage results (R1v2 p.9, Table 3):

| Benchmark | R1-Zero | R1-Dev1 | R1-Dev2 | R1-Dev3 | R1 |
|---|---|---|---|---|---|
| MMLU-Pro (EM) | 68.9 | 74.1 | 83.8 | 83.1 | 84.0 |
| IF-Eval (Prompt Strict) | 46.6 | 71.7 | 72.0 | 78.1 | 83.3 |
| GPQA Diamond | 75.8 | 66.1 | 70.7 | 71.2 | 71.5 |
| AlpacaEval2.0 (LC-winrate) | 24.7 | 50.1 | 55.8 | 62.1 | 87.6 |
| ArenaHard | 53.6 | 77.0 | 73.2 | 75.6 | 92.3 |
| LiveCodeBench (Pass@1-COT) | 50.0 | 57.5 | 63.5 | 64.6 | 65.9 |
| Codeforces (Rating) | 1444 | 1534 | 1687 | 1746 | 2029 |
| SWE Verified | 43.2 | 39.6 | 44.6 | 45.6 | 49.2 |
| Aider-Polyglot | 12.2 | 6.7 | 25.6 | 44.8 | 53.3 |
| AIME 2024 (Pass@1) | 77.9 | 59.0 | 74.0 | 78.1 | 79.8 |
| MATH-500 | 95.9 | 94.2 | 95.9 | 95.4 | 97.3 |
| CNMO 2024 | 88.1 | 58.0 | 73.9 | 77.3 | 78.8 |

Story in one line: the cold start (Dev1) makes the model readable but costs reasoning (AIME 77.9 to 59.0); reasoning RL (Dev2) wins it back; the final RL mostly lifts AlpacaEval (+25%) and ArenaHard (+17%) (R1v2 p.9).

### 4.4 Distillation

> v1 p.11: "For distilled models, we apply only SFT and do not include an RL stage, even though incorporating RL could substantially boost model performance."

Students: Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, Llama-3.3-70B-Instruct; data: the 800k samples (v1 p.11).

Distilled models (v1 p.14 Table 5; identical in R1v2 p.61 Table 15):

| Model | AIME 2024 pass@1 | AIME 2024 cons@64 | MATH-500 | GPQA Diamond | LiveCodeBench | Codeforces rating |
|---|---|---|---|---|---|---|
| GPT-4o-0513 | 9.3 | 13.4 | 74.6 | 49.9 | 32.9 | 759 |
| Claude-3.5-Sonnet-1022 | 16.0 | 26.7 | 78.3 | 65.0 | 38.9 | 717 |
| OpenAI-o1-mini | 63.6 | 80.0 | 90.0 | 60.0 | 53.8 | 1820 |
| QwQ-32B-Preview | 50.0 | 60.0 | 90.6 | 54.5 | 41.9 | 1316 |
| R1-Distill-Qwen-1.5B | 28.9 | 52.7 | 83.9 | 33.8 | 16.9 | 954 |
| R1-Distill-Qwen-7B | 55.5 | 83.3 | 92.8 | 49.1 | 37.6 | 1189 |
| R1-Distill-Qwen-14B | 69.7 | 80.0 | 93.9 | 59.1 | 53.1 | 1481 |
| R1-Distill-Qwen-32B | 72.6 | 83.3 | 94.3 | 62.1 | 57.2 | 1691 |
| R1-Distill-Llama-8B | 50.4 | 80.0 | 89.1 | 49.0 | 39.6 | 1205 |
| R1-Distill-Llama-70B | 70.0 | 86.7 | 94.5 | 65.2 | 57.5 | 1633 |

Distillation vs RL (v1 p.15 Table 6; R1v2 p.61 Table 16): RL on Qwen2.5-32B-Base for over 10K steps ("DeepSeek-R1-Zero-Qwen-32B" in v1, "Qwen2.5-32B-Zero" in v2) reaches AIME 47.0 / MATH-500 91.6 / GPQA 55.0 / LCB 40.2, about QwQ-32B-Preview, while R1-Distill-Qwen-32B gets 72.6 / 94.3 / 62.1 / 57.2. New in v2 (p.62): Qwen2-Math-7B-Zero (base released before o1) trained ~10,000 steps: AIME 2024 22.3%, AIME 2025 18.1% (Table 17); and small bases (7B dense, 16B MoE) "consistently failed to yield meaningful improvements" (p.62).

R1-0528 (HF card) also released DeepSeek-R1-0528-Qwen3-8B: AIME 2024 86.0, AIME 2025 76.3, HMMT Feb 2025 61.5, GPQA 61.1, LCB 60.5.

### 4.5 Unsuccessful attempts

Screenshot: v1 p.15-16 Section 4.2 (same text in R1v2 p.63-64, Supplementary G.2).
> v1 p.15 (PRM): "Third, once a model-based PRM is introduced, it inevitably leads to reward hacking (Gao et al., 2022), and retraining the reward model needs additional training resources and it complicates the whole training pipeline."
>
> v1 p.16 (MCTS): "In conclusion, while MCTS can improve performance during inference when paired with a pre-trained value model, iteratively boosting model performance through self-search remains a significant challenge."

PRM's three problems (v1 p.15): defining a fine-grained step; judging whether a step is correct; reward hacking. MCTS problems (v1 p.15-16): token search space much larger than chess, node-extension limits cause local optima, and a fine-grained value model is hard to train.

### 4.6 The Nature disclosures: cost, data, safety

**RL cost (verified, exact).** NatureSI p.27-28 (Supplementary section numbered B.4.4 in R1v2 p.35-37). Nature main text p.3: "The total training cost is listed in Supplementary Information, section 2."

> R1v2 p.35 / NatureSI p.27: "Regarding our research on DeepSeek-R1, we utilized the A100 GPUs to prepare for the experiments with a smaller model (30B parameters). The results from this smaller model have been promising, which has allowed us to confidently scale up to 660B R1-Zero and R1. For the training of DeepSeek-R1-Zero, we employed 64*8 H800 GPUs, and the process required approximately 198 hours. Additionally, during the training phase of DeepSeek-R1, we utilized the same 64*8 H800 GPUs, completing the process in about 4 days, or roughly 80 hours. To create the SFT datasets, we use 5K GPU hours. The details are shown in Table 7."

Table 7 (R1v2 p.37; NatureSI p.28), "Training costs of DeepSeek-R1, assuming the rental price of H800 is $2 per GPU hour.":

| Training Costs | DeepSeek-R1-Zero | SFT data creation | DeepSeek-R1 | Total |
|---|---|---|---|---|
| in H800 GPU Hours | 101K | 5K | 41K | 147K |
| in USD | $202K | $10K | $82K | $294K |

So the "$294K" is exact as printed ($294K, 147K GPU hours). Derived checks: 512 x 198 h = 101,376; 512 x 80 h = 40,960. This sits **on top of** the V3 base cost ($5.576M, which itself excludes research). Press stories that said "R1 cost $294K to train" drop that context. Several outlets also misreported the GPU count as 648; the paper says 64*8 = 512.

**Data disclosures** (all R1v2): V3-Base data is web pages and e-books with no intentional synthetic data, but with indirect OpenAI-generated content on the web (p.13); RL dataset composition (p.19-20, above); reward-model prompts from open datasets or consenting users (p.7); decontamination of ~6 million math pre-training texts and use of pre-2023 competitions for math SFT/RL (p.40); "We also release SFT and RL data to the public at xxx." (p.66; placeholder as printed in the arXiv v2).

**Safety disclosures** (R1v2 p.10 Section 5; Supplementary D.3, p.44-54): risk-control system with a "Risk Review Prompt" (Listing 8, p.44); 6 public safety benchmarks (Table 9, p.48); in-house taxonomy (Table 10, p.51); multilingual safety; jailbreak robustness (Table 11, p.54).
> R1v2 p.10: "These comprehensive safety analyses conclude that the inherent safety level of the DeepSeek-R1 model, compared to other state-of-the-art models, is generally at a moderate level (comparable to GPT-4o (2024-05-13)). Besides, when coupled with the risk control system, the model’s safety level is elevated to a superior standard."

The ethics statement (p.10) also names jailbreaks that yield "explosive manufacturing plans" and fine-tuning that removes safety as risks.

---

## 5. Later post-training

### 5.1 R1-0528 (May 2025, HF card)
"leveraging increased computational resources and introducing algorithmic optimization mechanisms during post-training". AIME 2025 70.0% to 87.5% (card). No paper.

### 5.2 V3.1: hybrid thinking (Aug 2025, HF card)
- "One model supports both thinking mode and non-thinking mode by changing the chat template."
- "DeepSeek-V3.1-Think achieves comparable answer quality to DeepSeek-R1-0528, while responding more quickly."
- Post-training improved tool use and agents; tool calls supported in non-thinking mode; search agent format in thinking mode.
- No training details beyond the context-extension and UE8M0 notes (1.5). The RL recipe is UNVERIFIED.
- V3.1-Terminus (Sep 2025, card): fixes "mixed Chinese-English text and occasional abnormal characters" and improves code and search agents.

### 5.3 V3.2 and V3.2-Speciale (arXiv 2512.02556)

Pipeline (p.5-6): specialist distillation, then one mixed RL stage.
- Specialists for math, programming, logic, general agentic tasks, agentic coding, agentic search, each "trained with large-scale Reinforcement Learning (RL) computing" (p.6). Their data trains the final checkpoint.
- "we merge reasoning, agent, and human alignment training into one RL stage." (p.6). Rewards: rule-based outcome, length penalty, language consistency; generative reward model with per-prompt rubrics for general tasks (p.6).
- Speciale: "This model was trained exclusively on reasoning data with a reduced length penalty during RL." (p.6). It also uses the DeepSeekMath-V2 dataset and reward method (p.6).

**RL compute share:**
> p.2: "Notably, this framework allocates a post-training computational budget exceeding 10% of the pre-training cost, unlocking advanced capabilities."
>
> p.13: "Over recent months, we have observed consistent performance improvements correlating with extended RL training budget, which already exceeds 10% of the pre-training cost."

No GPU-hour or dollar figure is given. "Pre-training cost" is not defined (whether it means V3's original 2.664M GPU hours or something else): UNVERIFIED.

**Agentic task synthesis:**
> p.2: "advance to large-scale agentic task synthesis, where we generate over 1,800 distinct environments and 85,000 complex prompts."
>
> p.11: "We then perform RL on this dataset using DeepSeek-V3.2 and retain only instances with non-zero pass@100, resulting in 1,827 environments and their corresponding tasks (4,417 in total)."

Table 1 (p.10), RL agent tasks:

| Task type | Number of tasks | Environment | Prompt |
|---|---|---|---|
| code agent | 24667 | real | extracted |
| search agent | 50275 | real | synthesized |
| general agent | 4417 | synthesized | synthesized |
| code interpreter | 5908 | real | extracted |

(24,667 + 50,275 + 4,417 + 5,908 = 85,267, which matches "85,000 complex prompts": derived.) Code-agent environments come from "millions of issue-Pull Request (PR) pairs from GitHub" and an environment-setup agent (p.11). The trip-planning example task (p.12) makes a good illustration of "hard to solve, easy to verify". Synthetic tasks are hard: DeepSeek-V3.2-Exp scores 12% on 50 sampled tasks, frontier closed models at most 62% (p.15).

### 5.4 DeepSeekMath-V2 (arXiv 2511.22570, Nov 2025)

- Built on DeepSeek-V3.2-Exp-Base (p.2).
- Method: train an LLM verifier that lists issues and scores proofs (1, 0.5, 0); add a **meta-verifier** that checks the verifier's analysis; train the generator with the verifier as reward and make it self-verify before finalizing; scale verification compute to auto-label hard proofs to keep improving the verifier (p.1-6).
- Meta-verification raised the quality score of verifier analyses "from 0.85 to 0.96" (p.4).
- Results (p.8-9, Table 1): IMO 2025 5 of 6 problems (83.3% of points); CMO 2024 73.8%; Putnam 2024 118/120 (98.3%).
> p.8: "As shown in Table 1, our approach solved 5 of 6 problems from IMO 2025 and 4 problems plus partial credit on another from CMO 2024, achieving gold medal performance in both pinnacle high-school competitions. On Putnam 2024, the preeminent undergraduate mathematics competition, our model solved 11 of 12 problems completely and the remaining problem with minor errors, scoring 118/120"
- High-compute search (p.8): 64 proofs x 64 verifications, up to 16 refinement iterations.
- Weights: Apache 2.0 (HF card).

### 5.5 DeepSeek-Prover-V2 (arXiv 2504.21801, Apr 2025)

- Lean 4 prover; cold-start data from recursive subgoal decomposition with DeepSeek-V3, then RL (p.1-2).
> p.1: "The resulting model, DeepSeek-Prover-V2-671B, achieves state-of-the-art performance in neural theorem proving, reaching 88.9% pass ratio on the MiniF2F-test and solving 47 out of 658 problems from PutnamBench."
- ProverBench: 325 problems, incl. 15 AIME 24-25; Prover-V2 solves 6 of 15, DeepSeek-V3 (informal, majority voting) 8 (p.1).
- Note: the cached PDF is arXiv v2 (18 Jul 2025), which says 47 PutnamBench problems. Earlier versions and press used 49: UNVERIFIED which number v1 printed. Cite v2.

### 5.6 V4 post-training (arXiv 2606.19348): RL replaced by on-policy distillation

> p.28: "Although the training pipeline largely mirrored that of DeepSeek-V3.2, a critical methodological substitution was made: the mixed Reinforcement Learning (RL) stage was entirely replaced by On-Policy Distillation (OPD; Gu et al., 2024; Lu and Lab, 2025)."
>
> p.33: "In this stage, more than ten teacher models covering various domains are employed to distill a single student model."

- Specialists (math, coding, agent, instruction following, ...) get SFT then GRPO (p.5, p.28).
- The student samples its own trajectories and minimizes reverse KL to the relevant teacher, with full-vocabulary logits (p.32-34).
- Infrastructure (p.33-35): FP4 QAT (MXFP4 for MoE expert weights and indexer), teacher scheduling for full-vocabulary OPD, preemptible fault-tolerant rollouts, RL framework for million-token context, sandbox infrastructure for agents.

### 5.7 V4.1-Flash post-training (arXiv 2609.19969)

> p.25: "We find that, under a fixed and unremarkable optimization procedure, systematic improvements in the scale, diversity, and verifiability of synthesized data and environments account for essentially all of the observed gains."

- SFT, then RL, then OPD (p.25). Tasks are (problem, environment, verification system) triplets; the model itself is trained to build tasks, with difficulty and correctness as rewards (p.25).
- General-agent environments use mocked tools based on interaction data that employees and partners return voluntarily; coding-agent environments come from internal and partner sessions and starred GitHub repos, built by cooperating agents (p.26).
- Large-scale asynchronous RL across many agent scaffolds; continuous controllable reasoning effort 1 to 100 trained in RL (p.26-29; card).

### 5.8 Other 2026 DeepSeek training research (verified to exist on arXiv)

- mHC: Manifold-Constrained Hyper-Connections, arXiv 2512.24880 (submitted 31 Dec 2025). Used in V4 and V4.1 (Sinkhorn-Knopp, 20 iterations; 2606.19348 p.24). Not downloaded or read in depth here.
- Conditional Memory via Scalable Lookup (Engram), arXiv 2601.07372 (12 Jan 2026). Used in V4.1-Flash (card: 196B Engram parameters). Not read in depth.
- Third-party safety evaluation of V4.1-Flash-1017: arXiv 2610.35671 (not by DeepSeek).
- I did not find a separate 2026 DeepSeek paper on a new RL algorithm. V4 and V4.1 both say they reuse GRPO-style RL and focus on OPD and data/environments.

---

## 6. Benchmarks

Always say which paper or card a number comes from; DeepSeek evaluation settings change between reports (e.g. LiveCodeBench date windows, output length caps of 8K for V3 vs 32K for R1 vs 64K for R1-0528).

### 6.1 DeepSeek-V3 chat (2412.19437, p.31, Table 6; output limited to 8K)

| Benchmark | V2.5-0905 | Qwen2.5 72B | LLaMA-3.1 405B | Claude-3.5-Sonnet-1022 | GPT-4o-0513 | DeepSeek-V3 |
|---|---|---|---|---|---|---|
| MMLU (EM) | 80.6 | 85.3 | 88.6 | 88.3 | 87.2 | 88.5 |
| MMLU-Pro (EM) | 66.2 | 71.6 | 73.3 | 78.0 | 72.6 | 75.9 |
| GPQA-Diamond | 41.3 | 49.0 | 51.1 | 65.0 | 49.9 | 59.1 |
| SimpleQA | 10.2 | 9.1 | 17.1 | 28.4 | 38.2 | 24.9 |
| LiveCodeBench (Pass@1-COT) | 29.2 | 31.1 | 28.4 | 36.3 | 33.4 | 40.5 |
| Codeforces (Percentile) | 35.6 | 24.8 | 25.3 | 20.3 | 23.6 | 51.6 |
| SWE Verified (Resolved) | 22.6 | 23.8 | 24.5 | 50.8 | 38.8 | 42.0 |
| Aider-Polyglot | 18.2 | 7.6 | 5.8 | 45.3 | 16.0 | 49.6 |
| AIME 2024 (Pass@1) | 16.7 | 23.3 | 23.3 | 16.0 | 9.3 | 39.2 |
| MATH-500 (EM) | 74.7 | 80.0 | 73.8 | 78.3 | 74.6 | 90.2 |
| CNMO 2024 | 10.8 | 15.9 | 6.8 | 13.1 | 10.8 | 43.2 |

V3-Base (p.25, Table 3) selected: MMLU 87.1, MMLU-Pro 64.4, BBH 87.5, HumanEval 65.2, GSM8K 89.3, MATH 61.6.

### 6.2 DeepSeek-R1 (v1 p.13, Table 4; same numbers in R1v2 p.41 Table 8)

| Benchmark | Claude-3.5-Sonnet-1022 | GPT-4o-0513 | DeepSeek-V3 | o1-mini | o1-1217 | DeepSeek-R1 |
|---|---|---|---|---|---|---|
| MMLU (Pass@1) | 88.3 | 87.2 | 88.5 | 85.2 | 91.8 | 90.8 |
| MMLU-Pro | 78.0 | 72.6 | 75.9 | 80.3 | - | 84.0 |
| GPQA Diamond | 65.0 | 49.9 | 59.1 | 60.0 | 75.7 | 71.5 |
| SimpleQA | 28.4 | 38.2 | 24.9 | 7.0 | 47.0 | 30.1 |
| LiveCodeBench (Pass@1-COT) | 38.9 | 32.9 | 36.2 | 53.8 | 63.4 | 65.9 |
| Codeforces (Percentile) | 20.3 | 23.6 | 58.7 | 93.4 | 96.6 | 96.3 |
| Codeforces (Rating) | 717 | 759 | 1134 | 1820 | 2061 | 2029 |
| SWE Verified | 50.8 | 38.8 | 42.0 | 41.6 | 48.9 | 49.2 |
| Aider-Polyglot | 45.3 | 16.0 | 49.6 | 32.9 | 61.7 | 53.3 |
| AIME 2024 (Pass@1) | 16.0 | 9.3 | 39.2 | 63.6 | 79.2 | 79.8 |
| MATH-500 | 78.3 | 74.6 | 90.2 | 90.0 | 96.4 | 97.3 |
| CNMO 2024 | 13.1 | 10.8 | 43.2 | 67.6 | - | 78.8 |

Note: the V3 column here (LCB 36.2, Codeforces percentile 58.7) differs from the V3 paper (40.5, 51.6) because R1 re-evaluated with different settings. Do not mix columns across papers.

R1-Zero (v1 p.7, Table 2): AIME 2024 pass@1 71.0, cons@64 86.7, MATH-500 95.9, GPQA 73.3, LCB 50.0, Codeforces 1444. R1v2 Table 3 gives 77.9 AIME and 75.8 GPQA (see 4.1).

R1 test-time scaling (R1v2 p.60): AIME 2024 pass@64 90.0% vs pass@1 79.8%; majority voting lifts 79.8% to 86.7%.

### 6.3 R1-0528 (HF card, max 64K generation)

MMLU-Redux 93.4, MMLU-Pro 85.0, GPQA 81.0, HLE 17.7, LiveCodeBench (2408-2505) 73.3, Codeforces-Div1 1930, SWE Verified 57.6, Aider-Polyglot 71.6, AIME 2024 91.4, AIME 2025 87.5, HMMT 2025 79.4.

### 6.4 V3.1 (HF card)

| Benchmark | V3.1 NonThinking | V3-0324 | V3.1 Thinking | R1-0528 |
|---|---|---|---|---|
| MMLU-Redux | 91.8 | 90.5 | 93.7 | 93.4 |
| MMLU-Pro | 83.7 | 81.2 | 84.8 | 85.0 |
| GPQA-Diamond | 74.9 | 68.4 | 80.1 | 81.0 |
| HLE | - | - | 15.9 | 17.7 |
| LiveCodeBench (2408-2505) | 56.4 | 43.0 | 74.8 | 73.3 |
| Codeforces-Div1 (Rating) | - | - | 2091 | 1930 |
| Aider-Polyglot | 68.4 | 55.1 | 76.3 | 71.6 |
| SWE Verified (Agent mode) | 66.0 | 45.4 | - | 44.6 |
| Terminal-bench (Terminus 1) | 31.3 | 13.3 | - | 5.7 |
| AIME 2024 | 66.3 | 59.4 | 93.1 | 91.4 |
| AIME 2025 | 49.8 | 51.3 | 88.4 | 87.5 |
| HMMT 2025 | 33.5 | 29.2 | 84.2 | 79.4 |
| BrowseComp | - | - | 30.0 | 8.9 |

V3.1-Terminus (card): GPQA 80.7, HLE 21.7, LCB 74.9, Codeforces 2046, SWE Verified 68.4, Terminal-bench 36.7, BrowseComp 38.5.

### 6.5 V3.2 (2512.02556, p.13 Table 2; thinking mode, 128K context, temperature 1.0)

| Benchmark | Claude-4.5-Sonnet | GPT-5 High | Gemini-3.0 Pro | Kimi-K2 Thinking | DeepSeek-V3.2 Thinking |
|---|---|---|---|---|---|
| MMLU-Pro | 88.2 | 87.5 | 90.1 | 84.6 | 85.0 |
| GPQA Diamond | 83.4 | 85.7 | 91.9 | 84.5 | 82.4 |
| HLE | 13.7 | 26.3 | 37.7 | 23.9 | 25.1 |
| LiveCodeBench | 64.0 | 84.5 | 90.7 | 82.6 | 83.3 |
| Codeforces (Rating) | 1480 | 2537 | 2708 | - | 2386 |
| AIME 2025 | 87.0 | 94.6 | 95.0 | 94.5 | 93.1 |
| HMMT Feb 2025 | 79.2 | 88.3 | 97.5 | 89.4 | 92.5 |
| Terminal Bench 2.0 | 42.8 | 35.2 | 54.2 | 35.7 | 46.4 |
| SWE Verified | 77.2 | 74.9 | 76.2 | 71.3 | 73.1 |
| SWE Multilingual | 68.0 | 55.3 | - | 61.1 | 70.2 |
| BrowseComp | 24.1 | 54.9 | - | -/60.2* | 51.4/67.6* |
| tau2-Bench | 84.7 | 80.2 | 85.4 | 74.3 | 80.3 |

(MiniMax M2 column omitted; * = with context management. Terminal Bench 46.4 used the Claude Code framework, p.14.)

V3.2-Speciale (p.15 Table 3, accuracy with output tokens in thousands): AIME 2025 96.0 (23k), HMMT Feb 2025 99.2 (27k), HMMT Nov 2025 94.4, IMOAnswerBench 84.5, LiveCodeBench 88.7, Codeforces 2701 (77k), GPQA 85.7, HLE 30.6. Competitions (p.15 Table 4): IMO 2025 35/42 gold; CMO 2025 102/126 gold; IOI 2025 492/600 gold (10th); ICPC WF 2025 10/12 gold (2nd).

### 6.6 V4-Pro-Max (2606.19348, p.38 Table 6)

| Benchmark | Opus-4.6 Max | GPT-5.4 xHigh | Gemini-3.1-Pro High | DS-V4-Pro Max |
|---|---|---|---|---|
| MMLU-Pro | 89.1 | 87.5 | 91.0 | 87.5 |
| GPQA Diamond | 91.3 | 93.0 | 94.3 | 90.1 |
| HLE | 40.0 | 39.8 | 44.4 | 37.7 |
| LiveCodeBench | 88.8 | - | 91.7 | 93.5 |
| Codeforces (Rating) | - | 3168 | 3052 | 3206 |
| HMMT 2026 Feb | 96.2 | 97.7 | 94.7 | 95.2 |
| SWE Verified | 80.8 | - | 80.6 | 80.6 |
| Terminal Bench 2.0 | 65.4 | 75.1 | 68.5 | 67.9 |
| BrowseComp | 83.7 | 82.7 | 85.9 | 83.4 |
| MRCR 1M | 92.9 | - | 76.3 | 83.5 |

Later cards: V4-Pro-0813 HLE 42.7 / 60.0 with tools, Terminal Bench 2.1 87.9; V4.1-Flash (max effort) GPQA 90.9, Codeforces 3471, Terminal-Bench 2.1 90.6, DeepSWE v1.1 74.2; V4.1-Flash-1017 DeepSWE v1.1 77.0, HLE 42.6 (all from HF cards).

---

## 7. Third-party cost estimates (clearly not DeepSeek's numbers)

- **SemiAnalysis, "DeepSeek Debates", 31 Jan 2025** (https://newsletter.semianalysis.com/p/deepseek-debates; Dylan Patel and co-authors): "We believe they have access to around 50,000 Hopper GPUs, which is not the same as 50,000 H100, as some have claimed." "We believe DeepSeek has access to around 10,000 of these H800s and about 10,000 H100s." "Our analysis shows that the total server CapEx for DeepSeek is ~$1.6B," "with a considerable cost of $944M associated with operating such clusters." "The $6M cost in the paper is attributed to just the GPU cost of the pre-training run," "Excluded are important pieces of the puzzle like R&D and TCO of the hardware itself." Caveat: these are estimates; the hardware is shared with High-Flyer and used for more than training, so $1.6B is not a training cost.
- **Dario Amodei, "On DeepSeek and Export Controls", Jan 2025** (https://www.darioamodei.com/post/on-deepseek-and-export-controls): "DeepSeek does not 'do for $6M' what cost US AI companies billions"; "Claude 3.5 Sonnet is a mid-sized model that cost a few $10M's to train"; V3 is "an expected point on an ongoing cost reduction curve." (Author is CEO of a competitor; say so.)
- **Epoch AI** training-compute estimate for V3: not checked: UNVERIFIED. Use the derived 3.3e24 to 3.7e24 FLOPs from section 1.4 and label it derived.
- Press for the Nature paper (Reuters-derived, e.g. ITPro, YourStory) repeated "$294,000" and a "$6 million" base-model figure. Treat press as secondary; cite NatureSI p.28.

---

## 8. Gaps and open items

- V3.1, V3.2, V4, V4.1: no GPU hours, cluster size, or dollar costs disclosed. V3.2 gives only the ">10% of pre-training cost" ratio.
- DeepSeek LLM 67B training hardware not stated in its paper.
- V2 total GPU hours not stated (only per-trillion); 1.40M is derived.
- mHC (2512.24880) and Engram (2601.07372) papers: existence verified, content not read. Not downloaded.
- Exact v1 PutnamBench count for Prover-V2 (49 vs 47) not verified.
- Nature main-text page mapping: only key items were checked against `nature-r1.pdf` (77.9% on p.2; cost pointer on p.3; dates p.1; peer review p.8). For other passages cite R1v2 pages or check the Nature PDF.
- DeepEP benchmark throughput numbers from the Feb 2025 README were not recaptured (the README was rewritten for V2/V2.5).
