# DeepSeek catalog and openness dossier

Research dossier for "DeepSeek, From the Ground Up". Compiled 2026-11-10.

How to read this file:

- Every fact has a source URL next to it. "UNVERIFIED" means I could not confirm it from a primary source; the note says what I found.
- "DeepSeek states" means the claim comes from DeepSeek's own paper, model card, repo or API docs. Third-party claims are marked as such.
- Hugging Face (HF) `createdAt` dates come from the HF API (`https://huggingface.co/api/models?author=deepseek-ai`). A repo can be created before it is made public, so I prefer the date in DeepSeek's API change log or news page when one exists.
- Local PDF copies of papers are in `~/.cache/papers/<arxiv id>.pdf`.

Primary indexes used throughout:

- HF org: https://huggingface.co/deepseek-ai (106 model repos, 2 dataset repos at time of writing)
- GitHub org: https://github.com/deepseek-ai (46 public repos at time of writing)
- API change log: https://api-docs.deepseek.com/updates
- API pricing page: https://api-docs.deepseek.com/quick_start/pricing
- arXiv API query for author "DeepSeek-AI": https://export.arxiv.org/api/query?search_query=au:%22DeepSeek-AI%22&max_results=200

---

## 1. Company background

| Fact | Value | Source | Status |
|---|---|---|---|
| Legal name | Hangzhou DeepSeek Artificial Intelligence Fundamental Technology Research Co., Ltd. (also a Beijing entity of the same name) | DeepSeek License Agreement, definition of "DeepSeek": https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL | Primary |
| Chinese name | 杭州深度求索人工智能基础技术研究有限公司 | https://en.wikipedia.org/wiki/DeepSeek | Secondary (Wikipedia) |
| Founded | 17 July 2023 | https://en.wikipedia.org/wiki/DeepSeek (infobox) | Secondary. Some outlets say May 2023 (the date Liang announced the AGI effort). DeepSeek's own HF org card says only "founded in 2023": https://huggingface.co/deepseek-ai |
| Founder and CEO | Liang Wenfeng | https://en.wikipedia.org/wiki/DeepSeek ; Liang is listed as an author and submitter on DeepSeek papers, for example the arXiv submission history of https://arxiv.org/abs/2606.19348 ("From: Wenfeng Liang") | Primary for his role as author; CEO title is secondary |
| Parent / funder | High-Flyer (幻方), a Chinese quantitative hedge fund co-founded by Liang Wenfeng | https://en.wikipedia.org/wiki/DeepSeek | Secondary. The Fire-Flyer paper thanks "our colleagues at DeepSeek-AI and High-Flyer Quant": https://arxiv.org/abs/2408.14158 (acknowledgment, p.12) |
| High-Flyer founding date | June 2015 (Wikipedia); other outlets say February 2015 or February 2016 | https://en.wikipedia.org/wiki/DeepSeek | UNVERIFIED: sources disagree; no primary filing found |
| Ownership | Wikipedia reports Liang held 84% through two shell companies as of May 2024 | https://en.wikipedia.org/wiki/DeepSeek | Secondary, UNVERIFIED |
| Stated mission | "DeepSeek (深度求索), founded in 2023, is a Chinese company dedicated to making AGI a reality. Unravel the mystery of AGI with curiosity. Answer the essential question with long-termism." | HF org card: https://huggingface.co/deepseek-ai | Primary |
| Website tagline | "DeepSeek \| Into the Unknown"; meta description: "an AI research company focused on building world-leading general artificial intelligence" | https://www.deepseek.com/en | Primary |
| Recurring phrase in releases | "Open-source spirit + Longtermism to inclusive AGI" (V3 launch, 2024-12-26); "We remain committed to longtermism, advancing steadily toward our ultimate goal of AGI" (V4 launch, 2026-04-24) | https://api-docs.deepseek.com/news/news1226 ; https://api-docs.deepseek.com/news/news260424 | Primary |

### Fire-Flyer clusters

| Cluster | Facts | Source | Status |
|---|---|---|---|
| Fire-Flyer 1 (萤火一号) | Built 2019, retired after about 1.5 years. Secondary sources give 1,100 GPUs and 200 million yuan cost. | https://en.wikipedia.org/wiki/DeepSeek (no GPU count in the current article text); 1,100 GPUs figure only in secondary summaries | UNVERIFIED for GPU count and cost. The Fire-Flyer paper mentions "insights gleaned from our Fire-Flyer 1 experiments" but gives no size: https://arxiv.org/abs/2408.14158 (p.4) |
| Fire-Flyer 2 (萤火二号) | DeepSeek states: "we deployed the Fire-Flyer 2 with 10,000 PCIe A100 GPUs, achieved performance approximating the DGX-A100 while reducing costs by half and energy consumption by 40%". About 1,250 GPU compute nodes and nearly 200 storage servers; two-layer fat-tree InfiniBand (200 Gbps); 122 switches vs 1,320 for an equivalent DGX-A100 design. "Four years of design, construction, and operation." Uses HFReduce (allreduce), HaiScale, 3FS. | Fire-Flyer AI-HPC paper (SC24): https://arxiv.org/abs/2408.14158 (abstract, p.5, p.12). PDF: `~/.cache/papers/2408.14158.pdf` | Primary |
| Fire-Flyer 2 earlier size | Wikipedia: 5,000 PCIe A100 GPUs in 625 nodes as of 2022 | https://en.wikipedia.org/wiki/DeepSeek | Secondary |
| V3/R1 training hardware | DeepSeek states V3 was trained on a cluster of 2,048 H800 GPUs; total 2.788M H800 GPU hours; "$5.576M" assuming $2 per GPU hour, "excluding the costs associated with prior research and ablation experiments" | https://arxiv.org/abs/2412.19437 (Table 1, p.5). PDF: `~/.cache/papers/2412.19437.pdf` | Primary |
| R1 incremental cost | Nature paper (17 Sep 2025) reports R1's reasoning training cost about US$294,000 on 512 H800s; also admits A100s used for smaller preparatory experiments | Nature: https://www.nature.com/articles/s41586-025-09422-z (DOI 10.1038/s41586-025-09422-z, Nature 645, 633-638). Press summary: https://www.theregister.com/2025/09/18/chinas_deepseek_ai_reasoning_research/ | Primary paper; the $294K number came to me via press coverage; check the Supplementary Information table before quoting |
| Inference fleet (Feb 2025) | DeepSeek states: V3/R1 served on H800; peak 278 nodes, average 226.75 nodes (8 H800 each) over 24 h on 27-28 Feb 2025; daily cost $87,072 at $2/GPU-hour; "theoretical" revenue $562,027/day if all tokens billed at R1 price; "cost profit margin of 545%"; DeepSeek adds that "actual revenue is substantially lower" | https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md | Primary |

### Third-party claims (keep separate from DeepSeek statements)

| Claim | Who | Source | Note |
|---|---|---|---|
| About 50,000 Hopper GPUs (10,000 H800, 10,000 H100, plus H20 orders), shared with High-Flyer; total server capex "~$1.6B"; operating cost about $944M | SemiAnalysis, 31 Jan 2025 | https://semianalysis.com/2025/01/31/deepseek-debates/ ; coverage: https://www.tomshardware.com/tech-industry/artificial-intelligence/deepseek-might-not-be-as-disruptive-as-claimed-firm-reportedly-has-50-000-nvidia-gpus-and-spent-usd1-6-billion-on-buildouts | Analyst estimate, not confirmed by DeepSeek. The report sits behind a partial paywall; figures above come from press quotes of it |
| Training data for V3 included web pages with "a significant number" of OpenAI-generated answers | Reported from the Nature paper's peer-review material | https://www.nature.com/articles/s41586-025-09422-z | Press-reported; verify wording in the paper's supplementary information |
| 2026 funding: Series A of US$7B at US$52B post-money (May 2026); IPO preparation reports (July 2026) | Press, via Wikipedia | https://en.wikipedia.org/wiki/DeepSeek | UNVERIFIED, secondary |
| Planned Inner Mongolia data center with 160,000 Huawei Ascend 950DT chips | Bloomberg, via Wikipedia | https://en.wikipedia.org/wiki/DeepSeek | UNVERIFIED. Note: DeepSeek did release Ascend 950 kernels in Sept 2026 (DeepGEMM-Ascend, DeepEP-Ascend), which is consistent with Ascend use but does not confirm the number |
| Headcount 160 (2025) | Wikipedia infobox | https://en.wikipedia.org/wiki/DeepSeek | UNVERIFIED. DeepSeek itself said in 2025 "We're a tiny team": https://github.com/deepseek-ai/open-infra-index |

---

## 2. Chronological catalog of model releases (2023 to 2026-11-10)

Notes on columns:

- Total / active params and context come from the model card unless marked "config" (from `config.json` `max_position_embeddings`) or "paper".
- Weight license: "DeepSeek License" means the DeepSeek License Agreement v1.0 (23 Oct 2023) with use restrictions (see Section 3). "MIT" and "Apache-2.0" are standard.
- Code license is the license of the GitHub repo or of the code in the HF repo.
- Paper column gives arXiv id when one exists; otherwise the PDF location.

### 2023

| Date | Model | Total | Active | Context | Family | Weights license | Code license | HF repo | Paper |
|---|---|---|---|---|---|---|---|---|---|
| 2023-10-23 to 2023-11-01 (HF repos created; GitHub repo 2023-10-20) | DeepSeek-Coder 1.3B / 5.7B-MQA / 6.7B / 33B (base, instruct) | 1.3B, 5.7B, 6.7B, 33B | dense | 16K (config 16384) | Dense, Llama-style, GQA (33B) | DeepSeek License (supports commercial use) | MIT | https://huggingface.co/deepseek-ai/deepseek-coder-33b-base | arXiv 2401.14196 https://arxiv.org/abs/2401.14196 |
| 2023-11-29 | DeepSeek LLM 7B / 67B (base, chat) | 7B, 67B | dense | 4K (config 4096) | Dense, Llama-style; 67B uses GQA; 2T tokens | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/deepseek-llm-67b-base | arXiv 2401.02954 https://arxiv.org/abs/2401.02954 |

Public announcement date of DeepSeek-Coder (often given as 2 Nov 2023): UNVERIFIED; I only have the repo creation dates above (HF API, GitHub API).

### 2024

| Date | Model | Total | Active | Context | Family | Weights license | Code license | HF repo | Paper |
|---|---|---|---|---|---|---|---|---|---|
| 2024-01-09 (HF; arXiv 2024-01-11) | DeepSeekMoE 16B (base, chat) | 16.4B | about 2.8B (UNVERIFIED: the paper and card say it matches LLaMA2 7B, which has about 2.5x the activated params; the exact 2.8B figure is not on the card) | 4K | MoE: 64 routed + 2 shared experts, top-6 (config) | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/deepseek-moe-16b-base | arXiv 2401.06066 https://arxiv.org/abs/2401.06066 |
| 2024-01-25 | DeepSeek-Coder 7B v1.5 (base, instruct) | 7B | dense | UNVERIFIED | Dense, continued from DeepSeek LLM 7B | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/deepseek-coder-7b-base-v1.5 | none |
| 2024-02-05 | DeepSeekMath 7B (base, instruct, RL) | 7B | dense | 4K (config) | Dense; introduces GRPO | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/deepseek-math-7b-rl | arXiv 2402.03300 https://arxiv.org/abs/2402.03300 |
| 2024-03-07 (HF; arXiv 2024-03-08) | DeepSeek-VL 1.3B / 7B (base, chat) | 1.3B, 7B (+vision encoder) | dense | config 16384 (LLM side) | VLM on DeepSeek LLM | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat | arXiv 2403.05525 https://arxiv.org/abs/2403.05525 |
| 2024-05 (arXiv 2024-05-07; HF repo created 2024-04-22) | DeepSeek-V2 (base, chat) | 236B | 21B | 128K | MoE + MLA (first MLA model); 160 routed + 2 shared, top-6 | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V2 | arXiv 2405.04434 https://arxiv.org/abs/2405.04434 |
| 2024-05-15 (HF) | DeepSeek-V2-Lite (base, chat) | 16B | 2.4B | 32K | MoE + MLA | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite | arXiv 2405.04434 |
| 2024-05-17 | DeepSeek-V2-0517 (API only update) | 236B | 21B | 128K | as V2 | API only, no separate weights | n/a | none | n/a. Source: https://api-docs.deepseek.com/updates |
| 2024-05-23 (arXiv); HF 2024-08-16 | DeepSeek-Prover (V1) | 7B | dense | UNVERIFIED | Dense, from DeepSeekMath 7B | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1 ; dataset https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1 | arXiv 2405.14333 https://arxiv.org/abs/2405.14333 |
| 2024-06-14 (API change log; arXiv 2024-06-17) | DeepSeek-Coder-V2 and Coder-V2-Lite (base, instruct) | 236B; 16B | 21B; 2.4B | 128K | MoE + MLA, continued from a V2 checkpoint | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct | arXiv 2406.11931 https://arxiv.org/abs/2406.11931 (the HF repo is tagged with 2401.06066 by mistake) |
| 2024-06-28 (API); HF 2024-07-18 | DeepSeek-V2-Chat-0628 | 236B | 21B | 128K | as V2 | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V2-Chat-0628 | arXiv 2405.04434 |
| 2024-07-04 | ESFT checkpoints (13 repos, expert-specialized fine-tuning on V2-Lite) | 16B | 2.4B | as V2-Lite | MoE | no license tag on HF | MIT | https://huggingface.co/deepseek-ai/ESFT-vanilla-lite | arXiv 2407.01906 https://arxiv.org/abs/2407.01906 |
| 2024-07-24 (API); HF 2024-09-05 | DeepSeek-Coder-V2-Instruct-0724 | 236B | 21B | 128K | as Coder-V2 | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct-0724 | arXiv 2406.11931 |
| 2024-08-15 | DeepSeek-Prover-V1.5 (base, SFT, RL) | 7B | dense | 4K (config) | Dense, from DeepSeekMath | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1.5-RL | arXiv 2408.08152 https://arxiv.org/abs/2408.08152 |
| 2024-09-05 | DeepSeek-V2.5 (merge of V2-Chat and Coder-V2) | 236B | 21B | 128K | MoE + MLA | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V2.5 | arXiv 2405.04434. Release: https://api-docs.deepseek.com/updates |
| 2024-10-18 (HF; arXiv 2024-10-17) | Janus 1.3B | about 2.09B stored params (HF safetensors) | dense | config 16384 | Unified multimodal understanding + generation | HF tag says MIT, but the card says "subject to DeepSeek Model License" | MIT | https://huggingface.co/deepseek-ai/Janus-1.3B | arXiv 2410.13848 https://arxiv.org/abs/2410.13848 |
| 2024-11-12 | JanusFlow 1.3B | about 2.05B stored params | dense | UNVERIFIED | AR + rectified flow | same as Janus (card: DeepSeek Model License) | MIT | https://huggingface.co/deepseek-ai/JanusFlow-1.3B | arXiv 2411.07975 https://arxiv.org/abs/2411.07975 |
| 2024-11-20 | DeepSeek-R1-Lite-Preview (web only) | UNVERIFIED | UNVERIFIED | n/a | reasoning preview | no weights released | n/a | none | none. Source: https://api-docs.deepseek.com/news/news1120 |
| 2024-12-10 | DeepSeek-V2.5-1210 ("the V2.5 series comes to an end") | 236B | 21B | 128K | MoE + MLA | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V2.5-1210 | Source: https://api-docs.deepseek.com/news/news1210 |
| 2024-12-13 | DeepSeek-VL2 Tiny / Small / VL2 | 3.37B / 16.1B / 27.5B (HF safetensors totals) | 1.0B / 2.8B / 4.5B (paper) | 4K (config) | MoE VLM (DeepSeekMoE LLM + SigLIP) | DeepSeek License | MIT | https://huggingface.co/deepseek-ai/deepseek-vl2 | arXiv 2412.10302 https://arxiv.org/abs/2412.10302 |
| 2024-12-26 | DeepSeek-V3 and V3-Base | 671B (+14B MTP module = 685B on HF) | 37B | 128K | MoE (256 routed + 1 shared, top-8) + MLA + MTP; FP8 training; aux-loss-free balancing | DeepSeek License (the HF repo still ships LICENSE-MODEL = DeepSeek License) | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3 | arXiv 2412.19437 https://arxiv.org/abs/2412.19437. Release: https://api-docs.deepseek.com/news/news1226 |

### 2025

| Date | Model | Total | Active | Context | Family | Weights license | Code license | HF repo | Paper |
|---|---|---|---|---|---|---|---|---|---|
| 2025-01-20 | DeepSeek-R1 and R1-Zero | 671B | 37B | 128K | V3 architecture; RL-trained reasoning | MIT (first MIT-licensed flagship weights) | MIT | https://huggingface.co/deepseek-ai/DeepSeek-R1 | arXiv 2501.12948 https://arxiv.org/abs/2501.12948 ; Nature 645, 633-638 (2025) https://www.nature.com/articles/s41586-025-09422-z |
| 2025-01-20 | R1-Distill-Qwen 1.5B / 7B / 14B / 32B, R1-Distill-Llama 8B / 70B | 1.5B to 70B | dense | per base model | Dense; SFT of Qwen2.5 / Llama 3.x on 800K R1 samples | MIT for DeepSeek's part, but card notes Qwen bases are Apache-2.0 and Llama bases keep Llama 3.1 / 3.3 licenses | MIT | https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B | arXiv 2501.12948 |
| 2025-01-27 (HF repo 2025-01-26 UTC; arXiv 2025-01-29) | Janus-Pro 1B / 7B | 1B / 7B class (built on DeepSeek-LLM-1.5B / 7B base) | dense | config 16384 | Unified multimodal | HF tag MIT; card says code MIT, model "subject to DeepSeek Model License" | MIT | https://huggingface.co/deepseek-ai/Janus-Pro-7B | arXiv 2501.17811 https://arxiv.org/abs/2501.17811 |
| 2025-03-24 | DeepSeek-V3-0324 | 671B | 37B | 128K | as V3 | MIT ("Models are now released under the MIT License, just like DeepSeek-R1!") | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3-0324 | arXiv 2412.19437. Release: https://api-docs.deepseek.com/news/news250325 |
| 2025-04-30 | DeepSeek-Prover-V2 7B and 671B | 7B; 671B | dense; 37B | 7B: 32K per card (config 65536); 671B: as V3 | 7B from Prover-V1.5-Base; 671B from V3-Base | Mixed: HF LICENSE file is MIT, but the card says "subject to the Model License (LICENSE-MODEL)", and the GitHub LICENSE-MODEL is the DeepSeek License. Treat as DeepSeek License until clarified | GitHub shows NOASSERTION | https://huggingface.co/deepseek-ai/DeepSeek-Prover-V2-671B | arXiv 2504.21801 https://arxiv.org/abs/2504.21801 |
| 2025-05-28 | DeepSeek-R1-0528 | 671B | 37B | 128K | as R1 | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-R1-0528 | arXiv 2501.12948. Release: https://api-docs.deepseek.com/news/news250528 |
| 2025-05-29 | DeepSeek-R1-0528-Qwen3-8B | 8B | dense | per Qwen3-8B | Distill into Qwen3-8B | MIT (HF tag) | MIT | https://huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B | arXiv 2501.12948 |
| 2025-08-21 (Base repo 2025-08-19) | DeepSeek-V3.1 and V3.1-Base | 671B | 37B | 128K | V3 + 840B tokens continued pretraining for long context; hybrid think / non-think | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3.1 | no new arXiv paper (card cites 2412.19437). Release: https://api-docs.deepseek.com/news/news250821 |
| 2025-09-22 | DeepSeek-V3.1-Terminus | 671B | 37B | 128K | as V3.1 | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3.1-Terminus | none. Release: https://api-docs.deepseek.com/news/news250922 |
| 2025-09-29 | DeepSeek-V3.2-Exp and V3.2-Exp-Base | 671B | 37B | 128K (API); config 163840 | V3.1-Terminus + DeepSeek Sparse Attention (DSA, lightning indexer, top-k 2048) | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp | Tech report PDF (no arXiv): https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf . Release: https://api-docs.deepseek.com/news/news250929 |
| 2025-10-17 (GitHub + HF repos; arXiv 2025-10-21) | DeepSeek-OCR | about 3.34B stored (HF); paper: DeepEncoder ~380M + 3B MoE decoder | 570M active in decoder | 8K (config) | Vision encoder + DeepSeek3B-MoE-A570M | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-OCR | arXiv 2510.18234 https://arxiv.org/abs/2510.18234 |
| 2025-11-27 | DeepSeekMath-V2 | 671B class (HF safetensors 685.4B incl. MTP) | 37B (UNVERIFIED on card; architecture is V3.2-Exp) | config 163840 | Built on V3.2-Exp-Base; self-verifying proofs | Apache-2.0 | Apache-2.0 | https://huggingface.co/deepseek-ai/DeepSeek-Math-V2 | arXiv 2511.22570 https://arxiv.org/abs/2511.22570 |
| 2025-12-01 | DeepSeek-V3.2 and V3.2-Speciale | 671B | 37B | 128K (API) | Same structure as V3.2-Exp (DSA); Speciale is a high-compute reasoning variant without tool calls | MIT (both) | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V3.2 ; https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale | arXiv 2512.02556 https://arxiv.org/abs/2512.02556 . Release: https://api-docs.deepseek.com/news/news251201 |

### 2026

| Date | Model | Total | Active | Context | Family | Weights license | Code license | HF repo | Paper |
|---|---|---|---|---|---|---|---|---|---|
| 2026-01-12 | Engram (code and demo only, no released weights found) | paper scales Engram to 27B | n/a | n/a | Conditional memory module (hashed N-gram lookup) | no weights | Apache-2.0 | none; code https://github.com/deepseek-ai/Engram | arXiv 2601.07372 https://arxiv.org/abs/2601.07372 |
| 2026-01-27 (arXiv 2026-01-28) | DeepSeek-OCR 2 | about 3.39B stored (HF) | UNVERIFIED (decoder as OCR v1) | 8K (config) | DeepEncoder V2 replaces CLIP with a Qwen2-500M-style LM encoder ("Visual Causal Flow") | Apache-2.0 | Apache-2.0 | https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 | arXiv 2601.20552 https://arxiv.org/abs/2601.20552 |
| 2026-04-24 | DeepSeek-V4 Preview: V4-Pro, V4-Flash, V4-Pro-Base, V4-Flash-Base | Pro 1.6T; Flash 284B | Pro 49B; Flash 13B | 1M | MoE with hybrid attention (Compressed Sparse Attention + Heavily Compressed Attention), mHC residuals, Muon optimizer, >32T tokens; FP4 experts + FP8 elsewhere (instruct) | MIT (Base repos ship an MIT LICENSE file but no model card) | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro ; https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash ; collection https://huggingface.co/collections/deepseek-ai/deepseek-v4 | arXiv 2606.19348 https://arxiv.org/abs/2606.19348 (arXiv page says "Submitted on 26 Apr 2026" even though the id prefix is 2606). Release: https://api-docs.deepseek.com/news/news260424 |
| 2026-06-27 to 06-28 | V4-Pro-DSpark, V4-Flash-DSpark (same checkpoints + speculative decoding module) and 12 small draft checkpoints (dspark_*, dflash_*, eagle3_* on Qwen3 4B/8B/14B and Gemma4 12B) | as V4 | as V4 | 1M | DSpark speculative decoding | MIT (V4 DSpark); no license tag on draft repos | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark | arXiv 2607.05147 https://arxiv.org/abs/2607.05147 ; code https://github.com/deepseek-ai/DeepSpec |
| 2026-07-31 | DeepSeek-V4-Flash-0731 (official V4-Flash, "only re-post-trained", ships with DSpark module) | 284B | 13B | 1M | as V4-Flash | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731 | arXiv 2606.19348. Change log: https://api-docs.deepseek.com/updates |
| 2026-08-13 | DeepSeek-V4-Pro-0813 (official V4-Pro GA) | 1.6T | 49B | 1M | as V4-Pro + DSpark | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813 | arXiv 2606.19348. Release: https://api-docs.deepseek.com/news/news260813 |
| 2026-08-21 | DeepSeek-V4-Flash-Vision-Exp (first multimodal V4) | V4-Flash + vision modules; exact total UNVERIFIED (HF reports 304.6B stored elements, but FP4 weights are packed so this is not a clean param count) | UNVERIFIED | 1M (config) | V4-Flash + vision encoder, continued training | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp | none. Release: https://api-docs.deepseek.com/news/news260821 |
| 2026-09-10 | DeepSeek-V4.1-Flash | 552B "backbone" plus Engram conditional memory of 196B | 8B in prefill, 16B in decode | 1M | New "Causal Encoder-Decoder" (20-layer causal encoder + 20-layer decoder), CSA2, Single-Pass mHC, Engram, DSpark, native vision (DeepSeek-ViT); 45T multimodal tokens; 1 shared + 384 routed experts, top-6 | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash | arXiv 2609.19969 https://arxiv.org/abs/2609.19969 ; PDF on HF https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf . Release: https://api-docs.deepseek.com/news/news260910 |
| 2026-10-17 | DeepSeek-V4.1-Flash-1017 (minor update) | as V4.1-Flash | as V4.1-Flash | 1M | as V4.1-Flash | MIT | MIT | https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017 | none new. Release: https://api-docs.deepseek.com/news/news261017 |

Not released as weights (as of 2026-11-10):

- DeepSeek-V4.1-Pro: the V4.1-Flash news says V4-Pro requests will route to V4.1-Flash "until V4.1-Pro launches", which implies a V4.1-Pro is planned. No release found. Source: https://api-docs.deepseek.com/news/news260910
- "Thinking with Visual Primitives" (around 30 Apr 2026): press and alphaXiv describe a DeepSeek paper with a 284B / 13B MoE vision model and a GitHub repo, but `https://github.com/deepseek-ai/Thinking-with-Visual-Primitives` returns 404 today and I found no arXiv entry. Some blogs say it was pulled. UNVERIFIED. Coverage: https://www.mindstudio.ai/blog/what-is-visual-primitives-reasoning-deepseek
- The "V4.1-Flash-Base" appears in the V4.1-Flash card's base-model table, but no Base repo exists on HF.

---

## 3. What is open: matrix per major model

### 3.1 Licenses and when MIT started

- DeepSeek License Agreement v1.0 (23 Oct 2023) is a responsible-AI license: it permits commercial use, but Attachment A forbids use "for military use in any way", to harm minors, to generate verifiably false information to harm others, "inappropriate content subject to applicable regulatory requirements", unauthorized PII, defamation, fully automated decisions that adversely affect legal rights, discrimination, and exploiting vulnerable groups. Users must pass these restrictions down to their own users. Governing law is PRC law, courts of Hangzhou. Source: https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL
- First MIT weights: Janus-1.3B carries an MIT tag on HF (2024-10-18), but its card still says the model is "subject to DeepSeek Model License"; I treat this as a labeling inconsistency. Sources: https://huggingface.co/deepseek-ai/Janus-1.3B
- First clean MIT flagship weights: DeepSeek-R1 (2025-01-20): "This code repository and the model weights are licensed under the MIT License ... supports commercial use, allow for any modifications and derivative works, including, but not limited to, distillation." Source: https://huggingface.co/deepseek-ai/DeepSeek-R1
- V3 line switched with V3-0324 (2025-03-24): "Models are now released under the MIT License, just like DeepSeek-R1!" Source: https://api-docs.deepseek.com/news/news250325 . The original DeepSeek-V3 and V3-Base repos still ship the DeepSeek License as LICENSE-MODEL today: https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL
- Apache-2.0 appears from late 2025: DeepSeekMath-V2 (https://huggingface.co/deepseek-ai/DeepSeek-Math-V2), DeepSeek-OCR-2 (https://huggingface.co/deepseek-ai/DeepSeek-OCR-2), Engram code (https://github.com/deepseek-ai/Engram), DeepInfer (https://github.com/deepseek-ai/DeepInfer).
- All V3.1, V3.2, V4 and V4.1 weights are MIT. Sources: each HF card listed in Section 2.

### 3.2 Matrix

Legend: "Inference code" means code shipped by DeepSeek to run the model (reference PyTorch or kernels), not third-party engines. "Training code" means pretraining or RL code for the model itself.

| Model | Weights | Weight license and restrictions | Inference code | Training code | Training data | Tech report |
|---|---|---|---|---|---|---|
| DeepSeek LLM 7B/67B | Yes (base, chat) | DeepSeek License, use restrictions | HF Transformers usage only (Llama arch) https://github.com/deepseek-ai/DeepSeek-LLM | No (repo has only images of loss curves) | Described only (2T tokens EN/ZH) | arXiv 2401.02954 |
| DeepSeek-Coder | Yes | DeepSeek License | Usage examples https://github.com/deepseek-ai/DeepSeek-Coder | Partial: a fine-tuning script `finetune/finetune_deepseekcoder.py` (DeepSpeed) for downstream SFT; no pretraining code | Described only (2T tokens, 87% code) | arXiv 2401.14196 |
| DeepSeekMoE 16B | Yes | DeepSeek License | HF remote code https://github.com/deepseek-ai/DeepSeek-MoE | No | Described only | arXiv 2401.06066 |
| DeepSeekMath 7B | Yes | DeepSeek License | Usage + eval https://github.com/deepseek-ai/DeepSeek-Math | No (GRPO described, not released) | Described only (DeepSeekMath corpus, 120B math tokens per paper) | arXiv 2402.03300 |
| DeepSeek-V2 / V2-Lite / V2.5 | Yes | DeepSeek License | HF remote code (`modeling_deepseek.py`) https://github.com/deepseek-ai/DeepSeek-V2 | No | Described only (8.1T tokens) | arXiv 2405.04434 |
| Coder-V2 | Yes | DeepSeek License | Usage https://github.com/deepseek-ai/DeepSeek-Coder-V2 | No | Described only | arXiv 2406.11931 |
| ESFT | Yes (13 adapters/checkpoints) | no license tag on HF | Yes https://github.com/deepseek-ai/ESFT | Yes: `esft.py`, `scripts/train.sh`, `scripts/train_ep.sh` plus small task datasets in `datasets/train` | Partly open (task datasets for the ESFT experiments) | arXiv 2407.01906 |
| Prover V1 / V1.5 | Yes | DeepSeek License | Yes (search/eval code) https://github.com/deepseek-ai/DeepSeek-Prover-V1.5 | No | Prover-V1 synthetic Lean 4 dataset released: https://huggingface.co/datasets/deepseek-ai/DeepSeek-Prover-V1 (the card describes 8M formal statements; check how much of it is in the HF repo) | arXiv 2405.14333, 2408.08152 |
| Janus / JanusFlow / Janus-Pro | Yes | Mixed tags (see 3.1) | Yes https://github.com/deepseek-ai/Janus | No training code found in repo tree | Described only | arXiv 2410.13848, 2411.07975, 2501.17811 |
| DeepSeek-VL / VL2 | Yes | DeepSeek License | Yes https://github.com/deepseek-ai/DeepSeek-VL2 | No | Described only | arXiv 2403.05525, 2412.10302 |
| DeepSeek-V3 / V3-0324 | Yes (FP8) | V3: DeepSeek License; V3-0324: MIT | Yes: reference PyTorch inference (`inference/model.py`, `generate.py`, `kernel.py`, FP8 to BF16 converter) https://github.com/deepseek-ai/DeepSeek-V3/tree/main/inference ; infra kernels (FlashMLA, DeepEP, DeepGEMM) | No. Repo has no training code. Training framework is internal ("HAI-LLM" per paper) | Described only (14.8T tokens) | arXiv 2412.19437 ; ISCA 2025 paper arXiv 2505.09343 |
| DeepSeek-R1 / R1-Zero / distills | Yes | MIT; distills inherit Qwen Apache-2.0 / Llama licenses | Same as V3 (repo has only README, PDF, license) https://github.com/deepseek-ai/DeepSeek-R1 | No RL code. Nature paper states it released "data samples" with the models: https://www.nature.com/articles/s41586-025-09422-z (UNVERIFIED where those samples live) | Not released (800K SFT samples described only) | arXiv 2501.12948 ; Nature 2025 |
| Prover-V2 | Yes | Mixed (see Section 2) | Usage https://github.com/deepseek-ai/DeepSeek-Prover-V2 | No | ProverBench eval set released: https://huggingface.co/datasets/deepseek-ai/DeepSeek-ProverBench ; miniF2F solutions zip | arXiv 2504.21801 |
| V3.1 / V3.1-Terminus | Yes | MIT | Terminus repo has `inference/`; V3.1 relies on V3 repo + DeepGEMM | No | Described only (840B extension tokens) | None new |
| V3.2-Exp / V3.2 / Speciale | Yes | MIT | Yes: `inference/` in HF repo; TileLang and CUDA kernels (FlashMLA PR #98, DeepGEMM PR #200, TileLang examples) https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp ; competition submissions released for IMO/IOI/ICPC/CMO 2025 | No | Described only | V3.2-Exp PDF on GitHub; V3.2 arXiv 2512.02556 |
| DeepSeek-OCR / OCR-2 | Yes | MIT / Apache-2.0 | Yes (vLLM + Transformers) https://github.com/deepseek-ai/DeepSeek-OCR | No training code found | Described only | arXiv 2510.18234, 2601.20552 |
| DeepSeekMath-V2 | Yes | Apache-2.0 | Yes (`inference/` in HF repo) | No | Described only | arXiv 2511.22570 |
| DeepSeek-V4 (Preview, -0731, -0813, DSpark) | Yes, including Base models | MIT | Yes: `inference/` and `encoding/` folders in HF repos (no Jinja chat template; Python encoder instead); kernels in TileKernels, DeepGEMM "mega_moe" backend referenced in vLLM recipe | Partial: DeepSpec releases full training code for the DSpark draft model, not for V4 itself https://github.com/deepseek-ai/DeepSpec | Described only (>32T tokens) | arXiv 2606.19348 |
| DeepSeek-V4.1-Flash / -1017 | Yes (no Base repo found) | MIT | Yes: `inference/`, `encoding/`, `evaluation/` in HF repo; DeepInfer production engine (supports V4.1-Flash only) https://github.com/deepseek-ai/DeepInfer ; prompt toolkit https://github.com/deepseek-ai/deepseek-recipe | No | Described only (45T multimodal tokens) | arXiv 2609.19969 |

Bottom line for writers: DeepSeek has never released pretraining code, RL training code for its LLMs, or pretraining data. Verified by listing the repo trees of DeepSeek-LLM, DeepSeek-V2, DeepSeek-V3, DeepSeek-R1, DeepSeek-V3.2-Exp, Janus, VL2, OCR, Math, Math-V2, Prover-V1.5, Prover-V2 via the GitHub API (search for "train|finetune|sft|grpo" in paths). The only training code it has released is narrow: the Coder fine-tuning script, ESFT, and DeepSpec (speculative-decoding drafters). The weights are open; the recipe is described in papers; the infrastructure kernels and systems are open (next table).

### 3.3 Infrastructure libraries

Open Source Week, 24 to 28 Feb 2025 (DeepSeek states: "we'll open-source 5 repos, one daily drop"), source https://github.com/deepseek-ai/open-infra-index :

| Day | Repo | What it is (DeepSeek's words, condensed) | License | Created |
|---|---|---|---|---|
| 1 | FlashMLA https://github.com/deepseek-ai/FlashMLA | MLA decoding kernel for Hopper; paged KV cache block 64; "3000 GB/s memory-bound, BF16 580 TFLOPS compute-bound on H800"; later gained sparse kernels for V3.2 (PR #98) | MIT | 2025-02-21 |
| 2 | DeepEP https://github.com/deepseek-ai/DeepEP | Expert-parallel all-to-all communication library (NVLink + RDMA, FP8 dispatch, low-latency decode kernels) | MIT | 2025-02-17 |
| 3 | DeepGEMM https://github.com/deepseek-ai/DeepGEMM | FP8 GEMM for dense and MoE, JIT-compiled, "up to 1350+ FP8 TFLOPS on Hopper" | MIT | 2025-02-13 |
| 4 | DualPipe https://github.com/deepseek-ai/DualPipe | Bidirectional pipeline-parallel schedule used in V3/R1 training | MIT | 2025-02-26 |
| 4 | EPLB https://github.com/deepseek-ai/EPLB | Expert-parallel load balancer for V3/R1 | MIT | 2025-02-26 |
| 4 | profile-data https://github.com/deepseek-ai/profile-data | Profiling traces of training and inference compute-communication overlap | no license file | 2025-02-26 |
| 5 | 3FS https://github.com/deepseek-ai/3FS | Fire-Flyer File System: "6.6 TiB/s aggregate read throughput in a 180-node cluster" | MIT | 2025-02-27 |
| 5 | smallpond https://github.com/deepseek-ai/smallpond | Data processing on DuckDB + 3FS | MIT | 2025-02-24 |
| 6 | Inference system overview (doc, not code) https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md | Prefill/decode disaggregation, cross-node EP, the 545% "theoretical" margin figure | CC0 (index repo) | 2025-02-28/03-01 |

Later infra and tooling repos (GitHub API, `created_at`):

| Date | Repo | What | License |
|---|---|---|---|
| 2025-04 | "The Path to Open-Sourcing the DeepSeek Inference Engine" https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md | DeepSeek states its engine is "an early fork of vLLM"; it decided not to release it and instead upstream features to vLLM/SGLang; training framework "relies on PyTorch" | CC0 |
| 2025-09-29 | TileLang + CUDA kernels for V3.2-Exp (in tile-ai/tilelang examples, FlashMLA, DeepGEMM PRs) https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32 | DSA kernels | per host repo |
| 2025-11-19 | LPLB https://github.com/deepseek-ai/LPLB | Linear-programming expert-parallel load balancer, "early research stage" | MIT |
| 2026-04-22 | TileKernels https://github.com/deepseek-ai/TileKernels | TileLang kernels for MoE routing, FP8/FP4 quant, Engram, mHC; Ascend backend added 2026-09-30; "All kernels have been used in our internal training and inference" | MIT |
| 2026-05-28 | clangd-ascend https://github.com/deepseek-ai/clangd-ascend | Clangd for Ascend code | NOASSERTION |
| 2026-06-26 | DeepSpec https://github.com/deepseek-ai/DeepSpec | Training + evaluation of draft models (DSpark, DFlash, Eagle3) | MIT |
| 2026-08-13 | deepseek-harness https://github.com/deepseek-ai/deepseek-harness | Open-source agent harness (`dsh`), developer preview | MIT |
| 2026-09-08 | DeepJIT https://github.com/deepseek-ai/DeepJIT | Header-only C++20 JIT runtime for CUDA and Ascend | MIT |
| 2026-09-09 | DeepSelect https://github.com/deepseek-ai/DeepSelect | Top-K kernels for DSA and samplers (CUDA + Ascend) | MIT |
| 2026-09-10 | deepseek-recipe https://github.com/deepseek-ai/deepseek-recipe | Rust + Python prompt encoding / response parsing for V4 and V4.1 | MIT |
| 2026-09-29 | DeepGEMM-Ascend https://github.com/deepseek-ai/DeepGEMM-Ascend | DeepGEMM port for Ascend 950 | MIT |
| 2026-09-30 | DeepEP-Ascend https://github.com/deepseek-ai/DeepEP-Ascend | DeepEP for Ascend 950DT | no license detected by GitHub |
| 2026-10-16 | DeepInfer https://github.com/deepseek-ai/DeepInfer | "the high-performance inference framework that powers DeepSeek's large-scale production services"; open release supports offline inference of V4.1-Flash | Apache-2.0 |
| 2026-10-31 | dsh-superagent-prototype https://github.com/deepseek-ai/dsh-superagent-prototype | Research prototype proactive coding agent for DeepSeek Harness | MIT |

Note the reversal: in April 2025 DeepSeek said it would not open-source its inference engine; in October 2026 it released DeepInfer, described as the production engine (open release limited to offline inference). Sources: the two READMEs above.

---

## 4. API product facts

### 4.1 Model names served over time

| From | `deepseek-chat` | `deepseek-reasoner` | Other names | Source |
|---|---|---|---|---|
| 2024-05-17 | V2-0517 | n/a | `deepseek-coder` | https://api-docs.deepseek.com/updates |
| 2024-06-14 / 06-28 / 07-24 | V2-0628 | n/a | `deepseek-coder` = Coder-V2-0614, then -0724 | same |
| 2024-09-05 | V2.5 (both `deepseek-chat` and `deepseek-coder`) | n/a | | same |
| 2024-12-10 | V2.5-1210 | n/a | | same |
| 2024-12-26 | V3 | n/a | | same |
| 2025-01-20 | V3 | R1 | | same |
| 2025-03-24 | V3-0324 | R1 | | same |
| 2025-05-28 | V3-0324 | R1-0528 | | same |
| 2025-08-21 | V3.1 non-thinking | V3.1 thinking | | same |
| 2025-09-22 | V3.1-Terminus | V3.1-Terminus thinking | | same |
| 2025-09-29 | V3.2-Exp | V3.2-Exp thinking | temporary V3.1-Terminus endpoint until 2025-10-15 | https://api-docs.deepseek.com/news/news250929 |
| 2025-12-01 | V3.2 | V3.2 thinking | V3.2-Speciale at temporary base_url until 2025-12-15 15:59 UTC | https://api-docs.deepseek.com/news/news251201 |
| 2026-04-24 | routes to V4-Flash non-thinking | routes to V4-Flash thinking | new `deepseek-v4-flash`, `deepseek-v4-pro`; chat/reasoner retired after 2026-07-24 15:59 UTC | https://api-docs.deepseek.com/news/news260424 |
| 2026-07-31 | retired (after 07-24) | retired | `deepseek-v4-flash` = V4-Flash-0731 | https://api-docs.deepseek.com/updates |
| 2026-08-13 | | | `deepseek-v4-pro` = V4-Pro-0813; reasoning_effort low / high / max; native OpenAI Responses API | https://api-docs.deepseek.com/news/news260813 |
| 2026-08-21 | | | `deepseek-v4-flash-vision-exp` (images up to 384 tokens each); Files API | https://api-docs.deepseek.com/news/news260821 |
| 2026-09-10 | | | `deepseek-flash` = V4.1-Flash; V4-Flash and Vision-Exp retired (names route to V4.1-Flash). DeepSeek first said V4-Pro would route to V4.1-Flash from 2026-09-14, then reversed: "we have decided to continue providing API services for DeepSeek V4 Pro after September 14, 2026" | https://api-docs.deepseek.com/news/news260910 ; https://api-docs.deepseek.com/updates |
| 2026-10-17 (current) | | | `deepseek-flash` = V4.1-Flash-1017; `deepseek-v4-pro` = V4-Pro-0813; Search and Fetch APIs | https://api-docs.deepseek.com/quick_start/pricing |

Formats: OpenAI-compatible base_url `https://api.deepseek.com`; Anthropic-compatible `https://api.deepseek.com/anthropic` (announced with V3.1, 2025-08-21: https://api-docs.deepseek.com/news/news250821 ); OpenAI Responses API from 2026-07-31 / 08-13.

### 4.2 Pricing history (USD per 1M tokens)

All snapshots are Wayback Machine captures of https://api-docs.deepseek.com/quick_start/pricing unless another source is given.

| Effective | Model | Input, cache hit | Input, cache miss | Output | Context / max output | Source |
|---|---|---|---|---|---|---|
| 2024-08-02 | Context caching on disk launched: cache hit $0.014 | $0.014 | | | | https://api-docs.deepseek.com/news/news0802 |
| as of 2024-11-07 | `deepseek-chat` (V2.5) | $0.014 | $0.14 | $0.28 | 128K / 4K (8K beta) | https://web.archive.org/web/20241107075055/https://api-docs.deepseek.com/quick_start/pricing |
| 2024-12-26 to 2025-02-08 16:00 UTC | `deepseek-chat` (V3), promo "same as V2" | $0.014 | $0.14 | $0.28 | 64K / 8K | https://api-docs.deepseek.com/news/news1226 ; https://web.archive.org/web/20250122193157/https://api-docs.deepseek.com/quick_start/pricing |
| from 2025-02-08 16:00 UTC | `deepseek-chat` (V3) | $0.07 | $0.27 | $1.10 | 64K / 8K | same two sources |
| from 2025-01-20 | `deepseek-reasoner` (R1), not in promo | $0.14 | $0.55 | $2.19 | 64K, max CoT 32K, output 8K | https://web.archive.org/web/20250122193157/https://api-docs.deepseek.com/quick_start/pricing |
| by 2025-02-26 (announcement date UNVERIFIED) | Off-peak discount 16:30 to 00:30 UTC daily: chat 50% off ($0.035 / $0.135 / $0.550), reasoner 75% off (same $0.035 / $0.135 / $0.550) | | | | | https://web.archive.org/web/20250226111135/https://api-docs.deepseek.com/quick_start/pricing |
| from 2025-09-05 16:00 UTC (off-peak discounts ended) | V3.1 (both modes, one price) | $0.07 | $0.56 | $1.68 | 128K; output chat 8K max, reasoner 64K max | https://api-docs.deepseek.com/news/news250821 ; https://web.archive.org/web/20250906202357/https://api-docs.deepseek.com/quick_start/pricing |
| from 2025-09-29 ("50%+" cut) | V3.2-Exp | $0.028 | $0.28 | $0.42 | 128K | https://api-docs.deepseek.com/news/news250929 ; https://web.archive.org/web/20251004112357/https://api-docs.deepseek.com/quick_start/pricing |
| from 2025-12-01 | V3.2 and V3.2-Speciale (same price; Speciale output up to 128K) | $0.028 | $0.28 | $0.42 | 128K | https://web.archive.org/web/20251203072137/https://api-docs.deepseek.com/quick_start/pricing |
| from 2026-04-24 | `deepseek-v4-flash` | $0.028 | $0.14 | $0.28 | 1M / 384K | https://web.archive.org/web/20260425165822/https://api-docs.deepseek.com/quick_start/pricing |
| from 2026-04-24 | `deepseek-v4-pro` list price (75% off until 2026-05-05 15:59 UTC: $0.03625 / $0.435 / $0.87) | $0.145 | $1.74 | $3.48 | 1M / 384K | same |
| from 2026-04-26 12:15 UTC | Cache-hit price cut to 1/10 for all models: Flash $0.0028, Pro $0.0145 (discounted $0.003625); Pro 75% discount extended to 2026-05-31 15:59 UTC | | | | | https://web.archive.org/web/20260501050639/https://api-docs.deepseek.com/quick_start/pricing |
| from 2026-08-16 16:00 UTC | Peak / off-peak pricing introduced. V4-Flash-0731 peak | $0.014 | $0.44 | $1.32 | 1M / 384K | https://api-docs.deepseek.com/news/news260813 ; https://web.archive.org/web/20260817102957/https://api-docs.deepseek.com/quick_start/pricing |
| same | V4-Flash-0731 off-peak | $0.007 | $0.22 | $0.66 | | same |
| same | V4-Pro-0813 peak | $0.044 | $1.32 | $3.96 | | same |
| same | V4-Pro-0813 off-peak | $0.022 | $0.66 | $1.98 | | same |
| from 2026-09-10 04:00 UTC | `deepseek-flash` (V4.1-Flash) peak | $0.006 | $0.30 | $1.20 | 1M / 384K | https://api-docs.deepseek.com/news/news260910 ; https://web.archive.org/web/20260911165508/https://api-docs.deepseek.com/quick_start/pricing |
| same | `deepseek-flash` off-peak | $0.003 | $0.15 | $0.60 | | same |
| 2026-10-17 to now | V4.1-Flash-1017 at unchanged prices; V4-Pro-0813 unchanged | as above | | | | https://api-docs.deepseek.com/quick_start/pricing (live, read 2026-11-10) ; https://api-docs.deepseek.com/updates |

Peak / off-peak rules (current): "Off-peak rates are half of the peak rates. Peak hours are 01:00 - 04:00 and 06:00 - 10:00 UTC, Monday through Friday, excluding Chinese public holidays." Concurrency limit: 2,500 (Flash), 500 (Pro). Source: https://api-docs.deepseek.com/quick_start/pricing

Search and Fetch APIs (from 2026-10-17): $0.75 per 1,000 successful requests each; server-side tool calls in DeepSeek Harness, Claude Code and Codex free until billing starts 2026-10-31 16:00 UTC. Source: https://api-docs.deepseek.com/news/news261017

Earlier pricing (V2 launch, May 2024): UNVERIFIED in this pass. The Nov 2024 snapshot above is the earliest captured page. The V2 README only says "pay-as-you-go at an unbeatable price": https://huggingface.co/deepseek-ai/DeepSeek-V2

Gap: no Wayback capture parsed between 2026-05-01 and 2026-08-17, so I cannot say whether V4 prices changed between the end of the Pro discount (2026-05-31) and 2026-08-16. UNVERIFIED.

Odd finding: `https://api-docs.deepseek.com/news/news250120` (the R1 release page) now serves the "Your First API Call" page. The R1 release is still recorded in the change log: https://api-docs.deepseek.com/updates

---

## 5. DeepSeek research papers

Verified against the arXiv API (`https://export.arxiv.org/api/query?id_list=...`) on 2026-11-10. Dates are arXiv first-submission dates. Affiliation was checked on page 1 of the PDF for the less obvious ones. All PDFs below are cached at `~/.cache/papers/<id>.pdf`.

| arXiv id | Title | Date | One-line contribution |
|---|---|---|---|
| 2310.16818 https://arxiv.org/abs/2310.16818 | DreamCraft3D: Hierarchical 3D Generation with Bootstrapped Diffusion Prior | 2023-10-25 | Text-to-3D generation (ICLR 2024); code in deepseek-ai org (https://github.com/deepseek-ai/DreamCraft3D), joint with Tsinghua |
| 2401.02954 https://arxiv.org/abs/2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | 2024-01-05 | Scaling-law study (hyperparameters, data quality) and 7B/67B dense models on 2T tokens |
| 2401.06066 https://arxiv.org/abs/2401.06066 | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | 2024-01-11 | Fine-grained experts plus shared experts |
| 2401.14196 https://arxiv.org/abs/2401.14196 | DeepSeek-Coder: When the Large Language Model Meets Programming, The Rise of Code Intelligence | 2024-01-25 | Repo-level code pretraining, FIM, 16K context |
| 2402.03300 https://arxiv.org/abs/2402.03300 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models | 2024-02-05 | Math corpus mined from Common Crawl, and GRPO (critic-free RL) |
| 2403.05525 https://arxiv.org/abs/2403.05525 | DeepSeek-VL: Towards Real-World Vision-Language Understanding | 2024-03-08 | Hybrid vision encoder VLM |
| 2405.04434 https://arxiv.org/abs/2405.04434 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | 2024-05-07 | Multi-head Latent Attention (MLA): "reduces the KV cache by 93.3%" vs 67B, "saves 42.5% of training costs" |
| 2405.14333 https://arxiv.org/abs/2405.14333 | DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data | 2024-05-23 | Large synthetic Lean 4 proof data |
| 2406.11931 https://arxiv.org/abs/2406.11931 | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence | 2024-06-17 | Continued pretraining of V2 for code, 338 languages, 128K |
| 2407.01906 https://arxiv.org/abs/2407.01906 | Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural LLMs | 2024-07-02 | Fine-tune only task-relevant experts (ESFT) |
| 2408.08152 https://arxiv.org/abs/2408.08152 | DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for RL and Monte-Carlo Tree Search | 2024-08-15 | RLPAF and RMaxTS tree search |
| 2408.14158 https://arxiv.org/abs/2408.14158 | Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning | 2024-08-26 | The 10,000 A100 PCIe cluster, HFReduce, 3FS (SC24) |
| 2408.15664 https://arxiv.org/abs/2408.15664 | Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts | 2024-08-28 | Per-expert bias on routing scores instead of an aux loss (used in V3) |
| 2410.13848 https://arxiv.org/abs/2410.13848 | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation | 2024-10-17 | Separate encoders for understanding vs generation |
| 2411.07975 https://arxiv.org/abs/2411.07975 | JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation | 2024-11-12 | Rectified flow inside an AR LLM |
| 2412.10302 https://arxiv.org/abs/2412.10302 | DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding | 2024-12-13 | Dynamic tiling + MoE VLM |
| 2412.19437 https://arxiv.org/abs/2412.19437 | DeepSeek-V3 Technical Report | 2024-12-27 | 671B MoE, FP8 training, MTP, DualPipe, $5.576M final-run cost claim |
| 2501.12948 https://arxiv.org/abs/2501.12948 | DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning | 2025-01-22 | Pure RL (R1-Zero) yields reasoning; R1 adds cold start; distillation. Peer-reviewed in Nature, 17 Sep 2025: https://www.nature.com/articles/s41586-025-09422-z |
| 2501.17811 https://arxiv.org/abs/2501.17811 | Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling | 2025-01-29 | Scaled Janus to 7B |
| 2502.07316 https://arxiv.org/abs/2502.07316 | CodeI/O: Condensing Reasoning Patterns via Code Input-Output Prediction | 2025-02-11 | Train reasoning by predicting code inputs/outputs in natural language |
| 2502.11089 https://arxiv.org/abs/2502.11089 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | 2025-02-16 | NSA: compressed + selected + sliding branches, trainable end to end (with PKU; ACL 2025) |
| 2504.02495 https://arxiv.org/abs/2504.02495 | Inference-Time Scaling for Generalist Reward Modeling | 2025-04-03 | DeepSeek-GRM and Self-Principled Critique Tuning (SPCT) |
| 2504.21801 https://arxiv.org/abs/2504.21801 | DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via RL for Subgoal Decomposition | 2025-04-30 | V3 decomposes problems into subgoals, 7B prover solves them, then RL |
| 2505.09343 https://arxiv.org/abs/2505.09343 | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures | 2025-05-14 | ISCA 2025 industry paper: hardware co-design lessons, multi-plane network |
| (no arXiv) https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf | DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention | 2025-09-29 | DSA: lightning indexer + top-k token selection on top of MLA |
| 2510.18234 https://arxiv.org/abs/2510.18234 | DeepSeek-OCR: Contexts Optical Compression | 2025-10-21 | Compress long text into few vision tokens (about 10x at ~97% OCR precision per DeepSeek) |
| 2511.22570 https://arxiv.org/abs/2511.22570 | DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning | 2025-11-27 | Proof generator trained with a learned verifier |
| 2512.02556 https://arxiv.org/abs/2512.02556 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | 2025-12-02 | DSA, scaled RL, agentic task synthesis (1,800+ environments) |
| 2512.24880 https://arxiv.org/abs/2512.24880 | mHC: Manifold-Constrained Hyper-Connections | 2025-12-31 | Constrain hyper-connection mixing matrices (Sinkhorn, doubly stochastic) to keep residual stream stable; used in V4 |
| 2601.07372 https://arxiv.org/abs/2601.07372 | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models | 2026-01-12 | Engram: hashed N-gram memory lookup as a second sparsity axis, U-shaped allocation law; Engram-27B; used in V4.1 |
| 2601.20552 https://arxiv.org/abs/2601.20552 | DeepSeek-OCR 2: Visual Causal Flow | 2026-01-28 | LLM-style encoder learns reading order |
| 2602.21548 https://arxiv.org/abs/2602.21548 | DualPath: Breaking the Storage Bandwidth Bottleneck in Agentic LLM Inference | 2026-02-25 | Load KV cache through decode-side storage NICs too (PKU, Tsinghua, DeepSeek-AI; affiliation verified on PDF p.1) |
| 2606.19348 https://arxiv.org/abs/2606.19348 | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | 2026-04-26 (per arXiv page) | CSA + HCA hybrid attention, mHC, Muon; 1M context at 27% FLOPs and 10% KV cache of V3.2 |
| 2607.05147 https://arxiv.org/abs/2607.05147 | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation | 2026-07-06 | Semi-autoregressive drafter, verify only high-confidence tokens |
| 2608.25512 https://arxiv.org/abs/2608.25512 | A Programming Paradigm for Spatiotemporal Composability | 2026-08-26 | Formal basis (revertible effects, coeffects) of Cordis, the plugin runtime under DeepSeek Harness (PKU + DeepSeek-AI, verified on PDF p.1) |
| 2609.19969 https://arxiv.org/abs/2609.19969 | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | 2026-09-17 | Causal encoder-decoder, CSA2, FP4 KV, 890 bytes/token global KV cache |
| 2609.22978 https://arxiv.org/abs/2609.22978 | DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale | 2026-09-19 | Production sandbox platform (FnCall, container, microVM, full VM) for agentic RL (DeepSeek-AI, verified on PDF p.1) |

Related, not arXiv:

- Nature version of R1: "DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning", Nature 645, 633-638 (2025), https://www.nature.com/articles/s41586-025-09422-z
- V4 tech report PDF was first linked from HF (`https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf`, per https://api-docs.deepseek.com/news/news260424); that file now returns 404, and the cards link to arXiv 2606.19348.
- "Exploring Super-agent Design: Towards a More Autonomous Agent System" is cited by https://github.com/deepseek-ai/dsh-superagent-prototype ; I did not find it on arXiv. UNVERIFIED as a paper.
- "Thinking with Visual Primitives" (around 2026-04-30): UNVERIFIED, see Section 2.

Third-party papers about DeepSeek that are easy to confuse with DeepSeek's own (do not cite as DeepSeek work): FlashMemory-DeepSeek-V4 (https://arxiv.org/abs/2606.09079), "Evaluating DeepSeek V4.1 Flash-1017 for loss of control and harmful manipulation risks" (https://arxiv.org/abs/2610.35671), "The Law of DeepSeek" (https://arxiv.org/abs/2610.05238).

Completeness caveat: alphaXiv's DeepSeek organization page reports 23 papers but shows 20 (https://www.alphaxiv.org/organizations/deepseek/papers); HF says 30 papers linked to the org. My list has 37 entries because I include pre-2025 papers with personal author lists. There may be DeepSeek-affiliated papers with only individual authors that I missed.

---

## 6. Items for writers to handle carefully

1. V4 arXiv date: the id is 2606.19348 but arXiv shows "Submitted on 26 Apr 2026". Quote the arXiv page, not the id prefix.
2. "V4.1-Flash total params": DeepSeek gives 552B backbone plus a 196B Engram table. Do not add them into one number without saying so. HF's safetensors total (763B stored elements) counts packed FP4 bytes, so it is not a parameter count either.
3. HF `createdAt` is not a release date (V2's repo was created 2024-04-22, two weeks before the paper).
4. License labels on HF are sometimes wrong or inconsistent (Janus, Prover-V2, original V3). Quote the LICENSE file and the card text.
5. The $5.576M (V3) and $294K (R1) numbers are DeepSeek's own and cover only final training runs. The $1.6B figure is SemiAnalysis's estimate of server capex, not a training cost.

## 7. PDF cache

Downloaded to `~/.cache/papers/`: 2310.16818, 2401.02954, 2401.06066, 2401.14196, 2402.03300, 2403.05525, 2405.04434, 2405.14333, 2406.11931, 2407.01906, 2408.08152, 2408.14158, 2408.15664, 2410.13848, 2411.07975, 2412.10302, 2412.19437, 2501.12948, 2501.17811, 2502.07316, 2502.11089, 2504.02495, 2504.21801, 2505.09343, 2510.18234, 2511.22570, 2512.02556, 2512.24880, 2601.07372, 2601.20552, 2602.21548, 2606.19348, 2607.05147, 2608.25512, 2609.19969, 2609.22978 (all `.pdf`).

## 8. URL check

On 2026-11-10 every URL in this file was requested with `curl -L`. All returned HTTP 200 except: `https://api.deepseek.com` and `https://api.deepseek.com/anthropic` (401, expected: they are API endpoints that need a key), and the two links recorded above as dead on purpose: `https://github.com/deepseek-ai/Thinking-with-Visual-Primitives` (404) and `https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/blob/main/DeepSeek_V4.pdf` (404). The `id_list=...` arXiv URL in the Section 5 header is a template, not a link.
