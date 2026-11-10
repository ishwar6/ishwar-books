# 05. DeepSeek's specialised model families and side research

Research dossier for "DeepSeek, From the Ground Up". Compiled 2026-11-10.

## How to read this file

- Scope: everything that is not the main V-series chat and reasoning line (V2, V2.5, V3, R1, V3.1, V3.2, V4, V4.1). Main-line facts appear only where a specialised family feeds into them.
- Every paper is cached at `~/.cache/papers/<arxiv id>.pdf`. Page numbers are 1-based PDF pages of the version named in the table below (the latest version on arXiv at download time). If a writer downloads a different version, page numbers can shift.
- Quotes were extracted with pymupdf. Whitespace is normalised. Line-break hyphens from the PDF are kept as they appear (for example "pre- training"), so a screenshot of the PDF matches the quote. Text in a quote is exact; `[...]` marks a cut.
- "HF createdAt" is the creation timestamp of the repo in the Hugging Face API (`https://huggingface.co/api/models?author=deepseek-ai`), fetched 2026-11-10. It is the best public proxy for the weight release date. It is not always the announcement date.
- License note: most 2023 to 2024 releases use two licenses: MIT for code, and the "DeepSeek Model License" (`LICENSE-MODEL`) for weights, which allows commercial use with use-based restrictions. From late 2025, several side releases are Apache 2.0 or MIT for weights too.

### Papers used (arXiv id, version cached, first-submission date from the arXiv API)

| arXiv id | Version cached | First submitted | Title |
|---|---|---|---|
| 2401.14196 | v2 | 2024-01-25 | DeepSeek-Coder: When the Large Language Model Meets Programming |
| 2406.11931 | v1 | 2024-06-17 | DeepSeek-Coder-V2: Breaking the Barrier of Closed-Source Models in Code Intelligence |
| 2402.03300 | v3 | 2024-02-05 | DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models |
| 2511.22570 | v1 | 2025-11-27 | DeepSeekMath-V2: Towards Self-Verifiable Mathematical Reasoning |
| 2405.14333 | v1 | 2024-05-23 | DeepSeek-Prover: Advancing Theorem Proving in LLMs through Large-Scale Synthetic Data |
| 2408.08152 | v1 | 2024-08-15 | DeepSeek-Prover-V1.5: Harnessing Proof Assistant Feedback for RL and MCTS |
| 2504.21801 | v2 | 2025-04-30 | DeepSeek-Prover-V2: Advancing Formal Mathematical Reasoning via RL for Subgoal Decomposition |
| 2403.05525 | v2 | 2024-03-08 | DeepSeek-VL: Towards Real-World Vision-Language Understanding |
| 2412.10302 | v1 | 2024-12-13 | DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding |
| 2410.13848 | v1 | 2024-10-17 | Janus: Decoupling Visual Encoding for Unified Multimodal Understanding and Generation |
| 2411.07975 | v2 | 2024-11-12 | JanusFlow: Harmonizing Autoregression and Rectified Flow for Unified Multimodal Understanding and Generation |
| 2501.17811 | v1 | 2025-01-29 | Janus-Pro: Unified Multimodal Understanding and Generation with Data and Model Scaling |
| 2510.18234 | v1 | 2025-10-21 | DeepSeek-OCR: Contexts Optical Compression |
| 2601.20552 | v1 | 2026-01-28 | DeepSeek-OCR 2: Visual Causal Flow |
| 2407.01906 | v2 | 2024-07-02 | Let the Expert Stick to His Last: Expert-Specialized Fine-Tuning for Sparse Architectural LLMs (ESFT) |
| 2504.02495 | v3 | 2025-04-03 | Inference-Time Scaling for Generalist Reward Modeling (DeepSeek-GRM, SPCT) |
| 2512.24880 | v2 | 2025-12-31 | mHC: Manifold-Constrained Hyper-Connections |
| 2601.07372 | v2 | 2026-01-12 | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs (Engram) |
| 2607.05147 | v1 | 2026-07-06 | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation |
| 2609.22978 | v1 | 2026-09-19 | DeepSeek Elastic Compute (DSec): A Sandbox Infrastructure for Effective Agentic Training at Scale |

Main-line papers used only for the "folded in" section: 2405.04434 (DeepSeek-V2), 2501.12948 (DeepSeek-R1, cached version is 86 pages, a 2026 revision), 2512.02556 (DeepSeek-V3.2), 2606.19348 (DeepSeek-V4).

Anomaly to flag: the arXiv abs page for 2606.19348 says "Submitted on 26 Apr 2026", but a "2606" id would normally mean June 2026. Source: https://arxiv.org/abs/2606.19348 (fetched 2026-11-10). Treat the exact submission date as UNVERIFIED. The HF repos for V4 were created 2026-04-22.

---

## Timeline at a glance

| Date (source) | Release | Type |
|---|---|---|
| 2023-10-28 to 2023-11-01 (HF createdAt) | DeepSeek-Coder 1.3B / 5.7B-MQA / 6.7B / 33B, base and instruct | Dense code LLM |
| 2024-01-25 (HF createdAt; arXiv same day) | DeepSeek-Coder-v1.5 7B; Coder paper | Dense code LLM |
| 2024-02-05 (HF createdAt, arXiv) | DeepSeekMath 7B base / instruct / RL; GRPO | Dense math LLM |
| 2024-03-07 (HF createdAt) | DeepSeek-VL 1.3B and 7B, base and chat | Dense VLM |
| 2024-05-23 (arXiv) / 2024-08-16 (HF createdAt) | DeepSeek-Prover-V1 7B | Lean 4 prover |
| 2024-06-14 (HF createdAt) | DeepSeek-Coder-V2 236B and Lite 16B | MoE code LLM |
| 2024-07-04 (HF createdAt) | ESFT checkpoints (13 "lite" repos) | PEFT method for MoE |
| 2024-08-15 (HF createdAt, arXiv) | DeepSeek-Prover-V1.5 Base / SFT / RL | Lean 4 prover |
| 2024-09-05 (HF createdAt) | DeepSeek-V2.5 (merges V2-Chat and Coder-V2-Instruct) | Main line, merge point |
| 2024-10-18 (HF createdAt) | Janus-1.3B | Unified multimodal |
| 2024-11-12 (HF createdAt) | JanusFlow-1.3B | Unified multimodal |
| 2024-12-13 (HF createdAt) | DeepSeek-VL2 Tiny / Small / VL2 | MoE VLM |
| 2025-01-26 (HF createdAt) | Janus-Pro 1B and 7B | Unified multimodal |
| 2025-04-03 (arXiv) | DeepSeek-GRM paper (SPCT) | Reward modelling |
| 2025-04-30 (HF createdAt) | DeepSeek-Prover-V2 7B and 671B | Lean 4 prover |
| 2025-10-17 (HF createdAt) | DeepSeek-OCR | MoE VLM, OCR |
| 2025-11-27 (HF createdAt) | DeepSeek-Math-V2 (685B) | Natural-language theorem proving |
| 2025-12-31 (arXiv) | mHC paper | Architecture research |
| 2026-01-12 (arXiv; GitHub repo created same day) | Engram paper and code | Architecture research |
| 2026-01-27 (HF createdAt) | DeepSeek-OCR-2 | MoE VLM, OCR |
| 2026-06-26 to 2026-06-28 (GitHub, HF createdAt) | DeepSpec codebase; DSpark / DFlash / Eagle3 draft checkpoints; V4-Pro-DSpark, V4-Flash-DSpark | Speculative decoding |
| 2026-07-06 (arXiv) | DSpark paper | Speculative decoding |
| 2026-08-31 (HF createdAt) | DeepSeek-V4-Flash-Vision-Exp | Experimental V4 multimodal |
| 2026-09-19 (arXiv) | DSec paper | RL sandbox infrastructure |

Sources: HF API listing `https://huggingface.co/api/models?author=deepseek-ai&sort=createdAt`; GitHub API listing `https://api.github.com/orgs/deepseek-ai/repos`; arXiv API `https://export.arxiv.org/api/query`. All fetched 2026-11-10.

---

## 1. DeepSeek-Coder (1.3B to 33B), and Coder-v1.5 7B

**Paper:** arXiv 2401.14196 (v2). https://arxiv.org/abs/2401.14196
**Code:** https://github.com/deepseek-ai/DeepSeek-Coder
**Weights:** https://huggingface.co/deepseek-ai/deepseek-coder-33b-instruct (and siblings)

**Release.** Weights first appear on HF 2023-10-28 (33B base, 1.3B base), 2023-10-23 (6.7B base), up to 2023-11-01 (33B instruct). A 5.7B MQA base also exists (2023-10-31). The paper came later, 2024-01-25. The v1.5 7B models appear 2024-01-25. Source: HF API listing.

**Sizes and architecture.** Dense decoder-only Transformers, same framework as DeepSeek LLM, RoPE, SwiGLU. 1.3B (24 layers, hidden 2048), 6.7B (32 layers, hidden 4096), 33B (62 layers, hidden 7168, GQA with group size 8). Trained from scratch. Source: 2401.14196 p8 (Section 3.3) and p9 (Table 2).

**Data and method.** 2T tokens; 87% source code, 10% English code-related natural language, 3% Chinese natural language (p1, p3). Repository-level (project-level) data with dependency ordering, Fill-in-the-Middle (FIM) objective, 16K context window (p1). Contamination filtering removes files that contain HumanEval, MBPP, GSM8K, MATH content (p5). The FIM ablation shows a trade-off: 100% FIM rate is best for infilling but worst for plain completion (p7).

**Coder-v1.5 7B.** Continued pre-training of DeepSeek-LLM-7B Base on 2T tokens, next-token prediction only, 4K context. Data: 70% source code, 10% Markdown and StackExchange, 7% code-related NL, 7% math-related NL, 6% bilingual NL. Source: 2401.14196 p15 (Section 5, Table 9). This 7B v1.5 base is the starting point of DeepSeekMath.

**Key results** (2401.14196 p11, Table 3; greedy decoding, HumanEval Python and MBPP):

| Model | HumanEval Python | Multilingual HumanEval avg | MBPP |
|---|---|---|---|
| DeepSeek-Coder-Base 1.3B | 34.8% | 28.3% | 46.2% |
| DeepSeek-Coder-Base 6.7B | 49.4% | 44.7% | 60.6% |
| DeepSeek-Coder-Base 33B | 56.1% | 50.3% | 66.0% |
| CodeLlama-Base 34B | 48.2% | 41.0% | 55.2% |
| DeepSeek-Coder-Instruct 6.7B | 78.6% | 66.1% | 65.4% |
| DeepSeek-Coder-Instruct 33B | 79.3% | 69.2% | 70.0% |
| GPT-3.5-Turbo | 76.2% | 64.9% | 70.8% |
| GPT-4 | 84.1% | 76.5% | 80.0% |

DS-1000 (33B base) average 40.2% (p12, Table 4). LeetCode Contest benchmark: 180 problems, July 2023 to January 2024 (p12).

**License and openness.** Code MIT; weights under the DeepSeek Model License; "DeepSeek Coder supports commercial use." Source: https://github.com/deepseek-ai/DeepSeek-Coder README, Section 9. Open: base and instruct weights for all sizes. Not open: the training corpus.

**Quotes for screenshots**
- 2401.14196, p1, Abstract: "we introduce the DeepSeek-Coder series, a range of open-source code models with sizes from 1.3B to 33B, trained from scratch on 2 trillion tokens. These models are pre-trained on a high-quality project-level code corpus and employ a fill-in-the-blank task with a 16K window to enhance code generation and infilling."
- 2401.14196, p3, Section 2 (Data Collection): "The training dataset of DeepSeek-Coder is composed of 87% source code, 10% English code- related natural language corpus, and 3% code-unrelated Chinese natural language corpus."
- 2401.14196, p11, Section 4.1: "It's worth noting that even our smaller model, DeepSeek-Coder-Base 6.7B, surpasses the performance of CodeLlama-Base 34B." (PDF uses a curly apostrophe in "It's".)
- 2401.14196, p15, Section 5: "we perform additional pre-training from the general language model DeepSeek-LLM-7B Base (DeepSeek-AI, 2024) on 2 trillion tokens, resulting in DeepSeek- Coder-v1.5 7B."

---

## 2. DeepSeek-Coder-V2 (16B Lite and 236B, MoE)

**Paper:** arXiv 2406.11931 (v1). https://arxiv.org/abs/2406.11931
**Code:** https://github.com/deepseek-ai/DeepSeek-Coder-V2
**Weights:** https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Instruct, -Base, -Lite-Base, -Lite-Instruct; later -Instruct-0724.

**Release.** HF createdAt 2024-06-14 (four repos); arXiv 2024-06-17. DeepSeek-Coder-V2-Instruct-0724 repo created 2024-09-05.

**Architecture.** MoE, the DeepSeek-V2 architecture (MLA plus DeepSeekMoE). 236B total, 21B active; Lite 16B total, 2.4B active. Source: 2406.11931 p3 and p5 (Table 2).

**Base model and data.** Continued pre-training from an intermediate DeepSeek-V2 checkpoint that had seen 4.2T tokens, plus 6T more, so 10.2T total (p5). New corpus mix: 60% source code, 10% math, 30% natural language (p2, p3). Code part is 1,170B code-related tokens from GitHub and CommonCrawl, built with the DeepSeekMath pipeline (p2, p4). Languages 86 to 338; context 16K to 128K via YaRN (p1, p5). FIM enabled for Lite, disabled for 236B (p5, Table 2).

**Post-training.** SFT on code, math and general data, then RL with GRPO. Reward: a reward model trained on compiler and test-case feedback; the paper reports the reward-model signal beats raw compiler signal (p2, p7).

**Key results** (2406.11931 p1 Figure 1 and p3):
- HumanEval 90.2%, MBPP+ 76.2%, LiveCodeBench 43.4% (Dec 2023 to Jun 2024), first open model above 10% on SWE-Bench (12.7% in Figure 1). Source: p3, p1.
- MATH 75.7%, GSM8K 94.9% (p1, Figure 1 bars for DeepSeek-Coder-V2).
- Aider 73.7 (p1, Figure 1).
- General: 8.77 MT-bench, 7.84 AlignBench (p3).

**License and openness.** Code MIT; weights under the Model License; "DeepSeek-Coder-V2 series (including Base and Instruct) supports commercial use." Source: https://github.com/deepseek-ai/DeepSeek-Coder-V2 README, Section 7. Open: base and instruct weights at both sizes.

**Folded in later.** Merged into DeepSeek-V2.5 (see Section 14).

**Quotes for screenshots**
- 2406.11931, p1, Abstract: "DeepSeek-Coder-V2 is further pre-trained from an intermediate checkpoint of DeepSeek-V2 with additional 6 trillion tokens."
- 2406.11931, p1, Abstract: "DeepSeek-Coder- V2 expands its support for programming languages from 86 to 338, while extending the context length from 16K to 128K."
- 2406.11931, p5, Section 3.3: "The intermediate checkpoint was initially trained on 4.2T tokens. Consequently, DeepSeek-Coder-V2 has been exposed to a total of 10.2T high-quality tokens during the pre-training phase."
- 2406.11931, p3, Section 1.2: "Additionally, DeepSeek-Coder-V2 is the first open-source model that surpasses a score of 10% on SWEBench (Jimenez et al., 2023)."

---

## 3. DeepSeekMath 7B (Feb 2024): the corpus and GRPO

**Paper:** arXiv 2402.03300 (v3). https://arxiv.org/abs/2402.03300
**Code:** https://github.com/deepseek-ai/DeepSeek-Math
**Weights:** https://huggingface.co/deepseek-ai/deepseek-math-7b-base, -instruct, -rl (HF createdAt 2024-02-05)

**Architecture.** Dense 7B. Initialised from DeepSeek-Coder-Base-v1.5 7B, because "starting from a code training model is a better choice compared to a general LLM" (p2).

**The DeepSeekMath Corpus (why it matters).**
- 120B math tokens from 35.5M Common Crawl web pages (p5, Section 2.1).
- Built with an iterative fastText classifier. Seed: OpenWebMath, 500,000 positive examples (p4). Each round finds math-heavy domains, humans annotate URL paths, new pages join the seed, the classifier retrains. Four iterations; by the fourth, nearly 98% of the data was already collected in the third, so they stopped (p5).
- Pre-training mix for DeepSeekMath-Base: 500B tokens; 56% DeepSeekMath Corpus, 4% AlgebraicStack, 10% arXiv, 20% GitHub code, 10% natural language (p8).
- Finding: arXiv papers gave "no notable improvements" on their math benchmarks (p3). Code training helps math reasoning (p3).
- The same pipeline later built the 1,170B code corpus and math data for DeepSeek-Coder-V2 (2406.11931 p2).

**GRPO (why it matters).** Group Relative Policy Optimization drops PPO's value network. For each question it samples a group of outputs and uses the group's average reward as the baseline (p13, Section 4.1). This cuts memory and compute. GRPO later became the RL algorithm of DeepSeek-V2 (2405.04434 p4), Coder-V2 (2406.11931 p7), R1 (2501.12948 p2) and V4 (2606.19348 p5). RL data for DeepSeekMath-RL: about 144K GSM8K and MATH style questions (p15).

**Key results.**
- Base 7B: GSM8K 64.2%, MATH 36.2%, beats Minerva 540B (p2).
- RL 7B: MATH 51.7% top-1 without tools or voting; 60.9% with self-consistency over 64 samples (p1).
- GRPO over Instruct: GSM8K 82.9% to 88.2%, MATH 46.8% to 51.7%, CMATH 84.6% to 88.8% (p2).
- Analysis: "RL enhances Maj@K but not Pass@K" (p21, Figure 7). RL sharpens the output distribution; it does not add new abilities. This is an often-quoted insight.

**License and openness.** Code MIT, weights Model License, commercial use allowed. Source: https://github.com/deepseek-ai/DeepSeek-Math README, Section 6. Open: base, instruct, RL weights. Not open: the 120B corpus itself.

**Quotes for screenshots**
- 2402.03300, p1, Abstract: "we introduce DeepSeekMath 7B, which continues pre- training DeepSeek-Coder-Base-v1.5 7B with 120B math-related tokens sourced from Common Crawl, together with natural language and code data."
- 2402.03300, p1, Abstract: "Second, we introduce Group Relative Policy Optimization (GRPO), a variant of Proximal Policy Optimization (PPO), that enhances mathematical reasoning abilities while concurrently optimizing the memory usage of PPO."
- 2402.03300, p5, Section 2.1: "After four iterations of data collection, we end up with 35.5M mathematical web pages, totaling 120B tokens."
- 2402.03300, p13, Section 4.1 (Group Relative Policy Optimization; Section 4.1 starts on p11): "we propose Group Relative Policy Optimization (GRPO), which obviates the need for additional value function approximation as in PPO, and instead uses the average reward of multiple sampled outputs, produced in response to the same question, as the baseline."
- 2402.03300, p21, Figure 7 caption and text: "It was noted that RL enhances Maj@K but not Pass@K."
- 2402.03300, p3: "This offers a partial answer to the long-standing question: does code training improve reasoning abilities? We believe it does, at least for mathematical reasoning."

---

## 4. DeepSeekMath-V2 (Nov 2025): self-verifiable proofs

**Paper:** arXiv 2511.22570 (v1). https://arxiv.org/abs/2511.22570
**Code:** https://github.com/deepseek-ai/DeepSeek-Math-V2
**Weights:** https://huggingface.co/deepseek-ai/DeepSeek-Math-V2 (HF createdAt 2025-11-27)

**Architecture and size.** Built on DeepSeek-V3.2-Exp-Base (p2), so it is a 671B-class MoE with DeepSeek Sparse Attention. HF safetensors total: 685,396,921,376 parameters (HF API `https://huggingface.co/api/models/deepseek-ai/DeepSeek-Math-V2`; this count includes the MTP module).

**Method.**
1. Train a verifier with RL to find issues in proofs and score them (0, 0.5, 1) against expert annotations (p3, Section 2.1.1). Problem set: 17,503 contest problems that require proofs (p3).
2. Meta-verification: a second check that the issues the verifier lists really exist and justify the score. This stops the verifier from earning reward while hallucinating issues (p3, Section 2.1.2).
3. Train the generator with the verifier as reward model. The generator writes a proof followed by a self-analysis in the verifier's format; the verifier scores both, so the generator is rewarded for faithful self-evaluation (p5).
4. Keep the "generation-verification gap": scale verification compute to auto-label new hard proofs, and use them to retrain the verifier (p1).
5. High-compute search: pool of 64 candidate proofs, 64 verifications each, iterative refinement (p8, Section 3.3.3).

**Key results.**
- IMO 2025: P1 to P5 solved, 83.3% of points; CMO 2024: 73.8%; Putnam 2024: 118/120 (98.3%), above the top human score of 90 (p8, Table 1; p2).
- IMO-ProofBench (expert graded): Basic 99.0, Advanced 61.9 for DeepSeekMath-V2 (Heavy). Gemini Deep Think (IMO Gold): 89.0 and 65.7. Source: p8, Figure 3 (other numbers in that figure come from Luong et al. 2025, as the caption says).

**License and openness.** Weights Apache 2.0 (HF card `license: apache-2.0`; GitHub LICENSE file is Apache License 2.0, https://github.com/deepseek-ai/DeepSeek-Math-V2/blob/main/LICENSE). Open: weights, inference code, and the model's IMO 2025 and CMO problem outputs (2512.02556 p14 footnote points to this repo for the problems and inference code).

**Folded in later.** DeepSeek-V3.2-Speciale used the DeepSeekMath-V2 dataset and reward method (2512.02556 p6). DeepSeek-V4's formal-math evaluation uses self-verification from DeepSeekMath-V2 to filter informal solutions before a Lean agent proves them (2606.19348 p37).

**Quotes for screenshots**
- 2511.22570, p1, Abstract: "Pursuing higher final answer accuracy doesn't address a key issue: correct answers don't guarantee correct reasoning." (curly apostrophes in PDF)
- 2511.22570, p1, Abstract: "We then train a proof generator using the verifier as the reward model, and incentivize the generator to identify and resolve as many issues as possible in their own proofs before finalizing them."
- 2511.22570, p2, Section 1: "On the Putnam 2024 undergraduate competition, it scored 118/120, exceeding the highest score of 90 1 obtained by human participants." (the "1" is a footnote marker)
- 2511.22570, p3, Section 2.1.2: "To address this problem, we introduce meta-verification: a secondary evaluation process that assesses whether issues identified by the verifier indeed exist and whether these issues logically justify the predicted proof score"
- 2511.22570, p8, Section 3.3.3: "our model solved 11 of 12 problems completely and the remaining problem with minor errors, scoring 118/120 and surpassing the highest human score of 90."

---

## 5. DeepSeek-Prover V1, V1.5, V2 (Lean 4)

### 5.1 DeepSeek-Prover-V1 (May 2024)

**Paper:** arXiv 2405.14333 (v1). **Weights:** https://huggingface.co/deepseek-ai/DeepSeek-Prover-V1 (HF createdAt 2024-08-16, three months after the paper).
- Base: DeepSeekMath 7B, fine-tuned on synthetic Lean 4 data (p1).
- Data: translate natural-language competition problems into Lean 4 statements, filter bad statements, generate proofs, verify with Lean, iterate. 8 million formal statements with proofs (p1).
- Results: miniF2F-test whole-proof 46.3% at 64 samples, 52% cumulative; GPT-4 23.0%; FIMO 5 of 148 solved, GPT-4 0 (p1).
- Quote, 2405.14333 p1, Abstract: "After fine- tuning the DeepSeekMath 7B model on this synthetic dataset, which comprises 8 million formal statements with proofs, our model achieved whole-proof gener- ation accuracies of 46.3% with 64 samples and 52% cumulatively on the Lean 4 miniF2F test, surpassing the baseline GPT-4 at 23.0% with 64 samples and a tree search reinforcement learning method at 41.0%."

### 5.2 DeepSeek-Prover-V1.5 (Aug 2024)

**Paper:** arXiv 2408.08152 (v1). **Code:** https://github.com/deepseek-ai/DeepSeek-Prover-V1.5. **Weights:** DeepSeek-Prover-V1.5-Base, -SFT, -RL (HF createdAt 2024-08-15).
- 7B dense. Further pre-trained from DeepSeekMath-Base on math and code with a focus on Lean, Isabelle, Metamath (p3).
- SFT on an improved Lean 4 dataset, then RLPAF (RL from proof assistant feedback, using GRPO, p4), then RMaxTS: Monte-Carlo tree search with an intrinsic, curiosity-style reward for diverse proof paths (p1).
- Results: miniF2F-test 60.2% single-pass whole-proof (V1 was 50.0%), 63.5% with tree search; ProofNet 25.3% (p1, p4).
- License: code MIT, weights Model License, commercial use allowed (GitHub README Section 7).
- Quote, 2408.08152 p4, Section 1.2: "In the single-pass whole-proof generation setting, DeepSeek-Prover-V1.5 achieved a pass rate of 60.2% on the test set of miniF2F, marking a significant improvement of abso- lute 10.2 percentage points over DeepSeek-Prover-V1's 50.0%. Incorporating tree search techniques further elevated the pass rate to a new state-of-the-art 63.5%."

### 5.3 DeepSeek-Prover-V2 (Apr 2025)

**Paper:** arXiv 2504.21801 (v2). **Code:** https://github.com/deepseek-ai/DeepSeek-Prover-V2. **Weights:** DeepSeek-Prover-V2-7B and -671B (HF createdAt 2025-04-30).
- 671B: MoE built on DeepSeek-V3. 7B: DeepSeek-Prover-V1.5-Base with context extended from 4,096 to 32,768 tokens, fine-tuned on rollout data from the RL phase of Prover-V2-671B (p7).
- Cold start: DeepSeek-V3 decomposes a theorem into subgoals (Lean `have ... := by sorry`), the 7B prover solves the subgoals recursively, solved pieces are stitched with V3's chain of thought into training data for RL (p1, p3 Figure 2).
- Results: miniF2F-test 88.9% (Pass@8192); ProofNet-test 37.1% (Pass@1024); PutnamBench 47 of 658 (p1, p3). ProverBench: new 325-problem set, including 15 AIME 2024 and 2025 problems; Prover-V2 solves 6, DeepSeek-V3 informally solves 8 with majority voting (p1).
- Discrepancy: the GitHub README says "solving 49 out of 658 problems from PutnamBench" (https://github.com/deepseek-ai/DeepSeek-Prover-V2 README, Section 1). The paper (v2, p1 and p3) says 47. Use 47 when citing the paper; note the README figure.
- License: weights under the Model License (README Section 6).
- Quote, 2504.21801 p1, Abstract: "The cold-start training procedure begins by prompting DeepSeek-V3 to decompose complex problems into a series of subgoals."
- Quote, 2504.21801 p1, Abstract: "In comparison, DeepSeek-V3 solves 8 of these problems using majority voting, highlighting that the gap between formal and informal mathematical reasoning in large language models is substantially narrowing."

---

## 6. DeepSeek-VL (Mar 2024)

**Paper:** arXiv 2403.05525 (v2). **Code:** https://github.com/deepseek-ai/DeepSeek-VL. **Weights:** deepseek-vl-1.3b-base/-chat, deepseek-vl-7b-base/-chat (HF createdAt 2024-03-07).
- Dense. LLM: DeepSeek LLM (LLaMA-like, RMSNorm, SwiGLU) (p10). 1.3B built on DeepSeek-LLM-1B trained on about 500B text tokens; 7B on DeepSeek-LLM-7B trained on about 2T tokens (p11).
- Hybrid vision encoder: SigLIP at 384 x 384 for semantics plus SAM-B at 1024 x 1024 for detail, joined by an MLP adaptor (p10; also described in 2412.10302 p4).
- Training idea: keep language ability by mixing LLM text data from the start and slowly shifting the vision:language ratio (p1).
- License: code MIT, weights DeepSeek Model License, commercial use allowed (GitHub README Section 5).
- Quote, 2403.05525 p1, Abstract: "DeepSeek-VL incorporates a hybrid vision encoder that efficiently processes high-resolution images (1024 x 1024) within a fixed token budget, while maintaining a relatively low computa- tional overhead."
- Quote, 2403.05525 p1, Abstract: "We posit that a proficient Vision-Language Model should, foremost, possess strong language abilities."

---

## 7. DeepSeek-VL2 (Dec 2024): MoE vision-language with dynamic tiling

**Paper:** arXiv 2412.10302 (v1). **Code:** https://github.com/deepseek-ai/DeepSeek-VL2. **Weights:** deepseek-vl2-tiny, -small, deepseek-vl2 (HF createdAt 2024-12-13).

**Architecture.** LLaVA-style: SigLIP-SO400M-384 encoder, VL adaptor, DeepSeekMoE LLM with MLA (p4).

| Variant | LLM total | LLM active | Total active (paper) |
|---|---|---|---|
| VL2-Tiny | 3B | 0.57B | 1.0B |
| VL2-Small | 16B | 2.4B | 2.8B |
| VL2 | 27B | 4.1B | 4.5B |

Source: 2412.10302 p3, p10 (Table 2), p1 (abstract).

**Dynamic tiling.** Candidate resolutions are (m x 384, n x 384) with 1 <= m, n and mn <= 9. The image is resized to the candidate needing least padding, cut into 384 x 384 local tiles plus one global thumbnail tile; SigLIP yields 27 x 27 = 729 embeddings per tile (p4). This replaces VL's two fixed resolutions (p3).

**Training.** Three stages: adaptor alignment (about 2B tokens), VL pre-training (about 800B image-text tokens), SFT (about 20B tokens) (p10, Table 2).

**Key results** (p11, OCR table): DeepSeek-VL2: DocVQA 93.3, ChartQA 86.0, InfoVQA 78.1, TextVQA 84.2, OCRBench 811. VL2-Small: 92.3 / 84.5 / 75.8 / 83.4 / 834. VL2-Tiny: 88.9 / 81.0 / 66.1 / 80.7 / 809.

**License:** code MIT, weights DeepSeek Model License, commercial use allowed (GitHub README Section 5).

**Quotes for screenshots**
- 2412.10302, p1, Abstract: "For the vision component, we incorporate a dynamic tiling vision encoding strategy designed for processing high-resolution images with different aspect ratios. For the language component, we leverage DeepSeekMoE models with the Multi-head Latent Attention mechanism, which compresses Key-Value cache into latent vectors, to enable efficient inference and high throughput."
- 2412.10302, p3: "Drawing inspiration from established slicing-tile methods, our system dynamically segments high-resolution inputs into local tiles, processes each tile through a shared vision transformer, and seamlessly integrates the extracted features within the language model."

---

## 8. Janus, JanusFlow, Janus-Pro: unified understanding and generation

**Code (all three):** https://github.com/deepseek-ai/Janus. License: code MIT, weights DeepSeek Model License (GitHub README Section 4). HF cards for Janus-Pro and JanusFlow show `license: mit` with `license_name: deepseek` (https://huggingface.co/deepseek-ai/Janus-Pro-7B). Treat the model license as the DeepSeek Model License.

**The core idea (decoupled visual encoding).** One autoregressive Transformer handles both tasks, but images enter through two different encoders: SigLIP for understanding (semantic features) and a VQ tokenizer for generation (discrete IDs). Earlier unified models (Chameleon) used one encoder for both, which hurt understanding.

### 8.1 Janus (Oct 2024)
- Paper: arXiv 2410.13848 (v1). Weights: Janus-1.3B (HF createdAt 2024-10-18).
- Base LLM: DeepSeek-LLM 1.3B, 4096 context; understanding encoder SigLIP-Large-Patch16-384; generation via a VQ tokenizer (p4, p7).
- Results: MMBench 69.4, SEED-Bench 63.7, POPE 87.0; MSCOCO-30K FID 8.53; GenEval 61% (p3).
- Quote, 2410.13848 p1, Abstract: "To address this issue, we decouple visual encoding into separate pathways, while still leveraging a single, unified transformer architecture for processing."

### 8.2 JanusFlow (Nov 2024)
- Paper: arXiv 2411.07975 (v2). Weights: JanusFlow-1.3B (HF createdAt 2024-11-12).
- Replaces the VQ autoregressive image head with rectified flow inside the LLM, in SDXL-VAE latent space (p5). Keeps decoupled encoders and adds representation alignment between them (p1).
- Results: MJHQ FID-30k 9.51, GenEval 0.63, DPG-Bench 80.09%; MMBench 74.9, SEED 70.5, GQA 60.3; 1.3B LLM (p2).
- Quote, 2411.07975 p1, Abstract: "Our key finding demonstrates that rectified flow can be straightforwardly trained within the large language model framework, eliminating the need for complex architectural modifications."

### 8.3 Janus-Pro (Jan 2025)
- Paper: arXiv 2501.17811 (v1). Weights: Janus-Pro-1B and -7B (HF createdAt 2025-01-26).
- Base LLM: DeepSeek-LLM 1.5B and 7B (p5). Changes: optimised training schedule, more data (about 72M synthetic aesthetic samples, real:synthetic 1:1 in pre-training, p4), bigger model.
- Results (7B): MMBench 79.2 (Janus 69.4); GenEval 0.80 (Janus 0.61, DALL-E 3 0.67, SD3-Medium 0.74); DPG-Bench 84.19 (p2, p6, p7).
- Limit: image resolution 384 x 384 (p2, Figure 2 caption).
- Quote, 2501.17811 p2: "Additionally, in the text-to- image instruction-following leaderboard GenEval [14], Janus-Pro-7B scores 0.80, outperforming Janus [46] (0.61), DALL-E 3 (0.67), and Stable Diffusion 3 Medium [11] (0.74)."
- Quote, 2501.17811 p4: "In Janus-Pro, we incorporate approximately 72 million samples of synthetic aesthetic data, bringing the ratio of real to synthetic data to 1:1 during the unified pretraining stage."

---

## 9. DeepSeek-OCR (Oct 2025): contexts optical compression

**Paper:** arXiv 2510.18234 (v1). **Code:** https://github.com/deepseek-ai/DeepSeek-OCR. **Weights:** https://huggingface.co/deepseek-ai/DeepSeek-OCR (HF createdAt 2025-10-17). About 3.34B parameters (HF API safetensors total 3,336,106,240).

**Idea.** Render long text as an image and let a small number of vision tokens stand for many text tokens. Test how far you can compress and still decode.

**Architecture.** DeepEncoder plus a DeepSeek3B-MoE-A570M decoder (p1). DeepEncoder = SAM-base (80M, window attention) then a 16x convolutional token compressor then CLIP-large (300M, global attention) (p5, Figure 3). The decoder has 12 layers (p10).

**Resolution modes** (p6, Table 1): Tiny 512 (64 tokens), Small 640 (100), Base 1024 (256), Large 1280 (400), Gundam n x 100 + 256, Gundam-M n x 256 + 400.

**Compression vs accuracy** (Fox benchmark, English pages with 600 to 1300 text tokens; 2510.18234 p10, Table 2):

| Text tokens per page | 64 vision tokens: precision | ratio | 100 vision tokens: precision | ratio | Pages |
|---|---|---|---|---|---|
| 600-700 | 96.5% | 10.5x | 98.5% | 6.7x | 7 |
| 700-800 | 93.8% | 11.8x | 97.3% | 7.5x | 28 |
| 800-900 | 83.8% | 13.2x | 96.8% | 8.5x | 28 |
| 900-1000 | 85.9% | 15.1x | 96.8% | 9.7x | 14 |
| 1000-1100 | 79.3% | 16.5x | 91.5% | 10.6x | 11 |
| 1100-1200 | 76.4% | 17.7x | 89.8% | 11.3x | 8 |
| 1200-1300 | 59.1% | 19.7x | 87.1% | 12.6x | 4 |

Note: Figure 1(a) on p1 prints 85.8% and 76.3% for two cells where Table 2 prints 85.9% and 76.4%. Use Table 2.

Headline: under 10x compression about 97% precision; at 20x about 60% (p1). OmniDocBench: beats GOT-OCR2.0 (256 tokens per page) with 100 vision tokens, beats MinerU2.0 (6000+ tokens per page) with under 800 (p1). Throughput: 200k+ pages per day on one A100-40G (p1). Training: 20 nodes of 8 A100-40G (p10).

**Why writers care.** The paper frames this as a possible memory mechanism: old context could be stored as progressively blurrier images, like forgetting (p12).

**License:** MIT (HF card; GitHub LICENSE "MIT License, Copyright (c) 2025 DeepSeek").

**Quotes for screenshots**
- 2510.18234, p1, Abstract: "Experiments show that when the number of text tokens is within 10 times that of vision tokens (i.e., a compression ratio < 10×), the model can achieve decoding (OCR) precision of 97%. Even at a compression ratio of 20×, the OCR accuracy still remains at about 60%."
- 2510.18234, p5, Section 3: "DeepEncoder is the core of DeepSeek-OCR, comprising three components: a SAM [17] for perception dominated by window attention, a CLIP [29] for knowledge with dense global attention, and a 16× token compressor that bridges between them."
- 2510.18234, p12: "When compressing tokens by nearly 20×, we find that precision can still approach 60%. These results indicate that optical contexts compression is a very promising and worthwhile research direction"

---

## 10. DeepSeek-OCR 2 (Jan 2026): visual causal flow

**Paper:** arXiv 2601.20552 (v1), submitted 2026-01-28. **Code:** https://github.com/deepseek-ai/DeepSeek-OCR-2. **Weights:** https://huggingface.co/deepseek-ai/DeepSeek-OCR-2 (HF createdAt 2026-01-27). License: Apache 2.0 per HF card (`license: apache-2.0`); no LICENSE file was found at the GitHub repo root on 2026-11-10, so the code license is UNVERIFIED.
- DeepEncoder V2 replaces the CLIP part with an LLM-style encoder (Qwen2 500M). Visual tokens attend bidirectionally; a set of learnable "causal flow" queries attends causally and learns a reading order; only these query outputs go to the decoder (p1 Figure 1, p3).
- Visual token budget 256 to 1120. 256 matches DeepSeek-OCR at 1024 x 1024; 1120 matches Gemini-3 Pro's maximum (p3).
- OmniDocBench v1.5 overall: 91.09 with 1120 tokens vs DeepSeek-OCR (9 crops) 87.36 with 1156 tokens, so +3.73; edit distance 0.073 to 0.048 (p9).
- Quote, 2601.20552 p3: "Com- pared to the DeepSeek-OCR baseline, DeepSeek-OCR 2 achieves 3.73% performance gains on OmniDocBench v1.5 [34] and yields considerable advances in visual reading logic."
- Quote, 2601.20552 p1, Abstract: "Conventional vision-language models (VLMs) invariably process visual tokens in a rigid raster-scan order (top-left to bottom-right) with fixed positional encoding when fed into LLMs."

---

## 11. ESFT: Expert-Specialized Fine-Tuning (Jul 2024)

**Paper:** arXiv 2407.01906 (v2). **Code:** https://github.com/deepseek-ai/ESFT. **Weights:** 13 repos on HF (ESFT-vanilla-lite plus token and gate variants for intent, law, math, code, summary, translation), HF createdAt 2024-07-04.
- Backbone: DeepSeek-V2-Lite architecture, 66 experts per layer, 8 of them active per token (p5, p2). (The HF ESFT-vanilla-lite card is very short, 368 bytes, and states no size; UNVERIFIED whether it is identical to the public V2-Lite weights.)
- Method: run a few task samples (32 concatenated samples of 4096 tokens), score each expert by average gate score (ESFT-Gate) or token selection ratio (ESFT-Token), fine-tune only the top experts; freeze all else including shared experts and attention (p4).
- Results (p6, Table 1, average over 8 tasks): vanilla 33.6, full fine-tune 51.0, LoRA 44.9, ESFT-Token 49.4, ESFT-Gate 50.2. General ability (Table 2 average): vanilla 62.4, FFT 58.8, LoRA 59.1, ESFT-Token 61.5, ESFT-Gate 60.6. ESFT forgets less.
- Efficiency: up to 90% less storage and up to 30% less training time vs full fine-tuning (p1).
- License: the ESFT GitHub README has no license section and no LICENSE file was found; UNVERIFIED.
- Quote, 2407.01906 p1, Abstract: "We find that MoE mod- els with finer-grained experts are more advan- tageous in selecting the combination of experts that are most relevant to downstream tasks, thereby enhancing both the training efficiency and effectiveness."
- Quote, 2407.01906 p1: "ESFT only trains the parameters of the selected experts, which effectively reduces the storage of up to 90% and training time up to 30% compared to full-parameter fine-tuning"

---

## 12. DeepSeek-GRM and SPCT (Apr 2025): generalist reward modelling

**Paper:** arXiv 2504.02495 (v3, 2025-09-25), first submitted 2025-04-03. https://arxiv.org/abs/2504.02495
- Pointwise generative reward model: the RM writes principles and a critique, then scores each response (p1).
- SPCT (Self-Principled Critique Tuning): two phases, rejective fine-tuning as cold start, then rule-based online RL (p5).
- Inference-time scaling: sample k reward generations in parallel and vote; a meta RM filters samples before voting (p1).
- Base models: DeepSeek-GRM-27B is post-trained from Gemma-2-27B (p2). They also train GRMs on DeepSeek-V2-Lite (16B MoE), DeepSeek-V2.5 (236B) and DeepSeek-V3 (671B) for comparison (p9).
- Results (p8, Table 2, overall over RewardBench, PPE Preference, PPE Correctness, RMB): DeepSeek-GRM-27B 69.9 greedy; 71.0 with Voting@32; 72.8 with MetaRM-guided voting@32. GPT-4o 71.3; Nemotron-4-340B-Reward 70.5.
- Claim: inference-time scaling of the 27B can beat training-time scaling to larger models (p3).
- Openness: the paper says "The models are released at Hugging Face and ModelScope" (p1). No DeepSeek-GRM repo exists in the deepseek-ai HF org (checked 2026-11-10). Repos BBQGOD/DeepSeek-GRM-27B, -16B, -27B-MetaRM exist (createdAt 2025-08-31, `license: gemma`, base google/gemma-2-27b), with the DeepSeek logo and a ModelScope link. That they are the official release by the first author is plausible but UNVERIFIED.
- Folded in later: DeepSeek-V4 uses a Generative Reward Model for hard-to-verify tasks, and the actor itself acts as the GRM (2606.19348 p29 to p30). The V4 report does not cite SPCT by name in the passage found; the lineage is an inference, mark as such.
- Quote, 2504.02495 p1, Abstract: "For the learning method, we propose Self-Principled Critique Tuning (SPCT) to foster scalable reward generation behaviors in GRMs through online RL, to generate principles adaptively and critiques accu- rately, resulting in DeepSeek-GRM models."
- Quote, 2504.02495 p5, Section 3: "SPCT consists of two phases: rejective fine-tuning, as the cold start, and rule-based online RL, reinforcing generalist reward generation by advancing the generated principles and critiques."
- Quote, 2606.19348 p30: "Instead, to address hard-to-verify tasks, we curate rubric-guided RL data and employ a Generative Reward Model (GRM) to evaluate policy trajectories. Crucially, we apply RL optimization directly to the GRM itself."

---

## 13. Side research and models, late 2025 to 2026 (verified)

### 13.1 mHC: Manifold-Constrained Hyper-Connections
- Paper: arXiv 2512.24880 (v2), first submitted 2025-12-31. Last author Wenfeng Liang.
- Hyper-Connections widen the residual stream into n streams with learned mixing matrices; unconstrained, the product of mixing matrices can amplify or shrink signals and break training (p4). mHC projects the residual mixing matrix onto the Birkhoff polytope (doubly stochastic matrices) with Sinkhorn-Knopp (p4). Overhead 6.7% at expansion rate n = 4 (p4).
- 27B results (p13, Table 4): BBH 43.8 baseline, 48.9 HC, 51.0 mHC; GSM8K 46.7 / 53.2 / 53.8; MMLU 59.0 / 63.0 / 63.4. Models tested at 3B, 9B, 27B (press: SCMP, https://www.scmp.com/tech/big-tech/article/3338427/deepseek-kicks-2026-paper-signalling-push-train-bigger-models-less).
- Folded in: DeepSeek-V4 uses mHC (2606.19348 abstract p1; V4 HF card).
- Quote, 2512.24880 p4: "Specifically, mHC utilizes the Sinkhorn-Knopp algorithm (Sinkhorn and Knopp, 1967) to entropically project Hres 𝑙 onto the Birkhoff polytope." (the symbol renders oddly in extraction; screenshot the PDF)
- Quote, 2512.24880 p4: "In- house large-scale training indicates that mHC supports training at scale and introduces only a 6.7% additional time overhead when expansion rate 𝑛= 4."

### 13.2 Engram: conditional memory
- Paper: arXiv 2601.07372 (v2), first submitted 2026-01-12. Code: https://github.com/deepseek-ai/Engram (Apache 2.0 LICENSE; demo code, no weights found on HF on 2026-11-10).
- A hashed N-gram embedding table with O(1) lookup, gated into the hidden state, as a second sparsity axis next to MoE (p1, p2). A U-shaped law splits a fixed sparse budget between experts and memory (p1).
- Engram-27B vs iso-parameter, iso-FLOPs MoE-27B: MMLU +3.4, CMMLU +4.0, BBH +5.0, ARC-Challenge +3.7, HumanEval +3.0, MATH +2.4; Multi-Query NIAH 84.2 to 97.0 (p1).
- Not in V4: the V4 technical report (2606.19348) contains no mention of "Engram" (full-text search, 2026-11-10). Blog claims that V4 uses Engram (for example https://kenhuangus.substack.com/p/deepseek-v4-the-next-frontier-of) are not supported by the report.
- Quote, 2601.07372 p1, Abstract: "we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic 𝑁-gram embedding for O(1) lookup."
- Quote, 2601.07372 p1, Abstract: "Mechanistic analyses reveal that Engram relieves the backbone's early layers from static reconstruction, effectively deepening the network for complex reasoning."

### 13.3 DSpark and DeepSpec: speculative decoding
- Paper: arXiv 2607.05147 (v1), 2026-07-06. Code: https://github.com/deepseek-ai/DeepSpec (MIT; repo created 2026-06-26).
- DSpark: semi-autoregressive drafter (parallel backbone plus a small sequential module) and confidence-scheduled verification that picks a per-request verify length from estimated prefix survival and engine throughput (p1).
- In the live DeepSeek-V4 serving system, vs the production MTP-1 baseline: 60% to 85% faster per-user generation at matched throughput (p1).
- Released: draft checkpoints for Eagle3, DFlash and DSpark against Qwen3-4B/8B/14B and Gemma-4-12B (12 repos, HF createdAt 2026-06-28), plus DeepSeek-V4-Pro-DSpark and DeepSeek-V4-Flash-DSpark (2026-06-27, MIT). DFlash (arXiv 2602.06036) and Eagle3 (arXiv 2503.01840) are other groups' methods that DeepSpec re-implements (DeepSpec README).
- Quote, 2607.05147 p1, Abstract: "Compared to the established production baseline (MTP-1), DSpark accelerates per-user generation speeds by 60%–85% at matched throughput levels." (contains an en dash, not an em dash)

### 13.4 DeepSeek-V4-Flash-Vision-Exp (Aug 2026)
- HF: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp (createdAt 2026-08-31, MIT, about 304.6B parameters by HF safetensors total). No paper found.
- Card: "our first experimental multimodal model in the DeepSeek-V4 family. It builds on the DeepSeek-V4-Flash architecture by incorporating visual modules and undergoing continued training to unlock visual understanding capabilities."
- Card numbers vs DeepSeek-V4-Flash-0731: ApexBench Pass@1 36.5 vs 26.2; Terminal Bench 2.1 83.9 vs 82.7. All vendor-reported, no paper.
- This is where the separate vision line (VL, VL2, Janus, OCR) starts to meet the main line. Whether DeepEncoder or Janus components are used inside is UNVERIFIED (the card mentions only "vision encoder and aligner").

### 13.5 DSec: DeepSeek Elastic Compute
- Paper: arXiv 2609.22978 (v1), 2026-09-19. Infrastructure, not a model.
- Sandbox platform for agentic RL: FnCall, container, microVM and full-VM backends behind one SDK; loads images from 3FS; one unit is about 160 nodes, about 3 million sandboxes per day, over 380,000 concurrent sandboxes, over 5,000 creations per second (p1).

### 13.6 Earlier side items, for completeness
- Native Sparse Attention, arXiv 2502.11089 (2025-02-16): research that preceded DeepSeek Sparse Attention in V3.2. Belongs to the main-line dossier; not analysed here.
- Fire-Flyer AI-HPC, arXiv 2408.14158; "Insights into DeepSeek-V3", arXiv 2505.09343: hardware and systems papers, not models.
- Open-infra releases (Feb 2025: FlashMLA, DeepEP, DeepGEMM, DualPipe, EPLB, 3FS, smallpond) and 2026 repos (TileKernels, DeepSelect, DeepGEMM-Ascend, DeepEP-Ascend, DeepInfer, deepseek-harness): code, not models. Source: GitHub org listing.
- DreamCraft3D (GitHub, 2023-10-23): ICLR 2024 3D generation work hosted by deepseek-ai. Not analysed.

### Searched for, not found
- No DeepSeek-Coder-V3, no DeepSeek-Prover-V3, no Janus-2, no DeepSeek-VL3 on HF, GitHub or arXiv as of 2026-11-10.
- No arXiv paper by DeepSeek-AI dated October or November 2026 surfaced in the arXiv API author query (au:Wenfeng_Liang) or web search. Papers without Liang as author (like DeepSeek-OCR) would not show in that query, so this is not a proof of absence.

---

## 14. What got folded into the main models

| Specialised work | Folded into | Evidence |
|---|---|---|
| DeepSeek-Coder-V2-Instruct (0724) | DeepSeek-V2.5 (Sept 2024), which "combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct" | https://huggingface.co/deepseek-ai/DeepSeek-V2.5 README, Introduction |
| DeepSeekMath data pipeline | Coder-V2 code and math corpus | 2406.11931 p2 |
| GRPO (from DeepSeekMath) | V2 Chat RL, Coder-V2, R1, V4 | 2405.04434 p4; 2406.11931 p7; 2501.12948 p2; 2606.19348 p5 |
| Coder-v1.5 7B | Initialisation of DeepSeekMath 7B | 2402.03300 p2 |
| DeepSeekMath 7B | Initialisation of Prover-V1 and Prover-V1.5 | 2405.14333 p1; 2408.08152 p1 |
| DeepSeek-V3 | Base of Prover-V2-671B and its cold-start data | 2504.21801 p1 |
| DeepSeek-V3.2-Exp-Base | Base of DeepSeekMath-V2 | 2511.22570 p2 |
| DeepSeekMath-V2 data and reward method | DeepSeek-V3.2-Speciale | 2512.02556 p6 |
| DeepSeekMath-V2 self-verification | V4 formal-math evaluation pipeline | 2606.19348 p37 |
| Generative reward modelling (GRM line) | V4 post-training uses a GRM, the actor is the GRM | 2606.19348 p29 to p30 (lineage to SPCT is inference) |
| mHC | DeepSeek-V4 architecture | 2606.19348 p1 |
| DSpark | DeepSeek-V4 serving; V4-Pro-DSpark and V4-Flash-DSpark checkpoints | 2607.05147 p1; HF listing |
| Vision line (VL, VL2, Janus, OCR) | First V4 multimodal model, DeepSeek-V4-Flash-Vision-Exp | HF card; direct reuse of earlier encoders UNVERIFIED |
| Engram | Not in V4 per the V4 report | full-text search of 2606.19348 |

Quote for the V2.5 merge (HF card, verbatim): "DeepSeek-V2.5 is an upgraded version that combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct. The new model integrates the general and coding abilities of the two previous versions."

Quote, 2512.02556 p6: "Additionally, we incorporated the dataset and reward method from DeepSeekMath-V2 (Shao et al., 2025) to enhance capabilities in mathematical proofs."

---

## 15. Open questions and UNVERIFIED items

1. arXiv 2606.19348 (V4) id vs "Submitted on 26 Apr 2026" mismatch.
2. Prover-V2 PutnamBench: paper 47/658, README 49/658.
3. DeepSeek-GRM weights: only under BBQGOD on HF; official status UNVERIFIED.
4. DeepSeek-OCR-2 code license and ESFT license: no LICENSE file found.
5. DeepSeek-V4-Flash-Vision-Exp: no paper; internal vision encoder origin UNVERIFIED.
6. Exact announcement dates (vs HF repo creation dates) for 2023 to 2024 releases were not cross-checked against DeepSeek's news page (https://api-docs.deepseek.com/news) except where noted.
