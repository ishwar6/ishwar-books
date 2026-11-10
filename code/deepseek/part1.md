---
title: "Every DeepSeek Model, and What Is Really Open"
description: "Every DeepSeek model from 2023 to 2026 with its size, context and license, and exactly what you can download, run or rebuild yourself."
date: 2026-11-10
tags: [deepseek, llm, open-weights, licenses]
series: "DeepSeek, From the Ground Up"
series_part: 1
motif: timeline
accent: "#5b8cff"
---

On 20 January 2025, a small lab in Hangzhou put a reasoning model with 671 billion numbers in it on a public website. Anyone could download it. The license was MIT, one of the shortest and most permissive licenses in common use. It allowed commercial use, changes, and even training other models on its answers.

That model was DeepSeek-R1. Its release started a long argument about what "open" means for an AI model. Since then, DeepSeek has put more than 100 model repositories on Hugging Face and 46 public code repositories on GitHub. Yet you cannot download the program that trained any of its main models, or the data it was trained on.

This part is a map. It lists every DeepSeek model from 2023 to November 2026, with its size, its context length and its license. Then it takes the word "open" apart, piece by piece: weights, code to run the model, code to train it, training data, papers and infrastructure. At the end you will know which model to look at for which job. You will also know exactly what you can download, what you can run, and what you can never rebuild from public material.

> [!TIP] The whole idea in one minute
> DeepSeek has one **main line** of general chat models. It runs from LLM 67B (2023) through V2, V2.5 and V3 (2024), R1, V3.1 and V3.2 (2025), to V4 and V4.1 (2026). Around it sit **specialised** models for code, maths proofs, vision and OCR.
>
> Since V2, the main models are **mixtures of experts**. They are huge in total, but each token uses only a small part: 37B of V3's 671B parameters. DeepSeek publishes the **weights** (under MIT since R1), the code to **run** them, the papers, and many fast **kernels and systems**. It does not publish the code that **trains** its main models or the **data**. So the models are "open weights": you can use, study and change them, but you cannot reproduce them.

## The lab behind the models

DeepSeek is a Chinese AI company. Its legal name, from its own license file, is Hangzhou DeepSeek Artificial Intelligence Fundamental Technology Research Co., Ltd. Its Hugging Face page says only that it was "founded in 2023" and is "dedicated to making AGI a reality". Its founder and CEO is Liang Wenfeng, who also appears as an author on its papers.

DeepSeek grew out of High-Flyer, a Chinese quantitative hedge fund that Liang co-founded. A quantitative fund trades with computer models, so it already owned many GPUs before large language models became popular. DeepSeek's own papers confirm the link: the Fire-Flyer paper thanks "our colleagues at DeepSeek-AI and High-Flyer Quant".

The clusters have names. Fire-Flyer 1 was built in 2019; DeepSeek's papers mention "our Fire-Flyer 1 experiments" but give no size for it. Fire-Flyer 2 is described in detail, in a paper presented at the SC24 supercomputing conference.

> [!PAPER] DeepSeek-AI (2024), Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning · Abstract · page 1
> [![The Fire-Flyer abstract, with deployed the Fire-Flyer 2 with 10,000 PCIe A100 GPUs and reducing costs by half highlighted](/img/deepseek/papers/p1-fireflyer.png)](/img/deepseek/papers/p1-fireflyer.png)
>
> **What it says:** "we deployed the Fire-Flyer 2 with 10,000 PCIe A100 GPUs, achieved performance approximating the DGX-A100 while reducing costs by half and energy consumption by 40%."
>
> **What it means:**
>
> NVIDIA sells a premium A100 server, the DGX-A100, with fast links between its GPUs. It also sells a cheaper A100 card that plugs into an ordinary server through a slot called PCIe. DeepSeek bought the cheaper cards, 10,000 of them, and wrote its own software to hide their slower links.
>
> This is a pattern you will see again and again in this series. DeepSeek often works under a hardware limit and wins with software that fits the hardware closely. The same paper describes HFReduce (a faster way for GPUs to add up their results), HaiScale (a training library) and 3FS (a file system). Of these, 3FS was later published as open source.

The V3 paper names a different cluster: 2,048 NVIDIA H800 GPUs. DeepSeek's later hardware paper notes that the H800's links between GPUs run at 400 GB/s, against 900 GB/s on the H100. [Part 3](deepseek-3-training-v3.md) covers that training run.

> [!WARNING] Third-party estimates are not DeepSeek's numbers
> Some numbers about DeepSeek that you will read online come from analysts, not from DeepSeek. In January 2025 the research firm SemiAnalysis estimated that DeepSeek and High-Flyer could use about 50,000 Hopper GPUs. It put their server spending at about US$1.6 billion. DeepSeek has not confirmed this. Press reports about funding rounds, headcount and new data centres are also unconfirmed. In this series, a number is DeepSeek's only when it comes from DeepSeek's own papers, model cards, repositories or API documents.

## Two ideas you need for the catalog

The catalog below rests on two ideas. Later parts go deep into each one.

The first is the most important for reading DeepSeek's numbers. An older model is **dense**: every parameter works on every token. A newer DeepSeek model is a **mixture of experts**.

> [!DEFINITION] Mixture of experts (MoE)
> A **mixture of experts** replaces one large block of the network with many small blocks, called **experts**. A small **router** picks a few experts for each token. Only those experts run. Some DeepSeek models also have **shared experts** that every token uses.

Think of a hospital. A dense model is one doctor who must know everything and sees every patient. An MoE model is a hospital with 256 specialists and a receptionist. Each patient sees the receptionist, a family doctor (the shared expert) and 8 specialists. The hospital holds far more knowledge than one doctor, but each patient costs only a few visits.

{{FIG:dense_vs_moe|A dense model sends each token through all of its weights. A DeepSeek-V3 mixture-of-experts layer sends each token to 1 shared expert and 8 of 256 routed experts, so most of the 671B weights are idle for any one token.}}

This gives the second idea: every MoE model has two sizes.

> [!DEFINITION] Total and active parameters
> **Total parameters** count every weight the model stores. **Active parameters** count only the weights that work on one token. Total size decides how much memory you need. Active size decides roughly how much computation each token costs.

### How 671B becomes 37B

The two sizes follow from a simple count. Split the model into the part that always runs and the expert part.

The always-on part holds attention, the word embeddings, the output layer, the first few dense layers and the routers. The expert part holds many experts in each MoE layer, and a token uses only a few of them.

$$N_{\text{total}} = N_{\text{always}} + (E + s)\, P\, L_{\text{moe}}$$

$$N_{\text{active}} = N_{\text{always}} + (k + s)\, P\, L_{\text{moe}}$$

Read them left to right. $$N_{\text{always}}$$ is the number of parameters that every token uses. $$E$$ is the number of routed experts in one MoE layer, and $$k$$ is how many of them the router picks per token. $$s$$ is the number of shared experts. $$P$$ is the size of one expert, and $$L_{\text{moe}}$$ is the number of MoE layers.

A tiny example first. Take a model with 10 always-on parameters, 2 MoE layers, and 4 routed experts of 5 parameters each, with no shared expert. The router picks 1 expert per token. Total: $$10 + 4 \times 5 \times 2 = 50$$. Active: $$10 + 1 \times 5 \times 2 = 20$$. The model stores 50 numbers but each token touches 20.

Now the real DeepSeek-V3, with values from its `config.json` file. One expert has three weight matrices of size $$7168 \times 2048$$, so $$P = 3 \times 7168 \times 2048 = 44{,}040{,}192$$. There are $$E = 256$$ routed experts, $$s = 1$$ shared expert, $$k = 8$$, and $$L_{\text{moe}} = 58$$ MoE layers (61 layers, of which the first 3 are dense). The expert part is $$257 \times 44.04\text{M} \times 58 = 656.5\text{B}$$ in total, but only $$9 \times 44.04\text{M} \times 58 = 23.0\text{B}$$ per token. The always-on part is about 14.6B. So the total is about 671B and the active count is about 37.6B.

The paper says "671B total parameters with 37B activated for each token", which matches.

The assumption under the formula is that every token uses the same number of experts, $$k$$, in every MoE layer. That holds for V2 and V3. In V4.1 it breaks in an interesting way, as the table below shows.

## Every model, in order

DeepSeek has released a model, an update or a paper almost every month since late 2023. The timeline below shows only the main line. The specialised models come in the next section.

{{FIG:timeline|The main DeepSeek language models from November 2023 to September 2026, with total and active parameters. The band underneath shows the weight license of the flagship models: the DeepSeek License until R1, then MIT.}}

> [!DEFINITION] Version suffixes like -0324
> DeepSeek names many updates by date. **V3-0324** is a V3 update from 24 March 2025; **R1-0528** is an R1 update from 28 May 2025. A suffix like **-Base** means the model before chat training; **-Exp** means experimental; **-Lite** and **-Flash** mean smaller, cheaper versions.

### 2023: a dense start

DeepSeek's first public models were **DeepSeek-Coder** and **DeepSeek LLM**. Coder, a set of code models from 1.3B to 33B, appeared in late October and early November 2023. LLM 7B and 67B followed on 29 November 2023. Both were dense models with the same layout as Meta's LLaMA. The 67B model was trained on 2 trillion tokens of English and Chinese text.

The DeepSeek LLM paper states its open stance in one sentence.

> [!PAPER] DeepSeek-AI (2024), DeepSeek LLM: Scaling Open-Source Language Models with Longtermism · §1 · page 3
> [![The introduction of the DeepSeek LLM paper, with release as much information as possible for community reference and 2 trillion tokens highlighted](/img/deepseek/papers/p1-llm-release.png)](/img/deepseek/papers/p1-llm-release.png)
>
> **What it says:** "we build from scratch open-source large language models, and release as much information as possible for community reference. We collect 2 trillion tokens for pre-training".
>
> **What it means:**
>
> "From scratch" is the key word. In 2023 many open models were fine-tuned copies of LLaMA. DeepSeek trained its own model from random numbers, on its own data, with its own tokenizer.
>
> "As much information as possible" is a promise about information, not about every artifact. The paper gives scaling laws, learning rates, batch sizes and data mixtures in detail. It does not ship the data or the training program. That split, generous with knowledge and closed on the full recipe, holds for every model in this part.

### 2024: experts, latent attention and V3

In January 2024 came **DeepSeekMoE 16B**, the first DeepSeek mixture of experts. It has 16.4B parameters and uses about 2.8B per token. Its paper introduced two ideas that every later main model kept. One is many small experts instead of a few big ones; the other is shared experts for every token. [Part 2](deepseek-2-the-v3-architecture.md) explains both.

In February came **DeepSeekMath 7B**. It is small, but its paper introduced GRPO, the reinforcement learning method behind R1 a year later. In March came **DeepSeek-VL**, the first vision-language model.

In May 2024, **DeepSeek-V2** joined the two new ideas into one large model.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model · Abstract · page 1
> [![The DeepSeek-V2 abstract, with 236B total parameters, of which 21B are activated for each token, context length of 128K tokens and The model checkpoints are available at highlighted](/img/deepseek/papers/p1-v2-abstract.png)](/img/deepseek/papers/p1-v2-abstract.png)
>
> **What it says:** "It comprises 236B total parameters, of which 21B are activated for each token, and supports a context length of 128K tokens." The last line: "The model checkpoints are available at https://github.com/deepseek-ai/DeepSeek-V2."
>
> **What it means:**
>
> Compare V2 with the 67B dense model from six months earlier. V2 stores 3.5 times more parameters but uses less than a third as many per token. Its context is 32 times longer: 128K tokens against 4K.
>
> The middle of the abstract names the second invention, Multi-head Latent Attention (MLA). It shrinks the memory that a model keeps for every token it has read, called the KV cache, by 93.3% against the 67B model. A small KV cache is what makes long context and cheap serving possible. MLA has its own page on this site: [MQA, GQA and MLA](attention-2-mqa-gqa-mla.md).
>
> The last line matters for this part. "Checkpoints are available" means the weights are downloadable. It says nothing about training code or data, and neither was released.

V2 was too big for most researchers to run. So DeepSeek also released a small sibling with the same design.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2 · §1 · page 5
> [![The paragraph that introduces DeepSeek-V2-Lite, with we also release DeepSeek-V2-Lite and for the open-source community highlighted](/img/deepseek/papers/p1-v2-lite.png)](/img/deepseek/papers/p1-v2-lite.png)
>
> **What it says:** "we also release DeepSeek-V2-Lite, a smaller model equipped with MLA and DeepSeekMoE, for the open-source community. It has a total of 15.7B parameters, where 2.4B are activated for each token."
>
> **What it means:**
>
> V2-Lite is the model to start with if you want to study DeepSeek's architecture on your own machine. With 16-bit numbers its weights take about 31 GB, which fits on one large GPU. It has the same MLA attention and the same kind of experts as the big model. Only the sizes are smaller.

The rest of 2024 filled in the line. **DeepSeek-Coder-V2** (June) continued training a V2 checkpoint on code. **DeepSeek-V2.5** (September) merged the chat and code models into one. **Janus** (October) generated and understood images in one model. **DeepSeek-VL2** (December) was a vision model built on experts. On 20 November a web-only preview called R1-Lite-Preview showed DeepSeek's first long-reasoning model; its weights were never released.

Then, on 26 December 2024, came **DeepSeek-V3**.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Abstract · page 1
> [![The DeepSeek-V3 abstract, with 671B total parameters with 37B activated for each token, 14.8 trillion, 2.788M H800 GPU hours and The model checkpoints are available at highlighted](/img/deepseek/papers/p1-v3-abstract.png)](/img/deepseek/papers/p1-v3-abstract.png)
>
> **What it says:** "a strong Mixture-of-Experts (MoE) language model with 671B total parameters with 37B activated for each token." It was pre-trained "on 14.8 trillion diverse and high-quality tokens" and "requires only 2.788M H800 GPU hours for its full training".
>
> **What it means:**
>
> V3 is the base of almost everything that followed. R1, V3.1, V3.2, Prover-V2 and Math-V2 all start from V3's architecture or a V3 checkpoint. If you learn one DeepSeek model in depth, learn this one. [Part 2](deepseek-2-the-v3-architecture.md) takes it apart.
>
> The GPU-hour number became famous. At $2 per GPU hour it is about $5.6 million. The paper itself warns what that number leaves out, as the next box shows.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §1 · page 5
> [![The cost paragraph, with excluding the costs associated with prior research and ablation experiments highlighted](/img/deepseek/papers/p1-v3-cost.png)](/img/deepseek/papers/p1-v3-cost.png)
>
> **What it says:** "Note that the aforementioned costs include only the official training of DeepSeek-V3, excluding the costs associated with prior research and ablation experiments on architectures, algorithms, or data."
>
> **What it means:**
>
> $5.576M is the rental price of the final training run only. It excludes the GPUs that DeepSeek owns, the failed and small trial runs, the salaries and the data work. It is like quoting the fuel for a rocket's last flight as the cost of the space programme. The number is honest, because the paper says exactly what it covers; many headlines dropped that sentence. [Part 3](deepseek-3-training-v3.md) breaks down the 2.788M GPU hours.

### 2025: reasoning, the MIT switch and sparse attention

On 20 January 2025 DeepSeek released **R1** and **R1-Zero**. Both have V3's architecture: 671B total, 37B active. R1-Zero learned to reason with reinforcement learning alone; R1 added a small amount of supervised data first. [Part 4](deepseek-4-r1-and-reasoning-by-rl.md) explains how.

The reasoning training was cheap next to the base model. The later Nature version of the R1 paper gives its cost: 147K H800 GPU hours, or about $294K at $2 per GPU hour. That is on top of V3's base training, not instead of it.

R1 came with six small distilled models: smaller open models fine-tuned on answers that R1 wrote.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning · §1 · page 2
> [![The end of the introduction, with we have distilled several smaller models, made them publicly available and We release DeepSeek-R1 series models to the public highlighted](/img/deepseek/papers/p1-r1-release.png)](/img/deepseek/papers/p1-r1-release.png)
>
> **What it says:** "To enable broader access to powerful AI at a lower energy cost, we have distilled several smaller models and made them publicly available." Then: "We release DeepSeek-R1 series models to the public at https://huggingface.co/deepseek-ai."
>
> **What it means:**
>
> The "smaller models" are not DeepSeek architectures. They are other companies' dense models, Qwen2.5 (1.5B, 7B, 14B, 32B) and Llama 3.1 and 3.3 (8B, 70B), fine-tuned on 800K samples that R1 wrote. Each one keeps its base model's license, so the Llama students also follow Meta's Llama license, not only MIT.
>
> This matters for "which model should I look at". If you want R1-style reasoning on a laptop or one GPU, a distilled model is the practical choice. If you want to study DeepSeek's own architecture, a distill teaches you nothing about it.

The rest of 2025 moved fast. **V3-0324** (March) improved V3 and moved the V3 line to the MIT license. **Prover-V2** (April) proved maths theorems in the Lean language. **R1-0528** (May) improved R1, with a distilled Qwen3-8B student. **V3.1** (August) joined chat and reasoning into one model with a "thinking" switch. **V3.1-Terminus** (September) was a polish of it.

On 29 September, **V3.2-Exp** added DeepSeek Sparse Attention: each token reads only the 2,048 most useful earlier tokens, not all of them. The final **V3.2** and **V3.2-Speciale** followed on 1 December. Between them came **DeepSeek-OCR** (October), which reads documents from images, and **DeepSeekMath-V2** (November), which checks its own proofs. [Part 5](deepseek-5-sparse-attention-and-v3-2.md) covers the sparse attention.

### 2026: V4, V4.1 and a million tokens

January 2026 brought two research releases: **Engram**, a memory module that looks up short word sequences in a large table, and **DeepSeek-OCR 2**. Then on 24 April, DeepSeek released a preview of **V4**.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence · Abstract · page 1
> [![The DeepSeek-V4 abstract, with 1.6T parameters (49B activated), 284B parameters (13B activated), one million tokens, more than 32T and The model checkpoints highlighted](/img/deepseek/papers/p1-v4-abstract.png)](/img/deepseek/papers/p1-v4-abstract.png)
>
> **What it says:** two models, "DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated)", both "supporting a context length of one million tokens", pre-trained "on more than 32T" tokens.
>
> **What it means:**
>
> V4 split the main line in two. Pro is the largest DeepSeek model so far, at 1.6 trillion parameters. Flash is smaller than V3 in total (284B against 671B) and uses about a third of V3's active parameters per token.
>
> The other headline is the context. V3 read 128K tokens; V4 reads one million, the length of several long novels. The abstract says that at that length, V4-Pro needs "only 27% of single-token inference FLOPs and 10% of KV cache" of V3.2. That saving came from a new attention design, which [Part 6](deepseek-6-v4-and-v4-1.md) explains.
>
> Note the date. The arXiv number of this paper starts with 2606, but arXiv lists it as submitted on 26 April 2026. Quote the date, not the number.

The April models were a preview. **V4-Flash-0731** and **V4-Pro-0813** replaced them as the official versions in July and August. Both carry a new speculative decoding module called DSpark, which guesses several tokens at once and checks them in one step. **V4-Flash-Vision-Exp** (August) was the first V4 model that sees images.

On 10 September 2026 came **V4.1-Flash**, the newest main model at the time of writing.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression · Abstract · page 1
> [![The V4.1-Flash abstract, with 552B backbone parameters, one million tokens, activates 16B parameters per token during decode but only 8B parameters during prefill and 45T tokens highlighted](/img/deepseek/papers/p1-v41-abstract.png)](/img/deepseek/papers/p1-v41-abstract.png)
>
> **What it says:** "a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens." The model "activates 16B parameters per token during decode but only 8B parameters during prefill".
>
> **What it means:**
>
> V4.1-Flash breaks the simple rule "one total, one active". Its layers form two halves, an encoder and a decoder. When the model reads your prompt (prefill), most prompt tokens go only through the first half, so they use about 8B parameters each. When it writes its answer (decode), each new token goes through both halves, so it uses about 16B.
>
> The total is also two numbers. The paper counts 552B "backbone" parameters plus 196B in two Engram memory tables. Do not add them into one figure without saying so; the tables are looked up, not multiplied like normal weights.
>
> It is also multimodal: it reads images natively. It was pre-trained on 45T tokens of text and images.

The minor update **V4.1-Flash-1017** came on 17 October 2026. Two things have not appeared as of 10 November 2026. DeepSeek's own news page refers to a coming "V4.1-Pro", but no weights exist. Press reports about a vision model called "Thinking with Visual Primitives" cite a GitHub repository that now returns "not found"; this series treats it as unconfirmed.

## The catalog in one table

Here is the main line, plus the most useful small models, in one table. Sizes come from the papers and model cards; contexts are the lengths the cards state. The license is the license of the weights.

| Date | Model | Total / active | Context | Weight license |
|---|---|---|---|---|
| Nov 2023 | DeepSeek-Coder 1.3B to 33B | dense | 16K | DeepSeek License |
| Nov 2023 | DeepSeek LLM 7B, 67B | dense | 4K | DeepSeek License |
| Jan 2024 | DeepSeekMoE 16B | 16.4B / 2.8B | 4K | DeepSeek License |
| Feb 2024 | DeepSeekMath 7B | dense | 4K | DeepSeek License |
| May 2024 | DeepSeek-V2 | 236B / 21B | 128K | DeepSeek License |
| May 2024 | DeepSeek-V2-Lite | 15.7B / 2.4B | 32K | DeepSeek License |
| Jun 2024 | DeepSeek-Coder-V2 (and Lite) | 236B / 21B (16B / 2.4B) | 128K | DeepSeek License |
| Sep 2024 | DeepSeek-V2.5 (and -1210) | 236B / 21B | 128K | DeepSeek License |
| Dec 2024 | DeepSeek-V3 | 671B / 37B | 128K | DeepSeek License |
| Jan 2025 | DeepSeek-R1, R1-Zero | 671B / 37B | 128K | MIT |
| Jan 2025 | R1-Distill (Qwen 1.5B to 32B, Llama 8B, 70B) | dense | as base model | MIT plus the base model's license |
| Mar 2025 | DeepSeek-V3-0324 | 671B / 37B | 128K | MIT |
| Apr 2025 | DeepSeek-Prover-V2 7B, 671B | dense; 671B / 37B | 32K; 128K | unclear (MIT file, card cites DeepSeek License) |
| May 2025 | DeepSeek-R1-0528 (and Qwen3-8B distill) | 671B / 37B | 128K | MIT |
| Aug 2025 | DeepSeek-V3.1 (Sep: Terminus) | 671B / 37B | 128K | MIT |
| Sep 2025 | DeepSeek-V3.2-Exp | 671B / 37B | 128K | MIT |
| Nov 2025 | DeepSeekMath-V2 | 671B class | 160K (config) | Apache-2.0 |
| Dec 2025 | DeepSeek-V3.2, V3.2-Speciale | 671B / 37B | 128K | MIT |
| Apr 2026 | DeepSeek-V4-Flash, V4-Pro (preview, with Base models) | 284B / 13B; 1.6T / 49B | 1M | MIT |
| Jul, Aug 2026 | DeepSeek-V4-Flash-0731, V4-Pro-0813 | 284B / 13B; 1.6T / 49B | 1M | MIT |
| Sep 2026 | DeepSeek-V4.1-Flash (Oct: -1017) | 552B + 196B Engram / 8B prefill, 16B decode | 1M | MIT |

### The same table, read from the real config files

Every model on Hugging Face ships a small text file, `config.json`, that describes its shape: the number of layers, the width, the number of experts. These files are a few kilobytes each, so anyone can download them without the weights. Our script `part1_configs.py` downloads 14 of them and counts the parameters from the config fields alone, with no published number as input.

The core of the count for the V2 and V3 families is a direct copy of the two formulas above:

```python
def count(c):
    d, L, V = c['hidden_size'], c['num_hidden_layers'], c['vocab_size']
    emb = V * d * (1 if c.get('tie_word_embeddings') else 2)      # input embedding + output head
    attn, norms = attention(c), 2 * d                              # MLA or standard attention, per layer
    ...
    expert = 3 * d * c['moe_intermediate_size']                    # one SwiGLU expert: 3 matrices
    E, S, k = c['n_routed_experts'], c['n_shared_experts'], c['num_experts_per_tok']
    common = emb + L * (attn + norms) + k_dense * dense_ffn + n_moe * router + d
    total = common + n_moe * (E + S) * expert
    active = common + n_moe * (k + S) * expert
    return total, active
```

And this is its real output, from configs fetched on 10 November 2026 (CPU only, a few seconds):

```text
model                      date     layers     d      experts ctx(config)  total(count) active(count)   published
-----------------------------------------------------------------------------------------------------------------
deepseek-coder-33b-base    2023-11      62  7168        dense         16K         33.3B         33.3B   33B / dense, 16K
deepseek-llm-67b-base      2023-11      95  8192        dense          4K         67.4B         67.4B   67B / dense, 4K
deepseek-moe-16b-base      2024-01      28  2048   64+2 top-6          4K         16.4B          2.8B   16.4B / 2.8B, 4K
DeepSeek-V2                2024-05      60  5120  160+2 top-6        160K        235.7B         21.4B   236B / 21B, 128K
DeepSeek-V2-Lite           2024-05      27  2048   64+2 top-6        160K         15.7B          2.7B   15.7B / 2.4B, 32K
DeepSeek-Coder-V2-Instruct 2024-06      60  5120  160+2 top-6        160K        235.7B         21.4B   236B / 21B, 128K
DeepSeek-V2.5              2024-09      60  5120  160+2 top-6        160K        235.7B         21.4B   236B / 21B, 128K
DeepSeek-V3                2024-12      61  7168  256+1 top-8        160K        671.0B         37.6B   671B / 37B, 128K
DeepSeek-R1                2025-01      61  7168  256+1 top-8        160K        671.0B         37.6B   671B / 37B, 128K
DeepSeek-V3.1              2025-08      61  7168  256+1 top-8        160K        671.0B         37.6B   671B / 37B, 128K
DeepSeek-V3.2              2025-12      61  7168  256+1 top-8        160K        671.0B         37.6B   671B / 37B, 128K
DeepSeek-V4-Flash          2026-04      43  4096  256+1 top-6          1M      >278.1B*        >7.6B*   284B / 13B, 1M
DeepSeek-V4-Pro            2026-04      61  7168  384+1 top-6          1M       >1.55T*       >28.2B*   1.6T / 49B, 1M
DeepSeek-V4.1-Flash        2026-09      40  5120  384+1 top-6          1M      >545.0B*        >9.9B*   552B + 196B Engram / 8B / 16B, 1M

* V4 and V4.1 rows count the expert weights only (a lower bound); attention, embeddings and the rest are not counted.
  V4.1-Flash: the 9.9B is all 40 layers; DeepSeek says prefill uses 8B because most prompt tokens skip the decoder half.
ctx(config) is max_position_embeddings; the model cards state the context the model was trained and tested for.

DeepSeek-V3 share of weights one token uses: 5.6%  (37.6B of 671.0B)
DeepSeek-V4-Flash experts alone: 278.1B, which is 98% of the published 284B
Config files read: 14, total size 21.4 KB
```

Read the output column by column. The counted totals match the published totals for every dense, V2 and V3 model, to the first decimal place. So the published sizes are not marketing round numbers; they follow from the shapes in the files.

The active column has one small mismatch: V2-Lite counts 2.7B, while the paper says 2.4B. Our count includes the input embedding and the output head, 0.42B together in V2-Lite. DeepSeek's papers are not consistent on this point, so small differences in active counts are a matter of convention.

Three more lessons hide in the output. First, R1, V3.1 and V3.2 have the same shape as V3; they differ in training, not in size. (V3.2 adds a small 0.85B attention indexer that this count leaves out.)

Second, the `max_position_embeddings` field is not the context length: V2-Lite's config says 160K, but its paper says 32K. Trust the card. Third, in V4-Flash the experts alone are 98% of all weights, so almost all of a modern DeepSeek model is experts.

{{FIG:sizes|Total and active parameters of eight DeepSeek models, on a log scale. From V2 on, each token uses only 2% to 9% of the stored weights. The dense 67B model uses all of its weights for every token.}}

## The families: main line and specialists

The catalog has two kinds of models. The **main line** is the general chat model that DeepSeek serves through its API. The **specialists** are trained for one job, and most of them start from a main-line model.

{{FIG:families|DeepSeek models in six lanes. The main line runs along the top; reasoning, code, maths, vision and research models sit below it. Arrows show where one model was built on another, for example Prover-V2 on V3 and Math-V2 on V3.2.}}

Over time the specialists flow back into the main line. Coder-V2 merged into V2.5, and R1's reasoning merged into V3.1, which has one switch for "thinking" and "non-thinking". Engram, first a research paper, became part of V4.1. Vision moved from separate VL and Janus models into V4.1-Flash itself. So in late 2026 the main line does most jobs, and the specialists remain for narrow tasks such as formal proofs and OCR. [Part 8](deepseek-8-the-specialised-models.md) covers each specialist.

Which model should you look at? It depends on what you want to do.

| You want to... | Look at | Why |
|---|---|---|
| Learn DeepSeek's architecture | V3 (paper and reference code), then V2-Lite to run it | V3 is the base of most later models; V2-Lite has the same ideas at 16B |
| Run the newest model yourself | V4.1-Flash-1017 or V4-Pro-0813 | Both MIT; V4.1-Flash uses 8B to 16B per token, V4-Pro is the largest |
| Get R1-style reasoning on one GPU | R1-Distill-Qwen 7B to 32B, or R1-0528-Qwen3-8B | Dense, small, fine-tuned on R1's answers |
| Study reinforcement learning for reasoning | R1 and R1-Zero, with the R1 paper | The reference case; [Part 4](deepseek-4-r1-and-reasoning-by-rl.md) |
| Study long context and sparse attention | V3.2, then V4 | [Part 5](deepseek-5-sparse-attention-and-v3-2.md) and [Part 6](deepseek-6-v4-and-v4-1.md) |
| Prove theorems in Lean | Prover-V2 (7B or 671B) | Trained for formal proofs |
| Read documents from images | DeepSeek-OCR 2 | Small (about 3B), Apache-2.0 |
| Generate and understand images | Janus-Pro 7B | One model for both directions |
| Call a model over the internet | `deepseek-flash` in the API | V4.1-Flash-1017, the cheapest current option |

## What "open" can mean

"DeepSeek is open source" is a common sentence, and it is partly wrong. To see why, follow a model from raw data to an answer on your screen.

{{FIG:release_pipeline|Six steps from training data to answers. DeepSeek publishes the last three for its main models: the weights, the code to run them, and the answers through its API. The first three, the data, the training code and the training run itself, stay in-house; papers describe them.}}

An analogy helps. A compiled program is a file that runs, but you cannot easily read or rebuild it. Its source code is what programmers edit, and anyone with the source can rebuild the program. Model weights are like the compiled program. The training data and training code are like the source.

There is a twist. With ordinary software, a program cannot be changed much without its source. With a model, you can change the weights directly: fine-tune them, prune them, quantize them or distill them. So weights alone give you far more freedom than a compiled program does. That is why open weights are so useful, even without the full recipe.

> [!DEFINITION] Open weights
> A model has **open weights** when anyone can download its trained parameters and use them under a published license. Open weights let you run, study and fine-tune the model. They do not let you rebuild it from scratch.

> [!DEFINITION] Open source (for AI)
> The Open Source Initiative published its **Open Source AI Definition 1.0** in October 2024. It asks for three things under open terms. The first is the **parameters** (weights), and the second is the **complete code** to train and run the system. The third is **data information** detailed enough "so that a skilled person can build a substantially equivalent system". The raw data itself need not be shared, but its sources, selection and processing must be described in full.

> [!DEFINITION] Source-available and responsible-AI licenses
> Some licenses let you read and use a model but add rules about how you may use it, such as a list of forbidden uses. These are not open source licenses in the classic sense, because open source licenses forbid such use restrictions. DeepSeek's first license was of this kind.

With these definitions, the question "is DeepSeek open?" becomes six smaller questions. The matrix answers them for the main line.

{{FIG:matrix|What DeepSeek released for nine main-line models. Papers and weights are always there; inference code is there from V2 on; training code and training data are never released, except narrow pieces for Coder and V4. The license changed from the DeepSeek License to MIT with R1.}}

The sections below take the columns one at a time.

## Weights and their licenses

Every DeepSeek main model has downloadable weights on Hugging Face, usually as both a Base model and a chat model. V3 and later models ship their main weights in FP8, an 8-bit number format, while V4 stores its experts in 4-bit FP4. [Part 7](deepseek-7-running-deepseek.md) shows what that means for memory.

The license changed over time.

{{FIG:licenses|When each kind of weight license first appeared on a DeepSeek model: the DeepSeek License from October 2023, MIT from R1 in January 2025 and for the V3 line from V3-0324, Apache-2.0 for some specialised models from late 2025, and MIT for all of V4.}}

### The DeepSeek License (2023 to early 2025)

The **DeepSeek License Agreement, Version 1.0**, is dated 23 October 2023. It covered all models up to and including the first V3. It allows commercial use. But its Attachment A lists forbidden uses, starting with "For military use in any way" and use to harm minors. It also forbids false information that harms others, discrimination, and fully automated decisions that harm people's legal rights.

Two clauses matter more than they look. First, you must pass the same restrictions on to all of your own users. Second, the license defines "Derivatives of the Model" to include "distillation methods" and models trained on "synthetic data by the Model". So under this license, a small model trained on DeepSeek-V3's answers also carries V3's restrictions. The governing law is the law of the People's Republic of China, with courts in Hangzhou.

### MIT (from R1, January 2025)

R1 changed this. Its model card says: "This code repository and the model weights are licensed under the MIT License." It adds that the R1 series supports "commercial use, allow for any modifications and derivative works, including, but not limited to, distillation for training other LLMs."

Compare that with the DeepSeek License. The old license treated distillation as a derivative that kept all the restrictions. The new one names distillation as explicitly allowed. That one change made R1 the default teacher for many small reasoning models in 2025.

The V3 line followed two months later. DeepSeek's release note for V3-0324 says: "Models are now released under the MIT License, just like DeepSeek-R1!" Every main model since then, V3.1, V3.2, V4 and V4.1, uses MIT, including the V4 Base models.

> [!WARNING] Check the license file, not the label
> Hugging Face shows a license tag on every model page, and DeepSeek's tags are sometimes wrong. The original DeepSeek-V3 repository still ships the DeepSeek License as its `LICENSE-MODEL` file; only V3-0324 and later are MIT. Janus is tagged MIT, but its card says it is "subject to DeepSeek Model License". Prover-V2 ships an MIT file, but its card points to the DeepSeek License. The R1 distills on Llama keep Meta's Llama license. Read the license file and the card text before you ship a product.

### Apache-2.0 (late 2025 on)

A third license appeared in November 2025. DeepSeekMath-V2, DeepSeek-OCR 2, the Engram code and the DeepInfer engine use **Apache-2.0**. It is as permissive as MIT, and it adds an explicit patent grant: contributors promise not to sue users over patents in the code. For a user, MIT and Apache-2.0 give the same practical freedom.

## Inference code: how to run the weights

Weights are only numbers. To get answers you need code that knows the model's shape. It must wire the layers, apply attention and route tokens to experts.

DeepSeek's practice here improved over time. For DeepSeek LLM 67B, no special code was needed: it has LLaMA's layout, and Hugging Face Transformers already ran it. For DeepSeekMoE and V2, DeepSeek shipped a `modeling_deepseek.py` file inside the Hugging Face repository.

For V3, DeepSeek published a short **reference implementation** in PyTorch, in the `inference/` folder of the GitHub repository. It has the model (`model.py`), a text generator, the FP8 kernels and a tool to convert FP8 weights to 16-bit. R1 uses the same code, because it has the same architecture. V3.2, V4 and V4.1 ship their own `inference/` folders inside the Hugging Face repositories.

V4 and V4.1 add an `encoding/` folder. Most models describe their chat format as a small template file. DeepSeek instead ships Python code (and later the `deepseek-recipe` library in Rust and Python) that turns a conversation into tokens. If you write your own serving code, use this encoder; a home-made template can silently break tool calls or thinking mode.

Reference code is written to be read, not to be fast. In practice people run DeepSeek models with open engines such as vLLM, SGLang and llama.cpp, which add DeepSeek support, often on release day. The V4 paper explains why the reference code exists.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §3.1 · page 16
> [![The mega-kernel paragraph, with We have open-sourced the CUDA-based mega-kernel implementation named MegaMoE as a component of DeepGEMM highlighted](/img/deepseek/papers/p1-v4-megamoe.png)](/img/deepseek/papers/p1-v4-megamoe.png)
>
> **What it says:** the fused expert-parallel scheme "achieves 1.50 ~ 1.73x speedup for general inference workloads", and "We have open-sourced the CUDA-based mega-kernel implementation named MegaMoE as a component of DeepGEMM."
>
> **What it means:**
>
> A paper can describe a fast kernel in words, but other engineers then spend months rebuilding it. Here DeepSeek ships the kernel itself, inside its DeepGEMM library. Elsewhere the same paper says it gives "an open-source implementation to specify more details unambiguously" for its new attention.
>
> This is DeepSeek's habit in 2025 and 2026: release the code that runs and serves the model, sometimes before the engines catch up. The paper also says the scheme was tested on both NVIDIA GPUs and Huawei Ascend chips. That second platform matters for the infrastructure section below.

## Training code: what is missing

This is the column where the answer is almost always "no", so it needs to be exact.

DeepSeek has never released the code that pre-trains its main models, nor the reinforcement learning code that turned V3 into R1. The repositories for LLM, V2, V3, R1, V3.2, the vision models, Math and Prover hold usage examples, inference code, evaluation scripts and papers. None holds a training program for the model itself.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.2 · page 12
> [![The training framework paragraph, with HAI-LLM framework and crafted by our engineers from the ground up highlighted](/img/deepseek/papers/p1-v3-hai-llm.png)](/img/deepseek/papers/p1-v3-hai-llm.png)
>
> **What it says:** "The training of DeepSeek-V3 is supported by the HAI-LLM framework, an efficient and lightweight training framework crafted by our engineers from the ground up."
>
> **What it means:**
>
> HAI-LLM is DeepSeek's in-house training program. It splits V3 across 2,048 GPUs, schedules the work and saves checkpoints. It is not public. So even with the data, you could not repeat V3's training run as DeepSeek did; you would have to write that layer yourself.
>
> The paper does describe how the framework splits the work, for example 16-way pipeline parallelism and 64-way expert parallelism. Some building blocks are public too, as the infrastructure section shows. You get the bricks and the drawings, but not the building crane. In 2025 DeepSeek also wrote that its "training framework relies on PyTorch", the common open library.

There are three narrow exceptions, and none of them trains a main model:

1. **DeepSeek-Coder** ships a fine-tuning script (`finetune_deepseekcoder.py`). It adapts a finished Coder model to your own data; it does not pre-train.
2. **ESFT** (2024) ships full training code for its method, which fine-tunes only the experts that matter for a task, plus small task datasets.
3. **DeepSpec** (2026) ships training and evaluation code for the small draft models used in speculative decoding, such as DSpark. It does not train V4 itself.

## Training data: described, never shipped

The data is the most closed part. DeepSeek describes its data in its papers: the size, the languages, the share of code and maths, the filtering steps. It has never released a pre-training corpus.

| Model | Pre-training tokens (as stated in the paper or card) |
|---|---|
| DeepSeek LLM 67B | 2T, mainly English and Chinese |
| DeepSeek-Coder | 2T, 87% code |
| DeepSeek-V2 | 8.1T |
| DeepSeek-V3 | 14.8T |
| DeepSeek-V3.1 | V3 plus 840B tokens of long-context training |
| DeepSeek-V4 | more than 32T |
| DeepSeek-V4.1-Flash | 45T, text and images |

The released datasets are small and special-purpose. They include a synthetic Lean proof dataset for Prover-V1, the ProverBench evaluation set, and the problems and code from V3.2's maths and programming competition runs. These help you test a model, not train one.

There are also good reasons for the gap, and the R1 paper shows one of them.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1 · §3.1 · page 7
> [![The reward-model data paragraph, with publicly available open-source datasets or from users who have explicitly consented to share their data highlighted](/img/deepseek/papers/p1-r1-data.png)](/img/deepseek/papers/p1-r1-data.png)
>
> **What it says:** the prompts for the helpfulness reward model "are sourced either from publicly available open-source datasets or from users who have explicitly consented to share their data for the purpose of model improvement."
>
> **What it means:**
>
> Part of DeepSeek's training data comes from its own users. Even with their consent to use the data for training, publishing it would be a different matter. Web text has a similar problem: it holds copyrighted pages and personal details that a lab may use but cannot redistribute.
>
> This is why the Open Source AI Definition asks for "data information" and not for the raw data itself. By that standard the question becomes: is DeepSeek's description detailed enough to build "a substantially equivalent system"? Our reading is no. The papers give token counts and filtering ideas, but not the source lists, the filter code or the exact mixtures.

## Papers: the most open part

If the data is the most closed part, the papers are the most open. DeepSeek has published more than 30 technical papers since 2023, most of them on arXiv and free to read. Some, such as Fire-Flyer, the R1 paper (in *Nature*, September 2025) and the hardware lessons of V3 (ISCA 2025), went through peer review.

The papers are unusually detailed for a commercial lab. The V3 report gives the full training cost breakdown, the FP8 recipe and its failure modes. The V4 and V4.1 papers give layer-by-layer plans that match the `config.json` files exactly.

A few releases have no paper of their own. V3-0324 and V3.1 have only model cards and release notes. The V3.2-Exp report was published as a PDF in the GitHub repository, not on arXiv. The V4 report first lived on Hugging Face and later moved to arXiv.

The V3 paper ends with a line that explains why DeepSeek publishes so much.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §6 · page 35
> [![The conclusion, with DeepSeek consistently adheres to the route of open-source models with longtermism highlighted](/img/deepseek/papers/p1-v3-longtermism.png)](/img/deepseek/papers/p1-v3-longtermism.png)
>
> **What it says:** "DeepSeek consistently adheres to the route of open-source models with longtermism, aiming to steadily approach the ultimate goal of AGI".
>
> **What it means:**
>
> "Longtermism" appears in DeepSeek's first paper title, its release notes and its V4 announcement. The bet seems to be that sharing ideas brings more back, in bug fixes, engine support and reputation, than it costs. The papers and weights fit that bet, and they cost DeepSeek little once the model is trained.
>
> Note the words, though: "open-source models". DeepSeek uses "open source" in the everyday sense of "downloadable". By the OSI definition, its models are open weights. Both statements can be true; they use one word with two meanings.

## The open infrastructure

The most surprising part of DeepSeek's openness is not a model at all. It is the low-level software that makes training and serving fast.

In the last week of February 2025, DeepSeek held an **Open Source Week**: "we'll open-source 5 repos, one daily drop". It released, among others:

| Day | Repository | What it does |
|---|---|---|
| 1 | FlashMLA | Fast MLA attention kernel for decoding on Hopper GPUs |
| 2 | DeepEP | Sends tokens between GPUs for expert parallelism |
| 3 | DeepGEMM | Fast FP8 matrix multiplication for dense and MoE layers |
| 4 | DualPipe, EPLB, profile-data | V3's pipeline schedule, its expert load balancer, and real profiling traces |
| 5 | 3FS, smallpond | The Fire-Flyer file system and a data processing tool on top of it |
| 6 | A design note | How DeepSeek serves V3 and R1 in production |

> [!DEFINITION] Expert parallelism
> **Expert parallelism** spreads the experts of one MoE layer across many GPUs. Each token travels to the GPUs that hold its chosen experts and back. DeepEP does this travel; [Beyond one GPU](llm-inference-7-beyond-one-gpu.md) on this site explains the idea.

These are parts of the real systems behind V3 and R1, not demos. DualPipe and EPLB are training pieces too: they are part of how DeepSeek trained V3. The full training program still stays closed.

{{FIG:infra|DeepSeek's open infrastructure in five layers. Open Source Week in February 2025 released the storage, kernel, communication and balancing pieces. Later releases added serving (DeepInfer, DeepSpec), more kernels (TileKernels, DeepSelect) and ports to Huawei Ascend chips.}}

The newer releases fill the gaps. **TileKernels** (April 2026) holds kernels for MoE routing, FP8 and FP4 quantization, Engram and mHC, V4's wider residual stream ([Part 6](deepseek-6-v4-and-v4-1.md)). Its README says "All kernels have been used in our internal training and inference". **DeepSpec** (June 2026) trains the draft models for speculative decoding. **DeepSelect** (September 2026) holds the top-k selection kernels for sparse attention. **DeepGEMM-Ascend** and **DeepEP-Ascend** (September 2026) port the main libraries to Huawei's Ascend 950 chips, a sign that DeepSeek now runs on Chinese hardware too.

### The inference engine: a reversal

One release shows how DeepSeek's position moved. In April 2025 it published a note called "The Path to Open-Sourcing the DeepSeek Inference Engine". It said its engine was "based on an early fork of vLLM from over a year ago", heavily changed and tied to internal tools. So DeepSeek decided not to release it. It chose instead to give features to the open engines vLLM and SGLang.

Eighteen months later, on 16 October 2026, DeepSeek released **DeepInfer** under Apache-2.0. Its README calls it "the high-performance inference framework that powers DeepSeek's large-scale production services". The release has limits. It supports one model, V4.1-Flash, on NVIDIA SM100 (Blackwell) GPUs and Ascend 950 chips, for offline inference only.

It leaves out "A KV cache system", the split between prefill and decode machines, and load balancing across replicas. So you get the engine that runs one machine, not the whole service.

## Open weights or open source? A scorecard

Now apply the OSI definition to DeepSeek's newest main model, V4.1-Flash, one requirement at a time.

| OSI requirement | What DeepSeek provides | Our reading |
|---|---|---|
| Parameters under open terms | Full weights, MIT | Yes |
| Code to run the system | Reference inference code, encoder, DeepInfer, kernels | Yes |
| Code to train the system | Not released; some kernels and balancers are | No |
| Code to process and filter data | Not released | No |
| Data information detailed enough to rebuild | Token counts and methods in the paper | No |

So DeepSeek's main models are **open-weight models with open inference and an open low-level stack**. They are not open source AI by the OSI definition.

> [!TIP] In production
> For almost every practical use, open weights under MIT are what you need. You can run the model on your own hardware, fine-tune it, quantize it, distill it and sell a product built on it. You cannot audit exactly what went into training, and you cannot retrain it with one part of the data removed. If your use needs those two things, for example a strict legal audit of training data, open weights are not enough.

## The API: what DeepSeek serves, and at what price

DeepSeek also sells access to its models through an API, a web service that programs can call. The API follows the OpenAI request format at `https://api.deepseek.com`. Since V3.1 (August 2025) it also accepts the Anthropic request format at `https://api.deepseek.com/anthropic`.

> [!DEFINITION] Context caching (cache hit and cache miss)
> Many requests start with the same text, such as a long system prompt. The server can keep the work it did on that text and reuse it. Tokens served from this store are a **cache hit** and cost much less. New tokens are a **cache miss**. DeepSeek launched context caching on disk in August 2024.

### Which model is behind each name

The model names in the API do not always match the model names on Hugging Face, and they changed often.

| From | Name | Model behind it |
|---|---|---|
| May 2024 | `deepseek-chat`, `deepseek-coder` | V2, then V2-0628 and Coder-V2 |
| Sep 2024 | `deepseek-chat` | V2.5, then V2.5-1210 |
| Dec 2024 | `deepseek-chat` | V3, then V3-0324 (March 2025) |
| Jan 2025 | `deepseek-reasoner` | R1, then R1-0528 (May 2025) |
| Aug 2025 | `deepseek-chat` and `deepseek-reasoner` | V3.1 without and with thinking, then Terminus, V3.2-Exp and V3.2 |
| Apr 2026 | `deepseek-v4-flash`, `deepseek-v4-pro` | V4 preview; the old names route to V4-Flash and retire on 24 July 2026 |
| Jul, Aug 2026 | same | V4-Flash-0731, V4-Pro-0813 |
| Sep 2026 | `deepseek-flash` | V4.1-Flash; V4-Flash names route here |
| Oct 2026 (now) | `deepseek-flash`, `deepseek-v4-pro` | V4.1-Flash-1017 and V4-Pro-0813 |

In September 2026 DeepSeek first said that V4-Pro requests would move to V4.1-Flash. It then changed its mind: "we have decided to continue providing API services for DeepSeek V4 Pro after September 14, 2026". So two models are served today.

### Prices over time

All prices below are in US dollars per one million tokens. They come from DeepSeek's pricing page and its archived copies.

{{FIG:prices|DeepSeek API prices per million tokens from November 2024 to November 2026. The default model's output price rose with V3 and V3.1, fell to $0.42 with V3.2-Exp and to $0.28 with V4-Flash, and is $1.20 at peak for V4.1-Flash. The premium line is R1, then V4-Pro.}}

| From | Model | Input, cache hit | Input, cache miss | Output |
|---|---|---|---|---|
| Nov 2024 (earliest capture) | V2.5 | $0.014 | $0.14 | $0.28 |
| 26 Dec 2024 to 8 Feb 2025 | V3, at V2's prices | $0.014 | $0.14 | $0.28 |
| 8 Feb 2025 | V3 | $0.07 | $0.27 | $1.10 |
| 20 Jan 2025 | R1 | $0.14 | $0.55 | $2.19 |
| 5 Sep 2025 | V3.1, one price for both modes | $0.07 | $0.56 | $1.68 |
| 29 Sep 2025 | V3.2-Exp (and V3.2 from December) | $0.028 | $0.28 | $0.42 |
| 24 Apr 2026 | V4-Flash | $0.028 | $0.14 | $0.28 |
| 24 Apr 2026 | V4-Pro, list price (75% off at launch) | $0.145 | $1.74 | $3.48 |
| 16 Aug 2026, peak | V4-Flash-0731 | $0.014 | $0.44 | $1.32 |
| 16 Aug 2026, peak | V4-Pro-0813 | $0.044 | $1.32 | $3.96 |
| 10 Sep 2026, peak (now) | V4.1-Flash (`deepseek-flash`) | $0.006 | $0.30 | $1.20 |

Two rules complete the picture. From 26 April 2026, every cache-hit price was cut to one tenth; the table shows the later prices. From 16 August 2026, prices have a peak and an off-peak level: "Off-peak rates are half of the peak rates". Peak hours are 01:00 to 04:00 and 06:00 to 10:00 UTC on weekdays. An earlier off-peak discount (16:30 to 00:30 UTC) ran in 2025 and ended on 5 September 2025.

A worked example shows what these prices mean.

Say an app sends 10,000 requests a day to `deepseek-flash` at peak. Each request has 2,000 input tokens, of which 1,500 are a cached system prompt, and 500 output tokens. Per day that is 15M cache-hit tokens, 5M cache-miss tokens and 5M output tokens. The bill is $$15 \times 0.006 + 5 \times 0.30 + 5 \times 1.20 = 0.09 + 1.50 + 6.00 = \$7.59$$ per day. At off-peak hours it would be half, $3.80. Output dominates the bill, and the cached prompt costs almost nothing.

> [!WARNING] Prices change often
> DeepSeek changed prices at least eight times in two years, both up and down. One gap is also open: we found no archived pricing page between May and mid-August 2026, so V4 prices in that window are not confirmed. Always read the live pricing page before you plan a budget.

## What you can download, run and rebuild

Here is the whole part as one checklist.

You **can** download the weights of every main DeepSeek model, from LLM 67B to V4.1-Flash-1017, plus most specialists. Since R1 (and since V3-0324 for the V3 line), almost all of them are MIT, so commercial use, fine-tuning and distillation are allowed. Read the license file of older models and of the few with mixed labels.

You **can** run them with DeepSeek's reference code, with DeepInfer (V4.1-Flash only), or with open engines such as vLLM, SGLang and llama.cpp. The big ones need many GPUs; [Part 7](deepseek-7-running-deepseek.md) does the memory arithmetic.

You **can** read how they were built in more than 30 papers. You can also reuse the kernels, communication libraries, schedulers and file system that DeepSeek trains and serves with.

You **cannot** download the pre-training data, the data filters, the HAI-LLM training framework, or the reinforcement learning code behind R1. So you cannot reproduce any DeepSeek main model exactly. Anything you train from the papers is a new model, not a copy.

> [!TAKEAWAYS]
> - DeepSeek's main line runs LLM 67B (Nov 2023), V2, V2.5, V3 (Dec 2024), R1 (Jan 2025), V3.1, V3.2, V4 (Apr 2026) and V4.1-Flash (Sep 2026).
> - From V2 on, the main models are mixtures of experts: V3 stores 671B parameters but uses about 37B per token, about 5.6% of the weights.
> - Counting parameters from the real `config.json` files reproduces the published totals of every dense, V2 and V3 model to the first decimal place.
> - V4-Pro (1.6T / 49B) is the largest DeepSeek model; V4.1-Flash (552B backbone plus 196B Engram) uses 8B per token in prefill and 16B in decode.
> - Weights were under the DeepSeek License, whose use restrictions even cover distilled models. R1 moved to MIT in January 2025, and the V3 line followed with V3-0324.
> - DeepSeek publishes weights, inference code, papers and much of its low-level infrastructure, including FlashMLA, DeepEP, DeepGEMM, 3FS and, since October 2026, DeepInfer.
> - It has never published the pre-training data, the HAI-LLM training framework, or R1's reinforcement learning code. So its models are open weights, not open source AI by the OSI definition.
> - The API now serves `deepseek-flash` (V4.1-Flash-1017, $0.30 input and $1.20 output per million tokens at peak) and `deepseek-v4-pro` (V4-Pro-0813); off-peak prices are half.

**Next:** [Part 2, the V3 architecture](deepseek-2-the-v3-architecture.md) opens the model that most of this catalog is built on. It covers many small experts, latent attention and load balancing without an extra loss. It also counts exactly where V3's 671 billion parameters live.

## References

1. DeepSeek-AI. [*DeepSeek LLM: Scaling Open-Source Language Models with Longtermism*](https://arxiv.org/abs/2401.02954). 2024.
2. DeepSeek-AI. [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434). 2024.
3. DeepSeek-AI. [*Fire-Flyer AI-HPC: A Cost-Effective Software-Hardware Co-Design for Deep Learning*](https://arxiv.org/abs/2408.14158). SC24, 2024.
4. DeepSeek-AI. [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437). 2024.
5. DeepSeek-AI. [*DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning*](https://arxiv.org/abs/2501.12948). 2025. Nature version: [*DeepSeek-R1 incentivizes reasoning in LLMs through reinforcement learning*](https://www.nature.com/articles/s41586-025-09422-z), Nature 645, 2025.
6. DeepSeek-AI. [*DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*](https://arxiv.org/abs/2512.02556). 2025.
7. DeepSeek-AI. [*DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*](https://arxiv.org/abs/2606.19348). 2026.
8. DeepSeek-AI. [*DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*](https://arxiv.org/abs/2609.19969). 2026.
9. Model weights and cards: [deepseek-ai on Hugging Face](https://huggingface.co/deepseek-ai); code: [deepseek-ai on GitHub](https://github.com/deepseek-ai); the [DeepSeek License Agreement](https://github.com/deepseek-ai/DeepSeek-V3/blob/main/LICENSE-MODEL); the [R1 model card](https://huggingface.co/deepseek-ai/DeepSeek-R1).
10. Infrastructure: [open-infra-index](https://github.com/deepseek-ai/open-infra-index) (Open Source Week and the inference engine note), [DeepInfer](https://github.com/deepseek-ai/DeepInfer), [TileKernels](https://github.com/deepseek-ai/TileKernels), [DeepSpec](https://github.com/deepseek-ai/DeepSpec).
11. API: [change log](https://api-docs.deepseek.com/updates), [pricing](https://api-docs.deepseek.com/quick_start/pricing), [V3-0324 release note](https://api-docs.deepseek.com/news/news250325), [V4.1-Flash release note](https://api-docs.deepseek.com/news/news260910).
12. Open Source Initiative. [*The Open Source AI Definition 1.0*](https://opensource.org/ai/open-source-ai-definition). 2024.
13. SemiAnalysis. [*DeepSeek Debates*](https://semianalysis.com/2025/01/31/deepseek-debates/). 2025 (third-party estimate, cited only as such).
