---
title: "SGLang Explained: How It Reuses Work, and How It Differs from vLLM"
description: "RadixAttention from a handbook example: KV reuse maths, real prefill timings, cache and scheduling simulations, and a fair comparison with today's vLLM."
date: 2026-10-08
tags: [inference, sglang, vllm, kv-cache, llm]
series: "LLM Inference from the Ground Up"
series_part: 5
motif: graph
accent: "#4fc3d9"
---

You are building an assistant for a company's employee handbook. Every request includes the same instructions and the same handbook, followed by a question:

```text
Request 1: instructions + handbook + "How much annual leave do I get?"
Request 2: instructions + handbook + "Can I carry unused leave forward?"
Request 3: instructions + handbook + "How do I apply for leave?"
```

The model must answer three different questions. But should it process the entire handbook from scratch three times?

**SGLang helps avoid that repeated work.** It is an inference engine: the software that loads a language model, accepts requests, runs the model, and sends back its answers. One of its central ideas, **RadixAttention**, finds prompt beginnings the model has already processed and reuses the saved computation.

vLLM, the engine from [Part 3](llm-inference-3-vllm.md), also reuses repeated prompt beginnings, and in its current engine it does so by default. So the interesting question is not "which one caches?" Both do. The interesting questions are how each engine finds the saved work, what it throws away when memory runs out, in what order it runs waiting requests, and when any of this actually makes the user wait less.

We will follow the handbook assistant through those decisions, one at a time:

1. What the engine saves, how big it is, and how much work it saves, with the maths and a **real measurement** on a real model.
2. How SGLang finds saved work with a **radix tree**, and how it decides what to evict.
3. How vLLM finds the same work with **hashed blocks**, and where the two designs really differ.
4. Why the **order** in which requests run can matter as much as the cache itself.
5. A second idea from the same paper: **jump-forward decoding** for JSON output.
6. A fair comparison table for the versions available today, and a recipe to test both engines yourself.

> [!NOTE] Three kinds of numbers in this article
> Every number comes with its source, and they are kept apart.
> **Measured:** real prefill timings of Qwen2.5-0.5B (BF16) on an Apple M5 Pro GPU with PyTorch, the same setup as Parts 1 to 3. This measures the reuse operation itself; it is *not* a run of SGLang or vLLM.
> **Simulated:** cache and scheduling simulations in pure Python on a CPU. Where they report milliseconds, the times come from a cost model fitted to the measured timings. No NVIDIA GPU was available for this article, so no engine was benchmarked.
> **Reported:** results that the SGLang authors, the SGLang team or the vLLM team published, always with the hardware and versions they used.
> Engine defaults and flags were checked in the source code of SGLang (commit `b2cb249`) and vLLM (commit `c23ca06`), both from 8 October 2026.

## 1. What the inference engine does with your question

A model is a set of learned weights plus the computations that use them. Serving that model to many people takes more than running one forward pass. Someone has to queue incoming requests, put work into batches, allocate memory, and return generated tokens to the right user. That is the inference engine's job.

> [!DEFINITION] Token
> A piece of text represented by an integer ID. A token might be a word, part of a word, or punctuation. The model processes token IDs rather than the characters you see on screen.

Our first handbook question passes through two main stages:

| Stage | What happens in our assistant | What the user sees |
|---|---|---|
| **Prefill** | The engine runs the model over the instructions, handbook, and question, building the state needed to answer. | Waiting for the answer to begin. |
| **Decode** | The model generates further answer tokens, using the prompt and the answer written so far. | The answer appearing a piece at a time. |

Prefill also provides the prediction used to select the first output token. Decode then continues from it. [Part 1](llm-inference-1-prefill-and-decode.md) explains these two stages in more depth.

<figure class="fig"><svg viewBox="0 0 760 345" role="img" aria-label="Instructions, handbook and question enter prefill, which builds KV and predicts the first output token. Decode continues the answer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="30.0" y="25.0" text-anchor="start">INPUT</text><rect class="box-1" x="20.0" y="40.0" width="720.0" height="52.0" rx="8"/><text class="t-note" x="380.0" y="73.0" text-anchor="middle">Instructions + handbook + question</text><line class="edge" x1="380.0" y1="92.0" x2="380.0" y2="118.0"/><rect class="box-2" x="100.0" y="118.0" width="560.0" height="60.0" rx="8"/><text class="t-note" x="380.0" y="144.0" text-anchor="middle">PREFILL: process the supplied text</text><text class="t-tick" x="380.0" y="165.0" text-anchor="middle">Build KV state and select the first answer token</text><line class="edge" x1="380.0" y1="178.0" x2="380.0" y2="204.0"/><rect class="box-1" x="100.0" y="204.0" width="560.0" height="60.0" rx="8"/><text class="t-note" x="380.0" y="230.0" text-anchor="middle">DECODE: continue the answer</text><text class="t-tick" x="380.0" y="251.0" text-anchor="middle">Read saved KV, generate the next token, repeat</text><line class="edge" x1="380.0" y1="264.0" x2="380.0" y2="291.0"/><text class="t-note" x="380.0" y="315.0" text-anchor="middle">OUTPUT: the answer streams back to the user</text></svg><figcaption>The same model does both jobs: process the supplied text, then continue it. Prefix reuse saves some of the first job; the new answer still has to be generated.</figcaption></figure>

SGLang accepts ordinary OpenAI-style serving requests, so you can put it behind an existing assistant. The original project also included a small Python language for describing connected model calls (for example "ask, then fork into three parallel judgements, then merge"). That frontend is optional; everything in this article is about the serving engine, which the paper calls the SGLang Runtime. [Sources: SGLang documentation](https://docs.sglang.io/), [original paper](https://arxiv.org/abs/2312.07104).

For now, focus on the first stage. Most of our second request's input is exactly the same as the first request's. To avoid processing it again, we need to know what can be saved.

## 2. Save the model's working state, not the answer

As the model reads the handbook, each attention layer computes arrays called **keys** and **values**, usually shortened to **K** and **V**. Later token positions use them to read information from earlier positions.

> [!DEFINITION] KV cache
> The stored keys and values from token positions the model has already processed. Keeping them lets later positions use that earlier work without rebuilding it. A cache is simply storage kept for possible reuse.

Within one answer, this is already useful. When generating the next word, the model reuses the saved state for the prompt and the answer so far. It does not rebuild that whole history at every step ([Part 2](llm-inference-2-kv-cache.md) measured this).

Now extend the idea **across requests**. After answering the annual-leave question, keep the handbook's KV state. When the carry-forward question arrives, reuse the handbook state and compute only the new question's state.

The cached object is not "the answer to the annual-leave question." It is the model's representation of the shared beginning. Different questions can use it to produce different answers.

> [!DEFINITION] Prefix
> A sequence starting at the very beginning. Here, the **shared prefix** is the identical run of token IDs at the start of two prompts. It is an exact match, not a judgement that two passages mean roughly the same thing.

### Why the later question does not change the earlier handbook

A standard causal language model reads in one direction: a token position can use itself and earlier positions, but cannot look at later positions.

In our prompt, the handbook comes **before** the question. While processing a handbook token, the model cannot look ahead at the question. Changing that later question therefore does not change the handbook's keys and values, provided the model, preceding tokens, and positional setup stay the same.

> [!PAPER] Zheng et al., SGLang · Section 3 · page 4
> [![The SGLang paper states: KV cache computation depends only on prefix tokens.](/img/sglang/paper-prefix.png)](/img/sglang/paper-prefix.png)
>
> **Context:** the authors are explaining when two model calls can share previously computed keys and values.
>
> **What it says:** the KV state at a position is determined by the input up to that position. In our example, the later question cannot change the earlier handbook state.
>
> **Why it matters:** keep the shared instructions and handbook before the changing question. The second request can then reuse the first request's handbook state and process only its new question. Putting different questions first would remove that shared beginning.
>
> [Read Section 3 of the paper](https://arxiv.org/pdf/2312.07104v2#page=4)

This arrangement can share the handbook:

```text
instructions → handbook → question that changes
```

This one usually cannot share the handbook across different questions:

```text
question that changes → instructions → handbook
```

In the second arrangement, the handbook can attend to a different earlier question, so its states may differ. Changing a timestamp, chat template, or tool definition near the beginning can also shorten the reusable prefix. Preserve the intended meaning when arranging a prompt; do not move information blindly just to improve a cache-hit number.

### A little maths: the same stored values, a different answer

Attention reads stored values by assigning them weights. A new **query** determines those weights. You can think of a query as what the current position wants to find in earlier positions.

Consider just two stored positions. To keep the arithmetic small, let their keys be 0 and 1, and their values be 2 and 6. Each vector has only one component in this example.

With query 0, both key matches score 0. Their attention weights are equal:

$$
o=\tfrac12(2)+\tfrac12(6)=4.
$$

Now choose query $$\ln 3$$, the number whose exponential is 3. The scores become 0 and $$\ln 3$$. Softmax exponentiates them, giving 1 and 3, then divides by their sum:

$$
\text{weights}=\left(\frac{e^0}{e^0+e^{\ln 3}},\frac{e^{\ln 3}}{e^0+e^{\ln 3}}\right)
=\left(\frac14,\frac34\right).
$$

The weighted result is now:

$$
o=\tfrac14(2)+\tfrac34(6)=5.
$$

**The keys and values stayed the same. The query changed, so the result changed.** Reusing KV does not force two questions to get the same answer. These two scalar calculations illustrate the attention operation; a language model repeats it with large vectors, many heads, and many layers.

The companion code runs this example, and it also checks two complete causal-attention layers: reusing an unchanged prefix agrees with recomputing the full sequence to floating-point precision (largest difference $$4.4\times10^{-16}$$), while deliberately reusing a *changed* prefix does not (largest difference 0.95). Section 4 repeats the check on a real model.

## 3. How big is the saved state?

Before deciding to keep the handbook's KV state around, ask what it costs to keep. Every token stores one key vector and one value vector in every layer, for every KV head.

**The idea.** Multiply the things a token stores.

$$
\text{bytes per token}=2\times L\times H_{\mathrm{KV}}\times d_h\times b
$$

**Symbol by symbol.** The 2 counts the two arrays, K and V. $$L$$ is the number of layers. $$H_{\mathrm{KV}}$$ is the number of KV heads per layer (with grouped-query attention this is smaller than the number of query heads, Part 2). $$d_h$$ is the length of one head's vector. $$b$$ is bytes per number: 2 for BF16 or FP16.

**Worked numbers for Llama 3.1 8B** (32 layers, 8 KV heads, head size 128, BF16, the same model as Parts 2 and 3):

$$
2\times32\times8\times128\times2=131{,}072\ \text{bytes}=128\ \text{KiB per token}.
$$

Our handbook plus instructions is about 6,000 tokens. Stored once, that is

$$
6{,}000\times128\ \text{KiB}=768{,}000\ \text{KiB}=750\ \text{MiB}.
$$

<figure class="fig"><svg viewBox="0 0 760 272" role="img" aria-label="KV bytes per token for Llama 3.1 8B are 2 times 32 times 8 times 128 times 2, which is 128 KiB. A 6,000-token handbook takes 750 MiB on Llama 3.1 8B, 328 MiB on Qwen2.5-7B and 70 MiB on Qwen2.5-0.5B."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Llama 3.1 8B: bytes of KV cache for one token</text><rect class="box-2" x="22.0" y="44.0" width="92.0" height="64.0" rx="8"/><text class="t-title" x="68.0" y="74.0" text-anchor="middle">2</text><text class="t-tick" x="68.0" y="96.0" text-anchor="middle">K and V</text><text class="t-title" x="128.0" y="82.0" text-anchor="middle">×</text><rect class="box-1" x="142.0" y="44.0" width="92.0" height="64.0" rx="8"/><text class="t-title" x="188.0" y="74.0" text-anchor="middle">32</text><text class="t-tick" x="188.0" y="96.0" text-anchor="middle">layers</text><text class="t-title" x="248.0" y="82.0" text-anchor="middle">×</text><rect class="box-1" x="262.0" y="44.0" width="92.0" height="64.0" rx="8"/><text class="t-title" x="308.0" y="74.0" text-anchor="middle">8</text><text class="t-tick" x="308.0" y="96.0" text-anchor="middle">KV heads</text><text class="t-title" x="368.0" y="82.0" text-anchor="middle">×</text><rect class="box-1" x="382.0" y="44.0" width="92.0" height="64.0" rx="8"/><text class="t-title" x="428.0" y="74.0" text-anchor="middle">128</text><text class="t-tick" x="428.0" y="96.0" text-anchor="middle">numbers/head</text><text class="t-title" x="488.0" y="82.0" text-anchor="middle">×</text><rect class="box-1" x="502.0" y="44.0" width="92.0" height="64.0" rx="8"/><text class="t-title" x="548.0" y="74.0" text-anchor="middle">2</text><text class="t-tick" x="548.0" y="96.0" text-anchor="middle">bytes (BF16)</text><text class="t-title" x="608.0" y="82.0" text-anchor="middle">=</text><rect class="box-3" x="622.0" y="44.0" width="118.0" height="64.0" rx="8"/><text class="t-title" x="681.0" y="74.0" text-anchor="middle">131,072 B</text><text class="t-tick" x="681.0" y="96.0" text-anchor="middle">= 128 KiB</text><text class="t-note" x="20.0" y="146.0" text-anchor="start">A 6,000-token handbook, stored once:</text><text class="t-note" x="150.0" y="179.0" text-anchor="end">Llama 3.1 8B</text><g class="mark"><title>Llama 3.1 8B: 128 KiB per token, 750 MiB per handbook</title><rect class="s1" x="162.0" y="165.0" width="430.0" height="20.0" rx="4"/></g><text class="t-tick" x="600.0" y="180.0" text-anchor="start">750 MiB  (128 KiB/token)</text><text class="t-note" x="150.0" y="213.0" text-anchor="end">Qwen2.5-7B</text><g class="mark"><title>Qwen2.5-7B: 56 KiB per token, 328 MiB per handbook</title><rect class="s1" x="162.0" y="199.0" width="188.1" height="20.0" rx="4"/></g><text class="t-tick" x="358.1" y="214.0" text-anchor="start">328 MiB  (56 KiB/token)</text><text class="t-note" x="150.0" y="247.0" text-anchor="end">Qwen2.5-0.5B</text><g class="mark"><title>Qwen2.5-0.5B: 12 KiB per token, 70 MiB per handbook</title><rect class="s1" x="162.0" y="233.0" width="40.3" height="20.0" rx="4"/></g><text class="t-tick" x="210.3" y="248.0" text-anchor="start">70 MiB  (12 KiB/token)</text></svg><figcaption>Top: the five factors that make up 128 KiB of KV cache per token for Llama 3.1 8B. Bottom: one 6,000-token handbook stored once, for three models. The small model we measure later needs 12 KiB per token.</figcaption></figure>

The same formula for the two Qwen models used in this series, from their published configurations ([Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json): 24 layers, 2 KV heads, head size 64; [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json): 28 layers, 4 KV heads, head size 128):

| Model | Calculation | Per token | 6,000-token handbook |
|---|---|---:|---:|
| Qwen2.5-0.5B | $$2\times24\times2\times64\times2$$ | 12 KiB | 70 MiB |
| Qwen2.5-7B | $$2\times28\times4\times128\times2$$ | 56 KiB | 328 MiB |
| Llama 3.1 8B | $$2\times32\times8\times128\times2$$ | 128 KiB | 750 MiB |

**What this does and does not include.** It is the KV payload only: no model weights, no temporary working memory, no page-table overhead. Quantized caches (for example FP8, which halves $$b$$), sliding-window layers, and multi-GPU sharding all change the accounting. The point to carry forward: a cached handbook is not free. It occupies memory that could otherwise hold more running requests, and that is why eviction (Section 7) exists.

## 4. What reuse saves: compute first, then time

### Counting the arithmetic

Suppose the cache already holds a prefix of $$P$$ tokens (instructions and handbook), and the new question adds $$S$$ suffix tokens. How much prefill work is left?

**The idea.** Each new token does two kinds of work. It is multiplied by every weight in the model, and in every layer its query is compared with the keys of every earlier position (and the result is used to mix their values).

**The equation.** For $$n$$ new tokens on top of $$c$$ cached ones,

$$
F(n,c)=\underbrace{2Nn}_{\text{weights}}+\underbrace{4Ld\left(nc+\tfrac{n(n+1)}{2}\right)}_{\text{attention}}.
$$

**Symbol by symbol.** $$N$$ is the number of non-embedding weights; each weight costs one multiply and one add per token, hence $$2N$$. $$L$$ is the number of layers and $$d$$ the query width (heads × head size). New token $$i$$ attends to $$c+i$$ positions; summing over $$i=1\ldots n$$ gives $$nc+n(n+1)/2$$ query-key pairs. Each pair costs $$2d$$ for the score and $$2d$$ for mixing the value, hence $$4Ld$$ per pair.

Without a cache, the request costs $$F(P+S,0)$$. With the prefix cached, it costs $$F(S,P)$$. The saving is the difference, and the speedup in arithmetic is their ratio:

$$
\text{FLOP ratio}=\frac{F(P+S,\,0)}{F(S,\,P)}.
$$

Notice what the cached version still pays: the $$nc$$ term. The new question **still attends to every cached handbook token**. Reuse removes the handbook's own computation, not the handbook from attention.

**Worked numbers for Qwen2.5-0.5B** ($$N=357{,}898{,}112$$ non-embedding weights, $$L=24$$, $$d=896$$), with a 2,048-token handbook and a 64-token question:

$$
F(2112,0)=2N(2112)+4(24)(896)\tfrac{2112\cdot2113}{2}\approx1{,}704\ \text{GFLOP}
$$

$$
F(64,2048)=2N(64)+4(24)(896)\left(64\cdot2048+\tfrac{64\cdot65}{2}\right)\approx57\ \text{GFLOP}
$$

That is a **30x** reduction in arithmetic. For scale, the same arithmetic for Llama 3.1 8B with an 8,192-token prefix is about 132 TFLOP, which would take 133 ms even at 100% of an H100's dense BF16 peak of 989 TFLOP/s (a bound, not a measurement).

The companion script prints these numbers for every prefix length, together with the KV sizes from Section 3 and the whole-request arithmetic used later in this section:

[![Terminal output of simulate.py sections 1 and 2: KV bytes per token for Qwen2.5-0.5B, Qwen2.5-7B and Llama 3.1 8B (12, 56 and 128 KiB); prefill FLOPs for P = 512 to 8,192 with S = 64, FLOP ratios 8.7x to 97.1x against measured time ratios 2.1x to 20.7x; Llama 3.1 8B 8,192-token prefix 132 TFLOP, 133 ms at H100 peak; whole-request times for answers of 1, 20, 200 and 1,000 tokens](/img/sglang/sim_math-run.png)](/img/sglang/sim_math-run.png)

### Measuring the time on a real model

Arithmetic is not time. So I ran the operation itself: Qwen2.5-0.5B in BF16 on an Apple M5 Pro GPU through PyTorch, the setup used in Parts 1 to 3. **Cold** means prefill handbook and question from nothing. **Warm** means the handbook's KV cache is already there and only the question is prefilled on top of it. Token IDs are random, because the content of the text does not change how long the arithmetic takes. Each time is the median of 9 runs after 3 warm-up runs.

```python
_, prefix_cache = prefill(prefix)                        # done once, then kept
cold_logits, _ = prefill(torch.cat([prefix, suffix], 1)) # everything from scratch
warm_logits, _ = prefill(suffix, copy.deepcopy(prefix_cache))  # only the new tokens
```

[![Terminal output of measure_prefix.py on Qwen2.5-0.5B, BF16, MPS: for a 64-token suffix, cold and warm prefill times are 30.2 and 14.7 ms at P=512, 51.2 and 12.9 at 1024, 100.4 and 13.2 at 2048, 211.2 and 16.9 at 4096, 462.3 and 22.3 at 8192, speedups 2.1x to 20.7x, same next token every time with logit difference 0.0000. Suffix sweep at P=2048: speedups 6.3x, 7.5x, 5.9x and 2.5x for S = 16, 64, 256 and 1024. Decode step at context 2,112: 12.71 ms](/img/sglang/measure_prefix-run.png)](/img/sglang/measure_prefix-run.png)

<figure class="fig"><svg viewBox="0 0 760 330" role="img" aria-label="Measured prefill time, cold versus warm, for shared prefixes of 512 to 8,192 tokens. The speedup grows from 2.1x to 20.7x."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><rect class="s2" x="170.0" y="12.0" width="12.0" height="12.0" rx="3"/><text class="t-tick" x="188.0" y="23.0" text-anchor="start">cold: prefill handbook + question</text><rect class="s1" x="430.0" y="12.0" width="12.0" height="12.0" rx="3"/><text class="t-tick" x="448.0" y="23.0" text-anchor="start">warm: handbook cached, prefill the question</text><text class="t-note" x="150.0" y="66.0" text-anchor="end">P = 512</text><g class="mark"><title>P=512, S=64: cold 30.25 ms</title><rect class="s2" x="162.0" y="44.0" width="28.8" height="17.0" rx="4"/></g><text class="t-tick" x="196.8" y="57.0" text-anchor="start">30 ms</text><g class="mark"><title>P=512, S=64: warm 14.65 ms</title><rect class="s1" x="162.0" y="64.0" width="13.9" height="17.0" rx="4"/></g><text class="t-tick" x="181.9" y="77.0" text-anchor="start">15 ms</text><text class="t-val" x="745.0" y="68.0" text-anchor="end">2.1x</text><text class="t-note" x="150.0" y="118.0" text-anchor="end">P = 1,024</text><g class="mark"><title>P=1024, S=64: cold 51.2 ms</title><rect class="s2" x="162.0" y="96.0" width="48.7" height="17.0" rx="4"/></g><text class="t-tick" x="216.7" y="109.0" text-anchor="start">51 ms</text><g class="mark"><title>P=1024, S=64: warm 12.94 ms</title><rect class="s1" x="162.0" y="116.0" width="12.3" height="17.0" rx="4"/></g><text class="t-tick" x="180.3" y="129.0" text-anchor="start">13 ms</text><text class="t-val" x="745.0" y="120.0" text-anchor="end">4.0x</text><text class="t-note" x="150.0" y="170.0" text-anchor="end">P = 2,048</text><g class="mark"><title>P=2048, S=64: cold 100.4 ms</title><rect class="s2" x="162.0" y="148.0" width="95.6" height="17.0" rx="4"/></g><text class="t-tick" x="263.6" y="161.0" text-anchor="start">100 ms</text><g class="mark"><title>P=2048, S=64: warm 13.15 ms</title><rect class="s1" x="162.0" y="168.0" width="12.5" height="17.0" rx="4"/></g><text class="t-tick" x="180.5" y="181.0" text-anchor="start">13 ms</text><text class="t-val" x="745.0" y="172.0" text-anchor="end">7.6x</text><text class="t-note" x="150.0" y="222.0" text-anchor="end">P = 4,096</text><g class="mark"><title>P=4096, S=64: cold 211.17 ms</title><rect class="s2" x="162.0" y="200.0" width="201.0" height="17.0" rx="4"/></g><text class="t-tick" x="369.0" y="213.0" text-anchor="start">211 ms</text><g class="mark"><title>P=4096, S=64: warm 16.88 ms</title><rect class="s1" x="162.0" y="220.0" width="16.1" height="17.0" rx="4"/></g><text class="t-tick" x="184.1" y="233.0" text-anchor="start">17 ms</text><text class="t-val" x="745.0" y="224.0" text-anchor="end">12.5x</text><text class="t-note" x="150.0" y="274.0" text-anchor="end">P = 8,192</text><g class="mark"><title>P=8192, S=64: cold 462.27 ms</title><rect class="s2" x="162.0" y="252.0" width="440.0" height="17.0" rx="4"/></g><text class="t-tick" x="608.0" y="265.0" text-anchor="start">462 ms</text><g class="mark"><title>P=8192, S=64: warm 22.34 ms</title><rect class="s1" x="162.0" y="272.0" width="21.3" height="17.0" rx="4"/></g><text class="t-tick" x="189.3" y="285.0" text-anchor="start">22 ms</text><text class="t-val" x="745.0" y="276.0" text-anchor="end">20.7x</text><line class="axis" x1="162.0" y1="40.0" x2="162.0" y2="294.0"/><text class="t-tick" x="380.0" y="318.0" text-anchor="middle">Qwen2.5-0.5B, BF16, Apple M5 Pro, PyTorch. S = 64 new tokens. Median of 9 runs.</text></svg><figcaption>Measured on a real model: prefill time with and without a cached prefix, for prefixes of 512 to 8,192 tokens and a 64-token question. The longer the shared beginning, the bigger the saving.</figcaption></figure>

Three things to read from this run:

1. **The answer did not change.** In every case the cold and warm runs chose the same next token, and the largest difference between their output scores was 0.0000 to four decimal places. Reuse is the same arithmetic done once, not an approximation.
2. **The saving grows with the shared prefix**, from 2.1x at 512 tokens to 20.7x at 8,192 tokens, because cold prefill grows with $$P$$ while the warm pass stays small.
3. **The time saving is smaller than the arithmetic saving.** At $$P=2048$$ the arithmetic falls 30x, but the time falls 7.6x. A 64-token pass on this GPU costs about 13 ms no matter how few FLOPs it has, about the same as one decode step (12.71 ms in the same run). Part 1 explained why: small passes are limited by reading the weights from memory and by fixed overheads, not by arithmetic. That is also why $$S=16$$ was no faster than $$S=64$$ here.

To use these timings in simulations later, I fitted a simple cost model to all nine cold and warm measurements:

$$
t(n,c)\approx 11.0+0.0391\,n+3.80\times10^{-6}\,n\left(c+\tfrac n2\right)\ \text{ms},
$$

with $$n$$ new tokens on top of $$c$$ cached ones. The 11.0 ms constant is that fixed floor; the last term is attention to everything already there. It fits the cold runs within 10% and every point within 30% (the worst is the warm run at $$P=8192$$, 22.3 ms measured against 15.5 ms predicted). It is good enough to compare scheduling orders on this machine, and not a model of any other GPU.

> [!WARNING] What this measurement can and cannot show
> It shows that reusing a prefix's KV cache gives identical output and how much prefill time it saves for one request on one small model and one Apple GPU. It cannot show SGLang or vLLM throughput, batching effects, memory pressure, or data-centre GPU timings. On an H100 with an 8B model the absolute numbers are very different; the shape (savings grow with $$P$$, small passes have a floor) is the part that carries over.

### What the user actually waits for

Prefill is only the beginning. The whole request costs the time to the first token (TTFT) plus one decode step for every later answer token (time per output token, TPOT):

$$
T=\text{TTFT}+(n_{\text{out}}-1)\times\text{TPOT}.
$$

Prefix reuse shrinks TTFT and leaves TPOT alone, so the speedup of the whole request is

$$
\text{speedup}=\frac{t_{\text{cold}}+(n_{\text{out}}-1)\,\tau}{t_{\text{warm}}+(n_{\text{out}}-1)\,\tau},
$$

where $$\tau$$ is the decode step. With the measured values for a 2,048-token handbook ($$t_{\text{cold}}=100.4$$ ms, $$t_{\text{warm}}=13.2$$ ms, $$\tau=12.71$$ ms):

| Answer length | Cold request | Warm request | Speedup |
|---:|---:|---:|---:|
| 1 token | 100.4 ms | 13.2 ms | 7.63x |
| 20 tokens | 341.9 ms | 254.6 ms | 1.34x |
| 200 tokens | 2,629.7 ms | 2,542.4 ms | 1.03x |
| 1,000 tokens | 12,797.7 ms | 12,710.4 ms | 1.01x |

Same cache hit, very different value. For a "yes / no / which section" style answer it is a 7.6x faster response. For a long essay it barely matters for the person waiting, although the GPU time saved can still be spent on other people's requests. vLLM's own documentation says the same thing about its prefix cache:

> [!PAPER] vLLM documentation · Automatic Prefix Caching · Limits
> [![vLLM docs: APC in general does not reduce the performance of vLLM. With that being said, APC only reduces the time of processing the queries (the prefilling phase) and does not reduce the time of generating new tokens (the decoding phase). So APC does not bring performance gain when vLLM spends most of the time generating answers to the queries, or new queries do not share the same prefix with any of existing queries.](/img/sglang/vllm-apc-limits.png)](/img/sglang/vllm-apc-limits.png)
>
> **Context:** the "Limits" section of vLLM's feature page for automatic prefix caching (APC).
>
> **What it says:** prefix caching only shortens prefill. It does not help when decoding dominates or when nothing is shared.
>
> **Why it matters:** this is engine-independent. It applies to SGLang's radix cache in exactly the same way, and it is what the table above computes. Measure TTFT and total time separately.
>
> [Read the vLLM page](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)

## 5. How SGLang finds the saved beginning: RadixAttention

Our assistant may answer thousands of questions. Some requests share the whole handbook; others share only the instructions. Some are follow-up turns of the same chat and share the whole previous conversation. Scanning every old prompt to find the longest match would be slow. We need an index.

> [!DEFINITION] Radix tree
> A **prefix tree** (trie) stores sequences so that sequences with a common beginning share a path from the root. A **radix tree** is the compact version: a stretch with no branch is stored as one edge labelled with the whole run of tokens, instead of one node per token.

Let us build one. For this small example, six token IDs stand for our shared instructions and handbook:

```text
shared beginning: 10 11 12 13 14 15
first question:                     20 21
second question:                    30 31
```

These are invented IDs for seeing how the algorithm works, not a real tokenizer's encoding of the handbook.

**First request.** The tree is empty. Compute all eight positions and record the sequence as one edge, pointing at its saved KV.

**Second request.** Follow the stored sequence from the root. The first six IDs match. The next ID is 30 instead of 20, so split the edge at that point. Reuse the six matching positions and compute the two new ones.

<figure class="fig"><svg viewBox="0 0 760 430" role="img" aria-label="First request stores eight tokens. The second shares token IDs 10 through 15 and splits into question branches 20 21 and 30 31."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="25.0" y="25.0" text-anchor="start">1. After the first request</text><rect class="box-1" x="90.0" y="48.0" width="580.0" height="50.0" rx="8"/><text class="t-note" x="380.0" y="79.0" text-anchor="middle">10 11 12 13 14 15 20 21</text><text class="t-tick" x="380.0" y="123.0" text-anchor="middle">One stored path: eight computed token positions</text><text class="t-note" x="25.0" y="177.0" text-anchor="start">2. The second request matches six IDs, then branches</text><rect class="box-1" x="210.0" y="198.0" width="340.0" height="52.0" rx="8"/><text class="t-note" x="380.0" y="230.0" text-anchor="middle">10 11 12 13 14 15</text><line class="edge" x1="380.0" y1="250.0" x2="190.0" y2="293.0"/><line class="edge" x1="380.0" y1="250.0" x2="570.0" y2="293.0"/><rect class="box-1" x="90.0" y="293.0" width="200.0" height="48.0" rx="8"/><text class="t-note" x="190.0" y="323.0" text-anchor="middle">20 21</text><rect class="box-2" x="470.0" y="293.0" width="200.0" height="48.0" rx="8"/><text class="t-note" x="570.0" y="323.0" text-anchor="middle">30 31</text><text class="t-tick" x="190.0" y="367.0" text-anchor="middle">First question: already stored</text><text class="t-tick" x="570.0" y="367.0" text-anchor="middle">Second question: compute now</text><text class="t-tick" x="380.0" y="405.0" text-anchor="middle">The shared path refers to one reusable set of KV states.</text></svg><figcaption>The branch appears exactly where the token sequences differ. Both questions refer to the same saved beginning, while their different continuations have separate state.</figcaption></figure>

The tree says where to find the states. It is not the large array of states itself: the index lives on the CPU, while the KV tensors live in GPU memory. Splitting an edge changes the index only; the shared prefix is not recalculated or copied.

> [!PAPER] Zheng et al., SGLang · Section 3, "RadixAttention" · page 4
> [![Paper paragraph: a radix tree manages a mapping between sequences of tokens and their KV cache tensors, stored in a non-contiguous paged layout with one token per page; a simple LRU eviction policy evicts the least recently used leaf first; each node maintains a reference counter, and a node is evictable if its reference counter is zero; cached tokens and running requests share one memory pool](/img/sglang/paper-radix-lru.png)](/img/sglang/paper-radix-lru.png)
>
> **Context:** the core of Section 3, right after the authors define the radix tree.
>
> **What it says:** keys are token sequences, values are KV tensors kept in a paged layout with **one token per page**. Eviction removes the least recently used **leaf** first. A **reference counter** protects nodes that running requests are using. There is no separate fixed-size cache: cached tokens and running requests share one memory pool, and when enough requests are waiting the cache shrinks to zero in favour of a bigger batch.
>
> **Why it matters:** this paragraph is the whole design in miniature. One-token pages explain why SGLang can reuse a prefix to the exact token (Section 8 compares this with vLLM's blocks). Leaf-first eviction explains why a shared handbook survives while old questions are dropped (Section 7). The shared pool explains why caching never blocks running work.
>
> [Read Section 3](https://arxiv.org/pdf/2312.07104v2#page=4)

### Run the three-request example

Here is the small radix-tree implementation from the companion code in use:

```python
from experiments import Radix

prefix = [10, 11, 12, 13, 14, 15]
prompts = [
    prefix + [20, 21],
    prefix + [30, 31],
    prefix + [20, 40],
]
cache = Radix()

for number, prompt in enumerate(prompts, 1):
    reused = cache.match(prompt)
    print(f"request {number}: reuse {reused}, compute {len(prompt) - reused}")
    cache.insert(prompt)
```

[![Terminal output of walkthrough.py: radix cache, request 1 reuse 0 compute 8, request 2 reuse 6 compute 2, request 3 reuse 7 compute 1; blocks of 4, request 1 reuse 0 compute 8, request 2 reuse 4 compute 4, request 3 reuse 4 compute 4; then the attention example, query 0 gives weights 0.50 and 0.50 and output 4.00, query ln(3) gives weights 0.25 and 0.75 and output 5.00](/img/sglang/walkthrough-run.png)](/img/sglang/walkthrough-run.png)

The third request is worth looking at. It shares the six-token beginning **and token 20** with the first request. Matching continues into that branch, stopping only when 40 differs from 21. The reusable part is discovered from the tokens; it does not have to end at a boundary we called "the handbook."

This example runs the lookup and insertion logic on the CPU. It counts token positions that need computing; it does not run a model for them. The full [`Radix` implementation](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/experiments.py) includes the edge splitting just illustrated, and it is checked against a brute-force longest-common-prefix search on 600 random requests.

### The paper's own picture

The paper draws the same process for a richer mix of traffic: two chat sessions, a batch of few-shot questions, and several samples of one question.

> [!PAPER] Zheng et al., SGLang · Figure 3 · page 5
> [![Figure 3 of the SGLang paper: nine snapshots of a radix tree. (1) empty; (2) one chat turn stored as a single edge; (3) a second turn appended; (4) a second chat session splits the system prompt edge; (5) memory runs out and node c is evicted; (6) a few-shot query adds a new branch at the root; (7) more few-shot queries split node e to share the examples; (8) the second chat session is evicted; (9) more samples for one question evict nodes i, k and l](/img/sglang/paper-fig3.png)](/img/sglang/paper-fig3.png)
>
> **Context:** Figure 3, with a long caption that walks through all nine steps. Green nodes are new, blue nodes are cache hits at that step, red nodes are evicted.
>
> **What it says:** the tree grows by appending and splitting, exactly as in our small example. When memory is short, whole branches that were not used recently disappear (steps 5, 8 and 9), while the shared system prompt at the top stays.
>
> **Why it matters:** look at step (7). Several few-shot questions share a long list of examples, and the tree splits node "e" so the examples are stored once and every question hangs below them. That is our handbook with several questions, drawn by the authors.
>
> [Read the figure on page 5](https://arxiv.org/pdf/2312.07104v2#page=5)

The SGLang team's launch post summarised which kinds of programs produce this sharing. Our handbook assistant is pattern (a); a support chat is pattern (c).

> [!PAPER] LMSYS blog, "Fast and Expressive LLM Inference with RadixAttention and SGLang" (January 2024) · Figure 3
> [![Four KV cache sharing patterns: (a) few-shot learning, where three prompts share the same few-shot examples; (b) self-consistency, where one question is answered three times; (c) multi-turn chat, where each turn shares the whole chat history; (d) tree-of-thought, where branches share a growing search history. Blue boxes are shareable, green and yellow are not](/img/sglang/blog-sharing-patterns.png)](/img/sglang/blog-sharing-patterns.png)
>
> **Context:** the section "Backend: Automatic KV Cache Reuse with RadixAttention".
>
> **What it says:** blue parts can be shared, green parts are each request's own input, yellow parts are each request's own output. The shareable part can be a fixed list of examples, one question, a growing chat history, or a branching search history.
>
> **Why it matters:** in (c) and (d) the shared part is a previous **output**. That is why both SGLang and vLLM also cache the KV of generated tokens: the next chat turn begins with the previous answer.
>
> [Read the post](https://www.lmsys.org/blog/2024-01-17-sglang/)

## 6. How much memory the tree shares

Reuse saves computation once. Sharing saves memory for as long as requests run together. If 32 people are asking about the handbook at the same moment, should there be 32 copies of its KV cache in GPU memory?

**The idea.** In a radix tree, every token position is stored once, on exactly one edge, however many requests pass through that edge.

**The equation.** Without sharing, memory is the sum of the request lengths. With a tree, it is the sum of the edge lengths:

$$
M_{\text{no sharing}}=\sum_{r}|r|,\qquad M_{\text{tree}}=\sum_{e\in\text{edges}}|e|.
$$

**Worked numbers.** Take 32 active requests, each with 512 instruction tokens, one of 4 handbook sections of 1,536 tokens, and its own 64-token question. Each request is $$512+1536+64=2112$$ tokens.

$$
M_{\text{no sharing}}=32\times2112=67{,}584\ \text{token slots}
$$

$$
M_{\text{tree}}=\underbrace{512}_{\text{instructions}}+\underbrace{4\times1536}_{\text{sections}}+\underbrace{32\times64}_{\text{questions}}=8{,}704\ \text{token slots}
$$

That is **7.8x** less KV memory. Multiply by bytes per token from Section 3: on Llama 3.1 8B it is 8.25 GiB without sharing against 1.06 GiB with the tree; on Qwen2.5-0.5B, 0.77 GiB against 0.10 GiB.

<figure class="fig"><svg viewBox="0 0 760 196" role="img" aria-label="Thirty-two requests need 67,584 KV slots without sharing and 8,704 in a radix tree, 7.8 times fewer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="140.0" y="42.0" text-anchor="end">no sharing</text><g class="mark"><title>no sharing: 67,584 token slots</title><rect class="s2" x="152.0" y="26.0" width="470.0" height="24.0" rx="4"/></g><text class="t-val" x="630.0" y="43.0" text-anchor="start">67,584 slots</text><text class="t-tick" x="152.0" y="68.0" text-anchor="start">8.25 GiB on Llama 3.1 8B</text><text class="t-note" x="140.0" y="106.0" text-anchor="end">radix tree</text><g class="mark"><title>radix tree: 8,704 token slots</title><rect class="s1" x="152.0" y="90.0" width="60.5" height="24.0" rx="4"/></g><text class="t-val" x="220.5" y="107.0" text-anchor="start">8,704 slots</text><text class="t-tick" x="152.0" y="132.0" text-anchor="start">1.06 GiB on Llama 3.1 8B</text><text class="t-tick" x="380.0" y="162.0" text-anchor="middle">32 active requests: 512 instruction + 1,536 handbook (one of 4) + 64 question tokens each.</text><text class="t-tick" x="380.0" y="182.0" text-anchor="middle">Tree = 512 + 4 × 1,536 + 32 × 64 = 8,704 slots, 7.8x fewer.</text></svg><figcaption>Thirty-two concurrent handbook requests. Without sharing every request holds its own copy of the instructions and section; the tree stores each once. Simulated with the companion RadixLRU class, which agrees with the formula.</figcaption></figure>

More free memory means more requests fit in a batch, and Part 1 showed that bigger decode batches mean more tokens per second. That is the second, less obvious benefit of prefix sharing.

**Be careful what this compares against.** "No sharing" means an engine that copies the prefix into every request. vLLM shares too: identical full blocks are stored once and reference-counted (Part 3). So against vLLM the difference is not 7.8x; it is at most the partial-block tokens that blocks cannot share (Section 8).

## 7. When the cache fills up: LRU eviction

GPU memory is finite. Sooner or later, keeping one more handbook means dropping something else. The paper's rule is simple to state: **evict the least recently used leaf first, and never evict anything a running request is using.**

**The equation.** Give every node $$v$$ a last-access time $$\tau(v)$$ and a reference count $$\text{ref}(v)$$ (how many running requests use it). While memory used exceeds capacity $$C$$, remove

$$
v^\star=\arg\min_{v\,\in\,\text{leaves},\ \text{ref}(v)=0}\ \tau(v).
$$

**Why leaves only, and why it is safe.** Every time a request uses a node, it walks through all the node's ancestors and touches them too. So a parent is always at least as recent as any of its children:

$$
\tau(\text{parent})\ \ge\ \max_{\text{children}}\tau(\text{child}).
$$

Plain LRU would therefore never want a parent gone before its children anyway. Restricting eviction to leaves keeps the tree well formed, and the shared handbook, which sits above many questions, can only be evicted after every question below it has gone.

A worked trace, with room for only 10 tokens:

<figure class="fig"><svg viewBox="0 0 760 244" role="img" aria-label="Leaf-first LRU eviction with room for 10 tokens. The third question evicts the oldest question branch; the handbook node stays."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="118.0" y="24.0" text-anchor="middle">q1, q2 stored</text><text class="t-tick" x="118.0" y="44.0" text-anchor="middle">10 of 10 slots used</text><rect class="box" x="48.0" y="62.0" width="140.0" height="36.0" rx="8"/><text class="t-note" x="118.0" y="85.0" text-anchor="middle">1 2 3 4 5 6</text><line class="edge" x1="118.0" y1="98.0" x2="79.0" y2="140.0"/><rect class="box-1" x="46.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="79.0" y="162.0" text-anchor="middle">20 21</text><text class="t-tick" x="79.0" y="194.0" text-anchor="middle">used: step 1</text><line class="edge" x1="118.0" y1="98.0" x2="157.0" y2="140.0"/><rect class="box-1" x="124.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="157.0" y="162.0" text-anchor="middle">30 31</text><text class="t-tick" x="157.0" y="194.0" text-anchor="middle">used: step 2</text><text class="t-note" x="372.0" y="24.0" text-anchor="middle">q3 arrives</text><text class="t-tick" x="372.0" y="44.0" text-anchor="middle">evict the LRU leaf</text><rect class="box" x="302.0" y="62.0" width="140.0" height="36.0" rx="8"/><text class="t-note" x="372.0" y="85.0" text-anchor="middle">1 2 3 4 5 6</text><line class="edge" x1="372.0" y1="98.0" x2="294.0" y2="140.0"/><rect class="waste" x="261.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="294.0" y="162.0" text-anchor="middle">20 21</text><text class="t-tick" x="294.0" y="194.0" text-anchor="middle">evicted</text><line class="edge" x1="372.0" y1="98.0" x2="372.0" y2="140.0"/><rect class="box-1" x="339.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="372.0" y="162.0" text-anchor="middle">30 31</text><text class="t-tick" x="372.0" y="194.0" text-anchor="middle">kept</text><line class="edge" x1="372.0" y1="98.0" x2="450.0" y2="140.0"/><rect class="box-2" x="417.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="450.0" y="162.0" text-anchor="middle">40 41</text><text class="t-tick" x="450.0" y="194.0" text-anchor="middle">new</text><text class="t-note" x="626.0" y="24.0" text-anchor="middle">q2 hits, then q1 returns</text><text class="t-tick" x="626.0" y="44.0" text-anchor="middle">evict the new LRU leaf</text><rect class="box" x="556.0" y="62.0" width="140.0" height="36.0" rx="8"/><text class="t-note" x="626.0" y="85.0" text-anchor="middle">1 2 3 4 5 6</text><line class="edge" x1="626.0" y1="98.0" x2="548.0" y2="140.0"/><rect class="box-1" x="515.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="548.0" y="162.0" text-anchor="middle">30 31</text><text class="t-tick" x="548.0" y="194.0" text-anchor="middle">hit: 8 of 8</text><line class="edge" x1="626.0" y1="98.0" x2="626.0" y2="140.0"/><rect class="waste" x="593.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="626.0" y="162.0" text-anchor="middle">40 41</text><text class="t-tick" x="626.0" y="194.0" text-anchor="middle">evicted</text><line class="edge" x1="626.0" y1="98.0" x2="704.0" y2="140.0"/><rect class="box-2" x="671.0" y="140.0" width="66.0" height="34.0" rx="8"/><text class="t-note" x="704.0" y="162.0" text-anchor="middle">20 21</text><text class="t-tick" x="704.0" y="194.0" text-anchor="middle">recomputed</text><line class="drop" x1="250.0" y1="14.0" x2="250.0" y2="200.0"/><line class="drop" x1="504.0" y1="14.0" x2="504.0" y2="200.0"/><text class="t-tick" x="380.0" y="228.0" text-anchor="middle">The shared handbook node is never a leaf while a question hangs below it, so it survives every eviction.</text></svg><figcaption>Leaf-first LRU with room for 10 tokens. When q3 arrives the oldest question branch (20 21) is evicted. Later q2 hits completely, and when q1 returns its branch must be recomputed; the least recently used leaf is now 40 41, so that goes. The handbook node is never evicted.</figcaption></figure>

[![Terminal output of simulate.py sections 3 and 4: memory sharing 67,584 token slots without sharing against 8,704 in the tree, ratio 7.8x, 8.25 GiB against 1.06 GiB on Llama 3.1 8B; then the LRU trace with capacity 10: q1 computes 8, q2 reuses 6, q3 reuses 6 and evicts 20 21, q2 again reuses all 8, q1 again reuses 6 and computes 2](/img/sglang/sim_memory-run.png)](/img/sglang/sim_memory-run.png)

Current SGLang keeps exactly this rule as its default and adds alternatives. Its documentation states the leaf-first walk directly:

> [!PAPER] SGLang documentation · Radix Cache Eviction Policies
> [![SGLang docs: How eviction picks a victim. Eviction only ever considers evictable leaves: nodes whose KV is present, unlocked (no in-flight request holds them), and not shadowed by a child that still holds KV. The root is never evictable. The policy scores each candidate and the lowest score is evicted first. Once a leaf is evicted, its parent may become a leaf and re-enter the candidate set, so eviction walks a branch from its tip toward the root.](/img/sglang/sglang-eviction-doc.png)](/img/sglang/sglang-eviction-doc.png)
>
> **Context:** the page for the `--radix-eviction-policy` flag.
>
> **What it says:** only unlocked leaves are candidates, the lowest score goes first, and eviction walks up a branch from its tip.
>
> **Why it matters:** the policy only decides the *score*. The default score is recency (`lru`); the same page lists `lfu`, `slru`, `priority` and `tlru` as alternatives, for workloads where a stable hot set or tail first-token latency matters more than recency.
>
> [Read the SGLang page](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy)

One detail matters for the comparison later: in the source we checked, SGLang's [`RadixCache.evict`](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/mem_cache/radix_cache.py#L533) frees a chosen leaf **whole**. If the least recently used leaf is a 1,500-token handbook section, all 1,500 tokens go at once. Keep that in mind for Section 8.

So a second handbook request can reuse the first request's work **only if that work is still in the cache**. A matching prompt alone does not guarantee a hit.

## 8. Hit rate and the expected saving

How do you know whether all this is working? The paper defines the number to watch:

$$
\text{cache hit rate}\ h=\frac{\text{cached prompt tokens}}{\text{total prompt tokens}}.
$$

It is a token-weighted rate: a request that reuses 2,000 of its 2,064 tokens counts far more than one that reuses 10 of 50. Two consequences follow directly.

**Expected computation.** The prompt tokens that still need prefill are

$$
\text{computed tokens}=(1-h)\times\sum_r|r_{\text{prompt}}|.
$$

**Expected first-token time.** Suppose each handbook request either finds the handbook cached (probability $$p$$) or does not. With the measured times from Section 4,

$$
\mathbb{E}[t]=p\,t_{\text{warm}}+(1-p)\,t_{\text{cold}}.
$$

At $$p=0.5$$ that is $$0.5(13.2)+0.5(100.4)=56.8$$ ms; at $$p=0.9$$ it is 21.9 ms. The expected time falls in a straight line as the hit probability rises, so each extra percentage point of hits is worth the same amount: here about 0.87 ms per point.

<figure class="fig"><svg viewBox="0 0 760 276" role="img" aria-label="Expected prefill time falls in a straight line from 100 ms with no hit to 13 ms with a certain hit; at a 50 percent hit rate it is 57 ms."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="90.0" y1="220.0" x2="700.0" y2="220.0"/><text class="t-tick" x="80.0" y="224.0" text-anchor="end">0</text><line class="grid" x1="90.0" y1="176.8" x2="700.0" y2="176.8"/><text class="t-tick" x="80.0" y="180.8" text-anchor="end">25</text><line class="grid" x1="90.0" y1="133.6" x2="700.0" y2="133.6"/><text class="t-tick" x="80.0" y="137.6" text-anchor="end">50</text><line class="grid" x1="90.0" y1="90.5" x2="700.0" y2="90.5"/><text class="t-tick" x="80.0" y="94.5" text-anchor="end">75</text><line class="grid" x1="90.0" y1="47.3" x2="700.0" y2="47.3"/><text class="t-tick" x="80.0" y="51.3" text-anchor="end">100</text><text class="t-tick" x="90.0" y="240.0" text-anchor="middle">0%</text><text class="t-tick" x="242.5" y="240.0" text-anchor="middle">25%</text><text class="t-tick" x="395.0" y="240.0" text-anchor="middle">50%</text><text class="t-tick" x="547.5" y="240.0" text-anchor="middle">75%</text><text class="t-tick" x="700.0" y="240.0" text-anchor="middle">100%</text><text class="t-tick" x="395.0" y="262.0" text-anchor="middle">probability h that the 2,048-token handbook is still cached</text><text class="t-tick" x="16.0" y="18.0" text-anchor="start">expected prefill, ms</text><polyline class="l1" points="90.0,46.6 700.0,197.3"/><g class="mark"><title>h = 0%: 100.4 ms</title><circle class="s1 ring" cx="90.0" cy="46.6" r="6"/></g><text class="t-val" x="100.0" y="36.6" text-anchor="start">100 ms</text><g class="mark"><title>h = 50%: 56.8 ms</title><circle class="s1 ring" cx="395.0" cy="121.9" r="6"/></g><text class="t-val" x="405.0" y="111.9" text-anchor="start">57 ms</text><g class="mark"><title>h = 90%: 21.9 ms</title><circle class="s1 ring" cx="639.0" cy="182.2" r="6"/></g><text class="t-val" x="635.0" y="168.2" text-anchor="end">22 ms</text><g class="mark"><title>h = 100%: 13.2 ms</title><circle class="s1 ring" cx="700.0" cy="197.3" r="6"/></g><text class="t-val" x="706.0" y="183.3" text-anchor="end">13 ms</text><line class="axis" x1="90.0" y1="220.0" x2="700.0" y2="220.0"/></svg><figcaption>Expected prefill time for one handbook request against the probability that its 2,048-token handbook is still cached, using the measured cold (100.4 ms) and warm (13.2 ms) times on Qwen2.5-0.5B.</figcaption></figure>

What hit rates happen in practice? The paper reports one production data point:

> [!PAPER] Zheng et al., SGLang · Section 6.2, "Production deployment" · page 8
> [![Paper paragraph: SGLang has been deployed in Chatbot Arena; after one month the authors observed a 52.4% RadixAttention cache hit rate for LLaVA-Next-34B and 74.1% for Vicuna-33B, from common system messages, reused example images and multi-turn chat histories, which reduces first-token latency by an average of 1.7x for Vicuna-33B](/img/sglang/paper-arena.png)](/img/sglang/paper-arena.png)
>
> **Context:** a live service, Chatbot Arena, with one SGLang worker per model because traffic per model was low.
>
> **What it says:** after one month, hit rates of 52.4% (LLaVA-Next-34B) and 74.1% (Vicuna-33B), and an average 1.7x lower first-token latency for Vicuna-33B.
>
> **Why it matters:** even real public chat traffic, with no deliberate prompt design, has a lot of repeated beginnings: system messages and chat histories. The numbers are specific to that service and those 2024 models; they tell you that measuring your own hit rate is worth it, not what it will be.
>
> [Read Section 6.2](https://arxiv.org/pdf/2312.07104v2#page=8)

## 9. How vLLM handles the same two requests

vLLM also reuses prefixes, and in its current engine this is **on by default**: [`enable_prefix_caching: bool = True`](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/cache.py#L141) in the version we checked. It finds saved work differently. It cuts every sequence into fixed-size **blocks** (16 tokens by default, the same blocks PagedAttention stores, Part 3) and gives each full block a **hash**: a compact fingerprint used to look it up.

> [!PAPER] vLLM documentation · Design · Automatic Prefix Caching
> [![vLLM design doc diagram: the sentence A gentle breeze stirred the leaves as children laughed in the distance split into three blocks; block 1 is identified by its own tokens; block 2 by the prefix plus its tokens; block 3 by the longer prefix plus its tokens](/img/sglang/vllm-hash-blocks.png)](/img/sglang/vllm-hash-blocks.png)
>
> **Context:** the opening of vLLM's design page for prefix caching, which describes vLLM's choice as "a hash-based approach".
>
> **What it says:** the third block is identified by its own tokens *and* all the tokens before it. In practice each block's hash combines the parent block's hash, the block's tokens, and extra values such as LoRA IDs, image hashes and a cache salt. The same page notes "We only cache full blocks" and that since v0.11 the default hash is SHA-256.
>
> **Why it matters:** chaining in the parent hash makes a block's key stand for the whole prefix up to that point, which is what a radix path does too. Two identical blocks that follow different beginnings get different keys and cannot be confused.
>
> [Read the vLLM design page](https://docs.vllm.ai/en/latest/design/prefix_caching/)

**The equation.** With block size $$B$$, block $$i$$'s key is

$$
k_i=H\big(k_{i-1},\ \text{tokens}_{(i-1)B+1\,\ldots\,iB},\ \text{extra}\big),\qquad k_0=\text{empty}.
$$

The reusable length for a new prompt is the longest run of blocks whose keys are all in the cache:

$$
\text{reused}=B\times\max\{\,j: k_1,\ldots,k_j\ \text{all cached}\,\}.
$$

With an unlimited cache, this equals the radix tree's exact match $$\ell$$ rounded down to whole blocks, $$B\lfloor \ell/B\rfloor$$. The companion code checks that identity on 300 random requests for each of $$B=1,4,16$$.

Use four-token blocks on our second request to see the rounding:

<figure class="fig"><svg viewBox="0 0 760 240" role="img" aria-label="With page size 1 the second request reuses six tokens. With four-token blocks it reuses four: tokens 14 and 15 match, but their block also holds 30 and 31, so the block hash differs."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="20.0" y="54.0" text-anchor="start">radix, page 1</text><rect class="box-1" x="170.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="196.0" y="54.0" text-anchor="middle">10</text><rect class="box-1" x="228.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="254.0" y="54.0" text-anchor="middle">11</text><rect class="box-1" x="286.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="312.0" y="54.0" text-anchor="middle">12</text><rect class="box-1" x="344.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="370.0" y="54.0" text-anchor="middle">13</text><rect class="box-1" x="402.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="428.0" y="54.0" text-anchor="middle">14</text><rect class="box-1" x="460.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="486.0" y="54.0" text-anchor="middle">15</text><rect class="box-2" x="518.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="544.0" y="54.0" text-anchor="middle">30</text><rect class="box-2" x="576.0" y="28.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="602.0" y="54.0" text-anchor="middle">31</text><text class="t-tick" x="170.0" y="90.0" text-anchor="start">reuse 6, compute 2</text><text class="t-note" x="20.0" y="146.0" text-anchor="start">blocks of 4</text><rect class="box-1" x="170.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="196.0" y="146.0" text-anchor="middle">10</text><rect class="box-1" x="228.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="254.0" y="146.0" text-anchor="middle">11</text><rect class="box-1" x="286.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="312.0" y="146.0" text-anchor="middle">12</text><rect class="box-1" x="344.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="370.0" y="146.0" text-anchor="middle">13</text><rect class="waste" x="402.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="428.0" y="146.0" text-anchor="middle">14</text><rect class="waste" x="460.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="486.0" y="146.0" text-anchor="middle">15</text><rect class="box-2" x="518.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="544.0" y="146.0" text-anchor="middle">30</text><rect class="box-2" x="576.0" y="120.0" width="52.0" height="40.0" rx="6"/><text class="t-note" x="602.0" y="146.0" text-anchor="middle">31</text><text class="t-tick" x="170.0" y="182.0" text-anchor="start">reuse 4, compute 4</text><line class="axis" x1="170.0" y1="112.0" x2="398.0" y2="112.0"/><line class="axis" x1="402.0" y1="112.0" x2="630.0" y2="112.0"/><text class="t-tick" x="284.0" y="107.0" text-anchor="middle">block 1: same</text><text class="t-tick" x="516.0" y="107.0" text-anchor="middle">block 2: differs at 30</text><rect class="box-1" x="170.0" y="214.0" width="14.0" height="12.0" rx="3"/><text class="t-tick" x="190.0" y="225.0" text-anchor="start">reused</text><rect class="box-2" x="270.0" y="214.0" width="14.0" height="12.0" rx="3"/><text class="t-tick" x="290.0" y="225.0" text-anchor="start">computed</text><rect class="waste" x="380.0" y="214.0" width="14.0" height="12.0" rx="3"/><text class="t-tick" x="400.0" y="225.0" text-anchor="start">matched, but its block is not full</text></svg><figcaption>The second request against the first. A radix tree with one-token pages reuses six tokens. With four-token blocks, block 1 matches, but block 2 holds 14 15 30 31 instead of 14 15 20 21, so its hash differs and tokens 14 and 15 are recomputed.</figcaption></figure>

### How much does rounding cost?

If the point where two prompts diverge falls at a random place inside a block, the reuse lost to rounding is $$\ell\bmod B$$, which is equally likely to be any of $$0,1,\ldots,B-1$$:

$$
\mathbb{E}[\text{lost}]=\frac{0+1+\cdots+(B-1)}{B}=\frac{B-1}{2}.
$$

For vLLM's default $$B=16$$, that is 7.5 tokens per request on average; the simulation over 100,000 random prefix lengths gives 7.52. Next to a 2,048-token handbook it is noise. It matters only when the shared parts are short, or very many short shared pieces are stitched together.

[![Terminal output of simulate.py sections 5 and 6: mean tokens lost to block rounding 0.00, 7.52, 15.55, 31.56 and 63.57 for block sizes 1, 16, 32, 64 and 128, matching (B-1)/2; then the capacity sweep table of hit rates for radix page 1, radix trimming leaf tails, blocks of 16 and blocks of 64, at capacities 4,096 to 65,536 tokens](/img/sglang/sim_capacity-run.png)](/img/sglang/sim_capacity-run.png)

This does **not** mean SGLang always reuses individual tokens while vLLM always uses blocks. SGLang's page size defaults to one token on NVIDIA GPUs in the source we checked, but it is configurable, some backends and models set larger pages, and its matcher rounds down to the page size exactly like blocks ([`RadixKey.match_at`](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/mem_cache/radix_cache.py#L182)).

### The vLLM picture: lookup, then allocation

vLLM's design page walks through a small example with four-token blocks. Here a second request shares its first 10 tokens with the first:

> [!PAPER] vLLM documentation · Design · Automatic Prefix Caching · Example, Time 3
> [![vLLM docs example: request 0 holds blocks 0 to 4 with hashes A-D, A-H, A-L, A-P; request 1, whose prompt shares 10 tokens, reuses blocks 0 and 1 (hashes A-D and A-H) and gets new blocks 5 and 6; a cache-blocks table maps hashes to block IDs; three free blocks wait in a doubly linked free queue](/img/sglang/vllm-example-time3.png)](/img/sglang/vllm-example-time3.png)
>
> **Context:** step "Time 3" of the worked example, with block size 4 and 10 blocks in total.
>
> **What it says:** "only the first 2 blocks (8 tokens) hit the cache, because the 3rd block only matches 2 of 4 tokens." The hash table maps a hash to a physical block; free blocks wait in a doubly linked queue.
>
> **Why it matters:** this is our `blocks` figure with real data structures: a dictionary from hash to block, plus a free queue. There is no tree, but the parent hash inside every key gives the same prefix meaning.
>
> [Read the example](https://docs.vllm.ai/en/latest/design/prefix_caching/#example)

Two more vLLM details complete the picture. When a whole prompt is already cached, vLLM still recomputes its **last token**, because it needs that position's output scores to pick the first answer token ([`max_cache_hit_length = request.num_tokens - 1`](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/v1/core/kv_cache_manager.py#L295)). Our simulations apply the same rule to both caches. And vLLM's eviction is also LRU, with a twist:

> [!PAPER] vLLM documentation · Design · Automatic Prefix Caching · Free
> [![vLLM docs: when a request is finished, we free all its blocks if no other requests are using them (reference count = 0). The freed blocks are added to the tail of the free queue in the reverse order. This is because the last block of a request must hash more tokens and is less likely to be reused by other requests. As a result, it should be evicted first.](/img/sglang/vllm-free-reverse.png)](/img/sglang/vllm-free-reverse.png)
>
> **Context:** the "Free" operation, just before the "Eviction (LRU)" section.
>
> **What it says:** a finished request's blocks join the LRU free queue last block first, so its tail is evicted before its head.
>
> **Why it matters:** this is vLLM's version of "leaf first". The tail block of a request is the most specific part of it, like a leaf; the first blocks are the most widely shared, like the root's children. The difference is granularity: vLLM can drop *one 16-token block* from the end of a handbook, while SGLang's leaf eviction drops a whole leaf.
>
> [Read the Free section](https://docs.vllm.ai/en/latest/design/prefix_caching/#free)

### Does the tree win when memory is tight?

It is tempting to assume the radix tree must give more hits. So I simulated both caches on one trace, under memory pressure. The trace has 600 requests: 300 instruction tokens, then one of six handbooks of 900 to 2,100 tokens (picked with Zipf-like popularity), then a 20 to 90 token question, with 35% of requests being follow-ups in an existing chat (so the previous answer becomes part of the prefix). Answers of 40 to 160 tokens are cached too. Requests run one at a time. Four caches see the same trace:

- **radix, page 1:** leaf-first LRU, evicting a whole leaf, as in SGLang;
- **radix, trims leaf tails:** an ablation that shortens the LRU leaf from its end by only as many tokens as needed;
- **blocks of 16** and **blocks of 64:** hash-chained full blocks with an LRU free queue in reverse order, as in vLLM.

<figure class="fig"><svg viewBox="0 0 760 332" role="img" aria-label="Hit rate against cache capacity for four caches on the same 600-request trace. Blocks beat the whole-leaf radix cache at small capacity; trimming leaf tails closes the gap."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="90.0" y1="272.0" x2="710.0" y2="272.0"/><text class="t-tick" x="80.0" y="276.0" text-anchor="end">50%</text><line class="grid" x1="90.0" y1="232.0" x2="710.0" y2="232.0"/><text class="t-tick" x="80.0" y="236.0" text-anchor="end">60%</text><line class="grid" x1="90.0" y1="192.0" x2="710.0" y2="192.0"/><text class="t-tick" x="80.0" y="196.0" text-anchor="end">70%</text><line class="grid" x1="90.0" y1="152.0" x2="710.0" y2="152.0"/><text class="t-tick" x="80.0" y="156.0" text-anchor="end">80%</text><line class="grid" x1="90.0" y1="112.0" x2="710.0" y2="112.0"/><text class="t-tick" x="80.0" y="116.0" text-anchor="end">90%</text><line class="grid" x1="90.0" y1="72.0" x2="710.0" y2="72.0"/><text class="t-tick" x="80.0" y="76.0" text-anchor="end">100%</text><text class="t-tick" x="90.0" y="294.0" text-anchor="middle">4,096</text><text class="t-tick" x="240.0" y="294.0" text-anchor="middle">8,192</text><text class="t-tick" x="390.0" y="294.0" text-anchor="middle">16,384</text><text class="t-tick" x="540.0" y="294.0" text-anchor="middle">32,768</text><text class="t-tick" x="690.0" y="294.0" text-anchor="middle">65,536</text><text class="t-tick" x="390.0" y="316.0" text-anchor="middle">cache capacity, tokens (log scale)</text><polyline class="l1" points="90.0,262.2 240.0,158.7 390.0,94.6 540.0,86.9 690.0,86.9"/><g class="mark"><title>radix, page 1, whole-leaf eviction, 4,096 tokens: 52.5%</title><circle class="s1 ring" cx="90.0" cy="262.2" r="5"/></g><g class="mark"><title>radix, page 1, whole-leaf eviction, 8,192 tokens: 78.3%</title><circle class="s1 ring" cx="240.0" cy="158.7" r="5"/></g><g class="mark"><title>radix, page 1, whole-leaf eviction, 16,384 tokens: 94.4%</title><circle class="s1 ring" cx="390.0" cy="94.6" r="5"/></g><g class="mark"><title>radix, page 1, whole-leaf eviction, 32,768 tokens: 96.3%</title><circle class="s1 ring" cx="540.0" cy="86.9" r="5"/></g><g class="mark"><title>radix, page 1, whole-leaf eviction, 65,536 tokens: 96.3%</title><circle class="s1 ring" cx="690.0" cy="86.9" r="5"/></g><rect class="s1" x="20" y="14" width="12" height="12" rx="3"/><text class="t-tick" x="38.0" y="25.0" text-anchor="start">radix, page 1, whole-leaf eviction</text><polyline class="l3" points="90.0,238.9 240.0,145.8 390.0,93.2 540.0,86.9 690.0,86.9"/><g class="mark"><title>radix, trims leaf tails (ablation), 4,096 tokens: 58.3%</title><circle class="s3 ring" cx="90.0" cy="238.9" r="5"/></g><g class="mark"><title>radix, trims leaf tails (ablation), 8,192 tokens: 81.6%</title><circle class="s3 ring" cx="240.0" cy="145.8" r="5"/></g><g class="mark"><title>radix, trims leaf tails (ablation), 16,384 tokens: 94.7%</title><circle class="s3 ring" cx="390.0" cy="93.2" r="5"/></g><g class="mark"><title>radix, trims leaf tails (ablation), 32,768 tokens: 96.3%</title><circle class="s3 ring" cx="540.0" cy="86.9" r="5"/></g><g class="mark"><title>radix, trims leaf tails (ablation), 65,536 tokens: 96.3%</title><circle class="s3 ring" cx="690.0" cy="86.9" r="5"/></g><rect class="s3" x="390" y="14" width="12" height="12" rx="3"/><text class="t-tick" x="408.0" y="25.0" text-anchor="start">radix, trims leaf tails (ablation)</text><polyline class="l2" points="90.0,240.0 240.0,146.8 390.0,94.3 540.0,88.3 690.0,88.3"/><g class="mark"><title>blocks of 16, LRU, 4,096 tokens: 58.0%</title><circle class="s2 ring" cx="90.0" cy="240.0" r="5"/></g><g class="mark"><title>blocks of 16, LRU, 8,192 tokens: 81.3%</title><circle class="s2 ring" cx="240.0" cy="146.8" r="5"/></g><g class="mark"><title>blocks of 16, LRU, 16,384 tokens: 94.4%</title><circle class="s2 ring" cx="390.0" cy="94.3" r="5"/></g><g class="mark"><title>blocks of 16, LRU, 32,768 tokens: 95.9%</title><circle class="s2 ring" cx="540.0" cy="88.3" r="5"/></g><g class="mark"><title>blocks of 16, LRU, 65,536 tokens: 95.9%</title><circle class="s2 ring" cx="690.0" cy="88.3" r="5"/></g><rect class="s2" x="20" y="34" width="12" height="12" rx="3"/><text class="t-tick" x="38.0" y="45.0" text-anchor="start">blocks of 16, LRU</text><polyline class="l4" points="90.0,244.1 240.0,150.0 390.0,97.6 540.0,92.2 690.0,92.2"/><g class="mark"><title>blocks of 64, LRU, 4,096 tokens: 57.0%</title><circle class="s4 ring" cx="90.0" cy="244.1" r="5"/></g><g class="mark"><title>blocks of 64, LRU, 8,192 tokens: 80.5%</title><circle class="s4 ring" cx="240.0" cy="150.0" r="5"/></g><g class="mark"><title>blocks of 64, LRU, 16,384 tokens: 93.6%</title><circle class="s4 ring" cx="390.0" cy="97.6" r="5"/></g><g class="mark"><title>blocks of 64, LRU, 32,768 tokens: 95.0%</title><circle class="s4 ring" cx="540.0" cy="92.2" r="5"/></g><g class="mark"><title>blocks of 64, LRU, 65,536 tokens: 95.0%</title><circle class="s4 ring" cx="690.0" cy="92.2" r="5"/></g><rect class="s4" x="390" y="34" width="12" height="12" rx="3"/><text class="t-tick" x="408.0" y="45.0" text-anchor="start">blocks of 64, LRU</text><line class="axis" x1="90.0" y1="272.0" x2="710.0" y2="272.0"/></svg><figcaption>Token hit rate against cache capacity on the same 600-request trace. With a small cache the 16-token block cache beats the whole-leaf radix cache (58.0% against 52.5% at 4,096 tokens). Letting the radix cache trim leaf tails (58.3%) closes the gap, so eviction granularity, not the index shape, is what differs. With room for the working set all four reach 95% to 96%.</figcaption></figure>

| Capacity (tokens) | Radix, page 1 | Radix, trims tails | Blocks of 16 | Blocks of 64 |
|---:|---:|---:|---:|---:|
| 4,096 | 52.5% | 58.3% | 58.0% | 57.0% |
| 8,192 | 78.3% | 81.6% | 81.3% | 80.5% |
| 16,384 | 94.4% | 94.7% | 94.4% | 93.6% |
| 32,768 | 96.3% | 96.3% | 95.9% | 95.0% |

Read it carefully, because it cuts against the easy story:

- **With enough memory, the index hardly matters.** At 32,768 tokens every cache is within 1.3 points. The radix tree's small edge is the block-rounding loss computed above.
- **Under memory pressure, the block cache won in this simulation.** At 4,096 tokens, it kept 58.0% against 52.5%. The ablation shows why: a radix cache that may trim a leaf's tail gets 58.3%. Whole-leaf eviction throws away the useful start of a long handbook together with its end; block eviction peels a sequence from its end, block by block.
- **This is a simulation, not an engine benchmark.** It runs one request at a time, has no concurrency, no batching, no real memory pool shared with running requests, and only one trace. It cannot tell you which engine has the higher hit rate on your traffic. What it does show is that "tree versus hash table" is the wrong question; granularity of eviction, block size and what you count as capacity matter more.

A simpler workload with every shared part a multiple of 16 tokens and unlimited memory shows the "enough memory" end of this table cleanly: both indexes compute exactly the same 6,144 tokens instead of 86,016.

<figure class="fig"><svg viewBox="0 0 760 255" role="img" aria-label="Both teaching cache indexes compute 6,144 tokens versus 86,016 without reuse."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="180.0" y="49.0" text-anchor="end">No reuse</text><rect class="box-2" x="200.0" y="25.0" width="450.0" height="36.0" rx="8"/><text class="t-note" x="660.0" y="49.0" text-anchor="start">86,016</text><text class="t-note" x="180.0" y="114.0" text-anchor="end">Radix index</text><rect class="box-1" x="200.0" y="90.0" width="32.1" height="36.0" rx="8"/><text class="t-note" x="242.1" y="114.0" text-anchor="start">6,144</text><text class="t-note" x="180.0" y="179.0" text-anchor="end">Block-hash index</text><rect class="box-1" x="200.0" y="155.0" width="32.1" height="36.0" rx="8"/><text class="t-note" x="242.1" y="179.0" text-anchor="start">6,144</text><text class="t-tick" x="380.0" y="235.0" text-anchor="middle">Prompt token positions computed across 64 requests</text></svg><figcaption>With 1,024 shared instruction tokens, four 256-token sections and 64 unique 64-token questions (all multiples of 16) and no capacity limit, both indexes avoid the same 79,872 repeated token computations.</figcaption></figure>

### Where PagedAttention fits

[Part 3](llm-inference-3-vllm.md) introduced vLLM's PagedAttention. It is easy to mix up two different jobs:

| Job | Question it answers |
|---|---|
| **Paged KV storage** | Where in memory should this request's keys and values go? |
| **Prefix-cache lookup** | Have these token positions already been computed, and where is their saved state? |

A radix tree can point to KV stored in pages; SGLang's paper says RadixAttention is "compatible with" paged attention and continuous batching, and its pages are simply one token long by default. RadixAttention does not replace the attention computation or make paging unnecessary. It is an index and an eviction policy on top of paged storage.

## 10. When the assistant gets busy: the order of requests matters

So far, one request finished before the next started, in arrival order. A real service has a queue. When several requests are waiting, the scheduler chooses which runs next, and that choice decides what is still in the cache when each request runs.

### The paper's idea: longest prefix first

> [!PAPER] Zheng et al., SGLang · Section 3, "Cache-aware scheduling" · pages 5 and 6
> [![Paper text: in the batch-processing setting we sort the requests by matched prefix length and prioritize requests with longer matched prefixes instead of using a first-come, first-served schedule. Theorem 3.1: for a batch of requests, we can achieve an optimal cache hit rate by visiting the radix tree of the requests in the depth-first search order, with a cache size at least the maximum request length; the longest-shared-prefix-first order is equivalent to a depth-first search order. While greedy cache-aware scheduling can achieve high throughput, it can lead to starvation; integration with fair scheduling is left as future work.](/img/sglang/paper-schedule.png)](/img/sglang/paper-schedule.png)
>
> **Context:** the paragraph after the hit-rate definition. The proof is in Appendix A.3, and the pseudocode (Algorithm 1) is in the research notes at the end of this article.
>
> **What it says:** sort the waiting requests by how much of each is already cached, and run the longest match first. For a batch known in advance, visiting the tree depth-first gives the best possible hit rate, as long as the cache can hold the longest single request; longest-prefix-first produces a depth-first order. The authors also warn that this greedy order "can lead to starvation".
>
> **Why it matters:** the theorem gives a lower bound on work. Every edge of the requests' tree must be computed at least once, so the minimum prefill is $$\sum_{e}|e|$$; depth-first order reaches it because it finishes everything under an edge before leaving it. The starvation warning is the price, and we will measure it.
>
> [Read pages 5 and 6](https://arxiv.org/pdf/2312.07104v2#page=6)

**The equations.** If one server prefills waiting requests one at a time, and request $$k$$ in the chosen order takes $$t_k$$, then request $$k$$ gets its first token at

$$
W_k=\sum_{j\le k}t_j,\qquad \overline{W}=\frac1n\sum_{k=1}^{n}W_k=\frac1n\sum_{k=1}^{n}(n-k+1)\,t_k.
$$

Every request's time $$t_j$$ is added into the wait of everyone after it. A schedule that turns misses into hits early shortens not only those requests but every request behind them. And the lower bound from the theorem says the total computed tokens can never go below $$\sum_e |e|$$.

### A simulation: four handbooks, room for two

Now 32 handbook questions are waiting at once, arriving in the order A B C D A B C D and so on (four different handbook sections). Each request is 512 instruction tokens, a 1,536-token section and a 64-token question. The cache holds 4,096 tokens: the instructions plus two sections and a few questions. One prefill runs at a time, timed with the cost model from Section 4.

The best possible work is the size of the tree: $$512+4\times1536+32\times64=8{,}704$$ tokens.

<figure class="fig"><svg viewBox="0 0 760 226" role="img" aria-label="Thirty-two waiting requests served in arrival order switch handbooks every request and take 2,630 ms. Longest-prefix-match order groups them and finishes in 740 ms."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="20.0" y="62.0" text-anchor="start">FCFS</text><g class="mark"><title>FCFS: handbook A, 102 ms, 2,112 tokens computed</title><rect class="s1" x="70.00" y="40" width="24.42" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="95.22" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="115.38" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="135.53" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="155.68" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="175.84" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="195.99" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="216.15" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="236.30" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="256.46" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="276.61" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="296.76" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="316.92" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="337.07" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="357.23" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="377.38" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="397.53" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="417.69" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="437.84" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="458.00" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="478.15" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="498.30" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="518.46" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="538.61" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="558.77" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="578.92" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="599.07" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="619.23" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook A, 82 ms, 1,600 tokens computed</title><rect class="s1" x="639.38" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="659.54" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="679.69" y="40" width="19.35" height="32" rx="2"/></g><g class="mark"><title>FCFS: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="699.85" y="40" width="19.35" height="32" rx="2"/></g><text class="t-tick" x="728.0" y="92.0" text-anchor="end">all 32 done at 2,630 ms · 51,712 tokens computed · hit rate 23%</text><text class="t-note" x="20.0" y="140.0" text-anchor="start">LPM</text><g class="mark"><title>LPM: handbook A, 102 ms, 2,112 tokens computed</title><rect class="s1" x="70.00" y="118" width="24.42" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="95.22" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="98.69" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="102.17" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="105.64" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="109.11" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="112.58" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook A, 14 ms, 64 tokens computed</title><rect class="s1" x="116.05" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 82 ms, 1,600 tokens computed</title><rect class="s2" x="119.52" y="118" width="19.35" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="139.68" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="143.15" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="146.62" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="150.10" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="153.57" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="157.04" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook B, 14 ms, 64 tokens computed</title><rect class="s2" x="160.51" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 82 ms, 1,600 tokens computed</title><rect class="s3" x="163.98" y="118" width="19.35" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="184.14" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="187.61" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="191.08" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="194.55" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="198.03" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="201.50" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook C, 14 ms, 64 tokens computed</title><rect class="s3" x="204.97" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 82 ms, 1,600 tokens computed</title><rect class="s4" x="208.44" y="118" width="19.35" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="228.59" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="232.07" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="235.54" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="239.01" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="242.48" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="245.95" y="118" width="2.67" height="32" rx="2"/></g><g class="mark"><title>LPM: handbook D, 14 ms, 64 tokens computed</title><rect class="s4" x="249.43" y="118" width="2.67" height="32" rx="2"/></g><text class="t-tick" x="260.9" y="170.0" text-anchor="start">all 32 done at 740 ms · 8,704 tokens computed · hit rate 87%</text><rect class="s1" x="70" y="14" width="12" height="12" rx="3"/><text class="t-tick" x="88.0" y="25.0" text-anchor="start">handbook A</text><rect class="s2" x="190" y="14" width="12" height="12" rx="3"/><text class="t-tick" x="208.0" y="25.0" text-anchor="start">handbook B</text><rect class="s3" x="310" y="14" width="12" height="12" rx="3"/><text class="t-tick" x="328.0" y="25.0" text-anchor="start">handbook C</text><rect class="s4" x="430" y="14" width="12" height="12" rx="3"/><text class="t-tick" x="448.0" y="25.0" text-anchor="start">handbook D</text><line class="axis" x1="70.0" y1="196.0" x2="720.0" y2="196.0"/><text class="t-tick" x="70.0" y="214.0" text-anchor="start">0 ms</text><text class="t-tick" x="720.0" y="214.0" text-anchor="end">2,630 ms</text></svg><figcaption>Thirty-two waiting requests, coloured by handbook section. In arrival order (FCFS) the scheduler switches section every request, so every section is evicted before it is needed again. Longest-prefix-match (LPM) runs all requests for one section together; each section is computed once.</figcaption></figure>

| Order | Tokens computed | Hit rate | Last request done | Mean wait | Worst wait |
|---|---:|---:|---:|---:|---:|
| FCFS (arrival order) | 51,712 | 23.5% | 2,630 ms | 1,366 ms | 2,630 ms |
| Random | 27,136 | 59.8% | 1,550 ms | 841 ms | 1,550 ms |
| LPM (longest prefix first) | **8,704** | **87.1%** | **740 ms** | **421 ms** | **740 ms** |

LPM reached the theorem's lower bound exactly: 8,704 tokens. The arithmetic of the timeline is easy to check. Under FCFS the first request costs about 102 ms and each of the other 31 misses its section and costs about 82 ms (instructions still cached), so $$102+31\times82\approx2{,}630$$ ms. Under LPM there are four section misses and 28 hits of about 14 ms: $$102+3\times82+28\times14\approx740$$ ms. The mean wait falls from 1,366 ms to 421 ms because, by the formula above, every saved millisecond early in the queue is saved again for everyone behind it.

Arrival order was the worst possible case here on purpose: it cycles through more sections than fit in the cache, which is exactly the "cache thrashing" the paper describes. With arrivals already grouped, FCFS would do as well as LPM.

### The price: starvation

Now a different situation. Handbook A is popular: 60 A questions arrive, one every 10 ms. Each takes about 14 ms with A cached, so a queue builds up. One handbook-D question arrives at 25 ms. D is not cached and needs a full section prefill.

<figure class="fig"><svg viewBox="0 0 760 214" role="img" aria-label="Under FCFS the handbook-D question waits 187 ms. Under longest-prefix-match it waits 987 ms because cached A questions keep jumping ahead."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="210.0" y="33.0" text-anchor="end">FCFS: the D question</text><g class="mark"><title>FCFS: the D question: 187 ms</title><rect class="s2" x="222.0" y="18.0" width="81.3" height="22.0" rx="4"/></g><text class="t-val" x="311.3" y="34.0" text-anchor="start">187 ms</text><text class="t-note" x="210.0" y="73.0" text-anchor="end">FCFS: A questions, mean</text><g class="mark"><title>FCFS: A questions, mean: 299 ms</title><rect class="s1" x="222.0" y="58.0" width="130.2" height="22.0" rx="4"/></g><text class="t-val" x="360.2" y="74.0" text-anchor="start">299 ms</text><text class="t-note" x="210.0" y="125.0" text-anchor="end">LPM: the D question</text><g class="mark"><title>LPM: the D question: 987 ms</title><rect class="s2" x="222.0" y="110.0" width="430.0" height="22.0" rx="4"/></g><text class="t-val" x="660.0" y="126.0" text-anchor="start">987 ms</text><text class="t-note" x="210.0" y="165.0" text-anchor="end">LPM: A questions, mean</text><g class="mark"><title>LPM: A questions, mean: 221 ms</title><rect class="s1" x="222.0" y="150.0" width="96.4" height="22.0" rx="4"/></g><text class="t-val" x="326.4" y="166.0" text-anchor="start">221 ms</text><text class="t-tick" x="380.0" y="200.0" text-anchor="middle">60 handbook-A questions arrive every 10 ms; one handbook-D question arrives at 25 ms.</text></svg><figcaption>Waiting times when a stream of cached handbook-A questions competes with one uncached handbook-D question. LPM keeps choosing the cached A requests, so D waits until the stream ends.</figcaption></figure>

| Order | D question waits | A questions, mean | A questions, worst |
|---|---:|---:|---:|
| FCFS | 187 ms | 299 ms | 422 ms |
| LPM | **987 ms** | 221 ms | 341 ms |

LPM made the average A request faster but made the D user wait 5.3 times longer, until the A stream ended. In a real service with a never-ending A stream, D could wait much longer. That is the starvation the paper warns about.

[![Terminal output of simulate.py sections 7 and 8: the fitted cost model; scheduling order table for fcfs, random and lpm with tokens computed 51,712, 27,136 and 8,704 and mean waits 1,366, 841 and 421 ms; starvation case where D waits 187 ms under fcfs and 987 ms under lpm](/img/sglang/sim_schedule-run.png)](/img/sglang/sim_schedule-run.png)

**What this simulation can and cannot show.** It isolates the effect of order on hit rate and waiting, with one server, one prefill at a time, no decode and no batching. Real engines batch many prefills and decodes per step (Part 3), so the gaps would be smaller and shaped differently. The direction of both effects, fewer misses with grouping and longer waits for the unlucky request, is what carries over.

### What the engines do today

This is where the paper and the current code differ, and where the comparison is most often overstated.

- **SGLang's default is first come, first served.** The `--schedule-policy` flag defaults to `fcfs` in the source we checked ([`schedule.py`](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/arg_groups/fields/schedule.py#L82)). The paper's longest-prefix-match is available as `lpm`, alongside `dfs-weight`, `lof`, `priority`, `random`, `routing-key` and others.
- **Even with `lpm`, very long queues fall back to FCFS.** When more than 128 requests are waiting, SGLang turns off the prefix matching and sorting because it is expensive ([`_determine_active_policy`](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/managers/schedule_policy.py#L271)).
- **SGLang also has an in-batch trick.** If several waiting requests share a prefix that is not cached yet, it can run one of them first so the others hit the cache afterwards (the "in-batch prefix caching" check in the same file).
- **vLLM's scheduler offers `fcfs` and `priority`** ([`SchedulerPolicy`](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/scheduler.py#L159)); neither sorts the queue by cached prefix length.

> [!PAPER] SGLang documentation · Hyperparameter Tuning
> [![SGLang docs: If the workload has many shared prefixes, try --schedule-policy lpm. Here, lpm stands for longest prefix match. It reorders requests to encourage more cache hits but introduces more scheduling overhead.](/img/sglang/sglang-lpm-doc.png)](/img/sglang/sglang-lpm-doc.png)
>
> **Context:** the tuning guide's list of scheduler settings.
>
> **What it says:** LPM is a setting you *try* for prefix-heavy workloads; it costs scheduling overhead.
>
> **Why it matters:** if you benchmark SGLang with default flags, you are not testing the paper's cache-aware scheduling. Turn it on deliberately, and watch the slowest requests as well as the average.
>
> [Read the tuning guide](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning)

### Several servers: route to the warm one

With several replicas, *placement* is the same problem one level up. A request sent to the worker that already holds its handbook avoids rebuilding it; a request sent round-robin probably lands on a cold worker.

> [!PAPER] LMSYS blog, "SGLang v0.4" (December 2024) · Cache-Aware Load Balancer
> [![Diagram: round-robin data parallel routing in SGLang v0.3 sends prefix 1 and prefix 2 requests to both workers, about 20% cache hit rate; the cache-aware router in v0.4 keeps an approximate tree per worker and sends each prefix to one worker, about 75% cache hit rate](/img/sglang/blog-cache-aware-router.png)](/img/sglang/blog-cache-aware-router.png)
>
> **Context:** the "Cache-Aware Load Balancer" section of the v0.4 release post.
>
> **What it says:** the router keeps an approximate radix tree for each worker and sends a request to the worker with the longest match, while balancing load. On their benchmark, throughput went from 82,665 to 158,596 tokens/s and the hit rate from 20% to 75%.
>
> **Why it matters:** the team notes the benchmark had "multiple long prefix groups" that were "perfectly balanced", which is close to the best case. The idea is sound; your gain depends on how concentrated your prefixes are and how evenly they load the workers. A warm worker with a long queue can still lose to an idle cold one.
>
> [Read the post](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)

## 11. A second idea from the paper: decode JSON faster

Suppose the assistant must answer in a fixed format so another program can read it:

```json
{"employee": "Asha Rao", "leave_type": "annual", "days": 12}
```

Engines enforce such a format with **constrained decoding**: the schema becomes a grammar or a finite state machine (FSM), and at each step tokens that would break the format get probability zero. Both SGLang and vLLM do this today through grammar libraries (SGLang: `xgrammar` by default, also `outlines` and `llguidance`; vLLM: `auto`, choosing between backends such as `xgrammar` and `guidance`).

The SGLang paper noticed a waste. Much of a JSON answer is not a choice at all: after `{` the schema forces `"employee": "`. Decoding those characters one token per forward pass spends a full model step on text the model has no say in.

> [!PAPER] Zheng et al., SGLang · Figure 4 · page 6
> [![Figure 4 of the SGLang paper: (a) a normal FSM for the regex {"summary": " has one state per character; (b) the compressed FSM has a single edge carrying the whole string; (c) decoding with the normal FSM alternates tokens and LLM calls; (d) with the compressed FSM the tokens are added together before one LLM call](/img/sglang/paper-fig4.png)](/img/sglang/paper-fig4.png)
>
> **Context:** Section 4, "Efficient Constrained Decoding with Compressed Finite State Machine".
>
> **What it says:** where an FSM state has only one way forward, consecutive edges are merged into one **compressed edge**. The whole forced string can then be added in a single forward pass instead of one pass per token.
>
> **Why it matters:** this is **jump-forward decoding**. The model still processes the forced tokens (they enter the KV cache like a short prefill), but in one step. On the paper's JSON benchmark the authors report 1.6x higher throughput from the compressed FSM.
>
> [Read Section 4](https://arxiv.org/pdf/2312.07104v2#page=6)

**The equation.** If an answer has $$n_f$$ forced tokens grouped into $$r$$ forced runs, and $$n_m$$ tokens the model chooses, then

$$
\text{passes}_{\text{token by token}}=n_f+n_m,\qquad \text{passes}_{\text{jump}}=n_m+r.
$$

The saving is $$n_f-r$$ passes. It is large when forced runs are long (verbose keys, fixed templates) and small when the model writes most of the text.

**Worked numbers.** I tokenized our JSON answer with the real Qwen2.5 tokenizer and marked each token as forced or chosen. The model chooses where the free string `Asha Rao` ends, so its closing quote counts as a choice; after the fixed value `annual` the closing quote is forced.

<figure class="fig"><svg viewBox="0 0 760 214" role="img" aria-label="The JSON answer is 22 tokens. Decoded one token at a time it needs 22 forward passes; merging the four forced runs needs 11."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="20.0" y="22.0" text-anchor="start">token by token: 22 forward passes</text><rect class="box-2" x="20.0" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="34.2" y="54.0" text-anchor="middle">{&quot;</text><rect class="box-2" x="52.4" y="34.0" width="77.6" height="30.0" rx="5"/><text class="t-tick" x="91.2" y="54.0" text-anchor="middle">employee</text><rect class="box-2" x="134.0" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="148.2" y="54.0" text-anchor="middle">&quot;:</text><rect class="box-2" x="166.4" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="180.6" y="54.0" text-anchor="middle">·&quot;</text><rect class="box-1" x="198.8" y="34.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="210.8" y="54.0" text-anchor="middle">A</text><rect class="box-1" x="226.8" y="34.0" width="36.6" height="30.0" rx="5"/><text class="t-tick" x="245.1" y="54.0" text-anchor="middle">sha</text><rect class="box-1" x="267.4" y="34.0" width="44.8" height="30.0" rx="5"/><text class="t-tick" x="289.8" y="54.0" text-anchor="middle">·Rao</text><rect class="box-1" x="316.2" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="330.4" y="54.0" text-anchor="middle">&quot;,</text><rect class="box-2" x="348.6" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="362.8" y="54.0" text-anchor="middle">·&quot;</text><rect class="box-2" x="381.0" y="34.0" width="53.0" height="30.0" rx="5"/><text class="t-tick" x="407.5" y="54.0" text-anchor="middle">leave</text><rect class="box-2" x="438.0" y="34.0" width="53.0" height="30.0" rx="5"/><text class="t-tick" x="464.5" y="54.0" text-anchor="middle">_type</text><rect class="box-2" x="495.0" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="509.2" y="54.0" text-anchor="middle">&quot;:</text><rect class="box-2" x="527.4" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="541.6" y="54.0" text-anchor="middle">·&quot;</text><rect class="box-1" x="559.8" y="34.0" width="61.2" height="30.0" rx="5"/><text class="t-tick" x="590.4" y="54.0" text-anchor="middle">annual</text><rect class="box-2" x="625.0" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="639.2" y="54.0" text-anchor="middle">&quot;,</text><rect class="box-2" x="657.4" y="34.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="671.6" y="54.0" text-anchor="middle">·&quot;</text><rect class="box-2" x="689.8" y="34.0" width="44.8" height="30.0" rx="5"/><text class="t-tick" x="712.2" y="54.0" text-anchor="middle">days</text><rect class="box-2" x="20.0" y="74.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="34.2" y="94.0" text-anchor="middle">&quot;:</text><rect class="box-2" x="52.4" y="74.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="64.4" y="94.0" text-anchor="middle">·</text><rect class="box-1" x="80.4" y="74.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="92.4" y="94.0" text-anchor="middle">1</text><rect class="box-1" x="108.4" y="74.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="120.4" y="94.0" text-anchor="middle">2</text><rect class="box-2" x="136.4" y="74.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="148.4" y="94.0" text-anchor="middle">}</text><text class="t-note" x="20.0" y="128.0" text-anchor="start">jump-forward: 11 forward passes (each forced run is one extend step)</text><rect class="box-2" x="20.0" y="140.0" width="126.8" height="30.0" rx="5"/><text class="t-tick" x="83.4" y="160.0" text-anchor="middle">{&quot;employee&quot;:·&quot;</text><rect class="box-1" x="150.8" y="140.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="162.8" y="160.0" text-anchor="middle">A</text><rect class="box-1" x="178.8" y="140.0" width="36.6" height="30.0" rx="5"/><text class="t-tick" x="197.1" y="160.0" text-anchor="middle">sha</text><rect class="box-1" x="219.4" y="140.0" width="44.8" height="30.0" rx="5"/><text class="t-tick" x="241.8" y="160.0" text-anchor="middle">·Rao</text><rect class="box-1" x="268.2" y="140.0" width="28.4" height="30.0" rx="5"/><text class="t-tick" x="282.4" y="160.0" text-anchor="middle">&quot;,</text><rect class="box-2" x="300.6" y="140.0" width="143.2" height="30.0" rx="5"/><text class="t-tick" x="372.2" y="160.0" text-anchor="middle">·&quot;leave_type&quot;:·&quot;</text><rect class="box-1" x="447.8" y="140.0" width="61.2" height="30.0" rx="5"/><text class="t-tick" x="478.4" y="160.0" text-anchor="middle">annual</text><rect class="box-2" x="513.0" y="140.0" width="102.2" height="30.0" rx="5"/><text class="t-tick" x="564.1" y="160.0" text-anchor="middle">&quot;,·&quot;days&quot;:·</text><rect class="box-1" x="619.2" y="140.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="631.2" y="160.0" text-anchor="middle">1</text><rect class="box-1" x="647.2" y="140.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="659.2" y="160.0" text-anchor="middle">2</text><rect class="box-2" x="675.2" y="140.0" width="24.0" height="30.0" rx="5"/><text class="t-tick" x="687.2" y="160.0" text-anchor="middle">}</text><rect class="box-2" x="20.0" y="190.0" width="14.0" height="12.0" rx="3"/><text class="t-tick" x="40.0" y="201.0" text-anchor="start">forced by the schema</text><rect class="box-1" x="210.0" y="190.0" width="14.0" height="12.0" rx="3"/><text class="t-tick" x="230.0" y="201.0" text-anchor="start">chosen by the model</text><text class="t-tick" x="400.0" y="201.0" text-anchor="start">Qwen2.5 tokenizer; · marks a space</text></svg><figcaption>The same 22-token JSON answer, decoded token by token (top) and with each run of forced tokens merged into one step (bottom): 15 forced tokens in 4 runs, 7 chosen tokens, so 22 passes become 11.</figcaption></figure>

$$
n_f=15,\quad n_m=7,\quad r=4:\qquad 15+7=22\ \text{passes}\ \longrightarrow\ 7+4=11\ \text{passes}.
$$

[![Terminal output of simulate.py section 9: the JSON answer, 22 tokens of which 15 forced and 7 free in 4 forced runs; forward passes 22 token by token and 11 with jump-forward](/img/sglang/sim_jump-run.png)](/img/sglang/sim_jump-run.png)

Look at the token `",` after `Rao`: one token holds the model's closing quote and a forced comma. Tokens do not respect the grammar's boundaries, so a real implementation must handle text that is half forced and half chosen. The SGLang team's post on this method describes exactly that "tokenization boundary" problem and re-tokenizes the jumped text.

> [!PAPER] LMSYS blog, "Fast JSON Decoding for Local LLMs with Compressed Finite State Machine" (February 2024) · Figure 5
> [![Comparison of jump-forward decoding with compressed FSM and normal decoding: for a Harry Potter character JSON, jump-forward decoding prefills whole forced chunks such as the "age" key in orange while the model decodes only Har, ry, Pot, ter, 1, 1 and G in blue; normal decoding with an FSM needs a decode step for every token including braces, newlines and keys](/img/sglang/blog-jump-forward.png)](/img/sglang/blog-jump-forward.png)
>
> **Context:** the method section of the post that introduced jump-forward decoding.
>
> **What it says:** orange chunks are forced text added in one step; blue boxes are real decode steps. Once the model writes `G` for the house, `ryffindor",` is forced and jumped.
>
> **Why it matters:** forced text is not only punctuation and keys. With an enum, the first few characters can force the rest of the value.
>
> [Read the post](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/)

**Current status, stated carefully.** In the SGLang source we checked, the grammar backends still define a `try_jump_forward` hook ([`base_grammar_backend.py`](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/constrained/base_grammar_backend.py#L119)), but no scheduler code calls it. So treat jump-forward decoding as a published idea with a measured result, not as something the default server does for you today. Both engines' structured-output speed now comes mainly from fast grammar engines such as XGrammar; the SGLang v0.4 post reported up to 10x faster structured outputs from adopting it. Valid JSON still needs correct facts, so check those separately. [Sources: SGLang structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs), [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).

## 12. Two more things a busy server needs

### Prepare the next batch while the GPU runs the current one

Even when KV reuse works, the GPU can sit idle while the CPU prepares the next batch: matching prefixes, allocating memory, building the input arrays.

**The equation.** If CPU preparation takes $$c$$ and GPU execution takes $$g$$ per batch, a serial loop costs $$c+g$$ per batch, and an overlapped loop costs $$\max(c,g)$$ once it is running. With $$c=2$$ ms and $$g=8$$ ms, 100 batches take 1,000 ms serially and $$2+8+99\times8=802$$ ms overlapped.

<figure class="fig"><svg viewBox="0 0 760 285" role="img" aria-label="CPU scheduling overlaps GPU execution. Ideal steady state changes from 10 milliseconds to 8 milliseconds per batch."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="15.0" y="18.0" text-anchor="start">Serial</text><text class="t-tick" x="15.0" y="43.0" text-anchor="start">CPU</text><rect class="box-2" x="110.0" y="25.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="130.0" y="57.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="210.0" y="25.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="230.0" y="57.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="310.0" y="25.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="330.0" y="57.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="410.0" y="25.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="430.0" y="57.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="510.0" y="25.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="530.0" y="57.0" width="80.0" height="25.0" rx="8"/><text class="t-tick" x="15.0" y="73.0" text-anchor="start">GPU</text><text class="t-note" x="15.0" y="123.0" text-anchor="start">Overlap</text><text class="t-tick" x="15.0" y="148.0" text-anchor="start">CPU</text><rect class="box-2" x="110.0" y="130.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="130.0" y="162.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="190.0" y="130.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="210.0" y="162.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="270.0" y="130.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="290.0" y="162.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="350.0" y="130.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="370.0" y="162.0" width="80.0" height="25.0" rx="8"/><rect class="box-2" x="430.0" y="130.0" width="20.0" height="25.0" rx="8"/><rect class="box-1" x="450.0" y="162.0" width="80.0" height="25.0" rx="8"/><text class="t-tick" x="15.0" y="178.0" text-anchor="start">GPU</text><text class="t-tick" x="380.0" y="240.0" text-anchor="middle">Illustrative: CPU preparation 2 ms, GPU work 8 ms; each unit is 1 ms.</text><text class="t-tick" x="380.0" y="263.0" text-anchor="middle">After startup, the CPU prepares the next batch while the GPU runs.</text></svg><figcaption>Orange is CPU preparation; blue is GPU execution. In the lower timeline, preparation for the next batch overlaps the current batch's GPU work. The durations are illustrative.</figcaption></figure>

> [!PAPER] LMSYS blog, "SGLang v0.4" (December 2024) · Zero-Overhead Batch Scheduler
> [![Timeline from the post: the CPU launches compute batch 2, processes results of batch 1, launches sample batch 2 and prepares compute batch 3 while the GPU runs compute batch 2 and sample batch 2, then compute batch 3, with no idle gap on the GPU](/img/sglang/blog-overlap-scheduler.png)](/img/sglang/blog-overlap-scheduler.png)
>
> **Context:** the "Zero-Overhead Batch Scheduler" section. The team says the idea had been proposed earlier in NanoFlow.
>
> **What it says:** the scheduler runs one batch ahead, so the CPU work for batch 3 happens while the GPU computes batch 2. The team reported a 1.1x speedup over their previous version and stated that the feature is on by default.
>
> **Why it matters:** this saves idle time, not repeated computation; it is independent of prefix caching. It is also not unique to SGLang: vLLM's current configuration enables asynchronous scheduling by default unless an incompatible option is set ([`vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/vllm.py#L1615)).
>
> [Read the post](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/)

### Keep more handbooks outside GPU memory

SGLang's **HiCache** extends the radix cache into host memory and optional storage backends (off by default, `--enable-hierarchical-cache`). vLLM has a native CPU KV offloading option ([`kv_offloading_size`](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/cache.py#L266)) and connectors such as [LMCache](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/). A host-memory hit still has to copy the KV back to the GPU before it can be used. [Source: HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design).

**The decision rule.** Fetching saved work pays off only if moving it is cheaper than recomputing it:

$$
\frac{\text{prefix bytes}}{\text{transfer bandwidth}}\ <\ t_{\text{recompute}}.
$$

For a 1 GiB prefix (8,192 tokens on Llama 3.1 8B) at an assumed effective 25 GB/s, the copy alone takes $$2^{30}/(25\times10^9)\approx43$$ ms. If recomputing that prefix takes 20 ms, a blocking copy loses; if it takes 200 ms, the copy wins. Lookup, allocation and contention add costs, and overlapping the copy with other work can hide some of them. The 25 GB/s is an assumption for the arithmetic, not a measurement.

## 13. What the paper measured, and what it does not tell you today

The headline numbers from the paper are real, but they compare SGLang with the vLLM of late 2023.

> [!PAPER] Zheng et al., SGLang · Figure 5 · page 7
> [![Figure 5 of the SGLang paper: normalized throughput on Llama-7B for SGLang, vLLM, Guidance and LMQL across MMLU, ReAct agents, generative agents, tree of thought, skeleton of thought, LLM judge, HellaSwag, JSON decoding, multi-turn chat short and long, and a DSPy RAG pipeline; SGLang is 1.0 on every task, vLLM ranges from near zero on HellaSwag to close to 1.0 on long multi-turn chat](/img/sglang/paper-fig5.png)](/img/sglang/paper-fig5.png)
>
> **Context:** Section 6. Llama-7B in FP16 on one NVIDIA A10G (24 GB); the baseline is vLLM **v0.2.5** with its default API server. A footnote says RadixAttention had by then been "partially integrated as an optional experimental feature" into vLLM, so the authors used the earlier version.
>
> **What it says:** up to 6.4x higher throughput and up to 3.7x lower latency across these workloads, from KV reuse, parallelism inside one program, and faster constrained decoding. On long multi-turn chat, where decoding dominates, there is almost no speedup.
>
> **Why it matters:** in late 2023 the baseline had no automatic prefix caching, so the gap measures "reuse versus no reuse" as much as "SGLang versus vLLM". Today vLLM caches prefixes by default. Use the figure to see *which workloads* benefit from reuse (few-shot, branching, agents), not as a current engine ranking.
>
> [Read Section 6](https://arxiv.org/pdf/2312.07104v2#page=7)

The same evaluation also measured what the tree costs when nothing is shared:

> [!PAPER] Zheng et al., SGLang · Section 6.3, "Overhead of RadixAttention" · page 9
> [![Paper paragraph: on a ShareGPT benchmark with no KV cache reuse opportunities, running 100 requests takes 74.3 seconds, while managing the RadixAttention data structures takes only 0.2 seconds, a negligible overhead of less than 0.3 percent, so RadixAttention can be on by default](/img/sglang/paper-overhead.png)](/img/sglang/paper-overhead.png)
>
> **Context:** the ablation section, testing the worst case for a cache: no reuse at all.
>
> **What it says:** 0.2 seconds of tree management out of 74.3 seconds, under 0.3%.
>
> **Why it matters:** the index is cheap compared with the model, which is why both engines can leave prefix caching on by default. Our own CPU simulation agrees in spirit: matching and inserting thousands of requests takes milliseconds of Python.
>
> [Read Section 6.3](https://arxiv.org/pdf/2312.07104v2#page=9)

## 14. So how does SGLang differ from vLLM?

We can now compare the engines on the problems we have actually worked through. Defaults are from the source code checked on 8 October 2026 (SGLang `b2cb249`, vLLM `c23ca06`) and can change between releases.

| Problem | SGLang | vLLM |
|---|---|---|
| Prefix caching on by default? | Yes (`--disable-radix-cache` turns it off) | Yes (`enable_prefix_caching=True`) |
| How saved work is found | Radix tree over token IDs | Hash table of chained full-block hashes (SHA-256 by default) |
| Reuse granularity | Page size; 1 token by default on NVIDIA GPUs | Block size; 16 tokens by default |
| Generated tokens cached? | Yes | Yes (full blocks) |
| Eviction | Leaf-first, whole leaf; `lru` default, also `lfu`, `slru`, `priority`, `tlru` | LRU free queue; a request's blocks freed last block first |
| Isolation between tenants | Extra key and cache salt in the radix key | `cache_salt` and extra hashes in the block key |
| Queue order | `fcfs` default; `lpm` and others opt-in; LPM falls back to FCFS above 128 waiting | `fcfs` or `priority` |
| Routing across replicas | `cache_aware` policy in SGLang Model Gateway, the router introduced in v0.4 | Not part of the engine; separate routing projects |
| CPU and GPU overlap | Overlap scheduler, on by default | Async scheduling, on by default when compatible |
| KV beyond GPU memory | HiCache (host memory, storage backends), off by default | Native CPU offloading, LMCache and other connectors |
| Structured output | Grammar backends, `xgrammar` default; jump-forward hook not called | Grammar backends, `auto` (for example `xgrammar`, `guidance`) |

What this table says, in one paragraph: **both engines avoid repeated prefill for shared beginnings, by default.** The real differences are smaller and more specific than "tree versus no tree": token-level versus block-level matching (a few tokens per request), whole-leaf versus block-by-block eviction (which mattered under tight memory in our simulation), an optional cache-aware queue order in SGLang (with a starvation risk), and a cache-aware router. Everything else on the list exists in both, under different names.

For our handbook assistant, the useful question is: **which configuration answers our actual requests within the target time using less hardware?** Test at least these three cases:

1. **One shared handbook, many questions.** This shows the value of retaining and reusing the prefix.
2. **Many unrelated handbooks, more than fit in memory.** This tests eviction and, for SGLang, whether `--schedule-policy lpm` helps or starves anyone.
3. **Long generated answers.** This shows whether decoding dominates once input work is reduced.

Use the same model snapshot, hardware, request trace, and generation settings. Record time to first token, time to finish, completed requests per second, errors, answer quality and the engines' own cache-hit metrics. Test both a fresh cache and a deliberately warmed one, and report the slowest requests, not only the average. There is no universal winner independent of workload.

## 15. Try it yourself

The [companion code](https://github.com/ishwar6/ishwar-books/tree/main/code/sglang) contains every script, saved output and figure source used above.

### The CPU examples and simulations

From the repository root:

```bash
python3 -m venv /tmp/sglang-article-env
/tmp/sglang-article-env/bin/pip install -r code/sglang/requirements.txt
/tmp/sglang-article-env/bin/python code/sglang/walkthrough.py     # three-request trace, attention example
/tmp/sglang-article-env/bin/python code/sglang/experiments.py     # 64-request workload, attention equivalence
/tmp/sglang-article-env/bin/python code/sglang/simulate.py        # sections 3 to 11 of this article
/tmp/sglang-article-env/bin/python -m unittest discover -s code/sglang -p 'test_*.py'
```

`simulate.py` needs the saved timings in `results/measure_prefix.json` and the Qwen2.5 tokenizer (for the JSON example). The unit tests check both caches against a brute-force longest-prefix search, the leaf-first eviction rule, reference-count locking, and block eviction order.

### The real prefill measurement

`measure_prefix.py` needs PyTorch and Transformers, and runs on Apple GPUs (`mps`), NVIDIA GPUs (`cuda`) or slowly on a CPU:

```bash
python code/sglang/measure_prefix.py
```

It prints the table shown in Section 4 and saves the hardware, versions and every timing to `results/measure_prefix.json`.

### Then try a real model server

On a compatible GPU machine, install each engine in its own environment using its [SGLang installation guide](https://docs.sglang.io/docs/get-started/install) or [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). Use the same downloaded model snapshot for both, and start **one** server at a time on the same GPU:

```bash
# SGLang (radix cache is on by default; add --schedule-policy lpm to test cache-aware order)
python -m sglang.launch_server \
  --model-path /models/qwen2.5-0.5b-instruct-snapshot \
  --served-model-name comparison-model \
  --host 127.0.0.1 --port 30000
```

```bash
# vLLM (prefix caching is on by default in current versions)
vllm serve /models/qwen2.5-0.5b-instruct-snapshot \
  --served-model-name comparison-model \
  --host 127.0.0.1 --port 8000
```

Check the installed version's `--help` for available flags. The official request guides cover [SGLang](https://docs.sglang.io/docs/basic_usage/send_request) and [vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/).

Then send two different questions after the same handbook-like prefix:

```bash
python3 code/sglang/probe_server.py \
  --url http://127.0.0.1:30000 --model comparison-model \
  --output /tmp/sglang-probe.json
```

Use port 8000 and a different output file for vLLM. The probe saves the answers, time to first nonempty text, total streaming time, and token usage when the server supplies it.

This checks that requests work. Two requests cannot establish a throughput winner, and a faster second response alone does not prove a cache hit; read the engine's cache metrics. **The GPU commands are a reproduction recipe and were not run for this article**, because no NVIDIA GPU was available. The [README](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/README.md) records what was run and the limits of each experiment.

## The whole part, on one page

| Question | Answer | Where the number comes from |
|---|---|---|
| What is saved? | The prefix's keys and values, not the answer | Causal attention; exact match on a real model |
| How big is it? | $$2LH_{\mathrm{KV}}d_hb$$ per token: 128 KiB on Llama 3.1 8B, 750 MiB per 6,000-token handbook | Arithmetic from model configs |
| How much prefill does it save? | 30x fewer FLOPs, 7.6x less time at $$P=2048$$, $$S=64$$ | Measured, Qwen2.5-0.5B on an M5 Pro |
| Does the whole request get faster? | 7.6x for a 1-token answer, 1.03x for 200 tokens | Measured prefill and decode, combined |
| How much memory does sharing save? | $$\sum_r|r|$$ down to $$\sum_e|e|$$: 7.8x for 32 handbook requests | Arithmetic, checked by simulation |
| Radix tree or hashed blocks? | Same reuse with ample memory; under pressure, eviction granularity matters more | Simulation, one trace |
| Does request order matter? | LPM reached the 8,704-token optimum (87.1% hits) against 23.5% for FCFS, but one request waited 5.3x longer | Simulation with a measured cost model |
| What did the paper report? | Up to 6.4x throughput against vLLM v0.2.5 (2023); 74.1% hits and 1.7x faster first token in Chatbot Arena | Reported by the authors |

Return to the three questions at the start. Both SGLang and vLLM can keep the handbook's working state, find it again, and spend their effort only on each new question. SGLang finds it through a radix tree and offers a cache-aware order; vLLM finds it through chained block hashes. The handbook is processed once either way. What decides how fast your assistant feels is how long the shared part is, how long the answers are, whether the handbook is still in memory when the next question comes, and who has to wait while others go first.

<details>
<summary>Research notes: the paper's scheduling algorithm, and a few details</summary>

> [!PAPER] Zheng et al., SGLang · Algorithm 1 · page 15
> [![Algorithm 1 of the SGLang paper, cache-aware scheduling for RadixAttention with continuous batching: get all waiting requests; match each request's prefix in the radix tree; sort the requests by matched prefix length; admit requests while their size fits the evictable plus available memory, increasing reference counters of their prefix nodes; merge them into the running batch; allocate memory, evicting from the tree if needed; run the batch; for finished requests decrease reference counters and insert them into the tree](/img/sglang/paper-alg1.png)](/img/sglang/paper-alg1.png)
>
> **Context:** Appendix A.2, the pseudocode behind Section 3's scheduling paragraph.
>
> **What it says:** matching, sorting (`requests.sort()`), admission against *evictable plus free* memory, reference-count increments, eviction on allocation failure, and insertion of finished requests back into the tree.
>
> **Why it matters:** admission counts evictable cached tokens as available memory. That is the "cached tokens and running requests share one pool" rule from Section 3 in code form: the cache gives way to running work.
>
> [Read Appendix A.2](https://arxiv.org/pdf/2312.07104v2#page=15)

- **Multi-GPU.** With tensor parallelism each GPU holds a shard of every KV tensor, and the tree is the same on every GPU, so no extra synchronisation is needed (paper, Section 3). For data parallelism the paper describes a router holding a meta-tree of the workers' trees (Appendix A.4); the v0.4 cache-aware router above follows the same idea with an approximate tree per worker.
- **Images.** For multimodal models, SGLang hashes each input image and uses the hash as the radix key for the image tokens, so the same image is reused (paper, Section 6.2). vLLM puts the image hash into the block's extra hash.
- **API models.** The paper's Section 5, "API speculative execution", is about black-box API models and does not apply to self-hosted serving, so it is left out here.
- **Source pins.** SGLang `b2cb24995d4adefb2628d73cbf9cf84dae896bc1` and vLLM `c23ca06b6d944579ad9d220297441236755b4744`, both 8 October 2026; the paper PDF is arXiv 2312.07104v2. Hashes are in `results/sources.json`.

</details>

## References

**Papers**

1. L. Zheng et al. [*SGLang: Efficient Execution of Structured Language Model Programs*](https://arxiv.org/abs/2312.07104). arXiv 2312.07104, version 2.
2. W. Kwon et al. [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180). SOSP 2023.

**Engineering blogs**

3. SGLang team. [*Fast and Expressive LLM Inference with RadixAttention and SGLang*](https://www.lmsys.org/blog/2024-01-17-sglang/). LMSYS blog, January 2024.
4. L. Yin, Y. Sheng, L. Zheng. [*Fast JSON Decoding for Local LLMs with Compressed Finite State Machine*](https://www.lmsys.org/blog/2024-02-05-compressed-fsm/). LMSYS blog, February 2024.
5. SGLang team. [*SGLang v0.4: Zero-Overhead Batch Scheduler, Cache-Aware Load Balancer, Faster Structured Outputs*](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/). LMSYS blog, December 2024.

**Documentation**

6. vLLM. [Design: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/design/prefix_caching/), [Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).
7. SGLang. [Radix Cache Eviction Policies](https://docs.sglang.io/docs/advanced_features/radix_eviction_policy), [Hyperparameter Tuning](https://docs.sglang.io/docs/advanced_features/hyperparameter_tuning), [Structured Outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs), [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design).
8. Model configurations: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B/blob/main/config.json), [Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct/blob/main/config.json).

**Source code checked (8 October 2026)**

9. SGLang `b2cb249`: [schedule policy flag](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/arg_groups/fields/schedule.py#L82), [eviction policy flag](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/arg_groups/fields/memory.py#L25), [LPM fallback](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/managers/schedule_policy.py#L271), [radix cache](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/mem_cache/radix_cache.py#L533), [jump-forward hook](https://github.com/sgl-project/sglang/blob/b2cb24995d4adefb2628d73cbf9cf84dae896bc1/python/sglang/srt/constrained/base_grammar_backend.py#L119).
10. vLLM `c23ca06`: [cache config](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/cache.py#L141), [scheduler config](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/scheduler.py#L159), [async scheduling default](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/config/vllm.py#L1615), [last-token rule](https://github.com/vllm-project/vllm/blob/c23ca06b6d944579ad9d220297441236755b4744/vllm/v1/core/kv_cache_manager.py#L295).

**Companion results**

11. [Prefill measurement](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/measure_prefix.json), [simulations](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/simulate.json), [small walkthrough](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/walkthrough_stdout.txt), [64-request experiment](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/experiments.json), [source pins](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/sources.json).
