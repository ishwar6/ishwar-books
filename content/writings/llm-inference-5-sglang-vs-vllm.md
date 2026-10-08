---
title: "SGLang Explained: How It Reuses Work, and How It Differs from vLLM"
description: "Follow a handbook assistant from its first request to a shared KV cache. Learn RadixAttention step by step, then compare SGLang with vLLM."
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

vLLM is another inference engine, and modern vLLM also reuses repeated prompt beginnings. The interesting difference is how the engines organise that work, and what happens when requests compete for memory and GPU time.

We will follow the handbook assistant through those decisions. First, see what happens to one request. Then add a second request, build a small cache, compare it with vLLM's approach, and work out when the saving actually makes the assistant faster. You do not need to know tree data structures beforehand.

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

<figure class="fig"><svg viewBox="0 0 760 345" role="img" aria-label="Instructions, handbook and question enter prefill, which builds KV and predicts the first output token. Decode continues the answer."><text class="t-tick" x="30" y="25" text-anchor="start">INPUT</text><rect class="box-1" x="20" y="40" width="720" height="52" rx="8"/><text class="t-note" x="380" y="73" text-anchor="middle">Instructions + handbook + question</text><line class="edge" x1="380" y1="92" x2="380" y2="118"/><rect class="box-2" x="100" y="118" width="560" height="60" rx="8"/><text class="t-note" x="380" y="144" text-anchor="middle">PREFILL: process the supplied text</text><text class="t-tick" x="380" y="165" text-anchor="middle">Build KV state and select the first answer token</text><line class="edge" x1="380" y1="178" x2="380" y2="204"/><rect class="box-1" x="100" y="204" width="560" height="60" rx="8"/><text class="t-note" x="380" y="230" text-anchor="middle">DECODE: continue the answer</text><text class="t-tick" x="380" y="251" text-anchor="middle">Read saved KV, generate the next token, repeat</text><line class="edge" x1="380" y1="264" x2="380" y2="291"/><text class="t-note" x="380" y="315" text-anchor="middle">OUTPUT: the answer streams back to the user</text></svg><figcaption>The same model does both jobs: process the supplied text, then continue it. Prefix reuse saves some of the first job; the new answer still has to be generated.</figcaption></figure>

SGLang can accept ordinary serving requests, so you can use it behind an existing assistant. Its original project also included a Python language for describing connected model calls, but that frontend is not required to use the serving engine. [Sources: SGLang project](https://docs.sglang.io/), [original paper](https://arxiv.org/abs/2312.07104).

For now, focus on the first stage. Most of our second request's input is exactly the same as the first request's. To avoid processing it again, we need to know what can be saved.

## 2. Save the model's working state, not the answer

As the model reads the handbook, each attention layer computes arrays called **keys** and **values**, usually shortened to **K** and **V**. Later token positions use them to read information from earlier positions.

> [!DEFINITION] KV cache
> The stored keys and values from token positions the model has already processed. Keeping them lets later positions use that earlier work without rebuilding it. A cache is simply storage kept for possible reuse.

Within one answer, this is already useful. When generating the next word, the model can reuse the saved state for the prompt and the answer so far. It does not need to rebuild that whole history at every step.

Now extend the idea **across requests**. After answering the annual-leave question, retain the handbook's KV state. When the carry-forward question arrives, reuse the handbook state and compute the new question's state.

The cached object is not “the answer to the annual-leave question.” It is the model's representation of the shared beginning. Different questions can use it to produce different answers.

> [!DEFINITION] Prefix
> A sequence starting at the very beginning. Here, the **shared prefix** is the identical run of token IDs at the start of two prompts. It is an exact match, not a judgement that two passages mean roughly the same thing.

### Why the later question does not change the earlier handbook

A standard causal language model reads in one direction: a token position can use itself and earlier positions, but cannot look at later positions.

In our prompt, the handbook comes **before** the question. While processing a handbook token, the model cannot look ahead at the question. Changing that later question therefore does not change the handbook's keys and values, provided the model, preceding tokens, and positional setup stay the same.

> [!PAPER] Zheng et al., SGLang · Section 3 · Page 4
> [![The SGLang paper states: KV cache computation depends only on prefix tokens.](/img/sglang/paper-prefix.png)](/img/sglang/paper-prefix.png)
>
> **Context:** the authors are explaining when two model calls can share previously computed keys and values.
>
> **What it says:** the KV state at a position is determined by the input up to that position. In our example, the later question cannot change the earlier handbook state.
>
> **Why it matters:** keep the shared instructions and handbook before the changing question. The second request can then reuse the first request's handbook state and process its new question. Putting different questions first would remove that shared beginning.
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

The companion code runs this example. It also checks two complete causal-attention layers: reusing an unchanged prefix agrees with recomputing the full sequence to floating-point precision; deliberately reusing a changed prefix does not. The [KV-cache article](llm-inference-2-kv-cache.md) develops the full mechanics.

## 3. How SGLang finds the saved beginning: RadixAttention

Our assistant may answer thousands of questions. Some requests share the whole handbook; others share only the instructions. Scanning every old prompt would be an awkward way to find reusable work.

A **prefix tree** groups sequences by their common beginning. A **radix tree** is a compact version: a stretch with no branch is stored as one edge rather than as a separate node for every token.

Let us build one. For this small example, six token IDs stand for our shared instructions and handbook:

```text
shared beginning: 10 11 12 13 14 15
first question:                     20 21
second question:                    30 31
```

These are invented IDs for seeing how the algorithm works, not a real tokenizer's encoding of the handbook.

**First request.** The tree is empty. Compute all eight positions and record the sequence with references to its saved KV.

**Second request.** Follow the stored sequence from the beginning. The first six IDs match. The next ID is 30 instead of 20, so split the path at that point. Reuse the six matching positions and compute the two new ones.

<figure class="fig"><svg viewBox="0 0 760 430" role="img" aria-label="First request stores eight tokens. The second shares token IDs 10 through 15 and splits into question branches 20 21 and 30 31."><text class="t-note" x="25" y="25" text-anchor="start">1. After the first request</text><rect class="box-1" x="90" y="48" width="580" height="50" rx="8"/><text class="t-note" x="380" y="79" text-anchor="middle">10 11 12 13 14 15 20 21</text><text class="t-tick" x="380" y="123" text-anchor="middle">One stored path: eight computed token positions</text><text class="t-note" x="25" y="177" text-anchor="start">2. The second request matches six IDs, then branches</text><rect class="box-1" x="210" y="198" width="340" height="52" rx="8"/><text class="t-note" x="380" y="230" text-anchor="middle">10 11 12 13 14 15</text><line class="edge" x1="380" y1="250" x2="190" y2="293"/><line class="edge" x1="380" y1="250" x2="570" y2="293"/><rect class="box-1" x="90" y="293" width="200" height="48" rx="8"/><text class="t-note" x="190" y="323" text-anchor="middle">20 21</text><rect class="box-2" x="470" y="293" width="200" height="48" rx="8"/><text class="t-note" x="570" y="323" text-anchor="middle">30 31</text><text class="t-tick" x="190" y="367" text-anchor="middle">First question: already stored</text><text class="t-tick" x="570" y="367" text-anchor="middle">Second question: compute now</text><text class="t-tick" x="380" y="405" text-anchor="middle">The shared path refers to one reusable set of KV states.</text></svg><figcaption>The branch appears exactly where the token sequences differ. Both questions refer to the same saved beginning, while their different continuations have separate state.</figcaption></figure>

The tree describes where to find the states. It is not the huge array of states itself: the index is maintained on the CPU, while the KV tensors can live in GPU memory. Splitting a tree edge changes the index; it does not require recalculating the shared prefix. [Source: SGLang paper, Section 3](https://arxiv.org/html/2312.07104v2#S3).

### Run the three-request example

Here is the actual use of the small radix-tree implementation in the companion code:

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

Running `python code/sglang/walkthrough.py` from the repository root gives this radix-cache output:

```text
request 1: reuse 0, compute 8
request 2: reuse 6, compute 2
request 3: reuse 7, compute 1
```

The third request is worth looking at. It shares the six-token beginning **and token 20** with the first request. Matching can continue into that branch, stopping only when 40 differs from 21. The reusable part is discovered from the tokens; it does not have to end at a boundary we called “the handbook.”

This example runs the lookup and insertion logic on the CPU. It counts token positions needing computation; it does not execute a model for those positions or measure SGLang's GPU speed. The full [`Radix` implementation](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/experiments.py) includes the edge splitting just illustrated.

### What if the cache fills up?

Saved work competes for limited memory. A running request's state must remain available, while unused state from finished requests can be removed to make room.

The original RadixAttention design tracks which nodes are in use and evicts unused leaves by recency. Removing an old question branch can preserve the handbook shared by newer questions. Current SGLang offers configurable eviction policies; our small teaching class leaves memory management out. [Sources: paper, Section 3](https://arxiv.org/html/2312.07104v2#S3), [SGLang cache source](https://github.com/sgl-project/sglang/blob/943621c3364de6948fc02ee13c55b4f1cdf97f3a/python/sglang/srt/mem_cache/radix_cache.py).

So a second handbook request can reuse the first request's work **if that work is still available**. A matching prompt alone does not guarantee a cache hit.

## 4. How vLLM handles the same two requests

vLLM also supports prefix reuse. Its documented approach identifies complete blocks of tokens using a **hash**: a compact fingerprint used to look up the cached block.

Crucially, the fingerprint includes the preceding prefix's identity. The same four tokens after two different beginnings cannot simply share a block. [Source: vLLM prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/).

Use four-token blocks to see the difference:

```text
First request:   [10 11 12 13] [14 15 20 21]
Second request:  [10 11 12 13] [14 15 30 31]
                 └ same block ┘ └ different ┘
```

Our exact-token radix example reuses six positions. The four-token block example reuses the first four: the second block differs, so it must be computed again.

A simplified block-key rule is:

$$
\text{block key}=H(\text{previous block key},\;\text{this block's tokens},\;\text{extra identity}).
$$

$$H$$ denotes the hash function. The previous key ties this block to everything before it. Extra identity can distinguish adapters or multimodal inputs. Actual implementations also need model-specific cache handling and suitable isolation between requests.

Run the same prompts through the companion `BlockHash(4)` class and the output is:

```text
request 1: reuse 0, compute 8
request 2: reuse 4, compute 4
request 3: reuse 4, compute 4
```

This does **not** mean SGLang always reuses individual tokens while vLLM always uses blocks of four. Four is our teaching choice. Current SGLang's radix implementation also rounds matches to the configured page size; backends and model architectures impose further constraints. [Source: SGLang's `RadixKey.match_at`](https://github.com/sgl-project/sglang/blob/943621c3364de6948fc02ee13c55b4f1cdf97f3a/python/sglang/srt/mem_cache/radix_cache.py).

### Where PagedAttention fits

[Part 3](llm-inference-3-vllm.md) introduced vLLM's PagedAttention. It is easy to mix up two different jobs:

| Job | Question it answers |
|---|---|
| **Paged KV storage** | Where in memory should this request's keys and values go? |
| **Prefix-cache lookup** | Have these token positions already been computed, and where is their saved state? |

A radix tree can point to KV stored in pages. Paging and radix lookup can therefore be used together. RadixAttention does not replace the attention equation or make paging unnecessary.

### When both approaches save exactly the same work

Now make the handbook example larger. We generate 64 synthetic requests with:

- 1,024 shared instruction tokens;
- one of four 256-token handbook sections;
- a unique 64-token question.

All requests finish sequentially, and the teaching cache is large enough to keep everything. Without reuse, there are:

$$
64\times(1024+256+64)=86{,}016
$$

prompt-token computations. With reuse, compute the common instructions once, each section once, and every question:

$$
1024+4\times256+64\times64=6{,}144.
$$

All shared lengths align with 16-token blocks. **Both the radix index and the 16-token block-hash index reach 6,144.**

<figure class="fig"><svg viewBox="0 0 760 255" role="img" aria-label="Both teaching cache indexes compute 6,144 tokens versus 86,016 without reuse."><text class="t-note" x="180" y="49" text-anchor="end">No reuse</text><rect class="box-2" x="200" y="25" width="450.0" height="36" rx="8"/><text class="t-note" x="660.0" y="49" text-anchor="start">86,016</text><text class="t-note" x="180" y="114" text-anchor="end">Radix index</text><rect class="box-1" x="200" y="90" width="32.142857142857146" height="36" rx="8"/><text class="t-note" x="242.14285714285714" y="114" text-anchor="start">6,144</text><text class="t-note" x="180" y="179" text-anchor="end">Block-hash index</text><rect class="box-1" x="200" y="155" width="32.142857142857146" height="36" rx="8"/><text class="t-note" x="242.14285714285714" y="179" text-anchor="start">6,144</text><text class="t-tick" x="380" y="235" text-anchor="middle">Prompt token positions computed across 64 requests</text></svg><figcaption>In this executed cache simulation, both indexes avoid 79,872 repeated token computations. Each bar shows how many token positions still need computation.</figcaption></figure>

The data structure matters, but shared workload, alignment, and available memory determine what can actually be reused. This is why “one uses a tree” is not enough to predict which engine will be faster.

## 5. What the handbook assistant actually gains

We have shown that the engine can avoid repeated work. Now ask what the user or operator gets from it.

### A shorter wait for the answer to start

If the handbook accounts for most of the prompt, reusing its KV can reduce prefill work substantially. But the engine must still process the new question. Those new positions still attend to the cached handbook; reusing the handbook does not make it disappear from attention.

The engine also has to generate the answer. Suppose, for illustration, a request spends 100 ms in prefill and 900 ms generating the answer. Prefix reuse reduces prefill to 20 ms:

| | Before reuse | After reuse |
|---|---:|---:|
| Prefill | 100 ms | 20 ms |
| Generate the answer | 900 ms | 900 ms |
| Total | 1,000 ms | 920 ms |

Prefill became five times faster, but the whole request became only:

$$
\frac{1000}{920}\approx1.09
$$

times faster. These are assumed times to explain the arithmetic, not measured server results.

For a long handbook and a short answer, prefill savings may be a large part of the total. For a short prompt and a long answer, generating the answer may dominate. Measure the wait for the first token separately from the time to finish. [Source: vLLM's prefix-caching limits](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

### Room for more concurrent requests

If many active requests share the same cached handbook state, they may reference one physical copy instead of storing duplicate copies.

For a conventional transformer, the unsharded KV payload per token is:

$$
\text{bytes per token}=2\times L\times H_{\mathrm{KV}}\times d_h\times b.
$$

Read the factors from left to right: two arrays (K and V), $$L$$ layers, $$H_{\mathrm{KV}}$$ KV heads per layer, $$d_h$$ numbers per head, and $$b$$ bytes per number.

For example, with 32 layers, eight KV heads, head width 128, and two bytes per number:

$$
2\times32\times8\times128\times2=131{,}072\;\text{bytes}=128\;\text{KiB}.
$$

An 8,192-token shared beginning then occupies **1 GiB**. Thirty-two independent copies would occupy 32 GiB; one shared copy occupies 1 GiB, plus each request's private question and answer state.

That calculation counts KV payload, not total GPU memory. Model weights and temporary working memory still take space. Other model architectures, quantized caches, and multi-GPU placement need different accounting. The useful lesson is simple: **sharing can save both computation and duplicate storage**.

## 6. What changes when the assistant gets busy?

So far, one request has finished before the next starts. A real service has requests arriving together. Prefix reuse is only one part of keeping that service responsive.

### Put related work near each other, without making people wait too long

Suppose there are four different handbooks, A through D, but the cache has room for only two. With requests arriving as `A B C D A B C D`, an old handbook may be evicted before it is needed again.

Our small capacity simulation makes that concrete. Each request needs a 1,024-token handbook and a fresh 64-token question:

| Order of 32 requests | Handbook misses | Handbook hits | Prompt tokens computed |
|---|---:|---:|---:|
| A, B, C, D, repeated | 32 | 0 | 34,816 |
| Eight A requests, then eight B, then C, then D | 4 | 28 | 6,144 |

The simulation keeps only two whole handbooks and evicts the least recently used one. It illustrates **locality**: doing related work close together makes saved state more likely to survive until its next use.

Grouping is possible if the requests are already waiting. It is not free: delaying one person's request to help other requests reuse a cache can increase that person's wait. SGLang's research considers cache-aware scheduling; the practical goal is to save work while still meeting users' latency targets. [Source: SGLang paper, Section 3](https://arxiv.org/html/2312.07104v2#S3).

If there are multiple server replicas, placement matters too. A request sent to a worker that already holds its handbook can avoid rebuilding it. SGLang's cache-aware routing work addresses this problem, balancing reuse against load. A warm worker with a long queue can still lose to a cold worker that is idle. [Source: SGLang team's scheduler and router report](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/).

### Prepare the next batch while the GPU runs the current one

Even when KV reuse works, the GPU can sit idle while the CPU prepares the next batch's metadata and memory mappings.

SGLang's overlap scheduler lets that preparation happen while the current GPU computation is still running. Imagine CPU preparation takes 2 ms and GPU execution takes 8 ms. In a serial schedule, each batch costs 10 ms. With ideal overlap, batches after startup can complete every 8 ms because preparation fits inside the device's working time.

<figure class="fig"><svg viewBox="0 0 760 285" role="img" aria-label="CPU scheduling overlaps GPU execution. Ideal steady state changes from 10 milliseconds to 8 milliseconds per batch."><text class="t-note" x="15" y="18" text-anchor="start">Serial</text><text class="t-tick" x="15" y="43" text-anchor="start">CPU</text><rect class="box-2" x="110" y="25" width="20" height="25" rx="8"/><rect class="box-1" x="130" y="57" width="80" height="25" rx="8"/><rect class="box-2" x="210" y="25" width="20" height="25" rx="8"/><rect class="box-1" x="230" y="57" width="80" height="25" rx="8"/><rect class="box-2" x="310" y="25" width="20" height="25" rx="8"/><rect class="box-1" x="330" y="57" width="80" height="25" rx="8"/><rect class="box-2" x="410" y="25" width="20" height="25" rx="8"/><rect class="box-1" x="430" y="57" width="80" height="25" rx="8"/><rect class="box-2" x="510" y="25" width="20" height="25" rx="8"/><rect class="box-1" x="530" y="57" width="80" height="25" rx="8"/><text class="t-tick" x="15" y="73" text-anchor="start">GPU</text><text class="t-note" x="15" y="123" text-anchor="start">Overlap</text><text class="t-tick" x="15" y="148" text-anchor="start">CPU</text><rect class="box-2" x="110" y="130" width="20" height="25" rx="8"/><rect class="box-1" x="130" y="162" width="80" height="25" rx="8"/><rect class="box-2" x="190" y="130" width="20" height="25" rx="8"/><rect class="box-1" x="210" y="162" width="80" height="25" rx="8"/><rect class="box-2" x="270" y="130" width="20" height="25" rx="8"/><rect class="box-1" x="290" y="162" width="80" height="25" rx="8"/><rect class="box-2" x="350" y="130" width="20" height="25" rx="8"/><rect class="box-1" x="370" y="162" width="80" height="25" rx="8"/><rect class="box-2" x="430" y="130" width="20" height="25" rx="8"/><rect class="box-1" x="450" y="162" width="80" height="25" rx="8"/><text class="t-tick" x="15" y="178" text-anchor="start">GPU</text><text class="t-tick" x="380" y="240" text-anchor="middle">Illustrative: CPU preparation 2 ms, GPU work 8 ms; each unit is 1 ms.</text><text class="t-tick" x="380" y="263" text-anchor="middle">After startup, the CPU prepares the next batch while the GPU runs.</text></svg><figcaption>Orange is CPU preparation; blue is GPU execution. In the lower timeline, preparation for the next batch overlaps the current batch's GPU work. The durations are illustrative.</figcaption></figure>

This is a different saving from prefix reuse: it reduces idle time rather than repeated prompt computation. Modern vLLM also supports asynchronous scheduling; it is not a feature exclusive to SGLang. [Sources: SGLang overlap explanation](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), [vLLM scheduler source](https://github.com/vllm-project/vllm/blob/458ba2edf85bc9b7ebc0d5141f34ea658520faf5/vllm/config/scheduler.py).

### Keep more handbooks outside GPU memory

SGLang's **HiCache** can extend saved KV into host memory and an optional storage backend. That can retain more handbooks, but a host-memory hit still requires moving the state back before the GPU can use it. [Source: HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design).

For our 1 GiB example, an assumed effective transfer rate of 25 GB/s gives about **43 ms** just to move the data. If recomputing that prefix takes 20 ms, a blocking 43 ms transfer loses. If recomputation takes 200 ms, the transfer may be worthwhile. Lookup, allocation, and contention add costs; overlap may hide some of them.

The decision is whether retrieving the saved work is cheaper than doing it again. vLLM has KV offload and transfer integrations too, including [LMCache](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/). Compare the configuration that fits your model and hardware, not just whether “offload” appears on a feature list.

## 7. So how does SGLang differ from vLLM?

We can now compare the engines in terms of the problems we have actually seen:

| Problem | SGLang | vLLM |
|---|---|---|
| Find the handbook already processed | Radix-tree cache family | Hash-linked full-block prefix cache |
| Fit requests into memory and run them together | KV allocation and request batching | Paged KV management and request batching |
| Reduce GPU idle time between batches | Overlap scheduling | Asynchronous scheduling support |
| Retain reusable KV beyond GPU memory | HiCache | Offload/transfer integrations such as LMCache |
| Produce output that follows a schema | Structured-output backends | Structured-output backends |

The last row addresses a separate requirement. If our assistant must return `{"answer": "...", "policy_section": "..."}`, a grammar can constrain which tokens are allowed next. Both engines offer this. Valid JSON still needs correct facts, so check those separately. [Sources: SGLang structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs), [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).

Both also support other serving optimizations; their exact coverage depends on the release, model, backend, and hardware. The comparison above is about the mechanisms discussed here, not every feature either project offers. Source revisions are recorded in the companion directory, checked on 8 October 2026.

For our handbook assistant, the useful question is: **which configuration answers our actual requests within the target time using less hardware?** Test at least these three cases:

1. **One shared handbook, many questions.** This reveals the value of retaining and reusing the prefix.
2. **Many unrelated handbooks.** This tests what happens when little is shared or the cache is too small.
3. **Long generated answers.** This shows whether generation time dominates after input work is reduced.

Use the same model snapshot, hardware, request trace, and generation settings. Record time to first token, time to finish, completed requests per second, errors, and answer quality. Test both a fresh cache and a deliberately warmed one. Report slow requests too, not only the average.

A server that produces more tokens per second but makes people wait much longer for the first one may be a poor trade for this assistant. A batch-processing job may accept that trade. There is no universal winner independent of workload.

## 8. Try it yourself

The [companion code](https://github.com/ishwar6/ishwar-books/tree/main/code/sglang) contains the complete cache implementations and saved outputs.

### Start with the small example in this article

From the repository root:

```bash
python3 -m venv /tmp/sglang-article-env
/tmp/sglang-article-env/bin/pip install -r code/sglang/requirements.txt
/tmp/sglang-article-env/bin/python code/sglang/walkthrough.py
```

It prints the three-request radix and block-cache results, followed by the two attention calculations. Then run the larger experiment:

```bash
/tmp/sglang-article-env/bin/python code/sglang/experiments.py
```

That reproduces the 64-prompt comparison, the cache-capacity simulation, and the attention-equivalence check. These examples were executed on the CPU. Their token counts explain the mechanisms; they are not GPU engine benchmarks.

### Then try a real model server

On a compatible GPU machine, install either engine in its own environment using its [SGLang installation guide](https://docs.sglang.io/docs/get-started/install) or [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/). Use the same downloaded model snapshot for both.

Start **one** of these servers at a time on the same GPU, replacing the model path:

```bash
# SGLang
python -m sglang.launch_server \
  --model-path /models/qwen2.5-0.5b-instruct-snapshot \
  --served-model-name comparison-model \
  --host 127.0.0.1 --port 30000
```

```bash
# vLLM
vllm serve /models/qwen2.5-0.5b-instruct-snapshot \
  --served-model-name comparison-model \
  --enable-prefix-caching \
  --host 127.0.0.1 --port 8000
```

Check the installed version's `--help` for available flags. The official request guides cover [SGLang](https://docs.sglang.io/docs/basic_usage/send_request) and [vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/).

Now send two different questions after the same handbook-like prefix:

```bash
python3 code/sglang/probe_server.py \
  --url http://127.0.0.1:30000 --model comparison-model \
  --output /tmp/sglang-probe.json
```

Use port 8000 and a different output filename for vLLM. The probe saves the answers, time to first nonempty text, total streaming time, and token usage when supplied by the server.

This checks that requests work. Two requests cannot establish a throughput winner, and a faster second response alone does not prove a cache hit; inspect the engine's cache metrics. The GPU commands are a reproduction recipe and were not run for this article. The [README](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/README.md) records what was run and the limits of each experiment.

Return to the three questions at the start. SGLang can retain the common handbook's working state, find it through the radix cache, and use it while processing each new question. vLLM can avoid much of the same repeated work using a different index. Once you understand that shared task, the comparison becomes concrete: how much work was reused, how much memory it occupied, and how long the user waited for a correct answer.

## Sources and experiment details

- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), especially Section 3 on RadixAttention.
- SGLang team, [scheduler overlap and cache-aware routing](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), December 2024.
- vLLM, [prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/) and [where prefix caching helps](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).
- [Small walkthrough output](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/walkthrough_stdout.txt), [larger experiment results](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/experiments.json), and [pinned source revisions](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/sources.json).
