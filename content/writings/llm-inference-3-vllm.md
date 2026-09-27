---
title: "How vLLM Serves Thousands: Paging, Batching and Caching"
description: "PagedAttention, continuous batching, chunked prefill and prefix caching, each built from the problem it solves and measured or simulated with real costs: from 48 to 415 requests on one GPU, pauses cut from 463 ms to 40 ms, and a 7x faster first token."
date: 2026-09-28
tags: [inference, vllm, llm]
series: "LLM Inference from the Ground Up"
series_part: 3
motif: blocks
accent: "#c4a1ff"
---

Part 1 ended with a rule: to make a GPU productive, **decode many sequences at once**. Part 2 ended with the catch: every sequence carries a **KV cache** that grows unpredictably, and the obvious way of storing it wasted most of the memory. Older serving systems used only 20.4% to 38.2% of their KV memory for real data.

In 2023 a team at UC Berkeley looked at this and noticed something familiar. A program that needs memory it cannot predict, a limited physical memory, lots of programs sharing it: that is exactly the problem operating systems solved in the 1960s. Their fix was called **paging**. The Berkeley team applied it to the KV cache, called it **PagedAttention**, and built a serving engine around it: **vLLM**.

This part walks through vLLM's four big ideas, each one built from the problem it solves:

1. **PagedAttention**: store the cache in small blocks, so almost nothing is wasted.
2. **Continuous batching**: let requests join and leave the batch at every step.
3. **Chunked prefill**: keep long prompts from freezing everyone else.
4. **Prefix caching**: never compute the same prompt twice.

## Idea 1: store the cache in pages

Your laptop runs dozens of programs, and none of them knows where in physical memory its data really lives. Each program sees a neat, continuous address space. Behind the scenes the operating system chops that space into small fixed-size **pages** and puts each page wherever there is room. A **page table** records where each one went. A program that needs more memory simply gets another page; nobody has to reserve the maximum up front.

vLLM does the same with the KV cache. The cache is cut into **blocks** of 16 tokens each (vLLM's default block size). A request's cache is a list of blocks, in order, but the blocks themselves can sit anywhere in GPU memory. A **block table** records the mapping.

<figure class="fig"><svg viewBox="0 0 760 302" role="img" aria-label="PagedAttention: each request has logical blocks that map through a block table to physical blocks scattered anywhere in GPU memory."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="20" y="24" text-anchor="start">logical blocks: what each request sees, in order</text><text class="t-strong" x="40" y="58" text-anchor="start">request A</text><rect class="box-1" x="40" y="70" width="88" height="34" rx="6"/><text class="t-note" x="84" y="92" text-anchor="middle">block 0</text><rect class="box-1" x="140" y="70" width="88" height="34" rx="6"/><text class="t-note" x="184" y="92" text-anchor="middle">block 1</text><rect class="box-1" x="240" y="70" width="88" height="34" rx="6"/><text class="t-note" x="284" y="92" text-anchor="middle">block 2</text><text class="t-strong" x="450" y="58" text-anchor="start">request B</text><rect class="box-2" x="450" y="70" width="88" height="34" rx="6"/><text class="t-note" x="494" y="92" text-anchor="middle">block 0</text><rect class="box-2" x="550" y="70" width="88" height="34" rx="6"/><text class="t-note" x="594" y="92" text-anchor="middle">block 1</text><text class="t-tick" x="20" y="190" text-anchor="start">physical blocks: where the K and V really live in GPU memory</text><rect class="box" x="40" y="208" width="72" height="36" rx="6"/><text class="t-tick" x="76" y="231" text-anchor="middle">free</text><text class="t-tick" x="76" y="262" text-anchor="middle">#0</text><line class="edge" x1="184" y1="106" x2="164" y2="204" marker-end="url(#ah)"/><rect class="box-1" x="128" y="208" width="72" height="36" rx="6"/><text class="t-strong" x="164" y="231" text-anchor="middle">A1</text><text class="t-tick" x="164" y="262" text-anchor="middle">#1</text><line class="edge" x1="494" y1="106" x2="252" y2="204" marker-end="url(#ah)"/><rect class="box-2" x="216" y="208" width="72" height="36" rx="6"/><text class="t-strong" x="252" y="231" text-anchor="middle">B0</text><text class="t-tick" x="252" y="262" text-anchor="middle">#2</text><rect class="box" x="304" y="208" width="72" height="36" rx="6"/><text class="t-tick" x="340" y="231" text-anchor="middle">free</text><text class="t-tick" x="340" y="262" text-anchor="middle">#3</text><line class="edge" x1="284" y1="106" x2="428" y2="204" marker-end="url(#ah)"/><rect class="box-1" x="392" y="208" width="72" height="36" rx="6"/><text class="t-strong" x="428" y="231" text-anchor="middle">A2</text><text class="t-tick" x="428" y="262" text-anchor="middle">#4</text><rect class="box" x="480" y="208" width="72" height="36" rx="6"/><text class="t-tick" x="516" y="231" text-anchor="middle">free</text><text class="t-tick" x="516" y="262" text-anchor="middle">#5</text><line class="edge" x1="594" y1="106" x2="604" y2="204" marker-end="url(#ah)"/><rect class="box-2" x="568" y="208" width="72" height="36" rx="6"/><text class="t-strong" x="604" y="231" text-anchor="middle">B1</text><text class="t-tick" x="604" y="262" text-anchor="middle">#6</text><line class="edge" x1="84" y1="106" x2="692" y2="204" marker-end="url(#ah)"/><rect class="box-1" x="656" y="208" width="72" height="36" rx="6"/><text class="t-strong" x="692" y="231" text-anchor="middle">A0</text><text class="t-tick" x="692" y="262" text-anchor="middle">#7</text><text class="t-tick" x="20" y="290" text-anchor="start">Each block holds K and V for 16 tokens. A request's block table records which physical block holds each logical one.</text></svg><figcaption>PagedAttention. Each request sees its cache as a neat sequence of blocks. Physically, the blocks are scattered wherever there was room, and a block table records which is which.</figcaption></figure>

Now look at what happens to the three kinds of waste from Part 2:

- **Reserved slots** disappear: a request gets a new block only when its last block fills up.
- **Internal fragmentation** shrinks to at most one partly filled block per request, fewer than 16 tokens.
- **External fragmentation** disappears: every block is the same size, so any free block fits any request.

The attention computation has to change a little, because a sequence's keys and values are no longer in one continuous stretch of memory. The PagedAttention kernel reads them block by block, following the block table. That indirection is the price of the idea: the paper measured the attention step itself at 20 to 26% slower than a highly optimised contiguous kernel. The authors judged it small because it only affects attention, not the rest of the model, and the memory it frees buys far more. Why 16 tokens per block? Big enough to keep the GPU busy, small enough that the last, partly filled block wastes little; the paper found 16 the best balance "in most workloads."

### How much memory does it save?

To see the effect, I simulated 200,000 requests on a single 80 GB GPU serving Llama 3.1 8B (128 KiB of cache per token, from Part 2). Prompt and answer lengths were drawn from wide, realistic spreads, and each request was caught at a random moment in its life. The question: of the memory each policy sets aside, how much actually holds tokens, and how many requests fit at once?

<figure class="fig"><svg viewBox="0 0 760 250" role="img" aria-label="Share of reserved KV memory that holds real tokens, and how many requests fit, by allocation policy."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><g class="mark"><title>reserve max length (8,192): 11.6%</title><rect class="hit-area" x="0" y="16.0" width="760" height="54.5"/><path class="s2 bar-mark" d="M230,31.25H274.61625533447267Q278.61625533447267,31.25 278.61625533447267,35.25V51.25Q278.61625533447267,55.25 274.61625533447267,55.25H230Z"/></g><text class="t-note" x="218" y="48.25" text-anchor="end">reserve max length (8,192)</text><text class="t-val" x="286.61625533447267" y="48.25" text-anchor="start">11.6%</text><text class="t-tick" x="286.61625533447267" y="65.25" text-anchor="start">48 requests fit</text><g class="mark"><title>reserve prompt + max_tokens: 33.7%</title><rect class="hit-area" x="0" y="70.5" width="760" height="54.5"/><path class="s2 bar-mark" d="M230,85.75H367.7328939744799Q371.7328939744799,85.75 371.7328939744799,89.75V105.75Q371.7328939744799,109.75 367.7328939744799,109.75H230Z"/></g><text class="t-note" x="218" y="102.75" text-anchor="end">reserve prompt + max_tokens</text><text class="t-val" x="379.7328939744799" y="102.75" text-anchor="start">33.7%</text><text class="t-tick" x="379.7328939744799" y="119.75" text-anchor="start">141 requests fit</text><g class="mark"><title>reserve exact length (oracle): 83.5%</title><rect class="hit-area" x="0" y="125.0" width="760" height="54.5"/><path class="s2 bar-mark" d="M230,140.25H576.5421434489501Q580.5421434489501,140.25 580.5421434489501,144.25V160.25Q580.5421434489501,164.25 576.5421434489501,164.25H230Z"/></g><text class="t-note" x="218" y="157.25" text-anchor="end">reserve exact length (oracle)</text><text class="t-val" x="588.5421434489501" y="157.25" text-anchor="start">83.5%</text><text class="t-tick" x="588.5421434489501" y="174.25" text-anchor="start">349 requests fit</text><g class="mark"><title>paged, 16-token blocks: 99.2%</title><rect class="hit-area" x="0" y="179.5" width="760" height="54.5"/><path class="s1 bar-mark" d="M230,194.75H642.7025864071361Q646.7025864071361,194.75 646.7025864071361,198.75V214.75Q646.7025864071361,218.75 642.7025864071361,218.75H230Z"/></g><text class="t-note" x="218" y="211.75" text-anchor="end">paged, 16-token blocks</text><text class="t-val" x="654.7025864071361" y="211.75" text-anchor="start">99.2%</text><text class="t-tick" x="654.7025864071361" y="228.75" text-anchor="start">415 requests fit</text><line class="axis" x1="230" y1="16" x2="230" y2="234"/></svg><figcaption>Share of KV memory that holds real tokens under four allocation policies, and how many concurrent requests fit in the same memory. Simulated with Llama 3.1 8B's cache size on an 80 GB GPU.</figcaption></figure>

| Allocation policy | Memory holding real tokens | Requests that fit |
|---|---|---|
| reserve the model's maximum length (8,192 tokens) | 11.6% | 48 |
| reserve prompt + the request's `max_tokens` | 33.7% | 141 |
| reserve the exact final length (impossible: needs the future) | 83.5% | 349 |
| **paged, 16-token blocks** | **99.2%** | **415** |

Paging does better than even the "oracle" that magically knows every answer's length, because the oracle still reserves space for tokens that have not been written yet. My simulation leaves out external fragmentation, which only makes the contiguous policies look better than they are.

The real system agrees. The vLLM paper measured existing systems at 20.4% to 38.2% useful KV memory and vLLM at **96.3%**, and reported **2 to 4 times** the throughput of the leading systems of the time "with the same level of latency."

## Sharing blocks for free

Blocks bring a bonus: two requests can point at **the same physical block**. Many requests share a beginning. Think of a chatbot's long system prompt, or one prompt sampled several times to get several answers. With paging, those shared tokens are stored once.

<figure class="fig"><svg viewBox="0 0 760 302" role="img" aria-label="Two requests share the physical blocks of an identical prompt prefix; only their own blocks differ."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="20" y="24" text-anchor="start">two requests that start with the same system prompt</text><text class="t-strong" x="30" y="58" text-anchor="start">request A</text><rect class="box-3" x="30" y="70" width="88" height="34" rx="6"/><text class="t-note" x="74" y="92" text-anchor="middle">system</text><rect class="box-3" x="130" y="70" width="88" height="34" rx="6"/><text class="t-note" x="174" y="92" text-anchor="middle">system</text><rect class="box-1" x="230" y="70" width="88" height="34" rx="6"/><text class="t-note" x="274" y="92" text-anchor="middle">question</text><text class="t-strong" x="420" y="58" text-anchor="start">request B</text><rect class="box-3" x="420" y="70" width="88" height="34" rx="6"/><text class="t-note" x="464" y="92" text-anchor="middle">system</text><rect class="box-3" x="520" y="70" width="88" height="34" rx="6"/><text class="t-note" x="564" y="92" text-anchor="middle">system</text><rect class="box-2" x="620" y="70" width="88" height="34" rx="6"/><text class="t-note" x="664" y="92" text-anchor="middle">question</text><rect class="box-3" x="130" y="208" width="110" height="36" rx="6"/><text class="t-strong" x="185" y="231" text-anchor="middle">#3</text><text class="t-tick" x="185" y="262" text-anchor="middle">shared · refs 2</text><rect class="box-3" x="280" y="208" width="110" height="36" rx="6"/><text class="t-strong" x="335" y="231" text-anchor="middle">#5</text><text class="t-tick" x="335" y="262" text-anchor="middle">shared · refs 2</text><rect class="box-1" x="430" y="208" width="110" height="36" rx="6"/><text class="t-strong" x="485" y="231" text-anchor="middle">#8</text><text class="t-tick" x="485" y="262" text-anchor="middle">A only</text><rect class="box-2" x="580" y="208" width="110" height="36" rx="6"/><text class="t-strong" x="635" y="231" text-anchor="middle">#9</text><text class="t-tick" x="635" y="262" text-anchor="middle">B only</text><line class="edge" x1="74" y1="106" x2="185" y2="204" marker-end="url(#ah)"/><line class="edge" x1="174" y1="106" x2="335" y2="204" marker-end="url(#ah)"/><line class="edge" x1="274" y1="106" x2="485" y2="204" marker-end="url(#ah)"/><line class="edge" x1="464" y1="106" x2="185" y2="204" marker-end="url(#ah)"/><line class="edge" x1="564" y1="106" x2="335" y2="204" marker-end="url(#ah)"/><line class="edge" x1="664" y1="106" x2="635" y2="204" marker-end="url(#ah)"/><text class="t-tick" x="20" y="290" text-anchor="start">The shared prompt is stored once. A block is copied only when one request must change it (copy-on-write).</text></svg><figcaption>Two requests with the same system prompt point at the same physical blocks. Each block keeps a count of how many requests use it. Only a block that one request needs to change gets copied.</figcaption></figure>

Each block carries a **reference count**. When a request needs to write into a block that others share, vLLM copies that one block first and then writes (**copy-on-write**, the same trick an operating system uses when a process forks). Everything else stays shared.

## Idea 2: prefix caching

Sharing within the requests running right now is good. Sharing **across time** is better. If a thousand users send requests that start with the same 2,000-token system prompt, why compute its keys and values a thousand times?

vLLM's **automatic prefix caching** keeps finished blocks around after a request ends. Each full 16-token block gets a hash built from its tokens and the hash of the block before it, so a hash identifies not just 16 tokens but the entire prefix leading up to them. When a new request arrives, vLLM looks up its blocks by hash, reuses every one it finds, and only runs prefill for the rest.

I measured the effect directly: a 2,048-token shared system prompt followed by a 64-token question.

| | Time to first token |
|---|---|
| full prefill of prompt + question | 98.3 ms |
| system prompt reused from cache, prefill only the question | **14.2 ms** |

That is **7 times faster**, and the next token it predicts is identical: the largest difference in any output score between the two runs was exactly 0.0. The saved prefill is not an approximation; it is the same arithmetic, done once instead of twice.

## Idea 3: continuous batching

Batching (Part 1) is only useful if the batch stays full. The simple way to batch is **static**: gather 64 requests, run them together until they are all finished, then take the next 64. The problem is that answers have wildly different lengths. A request that finishes after 20 tokens leaves its slot empty while the batch waits for the one that runs to 2,000.

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="Static batching leaves slots idle until the slowest request finishes; continuous batching refills a slot as soon as it frees up."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="20" y="24" text-anchor="start">static batching: a batch of 4 runs until its longest request ends</text><rect class="s1" x="120" y="36" width="151" height="16" rx="3"/><rect class="s2" x="120" y="58" width="49" height="16" rx="3"/><rect class="waste" x="171" y="58" width="100" height="16" rx="3"/><rect class="s3" x="120" y="80" width="100" height="16" rx="3"/><rect class="waste" x="222" y="80" width="49" height="16" rx="3"/><rect class="s1" x="120" y="102" width="32" height="16" rx="3"/><rect class="waste" x="154" y="102" width="117" height="16" rx="3"/><rect class="s2" x="273" y="36" width="134" height="16" rx="3"/><rect class="s3" x="273" y="58" width="66" height="16" rx="3"/><rect class="waste" x="341" y="58" width="66" height="16" rx="3"/><rect class="s1" x="273" y="80" width="83" height="16" rx="3"/><rect class="waste" x="358" y="80" width="49" height="16" rx="3"/><rect class="s2" x="273" y="102" width="49" height="16" rx="3"/><rect class="waste" x="324" y="102" width="83" height="16" rx="3"/><text class="t-tick" x="20" y="49" text-anchor="start">slot 1</text><text class="t-tick" x="20" y="71" text-anchor="start">slot 2</text><text class="t-tick" x="20" y="93" text-anchor="start">slot 3</text><text class="t-tick" x="20" y="115" text-anchor="start">slot 4</text><text class="t-note" x="20" y="158" text-anchor="start">continuous batching: a finished slot takes the next request at once</text><rect class="s1" x="120" y="170" width="151" height="16" rx="3"/><rect class="s2" x="120" y="192" width="49" height="16" rx="3"/><rect class="s3" x="120" y="214" width="100" height="16" rx="3"/><rect class="s1" x="120" y="236" width="32" height="16" rx="3"/><rect class="s2" x="154" y="236" width="134" height="16" rx="3"/><rect class="s3" x="171" y="192" width="66" height="16" rx="3"/><rect class="s1" x="222" y="214" width="83" height="16" rx="3"/><rect class="s2" x="239" y="192" width="49" height="16" rx="3"/><text class="t-tick" x="20" y="183" text-anchor="start">slot 1</text><text class="t-tick" x="20" y="205" text-anchor="start">slot 2</text><text class="t-tick" x="20" y="227" text-anchor="start">slot 3</text><text class="t-tick" x="20" y="249" text-anchor="start">slot 4</text><line class="drop" x1="409" y1="30" x2="409" y2="260"/><line class="drop" x1="307" y1="164" x2="307" y2="260"/><text class="t-tick" x="313" y="274" text-anchor="start">same 8 requests done in 11 steps instead of 17</text><rect class="waste" x="20" y="288" width="18" height="12" rx="3"/><text class="t-tick" x="44" y="298" text-anchor="start">slot idle, waiting for the slowest request in its batch</text></svg><figcaption>Static batching keeps a slot idle until the slowest request in its batch finishes. Continuous batching hands a freed slot to the next waiting request immediately.</figcaption></figure>

**Continuous batching**, introduced by the Orca system (Yu et al., OSDI 2022) as *iteration-level scheduling*, decides who is in the batch **at every single step**. The moment a request finishes, its slot goes to the next one in line.

I simulated 4,096 requests with realistic answer lengths through 64 slots:

| | Steps to finish all requests | Slots doing useful work |
|---|---|---|
| static batching | 114,566 | 21% |
| continuous batching | 25,127 | 96% |

Same requests, same slots, **4.6 times fewer steps**. The exact ratio depends on how uneven the answer lengths are, but the direction never changes. Orca reported a **36.9x** throughput improvement over NVIDIA FasterTransformer on GPT-3 175B "at the same level of latency."

Continuous batching and paging need each other. Requests joining and leaving at every step would shred a contiguous memory layout; with fixed-size blocks, a finished request just hands its blocks back to the free pool.

In vLLM's current engine (called V1, the default since version 0.8 in 2025), the batch for each step is flattened into one long "super sequence": every scheduled token from every request laid end to end, with position indices and attention masks keeping each request's tokens to themselves. No padding, no wasted slots.

## Idea 4: chunked prefill

There is one more collision to deal with. Prefill and decode have opposite personalities (Part 1). Decode steps are short and frequent. A prefill for a long prompt is one huge step. If a user sends an 8,192-token document while 16 other users are streaming answers, what happens?

Without any special handling, that prefill runs as one step, and **every streaming user freezes** until it is done. From their point of view the text just stops.

<figure class="fig"><svg viewBox="0 0 760 226" role="img" aria-label="Chunked prefill splits a long prompt into slices that ride along with ongoing decodes, so no step stalls the streaming users for long."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="20" y="24" text-anchor="start">without chunking: the 8,192-token prompt runs in one step; 16 streaming users wait</text><rect class="s2" x="120.0" y="36" width="9.4" height="30" rx="3"/><rect class="s2" x="131.4" y="36" width="9.4" height="30" rx="3"/><rect class="s2" x="142.7" y="36" width="9.4" height="30" rx="3"/><rect class="s2" x="154.1" y="36" width="9.4" height="30" rx="3"/><g class="mark"><title>one step with the whole prefill: 463 ms</title><rect class="s1" x="165.5" y="36" width="408.4" height="30" rx="3"/></g><text class="t-val" x="370.66324891522527" y="56" text-anchor="middle" style="fill:#fff">one step: 463 ms, no new tokens for anyone</text><text class="t-tick" x="20" y="56" text-anchor="start">one GPU</text><text class="t-note" x="20" y="112" text-anchor="start">with chunked prefill: each step = 16 decodes + a 512-token slice of the prompt</text><rect class="s2" x="120.0" y="124" width="9.4" height="30" rx="3"/><rect class="s2" x="131.4" y="124" width="9.4" height="30" rx="3"/><rect class="s2" x="142.7" y="124" width="9.4" height="30" rx="3"/><rect class="s2" x="154.1" y="124" width="9.4" height="30" rx="3"/><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="165.5" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="176.8" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="201.1" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="212.4" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="236.7" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="248.0" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="272.3" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="283.6" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="307.9" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="319.2" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="343.5" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="354.8" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="379.1" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="390.4" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="414.7" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="426.0" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="450.3" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="461.6" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="485.9" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="497.2" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="521.5" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="532.8" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="557.1" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="568.4" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="592.7" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="604.0" y="124" width="22.2" height="30" rx="3"/></g><g class="mark"><title>step: 16 decodes (13 ms) + 512 prompt tokens (27 ms) = 40 ms</title><rect class="s2" x="628.3" y="124" width="10.4" height="30" rx="3"/><rect class="s1" x="639.6" y="124" width="22.2" height="30" rx="3"/></g><text class="t-tick" x="20" y="144" text-anchor="start">one GPU</text><rect class="s2" x="120" y="176" width="12" height="12" rx="3"/><text class="t-tick" x="138" y="187" text-anchor="start">decode for the 16 streaming users</text><rect class="s1" x="380" y="176" width="12" height="12" rx="3"/><text class="t-tick" x="398" y="187" text-anchor="start">prefill of the new long prompt</text><text class="t-tick" x="20" y="214" text-anchor="start">Longest pause for streaming users: 463 ms without chunking, 40 ms with it. The long prompt's first token: 463 ms vs 643 ms.</text></svg><figcaption>Top: the 8,192-token prefill runs in a single step and all 16 streaming users wait. Bottom: chunked prefill cuts it into 512-token slices that ride along with the ongoing decodes.</figcaption></figure>

**Chunked prefill** cuts the long prompt into slices and adds one slice to each step, next to the regular decodes. Using the costs measured in Part 1 (16 decodes: 12.8 ms; a 512-token prefill: 27.4 ms; an 8,192-token prefill: 450.5 ms):

| | Longest pause for streaming users | Long prompt's first token |
|---|---|---|
| no chunking | 463 ms | 463 ms |
| 512-token chunks | **40 ms** | 643 ms |

(These are estimates built by adding measured step costs, not a measured mixed batch. Later slices cost a little more than the first, because they attend to the slices before them, so the real chunked numbers would be slightly higher. The trade they show is the real one.)

The streaming users' worst pause drops from almost half a second to 40 ms. The price is that the long prompt's first token arrives later, because its prefill is now spread over 16 steps. That is the trade vLLM's documentation describes for its main knob here, `--max-num-batched-tokens`: smaller values give better inter-token latency, larger values give better time to first token.

This idea was developed in Sarathi-Serve (Agrawal et al., OSDI 2024), which reported **2.6x** higher serving capacity than the vLLM of the time for Mistral-7B on one A100. In vLLM V1, chunked prefill is **on by default**, and the scheduler fills each step's token budget with decodes first, then prefill chunks.

## When memory runs out anyway

Paging makes memory go much further, but a GPU can still fill up: many long answers all growing at once. When a running request needs a new block and none is free, vLLM **preempts** someone. It frees that request's blocks and puts it back in the queue. When the request is scheduled again, its cache has to be rebuilt.

There are two ways to rebuild it: copy the blocks out to CPU memory and back (**swapping**), or simply throw them away and recompute them with a prefill (**recomputation**). Recomputation sounds wasteful, but the tokens already generated can be prefilled together with the original prompt in one compute-bound pass, and prefill is the fast phase. In vLLM V1 the default is **recompute**, which the docs say "has lower overhead in the V1 architecture."

## The whole engine, in one loop

Put it all together and vLLM's core is a surprisingly small loop that runs every few milliseconds:

<figure class="fig"><svg viewBox="0 0 760 210" role="img" aria-label="The vLLM engine loop: schedule, run one forward pass for everything scheduled, post-process, repeat."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-1" x="20" y="40" width="220" height="90" rx="14"/><text class="t-strong" x="130" y="74" text-anchor="middle">schedule</text><text class="t-tick" x="130" y="98" text-anchor="middle">decodes first, then prefill</text><text class="t-tick" x="130" y="114" text-anchor="middle">chunks, within a token budget</text><line class="edge" x1="242" y1="85" x2="268" y2="85" marker-end="url(#ah)"/><rect class="box-2" x="270" y="40" width="220" height="90" rx="14"/><text class="t-strong" x="380" y="74" text-anchor="middle">forward pass</text><text class="t-tick" x="380" y="98" text-anchor="middle">every chosen token in one</text><text class="t-tick" x="380" y="114" text-anchor="middle">flat batch, one model run</text><line class="edge" x1="492" y1="85" x2="518" y2="85" marker-end="url(#ah)"/><rect class="box-3" x="520" y="40" width="220" height="90" rx="14"/><text class="t-strong" x="630" y="74" text-anchor="middle">post-process</text><text class="t-tick" x="630" y="98" text-anchor="middle">append new tokens, free</text><text class="t-tick" x="630" y="114" text-anchor="middle">the blocks of finished requests</text><path class="path" d="M600,132 C600,190 130,190 130,134" fill="none" marker-end="url(#ah-on)"/><text class="t-note" x="365" y="196" text-anchor="middle">repeat every step, typically every few milliseconds</text></svg><figcaption>vLLM's engine loop: schedule this step's work, run one forward pass over all of it, append the new tokens and free finished requests, repeat.</figcaption></figure>

1. **Schedule.** Decide what runs this step: first the running requests that need their next token, then chunks of waiting prompts, within a token budget and the free blocks available.
2. **Forward pass.** Flatten everything scheduled into one batch and run the model once.
3. **Post-process.** Append each sampled token to its request, return finished requests' blocks to the pool, and loop.

## The knobs that matter

Most of these ideas are on by default. These are the settings people actually tune:

| Flag | What it controls |
|---|---|
| `--gpu-memory-utilization` | Fraction of GPU memory vLLM may use (0.92 in the current docs). Whatever the weights and working memory do not need becomes KV blocks. |
| `--max-model-len` | Longest prompt + answer allowed. Lower it if you do not need the model's full context. |
| `--max-num-seqs` | Maximum number of sequences in one step. |
| `--max-num-batched-tokens` | Token budget per step. Smaller: smoother streaming. Larger: faster first token. |
| `--block-size` | Tokens per KV block (16 by default). |
| `--kv-cache-dtype` | Store the cache in `fp8` to fit about twice as many tokens. |
| `--enable-prefix-caching` | Reuse cached blocks across requests. |

Engine arguments change between releases, so check them against the version you run (`vllm serve --help`).

## Beyond one GPU: two ideas to know

**Disaggregated prefill and decode.** Chunked prefill makes the two phases share a GPU politely. The more radical option is to put them on **different GPUs**: some GPUs only prefill, others only decode, and the KV cache is sent from one to the other. DistServe (Zhong et al., OSDI 2024) showed this can serve up to 7.4x more requests, or meet 12.6x tighter latency targets, because each phase can be sized and tuned for its own bottleneck.

**Speculative decoding.** Part 1 measured that processing 16 tokens costs about the same as processing one (10.8 ms against 10.0 ms). Speculative decoding (Leviathan et al. and Chen et al., 2023) exploits that directly. A cheap drafter proposes the next few tokens, and the big model checks them all in one pass, keeping every guess up to the first one it disagrees with and correcting that one. With the right acceptance rule, the output follows exactly the same distribution as if the big model had written every token itself, so it is faster without being approximate. The gain depends on how often the guesses are right, and it shrinks when the GPU is already busy with a large batch, because the spare compute it spends is no longer spare. vLLM supports several drafters, including n-gram lookup, EAGLE and Medusa.

## The series, on one page

| Idea | The problem | The fix | What we measured or simulated |
|---|---|---|---|
| prefill vs decode | writing is 190x slower than reading | understand it: decode is memory-bound | 27 ms to read 512 tokens, 5.2 s to write 512 |
| batching | one sequence wastes each weight read | decode many sequences per step | 44x throughput at 128 sequences |
| KV cache | recomputing the past every step | keep keys and values | next token at 8,192: 449 ms to 11.5 ms |
| PagedAttention | reserved memory sits empty | 16-token blocks, allocated on demand | 48 to 415 concurrent requests |
| continuous batching | slots idle behind the slowest request | refill slots every step | 4.6x fewer steps |
| chunked prefill | long prompts freeze streaming users | slice prompts into each step | worst pause 463 ms to 40 ms |
| prefix caching | the same prompt prefilled again and again | reuse blocks by hash | first token 98 ms to 14 ms |

Every one of these ideas comes back to the two facts from Part 1: **decode is limited by memory, not math**, and **a GPU is only productive when it is decoding many sequences at once**. vLLM's whole design is a way of fitting as many sequences as possible into memory, and keeping every step full.

<details>
<summary>The simulation code</summary>

```python
"""KV allocation policies and static vs continuous batching (pure Python + numpy)."""
import heapq, math
import numpy as np

rng = np.random.default_rng(42)

# ---- how much KV memory each policy wastes (Llama 3.1 8B on one 80 GB GPU)
PER_TOKEN = 2 * 32 * 8 * 128 * 2                    # 128 KiB, from the config
BUDGET = 0.9 * 80e9 - 8.03e9 * 2 - 4e9              # 90% of memory, minus weights and ~4 GB working memory
N, MAX_LEN, MAX_TOKENS, BLOCK = 200_000, 8192, 2048, 16
prompt = np.clip(rng.lognormal(math.log(512), 0.9, N), 16, 6000).astype(int)
output = np.clip(rng.lognormal(math.log(256), 0.9, N), 8, MAX_TOKENS).astype(int)
now = prompt + (rng.uniform(0, 1, N) * output).astype(int)   # a random moment in each request's life

policies = {
    "reserve max length": np.full(N, MAX_LEN),
    "reserve prompt + max_tokens": prompt + MAX_TOKENS,
    "reserve exact final length": prompt + output,
    "paged, 16-token blocks": np.ceil(now / BLOCK).astype(int) * BLOCK,
}
for name, reserved in policies.items():
    print(f"{name:28s} useful {now.sum() / reserved.sum():6.1%}   fits {BUDGET / (reserved.mean() * PER_TOKEN):5.0f} requests")

# ---- static vs continuous batching: 4,096 requests, 64 slots
SLOTS = 64
lens = np.clip(rng.lognormal(math.log(256), 0.9, 4096), 8, 2048).astype(int)
static = sum(lens[i:i + SLOTS].max() for i in range(0, len(lens), SLOTS))
free = [0] * SLOTS
for L in lens:
    heapq.heappush(free, heapq.heappop(free) + L)
continuous = max(free)
print(f"static {static:,} steps, continuous {continuous:,} steps, {static / continuous:.1f}x")
```

</details>

## References

1. W. Kwon et al. [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180). SOSP 2023.
2. G.-I. Yu et al. *Orca: A Distributed Serving System for Transformer-Based Generative Models.* OSDI 2022.
3. A. Agrawal et al. [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310). OSDI 2024.
4. Y. Zhong et al. [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*](https://arxiv.org/abs/2401.09670). OSDI 2024.
5. Y. Leviathan, M. Kalman, Y. Matias. [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192). ICML 2023.
6. C. Chen et al. [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318). 2023.
7. vLLM team. [*Inside vLLM: Anatomy of a High-Throughput LLM Inference System*](https://vllm.ai/blog/2025-09-05-anatomy-of-vllm). 2025.
8. vLLM documentation: [Optimization and tuning](https://docs.vllm.ai/en/stable/configuration/optimization/), [Engine arguments](https://docs.vllm.ai/en/stable/configuration/engine_args/).
