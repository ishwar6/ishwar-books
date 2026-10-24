---
title: "The Big Idea: Memory, Not Arithmetic, Limits LLM Serving"
description: "How a chat server answers many users at once, why its GPU runs out of memory before it runs out of speed, and the paper's fix: store each request's memory in small pages. We read the title, the abstract, Figures 1 and 2 and §1, and we measure the KV cache of a real model."
part: 1
covers: "Title, Abstract, Figures 1 and 2, §1"
date: 2026-10-19
tags: [llm-serving, inference, kv-cache, gpu-memory, systems]
---

When many people use a chatbot at once, one GPU answers many of them together. How many it can answer at once depends on memory, and this paper shows that most of that memory was wasted. Its fix comes from the way an operating system hands out memory to programs.

Each piece below shows the exact lines of the paper, then explains them.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## The title {§Title}

> [!PAPER] · Title
> [![The title of the paper: Efficient Memory Management for Large Language Model Serving with PagedAttention, with the author names and their universities](/img/papers/vllm/p1-title.png)](/img/papers/vllm/p1-title.png)
>
> **What it says:** "Efficient Memory Management for Large Language Model Serving with PagedAttention."
>
> **What it means:**
>
> Most papers about language models are about the model: a new layer, more data, a better score. This title has no new model in it. It is about the computer that runs a model after training, and about one resource of that computer: memory. The paper does not change a single weight. Every model gives exactly the same answers before and after.
>
> "Serving" is the job of answering requests from users, all day, for many users at once. "Memory management" is the job of deciding which piece of memory each request gets, and when it gives the memory back. "PagedAttention" is the paper's new idea. The word "paged" comes from operating systems, which have cut memory into small pages since the 1960s. The rest of the paper shows why that old idea fits a new problem so well.

> [!DEFINITION] Large language model (LLM)
> A neural network that reads text and writes the next words. Its knowledge sits in billions of numbers, its **weights**. In this paper the weights never change; only the way the computer stores the model's working memory changes. The [Background page](background-the-kv-cache-and-paging.md) shows how such a model writes, one token at a time.

> [!DEFINITION] LLM serving
> Running a trained model as a service: users send **requests** (a prompt), and the system sends back the model's answer. A serving system decides which requests run together, where their data sits in memory, and when each one finishes.

## How LLM serving works

The abstract assumes that you know how a chat server works. So we first follow one request from start to end, then add many users, then look at the parts of vLLM.

### One chat request

You type "What is the capital of France?" and press Enter. The text goes to a server with a GPU, and the server cuts it into tokens. Say the prompt is 7 tokens. The model reads all 7 in one pass and writes the first token of the answer, "The". This first pass is the **prompt phase**.

Then the model writes one token per pass: "capital", then "is", then "Paris", then ".". This is the **generation phase**. The server sends each new token back at once, which is why a chatbot's answer appears word by word. When the model writes a special end token, the request is done.

During all of this, the model keeps a record of every token it has seen: the **KV cache**. After the prompt phase it holds 7 slots, one per prompt token. Each step of the generation phase adds one slot. When the request ends, the server frees the memory for the next request.

{{FIG:p1_life|One chat request over time. The prompt phase reads 7 prompt tokens in one pass and writes the first answer token. Each later pass writes one more token, and the KV cache grows by one slot per step until the end token frees it.}}

> [!DEFINITION] KV cache
> The keys and values that the model computed for every earlier token of a request. They stay in GPU memory, so the model does not compute them again for each new token. "KV" stands for key and value, two vectors from the attention layers. The sections below show what they are and how large they get.

> [!DEFINITION] Prompt phase and generation phase
> The two phases of one request. In the **prompt phase**, the model reads all prompt tokens in one pass, because they are all known. In the **generation phase**, it writes one new token per pass, because each token depends on the one before it.

Look at the bottom row of the picture. The cache grows by one slot at every step, and nobody knows when it will stop. The answer could be 5 tokens or 500. The whole paper is about this growing, unpredictable piece of memory.

### Many users at once

A server does not answer one user and then the next. Many requests run at the same time, and the server moves them forward together. At every step, it takes the requests that are running and runs one pass of the model for all of them. That group is a **batch**.

Take four requests, A to D, that arrive at different times. In step 1 only A runs. In step 2, B arrives and joins, so the batch holds 2 requests. A ends after step 5 and leaves. D arrives later and joins in step 6. So the batch changes at every step:

{{FIG:p1_users|Four requests over eight passes of the model. Each request starts with its prompt phase, then writes one token per pass. Requests join when they arrive and leave when they finish, so the batch size changes from step to step.}}

> [!NOTE] Where this idea comes from: batching at every step
> Batching is older than language models. In training, a GPU always processes many images or sentences in one pass. Text generation adds a problem: requests end at different times. Earlier servers formed a batch and kept it until every request in it was done. A short request finished early and its place stayed idle, while new requests waited outside.
>
> Orca (OSDI 2022) changed this. It schedules the batch again after every pass of the model: finished requests leave, and waiting requests join. The paper calls this **iteration-level scheduling**, and vLLM uses it too. Think of a lift that stops at every floor, against a bus that only opens its doors at the last stop.

> [!DEFINITION] Batch and batch size
> A **batch** is a group of requests that the GPU processes in the same pass. The **batch size** is how many requests are in it. A larger batch uses the GPU better, until memory runs out.

> [!DEFINITION] Throughput
> How much work a system finishes per second. Here it is requests or tokens per second. Higher throughput means each request costs less.

Now connect the two pictures. Every request in the batch holds its own KV cache in GPU memory, for as long as it runs. A request can join the batch only if its cache fits. So memory sets the batch size, and the batch size sets the throughput. The rest of this part makes that chain exact.

### The map of vLLM

Here is the whole system that the paper builds, drawn from its Figure 4. Nothing on it is new yet. It is a map for the next parts, so you can see where each piece fits.

{{FIG:p1_arch|The parts of vLLM. Requests enter the scheduler. Inside it, the KV cache manager keeps a block table per request and two allocators. Workers run the model on the GPUs. New tokens go back to the users and to the next step.}}

Requests from users arrive on the left. In the centre, the **scheduler** decides which requests run in the next step, as in the grid above. Inside it, the **KV cache manager** decides where in memory each request's cache lives. It keeps a **block table** for each request. It also has two allocators: one hands out pieces of GPU memory, and one hands out CPU memory as an overflow area.

On the right are the **workers**. Each worker drives one GPU. Its **cache engine** holds the KV cache memory on that GPU, and its **model shard** holds its part of the weights. After each step, the new tokens go back to the users and to the scheduler.

This part explains why a KV cache manager is needed at all. Part 3 opens the block tables, the allocators and the attention kernel. Part 4 explains sharing, the scheduler's choices when memory runs out, and how the workers split a model across GPUs.

> [!DEFINITION] Worker and model shard
> A **worker** is the program that controls one GPU and runs the model on it. When a model is too large for one GPU, its weights are split into pieces, and each GPU holds one piece: a **shard**. With one GPU, there is one worker and the shard is the whole model.

## The abstract {§Abstract}

> [!PAPER] · Abstract
> [![The first sentences of the abstract, with requires batching sufficiently many requests at a time, is huge and grows and shrinks dynamically, and fragmentation and redundant duplication, limiting the batch size highlighted](/img/papers/vllm/p1-abstract-1.png)](/img/papers/vllm/p1-abstract-1.png)
>
> **What it says:** high throughput "requires batching sufficiently many requests at a time". But the key-value cache of each request "is huge and grows and shrinks dynamically", and when it is managed badly, memory is wasted by "fragmentation and redundant duplication, limiting the batch size".
>
> **What it means:**
>
> Think of a bus. Driving the bus costs about the same with one passenger or with forty. A GPU that serves a language model is the same. One step of the model costs almost the same for one request or for many. So the system wants to run many requests together, in one batch.
>
> The seats on this bus are memory. Each request needs space for its KV cache. The cache is large, and it grows with every token. Nobody knows in advance how long the answer will be. If the system books seats badly, the bus leaves half empty, and fewer requests run together.
>
> The abstract names two ways to book seats badly. **Fragmentation** means free memory that is cut into pieces nobody can use. **Redundant duplication** means two copies of the same data, where one copy shared by both would do.

> [!PAPER] · Abstract
> [![The middle of the abstract, with inspired by the classical virtual memory and paging techniques in operating systems, near-zero waste in KV cache memory, and flexible sharing of KV cache within and across requests highlighted](/img/papers/vllm/p1-abstract-2.png)](/img/papers/vllm/p1-abstract-2.png)
>
> **What it says:** the paper proposes PagedAttention, an attention algorithm "inspired by the classical virtual memory and paging techniques in operating systems". On top of it, vLLM achieves "near-zero waste in KV cache memory" and "flexible sharing of KV cache within and across requests".
>
> **What it means:**
>
> Your laptop runs a browser, an editor and a music player at the same time. Each program thinks it owns one long, unbroken stretch of memory. In fact, the operating system has cut the real memory into small pages of equal size and scattered each program's pages wherever there was room. A table per program records where each page really is. This trick is **virtual memory with paging**, and it is the reason a program almost never wastes more than part of one page.
>
> PagedAttention does the same with the KV cache. The cache of a request is cut into small **blocks** of a fixed number of tokens. The blocks can sit anywhere in GPU memory, and a table per request records where. A request gets a new block only when it needs one. So the waste per request is at most the unused part of its last block: this is "near-zero waste".
>
> Blocks also make sharing easy. If two answers start from the same prompt, both tables can point to the same blocks of that prompt. One copy serves both. That is the "sharing within and across requests".

> [!DEFINITION] Virtual memory and paging
> A method of operating systems. Memory is cut into **pages** of a fixed size, often 4 KB. Each program sees its own unbroken address space, and a **page table** maps each of its pages to a real place in memory. Pages that sit next to each other for the program can be far apart in reality. The [Background page](background-the-kv-cache-and-paging.md) builds it step by step.

> [!PAPER] · Abstract
> [![The end of the abstract, with by 2-4× with the same level of latency, and longer sequences, larger models, and more complex decoding algorithms highlighted](/img/papers/vllm/p1-abstract-3.png)](/img/papers/vllm/p1-abstract-3.png)
>
> **What it says:** vLLM improves the throughput of popular LLMs "by 2-4× with the same level of latency" compared with FasterTransformer and Orca. The gain is larger with "longer sequences, larger models, and more complex decoding algorithms".
>
> **What it means:**
>
> Throughput alone is easy to raise: make every user wait longer, and the GPU can collect huge batches. The claim is fair only because of the second half, "with the same level of latency". Each user waits about as long as before, and the same GPU serves two to four times as many of them. For a company that pays for GPUs by the hour, that is a cost cut of half or more.
>
> The three conditions tell you where the gain comes from. Longer sequences mean larger caches, so wasted memory hurts more. Larger models leave less room for the cache. Complex decoding, such as several answers to one prompt, gives more chances to share. In each case memory is the tighter limit, so better memory management helps more.

> [!DEFINITION] Latency
> How long one user waits for an answer. The paper measures **normalized latency**: the time of a request divided by the number of tokens in its answer, in seconds per token. Part 5 reads those curves.

> [!DEFINITION] FasterTransformer and Orca
> The two strongest serving systems of 2022. **FasterTransformer** is NVIDIA's fast inference library. **Orca** (OSDI 2022) added a better way to form batches. Orca's code was not public, so the authors built their own version, in three variants that reserve memory in different ways. Part 5 describes them.

> [!NOTE] How serving systems are compared
> A fair test works like a shop on a busy day. Requests arrive at a fixed rate, say a few per second, and the test measures the normalized latency of each one. At a low rate, latency stays flat, because the server keeps up. At some rate the server falls behind, a queue forms, and latency shoots up.
>
> The highest rate a system can serve before that jump is its throughput. So "2-4× with the same level of latency" means that vLLM's jump comes at a rate two to four times higher. Part 5 reads these curves for every model and dataset of the paper.

## Why serving is expensive {§1}

> [!PAPER] · §1
> [![The first paragraph of the introduction, with very expensive, 10× more expensive than a traditional keyword query, and increasing the throughput highlighted](/img/papers/vllm/p1-intro-cost.png)](/img/papers/vllm/p1-intro-cost.png)
>
> **What it says:** running LLM applications "is very expensive", and one LLM request can be "10× more expensive than a traditional keyword query". So "increasing the throughput", and with it reducing the cost per request, matters more and more.
>
> **What it means:**
>
> A web search looks up an index that was built in advance. An LLM request runs billions of multiplications for every word of the answer, on hardware that costs thousands of dollars per card. The 10× estimate comes from a news report that the paper cites, not from the authors' own measurement.
>
> The paper's goal follows from this. You cannot make the model smaller without changing its answers. You can serve more requests per GPU. Every gain in throughput is a direct cut in the cost of each request.

## Figure 1: where the memory goes {§Figure 1}

> [!PAPER] · Figure 1
> [![Figure 1 of the paper: left, the memory of a 40 GB A100 split into parameters (26 GB, 65%), KV cache (more than 30%) and others; right, memory usage and throughput against batch size for existing systems and vLLM](/img/papers/vllm/p1-figure1.png)](/img/papers/vllm/p1-figure1.png)
>
> **What it says:** the parameters of a 13B model "persist in GPU memory throughout serving". The memory for the KV cache "is (de)allocated per serving request". On the right, vLLM "smooths out the rapid growth" of KV cache memory, "leading to a notable boost in serving throughput".
>
> **What it means:**
>
> The left picture is a budget. A GPU with 40 GB holds a model of 13 billion parameters. The parameters take 26 GB, 65% of the card, and they never move. A small part, "Others", holds short-lived data of the computation. Everything else, a bit over 30%, is for the KV cache of the requests. That slice is the only part a serving system can manage, so it decides how many requests run together.
>
> The right picture has two panels that share one x-axis, the batch size. The top panel shows memory. With existing systems (orange), memory climbs steeply and hits the 40 GB ceiling at a batch of about 8 requests. With vLLM (blue), each request takes much less memory, so the line climbs slowly and reaches the ceiling at about 40. The bottom panel shows what that buys. Throughput rises with the batch size, so the system that fits 40 requests gets far more tokens per second than the one that fits 8.
>
> The figure states the whole paper in one picture. The model and the GPU are the same. Only the memory per request changes, and throughput follows it.

> [!DEFINITION] GPU memory
> The memory on the graphics card, separate from the computer's main memory. The NVIDIA A100 in the paper has 40 GB or 80 GB. The model weights and the KV cache must both sit in it during serving, because reading from main memory is far too slow.

> [!DEFINITION] FP16
> A way to store each number in 16 bits, which is 2 bytes. Models are usually served in FP16, so a model of 13 billion parameters needs 13 billion times 2 bytes, which is 26 GB.

## Why one token at a time {§1}

> [!PAPER] · §1
> [![The paragraph about autoregressive generation, with one at a time, repeated until the model outputs a termination token, and memory-bound, underutilizing the computation power of GPUs highlighted](/img/papers/vllm/p1-intro-autoregressive.png)](/img/papers/vllm/p1-intro-autoregressive.png)
>
> **What it says:** the model "generates words (tokens), one at a time", based on the prompt and the tokens it generated so far. The process is "repeated until the model outputs a termination token". This makes the workload "memory-bound, underutilizing the computation power of GPUs".
>
> **What it means:**
>
> A language model cannot write a whole answer at once. It writes one token, adds it to the text, and runs again to write the next one. An answer of 300 tokens needs 300 runs of the model, one after another. Each run needs the one before it, so they cannot happen at the same time.
>
> Each run is small work for a GPU: one new token per request. But to do it, the GPU must read all 26 GB of weights from its memory. A GPU can multiply much faster than it can read memory. So most of the time, its arithmetic units sit idle and wait for data. That is what **memory-bound** means. The GPU is not too slow at maths; it is starved of data.
>
> Batching is the cure. If 40 requests share one run, the weights are read once and used 40 times. The cost of the read is split 40 ways.

> [!DEFINITION] Autoregressive generation
> Writing text one token at a time, where each new token depends on all tokens before it. "Auto" (self) and "regressive" (predicting from earlier values): the model feeds its own output back in as input.

> [!DEFINITION] Memory-bound
> A computation whose speed is set by how fast data moves from memory, not by how fast the processor computes. The opposite is **compute-bound**. Writing one token for one request is memory-bound on a GPU.

{{FIG:p1_bound|Why serving is memory-bound. Each step reads every weight once, whatever the batch size. In our run, a batch of 1 takes 4.51 ms for 1 token, and a batch of 64 takes 10.01 ms for 64 tokens.}}

> [!NOTE] Where this idea comes from: the memory wall
> For decades, processors got faster much more quickly than memory did. In 1995, the computer architects William Wulf and Sally McKee called this gap the "memory wall". A GPU makes the gap extreme: thousands of arithmetic units wait on one memory.
>
> Think of a chef who can chop a carrot every second, with a helper who brings one carrot every ten seconds. The chef's speed does not matter; the helper's does. When a model writes one token for one request, it does about one multiply and one add for each weight it reads. With a batch of 64, it does 64 of them for the same read. That is why batching helps so much, and why the next limit is memory space.

### Why the KV cache exists, on a laptop

Before we measure the cache, we check why it exists. The attention layers of the model compare the newest token with every earlier token. For that, each earlier token needs two vectors, its **key** and its **value**. Without a cache, the model would compute the keys and values of the whole text again at every step. With a cache, it computes them once and keeps them.

We use OPT-125M, the smallest model of the OPT family that the paper serves, in FP16 on an Apple laptop GPU. We write greedy generation three ways: with no cache, with our own loop that keeps a cache, and with the library's `model.generate`. All three must give the same tokens. In the output, look for two things: the word True on both checks, and the times.

```python
@torch.no_grad()
def generate_no_cache(ids, n_new):
    """Recompute the keys and values of every earlier token at every step."""
    for _ in range(n_new):
        logits = model(ids, use_cache=False).logits
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
    return ids


@torch.no_grad()
def generate_with_cache(ids, n_new):
    """The prompt phase once, then one new token per step; the cache keeps every earlier key and value."""
    out = model(ids, use_cache=True)                       # prompt phase: all prompt tokens in one pass
    cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    toks = [nxt]
    for _ in range(n_new - 1):                             # generation phase: one token in, one token out
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        toks.append(nxt)
    return torch.cat([ids] + toks, dim=1), cache
```

The first function feeds the whole text into the model at every step. The second feeds the prompt once, then only the newest token, and passes the cache back in. The model reads the old keys and values from the cache instead of computing them again:

```text
== 3. Greedy generation three ways: no cache, our own KV cache loop, model.generate
64 new tokens; same tokens, no cache vs our cache: True; our cache vs generate: True
text: ', a new nation, a new nation, a new nation, a new nation, a new nation, a new nation, a ne' ...
cache after generation: 80 tokens = 2,949,120 bytes (the last token gets its key and value at the next step)
time: no cache 0.308 s, with cache 0.260 s (1.2x)
with a 400-token prompt: no cache 0.633 s, with cache 0.270 s (2.3x)
```

The three methods give the same 64 tokens, so the cache changes nothing about the answer. (A model of 125M parameters repeats itself; the text is not the point.) The time is what changes. With a short prompt the gain is small, 1.2 times, because each step of a tiny model is dominated by fixed costs. With a prompt of 400 tokens, the cache cuts the time by 2.3 times. The longer the context, the more work the cache saves.

So the cache trades memory for time. Every request keeps its keys and values in GPU memory for as long as it runs. The rest of the paper is about the memory side of that trade.

{{FIG:p1_cache|The KV cache of one request. The prompt phase writes the keys and values of all prompt tokens in one pass. Each step of the generation phase adds one more column.}}

## How large the KV cache is {§1}

> [!PAPER] · §1
> [![The paragraph about batching and Figure 1, with batching multiple requests, Approximately 65% of the memory, remain static, Close to 30% of the memory, dynamic states of the requests, and commonly referred to as KV cache highlighted](/img/papers/vllm/p1-intro-batching.png)](/img/papers/vllm/p1-intro-batching.png)
>
> **What it says:** throughput improves by "batching multiple requests together". For a 13B model on a 40 GB A100, "Approximately 65% of the memory" holds the weights, which "remain static". "Close to 30% of the memory" stores the "dynamic states of the requests": the key and value tensors, "commonly referred to as KV cache".
>
> **What it means:**
>
> The weights are the same for every request, so they are loaded once and stay. The KV cache is different: each request has its own, it starts small, it grows with every token, and it vanishes when the request ends. "Dynamic states" is a good name. It is the working memory of each conversation.
>
> The paper's §3 gives the size. For OPT-13B, one token needs 800 KB of cache. That is 2 (a key and a value) times 5,120 (the vector size) times 40 layers times 2 bytes. A full request of 2,048 tokens needs about 1.6 GB. So 12 GB holds only a few requests of full length.

We check this formula twice. Method 1 applies it to the real configuration file of OPT-13B. Method 2 runs OPT-125M on a prompt and adds up the bytes of the cache tensors that the library really keeps. If the formula is right, it predicts those bytes exactly:

```python
def kv_bytes_per_token(cfg, bytes_per_value=2):
    """The paper's formula (§3): 2 (key and value) x hidden size x number of layers x bytes per FP16 number."""
    return 2 * cfg['hidden_size'] * cfg['num_hidden_layers'] * bytes_per_value


def cache_bytes(cache):
    """Method 2: add up the bytes of every key and value tensor that the library really keeps."""
    return sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
```

```text
== 1. The KV cache of one token
OPT-13B config: 40 layers, hidden size 5120, max length 2048 tokens
Method 1, formula: 2 x 5120 x 40 x 2 bytes = 819,200 bytes = 800 KB per token
one request of 2048 tokens: 1,677,721,600 bytes = 1.68 GB
OPT-125M: 12 layers, hidden size 768; prompt of 17 tokens
shape of the keys of layer 0: (1, 12, 17, 64)  (batch, heads, tokens, head size), torch.float16
Method 1, formula: 36,864 bytes per token x 17 tokens = 626,688 bytes
Method 2, real tensors: 626,688 bytes; agree: True
```

The formula matches the real tensors to the byte. The shape shows where the "hidden size" goes: 12 attention heads of 64 numbers each make 768. The 800 KB of the paper is 819,200 bytes, which is 800 times 1,024. A full request of OPT-13B is 1.68 GB in decimal units, or 1.56 GiB; the paper rounds to "1.6 GB".

> [!DEFINITION] Key and value vectors
> In each attention layer, every token gets a **query**, a **key** and a **value** vector. The new token's query is compared with the keys of all earlier tokens to decide how much attention each one gets. The answer is a weighted average of their values. Keys and values of earlier tokens never change, which is why they can be cached.

Now Figure 1's budget, as arithmetic. The 13B model takes 26 GB. Suppose the cache gets 30% of the card, 12 GB:

```text
== 2. Figure 1 (left): a 13B model on a 40 GB A100
weights: 13B parameters x 2 bytes = 26 GB = 65% of 40 GB
KV cache room at 30% of the GPU: 12 GB = 14,648 tokens of OPT-13B
requests of 2048 tokens that fit: 7.2
```

Twelve gigabytes sounds large, but it holds only 14,648 tokens of OPT-13B. If every request books room for 2,048 tokens, only 7 requests fit. That matches the orange line of Figure 1, which hits the ceiling at a batch of about 8.

> [!PAPER] · §1
> [![The next lines of the introduction, with the way the KV cache is managed is critical in determining the maximum batch size highlighted](/img/papers/vllm/p1-intro-batch-limit.png)](/img/papers/vllm/p1-intro-batch-limit.png)
>
> **What it says:** because the weights are constant and the activations are small, "the way the KV cache is managed is critical in determining the maximum batch size".
>
> **What it means:**
>
> This is the key sentence of §1. Of the three slices of memory, two are fixed. The weights cannot shrink without changing the model. The activations are small and short-lived. Only the KV cache slice depends on decisions that the serving system makes. So the batch size, and with it the throughput, is decided by how well the system packs KV caches into that slice.

> [!DEFINITION] Activations
> The temporary results inside the model while it computes one step, such as the output of each layer. They are thrown away after the step. The paper's Figure 1 calls them "Others".

### The two phases, and what batching buys

The paper's §2 names the two phases that we met in the chat request above. The prompt phase reads all prompt tokens in one pass. The generation phase writes one token per pass. We time both on the same 400 tokens. If the GPU is starved of data, the second way should be far slower:

```text
== 4. The prompt phase against the generation phase (OPT-125M, 400 tokens)
prompt phase, 400 tokens in one pass:       9.6 ms =    41,536 tokens/s
the same 400 tokens one at a time:       1629.9 ms =       245 tokens/s
one pass is 169x faster for the same work
```

The work is the same, but one pass is 169 times faster. In one pass, the GPU reads the weights once and uses them for 400 tokens. One at a time, it reads them 400 times. On our laptop, part of each step is also the fixed cost of a call from Python. So the exact factor is larger than on a server GPU. The direction is the paper's point: the generation phase wastes the GPU.

Batching gives back what the generation phase loses. We run one decode step for 1 to 64 requests at once, each with 128 tokens of cache. The function below is one such step; after it, we cut the cache back, so every timed run does the same work. Watch the step time grow much more slowly than the batch:

```python
def step():
    with torch.no_grad():
        model(step_in, past_key_values=cache, use_cache=True)
        for layer in cache.layers:                      # drop the token just added, so every run is the same
            layer.keys, layer.values = layer.keys[:, :, :CTX], layer.values[:, :, :CTX]
```

```text
== 5. One decode step for a batch of requests (each with a 128-token context)
batch  step ms  tokens/s
    1     4.51       221
    2     5.45       367
    4     5.07       788
    8     6.18     1,294
   16     7.22     2,217
   32     7.93     4,033
   64    10.01     6,393
from batch 1 to 64: step time x2.2, throughput x28.9
```

{{FIG:p1_batch|Our run: one decode step of OPT-125M on a laptop GPU. A batch of 64 takes 2.2 times as long as a batch of 1, but it writes 64 tokens instead of 1, so throughput grows 29 times.}}

Sixty-four requests take 2.2 times as long as one request, and they produce 64 tokens. So throughput grows 29 times. This is the bottom panel of Figure 1 on a small scale: throughput climbs with the batch size. On a real server, what stops the climb is the memory for all those caches. That is the subject of the rest of the paper. Our timings are the fastest of 7 runs on a laptop, so treat them as a shape, not as server numbers.

## Why the existing systems waste memory {§1}

> [!PAPER] · §1
> [![The paragraph about contiguous memory, with contiguous memory space, tensors to be stored in contiguous memory, dynamically grows and shrinks over time, and lifetime and length are not known a priori highlighted](/img/papers/vllm/p1-intro-contiguous.png)](/img/papers/vllm/p1-intro-contiguous.png)
>
> **What it says:** existing systems store the KV cache of a request "in contiguous memory space", because most deep learning frameworks require "tensors to be stored in contiguous memory". But the KV cache "dynamically grows and shrinks over time", and its "lifetime and length are not known a priori".
>
> **What it means:**
>
> A tensor in PyTorch is one unbroken block of memory. The fast GPU code that computes attention expects the keys of a request to sit side by side in that block. So the simple design is one block per request, and the block must be big enough for the whole cache.
>
> The trouble is the size. When a request arrives, nobody knows how long the answer will be. It may be 10 tokens or 2,000. A block that is too small cannot grow in place, because other data may sit right after it. So the system books the largest size up front, just in case.
>
> Ordinary deep learning never hits this problem. When you train a model or classify an image, the tensor sizes are known before the computation starts. The KV cache is the unusual case: its size is decided by the model's own output, one token at a time.

> [!DEFINITION] Contiguous memory
> Memory addresses that follow each other without a gap, like seats in one row. A contiguous block cannot grow if the next seats are taken; it must move to a larger free row.

> [!DEFINITION] Tensor
> An array of numbers with any number of dimensions, the basic data type of PyTorch. A vector is a 1-D tensor and a matrix is a 2-D tensor. The KV cache of a request is stored as tensors.

> [!NOTE] Where this comes from: why tensors are contiguous
> A tensor library stores a matrix as one long row of numbers. To find the number in row i and column j, it computes one address: the start, plus i times the row length, plus j. No lookup is needed. GPU code is fast partly because of this. Threads that sit side by side read numbers that sit side by side, in one wide read.
>
> Think of a street where the house numbers run in order. You find number 40 without asking anyone. If the houses were scattered across town, you would need a directory first. PagedAttention accepts that small cost: Part 3 shows how its kernel reads a table first, then the keys and values.

> [!PAPER] · §1
> [![The paragraph about fragmentation, with pre-allocate a contiguous chunk, request's maximum length, internal fragmentation, other shorter requests cannot utilize any, external, and only 20.4% - 38.2% of the KV cache memory is used to store the actual token states highlighted](/img/papers/vllm/p1-intro-frag.png)](/img/papers/vllm/p1-intro-frag.png)
>
> **What it says:** existing systems "pre-allocate a contiguous chunk of memory with the request's maximum length (e.g., 2048 tokens)". This causes internal fragmentation, because the real length is often much shorter. Even with a known length, "other shorter requests cannot utilize any part of the chunk" that is still unused. External fragmentation adds more waste. In the authors' profiling, "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states".
>
> **What it means:**
>
> Picture a restaurant that books tables by the longest possible dinner. A couple that stays one hour holds a table for five hours. That is the first waste, **internal fragmentation**: memory inside a request's chunk that its tokens never fill.
>
> Now suppose the restaurant knows exactly how long each guest will stay. It still books the table from the start of the meal to the end. During the first hour the table is half used, but nobody else can sit there. That is the second waste, which the paper calls **reservation**: memory that this request will fill later, but which is blocked now.
>
> The third waste is **external fragmentation**. Chunks of different sizes come and go, and they leave gaps between them. A gap of 300 tokens is useless to a request that needs a chunk of 2,048. This is true even if many such gaps add up to more. All three together leave only 20% to 38% of the cache memory holding real data.

> [!DEFINITION] Internal and external fragmentation
> **Internal fragmentation** is unused memory inside a block that has been given to someone. **External fragmentation** is free memory between blocks, split into gaps that are each too small to use. Both are old terms from operating systems.

{{FIG:p1_paged|Contiguous chunks against blocks. Each request books one chunk for its longest possible answer, and gaps between chunks stay empty. With blocks, a request takes one small block at a time, wherever one is free.}}

> [!NOTE] Where these words come from: memory allocators
> Operating systems met this problem in the 1960s. Programs ask for pieces of memory of many sizes, and give them back in any order. The code that hands out the pieces is an **allocator**, and both kinds of fragmentation are its classic enemies.
>
> One old answer is to round every request up to a power of two. The **buddy allocator**, described by Kenneth Knowlton in 1965, keeps free memory in pieces of 2, 4, 8, 16 units and so on. Splitting and merging such pieces is fast. The price is internal fragmentation: a request for 17 units gets 32. Orca (Pow2) below books memory in the same way. The [Background page](background-the-kv-cache-and-paging.md) works through both kinds of waste with small numbers.

> [!PAPER] · Figure 2
> [![Figure 2 of the paper: stacked bars of KV cache usage for Orca (Max), Orca (Pow2), Orca (Oracle) and vLLM, split into token states, reservation, internal fragmentation, and external fragmentation and others; token states are 20.4, 26.8, 38.2 and 96.3 percent](/img/papers/vllm/p1-figure2.png)](/img/papers/vllm/p1-figure2.png)
>
> **What it says:** the average percentage of "memory wastes" in different serving systems, measured during the experiment of §6.2.
>
> **What it means:**
>
> Each bar is all of the KV cache memory of one system, split by use. Only the green part holds real tokens. Orca (Max) books 2,048 tokens per request: 20.4% green, and 57.3% internal fragmentation. Orca (Pow2) rounds the answer's real length up to a power of two, so it books at most twice too much. It gets 26.8% green, but now 41.6% is lost to external fragmentation. Chunks of many sizes leave odd gaps.
>
> Orca (Oracle) cheats on purpose: it knows each answer's length in advance and books exactly that. Even this impossible best case keeps only 38.2% green, because of reservation (25.2%) and external fragmentation (36.6%).
>
> vLLM is 96.3% green. The bars say that the problem was never the size of the GPU. In the earlier systems, 62% to 80% of the cache memory was empty or blocked. Fill it, and the same card runs two to four times as many requests.

> [!DEFINITION] Power of two
> Any of the numbers 2, 4, 8, 16, ..., 1024, 2048. Booking "the next power of two" for an answer of 25 tokens books 32. Memory allocators often round to powers of two because it makes free space easier to track.

### Figure 2 in miniature

We cannot run Orca, and we have no A100. But we can count where the memory goes under each booking rule, using the lengths of real requests. We take 2,000 random requests of Alpaca, one of the paper's two datasets. We measure their prompt and answer lengths with the OPT tokenizer. Then we walk each request through its life, token by token, and count three kinds of slots at every step:

```python
SCHEMES = {                                                # slots held, given current length, prompt and final length
    'max (2048)': lambda cur, p, final: MAXLEN,
    'pow2': lambda cur, p, final: p + 2 ** math.ceil(math.log2(final - p)),   # output booked up to 2x (25 -> 32)
    'oracle': lambda cur, p, final: final,
    f'paged ({BLOCK})': lambda cur, p, final: BLOCK * math.ceil(cur / BLOCK),
}


def simulate(alloc):
    """Method 1: walk every request step by step. At each step, count the slots that hold a token (used),
    the slots kept for tokens that will come later (reserved) and the slots that no token will ever fill (waste)."""
    used = reserved = waste = 0
    for p, o in REQS:
        final = p + o
        for cur in range(p, final + 1):                    # after the prompt phase, then after each new token
            a = alloc(cur, p, final)
            used += cur
            fut = min(a, final) - cur                       # slots that a later token of this request will fill
            reserved += fut
            waste += a - cur - fut
    return used, reserved, waste
```

Each rule takes the current length, the prompt length and the final length of a request, and returns how many slots it holds. The first three book once, at the start, as the paper's three versions of Orca do. The power-of-two rule rounds the answer's length up, so an answer of 25 tokens books 32 slots. The paged rule books blocks of 16 slots, one more each time the last block is full.

Method 2 computes the same totals with a closed formula instead of the loop. The sum of the current lengths over the steps $$p, p+1, \dots, p+o$$ is $$(o+1)(2p+o)/2$$, so no walk is needed. The two methods must agree to the slot:

```text
== 6. How much reserved KV cache memory holds real tokens (Alpaca lengths, OPT tokenizer)
2000 requests; mean prompt 18.9 tokens, mean output 58.4 tokens, longest 514 tokens
scheme       token states  reserved  never used  Method 2 agrees
max (2048)           3.6%      2.7%       93.7%             True
pow2                41.2%     31.3%       27.4%             True
oracle              56.8%     43.2%        0.0%             True
paged (16)          90.7%      8.4%        0.8%             True
external fragmentation is not simulated: we place no requests in a real address space
requests of OPT-13B that fit in 12 GB of KV cache, on average over their lifetime:
  max (2048)    2048.0 slots per request ->    7.2 requests
  pow2           177.5 slots per request ->   82.5 requests
  oracle         128.9 slots per request ->  113.7 requests
  paged (16)      80.7 slots per request ->  181.5 requests
```

{{FIG:p1_waste|Our simulation on 2,000 Alpaca requests: the share of booked KV cache memory that holds a token, that waits for a later token, and that no token ever fills. Blocks of 16 slots keep 90.7% of the memory full.}}

The order is the paper's order: max, then power of two, then oracle, then paged. Alpaca requests are short, 77 tokens on average, so booking 2,048 slots is extreme here: only 3.6% of the memory holds tokens. Even the oracle keeps 43.2% blocked as reservation, because a request books its final length on its first step. Paged blocks keep 90.7% of the memory full. On average a request holds 80.7 slots instead of 2,048, so 181 requests fit where 7 did.

Our numbers differ from Figure 2, and the reasons are worth knowing. We place no chunks in a real address space, so we count no external fragmentation; that is why our oracle shows 56.8% instead of 38.2%. The paper's bars come from a running system with ShareGPT or Alpaca traffic, a scheduler and requests that arrive over time. Our paged share is below the paper's 96.3% because our requests are short. The half-empty last block of 16 slots is a larger part of each one. The lesson is the same: booking ahead is what wastes memory.

Here is what those counts mean for one real card. We take Figure 1's budget, 12 GB of KV cache space, and fill it with requests under the first rule and under the paged rule:

{{FIG:p1_before|The same 40 GB GPU in our simulation. Before vLLM, 12 GB of KV cache space holds 7.2 chunks of 2,048 slots, and only 3.6% of it holds a token. With blocks of 16 slots, it holds 181.5 requests on average, and 90.7% of it holds a token.}}

The weights do not change, and neither does the card. Only the way the cache is booked changes, and the number of requests that fit grows from 7 to about 181. For these short Alpaca requests the gap is extreme. The paper's own measurement, with real traffic, is the 2 to 4 times of the abstract.

## The second problem: no sharing {§1}

> [!PAPER] · §1
> [![The start of the paragraph about memory sharing, with cannot exploit the opportunities for memory sharing highlighted](/img/papers/vllm/p1-intro-sharing.png)](/img/papers/vllm/p1-intro-sharing.png)
> [![The rest of the paragraph, with parallel sampling and beam search, partially share their KV cache, and separate contiguous spaces highlighted](/img/papers/vllm/p1-intro-sharing-2.png)](/img/papers/vllm/p1-intro-sharing-2.png)
>
> **What it says:** existing systems "cannot exploit the opportunities for memory sharing". Decoding algorithms such as "parallel sampling and beam search" produce several outputs per request, which could "partially share their KV cache". But each sequence is stored in "separate contiguous spaces", so sharing is impossible.
>
> **What it means:**
>
> A coding assistant often shows you three suggestions for the same line. All three start from the same prompt, so the keys and values of the prompt are identical in all three. Only the new tokens differ. In a contiguous design, each suggestion gets its own chunk, and the prompt's cache is copied three times. That is the "redundant duplication" of the abstract.
>
> Beam search shares even more. It keeps several candidate answers alive, and they share long common beginnings that change as the search goes on. With one chunk per candidate, the shared part cannot be stored once. The paper's §6.3 measures how much memory sharing saves: up to 55% with beam search.

{{FIG:p1_share|Three suggestions for one prompt. Before, each answer has its own chunk with its own copy of the prompt's cache. With vLLM, the three block tables all point to one shared copy of the prompt blocks.}}

> [!DEFINITION] Parallel sampling and beam search
> Two **decoding algorithms**, ways to choose the output tokens. **Parallel sampling** draws several different answers to the same prompt. **Beam search** keeps the few most likely partial answers at every step and extends each of them, then returns the best complete one.

## The fix: pages for the KV cache {§1}

> [!PAPER] · §1
> [![The paragraph that introduces PagedAttention, with virtual memory with paging, divides the request's KV cache into blocks, fixed number of tokens, not necessarily stored in contiguous space, and blocks as pages, tokens as bytes, and requests as processes highlighted](/img/papers/vllm/p1-intro-paged.png)](/img/papers/vllm/p1-intro-paged.png)
>
> **What it says:** PagedAttention borrows "virtual memory with paging" from operating systems. It "divides the request's KV cache into blocks", each holding the keys and values of a "fixed number of tokens". The blocks are "not necessarily stored in contiguous space". Think of "blocks as pages, tokens as bytes, and requests as processes".
>
> **What it means:**
>
> The analogy is exact, and it is worth reading slowly. In an operating system, a **process** (a running program) sees its memory as one long row of bytes. Behind the scenes the row is cut into pages, and the pages sit anywhere. In vLLM, a request sees its cache as one long row of tokens. Behind the scenes the row is cut into blocks, and the blocks sit anywhere.
>
> The hard part is in the name: Paged**Attention**. The attention computation must now read keys and values from blocks scattered across memory, not from one tidy tensor. The authors wrote a new GPU kernel that follows each request's block table while it computes attention. Part 3 opens that kernel.

> [!NOTE] Where this idea comes from: paging
> Around 1962, the Atlas computer at the University of Manchester had a small, fast core memory and a large, slow magnetic drum. Programs were larger than the fast memory. Atlas cut memory into pages of a fixed size and moved them between core and drum when needed. A table recorded where each page was, so a programmer could write as if all memory were one large, fast store.
>
> This is **virtual memory with paging**, and every laptop and phone uses it today. Think of a binder of loose pages. The pages of one chapter can sit anywhere, because the table of contents lists where each one is. The paper makes the same move with the KV cache:

{{FIG:p1_os|Paging in an operating system and in vLLM. A program sees pages 0, 1 and 2 in order, and its page table maps them to scattered frames. A request sees blocks 0, 1 and 2, and its block table maps them to scattered GPU blocks.}}

> [!DEFINITION] Process
> A running program, as the operating system sees it. Each process has its own address space and its own page table.

> [!DEFINITION] GPU kernel
> A function that runs on the GPU, on thousands of threads at once. Attention in a serving system is one kernel. PagedAttention is a kernel that reads keys and values through a block table.

> [!PAPER] · §1
> [![The next lines, with relatively small blocks and allocating them on demand, eliminates external fragmentation, all blocks have the same size, and at the granularity of a block highlighted](/img/papers/vllm/p1-intro-paged-2.png)](/img/papers/vllm/p1-intro-paged-2.png)
>
> **What it says:** the design reduces internal fragmentation "by using relatively small blocks and allocating them on demand". It "eliminates external fragmentation as all blocks have the same size". And it enables memory sharing "at the granularity of a block".
>
> **What it means:**
>
> Each of the three wastes falls to one property of blocks. Blocks are given **on demand**, so nothing is booked for tokens that do not exist yet: no reservation. Blocks are **small**, so the empty part of the last block is small: almost no internal fragmentation. Blocks are **all the same size**, so any free block fits any request: no external fragmentation.
>
> Sharing follows from the block table. Two tables can point to the same block. When one of the two sequences must write into a shared block, vLLM copies that one block first, a trick from operating systems called **copy-on-write**. Part 4 shows it at work.

> [!NOTE] Where this idea comes from: copy-on-write
> In Unix, the call `fork()` makes a copy of a running program. Copying all of its memory would be slow, and the copy often replaces itself with another program right away. So modern Unix systems let the two programs share every page, marked read-only. When one program writes to a page, the system copies that one page first.
>
> Think of two students who share one textbook. When one of them wants to write notes in a chapter, they photocopy that chapter only. The paper does the same with blocks. It counts how many tables point to each block, and copies a block only when a shared block must change.

## vLLM, and the claim {§1}

> [!PAPER] · §1
> [![The paragraph about vLLM, with near-zero waste in KV cache memory, preemptive request scheduling, by 2-4×, and without affecting the model accuracy at all highlighted](/img/papers/vllm/p1-intro-vllm.png)](/img/papers/vllm/p1-intro-vllm.png)
>
> **What it says:** vLLM is "a high-throughput distributed LLM serving engine" built on PagedAttention that achieves "near-zero waste in KV cache memory". It uses block-level memory management and "preemptive request scheduling". It improves throughput "by 2-4×" over the state of the art "without affecting the model accuracy at all".
>
> **What it means:**
>
> PagedAttention is a kernel. vLLM is the whole engine around it. It has a scheduler that decides which requests run in each step. A manager hands out blocks, and workers run the model on one or more GPUs. "Preemptive" means that when memory runs out, the scheduler can pause a request, free or move its blocks, and resume it later.
>
> "Without affecting the model accuracy at all" is an important promise. Many speed-ups for language models change the numbers: smaller number formats, pruned weights, approximate attention. PagedAttention changes only where the keys and values are stored. The attention result is mathematically the same, so the answers are the same. Our own check above showed the same idea on a small scale: a cache changes the time, never the tokens.

> [!DEFINITION] Preemption
> Stopping a running job before it finishes, to give its resources to another job, and resuming it later. Operating systems preempt programs many times per second. When memory is short, they can also move a paused program's pages to disk and bring them back later; this is **swapping**. The paper can move blocks to CPU memory in the same way, with the CPU block allocator of the map above.

### The paper's numbers, checked

The last section of our script turns Figure 2's percentages into ratios, so we can compare them with the claim of "2-4×". Look at the line that starts with "vLLM keeps":

```text
== 7. The paper's headline numbers, checked
Figure 2, token states in Orca (Max)      20.4%   -> waste  79.6%
Figure 2, token states in Orca (Pow2)     26.8%   -> waste  73.2%
Figure 2, token states in Orca (Oracle)   38.2%   -> waste  61.8%
Figure 2, token states in vLLM            96.3%   -> waste   3.7%
Figure 2: vLLM keeps 4.7x to 2.5x more of the memory for real tokens
§3: 819,200 bytes x 2048 tokens = 1.56 GiB = 1.68 GB ("1.6 GB")
```

These ratios explain the size of the headline claim. vLLM keeps 2.5 to 4.7 times more of the cache memory full of real tokens. More full memory means larger batches, and larger batches mean more throughput. The measured gain of "2-4×" sits inside that range. It cannot be the full ratio, because other limits, such as compute, start to matter when batches get large.

## What our small run cannot show

Our run measured the KV cache of a real model, checked the paper's formula to the byte, and showed that batching raises throughput. It did not run vLLM or Orca. The paper's PagedAttention kernel is CUDA code for NVIDIA GPUs, and our laptop has an Apple GPU. Our timings come from a model of 125 million parameters, where fixed costs per step are large. On an A100 with a 13B model, the numbers would differ, though the direction would not.

Our Figure 2 is a count, not a measurement of a running server. It has no external fragmentation, no arrival times and no scheduler. Part 6 builds a small paged cache and a serving loop that do allocate real blocks, so we can see the block table at work.

> [!TAKEAWAYS] Key takeaways
> - **A serving step** runs one pass of the model for every running request. Requests join and leave the batch after every step, as in Orca's iteration-level scheduling.
> - **Serving is memory-bound.** Each new token needs a pass over all the weights, so one request at a time leaves the GPU idle. Batching many requests fixes this, until memory runs out.
> - **The KV cache** holds the keys and values of every earlier token. It costs 2 × hidden size × layers × 2 bytes per token: 800 KB for OPT-13B. In our run, the formula matched the real tensors to the byte.
> - **On a 40 GB A100** with a 13B model, the weights take 26 GB (65%). About 12 GB is left for the cache, which is 14,648 tokens, or 7 requests if each books 2,048.
> - **Existing systems book one contiguous chunk per request,** sized for the longest answer. Reservation, internal fragmentation and external fragmentation leave only 20.4% to 38.2% of the cache memory holding real tokens.
> - **They also cannot share.** Several answers to one prompt each get a full copy of the prompt's cache.
> - **PagedAttention** cuts the cache into small blocks of a fixed size. Blocks are given on demand and placed anywhere, with a block table per request. vLLM keeps 96.3% of the cache memory holding real tokens.
> - **In our simulation** on 2,000 Alpaca requests, blocks of 16 slots kept 90.7% of the memory full, against 3.6% when booking 2,048.
> - **The claim:** 2 to 4 times the throughput of FasterTransformer and Orca at the same latency, with exactly the same model outputs.

**Next, in Part 2:** §2 and §3 of the paper. Part 2 reads the attention equations behind the KV cache, the prompt and generation phases, and the batching methods of earlier systems. It then takes Figure 3 apart, slot by slot.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/vllm/vllm_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part1.py). It loads OPT-125M and the OPT-13B configuration file, and measures the KV cache two ways. It times generation with and without a cache, the two phases and a batched decode step. Then it runs the memory simulation on 2,000 Alpaca requests. Every number is saved in `results/part1.json`. On an Apple M-series laptop it takes about two minutes.

```bash
pip install torch transformers datasets
python vllm_part1.py   # downloads facebook/opt-125m (about 250 MB), the OPT-13B config, and tatsu-lab/alpaca
```

<figure><img src="/img/papers/vllm/part1-run.png" alt="Terminal output of vllm_part1.py: the KV cache sizes, the Figure 1 arithmetic, generation with and without a cache, the two phases, the batch timings, the memory simulation and the paper's numbers" loading="lazy" /><figcaption>The full output of vllm_part1.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
