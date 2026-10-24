---
title: "PagedAttention"
description: "How vLLM stores each request's keys and values in small blocks that can sit anywhere in memory, and still computes the same attention. We read §4.1 to §4.3 and build a block table that drives a real model."
part: 3
covers: "§4 intro, §4.1 to §4.3, Figures 4 to 7"
date: 2026-10-24
tags: [llm-serving, pagedattention, kv-cache, gpu-memory, block-table]
---

Part 2 ended with a diagnosis. Earlier systems kept each request's KV cache in one long chunk, booked for the longest possible answer, and most of that memory sat empty. This part reads the paper's cure. There the authors cut the cache into small blocks and scatter the blocks across memory. Then they change attention so that it can still read them.

Each piece below shows the exact lines of the paper, then explains them. We also build the whole mechanism in Python and run a real model through it.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## The problem, with four small requests

Start with a tiny memory that we can draw. It has room for the keys and values of 24 tokens. We call each such place a **slot**. Four requests want to use it: A has 5 tokens so far, B has 2, C has 7 and D has 3.

Before vLLM, a serving system booked one chunk per request, large enough for the longest answer it might write. Say that limit is 8 tokens. Then A, B and C each take a chunk of 8 slots, and the 24 slots are gone. Only 14 slots hold a token. The other 10 are booked for tokens that may never come, and request D cannot start.

vLLM cuts the same memory into 6 blocks of 4 slots. A request gets one block at a time, and a new block only when its last block is full. A needs 2 blocks, B needs 1, C needs 2 and D needs 1. All four requests now run in the same 24 slots.

{{FIG:p3_before_after|The same 24 slots, before and with vLLM. Top: one chunk of 8 slots per request leaves 10 slots booked but empty, and request D must wait. Bottom: blocks of 4, handed out one at a time, hold all four requests. A request's blocks need not be neighbours.}}

Two things changed. First, memory is given when it is needed, not booked at the start. Second, the blocks of one request can sit anywhere: A's two blocks are block 0 and block 3. The rest of this part explains what that costs. Attention must still find every key and value of a request, even when they are scattered.

> [!DEFINITION] Slot
> In this series, a **slot** is the room for the keys and values of one token in the KV cache. A block of 4 slots holds 4 tokens. The size of one slot depends on the model; [Part 1](part-1-the-big-idea.md) measured it.

## The parts of vLLM {§4}

Before the paper's own diagram, here is our redraw of it. It shows the whole engine as one map. The parts that this page explains are highlighted.

{{FIG:p3_map|The vLLM engine, redrawn from the paper's Figure 4. Requests come in on the left. The scheduler in the centre holds the KV cache manager, with the block tables and the block allocators. The workers on the right each hold a cache engine and a shard of the model. This part explains the highlighted pieces.}}

Follow one step of the engine with the four requests above. The step makes one new token for every request that runs.

1. The **scheduler** looks at the waiting and running requests and picks the ones that run in this step.
2. For each picked request, it asks the **KV cache manager** for a slot for the new token. The manager looks at the request's **block table**. If the last block has a free slot, the token goes there. If not, the **GPU block allocator** hands out one free block.
3. The scheduler sends the new tokens and the block tables to the **workers**.
4. Each worker runs the model. Its **cache engine** owns the GPU memory that holds the blocks. Attention reads the earlier keys and values through the block tables, and writes the new ones into the slots it was given.
5. The new tokens go back to the users. When a request ends, its blocks go back to the allocator.

Most of this map is bookkeeping: lists of numbers that say which block belongs to whom. Only step 4 touches the keys and values themselves. That split is the design. The scheduler and the manager only handle numbers; they never move a key. The workers move keys, but never decide where they go.

> [!DEFINITION] Scheduler and worker
> The **scheduler** is the part of a serving engine that decides, at every step, which requests run and which wait. A **worker** is a process that drives one GPU: it runs the model on the inputs that the scheduler sends.

> [!DEFINITION] Block allocator and free list
> A **block allocator** hands out block numbers and takes them back. It keeps a **free list**, the numbers of the blocks that nobody uses. To allocate is to take a number from that list; to free is to put it back.

> [!DEFINITION] Model shard and cache engine
> When a model is too large for one GPU, each worker holds a slice of it, a **model shard**. [Part 4](part-4-sharing-scheduling-and-many-gpus.md) explains the split. The **cache engine** is the part of a worker that owns its GPU's memory for keys and values, cut into blocks.

Now the paper's own words. The first paragraph of §4 names the same pieces.

> [!PAPER] · §4 · page 5
> [![The opening of §4, with new attention algorithm, centralized scheduler, KV cache manager, paged fashion and instructions sent by the highlighted](/img/papers/vllm/p3-method.png)](/img/papers/vllm/p3-method.png)
>
> **What it says:** the authors build a "new attention algorithm", PagedAttention, and a serving engine, vLLM. vLLM uses a "centralized scheduler" to drive the GPU workers. A "KV cache manager" manages the cache "in a paged fashion", through the "instructions sent by the" scheduler.
>
> **What it means:**
>
> There are two inventions here, and they depend on each other. The first is a way of storing memory. The KV cache manager hands out small fixed-size pieces of memory, the way an operating system hands out pages. The second is a way of computing: PagedAttention is an attention kernel that can read keys and values from those scattered pieces.
>
> Neither is useful alone. Scattered storage would break ordinary attention, which expects one tidy tensor. "Centralized" means that there is one scheduler for all GPUs, so every worker follows the same plan in every step.

> [!PAPER] · Figure 4 · page 5
> [![Figures 4 and 5 side by side. Figure 4: a scheduler on top, a KV cache manager with block tables, CPU and GPU block allocators below it, and N workers on the right, each with a cache engine and a model shard. Figure 5: a query vector forth with arrows to three blocks of keys and values](/img/papers/vllm/p3-figure4.png)](/img/papers/vllm/p3-figure4.png)
>
> **What it says:** Figure 4, "vLLM system overview": a scheduler, a KV cache manager that holds the "Block tables", a CPU block allocator and a GPU block allocator, and workers 0 to N − 1, each with a "Cache Engine" and a "Model Shard". (Figure 5, on the right, is the subject of §4.1 below.)
>
> **What it means:**
>
> This is the map that we drew above, in the paper's layout. Read it as a small company. The scheduler is the manager: at every step it decides which requests run. The KV cache manager is the clerk with the ledger. For each request, the ledger lists where that request's memory lives. The two allocators are the store rooms: one hands out blocks of GPU memory, the other blocks of CPU memory for later use.
>
> The workers are the GPUs. Each one holds a shard of the model, and a cache engine that owns that GPU's memory for keys and values. The workers never decide where memory goes; they follow the block tables that the scheduler sends. This part covers the block tables, the GPU allocator and the attention that reads them. Part 4 covers the CPU allocator and the many workers.

## Background: how a computer finds a byte

> [!NOTE] Where this idea comes from: paging
> The block idea is about sixty years old. Early computers had little fast memory, and programs did not fit. The Atlas computer, built at the University of Manchester in the early 1960s, was the first to cut memory into fixed-size **pages**. It kept a table of where each page really was. Every modern operating system still works this way. vLLM copies the idea for the KV cache. The [Background page](background-the-kv-cache-and-paging.md) tells the story in full; here we need only the lookup.

A program sees its memory as one long row of numbered bytes, starting at 0. A byte's number in that row is its **address**. The row is a fiction. The real memory is cut into **frames** of 4,096 bytes each, and the program's row is cut into **pages** of the same size. Any page can sit in any frame.

A **page table** records the frame of each page. Take a small one. Page 0 is in frame 5, page 1 in frame 2, page 2 in frame 7 and page 3 in frame 0. The neighbours in the program's row are not neighbours in real memory.

To find a byte, the computer makes three moves. Take the address 10,000.

1. Split it: 10,000 = 2 × 4,096 + 1,808. So the byte is in page 2, at **offset** 1,808 inside the page.
2. Look up page 2 in the page table: frame 7.
3. Put it back together: 7 × 4,096 + 1,808 = 30,480. That is the real address.

A street address works the same way. "House 8 on Elm Street" is two parts: the street, which a map turns into a place, and the house number, which you count along the street. The split costs one look at the map. In return, the streets can be anywhere.

> [!DEFINITION] Page, frame and offset
> A **page** is a fixed-size piece of a program's memory as the program sees it. A **frame** is a piece of real memory of the same size. The **offset** is the position of a byte inside its page; it stays the same when the page moves to a frame.

vLLM does exactly this with tokens. A token's position plays the role of the address. A KV block plays the role of a page, and the block table plays the role of the page table. We draw the two lookups side by side in §4.2 below, after the paper has named the parts.

## Background: what an attention kernel is

A program that uses a GPU runs on the CPU. When it needs heavy maths, it asks the GPU to run a small function on thousands of threads at once. That function is a **kernel**. Matrix products, softmax and attention are each one or a few kernels.

A kernel gets its data as an address in GPU memory plus a shape, such as "10 keys of 768 numbers each, starting here". From that it computes where every number is. Key $$t$$ starts at the start address plus $$t$$ times the size of one key. One multiplication finds any key. This only works if the keys sit one after another in one unbroken run of memory, a **contiguous tensor**.

That is why earlier systems kept each request's KV cache in one chunk. The attention kernels they used, such as FasterTransformer's, expect the keys and values of each request in one contiguous run of memory. A tensor cannot grow into memory that another request already uses. So the chunk had to be booked for the longest answer at the start.

{{FIG:p3_kernel|What the kernel has to know. Top: an ordinary attention kernel reads the keys of a request from one contiguous row, so a single multiplication finds key t. Bottom: with blocks, the same keys sit in physical blocks 7, 1 and 3 of a pool; the kernel must look up the block table first, then find the slot inside the block.}}

With blocks, the simple formula breaks. Key 9 is not 9 keys after key 0; it is in some block, wherever that block happens to be. There are two ways out. One is to copy the blocks into one contiguous tensor before each attention call, which costs a full copy of the cache at every step. The other is a new kernel that takes the block table as an extra input and follows it. PagedAttention is the second way.

> [!NOTE] Where this idea comes from: GPU kernels for Transformers
> FasterTransformer is a library from NVIDIA with hand-written GPU kernels for running Transformers fast. Its attention kernel expects the keys and values of a request in one tensor. The paper says in §5.1 that vLLM's PagedAttention kernel started as a copy of it. The authors changed it "to read KV cache according to the block table". Part 5 compares the two.

> [!DEFINITION] GPU kernel and contiguous tensor
> A **GPU kernel** is a function that runs on the GPU, on thousands of threads at once. A **tensor** is a grid of numbers, such as a matrix. It is **contiguous** when its numbers fill one unbroken run of memory, in order, so that the position of any number follows from its index.

## PagedAttention {§4.1}

> [!PAPER] · §4.1 · page 5
> [![The opening of §4.1, with of paging, continuous keys and values in non-contiguous memory space, KV blocks and vectors for a fixed number of highlighted](/img/papers/vllm/p3-s41.png)](/img/papers/vllm/p3-s41.png)
>
> **What it says:** PagedAttention is an attention algorithm inspired by the classic idea "of paging" in operating systems. It allows storing "continuous keys and values in non-contiguous memory space". It splits the cache of each sequence into "KV blocks", each holding the key and value "vectors for a fixed number of" tokens.
>
> **What it means:**
>
> Look at the two words "continuous" and "non-contiguous". The keys and values of a sequence are continuous in meaning: token 1, token 2, token 3, in order. They do not need to be contiguous in memory: token 5 can sit far from token 4. The order lives in a table, not in the addresses.
>
> A book in a library is a good picture. A long novel can be printed in four volumes that stand on four different shelves. The story still runs from volume 1 to volume 4, because the catalogue tells you where each volume is. The volumes are the KV blocks. The catalogue is the block table.

> [!DEFINITION] KV block and block size
> A **KV block** is a fixed-size piece of memory that holds the keys and values of a fixed number of tokens. That number is the **block size**, written $$B$$. The figures of the paper use $$B = 4$$; vLLM uses $$B = 16$$ by default.

The mapping from the background above is short: blocks for pages, token positions for byte addresses, requests for programs. Here we need one extra fact. A key and a value exist for every token in every layer and every attention head. The footnote says how vLLM groups them.

> [!PAPER] · §4.1, footnote 1 · page 5
> [![Footnote 1, with separate block, no performance difference and choose the second one highlighted](/img/papers/vllm/p3-footnote.png)](/img/papers/vllm/p3-footnote.png)
>
> **What it says:** all keys and values of a token, across layers and heads, could live in one block. Or the keys and values of each head and layer could each have a "separate block" and block table. The two designs have "no performance difference", and the authors "choose the second one" for easy implementation.
>
> **What it means:**
>
> OPT-125M has 12 layers with 12 heads each, so every token makes 144 keys and 144 values. A block of 4 tokens can be cut along the layers or not. Either way, a block covers the same 4 token positions, and the same rule decides when a new block is needed. Only the bookkeeping differs, which is why the speed does not.
>
> Our own code keeps one pool of blocks per layer, and each slot of a block holds all 12 heads of one token. The block numbers are shared by all layers.

## Eq. 4: attention, block by block {§4.1}

Attention, as the [Background page](background-the-kv-cache-and-paging.md) worked it, has three moves. The query of the new token gets a score against every earlier key. A **softmax** turns the scores into weights: take the exponential of each score, then divide each one by the sum of all of them. The output mixes the values with those weights. Eq. 4 writes the same three moves for keys and values in blocks.

> [!PAPER] · §4.1, Eq. 4 · page 5
> [![The block definitions and Equation 4, with block size, block wise computation and row vector of attention score on highlighted](/img/papers/vllm/p3-eq4.png)](/img/papers/vllm/p3-eq4.png)
>
> **What it says:** with "block size" $$B$$, the key block $$K_j$$ holds the keys of tokens $$(j-1)B+1$$ to $$jB$$, and the value block $$V_j$$ their values. The attention of Part 2 can be rewritten as a "blockwise computation" (Eq. 4). $$A_{ij}$$ is the "row vector of attention score on" the $$j$$-th block.
>
> **What it means:**
>
> Attention adds up over all earlier tokens, and addition does not care how you group the terms. You can add 1 + 2 + 3 + 4 + 5 + 6 as (1 + 2 + 3) + (4 + 5 + 6). Eq. 4 is the same attention as before, with the sums grouped by block. Nothing about the result changes. Only the way the computer reaches the numbers changes: one block at a time.
>
> The grouping has a catch, and Eq. 4 shows it. The softmax needs one denominator for all tokens, so the scores of every block must be known before any weight is final. A block cannot be finished on its own.

Here is Eq. 4, written with the bold $$\mathbf{1}$$ of the paper:

$$
A_{ij} = \frac{\exp\left(q_i^\top K_j / \sqrt{d}\right)}{\sum_{t=1}^{\lceil i/B \rceil} \exp\left(q_i^\top K_t / \sqrt{d}\right) \mathbf{1}}, \qquad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top
$$

Read it from left to right. The query $$q_i$$ belongs to the token being written, token $$i$$. The key block $$K_j$$ is a small matrix: the keys of the $$B$$ tokens in block $$j$$. So $$q_i^\top K_j$$ is not one number but a row of $$B$$ numbers, one score per token of the block. Dividing by $$\sqrt{d}$$ and taking the exponential works on each number of the row.

The denominator adds up those exponentials over all blocks $$t$$ from 1 to $$\lceil i/B \rceil$$. The bold $$\mathbf{1}$$ is a column of ones: multiplying a row by it adds up the row. The symbol $$\lceil i/B \rceil$$ means "$$i/B$$ rounded up", the number of blocks that token $$i$$ needs. With $$B = 4$$ and $$i = 10$$, that is 3 blocks.

The result $$A_{ij}$$ is the row of attention weights for the tokens of block $$j$$. On the right, $$V_j A_{ij}^\top$$ mixes the values of block $$j$$ with those weights. The output $$o_i$$ adds up the mixes of all blocks. (The paper's text calls the attention it rewrites "Eq. 4" too; it means the attention equation of Part 2.)

> [!DEFINITION] Ceiling
> $$\lceil x \rceil$$, "the ceiling of $$x$$", is the smallest whole number that is not less than $$x$$. $$\lceil 2.5 \rceil = 3$$ and $$\lceil 3 \rceil = 3$$. Seven tokens in blocks of 4 need $$\lceil 7/4 \rceil = 2$$ blocks.

### Why splitting the softmax gives the exact answer

A softmax over blocks sounds risky. Each weight depends on every score, and the scores now live in different places. The trick is that the denominator is only a sum. Each block can add up its own exponentials, and the block sums can then be added together. The total is the same number that one long sum would give, because addition does not care about grouping.

A tiny example you can check by hand. The [Background page](background-the-kv-cache-and-paging.md) worked attention for a query $$(1, 0)$$ and three tokens with keys $$(1, 0)$$, $$(0, 1)$$ and $$(1, 1)$$ and values $$(1, 0)$$, $$(0, 1)$$ and $$(1, 1)$$. The weights were 0.401, 0.198 and 0.401, and the output $$(0.802, 0.599)$$. Now store the same tokens in blocks of $$B = 2$$. Block 1 holds tokens 1 and 2, block 2 holds token 3, with one empty slot.

Our script does Eq. 4 on these numbers, one block at a time, and prints every middle result:

```text
== 9. Eq. 4 by hand: the Background example (query (1, 0), three tokens) in blocks of 2
block 1: scores 0.707, 0.000; exp 2.028, 1.000; block sum 3.028; weights 0.401, 0.198; adds (0.401, 0.198)
block 2: scores 0.707; exp 2.028; block sum 2.028; weights 0.401; adds (0.401, 0.401)
shared denominator 3.028 + 2.028 = 5.056; output (0.802, 0.599); torch softmax over all three tokens: (0.802, 0.599)
```

{{FIG:p3_softmax|Eq. 4 in three moves, on the Background example. First, each block scores its own tokens and adds up its own exponentials. Second, one addition joins the block sums into the shared denominator. Third, each block divides by it and adds its share of the output.}}

Read the picture from left to right. The scores are $$q \cdot k / \sqrt{2}$$: 0.707, 0 and 0.707. Each block adds its own exponentials: 3.028 for block 1 and 2.028 for block 2. One addition gives the shared denominator, 5.056. Each block then divides its exponentials by 5.056 to get its weights, and mixes its own values.

Block 1 contributes $$0.401 \cdot (1, 0) + 0.198 \cdot (0, 1) = (0.401, 0.198)$$. Block 2 contributes $$0.401 \cdot (1, 1) = (0.401, 0.401)$$. The sum is $$(0.802, 0.599)$$, the same output as PyTorch's softmax over all three tokens at once.

The only step that needs all blocks is the one addition in the middle. Everything else is local to a block. That is the "catch" of Eq. 4 above, and it is a small one: each block passes one number to the others, not its keys.

### The paper's picture of it

> [!PAPER] · Figure 5 · page 5
> [![Figure 5: a query vector forth with arrows to block 1 holding years ago our fathers, block 2 holding brought forth, and block 0 holding Four score and seven, with non-contiguous blocks in the memory highlighted](/img/papers/vllm/p3-figure5.png)](/img/papers/vllm/p3-figure5.png)
>
> **What it says:** Figure 5 illustrates PagedAttention: the key and value vectors "are stored as non-contiguous blocks in the memory".
>
> **What it means:**
>
> The model is writing the word "forth". Its query must meet the keys of all ten tokens so far. They sit in three blocks: "Four score and seven" in block 0, "years ago our fathers" in block 1, "brought forth" in block 2. In memory the blocks are out of order: block 1 is drawn on top and block 0 at the bottom, with empty rows between them.
>
> Block 2 has two empty slots. They are booked for the next two tokens of this request. That is the only memory this request holds and does not use.

> [!PAPER] · §4.1 · page 5
> [![The paragraph on the PagedAttention kernel, with fetches different KV blocks separately, not contiguous on the physical memory, multiplies the query vector and derive the final attention output highlighted](/img/papers/vllm/p3-kernel.png)](/img/papers/vllm/p3-kernel.png)
>
> **What it says:** the PagedAttention kernel "fetches different KV blocks separately". The three blocks of Figure 5 are "not contiguous on the physical memory". The kernel "multiplies the query vector" with the keys of one block at a time to get $$A_{ij}$$, then multiplies $$A_{ij}$$ with the values of that block to "derive the final attention output" $$o_i$$.
>
> **What it means:**
>
> This is Eq. 4 as a procedure. For each block, look up where it is, load its keys, and score them against the query. Then turn all scores into weights with one softmax. Last, load each block's values and add them up with their weights.
>
> The kernel never needs the blocks to be neighbours. It needs only a list of where they are. That list is the request's block table, which §4.2 introduces.

We wrote that procedure in Python. This is Method 1, our own PagedAttention, verbatim from the script. Look for the two loops over the block table. The block table is a list of pairs: the physical block number and how many slots of it are filled.

```python
def paged_attention(q, cache, layer, table):
    """Eq. 4 for one query token. q: (H, d). table: [[physical block, # filled], ...] in logical order.
    Pass 1: the scores of each KV block. Pass 2: each block's values, weighted by its share of the softmax."""
    scale = 1 / math.sqrt(DH)
    scores = []
    for phys, filled in table:                                  # logical block j lives in physical block `phys`
        K_j = cache.K[layer, phys, :filled]                     # (filled, H, d): fetched from wherever it sits
        scores.append(torch.einsum('hd,thd->ht', q, K_j) * scale)          # q_i^T K_j / sqrt(d), (H, filled)
    m = torch.stack([s.max(-1).values for s in scores]).max(0).values      # subtract the max: same result, no overflow
    exps = [torch.exp(s - m[:, None]) for s in scores]
    denom = sum(e.sum(-1) for e in exps)                        # the sum over all ceil(i / B) blocks, (H,)
    o = torch.zeros(H, DH)
    for (phys, filled), e in zip(table, exps):
        A_ij = e / denom[:, None]                               # attention on block j, (H, filled)
        o += torch.einsum('ht,thd->hd', A_ij, cache.V[layer, phys, :filled])   # V_j A_ij^T
    return o
```

The first loop computes $$q_i^\top K_j / \sqrt{d}$$ for every block. The line with `denom` is the one addition of the picture above. The line with `m` subtracts the largest score before the exponential. That changes no weight, because the same factor cancels above and below the fraction. It only keeps the exponentials from overflowing.

The second loop computes $$V_j A_{ij}^\top$$ and adds it up. The `H` dimension runs over the 12 heads of OPT-125M, each with its own query.

> [!DEFINITION] einsum
> A compact way to write sums of products in PyTorch. `'hd,thd->ht'` says: for each head `h` and token `t`, multiply along the shared size `d` and add up. Here it is the dot product of each head's query with each key of the block.

We ran OPT-125M on the paper's own sentence, with blocks of 4, and looked at the query of "forth" in layer 6. The model has 12 heads; we print the weight of each token averaged over them. Check that the three block totals add up to 1:

```text
== 2. Eq. 4: the query of "forth" against KV blocks 0, 1 and 2 (layer 6, mean over the 12 heads)
block 0 (physical 7): Four 0.731, score 0.013, and 0.009, seven 0.012   block total 0.765
block 1 (physical 1): years 0.020, ago 0.024, our 0.032, fathers 0.041   block total 0.118
block 2 (physical 3): brought 0.074, forth 0.044   block total 0.118
all ten weights add up to 1.000000
a softmax inside block 0 alone would give 0.948, 0.022, 0.013, 0.017: the denominator must run over every block
```

{{FIG:p3_eq4|Eq. 4 on a real model. The query of "forth" meets three blocks stored in physical blocks 7, 1 and 3. Each slot is shaded by its attention weight. One softmax covers all ten slots, so the three block totals add up to 1.}}

The three block totals, 0.765, 0.118 and 0.118, add up to 1. That is the shared denominator at work. The last line shows what goes wrong without it. A softmax inside block 0 alone would give "Four" 0.948 instead of 0.731, because it ignores the six tokens in the other blocks.

The large weight on "Four" is not about its meaning. Many language models park much of their attention on the first token of a text. It is a habit learned in training, and OPT-125M shows it here in layer 6. For this part it is just a number that the paged and the contiguous computation must both reproduce.

## Does paged attention give the same answer? {§4.1}

The paper's claim is that PagedAttention changes where keys and values live, not what attention computes. We test that claim in three ways.

Method 2 uses the library. We run the same ten tokens through the model with its own cache, `past_key_values`, which holds each layer's keys and values in one contiguous tensor. Then we call PyTorch's built-in `scaled_dot_product_attention` (SDPA) with the query of "forth". We compare the result with our paged output in each of the 12 layers:

```python
    K, V = kv.layers[l].keys, kv.layers[l].values
    o_lib = F.scaled_dot_product_attention(q[None, :, None], K, V)[0, :, 0]   # (H, d), one query, all 10 keys
    diffs.append((o_lib - o_paged).abs().max().item())
```

```text
== 3. The same attention with the library: past_key_values + torch scaled_dot_product_attention
attention output of "forth", paged (Method 1) against SDPA on the contiguous cache (Method 2), 12 layers:
  max difference 7.7e-07 (layer 10), smallest 9.5e-08
logits after "forth": our paged engine against the library, max difference 1.8e-05; same top token: True
```

The largest difference in any layer is $$7.7 \times 10^{-7}$$. That is rounding: 32-bit numbers keep about 7 significant digits, and the two methods add the same terms in a different order. The final scores of the model for the next token, its **logits**, differ by at most $$1.8 \times 10^{-5}$$, and both pick the same next token.

> [!DEFINITION] Logits
> The raw scores that a language model gives every token of its vocabulary as the next token. A softmax turns them into probabilities. The token with the largest logit is the most likely one.

The second test is harder. We let our paged engine write 48 tokens on its own, picking the most likely token each time. We compare with the library's `model.generate`. To make sure that order in memory does not matter, the free blocks are handed out in a random order:

```text
== 4. Greedy generation of 48 tokens, our paged engine against model.generate (random free-block order)
prompt 10 tokens; library text: " to learn a new language.\nI'm not sure if you're being sarcastic or no"...
B =  1: 57 blocks, first physical blocks [25, 61, 35, 9, 2, 5]; same 48 tokens: True; max logit difference 1.6e-05
B =  4: 15 blocks, first physical blocks [16, 5, 14, 13, 11, 10]; same 48 tokens: True; max logit difference 1.6e-05
B = 16:  4 blocks, first physical blocks [4, 1, 0, 7]; same 48 tokens: True; max logit difference 1.6e-05
```

With blocks of 1, 4 and 16 tokens, scattered over memory in a random order, the engine writes the same 48 tokens as the library. The logits never differ by more than $$1.6 \times 10^{-5}$$. This is what Part 1 called "without affecting the model accuracy at all". The block layout is invisible to the model.

The third test pushes the softmax argument to its limit. Our `paged_attention` still looks at all scores before it divides. Can each block be finished completely alone, and the results joined later? Yes, if each block keeps three things: its largest score, its own sum of exponentials, and its own unweighted mix of values. Joining them needs only a rescale to a common largest score:

```python
def merged_attention(q, cache, layer, table):
    """Each block alone keeps three things: its largest score m_j, its sum s_j of exp(score - m_j) and its
    unnormalised output u_j. Merging rescales every block to the largest m and divides once at the end."""
    scale = 1 / math.sqrt(DH)
    parts = []
    for phys, filled in table:
        s = torch.einsum('hd,thd->ht', q, cache.K[layer, phys, :filled]) * scale       # (H, filled)
        m = s.max(-1).values                                                           # (H,)
        e = torch.exp(s - m[:, None])
        parts.append((m, e.sum(-1), torch.einsum('ht,thd->hd', e, cache.V[layer, phys, :filled])))
    M = torch.stack([m for m, _, _ in parts]).max(0).values
    S = sum(s_ * torch.exp(m - M) for m, s_, _ in parts)                               # the shared denominator
    U = sum(u * torch.exp(m - M)[:, None] for m, _, u in parts)
    return U / S[:, None]
```

```text
blocks finished one at a time and merged at the end, query of "forth", 12 layers of OPT-125M:
  merged against SDPA on the contiguous cache: max difference 7.2e-07
```

The merged result matches SDPA to $$7.2 \times 10^{-7}$$, the same rounding as before. So the order of work is free. A GPU can give each block to a different group of threads, and join the pieces at the end. The paper does not describe this merge; we add it to show why the blockwise sum is safe.

## The KV cache manager {§4.2}

> [!PAPER] · §4.2 · page 5
> [![The opening of §4.2, with virtual memory, logical pages to physical pages, non-contiguous physical memory pages, needs not to be fully reserved in advance and fixed-size KV blocks, like pages in virtual memory highlighted](/img/papers/vllm/p3-vm.png)](/img/papers/vllm/p3-vm.png)
>
> **What it says:** the memory manager is analogous to "virtual memory" in operating systems, which maps a program's "logical pages to physical pages". Contiguous logical pages can sit in "non-contiguous physical memory pages", and physical memory "needs not to be fully reserved in advance". vLLM organizes the KV cache as "fixed-size KV blocks, like pages in virtual memory".
>
> **What it means:**
>
> This is the page table of the background section above. The paragraph names the two gifts of paging that vLLM wants. The first is freedom of place: a request sees its cache as one row, but the pieces sit anywhere. The second is freedom of time: memory is given when it is needed, not booked at the start.
>
> Part 2 showed what the lack of each gift cost. Without freedom of place, gaps between chunks were external fragmentation. Without freedom of time, the slots booked for future tokens were reserved memory, and the slots never reached were internal fragmentation.

> [!PAPER] · §4.2 · page 5
> [![The paragraph on logical KV blocks, with logical KV blocks, filled from left to right, unfilled positions are reserved for future generations, block engine and contiguous chunk of GPU DRAM highlighted](/img/papers/vllm/p3-logical.png)](/img/papers/vllm/p3-logical.png)
>
> **What it says:** a request's cache is a series of "logical KV blocks", "filled from left to right" as tokens arrive. The last block's "unfilled positions are reserved for future generations". On each GPU, a "block engine" allocates a "contiguous chunk of GPU DRAM" and divides it into physical blocks (continued on the next page).
>
> **What it means:**
>
> A **logical block** is a block as the request sees it. Block 0 holds its tokens 1 to 4, block 1 its tokens 5 to 8, and so on, always in order. A **physical block** is a real piece of GPU memory, with a number. The two are different things, and keeping them apart is the whole trick.
>
> Note the twist in the last sentence. vLLM does allocate one large contiguous chunk, once, when the server starts. It then cuts that chunk into equal physical blocks and never asks the GPU for memory again. The fragmentation problem of Part 2 came from allocating one chunk per request. One chunk for the whole server, cut into equal blocks, has no such problem.

> [!DEFINITION] DRAM
> Dynamic random-access memory, the main memory of a computer or a GPU. On a GPU it is the memory that holds the weights and the KV cache: 40 or 80 GB on an A100.

> [!PAPER] · §4.2 · page 6
> [![The end of §4.2, with physical KV blocks, block tables, number of filled positions and without reserving it for all positions in advance highlighted](/img/papers/vllm/p3-block-table.png)](/img/papers/vllm/p3-block-table.png)
>
> **What it says:** the KV block manager keeps "block tables", the mapping between logical and "physical KV blocks" of each request. Each entry records the physical block of a logical block and the "number of filled positions". This lets vLLM grow the cache "without reserving it for all positions in advance".
>
> **What it means:**
>
> A block table is a short list, one row per logical block. Each row has two numbers: where the block is, and how full it is. A request of 1,000 tokens in blocks of 16 needs a table of 63 rows. That table is all the bookkeeping that paging costs.
>
> The "number of filled positions" matters for two reasons. Attention must read only the filled slots of the last block, not the empty ones. And the manager must know when the last block is full, because that is the moment to map a new one.

> [!DEFINITION] Block table
> The list that maps each logical block of a request to a physical block of GPU memory, with the number of filled slots in each. It plays the role of an operating system's page table.

### Finding one token: the page-table lookup again

Now we can put the two lookups side by side. On the left, the operating system finds byte 10,000. On the right, vLLM finds the key of token 9, "forth", for request A of the paper's Figure 6. Its block table sends logical blocks 0, 1 and 2 to physical blocks 7, 1 and 3. We count positions from 0, as the code does.

{{FIG:p3_lookup|The same lookup twice. Left: address 10,000 splits into page 2 and offset 1,808, the page table sends page 2 to frame 7, and the byte is at 30,480. Right: token 9 splits into logical block 2 and offset 1, the block table sends logical block 2 to physical block 3, and the token is in slot 13 of the pool.}}

The three moves are the same. Split the position by the block size: 9 = 2 × 4 + 1, so token 9 is in logical block 2 at offset 1. Look up logical block 2 in the block table: physical block 3. Put it back together: 3 × 4 + 1 = 13, the slot number if the pool is seen as one long row.

Our script does the lookup with one small function. We call it once with the page table of the background section and once with request A's block table. Then we check the key and value that it finds against the library's contiguous cache, for all 10 tokens in all 12 layers:

```python
def locate(t, table, B):
    """The address split of paging. t: a position (a token, or a byte address). table: physical number of each
    logical block (or page). Returns (logical block, offset, physical block, slot in the pool as one long row)."""
    j, off = t // B, t % B                                      # which block, and where inside it
    phys = table[j]                                             # one look into the table
    return j, off, phys, phys * B + off
```

```text
== 8. Finding a token through the block table, like an address through a page table
the operating system (Background page): page size 4,096 bytes, page table [5, 2, 7, 0]
  address 10,000 -> page 2, offset 1,808 -> frame 7 -> physical address 30,480
vLLM (request A of Figure 6): block size 4, block table [7, 1, 3]
  token 0     'Four' -> logical block 0, offset 0 -> physical block 7 -> slot 28 of the pool
  token 4    'years' -> logical block 1, offset 0 -> physical block 1 -> slot  4 of the pool
  token 7  'fathers' -> logical block 1, offset 3 -> physical block 1 -> slot  7 of the pool
  token 8  'brought' -> logical block 2, offset 0 -> physical block 3 -> slot 12 of the pool
  token 9    'forth' -> logical block 2, offset 1 -> physical block 3 -> slot 13 of the pool
keys and values found this way against the library's contiguous cache, 10 tokens x 12 layers: max difference 8.6e-06
```

One function serves both worlds. The same three lines turn address 10,000 into 30,480 and token 9 into slot 13. The keys found through the table match the library's keys to $$8.6 \times 10^{-6}$$. They are not equal to the last bit. Our engine computed them a step at a time, and the library computed all ten in one pass. So the additions ran in a different order.

Neighbours in the request are not neighbours in memory. Tokens 7 and 8 are next to each other in the sentence, but they sit in slots 7 and 12. That is the price of paging, and the reason the kernel needs the table.

### The manager in code

Here is our KV cache manager, the `PagedKVCache` class of the script. It owns the physical blocks of every layer, a free list, and one block table per request. The first block shows the allocator. The second shows the one important method, `append_slot`, which finds the slot for the next token:

```python
class BlockAllocator:
    """Hands out physical block numbers from a free list, and takes them back."""
    def __init__(self, order):
        self.free = list(order)

    def allocate(self):
        if not self.free:
            raise MemoryError('no free KV block')
        return self.free.pop(0)

    def release(self, block):
        self.free.append(block)
```

```python
    def append_slot(self, req):
        """Reserve the slot for one new token: the next free position of the last logical block,
        or a new physical block when the last one is full. Returns (physical block, offset)."""
        table = self.tables.setdefault(req, [])
        if not table or table[-1][1] == self.B:                 # last block full (or none yet): map a new one
            table.append([self.alloc.allocate(), 0])
        phys, filled = table[-1]
        table[-1][1] += 1                                       # the "# filled" column of the block table
        return phys, filled
```

Six lines of logic carry the whole policy of §4.2. A new physical block is taken from the free list only when the last block is full. The keys and values themselves live in two tensors of shape (layers, physical blocks, $$B$$, heads, head size). The model writes each new token's key and value at the slot that `append_slot` returned.

## Decoding, step by step: Figure 6 {§4.3}

> [!PAPER] · Figure 6 · page 6
> [![Figures 6 and 7 side by side. Figure 6: request A with the prompt Four score and seven years ago our, its logical KV blocks, a block table with physical block numbers 7, 1, 3 and filled counts 4, 3 to 4, 1, and nine physical blocks on GPU DRAM. Figure 7 on the right shows two requests in one set of physical blocks](/img/papers/vllm/p3-figure6.png)](/img/papers/vllm/p3-figure6.png)
>
> **What it says:** Figure 6, "Block table translation in vLLM": request A with the prompt "Four score and seven years ago our" and the outputs "fathers", "brought". On the left, its logical KV blocks. In the middle, the block table: physical block 7 with 4 filled, physical block 1 with "3 → 4" filled, physical block 3 with 1 filled. On the right, nine physical KV blocks "on GPU DRAM". (Figure 7, on the right of the crop, comes at the end of this part.)
>
> **What it means:**
>
> Follow the arrows. Logical block 0, "Four score and seven", is stored in physical block 7. Logical block 1, "years ago our fathers", is in physical block 1. Logical block 2, "brought", is in physical block 3. Physical blocks 0, 2, 4, 5, 6 and 8 are free for other requests.
>
> The circled numbers 1, 2 and 3 mark the order in which the slots were filled. The text of §4.3 walks through them. We do the same with our code and a real model.

> [!PAPER] · §4.3 · page 6
> [![The first step of the walk-through, with does not require reserving the memory for the maximum possible generated, reserves only the necessary KV blocks, 7 and 1, respectively, conventional self-attention and The remaining slot is reserved highlighted](/img/papers/vllm/p3-step1.png)](/img/papers/vllm/p3-step1.png)
>
> **What it says:** step 1: vLLM "does not require reserving the memory for the maximum possible generated" length. It "reserves only the necessary KV blocks" for the prompt. The prompt has 7 tokens, so logical blocks 0 and 1 map to physical blocks "7 and 1, respectively". The prefill step uses a "conventional self-attention" algorithm. "The remaining slot is reserved" for the generation phase.
>
> **What it means:**
>
> Seven tokens need $$\lceil 7/4 \rceil = 2$$ blocks: four tokens in the first, three in the second. The request now holds 8 slots for 7 tokens. Compare that with Part 2's request A, which held 2,048 slots from the start.
>
> The prefill step reads the whole prompt at once, so it does not need PagedAttention. It computes attention the ordinary way, over keys and values it has just made. On the side, it writes those keys and values into the blocks, where the later steps will find them.

> [!DEFINITION] Prefill
> Another name for the prompt phase of Part 2. It is the first pass: it reads all prompt tokens at once and fills the KV cache with their keys and values.

> [!PAPER] · §4.3 · page 6
> [![Steps 2 and 3 of the walk-through, with PagedAttention algorithm on physical blocks 7 and 1, #filled record is updated, allocates a new physical block (physical block 3) highlighted](/img/papers/vllm/p3-step23.png)](/img/papers/vllm/p3-step23.png)
>
> **What it says:** step 2: the first decoding step runs the "PagedAttention algorithm on physical blocks 7 and" 1. One slot is still free in the last logical block, so the new token's KV cache goes there, and the block table's "#filled record is updated". Step 3: the last block is full, so vLLM "allocates a new physical block (physical block 3)" and records it in the block table.
>
> **What it means:**
>
> Step 2 is the cheap case: the new token fits in a block the request already has. The only change in the ledger is one number, 3 becomes 4. Step 3 is the expensive case, and it is still cheap: take one block from the free list and add one row to the table.
>
> This is memory "on demand". At no point does the request hold more than one block it has not filled. The next new block is mapped only at the step that needs it.

We ran this walk-through on OPT-125M. The prompt is the paper's own, exactly 7 tokens in the OPT tokenizer. The model then gets the paper's outputs "fathers", "brought" and "forth", one per step. We feed those words ourselves, because the small model would write something else (its own first choice is "team"). Our free list starts with blocks 7, 1 and 3, so the numbers match the figure. The core of each step is the start of our `forward` function:

```python
    n = len(ids)
    slots = [cache.append_slot(req) for _ in range(n)]          # the KV cache manager reserves the slots first
```

Then, in each layer, the new keys and values go into those slots. A step with many tokens uses ordinary attention; a step with one token uses `paged_attention`:

```python
        for t, (phys, off) in enumerate(slots):                 # write the new keys and values into their blocks
            cache.K[l, phys, off], cache.V[l, phys, off] = k[t], v[t]
        if n > 1:
            o = causal_attention(q, k, v)
        else:
            o = paged_attention(q[0], cache, l, cache.tables[req])[None]
```

In the output, watch the block table grow by one row only at "brought". At the end, the script prints what sits in each of the 9 physical blocks:

```text
== 1. Figure 6: the block table of request A, step by step (block size 4, real OPT-125M)
prompt: 7 tokens ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our']; we then feed the outputs "fathers", "brought", "forth"
prefill  (7 tokens): logical 0 -> physical 7 (4 filled)  logical 1 -> physical 1 (3 filled)
          model's top next token: ' team'
decode 1 ('fathers'): logical 0 -> physical 7 (4 filled)  logical 1 -> physical 1 (4 filled)
decode 2 ('brought'): logical 0 -> physical 7 (4 filled)  logical 1 -> physical 1 (4 filled)  logical 2 -> physical 3 (1 filled)
decode 3 (  'forth'): logical 0 -> physical 7 (4 filled)  logical 1 -> physical 1 (4 filled)  logical 2 -> physical 3 (2 filled)
physical blocks now (9 blocks of 4 slots; "." = free slot, "-" = free block):
  block 0:        -
  block 1:    years      ago      our  fathers
  block 2:        -
  block 3:  brought    forth        .        .
  block 4:        -
  block 5:        -
  block 6:        -
  block 7:     Four    score      and    seven
  block 8:        -
```

{{FIG:p3_fig6_steps|The walk-through of Figure 6, from our run. Each row is one step: the block table in the middle and the nine physical blocks on the right. The table row that changed and the slot written in that step are highlighted. Only "brought" needs a new physical block.}}

Read the picture one row at a time. After the prefill, the table has two rows and the memory has 7 filled slots, in blocks 7 and 1. "fathers" changes one number in the table, 3 to 4, and fills the last slot of block 1. "brought" finds no free slot, so the table gets a third row and block 3 appears in memory. "forth" again changes only a number.

The block table after "brought" is the paper's table, row for row: 7 with 4 filled, 1 with 4 filled, 3 with 1 filled. We went one step further, to "forth", which fills a second slot of block 3. That is the state of Figure 5, and the step whose attention we checked against the library above. The physical blocks hold the words exactly where Figure 6 draws them.

### One step for many requests {§4.3}

> [!NOTE] Where this idea comes from: Orca
> Orca, a serving system from Yu and colleagues at OSDI 2022, introduced iteration-level scheduling. After every model step, finished requests leave the batch and new ones join. [Part 2](part-2-why-memory-is-the-bottleneck.md) explained it. Orca still booked a contiguous chunk per request. vLLM keeps Orca's step-by-step batching and gives it blocks.

> [!PAPER] · §4.3 · page 6
> [![The paragraph on each decoding iteration, with allocates the physical blocks for the newly required logical blocks and concatenates all the input tokens highlighted](/img/papers/vllm/p3-iteration.png)](/img/papers/vllm/p3-iteration.png)
> [![The continuation in the next column, with as one sequence and uses the PagedAttention kernel to access highlighted](/img/papers/vllm/p3-iteration-2.png)](/img/papers/vllm/p3-iteration-2.png)
>
> **What it says:** in each decoding iteration, vLLM selects the sequences to batch, then "allocates the physical blocks for the newly required logical blocks". It "concatenates all the input tokens" of the iteration, all prompt tokens of new requests and the latest token of running ones, "as one sequence" and feeds it to the model. The model "uses the PagedAttention kernel to access" the earlier KV cache in its blocks.
>
> **What it means:**
>
> This is iteration-level batching from Part 2, now with blocks. Before the step, the manager makes sure every running request has a slot for its next token. Then all the new tokens go through the model together, in one long row. The work that each token does alone, such as the feed-forward layers, runs on the whole row at once and uses the GPU well.
>
> Only attention is per request. Each token's query must meet its own request's keys and values, which sit in that request's blocks. The PagedAttention kernel gets every request's block table and follows each one.

Take the four requests from the start of this page. In one step, A, B, C and D each add one token. The model gets a row of 4 tokens and runs its big matrix products once for all of them. In attention, A's token reads A's blocks through A's table, and D's token reads D's blocks through D's table. The tables keep the requests apart, even though their blocks are mixed in one pool.

### The block size {§4.3}

> [!PAPER] · §4.3 · page 6
> [![The sentences on block size, with more positions in parallel and larger block size also increases memory fragmentation highlighted](/img/papers/vllm/p3-blocksize.png)](/img/papers/vllm/p3-blocksize.png)
>
> **What it says:** more than one token per block lets the kernel process the KV cache across "more positions in parallel", which raises hardware use and lowers latency. But "a larger block size also increases memory fragmentation". §7.2 studies the effect.
>
> **What it means:**
>
> There are two forces, and they pull in opposite ways. A GPU is fast when it can load a large piece of memory in one go and work on many numbers at once. Large blocks give it that. Each block is also one more lookup in the table and one more separate load. So the kernel wants large blocks.
>
> Memory wants small blocks. The last block of a request is, on average, half empty, and a larger block makes that half larger. Part 5 reads the paper's measurements; vLLM chose 16 tokens as its default.

> [!DEFINITION] Latency and throughput
> **Latency** is the time one request waits for its answer. **Throughput** is how much work the server finishes per second, for example requests or tokens per second. A server can raise throughput by running more requests at once, which is what free memory allows.

We can see both forces on a laptop. For the memory side, we take 2,000 real requests from the Alpaca dataset, as in Parts 1 and 2. We count the slots held at the end of each request. For the speed side, we time our own `paged_attention` for one query over 2,048 keys, with the keys cut into blocks of different sizes. In the output, look at how the time falls as the blocks grow, while the answer stays right:

```text
== 7. Block size against speed: our block-by-block attention, one query, a 2,048-token cache, one layer (CPU)
B =   1: 2048 blocks,   91.29 ms; max difference from SDPA 1.6e-07
B =   4:  512 blocks,   25.11 ms; max difference from SDPA 1.2e-07
B =  16:  128 blocks,    8.33 ms; max difference from SDPA 8.9e-08
B =  64:   32 blocks,    2.85 ms; max difference from SDPA 6.0e-08
B = 256:    8 blocks,    0.99 ms; max difference from SDPA 5.2e-08
contiguous SDPA, one call: 0.10 ms
```

{{FIG:p3_tradeoff|Both sides of the block size. Left: the share of held memory that holds tokens, for 2,000 Alpaca requests at their full length. Right: the time of our block-by-block attention over 2,048 keys. Small blocks save memory and cost time; large blocks do the reverse.}}

Every block size gives the right answer, within $$1.6 \times 10^{-7}$$ of SDPA. The time falls as the blocks grow, from 91 ms with blocks of one token to 0.99 ms with blocks of 256. One call of the library's contiguous SDPA takes 0.10 ms.

Be careful with what this shows.

Our loop is Python. Most of its cost per block is Python and the start of a small operation, not the GPU effect that the paper describes. The direction is the same, but the size is not. A real PagedAttention kernel runs all blocks inside one GPU program. The paper reports (§7.1) that its attention kernel takes 20% to 26% longer than FasterTransformer's, not ten times longer. The timings were the fastest of 7 runs on a laptop shared with other work, so treat them as a shape.

## Waste is bounded by one block {§4.3}

> [!PAPER] · §4.3 · page 6
> [![The paragraph on dynamic block assignment, with a new physical block is only allocated when all previous blocks are full, within one block and its KV blocks can be freed highlighted](/img/papers/vllm/p3-one-block.png)](/img/papers/vllm/p3-one-block.png)
>
> **What it says:** blocks are filled from left to right, and "a new physical block is only allocated when all previous blocks are full". So vLLM limits the waste of a request "within one block". More requests fit in memory, which raises throughput. When a request finishes, "its KV blocks can be freed" for other requests.
>
> **What it means:**
>
> This is the key promise of the design, and it is easy to prove. Every block of a request except the last one is full, because a new block is only taken when the previous one is full. So the only empty slots are in the last block. That block holds at least one token, so at most $$B - 1$$ slots are empty.
>
> Compare this with Part 2. There, a request could waste 2,037 of its 2,048 slots. Here, with $$B = 16$$, it can never waste more than 15, whatever its length.

The rule fits in one formula. A request of $$L$$ tokens in blocks of $$B$$ holds $$\lceil L/B \rceil$$ blocks, so its waste is:

$$
\text{waste}(L) = B \left\lceil \frac{L}{B} \right\rceil - L \;\le\; B - 1
$$

Read it this way. $$B \lceil L/B \rceil$$ is the number of slots held: the number of blocks times the slots per block. Subtract the $$L$$ slots in use, and what is left is the empty end of the last block. For example, $$L = 7$$ and $$B = 4$$ give $$4 \cdot 2 - 7 = 1$$, the one free slot of Figure 6. $$L = 8$$ gives $$4 \cdot 2 - 8 = 0$$, and $$L = 9$$ gives $$4 \cdot 3 - 9 = 3$$, the largest waste possible for $$B = 4$$.

We check it on real lengths in two ways. Method 1 builds a block table token by token, as `append_slot` does, and counts the empty slots at the end. Method 2 is the formula:

```python
def waste_by_table(L, B):
    """Method 1: append L tokens one by one through a block table; count the empty slots at the end."""
    table = []
    for _ in range(L):
        if not table or table[-1] == B:
            table.append(0)                                     # a new block, only when the last one is full
        table[-1] += 1
    return len(table) * B - L


def waste_closed(L, B):
    """Method 2: ceil(L / B) blocks of B slots, minus the L slots in use."""
    return B * math.ceil(L / B) - L
```

```text
== 6. Waste per request at the end of generation, 2,000 Alpaca requests (prompt + answer, OPT tokenizer)
lengths: mean 76.3, shortest 8, longest 513 tokens; 152,590 tokens in all
block size max waste bound B-1 mean waste slots held  in use  methods agree
         1         0         0        0.0    152,590  100.0%  True
         2         1         1        0.5    153,576   99.4%  True
         4         3         3        1.5    155,568   98.1%  True
         8         7         7        3.4    159,480   95.7%  True
        16        15        15        7.4    167,424   91.1%  True
        32        31        31       14.8    182,240   83.7%  True
        64        63        63       32.2    216,960   70.3%  True
       128       127       127       72.1    296,704   51.4%  True
contiguous      2040               1971.7  4,096,000    3.7%  (one chunk of 2,048 slots per request)
OPT-13B with 12 GB of KV cache = 14,648 slots: chunks of 2,048 fit 7 requests; blocks of 16 fit 175 requests of these lengths, even at their full length
```

The two methods agree for every block size. The largest waste of any request is exactly $$B - 1$$, the bound, and the mean waste is close to $$B/2$$, as the "half-empty last block" argument predicts. With blocks of 16, 91.1% of the held memory holds tokens.

The last two rows show the scale of the change. Chunks of 2,048 slots hold 4,096,000 slots for 152,590 tokens, so only 3.7% of them hold anything. With 12 GB of KV cache for OPT-13B, Part 1's budget on a 40 GB A100, chunks of 2,048 fit 7 requests at once. Blocks of 16 fit 175 requests of these lengths, even when each one has reached its full length.

Two notes keep this honest. First, Alpaca requests are short, 76 tokens on average with the prompt, so the gain is large. Longer requests waste a smaller share in chunks too. Second, this count looks at one moment, the end of each request. Part 1 followed requests over their whole life and found the same picture: 3.6% with chunks of 2,048 against 90.7% with blocks of 16.

## Two requests in one memory: Figure 7 {§4.3}

> [!PAPER] · Figure 7 and §4.3 · page 6
> [![Figure 7: request A in physical blocks 7, 1 and 3 and request B in physical blocks 5 and 2 of one set of nine physical KV blocks](/img/papers/vllm/p3-figure7.png)](/img/papers/vllm/p3-figure7.png)
> [![The paragraph on Figure 7, with mapped to different physical blocks, do not need to be contiguous in physical GPU, effectively and utilized by both sequences highlighted](/img/papers/vllm/p3-fig7-text.png)](/img/papers/vllm/p3-fig7-text.png)
>
> **What it says:** Figure 7 shows vLLM managing the memory of two sequences. Their logical blocks are "mapped to different physical blocks". The neighbouring logical blocks of both sequences "do not need to be contiguous in physical GPU" memory, and the physical blocks can be "effectively utilized by both sequences".
>
> **What it means:**
>
> Request A holds physical blocks 7, 1 and 3. Request B, "It was the best of times", holds blocks 5 and 2. The two requests are interleaved: going down the memory, the blocks belong to A, B, A, B, A. Neither request cares, because each one reads its blocks through its own table.
>
> This is what kills external fragmentation. When a request ends, its blocks return to the free list. Every block has the same size, so any free block fits the next request that needs one. There is never a gap too small to use.

We built Figure 7 with our code. Both requests run in one pool of 9 physical blocks, one iteration at a time. In iteration 1 both prompts are prefilled. In iterations 2 and 3, request A writes "fathers" and "brought" while request B writes "of" and "times". The free list starts 7, 1, 5, 2, 3, so the numbers match the figure. Then request A ends, and a third request C arrives:

```text
== 5. Figure 7: requests A and B share one pool of 9 physical blocks (block size 4)
iteration 1 (both prefills): A: logical 0 -> physical 7 (4 filled)  logical 1 -> physical 1 (3 filled)
                             B: logical 0 -> physical 5 (4 filled)
iteration 2 (A writes 'fathers', B writes 'of'): A: [7, 1], B: [5, 2]
iteration 3 (A writes 'brought', B writes 'times'): A: [7, 1, 3], B: [5, 2]
physical blocks:
  block 0:    -
  block 1: A     years      ago      our  fathers
  block 2: B        of    times        .        .
  block 3: A   brought        .        .        .
  block 4:    -
  block 5: B        It      was      the     best
  block 6:    -
  block 7: A      Four    score      and    seven
  block 8:    -
each request against the library run alone: A max difference 1.7e-05, B max difference 1.5e-05
free list before A ends: [0, 4, 6, 8]
A ends; its blocks go back. free list now: [0, 4, 6, 8, 7, 1, 3]
a new request C of 9 tokens takes physical blocks [0, 4, 6]
```

{{FIG:p3_fig7_steps|Figure 7 from our run, step by step. Each row shows both block tables, with the filled count of each block in brackets, and the nine physical blocks. The slots written in that step are highlighted. When A ends, its blocks 7, 1 and 3 go back to the free list; C then takes blocks 0, 4 and 6.}}

Read the picture from the top. In iteration 1, A's prompt fills blocks 7 and 1, and B's four-token prompt fills block 5 exactly. In iteration 2, B's new token "of" finds block 5 full, so B gets block 2. A's "fathers" fits in block 1. In iteration 3 it is the other way round: A's "brought" needs a new block, 3, and B's "times" fits in block 2.

After iteration 3, the pool matches Figure 7 block for block. Each request's logits agree with the library run on that request alone, to within $$1.7 \times 10^{-5}$$. Sharing one pool did not let one request disturb the other.

Then request A ends. Its blocks 7, 1 and 3 go back to the end of the free list. A new request C of 9 tokens needs 3 blocks and gets 0, 4 and 6, the next ones on the list. Blocks 7, 1 and 3 wait for the request after it. No copy, no compaction, no search for a gap of the right size: a free block is a free block.

## What this part sets up

§4.1 to §4.3 build one mechanism from three pieces. PagedAttention computes attention block by block, with one softmax over all blocks, and gives the same result as attention over one contiguous tensor. The block table records where each logical block of a request lives, and finds any token with the same split-and-look-up as a page table. The KV cache manager hands out a physical block only when the last one is full, and takes blocks back when a request ends.

Because a block table is only a list of numbers, two requests can point at the same physical block. Part 4 builds on that: shared prompts, beams, copy-on-write, and what happens when the free list runs empty.

## What our small run cannot show

Our engine is Python on a laptop CPU, with OPT-125M in 32-bit numbers. It reproduces the logic of §4.1 to §4.3 exactly: the block table, the allocation rule and the attention result. It does not reproduce the speed. The paper's PagedAttention is a CUDA kernel that reads blocks inside one GPU program; our loop starts a separate small operation for every block. Our timings show the direction of the block size trade-off, not its real size.

We chose the order of the free list so that the block numbers match Figures 6 and 7. The paper does not say how vLLM picks a free block. The random orders of our generation test show that the choice does not change the result. We also fed the paper's output words to the model ourselves, because the small model would not write them.

> [!TAKEAWAYS] Key takeaways
> - **vLLM has two halves.** A KV cache manager stores keys and values in fixed-size blocks. PagedAttention, an attention kernel, reads them wherever they are.
> - **The idea is paging,** from operating systems since the Atlas computer. A token's position splits into a block and an offset, as a byte address splits into a page and an offset. In our run, one function turned address 10,000 into 30,480 and token 9 into slot 13.
> - **An ordinary attention kernel needs one contiguous tensor,** because it finds key $$t$$ by arithmetic. PagedAttention takes the block table as an extra input instead.
> - **Eq. 4** is ordinary attention with the sums grouped by block. The block sums of the exponentials add up to one shared denominator. In our run, a softmax inside one block would give "Four" 0.948 instead of 0.731.
> - **Same answer:** our paged attention matched PyTorch's SDPA on the contiguous cache to within $$7.7 \times 10^{-7}$$ in all 12 layers of OPT-125M. With blocks of 1, 4 and 16 in random order, our engine wrote the same 48 tokens as `model.generate`.
> - **A block table** maps each logical block to a physical block and counts its filled slots. Our walk-through reproduced Figure 6 row for row: physical blocks 7, 1 and 3.
> - **A new block is taken only when the last one is full,** so a request wastes at most $$B - 1$$ slots. On 2,000 Alpaca requests, blocks of 16 kept 91.1% of the held memory full, against 3.7% for chunks of 2,048.
> - **Blocks of many requests interleave in one pool,** as in Figure 7, and freed blocks go straight to the next request. There is no external fragmentation.
> - **Block size is a trade-off:** small blocks waste less memory, large blocks are faster to read. vLLM uses 16.

**Next, in Part 4:** §4.4 to §4.6 and §5. Sharing blocks between samples and beams, copy-on-write, preemption, swapping, and running on many GPUs.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/vllm/vllm_part3.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part3.py). It builds a block allocator, a paged KV cache and PagedAttention from scratch, and runs OPT-125M through them. It walks through Figures 6 and 7 and checks every result against the library. It also counts the waste per request on Alpaca, times the block loop, does the page-table lookup for tokens, and works Eq. 4 by hand. Every number is saved in `results/part3.json`. On an Apple M-series laptop it takes under a minute.

```bash
pip install torch transformers datasets
python vllm_part3.py   # downloads facebook/opt-125m (about 250 MB) and tatsu-lab/alpaca
```

<figure><img src="/img/papers/vllm/part3-run.png" alt="Terminal output of vllm_part3.py: the block table of Figure 6 step by step, Eq. 4 on the query forth, the check against the library, greedy generation with three block sizes, the two requests of Figure 7, the waste per request, the block size timings, the token lookup and Eq. 4 by hand" loading="lazy" /><figcaption>The full output of vllm_part3.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
