---
title: "Build It from Scratch"
description: "We write a small vLLM in about 500 lines of Python: blocks, block tables, PagedAttention, a scheduler and parallel sampling. It runs the real OPT-125M, matches the transformers library and serves real requests."
part: 6
covers: "§4 and §5, rebuilt in Python"
date: 2026-10-24
tags: [llm-serving, kv-cache, pagedattention, from-scratch, python]
---

Parts 1 to 5 read the paper. In this part we build its core ourselves, in about 500 lines of Python. Then we run it on the real weights of a small OPT model.

The proof has three steps. First, each piece does what the paper says, and we print it doing so. Second, the whole engine writes the same tokens as the transformers library, to the last token. Third, it serves a batch of real requests faster than the same code with one contiguous chunk per request.

Each piece below shows the exact lines of the paper, the code that builds them, and what the code printed.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## The map: Figure 4 as seven files {§4}

A serving engine is a program that takes requests from many users and runs one model for all of them. Before we write one, we need a map. The paper gives it in Figure 4.

> [!PAPER] · Figure 4 · page 5
> [![Figure 4 of the paper: a scheduler on the left, a KV cache manager with block tables below it, CPU and GPU block allocators, and workers 0 to N-1 on the right, each with a cache engine and a model shard](/img/papers/vllm/p6-figure4.png)](/img/papers/vllm/p6-figure4.png)
>
> **What it says:** "vLLM system overview."
>
> **What it means:**
>
> Read the figure as an office. The scheduler is the manager who decides, at every step, which requests to work on. The KV cache manager is the clerk who keeps the block tables and knows which memory is free.
>
> On the right are the workers, one per GPU. Each holds a part of the model, its "model shard", and a "cache engine" that owns its piece of KV cache memory. The workers only do the arithmetic.

> [!PAPER] · §4 · page 5
> [![The opening of §4, with centralized scheduler to, KV cache manager effectively manages the KV cache in a and paged fashion highlighted](/img/papers/vllm/p6-method.png)](/img/papers/vllm/p6-method.png)
>
> **What it says:** vLLM "adopts a centralized scheduler to" coordinate the GPU workers. The "KV cache manager effectively manages the KV cache in a paged fashion", and it acts on the instructions of the scheduler.
>
> **What it means:**
>
> This paragraph is the plan of our package. One boss, the scheduler, decides at every step which requests run. One bookkeeper, the KV cache manager, decides where each request's keys and values live. The GPU only does arithmetic on the memory it is told to use.
>
> A real server spreads the model over many GPUs. We keep the same split of jobs in one Python process.

> [!NOTE] Where this idea comes from
> The split between "who decides" and "who computes" comes from operating systems. Their scheduler decides which program runs next, and their memory manager decides which pages each program gets. The programs only compute. In vLLM, the scheduler and the KV cache manager play the operating system, and the GPU workers play the programs.

Each box of Figure 4 is a file of our package, minivllm. Dashed boxes are in the paper but not in our code:

{{FIG:p6_arch|Figure 4 redrawn as a map of our package. Requests enter on the left. The scheduler holds the KV cache manager, with its block tables and its GPU block allocator. Worker 0 holds the cache engine and the model. engine.py is the loop that drives all of them. Dashed boxes are in the paper but not in our code.}}

We left out two boxes. The CPU block allocator serves swapping, which our laptop cannot use, as we explain below. Workers 1 to N-1 serve a model split over many GPUs (Part 4).

The code lives in [code/papers/vllm/minivllm](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm/minivllm). It uses PyTorch for arithmetic and transformers only for the weights. The first lines of our script count the lines of each file:

```text
== 1. The package minivllm
  block_manager.py   151 lines
  kv_cache.py         27 lines
  attention.py        49 lines
  model.py            64 lines
  sequence.py         31 lines
  scheduler.py        71 lines
  engine.py          108 lines
  total              501 lines of Python (no CUDA)
```

| File | Box of Figure 4 | What it holds | Paper |
|---|---|---|---|
| `sequence.py` | Requests | Requests, their samples and the sampling settings | §4.4, §4.5 |
| `scheduler.py` | Scheduler | First come, first served; preemption by recomputation | §4.5 |
| `block_manager.py` | KV cache manager | Free list, reference counts, block tables, copy-on-write | §4.2 to §4.4 |
| `kv_cache.py` | Cache engine | One big tensor cut into blocks; the block copy | §4.2, §5.1 |
| `model.py` | Model shard | OPT written out by hand, reading its keys and values through block tables | §4.3 |
| `attention.py` | Model shard | PagedAttention: Eq. 4, and the fast versions the engine uses | §4.1 |
| `engine.py` | the arrows | The loop: schedule, run the model, sample, fork, free | §4.3, §5.2 |

A map does not show the order in which things act. The second picture does: one step of the engine, as calls from file to file.

{{FIG:p6_flow|One step of the engine as a sequence diagram. Time runs down. The engine asks the scheduler for work, the scheduler asks the block manager for slots, and the engine runs the model, which writes to the cache and reads blocks through PagedAttention. The logits go back to the engine, which samples, forks and frees.}}

The engine asks the scheduler what to run (1 to 4), makes any block copies (5) and calls the model (6). In each layer, the model stores the new keys and values (7), and PagedAttention reads the old ones block by block (8, 9). The engine then picks the next tokens and tells the block manager about new samples and finished requests (10, 11).

The model is facebook/opt-125m, the smallest OPT, built like the OPT-13B of the paper. It has 12 layers and 12 attention heads of 64 numbers. We run it in FP32 on the CPU, so that our numbers can match the library's closely.

> [!DEFINITION] FP32
> A 32-bit floating-point number, 4 bytes, with about 7 correct decimal digits. Servers usually run models in FP16, 2 bytes.

## The KV cache as blocks {§4.1}

> [!PAPER] · §4.1 · page 5
> [![The opening of §4.1, with non-contiguous memory space, partitions the KV cache of each and fixed number of highlighted](/img/papers/vllm/p6-blocks.png)](/img/papers/vllm/p6-blocks.png)
>
> **What it says:** PagedAttention stores "continuous keys and values in non-contiguous memory space". It "partitions the KV cache of each" sequence into KV blocks, and each block holds the keys and values of a "fixed number of" tokens, the block size $$B$$.
>
> **What it means:**
>
> Think of a long text written on index cards instead of one long scroll. Each card holds four lines. The cards can lie anywhere on the desk, as long as a list says which card comes first, second and third.
>
> In code, we book one large tensor when the server starts and cut it into equal blocks. A block is not a separate allocation; it is a row number in that tensor.

> [!DEFINITION] KV block and block size
> A **KV block** is a fixed-size piece of the KV cache that holds the keys and values of $$B$$ consecutive tokens of one sequence. $$B$$ is the **block size**. vLLM uses $$B = 16$$ by default; Figures 5 to 8 of the paper use 4.

> [!DEFINITION] Tensor
> A block of numbers with a fixed number of axes, as PyTorch stores it. A list of numbers is a tensor with one axis, a table has two. Its **shape** lists the length of each axis, for example (12, 2, 9, 4, 12, 64).

> [!INFO] kv_cache.py: the memory itself
> **What it does.** It holds the whole KV cache in one tensor, stores new keys and values, and copies whole blocks. It is the "cache engine" of Figure 4.
>
> **Why it exists.** Asking the GPU for memory while a server runs is slow, and many small requests leave gaps. So the cache is booked once, at start-up, and cut into equal blocks. After that, "allocating" a block only means choosing a number.
>
> **Where the idea comes from.** Operating systems divide physical memory into equal **page frames** and hand them out by number. Equal sizes mean that any free piece fits any request.

Here is the whole cache, from `kv_cache.py`. Look at the shape in the docstring first; the rest follows from it:

```python
class PagedKVCache:
    """GPU memory for keys and values, booked once at start-up and cut into num_blocks physical blocks.

    Shape: (layers, 2, num_blocks, block_size, heads, head_dim). Index 0 of the second axis holds keys, 1 values.
    Every layer uses the same block numbers, so one block table per sequence serves all layers."""

    def __init__(self, num_layers, num_blocks, block_size, num_heads, head_dim, dtype=torch.float32, device='cpu'):
        self.num_blocks, self.block_size = num_blocks, block_size
        self.data = torch.zeros(num_layers, 2, num_blocks, block_size, num_heads, head_dim, dtype=dtype, device=device)

    def blocks(self, layer):
        """Keys and values of one layer, each (num_blocks, block_size, heads, head_dim)."""
        return self.data[layer, 0], self.data[layer, 1]

    def write(self, layer, slots, k, v):
        """Store the new keys and values (t, heads, head_dim) at their slots (t,) = block * block_size + offset."""
        keys, values = self.blocks(layer)
        keys.view(-1, *keys.shape[2:])[slots] = k             # view the blocks as one long list of slots
        values.view(-1, *values.shape[2:])[slots] = v
```

Read the shape from left to right. There are 12 layers. Each layer has two halves, keys and values. Each half is cut into `num_blocks` blocks. Each block has `block_size` places for tokens, and each place holds 12 heads of 64 numbers.

> [!DEFINITION] Slot
> One token's place in the cache: place $$r$$ of physical block $$b$$. We number the slots one after the other, so slot $$b \cdot B + r$$ is place $$r$$ of block $$b$$. With $$B = 4$$, slot 15 is place 3 of block 3.

Now the method `write`, for a tiny cache of 9 blocks of 4. `view(-1, ...)` lays the 9 blocks end to end as one list of 36 slots, without copying. Then `[slots] = k` puts each new key at its slot. Two new tokens at slots 7 and 15 are stored by one line.

Every layer uses the same block number, so one block table per sequence serves all 12 layers. For OPT-125M in FP32, one token takes $$2 \times 12 \times 768 \times 4 = 73{,}728$$ bytes, 72 KB, and a block of 16 tokens takes 1.2 MB. Read the product as: keys and values, 12 layers, 768 numbers per layer (12 heads of 64), 4 bytes per number.

## PagedAttention, Eq. 4 {§4.1}

> [!PAPER] · §4.1 · page 5
> [![Equation 4 of the paper, with Denote the key block, wise computation and row vector of attention highlighted](/img/papers/vllm/p6-eq4.png)](/img/papers/vllm/p6-eq4.png)
>
> **What it says:** with the key block $$K_j$$ and value block $$V_j$$ of block $$j$$, attention "can be transformed into the following block-wise computation" (Eq. 4). $$A_{ij}$$ is the "row vector of attention score on $$j$$-th KV block".
>
> **What it means:**
>
> Ordinary attention, from Part 2, compares the query of token $$i$$ with every earlier key. It turns the scores into weights that add up to 1, and mixes the values with those weights. Eq. 4 does exactly the same arithmetic, only grouped by block. It takes one block of keys, gets its scores, and later mixes that block's values.
>
> Nothing about the result changes. The weights still add up to 1 over all earlier tokens, because the bottom of the fraction sums over every block. What changes is that the computation never needs the keys side by side in one piece of memory. It needs only a way to find block 1, block 2 and block 3.

Here is Eq. 4 as the paper writes it. (The $$\mathbf{1}$$ in the bottom is a vector of ones; it adds up the $$B$$ numbers of block $$t$$.)

$$
A_{ij} = \frac{\exp\left(q_i^\top K_j / \sqrt{d}\right)}{\sum_{t=1}^{\lceil i/B \rceil} \exp\left(q_i^\top K_t \mathbf{1} / \sqrt{d}\right)}, \qquad o_i = \sum_{j=1}^{\lceil i/B \rceil} V_j A_{ij}^\top
$$

Read it from left to right. $$q_i$$ is the query of token $$i$$. $$K_j$$ holds the $$B$$ keys of block $$j$$, so $$q_i^\top K_j$$ is a row of $$B$$ scores, one per token of the block. Dividing by $$\sqrt{d}$$ keeps the scores in a good range.

$$\lceil i/B \rceil$$ is the number of blocks that hold tokens 1 to $$i$$ ("ceil" rounds up). The bottom adds the exponentials of all scores in all those blocks. So $$A_{ij}$$ is the row of $$B$$ weights for block $$j$$. Last, $$V_j A_{ij}^\top$$ mixes the $$B$$ values of block $$j$$ with those weights, and $$o_i$$ adds the blocks together.

A tiny example you can check by hand. Take $$B = 2$$ and three tokens, so block 1 holds tokens 1 and 2 and block 2 holds token 3. Let the scores be 0, 0 and 0.693, whose exponentials are 1, 1 and 2. The bottom is $$(1 + 1) + 2 = 4$$.

So $$A_{i1} = (0.25, 0.25)$$ and $$A_{i2} = (0.5)$$. With values 4, 8 and 2, the output is $$4 \cdot 0.25 + 8 \cdot 0.25 + 2 \cdot 0.5 = 4$$. A softmax over all three tokens at once gives the same weights and the same 4.

> [!PAPER] · §4.1 · page 5
> [![The paragraph on the PagedAttention kernel, with identifies and fetches different KV blocks separately and not contiguous on the physical memory highlighted](/img/papers/vllm/p6-kernel.png)](/img/papers/vllm/p6-kernel.png)
>
> **What it says:** the PagedAttention kernel "identifies and fetches different KV blocks separately". In Figure 5 the three blocks of "Four score and seven years ago our fathers brought forth" are "not contiguous on the physical memory".
>
> **What it means:**
>
> The kernel walks the block table. For each entry it goes to that block, multiplies the query with the block's keys, and keeps the scores. Then it mixes each block's values. The order of the blocks in memory does not matter, only the order in the table.
>
> Our Method 1 below is this walk, written as a Python loop. A real kernel runs the blocks in parallel on the GPU, but it computes the same numbers.

> [!DEFINITION] GPU kernel
> A function that runs on the GPU, with many threads working on it at once. Part 1 met the word; here it is the code that computes PagedAttention.

> [!INFO] attention.py: reading a scattered context
> **What it does.** It computes attention when the keys and values sit in scattered blocks. It has Eq. 4 block by block, a version for a prompt, and a version for many requests.
>
> **Why it exists.** Ordinary attention code expects the keys of one sequence side by side in memory. Paging breaks that, so attention must take a block table and find the keys itself.
>
> **Where the idea comes from.** Attention comes from the Transformer (Vaswani et al., 2017). FlashAttention (Dao et al., 2022) already split its softmax into tiles, to keep each tile in fast on-chip memory. The vLLM authors split it by block because the blocks live in different places.

Method 1 is Eq. 4 word for word, from `attention.py`. It handles one query token. Watch for two loops: the first gets the scores of each block, the second mixes the values of each block.

```python
def paged_attention_eq4(q, k_blocks, v_blocks, block_table, context_len):
    """Eq. 4 for one query token i: q (H, D), the sequence's block table, tokens 1..context_len. Returns (H, D)."""
    B, D = k_blocks.shape[1], q.shape[-1]
    n_blocks = -(-context_len // B)                         # ceil(i / B) blocks hold the context
    scores = []
    for j in range(n_blocks):
        K_j = k_blocks[block_table[j]]                      # fetch block j from wherever it sits: (B, H, D)
        n_j = min(B, context_len - j * B)                   # the last block may be only partly filled
        scores.append(torch.einsum('hd,bhd->hb', q, K_j[:n_j]) / D ** 0.5)   # q_i^T K_j / sqrt(d): (H, n_j)
    # The denominator of Eq. 4 sums exp(...) over all blocks; we subtract the largest score first so that
    # exp() cannot overflow. This changes no result: it cancels between the top and the bottom of the fraction.
    m = torch.cat(scores, -1).max(-1, keepdim=True).values  # (H, 1)
    total = sum(torch.exp(s - m).sum(-1, keepdim=True) for s in scores)
    o = torch.zeros_like(q)
    for j, s in enumerate(scores):
        A_ij = torch.exp(s - m) / total                     # the attention scores on block j: (H, n_j)
        V_j = v_blocks[block_table[j]][:s.shape[-1]]        # (n_j, H, D)
        o += torch.einsum('hb,bhd->hd', A_ij, V_j)          # o_i = sum_j V_j A_ij^T
    return o
```

Walk through it with the test below: 10 tokens, $$B = 4$$, block table [7, 1, 3]. `n_blocks` is $$\lceil 10/4 \rceil = 3$$. For $$j = 0$$ the loop fetches physical block 7 and uses all 4 of its tokens. For $$j = 2$$ it fetches block 3, and `n_j` is $$10 - 8 = 2$$, because the last block holds only 2 tokens.

`einsum('hd,bhd->hb', ...)` means "for each head, the dot product of the query with each key". So each block gives 12 heads by `n_j` scores. The variable `total` is the bottom of Eq. 4, and the second loop is the top, block by block. Subtracting the largest score `m` keeps every exponential at 1 or below; in FP32, $$e^{89}$$ is already too large to store.

The engine uses a faster form: it gathers a sequence's blocks in table order, `blocks[block_table]`, then runs one ordinary attention. Method 2 is the library: we copy the blocks into one contiguous tensor and call PyTorch's `scaled_dot_product_attention`. We put 10 tokens in physical blocks 7, 1 and 3, as in Figure 5. In the output, look for the size of the differences, and for the last line:

```text
== 3. PagedAttention (Eq. 4): one query against 10 tokens in 3 scattered blocks of 4 (12 heads of 64)
  Eq. 4 block by block vs torch on a contiguous copy: max difference 4.2e-07
  gathered version (used by the engine) vs torch:     max difference 2.4e-07
  same blocks moved to physical 0, 5, 8 (new block table): max difference 0.0e+00
```

All three agree to within $$4.2 \times 10^{-7}$$, the size of rounding errors in FP32. The last line is the point of paging. We moved the same three blocks to other places in memory and changed only the block table. The output did not change at all, not even by one rounding error.

## Physical blocks and block tables {§4.2}

> [!PAPER] · §4.2 · page 5
> [![The opening of §4.2, with logical pages to physical pages, non-contiguous physical memory pages and fixed-size KV blocks, like pages in virtual memory highlighted](/img/papers/vllm/p6-vm.png)](/img/papers/vllm/p6-vm.png)
>
> **What it says:** an operating system maps a program's "logical pages to physical pages", and contiguous logical pages can sit in "non-contiguous physical memory pages". vLLM organizes the KV cache as "fixed-size KV blocks, like pages in virtual memory".
>
> **What it means:**
>
> A program believes that its memory is one unbroken range. In fact its pages are scattered, and a page table translates each address. Each sequence gets the same illusion: logical blocks 0, 1, 2 in order, each stored in any physical block. The [Background page](background-the-kv-cache-and-paging.md) explains virtual memory.

> [!PAPER] · §4.2 · page 5
> [![The paragraph on logical KV blocks, with filled from left to right, unfilled positions, reserved for future generations and allocates a contiguous chunk of GPU DRAM highlighted](/img/papers/vllm/p6-logical.png)](/img/papers/vllm/p6-logical.png)
> [![The rest of the paragraph, with maintains block tables, the number of filled positions and without reserving it for all positions in highlighted](/img/papers/vllm/p6-table.png)](/img/papers/vllm/p6-table.png)
>
> **What it says:** logical blocks are "filled from left to right", and the "unfilled positions" of the last block are "reserved for future generations". A block engine "allocates a contiguous chunk of GPU DRAM" and cuts it into physical blocks. The manager "maintains block tables", which record the physical block of each logical block and "the number of filled positions". So the cache can grow "without reserving it for all positions in" advance.
>
> **What it means:**
>
> Three objects appear here, and each becomes a class in our code. The contiguous chunk is our `PagedKVCache` tensor. The list of free physical blocks is the `BlockAllocator`. The table per sequence is the `BlockTable`.

> [!DEFINITION] Logical and physical block
> A **logical block** is a block as the sequence sees it. Block 0 holds its first $$B$$ tokens, block 1 the next $$B$$, and so on. A **physical block** is a real block of the cache tensor, named by its number. The block table maps each logical block to a physical block.

> [!INFO] block_manager.py: the bookkeeper
> **What it does.** It keeps the list of free blocks, a count of owners for each block, and one block table per sequence. It never touches the numbers inside the blocks; it only hands out block numbers and slots.
>
> **Why it exists.** The scheduler must know whether the new tokens fit, and the model must know where they go. Both answers come from here, in plain Python.
>
> **Where the idea comes from.** This is paging, from virtual memory. The Atlas computer, built at the University of Manchester in the early 1960s, was the first machine to use it. Programs saw one large memory, and a table mapped fixed-size pages onto real memory. So vLLM moves the idea from a program's pages to a request's KV blocks.

The allocator and the block table, from `block_manager.py`. The allocator has three operations, take a block, share it and let go of it; the table translates a token's position into a slot:

```python
class BlockAllocator:
    """Hands out physical KV blocks by number and counts how many logical blocks point at each one (§4.4)."""

    def __init__(self, num_blocks):
        self.num_blocks = num_blocks
        self.free_list = list(range(num_blocks))[::-1]   # pop() hands out block 0 first, then 1, 2, ...
        self.ref_count = [0] * num_blocks

    def allocate(self):
        if not self.free_list:
            raise MemoryError('no free KV block')
        block = self.free_list.pop()
        self.ref_count[block] = 1
        return block

    def share(self, block):
        """One more logical block points at this physical block."""
        self.ref_count[block] += 1

    def free(self, block):
        """One logical block lets go; the physical block returns to the free list when nobody points at it."""
        self.ref_count[block] -= 1
        if self.ref_count[block] == 0:
            self.free_list.append(block)

    @property
    def num_free(self):
        return len(self.free_list)


@dataclass
class BlockTable:
    """The block table of one sequence (Figure 6): logical block j lives in physical block blocks[j]."""
    blocks: list = field(default_factory=list)
    num_tokens: int = 0                          # tokens whose keys and values are stored

    def filled(self, j, block_size):
        """The "# filled" column of Figure 6: how many slots of logical block j hold a token."""
        return min(block_size, self.num_tokens - j * block_size)

    def slot(self, pos, block_size):
        """Where token number pos lives: its physical block times the block size, plus its offset."""
        return self.blocks[pos // block_size] * block_size + pos % block_size
```

Follow a cache of 4 blocks. At the start, `free_list` is [3, 2, 1, 0]. The first `allocate` pops 0 from the end and sets its count to 1. If a second sequence shares block 0, `share` raises the count to 2. Each `free` lowers it by 1. Only when it reaches 0 does block 0 go back on the list.

So the allocator is a stack of free block numbers: to allocate is to pop one, to free is to push it back. Both take the same tiny time, whatever the size of the cache.

The block table is a plain list plus a token count, because every block but the last is full. The method `slot` is the address translation of virtual memory, shown here for two tokens of our run:

{{FIG:p6_slot|Address translation for two tokens of request B in our run. The position, divided by the block size, gives the logical block and the place in it. The block table turns the logical block into a physical block, and the slot follows. "was" lives at slot 10, "the" at slot 20.}}

Take "the" at position 8. With $$B = 4$$, $$8 \,/\!/\, 4 = 2$$ and $$8 \bmod 4 = 0$$, so it is place 0 of logical block 2. Entry 2 of the table [2, 3, 5] is 5. So its slot is $$5 \times 4 + 0 = 20$$. Two divisions and one lookup find any token, however long the sequence.

## Decoding one request, step by step {§4.3}

> [!DEFINITION] Prompt phase and generation phase
> In the **prompt phase**, the model reads the whole prompt in one pass and stores the keys and values of all its tokens. In the **generation phase**, also called decoding, it writes one new token per step and stores that token's keys and values. Part 2 explains both.

> [!PAPER] · Figure 6 · page 6
> [![Figures 6 and 7 of the paper. Figure 6: request A with prompt Four score and seven years ago our; logical blocks 0 to 2 map to physical blocks 7, 1 and 3 through a block table with a filled column. Figure 7: requests A and B share the physical blocks](/img/papers/vllm/p6-figure6.png)](/img/papers/vllm/p6-figure6.png)
>
> **What it says:** "Block table translation in vLLM" (Figure 6), and "Storing the KV cache of two requests at the same time" (Figure 7).
>
> **What it means:**
>
> Figure 6 follows one request. The prompt "Four score and seven years ago our" has 7 tokens, so logical blocks 0 and 1 map to physical blocks 7 and 1. The circled numbers are the steps of the text below. Figure 7 adds request B, "It was the best of times", whose blocks sit between those of A.

> [!PAPER] · §4.3 · page 6
> [![Step 1 of §4.3, with does not require reserving the memory for the maximum possible generated, necessary KV blocks and The remaining slot is reserved for the highlighted](/img/papers/vllm/p6-step1.png)](/img/papers/vllm/p6-step1.png)
>
> **What it says:** vLLM "does not require reserving the memory for the maximum possible generated" length. It reserves "only the necessary KV blocks" for the prompt: 7 tokens need 2 blocks, the first 4 tokens in block 0 and the next 3 in block 1. "The remaining slot is reserved for the" generation phase.
>
> **What it means:**
>
> This is the whole difference from the systems of §3. They booked 2,048 slots the moment a request arrived. vLLM books 8, because 8 slots hold the 7 tokens it knows about. The one empty slot will be used at the very next step.

> [!PAPER] · §4.3 · page 6
> [![Steps 2 and 3 of §4.3, with one slot remains available in the last logical block, allocates a and new physical block highlighted](/img/papers/vllm/p6-step23.png)](/img/papers/vllm/p6-step23.png)
>
> **What it says:** at the first decoding step, "one slot remains available in the last logical block", so the new key and value go there and the "#filled" record goes from 3 to 4. At the second step the last block is full, so vLLM "allocates a new physical block" (block 3) and records it in the block table.
>
> **What it means:**
>
> A new block is taken only when a token has nowhere else to go. Taking a block is one pop from a list, so the question of how long the answer will be never needs an answer.

> [!NOTE] Where this idea comes from
> Taking memory only when it is first needed is **demand paging**. An operating system loads a page of a program the first time the program touches it. In the same way, vLLM takes a KV block the first time a token needs a place.

Steps 1 to 3 are two methods of our `BlockManager`. The prompt step calls `allocate`. Each decoding step calls `append_slot`. Both return slots, the addresses that `PagedKVCache.write` needs:

```python
    def allocate(self, seq_id, num_tokens):
        """Prompt phase (§4.3, step 1): book only the blocks that the prompt fills. Returns the prompt's slots."""
        t = BlockTable([self.allocator.allocate() for _ in range(cdiv(num_tokens, self.block_size))], num_tokens)
        self.tables[seq_id] = t
        return [t.slot(p, self.block_size) for p in range(num_tokens)]

    def append_slot(self, seq_id):
        """Find the slot of the next token (§4.3, steps 2 and 3; §4.4, copy-on-write). Returns the slot."""
        t, B = self.tables[seq_id], self.block_size
        if t.num_tokens % B == 0:                          # last block full: map a new logical block
            t.blocks.append(self.allocator.allocate())
        else:
            last = t.blocks[-1]
            if self.allocator.ref_count[last] > 1:         # the last block is shared: copy it before writing
                new = self.allocator.allocate()
                self.copies.append((last, new))            # the block engine copies the data later
                self.allocator.free(last)                  # ref count of the old block goes down by one
                t.blocks[-1] = new
        slot = t.slot(t.num_tokens, B)
        t.num_tokens += 1
        return slot
```

`cdiv` is division that rounds up: 7 tokens need $$\lceil 7/4 \rceil = 2$$ blocks. So `allocate` pops 2 blocks, builds the table, and returns 7 slots, one per prompt token.

Now `append_slot`, for request A after its 7-token prompt. `num_tokens` is 7, and $$7 \bmod 4 = 3$$, not 0. So the last block has room: this is step 2 of the paper. The block is not shared, so no copy. The slot is place 3 of the last block, and the count becomes 8.

At the next call, $$8 \bmod 4 = 0$$: the last block is full. The first branch takes a new block, which is step 3 of the paper. The branch for a shared block belongs to §4.4, further down.

Now we replay Figures 6 and 7 with the real model. OPT puts a start token, `</s>`, before every prompt. So "Four score and seven years ago" is 7 tokens, like the paper's prompt. B is "It was the best of times", also 7 tokens. Block size 4, a cache of 9 physical blocks, as in the figures. In the output, watch the "filled" count of each request's last block, and the count of free blocks:

```text
== 2. Block tables (Figures 6 and 7): block size 4, a cache of 9 physical blocks
  after the prompt step:
    A logical 0 -> physical 0, filled 4: </s> | Four | score | and
    A logical 1 -> physical 1, filled 3: seven | years | ago
    B logical 0 -> physical 2, filled 4: </s> | It | was | the
    B logical 1 -> physical 3, filled 3: best | of | times
    free blocks: 5 of 9
  after the decoding step 1:
    A logical 0 -> physical 0, filled 4: </s> | Four | score | and
    A logical 1 -> physical 1, filled 4: seven | years | ago | ,
    B logical 0 -> physical 2, filled 4: </s> | It | was | the
    B logical 1 -> physical 3, filled 4: best | of | times | ,
    free blocks: 5 of 9
  after the decoding step 2:
    A logical 0 -> physical 0, filled 4: </s> | Four | score | and
    A logical 1 -> physical 1, filled 4: seven | years | ago | ,
    A logical 2 -> physical 4, filled 1: the
    B logical 0 -> physical 2, filled 4: </s> | It | was | the
    B logical 1 -> physical 3, filled 4: best | of | times | ,
    B logical 2 -> physical 5, filled 1: the
    free blocks: 3 of 9
```

{{FIG:p6_tables|Our run after decoding step 2. Each request has three logical blocks in order. Their physical blocks interleave: request A owns blocks 0, 1 and 4, request B owns 2, 3 and 5. Blocks 6 to 8 are still free.}}

This is Figure 6, step by step. After the prompt, each request holds 2 blocks, and its second block is filled 3 of 4. In step 1 the model reads ",", the token it chose in the prompt step, and its key and value fill the free slot. Filled goes from 3 to 4, and no block is taken. In step 2 it reads "the", the last block is full, and each request gets a new physical block.

Our allocator hands out the lowest free number, so the numbers differ from the paper's 7, 1 and 3. That changes nothing. Request A's logical block 1 is in physical block 1 and its logical block 2 is in physical block 4, with B's blocks in between. This is Figure 7.

## One iteration for many requests {§4.3}

> [!DEFINITION] Batch and iteration
> A **batch** is a group of inputs that a model processes in one pass. An **iteration**, or **step**, is one pass of the model. In a serving engine, each step writes one new token for every running request.

> [!PAPER] · §4.3 · page 6
> [![The paragraph on one decoding iteration, with selects a set of candidate sequences for batching and concatenates all the input tokens highlighted](/img/papers/vllm/p6-iteration.png)](/img/papers/vllm/p6-iteration.png)
> [![Its continuation in the next column, with as one sequence and feeds it into the LLM and PagedAttention kernel to access highlighted](/img/papers/vllm/p6-iteration-2.png)](/img/papers/vllm/p6-iteration-2.png)
>
> **What it says:** in each iteration vLLM "selects a set of candidate sequences for batching" and allocates their new blocks. It then "concatenates all the input tokens" of the iteration, all prompt tokens of new requests and the latest token of running ones, "as one sequence and feeds it into the LLM". The "PagedAttention kernel" reads the old cache and stores the new keys and values.
>
> **What it means:**
>
> Part 2 showed that every layer except attention treats each token on its own. So a server can put the tokens of many requests in one long list and run each weight matrix once over it. Only attention must know which tokens belong together, and that is where the block tables enter the model.

> [!NOTE] Where this idea comes from
> Joining the tokens of many requests into one list comes from Orca (Yu et al., OSDI 2022), as **selective batching**. Every operation that treats tokens one by one is batched; only attention is split. Orca also picks the batch again at every step, which is **iteration-level scheduling**.
>
> Think of a bus that stops at every station: passengers get on and off at each stop. So vLLM keeps both ideas and adds paging, so that more passengers fit in the same bus.

> [!INFO] model.py: the arithmetic
> **What it does.** It runs OPT-125M for one step: the joined list of tokens goes in, one row of scores per request comes out. It is the "model shard" of Figure 4, with one shard holding the whole model.
>
> **Why it exists.** The transformers library keeps its own cache, one contiguous tensor per request, and cannot read a paged cache. So we copy the weights out of the library and write the forward pass by hand.
>
> **Where the idea comes from.** OPT is a family of open models from Meta AI (Zhang et al., 2022), from 125 million to 175 billion parameters. All sizes share one design, so our code for the smallest shows the same steps as OPT-13B.

Here is one iteration of our model, from `model.py`. The `Batch` holds the joined list of tokens, the position of each token in its own sequence, its slot, and the block tables. Before the code, look at what goes in, for the prompt step of requests A and B above:

{{FIG:p6_batch|The input of our prompt step. The 7 tokens of A and the 7 tokens of B form one list of 14. Each token carries its position in its own request and its slot in the cache. A prefill range and a block table tell attention which tokens belong together.}}

A and B both start at position 0, because a position counts inside one request. But their slots differ. A's prompt goes to slots 0 to 6, in blocks 0 and 1. B's prompt goes to slots 8 to 14, in blocks 2 and 3. Slot 7 stays empty until the next step. Now the code:

```python
    @torch.no_grad()
    def forward(self, batch, cache):
        """Run one iteration. Returns the logits (R, vocab) of the last token of each scheduled sequence."""
        T, H, D = batch.input_ids.shape[0], self.h, self.dh
        x = self.embed[batch.input_ids] + self.pos_embed[batch.positions + POS_OFFSET]    # (T, d)
        for l, W in enumerate(self.layers):
            # ---- attention: the only place where the tokens of one sequence meet
            a = F.layer_norm(x, (self.d,), W['self_attn_layer_norm.weight'], W['self_attn_layer_norm.bias'])
            q = F.linear(a, W['self_attn.q_proj.weight'], W['self_attn.q_proj.bias']).view(T, H, D)
            k = F.linear(a, W['self_attn.k_proj.weight'], W['self_attn.k_proj.bias']).view(T, H, D)
            v = F.linear(a, W['self_attn.v_proj.weight'], W['self_attn.v_proj.bias']).view(T, H, D)
            cache.write(l, batch.slots, k, v)                        # store the new keys and values first
            k_blocks, v_blocks = cache.blocks(l)
            o = torch.empty_like(q)
            for start, end, table in batch.prefills:                 # prompt phase: many tokens of one sequence
                o[start:end] = paged_attention(q[start:end], k_blocks, v_blocks, table, batch.positions[start:end])
            if len(batch.decode_idx):                                # generation phase: one token per sequence
                o[batch.decode_idx] = paged_attention_batched(q[batch.decode_idx], k_blocks, v_blocks,
                                                              batch.decode_tables, batch.decode_lens)
            x = x + F.linear(o.reshape(T, self.d), W['self_attn.out_proj.weight'], W['self_attn.out_proj.bias'])
            # ---- feed-forward: each token on its own (position-wise)
            f = F.layer_norm(x, (self.d,), W['final_layer_norm.weight'], W['final_layer_norm.bias'])
            f = F.linear(F.relu(F.linear(f, W['fc1.weight'], W['fc1.bias'])), W['fc2.weight'], W['fc2.bias'])
            x = x + f
        x = F.layer_norm(x[batch.last_idx], (self.d,), *self.final_ln)
        return x @ self.embed.T                                      # (R, vocab): the output layer reuses the embedding
```

Follow the 14 tokens of the picture. The first line turns each token and its position into 768 numbers, so `x` is (14, 768). In each layer, the layer norm and the three projections run over all 14 rows at once. `view(T, H, D)` cuts each row into 12 heads of 64.

Then `cache.write` stores the 14 new keys and values at their 14 slots, and only then does attention read them. The loop over `batch.prefills` runs twice. First it takes A's range (0, 7) with table [0, 1], then B's range (7, 14) with table [2, 3]. Each prompt token sees only the earlier tokens of its own request; this rule is the **causal mask**.

In a decoding step, the tokens go through `paged_attention_batched` instead, one per request, all in one call. The rest of the layer again runs over the whole list. At the end, `last_idx` is [6, 13], because only the last token of each request needs scores. So the output has 2 rows.

> [!DEFINITION] Logits
> The scores that a model gives to every token of its vocabulary as the next token, before the softmax turns them into probabilities. OPT has 50,272 logits per position. Greedy decoding takes the token with the largest logit.

Is it right? Method 2 is the library: transformers' own OPT with its own cache, which is one contiguous tensor per request. First we compare the logits at every step for one prompt. Then we serve 8 Alpaca prompts together and compare 48 greedy tokens of each with the library, which runs them one at a time. Alpaca is a public set of 52,000 instructions with answers, made to teach models to follow instructions. In the output, look for "identical":

```text
== 4. Our OPT on the paged cache against transformers (FP32, CPU)
  prompt of 9 tokens, then 5 decoding steps: max logit difference per step 2.0e-05, 6.7e-06, 4.8e-06, 4.3e-06, 5.7e-06, 3.8e-06
  greedy tokens: ours ['were', 'the', 'first', 'to', 'have', 'a']; library ['were', 'the', 'first', 'to', 'have', 'a']
  8 Alpaca prompts (8 to 21 tokens) served together, 48 greedy tokens each:
  identical outputs 8 of 8; identical tokens 384 of 384; steps 48
  first prompt: 'Give three tips for staying healthy.'
  our answer:   '\n\n1. Eat healthy.\n\nEat healthy. Eat healthy. Eat healthy. Eat healthy. Eat healthy.' ...
```

The logits differ by at most $$2 \times 10^{-5}$$, on values that are tens in size. That is rounding: we add and multiply in a different order from the library. All 384 tokens are identical, although the 8 prompts had different lengths and filled their blocks at different steps. The answer to "Give three tips for staying healthy." is poor, because OPT-125M is very small, but it is exactly the library's answer.

> [!PAPER] · §4.3 · page 6
> [![The paragraph on lazy allocation, with a new physical block is only allocated when all previous blocks, limits all the memory wastes for a request, within one block and its KV blocks can be freed highlighted](/img/papers/vllm/p6-one-block.png)](/img/papers/vllm/p6-one-block.png)
>
> **What it says:** a new "physical block is only allocated when all previous blocks" are full, so vLLM "limits all the memory wastes for a request within one block". When a request finishes, "its KV blocks can be freed" for other requests.
>
> **What it means:**
>
> The waste of a request is the empty end of its last block, and nothing else: with $$B = 16$$, at most 15 slots. There are no gaps between chunks either, because all blocks have the same size. Our engine frees a sequence's blocks in the step of its last token, so a waiting request can use them at once.

## Parallel sampling and copy-on-write {§4.4}

> [!DEFINITION] Greedy decoding and sampling
> **Greedy decoding** always takes the token with the largest logit, so the same prompt always gives the same answer. **Sampling** draws the next token at random from the probabilities. The **temperature** divides the logits first: below 1 the draw prefers likely tokens more, above 1 less. **Parallel sampling** draws several answers from one prompt.

> [!PAPER] · §4.4 · page 7
> [![The paragraph on parallel sampling, with reserve space for one copy of the, reference count for and each physical block highlighted](/img/papers/vllm/p6-refcount.png)](/img/papers/vllm/p6-refcount.png)
>
> **What it says:** when two samples share one prompt, vLLM reserves "space for one copy of the" prompt's state. The logical blocks of both samples map to the same physical blocks. Because one physical block can now belong to several logical blocks, vLLM keeps a "reference count for each physical block"; here both counts are 2.
>
> **What it means:**
>
> Part 2 showed that the keys and values of a prompt are the same for every sample, bit for bit. So two samples can point at the same blocks, as two people can read one book from a library shelf. The reference count records how many block tables point at a block, and a block is free only when it reaches 0.

> [!DEFINITION] Reference count
> A number kept for each shared object that says how many owners point at it. An owner who lets go lowers it by 1. At 0, nobody uses the object and its memory can be reused.

> [!PAPER] · §4.4 · page 7
> [![The copy-on-write paragraph, with copy-on-write mechanism at the block granularity, reference count of the corresponding, decreases the reference count to 1 and directly writes highlighted](/img/papers/vllm/p6-cow.png)](/img/papers/vllm/p6-cow.png)
>
> **What it says:** vLLM uses a "copy-on-write mechanism at the block granularity". When sample A1 must write to its last logical block, vLLM sees that the "reference count of the corresponding" physical block is above 1. It allocates a new block, copies the data, and "decreases the reference count to 1". Then sample A2 "directly writes" to the old block.
>
> **What it means:**
>
> Two samples share the last prompt block, and it still has free slots. Each sample wants to write its own new token there; if both did, each would spoil the other's cache. So the first writer makes a private copy. The old block then has one owner, and the second writer needs no copy. Only one block is copied, not the whole prompt.

> [!DEFINITION] Copy-on-write
> A rule for shared memory: share it while everyone only reads, and make a private copy at the moment someone wants to change it. Operating systems use it when a process forks; vLLM uses it for KV blocks.

> [!NOTE] Where this idea comes from
> In Unix, the call `fork()` makes a new process as a copy of a running one. Copying all its memory at once would be slow. So modern Unix systems let parent and child share every page, and copy a page only when one of them first writes to it.
>
> Reference counts are just as common: Python itself frees an object when its count reaches 0. So vLLM puts both ideas on KV blocks. Samples share blocks, a count says how many, and a write to a shared block makes a copy.

> [!INFO] sequence.py: requests and their samples
> **What it does.** It defines a request, called a **sequence group**, its sequences, one per sample, and the sampling settings.
>
> **Why it exists.** The block manager works per sequence, but the scheduler must start and stop the samples of one request together. So the code needs both levels.
>
> **Where the idea comes from.** vLLM's own code has a `SequenceGroup` class for the same purpose. Running related tasks together is **gang scheduling**, a term from parallel computers.

The data classes, from `sequence.py`. A `Sequence` is one answer; `num_cached` counts how many of its tokens have keys and values in the cache:

```python
@dataclass
class SamplingParams:
    n: int = 1                       # number of samples from one prompt (parallel sampling)
    max_tokens: int = 16             # stop after this many new tokens
    temperature: float = 0.0         # 0 means greedy: always take the most likely token
    ignore_eos: bool = False         # True: keep writing past </s> (to replay known output lengths)
    seed: int = 0


class Sequence:
    """One prompt plus the tokens written so far, and how many of them have keys and values in the cache."""

    def __init__(self, seq_id, token_ids, prompt_len, group):
        self.seq_id = seq_id
        self.token_ids = list(token_ids)
        self.prompt_len = prompt_len
        self.group = group
        self.num_cached = 0          # tokens whose keys and values are stored in the KV cache
        self.status = WAITING
        self.logprobs = []           # log-probability of each written token, for the checks

    @property
    def output_ids(self):
        return self.token_ids[self.prompt_len:]


class SequenceGroup:
    """One request. Its sequences are scheduled and preempted together (gang scheduling, §4.5)."""

    def __init__(self, request_id, params, arrival):
        self.request_id, self.params, self.arrival = request_id, params, arrival
        self.seqs = []
        self.first_step = self.finish_step = None

    def unfinished(self):
        return [s for s in self.seqs if s.status != FINISHED]
```

A sequence keeps its prompt and its answer in one list, `token_ids`; `prompt_len` says where the answer starts. A request with `n = 4` starts as one sequence and becomes four after its prompt step. `arrival` is its place in the queue.

> [!PAPER] · Figure 8 · page 7
> [![Figure 8 of the paper: samples A1 and A2 share physical blocks 7 and 1; A1 gets a copy in physical block 3 and writes fathers, A2 writes mothers in physical block 1, whose reference count goes from 2 to 1](/img/papers/vllm/p6-figure8.png)](/img/papers/vllm/p6-figure8.png)
>
> **What it says:** "Parallel sampling example."
>
> **What it means:**
>
> Both samples point at physical blocks 7 and 1. Sample A1 writes "fathers": block 1 is shared, so its content is copied to block 3 and A1 writes there. Sample A2 writes "mothers" into block 1, which it now owns alone. The shared block 7 is never copied, because neither sample writes into it again.

In our code, copy-on-write is the middle branch of `append_slot` above. It needs only two more pieces. `fork` makes a new sequence that shares every block of its parent, and `copy_blocks` in the cache does the copies that `append_slot` asked for:

```python
    def fork(self, parent_id, child_id):
        """A new sequence that shares every block of its parent (parallel sampling, §4.4)."""
        p = self.tables[parent_id]
        for b in p.blocks:
            self.allocator.share(b)
        self.tables[child_id] = BlockTable(list(p.blocks), p.num_tokens)
```

```python
    def copy_blocks(self, pairs):
        """Copy-on-write (§4.4): copy whole blocks src -> dst, in every layer, keys and values."""
        if pairs:
            src = torch.tensor([s for s, _ in pairs])
            dst = torch.tensor([d for _, d in pairs])
            self.data[:, :, dst] = self.data[:, :, src]       # one batched copy, as in the fused kernel of §5.1
```

Take a parent with table [0, 1]. `fork` raises the counts of blocks 0 and 1 from 1 to 2. The child gets its own list, [0, 1], a copy of the numbers only. No key or value moves.

`copy_blocks` takes pairs such as (1, 2): "copy block 1 into block 2". The `:` on the first two axes means "in every layer, keys and values". All the pairs of a step are done in one indexed copy.

The engine runs the prompt once, with one sequence. Right after that step it forks the other samples, and each one samples its own first token from the same logits:

```python
            if g.first_step is None:                           # first step of a request with n samples:
                g.first_step = self.step_no                    # fork n - 1 children that share every block
                for _ in range(g.params.n - 1):
                    child = self._new_seq(seq.token_ids, seq.prompt_len, g)
                    child.num_cached, child.status = seq.num_cached, RUNNING
                    self.bm.fork(seq.seq_id, child.seq_id)
                    g.seqs.append(child)
                    self._sample(logits[row], child)
            self._sample(logits[row], seq)
```

For $$n = 2$$, the loop runs once. It makes a child with the same tokens and forks the block table. Then child and parent each draw a token from the same logits, each with its own random generator.

We replay Figure 8 with the real model: the same 7-token prompt, block size 4, two samples drawn at random (temperature 1). In the output, follow the block lists and the reference counts from step 0 to step 2:

```text
== 5. Parallel sampling and copy-on-write (Figure 8): 2 samples, block size 4
  step 0: A1 blocks [0, 1], A2 blocks [0, 1]; ref counts {0: 2, 1: 2}; copies []
  step 1: A1 blocks [0, 2], A2 blocks [0, 1]; ref counts {0: 2, 1: 1, 2: 1}; copies [(1, 2)]
  step 2: A1 blocks [0, 2, 3], A2 blocks [0, 1, 4]; ref counts {0: 2, 1: 1, 2: 1, 3: 1, 4: 1}; copies []
  sample A1: ': Tahitians reach WC 180\nTony Kanaan'
  sample A2: " right now your dad hadn't lived in Bortles easily"
  log-probability of all 24 sampled tokens, ours vs the library on the full text: max difference 6.2e-06
```

{{FIG:p6_cow|Our run of Figure 8. Left: after the prompt step both samples point at blocks 0 and 1, each with a reference count of 2. Right: after the first decoding step, A1 has a copy of block 1 in block 2, and A2 still writes into block 1.}}

It is Figure 8, line for line. After the prompt, both samples point at blocks 0 and 1, and both counts are 2. In step 1, A1 writes first. Block 1 is shared, so it is copied to block 2, A1 moves to block 2, and the count of block 1 drops to 1. A2 then writes into block 1 with no copy.

Block 0 stays shared to the end. In step 2 both last blocks are full, so each sample takes a fresh block of its own.

The last line checks that sharing did not damage anything. For each sample we ran the library's model once on the prompt plus all 12 written tokens. We compared its log-probability for every sampled token with the one our engine had when it sampled it. They agree to $$6.2 \times 10^{-6}$$, so each sample read its own correct cache, shared blocks and copied block included.

How much memory does sharing save? We gave 16 Alpaca prompts 4 samples each, with output lengths up to 64. At every step we counted the blocks held, and the blocks the same sequences would hold with no sharing:

```text
== 8. Parallel sampling, 4 samples per prompt: blocks held with sharing against without (16 Alpaca prompts)
  63 steps; mean blocks held 107.0, without sharing 118.9; mean saving 11.2%; copy-on-write copies 42
```

Sharing saved 11.2% of the blocks on average. The paper measured 6.1% to 9.8% for parallel sampling on Alpaca with OPT-13B (Figure 15, Part 5). It used the same formula: blocks saved over blocks without sharing. Alpaca prompts are short, so only a small part of each sequence is shared; long prompts save much more.

The run made 42 block copies. A prompt whose last block has room makes 3. The first three samples to write each make a copy; the fourth finds a count of 1 and writes in place.

> [!PAPER] · §4.4 · page 8
> [![The paragraph on mixed decoding methods, with common mapping layer and only see a list of physical block IDs highlighted](/img/papers/vllm/p6-mixed.png)](/img/papers/vllm/p6-mixed.png)
>
> **What it says:** vLLM hides all sharing behind "a common mapping layer" from logical to physical blocks. The model and its kernel "only see a list of physical block IDs" for each sequence.
>
> **What it means:**
>
> Look back at `forward`. It has no idea that two samples share a prompt, or that a block was copied. It receives slots and block tables, and does arithmetic. So one batch can mix greedy requests, sampled requests and recomputed requests, and all the complexity stays in the block manager.

## Scheduling and preemption {§4.5}

> [!PAPER] · §4.5 · page 8
> [![The opening of §4.5, with first-come-first-serve (FCFS) scheduling policy, arrived requests are served first and the latest requests are and preempted first highlighted](/img/papers/vllm/p6-fcfs.png)](/img/papers/vllm/p6-fcfs.png)
>
> **What it says:** vLLM uses "the first-come-first-serve (FCFS) scheduling policy". When it must preempt, the earliest "arrived requests are served first and the latest requests are preempted first".
>
> **What it means:**
>
> Blocks are taken one at a time, as answers grow, so a server can run out of blocks halfway. Then the rule is the queue at a bakery. The customer who came first is served first. When there is not enough bread, the one who came last steps back and waits, so nobody waits forever.

> [!DEFINITION] Preemption
> Taking resources away from a running task before it is done, so that others can continue. The task goes back to the queue and resumes later.

> [!PAPER] · §4.5 · page 8
> [![The paragraph on eviction, with Which blocks should it evict, How to recover, all-or-nothing eviction policy and gang-scheduled as a sequence highlighted](/img/papers/vllm/p6-evict.png)](/img/papers/vllm/p6-evict.png)
>
> **What it says:** two questions follow: "Which blocks should it evict?" and "How to recover" them. Because all blocks of a sequence are used together, vLLM uses an "all-or-nothing eviction policy". The sequences of one request, such as its samples, are "gang-scheduled as a sequence group" and are preempted together.
>
> **What it means:**
>
> An operating system can take away a single page that a program may not touch for a while. A sequence touches all its blocks at every step, since attention reads the whole context. So vLLM takes all of them, and the samples of a request go together, because they share blocks.

> [!PAPER] · §4.5 · page 8
> [![The recomputation paragraph, with recompute the KV cache when the preempted sequences are rescheduled, concatenated with the original user prompt as a new and one prompt phase iteration highlighted](/img/papers/vllm/p6-recompute.png)](/img/papers/vllm/p6-recompute.png)
>
> **What it says:** with recomputation, vLLM simply "recompute[s] the KV cache when the preempted sequences are rescheduled". The tokens already written are "concatenated with the original user prompt as a new" prompt, so the whole cache comes back in "one prompt phase iteration".
>
> **What it means:**
>
> The cache is only a record of work, and the work can be done again; the written tokens themselves are kept. Writing 100 tokens took 100 passes of the model, but reading them back as a prompt takes one pass. So rebuilding costs far less than building.

> [!INFO] scheduler.py: who runs in this step
> **What it does.** Before every step, it picks which running requests continue, which waiting requests start, and which request must give up its blocks. It hands the engine a plan with the slot of every new token.
>
> **Why it exists.** Paging books memory as answers grow, so nobody knows in advance whether all requests will fit. Something must check at every step.
>
> **Where the idea comes from.** First come, first served is the oldest rule for a queue. Preemption comes from operating systems, which stop a program to give the processor to another. Recomputing instead of storing is known from training as "gradient checkpointing" (Chen et al., 2016).

Our scheduler, from `scheduler.py`. `preempt` is recomputation: it frees every block of the request and puts it back in the queue by arrival time. `schedule` picks the work of one step. The two numbered comments in `schedule` are its two parts:

```python
    def preempt(self, group):
        """Recomputation: free every block of the group (all or nothing), and queue it again in arrival order.
        Later its prompt plus the tokens it already wrote are computed again as one longer prompt."""
        for s in group.unfinished():
            self.bm.free(s.seq_id)
            s.num_cached, s.status = 0, WAITING
        self.running.remove(group)
        i = 0
        while i < len(self.waiting) and self.waiting[i].arrival < group.arrival:
            i += 1
        self.waiting.insert(i, group)                 # it arrived early, so it goes near the front
        self.num_preemptions += 1

    def schedule(self):
        """Pick the work of one iteration. Running requests first (oldest first), then new ones (FCFS)."""
        out = SchedulerOutput()
        # 1. Each running sequence needs a slot for the token it feeds now. When the blocks run out,
        #    preempt the request that arrived last, until the older ones fit.
        self.running.sort(key=lambda g: g.arrival)
        queue = list(self.running)
        while queue:
            g = queue.pop(0)
            seq_ids = [s.seq_id for s in g.unfinished()]
            while not self.bm.can_append(seq_ids):
                victim = queue.pop() if queue else g
                self.preempt(victim)
                out.preempted.append(victim)
                if victim is g:
                    break
            else:
                out.decode += [(s, self.bm.append_slot(s.seq_id)) for s in g.unfinished()]
        # 2. Admit waiting requests in arrival order while their tokens fit. Not in a step that preempted.
        budget = self.max_prefill_tokens
        num_seqs = sum(len(g.unfinished()) for g in self.running)
        while self.waiting and not out.preempted:
            g = self.waiting[0]
            lengths = [len(s.token_ids) for s in g.unfinished()]
            if not g.first_step and g.params.n > 1:   # samples not forked yet: count them for a non-sharing manager
                lengths += lengths[:1] * (g.params.n - 1)
            if (num_seqs + len(lengths) > self.max_num_seqs or sum(lengths) > budget
                    or not self.bm.can_allocate(lengths)):
                break
            self.waiting.popleft()
            budget -= sum(len(s.token_ids) for s in g.unfinished())
            num_seqs += len(lengths)
            for s in g.unfinished():
                out.prefill.append((s, self.bm.allocate(s.seq_id, len(s.token_ids))))
                s.status = RUNNING
            self.running.append(g)
        out.copies, self.bm.copies = self.bm.copies, []
        return out
```

The schedule runs every step: this is iteration-level scheduling, as in Part 2. Take three running requests, R0, R1 and R2, in arrival order, and one free block. R0 and R1 each need a new block for their next token.

Part 1 of `schedule` takes R0 first, and `append_slot` gives it the free block. Then R1 asks, and no block is left. The `while` loop preempts the last request in the queue, R2. Its blocks return to the free list, and R1 now fits.

Part 2 admits waiting requests in arrival order while their blocks fit, and stops at the first one that does not. It admits nothing in a step that preempted. `max_num_seqs` and `budget` cap a step at 256 sequences and 4,096 prompt tokens.

A recomputed request needs no special code. Its `token_ids` hold its prompt plus the tokens it wrote. So `allocate` treats them as one longer prompt, and the next step rebuilds the whole cache in one pass.

Now we squeeze the memory. Six Alpaca requests must each write 64 greedy tokens. The cache has 12 blocks of 16, which is 192 slots, while the six requests need about 460 slots at the end. Then we compare all outputs with the library, which never preempts. In the output, watch which request is preempted (always the newest running one), and the line that says "identical":

```text
== 6. Scheduling and preemption by recomputation: 6 requests, 64 greedy tokens each, a cache of 12 blocks of 16
  step  10: preempt request 5 (it has 10 new tokens); free blocks now 1
  step  24: preempt request 4 (it has 24 new tokens); free blocks now 2
  step  26: preempt request 3 (it has 26 new tokens); free blocks now 2
  step  42: preempt request 2 (it has 42 new tokens); free blocks now 3
  step  64: request 2 back: recompute 49 tokens (7 prompt + 42 written) in one pass
  step  64: request 3 back: recompute 34 tokens (8 prompt + 26 written) in one pass
  step  64: request 4 back: recompute 40 tokens (16 prompt + 24 written) in one pass
  step  64: request 5 back: recompute 18 tokens (8 prompt + 10 written) in one pass
  step  73: preempt request 5 (it has 19 new tokens); free blocks now 1
  step  80: preempt request 4 (it has 40 new tokens); free blocks now 3
  ... 4 more events
  prompts [9, 24, 7, 8, 16, 8] tokens; 138 steps; 7 preemptions
  outputs identical to the library (which never preempts): 6 of 6
  the cache of 128 tokens: one pass 54 ms; a 16-token prompt then 112 tokens one by one 1490 ms (28x longer)
```

{{FIG:p6_preempt|Our preemption run. Each row is a request and each column a step. Filled cells: the request runs. Hatched cells: it waits after a preemption, until enough blocks are free to recompute its cache in one pass.}}

All six requests start at step 0. As their answers grow, the cache fills, and at step 10 request 5, the last to arrive, is preempted with 10 tokens written. Requests 4, 3 and 2 follow at steps 24, 26 and 42. Each time the newest running request gives way.

At step 63, requests 0 and 1 write their 64th token and free their blocks. In the next step, the four waiting requests come back. Each one rebuilds its whole cache in one pass: request 2 recomputes 49 tokens, its 7-token prompt plus the 42 tokens it had written. The memory is still too small for all four to finish together, so requests 5 and 4 are preempted again later. In total there are 7 preemptions, and the work takes 138 steps instead of 64.

The line that matters is the last but one: all six outputs are identical to the library's, which never preempts. Throwing away a cache and computing it again changes nothing in the result; it only costs time.

And that time is small. The last line compares the two ways to build the cache of 128 tokens. Computing it in one pass took 54 ms. Writing it token by token took 1,490 ms, 28 times longer.

> [!NOTE] Where this idea comes from
> Swapping, the other recovery of §4.5, is the classic one. An operating system short of memory writes pages to disk and reads them back later. In vLLM, KV blocks go to CPU memory instead, through the CPU block allocator of Figure 4.

We did not build swapping. On our laptop, an Apple M-series machine, the CPU and the GPU share one memory, so there is nothing to swap to. Part 4 explains swapping and Part 5 compares the two methods.

## One engine step, all pieces together {§4.3, §5.2}

> [!INFO] engine.py: the loop
> **What it does.** It runs every other piece in order, once per step: schedule, copy blocks, build the batch, run the model, sample, tidy up.
>
> **Why it exists.** Each piece answers one question. Somebody must ask the questions in the right order. In Figure 4 this is not a box; it is the arrows.
>
> **Where the idea comes from.** A loop that repeats "take work, do one slice, give results" is how operating system schedulers and network servers run. The vLLM code has an `LLMEngine` class with a `step` method, and our names follow it.

Here is the whole step, from `engine.py`. You have met every call in it. Read it once from top to bottom, then compare it with the picture below:

```python
    def step(self):
        out = self.scheduler.schedule()
        self.cache.copy_blocks(out.copies)                    # copy-on-write copies, before anyone writes
        if not out.prefill and not out.decode:
            if self.scheduler.has_work():
                raise RuntimeError('a request does not fit in the KV cache even when it runs alone')
            return out
        batch, order = self._batch(out)
        logits = self.model.forward(batch, self.cache)        # (sequences, vocab)
        self.last_logits = {seq.seq_id: logits[row] for row, seq in enumerate(order)}   # kept for the checks
        for row, seq in enumerate(order):
            seq.num_cached = len(seq.token_ids)
            g = seq.group
            if g.first_step is None:                           # first step of a request with n samples:
                g.first_step = self.step_no                    # fork n - 1 children that share every block
                for _ in range(g.params.n - 1):
                    child = self._new_seq(seq.token_ids, seq.prompt_len, g)
                    child.num_cached, child.status = seq.num_cached, RUNNING
                    self.bm.fork(seq.seq_id, child.seq_id)
                    g.seqs.append(child)
                    self._sample(logits[row], child)
            self._sample(logits[row], seq)
        for g in list(self.scheduler.running):
            if not g.unfinished():
                g.finish_step = self.step_no
                self.scheduler.running.remove(g)
        self.stats.append(dict(step=self.step_no, prefill_tokens=sum(len(s.token_ids) - 1 for s, _ in out.prefill),
                               decode=len(out.decode), running=sum(len(g.unfinished()) for g in self.scheduler.running),
                               blocks_held=self.bm.blocks_held(), blocks_no_sharing=self.bm.blocks_without_sharing(),
                               tokens=self.bm.tokens_stored(), preempted=len(out.preempted), copies=len(out.copies)))
        self.step_no += 1
        return out
```

{{FIG:p6_step|Decoding step 1 of our two-request run, stage by stage. The scheduler finds one free place in each last block. The batch has 2 tokens. The model writes their keys and values and reads blocks through the tables. Greedy sampling picks "the" for both. Nothing finishes, so nothing is freed.}}

The picture is decoding step 1 of the run with requests A and B. First, `schedule` returns the plan: A's next token goes to slot 7 and B's to slot 15. Second, any copies are made, before any write. If no request fits even alone, the engine stops with an error.

Third, `_batch` joins the 2 tokens into one `Batch`, and `forward` returns 2 rows of logits. Fourth, `_sample` picks each next token; a new request with several samples forks here. Last, finished requests leave the running list. Their blocks were freed already, inside `_sample`. The `stats` line records the numbers for our charts.

The method `run` repeats `step` until no request is left. That is the whole engine.

## Fork, append, free, and the kernels {§5}

> [!PAPER] · §5 · page 9
> [![The opening of §5, with 8.5K lines of Python and 2K lines of, scheduler and the block manager in Python and custom CUDA kernels for key operations highlighted](/img/papers/vllm/p6-impl.png)](/img/papers/vllm/p6-impl.png)
>
> **What it says:** the vLLM engine has "8.5K lines of Python and 2K lines of" C++ and CUDA. The authors wrote the "scheduler and the block manager in Python" and "custom CUDA kernels for key operations" such as PagedAttention.
>
> **What it means:**
>
> The split is the same as ours. Decisions about memory happen once per step, so Python is fast enough. Arithmetic on the cache happens billions of times, so it needs GPU code. Our 501 lines are all Python, because PyTorch's indexing does our block reads and writes.

> [!PAPER] · §5.2 · page 9
> [![Section 5.2, with fork, append, and free, creates a new sequence from an existing one, appends a new token to the sequence and deletes the sequence highlighted](/img/papers/vllm/p6-fork.png)](/img/papers/vllm/p6-fork.png)
>
> **What it says:** vLLM supports decoding algorithms with three methods: "fork, append, and free". Fork "creates a new sequence from an existing one", append "appends a new token to the sequence", and free "deletes the sequence".
>
> **What it means:**
>
> These three verbs are a small language. Parallel sampling is one fork per extra sample, then appends, then frees. Beam search forks the good beams and frees the bad ones at every step. Our `BlockManager` has exactly these three, as `fork`, `append_slot` and `free`, plus `allocate` for the first prompt.

> [!PAPER] · §5.1 · page 9
> [![Section 5.1, with Fused reshape and block write, Fusing block read and attention, Fused block copy and batches the copy operations for different blocks into a single highlighted](/img/papers/vllm/p6-kernels.png)](/img/papers/vllm/p6-kernels.png)
>
> **What it says:** three kernels make paging fast: "Fused reshape and block write", "Fusing block read and attention", and "Fused block copy", which "batches the copy operations for different blocks into a single" kernel launch.
>
> **What it means:**
>
> Each kernel has a twin in our code. `cache.write` stores all new keys and values of a step with one indexed write. `paged_attention_batched` reads the blocks and computes attention for all running sequences in one call. `copy_blocks` does all copies of a step at once.
>
> The difference is "fused". Our versions gather the blocks into a new tensor, then compute. A fused CUDA kernel reads each block straight into the arithmetic, in one GPU function, which saves memory traffic.

> [!NOTE] Where this idea comes from
> Each GPU kernel has a start-up cost, and reads and writes GPU memory. **Kernel fusion** joins several small steps into one kernel, so the data stays on the chip between them. NVIDIA's FasterTransformer, one of the paper's baselines, is built on fused kernels. So vLLM adds fused kernels that understand block tables.

## Serving real requests

> [!NOTE] Where this idea comes from
> A server is judged by **throughput**, the work it finishes per second, and **latency**, how long one user waits. Part 5 shows how the paper measures both, with requests sent at a fixed rate. Our test is simpler: all requests arrive at time 0, and we time the whole queue. That measures throughput only.

Now the whole engine against the baseline of §3.1. We take 64 Alpaca instructions as prompts. Each request writes as many tokens as the dataset's own answer has, capped at 128. It ignores the end token, so both runs do exactly the same work. The paper builds its workloads from dataset lengths in the same way.

Both runs use the same code and the same KV cache of 128 blocks of 16, which is 2,048 slots. Only the block manager differs. The paged run uses `BlockManager`. The contiguous run uses `ContiguousManager`. It books one chunk of 512 slots for each request when it starts, like the systems of §3.1 with a maximum length of 512. (The real OPT maximum is 2,048; we are kind to the baseline.)

`ContiguousManager` answers the same questions as `BlockManager`, so the engine can use either. Its `allocate` shows the difference:

```python
    def allocate(self, seq_id, num_tokens):
        start = self._find_chunks(1)[0]
        self.used[start:start + self.chunk] = [True] * self.chunk
        t = BlockTable(list(range(start, start + self.chunk)), num_tokens)
        self.tables[seq_id] = t
        return [t.slot(p, self.block_size) for p in range(num_tokens)]
```

`self.chunk` is $$512 / 16 = 32$$ blocks. `_find_chunks` looks for 32 free blocks in a row and returns where the run starts. The request gets all 32 at once, whatever it will need, and its table is simply 32 consecutive numbers. A request that writes 20 tokens still holds 512 slots until it finishes.

The picture shows what this means for the cache, after the first step of each run:

{{FIG:p6_first|The same 2,048-slot cache after the first step. With one 512-slot chunk per request, 4 requests fill the whole cache, and almost all of it is booked but empty. With blocks of 16, 63 requests run, most booked slots hold a token, and some blocks are still free.}}

With chunks, 4 requests fill all 2,048 slots, but only 81 slots hold a token. With blocks, 63 requests run at once, 1,324 slots hold a token, and 272 slots are still free.

> [!DEFINITION] Throughput
> The amount of work a server finishes per second. Here it is generated tokens per second, or finished requests per second.

The table below has one row per run. Look first at "mean batch", then at "steps", then at "tokens/s":

```text
== 7. Serving 64 Alpaca requests: KV cache of 128 blocks x 16 = 2048 token slots
  prompts: mean 21.2 tokens; outputs: the length of the dataset answer, at most 128, mean 30.0; 1917 tokens to write
               seconds  tokens/s  requests/s  steps  mean batch  max batch  slots holding a token  preemptions
  contiguous      25.2        76        2.54    494         3.8          4                  10.2%            0
  paged            7.4       259        8.64    128        14.5         63                  87.3%            0
  runs (seconds): contiguous 25.9, 25.2; paged 7.4, 7.4
  paged / contiguous: throughput x3.4, mean batch x3.9
  the KV cache is 151 MB in both runs (72 KB per token in FP32); booking 512 slots each, the paged mean batch would need 546 MB
```

{{FIG:p6_serving|The number of sequences in the batch at each step of our serving run. The contiguous run never holds more than 4, because 4 chunks of 512 fill the cache. The paged run starts 63 of the 64 requests in its first step and finishes in far fewer steps.}}

{{FIG:p6_memory|Of the KV slots that each run had booked, the share that held a token, averaged over the steps.}}

The paged engine finished the same 1,917 tokens 3.4 times faster: 259 tokens per second against 76. The reason is in the batch column. With 512 slots booked per request, the 2,048-slot cache holds only 4 requests. So the contiguous run worked through the queue 4 at a time, in 494 steps. The paged run gave each request only the blocks it filled. It started 63 requests in the first step and finished in 128 steps, the length of the longest answer.

Each paged step did more work, but it did not take 3.9 times longer. Part 2 showed why: a weight matrix takes almost the same time for 1 token as for 64. More requests per step from the same memory means more throughput.

The memory column says where the gain comes from. In the contiguous run, only 10.2% of the booked slots held a token, on average over the steps. The rest was reserved or never used, the waste of Figure 3.

In the paged run, 87.3% held a token. The rest is the empty end of last blocks, at most 15 slots per sequence. The paper reports 20.4% to 38.2% of real token states for its baselines and 96.3% for vLLM (Figure 2, Part 1). Our requests are much shorter than ShareGPT conversations, so one partial block per request weighs more.

Seen the other way around, take the paged run's mean batch of 14.5 requests. With 512 slots booked each, the contiguous engine would need 546 MB of KV cache for it, instead of 151 MB. The timings are noisy, because the laptop was busy with other work. The step counts and the memory shares do not depend on the timing at all.

## What this code does not show

Our engine runs one small model in FP32 on a laptop CPU; the paper serves OPT-13B to OPT-175B in FP16 on A100 GPUs. The checks against the library are measured, not estimated. The timings are the fastest of two runs on a busy laptop, so treat the ratios as shapes, not server numbers.

We left out swapping, beam search, shared prefixes, many GPUs, the HTTP front end and the CUDA kernels. Our requests all arrive at time 0, while the paper sends them at a random rate and measures latency per token.

> [!TAKEAWAYS] Key takeaways
> - The core of vLLM fits in about **500 lines of Python**: blocks, reference counts, block tables, PagedAttention, a scheduler and a loop. Each file is one box of Figure 4.
> - **Eq. 4** is ordinary attention grouped by block. Our block-by-block loop matches PyTorch's attention to $$4.2 \times 10^{-7}$$, and moving the blocks elsewhere changes the output by exactly 0.
> - **A block table** is the address translation of virtual memory: token $$p$$ lives in physical block `blocks[p // B]` at place `p % B`.
> - **Blocks are taken only when a token needs one** (§4.3), as our replay of Figures 6 and 7 shows.
> - **Our OPT on the paged cache equals the library**: logits to $$2 \times 10^{-5}$$ and 384 of 384 greedy tokens identical, for 8 requests served together.
> - **Copy-on-write** copies one block, once, when two samples first write to a shared block. Sharing saved 11.2% of the blocks for 4 samples on Alpaca prompts; the paper reports 6.1% to 9.8%.
> - **Preemption by recomputation** drops a request's blocks and rebuilds them later in one pass. In a 192-slot cache, 6 requests went through 7 preemptions and still matched the library exactly.
> - **Paging raises throughput by fitting more requests.** In the same 2,048-slot cache, 64 Alpaca requests ran 3.4 times faster paged than with 512 slots booked each. 87.3% of booked slots held a token, against 10.2%.

<details>
<summary>Run it yourself</summary>

The package is [`code/papers/vllm/minivllm`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm/minivllm), and the script that runs every check on this page is [`code/papers/vllm/vllm_part6.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part6.py). Every number is saved in `results/part6.json`. On an Apple M-series laptop the script takes about five minutes, most of it in the contiguous serving run.

```bash
pip install torch transformers datasets
cd code/papers/vllm
python vllm_part6.py   # downloads facebook/opt-125m (about 250 MB) and tatsu-lab/alpaca
```

To use the engine on your own prompts:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from minivllm import LLMEngine, SamplingParams

tok = AutoTokenizer.from_pretrained('facebook/opt-125m')
hf = AutoModelForCausalLM.from_pretrained('facebook/opt-125m', dtype=torch.float32)
engine = LLMEngine(hf, num_blocks=128, block_size=16)
for p in ['The capital of France is', 'Write a poem about the sea.']:
    engine.add_request(tok(p).input_ids, SamplingParams(n=2, max_tokens=30, temperature=0.8))
engine.run()
for g in engine.groups:
    for s in g.seqs:
        print(tok.decode(s.output_ids))
```

<figure><img src="/img/papers/vllm/part6-run.png" alt="Terminal output of vllm_part6.py: the package size, the block tables of Figures 6 and 7, the Eq. 4 check, the comparison with the library, copy-on-write, preemption, the serving run and the sharing count" loading="lazy" /><figcaption>The output of vllm_part6.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our minivllm package and scripts](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
