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

<figure class="fig"><svg viewBox="0 0 720 430" role="img" aria-label="The vLLM architecture of Figure 4 as a map. Requests enter on the left (sequence.py). The scheduler (scheduler.py) holds the KV cache manager (block_manager.py) with block tables and the GPU block allocator; the CPU block allocator is not built. Worker 0 holds the cache engine (kv_cache.py) and the model (model.py, attention.py); workers 1 to N-1 are not built. engine.py runs the loop."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Figure 4 of the paper, with the file of minivllm that builds each box</text><rect class="box-ghost" x="10.0" y="38.0" width="700.0" height="336.0" rx="12"/><text class="t-label" x="24.0" y="58.0" text-anchor="start">engine.py: the loop that drives every step (LLMEngine)</text><rect class="box-on" x="24.0" y="170.0" width="130.0" height="96.0" rx="10"/><text class="t-note" x="89.0" y="196.0" text-anchor="middle">Requests</text><text class="t-mono" x="89.0" y="218.0" text-anchor="middle">sequence.py</text><text class="t-muted" x="89.0" y="238.0" text-anchor="middle">add_request()</text><text class="t-muted" x="89.0" y="254.0" text-anchor="middle">prompt + settings</text><rect class="box-1" x="184.0" y="72.0" width="290.0" height="290.0" rx="12"/><text class="t-note" x="198.0" y="96.0" text-anchor="start">Scheduler</text><text class="t-mono" x="460.0" y="96.0" text-anchor="end">scheduler.py</text><rect class="box" x="198.0" y="110.0" width="262.0" height="238.0" rx="10"/><text class="t-note" x="212.0" y="132.0" text-anchor="start">KV cache manager</text><text class="t-mono" x="212.0" y="150.0" text-anchor="start">block_manager.py</text><rect class="box-on" x="212.0" y="162.0" width="234.0" height="50.0" rx="8"/><text class="t-tick t-strong" x="224.0" y="183.0" text-anchor="start">Block tables</text><text class="t-muted" x="224.0" y="201.0" text-anchor="start">BlockTable</text><rect class="box-on" x="212.0" y="222.0" width="234.0" height="50.0" rx="8"/><text class="t-tick t-strong" x="224.0" y="243.0" text-anchor="start">GPU block allocator</text><text class="t-muted" x="224.0" y="261.0" text-anchor="start">BlockAllocator</text><rect class="box-ghost" x="212.0" y="282.0" width="234.0" height="50.0" rx="8"/><text class="t-tick t-strong" x="224.0" y="303.0" text-anchor="start">CPU block allocator</text><text class="t-muted" x="224.0" y="321.0" text-anchor="start">not built (no swapping)</text><rect class="box-1" x="508.0" y="72.0" width="190.0" height="190.0" rx="12"/><text class="t-note" x="522.0" y="96.0" text-anchor="start">Worker 0</text><rect class="box-on" x="522.0" y="108.0" width="162.0" height="62.0" rx="8"/><text class="t-tick t-strong" x="534.0" y="130.0" text-anchor="start">Cache engine</text><text class="t-mono" x="534.0" y="150.0" text-anchor="start">kv_cache.py</text><rect class="box-on" x="522.0" y="180.0" width="162.0" height="70.0" rx="8"/><text class="t-tick t-strong" x="534.0" y="202.0" text-anchor="start">Model shard</text><text class="t-mono" x="534.0" y="222.0" text-anchor="start">model.py</text><text class="t-mono" x="534.0" y="240.0" text-anchor="start">attention.py</text><rect class="box-ghost" x="508.0" y="276.0" width="190.0" height="86.0" rx="12"/><text class="t-note" x="522.0" y="300.0" text-anchor="start">Workers 1 to N-1</text><text class="t-muted" x="522.0" y="320.0" text-anchor="start">not built: one process,</text><text class="t-muted" x="522.0" y="338.0" text-anchor="start">one device, no all-reduce</text><line class="edge" x1="156.0" y1="218.0" x2="182.0" y2="218.0" marker-end="url(#ah)"/><line class="edge" x1="476.0" y1="160.0" x2="506.0" y2="160.0" marker-end="url(#ah)"/><line class="edge" x1="506.0" y1="228.0" x2="476.0" y2="228.0" marker-end="url(#ah)"/><text class="t-muted" x="491.0" y="150.0" text-anchor="middle">slots</text><text class="t-muted" x="491.0" y="246.0" text-anchor="middle">logits</text><rect class="box-on" x="20.0" y="392.0" width="22.0" height="16.0" rx="4"/><text class="t-tick" x="50.0" y="405.0" text-anchor="start">built in minivllm</text><rect class="box-ghost" x="190.0" y="392.0" width="22.0" height="16.0" rx="4"/><text class="t-tick" x="220.0" y="405.0" text-anchor="start">in the paper, not built here</text></svg><figcaption>Figure 4 redrawn as a map of our package. Requests enter on the left. The scheduler holds the KV cache manager, with its block tables and its GPU block allocator. Worker 0 holds the cache engine and the model. engine.py is the loop that drives all of them. Dashed boxes are in the paper but not in our code.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 446" role="img" aria-label="A sequence diagram of one step. The engine asks the scheduler for work; the scheduler gets slots and copies from the block manager. The engine copies blocks, then runs the model. In each layer the model writes keys and values to the cache and PagedAttention reads blocks. The logits go back to the engine, which forks and frees through the block manager."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One step, in time order: which file calls which (read from top to bottom)</text><rect class="box-on" x="12.0" y="38.0" width="116.0" height="28.0" rx="6"/><text class="t-tick t-strong" x="70.0" y="57.0" text-anchor="middle">engine.py</text><line class="edge-dim" x1="70" y1="68" x2="70" y2="416"/><rect class="box-1" x="136.0" y="38.0" width="116.0" height="28.0" rx="6"/><text class="t-tick t-strong" x="194.0" y="57.0" text-anchor="middle">scheduler.py</text><line class="edge-dim" x1="194" y1="68" x2="194" y2="416"/><rect class="box-1" x="260.0" y="38.0" width="116.0" height="28.0" rx="6"/><text class="t-tick t-strong" x="318.0" y="57.0" text-anchor="middle">block_manager.py</text><line class="edge-dim" x1="318" y1="68" x2="318" y2="416"/><rect class="box-1" x="384.0" y="38.0" width="116.0" height="28.0" rx="6"/><text class="t-tick t-strong" x="442.0" y="57.0" text-anchor="middle">model.py</text><line class="edge-dim" x1="442" y1="68" x2="442" y2="416"/><rect class="box-1" x="508.0" y="38.0" width="116.0" height="28.0" rx="6"/><text class="t-tick t-strong" x="566.0" y="57.0" text-anchor="middle">attention.py</text><line class="edge-dim" x1="566" y1="68" x2="566" y2="416"/><rect class="box-1" x="632.0" y="38.0" width="116.0" height="28.0" rx="6"/><text class="t-tick t-strong" x="690.0" y="57.0" text-anchor="middle">kv_cache.py</text><line class="edge-dim" x1="690" y1="68" x2="690" y2="416"/><line class="edge-on" x1="74" y1="94" x2="190" y2="94" marker-end="url(#ah-on)"/><text class="t-tick" x="132.0" y="87.0" text-anchor="middle">1. schedule()</text><line class="edge-on" x1="198" y1="124" x2="314" y2="124" marker-end="url(#ah-on)"/><text class="t-tick" x="256.0" y="117.0" text-anchor="middle">2. allocate / append</text><line class="edge" x1="314" y1="154" x2="198" y2="154" marker-end="url(#ah)" stroke-dasharray="5 4"/><text class="t-tick" x="264.0" y="147.0" text-anchor="middle">3. slots, copies</text><line class="edge" x1="190" y1="184" x2="74" y2="184" marker-end="url(#ah)" stroke-dasharray="5 4"/><text class="t-tick" x="140.0" y="177.0" text-anchor="middle">4. SchedulerOutput</text><line class="edge-on" x1="74" y1="214" x2="686" y2="214" marker-end="url(#ah-on)"/><text class="t-tick" x="380.0" y="207.0" text-anchor="middle">5. copy_blocks(copies): copy-on-write first</text><line class="edge-on" x1="74" y1="244" x2="438" y2="244" marker-end="url(#ah-on)"/><text class="t-tick" x="256.0" y="237.0" text-anchor="middle">6. forward(Batch)</text><line class="edge-on" x1="446" y1="274" x2="686" y2="274" marker-end="url(#ah-on)"/><text class="t-tick" x="566.0" y="267.0" text-anchor="middle">7. write(slots, k, v)</text><line class="edge-on" x1="446" y1="304" x2="562" y2="304" marker-end="url(#ah-on)"/><text class="t-tick" x="504.0" y="297.0" text-anchor="middle">8. paged_attention</text><line class="edge-on" x1="570" y1="334" x2="686" y2="334" marker-end="url(#ah-on)"/><text class="t-tick" x="628.0" y="327.0" text-anchor="middle">9. read blocks</text><line class="edge" x1="438" y1="364" x2="74" y2="364" marker-end="url(#ah)" stroke-dasharray="5 4"/><text class="t-tick" x="264.0" y="357.0" text-anchor="middle">10. logits, one row per sequence</text><line class="edge-on" x1="74" y1="394" x2="314" y2="394" marker-end="url(#ah-on)"/><text class="t-tick" x="194.0" y="387.0" text-anchor="middle">11. fork() and free()</text><rect class="box-ghost" x="392" y="260" width="348" height="94" rx="6"/><text class="t-muted" x="740.0" y="370.0" text-anchor="end">steps 7 to 9 repeat in each of the 12 layers</text></svg><figcaption>One step of the engine as a sequence diagram. Time runs down. The engine asks the scheduler for work, the scheduler asks the block manager for slots, and the engine runs the model, which writes to the cache and reads blocks through PagedAttention. The logits go back to the engine, which samples, forks and frees.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 244" role="img" aria-label="Two tokens of request B. "was" at position 2 is in logical block 0, place 2, which the block table maps to physical block 2, so its slot is 10. "the" at position 8 is in logical block 2, place 0, physical block 5, slot 20."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Where does a token live? Request B, block size 4, block table [2, 3, 5]</text><text class="t-label" x="65.0" y="56.0" text-anchor="middle">token</text><text class="t-label" x="170.0" y="56.0" text-anchor="middle">position p</text><text class="t-label" x="290.0" y="56.0" text-anchor="middle">p // 4, p % 4</text><text class="t-label" x="415.0" y="56.0" text-anchor="middle">block table</text><text class="t-label" x="530.0" y="56.0" text-anchor="middle">physical block</text><text class="t-label" x="650.0" y="56.0" text-anchor="middle">slot</text><rect class="box-2" x="20.0" y="70.0" width="90.0" height="40.0" rx="8"/><text class="t-tick" x="65.0" y="95.0" text-anchor="middle">"was"</text><line class="edge" x1="111.0" y1="90.0" x2="118.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-2" x="120.0" y="70.0" width="100.0" height="40.0" rx="8"/><text class="t-tick" x="170.0" y="95.0" text-anchor="middle">2</text><line class="edge" x1="221.0" y1="90.0" x2="228.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-2" x="230.0" y="70.0" width="120.0" height="40.0" rx="8"/><text class="t-tick" x="290.0" y="95.0" text-anchor="middle">logical 0, place 2</text><line class="edge" x1="351.0" y1="90.0" x2="358.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-2" x="360.0" y="70.0" width="110.0" height="40.0" rx="8"/><text class="t-tick" x="415.0" y="95.0" text-anchor="middle">entry 0 = 2</text><line class="edge" x1="471.0" y1="90.0" x2="478.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-2" x="480.0" y="70.0" width="100.0" height="40.0" rx="8"/><text class="t-tick" x="530.0" y="95.0" text-anchor="middle">block 2</text><line class="edge" x1="581.0" y1="90.0" x2="588.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-on" x="590.0" y="70.0" width="120.0" height="40.0" rx="8"/><text class="t-tick" x="650.0" y="95.0" text-anchor="middle">2 x 4 + 2 = 10</text><rect class="box-2" x="20.0" y="136.0" width="90.0" height="40.0" rx="8"/><text class="t-tick" x="65.0" y="161.0" text-anchor="middle">"the"</text><line class="edge" x1="111.0" y1="156.0" x2="118.0" y2="156.0" marker-end="url(#ah)"/><rect class="box-2" x="120.0" y="136.0" width="100.0" height="40.0" rx="8"/><text class="t-tick" x="170.0" y="161.0" text-anchor="middle">8</text><line class="edge" x1="221.0" y1="156.0" x2="228.0" y2="156.0" marker-end="url(#ah)"/><rect class="box-2" x="230.0" y="136.0" width="120.0" height="40.0" rx="8"/><text class="t-tick" x="290.0" y="161.0" text-anchor="middle">logical 2, place 0</text><line class="edge" x1="351.0" y1="156.0" x2="358.0" y2="156.0" marker-end="url(#ah)"/><rect class="box-2" x="360.0" y="136.0" width="110.0" height="40.0" rx="8"/><text class="t-tick" x="415.0" y="161.0" text-anchor="middle">entry 2 = 5</text><line class="edge" x1="471.0" y1="156.0" x2="478.0" y2="156.0" marker-end="url(#ah)"/><rect class="box-2" x="480.0" y="136.0" width="100.0" height="40.0" rx="8"/><text class="t-tick" x="530.0" y="161.0" text-anchor="middle">block 5</text><line class="edge" x1="581.0" y1="156.0" x2="588.0" y2="156.0" marker-end="url(#ah)"/><rect class="box-on" x="590.0" y="136.0" width="120.0" height="40.0" rx="8"/><text class="t-tick" x="650.0" y="161.0" text-anchor="middle">5 x 4 + 0 = 20</text><text class="t-note" x="20.0" y="212.0" text-anchor="start">slot = physical block x block size + place</text><text class="t-tick" x="20.0" y="232.0" text-anchor="start">The slot is the token's address in the cache tensor. Every layer uses the same slot.</text></svg><figcaption>Address translation for two tokens of request B in our run. The position, divided by the block size, gives the logical block and the place in it. The block table turns the logical block into a physical block, and the slot follows. "was" lives at slot 10, "the" at slot 20.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 400" role="img" aria-label="Two requests with their logical blocks on the left and the nine physical blocks on the right. Each logical block points to a physical block; neighbouring logical blocks sit in physical blocks that are not neighbours."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Our run: two requests, block size 4, 9 physical blocks (after decoding step 2)</text><text class="t-tick" x="586.0" y="48.0" text-anchor="middle">physical KV blocks</text><text class="t-tick" x="462.0" y="73.0" text-anchor="end">block 0</text><rect class="box-1" x="470.0" y="56.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="73.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-1" x="528.0" y="56.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="73.5" text-anchor="middle">Four</text><rect class="box-1" x="586.0" y="56.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="73.5" text-anchor="middle">score</text><rect class="box-1" x="644.0" y="56.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="73.5" text-anchor="middle">and</text><text class="t-tick" x="462.0" y="105.0" text-anchor="end">block 1</text><rect class="box-1" x="470.0" y="88.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="105.5" text-anchor="middle">seven</text><rect class="box-1" x="528.0" y="88.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="105.5" text-anchor="middle">years</text><rect class="box-1" x="586.0" y="88.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="105.5" text-anchor="middle">ago</text><rect class="box-1" x="644.0" y="88.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="105.5" text-anchor="middle">,</text><text class="t-tick" x="462.0" y="137.0" text-anchor="end">block 2</text><rect class="box-2" x="470.0" y="120.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="137.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-2" x="528.0" y="120.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="137.5" text-anchor="middle">It</text><rect class="box-2" x="586.0" y="120.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="137.5" text-anchor="middle">was</text><rect class="box-2" x="644.0" y="120.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="137.5" text-anchor="middle">the</text><text class="t-tick" x="462.0" y="169.0" text-anchor="end">block 3</text><rect class="box-2" x="470.0" y="152.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="169.5" text-anchor="middle">best</text><rect class="box-2" x="528.0" y="152.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="169.5" text-anchor="middle">of</text><rect class="box-2" x="586.0" y="152.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="169.5" text-anchor="middle">times</text><rect class="box-2" x="644.0" y="152.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="169.5" text-anchor="middle">,</text><text class="t-tick" x="462.0" y="201.0" text-anchor="end">block 4</text><rect class="box-1" x="470.0" y="184.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="201.5" text-anchor="middle">the</text><rect class="box" x="528.0" y="184.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="201.5" text-anchor="middle"></text><rect class="box" x="586.0" y="184.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="201.5" text-anchor="middle"></text><rect class="box" x="644.0" y="184.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="201.5" text-anchor="middle"></text><text class="t-tick" x="462.0" y="233.0" text-anchor="end">block 5</text><rect class="box-2" x="470.0" y="216.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="233.5" text-anchor="middle">the</text><rect class="box" x="528.0" y="216.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="233.5" text-anchor="middle"></text><rect class="box" x="586.0" y="216.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="233.5" text-anchor="middle"></text><rect class="box" x="644.0" y="216.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="233.5" text-anchor="middle"></text><text class="t-tick" x="462.0" y="265.0" text-anchor="end">block 6</text><rect class="box" x="470.0" y="248.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="265.5" text-anchor="middle"></text><rect class="box" x="528.0" y="248.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="265.5" text-anchor="middle"></text><rect class="box" x="586.0" y="248.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="265.5" text-anchor="middle"></text><rect class="box" x="644.0" y="248.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="265.5" text-anchor="middle"></text><text class="t-tick" x="462.0" y="297.0" text-anchor="end">block 7</text><rect class="box" x="470.0" y="280.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="297.5" text-anchor="middle"></text><rect class="box" x="528.0" y="280.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="297.5" text-anchor="middle"></text><rect class="box" x="586.0" y="280.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="297.5" text-anchor="middle"></text><rect class="box" x="644.0" y="280.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="297.5" text-anchor="middle"></text><text class="t-tick" x="462.0" y="329.0" text-anchor="end">block 8</text><rect class="box" x="470.0" y="312.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="497.0" y="329.5" text-anchor="middle"></text><rect class="box" x="528.0" y="312.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="555.0" y="329.5" text-anchor="middle"></text><rect class="box" x="586.0" y="312.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="613.0" y="329.5" text-anchor="middle"></text><rect class="box" x="644.0" y="312.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="671.0" y="329.5" text-anchor="middle"></text><text class="t-label" x="40.0" y="60.0" text-anchor="start">request A: logical blocks</text><rect class="box-1" x="80.0" y="70.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="107.0" y="87.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-1" x="138.0" y="70.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="165.0" y="87.5" text-anchor="middle">Four</text><rect class="box-1" x="196.0" y="70.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="223.0" y="87.5" text-anchor="middle">score</text><rect class="box-1" x="254.0" y="70.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="281.0" y="87.5" text-anchor="middle">and</text><text class="t-tick" x="70.0" y="87.0" text-anchor="end">0</text><text class="t-mono" x="318.0" y="87.0" text-anchor="start">-&gt; 0</text><line class="edge" x1="352.0" y1="83.0" x2="410.0" y2="69.0" marker-end="url(#ah)"/><rect class="box-1" x="80.0" y="102.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="107.0" y="119.5" text-anchor="middle">seven</text><rect class="box-1" x="138.0" y="102.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="165.0" y="119.5" text-anchor="middle">years</text><rect class="box-1" x="196.0" y="102.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="223.0" y="119.5" text-anchor="middle">ago</text><rect class="box-1" x="254.0" y="102.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="281.0" y="119.5" text-anchor="middle">,</text><text class="t-tick" x="70.0" y="119.0" text-anchor="end">1</text><text class="t-mono" x="318.0" y="119.0" text-anchor="start">-&gt; 1</text><line class="edge" x1="352.0" y1="115.0" x2="410.0" y2="101.0" marker-end="url(#ah)"/><rect class="box-1" x="80.0" y="134.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="107.0" y="151.5" text-anchor="middle">the</text><rect class="box" x="138.0" y="134.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="165.0" y="151.5" text-anchor="middle"></text><rect class="box" x="196.0" y="134.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="223.0" y="151.5" text-anchor="middle"></text><rect class="box" x="254.0" y="134.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="281.0" y="151.5" text-anchor="middle"></text><text class="t-tick" x="70.0" y="151.0" text-anchor="end">2</text><text class="t-mono" x="318.0" y="151.0" text-anchor="start">-&gt; 4</text><line class="edge" x1="352.0" y1="147.0" x2="410.0" y2="197.0" marker-end="url(#ah)"/><text class="t-label" x="40.0" y="240.0" text-anchor="start">request B: logical blocks</text><rect class="box-2" x="80.0" y="250.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="107.0" y="267.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-2" x="138.0" y="250.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="165.0" y="267.5" text-anchor="middle">It</text><rect class="box-2" x="196.0" y="250.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="223.0" y="267.5" text-anchor="middle">was</text><rect class="box-2" x="254.0" y="250.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="281.0" y="267.5" text-anchor="middle">the</text><text class="t-tick" x="70.0" y="267.0" text-anchor="end">0</text><text class="t-mono" x="318.0" y="267.0" text-anchor="start">-&gt; 2</text><line class="edge" x1="352.0" y1="263.0" x2="410.0" y2="133.0" marker-end="url(#ah)"/><rect class="box-2" x="80.0" y="282.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="107.0" y="299.5" text-anchor="middle">best</text><rect class="box-2" x="138.0" y="282.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="165.0" y="299.5" text-anchor="middle">of</text><rect class="box-2" x="196.0" y="282.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="223.0" y="299.5" text-anchor="middle">times</text><rect class="box-2" x="254.0" y="282.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="281.0" y="299.5" text-anchor="middle">,</text><text class="t-tick" x="70.0" y="299.0" text-anchor="end">1</text><text class="t-mono" x="318.0" y="299.0" text-anchor="start">-&gt; 3</text><line class="edge" x1="352.0" y1="295.0" x2="410.0" y2="165.0" marker-end="url(#ah)"/><rect class="box-2" x="80.0" y="314.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="107.0" y="331.5" text-anchor="middle">the</text><rect class="box" x="138.0" y="314.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="165.0" y="331.5" text-anchor="middle"></text><rect class="box" x="196.0" y="314.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="223.0" y="331.5" text-anchor="middle"></text><rect class="box" x="254.0" y="314.0" width="54.0" height="26.0" rx="4"/><text class="t-tick" x="281.0" y="331.5" text-anchor="middle"></text><text class="t-tick" x="70.0" y="331.0" text-anchor="end">2</text><text class="t-mono" x="318.0" y="331.0" text-anchor="start">-&gt; 5</text><line class="edge" x1="352.0" y1="327.0" x2="410.0" y2="229.0" marker-end="url(#ah)"/></svg><figcaption>Our run after decoding step 2. Each request has three logical blocks in order. Their physical blocks interleave: request A owns blocks 0, 1 and 4, request B owns 2, 3 and 5. Blocks 6 to 8 are still free.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 222" role="img" aria-label="The 14 tokens of the prompt step in one list. Request A fills positions 0 to 6 and slots 0 to 6; request B fills positions 0 to 6 and slots 8 to 14. Each request is a prefill range with its own block table."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The prompt step: two prompts joined into one list of 14 tokens</text><text class="t-label" x="82.0" y="54.0" text-anchor="end">token</text><text class="t-label" x="82.0" y="90.0" text-anchor="end">position</text><text class="t-label" x="82.0" y="120.0" text-anchor="end">slot</text><rect class="box-1" x="92.0" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="111.9" y="53.5" text-anchor="middle">&lt;/s&gt;</text><text class="t-tick" x="111.9" y="90.0" text-anchor="middle">0</text><text class="t-mono" x="111.9" y="120.0" text-anchor="middle">0</text><rect class="box-1" x="135.7" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="155.6" y="53.5" text-anchor="middle">Four</text><text class="t-tick" x="155.6" y="90.0" text-anchor="middle">1</text><text class="t-mono" x="155.6" y="120.0" text-anchor="middle">1</text><rect class="box-1" x="179.4" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="199.3" y="53.5" text-anchor="middle">score</text><text class="t-tick" x="199.3" y="90.0" text-anchor="middle">2</text><text class="t-mono" x="199.3" y="120.0" text-anchor="middle">2</text><rect class="box-1" x="223.1" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="243.0" y="53.5" text-anchor="middle">and</text><text class="t-tick" x="243.0" y="90.0" text-anchor="middle">3</text><text class="t-mono" x="243.0" y="120.0" text-anchor="middle">3</text><rect class="box-1" x="266.9" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="286.7" y="53.5" text-anchor="middle">seven</text><text class="t-tick" x="286.7" y="90.0" text-anchor="middle">4</text><text class="t-mono" x="286.7" y="120.0" text-anchor="middle">4</text><rect class="box-1" x="310.6" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="330.4" y="53.5" text-anchor="middle">years</text><text class="t-tick" x="330.4" y="90.0" text-anchor="middle">5</text><text class="t-mono" x="330.4" y="120.0" text-anchor="middle">5</text><rect class="box-1" x="354.3" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="374.1" y="53.5" text-anchor="middle">ago</text><text class="t-tick" x="374.1" y="90.0" text-anchor="middle">6</text><text class="t-mono" x="374.1" y="120.0" text-anchor="middle">6</text><rect class="box-2" x="398.0" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="417.9" y="53.5" text-anchor="middle">&lt;/s&gt;</text><text class="t-tick" x="417.9" y="90.0" text-anchor="middle">0</text><text class="t-mono" x="417.9" y="120.0" text-anchor="middle">8</text><rect class="box-2" x="441.7" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="461.6" y="53.5" text-anchor="middle">It</text><text class="t-tick" x="461.6" y="90.0" text-anchor="middle">1</text><text class="t-mono" x="461.6" y="120.0" text-anchor="middle">9</text><rect class="box-2" x="485.4" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="505.3" y="53.5" text-anchor="middle">was</text><text class="t-tick" x="505.3" y="90.0" text-anchor="middle">2</text><text class="t-mono" x="505.3" y="120.0" text-anchor="middle">10</text><rect class="box-2" x="529.1" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="549.0" y="53.5" text-anchor="middle">the</text><text class="t-tick" x="549.0" y="90.0" text-anchor="middle">3</text><text class="t-mono" x="549.0" y="120.0" text-anchor="middle">11</text><rect class="box-2" x="572.9" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="592.7" y="53.5" text-anchor="middle">best</text><text class="t-tick" x="592.7" y="90.0" text-anchor="middle">4</text><text class="t-mono" x="592.7" y="120.0" text-anchor="middle">12</text><rect class="box-2" x="616.6" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="636.4" y="53.5" text-anchor="middle">of</text><text class="t-tick" x="636.4" y="90.0" text-anchor="middle">5</text><text class="t-mono" x="636.4" y="120.0" text-anchor="middle">13</text><rect class="box-2" x="660.3" y="34.0" width="39.7" height="30.0" rx="4"/><text class="t-tick" x="680.1" y="53.5" text-anchor="middle">times</text><text class="t-tick" x="680.1" y="90.0" text-anchor="middle">6</text><text class="t-mono" x="680.1" y="120.0" text-anchor="middle">14</text><path class="edge" d="M92.0,136 v8 H394.0 v-8"/><text class="t-tick" x="243.0" y="162.0" text-anchor="middle">request A: prefill (0, 7), block table [0, 1]</text><text class="t-muted" x="243.0" y="180.0" text-anchor="middle">its logits come from token 6 only</text><path class="edge" d="M398.0,136 v8 H700.0 v-8"/><text class="t-tick" x="549.0" y="162.0" text-anchor="middle">request B: prefill (7, 14), block table [2, 3]</text><text class="t-muted" x="549.0" y="180.0" text-anchor="middle">its logits come from token 13 only</text><text class="t-note" x="20.0" y="210.0" text-anchor="start">Layer norms, projections and the feed-forward run once over all 14 tokens. Only attention keeps A and B apart.</text></svg><figcaption>The input of our prompt step. The 7 tokens of A and the 7 tokens of B form one list of 14. Each token carries its position in its own request and its slot in the cache. A prefill range and a block table tell attention which tokens belong together.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="Left: after the prompt step, both samples point at the same two physical blocks with a reference count of 2. Right: after the first decoding step, sample A1 got a copy of the shared last block and wrote its token there; sample A2 wrote into the original block, whose count is now 1."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Parallel sampling in our run: two samples, one prompt (block size 4)</text><text class="t-label" x="20.0" y="60.0" text-anchor="start">after the prompt step</text><text class="t-tick" x="20.0" y="96.0" text-anchor="start">sample A1</text><rect class="box-on" x="20.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="41.0" y="119.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-on" x="66.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="87.0" y="119.5" text-anchor="middle">Four</text><rect class="box-on" x="112.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="133.0" y="119.5" text-anchor="middle">score</text><rect class="box-on" x="158.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="179.0" y="119.5" text-anchor="middle">and</text><text class="t-mono" x="210.0" y="119.0" text-anchor="start">block 0, ref 2</text><rect class="box-on" x="20.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="41.0" y="149.5" text-anchor="middle">seven</text><rect class="box-on" x="66.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="87.0" y="149.5" text-anchor="middle">years</text><rect class="box-on" x="112.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="133.0" y="149.5" text-anchor="middle">ago</text><rect class="box" x="158.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="179.0" y="149.5" text-anchor="middle"></text><text class="t-mono" x="210.0" y="149.0" text-anchor="start">block 1, ref 2</text><text class="t-tick" x="20.0" y="192.0" text-anchor="start">sample A2</text><rect class="box-on" x="20.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="41.0" y="215.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-on" x="66.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="87.0" y="215.5" text-anchor="middle">Four</text><rect class="box-on" x="112.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="133.0" y="215.5" text-anchor="middle">score</text><rect class="box-on" x="158.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="179.0" y="215.5" text-anchor="middle">and</text><text class="t-mono" x="210.0" y="215.0" text-anchor="start">block 0, ref 2</text><rect class="box-on" x="20.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="41.0" y="245.5" text-anchor="middle">seven</text><rect class="box-on" x="66.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="87.0" y="245.5" text-anchor="middle">years</text><rect class="box-on" x="112.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="133.0" y="245.5" text-anchor="middle">ago</text><rect class="box" x="158.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="179.0" y="245.5" text-anchor="middle"></text><text class="t-mono" x="210.0" y="245.0" text-anchor="start">block 1, ref 2</text><text class="t-note" x="20.0" y="290.0" text-anchor="start">no copy yet: both samples point at the same 2 blocks</text><text class="t-label" x="380.0" y="60.0" text-anchor="start">after decoding step 1</text><text class="t-tick" x="380.0" y="96.0" text-anchor="start">sample A1</text><rect class="box-on" x="380.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="401.0" y="119.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-on" x="426.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="447.0" y="119.5" text-anchor="middle">Four</text><rect class="box-on" x="472.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="493.0" y="119.5" text-anchor="middle">score</text><rect class="box-on" x="518.0" y="102.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="539.0" y="119.5" text-anchor="middle">and</text><text class="t-mono" x="570.0" y="119.0" text-anchor="start">block 0, ref 2</text><rect class="box-1" x="380.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="401.0" y="149.5" text-anchor="middle">seven</text><rect class="box-1" x="426.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="447.0" y="149.5" text-anchor="middle">years</text><rect class="box-1" x="472.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="493.0" y="149.5" text-anchor="middle">ago</text><rect class="box-1" x="518.0" y="132.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="539.0" y="149.5" text-anchor="middle">:</text><text class="t-mono" x="570.0" y="149.0" text-anchor="start">block 2, ref 1</text><text class="t-tick" x="380.0" y="192.0" text-anchor="start">sample A2</text><rect class="box-on" x="380.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="401.0" y="215.5" text-anchor="middle">&lt;/s&gt;</text><rect class="box-on" x="426.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="447.0" y="215.5" text-anchor="middle">Four</text><rect class="box-on" x="472.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="493.0" y="215.5" text-anchor="middle">score</text><rect class="box-on" x="518.0" y="198.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="539.0" y="215.5" text-anchor="middle">and</text><text class="t-mono" x="570.0" y="215.0" text-anchor="start">block 0, ref 2</text><rect class="box-1" x="380.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="401.0" y="245.5" text-anchor="middle">seven</text><rect class="box-1" x="426.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="447.0" y="245.5" text-anchor="middle">years</text><rect class="box-1" x="472.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="493.0" y="245.5" text-anchor="middle">ago</text><rect class="box-1" x="518.0" y="228.0" width="42.0" height="26.0" rx="4"/><text class="t-tick" x="539.0" y="245.5" text-anchor="middle">right</text><text class="t-mono" x="570.0" y="245.0" text-anchor="start">block 1, ref 1</text><text class="t-note" x="380.0" y="290.0" text-anchor="start">copy-on-write: block 1 copied to block 2</text></svg><figcaption>Our run of Figure 8. Left: after the prompt step both samples point at blocks 0 and 1, each with a reference count of 2. Right: after the first decoding step, A1 has a copy of block 1 in block 2, and A2 still writes into block 1.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 262" role="img" aria-label="A row per request and a column per step. Filled cells are steps in which the request runs; hatched cells are steps in which it waits after being preempted."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Six requests, a cache of 12 blocks: who runs at each step</text><rect class="s1" x="20" y="36" width="12" height="12" rx="2"/><text class="t-tick" x="38.0" y="46.0" text-anchor="start">running</text><rect class="waste" x="110" y="36" width="12" height="12" rx="2"/><text class="t-tick" x="128.0" y="46.0" text-anchor="start">waiting after a preemption</text><text class="t-tick" x="100.0" y="75.0" text-anchor="end">request 0</text><rect class="s1" x="110.0" y="60" width="4.58" height="20"/><rect class="s1" x="114.3" y="60" width="4.58" height="20"/><rect class="s1" x="118.6" y="60" width="4.58" height="20"/><rect class="s1" x="122.8" y="60" width="4.58" height="20"/><rect class="s1" x="127.1" y="60" width="4.58" height="20"/><rect class="s1" x="131.4" y="60" width="4.58" height="20"/><rect class="s1" x="135.7" y="60" width="4.58" height="20"/><rect class="s1" x="139.9" y="60" width="4.58" height="20"/><rect class="s1" x="144.2" y="60" width="4.58" height="20"/><rect class="s1" x="148.5" y="60" width="4.58" height="20"/><rect class="s1" x="152.8" y="60" width="4.58" height="20"/><rect class="s1" x="157.0" y="60" width="4.58" height="20"/><rect class="s1" x="161.3" y="60" width="4.58" height="20"/><rect class="s1" x="165.6" y="60" width="4.58" height="20"/><rect class="s1" x="169.9" y="60" width="4.58" height="20"/><rect class="s1" x="174.1" y="60" width="4.58" height="20"/><rect class="s1" x="178.4" y="60" width="4.58" height="20"/><rect class="s1" x="182.7" y="60" width="4.58" height="20"/><rect class="s1" x="187.0" y="60" width="4.58" height="20"/><rect class="s1" x="191.2" y="60" width="4.58" height="20"/><rect class="s1" x="195.5" y="60" width="4.58" height="20"/><rect class="s1" x="199.8" y="60" width="4.58" height="20"/><rect class="s1" x="204.1" y="60" width="4.58" height="20"/><rect class="s1" x="208.3" y="60" width="4.58" height="20"/><rect class="s1" x="212.6" y="60" width="4.58" height="20"/><rect class="s1" x="216.9" y="60" width="4.58" height="20"/><rect class="s1" x="221.2" y="60" width="4.58" height="20"/><rect class="s1" x="225.4" y="60" width="4.58" height="20"/><rect class="s1" x="229.7" y="60" width="4.58" height="20"/><rect class="s1" x="234.0" y="60" width="4.58" height="20"/><rect class="s1" x="238.3" y="60" width="4.58" height="20"/><rect class="s1" x="242.5" y="60" width="4.58" height="20"/><rect class="s1" x="246.8" y="60" width="4.58" height="20"/><rect class="s1" x="251.1" y="60" width="4.58" height="20"/><rect class="s1" x="255.4" y="60" width="4.58" height="20"/><rect class="s1" x="259.6" y="60" width="4.58" height="20"/><rect class="s1" x="263.9" y="60" width="4.58" height="20"/><rect class="s1" x="268.2" y="60" width="4.58" height="20"/><rect class="s1" x="272.5" y="60" width="4.58" height="20"/><rect class="s1" x="276.7" y="60" width="4.58" height="20"/><rect class="s1" x="281.0" y="60" width="4.58" height="20"/><rect class="s1" x="285.3" y="60" width="4.58" height="20"/><rect class="s1" x="289.6" y="60" width="4.58" height="20"/><rect class="s1" x="293.8" y="60" width="4.58" height="20"/><rect class="s1" x="298.1" y="60" width="4.58" height="20"/><rect class="s1" x="302.4" y="60" width="4.58" height="20"/><rect class="s1" x="306.7" y="60" width="4.58" height="20"/><rect class="s1" x="310.9" y="60" width="4.58" height="20"/><rect class="s1" x="315.2" y="60" width="4.58" height="20"/><rect class="s1" x="319.5" y="60" width="4.58" height="20"/><rect class="s1" x="323.8" y="60" width="4.58" height="20"/><rect class="s1" x="328.0" y="60" width="4.58" height="20"/><rect class="s1" x="332.3" y="60" width="4.58" height="20"/><rect class="s1" x="336.6" y="60" width="4.58" height="20"/><rect class="s1" x="340.9" y="60" width="4.58" height="20"/><rect class="s1" x="345.1" y="60" width="4.58" height="20"/><rect class="s1" x="349.4" y="60" width="4.58" height="20"/><rect class="s1" x="353.7" y="60" width="4.58" height="20"/><rect class="s1" x="358.0" y="60" width="4.58" height="20"/><rect class="s1" x="362.2" y="60" width="4.58" height="20"/><rect class="s1" x="366.5" y="60" width="4.58" height="20"/><rect class="s1" x="370.8" y="60" width="4.58" height="20"/><rect class="s1" x="375.1" y="60" width="4.58" height="20"/><text class="t-tick" x="100.0" y="101.0" text-anchor="end">request 1</text><rect class="s1" x="110.0" y="86" width="4.58" height="20"/><rect class="s1" x="114.3" y="86" width="4.58" height="20"/><rect class="s1" x="118.6" y="86" width="4.58" height="20"/><rect class="s1" x="122.8" y="86" width="4.58" height="20"/><rect class="s1" x="127.1" y="86" width="4.58" height="20"/><rect class="s1" x="131.4" y="86" width="4.58" height="20"/><rect class="s1" x="135.7" y="86" width="4.58" height="20"/><rect class="s1" x="139.9" y="86" width="4.58" height="20"/><rect class="s1" x="144.2" y="86" width="4.58" height="20"/><rect class="s1" x="148.5" y="86" width="4.58" height="20"/><rect class="s1" x="152.8" y="86" width="4.58" height="20"/><rect class="s1" x="157.0" y="86" width="4.58" height="20"/><rect class="s1" x="161.3" y="86" width="4.58" height="20"/><rect class="s1" x="165.6" y="86" width="4.58" height="20"/><rect class="s1" x="169.9" y="86" width="4.58" height="20"/><rect class="s1" x="174.1" y="86" width="4.58" height="20"/><rect class="s1" x="178.4" y="86" width="4.58" height="20"/><rect class="s1" x="182.7" y="86" width="4.58" height="20"/><rect class="s1" x="187.0" y="86" width="4.58" height="20"/><rect class="s1" x="191.2" y="86" width="4.58" height="20"/><rect class="s1" x="195.5" y="86" width="4.58" height="20"/><rect class="s1" x="199.8" y="86" width="4.58" height="20"/><rect class="s1" x="204.1" y="86" width="4.58" height="20"/><rect class="s1" x="208.3" y="86" width="4.58" height="20"/><rect class="s1" x="212.6" y="86" width="4.58" height="20"/><rect class="s1" x="216.9" y="86" width="4.58" height="20"/><rect class="s1" x="221.2" y="86" width="4.58" height="20"/><rect class="s1" x="225.4" y="86" width="4.58" height="20"/><rect class="s1" x="229.7" y="86" width="4.58" height="20"/><rect class="s1" x="234.0" y="86" width="4.58" height="20"/><rect class="s1" x="238.3" y="86" width="4.58" height="20"/><rect class="s1" x="242.5" y="86" width="4.58" height="20"/><rect class="s1" x="246.8" y="86" width="4.58" height="20"/><rect class="s1" x="251.1" y="86" width="4.58" height="20"/><rect class="s1" x="255.4" y="86" width="4.58" height="20"/><rect class="s1" x="259.6" y="86" width="4.58" height="20"/><rect class="s1" x="263.9" y="86" width="4.58" height="20"/><rect class="s1" x="268.2" y="86" width="4.58" height="20"/><rect class="s1" x="272.5" y="86" width="4.58" height="20"/><rect class="s1" x="276.7" y="86" width="4.58" height="20"/><rect class="s1" x="281.0" y="86" width="4.58" height="20"/><rect class="s1" x="285.3" y="86" width="4.58" height="20"/><rect class="s1" x="289.6" y="86" width="4.58" height="20"/><rect class="s1" x="293.8" y="86" width="4.58" height="20"/><rect class="s1" x="298.1" y="86" width="4.58" height="20"/><rect class="s1" x="302.4" y="86" width="4.58" height="20"/><rect class="s1" x="306.7" y="86" width="4.58" height="20"/><rect class="s1" x="310.9" y="86" width="4.58" height="20"/><rect class="s1" x="315.2" y="86" width="4.58" height="20"/><rect class="s1" x="319.5" y="86" width="4.58" height="20"/><rect class="s1" x="323.8" y="86" width="4.58" height="20"/><rect class="s1" x="328.0" y="86" width="4.58" height="20"/><rect class="s1" x="332.3" y="86" width="4.58" height="20"/><rect class="s1" x="336.6" y="86" width="4.58" height="20"/><rect class="s1" x="340.9" y="86" width="4.58" height="20"/><rect class="s1" x="345.1" y="86" width="4.58" height="20"/><rect class="s1" x="349.4" y="86" width="4.58" height="20"/><rect class="s1" x="353.7" y="86" width="4.58" height="20"/><rect class="s1" x="358.0" y="86" width="4.58" height="20"/><rect class="s1" x="362.2" y="86" width="4.58" height="20"/><rect class="s1" x="366.5" y="86" width="4.58" height="20"/><rect class="s1" x="370.8" y="86" width="4.58" height="20"/><rect class="s1" x="375.1" y="86" width="4.58" height="20"/><text class="t-tick" x="100.0" y="127.0" text-anchor="end">request 2</text><rect class="s1" x="110.0" y="112" width="4.58" height="20"/><rect class="s1" x="114.3" y="112" width="4.58" height="20"/><rect class="s1" x="118.6" y="112" width="4.58" height="20"/><rect class="s1" x="122.8" y="112" width="4.58" height="20"/><rect class="s1" x="127.1" y="112" width="4.58" height="20"/><rect class="s1" x="131.4" y="112" width="4.58" height="20"/><rect class="s1" x="135.7" y="112" width="4.58" height="20"/><rect class="s1" x="139.9" y="112" width="4.58" height="20"/><rect class="s1" x="144.2" y="112" width="4.58" height="20"/><rect class="s1" x="148.5" y="112" width="4.58" height="20"/><rect class="s1" x="152.8" y="112" width="4.58" height="20"/><rect class="s1" x="157.0" y="112" width="4.58" height="20"/><rect class="s1" x="161.3" y="112" width="4.58" height="20"/><rect class="s1" x="165.6" y="112" width="4.58" height="20"/><rect class="s1" x="169.9" y="112" width="4.58" height="20"/><rect class="s1" x="174.1" y="112" width="4.58" height="20"/><rect class="s1" x="178.4" y="112" width="4.58" height="20"/><rect class="s1" x="182.7" y="112" width="4.58" height="20"/><rect class="s1" x="187.0" y="112" width="4.58" height="20"/><rect class="s1" x="191.2" y="112" width="4.58" height="20"/><rect class="s1" x="195.5" y="112" width="4.58" height="20"/><rect class="s1" x="199.8" y="112" width="4.58" height="20"/><rect class="s1" x="204.1" y="112" width="4.58" height="20"/><rect class="s1" x="208.3" y="112" width="4.58" height="20"/><rect class="s1" x="212.6" y="112" width="4.58" height="20"/><rect class="s1" x="216.9" y="112" width="4.58" height="20"/><rect class="s1" x="221.2" y="112" width="4.58" height="20"/><rect class="s1" x="225.4" y="112" width="4.58" height="20"/><rect class="s1" x="229.7" y="112" width="4.58" height="20"/><rect class="s1" x="234.0" y="112" width="4.58" height="20"/><rect class="s1" x="238.3" y="112" width="4.58" height="20"/><rect class="s1" x="242.5" y="112" width="4.58" height="20"/><rect class="s1" x="246.8" y="112" width="4.58" height="20"/><rect class="s1" x="251.1" y="112" width="4.58" height="20"/><rect class="s1" x="255.4" y="112" width="4.58" height="20"/><rect class="s1" x="259.6" y="112" width="4.58" height="20"/><rect class="s1" x="263.9" y="112" width="4.58" height="20"/><rect class="s1" x="268.2" y="112" width="4.58" height="20"/><rect class="s1" x="272.5" y="112" width="4.58" height="20"/><rect class="s1" x="276.7" y="112" width="4.58" height="20"/><rect class="s1" x="281.0" y="112" width="4.58" height="20"/><rect class="s1" x="285.3" y="112" width="4.58" height="20"/><rect class="waste" x="289.6" y="112" width="4.58" height="20"/><rect class="waste" x="293.8" y="112" width="4.58" height="20"/><rect class="waste" x="298.1" y="112" width="4.58" height="20"/><rect class="waste" x="302.4" y="112" width="4.58" height="20"/><rect class="waste" x="306.7" y="112" width="4.58" height="20"/><rect class="waste" x="310.9" y="112" width="4.58" height="20"/><rect class="waste" x="315.2" y="112" width="4.58" height="20"/><rect class="waste" x="319.5" y="112" width="4.58" height="20"/><rect class="waste" x="323.8" y="112" width="4.58" height="20"/><rect class="waste" x="328.0" y="112" width="4.58" height="20"/><rect class="waste" x="332.3" y="112" width="4.58" height="20"/><rect class="waste" x="336.6" y="112" width="4.58" height="20"/><rect class="waste" x="340.9" y="112" width="4.58" height="20"/><rect class="waste" x="345.1" y="112" width="4.58" height="20"/><rect class="waste" x="349.4" y="112" width="4.58" height="20"/><rect class="waste" x="353.7" y="112" width="4.58" height="20"/><rect class="waste" x="358.0" y="112" width="4.58" height="20"/><rect class="waste" x="362.2" y="112" width="4.58" height="20"/><rect class="waste" x="366.5" y="112" width="4.58" height="20"/><rect class="waste" x="370.8" y="112" width="4.58" height="20"/><rect class="waste" x="375.1" y="112" width="4.58" height="20"/><rect class="waste" x="379.3" y="112" width="4.58" height="20"/><rect class="s1" x="383.6" y="112" width="4.58" height="20"/><rect class="s1" x="387.9" y="112" width="4.58" height="20"/><rect class="s1" x="392.2" y="112" width="4.58" height="20"/><rect class="s1" x="396.4" y="112" width="4.58" height="20"/><rect class="s1" x="400.7" y="112" width="4.58" height="20"/><rect class="s1" x="405.0" y="112" width="4.58" height="20"/><rect class="s1" x="409.3" y="112" width="4.58" height="20"/><rect class="s1" x="413.6" y="112" width="4.58" height="20"/><rect class="s1" x="417.8" y="112" width="4.58" height="20"/><rect class="s1" x="422.1" y="112" width="4.58" height="20"/><rect class="s1" x="426.4" y="112" width="4.58" height="20"/><rect class="s1" x="430.7" y="112" width="4.58" height="20"/><rect class="s1" x="434.9" y="112" width="4.58" height="20"/><rect class="s1" x="439.2" y="112" width="4.58" height="20"/><rect class="s1" x="443.5" y="112" width="4.58" height="20"/><rect class="s1" x="447.8" y="112" width="4.58" height="20"/><rect class="s1" x="452.0" y="112" width="4.58" height="20"/><rect class="s1" x="456.3" y="112" width="4.58" height="20"/><rect class="s1" x="460.6" y="112" width="4.58" height="20"/><rect class="s1" x="464.9" y="112" width="4.58" height="20"/><rect class="s1" x="469.1" y="112" width="4.58" height="20"/><text class="t-tick" x="100.0" y="153.0" text-anchor="end">request 3</text><rect class="s1" x="110.0" y="138" width="4.58" height="20"/><rect class="s1" x="114.3" y="138" width="4.58" height="20"/><rect class="s1" x="118.6" y="138" width="4.58" height="20"/><rect class="s1" x="122.8" y="138" width="4.58" height="20"/><rect class="s1" x="127.1" y="138" width="4.58" height="20"/><rect class="s1" x="131.4" y="138" width="4.58" height="20"/><rect class="s1" x="135.7" y="138" width="4.58" height="20"/><rect class="s1" x="139.9" y="138" width="4.58" height="20"/><rect class="s1" x="144.2" y="138" width="4.58" height="20"/><rect class="s1" x="148.5" y="138" width="4.58" height="20"/><rect class="s1" x="152.8" y="138" width="4.58" height="20"/><rect class="s1" x="157.0" y="138" width="4.58" height="20"/><rect class="s1" x="161.3" y="138" width="4.58" height="20"/><rect class="s1" x="165.6" y="138" width="4.58" height="20"/><rect class="s1" x="169.9" y="138" width="4.58" height="20"/><rect class="s1" x="174.1" y="138" width="4.58" height="20"/><rect class="s1" x="178.4" y="138" width="4.58" height="20"/><rect class="s1" x="182.7" y="138" width="4.58" height="20"/><rect class="s1" x="187.0" y="138" width="4.58" height="20"/><rect class="s1" x="191.2" y="138" width="4.58" height="20"/><rect class="s1" x="195.5" y="138" width="4.58" height="20"/><rect class="s1" x="199.8" y="138" width="4.58" height="20"/><rect class="s1" x="204.1" y="138" width="4.58" height="20"/><rect class="s1" x="208.3" y="138" width="4.58" height="20"/><rect class="s1" x="212.6" y="138" width="4.58" height="20"/><rect class="s1" x="216.9" y="138" width="4.58" height="20"/><rect class="waste" x="221.2" y="138" width="4.58" height="20"/><rect class="waste" x="225.4" y="138" width="4.58" height="20"/><rect class="waste" x="229.7" y="138" width="4.58" height="20"/><rect class="waste" x="234.0" y="138" width="4.58" height="20"/><rect class="waste" x="238.3" y="138" width="4.58" height="20"/><rect class="waste" x="242.5" y="138" width="4.58" height="20"/><rect class="waste" x="246.8" y="138" width="4.58" height="20"/><rect class="waste" x="251.1" y="138" width="4.58" height="20"/><rect class="waste" x="255.4" y="138" width="4.58" height="20"/><rect class="waste" x="259.6" y="138" width="4.58" height="20"/><rect class="waste" x="263.9" y="138" width="4.58" height="20"/><rect class="waste" x="268.2" y="138" width="4.58" height="20"/><rect class="waste" x="272.5" y="138" width="4.58" height="20"/><rect class="waste" x="276.7" y="138" width="4.58" height="20"/><rect class="waste" x="281.0" y="138" width="4.58" height="20"/><rect class="waste" x="285.3" y="138" width="4.58" height="20"/><rect class="waste" x="289.6" y="138" width="4.58" height="20"/><rect class="waste" x="293.8" y="138" width="4.58" height="20"/><rect class="waste" x="298.1" y="138" width="4.58" height="20"/><rect class="waste" x="302.4" y="138" width="4.58" height="20"/><rect class="waste" x="306.7" y="138" width="4.58" height="20"/><rect class="waste" x="310.9" y="138" width="4.58" height="20"/><rect class="waste" x="315.2" y="138" width="4.58" height="20"/><rect class="waste" x="319.5" y="138" width="4.58" height="20"/><rect class="waste" x="323.8" y="138" width="4.58" height="20"/><rect class="waste" x="328.0" y="138" width="4.58" height="20"/><rect class="waste" x="332.3" y="138" width="4.58" height="20"/><rect class="waste" x="336.6" y="138" width="4.58" height="20"/><rect class="waste" x="340.9" y="138" width="4.58" height="20"/><rect class="waste" x="345.1" y="138" width="4.58" height="20"/><rect class="waste" x="349.4" y="138" width="4.58" height="20"/><rect class="waste" x="353.7" y="138" width="4.58" height="20"/><rect class="waste" x="358.0" y="138" width="4.58" height="20"/><rect class="waste" x="362.2" y="138" width="4.58" height="20"/><rect class="waste" x="366.5" y="138" width="4.58" height="20"/><rect class="waste" x="370.8" y="138" width="4.58" height="20"/><rect class="waste" x="375.1" y="138" width="4.58" height="20"/><rect class="waste" x="379.3" y="138" width="4.58" height="20"/><rect class="s1" x="383.6" y="138" width="4.58" height="20"/><rect class="s1" x="387.9" y="138" width="4.58" height="20"/><rect class="s1" x="392.2" y="138" width="4.58" height="20"/><rect class="s1" x="396.4" y="138" width="4.58" height="20"/><rect class="s1" x="400.7" y="138" width="4.58" height="20"/><rect class="s1" x="405.0" y="138" width="4.58" height="20"/><rect class="s1" x="409.3" y="138" width="4.58" height="20"/><rect class="s1" x="413.6" y="138" width="4.58" height="20"/><rect class="s1" x="417.8" y="138" width="4.58" height="20"/><rect class="s1" x="422.1" y="138" width="4.58" height="20"/><rect class="s1" x="426.4" y="138" width="4.58" height="20"/><rect class="s1" x="430.7" y="138" width="4.58" height="20"/><rect class="s1" x="434.9" y="138" width="4.58" height="20"/><rect class="s1" x="439.2" y="138" width="4.58" height="20"/><rect class="s1" x="443.5" y="138" width="4.58" height="20"/><rect class="s1" x="447.8" y="138" width="4.58" height="20"/><rect class="s1" x="452.0" y="138" width="4.58" height="20"/><rect class="s1" x="456.3" y="138" width="4.58" height="20"/><rect class="s1" x="460.6" y="138" width="4.58" height="20"/><rect class="s1" x="464.9" y="138" width="4.58" height="20"/><rect class="s1" x="469.1" y="138" width="4.58" height="20"/><rect class="s1" x="473.4" y="138" width="4.58" height="20"/><rect class="s1" x="477.7" y="138" width="4.58" height="20"/><rect class="s1" x="482.0" y="138" width="4.58" height="20"/><rect class="s1" x="486.2" y="138" width="4.58" height="20"/><rect class="s1" x="490.5" y="138" width="4.58" height="20"/><rect class="s1" x="494.8" y="138" width="4.58" height="20"/><rect class="s1" x="499.1" y="138" width="4.58" height="20"/><rect class="s1" x="503.3" y="138" width="4.58" height="20"/><rect class="s1" x="507.6" y="138" width="4.58" height="20"/><rect class="s1" x="511.9" y="138" width="4.58" height="20"/><rect class="s1" x="516.2" y="138" width="4.58" height="20"/><rect class="s1" x="520.4" y="138" width="4.58" height="20"/><rect class="s1" x="524.7" y="138" width="4.58" height="20"/><rect class="s1" x="529.0" y="138" width="4.58" height="20"/><rect class="s1" x="533.3" y="138" width="4.58" height="20"/><rect class="s1" x="537.5" y="138" width="4.58" height="20"/><text class="t-tick" x="100.0" y="179.0" text-anchor="end">request 4</text><rect class="s1" x="110.0" y="164" width="4.58" height="20"/><rect class="s1" x="114.3" y="164" width="4.58" height="20"/><rect class="s1" x="118.6" y="164" width="4.58" height="20"/><rect class="s1" x="122.8" y="164" width="4.58" height="20"/><rect class="s1" x="127.1" y="164" width="4.58" height="20"/><rect class="s1" x="131.4" y="164" width="4.58" height="20"/><rect class="s1" x="135.7" y="164" width="4.58" height="20"/><rect class="s1" x="139.9" y="164" width="4.58" height="20"/><rect class="s1" x="144.2" y="164" width="4.58" height="20"/><rect class="s1" x="148.5" y="164" width="4.58" height="20"/><rect class="s1" x="152.8" y="164" width="4.58" height="20"/><rect class="s1" x="157.0" y="164" width="4.58" height="20"/><rect class="s1" x="161.3" y="164" width="4.58" height="20"/><rect class="s1" x="165.6" y="164" width="4.58" height="20"/><rect class="s1" x="169.9" y="164" width="4.58" height="20"/><rect class="s1" x="174.1" y="164" width="4.58" height="20"/><rect class="s1" x="178.4" y="164" width="4.58" height="20"/><rect class="s1" x="182.7" y="164" width="4.58" height="20"/><rect class="s1" x="187.0" y="164" width="4.58" height="20"/><rect class="s1" x="191.2" y="164" width="4.58" height="20"/><rect class="s1" x="195.5" y="164" width="4.58" height="20"/><rect class="s1" x="199.8" y="164" width="4.58" height="20"/><rect class="s1" x="204.1" y="164" width="4.58" height="20"/><rect class="s1" x="208.3" y="164" width="4.58" height="20"/><rect class="waste" x="212.6" y="164" width="4.58" height="20"/><rect class="waste" x="216.9" y="164" width="4.58" height="20"/><rect class="waste" x="221.2" y="164" width="4.58" height="20"/><rect class="waste" x="225.4" y="164" width="4.58" height="20"/><rect class="waste" x="229.7" y="164" width="4.58" height="20"/><rect class="waste" x="234.0" y="164" width="4.58" height="20"/><rect class="waste" x="238.3" y="164" width="4.58" height="20"/><rect class="waste" x="242.5" y="164" width="4.58" height="20"/><rect class="waste" x="246.8" y="164" width="4.58" height="20"/><rect class="waste" x="251.1" y="164" width="4.58" height="20"/><rect class="waste" x="255.4" y="164" width="4.58" height="20"/><rect class="waste" x="259.6" y="164" width="4.58" height="20"/><rect class="waste" x="263.9" y="164" width="4.58" height="20"/><rect class="waste" x="268.2" y="164" width="4.58" height="20"/><rect class="waste" x="272.5" y="164" width="4.58" height="20"/><rect class="waste" x="276.7" y="164" width="4.58" height="20"/><rect class="waste" x="281.0" y="164" width="4.58" height="20"/><rect class="waste" x="285.3" y="164" width="4.58" height="20"/><rect class="waste" x="289.6" y="164" width="4.58" height="20"/><rect class="waste" x="293.8" y="164" width="4.58" height="20"/><rect class="waste" x="298.1" y="164" width="4.58" height="20"/><rect class="waste" x="302.4" y="164" width="4.58" height="20"/><rect class="waste" x="306.7" y="164" width="4.58" height="20"/><rect class="waste" x="310.9" y="164" width="4.58" height="20"/><rect class="waste" x="315.2" y="164" width="4.58" height="20"/><rect class="waste" x="319.5" y="164" width="4.58" height="20"/><rect class="waste" x="323.8" y="164" width="4.58" height="20"/><rect class="waste" x="328.0" y="164" width="4.58" height="20"/><rect class="waste" x="332.3" y="164" width="4.58" height="20"/><rect class="waste" x="336.6" y="164" width="4.58" height="20"/><rect class="waste" x="340.9" y="164" width="4.58" height="20"/><rect class="waste" x="345.1" y="164" width="4.58" height="20"/><rect class="waste" x="349.4" y="164" width="4.58" height="20"/><rect class="waste" x="353.7" y="164" width="4.58" height="20"/><rect class="waste" x="358.0" y="164" width="4.58" height="20"/><rect class="waste" x="362.2" y="164" width="4.58" height="20"/><rect class="waste" x="366.5" y="164" width="4.58" height="20"/><rect class="waste" x="370.8" y="164" width="4.58" height="20"/><rect class="waste" x="375.1" y="164" width="4.58" height="20"/><rect class="waste" x="379.3" y="164" width="4.58" height="20"/><rect class="s1" x="383.6" y="164" width="4.58" height="20"/><rect class="s1" x="387.9" y="164" width="4.58" height="20"/><rect class="s1" x="392.2" y="164" width="4.58" height="20"/><rect class="s1" x="396.4" y="164" width="4.58" height="20"/><rect class="s1" x="400.7" y="164" width="4.58" height="20"/><rect class="s1" x="405.0" y="164" width="4.58" height="20"/><rect class="s1" x="409.3" y="164" width="4.58" height="20"/><rect class="s1" x="413.6" y="164" width="4.58" height="20"/><rect class="s1" x="417.8" y="164" width="4.58" height="20"/><rect class="s1" x="422.1" y="164" width="4.58" height="20"/><rect class="s1" x="426.4" y="164" width="4.58" height="20"/><rect class="s1" x="430.7" y="164" width="4.58" height="20"/><rect class="s1" x="434.9" y="164" width="4.58" height="20"/><rect class="s1" x="439.2" y="164" width="4.58" height="20"/><rect class="s1" x="443.5" y="164" width="4.58" height="20"/><rect class="s1" x="447.8" y="164" width="4.58" height="20"/><rect class="waste" x="452.0" y="164" width="4.58" height="20"/><rect class="waste" x="456.3" y="164" width="4.58" height="20"/><rect class="waste" x="460.6" y="164" width="4.58" height="20"/><rect class="waste" x="464.9" y="164" width="4.58" height="20"/><rect class="waste" x="469.1" y="164" width="4.58" height="20"/><rect class="waste" x="473.4" y="164" width="4.58" height="20"/><rect class="s1" x="477.7" y="164" width="4.58" height="20"/><rect class="s1" x="482.0" y="164" width="4.58" height="20"/><rect class="s1" x="486.2" y="164" width="4.58" height="20"/><rect class="s1" x="490.5" y="164" width="4.58" height="20"/><rect class="s1" x="494.8" y="164" width="4.58" height="20"/><rect class="s1" x="499.1" y="164" width="4.58" height="20"/><rect class="s1" x="503.3" y="164" width="4.58" height="20"/><rect class="s1" x="507.6" y="164" width="4.58" height="20"/><rect class="s1" x="511.9" y="164" width="4.58" height="20"/><rect class="s1" x="516.2" y="164" width="4.58" height="20"/><rect class="s1" x="520.4" y="164" width="4.58" height="20"/><rect class="s1" x="524.7" y="164" width="4.58" height="20"/><rect class="s1" x="529.0" y="164" width="4.58" height="20"/><rect class="s1" x="533.3" y="164" width="4.58" height="20"/><rect class="s1" x="537.5" y="164" width="4.58" height="20"/><rect class="s1" x="541.8" y="164" width="4.58" height="20"/><rect class="s1" x="546.1" y="164" width="4.58" height="20"/><rect class="s1" x="550.4" y="164" width="4.58" height="20"/><rect class="s1" x="554.6" y="164" width="4.58" height="20"/><rect class="s1" x="558.9" y="164" width="4.58" height="20"/><rect class="s1" x="563.2" y="164" width="4.58" height="20"/><rect class="s1" x="567.5" y="164" width="4.58" height="20"/><rect class="s1" x="571.7" y="164" width="4.58" height="20"/><text class="t-tick" x="100.0" y="205.0" text-anchor="end">request 5</text><rect class="s1" x="110.0" y="190" width="4.58" height="20"/><rect class="s1" x="114.3" y="190" width="4.58" height="20"/><rect class="s1" x="118.6" y="190" width="4.58" height="20"/><rect class="s1" x="122.8" y="190" width="4.58" height="20"/><rect class="s1" x="127.1" y="190" width="4.58" height="20"/><rect class="s1" x="131.4" y="190" width="4.58" height="20"/><rect class="s1" x="135.7" y="190" width="4.58" height="20"/><rect class="s1" x="139.9" y="190" width="4.58" height="20"/><rect class="s1" x="144.2" y="190" width="4.58" height="20"/><rect class="s1" x="148.5" y="190" width="4.58" height="20"/><rect class="waste" x="152.8" y="190" width="4.58" height="20"/><rect class="waste" x="157.0" y="190" width="4.58" height="20"/><rect class="waste" x="161.3" y="190" width="4.58" height="20"/><rect class="waste" x="165.6" y="190" width="4.58" height="20"/><rect class="waste" x="169.9" y="190" width="4.58" height="20"/><rect class="waste" x="174.1" y="190" width="4.58" height="20"/><rect class="waste" x="178.4" y="190" width="4.58" height="20"/><rect class="waste" x="182.7" y="190" width="4.58" height="20"/><rect class="waste" x="187.0" y="190" width="4.58" height="20"/><rect class="waste" x="191.2" y="190" width="4.58" height="20"/><rect class="waste" x="195.5" y="190" width="4.58" height="20"/><rect class="waste" x="199.8" y="190" width="4.58" height="20"/><rect class="waste" x="204.1" y="190" width="4.58" height="20"/><rect class="waste" x="208.3" y="190" width="4.58" height="20"/><rect class="waste" x="212.6" y="190" width="4.58" height="20"/><rect class="waste" x="216.9" y="190" width="4.58" height="20"/><rect class="waste" x="221.2" y="190" width="4.58" height="20"/><rect class="waste" x="225.4" y="190" width="4.58" height="20"/><rect class="waste" x="229.7" y="190" width="4.58" height="20"/><rect class="waste" x="234.0" y="190" width="4.58" height="20"/><rect class="waste" x="238.3" y="190" width="4.58" height="20"/><rect class="waste" x="242.5" y="190" width="4.58" height="20"/><rect class="waste" x="246.8" y="190" width="4.58" height="20"/><rect class="waste" x="251.1" y="190" width="4.58" height="20"/><rect class="waste" x="255.4" y="190" width="4.58" height="20"/><rect class="waste" x="259.6" y="190" width="4.58" height="20"/><rect class="waste" x="263.9" y="190" width="4.58" height="20"/><rect class="waste" x="268.2" y="190" width="4.58" height="20"/><rect class="waste" x="272.5" y="190" width="4.58" height="20"/><rect class="waste" x="276.7" y="190" width="4.58" height="20"/><rect class="waste" x="281.0" y="190" width="4.58" height="20"/><rect class="waste" x="285.3" y="190" width="4.58" height="20"/><rect class="waste" x="289.6" y="190" width="4.58" height="20"/><rect class="waste" x="293.8" y="190" width="4.58" height="20"/><rect class="waste" x="298.1" y="190" width="4.58" height="20"/><rect class="waste" x="302.4" y="190" width="4.58" height="20"/><rect class="waste" x="306.7" y="190" width="4.58" height="20"/><rect class="waste" x="310.9" y="190" width="4.58" height="20"/><rect class="waste" x="315.2" y="190" width="4.58" height="20"/><rect class="waste" x="319.5" y="190" width="4.58" height="20"/><rect class="waste" x="323.8" y="190" width="4.58" height="20"/><rect class="waste" x="328.0" y="190" width="4.58" height="20"/><rect class="waste" x="332.3" y="190" width="4.58" height="20"/><rect class="waste" x="336.6" y="190" width="4.58" height="20"/><rect class="waste" x="340.9" y="190" width="4.58" height="20"/><rect class="waste" x="345.1" y="190" width="4.58" height="20"/><rect class="waste" x="349.4" y="190" width="4.58" height="20"/><rect class="waste" x="353.7" y="190" width="4.58" height="20"/><rect class="waste" x="358.0" y="190" width="4.58" height="20"/><rect class="waste" x="362.2" y="190" width="4.58" height="20"/><rect class="waste" x="366.5" y="190" width="4.58" height="20"/><rect class="waste" x="370.8" y="190" width="4.58" height="20"/><rect class="waste" x="375.1" y="190" width="4.58" height="20"/><rect class="waste" x="379.3" y="190" width="4.58" height="20"/><rect class="s1" x="383.6" y="190" width="4.58" height="20"/><rect class="s1" x="387.9" y="190" width="4.58" height="20"/><rect class="s1" x="392.2" y="190" width="4.58" height="20"/><rect class="s1" x="396.4" y="190" width="4.58" height="20"/><rect class="s1" x="400.7" y="190" width="4.58" height="20"/><rect class="s1" x="405.0" y="190" width="4.58" height="20"/><rect class="s1" x="409.3" y="190" width="4.58" height="20"/><rect class="s1" x="413.6" y="190" width="4.58" height="20"/><rect class="s1" x="417.8" y="190" width="4.58" height="20"/><rect class="waste" x="422.1" y="190" width="4.58" height="20"/><rect class="waste" x="426.4" y="190" width="4.58" height="20"/><rect class="waste" x="430.7" y="190" width="4.58" height="20"/><rect class="waste" x="434.9" y="190" width="4.58" height="20"/><rect class="waste" x="439.2" y="190" width="4.58" height="20"/><rect class="waste" x="443.5" y="190" width="4.58" height="20"/><rect class="waste" x="447.8" y="190" width="4.58" height="20"/><rect class="waste" x="452.0" y="190" width="4.58" height="20"/><rect class="waste" x="456.3" y="190" width="4.58" height="20"/><rect class="waste" x="460.6" y="190" width="4.58" height="20"/><rect class="waste" x="464.9" y="190" width="4.58" height="20"/><rect class="waste" x="469.1" y="190" width="4.58" height="20"/><rect class="waste" x="473.4" y="190" width="4.58" height="20"/><rect class="s1" x="477.7" y="190" width="4.58" height="20"/><rect class="s1" x="482.0" y="190" width="4.58" height="20"/><rect class="s1" x="486.2" y="190" width="4.58" height="20"/><rect class="s1" x="490.5" y="190" width="4.58" height="20"/><rect class="s1" x="494.8" y="190" width="4.58" height="20"/><rect class="s1" x="499.1" y="190" width="4.58" height="20"/><rect class="s1" x="503.3" y="190" width="4.58" height="20"/><rect class="s1" x="507.6" y="190" width="4.58" height="20"/><rect class="s1" x="511.9" y="190" width="4.58" height="20"/><rect class="waste" x="516.2" y="190" width="4.58" height="20"/><rect class="waste" x="520.4" y="190" width="4.58" height="20"/><rect class="waste" x="524.7" y="190" width="4.58" height="20"/><rect class="waste" x="529.0" y="190" width="4.58" height="20"/><rect class="waste" x="533.3" y="190" width="4.58" height="20"/><rect class="waste" x="537.5" y="190" width="4.58" height="20"/><rect class="waste" x="541.8" y="190" width="4.58" height="20"/><rect class="s1" x="546.1" y="190" width="4.58" height="20"/><rect class="s1" x="550.4" y="190" width="4.58" height="20"/><rect class="s1" x="554.6" y="190" width="4.58" height="20"/><rect class="s1" x="558.9" y="190" width="4.58" height="20"/><rect class="s1" x="563.2" y="190" width="4.58" height="20"/><rect class="s1" x="567.5" y="190" width="4.58" height="20"/><rect class="s1" x="571.7" y="190" width="4.58" height="20"/><rect class="s1" x="576.0" y="190" width="4.58" height="20"/><rect class="s1" x="580.3" y="190" width="4.58" height="20"/><rect class="s1" x="584.6" y="190" width="4.58" height="20"/><rect class="s1" x="588.8" y="190" width="4.58" height="20"/><rect class="s1" x="593.1" y="190" width="4.58" height="20"/><rect class="s1" x="597.4" y="190" width="4.58" height="20"/><rect class="s1" x="601.7" y="190" width="4.58" height="20"/><rect class="s1" x="605.9" y="190" width="4.58" height="20"/><rect class="s1" x="610.2" y="190" width="4.58" height="20"/><rect class="s1" x="614.5" y="190" width="4.58" height="20"/><rect class="s1" x="618.8" y="190" width="4.58" height="20"/><rect class="s1" x="623.0" y="190" width="4.58" height="20"/><rect class="s1" x="627.3" y="190" width="4.58" height="20"/><rect class="s1" x="631.6" y="190" width="4.58" height="20"/><rect class="s1" x="635.9" y="190" width="4.58" height="20"/><rect class="s1" x="640.1" y="190" width="4.58" height="20"/><rect class="s1" x="644.4" y="190" width="4.58" height="20"/><rect class="s1" x="648.7" y="190" width="4.58" height="20"/><rect class="s1" x="653.0" y="190" width="4.58" height="20"/><rect class="s1" x="657.2" y="190" width="4.58" height="20"/><rect class="s1" x="661.5" y="190" width="4.58" height="20"/><rect class="s1" x="665.8" y="190" width="4.58" height="20"/><rect class="s1" x="670.1" y="190" width="4.58" height="20"/><rect class="s1" x="674.3" y="190" width="4.58" height="20"/><rect class="s1" x="678.6" y="190" width="4.58" height="20"/><rect class="s1" x="682.9" y="190" width="4.58" height="20"/><rect class="s1" x="687.2" y="190" width="4.58" height="20"/><rect class="s1" x="691.4" y="190" width="4.58" height="20"/><line class="axis" x1="110" y1="216" x2="700" y2="216"/><text class="t-tick" x="110.0" y="232.0" text-anchor="middle">0</text><text class="t-tick" x="216.9" y="232.0" text-anchor="middle">25</text><text class="t-tick" x="323.8" y="232.0" text-anchor="middle">50</text><text class="t-tick" x="430.7" y="232.0" text-anchor="middle">75</text><text class="t-tick" x="537.5" y="232.0" text-anchor="middle">100</text><text class="t-tick" x="644.4" y="232.0" text-anchor="middle">125</text><text class="t-tick" x="405.0" y="252.0" text-anchor="middle">step</text></svg><figcaption>Our preemption run. Each row is a request and each column a step. Filled cells: the request runs. Hatched cells: it waits after a preemption, until enough blocks are free to recompute its cache in one pass.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 498" role="img" aria-label="Five stages of one engine step. Schedule: A gets slot 7, B slot 15. Build the batch of 2 tokens. Run the model: write keys and values, read blocks through the block tables. Sample: both write "the". Tidy up: nothing finished."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One engine step with real values: decoding step 1 of requests A and B (block size 4)</text><rect class="box-1" x="20.0" y="46.0" width="680.0" height="88.0" rx="8"/><text class="t-note" x="34.0" y="68.0" text-anchor="start">1. schedule</text><text class="t-mono" x="686.0" y="68.0" text-anchor="end">scheduler.py, block_manager.py</text><text class="t-tick" x="34.0" y="88.0" text-anchor="start">A and B are running; each needs one slot for the token it feeds.</text><text class="t-tick" x="34.0" y="106.0" text-anchor="start">A: last block 1 has 1 free place -&gt; slot 7.  B: block 3 -&gt; slot 15.</text><text class="t-tick" x="34.0" y="124.0" text-anchor="start">No block taken, no copy, no preemption.</text><line class="edge" x1="360.0" y1="135.0" x2="360.0" y2="149.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="150.0" width="680.0" height="70.0" rx="8"/><text class="t-note" x="34.0" y="172.0" text-anchor="start">2. build the batch</text><text class="t-mono" x="686.0" y="172.0" text-anchor="end">engine.py</text><text class="t-tick" x="34.0" y="192.0" text-anchor="start">tokens [",", ","], positions [7, 7], slots [7, 15]</text><text class="t-tick" x="34.0" y="210.0" text-anchor="start">block tables [0, 1] and [2, 3], context lengths [8, 8]</text><line class="edge" x1="360.0" y1="221.0" x2="360.0" y2="235.0" marker-end="url(#ah)"/><rect class="box-on" x="20.0" y="236.0" width="680.0" height="88.0" rx="8"/><text class="t-note" x="34.0" y="258.0" text-anchor="start">3. run the model</text><text class="t-mono" x="686.0" y="258.0" text-anchor="end">model.py, attention.py, kv_cache.py</text><text class="t-tick" x="34.0" y="278.0" text-anchor="start">In each of 12 layers: write the 2 new keys and values at slots 7 and 15,</text><text class="t-tick" x="34.0" y="296.0" text-anchor="start">then A attends over blocks [0, 1] (8 tokens) and B over blocks [2, 3].</text><text class="t-tick" x="34.0" y="314.0" text-anchor="start">Out: logits, 2 rows of 50,272 scores.</text><line class="edge" x1="360.0" y1="325.0" x2="360.0" y2="339.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="340.0" width="680.0" height="52.0" rx="8"/><text class="t-note" x="34.0" y="362.0" text-anchor="start">4. sample</text><text class="t-mono" x="686.0" y="362.0" text-anchor="end">engine.py</text><text class="t-tick" x="34.0" y="382.0" text-anchor="start">Greedy: the largest score of each row. A writes "the", B writes "the".</text><line class="edge" x1="360.0" y1="393.0" x2="360.0" y2="407.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="408.0" width="680.0" height="70.0" rx="8"/><text class="t-note" x="34.0" y="430.0" text-anchor="start">5. tidy up</text><text class="t-mono" x="686.0" y="430.0" text-anchor="end">engine.py, block_manager.py</text><text class="t-tick" x="34.0" y="450.0" text-anchor="start">Nobody finished, so nothing is freed. Both last blocks are now full,</text><text class="t-tick" x="34.0" y="468.0" text-anchor="start">so in the next step append_slot takes a new block for each request.</text></svg><figcaption>Decoding step 1 of our two-request run, stage by stage. The scheduler finds one free place in each last block. The batch has 2 tokens. The model writes their keys and values and reads blocks through the tables. Greedy sampling picks "the" for both. Nothing finishes, so nothing is freed.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 250" role="img" aria-label="After the first step, the contiguous run has 4 requests and 81 slots holding a token, with the rest booked but empty. The paged run has 63 requests and 1,324 slots holding a token, 452 booked but empty and 272 free."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The same cache of 2,048 slots after the first step of the serving run</text><text class="t-note" x="208.0" y="68.0" text-anchor="end">one chunk each</text><text class="t-muted" x="208.0" y="86.0" text-anchor="end">before vLLM (512 slots booked)</text><rect class="s1" x="220.0" y="50" width="19.0" height="28" rx="2"/><rect class="waste" x="239.0" y="50" width="461.0" height="28" rx="2"/><text class="t-tick" x="220.0" y="100.0" text-anchor="start">4 requests running; 81 slots hold a token; 1,967 booked but empty; 0 free</text><text class="t-note" x="208.0" y="158.0" text-anchor="end">paged</text><text class="t-muted" x="208.0" y="176.0" text-anchor="end">with vLLM (blocks of 16)</text><rect class="s1" x="220.0" y="140" width="310.3" height="28" rx="2"/><rect class="waste" x="530.3" y="140" width="105.9" height="28" rx="2"/><rect class="box" x="636.2" y="140" width="63.8" height="28" rx="2"/><text class="t-tick" x="220.0" y="190.0" text-anchor="start">63 requests running; 1,324 slots hold a token; 452 booked but empty; 272 free</text><rect class="s1" x="220" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="238.0" y="236.0" text-anchor="start">holds a token</text><rect class="waste" x="350" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="368.0" y="236.0" text-anchor="start">booked, empty</text><rect class="box" x="490" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="508.0" y="236.0" text-anchor="start">free</text></svg><figcaption>The same 2,048-slot cache after the first step. With one 512-slot chunk per request, 4 requests fill the whole cache, and almost all of it is booked but empty. With blocks of 16, 63 requests run, most booked slots hold a token, and some blocks are still free.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="Sequences in the batch at each step. Paged: mean 14.5, done after 128 steps. Contiguous: mean 3.8, done after 494 steps."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Sequences in the batch at each step (64 Alpaca requests, same 2,048-slot cache)</text><line class="grid" x1="60" y1="254.0" x2="700" y2="254.0"/><text class="t-tick" x="52.0" y="258.0" text-anchor="end">0</text><line class="grid" x1="60" y1="211.2" x2="700" y2="211.2"/><text class="t-tick" x="52.0" y="215.2" text-anchor="end">14</text><line class="grid" x1="60" y1="168.4" x2="700" y2="168.4"/><text class="t-tick" x="52.0" y="172.4" text-anchor="end">28</text><line class="grid" x1="60" y1="125.6" x2="700" y2="125.6"/><text class="t-tick" x="52.0" y="129.6" text-anchor="end">42</text><line class="grid" x1="60" y1="82.8" x2="700" y2="82.8"/><text class="t-tick" x="52.0" y="86.8" text-anchor="end">56</text><line class="grid" x1="60" y1="40.0" x2="700" y2="40.0"/><text class="t-tick" x="52.0" y="44.0" text-anchor="end">70</text><text class="t-tick" x="60.0" y="272.0" text-anchor="middle">0</text><text class="t-tick" x="189.6" y="272.0" text-anchor="middle">100</text><text class="t-tick" x="319.1" y="272.0" text-anchor="middle">200</text><text class="t-tick" x="448.7" y="272.0" text-anchor="middle">300</text><text class="t-tick" x="578.2" y="272.0" text-anchor="middle">400</text><text class="t-tick" x="380.0" y="292.0" text-anchor="middle">step</text><polyline class="l1" points="60.0,61.4 61.3,61.4 61.3,64.5 62.6,64.5 62.6,67.5 63.9,67.5 63.9,70.6 65.2,70.6 65.2,76.7 66.5,76.7 66.5,88.9 67.8,88.9 67.8,101.1 69.1,101.1 69.1,110.3 70.4,110.3 70.4,119.5 71.7,119.5 71.7,128.7 73.0,128.7 73.0,131.7 74.3,131.7 74.3,137.8 75.5,137.8 75.5,143.9 76.8,143.9 76.8,150.1 78.1,150.1 78.1,156.2 79.4,156.2 79.4,156.2 80.7,156.2 80.7,159.2 82.0,159.2 82.0,162.3 83.3,162.3 83.3,162.3 84.6,162.3 84.6,177.6 85.9,177.6 85.9,177.6 87.2,177.6 87.2,183.7 88.5,183.7 88.5,189.8 89.8,189.8 89.8,189.8 91.1,189.8 91.1,189.8 92.4,189.8 92.4,192.9 93.7,192.9 93.7,192.9 95.0,192.9 95.0,192.9 96.3,192.9 96.3,195.9 97.6,195.9 97.6,199.0 98.9,199.0 98.9,199.0 100.2,199.0 100.2,202.0 101.5,202.0 101.5,202.0 102.8,202.0 102.8,205.1 104.0,205.1 104.0,205.1 105.3,205.1 105.3,205.1 106.6,205.1 106.6,205.1 107.9,205.1 107.9,205.1 109.2,205.1 109.2,208.1 110.5,208.1 110.5,211.2 111.8,211.2 111.8,211.2 113.1,211.2 113.1,211.2 114.4,211.2 114.4,211.2 115.7,211.2 115.7,211.2 117.0,211.2 117.0,211.2 118.3,211.2 118.3,211.2 119.6,211.2 119.6,211.2 120.9,211.2 120.9,211.2 122.2,211.2 122.2,211.2 123.5,211.2 123.5,214.3 124.8,214.3 124.8,214.3 126.1,214.3 126.1,214.3 127.4,214.3 127.4,214.3 128.7,214.3 128.7,214.3 130.0,214.3 130.0,214.3 131.3,214.3 131.3,217.3 132.6,217.3 132.6,220.4 133.8,220.4 133.8,220.4 135.1,220.4 135.1,220.4 136.4,220.4 136.4,220.4 137.7,220.4 137.7,220.4 139.0,220.4 139.0,220.4 140.3,220.4 140.3,220.4 141.6,220.4 141.6,220.4 142.9,220.4 142.9,223.4 144.2,223.4 144.2,226.5 145.5,226.5 145.5,226.5 146.8,226.5 146.8,226.5 148.1,226.5 148.1,226.5 149.4,226.5 149.4,226.5 150.7,226.5 150.7,226.5 152.0,226.5 152.0,226.5 153.3,226.5 153.3,229.5 154.6,229.5 154.6,229.5 155.9,229.5 155.9,229.5 157.2,229.5 157.2,229.5 158.5,229.5 158.5,232.6 159.8,232.6 159.8,232.6 161.1,232.6 161.1,232.6 162.3,232.6 162.3,232.6 163.6,232.6 163.6,232.6 164.9,232.6 164.9,232.6 166.2,232.6 166.2,232.6 167.5,232.6 167.5,235.7 168.8,235.7 168.8,235.7 170.1,235.7 170.1,235.7 171.4,235.7 171.4,235.7 172.7,235.7 172.7,238.7 174.0,238.7 174.0,238.7 175.3,238.7 175.3,238.7 176.6,238.7 176.6,238.7 177.9,238.7 177.9,238.7 179.2,238.7 179.2,238.7 180.5,238.7 180.5,238.7 181.8,238.7 181.8,238.7 183.1,238.7 183.1,241.8 184.4,241.8 184.4,241.8 185.7,241.8 185.7,241.8 187.0,241.8 187.0,241.8 188.3,241.8 188.3,241.8 189.6,241.8 189.6,241.8 190.9,241.8 190.9,244.8 192.1,244.8 192.1,244.8 193.4,244.8 193.4,244.8 194.7,244.8 194.7,244.8 196.0,244.8 196.0,244.8 197.3,244.8 197.3,244.8 198.6,244.8 198.6,244.8 199.9,244.8 199.9,244.8 201.2,244.8 201.2,244.8 202.5,244.8 202.5,244.8 203.8,244.8 203.8,244.8 205.1,244.8 205.1,244.8 206.4,244.8 206.4,244.8 207.7,244.8 207.7,244.8 209.0,244.8 209.0,244.8 210.3,244.8 210.3,244.8 211.6,244.8 211.6,244.8 212.9,244.8 212.9,244.8 214.2,244.8 214.2,247.9 215.5,247.9 215.5,247.9 216.8,247.9 216.8,247.9 218.1,247.9 218.1,247.9 219.4,247.9 219.4,247.9 220.6,247.9 220.6,247.9 221.9,247.9 221.9,247.9 223.2,247.9 223.2,247.9 224.5,247.9 224.5,254.0 225.8,254.0"/><polyline class="l2" points="60.0,241.8 61.3,241.8 61.3,241.8 62.6,241.8 62.6,241.8 63.9,241.8 63.9,241.8 65.2,241.8 65.2,241.8 66.5,241.8 66.5,241.8 67.8,241.8 67.8,244.8 69.1,244.8 69.1,241.8 70.4,241.8 70.4,241.8 71.7,241.8 71.7,241.8 73.0,241.8 73.0,241.8 74.3,241.8 74.3,241.8 75.5,241.8 75.5,241.8 76.8,241.8 76.8,241.8 78.1,241.8 78.1,241.8 79.4,241.8 79.4,241.8 80.7,241.8 80.7,241.8 82.0,241.8 82.0,241.8 83.3,241.8 83.3,241.8 84.6,241.8 84.6,241.8 85.9,241.8 85.9,241.8 87.2,241.8 87.2,241.8 88.5,241.8 88.5,241.8 89.8,241.8 89.8,241.8 91.1,241.8 91.1,241.8 92.4,241.8 92.4,241.8 93.7,241.8 93.7,241.8 95.0,241.8 95.0,241.8 96.3,241.8 96.3,244.8 97.6,244.8 97.6,241.8 98.9,241.8 98.9,241.8 100.2,241.8 100.2,244.8 101.5,244.8 101.5,241.8 102.8,241.8 102.8,241.8 104.0,241.8 104.0,241.8 105.3,241.8 105.3,241.8 106.6,241.8 106.6,241.8 107.9,241.8 107.9,241.8 109.2,241.8 109.2,241.8 110.5,241.8 110.5,241.8 111.8,241.8 111.8,241.8 113.1,241.8 113.1,244.8 114.4,244.8 114.4,241.8 115.7,241.8 115.7,241.8 117.0,241.8 117.0,241.8 118.3,241.8 118.3,241.8 119.6,241.8 119.6,241.8 120.9,241.8 120.9,241.8 122.2,241.8 122.2,241.8 123.5,241.8 123.5,241.8 124.8,241.8 124.8,241.8 126.1,241.8 126.1,241.8 127.4,241.8 127.4,241.8 128.7,241.8 128.7,241.8 130.0,241.8 130.0,241.8 131.3,241.8 131.3,244.8 132.6,244.8 132.6,241.8 133.8,241.8 133.8,241.8 135.1,241.8 135.1,241.8 136.4,241.8 136.4,241.8 137.7,241.8 137.7,244.8 139.0,244.8 139.0,244.8 140.3,244.8 140.3,241.8 141.6,241.8 141.6,241.8 142.9,241.8 142.9,241.8 144.2,241.8 144.2,241.8 145.5,241.8 145.5,241.8 146.8,241.8 146.8,241.8 148.1,241.8 148.1,241.8 149.4,241.8 149.4,241.8 150.7,241.8 150.7,241.8 152.0,241.8 152.0,241.8 153.3,241.8 153.3,244.8 154.6,244.8 154.6,241.8 155.9,241.8 155.9,244.8 157.2,244.8 157.2,241.8 158.5,241.8 158.5,241.8 159.8,241.8 159.8,241.8 161.1,241.8 161.1,241.8 162.3,241.8 162.3,244.8 163.6,244.8 163.6,241.8 164.9,241.8 164.9,244.8 166.2,244.8 166.2,241.8 167.5,241.8 167.5,241.8 168.8,241.8 168.8,241.8 170.1,241.8 170.1,241.8 171.4,241.8 171.4,241.8 172.7,241.8 172.7,250.9 174.0,250.9 174.0,241.8 175.3,241.8 175.3,241.8 176.6,241.8 176.6,241.8 177.9,241.8 177.9,241.8 179.2,241.8 179.2,244.8 180.5,244.8 180.5,244.8 181.8,244.8 181.8,241.8 183.1,241.8 183.1,241.8 184.4,241.8 184.4,241.8 185.7,241.8 185.7,241.8 187.0,241.8 187.0,241.8 188.3,241.8 188.3,241.8 189.6,241.8 189.6,241.8 190.9,241.8 190.9,247.9 192.1,247.9 192.1,241.8 193.4,241.8 193.4,241.8 194.7,241.8 194.7,241.8 196.0,241.8 196.0,241.8 197.3,241.8 197.3,241.8 198.6,241.8 198.6,247.9 199.9,247.9 199.9,241.8 201.2,241.8 201.2,241.8 202.5,241.8 202.5,241.8 203.8,241.8 203.8,244.8 205.1,244.8 205.1,241.8 206.4,241.8 206.4,241.8 207.7,241.8 207.7,241.8 209.0,241.8 209.0,241.8 210.3,241.8 210.3,241.8 211.6,241.8 211.6,241.8 212.9,241.8 212.9,241.8 214.2,241.8 214.2,241.8 215.5,241.8 215.5,241.8 216.8,241.8 216.8,247.9 218.1,247.9 218.1,241.8 219.4,241.8 219.4,241.8 220.6,241.8 220.6,241.8 221.9,241.8 221.9,241.8 223.2,241.8 223.2,241.8 224.5,241.8 224.5,241.8 225.8,241.8 225.8,241.8 227.1,241.8 227.1,241.8 228.4,241.8 228.4,241.8 229.7,241.8 229.7,244.8 231.0,244.8 231.0,241.8 232.3,241.8 232.3,241.8 233.6,241.8 233.6,244.8 234.9,244.8 234.9,241.8 236.2,241.8 236.2,244.8 237.5,244.8 237.5,241.8 238.8,241.8 238.8,241.8 240.1,241.8 240.1,244.8 241.4,244.8 241.4,241.8 242.7,241.8 242.7,241.8 244.0,241.8 244.0,241.8 245.3,241.8 245.3,241.8 246.6,241.8 246.6,241.8 247.9,241.8 247.9,241.8 249.1,241.8 249.1,241.8 250.4,241.8 250.4,241.8 251.7,241.8 251.7,244.8 253.0,244.8 253.0,241.8 254.3,241.8 254.3,241.8 255.6,241.8 255.6,244.8 256.9,244.8 256.9,241.8 258.2,241.8 258.2,241.8 259.5,241.8 259.5,241.8 260.8,241.8 260.8,241.8 262.1,241.8 262.1,241.8 263.4,241.8 263.4,241.8 264.7,241.8 264.7,241.8 266.0,241.8 266.0,244.8 267.3,244.8 267.3,244.8 268.6,244.8 268.6,247.9 269.9,247.9 269.9,241.8 271.2,241.8 271.2,241.8 272.5,241.8 272.5,241.8 273.8,241.8 273.8,241.8 275.1,241.8 275.1,244.8 276.4,244.8 276.4,241.8 277.7,241.8 277.7,241.8 278.9,241.8 278.9,241.8 280.2,241.8 280.2,241.8 281.5,241.8 281.5,244.8 282.8,244.8 282.8,241.8 284.1,241.8 284.1,241.8 285.4,241.8 285.4,241.8 286.7,241.8 286.7,241.8 288.0,241.8 288.0,241.8 289.3,241.8 289.3,241.8 290.6,241.8 290.6,241.8 291.9,241.8 291.9,241.8 293.2,241.8 293.2,241.8 294.5,241.8 294.5,247.9 295.8,247.9 295.8,241.8 297.1,241.8 297.1,241.8 298.4,241.8 298.4,244.8 299.7,244.8 299.7,241.8 301.0,241.8 301.0,241.8 302.3,241.8 302.3,241.8 303.6,241.8 303.6,241.8 304.9,241.8 304.9,241.8 306.2,241.8 306.2,241.8 307.4,241.8 307.4,250.9 308.7,250.9 308.7,241.8 310.0,241.8 310.0,241.8 311.3,241.8 311.3,241.8 312.6,241.8 312.6,241.8 313.9,241.8 313.9,241.8 315.2,241.8 315.2,241.8 316.5,241.8 316.5,244.8 317.8,244.8 317.8,241.8 319.1,241.8 319.1,241.8 320.4,241.8 320.4,241.8 321.7,241.8 321.7,241.8 323.0,241.8 323.0,244.8 324.3,244.8 324.3,241.8 325.6,241.8 325.6,241.8 326.9,241.8 326.9,241.8 328.2,241.8 328.2,241.8 329.5,241.8 329.5,241.8 330.8,241.8 330.8,241.8 332.1,241.8 332.1,241.8 333.4,241.8 333.4,244.8 334.7,244.8 334.7,241.8 336.0,241.8 336.0,241.8 337.2,241.8 337.2,241.8 338.5,241.8 338.5,241.8 339.8,241.8 339.8,241.8 341.1,241.8 341.1,241.8 342.4,241.8 342.4,241.8 343.7,241.8 343.7,241.8 345.0,241.8 345.0,244.8 346.3,244.8 346.3,241.8 347.6,241.8 347.6,241.8 348.9,241.8 348.9,241.8 350.2,241.8 350.2,241.8 351.5,241.8 351.5,241.8 352.8,241.8 352.8,241.8 354.1,241.8 354.1,241.8 355.4,241.8 355.4,244.8 356.7,244.8 356.7,244.8 358.0,244.8 358.0,241.8 359.3,241.8 359.3,241.8 360.6,241.8 360.6,241.8 361.9,241.8 361.9,241.8 363.2,241.8 363.2,241.8 364.5,241.8 364.5,241.8 365.7,241.8 365.7,241.8 367.0,241.8 367.0,241.8 368.3,241.8 368.3,241.8 369.6,241.8 369.6,241.8 370.9,241.8 370.9,241.8 372.2,241.8 372.2,241.8 373.5,241.8 373.5,244.8 374.8,244.8 374.8,241.8 376.1,241.8 376.1,241.8 377.4,241.8 377.4,241.8 378.7,241.8 378.7,244.8 380.0,244.8 380.0,241.8 381.3,241.8 381.3,241.8 382.6,241.8 382.6,241.8 383.9,241.8 383.9,241.8 385.2,241.8 385.2,244.8 386.5,244.8 386.5,241.8 387.8,241.8 387.8,241.8 389.1,241.8 389.1,241.8 390.4,241.8 390.4,241.8 391.7,241.8 391.7,241.8 393.0,241.8 393.0,241.8 394.3,241.8 394.3,241.8 395.5,241.8 395.5,241.8 396.8,241.8 396.8,241.8 398.1,241.8 398.1,241.8 399.4,241.8 399.4,241.8 400.7,241.8 400.7,241.8 402.0,241.8 402.0,241.8 403.3,241.8 403.3,241.8 404.6,241.8 404.6,241.8 405.9,241.8 405.9,241.8 407.2,241.8 407.2,241.8 408.5,241.8 408.5,241.8 409.8,241.8 409.8,241.8 411.1,241.8 411.1,241.8 412.4,241.8 412.4,241.8 413.7,241.8 413.7,241.8 415.0,241.8 415.0,241.8 416.3,241.8 416.3,241.8 417.6,241.8 417.6,241.8 418.9,241.8 418.9,241.8 420.2,241.8 420.2,241.8 421.5,241.8 421.5,241.8 422.8,241.8 422.8,241.8 424.0,241.8 424.0,241.8 425.3,241.8 425.3,241.8 426.6,241.8 426.6,241.8 427.9,241.8 427.9,241.8 429.2,241.8 429.2,241.8 430.5,241.8 430.5,241.8 431.8,241.8 431.8,241.8 433.1,241.8 433.1,241.8 434.4,241.8 434.4,241.8 435.7,241.8 435.7,241.8 437.0,241.8 437.0,241.8 438.3,241.8 438.3,241.8 439.6,241.8 439.6,241.8 440.9,241.8 440.9,241.8 442.2,241.8 442.2,241.8 443.5,241.8 443.5,241.8 444.8,241.8 444.8,241.8 446.1,241.8 446.1,241.8 447.4,241.8 447.4,241.8 448.7,241.8 448.7,241.8 450.0,241.8 450.0,241.8 451.3,241.8 451.3,241.8 452.6,241.8 452.6,241.8 453.8,241.8 453.8,241.8 455.1,241.8 455.1,241.8 456.4,241.8 456.4,241.8 457.7,241.8 457.7,241.8 459.0,241.8 459.0,241.8 460.3,241.8 460.3,241.8 461.6,241.8 461.6,241.8 462.9,241.8 462.9,241.8 464.2,241.8 464.2,241.8 465.5,241.8 465.5,241.8 466.8,241.8 466.8,241.8 468.1,241.8 468.1,241.8 469.4,241.8 469.4,241.8 470.7,241.8 470.7,244.8 472.0,244.8 472.0,244.8 473.3,244.8 473.3,241.8 474.6,241.8 474.6,241.8 475.9,241.8 475.9,244.8 477.2,244.8 477.2,241.8 478.5,241.8 478.5,241.8 479.8,241.8 479.8,241.8 481.1,241.8 481.1,244.8 482.3,244.8 482.3,241.8 483.6,241.8 483.6,244.8 484.9,244.8 484.9,241.8 486.2,241.8 486.2,241.8 487.5,241.8 487.5,241.8 488.8,241.8 488.8,241.8 490.1,241.8 490.1,241.8 491.4,241.8 491.4,241.8 492.7,241.8 492.7,241.8 494.0,241.8 494.0,241.8 495.3,241.8 495.3,241.8 496.6,241.8 496.6,241.8 497.9,241.8 497.9,241.8 499.2,241.8 499.2,241.8 500.5,241.8 500.5,241.8 501.8,241.8 501.8,241.8 503.1,241.8 503.1,241.8 504.4,241.8 504.4,241.8 505.7,241.8 505.7,241.8 507.0,241.8 507.0,241.8 508.3,241.8 508.3,241.8 509.6,241.8 509.6,241.8 510.9,241.8 510.9,244.8 512.1,244.8 512.1,241.8 513.4,241.8 513.4,241.8 514.7,241.8 514.7,241.8 516.0,241.8 516.0,241.8 517.3,241.8 517.3,241.8 518.6,241.8 518.6,241.8 519.9,241.8 519.9,241.8 521.2,241.8 521.2,244.8 522.5,244.8 522.5,241.8 523.8,241.8 523.8,241.8 525.1,241.8 525.1,241.8 526.4,241.8 526.4,241.8 527.7,241.8 527.7,241.8 529.0,241.8 529.0,241.8 530.3,241.8 530.3,241.8 531.6,241.8 531.6,241.8 532.9,241.8 532.9,241.8 534.2,241.8 534.2,241.8 535.5,241.8 535.5,241.8 536.8,241.8 536.8,241.8 538.1,241.8 538.1,241.8 539.4,241.8 539.4,241.8 540.6,241.8 540.6,241.8 541.9,241.8 541.9,241.8 543.2,241.8 543.2,241.8 544.5,241.8 544.5,241.8 545.8,241.8 545.8,241.8 547.1,241.8 547.1,241.8 548.4,241.8 548.4,241.8 549.7,241.8 549.7,241.8 551.0,241.8 551.0,241.8 552.3,241.8 552.3,241.8 553.6,241.8 553.6,241.8 554.9,241.8 554.9,241.8 556.2,241.8 556.2,241.8 557.5,241.8 557.5,244.8 558.8,244.8 558.8,241.8 560.1,241.8 560.1,241.8 561.4,241.8 561.4,241.8 562.7,241.8 562.7,241.8 564.0,241.8 564.0,241.8 565.3,241.8 565.3,241.8 566.6,241.8 566.6,244.8 567.9,244.8 567.9,241.8 569.1,241.8 569.1,241.8 570.4,241.8 570.4,241.8 571.7,241.8 571.7,241.8 573.0,241.8 573.0,241.8 574.3,241.8 574.3,241.8 575.6,241.8 575.6,241.8 576.9,241.8 576.9,241.8 578.2,241.8 578.2,241.8 579.5,241.8 579.5,241.8 580.8,241.8 580.8,241.8 582.1,241.8 582.1,241.8 583.4,241.8 583.4,241.8 584.7,241.8 584.7,241.8 586.0,241.8 586.0,241.8 587.3,241.8 587.3,241.8 588.6,241.8 588.6,241.8 589.9,241.8 589.9,244.8 591.2,244.8 591.2,241.8 592.5,241.8 592.5,241.8 593.8,241.8 593.8,241.8 595.1,241.8 595.1,241.8 596.4,241.8 596.4,241.8 597.7,241.8 597.7,241.8 598.9,241.8 598.9,241.8 600.2,241.8 600.2,241.8 601.5,241.8 601.5,241.8 602.8,241.8 602.8,241.8 604.1,241.8 604.1,241.8 605.4,241.8 605.4,241.8 606.7,241.8 606.7,241.8 608.0,241.8 608.0,241.8 609.3,241.8 609.3,241.8 610.6,241.8 610.6,241.8 611.9,241.8 611.9,241.8 613.2,241.8 613.2,241.8 614.5,241.8 614.5,241.8 615.8,241.8 615.8,241.8 617.1,241.8 617.1,241.8 618.4,241.8 618.4,241.8 619.7,241.8 619.7,241.8 621.0,241.8 621.0,241.8 622.3,241.8 622.3,241.8 623.6,241.8 623.6,241.8 624.9,241.8 624.9,241.8 626.2,241.8 626.2,241.8 627.4,241.8 627.4,241.8 628.7,241.8 628.7,241.8 630.0,241.8 630.0,241.8 631.3,241.8 631.3,241.8 632.6,241.8 632.6,241.8 633.9,241.8 633.9,241.8 635.2,241.8 635.2,241.8 636.5,241.8 636.5,241.8 637.8,241.8 637.8,241.8 639.1,241.8 639.1,241.8 640.4,241.8 640.4,241.8 641.7,241.8 641.7,241.8 643.0,241.8 643.0,241.8 644.3,241.8 644.3,241.8 645.6,241.8 645.6,241.8 646.9,241.8 646.9,241.8 648.2,241.8 648.2,241.8 649.5,241.8 649.5,241.8 650.8,241.8 650.8,241.8 652.1,241.8 652.1,241.8 653.4,241.8 653.4,241.8 654.7,241.8 654.7,241.8 656.0,241.8 656.0,241.8 657.2,241.8 657.2,241.8 658.5,241.8 658.5,241.8 659.8,241.8 659.8,241.8 661.1,241.8 661.1,241.8 662.4,241.8 662.4,241.8 663.7,241.8 663.7,241.8 665.0,241.8 665.0,241.8 666.3,241.8 666.3,247.9 667.6,247.9 667.6,247.9 668.9,247.9 668.9,247.9 670.2,247.9 670.2,247.9 671.5,247.9 671.5,247.9 672.8,247.9 672.8,247.9 674.1,247.9 674.1,247.9 675.4,247.9 675.4,247.9 676.7,247.9 676.7,247.9 678.0,247.9 678.0,247.9 679.3,247.9 679.3,247.9 680.6,247.9 680.6,247.9 681.9,247.9 681.9,247.9 683.2,247.9 683.2,247.9 684.5,247.9 684.5,247.9 685.7,247.9 685.7,247.9 687.0,247.9 687.0,250.9 688.3,250.9 688.3,250.9 689.6,250.9 689.6,250.9 690.9,250.9 690.9,250.9 692.2,250.9 692.2,250.9 693.5,250.9 693.5,250.9 694.8,250.9 694.8,250.9 696.1,250.9 696.1,250.9 697.4,250.9 697.4,250.9 698.7,250.9 698.7,254.0 700.0,254.0"/><rect class="s1" x="390" y="50" width="12" height="12" rx="2"/><text class="t-note" x="408.0" y="60.0" text-anchor="start">paged: mean 14.5, done after 128 steps</text><rect class="s2" x="390" y="72" width="12" height="12" rx="2"/><text class="t-note" x="408.0" y="82.0" text-anchor="start">contiguous: mean 3.8, done after 494 steps</text></svg><figcaption>The number of sequences in the batch at each step of our serving run. The contiguous run never holds more than 4, because 4 chunks of 512 fill the cache. The paged run starts 63 of the 64 requests in its first step and finishes in far fewer steps.</figcaption></figure>

<figure class="fig"><svg viewBox="0 0 720 170" role="img" aria-label="Paged: 87.3 percent of booked slots hold a token. Contiguous: 10.2 percent."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Of the KV slots booked at each step, the share that holds a token (mean over steps)</text><text class="t-tick" x="122.0" y="71.0" text-anchor="end">paged</text><g class="mark"><title>paged: 87.3%</title><rect class="s1" x="130.0" y="53.0" width="419.1" height="26.0" rx="3"/></g><text class="t-val" x="555.1" y="71.0" text-anchor="start">87.3%</text><text class="t-tick" x="122.0" y="105.0" text-anchor="end">contiguous</text><g class="mark"><title>contiguous: 10.2%</title><rect class="s1" x="130.0" y="87.0" width="48.8" height="26.0" rx="3"/></g><text class="t-val" x="184.8" y="105.0" text-anchor="start">10.2%</text><text class="t-tick" x="20.0" y="150.0" text-anchor="start">Same 2,048-slot cache in both runs. The contiguous run books 512 slots per request.</text></svg><figcaption>Of the KV slots that each run had booked, the share that held a token, averaged over the steps.</figcaption></figure>

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
