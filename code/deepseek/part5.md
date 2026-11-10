---
title: "Sparse Attention in DeepSeek: NSA, DSA and V3.2"
description: "How DeepSeek stopped each new token from reading the whole conversation: NSA, the token scorer of V3.2, its measured costs and the price cut."
date: 2026-11-10
tags: [deepseek, sparse-attention, long-context, attention, inference]
series: "DeepSeek, From the Ground Up"
series_part: 5
motif: grid
accent: "#5fb3a1"
---

On 29 September 2025, DeepSeek released a model called V3.2-Exp and announced "API prices cut by 50%+". Output tokens went from $1.68 to $0.42 per million. The experts were the same as before, and so was the size: 671 billion parameters.

The only change to the architecture was in one place: how each new token looks back at the tokens before it.

That change is called **DeepSeek Sparse Attention**, or DSA. It came out of a research paper from February 2025, **Native Sparse Attention** (NSA). Both answer one question. When a model has read 128,000 tokens, which of them does the next token really need to look at? DeepSeek shipped its answer together with a 75% cut in the output price.

> [!TIP] The whole idea in one minute
> In normal attention, every new token compares itself with every earlier token. At 128K tokens of context, that is 128K comparisons per token, per layer. DSA adds a very small scorer, the **lightning indexer**, in front of the attention. It gives every earlier token a quick score and keeps only the best 2,048. The big attention then reads only those 2,048.
>
> The scorer still looks at everything, but it is about 17 times cheaper per token pair and runs in 8-bit numbers. DeepSeek's measured cost per token at 128K fell by about 3.6 times for reading the prompt and about 8.6 times for writing the answer. The memory to store the conversation did not shrink. It grew by 20%.

This part builds on two earlier pages. [Attention, Part 3](attention-3-sliding-window-sparse.md) covers sliding windows and a first look at DSA. [Part 2](deepseek-2-the-v3-architecture.md) of this series covers the MLA cache. Short reminders appear where they are needed.

## Why attention gets expensive at long context

### Two phases, two different costs

A model that answers a question works in two phases. First it reads the whole prompt in one pass. Then it writes the answer, one token at a time. The two phases have very different costs. The first is **prefill**, the second is **decode** ([LLM Inference, Part 1](llm-inference-1-prefill-and-decode.md) explains both).



In attention, each token makes a **query**. It compares the query with the **key** of every earlier token. The comparison gives a score. A softmax turns the scores into weights that add up to 1. The weights then mix the **values** of the earlier tokens into one output. So the work grows with the number of (query, key) pairs.

### Prefill: the pairs grow with the square

Count the pairs in a prompt of 4 tokens. Token 1 looks at 1 token (itself), token 2 at 2, token 3 at 3, token 4 at 4. That is 1 + 2 + 3 + 4 = 10 pairs. For a prompt of $$L$$ tokens the count is:

$$
P(L) = 1 + 2 + \dots + L = \frac{L\,(L+1)}{2}
$$

Here $$P(L)$$ is the number of (query, key) pairs and $$L$$ is the prompt length. For $$L = 4$$ it gives $$4 \times 5 / 2 = 10$$, as counted. For $$L = 131{,}072$$ (128K) it gives about 8.6 billion pairs, in every layer. Double the prompt and the pairs grow four times. This is the **quadratic** cost of prefill.

### Decode: every new token reads the whole cache

Decode looks cheaper at first. One new token at position $$L$$ makes one query and compares it with $$L$$ keys. That is linear in $$L$$, not quadratic. But it happens for every token of the answer, and each comparison needs the earlier tokens' records from memory.

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §3.1 · page 5
> [![The paragraph on arithmetic intensity, with auto-regressive decoding becomes memory-bandwidth constrained and requiring loading the entire key-value cache highlighted](/img/deepseek/papers/p5-nsa-decode-memory.png)](/img/deepseek/papers/p5-nsa-decode-memory.png)
>
> **What it says:** in training and prefill, attention has high arithmetic intensity and is compute-bound. In contrast, "auto-regressive decoding becomes memory-bandwidth constrained" because it makes one token per pass "while requiring loading the entire key-value cache".
>
> **What it means:**
>
> Think of a cook who reads a recipe book. In prefill, the cook reads one page and makes a hundred dishes from it, so the cooking is the slow part. In decode, the cook reads the whole book to make one dish. Now the reading is the slow part, and a faster stove does not help.
>
> So the two phases need different fixes. Prefill needs fewer multiplications. Decode needs fewer bytes read from memory. A good sparse attention must cut both. The NSA paper makes this point because many earlier methods cut only one of them.

> [!DEFINITION] Arithmetic intensity, compute-bound, memory-bound
> **Arithmetic intensity** is the number of arithmetic operations a program does per byte it reads from memory. When it is high, the chip's arithmetic units are the limit: the work is **compute-bound**. When it is low, the memory bus is the limit: the work is **memory-bound**.

### The real numbers for DeepSeek-V3

Now put in V3's real sizes. V3 uses **Multi-head Latent Attention** (MLA, see [Part 2](deepseek-2-the-v3-architecture.md)). Instead of full keys and values per head, MLA stores one small **latent** vector per token per layer. It has 512 numbers, plus a 64-number key part that carries position (the RoPE key). All 128 query heads read that same 576-number entry.

MLA can run in two forms, and the DeepSeek-V3.2 paper draws them side by side.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · Appendix A, Figure 7 · page 20
> [![Figure 7 of the DeepSeek-V3.2 paper: the MHA mode of MLA on the left with per-head keys and values rebuilt from the latent, the MQA mode on the right with one shared latent and key, and the caption sentence MHA mode is used for training and prefilling, while the MQA mode is used for decoding highlighted](/img/deepseek/papers/p5-v32-mla-modes.png)](/img/deepseek/papers/p5-v32-mla-modes.png)
>
> **What it says:** MLA has an MHA mode and an MQA mode. For DeepSeek-V3.1-Terminus, "the MHA mode is used for training and prefilling, while the MQA mode is used for decoding."
>
> **What it means:**
>
> In **MHA mode** (left), the model rebuilds a full key and value for each of the 128 heads from the cached latent. A key has 192 numbers and a value 128. That costs work up front, but each comparison is then small. In **MQA mode** (right), it does the reverse. It folds the rebuilding matrices into the query and the output, so every head compares directly against the one 576-number latent. No per-head keys are ever made.
>
> Prefill handles many queries at once, so rebuilding keys once and reusing them is cheap: MHA mode wins. Decode has one query per step, so rebuilding a full set of keys for one query is waste: MQA mode wins. Keep this split in mind. DSA chooses MQA mode on purpose, and that choice explains a kink in DeepSeek's cost curves later.

> [!DEFINITION] MQA (multi-query attention)
> An attention layout where many query heads share one key and one value per token. The cache is small, and one memory read serves all heads. [Attention, Part 2](attention-2-mqa-gqa-mla.md) covers MQA, GQA and MLA.

Per (query, key) pair and per layer, the two forms cost these multiply-adds:

$$
\text{MQA mode: } 128 \times (576 + 512) = 139{,}264
\qquad
\text{MHA mode: } 128 \times (192 + 128) = 40{,}960
$$

Read the first one left to right. There are 128 heads. Each head takes a 576-number dot product for the score, then adds a 512-number latent, scaled by its weight, into its output. The second one is the same with 192-number keys and 128-number values.

For one new token at position $$L$$, the decode cost of attention over all 61 layers is then:

$$
\text{FLOPs per new token} = 2 \times 61 \times 139{,}264 \times L
$$

The 2 turns multiply-adds into FLOPs. At $$L = 131{,}072$$ this is about 2.2 trillion FLOPs. Compare it with the rest of the model. V3 activates about 37 billion parameters per token, and each costs about 2 FLOPs, so the experts and projections cost about 74 billion FLOPs. Our calculator (`part5_cost.py`) prints both:

```text
DECODE: one new token at position L, whole model (61 layers), attention core only
       L  dense GFLOP   DSA idx  DSA main   DSA tot dense/DSA  weights attn/weights
   8,192        139.2       8.2      34.8      43.0      3.2x       74         1.9x
  16,384        278.4      16.4      34.8      51.2      5.4x       74         3.8x
  32,768        556.7      32.7      34.8      67.5      8.2x       74         7.5x
  65,536      1,113.5      65.5      34.8     100.3     11.1x       74        15.0x
 131,072      2,226.9     131.0      34.8     165.8     13.4x       74        30.1x
Dense attention FLOPs equal the 2 x 37B weight FLOPs at L = 4,355 tokens
```

Look at the last column. At 8K tokens of context, dense attention already costs 1.9 times the whole rest of the model. At 128K it costs 30 times more. The mixture of experts made V3 cheap per token, but at long context attention eats that saving many times over. Above about 4,400 tokens, attention is the bigger bill.

{{FIG:decode_flops|Attention FLOPs for one new token of DeepSeek-V3, by context length (our calculator, attention core only). Dense MLA grows in a straight line with the context; DSA grows much more slowly; the expert and projection FLOPs stay at about 74 billion.}}

### The bytes matter even more in decode

FLOPs are only half of the decode story. Each new token must also read the cache of every earlier token from memory. With the FP8 layout of DeepSeek's FlashMLA kernels, one token's entry is 656 bytes per layer. With plain BF16 it is 1,152 bytes.

```text
DECODE: cache bytes read for one new token of one sequence (61 layers)
       L  dense FP8 MB  dense BF16 MB  DSA idx MB  DSA main MB  DSA tot MB dense/DSA
   8,192         327.8          575.7        66.0         82.0       147.9      2.2x
  16,384         655.6        1,151.3       131.9         82.0       213.9      3.1x
  32,768       1,311.2        2,302.7       263.8         82.0       345.8      3.8x
  65,536       2,622.5        4,605.3       527.7         82.0       609.6      4.3x
 131,072       5,245.0        9,210.7     1,055.4         82.0     1,137.3      4.6x
At 128K on one H200 (4.8 TB/s): reading the dense FP8 cache takes 1.09 ms per new token, the DSA reads 0.24 ms (one sequence, before any batching)
```

At 128K, one new token of one conversation reads 5.2 GB of cache. On an H200, whose memory delivers 4.8 TB per second, that is about 1.1 ms of pure reading per token. A server decodes many conversations in one step, and each one brings its own cache. The weights are read once per step and shared by the whole batch, but the caches are not shared. So at long context the cache reads, not the weights, set the speed.

{{FIG:decode_bytes|Cache bytes read for one new token of one sequence, by context length (our calculator). The dense lines grow in proportion to the context. DSA still grows, because the indexer reads a small key for every token, but about 4.6 times more slowly at 128K.}}

## The way out: most of the weight lands on a few tokens

Attention weights are not spread evenly. In a long document, a token puts most of its weight on a few places. These are often the last few words, a name mentioned earlier, or the question at the top. The other tokens get tiny weights. If the model knew in advance which tokens matter, it could skip the rest.

> [!DEFINITION] Sparse attention
> Any attention where each query computes scores against only a **subset** of the earlier tokens, not all of them. The rest are treated as if their weight were zero.

The hard part is the "in advance". To know which tokens get large weights, you normally have to compute all the scores, and that is the cost you wanted to avoid. Each method below is an answer to this puzzle. The picture shows the patterns on 16 tokens.

{{FIG:patterns|Which (query, key) pairs get computed in four kinds of attention on 16 tokens. Dense computes every pair. A sliding window keeps the last few tokens. NSA's selection branch picks whole blocks. DSA picks single tokens anywhere.}}

A sliding window (second panel, see [Attention, Part 3](attention-3-sliding-window-sparse.md)) is cheap, but it forgets anything older than the window. DeepSeek wanted to reach a fact 100,000 tokens back, so it needed selection that depends on the content.

There is a second trap. A model trained with full attention learns to rely on all its tokens. Cut most of them away after training, and quality drops.

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §2.2 · page 4
> [![The paragraph on performance degradation, with top 20% attention can only cover 70% of the total attention scores highlighted](/img/deepseek/papers/p5-nsa-myth.png)](/img/deepseek/papers/p5-nsa-myth.png)
>
> **What it says:** applying sparsity after training "forces models to deviate from their pretrained optimization trajectory". Citing earlier work, the authors note that the "top 20% attention can only cover 70% of the total attention scores".
>
> **What it means:**
>
> Imagine a student who learned to answer exam questions with the whole textbook open. On exam day you give them only the 20% of pages they used most. They lose 30% of what they used to read, and some of it was needed. The model is the same. It was never trained to live with a cut-down view.
>
> The fix is in the paper's title. Train the model with the sparse attention from the start, or at least for a long time. Then the model learns to put its important information where the selection will find it. DeepSeek followed this rule twice: NSA trained a model from scratch, and DSA trained V3.2 for almost a trillion tokens with the selection switched on.

## Native Sparse Attention (February 2025)

NSA is a research paper by DeepSeek-AI and Peking University, posted on 16 February 2025 and published at ACL 2025. No released DeepSeek model used NSA exactly as published. But its ideas are the direct parents of DSA, and of the compressed attention in [V4](deepseek-6-v4-and-v4-1.md).

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · Figure 2 · page 3
> [![Figure 2 of the NSA paper: keys and values split into continuous blocks feed three branches, compression, selection with top-n block selection, and a sliding window, whose outputs are combined into a gated output; on the right, the attention masks of the three branches; the phrase three parallel attention branches is highlighted](/img/deepseek/papers/p5-nsa-figure2.png)](/img/deepseek/papers/p5-nsa-figure2.png)
>
> **What it says:** NSA processes the input "through three parallel attention branches". Earlier keys and values become compressed attention "for coarse-grained patterns", selected attention "for important token blocks", and sliding attention "for local context".
>
> **What it means:**
>
> Think of how you search a long book for one fact. First you skim the chapter titles (compression). Then you open the two or three chapters that look right and read them closely (selection). All the while you remember the last page you read (sliding window). NSA gives each query these three views and lets it mix them.

{{FIG:nsa|NSA in one picture. The past is cut into blocks. Compression turns each block into one summary key. Selection reads a few whole blocks in full. A sliding window keeps the most recent tokens. Learned gates mix the three outputs.}}

### The gated sum

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §3.2 · page 6
> [![Equation 5 of the NSA paper, the output as a sum over strategies of a gate times attention over remapped keys and values, with compression, selection, and sliding window and derived from input features via an MLP and sigmoid activation highlighted](/img/deepseek/papers/p5-nsa-framework.png)](/img/deepseek/papers/p5-nsa-framework.png)
>
> **What it says:** the output is $$o^*_t = \sum_{c \in C} g^c_t \cdot \mathrm{Attn}(q_t, \tilde K^c_t, \tilde V^c_t)$$ with $$C$$ = {cmp, slc, win}, "compression, selection, and sliding window". Each gate $$g^c_t \in [0, 1]$$ is "derived from input features via an MLP and sigmoid activation".
>
> **What it means:**
>
> Each branch runs its own ordinary attention over its own small set of keys and values. Then a tiny network looks at the current token and decides how much to trust each branch. A token that continues a sentence may turn up the sliding window. A token that answers a question about page 3 may turn up selection.
>
> The gates are numbers between 0 and 1, not a choice of one branch. So the model can blend the three views, and gradients flow through all of them during training. That last point is what "natively trainable" means.

### Branch 1: compression

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §3.3.1 · page 6
> [![Equation 7 of the NSA paper, compressed keys as a learnable map phi applied to blocks of keys, with phi is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key highlighted](/img/deepseek/papers/p5-nsa-compression.png)](/img/deepseek/papers/p5-nsa-compression.png)
>
> **What it says:** compressed keys are $$\varphi(k_{id+1:id+l})$$ for each block, where $$l$$ is the block length, $$d$$ the stride between blocks, and "φ is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key".
>
> **What it means:**
>
> Take 32 consecutive keys, feed them to a small network, and get one key that stands for the whole block. It works like a chapter summary. The query attends to these summaries, so it sees the whole past at low resolution. With blocks of 32 starting every 16 tokens, 64K tokens become about 4,096 summary keys.

### Branch 2: selection, in whole blocks

The compression branch does more than give a coarse view. Its attention scores already say which parts of the past matter. NSA reuses them as importance scores for **selection blocks** of 64 tokens. It then reads the top 16 blocks in full. So the scorer is free: it is a by-product of the compression branch.

Why whole blocks and not single tokens? The answer is the hardware.

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §3.3.2 · page 7
> [![The blockwise selection paragraph, with significantly higher throughput for continuous block accesses compared to random index-based reads highlighted](/img/deepseek/papers/p5-nsa-blockwise.png)](/img/deepseek/papers/p5-nsa-blockwise.png)
>
> **What it says:** "Blockwise selection is crucial to achieve efficient computation on modern GPUs", because GPUs show "significantly higher throughput for continuous block accesses compared to random index-based reads". Blocks also suit the GPU's Tensor Cores.
>
> **What it means:**
>
> GPU memory works like a warehouse where a forklift carries a full pallet in one trip. Ask for 64 boxes that sit next to each other and you get them in one trip. Ask for 64 boxes scattered over the warehouse and you need many trips for the same amount of goods. Reading contiguous blocks keeps the memory bus full.
>
> This is what "hardware-aligned" in the title means. The method is designed around how the chip moves data, not only around how many multiplications it saves. DSA, a few months later, would drop the blocks and pick single tokens. It could do that only because its latent entries are large and shared by all heads, as the next sections show.

A second hardware rule decides **who** chooses the blocks. When heads share one cache, as in GQA or MQA, NSA adds up the importance scores of all heads in a group. So the whole group picks the same blocks. If 16 heads each picked their own 16 blocks, the group could load 256 blocks, and the saving would be gone. DSA takes this rule to the end: in V3.2, all 128 heads of a layer share one selection.

### Branch 3: the sliding window

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §3.3.3 · page 8
> [![The sliding window paragraph, with without being shortcutted by local patterns highlighted](/img/deepseek/papers/p5-nsa-sliding.png)](/img/deepseek/papers/p5-nsa-sliding.png)
>
> **What it says:** local patterns "adapt faster and can dominate the learning process". A dedicated window branch lets the other two branches learn "without being shortcutted by local patterns". Each branch also gets its own keys and values.
>
> **What it means:**
>
> The nearest words are the easiest signal in language. If the selection branch could see them, it would learn to pick them and nothing else, and never find the fact on page 3. A separate window branch takes that easy win away. It is a training trick, not a cost trick, and it shows why "trained natively" matters.

### The numbers

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §4.1 · page 10
> [![The NSA settings: compression block size 32, sliding stride 16, selected block size 64, selected block count 16 and sliding window size 512, with three phrases highlighted](/img/deepseek/papers/p5-nsa-hparams.png)](/img/deepseek/papers/p5-nsa-hparams.png)
>
> **What it says:** "compression block size $$l$$ = 32, sliding stride $$d$$ = 16, selected block size $$l'$$ = 64, selected block count $$n$$ = 16 (including fixed activating the 1 initial block and 2 local blocks), and sliding window size $$w$$ = 512."
>
> **What it means:**
>
> These five numbers fix how many keys one query reads. The 16 selected blocks always include the very first block and the two most recent ones.

From the settings, the number of keys one query reads at position $$t$$ is:

$$
N_t = \frac{t}{d} + n\,l' + w
$$

Read it left to right. There are $$t/d$$ compressed keys, one per stride of 16 tokens. There are $$n\,l'$$ selected keys, 16 blocks of 64. And there are $$w$$ window keys, 512 of them. At $$t = 65{,}536$$:

$$
N_t = \frac{65{,}536}{16} + 16 \times 64 + 512 = 4{,}096 + 1{,}024 + 512 = 5{,}632
$$

Full attention reads 65,536 keys at that position, so NSA reads 11.6 times fewer. The exact count of compressed blocks is $$\lfloor (t-l)/d \rfloor + 1 = 4{,}095$$, one fewer. The paper rounds to $$t/d$$, and our calculator prints both. Table 4 of the paper gives the same numbers.

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · Table 4 · page 14
> [![Table 4 of the NSA paper: full attention reads 8192, 16384, 32768 and 65536 tokens; NSA reads 2048, 2560, 3584 and 5632; expected speedup 4x, 6.4x, 9.1x, 11.6x; the caption phrase expected speedup is approximately linear with the volume of memory access is highlighted](/img/deepseek/papers/p5-nsa-table4.png)](/img/deepseek/papers/p5-nsa-table4.png)
>
> **What it says:** memory access per decoding step, in tokens: 2,048 to 5,632 for NSA against 8,192 to 65,536 for full attention. Because decoding is memory-bound, "the expected speedup is approximately linear with the volume of memory access".
>
> **What it means:**
>
> Decode is limited by reading, so reading 11.6 times less should make the attention step about 11.6 times faster. Note the word "expected". The 11.6 in the abstract and in Figure 1 comes from this count, not from a timed decoding kernel. It is a sound estimate, but it is an estimate.
>
> The ratio also grows with the context. At 8K the fixed part (1,024 selected plus 512 window keys) is a large share of the reads. At 64K it is small, and the compressed keys, one per 16 tokens, dominate. NSA's decode cost still grows in proportion to $$t$$, only 16 times more slowly.

{{FIG:nsa_reads|Keys read per decoding step by full attention and by NSA, at 8K to 64K tokens (our calculator with the paper's settings; the same numbers as the paper's Table 4).}}

The training kernels were timed for real.

> [!PAPER] Yuan et al. (2025), Native Sparse Attention · §5.1 · page 13
> [![The training speed paragraph, with up to 9.0x forward and 6.0x backward speedup at 64k context-length highlighted](/img/deepseek/papers/p5-nsa-speed.png)](/img/deepseek/papers/p5-nsa-speed.png)
>
> **What it says:** against Triton-based FlashAttention-2, NSA "achieves progressively greater speedups as context length increases, up to 9.0× forward and 6.0× backward speedup at 64k context-length."
>
> **What it means:**
>
> This is measured, on an 8-GPU A100 machine, with both kernels written in the same language (Triton). The forward pass is 2.1, 3.8, 6.3 and 9.0 times faster at 8K, 16K, 32K and 64K. The backward pass gains less: 1.1, 2.0, 3.4 and 6.0 times. The paper does not explain the gap.
>
> At 8K the backward pass is barely faster. Sparse attention pays off only at lengths where the skipped work outweighs the fixed overheads. DSA meets the same rule.

And quality? The test model was a 27-billion-parameter mixture of experts with 3 billion active parameters, trained from scratch. NSA matched or beat the same model with full attention. It scored higher on 7 of 9 general benchmarks and got perfect retrieval in a 64K needle-in-a-haystack test. On the LongBench long-context suite it averaged 0.469 against 0.437 for full attention.

These are the authors' own results on their own model. They show that native training works at that scale, not that NSA beats full attention in general.

## From NSA to DSA: what DeepSeek kept and what it dropped

Seven months later, DSA appeared in DeepSeek-V3.2-Exp. It is not NSA with new numbers. It keeps one idea, drops two, and adds one.

| | NSA (Feb 2025) | DSA (Sep 2025) |
|---|---|---|
| What is selected | whole blocks of 64 tokens | single tokens |
| Who scores | the compression branch, for free | a separate lightning indexer |
| Branches | compression + selection + window, gated | one: attention over the top-k |
| Shared selection | per group of heads | all 128 heads of a layer |
| How trained | from scratch | continued training of an existing model |
| Tested on | 27B research model | the 671B production model |

The biggest change is the last row. DeepSeek did not want to train a new 671B model from scratch to try sparse attention. It wanted to add sparsity to V3.1-Terminus, a model already trained with dense MLA. That decided the design. The new parts had to fit around MLA without changing it, and they had to learn in continued training.

## DeepSeek Sparse Attention in V3.2-Exp

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.1 · page 3
> [![The opening of section 2.1, with the only architectural modification and is the introduction of DeepSeek Sparse Attention (DSA) through continued training highlighted](/img/deepseek/papers/p5-v32-same-arch.png)](/img/deepseek/papers/p5-v32-same-arch.png)
>
> **What it says:** V3.2 uses exactly the same architecture as V3.2-Exp. Compared with V3.1-Terminus, "the only architectural modification" of V3.2 "is the introduction of DeepSeek Sparse Attention (DSA) through continued training."
>
> **What it means:**
>
> This sentence makes V3.2 a clean experiment. The experts, the router, the multi-token prediction module, the tokenizer and the 128K context are all unchanged. So any change in cost comes from attention. The configuration files confirm it. The V3.2 config is V3's plus three fields: `index_n_heads: 64`, `index_head_dim: 128` and `index_topk: 2048`.

{{FIG:block|Where DSA sits in one of DeepSeek-V3.2's 61 layers. The lightning indexer and top-k selector attach to the MLA attention and tell it which cached entries to read. Everything else is V3.1-Terminus.}}

The new weights are small. By our count from the V3.2-Exp reference code, the indexer adds about 14 million weights per layer, about 0.85 billion over 61 layers. That is 0.13% on top of 671 billion. DSA has two parts: the lightning indexer, and the top-k selection that uses its scores.

### Part 1: the lightning indexer

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.1 · page 3
> [![Equation 1 of the DeepSeek-V3.2 paper, the index score I as a sum over indexer heads of a weight times ReLU of a query-key dot product, with We choose ReLU as the activation function for throughput consideration and can be implemented in FP8 highlighted](/img/deepseek/papers/p5-v32-indexer.png)](/img/deepseek/papers/p5-v32-indexer.png)
>
> **What it says:** the index score between query token $$h_t$$ and earlier token $$h_s$$ is $$I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}(q^I_{t,j} \cdot k^I_s)$$. "We choose ReLU as the activation function for throughput consideration." The indexer "has a small number of heads and can be implemented in FP8".
>
> **What it means:**
>
> The indexer is a second, tiny attention that runs next to the real one. It does not mix any values. It only produces a score for each earlier token, to decide who gets in. Its job is ranking, so its scores never need to be exact. They only need to put the right tokens near the top.
>
> Every design choice serves speed. ReLU is a cheap function, unlike the exponentials of a softmax. FP8 doubles the arithmetic speed. And the key $$k^I_s$$ has no head index: each earlier token stores **one** small indexer key, shared by all indexer heads.

> [!DEFINITION] Lightning indexer
> DSA's small scoring network. For each query token it computes one score per earlier token, with a few small FP8 heads and one cached key per earlier token. In V3.2: 64 heads of 128 numbers.

Read the formula left to right:

$$
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \mathrm{ReLU}\big(q^I_{t,j} \cdot k^I_s\big)
$$

- $$I_{t,s}$$ is the score that query token $$t$$ gives earlier token $$s$$.
- $$H^I$$ is the number of indexer heads (64 in V3.2), and $$j$$ counts them.
- $$q^I_{t,j}$$ is head $$j$$'s query, a vector of $$d^I = 128$$ numbers made from token $$t$$.
- $$k^I_s$$ is the indexer key of token $$s$$, also 128 numbers, and the same for every head.
- $$w^I_{t,j}$$ is one number per head, also made from token $$t$$: how much this query trusts head $$j$$.

**A tiny example by hand.** Take 2 heads and vectors of 2 numbers. The query token makes $$q_1 = (1, 0)$$, $$q_2 = (0, 1)$$ and weights $$w = (2, 1)$$. There are three earlier tokens with keys $$k_A = (3, -1)$$, $$k_B = (-2, 2)$$ and $$k_C = (1, 1)$$.

| Token | $$q_1 \cdot k$$ | ReLU | $$q_2 \cdot k$$ | ReLU | $$I = 2 \times \text{ReLU}_1 + 1 \times \text{ReLU}_2$$ |
|---|---|---|---|---|---|
| A | 3 | 3 | -1 | 0 | 6 |
| B | -2 | 0 | 2 | 2 | 2 |
| C | 1 | 1 | 1 | 1 | 3 |

With $$k = 2$$, the selection keeps A and C. Head 1 liked A strongly, and head 2's dislike of A was cut to 0 by ReLU, not counted against it. Each head can vote for tokens, but it cannot veto them. The per-query weights then decide whose votes count more.

**The real thing.** The V3.2-Exp reference code on Hugging Face shows how each piece is built. This is the core of its `Indexer.forward`, unchanged:

```python
q = self.wq_b(qr)                                   # qr: MLA's 1,536-number query latent
q = q.view(bsz, seqlen, self.n_heads, self.head_dim)  # 64 heads x 128
...
k = self.wk(x)                                      # x: the 7,168-number hidden state -> ONE 128-number key
k = self.k_norm(k)
...
q = rotate_activation(q)                            # a Hadamard rotation before 8-bit rounding
k = rotate_activation(k)
q_fp8, q_scale = act_quant(q, block_size, self.scale_fmt)
k_fp8, k_scale = act_quant(k, block_size, self.scale_fmt)
self.k_cache[:bsz, start_pos:end_pos] = k_fp8       # the indexer keeps its own FP8 key cache
...
weights = self.weights_proj(x.float()) * self.n_heads ** -0.5
...
topk_indices = index_score.topk(min(self.index_topk, end_pos), dim=-1)[1]
```

Four details are worth seeing. First, the indexer query is not made from scratch. It reuses MLA's compressed query, so it costs one extra matrix, 1,536 by 8,192 numbers. Second, the key is one 128-number vector per token, with a normalisation, and RoPE on its first 64 numbers (left out above).

Third, both are rotated by a Hadamard transform and then rounded to FP8. Our reading: the rotation spreads large outlier values over all 128 numbers, so 8-bit rounding loses less. Fourth, the indexer keeps its **own** key cache in FP8, next to MLA's cache. Remember that cache; it comes back when we count bytes.

> [!DEFINITION] Hadamard transform
> A fixed rotation of a vector made only of +1 and -1 entries (scaled). It keeps lengths and dot products the same, but mixes every number into every other.

### Part 2: top-k selection, shared by all heads

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.1 · page 3
> [![The fine-grained token selection paragraph, with retrieves only the key-value entries and corresponding to the top-k index scores highlighted](/img/deepseek/papers/p5-v32-topk.png)](/img/deepseek/papers/p5-v32-topk.png)
>
> **What it says:** the "fine-grained token selection mechanism retrieves only the key-value entries $$\{c_s\}$$ corresponding to the top-k index scores." The attention output is then computed between the query token and these selected entries only.
>
> **What it means:**
>
> "Fine-grained" is the contrast with NSA. NSA picked blocks of 64 neighbours. DSA picks single tokens, anywhere in the past. If the 2,048 most useful tokens are scattered across 100,000, DSA can take exactly those 2,048 and nothing else.
>
> The entries $$c_s$$ are MLA's cached latents, the 576-number records from Part 2. DSA does not change what is stored for the main attention. It only changes which stored entries get read.

With $$k = 2{,}048$$ and the selection in place, the attention of query token $$t$$ becomes:

$$
u_t = \mathrm{Attn}\big(h_t,\ \{\, c_s \mid I_{t,s} \in \text{Top-}k(I_{t,:}) \,\}\big)
$$

Here $$u_t$$ is the attention output for token $$t$$ and $$h_t$$ is its hidden state. The set holds the cached entries whose index score is among the $$k$$ largest. For a query at position $$t \le 2{,}048$$ the set holds every earlier token, so short contexts get dense attention.

Why can DSA pick single scattered tokens when NSA insisted on blocks? Because of the MQA mode.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.1 · page 3
> [![The paragraph Instantiate DSA Under MLA, with each key-value entry must be shared across multiple queries for computational efficiency and where each latent vector (the key-value entry of MLA) will be shared across highlighted](/img/deepseek/papers/p5-v32-mqa-mode.png)](/img/deepseek/papers/p5-v32-mqa-mode.png)
>
> **What it says:** "At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency", citing the NSA paper. So DSA is built on "the MQA mode of MLA", "where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token."
>
> **What it means:**
>
> This is NSA's shared-heads rule, pushed to the end. In MQA mode, one selected entry is one 656-byte row that all 128 heads use. Fetching it once feeds 128 dot products. That is enough work per fetch to keep the GPU busy, even though the rows are scattered.
>
> In MHA mode, each head would need its own rebuilt key and value for each selected token. Selection would save reads but not much else. So DSA's kernels run the core attention in MQA mode, in long prefill as well as in decode. That is a change from V3.1-Terminus, which used MHA mode in prefill.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · Figure 2 · page 4
> [![Figure 2 of the DeepSeek-V3.2 paper: MLA's query path, latent and RoPE key feed a multi-query core attention; in green, the indexer queries, key and weights feed the lightning indexer, whose scores go to a top-k selector in front of the core attention; the caption phrase green part illustrates how DSA selects the top-k key-value entries according to the indexer is highlighted](/img/deepseek/papers/p5-v32-figure2.png)](/img/deepseek/papers/p5-v32-figure2.png)
>
> **What it says:** the attention architecture of V3.2, where DSA is placed under MLA. "The green part illustrates how DSA selects the top-k key-value entries according to the indexer."
>
> **What it means:**
>
> Follow the green lines. The indexer query comes from $$c^Q_t$$, MLA's query latent. The indexer key and the weights come from the input hidden state $$h_t$$. The indexer's scores go to the top-k selector. The selector stands between the cached entries $$[c^{KV}_t; k^R_t]$$ and the core attention, like a gatekeeper.
>
> Everything in black is MLA in MQA mode, exactly as in V3.1-Terminus. That is why DeepSeek could start from V3.1-Terminus's weights. Nothing the old model learned had to be thrown away.

{{FIG:indexer|One query token through DSA in one layer of DeepSeek-V3.2. The indexer path is cheap but touches every earlier token. The core attention is expensive but touches only the 2,048 selected latents.}}

### How much does this save?

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.3 · page 5
> [![Section 2.3, Inference Costs: DSA reduces the core attention complexity from O(L squared) to O(Lk); the phrases still has a complexity, estimated from benchmarking the actual service deployed on H800 GPUs at a rental price of 2 USD per GPU, masked MHA mode to and simulate DSA are highlighted](/img/deepseek/papers/p5-v32-costs.png)](/img/deepseek/papers/p5-v32-costs.png)
>
> **What it says:** "DSA reduces the core attention complexity of the main model from $$O(L^2)$$ to $$O(Lk)$$". The lightning indexer "still has a complexity of $$O(L^2)$$", but needs much less computation than MLA. For short prefills, DeepSeek implements "a masked MHA mode to simulate DSA".
>
> **What it means:**
>
> The expensive part, the 128-head attention, now does a fixed 2,048 comparisons per query, so its total grows only in proportion to $$L$$. The cheap part, the indexer, still compares every pair, so it still grows with $$L^2$$. DSA does not remove the quadratic term. It moves it to a component that is about 17 times cheaper per pair.
>
> The last sentence hides a real engineering decision, explained just below. For short prompts, DeepSeek does not run DSA as written. It runs the old dense MHA mode with a mask that gives the same result, because that is faster.

Per (query, key) pair and layer, the indexer does $$64 \times 128 = 8{,}192$$ multiply-adds, in FP8. MQA-mode MLA does 139,264, in BF16. That is 17 times fewer operations, and FP8 arithmetic runs about twice as fast. So the quadratic part is roughly 30 to 35 times cheaper per pair. This is our estimate from the configuration, not a number from the paper.

Now redo the prefill count for a whole prompt, with the calculator:

```text
PREFILL: a whole prompt of L tokens, whole model, attention core only (TFLOP)
       L  dense MHA  dense MQA   DSA idx  DSA main   DSA tot  MHA/DSA  weights
   8,192      167.7      570.2      33.5     249.4     283.0     0.6x    606.2
  16,384      670.7    2,280.5     134.1     534.5     668.6     1.0x  1,212.4
  32,768    2,682.9    9,121.8     536.6   1,104.6   1,641.2     1.6x  2,424.8
  65,536   10,731.4   36,486.8   2,146.3   2,244.8   4,391.1     2.4x  4,849.7
 131,072   42,925.3  145,945.9   8,585.1   4,525.2  13,110.2     3.3x  9,699.3

Per-query break-even, dense MHA-mode vs DSA: t = 8,704 tokens (7,737 if the FP8 indexer counts half)
Below that, a dense pass is cheaper: why V3.2 runs short prefills in a masked MHA mode.
```

Look at the 8K row. DSA costs **more** than dense prefill: 283 against 168 trillion FLOPs. The reason is the MHA/MQA split from earlier. Dense prefill uses the cheap MHA mode at 40,960 multiply-adds per pair. DSA's selected attention uses MQA mode at 139,264 per pair, 3.4 times more. At short lengths, the saving from skipping tokens is smaller than that penalty.

The break-even per query follows from one equation. Dense MHA mode costs $$40{,}960\,t$$ for a query at position $$t$$. DSA costs $$8{,}192\,t + 139{,}264 \times 2{,}048$$. They are equal at $$t = 8{,}704$$, or 7,737 if FP8 counts half.

Below that, dense is cheaper. That is exactly the masked MHA mode in the paper. For short prompts, run dense and mask out the unselected tokens. It gives the same answer for less work.

At 128K, DSA's prefill is 3.3 times cheaper than dense MHA mode, by our count.

{{FIG:prefill_flops|Attention compute for a whole prompt of DeepSeek-V3 (our calculator). Dense MHA-mode prefill grows with the square of the length. DSA starts more expensive and wins after a few thousand tokens. At 128K it is 3.3 times cheaper.}}

> [!WARNING] The reference code is not the fast path
> In the public V3.2-Exp `model.py`, the attention still computes **all** scores and then adds minus infinity to the unselected ones (`index_mask`). That gives the right answer for checking, but no speed. The speed comes from separate kernels. FlashMLA's sparse attention kernels reach up to 640 TFLOPS in prefill and 410 TFLOPS in decode on H800, by DeepSeek's report. DeepGEMM adds indexer scoring kernels, and there are TileLang versions too. In September 2026 DeepSeek also released DeepSelect, a library of top-k kernels for DSA.

## Teaching the indexer: two stages of continued training

A new indexer starts with random weights. It must learn which tokens the model's own attention cares about. DeepSeek teaches it by imitation, in two stages, starting from the V3.1-Terminus checkpoint with its 128K context.

{{FIG:stages|The path from V3.1-Terminus to V3.2. A short dense warm-up trains only the indexer. A long sparse stage trains everything with top-2,048 selection switched on. Post-training then runs with DSA as well. The bar shows the tokens of the two stages to scale.}}

### Stage 1: dense warm-up

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.1.1 · page 4
> [![The dense warm-up stage: freeze all model parameters except for the lightning indexer, sum the main attention over heads, L1-normalise it, train the indexer with a KL loss (Equation 3), 1000 steps of 16 sequences of 128K tokens; freeze all model parameters except for the lightning and resulting in a total of 2.1B tokens are highlighted](/img/deepseek/papers/p5-v32-warmup.png)](/img/deepseek/papers/p5-v32-warmup.png)
>
> **What it says:** keep dense attention and "freeze all model parameters except for the lightning indexer". Sum the main attention scores over all heads and L1-normalise them into a target $$p_{t,:}$$. Train the indexer with the loss $$\mathcal{L}^I = \sum_t D_{\mathrm{KL}}(p_{t,:} \,\|\, \mathrm{Softmax}(I_{t,:}))$$. It runs for 1,000 steps of 16 sequences of 128K tokens, "resulting in a total of 2.1B tokens".
>
> **What it means:**
>
> The model is the teacher and the indexer is the student. For each query, the teacher shows where its 128 heads put their attention, added up into one distribution over earlier tokens. The student must predict that distribution from its own cheap scores. Nothing else in the model moves, so the teacher stays the same while the student learns.
>
> The check of the arithmetic: 1,000 steps times 16 sequences times 131,072 tokens is 2.1 billion tokens. That is tiny for a model of this size: V3's original pre-training used 14.8 trillion tokens. It works because the student is small and the target is clear.

**A tiny example.** Say the target over three tokens is $$p = (0.7, 0.2, 0.1)$$ and the indexer's softmax gives $$q = (0.5, 0.3, 0.2)$$. Then:

$$
D_{\mathrm{KL}}(p \,\|\, q) = 0.7 \ln\tfrac{0.7}{0.5} + 0.2 \ln\tfrac{0.2}{0.3} + 0.1 \ln\tfrac{0.1}{0.2} = 0.236 - 0.081 - 0.069 = 0.085
$$

The first term dominates: the token the teacher cares about most is under-scored. The gradient pushes the indexer to raise that token's score. Note the direction of the KL. With $$p$$ first, missing a token the teacher needs costs a lot, while giving a little score to an unneeded token costs little. For a selector that keeps 2,048 tokens, that is the right bias.

### Stage 2: sparse training

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.1.1 · page 4
> [![The sparse training stage, Equation 4 with the KL taken only over the selected set S_t; we detach the indexer input from the computational graph, select 2048 key-value tokens for each query token, and resulting in a total of 943.7B tokens are highlighted](/img/deepseek/papers/p5-v32-sparse-train.png)](/img/deepseek/papers/p5-v32-sparse-train.png)
>
> **What it says:** switch on token selection and train all parameters. The indexer loss now uses only the selected set $$S_t$$. "We detach the indexer input from the computational graph": the indexer learns only from its KL loss, and the main model only from the language-modelling loss. The run uses a learning rate of $$7.3 \times 10^{-6}$$, "2048 key-value tokens for each query token", and 15,000 steps of 480 sequences of 128K tokens, "a total of 943.7B tokens".
>
> **What it means:**
>
> Now the main model must learn to live with seeing only 2,048 tokens per query. This is NSA's lesson about native training, applied as a long continued run instead of from scratch. 943.7 billion tokens is a serious amount: about 6% of V3's original 14.8 trillion pre-training tokens.
>
> The "detach" matters. Without it, the main model could bend its hidden states to make the indexer's job easier. Or the indexer's loss could pull on the main model's weights. Keeping the two signals apart makes each one do one job. The very low learning rate tells the same story: $$7.3 \times 10^{-6}$$ is the rate V3 used at the very end of its pre-training. The model should adjust gently, not relearn.

Check the arithmetic again: 15,000 steps times 480 sequences is 7.2 million sequences, and times 131,072 tokens that is 943.7 billion tokens. The warm-up is 0.2% of the total, as the bar in the figure shows. The data in both stages is "totally aligned" with V3.1-Terminus's 128K long-context extension data, so the only new thing the model sees is the sparsity.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §2.2 · page 5
> [![Section 2.2, Parity Evaluation, with we do not observe substantial performance degradation highlighted](/img/deepseek/papers/p5-v32-parity.png)](/img/deepseek/papers/p5-v32-parity.png)
>
> **What it says:** V3.2-Exp was compared with V3.1-Terminus on a suite of benchmarks in September 2025, and "we do not observe substantial performance degradation compared with DeepSeek-V3.1-Terminus, on both short- and long-context tasks."
>
> **What it means:**
>
> This is the bar DeepSeek set: same quality, lower cost. It is a claim of no large loss, not of better quality, and not of zero loss on every benchmark. The same section adds outside checks. On the AA-LCR long-context benchmark, V3.2-Exp scored four points higher in reasoning mode.

## Our experiment: a lightning indexer from scratch

[Attention, Part 3](attention-3-sliding-window-sparse.md) trained an indexer on a real 0.5B model. Here we do something smaller and more controlled. We build the whole DSA path from scratch on random tensors, with V3.2's shape in miniature, and measure everything. The script is `part5_indexer.py` and runs on a laptop CPU in under a minute.

The setting has 4,096 cached tokens and 16 query heads that share one 32-number latent per token (MQA mode). The indexer has 16 heads of 16 numbers. Each query keeps $$k = 64$$ tokens. That is 1.56% of the cache, the same share as 2,048 of 131,072 in V3.2. The "main model" is two fixed random matrices. The hidden states are random, plus one shared mean direction, because real hidden states are not centred on zero.

The indexer is the formula, line for line:

```python
class Indexer(torch.nn.Module):
    """I[t,s] = sum_j w[t,j] * ReLU(q_I[t,j] . k_I[s])."""
    def __init__(self):
        super().__init__()
        self.wq = torch.nn.Linear(D, HI * DI, bias=False)          # query token -> HI small queries
        self.wk = torch.nn.Linear(D, DI, bias=False)               # earlier token -> ONE small key
        self.ww = torch.nn.Linear(D, HI, bias=False)               # query token -> one weight per head

    def forward(self, h_q, h_kv, fp8=False):
        B = h_q.shape[0]
        q = self.wq(h_q).view(B, T, HI, DI)                        # (B,T,HI,DI)
        k = self.wk(h_kv)                                          # (B,L,DI)
        if fp8:                                                    # store and multiply in 8-bit floats
            q = q.to(torch.float8_e4m3fn).float()
            k = k.to(torch.float8_e4m3fn).float()
        w = self.ww(h_q)                                           # (B,T,HI)
        dots = torch.einsum('btjd,bsd->btjs', q, k)                # (B,T,HI,L)
        return torch.einsum('btj,btjs->bts', w, F.relu(dots))      # (B,T,L): the index scores
```

The sparse attention gathers the 64 selected latents and attends over them only:

```python
def sparse_attention(q, c, idx):
    """Method 1: gather the k selected latents and attend over them only. idx: (B,T,K)."""
    B = q.shape[0]
    sel = torch.gather(c.unsqueeze(1).expand(B, T, L, DC), 2, idx.unsqueeze(-1).expand(B, T, K, DC))  # (B,T,K,DC)
    s = TEMP * torch.einsum('bhtc,btkc->bhtk', q, sel) / math.sqrt(DC)
    a = torch.softmax(s, -1)
    return torch.einsum('bhtk,btkc->bthc', a, sel)                 # (B,T,H,DC)
```

Then the script runs the dense warm-up of Stage 1. The main model is frozen, and only the indexer trains, with $$D_{\mathrm{KL}}(p \,\|\, \mathrm{Softmax}(I))$$, for 400 steps on fresh random sequences. It tests on a fixed batch that it never trained on. Here is the full output:

```text
Setup: 4,096 cached tokens, 32 queries per sequence, 16 heads sharing one 32-dim latent, indexer 16 heads x 16 dims, keep k = 64 (1.56%)
Dense attention: the best 64 tokens hold 90.1% of the head-summed attention on average

warm-up step    1: KL loss 11.653 per query
warm-up step   25: KL loss 4.727 per query
warm-up step  100: KL loss 2.308 per query
warm-up step  200: KL loss 1.855 per query
warm-up step  400: KL loss 1.134 per query
warm-up took 37 s on CPU

selector                attention kept  recall of true top-k  output error
random tokens                     1.5%                  1.6%         71.9%
untrained indexer                 2.0%                  2.0%         74.1%
trained indexer                  77.1%                 56.6%         17.2%
oracle (true top-k)              90.1%                100.0%          5.0%

k =    16 ( 0.39% of tokens): attention kept, trained indexer  52.9%, oracle  73.2%, random   0.4%
k =    32 ( 0.78% of tokens): attention kept, trained indexer  65.6%, oracle  83.0%, random   0.8%
k =    64 ( 1.56% of tokens): attention kept, trained indexer  77.1%, oracle  90.1%, random   1.6%
k =   128 ( 3.12% of tokens): attention kept, trained indexer  86.2%, oracle  94.8%, random   3.1%
k =   256 ( 6.25% of tokens): attention kept, trained indexer  92.8%, oracle  97.7%, random   6.2%
k =   512 (12.50% of tokens): attention kept, trained indexer  97.0%, oracle  99.1%, random  12.5%
k =  1024 (25.00% of tokens): attention kept, trained indexer  99.0%, oracle  99.8%, random  25.0%

FP8 indexer (float8_e4m3fn q and k): 95.2% of the selected tokens are the same as in FP32
Method 1 (gather 64 latents) vs Method 2 (SDPA with a mask over all 4,096): max difference 4.8e-06

Multiply-adds per query token:
  dense attention:          16 x (32+32) x 4,096 = 4,194,304
  DSA indexer:              16 x 16 x 4,096      = 1,048,576
  DSA main (top-64):        16 x (32+32) x 64     = 65,536
  DSA total:                1,114,112  (3.8x fewer than dense)
```

"Attention kept" is the share of the true head-summed attention that falls on the selected tokens. "Output error" is the gap between the sparse and the dense output, relative to the dense output.

- **Random tokens** keep 1.5% of the attention, as expected from picking 1.56% of tokens blindly. The output is badly wrong (72% error).
- **The untrained indexer** is no better than random.
- **The trained indexer** keeps 77.1% of the attention after 37 seconds of warm-up. The output error falls to 17.2%. It finds only 56.6% of the exact top 64. But the tokens it misses have small weights, so most of the attention mass is still covered.
- **The oracle**, the true top 64, keeps 90.1% with a 5.0% error. That is the ceiling for any selector at this $$k$$.

{{FIG:toy|Share of the true attention held by the selected tokens, against k, in our toy (4,096 cached tokens). The trained indexer closes most of the gap to the oracle as k grows. Random selection keeps only k divided by 4,096.}}

Three more lines answer three more questions. FP8: rounding the indexer's queries and keys to 8 bits changes only 4.8% of the selected tokens. A ranking survives coarse numbers well, which is why DeepSeek can run the indexer in FP8. Method 2: PyTorch's own `scaled_dot_product_attention`, given a mask that allows only the selected tokens, gives the same output as our gather to within $$4.8 \times 10^{-6}$$. So the from-scratch sparse attention is correct. And the multiply-adds: 3.8 times fewer than dense here.

Why only 3.8 times, against 13.4 for V3.2 at 128K? Our toy indexer is large next to its main attention: a ratio of 4 per pair, against 17 in V3.2. And its context is 32 times shorter.

> [!WARNING] What the toy cannot show
> Random hidden states have no meaning, so the toy shows the mechanism, not the quality of a language model. Our "main model" never trains, so there is no Stage 2. Real attention has structure that our random matrices lack, such as recent tokens and repeated names. And our hidden states needed a shared mean direction. With pure zero-mean noise, the per-query weights $$w$$ flip sign at random, and the indexer failed to learn. Real hidden states have such shared directions, but this is a property of our toy data, not a claim about DeepSeek's.

## What it costs: the measured curves

The calculator counts operations. DeepSeek measured money.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · Figure 3 · page 6
> [![Figure 3 of the DeepSeek-V3.2 paper: cost per million tokens against token position from 0K to 128K; in prefilling, V3.1-Terminus rises in a straight line to about 0.67 dollars and V3.2 to about 0.19; in decoding, V3.1-Terminus rises to about 2.15 dollars and V3.2 stays near 0.25](/img/deepseek/papers/p5-v32-figure3.png)](/img/deepseek/papers/p5-v32-figure3.png)
>
> **What it says:** the inference costs of V3.1-Terminus and V3.2 on H800 clusters, per million tokens, by the token's position in the sequence. The text says the costs are "estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour".
>
> **What it means:**
>
> The x-axis is the position of a token, not the length of a request. The y-axis is what it costs to process a million tokens that sit at that position. For V3.1-Terminus, both lines climb straight up. A token at position 128K costs about 12 times more to prefill than one at the start.
>
> It costs about 30 times more to decode. That is the linear growth per token from the first section.
>
> For V3.2, both lines are almost flat after a short start. A token at 128K costs only two to three times more than one near the start. That flatness is DSA: the main attention reads 2,048 entries no matter where the token sits.

The figure has no table of numbers, but it is a vector drawing. The PDF stores each line as exact coordinates. So instead of reading the plot by eye, we read the coordinates with `part5_fig3_read.py`. It finds the grid lines for the scale, finds the blue and orange paths, and converts their corner points.

```text
prefill  V3.1-Terminus  corner points (token position, $ per 1M tokens): [(0, 0.0529), (131072, 0.6657)]
prefill  V3.2           corner points (token position, $ per 1M tokens): [(0, 0.0611), (6746, 0.0999), (131072, 0.1871)]
prefill  at 128K: Terminus $0.666, V3.2 $0.187, ratio 3.6x; curves cross near 10,688 tokens
prefill  at 32K:  Terminus $0.206, V3.2 $0.118; at 64K: Terminus $0.359, V3.2 $0.141
decode   V3.1-Terminus  corner points (token position, $ per 1M tokens): [(2, 0.0706), (131070, 2.1543)]
decode   V3.2           corner points (token position, $ per 1M tokens): [(2, 0.0953), (2107, 0.1458), (104046, 0.2275), (131070, 0.2492)]
decode   at 128K: Terminus $2.154, V3.2 $0.249, ratio 8.6x; curves cross near 4,928 tokens
decode   at 32K:  Terminus $0.592, V3.2 $0.170; at 64K: Terminus $1.112, V3.2 $0.197
```

The corner points tell a story. Each curve is made of only two to four straight segments. So these are fitted cost models drawn as lines, not raw measurements with noise. Inside them, three facts stand out.

1. **At 128K, V3.2 is 3.6 times cheaper in prefill and 8.6 times cheaper in decode.** Decode gains more because decode is memory-bound, and DSA cuts reads hardest.
2. **The decode curve of V3.2 bends at about 2,100 tokens.** That is $$k = 2{,}048$$. Below it, every token is selected anyway, and the cost rises like dense. Above it, the main attention stops growing.
3. **The prefill curve of V3.2 bends at about 6,700 tokens, and the two prefill curves cross near 10,700.** Below the bend, DeepSeek runs the masked MHA mode, which costs about the same as Terminus. Our break-even estimate from operation counts was 7,700 to 8,700 tokens, in the same range. The real crossing depends on kernels, which our count ignores.

The decode curve of V3.2 also has a small bend at about 104,000 tokens, where it gets slightly steeper. The paper does not explain it, and we do not guess.

{{FIG:fig3_ratio|How many times cheaper a token is on V3.2 than on V3.1-Terminus, by its position, read from the vector data of the paper's Figure 3. Prefill crosses 1x near 10,700 tokens; decode gains much more at long positions.}}

### What happened to the price

DeepSeek's release note for V3.2-Exp says it plainly: "DeepSeek API prices drop 50%+, effective immediately". DeepSeek also kept V3.1-Terminus on a temporary endpoint until 15 October 2025, so users could compare the two. The archived pricing pages give the exact prices in USD per million tokens:

| | V3.1 (from 5 Sep 2025) | V3.2-Exp (from 29 Sep 2025) | Cut |
|---|---|---|---|
| Input, cache hit | $0.07 | $0.028 | 60% |
| Input, cache miss | $0.56 | $0.28 | 50% |
| Output | $1.68 | $0.42 | 75% |

{{FIG:prices|DeepSeek API prices per million tokens for V3.1 and for V3.2-Exp, from the archived pricing pages before and after the launch on 29 September 2025.}}

The output price fell the most, and decode is where Figure 3 shows the biggest saving. That fits, but keep the two things apart. A price is a business decision; Figure 3 is DeepSeek's own cost estimate at an assumed rental price.

The price is flat across positions, while the cost saving depends on context length. Short chats save little; long agent sessions and long reasoning chains save a lot. DeepSeek kept the same prices for V3.2 and V3.2-Speciale in December 2025.

## DSA saves compute and reads, not storage

It is easy to read "sparse attention" as "smaller cache". For DSA, that is wrong. Every token's MLA latent must stay in memory, because any of them might be selected by a later query. And the indexer adds its own key per token per layer.

```text
KV cache per token (61 layers):
  V3, BF16 latent:            61 x 576 x 2 B = 70,272 B
  V3, FlashMLA FP8 row:       61 x 656 B     = 40,016 B
  V3.2, FP8 row + indexer:    61 x (656 + 132) B = 48,068 B (+20.1%)
  at 128K tokens: V3 FP8 5.24 GB, V3.2 6.30 GB per sequence
```

The indexer key is 128 FP8 bytes plus one 4-byte scale per layer: 132 bytes. Added to the 656-byte FP8 row of MLA, V3.2 stores 48,068 bytes per token. This number is not only ours. The DeepSeek-V4.1-Flash paper's Figure 1 shows a bar for "DeepSeek-V3.2" at exactly 48,068 bytes per token. Our sum from the code matches it to the byte.

{{FIG:kv|Bytes of cache per token. V3's BF16 latent, the same latent in DeepSeek's FP8 serving layout, and V3.2, which adds a 132-byte indexer key per layer. DSA makes the cache 20% larger, not smaller.}}

So what does DSA save? Look again at the bytes table from the start. At 128K, a new token reads 1,055 MB of indexer keys and only 82 MB of selected latents. That is 1,137 MB in all, against 5,245 MB for dense. Reads fall 4.6 times.

But 93% of what DSA still reads is the indexer's key cache, and that part still grows with the context. Decode with DSA is still linear in $$L$$; the slope is about 5 times smaller.

> [!WARNING] More context does not fit in the same memory
> With DSA, a GPU does not hold more conversations, or longer ones. The capacity limit of [LLM Inference, Part 2](llm-inference-2-kv-cache.md) is unchanged or slightly worse. What gets better is how fast each step runs once the caches are there. A server that was out of memory at 128K before V3.2 is still out of memory after it.

That remaining linear term, the full-length indexer pass, is exactly what DeepSeek attacked next. V4's compressed sparse attention shrinks the cache itself, and V4.1 goes further. [Part 6](deepseek-6-v4-and-v4-1.md) follows that path.

## DeepSeek-V3.2 and V3.2-Speciale (December 2025)

On 1 December 2025 DeepSeek released **V3.2** and **V3.2-Speciale**, both under the MIT license, and the arXiv paper followed the next day. The architecture is the same as V3.2-Exp: DSA on MLA, 671B parameters, 37B active, 128K context. The difference is post-training. The paper says post-training also "employs sparse attention in the same way as the sparse continued pre-training stage".

Specialist models for maths, coding, reasoning and agent tasks generate training data. Then one mixed reinforcement-learning stage with GRPO trains reasoning, agent skills and alignment together. DeepSeek says its RL budget now "exceeds 10% of the pre-training cost". [Part 4](deepseek-4-r1-and-reasoning-by-rl.md) covers that training in detail.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · §3 · page 6
> [![The paragraph on DeepSeek-V3.2 and DeepSeek-V3.2-Speciale, with trained exclusively on reasoning data with a reduced length penalty during RL highlighted](/img/deepseek/papers/p5-v32-speciale.png)](/img/deepseek/papers/p5-v32-speciale.png)
>
> **What it says:** V3.2-Speciale is "an experimental variant" built to explore extended thinking. It "was trained exclusively on reasoning data with a reduced length penalty during RL", and it also uses the dataset and reward method of DeepSeekMath-V2 for mathematical proofs.
>
> **What it means:**
>
> A length penalty tells a model in RL training that long answers cost something. Reduce it, and the model learns to think for much longer before it answers. Speciale is the same network as V3.2, pushed to think as long as it likes.
>
> Long thinking means long decode. On the AIME 2025 maths contest, Speciale used about 23,000 output tokens per problem on average, and about 77,000 on Codeforces problems. Every one of those tokens is a decode step at a growing position. That is the right side of the decode curve in Figure 3, where DSA cuts cost the most.

The paper reports Speciale at gold-medal level in four 2025 competitions. It scored 35 of 42 points at the IMO and 492 of 600 at the IOI. It also reached gold at the Chinese Mathematical Olympiad and solved 10 of 12 problems at the ICPC World Finals. It scored 96.0 on AIME 2025.

Speciale had no tool calls. It was served only through a temporary API endpoint until 15 December 2025, at the same price as V3.2. The regular V3.2 became the model behind DeepSeek's `deepseek-chat` and `deepseek-reasoner` API names.

> [!TIP] In production
> DeepSeek's V3.2-Exp page lists SGLang images for H200, AMD MI350 and Huawei Ascend, launched with `--tp 8 --dp 8 --enable-dp-attention`. The vLLM recipe uses 8 H200, H20 or B200 GPUs with `-dp 8 --enable-expert-parallel` and an FP8 KV cache. Each GPU runs the attention of its own requests, and the experts are spread across GPUs. [Part 7](deepseek-7-running-deepseek.md) explains these setups.

<details><summary>Research notes: sources, versions and loose ends</summary>

- **Which document.** V3.2-Exp came with its own technical report as a PDF in the GitHub repository (29 September 2025), "DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention". The arXiv paper 2512.02556 (2 December 2025) repeats its DSA sections and extends them. All quotes and page numbers here are from the arXiv version.
- **NSA training tokens.** The NSA paper says the 27B model was pre-trained on "260B tokens" in §1 (page 3) and on "270B tokens of 8k-length texts" in §4.1 (page 10). We do not know which is right.
- **NSA decode speed.** The 11.6x for decoding is the memory-access ratio of Table 4 ("expected speedup"), not a timed kernel. The forward (9.0x) and backward (6.0x) numbers are timed, on A100 GPUs with Triton kernels.
- **Indexer size.** From the V3.2-Exp `model.py` shapes: `wq_b` 1,536 x 8,192, `wk` 7,168 x 128, `weights_proj` 7,168 x 64, plus a LayerNorm of 128. About 14.0M weights per layer, 0.85B over 61 layers. The model card still calls V3.2 a 671B model.
- **The 48,068 bytes.** The indexer key cache layout (128 FP8 bytes + one float32 scale per 128 numbers) comes from the `k_cache` and `k_scale_cache` buffers in `model.py`. The MLA row (512 FP8 bytes, 16 bytes of scales, 128 bytes of BF16 RoPE key) comes from FlashMLA's README at commit ba89a34. The sum matches the V4.1-Flash paper's Figure 1(b).
- **Figure 3 values.** Read from the PDF's vector paths, not by eye. The strokes of the decode panel stop 2 tokens short of the axis ends; we clamp to the drawn range.
- **Our calculator's limits.** It counts the attention core only, in multiply-adds. It ignores the indexer's projections, softmax, top-k sorting, and kernel efficiency. It treats FP8 and BF16 operations as equal unless stated. "Weights" uses 2 x 37B FLOPs per token.
- **The API cut.** Verified on DeepSeek's release note (news250929): "API prices cut by 50%+". The exact prices come from Wayback Machine captures of the pricing page on 6 September 2025 and 4 October 2025.

</details>

> [!TAKEAWAYS]
> - At 128K context, dense MLA attention in DeepSeek-V3 costs about 2.2 trillion FLOPs per new token. That is 30 times the 74 billion of the experts and projections.
> - Decode is memory-bound: at 128K one new token reads 5.2 GB of FP8 cache per conversation, about 1.1 ms on an H200.
> - NSA (Feb 2025) mixes three gated branches: compressed block summaries, a few selected blocks of 64 tokens, and a 512-token window. It reads 5,632 keys instead of 65,536 at 64K and trains 9.0x faster forward on A100s. No released model used it as published.
> - DSA (V3.2-Exp, Sep 2025) adds a lightning indexer: 64 FP8 heads of 128 numbers, ReLU scores, one cached 128-number key per token. Each query keeps the top 2,048 tokens, shared by all 128 heads in MLA's MQA mode.
> - The main attention drops from $$O(L^2)$$ to $$O(Lk)$$; the indexer stays $$O(L^2)$$ but is about 17 times cheaper per pair, in FP8.
> - Training starts from V3.1-Terminus. A dense warm-up trains the indexer alone with a KL loss (2.1B tokens). Sparse training then trains the whole model (943.7B tokens).
> - DeepSeek's measured cost at position 128K fell 3.6x in prefill and 8.6x in decode. Short prompts use a masked dense mode, because dense is cheaper below about 10K tokens.
> - API prices fell 50% for input and 75% for output on 29 September 2025.
> - DSA makes the KV cache 20% larger (48,068 bytes per token): it saves compute and reads, not storage.
> - V3.2 and V3.2-Speciale (Dec 2025) keep the V3.2-Exp architecture; Speciale's very long reasoning is the kind of decode that DSA makes cheap.

**Next:** [Part 6, V4 and V4.1](deepseek-6-v4-and-v4-1.md). DSA left the cache as large as before and the indexer still reading every token. V4 compresses the cache itself with hybrid compressed and sparse attention, and reaches a million tokens of context.

## References

1. J. Yuan et al. [*Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention*](https://arxiv.org/abs/2502.11089). ACL 2025.
2. DeepSeek-AI. [*DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*](https://arxiv.org/abs/2512.02556). 2025.
3. DeepSeek-AI. [*DeepSeek-V3.2-Exp technical report (PDF)*](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp/blob/main/DeepSeek_V3_2.pdf) and [repository](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp). 2025.
4. DeepSeek-AI. [*DeepSeek-V2*](https://arxiv.org/abs/2405.04434) (MLA) and [*DeepSeek-V3 Technical Report*](https://arxiv.org/abs/2412.19437). 2024.
5. DeepSeek-AI. [*DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression*](https://arxiv.org/abs/2609.19969), Figure 1(b). 2026.
6. Hugging Face: [DeepSeek-V3.2-Exp inference code](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/tree/main/inference), [DeepSeek-V3.2](https://huggingface.co/deepseek-ai/DeepSeek-V3.2), [DeepSeek-V3.2-Speciale](https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Speciale).
7. Kernels: [FlashMLA](https://github.com/deepseek-ai/FlashMLA), [DeepGEMM](https://github.com/deepseek-ai/DeepGEMM), [TileLang DSA examples](https://github.com/tile-ai/tilelang/tree/main/examples/deepseek_v32), [DeepSelect](https://github.com/deepseek-ai/DeepSelect).
8. DeepSeek API news: [V3.2-Exp release, 29 Sep 2025](https://api-docs.deepseek.com/news/news250929) and [V3.2 release, 1 Dec 2025](https://api-docs.deepseek.com/news/news251201). Pricing captures: [6 Sep 2025](https://web.archive.org/web/20250906202357/https://api-docs.deepseek.com/quick_start/pricing), [4 Oct 2025](https://web.archive.org/web/20251004112357/https://api-docs.deepseek.com/quick_start/pricing).
9. vLLM recipe: [DeepSeek-V3.2-Exp](https://docs.vllm.ai/projects/recipes/en/latest/DeepSeek/DeepSeek-V3_2-Exp.html).
10. Our code, in `code/deepseek/`: `part5_cost.py` (the calculator), `part5_indexer.py` (the from-scratch indexer), `part5_fig3_read.py` (Figure 3 from the PDF's vectors).
