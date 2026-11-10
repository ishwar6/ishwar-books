---
title: "DeepSeek-V4 and V4.1: A Million Tokens of Context"
description: "How DeepSeek cut the memory per token of context from 389,120 bytes to 890: compressed attention, stable wide residuals, FP4 weights and hashed n-gram memory."
date: 2026-11-10
tags: [deepseek, attention, kv-cache, llm, long-context]
series: "DeepSeek, From the Ground Up"
series_part: 6
motif: kv
accent: "#7c9cff"
---

A coding agent opens a large repository, reads every file, and keeps the result in its working memory: about one million tokens. With DeepSeek-V3's attention, the stored context for that one conversation takes 73.7 GB. That is almost all of an 80 GB GPU, for one user, before the model weights.

DeepSeek-V4.1-Flash, released in September 2026, stores the same million tokens in 0.93 GB. Per token, it keeps 890 bytes where DeepSeek's first big model kept 389,120. This part explains how each of those bytes went away. It also covers what else changed on the way: a new residual connection, 4-bit expert weights, a faster drafter and a hash-table memory.

[Part 5](deepseek-5-sparse-attention-and-v3-2.md) ended with V3.2, which made long-context attention cheap to compute but not cheap to store. This part starts there.

> [!TIP] The whole idea in one minute
> To answer, a model reads a stored summary of every earlier token, the **KV cache**. At one million tokens that cache, not the arithmetic, is the wall. **V4** (April 2026) shrinks it along the sequence. One layer type merges every 4 tokens into one stored entry and reads only the best 512 or 1,024 entries. The other merges every 128 tokens into one and reads them all.
>
> **mHC** widens the residual stream to four copies. It forces the mixing matrices to be doubly stochastic (rows and columns sum to 1), so signals cannot explode with depth. Expert weights are stored in 4 bits. **V4.1-Flash** (September 2026) shares one cache between many layers and stores it in 4 bits. Its top half takes its cache from the bottom half. It also adds **Engram**: 196B parameters of hash tables that are looked up, not multiplied.
>
> Our script rebuilds every bar of DeepSeek's chart from the config files: 389,120, 70,272, 48,068, 3,514 and 890 bytes per token.

## The 2026 releases

DeepSeek published two model generations in 2026, and several updates of each. The weights of all of them are open under the MIT license.

{{FIG:releases|The 2026 releases. V4 came as a preview in April and as official releases in July and August. V4.1-Flash followed in September (dates from DeepSeek's release notes and Hugging Face).}}

| Model | Released | Total parameters | Active per token | Context | Routed experts (top-k) | Layers |
|---|---|---|---|---|---|---|
| V3 / V3.2 (for comparison) | Dec 2024 / Dec 2025 | 671B | 37B | 128K | 256 (8) | 61 |
| V4-Flash | Apr 2026 | 284B | 13B | 1M | 256 (6) | 43 |
| V4-Pro | Apr 2026 | 1.6T | 49B | 1M | 384 (6) | 61 |
| V4.1-Flash | Sep 2026 | 552B backbone + 196B Engram | 8B in prefill, 16B in decode | 1M | 384 (6) | 40 |

The totals and active counts are the papers' own numbers (V4 paper pages 1, 24 and 25; V4.1 paper pages 1 and 7). The layer and expert counts come from the `config.json` files on Hugging Face.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence · Abstract · page 1
> [![The V4 abstract with the sizes of V4-Pro and V4-Flash, the one-million-token context and the 27 percent FLOPs and 10 percent KV cache claim highlighted](/img/deepseek/papers/p6-v4-abstract.png)](/img/deepseek/papers/p6-v4-abstract.png)
>
> **What it says:** two MoE models, "1.6T parameters (49B activated)" and "284B parameters (13B activated)", both with a "context length of one million tokens". At that length V4-Pro needs "27% of single-token inference FLOPs and 10% of KV cache" of V3.2.
>
> **What it means:**
>
> Notice the odd pair: V4-Pro is more than twice the size of V3, yet it is cheaper to run at long context. The saving does not come from the experts. It comes from attention, the one part of a Transformer whose cost grows with the length of the conversation.
>
> The paper calls these models a "preview". The official V4-Flash-0731 and V4-Pro-0813 keep the same backbone.

Three upgrades carry V4: a hybrid attention (CSA and HCA), a new residual connection (mHC), and a new optimizer, Muon. This part covers the first two; the optimizer changes training, not the model's shape.

> [!WARNING] Engram is not part of V4
> Many articles written before the V4 release said V4 would use Engram, DeepSeek's January 2026 memory module. It does not. The V4 paper never mentions Engram, and the V4-Flash and V4-Pro configs have no Engram fields. Engram first appears in **V4.1-Flash** (`engram_layer_ids: [1, 14]` in its config). The same holds for the causal encoder-decoder and CSA2 (both explained below): they are V4.1 only.

## Why a million tokens needs a new attention

### The KV cache, again

When a model writes a token, attention compares the new token with every earlier token. The earlier tokens' keys and values never change, so the model stores them instead of computing them again.

We measure it in **bytes per token**: how much memory each token of context costs. The simplest case, a model where every layer stores a key and a value for each of its KV heads, gives:

$$
\text{bytes per token} = 2 \times L \times H_{kv} \times d_h \times b
$$

Read it left to right. The $$2$$ counts one key and one value. $$L$$ is the number of layers. $$H_{kv}$$ is the number of key-value heads per layer. $$d_h$$ is the length of each head's vector. $$b$$ is the bytes per stored number: 2 for BF16, 1 for FP8, half a byte for FP4.

A tiny example first. A 2-layer model with 1 KV head of 4 numbers in BF16 stores $$2 \times 2 \times 1 \times 4 \times 2 = 32$$ bytes per token. A 10-token prompt costs 320 bytes.

Now the real models. DeepSeek LLM 67B (2023) has 95 layers and 8 KV heads of 128 numbers in BF16:

$$
2 \times 95 \times 8 \times 128 \times 2 = 389{,}120 \text{ bytes}
$$

DeepSeek-V3 uses multi-head latent attention (MLA). Each layer stores one 512-number latent plus one 64-number position key, in BF16 ([Part 2](deepseek-2-the-v3-architecture.md); [attention, Part 2](attention-2-mqa-gqa-mla.md)):

$$
61 \times (512 + 64) \times 2 = 70{,}272 \text{ bytes}
$$

### What 70,272 bytes means at one million tokens

Multiply by the length. At 128K tokens, V3's cache is 9.21 GB per conversation. At 1M tokens ($$1{,}048{,}576$$), it is 73.69 GB. That is one conversation filling a GPU.

Memory is only the first problem. To write each new token, the model must also **read** the cache, and reading memory is slow compared with arithmetic.

V3.2 attacked the reading cost with DeepSeek Sparse Attention ([Part 5](deepseek-5-sparse-attention-and-v3-2.md)). A small "lightning indexer" scores every earlier token, and the main attention reads only the best 2,048. But V3.2 still **stores** every token, and the indexer adds its own key per token. In DeepSeek's serving format that is 656 bytes of FP8 latent plus 132 bytes of indexer key per layer:

$$
61 \times (656 + 132) = 48{,}068 \text{ bytes}
$$

So V3.2 at one million tokens would still need 50.4 GB (V3.2 itself only supports 128K). To make a million tokens routine, the model must store fewer entries, not only read fewer.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · Figure 1 · page 1
> [![Figure 1 of the V4 paper: benchmark bars on the left; on the right, single-token FLOPs and accumulated KV cache versus context length for V3.2, V4-Pro and V4-Flash](/img/deepseek/papers/p6-v4-fig1.png)](/img/deepseek/papers/p6-v4-fig1.png)
>
> **What it says:** at one million tokens, per-token FLOPs are "3.7x lower" for V4-Pro and "9.8x lower" for V4-Flash than for V3.2; the accumulated KV cache is "9.5x smaller" and "13.7x smaller".
>
> **What it means:**
>
> Look at the lower right plot. The V3.2 line climbs straight to about 50 GB at 1,024K tokens; the V4 lines barely leave the floor. A straight line means "a fixed cost per token". V4 did not bend the line; it made the slope about ten times flatter.
>
> The upper plot shows the same for compute: V3.2's indexer scores every earlier token, V4's scores compressed entries. Our script below reproduces the 13.7x and the 9.5x from the configs.

## V4's hybrid attention: CSA and HCA

### The idea: store summaries, not every token

Think of minutes of a long meeting. You could keep a recording of every sentence. Or you could write one line for every four sentences, and one paragraph for every hour. To answer a question, you scan the hourly paragraphs for the right hour. Then you read the few one-line notes that matter, plus the last few sentences word for word.

V4 does exactly this with two kinds of attention layer.

> [!DEFINITION] Compressed KV entry and compression ratio
> A **compressed entry** is one stored vector that stands for a group of consecutive tokens. It is a learned weighted mix of their keys and values. The **compression ratio** is how many tokens go into one entry: 4 for CSA, 128 for HCA. A ratio of $$m$$ divides the number of stored entries by $$m$$.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.3 · page 9
> [![The V4 paper introduces CSA and HCA; highlighted: CSA compresses every m tokens into one entry and applies DeepSeek Sparse Attention, HCA consolidates every m-prime tokens into one entry](/img/deepseek/papers/p6-v4-hybrid.png)](/img/deepseek/papers/p6-v4-hybrid.png)
>
> **What it says:** CSA "first compresses the Key-Value (KV) cache of every m tokens into one entry, and then applies DeepSeek Sparse Attention (DSA)". HCA "aims for extreme compression by consolidating the KV cache of every m' (≫ m) tokens into a single entry".
>
> **What it means:**
>
> These answer two needs. CSA keeps fine detail but cannot afford to read all of it, so it borrows V3.2's indexer to pick the best entries. HCA keeps only a coarse view, small enough to read entirely. Because the layers are "interleaved", the network gets a sharp but partial view and a blurry but complete view, again and again.

> [!DEFINITION] Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA)
> **CSA**: merge every 4 tokens into one stored entry. A lightning indexer scores the entries, and the query reads only the top-k (512 in V4-Flash, 1,024 in V4-Pro). **HCA**: merge every 128 tokens into one entry; the query reads all of them, with no indexer. Both also read the last 128 tokens uncompressed.

{{FIG:csa_hca|What one query can read. In a CSA layer, groups of 4 tokens become entries and the indexer picks a few; in an HCA layer, 128 tokens become one entry and all entries are read. Both layers add a window of recent tokens. Drawn small; real sizes in the note.}}

### CSA, step by step

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.3.1, Figure 3 · page 9
> [![Figure 3 of the V4 paper: hidden states pass through a token-level compressor to compressed KV entries; a lightning indexer with its own compressor scores them; a top-k selector picks entries, which are joined with sliding-window entries for shared key-value multi-query attention](/img/deepseek/papers/p6-v4-csa-fig.png)](/img/deepseek/papers/p6-v4-csa-fig.png)
>
> **What it says:** CSA "compresses the number of KV entries to 1/m times, and then applies DeepSeek Sparse Attention for further acceleration". A small set of sliding-window entries joins the selected compressed entries.
>
> **What it means:**
>
> Follow the picture from the bottom. One compressor turns the earlier tokens into compressed KV entries. A second one, inside the dashed "Lightning Indexer" box, makes small keys for scoring. The top-k selector keeps the best entries, and only those reach the main attention.
>
> Compare it with V3.2 in Part 5. The indexer is the same idea, but it now ranks entries that each stand for 4 tokens. V4-Flash keeps 512 entries, which cover about 2,048 tokens of text. That is the same reach as V3.2's top-2,048, from a quarter of the stored entries. That last point is our reading of the numbers, not a sentence in the paper.

How does the compressor turn 4 tokens into one entry? It computes two candidate KV vectors for every token, series $$a$$ and series $$b$$, plus a score for each. Entry $$i$$ mixes the 4 tokens of its own group from series $$a$$ and the 4 tokens of the previous group from series $$b$$:

$$
C^{\text{Comp}}_i = \sum_{j \in \text{own } m} S^a_j \odot C^a_j \; + \sum_{j \in \text{previous } m} S^b_j \odot C^b_j
$$

Here $$C^a_j$$ and $$C^b_j$$ are the two candidate vectors of token $$j$$, each 512 numbers long. $$S^a_j$$ and $$S^b_j$$ are weights from one softmax over all $$2m = 8$$ scores, with a learned bias for each slot. The symbol $$\odot$$ means "multiply number by number", so each of the 512 channels gets its own weights.

A tiny example with $$m = 2$$ and one channel. Own tokens have values 4 and 8 with scores 0 and $$\ln 3$$. The previous tokens have values 2 and 6 with scores 0 and 0. The softmax over the four scores turns $$e^0, e^{\ln 3}, e^0, e^0 = 1, 3, 1, 1$$ into weights $$\tfrac16, \tfrac12, \tfrac16, \tfrac16$$. The entry is $$\tfrac16 \cdot 4 + \tfrac12 \cdot 8 + \tfrac16 \cdot 2 + \tfrac16 \cdot 6 = 6$$. The token with the highest score dominates, but every token contributes.

{{FIG:csa_overlap|CSA builds entry i from its own 4 tokens and the 4 tokens before them. Neighbouring entries overlap, but there is still one entry per 4 tokens.}}

Why read 8 tokens to make an entry that replaces 4? A hard cut every 4 tokens would split phrases at random: "New" could land in one group and "York" in the next. With the overlap, a phrase that crosses a boundary still reaches one entry whole. The paper notes that CSA "in fact compresses the sequence length to 1/m times": the overlap costs compute, not memory.

Two more details make the stored entries small. First, CSA and HCA use one shared KV head, as in multi-query attention. All 64 or 128 query heads read the same entry.

V4 goes one step further than classic multi-query attention: the same 512-number entry serves as both the key and the value ([attention, Part 2](attention-2-mqa-gqa-mla.md)).

Second, V4 stores the entry in mixed precision. The 448 numbers that carry content go in FP8, and the 64 numbers that carry position (RoPE) stay in BF16. With 8 bytes of scale factors, one entry takes $$448 + 64 \times 2 + 8 = 584$$ bytes (FlashMLA's documented layout).

> [!DEFINITION] RoPE (rotary position embedding)
> A way to tell attention where each token sits. Part of every query and key vector is rotated by an angle that depends on the token's position. The match between a query and a key then depends on their distance. V4 applies it to the last 64 of the 512 numbers.

### HCA: read everything, because there is little to read

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.3.2, Figure 4 · page 11
> [![Figure 4 of the V4 paper: HCA has one token-level compressor that produces heavily compressed KV entries, joined with sliding-window entries, and no indexer](/img/deepseek/papers/p6-v4-hca-fig.png)](/img/deepseek/papers/p6-v4-hca-fig.png)
>
> **What it says:** HCA "performs heavier compression, where the KV entries of m' (≫ m) tokens will be consolidated into one", and adds sliding-window entries.
>
> **What it means:**
>
> Compared with CSA, the whole right half is gone: no indexer, no top-k. With $$m' = 128$$, a million-token context becomes $$1{,}048{,}576 / 128 = 8{,}192$$ entries, cheap enough to read in full.
>
> Each entry mixes exactly its own 128 tokens, with no overlap. That is blurry: it can say "the test file is around here" but not which line. So HCA never works alone; CSA layers and the window supply the detail.

### The window, and why it is needed

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.3.3 · page 13
> [![The V4 paper on the additional sliding-window branch; highlighted: a query cannot access other tokens within its own compressed block, and recent tokens usually possess greater relevance](/img/deepseek/papers/p6-v4-swa.png)](/img/deepseek/papers/p6-v4-swa.png)
>
> **What it says:** because a query reads only completed compressed blocks, "a query cannot access information from other tokens within its own compressed block", and "recent tokens usually possess greater relevance". So each query also reads the last $$n_{\text{win}}$$ tokens uncompressed.
>
> **What it means:**
>
> Without the window, a token in the middle of an HCA block could not see the sentence it is writing. Its neighbours become an entry only when the block is complete.
>
> The window is 128 tokens in every V4 layer. It costs a fixed amount per conversation, not per token. For V4-Flash our script estimates $$128 \times 43 \times 584 = 3{,}214{,}336$$ bytes, about 3.2 MB, whatever the length.

V4 adds two smaller tricks to the core attention. An **attention sink** is one learned score per head, added to the softmax's denominator. With it, a head can give almost no weight to anything when nothing is relevant. And the output of each head is rotated back by its own position, so the result carries relative, not absolute, position.

### How the layers combine

{{FIG:layers|The layer plans, read from compress_ratios in the configs. V4-Flash starts with two window-only layers, V4-Pro with two HCA layers; then both alternate CSA and HCA. V4.1-Flash is covered further down.}}

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §4.2.1 · page 24
> [![The V4-Flash model setup: 43 layers, d 4096, the first two layers use pure sliding window attention, then CSA and HCA interleave; CSA top-k 512; 284B total and 13B activated parameters](/img/deepseek/papers/p6-v4-flash-setup.png)](/img/deepseek/papers/p6-v4-flash-setup.png)
>
> **What it says:** "For the first two layers, we use pure sliding window attention. For the subsequent layers, CSA and HCA are used in an interleaved manner." CSA uses $$m = 4$$ and "attention top-k) to 512"; HCA uses $$m' = 128$$.
>
> **What it means:**
>
> The first layers of a Transformer mostly build local features, so V4-Flash gives them only the window. V4-Pro uses HCA there instead.
>
> The config file says the same thing in numbers. V4-Flash's `compress_ratios` list starts `[0, 0, 4, 128, 4, 128, ...]`: 0 means window only, 4 means CSA, 128 means HCA. Counting the list gives 2 window-only layers, 21 CSA layers and 20 HCA layers. The last entry of the list belongs to the MTP module, not to the 43 main layers.

### Counting the bytes

Now we can write V4's bytes per token. Each layer stores entries of $$e$$ bytes, one per $$m$$ tokens, so it costs $$e / m$$ bytes per token:

$$
\text{bytes per token} = \sum_{\text{layers}} \frac{e_{\text{main}} + e_{\text{indexer}}}{m}
$$

For a CSA layer, $$e_{\text{main}} = 584$$ bytes. The indexer key is 128 numbers in MXFP4: 64 bytes of 4-bit values plus one 1-byte scale per 32 values, so $$e_{\text{indexer}} = 68$$. For an HCA layer, $$e_{\text{indexer}} = 0$$. The window layers add nothing per token.

> [!DEFINITION] FP8, FP4 and MXFP4
> Number formats with 8 or 4 bits per value instead of 16 (BF16). FP4 (E2M1) can only hold 16 different values, so it is used with **scales**. A group of values shares one scale factor that sets their range. **MXFP4** is the open standard with one 8-bit scale per 32 values. The scales cost $$8 / 32 = 0.25$$ extra bits per value.

Worked by hand for V4-Flash: $$21 \times (584 + 68) / 4 = 3{,}423$$ bytes from the CSA layers, and $$20 \times 584 / 128 = 91.25$$ bytes from the HCA layers. The total is **3,514 bytes**, and that is exactly the number DeepSeek prints for V4-Flash. The indexer layout of 68 bytes is our reading of the MXFP4 format. The exact match with the published total is good evidence that it is right.

Here is the part of our script that does this from the config:

```python
V4_MAIN = 448 * 1 + 64 * 2 + 8               # 448 FP8 + 64 BF16 RoPE + 8-byte scale row = 584
FP4_INDEX = 128 // 2 + 128 // 32             # 128 MXFP4 values (64 B) + 4 one-byte scales = 68

def v4(name):
    c = cfg(name)
    L = c['num_hidden_layers']
    ratios = c['compress_ratios'][:L]          # the last entry belongs to the MTP layer
    csa, hca, swa = ratios.count(4), ratios.count(128), ratios.count(0)
    b_csa = csa * (V4_MAIN + FP4_INDEX) / 4
    b_hca = hca * V4_MAIN / 128
```

```text
  DeepSeek-V4-Flash: 43 layers = 2 sliding-window only + 21 CSA (ratio 4) + 20 HCA (ratio 128)
     CSA: 21 x (584 + 68) B / 4  = 3,423.00 B
     HCA: 20 x 584 B / 128       = 91.25 B
  DeepSeek-V4-Pro: 61 layers = 0 sliding-window only + 30 CSA (ratio 4) + 31 HCA (ratio 128)
     CSA: 30 x (584 + 68) B / 4  = 4,890.00 B
     HCA: 31 x 584 B / 128       = 141.44 B
V4-Pro (not in the chart): 5,031 B per token; V3.2 / V4-Pro = 9.55x, V3.2 / V4-Flash = 13.68x  (V4 paper Figure 1: "9.5x smaller", "13.7x smaller")
```

{{FIG:bytes_split|Where the bytes come from. In V4 the CSA layers hold almost everything; twenty or more HCA layers add under 150 bytes per token together. V4.1-Flash stores KV in only four layers (explained below).}}

The ratios match the paper's Figure 1: 48,068 / 3,514 is 13.68, and 48,068 / 5,031 is 9.55. Notice what the split shows. The HCA layers are almost free, so the 128-token view of the whole context costs nearly nothing. The real budget is the CSA layers, and that is where V4.1 will cut next.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.3.4 · page 13
> [![The V4 efficiency discussion; highlighted: BF16 for the RoPE dimensions, FP8 for the remaining dimensions, the indexer computes in FP4, and the KV cache is about 2 percent of a BF16 GQA8 baseline at 1M context](/img/deepseek/papers/p6-v4-efficiency.png)](/img/deepseek/papers/p6-v4-efficiency.png)
>
> **What it says:** "BF16 precision is used for the rotary positional embedding" dimensions while "FP8 precision is applied to the remaining dimensions"; the indexer's attention is "performed in FP4 precision"; against a BF16 GQA8 model with head dimension 128 the KV cache is "approximately 2% times of that baseline in the 1M-context setting".
>
> **What it means:**
>
> Why keep the position part in BF16? The rotation angles of RoPE encode distance, and small rounding errors there turn into errors of position. The content part tolerates FP8, so it gets half the bytes.
>
> The 2% claim checks out. A GQA8 model with V4-Flash's 43 layers stores $$2 \times 8 \times 128 \times 2 \times 43 = 176{,}128$$ bytes per token. V4-Flash's 3,514 bytes are 2.0% of that; V4-Pro's 5,031 bytes are also 2.0% of its 61-layer baseline.

## mHC: a wider residual stream that cannot explode

### Residual connections

Every Transformer layer has a bypass around it. The layer's input skips past the layer and is added to its output.

> [!DEFINITION] Residual connection
> The rule $$x_{l+1} = x_l + F(x_l)$$: the output of layer $$l$$ is its input plus what the layer computes. The stream $$x$$ that runs through all layers is the **residual stream**. Because the input passes through untouched, a signal from layer 1 reaches layer 60 without being multiplied by anything.

That untouched path is the reason deep networks train at all. Unroll the rule over many layers and the first input is still there, unchanged, in the last layer: $$x_L = x_l + \sum F(\ldots)$$. Gradients flow back the same way. The mHC paper calls this the **identity mapping** property.

### Hyper-connections: four streams instead of one

In 2024, researchers proposed widening the stream: keep $$n$$ copies of it, and let each layer mix them.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.2 · page 7
> [![The V4 paper's description of standard hyper-connections; highlighted: the residual stream is expanded by a factor n_hc, with an input mapping A, a residual transformation B and an output mapping C, and the update X_l+1 = B X_l + C F(A X_l)](/img/deepseek/papers/p6-v4-hc.png)](/img/deepseek/papers/p6-v4-hc.png)
>
> **What it says:** standard HC "expands the width of the residual stream by a factor of" $$n_{hc}$$, with an "input mapping" $$A_l$$, a "residual transformation" $$B_l$$ and an "output mapping" $$C_l$$, and updates $$X_{l+1} = B_l X_l + C_l F_l(A_l X_l)$$.
>
> **What it means:**
>
> The layer itself does not get wider; only the bypass does. V4 uses $$n_{hc} = 4$$ (`hc_mult: 4` in the config). The mixing weights are produced from the current hidden state, so each token gets its own mixing. The extra arithmetic is tiny, because 4 is tiny next to a hidden size of 4,096 or 7,168.

{{FIG:residual_hc|A residual connection adds the layer's output to an untouched input. Hyper-connections keep four copies of the stream; B mixes them on the bypass, A reads one input from them and C writes the output back.}}

Read the update rule symbol by symbol:

$$
X_{l+1} = B_l X_l + C_l \, F_l(A_l X_l)
$$

$$X_l$$ is the residual state before layer $$l$$: $$n$$ rows of $$d$$ numbers. $$A_l X_l$$ is one row of $$d$$ numbers, the input that layer $$F_l$$ sees. $$C_l F_l(\cdot)$$ copies the layer's output into the $$n$$ streams with $$n$$ weights. $$B_l X_l$$ is the bypass, now a mix of streams instead of a plain copy. With $$n = 1$$ and $$B = C = A = 1$$, the rule is the ordinary residual connection.

### Why unconstrained mixing explodes

The bypass is now a matrix. Unroll it over many layers and the first input reaches layer $$L$$ multiplied by the **product** $$B_{L-1} \cdots B_{l+1} B_l$$. Products of matrices are dangerous.

A tiny example. Let every layer use $$B = \begin{pmatrix} 1.1 & 0.1 \\ 0.1 & 1.1 \end{pmatrix}$$. Each row sums to 1.2, so a signal $$[1, 1]$$ grows by 1.2 per layer. After 40 layers it is $$1.2^{40} \approx 1{,}470$$:

```text
1) A 2 x 2 matrix that is a little too big: H = [[1.1, 0.1], [0.1, 1.1]], every row sums to 1.2
   after  1 layers the signal [1, 1] becomes [1.2, 1.2]
   after 10 layers the signal [1, 1] becomes [6.2, 6.2]
   after 20 layers the signal [1, 1] becomes [38.3, 38.3]
   after 40 layers the signal [1, 1] becomes [1,469.8, 1,469.8]
```

Each matrix looks harmless, yet 20% too much per layer becomes a factor of a thousand over depth. Learned matrices differ from layer to layer, so real growth is a random walk, but the danger is the same.

> [!PAPER] Xie et al. (2025), mHC: Manifold-Constrained Hyper-Connections · §3.1 · page 6
> [![The mHC paper's instability evidence; highlighted: HC exhibits an unexpected loss surge around the 12k step](/img/deepseek/papers/p6-mhc-amax.png)](/img/deepseek/papers/p6-mhc-amax.png)
>
> **What it says:** in a 27B model, "HC exhibits an unexpected loss surge around the 12k step", which tracks the gradient norm. To measure the cause, the authors take the product of the mixing matrices and compute the largest row sum (forward gain) and the largest column sum (backward gain), which they call the "Amax Gain Magnitude".
>
> **What it means:**
>
> A loss surge 12,000 steps into a run costs real money: the run looks healthy, then breaks. The Amax gain asks "how much can this product amplify anything?". A row sum of 3 means some stream can come out three times stronger. For a plain residual connection the answer is always 1.

> [!PAPER] Xie et al. (2025), mHC · Figure 3 · page 7
> [![Figure 3 of the mHC paper: single-layer gains of HC stay near 1, but the composite backward gain climbs to about 3,000 around layer 45 on a log scale](/img/deepseek/papers/p6-mhc-fig3.png)](/img/deepseek/papers/p6-mhc-fig3.png)
>
> **What it says:** the single-layer gain (left) stays near 1 for most layers. The composite gain (right) rises on a log scale, with "peaks of 3000" (page 6).
>
> **What it means:**
>
> This is the 2 x 2 example at full scale. Every single matrix looks fine on the left; the danger appears only in the product, on the right.

We reproduced the effect with random matrices: 4 streams, 120 mixing layers and 5 random seeds. A 60-block model has an attention and an MoE layer per block, so 120 mixing layers. HC uses $$I + 0.2\,G$$, a matrix near the identity with random noise $$G$$. mHC uses the projected matrix described next.

```text
3) n = 4 streams, 120 mixing layers (a 60-block model: attention + MoE per block), noise scale 0.2, 5 seeds
   layer   HC gain (median / max of seeds)   mHC gain (median / max)
       1            1.8 /          2.3         1.000 /  1.000
      10            4.9 /          8.8         1.000 /  1.000
      30            7.1 /         14.1         1.000 /  1.000
      60           50.1 /         98.1         1.000 /  1.000
      90          165.3 /        330.5         1.000 /  1.000
     120          439.0 /        661.7         1.000 /  1.000
   largest single-layer spectral norm: HC 1.824, mHC 1.000104
   NumPy Sinkhorn vs PyTorch log-domain Sinkhorn: largest difference 4.4e-16
```

{{FIG:mhc_gain|Our toy, on a log scale. The gain of unconstrained mixing matrices grows to hundreds over 120 layers; the projected (mHC) matrices hold it at 1 in every seed.}}

Our toy is not a trained model: the noise is random, not learned. It shows only the mechanism: small errors per layer compound. The paper's 3,000 comes from a real 27B training run.

### The constraint: doubly stochastic mixing

mHC keeps the four streams and the mixing, but restricts what $$B_l$$ can be.

> [!DEFINITION] Doubly stochastic matrix
> A square matrix with no negative entries in which **every row sums to 1 and every column sums to 1**. Example: $$\begin{pmatrix} 0.366 & 0.634 \\ 0.634 & 0.366 \end{pmatrix}$$. The set of all of them is called the Birkhoff polytope.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.2 · page 8
> [![The V4 paper's mHC section; highlighted: HC training frequently exhibits numerical instability when stacking layers; the residual mapping is constrained to doubly stochastic matrices; its spectral norm is bounded by 1; the set is closed under multiplication](/img/deepseek/papers/p6-v4-mhc-constraint.png)](/img/deepseek/papers/p6-v4-mhc-constraint.png)
>
> **What it says:** with HC, DeepSeek finds "the training will frequently exhibit numerical instability when stacking multiple layers". mHC constrains $$B_l$$ to "doubly stochastic matrices"; the spectral norm "is bounded by 1", and the set "is closed under multiplication".
>
> **What it means:**
>
> Two facts, and both are about products. "Spectral norm at most 1" means one matrix can never make any signal longer. "Closed under multiplication" means the product of many doubly stochastic matrices is itself doubly stochastic. So the product over 60 or 120 layers obeys the same limit as one layer. The 2 x 2 example cannot happen.
>
> A third fact: because every column sums to 1, the **sum** of the streams is kept exactly. With one stream, the only doubly stochastic "matrix" is the number 1, and mHC becomes the plain residual connection.

Here is the mean-keeping property on numbers from our script. Mix the streams $$[4, 2]$$ (mean 3) with the matrix above:

$$
\begin{pmatrix} 0.366 & 0.634 \\ 0.634 & 0.366 \end{pmatrix} \begin{pmatrix} 4 \\ 2 \end{pmatrix} = \begin{pmatrix} 2.732 \\ 3.268 \end{pmatrix}
$$

The streams changed; their mean is still 3.000. Each output is a weighted average of the inputs, with weights that sum to 1, so no output can be bigger than the biggest input.

### Getting a doubly stochastic matrix: Sinkhorn-Knopp

The network produces a raw matrix of any numbers. A 1967 algorithm turns it into a doubly stochastic one.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §2.2, Eq. 8 · page 8
> [![The V4 paper's projection step; highlighted: the Sinkhorn-Knopp algorithm, which applies an exponential and then alternates column and row normalization, with t_max = 20 as a practical value](/img/deepseek/papers/p6-v4-sinkhorn.png)](/img/deepseek/papers/p6-v4-sinkhorn.png)
>
> **What it says:** the "Sinkhorn-Knopp algorithm" first applies an exponential, $$M^{(0)} = \exp(\tilde B_l)$$, "and then iteratively performs column and row normalization". "We choose" $$t_{max} = 20$$ "as a practical value".
>
> **What it means:**
>
> The exponential makes every entry positive. Then divide each column by its sum, then each row by its sum, and repeat. Each step fixes one kind of sum and slightly breaks the other, but the error shrinks every round.
>
> Twenty rounds is a trade. The result is close to, not exactly, doubly stochastic. Our toy measured the largest spectral norm after 20 rounds at 1.000104, not 1. The mHC paper reports the same effect at scale: the composite gain reaches about 1.6 instead of 1, far below HC's 3,000.

Run it by hand on $$\exp(\text{raw}) = \begin{pmatrix} 1 & 3 \\ 1 & 1 \end{pmatrix}$$. The column sums are 2 and 4. Dividing gives $$\begin{pmatrix} 0.5 & 0.75 \\ 0.5 & 0.25 \end{pmatrix}$$, whose rows sum to 1.25 and 0.75. Dividing each row by its sum gives $$\begin{pmatrix} 0.4 & 0.6 \\ 0.667 & 0.333 \end{pmatrix}$$. Now the columns are off a little: 1.067 and 0.933. Keep going:

```python
def sinkhorn(raw, iters=20):
    """Method 1 (NumPy). exp() makes every entry positive; then alternate column and row normalisation.
    Same order as the paper's Eq. 8/9: M(t) = T_r(T_c(M(t-1)))."""
    M = np.exp(raw)
    for _ in range(iters):
        M = M / M.sum(axis=0, keepdims=True)     # T_c: every column sums to 1
        M = M / M.sum(axis=1, keepdims=True)     # T_r: every row sums to 1
    return M
```

```text
2) Sinkhorn-Knopp on raw = [[0, ln 3], [0, 0]], so exp(raw) = [[1, 3], [1, 1]]
   step 1 columns: [[0.500, 0.750], [0.500, 0.250]]  row sums [1.25, 0.75]
   step 1 rows   : [[0.400, 0.600], [0.667, 0.333]]  column sums [1.067, 0.933]
   step 2 columns: [[0.375, 0.643], [0.625, 0.357]]  row sums [1.018, 0.982]
   step 2 rows   : [[0.368, 0.632], [0.636, 0.364]]  column sums [1.005, 0.995]
   step 3 columns: [[0.367, 0.635], [0.633, 0.365]]  row sums [1.001, 0.999]
   step 3 rows   : [[0.366, 0.634], [0.634, 0.366]]  column sums [1.0, 1.0]
   after 20 steps: [[0.3660, 0.6340], [0.6340, 0.3660]], rows [1.0, 1.0], columns [1.0, 1.0]
   mixing the streams [4, 2]: [2.732, 3.268], mean before 3.000, mean after 3.000
```

{{FIG:sinkhorn|Sinkhorn-Knopp on a 2 x 2 matrix. Three rounds already give sums within 0.001 of 1; V4 runs 20 rounds for every 4 x 4 matrix, at every layer, for every token.}}

The error falls about tenfold per round, so three rounds are already close. The script also checks this NumPy version against a PyTorch version written in log space, the safe form for large exponents: they agree to $$4.4 \times 10^{-16}$$.

> [!PAPER] Xie et al. (2025), mHC · §4.1 · page 8
> [![The mHC paper's list of properties; highlighted: norm preservation, the spectral norm is bounded by 1; compositional closure, closed under matrix multiplication; and the mapping acts as a convex combination of permutations](/img/deepseek/papers/p6-mhc-props.png)](/img/deepseek/papers/p6-mhc-props.png)
>
> **What it says:** "Norm Preservation", "Compositional Closure", and a geometric view: the residual mapping "acts as a convex combination of permutations".
>
> **What it means:**
>
> A permutation matrix just reorders the streams: stream 1 goes to slot 3, and so on. A doubly stochastic matrix is always a weighted average of such reorderings (Birkhoff's theorem). So mHC's bypass can shuffle and blend the streams, but never amplify them.
>
> The same property has a cost that the paper states gently: repeated averaging makes the streams more and more alike. In our toy, after 120 layers of pure mixing the four streams $$[3, -1, 2, 0]$$ all became 1.000, their mean. In a real model the layer outputs $$C_l F_l(\cdot)$$ keep adding new differences, so the streams stay useful.

### What mHC costs and what it buys

The constraint has a price in time and memory. Four streams mean four times more data on the bypass, and the 20 Sinkhorn rounds run at every layer.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §3.4.2 · page 20
> [![The V4 paper's mHC engineering: fused kernels, selective recomputation and an adjusted DualPipe schedule; highlighted: the wall-time overhead of mHC is only 6.7 percent of the overlapped 1F1B pipeline stage](/img/deepseek/papers/p6-v4-mhc-cost.png)](/img/deepseek/papers/p6-v4-mhc-cost.png)
>
> **What it says:** with fused kernels, recomputation and a changed pipeline schedule, "the wall-time overhead of mHC" is "only 6.7% of the overlapped 1F1B pipeline stage".
>
> **What it means:**
>
> The arithmetic of mHC is small; the memory traffic is not, because every layer reads and writes four copies of the hidden state. 6.7% more training time is a real cost. DeepSeek pays it because the mHC paper's 27B tests showed clear gains over a plain residual baseline. Examples: BBH 43.8 to 51.0 and DROP 47.0 to 53.9 (mHC paper, Table 4). V4.1 later cuts the traffic further (Single-Pass mHC, below).

## FP4 expert weights

Most of a V4 model's parameters sit in its routed experts. In V4-Flash they are $$43 \times 256 \times 3 \times 4{,}096 \times 2{,}048 = 277.0$$ billion of the 284 billion, by our count from the config. Shrinking the experts shrinks the model. V4 stores them in FP4. On today's chips an FP4 x FP8 product runs at FP8 speed, so the gain is memory, not speed (V4 paper, page 5).

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4 · §5.2.1 · page 33
> [![The V4 paper's FP4 quantization-aware training; highlighted: MoE expert weights are a major source of GPU memory occupancy, the FP4-to-FP8 dequantization is lossless, and FP8 has 2 additional exponent bits](/img/deepseek/papers/p6-v4-qat.png)](/img/deepseek/papers/p6-v4-qat.png)
>
> **What it says:** FP4 (MXFP4) is applied to "MoE expert weights, which are a major source of GPU memory occupancy", and to the indexer's query-key path, during post-training. "Our FP4-to-FP8 dequantization is lossless" because FP8 has "2 additional exponent bits".
>
> **What it means:**
>
> The model is not trained in FP4 from the start. In post-training, the full-precision weights are rounded to FP4 at every step, and the model learns to work well with the rounded version. This is **quantization-aware training**: the model sees its own rounding errors while it can still adapt to them.
>
> "Lossless" means an FP4 value with its scale converts into FP8 exactly, because FP8 has more exponent range. So the FP8 matrix units compute with exactly the FP4 weights the model was trained on.

Our script checks the size this gives:

```text
V4-Flash routed experts: 43 layers x 256 x 3 x 4096 x 2048 = 277.0B parameters
  in FP4 with one 1-byte scale per 32 values: 147.2 GB;  the other ~7B in FP8: ~7 GB;  estimate 154 GB  (published files: 159.6 GB)
  the same 284B in BF16 would be 568 GB
```

Our estimate of 154 GB is close to the 159.6 GB of the published V4-Flash files. We do not count the shared experts, norms, scales of the FP8 parts and file overhead, so the small gap is expected. The point is the other line: in BF16 the same model would be 568 GB. [Part 7](deepseek-7-running-deepseek.md) turns these sizes into GPU counts.

## DSpark and the official V4 releases

### What DSpark adds

DSpark is V4's new draft model for speculative decoding ([Part 2](deepseek-2-the-v3-architecture.md) shows how V3 drafted with its MTP module). It replaces that one-token drafter.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · §2.4.3 · page 14
> [![The V4.1 paper's description of DSpark; highlighted: three Transformer blocks, logits for five draft positions in parallel, a lightweight Markov head, and a confidence head](/img/deepseek/papers/p6-v41-dspark.png)](/img/deepseek/papers/p6-v41-dspark.png)
>
> **What it says:** the drafter has "three Transformer blocks" with a 128-token window. One pass gives base logits for "five draft positions in parallel", a "lightweight Markov head" models dependencies between the draft tokens, and a "confidence head predicts" how likely each draft token is to be accepted.
>
> **What it means:**
>
> Drafting five tokens at once is fast, but each position guesses without knowing the guesses before it. Token 4 might assume a different token 3. The Markov head is a small sequential layer that lets each draft position look at the previous one, which keeps late positions from decaying.
>
> The confidence head serves the server. When it is busy, checking drafts that will probably fail wastes batch space. So DSpark verifies only as far as a draft is likely to survive.

> [!PAPER] Cheng et al. (2026), DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation · Abstract · page 1
> [![The DSpark abstract; highlighted: compared with the MTP-1 production baseline, DSpark accelerates per-user generation speeds by 60 to 85 percent at matched throughput levels](/img/deepseek/papers/p6-dspark-abstract.png)](/img/deepseek/papers/p6-dspark-abstract.png)
>
> **What it says:** in the DeepSeek-V4 serving system, under live traffic, compared with the MTP-1 baseline, DSpark "accelerates per-user generation speeds by 60%–85% at matched throughput levels".
>
> **What it means:**
>
> "At matched throughput" is the honest way to state a speed-up. A server can always make one user faster by serving fewer users. Here the total tokens per second stays the same, and each user still sees text 1.6 to 1.85 times faster. This is a measurement on DeepSeek's own service, not a public benchmark.

DeepSeek first shipped DSpark checkpoints for the V4 preview in June. Then came the official releases. **V4-Flash-0731** "is the official release of DeepSeek-V4-Flash, superseding the preview version", with "a speculative decoding module attached" (model card). **V4-Pro-0813** is "built on the DeepSeek-V4-Pro (Preview) model structure, with a DSpark speculative decoding module attached". Its config adds DSpark fields such as `dspark_block_size: 5`. In V4.1-Flash, DSpark replaces MTP completely: the backbone trains without MTP, and the drafter trains afterwards with the backbone frozen.

## V4.1-Flash: 890 bytes per token

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression · Abstract · page 1
> [![The V4.1-Flash abstract; highlighted: 552B backbone parameters, 16B parameters active per token in decode and 8B in prefill, and a global KV cache of 890 bytes per token, about 1/4 of V4-Flash](/img/deepseek/papers/p6-v41-abstract.png)](/img/deepseek/papers/p6-v41-abstract.png)
>
> **What it says:** "552B backbone parameters"; the model "activates 16B parameters per token during decode but only 8B parameters during prefill"; its global KV cache is "890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash".
>
> **What it means:**
>
> The first sentence of this abstract names the workload: agents. An agent calls a tool, gets a long result, and the model must read it before writing a few tokens. The input is huge and the output is small. So V4.1 attacks prefill cost and stored cache, more than writing speed.
>
> Note the two active counts: V4.1 uses half its layers to read most of a prompt.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · Figure 3 · page 7
> [![Figure 3 of the V4.1 paper: a causal encoder of SWA and CSA2 layers with Engram, text and vision embeddings, feeding its hidden states to a decoder of CSA2 layers in Full, Reindex and Reuse modes, with a candidate pool and DSpark on top](/img/deepseek/papers/p6-v41-arch.png)](/img/deepseek/papers/p6-v41-arch.png)
>
> **What it says:** "The 40-layer network is divided into a causal encoder and a decoder, each with 20 layers." All feed-forward layers are DeepSeekMoE. The model also uses Single-Pass mHC, Engram, DSpark and a Hierarchical Sparse Indexer.
>
> **What it means:**
>
> Read the picture from the bottom left. Text and image embeddings, and Engram, enter the encoder. The encoder's output, "Enc. Hidden States", crosses to the right and feeds the decoder's first layer. Inside each half, labels like "CSA2(2, Reuse) x5" mean five layers with compression ratio 2 that reuse an earlier layer's cache.
>
> "Causal" matters: unlike a translation encoder, every layer here still looks only backwards. It remains a normal left-to-right language model.

### The causal encoder-decoder

{{FIG:ced|In the causal encoder-decoder, the decoder layers project their long-range KV from the output of the last encoder layer. A long prompt mostly needs only the encoder half during prefill.}}

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · §2.2, Eq. 1 · page 9
> [![The V4.1 paper's CED section; highlighted: decoder KV entries are projected directly from the hidden state of layer L/2, so CED computes only the first half of the layers during prefill](/img/deepseek/papers/p6-v41-ced.png)](/img/deepseek/papers/p6-v41-ced.png)
>
> **What it says:** for decoder layers ($$l > L/2$$) the KV entries "are projected directly from the hidden state of" layer $$L/2$$: $$C_l = H_{L/2} W^{KV}_l$$. This lets CED "compute only the first half of the layers during the prefill phase".
>
> **What it means:**
>
> In a normal model, layer 30's cache comes from layer 30's own hidden state, which needs layers 0 to 29 first. Here, layer 30's long-range cache is a projection of layer 19's output. So for a prompt token, running layers 0 to 19 is enough to fill every layer's long-range cache. The decoder layers run only for the tokens the model writes, and for a short recent stretch of the prompt.
>
> The recent stretch remains because each decoder layer keeps its own sliding window. "SWA Bounded Replay" runs the decoder over only the last 128 prompt tokens, a small approximation. The idea builds on YOCO, where the top half of a model shares one cache built by the bottom half.

### CSA2: one cache, many layers

V4 stored a separate cache in every CSA and HCA layer. V4.1 asks a sharper question: do neighbouring layers need different caches at all?

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · Figure 4 · page 10
> [![Figure 4 of the V4.1 paper: three CSA2 modes. Full mode computes main KV, indexer K and top-k indices itself; Reindex mode reuses main KV and indexer K but computes new indices; Reuse mode reuses KV and indices](/img/deepseek/papers/p6-v41-modes.png)](/img/deepseek/papers/p6-v41-modes.png)
>
> **What it says:** "Green blocks indicate quantities computed in the current layer; yellow blocks indicate main KV and indexer K reused from the most recent Full Mode layer; while red blocks indicate Top-K indices reused". Every mode computes its own query and window KV.
>
> **What it means:**
>
> A **Full** layer is a complete CSA layer: it builds and stores the cache, and its indexer picks the top-k. A **Reindex** layer reads the cache of the last Full layer but asks its own question: new indexer query, new top-k. A **Reuse** layer copies both the cache and the choice of entries from an earlier layer, and only brings its own query.
>
> Only Full layers add bytes. In V4.1-Flash, 4 of the 40 layers are Full: encoder layers 2, 8 and 14 (ratio 2), and decoder layer 20 (ratio 1). The config lists them as `kv_source_layer_ids: [2, 8, 14, 20]`. V4.1 also drops HCA entirely: every global layer is CSA2.

The decoder adds one more trick, the **Hierarchical Sparse Indexer**. The first decoder layer scores every position, then keeps the 2,048 best blocks of 8 positions: a pool of 16,384 candidates. Later Reindex layers score only that pool. Their cost per query no longer grows with the length of the context.

The last cut is precision. V4 stored the main entry in FP8 with BF16 for the position part. V4.1 stores all of it in FP4.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · §2.4.4 · page 14
> [![The V4.1 paper's FP4 main KV cache; highlighted: E2M1 with one E4M3 scale per 16 channels, FP8 kept for the SWA KV cache, and the format nearly halves the storage footprint](/img/deepseek/papers/p6-v41-fp4kv.png)](/img/deepseek/papers/p6-v41-fp4kv.png)
>
> **What it says:** the main KV cache uses "E2M1 with one E4M3 scale per 16 channels". "We retain FP8 for the SWA KV cache". Compared with V4's FP8 main cache, this "nearly halves the storage footprint".
>
> **What it means:**
>
> 512 FP4 values take 256 bytes, and 32 scales of one byte take 32 more: 288 bytes per entry, against V4's 584. The paper also explains why one scale level is enough. The entry is RMS-normalized, so no value can exceed about $$\sqrt{512} \approx 22.6$$, and the largest value seen in training was about 10. A format that reaches 2,688 has plenty of room.
>
> The window stays in FP8 because, in the paper's words, it is sensitive to quantization. That fits the design: the window holds the exact recent text, and errors there show up right away in the next token.

Now count V4.1's bytes. Each Full layer stores a 288-byte entry plus a 68-byte indexer key, once per $$m$$ tokens:

```python
def v41():
    c = cfg('DeepSeek-V4.1-Flash')
    L = c['num_hidden_layers']
    ratios = c['compress_ratios'][:L]
    src = c['kv_source_layer_ids']
    total = 0
    for l in src:
        m = ratios[l]
        b = (V41_MAIN_FP4 + FP4_INDEX) / m
        total += b
```

```text
  V4.1-Flash: 40 layers, compress ratios [0, 1, 2], KV stored only in layers [2, 8, 14, 20]
     layer  2 (encoder, ratio 2): (288 + 68) B / 2 = 178.0 B
     layer  8 (encoder, ratio 2): (288 + 68) B / 2 = 178.0 B
     layer 14 (encoder, ratio 2): (288 + 68) B / 2 = 178.0 B
     layer 20 (decoder, ratio 1): (288 + 68) B / 1 = 356.0 B
```

Three encoder layers give $$3 \times 356 / 2 = 534$$ bytes, the decoder layer gives 356, and the total is **890 bytes**: the paper's number exactly. Notice that the compression ratios went **down**, from 4 to 2 and even 1. V4.1 stores finer detail than V4's CSA layers. It still uses a quarter of the memory, because it stores it in 4 layers instead of 41.

### Single-Pass mHC

V4.1 keeps mHC but changes one index in the formula, to save memory traffic at inference.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · §2.4.1, Eq. 6 · page 13
> [![The V4.1 paper's Single-Pass mHC: every block consumes the mixing coefficients produced by the previous one, so X_l+1 = B_l X_l + C_l F_l(A_l-1 X_l)](/img/deepseek/papers/p6-v41-single-pass.png)](/img/deepseek/papers/p6-v41-single-pass.png)
>
> **What it says:** in Single-Pass mHC "every block consumes the mixing coefficients produced by the previous one": the input uses $$A_{l-1}$$ instead of $$A_l$$. The fused Mega-mHC kernel reads and writes the residual once, "halving the activation memory traffic of our original implementation".
>
> **What it means:**
>
> The problem was a wait. To compute $$A_l$$, the kernel must read all of $$X_l$$ first. Only then can it read $$X_l$$ again to mix the input. Two reads of four streams, at every layer, for every token.
>
> Using last layer's $$A_{l-1}$$ removes the wait: the kernel can mix the input and compute the next coefficients in the same single pass. The paper reports negligible quality loss. The traffic falls from $$(4n + 4)d$$ to the lower bound $$(2n + 2)d$$ values per layer. With $$n = 4$$ that is 20d down to 10d.

### Engram: memory you look up, not compute

Some knowledge is a fixed fact about a short phrase. After "Alexander the", "Great" is very likely. After "import numpy as", "np" is almost certain. A Transformer stores such facts in its weights, and finds them by running several layers of attention and MoE. Engram asks: why compute a lookup?

> [!PAPER] Cheng et al. (2026), Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models · Figure 1 · page 3
> [![Figure 1 of the Engram paper: before attention and MoE, an Engram block hashes the 2-gram "the Great" and the 3-gram "Alexander the Great", looks up embeddings, and gates them with the hidden state](/img/deepseek/papers/p6-engram-fig1.png)](/img/deepseek/papers/p6-engram-fig1.png)
>
> **What it says:** Engram "augments the backbone by retrieving static N-gram memory and fusing it with dynamic hidden states via context-aware gating". It sits only in some layers.
>
> **What it means:**
>
> At the token "Great", Engram takes the last two tokens ("the Great") and the last three ("Alexander the Great"). A hash turns each into a row number, and the row holds a learned vector for that exact phrase.
>
> The paper frames this as a second kind of sparsity. MoE is sparse **computation**: only a few experts run. Engram is sparse **memory**: only a few table rows are read.

> [!PAPER] Cheng et al. (2026), Engram · §2.2, Eq. 1 · page 4
> [![The Engram paper's multi-head hashing; highlighted: parameterizing all possible N-grams is intractable, so K hash heads for each N-gram order map the context to a row of a prime-sized table with a lightweight multiplicative-XOR hash](/img/deepseek/papers/p6-engram-hash.png)](/img/deepseek/papers/p6-engram-hash.png)
>
> **What it says:** "Directly parameterizing the combinatorial space of all possible" N-grams is impossible, so Engram uses several "hash heads for each" N-gram order. Each head maps the N-gram to a row of a table of prime size with "a lightweight multiplicative-XOR hash".
>
> **What it means:**
>
> With about 100,000 tokens there are $$10^{10}$$ possible 2-grams and $$10^{15}$$ 3-grams. A hash folds them into a fixed-size table, and some share a row. Two N-grams may collide in one table, but rarely in all of them at once, so the joined vector still tells them apart.

Here is the hash from our toy, written like the official Engram demo code. It multiplies each token ID by a fixed random odd number, combines with XOR, and takes the remainder modulo a prime.

```python
def rows_for(seq, t, n):
    """The suffix n-gram ending at position t -> one row number per hash head."""
    gram = seq[max(0, t - n + 1): t + 1]
    gram = [0] * (n - len(gram)) + gram                                  # pad at the start of the text
    mix = 0
    for k, tok in enumerate(reversed(gram)):                             # offset 0 = current token
        mix ^= int(tok) * int(mult[k])
    return [mix % p for p in PRIMES[n]], gram
```

```text
1) tokenizer compression
   raw IDs      : [0, 1, 2, 3, 4, 5, 2, 6, 7, 8, 9, 6, 10, 1, 2, 3]   (11 distinct)
   canonical IDs: [0, 1, 2, 3, 4, 5, 2, 6, 7, 8, 2, 6, 9, 1, 2, 3]   (10 distinct; "The" and "the" now share one ID)

2) lookups for the last tokens of "... obeyed Alexander the Great"
   2-gram "the Great" -> canonical IDs [2, 3] -> rows [283, 142] in tables of size [1009, 1013]
   3-gram "Alexander the Great" -> canonical IDs [1, 2, 3] -> rows [83, 847] in tables of size [1019, 1021]
   memory vector e_t = concatenation of 4 rows x 4 = 16 numbers
   "Alexander the Great" at position 3 and at the end give the same rows: [83, 847] and [83, 847]: same memory, both times

3) 50,000 distinct 2-grams hashed into tables of 65,537 and 65,539 rows
   share a row with another 2-gram in head 1: 53.3%   in head 2: 53.7%
   share BOTH rows with the same other 2-gram:  0.000%
```

Step 1 is the tokenizer compression from the paper. IDs that differ only in case or spacing map to one canonical ID, so "The" and "the" share their memory. Step 2 shows the key property: the same phrase always lands on the same rows, at any position, with no search. Step 3 shows why there are several heads. We packed 50,000 2-grams into tables of about 65,000 rows, much tighter than a real table. More than half share a row in one head, but none share both rows with the same partner.

{{FIG:engram|Engram at one position. Hash the last two and three tokens with two heads each, read one row from each table, join the rows, gate the result with the hidden state, and add it to the stream.}}

The retrieved vector is a guess made without context, so Engram checks it against the hidden state before use. The gate is a scaled dot product passed through a sigmoid:

$$
\alpha_t = \sigma\!\left(\frac{\text{RMSNorm}(h_t)^\top \, \text{RMSNorm}(k_t)}{\sqrt{d}}\right)
$$

$$h_t$$ is the hidden state at position $$t$$, which already carries the context. $$k_t$$ is a key projected from the retrieved memory. RMSNorm scales each vector so that its numbers have an average square of 1. $$\sigma$$ squashes the result into the range 0 to 1. The output added to the stream is $$\alpha_t$$ times a value projected from the memory. Our toy with $$d = 16$$:

```text
4) gate alpha = sigmoid(RMSNorm(h) . RMSNorm(k) / sqrt(16))
   hidden state agrees with the memory (h = k)   : alpha = 0.982
   unrelated hidden state (random h)             : alpha = 0.451
   hidden state contradicts the memory (h = -k)  : alpha = 0.018
```

When the memory fits the context, it passes almost whole. When it contradicts the context, for example "Great" in "the Great Barrier Reef", the gate closes. With $$h = k$$ the dot product of the normalized vectors is exactly $$d = 16$$, so the sigmoid sees $$16 / 4 = 4$$ and gives 0.982.

### Why 196B parameters are not "active"

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · §2.4.2 · page 13
> [![The V4.1 paper's Engram configuration; highlighted: 196B Engram parameters evenly across two modules, 8 hash heads, tables of approximately 16M entries, and deterministic addressing that lets embeddings be prefetched from host memory](/img/deepseek/papers/p6-v41-engram.png)](/img/deepseek/papers/p6-v41-engram.png)
>
> **What it says:** "We allocate 196B Engram parameters evenly across two modules." Each uses N-gram orders 2, 3 and 4, "with 8 hash heads", and each head's table has "approximately 16M entries". The modules sit at layers 1 and 14. "Deterministic addressing enables embeddings to be prefetched from host" memory.
>
> **What it means:**
>
> The config lets us check the 196B. It lists the rows of the two modules, 384,006,168 and 384,016,682, and a head width of 256. Multiply and you get 196.6B parameters. Each module has $$3 \times 8 = 24$$ tables, so each table averages 16,000,476 rows: the "approximately 16M".
>
> The last sentence is the important one. Engram's row numbers depend only on the input tokens, which are known before the first layer runs. So the server can fetch the needed rows from CPU memory while the GPU is busy with layer 0. A learned MoE router cannot do this: which expert a token needs is known only when that layer's router has run.

```text
Engram (V4.1-Flash): layers [1, 14], 3 n-gram orders x 8 hash heads per module
  table rows  = 384,006,168 + 384,016,682 = 768,022,850
  parameters  = 768,022,850 rows x 256 dims = 196.6B   (paper: "196B Engram parameters")
  per table   = 16,000,476 rows on average   (paper: "approximately 16M entries")
  read per token = 48 rows x 256 = 12,288 numbers = 12,288 bytes in FP8 (0.0000062% of the tables)
```

Here is why "active" is the wrong word. Per token, Engram reads 48 rows: 2 modules x 3 orders x 8 heads. That is 12,288 numbers, about 12 KB in FP8, out of 196.6B. A dense layer with 196.6B weights would need about 393 GFLOPs per token. So DeepSeek counts the 196B separately from both the 552B backbone and the 8B or 16B active.

Does a lookup really cost the same in a huge table? On our CPU, gathering 8,192 rows took the same time from a table of a thousand rows as from one of four million:

```text
5) gather 8,192 rows of 64 FP16 numbers (CPU, best of 20 runs)
   table of     1,000 rows (    0.1 MB): 0.113 ms
   table of   100,000 rows (   12.8 MB): 0.118 ms
   table of 4,000,000 rows (  512.0 MB): 0.080 ms
```

The small differences are timing noise on a laptop CPU, not a trend. The table's size costs memory, not time, and that memory can be cheap CPU memory. The Engram paper reports that a 100B-parameter table in host memory costs "negligible overhead (< 3%)". Its own test model, Engram-27B, swapped part of its experts for a 5.7B-parameter table at equal size and compute. It reports MMLU +3.4, BBH +5.0 and long-context retrieval (Multi-Query NIAH) from 84.2 to 97.0.

### Multimodality

V4.1-Flash is DeepSeek's first main model trained on images from the start. A vision encoder trained from scratch, DeepSeek-ViT (32 layers, 14-pixel patches), turns an image into features. A 3 x 3 "pixel-unshuffle" folds each 3 x 3 neighbourhood into one token, so an image costs nine times fewer tokens. The pre-training corpus is 45T tokens, 7 parts text-only to 1 part multimodal. Image and text tokens also keep separate load-balancing biases for the experts ([Part 2](deepseek-2-the-v3-architecture.md)), because they prefer different experts.

## The headline chart

Now every number in DeepSeek's own chart has an explanation, and a line of arithmetic behind it.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash · Figure 1 · page 1
> [![Figure 1 of the V4.1 paper: agentic benchmark bars on the left; on the right, global KV cache per token in bytes for DeepSeek-V1 389,120, V3.2 48,068, V4-Flash 3,514 and V4.1-Flash 890, with 8.1x, 13.7x and 3.9x smaller steps](/img/deepseek/papers/p6-v41-fig1.png)](/img/deepseek/papers/p6-v41-fig1.png)
>
> **What it says:** global KV cache per token: "DeepSeek-V1" 389,120 bytes (2023.11), V3.2 48,068, V4-Flash 3,514, V4.1-Flash 890; overall "437-fold reductions in per-token global KV cache size" relative to V1.
>
> **What it means:**
>
> "DeepSeek-V1" here is the 67B model of November 2023. The word "global" is precise: the chart counts the cache that grows with length and leaves out the fixed sliding-window cache, as our script does.
>
> Each step used a different lever. 67B to V3.2 changed **what** each layer stores (MLA's shared latent, then FP8). V3.2 to V4-Flash changed **how many** tokens get an entry (compression). V4-Flash to V4.1-Flash changed **how many layers** store one, and the bits per number. The V4.1 paper calls these the entry size, the sequence and the layer dimension.

{{FIG:kvgen|Our script's reconstruction of the chart, with V3 added. Every bar comes from config.json and the published cache layouts; four of the five match the paper's figure to the byte.}}

```text
model                  date      bytes/token   paper Fig 1b   x smaller   GB at 128K   GB at 1M
DeepSeek LLM 67B       2023-11      389,120        389,120                   51.00     408.02
DeepSeek-V3            2024-12       70,272              -        5.5x        9.21      73.69
DeepSeek-V3.2          2025-12       48,068         48,068        1.5x        6.30      50.40
DeepSeek-V4-Flash      2026-04        3,514          3,514       13.7x        0.46       3.68
DeepSeek-V4.1-Flash    2026-09          890            890        3.9x        0.12       0.93

V4-Pro (not in the chart): 5,031 B per token; V3.2 / V4-Pro = 9.55x, V3.2 / V4-Flash = 13.68x  (V4 paper Figure 1: "9.5x smaller", "13.7x smaller")
67B / V3.2 = 8.1x  (V4.1 paper Figure 1b: "8.1x smaller")
67B / V4.1-Flash = 437.2x  (V4.1 paper: "437-fold");  V4-Flash / V4.1-Flash = 3.95x  ("approximately 4-fold")
```

V3 is not in DeepSeek's chart; we added it because it is the model most readers know. At one million tokens, 67B would need 408 GB per conversation, V3 73.7 GB, and V4.1-Flash 0.93 GB.

> [!TIP] In production
> A small cache changes what a GPU can hold. At 890 bytes per token, 100 users with 128K-token contexts need about 12 GB of global cache, plus a fixed window cache per user.
>
> DeepSeek also keeps caches on SSD, so a repeated prefix is read back instead of computed again. V4.1's SSD cache is about 1/8 of V4-Flash's. [Part 7](deepseek-7-running-deepseek.md) does the full memory arithmetic for running these models.

<details><summary>Research notes: assumptions and what we could not check</summary>

- The main entry layouts (656 B for V3.2, 584 B for V4, 288 B for V4.1 FP4) are from the FlashMLA README at commit ba89a34. The 68-byte indexer key is our reading of MXFP4 (128 four-bit values plus one 8-bit scale per 32). The V4 reference code uses an FP4 block size of 32 (`fp4_block_size = 32`). This assumption reproduces the published 3,514 and 890 exactly, which is strong but indirect evidence.
- V4-Pro's 5,031 bytes per token is our number; DeepSeek does not print it. It reproduces the paper's "9.5x smaller".
- The sliding-window estimate (3.2 MB per request for V4-Flash) assumes the window uses the same 584-byte layout as the compressed entries.
- The arXiv identifier of the V4 paper starts with 2606, but arXiv shows "Submitted on 26 Apr 2026".
- DeepSeek-V4-Flash-Vision-Exp exists on Hugging Face; we did not read its config, and there is no paper for it.
- The V4.1-Flash release note on the API site is dated 10 September 2026; the arXiv paper is dated 17 September 2026.
- Our mHC toy uses random, not learned, matrices; our Engram toy uses tiny tables. Both show mechanisms, not model behaviour.

</details>

> [!TAKEAWAYS]
> - V4 (April 2026): V4-Pro 1.6T total / 49B active, V4-Flash 284B / 13B, both 1M context. V4.1-Flash (September 2026): 552B backbone plus 196B Engram, 8B active in prefill and 16B in decode.
> - Global KV cache per token, rebuilt from configs: 389,120 B (67B), 70,272 (V3), 48,068 (V3.2), 3,514 (V4-Flash), 890 (V4.1-Flash). At 1M tokens: 408 GB down to 0.93 GB.
> - CSA merges every 4 tokens into one entry and reads only the top 512 or 1,024; HCA merges every 128 and reads all. Layers alternate, and every layer also reads a 128-token window.
> - mHC widens the residual stream to 4 copies and projects the mixing matrix onto doubly stochastic matrices with 20 Sinkhorn-Knopp rounds. Products of such matrices cannot amplify, so the gain stays near 1 instead of reaching 3,000.
> - mHC costs 6.7% of training time in V4. V4.1's Single-Pass mHC halves its memory traffic at inference.
> - Routed experts are stored in FP4 (MXFP4), trained with quantization-aware training; V4-Flash's files are 159.6 GB instead of 568 GB in BF16.
> - DSpark drafts 5 tokens per pass with a Markov head and a confidence head; DeepSeek measured 60 to 85% faster per-user generation than MTP-1 at the same throughput.
> - V4.1 stores KV in only 4 of 40 layers (CSA2 Full mode), in FP4. Its decoder takes its long-range KV from the encoder's output, which halves prefill work.
> - Engram is in V4.1 only, not in V4. It reads 48 hashed table rows per token out of 196.6B parameters. So its parameters are stored but not "active", and they can live in CPU memory.

## Next

[Part 7](deepseek-7-running-deepseek.md) turns all these numbers into hardware. It counts the GPUs each model needs, and what fits on one machine or on a CPU with lots of RAM. It also covers the quantized files, and how vLLM, SGLang, llama.cpp and DeepSeek's own serving system run them.

## References

- DeepSeek-AI (2026). DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence. [arXiv:2606.19348](https://arxiv.org/abs/2606.19348)
- DeepSeek-AI (2026). DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression. [arXiv:2609.19969](https://arxiv.org/abs/2609.19969)
- Xie et al. (2025). mHC: Manifold-Constrained Hyper-Connections. [arXiv:2512.24880](https://arxiv.org/abs/2512.24880)
- Cheng et al. (2026). Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram). [arXiv:2601.07372](https://arxiv.org/abs/2601.07372); code: [github.com/deepseek-ai/Engram](https://github.com/deepseek-ai/Engram)
- Cheng et al. (2026). DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. [arXiv:2607.05147](https://arxiv.org/abs/2607.05147)
- Model configs and cards: [DeepSeek-V4-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash), [DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro), [DeepSeek-V4-Pro-0813](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813), [DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)
- KV cache byte layouts: [FlashMLA README (commit ba89a34)](https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7)
- Our scripts: `part6_kv_per_token.py`, `part6_mhc_toy.py`, `part6_engram_toy.py` in `code/deepseek/`
