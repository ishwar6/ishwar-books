---
title: "Attention, Part 2: MQA, GQA and MLA"
description: "Every token leaves its keys and values behind in memory (the KV cache), and plain multi-head attention leaves a lot. How multi-query, grouped-query and multi-head latent attention shrink it, explained from zero, built from scratch, checked against a real model's layer, and measured: DeepSeek-V3 stores 68.6 KiB per token where plain attention would need 4,880."
date: 2026-10-04
tags: [attention, kv-cache, llm]
series: "Attention, From the Ground Up"
series_part: 2
motif: kv
accent: "#5fd4b0"
---

[Part 1](attention-1-self-attention.md) built attention from scratch and ended on its cost: memory. This part is about three ideas that cut that memory, sometimes by more than 60 times, and which ones today's models use.

As before, every technical word gets a yellow box the first time it appears.

> [!TIP] Quick recap of Part 1
> Every token makes a **query** (what am I looking for?), a **key** (what do I contain?) and a **value** (what do I hand over?). A token compares its query with the keys of all earlier tokens, turns the scores into weights with softmax, and takes a weighted mix of their values. A model runs several of these side by side, called **heads**.

## The memory problem: the KV cache

A chatbot writes its answer **one token at a time**. To write each new token, it runs attention: the new token's query is compared with the keys of every earlier token, and their values are mixed.

Those earlier keys and values never change. So instead of recomputing them at every step, the model computes them once and keeps them in memory.

> [!DEFINITION] KV cache
> The stored **keys (K)** and **values (V)** of every token so far. Each new token adds its own key and value, so the cache grows by one token at every step. It lives in the GPU's memory for as long as the conversation runs.

{{FIG:p2_cache|How the KV cache works. Every earlier token has left a key and a value in memory. The newest token adds its own, and its query is compared with every stored key.}}

Notice what is **not** in the cache: the query. A query belongs to the token being written right now. It is used once, for this one step, and thrown away. Only keys and values are kept.

That is the opening for this whole part: **a model can keep many query heads while storing far fewer key and value heads**.

### How big is it?

With plain multi-head attention (MHA), every head in every layer stores one key and one value for every token:

$$
\text{KV cache per token} = 2 \times L \times H \times d_h \times b
$$

where:

- the $$2$$ counts one key and one value;
- $$L$$ is the number of layers;
- $$H$$ is the number of key/value heads in each layer (in plain MHA, the same as the number of query heads);
- $$d_h$$ is the length of each head's key and value vectors;
- $$b$$ is the bytes per number (2 for the usual 16-bit numbers).

For Llama 2 7B, $$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$$ bytes: **512 KiB for every single token**. A 4,096-token conversation needs 2 GiB, and a GPU serving 30 conversations at once needs 60 GiB just for caches.

> [!DEFINITION] KiB, MiB and GiB
> Units of memory. 1 KiB = 1,024 bytes, 1 MiB = 1,024 KiB, 1 GiB = 1,024 MiB. One 16-bit number takes 2 bytes.

### Why reading the cache is the real cost

When a model writes one token, the arithmetic is small, but it must **read** all its weights and the whole KV cache from GPU memory. Reading memory has a speed limit, called **memory bandwidth**, and for writing it is usually the limit that matters:

$$
t_{\text{step}} \;\gtrsim\; \frac{\text{bytes of weights} + \text{bytes of KV cache}}{\text{memory bandwidth}}
$$

where $$t_{\text{step}}$$ is the time to write one token, and memory bandwidth is how many bytes per second the GPU can read (about 2,000 to 3,000 GB per second on a modern data-centre GPU).

> [!DEFINITION] Memory bandwidth
> How fast a chip can move data from its memory to its calculators, in bytes per second. While writing text, a GPU spends most of its time waiting for data to arrive, not calculating, so this number, not raw calculating speed, usually decides how fast it writes.

Every byte the cache saves is a byte that does not need to be read, at every step, for every conversation on that GPU. That is why the three ideas below matter so much.

The cache, more than the arithmetic, limits how many people a GPU can serve and how long their conversations can be. (The [LLM inference series](llm-inference-2-kv-cache.md) goes deeper.) Look at the formula again: the only part we can shrink without changing the model's size much is $$H$$, the number of key/value heads, or what each head stores. That is exactly what the three ideas do.

{{FIG:p2_sharing|Four ways to organise the heads. Orange dots are query heads: they are computed fresh and never stored. Aqua boxes are key/value heads: these are what each token leaves in the cache. MLA stores one compressed vector (blue) instead.}}

## Multi-query attention (MQA): one key/value head for everyone

In 2019, Noam Shazeer proposed the most extreme answer: keep all the query heads, but give the whole layer **just one key head and one value head**, shared by every query head.

> [!PAPER] Shazeer (2019), Fast Transformer Decoding: One Write-Head is All You Need · Abstract · page 1
> [![The abstract of the multi-query attention paper, highlighting that incremental inference is slow because of memory bandwidth, and the proposal to share keys and values across heads](/img/attention/papers/shazeer-abstract.png)](/img/attention/papers/shazeer-abstract.png)
>
> **Context:** A short, single-author paper from Google, written two years after the transformer, when people started using transformers to *generate* text one token at a time.
>
> **What it says:** Training is fast because it runs in parallel, but writing ("incremental inference") is slow "due to the memory-bandwidth cost of repeatedly loading the large keys and values tensors". The fix: share one set of keys and values across all heads.
>
> **Why it matters:** This is the first paper to say clearly that **memory reads, not arithmetic**, are the bottleneck of text generation. That insight drives every idea in Parts 2 to 4.
>
> [Read the paper on arXiv](https://arxiv.org/abs/1911.02150)

> [!PAPER] Shazeer (2019) · Section 3, Multi-Query Attention · page 5
> [![The definition of multi-query attention: identical to multi-head attention except that the heads share a single set of keys and values](/img/attention/papers/shazeer-definition.png)](/img/attention/papers/shazeer-definition.png)
>
> **Context:** The definition, right after the paper walks through code for ordinary multi-head attention.
>
> **What it says:** Multi-query attention "is identical except that the different heads share a single set of keys and values". In the code, you simply delete the head letter "h" from the key and value shapes.
>
> **Why it matters:** One deleted letter, and the cache shrinks by the number of heads. Few ideas in deep learning have such a large effect for such a small change.

In the formula, $$H$$ becomes $$1$$:

$$
\text{MQA cache per token} = 2 \times L \times 1 \times d_h \times b
$$

So the cache shrinks by a factor equal to the number of heads.

Falcon-7B is a real example: 71 query heads sharing one key/value head. It stores **8 KiB per token**, against 512 KiB for Llama 2 7B.

The price is quality. Every query head must find what it needs in the very same keys and values, and models trained this way tend to be measurably worse. The DeepSeek-V2 paper says it plainly: MQA and GQA "require a smaller magnitude of KV cache, but their performance does not match MHA."

The original paper measured both sides of that trade, on English-to-German translation:

> [!PAPER] Shazeer (2019) · Section 4, Experiments · page 8, Tables 1 and 2
> [![Table 1 of the multi-query paper: translation quality of multi-head and multi-query models, with the multi-query row highlighted](/img/attention/papers/shazeer-table1.png)](/img/attention/papers/shazeer-table1.png)
> [![Table 2 of the multi-query paper: training and inference costs in microseconds per output token, with the multi-query row highlighted](/img/attention/papers/shazeer-table2.png)](/img/attention/papers/shazeer-table2.png)
>
> **Context:** Table 1 is quality (lower ln(PPL) and higher BLEU are better). Table 2 is cost, in TPU-microseconds per output token. "enc. + dec." splits the time between reading the input (encoder) and writing the output (decoder).
>
> **What it says:** Quality barely moves: BLEU 26.5 against 26.7 on the dev set. But the decoder's time per token drops from **46 to 3.8 microseconds**, about **12 times faster**, and beam search from 203 to 32.
>
> **Why it matters:** A 12× faster decoder for a 0.2 BLEU loss is the trade that made MQA, and then GQA, standard. Notice that **training** time is unchanged (13.2 against 13.0): the saving is only in writing, exactly where the cache is read again and again.

## Grouped-query attention (GQA): the compromise everyone adopted

In 2023, Ainslie et al. proposed a middle ground: split the query heads into **groups**, and give each group its own key/value head.

> [!DEFINITION] Group
> A set of query heads that share one key head and one value head. With 32 query heads and 8 key/value heads, there are 8 groups of 4 query heads each.

$$
\text{GQA cache per token} = 2 \times L \times H_{kv} \times d_h \times b, \qquad \text{group size } g = \frac{H_q}{H_{kv}}
$$

where $$H_q$$ is the number of query heads and $$H_{kv}$$ the number of key/value heads.

GQA is a **dial** between the two extremes:

- With $$H_{kv} = H_q$$ (every query head has its own), GQA *is* plain multi-head attention.
- With $$H_{kv} = 1$$ (one for everyone), GQA *is* multi-query attention.

The paper showed that GQA gets "quality close to multi-head attention with comparable speed to MQA". It also showed that an existing MHA model can be converted to GQA: average the key and value heads inside each group, then train briefly (about 5% of the original training) to recover.

> [!DEFINITION] Uptraining
> Taking an already-trained model, changing part of its design, and training it a little more so it adapts. Much cheaper than training from scratch: the GQA paper used 5% of the original training compute.

As an equation, converting one group of $$g$$ key heads into a single shared key head is just an average of their weight matrices:

$$
W_k^{\text{group}} = \frac{1}{g} \sum_{i \in \text{group}} W_k^{(i)}, \qquad W_v^{\text{group}} = \frac{1}{g} \sum_{i \in \text{group}} W_v^{(i)}
$$

where $$W_k^{(i)}$$ and $$W_v^{(i)}$$ are the key and value matrices of head $$i$$ in the original model.

> [!PAPER] Ainslie et al. (2023), GQA · Abstract · page 1
> [![The abstract of the GQA paper, highlighting uptraining with 5% of the pre-training compute and the definition of grouped-query attention](/img/attention/papers/gqa-abstract.png)](/img/attention/papers/gqa-abstract.png)
>
> **Context:** A 2023 paper from Google Research. The two contributions are listed as (1) and (2).
>
> **What it says:** (1) An existing multi-head model can be "uptrained" into MQA using 5% of its original pre-training compute. (2) Grouped-query attention, "a generalization of multi-query attention which uses an intermediate (more than one, less than number of query heads) number of key-value heads".
>
> **Why it matters:** Point (1) meant companies did not have to retrain their models from scratch to get the speed-up. Point (2) gave them a dial instead of an on/off switch.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2305.13245)

> [!PAPER] Ainslie et al. (2023) · Section 2.2 · page 2, Figure 2
> [![Figure 2 of the GQA paper: diagrams of multi-head, grouped-query and multi-query attention showing how query heads share key and value heads](/img/attention/papers/gqa-figure2.png)](/img/attention/papers/gqa-figure2.png)
>
> **Context:** The paper's picture of the three designs, with 8 query heads each.
>
> **What it says:** Multi-head: 8 keys and 8 values. Grouped-query: 4 of each, shared by pairs of queries. Multi-query: 1 of each, shared by all 8. In the caption's words, GQA is "interpolating between multi-head and multi-query attention".
>
> **Why it matters:** Compare it with this article's own figure above: it is the same picture. Only the number of orange and red boxes (keys and values) changes the cache size; the blue boxes (queries) cost no memory.

> [!PAPER] Ainslie et al. (2023) · Section 3.2, Main results · page 3, Figure 3
> [![Figure 3 of the GQA paper: performance against inference time for MHA-Large, MHA-XXL, MQA-XXL and GQA-XXL](/img/attention/papers/gqa-figure3.png)](/img/attention/papers/gqa-figure3.png)
>
> **Context:** Average score over many tasks (summaries, translation, question answering) against time per sample, for T5 models. Up and to the left is better.
>
> **What it says:** The big multi-head model (MHA-XXL) scores 47.2 but takes 1.51 ms. GQA-XXL scores **47.1 in 0.28 ms**: almost the same quality, more than **5 times faster**. MQA-XXL is slightly faster still (0.24 ms) but loses more quality (46.6).
>
> **Why it matters:** This plot is why GQA "won". It sits almost exactly at MHA's quality and MQA's speed, the best of both.

It became the default. Llama 3 (8 key/value heads), Qwen2.5 (Qwen2.5-0.5B: 14 query heads, 2 key/value heads), Gemma 3 (27B: 32 query heads, 16 key/value heads) and Mistral all use it.

### GQA from scratch

One way to build GQA is to copy each key/value head once for every query head in its group, then run normal attention. That works, but making those copies wastes the memory we just saved. The better way is to **group the query heads** instead, so the small cache is read as it is:

> [!DEFINITION] einsum
> A compact way to write matrix multiplications in PyTorch. `torch.einsum("bhgqd,bhkd->bhgqk", a, b)` means: multiply `a` and `b` and add up over the letter that disappears (`d`, the vector length). It is just "dot product of every query with every key", with the head and group bookkeeping spelled out by letters.

```python
def gqa(q, k, v, causal=True):
    """q: (B, Hq, T, d); k, v: (B, Hkv, T, d) with Hq a multiple of Hkv.
    Each group of Hq/Hkv query heads shares one key/value head. No copy of K or V is made."""
    B, Hq, T, d = q.shape
    Hkv = k.shape[1]
    g = Hq // Hkv
    qg = q.view(B, Hkv, g, T, d)                                       # group the query heads
    scores = torch.einsum('bhgqd,bhkd->bhgqk', qg, k) / math.sqrt(d)
    if causal:
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=q.device), 1)
        scores = scores.masked_fill(mask, float('-inf'))
    out = torch.einsum('bhgqk,bhkd->bhgqd', torch.softmax(scores, -1), v)
    return out.reshape(B, Hq, T, d)
```

The shapes read like this: `B` is how many texts at once, `T` the number of tokens, `d` the head length, and `Hq` and `Hkv` the numbers of query and key/value heads.

**Proof against PyTorch.** PyTorch's built-in attention function needs the copying approach, so I gave it copied keys and values and compared it with the function above, for all three cases:

```text
MHA (8 KV heads)   vs PyTorch with repeated K/V: max |diff| = 4.4e-16
GQA (2 KV heads)   vs PyTorch with repeated K/V: max |diff| = 4.4e-16
MQA (1 KV head)    vs PyTorch with repeated K/V: max |diff| = 6.7e-16
```

Differences around $$10^{-16}$$ are the smallest rounding errors a computer can make here: same answers.

### Proof on a real model

A toy test only proves the toy. So I opened **Qwen2.5-0.5B**, a real model that uses GQA with 14 query heads sharing 2 key/value heads. I captured the input to its first attention layer and recomputed that layer **by hand, from the model's own weights**:

1. the query, key and value matrices (including their small added constants, called biases);
2. the rotary position embedding, **RoPE** (explained below);
3. the grouped attention function above;
4. the output matrix.

Then I compared my result with what the model itself produced:

```text
Qwen2.5-0.5B layer 0 (14 query heads share 2 KV heads): our GQA vs the model's own output,
max |diff| = 6.6e-07 (outputs are up to 0.5)
```

$$6.6 \times 10^{-7}$$ is the rounding noise of 32-bit numbers. So the real layer computes exactly what those 13 lines compute.

### Does a smaller cache make writing faster?

At every step, the model reads the whole KV cache. Fewer key/value heads means less to read, so each step should be faster. I measured one attention step for 8 conversations at once, each with 4,096 tokens of history and 32 query heads, on an Apple M5 Pro GPU:

{{FIG:p2_speed|Time for one attention step over a 4,096-token cache. Fewer key/value heads means less to read and a faster step, but not in proportion.}}

| Key/value heads | Cache read per step | Time per attention step |
|---|---|---|
| 32 (MHA) | 537 MB | 3.08 ms |
| 8 (GQA) | 134 MB | 1.85 ms |
| 1 (MQA) | 17 MB | 1.24 ms |

Faster, clearly, but not 4 or 32 times faster. Each step has a fixed cost (starting the work on the GPU) that does not shrink, and once the cache is small that fixed cost dominates.

The bigger win is **memory**: a cache 4 times smaller fits 4 times more conversations, or 4 times more history, on the same GPU.

## Multi-head latent attention (MLA): compress instead of share

GQA saves memory by **throwing information away**: query heads in the same group are forced to use identical keys and values. DeepSeek-V2 (2024) tried something different: what if each token stored a **compressed** version of its keys and values, from which every head's own keys and values can be rebuilt?

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2 · Section 2.1 · page 7, Figure 3
> [![Figure 3 of the DeepSeek-V2 paper: MHA, GQA, MQA and MLA side by side, with the parts cached during inference shaded](/img/attention/papers/deepseek-figure3.png)](/img/attention/papers/deepseek-figure3.png)
>
> **Context:** The paper's comparison of all four designs in this part. Shaded boxes are what is **cached during inference**.
>
> **What it says:** MHA caches every key and value; GQA caches a few; MQA caches one. MLA caches only a single small "compressed latent KV" (far right) and rebuilds the keys and values from it by a projection.
>
> **Why it matters:** This one figure is the whole of Part 2. The question every design answers is: how little can you cache and still let every query head see what it needs?
>
> [Read the paper on arXiv](https://arxiv.org/abs/2405.04434)

> [!DEFINITION] Latent vector (compression)
> A short vector that holds the important information of a longer one, like a zip file. "Latent" means hidden: the model learns what to keep. MLA squeezes all of a token's keys and values (thousands of numbers) into one latent of 512 numbers in DeepSeek-V3.

{{FIG:p2_mla|Multi-head latent attention. Each token is compressed into a small latent vector c, plus a small key that carries position (RoPE). Only those two are cached. Every head's keys and values can be rebuilt from c.}}

### The MLA equations

For each token with vector $$x$$, MLA computes two small things and caches only them:

$$
c = x\,W_{dkv} \qquad\qquad k^{R} = \operatorname{RoPE}(x\,W_{kr})
$$

where:

- $$c$$ is the **latent**: $$W_{dkv}$$ ("down-projection for keys and values") shrinks the token to $$d_c$$ numbers (512 in DeepSeek-V3);
- $$k^R$$ is a small **position key** of $$d_r$$ numbers (64 in DeepSeek-V3), shared by all heads.

> [!DEFINITION] Projection
> Multiplying a vector by a learned matrix to get a new vector, usually of a different length. A **down-projection** makes it shorter (compresses); an **up-projection** makes it longer (expands).

When a head $$h$$ needs its keys and values, it rebuilds them from the latent with two up-projections:

$$
k_h = c\,W_{uk}^{(h)} \qquad\qquad v_h = c\,W_{uv}^{(h)}
$$

Here are the same three equations in the paper, numbered (9), (10) and (11):

> [!PAPER] DeepSeek-V2 · Section 2.1.2, Low-Rank Key-Value Joint Compression · page 7
> [![Equations 9 to 11 of the DeepSeek-V2 paper: the down-projection to a compressed latent and the up-projections to keys and values](/img/attention/papers/deepseek-mla-equations.png)](/img/attention/papers/deepseek-mla-equations.png)
>
> **Context:** The core definition of MLA. The paper writes vectors as columns ($$W h_t$$) where this article writes rows ($$x W$$); it is the same thing.
>
> **What it says:** $$c_t^{KV} = W^{DKV} h_t$$ compresses the token into a latent of size $$d_c$$ ("much smaller than" all heads' keys together, $$d_h n_h$$). $$W^{UK}$$ and $$W^{UV}$$ rebuild the keys and values. "During inference, MLA only needs to cache $$c_t^{KV}$$".
>
> **Why it matters:** "Low-rank" is the key phrase: the paper bets that all the heads' keys and values together contain much less real information than their size suggests, so a short latent can hold it. DeepSeek's results suggest the bet was right.

> [!DEFINITION] Low-rank
> A big matrix is **low-rank** if it can be written as the product of two thin matrices, $$W \approx A\,B$$. MLA's key matrix is effectively $$W_{dkv} W_{uk}$$: a squeeze down to $$d_c$$ numbers, then an expansion back up. If the information really fits in $$d_c$$ numbers, nothing is lost.

So the cache per token is just:

$$
\text{MLA cache per token} = L \times (d_c + d_r) \times b
$$

For DeepSeek-V3: $$61 \times (512 + 64) \times 2 = 70{,}272$$ bytes, or **68.6 KiB**.

(DeepSeek also compresses the *queries* through a small latent, to save memory during training. Queries are never cached, so that part does not change the KV cache, and my implementation leaves it out.)

### The trick that makes it fast: never rebuild at all

Rebuilding every head's keys and values at every step would cost a lot of extra work. MLA avoids it with a small piece of algebra called **absorption**.

> [!DEFINITION] Absorption
> Two matrix multiplications in a row can be merged into one. If the key is always built as $$c\,W_{uk}$$, we can move $$W_{uk}$$ over to the query side instead, multiply it in once, and then compare the query with the small latent $$c$$ directly.

A head's attention score is its query dotted with a key. Put in the rebuilt key and rearrange:

$$
q_h \cdot k_h \;=\; q_h \cdot \big(c\,W_{uk}^{(h)}\big) \;=\; \big(q_h\,W_{uk}^{(h)\top}\big) \cdot c
$$

The left side needs every cached token's key to be rebuilt. The right side multiplies the **one** new query by $$W_{uk}^{(h)\top}$$ once, then compares it with the cached latents directly. Same number, much less work.

The value side works the same way. The head's output is a weighted sum of values, with weights $$a_j$$ from softmax:

$$
\sum_j a_j\, v_{h,j} \;=\; \sum_j a_j\, c_j W_{uv}^{(h)} \;=\; \Big(\sum_j a_j\, c_j\Big)\, W_{uv}^{(h)}
$$

So the weighted sum is taken over the small latents, and $$W_{uv}^{(h)}$$ is applied just once at the end. **The full keys and values are never built.**

Here is that inference path from my implementation:

```python
def absorbed(self, x):
    """Inference path: cache only c (d_c) and the RoPE key (d_r) per token; never rebuild K or V."""
    T = x.shape[0]; pos = torch.arange(T)
    c, kr = self.W_dkv(x), rope(self.W_kr(x), pos)                     # <- the entire KV cache
    W_uk = self.W_uk.weight.view(self.h, self.dn, self.dc)
    W_uv = self.W_uv.weight.view(self.h, self.dv, self.dc)
    q_nope = self.W_q(x).view(T, self.h, self.dn)
    q_lat = torch.einsum('qhd,hdc->qhc', q_nope, W_uk)                  # absorb W_uk into the query
    q_rope = torch.stack([rope(t, pos) for t in self.W_qr(x).view(T, self.h, self.dr).unbind(1)], 1)
    scores = (torch.einsum('qhc,kc->hqk', q_lat, c) + torch.einsum('qhd,kd->hqk', q_rope, kr)) / math.sqrt(self.dn + self.dr)
    mask = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
    A = torch.softmax(scores.masked_fill(mask, float('-inf')), -1)
    o_lat = torch.einsum('hqk,kc->qhc', A, c)                            # attend in latent space
    o = torch.einsum('qhc,hvc->qhv', o_lat, W_uv)                        # then up-project once
    return self.W_o(o.reshape(T, -1))
```

**Proof.** I compared it with the slow path that rebuilds every key and value, using the same weights:

```text
MLA: full path vs compressed-cache path, max |diff| = 3.9e-16
numbers cached per token per layer: full K and V 640, latent + RoPE key 80 (8.0x smaller)
```

Identical answers, with an eighth of the cache in this small example.

### Why position needs its own key

You may wonder why MLA has that separate little position key. The reason is RoPE.

> [!DEFINITION] RoPE (rotary position embedding)
> Attention by itself does not know the order of words. RoPE adds order by **turning** each query and key vector by an angle that grows with the token's position. When two turned vectors are compared with a dot product, the result depends on **how far apart** the two tokens are. Most modern models use it.

{{FIG:p2_rope|RoPE in a picture. The same vector is turned a little more at each position. Comparing two turned vectors tells the model how far apart they are.}}

For one pair of numbers $$(x_1, x_2)$$ inside a vector at position $$m$$, RoPE does:

$$
\begin{pmatrix} x_1' \\ x_2' \end{pmatrix} = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \end{pmatrix}
$$

where $$m\theta$$ is the angle: $$\theta$$ is a fixed small angle (different for each pair of numbers), and $$m$$ is the position.

Here is the problem. If RoPE were applied to the rebuilt keys $$k_h = c\,W_{uk}^{(h)}$$, a rotation would sit **between** $$W_{uk}^{(h)}$$ and the query, and that rotation is different for every position. Then there is no single matrix to move over to the query side, and the absorption trick breaks.

I tested exactly that mistake: apply RoPE to the rebuilt keys and the queries, then compare with the compressed path:

```text
with RoPE applied to the reconstructed keys, the compressed path is wrong by up to 0.03
(outputs up to 0.53)
```

An error of about 6%, from one misplaced rotation.

The DeepSeek-V2 paper found exactly this problem and states it in one sentence:

> [!PAPER] DeepSeek-V2 · Section 2.1.3, Decoupled Rotary Position Embedding · page 8
> [![The DeepSeek-V2 paragraph explaining that RoPE is incompatible with low-rank KV compression and proposing the decoupled RoPE strategy](/img/attention/papers/deepseek-rope.png)](/img/attention/papers/deepseek-rope.png)
>
> **Context:** DeepSeek wanted to keep RoPE, which almost every model uses, but it collides with MLA's trick.
>
> **What it says:** "RoPE is incompatible with low-rank KV compression": with RoPE on the keys, $$W^{UK}$$ "cannot be absorbed into $$W^Q$$ any more", because a position-dependent RoPE matrix "will lie between $$W^Q$$ and $$W^{UK}$$", and matrix multiplication "does not obey a commutative law". The fix: "the decoupled RoPE strategy", extra small queries and one shared key that carry RoPE alone.
>
> **Why it matters:** My test above (6% error when RoPE is put on the rebuilt keys) is this paragraph, measured. It is a nice example of a design forced by algebra: the separate RoPE key exists only because $$AB \ne BA$$ for matrices.

> [!DEFINITION] Commutative
> An operation is commutative if order does not matter: $$3 \times 5 = 5 \times 3$$. Matrix multiplication is **not**: in general $$AB \ne BA$$. So a rotation stuck between two matrices cannot just be moved out of the way.

RoPE itself comes from the RoFormer paper, which explains it with this picture:

> [!PAPER] Su et al. (2021), RoFormer · Section 3.2 · page 5, Figure 1
> [![Figure 1 of the RoFormer paper: each pair of numbers in a query or key is rotated by an angle m theta that depends on the token position m](/img/attention/papers/roformer-figure1.png)](/img/attention/papers/roformer-figure1.png)
>
> **Context:** The paper that introduced rotary position embedding (RoPE).
>
> **What it says:** Top: a pair of numbers $$(x_1, x_2)$$ is rotated by the angle $$m\theta_1$$, where $$m$$ is the position. Bottom: a whole query or key is split into pairs, each pair with its own angle $$\theta_1, \theta_2, \dots$$, and each token (Enhanced, Transformer, with, ...) is rotated according to its position 1, 2, 3, ...
>
> **Why it matters:** Because the rotation depends on the position, it cannot be folded into fixed matrices, which is exactly the clash with MLA described above.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2104.09864)

DeepSeek's fix is to **split the job**. The latent carries the **content** and is never rotated. A small separate key carries the **position**, and only it gets RoPE. The score simply adds the two parts:

$$
\text{score}_{h,j} = \frac{\big(q_h W_{uk}^{(h)\top}\big)\cdot c_j \;+\; q^R_h \cdot k^R_j}{\sqrt{d_n + d_r}}
$$

where $$q^R_h$$ is the head's small position query, $$k^R_j$$ is token $$j$$'s cached position key, and $$d_n$$ and $$d_r$$ are the lengths of the content and position parts. (This is exactly the `scores = ...` line in the code above.)

## How big is the difference?

Here are four real models, using each one's published settings and 2 bytes per number:

{{FIG:p2_kv|KV cache per token for four real models. Llama 2 7B uses plain multi-head attention; Llama 3.1 8B uses GQA; DeepSeek-V3 uses MLA; Falcon 7B uses MQA.}}

| Model | Attention | Formula | Cached per token |
|---|---|---|---|
| Llama 2 7B | MHA | 2 × 32 layers × 32 heads × 128 × 2 B | 512 KiB |
| Llama 3.1 8B | GQA | 2 × 32 layers × 8 heads × 128 × 2 B | 128 KiB |
| DeepSeek-V3 (671B) | MLA | 61 layers × (512 + 64) × 2 B | 68.6 KiB |
| Falcon 7B | MQA | 2 × 32 layers × 1 head × 64 × 2 B | 8 KiB |

The DeepSeek-V3 row is the striking one. It is a 671-billion-parameter model with 128 attention heads, yet it stores less per token than an 8-billion-parameter Llama. With plain multi-head attention and the same heads, it would need **4,880 KiB per token**, about **71 times** more.

The DeepSeek-V2 paper says MLA's cache is as small as GQA with only **2.25 groups**, and reports that MLA, unlike GQA and MQA, "achieves better performance than MHA". That claim comes from their own tests at large scale. MLA is also harder to build and serve, which is part of why GQA is still the most common choice.

Where does 2.25 come from? DeepSeek-V2 sets the latent to $$d_c = 4 d_h$$ and the RoPE key to $$d_h^R = d_h / 2$$, where $$d_h$$ is one head's size. Set MLA's cache equal to GQA's and solve for the number of groups $$n_g$$:

$$
(d_c + d_h^R)\, l = \left(4 + \tfrac{1}{2}\right) d_h\, l = \tfrac{9}{2}\, d_h\, l = 2\, n_g\, d_h\, l
\qquad\Longrightarrow\qquad
n_g = \tfrac{9}{4} = 2.25
$$

where $$l$$ is the number of layers and the factor 2 on the GQA side counts keys and values.

> [!PAPER] DeepSeek-V2 · Section 2.1.4, Comparison of Key-Value Cache · page 9, Table 1
> [![Table 1 of the DeepSeek-V2 paper: KV cache per token and capability for MHA, GQA, MQA and MLA, with the MLA row highlighted](/img/attention/papers/deepseek-table1.png)](/img/attention/papers/deepseek-table1.png)
>
> **Context:** The paper's summary of the four designs: cache per token (counted in numbers, not bytes) and a one-word verdict on quality.
>
> **What it says:** MHA $$2 n_h d_h l$$ "Strong"; GQA $$2 n_g d_h l$$ "Moderate"; MQA $$2 d_h l$$ "Weak"; MLA $$(d_c + d_h^R) l \approx \frac{9}{2} d_h l$$ "Stronger".
>
> **Why it matters:** These are exactly the formulas in this article (without the bytes per number). The "Stronger" is DeepSeek's own claim from their experiments, and the main reason later models such as DeepSeek-V3, R1, Kimi K2 and Kimi Linear adopted MLA.

## Which one to use?

| | Cache per token | Quality | How hard | Used by |
|---|---|---|---|---|
| **MHA** | largest | the baseline | simplest | older models (Llama 2 7B, GPT-2) |
| **MQA** | smallest | noticeably worse | simple | Falcon, PaLM |
| **GQA** | in between, a dial | close to MHA | simple | Llama 3, Qwen2.5, Gemma 3, Mistral |
| **MLA** | small | as good as MHA or better, per DeepSeek | harder: latent, absorption, separate RoPE key | DeepSeek-V2/V3/R1, Kimi K2 |

All four keep one thing the same: **every token still looks at every earlier token**. They shrink what each token stores, not how many tokens are looked at. Cutting *that* is the subject of Part 3.

## The impact, and where you meet it

These three ideas changed how every large model is built and served:

- **GQA is the default.** Llama 2's largest models adopted it, and Llama 3, Qwen2.5, Gemma 3 and Mistral followed. If you run an open model today, it almost certainly uses GQA.
- **MLA made very large models cheap to serve.** DeepSeek-V2 and V3 used it to serve very large mixture-of-experts models with a small cache, and Moonshot's Kimi models adopted it.
- **MQA** was used by Falcon, PaLM and the original small Gemma 2B, where speed and memory mattered most.

> [!PAPER] Touvron et al. (2023), Llama 2 · Section 2 · page 5
> [![The Llama 2 paper's list of changes from Llama 1, highlighting the adoption of grouped-query attention to improve inference scalability](/img/attention/papers/llama2-gqa.png)](/img/attention/papers/llama2-gqa.png)
>
> **Context:** Meta's list of what changed from Llama 1 to Llama 2.
>
> **What it says:** Alongside more data and longer context, Llama 2 "used grouped-query attention (GQA) to improve inference scalability for our larger models".
>
> **Why it matters:** This is GQA leaving the research paper and entering one of the most widely used model families, only a few months after it was published. "Inference scalability" means exactly what this part is about: serving more users per GPU.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2307.09288)

> [!PAPER] DeepSeek-V2 · Abstract · page 1
> [![The DeepSeek-V2 abstract: compared with DeepSeek 67B it saves 42.5% of training costs, reduces the KV cache by 93.3% and boosts maximum generation throughput to 5.76 times](/img/attention/papers/deepseek-abstract.png)](/img/attention/papers/deepseek-abstract.png)
> [![Figure 1 of the DeepSeek-V2 paper: accuracy against activated parameters, and bar charts of training cost, KV cache per token and generation throughput](/img/attention/papers/deepseek-figure1.png)](/img/attention/papers/deepseek-figure1.png)
>
> **Context:** DeepSeek-V2 compared with their previous dense model, DeepSeek 67B, which used ordinary attention. (V2 also changed other things, notably a mixture-of-experts design, so not all of the gain is MLA.)
>
> **What it says:** The KV cache drops by **93.3%** and the maximum generation throughput rises to **5.76 times**. In Figure 1(b), the middle bar chart is the KV cache per token: a long grey bar for the old model, a tiny blue one for V2.
>
> **Why it matters:** This is the real-world use case: a smaller cache means many more conversations fit on each GPU at once, which is what "throughput" measures.

**Use cases in one line each:**

- **Chat services:** a 4 to 8 times smaller cache (GQA) means 4 to 8 times more simultaneous users per GPU, or much longer conversations.
- **Long documents and code:** at 128K tokens the cache, not the model, fills the GPU; MLA and GQA make such contexts affordable.
- **On-device models:** phones and laptops have little memory, so small KV caches (GQA or MQA) decide how long a conversation can be.
- **Reasoning models:** models that "think" for thousands of tokens before answering write very long outputs, and every one of those tokens reads the cache.

## Summary

- When a model writes, it keeps every earlier token's **keys and values** in the **KV cache**. Queries are used once and never stored.
- Plain MHA stores $$2 \times L \times H \times d_h \times b$$ bytes per token: 512 KiB for Llama 2 7B.
- **MQA** shares one key/value head across all query heads (Falcon 7B: 8 KiB per token) but loses quality. **GQA** shares in groups, a dial between MHA and MQA.
- Our GQA matched PyTorch to $$10^{-16}$$ and reproduced **Qwen2.5-0.5B's real first layer** to $$6.6 \times 10^{-7}$$.
- Fewer key/value heads made each step faster (3.08, 1.85, 1.24 ms) and, more importantly, the cache smaller.
- **MLA** stores a small compressed latent plus a small RoPE key, and **absorbs** the up-projections into the query and output so keys and values are never rebuilt. The fast path matched the slow path to $$3.9 \times 10^{-16}$$.
- RoPE must stay out of the latent: putting it on the rebuilt keys broke the shortcut by about 6%.
- DeepSeek-V3 stores **68.6 KiB per token**; plain MHA would need **4,880 KiB**.
- Writing is limited by **memory reads**, $$t_{\text{step}} \gtrsim \text{bytes read} / \text{bandwidth}$$, which is why shrinking the cache speeds it up (MQA's decoder became about 12× faster in its paper).
- In the papers: MQA lost 0.2 BLEU for a 12× faster decoder; GQA-XXL matched MHA-XXL (47.1 vs 47.2) more than 5 times faster; DeepSeek-V2 cut its KV cache by 93.3%.

<details>
<summary>Run it yourself</summary>

Everything above comes from [`code/attention/part2_kv_variants.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part2_kv_variants.py), which also contains the full `MLA` class with both the slow (rebuild) path and the fast (absorbed) path.

```bash
pip install torch transformers
python part2_kv_variants.py     # writes results/part2.json
```

</details>

## References

1. N. Shazeer. [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150). 2019.
2. J. Ainslie et al. [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). EMNLP 2023.
3. DeepSeek-AI. [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434). 2024.
4. J. Su et al. [*RoFormer: Enhanced Transformer with Rotary Position Embedding*](https://arxiv.org/abs/2104.09864). 2021.
5. Model configurations: [Llama 2 7B](https://huggingface.co/meta-llama/Llama-2-7b-hf), [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B), [DeepSeek-V3](https://huggingface.co/deepseek-ai/DeepSeek-V3), [Falcon 7B](https://huggingface.co/tiiuae/falcon-7b), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
