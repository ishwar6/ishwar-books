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

The cache, more than the arithmetic, limits how many people a GPU can serve and how long their conversations can be. (The [LLM inference series](llm-inference-2-kv-cache.md) goes deeper.) Look at the formula again: the only part we can shrink without changing the model's size much is $$H$$, the number of key/value heads, or what each head stores. That is exactly what the three ideas do.

{{FIG:p2_sharing|Four ways to organise the heads. Orange dots are query heads: they are computed fresh and never stored. Aqua boxes are key/value heads: these are what each token leaves in the cache. MLA stores one compressed vector (blue) instead.}}

## Multi-query attention (MQA): one key/value head for everyone

In 2019, Noam Shazeer proposed the most extreme answer: keep all the query heads, but give the whole layer **just one key head and one value head**, shared by every query head.

In the formula, $$H$$ becomes $$1$$:

$$
\text{MQA cache per token} = 2 \times L \times 1 \times d_h \times b
$$

So the cache shrinks by a factor equal to the number of heads.

Falcon-7B is a real example: 71 query heads sharing one key/value head. It stores **8 KiB per token**, against 512 KiB for Llama 2 7B.

The price is quality. Every query head must find what it needs in the very same keys and values, and models trained this way tend to be measurably worse. The DeepSeek-V2 paper says it plainly: MQA and GQA "require a smaller magnitude of KV cache, but their performance does not match MHA."

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

## Which one to use?

| | Cache per token | Quality | How hard | Used by |
|---|---|---|---|---|
| **MHA** | largest | the baseline | simplest | older models (Llama 2 7B, GPT-2) |
| **MQA** | smallest | noticeably worse | simple | Falcon, PaLM |
| **GQA** | in between, a dial | close to MHA | simple | Llama 3, Qwen2.5, Gemma 3, Mistral |
| **MLA** | small | as good as MHA or better, per DeepSeek | harder: latent, absorption, separate RoPE key | DeepSeek-V2/V3/R1, Kimi K2 |

All four keep one thing the same: **every token still looks at every earlier token**. They shrink what each token stores, not how many tokens are looked at. Cutting *that* is the subject of Part 3.

## Summary

- When a model writes, it keeps every earlier token's **keys and values** in the **KV cache**. Queries are used once and never stored.
- Plain MHA stores $$2 \times L \times H \times d_h \times b$$ bytes per token: 512 KiB for Llama 2 7B.
- **MQA** shares one key/value head across all query heads (Falcon 7B: 8 KiB per token) but loses quality. **GQA** shares in groups, a dial between MHA and MQA.
- Our GQA matched PyTorch to $$10^{-16}$$ and reproduced **Qwen2.5-0.5B's real first layer** to $$6.6 \times 10^{-7}$$.
- Fewer key/value heads made each step faster (3.08, 1.85, 1.24 ms) and, more importantly, the cache smaller.
- **MLA** stores a small compressed latent plus a small RoPE key, and **absorbs** the up-projections into the query and output so keys and values are never rebuilt. The fast path matched the slow path to $$3.9 \times 10^{-16}$$.
- RoPE must stay out of the latent: putting it on the rebuilt keys broke the shortcut by about 6%.
- DeepSeek-V3 stores **68.6 KiB per token**; plain MHA would need **4,880 KiB**.

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
