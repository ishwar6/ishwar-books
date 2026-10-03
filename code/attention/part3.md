---
title: "Attention, Part 3: Sliding-Window and Sparse Attention"
description: "Why looking at every earlier token gets too expensive, and the three ways today's models avoid it: sliding windows (Mistral), mixing local and global layers (Gemma 2, Gemma 3, gpt-oss), and letting the model pick its own tokens (DeepSeek Sparse Attention). Explained from zero, with equations, quotes from the papers, and real experiments: a lightning indexer trained on Qwen2.5-0.5B keeps quality close to full attention while each token reads only 64 of up to 2,048 earlier tokens."
date: 2026-10-04
tags: [attention, sparse-attention, long-context, llm]
series: "Attention, From the Ground Up"
series_part: 3
motif: graph
accent: "#f0a35e"
---

[Part 2](attention-2-mqa-gqa-mla.md) made each token **store less**: fewer key/value heads (MQA, GQA) or a compressed latent (MLA). But every token still looked at **every** earlier token.

This part attacks that second problem: **how many tokens each token looks at**. It is the biggest cost of long conversations, long documents and long chains of reasoning, and it is why every model that handles 128,000 tokens or more uses at least one of the ideas below.

This is a long part, so here is the map.

> [!TIP] What you will learn
> 1. **Why full attention gets expensive:** the cost grows with the *square* of the length.
> 2. **Sliding-window attention:** each token looks only at the last W tokens (Mistral 7B).
> 3. **How stacked layers still see far**, and the **rolling buffer cache** that keeps memory fixed.
> 4. **Local and global layers mixed together** (Gemma 2, Gemma 3, gpt-oss), with real memory numbers.
> 5. **Attention sinks again:** a real experiment where cutting off the first token breaks a model.
> 6. **DeepSeek Sparse Attention:** a tiny "lightning indexer" picks which tokens to read. I train one on a real model and test it.

As in the earlier parts, every technical word gets a yellow box the first time it appears, every number comes from code I ran, and every quote links to its paper.

> [!TIP] Quick recap
> Each token makes a **query**, a **key** and a **value**. Its query is compared with the keys of earlier tokens (dot products), softmax turns the scores into weights that add up to 1, and the weights mix the values. While a model writes, it keeps all earlier keys and values in the **KV cache** ([Part 2](attention-2-mqa-gqa-mla.md)).

## 1. The problem: everyone looks at everyone

In full attention, token number $$i$$ compares its query with the keys of tokens $$1, 2, \dots, i$$. So for a text of $$T$$ tokens, the number of query-key comparisons in **one head of one layer** is:

$$
1 + 2 + 3 + \dots + T = \frac{T\,(T+1)}{2}
$$

> [!DEFINITION] Quadratic growth
> When a cost grows with the **square** of the size ($$T^2$$), doubling the length makes it about **four times** bigger, and ten times the length makes it about **a hundred times** bigger. Attention's comparisons grow this way.

Here is what that means in numbers:

| Text length $$T$$ | Comparisons, full attention | Comparisons, window of 1,024 | Full ÷ window |
|---|---|---|---|
| 1,024 tokens | 524,800 | 524,800 | 1.0× |
| 8,192 tokens | 33,558,528 | 7,864,832 | 4.3× |
| 131,072 tokens (128K) | 8,590,000,128 | 133,693,952 | 64.3× |

At 128K tokens, full attention makes **8.6 billion** comparisons, in every head of every layer. A window of 1,024 tokens makes 64 times fewer.

There is also a memory cost while the model writes. The KV cache grows by one token at every step, and **every step reads all of it**. Twice the conversation means twice the memory and twice the reading per step.

> [!NOTE] The hopeful observation
> Most attention is **local**. In [Part 1](attention-1-self-attention.md) we saw a head that put 84% of its attention on the *previous token*, and many heads that dump attention on the *first token*. In the Qwen experiment later in this part, just the last 64 tokens plus the first 4 already catch **72%** of all attention, averaged over every layer (all heads together). So most of those 8.6 billion comparisons produce weights close to zero.

The rest of this part is about skipping the comparisons that do not matter, without skipping the ones that do.

{{FIG:p3_masks|Three patterns on 12 tokens. Full attention: every token sees all earlier tokens. Sliding window: each token sees only itself and the 3 before it. Sparse top-k: each token sees a small, chosen set (here an illustrative random pick).}}

## 2. Sliding-window attention (SWA)

The simplest fix: **each token looks only at the last W tokens**, itself included. Nothing older.

> [!DEFINITION] Sliding window and window size W
> A fixed-size "view" that moves along with the text. With window size $$W$$, token $$i$$ may attend to tokens $$i-W+1, \dots, i$$. As the text moves forward, the window slides with it. Typical sizes: 128 (gpt-oss), 1,024 (Gemma 3), 4,096 (Mistral 7B, Gemma 2).

In Part 1 we wrote attention with a **mask** $$M$$ that blocks the future. Sliding-window attention is the same formula with a stricter mask:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V,
\qquad
M_{ij} = \begin{cases} 0 & \text{if } 0 \le i - j < W \\ -\infty & \text{otherwise} \end{cases}
$$

where:

- $$i$$ is the position of the query token and $$j$$ the position of a key token;
- $$i - j$$ is how far back token $$j$$ is from token $$i$$;
- $$i - j < 0$$ would be the future (blocked, as before);
- $$i - j \ge W$$ is too far back (newly blocked);
- $$-\infty$$ becomes a weight of exactly 0 after softmax.

That is the whole idea. In code it is one extra condition on the mask:

```python
def window_mask(T, W, device=None):
    """True where query i may look at key j: j <= i (causal) and i - j < W (inside the window)."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (i - j < W)


def attend(q, k, v, allowed):
    """Plain attention with a boolean 'allowed' mask. q, k, v: (..., T, d)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float('-inf'))
    return torch.softmax(scores, -1) @ v
```

For 10 tokens and $$W = 4$$, the mask looks like this (1 = may look, 0 = may not). Each row is one token; it sees itself and the three before it:

```text
1 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0
1 1 1 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0
0 1 1 1 1 0 0 0 0 0
0 0 1 1 1 1 0 0 0 0
0 0 0 1 1 1 1 0 0 0
0 0 0 0 1 1 1 1 0 0
0 0 0 0 0 1 1 1 1 0
0 0 0 0 0 0 1 1 1 1
```

> [!NOTE] Two ways of counting the window
> The Mistral paper says position $$i$$ attends to positions "between $$i - W$$ and $$i$$", which is $$W + 1$$ tokens if both ends are included. The Hugging Face library, which most people use to run these models, uses $$i - j < W$$: exactly $$W$$ tokens, the current one included. I use the Hugging Face rule everywhere in this part. The difference of one token never matters in practice, but it explains why formulas in different sources differ by one.

### Proof that it works

I checked three things in `part3_sparse.py` (24 tokens, $$W = 6$$, 64-bit numbers):

```text
1. SWA vs PyTorch SDPA with the same mask:  max |diff| = 4.4e-16
   window wider than the text vs causal:   max |diff| = 0.0e+00
   change all 18 tokens outside the window of the last token:
      last output with sliding window moves by 0.0e+00
      last output with full attention moves by 0.65
```

1. My function agrees with PyTorch's built-in attention to $$4.4 \times 10^{-16}$$ (rounding noise).
2. A window wider than the text is exactly ordinary causal attention (difference 0).
3. The real test of "only the last W tokens matter": I replaced **all 18 tokens outside the last token's window** with random new ones. With the sliding window, the last token's output did not move at all (difference exactly 0). With full attention, it moved by 0.65.

## 3. "But then the model forgets everything older than W?"

Not quite, and this is the clever part. A model has many layers stacked on top of each other. Layer 2 reads the *outputs* of layer 1, and each of those outputs already mixed information from its own window.

> [!DEFINITION] Receptive field
> All the input tokens that can affect one output, directly or indirectly. In a single sliding-window layer it is just the window. Across stacked layers it grows, like a rumour passed from person to person.

{{FIG:p3_receptive|How the reach grows. Window W = 3. The last token after layer 3 reads 3 tokens of layer 2, each of which read 3 tokens of layer 1, and so on. By the input, 7 tokens can affect it.}}

Each layer lets information travel $$W - 1$$ more tokens back. After $$L$$ layers:

$$
\text{reach} = L \times (W - 1) + 1 \ \text{tokens}
$$

The Mistral 7B paper says the same thing (counting the window with its $$W+1$$ convention):

> [!QUOTE] Mistral 7B paper
> "At each attention layer, information can move forward by W tokens. Hence, after k attention layers, information can move forward by up to k × W tokens."
>
> "At the last layer, using a window size of W = 4096, we have a theoretical attention span of approximately 131K tokens."
>
> Source: [Jiang et al., 2023](https://arxiv.org/abs/2310.06825)

(32 layers × 4,096 = 131,072.)

**Proof.** I stacked 1 to 6 sliding-window layers ($$W = 4$$, with the usual "add the input back" connection that real models use), and asked PyTorch which input tokens the last output depends on. It can tell exactly, by computing the **gradient**.

> [!DEFINITION] Gradient (as a dependency detector)
> The gradient of an output with respect to an input says how much the output would change if that input changed slightly. If it is exactly zero for an input, the output does not depend on that input at all.

```text
2. Receptive field of the last token, window W = 4
   1 layer(s): depends on  4 tokens (formula L*(W-1)+1 =  4), reaches  3 tokens back
   2 layer(s): depends on  7 tokens (formula L*(W-1)+1 =  7), reaches  6 tokens back
   3 layer(s): depends on 10 tokens (formula L*(W-1)+1 = 10), reaches  9 tokens back
   4 layer(s): depends on 13 tokens (formula L*(W-1)+1 = 13), reaches 12 tokens back
   5 layer(s): depends on 16 tokens (formula L*(W-1)+1 = 16), reaches 15 tokens back
   6 layer(s): depends on 19 tokens (formula L*(W-1)+1 = 19), reaches 18 tokens back
```

The measured reach matches the formula exactly, at every depth.

> [!WARNING] "Can reach" is not "remembers well"
> The formula gives the *theoretical* reach. Information that travels through many layers gets blended with everything else on the way, like a message whispered down a long line. A sliding-window-only model can technically be influenced by a token 100,000 positions back, but it cannot *look it up* directly. That is why most recent models add some global layers (section 5).

## 4. The rolling buffer cache: memory that never grows

With a window, a token never needs keys and values older than $$W$$ steps. So the KV cache can be a fixed-size **ring**.

> [!DEFINITION] Rolling (ring) buffer
> A fixed number of slots used in a circle. When the last slot is filled, writing goes back to slot 0 and overwrites the oldest item. Token $$i$$ goes into slot $$i \bmod W$$ ("the remainder when $$i$$ is divided by $$W$$").

> [!QUOTE] Mistral 7B paper
> "The cache has a fixed size of W, and the keys and values for the timestep i are stored in position i mod W of the cache. As a result, when the position i is larger than W, past values in the cache are overwritten, and the size of the cache stops increasing."

{{FIG:p3_ring|A rolling buffer with W = 4 slots while writing token 9. Token i goes into slot i mod 4. Tokens 0 to 5 have been overwritten; the cache always holds just the last 4 tokens.}}

Here is the decoding loop from `part3_sparse.py`. It writes one token at a time into a cache of only $$W$$ slots:

```python
k_cache = torch.zeros(H, W, d, dtype=torch.float64)                     # the whole cache: W slots, never more
v_cache = torch.zeros(H, W, d, dtype=torch.float64)
step_out = []
for i in range(T):
    xi = x[i:i + 1]
    qi = rope((xi @ Wq).view(1, H, d).transpose(0, 1), pos[i:i + 1])
    ki = rope((xi @ Wk).view(1, H, d).transpose(0, 1), pos[i:i + 1])
    vi = (xi @ Wv).view(1, H, d).transpose(0, 1)
    slot = i % W                                                        # position i goes to slot i mod W
    k_cache[:, slot], v_cache[:, slot] = ki[:, 0], vi[:, 0]
    n = min(i + 1, W)                                                   # how many slots are filled
    step_out.append(attend(qi, k_cache[:, :n], v_cache[:, :n], torch.ones(1, n, dtype=torch.bool)))
```

Notice that the slots are **out of order**: in the picture, slot 0 holds token 8 and slot 2 holds token 6. Does that break anything? No, for two reasons:

1. **Softmax and the weighted sum do not care about order.** Adding up "weight × value" over a set of tokens gives the same answer in any order.
2. **Position is already baked into each key.** RoPE (explained in [Part 2](attention-2-mqa-gqa-mla.md)) rotates each key by its position *before* it is stored, so the key remembers where it came from, whatever slot it sits in.

**Proof.** I compared this token-by-token ring cache with computing all 50 tokens at once using the sliding-window mask (2 heads, RoPE on):

```text
3. Rolling buffer cache (W = 8 slots) vs computing all 50 tokens at once: max |diff| = 1.1e-15
```

Identical, while the cache held 8 slots instead of 50.

### How much memory does it save?

The KV cache formula from Part 2, with the number of stored tokens capped at $$W$$:

$$
\text{KV cache} = 2 \times L \times H_{kv} \times d_h \times \min(n, W) \times b
$$

where $$n$$ is the number of tokens so far, and the other symbols are as in Part 2 (layers, key/value heads, head size, bytes per number).

For Mistral 7B (32 layers, 8 key/value heads, head size 128, window 4,096) at 32,768 tokens:

```text
   Mistral 7B at 32K: full 4.00 GiB, window 4096 0.50 GiB (8x smaller)
```

> [!QUOTE] Mistral 7B paper
> "On a sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model quality."

My calculation gives the same 8×.

### And speed?

Every writing step reads the whole cache. With full attention that read keeps growing; with a window it stays the same size. I timed one attention step (16 heads, head size 128, 16-bit numbers) on an Apple M5 Pro GPU:

{{FIG:p3_decode|Time for one decode step of attention. Full attention gets slower as the conversation grows, because it reads the whole cache. With a 1,024-token window, the step time stays flat.}}

| Tokens so far | Full attention | Window of 1,024 |
|---|---|---|
| 1,024 | 0.057 ms | 0.041 ms |
| 4,096 | 0.137 ms | 0.033 ms |
| 16,384 | 0.521 ms | 0.030 ms |
| 65,536 | 2.019 ms | 0.034 ms |
| 131,072 | 3.893 ms | 0.031 ms |

At 128K tokens, the windowed step is about **125 times faster**. (At 1,024 tokens both read the same amount; the small gap there is timing noise.)

## 5. Mixing local and global layers

A pure sliding-window model has the weakness from the warning above: it can never look up an old token directly. The fix that most recent models use is to **mix** two kinds of layers:

> [!DEFINITION] Local layer and global layer
> A **local** layer uses sliding-window attention: cheap, short memory. A **global** layer uses ordinary full attention: expensive, but it can look directly at any earlier token. A model can stack them in a fixed pattern, for example 5 local then 1 global.

This idea is older than chatbots. Longformer (2020) already combined the two inside one layer:

> [!QUOTE] Longformer
> "Longformer's attention mechanism is a drop-in replacement for the standard self-attention and combines a local windowed attention with a task motivated global attention."
>
> Source: [Beltagy et al., 2020](https://arxiv.org/abs/2004.05150)

Today's models do it **layer by layer**:

- **Gemma 2** (2024) alternates **1 local : 1 global**, with a local window of 4,096 tokens.
- **Gemma 3** (2025) goes further: **5 local : 1 global**, and shrinks the window to **1,024**.
- **gpt-oss** (OpenAI, 2025) alternates **1 : 1** with a tiny window of just **128** tokens.

> [!QUOTE] Gemma 3 Technical Report
> "We alternate between a local sliding window self-attention and global self-attention, with a pattern of 5 local layers for every global layer, starting with a local layer as the first layer of the model."
>
> Source: [Gemma Team, 2025](https://arxiv.org/abs/2503.19786)

They also say why: long context causes "the memory explosion of the KV cache during inference". In their tests, changing the local:global ratio had "minimal impact on perplexity".

{{FIG:p3_gemma|The real layer pattern of Gemma 3 27B: five local layers (blue, window 1,024), then one global layer (orange), repeated. 52 local and 10 global layers in total. Hover a bar to see the layer.}}

Here is the published configuration (from Hugging Face), which is where those numbers come from:

```json
{
  "num_hidden_layers": 62,
  "num_attention_heads": 32,
  "num_key_value_heads": 16,
  "head_dim": 128,
  "sliding_window": 1024,
  "sliding_window_pattern": 6,
  "rope_theta": 1000000.0,
  "rope_local_base_freq": 10000.0
}
```

`sliding_window_pattern: 6` means every 6th layer is global: layers 6, 12, 18, ..., 60. That gives 10 global and 52 local layers.

### The memory math

Global layers store every token; local layers store at most $$W$$:

$$
\text{KV cache} = 2 \, H_{kv} \, d_h \, b \, \Big( L_{\text{global}} \cdot n \;+\; L_{\text{local}} \cdot \min(n, W) \Big)
$$

where $$L_{\text{global}}$$ and $$L_{\text{local}}$$ are the numbers of global and local layers. For Gemma 3 27B ($$H_{kv} = 16$$, $$d_h = 128$$, 2 bytes per number):

```text
   Gemma 3 27B (52 local + 10 global layers):
         1024 tokens: all global   0.48 GiB   5 local : 1 global  0.48 GiB   (1.0x smaller)
         2048 tokens: all global   0.97 GiB   5 local : 1 global  0.56 GiB   (1.7x smaller)
         4096 tokens: all global   1.94 GiB   5 local : 1 global  0.72 GiB   (2.7x smaller)
         8192 tokens: all global   3.88 GiB   5 local : 1 global  1.03 GiB   (3.8x smaller)
        16384 tokens: all global   7.75 GiB   5 local : 1 global  1.66 GiB   (4.7x smaller)
        32768 tokens: all global  15.50 GiB   5 local : 1 global  2.91 GiB   (5.3x smaller)
        65536 tokens: all global  31.00 GiB   5 local : 1 global  5.41 GiB   (5.7x smaller)
       131072 tokens: all global  62.00 GiB   5 local : 1 global 10.41 GiB   (6.0x smaller)
```

{{FIG:p3_kv|KV cache of Gemma 3 27B for one conversation. If every layer were global, 128K tokens would need 62 GiB. With the real 5:1 pattern it needs 10.4 GiB.}}

At the full 128K context: **62 GiB → 10.4 GiB**. The savings can never pass 6.2× (62 layers ÷ 10 global layers), because the global layers still grow with the text. They become the main cost.

For gpt-oss-20b (24 layers alternating, 8 key/value heads, head size 64, window 128), at 128K tokens the cache drops from 6.00 GiB to 3.00 GiB: half the layers keep almost nothing.

> [!NOTE] A detail worth knowing: two RoPE speeds
> Gemma 3 uses different RoPE settings for the two kinds of layer: "We increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local layers at 10k." Global layers must tell apart positions up to 128K tokens apart, so their rotation turns more slowly (a larger base). Local layers only ever see 1,024 tokens, so the usual setting is fine.

| Model | Pattern | Window | Why it matters |
|---|---|---|---|
| Mistral 7B | every layer local | 4,096 | made SWA and the rolling cache popular in open models |
| Gemma 2 | 1 local : 1 global | 4,096 | half the layers stay small |
| Gemma 3 | 5 local : 1 global | 1,024 | 6× smaller cache at 128K |
| gpt-oss | 1 local : 1 global | 128 | tiny windows, half the layers almost free |

## 6. Attention sinks strike again: a real experiment

All the models above were **trained** with their windows, so they learned to live with them. What happens if you take a normal model, trained with full attention, and simply force a window on it? This is what "StreamingLLM" studied, and the answer surprised people.

> [!QUOTE] StreamingLLM
> "Window attention, where only the most recent KVs are cached, is a natural approach -- but we show that it fails when the text length surpasses the cache size. We observe an interesting phenomenon, namely attention sink, that keeping the KV of initial tokens will largely recover the performance of window attention."
>
> Source: [Xiao et al., 2023](https://arxiv.org/abs/2309.17453)

Remember the **attention sink** from [Part 1](attention-1-self-attention.md): in Qwen2.5-0.5B, 68% of heads put most of their attention on the very first token, because softmax forces the weights to add up to 1 and the heads need somewhere harmless to "park" them. A sliding window cuts that first token off.

I tested this myself.

> [!DEFINITION] Perplexity
> The standard score for how well a language model predicts text. Roughly: "on average, how many tokens was the model choosing between?" **Lower is better.** It is $$e^{\text{average loss}}$$, where the loss for each token is $$-\log(\text{probability the model gave the correct next token})$$.

**Setup** (`part3_qwen.py`):

- Model: **Qwen2.5-0.5B**, trained with full attention.
- Text: four chunks of 2,048 tokens from *Pride and Prejudice* (public domain, from Project Gutenberg).
- I score only tokens 1,024 to 2,047, where every window really cuts something off.
- Each rule gets the **same budget**: each token may look at exactly 64, 256 or 1,024 tokens.
- "4 sinks + window" means: the first 4 tokens of the text, plus the most recent (budget − 4) tokens.
- First, a sanity check: passing my own full-attention mask gives exactly the model's normal output (max difference 0.0).

```text
full attention: perplexity 17.39
budget   64 tokens: window only   132.70   4 sinks + window 60:  22.45
budget  256 tokens: window only    67.90   4 sinks + window 252:  18.83
budget 1024 tokens: window only   508.88   4 sinks + window 1020:  17.80
```

What this shows:

- **A plain window destroys the model.** Perplexity jumps from 17.4 to between 68 and 509.
- **Keeping just 4 sink tokens fixes almost all of it.** With 1,024 tokens of budget, 4 sinks plus a window give 17.80, very close to full attention's 17.39.
- **A bigger window does not save you if the sink is gone.** The 1,024-token window without sinks was the *worst* of all (508.9), even though it sees 16 times more text than the 64-token window. I did not expect this. It shows the problem is not missing information: the model is thrown off by losing its "parking spot".

> [!NOTE] How this differs from StreamingLLM exactly
> StreamingLLM also renumbers the positions inside the cache when it evicts tokens. I kept the original positions and only changed which tokens are visible. The lesson is the same: a model trained with full attention depends on its first tokens, and **"just use a window" is not safe for a model that was not trained with one**. Models like Mistral and Gemma learn to cope with windows during training.

## 7. Sparse attention: let the model choose

Windows are **fixed** patterns: they always keep the most recent tokens. But sometimes the important token is far away. A character's name introduced in chapter 1, a function defined at the top of a file, the original question at the start of a long chat.

> [!DEFINITION] Sparse attention
> Attention where each token looks at only a small **subset** of earlier tokens instead of all of them. "Sparse" means mostly empty: in the attention grid, most squares are skipped. The subset can be fixed (a window, a stride) or chosen by the model.

Early sparse transformers ([Child et al., 2019](https://arxiv.org/abs/1904.10509)) used fixed patterns, like "the last few tokens plus every 64th token". The newer idea is to let the model **pick** the tokens that matter for each query. The problem: to know which tokens score highest, you seem to need the scores, which is the very thing you were trying to avoid computing.

DeepSeek's answer: compute the scores with a **much smaller, much cheaper** model first, then do the expensive attention only on the winners.

> [!DEFINITION] Top-k selection
> From a list of scores, keep the $$k$$ largest and drop the rest. If $$k = 2{,}048$$, each token reads at most 2,048 earlier tokens, no matter whether the text is 10,000 or 128,000 tokens long.

## 8. DeepSeek Sparse Attention (DSA)

DSA arrived with DeepSeek-V3.2 (2025). It has two pieces.

> [!QUOTE] DeepSeek-V3.2 paper
> "The prototype of DSA primarily consists of two components: a lightning indexer and a fine-grained token selection mechanism."
>
> Source: [DeepSeek-AI, 2025](https://arxiv.org/abs/2512.02556)

{{FIG:p3_dsa|DeepSeek Sparse Attention. The lightning indexer is small and cheap, but it scores every earlier token. Top-k keeps the best k. The big main attention then runs only over those k tokens.}}

### Piece 1: the lightning indexer

> [!DEFINITION] Lightning indexer
> A tiny attention-like scorer that runs **before** the real attention. For the current token $$t$$ and each earlier token $$s$$, it produces one number, the **index score** $$I_{t,s}$$: "how useful will token $$s$$ be for token $$t$$?" It has only a few heads and can run in low precision, so it is very fast.

The index score is:

$$
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \operatorname{ReLU}\!\left(q^I_{t,j} \cdot k^I_s\right)
$$

where:

- $$t$$ is the current (query) token and $$s$$ an earlier token;
- $$H^I$$ is the number of **indexer heads** (a small number);
- $$q^I_{t,j}$$ is the indexer query of token $$t$$ for indexer head $$j$$;
- $$k^I_s$$ is the indexer key of token $$s$$ (one key per token, shared by all indexer heads);
- $$w^I_{t,j}$$ is a weight that token $$t$$ gives to indexer head $$j$$ (how much to trust that head for this token);
- $$\operatorname{ReLU}(x) = \max(0, x)$$.

> [!DEFINITION] ReLU
> A very simple function: negative numbers become 0, positive numbers stay as they are. $$\operatorname{ReLU}(-3) = 0$$, $$\operatorname{ReLU}(2) = 2$$. It is cheaper than softmax because it needs no exponentials and no sum over all tokens.

> [!DEFINITION] FP8
> Numbers stored in 8 bits (1 byte) instead of the usual 16. Less precise, but twice as small and much faster on modern GPUs. Good enough for *ranking* tokens, which is all the indexer must do.

The paper explains both choices: ReLU was chosen "for throughput consideration", and the indexer "can be implemented in FP8".

### Piece 2: top-k token selection

For each query token $$t$$, keep only the $$k$$ earlier tokens with the highest index scores, and run the real attention over just those:

$$
S_t = \operatorname{Top\text{-}k}\big(I_{t,:}\big), \qquad u_t = \operatorname{Attn}\big(h_t, \{\, c_s : s \in S_t \,\}\big)
$$

where:

- $$I_{t,:}$$ means all of token $$t$$'s index scores;
- $$S_t$$ is the set of selected positions;
- $$h_t$$ is the current token's hidden vector;
- $$c_s$$ is the cached key-value entry of token $$s$$ (in DeepSeek, the MLA latent from [Part 2](attention-2-mqa-gqa-mla.md));
- $$u_t$$ is the attention output.

DeepSeek-V3.2 uses $$k = 2{,}048$$.

### What it saves

> [!QUOTE] DeepSeek-V3.2 paper
> DSA "reduces the core attention complexity of the main model from O(L²) to O(Lk)". And: "Although the lightning indexer still has a complexity of O(L²), it requires much less computation compared with MLA."

> [!DEFINITION] Big-O notation
> A shorthand for how cost grows with size. $$O(L^2)$$: grows with the square of the length $$L$$. $$O(Lk)$$: grows with $$L$$ times a fixed number $$k$$, so only linearly in $$L$$.

So the expensive part (the big attention with 128 heads) becomes linear in length. The cheap part (the indexer) is still quadratic, but it is so small and so fast that it costs much less.

### How the indexer learns: two training stages

The indexer starts out random. How does it learn which tokens matter? It copies the model's own attention. DeepSeek trains it in two stages:

**Stage 1, dense warm-up.** Keep normal full attention, **freeze the whole model**, and train only the indexer to predict where the model's attention goes.

> [!QUOTE] DeepSeek-V3.2 paper
> "for the t-th query token, we first aggregate the main attention scores by summing across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a target distribution $$p_{t,:}$$"

The training loss compares the indexer's scores (turned into a distribution by softmax) with that target:

$$
\mathcal{L}^I = \sum_t D_{\mathrm{KL}}\Big(p_{t,:} \,\Big\|\, \operatorname{Softmax}(I_{t,:})\Big)
$$

> [!DEFINITION] L1-normalised and KL divergence
> **L1-normalised:** divided by its total, so the numbers add up to 1, like percentages. **KL divergence** $$D_{\mathrm{KL}}(p \,\|\, q)$$: a measure of how different two such distributions are. It is 0 when they are identical and grows as they differ. Training makes it as small as possible, so the indexer's ranking matches the real attention.

This stage is short: 1,000 steps, 2.1 billion tokens, learning rate $$10^{-3}$$.

**Stage 2, sparse training.** Switch on top-k selection and train the **whole** model to work with it: 15,000 steps, 943.7 billion tokens. The indexer keeps learning from the KL loss, but it is trained separately: the paper says "we detach the indexer input from the computational graph for separate optimization", meaning the indexer's learning signal does not flow back into the main model.

> [!NOTE] Built on MLA
> DSA is added on top of the MLA attention from Part 2, in "MQA mode": each cached latent is shared by all query heads. That way, choosing a token means fetching **one** small latent, which all 128 heads then use. DeepSeek reports no substantial quality drop against the dense model it started from (DeepSeek-V3.1-Terminus), with large speedups at long context.

## 9. I trained a lightning indexer on a real model

To see whether this really works, and not just trust the paper, I built a small DSA for **Qwen2.5-0.5B** and ran **stage 1** (the dense warm-up) exactly as described: model frozen, indexer trained with the KL loss against the model's own head-summed, L1-normalised attention.

**The indexer**, one per layer, 24 in total:

- 4 indexer heads of size 32, one shared key per token, ReLU, and the weights $$w$$;
- RoPE on its queries and keys, so it knows positions;
- a LayerNorm on its input (more on this below);
- **148,736 parameters per layer**, about 8% of the size of the attention layer it serves (1,836,160).

```python
class LightningIndexer(torch.nn.Module):
    """I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])   (DeepSeek-V3.2, eq. 1). One shared key per token."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)                        # Qwen's hidden states have a few huge values; tame them
        self.q = torch.nn.Linear(d_model, HI * DI, bias=False)
        self.k = torch.nn.Linear(d_model, DI, bias=False)
        self.w = torch.nn.Linear(d_model, HI, bias=False)

    def forward(self, h):                                              # h: (T, d_model)
        n, h = h.shape[0], self.norm(h)
        q = rope(self.q(h).view(n, HI, DI).transpose(0, 1))            # (HI, T, DI)
        k = rope(self.k(h))                                            # (T, DI)
        return torch.einsum('jt,jts->ts', self.w(h).T, F.relu(q @ k.T))   # (T, T)
```

The target and the loss, as in the paper (summing the 14 heads, then dividing by the total; the KL loss written as cross-entropy, which differs from KL only by a constant):

```python
        A = output[1][0]                                               # (heads, T, T)
        captured[layer]['p'] = (A.sum(0) / A.sum(0).sum(-1, keepdim=True)).detach()   # sum over heads, L1-normalise
```

```python
def kl_loss(ix, h, p):
    I = ix(h).masked_fill(~CAUSAL, float('-inf'))
    logq = torch.log_softmax(I, -1)
    return -(p * logq.masked_fill(~CAUSAL, 0)).sum(-1).mean()          # KL(p || softmax(I)) up to a constant
```

**Training:** 400 steps per layer, learning rate $$10^{-3}$$ (the paper's warm-up rate), on six 2,048-token chunks of *The Adventures of Sherlock Holmes*. **Testing** used a different book, *Pride and Prejudice*, so the indexer never saw the test text.

> [!WARNING] A real problem I hit: "dead" indexers
> My first run had no LayerNorm. In layers 22 and 23, the training loss got stuck at exactly 6.627. That number is the loss of a completely flat guess over 2,048 tokens: the indexer had "died". The ReLU outputs were all zero, so every token got the same score and nothing could be learned. The cause: Qwen's hidden vectors contain a few enormous values, especially in late layers, which pushed the first updates too far. Normalising the input (LayerNorm) and limiting the size of each update (gradient clipping) fixed it, and all 24 layers then trained normally. ReLU's "nothing below zero" makes this failure possible, so it is worth knowing about.

### Result 1: does it pick the tokens the model really attends to?

For each layer and each query token (positions 1,024 to 2,047), I measured **what share of the model's real attention lands on the chosen tokens**. 1.0 would mean the chosen tokens hold all of the attention.

```text
budget 64: attention captured  window 0.423  sinks+window 0.721  indexer 0.776  (untrained, layer 0: 0.042)
budget 256: attention captured  window 0.520  sinks+window 0.820  indexer 0.887  (untrained, layer 0: 0.177)
```

| Budget per token | Last k tokens | 4 sinks + window | Trained indexer | Untrained indexer (layer 1) |
|---|---|---|---|---|
| 64 | 42.3% | 72.1% | **77.6%** | 4.2% |
| 256 | 52.0% | 82.0% | **88.7%** | 17.7% |

- An **untrained** indexer catches almost nothing (4.2%), so the training clearly did the work.
- The **trained** indexer beats the strong "sinks + window" rule at both budgets. It found the sinks and the recent tokens by itself, *plus* some important far-away tokens.

{{FIG:p3_cov|Layer by layer, the share of real attention caught by 64 tokens. The trained indexer (blue) is above "4 sinks + window" (orange) in every layer. The last two layers spread their attention widely, so 64 tokens catch less there.}}

Here is one real example: the very last token of a test chunk, in layer 13.

{{FIG:p3_pick|For one real query: the 64 tokens the model actually attends to most (top) and the 64 tokens the trained indexer picked (bottom), across positions 0 to 2,047. They agree on 58 of 64. Both include the first token (the sink) and a cluster of recent tokens, plus a few far-away ones.}}

### Result 2: does the model still work with sparse attention?

The real test: run Qwen with **every layer** using top-k attention chosen by its indexer, and measure perplexity on *Pride and Prejudice*.

{{FIG:p3_ppl|Perplexity when each token may look at only 64 or 256 earlier tokens, chosen three ways, against full attention (lower is better). The trained indexer comes closest to full attention at both budgets.}}

| Budget per token | Last k tokens | 4 sinks + window | Trained indexer, top-k | Full attention |
|---|---|---|---|---|
| 64 | 132.7 | 22.45 | **18.93** | 17.39 |
| 256 | 67.9 | 18.83 | **17.62** | 17.39 |

With 256 tokens per query (full attention reads 1,025 to 2,048 at these positions), the indexer version reaches **17.62**, against 17.39 for full attention, **without retraining the model at all**. With 64 tokens it stays at 18.93, clearly better than the best fixed pattern (22.45).

> [!NOTE] What this experiment does and does not show
> - It shows the **quality** side: a small trained indexer can pick the right tokens, even in a model never trained for sparse attention.
> - It is only **stage 1**. DeepSeek's stage 2 retrains the whole model with sparse attention, which should close the remaining gap.
> - It does **not** show speed. My version still computes the full score grid and then hides most of it with a mask, so it is not faster. Real speedups need special GPU code that fetches only the selected tokens.
> - It is one small model, one test book and 2,048-token texts, so read the exact numbers as an illustration, not a benchmark.

## 10. Proof of the runs

Both scripts, exactly as they ran (the same outputs quoted above):

<figure class="fig"><img src="/img/attention/part3-sparse-run.png" alt="Terminal output of part3_sparse.py: sliding-window checks, receptive field, rolling cache, KV memory and decode timing" loading="lazy" /><figcaption>Output of part3_sparse.py on an Apple M5 Pro.</figcaption></figure>

<figure class="fig"><img src="/img/attention/part3-qwen-run.png" alt="Terminal output of part3_qwen.py: window and sink perplexities, indexer warm-up losses per layer, attention captured and sparse perplexities" loading="lazy" /><figcaption>Output of part3_qwen.py: Qwen2.5-0.5B with windows, sinks and trained lightning indexers. The whole run took 106 seconds.</figcaption></figure>

## Which one to use?

| | Each token looks at | KV cache | Can look far back directly? | Used by |
|---|---|---|---|---|
| **Full attention** | all earlier tokens | grows with length | yes | most models, in at least some layers |
| **Sliding window** | the last W | fixed at W | no (only indirectly) | Mistral 7B |
| **Local + global layers** | last W in local layers, all in global | local layers fixed, global grow | yes, in global layers | Gemma 2, Gemma 3, gpt-oss |
| **Window + sinks** | first few + last W | fixed | no | StreamingLLM (inference trick) |
| **DSA (top-k)** | k tokens chosen by the indexer | all kept, but only k read per step | yes | DeepSeek-V3.2 |

Notice that DSA still **keeps** every token's latent in memory (any of them might be chosen), but reads only $$k$$ of them per step. Sliding windows save memory too; DSA saves mainly reading and computing.

All of these still use softmax attention over *some* set of tokens. Part 4 goes one step further: layers that replace softmax attention with a fixed-size **memory** that never grows at all (linear attention, Gated DeltaNet), and the hybrid models that mix them with ordinary attention.

## Summary

- Full attention compares every token with every earlier token: $$T(T+1)/2$$ comparisons per head per layer, **8.6 billion** at 128K tokens.
- **Sliding-window attention** adds one condition to the mask, $$0 \le i - j < W$$. I checked it against PyTorch ($$4.4 \times 10^{-16}$$) and showed that tokens outside the window have exactly zero effect.
- Stacked layers still see far: the reach is $$L(W-1)+1$$, measured exactly with gradients for 1 to 6 layers.
- A **rolling buffer** stores token $$i$$ in slot $$i \bmod W$$. It matched full computation to $$1.1 \times 10^{-15}$$, cut Mistral 7B's cache **8×** at 32K, and kept a decode step flat (0.031 ms vs 3.893 ms at 128K).
- **Local + global layers** keep some full-attention layers for direct long-range lookups. Gemma 3 27B: **62 GiB → 10.4 GiB** at 128K.
- A model trained with full attention **breaks** under a plain window (perplexity 17.4 → up to 509) because it loses its **attention sink**. Keeping 4 first tokens fixes most of it (17.80).
- **DeepSeek Sparse Attention** uses a tiny **lightning indexer**, $$I_{t,s} = \sum_j w^I_{t,j}\,\operatorname{ReLU}(q^I_{t,j} \cdot k^I_s)$$, to pick the top-k tokens, so the main attention costs $$O(Lk)$$ instead of $$O(L^2)$$.
- My indexer for Qwen2.5-0.5B, trained only with the warm-up KL loss, caught **77.6%** of the real attention with 64 tokens and brought perplexity to **17.62** with 256 tokens (full: 17.39).

<details>
<summary>Run it yourself</summary>

- [`code/attention/part3_sparse.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part3_sparse.py): sliding-window checks, receptive field, rolling cache, KV memory, decode timing. Runs in seconds; the timing part uses a GPU if one is available.
- [`code/attention/part3_qwen.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part3_qwen.py): Qwen2.5-0.5B with windows and sinks, plus training and testing the 24 lightning indexers. Downloads the model (about 1 GB) and two public-domain books from Project Gutenberg. About 2 minutes on an Apple M5 Pro.

```bash
pip install torch transformers
python part3_sparse.py     # writes results/part3.json
python part3_qwen.py       # writes results/part3_qwen.json
```

</details>

## References

1. A. Q. Jiang et al. [*Mistral 7B*](https://arxiv.org/abs/2310.06825). 2023.
2. I. Beltagy, M. E. Peters, A. Cohan. [*Longformer: The Long-Document Transformer*](https://arxiv.org/abs/2004.05150). 2020.
3. R. Child, S. Gray, A. Radford, I. Sutskever. [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509). 2019.
4. G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. [*Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453). ICLR 2024.
5. Gemma Team. [*Gemma 2: Improving Open Language Models at a Practical Size*](https://arxiv.org/abs/2408.00118). 2024.
6. Gemma Team. [*Gemma 3 Technical Report*](https://arxiv.org/abs/2503.19786). 2025.
7. DeepSeek-AI. [*DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*](https://arxiv.org/abs/2512.02556). 2025.
8. Model configurations: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it), [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
9. Texts: [*Pride and Prejudice*](https://www.gutenberg.org/ebooks/1342) and [*The Adventures of Sherlock Holmes*](https://www.gutenberg.org/ebooks/1661), Project Gutenberg.
