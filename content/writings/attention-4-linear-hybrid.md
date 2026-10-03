---
title: "Attention, Part 4: Linear Attention, Gated DeltaNet and Hybrid Models"
description: "The newest idea in attention: replace the ever-growing KV cache with a fixed-size memory. Linear attention, the delta rule, forget gates, Gated DeltaNet, Kimi Delta Attention, gated attention, and the hybrid models built from them (Qwen3-Next, Kimi Linear, Nemotron-H). Explained from zero with equations, paper screenshots and real runs, including three small models trained side by side."
date: 2026-10-04
tags: [attention, linear-attention, gated-deltanet, llm]
series: "Attention, From the Ground Up"
series_part: 4
motif: waves
accent: "#9b8cff"
---

In [Part 2](attention-2-mqa-gqa-mla.md), each token stored **less**. In [Part 3](attention-3-sliding-window-sparse.md), each token **looked at fewer** earlier tokens. Both kept the same basic picture: a list of keys and values that grows by one entry for every token.

This final part changes that picture. The idea: **give each layer a memory of fixed size**, and update it once per token, the way a person keeps notes on one page instead of re-reading the whole book. A 1,000-token chat and a 1,000,000-token chat then need **the same memory** in those layers.

That idea is old (2020), but it only became good enough for frontier models in 2025, thanks to two additions: a smarter way to **write** (the delta rule) and a way to **forget** (gates). Today Qwen3-Next, Kimi Linear and Nemotron-H all use it for **most** of their layers, keeping only a few ordinary attention layers.

> [!TIP] What you will learn
> 1. **Gated attention:** a one-line fix to ordinary attention (used in Qwen3-Next).
> 2. **Linear attention:** remove softmax, and attention becomes a fixed-size memory.
> 3. **Why that memory gets confused**, shown with a tiny example and a capacity test.
> 4. **The delta rule:** write by *correcting*, not just adding.
> 5. **Forget gates**, and **Gated DeltaNet**, which combines both (checked against Qwen3-Next's own code).
> 6. **Kimi Delta Attention:** a forget rate for every channel.
> 7. **Hybrid models:** why the best models still keep a few full-attention layers.
> 8. **An experiment:** three small models, identical except for their attention, trained on the same data.

As in every part: each new word gets a yellow box, each equation gets a list explaining every symbol, every quote is checked against the paper, and every number comes from code I ran.

> [!TIP] Quick recap
> **Softmax attention:** each token compares its **query** with the **keys** of all earlier tokens, turns the scores into weights with **softmax**, and mixes the **values**. While writing, the model keeps all earlier keys and values in the **KV cache**, which grows by one token every step ([Part 2](attention-2-mqa-gqa-mla.md)).

## 1. Gated attention: a small fix for softmax attention

Before replacing attention, here is the one improvement to ordinary attention that the newest hybrid models use.

Recall the **attention sink** from [Part 1](attention-1-self-attention.md) and [Part 3](attention-3-sliding-window-sparse.md): softmax forces every token's weights to add up to 1, so a head that has nothing useful to say still has to put its weight *somewhere*. Trained models learn to dump it on the first token. In Part 3, cutting that first token off broke a model completely.

A 2025 paper from the Qwen team tested a simple fix: let each head **turn its own output down** when it has nothing useful.

> [!DEFINITION] Sigmoid and gate
> The **sigmoid** function $$\sigma(x) = \dfrac{1}{1 + e^{-x}}$$ squeezes any number into the range 0 to 1: large negative numbers become almost 0, large positive numbers almost 1. A **gate** is a number between 0 and 1 that multiplies a signal: 1 lets it through fully, 0 blocks it, 0.5 lets half through. Gates are usually made with sigmoid.

The gate is applied right after attention, before the output matrix:

$$
Y' = Y \odot \sigma(X W_\theta)
$$

where:

- $$Y$$ is the normal attention output (what Part 1 called "the weighted mix of values"), one vector per token and head;
- $$X$$ is the layer's input (the token vectors);
- $$W_\theta$$ is a new learned matrix, so the gate is computed from the token itself;
- $$\sigma$$ is the sigmoid, so every gate value is between 0 and 1;
- $$\odot$$ means "multiply number by number" (each output number has its own gate);
- $$Y'$$ is the gated output, which then goes through the usual output matrix $$W_o$$.

Here is that equation as it appears in the paper:

<figure class="fig paper"><img src="/img/attention/paper-eq-gate.png" width="558" alt="Equation 5 from the Gated Attention paper: Y prime equals g of Y, X, W theta, sigma, which equals Y elementwise times sigma of X W theta" loading="lazy" /><figcaption>Equation (5) of Qiu et al., 2025, "Gated Attention for Large Language Models".</figcaption></figure>

<figure class="fig"><svg viewBox="0 0 760 236" role="img" aria-label="Gated attention: the attention output Y is multiplied elementwise by a sigmoid gate computed from the input, before the output projection."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="70.0" width="90.0" height="44.0" rx="10"/><text class="t-math" x="65.0" y="97.0" text-anchor="middle">x</text><line class="edge" x1="112.0" y1="92.0" x2="160.0" y2="92.0" marker-end="url(#ah)"/><rect class="box-1" x="160.0" y="70.0" width="140.0" height="44.0" rx="10"/><text class="t-note" x="230.0" y="90.0" text-anchor="middle">attention</text><text class="t-tick" x="230.0" y="106.0" text-anchor="middle">(softmax)</text><line class="edge" x1="302.0" y1="92.0" x2="380.0" y2="92.0" marker-end="url(#ah)"/><text class="t-math" x="340.0" y="84.0" text-anchor="middle">Y</text><circle class="node" cx="400" cy="92" r="18"/><text class="t-note" x="400.0" y="98.0" text-anchor="middle">×</text><path class="edge" d="M65,116 V170 H400 V112" fill="none" marker-end="url(#ah)"/><rect class="box-2" x="250.0" y="150.0" width="120.0" height="40.0" rx="8"/><text class="t-note" x="310.0" y="175.0" text-anchor="middle">σ(x Wθ)</text><line class="edge" x1="420.0" y1="92.0" x2="470.0" y2="92.0" marker-end="url(#ah)"/><rect class="box-3" x="470.0" y="70.0" width="110.0" height="44.0" rx="10"/><text class="t-note" x="525.0" y="97.0" text-anchor="middle">× Wo</text><line class="edge" x1="582.0" y1="92.0" x2="620.0" y2="92.0" marker-end="url(#ah)"/><text class="t-note" x="630.0" y="97.0" text-anchor="start">output</text><text class="t-tick" x="20.0" y="30.0" text-anchor="start">Gated attention: each output number is multiplied by a gate between 0 and 1, computed from the token x.</text><text class="t-tick" x="20.0" y="222.0" text-anchor="start">A gate near 0 lets a head say "nothing useful here" without dumping attention on the first token.</text></svg><figcaption>Gated attention. The attention output Y is multiplied, number by number, by a gate between 0 and 1 computed from the input x. Only then does it go through the output matrix.</figcaption></figure>

> [!QUOTE] Gated Attention paper
> "Our central finding is that a simple modification ... applying a head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA) ... consistently improves performance."
>
> On attention sinks: "The baseline model suffers from a significant attention sink, with an average of 46.7% of attention scores across layers directed towards the first token. Introducing a gate effectively alleviates this, reducing the proportion to 4.8%."
>
> Source: [Qiu et al., 2025](https://arxiv.org/abs/2505.06708)

<figure class="fig paper"><img src="/img/attention/paper-gated-attention.png" width="841" alt="arXiv page of Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free, with authors and abstract" loading="lazy" /><figcaption>The paper on arXiv (2505.06708). They tested 30 variants on models up to 15B parameters trained on 3.5 trillion tokens.</figcaption></figure>

Why does it help with sinks? With a gate, a head can say "nothing here" by setting its gate near 0. It no longer needs the trick of parking its attention on the first token.

**It is in a real model.** Qwen3-Next's attention layers do exactly this. From its code in the Hugging Face `transformers` library, the query projection produces both the query and the gate, and the gate is applied right after attention:

```python
query_states, gate = torch.chunk(
    self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
)
...
attn_output = attn_output.reshape(*input_shape, -1).contiguous()
attn_output = attn_output * torch.sigmoid(gate)

attn_output = self.o_proj(attn_output)
```

We will measure the sink effect ourselves in section 10.

## 2. Linear attention: remove softmax, get a fixed-size memory

Now the big idea. Look again at the attention formula without the mask:

$$
\text{output} = \operatorname{softmax}\!\left(QK^\top\right) V
$$

Softmax sits **between** $$QK^\top$$ and $$V$$, so we must first build the full $$T \times T$$ grid of scores. What if there were no softmax?

> [!DEFINITION] Associativity
> For matrix multiplication, the brackets can move: $$(AB)C = A(BC)$$. The answer is the same; only the amount of work changes. For example, $$(2 \times 3) \times 4 = 2 \times (3 \times 4) = 24$$.

Without softmax, $$(QK^\top)V = Q(K^\top V)$$. And $$K^\top V$$ is only $$d \times d$$, whatever the length $$T$$:

<figure class="fig"><svg viewBox="0 0 760 272" role="img" aria-label="Matrix multiplication is associative: (Q K transposed) V equals Q (K transposed V). The second order never builds the T by T grid."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="190.0" y="24.0" text-anchor="middle">softmax: (Q Kᵀ) first</text><text class="t-title" x="580.0" y="24.0" text-anchor="middle">linear: (Kᵀ V) first</text><rect class="box-2" x="40.0" y="50.0" width="40.0" height="120.0" rx="4"/><text class="t-math" x="60.0" y="190.0" text-anchor="middle">Q</text><text class="t-tick" x="60.0" y="206.0" text-anchor="middle">T × d</text><text class="t-note" x="95.0" y="115.0" text-anchor="middle">×</text><rect class="box-1" x="110.0" y="90.0" width="120.0" height="40.0" rx="4"/><text class="t-math" x="170.0" y="150.0" text-anchor="middle">Kᵀ</text><text class="t-tick" x="170.0" y="166.0" text-anchor="middle">d × T</text><text class="t-note" x="245.0" y="115.0" text-anchor="middle">=</text><rect class="box" x="260.0" y="50.0" width="120.0" height="120.0" rx="4"/><text class="t-note" x="320.0" y="115.0" text-anchor="middle">T × T</text><text class="t-tick" x="320.0" y="190.0" text-anchor="middle">grows with T²</text><rect class="box-1" x="440.0" y="90.0" width="120.0" height="40.0" rx="4"/><text class="t-math" x="500.0" y="150.0" text-anchor="middle">Kᵀ</text><text class="t-tick" x="500.0" y="166.0" text-anchor="middle">d × T</text><text class="t-note" x="575.0" y="115.0" text-anchor="middle">×</text><rect class="box-3" x="590.0" y="50.0" width="40.0" height="120.0" rx="4"/><text class="t-math" x="610.0" y="190.0" text-anchor="middle">V</text><text class="t-tick" x="610.0" y="206.0" text-anchor="middle">T × d</text><text class="t-note" x="645.0" y="115.0" text-anchor="middle">=</text><rect class="box" x="660.0" y="90.0" width="40.0" height="40.0" rx="4"/><text class="t-note" x="680.0" y="150.0" text-anchor="middle">d × d</text><text class="t-tick" x="680.0" y="166.0" text-anchor="middle">fixed size</text><text class="t-tick" x="20.0" y="240.0" text-anchor="start">Same numbers, different order. With softmax in the middle you must build the T × T grid;</text><text class="t-tick" x="20.0" y="258.0" text-anchor="start">without softmax you can multiply Kᵀ V first and only ever keep a small d × d matrix.</text></svg><figcaption>Same numbers, different order. Left: softmax attention must build the T × T grid, which grows with the square of the text length. Right: without softmax, Kᵀ V is computed first, and it is a small d × d matrix whose size never changes.</figcaption></figure>

But we cannot just delete softmax: softmax made all weights **positive**, and that matters. Katharopoulos et al. (2020) replaced it with a **feature map**.

> [!DEFINITION] Feature map φ
> A simple function applied to every query and key before comparing them, chosen so the results are always positive. The common choice is $$\phi(x) = \operatorname{elu}(x) + 1$$, which equals $$x + 1$$ for positive $$x$$ and $$e^x$$ for negative $$x$$, so it is always above 0.

Linear attention for token $$t$$ is then:

$$
o_t = \frac{\sum_{i \le t} \big(\phi(q_t) \cdot \phi(k_i)\big)\, v_i}{\sum_{i \le t} \phi(q_t) \cdot \phi(k_i)}
$$

where:

- $$q_t$$ is the current token's query, and $$k_i, v_i$$ are the key and value of an earlier token $$i$$;
- $$\phi(q_t) \cdot \phi(k_i)$$ is the (always positive) score, replacing $$e^{q_t \cdot k_i}$$;
- the bottom line divides by the total score, so the weights still add up to 1, just like softmax did.

Now move the brackets. Because $$\phi(q_t)$$ is the same for every $$i$$, it can come out of the sum:

$$
o_t = \frac{\phi(q_t)^\top S_t}{\phi(q_t)^\top z_t},
\qquad
S_t = \sum_{i \le t} \phi(k_i)\, v_i^\top,
\qquad
z_t = \sum_{i \le t} \phi(k_i)
$$

where:

- $$S_t$$ is a $$d_k \times d_v$$ matrix: **the memory**. It is the sum of every token's key-value pair;
- $$z_t$$ is a vector of length $$d_k$$ (the running total used for dividing);
- $$\phi(k_i)\, v_i^\top$$ is an **outer product**.

> [!DEFINITION] Outer product
> Multiplying a column vector by a row vector, $$a\,b^\top$$, gives a whole matrix: entry $$(r, c)$$ is $$a_r \times b_c$$. It is how one key-value pair is "written" into a matrix memory: the key decides *where*, the value decides *what*.

And here is the punchline: $$S_t$$ can be **updated one token at a time**:

$$
S_t = S_{t-1} + \phi(k_t)\, v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t)
$$

That is an **RNN**: a fixed-size state, updated once per token. Each step costs the same, no matter how long the text is, and there is no KV cache that grows.

> [!DEFINITION] RNN and state
> A **recurrent neural network** reads a sequence one step at a time and carries a fixed-size **state** from step to step. Each new input updates the state; the output is read from the state. Part 1 met RNNs as the thing attention replaced; linear attention brings them back in a new form.

<figure class="fig"><svg viewBox="0 0 780 280" role="img" aria-label="Softmax attention keeps every key and value in a growing list; linear attention and its successors keep one fixed-size memory matrix and update it once per token."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Softmax attention: a list that keeps growing</text><text class="t-title" x="20.0" y="168.0" text-anchor="start">Linear attention / Gated DeltaNet: one fixed-size memory</text><rect class="box-3" x="20.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="46.0" y="58.0" text-anchor="middle">k1 v1</text><rect class="box-3" x="82.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="108.0" y="58.0" text-anchor="middle">k2 v2</text><rect class="box-3" x="144.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="170.0" y="58.0" text-anchor="middle">k3 v3</text><rect class="box-3" x="206.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="232.0" y="58.0" text-anchor="middle">k4 v4</text><rect class="box-3" x="268.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="294.0" y="58.0" text-anchor="middle">k5 v5</text><rect class="box-3" x="330.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="356.0" y="58.0" text-anchor="middle">k6 v6</text><rect class="box-3" x="392.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="418.0" y="58.0" text-anchor="middle">k7 v7</text><rect class="box-3" x="454.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="480.0" y="58.0" text-anchor="middle">k8 v8</text><rect class="box-3" x="516.0" y="40.0" width="52.0" height="26.0" rx="5"/><text class="t-tick" x="542.0" y="58.0" text-anchor="middle">k9 v9</text><text class="t-tick" x="590.0" y="58.0" text-anchor="start">… + one more per token</text><text class="t-tick" x="20.0" y="92.0" text-anchor="start">Each new token reads ALL stored keys and values. Memory and work grow with the text.</text><text class="t-tick" x="20.0" y="112.0" text-anchor="start">Nothing is ever forgotten, so recall is exact, but at 1M tokens the list is huge.</text><rect class="box-1" x="20.0" y="186.0" width="120.0" height="76.0" rx="8"/><text class="t-note" x="80.0" y="222.0" text-anchor="middle">memory S</text><text class="t-tick" x="80.0" y="242.0" text-anchor="middle">d × d numbers</text><line class="path" x1="150.0" y1="224.0" x2="210.0" y2="224.0" marker-end="url(#ah-on)"/><rect class="box" x="210.0" y="196.0" width="150.0" height="56.0" rx="8"/><text class="t-tick" x="285.0" y="220.0" text-anchor="middle">write: add or</text><text class="t-tick" x="285.0" y="238.0" text-anchor="middle">correct one fact</text><line class="path" x1="362.0" y1="224.0" x2="420.0" y2="224.0" marker-end="url(#ah-on)"/><rect class="box-1" x="420.0" y="186.0" width="120.0" height="76.0" rx="8"/><text class="t-note" x="480.0" y="222.0" text-anchor="middle">memory S</text><text class="t-tick" x="480.0" y="242.0" text-anchor="middle">same size</text><text class="t-tick" x="560.0" y="214.0" text-anchor="start">Every token does one small</text><text class="t-tick" x="560.0" y="232.0" text-anchor="start">update. The size never changes,</text><text class="t-tick" x="560.0" y="250.0" text-anchor="start">so old facts must share space.</text></svg><figcaption>Two kinds of memory. Softmax attention keeps every key and value in a list that grows forever. Linear attention (and everything after it in this part) keeps one fixed-size matrix and updates it once per token.</figcaption></figure>

> [!QUOTE] Transformers are RNNs (Katharopoulos et al., 2020)
> "we express the self-attention as a linear dot-product of kernel feature maps and make use of the associativity property of matrix products to reduce the complexity from O(N²) to O(N)"
>
> "Our linear transformers achieve similar performance to vanilla transformers and they are up to 4000x faster on autoregressive prediction of very long sequences."
>
> Source: [Katharopoulos et al., 2020](https://arxiv.org/abs/2006.16236)

<figure class="fig paper"><img src="/img/attention/paper-linear-transformers.png" width="841" alt="arXiv page of Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" loading="lazy" /><figcaption>The 2020 paper that started linear attention for transformers (arXiv 2006.16236).</figcaption></figure>

### Proof: the two forms give the same answer

I wrote both forms in `part4_linear.py`: the **parallel** one (build the masked $$T \times T$$ grid, like attention) and the **recurrent** one (one token at a time, keeping only $$S$$ and $$z$$):

```python
def linear_attention_parallel(q, k, v):
    """Like causal attention, but exp(q.k) is replaced by phi(q).phi(k), and there is no softmax."""
    Q, K = phi(q), phi(k)
    scores = (Q @ K.T).tril()                                  # (T, T), future set to 0
    return (scores @ v) / scores.sum(-1, keepdim=True)


def linear_attention_recurrent(q, k, v):
    """The same thing as an RNN: a fixed-size state S (d_k x d_v) and a normaliser z (d_k)."""
    S = torch.zeros(k.shape[-1], v.shape[-1], dtype=q.dtype)
    z = torch.zeros(k.shape[-1], dtype=q.dtype)
    out = []
    for t in range(q.shape[0]):
        S = S + torch.outer(phi(k[t]), v[t])                   # write: add this token's key-value pair
        z = z + phi(k[t])
        out.append((phi(q[t]) @ S) / (phi(q[t]) @ z))          # read: one small matrix-vector product
    return torch.stack(out)
```

```text
1. Linear attention, parallel form vs recurrent form (T = 200): max |diff| = 4.4e-16
   the recurrent form keeps only 272 numbers, however long the text
```

Identical (the difference is rounding noise). And the recurrent form kept just $$16 \times 16 + 16 = 272$$ numbers for 200 tokens. For 200,000 tokens it would still be 272.

> [!NOTE] Training uses the parallel form, writing uses the recurrent form
> During **training**, the whole text is known in advance, so models use a parallel (or "chunked") form that keeps the GPU busy. During **generation**, they switch to the recurrent form: one small update per new token. Same math, two schedules.

## 3. The catch: a fixed-size memory gets confused

A memory that never grows sounds too good to be true. It is. Softmax attention keeps every key and value **separately**, so it can always find an old fact exactly. Linear attention **adds everything into one matrix**, and things start to blur.

To see why, write a few facts into $$S$$ (I drop $$\phi$$ and the divisor here to keep it simple) and read one back with its own key $$k_i$$:

$$
S\,k_i \;=\; \sum_j v_j\,(k_j \cdot k_i) \;=\; \underbrace{v_i\,(k_i \cdot k_i)}_{\text{the fact you wanted}} \;+\; \underbrace{\sum_{j \ne i} v_j\,(k_j \cdot k_i)}_{\text{noise from every other fact}}
$$

where $$S = \sum_j v_j k_j^\top$$ is the memory (in the "value × key" layout used by the Gated DeltaNet paper), and $$k_j \cdot k_i$$ is how much key $$j$$ overlaps with key $$i$$.

> [!DEFINITION] Interference (crosstalk)
> When facts share one memory, reading one fact also picks up a little of every other fact whose key points in a similar direction. If the keys were perfectly perpendicular (dot product 0), there would be no noise. But a $$d$$-dimensional space has room for only $$d$$ perpendicular directions, so after about $$d$$ facts, interference is unavoidable.

There is a second problem: linear attention can only **add**. It has no way to **change** a fact.

**Test 1: overwriting.** I wrote `x = 1`, then `y = 7`, then `x = 5` (each fact stored as an outer product of a key and a value, with perpendicular keys for x and y), and read x and y back:

```text
2. Write "x = 1", "y = 7", then "x = 5". Read x and y back:
   linear attention: x = 6.00, y = 7.00
   delta rule:       x = 5.00, y = 7.00
```

<figure class="fig"><svg viewBox="0 0 760 222" role="img" aria-label="Overwriting a fact: linear attention returns the sum of the old and new value, the delta rule returns the new value."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Write "x = 1", "y = 7", then "x = 5". What comes back?</text><g class="mark"><title>read x, linear attention: 6.00</title><path class="s2 bar-mark" d="M230,46H511.0Q515.0,46 515.0,50V64Q515.0,68 511.0,68H230Z"/></g><text class="t-tick" x="220.0" y="62.0" text-anchor="end">read x, linear attention</text><text class="t-val" x="523.0" y="62.0" text-anchor="start">6.00</text><g class="mark"><title>read x, delta rule: 5.00</title><path class="s1 bar-mark" d="M230,82H463.5Q467.5,82 467.5,86V100Q467.5,104 463.5,104H230Z"/></g><text class="t-tick" x="220.0" y="98.0" text-anchor="end">read x, delta rule</text><text class="t-val" x="475.5" y="98.0" text-anchor="start">5.00</text><g class="mark"><title>read y, linear attention: 7.00</title><path class="s2 bar-mark" d="M230,118H558.5Q562.5,118 562.5,122V136Q562.5,140 558.5,140H230Z"/></g><text class="t-tick" x="220.0" y="134.0" text-anchor="end">read y, linear attention</text><text class="t-val" x="570.5" y="134.0" text-anchor="start">7.00</text><g class="mark"><title>read y, delta rule: 7.00</title><path class="s1 bar-mark" d="M230,154H558.5Q562.5,154 562.5,158V172Q562.5,176 558.5,176H230Z"/></g><text class="t-tick" x="220.0" y="170.0" text-anchor="end">read y, delta rule</text><text class="t-val" x="570.5" y="170.0" text-anchor="start">7.00</text><line class="axis" x1="467.5" y1="40" x2="467.5" y2="110" stroke-dasharray="4 3"/><text class="t-tick" x="471.5" y="36.0" text-anchor="start">correct x = 5</text><text class="t-tick" x="20.0" y="208.0" text-anchor="start">Linear attention can only ADD, so x becomes 1 + 5 = 6. The delta rule replaces the old value: x = 5.</text></svg><figcaption>Overwriting a fact. Linear attention can only add, so x comes back as 1 + 5 = 6. The delta rule (next section) replaces the old value and returns 5.</figcaption></figure>

Linear attention says **x = 6**: it added the new value on top of the old one.

**Test 2: capacity.** I stored $$n$$ random facts (random unit-length keys, random values) in a $$64 \times 64$$ memory, then read all of them back and measured the error (0 = perfect, 1 = as wrong as the value itself):

```text
3. Store n random facts in a 64 x 64 memory, then read all of them back (relative error, 0 = perfect):
   n =   4: linear attention 0.21   delta rule 0.10
   n =   8: linear attention 0.32   delta rule 0.20
   n =  16: linear attention 0.49   delta rule 0.33
   n =  32: linear attention 0.70   delta rule 0.50
   n =  48: linear attention 0.86   delta rule 0.63
   n =  64: linear attention 1.00   delta rule 0.73
   n =  96: linear attention 1.22   delta rule 0.89
   n = 128: linear attention 1.41   delta rule 0.99
   n = 192: linear attention 1.74   delta rule 1.12
   n = 256: linear attention 2.01   delta rule 1.20
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Error when reading back n random facts from a 64 by 64 memory, for linear attention and the delta rule."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="570" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="200.0" x2="570" y2="200.0"/><text class="t-tick" x="56.0" y="204.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="150.0" x2="570" y2="150.0"/><text class="t-tick" x="56.0" y="154.0" text-anchor="end">1.00</text><line class="grid" x1="64" y1="100.0" x2="570" y2="100.0"/><text class="t-tick" x="56.0" y="104.0" text-anchor="end">1.50</text><line class="grid" x1="64" y1="50.0" x2="570" y2="50.0"/><text class="t-tick" x="56.0" y="54.0" text-anchor="end">2.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">4</text><text class="t-tick" x="232.7" y="270.0" text-anchor="middle">16</text><text class="t-tick" x="401.3" y="270.0" text-anchor="middle">64</text><text class="t-tick" x="570.0" y="270.0" text-anchor="middle">256</text><line class="axis" x1="64" y1="250" x2="570" y2="250"/><polyline class="l2" points="64.0,229.2 148.3,218.0 232.7,200.7 317.0,179.5 366.3,164.5 401.3,149.7 450.7,127.6 485.7,108.7 535.0,75.9 570.0,48.6"/><g class="mark"><title>linear attention, 4: 0.21</title><circle class="s2 ring" cx="64.0" cy="229.2" r="5"/></g><g class="mark"><title>linear attention, 8: 0.32</title><circle class="s2 ring" cx="148.3" cy="218.0" r="5"/></g><g class="mark"><title>linear attention, 16: 0.49</title><circle class="s2 ring" cx="232.7" cy="200.7" r="5"/></g><g class="mark"><title>linear attention, 32: 0.70</title><circle class="s2 ring" cx="317.0" cy="179.5" r="5"/></g><g class="mark"><title>linear attention, 48: 0.86</title><circle class="s2 ring" cx="366.3" cy="164.5" r="5"/></g><g class="mark"><title>linear attention, 64: 1.00</title><circle class="s2 ring" cx="401.3" cy="149.7" r="5"/></g><g class="mark"><title>linear attention, 96: 1.22</title><circle class="s2 ring" cx="450.7" cy="127.6" r="5"/></g><g class="mark"><title>linear attention, 128: 1.41</title><circle class="s2 ring" cx="485.7" cy="108.7" r="5"/></g><g class="mark"><title>linear attention, 192: 1.74</title><circle class="s2 ring" cx="535.0" cy="75.9" r="5"/></g><g class="mark"><title>linear attention, 256: 2.01</title><circle class="s2 ring" cx="570.0" cy="48.6" r="5"/></g><polyline class="l1" points="64.0,239.8 148.3,229.6 232.7,217.3 317.0,199.6 366.3,187.0 401.3,176.9 450.7,161.1 485.7,150.8 535.0,137.7 570.0,130.4"/><g class="mark"><title>delta rule, 4: 0.10</title><circle class="s1 ring" cx="64.0" cy="239.8" r="5"/></g><g class="mark"><title>delta rule, 8: 0.20</title><circle class="s1 ring" cx="148.3" cy="229.6" r="5"/></g><g class="mark"><title>delta rule, 16: 0.33</title><circle class="s1 ring" cx="232.7" cy="217.3" r="5"/></g><g class="mark"><title>delta rule, 32: 0.50</title><circle class="s1 ring" cx="317.0" cy="199.6" r="5"/></g><g class="mark"><title>delta rule, 48: 0.63</title><circle class="s1 ring" cx="366.3" cy="187.0" r="5"/></g><g class="mark"><title>delta rule, 64: 0.73</title><circle class="s1 ring" cx="401.3" cy="176.9" r="5"/></g><g class="mark"><title>delta rule, 96: 0.89</title><circle class="s1 ring" cx="450.7" cy="161.1" r="5"/></g><g class="mark"><title>delta rule, 128: 0.99</title><circle class="s1 ring" cx="485.7" cy="150.8" r="5"/></g><g class="mark"><title>delta rule, 192: 1.12</title><circle class="s1 ring" cx="535.0" cy="137.7" r="5"/></g><g class="mark"><title>delta rule, 256: 1.20</title><circle class="s1 ring" cx="570.0" cy="130.4" r="5"/></g><line class="grid" x1="576.0" y1="48.6" x2="586.0" y2="48.6"/><rect class="s2" x="590.0" y="43.6" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="52.6" text-anchor="start">linear attention: 2.01</text><line class="grid" x1="576.0" y1="130.4" x2="586.0" y2="130.4"/><rect class="s1" x="590.0" y="125.4" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="134.4" text-anchor="start">delta rule: 1.20</text><text class="t-tick" x="317.0" y="290.0" text-anchor="middle">facts stored in a 64 × 64 memory (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">read-back error (0 = perfect)</text></svg><figcaption>Reading back n random facts from a 64 × 64 memory. The error grows as the memory fills up. The delta rule is always better than plain adding, but neither is perfect: a fixed-size memory has a fixed capacity.</figcaption></figure>

Two lessons. Plain adding gets worse with every fact (by 64 facts, the error is as big as the signal). And even the better rule cannot escape the limit: **a fixed-size memory has a fixed capacity**. Keep this in mind; it is why hybrid models exist (section 9).

## 4. The delta rule: write by correcting

The fix for overwriting is old: the **delta rule** (Widrow and Hoff, 1960), brought to transformers as "DeltaNet" (Schlag et al., 2021; Yang et al., 2024). The idea in plain words:

1. Before writing, **look up** what the memory currently says for this key: $$S_{t-1} k_t$$.
2. Compare it with the value you want to store, $$v_t$$. The difference is the **error** ("delta").
3. Change the memory by **just that error**, scaled by a writing strength $$\beta_t$$.

As an equation (memory in "value × key" layout, keys of length 1):

$$
S_t \;=\; S_{t-1} + \beta_t \big(v_t - S_{t-1} k_t\big)\, k_t^\top
\;=\; S_{t-1}\big(I - \beta_t\, k_t k_t^\top\big) + \beta_t\, v_t k_t^\top
$$

where:

- $$S_{t-1} k_t$$ is the **old value** the memory holds for key $$k_t$$;
- $$v_t - S_{t-1} k_t$$ is the **error**: what we want minus what is there;
- $$\beta_t$$, between 0 and 1, is the **writing strength** (1 = replace completely, 0.5 = move halfway);
- $$I$$ is the identity matrix (the matrix that changes nothing); $$I - \beta_t k_t k_t^\top$$ removes part of whatever was stored along the direction $$k_t$$;
- $$\beta_t\, v_t k_t^\top$$ then writes the new value there.

With $$\beta_t = 1$$ and a unit-length key, reading right after writing gives exactly $$S_t k_t = v_t$$: the old value is gone. That is why the test above returned x = 5.

> [!DEFINITION] Gradient descent (one step)
> A way to improve something by repeatedly taking a small step "downhill" on an error. If the error is $$\mathcal{L}$$, a step is: new = old − step size × (slope of $$\mathcal{L}$$). This is how all neural networks are trained.

Here is the beautiful part: the delta rule **is** one step of gradient descent, taken at every token, on the error "how wrong is my memory for this key?". The Kimi Linear paper writes it this way (in their layout, keys and values swap sides):

<figure class="fig paper"><img src="/img/attention/paper-eq-delta-gradient.png" width="411" alt="Equation from the Kimi Linear paper: S t equals S t minus 1 minus beta t times the gradient of L t, which equals I minus beta t k t k t transposed, times S t minus 1, plus beta t k t v t transposed" loading="lazy" /><figcaption>The delta rule as one gradient-descent step, from the Kimi Linear paper (Section 2). The loss being reduced is ½‖Sᵀk − v‖².</figcaption></figure>

$$
\mathcal{L}_t(S) = \tfrac{1}{2}\,\big\lVert S\,k_t - v_t \big\rVert^2
$$

where $$\mathcal{L}_t$$ is the squared error between what the memory returns for key $$k_t$$ and the value $$v_t$$ it should return, and $$\beta_t$$ plays the role of the step size. So the memory is literally **learning while it reads**, one token at a time. This is why this family is sometimes called "test-time training".

In code it is one line:

```python
def write_delta(S, k, v, beta=1.0):
    return S - beta * torch.outer(S @ k - v, k)                # S (I - beta k k^T) + beta v k^T
```

## 5. Forget gates: clearing out old memories

The delta rule fixes one fact at a time. But sometimes the model needs to forget **a lot** at once: a new paragraph, a new topic, a new document. For that, models add a **forget gate**.

> [!DEFINITION] Forget gate (decay)
> A number $$\alpha_t$$ between 0 and 1 that multiplies the whole memory at each step. $$\alpha_t = 1$$: keep everything. $$\alpha_t = 0$$: wipe the memory clean. In between, old facts **fade**: after $$t$$ steps with the same $$\alpha$$, a fact keeps a fraction $$\alpha^t$$ of its strength.

Mamba-2 (2024) can be written as linear attention with exactly this gate. As written in the Gated DeltaNet paper:

<figure class="fig paper"><img src="/img/attention/paper-eq-mamba2.png" width="254" alt="Equation from the Gated DeltaNet paper: S t equals alpha t times S t minus 1 plus v t k t transposed, and o t equals S t q t" loading="lazy" /><figcaption>Mamba-2 as a gated linear attention, from the Gated DeltaNet paper (Section 2).</figcaption></figure>

$$
S_t = \alpha_t\, S_{t-1} + v_t k_t^\top, \qquad o_t = S_t\, q_t
$$

where $$\alpha_t$$ is the forget gate (computed from token $$t$$), $$v_t k_t^\top$$ writes the new fact, and $$o_t = S_t q_t$$ reads the memory with the query.

How fast do facts fade? It depends strongly on $$\alpha$$:

```text
4. Strength of a fact after t more tokens, with forget gate alpha (alpha^t):
   alpha = 0.9  : t=0: 1.000  t=10: 0.349  t=50: 0.005  t=100: 0.000  t=500: 0.000
   alpha = 0.99 : t=0: 1.000  t=10: 0.904  t=50: 0.605  t=100: 0.366  t=500: 0.007
   alpha = 0.999: t=0: 1.000  t=10: 0.990  t=50: 0.951  t=100: 0.905  t=500: 0.606
```

<figure class="fig"><svg viewBox="0 0 760 290" role="img" aria-label="How much of a stored fact remains after t more tokens, for forget gates alpha of 0.9, 0.99 and 0.999."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="240.0" x2="600" y2="240.0"/><text class="t-tick" x="56.0" y="244.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="187.5" x2="600" y2="187.5"/><text class="t-tick" x="56.0" y="191.5" text-anchor="end">0.25</text><line class="grid" x1="64" y1="135.0" x2="600" y2="135.0"/><text class="t-tick" x="56.0" y="139.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="82.5" x2="600" y2="82.5"/><text class="t-tick" x="56.0" y="86.5" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="600" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="260.0" text-anchor="middle">10</text><text class="t-tick" x="162.4" y="260.0" text-anchor="middle">100</text><text class="t-tick" x="326.5" y="260.0" text-anchor="middle">250</text><text class="t-tick" x="600.0" y="260.0" text-anchor="middle">500</text><line class="axis" x1="64" y1="240" x2="600" y2="240"/><polyline class="l1" points="64.0,166.8 107.8,238.9 162.4,240.0 600.0,240.0"/><g class="mark"><title>α = 0.9, 10: 0.35</title><circle class="s1 ring" cx="64.0" cy="166.8" r="5"/></g><g class="mark"><title>α = 0.9, 50: 0.01</title><circle class="s1 ring" cx="107.8" cy="238.9" r="5"/></g><g class="mark"><title>α = 0.9, 100: 0.00</title><circle class="s1 ring" cx="162.4" cy="240.0" r="5"/></g><g class="mark"><title>α = 0.9, 500: 0.00</title><circle class="s1 ring" cx="600.0" cy="240.0" r="5"/></g><polyline class="l2" points="64.0,50.1 107.8,112.9 162.4,163.1 600.0,238.6"/><g class="mark"><title>α = 0.99, 10: 0.90</title><circle class="s2 ring" cx="64.0" cy="50.1" r="5"/></g><g class="mark"><title>α = 0.99, 50: 0.61</title><circle class="s2 ring" cx="107.8" cy="112.9" r="5"/></g><g class="mark"><title>α = 0.99, 100: 0.37</title><circle class="s2 ring" cx="162.4" cy="163.1" r="5"/></g><g class="mark"><title>α = 0.99, 500: 0.01</title><circle class="s2 ring" cx="600.0" cy="238.6" r="5"/></g><polyline class="l3" points="64.0,32.1 107.8,40.2 162.4,50.0 600.0,112.7"/><g class="mark"><title>α = 0.999, 10: 0.99</title><circle class="s3 ring" cx="64.0" cy="32.1" r="5"/></g><g class="mark"><title>α = 0.999, 50: 0.95</title><circle class="s3 ring" cx="107.8" cy="40.2" r="5"/></g><g class="mark"><title>α = 0.999, 100: 0.90</title><circle class="s3 ring" cx="162.4" cy="50.0" r="5"/></g><g class="mark"><title>α = 0.999, 500: 0.61</title><circle class="s3 ring" cx="600.0" cy="112.7" r="5"/></g><line class="grid" x1="606.0" y1="112.7" x2="616.0" y2="112.7"/><rect class="s3" x="620.0" y="107.7" width="10" height="10" rx="2"/><text class="t-note" x="636.0" y="116.7" text-anchor="start">α = 0.999: 0.61</text><line class="grid" x1="606.0" y1="238.6" x2="616.0" y2="238.6"/><rect class="s2" x="620.0" y="233.6" width="10" height="10" rx="2"/><text class="t-note" x="636.0" y="242.6" text-anchor="start">α = 0.99: 0.01</text><line class="grid" x1="606.0" y1="240.0" x2="616.0" y2="255.6"/><rect class="s1" x="620.0" y="250.6" width="10" height="10" rx="2"/><text class="t-note" x="636.0" y="259.6" text-anchor="start">α = 0.9: 0.00</text><text class="t-tick" x="332.0" y="280.0" text-anchor="middle">tokens since the fact was written</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">strength left (α^t)</text></svg><figcaption>How much of a stored fact is left after t more tokens. With α = 0.9 a fact is almost gone after 50 tokens; with α = 0.999 more than half survives 500 tokens. Because α is computed from each token, the model can choose: remember (α near 1) or wipe (α near 0).</figcaption></figure>

The important word is **data-dependent**: $$\alpha_t$$ is computed from token $$t$$ itself. So the model can keep $$\alpha$$ near 1 inside a paragraph and drop it towards 0 at a topic change.

## 6. Gated DeltaNet: correct AND forget

Gating is good at forgetting **everything a bit**. The delta rule is good at changing **one fact exactly**. Gated DeltaNet (Yang, Kautz and Hatamizadeh, ICLR 2025) uses both:

> [!QUOTE] Gated DeltaNet paper
> "gating enables rapid memory erasure while the delta rule facilitates targeted updates."
>
> Source: [Yang, Kautz, Hatamizadeh, 2025](https://arxiv.org/abs/2412.06464)

<figure class="fig paper"><img src="/img/attention/paper-gated-deltanet.png" width="841" alt="arXiv page of Gated Delta Networks: Improving Mamba2 with Delta Rule, by Songlin Yang, Jan Kautz and Ali Hatamizadeh, with the abstract" loading="lazy" /><figcaption>The Gated DeltaNet paper on arXiv (2412.06464).</figcaption></figure>

The **gated delta rule**, as printed in the paper:

<figure class="fig paper"><img src="/img/attention/paper-eq-gated-delta-rule.png" width="565" alt="Equation 10 from the Gated DeltaNet paper: S t equals S t minus 1 times alpha t times I minus beta t k t k t transposed, plus beta t v t k t transposed" loading="lazy" /><figcaption>Equation (10) of the Gated DeltaNet paper: the gated delta rule.</figcaption></figure>

$$
S_t = S_{t-1}\Big(\alpha_t\big(I - \beta_t\, k_t k_t^\top\big)\Big) + \beta_t\, v_t k_t^\top,
\qquad
o_t = S_t\, q_t
$$

where:

- $$S_t$$ is the memory, a $$d_v \times d_k$$ matrix (one per head);
- $$\alpha_t \in (0, 1)$$ is the **forget gate**: shrink the whole memory a little (or a lot);
- $$\beta_t \in (0, 1)$$ is the **writing strength** of the delta rule;
- $$I - \beta_t k_t k_t^\top$$ removes (part of) the old value stored at key $$k_t$$;
- $$\beta_t\, v_t k_t^\top$$ writes the new value at key $$k_t$$;
- $$q_t$$ is the query; $$o_t = S_t q_t$$ reads the memory;
- if $$\alpha_t = 1$$ it becomes the plain delta rule; if $$\beta_t = 0$$ it only forgets.

<figure class="fig"><svg viewBox="0 0 780 214" role="img" aria-label="One step of Gated DeltaNet: decay the memory by alpha, look up the value stored under the key, correct it towards the new value by beta, then read with the query."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One Gated DeltaNet step for token t</text><rect class="box-2" x="20.0" y="44.0" width="166.0" height="86.0" rx="10"/><text class="t-note" x="103.0" y="68.0" text-anchor="middle">1. forget</text><text class="t-tick" x="103.0" y="94.0" text-anchor="middle">S ← αₜ · S</text><text class="t-tick" x="103.0" y="116.0" text-anchor="middle">shrink every old fact</text><line class="edge" x1="188.0" y1="87.0" x2="204.0" y2="87.0" marker-end="url(#ah)"/><rect class="box" x="206.0" y="44.0" width="166.0" height="86.0" rx="10"/><text class="t-note" x="289.0" y="68.0" text-anchor="middle">2. look up</text><text class="t-tick" x="289.0" y="94.0" text-anchor="middle">old = S kₜ</text><text class="t-tick" x="289.0" y="116.0" text-anchor="middle">what is stored at kₜ now?</text><line class="edge" x1="374.0" y1="87.0" x2="390.0" y2="87.0" marker-end="url(#ah)"/><rect class="box-1" x="392.0" y="44.0" width="166.0" height="86.0" rx="10"/><text class="t-note" x="475.0" y="68.0" text-anchor="middle">3. correct</text><text class="t-tick" x="475.0" y="94.0" text-anchor="middle">S ← S + βₜ (vₜ − old) kₜᵀ</text><text class="t-tick" x="475.0" y="116.0" text-anchor="middle">move it toward vₜ</text><line class="edge" x1="560.0" y1="87.0" x2="576.0" y2="87.0" marker-end="url(#ah)"/><rect class="box-3" x="578.0" y="44.0" width="166.0" height="86.0" rx="10"/><text class="t-note" x="661.0" y="68.0" text-anchor="middle">4. read</text><text class="t-tick" x="661.0" y="94.0" text-anchor="middle">oₜ = S qₜ</text><text class="t-tick" x="661.0" y="116.0" text-anchor="middle">answer the query</text><text class="t-tick" x="20.0" y="160.0" text-anchor="start">αₜ (between 0 and 1): how much of the old memory to keep. Near 1 = remember, near 0 = wipe.</text><text class="t-tick" x="20.0" y="180.0" text-anchor="start">βₜ (between 0 and 1): how strongly to write. 1 = fully replace what was stored at kₜ.</text><text class="t-tick" x="20.0" y="200.0" text-anchor="start">Both are computed from the token itself, so the model decides, token by token, what to forget and what to write.</text></svg><figcaption>One Gated DeltaNet step, in four moves: forget a little, look up what is stored at the key, correct it towards the new value, and read the answer with the query.</figcaption></figure>

### Where α and β come from

Both are computed from the token by small learned layers, so the model decides token by token. In Qwen3-Next's code:

```python
beta = b.sigmoid()
# If the model is loaded in fp16, without the .float() here, A might be -inf
g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
```

where:

- `b` and `a` are produced from the token by a learned matrix;
- `beta = sigmoid(b)` keeps $$\beta_t$$ between 0 and 1;
- `g` is $$\log \alpha_t$$, the forget gate stored as a logarithm, and $$\alpha_t = e^{g}$$;
- `softplus(x)` $$= \log(1 + e^x)$$ is always positive and `A_log.exp()` is positive, so `g` is always negative and $$\alpha_t = e^g$$ is always between 0 and 1.

> [!DEFINITION] Softplus
> A smooth version of ReLU: $$\operatorname{softplus}(x) = \log(1 + e^x)$$. It is always positive, close to 0 for very negative $$x$$, and close to $$x$$ for large $$x$$.

A real Gated DeltaNet layer adds a few more parts around this core: queries and keys are normalised to length 1 (so $$k_t k_t^\top$$ behaves well), a **short convolution** mixes each token with its 3 neighbours before the memory step, and the output passes through a norm and an output gate. The Gated DeltaNet paper describes the query and key path as "linear proj., shortconv., SiLU and L2 norm".

### Proof: my version matches Qwen3-Next's own code

I wrote the gated delta rule from scratch, line by line from the equation:

```python
def gated_delta_rule(q, k, v, alpha, beta):
    """S_t = alpha_t * S_{t-1} (I - beta_t k_t k_t^T) + beta_t v_t k_t^T ;  o_t = S_t q_t.
    q, k: (H, T, d_k) L2-normalised; v: (H, T, d_v); alpha, beta: (H, T) in (0, 1)."""
    H, T, dk = k.shape
    S = torch.zeros(H, v.shape[-1], dk, dtype=q.dtype)
    out = []
    for t in range(T):
        kt, vt = k[:, t], v[:, t]
        S = alpha[:, t, None, None] * S                                         # 1. forget a little (gate)
        S = S - beta[:, t, None, None] * torch.einsum('hv,hk->hvk', torch.einsum('hvk,hk->hv', S, kt) - vt, kt)   # 2. correct (delta)
        out.append(torch.einsum('hvk,hk->hv', S, q[:, t]))                       # 3. read
    return torch.stack(out, 1)
```

The `transformers` library ships Qwen3-Next's reference implementation in two versions: a step-by-step one (used when generating) and a **chunked** one (used for long inputs: it processes 64 tokens at a time with clever algebra, giving the same result). I compared mine with both, on 4 heads and 256 tokens of random inputs:

```text
5. Our gated delta rule vs the Qwen3-Next reference code in transformers (float32 inside):
   vs step-by-step version: max |diff| = 3.0e-08
   vs chunked version:      max |diff| = 2.3e-07   (outputs up to 0.17)
```

Differences around $$10^{-7}$$ are the rounding noise of 32-bit numbers, which their code uses internally. So the equation above really is what runs inside Qwen3-Next.

## 7. Kimi Delta Attention: a forget rate for every channel

Kimi Linear (Moonshot AI, 2025) makes one more change. In Gated DeltaNet, $$\alpha_t$$ is **one number per head**: the whole memory of a head fades at the same speed. Kimi Delta Attention (KDA) gives **every channel its own** forget rate.

> [!DEFINITION] Channel and diagonal matrix
> A **channel** is one of the $$d$$ numbers in a vector (one "slot" of the key). A **diagonal matrix** $$\operatorname{Diag}(\alpha_t)$$ has the numbers of $$\alpha_t$$ on its diagonal and zeros elsewhere. Multiplying by it scales channel 1 by $$\alpha_{t,1}$$, channel 2 by $$\alpha_{t,2}$$, and so on: different forgetting speeds for different slots.

<figure class="fig paper"><img src="/img/attention/paper-eq-kda.png" width="692" alt="Equation 1 from the Kimi Linear paper: S t equals I minus beta t k t k t transposed, times Diag of alpha t, times S t minus 1, plus beta t k t v t transposed; o t equals S t transposed q t" loading="lazy" /><figcaption>Equation (1) of the Kimi Linear paper: Kimi Delta Attention. (Their memory is stored as d_k × d_v, the transpose of the Gated DeltaNet layout, so keys and values swap sides.)</figcaption></figure>

$$
S_t = \big(I - \beta_t\, k_t k_t^\top\big)\, \operatorname{Diag}(\alpha_t)\, S_{t-1} + \beta_t\, k_t v_t^\top,
\qquad
o_t = S_t^\top q_t
$$

where everything is as in Gated DeltaNet, except that $$\alpha_t$$ is now a **vector** (one forget rate per key channel) instead of a single number.

> [!QUOTE] Kimi Linear paper
> "While GDN, similar to Mamba2, employs a coarse head-wise forget gate, KDA introduces a channel-wise variant in which each feature dimension maintains an independent forgetting rate"
>
> Source: [Kimi Team, 2025](https://arxiv.org/abs/2510.26692)

Why would that help? Some channels can hold **long-lived** facts (α near 1) while others act as **scratch space** that is quickly cleared (α small), all inside one head.

<figure class="fig paper"><img src="/img/attention/paper-kimi-linear.png" width="841" alt="arXiv page of Kimi Linear: An Expressive, Efficient Attention Architecture, with the abstract" loading="lazy" /><figcaption>The Kimi Linear paper on arXiv (2510.26692).</figcaption></figure>

## 8. Speed and memory: the payoff

Each new token costs a linear layer **one memory update**, the same size every time. Softmax attention instead reads the whole cache. I timed one decode step of one layer, using Qwen3-Next's real layer shapes (gated attention: 16 query heads sharing 2 key/value heads of size 256; Gated DeltaNet: 32 heads with a $$128 \times 128$$ memory each), on an Apple M5 Pro GPU:

```text
7. One decode step on mps (Qwen3-Next layer shapes):
     4096 tokens so far: gated attention 0.097 ms   Gated DeltaNet 0.180 ms
    16384 tokens so far: gated attention 0.284 ms   Gated DeltaNet 0.181 ms
    65536 tokens so far: gated attention 1.238 ms   Gated DeltaNet 0.180 ms
   262144 tokens so far: gated attention 5.831 ms   Gated DeltaNet 0.180 ms
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Time for one decode step of one layer: gated softmax attention over a growing cache versus one Gated DeltaNet state update."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="570" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="176.7" x2="570" y2="176.7"/><text class="t-tick" x="56.0" y="180.7" text-anchor="end">2.00</text><line class="grid" x1="64" y1="103.3" x2="570" y2="103.3"/><text class="t-tick" x="56.0" y="107.3" text-anchor="end">4.00</text><line class="grid" x1="64" y1="30.0" x2="570" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">6.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">4,096</text><text class="t-tick" x="232.7" y="270.0" text-anchor="middle">16,384</text><text class="t-tick" x="401.3" y="270.0" text-anchor="middle">65,536</text><text class="t-tick" x="570.0" y="270.0" text-anchor="middle">262,144</text><line class="axis" x1="64" y1="250" x2="570" y2="250"/><polyline class="l2" points="64.0,246.4 232.7,239.6 401.3,204.6 570.0,36.2"/><g class="mark"><title>gated attention, 4,096: 0.10</title><circle class="s2 ring" cx="64.0" cy="246.4" r="5"/></g><g class="mark"><title>gated attention, 16,384: 0.28</title><circle class="s2 ring" cx="232.7" cy="239.6" r="5"/></g><g class="mark"><title>gated attention, 65,536: 1.24</title><circle class="s2 ring" cx="401.3" cy="204.6" r="5"/></g><g class="mark"><title>gated attention, 262,144: 5.83</title><circle class="s2 ring" cx="570.0" cy="36.2" r="5"/></g><polyline class="l1" points="64.0,243.4 232.7,243.4 401.3,243.4 570.0,243.4"/><g class="mark"><title>Gated DeltaNet, 4,096: 0.18</title><circle class="s1 ring" cx="64.0" cy="243.4" r="5"/></g><g class="mark"><title>Gated DeltaNet, 16,384: 0.18</title><circle class="s1 ring" cx="232.7" cy="243.4" r="5"/></g><g class="mark"><title>Gated DeltaNet, 65,536: 0.18</title><circle class="s1 ring" cx="401.3" cy="243.4" r="5"/></g><g class="mark"><title>Gated DeltaNet, 262,144: 0.18</title><circle class="s1 ring" cx="570.0" cy="243.4" r="5"/></g><line class="grid" x1="576.0" y1="36.2" x2="586.0" y2="36.2"/><rect class="s2" x="590.0" y="31.2" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="40.2" text-anchor="start">gated attention: 5.83</text><line class="grid" x1="576.0" y1="243.4" x2="586.0" y2="243.4"/><rect class="s1" x="590.0" y="238.4" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="247.4" text-anchor="start">Gated DeltaNet: 0.18</text><text class="t-tick" x="317.0" y="290.0" text-anchor="middle">tokens so far (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">milliseconds per step</text></svg><figcaption>Time for one decode step of one layer. Attention grows with the conversation; the Gated DeltaNet step stays flat at 0.18 ms. Below about 10,000 tokens attention is actually faster in this test; beyond that, the fixed-size memory wins by more and more.</figcaption></figure>

Note the honest detail: at 4,096 tokens, attention was **faster** (0.097 vs 0.180 ms). My Gated DeltaNet step is several small unfused operations, and a $$32 \times 128 \times 128$$ memory is not tiny. The advantage appears only once the conversation is long, and then it keeps growing: **32× faster at 262K tokens**.

## 9. Hybrid models: a few attention layers keep recall sharp

Section 3 showed the weakness: a fixed-size memory has fixed capacity, so it cannot recall *every* detail of a long text exactly. Softmax attention can. The solution all three of today's big designs use: **mostly linear layers, plus a few full attention layers**.

<figure class="fig"><svg viewBox="0 0 760 252" role="img" aria-label="Layer patterns of three hybrid models: Qwen3-Next, Kimi Linear and Nemotron-H. Most layers are linear or Mamba-2; only a few are full attention."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="20.0" y="30.0" text-anchor="start">Qwen3-Next-80B (48 layers)</text><g class="mark"><title>layer 1: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="20.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 2: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="32.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 3: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="44.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 4: full attention</title><rect class="s2" x="56.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 5: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="68.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 6: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="80.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 7: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="92.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 8: full attention</title><rect class="s2" x="104.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 9: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="116.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 10: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="128.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 11: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="140.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 12: full attention</title><rect class="s2" x="152.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 13: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="164.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 14: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="176.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 15: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="188.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 16: full attention</title><rect class="s2" x="200.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 17: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="212.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 18: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="224.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 19: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="236.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 20: full attention</title><rect class="s2" x="248.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 21: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="260.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 22: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="272.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 23: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="284.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 24: full attention</title><rect class="s2" x="296.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 25: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="308.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 26: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="320.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 27: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="332.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 28: full attention</title><rect class="s2" x="344.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 29: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="356.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 30: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="368.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 31: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="380.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 32: full attention</title><rect class="s2" x="392.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 33: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="404.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 34: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="416.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 35: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="428.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 36: full attention</title><rect class="s2" x="440.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 37: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="452.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 38: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="464.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 39: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="476.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 40: full attention</title><rect class="s2" x="488.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 41: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="500.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 42: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="512.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 43: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="524.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 44: full attention</title><rect class="s2" x="536.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 45: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="548.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 46: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="560.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 47: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="572.0" y="40" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 48: full attention</title><rect class="s2" x="584.0" y="40" width="10.0" height="26" rx="2"/></g><text class="t-note" x="20.0" y="94.0" text-anchor="start">Kimi Linear 48B (27 layers)</text><g class="mark"><title>layer 1: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="20.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 2: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="32.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 3: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="44.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 4: full attention</title><rect class="s2" x="56.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 5: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="68.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 6: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="80.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 7: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="92.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 8: full attention</title><rect class="s2" x="104.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 9: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="116.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 10: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="128.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 11: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="140.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 12: full attention</title><rect class="s2" x="152.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 13: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="164.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 14: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="176.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 15: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="188.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 16: full attention</title><rect class="s2" x="200.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 17: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="212.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 18: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="224.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 19: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="236.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 20: full attention</title><rect class="s2" x="248.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 21: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="260.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 22: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="272.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 23: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="284.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 24: full attention</title><rect class="s2" x="296.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 25: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="308.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 26: linear (Gated DeltaNet / KDA)</title><rect class="s1" x="320.0" y="104" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 27: full attention</title><rect class="s2" x="332.0" y="104" width="10.0" height="26" rx="2"/></g><text class="t-note" x="20.0" y="158.0" text-anchor="start">Nemotron-H-8B (52 layers)</text><g class="mark"><title>layer 1: Mamba-2</title><rect class="s3" x="20.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 2: MLP only (no token mixing)</title><rect class="box" x="32.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 3: Mamba-2</title><rect class="s3" x="44.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 4: MLP only (no token mixing)</title><rect class="box" x="56.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 5: Mamba-2</title><rect class="s3" x="68.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 6: MLP only (no token mixing)</title><rect class="box" x="80.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 7: Mamba-2</title><rect class="s3" x="92.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 8: full attention</title><rect class="s2" x="104.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 9: MLP only (no token mixing)</title><rect class="box" x="116.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 10: Mamba-2</title><rect class="s3" x="128.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 11: MLP only (no token mixing)</title><rect class="box" x="140.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 12: Mamba-2</title><rect class="s3" x="152.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 13: MLP only (no token mixing)</title><rect class="box" x="164.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 14: Mamba-2</title><rect class="s3" x="176.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 15: MLP only (no token mixing)</title><rect class="box" x="188.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 16: Mamba-2</title><rect class="s3" x="200.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 17: MLP only (no token mixing)</title><rect class="box" x="212.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 18: Mamba-2</title><rect class="s3" x="224.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 19: full attention</title><rect class="s2" x="236.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 20: MLP only (no token mixing)</title><rect class="box" x="248.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 21: Mamba-2</title><rect class="s3" x="260.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 22: MLP only (no token mixing)</title><rect class="box" x="272.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 23: Mamba-2</title><rect class="s3" x="284.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 24: MLP only (no token mixing)</title><rect class="box" x="296.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 25: Mamba-2</title><rect class="s3" x="308.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 26: MLP only (no token mixing)</title><rect class="box" x="320.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 27: Mamba-2</title><rect class="s3" x="332.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 28: MLP only (no token mixing)</title><rect class="box" x="344.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 29: Mamba-2</title><rect class="s3" x="356.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 30: full attention</title><rect class="s2" x="368.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 31: MLP only (no token mixing)</title><rect class="box" x="380.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 32: Mamba-2</title><rect class="s3" x="392.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 33: MLP only (no token mixing)</title><rect class="box" x="404.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 34: Mamba-2</title><rect class="s3" x="416.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 35: MLP only (no token mixing)</title><rect class="box" x="428.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 36: Mamba-2</title><rect class="s3" x="440.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 37: MLP only (no token mixing)</title><rect class="box" x="452.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 38: Mamba-2</title><rect class="s3" x="464.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 39: MLP only (no token mixing)</title><rect class="box" x="476.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 40: Mamba-2</title><rect class="s3" x="488.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 41: full attention</title><rect class="s2" x="500.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 42: MLP only (no token mixing)</title><rect class="box" x="512.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 43: Mamba-2</title><rect class="s3" x="524.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 44: MLP only (no token mixing)</title><rect class="box" x="536.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 45: Mamba-2</title><rect class="s3" x="548.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 46: MLP only (no token mixing)</title><rect class="box" x="560.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 47: Mamba-2</title><rect class="s3" x="572.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 48: MLP only (no token mixing)</title><rect class="box" x="584.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 49: Mamba-2</title><rect class="s3" x="596.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 50: MLP only (no token mixing)</title><rect class="box" x="608.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 51: Mamba-2</title><rect class="s3" x="620.0" y="168" width="10.0" height="26" rx="2"/></g><g class="mark"><title>layer 52: MLP only (no token mixing)</title><rect class="box" x="632.0" y="168" width="10.0" height="26" rx="2"/></g><rect class="s1" x="20" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="38.0" y="237.0" text-anchor="start">linear: Gated DeltaNet or KDA</text><rect class="s3" x="210" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="228.0" y="237.0" text-anchor="start">Mamba-2</text><rect class="s2" x="400" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="418.0" y="237.0" text-anchor="start">full attention</text><rect class="box" x="590" y="226" width="12" height="12" rx="2"/><text class="t-tick" x="608.0" y="237.0" text-anchor="start">MLP only</text></svg><figcaption>Layer patterns from the published configurations. Qwen3-Next: 3 Gated DeltaNet layers, then 1 gated attention layer, repeated (36 + 12). Kimi Linear: 3 KDA layers, then 1 MLA layer (20 + 7). Nemotron-H-8B: 24 Mamba-2 layers, 24 MLP-only layers, and just 4 attention layers. Hover over a layer to see its type.</figcaption></figure>

**Qwen3-Next-80B** (Qwen, 2025). From its configuration on Hugging Face:

```json
{
  "num_hidden_layers": 48,
  "full_attention_interval": 4,
  "num_attention_heads": 16,
  "num_key_value_heads": 2,
  "head_dim": 256,
  "linear_num_key_heads": 16,
  "linear_num_value_heads": 32,
  "linear_key_head_dim": 128,
  "linear_value_head_dim": 128,
  "linear_conv_kernel_dim": 4,
  "max_position_embeddings": 262144
}
```

`full_attention_interval: 4` means every 4th layer is (gated) full attention: 12 of 48. The other 36 are Gated DeltaNet, each with 32 heads holding a $$128 \times 128$$ memory, and a short convolution of 4 tokens.

**Kimi Linear 48B** (Moonshot AI, 2025): 27 layers; layers 4, 8, 12, 16, 20, 24 and 27 are MLA (the compressed attention from [Part 2](attention-2-mqa-gqa-mla.md)), the other 20 are KDA.

> [!QUOTE] Kimi Linear paper
> "Kimi Linear interleaves KDA with periodic full attention layers in a uniform 3:1 ratio. This hybrid structure reduces memory and KV-cache usage by up to 75% during long-sequence generation while preserving global information flow via the full attention layers."
>
> And a surprising detail: "In Kimi Linear, we apply NoPE to all full attention (MLA) layers. This design delegates the entire responsibility for encoding positional information and recency bias (see § 6.1) to the KDA layers."
>
> Source: [Kimi Team, 2025](https://arxiv.org/abs/2510.26692)

> [!DEFINITION] NoPE
> "No Position Encoding": attention layers without RoPE or any other position signal. It works in a hybrid because the KDA layers, which read tokens in order, already know where everything is. The attention layers then only need to find **what** matches, not **where** it is.

**Nemotron-H** (NVIDIA, 2025) uses Mamba-2 (the gated linear attention from section 5) instead of a delta rule:

> [!QUOTE] Nemotron-H paper
> "we replace the majority of self-attention layers in the common Transformer model architecture with Mamba layers that perform constant computation and require constant memory per generated token."
>
> Source: [NVIDIA, 2025](https://arxiv.org/abs/2504.03624)

### How much memory does the hybrid save?

For attention layers the memory grows with the text; for linear layers it is fixed. With 2 bytes per number:

$$
\text{memory} = \underbrace{L_{\text{attn}} \times 2 \times H_{kv} \times d_h \times n \times 2\,\text{B}}_{\text{attention layers: grows with } n} \;+\; \underbrace{L_{\text{lin}} \times H \times d_k \times d_v \times 2\,\text{B}}_{\text{linear layers: fixed}}
$$

where $$n$$ is the number of tokens, $$L_{\text{attn}}$$ and $$L_{\text{lin}}$$ count the two kinds of layers, and the other symbols are the head counts and sizes from the configuration.

```text
6. Memory, 2 bytes per number
   Qwen3-Next-80B at 262144 tokens: 12 attention layers 6.00 GiB (all 48 as attention: 24.00 GiB), plus a fixed 36 MiB of Gated DeltaNet state
   Kimi Linear 48B: 8064 B per token with 7 MLA layers vs 31104 B if all 27 were MLA (74% less), plus a fixed 20 MiB of KDA state
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Memory for one conversation in Qwen3-Next-80B: if all 48 layers were attention, versus the real hybrid with 12 attention layers plus fixed Gated DeltaNet state."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="540" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="195.0" x2="540" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">6.00</text><line class="grid" x1="64" y1="140.0" x2="540" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">12.00</text><line class="grid" x1="64" y1="85.0" x2="540" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">18.00</text><line class="grid" x1="64" y1="30.0" x2="540" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">24.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">4,096</text><text class="t-tick" x="222.7" y="270.0" text-anchor="middle">16,384</text><text class="t-tick" x="381.3" y="270.0" text-anchor="middle">65,536</text><text class="t-tick" x="540.0" y="270.0" text-anchor="middle">262,144</text><line class="axis" x1="64" y1="250" x2="540" y2="250"/><polyline class="l2" points="64.0,246.6 222.7,236.2 381.3,195.0 540.0,30.0"/><g class="mark"><title>all 48 layers attention, 4,096: 0.38</title><circle class="s2 ring" cx="64.0" cy="246.6" r="5"/></g><g class="mark"><title>all 48 layers attention, 16,384: 1.50</title><circle class="s2 ring" cx="222.7" cy="236.2" r="5"/></g><g class="mark"><title>all 48 layers attention, 65,536: 6.00</title><circle class="s2 ring" cx="381.3" cy="195.0" r="5"/></g><g class="mark"><title>all 48 layers attention, 262,144: 24.00</title><circle class="s2 ring" cx="540.0" cy="30.0" r="5"/></g><polyline class="l1" points="64.0,248.8 222.7,246.2 381.3,235.9 540.0,194.7"/><g class="mark"><title>real hybrid (12 attention), 4,096: 0.13</title><circle class="s1 ring" cx="64.0" cy="248.8" r="5"/></g><g class="mark"><title>real hybrid (12 attention), 16,384: 0.41</title><circle class="s1 ring" cx="222.7" cy="246.2" r="5"/></g><g class="mark"><title>real hybrid (12 attention), 65,536: 1.54</title><circle class="s1 ring" cx="381.3" cy="235.9" r="5"/></g><g class="mark"><title>real hybrid (12 attention), 262,144: 6.04</title><circle class="s1 ring" cx="540.0" cy="194.7" r="5"/></g><line class="grid" x1="546.0" y1="30.0" x2="556.0" y2="30.0"/><rect class="s2" x="560.0" y="25.0" width="10" height="10" rx="2"/><text class="t-note" x="576.0" y="34.0" text-anchor="start">all 48 layers attention: 24.00</text><line class="grid" x1="546.0" y1="194.7" x2="556.0" y2="194.7"/><rect class="s1" x="560.0" y="189.7" width="10" height="10" rx="2"/><text class="t-note" x="576.0" y="198.7" text-anchor="start">real hybrid (12 attention): 6.04</text><text class="t-tick" x="302.0" y="290.0" text-anchor="middle">tokens in the conversation (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">memory, GiB</text></svg><figcaption>Memory for one conversation in Qwen3-Next-80B: if all 48 layers were attention (orange) versus the real hybrid (blue). At 262K tokens: 24 GiB versus about 6 GiB.</figcaption></figure>

- **Qwen3-Next at 262K tokens:** 24 GiB if every layer were attention, 6 GiB as built, plus a fixed 36 MiB for all the Gated DeltaNet memories together.
- **Kimi Linear:** 7 MLA layers out of 27 means **74% less** cache per token, matching the paper's "up to 75%".

## 10. Experiment: three small models, one difference

Equations and timings are one thing. Do these layers actually **learn** as well as attention? I trained three small models that are **identical except for their token-mixing layers**:

| Model | Its 4 layers |
|---|---|
| softmax attention | 4 × softmax attention (with RoPE) |
| gated attention | 4 × softmax attention + output gate (section 1) |
| linear attention | 4 × linear attention, $$\phi = \operatorname{elu} + 1$$ (section 2) |

Everything else is the same: 4 layers, width 256, 4 heads, the same feed-forward blocks, the same optimizer, the same random seed, the same data, the same number of steps. Each model has about 3.3 to 3.5 million parameters.

**Task 1: predicting text.** Each model reads public-domain books one byte at a time and learns to predict the next byte (1,500 steps of 32 × 256 bytes, from six books). It is then tested on a book it never saw, *The Adventures of Tom Sawyer*. The score is **bits per byte**: lower is better.

> [!DEFINITION] Bits per byte
> How many yes/no questions, on average, the model would need to guess each next character. A model that knows nothing needs 8 (one byte = 8 bits). Very good large models get to around 1 bit per character on English text.

**Task 2: recall.** The model sees $$n$$ facts (a key and its value at each position), then $$n$$ questions (just a key, in a new order), and must answer each with the right value. Keys come from 1,024 possible tokens and values from 256. Each model trains for 3,000 steps on random sets of 32 to 256 facts, then is tested on fresh ones. This is a simplified, one-step version of the "multi-query associative recall" test from the Zoology paper ([Arora et al., 2023](https://arxiv.org/abs/2312.04927)), which found that recall explains most of the gap between attention and its cheaper rivals.

> [!NOTE] Why not Gated DeltaNet too?
> The script can also train a Gated DeltaNet model and a 3:1 hybrid, using the Qwen3-Next reference code from section 6. But that code is plain PyTorch with a loop inside (the fast version needs special GPU kernels that do not run on a Mac), and after an hour on my laptop those two runs were still going. So this experiment compares the three attention types in the **first half** of this part. Gated DeltaNet's correctness was checked directly in section 6, against Qwen3-Next's own code.

### Result 1: predicting text

```text
softmax  params 3.28M | books: 1.994 bits/byte (558 s)
gated    params 3.54M | books: 1.989 bits/byte (489 s)
linear   params 3.28M | books: 2.267 bits/byte (482 s)
```

<figure class="fig"><svg viewBox="0 0 760 152" role="img" aria-label="Next-byte prediction on a held-out book for three small models that differ only in their token-mixing layers."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><g class="mark"><title>softmax attention: 1.994 bits per byte</title><path class="s2 bar-mark" d="M210,14H458.9581524732Q462.9581524732,14 462.9581524732,18V32Q462.9581524732,36 458.9581524732,36H210Z"/></g><text class="t-tick" x="200.0" y="30.0" text-anchor="end">softmax attention</text><text class="t-val" x="471.0" y="30.0" text-anchor="start">1.994</text><g class="mark"><title>gated attention: 1.989 bits per byte</title><path class="s4 bar-mark" d="M210,50H455.6217961129347Q459.6217961129347,50 459.6217961129347,54V68Q459.6217961129347,72 455.6217961129347,72H210Z"/></g><text class="t-tick" x="200.0" y="66.0" text-anchor="end">gated attention</text><text class="t-val" x="467.6" y="66.0" text-anchor="start">1.989</text><g class="mark"><title>linear attention: 2.267 bits per byte</title><path class="s3 bar-mark" d="M210,86H633.909612276604Q637.909612276604,86 637.909612276604,90V104Q637.909612276604,108 633.909612276604,108H210Z"/></g><text class="t-tick" x="200.0" y="102.0" text-anchor="end">linear attention</text><text class="t-val" x="645.9" y="102.0" text-anchor="start">2.267</text><text class="t-tick" x="210.0" y="140.0" text-anchor="start">bits per byte on a held-out book (lower is better); bars start at 1.6</text></svg><figcaption>Bits per byte on a book none of the models saw during training (lower is better). Softmax and gated attention are almost tied; linear attention is clearly behind.</figcaption></figure>

| Model | Bits per byte (lower is better) |
|---|---|
| softmax attention | 1.994 |
| gated attention | **1.989** |
| linear attention | 2.267 |

- **Gated attention is slightly better than plain softmax** (1.989 vs 1.994). That is the direction the paper reports, but the gap here is tiny, and with one run per model I would not call it a real win at this scale.
- **Linear attention is clearly worse** (2.267, about 14% more bits). Its fixed memory cannot keep exact track of the recent letters and words the way softmax attention can. This is the "fixed memory gets confused" problem from section 3, showing up in real training. It is exactly what the delta rule and gates were invented to fix.

### Result 2: recall

```text
softmax  recall accuracy  32 facts 1.000  64 facts 1.000  128 facts 1.000  256 facts 1.000
gated    recall accuracy  32 facts 0.159  64 facts 0.147  128 facts 0.122  256 facts 0.079
linear   recall accuracy  32 facts 0.998  64 facts 0.993  128 facts 0.956  256 facts 0.781
```

<figure class="fig"><svg viewBox="0 0 760 320" role="img" aria-label="Recall accuracy as the number of stored facts grows, for the three small models (seed 0, 3,000 training steps)."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="270.0" x2="530" y2="270.0"/><text class="t-tick" x="56.0" y="274.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="210.0" x2="530" y2="210.0"/><text class="t-tick" x="56.0" y="214.0" text-anchor="end">0.25</text><line class="grid" x1="64" y1="150.0" x2="530" y2="150.0"/><text class="t-tick" x="56.0" y="154.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="90.0" x2="530" y2="90.0"/><text class="t-tick" x="56.0" y="94.0" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="530" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="290.0" text-anchor="middle">32</text><text class="t-tick" x="219.3" y="290.0" text-anchor="middle">64</text><text class="t-tick" x="374.7" y="290.0" text-anchor="middle">128</text><text class="t-tick" x="530.0" y="290.0" text-anchor="middle">256</text><line class="axis" x1="64" y1="270" x2="530" y2="270"/><polyline class="l2" points="64.0,30.0 219.3,30.0 374.7,30.0 530.0,30.1"/><g class="mark"><title>softmax attention, 32: 1.00</title><circle class="s2 ring" cx="64.0" cy="30.0" r="5"/></g><g class="mark"><title>softmax attention, 64: 1.00</title><circle class="s2 ring" cx="219.3" cy="30.0" r="5"/></g><g class="mark"><title>softmax attention, 128: 1.00</title><circle class="s2 ring" cx="374.7" cy="30.0" r="5"/></g><g class="mark"><title>softmax attention, 256: 1.00</title><circle class="s2 ring" cx="530.0" cy="30.1" r="5"/></g><polyline class="l4" points="64.0,231.7 219.3,234.8 374.7,240.8 530.0,250.9"/><g class="mark"><title>gated attention, 32: 0.16</title><circle class="s4 ring" cx="64.0" cy="231.7" r="5"/></g><g class="mark"><title>gated attention, 64: 0.15</title><circle class="s4 ring" cx="219.3" cy="234.8" r="5"/></g><g class="mark"><title>gated attention, 128: 0.12</title><circle class="s4 ring" cx="374.7" cy="240.8" r="5"/></g><g class="mark"><title>gated attention, 256: 0.08</title><circle class="s4 ring" cx="530.0" cy="250.9" r="5"/></g><polyline class="l3" points="64.0,30.4 219.3,31.8 374.7,40.5 530.0,82.6"/><g class="mark"><title>linear attention, 32: 1.00</title><circle class="s3 ring" cx="64.0" cy="30.4" r="5"/></g><g class="mark"><title>linear attention, 64: 0.99</title><circle class="s3 ring" cx="219.3" cy="31.8" r="5"/></g><g class="mark"><title>linear attention, 128: 0.96</title><circle class="s3 ring" cx="374.7" cy="40.5" r="5"/></g><g class="mark"><title>linear attention, 256: 0.78</title><circle class="s3 ring" cx="530.0" cy="82.6" r="5"/></g><line class="grid" x1="536.0" y1="30.1" x2="546.0" y2="30.1"/><rect class="s2" x="550.0" y="25.1" width="10" height="10" rx="2"/><text class="t-note" x="566.0" y="34.1" text-anchor="start">softmax attention: 1.00</text><line class="grid" x1="536.0" y1="82.6" x2="546.0" y2="82.6"/><rect class="s3" x="550.0" y="77.6" width="10" height="10" rx="2"/><text class="t-note" x="566.0" y="86.6" text-anchor="start">linear attention: 0.78</text><line class="grid" x1="536.0" y1="250.9" x2="546.0" y2="250.9"/><rect class="s4" x="550.0" y="245.9" width="10" height="10" rx="2"/><text class="t-note" x="566.0" y="254.9" text-anchor="start">gated attention: 0.08</text><text class="t-tick" x="297.0" y="310.0" text-anchor="middle">facts to remember (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">share of lookups answered correctly</text></svg><figcaption>Share of lookups answered correctly as the number of facts grows. Softmax attention stays at 100%. Linear attention starts near 100% but falls to 78% at 256 facts: its memory is full. Gated attention (this run) failed to learn the task in 3,000 steps; see the note below.</figcaption></figure>

| Facts to remember | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| softmax attention | 100% | 100% | 100% | 100% |
| linear attention | 99.8% | 99.3% | 95.6% | **78.1%** |
| gated attention (seed 0) | 15.9% | 14.7% | 12.2% | 7.9% |

**Linear attention shows its capacity limit.** With few facts it is nearly perfect, but as the facts pile up, its fixed memory (4 heads of $$64 \times 64$$ per layer) starts mixing them, exactly like the capacity test in section 3. Softmax attention keeps every fact separately and stays at 100%.

> [!WARNING] The gated attention result is a training accident, and I checked
> Gated attention scoring 16% looked wrong, since its attention is the same as softmax's plus a gate. So I ran two extra checks on the recall task:
>
> - **Same model, a different random start (seed 1), 3,000 steps:** softmax 100% and gated attention **100%** at every size (99.99% at 256 facts).
> - **Same start (seed 0), but 6,000 steps:** gated attention **100%** at every size.
>
> So gated attention can learn the task perfectly. With the first random start it simply had not "clicked" yet after 3,000 steps. Small models often learn recall suddenly, after a long flat stretch, and the timing varies from run to run. The honest conclusion: **one run per model is not enough to rank softmax and gated attention**, but it is enough to see linear attention's capacity limit, because that shows up as a smooth decline with more facts, not as a failure to learn.

### Result 3: attention sinks

The gated attention paper says the gate removes attention sinks. I measured how much attention the first token receives in the trained softmax and gated models (queries from position 64 on, averaged per layer):

```text
softmax  first-token attention per softmax layer: 0.003 0.001 0.000 0.000
gated    first-token attention per softmax layer: 0.003 0.000 0.000 0.000
```

Both are near zero: **these tiny models never formed an attention sink at all**, so there was nothing for the gate to remove. Sinks are known to appear in larger models trained for much longer, and my byte-level texts start mid-sentence with no special first token. So this experiment cannot confirm or refute the paper's 46.7% → 4.8% result; it only shows that sinks are not automatic.

## 11. Proof of the runs

Both scripts, with their real outputs:

<figure class="fig"><img src="/img/attention/part4-linear-run.png" alt="Terminal output of part4_linear.py: linear attention equivalence, overwrite test, capacity test, forget gate fading, comparison with Qwen3-Next reference code, memory and decode timing" loading="lazy" /><figcaption>Output of part4_linear.py on an Apple M5 Pro.</figcaption></figure>

<figure class="fig"><img src="/img/attention/part4-train-run.png" alt="Terminal output of part4_train.py for the three small models: parameters, bits per byte, recall accuracy and first-token attention" loading="lazy" /><figcaption>Output of part4_train.py: the three models, each trained from scratch, plus the two extra recall checks.</figcaption></figure>

## The whole series in one table

| Idea | Part | What it cuts | Used by |
|---|---|---|---|
| Multi-head attention | 1 | nothing (the baseline) | every transformer |
| MQA / GQA | 2 | KV cache size: fewer key/value heads | Llama 3, Qwen2.5, Gemma 3, Mistral |
| MLA | 2 | KV cache size: compressed latent | DeepSeek-V2/V3, Kimi K2, Kimi Linear |
| Sliding window | 3 | tokens looked at, and cache | Mistral 7B, Gemma 2/3, gpt-oss |
| DeepSeek Sparse Attention | 3 | tokens looked at (top-k) | DeepSeek-V3.2 |
| Gated attention | 4 | attention sinks (quality fix) | Qwen3-Next |
| Linear attention, Mamba-2 | 4 | the cache itself: fixed-size memory | Nemotron-H (Mamba-2) |
| Gated DeltaNet, KDA | 4 | the cache itself, with better recall | Qwen3-Next, Kimi Linear |
| Hybrids | 4 | most of the cache, keeping exact recall | Qwen3-Next, Kimi Linear, Nemotron-H |

## Summary

- **Gated attention** multiplies each attention output by a sigmoid gate, $$Y' = Y \odot \sigma(XW_\theta)$$. The paper cut first-token attention from 46.7% to 4.8%; Qwen3-Next uses it.
- **Linear attention** replaces softmax with $$\phi(q) \cdot \phi(k)$$. Moving the brackets turns attention into an RNN with a fixed memory $$S_t = S_{t-1} + \phi(k_t) v_t^\top$$. The parallel and recurrent forms matched to $$4.4 \times 10^{-16}$$.
- A fixed memory **gets confused**: plain adding cannot overwrite (x came back as 6 instead of 5), and errors grow as facts pile up.
- The **delta rule** writes by correcting: $$S_t = S_{t-1} + \beta_t (v_t - S_{t-1}k_t) k_t^\top$$, one step of gradient descent per token. It overwrote correctly and stored random facts with less error.
- **Forget gates** $$\alpha_t$$ fade old memories ($$\alpha^t$$), chosen token by token.
- **Gated DeltaNet** combines both: $$S_t = S_{t-1}\,\alpha_t(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top$$. My version matched Qwen3-Next's reference code to $$3 \times 10^{-8}$$. **KDA** gives every channel its own forget rate.
- One Gated DeltaNet decode step stayed at **0.18 ms** at any length; attention grew to 5.8 ms at 262K tokens.
- **Hybrids** keep a few full attention layers for exact recall: Qwen3-Next (36 + 12), Kimi Linear (20 + 7, 74% less cache), Nemotron-H (4 attention layers of 52).
- In my three small models, **linear attention** was clearly worse at predicting text (2.267 vs 1.994 bits per byte) and its recall fell to 78% at 256 facts, while softmax attention stayed at 100%. Gated and plain softmax were too close to rank from single runs.

That is the end of the series. Attention started in 2014 as "let the decoder look back at every word"; in 2025 the best models look back with only a handful of layers, and remember everything else in memories that never grow.

<details>
<summary>Run it yourself</summary>

- [`code/attention/part4_linear.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part4_linear.py): linear attention forms, overwrite and capacity tests, forget-gate fading, the comparison with Qwen3-Next's reference code, memory and decode timing. Runs in about a minute.
- [`code/attention/part4_train.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part4_train.py): the small models, trained on books and on the recall task. Pass model names to choose: `python part4_train.py softmax gated linear` (what this article reports). It can also train `gdn` and `hybrid`, but those are very slow without the special GPU kernels. `SEED=1` changes the random start and `RECALL_ONLY=1` skips the book task, as in the extra checks.

```bash
pip install torch transformers
python part4_linear.py     # writes results/part4.json
python part4_train.py softmax gated linear   # writes results/part4_train.json
```

</details>

## References

1. A. Katharopoulos, A. Vyas, N. Pappas, F. Fleuret. [*Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention*](https://arxiv.org/abs/2006.16236). ICML 2020.
2. I. Schlag, K. Irie, J. Schmidhuber. [*Linear Transformers Are Secretly Fast Weight Programmers*](https://arxiv.org/abs/2102.11174). ICML 2021.
3. S. Yang, B. Wang, Y. Zhang, Y. Shen, Y. Kim. [*Parallelizing Linear Transformers with the Delta Rule over Sequence Length*](https://arxiv.org/abs/2406.06484). NeurIPS 2024.
4. T. Dao, A. Gu. [*Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality*](https://arxiv.org/abs/2405.21060) (Mamba-2). ICML 2024.
5. S. Yang, J. Kautz, A. Hatamizadeh. [*Gated Delta Networks: Improving Mamba2 with Delta Rule*](https://arxiv.org/abs/2412.06464). ICLR 2025.
6. Z. Qiu et al. [*Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free*](https://arxiv.org/abs/2505.06708). 2025.
7. Kimi Team. [*Kimi Linear: An Expressive, Efficient Attention Architecture*](https://arxiv.org/abs/2510.26692). 2025.
8. NVIDIA. [*Nemotron-H: A Family of Accurate and Efficient Hybrid Mamba-Transformer Models*](https://arxiv.org/abs/2504.03624). 2025.
9. S. Arora et al. [*Zoology: Measuring and Improving Recall in Efficient Language Models*](https://arxiv.org/abs/2312.04927). 2023.
10. Model configurations: [Qwen3-Next-80B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct), [Kimi-Linear-48B-A3B-Instruct](https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct), [Nemotron-H-8B-Base-8K](https://huggingface.co/nvidia/Nemotron-H-8B-Base-8K). Qwen3-Next code: Hugging Face `transformers`, `models/qwen3_next/modeling_qwen3_next.py`.
11. Texts: Project Gutenberg books [1342](https://www.gutenberg.org/ebooks/1342), [1661](https://www.gutenberg.org/ebooks/1661), [11](https://www.gutenberg.org/ebooks/11), [84](https://www.gutenberg.org/ebooks/84), [2701](https://www.gutenberg.org/ebooks/2701), [98](https://www.gutenberg.org/ebooks/98) for training and [74](https://www.gutenberg.org/ebooks/74) for testing.
