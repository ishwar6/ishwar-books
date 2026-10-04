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

{{FIG:p4_gate|Gated attention. The attention output Y is multiplied, number by number, by a gate between 0 and 1 computed from the input x. Only then does it go through the output matrix.}}

> [!PAPER] Qiu et al. (2025), Gated Attention for LLMs · Section 1 · page 2, Figure 1
> [![Figure 1 of the gated attention paper: the five positions where a gate was tested, the change in perplexity and MMLU for each, and the training loss with and without the gate](/img/attention/papers/ga-figure1.png)](/img/attention/papers/ga-figure1.png)
>
> **Context:** Where should a gate go? The authors tried five places, marked $$G_1$$ to $$G_5$$ on the left: after the attention output ($$G_1$$), on the values ($$G_2$$), keys ($$G_3$$), queries ($$G_4$$), and after the output matrix ($$G_5$$).
>
> **What it says:** Middle: on 15B mixture-of-experts models, the gate after attention ($$G_1$$) lowered average perplexity by 0.265 and raised MMLU by 2.03 points, the best of the five. Right: on a 1.7B dense model trained on 3.5 trillion tokens, the gated model's training loss (blue) is lower and has far fewer **loss spikes** than the baseline (grey).
>
> **Why it matters:** One extra matrix and a sigmoid, placed at exactly the right spot, improved quality **and** made training more stable. "Our central finding is that a simple modification ... applying a head-specific sigmoid gate after the Scaled Dot-Product Attention (SDPA) ... consistently improves performance."
>
> [Read the paper on arXiv](https://arxiv.org/abs/2505.06708)

> [!DEFINITION] Loss spike
> A sudden jump in the training error, usually caused by a few numbers inside the model growing too large. Spikes waste training time and can ruin a run, so anything that removes them is valuable at large scale.

> [!PAPER] Gated Attention for LLMs · Section 4 · page 3, Figure 2
> [![Figure 2 of the gated attention paper: share of attention on the first token per layer, about 47% on average without the gate and about 5% with it, plus attention maps of layers 21 and 23](/img/attention/papers/ga-figure2.png)](/img/attention/papers/ga-figure2.png)
>
> **Context:** Left column: how much attention each layer puts on the very first token, without the gate (top) and with it (bottom). Right: attention maps (query rows, key columns) of layers 21 and 23.
>
> **What it says:** Without the gate, "an average of 46.7% of attention scores across layers" go to the first token, and layer 21 puts 83% there (the bright left column). With the gate this drops to 4.8% on average, and layer 21 to 4%.
>
> **Why it matters:** This is the attention sink of Parts 1 and 3, removed by design. A model without sinks no longer depends on keeping its first token, which matters for long texts and sliding windows.

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

{{FIG:p4_assoc|Same numbers, different order. Left: softmax attention must build the T × T grid, which grows with the square of the text length. Right: without softmax, Kᵀ V is computed first, and it is a small d × d matrix whose size never changes.}}

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

This is exactly what the linear transformer paper writes (its Equation 5 is the version without the mask; Equation 9 below adds it):

> [!PAPER] Katharopoulos et al. (2020), Transformers are RNNs · Section 3.2 · page 3
> [![Equations 5 and 6 of the Transformers are RNNs paper: linear attention, and the regrouping that computes phi of K transposed times V once and reuses it for every query](/img/attention/papers/kath-linearized.png)](/img/attention/papers/kath-linearized.png)
>
> **Context:** Section 3.2, "Linearized Attention", after the authors replace the softmax similarity with $$\phi(q)^\top \phi(k)$$.
>
> **What it says:** Equation (6) is the bracket move: $$(\phi(Q)\phi(K)^T)V = \phi(Q)(\phi(K)^T V)$$. Softmax attention costs $$O(N^2)$$, but the linear version is $$O(N)$$ because the sums over keys can be computed "once and reuse[d] for every query".
>
> **Why it matters:** That single regrouping is the whole idea of linear attention, and everything in this part (DeltaNet, Mamba-2, Gated DeltaNet, KDA) starts from it.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2006.16236)

The paper also gives the cost in multiplications (Section 3.2.1). With $$N$$ tokens, keys and queries of length $$D$$ and values of length $$M$$:

$$
\text{softmax attention: } O\big(N^2 \max(D, M)\big)
\qquad\qquad
\text{linear attention: } O\big(N\,D\,M\big)
$$

where $$N^2$$ comes from comparing every pair of tokens, and $$D\,M$$ is the size of the memory matrix that each token updates. When the text is longer than the head size ($$N > D$$), linear attention does less work, and the gap grows with $$N$$.

For text written left to right, the sums only run over earlier tokens, which the paper writes like this:

> [!PAPER] Transformers are RNNs · Section 3.3, Causal Masking · page 4
> [![Equations 8 and 9 of the Transformers are RNNs paper: masked attention summing only over earlier positions j up to i, and its linearized version](/img/attention/papers/kath-masked.png)](/img/attention/papers/kath-masked.png)
>
> **Context:** Generating text needs a causal mask (Part 1): token $$i$$ may only use tokens $$j \le i$$.
>
> **What it says:** Equation (8) is ordinary masked attention with a general similarity; Equation (9) is the linearized version, where the sums run from $$j = 1$$ to $$i$$.
>
> **Why it matters:** Because each sum only adds one new term per token, it can be kept as a running total. That is the step that turns attention into an RNN, just below.

And here is the punchline: $$S_t$$ can be **updated one token at a time**:

$$
S_t = S_{t-1} + \phi(k_t)\, v_t^\top, \qquad z_t = z_{t-1} + \phi(k_t)
$$

That is an **RNN**: a fixed-size state, updated once per token. Each step costs the same, no matter how long the text is, and there is no KV cache that grows.

> [!PAPER] Transformers are RNNs · Section 3.4, Transformers are RNNs · page 5
> [![Equations 16 to 20 of the Transformers are RNNs paper: the attention memory s and normalizer memory z updated at every step, and the output y computed from them](/img/attention/papers/kath-rnn.png)](/img/attention/papers/kath-rnn.png)
>
> **Context:** The section that gives the paper its title.
>
> **What it says:** "The resulting RNN has two hidden states, namely the attention memory $$s$$ and the normalizer memory $$z$$." Each step adds $$\phi(x_i W_K)(x_i W_V)^T$$ to $$s$$ (Equation 18) and $$\phi(x_i W_K)$$ to $$z$$ (Equation 19), then reads the output (Equation 20).
>
> **Why it matters:** These are exactly the $$S_t$$ and $$z_t$$ above, in the paper's notation ($$x_i W_K$$ is the key, $$x_i W_V$$ the value, $$x_i W_Q$$ the query). The paper's $$f_l$$ is the rest of the transformer layer (the feed-forward part), and $$+ x_i$$ is the usual "add the input back".

> [!DEFINITION] RNN and state
> A **recurrent neural network** reads a sequence one step at a time and carries a fixed-size **state** from step to step. Each new input updates the state; the output is read from the state. Part 1 met RNNs as the thing attention replaced; linear attention brings them back in a new form.

{{FIG:p4_kinds|Two kinds of memory. Softmax attention keeps every key and value in a list that grows forever. Linear attention (and everything after it in this part) keeps one fixed-size matrix and updates it once per token.}}

> [!PAPER] Transformers are RNNs · Section 4.1 · page 6, Figure 1
> [![Figure 1 of the Transformers are RNNs paper: time and GPU memory for a forward and backward pass against sequence length, for softmax, linear and Reformer attention](/img/attention/papers/kath-figure1.png)](/img/attention/papers/kath-figure1.png)
>
> **Context:** Time and GPU memory for one training step (forward and backward pass), as the sequence grows from $$2^9 = 512$$ to $$2^{16} = 65{,}536$$ tokens. Both axes are logarithmic.
>
> **What it says:** Softmax attention (red, dashed) rises steeply and stops at $$2^{12}$$ tokens, where it runs out of memory. Linear attention (black) rises along a straight line of slope 1: double the length, double the cost.
>
> **Why it matters:** On a log-log plot, slope 2 means $$N^2$$ and slope 1 means $$N$$. This plot is the $$O(N^2)$$ versus $$O(N)$$ formulas above, measured. The abstract's headline: linear transformers are "up to 4000x faster on autoregressive prediction of very long sequences".

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

{{FIG:p4_overwrite|Overwriting a fact. Linear attention can only add, so x comes back as 1 + 5 = 6. The delta rule (next section) replaces the old value and returns 5.}}

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

{{FIG:p4_capacity|Reading back n random facts from a 64 × 64 memory. The error grows as the memory fills up. The delta rule is always better than plain adding, but neither is perfect: a fixed-size memory has a fixed capacity.}}

Two lessons. Plain adding gets worse with every fact (by 64 facts, the error is as big as the signal). And even the better rule cannot escape the limit: **a fixed-size memory has a fixed capacity**. Keep this in mind; it is why hybrid models exist (section 9).

A 2023 study found that this is exactly where cheaper models lose to attention on real text:

> [!PAPER] Arora et al. (2023), Zoology · Section 3 · page 2, Figure 1
> [![Figure 1 of the Zoology paper: perplexity of attention, Hyena and RWKV on tokens that repeat an earlier bigram versus first occurrences, and diagrams of how attention and gated convolutions aggregate information](/img/attention/papers/zoology-figure1.png)](/img/attention/papers/zoology-figure1.png)
>
> **Context:** The authors split real text into two kinds of tokens: "associative recall hits", where the next word repeats a pair of words seen earlier in the same text ("common buzzard ... a common buzzard"), and everything else.
>
> **What it says:** On ordinary tokens (middle plot), attention (blue) and two attention-free models (Hyena, RWKV) are about equally good. On recall hits (left plot), the attention-free models are clearly worse, especially for word pairs that were rare in training. In the abstract's words, "82% of the gap is explained by each model's ability to recall information that is previously mentioned in-context".
>
> **Why it matters:** This is the capacity problem of my 64 × 64 test, showing up in real language: a fixed memory struggles to look up an exact earlier detail. It is the main reason the best models keep a few attention layers.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2312.04927)

> [!DEFINITION] Associative recall
> Remembering what went with what: if the text said "Kim met Tim Rice in March 2018", and later says "Tim", the model should recall "Rice". Attention can look the pair up directly; a fixed-size memory has to have stored it well.

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

{{FIG:p4_fade|How much of a stored fact is left after t more tokens. With α = 0.9 a fact is almost gone after 50 tokens; with α = 0.999 more than half survives 500 tokens. Because α is computed from each token, the model can choose: remember (α near 1) or wipe (α near 0).}}

The important word is **data-dependent**: $$\alpha_t$$ is computed from token $$t$$ itself. So the model can keep $$\alpha$$ near 1 inside a paragraph and drop it towards 0 at a topic change.

## 6. Gated DeltaNet: correct AND forget

Gating is good at forgetting **everything a bit**. The delta rule is good at changing **one fact exactly**. Gated DeltaNet (Yang, Kautz and Hatamizadeh, ICLR 2025) uses both:

> [!PAPER] Yang, Kautz, Hatamizadeh (2025), Gated Delta Networks · Section 1 · page 2
> [![The Gated DeltaNet introduction proposing the gated delta rule, which can clear memory by setting alpha to zero or switch to the pure delta rule by setting alpha to one](/img/attention/papers/gdn-intro.png)](/img/attention/papers/gdn-intro.png)
>
> **Context:** The introduction, right after the authors explain that gating and the delta rule each solve half of the memory problem.
>
> **What it says:** The gated delta rule "can promptly clear memory by setting $$\alpha_t \to 0$$, while selectively updating specific content without affecting other information by setting $$\alpha_t \to 1$$ (effectively switching to the pure delta rule)". The abstract sums it up: "gating enables rapid memory erasure while the delta rule facilitates targeted updates."
>
> **Why it matters:** One rule, two behaviours, chosen token by token. That flexibility is why it beat both of its parents (Mamba-2 and DeltaNet) in their experiments.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2412.06464)

<figure class="fig paper"><img src="/img/attention/paper-gated-deltanet.png" width="841" alt="arXiv page of Gated Delta Networks: Improving Mamba2 with Delta Rule, by Songlin Yang, Jan Kautz and Ali Hatamizadeh, with the abstract" loading="lazy" /><figcaption>The Gated DeltaNet paper on arXiv (2412.06464).</figcaption></figure>

The **gated delta rule**, as printed in the paper:

> [!PAPER] Gated Delta Networks · Section 3.1 · page 4, Equation (10)
> [![Equation 10 of the Gated DeltaNet paper, the gated delta rule, with the sentence saying alpha controls state decay](/img/attention/papers/gdn-rule.png)](/img/attention/papers/gdn-rule.png)
>
> **Context:** The formal definition of the gated delta rule.
>
> **What it says:** $$S_t = S_{t-1}\big(\alpha_t(I - \beta_t k_t k_t^\top)\big) + \beta_t v_t k_t^\top$$, where "the data-dependent gating term $$\alpha_t \in (0, 1)$$ controls state decay".
>
> **Why it matters:** This is the equation that runs in 36 of Qwen3-Next's 48 layers. It is checked against Qwen3-Next's own code further below.

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

{{FIG:p4_gdn|One Gated DeltaNet step, in four moves: forget a little, look up what is stored at the key, correct it towards the new value, and read the answer with the query.}}

### Every rule as "learning while reading"

There is a neat way to see all these memories as one family. At each token, the new memory $$S_t$$ is the answer to a tiny optimisation problem: stay close to the old memory, but fit the new fact.

For plain linear attention:

$$
S_t = \arg\min_S \;\underbrace{\lVert S - S_{t-1} \rVert_F^2}_{\text{stay close}} \;-\; 2\,\underbrace{\langle S\,k_t,\; v_t \rangle}_{\text{point toward } v_t}
\quad\Longrightarrow\quad
S_t = S_{t-1} + v_t k_t^\top
$$

For the delta rule, the second term instead points toward the **error** $$v_t - S_{t-1}k_t$$, scaled by $$\beta_t$$:

$$
S_t = \arg\min_S \;\lVert S - S_{t-1} \rVert_F^2 \;-\; 2\,\big\langle S\,k_t,\; \beta_t (v_t - S_{t-1} k_t) \big\rangle
\quad\Longrightarrow\quad
S_t = S_{t-1}(I - \beta_t k_t k_t^\top) + \beta_t v_t k_t^\top
$$

where:

- $$\arg\min_S$$ means "the $$S$$ that makes this smallest";
- $$\lVert A \rVert_F^2$$ (the squared **Frobenius norm**) is the sum of the squares of all entries of a matrix: here, how much the memory changed;
- $$\langle a, b \rangle$$ is the dot product of two vectors;
- setting the slope (gradient) to zero gives the update on the right. For the first one: $$2(S - S_{t-1}) - 2\,v_t k_t^\top = 0$$, so $$S = S_{t-1} + v_t k_t^\top$$.

The Gated DeltaNet paper lists the whole family in one table:

> [!PAPER] Gated Delta Networks · Section 2 · page 5, Table 1
> [![Table 1 of the Gated DeltaNet paper: the online learning objective and the resulting online update for linear attention, Mamba2, Longhorn, DeltaNet and Gated DeltaNet](/img/attention/papers/gdn-table1.png)](/img/attention/papers/gdn-table1.png)
>
> **Context:** Five recent designs written in the same "online learning" form (after Liu et al., 2024).
>
> **What it says:** Linear attention (LA) and Mamba2 just add $$v_t k_t^\top$$ (Mamba2 first multiplies the old memory by $$\alpha_t$$). DeltaNet and Gated DeltaNet subtract the part of the old memory along $$k_t$$ first, the $$(I - \beta_t k_t k_t^\top)$$ factor. Gated DeltaNet is the only row with both $$\alpha_t$$ and $$\beta_t$$.
>
> **Why it matters:** It shows that these "new architectures" differ only in the small loss each token minimises. Read down the Online Update column and you have sections 2 to 6 of this article in five lines.

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

> [!PAPER] Gated Delta Networks · Section 3.3 · page 7, Figure 1
> [![Figure 1 of the Gated DeltaNet paper: hybrid layer stacks H1 and H2 with sliding window attention and Mamba2, and the block design with linear projections, short convolutions, L2 norm, the gated delta rule, a norm and an output gate](/img/attention/papers/gdn-figure1.png)](/img/attention/papers/gdn-figure1.png)
>
> **Context:** Left: two hybrid stacks the paper tested (H1: Gated DeltaNet + sliding-window attention; H2: Mamba2 + Gated DeltaNet + sliding-window attention). Right: what is inside one Gated DeltaNet block.
>
> **What it says:** Queries and keys go through a linear projection, a short convolution ("Conv"), SiLU and L2 normalisation; values through a projection, convolution and SiLU; $$\alpha$$ and $$\beta$$ through their own small projections. The memory update is the "Gated Delta Rule" box, followed by a norm and an output gate ($$\otimes$$).
>
> **Why it matters:** This is the block Qwen3-Next uses, and the block I trained in my own code. Even the paper already proposed mixing it with attention layers (H1, H2): hybrids were part of the design from the start.

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

Written out entry by entry, the forget step of KDA is

$$
\big(\operatorname{Diag}(\alpha_t)\, S_{t-1}\big)_{ij} = \alpha_{t,i} \cdot (S_{t-1})_{ij}
$$

where $$i$$ indexes the key channels (rows of KDA's $$d_k \times d_v$$ memory) and $$j$$ the value channels. Every entry in row $$i$$ fades at its own rate $$\alpha_{t,i}$$. In Gated DeltaNet the same step is $$\alpha_t \cdot (S_{t-1})_{ij}$$: one rate for the whole head.

> [!PAPER] Kimi Linear · Section 4 · page 6, Figure 3
> [![Figure 3 of the Kimi Linear paper: the model stack with N KDA layers and one MLA layer, each followed by a mixture-of-experts layer, and the inside of a KDA block](/img/attention/papers/kimi-figure3.png)](/img/attention/papers/kimi-figure3.png)
>
> **Context:** The whole Kimi Linear model. Left: the stack, with $$N$$ KDA blocks for every one MLA block ($$N = 3$$), each followed by a mixture-of-experts (MoE) layer. Right: what is inside MoE (top) and KDA (bottom).
>
> **What it says:** The KDA block looks very much like Gated DeltaNet's: projections, short convolutions, L2 norm, the "Kimi Delta Attention" memory, a norm and a sigmoid output gate. The differences are inside the memory update (the per-channel $$\alpha$$) and in small details of the projections.
>
> **Why it matters:** Comparing this with Gated DeltaNet's Figure 1 above shows how quickly the field converged: two independent teams arrived at nearly the same block.

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

{{FIG:p4_decode|Time for one decode step of one layer. Attention grows with the conversation; the Gated DeltaNet step stays flat at 0.18 ms. Below about 10,000 tokens attention is actually faster in this test; beyond that, the fixed-size memory wins by more and more.}}

Note the honest detail: at 4,096 tokens, attention was **faster** (0.097 vs 0.180 ms). My Gated DeltaNet step is several small unfused operations, and a $$32 \times 128 \times 128$$ memory is not tiny. The advantage appears only once the conversation is long, and then it keeps growing: **32× faster at 262K tokens**.

## 9. Hybrid models: a few attention layers keep recall sharp

Section 3 showed the weakness: a fixed-size memory has fixed capacity, so it cannot recall *every* detail of a long text exactly. Softmax attention can. The solution all three of today's big designs use: **mostly linear layers, plus a few full attention layers**.

{{FIG:p4_hybrids|Layer patterns from the published configurations. Qwen3-Next: 3 Gated DeltaNet layers, then 1 gated attention layer, repeated (36 + 12). Kimi Linear: 3 KDA layers, then 1 MLA layer (20 + 7). Nemotron-H-8B: 24 Mamba-2 layers, 24 MLP-only layers, and just 4 attention layers. Hover over a layer to see its type.}}

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

> [!PAPER] Kimi Linear · Section 4 · page 6
> [![The Kimi Linear paragraphs on the hybrid architecture, with the 3 to 1 KDA to MLA ratio and the use of no position encoding in the MLA layers highlighted](/img/attention/papers/kimi-nope.png)](/img/attention/papers/kimi-nope.png)
>
> **Context:** How the KDA and MLA layers are combined.
>
> **What it says:** "Long-context retrieval remains the primary bottleneck for pure linear attention", so they add a few full attention (MLA) layers; "a uniform 3:1 ratio, i.e., repeating 3 KDA layers to 1 full MLA layer, provided the best quality–throughput trade-off". And: "we apply NoPE to all full attention (MLA) layers", leaving all position information to the KDA layers.
>
> **Why it matters:** The first sentence is the Zoology finding (section 3) stated as a design rule. The NoPE choice is a nice side effect of hybrids: the linear layers read in order, so the attention layers do not need RoPE at all, which also removes the RoPE complication from Part 2's MLA.

> [!DEFINITION] NoPE
> "No Position Encoding": attention layers without RoPE or any other position signal. It works in a hybrid because the KDA layers, which read tokens in order, already know where everything is. The attention layers then only need to find **what** matches, not **where** it is.

**Nemotron-H** (NVIDIA, 2025) uses Mamba-2 (the gated linear attention from section 5) instead of a delta rule:

> [!PAPER] NVIDIA (2025), Nemotron-H · Section 2 · page 3, Figure 2
> [![Figure 2 of the Nemotron-H paper: the layer layout of the 8B and 56B models, mostly Mamba-2 and feed-forward layers with a few attention layers spread evenly](/img/attention/papers/nemo-figure2.png)](/img/attention/papers/nemo-figure2.png)
>
> **Context:** The layer layout of the 8B and 56B models, read left to right; "x4" and "x10" mean the middle group is repeated.
>
> **What it says:** "Roughly 8% of the total layers in the model are self-attention layers; these layers are evenly dispersed throughout the model. The rest of the model is made up of alternating Mamba-2 and FFN layers." The abstract adds that the Mamba layers "perform constant computation and require constant memory per generated token".
>
> **Why it matters:** Nemotron-H is the most extreme hybrid of the three: only 4 attention layers out of 52 in the 8B model. It shows how few attention layers a strong model can get away with.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2504.03624)

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

{{FIG:p4_memory|Memory for one conversation in Qwen3-Next-80B: if all 48 layers were attention (orange) versus the real hybrid (blue). At 262K tokens: 24 GiB versus about 6 GiB.}}

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

{{FIG:p4_lm|Bits per byte on a book none of the models saw during training (lower is better). Softmax and gated attention are almost tied; linear attention is clearly behind.}}

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

{{FIG:p4_recall|Share of lookups answered correctly as the number of facts grows. Softmax attention stays at 100%. Linear attention starts near 100% but falls to 78% at 256 facts: its memory is full. Gated attention (this run) failed to learn the task in 3,000 steps; see the note below.}}

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

## The impact, and where you meet it

In 2024, linear attention was mostly a research topic. By late 2025, three major model families shipped it in most of their layers. The results the companies reported are what made the difference:

> [!PAPER] Kimi Linear · Abstract and Section 5 · page 1, Figure 1
> [![Figure 1 of the Kimi Linear paper: performance against decoding acceleration for Kimi Linear, GDN-H and MLA, and time per output token against decoding length up to 1 million tokens](/img/attention/papers/kimi-figure1.png)](/img/attention/papers/kimi-figure1.png)
>
> **Context:** Kimi Linear against a full-attention MLA model and a Gated DeltaNet hybrid (GDN-H), all trained the same way on 1.4 trillion tokens.
>
> **What it says:** (a) On long-context tests (RULER at 128K) Kimi Linear scores highest (84.3) while decoding about 4 times faster than MLA. (b) Time per output token: at 1 million tokens, Kimi Linear takes 1.84 ms against 11.48 ms for MLA, "6.3× faster".
>
> **Why it matters:** "Faster" is expected; "better" is the surprise. The abstract claims it is the first hybrid linear design that "outperforms full attention under fair comparisons".
>
> [Read the paper on arXiv](https://arxiv.org/abs/2510.26692)

> [!PAPER] Kimi Linear · Section 5 · page 13
> [![The Kimi Linear efficiency paragraph: comparable to MLA at short lengths, 2.3 and 2.9 times faster at 512k and 1M for prefilling, and 6 times faster decoding at 1M context](/img/attention/papers/kimi-6x.png)](/img/attention/papers/kimi-6x.png)
>
> **Context:** The efficiency results, in words.
>
> **What it says:** At short lengths (4K to 16K) Kimi Linear is about as fast as MLA. It pulls ahead from 128K: 2.3× faster at 512K and 2.9× at 1M tokens for processing the prompt, and "for decoding at 1M context length, Kimi Linear is 6× faster than full attention".
>
> **Why it matters:** The same pattern as my own timing in section 8: no gain for short texts, a growing gain for long ones.

> [!PAPER] Nemotron-H · Section 1 · page 2, Figure 1
> [![Figure 1 of the Nemotron-H paper: MMLU-Pro accuracy against output tokens per second per GPU for Nemotron-H 56B and 47B, Qwen-2.5-72B and Llama-3.1 models, with a 65,536-token input](/img/attention/papers/nemo-figure1.png)](/img/attention/papers/nemo-figure1.png)
>
> **Context:** Accuracy (MMLU-Pro) against throughput (output tokens per second per GPU), with a 65,536-token input and 1,024 output tokens.
>
> **What it says:** Nemotron-H-56B matches or beats the transformer models of similar size in accuracy while generating about **2.4 times** as many tokens per second as Qwen-2.5-72B and Llama-3.1-70B. The smaller 47B version is faster still.
>
> **Why it matters:** For a company serving millions of long requests, 2.4 times more tokens per GPU is 2.4 times fewer GPUs for the same work.

> [!PAPER] Transformers are RNNs · Section 4.2.1 · page 7, Table 1
> [![Table 1 of the Transformers are RNNs paper: bits per dimension and images per second for softmax, LSH and linear attention on MNIST image generation, with linear attention 317 times faster](/img/attention/papers/kath-table1.png)](/img/attention/papers/kath-table1.png)
>
> **Context:** The 2020 paper's own use case: generating images of handwritten digits one pixel at a time (each pixel is a "token").
>
> **What it says:** Linear attention reaches almost the same quality as softmax (0.644 against 0.621 bits per dimension, lower is better) but generates 142.8 images per second against 0.45: **317 times faster**.
>
> **Why it matters:** Generating one pixel or one word at a time is exactly where the fixed-size memory shines: every new step costs the same, however much has been generated.

**Use cases in one line each:**

- **Very long documents and codebases** (hundreds of thousands to millions of tokens): hybrids keep memory and per-token cost almost flat.
- **Reasoning models and agents** that write very long outputs: each new token stays cheap, as in Kimi Linear's 6× faster decoding at 1M tokens.
- **High-volume serving**: more requests per GPU (Nemotron-H's 2.4× throughput) means lower cost per answer.
- **Devices with little memory**: a fixed-size state means a long conversation does not need a growing cache.
- **Where full attention still wins**: exact lookups of details far back in the text. That is why every model above keeps some attention layers.

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
