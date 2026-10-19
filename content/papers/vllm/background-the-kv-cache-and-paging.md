---
title: "The KV Cache and Paging"
label: Background
description: "The two ideas the paper joins: the cache of keys and values that a language model keeps while it writes, and the pages an operating system uses to hand out memory. With small runs you can check."
covers: "Vaswani et al. (2017) §3.1 and §3.2; virtual memory, paging and memory allocators"
date: 2026-10-19
tags: [vllm, kv-cache, attention, virtual-memory, paging, background]
---

The vLLM paper takes an idea from operating systems and puts it inside a language model. To follow it, you need both halves. One half is how a model remembers the text it has already read: the KV cache. The other half is how a computer hands out memory to programs that grow and shrink: paging.

This page builds both from the ground up, with small runs that you can check by hand. If you know one half already, skip to the other.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## A model writes one token at a time

A language model does one thing. You give it a piece of text, and it gives back a probability for every possible next token. A **token** is a piece of text: a word, part of a word, or a sign. The model picks one token, adds it to the text, and asks itself again. An answer of 300 tokens takes 300 rounds.

The probability of a whole text is built from these one-step guesses. For a text of tokens $$x_1, x_2, \dots, x_n$$:

$$
P(x_1, \dots, x_n) = P(x_1) \cdot P(x_2 \mid x_1) \cdot P(x_3 \mid x_1, x_2) \cdots P(x_n \mid x_1, \dots, x_{n-1})
$$

Read it left to right. The chance of the whole text is the chance of the first token, times the chance of the second given the first, and so on. The bar $$\mid$$ means "given". Each factor is one round of the model. Suppose the model gives "Four" a chance of 0.01, then "score" after "Four" a chance of 0.5, then "and" after "Four score" a chance of 0.9. The three tokens together get 0.01 × 0.5 × 0.9 = 0.0045.

The point for this paper is in the shape of the formula. Every factor looks back at **all** earlier tokens. The model must know about the whole text so far, every round. The question is what it keeps in memory so that it does not read the whole text again each time.

> [!DEFINITION] Autoregressive
> A model that writes one output at a time and feeds each output back as input for the next one. Language models such as GPT and OPT are autoregressive.

## Attention: a query, keys and values

The part of the model that looks back at earlier tokens is **attention**. The Transformer paper of 2017 describes it in two sentences.

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · §3.2 · page 3
> [![The definition of attention, with mapping a query and a set of key-value pairs to an output and weighted sum highlighted](/img/papers/vllm/pb-attn-def.png)](/img/papers/vllm/pb-attn-def.png)
> [![The continuation on the next page, with of the values, compatibility function of the query with the corresponding key highlighted](/img/papers/vllm/pb-attn-def-2.png)](/img/papers/vllm/pb-attn-def-2.png)
>
> **What it says:** attention is "mapping a query and a set of key-value pairs to an output". The output is a "weighted sum of the values", where each weight comes from a "compatibility function of the query with the corresponding key".
>
> **What it means:**
>
> Think of a library. You walk in with a question: that is the **query**. Every book has a label on its spine: that is its **key**. Every book has content: that is its **value**. You compare your question with every label. The books whose labels match best get most of your attention, and you take your answer mostly from them.
>
> In the model, every token makes all three. Each token has a key, which says "this is what I am about", and a value, which says "this is what I contribute". The newest token makes a query, which says "this is what I am looking for". The query is compared with every key. Good matches get large weights. The output is a blend of the values, mixed by those weights.

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · §3.2.1 · page 4
> [![Scaled dot-product attention, with dot products of the query with all keys, divide each by the square root of d_k, and apply a softmax function to obtain the weights on the values highlighted, and equation 1](/img/papers/vllm/pb-attn-eq.png)](/img/papers/vllm/pb-attn-eq.png)
>
> **What it says:** compute "the dot products of the query with all keys, divide each by $$\sqrt{d_k}$$, and apply a softmax function to obtain the weights on the values". In matrix form, equation (1).
>
> **What it means:**
>
> The match between a query and a key is a **dot product**: multiply the two vectors number by number and add the results. Vectors that point the same way give a large number. Dividing by $$\sqrt{d_k}$$, the square root of the vector length, keeps these numbers in a sensible range for long vectors. The **softmax** then turns the scores into weights that are positive and add up to 1.
>
> The equation packs many queries into a matrix $$Q$$, so a whole text is computed in one go during training. When a model writes, it has one new token, so $$Q$$ is a single row. That single row is all the new work. The keys and values it is compared with all belong to tokens that came earlier.

For one new token with query $$q$$, and earlier tokens with keys $$k_j$$ and values $$v_j$$, the equation reads:

$$
o = \sum_{j} a_j v_j, \qquad a_j = \frac{\exp(q \cdot k_j / \sqrt{d})}{\sum_t \exp(q \cdot k_t / \sqrt{d})}
$$

Read it right to left. Take the dot product of the query with key $$j$$, divide by $$\sqrt{d}$$, and raise $$e$$ to that power. Divide by the same thing summed over all tokens $$t$$: that is the softmax, and it gives the weight $$a_j$$. The output $$o$$ is the sum of the values, each times its weight.

Now a tiny example that you can do on paper. The vectors have length $$d = 2$$, so we divide by $$\sqrt{2} \approx 1.414$$. The query is $$q = [1, 0]$$. There are three earlier tokens:

| token | key | value | $$q \cdot k / \sqrt{2}$$ |
|---|---|---|---|
| 1 | [1, 0] | [1, 0] | 1 / 1.414 = 0.707 |
| 2 | [0, 1] | [0, 1] | 0 / 1.414 = 0 |
| 3 | [1, 1] | [1, 1] | 1 / 1.414 = 0.707 |

The softmax: $$e^{0.707} \approx 2.03$$ and $$e^{0} = 1$$. The sum is 2.03 + 1 + 2.03 = 5.06. So the weights are 2.03 / 5.06 = 0.401, then 0.198, then 0.401. The output is 0.401 × [1, 0] + 0.198 × [0, 1] + 0.401 × [1, 1] = [0.802, 0.599]. Tokens 1 and 3 match the query, so the output leans towards their values.

We check this with code, two ways. Method 1 follows the equation step by step. Method 2 is PyTorch's built-in attention:

```python
def attention_by_hand(q, K, V):
    """softmax(q K^T / sqrt(d)) V, one step at a time."""
    d = q.shape[-1]
    scores = K @ q / math.sqrt(d)                         # how well the query matches each key
    weights = torch.exp(scores) / torch.exp(scores).sum() # softmax: positive, sums to 1
    return scores, weights, weights @ V                   # weighted average of the values
```

```text
== 1. Attention for one query over three tokens (d = 2)
scores  q.k/sqrt(2): [0.707, 0.0, 0.707]
weights softmax    : [0.401, 0.198, 0.401] sum 1.000
output, Method 1   : [0.802, 0.599]
output, Method 2   : [0.802, 0.599] agree: True
```

<figure class="fig"><svg viewBox="0 0 720 250" role="img" aria-label="A query vector [1, 0] compared with three keys. Keys 1 and 3 match it equally well and get weight 0.401 each; key 2 gets 0.198. The output is the weighted average of the three values, [0.802, 0.599]."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">One query looks at three tokens</text><rect class="box-on" x="30.0" y="110.0" width="120.0" height="34.0" rx="6"/><text class="t-mono" x="90.0" y="131.5" text-anchor="middle">q = [1, 0]</text><rect class="box-1" x="250.0" y="50.0" width="130.0" height="34.0" rx="6"/><text class="t-mono" x="315.0" y="71.5" text-anchor="middle">k1 = [1, 0]</text><rect class="box-2" x="400.0" y="50.0" width="130.0" height="34.0" rx="6"/><text class="t-mono" x="465.0" y="71.5" text-anchor="middle">v1 = [1, 0]</text><line class="edge" x1="152.0" y1="127.0" x2="246.0" y2="67.0" marker-end="url(#ah)"/><text class="t-tick t-strong" x="560.0" y="72.0" text-anchor="start">weight 0.401</text><rect class="box-1" x="250.0" y="110.0" width="130.0" height="34.0" rx="6"/><text class="t-mono" x="315.0" y="131.5" text-anchor="middle">k2 = [0, 1]</text><rect class="box-2" x="400.0" y="110.0" width="130.0" height="34.0" rx="6"/><text class="t-mono" x="465.0" y="131.5" text-anchor="middle">v2 = [0, 1]</text><line class="edge" x1="152.0" y1="127.0" x2="246.0" y2="127.0" marker-end="url(#ah)"/><text class="t-tick t-strong" x="560.0" y="132.0" text-anchor="start">weight 0.198</text><rect class="box-1" x="250.0" y="170.0" width="130.0" height="34.0" rx="6"/><text class="t-mono" x="315.0" y="191.5" text-anchor="middle">k3 = [1, 1]</text><rect class="box-2" x="400.0" y="170.0" width="130.0" height="34.0" rx="6"/><text class="t-mono" x="465.0" y="191.5" text-anchor="middle">v3 = [1, 1]</text><line class="edge" x1="152.0" y1="127.0" x2="246.0" y2="187.0" marker-end="url(#ah)"/><text class="t-tick t-strong" x="560.0" y="192.0" text-anchor="start">weight 0.401</text><text class="t-mono" x="250.0" y="236.0" text-anchor="start">output = 0.401 v1 + 0.198 v2 + 0.401 v3 = [0.802, 0.599]</text></svg><figcaption>Our tiny example. The query matches keys 1 and 3 equally, so they get the same weight, 0.401. The output is the weighted average of the values.</figcaption></figure>

> [!DEFINITION] Query, key and value
> Three vectors that each attention layer computes for every token, by multiplying the token's vector with three learned weight matrices. The **query** asks, the **key** is matched against queries, and the **value** is the content that gets blended into the output.

> [!DEFINITION] Softmax
> A function that turns a list of numbers into weights: raise $$e$$ to each number, then divide each by the total. The weights are positive and add up to 1, and a larger number always gets a larger weight.

## Why keys and values can be kept

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · §3.2.3 · page 5
> [![The decoder self-attention bullet, with up to and including that position, preserve the auto-regressive property and masking out (setting to minus infinity) highlighted](/img/papers/vllm/pb-mask.png)](/img/papers/vllm/pb-mask.png)
>
> **What it says:** in the decoder, each position may attend to positions "up to and including that position". To "preserve the auto-regressive property", the model implements this by "masking out (setting to −∞)" the scores of later positions.
>
> **What it means:**
>
> A model that writes text must not peek at the future. When it computes token 5, tokens 6 and 7 do not exist yet. So during training, every token is blocked from reading the tokens after it. The scores for later tokens are set to minus infinity, and the softmax turns them into weights of exactly zero. This is the **causal mask**.
>
> The mask has a consequence that the Transformer paper does not spell out, and the vLLM paper depends on it. Token 3 never reads token 4. So when token 4 arrives, nothing about token 3 changes: not its key, not its value. Once the keys and values of a token are computed, they stay the same for the rest of the text. They can be stored and reused. That store is the **KV cache**.

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="A six by six grid. Each row is a token; it may read the columns up to and including itself. The upper right triangle is empty. The last row, the newest token, reads all earlier tokens."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The causal mask: each token reads only itself and the tokens before it</text><text class="t-tick" x="160.0" y="52.0" text-anchor="middle">Four</text><text class="t-tick" x="200.0" y="52.0" text-anchor="middle">score</text><text class="t-tick" x="240.0" y="52.0" text-anchor="middle">and</text><text class="t-tick" x="280.0" y="52.0" text-anchor="middle">seven</text><text class="t-tick" x="320.0" y="52.0" text-anchor="middle">years</text><text class="t-tick" x="360.0" y="52.0" text-anchor="middle">ago</text><text class="t-tick" x="130.0" y="84.0" text-anchor="end">Four</text><rect class="box-1" x="142.0" y="62.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="182.0" y="62.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="222.0" y="62.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="262.0" y="62.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="302.0" y="62.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="342.0" y="62.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="130.0" y="124.0" text-anchor="end">score</text><rect class="box-1" x="142.0" y="102.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="182.0" y="102.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="222.0" y="102.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="262.0" y="102.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="302.0" y="102.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="342.0" y="102.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="130.0" y="164.0" text-anchor="end">and</text><rect class="box-1" x="142.0" y="142.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="182.0" y="142.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="222.0" y="142.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="262.0" y="142.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="302.0" y="142.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="342.0" y="142.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="130.0" y="204.0" text-anchor="end">seven</text><rect class="box-1" x="142.0" y="182.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="182.0" y="182.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="222.0" y="182.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="262.0" y="182.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="302.0" y="182.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="342.0" y="182.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="130.0" y="244.0" text-anchor="end">years</text><rect class="box-1" x="142.0" y="222.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="182.0" y="222.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="222.0" y="222.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="262.0" y="222.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="302.0" y="222.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="342.0" y="222.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="130.0" y="284.0" text-anchor="end">ago</text><rect class="box-2" x="142.0" y="262.0" width="36.0" height="36.0" rx="4"/><rect class="box-2" x="182.0" y="262.0" width="36.0" height="36.0" rx="4"/><rect class="box-2" x="222.0" y="262.0" width="36.0" height="36.0" rx="4"/><rect class="box-2" x="262.0" y="262.0" width="36.0" height="36.0" rx="4"/><rect class="box-2" x="302.0" y="262.0" width="36.0" height="36.0" rx="4"/><rect class="box-2" x="342.0" y="262.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="410.0" y="80.0" text-anchor="start">Row = the token that reads.</text><text class="t-tick" x="410.0" y="100.0" text-anchor="start">Column = the token it may read.</text><text class="t-tick" x="410.0" y="140.0" text-anchor="start">The last row (orange) is the newest</text><text class="t-tick" x="410.0" y="160.0" text-anchor="start">token. It adds one row and reads</text><text class="t-tick" x="410.0" y="180.0" text-anchor="start">all earlier columns.</text><text class="t-tick" x="410.0" y="220.0" text-anchor="start">No earlier row ever reads a later</text><text class="t-tick" x="410.0" y="240.0" text-anchor="start">token, so earlier keys and values</text><text class="t-tick t-strong" x="410.0" y="260.0" text-anchor="start">never change: they can be cached.</text></svg><figcaption>The causal mask. Each row reads only the columns up to itself. A new token adds a row at the bottom and never changes the rows above it.</figcaption></figure>

We test this claim on a real model, OPT-125M, the smallest model of the OPT family that the vLLM paper serves. We run it on the first 8 tokens of a sentence. Then we run it on the same 8 tokens followed by 9 more. If the claim is right, the keys and values of the first 8 tokens are the same in both runs, in all 12 layers:

```text
== 2. Do the keys of earlier tokens change when more tokens follow? (OPT-125M, FP32, CPU)
8 tokens alone, then the same 8 tokens followed by 9 more
largest change of a key or value of the first 8 tokens, over all 12 layers: 7.39e-06
largest key value, for scale: 6.5
```

The largest change is 0.0000074, against values as large as 6.5. That is rounding noise from adding numbers in a different order, not a real change. Later tokens do not touch earlier keys and values.

### The new token needs only its own query

So each step of writing needs three things: the query of the new token, and the keys and values of all tokens so far. Only the query is new work. We check this inside OPT-125M. We run the model on all tokens but the last, which fills the cache. Then we feed the last token alone and record what the first attention layer receives and returns.

Method 1 rebuilds that layer by hand. It computes the query of the new token, reads every key and value from the cache, and applies the attention equation. Method 2 is the library's own output for the same layer:

```python
def layer0_from_cache(h, cache):
    """Method 1: query of the new token only; keys and values of all tokens read from the cache."""
    H, D = attn.num_heads, attn.head_dim
    qn = (attn.q_proj(h) * attn.scaling).view(1, 1, H, D).transpose(1, 2)   # (1, 12, 1, 64); OPT scales q
    Kc, Vc = cache.layers[0].keys, cache.layers[0].values                   # (1, 12, n, 64), new token included
    w = torch.softmax(qn @ Kc.transpose(-1, -2), dim=-1)                    # (1, 12, 1, n): one weight per token
    o = (w @ Vc).transpose(1, 2).reshape(1, 1, H * D)                       # weighted values, heads joined
    return attn.out_proj(o), w
```

```text
== 3. Layer 0 for the newest token, rebuilt from the cache by hand
input to layer 0: (1, 1, 768) (one token); cached keys: (1, 12, 17, 64)
attention weights of head 0 over the 17 tokens: 0.04 0.05 0.01 0.03 0.02 0.01 0.04 0.02 0.02 0.01 0.01 0.02 0.02 0.03 0.07 0.58 0.03
Method 1 (by hand from the cache) vs Method 2 (the library): max difference 5.96e-08
```

The input to the layer is one token, a vector of 768 numbers. The cache holds keys for all 17 tokens, split into 12 **heads** of 64 numbers each. The two methods agree to 0.00000006. This is the whole job of attention while a model writes: one query against a growing cache. In head 0, most of the weight (0.58) goes to one earlier token.

> [!DEFINITION] Attention head
> A model runs several attention computations side by side, each on its own slice of the vector. Each one is a **head**. OPT-125M has 12 heads of 64 numbers, which together make the 768 numbers of each token. Each head has its own queries, keys and values, so the cache stores keys and values for every head of every layer.

### How large the cache gets

Each token stores one key and one value, each as long as the model's hidden size, in every layer. With 2 bytes per number (FP16), the cache of one token is:

$$
\text{bytes per token} = 2 \times d_{\text{model}} \times L \times 2
$$

The first 2 counts the key and the value. $$d_{\text{model}}$$ is the hidden size, $$L$$ the number of layers, and the last 2 is bytes per number. For OPT-125M: 2 × 768 × 12 × 2 = 36,864 bytes, about 36 KB per token. For OPT-13B, the paper's main model: 2 × 5,120 × 40 × 2 = 819,200 bytes, which is 800 KB per token. A single answer of 2,048 tokens then needs 1.68 GB. [Part 1](part-1-the-big-idea.md) checks this formula against the real tensors.

So the cache is large, it grows by one token per step, and nobody knows its final size until the answer ends. That is a memory problem, and computers have faced it before.

## How a computer hands out memory

Memory is a long row of numbered slots. When a program needs space, it asks an **allocator** for a run of slots. The allocator finds a free run, marks it as taken, and returns where it starts. When the program is done, the run goes back.

Most data in deep learning lives in **contiguous** runs: one unbroken stretch of slots per tensor. That is simple and fast, because the next element is always in the next slot. But it has a weakness. Requests of different sizes come and go, and they leave holes between them.

> [!DEFINITION] Allocator
> The part of a program or operating system that hands out memory and takes it back. It keeps track of which slots are free.

Here is the weakness in 32 slots. Four requests take 8, 6, 8 and 6 slots. Then B and D finish:

```python
class FirstFit:
    """Give each request one contiguous run of slots: the first free gap that is large enough."""
    ...
    def alloc(self, name, n):
        for s, g in self.gaps():
            if g >= n:
                self.used[name] = (s, n)
                return True
        return False
```

```text
== 4. Contiguous memory: a first-fit allocator (memory of 32 slots)
after A=8, B=6, C=8, D=6:  AAAAAAAABBBBBBCCCCCCCCDDDDDD....
after B and D finish:      AAAAAAAA......CCCCCCCC..........   free 16, largest gap 10
request E needs 12 slots in one piece: refused, although 16 slots are free
the same memory in blocks of 2 slots, any block anywhere:
  free blocks 8; E needs 6 blocks: granted
```

<figure class="fig"><svg viewBox="0 0 720 190" role="img" aria-label="Two rows of 32 slots. First, A, B, C and D fill 28 slots. Then B and D leave, which leaves a gap of 6 and a gap of 10. A request for 12 contiguous slots is refused, although 16 slots are free."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">External fragmentation in 32 slots</text><text class="t-tick t-strong" x="30.0" y="52.0" text-anchor="start">A=8, B=6, C=8, D=6</text><rect class="box-1" x="30.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="38.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="50.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="58.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="70.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="78.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="90.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="98.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="110.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="118.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="130.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="138.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="150.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="158.5" y="80.0" text-anchor="middle">A</text><rect class="box-1" x="170.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="178.5" y="80.0" text-anchor="middle">A</text><rect class="box-3" x="190.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="198.5" y="80.0" text-anchor="middle">B</text><rect class="box-3" x="210.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="218.5" y="80.0" text-anchor="middle">B</text><rect class="box-3" x="230.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="238.5" y="80.0" text-anchor="middle">B</text><rect class="box-3" x="250.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="258.5" y="80.0" text-anchor="middle">B</text><rect class="box-3" x="270.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="278.5" y="80.0" text-anchor="middle">B</text><rect class="box-3" x="290.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="298.5" y="80.0" text-anchor="middle">B</text><rect class="box-1" x="310.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="318.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="330.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="338.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="350.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="358.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="370.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="378.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="390.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="398.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="410.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="418.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="430.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="438.5" y="80.0" text-anchor="middle">C</text><rect class="box-1" x="450.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="458.5" y="80.0" text-anchor="middle">C</text><rect class="box-3" x="470.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="478.5" y="80.0" text-anchor="middle">D</text><rect class="box-3" x="490.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="498.5" y="80.0" text-anchor="middle">D</text><rect class="box-3" x="510.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="518.5" y="80.0" text-anchor="middle">D</text><rect class="box-3" x="530.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="538.5" y="80.0" text-anchor="middle">D</text><rect class="box-3" x="550.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="558.5" y="80.0" text-anchor="middle">D</text><rect class="box-3" x="570.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="578.5" y="80.0" text-anchor="middle">D</text><rect class="box-ghost" x="590.0" y="60.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="610.0" y="60.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="630.0" y="60.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="650.0" y="60.0" width="17.0" height="30.0" rx="3"/><text class="t-tick t-strong" x="30.0" y="116.0" text-anchor="start">B and D finish</text><rect class="box-1" x="30.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="38.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="50.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="58.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="70.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="78.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="90.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="98.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="110.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="118.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="130.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="138.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="150.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="158.5" y="144.0" text-anchor="middle">A</text><rect class="box-1" x="170.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="178.5" y="144.0" text-anchor="middle">A</text><rect class="box-ghost" x="190.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="210.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="230.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="250.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="270.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="290.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-1" x="310.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="318.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="330.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="338.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="350.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="358.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="370.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="378.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="390.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="398.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="410.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="418.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="430.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="438.5" y="144.0" text-anchor="middle">C</text><rect class="box-1" x="450.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="458.5" y="144.0" text-anchor="middle">C</text><rect class="box-ghost" x="470.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="490.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="510.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="530.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="550.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="570.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="590.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="610.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="630.0" y="124.0" width="17.0" height="30.0" rx="3"/><rect class="box-ghost" x="650.0" y="124.0" width="17.0" height="30.0" rx="3"/><text class="t-tick" x="30.0" y="178.0" text-anchor="start">Free: 6 + 10 = 16 slots. Request E needs 12 in one piece: refused. In blocks of 2 slots it fits.</text></svg><figcaption>Our 32-slot run. After B and D leave, 16 slots are free in two gaps of 6 and 10. A request for 12 slots in one piece is refused. If memory is handed out in small blocks that can sit anywhere, the same request fits.</figcaption></figure>

Sixteen slots are free, and a request for 12 is refused, because no single gap holds 12. This is **external fragmentation**: free memory that exists but is cut into pieces too small to use. The fix in the last line is the fix of this whole page. If a request may take small blocks from anywhere, every free block is useful.

The same effect shows over a long run. We replay 5,000 requests of 20 to 400 slots in a memory of 4,096 slots, with old requests finishing at random:

```text
trace of 5000 requests in 4096 slots, sizes 20 to 400; a request that does not fit is refused:
  contiguous first fit: memory holding data 81.6% on average; 2468 refused, 2456 of them while enough slots were free in total; slots not holding data at a refusal: 746 on average
  blocks of 16:         memory holding data 91.9% on average; 2473 refused; slots not holding data at a refusal: 305 on average
```

This trace sends far more work than the memory can hold, so both designs refuse about half of the requests. The difference is in how full the memory is while they work. Contiguous runs keep 81.6% of the memory holding data; blocks keep 91.9%. In the contiguous design, 2,456 of the 2,468 refusals happened while enough slots were free in total, just not in one piece. Blocks waste only the unused end of each request's last block.

> [!DEFINITION] Internal and external fragmentation
> **Internal fragmentation** is memory given to a request that the request never fills. **External fragmentation** is free memory between taken runs, split into gaps too small to use.

### Rounding up: the buddy allocator

Many allocators make bookkeeping easier by rounding every request up to a power of two: 32, 64, 128 and so on. A well-known design, the **buddy allocator**, splits memory in halves until a piece of the right power of two remains. The vLLM paper assumes that its baseline, Orca, uses one. Rounding has a price:

```text
== 4b. The buddy allocator rounds every request up to a power of two
  request    25 -> chunk    32:    7 slots never used (22% of the chunk)
  request   300 -> chunk   512:  212 slots never used (41% of the chunk)
  request   513 -> chunk  1024:  511 slots never used (50% of the chunk)
  request  1500 -> chunk  2048:  548 slots never used (27% of the chunk)
```

A request just above a power of two wastes almost half of its chunk: 513 slots get 1,024. That is internal fragmentation, created on purpose to make the allocator simple.

## Paging: the operating system's answer

Operating systems met this problem in the 1960s, with programs instead of requests. Their answer is **paging**, and every laptop and phone uses it today.

Memory is cut into **pages** of one fixed size. A program does not get one long run of real memory. It gets a list of pages, and each page can sit anywhere. The program still sees one unbroken row of addresses, its **virtual** memory. A table, the **page table**, records where each of its pages really is.

<figure class="fig"><svg viewBox="0 0 720 260" role="img" aria-label="Four virtual pages of one program map through a page table to frames 5, 2, 7 and 0 of physical memory. Address 10,000 is in page 2 at offset 1,808, which maps to physical address 30,480."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Paging: the program sees one row; the pages sit anywhere</text><text class="t-tick t-strong" x="30.0" y="56.0" text-anchor="start">virtual pages (the program)</text><rect class="box-1" x="30.0" y="66.0" width="120.0" height="34.0" rx="6"/><text class="t-mono" x="90.0" y="87.5" text-anchor="middle">page 0</text><rect class="box-1" x="30.0" y="108.0" width="120.0" height="34.0" rx="6"/><text class="t-mono" x="90.0" y="129.5" text-anchor="middle">page 1</text><rect class="box-1" x="30.0" y="150.0" width="120.0" height="34.0" rx="6"/><text class="t-mono" x="90.0" y="171.5" text-anchor="middle">page 2</text><rect class="box-1" x="30.0" y="192.0" width="120.0" height="34.0" rx="6"/><text class="t-mono" x="90.0" y="213.5" text-anchor="middle">page 3</text><text class="t-tick t-strong" x="240.0" y="56.0" text-anchor="start">page table</text><rect class="box" x="240.0" y="66.0" width="110.0" height="34.0" rx="6"/><text class="t-mono" x="295.0" y="87.5" text-anchor="middle">0 -&gt; 5</text><line class="edge" x1="152.0" y1="83.0" x2="236.0" y2="83.0" marker-end="url(#ah)"/><rect class="box" x="240.0" y="108.0" width="110.0" height="34.0" rx="6"/><text class="t-mono" x="295.0" y="129.5" text-anchor="middle">1 -&gt; 2</text><line class="edge" x1="152.0" y1="125.0" x2="236.0" y2="125.0" marker-end="url(#ah)"/><rect class="box" x="240.0" y="150.0" width="110.0" height="34.0" rx="6"/><text class="t-mono" x="295.0" y="171.5" text-anchor="middle">2 -&gt; 7</text><line class="edge" x1="152.0" y1="167.0" x2="236.0" y2="167.0" marker-end="url(#ah)"/><rect class="box" x="240.0" y="192.0" width="110.0" height="34.0" rx="6"/><text class="t-mono" x="295.0" y="213.5" text-anchor="middle">3 -&gt; 0</text><line class="edge" x1="152.0" y1="209.0" x2="236.0" y2="209.0" marker-end="url(#ah)"/><text class="t-tick t-strong" x="450.0" y="56.0" text-anchor="start">physical frames (real memory)</text><rect class="box-1" x="450.0" y="66.0" width="56.0" height="34.0" rx="6"/><text class="t-mono" x="478.0" y="87.5" text-anchor="middle">p3</text><rect class="box-ghost" x="512.0" y="66.0" width="56.0" height="34.0" rx="6"/><text class="t-muted" x="540.0" y="88.0" text-anchor="middle">1</text><rect class="box-1" x="574.0" y="66.0" width="56.0" height="34.0" rx="6"/><text class="t-mono" x="602.0" y="87.5" text-anchor="middle">p1</text><rect class="box-ghost" x="636.0" y="66.0" width="56.0" height="34.0" rx="6"/><text class="t-muted" x="664.0" y="88.0" text-anchor="middle">3</text><rect class="box-ghost" x="450.0" y="116.0" width="56.0" height="34.0" rx="6"/><text class="t-muted" x="478.0" y="138.0" text-anchor="middle">4</text><rect class="box-1" x="512.0" y="116.0" width="56.0" height="34.0" rx="6"/><text class="t-mono" x="540.0" y="137.5" text-anchor="middle">p0</text><rect class="box-ghost" x="574.0" y="116.0" width="56.0" height="34.0" rx="6"/><text class="t-muted" x="602.0" y="138.0" text-anchor="middle">6</text><rect class="box-1" x="636.0" y="116.0" width="56.0" height="34.0" rx="6"/><text class="t-mono" x="664.0" y="137.5" text-anchor="middle">p2</text><text class="t-tick" x="450.0" y="190.0" text-anchor="start">frame numbers 0 to 7; empty frames</text><text class="t-tick" x="450.0" y="208.0" text-anchor="start">belong to other programs or are free</text><text class="t-mono" x="30.0" y="248.0" text-anchor="start">address 10,000: page 2, offset 1,808 -&gt; frame 7 -&gt; 7 x 4,096 + 1,808 = 30,480</text></svg><figcaption>Paging. The program sees pages 0 to 3 in a row. The page table sends them to frames 5, 2, 7 and 0 of real memory. Address 10,000 lands in page 2 and comes out at physical address 30,480.</figcaption></figure>

To find a byte, the computer splits the virtual address into a page number and an offset inside the page. With a page size $$P$$:

$$
\text{page} = \lfloor v / P \rfloor, \qquad \text{offset} = v \bmod P, \qquad \text{physical} = \text{table}[\text{page}] \times P + \text{offset}
$$

Read it in order. Divide the virtual address $$v$$ by the page size and round down: that is the page number. The remainder is the offset, the position inside the page. Look the page up in the table to get its real frame, multiply by the page size, and add the offset back.

A worked example with $$P = 4{,}096$$ bytes. The address 10,000 divided by 4,096 is 2, remainder 1,808. So it is in page 2 at offset 1,808. Our table sends page 2 to frame 7. The physical address is 7 × 4,096 + 1,808 = 30,480.

Hardware does not divide. The number 4,096 is $$2^{12}$$, so it shifts the address right by 12 bits to get the page. It keeps the low 12 bits as the offset. Our run does it both ways:

```python
def translate_div(v):
    """Method 1: page number and offset by division."""
    page, offset = v // P, v % P
    return page, offset, page_table[page] * P + offset


def translate_bits(v):
    """Method 2: what the hardware does. 4096 = 2^12, so shift by 12 bits and keep the low 12 bits."""
    page, offset = v >> 12, v & 0xFFF
    return page, offset, (page_table[page] << 12) | offset
```

```text
== 5. Paging: from a virtual address to a physical one
virtual  10000 -> page 2, offset 1808 -> frame 7 -> physical  30480; bit method agrees: True
virtual   4095 -> page 0, offset 4095 -> frame 5 -> physical  24575; bit method agrees: True
virtual   4096 -> page 1, offset    0 -> frame 2 -> physical   8192; bit method agrees: True
this laptop's page size: 16384 bytes (16 KB)
```

Look at addresses 4,095 and 4,096. They sit next to each other for the program, but they land in frames 5 and 2, far apart in real memory. The program never knows. The laptop that ran this uses pages of 16 KB; many other computers use 4 KB.

> [!DEFINITION] Virtual memory and page table
> **Virtual memory** is the address space a program sees: one unbroken row, starting at 0. The **page table** maps each virtual page to a real **frame** of physical memory. The processor reads the table on every memory access, with a small cache of recent lookups to keep it fast.

### Pages on demand

Paging gives a second gift. A page does not need real memory until the program uses it. The program can ask for a gigabyte, and the operating system only writes that down. Real memory is given page by page, at the moment a page is first touched. We can watch this on the laptop. We ask for 1 GB, then write one byte into each page of the first 256 MB:

```text
== 5b. Pages on demand: map 1 GB, then touch 256 MB of it
resident memory before:   1022.7 MB
after mapping 1 GB:       1022.7 MB   (+0.0 MB)
after touching 256 MB:    1278.7 MB   (+256.0 MB, 16384 pages written)
```

Asking for 1 GB cost nothing. Touching 16,384 pages of 16 KB cost exactly 256 MB. The **resident memory** is the real memory a program holds, and it grew only where pages were used. A program that books a lot but uses little costs only what it uses.

The waste of paging is small and has a clear bound. A program's last page may be partly empty, so it wastes less than one page:

```text
== 5c. The waste of paging is the unused end of the last page
       1 bytes -> 1 pages of 4096 =   4096 bytes, 4095 unused
    4096 bytes -> 1 pages of 4096 =   4096 bytes,    0 unused
    4097 bytes -> 2 pages of 4096 =   8192 bytes, 4095 unused
   10000 bytes -> 3 pages of 4096 =  12288 bytes, 2288 unused
```

There is no external fragmentation at all, because every page has the same size and any free frame fits any page.

### Sharing and copy-on-write

Paging gives a third gift: sharing. Two page tables can point to the same frame. On Linux or macOS, a program can start a copy of itself. The copy gets a new page table that points to the same frames as the original.

Nothing is copied. Only when one of them writes to a shared page does the operating system copy that one page first. This is **copy-on-write**. Memory is copied only where the two really differ.

> [!DEFINITION] Copy-on-write
> A way to share memory safely. Several owners point to the same page. When one owner wants to change it, that owner gets a private copy of that page first, and the others keep the original.

## The two halves together

Put the halves side by side. A language model's KV cache grows one token at a time, its final size is unknown, and many requests of different sizes come and go. Serving systems before vLLM stored each cache in one contiguous run, booked for the longest possible answer. That is the old allocator with all its waste: booked space that is not used yet, rounding, and gaps between runs.

The vLLM paper's move is to treat the KV cache the way an operating system treats a program's memory:

| Operating system | vLLM |
|---|---|
| a process (a running program) | a request |
| a byte | a token's keys and values |
| a page of 4 KB or 16 KB | a block of a fixed number of tokens (16 by default) |
| the page table | the block table of each request |
| a page given when first touched | a block given when the last one is full |
| copy-on-write when a process forks | copy-on-write when several answers share one prompt |

One thing does not carry over for free. A program reads memory through the processor, which follows the page table in hardware. Attention reads keys and values in a GPU kernel, which expected them in one contiguous tensor. So the paper had to write a new attention kernel that follows the block table: **PagedAttention**. [Part 1](part-1-the-big-idea.md) starts the paper there.

> [!TAKEAWAYS] Key takeaways
> - **A model writes one token at a time,** and each step looks back at every earlier token through attention.
> - **Attention** compares the new token's query with every earlier key, turns the scores into weights with a softmax, and blends the values. In our tiny example: weights 0.401, 0.198 and 0.401, output [0.802, 0.599].
> - **The causal mask** stops tokens from reading later tokens, so earlier keys and values never change. In OPT-125M they changed by at most 0.0000074 when 9 tokens were added.
> - **The KV cache** stores them, so each step computes only the new query. Our hand-built layer matched the library to 0.00000006.
> - **The cache costs** 2 × hidden size × layers × 2 bytes per token: 36 KB for OPT-125M, 800 KB for OPT-13B.
> - **Contiguous allocation fragments memory.** In 32 slots, 16 were free but a request for 12 was refused. Over a long trace, blocks kept 91.9% of memory holding data against 81.6%.
> - **Paging** cuts memory into equal pages and places them anywhere, on demand. It wastes less than one page per program, and pages can be shared with copy-on-write.
> - **vLLM** applies paging to the KV cache: blocks for pages, tokens for bytes, requests for processes.

<details>
<summary>Run it yourself</summary>

The script behind this page is [`code/papers/vllm/vllm_partb.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_partb.py). It runs the tiny attention example, tests the cached keys of OPT-125M, rebuilds one attention layer from the cache, and replays the allocator traces. It also translates addresses and maps 1 GB of memory to watch pages arrive. Every number is saved in `results/partb.json`. It runs on a CPU in under a minute.

```bash
pip install torch transformers psutil
python vllm_partb.py   # downloads facebook/opt-125m (about 250 MB)
```

<figure><img src="/img/papers/vllm/partb-run.png" alt="Terminal output of vllm_partb.py: the tiny attention example, the cached keys test, the layer rebuilt from the cache, the allocator runs, address translation and pages on demand" loading="lazy" /><figcaption>The full output of vllm_partb.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[Attention Is All You Need on arXiv](https://arxiv.org/abs/1706.03762) · [vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [our scripts for this page](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
