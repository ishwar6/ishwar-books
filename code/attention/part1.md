---
title: "Attention, Part 1: Self-Attention and Multi-Head Attention"
description: "The one mechanism every modern LLM is built on, from the ground up: why attention was invented, how queries, keys and values work, why scores are divided by √d, how multi-head attention works, and what real attention heads in Qwen2.5 actually learn. Every claim checked in code."
date: 2026-10-04
tags: [attention, transformers, llm]
series: "Attention, From the Ground Up"
series_part: 1
motif: graph
accent: "#7c9cff"
---

Every large language model you have used, from GPT to Llama, Qwen, Gemma and DeepSeek, is built around one mechanism: **attention**. The 2017 paper that introduced the transformer was called *Attention Is All You Need*, and for a few years everyone used the same version of it.

That is no longer true. Look inside the models released in the last two years and you find a zoo of variants: grouped-query attention, multi-head latent attention, sliding-window attention, sparse attention, gated attention, linear attention, and hybrids that mix several of them. They all exist for the same reason: the original version is expensive, and it gets more expensive the longer the text.

This series builds all of them from the ground up, one idea at a time, with code that is tested against reference implementations and real models.

> [!TIP] The series in one minute
> **Part 1 (this one):** self-attention and multi-head attention, the original recipe, and what real attention heads learn.
>
> **Part 2:** shrinking the memory each token leaves behind: **MQA**, **GQA** and **MLA**.
>
> **Part 3:** attending to fewer tokens: **sliding-window attention** and **DeepSeek Sparse Attention**.
>
> **Part 4:** replacing softmax attention in most layers: **gated attention**, **linear attention**, **Gated DeltaNet** and the **hybrid** architectures built from them.

## The problem attention solved

Before transformers, translation models were **recurrent neural networks** (RNNs). An encoder read the source sentence one word at a time, updating a hidden state as it went, and a decoder then wrote the translation from that state.

The flaw was the squeeze. Everything the decoder knew about the sentence had to pass through a fixed-size vector, whether the sentence had five words or fifty.

{{FIG:p1_bottleneck|Left: a classic encoder-decoder squeezes the whole sentence into one vector. Right: with attention, the decoder looks back at every input word and decides how much each one matters right now (the weights shown are illustrative).}}

In 2014, Bahdanau, Cho and Bengio proposed a fix: let the decoder **look back** at all of the encoder's states at every step, and learn how much to weigh each one. To write "cat", it should look mostly at "chat". To write "black", at "noir". That weighting is **attention**.

The transformer took the idea one step further. Instead of bolting attention onto an RNN, it threw the recurrence away and made attention the main way tokens exchange information. When the tokens attend to each other within the same sequence, it is called **self-attention**.

## Self-attention, in one picture

Self-attention answers one question for every token: *of all the tokens I am allowed to see, which ones should I gather information from, and how much?*

{{FIG:p1_pipeline|The self-attention pipeline. Each token is projected into a query, a key and a value; every query is scored against every key; the future is masked out; a softmax turns scores into weights; and the weights mix the values.}}

Each token's vector is multiplied by three learned matrices to make three new vectors:

- a **query** (Q): what this token is looking for;
- a **key** (K): what this token offers to others, used for matching;
- a **value** (V): the information this token hands over once it is chosen.

Then:

1. **Score.** Compare every query with every key using a dot product: $$QK^\top$$. A high score means "this key matches what I am looking for".
2. **Scale.** Divide by $$\sqrt{d}$$, where $$d$$ is the size of the key vectors. (Why this matters gets its own section below.)
3. **Mask.** In a model that writes left to right, a token must not see the future, so every score for a later position is set to $$-\infty$$.
4. **Softmax.** Turn each row of scores into weights that are positive and add up to 1.
5. **Mix.** Each token's output is the weighted average of the values: $$Z = AV$$.

All of it fits in one line:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
$$

where $$M$$ is the mask: 0 where looking is allowed, $$-\infty$$ where it is not.

## A tiny example, step by step

Here is that recipe in PyTorch, on four tokens with random weights:

```python
import math, torch

def self_attention(x, Wq, Wk, Wv, causal=True):
    """x: (T, d_model). Returns the attention weights A (T, T) and the output Z."""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T / math.sqrt(K.shape[-1])          # how well each pair matches
    if causal:                                         # a token may not look at the future
        T = x.shape[0]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    A = torch.softmax(scores, dim=-1)                  # each row sums to 1
    return A, A @ V
```

Running it on the tokens `The cat sat down` prints:

```text
attention weights (rows = query token, columns = key token):
     The  1.00  0.00  0.00  0.00
     cat  0.25  0.75  0.00  0.00
     sat  0.31  0.26  0.42  0.00
    down  0.08  0.29  0.09  0.54
   row sums: [1.0, 1.0, 1.0, 1.0]
```

{{FIG:p1_toy|The same matrix as a picture. Each row is one token deciding where to look. The empty upper triangle is the future, masked out, and every row sums to 1.}}

Three things to notice:

- **The first token can only look at itself**, so its row is a single 1.00. The mask is doing its job.
- **Every row sums to exactly 1.** Attention is always a weighted *average*: a token can choose where to look, but not how much to look in total.
- **The matrix is T × T.** Four tokens, sixteen numbers. A thousand tokens, a million numbers, in every head of every layer. Hold on to that: it is the cost every variant in this series is trying to reduce.

## Why divide by √d?

The scaling step looks like a detail, but leaving it out breaks training. Here is why, measured.

If the entries of a query and a key are random numbers with variance 1, their dot product is a sum of $$d$$ such products, so its variance is $$d$$ and its spread (standard deviation) is $$\sqrt{d}$$. Bigger vectors produce bigger scores. And softmax reacts to big scores by putting almost all of the weight on the single largest one.

I measured it with 10,000 random queries, each scored against 64 random keys:

| Vector size d | Spread of q·k | Top weight, no scaling | Top weight, divided by √d |
|---|---|---|---|
| 16 | 4.0 | 0.576 | 0.107 |
| 64 | 8.0 | 0.796 | 0.108 |
| 256 | 16.0 | 0.896 | 0.107 |
| 1,024 | 32.0 | 0.950 | 0.108 |

{{FIG:p1_scaling|The largest softmax weight in a 64-key attention row. Without scaling, it climbs towards 1 as the vectors get bigger, so attention collapses onto one token. Dividing by √d keeps it steady at every size.}}

The spread grows exactly as $$\sqrt{d}$$ (4, 8, 16, 32). Without scaling, by $$d = 1{,}024$$ one key takes 95% of the weight before the model has learned anything: the softmax is saturated, its gradients are tiny, and learning stalls. Dividing by $$\sqrt{d}$$ undoes the growth, and the weights stay spread out at every size.

## Many heads instead of one

One attention head computes one pattern of weights. But a token often needs several kinds of information at once: the previous word, the subject of the sentence, the matching bracket in a piece of code. **Multi-head attention** (MHA) simply runs several heads side by side.

{{FIG:p1_heads|Multi-head attention. Each head has its own query, key and value projections and computes its own attention matrix. The outputs are concatenated and mixed by one more matrix, Wo.}}

Each head works in a smaller space (with 8 heads and a 64-dimensional model, each head's vectors have 8 numbers), so running 8 heads costs about the same as one big head. Then the heads' outputs are concatenated and multiplied by an output matrix $$W_o$$ that blends them back together.

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                   # x: (B, T, d_model)
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.d).transpose(1, 2)   # (B, heads, T, d)
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)             # one T x T matrix per head
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        A = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
        heads = A @ v                                                    # (B, heads, T, d)
        return self.Wo(heads.transpose(1, 2).reshape(B, T, self.h * self.d))
```

### Proof that it is right

A hand-written implementation is only worth trusting if it agrees with a reference. I ran this class (8 heads, 64 dimensions, in 64-bit precision) and compared it with PyTorch's two built-in versions, using the same weights:

```text
our MHA vs F.scaled_dot_product_attention: max |diff| = 1.7e-16
our MHA vs torch.nn.MultiheadAttention:     max |diff| = 1.7e-16
```

A difference of $$10^{-16}$$ is the last digit of 64-bit floating point: the three implementations compute exactly the same thing.

## What real attention heads learn

Random weights show the mechanics. To see what attention *does*, look inside a trained model. I ran **Qwen2.5-0.5B**, which has 24 layers of 14 attention heads (336 heads in total), on the sentence:

> The cat sat on the mat because it was tired.

and pulled out every head's attention matrix. Three kinds of behaviour stand out:

{{FIG:p1_real|Three real heads from Qwen2.5-0.5B. Left: an attention sink, where almost every token looks at the first token. Middle: a previous-token head. Right: a head that links tokens, including "it", back to "cat". Hover a cell for its weight.}}

**Attention sinks.** In **68% of all heads**, more than half of the attention goes to the very first token, whatever that token is. This is not a bug. Softmax must put its weight *somewhere*; when a head has nothing useful to say for a token, it parks the weight on the first position, which every token can see. Xiao et al. (2023) named these **attention sinks** and showed that keeping the first few tokens around is essential when you trim a long context.

**Previous-token heads.** Layer 8, head 7 puts on average 84% of each token's attention on the token right before it: a clean diagonal. Heads like this are building blocks for copying and pattern-matching.

**A pronoun head.** Layer 5, head 5 sends 83% of the weight from "it" to "cat". One sentence could be a coincidence, so I tested the same head on six sentences it had never seen:

| Sentence | Pronoun | Head's top choice | Weight |
|---|---|---|---|
| The dog chased the ball because it was bored. | it | **dog** | 0.90 |
| The trophy did not fit in the suitcase because it was too big. | it | **trophy** | 0.62 |
| My sister bought a new car and she loves driving it. | she | **sister** | 0.86 |
| The engineers fixed the server after it crashed twice. | it | **server** | 0.40 |
| The scientist published the paper because she was proud of it. | she | **scientist** | 0.78 |
| The children ate the cake because they were hungry. | they | **children** | 0.72 |

Six out of six. One honest caveat: in all of these sentences the right noun is also a subject or a main noun near the start, so this head may be finding "the main noun" rather than truly resolving references. Either way, nobody programmed it. It emerged from training.

## The cost that drives everything else

Now the problem. Two costs grow with the length of the text:

1. **Compute.** Every token scores every earlier token: a $$T \times T$$ matrix in every head of every layer. Double the text, quadruple the work.
2. **Memory.** To write each new token, the model needs the keys and values of every token before it (the **KV cache**, covered in depth in [Part 2 of the LLM inference series](llm-inference-2-kv-cache.md)). With plain multi-head attention, every head of every layer stores its own key and value for every token. For Llama 2 7B, that is **512 KiB per token**, so one sequence at its full 4,096-token context needs **2 GiB**, on top of the model's weights.

Every variant in this series attacks one of those costs:

| Part | Idea | What it reduces |
|---|---|---|
| 2 | MQA, GQA, MLA | **what each token stores**: fewer or compressed keys and values |
| 3 | sliding-window and sparse attention | **which tokens are looked at**: a window, or a learned subset |
| 4 | linear attention, Gated DeltaNet, hybrids | **whether a growing cache is needed at all** in most layers |

## Summary

- Attention was invented to remove the bottleneck of squeezing a whole sentence into one vector: let every step **look back** at every input.
- **Self-attention**: queries match keys, softmax turns the scores into weights, the weights mix the values. One line: $$\operatorname{softmax}(QK^\top/\sqrt{d} + M)V$$.
- The **causal mask** hides the future; every row of weights sums to 1.
- **Dividing by √d** keeps the scores from growing with the vector size. Without it, the top weight reached 0.95 at d = 1,024; with it, 0.107.
- **Multi-head attention** runs several heads in parallel. Our implementation matched PyTorch's to 1.7 × 10⁻¹⁶.
- Real heads specialise. In Qwen2.5-0.5B, **68%** of heads act as attention sinks, one head tracks the previous token (84%), and one links pronouns to nouns (6 out of 6 test sentences).
- The $$T \times T$$ compute and the per-token KV cache are what every later variant tries to shrink.

<details>
<summary>Run it yourself</summary>

The full script that produced every number above is [`code/attention/part1_attention.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part1_attention.py), and the pronoun test is [`part1_coref_check.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part1_coref_check.py). It needs `torch` and `transformers`, and downloads Qwen2.5-0.5B (about 1 GB) on first run.

```bash
pip install torch transformers
python part1_attention.py      # sections 1 to 4, writes results/part1.json
python part1_coref_check.py    # the six pronoun sentences
```

Reading a real model's attention weights only takes one flag:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
ids = tok("The cat sat on the mat because it was tired.", return_tensors="pt").input_ids
attentions = model(ids, output_attentions=True).attentions   # one (batch, heads, T, T) tensor per layer
```

</details>

## References

1. D. Bahdanau, K. Cho, Y. Bengio. [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473). 2014.
2. A. Vaswani et al. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
3. G. Xiao et al. [*Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453). ICLR 2024.
4. K. Clark, U. Khandelwal, O. Levy, C. Manning. [*What Does BERT Look At? An Analysis of BERT's Attention*](https://arxiv.org/abs/1906.04341). 2019.
5. Qwen Team. [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
