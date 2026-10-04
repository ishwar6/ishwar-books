---
title: "Attention, Part 1: Self-Attention and Multi-Head Attention"
description: "The one idea every modern AI language model is built on, explained from zero: why attention was invented, how a word decides which other words to look at, why the scores are divided by √d, how several heads work together, and what real attention heads inside Qwen2.5 actually learn. Every claim checked in code."
date: 2026-10-04
tags: [attention, transformers, llm]
series: "Attention, From the Ground Up"
series_part: 1
motif: graph
accent: "#7c9cff"
---

Every AI chatbot you have used (ChatGPT, Gemini, Llama, Qwen, Gemma, DeepSeek) is built around one idea called **attention**. It is the part of the model that lets each word look at the other words around it and decide which ones matter.

This series explains attention from zero, and then every important variation of it that today's models use. No background is assumed. Whenever a technical word appears, there is a box like this one that explains it:

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word in plain language. If you already know the word, skip the box.

> [!TIP] The series in one minute
> **Part 1 (this one):** how attention works, from scratch, and what it learns inside a real model.
>
> **Part 2:** making attention use less memory: **MQA**, **GQA** and **MLA**.
>
> **Part 3:** making each word look at fewer words: **sliding-window** and **sparse** attention.
>
> **Part 4:** replacing attention in most layers with something cheaper: **linear attention**, **Gated DeltaNet** and **hybrid** models.

## First: words become numbers

A computer cannot read words. So before anything else, a language model turns text into **tokens**, and every token into a list of numbers.

> [!DEFINITION] Token
> A small piece of text: usually a word, sometimes part of a word or a punctuation mark. "The cat sat." is four tokens: `The`, `cat`, `sat`, `.`.

> [!DEFINITION] Vector
> A list of numbers, like `[0.8, -1.2, 0.3, 2.1]`. A model represents every token as a vector. Real models use hundreds or thousands of numbers per token.

{{FIG:p1_vectors|Every token becomes a vector. The model learns these numbers during training so that words with similar meanings get similar numbers.}}

From here on, "a token" and "the token's vector" mean the same thing: attention only ever works with the numbers.

## The problem attention solved

Before 2014, translation models read a sentence **one word at a time** and squeezed everything into a single vector, a kind of running summary. Then they wrote the translation from that one summary.

> [!DEFINITION] RNN (recurrent neural network)
> An older kind of model that reads text one token at a time and keeps a running summary (its "hidden state"). Each new token updates the summary.

The problem: one fixed-size summary has to hold the whole sentence, whether it has five words or fifty. Long sentences get blurry.

{{FIG:p1_bottleneck|Left: the old way squeezes the whole sentence into one vector. Right: with attention, the model looks back at every input word and decides how much each one matters right now (the weights shown are illustrative).}}

Here is how the paper that introduced attention describes that problem, in its very first section:

> [!PAPER] Bahdanau, Cho, Bengio (2014) · Section 1, Introduction · page 1
> [![The introduction of the Bahdanau paper, with the sentence about compressing a whole sentence into a fixed-length vector highlighted](/img/attention/papers/bahdanau-bottleneck.png)](/img/attention/papers/bahdanau-bottleneck.png)
>
> **Context:** The authors first describe the standard translation model of 2014: an *encoder* RNN reads the source sentence, a *decoder* RNN writes the translation. Then they point at its weak spot.
>
> **What it says:** All the information of the source sentence must be squeezed into one fixed-length vector. Long sentences do not fit, and translation quality "deteriorates rapidly" as sentences get longer.
>
> **Why it matters:** This one paragraph is the reason attention exists. Every later idea in this series is still fighting the same enemy: how to give a model access to a lot of text without one fixed-size bottleneck.
>
> [Read the paper on arXiv](https://arxiv.org/abs/1409.0473)

In 2014, Bahdanau, Cho and Bengio had a simple idea: **let the model look back at every word**, every time it writes a word, and learn how much to pay attention to each one. To write "cat" it looks mostly at "chat"; to write "black", mostly at "noir". That weighting is **attention**.

### The first attention equations

Their version works like this. The encoder leaves one vector $$h_j$$ for each source word $$j$$ (they call these **annotations**). When the decoder is about to write target word $$i$$, it does three things:

$$
e_{ij} = a(s_{i-1}, h_j)
\qquad\qquad
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
\qquad\qquad
c_i = \sum_{j=1}^{T_x} \alpha_{ij}\, h_j
$$

where:

- $$s_{i-1}$$ is the decoder's current state (what it has written so far);
- $$a$$ is a small learned network, the **alignment model**, that scores how well source word $$j$$ fits the next target word;
- $$e_{ij}$$ is that score, and $$\alpha_{ij}$$ is the score turned into a weight by softmax (the same softmax you will meet below);
- $$T_x$$ is the number of source words;
- $$c_i$$ is the **context vector**: a weighted mix of all the source words, made fresh for every target word.

> [!DEFINITION] Encoder and decoder
> An **encoder** reads the input (for example an English sentence) and turns it into vectors. A **decoder** writes the output (the French sentence) one word at a time. Translation models of 2014 had both. Today's chatbots are decoder-only: they read and write with the same stack of layers.

> [!PAPER] Bahdanau et al. (2014) · Section 3.1, Decoder · page 3, Equation (6)
> [![Equation 6 of the Bahdanau paper, the softmax that turns alignment scores into attention weights, highlighted](/img/attention/papers/bahdanau-weights.png)](/img/attention/papers/bahdanau-weights.png)
>
> **Context:** This is the heart of the paper: how the decoder decides how much weight $$\alpha_{ij}$$ to give each source word.
>
> **What it says:** The weight is a softmax over alignment scores $$e_{ij}$$, and each score comes from a small network $$a$$ that compares the decoder state with one source word.
>
> **Why it matters:** "Score every input, softmax the scores, take the weighted average" is still exactly what every attention layer does today. The 2017 transformer only changed *how* the score is computed (a dot product instead of a small network).

The paper also drew the weights as pictures, and they show something nobody had programmed:

> [!PAPER] Bahdanau et al. (2014) · Section 5.2.1, Alignment · page 6, Figure 3
> [![Figure 3 of the Bahdanau paper: four attention matrices between English source words and French output words](/img/attention/papers/bahdanau-alignments.png)](/img/attention/papers/bahdanau-alignments.png)
>
> **Context:** Each square is one translation. Columns are English words, rows are the French words the model wrote, and a white pixel means "a lot of attention".
>
> **What it says:** The bright cells mostly follow the diagonal, because French and English often use the same word order. But look at panel (a): "European Economic Area" became "zone économique européenne", in **reverse** order, and the bright cells turn around to follow it.
>
> **Why it matters:** The model learned word alignment, a classic hard problem in translation, purely from example translations. This was the first clear evidence that attention weights are not just a trick: they can be read, and they often mean something.

In 2017 the paper *Attention Is All You Need* went further. It threw the word-by-word reading away and made attention the **main** way words share information. That design is called the **transformer**, and every modern chatbot is one. When the words of one text attend to each other, it is called **self-attention**.

## Two small tools: the dot product and softmax

Attention is built from two simple tools. It is worth meeting them first.

### Tool 1: the dot product measures "how similar"

> [!DEFINITION] Dot product
> Take two vectors of the same length, multiply them number by number, and add up the results. The answer is one number.

As an equation, for two vectors $$a$$ and $$b$$ with $$d$$ numbers each:

$$
a \cdot b = a_1 b_1 + a_2 b_2 + \dots + a_d b_d
$$

where $$a_1$$ is the first number of $$a$$, $$b_1$$ the first number of $$b$$, and so on.

For example, $$[1, 2] \cdot [3, 4] = 1 \times 3 + 2 \times 4 = 11$$.

Why is this useful? Because it tells you **how much two vectors point the same way**:

{{FIG:p1_dot|The dot product is big when two vectors point the same way, zero when they are at a right angle, and negative when they point in opposite directions.}}

So "is this word relevant to that word?" can become "is the dot product of their vectors big?"

### Tool 2: softmax turns scores into percentages

Dot products can be any number: big, small, or negative. Attention needs **weights** instead: positive numbers that add up to 1, like percentages.

> [!DEFINITION] Softmax
> A function that turns any list of numbers into positive numbers that add up to 1. Bigger inputs get bigger shares. It keeps the order: the largest score always gets the largest weight.

$$
\operatorname{softmax}(s_i) = \frac{e^{s_i}}{e^{s_1} + e^{s_2} + \dots + e^{s_n}}
$$

where $$s_1, \dots, s_n$$ are the scores, and $$e \approx 2.718$$. Raising $$e$$ to a power makes every score positive; dividing by the total makes them add up to 1.

{{FIG:p1_softmax|Softmax in action. Four scores, one of them negative, become four weights that are all positive and add up to 1.}}

## Self-attention, step by step

Now we can build attention. It answers one question for every token: *of the tokens I am allowed to see, which ones should I take information from, and how much?*

Each token's vector is turned into **three new vectors**, each made by multiplying it by a different grid of numbers:

> [!DEFINITION] Matrix and learned weights
> A **matrix** is a grid of numbers. Multiplying a vector by a matrix turns it into a new vector by mixing its numbers in a fixed way. The numbers inside the matrices are **learned weights**: the model adjusts them during training until it works well.

- a **query** (Q): *what am I looking for?*
- a **key** (K): *what do I contain?* (used for matching)
- a **value** (V): *what will I hand over if I am chosen?*

A library is a good picture. Your query is the question you bring. Every book has a key (its title and topic) and a value (its contents). You compare your question with every title, then read mostly from the books that match best.

{{FIG:p1_pipeline|The self-attention pipeline. Each token makes a query, a key and a value. Every query is compared with every key, the future is hidden, softmax turns the scores into weights, and the weights mix the values.}}

The steps:

1. **Score.** Compare every query with every key using the dot product. A big score means "this key matches what I am looking for".
2. **Scale.** Divide the scores by $$\sqrt{d}$$ (the square root of the vector length). The next section shows why.
3. **Hide the future.** A model that writes left to right must not peek at words that come later, so those scores are blocked.
4. **Softmax.** Turn each token's scores into weights that add up to 1.
5. **Mix.** Each token's new vector is the weighted average of the values.

> [!DEFINITION] Causal mask
> The rule that a token may only look at itself and the tokens **before** it, never after. "Causal" because the future cannot cause the past. It is applied by setting future scores to minus infinity, which softmax turns into a weight of exactly 0.

All five steps fit in one equation, the most important equation in this series:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V
$$

where:

- $$Q$$, $$K$$, $$V$$ are the queries, keys and values of all the tokens, stacked as rows;
- $$QK^\top$$ is every query's dot product with every key (the $$^\top$$, "transpose", just flips $$K$$ so the multiplication lines up);
- $$d$$ is the length of each key vector;
- $$M$$ is the causal mask: $$0$$ where looking is allowed, $$-\infty$$ where it is not.

This is the equation from *Attention Is All You Need* (Vaswani et al., 2017), the paper that introduced the transformer. Here it is in the original:

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · Section 3.2.1 · page 4, Equation (1)
> [![Section 3.2.1 of Attention Is All You Need, with the definition of scaled dot-product attention and Equation 1 highlighted](/img/attention/papers/vaswani-scaled-dot.png)](/img/attention/papers/vaswani-scaled-dot.png)
>
> **Context:** Section 3.2 defines the attention used inside the transformer. The authors call it **scaled dot-product attention**.
>
> **What it says:** Take the dot products of a query with all keys, divide each by $$\sqrt{d_k}$$, apply softmax, and use the result to weight the values. For many queries at once, stack them into matrices: $$\operatorname{softmax}(QK^T/\sqrt{d_k})V$$.
>
> **Why it matters:** This one line replaced Bahdanau's small scoring network with a single matrix multiplication. Matrix multiplications are exactly what GPUs are fastest at, which is a big reason transformers could be trained on so much more data. The paper writes $$d_k$$ where this article writes $$d$$, and leaves the mask out of the equation (Figure 2 shows it as an optional "Mask" step).

> [!PAPER] Vaswani et al. (2017) · Section 3.2 · page 4, Figure 2
> [![Figure 2 of Attention Is All You Need: block diagrams of scaled dot-product attention and multi-head attention](/img/attention/papers/vaswani-figure2.png)](/img/attention/papers/vaswani-figure2.png)
>
> **Context:** The paper's own diagram of the two pieces this article builds: the attention calculation (left) and several of them in parallel (right).
>
> **What it says:** Read the left diagram from the bottom up: multiply Q and K (MatMul), Scale by $$1/\sqrt{d_k}$$, Mask (optional), SoftMax, then multiply by V. Those are exactly steps 1 to 5 above.
>
> **Why it matters:** This figure is probably the most reproduced diagram in modern AI. If you can read it, you can read the attention part of almost any model's code.

### A tiny example you can check

Here is the same thing in a few lines of Python, using PyTorch:

```python
import math, torch

def self_attention(x, Wq, Wk, Wv, causal=True):
    """x holds one vector per token. Returns the attention weights A and the new vectors Z."""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv                   # 1. make queries, keys and values
    scores = Q @ K.T / math.sqrt(K.shape[-1])          # 2. score every pair, divided by sqrt(d)
    if causal:                                         # 3. hide the future
        T = x.shape[0]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float("-inf"))
    A = torch.softmax(scores, dim=-1)                  # 4. scores -> weights; each row adds up to 1
    return A, A @ V                                    # 5. mix the values
```

> [!DEFINITION] PyTorch
> A free Python library for building neural networks. `@` means matrix multiplication, and `torch.softmax` is the softmax from above.

I ran it on four tokens, `The cat sat down`, with random weights. Here are the attention weights it printed: each row is one token, showing how much it looks at each token in the columns.

```text
attention weights (rows = query token, columns = key token):
     The  1.00  0.00  0.00  0.00
     cat  0.25  0.75  0.00  0.00
     sat  0.31  0.26  0.42  0.00
    down  0.08  0.29  0.09  0.54
   row sums: [1.0, 1.0, 1.0, 1.0]
```

{{FIG:p1_toy|The same weights as a picture. Each row is one token deciding where to look. The empty cells are the future, which is hidden. Every row adds up to 1.}}

Three things to notice:

- **`The` can only look at itself**, so its row is a single 1.00. The causal mask is working.
- **Every row adds up to exactly 1.** A token can choose *where* to look, but its total attention is always 100%.
- **There is one number for every pair of tokens.** 4 tokens make 16 numbers. 1,000 tokens make 1,000,000, in every head of every layer. Remember this: it is the cost that the whole rest of this series is about.

## Why divide by √d?

Step 2 looks like a small detail. It is not: leave it out and the model cannot learn properly. Here is why, with measurements.

A dot product adds up $$d$$ multiplications. If the numbers are random, adding more of them makes the total swing more widely: the typical size of a dot product grows like $$\sqrt{d}$$.

> [!DEFINITION] Spread (standard deviation)
> How far numbers typically are from their average. A spread of 4 means values usually land within a few units of the average; a spread of 32 means they swing much more wildly.

Softmax is very sensitive to big numbers. If one score is much bigger than the others, softmax gives it almost all the weight, and the model effectively stops looking at anything else.

I measured it: 10,000 random queries, each scored against 64 random keys, for four vector lengths.

| Vector length d | Spread of the dot products | Top weight without ÷√d | Top weight with ÷√d |
|---|---|---|---|
| 16 | 4.0 | 0.576 | 0.107 |
| 64 | 8.0 | 0.796 | 0.108 |
| 256 | 16.0 | 0.896 | 0.107 |
| 1,024 | 32.0 | 0.950 | 0.108 |

{{FIG:p1_scaling|The biggest weight in a softmax over 64 keys. Without dividing by √d, it climbs towards 1 as the vectors get longer, so attention collapses onto a single token. Dividing by √d keeps it steady.}}

The spread grows exactly as $$\sqrt{d}$$: 4, 8, 16, 32. Without the division, at $$d = 1{,}024$$ one random key already takes 95% of the attention before the model has learned anything. When softmax is that lopsided, its learning signals (gradients) become tiny and training stalls. Dividing by $$\sqrt{d}$$ cancels the growth: the top weight stays near 0.107 at every size.

### Where the √d comes from (the math in four lines)

> [!DEFINITION] Mean and variance
> The **mean** is the average value. The **variance** is the average of the squared distance from the mean; it measures spread. The standard deviation (the "spread" in the table above) is the square root of the variance.

Suppose every number in $$q$$ and $$k$$ is random, with mean 0 and variance 1, and all are independent. Look at one term of the dot product, $$q_i k_i$$:

$$
\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\,\mathbb{E}[k_i] = 0 \cdot 0 = 0,
\qquad
\operatorname{Var}(q_i k_i) = \mathbb{E}[q_i^2]\,\mathbb{E}[k_i^2] = 1 \cdot 1 = 1
$$

The dot product adds $$d$$ such independent terms, and variances of independent terms simply add up:

$$
\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d} \operatorname{Var}(q_i k_i) = d
\qquad\Longrightarrow\qquad
\text{spread}(q \cdot k) = \sqrt{d}
$$

So dividing by $$\sqrt{d}$$ brings the spread back to exactly 1, whatever the vector length:

$$
\operatorname{Var}\!\left(\frac{q \cdot k}{\sqrt{d}}\right) = \frac{d}{d} = 1
$$

where $$\mathbb{E}$$ means "average value" (expected value) and $$\operatorname{Var}$$ means variance. That is exactly what my measurement showed: spreads of 4, 8, 16, 32 for $$d$$ = 16, 64, 256, 1,024, which are $$\sqrt{d}$$.

The transformer paper gives the same reasoning, in one sentence and one footnote:

> [!PAPER] Vaswani et al. (2017) · Section 3.2.1 · page 4
> [![The paragraph of Attention Is All You Need explaining that large dot products push softmax into regions with extremely small gradients](/img/attention/papers/vaswani-why-sqrt.png)](/img/attention/papers/vaswani-why-sqrt.png)
> [![Footnote 4 of Attention Is All You Need: the dot product of two random vectors has mean 0 and variance d_k](/img/attention/papers/vaswani-footnote.png)](/img/attention/papers/vaswani-footnote.png)
>
> **Context:** Right after defining attention, the authors explain why they divide by $$\sqrt{d_k}$$, which earlier dot-product attention did not do.
>
> **What it says:** For large $$d_k$$ the dot products "grow large in magnitude", pushing softmax into regions with "extremely small gradients". Footnote 4 gives the reason: for random $$q$$ and $$k$$, the dot product has mean 0 and variance $$d_k$$.
>
> **Why it matters:** Note the word "suspect": the authors gave the argument, not a measurement. The experiment above measures it directly: without the division, one random key takes 95% of the attention at $$d = 1{,}024$$.

## Many heads instead of one

One attention calculation learns one pattern. But a word often needs several kinds of information at once: the word just before it, the subject of the sentence, the matching bracket in some code. So models run several attention calculations side by side.

> [!DEFINITION] Attention head
> One complete attention calculation, with its own query, key and value matrices. A model with 8 heads runs 8 of them in parallel, and each one can learn to look for something different.

{{FIG:p1_heads|Multi-head attention. Each head has its own matrices and its own attention pattern. The results are joined together and mixed by one more matrix, called Wo.}}

As equations:

$$
\text{head}_i = \text{Attention}(X W_q^{(i)},\; X W_k^{(i)},\; X W_v^{(i)})
$$

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_o
$$

where:

- $$X$$ holds the vectors of all the tokens;
- $$W_q^{(i)}, W_k^{(i)}, W_v^{(i)}$$ are head $$i$$'s own query, key and value matrices;
- $$h$$ is the number of heads;
- **Concat** places the heads' results side by side into one long vector per token;
- $$W_o$$ is the **output matrix** that mixes the heads back together.

Each head works with shorter vectors, so $$h$$ heads cost about the same as one big head:

$$
d_{\text{head}} = \frac{d_{\text{model}}}{h}
\qquad\text{for example}\qquad
\frac{512}{8} = 64
$$

where $$d_{\text{model}}$$ is the length of each token's vector and $$h$$ the number of heads. The original transformer used exactly these numbers: 512 per token, 8 heads, 64 per head. Total work is $$h \times d_{\text{head}} = d_{\text{model}}$$, the same as one head of full size.

> [!PAPER] Vaswani et al. (2017) · Section 3.2.2, Multi-Head Attention · page 5
> [![Section 3.2.2 of Attention Is All You Need: the sentence on attending to different representation subspaces, and the MultiHead equations highlighted](/img/attention/papers/vaswani-multihead.png)](/img/attention/papers/vaswani-multihead.png)
>
> **Context:** After defining one attention calculation, the paper explains why it runs several in parallel.
>
> **What it says:** "Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this." Then the two equations: the heads are computed separately, concatenated, and multiplied by $$W^O$$.
>
> **Why it matters:** "Averaging inhibits this" is the key idea. One softmax produces one weighted average, which blurs together everything a token might want. Several heads let it gather several different things at once, and the real heads below show exactly that kind of specialisation.

```python
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads          # number of heads, numbers per head
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                     # x: (batch, tokens, d_model)
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.d).transpose(1, 2)   # cut into h heads
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)             # one score grid per head
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        A = torch.softmax(scores.masked_fill(mask, float("-inf")), -1)
        heads = A @ v
        return self.Wo(heads.transpose(1, 2).reshape(B, T, self.h * self.d))   # join and mix
```

### Proof that it is right

Code you write yourself is only trustworthy if it agrees with a trusted version. PyTorch has two built-in attention functions, so I gave them exactly the same weights as the class above and compared the results:

```text
our MHA vs F.scaled_dot_product_attention: max |diff| = 1.7e-16
our MHA vs torch.nn.MultiheadAttention:     max |diff| = 1.7e-16
```

The largest difference is $$1.7 \times 10^{-16}$$, that is 0.00000000000000017. That is the smallest rounding error a computer can make with these numbers: the three versions compute exactly the same thing.

## What real attention heads learn

Random weights show *how* attention works. To see *what* it does, we have to look inside a trained model. I used **Qwen2.5-0.5B**, a small open model with 24 layers and 14 heads in each layer: **336 heads** in total.

> [!DEFINITION] Layer
> A model stacks the same kind of block many times. Each block (a layer) contains attention plus some other processing, and passes its result to the next one. Qwen2.5-0.5B has 24 layers.

I gave it this sentence and read out the attention weights of every head:

> The cat sat on the mat because it was tired.

Three kinds of head stood out:

{{FIG:p1_real|Three real heads from Qwen2.5-0.5B. Left: almost every token looks at the first token (an attention sink). Middle: every token looks at the token just before it. Right: many tokens, including "it", look back at "cat". Hover over a cell to see its weight.}}

**1. Attention sinks.** In **68% of all heads**, more than half of the attention goes to the very first token, whatever it is.

> [!DEFINITION] Attention sink
> Softmax forces every token's weights to add up to 1, so a head must look *somewhere*, even when it has nothing useful to find. Trained models learn to park that spare attention on the first token, which every token can see. Xiao et al. (2023) named this an attention sink.

**2. A previous-token head.** Layer 8, head 7 puts on average **84%** of each token's attention on the token just before it: the clean diagonal in the middle picture. Heads like this help with copying and spotting patterns.

**3. A pronoun head.** In layer 5, head 5, the word "it" puts **83%** of its attention on "cat", the thing "it" refers to. One sentence could be luck, so I tested the same head on six sentences it had never seen:

| Sentence | Word | Where it looked most | Weight |
|---|---|---|---|
| The dog chased the ball because it was bored. | it | **dog** | 0.90 |
| The trophy did not fit in the suitcase because it was too big. | it | **trophy** | 0.62 |
| My sister bought a new car and she loves driving it. | she | **sister** | 0.86 |
| The engineers fixed the server after it crashed twice. | it | **server** | 0.40 |
| The scientist published the paper because she was proud of it. | she | **scientist** | 0.78 |
| The children ate the cake because they were hungry. | they | **children** | 0.72 |

Six out of six. One honest note: in every one of these sentences, the right word is also the main noun near the start, so this head might be finding "the main noun" rather than truly understanding who "it" is. Either way, nobody programmed this. It appeared on its own during training.

### The same patterns, found by researchers

These findings are not special to Qwen. Two well-known papers found the same kinds of heads in other models.

> [!PAPER] Clark, Khandelwal, Levy, Manning (2019), What Does BERT Look At? · Figure 5 · page 5
> [![The coreference row of Figure 5 of the Clark et al. paper: BERT head 5-4 links pronouns such as she and her to the noun they refer to](/img/attention/papers/clark-coref.png)](/img/attention/papers/clark-coref.png)
>
> **Context:** The authors examined every attention head in BERT, a 2018 transformer, and checked which heads line up with grammar relations. Figure 5 shows examples; this crop shows its last row.
>
> **What it says:** Head 5-4 (layer 5, head 4) links "coreferent mentions" to their antecedents: "she" and "her" point back to "Kim", "negotiations" points back to "talks". It gets the right antecedent 65.1% of the time. Other heads in the same figure find verbs' objects (86.8%) and nouns' determiners (94.3%).
>
> **Why it matters:** This is the same kind of "pronoun head" we found in Qwen2.5-0.5B (layer 5, head 5), in a different model trained in a different way. It suggests that heads like this are something transformers reliably discover, not an accident of one model.

> [!PAPER] Xiao et al. (2023), Efficient Streaming Language Models with Attention Sinks · Figure 2 · page 3
> [![Figure 2 of the StreamingLLM paper: attention maps of Llama-2-7B where most layers put heavy attention on the first token](/img/attention/papers/xiao-sinks-figure.png)](/img/attention/papers/xiao-sinks-figure.png)
>
> **Context:** The authors averaged Llama-2-7B's attention over 256 short sentences and drew the maps for several layers (red = high, blue = low).
>
> **What it says:** Layers 0 and 1 attend mostly to nearby tokens (the red diagonal). From layer 2 on, almost every head puts a lot of attention on the **first** token: the red column on the left.
>
> **Why it matters:** That red first column is the attention sink, the same pattern we measured in 68% of Qwen2.5-0.5B's heads. It looks like a harmless quirk, but Part 3 shows that cutting off that first token breaks a model completely, and Part 4 shows a fix (gated attention).

## The cost that every other part tries to cut

Attention has two costs, and both grow with the length of the text:

1. **Work.** Every token compares itself with every earlier token. Twice as much text means **four times** as many comparisons.
2. **Memory.** When the model writes a reply one token at a time, it keeps every earlier token's keys and values in memory so it does not have to recompute them.

> [!DEFINITION] KV cache
> The stored keys (K) and values (V) of all the tokens so far, kept in memory while the model writes. Each new token adds its own keys and values to it. ([Part 2 of the LLM inference series](llm-inference-2-kv-cache.md) explains it in depth.)

With plain multi-head attention, every head in every layer stores a key and a value for every token. The memory per token is:

$$
\text{KV memory per token} = 2 \times \text{layers} \times \text{heads} \times d_{\text{head}} \times \text{bytes per number}
$$

where the 2 counts one key and one value, and $$d_{\text{head}}$$ is the length of each head's vectors. For Llama 2 7B that is $$2 \times 32 \times 32 \times 128 \times 2 = 524{,}288$$ bytes, or **512 KiB per token**. At its full 4,096-token length, one conversation needs **2 GiB** just for this memory.

The transformer paper itself lists this cost, comparing attention with the older layer types:

> [!PAPER] Vaswani et al. (2017) · Section 4, Why Self-Attention · page 6, Table 1
> [![Table 1 of Attention Is All You Need comparing self-attention, recurrent and convolutional layers, with the self-attention row highlighted](/img/attention/papers/vaswani-table1.png)](/img/attention/papers/vaswani-table1.png)
>
> **Context:** Section 4 argues why a network built only from attention is a good idea, by comparing three kinds of layer. $$n$$ is the text length and $$d$$ the vector length.
>
> **What it says:** Self-attention costs $$O(n^2 \cdot d)$$ per layer, but needs only $$O(1)$$ sequential steps, and any two tokens are connected in $$O(1)$$ steps. A recurrent layer costs $$O(n \cdot d^2)$$ but needs $$n$$ steps one after another, and information between distant tokens must pass through $$O(n)$$ steps.
>
> **Why it matters:** This table is the whole trade-off of the transformer. **No sequential steps** means training runs in parallel on GPUs, which is why transformers won. **$$n^2$$** is the bill, paid later, when texts became 100,000 tokens long. Parts 2 to 4 are about that $$n^2$$.

> [!DEFINITION] Big-O notation
> A shorthand for how a cost grows with size, ignoring constant factors. $$O(n^2)$$: double the text, four times the work. $$O(n)$$: double the text, double the work. $$O(1)$$: the cost does not depend on $$n$$ at all.

Every idea in the rest of the series cuts one of these two costs:

| Part | Idea | What it cuts |
|---|---|---|
| 2 | MQA, GQA, MLA | **memory**: each token stores fewer or smaller keys and values |
| 3 | sliding-window and sparse attention | **work and memory**: each token looks at fewer tokens |
| 4 | linear attention, Gated DeltaNet, hybrids | **both**: most layers stop keeping a growing memory at all |

## How attention changed everything

It is hard to overstate what these two papers started. A short timeline of what happened next:

| Year | What happened | Role of attention |
|---|---|---|
| 2014 | Bahdanau et al.: attention for translation | a helper for an RNN translator |
| 2017 | Vaswani et al.: the transformer | attention **replaces** the RNN entirely |
| 2018 | BERT and GPT | transformers pre-trained on huge amounts of text |
| 2020 | Vision Transformer (ViT) | images cut into patches and treated as tokens |
| 2021 to 2022 | AlphaFold 2, Whisper | attention for protein structures and for speech |
| 2022 onward | ChatGPT, Llama, Qwen, Gemma, DeepSeek | every large chatbot is a stack of attention layers |

> [!PAPER] Vaswani et al. (2017) · Abstract · page 1
> [![The abstract of Attention Is All You Need, with the English-to-French result of 41.8 BLEU after 3.5 days of training on eight GPUs highlighted](/img/attention/papers/vaswani-abstract-result.png)](/img/attention/papers/vaswani-abstract-result.png)
>
> **Context:** The abstract's main claim: the new model is both better and cheaper to train than the best translation systems of the time.
>
> **What it says:** A new single-model record of 41.8 BLEU on English-to-French "after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature".
>
> **Why it matters:** "Better" got attention noticed; "cheaper to train" made it take over. Because attention trains in parallel, researchers could afford much bigger models on much more text, and that scaling is what eventually produced today's chatbots.

> [!DEFINITION] BLEU score
> A standard score for machine translation, from 0 to 100. It counts how many short word sequences in the model's translation also appear in human reference translations. Higher is better; a gain of 1 to 2 points was considered a big step at the time.

## Use cases: where you meet this today

Everything in this part runs, almost unchanged, inside products you use:

- **Translation.** The original use case. Attention lets the model align words across languages, as Figure 3 of Bahdanau et al. showed above.
- **Chatbots and writing assistants.** ChatGPT, Gemini, Llama, Qwen and DeepSeek are stacks of the multi-head attention built in this part (plus the efficiency tricks of Parts 2 to 4). Each word they write is one round of exactly this attention over the conversation so far.
- **Search and understanding.** Encoder transformers such as BERT read a query and documents and match their meaning, not just their words. The heads Clark et al. found (objects, determiners, coreference) are part of why this works.
- **Code assistants.** The same attention links a variable to where it was defined, or a closing bracket to its opening one, exactly like the previous-token and pronoun heads above.
- **Long conversations.** Attention sinks matter in practice: some systems that keep a chat going for a very long time keep the first few tokens in memory for exactly this reason (Part 3 tests it).

The translation results that started it all:

> [!PAPER] Vaswani et al. (2017) · Section 6.1, Machine Translation · page 8, Table 2
> [![Table 2 of Attention Is All You Need: BLEU scores and training costs of earlier translation systems and the Transformer, with the Transformer (big) row highlighted](/img/attention/papers/vaswani-bleu.png)](/img/attention/papers/vaswani-bleu.png)
>
> **Context:** The main results table: earlier systems, including ensembles of several models, against the transformer on two standard translation tests (English to German, English to French).
>
> **What it says:** The big transformer scores 28.4 (EN-DE) and 41.8 (EN-FR), beating every earlier system including ensembles. Its training cost (last two columns, in FLOPs) is about 3 to 50 times lower than those ensembles, and the smaller base model is cheaper still.
>
> **Why it matters:** This is the use case that made attention famous. Translation was the hardest benchmark of its day, and a model built *only* from the attention in this part won it.

## Summary

- Models turn text into **tokens** and tokens into **vectors** (lists of numbers).
- Attention was invented so a model could **look back** at every word instead of squeezing a sentence into one summary.
- **Self-attention:** each token makes a query, a key and a value. Queries are compared with keys (dot products), softmax turns the scores into weights, and the weights mix the values: $$\operatorname{softmax}(QK^\top/\sqrt{d} + M)\,V$$.
- The **causal mask** stops tokens from seeing the future; every row of weights adds up to 1.
- **Dividing by √d** keeps scores from growing with vector length. Without it, the top weight reached 0.95; with it, 0.107.
- **Multi-head attention** runs several heads side by side. Our version matched PyTorch's to $$1.7 \times 10^{-16}$$.
- Real heads specialise: in Qwen2.5-0.5B, **68%** are attention sinks, one follows the previous token (84%), and one links pronouns to nouns (6 out of 6 test sentences). BERT (Clark et al.) and Llama-2 (Xiao et al.) show the same patterns.
- The √d comes from simple statistics: the dot product of two random vectors has variance $$d$$, so dividing by $$\sqrt{d}$$ brings it back to 1.
- The transformer won because attention needs no sequential steps (fast, parallel training), at the price of $$O(n^2)$$ work.
- Attention's work grows with the square of the text length, and its memory (the KV cache) with the length. Parts 2 to 4 are about cutting both.

<details>
<summary>Run it yourself</summary>

The script that produced every number above is [`code/attention/part1_attention.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part1_attention.py), and the six-sentence test is [`part1_coref_check.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part1_coref_check.py). They need Python with `torch` and `transformers`, and download Qwen2.5-0.5B (about 1 GB) the first time.

```bash
pip install torch transformers
python part1_attention.py      # writes results/part1.json
python part1_coref_check.py    # the six pronoun sentences
```

Reading a real model's attention weights takes one setting:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", attn_implementation="eager")
ids = tok("The cat sat on the mat because it was tired.", return_tensors="pt").input_ids
attentions = model(ids, output_attentions=True).attentions   # one tensor per layer: (batch, heads, tokens, tokens)
```

</details>

## References

1. D. Bahdanau, K. Cho, Y. Bengio. [*Neural Machine Translation by Jointly Learning to Align and Translate*](https://arxiv.org/abs/1409.0473). 2014.
2. A. Vaswani et al. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
3. G. Xiao et al. [*Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453). ICLR 2024.
4. K. Clark, U. Khandelwal, O. Levy, C. Manning. [*What Does BERT Look At? An Analysis of BERT's Attention*](https://arxiv.org/abs/1906.04341). 2019.
5. Qwen Team. [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
