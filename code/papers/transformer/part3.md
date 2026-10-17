---
title: "Attention: Queries, Keys and Values"
description: "How one attention step works, why the scores are divided by the square root of the key size, and why the model runs eight heads side by side. We write attention and multi-head attention from scratch in Python, match them with the PyTorch library and with a released translation model, and show three common bugs."
part: 3
covers: "§3.2, Figure 2, Equation 1, footnote 4, §3.2.2, §3.2.3"
date: 2026-10-18
tags: [transformer, attention, multi-head-attention, nlp, deep-learning]
---

Attention is the one step in the Transformer where a word gets information from other words. Each piece below shows the exact lines of the paper, then explains them, and then checks them on a real translation model.

## What an attention function is {§3.2}

> [!PAPER] · §3.2
> [![The first lines of Section 3.2, with mapping a query and a set of key-value pairs to an output and weighted sum highlighted](/img/papers/transformer/p3-attention-def.png)](/img/papers/transformer/p3-attention-def.png)
> [![The next line on page 4, with compatibility function of the query with the corresponding key highlighted](/img/papers/transformer/p3-weights.png)](/img/papers/transformer/p3-weights.png)
>
> **What it says:** attention is "mapping a query and a set of key-value pairs to an output". The output is "a weighted sum of the values", and each weight comes from "a compatibility function of the query with the corresponding key".
>
> **What it means:**
>
> Think of a library. You walk in with a question: that is the **query**. Each book has a label on its spine: that is its **key**. Each book also has content inside: that is its **value**. You compare your question with every label, and you read the best matches most carefully.
>
> The paper says the same thing with numbers. Each query, key and value is a vector, a list of numbers. The "compatibility function" gives one score for each key: how well that key matches the query. The scores become weights that are positive and add up to 1. The output is the values mixed with those weights.
>
> The important word is "sum". Attention does not pick one book. It takes a little from every book, and a lot from the books that match. Because the mix is smooth, the network can learn the weights with gradient descent.

> [!DEFINITION] Query, key and value
> Three vectors that the network makes from each token with three learned matrices. The **query** asks "what am I looking for?". The **key** says "this is what I contain". The **value** is "this is what I give if you pick me". A token can have a key that is quite different from its value.

> [!DEFINITION] Token
> One piece of the text that the model reads as a unit: a word or a part of a word. Our translation model cuts "The cat sat on the mat because it was tired." into 12 tokens, including the full stop and an end marker.

## Figure 2: the two pictures of attention {§Figure 2}

> [!PAPER] · Figure 2
> [![Figure 2 of the paper. Left: Scaled Dot-Product Attention, a stack of boxes MatMul, Scale, Mask (opt.), SoftMax, MatMul, with Q and K entering the first MatMul and V entering the last. Right: Multi-Head Attention, with V, K and Q each passing through a Linear box, then h copies of Scaled Dot-Product Attention, then Concat and a final Linear box](/img/papers/transformer/p3-figure2.png)](/img/papers/transformer/p3-figure2.png)
>
> **What it says:** "(left) Scaled Dot-Product Attention. (right) Multi-Head Attention consists of several attention layers running in parallel."
>
> **What it means:**
>
> Read the left picture from the bottom up. $$Q$$ and $$K$$ meet in the first "MatMul": every query is multiplied with every key, which gives a table of scores. "Scale" divides the scores by a fixed number. "Mask (opt.)" hides some scores; only the decoder uses it. "SoftMax" turns each row of scores into weights. The last "MatMul" mixes the values $$V$$ with those weights.
>
> The right picture wraps the left one. The three "Linear" boxes make new queries, keys and values from the inputs. The stacked shadows behind each box, and the letter $$h$$, say that this happens $$h$$ times with different matrices. "Concat" glues the $$h$$ results together, and a last "Linear" box mixes them.
>
> The rest of this page follows the figure. First comes the left picture and the reason for "Scale", then the right picture, and last the "Mask".

## Scaled dot-product attention {§3.2.1}

> [!PAPER] · §3.2.1
> [![The first two paragraphs of Section 3.2.1, with queries and keys of dimension dk, values of dimension dv, dot products of the query with all keys, and apply a softmax function highlighted](/img/papers/transformer/p3-sdpa.png)](/img/papers/transformer/p3-sdpa.png)
>
> **What it says:** the queries and keys have "dimension $$d_k$$" and the values have "dimension $$d_v$$". The model computes "the dot products of the query with all keys", divides each by $$\sqrt{d_k}$$, and applies a softmax to get the weights on the values. In practice, all the queries are "packed together into a matrix $$Q$$".
>
> **What it means:**
>
> The "compatibility function" is now concrete: it is the **dot product**. Multiply the two vectors number by number, then add up the results. Two vectors that point in the same direction give a large dot product. Two vectors at right angles give zero.
>
> The second paragraph is about speed. One query at a time would be a slow loop. So the paper stacks all queries as rows of one matrix $$Q$$, and all keys and values as rows of $$K$$ and $$V$$. A graphics card then computes every score at once with one matrix product.

> [!DEFINITION] Dot product
> For two vectors of the same length, $$q \cdot k = q_1 k_1 + q_2 k_2 + \dots + q_d k_d$$. For example, $$(1, 2) \cdot (3, 4) = 1 \times 3 + 2 \times 4 = 11$$. A large positive value means the two vectors point in a similar direction.

> [!DEFINITION] Softmax
> A function that turns a list of numbers into weights that are positive and add up to 1: $$\text{softmax}(s)_i = e^{s_i} / \sum_j e^{s_j}$$. The largest number gets the largest weight. For the scores (2, 1, 0), the weights are 0.665, 0.245 and 0.090.

> [!PAPER] · Equation 1
> [![Equation 1 of the paper, highlighted: Attention(Q, K, V) equals softmax of Q K transpose divided by the square root of d k, times V](/img/papers/transformer/p3-eq1.png)](/img/papers/transformer/p3-eq1.png)
>
> **What it says:** "We compute the matrix of outputs as" $$\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k})V$$.
>
> **What it means:**
>
> This one line is the whole left half of Figure 2. Each piece of the line is one box of the figure. The next section reads it symbol by symbol and runs it on three tiny tokens.

### Equation 1, step by step

The equation does four things in order: score every pair, shrink the scores, turn them into weights, mix the values.

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left( \frac{Q K^\top}{\sqrt{d_k}} \right) V
$$

Read it from left to right:

- $$Q$$ is a table with one row for each query token. Each row has $$d_k$$ numbers.
- $$K$$ is a table with one row for each key token, also $$d_k$$ numbers per row. $$K^\top$$ is the same table turned on its side.
- $$Q K^\top$$ is the table of all dot products. Row $$i$$, column $$j$$ is the score of query $$i$$ against key $$j$$.
- $$\sqrt{d_k}$$ divides every score by the same number. The next section explains why.
- $$\text{softmax}$$ works on each row on its own. Each row becomes weights that add up to 1.
- $$V$$ is the table of values. The weights multiply $$V$$, so each output row is a weighted average of the value rows.

A tiny example with three tokens and $$d_k = 2$$. The three queries are $$(1, 0)$$, $$(0, 1)$$ and $$(1, 1)$$. The three keys are $$(2, 0)$$, $$(0, 2)$$ and $$(1, 1)$$. The three values are $$(10, 0)$$, $$(0, 10)$$ and $$(5, 5)$$.

Take the first query, $$(1, 0)$$. Its dot products with the three keys are 2, 0 and 1. Divide by $$\sqrt{2} = 1.414$$, and the scores are 1.414, 0 and 0.707. The softmax gives:

$$
\frac{e^{1.414}}{e^{1.414} + e^{0} + e^{0.707}} = \frac{4.113}{7.141} = 0.576, \qquad \frac{1}{7.141} = 0.140, \qquad \frac{2.028}{7.141} = 0.284
$$

The output is $$0.576 \times (10, 0) + 0.140 \times (0, 10) + 0.284 \times (5, 5) = (7.18, 2.82)$$. The query points along the first axis, like the first key, so the output is mostly the first value. Our script gives the same numbers:

```text
== 1. Eq. 1 on three tokens, d_k = 2
scores QK^T/sqrt(2): [[1.414, 0.0, 0.707], [0.0, 1.414, 0.707], [1.414, 1.414, 1.414]]
weights (softmax, row 1): [0.576, 0.14, 0.284]
output row 1: [7.18, 2.82]
```

Look at the third row of scores: 1.414 three times. The query $$(1, 1)$$ matches all three keys equally, so it gets one third of each value. Attention can share its focus when nothing stands out.

### Method 1: Equation 1 from scratch, every matrix printed

Here is the same computation as code, so you can follow each table. NumPy is a Python library for tables of numbers. In NumPy, `@` is the matrix product and `.T` turns a table on its side.

```python
Q = np.array([[1., 0.], [0., 1.], [1., 1.]])     # 3 queries, one row per token
K = np.array([[2., 0.], [0., 2.], [1., 1.]])     # 3 keys
V = np.array([[10., 0.], [0., 10.], [5., 5.]])   # 3 values
d_k = Q.shape[1]                                  # 2 numbers per query and key

scores = Q @ K.T                                  # (3, 3): row i, column j = query i . key j
scaled = scores / np.sqrt(d_k)                    # divide every score by sqrt(2) = 1.414
e = np.exp(scaled - scaled.max(axis=1, keepdims=True))  # subtract the row maximum first: same softmax, no overflow
weights = e / e.sum(axis=1, keepdims=True)        # each row now adds up to 1
out = weights @ V                                 # (3, 2): each output row is a weighted average of the value rows
```

The script prints every table on the way:

```text
Q K^T  shape (3, 3)
     2.000   0.000   1.000
     0.000   2.000   1.000
     2.000   2.000   2.000
Q K^T / sqrt(d_k)  shape (3, 3)
     1.414   0.000   0.707
     0.000   1.414   0.707
     1.414   1.414   1.414
weights = softmax(rows)  shape (3, 3)
     0.576   0.140   0.284
     0.140   0.576   0.284
     0.333   0.333   0.333
row sums: [1.0, 1.0, 1.0]
output = weights V  shape (3, 2)
      7.18    2.82
      2.82    7.18
      5.00    5.00
```

Read the tables in order. `Q K^T` holds the raw dot products: row 1 is 2, 0 and 1, as we computed by hand. The next table is the same divided by 1.414. The softmax turns each row into weights, and the row sums show that each row adds up to 1. The last table is the output, and its row 1 is (7.18, 2.82), as above.

One line needs a word. Before the exponential, the code subtracts the largest score of each row. This does not change the softmax, because the same factor $$e^{-m}$$ appears above and below the fraction. It only keeps $$e^{s}$$ from growing too large for the computer when a score is big.

> [!DEFINITION] Tensor and shape
> A **tensor** is a table of numbers with any number of axes: a vector has one axis, a matrix has two. Its **shape** lists the size of each axis. A shape of (2, 8, 10, 64) means 2 sentences, 8 heads, 10 tokens and 64 numbers per token.

### Method 2: the library function

PyTorch is the library that most Transformer code uses today. Below is Equation 1 as a PyTorch function, which the rest of this page uses again and again. It also takes an optional mask, which the section on the decoder mask explains. The lines of the body follow the boxes of Figure 2: MatMul and Scale, then Mask, SoftMax and the last MatMul.

```python
def attention(Q, K, V, mask=None):
    """Equation 1: softmax(Q K^T / sqrt(d_k)) V.
    Q: (..., n_q, d_k)   K: (..., n_k, d_k)   V: (..., n_k, d_v)
    mask: True where a query may look at a key, False where it may not (None = no mask)."""
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)       # (..., n_q, n_k): every query against every key
    if mask is not None:
        scores = scores.masked_fill(~mask, float('-inf'))    # forbidden pairs get minus infinity, BEFORE the softmax
    weights = torch.softmax(scores, dim=-1)                  # (..., n_q, n_k): each row adds up to 1
    return weights @ V, weights                              # (..., n_q, d_v), (..., n_q, n_k)
```

PyTorch also has its own version, `torch.nn.functional.scaled_dot_product_attention`. We call both on the three tokens, and then on random tensors with the shape of a real attention step:

```python
Qt, Kt, Vt = (torch.tensor(a, dtype=torch.float32) for a in (Q, K, V))
ours, _ = attention(Qt, Kt, Vt)                       # Method 1: our function
lib = F.scaled_dot_product_attention(Qt, Kt, Vt)       # Method 2: the library
d1 = (ours - lib).abs().max().item()                   # the largest difference over all outputs
q, k, v = torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64), torch.randn(2, 8, 10, 64)   # (batch, heads, n, d_k)
ours, _ = attention(q, k, v)
lib = F.scaled_dot_product_attention(q, k, v)
d2 = (ours - lib).abs().max().item()
```

```text
three tokens, ours   : [[7.1797, 2.8203], [2.8203, 7.1797], [5.0, 5.0]]
three tokens, library: [[7.1797, 2.8203], [2.8203, 7.1797], [5.0, 5.0]]
max abs difference: 4.8e-07
random (batch 2, heads 8, n 10, d_k 64): output (2, 8, 10, 64), max abs difference 6.0e-07
```

The two methods agree to within $$6 \times 10^{-7}$$. That is not an error in either one. A computer stores each number of this kind with about 7 correct digits, and the library can add the products in a different order. The second test has 2 sentences, 8 heads, 10 tokens and 64 numbers per head. The `@` product works on the last two axes and repeats for every sentence and head, so one function handles both cases.

## Dot product or additive attention {§3.2.1}

> [!PAPER] · §3.2.1
> [![The paragraph that compares additive attention and dot-product attention, with additive attention [2], except for the scaling factor and much faster and more space-efficient in practice highlighted](/img/papers/transformer/p3-additive.png)](/img/papers/transformer/p3-additive.png)
>
> **What it says:** the two common kinds of attention are "additive attention [2]" and dot-product attention. Dot-product attention is the same as theirs "except for the scaling factor". It is "much faster and more space-efficient in practice", because it uses "highly optimized matrix multiplication code".
>
> **What it means:**
>
> Reference [2] is Bahdanau attention, from the [Background page](background-the-road-to-the-transformer.md). It scores a query against a key with a small neural network. Each vector goes through a matrix, the two are added, and tanh and one more dot product give the score. That is a good score, but each pair of tokens needs its own trip through the network.
>
> The dot product needs no network. All scores come out of one matrix product, $$Q K^\top$$. Graphics cards are built for exactly this operation, so it runs fast and needs no extra memory for hidden layers. The authors chose the cheaper score and fixed its one weakness with the scale factor.

## Why divide by the square root of d_k {§3.2.1}

> [!PAPER] · §3.2.1
> [![The paragraph about large values of dk, with additive attention outperforms, dot products grow large in magnitude and extremely small gradients highlighted](/img/papers/transformer/p3-scale-why.png)](/img/papers/transformer/p3-scale-why.png)
>
> **What it says:** for small $$d_k$$ the two kinds of attention perform alike, but "additive attention outperforms" unscaled dot-product attention for large $$d_k$$. The authors suspect that the "dot products grow large in magnitude", which pushes the softmax into regions with "extremely small gradients". So they scale the dot products by $$1/\sqrt{d_k}$$.
>
> **What it means:**
>
> A softmax that gets very large scores stops being a soft choice. One weight goes to almost 1 and the others go to almost 0. Think of a thermostat that only knows "off" and "full power". You cannot tune it by a small step, because a small step changes nothing.
>
> This is the problem with gradients. Training changes the scores by small steps and watches how the loss changes. When the softmax is already at 0 or 1, a small change of a score does not move the weights at all. The gradient is close to zero, and the attention stops learning.
>
> Note the word "suspect". The authors give a reason that fits the facts, and the footnote below backs it with a short calculation. It is an argument, not a proof.

> [!DEFINITION] Gradient
> How much the output of a function changes when one input changes by a tiny amount. Training uses gradients to decide which way to move each weight. A gradient close to zero means "a change here does nothing", so that weight stops learning.

> [!PAPER] · Footnote 4
> [![Footnote 4 of the paper, with mean 0 and variance 1 and has mean 0 and variance dk highlighted](/img/papers/transformer/p3-footnote.png)](/img/papers/transformer/p3-footnote.png)
>
> **What it says:** "assume that the components of $$q$$ and $$k$$ are independent random variables with mean 0 and variance 1. Then their dot product, $$q \cdot k = \sum_{i=1}^{d_k} q_i k_i$$, has mean 0 and variance $$d_k$$."
>
> **What it means:**
>
> A dot product is a sum of $$d_k$$ small products. Each product $$q_i k_i$$ is random, sometimes positive and sometimes negative, with variance 1. When you add up independent random numbers, their variances add up. So the sum of 64 of them has variance 64, and a typical size of $$\sqrt{64} = 8$$.
>
> Now the fix is clear. Divide by $$\sqrt{d_k}$$, and the typical size goes back to 1, whatever $$d_k$$ is. The scale factor is not a tuned number. It is exactly the number that cancels the growth.

> [!DEFINITION] Variance and standard deviation
> The **variance** measures how far random values spread around their mean: the average of the squared distances from the mean. The **standard deviation** (std) is its square root, in the same units as the values. For a variance of 64, the std is 8.

The equation behind the footnote, with its assumption:

$$
\operatorname{Var}(q \cdot k) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k \quad\Rightarrow\quad \operatorname{Var}\!\left( \frac{q \cdot k}{\sqrt{d_k}} \right) = \frac{d_k}{d_k} = 1
$$

Read it: the variance of the dot product is the sum of the variances of its $$d_k$$ terms. This holds when the terms are independent. Each term has variance 1 when $$q_i$$ and $$k_i$$ have mean 0 and variance 1. Dividing a random number by $$c$$ divides its variance by $$c^2$$, so dividing by $$\sqrt{d_k}$$ divides the variance by $$d_k$$.

A tiny check by hand with $$d_k = 4$$: each term is about $$\pm 1$$, so a sum of four terms often lands near $$\pm 2$$, which is $$\sqrt{4}$$. Our script drew 100,000 random pairs for four sizes:

```text
== 2. dot products of random vectors (mean 0, variance 1), 100,000 pairs each
d_k=   4  mean +0.003  variance     3.99  std   2.00  scaled std 0.999
d_k=  16  mean +0.012  variance    15.92  std   3.99  scaled std 0.997
d_k=  64  mean -0.034  variance    64.01  std   8.00  scaled std 1.000
d_k= 512  mean -0.077  variance   514.16  std  22.68  scaled std 1.002
```

The variance follows $$d_k$$, as the footnote says, and the scaled std stays at 1.00. Next, what this does to the softmax. We took one random query and ten random keys, and computed the weights with and without the scale. We also computed the largest entry of the softmax gradient (its Jacobian matrix). For a softmax, no entry can be larger than 0.25.

```text
== softmax over 10 keys, one query, d_k = 64 (same random draw, scaled and not scaled)
unscaled64  scores range [-15.8, +9.5]  top weight 0.8407  largest gradient entry 0.1339
scaled64    scores range [-2.0, +1.2]  top weight 0.2484  largest gradient entry 0.1867
== softmax over 10 keys, one query, d_k = 512 (same random draw, scaled and not scaled)
unscaled512 scores range [-60.1, +46.0]  top weight 1.0000  largest gradient entry 0.0000
scaled512   scores range [-2.7, +2.0]  top weight 0.5176  largest gradient entry 0.2497
```

With $$d_k = 512$$ and no scale, the scores spread from $$-60$$ to $$+46$$. The softmax puts a weight of 1.0000 on one key, and every gradient entry rounds to 0.0000. That attention head cannot learn. With the scale, the same vectors give a top weight of 0.52 and a gradient entry of 0.25, the largest a softmax allows. This is a toy with random vectors, not a trained model, but it shows the effect the paper describes.

## Multi-head attention {§3.2.2}

> [!PAPER] · §3.2.2
> [![The first paragraph of Section 3.2.2, with linearly project the queries, keys and values h times and perform the attention function in parallel highlighted](/img/papers/transformer/p3-multihead-1.png)](/img/papers/transformer/p3-multihead-1.png)
>
> **What it says:** instead of one attention function with $$d_{\text{model}}$$-dimensional vectors, the authors "linearly project the queries, keys and values $$h$$ times" with different, learned projections to $$d_k$$, $$d_k$$ and $$d_v$$ dimensions. On each version they "perform the attention function in parallel".
>
> **What it means:**
>
> One attention step gives each token one set of weights: one way to read the sentence. But a word often needs several things at once. "it" needs to know what it refers to. It also needs to know which verb comes next, and whether it is a subject. One weighted average must mix all of these into one blend.
>
> So the model runs several small attention steps side by side, the **heads**. Each head has its own three matrices, so it asks its own question with its own queries and keys. Think of a team of eight readers. One looks for the subject, one for the verb, one for the punctuation. Each writes a short note, and the notes are put together.

> [!DEFINITION] Linear projection
> Multiplying a vector by a learned matrix to get a new vector, often shorter. A 512-number vector times a 512 × 64 matrix gives a 64-number vector. Each head has its own matrices, so each head sees a different short summary of the same token.

> [!PAPER] · §3.2.2
> [![The rest of Section 3.2.2: the outputs are concatenated and projected; the MultiHead equation with Concat of head 1 to head h times W O, where head i is Attention of Q W i Q, K W i K, V W i V; the shapes of the projection matrices; and h = 8 heads with d k = d v = d model / h = 64](/img/papers/transformer/p3-multihead-2.png)](/img/papers/transformer/p3-multihead-2.png)
>
> **What it says:** the head outputs are "concatenated and once again projected". This lets the model attend to "different representation subspaces at different positions"; with a single head, "averaging inhibits this". The paper uses "$$h = 8$$ parallel attention layers", with $$d_k = d_v = d_{\text{model}}/h = 64$$, so the cost is "similar to that of single-head attention with full dimensionality".
>
> **What it means:**
>
> "Averaging inhibits this" is the key phrase. One head that must attend to the subject and to the verb ends up with a blurred average of both. Eight heads can each keep one sharp focus.
>
> The cost does not grow. The paper does not give each head the full 512 numbers. It cuts them into 8 pieces of 64. Eight heads of 64 numbers do about the same work as one head of 512 numbers. The model gets eight views for the price of one.
>
> After the heads, "Concat" puts the eight outputs of 64 numbers back next to each other: 512 numbers again. The matrix $$W^O$$ then mixes the notes of the eight heads into one vector for each token.

### The multi-head equation

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O, \qquad \text{head}_i = \text{Attention}(Q W_i^Q,\; K W_i^K,\; V W_i^V)
$$

Read it from the right. $$W_i^Q$$, $$W_i^K$$ and $$W_i^V$$ are the three matrices of head $$i$$, each of size $$512 \times 64$$. They turn the inputs into the queries, keys and values of that head. $$\text{Attention}$$ is Equation 1. $$\text{Concat}$$ puts the $$h$$ outputs side by side. $$W^O$$, of size $$512 \times 512$$, mixes them.

The parameter count with the base shapes: each head has three matrices of $$512 \times 64$$, and eight heads give $$3 \times 8 \times 512 \times 64 = 786{,}432$$ numbers. That is exactly three matrices of $$512 \times 512$$. Add $$W^O$$ and one bias vector of 512 for each of the four matrices:

$$
4 \times 512 \times 512 + 4 \times 512 = 1{,}050{,}624
$$

### Multi-head attention from scratch, line by line

The multi-head equation hides one trick of the code. The paper writes $$h$$ separate matrices $$W_i^Q$$. Code keeps them side by side as one wide matrix $$W^Q$$ of size $$d_{\text{model}} \times d_{\text{model}}$$. It makes the queries of all heads with one product, and then cuts the result into $$h$$ pieces. The cut is two calls: `view` and `transpose`.

We follow a small example with real tensors: 2 sentences of 5 tokens, $$d_{\text{model}} = 16$$ and $$h = 4$$ heads, so $$d_k = 4$$. The comment at the end of a line says what it does.

```python
batch, n, d_model, h = 2, 5, 16, 4
d_k = d_model // h                                  # 4 numbers per head
x = torch.randn(batch, n, d_model)                  # the input: 2 sentences of 5 tokens, 16 numbers each
p = {f'W_{m}': torch.randn(d_model, d_model) / math.sqrt(d_model) for m in 'QKVO'}   # paper layout: x @ W
p.update({f'b_{m}': torch.randn(d_model) * 0.1 for m in 'QKVO'})

Qm = x @ p['W_Q'] + p['b_Q']                         # all 4 heads' queries in one product
Qs = Qm.view(batch, n, h, d_k)                       # cut the last axis into h pieces of d_k
Qh = Qs.transpose(1, 2)                              # move heads before tokens: each head is its own problem
Kh = (x @ p['W_K'] + p['b_K']).view(batch, n, h, d_k).transpose(1, 2)   # keys: the same three steps
Vh = (x @ p['W_V'] + p['b_V']).view(batch, n, h, d_k).transpose(1, 2)   # values: the same three steps
S = Qh @ Kh.transpose(-2, -1) / math.sqrt(d_k)       # one n x n score table per head
Wt = torch.softmax(S, dim=-1)                        # each row of each table adds up to 1
heads = Wt @ Vh                                      # each head mixes its own values
merged = heads.transpose(1, 2)                       # tokens first again
concat = merged.reshape(batch, n, h * d_k)           # the h pieces side by side: the paper's Concat
y_steps = concat @ p['W_O'] + p['b_O']               # W^O mixes what the heads found
```

The script prints the shape after every line:

```text
x                                    (2, 5, 16)   (batch, n, d_model)
Q = x W_Q + b_Q                      (2, 5, 16)   (batch, n, d_model)
Q.view(batch, n, h, d_k)             (2, 5, 4, 4)   (batch, n, h, d_k)
.transpose(1, 2)                     (2, 4, 5, 4)   (batch, h, n, d_k)
K, V the same way                    (2, 4, 5, 4)   (batch, h, n, d_k)
Q K^T / sqrt(d_k)                    (2, 4, 5, 5)   (batch, h, n, n)
softmax over the last axis           (2, 4, 5, 5)   (batch, h, n, n)
weights V                            (2, 4, 5, 4)   (batch, h, n, d_k)
.transpose(1, 2)                     (2, 5, 4, 4)   (batch, n, h, d_k)
.reshape(batch, n, h * d_k)          (2, 5, 16)   (batch, n, d_model)
concat W_O + b_O                     (2, 5, 16)   (batch, n, d_model)
```

{{FIG:p3_split|The split of multi-head attention, for 5 tokens of 16 numbers and 4 heads. View cuts each row of 16 numbers into 4 groups of 4, one colour per head. Transpose then makes 4 separate tables, one per head, each with the same 5 tokens and 4 numbers per token.}}

Three lines do the real work. `view(batch, n, h, d_k)` reads the 16 numbers of each token as 4 groups of 4. No number moves; only the way we count them changes. `transpose(1, 2)` swaps the token axis and the head axis. Now the tensor is a stack of $$2 \times 4 = 8$$ small problems, one for each sentence and head.

Equation 1 then runs on all 8 problems at once, because `@` repeats over the leading axes. The scores have the shape (2, 4, 5, 5): one 5 × 5 table for each sentence and head. The way back is the mirror image. `transpose(1, 2)` puts the tokens first again, and `reshape` glues the 4 pieces of each token into 16 numbers. That is the paper's Concat, and $$W^O$$ then mixes the pieces.

> [!DEFINITION] Batch
> A group of inputs that the model processes together, as one tensor. Here the batch has 2 sentences. The batch axis comes first, and every step works on each sentence on its own.

### The block as three functions

The same steps fit in three short functions. `split_heads` and `merge_heads` are the two reshapes. `multi_head_attention` is the multi-head equation. It takes two inputs: `x_q` makes the queries, and `x_kv` makes the keys and values. For self-attention we pass the same tensor twice; for cross-attention we pass two different ones.

```python
def split_heads(t, h):
    """(batch, n, d_model) -> (batch, h, n, d_k): cut each token vector into h pieces, one per head."""
    batch, n, d_model = t.shape
    return t.view(batch, n, h, d_model // h).transpose(1, 2)

def merge_heads(t):
    """(batch, h, n, d_k) -> (batch, n, h * d_k): put the h pieces of each token side by side (Concat)."""
    batch, h, n, d_k = t.shape
    return t.transpose(1, 2).reshape(batch, n, h * d_k)

def multi_head_attention(x_q, x_kv, p, h, mask=None):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O, with head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V).
    x_q (batch, n_q, d_model) makes the queries. x_kv (batch, n_k, d_model) makes the keys and values:
    the same tensor for self-attention, the encoder output for cross-attention.
    p: W_Q, W_K, W_V, W_O of shape (d_model, d_model) and b_Q, b_K, b_V, b_O of shape (d_model,).
    The h matrices W_i^Q of the paper are the h column blocks of W_Q, each d_model x d_k."""
    Q = split_heads(x_q @ p['W_Q'] + p['b_Q'], h)     # (batch, h, n_q, d_k)
    K = split_heads(x_kv @ p['W_K'] + p['b_K'], h)    # (batch, h, n_k, d_k)
    V = split_heads(x_kv @ p['W_V'] + p['b_V'], h)    # (batch, h, n_k, d_v)
    heads, weights = attention(Q, K, V, mask)         # (batch, h, n_q, d_v), (batch, h, n_q, n_k)
    return merge_heads(heads) @ p['W_O'] + p['b_O'], weights   # (batch, n_q, d_model)
```

```text
step by step vs multi_head_attention(): max abs difference 0.0e+00
```

The function and the line-by-line code give the same output, with a difference of exactly 0. They do the same products in the same order.

### Method 2: torch.nn.MultiheadAttention

PyTorch has a ready-made block, `torch.nn.MultiheadAttention`. To compare it with our code, we copy our matrices into it. There is one trap. The library stores each matrix as (output size, input size) and computes $$x W^\top$$, while the paper writes $$x W$$. So we copy the transpose. The library also keeps $$W^Q$$, $$W^K$$ and $$W^V$$ in one tall matrix, `in_proj_weight`.

```python
mha = torch.nn.MultiheadAttention(d_model, h, batch_first=True)
with torch.no_grad():
    # The library stores each matrix as (out, in) and computes x @ W.T, so we copy the transpose.
    mha.in_proj_weight.copy_(torch.cat([p['W_Q'].T, p['W_K'].T, p['W_V'].T]))   # (3 d_model, d_model)
    mha.in_proj_bias.copy_(torch.cat([p['b_Q'], p['b_K'], p['b_V']]))
    mha.out_proj.weight.copy_(p['W_O'].T)
    mha.out_proj.bias.copy_(p['b_O'])
    y_lib, w_lib = mha(x, x, x, need_weights=True, average_attn_weights=False)
d_out = (y_ours - y_lib).abs().max().item()                  # Method 1 against Method 2: outputs
d_w = (w_ours - w_lib).abs().max().item()                      # and the attention weights of every head
causal = torch.tril(torch.ones(n, n, dtype=torch.bool))       # True = may look
with torch.no_grad():
    y_ours_c, _ = multi_head_attention(x, x, p, h, mask=causal)
    y_lib_c, _ = mha(x, x, x, attn_mask=~causal)                # nn.MultiheadAttention: True = may NOT look
d_causal = (y_ours_c - y_lib_c).abs().max().item()
n_par = sum(t.numel() for t in mha.parameters())
```

```text
library output (2, 5, 16), weights (2, 4, 5, 5)
output : max abs difference 4.8e-07
weights: max abs difference 1.2e-07
with the causal mask (library gets ~mask): max abs difference 3.6e-07
parameters: 1088 = 4 x 16 x 16 + 4 x 16
```

The outputs agree to $$4.8 \times 10^{-7}$$, and so do the attention weights of all 4 heads. The parameter count follows the formula above: 4 matrices of 16 × 16 and 4 bias vectors of 16 make 1,088.

Look at the mask line. For `nn.MultiheadAttention`, True in `attn_mask` means "may NOT look". That is the opposite of `scaled_dot_product_attention`, where True means "may look". So we pass `~causal`, the same table with True and False swapped. Libraries do not agree on this point, so check the meaning of each mask you use.

### Multi-head attention by hand, on a real model

The paper's own weights are not easy to run today. So we used a released model with the same shapes: Helsinki-NLP/opus-mt-en-de, an English-to-German translation Transformer. It has 6 encoder and 6 decoder layers, $$d_{\text{model}} = 512$$ and 8 heads. It is not the paper's model: it was trained on other data, with another tokenizer and other settings.

```text
== 3. Helsinki-NLP/opus-mt-en-de (MarianMT)
d_model 512, heads 8, d_k = d_v = 64, encoder layers 6, decoder layers 6
  q_proj   weight (512, 512)
  k_proj   weight (512, 512)
  v_proj   weight (512, 512)
  out_proj weight (512, 512)
parameters in one multi-head attention block: 1,050,624 (4 x 512 x 512 + 4 x 512 = 1,050,624)
source tokens: ['▁The', '▁cat', '▁sat', '▁on', '▁the', '▁mat', '▁because', '▁it', '▁was', '▁tired', '.', '</s>']
translation (greedy): Die Katze saß auf der Matte, weil sie müde war.
```

The library stores the eight $$W_i^Q$$ matrices as one $$512 \times 512$$ matrix, `q_proj`. Columns 1 to 64 belong to head 1, columns 65 to 128 to head 2, and so on. The count of 1,050,624 matches our hand count.

Then we caught the input of the first encoder attention block, and computed Equation 1 and the multi-head equation ourselves, with the model's weights:

```text
== 4. encoder layer 1 self-attention rebuilt by hand
x (12, 512) -> Q,K,V per head (8, 12, 64) -> weights (8, 12, 12) -> heads (8, 12, 64) -> concat (12, 512) -> W^O -> (12, 512)
largest difference from the library output: 0.0e+00
```

Follow the shapes. Twelve tokens of 512 numbers become, for each of 8 heads, twelve queries, keys and values of 64 numbers. Each head makes a $$12 \times 12$$ table of weights. Each head outputs twelve vectors of 64, and the concat brings back 512. Our result equals the library output exactly. The two lines of the paper are the whole computation.

That rebuild used the library's own `q_proj` modules, so it did the same products as the library, in the same order. Next, we ran our own `multi_head_attention` function on the model's weights. A small helper reads the four matrices of a block and turns them into the layout of the paper:

```python
def marian_params(att):
    """Read one Marian attention block as the paper's matrices. nn.Linear stores (out, in): transpose to x @ W."""
    return dict(W_Q=att.q_proj.weight.T, b_Q=att.q_proj.bias, W_K=att.k_proj.weight.T, b_K=att.k_proj.bias,
                W_V=att.v_proj.weight.T, b_V=att.v_proj.bias, W_O=att.out_proj.weight.T, b_O=att.out_proj.bias)
```

> [!DEFINITION] Hook
> A small function that PyTorch calls each time a given module runs. We use hooks to save the input and the output of an attention block during a normal run of the model. Then we feed the same input to our own code.

With hooks on three blocks, we ran the model once on the English sentence and its German translation. Then we called our function on the input of the first encoder block:

```python
s = seen['encoder self-attention']
y, w = multi_head_attention(s['x_q'], s['x_q'], marian_params(blocks['encoder self-attention']), H)
diff = (y - s['y']).abs().max().item()                # against the output of the library's own block
```

```text
source 12 tokens, decoder input 14 tokens
encoder self-attention : x (1, 12, 512) -> weights (1, 8, 12, 12), max abs difference 1.5e-06
```

Our function matches the library to $$1.5 \times 10^{-6}$$, on outputs that reach 4.3. The products are the same, but our code orders the steps a little differently, so the last digits round differently. The same code later handles the decoder and cross-attention blocks; only the inputs and the mask change.

[![Terminal output of tx_part3.py with every result on this page](/img/papers/transformer/part3-run.png)](/img/papers/transformer/part3-run.png)

What do the eight heads look at? Here is the top key of each head for the query "it", in the first encoder layer:

```text
layer 1, query "it" (position 7): top key per head
  head 1: [('▁tired', 0.75), ('</s>', 0.08)]
  head 2: [('▁it', 0.2), ('▁tired', 0.19)]
  head 3: [('</s>', 0.6), ('▁the', 0.1)]
  head 4: [('▁was', 0.33), ('▁mat', 0.17)]
  head 5: [('</s>', 0.41), ('▁the', 0.18)]
  head 6: [('</s>', 0.19), ('▁tired', 0.18)]
  head 7: [('</s>', 0.26), ('.', 0.2)]
  head 8: [('▁was', 0.68), ('</s>', 0.09)]
```

The heads do not agree, which is the point. Head 1 puts 0.75 of its weight on "tired". Head 8 puts 0.68 on "was", the next word. Head 2 spreads its weight. Several heads park weight on the end marker `</s>`, a common habit of trained models when a head has nothing specific to find. This is one sentence and the first layer only, so do not read a fixed job into each head.

## The three uses of attention {§3.2.3}

The same multi-head block appears three times in the model. Only the source of the queries, keys and values changes.

> [!PAPER] · §3.2.3
> [![The start of Section 3.2.3 and its first bullet, with queries come from the previous decoder layer, memory keys and values come from the output of the encoder and attend over all positions in the input sequence highlighted](/img/papers/transformer/p3-uses-1.png)](/img/papers/transformer/p3-uses-1.png)
>
> **What it says:** in "encoder-decoder attention" layers, "the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder". So every position in the decoder can "attend over all positions in the input sequence".
>
> **What it means:**
>
> This is the bridge between the two languages. The decoder is writing German. At each step it asks a question about the English sentence: "which English words matter for the word I write now?". The questions are German (the queries); the labels and the content are English (the keys and values).
>
> This is the same job as Bahdanau attention in older translation models. The difference is the score: a scaled dot product in place of a small network, and eight heads in place of one. Many people call this layer **cross-attention**.

We ran the source sentence and its translation through the model and asked for every attention weight. The shapes show the three uses: the encoder table is English by English, the cross table is German by English.

```text
== 5. the three uses (full forward pass on the source and the translation)
decoder input tokens: ['<pad>', '▁Die', '▁Katze', '▁saß', '▁auf', '▁der', '▁Matt', 'e', ',', '▁weil', '▁sie', '▁müde', '▁war', '.']
encoder self-attention weights per layer: (1, 8, 12, 12)  (batch, heads, n_src, n_src)
decoder self-attention weights per layer: (1, 8, 14, 14)  (batch, heads, n_tgt, n_tgt)
cross-attention weights per layer:        (1, 8, 14, 12)  (batch, heads, n_tgt, n_src)
```

The first decoder token, `<pad>`, is the start signal that this model uses. Below is the cross-attention of decoder layer 3, as the mean of its 8 heads. For each German query, it shows the English token with the most weight:

```text
cross-attention, decoder layer 3, mean of heads: the source token each German token attends to most
  query ▁Die         -> ▁The       (0.31)
  query ▁Katze       -> ▁cat       (0.25)
  query ▁saß         -> </s>       (0.37)
  query ▁auf         -> ▁on        (0.31)
  query ▁der         -> ▁the       (0.26)
  query ▁Matt        -> ▁mat       (0.43)
  query e            -> ▁mat       (0.27)
  query ,            -> ▁because   (0.55)
  query ▁weil        -> </s>       (0.27)
  query ▁sie         -> ▁it        (0.30)
  query ▁müde        -> ▁tired     (0.33)
  query ▁war         -> </s>       (0.37)
  query .            -> .          (0.52)
```

{{FIG:p3_cross|Cross-attention in decoder layer 3 of the translation model, mean of 8 heads. Rows are the German tokens (queries), columns are the English tokens (keys). A darker cell means more weight. Most rows are darkest at the English word with the same meaning.}}

Most German words find their English partner with no one telling the model which word matches which. "sie" finds "it", which is a nice case: German uses "sie" because "Katze" is feminine. "Matt" and "e" are two pieces of "Matte", and both look at "mat". Some rows go to the end marker instead, as in the heads above. One layer and the mean of heads hide a lot, so this table is a picture of one sentence, not a rule.

### Cross-attention in code

In code, cross-attention is the same function with a different second input. The queries come from the decoder: 14 German positions. The keys and values come from the encoder output: 12 English positions. Our script caught both in decoder layer 1 of the model and called our function:

```python
s = seen['cross-attention']
y, w = multi_head_attention(s['x_q'], s['x_kv'], marian_params(blocks['cross-attention']), H)
diff = (y - s['y']).abs().max().item()
```

```text
cross-attention        : queries (1, 14, 512) from the decoder, keys and values (1, 12, 512) from the encoder
                         -> weights (1, 8, 14, 12) -> output (1, 14, 512), max abs difference 1.2e-05
largest output value of each block: encoder self-attention 4.3, decoder self-attention 8.0, cross-attention 26.2
cross-attention layer 1, query "Katze", mean of 8 heads, top keys: [('</s>', 0.64), ('▁cat', 0.15), ('.', 0.05)]
```

Look at the shapes. The weights are (1, 8, 14, 12). Each of the 8 heads has a table with a row for each German query and a column for each English key. The table is not square, because the two sentences have different lengths. The output is (1, 14, 512): one new vector for each German position, filled with information from the English sentence.

Our function matches the library to $$1.2 \times 10^{-5}$$. That is larger than before, but the outputs of this block are also larger: up to 26.2. Relative to the size of the values, it is the same rounding as before. In this first decoder layer, "Katze" puts 0.64 of its weight on the end marker and 0.15 on "cat". The clear match of layer 3, above, appears only after a few layers.

> [!PAPER] · §3.2.3
> [![The second bullet of Section 3.2.3, with all of the keys, values and queries come from the same place highlighted](/img/papers/transformer/p3-uses-2.png)](/img/papers/transformer/p3-uses-2.png)
>
> **What it says:** "The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place", the output of the previous encoder layer. "Each position in the encoder can attend to all positions in the previous layer of the encoder."
>
> **What it means:**
>
> In self-attention, the sentence talks to itself. Each English token asks a question of all the English tokens, itself included. This is how "it" can collect information from "cat" and "tired" before any German is written.
>
> "All positions" means there is no limit on distance. The first word and the last word are one step apart. A recurrent network needs one step per word to carry the same information. [Part 4](part-4-why-self-attention.md) counts this cost in detail.

> [!DEFINITION] Self-attention and cross-attention
> In **self-attention**, the queries, keys and values all come from the same sequence. In **cross-attention** (the paper's "encoder-decoder attention"), the queries come from one sequence and the keys and values from another.

## The mask: no looking ahead {§3.2.3}

> [!PAPER] · §3.2.3
> [![The third bullet of Section 3.2.3, with up to and including that position, prevent leftward information flow, auto-regressive property and masking out (setting to minus infinity) highlighted](/img/papers/transformer/p3-uses-3.png)](/img/papers/transformer/p3-uses-3.png)
>
> **What it says:** decoder self-attention lets each position attend to all positions "up to and including that position". The model must "prevent leftward information flow" to keep the "auto-regressive property". It does this by "masking out (setting to $$-\infty$$)" the inputs of the softmax that "correspond to illegal connections".
>
> **What it means:**
>
> The decoder writes one word at a time, from left to right. When it chooses word 4, words 5, 6 and 7 do not exist yet. In training, though, the whole German sentence is in the input at once, for speed. Without a rule, word 4 could simply read word 5 and copy it. That is cheating, and the model would learn nothing useful.
>
> The mask is that rule. Before the softmax, every score where a query looks at a later position is set to $$-\infty$$. The paper says "leftward" because, in their pictures, information about later words would flow to the left. This is the "Mask (opt.)" box in Figure 2.

> [!DEFINITION] Auto-regressive
> A model that makes its output one piece at a time, and feeds each piece back in as input for the next piece. A translation model of this kind writes German word 1, then reads it to write word 2, and so on.

### Why minus infinity works

The softmax uses $$e^{s}$$, and $$e^{-\infty} = 0$$. So a masked score gives a weight of exactly zero, and the other weights still add up to 1.

$$
\text{softmax}(s_1, \dots, s_n)_j = \frac{e^{s_j}}{\sum_{m} e^{s_m}}, \qquad s_j = -\infty \;\Rightarrow\; e^{s_j} = 0
$$

A tiny example: the scores of word 2 against words 1, 2 and 3 are (2, 1, 5). Word 3 is in the future, so its score becomes $$-\infty$$. The weights are $$e^2 / (e^2 + e^1) = 7.389 / 10.107 = 0.731$$, then $$0.269$$, then $$0$$. Without the mask, the large future score 5 would have taken most of the weight.

### The mask in code

The mask is a table of True and False: True where a query may look at a key. Our code builds it with two loops, so the rule is visible: query $$i$$ may look at key $$j$$ only if $$j \le i$$. The same table comes from `torch.tril`, which keeps the lower triangle of a matrix.

```python
n = 5
allowed = torch.zeros(n, n, dtype=torch.bool)
for i in range(n):                 # query position i ...
    for j in range(n):             # ... may look at key position j only if j is not in the future
        allowed[i, j] = j <= i
```

Then we make random queries, keys and values for 5 tokens and follow the scores through the mask. `masked_fill` writes minus infinity into every cell where the mask is False. Our `attention` function does exactly this, before the softmax.

```python
g = torch.Generator().manual_seed(3)
q, k, v = torch.randn(n, 4, generator=g), torch.randn(n, 4, generator=g), torch.randn(n, 4, generator=g)
scores = q @ k.T / math.sqrt(4)
masked = scores.masked_fill(~allowed, float('-inf'))
out_mask, w_mask = attention(q, k, v, allowed)
lib_causal = F.scaled_dot_product_attention(q, k, v, is_causal=True)
lib_bool = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
d_causal = (out_mask - lib_causal).abs().max().item()
d_bool = (out_mask - lib_bool).abs().max().item()
```

```text
same as torch.tril(ones): True
allowed (1 = may look, 0 = future)  shape (5, 5)
    1  0  0  0  0
    1  1  0  0  0
    1  1  1  0  0
    1  1  1  1  0
    1  1  1  1  1
scores before the mask  shape (5, 5)
    -0.212   0.955   0.141   0.252  -0.031
    -0.057   0.968   0.105   0.652  -1.325
     0.165   0.101  -0.190   0.419  -0.887
     0.589  -1.212  -0.498   0.380  -1.627
    -0.513   0.422   0.423   0.672  -0.942
scores after the mask  shape (5, 5)
    -0.212    -inf    -inf    -inf    -inf
    -0.057   0.968    -inf    -inf    -inf
     0.165   0.101  -0.190    -inf    -inf
     0.589  -1.212  -0.498   0.380    -inf
    -0.513   0.422   0.423   0.672  -0.942
weights  shape (5, 5)
     1.000   0.000   0.000   0.000   0.000
     0.264   0.736   0.000   0.000   0.000
     0.379   0.355   0.266   0.000   0.000
     0.432   0.071   0.146   0.351   0.000
     0.100   0.254   0.255   0.326   0.065
ours vs library is_causal=True: max abs difference 1.5e-07
ours vs library attn_mask=allowed: max abs difference 1.5e-07
```

{{FIG:p3_mask_steps|Masked attention on 5 tokens in three steps. Left: the scores of every query against every key. Middle: every score of a future key is replaced by minus infinity. Right: the weights after the softmax. The future weights are exactly 0, and each row adds up to 1.}}

Follow row 2. Its scores are -0.06 and 0.97 for the two allowed keys, and three future scores. The mask turns the future into minus infinity, and the softmax gives 0.264 and 0.736 to the two allowed keys. Row 1 has only one allowed key, so it gets the full weight of 1. Row 5 is the last token: it sees every key, and its weights are the same as without a mask.

The library offers two ways to say the same thing. `is_causal=True` builds the triangle for you. `attn_mask=allowed` takes your own table, where True also means "may look". Both match our output to $$1.5 \times 10^{-7}$$.

In the real model, we checked every weight above the diagonal of the decoder self-attention table:

```text
decoder self-attention, weight above the diagonal (future tokens): max 0.0e+00
decoder self-attention, last layer, mean of heads, first 5 rows x 5 columns:
  1.00 0.00 0.00 0.00 0.00   <- query <pad>
  0.59 0.41 0.00 0.00 0.00   <- query ▁Die
  0.63 0.04 0.32 0.00 0.00   <- query ▁Katze
  0.53 0.04 0.12 0.31 0.00   <- query ▁saß
  0.52 0.06 0.06 0.12 0.25   <- query ▁auf
```

{{FIG:p3_causal|Decoder self-attention in the last decoder layer of the translation model, mean of 8 heads. Each row is a German query, each column a German key. The empty cells above the diagonal are the masked future positions: their weight is exactly zero.}}

Every future weight is exactly zero. The first row has only one place to look, so it puts 1.00 on itself. Each later row spreads its weight over the past, with much of it on the start token and on itself. The table is a triangle, and that triangle is the whole idea of the decoder mask.

Our own function gives the same decoder self-attention as the library, with the mask from `torch.tril`:

```python
s = seen['decoder self-attention']
m = s['x_q'].shape[1]
y, w = multi_head_attention(s['x_q'], s['x_q'], marian_params(blocks['decoder self-attention']), H,
                            mask=torch.tril(torch.ones(m, m, dtype=torch.bool)))
diff = (y - s['y']).abs().max().item()
```

```text
decoder self-attention : x (1, 14, 512) -> weights (1, 8, 14, 14), max abs difference 3.3e-06
```

The mask also has a partner trick. The German input to the decoder is shifted right by one position: the start token comes first. So the query at "Katze" predicts the next word, "saß", and sees only words up to "Katze". [Part 2](part-2-the-architecture.md) shows where this mask and this shift sit in the decoder stack.

## Three common mistakes, in code

Attention code often fails without a sound. The shapes are right, nothing crashes, and the numbers are wrong. Here are three mistakes that we made on purpose, each with its wrong output. The tests use the real encoder block from above, or the 5-token example of the mask.

### Mistake 1: forgetting the scale

The same code as before, but the scores are not divided by $$\sqrt{d_k}$$:

```python
Qh, Kh, Vh = (split_heads(s['x_q'] @ pm[f'W_{c}'] + pm[f'b_{c}'], H) for c in 'QKV')
w_bad = torch.softmax(Qh @ Kh.transpose(-2, -1), dim=-1)          # no / math.sqrt(d_k)
y_bad = merge_heads(w_bad @ Vh) @ pm['W_O'] + pm['b_O']
```

```text
mistake 1, no scale:
  query "it", head 1: top weight 0.75 with the scale, 1.00 without
  largest weight in any row, all heads: 0.866 with the scale, 1.000 without
  output: max abs difference from the library 3.36
```

Without the division by $$\sqrt{64} = 8$$, every score is 8 times larger. Head 1 at the query "it" had 0.75 of its weight on "tired". Without the scale, it puts 1.00 on one key, and the soft choice becomes a hard one. This is the effect of footnote 4, now in a trained model. The output moves by up to 3.36, which is a different layer, not a rounding error. A new model with this bug would also train badly, because of the small gradients.

### Mistake 2: masking after the softmax

Two wrong versions. The first applies the softmax and then sets the future weights to 0. The second masks before the softmax, but with 0 in place of minus infinity:

```python
w_after = torch.softmax(scores, dim=-1) * allowed                   # softmax first, then zero the future
w_zero = torch.softmax(scores.masked_fill(~allowed, 0.0), dim=-1)   # "mask" with 0 in place of minus infinity
```

```text
mistake 2, mask after the softmax (5-token example of section 3):
  weights, mask after softmax  shape (5, 5)
     0.119   0.000   0.000   0.000   0.000
     0.137   0.383   0.000   0.000   0.000
     0.234   0.219   0.164   0.000   0.000
     0.413   0.068   0.139   0.335   0.000
     0.100   0.254   0.255   0.326   0.065
  row sums: [0.119, 0.52, 0.617, 0.955, 1.0]  (correct: all 1.0)
  variant: masked scores set to 0, not -inf; weight on the future per row: [0.832, 0.456, 0.391, 0.193, 0.0]
```

In the first version, the future weights are 0, so the mask seems to work. But the softmax already shared the weight among all 5 keys, the future ones too. Row 1 keeps only 0.119 of its weight, and the rows no longer add up to 1. The output is a shrunken average, and the amount of the shrink depends on the future scores. So the future still leaks in. Only row 5, which has no future, is right.

The second version is just as bad. A score of 0 is not "no score", because $$e^0 = 1$$. Row 1 gives 0.832 of its weight to future keys. The mask must use minus infinity, and it must come before the softmax.

### Mistake 3: transposing the wrong axes

The first wrong version splits the heads with one `view` and no `transpose`. The second splits correctly, but merges with one `reshape` and no `transpose`:

```python
xq = s['x_q']
b_, n_, dm = xq.shape
Qw, Kw, Vw = ((xq @ pm[f'W_{c}'] + pm[f'b_{c}']).view(b_, H, n_, dm // H) for c in 'QKV')   # wrong split
heads_w, _ = attention(Qw, Kw, Vw)
y_w1 = heads_w.reshape(b_, n_, dm) @ pm['W_O'] + pm['b_O']
Qr, Kr, Vr = (split_heads(xq @ pm[f'W_{c}'] + pm[f'b_{c}'], H) for c in 'QKV')            # right split
heads_r, _ = attention(Qr, Kr, Vr)
y_w2 = heads_r.reshape(b_, n_, dm) @ pm['W_O'] + pm['b_O']                                # wrong merge
```

```text
mistake 3, wrong axes:
  split with view(batch, h, n, d_k), no transpose: shapes (1, 8, 12, 64) (look right), max abs difference 3.84
  right split, but merge with reshape and no transpose: output (1, 12, 512), max abs difference 4.57
  right split and right merge: max abs difference 1.5e-06
```

This one is the hardest to see. `view(batch, h, n, d_k)` gives the right shape, (1, 8, 12, 64), so the rest of the code runs. But `view` only reads the numbers again in their order in memory. The 512 numbers of token 1 become rows 1 to 8 of "head 1", and the first half of token 2 becomes rows 9 to 12. Each "head" now mixes pieces of several tokens, and attention compares the wrong things.

The right order is to cut each token first, with `view(batch, n, h, d_k)`, and then move the heads to the front with `transpose`. The merge has the same trap in reverse. Both wrong versions change the output by about 4, while the right code matches the library to $$1.5 \times 10^{-6}$$. A comparison with a trusted library, as on this page, catches such bugs at once.

[![Terminal output of tx_part3b.py, the code walkthroughs of this page](/img/papers/transformer/part3b-run.png)](/img/papers/transformer/part3b-run.png)

[Part 7](part-7-build-it-from-scratch.md) puts these functions together with the rest of the model and builds the whole Transformer from scratch.

> [!TAKEAWAYS] Key takeaways
> - Attention turns a query and a set of key-value pairs into a weighted sum of the values. The weights come from how well each key matches the query.
> - Equation 1, $$\text{softmax}(QK^\top / \sqrt{d_k})V$$, scores all pairs with one matrix product, turns each row into weights, and mixes the values.
> - Dot-product scores are much faster than additive (Bahdanau) scores, because they are one matrix product.
> - The dot product of random vectors has variance $$d_k$$. With $$d_k = 512$$ and no scale, our softmax put 1.0000 on one key and its gradient rounded to zero.
> - Dividing by $$\sqrt{d_k}$$ brings the typical score back to size 1 and keeps the softmax trainable.
> - Multi-head attention runs $$h = 8$$ heads of 64 numbers side by side, for about the cost of one head of 512. One block of the base model has 1,050,624 parameters.
> - We rebuilt one multi-head block of a released translation model by hand, and it matched the library output exactly.
> - Our own `attention` and `multi_head_attention` functions match `scaled_dot_product_attention` and `nn.MultiheadAttention` to within $$6 \times 10^{-7}$$. On the real model's three kinds of attention block, they match to $$1.2 \times 10^{-5}$$ or better.
> - Multi-head attention is one wide product, then `view` and `transpose` to split the heads, and `transpose` and `reshape` to merge them.
> - The same block is used three ways: encoder self-attention, decoder self-attention, and encoder-decoder (cross) attention.
> - The decoder mask sets future scores to minus infinity, so their weights are exactly zero and the model cannot copy the answer.
> - Three silent bugs: no scale, a mask after the softmax, and a split without the transpose. Without the scale, head 1 put 1.00 on one key in place of 0.75.

## The paper and the code

[Paper (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) · [Official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
