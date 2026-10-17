---
title: "Attention: Queries, Keys and Values"
description: "How one attention step works, why the scores are divided by the square root of the key size, and why the model runs eight heads side by side. We rebuild the attention of a released translation model by hand and look at its real weights."
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

The mask also has a partner trick. The German input to the decoder is shifted right by one position: the start token comes first. So the query at "Katze" predicts the next word, "saß", and sees only words up to "Katze". [Part 2](part-2-the-architecture.md) shows where this mask and this shift sit in the decoder stack.

> [!TAKEAWAYS] Key takeaways
> - Attention turns a query and a set of key-value pairs into a weighted sum of the values. The weights come from how well each key matches the query.
> - Equation 1, $$\text{softmax}(QK^\top / \sqrt{d_k})V$$, scores all pairs with one matrix product, turns each row into weights, and mixes the values.
> - Dot-product scores are much faster than additive (Bahdanau) scores, because they are one matrix product.
> - The dot product of random vectors has variance $$d_k$$. With $$d_k = 512$$ and no scale, our softmax put 1.0000 on one key and its gradient rounded to zero.
> - Dividing by $$\sqrt{d_k}$$ brings the typical score back to size 1 and keeps the softmax trainable.
> - Multi-head attention runs $$h = 8$$ heads of 64 numbers side by side, for about the cost of one head of 512. One block of the base model has 1,050,624 parameters.
> - We rebuilt one multi-head block of a released translation model by hand, and it matched the library output exactly.
> - The same block is used three ways: encoder self-attention, decoder self-attention, and encoder-decoder (cross) attention.
> - The decoder mask sets future scores to minus infinity, so their weights are exactly zero and the model cannot copy the answer.

## The paper and the code

[Paper (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) · [Official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
