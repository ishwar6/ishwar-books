---
title: "Paths Through the Model: the Path Expansion Trick"
description: "How the paper turns a one-layer model into a sum of end-to-end paths, one for the token itself and one for each head. We expand the real model, freeze its attention, and check every term."
part: 3
covers: "Conceptual take-aways, One-Layer Attention-Only Transformers, The Path Expansion Trick, OV and QK Independence, Path Expansion of Logits"
date: 2026-10-17
tags: [interpretability, transformers, path-expansion, circuits, attention]
---

Parts 1 and 2 took the model apart piece by piece: the residual stream, then one attention head. This part puts the pieces back together in a new shape: a sum of paths from the input tokens to the output. Every path can be read on its own.

We use the same one-layer model, `attn-only-1l`, and the same prompt as Part 2: *"I like green tea, so I drink green"*. The model's next word is " tea", with probability 0.897. By the end of this part we will know exactly how much each path adds to that guess.

## The promise: a sum of simple functions {§Summary of Results}

The paper states its main method early, in a short list of "conceptual take-aways". Two items of that list are the subject of this part.

> [!PAPER] · Summary of Results
> [![Two take-aways: attention-only models can be written as a sum of interpretable end-to-end functions, linear if one freezes the attention patterns; transformers have an enormous amount of linear structure](/img/papers/circuits/p3-takeaways.png)](/img/papers/circuits/p3-takeaways.png)
>
> **What it says:** "Attention-only models can be written as a sum of interpretable end-to-end functions mapping tokens to changes in logits." These functions "correspond to 'paths' through the model, and are linear if one freezes the attention patterns." Also: "Transformers have an enormous amount of linear structure", and one can learn a lot "by breaking apart sums and multiplying together chains of matrices".
>
> **What it means:**
>
> Think of a big river delta. Water from the mountain reaches the sea along many channels. You cannot see much by looking at the whole delta at once. But you can follow one channel from the source to the sea, measure its flow, and then add up the channels.
>
> The paper says that a small transformer is like that delta. The tokens are the source, and the logits are the sea. Each "path" is one channel through the layers. Each path is an "end-to-end function": it starts at the tokens and ends at the logits, so a person can read it in tokens.
>
> The second sentence names the tool. Most of a transformer is matrix multiplication and addition. Only the attention pattern is not. So if we hold the attention pattern fixed, everything left is linear algebra, and linear algebra can be taken apart. This part shows how.

The two words "freezes" and "linear" carry the whole idea. We meet both again below, and we test them on the real model.

## One layer: a bigram model plus skip-trigrams {§One-Layer Attention-Only Transformers}

The section on one-layer models opens with a claim, and then says what it will do with it.

> [!PAPER] · One-Layer Attention-Only Transformers
> [![The opening of the one-layer section: one-layer attention-only transformers are an ensemble of a bigram model and several skip-trigram models; the goal is to convert raw weights into interpretable tables](/img/papers/circuits/p3-claim.png)](/img/papers/circuits/p3-claim.png)
>
> **What it says:** a one-layer attention-only transformer is "an ensemble of a bigram model and several 'skip-trigram' models". They change the probabilities of sequences "A… BC". Each head can "selectively attend from the present token ('B') to a previous token ('A')". It copies information that changes the probability of the next token ("C"). The goal is to "convert the raw weights of a transformer into interpretable tables".
>
> **What it means:**
>
> Think of a panel of judges who each give a score, and the scores are added. One judge only looks at the current word. That is the bigram model from Part 1: "after Barack, say Obama". Each other judge is a head. A head looks back at one earlier word and then votes for some next words.
>
> The phrase "A… BC" is the pattern a head can learn. "A" is an earlier token, "B" is the current token, and "C" is the next token. The dots mean that any number of tokens can sit between A and B. In our prompt, A is " tea", B is the last " green", and C is " tea" again.
>
> The last sentence is the real promise. The paper does not only want to describe the model. It wants tables that you can read straight from the weights, without running any text. This part builds the sum that holds those tables, and Part 4 reads them.

> [!DEFINITION] Skip-trigram
> A pattern of three tokens, written "A… BC". If token A appeared somewhere earlier and the current token is B, change the chance that the next token is C. A **trigram** is three tokens in a row; "skip" means that A does not have to be next to B. The paper borrows the name from "skip-gram", a classic word-embedding method.

> [!DEFINITION] Ensemble
> A group of models whose outputs are combined into one answer, here by adding their logits. Each member of the ensemble can be studied alone, and the final answer is their sum.

"Ensemble" is a strong word. It says that the one-layer model is nothing more than its members, added together. To check this, we need to write the model in a form where the sum is visible. That form is the **path expansion**.

## The model as a product {§The Path Expansion Trick}

Before the trick, the paper writes the one-layer model again, in three lines.

> [!PAPER] · The Path Expansion Trick
> [![The one-layer model as three equations: the token embedding x0 = W_E t, each attention head is run and added to the residual stream, x1 = x0 + sum of h(x0), and the logits T(t) = W_U x1; then the remark that the transformer can be written as a product of three terms](/img/papers/circuits/p3-recall.png)](/img/papers/circuits/p3-recall.png)
>
> **What it says:** a one-layer attention-only transformer is "a token embedding, followed by an attention layer" and "finally an unembedding". In equations: $$x_0=W_{\mathrm{E}}\,t$$, then $$x_1=x_0+\sum_{h\in H}h(x_0)$$, then $$T(t)=W_{\mathrm{U}}\,x_1$$. With the tensor notation of Part 2, the model is "a product of three terms".
>
> **What it means:**
>
> Read the figure from the bottom up, like the model runs. The tokens become vectors. Every head reads those vectors and adds its output. The unembedding turns the result into logits. These are the equations of Part 1, with only one layer.
>
> The plus sign in the middle is important. The heads do not replace the stream; they add to it. The original $$x_0$$ is still there in $$x_1$$. This is the residual stream at work, and it is the reason the expansion below gives more than one path.
>
> "A product of three terms" means one factor for each stage: embed, attend, unembed. The next figure of the paper writes the three factors out.

Here $$t$$ is the whole prompt as one table of one-hot vectors, one column per position. So $$x_0$$ and $$x_1$$ are tables too, with one stream vector per position. Each head works on the whole table, because it reads across positions.

> [!PAPER] · The Path Expansion Trick
> [![The product of three terms: T = (Id ⊗ W_U) · (Id + sum over heads of A^h ⊗ W_OV^h) · (Id ⊗ W_E), where A^h is the masked softmax of t^T W_E^T W_QK^h W_E t](/img/papers/circuits/p3-product.png)](/img/papers/circuits/p3-product.png)
>
> **What it says:** $$T=(\text{Id}\otimes W_{\mathrm{U}})\cdot\big(\text{Id}+\sum_{h\in H_1}A^h\otimes W_{\mathrm{OV}}^h\big)\cdot(\text{Id}\otimes W_{\mathrm{E}})$$. The unembedding "maps residual stream vectors to logits". The attention layer adds the result of each head. The embedding "maps tokens to residual stream vectors". The pattern is $$A^h=\text{softmax}^*(t^\top W_{\mathrm{E}}^\top W_{\mathrm{QK}}^h W_{\mathrm{E}}\,t)$$.
>
> **What it means:**
>
> Each factor is one stage of the figure above, written as a map from tables to tables. The right factor embeds every token. The middle factor runs the attention layer. The left factor unembeds every position. A product of maps means "do the right one first, then the next".
>
> The notation comes from Part 2. $$\text{Id}\otimes W$$ means "apply $$W$$ to each position, and do not mix positions". $$A^h\otimes W_{\mathrm{OV}}^h$$ is one whole head: the pattern mixes positions, and $$W_{\mathrm{OV}}^h$$ changes each vector. The lone $$\text{Id}$$ in the middle is the plus sign of the residual stream: "keep what was there".
>
> The second line says where the pattern comes from. It is the masked softmax of scores, and each score multiplies two tokens through $$W_{\mathrm{E}}^\top W_{\mathrm{QK}}^h W_{\mathrm{E}}$$. That is the QK circuit of Part 2. So the pattern depends on the tokens too, which is the only non-linear part of the model.

> [!DEFINITION] Identity map (Id)
> The map that changes nothing: $$\text{Id}\,x=x$$. On the left side of $$\otimes$$ it means "do not mix positions"; on the right side it means "do not change the vectors". In a sum like $$\text{Id}+\sum_h(\dots)$$, it stands for the original stream that passes through the layer unchanged.

### Check the product on the real model

Is this product really the model? We computed it by hand, stage by stage, at all ten positions of our prompt:

```python
x0  = E + P                                                     # (Id ⊗ W_E) t, plus the position embedding
mid = x0 + sum(A[h] @ (x0 / s1[:, None]) @ OV[h] for h in range(8)) + bias_O   # (Id + Σ_h A^h ⊗ W_OV^h) x0
prod = (mid / sf[:, None]) @ W_U + b_U                          # (Id ⊗ W_U) x1
```

```text
== 1. THE ONE-LAYER MODEL AS A PRODUCT OF THREE TERMS (attn-only-1l)
  prompt: ['<|BOS|>', 'I', ' like', ' green', ' tea', ',', ' so', ' I', ' drink', ' green']  (10 tokens); the model predicts ' tea' next
  embed -> (Id + sum of 8 heads) -> unembed, by hand at all 10 positions: |by hand - model logits| max = 2.6e-05
  layer-norm scales at the last position: before attention 0.238, before the unembedding 0.493
```

The largest difference over all 48,262 logits at all ten positions is 0.000026, which is rounding noise. Three things in the code are not in the paper's formula. The real model adds a **position embedding** `P` to each token's vector. It has **biases** (`bias_O`, `b_U`). And it has the leftover **layer norm** from Part 1, which divides each vector by a scale (`s1`, `sf`).

We read the layer-norm scales from the real run and then hold them fixed. With fixed scales, a division is just a multiplication by a number, so it fits inside the product. The scales at the last position are 0.238 and 0.493. The paper's models have no layer norm, so it does not need this step.

## Expand the product {§The Path Expansion Trick}

Now comes the trick. It is the oldest move in algebra: multiply out the brackets.

Take a product of two sums of numbers, $$(1+2)\times(1+3)$$. You can add first and then multiply: $$3\times4=12$$. Or you can multiply every part of the first bracket by every part of the second and add the results: $$1+2+3+2\times3=12$$. The answer is the same. The second way keeps a record of **where each part of the answer came from**.

> [!DEFINITION] Distributive law (expanding a product)
> For numbers and for matrices, $$a\,(b+c)=ab+ac$$ and $$(b+c)\,a=ba+ca$$. Applied again and again, a product of sums becomes a sum of products: one product for each way to pick one part from every bracket.

> [!PAPER] · The Path Expansion Trick
> [![The key trick: expand the product into a sum where every term is an end-to-end path, T = Id ⊗ W_U W_E plus the sum over heads of A^h ⊗ (W_U W_OV^h W_E), with the direct path term and the attention head terms labelled](/img/papers/circuits/p3-expand.png)](/img/papers/circuits/p3-expand.png)
>
> **What it says:** "Our key trick is to simply expand the product." This turns a product, "where every term corresponds to a layer", into "a sum where every term corresponds to an end-to-end path": $$T=\text{Id}\otimes W_{\mathrm{U}} W_{\mathrm{E}}+\sum_{h\in H}A^h\otimes(W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}})$$. Each term is "tractable to understand, can be reasoned about independently, and additively combine[s] to create model behavior".
>
> **What it means:**
>
> Before the trick, the model is a pipeline: every token passes through every stage. After the trick, it is a set of separate routes. The direct path goes from the token straight to the logits. Each head term goes from the tokens, through one head, to the logits.
>
> The key change is in what each piece means. A layer is hard to read, because its input and output are 512-number vectors. A path starts at tokens and ends at logits, so it is a table from tokens to tokens. A person can read that.
>
> The last claim has three parts, and this part tests each one. "Tractable": each term is a product of known matrices. "Independently": we can compute one term without the others. "Additively": the terms add up to the model, as the check below shows.

### The derivation

The middle factor is a sum of nine parts: $$\text{Id}$$ and one term per head. The outer factors have one part each. So the expansion has $$1\times9\times1=9$$ products. Take the first one, with $$\text{Id}$$ from the middle:

$$
\begin{aligned}
&(\text{Id}\otimes W_{\mathrm{U}})\cdot\text{Id}\cdot(\text{Id}\otimes W_{\mathrm{E}})\\
&=(\text{Id}\cdot\text{Id})\otimes(W_{\mathrm{U}} W_{\mathrm{E}})\\
&=\text{Id}\otimes W_{\mathrm{U}} W_{\mathrm{E}}
\end{aligned}
$$

The middle $$\text{Id}$$ changes nothing, so it disappears. The second line is the **mixed-product property** from Part 2: $$(A\otimes B)(C\otimes D)=(AC)\otimes(BD)$$. The left sides of the two factors multiply, and the right sides multiply. Now take one head $$h$$ from the middle:

$$
\begin{aligned}
&(\text{Id}\otimes W_{\mathrm{U}})(A^h\otimes W_{\mathrm{OV}}^h)(\text{Id}\otimes W_{\mathrm{E}})\\
&=(\text{Id}\,A^h\,\text{Id})\otimes(W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}})\\
&=A^h\otimes W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}}
\end{aligned}
$$

The same rule puts everything that acts on positions on the left of $$\otimes$$ and everything that acts on vectors on the right. On the left, only the pattern $$A^h$$ is left. On the right, the chain of matrices runs from the token to the logits. Add the nine products, and you get the paper's sum.

Read each symbol once more. $$\text{Id}\otimes W_{\mathrm{U}} W_{\mathrm{E}}$$ is the **direct path**: no mixing of positions, and the table $$W_{\mathrm{U}} W_{\mathrm{E}}$$ from Part 1. $$A^h\otimes W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}}$$ is the **head term** of head $$h$$. The pattern $$A^h$$ says which positions are read. The table $$W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}}$$ says what reading each token does to the logits. That table is the OV circuit of Part 2.

> [!DEFINITION] Path term
> One term of the expanded model. It follows one route from the input tokens to the logits: the direct route, or a route through one head (later, through two heads). Each path term is a function from tokens to logit changes, and the logits are the sum of all path terms.

### A toy you can check by hand

Our toy model has three tokens, a, b and c, a stream of 2 numbers and one head. The embedding puts a at $$(1,0)$$, b at $$(0,1)$$ and c at $$(1,1)$$. The unembedding reads the logit of a from number 1 and the logit of b from number 2, and c always gets 0. The head swaps the two numbers. On the text "a b", position 2 reads both tokens, half each.

<figure class="fig"><svg viewBox="0 0 760 336" role="img" aria-label="A toy one-layer model with tokens a, b, c on the text "a b", at position 2. Left, the product: the embedding gives (0, 1), the head adds (0.5, 0.5) so the stream is (0.5, 1.5), and the unembedding gives logits (0.5, 1.5, 0). Right, the sum of paths: the direct path gives (0, 1, 0), the head term gives (0.5, 0.5, 0), and their sum is (0.5, 1.5, 0), the same."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">A toy model on the text "a b": the product and the sum of paths give the same logits</text><text class="t-note" x="24.0" y="62.0" text-anchor="start">The product: layer after layer (position 2)</text><rect class="box" x="24.0" y="80.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="99.0" y="99.5" text-anchor="middle">x0 = W<tspan baseline-shift="sub" style="font-size:72%">E</tspan> t</text><rect class="s1" x="190.0" y="81.0" width="42" height="28" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="211.0" y="100.0" text-anchor="middle">0</text><rect class="s1" x="236.0" y="81.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="257.0" y="100.0" text-anchor="middle">1</text><line class="edge" x1="99.0" y1="114.0" x2="99.0" y2="142.0" marker-end="url(#ah)"/><rect class="box" x="24.0" y="146.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="99.0" y="165.5" text-anchor="middle">x1 = x0 + head</text><rect class="s1" x="190.0" y="147.0" width="42" height="28" rx="4" style="fill-opacity:0.28"/><text class="t-val" x="211.0" y="166.0" text-anchor="middle">0.5</text><rect class="s1" x="236.0" y="147.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="257.0" y="166.0" text-anchor="middle">1.5</text><line class="edge" x1="99.0" y1="180.0" x2="99.0" y2="208.0" marker-end="url(#ah)"/><rect class="box-3" x="24.0" y="212.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="99.0" y="231.5" text-anchor="middle">logits = W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> x1</text><rect class="s3" x="190.0" y="213.0" width="42" height="28" rx="4" style="fill-opacity:0.28"/><text class="t-val" x="211.0" y="232.0" text-anchor="middle">0.5</text><rect class="s3" x="236.0" y="213.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="257.0" y="232.0" text-anchor="middle">1.5</text><rect class="s3" x="282.0" y="213.0" width="42" height="28" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="303.0" y="232.0" text-anchor="middle">0</text><text class="t-muted" x="24.0" y="300.0" text-anchor="start">the head at position 2 reads a and b, half each,</text><text class="t-muted" x="24.0" y="318.0" text-anchor="start">and swaps their numbers: (0.5, 0.5)</text><text class="t-note" x="400.0" y="62.0" text-anchor="start">The sum: one term per end-to-end path</text><rect class="box-1" x="400.0" y="80.0" width="170.0" height="30.0" rx="6"/><text class="t-tick" x="485.0" y="99.5" text-anchor="middle">direct: W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> W<tspan baseline-shift="sub" style="font-size:72%">E</tspan> t</text><rect class="s1" x="586.0" y="81.0" width="42" height="28" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="607.0" y="100.0" text-anchor="middle">0</text><rect class="s1" x="632.0" y="81.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="653.0" y="100.0" text-anchor="middle">1</text><rect class="s1" x="678.0" y="81.0" width="42" height="28" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="699.0" y="100.0" text-anchor="middle">0</text><text class="t-muted" x="400.0" y="132.0" text-anchor="start">the current token b, straight to the logits</text><rect class="box-2" x="400.0" y="150.0" width="170.0" height="30.0" rx="6"/><text class="t-tick" x="485.0" y="169.5" text-anchor="middle">head: A ⊗ W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> W<tspan baseline-shift="sub" style="font-size:72%">OV</tspan> W<tspan baseline-shift="sub" style="font-size:72%">E</tspan></text><rect class="s2" x="586.0" y="151.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="607.0" y="170.0" text-anchor="middle">0.5</text><rect class="s2" x="632.0" y="151.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="653.0" y="170.0" text-anchor="middle">0.5</text><rect class="s2" x="678.0" y="151.0" width="42" height="28" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="699.0" y="170.0" text-anchor="middle">0</text><text class="t-muted" x="400.0" y="202.0" text-anchor="start">0.5 × column a + 0.5 × column b of the OV table</text><line class="axis" x1="400.0" y1="222.0" x2="730.0" y2="222.0"/><rect class="box-3" x="400.0" y="236.0" width="170.0" height="30.0" rx="6"/><text class="t-tick" x="485.0" y="255.5" text-anchor="middle">sum</text><rect class="s3" x="586.0" y="237.0" width="42" height="28" rx="4" style="fill-opacity:0.28"/><text class="t-val" x="607.0" y="256.0" text-anchor="middle">0.5</text><rect class="s3" x="632.0" y="237.0" width="42" height="28" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="653.0" y="256.0" text-anchor="middle">1.5</text><rect class="s3" x="678.0" y="237.0" width="42" height="28" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="699.0" y="256.0" text-anchor="middle">0</text><text class="t-muted" x="400.0" y="290.0" text-anchor="start">the same logits as the product</text><text class="t-tick" x="400.0" y="318.0" text-anchor="start">columns of each vector: logit of a, of b, of c</text></svg><figcaption>The toy model on the text "a b", at position 2. Left, the product, layer after layer: the embedding gives (0, 1), the head adds (0.5, 0.5), so the stream is (0.5, 1.5), and the unembedding gives the logits (0.5, 1.5, 0). Right, the sum of paths: the direct path gives (0, 1, 0) and the head term gives (0.5, 0.5, 0). Their sum is the same (0.5, 1.5, 0).</figcaption></figure>

Work the head term out by hand. The OV table $$W_{\mathrm{U}} W_{\mathrm{OV}} W_{\mathrm{E}}$$ has one column per source token. Reading a gives $$(0,1,0)$$, because the head swaps $$(1,0)$$ to $$(0,1)$$, which is the logit of b. Reading b gives $$(1,0,0)$$. Position 2 takes half of each column: $$0.5\times(0,1,0)+0.5\times(1,0,0)=(0.5,0.5,0)$$. The script prints the same numbers:

```text
  W_U W_OV W_E (OV circuit, column = source token)     = [[0.0, 1.0, 1.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]
  direct path term = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];  head term = [[0.0, 1.0, 0.0], [0.5, 0.5, 0.0]];  sum = [[1.0, 1.0, 0.0], [0.5, 1.5, 0.0]]
  as one 6 x 6 matrix (Id (x) W_U W_E + A (x) W_U W_OV W_E) times the flattened tokens [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]: [1.0, 1.0, 0.0, 0.5, 1.5, 0.0]
```

The last line writes the whole toy model, with $$A$$ fixed, as one 6 × 6 matrix. It acts on the two one-hot tokens laid end to end. With $$A$$ fixed, the model really is one matrix times the tokens.

### The paths of the real model

Now the real model. At the last position, the one that predicts " tea", we computed every path term over all 48,262 logits. Each head term is computed in tokens, as the formula says: the pattern of the head times the token columns of its OV table.

```python
terms['direct path (token)'] = (E[i] / sf[i]) @ W_U                                     # Id ⊗ W_U W_E
for h in range(8):
    terms[f'head 0.{h} (tokens)'] = (A[h, i] @ (E / s1[:, None]) @ OV[h] / sf[i]) @ W_U   # A^h ⊗ W_U W_OV^h W_E
```

```text
== 2. EXPAND THE PRODUCT: A SUM OF END-TO-END PATHS (logits at the last position, " green")
  12 terms: the direct path, 8 head terms (the paper's 9 terms), and 3 parts the paper leaves out
  |sum of the terms - model logits| max over all 48,262 logits = 2.1e-05
  term                        adds to logit of tea   tokens it pushes up most
  direct path (token)                       +5.501   ['ery', 'houses', 'belt', 'wich']
  head 0.0 (tokens)                         +1.321   [' moderation', 'avour', 'mented', ' replen']
  head 0.1 (tokens)                         -1.466   ['ated', 'ided', 'rices', 'istant']
  head 0.2 (tokens)                         +1.545   [' moderation', ' wouldn', ' gotta', '))))']
  head 0.3 (tokens)                         +5.465   [' tea', ' Tea', ' teac', ' brew']
  head 0.4 (tokens)                         +0.012   ['\n\n\n       ', '\n\n\n\n', '\n    ', '\n\n        ']
  head 0.5 (tokens)                         +1.146   ['MENT', 'MENTS', 'vent', ' WARRANTIES']
  head 0.6 (tokens)                         +0.317   [')],', '”).', ']),', '”)']
  head 0.7 (tokens)                         +3.353   [' vinegar', ' snacks', ' cigarettes', ' candles']
  direct path (position)                    +0.653   [' yang', '”),', '?”', '”).']
  heads (positions)                         -1.215   ['aliana', 'gmail', 'visors', 'vironments']
  biases b_O and b_U                        +3.495   [',', '.', ' and', ' in']
  sum                                      +20.128   (model logit of " tea": 20.128)
```

<figure class="fig"><svg viewBox="0 0 760 412" role="img" aria-label="Diverging bars: how much each path term of attn-only-1l adds to the logit of tea at the last position: direct path +5.501, head 0.0 +1.321, head 0.1 -1.466, head 0.2 +1.545, head 0.3 +5.465, head 0.4 +0.012, head 0.5 +1.146, head 0.6 +0.317, head 0.7 +3.353, position, direct +0.653, position, via heads -1.215, biases +3.495. They add up to 20.128, the model logit."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">attn-only-1l at the last "green": how much each path adds to the logit of "tea"</text><text class="t-muted" x="24.0" y="48.0" text-anchor="start">the first 9 rows are the paper's terms; the last 3 are parts the paper leaves out</text><line class="axis" x1="444.0" y1="62.0" x2="444.0" y2="378.0"/><text class="t-tick t-strong" x="204.0" y="83.0" text-anchor="end">direct path</text><g class="mark"><title>direct path: +5.501</title><rect class="s1" x="444.0" y="70.0" width="210.9" height="17.0" rx="3"/></g><text class="t-val" x="660.9" y="83.0" text-anchor="start">+5.501</text><text class="t-tick" x="204.0" y="109.0" text-anchor="end">head 0.0</text><g class="mark"><title>head 0.0: +1.321</title><rect class="s1" x="444.0" y="96.0" width="50.6" height="17.0" rx="3"/></g><text class="t-val" x="500.6" y="109.0" text-anchor="start">+1.321</text><text class="t-tick" x="204.0" y="135.0" text-anchor="end">head 0.1</text><g class="mark"><title>head 0.1: -1.466</title><rect class="s2" x="387.8" y="122.0" width="56.2" height="17.0" rx="3"/></g><text class="t-val" x="381.8" y="135.0" text-anchor="end">-1.466</text><text class="t-tick" x="204.0" y="161.0" text-anchor="end">head 0.2</text><g class="mark"><title>head 0.2: +1.545</title><rect class="s1" x="444.0" y="148.0" width="59.2" height="17.0" rx="3"/></g><text class="t-val" x="509.2" y="161.0" text-anchor="start">+1.545</text><text class="t-tick t-strong" x="204.0" y="187.0" text-anchor="end">head 0.3</text><g class="mark"><title>head 0.3: +5.465</title><rect class="s1" x="444.0" y="174.0" width="209.5" height="17.0" rx="3"/></g><text class="t-val" x="659.5" y="187.0" text-anchor="start">+5.465</text><text class="t-tick" x="204.0" y="213.0" text-anchor="end">head 0.4</text><g class="mark"><title>head 0.4: +0.012</title><rect class="s1" x="444.0" y="200.0" width="2.0" height="17.0" rx="3"/></g><text class="t-val" x="452.0" y="213.0" text-anchor="start">+0.012</text><text class="t-tick" x="204.0" y="239.0" text-anchor="end">head 0.5</text><g class="mark"><title>head 0.5: +1.146</title><rect class="s1" x="444.0" y="226.0" width="43.9" height="17.0" rx="3"/></g><text class="t-val" x="493.9" y="239.0" text-anchor="start">+1.146</text><text class="t-tick" x="204.0" y="265.0" text-anchor="end">head 0.6</text><g class="mark"><title>head 0.6: +0.317</title><rect class="s1" x="444.0" y="252.0" width="12.2" height="17.0" rx="3"/></g><text class="t-val" x="462.2" y="265.0" text-anchor="start">+0.317</text><text class="t-tick" x="204.0" y="291.0" text-anchor="end">head 0.7</text><g class="mark"><title>head 0.7: +3.353</title><rect class="s1" x="444.0" y="278.0" width="128.5" height="17.0" rx="3"/></g><text class="t-val" x="578.5" y="291.0" text-anchor="start">+3.353</text><text class="t-tick" x="204.0" y="317.0" text-anchor="end">position, direct</text><g class="mark"><title>position, direct: +0.653</title><rect class="s1" x="444.0" y="304.0" width="25.0" height="17.0" rx="3"/></g><text class="t-val" x="475.0" y="317.0" text-anchor="start">+0.653</text><text class="t-tick" x="204.0" y="343.0" text-anchor="end">position, via heads</text><g class="mark"><title>position, via heads: -1.215</title><rect class="s2" x="397.4" y="330.0" width="46.6" height="17.0" rx="3"/></g><text class="t-val" x="391.4" y="343.0" text-anchor="end">-1.215</text><text class="t-tick" x="204.0" y="369.0" text-anchor="end">biases</text><g class="mark"><title>biases: +3.495</title><rect class="s1" x="444.0" y="356.0" width="134.0" height="17.0" rx="3"/></g><text class="t-val" x="584.0" y="369.0" text-anchor="start">+3.495</text><text class="t-note" x="24.0" y="396.0" text-anchor="start">sum of all 12 rows = the model's logit of "tea", 20.128</text></svg><figcaption>How much each path of attn-only-1l adds to the logit of "tea" at the last "green". The direct path adds +5.501 and head 0.3 adds +5.465; head 0.7 adds +3.353 and the biases +3.495; head 0.1 takes away 1.466. The first nine rows are the paper's terms, and all twelve add up to 20.128, the model's real logit.</figcaption></figure>

The twelve terms add up to the model's logits. The largest error over all 48,262 logits is 0.000021. So the expansion is not an approximation. It is the model, written another way.

Now the table can be read. Two paths carry most of the vote for " tea". The **direct path** adds +5.501, and **head 0.3** adds +5.465. Head 0.7 adds +3.353, and head 0.1 votes against with −1.466. The other heads add a little.

The last three rows are parts the paper leaves out: the position embedding, through the direct path and through the heads, and the biases. They do not depend on which tokens are in the prompt, only on where they are. That is why the paper can drop them and still describe what the model does with tokens.

### The direct path {§The Path Expansion Trick}

> [!PAPER] · The Path Expansion Trick
> [![The direct path term Id ⊗ W_U W_E also occurred in the zero-layer transformer; it doesn't move information between positions, so it can only contribute to bigram statistics and fills gaps other terms don't handle](/img/papers/circuits/p3-direct.png)](/img/papers/circuits/p3-direct.png)
>
> **What it says:** the direct path term, $$\text{Id}\otimes W_{\mathrm{U}} W_{\mathrm{E}}$$, "also occurred when we looked at the zero-layer transformer". "Because it doesn't move information between positions", "the only thing it can contribute to is the bigram statistics". It will "fill in missing gaps that other terms don't handle".
>
> **What it means:**
>
> The $$\text{Id}$$ on the left of $$\otimes$$ says it all. The direct path at position $$i$$ only sees the token at position $$i$$. It is the zero-layer model of Part 1, still sitting inside the one-layer model.
>
> A path that sees one token can only store "after this token, that token is likely". So it can only hold bigrams. And because the heads also predict some bigrams, the direct path does not hold the full bigram table. It holds what is left: the bigrams that the heads do not already cover.
>
> This is how the terms share work. No term is told what to do. Training shapes all of them together, and each one ends up with the part of the job that suits its shape.

Look at the top tokens of the direct path at the last " green": "ery", "houses", "belt" and "wich". These pieces complete "greenery", "greenhouses", "greenbelt" and "Greenwich". That is bigram knowledge in its purest form: it only needs the current token.

But the direct path also adds +5.501 to " tea". "Green tea" is a common phrase, so " tea" after " green" is a good bigram too. Without the head, this is the only reason the model would think of tea. The head adds a second, separate reason: "tea" appeared earlier in this text.

### The head terms {§Splitting Attention Head terms into Query-Key and Output-Value Circuits}

The section after the trick starts from one head term and asks what it means.

> [!PAPER] · Splitting Attention Head terms into Query-Key and Output-Value Circuits
> [![For each attention head h we have a term A^h ⊗ (W_U W_OV^h W_E) where A^h = softmax(t^T W_E^T W_QK^h W_E t); how can we map these terms to model behavior, and why these particular products of matrices](/img/papers/circuits/p3-head-term.png)](/img/papers/circuits/p3-head-term.png)
>
> **What it says:** "For each attention head $$h$$ we have a term $$A^h\otimes(W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}})$$ where $$A^h=\text{softmax}(t^\top\cdot W_{\mathrm{E}}^\top W_{\mathrm{QK}}^h W_{\mathrm{E}}\cdot t)$$." Then two questions: "How can we map these terms to model behavior?" and "why do we get these particular products of matrices on our equations?"
>
> **What it means:**
>
> The head term has two halves, and each half comes from a chain of matrices. Both chains begin with $$W_{\mathrm{E}}$$, because both begin at tokens. The OV chain ends with $$W_{\mathrm{U}}$$, because it ends at logits. The QK chain ends with a token on each side, because a score compares two tokens.
>
> That is the answer to "why these products". Each product is a path traced through the model, from where the information enters to where it is used. Part 2 named the two tables: the QK circuit and the OV circuit.
>
> The first question, about behaviour, is what Part 4 answers with skip-trigrams. Here we do one smaller step: we take one real head term apart into its tokens.

What does $$A^h\otimes W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}}$$ do at one position $$i$$? Apply it to the prompt $$t$$, and read row $$i$$. The result is a sum over the source positions $$j$$:

$$
\big[(A^h\otimes W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}})\,t\big]_i=\sum_{j\le i}A^h_{ij}\,\big(W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}}\big)\,t_j
$$

Read it left to right. The head term at position $$i$$ is a weighted sum. Each source position $$j$$ gives one column of the OV table: the column of its own token $$t_j$$. The weight of that column is $$A^h_{ij}$$, how much position $$i$$ reads position $$j$$.

In the toy, position 2 had weights 0.5 and 0.5 on a and b. So its head term was half of column a plus half of column b. On the real model, here is head 0.3 at the last " green", for the logit of " tea":

```text
== 3. ONE HEAD TERM IN TOKENS: A^h (x) W_U W_OV^h W_E (head 0.3, destination = last " green")
  each source token j adds  A[last, j] x (column t_j of the OV circuit)  to the logits; its effect on " tea":
    j=0 '<|BOS|>'  weight 0.366 x column value  -0.456 = -0.167
    j=1 'I'        weight 0.003 x column value  +3.673 = +0.011
    j=2 ' like'    weight 0.001 x column value  -6.281 = -0.008
    j=3 ' green'   weight 0.013 x column value  +6.110 = +0.077
    j=4 ' tea'     weight 0.370 x column value +12.150 = +4.497
    j=5 ','        weight 0.087 x column value  +0.040 = +0.003
    j=6 ' so'      weight 0.001 x column value  +2.496 = +0.001
    j=7 ' I'       weight 0.002 x column value  +1.022 = +0.002
    j=8 ' drink'   weight 0.147 x column value  +6.649 = +0.979
    j=9 ' green'   weight 0.011 x column value  +6.281 = +0.070
  sum = +5.465  (term "head 0.3 (tokens)" above: +5.465)
```

<figure class="fig"><svg viewBox="0 0 760 364" role="img" aria-label="A table for head 0.3 at the last position. For each of the 10 source tokens: the attention weight, the value of its OV column in the row for tea, and their product. Tea has weight 0.370 and column value +12.15, so it adds +4.497. Drink adds +0.979, BOS -0.167, the rest almost nothing. The total is +5.465."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">Head 0.3 at the last "green": weight × OV column value = what each source adds to "tea"</text><text class="t-note" x="24.0" y="56.0" text-anchor="start">source token</text><text class="t-note" x="170.0" y="56.0" text-anchor="start">weight A</text><text class="t-note" x="290.0" y="56.0" text-anchor="start">OV column, tea row</text><text class="t-note" x="450.0" y="56.0" text-anchor="start">adds to "tea"</text><text class="t-tick" x="24.0" y="85.0" text-anchor="start">BOS  (j=0)</text><rect class="cell" x="170" y="72" width="36.6" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="212.6" y="85.0" text-anchor="start">0.366</text><text class="t-val" x="290.0" y="85.0" text-anchor="start">-0.46</text><rect class="s2" x="450" y="72" width="8.5" height="17" rx="3"/><text class="t-val" x="464.5" y="85.0" text-anchor="start">-0.167</text><text class="t-tick" x="24.0" y="111.0" text-anchor="start">I  (j=1)</text><rect class="cell" x="170" y="98" width="2.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="178.0" y="111.0" text-anchor="start">0.003</text><text class="t-val" x="290.0" y="111.0" text-anchor="start">+3.67</text><rect class="s1" x="450" y="98" width="2.0" height="17" rx="3"/><text class="t-val" x="458.0" y="111.0" text-anchor="start">+0.011</text><text class="t-tick" x="24.0" y="137.0" text-anchor="start">like  (j=2)</text><rect class="cell" x="170" y="124" width="2.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="178.0" y="137.0" text-anchor="start">0.001</text><text class="t-val" x="290.0" y="137.0" text-anchor="start">-6.28</text><rect class="s2" x="450" y="124" width="2.0" height="17" rx="3"/><text class="t-val" x="458.0" y="137.0" text-anchor="start">-0.008</text><text class="t-tick" x="24.0" y="163.0" text-anchor="start">green  (j=3)</text><rect class="cell" x="170" y="150" width="2.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="178.0" y="163.0" text-anchor="start">0.013</text><text class="t-val" x="290.0" y="163.0" text-anchor="start">+6.11</text><rect class="s1" x="450" y="150" width="3.9" height="17" rx="3"/><text class="t-val" x="459.9" y="163.0" text-anchor="start">+0.077</text><text class="t-tick t-strong" x="24.0" y="189.0" text-anchor="start">tea  (j=4)</text><rect class="cell" x="170" y="176" width="37.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="213.0" y="189.0" text-anchor="start">0.370</text><text class="t-val" x="290.0" y="189.0" text-anchor="start">+12.15</text><rect class="s1" x="450" y="176" width="230.0" height="17" rx="3"/><text class="t-val" x="686.0" y="189.0" text-anchor="start">+4.497</text><text class="t-tick" x="24.0" y="215.0" text-anchor="start">,  (j=5)</text><rect class="cell" x="170" y="202" width="8.7" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="184.7" y="215.0" text-anchor="start">0.087</text><text class="t-val" x="290.0" y="215.0" text-anchor="start">+0.04</text><rect class="s1" x="450" y="202" width="2.0" height="17" rx="3"/><text class="t-val" x="458.0" y="215.0" text-anchor="start">+0.003</text><text class="t-tick" x="24.0" y="241.0" text-anchor="start">so  (j=6)</text><rect class="cell" x="170" y="228" width="2.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="178.0" y="241.0" text-anchor="start">0.001</text><text class="t-val" x="290.0" y="241.0" text-anchor="start">+2.50</text><rect class="s1" x="450" y="228" width="2.0" height="17" rx="3"/><text class="t-val" x="458.0" y="241.0" text-anchor="start">+0.001</text><text class="t-tick" x="24.0" y="267.0" text-anchor="start">I  (j=7)</text><rect class="cell" x="170" y="254" width="2.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="178.0" y="267.0" text-anchor="start">0.002</text><text class="t-val" x="290.0" y="267.0" text-anchor="start">+1.02</text><rect class="s1" x="450" y="254" width="2.0" height="17" rx="3"/><text class="t-val" x="458.0" y="267.0" text-anchor="start">+0.002</text><text class="t-tick" x="24.0" y="293.0" text-anchor="start">drink  (j=8)</text><rect class="cell" x="170" y="280" width="14.7" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="190.7" y="293.0" text-anchor="start">0.147</text><text class="t-val" x="290.0" y="293.0" text-anchor="start">+6.65</text><rect class="s1" x="450" y="280" width="50.1" height="17" rx="3"/><text class="t-val" x="506.1" y="293.0" text-anchor="start">+0.979</text><text class="t-tick" x="24.0" y="319.0" text-anchor="start">green  (j=9)</text><rect class="cell" x="170" y="306" width="2.0" height="17" rx="3" style="fill-opacity:0.7"/><text class="t-val" x="178.0" y="319.0" text-anchor="start">0.011</text><text class="t-val" x="290.0" y="319.0" text-anchor="start">+6.28</text><rect class="s1" x="450" y="306" width="3.6" height="17" rx="3"/><text class="t-val" x="459.6" y="319.0" text-anchor="start">+0.070</text><text class="t-note" x="24.0" y="350.0" text-anchor="start">sum +5.465: the head 0.3 term of the expansion. Almost all of it comes from "tea" itself.</text></svg><figcaption>Head 0.3 at the last "green", taken apart by source token. For each token: the attention weight, the value of that token's OV column in the row for "tea", and their product. "Tea" has weight 0.370 and column value +12.15, so it adds +4.497. "Drink" adds +0.979 and BOS takes away 0.167. The total, +5.465, is the head 0.3 term.</figcaption></figure>

Almost all of the head's vote comes from one source: " tea" at position 4. Its column has the value +12.15 in the row for " tea", the largest of all ten sources. That is the copying behaviour of Part 2: reading " tea" pushes up " tea". The weight 0.370 decides how much of that column arrives.

The two rows for " green" have different column values, +6.110 and +6.281. The token is the same, but the layer-norm scales of the two positions differ a little, because their position embeddings differ. Our columns include the scales. In the paper's model, with no layer norm, the two columns would be equal.

### Where and what, once more

The paper then explains why the two halves of the head term can be thought about apart.

> [!PAPER] · Splitting Attention Head terms into Query-Key and Output-Value Circuits
> [![The attention pattern is a function of both the source and destination token, but once a destination token has decided how much to attend to a source token, the effect on the output is solely a function of that source token](/img/papers/circuits/p3-separable.png)](/img/papers/circuits/p3-separable.png)
>
> **What it says:** the attention pattern depends on both the source and the destination token. But once the weight is set, the effect on the output "is solely a function of that source token". If several destinations read the same source by the same amount, "the source token will have the same effect on the logits". A footnote adds that, technically, the pattern depends on "all possible source tokens from the start to the destination token", because the softmax normalises.
>
> **What it means:**
>
> Think of a radio. The volume knob depends on who is listening and what they want. But the song on the station does not depend on the listener. Two people with the knob at the same level hear exactly the same song.
>
> In the head, the knob is the weight $$A^h_{ij}$$, and the song is the OV column of the source token. Look at the formula above: the column $$(W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}})\,t_j$$ contains only $$t_j$$. Nothing about position $$i$$ enters it. Only the weight knows about the destination.
>
> The footnote is a fair warning. The weight on one source is not decided by that source alone. The softmax divides by a sum over all sources, so a change anywhere in the row moves every weight. We test this next.

We changed one token of the prompt, " drink", to " want", and looked at the last row of head 0.3 again:

```text
== 5. ONE ROW OF THE PATTERN DEPENDS ON EVERY SOURCE TOKEN (softmax), head 0.3, last row
  before: "I like green tea, so I drink green"   weight on " tea" 0.370, on BOS 0.366, on token 8 ' drink' 0.147
  after : "I like green tea, so I want green"   weight on " tea" 0.434, on BOS 0.429, on token 8 ' want' 0.000
  the raw score of " tea" is the same in both runs: 2.929 and 2.929
```

<figure class="fig"><svg viewBox="0 0 780 278" role="img" aria-label="Two rows of attention weights of head 0.3 from the last token. Before: BOS 0.37, tea 0.37, drink 0.15, comma 0.09, the rest near zero. After drink is replaced by want: BOS 0.43, tea 0.43, want 0.00. The scores of tea and BOS are unchanged, but their weights grow because the softmax divides by a smaller total."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">Head 0.3, last row of the pattern: change one source token, every weight moves</text><text class="t-note" x="24.0" y="90.0" text-anchor="start">before</text><text class="t-tick" x="140.0" y="64.0" text-anchor="middle">BOS</text><g class="mark"><title>BOS: 0.366</title><rect class="cell" x="110.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.366"/></g><text class="t-cell" x="140.0" y="94.0" text-anchor="middle">0.37</text><text class="t-tick" x="206.0" y="64.0" text-anchor="middle">I</text><g class="mark"><title>I: 0.003</title><rect class="cell" x="176.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="206.0" y="94.0" text-anchor="middle">0.00</text><text class="t-tick" x="272.0" y="64.0" text-anchor="middle">like</text><g class="mark"><title>like: 0.001</title><rect class="cell" x="242.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="272.0" y="94.0" text-anchor="middle">0.00</text><text class="t-tick" x="338.0" y="64.0" text-anchor="middle">green</text><g class="mark"><title>green: 0.013</title><rect class="cell" x="308.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="338.0" y="94.0" text-anchor="middle">0.01</text><text class="t-tick" x="404.0" y="64.0" text-anchor="middle">tea</text><g class="mark"><title>tea: 0.370</title><rect class="cell" x="374.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.370"/></g><text class="t-cell" x="404.0" y="94.0" text-anchor="middle">0.37</text><text class="t-tick" x="470.0" y="64.0" text-anchor="middle">,</text><g class="mark"><title>,: 0.087</title><rect class="cell" x="440.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.087"/></g><text class="t-cell" x="470.0" y="94.0" text-anchor="middle">0.09</text><text class="t-tick" x="536.0" y="64.0" text-anchor="middle">so</text><g class="mark"><title>so: 0.001</title><rect class="cell" x="506.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="536.0" y="94.0" text-anchor="middle">0.00</text><text class="t-tick" x="602.0" y="64.0" text-anchor="middle">I</text><g class="mark"><title>I: 0.002</title><rect class="cell" x="572.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="602.0" y="94.0" text-anchor="middle">0.00</text><text class="t-tick t-strong" x="668.0" y="64.0" text-anchor="middle">drink</text><g class="mark"><title>drink: 0.147</title><rect class="cell" x="638.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.147"/></g><text class="t-cell" x="668.0" y="94.0" text-anchor="middle">0.15</text><text class="t-tick" x="734.0" y="64.0" text-anchor="middle">green</text><g class="mark"><title>green: 0.011</title><rect class="cell" x="704.0" y="72" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="734.0" y="94.0" text-anchor="middle">0.01</text><text class="t-note" x="24.0" y="182.0" text-anchor="start">after</text><text class="t-tick" x="140.0" y="156.0" text-anchor="middle">BOS</text><g class="mark"><title>BOS: 0.429</title><rect class="cell" x="110.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.429"/></g><text class="t-cell" x="140.0" y="186.0" text-anchor="middle">0.43</text><text class="t-tick" x="206.0" y="156.0" text-anchor="middle">I</text><g class="mark"><title>I: 0.003</title><rect class="cell" x="176.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="206.0" y="186.0" text-anchor="middle">0.00</text><text class="t-tick" x="272.0" y="156.0" text-anchor="middle">like</text><g class="mark"><title>like: 0.001</title><rect class="cell" x="242.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="272.0" y="186.0" text-anchor="middle">0.00</text><text class="t-tick" x="338.0" y="156.0" text-anchor="middle">green</text><g class="mark"><title>green: 0.015</title><rect class="cell" x="308.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="338.0" y="186.0" text-anchor="middle">0.01</text><text class="t-tick" x="404.0" y="156.0" text-anchor="middle">tea</text><g class="mark"><title>tea: 0.434</title><rect class="cell" x="374.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.434"/></g><text class="t-cell" x="404.0" y="186.0" text-anchor="middle">0.43</text><text class="t-tick" x="470.0" y="156.0" text-anchor="middle">,</text><g class="mark"><title>,: 0.102</title><rect class="cell" x="440.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.102"/></g><text class="t-cell" x="470.0" y="186.0" text-anchor="middle">0.10</text><text class="t-tick" x="536.0" y="156.0" text-anchor="middle">so</text><g class="mark"><title>so: 0.001</title><rect class="cell" x="506.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="536.0" y="186.0" text-anchor="middle">0.00</text><text class="t-tick" x="602.0" y="156.0" text-anchor="middle">I</text><g class="mark"><title>I: 0.002</title><rect class="cell" x="572.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="602.0" y="186.0" text-anchor="middle">0.00</text><text class="t-tick t-strong" x="668.0" y="156.0" text-anchor="middle">want</text><g class="mark"><title>want: 0.000</title><rect class="cell" x="638.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="668.0" y="186.0" text-anchor="middle">0.00</text><text class="t-tick" x="734.0" y="156.0" text-anchor="middle">green</text><g class="mark"><title>green: 0.013</title><rect class="cell" x="704.0" y="164" width="60" height="34" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="734.0" y="186.0" text-anchor="middle">0.01</text><text class="t-muted" x="24.0" y="246.0" text-anchor="start">Only "drink" became "want", and the head now ignores it. The scores of "tea" and BOS did not change,</text><text class="t-muted" x="24.0" y="264.0" text-anchor="start">but the softmax shares out the freed weight: "tea" goes from 0.37 to 0.43.</text></svg><figcaption>The last row of head 0.3's pattern, before and after one token changes. Before, BOS and "tea" each get about 0.37 and "drink" gets 0.15. After "drink" becomes "want", the head gives "want" no weight, and BOS and "tea" rise to 0.43 each. The score of "tea" did not change, but the softmax shares out the weight that "want" left free.</figcaption></figure>

The score between the last " green" and " tea" did not change: 2.929 in both runs. Only the token at position 8 changed. Yet the weight on " tea" rose from 0.370 to 0.434. The new token " want" took no weight, and the softmax gave its share to the others. This is what the footnote means: the QK circuit gives each pair a score, but the weight of a pair depends on the whole row.

## Freezing the attention patterns {§OV and QK Independence (The Freezing Attention Patterns Trick)}

The head term has one non-linear piece: the pattern $$A^h$$, which comes from a softmax of the tokens. The paper now offers a way to remove even that.

> [!PAPER] · OV and QK Independence (The Freezing Attention Patterns Trick)
> [![The freezing trick: imagine running the model twice; the first time collect the attention patterns, which only depend on the QK circuit; the second time replace them with the frozen patterns, which gives logits that are a linear function of the tokens](/img/papers/circuits/p3-freeze.png)](/img/papers/circuits/p3-freeze.png)
>
> **What it says:** the OV and QK circuits are each "functions we can understand (linear or bilinear functions operating on matrices we understand)". But "is it really principled to think about them independently?" The paper suggests to "imagine running the model twice". The first run collects the attention patterns, and "this only depends on the QK circuit". The second run replaces the patterns with the "frozen" ones. "This gives you a function where the logits are a linear function of the tokens!"
>
> **What it means:**
>
> Think of a factory with a sorting machine and a painting machine. The sorter decides which parts go to which station, and it is hard to predict. The painter is simple: twice the paint in, twice the colour out. To study the painter, you film the sorter once. Then you replay the film and feed in new parts.
>
> The sorter is the QK circuit, and its film is the attention pattern. Once the pattern is recorded, it is a fixed table of numbers. Every path term becomes "a fixed table times the tokens". A sum of such terms is linear in the tokens.
>
> The trick answers the question in the box. Thinking about OV and QK apart is principled, because it is exact once the pattern is fixed. The QK circuit decides the pattern, and the OV circuit does everything else.

> [!DEFINITION] Frozen attention pattern
> An attention pattern recorded in one run of the model and then reused, unchanged, in a second run. In the second run the heads do not compute their patterns from the input; they use the recorded ones. The second run is not the real model, but it equals the real model on the input that made the patterns.

What does "linear in the tokens" promise? A **linear map** (Part 1) keeps sums and scales. The output for $$a+b$$ is the output for $$a$$ plus the output for $$b$$. The output for $$2a$$ is twice the output for $$a$$. The toy shows it. We keep $$A$$ fixed and swap the first token from a to c:

```text
== C. FREEZING THE PATTERN
  "c b" with A frozen: logits [[2.0, 2.0, 0.0], [1.0, 1.5, 0.0]]; change at position 2 = [0.5, 0.0, 0.0]
  A[2,1] x (OV column c - OV column a) = 0.5 x ([1.0, 1.0, 0.0] - [0.0, 1.0, 0.0]) = [0.5, 0.0, 0.0]
  linear: T("a b" + "c b") = [3.0, 3.0, 0.0, 1.5, 3.0, 0.0];  T("a b") + T("c b") = [3.0, 3.0, 0.0, 1.5, 3.0, 0.0]
```

The change at position 2 is exactly the weight, 0.5, times the difference between two OV columns. You can check it with a pencil: column c is $$(1,1,0)$$, column a is $$(0,1,0)$$, so the change is $$(0.5,0,0)$$.

### The test on the real model

On the real model we froze the patterns of all eight heads, recorded on our prompt. We also froze the layer-norm scales, because the paper's model has no layer norm. Then we fed in token embeddings that are not real text: the sum of two prompts, and twice one prompt.

We call $$g(t)$$ the part of the logits that depends on the tokens. It is the logits for $$t$$ minus the logits with every token embedding set to zero. The zero run keeps the positions and biases, so subtracting it leaves only the token paths. Prompt $$a$$ is ours, and prompt $$b$$ is *"The cat sat on the mat and the dog"*, also ten tokens.

```python
frozen = {'blocks.0.attn.hook_pattern': ..., 'blocks.0.ln1.hook_scale': ..., 'ln_final.hook_scale': ...}  # from run 1
g = lambda e: run(e, freeze) - run(zero, freeze)        # run 2: replace the token embeddings, reuse the patterns
```

```text
== 4. FREEZE THE ATTENTION PATTERNS: THE LOGITS BECOME LINEAR IN THE TOKENS
  frozen    : g(a + b) vs g(a) + g(b): max diff 4.4e-05;  g(2a) vs 2 g(a): max diff 5.4e-05  (typical |g(a)| = 1.73)
  not frozen: g(a + b) vs g(a) + g(b): max diff 1.7e+01;  g(2a) vs 2 g(a): max diff 2.7e+01  (typical |g(a)| = 2.14)
```

<figure class="fig"><svg viewBox="0 0 760 312" role="img" aria-label="A diagram of the freezing trick. Run 1 on real text records the attention pattern A of every head, which depends only on the QK circuits. Run 2 on any tokens reuses the recorded patterns, and the logits become a linear function of the tokens. Below, a test on attn-only-1l: with frozen patterns, g(a + b) and g(a) + g(b) differ by at most 0.00004; with recomputed patterns they differ by up to 17."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">The freezing trick: run the model twice</text><rect class="box" x="24.0" y="50.0" width="190.0" height="32.0" rx="6"/><text class="t-tick" x="119.0" y="70.5" text-anchor="middle">run 1: real text</text><line class="edge" x1="218.0" y1="66.0" x2="262.0" y2="66.0" marker-end="url(#ah)"/><rect class="box-1" x="266.0" y="50.0" width="220.0" height="32.0" rx="6"/><text class="t-tick" x="376.0" y="70.5" text-anchor="middle">record A of every head</text><text class="t-muted" x="496.0" y="71.0" text-anchor="start">this run uses only</text><text class="t-muted" x="496.0" y="89.0" text-anchor="start">the QK circuits</text><rect class="box" x="24.0" y="124.0" width="190.0" height="32.0" rx="6"/><text class="t-tick" x="119.0" y="144.5" text-anchor="middle">run 2: any tokens</text><line class="edge" x1="218.0" y1="140.0" x2="262.0" y2="140.0" marker-end="url(#ah)"/><rect class="box-2" x="266.0" y="124.0" width="220.0" height="32.0" rx="6"/><text class="t-tick" x="376.0" y="144.5" text-anchor="middle">heads use the recorded A</text><line class="edge" x1="490.0" y1="140.0" x2="534.0" y2="140.0" marker-end="url(#ah)"/><rect class="box-3" x="538.0" y="124.0" width="196.0" height="32.0" rx="6"/><text class="t-tick" x="636.0" y="144.5" text-anchor="middle">logits: linear in tokens</text><line class="path" x1="376.0" y1="86.0" x2="376.0" y2="120.0" marker-end="url(#ah-on)"/><text class="t-note" x="24.0" y="196.0" text-anchor="start">Test on attn-only-1l: is g(a + b) = g(a) + g(b)?  (g = the part of the logits that depends on the tokens)</text><text class="t-tick" x="24.0" y="232.0" text-anchor="start">patterns frozen</text><rect class="s1" x="190" y="219" width="3.0" height="20" rx="3"/><text class="t-val" x="203.0" y="233.0" text-anchor="start">largest error 4.4e-05</text><text class="t-tick" x="24.0" y="266.0" text-anchor="start">patterns recomputed</text><rect class="s2" x="190" y="253" width="389.1" height="20" rx="3"/><text class="t-val" x="589.1" y="267.0" text-anchor="start">largest error 1.7e+01</text><text class="t-muted" x="24.0" y="296.0" text-anchor="start">typical size of g(a): 1.73. Frozen, the error is rounding noise; recomputed, it is larger than the logits.</text></svg><figcaption>The freezing trick as two runs. Run 1 on real text records the pattern of every head; that run uses only the QK circuits. Run 2 feeds in any tokens and reuses the recorded patterns, so the logits become linear in the tokens. The test on attn-only-1l: with frozen patterns, g(a + b) and g(a) + g(b) differ by at most 0.000044; with patterns computed again, by up to 17.</figcaption></figure>

With frozen patterns, the largest error is 0.000044, while a typical logit part is 1.73. That is rounding noise: the frozen model is exactly linear. Without freezing, the error reaches 17 for sums and 27 for doubling, larger than the logits themselves. The only difference between the two rows is the softmax of the patterns.

Now a more useful use of the trick. We swap one source token, " tea" at position 4, for " coffee", with the patterns frozen. The formula above predicts the change at the last position: for each head, the weight on position 4 times the difference of two OV columns.

```text
  swap source token 4 " tea" -> " coffee", patterns and scales frozen, watch the last position:
    change of all logits = sum over heads of A[last, 4] x (OV column coffee - OV column tea): max diff 2.9e-05
    frozen run: logit of " tea" changes by -2.361, logit of " coffee" by +2.739
    the real model on "I like green coffee, so I drink green" predicts: [(' coffee', 0.562), (' tea', 0.241), (' and', 0.015)]
```

The prediction from the OV columns matches the frozen run over all 48,262 logits, to 0.000029. The swap lowers " tea" by 2.361 and raises " coffee" by 2.739. So, with the reading fixed, what the heads move depends only on which token they read.

The real model, with its patterns free, agrees in direction: it now predicts " coffee" with probability 0.562, and " tea" falls to 0.241. The direct path still sees only " green", so its vote for " tea" is unchanged. That is one reason tea keeps some probability.

**What the frozen test cannot show.** A frozen run equals the real model only on the text that made the patterns. On new text the real patterns change, and the frozen answer can be wrong. We also froze the layer-norm scales; the real model computes them again. The trick is a tool for reading the OV side, not a replacement for the model.

## Two layers, in outline {§Path Expansion of Logits}

The trick does not stop at one layer. The two-layer section of the paper applies it again, and the result is the map for Parts 5 and 6.

> [!PAPER] · Path Expansion of Logits
> [![The two-layer model as a product of four factors, expanded into the direct path, the individual attention head terms of both layers, and the virtual attention head terms (A^h2 A^h1) ⊗ (W_U W_OV^h2 W_OV^h1 W_E)](/img/papers/circuits/p3-two-layer-fig.png)](/img/papers/circuits/p3-two-layer-fig.png)
>
> **What it says:** with two layers, $$T=(\text{Id}\otimes W_{\mathrm{U}})\big(\text{Id}+\sum_{h\in H_2}A^h\otimes W_{\mathrm{OV}}^h\big)\big(\text{Id}+\sum_{h\in H_1}A^h\otimes W_{\mathrm{OV}}^h\big)(\text{Id}\otimes W_{\mathrm{E}})$$. Expanded, it is the direct path, plus "individual attention head" terms for the heads of both layers, plus "virtual attention head" terms $$(A^{h_2}A^{h_1})\otimes(W_{\mathrm{U}} W_{\mathrm{OV}}^{h_2} W_{\mathrm{OV}}^{h_1} W_{\mathrm{E}})$$.
>
> **What it means:**
>
> Now there are two middle brackets, and each has a choice: pass through with $$\text{Id}$$, or go through one of its heads. Pass through both, and you get the direct path. Go through one head in either layer, and you get a single-head term, just like the one-layer model.
>
> The new kind of path goes through a head in layer 1 and then a head in layer 2. The mixed-product property multiplies the two patterns and the two OV matrices. The result looks like one head with pattern $$A^{h_2}A^{h_1}$$ and OV matrix $$W_{\mathrm{OV}}^{h_2}W_{\mathrm{OV}}^{h_1}$$. That is the virtual head that Part 2 met at the end.

Count the paths like the brackets in $$(1+2)(1+3)$$. Each middle bracket has $$1+8$$ parts, so there are $$9\times9=81$$ products. The toy script counts them for a few head numbers:

```text
== D. COUNTING PATHS
   8 heads per layer: one layer 9 terms; two layers 81 terms (1 + 16 + 64)
  12 heads per layer: one layer 13 terms; two layers 169 terms (1 + 24 + 144)
```

So the two-layer model has 1 direct path, 16 single-head terms and 64 virtual-head terms. The paper's numbering of layers starts at 1; TransformerLens names the same layers 0 and 1.

> [!PAPER] · Path Expansion of Logits
> [![Two of these terms, the direct path and the individual head terms, are identical to the one-layer model; the virtual attention head term corresponds to V-Composition, and virtual heads tend to not play a significant role in small two-layer models](/img/papers/circuits/p3-two-layer.png)](/img/papers/circuits/p3-two-layer.png)
>
> **What it says:** the direct path and the individual head terms "are identical to the one-layer model". The virtual attention head term "corresponds to V-Composition". Virtual heads are "conceptually very interesting", but in practice "they tend to not play a significant role in small two-layer models".
>
> **What it means:**
>
> The first sentence is the payoff of the method. Everything we learned to read on one layer can be read again on two layers, unchanged. Only the new terms need new tools.
>
> "V-composition" means that a head in layer 2 reads, as its value, something a head in layer 1 wrote. The information hops twice: from token A to token B, then from token B to token C. Part 5 explains this and the two other kinds of composition.
>
> The last sentence is a measured result of the paper, not a rule. Part 5 shows how the paper measures how much each group of terms matters.

> [!DEFINITION] Virtual attention head
> A path through one head of an early layer and then one head of a later layer. It acts like a single head with pattern $$A^{h_2}A^{h_1}$$ and OV matrix $$W_{\mathrm{OV}}^{h_2}W_{\mathrm{OV}}^{h_1}$$, although no such head exists in the weights.

There is one catch, and a footnote of the freezing section flags it. In models with more than one layer, the QK circuit "can be more complicated". A layer-2 head computes its pattern from the stream, and the stream already holds what layer-1 heads wrote. So the layer-2 patterns depend on layer 1. This is Q- and K-composition, and the induction heads of Part 6 are built from it. With frozen patterns the expansion above is still exact; the patterns themselves are what need explaining.

### A first look at the real two-layer model

We expanded `attn-only-2l`, a two-layer model with eight heads per layer, on the same prompt. With frozen patterns and scales, we computed all 81 path terms, plus the position and bias parts, and grouped them:

```text
== 6. TWO LAYERS IN OUTLINE (attn-only-2l): 1 + 16 + 64 PATH TERMS
  same prompt; attn-only-2l predicts ' tea' after the last " green"
  terms: direct path 1, layer-0 heads 8, layer-1 heads 8, virtual heads 64, positions and biases 92  (paper's terms: 1 + 16 + 64 = 81)
  |sum of all terms - model logits| max = 2.7e-05
  group                    adds to logit of ' tea'   spread of the 48,262 logits (std)
  direct path                               +3.068                               0.907
  layer-0 heads                             +0.376                               0.881
  layer-1 heads                            +15.291                               2.002
  virtual heads                             +1.061                               1.082
  positions and biases                      +2.962                               2.933
  sum                                      +22.758   (model logit 22.758, probability 0.986)
```

<figure class="fig"><svg viewBox="0 0 760 250" role="img" aria-label="Diverging bars for attn-only-2l at the last position: the logit of tea split into groups of path terms: direct path (1) +3.068, layer-0 heads (8) +0.376, layer-1 heads (8) +15.291, virtual heads (64) +1.061, positions and biases +2.962."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">attn-only-2l at the last "green": what each group of paths adds to the logit of "tea"</text><text class="t-muted" x="24.0" y="48.0" text-anchor="start">the paper's 81 terms in four groups, plus the parts the paper leaves out</text><line class="axis" x1="464.0" y1="62.0" x2="464.0" y2="216.0"/><text class="t-tick" x="204.0" y="85.0" text-anchor="end">direct path (1)</text><g class="mark"><title>direct path (1): +3.068</title><rect class="s1" x="464.0" y="70.0" width="47.9" height="21.0" rx="3"/></g><text class="t-val" x="517.9" y="85.0" text-anchor="start">+3.068</text><text class="t-tick" x="204.0" y="115.0" text-anchor="end">layer-0 heads (8)</text><g class="mark"><title>layer-0 heads (8): +0.376</title><rect class="s1" x="464.0" y="100.0" width="5.9" height="21.0" rx="3"/></g><text class="t-val" x="475.9" y="115.0" text-anchor="start">+0.376</text><text class="t-tick t-strong" x="204.0" y="145.0" text-anchor="end">layer-1 heads (8)</text><g class="mark"><title>layer-1 heads (8): +15.291</title><rect class="s1" x="464.0" y="130.0" width="238.9" height="21.0" rx="3"/></g><text class="t-val" x="708.9" y="145.0" text-anchor="start">+15.291</text><text class="t-tick" x="204.0" y="175.0" text-anchor="end">virtual heads (64)</text><g class="mark"><title>virtual heads (64): +1.061</title><rect class="s1" x="464.0" y="160.0" width="16.6" height="21.0" rx="3"/></g><text class="t-val" x="486.6" y="175.0" text-anchor="start">+1.061</text><text class="t-tick" x="204.0" y="205.0" text-anchor="end">positions and biases</text><g class="mark"><title>positions and biases: +2.962</title><rect class="s1" x="464.0" y="190.0" width="46.3" height="21.0" rx="3"/></g><text class="t-val" x="516.3" y="205.0" text-anchor="start">+2.962</text><text class="t-note" x="24.0" y="234.0" text-anchor="start">sum: +22.758 = the model's logit. Part 5 takes the groups apart.</text></svg><figcaption>The logit of "tea" in attn-only-2l at the last "green", split into groups of paths. The eight layer-1 heads add +15.291, the direct path +3.068, the 64 virtual heads together +1.061, the layer-0 heads +0.376, and the positions and biases +2.962. The groups add up to the model's logit, 22.758.</figcaption></figure>

The sum again matches the model, to 0.000027. On this prompt, the second-layer heads carry most of the vote for " tea": +15.291 of 22.758. The 64 virtual heads together add +1.061, less than a tenth of what the layer-1 heads add. That fits the paper's remark, but one prompt cannot test it. Part 5 does the real measurement, and Part 6 shows which second-layer heads do the work and why.

> [!TAKEAWAYS] Key takeaways
> - **A one-layer attention-only model is a product of three maps:** $$(\text{Id}\otimes W_{\mathrm{U}})\big(\text{Id}+\sum_h A^h\otimes W_{\mathrm{OV}}^h\big)(\text{Id}\otimes W_{\mathrm{E}})$$. Computed by hand on `attn-only-1l`, it matched the model's logits to 0.000026.
> - **The path expansion trick multiplies out the brackets.** A product where each factor is a layer becomes a sum where each term is a path from tokens to logits: $$\text{Id}\otimes W_{\mathrm{U}} W_{\mathrm{E}}+\sum_h A^h\otimes W_{\mathrm{U}} W_{\mathrm{OV}}^h W_{\mathrm{E}}$$.
> - **The terms really add up to the model.** At the last " green", the 9 paper terms plus position and bias parts summed to the real logit of " tea", 20.128.
> - **The direct path is a bigram table.** It sees only the current token: after " green" it pushes up "ery", "houses" and "wich", and it adds +5.501 to " tea".
> - **A head term is weight times OV column.** Head 0.3 adds +5.465 to " tea"; +4.497 of it comes from reading " tea" itself (weight 0.370, column value +12.15).
> - **The pattern of a row depends on every source.** Changing " drink" to " want" left the score of " tea" at 2.929 but raised its weight from 0.370 to 0.434.
> - **Freezing the patterns makes the logits linear in the tokens.** Frozen, the sum test failed by at most 0.000044; recomputed, by up to 17. Swapping " tea" for " coffee" then changes the logits by exactly weight times the difference of two OV columns.
> - **Two layers give 81 path terms:** 1 direct, 16 single-head and 64 virtual-head terms. On our prompt in `attn-only-2l`, the layer-1 heads add +15.291 of the logit 22.758, and the 64 virtual heads together +1.061.

**Next, in Part 4:** we read the head terms of the one-layer model. Each head's QK and OV tables become lists of skip-trigrams. The paper shows copying, simple in-context learning, and the bugs that this design cannot avoid.

<details>
<summary>Run it yourself</summary>

The two scripts behind this part are [`circuits_part3.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/circuits/circuits_part3.py) (every real-model number: the product by hand, the expanded terms, one head term by source token, the freezing tests, the softmax row, and the two-layer outline) and [`circuits_part3_toy.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/circuits/circuits_part3_toy.py) (the hand-sized examples). They use `attn-only-1l` and `attn-only-2l` and run on the CPU.

```bash
pip install torch "transformer_lens<4"
python circuits_part3.py        # writes results/part3.json and results/part3_stdout.txt
python circuits_part3_toy.py    # writes results/part3_toy.json
```

<figure><img src="/img/papers/circuits/part3-run.png" alt="Terminal output of circuits_part3.py: the product of three terms checked against the model, the twelve expanded terms and their effect on the logit of tea, head 0.3 taken apart by source token, the freezing tests, the softmax row test, and the two-layer outline" loading="lazy" /><figcaption>The real output of circuits_part3.py.</figcaption></figure>

<figure><img src="/img/papers/circuits/part3_toy-run.png" alt="Terminal output of circuits_part3_toy.py: a product of sums expanded, the toy model as a product and as a sum of paths, the freezing test, and the number of path terms for one and two layers" loading="lazy" /><figcaption>The real output of circuits_part3_toy.py.</figcaption></figure>

</details>

## The paper and the code

[A Mathematical Framework for Transformer Circuits](https://transformer-circuits.pub/2021/framework/index.html) (Transformer Circuits Thread, 2021) · [our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/circuits)
