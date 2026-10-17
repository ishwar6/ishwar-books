---
title: "The Architecture: Stacks, Feed-Forward Layers and Positions"
description: "How the Transformer is built, block by block: the encoder and decoder stacks, the add and norm step, the feed-forward layer, the embeddings and the sine waves that give each word its place. We open a released English to German model and check each piece."
part: 2
covers: "§3, Figure 1, §3.1, §3.3, §3.4, §3.5"
date: 2026-10-18
tags: [transformer, attention, architecture, positional-encoding, nlp]
---

Part 1 showed the promise: a translation model with no recurrence, built on attention. This part opens the box and names every block in Figure 1, except attention itself, which gets [Part 3](part-3-attention.md) for itself.

To make each block concrete, we load a released English to German Transformer and send one sentence through it. Each number below comes from our script.

## An encoder and a decoder {§3}

> [!PAPER] · §3
> [![The first paragraph of Section 3, with encoder-decoder structure, continuous representations z, one element at a time and auto-regressive highlighted](/img/papers/transformer/p2-encoder-decoder.png)](/img/papers/transformer/p2-encoder-decoder.png)
>
> **What it says:** most strong translation models "have an encoder-decoder structure". The encoder maps the input symbols to "continuous representations z = (z1, ..., zn)". The decoder then makes the output "one element at a time", and it is "auto-regressive".
>
> **What it means:**
>
> Think of a human interpreter. First she listens to the whole English sentence and builds an understanding of it in her head. Then she speaks the German sentence, one word after the other. The encoder is the listening half. The decoder is the speaking half.
>
> The "continuous representations" are the understanding. For each input word, the encoder writes one list of numbers, a vector. So a sentence of 12 tokens becomes 12 vectors. The paper calls this set $$z$$. The decoder can look at all of $$z$$ at any time, as the interpreter can think back to any part of the sentence she heard.
>
> "Auto-regressive" says how the speaking works. To choose word 5, the decoder reads $$z$$ and also the 4 words it already said. Then word 5 joins the input for word 6. The Transformer keeps this outer design from the older models of the [Background](background-the-road-to-the-transformer.md) page. What it changes is what is inside each half.

> [!DEFINITION] Token
> One piece of text that the model reads as a unit. It can be a whole word, part of a word or a punctuation mark. A tokenizer cuts the text into tokens, and each token has a number, its **id**, in a fixed **vocabulary**.

> [!DEFINITION] Vector
> An ordered list of numbers. In this paper, almost every vector has 512 numbers. The length of the list is its **dimension**.

### Our model and one sentence

The paper's own trained weights are not easy to run today. So we use `Helsinki-NLP/opus-mt-en-de`, a released MarianMT model that translates English to German. It has the paper's base shapes: 6 encoder layers, 6 decoder layers, vectors of 512 numbers, 8 attention heads and a feed-forward size of 2048. It is a different model, trained on different data, and it has a few small differences that we point out when we meet them.

Our script reads its configuration:

```text
== 1. configuration of Helsinki-NLP/opus-mt-en-de
  d_model                    512
  encoder_layers             6
  decoder_layers             6
  encoder_attention_heads    8
  encoder_ffn_dim            2048
  vocab_size                 58101
  activation_function        swish
  scale_embedding            True
  max_position_embeddings    512
```

Then it translates one sentence:

```text
  input  : The cat sat on the mat because it was tired.
  tokens : ▁The ▁cat ▁sat ▁on ▁the ▁mat ▁because ▁it ▁was ▁tired . </s> (12 tokens)
  ids    : [36, 12064, 10369, 32, 4, 19160, 433, 56, 58, 15871, 3, 0]
  output : Die Katze saß auf der Matte, weil sie müde war.
```

The `▁` sign marks the start of a word. The tokenizer adds `</s>`, an "end of sentence" token, so the encoder sees 12 tokens. The German output is correct. The model even chose "sie" for "it", because "Katze" is a feminine noun in German.

## Figure 1: the whole model in one picture {§Figure 1}

> [!PAPER] · Figure 1
> [![Figure 1 of the paper: on the left, the encoder: Inputs go into Input Embedding, a Positional Encoding is added, then a grey box repeated N times holds Multi-Head Attention with Add and Norm, and Feed Forward with Add and Norm. On the right, the decoder: Outputs shifted right go into Output Embedding plus Positional Encoding, then a grey box repeated N times holds Masked Multi-Head Attention, Multi-Head Attention that also takes the encoder output, and Feed Forward, each with Add and Norm. On top, Linear and Softmax give Output Probabilities](/img/papers/transformer/p2-figure1.png)](/img/papers/transformer/p2-figure1.png)
>
> **What it says:** "Figure 1: The Transformer - model architecture."
>
> **What it means:**
>
> Read the figure from the bottom up. On the left, the input words become vectors (Input Embedding), and a wave sign adds their positions. The vectors then pass through a grey box. The "N×" next to it means that the box is repeated: the paper uses N = 6 copies, one on top of the other.
>
> On the right, the decoder does the same with the words it already wrote ("Outputs (shifted right)"). Its grey box has three parts, not two. The middle part receives an arrow from the encoder: this is where the German side reads the English side. At the top, a Linear layer and a Softmax turn each decoder vector into a probability for each word of the vocabulary.
>
> Notice what is missing. There is no loop and no arrow from one word to the next word. Each yellow "Add & Norm" box has a second arrow that goes around the block below it. These two facts are the whole story of this part.

> [!PAPER] · §3
> [![The sentence under Figure 1, with stacked self-attention and point-wise, fully connected layers and left and right halves of Figure 1 highlighted](/img/papers/transformer/p2-overall.png)](/img/papers/transformer/p2-overall.png)
>
> **What it says:** the Transformer uses "stacked self-attention and point-wise, fully connected layers for both the encoder and decoder", shown in the "left and right halves of Figure 1".
>
> **What it means:**
>
> This sentence lists the only two kinds of blocks in the model. "Self-attention" lets the words of one sentence exchange information. "Point-wise, fully connected layers" then work on each word alone. "Stacked" means that the model repeats this pair many times.
>
> A useful way to see it: attention is a meeting where everyone listens to everyone. The point-wise layer is the time after the meeting, where each person thinks alone about what they heard. The Transformer alternates meeting and thinking, six times on each side.

## The encoder stack {§3.1}

> [!PAPER] · §3.1
> [![The Encoder paragraph of Section 3.1, with N = 6 identical layers, multi-head self-attention mechanism, position-wise, residual connection, LayerNorm(x + Sublayer(x)) and d_model = 512 highlighted](/img/papers/transformer/p2-encoder.png)](/img/papers/transformer/p2-encoder.png)
>
> **What it says:** the encoder is "a stack of N = 6 identical layers". Each layer has two sub-layers: "multi-head self-attention" and a "position-wise fully connected feed-forward network". Each sub-layer has a "residual connection" and layer normalization, so its output is "LayerNorm(x + Sublayer(x))". All outputs have "dmodel = 512" numbers.
>
> **What it means:**
>
> "Identical" means the same design, not the same weights. Each of the six layers has its own learned numbers. A sentence goes into layer 1, and its output goes into layer 2, and so on up to layer 6.
>
> The formula $$\text{LayerNorm}(x + \text{Sublayer}(x))$$ is the "Add & Norm" box of Figure 1. The sub-layer does not replace its input. It computes a change, and the model adds that change to the input. Then a layer norm puts the numbers back into a standard range. We explain both steps below.
>
> The last sentence is a rule that makes the additions possible. You can add two vectors only if they have the same length. So every sub-layer, and the embedding too, must give vectors of exactly 512 numbers. The paper calls this number $$d_{\text{model}}$$.

> [!DEFINITION] Layer and sub-layer
> A **layer** is one copy of the grey box of Figure 1. A **sub-layer** is one block inside it, such as attention or the feed-forward network, together with its Add & Norm. An encoder layer has two sub-layers; a decoder layer has three.

<figure class="fig"><svg viewBox="0 0 680 460" role="img" aria-label="One encoder layer of the released English to German model: multi-head self-attention, add the input, layer norm, feed-forward 512 to 2048 to 512, add the input, layer norm. Every arrow carries 12 vectors of 512 numbers."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One encoder layer (the model has 6), for the 12 tokens of our sentence</text><rect class="box" x="210.0" y="48.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="71.0" text-anchor="middle">x: 12 × 512</text><rect class="box-2" x="210.0" y="108.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="131.0" text-anchor="middle">multi-head self-attention</text><line class="edge" x1="330.0" y1="84.0" x2="330.0" y2="106.0" marker-end="url(#ah)"/><rect class="box" x="210.0" y="168.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="191.0" text-anchor="middle">add: x + Sublayer(x)</text><line class="edge" x1="330.0" y1="144.0" x2="330.0" y2="166.0" marker-end="url(#ah)"/><rect class="box-1" x="210.0" y="228.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="251.0" text-anchor="middle">layer norm → 12 × 512</text><line class="edge" x1="330.0" y1="204.0" x2="330.0" y2="226.0" marker-end="url(#ah)"/><rect class="box-2" x="210.0" y="288.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="311.0" text-anchor="middle">feed-forward: 512 → 2048 → 512</text><line class="edge" x1="330.0" y1="264.0" x2="330.0" y2="286.0" marker-end="url(#ah)"/><rect class="box" x="210.0" y="348.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="371.0" text-anchor="middle">add: x + Sublayer(x)</text><line class="edge" x1="330.0" y1="324.0" x2="330.0" y2="346.0" marker-end="url(#ah)"/><rect class="box-1" x="210.0" y="408.0" width="240.0" height="36.0" rx="8"/><text class="t-note" x="330.0" y="431.0" text-anchor="middle">layer norm → 12 × 512</text><line class="edge" x1="330.0" y1="384.0" x2="330.0" y2="406.0" marker-end="url(#ah)"/><path class="edge" d="M210.0,66 H170.0 V186 H208.0" fill="none" marker-end="url(#ah)"/><text class="t-tick" x="162.0" y="130.0" text-anchor="end">residual</text><path class="edge" d="M210.0,246 H170.0 V366 H208.0" fill="none" marker-end="url(#ah)"/><text class="t-tick" x="162.0" y="310.0" text-anchor="end">residual</text><text class="t-tick" x="466.0" y="130.0" text-anchor="start">sub-layer 1: 1,050,624 weights</text><text class="t-tick" x="466.0" y="310.0" text-anchor="start">sub-layer 2: 2,099,712 weights</text><text class="t-tick" x="466.0" y="250.0" text-anchor="start">layer norm: 1,024 weights</text><text class="t-tick" x="466.0" y="430.0" text-anchor="start">layer norm: 1,024 weights</text></svg><figcaption>One encoder layer of our model, with the real shapes for our sentence of 12 tokens. The two curved arrows are the residual connections. The weight counts come from the released model.</figcaption></figure>

Our script follows the 12 tokens through the six encoder layers:

```text
  embedding lookup           (1, 12, 512)
  times sqrt(d_model) = 22.627, plus positions (12, 512)  -> (1, 12, 512)
  after encoder layer 1      (1, 12, 512)
  after encoder layer 2      (1, 12, 512)
  after encoder layer 3      (1, 12, 512)
  after encoder layer 4      (1, 12, 512)
  after encoder layer 5      (1, 12, 512)
  after encoder layer 6      (1, 12, 512)
  encoder output z           (1, 12, 512)  (one vector of 512 numbers per input token)
```

The first number, 1, is the batch: one sentence. The shape never changes. Twelve vectors of 512 numbers go in, and twelve vectors of 512 numbers come out of each layer. What changes is the content: at the end, each vector also carries information about the other words of the sentence. This final set is the $$z$$ of Section 3.

### The residual connection

Imagine you edit a document with "track changes" on. You do not write the document again. You only write your changes, and the editor adds them to the old text. If you have nothing useful to add, the document stays as it was.

A residual connection does the same for a vector. The sub-layer reads $$x$$ and computes a change. The output is the old vector plus the change:

$$
y = x + \text{Sublayer}(x)
$$

Here $$x$$ is the vector that enters, $$\text{Sublayer}(x)$$ is what attention or the feed-forward network computes from it, and $$y$$ is the result. For example, if $$x = (2, 0)$$ and the sub-layer returns $$(0.5, -1)$$, then $$y = (2.5, -1)$$.

Why is this useful? In a deep stack, a network without this path must pass all the information through every block. The path "around" the block keeps the original vector always available. In training, the error signal can also flow back along this direct path, so the lower layers keep learning. The paper takes the idea from ResNet, its reference [11].

Our model shows how much each sub-layer adds. The vectors that enter layer 1 have a length of 11 to 20. The attention outputs have a length of about 3 to 10:

```text
  x (layer input) norm per token: 19.5 18.9 19.6 19.7 14.0 18.3 19.8 20.2 20.3 18.5 15.0 11.4
  attention output norm per token: 9.7 3.7 3.7 9.3 9.3 3.2 3.9 10.4 6.5 2.9 10.1 7.2
  cosine(x, LayerNorm(x + attn)) per token: 0.67 0.71 0.76 0.71 0.62 0.74 0.77 0.72 0.81 0.74 0.66 0.68
```

The change is smaller than the input, so the output still points in a similar direction. The cosine of 0.62 to 0.81 between input and output measures this (1 would be "same direction"). The layer edits the vector; it does not rewrite it.

> [!DEFINITION] Norm (length) of a vector
> The square root of the sum of the squares of its numbers. The vector $$(3, 4)$$ has the norm $$\sqrt{9 + 16} = 5$$. **Cosine similarity** compares two vectors by direction only: 1 means the same direction, 0 means at right angles.

### Layer normalization

After each addition, the numbers of a vector can drift: one layer may make them large, another small. Layer normalization resets each vector to a standard scale, like a sound engineer who sets each track to the same volume before the mix.

For one vector $$v$$ of $$d$$ numbers:

$$
\text{LayerNorm}(v) = \gamma \odot \frac{v - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta,
\qquad \mu = \frac{1}{d}\sum_{j=1}^{d} v_j,
\qquad \sigma^2 = \frac{1}{d}\sum_{j=1}^{d} (v_j - \mu)^2
$$

Read it from left to right. $$\mu$$ is the mean of the $$d$$ numbers, and $$\sigma^2$$ is their variance (the mean squared distance from $$\mu$$). Subtract the mean and divide by the standard deviation $$\sigma$$: now the vector has mean 0 and standard deviation 1. The small $$\epsilon$$ avoids a division by zero. Then $$\gamma$$ and $$\beta$$, two learned vectors of $$d$$ numbers, let the model scale and move each number again where that helps. The sign $$\odot$$ means "multiply number by number".

A tiny example with $$d = 4$$ and $$v = (2, 0, -1, 3)$$. The mean is $$\mu = 4/4 = 1$$. The squared distances are $$1, 1, 4, 4$$, so $$\sigma^2 = 10/4 = 2.5$$ and $$\sigma = 1.581$$. The normalized vector is $$(0.632, -0.632, -1.265, 1.265)$$, with mean 0 and standard deviation 1.

The word "layer" in the name means that the statistics come from the numbers of one vector, for one token. They do not mix tokens or sentences. In our model, for the token "sat" after the first addition, the 512 numbers had mean 0.208 and standard deviation 0.836. We computed the layer norm by hand and compared it with the model's own layer:

```text
  token 2 (▁sat): x + attn has mean 0.208 and std 0.836
  max |layer norm by hand - library layer norm| = 4.77e-07
  max |our layer 1 output - library layer 1 output| = 0.00e+00
```

The second line is the full test. We rebuilt encoder layer 1 from the formula $$\text{LayerNorm}(x + \text{Sublayer}(x))$$, twice, with the model's own weights. Our output and the library's output agree exactly. This confirms that the released model uses the same order as the paper: first add, then normalize.

The order matters. Here, the layer norm sits after the addition, on the main path. Many later models move it before the sub-layer, to the form $$x + \text{Sublayer}(\text{LayerNorm}(x))$$, which is easier to train in very deep stacks. The 2017 paper uses the form after the addition.

> [!DEFINITION] Mean, variance and standard deviation
> The **mean** is the average. The **variance** is the average of the squared distances from the mean. The **standard deviation** is the square root of the variance: a typical distance from the mean.

## The decoder stack {§3.1}

> [!PAPER] · §3.1
> [![The Decoder paragraph of Section 3.1, with inserts a third sub-layer, over the output of the encoder stack, prevent positions from attending to subsequent positions and offset by one position highlighted](/img/papers/transformer/p2-decoder.png)](/img/papers/transformer/p2-decoder.png)
>
> **What it says:** the decoder is also "a stack of N = 6 identical layers". It "inserts a third sub-layer" that attends "over the output of the encoder stack". Its self-attention is changed "to prevent positions from attending to subsequent positions", and the output embeddings are "offset by one position".
>
> **What it means:**
>
> The decoder layer has the same two sub-layers as the encoder layer, plus one in the middle. That middle sub-layer is the bridge of Figure 1. Its questions come from the German side, and its answers come from the 12 English vectors of $$z$$. Our script counts its cost: a decoder layer has 1,051,648 more weights than an encoder layer, which is one more attention block.
>
> The two last phrases solve one problem. In training, the model sees the whole German sentence at once, to train all positions in parallel. But at position 5, it must not read word 5 or any later word. That would be cheating, and in real use those words do not exist yet. So the self-attention of the decoder hides all later positions. [Part 3](part-3-attention.md) shows this mask in detail.
>
> "Offset by one position" is the other half of the trick. The decoder input is the German sentence moved one step to the right, with a start token in front. So at each position, the model reads the words before it and predicts the word at that position.

Our model shows the shift. The decoder starts from a start token, which is `<pad>` in MarianMT, and the output ends with `</s>`:

```text
  output tokens: <pad> ▁Die ▁Katze ▁saß ▁auf ▁der ▁Matt e , ▁weil ▁sie ▁müde ▁war . </s>
```

From `<pad>`, the model predicted `▁Die`. From `<pad> ▁Die`, it predicted `▁Katze`, and so on. The word "Matte" became two tokens, `▁Matt` and `e`, because the vocabulary has no single token for it.

## The feed-forward network {§3.3}

> [!PAPER] · §3.3
> [![Section 3.3, Position-wise Feed-Forward Networks, with applied to each position separately and identically, two linear transformations with a ReLU activation, different parameters, two convolutions with kernel size 1 and d_ff = 2048 highlighted, and Equation 2](/img/papers/transformer/p2-ffn.png)](/img/papers/transformer/p2-ffn.png)
>
> **What it says:** each layer has a feed-forward network, "applied to each position separately and identically". It is "two linear transformations with a ReLU activation" in between: $$\text{FFN}(x) = \max(0, xW_1 + b_1)W_2 + b_2$$. Each layer has "different parameters". The inner size is "dff = 2048".
>
> **What it means:**
>
> After the meeting (attention), each token thinks alone. The same small network runs on each of the 12 vectors, one at a time. "Identically" means that all positions use the same weights, so the word in position 1 and the word in position 9 get the same treatment.
>
> The network makes each vector wider and then narrow again: 512 numbers go to 2048, and then back to 512. The wide middle gives the model room to compute many features of each token. The ReLU in the middle sets the negative numbers to zero, so the network is not one big linear function.
>
> "Two convolutions with kernel size 1" is the same idea in the words of image models. A convolution of size 1 looks at one position at a time. It is a second name for "the same small network at each position".

> [!DEFINITION] Linear transformation
> A multiplication by a matrix of learned weights, plus a learned vector called the **bias**: $$xW + b$$. It turns a vector of one size into a vector of another size. Each number of the output is a weighted sum of all the numbers of the input.

> [!DEFINITION] ReLU
> The function $$\max(0, x)$$: it keeps a positive number and changes a negative number to 0. Without a function like this between two linear transformations, the two would collapse into one linear transformation.

Equation 2 reads from the inside out. Multiply the vector $$x$$ by $$W_1$$ (512 × 2048) and add $$b_1$$: now there are 2048 numbers. Apply the ReLU: each negative number becomes 0. Multiply by $$W_2$$ (2048 × 512) and add $$b_2$$: we are back to 512 numbers.

$$
\text{FFN}(x) = \max(0,\ xW_1 + b_1)\,W_2 + b_2
$$

A tiny example with 2 numbers in and 3 in the middle. Let $$x = (1, 2)$$ and

$$
W_1 = \begin{pmatrix} 1 & 0 & -1 \\ 0 & 1 & 1 \end{pmatrix},\quad b_1 = (0, -3, 0),\quad
W_2 = \begin{pmatrix} 1 & 0 \\ 0 & 1 \\ 1 & 1 \end{pmatrix},\quad b_2 = (0, 0)
$$

Then $$xW_1 = (1, 2, 1)$$, and with $$b_1$$ we get $$(1, -1, 1)$$. The ReLU changes the $$-1$$ to $$0$$: $$(1, 0, 1)$$. Multiply by $$W_2$$: $$1 \times (1, 0) + 0 \times (0, 1) + 1 \times (1, 1) = (2, 1)$$. The middle unit with a negative value had no effect on the output. This on and off switching is what makes the network more than a linear map.

Our script checks the shapes and the "separately" claim in the released model:

```text
  W1 (512, 2048)  b1 (2048,)  W2 (2048, 512)  b2 (512,)
  one token: 512 -> 2048 -> 512
  max |each token alone - whole sentence at once| = 5.72e-05
  hidden units near zero (|h| < 0.001) for this sentence: 0.2% (Marian uses swish, the paper uses ReLU)
```

We ran the network on each token alone and on all 12 tokens at once. The results agree to about 0.00006, the size of rounding differences in 32-bit numbers. So the feed-forward network never mixes tokens: only attention does that.

The last line shows one difference from the paper. MarianMT uses the **swish** function, $$x \cdot \text{sigmoid}(x)$$, in place of ReLU. Swish is close to 0 for negative inputs but not exactly 0. So only 0.2% of the hidden numbers are near zero here; with ReLU, many would be exactly zero.

The feed-forward network is small in idea but large in weights. Our script counts them:

```text
  encoder layer: attention 1,050,624  feed-forward 2,099,712  layer norms 2,048  total 3,152,384
  feed-forward share of an encoder layer: 66.6%
```

Check the feed-forward count by hand: $$512 \times 2048 + 2048 + 2048 \times 512 + 512 = 2{,}099{,}712$$. Attention has four matrices of 512 × 512 plus their biases: $$4 \times 262{,}656 = 1{,}050{,}624$$. So two thirds of each encoder layer is the feed-forward network, not attention.

## Embeddings and the shared matrix {§3.4}

> [!PAPER] · §3.4
> [![Section 3.4, Embeddings and Softmax, with learned embeddings, predicted next-token probabilities, share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, and multiply those weights by the square root of d_model highlighted](/img/papers/transformer/p2-embeddings.png)](/img/papers/transformer/p2-embeddings.png)
>
> **What it says:** the model uses "learned embeddings" to turn tokens into vectors of $$d_{\text{model}}$$ numbers, and a linear layer and softmax to give "predicted next-token probabilities". It shares "the same weight matrix between the two embedding layers and the pre-softmax linear transformation". In the embeddings, it multiplies "those weights by √dmodel".
>
> **What it means:**
>
> An embedding is a table with one row for each token of the vocabulary. Each row is a vector of 512 learned numbers. To read the token with id 12064 ("cat"), the model takes row 12064. Training moves the rows so that tokens with similar uses get similar vectors.
>
> At the top of the decoder, the model must do the reverse. It has a vector of 512 numbers and must give a score to each token of the vocabulary. The paper uses the same table for this. The score of a token is the dot product of the decoder vector with that token's row. So a vector that is close to the row of "Katze" gives a high score to "Katze".
>
> One table does three jobs: the encoder input, the decoder input and the output scores. This needs a vocabulary that both languages share, which the paper builds in Section 5. The saving is large, as our numbers show.

> [!DEFINITION] Embedding
> A learned table that maps each token id to a vector. The table has one row per token in the vocabulary and $$d_{\text{model}}$$ columns.

> [!DEFINITION] Softmax
> A function that turns a list of scores into probabilities. It takes $$e$$ to the power of each score and divides each result by their sum. The outputs are positive and add up to 1, and the largest score gets the largest probability.

Our model shares the table in the same way:

```text
  shared embedding matrix   58,101 x 512 = 29,747,712
  encoder embedding is the shared matrix: True; decoder embedding: True; output layer: True
  without sharing: 3 x 29,747,712 = 89,243,136 numbers; with sharing: 29,747,712
```

The script checked that the three layers point to the same memory, not to three equal copies. Sharing saves about 59.5 million numbers. The whole model has 74,410,496 weights, so the shared table alone is 40% of it. For comparison, Table 3 of the paper gives 65 million weights for its base model. Its vocabulary of about 37,000 tokens is smaller than Marian's 58,101.

### Why multiply by $$\sqrt{d_{\text{model}}}$$?

The paper gives no reason for the scale, so here is our reading. The embedding and the position vector are added. If one of them is much larger than the other, the sum is mostly that one.

In our model, the rows of the table are small, and the position vectors are not:

```text
  embedding row norm: median 1.08  (min 0.00, max 2.21)
  times sqrt(512) = 22.63: median 24.5
  position vector norm: 16.00 at pos 0, 16.00 at pos 10, sqrt(256) = 16
```

A row has a typical length of 1.08. A position vector always has the length 16, because it holds 256 pairs of $$\sin$$ and $$\cos$$, and each pair gives $$\sin^2 + \cos^2 = 1$$. Without the scale, the word would be a whisper next to its position. With $$\sqrt{512} = 22.63$$, the word has a typical length of 24.5 and the position 16. The word now leads, and the position is still clearly present. The row with length 0.00 is the padding token, which the model never needs to read.

The shared table explains why the rows are small. The same rows also make the output scores, and large rows would give extreme scores. So the table stays small, and the scale is applied only where the table is used as an embedding.

## Positional encoding {§3.5}

> [!PAPER] · §3.5
> [![The first paragraph of Section 3.5, Positional Encoding, with no recurrence and no convolution, relative or absolute position, add positional encodings and same dimension d_model highlighted](/img/papers/transformer/p2-pe-1.png)](/img/papers/transformer/p2-pe-1.png)
>
> **What it says:** because the model has "no recurrence and no convolution", it must get information about the "relative or absolute position" of the tokens. So the paper adds "positional encodings" to the input embeddings. They have the "same dimension dmodel", so that the two can be summed.
>
> **What it means:**
>
> Attention compares every token with every token, but it does not know which token came first. To attention, a sentence is a bag of words. "Dog bites man" and "man bites dog" would look the same. A recurrent model knows the order because it reads word after word. A convolution knows it because it looks at neighbours. The Transformer has neither, so it must get the order from outside.
>
> The solution is a label of position. For each position (0, 1, 2 and so on), the model has a vector of 512 numbers. It adds this vector to the embedding of the token in that position. The token "cat" in position 1 then has a different input vector than "cat" in position 7.

Our script tests the bag-of-words claim. It takes encoder layer 1 and swaps the first and fourth tokens ("The" and "on"). Without positions, the layer gives exactly the same vectors, only in the swapped order. With positions, the outputs change:

```text
  swap tokens 0 and 3 (▁The <-> ▁on), no positions: outputs are the same vectors, swapped; max difference 1.4e-06
  same swap with positions added: max difference 2.67
```

The first line is the problem in numbers: without positions, the layer cannot see that the order changed. The difference of 0.0000014 is rounding only.

> [!PAPER] · §3.5
> [![The sine and cosine formulas of Section 3.5 and the paragraph under them, with geometric progression from 2π to 10000 · 2π, attend by relative positions and PE pos+k can be represented as a linear function highlighted](/img/papers/transformer/p2-pe-2.png)](/img/papers/transformer/p2-pe-2.png)
>
> **What it says:** the encoding uses "sine and cosine functions of different frequencies": $$PE_{(pos,2i)} = \sin(pos/10000^{2i/d_{\text{model}}})$$ and $$PE_{(pos,2i+1)} = \cos(pos/10000^{2i/d_{\text{model}}})$$. The wavelengths form "a geometric progression from 2π to 10000 · 2π". The reason: "for any fixed offset k, PEpos+k can be represented as a linear function of PEpos".
>
> **What it means:**
>
> Think of a clock with many hands. The second hand turns fast, the minute hand slower, the hour hand slower still. From the positions of all hands you can read the time exactly. The position encoding is a clock with 256 hands. Each hand is one pair of numbers, a sine and a cosine, and each turns at its own speed as the position grows.
>
> The fastest hand makes a full turn every $$2\pi = 6.28$$ positions, so it tells near neighbours apart. The slowest needs about $$10000 \cdot 2\pi$$ positions, so over a sentence it moves slowly and tells the beginning from the end. Each hand is a fixed factor slower than the one before: that is a "geometric progression".
>
> The last sentence is the deep reason. To move a clock forward by 5 minutes, you turn each hand by a fixed angle, the same for any start time. In the same way, $$PE_{pos+k}$$ is $$PE_{pos}$$ turned by fixed angles that depend only on $$k$$. So "the word 3 places back" looks the same everywhere in the sentence, and a linear layer can learn to find it.

The formula, read from left to right:

$$
PE_{(pos,\,2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad
PE_{(pos,\,2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

- $$pos$$ is the position of the token: 0, 1, 2 and so on.
- $$i$$ numbers the pairs, from 0 to 255. Pair $$i$$ fills dimensions $$2i$$ (the sine) and $$2i + 1$$ (the cosine).
- $$10000^{2i/d_{\text{model}}}$$ sets the speed of pair $$i$$. For $$i = 0$$ it is 1, so the angle is the position itself, in radians. For larger $$i$$ it grows, and the hand turns more slowly.

A tiny example: position 1, pair 0. The angle is $$1 / 10000^{0} = 1$$ radian, so $$PE_{(1,0)} = \sin 1 = 0.8415$$ and $$PE_{(1,1)} = \cos 1 = 0.5403$$. For pair 1, the angle is $$1 / 10000^{2/512} = 0.9647$$, so $$PE_{(1,2)} = \sin 0.9647 = 0.8219$$. Our script computes the same table:

```text
  pos  dim0 sin   dim1 cos   dim2 sin   dim3 cos  ...  dim510 sin  dim511 cos
    0    0.0000    1.0000    0.0000    1.0000  ...   0.000000   1.000000
    1    0.8415    0.5403    0.8219    0.5697  ...   0.000104   1.000000
    2    0.9093   -0.4161    0.9364   -0.3509  ...   0.000207   1.000000
    3    0.1411   -0.9900    0.2451   -0.9695  ...   0.000311   1.000000
   10   -0.5440   -0.8391   -0.2200   -0.9755  ...   0.001037   0.999999
   50   -0.2624    0.9650   -0.8953   -0.4454  ...   0.005183   0.999987
```

Read the columns. The first two dimensions change a lot from one position to the next. The last two barely move: even at position 50, dimension 511 is still 0.99999. Position 0 is always $$(0, 1, 0, 1, \ldots)$$, because $$\sin 0 = 0$$ and $$\cos 0 = 1$$.

<figure class="fig"><svg viewBox="0 0 780 300" role="img" aria-label="Three dimensions of the sinusoidal position encoding for positions 0 to 63. Dimension 0 turns once every 6.3 positions, dimension 64 every 19.9 positions and dimension 128 every 62.8 positions."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="530" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">-1</text><line class="grid" x1="64" y1="195.0" x2="530" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">-0.5</text><line class="grid" x1="64" y1="140.0" x2="530" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">0</text><line class="grid" x1="64" y1="85.0" x2="530" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">0.5</text><line class="grid" x1="64" y1="30.0" x2="530" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">0</text><text class="t-tick" x="138.0" y="270.0" text-anchor="middle">10</text><text class="t-tick" x="211.9" y="270.0" text-anchor="middle">20</text><text class="t-tick" x="285.9" y="270.0" text-anchor="middle">30</text><text class="t-tick" x="359.9" y="270.0" text-anchor="middle">40</text><text class="t-tick" x="433.8" y="270.0" text-anchor="middle">50</text><text class="t-tick" x="530.0" y="270.0" text-anchor="middle">63</text><line class="axis" x1="64" y1="250" x2="530" y2="250"/><polyline class="l1" points="64.0,140.0 71.4,47.4 78.8,40.0 86.2,124.5 93.6,223.2 101.0,245.5 108.4,170.7 115.8,67.7 123.2,31.2 130.6,94.7 138.0,199.8 145.4,250.0 152.8,199.0 160.2,93.8 167.6,31.0 175.0,68.5 182.3,171.7 189.7,245.8 197.1,222.6 204.5,123.5 211.9,39.6 219.3,48.0 226.7,141.0 234.1,233.1 241.5,239.6 248.9,154.6 256.3,56.1 263.7,34.8 271.1,110.2 278.5,213.0 285.9,248.7 293.3,184.4 300.7,79.3 308.1,30.0 315.5,81.8 322.9,187.1 330.3,249.1 337.7,210.8 345.1,107.4 352.5,34.0 359.9,58.0 367.3,157.4 374.7,240.8 382.1,231.5 389.5,138.1 396.9,46.4 404.3,40.8 411.7,126.4 419.0,224.5 426.4,244.9 433.8,168.9 441.2,66.3 448.6,31.5 456.0,96.5 463.4,201.5 470.8,250.0 478.2,197.4 485.6,92.0 493.0,30.8 500.4,70.0 507.8,173.5 515.2,246.3 522.6,221.3 530.0,121.6"/><g class="mark"><title>dim 0, turn 6.3, 0: 0</title><circle class="s1 ring" cx="64.0" cy="140.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 1: 0.8415</title><circle class="s1 ring" cx="71.4" cy="47.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 2: 0.9093</title><circle class="s1 ring" cx="78.8" cy="40.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 3: 0.1411</title><circle class="s1 ring" cx="86.2" cy="124.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 4: -0.7568</title><circle class="s1 ring" cx="93.6" cy="223.2" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 5: -0.9589</title><circle class="s1 ring" cx="101.0" cy="245.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 6: -0.2794</title><circle class="s1 ring" cx="108.4" cy="170.7" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 7: 0.657</title><circle class="s1 ring" cx="115.8" cy="67.7" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 8: 0.9894</title><circle class="s1 ring" cx="123.2" cy="31.2" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 9: 0.4121</title><circle class="s1 ring" cx="130.6" cy="94.7" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 10: -0.544</title><circle class="s1 ring" cx="138.0" cy="199.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 11: -1</title><circle class="s1 ring" cx="145.4" cy="250.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 12: -0.5366</title><circle class="s1 ring" cx="152.8" cy="199.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 13: 0.4202</title><circle class="s1 ring" cx="160.2" cy="93.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 14: 0.9906</title><circle class="s1 ring" cx="167.6" cy="31.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 15: 0.6503</title><circle class="s1 ring" cx="175.0" cy="68.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 16: -0.2879</title><circle class="s1 ring" cx="182.3" cy="171.7" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 17: -0.9614</title><circle class="s1 ring" cx="189.7" cy="245.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 18: -0.751</title><circle class="s1 ring" cx="197.1" cy="222.6" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 19: 0.1499</title><circle class="s1 ring" cx="204.5" cy="123.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 20: 0.9129</title><circle class="s1 ring" cx="211.9" cy="39.6" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 21: 0.8367</title><circle class="s1 ring" cx="219.3" cy="48.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 22: -0.0089</title><circle class="s1 ring" cx="226.7" cy="141.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 23: -0.8462</title><circle class="s1 ring" cx="234.1" cy="233.1" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 24: -0.9056</title><circle class="s1 ring" cx="241.5" cy="239.6" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 25: -0.1324</title><circle class="s1 ring" cx="248.9" cy="154.6" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 26: 0.7626</title><circle class="s1 ring" cx="256.3" cy="56.1" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 27: 0.9564</title><circle class="s1 ring" cx="263.7" cy="34.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 28: 0.2709</title><circle class="s1 ring" cx="271.1" cy="110.2" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 29: -0.6636</title><circle class="s1 ring" cx="278.5" cy="213.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 30: -0.988</title><circle class="s1 ring" cx="285.9" cy="248.7" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 31: -0.404</title><circle class="s1 ring" cx="293.3" cy="184.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 32: 0.5514</title><circle class="s1 ring" cx="300.7" cy="79.3" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 33: 0.9999</title><circle class="s1 ring" cx="308.1" cy="30.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 34: 0.5291</title><circle class="s1 ring" cx="315.5" cy="81.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 35: -0.4282</title><circle class="s1 ring" cx="322.9" cy="187.1" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 36: -0.9918</title><circle class="s1 ring" cx="330.3" cy="249.1" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 37: -0.6435</title><circle class="s1 ring" cx="337.7" cy="210.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 38: 0.2964</title><circle class="s1 ring" cx="345.1" cy="107.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 39: 0.9638</title><circle class="s1 ring" cx="352.5" cy="34.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 40: 0.7451</title><circle class="s1 ring" cx="359.9" cy="58.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 41: -0.1586</title><circle class="s1 ring" cx="367.3" cy="157.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 42: -0.9165</title><circle class="s1 ring" cx="374.7" cy="240.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 43: -0.8318</title><circle class="s1 ring" cx="382.1" cy="231.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 44: 0.0177</title><circle class="s1 ring" cx="389.5" cy="138.1" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 45: 0.8509</title><circle class="s1 ring" cx="396.9" cy="46.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 46: 0.9018</title><circle class="s1 ring" cx="404.3" cy="40.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 47: 0.1236</title><circle class="s1 ring" cx="411.7" cy="126.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 48: -0.7683</title><circle class="s1 ring" cx="419.0" cy="224.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 49: -0.9538</title><circle class="s1 ring" cx="426.4" cy="244.9" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 50: -0.2624</title><circle class="s1 ring" cx="433.8" cy="168.9" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 51: 0.6702</title><circle class="s1 ring" cx="441.2" cy="66.3" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 52: 0.9866</title><circle class="s1 ring" cx="448.6" cy="31.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 53: 0.3959</title><circle class="s1 ring" cx="456.0" cy="96.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 54: -0.5588</title><circle class="s1 ring" cx="463.4" cy="201.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 55: -0.9998</title><circle class="s1 ring" cx="470.8" cy="250.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 56: -0.5216</title><circle class="s1 ring" cx="478.2" cy="197.4" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 57: 0.4362</title><circle class="s1 ring" cx="485.6" cy="92.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 58: 0.9929</title><circle class="s1 ring" cx="493.0" cy="30.8" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 59: 0.6367</title><circle class="s1 ring" cx="500.4" cy="70.0" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 60: -0.3048</title><circle class="s1 ring" cx="507.8" cy="173.5" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 61: -0.9661</title><circle class="s1 ring" cx="515.2" cy="246.3" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 62: -0.7392</title><circle class="s1 ring" cx="522.6" cy="221.3" r="5"/></g><g class="mark"><title>dim 0, turn 6.3, 63: 0.1674</title><circle class="s1 ring" cx="530.0" cy="121.6" r="5"/></g><polyline class="l2" points="64.0,140.0 71.4,105.8 78.8,75.0 86.2,50.6 93.6,35.1 101.0,30.0 108.4,35.8 115.8,52.0 123.2,76.8 130.6,108.0 138.0,142.3 145.4,176.4 152.8,206.8 160.2,230.7 167.6,245.6 175.0,249.9 182.3,243.4 189.7,226.7 197.1,201.3 204.5,169.9 211.9,135.4 219.3,101.5 226.7,71.4 234.1,48.0 241.5,33.8 248.9,30.1 256.3,37.4 263.7,54.8 271.1,80.6 278.5,112.3 285.9,146.8 293.3,180.6 300.7,210.4 308.1,233.2 315.5,246.7 322.9,249.7 330.3,241.8 337.7,223.8 345.1,197.5 352.5,165.5 359.9,130.9 367.3,97.3 374.7,67.9 382.1,45.6 389.5,32.7 396.9,30.5 404.3,39.1 411.7,57.7 419.0,84.5 426.4,116.8 433.8,151.4 441.2,184.8 448.6,213.8 456.0,235.5 463.4,247.8 470.8,249.3 478.2,240.0 485.6,220.8 493.0,193.5 500.4,161.0 507.8,126.4 515.2,93.1 522.6,64.5 530.0,43.4"/><g class="mark"><title>dim 64, turn 19.9, 0: 0</title><circle class="s2 ring" cx="64.0" cy="140.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 1: 0.311</title><circle class="s2 ring" cx="71.4" cy="105.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 2: 0.5911</title><circle class="s2 ring" cx="78.8" cy="75.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 3: 0.8126</title><circle class="s2 ring" cx="86.2" cy="50.6" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 4: 0.9536</title><circle class="s2 ring" cx="93.6" cy="35.1" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 5: 0.9999</title><circle class="s2 ring" cx="101.0" cy="30.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 6: 0.9471</title><circle class="s2 ring" cx="108.4" cy="35.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 7: 0.8004</title><circle class="s2 ring" cx="115.8" cy="52.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 8: 0.5743</title><circle class="s2 ring" cx="123.2" cy="76.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 9: 0.2913</title><circle class="s2 ring" cx="130.6" cy="108.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 10: -0.0207</title><circle class="s2 ring" cx="138.0" cy="142.3" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 11: -0.3306</title><circle class="s2 ring" cx="145.4" cy="176.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 12: -0.6077</title><circle class="s2 ring" cx="152.8" cy="206.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 13: -0.8245</title><circle class="s2 ring" cx="160.2" cy="230.7" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 14: -0.9596</title><circle class="s2 ring" cx="167.6" cy="245.6" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 15: -0.9995</title><circle class="s2 ring" cx="175.0" cy="249.9" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 16: -0.9403</title><circle class="s2 ring" cx="182.3" cy="243.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 17: -0.7879</title><circle class="s2 ring" cx="189.7" cy="226.7" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 18: -0.5573</title><circle class="s2 ring" cx="197.1" cy="201.3" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 19: -0.2714</title><circle class="s2 ring" cx="204.5" cy="169.9" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 20: 0.0414</title><circle class="s2 ring" cx="211.9" cy="135.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 21: 0.35</title><circle class="s2 ring" cx="219.3" cy="101.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 22: 0.624</title><circle class="s2 ring" cx="226.7" cy="71.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 23: 0.8361</title><circle class="s2 ring" cx="234.1" cy="48.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 24: 0.9652</title><circle class="s2 ring" cx="241.5" cy="33.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 25: 0.9987</title><circle class="s2 ring" cx="248.9" cy="30.1" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 26: 0.9331</title><circle class="s2 ring" cx="256.3" cy="37.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 27: 0.7749</title><circle class="s2 ring" cx="263.7" cy="54.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 28: 0.54</title><circle class="s2 ring" cx="271.1" cy="80.6" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 29: 0.2514</title><circle class="s2 ring" cx="278.5" cy="112.3" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 30: -0.062</title><circle class="s2 ring" cx="285.9" cy="146.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 31: -0.3693</title><circle class="s2 ring" cx="293.3" cy="180.6" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 32: -0.64</title><circle class="s2 ring" cx="300.7" cy="210.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 33: -0.8472</title><circle class="s2 ring" cx="308.1" cy="233.2" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 34: -0.9704</title><circle class="s2 ring" cx="315.5" cy="246.7" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 35: -0.9974</title><circle class="s2 ring" cx="322.9" cy="249.7" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 36: -0.9254</title><circle class="s2 ring" cx="330.3" cy="241.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 37: -0.7617</title><circle class="s2 ring" cx="337.7" cy="223.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 38: -0.5224</title><circle class="s2 ring" cx="345.1" cy="197.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 39: -0.2314</title><circle class="s2 ring" cx="352.5" cy="165.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 40: 0.0826</title><circle class="s2 ring" cx="359.9" cy="130.9" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 41: 0.3885</title><circle class="s2 ring" cx="367.3" cy="97.3" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 42: 0.6558</title><circle class="s2 ring" cx="374.7" cy="67.9" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 43: 0.858</title><circle class="s2 ring" cx="382.1" cy="45.6" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 44: 0.9752</title><circle class="s2 ring" cx="389.5" cy="32.7" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 45: 0.9957</title><circle class="s2 ring" cx="396.9" cy="30.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 46: 0.9174</title><circle class="s2 ring" cx="404.3" cy="39.1" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 47: 0.7481</title><circle class="s2 ring" cx="411.7" cy="57.7" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 48: 0.5047</title><circle class="s2 ring" cx="419.0" cy="84.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 49: 0.2112</title><circle class="s2 ring" cx="426.4" cy="116.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 50: -0.1032</title><circle class="s2 ring" cx="433.8" cy="151.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 51: -0.4074</title><circle class="s2 ring" cx="441.2" cy="184.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 52: -0.6712</title><circle class="s2 ring" cx="448.6" cy="213.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 53: -0.8685</title><circle class="s2 ring" cx="456.0" cy="235.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 54: -0.9796</title><circle class="s2 ring" cx="463.4" cy="247.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 55: -0.9935</title><circle class="s2 ring" cx="470.8" cy="249.3" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 56: -0.909</title><circle class="s2 ring" cx="478.2" cy="240.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 57: -0.7343</title><circle class="s2 ring" cx="485.6" cy="220.8" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 58: -0.4867</title><circle class="s2 ring" cx="493.0" cy="193.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 59: -0.1909</title><circle class="s2 ring" cx="500.4" cy="161.0" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 60: 0.1238</title><circle class="s2 ring" cx="507.8" cy="126.4" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 61: 0.4262</title><circle class="s2 ring" cx="515.2" cy="93.1" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 62: 0.6864</title><circle class="s2 ring" cx="522.6" cy="64.5" r="5"/></g><g class="mark"><title>dim 64, turn 19.9, 63: 0.8785</title><circle class="s2 ring" cx="530.0" cy="43.4" r="5"/></g><polyline class="l3" points="64.0,140.0 71.4,129.0 78.8,118.1 86.2,107.5 93.6,97.2 101.0,87.3 108.4,77.9 115.8,69.1 123.2,61.1 130.6,53.8 138.0,47.4 145.4,42.0 152.8,37.5 160.2,34.0 167.6,31.6 175.0,30.3 182.3,30.0 189.7,30.9 197.1,32.9 204.5,35.9 211.9,40.0 219.3,45.0 226.7,51.1 234.1,58.0 241.5,65.7 248.9,74.2 256.3,83.3 263.7,93.0 271.1,103.2 278.5,113.7 285.9,124.5 293.3,135.4 300.7,146.4 308.1,157.3 315.5,168.1 322.9,178.6 330.3,188.7 337.7,198.3 345.1,207.3 352.5,215.7 359.9,223.2 367.3,230.0 374.7,235.9 382.1,240.8 389.5,244.7 396.9,247.5 404.3,249.3 411.7,250.0 419.0,249.6 426.4,248.1 433.8,245.5 441.2,241.8 448.6,237.2 456.0,231.6 463.4,225.0 470.8,217.6 478.2,209.4 485.6,200.6 493.0,191.1 500.4,181.1 507.8,170.7 515.2,160.0 522.6,149.1 530.0,138.2"/><g class="mark"><title>dim 128, turn 62.8, 0: 0</title><circle class="s3 ring" cx="64.0" cy="140.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 1: 0.0998</title><circle class="s3 ring" cx="71.4" cy="129.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 2: 0.1987</title><circle class="s3 ring" cx="78.8" cy="118.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 3: 0.2955</title><circle class="s3 ring" cx="86.2" cy="107.5" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 4: 0.3894</title><circle class="s3 ring" cx="93.6" cy="97.2" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 5: 0.4794</title><circle class="s3 ring" cx="101.0" cy="87.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 6: 0.5646</title><circle class="s3 ring" cx="108.4" cy="77.9" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 7: 0.6442</title><circle class="s3 ring" cx="115.8" cy="69.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 8: 0.7174</title><circle class="s3 ring" cx="123.2" cy="61.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 9: 0.7833</title><circle class="s3 ring" cx="130.6" cy="53.8" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 10: 0.8415</title><circle class="s3 ring" cx="138.0" cy="47.4" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 11: 0.8912</title><circle class="s3 ring" cx="145.4" cy="42.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 12: 0.932</title><circle class="s3 ring" cx="152.8" cy="37.5" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 13: 0.9636</title><circle class="s3 ring" cx="160.2" cy="34.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 14: 0.9854</title><circle class="s3 ring" cx="167.6" cy="31.6" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 15: 0.9975</title><circle class="s3 ring" cx="175.0" cy="30.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 16: 0.9996</title><circle class="s3 ring" cx="182.3" cy="30.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 17: 0.9917</title><circle class="s3 ring" cx="189.7" cy="30.9" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 18: 0.9738</title><circle class="s3 ring" cx="197.1" cy="32.9" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 19: 0.9463</title><circle class="s3 ring" cx="204.5" cy="35.9" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 20: 0.9093</title><circle class="s3 ring" cx="211.9" cy="40.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 21: 0.8632</title><circle class="s3 ring" cx="219.3" cy="45.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 22: 0.8085</title><circle class="s3 ring" cx="226.7" cy="51.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 23: 0.7457</title><circle class="s3 ring" cx="234.1" cy="58.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 24: 0.6755</title><circle class="s3 ring" cx="241.5" cy="65.7" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 25: 0.5985</title><circle class="s3 ring" cx="248.9" cy="74.2" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 26: 0.5155</title><circle class="s3 ring" cx="256.3" cy="83.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 27: 0.4274</title><circle class="s3 ring" cx="263.7" cy="93.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 28: 0.335</title><circle class="s3 ring" cx="271.1" cy="103.2" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 29: 0.2392</title><circle class="s3 ring" cx="278.5" cy="113.7" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 30: 0.1411</title><circle class="s3 ring" cx="285.9" cy="124.5" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 31: 0.0416</title><circle class="s3 ring" cx="293.3" cy="135.4" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 32: -0.0584</title><circle class="s3 ring" cx="300.7" cy="146.4" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 33: -0.1577</title><circle class="s3 ring" cx="308.1" cy="157.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 34: -0.2555</title><circle class="s3 ring" cx="315.5" cy="168.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 35: -0.3508</title><circle class="s3 ring" cx="322.9" cy="178.6" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 36: -0.4425</title><circle class="s3 ring" cx="330.3" cy="188.7" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 37: -0.5298</title><circle class="s3 ring" cx="337.7" cy="198.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 38: -0.6119</title><circle class="s3 ring" cx="345.1" cy="207.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 39: -0.6878</title><circle class="s3 ring" cx="352.5" cy="215.7" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 40: -0.7568</title><circle class="s3 ring" cx="359.9" cy="223.2" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 41: -0.8183</title><circle class="s3 ring" cx="367.3" cy="230.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 42: -0.8716</title><circle class="s3 ring" cx="374.7" cy="235.9" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 43: -0.9162</title><circle class="s3 ring" cx="382.1" cy="240.8" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 44: -0.9516</title><circle class="s3 ring" cx="389.5" cy="244.7" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 45: -0.9775</title><circle class="s3 ring" cx="396.9" cy="247.5" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 46: -0.9937</title><circle class="s3 ring" cx="404.3" cy="249.3" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 47: -0.9999</title><circle class="s3 ring" cx="411.7" cy="250.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 48: -0.9962</title><circle class="s3 ring" cx="419.0" cy="249.6" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 49: -0.9825</title><circle class="s3 ring" cx="426.4" cy="248.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 50: -0.9589</title><circle class="s3 ring" cx="433.8" cy="245.5" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 51: -0.9258</title><circle class="s3 ring" cx="441.2" cy="241.8" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 52: -0.8835</title><circle class="s3 ring" cx="448.6" cy="237.2" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 53: -0.8323</title><circle class="s3 ring" cx="456.0" cy="231.6" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 54: -0.7728</title><circle class="s3 ring" cx="463.4" cy="225.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 55: -0.7055</title><circle class="s3 ring" cx="470.8" cy="217.6" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 56: -0.6313</title><circle class="s3 ring" cx="478.2" cy="209.4" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 57: -0.5507</title><circle class="s3 ring" cx="485.6" cy="200.6" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 58: -0.4646</title><circle class="s3 ring" cx="493.0" cy="191.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 59: -0.3739</title><circle class="s3 ring" cx="500.4" cy="181.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 60: -0.2794</title><circle class="s3 ring" cx="507.8" cy="170.7" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 61: -0.1822</title><circle class="s3 ring" cx="515.2" cy="160.0" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 62: -0.0831</title><circle class="s3 ring" cx="522.6" cy="149.1" r="5"/></g><g class="mark"><title>dim 128, turn 62.8, 63: 0.0168</title><circle class="s3 ring" cx="530.0" cy="138.2" r="5"/></g><line class="grid" x1="536.0" y1="43.4" x2="546.0" y2="43.4"/><rect class="s2" x="550.0" y="38.4" width="10" height="10" rx="2"/><text class="t-note" x="566.0" y="47.4" text-anchor="start">dim 64, turn 19.9: 0.8785</text><line class="grid" x1="536.0" y1="121.6" x2="546.0" y2="121.6"/><rect class="s1" x="550.0" y="116.6" width="10" height="10" rx="2"/><text class="t-note" x="566.0" y="125.6" text-anchor="start">dim 0, turn 6.3: 0.1674</text><line class="grid" x1="536.0" y1="138.2" x2="546.0" y2="138.6"/><rect class="s3" x="550.0" y="133.6" width="10" height="10" rx="2"/><text class="t-note" x="566.0" y="142.6" text-anchor="start">dim 128, turn 62.8: 0.0168</text><text class="t-tick" x="297.0" y="290.0" text-anchor="middle">position</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">value</text></svg><figcaption>Three dimensions of the position encoding over positions 0 to 63. Each is a sine wave with its own wavelength: 6.3, 19.9 and 62.8 positions. Low dimensions turn fast; high dimensions turn slowly.</figcaption></figure>

The wavelength of pair $$i$$ is $$2\pi \cdot 10000^{2i/d_{\text{model}}}$$:

```text
  wavelengths (positions per full turn) for i = 0, 1, 2, 32, 64, 128, 192, 255: 6.3, 6.5, 6.8, 19.9, 62.8, 628.3, 6,283.2, 60,611.5
  shortest 2*pi = 6.28; longest limit 10000*2*pi = 62,832
```

Every 64 pairs, the wavelength grows by a factor of 10: 6.3, 62.8, 628.3, 6,283.2. The last pair, $$i = 255$$, has 60,611.5, just under the limit of 62,832 that the paper names.

The released model stores the same numbers in a different order: first all 256 sines, then all 256 cosines. Our script put the paper's table in that order and compared:

```text
  max |paper formula (reordered) - model table| = 2.98e-08
```

The order of the dimensions does not matter to the model, because the first linear layer can learn any order. The values are the paper's formula.

### Why a shift is a rotation

Take one pair with the speed $$\omega = 1/10000^{2i/d_{\text{model}}}$$. At position $$pos$$ it holds $$(\sin \omega\, pos,\ \cos \omega\, pos)$$. The angle-sum rules of trigonometry give the pair at position $$pos + k$$:

$$
\begin{pmatrix} \sin \omega(pos+k) \\ \cos \omega(pos+k) \end{pmatrix}
=
\begin{pmatrix} \cos \omega k & \sin \omega k \\ -\sin \omega k & \cos \omega k \end{pmatrix}
\begin{pmatrix} \sin \omega\, pos \\ \cos \omega\, pos \end{pmatrix}
$$

The matrix in the middle depends on $$k$$ and $$\omega$$ but not on $$pos$$. It is a rotation by the angle $$\omega k$$. So "move $$k$$ places forward" is the same matrix at every position. This is the "linear function" of the paper. Our script applied this matrix with $$k = 5$$ to every pair and every position from 0 to 49:

```text
  PE(pos + 5) = rotation(k=5) applied to PE(pos), max error over pos 0..49 and all 256 pairs: 6.2e-15
```

The error is at the limit of 64-bit arithmetic: the rule is exact.

A second consequence is useful for attention, which compares vectors with dot products. The dot product of two position vectors is $$\sum_i \cos(\omega_i k)$$, which depends only on the distance $$k$$:

```text
  dot product PE(pos) . PE(pos + k) for pos = 0, 7, 20, 40:
    k =  0:  256.00   256.00   256.00   256.00
    k =  1:  249.10   249.10   249.10   249.10
    k =  2:  231.73   231.73   231.73   231.73
    k =  5:  189.60   189.60   189.60   189.60
    k = 10:  173.79   173.79   173.79   173.79
    k = 20:  157.57   157.57   157.57   157.57
```

Each row has four equal numbers. The starting position does not matter; only the distance does. And the number falls as the distance grows, so near positions look more alike than far positions.

<figure class="fig"><svg viewBox="0 0 780 300" role="img" aria-label="The dot product of two position vectors depends only on how far apart they are. From position 0 and from position 3 the two curves lie on top of each other: 256 at offset 0, then 144.2 at offset 30 and 125.1 at offset 60."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="580" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">100</text><line class="grid" x1="64" y1="181.2" x2="580" y2="181.2"/><text class="t-tick" x="56.0" y="185.2" text-anchor="end">150</text><line class="grid" x1="64" y1="112.5" x2="580" y2="112.5"/><text class="t-tick" x="56.0" y="116.5" text-anchor="end">200</text><line class="grid" x1="64" y1="43.8" x2="580" y2="43.8"/><text class="t-tick" x="56.0" y="47.8" text-anchor="end">250</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">0</text><text class="t-tick" x="151.5" y="270.0" text-anchor="middle">10</text><text class="t-tick" x="238.9" y="270.0" text-anchor="middle">20</text><text class="t-tick" x="326.4" y="270.0" text-anchor="middle">30</text><text class="t-tick" x="413.8" y="270.0" text-anchor="middle">40</text><text class="t-tick" x="501.3" y="270.0" text-anchor="middle">50</text><text class="t-tick" x="588.7" y="270.0" text-anchor="middle">60</text><line class="axis" x1="64" y1="250" x2="580" y2="250"/><polyline class="l1" points="64.0,35.5 72.7,45.0 81.5,68.9 90.2,96.3 99.0,117.1 107.7,126.8 116.5,128.7 125.2,129.2 134.0,133.2 142.7,140.7 151.5,148.5 160.2,153.3 168.9,154.5 177.7,154.6 186.4,156.4 195.2,160.5 203.9,165.4 212.7,168.8 221.4,169.9 230.2,169.9 238.9,170.8 247.7,173.5 256.4,177.0 265.2,179.7 273.9,180.8 282.6,180.8 291.4,181.3 300.1,183.1 308.9,185.9 317.6,188.2 326.4,189.3 335.1,189.3 343.9,189.6 352.6,190.9 361.4,193.1 370.1,195.1 378.8,196.2 387.6,196.3 396.3,196.4 405.1,197.3 413.8,199.1 422.6,201.0 431.3,202.0 440.1,202.2 448.8,202.2 457.6,202.8 466.3,204.3 475.1,206.0 483.8,207.1 492.5,207.3 501.3,207.3 510.0,207.7 518.8,208.9 527.5,210.4 536.3,211.5 545.0,211.8 553.8,211.7 562.5,212.0 571.3,213.0 580.0,214.4"/><g class="mark"><title>from position 0, 0: 256</title><circle class="s1 ring" cx="64.0" cy="35.5" r="5"/></g><g class="mark"><title>from position 0, 1: 249.1</title><circle class="s1 ring" cx="72.7" cy="45.0" r="5"/></g><g class="mark"><title>from position 0, 2: 231.73</title><circle class="s1 ring" cx="81.5" cy="68.9" r="5"/></g><g class="mark"><title>from position 0, 3: 211.75</title><circle class="s1 ring" cx="90.2" cy="96.3" r="5"/></g><g class="mark"><title>from position 0, 4: 196.69</title><circle class="s1 ring" cx="99.0" cy="117.1" r="5"/></g><g class="mark"><title>from position 0, 5: 189.6</title><circle class="s1 ring" cx="107.7" cy="126.8" r="5"/></g><g class="mark"><title>from position 0, 6: 188.25</title><circle class="s1 ring" cx="116.5" cy="128.7" r="5"/></g><g class="mark"><title>from position 0, 7: 187.86</title><circle class="s1 ring" cx="125.2" cy="129.2" r="5"/></g><g class="mark"><title>from position 0, 8: 184.97</title><circle class="s1 ring" cx="134.0" cy="133.2" r="5"/></g><g class="mark"><title>from position 0, 9: 179.46</title><circle class="s1 ring" cx="142.7" cy="140.7" r="5"/></g><g class="mark"><title>from position 0, 10: 173.79</title><circle class="s1 ring" cx="151.5" cy="148.5" r="5"/></g><g class="mark"><title>from position 0, 11: 170.35</title><circle class="s1 ring" cx="160.2" cy="153.3" r="5"/></g><g class="mark"><title>from position 0, 12: 169.45</title><circle class="s1 ring" cx="168.9" cy="154.5" r="5"/></g><g class="mark"><title>from position 0, 13: 169.35</title><circle class="s1 ring" cx="177.7" cy="154.6" r="5"/></g><g class="mark"><title>from position 0, 14: 168.06</title><circle class="s1 ring" cx="186.4" cy="156.4" r="5"/></g><g class="mark"><title>from position 0, 15: 165.06</title><circle class="s1 ring" cx="195.2" cy="160.5" r="5"/></g><g class="mark"><title>from position 0, 16: 161.53</title><circle class="s1 ring" cx="203.9" cy="165.4" r="5"/></g><g class="mark"><title>from position 0, 17: 159.08</title><circle class="s1 ring" cx="212.7" cy="168.8" r="5"/></g><g class="mark"><title>from position 0, 18: 158.28</title><circle class="s1 ring" cx="221.4" cy="169.9" r="5"/></g><g class="mark"><title>from position 0, 19: 158.25</title><circle class="s1 ring" cx="230.2" cy="169.9" r="5"/></g><g class="mark"><title>from position 0, 20: 157.57</title><circle class="s1 ring" cx="238.9" cy="170.8" r="5"/></g><g class="mark"><title>from position 0, 21: 155.64</title><circle class="s1 ring" cx="247.7" cy="173.5" r="5"/></g><g class="mark"><title>from position 0, 22: 153.09</title><circle class="s1 ring" cx="256.4" cy="177.0" r="5"/></g><g class="mark"><title>from position 0, 23: 151.11</title><circle class="s1 ring" cx="265.2" cy="179.7" r="5"/></g><g class="mark"><title>from position 0, 24: 150.34</title><circle class="s1 ring" cx="273.9" cy="180.8" r="5"/></g><g class="mark"><title>from position 0, 25: 150.3</title><circle class="s1 ring" cx="282.6" cy="180.8" r="5"/></g><g class="mark"><title>from position 0, 26: 149.95</title><circle class="s1 ring" cx="291.4" cy="181.3" r="5"/></g><g class="mark"><title>from position 0, 27: 148.62</title><circle class="s1 ring" cx="300.1" cy="183.1" r="5"/></g><g class="mark"><title>from position 0, 28: 146.64</title><circle class="s1 ring" cx="308.9" cy="185.9" r="5"/></g><g class="mark"><title>from position 0, 29: 144.94</title><circle class="s1 ring" cx="317.6" cy="188.2" r="5"/></g><g class="mark"><title>from position 0, 30: 144.17</title><circle class="s1 ring" cx="326.4" cy="189.3" r="5"/></g><g class="mark"><title>from position 0, 31: 144.12</title><circle class="s1 ring" cx="335.1" cy="189.3" r="5"/></g><g class="mark"><title>from position 0, 32: 143.95</title><circle class="s1 ring" cx="343.9" cy="189.6" r="5"/></g><g class="mark"><title>from position 0, 33: 143</title><circle class="s1 ring" cx="352.6" cy="190.9" r="5"/></g><g class="mark"><title>from position 0, 34: 141.41</title><circle class="s1 ring" cx="361.4" cy="193.1" r="5"/></g><g class="mark"><title>from position 0, 35: 139.91</title><circle class="s1 ring" cx="370.1" cy="195.1" r="5"/></g><g class="mark"><title>from position 0, 36: 139.14</title><circle class="s1 ring" cx="378.8" cy="196.2" r="5"/></g><g class="mark"><title>from position 0, 37: 139.05</title><circle class="s1 ring" cx="387.6" cy="196.3" r="5"/></g><g class="mark"><title>from position 0, 38: 138.99</title><circle class="s1 ring" cx="396.3" cy="196.4" r="5"/></g><g class="mark"><title>from position 0, 39: 138.33</title><circle class="s1 ring" cx="405.1" cy="197.3" r="5"/></g><g class="mark"><title>from position 0, 40: 137.03</title><circle class="s1 ring" cx="413.8" cy="199.1" r="5"/></g><g class="mark"><title>from position 0, 41: 135.67</title><circle class="s1 ring" cx="422.6" cy="201.0" r="5"/></g><g class="mark"><title>from position 0, 42: 134.89</title><circle class="s1 ring" cx="431.3" cy="202.0" r="5"/></g><g class="mark"><title>from position 0, 43: 134.76</title><circle class="s1 ring" cx="440.1" cy="202.2" r="5"/></g><g class="mark"><title>from position 0, 44: 134.77</title><circle class="s1 ring" cx="448.8" cy="202.2" r="5"/></g><g class="mark"><title>from position 0, 45: 134.31</title><circle class="s1 ring" cx="457.6" cy="202.8" r="5"/></g><g class="mark"><title>from position 0, 46: 133.24</title><circle class="s1 ring" cx="466.3" cy="204.3" r="5"/></g><g class="mark"><title>from position 0, 47: 132.01</title><circle class="s1 ring" cx="475.1" cy="206.0" r="5"/></g><g class="mark"><title>from position 0, 48: 131.22</title><circle class="s1 ring" cx="483.8" cy="207.1" r="5"/></g><g class="mark"><title>from position 0, 49: 131.04</title><circle class="s1 ring" cx="492.5" cy="207.3" r="5"/></g><g class="mark"><title>from position 0, 50: 131.09</title><circle class="s1 ring" cx="501.3" cy="207.3" r="5"/></g><g class="mark"><title>from position 0, 51: 130.79</title><circle class="s1 ring" cx="510.0" cy="207.7" r="5"/></g><g class="mark"><title>from position 0, 52: 129.92</title><circle class="s1 ring" cx="518.8" cy="208.9" r="5"/></g><g class="mark"><title>from position 0, 53: 128.79</title><circle class="s1 ring" cx="527.5" cy="210.4" r="5"/></g><g class="mark"><title>from position 0, 54: 127.99</title><circle class="s1 ring" cx="536.3" cy="211.5" r="5"/></g><g class="mark"><title>from position 0, 55: 127.76</title><circle class="s1 ring" cx="545.0" cy="211.8" r="5"/></g><g class="mark"><title>from position 0, 56: 127.83</title><circle class="s1 ring" cx="553.8" cy="211.7" r="5"/></g><g class="mark"><title>from position 0, 57: 127.66</title><circle class="s1 ring" cx="562.5" cy="212.0" r="5"/></g><g class="mark"><title>from position 0, 58: 126.94</title><circle class="s1 ring" cx="571.3" cy="213.0" r="5"/></g><g class="mark"><title>from position 0, 59: 125.92</title><circle class="s1 ring" cx="580.0" cy="214.4" r="5"/></g><polyline class="l2" points="64.0,35.5 72.7,45.0 81.5,68.9 90.2,96.3 99.0,117.1 107.7,126.8 116.5,128.7 125.2,129.2 134.0,133.2 142.7,140.7 151.5,148.5 160.2,153.3 168.9,154.5 177.7,154.6 186.4,156.4 195.2,160.5 203.9,165.4 212.7,168.8 221.4,169.9 230.2,169.9 238.9,170.8 247.7,173.5 256.4,177.0 265.2,179.7 273.9,180.8 282.6,180.8 291.4,181.3 300.1,183.1 308.9,185.9 317.6,188.2 326.4,189.3 335.1,189.3 343.9,189.6 352.6,190.9 361.4,193.1 370.1,195.1 378.8,196.2 387.6,196.3 396.3,196.4 405.1,197.3 413.8,199.1 422.6,201.0 431.3,202.0 440.1,202.2 448.8,202.2 457.6,202.8 466.3,204.3 475.1,206.0 483.8,207.1 492.5,207.3 501.3,207.3 510.0,207.7 518.8,208.9 527.5,210.4 536.3,211.5 545.0,211.8 553.8,211.7 562.5,212.0 571.3,213.0 580.0,214.4"/><g class="mark"><title>from position 3, 0: 256</title><circle class="s2 ring" cx="64.0" cy="35.5" r="5"/></g><g class="mark"><title>from position 3, 1: 249.1</title><circle class="s2 ring" cx="72.7" cy="45.0" r="5"/></g><g class="mark"><title>from position 3, 2: 231.73</title><circle class="s2 ring" cx="81.5" cy="68.9" r="5"/></g><g class="mark"><title>from position 3, 3: 211.75</title><circle class="s2 ring" cx="90.2" cy="96.3" r="5"/></g><g class="mark"><title>from position 3, 4: 196.69</title><circle class="s2 ring" cx="99.0" cy="117.1" r="5"/></g><g class="mark"><title>from position 3, 5: 189.6</title><circle class="s2 ring" cx="107.7" cy="126.8" r="5"/></g><g class="mark"><title>from position 3, 6: 188.25</title><circle class="s2 ring" cx="116.5" cy="128.7" r="5"/></g><g class="mark"><title>from position 3, 7: 187.86</title><circle class="s2 ring" cx="125.2" cy="129.2" r="5"/></g><g class="mark"><title>from position 3, 8: 184.97</title><circle class="s2 ring" cx="134.0" cy="133.2" r="5"/></g><g class="mark"><title>from position 3, 9: 179.46</title><circle class="s2 ring" cx="142.7" cy="140.7" r="5"/></g><g class="mark"><title>from position 3, 10: 173.79</title><circle class="s2 ring" cx="151.5" cy="148.5" r="5"/></g><g class="mark"><title>from position 3, 11: 170.35</title><circle class="s2 ring" cx="160.2" cy="153.3" r="5"/></g><g class="mark"><title>from position 3, 12: 169.45</title><circle class="s2 ring" cx="168.9" cy="154.5" r="5"/></g><g class="mark"><title>from position 3, 13: 169.35</title><circle class="s2 ring" cx="177.7" cy="154.6" r="5"/></g><g class="mark"><title>from position 3, 14: 168.06</title><circle class="s2 ring" cx="186.4" cy="156.4" r="5"/></g><g class="mark"><title>from position 3, 15: 165.06</title><circle class="s2 ring" cx="195.2" cy="160.5" r="5"/></g><g class="mark"><title>from position 3, 16: 161.53</title><circle class="s2 ring" cx="203.9" cy="165.4" r="5"/></g><g class="mark"><title>from position 3, 17: 159.08</title><circle class="s2 ring" cx="212.7" cy="168.8" r="5"/></g><g class="mark"><title>from position 3, 18: 158.28</title><circle class="s2 ring" cx="221.4" cy="169.9" r="5"/></g><g class="mark"><title>from position 3, 19: 158.25</title><circle class="s2 ring" cx="230.2" cy="169.9" r="5"/></g><g class="mark"><title>from position 3, 20: 157.57</title><circle class="s2 ring" cx="238.9" cy="170.8" r="5"/></g><g class="mark"><title>from position 3, 21: 155.64</title><circle class="s2 ring" cx="247.7" cy="173.5" r="5"/></g><g class="mark"><title>from position 3, 22: 153.09</title><circle class="s2 ring" cx="256.4" cy="177.0" r="5"/></g><g class="mark"><title>from position 3, 23: 151.11</title><circle class="s2 ring" cx="265.2" cy="179.7" r="5"/></g><g class="mark"><title>from position 3, 24: 150.34</title><circle class="s2 ring" cx="273.9" cy="180.8" r="5"/></g><g class="mark"><title>from position 3, 25: 150.3</title><circle class="s2 ring" cx="282.6" cy="180.8" r="5"/></g><g class="mark"><title>from position 3, 26: 149.95</title><circle class="s2 ring" cx="291.4" cy="181.3" r="5"/></g><g class="mark"><title>from position 3, 27: 148.62</title><circle class="s2 ring" cx="300.1" cy="183.1" r="5"/></g><g class="mark"><title>from position 3, 28: 146.64</title><circle class="s2 ring" cx="308.9" cy="185.9" r="5"/></g><g class="mark"><title>from position 3, 29: 144.94</title><circle class="s2 ring" cx="317.6" cy="188.2" r="5"/></g><g class="mark"><title>from position 3, 30: 144.17</title><circle class="s2 ring" cx="326.4" cy="189.3" r="5"/></g><g class="mark"><title>from position 3, 31: 144.12</title><circle class="s2 ring" cx="335.1" cy="189.3" r="5"/></g><g class="mark"><title>from position 3, 32: 143.95</title><circle class="s2 ring" cx="343.9" cy="189.6" r="5"/></g><g class="mark"><title>from position 3, 33: 143</title><circle class="s2 ring" cx="352.6" cy="190.9" r="5"/></g><g class="mark"><title>from position 3, 34: 141.41</title><circle class="s2 ring" cx="361.4" cy="193.1" r="5"/></g><g class="mark"><title>from position 3, 35: 139.91</title><circle class="s2 ring" cx="370.1" cy="195.1" r="5"/></g><g class="mark"><title>from position 3, 36: 139.14</title><circle class="s2 ring" cx="378.8" cy="196.2" r="5"/></g><g class="mark"><title>from position 3, 37: 139.05</title><circle class="s2 ring" cx="387.6" cy="196.3" r="5"/></g><g class="mark"><title>from position 3, 38: 138.99</title><circle class="s2 ring" cx="396.3" cy="196.4" r="5"/></g><g class="mark"><title>from position 3, 39: 138.33</title><circle class="s2 ring" cx="405.1" cy="197.3" r="5"/></g><g class="mark"><title>from position 3, 40: 137.03</title><circle class="s2 ring" cx="413.8" cy="199.1" r="5"/></g><g class="mark"><title>from position 3, 41: 135.67</title><circle class="s2 ring" cx="422.6" cy="201.0" r="5"/></g><g class="mark"><title>from position 3, 42: 134.89</title><circle class="s2 ring" cx="431.3" cy="202.0" r="5"/></g><g class="mark"><title>from position 3, 43: 134.76</title><circle class="s2 ring" cx="440.1" cy="202.2" r="5"/></g><g class="mark"><title>from position 3, 44: 134.77</title><circle class="s2 ring" cx="448.8" cy="202.2" r="5"/></g><g class="mark"><title>from position 3, 45: 134.31</title><circle class="s2 ring" cx="457.6" cy="202.8" r="5"/></g><g class="mark"><title>from position 3, 46: 133.24</title><circle class="s2 ring" cx="466.3" cy="204.3" r="5"/></g><g class="mark"><title>from position 3, 47: 132.01</title><circle class="s2 ring" cx="475.1" cy="206.0" r="5"/></g><g class="mark"><title>from position 3, 48: 131.22</title><circle class="s2 ring" cx="483.8" cy="207.1" r="5"/></g><g class="mark"><title>from position 3, 49: 131.04</title><circle class="s2 ring" cx="492.5" cy="207.3" r="5"/></g><g class="mark"><title>from position 3, 50: 131.09</title><circle class="s2 ring" cx="501.3" cy="207.3" r="5"/></g><g class="mark"><title>from position 3, 51: 130.79</title><circle class="s2 ring" cx="510.0" cy="207.7" r="5"/></g><g class="mark"><title>from position 3, 52: 129.92</title><circle class="s2 ring" cx="518.8" cy="208.9" r="5"/></g><g class="mark"><title>from position 3, 53: 128.79</title><circle class="s2 ring" cx="527.5" cy="210.4" r="5"/></g><g class="mark"><title>from position 3, 54: 127.99</title><circle class="s2 ring" cx="536.3" cy="211.5" r="5"/></g><g class="mark"><title>from position 3, 55: 127.76</title><circle class="s2 ring" cx="545.0" cy="211.8" r="5"/></g><g class="mark"><title>from position 3, 56: 127.83</title><circle class="s2 ring" cx="553.8" cy="211.7" r="5"/></g><g class="mark"><title>from position 3, 57: 127.66</title><circle class="s2 ring" cx="562.5" cy="212.0" r="5"/></g><g class="mark"><title>from position 3, 58: 126.94</title><circle class="s2 ring" cx="571.3" cy="213.0" r="5"/></g><g class="mark"><title>from position 3, 59: 125.92</title><circle class="s2 ring" cx="580.0" cy="214.4" r="5"/></g><line class="grid" x1="586.0" y1="214.4" x2="596.0" y2="214.4"/><rect class="s1" x="600.0" y="209.4" width="10" height="10" rx="2"/><text class="t-note" x="616.0" y="218.4" text-anchor="start">from position 0: 125.92</text><line class="grid" x1="586.0" y1="214.4" x2="596.0" y2="231.4"/><rect class="s2" x="600.0" y="226.4" width="10" height="10" rx="2"/><text class="t-note" x="616.0" y="235.4" text-anchor="start">from position 3: 125.92</text><text class="t-tick" x="322.0" y="290.0" text-anchor="middle">offset k</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">PE(pos) · PE(pos + k)</text></svg><figcaption>The dot product of two position vectors against their distance k. The curves from position 0 and from position 3 are the same: the dot product depends only on the distance.</figcaption></figure>

> [!PAPER] · §3.5
> [![The last paragraph of Section 3.5, with nearly identical results and extrapolate to sequence lengths longer highlighted](/img/papers/transformer/p2-pe-3.png)](/img/papers/transformer/p2-pe-3.png)
>
> **What it says:** the team also tried "learned positional embeddings" and got "nearly identical results" (Table 3, row E). They chose the sine version because "it may allow the model to extrapolate to sequence lengths longer than the ones encountered during training".
>
> **What it means:**
>
> The other choice is a learned table, like the word embedding: one row of 512 free numbers for each position. Training fills the rows. It worked just as well, which tells us that the exact form of the label is not critical. The model mostly needs some distinct label for each position.
>
> The sine version has one advantage in theory. A learned table has rows only for the positions seen in training; a sentence longer than that has no label. The formula gives a value for any position, even position 10,000. Note the word "may": the paper does not test this claim. [Part 6](part-6-results-and-impact.md) reads row E of Table 3.

## What our model cannot show

MarianMT has the paper's shapes and the paper's order of operations, and it shares one table across its three jobs. But it is not the paper's model. It was trained on other data, with a larger vocabulary, and it uses swish where the paper uses ReLU. Its weights tell us how the design works; they do not tell us what the paper's own weights looked like. We report no translation scores of our own.

> [!TAKEAWAYS] Key takeaways
> - **Two halves.** The encoder reads the whole input and writes one vector of 512 numbers per token ($$z$$). The decoder writes the output one token at a time, reading $$z$$ and its own earlier tokens.
> - **Six identical layers on each side,** each with its own weights. An encoder layer has two sub-layers (self-attention, feed-forward); a decoder layer has three (masked self-attention, attention over $$z$$, feed-forward).
> - **Add, then normalize.** Each sub-layer output is $$\text{LayerNorm}(x + \text{Sublayer}(x))$$. The residual path keeps the input; the layer norm resets each vector to mean 0 and standard deviation 1. We rebuilt layer 1 of the released model by hand with an exact match.
> - **The feed-forward network works on each token alone:** 512 to 2048 to 512 with a ReLU. It holds two thirds of the weights of an encoder layer (2,099,712 of 3,152,384).
> - **One table, three jobs.** In our model, the same 58,101 × 512 matrix is the encoder embedding, the decoder embedding and the output layer. This saves about 59.5 million numbers.
> - **The $$\sqrt{512} = 22.63$$ scale** lifts the small embedding rows (median length 1.08) above the position vectors (length 16).
> - **Without positions, attention sees a bag of words.** Swapping two tokens only swapped the outputs.
> - **The position encoding is a clock with 256 hands,** with wavelengths from 6.3 to about 60,612 positions. A shift by $$k$$ is a fixed rotation, and the dot product of two positions depends only on their distance.

**Next, in Part 3:** attention itself. The scaled dot-product formula, why it divides by $$\sqrt{d_k}$$, the 8 heads and the mask of the decoder.

<details>
<summary>Run it yourself</summary>

The script [`code/papers/transformer/tx_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/transformer/tx_part2.py) loads the released English to German model, counts its weights and translates one sentence. It rebuilds encoder layer 1 by hand, tests the feed-forward network token by token and checks the shared table. It also computes the position encoding from the formula and tests the rotation rule. Every number of this part is in `results/part2_stdout.txt`.

```bash
pip install torch transformers sentencepiece
python tx_part2.py        # uses Helsinki-NLP/opus-mt-en-de (about 300 MB), CPU, a few seconds
```

<figure><img src="/img/papers/transformer/part2-run.png" alt="Terminal output of tx_part2.py: the configuration and weight counts, the translation of one sentence, the shapes through the encoder, layer 1 rebuilt by hand, the feed-forward and shared-table checks, and the position encoding table" loading="lazy" /><figcaption>The first part of the output of tx_part2.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017) · [official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
