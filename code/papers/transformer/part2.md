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

{{FIG:p2_layer|One encoder layer of our model, with the real shapes for our sentence of 12 tokens. The two curved arrows are the residual connections. The weight counts come from the released model.}}

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

{{FIG:p2_waves|Three dimensions of the position encoding over positions 0 to 63. Each is a sine wave with its own wavelength: 6.3, 19.9 and 62.8 positions. Low dimensions turn fast; high dimensions turn slowly.}}

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

{{FIG:p2_dots|The dot product of two position vectors against their distance k. The curves from position 0 and from position 3 are the same: the dot product depends only on the distance.}}

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
