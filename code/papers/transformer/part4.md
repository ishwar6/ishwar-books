---
title: "Why Self-Attention: Cost, Steps and Paths"
description: "Section 4 compares self-attention with recurrent and convolutional layers on three measures: work per layer, steps that must wait for each other, and the distance a signal must travel. We work out Table 1 with real numbers and measure all three on a laptop CPU."
part: 4
covers: "§4, Table 1"
date: 2026-10-18
tags: [transformer, attention, complexity, rnn, deep-learning]
---

Parts 2 and 3 showed what the Transformer is. Section 4 answers a different question: why build a layer out of attention, and not out of a recurrent or a convolutional network? The paper gives three measures and one table, and we put real numbers into each cell.

## Three measures for a layer {§4}

> [!PAPER] · §4
> [![The first paragraph of Section 4, with recurrent and convolutional layers, one variable-length sequence of symbol representations and three desiderata highlighted](/img/papers/transformer/p4-intro.png)](/img/papers/transformer/p4-intro.png)
>
> **What it says:** the section compares "self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $$(x_1, ..., x_n)$$ to another sequence of equal length $$(z_1, ..., z_n)$$". To motivate self-attention, the authors "consider three desiderata".
>
> **What it means:**
>
> Think of one layer as a machine on a conveyor belt. A row of $$n$$ vectors comes in, one vector for each word. A row of $$n$$ new vectors goes out, again one for each word. Each vector has $$d$$ numbers. In the base Transformer, $$d = 512$$. All three kinds of layer do this same job, so we can compare them fairly.
>
> The three candidates are the three ways people knew in 2017 to let words share information. A recurrent layer reads the words one by one and carries a memory. A convolutional layer looks at a small window of neighbours. A self-attention layer lets every word look at every other word at once.
>
> "Desiderata" is Latin for "things we want". The paper wants a layer that is cheap, that a computer can run in parallel, and that lets distant words affect each other easily. The rest of the section scores each layer on these three wishes.

> [!DEFINITION] Representation dimension $$d$$
> The number of values in the vector that stands for one word inside a layer. The base Transformer uses $$d = 512$$, so each word is a list of 512 decimal numbers. The paper also writes $$d_{\text{model}}$$ for it.

> [!DEFINITION] Sequence length $$n$$
> The number of tokens in the input. A token is one word or one piece of a word. The sentence "The cat sat on the mat." is 8 tokens for the tokenizer of the translation model that we use below.

> [!PAPER] · §4
> [![The paragraphs of Section 4 that name the three measures: total computational complexity per layer, minimum number of sequential operations required, and the path length between long-range dependencies](/img/papers/transformer/p4-desiderata.png)](/img/papers/transformer/p4-desiderata.png)
>
> **What it says:** the three measures are "the total computational complexity per layer", the amount of parallel work "as measured by the minimum number of sequential operations required", and "the path length between long-range dependencies in the network". The authors add: "The shorter these paths ... the easier it is to learn long-range dependencies."
>
> **What it means:**
>
> The first measure is plain cost: how many multiplications one layer needs. The second is about waiting. Some steps must wait for the result of an earlier step, and no number of processors can make them go faster. A GPU has thousands of small cores, so a layer with few waiting steps uses the hardware well.
>
> The third measure is the most subtle. Take the sentence "The keys that the man left on the table are lost". The verb "are" must agree with "keys", eight words back. To learn this, the training signal must travel from "are" back to "keys" through the network. Each hop on the way can weaken the signal or add noise. A short path means a strong, clean signal.
>
> The last sentence cites earlier work on recurrent networks (reference [12]). It showed that gradients in long chains tend to vanish. We test this claim ourselves later on this page.

> [!DEFINITION] Sequential operation
> A step that cannot start before an earlier step ends. A recurrent network must compute the memory of word 1 before word 2, and word 2 before word 3. Such steps form a chain. The length of the chain sets a floor on the time, however many processors you have.

> [!DEFINITION] Path length
> The number of layers or steps that a signal crosses to go from one input position to one output position. The paper uses the longest such path between any two positions, the **maximum path length**.

## Table 1, cell by cell {§Table 1}

> [!PAPER] · Table 1
> [![Table 1 of the paper: maximum path lengths, per-layer complexity and minimum number of sequential operations for self-attention, recurrent, convolutional and restricted self-attention layers](/img/papers/transformer/p4-table1.png)](/img/papers/transformer/p4-table1.png)
>
> **What it says:** "n is the sequence length, d is the representation dimension, k is the kernel size of convolutions and r the size of the neighborhood in restricted self-attention." Self-attention costs $$O(n^2 \cdot d)$$ with $$O(1)$$ sequential operations and path length $$O(1)$$. A recurrent layer costs $$O(n \cdot d^2)$$ with $$O(n)$$ and $$O(n)$$.
>
> **What it means:**
>
> Read the table as a scorecard with three columns. In the two right columns, self-attention gets the best possible score: a constant. It does not matter if the sentence has 10 words or 1,000. All words are handled in one parallel step, and any word reaches any other word in one hop.
>
> The recurrent layer gets the worst score in both columns: both grow with $$n$$. Its only win is in the first column, and only when sentences are long. The convolutional layer sits in between: no waiting, but a path that grows slowly with $$n$$.
>
> The last row is a variant that the paper proposes but does not use: attention restricted to $$r$$ neighbours. It trades a longer path for a lower cost. We come back to it below.

> [!DEFINITION] Big-O notation
> A way to say how a cost grows, without the exact count. $$O(n^2 \cdot d)$$ means "about a fixed number times $$n^2 \cdot d$$". If $$n$$ doubles, the cost becomes about four times larger. $$O(1)$$ means the cost does not grow with $$n$$ at all.

### The first column: work per layer

Where do the two formulas come from? Count the multiplications.

A recurrent layer updates its memory once for each word. Each update multiplies a vector of $$d$$ numbers by a $$d \times d$$ matrix, which is $$d^2$$ multiplications. With $$n$$ words, that is $$n \cdot d^2$$.

Self-attention compares each word with each other word. That is $$n \times n$$ pairs. Each comparison is a dot product of two vectors of $$d$$ numbers, so it costs $$d$$ multiplications. The mix of the values costs the same again. So the core of attention costs about $$n^2 \cdot d$$.

$$
\text{self-attention: } n^2 \cdot d \qquad \text{recurrent: } n \cdot d^2 \qquad \frac{n^2 \cdot d}{n \cdot d^2} = \frac{n}{d}
$$

Read the ratio on the right. The two costs share a factor $$n \cdot d$$, so it cancels. What is left is $$n / d$$: sentence length over vector size. When $$n$$ is smaller than $$d$$, attention does less work. When $$n$$ is larger, the recurrent layer does less.

A tiny example you can check by hand. Take $$n = 4$$ words and vectors of $$d = 3$$ numbers. The attention scores are a $$4 \times 4$$ grid of 16 dot products, each with 3 multiplications: 48 in all. The recurrent layer does 4 updates of $$3 \times 3 = 9$$ multiplications: 36 in all. The ratio is $$48 / 36 = 4 / 3$$, which is exactly $$n / d$$.

Now the real sizes, with $$d = 512$$ as in the base model. Our script `tx_part4.py` prints the counts:

| $$n$$ | self-attention $$n^2 d$$ | recurrent $$n d^2$$ | convolution $$k n d^2$$, $$k=3$$ | restricted $$r n d$$, $$r=16$$ | ratio $$n/d$$ |
|---:|---:|---:|---:|---:|---:|
| 10 | 51,200 | 2,621,440 | 7,864,320 | 81,920 | 0.020 |
| 27 | 373,248 | 7,077,888 | 21,233,664 | 221,184 | 0.053 |
| 100 | 5,120,000 | 26,214,400 | 78,643,200 | 819,200 | 0.195 |
| 512 | 134,217,728 | 134,217,728 | 402,653,184 | 4,194,304 | 1.000 |
| 2,048 | 2,147,483,648 | 536,870,912 | 1,610,612,736 | 16,777,216 | 4.000 |
| 8,192 | 34,359,738,368 | 2,147,483,648 | 6,442,450,944 | 67,108,864 | 16.000 |

For a sentence of 27 tokens, the attention core needs about 19 times less work than a recurrent layer. The two meet at exactly $$n = 512 = d$$. At 8,192 tokens, attention needs 16 times more.

One honest note. Table 1 counts only the part that differs between the layers. A real attention layer also multiplies every vector by the matrices $$W^Q$$, $$W^K$$, $$W^V$$ and $$W^O$$ of Part 3. That adds about $$4 \cdot n \cdot d^2$$, the same size as a recurrent layer. A real recurrent layer, such as an LSTM, also has several matrices. So the table compares the parts that grow differently, and leaves out the shared $$n \cdot d^2$$ terms.

> [!PAPER] · §4
> [![The lines of Section 4 that say a self-attention layer connects all positions with a constant number of sequentially executed operations, a recurrent layer requires O(n), and self-attention is faster when the sequence](/img/papers/transformer/p4-constant.png)](/img/papers/transformer/p4-constant.png)
>
> **What it says:** "a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires $$O(n)$$ sequential operations." Also, "self-attention layers are faster than recurrent layers when the sequence"...
>
> **What it means:**
>
> Picture a relay race and a swimming race. In a relay, runner 2 cannot start before runner 1 hands over the baton. Four runners take four legs of time. In a swimming race, all swimmers start at the same moment. Eight swimmers take the time of one length.
>
> A recurrent layer is a relay: the memory after word $$t$$ needs the memory after word $$t-1$$. Self-attention is the swimming race. The scores of all $$n \times n$$ pairs are independent, so one large matrix product computes them all at once. That is why "sequential operations" is $$O(n)$$ for one and $$O(1)$$ for the other.

> [!PAPER] · §4
> [![The lines of Section 4 that end the sentence: the sequence length n is smaller than the representation dimensionality d, which is most often the case, with word-piece and byte-pair representations](/img/papers/transformer/p4-n-smaller-d.png)](/img/papers/transformer/p4-n-smaller-d.png)
>
> **What it says:** self-attention is faster "when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [38] and byte-pair [31] representations."
>
> **What it means:**
>
> This is the $$n / d$$ ratio in words. The claim depends on one fact about the data: real sentences are short compared to 512. Word-piece and byte-pair are two ways to cut text into tokens. Common words stay whole, and rare words break into pieces. So the number of tokens is a little larger than the number of words, but not by much.
>
> We can check the claim. The translation model that we use in this series, Helsinki-NLP/opus-mt-en-de, has $$d = 512$$ like the base Transformer. Its tokenizer is a close relative of byte-pair encoding. We gave it six English sentences, from 5 to 32 words long.

> [!DEFINITION] Subword token (word-piece, byte-pair)
> A piece of text that the model treats as one unit. A tokenizer learns a vocabulary of frequent pieces from data. "dimensionality" becomes the two tokens "▁dimension" and "ality". The mark "▁" shows the start of a word.

The tokenizer gives these lengths:

```
2. Real sentence lengths (Marian en-de tokenizer, d_model = 512)
  n =   8  words =  6  n/d = 0.016  The cat sat on the mat.
  n =   7  words =  5  n/d = 0.014  Attention is all you need.
  n =  13  words = 12  n/d = 0.025  The agreement on the European Economic Area was signed in Au...
  n =  23  words = 17  n/d = 0.045  Self-attention layers are faster than recurrent layers when ...
  n =  33  words = 26  n/d = 0.064  Learning long-range dependencies is a key challenge in many ...
  n =  39  words = 32  n/d = 0.076  The Transformer allows for significantly more parallelizatio...
  tokens of sentence 4: ▁Self - at tention ▁layers ▁are ▁faster ▁than ▁recurrent ▁layers ▁when ▁the ▁sequence ▁length ▁is ▁smaller ▁than ▁the ▁representation ▁dimension ality . </s>
```

Even the longest sentence, at 39 tokens, has $$n / d = 0.076$$. So for this sentence the attention core does about 13 times less work than a recurrent layer of the same width. Each count includes the end-of-sentence token `</s>`. These six sentences are our own choice, not the paper's test set. They show the order of size, not an average over a corpus.

### Time on a real computer

Counting multiplications is one thing. Does a real computer agree? Our script times one self-attention step against one recurrent loop, both with $$d = 512$$. The attention step includes the three projections to queries, keys and values. The recurrent loop does its input projection for all words at once, then runs the chain of $$n$$ updates.

[![Terminal output of tx_part4.py with the work counts, sentence lengths, CPU timings, gradient sizes and layer counts](/img/papers/transformer/part4-run.png)](/img/papers/transformer/part4-run.png)

```
3. Wall time on CPU (4 threads), d = 512, median of 7 runs, milliseconds
     n  self-attn  recurrent loop  steps attn  steps rnn
    16      0.059           0.156           1         16
    64      0.147           0.573           1         64
   256      0.578           2.449           1        256
  1024      4.568           8.970           1       1024
  4096     42.882          37.640           1       4096
```

At 16 tokens, attention takes 0.059 ms and the loop takes 0.156 ms. At 1,024 tokens, attention is still about two times faster. At 4,096 tokens, the $$n^2$$ term wins and attention becomes the slower one, as the arithmetic predicts.

Two limits apply to this test. It ran on a laptop CPU with 4 threads, in PyTorch, with random weights and one batch of one sentence. A GPU runs the large matrix products of attention much faster, but cannot speed up the 4,096 small waiting steps of the loop. So we expect the gap at short lengths to be larger on a GPU, not smaller. We did not measure a GPU here.

## The path between two words {§4}

The third column of Table 1 is about learning, not speed. The figure below draws the three layers for a sentence of 8 words. The bold lines show the shortest route from word 1 to the output at word 8.

{{FIG:p4_paths|The maximum path length of Table 1, drawn for 8 words. A recurrent layer passes word 1 through a chain of 7 memory updates. A dilated convolution with kernel 3 reaches it in 2 layers. Self-attention reaches it in 1 step.}}

Why does a long path hurt? During training, the network learns from the gradient: a signal that says how much each input should change. In a recurrent layer, the gradient from word 8 to word 1 passes through 7 updates. Each update multiplies it by a matrix and by the slope of the tanh function. If these factors are a little smaller than 1, the product shrinks fast, like $$0.6^7 \approx 0.03$$.

> [!DEFINITION] Gradient
> For each number in the network, how much the loss changes when that number changes a little. Training moves each number against its gradient. Here we use the gradient of the last output with respect to each input word: it shows how strongly that word can affect the output.

We measured this. Our script builds a recurrent network with tanh and a single self-attention layer, both with random weights. Each has vectors of 64 numbers and reads 50 positions. We compute the size of the gradient of the last output with respect to each input, and take the mean of 20 random networks.

```
4. Gradient norm |d out_last / d x_i| (random weights, d = 64, n = 50), mean of 20 seeds
 input i  distance          RNN    self-attn
       1        49     1.55e-09     2.97e-01
      10        40     7.31e-08     2.76e-01
      20        30     5.23e-06     2.13e-01
      30        20     6.06e-04     1.77e-01
      40        10     4.66e-02     2.14e-01
      45         5     5.58e-01     3.11e-01
      49         1     3.33e+00     1.99e-01
      50         0     5.02e+00     1.87e+00
```

{{FIG:p4_grad|How strongly the last output depends on each input, against the distance between them (log scale). In the recurrent network the gradient falls by about 10 to the power 9 over 49 positions. In one self-attention layer it stays between about 0.18 and 0.31 at every distance above 0. Random weights, 64 numbers per vector, mean of 20 networks.}}

The numbers tell the story of the third column. In the recurrent network, the first word has a gradient of $$1.55 \times 10^{-9}$$, about three billion times smaller than the last word. In practice, such a word cannot teach the network anything. In the attention layer, the first word gets 0.297 and the 49th word gets 0.199. Distance does not matter.

This test has limits too. The weights are random, not trained. A trained LSTM keeps gradients much better than our plain tanh network, because its gates were designed for this problem. And the input at distance 0 is larger in attention because it also feeds the query of the last position. The test shows the shape of the effect, not the size it has in a real trained model.

## When sentences get long {§4}

> [!PAPER] · §4
> [![The lines of Section 4 about restricting self-attention to a neighborhood of size r, which would increase the maximum path length to O(n/r)](/img/papers/transformer/p4-restricted.png)](/img/papers/transformer/p4-restricted.png)
>
> **What it says:** "To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r ... This would increase the maximum path length to $$O(n/r)$$. We plan to investigate this approach further in future work."
>
> **What it means:**
>
> The authors saw the weak point in their own table. The $$n^2$$ term is harmless for sentences but grows fast for whole documents. Their fix is a window. Each word looks only at the $$r$$ words around it, so the cost falls from $$n^2 \cdot d$$ to $$r \cdot n \cdot d$$.
>
> The price is the path. Word 1 can no longer see word 1,000 in one step. The signal must hop from window to window, about $$n / r$$ hops. With $$n = 8{,}192$$ and $$r = 16$$, the core cost falls from about 34 billion to about 67 million multiplications, 512 times less. But the longest path grows from 1 to about 512 hops.
>
> "Future work" turned out to be a large field. Many later models for long text use some form of local or sparse attention, often with a few global tokens to keep paths short. The trade-off in this one sentence is the same trade-off they all balance.

## Why not convolutions? {§4}

> [!PAPER] · §4
> [![The paragraph of Section 4 on convolutional layers: a kernel of width k does not connect all pairs, a stack of O(n/k) layers or O(log_k(n)) dilated layers is needed, convolutions cost k times more than recurrent layers, and separable convolutions cost the same as self-attention plus a feed-forward layer](/img/papers/transformer/p4-conv.png)](/img/papers/transformer/p4-conv.png)
>
> **What it says:** "A single convolutional layer with kernel width k < n does not connect all pairs of input and output positions. Doing so requires a stack of $$O(n/k)$$ convolutional layers in the case of contiguous kernels, or $$O(\log_k(n))$$ in the case of dilated convolutions." They are "more expensive than recurrent layers, by a factor of k". Even a separable convolution with $$k = n$$ costs the same as "the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model."
>
> **What it means:**
>
> A convolution is a small window that slides along the sentence. With kernel width $$k = 3$$, each output sees one word on each side. To see further, you stack layers: two layers see 5 words, three layers see 7. Each layer adds $$k - 1 = 2$$ words of reach. A dilated convolution spreads its window out: layer 1 looks at neighbours, layer 2 at positions 3 apart, layer 3 at positions 9 apart. Then the reach grows as $$k^L$$, much faster.
>
> The cost is $$k$$ times the recurrent cost, because each output mixes $$k$$ input vectors with a $$d \times d$$ matrix each. ConvS2S and ByteNet, the models from Part 1 that used convolutions, paid this cost to get rid of the waiting steps of recurrence. Self-attention removes the waiting steps too, and also makes the path short.
>
> The last sentence is a neat argument. A separable convolution splits the work into a cheap per-channel window and a $$d \times d$$ mix. With a window as wide as the whole sentence, its cost equals one attention layer plus one feed-forward layer. That pair is exactly one Transformer block from Part 2. So the Transformer reaches the full sentence for the price of the most efficient convolution.

> [!DEFINITION] Dilated convolution
> A convolution whose window has gaps. With kernel 3 and dilation 3, an output looks at positions $$t - 3$$, $$t$$ and $$t + 3$$. Stacking layers with dilations 1, 3, 9, ... lets $$L$$ layers see $$3^L$$ positions.

Our script counts the layers that a kernel of width 3 needs to connect the first and the last word:

```
5. Layers needed to connect position 1 and position n (kernel k = 3)
  n =   10: contiguous    5 layers, dilated  3 layers, self-attention 1 layer, RNN 9 steps
  n =   27: contiguous   13 layers, dilated  3 layers, self-attention 1 layer, RNN 26 steps
  n =  100: contiguous   50 layers, dilated  5 layers, self-attention 1 layer, RNN 99 steps
  n =  512: contiguous  256 layers, dilated  6 layers, self-attention 1 layer, RNN 511 steps
```

The formulas behind the counts: $$L$$ contiguous layers reach $$1 + L(k-1)$$ positions, and $$L$$ dilated layers reach $$k^L$$. For $$n = 27$$, contiguous layers need $$\lceil 26 / 2 \rceil = 13$$, and dilated layers need $$\log_3 27 = 3$$, because $$3^3 = 27$$. Self-attention needs 1 layer for every $$n$$.

## A side benefit: we can look inside {§4}

> [!PAPER] · §4
> [![The paragraph of Section 4 that says self-attention could yield more interpretable models, and that individual attention heads learn different tasks related to the syntactic and semantic structure of sentences](/img/papers/transformer/p4-interpret.png)](/img/papers/transformer/p4-interpret.png)
>
> **What it says:** "As side benefit, self-attention could yield more interpretable models." The authors inspect attention distributions and find that "individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences."
>
> **What it means:**
>
> In a recurrent network, the memory is one vector of 512 numbers that mixes everything read so far. It is hard to say which earlier word caused what. In attention, each head writes down an explicit table: for this word, how much weight on each other word. We can print that table and read it.
>
> The paper shows such tables in its appendix, with heads that link a pronoun to its noun, and Part 6 looks at them. Be careful with the word "could". An attention weight shows where a head looked, not always why the model gave its answer. Later research found that attention maps are a useful clue, but not a full explanation.

## The three wishes, scored

Put the three measures side by side for a typical sentence of 27 tokens and $$d = 512$$:

| layer | work of the core | waiting steps | longest path |
|---|---:|---:|---:|
| self-attention | 373,248 | 1 | 1 |
| recurrent | 7,077,888 | 27 | 27 |
| convolution, $$k = 3$$, contiguous | 21,233,664 | 1 | 13 layers |
| convolution, $$k = 3$$, dilated | 21,233,664 | 1 | 3 layers |

For sentences, self-attention wins all three columns. That is the whole argument of Section 4. It does not say that attention is better for every input. For very long inputs, the $$n^2$$ cost returns, and the authors say so.

> [!TAKEAWAYS] Key takeaways
> - Section 4 scores a layer on three measures: work per layer, steps that must wait for each other, and the longest path between two positions.
> - Self-attention costs $$O(n^2 \cdot d)$$ and a recurrent layer costs $$O(n \cdot d^2)$$. Their ratio is $$n / d$$.
> - With $$d = 512$$, self-attention does less core work whenever a sentence has fewer than 512 tokens. Our six test sentences had 7 to 39 tokens.
> - Self-attention has one sequential step; a recurrent layer has $$n$$. On our CPU, attention was faster up to 1,024 tokens and slower at 4,096.
> - Any word reaches any other word in one attention step. In our random networks, the recurrent gradient fell to $$1.55 \times 10^{-9}$$ over 49 positions; the attention gradient stayed near 0.2.
> - Convolutions need a stack of layers to connect distant words: 13 contiguous or 3 dilated layers for 27 tokens with $$k = 3$$.
> - Restricted attention with a window $$r$$ cuts the cost to $$O(r \cdot n \cdot d)$$ but makes the path $$O(n / r)$$.
> - Table 1 counts only the parts that grow differently. The projections add about $$4 \cdot n \cdot d^2$$ to a real attention layer.

## The paper and the code

[Paper (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) · [Official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
