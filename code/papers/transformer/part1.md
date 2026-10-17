---
title: "The Big Idea: Attention Is All You Need"
description: "Why the best translation models of 2016 were slow to train, and the claim that attention alone can replace them. We read the title, the abstract and the first two sections, and we run a released Transformer that translates English into German."
part: 1
covers: "Title, Abstract, §1, §2"
date: 2026-10-18
tags: [transformer, attention, machine-translation, recurrent-networks, nlp]
---

In June 2017, a team from Google put a short paper on arXiv. It described a translation model that reads a whole sentence in one step. The best models of the time read it one word after the other. The model is the **Transformer**, and its design now runs almost every large language model.

Each piece below shows the exact lines of the paper, then explains them.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## The title {§Title}

> [!PAPER] · Title
> [![The title of the paper: Attention Is All You Need](/img/papers/transformer/p1-title.png)](/img/papers/transformer/p1-title.png)
>
> **What it says:** "Attention Is All You Need."
>
> **What it means:**
>
> The title is a claim, and a bold one. In 2017, a translation model was a large machine with many parts. Recurrent layers read the words in order. Convolutional layers looked at small windows of words. An attention step helped the two halves of the model talk to each other. Attention was a helper part, added to a recurrent network.
>
> The title says that the helper can do the whole job. Remove the recurrent layers and the convolutions, keep attention, and the model gets better and faster. The rest of the paper is the evidence for this claim.
>
> The word "need" is also exact. The paper does not say that attention is the best part of a model. It says that the other parts are not necessary. The model still has small feed-forward layers and some other pieces (Part 2 shows all of them). But the only step that moves information between words is attention.

> [!DEFINITION] Attention
> A step in which one word vector looks at a set of other word vectors and takes a weighted average of them. The weights say how much each other word matters for this word. They are positive and add up to 1. The network computes the weights from the vectors themselves, so they change from sentence to sentence. [Part 3](part-3-attention.md) gives the exact equation.

## The abstract {§Abstract}

> [!PAPER] · Abstract
> [![The first sentences of the abstract, with sequence transduction models, recurrent or convolutional neural networks, an encoder and a decoder, and through an attention mechanism highlighted](/img/papers/transformer/p1-abstract-1.png)](/img/papers/transformer/p1-abstract-1.png)
>
> **What it says:** "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism."
>
> **What it means:**
>
> "Sequence transduction" is a long name for a familiar job: a sequence goes in, and a different sequence comes out. Translation is the main example. "The dog sat on the mat" goes in, and "Der Hund saß auf der Matte" comes out. The two sentences can have different lengths and a different word order.
>
> The first sentence describes the standard machine for this job in 2017. An **encoder** reads the input sentence and turns it into numbers. A **decoder** reads those numbers and writes the output sentence, one word at a time. Both halves were recurrent or convolutional networks. The [Background page](background-the-road-to-the-transformer.md) explains both kinds.
>
> The second sentence names the best version of that machine. Its decoder does not only get one summary of the input. At each output word, it looks back at all the input words and decides which ones matter now. That look-back is the attention step, and it came from translation research in 2014 and 2015.

> [!DEFINITION] Encoder and decoder
> The two halves of a translation model. The **encoder** reads the input sentence and makes one vector for each input word. The **decoder** uses those vectors to write the output sentence, one word at a time. Each new word depends on the input and on the words that the decoder already wrote.

> [!DEFINITION] Token
> One unit of text that a model reads or writes. It is often a whole word, but it can also be a piece of a word or a punctuation mark. Our translation model below splits "Matte" into the two tokens "Matt" and "e".

> [!PAPER] · Abstract
> [![The middle of the abstract, with the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions, more parallelizable and significantly less time to train highlighted](/img/papers/transformer/p1-abstract-2.png)](/img/papers/transformer/p1-abstract-2.png)
>
> **What it says:** the paper proposes "the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely". On two translation tasks, the models are "superior in quality while being more parallelizable and requiring significantly less time to train".
>
> **What it means:**
>
> This is the big idea in one sentence. Keep the encoder and the decoder. Keep attention. Remove the recurrent layers and the convolutions. Attention is no longer a bridge between two recurrent networks. It is the material of the whole building.
>
> The second sentence makes two claims, and both are surprising. Usually, a faster model is a worse model, and you pay for speed with quality. Here the paper claims better translations and a shorter training time at the same moment.
>
> "More parallelizable" is the reason for the speed. A recurrent network must finish word 1 before it can start word 2. Attention has no such order, so a GPU can process all the words of a sentence at the same time. The next section of this page shows why, and our run measures it.

> [!DEFINITION] Parallel computation
> Work that a computer can do at the same time on many cores, because no piece of the work waits for another piece. A GPU has thousands of small cores. It is fast only when the work can be split in this way. Work in which each step needs the result of the step before it is **sequential**: more cores do not make it faster.

> [!PAPER] · Abstract
> [![The results sentences of the abstract, with 28.4 BLEU, by over 2 BLEU, 41.8, 3.5 days on eight GPUs and small fraction of the training costs highlighted](/img/papers/transformer/p1-abstract-3.png)](/img/papers/transformer/p1-abstract-3.png)
>
> **What it says:** the model gets "28.4 BLEU on the WMT 2014 English-to-German translation task", better than the best earlier results, "including ensembles, by over 2 BLEU". On English-to-French, it gets "41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature".
>
> **What it means:**
>
> **BLEU** is the score that translation research uses (the yellow box below explains it). WMT 2014 is a public competition with fixed training data and fixed test sentences, so all teams measure on the same sentences. A gain of 2 BLEU on a mature benchmark is a large step.
>
> The word "ensembles" matters. An **ensemble** is several models that vote on each translation. Ensembles score higher than one model, but they cost several times more to train and to run. A single Transformer beat the earlier ensembles on English-to-German.
>
> The last claim is about cost. Eight GPUs for 3.5 days is a modest budget for a model that sets a new record. [Part 6](part-6-results-and-impact.md) reads Table 2 of the paper, which compares the training cost of each model in floating-point operations.

> [!DEFINITION] BLEU
> A score from 0 to 100 for a machine translation. It counts the word sequences of 1, 2, 3 and 4 words in the output that also occur in a human translation. Outputs that are too short lose points. Take the output "the cat sat on mat" and the human reference "the cat sat on the mat". All 5 single words occur in the reference, and 3 of the 4 pairs occur ("on mat" does not). Higher BLEU means more overlap with the human translation.

> [!PAPER] · Abstract
> [![The last sentence of the abstract, with generalizes well and English constituency parsing highlighted](/img/papers/transformer/p1-abstract-4.png)](/img/papers/transformer/p1-abstract-4.png)
>
> **What it says:** "We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data."
>
> **What it means:**
>
> A new model that wins on one task can be a lucky fit for that task. So the paper tries a second task that is very different from translation. In **constituency parsing**, the input is a sentence and the output is its grammar tree. The tree shows which words form a phrase, and how the phrases nest.
>
> "Limited training data" is the hard test. Large models with few examples often memorise the examples and fail on new sentences. The paper reports that the Transformer did well in both settings. Part 6 reads the numbers. This sentence was the first hint that the design is general, and the years after 2017 confirmed it.

> [!DEFINITION] Constituency parsing
> The task of finding the grammar tree of a sentence. For "The cat sat on the mat", the tree groups "The cat" as a noun phrase and "on the mat" as a prepositional phrase. It then groups "sat on the mat" as the verb phrase. A model can write the tree as a sequence of brackets, so parsing is also a sequence transduction task.

### A released Transformer, translating one sentence

The paper's own weights are not on the public model hubs. So we use a released model with the same shapes as the paper's base model: Helsinki-NLP/opus-mt-en-de, a Transformer (MarianMT) that translates English into German. Our script loads it and translates one sentence:

```text
== 1. A released Transformer: Helsinki-NLP/opus-mt-en-de
encoder layers 6, decoder layers 6, d_model 512, heads 8, d_ff 2048, vocabulary 58,101
parameters: 74,410,496
English: The dog sat on the mat because it was tired.
source tokens (12): ['▁The', '▁dog', '▁sat', '▁on', '▁the', '▁mat', '▁because', '▁it', '▁was', '▁tired', '.', '</s>']
German : Der Hund saß auf der Matte, weil er müde war.
output tokens (15): ['<pad>', '▁Der', '▁Hund', '▁saß', '▁auf', '▁der', '▁Matt', 'e', ',', '▁weil', '▁er', '▁müde', '▁war', '.', '</s>']
```

The shapes match the base model of the paper. It has 6 encoder layers and 6 decoder layers, with vectors of 512 numbers. It has 8 attention heads and 2,048 numbers inside each feed-forward layer. Part 2 explains each of these numbers. The English sentence becomes 12 tokens. The mark "▁" means "a space comes before this token", and `</s>` marks the end of the sentence.

The translation is correct, and one detail shows real understanding. In German, "Hund" (dog) is masculine and "Matte" (mat) is feminine. So "it" becomes "er" for the dog and "sie" for the mat. The model wrote "er": it linked "it" with "dog", six words earlier, and not with the closer "mat". This kind of link between distant words is what the rest of this page is about.

This model is not the paper's model. Marian was trained later, on different data, with its own vocabulary. It shows the design and the shapes, not the paper's BLEU scores.

## The problem with recurrence {§1}

> [!PAPER] · §1
> [![The first paragraph of the Introduction, with recurrent neural networks, long short-term memory, gated recurrent and language modeling and machine translation highlighted](/img/papers/transformer/p1-intro-1.png)](/img/papers/transformer/p1-intro-1.png)
>
> **What it says:** "Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation."
>
> **What it means:**
>
> The introduction starts with the opponent. In 2017, the recurrent network was the standard tool for text. The **LSTM** (1997) and the **GRU** (2014) are improved recurrent networks that remember information over more words. Google's own translation service used a large LSTM model in 2016.
>
> "Firmly established" tells you how big the claim of the title is. The paper does not attack a weak method. It attacks the method that every strong system of the time used. To replace it, the new model must be better on the same tasks, and the paper must show it with numbers.

> [!DEFINITION] Recurrent neural network (RNN)
> A network that reads a sequence one element at a time. It keeps a vector called the **hidden state**, its memory of everything that it has read. At each step, it combines the old hidden state with the new word and makes the next hidden state. The **LSTM** and the **GRU** are recurrent networks with gates: small learned switches that decide what to keep and what to forget.

> [!PAPER] · §1
> [![The second paragraph of the Introduction, with the hidden states h_t as a function of h_(t-1) and the input for position t, the sequential nature that precludes parallelization within training examples, and the fundamental constraint of sequential computation highlighted](/img/papers/transformer/p1-intro-2.png)](/img/papers/transformer/p1-intro-2.png)
>
> **What it says:** recurrent models "generate a sequence of hidden states $$h_t$$, as a function of the previous hidden state $$h_{t-1}$$ and the input for position $$t$$". "This inherently sequential nature precludes parallelization within training examples." Tricks have made recurrent models faster, but "the fundamental constraint of sequential computation, however, remains."
>
> **What it means:**
>
> This paragraph is the heart of the paper's argument. Think of a line of people who pass a message by whispering. Person 5 cannot start before person 4 finishes. A hundred people in the line need a hundred turns, and more people do not shorten the line.
>
> A recurrent network is that line. To compute the state of word 5, it needs the state of word 4. So a sentence of 50 words needs 50 steps, one after the other, even on a GPU with thousands of cores. "Within training examples" is the exact point. The GPU can still work on many sentences at the same time, but not on the words of one sentence.
>
> Why not just process more sentences at the same time? The paper answers: "memory constraints limit batching across examples". Each sentence in a batch needs memory for all its hidden states. Long sentences fill the memory of the GPU quickly. So the number of sentences in a batch is limited, and the GPU waits.

> [!DEFINITION] Batch
> A group of training examples that the network processes together, in one pass. A GPU is fast with large batches, because it can work on all the examples at the same time. The size of a batch is limited by the memory of the GPU.

The paper writes the recurrence in words. In symbols, it is this:

$$
h_t = f(h_{t-1},\, x_t)
$$

Read it from left to right. $$h_t$$ is the hidden state after word $$t$$, a vector. $$h_{t-1}$$ is the hidden state after the word before. $$x_t$$ is the vector of word $$t$$. $$f$$ is a function with learned weights, the same function at every step. The arrow of time is inside the formula: you cannot compute $$h_t$$ before you have $$h_{t-1}$$.

The most basic recurrent network uses this $$f$$:

$$
h_t = \tanh\left(W h_{t-1} + U x_t\right)
$$

Here $$W$$ and $$U$$ are learned matrices, and $$\tanh$$ squeezes each number into the range from -1 to 1. To see the problem with long sentences, shrink everything to one number. Let $$W = 0.5$$, $$U = 1$$ and $$h_0 = 0$$. The first word has the value 1, and every later word has the value 0. Our script follows the trace of the first word:

```text
== 2. The recurrence h_t = tanh(w * h_(t-1) + u * x_t), with w = 0.5, u = 1.0, h_0 = 0
  t=1: x=1  h = tanh(0.5 x 0.0000 + 1.0 x 1) = tanh(1.0000) = 0.7616
  t=2: x=0  h = tanh(0.5 x 0.7616 + 1.0 x 0) = tanh(0.3808) = 0.3634
  t=3: x=0  h = tanh(0.5 x 0.3634 + 1.0 x 0) = tanh(0.1817) = 0.1797
  t=4: x=0  h = tanh(0.5 x 0.1797 + 1.0 x 0) = tanh(0.0899) = 0.0896
  t=5: x=0  h = tanh(0.5 x 0.0896 + 1.0 x 0) = tanh(0.0448) = 0.0448
  t=6: x=0  h = tanh(0.5 x 0.0448 + 1.0 x 0) = tanh(0.0224) = 0.0224
  the trace of word 1 shrinks by about half at each step: it must pass through every later state
```

Check the first line by hand: $$0.5 \times 0 + 1 \times 1 = 1$$, and $$\tanh(1) = 0.7616$$. At each later step, the old value is multiplied by 0.5, so the trace of word 1 halves. After five more words, only 0.0224 is left. Real networks have learned weights and gates, and the LSTM was designed to slow this fading. But two facts stay: word 1 reaches word 6 only through all the states between them, and the six steps must run in order.

{{FIG:p1_loop|Top: a recurrent network computes the states h1 to h6 one after the other, so the first word reaches h6 only through five later states. Bottom: self-attention computes six new vectors together, and each one reads all six words directly. The highlighted lines show the six links of the last word.}}

### A loop against one attention step

How much does the order cost on real hardware? We timed one GRU layer and one self-attention layer, both with vectors of 512 numbers, on batches of 32 random sentences. The GRU must loop over the words. The attention layer handles all the words in one step of matrix products.

```text
== 3. One GRU layer (a loop over the words) against one self-attention layer (all words at once)
   width 512, batch 32, device mps (Apple M5 Pro laptop GPU), forward pass only, fastest of 20 runs
 words n  GRU steps   GRU ms  attention steps  attention ms  GRU / attention
      16         16      2.3                1           0.7             3.3x
      64         64      6.8                1           2.0             3.4x
     256        256     18.4                1           4.8             3.8x
    1024       1024     75.5                1          39.1             1.9x
check: the library GRU equals our step-by-step loop, max difference 3.3e-07
```

The GRU needs $$n$$ steps in order; the attention layer needs one. On our laptop GPU, the attention layer was 1.9 to 3.8 times faster at every length.

The last line checks that the library GRU is the loop that we wrote by hand. The two results differ by less than one part in a million.

This table needs two warnings. First, attention is not free. Each word compares itself with every other word, so the work grows with $$n^2$$, and the gap gets smaller at 1,024 words. [Part 4](part-4-why-self-attention.md) counts this cost exactly.

Second, these are single layers with random weights, timed on a laptop while other jobs ran. The ratio jumps from run to run. The table shows the shape of the problem, not the paper's training speed-up.

### Training is parallel, writing is not

There is a subtle point that the abstract does not say. The Transformer's decoder still writes a translation one token at a time, because each new word depends on the words before it. The parallel speed-up is in **training**. During training, the correct German sentence is known in advance, so the model can predict every position at the same time. This trick is **teacher forcing**.

```text
== 4. Marian: score a known German sentence in one call, or write it one token per call
German tokens m = 14
teacher forcing: 1 decoder call gives logits of shape (1, 14, 58101) (one row per position), 67 ms
  the 14 predicted tokens equal the generated ones: True; mean loss 0.203
generation: 14 decoder calls, one per token, 146 ms
```

With the known German sentence of 14 tokens, one call of the decoder gives 14 rows of scores, one row for each position. Each row has 58,101 scores, one for each token in the vocabulary. The best token in each row is exactly the token that the model generated. Writing the same sentence from nothing needed 14 calls, one after the other. A recurrent decoder needs the loop in both cases; the Transformer needs it only when it writes.

> [!DEFINITION] Teacher forcing
> A way to train a decoder. At each position, the decoder gets the correct earlier words, not its own guesses, and it predicts the next word. Because all the inputs are known, a Transformer can compute the predictions for all positions in one pass. A mask stops each position from looking at later words ([Part 3](part-3-attention.md) shows it).

> [!PAPER] · §1
> [![The third paragraph of the Introduction, with without regard to their distance and used in conjunction with a recurrent network highlighted](/img/papers/transformer/p1-intro-3.png)](/img/papers/transformer/p1-intro-3.png)
>
> **What it says:** attention mechanisms allow "modeling of dependencies without regard to their distance in the input or output sequences". But "in all but a few cases", they "are used in conjunction with a recurrent network".
>
> **What it means:**
>
> Here is the good news that sits next to the bad news. In a recurrent network, word 1 reaches word 30 only through 29 states, and the signal fades on the way. Attention has no such road. When the decoder writes a word, it can look straight at any input word, the first one or the last one, at the same cost.
>
> A **dependency** is a link between two words that the model must get right. "It" depends on "dog" in our example, because the dog decides the gender of "er". Some dependencies span many words, for example a verb at the end of a long German sentence.
>
> The last sentence names the strange situation of 2017. The strong part, attention, was used as an addition to the weak part, the recurrent network. The loop was still there, and so was its speed limit. The next paragraph draws the obvious conclusion.

> [!PAPER] · §1
> [![The last paragraph of the Introduction, with eschewing recurrence, relying entirely on an attention mechanism, global dependencies and twelve hours on eight P100 GPUs highlighted](/img/papers/transformer/p1-intro-4.png)](/img/papers/transformer/p1-intro-4.png)
>
> **What it says:** the Transformer is "a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output". It reaches a new state of the art "after being trained for as little as twelve hours on eight P100 GPUs".
>
> **What it means:**
>
> "Eschewing" means "keeping away from". The paper removes the loop completely. Every link between words, inside the input, inside the output and between the two, goes through attention. "Global" means that each word can reach every other word in the sentence, not only its neighbours.
>
> Twelve hours on eight GPUs is the training time of the smaller base model (§5.2 gives it, and [Part 5](part-5-training.md) reads it). The P100 was a strong NVIDIA data-centre GPU in 2016. Earlier top systems, such as Google's LSTM translation model of 2016, trained for days on many more GPUs. So the claim is not "slightly faster". It is a model that a single machine can train overnight.

## The models that came closest {§2}

The second section of the paper is short. It names the models that had already tried to remove the loop, and it says why they were not enough.

> [!PAPER] · §2
> [![The first paragraph of the Background section, with Extended Neural GPU, ByteNet, ConvS2S, computing hidden representations in parallel, linearly for ConvS2S and logarithmically for ByteNet, constant number of operations, reduced effective resolution and Multi-Head Attention highlighted](/img/papers/transformer/p1-background-1.png)](/img/papers/transformer/p1-background-1.png)
>
> **What it says:** the Extended Neural GPU, ByteNet and ConvS2S use convolutional networks, "computing hidden representations in parallel for all input and output positions". But the number of operations to relate two positions "grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet". In the Transformer, it is "a constant number of operations", at the cost of "reduced effective resolution", which "Multi-Head Attention" counteracts.
>
> **What it means:**
>
> Convolutional models had already solved the speed problem. A convolution looks at a small window of neighbouring words, and it can process all the windows of a sentence at the same time. So these models trained in parallel, like the Transformer.
>
> Their weakness is reach. One layer with a window of 3 words links each word only with its close neighbours. To connect two words far apart, the signal must climb through many layers, like news that travels from village to village. ConvS2S needs more layers in proportion to the distance. ByteNet spaces its windows wider in each layer, so it needs fewer layers, but the number still grows. The paper cites earlier work: the longer this path, the harder it is to learn the link.
>
> In the Transformer, any two words are one attention step apart. The cost is a blur. Attention mixes many words into one weighted average, and an average loses detail. Imagine that "it" needs "dog" for its gender and "tired" for its meaning. One average of the two is neither. The fix is several attention steps side by side, each free to look at a different word: **multi-head attention**, which [Part 3](part-3-attention.md) explains.

> [!DEFINITION] Convolutional layer (for text)
> A layer that slides a small window along the sentence, for example 3 words wide. At each place, it computes a weighted sum of the vectors in the window with learned weights. All the places can be computed at the same time. A **dilated** convolution skips words. Its window covers, for example, every second or every fourth word. So it reaches further with the same number of weights.

How many layers does a signal climb? Take a causal convolution of width $$k$$, as in ConvS2S's decoder: each layer reaches $$k - 1$$ more words back. With $$L$$ layers, the reach is $$L(k-1)$$ words. Now take ByteNet-style dilated layers of width 2, with gaps of 1, 2, 4, 8 words: the reach after $$L$$ layers is $$1 + 2 + 4 + \dots + 2^{L-1} = 2^L - 1$$. To link two words $$d$$ positions apart, the number of layers is:

$$
L_{\text{conv}} = \left\lceil \frac{d}{k-1} \right\rceil, \qquad L_{\text{dilated}} = \left\lceil \log_2 (d+1) \right\rceil, \qquad L_{\text{attention}} = 1
$$

The brackets $$\lceil \cdot \rceil$$ round up to the next whole number. A worked case: $$d = 10$$ and $$k = 3$$ give $$\lceil 10 / 2 \rceil = 5$$ convolution layers and $$\lceil \log_2 11 \rceil = \lceil 3.46 \rceil = 4$$ dilated layers. A recurrent network needs 10 steps. Our script prints the table for four distances:

```text
== 5. Layers a signal crosses between two words d positions apart
   ConvS2S: kernel width 3, each layer reaches 2 more words back; ByteNet: kernel width 2, dilations 1, 2, 4, ...
    d  recurrent  ConvS2S  ByteNet  self-attention
    2          2        1        2               1
   10         10        5        4               1
   50         50       25        6               1
  200        200      100        8               1
```

At 200 words apart, a recurrent network needs 200 steps and a ConvS2S-style stack needs 100 layers. The dilated stack needs 8. Self-attention needs 1 at every distance. These are idealised counts for one layout of each model; the real ConvS2S and ByteNet use other widths. [Part 4](part-4-why-self-attention.md) reads the paper's own table of these costs, Table 1.

> [!PAPER] · §2
> [![The self-attention paragraph of the Background section, with relating different positions of a single sequence and compute a representation of the sequence highlighted](/img/papers/transformer/p1-background-2.png)](/img/papers/transformer/p1-background-2.png)
>
> **What it says:** "Self-attention, sometimes called intra-attention is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence."
>
> **What it means:**
>
> In translation attention, the decoder looks at the encoder: a German word looks at English words. That is attention between two sequences. **Self-attention** turns the look inward. The words of one sentence look at the other words of the same sentence.
>
> Why look inward? Because the meaning of a word depends on its neighbours. "Bank" next to "river" is not "bank" next to "money". After one step of self-attention, the vector of "bank" carries some of the vector of "river". It becomes a vector for "bank in this sentence". That is what "a representation of the sequence" means: one vector for each word, in its context.
>
> The idea was not new. The paper lists earlier uses in reading comprehension, summaries and sentence vectors. What was new is the next paragraph.

> [!DEFINITION] Self-attention
> Attention in which the words of one sequence attend to the words of the same sequence. Each word gets a new vector: a weighted average of the vectors of all the words, with the weights computed from the words themselves. The encoder of the Transformer uses only self-attention to read the input.

> [!PAPER] · §2
> [![The last paragraph of the Background section, with first transduction model relying entirely on self-attention and without using sequence-aligned RNNs or convolution highlighted](/img/papers/transformer/p1-background-3.png)](/img/papers/transformer/p1-background-3.png)
>
> **What it says:** "the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution."
>
> **What it means:**
>
> This sentence states what is new, and it is careful. Attention was not new. Self-attention was not new. What was new is a full translation model, encoder and decoder, with no recurrent layer and no convolution at all.
>
> "Sequence-aligned" is the precise word for the loop. A recurrent network aligns its computation with the order of the words: step 1 for word 1, step 2 for word 2. The Transformer drops this alignment.
>
> But word order still matters in language, and attention by itself ignores it. "dog bites man" and "man bites dog" contain the same words. So the model must get the order from somewhere else. [Part 2](part-2-the-architecture.md) shows the answer: position encodings, added to each word vector.

> [!TAKEAWAYS] Key takeaways
> - **The claim:** a translation model needs no recurrent layers and no convolutions. Attention alone can move all the information between words.
> - **Sequence transduction** turns one sequence into another, for example English into German. An encoder reads the input, and a decoder writes the output one token at a time.
> - **The problem with recurrence:** $$h_t = f(h_{t-1}, x_t)$$. Each state waits for the one before it, so the words of one sentence cannot be processed in parallel. In our one-number example, the trace of word 1 fell from 0.7616 to 0.0224 in six steps.
> - **On our laptop GPU,** one self-attention layer was 1.9 to 3.8 times faster than one GRU layer for 16 to 1,024 words. Attention's work grows with $$n^2$$, which Part 4 counts.
> - **Training is parallel, writing is not:** with teacher forcing, Marian scored a 14-token German sentence in 1 decoder call. Writing it took 14 calls.
> - **Path length:** between two words 200 positions apart, a recurrent network needs 200 steps, a width-3 convolution stack 100 layers and a dilated stack 8. Self-attention needs 1.
> - **The price** of one-step links is a blur from averaging. Multi-head attention fixes it.
> - **The results:** 28.4 BLEU on WMT 2014 English-to-German, over 2 BLEU above the best earlier results, ensembles included. 41.8 BLEU on English-to-French after 3.5 days on 8 GPUs.
> - **A released model with the paper's base shapes** (opus-mt-en-de: 6 + 6 layers, d_model 512, 8 heads, d_ff 2048) has 74.4 million parameters. It chose "er" for "it", which links "it" to the dog.

**Next, in Part 2:** §3 and Figure 1 of the paper, the architecture. Part 2 opens the encoder and the decoder and follows each shape through the released model. It covers the residual connections, layer normalisation, the feed-forward layers, the embeddings and the position encodings.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/transformer/tx_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/transformer/tx_part1.py). It loads the released translation model and translates one sentence. It works the recurrence by hand and times a GRU layer against an attention layer. It also compares teacher forcing with generation, and prints the path lengths. Every number is saved in `results/part1.json`.

```bash
pip install torch transformers sentencepiece
python tx_part1.py        # downloads Helsinki-NLP/opus-mt-en-de (about 300 MB)
```

<figure><img src="/img/papers/transformer/part1-run.png" alt="Terminal output of tx_part1.py: the shapes of the released model and its translation, the recurrence worked by hand, the GRU and attention timing table, teacher forcing against generation, and the path-length table" loading="lazy" /><figcaption>The full output of tx_part1.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017) · [official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
