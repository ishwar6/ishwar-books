---
title: "Inside BERT: Architecture and Input"
description: "Section 2 and the model half of Section 3, line by line: the ideas BERT builds on, the stack of Transformer layers, where the 110 million parameters come from (counted in the real model), WordPiece tokens, [CLS], [SEP], and the three embeddings that are added together for every token."
part: 2
covers: "§2, §3 (model, input)"
date: 2026-10-04
tags: [bert, nlp, transformers]
---

In Part 1 we read the paper's big idea: pre-train a Transformer that looks at both sides of every word, then fine-tune it. This part opens the box. First we read Section 2, the history BERT builds on. Then the first half of Section 3: what the model is made of, how big it is, and how text is turned into the numbers it reads.

Everything that can be checked, we check on the real released model, `bert-base-uncased`.

## Related work: a long history of pre-training {§2}

> [!PAPER] Devlin et al. (2018), BERT · Section 2 · page 2
> [![The opening of Section 2, Related Work: there is a long history of pre-training general language representations, and we briefly review the most widely-used approaches in this section](/img/papers/bert/p2-related-intro.png)](/img/papers/bert/p2-related-intro.png)
>
> **Context:** right after the introduction. Before describing BERT, the authors place it in the history of the field.
>
> **What it says:** "There is a long history of pre-training general language representations." The section reviews the most used approaches, in three groups.
>
> **Why it matters:** BERT is not one sudden invention. It combines ideas that already worked: learned word vectors, contextual vectors, fine-tuning, and transfer learning. Knowing them makes clear what is actually new.

The three groups are: **feature-based** approaches (§2.1), **fine-tuning** approaches (§2.2), and **transfer from labelled data** (§2.3). We take them one at a time.

### Word embeddings {§2.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 2.1 · page 2
> [![Section 2.1, Unsupervised Feature-based Approaches: learning widely applicable representations of words has been an active area of research for decades. Pre-trained word embeddings are an integral part of modern NLP systems, offering significant improvements over embeddings learned from scratch](/img/papers/bert/p2-feature-words.png)](/img/papers/bert/p2-feature-words.png)
>
> **Context:** the first paragraph of Section 2.1.
>
> **What it says:** people have learned vectors for words for decades, first without neural networks and then with them (Mikolov et al., 2013, known as word2vec; Pennington et al., 2014, known as GloVe). "Pre-trained word embeddings" give "significant improvements over embeddings learned from scratch".
>
> **Why it matters:** this is the oldest form of pre-training: learn one vector per word from a lot of text, then reuse those vectors in other models.

> [!DEFINITION] Word embedding
> A fixed list of numbers for each word in a vocabulary, learned from a large amount of text so that words used in similar ways get similar lists. word2vec (2013) and GloVe (2014) are the two most famous methods.

> [!DEFINITION] Learned from scratch
> Starting from random numbers and learning only from the task's own (usually small) labelled data. The opposite of starting from pre-trained numbers.

A word embedding is a **lookup table**: one row per word. The word "bank" always gets the same row, whether it is a river bank or a money bank. Keep this weakness in mind; the next two paragraphs are about fixing it.

### Sentence and paragraph embeddings {§2.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 2.1 · page 2
> [![These approaches have been generalized to coarser granularities, such as sentence embeddings or paragraph embeddings. Prior work trained sentence representations by ranking candidate next sentences, generating the next sentence, or with denoising auto-encoder objectives](/img/papers/bert/p2-feature-sentences.png)](/img/papers/bert/p2-feature-sentences.png)
>
> **Context:** the second paragraph of Section 2.1.
>
> **What it says:** the same idea was "generalized to coarser granularities": one vector for a whole **sentence** or **paragraph**. To learn them, earlier work ranked candidate next sentences, generated the next sentence's words, or used "denoising auto-encoder" objectives.
>
> **Why it matters:** two of these ideas come back in BERT. Guessing **which sentence comes next** returns as next sentence prediction (Part 3), and **repairing a damaged input** is close to masked language modelling.

> [!DEFINITION] Denoising auto-encoder
> A model that is given a damaged input (some words deleted or swapped) and is trained to rebuild the clean original. BERT's masked language model is similar, but it only has to rebuild the hidden words, not the whole input.

### ELMo: a word's vector depends on its sentence {§2.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 2.1 · page 2
> [![The ELMo paragraph: ELMo extracts context-sensitive features from a left-to-right and a right-to-left language model. The contextual representation of each token is the concatenation of the left-to-right and right-to-left representations. Melamud et al. 2016 predicted a word from both sides with LSTMs, but their model is feature-based and not deeply bidirectional](/img/papers/bert/p2-elmo.png)](/img/papers/bert/p2-elmo.png)
>
> **Context:** the third paragraph of Section 2.1, about the strongest feature-based model of 2018.
>
> **What it says:** ELMo extracts "context-sensitive features" from a left-to-right and a right-to-left language model. Each token's vector "is the concatenation of the left-to-right and right-to-left representations". Plugged into existing task models, ELMo set new records in question answering, sentiment analysis and named entity recognition. Melamud et al. (2016, called context2vec) predicted a word from both sides with LSTMs, but, like ELMo, it is "not deeply bidirectional". Fedus et al. (2018) used the Cloze task to improve text generation.
>
> **Why it matters:** ELMo fixed the "bank" problem: the vector of a word now depends on its sentence. BERT keeps that, and makes the two directions mix in every layer instead of only at the end.

> [!DEFINITION] Contextual embedding
> A word vector that is computed fresh for each sentence, from the words around it. The same word in two sentences gets two different vectors. ELMo and BERT both produce contextual embeddings.

> [!DEFINITION] Sentiment analysis
> Deciding whether a piece of text is positive or negative, for example a movie review.

We can see the difference between a static and a contextual vector in the real BERT. BERT has both kinds inside it: its first step is a lookup table (a static embedding, one row per token), and its output is a contextual vector. I took the word "bank" in four sentences, two about rivers and two about money, and compared the vectors with **cosine similarity**.

> [!DEFINITION] Cosine similarity
> A number from -1 to 1 that says how much two vectors point the same way. 1 means the same direction; values near 0 mean unrelated. It ignores the length of the vectors and only looks at their direction.

```python
import torch, torch.nn.functional as F
from transformers import BertModel, BertTokenizer

tok = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased").eval()

def bank_vector(sentence):
    enc = tok(sentence, return_tensors="pt")
    pos = enc.input_ids[0].tolist().index(tok.vocab["bank"])   # where "bank" is
    return model(**enc).last_hidden_state[0, pos]              # BERT's output vector for it (768 numbers)

a = bank_vector("he sat on the river bank and watched the water.")
b = bank_vector("she opened a savings account at the bank.")
print(F.cosine_similarity(a, b, dim=0))
```

The real output, for all four sentences:

```text
the input (static) vector of "bank" is the same row of the table in every sentence: cosine 1.000
cosine similarity of BERT's output vectors for "bank":
[1] 1.000  0.747  0.477  0.432   he sat on the river bank and watched the water.
[2] 0.747  1.000  0.458  0.479   the bank of the river was covered in mud.
[3] 0.477  0.458  1.000  0.786   she opened a savings account at the bank.
[4] 0.432  0.479  0.786  1.000   the bank approved my loan yesterday.
average, same meaning (river-river, money-money): 0.766
average, different meaning (river-money):         0.462
```

<figure class="fig"><svg viewBox="0 0 760 380" role="img" aria-label="Cosine similarity between BERT output vectors for the word bank in four sentences. The two river sentences are close to each other, the two money sentences are close to each other, and river versus money pairs are much less similar."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="386.0" y="28.0" text-anchor="middle">cosine similarity of the output vectors of "bank"</text><text class="t-tick" x="262.0" y="73.0" text-anchor="end">[1] river bank</text><g class="mark"><title>[1] river bank → [1] river bank: 1.00</title><rect class="cell" x="271.0" y="41.0" width="56.0" height="56.0" rx="2" style="fill-opacity:1.000"/></g><text class="t-cell on" x="299.0" y="73.0" text-anchor="middle">1.00</text><g class="mark"><title>[1] river bank → [2] bank of river: 0.75</title><rect class="cell" x="329.0" y="41.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.747"/></g><text class="t-cell on" x="357.0" y="73.0" text-anchor="middle">0.75</text><g class="mark"><title>[1] river bank → [3] savings bank: 0.48</title><rect class="cell" x="387.0" y="41.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.477"/></g><text class="t-cell" x="415.0" y="73.0" text-anchor="middle">0.48</text><g class="mark"><title>[1] river bank → [4] bank loan: 0.43</title><rect class="cell" x="445.0" y="41.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.432"/></g><text class="t-cell" x="473.0" y="73.0" text-anchor="middle">0.43</text><text class="t-tick" x="262.0" y="131.0" text-anchor="end">[2] bank of river</text><g class="mark"><title>[2] bank of river → [1] river bank: 0.75</title><rect class="cell" x="271.0" y="99.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.747"/></g><text class="t-cell on" x="299.0" y="131.0" text-anchor="middle">0.75</text><g class="mark"><title>[2] bank of river → [2] bank of river: 1.00</title><rect class="cell" x="329.0" y="99.0" width="56.0" height="56.0" rx="2" style="fill-opacity:1.000"/></g><text class="t-cell on" x="357.0" y="131.0" text-anchor="middle">1.00</text><g class="mark"><title>[2] bank of river → [3] savings bank: 0.46</title><rect class="cell" x="387.0" y="99.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.458"/></g><text class="t-cell" x="415.0" y="131.0" text-anchor="middle">0.46</text><g class="mark"><title>[2] bank of river → [4] bank loan: 0.48</title><rect class="cell" x="445.0" y="99.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.479"/></g><text class="t-cell" x="473.0" y="131.0" text-anchor="middle">0.48</text><text class="t-tick" x="262.0" y="189.0" text-anchor="end">[3] savings bank</text><g class="mark"><title>[3] savings bank → [1] river bank: 0.48</title><rect class="cell" x="271.0" y="157.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.477"/></g><text class="t-cell" x="299.0" y="189.0" text-anchor="middle">0.48</text><g class="mark"><title>[3] savings bank → [2] bank of river: 0.46</title><rect class="cell" x="329.0" y="157.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.458"/></g><text class="t-cell" x="357.0" y="189.0" text-anchor="middle">0.46</text><g class="mark"><title>[3] savings bank → [3] savings bank: 1.00</title><rect class="cell" x="387.0" y="157.0" width="56.0" height="56.0" rx="2" style="fill-opacity:1.000"/></g><text class="t-cell on" x="415.0" y="189.0" text-anchor="middle">1.00</text><g class="mark"><title>[3] savings bank → [4] bank loan: 0.79</title><rect class="cell" x="445.0" y="157.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.786"/></g><text class="t-cell on" x="473.0" y="189.0" text-anchor="middle">0.79</text><text class="t-tick" x="262.0" y="247.0" text-anchor="end">[4] bank loan</text><g class="mark"><title>[4] bank loan → [1] river bank: 0.43</title><rect class="cell" x="271.0" y="215.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.432"/></g><text class="t-cell" x="299.0" y="247.0" text-anchor="middle">0.43</text><g class="mark"><title>[4] bank loan → [2] bank of river: 0.48</title><rect class="cell" x="329.0" y="215.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.479"/></g><text class="t-cell" x="357.0" y="247.0" text-anchor="middle">0.48</text><g class="mark"><title>[4] bank loan → [3] savings bank: 0.79</title><rect class="cell" x="387.0" y="215.0" width="56.0" height="56.0" rx="2" style="fill-opacity:0.786"/></g><text class="t-cell on" x="415.0" y="247.0" text-anchor="middle">0.79</text><g class="mark"><title>[4] bank loan → [4] bank loan: 1.00</title><rect class="cell" x="445.0" y="215.0" width="56.0" height="56.0" rx="2" style="fill-opacity:1.000"/></g><text class="t-cell on" x="473.0" y="247.0" text-anchor="middle">1.00</text><text class="t-tick" x="299.0" y="282.0" text-anchor="end" transform="rotate(-50 299.0 282.0)">[1] river bank</text><text class="t-tick" x="357.0" y="282.0" text-anchor="end" transform="rotate(-50 357.0 282.0)">[2] bank of river</text><text class="t-tick" x="415.0" y="282.0" text-anchor="end" transform="rotate(-50 415.0 282.0)">[3] savings bank</text><text class="t-tick" x="473.0" y="282.0" text-anchor="end" transform="rotate(-50 473.0 282.0)">[4] bank loan</text><text class="t-tick" x="530.0" y="70.0" text-anchor="start">river vs river: 0.747</text><text class="t-tick" x="530.0" y="92.0" text-anchor="start">money vs money: 0.786</text><text class="t-tick" x="530.0" y="114.0" text-anchor="start">river vs money: 0.432 to 0.479</text><text class="t-note" x="530.0" y="160.0" text-anchor="start">The input (static) vector</text><text class="t-note" x="530.0" y="180.0" text-anchor="start">of "bank" is one fixed row</text><text class="t-note" x="530.0" y="200.0" text-anchor="start">of the table: similarity</text><text class="t-note" x="530.0" y="220.0" text-anchor="start">1.000 for every pair.</text></svg><figcaption>Cosine similarity of BERT's output vectors for "bank". The two river sentences are close (0.747), the two money sentences are close (0.786), and every river and money pair is far apart (0.432 to 0.479). A static word embedding would give 1.000 everywhere.</figcaption></figure>

The static row of the table cannot tell the two banks apart: it is literally the same vector. After 12 layers of looking at the neighbours, BERT's output vectors split cleanly into a "river" group and a "money" group. This is what "contextual" means, and it is what made ELMo, and then BERT, so useful.

**Use case.** A search engine that knows "bank" in "fishing spots on the river bank" is not about money will not show loan offers for that query.

### Fine-tuning approaches {§2.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 2.2 · pages 2 and 3
> [![Section 2.2, Unsupervised Fine-tuning Approaches: the first works only pre-trained word embedding parameters. More recently, sentence or document encoders have been pre-trained from unlabeled text and fine-tuned for a supervised downstream task. The advantage is that few parameters need to be learned from scratch. OpenAI GPT achieved previously state-of-the-art results on many sentence-level tasks from the GLUE benchmark](/img/papers/bert/p2-finetune.png)](/img/papers/bert/p2-finetune.png)
> [![The paragraph continues on page 3: left-to-right language modeling and auto-encoder objectives have been used for pre-training such models](/img/papers/bert/p2-finetune2.png)](/img/papers/bert/p2-finetune2.png)
>
> **Context:** Section 2.2, the family BERT belongs to. It runs from page 2 onto page 3.
>
> **What it says:** the first works (Collobert and Weston, 2008) "only pre-trained word embedding parameters". Later, whole encoders were pre-trained on unlabeled text and "fine-tuned for a supervised downstream task" (Dai and Le, 2015; Howard and Ruder, 2018; Radford et al., 2018). The advantage: "few parameters need to be learned from scratch". This is partly why OpenAI GPT held the previous best results on many GLUE tasks. They were pre-trained with left-to-right language modelling and auto-encoder objectives.
>
> **Why it matters:** BERT takes this recipe unchanged (pre-train everything, fine-tune everything) and only changes what happens during pre-training.

> [!DEFINITION] Supervised learning
> Learning from labelled examples: inputs paired with the right answers. Fine-tuning on a task is supervised. Pre-training on plain text is called **unsupervised** (or self-supervised), because the "answers" come from the text itself.

The key phrase is "**few parameters need to be learned from scratch**". With the feature-based approach, the whole task model above the features starts from random numbers and must learn from the task's small labelled set. With fine-tuning, almost every weight starts pre-trained; only a tiny output layer is new. Less to learn from scratch means less labelled data is needed. Howard and Ruder's 2018 paper (ULMFiT) showed this works well for text classification with LSTMs; GPT showed it with a Transformer.

### Transfer learning from labelled data {§2.3}

> [!PAPER] Devlin et al. (2018), BERT · Section 2.3 · page 3
> [![Section 2.3, Transfer Learning from Supervised Data: effective transfer from supervised tasks with large datasets, such as natural language inference and machine translation. Computer vision has shown the importance of transfer learning from large pre-trained models, where an effective recipe is to fine-tune models pre-trained with ImageNet](/img/papers/bert/p2-supervised.png)](/img/papers/bert/p2-supervised.png)
>
> **Context:** the last part of Related Work.
>
> **What it says:** models can also be pre-trained on a big **labelled** task and transferred: natural language inference (Conneau et al., 2017) and machine translation (McCann et al., 2017). In computer vision, "an effective recipe is to fine-tune models pre-trained with ImageNet".
>
> **Why it matters:** the ImageNet recipe was the model to copy. In vision, nobody trained an image classifier from zero anymore; everyone started from a network pre-trained on ImageNet. BERT brought that habit to language, but with **unlabeled** text, which is far more plentiful than any labelled dataset.

> [!DEFINITION] Transfer learning
> Reusing what a model learned on one task to do better on another task. Pre-training and fine-tuning is one form of transfer learning.

> [!DEFINITION] ImageNet
> A large collection of labelled photos (Deng et al., 2009) used to train image classifiers. A network pre-trained on ImageNet became the standard starting point for almost any vision task.

> [!DEFINITION] Machine translation (MT)
> Translating text from one language to another automatically.

## BERT in two steps, with one architecture {§3}

Now Section 3, the description of BERT itself. It opens with the two steps and with the paper's main picture, Figure 1.

> [!PAPER] Devlin et al. (2018), BERT · Section 3 · page 3
> [![The opening of Section 3: there are two steps in our framework, pre-training and fine-tuning. During pre-training the model is trained on unlabeled data over different pre-training tasks. For fine-tuning, the model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks. Each downstream task has separate fine-tuned models](/img/papers/bert/p2-bert-intro.png)](/img/papers/bert/p2-bert-intro.png)
>
> **Context:** the first paragraph of Section 3.
>
> **What it says:** "There are two steps in our framework: pre-training and fine-tuning." For fine-tuning, BERT starts from the pre-trained parameters and "all of the parameters are fine-tuned using labeled data from the downstream tasks". "Each downstream task has separate fine-tuned models", all starting from the same pre-trained parameters.
>
> **Why it matters:** this fixes the vocabulary for the rest of the paper. One pre-training run, then one separate copy per task.

> [!DEFINITION] Initialized with
> The numbers a model starts from before training. "Initialized with the pre-trained parameters" means the copy starts as an exact clone of pre-trained BERT, not from random numbers.

> [!PAPER] Devlin et al. (2018), BERT · Section 3, Figure 1 · page 3
> [![Figure 1 of the BERT paper. Left, pre-training: an unlabeled sentence A and B pair goes in as [CLS], the tokens of masked sentence A, [SEP], and the tokens of masked sentence B. BERT turns input embeddings E into outputs C and T; C feeds NSP and the T vectors feed Mask LM. Right, fine-tuning: the same BERT is copied for MNLI, NER and SQuAD; for SQuAD a question and a paragraph go in and start and end span outputs come out](/img/papers/bert/p2-figure1.png)](/img/papers/bert/p2-figure1.png)
>
> **Context:** Figure 1 sits at the top of page 3. The question-answering example in it is the "running example" for Section 3.
>
> **What it says:** "Apart from output layers, the same architectures are used in both pre-training and fine-tuning." "During fine-tuning, all parameters are fine-tuned." `[CLS]` is added in front of every input, and `[SEP]` separates two pieces of text, such as a question and its answer.
>
> **Why it matters:** this one picture shows the whole system. Learn to read it and the rest of the paper is easy to follow.

How to read Figure 1, from bottom to top:

- **Bottom (pink):** the input tokens. `[CLS]`, then the tokens of sentence A (Tok 1 to Tok N), then `[SEP]`, then the tokens of sentence B (Tok 1 to Tok M).
- **Yellow:** **E**, the input embedding of each token. We build these by hand later in this part.
- **Blue box:** BERT itself, the stack of layers. Every position is connected to every other (the faint lines).
- **Green (top):** the outputs. **C** is the output for `[CLS]`. **T₁, …, T_N** and **T₁′, …, T_M′** are the outputs for the other tokens.
- **Red arrows (top):** the output layers. In pre-training, C feeds **NSP** (next sentence prediction) and the T vectors feed **Mask LM**. In fine-tuning, the output layer depends on the task: for **SQuAD** (question answering), the T vectors of the paragraph predict where the answer starts and ends.

> [!DEFINITION] MNLI, NER, SQuAD
> Three of the downstream tasks in Figure 1. **MNLI** (MultiNLI): natural language inference on sentence pairs. **NER**: named entity recognition, a label for every token. **SQuAD**: question answering, find the answer span in a paragraph. Part 4 covers all of them.

> [!PAPER] Devlin et al. (2018), BERT · Section 3 · page 3
> [![A distinctive feature of BERT is its unified architecture across different tasks](/img/papers/bert/p2-unified.png)](/img/papers/bert/p2-unified.png)
> [![There is minimal difference between the pre-trained architecture and the final downstream architecture. Then the Model Architecture paragraph: a multi-layer bidirectional Transformer encoder based on the original implementation of Vaswani et al. 2017, almost identical to the original](/img/papers/bert/p2-architecture.png)](/img/papers/bert/p2-architecture.png)
>
> **Context:** the second paragraph of Section 3 (it starts at the bottom of the left column and ends at the top of the right column), followed by the "Model Architecture" paragraph.
>
> **What it says:** "A distinctive feature of BERT is its unified architecture across different tasks." There is "minimal difference between the pre-trained architecture and the final downstream architecture". The model is "a multi-layer bidirectional Transformer encoder" based on Vaswani et al. (2017), released in the `tensor2tensor` library, and "almost identical to the original".
>
> **Why it matters:** BERT invents no new kind of layer. The architecture is the standard 2017 Transformer encoder. Everything new is in how it is trained.

## The model: a stack of Transformer encoder layers {§3}

The paper skips the details of the Transformer and points to Vaswani et al. (2017) and the guide "The Annotated Transformer". Here is the short version, enough to follow the rest of the paper.

> [!DEFINITION] Layer (Transformer block)
> One repeated unit of the model. Each layer takes one vector per token and returns one improved vector per token, of the same size. BERT stacks 12 or 24 identical layers, each with its own weights.

> [!DEFINITION] Hidden size
> How many numbers are in each token's vector inside the model. BERT-base uses 768. It stays the same from the first layer to the last.

> [!DEFINITION] Self-attention head
> One attention calculation: every token compares itself with every other token, decides how much to look at each, and mixes in their information. A layer runs several heads side by side; each can learn a different pattern (one may follow the previous word, another may link a pronoun to its noun). See the [attention series](/writings/attention-1-self-attention/) for the full story.

Every BERT layer has the same two parts:

1. **Self-attention** with A heads: each token gathers information from the other tokens.
2. **A feed-forward network**: each token's vector is processed on its own by two matrix multiplications with a non-linear step between them.

After each part there is an **add-and-normalise** step: the part's output is added to its input (a "residual connection"), then normalised with **LayerNorm**. Written as equations, for the vectors $$h$$ of all tokens entering a layer:

$$
h' = \text{LayerNorm}\big(h + \text{MultiHeadAttention}(h)\big)
$$

$$
\text{output} = \text{LayerNorm}\big(h' + \text{FFN}(h')\big), \qquad \text{FFN}(x) = \text{GELU}(x W_1 + b_1)\, W_2 + b_2
$$

where:

- $$h$$ holds one $$H$$-number vector per token (the input of the layer), and $$h'$$ is the result after attention;
- $$\text{MultiHeadAttention}$$ runs $$A$$ attention heads in parallel and joins their results;
- $$W_1$$ is an $$H \times 4H$$ matrix and $$W_2$$ a $$4H \times H$$ matrix, with bias vectors $$b_1$$ and $$b_2$$: the feed-forward network makes each vector four times wider, then narrows it back;
- $$\text{GELU}$$ is a smooth version of "keep positive numbers, set negative ones to zero" (Part 3 shows its shape);
- $$\text{LayerNorm}$$ rescales each vector to have mean 0 and spread 1, then applies a learned scale and shift.

> [!DEFINITION] Residual connection
> Adding a block's input to its output: output = input + block(input). The block only has to learn a correction. This keeps deep stacks of layers trainable.

> [!DEFINITION] LayerNorm (layer normalisation)
> A step that rescales each token's vector so its numbers have average 0 and spread 1, then multiplies and shifts them by two learned vectors. It keeps the numbers in a stable range from layer to layer.

<figure class="fig"><svg viewBox="0 0 760 410" role="img" aria-label="The BERT-base encoder. Tokens become input embeddings, pass up through 12 identical layers of self-attention and feed-forward, and come out as one 768-number vector per token: C for [CLS] and T for the others."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><g transform="translate(-60,0)"><rect class="box-2" x="150.0" y="360.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="185.0" y="379.5" text-anchor="middle">[CLS]</text><rect class="box" x="230.0" y="360.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="265.0" y="379.5" text-anchor="middle">my</text><rect class="box" x="310.0" y="360.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="345.0" y="379.5" text-anchor="middle">dog</text><rect class="box" x="390.0" y="360.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="425.0" y="379.5" text-anchor="middle">is</text><rect class="box" x="470.0" y="360.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="505.0" y="379.5" text-anchor="middle">cute</text><rect class="box-2" x="550.0" y="360.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="585.0" y="379.5" text-anchor="middle">[SEP]</text><text class="t-muted" x="140.0" y="380.0" text-anchor="end">tokens</text><rect class="box-4" x="140.0" y="300.0" width="490.0" height="34.0" rx="8"/><text class="t-tick" x="385.0" y="322.0" text-anchor="middle">input embeddings: token + segment + position (768 numbers each)</text><line class="edge" x1="185.0" y1="358.0" x2="185.0" y2="336.0" marker-end="url(#ah)"/><line class="edge" x1="265.0" y1="358.0" x2="265.0" y2="336.0" marker-end="url(#ah)"/><line class="edge" x1="345.0" y1="358.0" x2="345.0" y2="336.0" marker-end="url(#ah)"/><line class="edge" x1="425.0" y1="358.0" x2="425.0" y2="336.0" marker-end="url(#ah)"/><line class="edge" x1="505.0" y1="358.0" x2="505.0" y2="336.0" marker-end="url(#ah)"/><line class="edge" x1="585.0" y1="358.0" x2="585.0" y2="336.0" marker-end="url(#ah)"/><rect class="box-1" x="140.0" y="236.0" width="490.0" height="44.0" rx="10"/><text class="t-note" x="150.0" y="254.0" text-anchor="start">layer 1</text><rect class="box" x="240.0" y="244.0" width="180.0" height="28.0" rx="6"/><text class="t-tick" x="330.0" y="262.0" text-anchor="middle">self-attention, 12 heads</text><rect class="box" x="432.0" y="244.0" width="188.0" height="28.0" rx="6"/><text class="t-tick" x="526.0" y="262.0" text-anchor="middle">feed-forward 768→3072→768</text><rect class="box-1" x="140.0" y="176.0" width="490.0" height="44.0" rx="10"/><text class="t-note" x="150.0" y="194.0" text-anchor="start">layer 2</text><rect class="box" x="240.0" y="184.0" width="180.0" height="28.0" rx="6"/><text class="t-tick" x="330.0" y="202.0" text-anchor="middle">self-attention, 12 heads</text><rect class="box" x="432.0" y="184.0" width="188.0" height="28.0" rx="6"/><text class="t-tick" x="526.0" y="202.0" text-anchor="middle">feed-forward 768→3072→768</text><rect class="box-1" x="140.0" y="116.0" width="490.0" height="44.0" rx="10"/><text class="t-note" x="150.0" y="134.0" text-anchor="start">layer 12</text><rect class="box" x="240.0" y="124.0" width="180.0" height="28.0" rx="6"/><text class="t-tick" x="330.0" y="142.0" text-anchor="middle">self-attention, 12 heads</text><rect class="box" x="432.0" y="124.0" width="188.0" height="28.0" rx="6"/><text class="t-tick" x="526.0" y="142.0" text-anchor="middle">feed-forward 768→3072→768</text><line class="edge" x1="385.0" y1="298.0" x2="385.0" y2="282.0" marker-end="url(#ah)"/><line class="edge" x1="385.0" y1="234.0" x2="385.0" y2="222.0" marker-end="url(#ah)"/><text class="t-muted" x="150.0" y="172.5" text-anchor="start">(9 more layers in between)</text><rect class="box-3" x="150.0" y="66.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="185.0" y="85.5" text-anchor="middle">C</text><rect class="box-3" x="230.0" y="66.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="265.0" y="85.5" text-anchor="middle">T₁</text><rect class="box-3" x="310.0" y="66.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="345.0" y="85.5" text-anchor="middle">T₂</text><rect class="box-3" x="390.0" y="66.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="425.0" y="85.5" text-anchor="middle">T₃</text><rect class="box-3" x="470.0" y="66.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="505.0" y="85.5" text-anchor="middle">T₄</text><rect class="box-3" x="550.0" y="66.0" width="70.0" height="30.0" rx="6"/><text class="t-tick" x="585.0" y="85.5" text-anchor="middle">T₅</text><text class="t-muted" x="140.0" y="86.0" text-anchor="end">outputs</text><line class="edge" x1="185.0" y1="114.0" x2="185.0" y2="98.0" marker-end="url(#ah)"/><line class="edge" x1="265.0" y1="114.0" x2="265.0" y2="98.0" marker-end="url(#ah)"/><line class="edge" x1="345.0" y1="114.0" x2="345.0" y2="98.0" marker-end="url(#ah)"/><line class="edge" x1="425.0" y1="114.0" x2="425.0" y2="98.0" marker-end="url(#ah)"/><line class="edge" x1="505.0" y1="114.0" x2="505.0" y2="98.0" marker-end="url(#ah)"/><line class="edge" x1="585.0" y1="114.0" x2="585.0" y2="98.0" marker-end="url(#ah)"/></g><text class="t-note" x="590.0" y="84.0" text-anchor="start">H = 768</text><text class="t-tick" x="590.0" y="102.0" text-anchor="start">numbers per token</text><text class="t-tick" x="590.0" y="120.0" text-anchor="start">at every level</text><text class="t-note" x="590.0" y="186.0" text-anchor="start">L = 12 layers</text><text class="t-tick" x="590.0" y="204.0" text-anchor="start">stacked</text><text class="t-note" x="590.0" y="250.0" text-anchor="start">A = 12 heads</text><text class="t-tick" x="590.0" y="268.0" text-anchor="start">in every layer</text><text class="t-title" x="20.0" y="30.0" text-anchor="start">BERT-base: L = 12, H = 768, A = 12</text></svg><figcaption>The BERT-base encoder. Tokens become input embeddings, go up through 12 identical layers (each with 12-head self-attention and a feed-forward network), and come out as one 768-number vector per token: C for [CLS] and T for every other token.</figcaption></figure>

To check that this short description is complete, I recomputed layer 1 of the real `bert-base-uncased` by hand, using only its weights and the equations above:

```python
x = model.embeddings(input_ids=ids, token_type_ids=seg)[0]     # (10 tokens, 768)
lay = model.encoder.layer[0]
A, d = 12, 64                                                  # 12 heads of 64 numbers
q = lay.attention.self.query(x).view(-1, A, d).transpose(0, 1) # (12, 10, 64)
k = lay.attention.self.key(x).view(-1, A, d).transpose(0, 1)
v = lay.attention.self.value(x).view(-1, A, d).transpose(0, 1)
att = torch.softmax(q @ k.transpose(1, 2) / d ** 0.5, dim=-1)  # no mask: both directions
heads = (att @ v).transpose(0, 1).reshape(-1, 768)             # join the 12 heads
h1 = lay.attention.output.LayerNorm(x + lay.attention.output.dense(heads))
out = lay.output.LayerNorm(h1 + lay.output.dense(F.gelu(lay.intermediate.dense(h1))))
```

```text
== 7. layer 1 of BERT, recomputed by hand ==
  12 heads of 64 numbers each; every attention row sums to 1: True
  max |difference|, our layer vs the model's layer 1: 8.34e-07
```

The largest difference from the model's own layer is $$8.34 \times 10^{-7}$$, the size of normal rounding errors in 32-bit arithmetic. So the equations above are the whole layer: attention, add and normalise, feed-forward, add and normalise.

## Two model sizes: L, H and A {§3}

> [!PAPER] Devlin et al. (2018), BERT · Section 3 · page 3
> [![We denote the number of layers as L, the hidden size as H, and the number of self-attention heads as A. BERT-BASE: L=12, H=768, A=12, Total Parameters=110M. BERT-LARGE: L=24, H=1024, A=16, Total Parameters=340M. BERT-BASE was chosen to have the same model size as OpenAI GPT. BERT uses bidirectional self-attention, while the GPT Transformer uses constrained self-attention where every token can only attend to context to its left](/img/papers/bert/p2-sizes.png)](/img/papers/bert/p2-sizes.png)
>
> **Context:** the second paragraph of "Model Architecture".
>
> **What it says:** **L** is the number of layers, **H** the hidden size, **A** the number of attention heads. Two sizes: **BERT-base** (L=12, H=768, A=12, 110M parameters) and **BERT-large** (L=24, H=1024, A=16, 340M parameters). BERT-base "was chosen to have the same model size as OpenAI GPT for comparison purposes". The one critical difference: "bidirectional self-attention" in BERT versus "constrained self-attention where every token can only attend to context to its left" in GPT.
>
> **Why it matters:** because BERT-base and GPT have the same size, a fair comparison is possible. When BERT-base wins, the size is not the reason.

| | Layers L | Hidden size H | Heads A | Numbers per head H/A | Feed-forward size 4H | Parameters (paper) |
|---|---|---|---|---|---|---|
| BERT-base | 12 | 768 | 12 | 64 | 3,072 | 110M |
| BERT-large | 24 | 1,024 | 16 | 64 | 4,096 | 340M |

> [!DEFINITION] Parameter
> One learned number inside the model (a weight or a bias). "110M parameters" means about 110 million such numbers, all set by training.

The feed-forward size comes from a footnote:

> [!PAPER] Devlin et al. (2018), BERT · Section 3, footnote 3 · page 3
> [![Footnote 3: In all cases we set the feed-forward/filter size to be 4H, i.e., 3072 for the H = 768 and 4096 for the H = 1024](/img/papers/bert/p2-footnote3.png)](/img/papers/bert/p2-footnote3.png)
>
> **Context:** footnote 3, attached to the definition of L, H and A.
>
> **What it says:** the feed-forward ("filter") size is always 4H: 3,072 for BERT-base and 4,096 for BERT-large.
>
> **Why it matters:** with this number we can compute the exact parameter count ourselves, which we do next.

And footnote 4 explains two names you will see everywhere:

> [!PAPER] Devlin et al. (2018), BERT · Section 3, footnote 4 · pages 3 and 4
> [![Footnote 4, first line: We note that in the literature the bidirectional Trans-](/img/papers/bert/p2-footnote4a.png)](/img/papers/bert/p2-footnote4a.png)
> [![Footnote 4, continued on page 4: former is often referred to as a Transformer encoder while the left-context-only version is referred to as a Transformer decoder since it can be used for text generation](/img/papers/bert/p2-footnote4.png)](/img/papers/bert/p2-footnote4.png)
>
> **Context:** footnote 4, attached to the sentence about GPT's "constrained self-attention". It starts on page 3 and ends on page 4.
>
> **What it says:** the bidirectional Transformer "is often referred to as a **Transformer encoder**", and the left-context-only version "as a **Transformer decoder** since it can be used for text generation".
>
> **Why it matters:** this is why BERT is called an *encoder* model and GPT a *decoder* model. The layers are almost the same; the difference is the attention mask.

The mask is easiest to see as a grid. Each row is a token doing the looking; each column a token being looked at.

<figure class="fig"><svg viewBox="0 0 760 362" role="img" aria-label="Attention masks. In BERT every row is full: each token may attend to all tokens. In GPT the upper triangle is blocked, so each token attends only to itself and tokens on its left."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="245.0" y="26.0" text-anchor="middle">BERT: every token sees every token</text><text class="t-tick" x="120.0" y="84.0" text-anchor="end">[CLS]</text><rect class="box-1" x="126.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="166.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="206.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="246.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="286.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="326.0" y="60.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="120.0" y="124.0" text-anchor="end">my</text><rect class="box-1" x="126.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="166.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="206.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="246.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="286.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="326.0" y="100.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="120.0" y="164.0" text-anchor="end">dog</text><rect class="box-1" x="126.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="166.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="206.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="246.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="286.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="326.0" y="140.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="120.0" y="204.0" text-anchor="end">is</text><rect class="box-1" x="126.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="166.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="206.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="246.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="286.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="326.0" y="180.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="120.0" y="244.0" text-anchor="end">cute</text><rect class="box-1" x="126.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="166.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="206.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="246.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="286.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="326.0" y="220.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="120.0" y="284.0" text-anchor="end">[SEP]</text><rect class="box-1" x="126.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="166.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="206.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="246.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="286.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="326.0" y="260.0" width="36.0" height="36.0" rx="4"/><text class="t-muted" x="246.0" y="322.0" text-anchor="middle">columns: the token being looked at</text><text class="t-title" x="615.0" y="26.0" text-anchor="middle">GPT: each token sees only its left</text><text class="t-tick" x="490.0" y="84.0" text-anchor="end">[CLS]</text><rect class="box-1" x="496.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="536.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="576.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="616.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="656.0" y="60.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="696.0" y="60.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="490.0" y="124.0" text-anchor="end">my</text><rect class="box-1" x="496.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="536.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="576.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="616.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="656.0" y="100.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="696.0" y="100.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="490.0" y="164.0" text-anchor="end">dog</text><rect class="box-1" x="496.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="536.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="576.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="616.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="656.0" y="140.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="696.0" y="140.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="490.0" y="204.0" text-anchor="end">is</text><rect class="box-1" x="496.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="536.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="576.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="616.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="656.0" y="180.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="696.0" y="180.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="490.0" y="244.0" text-anchor="end">cute</text><rect class="box-1" x="496.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="536.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="576.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="616.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="656.0" y="220.0" width="36.0" height="36.0" rx="4"/><rect class="box-ghost" x="696.0" y="220.0" width="36.0" height="36.0" rx="4"/><text class="t-tick" x="490.0" y="284.0" text-anchor="end">[SEP]</text><rect class="box-1" x="496.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="536.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="576.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="616.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="656.0" y="260.0" width="36.0" height="36.0" rx="4"/><rect class="box-1" x="696.0" y="260.0" width="36.0" height="36.0" rx="4"/><text class="t-muted" x="616.0" y="322.0" text-anchor="middle">columns: the token being looked at</text><text class="t-tick" x="20.0" y="348.0" text-anchor="start">Rows: the token doing the looking. Filled square: allowed. Dashed square: blocked by the mask.</text></svg><figcaption>Attention masks. In BERT (left) every square is allowed: every token can attend to every token. In GPT (right) the upper triangle is blocked, so each token attends only to itself and to the tokens on its left. That one triangle is the difference between an encoder and a decoder.</figcaption></figure>

A model that can only look left can **write** text one token at a time, because it never needs the future. A model that looks both ways can **understand** text better, but cannot simply write the next word. That trade-off is why BERT is used for understanding tasks and GPT-style models for generation.

### Where "110 million" comes from {§3}

The paper gives the parameter counts as rounded totals. We can rebuild them from L, H, A, the 4H footnote and the vocabulary size. Every count below is checked against the real model.

**Embeddings.** Three lookup tables of $$H$$ numbers per row, plus one LayerNorm (a scale and a shift, $$2H$$):

$$
\underbrace{30{,}522 \times 768}_{\text{tokens}} + \underbrace{512 \times 768}_{\text{positions}} + \underbrace{2 \times 768}_{\text{segments}} + \underbrace{2 \times 768}_{\text{LayerNorm}} = 23{,}837{,}184
$$

**One layer.** Attention has four $$H \times H$$ matrices (query, key, value, output), each with a bias of $$H$$. The feed-forward network has an $$H \times 4H$$ matrix with bias $$4H$$ and a $$4H \times H$$ matrix with bias $$H$$. Two LayerNorms add $$4H$$:

$$
\underbrace{4(H^2 + H)}_{\text{attention}} + \underbrace{2 \cdot 4H^2 + 4H + H}_{\text{feed-forward}} + \underbrace{4H}_{\text{2 LayerNorms}} = 2{,}362{,}368 + 4{,}722{,}432 + 3{,}072 = 7{,}087{,}872
$$

where $$H = 768$$. Twelve layers give $$12 \times 7{,}087{,}872 = 85{,}054{,}464$$.

**Pooler.** One more $$H \times H$$ matrix with a bias, $$H^2 + H = 590{,}592$$ (more on it at the end of this part).

**Total:** $$23{,}837{,}184 + 85{,}054{,}464 + 590{,}592 = 109{,}482{,}240$$.

The script computes the same formula and also counts every parameter in the downloaded model. For BERT-large, it builds the model's shape from its configuration file on PyTorch's "meta" device, which creates the layers without storing any weights, so nothing big is downloaded:

```python
from transformers import BertConfig, BertModel
import torch

c = BertConfig.from_pretrained("bert-large-uncased")   # only the small config file
with torch.device("meta"):                             # shapes only, no memory used for weights
    large = BertModel(c)
print(sum(p.numel() for p in large.parameters()))
```

```text
bert-base-uncased: L=12 H=768 A=12 feed-forward=3072
  embeddings   23,837,184   one layer   7,087,872   x12 =   85,054,464   pooler   590,592
  formula total  109,482,240   counted in the model  109,482,240   same: True
  without the pooler 108,891,648
bert-large-uncased: L=24 H=1024 A=16 feed-forward=4096
  embeddings   31,782,912   one layer  12,596,224   x24 =  302,309,376   pooler 1,049,600
  formula total  335,141,888   counted in the model  335,141,888   same: True
  without the pooler 334,092,288
```

<figure class="fig"><svg viewBox="0 0 760 240" role="img" aria-label="Parameters of BERT-base by part: embeddings 23.84 million, attention 28.35 million, feed-forward 56.67 million, layer norms 0.04 million, pooler 0.59 million."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="40.0" text-anchor="start">Where the 109,482,240 parameters of BERT-base live</text><text class="t-tick" x="202.0" y="71.0" text-anchor="end">embeddings</text><g class="mark"><title>embeddings: 23.84 M</title><rect class="s1" x="210.0" y="53.0" width="151.0" height="26.0" rx="3"/></g><text class="t-val" x="367.0" y="71.0" text-anchor="start">23.84 M</text><text class="t-tick" x="202.0" y="105.0" text-anchor="end">attention, 12 layers</text><g class="mark"><title>attention, 12 layers: 28.35 M</title><rect class="s1" x="210.0" y="87.0" width="179.5" height="26.0" rx="3"/></g><text class="t-val" x="395.5" y="105.0" text-anchor="start">28.35 M</text><text class="t-tick" x="202.0" y="139.0" text-anchor="end">feed-forward, 12 layers</text><g class="mark"><title>feed-forward, 12 layers: 56.67 M</title><rect class="s1" x="210.0" y="121.0" width="358.9" height="26.0" rx="3"/></g><text class="t-val" x="574.9" y="139.0" text-anchor="start">56.67 M</text><text class="t-tick" x="202.0" y="173.0" text-anchor="end">layer norms, 12 layers</text><g class="mark"><title>layer norms, 12 layers: 0.04 M</title><rect class="s1" x="210.0" y="155.0" width="2.0" height="26.0" rx="3"/></g><text class="t-val" x="218.0" y="173.0" text-anchor="start">0.04 M</text><text class="t-tick" x="202.0" y="207.0" text-anchor="end">pooler</text><g class="mark"><title>pooler: 0.59 M</title><rect class="s1" x="210.0" y="189.0" width="3.7" height="26.0" rx="3"/></g><text class="t-val" x="219.7" y="207.0" text-anchor="start">0.59 M</text></svg><figcaption>Where BERT-base's 109,482,240 parameters live. The feed-forward networks hold about half (56.67 million), attention about a quarter (28.35 million), the embedding tables about a fifth (23.84 million).</figcaption></figure>

So the formula and the real models agree exactly: **109,482,240** for BERT-base and **335,141,888** for BERT-large. The paper's "110M" and "340M" are both what you get by rounding these to the nearest 10 million (109.5 → 110, 335.1 → 340). The paper does not say how it rounded, so treat this as arithmetic, not as the authors' stated method.

Two things stand out in the picture:

- **The feed-forward networks are the biggest part**, about 52% of all parameters. Making them 4H wide is expensive.
- **The token table alone is about a fifth** of BERT-base: $$30{,}522 \times 768 = 23{,}440{,}896$$ numbers just to look up one vector per token. Later models (ALBERT, in Part 6) attack exactly this.

## Input: sentences and sequences {§3}

> [!PAPER] Devlin et al. (2018), BERT · Section 3, Input/Output Representations · page 4
> [![Input/Output Representations: the input representation can unambiguously represent both a single sentence and a pair of sentences in one token sequence. A sentence can be an arbitrary span of contiguous text, rather than an actual linguistic sentence. A sequence refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together](/img/papers/bert/p2-input.png)](/img/papers/bert/p2-input.png)
>
> **Context:** the first paragraph of "Input/Output Representations", at the top of page 4.
>
> **What it says:** one input format must handle both a single sentence and a pair, such as ⟨Question, Answer⟩. In this paper, "a 'sentence' can be an arbitrary span of contiguous text, rather than an actual linguistic sentence". A "sequence" is the whole input: one sentence, or two packed together.
>
> **Why it matters:** two words with special meanings for the rest of the paper. A "sentence" may be a whole paragraph. A "sequence" is what BERT actually reads.

> [!DEFINITION] Contiguous
> Next to each other, without gaps. A contiguous span of text is a piece cut out in one go, from some word to some later word.

## WordPiece: how text becomes tokens {§3}

> [!PAPER] Devlin et al. (2018), BERT · Section 3, Input/Output Representations · page 4
> [![We use WordPiece embeddings with a 30,000 token vocabulary. The first token of every sequence is always a special classification token [CLS], whose final hidden state is used as the aggregate sequence representation for classification tasks. Sentence pairs are separated with a special token [SEP], and a learned embedding is added to every token indicating whether it belongs to sentence A or sentence B. The input embedding is E, the final hidden vector of [CLS] is C in R to the H, and the final hidden vector of the i-th token is T i in R to the H](/img/papers/bert/p2-wordpiece.png)](/img/papers/bert/p2-wordpiece.png)
>
> **Context:** the second paragraph of "Input/Output Representations". It is short but dense: five ideas in nine lines.
>
> **What it says:** (1) "WordPiece embeddings (Wu et al., 2016) with a 30,000 token vocabulary". (2) Every sequence starts with `[CLS]`, whose final vector is "the aggregate sequence representation for classification tasks". (3) Two sentences are separated by `[SEP]`. (4) A "learned embedding" on every token says whether it belongs to sentence A or B. (5) Names: **E** for input embeddings, $$C \in \mathbb{R}^H$$ for the final vector of `[CLS]`, $$T_i \in \mathbb{R}^H$$ for the final vector of token $$i$$.
>
> **Why it matters:** these are the symbols used in every formula from here to the end of the paper.

We take the five ideas one at a time, starting with WordPiece.

> [!DEFINITION] Vocabulary
> The fixed list of all tokens a model knows, each with an id number. Any text must be cut into pieces from this list before the model can read it.

> [!DEFINITION] WordPiece
> A way to cut words into pieces from a fixed vocabulary of about 30,000 entries (Wu et al., 2016, from Google's translation system). Common words are one piece. Rare words are split into smaller pieces; a piece that continues a word starts with `##`. "embeddings" becomes `em ##bed ##ding ##s`. No word is ever "unknown" as long as its characters are in the vocabulary.

Why not just use whole words? Because there are too many: names, typos, technical terms, word forms. A whole-word vocabulary would either be enormous or would map many words to "unknown". Why not single letters? Because sequences would get very long. WordPiece is the middle path: about 30,000 pieces cover everything.

The released tokenizer cuts each word with a simple rule, which its code comment calls "a greedy longest-match-first algorithm": take the longest piece at the start of the word that is in the vocabulary, then repeat on the rest with a `##` in front. Here it is in a few lines of Python:

```python
def wordpiece(word, vocab, unk="[UNK]"):
    pieces, start = [], 0
    while start < len(word):
        end = len(word)
        while end > start:                       # try the longest piece first
            piece = ("##" if start > 0 else "") + word[start:end]
            if piece in vocab:
                break
            end -= 1
        if end == start:                         # not even one character matched
            return [unk]
        pieces.append(piece)
        start = end
    return pieces
```

Before this step, the real tokenizer lowercases the text (this is the "uncased" model: "`Uncased` means that the text has been lowercased before WordPiece tokenization", says the release README), strips accents, and splits on spaces and punctuation. I ran our function on every word of pages 1 to 9 of the BERT paper and compared with the real tokenizer:

```text
our WordPiece function vs the real tokenizer, pages 1-9 of the BERT paper:
  8,525 words -> 9,878 pieces (ours), 9,878 pieces (real); identical: True
  distinct words that needed more than one piece: 416
```

Identical, piece for piece. (One detail: the paper's text contains the literal strings `[CLS]`, `[SEP]` and `[MASK]`, which the real tokenizer keeps as special tokens before WordPiece runs. The script removes their brackets so only the WordPiece step is compared.)

Some real splits:

```text
  playing         -> playing
  likes           -> likes
  embeddings      -> em ##bed ##ding ##s
  unaffable       -> una ##ffa ##ble
  tokenization    -> token ##ization
  bidirectional   -> bid ##ire ##ction ##al
```

Two honest notes. First, the pieces are chosen by frequency, not by meaning: "bidirectional" becomes `bid ##ire ##ction ##al`, which a person would never choose. Second, Figure 2 of the paper (below) shows "playing" split into `play ##ing`, but in the released uncased vocabulary "playing" is a single token. The figure illustrates the idea; it is not the output of the real tokenizer. (Even the example in the tokenizer's own code comment, "unaffable" → `un ##aff ##able`, differs from what the released vocabulary gives: `una ##ffa ##ble`.)

### The vocabulary: "30,000" is 30,522 {§3}

The paper says "a 30,000 token vocabulary". The released `vocab.txt` of `bert-base-uncased` has **30,522** lines. I sorted every entry into groups:

| Group | Entries | Examples |
|---|---|---|
| special tokens | 5 | `[PAD]` `[UNK]` `[CLS]` `[SEP]` `[MASK]` |
| unused placeholders | 994 | `[unused0]` `[unused1]` … |
| single characters | 997 | `!` `"` `#` `a` … |
| `##` + one character | 997 | `##s` `##a` `##e` … |
| `##` pieces, longer | 4,831 | `##ing` `##ed` `##er` `##ly` … |
| whole tokens of 2+ characters | 22,698 | `the` `of` `and` `in` … |
| **total** | **30,522** | |

So "30,000" is a round number. 994 entries are empty placeholder slots (`[unused0]` to `[unused993]`) that the paper does not mention, and 5 are special tokens. The special tokens have fixed ids: `[PAD]`=0, `[UNK]`=100, `[CLS]`=101, `[SEP]`=102, `[MASK]`=103.

> [!DEFINITION] [PAD] and [UNK]
> `[PAD]` fills the empty places when several sequences of different lengths are put in one batch. `[UNK]` ("unknown") stands for a piece of text that cannot be built from the vocabulary at all, for example a character the vocabulary does not contain.

## [CLS], [SEP] and the two segments {§3}

Back to the paragraph. Three special tokens shape every input:

> [!DEFINITION] [CLS]
> "Classification" token. Always the first token of every sequence. It is not a word, so it has no meaning of its own; through self-attention it can gather information from the whole sequence. Its final vector, **C**, is used for whole-sequence tasks like sentiment or next sentence prediction.

> [!DEFINITION] [SEP]
> "Separator" token. Marks the end of sentence A, and the end of sentence B if there is one.

> [!DEFINITION] Segment embedding
> A learned vector added to every token to say which sentence it belongs to: vector $$E_A$$ for every token of sentence A (including the first `[SEP]`), vector $$E_B$$ for every token of sentence B. There are only two of these vectors in the whole model.

> [!DEFINITION] R to the H
> $$\mathbb{R}^H$$ is the set of all lists of $$H$$ real numbers. $$C \in \mathbb{R}^H$$ just says C is a vector of $$H$$ numbers (768 for BERT-base).

Here is what the real tokenizer makes of the pair from Figure 2, `my dog is cute` and `he likes playing`:

```text
Figure 2 pair ("my dog is cute", "he likes playing"):
  tokens:     [CLS]      my     dog      is    cute   [SEP]      he   likes playing   [SEP]
  ids:          101    2026    3899    2003   10140     102    2002    7777    2652     102
  segment:        A       A       A       A       A       A       B       B       B       B
  position:       0       1       2       3       4       5       6       7       8       9
```

Two ways tell the sentences apart, exactly as the paper says: the `[SEP]` token between them, and the segment row (A for the first six tokens, B for the last four).

## Three embeddings, added together {§3}

> [!PAPER] Devlin et al. (2018), BERT · Section 3, Input/Output Representations · page 4
> [![For a given token, its input representation is constructed by summing the corresponding token, segment, and position embeddings. A visualization of this construction can be seen in Figure 2](/img/papers/bert/p2-sum.png)](/img/papers/bert/p2-sum.png)
>
> **Context:** the last paragraph of "Input/Output Representations".
>
> **What it says:** a token's input representation is made "by summing the corresponding token, segment, and position embeddings".
>
> **Why it matters:** this is the very first computation BERT does. Everything else builds on these vectors.

> [!PAPER] Devlin et al. (2018), BERT · Section 3, Figure 2 · page 5
> [![Figure 2 of the BERT paper: the input [CLS] my dog is cute [SEP] he likes play ##ing [SEP]. Below each token: its token embedding E with the token as subscript, plus a segment embedding E A for the first six tokens and E B for the rest, plus a position embedding E 0 to E 10](/img/papers/bert/p2-figure2.png)](/img/papers/bert/p2-figure2.png)
>
> **Context:** Figure 2, at the top of page 5, the picture for the paragraph above.
>
> **What it says:** "The input embeddings are the sum of the token embeddings, the segmentation embeddings and the position embeddings." Each column is one token; each row is one kind of embedding.
>
> **Why it matters:** three tables, one lookup in each, one addition. That is the whole input layer.

> [!DEFINITION] Position embedding
> A learned vector for each position in the sequence: one for position 0, one for position 1, and so on. Attention by itself does not know word order ("dog bites man" and "man bites dog" would look the same), so the position vector is added to tell the model where each token sits. BERT learns a table of 512 of them, which is why its inputs are at most 512 tokens long.

As an equation, the input embedding of the token at position $$i$$ is:

$$
E_i = \text{LayerNorm}\big(\, \text{Tok}[\,t_i\,] + \text{Seg}[\,s_i\,] + \text{Pos}[\,i\,] \,\big)
$$

where:

- $$t_i$$ is the token's id (for example 3899 for "dog"), and $$\text{Tok}$$ is the token table, 30,522 rows of 768 numbers;
- $$s_i$$ is the segment (A or B), and $$\text{Seg}$$ is the segment table, 2 rows of 768 numbers;
- $$i$$ is the position (0 to 511), and $$\text{Pos}$$ is the position table, 512 rows of 768 numbers;
- $$\text{Tok}[\,t_i\,]$$ means "row $$t_i$$ of the table", a simple lookup;
- $$\text{LayerNorm}$$ is the same normalisation as inside the layers.

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="BERT input representation for a real sentence pair. Each token adds three learned vectors: its token embedding, a segment embedding (A or B) and a position embedding (0 to 9). The sum is normalised and sent to layer 1."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="30.0" text-anchor="start">The real input for the pair ("my dog is cute", "he likes playing")</text><rect class="box" x="150.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="176.0" y="71.5" text-anchor="middle">[CLS]</text><rect class="box" x="208.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="234.0" y="71.5" text-anchor="middle">my</text><rect class="box" x="266.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="292.0" y="71.5" text-anchor="middle">dog</text><rect class="box" x="324.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="350.0" y="71.5" text-anchor="middle">is</text><rect class="box" x="382.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="408.0" y="71.5" text-anchor="middle">cute</text><rect class="box" x="440.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="466.0" y="71.5" text-anchor="middle">[SEP]</text><rect class="box" x="498.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="71.5" text-anchor="middle">he</text><rect class="box" x="556.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="582.0" y="71.5" text-anchor="middle">likes</text><rect class="box" x="614.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="640.0" y="71.5" text-anchor="middle">playing</text><rect class="box" x="672.0" y="52.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="698.0" y="71.5" text-anchor="middle">[SEP]</text><text class="t-note" x="140.0" y="72.0" text-anchor="end">input</text><rect class="box-4" x="150.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="176.0" y="123.5" text-anchor="middle">101</text><rect class="box-4" x="208.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="234.0" y="123.5" text-anchor="middle">2026</text><rect class="box-4" x="266.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="292.0" y="123.5" text-anchor="middle">3899</text><rect class="box-4" x="324.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="350.0" y="123.5" text-anchor="middle">2003</text><rect class="box-4" x="382.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="408.0" y="123.5" text-anchor="middle">10140</text><rect class="box-4" x="440.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="466.0" y="123.5" text-anchor="middle">102</text><rect class="box-4" x="498.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="123.5" text-anchor="middle">2002</text><rect class="box-4" x="556.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="582.0" y="123.5" text-anchor="middle">7777</text><rect class="box-4" x="614.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="640.0" y="123.5" text-anchor="middle">2652</text><rect class="box-4" x="672.0" y="104.0" width="52.0" height="30.0" rx="6"/><text class="t-tick" x="698.0" y="123.5" text-anchor="middle">102</text><text class="t-note" x="140.0" y="124.0" text-anchor="end">token id</text><rect class="box-3" x="150.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="176.0" y="175.5" text-anchor="middle">A</text><rect class="box-3" x="208.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="234.0" y="175.5" text-anchor="middle">A</text><rect class="box-3" x="266.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="292.0" y="175.5" text-anchor="middle">A</text><rect class="box-3" x="324.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="350.0" y="175.5" text-anchor="middle">A</text><rect class="box-3" x="382.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="408.0" y="175.5" text-anchor="middle">A</text><rect class="box-3" x="440.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="466.0" y="175.5" text-anchor="middle">A</text><rect class="box-1" x="498.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="524.0" y="175.5" text-anchor="middle">B</text><rect class="box-1" x="556.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="582.0" y="175.5" text-anchor="middle">B</text><rect class="box-1" x="614.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="640.0" y="175.5" text-anchor="middle">B</text><rect class="box-1" x="672.0" y="156.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="698.0" y="175.5" text-anchor="middle">B</text><text class="t-note" x="140.0" y="176.0" text-anchor="end">segment</text><text class="t-muted" x="176.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="234.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="292.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="350.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="408.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="466.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="524.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="582.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="640.0" y="150.0" text-anchor="middle">+</text><text class="t-muted" x="698.0" y="150.0" text-anchor="middle">+</text><rect class="box" x="150.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="176.0" y="227.5" text-anchor="middle">0</text><rect class="box" x="208.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="234.0" y="227.5" text-anchor="middle">1</text><rect class="box" x="266.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="292.0" y="227.5" text-anchor="middle">2</text><rect class="box" x="324.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="350.0" y="227.5" text-anchor="middle">3</text><rect class="box" x="382.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="408.0" y="227.5" text-anchor="middle">4</text><rect class="box" x="440.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="466.0" y="227.5" text-anchor="middle">5</text><rect class="box" x="498.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="524.0" y="227.5" text-anchor="middle">6</text><rect class="box" x="556.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="582.0" y="227.5" text-anchor="middle">7</text><rect class="box" x="614.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="640.0" y="227.5" text-anchor="middle">8</text><rect class="box" x="672.0" y="208.0" width="52.0" height="30.0" rx="6"/><text class="t-mono" x="698.0" y="227.5" text-anchor="middle">9</text><text class="t-note" x="140.0" y="228.0" text-anchor="end">position</text><text class="t-muted" x="176.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="234.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="292.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="350.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="408.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="466.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="524.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="582.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="640.0" y="202.0" text-anchor="middle">+</text><text class="t-muted" x="698.0" y="202.0" text-anchor="middle">+</text><text class="t-note" x="140.0" y="280.0" text-anchor="end">sum</text><rect class="box-on" x="150.0" y="260.0" width="574.0" height="30.0" rx="8"/><text class="t-tick" x="437.0" y="280.0" text-anchor="middle">add the three 768-number vectors, then LayerNorm (and dropout in training)</text></svg><figcaption>The real input for the Figure 2 pair. Each token looks up three vectors (its token id, its segment, its position), the three are added, then normalised. Note that the real tokenizer keeps "playing" as one token, so there are 10 tokens, not the 11 drawn in the paper.</figcaption></figure>

The paper says "summing". The released model also applies **LayerNorm** after the sum (and **dropout** during training), which the paper does not mention in this paragraph. To check exactly what happens, I rebuilt the input embeddings from the model's own three tables and compared them with the model's embedding layer:

```python
E = model.embeddings
s = E.word_embeddings(ids) + E.token_type_embeddings(seg) + E.position_embeddings(pos)
mine = E.LayerNorm(s)
theirs = E(input_ids=ids, token_type_ids=seg)
```

```text
== 4. input embedding = LayerNorm(token + segment + position) ==
  tables: token (30522, 768), segment (2, 768), position (512, 768)
  output shape (1, 10, 768) (1 sequence, 10 tokens, 768 numbers each)
  max |difference|, our LayerNorm(sum) vs the model: 0.00e+00
  max |difference|, plain sum without LayerNorm vs the model: 9.15
```

> [!DEFINITION] Dropout
> During training, randomly setting some numbers to zero (10% of them in BERT) so the model cannot rely too much on any single number. It is switched off when the model is used, which is why our check (done in evaluation mode) has no dropout.

With LayerNorm, the difference is exactly zero: this is the model's embedding layer, bit for bit. Without LayerNorm, values differ by up to 9.15. So the precise rule is "sum, then LayerNorm". The three tables themselves have the shapes we used in the parameter count: (30522, 768), (2, 768) and (512, 768).

A note on positions: the original Transformer (Vaswani et al., 2017) used fixed sine and cosine patterns for positions. BERT instead **learns** its position vectors; they are ordinary weights in a table of 512 rows. Appendix A.2 says the last 10% of pre-training used length 512 "to learn the positional embeddings" (Part 3 covers this).

**Use case.** Every BERT-based system, from a spam filter to a search ranker, starts with exactly these three lookups. When an input is longer than 512 tokens, there is no position vector for token 513, so long documents must be cut into pieces. This is one of BERT's practical limits (Part 6).

## The outputs: C, Tᵢ and the pooler {§3}

After 12 layers, every token has an output vector of 768 numbers: **C** for `[CLS]`, and $$T_i$$ for token $$i$$. The paper uses C for whole-sequence tasks and the T vectors for token-level tasks.

There is one honest detail here. The paper defines C as the final hidden vector of `[CLS]`. The released code adds one more small layer on top of it, called the **pooler**: a 768 × 768 matrix and a `tanh`. Its comment in `modeling.py` says:

```python
# We "pool" the model by simply taking the hidden state corresponding
# to the first token. We assume that this has been pre-trained
first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
self.pooled_output = tf.layers.dense(first_token_tensor, config.hidden_size,
                                     activation=tf.tanh, ...)   # (initializer argument shortened)
```

That is the 590,592 parameters of the "pooler" in our count. I checked the formula on the downloaded model:

```text
== 6. the [CLS] vector C and the released pooler ==
  C = final hidden vector of [CLS], shape (768,)
  pooler = tanh(W C + b), W shape (768, 768); max |difference| vs model.pooler_output: 0.00e+00
```

> [!DEFINITION] tanh
> A smooth function that squeezes any number into the range from -1 to 1. Large positive numbers become close to 1, large negative numbers close to -1.

So in the released model, $$\text{pooled} = \tanh(W C + b)$$. In the original pre-training code, the next sentence prediction layer reads this pooled vector (`model.get_pooled_output()` in `run_pretraining.py`), and so does the classifier in `run_classifier.py`. So the pooler is trained during pre-training, through next sentence prediction (Part 3). For understanding the paper, think of it as part of the small output layer on top of C.

> [!TAKEAWAYS] Key takeaways
> - BERT builds on a long line of ideas: **word embeddings** (one fixed vector per word), **contextual embeddings** like ELMo (a vector that depends on the sentence), **fine-tuning** whole pre-trained models (ULMFiT, GPT), and the **ImageNet** habit of transfer learning.
> - In real BERT, "bank" in river sentences and in money sentences gets output vectors with cosine similarity **0.43 to 0.48**, while same-meaning pairs reach **0.75 to 0.79**. The static input row is identical everywhere.
> - The architecture is the **standard Transformer encoder**: L layers, each with A-head self-attention and a 4H-wide feed-forward network, with add-and-LayerNorm after each. Our hand-written layer matched the real one to $$8.34 \times 10^{-7}$$.
> - **BERT-base**: L=12, H=768, A=12. **BERT-large**: L=24, H=1024, A=16. BERT-base matches GPT's size; the difference is the attention mask: **encoder** (both directions) versus **decoder** (left only).
> - The parameter count, by formula and by counting the real models: **109,482,240** (base) and **335,141,888** (large); the paper rounds to 110M and 340M. About half sits in the feed-forward networks.
> - Text becomes **WordPiece** tokens with greedy longest-match; our short version matched the real tokenizer on 9,878 pieces. The "30,000" vocabulary is **30,522** entries, 994 of them unused placeholders.
> - Every input is `[CLS]` sentence A `[SEP]` (sentence B `[SEP]`). Each token's input is **token + segment + position** embeddings, then LayerNorm (exactly matched: difference 0). Positions are learned, up to **512**.
> - Outputs are **C** (for `[CLS]`) and $$T_i$$ (for each token). The released code adds a small **pooler** (dense + tanh) on top of C.

**Next, in Part 3:** how BERT is pre-trained. The masked language model with its strange 80/10/10 rule, next sentence prediction, the data, and the training recipe, all run on the real model.

<details>
<summary>Run it yourself</summary>

Every number in this part comes from [`code/papers/bert/bert_part2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part2.py). It runs on a laptop CPU in under a minute. It downloads `bert-base-uncased` (about 440 MB) and only the configuration file of `bert-large-uncased`. The WordPiece check reads the BERT paper PDF from `~/.cache/papers/1810.04805.pdf`, which `paper_shots.py` downloads.

```bash
pip install torch transformers pymupdf
python bert_part2.py      # prints everything below, writes results/part2.json
```

<figure><img src="/img/papers/bert/part2-run.png" alt="Terminal output of bert_part2.py: parameter counts by formula and by the real models, the vocabulary groups, WordPiece splits and the check against the real tokenizer, the Figure 2 pair with ids, segments and positions, the embedding check, the bank cosine similarities, the pooler check and the hand-computed layer" loading="lazy" /><figcaption>The real output of bert_part2.py.</figcaption></figure>

</details>

## References

1. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019.
2. T. Mikolov et al. *Distributed representations of words and phrases and their compositionality* (word2vec). NeurIPS 2013.
3. J. Pennington, R. Socher, C. Manning. *GloVe: Global vectors for word representation*. EMNLP 2014.
4. M. Peters et al. [*Deep contextualized word representations*](https://arxiv.org/abs/1802.05365) (ELMo). NAACL 2018.
5. J. Howard, S. Ruder. *Universal language model fine-tuning for text classification* (ULMFiT). ACL 2018.
6. A. Radford et al. *Improving Language Understanding by Generative Pre-Training* (OpenAI GPT). OpenAI, 2018.
7. A. Vaswani et al. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
8. Y. Wu et al. *Google's neural machine translation system: Bridging the gap between human and machine translation* (WordPiece). arXiv:1609.08144, 2016.
9. Google Research. [BERT code and pre-trained models](https://github.com/google-research/bert): `modeling.py` (the pooler), `tokenization.py` (WordPiece), `README.md`.
