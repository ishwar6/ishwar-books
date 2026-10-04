---
title: "The Big Idea: Reading in Both Directions"
description: "The title, the abstract and the introduction of the BERT paper, line by line: what pre-training is, the two ways to reuse a pre-trained model, why reading only left to right is a real limit, and BERT's fix of filling in blanks. With a real run of GPT-2 and BERT on the same sentences."
part: 1
covers: "Title, Abstract, §1"
date: 2026-10-04
tags: [bert, nlp, transformers]
---

In October 2018, four researchers at Google posted a paper called **BERT**. It reported new best results on eleven language tasks at once. A year later, in October 2019, Google wrote that BERT would help its search engine understand about one in ten English searches in the US. BERT-style models are still used today inside search engines, spam filters, support-ticket routers and retrieval systems.

This series reads the paper slowly, in the paper's own order. Each piece follows the same pattern:

1. **The exact lines from the paper**, as a highlighted screenshot in a teal box like the one below.
2. **A plain-English explanation**, with a yellow box for every new word.
3. **A picture**, and **real code** when it helps, with its real output.
4. **Why it matters**, and only then the next piece.

The small `Paper §1` tag above each heading tells you which section of the paper you are reading. The screenshots come from the paper's second arXiv version (May 2019), the one most people read today.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word in plain language, the first time it appears. If you already know the word, skip the box.

## The title {§Title}

> [!PAPER] Devlin et al. (2018), BERT · Title · page 1
> [![The title of the BERT paper: BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding, by Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova, Google AI Language](/img/papers/bert/p1-title.png)](/img/papers/bert/p1-title.png)
>
> **Context:** the first thing on page 1. The paper was posted to arXiv in October 2018 and published at NAACL 2019, a major conference on language technology.
>
> **What it says:** "BERT: Pre-training of Deep **Bidirectional Transformers** for Language Understanding."
>
> **Why it matters:** every word of the title is one idea of the paper. If you understand the title, you understand the paper.

Let us take the title apart, one word at a time.

- **Pre-training.** First, train a model on a huge pile of ordinary text, before it ever sees the real task. This is the expensive part, and it is done once.
- **Deep.** The model is a tall stack of layers (12 or 24 of them). Each layer refines what the layer below understood.
- **Bidirectional.** Every word looks at the words on its **left and** on its **right**. This is the key idea of the paper.
- **Transformers.** The kind of neural network used. It was introduced in 2017 in the paper *Attention Is All You Need*.
- **Language Understanding.** The goal: tasks where a model must understand text (classify it, answer questions about it, find names in it), not write new text.

> [!DEFINITION] Neural network and model
> A **neural network** is a program made of many simple maths steps whose numbers (called **weights** or **parameters**) are learned from examples. A **model** is one such network with its learned weights. BERT is a model with about 110 million weights in its smaller version.

> [!DEFINITION] Transformer
> A neural network design built around **attention**: every word computes how much it should "look at" every other word, and mixes in information from the words it looks at. If attention is new to you, the [attention series](/writings/attention-1-self-attention/) explains it from zero. For this paper you only need the idea: words look at other words.

> [!DEFINITION] Pre-training
> Training a model on a large amount of general text first, so it learns how language works. The model is later adapted to a specific job. It is like a student who reads thousands of books before taking any particular exam.

## What BERT is {§Abstract}

> [!PAPER] Devlin et al. (2018), BERT · Abstract · page 1
> [![The first half of the BERT abstract, with three highlights: Bidirectional Encoder Representations from Transformers; jointly conditioning on both left and right context in all layers; one additional output layer](/img/papers/bert/p1-abstract-what.png)](/img/papers/bert/p1-abstract-what.png)
>
> **Context:** the abstract, the paper's summary of itself in two paragraphs. This is the first paragraph.
>
> **What it says:** BERT stands for **B**idirectional **E**ncoder **R**epresentations from **T**ransformers. It is pre-trained on "unlabeled text by jointly conditioning on both left and right context in all layers". Afterwards it can be fine-tuned "with just one additional output layer" for many tasks, such as question answering.
>
> **Why it matters:** these three sentences are the whole recipe. Learn from plain text, look both ways, then add one small layer per task.

The name has four parts. You already know **bidirectional** and **transformers** from the title. The two new words are **encoder** and **representations**.

> [!DEFINITION] Representation (embedding)
> A list of numbers that stands for a piece of text. A model turns each word into such a list, for example 768 numbers. Words with similar meanings in similar contexts get similar lists. "Representation", "embedding" and "vector" are used almost interchangeably in this paper.

> [!DEFINITION] Encoder
> The part of a Transformer that **reads** text and turns every word into a representation. (The other part, a **decoder**, writes text one word at a time.) BERT is only an encoder: it understands, it does not write.

### Encoder and decoder: the two halves of the Transformer

"Encoder" is a precise word here. It points at one half of the Transformer from *Attention Is All You Need* (Vaswani et al., 2017), the paper BERT is built on.

> [!PAPER] Vaswani et al. (2017), Attention Is All You Need · Section 3 · page 3, Figure 1
> [![Figure 1 of Attention Is All You Need: the Transformer, an encoder stack on the left with multi-head attention and feed-forward blocks, and a decoder stack on the right with masked multi-head attention, cross-attention and feed-forward blocks, ending in a linear layer and softmax](/img/papers/bert/p1-transformer-fig1.png)](/img/papers/bert/p1-transformer-fig1.png)
> [![Section 3.1 of Attention Is All You Need: the encoder and decoder are the left and right halves of Figure 1; the decoder's self-attention is modified to prevent positions from attending to subsequent positions, so predictions for position i depend only on outputs at positions before i](/img/papers/bert/p1-transformer-halves.png)](/img/papers/bert/p1-transformer-halves.png)
>
> **Context:** the Transformer was built for translation. The left half (the **encoder**) reads the source sentence. The right half (the **decoder**) writes the translation, one word at a time.
>
> **What it says:** both halves are stacks of $$N$$ identical layers. The decoder's self-attention is "modified to prevent positions from attending to subsequent positions", so "the predictions for position $$i$$ can depend only on the known outputs at positions less than $$i$$". The encoder has no such rule.
>
> **Why it matters:** BERT is the left half. OpenAI GPT is (almost) the right half, without the cross-attention block. The single difference between them is that rule about "subsequent positions".

{{FIG:p1_enc_dec|The two halves of the original Transformer. BERT keeps only the encoder (every word sees every word). GPT keeps a decoder-style stack without the cross-attention, so a word sees only itself and the words before it.}}

That rule is implemented by a **mask** inside attention. Here is attention with the mask written in. (The [attention series](/writings/attention-1-self-attention/) derives this equation step by step.)

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right) V
$$

where, for a text of $$n$$ tokens:

- $$Q$$, $$K$$, $$V$$ are the **queries**, **keys** and **values**: three $$n \times d_k$$ tables made from the token vectors. Row $$i$$ of $$Q$$ asks "what is token $$i$$ looking for?"; row $$j$$ of $$K$$ says "what does token $$j$$ offer?";
- $$QK^\top$$ is an $$n \times n$$ table of scores: entry $$(i, j)$$ says how well token $$i$$'s question matches token $$j$$;
- $$\sqrt{d_k}$$ (here $$\sqrt{64} = 8$$) keeps the scores from growing too large;
- $$M$$ is the **mask**, also $$n \times n$$: 0 where looking is allowed, $$-\infty$$ where it is blocked;
- softmax turns each row into weights that add up to 1, and $$e^{-\infty} = 0$$, so a blocked position gets weight exactly 0.

The two models differ only in $$M$$:

$$
M^{\text{GPT}}_{ij} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
\qquad\qquad
M^{\text{BERT}}_{ij} = 0 \ \text{ for every } i, j
$$

In words: GPT blocks every key to the right of the query (the upper triangle of the table). BERT blocks nothing. (BERT's real code also masks padding, the empty slots that fill short texts up to a common length, but never real words.)

{{FIG:p1_kid_masks|The mask for "the kid smiles". Rows are queries (the word that looks), columns are keys (the word looked at). GPT blocks the upper triangle: "kid" may read "the" and itself, but not "smiles". BERT allows every square.}}

**A worked example with real numbers.** I took the first attention head of GPT-2's first layer, fed it "the kid smiles", and recomputed the attention by hand ([`bert_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part1_math.py)):

```python
q, k, v = blk.attn.c_attn(blk.ln_1(x)).split(768, dim=2)      # GPT-2 layer 1: queries, keys, values
qh, kh = q[0, :, 0:64], k[0, :, 0:64]                          # head 1 uses 64 of the 768 numbers
S = (qh @ kh.T) / math.sqrt(64)                                # 3 x 3 table of scores
M = torch.triu(torch.full((3, 3), float('-inf')), diagonal=1)  # -inf above the diagonal: the future
A = torch.softmax(S + M, -1)                                   # each row adds up to 1
```

```text
scores s = q.k / sqrt(64):
  the     -0.618   -1.104   -0.860
  kid      0.770   -0.374   -0.562
  smiles   0.345   -0.316   -0.987
mask M (0 = allowed, -inf = blocked: a key to the right of the query):
  the      0.000     -inf     -inf
  kid      0.000    0.000     -inf
  smiles   0.000    0.000    0.000
weights a = softmax(s + M), row by row:
  the      1.000    0.000    0.000   (row sum 1.000)
  kid      0.758    0.242    0.000   (row sum 1.000)
  smiles   0.562    0.290    0.148   (row sum 1.000)
max difference from the library's own attention weights: 0.0e+00
```

Follow the row for "kid" by hand. Its score for "smiles" (-0.562) is replaced by $$-\infty$$, so only two scores are left:

$$
a_{\text{kid},\text{the}} = \frac{e^{0.770}}{e^{0.770} + e^{-0.374}} = \frac{2.160}{2.160 + 0.688} = 0.758,
\qquad
a_{\text{kid},\text{kid}} = \frac{0.688}{2.848} = 0.242
$$

The first row is even simpler: "the" can see only itself, so its weight on itself is 1.000 whatever its score is. Our hand computation matches the library exactly (difference 0.0).

{{FIG:p1_kid_weights|Real attention weights on the same words. Left: GPT-2, layer 1, head 1, with the blocked squares forced to 0. Right: BERT, layer 1, averaged over its 12 heads, on "[CLS] the kid smiles [SEP]" (BERT adds those two special tokens, explained in Part 2). Every square is filled, and every row still adds up to 1.}}

So when the paper says BERT is "bidirectional", this is the concrete meaning: an all-zero mask. The hard part, which the rest of the paper solves, is how to *train* such a model, since the usual training game (predict the next word) breaks when the model can see the next word.

The abstract makes three claims. Let us translate each one.

**1. "Pre-train deep bidirectional representations from unlabeled text."** BERT learns from text that nobody has labelled: Wikipedia and a collection of books. Nobody had to mark which sentences are happy or sad. Labelled data is expensive. Plain text is almost free, and there is a lot of it.

> [!DEFINITION] Labeled and unlabeled data
> **Labeled** data has the right answer attached by a person, like a movie review marked "positive". **Unlabeled** data is just text, with no answers attached.

How can a model learn anything from text that has no answers attached? The trick is to make the answers out of the text itself. Hide a piece of the text, and the hidden piece *is* the answer.

{{FIG:p1_selfsup|Where the "labels" come from when nobody labels anything. A language model like GPT uses each next word as the answer for the words before it. BERT's masked language model hides a word in the middle and uses it as the answer, with both sides as input. Either way, every sentence ever written becomes training data.}}

> [!DEFINITION] Self-supervised learning
> Learning from data whose labels are made automatically from the data itself, for example by hiding a word and asking for it back. The BERT paper calls this "unsupervised" (no person supervises it). Today it is usually called **self-supervised**.

**2. "Jointly conditioning on both left and right context in all layers."** When BERT builds the representation of a word, it uses the words before it *and* the words after it, and it does this at every layer, not just at the end. "Conditioning on" simply means "using as input".

**3. "Fine-tuned with just one additional output layer."** After pre-training, you take BERT, put one small new layer on top, and keep training the whole thing a little on your task. The same pre-trained BERT can become a sentiment classifier, a question-answering system or a name finder.

> [!DEFINITION] Fine-tuning
> Taking a pre-trained model and training it a bit more on a specific task with labeled examples. All the weights move a little; nothing starts from zero except the small new layer. It is much cheaper than pre-training: minutes or hours instead of days.

{{FIG:p1_two_steps|The whole BERT recipe. Pre-train once on unlabeled text (the expensive step). Then, for every task, start from a copy of the pre-trained model, add one small output layer, and fine-tune.}}

**Use case.** A company that wants to sort support emails into "billing", "bug report" and "feature request" does not need to teach a model English. It starts from pre-trained BERT and fine-tunes it on a few thousand labelled emails.

### Why an "understanding" model, when GPT-style models can write?

A fair question today: chat models write fluent text, so why read a paper about a model that does not write? Because writing is only one job. Many everyday language jobs are about **reading and deciding**: is this review positive, which department should get this email, where are the names in this contract, which passage answers this question. Images have the same split: generating a new picture is one job, recognising what is in a photo is another, and most practical vision systems do the second.

{{FIG:p1_understand_gen|Two families of jobs. Understanding: read the whole input, then output a label, a span or a tag per word (BERT's home ground). Generation: write new text one word at a time (GPT's home ground). The same split exists for images.}}

For understanding jobs, a model that sees the whole input at once is a natural fit, and it can be small and fast. That is why BERT-style models still run inside many search, classification and retrieval systems (Part 6 shows where).

### What exactly gets trained when you fine-tune?

This is a common point of confusion, so it is worth stating carefully. The paper is explicit (we will see it in Section 3): "all of the parameters are fine-tuned". During fine-tuning, **every one of BERT's roughly 110 million weights keeps learning**, together with the small new output layer, which is the only part that starts from random numbers.

{{FIG:p1_what_updates|Three ways to reuse a pre-trained model. Left, feature-based (ELMo-style): the pre-trained layers are frozen and a separate task model is trained from zero. Middle, a common misreading of BERT: freeze everything and train only a new top layer. Right, what BERT actually does: all layers keep training, a little, together with the new layer.}}

> [!WARNING] A common misreading
> "Fine-tuning BERT means freezing it and training only the last layer" is **not** what the paper does. Freezing BERT and training something on top is the *feature-based* approach, which the paper tests separately in Section 5.3 (Part 5). In fine-tuning, all weights move, but only a little, because training is short and the learning rate is small.

The idea itself came from computer vision, which the paper mentions in its related work (Section 2.3, Part 2): train a big network once on ImageNet, a large labelled photo collection, then reuse it for many smaller tasks. BERT brings that recipe to language, with one difference: its first stage needs no labels at all.

{{FIG:p1_transfer|Transfer learning in two fields. Vision: pre-train on ImageNet's labelled photos, then fine-tune for a narrow task such as reading chest X-rays. Language: pre-train BERT on unlabeled Wikipedia and books, then fine-tune for a narrow task such as routing support emails.}}

> [!DEFINITION] Transfer learning
> Reusing what a model learned on one task (usually a big, general one) to do better on another task (usually a small, specific one). Pre-training followed by fine-tuning is transfer learning.

## The results in the abstract {§Abstract}

> [!PAPER] Devlin et al. (2018), BERT · Abstract · page 1
> [![The second paragraph of the BERT abstract: new state-of-the-art results on eleven tasks, GLUE score 80.5 percent, MultiNLI accuracy 86.7 percent, SQuAD v1.1 Test F1 93.2 and SQuAD v2.0 Test F1 83.1](/img/papers/bert/p1-abstract-results.png)](/img/papers/bert/p1-abstract-results.png)
>
> **Context:** the second paragraph of the abstract, right after the description of BERT.
>
> **What it says:** BERT is "conceptually simple and empirically powerful". It sets new best results on eleven tasks, and the abstract lists four of them with how much it improved.
>
> **Why it matters:** these were large jumps. Benchmarks usually move by fractions of a point at a time.

> [!DEFINITION] State of the art (SOTA)
> The best result anyone has published so far on a task. "New state of the art" means "better than every earlier system".

> [!DEFINITION] Benchmark
> A fixed, public test that everyone uses, so results can be compared fairly. A model is given the test questions, and its answers are scored.

Here are the four results from the abstract, in plain words:

| Benchmark | What it tests | BERT's score | Improvement |
|---|---|---|---|
| GLUE | nine sentence-understanding tasks, averaged | 80.5 | +7.7 points |
| MultiNLI | does sentence B follow from sentence A? | 86.7% accuracy | +4.6 points |
| SQuAD v1.1 | find the answer to a question in a paragraph | 93.2 Test F1 | +1.5 points |
| SQuAD v2.0 | the same, but some questions have no answer | 83.1 Test F1 | +5.1 points |

> [!DEFINITION] Accuracy and F1
> **Accuracy** is the share of answers that are exactly right. **F1** is a score between 0 and 100 that gives partial credit: for question answering, it measures how many words of the predicted answer overlap with the true answer. Part 4 explains both properly.

A note on the wording "7.7% point absolute improvement": it means the score went up by 7.7 **points** (for example from 72.8 to 80.5), not by 7.7 percent of the old score. Part 4 goes through every one of these tasks and every table of results.

## Pre-training already worked, for two kinds of tasks {§1}

Now the introduction. Its first paragraph sets the scene.

> [!PAPER] Devlin et al. (2018), BERT · Section 1 · page 1
> [![The first paragraph of the introduction: language model pre-training has been shown to be effective for sentence-level tasks such as natural language inference and paraphrasing, and token-level tasks such as named entity recognition and question answering](/img/papers/bert/p1-intro-tasks.png)](/img/papers/bert/p1-intro-tasks.png)
>
> **Context:** the opening paragraph of Section 1 (Introduction).
>
> **What it says:** "Language model pre-training" already helps many tasks. These come in two kinds: **sentence-level tasks**, which look at whole sentences ("analyzing them holistically"), and **token-level tasks**, which need an answer for every single word ("fine-grained output at the token level").
>
> **Why it matters:** BERT wants to be good at **both** kinds with the same model. Keep the two kinds in mind; the argument of the next paragraphs depends on them.

> [!DEFINITION] Token
> A small piece of text: usually a word, sometimes part of a word or a punctuation mark. Models read tokens, not letters. Part 2 shows exactly how BERT cuts text into tokens.

The paper names four example tasks:

> [!DEFINITION] Natural language inference (NLI)
> Given two sentences, decide if the second one **follows from** the first (entailment), **contradicts** it, or is **neutral**. "A man is playing a guitar" → "A person is making music": entailment.

> [!DEFINITION] Paraphrasing
> Deciding whether two sentences mean the same thing, even with different words.

> [!DEFINITION] Named entity recognition (NER)
> Marking every word that is part of a name, and what kind of name it is: a person, a place, an organisation, and so on.

> [!DEFINITION] Question answering (QA)
> Given a question and a paragraph, find the words in the paragraph that answer the question.

{{FIG:p1_tasks|Two kinds of tasks. A sentence-level task (here, natural language inference) gives one answer for the whole input. A token-level task (here, named entity recognition) gives an answer for every token.}}

The two token-level examples in that paragraph deserve a closer look, because both come back in Part 4.

{{FIG:p1_token_tasks|The two token-level tasks the paragraph names. Named entity recognition gives every word a tag: B-PER starts a person's name, I-PER continues it, B-LOC starts a place, O means "not a name". Question answering points at two positions in the passage: where the answer starts and where it ends.}}

> [!DEFINITION] Named entity recognition (NER)
> Finding the names of people, places, organisations and similar things in a text, and labelling each word. The usual tags are **B-** (beginning of a name), **I-** (inside a name) and **O** (outside any name), so "Barack Obama" becomes B-PER I-PER.

## Two ways to reuse a pre-trained model {§1}

> [!PAPER] Devlin et al. (2018), BERT · Section 1 · page 1
> [![The second paragraph of the introduction: there are two existing strategies, feature-based and fine-tuning. ELMo uses task-specific architectures that include the pre-trained representations as additional features. OpenAI GPT is trained by simply fine-tuning all pre-trained parameters. Both use unidirectional language models](/img/papers/bert/p1-intro-strategies.png)](/img/papers/bert/p1-intro-strategies.png)
>
> **Context:** the second paragraph of the introduction. It describes the two ways people used pre-trained models in 2018.
>
> **What it says:** the **feature-based** approach (ELMo) feeds the pre-trained representations as extra inputs into a model built specially for each task. The **fine-tuning** approach (OpenAI GPT) adds very few new weights and fine-tunes "all pre-trained parameters". Both learn during pre-training with "unidirectional language models".
>
> **Why it matters:** BERT picks the fine-tuning side. And the last sentence names the weakness BERT attacks: both approaches pre-train in only one direction.

> [!DEFINITION] Downstream task
> The real task you care about in the end (sentiment, question answering, and so on), as opposed to the pre-training task. It sits "downstream" of pre-training.

> [!DEFINITION] ELMo
> A 2018 model by Peters et al. (the "a" in "2018a" just tells two papers by the same authors apart). It reads text with two separate networks, one left to right and one right to left, and gives every word a vector that depends on its sentence. Its vectors are used as **features**: extra inputs for another model.

> [!DEFINITION] OpenAI GPT
> A 2018 model by Radford et al. at OpenAI (the first "Generative Pre-trained Transformer"). A Transformer trained to predict the next word, left to right. To use it for a task, you fine-tune the whole model with a small extra layer. BERT copies this recipe, and changes the direction.

{{FIG:p1_strategies|The two strategies. Feature-based: the pre-trained model is frozen and only supplies features to a separate task model. Fine-tuning: the whole pre-trained model keeps training on the task, and only a tiny layer is new.}}

Both approaches, the paper says, pre-train with a **language model**. That word is the key to the whole argument, so let us define it carefully.

> [!DEFINITION] Language model (LM)
> A model that gives a probability to the next word, given the words before it. Shown "I went to the", a good language model gives high probability to "store", "park" or "bank", and low probability to "elephant".

A language model reads left to right and predicts each word from the words before it. Written as an equation, the probability of a whole sentence is built up one word at a time:

$$
P(w_1, w_2, \dots, w_n) = \prod_{i=1}^{n} P(w_i \mid w_1, \dots, w_{i-1})
$$

where:

- $$w_1, \dots, w_n$$ are the words of the sentence, in order, and $$n$$ is how many there are;
- $$P(w_i \mid w_1, \dots, w_{i-1})$$ is the probability of word $$i$$ **given** (that is what the bar $$\mid$$ means) all the words before it;
- $$\prod$$ means "multiply all of these together", for $$i$$ from 1 to $$n$$.

Look at what is on the right of the bar: only earlier words. That is what **unidirectional** means. The training signal is free (the next word is always in the text), which is why language models are such a good way to pre-train. But each word only ever learns from its left.

**The chain rule with real numbers.** Here is the equation applied to "The kid smiles at the dog." by GPT-2 ([`bert_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part1_math.py)). Each row is one factor of the product: the probability GPT-2 gave the real next word, seeing only the words on its left.

```text
 i  word     given (left side only)     P(word | left)    log P
 1  The      (start)                          0.037700   -3.278
 2  kid      The                              0.000040  -10.114
 3  smiles   The kid                          0.000257   -8.268
 4  at       The kid smiles                   0.087426   -2.437
 5  the      The kid smiles at                0.150418   -1.894
 6  dog      The kid smiles at the            0.002395   -6.034
 7  .        The kid smiles at the dog        0.177917   -1.726
sum of log P = -33.752
P(sentence) = exp(-33.752) = 2.196e-15
average negative log-likelihood = 4.822  (perplexity exp of that = 124.2)
```

{{FIG:p1_chain|The chain rule, one factor per word. Each bar is the probability GPT-2 gave the real next word from its left side only. Multiplying all seven gives the probability of the whole sentence, 2.2 × 10⁻¹⁵.}}

Three things to notice:

- **Multiplying tiny numbers underflows**, so in practice everyone adds logarithms instead: $$\log \prod_i p_i = \sum_i \log p_i$$. Here the seven logs add up to $$-33.752$$, and $$e^{-33.752} = 2.2 \times 10^{-15}$$, the same probability.
- **Training a language model** means making this sum as large as possible over billions of sentences. Written as a loss to make small, it is the **negative log-likelihood**:

$$
\mathcal{L}_{\text{LM}} = -\sum_{i=1}^{n} \log P(w_i \mid w_1, \dots, w_{i-1})
$$

  where $$\mathcal{L}$$ is the loss (lower is better) and the other symbols are as above. For our sentence, $$\mathcal{L} = 33.752$$, or $$4.822$$ per word.
- **"kid" got only 0.00004.** After "The", thousands of words are possible, so no single one gets much probability. Every word is predicted from its left only; the model never got to use "smiles at the dog" to help with "kid".

> [!DEFINITION] Logarithm and log-likelihood
> The natural logarithm $$\log x$$ answers "$$e$$ to what power gives $$x$$?", with $$e \approx 2.718$$. It turns multiplication into addition, and small probabilities into manageable negative numbers ($$\log 0.00004 = -10.1$$). The **log-likelihood** of a text is the sum of the logs of the probabilities a model gave its words.

> [!DEFINITION] Perplexity
> $$e$$ raised to the average negative log-likelihood per word. It reads as "on average, the model was as unsure as if it were choosing among this many equally likely words". Here 124.2. Lower is better. Part 5 uses perplexity to compare BERT sizes.

> [!DEFINITION] Objective function (training objective)
> The score a model is trained to improve. For a language model, it is "give high probability to the real next word". Training means nudging the weights, millions of times, so this score gets better.

## The problem: reading in only one direction {§1}

> [!PAPER] Devlin et al. (2018), BERT · Section 1 · page 1
> [![The third paragraph of the introduction: the major limitation is that standard language models are unidirectional. In OpenAI GPT every token can only attend to previous tokens. This is sub-optimal for sentence-level tasks and could be very harmful for token-level tasks such as question answering, where it is crucial to incorporate context from both directions](/img/papers/bert/p1-intro-limitation.png)](/img/papers/bert/p1-intro-limitation.png)
>
> **Context:** the third paragraph of the introduction: the problem statement of the whole paper.
>
> **What it says:** "The major limitation is that standard language models are unidirectional." In OpenAI GPT, "every token can only attend to previous tokens". This is "sub-optimal" for sentence-level tasks and "could be very harmful" for token-level tasks like question answering, "where it is crucial to incorporate context from both directions".
>
> **Why it matters:** this is the problem BERT solves. Everything else in the paper is the solution and the evidence.

> [!DEFINITION] Attend to
> In a Transformer, "token A attends to token B" means A looks at B and takes information from it. In GPT, a token may attend only to itself and the tokens before it. A **mask** blocks every token to the right.

Why is one direction a problem? Because the meaning of a word often depends on what comes **after** it. Read this sentence and stop at the blank:

> I went to the ____

You cannot know the missing word. Now read the whole sentence:

> I went to the ____ to deposit my paycheck.

Now it is obviously "bank". The clue was on the right. A left-to-right model building the representation of that position has not seen "deposit my paycheck" yet, so it cannot use it.

{{FIG:p1_context|What the blank can see in three kinds of model. Left to right (GPT): only the earlier words. ELMo: a left reader and a right reader that work separately and are joined only at the very end. BERT: both sides, in every layer.}}

### Let us test it on real models

I gave the same three sentences to two real, public models:

- **GPT-2**, a left-to-right model from 2019 (the small version). It sees only the words before the blank and predicts the next word.
- **BERT** (`bert-base-uncased`, the model released with this paper). It sees the whole sentence with the blank replaced by a special `[MASK]` token, and predicts the missing word.

```python
from transformers import AutoTokenizer, BertForMaskedLM, GPT2LMHeadModel
import torch

btok = AutoTokenizer.from_pretrained("bert-base-uncased")
bert = BertForMaskedLM.from_pretrained("bert-base-uncased").eval()
gtok = AutoTokenizer.from_pretrained("openai-community/gpt2")
gpt2 = GPT2LMHeadModel.from_pretrained("openai-community/gpt2").eval()

# GPT-2: only the left side, predict the next token
ids = gtok("I went to the", return_tensors="pt").input_ids
p_gpt2 = torch.softmax(gpt2(ids).logits[0, -1], -1)

# BERT: the whole sentence, predict the [MASK]
enc = btok("i went to the [MASK] to deposit my paycheck.", return_tensors="pt")
pos = (enc.input_ids[0] == btok.mask_token_id).nonzero().item()
p_bert = torch.softmax(bert(**enc).logits[0, pos], -1)
```

These are the five most likely words each model gave, with their probabilities (the real output of [`bert_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part1.py)):

```text
sentence: i went to the ____ to deposit my paycheck.   (hidden word: bank)
  GPT-2, left side only: hospital 0.033, doctor 0.024, store 0.022, gym 0.018, office 0.015
  BERT, both sides:      bank 0.901, office 0.009, store 0.008, teller 0.008, atm 0.006

sentence: my neighbor's ____ barked at the mailman all morning.   (hidden word: dog)
  GPT-2, left side only: house 0.071, dog 0.037, son 0.026, car 0.024, daughter 0.023
  BERT, both sides:      dog 0.651, dogs 0.142, voice 0.020, phone 0.018, had 0.014

sentence: she picked up her ____ and started to play a song.   (hidden word: guitar)
  GPT-2, left side only: phone 0.100, bag 0.033, gun 0.018, daughter 0.014, own 0.013
  BERT, both sides:      guitar 0.677, phone 0.081, ipod 0.050, fiddle 0.017, instrument 0.013
```

{{FIG:p1_fill|The first sentence as a picture. GPT-2 cannot see "to deposit my paycheck" and spreads its guesses over places you might go. BERT sees both sides and puts 0.901 on "bank".}}

Both sides do not make every word easy, though. Take "smiles" from our chain-rule sentence:

```text
GPT-2  P(smiles | "The kid")                    = 0.0003   top 5: who 0.1939, is 0.0687, in 0.0596, was 0.0550, 's 0.0507
BERT   P(smiles | "the kid [MASK] at the dog.") = 0.0008   top 5: looked 0.3233, stared 0.1029, glanced 0.0795, pointed 0.0620, glared 0.0595
```

{{FIG:p1_smiles|One hidden word, predicted with one side and with both. GPT-2, seeing "The kid", guesses words that can follow any noun (who, is, in). BERT, seeing "the kid ___ at the dog.", guesses only verbs that take "at": looked, stared, glanced. Neither gets "smiles" right, because many verbs fit.}}

BERT's probability for "smiles" is still small, because "looked at the dog" fits just as well. But look at the *kind* of guesses. With only the left side, GPT-2's guesses are almost generic. With both sides, every one of BERT's top five is a verb that can be followed by "at". The right side did not reveal the answer; it narrowed the space of sensible answers. That narrowing is what makes BERT's vectors useful for understanding tasks.

> [!DEFINITION] Probability
> A number from 0 to 1 that says how likely something is. 0.901 means "about 90% sure". The probabilities over all possible words add up to 1.

What this shows, and what it does not:

- With the right side hidden, the guess is a guess. GPT-2's top answer for the first sentence has a probability of only 0.033.
- With both sides visible, the answer is nearly certain: BERT gives "bank" 0.901 and "guitar" 0.677.
- This is **not** a contest between two models. GPT-2 was simply not given the words on the right, because a left-to-right model never gets them at this position. That missing information is exactly the paper's point.

**Use case.** Question answering is the paper's own example. In "Who founded the company that makes the iPhone?", the word "company" needs the words after it to know which company. A model that builds every word's meaning from both sides can answer such questions far better.

## BERT's fix: fill in the blanks {§1}

If language models are one-directional by nature, how do you pre-train a two-directional model? The paper's answer is in the next paragraph.

> [!PAPER] Devlin et al. (2018), BERT · Section 1 · pages 1 and 2
> [![The fourth paragraph of the introduction: BERT alleviates the unidirectionality constraint by using a masked language model pre-training objective, inspired by the Cloze task. The masked language model randomly masks some of the tokens from the input, and the objective is to predict the original vocabulary id of the masked word](/img/papers/bert/p1-intro-mlm.png)](/img/papers/bert/p1-intro-mlm.png)
> [![The paragraph continues on page 2: based only on its context. The MLM objective enables the representation to fuse the left and the right context. In addition, a next sentence prediction task jointly pre-trains text-pair representations](/img/papers/bert/p1-intro-mlm2.png)](/img/papers/bert/p1-intro-mlm2.png)
>
> **Context:** the fourth paragraph of the introduction, which runs from the bottom of page 1 onto page 2. After the problem, the solution.
>
> **What it says:** BERT uses a **"masked language model" (MLM)**, "inspired by the Cloze task". It "randomly masks some of the tokens from the input", and the model must "predict the original vocabulary id of the masked word based only on its context". This lets the representation "fuse the left and the right context". A second task, **"next sentence prediction"**, teaches the model about pairs of sentences.
>
> **Why it matters:** this is the trick that makes deep bidirectional pre-training possible. Instead of predicting the *next* word, predict a *hidden* word, using everything around it.

> [!DEFINITION] Cloze task
> A fill-in-the-blank test. Wilson Taylor described it in 1953 as a way to measure how readable a text is. "The cat sat on the ____." Readers fill in the missing word from the context.

> [!DEFINITION] Masking
> Hiding a token from the model by replacing it with the special token `[MASK]`. The model must work out what was there.

> [!DEFINITION] Vocabulary id
> Every token the model knows has a number, its **id**, in a fixed list called the **vocabulary**. BERT's vocabulary has about 30,000 tokens. "Predict the original vocabulary id" just means "say which word was hidden".

You already saw masked language modelling in action: the `[MASK]` sentences above are exactly this pre-training task. The trick works because a hidden word cannot leak its own answer, so the model is free to look in both directions. In Part 3 we will see *why* an ordinary language model cannot simply look both ways (each word would "see itself"), and we will run the exact masking recipe, which has a few surprising details.

The paragraph ends with a second task, **next sentence prediction**: given two pieces of text, guess whether the second really came right after the first. Many tasks are about *pairs* of sentences (does this answer this question? does this sentence follow from that one?), and plain language modelling never practises that. Part 3 covers it too.

## The three contributions {§1}

> [!PAPER] Devlin et al. (2018), BERT · Section 1 · page 2
> [![The contributions of the paper as three bullet points: the importance of bidirectional pre-training; pre-trained representations reduce the need for heavily-engineered task-specific architectures; BERT advances the state of the art for eleven NLP tasks. The code and models are released on GitHub](/img/papers/bert/p1-contributions.png)](/img/papers/bert/p1-contributions.png)
>
> **Context:** the end of the introduction, where the authors list what is new in their paper.
>
> **What it says:** three contributions. (1) Bidirectional pre-training matters, and BERT is deeply bidirectional, unlike GPT (one direction) and unlike ELMo's "shallow concatenation of independently trained left-to-right and right-to-left LMs". (2) Pre-trained representations "reduce the need for many heavily-engineered task-specific architectures". (3) BERT "advances the state of the art for eleven NLP tasks", and the code and models are public.
>
> **Why it matters:** each claim is tested later. Claim 1 by the ablation studies in Part 5, claims 2 and 3 by the experiments in Part 4.

Let us unpack each contribution.

**1. Deep, not shallow, bidirectionality.** ELMo does look both ways, but in a shallow way: one network reads left to right, another reads right to left, and their outputs are glued together ("concatenated") at the very end. Inside each network, information still flows one way only. In BERT, every layer of one single network mixes both sides. The paper argues this is "strictly more powerful".

> [!DEFINITION] Concatenation
> Joining two lists of numbers end to end. A list of 512 numbers concatenated with another list of 512 numbers gives one list of 1,024 numbers. Nothing is mixed; the two halves just sit side by side.

To see what "shallow" means exactly, look at how ELMo is trained, in its own paper:

> [!PAPER] Peters et al. (2018a), Deep contextualized word representations (ELMo) · Section 3.1 · page 2
> [![Section 3.1 of the ELMo paper: a forward language model predicts each token from the tokens before it, a backward language model predicts each token from the tokens after it, and the biLM maximizes the sum of both log likelihoods, with separate parameters for the LSTMs in each direction](/img/papers/bert/p1-elmo-bilm.png)](/img/papers/bert/p1-elmo-bilm.png)
>
> **Context:** ELMo's training objective, the paper BERT calls "Peters et al. (2018a)".
>
> **What it says:** a **forward** language model predicts each token from the tokens before it; a **backward** one predicts "the previous token given the future context". ELMo "jointly maximizes the log likelihood of the forward and backward directions", sharing only the input layer and the final softmax, while keeping "separate parameters for the LSTMs in each direction".
>
> **Why it matters:** "separate parameters for the LSTMs in each direction" is the shallow part. Each reader is still one-directional. They meet only when their outputs are glued together at the end.

ELMo's objective, in the notation we used for GPT:

$$
\mathcal{L}_{\text{ELMo}} = -\sum_{k=1}^{N} \Big( \log p(t_k \mid t_1, \dots, t_{k-1}) + \log p(t_k \mid t_{k+1}, \dots, t_N) \Big)
$$

where $$t_1, \dots, t_N$$ are the tokens, the first term is the forward model (left side only) and the second the backward model (right side only). Each term on its own is an ordinary one-directional language model. No single term ever conditions on both sides at once, which is the thing BERT's masked language model does (Part 3).

{{FIG:p1_layers|"In all layers", drawn. Highlighted lines show what feeds the word "kid" at the top. BERT: both neighbours, at every layer, so after two layers "kid" has mixed in everything. GPT: only "the" and itself. ELMo: the left tower sees "the", the right tower sees "smiles", and the two halves meet only in the final concatenation.}}

> [!DEFINITION] Architecture
> The design of a model: which layers it has, in what order, and how they connect. A "task-specific architecture" is a design built by hand for one task.

**2. Less hand-made machinery.** Before BERT, the best question-answering systems and the best sentiment classifiers were different, carefully designed networks. With BERT, the same model plus one small output layer does both. That saves months of engineering per task.

**3. Eleven state-of-the-art results**, plus public code and weights. The public release is a big reason BERT spread so fast: anyone could download the model and fine-tune it on a single GPU. The model we ran above is that release.

> [!DEFINITION] NLP
> Natural language processing: the field of making computers work with human language (text and speech).

## Who's who: the papers BERT is talking to

The introduction names a handful of earlier works again and again. Here they are in one place, so the rest of the series can refer back to them. Each entry is listed in full in the references at the end of this part.

| Short name | Paper | Year | What it is, in one line | Role in the BERT paper |
|---|---|---|---|---|
| Transformer | Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) | 2017 | the attention-only network for translation | BERT is its encoder half |
| ELMo | Peters et al. (2018a), [*Deep contextualized word representations*](https://arxiv.org/abs/1802.05365) | 2018 | two one-way LSTM language models, concatenated | the feature-based rival; "shallow" bidirectional |
| OpenAI GPT | Radford et al., [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 2018 | a left-to-right Transformer, fine-tuned per task | the fine-tuning rival; BERT-base copies its size |
| ULMFiT | Howard and Ruder, [*Universal Language Model Fine-tuning for Text Classification*](https://arxiv.org/abs/1801.06146) | 2018 | an LSTM language model fine-tuned per task | an earlier fine-tuning approach |
| Semi-supervised sequence learning | Dai and Le, [*Semi-supervised Sequence Learning*](https://arxiv.org/abs/1511.01432) | 2015 | pre-train an LSTM, then fine-tune it | one of the first pre-train-then-fine-tune papers |
| word2vec | Mikolov et al., [*Distributed Representations of Words and Phrases and their Compositionality*](https://arxiv.org/abs/1310.4546) | 2013 | one fixed vector per word, learned from its neighbours | the classic pre-trained word embeddings |
| GloVe | Pennington et al., [*GloVe: Global Vectors for Word Representation*](https://aclanthology.org/D14-1162/) | 2014 | word vectors from word co-occurrence counts | the other classic word embeddings |
| Skip-thought | Kiros et al., [*Skip-Thought Vectors*](https://arxiv.org/abs/1506.06726) | 2015 | a sentence vector trained to predict nearby sentences | sentence-level pre-training (Part 2) |
| CoVe | McCann et al., [*Learned in Translation: Contextualized Word Vectors*](https://arxiv.org/abs/1708.00107) | 2017 | word vectors from a translation encoder | transfer from a supervised task (Part 2) |
| Cloze | Taylor, [*"Cloze Procedure": A New Tool for Measuring Readability*](https://doi.org/10.1177/107769905303000401) | 1953 | the fill-in-the-blank reading test | the inspiration for the masked LM |

GPT-2 (Radford et al., 2019, [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)), which we ran above, came out a few months after BERT and is not cited in the paper. We use it only because it is a freely available left-to-right model of similar size.

## Figure 3: the three designs side by side {§A.4}

The paper puts its clearest picture of this argument in the appendix (Appendix A.4), so we read it now, where it belongs in the story.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.4, Figure 3 · page 13
> [![Figure 3 of the BERT paper: three diagrams. BERT, a stack of Transformer blocks where every block connects to every position below. OpenAI GPT, a stack of Transformer blocks where each block connects only to positions on its left. ELMo, a left-to-right LSTM and a right-to-left LSTM whose outputs are combined. The caption says only BERT representations are jointly conditioned on both left and right context in all layers](/img/papers/bert/p1-figure3.png)](/img/papers/bert/p1-figure3.png)
>
> **Context:** Appendix A.4 compares BERT, OpenAI GPT and ELMo. Figure 3 draws the three of them.
>
> **What it says:** BERT uses a bidirectional Transformer; GPT a left-to-right Transformer; ELMo two separately trained LSTMs, one per direction. "Among the three, only BERT representations are jointly conditioned on both left and right context in all layers." Also, BERT and GPT are fine-tuning approaches, while ELMo is feature-based.
>
> **Why it matters:** the whole paper in one picture. Look at the arrows.

How to read the figure:

- **E₁, E₂, …, E_N** (yellow, bottom) are the input embeddings: one vector per input token.
- **Trm** is one Transformer block (a layer). **Lstm** is one step of an LSTM, an older kind of network that reads one token at a time.
- **T₁, T₂, …, T_N** (green, top) are the output vectors, one per token.
- **The arrows** show who can see whom. In the BERT panel, every Trm connects to *every* position in the layer below. In the GPT panel, each Trm connects only to positions on its own left. In the ELMo panel, the arrows inside each LSTM chain all point one way; the two directions only meet at the top.

> [!DEFINITION] LSTM
> Long short-term memory: a kind of recurrent neural network, the standard way to read text before Transformers. It reads tokens one at a time, carrying a running memory from each token to the next.

Notice also that BERT and GPT have the **same** shape: a stack of Transformer blocks. The only difference is the arrows, that is, which positions each token may look at. This is deliberate. The authors made BERT's smaller version the same size as GPT so the two could be compared fairly. Part 2 opens up that stack.

> [!TAKEAWAYS] Key takeaways
> - **BERT = Bidirectional Encoder Representations from Transformers**: a Transformer encoder, pre-trained on unlabeled text, that builds every word's representation from both its left and its right, in every layer.
> - **Two steps:** pre-train once (expensive), then fine-tune a copy for each task with one small extra layer (cheap).
> - In 2018 there were **two ways** to reuse a pre-trained model: **feature-based** (ELMo: frozen features fed to a task model) and **fine-tuning** (GPT: train everything a little). BERT fine-tunes.
> - The problem: standard **language models are unidirectional**, so each word learns only from its left. For tasks like question answering, the clue is often on the right.
> - We checked it on real models: with the right side hidden, GPT-2's best guess for "I went to the ____" had probability 0.033; BERT, seeing "to deposit my paycheck", put **0.901** on "bank".
> - BERT's fix is the **masked language model**: hide some tokens and predict them from both sides (a Cloze task), plus **next sentence prediction** for sentence pairs.
> - Concretely, BERT and GPT differ only in the **attention mask** $$M$$: GPT blocks every key to the right of the query ($$-\infty$$ in the upper triangle); BERT blocks nothing. We recomputed a real GPT-2 head by hand and matched the library exactly.
> - In **fine-tuning, all of BERT's weights keep training**, not just the new top layer. Freezing BERT is the separate feature-based approach (Part 5).
> - The results: new state of the art on **eleven tasks**, including GLUE **80.5** (+7.7) and SQuAD v1.1 Test F1 **93.2**.

**Next, in Part 2:** the related work the paper builds on, and the model itself: its layers, its size (and where "110 million parameters" comes from), and how text becomes the numbers BERT reads.

<details>
<summary>Run it yourself</summary>

The script behind the GPT-2 and BERT comparison is [`code/papers/bert/bert_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part1.py). It runs on a laptop CPU in under a minute and downloads `bert-base-uncased` (about 440 MB) and GPT-2 (about 550 MB) the first time.

```bash
pip install torch transformers
python bert_part1.py      # prints the table above, writes results/part1.json
```

<figure><img src="/img/papers/bert/part1-run.png" alt="Terminal output of bert_part1.py: for each of three sentences, GPT-2's top five next-word guesses from the left side only, and BERT's top five guesses for the masked word using both sides" loading="lazy" /><figcaption>The real output of bert_part1.py.</figcaption></figure>

</details>

## References

**The BERT paper**

1. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019 ([ACL Anthology](https://aclanthology.org/N19-1423/)). arXiv:1810.04805, version 2 (May 2019), which is the version shown in the screenshots.
2. Google Research. [BERT code and pre-trained models](https://github.com/google-research/bert). GitHub, 2018.

**Papers the BERT paper cites in this part**

3. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
4. M. E. Peters, M. Neumann, M. Iyyer, M. Gardner, C. Clark, K. Lee, L. Zettlemoyer. [*Deep contextualized word representations*](https://arxiv.org/abs/1802.05365) (ELMo; "Peters et al., 2018a" in the paper). NAACL 2018.
5. A. Radford, K. Narasimhan, T. Salimans, I. Sutskever. [*Improving Language Understanding by Generative Pre-Training*](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) (OpenAI GPT). OpenAI technical report, 2018. The BERT paper's reference list gives it the title "Improving language understanding with unsupervised learning".
6. J. Howard, S. Ruder. [*Universal Language Model Fine-tuning for Text Classification*](https://arxiv.org/abs/1801.06146) (ULMFiT). ACL 2018.
7. A. M. Dai, Q. V. Le. [*Semi-supervised Sequence Learning*](https://arxiv.org/abs/1511.01432). NeurIPS 2015.
8. W. L. Taylor. [*"Cloze Procedure": A New Tool for Measuring Readability*](https://doi.org/10.1177/107769905303000401). Journalism Quarterly 30(4), 1953 (cited as "Journalism Bulletin" in the BERT paper).
9. T. Mikolov, I. Sutskever, K. Chen, G. Corrado, J. Dean. [*Distributed Representations of Words and Phrases and their Compositionality*](https://arxiv.org/abs/1310.4546) (word2vec). NeurIPS 2013.
10. J. Pennington, R. Socher, C. D. Manning. [*GloVe: Global Vectors for Word Representation*](https://aclanthology.org/D14-1162/). EMNLP 2014.
11. R. Kiros, Y. Zhu, R. Salakhutdinov, R. Zemel, A. Torralba, R. Urtasun, S. Fidler. [*Skip-Thought Vectors*](https://arxiv.org/abs/1506.06726). NeurIPS 2015.
12. B. McCann, J. Bradbury, C. Xiong, R. Socher. [*Learned in Translation: Contextualized Word Vectors*](https://arxiv.org/abs/1708.00107) (CoVe). NeurIPS 2017.
13. J. Deng, W. Dong, R. Socher, L.-J. Li, K. Li, L. Fei-Fei. [*ImageNet: A Large-Scale Hierarchical Image Database*](https://doi.org/10.1109/CVPR.2009.5206848). CVPR 2009.

**Other sources used in this part**

14. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2). OpenAI, 2019. The `openai-community/gpt2` model used in the code.
15. P. Nayak. [*Understanding searches better than ever before*](https://blog.google/products/search/search-language-understanding-bert/). Google blog, 25 October 2019.
16. Code for this part: [`bert_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part1.py) and [`bert_part1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part1_math.py).
