---
title: "After BERT: Impact, Limits and Summary"
description: "The paper's conclusion, what came next (RoBERTa, ALBERT, DistilBERT, ELECTRA, Sentence-BERT and more), BERT's real limits measured on real models, how BERT-style models are used today, and the whole paper summarised on one page."
part: 6
covers: "§6 and later work"
date: 2026-10-04
tags: [bert, nlp, transformers]
---

This is the last part. We have read every section of the paper: the idea (Part 1), the model and its input (Part 2), pre-training (Part 3), fine-tuning and results (Part 4) and the ablations (Part 5). Only the short conclusion is left.

After that, this part steps outside the paper. It answers three questions: what did other researchers change after BERT, what can BERT **not** do, and how are BERT-style models used today. Every claim about later work comes from that work's own paper or post, linked at the end. The measurements come from a script you can run.

## The conclusion {§6}

> [!PAPER] Devlin et al. (2018), BERT · Section 6 (Conclusion) · page 9
> [![The conclusion of the BERT paper: rich, unsupervised pre-training is an integral part of many language understanding systems; these results enable even low-resource tasks to benefit from deep unidirectional architectures; our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks](/img/papers/bert/p6-conclusion.png)](/img/papers/bert/p6-conclusion.png)
>
> **Context:** Section 6, the whole conclusion. It is one paragraph of three sentences, right after the ablation studies of Section 5.
>
> **What it says:** earlier work showed that "rich, unsupervised pre-training is an integral part of many language understanding systems", and that it lets "even low-resource tasks" benefit from "deep unidirectional architectures". BERT's "major contribution is further generalizing these findings to deep *bidirectional* architectures", so "the same pre-trained model" can handle "a broad set of NLP tasks".
>
> **Why it matters:** the authors describe their own contribution modestly. They did not invent pre-training or fine-tuning. They made it work with a model that looks both ways.

Three phrases need unpacking.

> [!DEFINITION] Transfer learning
> Learning something on one task (here, filling in blanks in Wikipedia and books) and reusing that knowledge on a different task (here, sentiment or question answering). Pre-training followed by fine-tuning is transfer learning.

> [!DEFINITION] Low-resource task
> A task with only a few labelled examples. Collecting labels is slow and expensive, so many real tasks only have a few thousand, or even a few hundred. For example, the RTE task in Part 4 has 2.5k training examples.

**"Even low-resource tasks benefit from deep unidirectional architectures."** This sentence is about the work *before* BERT, mainly OpenAI GPT. GPT showed that a deep, left-to-right model, pre-trained on plain text, makes even small tasks work well, because the task no longer has to teach the model language from scratch.

**"Our major contribution is further generalizing these findings to deep bidirectional architectures."** BERT keeps that recipe and changes one thing: the direction. The masked language model (Part 3) is what made it possible to pre-train a model that looks both ways in every layer. The ablations in Part 5 showed that this one change is where most of the gain comes from.

**"The same pre-trained model."** One download, many tasks. This is the practical part of the conclusion, and it is why BERT spread so fast.

{{FIG:p6_quadrant|Where the conclusion places BERT. Two questions: does the model read one direction or both, and do the directions meet only at the end (shallow) or in every layer (deep)? ELMo is both-directions but shallow; OpenAI GPT is deep but one-directional; BERT is the first that is deep and bidirectional, which the conclusion calls "our major contribution".}}

## What happened next

The paper's code and weights were public from the start, so other researchers could build on it right away. The years 2019 and 2020 brought a wave of follow-up models.

{{FIG:p6_timeline|From the Transformer to ModernBERT. Most dates are the month of the first arXiv version; the award and the Search post have their own dates. The short notes are explained below.}}

Two events show how quickly BERT mattered outside research:

- In June 2019 the paper won the **Best Long Paper** award at NAACL 2019, the conference where it was published.
- On 25 October 2019, Google wrote that BERT "will help Search better understand one in 10 searches in the U.S. in English". The same post says Google also applied a BERT model to improve featured snippets "in the two dozen countries where this feature is available".

> [!DEFINITION] Featured snippet
> The short answer box that a search engine sometimes shows above the normal results, quoted from a web page.

The follow-up models changed BERT in four different directions.

### 1. Train the same model better: RoBERTa and SpanBERT

> [!DEFINITION] Hyperparameter
> A setting that people choose before training rather than the model learning it: the batch size, the learning rate, the number of steps, how much data to use.

**RoBERTa** (Liu et al., July 2019) is a careful replication of BERT. Its abstract says plainly: "We find that BERT was significantly undertrained." With the **same architecture**, they changed only the training: "(1) training the model longer, with bigger batches, over more data; (2) removing the next sentence prediction objective; (3) training on longer sequences; and (4) dynamically changing the masking pattern applied to the training data." They trained on over 160GB of text instead of BERT's 16GB (Wikipedia plus books, in their measurement).

> [!PAPER] Liu et al. (2019), RoBERTa · Abstract
> [![The RoBERTa abstract: a replication study of BERT pretraining that measures the impact of key hyperparameters and training data size; we find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it](/img/papers/bert/p6-roberta.png)](/img/papers/bert/p6-roberta.png)
>
> **Context:** the first big follow-up from another lab (Facebook AI), nine months after BERT.
>
> **What it says:** "We find that BERT was significantly undertrained, and can match or exceed the performance of every model published after it."
>
> **Why it matters:** the architecture was fine; the training recipe had room to grow. This is why later "BERT-style" models usually mean RoBERTa-style training.

> [!DEFINITION] Dynamic masking
> Choosing new positions to mask every time a sentence is shown to the model. The original BERT pipeline chose the masked positions once, when the training data was prepared (RoBERTa calls this "static" masking). RoBERTa notes that BERT's data was duplicated 10 times "so that each sequence is masked in 10 different ways over the 40 epochs of training", so each fixed mask was still seen several times.

The NSP result is interesting, because it seems to contradict BERT's own Table 5 (Part 5), where removing NSP hurt. RoBERTa found that "removing the NSP loss matches or slightly improves downstream task performance". They offer a possible reason: "the original BERT implementation may only have removed the loss term while still retaining the SEGMENT-PAIR input format". In other words, BERT's "No NSP" model still trained on pairs of shorter text pieces; RoBERTa's trained on long, full blocks of text. Both results can be true, because they removed NSP in different ways.

**SpanBERT** (Joshi et al., July 2019) masks "contiguous random spans, rather than random tokens", and gains most on "span selection tasks such as question answering and coreference resolution". Its abstract reports 94.6 F1 on SQuAD 1.1 and 88.7 on SQuAD 2.0, "with the same training data and model size as BERT-large".

### 2. Make it smaller: ALBERT and DistilBERT

**ALBERT** (Lan et al., September 2019) uses "two parameter-reduction techniques":

> [!PAPER] Lan et al. (2019), ALBERT · Abstract
> [![The ALBERT abstract: two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT, and a self-supervised loss that focuses on modeling inter-sentence coherence](/img/papers/bert/p6-albert.png)](/img/papers/bert/p6-albert.png)
>
> **Context:** a Google follow-up aimed at size.
>
> **What it says:** "two parameter-reduction techniques to lower memory consumption and increase the training speed of BERT", plus "a self-supervised loss that focuses on modeling inter-sentence coherence".
>
> **Why it matters:** the second half replaces NSP, the part of BERT that later work found weakest.

- **factorized embedding parameterization**: the big vocabulary table is split into two small matrices, so its size no longer grows with the hidden size;
- **cross-layer parameter sharing**: all layers share the same weights, so the size no longer grows with the depth.

ALBERT also replaces NSP with **sentence-order prediction (SOP)**, a task about "inter-sentence coherence": are these two consecutive pieces in the right order or swapped? The paper describes it as designed to address "the ineffectiveness" of NSP. Their example of the size saving: "ALBERT-large has about 18x fewer parameters compared to BERT-large, 18M versus 334M".

> [!DEFINITION] Knowledge distillation
> Training a small "student" model to copy the outputs of a big "teacher" model. The student learns from the teacher's full probability lists, not just from the right answers, which is a richer signal.

**DistilBERT** (Sanh et al., October 2019) uses distillation during pre-training. Its abstract: "it is possible to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster."

> [!PAPER] Sanh et al. (2019), DistilBERT · Abstract
> [![The DistilBERT abstract: knowledge distillation during the pre-training phase reduces the size of a BERT model by 40 percent while retaining 97 percent of its language understanding capabilities and being 60 percent faster](/img/papers/bert/p6-distilbert.png)](/img/papers/bert/p6-distilbert.png)
>
> **Context:** a Hugging Face follow-up aimed at speed.
>
> **What it says:** "knowledge distillation during the pre-training phase" makes it possible "to reduce the size of a BERT model by 40%, while retaining 97% of its language understanding capabilities and being 60% faster".
>
> **Why it matters:** a much cheaper model to run, for a small loss in quality. Below we check the size claim and show what distillation trains on.

The distillation loss, for one masked position, compares the student's probabilities $$s$$ with the teacher's $$t$$ over the vocabulary:

$$
\mathcal{L}_{\text{hard}} = -\log s(y), \qquad \mathcal{L}_{\text{soft}} = -\sum_{w} t(w) \log s(w)
$$

where $$y$$ is the true word and the sum runs over every word $$w$$ of the vocabulary. The hard loss only rewards the right answer; the soft loss also rewards giving the teacher's runner-ups some probability. (DistilBERT combines this soft loss with the masked-LM loss and a cosine loss on the hidden vectors; here we show only the idea.) Measured on "i went to the [MASK] to deposit my paycheck." ([`bert_part6_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part6_math.py)):

```text
encoder parameters: teacher 108,891,648 (12 layers), student 66,362,880 (6 layers), ratio 0.609
teacher top 5: bank 0.901, office 0.009, store 0.008, teller 0.008, atm 0.006
student top 5: bank 0.365, store 0.035, mall 0.025, hotel 0.022, supermarket 0.020
hard-label loss  -log s(bank)               = 1.009
soft-target loss -sum_w t(w) log s(w)       = 1.467   (teacher entropy, the smallest possible value: 0.774)
```

The student has 0.609 of the teacher's encoder weights, the "40% smaller" of the abstract. The soft loss can never go below the teacher's own **entropy** (0.774 here), reached only when the student copies the teacher exactly.

{{FIG:p6_distill|Distillation on one sentence. The teacher (BERT-base) puts 0.901 on "bank" and small amounts on office, store, teller, atm. The student (DistilBERT) also picks "bank" but with 0.365. Training on the teacher's whole list (soft targets) gives the student more to learn from than the single right answer.}}

> [!DEFINITION] Entropy
> How spread out a probability list is: $$-\sum_w t(w) \log t(w)$$. 0 for a list that puts everything on one word; larger when the probability is spread over many words.

### 3. Fix the masked language model: XLNet, ELECTRA and DeBERTa

The masked LM has two weak spots, and two papers named them directly. We measure both later in this part.

**XLNet** (Yang et al., June 2019) says that "relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy". XLNet keeps bidirectional context but predicts words in many different random orders instead of using `[MASK]`.

> [!PAPER] Yang et al. (2019), XLNet · Abstract
> [![The XLNet abstract: relying on corrupting the input with masks, BERT neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy; XLNet is a generalized autoregressive pretraining method that maximizes the expected likelihood over all permutations of the factorization order](/img/papers/bert/p6-xlnet.png)](/img/papers/bert/p6-xlnet.png)
>
> **Context:** a CMU and Google follow-up that names two of BERT's weaknesses.
>
> **What it says:** BERT "neglects dependency between the masked positions and suffers from a pretrain-finetune discrepancy". XLNet instead maximises the likelihood "over all permutations of the factorization order".
>
> **Why it matters:** these are Limits 2 and 4 below, measured on the real model.

**ELECTRA** (Clark et al., March 2020) points at a different cost: BERT only learns from the masked tokens. ELECTRA replaces some tokens with "plausible alternatives sampled from a small generator network", then trains the main model to say, for every token, whether it was replaced. This is "more efficient than MLM because the task is defined over all input tokens rather than just the small subset that was masked out". Its abstract gives an example: "we train a model on one GPU for 4 days that outperforms GPT (trained using 30x more compute) on the GLUE natural language understanding benchmark".

> [!PAPER] Clark et al. (2020), ELECTRA · Abstract
> [![The ELECTRA abstract: replaced token detection; instead of masking, corrupt the input by replacing some tokens with plausible alternatives from a small generator network, then train a discriminator that predicts whether each token was replaced; more efficient than MLM because the task is defined over all input tokens rather than just the small subset that was masked out](/img/papers/bert/p6-electra.png)](/img/papers/bert/p6-electra.png)
>
> **Context:** a Stanford and Google follow-up aimed at training efficiency.
>
> **What it says:** "replaced token detection": a small generator fills some positions with plausible words, and the main model judges every token, original or replaced. This is efficient "because the task is defined over all input tokens rather than just the small subset that was masked out".
>
> **Why it matters:** it attacks Limit 3 head on: every position gives a training signal.

{{FIG:p6_electra|ELECTRA in three steps. Mask a few tokens; a small generator fills them ("chef" guessed right, "soup" plausible but wrong); the main model labels every token as original or replaced. BERT learns from 2 of these 5 positions, ELECTRA's main model from all 5.}}

**DeBERTa** (He et al., June 2020) changes attention itself: "each word is represented using two vectors that encode its content and position", instead of one vector that adds them together as BERT does (Part 2).

### 4. Use it differently: Sentence-BERT, and a modern BERT

**Sentence-BERT** (Reimers and Gurevych, August 2019) fixed a problem the paper itself warns about in footnote 6: BERT's `[CLS]` vector is not a good sentence vector without fine-tuning. Sentence-BERT says that averaging BERT's outputs or using the `[CLS]` output "yields rather bad sentence embeddings, often worse than averaging GloVe embeddings". It fine-tunes BERT so that sentences with similar meanings get similar vectors. We test this ourselves below.

> [!PAPER] Reimers and Gurevych (2019), Sentence-BERT · Abstract
> [![The Sentence-BERT abstract: finding the most similar pair in 10,000 sentences requires about 50 million inference computations, about 65 hours, with BERT; SBERT uses siamese and triplet networks to derive semantically meaningful sentence embeddings that can be compared using cosine-similarity, reducing this to about 5 seconds](/img/papers/bert/p6-sbert.png)](/img/papers/bert/p6-sbert.png)
>
> **Context:** the paper that made BERT useful for search and similarity.
>
> **What it says:** with BERT reading every pair together, finding the most similar pair among 10,000 sentences "requires about 50 million inference computations (~65 hours)". SBERT produces "semantically meaningful sentence embeddings that can be compared using cosine-similarity", which cuts this "to about 5 seconds".
>
> **Why it matters:** 50 million is the number of pairs: $$10{,}000 \times 9{,}999 / 2 \approx 5 \times 10^7$$. One vector per sentence turns that into 10,000 BERT runs plus cheap cosine comparisons.

{{FIG:p6_crossbi|Two ways to compare texts. Cross-encoder (BERT as in the paper): read the query and the passage together, one score per pair; accurate, but one BERT run for every pair. Bi-encoder (Sentence-BERT): one vector per text, compared by cosine; passage vectors can be computed once, ahead of time.}}

**ModernBERT** (Warner et al., December 2024) shows the idea is still alive. It calls encoder-only models like BERT "the workhorse of numerous production pipelines", and brings modern training to them: "Trained on 2 trillion tokens with a native 8192 sequence length". Compare that with BERT's 512 tokens.

> [!PAPER] Warner et al. (2024), ModernBERT · Abstract
> [![The ModernBERT abstract: encoder-only transformer models such as BERT are the workhorse of numerous production pipelines; ModernBERT is trained on 2 trillion tokens with a native 8192 sequence length](/img/papers/bert/p6-modernbert.png)](/img/papers/bert/p6-modernbert.png)
>
> **Context:** a 2024 encoder from Answer.AI, LightOn and others.
>
> **What it says:** encoders like BERT are "the workhorse of numerous production pipelines". ModernBERT is "Trained on 2 trillion tokens with a native 8192 sequence length".
>
> **Why it matters:** six years later, the BERT design is still in production use, now with modern training and 16 times the context.

{{FIG:p6_family|Which follow-up attacks which weakness. 512-token wall: ModernBERT (8,192 tokens). [MASK] mismatch: XLNet, ELECTRA. Only 15% of tokens teach: ELECTRA. Blanks filled independently: XLNet. No sentence vector: Sentence-BERT. Too big or slow: ALBERT, DistilBERT. Undertrained recipe: RoBERTa, SpanBERT.}}

## BERT's limits, measured

Every model has limits. Here are BERT's main ones. Where the paper itself mentions a limit, its lines are shown; where a limit can be measured, I measured it with [`bert_part6.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part6.py) on the public `bert-base-uncased` model.

### Limit 1: at most 512 tokens

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.2 · page 13
> [![Appendix A.2: longer sequences are disproportionately expensive because attention is quadratic to the sequence length. BERT is pre-trained with sequence length 128 for 90 percent of the steps, then 512 for the last 10 percent to learn the positional embeddings](/img/papers/bert/p6-length.png)](/img/papers/bert/p6-length.png)
>
> **Context:** Appendix A.2, the pre-training procedure (covered in Part 3).
>
> **What it says:** "attention is quadratic to the sequence length", so long inputs are expensive. BERT trains on 128-token sequences for 90% of the steps and on 512-token sequences for the last 10%, "to learn the positional embeddings".
>
> **Why it matters:** BERT has learned position vectors for positions 0 to 511, and no more. 512 is a hard wall.

> [!DEFINITION] Quadratic
> Growing with the square. If attention compares every token with every token, then 2 times more tokens means 2 × 2 = 4 times more comparisons, and 4 times more tokens means 16 times more.

Recall from Part 2 that BERT adds a **position embedding** to every token, looked up in a table with one row per position. The code reads the size of that table and then tries a long input:

```python
bert = AutoModel.from_pretrained("bert-base-uncased")
print(bert.embeddings.position_embeddings.weight.shape)     # rows = positions it knows

long_text = " ".join(["BERT reads the whole input at once, so every token can look at every other token."] * 40)
print(len(tok(long_text).input_ids))                                   # all tokens
print(len(tok(long_text, truncation=True, max_length=512).input_ids))  # what is kept
bert(**tok(long_text, return_tensors="pt"))                            # try all of them
```

```text
1. The 512-token limit
   position embedding table: 512 rows x 768 numbers
   a long text has 722 tokens; with truncation it keeps 512; the rest is thrown away
   feeding all 722 tokens: RuntimeError: The size of tensor a (722) must match the size of tensor b (512) at non-singleton dimension 1
```

The table has exactly 512 rows. A 722-token text either crashes the model or loses its last 210 tokens. A few long paragraphs are already too much, so long documents have to be cut into pieces. This is the limit ModernBERT's 8192-token context attacks.

> [!DEFINITION] Truncation
> Cutting an input to a maximum length and throwing away the rest.

The usual workaround, used by the released SQuAD code (Part 4), is to read a long text in **overlapping windows**. With window size $$w$$ content tokens (510, leaving room for `[CLS]` and `[SEP]`) and step $$s$$ between window starts, a text of $$N$$ tokens needs

$$
k = 1 + \left\lceil \frac{N - w}{s} \right\rceil \text{ windows}
$$

where $$\lceil \cdot \rceil$$ rounds up. For $$N = 720$$, $$w = 510$$, $$s = 382$$: $$k = 1 + \lceil 210 / 382 \rceil = 2$$. Windows also keep the attention cost linear in $$N$$: each window costs $$512^2$$ per head and layer, so $$k \times 512^2$$ in total, against $$(N + 2)^2$$ for one long input (if the model could take it):

```text
N =    720: windows =   2; one long input (N+2)^2 =     521,284; windows k*512^2 =     524,288; ratio 0.99
N =   2000: windows =   5; one long input (N+2)^2 =   4,008,004; windows k*512^2 =   1,310,720; ratio 3.06
N =  10000: windows =  26; one long input (N+2)^2 = 100,040,004; windows k*512^2 =   6,815,744; ratio 14.68
```

{{FIG:p6_windows|Reading a 720-token text as two overlapping windows of 512. Window 1 covers tokens 0 to 509, window 2 tokens 382 to 719; the 128 tokens in between are seen by both. Below: attention entries per head and layer, one long input against windows, for 720, 2,000 and 10,000 tokens.}}

The price: a token in window 1 never sees a token in window 2. Facts that are far apart in a long document cannot be combined. That is the real cost of the 512 wall, and why later models raised it.

### Limit 2: `[MASK]` never appears after pre-training

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1 · page 4
> [![Section 3.1: a downside is that we are creating a mismatch between pre-training and fine-tuning, since the MASK token does not appear during fine-tuning](/img/papers/bert/p6-mismatch.png)](/img/papers/bert/p6-mismatch.png)
>
> **Context:** Section 3.1, right after the masked LM is introduced (Part 3).
>
> **What it says:** "a downside is that we are creating a mismatch between pre-training and fine-tuning, since the [MASK] token does not appear during fine-tuning."
>
> **Why it matters:** the authors saw this limit themselves. Their fix, the 80/10/10 rule, softens it but does not remove it.

During pre-training, BERT sees `[MASK]` in about 12% of positions (80% of the 15% chosen). In real use it never sees `[MASK]` at all. Part 3 explained the 80% / 10% / 10% trick that reduces the gap, and Part 5 (Table 8) measured how much it matters. XLNet named this the "pretrain-finetune discrepancy".

### Limit 3: only 15% of tokens teach the model

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.1 · page 12
> [![Appendix A.1: compared to standard language model training, the masked LM only makes predictions on 15 percent of tokens in each batch, which suggests that more pre-training steps may be required for the model to converge](/img/papers/bert/p6-fifteen.png)](/img/papers/bert/p6-fifteen.png)
>
> **Context:** Appendix A.1, after the masking examples. The sentence continues on page 13 with "to converge".
>
> **What it says:** the masked LM "only make[s] predictions on 15% of tokens in each batch", so "more pre-training steps may be required".
>
> **Why it matters:** in a left-to-right model, every token is a prediction target. In BERT, 85% of the tokens give no training signal of their own.

{{FIG:p6_signal|Where the learning signal comes from. In BERT's masked LM only the masked positions get a loss. In ELECTRA every position is checked as real or replaced (shown here as "fake"), so every token teaches the model something.}}

Part 5 (Appendix C.1) showed that BERT does learn a little more slowly because of this, but still wins. ELECTRA turned this observation into a new pre-training task.

> [!DEFINITION] Loss
> A number that measures how wrong the model's predictions are. Training changes the weights to make the loss smaller. A position "gets a loss" if its prediction is scored and used for learning.

### Limit 4: two blanks are filled separately

When BERT fills several `[MASK]`s at once, it predicts each one **on its own**. It does not know what it put in the other blank. I masked both words of a two-word city name:

```text
3. Two [MASK]s at once: each one is predicted on its own
   i flew from [MASK] [MASK] to london last week.
     mask 1: new 0.042, the 0.037, chicago 0.037, la 0.028, portland 0.019
     mask 2: town 0.068, york 0.044, beach 0.025, city 0.021, ##town 0.017
   i flew from new [MASK] to london last week.  -> york 0.979, orleans 0.012, jersey 0.003
   i flew from los [MASK] to london last week.  -> angeles 1.000, vegas 0.000, ' 0.000
   i flew from hong [MASK] to london last week. -> kong 0.999, ##kong 0.001, china 0.000
```

Taking the top guess for each blank separately gives "new town". But once the first word is filled in, the second becomes almost certain: "york" (0.979) after "new", "angeles" after "los", "kong" after "hong". The two blanks depend on each other, and BERT's masked LM cannot use that when both are hidden. This is what XLNet meant by "neglects dependency between the masked positions".

In equations, the masked LM scores the two blanks $$a$$ and $$b$$ as if they were independent, while the true probability follows the chain rule of Part 1:

$$
\underbrace{p(a)\, p(b)}_{\text{what BERT's two guesses give}} \qquad \text{versus} \qquad \underbrace{p(a)\, p(b \mid a)}_{\text{the chain rule}}
$$

```text
"new town":  p(a) = 0.0424, p(b) = 0.0678, p(b | a) = 0.0000
   independent p(a) p(b) = 0.002876;  chain rule p(a) p(b | a) = 0.000000
"new york":  p(a) = 0.0424, p(b) = 0.0440, p(b | a) = 0.9794
   independent p(a) p(b) = 0.001868;  chain rule p(a) p(b | a) = 0.041548
best pair by the independent score (over 29 candidate pairs): "new town" 0.002876
best pair by the chain rule:                                  "new york" 0.041548
```

{{FIG:p6_twomask|The two-blank problem in four panels. Both blanks hidden: BERT guesses "new" for blank 1 and "town" for blank 2. Multiplying the two guesses picks "new town". Filling blank 1 first makes blank 2 "york" with 0.979. The chain rule picks "new york" by far.}}

> [!NOTE] Why "1.000" and "0.000"?
> The probabilities are rounded to three decimals. "1.000" means more than 0.9995, not exactly 1; "0.000" means less than 0.0005.

### Limit 5: no ready-made sentence vector

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1, footnote 6 · page 4
> [![Footnote 6: the vector C is not a meaningful sentence representation without fine-tuning, since it was trained with NSP](/img/papers/bert/p6-footnote6.png)](/img/papers/bert/p6-footnote6.png)
>
> **Context:** a footnote to the next sentence prediction task in Section 3.1.
>
> **What it says:** "The vector C is not a meaningful sentence representation without fine-tuning, since it was trained with NSP."
>
> **Why it matters:** many people still take BERT's `[CLS]` vector as "the meaning of the sentence". The paper warns against exactly that.

> [!DEFINITION] Cosine similarity
> A score for how much two vectors point the same way: 1 means the same direction, 0 means unrelated (at a right angle), and negative means opposite. It is the standard way to compare sentence vectors.

The formula, $$\cos(a, b) = a \cdot b \,/\, (\lVert a \rVert \lVert b \rVert)$$, on one real pair of `[CLS]` vectors:

```text
a = C("A man is playing a guitar."), b = C("A person is making music."), each with 768 numbers
a . b = 199.086;  |a| = 15.498;  |b| = 15.086
cos = a . b / (|a| |b|) = 199.086 / (15.498 x 15.086) = 0.851   (torch: 0.851)
toy check in 2-D: a = (3, 4), b = (4, 3): a . b = 24, |a| = |b| = 5, cos = 24 / 25 = 0.96
```

{{FIG:p6_cosine|Cosine similarity is the angle between two vectors. Toy, in 2-D: (3, 4) and (4, 3) have cos 0.96. Real, in 768-D: the raw [CLS] vectors of "A man is playing a guitar." and "A person is making music." have cos 0.851.}}

> [!DEFINITION] Sentence embedding
> One vector for a whole sentence, built so that sentences with similar meanings get vectors that point in similar directions. Search engines and chat-with-your-documents systems compare such vectors to find related text.

I compared six sentence pairs, three related and three unrelated, with three kinds of vector:

- **raw BERT [CLS]**: the vector C of `bert-base-uncased`, as the footnote warns against;
- **raw BERT, mean of tokens**: the average of all of BERT's output vectors;
- **all-MiniLM-L6-v2**: a public model from the Sentence-BERT authors' library. Its config says `model_type=bert`, so it has BERT's architecture, but it is a small distilled model (6 layers, 384 numbers per token, 22.7M parameters, from the script's output), fine-tuned on over 1 billion sentence pairs (its model card) to make similar sentences close.

```text
   pair (R = related, U = unrelated)                                   raw [CLS]  raw mean  MiniLM
   R A man is playing a guitar.   | A person is making music.              0.851     0.746   0.590
   R How do I reset my password?  | I forgot my login details.             0.929     0.672   0.541
   R The weather is lovely today. | It is sunny and warm outside.          0.951     0.857   0.705
   U A man is playing a guitar.   | The stock market fell sharply.         0.776     0.584  -0.064
   U How do I reset my password?  | Penguins live in Antarctica.           0.795     0.517  -0.035
   U The weather is lovely today. | He parked the truck in the garage.     0.873     0.598   0.013
   average related / unrelated:  raw [CLS] 0.910 / 0.815,  raw mean 0.758 / 0.566,  MiniLM 0.612 / -0.029
```

{{FIG:p6_similarity|The same numbers as a picture. With raw BERT's [CLS] vector, every pair looks similar (0.776 to 0.951), and an unrelated pair (0.873) scores higher than a related one (0.851). The model fine-tuned for similarity puts all unrelated pairs near 0 and all related pairs above 0.5.}}

What the numbers say:

- **Raw [CLS] says everything is similar.** Even "the weather is lovely" and "he parked the truck in the garage" get 0.873, higher than the related guitar pair (0.851). You cannot pick a threshold that separates the two groups. The footnote is right.
- **Averaging BERT's outputs is better but not clean**: related pairs average 0.758 and unrelated 0.566, but the gap is small.
- **A model fine-tuned for similarity separates them clearly**: unrelated pairs score between -0.064 and 0.013, related pairs between 0.541 and 0.705.

This is only six pairs, so treat it as an illustration, not a benchmark. Sentence-BERT's paper measures the same effect on standard similarity datasets.

### Limit 6: it does not write text

BERT is an encoder. It reads a whole input and gives back one vector per token; it has no natural way to write a long answer one word at a time. As footnote 4 of the paper puts it (Part 2), the left-context-only Transformer is called a "Transformer decoder" "since it can be used for text generation". Chatbots are built on decoders, in the line of GPT, not on BERT.

> [!PAPER] Devlin et al. (2018), BERT · Section 3, footnote 4 · page 4
> [![Footnote 4 of the BERT paper: the bidirectional Transformer is often referred to as a Transformer encoder while the left-context-only version is referred to as a Transformer decoder since it can be used for text generation](/img/papers/bert/p6-footnote4.png)](/img/papers/bert/p6-footnote4.png)
>
> **Context:** a footnote to the model description in Section 3.
>
> **What it says:** the bidirectional version "is often referred to as a 'Transformer encoder'", while the left-context-only version is a "'Transformer decoder' since it can be used for text generation".
>
> **Why it matters:** the paper itself draws the line: generation needs left-only context. BERT is on the other side of that line by design.

### Limit 7: pre-training is expensive

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.2 · page 13
> [![Appendix A.2: training of BERT-base was performed on 4 Cloud TPUs in Pod configuration, 16 TPU chips total; BERT-large on 16 Cloud TPUs, 64 TPU chips total; each pre-training took 4 days to complete](/img/papers/bert/p6-compute.png)](/img/papers/bert/p6-compute.png)
>
> **Context:** Appendix A.2, the pre-training procedure.
>
> **What it says:** BERT-base was trained on "4 Cloud TPUs in Pod configuration (16 TPU chips total)", BERT-large on "16 Cloud TPUs (64 TPU chips total)", and "each pre-training took 4 days to complete".
>
> **Why it matters:** in 2018, few people outside large labs could pre-train BERT themselves. That is why the public release of the weights mattered so much.

> [!DEFINITION] TPU
> Tensor Processing Unit: a chip designed by Google for neural network maths, similar in role to a GPU.

The good news is the paper's own: fine-tuning is cheap. Section 3.2 says every result in the paper "can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model" (Part 4).

> [!PAPER] Devlin et al. (2018), BERT · Section 3.2 · page 5
> [![From Section 3.2: compared to pre-training, fine-tuning is relatively inexpensive; all of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model](/img/papers/bert/p6-finetune-cost.png)](/img/papers/bert/p6-finetune-cost.png)
>
> **Context:** the end of Section 3.2.
>
> **What it says:** "All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model."
>
> **Why it matters:** the expensive step is done once, by someone else; everyone downstream pays only the cheap step.

How big was the pre-training budget, in tokens? From the numbers in Appendix A.2:

```text
tokens per batch = 256 x 512 = 131,072  (the paper rounds to 128,000)
tokens seen      = 1,000,000 steps x 131,072 = 131,072,000,000  (an upper bound: 90% of steps use length 128)
with 90% of steps at length 128 and 10% at 512: 1,000,000 x 256 x (0.9 x 128 + 0.1 x 512) = 42,598,400,000
predicted tokens (15%) of that: 6,389,760,000
chip-days: BERT-base 16 chips x 4 days = 64; BERT-large 64 chips x 4 days = 256
```

One subtlety: the paper's "approximately 40 epochs over the 3.3 billion word corpus" matches the upper bound (131 billion tokens ÷ 3.3 billion words ≈ 40). If the batch stayed at 256 sequences during the length-128 phase, the model saw about 42.6 billion tokens, roughly 13 passes. The paper does not say whether the batch size changed between the two phases, so the exact number of epochs cannot be recovered from the text.

## How BERT-style models are used today

Large chatbots get the headlines, but small encoders like BERT do a lot of quiet work, because they are fast and cheap to run on every piece of text. ModernBERT's authors call them "the workhorse of numerous production pipelines". Typical jobs:

- **Classification**: spam or not, which team should handle this support ticket, is this review positive. This is the GLUE recipe from Part 4: `[CLS]` vector, one layer, fine-tune.
- **Finding names and things (NER)**: people, places, products, drug names in text. This is the token-level recipe from Part 4.
- **Search and retrieval**: sentence-embedding models in the Sentence-BERT line turn every document into a vector once, then compare vectors at search time. This is how many "chat with your documents" systems find the right passages.
- **Reranking**: a **cross-encoder** reads the question and one candidate passage together, exactly like BERT's sentence pairs, and scores how well they match. Sentence-BERT describes BERT itself as a cross-encoder: "Two sentences are passed to the transformer network". It is accurate but slow, so it is used to re-order a short list found by the faster vector search.

> [!DEFINITION] Cross-encoder and bi-encoder
> A **cross-encoder** reads two texts together in one input (like BERT's sentence A and sentence B) and outputs one score. A **bi-encoder** turns each text into its own vector separately, so vectors can be computed ahead of time and compared quickly. Sentence-BERT made BERT into a good bi-encoder.

Here is the NER use case with a real public checkpoint, `dslim/bert-base-NER`. Its model card says it is `bert-base-cased` fine-tuned on the CoNLL-2003 dataset (the same NER task as the paper's Section 5.3), with four entity types: person (PER), location (LOC), organisation (ORG) and miscellaneous (MISC).

```python
from transformers import pipeline
ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
ner("Ada Lovelace was born in London and worked with Charles Babbage on the Analytical Engine.")
```

```text
4. Named entities with dslim/bert-base-NER (BERT-base-cased fine-tuned on CoNLL-2003)
   Ada Lovelace was born in London and worked with Charles Babbage on the Analytical Engine.
     Ada Lovelace           PER   0.999
     London                 LOC   0.999
     Charles Babbage        PER   0.932
     Analy                  ORG   0.723
     Engine                 ORG   0.781
   token by token, for "Analytical Engine":
     Ana                    B-ORG  0.901
     ##ly                   I-ORG  0.544
     Engine                 I-ORG  0.781
```

The people and the place are found with high confidence. "Analytical Engine" (a machine, not really an organisation) shows a real rough edge: WordPiece split "Analytical" into `Ana`, `##ly` and `##tical`, the last piece got no entity label, and the result came out broken into "Analy" and "Engine". The model card warns about exactly this: the model "occassionally tags subword tokens as entities". This is why Section 5.3 of the paper labels only the **first** sub-token of each word (Part 5).

## The whole paper on one page

{{FIG:p6_napkin|BERT on a napkin. Text becomes WordPiece tokens with [CLS] and [SEP], plus segment and position embeddings. A deep bidirectional encoder is pre-trained with the masked LM and next sentence prediction. For each task, one small layer is added on top and everything is fine-tuned.}}

{{FIG:p6_pipeline|One input all the way through BERT-base. 1: WordPiece tokens with [CLS], [SEP] and a [MASK]. 2: three 768-number vectors per token (token, segment, position), added. 3: twelve identical encoder layers, every token attending to every token. 4: one output vector per token, C and T₁ to T₁₀. 5: a small head reads what the task needs: C for classification and NSP, T at [MASK] for the masked LM, every Tᵢ for tagging, Tᵢ·S and Tᵢ·E for answer spans.}}

Here is the paper, part by part, with its key numbers. Every number is from the paper, with its location.

**[Part 1: The Big Idea](part-1-the-big-idea.md)** (Abstract, §1). Language models read in one direction, which is a real limit for tasks like question answering, where "it is crucial to incorporate context from both directions" (§1). BERT pre-trains a deep bidirectional Transformer by hiding words and predicting them. It set new state-of-the-art results on eleven tasks, including GLUE 80.5 (+7.7 points), MultiNLI 86.7% (+4.6), SQuAD v1.1 Test F1 93.2 (+1.5) and SQuAD v2.0 Test F1 83.1 (+5.1) (Abstract).

**[Part 2: Inside BERT](part-2-architecture-and-input.md)** (§2, §3). BERT is a Transformer encoder. BERT-base has L=12 layers, H=768, A=12 heads and 110M parameters; BERT-large has L=24, H=1024, A=16 and 340M (§3). The feed-forward size is 4H: 3072 and 4096 (footnote 3). Text is cut into WordPiece tokens from a 30,000-token vocabulary; every input starts with `[CLS]`, sentences are separated by `[SEP]`, and each token's input is the sum of a token, a segment and a position embedding (§3, Figure 2).

**[Part 3: Pre-training](part-3-pre-training.md)** (§3.1, A.1, A.2). Masked LM: choose 15% of the WordPiece tokens; replace them with `[MASK]` 80% of the time, a random token 10% and leave them unchanged 10% (§3.1); random replacement is then only 1.5% of all tokens (A.1). Next sentence prediction: half the time sentence B really follows A (IsNext), half the time it is random (NotNext); the final model reaches 97%-98% on it (footnote 5). Data: BooksCorpus (800M words) and English Wikipedia (2,500M words) (§3.1). Training: batches of 256 sequences of 512 tokens ("128,000 tokens/batch") for 1,000,000 steps, "approximately 40 epochs over the 3.3 billion word corpus", Adam with learning rate 1e-4, 10,000 warm-up steps, dropout 0.1, GELU, 90% of the steps at length 128 (A.2).

**[Part 4: Fine-tuning and Results](part-4-fine-tuning-and-results.md)** (§3.2, §4, A.3, A.5, B.1). For classification, the only new weights are W of size K×H, and the loss is $$\log(\text{softmax}(CW^\top))$$ (§4.1). On GLUE, BERT-base and BERT-large average 79.6 and 82.1, against 75.1 for OpenAI GPT (Table 1); on the official leaderboard BERT-large scores 80.5 against GPT's 72.8 (§4.1). For SQuAD, only a start vector S and an end vector E are new; a span from i to j scores $$S \cdot T_i + E \cdot T_j$$ with $$j \ge i$$ (§4.2). SQuAD v1.1 best Test F1 93.2 (ensemble with TriviaQA, Table 2); SQuAD v2.0 Test F1 83.1, +5.1 over the previous best (§4.3, Table 3); SWAG test accuracy 86.3, +27.1% over ESIM+ELMo and +8.3% over GPT (§4.4, Table 4). Good fine-tuning settings: batch size 16 or 32, learning rate 5e-5, 3e-5 or 2e-5, and 2, 3 or 4 epochs (A.3).

**[Part 5: Ablations](part-5-ablations.md)** (§5, A.4, C.1, C.2). Removing NSP hurts (MNLI 84.4 → 83.9, QNLI 88.4 → 84.9); training left-to-right hurts much more (MRPC 77.5, SQuAD F1 77.8 instead of 86.7 and 88.5) (Table 5). Bigger models are better on every task tried, from 3 layers to 24 (Table 6). Using BERT's frozen features (concatenating the last four layers) is "only 0.3 F1 behind fine-tuning the entire model" on named entity recognition (§5.3, Table 7). Training for 1M steps instead of 500k adds "almost 1.0%" on MNLI (C.1), and the 80/10/10 masking mix gives the best named entity recognition results, especially in the feature-based setting, while MNLI barely changes (Table 8).

**This part** (§6). The contribution is to generalise unsupervised pre-training "to deep *bidirectional* architectures" so "the same pre-trained model" can do many tasks (§6).

## The key words, in one list

Each term links to the part where it is first explained.

- **Pre-training**, **fine-tuning**, **encoder**, **feature-based approach**, **language model**, **unidirectional**, **Cloze task**: [Part 1](part-1-the-big-idea.md)
- **Transfer learning**, **L / H / A**, **parameters**, **WordPiece**, **`[CLS]` and `[SEP]`**, **segment and position embeddings**: [Part 2](part-2-architecture-and-input.md)
- **Masked LM**, **the 80 / 10 / 10 rule**, **next sentence prediction**, **cross-entropy loss**, **perplexity**, **warm-up**, **GELU**: [Part 3](part-3-pre-training.md)
- **GLUE**, **SQuAD**, **span**, **F1 and exact match**, **SWAG**, **dev and test sets**: [Part 4](part-4-fine-tuning-and-results.md)
- **Ablation**, **feature extraction from layers**, **masking strategies**: [Part 5](part-5-ablations.md)
- **Dynamic masking**, **distillation**, **sentence embedding**, **cross-encoder**: this part

> [!TAKEAWAYS] Key takeaways
> - The conclusion in one line: unsupervised pre-training already worked for one-directional models; BERT's "major contribution" is making it work for **deep bidirectional** models, so one pre-trained model handles many tasks.
> - BERT's impact was fast: **Best Long Paper** at NAACL 2019, and in October 2019 Google said BERT would help understand **one in 10** US English searches.
> - Later work kept the idea and fixed the training: **RoBERTa** (train longer on more data, drop NSP), **SpanBERT** (mask spans), **ALBERT** and **DistilBERT** (smaller), **XLNet**, **ELECTRA** and **DeBERTa** (better than plain masking), **Sentence-BERT** (good sentence vectors), **ModernBERT** (8192-token inputs).
> - Measured limits: a hard **512-token** wall (a 722-token input crashes or loses 210 tokens); `[MASK]` never appears after pre-training; only **15%** of tokens give a training signal; two blanks are filled independently ("new town", while "new" alone gives "york" at 0.979).
> - The paper's footnote 6 holds: raw `[CLS]` vectors scored unrelated pairs up to **0.873**, above a related pair; a model fine-tuned for similarity put unrelated pairs near 0.
> - BERT-style encoders are still used every day for classification, NER, search and reranking, because they are small, fast and cheap to fine-tune.

<details>
<summary>Run it yourself</summary>

Every measurement in this part comes from [`code/papers/bert/bert_part6.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part6.py). It runs on a laptop CPU in about a minute and downloads `bert-base-uncased`, `sentence-transformers/all-MiniLM-L6-v2` and `dslim/bert-base-NER` the first time.

```bash
pip install torch transformers
python bert_part6.py      # prints everything above, writes results/part6.json
```

<figure><img src="/img/papers/bert/part6-run.png" alt="Terminal output of bert_part6.py: the 512-token limit, cosine similarities for six sentence pairs with three kinds of vectors, two masks predicted separately and then one at a time, and named entities found by a fine-tuned BERT-base model" loading="lazy" /><figcaption>The real output of bert_part6.py.</figcaption></figure>

</details>

## References

**The BERT paper**

1. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019 ([ACL Anthology](https://aclanthology.org/N19-1423/)). Best Long Paper at NAACL 2019.
2. Google Research. [BERT code and pre-trained models](https://github.com/google-research/bert).

**Later work discussed in this part**

3. Y. Liu et al. [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692). arXiv 2019.
4. M. Joshi, D. Chen, Y. Liu, D. S. Weld, L. Zettlemoyer, O. Levy. [*SpanBERT: Improving Pre-training by Representing and Predicting Spans*](https://arxiv.org/abs/1907.10529). TACL 2020.
5. Z. Lan et al. [*ALBERT: A Lite BERT for Self-supervised Learning of Language Representations*](https://arxiv.org/abs/1909.11942). ICLR 2020.
6. V. Sanh, L. Debut, J. Chaumond, T. Wolf. [*DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter*](https://arxiv.org/abs/1910.01108). NeurIPS EMC² workshop 2019.
7. Z. Yang, Z. Dai, Y. Yang, J. Carbonell, R. Salakhutdinov, Q. V. Le. [*XLNet: Generalized Autoregressive Pretraining for Language Understanding*](https://arxiv.org/abs/1906.08237). NeurIPS 2019.
8. K. Clark, M.-T. Luong, Q. V. Le, C. D. Manning. [*ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators*](https://arxiv.org/abs/2003.10555). ICLR 2020.
9. P. He, X. Liu, J. Gao, W. Chen. [*DeBERTa: Decoding-enhanced BERT with Disentangled Attention*](https://arxiv.org/abs/2006.03654). ICLR 2021.
10. N. Reimers, I. Gurevych. [*Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*](https://arxiv.org/abs/1908.10084). EMNLP 2019.
11. B. Warner et al. [*Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference*](https://arxiv.org/abs/2412.13663) (ModernBERT). arXiv 2024.
12. G. Hinton, O. Vinyals, J. Dean. [*Distilling the Knowledge in a Neural Network*](https://arxiv.org/abs/1503.02531). NeurIPS Deep Learning workshop 2014.

**Other sources used in this part**

13. P. Nayak. [*Understanding searches better than ever before*](https://blog.google/products/search/search-language-understanding-bert/). Google blog, 25 October 2019.
14. Public models used in the code: [distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) and [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
15. Code for this part: [`bert_part6.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part6.py) and [`bert_part6_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part6_math.py).
