---
title: "Fine-tuning and Results: GLUE, SQuAD and SWAG"
description: "Sections 3.2 and 4 of the BERT paper, line by line: how one pre-trained model becomes a classifier, a question answerer and a multiple-choice solver by adding a tiny output layer, and what the results tables really say. With a real fine-tune of BERT-base on SST-2 and MRPC, and our own span search scored on the full SQuAD dev sets."
part: 4
covers: "§3.2, §4, A.3, A.5, B.1"
date: 2026-10-04
tags: [bert, fine-tuning, glue, squad]
---

Part 3 built a pre-trained BERT: a model that has read 3.3 billion words and learned to fill in blanks and to tell whether two sentences belong together. Nobody has asked it to do anything useful yet.

This part is about the second step of the recipe: **fine-tuning**. We read Section 3.2 (how fine-tuning works), then Section 4 (the experiments on eleven tasks), and the appendix sections that go with them (A.3, A.5 and B.1). Along the way we fine-tune BERT ourselves on a laptop GPU and check every formula in code.

## One model, many tasks {§3.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.2 · page 5
> [![Section 3.2, first paragraph: fine-tuning is straightforward since self-attention lets BERT model many downstream tasks by swapping out the appropriate inputs and outputs. Encoding a concatenated text pair with self-attention effectively includes bidirectional cross attention between two sentences](/img/papers/bert/p4-finetune-swap.png)](/img/papers/bert/p4-finetune-swap.png)
>
> **Context:** the start of Section 3.2, "Fine-tuning BERT", right after the pre-training data.
>
> **What it says:** fine-tuning is "straightforward": BERT handles many tasks, with one text or two, "by swapping out the appropriate inputs and outputs". Older systems first encoded the two texts separately and then compared them with a separate step called cross attention. BERT does both at once, because "encoding a concatenated text pair with self-attention effectively includes *bidirectional* cross attention between two sentences".
>
> **Why it matters:** this is why one architecture fits so many tasks. Nothing inside BERT changes from task to task; only what goes in and what comes out.

> [!DEFINITION] Cross attention
> Attention where the words of one text look at the words of another text. In older question-answering systems, the question was read by one network, the passage by another, and a separate cross-attention layer let them look at each other.

The key sentence is the last one. Put a question and a passage into one sequence, `[CLS] question [SEP] passage [SEP]`, and let every token attend to every other token. A question word can now look at passage words, and a passage word can look at question words, in all 12 layers. That *is* cross attention, in both directions, for free. BERT needs no special pair-matching layer.

> [!DEFINITION] Concatenated
> Joined end to end. Two texts concatenated into one sequence sit side by side, separated by a `[SEP]` token.

## Plugging a task into BERT {§3.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.2 · page 5
> [![Section 3.2, second paragraph, first half: for each task, plug in the task-specific inputs and outputs and fine-tune all the parameters end-to-end. Sentence A and B are analogous to sentence pairs in paraphrasing, hypothesis-premise pairs in entailment, question-passage pairs in question answering](/img/papers/bert/p4-finetune-inputs.png)](/img/papers/bert/p4-finetune-inputs.png)
> [![The paragraph continues in the right column: a degenerate text-empty pair in text classification or sequence tagging. Token representations go into an output layer for token-level tasks, the CLS representation into an output layer for classification](/img/papers/bert/p4-finetune-inputs2.png)](/img/papers/bert/p4-finetune-inputs2.png)
>
> **Context:** the second paragraph of Section 3.2. It runs from the bottom of the left column into the right column.
>
> **What it says:** for each task, "plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end". Sentence A and sentence B from pre-training become (1) the two sentences in paraphrasing, (2) the hypothesis and premise in entailment, (3) the question and passage in question answering, and (4) "a degenerate text-∅ pair" (one text and nothing) for classification or tagging. At the output, token vectors feed token-level tasks, and the `[CLS]` vector feeds classification.
>
> **Why it matters:** this is the whole interface. Every task in the paper is described by two choices: what goes into slots A and B, and which output vectors you read.

> [!DEFINITION] End-to-end
> Training every weight of the whole system together, from the input tokens to the final answer, with one loss. Nothing is frozen.

> [!DEFINITION] Degenerate
> A mathematician's word for "a trivial special case". A text-∅ pair is a pair whose second half is empty (∅ is the symbol for the empty set).

<figure class="fig"><svg viewBox="0 0 760 282" role="img" aria-label="Every task is written as a pair in the same input format BERT saw during pre-training: sentence A in segment A, sentence B in segment B. A single-text task leaves B empty."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Pre-training saw pairs: [CLS] sentence A [SEP] sentence B [SEP]. Fine-tuning fills the same two slots.</text><text class="t-note" x="20.0" y="69.0" text-anchor="start">paraphrase (MRPC, QQP)</text><rect class="box" x="222.0" y="48.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="246.0" y="68.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="274.0" y="48.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="322.0" y="68.5" text-anchor="middle">sentence 1</text><rect class="box" x="374.0" y="48.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="398.0" y="68.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="426.0" y="48.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="474.0" y="68.5" text-anchor="middle">sentence 2</text><rect class="box" x="526.0" y="48.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="550.0" y="68.5" text-anchor="middle">[SEP]</text><line class="edge" x1="580.0" y1="64.0" x2="598.0" y2="64.0" marker-end="url(#ah)"/><text class="t-tick" x="602.0" y="69.0" text-anchor="start">same meaning?</text><text class="t-note" x="20.0" y="121.0" text-anchor="start">entailment (MNLI, RTE)</text><rect class="box" x="222.0" y="100.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="246.0" y="120.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="274.0" y="100.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="322.0" y="120.5" text-anchor="middle">premise</text><rect class="box" x="374.0" y="100.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="398.0" y="120.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="426.0" y="100.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="474.0" y="120.5" text-anchor="middle">hypothesis</text><rect class="box" x="526.0" y="100.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="550.0" y="120.5" text-anchor="middle">[SEP]</text><line class="edge" x1="580.0" y1="116.0" x2="598.0" y2="116.0" marker-end="url(#ah)"/><text class="t-tick" x="602.0" y="121.0" text-anchor="start">entail / contradict / neutral</text><text class="t-note" x="20.0" y="173.0" text-anchor="start">question answering (SQuAD)</text><rect class="box" x="222.0" y="152.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="246.0" y="172.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="274.0" y="152.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="322.0" y="172.5" text-anchor="middle">question</text><rect class="box" x="374.0" y="152.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="398.0" y="172.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="426.0" y="152.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="474.0" y="172.5" text-anchor="middle">passage</text><rect class="box" x="526.0" y="152.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="550.0" y="172.5" text-anchor="middle">[SEP]</text><line class="edge" x1="580.0" y1="168.0" x2="598.0" y2="168.0" marker-end="url(#ah)"/><text class="t-tick" x="602.0" y="173.0" text-anchor="start">answer span</text><text class="t-note" x="20.0" y="225.0" text-anchor="start">one text (SST-2, CoLA, NER)</text><rect class="box" x="222.0" y="204.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="246.0" y="224.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="274.0" y="204.0" width="96.0" height="32.0" rx="6"/><text class="t-tick" x="322.0" y="224.5" text-anchor="middle">the text</text><rect class="box" x="374.0" y="204.0" width="48.0" height="32.0" rx="6"/><text class="t-muted" x="398.0" y="224.5" text-anchor="middle">[SEP]</text><rect class="box-ghost" x="426.0" y="204.0" width="96.0" height="32.0" rx="6"/><text class="t-muted" x="474.0" y="224.5" text-anchor="middle">empty</text><line class="edge" x1="580.0" y1="220.0" x2="598.0" y2="220.0" marker-end="url(#ah)"/><text class="t-tick" x="602.0" y="225.0" text-anchor="start">label or tags</text><text class="t-s1" x="322.0" y="268.0" text-anchor="middle">segment A</text><text class="t-s2" x="474.0" y="268.0" text-anchor="middle">segment B</text></svg><figcaption>The input side of every task. Pre-training taught BERT the format [CLS] A [SEP] B [SEP]. A paraphrase task puts its two sentences there; entailment puts the premise and the hypothesis; question answering puts the question and the passage; a single-text task leaves B empty.</figcaption></figure>

The output side has only two cases:

- **Sentence-level tasks** (one answer per input, like "positive" or "entailment") read the final vector of `[CLS]`, called $$C$$, and add one small classification layer.
- **Token-level tasks** (one answer per token, like named-entity tags or the start and end of an answer) read the final vector of every token, $$T_1, T_2, \dots$$, and add one small layer that is applied to each token.

### Figure 4: the four shapes of a task {§A.5}

The paper draws these cases in Appendix A.5.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.5 · page 14
> [![Appendix A.5: task-specific models are formed by incorporating BERT with one additional output layer, so a minimal number of parameters are learned from scratch. Tasks (a) and (b) are sequence-level, (c) and (d) are token-level](/img/papers/bert/p4-a5.png)](/img/papers/bert/p4-a5.png)
>
> **Context:** Appendix A.5, which exists only to introduce Figure 4.
>
> **What it says:** each task model is BERT plus "one additional output layer, so a minimal number of parameters need to be learned from scratch". In Figure 4, "(a) and (b) are sequence-level tasks while (c) and (d) are token-level tasks". $$E$$ is the input embedding and $$T_i$$ the output vector of token $$i$$.
>
> **Why it matters:** "learned from scratch" is the key phrase. Only the tiny output layer starts random; everything else starts from pre-training.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.5, Figure 4 · page 15
> [![Figure 4 of the BERT paper: four diagrams. (a) Sentence pair classification (MNLI, QQP, QNLI, STS-B, MRPC, RTE, SWAG): the class label comes from C. (b) Single sentence classification (SST-2, CoLA): class label from C. (c) Question answering (SQuAD v1.1): start and end span read from the passage tokens. (d) Single sentence tagging (CoNLL-2003 NER): one tag per token, like O and B-PER](/img/papers/bert/p4-figure4.png)](/img/papers/bert/p4-figure4.png)
>
> **Context:** Figure 4, the picture for all of Section 4.
>
> **What it says:** the same blue BERT box appears four times. Only the input (one sentence or two) and the red arrows at the top (which output vectors are used) change.
>
> **Why it matters:** compare the four boxes: they are identical. That is the paper's claim of "minimal difference between the pre-trained architecture and the final downstream architecture", drawn.

How to read each panel:

- **(a) Sentence pair classification.** Two sentences go in; the red arrow comes out of $$C$$ only. Tasks: MNLI, QQP, QNLI, STS-B, MRPC, RTE, and SWAG.
- **(b) Single sentence classification.** One sentence; the arrow comes out of $$C$$. Tasks: SST-2, CoLA.
- **(c) Question answering.** Question and paragraph; the arrows come out of the *paragraph* tokens, marking where the answer starts and ends. Task: SQuAD v1.1.
- **(d) Single sentence tagging.** One sentence; an arrow comes out of *every* token with a tag such as `B-PER` (beginning of a person's name) or `O` (outside any name). Task: CoNLL-2003 NER, which Part 5 covers.

## Fine-tuning is cheap {§3.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.2 and footnote 7 · page 5
> [![Section 3.2, last paragraph: compared to pre-training, fine-tuning is relatively inexpensive. All results can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model](/img/papers/bert/p4-finetune-cost.png)](/img/papers/bert/p4-finetune-cost.png)
> [![Footnote 7: for example, the BERT SQuAD model can be trained in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of 91.0%](/img/papers/bert/p4-footnote7.png)](/img/papers/bert/p4-footnote7.png)
>
> **Context:** the end of Section 3.2, with its footnote.
>
> **What it says:** "All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model." Footnote 7 gives an example: the SQuAD model trains "in around 30 minutes on a single Cloud TPU to achieve a Dev F1 score of 91.0%".
>
> **Why it matters:** pre-training took 4 days on 16 to 64 TPU chips (Part 3). Fine-tuning takes an hour. That gap is what made BERT practical for everyone: one team pays for pre-training once, and everyone else fine-tunes.

> [!DEFINITION] TPU and GPU
> Chips built to do the big matrix multiplications of neural networks very fast. A **GPU** (graphics processing unit) started life drawing video-game graphics. A **TPU** (tensor processing unit) is Google's own chip, rented through Google Cloud.

We will test the "few hours on a GPU" claim ourselves later in this part, on a laptop.

## The fine-tuning settings {§A.3}

Which settings did the authors use for fine-tuning? Appendix A.3 answers.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.3 · pages 13 and 14
> [![Appendix A.3: for fine-tuning, most hyperparameters are the same as in pre-training, except the batch size, learning rate and number of training epochs. Dropout is kept at 0.1. Batch size 16 or 32](/img/papers/bert/p4-a3.png)](/img/papers/bert/p4-a3.png)
> [![Appendix A.3 continued: learning rate 5e-5, 3e-5 or 2e-5; number of epochs 2, 3 or 4. Large data sets with 100k+ labeled examples were far less sensitive to the choice than small data sets; it is reasonable to run an exhaustive search and pick the best model on the development set](/img/papers/bert/p4-a3b.png)](/img/papers/bert/p4-a3b.png)
>
> **Context:** Appendix A.3, "Fine-tuning Procedure". It starts at the bottom of page 13 and ends at the top of page 14.
>
> **What it says:** everything is as in pre-training "with the exception of the batch size, learning rate, and number of training epochs". Dropout stays at 0.1. Good values: batch size 16 or 32; learning rate 5e-5, 3e-5 or 2e-5; 2, 3 or 4 epochs. Big datasets (100k+ examples) "were far less sensitive to hyperparameter choice than small data sets", and since fine-tuning is fast, you can simply try all combinations.
>
> **Why it matters:** this small grid is the recipe almost everyone used for BERT fine-tuning for years.

> [!DEFINITION] Hyperparameter
> A setting you choose before training, as opposed to a weight the model learns. Batch size, learning rate and number of epochs are hyperparameters.

> [!DEFINITION] Batch, learning rate and epoch
> A **batch** is the group of examples the model looks at before each weight update (here 16 or 32). The **learning rate** is how big each update step is; 2e-5 means 0.00002. An **epoch** is one full pass over the training data.

> [!DEFINITION] Development (dev) set and test set
> Labelled examples kept out of training. You use the **dev set** to make choices, like which learning rate is best. The **test set** is used once, at the end, to report the final score. For GLUE and SQuAD the test answers are hidden on an evaluation server.

The learning rates are tiny compared with pre-training (1e-4, Part 3). That is on purpose: the pre-trained weights are already good, and fine-tuning should nudge them, not overwrite them.

The size of the grid is 2 batch sizes × 3 learning rates × 3 epoch counts = 18 runs per task. With runs that take minutes, that is affordable.

## Section 4: the experiments {§4}

> [!PAPER] Devlin et al. (2018), BERT · Section 4 and 4.1 · page 5
> [![The start of Section 4, Experiments: fine-tuning results on 11 NLP tasks. Section 4.1 GLUE: the General Language Understanding Evaluation benchmark is a collection of diverse natural language understanding tasks](/img/papers/bert/p4-glue-intro.png)](/img/papers/bert/p4-glue-intro.png)
>
> **Context:** the beginning of Section 4.
>
> **What it says:** this section presents "BERT fine-tuning results on 11 NLP tasks". The first group is GLUE, "a collection of diverse natural language understanding tasks", described in Appendix B.1.
>
> **Why it matters:** the eleven tasks are the eight GLUE tasks in Table 1, SQuAD v1.1, SQuAD v2.0 and SWAG.

> [!DEFINITION] GLUE
> The General Language Understanding Evaluation benchmark (Wang et al., 2018): nine English tasks about single sentences or pairs of sentences, with one public leaderboard. A model's GLUE score is the average over the tasks.

## The nine GLUE tasks {§B.1}

Before the results, let us meet the tasks. Appendix B.1 describes each one.

> [!PAPER] Devlin et al. (2018), BERT · Appendix B.1 · page 14
> [![Appendix B.1: GLUE results come from the GLUE leaderboard and the OpenAI blog. MNLI: given a pair of sentences, predict entailment, contradiction or neutral. QQP: are two Quora questions semantically equivalent. QNLI: a version of SQuAD converted to binary classification, does the sentence contain the answer to the question](/img/papers/bert/p4-b1a.png)](/img/papers/bert/p4-b1a.png)
> [![Appendix B.1 continued: SST-2, binary sentiment of movie review sentences. CoLA, is an English sentence linguistically acceptable. STS-B, sentence pairs scored 1 to 5 for similarity. MRPC, are two news sentences semantically equivalent](/img/papers/bert/p4-b1b.png)](/img/papers/bert/p4-b1b.png)
> [![Appendix B.1 continued: RTE, binary entailment like MNLI with much less training data. WNLI, a small inference dataset with construction issues; every submitted system performed worse than the 65.1 majority-class baseline, so it is excluded to be fair to OpenAI GPT](/img/papers/bert/p4-b1c.png)](/img/papers/bert/p4-b1c.png)
>
> **Context:** Appendix B.1, two pages of short task descriptions, originally summarised in the GLUE paper.
>
> **What it says:** one paragraph per dataset. For WNLI, "every trained system that's been submitted to GLUE has performed worse than the 65.1 baseline accuracy of predicting the majority class. We therefore exclude this set to be fair to OpenAI GPT." (The sentence continues on the next page: for the GLUE submission they "always predicted the majority class".)
>
> **Why it matters:** you cannot read Table 1 without knowing what each column measures.

Here are all nine in plain words, with one made-up example each:

| Task | Input | Question the model answers | Example |
|---|---|---|---|
| **MNLI** | premise + hypothesis | entailment, contradiction or neutral? | "A man plays guitar." / "A person makes music." → entailment |
| **QQP** | two Quora questions | do they ask the same thing? | "How do I learn Python?" / "What is the best way to learn Python?" → same |
| **QNLI** | question + one sentence | does the sentence contain the answer? | "Where was Ada born?" / "Ada was born in London." → yes |
| **SST-2** | one movie-review sentence | positive or negative? | "A gorgeous, witty film." → positive |
| **CoLA** | one sentence | is it acceptable English? | "The more you read, the more you learn." → yes; "Read more you the." → no |
| **STS-B** | two sentences | how similar are they, from 1 to 5? | "A dog runs." / "A puppy is running." → about 4.5 |
| **MRPC** | two news sentences | do they mean the same? | → yes or no |
| **RTE** | premise + hypothesis | does the first imply the second? | like MNLI, two classes, small data |
| **WNLI** | two sentences with a pronoun | (left out by the paper) | |

> [!DEFINITION] Binary classification
> A task with exactly two possible answers, like positive or negative, yes or no.

> [!DEFINITION] Majority-class baseline
> The score you get by always answering with the most common label, without reading the input. On WNLI that already gives 65.1% accuracy, and in 2018 no submitted model beat it, so the paper leaves WNLI out of its average.

## The classification layer {§4.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.1 · page 5
> [![Section 4.1: to fine-tune on GLUE, use the final hidden vector C corresponding to the first input token, CLS, as the aggregate representation. The only new parameters are classification layer weights W in R K by H, where K is the number of labels. Compute a standard classification loss with C and W, log of softmax of C W transpose](/img/papers/bert/p4-glue-head.png)](/img/papers/bert/p4-glue-head.png)
>
> **Context:** the second paragraph of Section 4.1, the first concrete fine-tuning recipe of the paper.
>
> **What it says:** use "the final hidden vector $$C \in \mathbb{R}^H$$ corresponding to the first input token ([CLS]) as the aggregate representation". "The only new parameters introduced during fine-tuning are classification layer weights $$W \in \mathbb{R}^{K \times H}$$, where $$K$$ is the number of labels." The loss is "$$\log(\text{softmax}(CW^T))$$".
>
> **Why it matters:** this one line turns BERT into any classifier. For MNLI, $$K = 3$$; for SST-2, $$K = 2$$.

> [!DEFINITION] Logits
> The raw scores a model gives each possible label, before softmax turns them into probabilities. They can be any number, positive or negative.

> [!DEFINITION] Loss (cross-entropy)
> A number that measures how wrong the model is on an example; training lowers it. For classification, the loss is $$-\log$$ of the probability the model gave to the right label. If the model gives the right label probability 1, the loss is 0. If it gives it 0.01, the loss is about 4.6.

The paper writes the loss in one breath. Let us write it out, for one example whose correct label is $$y$$:

$$
\text{logits} = C\,W^\top \in \mathbb{R}^{K},
\qquad
P(\text{label } k) = \frac{e^{\text{logits}_k}}{\sum_{m=1}^{K} e^{\text{logits}_m}},
\qquad
\text{loss} = -\log P(\text{label } y)
$$

where:

- $$C$$ is the final vector of the `[CLS]` token, a row of $$H = 768$$ numbers (for BERT-base);
- $$W$$ is the new weight matrix with $$K$$ rows (one per label) and $$H$$ columns, and $$W^\top$$ is it flipped, so $$C\,W^\top$$ gives one number per label;
- $$\text{logits}_k$$ is the score of label $$k$$, the dot product of $$C$$ with row $$k$$ of $$W$$;
- the fraction is the softmax, which turns the $$K$$ scores into probabilities that add up to 1;
- $$y$$ is the correct label, and the loss is minus the log of its probability.

The paper's "$$\log(\text{softmax}(CW^T))$$" is the log-probability of the right label; training maximises it, which is the same as minimising the loss above. (Real implementations also add a bias of $$K$$ numbers to the logits; the paper leaves it out of the formula.)

<figure class="fig"><svg viewBox="0 0 760 280" role="img" aria-label="Sentence-level fine-tuning. The final vector of the [CLS] token goes through one new layer W (K rows, one per label) and a softmax. The probabilities on the right are an illustration."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Sentence-level tasks read one vector: the final vector of [CLS]</text><rect class="box-on" x="40.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="72.0" y="249.5" text-anchor="middle">[CLS]</text><rect class="box" x="112.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="144.0" y="249.5" text-anchor="middle">a</text><rect class="box" x="184.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="216.0" y="249.5" text-anchor="middle">man</text><rect class="box" x="256.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="288.0" y="249.5" text-anchor="middle">plays</text><rect class="box" x="328.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="360.0" y="249.5" text-anchor="middle">[SEP]</text><rect class="box" x="400.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="432.0" y="249.5" text-anchor="middle">a</text><rect class="box" x="472.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="504.0" y="249.5" text-anchor="middle">person</text><rect class="box" x="544.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="576.0" y="249.5" text-anchor="middle">...</text><rect class="box" x="616.0" y="230.0" width="64.0" height="30.0" rx="6"/><text class="t-tick" x="648.0" y="249.5" text-anchor="middle">[SEP]</text><rect class="box-1" x="30.0" y="140.0" width="650.0" height="56.0" rx="12"/><text class="t-big" x="355.0" y="166.0" text-anchor="middle">BERT (12 layers, all fine-tuned)</text><text class="t-tick" x="355.0" y="184.0" text-anchor="middle">every token looks at every other token</text><line class="edge" x1="72.0" y1="228.0" x2="72.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="144.0" y1="228.0" x2="144.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="216.0" y1="228.0" x2="216.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="288.0" y1="228.0" x2="288.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="360.0" y1="228.0" x2="360.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="432.0" y1="228.0" x2="432.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="504.0" y1="228.0" x2="504.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="576.0" y1="228.0" x2="576.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="648.0" y1="228.0" x2="648.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="72.0" y1="138.0" x2="72.0" y2="112.0" marker-end="url(#ah)"/><rect class="box-on" x="40.0" y="80.0" width="64.0" height="30.0" rx="6"/><text class="t-math" x="72.0" y="100.0" text-anchor="middle">C</text><text class="t-muted" x="112.0" y="92.0" text-anchor="start">768 numbers</text><text class="t-muted" x="112.0" y="106.0" text-anchor="start">(pooler: dense + tanh)</text><line class="edge" x1="106.0" y1="70.0" x2="300.0" y2="56.0" marker-end="url(#ah)"/><rect class="box-3" x="300.0" y="36.0" width="130.0" height="40.0" rx="8"/><text class="t-note" x="365.0" y="54.0" text-anchor="middle">W: K x 768</text><text class="t-muted" x="365.0" y="69.0" text-anchor="middle">the only new weights</text><line class="edge" x1="432.0" y1="56.0" x2="470.0" y2="56.0" marker-end="url(#ah)"/><rect class="box" x="470.0" y="36.0" width="100.0" height="40.0" rx="8"/><text class="t-note" x="520.0" y="61.0" text-anchor="middle">softmax</text><line class="edge" x1="572.0" y1="56.0" x2="600.0" y2="56.0" marker-end="url(#ah)"/><rect class="s3" x="604" y="34" width="54.6" height="11" rx="2"/><text class="t-tick" x="668.0" y="44.0" text-anchor="start">entailment</text><rect class="s3" x="604" y="51" width="4.2" height="11" rx="2"/><text class="t-tick" x="668.0" y="61.0" text-anchor="start">neutral</text><rect class="s3" x="604" y="68" width="1.2" height="11" rx="2"/><text class="t-tick" x="668.0" y="78.0" text-anchor="start">contradiction</text><text class="t-muted" x="740.0" y="92.0" text-anchor="end">example probabilities</text></svg><figcaption>The sentence-level head. All tokens go through BERT; only the final [CLS] vector C is read. One new matrix W turns it into K scores, and softmax turns the scores into probabilities. The probabilities drawn are an illustration.</figcaption></figure>

### Checking the formula in code

How many new weights is that? For a 3-label task, $$W$$ has $$3 \times 768 = 2{,}304$$ numbers, plus 3 for the bias: 2,307. I built the classifier by hand and compared it with the Hugging Face `BertForSequenceClassification` on the same weights:

```python
from transformers import AutoTokenizer, BertForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("bert-base-uncased")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3).eval()
enc = tok("A man is playing a guitar.", "A person is making music.", return_tensors="pt")
label = torch.tensor([0])

out = model(**enc, labels=label, output_hidden_states=True)
C = out.hidden_states[-1][:, 0]                       # final vector of [CLS], shape (1, 768)
pooled = torch.tanh(model.bert.pooler.dense(C))       # the "pooler" (see below)
W, b = model.classifier.weight, model.classifier.bias # W: (3, 768)
logits = pooled @ W.T + b                             # C W^T (+ bias)
loss = -torch.log_softmax(logits, -1)[0, label]       # -log softmax(C W^T)[y]
```

The real output (from [`bert_part4_heads.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part4_heads.py)):

```text
input tokens: ['[CLS]', 'a', 'man', 'is', 'playing', 'a', 'guitar', '.', '[SEP]', 'a', 'person', 'is', 'making', 'music', '.', '[SEP]']
C (final [CLS] vector): shape (1, 768);  W: shape (3, 768)  (K = 3 labels, H = 768)
logits from the library:  [-0.1763, -0.9611, 0.0659]
logits by hand (pooled C): [-0.1763, -0.9611, 0.0659]
loss from the library 1.004374  vs  -log softmax(C W^T)[label] by hand 1.004374
probabilities (untrained W, so they mean nothing yet): [0.366, 0.167, 0.467]
with the raw C instead of the pooled C the logits would be [-0.0016, -0.1106, -0.1406] (the library uses the pooled C)
new parameters for this task: 2,307 out of 109,484,547 (0.0021%)
```

The hand-written loss and the library's loss agree to all six printed decimals. The probabilities are meaningless here, because $$W$$ is still random: fine-tuning has not started.

> [!WARNING] One honest detail: the pooler
> The paper says the classifier reads $$C$$, the final `[CLS]` vector. The code Google released (and Hugging Face, which copies it) first passes $$C$$ through a **pooler**: one more dense layer with a tanh, which was trained during pre-training together with next sentence prediction. The classifier reads that pooled vector. The output above shows that the raw $$C$$ would give different logits. The idea is unchanged (one vector per sequence, one new matrix $$W$$), but if you implement the formula literally you will not match the released models.

> [!DEFINITION] tanh
> A smooth function that squashes any number into the range from -1 to 1. "Dense layer + tanh" means: multiply by a matrix, add a bias, then squash every number.

## Table 1: the GLUE results {§4.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.1, Table 1 · page 6
> [![Table 1: GLUE test results. Columns MNLI matched and mismatched, QQP, QNLI, SST-2, CoLA, STS-B, MRPC, RTE and Average, with training set sizes below each name. Rows: Pre-OpenAI SOTA 74.0 average, BiLSTM+ELMo+Attn 71.0, OpenAI GPT 75.1, BERT-base 79.6, BERT-large 82.1](/img/papers/bert/p4-table1.png)](/img/papers/bert/p4-table1.png)
>
> **Context:** the main results table of the paper, at the top of page 6.
>
> **What it says:** GLUE **test** scores, "scored by the evaluation server". The number under each task name is its number of training examples. The Average leaves out WNLI. "F1 scores are reported for QQP and MRPC, Spearman correlations are reported for STS-B, and accuracy scores are reported for the other tasks." BERT-base averages **79.6** and BERT-large **82.1**, against 75.1 for OpenAI GPT.
>
> **Why it matters:** BERT-base wins every column. BERT-large wins by more.

How to read it, column by column:

- **The small numbers under the names** are training-set sizes: MNLI has 392k examples, RTE only 2.5k. Keep these in mind: they explain where the biggest gains are.
- **MNLI-(m/mm)** has two numbers: accuracy on the "matched" test set (the same genres of text as the training data) and the "mismatched" set (different genres). BERT-large: 86.7/85.9.
- **QQP, MRPC**: F1 scores. **STS-B**: Spearman correlation. **The rest**: accuracy.
- **Average**: the plain average of the columns, which the caption says is "slightly different than the official GLUE score" because WNLI is left out.

> [!DEFINITION] F1 for classification
> A score that balances two kinds of mistakes. **Precision**: of the pairs the model called "same meaning", how many really were? **Recall**: of the pairs that really were, how many did it find? F1 is their harmonic mean: $$F_1 = 2PR/(P+R)$$. It is used when one label is much more common than the other, where plain accuracy can look good by always guessing the common label.

> [!DEFINITION] Spearman correlation
> A number from -1 to 1 that says how well two rankings agree. STS-B asks for a similarity score; Spearman compares the order of the model's scores with the order of the human scores. 1 means the same order exactly.

> [!NOTE] A small inaccuracy in the caption
> The caption says accuracy is reported for "the other tasks", which would include CoLA. The GLUE benchmark itself scores CoLA with the **Matthews correlation** (a balanced score from -1 to 1), as listed in Table 1 of the GLUE paper (Wang et al.). The CoLA column (52.1 and 60.5 for BERT) is most likely that Matthews correlation, as reported by the GLUE server, not an accuracy.

And the rows:

- **Pre-OpenAI SOTA**: the best published result on each task before OpenAI GPT, each from a different specialised system.
- **BiLSTM+ELMo+Attn**: the GLUE paper's own baseline, a recurrent network using ELMo features.
- **OpenAI GPT**: the left-to-right Transformer from Part 1, fine-tuned the same way.
- **BERT-base and BERT-large**: one model each, one task at a time ("single-model, single task").

> [!PAPER] Devlin et al. (2018), BERT · Section 4.1 · page 6
> [![Section 4.1: batch size 32, fine-tune for 3 epochs for all GLUE tasks; for each task the best learning rate among 5e-5, 4e-5, 3e-5 and 2e-5 on the Dev set. For BERT-large, fine-tuning was sometimes unstable on small datasets, so several random restarts were run and the best model on Dev selected](/img/papers/bert/p4-glue-setup.png)](/img/papers/bert/p4-glue-setup.png)
> [![Section 4.1 results: both BERT models outperform all systems on all tasks, with 4.5% and 7.0% average improvement over the prior state of the art. On the official GLUE leaderboard, BERT-large obtains 80.5, compared to OpenAI GPT with 72.8](/img/papers/bert/p4-glue-results.png)](/img/papers/bert/p4-glue-results.png)
>
> **Context:** the paragraphs under Table 1.
>
> **What it says:** the settings: batch size 32, 3 epochs, and the best learning rate "(among 5e-5, 4e-5, 3e-5, and 2e-5) on the Dev set". For BERT-large, "fine-tuning was sometimes unstable on small datasets, so we ran several random restarts and selected the best model on the Dev set". The results: "4.5% and 7.0% respective average accuracy improvement over the prior state of the art", and on the official leaderboard BERT-large scores **80.5** against GPT's **72.8**.
>
> **Why it matters:** the 80.5 is the "GLUE score 80.5%" from the abstract (Part 1), with its "7.7 point" improvement: $$80.5 - 72.8 = 7.7$$.

The two averages check out against the table: $$79.6 - 75.1 = 4.5$$ and $$82.1 - 75.1 = 7.0$$, where 75.1 is OpenAI GPT's average (the best earlier row). Note these are **points**, though the paper writes "%".

> [!DEFINITION] Random restart
> Running the same fine-tuning again with a different random seed: a different shuffle of the training data and a different random starting $$W$$. Small datasets can give very different results from run to run; restarting several times and keeping the best dev score hides the bad runs.

Two more sentences in this paragraph matter. First, "BERT-base and OpenAI GPT are nearly identical in terms of model architecture apart from the attention masking": the 4.5-point gap comes mostly from reading in both directions and from the pre-training tasks (Part 5 tests this). Second, footnote 9 says the GLUE test labels are hidden, and the authors "only made a single GLUE evaluation server submission for each of BERT-base and BERT-large". They did not tune on the test set.

> [!PAPER] Devlin et al. (2018), BERT · Section 4.1 · page 6
> [![BERT-large significantly outperforms BERT-base across all tasks, especially those with very little training data. The effect of model size is explored in Section 5.2](/img/papers/bert/p4-glue-size.png)](/img/papers/bert/p4-glue-size.png)
>
> **Context:** the last sentence of Section 4.1.
>
> **What it says:** BERT-large beats BERT-base everywhere, "especially those with very little training data".
>
> **Why it matters:** look at the small tasks in Table 1: CoLA (8.5k examples) goes from 52.1 to 60.5, RTE (2.5k) from 66.4 to 70.1. A bigger pre-trained model helps most when labels are scarce. Part 5 looks at model size in detail.

## Our own fine-tune: SST-2 and MRPC on a laptop GPU {§4.1}

Reading a results table is one thing. Let us run the recipe. I fine-tuned the released `bert-base-uncased` on two GLUE tasks with the paper's settings, on the GPU of an Apple M5 Pro laptop:

- **SST-2** (sentiment, 67,349 training sentences, 872 dev sentences);
- **MRPC** (paraphrase, 3,668 training pairs, 408 dev pairs).

The settings: batch size 32, 3 epochs, learning rate 2e-5 for SST-2 (one value from the paper's grid; I did not search), Adam with weight decay 0.01, the learning rate rising linearly over the first 10% of steps and then falling linearly to zero (as in Google's released `run_classifier.py`), dropout 0.1, one fixed seed. The classifier reads the pooled `[CLS]` vector, like the released code.

```python
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2).to("mps")
opt = torch.optim.AdamW(groups, lr=2e-5, weight_decay=0.01)   # no decay on biases and LayerNorm
steps = 3 * math.ceil(len(train) / 32)                         # 3 epochs, batch 32
warm = int(0.1 * steps)                                        # 10% warmup, then linear decay
sched = LambdaLR(opt, lambda s: s / warm if s < warm else (steps - s) / (steps - warm))

for ids, tt, am, y in batches(train):                          # every weight is trained
    loss = model(input_ids=ids, token_type_ids=tt, attention_mask=am, labels=y).loss
    loss.backward(); clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
```

> [!DEFINITION] Warmup
> Starting training with a very small learning rate and raising it step by step. Big early steps on a fresh random $$W$$ can damage the pre-trained weights below it; warmup avoids that.

> [!DEFINITION] Weight decay
> A small pull of every weight towards zero at each step, which keeps weights from growing without need. 0.01 is the value from pre-training (Part 3).

Here is the SST-2 run, the first and last lines of the real log ([`bert_part4_finetune.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part4_finetune.py)):

```text
task sst2: 67349 training examples, 872 dev examples, device mps
longest training input: 66 tokens; inputs cut at 128: 0
batch 32, 3 epochs = 6315 steps, learning rate 2e-05, warmup 631 steps, weight decay 0.01
before fine-tuning (random classifier layer): dev accuracy 0.4908
step   210/6315  epoch 1  train loss 0.5933  dev accuracy 0.8544  dev F1 0.8486  (82 s)
...
step  6300/6315  epoch 3  train loss 0.0790  dev accuracy 0.9255  dev F1 0.9279  (2888 s)
step  6315/6315  epoch 3  train loss 0.0487  dev accuracy 0.9255  dev F1 0.9279  (2894 s)
FINAL sst2 dev accuracy 0.9255 (807/872), dev F1 0.9279, training time 48.3 minutes
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Our SST-2 fine-tuning run: dev accuracy (blue, left axis) and training loss (orange, right axis, averaged between evaluations) over 6,315 steps."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="570" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.80</text><line class="grid" x1="64" y1="176.7" x2="570" y2="176.7"/><text class="t-tick" x="56.0" y="180.7" text-anchor="end">0.85</text><line class="grid" x1="64" y1="103.3" x2="570" y2="103.3"/><text class="t-tick" x="56.0" y="107.3" text-anchor="end">0.90</text><line class="grid" x1="64" y1="30.0" x2="570" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">0.95</text><text class="t-tick" x="578.0" y="254.0" text-anchor="start">0.0</text><text class="t-tick" x="578.0" y="180.7" text-anchor="start">0.2</text><text class="t-tick" x="578.0" y="107.3" text-anchor="start">0.4</text><text class="t-tick" x="578.0" y="34.0" text-anchor="start">0.6</text><line class="base-line" x1="232.7" y1="30" x2="232.7" y2="250"/><line class="base-line" x1="401.3" y1="30" x2="401.3" y2="250"/><text class="t-tick" x="148.3" y="270.0" text-anchor="middle">epoch 1</text><text class="t-tick" x="317.0" y="270.0" text-anchor="middle">epoch 2</text><text class="t-tick" x="485.7" y="270.0" text-anchor="middle">epoch 3</text><polyline class="l2" points="80.8,32.5 97.7,133.1 114.5,150.3 131.3,163.1 148.1,171.3 165.0,173.5 181.8,181.4 198.6,178.3 215.4,185.6 232.3,185.4 249.1,204.1 265.9,207.5 282.7,209.3 299.6,199.8 316.4,205.1 333.2,205.1 350.1,208.0 366.9,207.9 383.7,207.8 400.5,209.3 417.4,222.4 434.2,221.2 451.0,222.9 467.8,221.7 484.7,219.7 501.5,222.6 518.3,227.5 535.1,219.2 552.0,224.2 568.8,221.0 570.0,232.1"/><polyline class="l1" points="80.8,170.3 97.7,121.5 114.5,129.9 131.3,77.8 148.1,71.0 165.0,82.8 181.8,71.0 198.6,77.8 215.4,69.4 232.3,77.8 249.1,60.9 265.9,52.5 282.7,74.4 299.6,64.3 316.4,77.8 333.2,66.0 350.1,67.7 366.9,71.0 383.7,67.7 400.5,74.4 417.4,67.7 434.2,59.3 451.0,62.6 467.8,71.0 484.7,71.0 501.5,64.3 518.3,59.3 535.1,60.9 552.0,64.3 568.8,66.0 570.0,66.0"/><g class="mark"><title>step 210: dev accuracy 0.8544, train loss 0.5933</title><circle class="s1 ring" cx="80.8" cy="170.3" r="3.5"/></g><g class="mark"><title>step 420: dev accuracy 0.8876, train loss 0.3189</title><circle class="s1 ring" cx="97.7" cy="121.5" r="3.5"/></g><g class="mark"><title>step 630: dev accuracy 0.8819, train loss 0.2718</title><circle class="s1 ring" cx="114.5" cy="129.9" r="3.5"/></g><g class="mark"><title>step 840: dev accuracy 0.9174, train loss 0.2369</title><circle class="s1 ring" cx="131.3" cy="77.8" r="3.5"/></g><g class="mark"><title>step 1050: dev accuracy 0.9220, train loss 0.2145</title><circle class="s1 ring" cx="148.1" cy="71.0" r="3.5"/></g><g class="mark"><title>step 1260: dev accuracy 0.9140, train loss 0.2088</title><circle class="s1 ring" cx="165.0" cy="82.8" r="3.5"/></g><g class="mark"><title>step 1470: dev accuracy 0.9220, train loss 0.1871</title><circle class="s1 ring" cx="181.8" cy="71.0" r="3.5"/></g><g class="mark"><title>step 1680: dev accuracy 0.9174, train loss 0.1956</title><circle class="s1 ring" cx="198.6" cy="77.8" r="3.5"/></g><g class="mark"><title>step 1890: dev accuracy 0.9232, train loss 0.1755</title><circle class="s1 ring" cx="215.4" cy="69.4" r="3.5"/></g><g class="mark"><title>step 2100: dev accuracy 0.9174, train loss 0.1761</title><circle class="s1 ring" cx="232.3" cy="77.8" r="3.5"/></g><g class="mark"><title>step 2310: dev accuracy 0.9289, train loss 0.1251</title><circle class="s1 ring" cx="249.1" cy="60.9" r="3.5"/></g><g class="mark"><title>step 2520: dev accuracy 0.9346, train loss 0.1159</title><circle class="s1 ring" cx="265.9" cy="52.5" r="3.5"/></g><g class="mark"><title>step 2730: dev accuracy 0.9197, train loss 0.1110</title><circle class="s1 ring" cx="282.7" cy="74.4" r="3.5"/></g><g class="mark"><title>step 2940: dev accuracy 0.9266, train loss 0.1369</title><circle class="s1 ring" cx="299.6" cy="64.3" r="3.5"/></g><g class="mark"><title>step 3150: dev accuracy 0.9174, train loss 0.1226</title><circle class="s1 ring" cx="316.4" cy="77.8" r="3.5"/></g><g class="mark"><title>step 3360: dev accuracy 0.9255, train loss 0.1224</title><circle class="s1 ring" cx="333.2" cy="66.0" r="3.5"/></g><g class="mark"><title>step 3570: dev accuracy 0.9243, train loss 0.1145</title><circle class="s1 ring" cx="350.1" cy="67.7" r="3.5"/></g><g class="mark"><title>step 3780: dev accuracy 0.9220, train loss 0.1147</title><circle class="s1 ring" cx="366.9" cy="71.0" r="3.5"/></g><g class="mark"><title>step 3990: dev accuracy 0.9243, train loss 0.1150</title><circle class="s1 ring" cx="383.7" cy="67.7" r="3.5"/></g><g class="mark"><title>step 4200: dev accuracy 0.9197, train loss 0.1109</title><circle class="s1 ring" cx="400.5" cy="74.4" r="3.5"/></g><g class="mark"><title>step 4410: dev accuracy 0.9243, train loss 0.0752</title><circle class="s1 ring" cx="417.4" cy="67.7" r="3.5"/></g><g class="mark"><title>step 4620: dev accuracy 0.9300, train loss 0.0785</title><circle class="s1 ring" cx="434.2" cy="59.3" r="3.5"/></g><g class="mark"><title>step 4830: dev accuracy 0.9278, train loss 0.0738</title><circle class="s1 ring" cx="451.0" cy="62.6" r="3.5"/></g><g class="mark"><title>step 5040: dev accuracy 0.9220, train loss 0.0771</title><circle class="s1 ring" cx="467.8" cy="71.0" r="3.5"/></g><g class="mark"><title>step 5250: dev accuracy 0.9220, train loss 0.0826</title><circle class="s1 ring" cx="484.7" cy="71.0" r="3.5"/></g><g class="mark"><title>step 5460: dev accuracy 0.9266, train loss 0.0747</title><circle class="s1 ring" cx="501.5" cy="64.3" r="3.5"/></g><g class="mark"><title>step 5670: dev accuracy 0.9300, train loss 0.0614</title><circle class="s1 ring" cx="518.3" cy="59.3" r="3.5"/></g><g class="mark"><title>step 5880: dev accuracy 0.9289, train loss 0.0839</title><circle class="s1 ring" cx="535.1" cy="60.9" r="3.5"/></g><g class="mark"><title>step 6090: dev accuracy 0.9266, train loss 0.0703</title><circle class="s1 ring" cx="552.0" cy="64.3" r="3.5"/></g><g class="mark"><title>step 6300: dev accuracy 0.9255, train loss 0.0790</title><circle class="s1 ring" cx="568.8" cy="66.0" r="3.5"/></g><g class="mark"><title>step 6315: dev accuracy 0.9255, train loss 0.0487</title><circle class="s1 ring" cx="570.0" cy="66.0" r="3.5"/></g><text class="t-s1" x="610.0" y="70.0" text-anchor="start">dev accuracy 0.9255</text><text class="t-s2" x="610.0" y="236.1" text-anchor="start">train loss 0.049</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">dev accuracy</text><text class="t-tick" x="578.0" y="18.0" text-anchor="start">loss</text></svg><figcaption>Our SST-2 run. Dev accuracy (blue) climbs above 0.9 within the first few hundred steps; training loss (orange) keeps falling for all three epochs.</figcaption></figure>

**SST-2: 92.55% dev accuracy** (807 of 872 sentences right), after 6,315 steps. Before fine-tuning, with a random $$W$$, the same model scored 49.1%: a coin flip. The paper reports **92.7** for BERT-base on the SST-2 **dev** set (Table 5, in Part 5) and 93.5 on the hidden **test** set (Table 1). Our single run, with one learning rate and no search, lands at 92.55. (During training, dev accuracy touched 93.5% at step 2,520; I report the model at the end of the 3 epochs, which is what a fixed number of epochs means, not the best checkpoint: picking the best checkpoint by its dev score would flatter the dev score.)

Note how fast it learns: dev accuracy was already 85.4% after the first 210 steps, which is about 10% of one epoch. Almost all of the knowledge was already in the pre-trained weights; fine-tuning only has to connect it to the two labels.

The run took **48 minutes** of wall time on the laptop GPU, including 31 evaluations on the dev set, and other experiments were sharing the GPU for most of it, so treat the time as an upper bound. That matches the paper's "a few hours on a GPU" comfortably.

**MRPC** is the opposite kind of task: only 3,668 training pairs. Here I did what the paper did for GLUE and ran all four learning rates from its grid, one run each, keeping everything else fixed:

| Learning rate | MRPC dev accuracy | F1 | Time |
|---|---|---|---|
| 5e-5 | 87.0% (355/408) | 90.9 | 2.7 min |
| 4e-5 | 84.8% (346/408) | 89.5 | 2.7 min |
| 3e-5 | 83.3% (340/408) | 88.6 | 2.7 min |
| 2e-5 | 83.3% (340/408) | 88.5 | 2.8 min |

<figure class="fig"><svg viewBox="0 0 760 206" role="img" aria-label="Our MRPC grid: the same fine-tuning run with each of the paper's four learning rates. Like the paper, we would keep the best one on the dev set."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="40.0" text-anchor="start">MRPC dev accuracy, one run per learning rate</text><text class="t-tick t-strong" x="162.0" y="69.0" text-anchor="end">lr 5e-5</text><g class="mark"><title>5e-5: 87.0%</title><rect class="s1" x="170.0" y="53.0" width="330.6" height="22.0" rx="3"/></g><text class="t-val" x="506.6" y="69.0" text-anchor="start">87.0%</text><text class="t-tick" x="162.0" y="99.0" text-anchor="end">lr 4e-5</text><g class="mark"><title>4e-5: 84.8%</title><rect class="s1" x="170.0" y="83.0" width="322.3" height="22.0" rx="3"/></g><text class="t-val" x="498.3" y="99.0" text-anchor="start">84.8%</text><text class="t-tick" x="162.0" y="129.0" text-anchor="end">lr 3e-5</text><g class="mark"><title>3e-5: 83.3%</title><rect class="s1" x="170.0" y="113.0" width="316.7" height="22.0" rx="3"/></g><text class="t-val" x="492.7" y="129.0" text-anchor="start">83.3%</text><text class="t-tick" x="162.0" y="159.0" text-anchor="end">lr 2e-5</text><g class="mark"><title>2e-5: 83.3%</title><rect class="s1" x="170.0" y="143.0" width="316.7" height="22.0" rx="3"/></g><text class="t-val" x="492.7" y="159.0" text-anchor="start">83.3%</text><text class="t-muted" x="20.0" y="192.0" text-anchor="start">the paper reports 86.7 for BERT-base (Table 5, dev); bars start at 0</text></svg><figcaption>MRPC dev accuracy for each of the paper's four learning rates, one run each with the same seed.</figcaption></figure>

The best learning rate here is **5e-5**, with **87.0%** dev accuracy (355/408) and F1 90.9. The paper reports **86.7** dev accuracy for BERT-base on MRPC (Table 5). Three honest remarks:

- The same job is not even exactly repeatable. Before the grid, I had run the 2e-5 job once with the same seed and got **81.9%** (334/408), against 83.3% in the grid. GPU arithmetic on the Apple GPU is not bit-for-bit repeatable, and on 408 dev examples one changed answer moves accuracy by 0.25 points. This is the instability the paper mentions, and why it reports "average Dev Set accuracy from 5 random restarts" in its model-size study (Part 5).
- The spread between learning rates is large for such a small dataset, which is exactly what Appendix A.3 warns about ("large data sets ... were far less sensitive to hyperparameter choice than small data sets").
- MRPC is lopsided: 68% of its pairs are paraphrases (the GLUE paper notes this), so always answering "yes" already scores about 68% on dev. Before fine-tuning, our random head answered "no" every time and scored 31.6%, the exact mirror image. This is why GLUE reports F1 next to accuracy for MRPC.

## SQuAD v1.1: finding the answer in a paragraph {§4.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.2 · page 6
> [![Section 4.2: the Stanford Question Answering Dataset, SQuAD v1.1, is a collection of 100k crowd-sourced question and answer pairs. Given a question and a passage from](/img/papers/bert/p4-squad-intro.png)](/img/papers/bert/p4-squad-intro.png)
> [![Section 4.2 continued: predict the answer text span in the passage. The question uses the A embedding and the passage the B embedding. Only a start vector S and an end vector E are introduced. The probability of word i being the start is a softmax of S dot T i over the paragraph. The score of a span from i to j is S dot T i plus E dot T j, and the maximum scoring span with j at least i is the prediction. The training objective is the sum of the log-likelihoods of the correct start and end positions. 3 epochs, learning rate 5e-5, batch size 32](/img/papers/bert/p4-squad-se.png)](/img/papers/bert/p4-squad-se.png)
>
> **Context:** Section 4.2, the first token-level task: panel (c) of Figure 4.
>
> **What it says:** SQuAD v1.1 has "100k crowd-sourced question/answer pairs". Given a question and a Wikipedia passage that contains the answer, predict "the answer text span in the passage". The question goes into segment A, the passage into segment B. The only new weights are "a start vector $$S \in \mathbb{R}^H$$ and an end vector $$E \in \mathbb{R}^H$$". The start probability of word $$i$$ is $$P_i = e^{S \cdot T_i} / \sum_j e^{S \cdot T_j}$$, and the same for the end with $$E$$. A span from $$i$$ to $$j$$ scores $$S \cdot T_i + E \cdot T_j$$, and the best span with $$j \ge i$$ is the answer. Training maximises the log-likelihood of the correct start plus that of the correct end. Settings: 3 epochs, learning rate 5e-5, batch size 32.
>
> **Why it matters:** a whole question-answering system from two vectors of 768 numbers each, 1,536 new weights in total.

> [!DEFINITION] Span
> A continuous stretch of tokens inside a text, given by where it starts and where it ends. SQuAD answers are always spans of the passage, like "Bidirectional Encoder Representations from Transformers".

> [!DEFINITION] Crowd-sourced
> Written by many paid online workers rather than by the researchers. For SQuAD, workers read Wikipedia paragraphs and wrote questions whose answers are in the paragraph.

The start probability, written out:

$$
P^{\text{start}}_i = \frac{e^{S \cdot T_i}}{\sum_{j} e^{S \cdot T_j}}
$$

where:

- $$T_i$$ is the final vector of passage token $$i$$ (768 numbers);
- $$S$$ is the learned start vector (768 numbers), and $$S \cdot T_i$$ their dot product: one score per token, high when token $$i$$ "looks like the start of an answer to this question";
- the sum in the bottom runs over all tokens $$j$$ of the paragraph, so the probabilities over the paragraph add up to 1.

The end probability $$P^{\text{end}}_i$$ is the same with $$E$$ in place of $$S$$. To answer, pick the span that maximises

$$
\text{score}(i, j) = S \cdot T_i + E \cdot T_j \qquad \text{with } j \ge i
$$

where $$i$$ is the start token and $$j$$ the end token. The condition $$j \ge i$$ just says an answer cannot end before it starts. Adding the two dot products is the same as multiplying the two probabilities (the softmax bottoms are the same for every span), so this picks the most likely start-and-end pair.

The training loss for one example whose true answer starts at token $$s$$ and ends at token $$e$$:

$$
\text{loss} = -\log P^{\text{start}}_{s} - \log P^{\text{end}}_{e}
$$

That is the paper's "sum of the log-likelihoods of the correct start and end positions", with a minus sign so that lower is better.

### Running it

I did **not** fine-tune a SQuAD model myself. I used a public BERT-base checkpoint that someone else fine-tuned on SQuAD v1.1, [`csarron/bert-base-uncased-squad-v1`](https://huggingface.co/csarron/bert-base-uncased-squad-v1), and wrote the span search myself. First, a check that the model really is the paper's design: its output layer is a $$2 \times 768$$ matrix, row 0 is $$S$$ and row 1 is $$E$$. Computing $$S \cdot T_i$$ by hand from the hidden states reproduces the model's start scores:

```python
W = model.qa_outputs.weight          # (2, 768): row 0 = S, row 1 = E
out = model(**enc, output_hidden_states=True)
T = out.hidden_states[-1][0]         # T_i for every token, (tokens, 768)
start_by_hand = T @ W[0] + model.qa_outputs.bias[0]     # S . T_i

def best_span(start, end, passage_tokens, max_len=30):
    """max over i <= j of S.T_i + E.T_j, inside the passage, at most 30 tokens long"""
    best = None
    for i in passage_tokens:
        for j in passage_tokens:
            if i <= j < i + max_len and (best is None or start[i] + end[j] > best[0]):
                best = (start[i] + end[j], i, j)
    return best
```

(The real script only tries the 20 best starts and 20 best ends, like Google's released `run_squad.py`, which is much faster and finds the same answer in practice.) I used the paper's own abstract as the passage. The real output:

```text
SQuAD v1.1 model: the output layer is (2, 768) -> row 0 is the start vector S, row 1 the end vector E (H = 768)
  it also has a bias: start +0.0049, end +0.0048 (a constant added to every position, so it cancels in the softmax)
Q: What does BERT stand for?
   answer: "Bidirectional Encoder Representations from Transformers"  (tokens 21..30, span score S.T_i + E.T_j = 15.16)
   runner-up spans: "Bidirectional Encoder Representations from Transformers." 11.21; "Bidirectional Encoder Representations" 10.90
   most likely start tokens: bid 0.989, transformers 0.009, en 0.001
   check: |S.T_i by hand - model start logit| max = 1.0e-05; softmax with vs without bias differs by 3.6e-07
Q: What is the GLUE score of BERT?
   answer: "80.5%"  (tokens 91..94, span score S.T_i + E.T_j = 16.81)
   runner-up spans: "to 80.5%" 10.68; "80.5% and SQuAD v1.1 question answering Test F1 to 93.2" 10.14
   most likely start tokens: 80 0.996, to 0.002, pushing 0.001
   check: |S.T_i by hand - model start logit| max = 7.6e-06; softmax with vs without bias differs by 1.2e-08
Q: On how many tasks does BERT get new results?
   answer: "eleven"  (tokens 81..81, span score S.T_i + E.T_j = 11.59)
   runner-up spans: "eleven natural language processing tasks" 9.38; "eleven natural language processing" 6.92
   most likely start tokens: eleven 0.990, on 0.002, natural 0.002
   check: |S.T_i by hand - model start logit| max = 8.6e-06; softmax with vs without bias differs by 1.2e-07
Q: What does BERT learn from?
   answer: "Transformers"  (tokens 30..30, span score S.T_i + E.T_j = 9.87)
   runner-up spans: "Bidirectional Encoder Representations from Transformers" 7.65; "Transformers. BERT is designed to pre-train deep bidirectional representations from unlabeled text" 6.76
   most likely start tokens: transformers 0.810, bid 0.088, un 0.061
   check: |S.T_i by hand - model start logit| max = 6.7e-06; softmax with vs without bias differs by 7.5e-08
```

<figure class="fig"><svg viewBox="0 0 780 236" role="img" aria-label="Start and end probabilities over part of the passage, from the public SQuAD v1.1 checkpoint. The start of "Bidirectional" and the end of "transformers" get almost all the probability, so the best span is the expansion of BERT."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">Q: What does BERT stand for?</text><text class="t-tick" x="88.0" y="66.0" text-anchor="end">P(start)</text><g class="mark"><title>bert: 0.000</title><rect class="cell" x="97" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>,: 0.000</title><rect class="cell" x="141" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>which: 0.000</title><rect class="cell" x="185" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>stands: 0.000</title><rect class="cell" x="229" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>for: 0.000</title><rect class="cell" x="273" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>bid: 0.989</title><rect class="cell" x="317" y="41" width="42" height="38" rx="3" style="fill-opacity:0.989"/></g><text class="t-cell on" x="338.0" y="65.0" text-anchor="middle">0.99</text><g class="mark"><title>##ire: 0.000</title><rect class="cell" x="361" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##ction: 0.000</title><rect class="cell" x="405" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##al: 0.000</title><rect class="cell" x="449" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>en: 0.001</title><rect class="cell" x="493" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##code: 0.000</title><rect class="cell" x="537" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##r: 0.000</title><rect class="cell" x="581" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>representations: 0.000</title><rect class="cell" x="625" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>from: 0.000</title><rect class="cell" x="669" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>transformers: 0.009</title><rect class="cell" x="713" y="41" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><text class="t-tick" x="88.0" y="112.0" text-anchor="end">P(end)</text><g class="mark"><title>bert: 0.000</title><rect class="cell" x="97" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>,: 0.000</title><rect class="cell" x="141" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>which: 0.000</title><rect class="cell" x="185" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>stands: 0.000</title><rect class="cell" x="229" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>for: 0.000</title><rect class="cell" x="273" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>bid: 0.000</title><rect class="cell" x="317" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##ire: 0.000</title><rect class="cell" x="361" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##ction: 0.000</title><rect class="cell" x="405" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##al: 0.000</title><rect class="cell" x="449" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>en: 0.000</title><rect class="cell" x="493" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##code: 0.000</title><rect class="cell" x="537" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>##r: 0.001</title><rect class="cell" x="581" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>representations: 0.014</title><rect class="cell" x="625" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><text class="t-cell" x="646.0" y="111.0" text-anchor="middle">0.01</text><g class="mark"><title>from: 0.000</title><rect class="cell" x="669" y="87" width="42" height="38" rx="3" style="fill-opacity:0.050"/></g><g class="mark"><title>transformers: 0.966</title><rect class="cell" x="713" y="87" width="42" height="38" rx="3" style="fill-opacity:0.966"/></g><text class="t-cell on" x="734.0" y="111.0" text-anchor="middle">0.97</text><text class="t-tick" x="118.0" y="150.0" text-anchor="end" transform="rotate(-45 118.0 150)">bert</text><text class="t-tick" x="162.0" y="150.0" text-anchor="end" transform="rotate(-45 162.0 150)">,</text><text class="t-tick" x="206.0" y="150.0" text-anchor="end" transform="rotate(-45 206.0 150)">which</text><text class="t-tick" x="250.0" y="150.0" text-anchor="end" transform="rotate(-45 250.0 150)">stands</text><text class="t-tick" x="294.0" y="150.0" text-anchor="end" transform="rotate(-45 294.0 150)">for</text><text class="t-tick" x="338.0" y="150.0" text-anchor="end" transform="rotate(-45 338.0 150)">bid</text><text class="t-tick" x="382.0" y="150.0" text-anchor="end" transform="rotate(-45 382.0 150)">##ire</text><text class="t-tick" x="426.0" y="150.0" text-anchor="end" transform="rotate(-45 426.0 150)">##ction</text><text class="t-tick" x="470.0" y="150.0" text-anchor="end" transform="rotate(-45 470.0 150)">##al</text><text class="t-tick" x="514.0" y="150.0" text-anchor="end" transform="rotate(-45 514.0 150)">en</text><text class="t-tick" x="558.0" y="150.0" text-anchor="end" transform="rotate(-45 558.0 150)">##code</text><text class="t-tick" x="602.0" y="150.0" text-anchor="end" transform="rotate(-45 602.0 150)">##r</text><text class="t-tick" x="646.0" y="150.0" text-anchor="end" transform="rotate(-45 646.0 150)">representations</text><text class="t-tick" x="690.0" y="150.0" text-anchor="end" transform="rotate(-45 690.0 150)">from</text><text class="t-tick" x="734.0" y="150.0" text-anchor="end" transform="rotate(-45 734.0 150)">transformers</text></svg><figcaption>Start and end probabilities over part of the passage for "What does BERT stand for?". Almost all the start probability sits on "bid" (the first piece of "Bidirectional") and almost all the end probability on "transformers", so the best span is the expansion of BERT.</figcaption></figure>

Three things to notice:

- **The formula is exactly the model.** $$S \cdot T_i$$ computed by hand matches the model's start scores to about $$10^{-5}$$ (rounding noise of the GPU). The checkpoint also has a bias number added to every start score; adding the same number to every token does not change a softmax, and the output shows the probabilities differ by less than $$10^{-6}$$.
- **"bid" is a word piece.** The tokenizer cut "Bidirectional" into pieces (Part 2), and the answer starts at the first piece.
- **It can be wrong.** For "What does BERT learn from?", the model answered "Transformers". The right answer, "unlabeled text", is in the passage, but the phrase "Representations **from** Transformers" fooled it. Fine-tuned models match patterns; they do not reason like a reader.

### Scoring on the full SQuAD v1.1 dev set

One passage proves nothing about quality. So I ran the same checkpoint, with my own span search, on **all** questions of the SQuAD v1.1 dev set, using the official answer normalisation (lower case, no punctuation, no "a/an/the").

> [!DEFINITION] Exact match (EM) and F1 for SQuAD
> **EM** is 1 if the predicted answer is exactly one of the human answers (after normalisation), else 0. **F1** gives partial credit: it counts the words the prediction shares with a human answer, and combines precision and recall. Both are averaged over all questions and shown out of 100.

Long passages do not fit in 384 tokens, so each one is cut into overlapping windows with a stride of 128 tokens (the settings of the released code), and the best span over all windows wins.

```text
SQuAD v1.1 dev: 10570 questions (10753 windows of 384 tokens, stride 128): EM 80.79, F1 88.08  (9.3 min)
   a miss: Q "What was the theme of Super Bowl 50?" predicted "to determine the champion of the National Football League (NFL) for the 2015 season", gold ""golden anniversary""
   a miss: Q "What was the theme of Super Bowl 50?" predicted "to determine the champion of the National Football League (NFL) for the 2015 season", gold ""golden anniversary""
   a miss: Q "How many times have the Panthers been in the Super Bowl?" predicted "eight", gold "2"
```

On all **10,570** dev questions, this checkpoint with my decoding scores **EM 80.79** and **F1 88.08**. Table 2 (next) lists **80.8 EM and 88.5 F1** for the authors' own BERT-base on the same dev set. So a BERT-base fine-tuned by someone else, decoded by my 60 lines of span search, lands within half a point of the paper. (The checkpoint's model page reports EM 80.91 and F1 88.23 with its own evaluation code; the small gap is down to decoding details such as how many candidate spans are tried.)

The misses are instructive. "What was the theme of Super Bowl 50?" appears twice because the dev set really contains two copies of that question (two separate entries with slightly different gold answers); both times the model picked a long span about the purpose of the game instead of "golden anniversary". And for "How many times have the Panthers been in the Super Bowl?" it answered "eight" where the answer is "2": a number from the right paragraph, attached to the wrong fact.

### Table 2 {§4.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.2, Table 2 · page 7
> [![Table 2: SQuAD 1.1 results. Top leaderboard systems on Dec 10th 2018: Human test EM 82.3 F1 91.2, #1 ensemble nlnet 86.0 and 91.7. Published: BiDAF+ELMo, R.M. Reader. Ours: BERT-base single dev EM 80.8 F1 88.5; BERT-large single 84.1 and 90.9; BERT-large ensemble 85.8 and 91.8; BERT-large single plus TriviaQA dev 84.2 and 91.1, test 85.1 and 91.8; BERT-large ensemble plus TriviaQA dev 86.2 and 92.2, test 87.4 and 93.2. The ensemble is 7 systems with different pre-training checkpoints and fine-tuning seeds](/img/papers/bert/p4-table2.png)](/img/papers/bert/p4-table2.png)
>
> **Context:** the SQuAD v1.1 results table.
>
> **What it says:** dev and test scores, EM and F1, for the top leaderboard systems on 10 December 2018, the best published systems, and five BERT variants. The best BERT (an ensemble of 7, with TriviaQA) reaches **93.2 test F1**. BERT-base alone reaches 88.5 dev F1.
>
> **Why it matters:** 93.2 is the abstract's headline SQuAD number, and it is above the human F1 of 91.2 in the same table.

> [!DEFINITION] Ensemble
> Several models trained separately whose answers are combined. Here, "7x systems which use different pre-training checkpoints and fine-tuning seeds". Ensembles are usually a point or two better than any single member, and cost 7 times as much to run.

> [!PAPER] Devlin et al. (2018), BERT · Section 4.2 · pages 6 and 7
> [![Section 4.2: the top leaderboard systems may use any public data, so BERT first fine-tunes on TriviaQA before SQuAD. The best system outperforms the top leaderboard system by +1.5 F1 as an ensemble and +1.3 F1 as a single system; the single BERT model beats the top ensemble in F1](/img/papers/bert/p4-squad-trivia.png)](/img/papers/bert/p4-squad-trivia.png)
> [![Without TriviaQA fine-tuning data, we only lose 0.1-0.4 F1, still outperforming all existing systems by a wide margin](/img/papers/bert/p4-squad-trivia2.png)](/img/papers/bert/p4-squad-trivia2.png)
>
> **Context:** the paragraph under Table 2, which explains the "+TriviaQA" rows.
>
> **What it says:** leaderboard systems "are allowed to use any public data", so BERT is first fine-tuned on **TriviaQA**, another question-answering dataset, and then on SQuAD. The result beats the top leaderboard system "by +1.5 F1 in ensembling and +1.3 F1 as a single system". Without TriviaQA, "we only lose 0.1-0.4 F1".
>
> **Why it matters:** the extra data is a small boost, not the reason for the win. BERT-large alone, without TriviaQA, already reaches 90.9 dev F1.

The arithmetic: the ensemble's 93.2 test F1 minus the top leaderboard ensemble's 91.7 is the "+1.5". And the single model with TriviaQA (91.8 test F1) beats that top *ensemble* (91.7), which is the sentence "our single BERT model outperforms the top ensemble system in terms of F1 score".

## SQuAD v2.0: when there is no answer {§4.3}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.3 · page 7
> [![Section 4.3: SQuAD 2.0 allows that no short answer exists in the paragraph. Questions without an answer are treated as having an answer span that starts and ends at the CLS token. For prediction, compare the score of the no-answer span s null equal to S dot C plus E dot C to the score of the best non-null span](/img/papers/bert/p4-squad2.png)](/img/papers/bert/p4-squad2.png)
> [![Section 4.3 continued: s hat i j equals the max over j at least i of S dot T i plus E dot T j. Predict a non-null answer when s hat is greater than s null plus tau, where the threshold tau is selected on the dev set to maximize F1. No TriviaQA. 2 epochs, learning rate 5e-5, batch size 48. A +5.1 F1 improvement over the previous best system](/img/papers/bert/p4-squad2b.png)](/img/papers/bert/p4-squad2b.png)
>
> **Context:** Section 4.3. The paragraph runs from the bottom of the left column to the top of the right column.
>
> **What it says:** SQuAD 2.0 adds questions whose answer is **not** in the paragraph, "making the problem more realistic". BERT's fix is "simple": an unanswerable question is trained as if its answer were the span that starts and ends at `[CLS]`. At prediction time, compare the no-answer score $$s_{\text{null}} = S \cdot C + E \cdot C$$ with the best real span score $$\hat{s}_{i,j} = \max_{j \ge i} S \cdot T_i + E \cdot T_j$$, and answer only when $$\hat{s}_{i,j} > s_{\text{null}} + \tau$$, with the threshold $$\tau$$ "selected on the dev set to maximize F1". Settings: 2 epochs, learning rate 5e-5, batch size 48, no TriviaQA. Result: "+5.1 F1" over the previous best system.
>
> **Why it matters:** not a single new weight. The same $$S$$ and $$E$$ learn to point at `[CLS]` when there is nothing to find.

The decision rule, written out:

$$
\text{answer with the best span if } \;\hat{s}_{i,j} > s_{\text{null}} + \tau, \quad \text{otherwise say "no answer"}
$$

where:

- $$s_{\text{null}} = S \cdot C + E \cdot C$$ is the score of the "span" that starts and ends at `[CLS]` (recall that $$C$$ is the final vector of `[CLS]`);
- $$\hat{s}_{i,j}$$ is the score of the best real span inside the passage, with $$j \ge i$$;
- $$\tau$$ (the Greek letter tau) is a **threshold**: a number you choose. A larger $$\tau$$ makes the model more careful (it answers less often).

> [!DEFINITION] Threshold
> A cut-off value. The model answers only if the best span beats the no-answer option by more than $$\tau$$. The paper tries values of $$\tau$$ on the dev set and keeps the one with the best F1.

Again I used a public checkpoint, [`deepset/bert-base-uncased-squad2`](https://huggingface.co/deepset/bert-base-uncased-squad2) (BERT-base fine-tuned on SQuAD 2.0 by its authors, not by me), and computed both scores myself for two answerable and two unanswerable questions about the same abstract:

```python
s_null = start[0] + end[0]                 # S.C + E.C: [CLS] is token 0
s_hat, i, j = best_span(start, end, passage_tokens)
answer = passage[i..j] if s_hat > s_null + tau else "no answer"
```

```text
SQuAD 2.0 Q: What does BERT stand for?
   s_null = S.C + E.C = 10.18;  best span "Bidirectional Encoder Representations from Transformers" scores 23.25;  difference +13.07
SQuAD 2.0 Q: What is the GLUE score of BERT?
   s_null = S.C + E.C = 11.64;  best span "80.5%" scores 22.88;  difference +11.24
SQuAD 2.0 Q: Who won the football world cup in 2018?
   s_null = S.C + E.C = 21.13;  best span "BERT" scores 1.22;  difference -19.92
SQuAD 2.0 Q: How many layers does BERT have?
   s_null = S.C + E.C = 13.19;  best span "all" scores 16.16;  difference +2.97
```

<figure class="fig"><svg viewBox="0 0 760 236" role="img" aria-label="For each question, the best span score minus the no-answer score. Positive: the model answers with a span. Negative: the no-answer option at [CLS] wins, and the model says there is no answer. Here the threshold tau is 0."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">SQuAD 2.0 model: best span score minus the no-answer score</text><line class="base-line" x1="540.0" y1="36.0" x2="540.0" y2="212.0"/><text class="t-tick" x="20.0" y="59.0" text-anchor="start">What does BERT stand for?</text><g class="mark"><title>What does BERT stand for?: +13.07</title><rect class="s3" x="540.0" y="44" width="117.6" height="22" rx="3"/></g><text class="t-val" x="663.6" y="59.0" text-anchor="start">+13.07</text><text class="t-tick" x="20.0" y="103.0" text-anchor="start">What is the GLUE score of BERT?</text><g class="mark"><title>What is the GLUE score of BERT?: +11.24</title><rect class="s3" x="540.0" y="88" width="101.2" height="22" rx="3"/></g><text class="t-val" x="647.2" y="103.0" text-anchor="start">+11.24</text><text class="t-tick" x="20.0" y="147.0" text-anchor="start">Who won the football world cup in 2018?</text><g class="mark"><title>Who won the football world cup in 2018?: -19.92</title><rect class="s2" x="360.8" y="132" width="179.2" height="22" rx="3"/></g><text class="t-val" x="354.8" y="147.0" text-anchor="end">-19.92</text><text class="t-tick" x="20.0" y="191.0" text-anchor="start">How many layers does BERT have?</text><g class="mark"><title>How many layers does BERT have?: +2.97</title><rect class="s3" x="540.0" y="176" width="26.7" height="22" rx="3"/></g><text class="t-val" x="572.7" y="191.0" text-anchor="start">+2.97</text><text class="t-s3" x="550.0" y="222.0" text-anchor="start">answer: a span wins</text><text class="t-s2" x="530.0" y="222.0" text-anchor="end">abstain: [CLS] wins</text></svg><figcaption>Best span score minus the no-answer score for four questions. Positive bars: a span wins and the model answers. Negative bars: the [CLS] option wins and the model says there is no answer. This uses a threshold of 0.</figcaption></figure>

Read the differences:

- For the two answerable questions, the best span beats the no-answer score easily (+13.07 and +11.24).
- For the football question, nothing in the passage fits: the best span ("BERT") scores only 1.22, while `[CLS]` scores 21.13. The model says "no answer", which is right.
- The last question is the interesting one. "How many layers does BERT have?" sounds like it should be answerable, but the abstract never says. With $$\tau = 0$$ the model answers "**all**" (from "in all layers"), because that span beats $$s_{\text{null}}$$ by +2.97. That is a confident wrong answer, and exactly the kind of mistake the threshold $$\tau$$ exists to reduce: with $$\tau$$ above 2.97, the model would abstain.

### Choosing τ on the dev set

The paper picks $$\tau$$ "on the dev set to maximize F1". I did exactly that on the full SQuAD 2.0 dev set: for every question, store $$\hat{s}_{i,j} - s_{\text{null}}$$, then try thresholds from -6 to +6 in steps of 0.25.

```text
SQuAD v2.0 dev: 11873 questions, 5945 have no answer
   tau = 0:    F1 78.47, EM 75.47, answers 56.7% of questions
   best tau = +4.00: F1 78.98, EM 76.22, answers 52.1% of questions  (8.2 min)
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="SQuAD 2.0 dev F1 of the public checkpoint as the threshold tau changes. The paper picks tau on the dev set to maximise F1; this is that search."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="590" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">75</text><line class="grid" x1="64" y1="206.0" x2="590" y2="206.0"/><text class="t-tick" x="56.0" y="210.0" text-anchor="end">76</text><line class="grid" x1="64" y1="162.0" x2="590" y2="162.0"/><text class="t-tick" x="56.0" y="166.0" text-anchor="end">77</text><line class="grid" x1="64" y1="118.0" x2="590" y2="118.0"/><text class="t-tick" x="56.0" y="122.0" text-anchor="end">78</text><line class="grid" x1="64" y1="74.0" x2="590" y2="74.0"/><text class="t-tick" x="56.0" y="78.0" text-anchor="end">79</text><line class="grid" x1="64" y1="30.0" x2="590" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">80</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">-6</text><text class="t-tick" x="151.7" y="270.0" text-anchor="middle">-4</text><text class="t-tick" x="239.3" y="270.0" text-anchor="middle">-2</text><text class="t-tick" x="327.0" y="270.0" text-anchor="middle">0</text><text class="t-tick" x="414.7" y="270.0" text-anchor="middle">+2</text><text class="t-tick" x="502.3" y="270.0" text-anchor="middle">+4</text><text class="t-tick" x="590.0" y="270.0" text-anchor="middle">+6</text><polyline class="l1" points="64.0,189.4 75.0,185.0 85.9,179.6 96.9,174.1 107.8,167.3 118.8,161.4 129.8,155.8 140.7,148.5 151.7,143.4 162.6,143.8 173.6,142.0 184.5,135.7 195.5,132.1 206.5,129.8 217.4,126.9 228.4,125.2 239.3,123.2 250.3,118.2 261.2,115.7 272.2,110.7 283.2,108.7 294.1,105.2 305.1,103.8 316.0,102.2 327.0,97.5 338.0,98.5 348.9,98.3 359.9,98.2 370.8,94.3 381.8,94.8 392.8,91.4 403.7,90.1 414.7,87.6 425.6,89.1 436.6,87.4 447.5,86.1 458.5,81.2 469.5,79.0 480.4,77.5 491.4,74.9 502.3,74.8 513.3,80.1 524.2,80.4 535.2,81.1 546.2,83.1 557.1,85.2 568.1,85.6 579.0,88.0 590.0,94.5"/><circle class="s1 ring" cx="502.3" cy="74.8" r="6"/><text class="t-note" x="512.3" y="64.8" text-anchor="start">best τ +4.00: F1 78.98</text><circle class="s2 ring" cx="327.0" cy="97.5" r="5"/><text class="t-tick" x="337.0" y="115.5" text-anchor="start">τ = 0: F1 78.47</text><text class="t-tick" x="327.0" y="290.0" text-anchor="middle">threshold τ (answer only if the best span beats the no-answer score by more than τ)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">SQuAD 2.0 dev F1</text></svg><figcaption>SQuAD 2.0 dev F1 as the threshold changes. Too low and the model answers questions that have no answer; too high and it refuses questions it could answer. The best tau sits in between.</figcaption></figure>

Of the 11,873 dev questions, 5,945 have no answer. With $$\tau = 0$$, the checkpoint gets F1 78.47. The best threshold on this dev set is $$\tau = +4.00$$, which gives **F1 78.98** (EM 76.22) and makes the model answer 52.1% of the questions. The paper only reports BERT-large on SQuAD 2.0 (81.9 dev F1 in Table 3), so there is no BERT-base number to compare with directly; the checkpoint's model page reports 78.62 F1 with its own evaluation code.

One caution, which applies to the paper too: choosing $$\tau$$ on the dev set and then reporting the dev score flatters the dev score a little. That is why the paper's headline number is the **test** score, computed by the SQuAD organisers on hidden answers with the $$\tau$$ chosen on dev.

### Table 3 {§4.3}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.3, Table 3 · page 7
> [![Table 3: SQuAD 2.0 results, excluding systems that use BERT. Human dev EM 86.3 F1 89.0, test 86.9 and 89.5. Number 1 single MIR-MRC test 74.8 and 78.0. Number 2 nlnet 74.2 and 77.1. Published unet ensemble 71.4 and 74.9, SLQA+ 71.4 and 74.4. BERT-large single dev 78.7 and 81.9, test 80.0 and 83.1](/img/papers/bert/p4-table3.png)](/img/papers/bert/p4-table3.png)
>
> **Context:** the SQuAD 2.0 results table.
>
> **What it says:** one BERT row: BERT-large, a single model, **80.0 EM and 83.1 F1** on the test set. The best earlier single system on the leaderboard had 78.0 test F1.
>
> **Why it matters:** $$83.1 - 78.0 = 5.1$$: the "+5.1 F1" in the text and in the abstract. Humans are still ahead (89.5 F1).

## SWAG: choosing the best ending {§4.4}

> [!PAPER] Devlin et al. (2018), BERT · Section 4.4 · page 7
> [![Section 4.4: SWAG contains 113k sentence-pair completion examples for grounded commonsense inference; choose the most plausible continuation among four choices. Construct four input sequences, each the given sentence plus one possible continuation. The only new parameter is a vector whose dot product with C gives a score for each choice, normalized with softmax. 3 epochs, learning rate 2e-5, batch size 16. BERT-large outperforms ESIM+ELMo by +27.1% and OpenAI GPT by 8.3%](/img/papers/bert/p4-swag.png)](/img/papers/bert/p4-swag.png)
>
> **Context:** Section 4.4, the last task of Section 4.
>
> **What it says:** SWAG has "113k sentence-pair completion examples that evaluate grounded commonsense inference". Given a sentence, "choose the most plausible continuation among four choices". BERT reads four sequences, each "the given sentence (sentence A) and a possible continuation (sentence B)". The only new parameter is "a vector whose dot product with the [CLS] token representation $$C$$ denotes a score for each choice", followed by a softmax over the four. Settings: 3 epochs, learning rate 2e-5, batch size 16.
>
> **Why it matters:** a multiple-choice task, solved with one vector. It shows how far "swap the inputs and outputs" stretches.

> [!DEFINITION] Commonsense inference
> Using everyday knowledge about the world to guess what happens next. "She opened the fridge and..." is much more likely to continue with "took out the milk" than "flew to the moon". "Grounded" means the questions are about concrete, everyday situations.

The scoring, written out for the four choices $$k = 1, \dots, 4$$:

$$
s_k = C_k \cdot w, \qquad P(\text{choice } k) = \frac{e^{s_k}}{\sum_{m=1}^{4} e^{s_m}}
$$

where $$C_k$$ is the `[CLS]` vector when BERT reads "sentence + ending $$k$$", $$w$$ is the one new vector (768 numbers), and the softmax runs across the four choices. Unlike GLUE, the softmax is not over labels of one input, but over four separate inputs.

<figure class="fig"><svg viewBox="0 0 760 272" role="img" aria-label="SWAG fine-tuning. Each of the four endings is paired with the sentence and read by BERT separately. The [CLS] vector of each pair is multiplied by one learned vector w to give a score, and a softmax over the four scores picks the ending."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">SWAG: one sequence per choice, one score per sequence, softmax across the four</text><rect class="box" x="20.0" y="46.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="43.0" y="66.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="70.0" y="46.0" width="190.0" height="32.0" rx="6"/><text class="t-tick" x="165.0" y="66.5" text-anchor="middle">She opened the fridge and</text><rect class="box" x="264.0" y="46.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="287.0" y="66.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="314.0" y="46.0" width="180.0" height="32.0" rx="6"/><text class="t-tick" x="404.0" y="66.5" text-anchor="middle">took out a carton of milk.</text><line class="edge" x1="498.0" y1="62.0" x2="528.0" y2="62.0" marker-end="url(#ah)"/><rect class="box-1" x="530.0" y="48.0" width="64.0" height="28.0" rx="6"/><text class="t-note" x="562.0" y="67.0" text-anchor="middle">BERT</text><line class="edge" x1="596.0" y1="62.0" x2="616.0" y2="62.0" marker-end="url(#ah)"/><text class="t-tick" x="620.0" y="67.0" text-anchor="start">C1 . w = s1</text><rect class="box" x="20.0" y="96.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="43.0" y="116.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="70.0" y="96.0" width="190.0" height="32.0" rx="6"/><text class="t-tick" x="165.0" y="116.5" text-anchor="middle">She opened the fridge and</text><rect class="box" x="264.0" y="96.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="287.0" y="116.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="314.0" y="96.0" width="180.0" height="32.0" rx="6"/><text class="t-tick" x="404.0" y="116.5" text-anchor="middle">flew to the moon.</text><line class="edge" x1="498.0" y1="112.0" x2="528.0" y2="112.0" marker-end="url(#ah)"/><rect class="box-1" x="530.0" y="98.0" width="64.0" height="28.0" rx="6"/><text class="t-note" x="562.0" y="117.0" text-anchor="middle">BERT</text><line class="edge" x1="596.0" y1="112.0" x2="616.0" y2="112.0" marker-end="url(#ah)"/><text class="t-tick" x="620.0" y="117.0" text-anchor="start">C2 . w = s2</text><rect class="box" x="20.0" y="146.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="43.0" y="166.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="70.0" y="146.0" width="190.0" height="32.0" rx="6"/><text class="t-tick" x="165.0" y="166.5" text-anchor="middle">She opened the fridge and</text><rect class="box" x="264.0" y="146.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="287.0" y="166.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="314.0" y="146.0" width="180.0" height="32.0" rx="6"/><text class="t-tick" x="404.0" y="166.5" text-anchor="middle">the fridge sang a song.</text><line class="edge" x1="498.0" y1="162.0" x2="528.0" y2="162.0" marker-end="url(#ah)"/><rect class="box-1" x="530.0" y="148.0" width="64.0" height="28.0" rx="6"/><text class="t-note" x="562.0" y="167.0" text-anchor="middle">BERT</text><line class="edge" x1="596.0" y1="162.0" x2="616.0" y2="162.0" marker-end="url(#ah)"/><text class="t-tick" x="620.0" y="167.0" text-anchor="start">C3 . w = s3</text><rect class="box" x="20.0" y="196.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="43.0" y="216.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="70.0" y="196.0" width="190.0" height="32.0" rx="6"/><text class="t-tick" x="165.0" y="216.5" text-anchor="middle">She opened the fridge and</text><rect class="box" x="264.0" y="196.0" width="46.0" height="32.0" rx="6"/><text class="t-muted" x="287.0" y="216.5" text-anchor="middle">[SEP]</text><rect class="box-2" x="314.0" y="196.0" width="180.0" height="32.0" rx="6"/><text class="t-tick" x="404.0" y="216.5" text-anchor="middle">painted the ocean blue.</text><line class="edge" x1="498.0" y1="212.0" x2="528.0" y2="212.0" marker-end="url(#ah)"/><rect class="box-1" x="530.0" y="198.0" width="64.0" height="28.0" rx="6"/><text class="t-note" x="562.0" y="217.0" text-anchor="middle">BERT</text><line class="edge" x1="596.0" y1="212.0" x2="616.0" y2="212.0" marker-end="url(#ah)"/><text class="t-tick" x="620.0" y="217.0" text-anchor="start">C4 . w = s4</text><line class="edge" x1="708.0" y1="60.0" x2="708.0" y2="216.0"/><line class="edge" x1="708.0" y1="138.0" x2="724.0" y2="138.0" marker-end="url(#ah)"/><text class="t-note" x="728.0" y="134.0" text-anchor="start">soft-</text><text class="t-note" x="728.0" y="150.0" text-anchor="start">max</text><text class="t-tick" x="20.0" y="258.0" text-anchor="start">The same BERT runs four times. The only new weights: one vector w (768 numbers) and one bias.</text></svg><figcaption>SWAG. Each ending is paired with the sentence and read by BERT on its own. One learned vector w turns each [CLS] vector into a score, and a softmax across the four scores picks the ending.</figcaption></figure>

The head in code, checked against `BertForMultipleChoice` (untrained, so the probabilities mean nothing yet; the point is the shapes and the arithmetic):

```python
e = tok([ctx] * 4, endings, return_tensors="pt", padding=True)
e = {k: v.unsqueeze(0) for k, v in e.items()}                # (1 example, 4 choices, tokens)
C = model.bert(**{k: v.view(4, -1) for k, v in e.items()}).pooler_output   # (4, 768)
scores = C @ model.classifier.weight.T + model.classifier.bias           # (4, 1): one score each
probs = torch.softmax(scores.view(1, 4), -1)                 # softmax across the 4 choices
```

```text
SWAG input: (1, 4, 16) = (batch, 4 choices, tokens); the vector w: (1, 768)
scores C.w by hand: [-0.3572, 0.141, -0.337, -0.0369];  library: [-0.3572, 0.141, -0.337, -0.0369]
softmax over the 4 choices (untrained w): [0.198, 0.326, 0.202, 0.273]
```

> [!PAPER] Devlin et al. (2018), BERT · Section 4.4, Table 4 · page 7
> [![Table 4: SWAG dev and test accuracies. ESIM+GloVe 51.9 and 52.7, ESIM+ELMo 59.1 and 59.2, OpenAI GPT test 78.0, BERT-base dev 81.6, BERT-large dev 86.6 and test 86.3. Human expert 85.0, human with 5 annotations 88.0, measured with 100 samples](/img/papers/bert/p4-table4.png)](/img/papers/bert/p4-table4.png)
>
> **Context:** the SWAG results.
>
> **What it says:** BERT-large reaches **86.3%** test accuracy. The SWAG authors' own baseline (ESIM+ELMo) has 59.2, OpenAI GPT 78.0. A human expert, measured on 100 samples, scored 85.0.
>
> **Why it matters:** $$86.3 - 59.2 = 27.1$$ and $$86.3 - 78.0 = 8.3$$: the "+27.1%" and "8.3%" of the text (points again). On this test, a single model matched an expert human.

> [!DEFINITION] ESIM and GloVe
> **ESIM** is a 2017 network designed specially for sentence-pair inference. **GloVe** is a classic set of fixed word vectors from 2014. "ESIM+ELMo" is ESIM with ELMo's contextual vectors as input features: exactly the feature-based approach from Part 1.

A careful reading note: SWAG was released in 2018 and built to be hard for the models of its day (its authors filtered out endings that their own models found easy, a method they call Adversarial Filtering). Within months, a fine-tuned BERT came close to human scores on it. This pattern (a benchmark is published, a bigger pre-trained model nearly solves it) repeated many times after BERT.

> [!TAKEAWAYS] Key takeaways
> - **Fine-tuning = swap the inputs and outputs.** Every task fills the same `[CLS] A [SEP] B [SEP]` format and reads either the `[CLS]` vector (sentence-level) or every token's vector (token-level). Self-attention over the pair gives cross attention in both directions for free.
> - **Tiny new layers.** GLUE adds $$W \in \mathbb{R}^{K \times H}$$ (2,307 numbers for 3 labels, about 0.002% of BERT-base); SQuAD adds two vectors $$S$$ and $$E$$; SWAG adds one vector. Everything else starts pre-trained and all of it is trained.
> - **Cheap.** Batch 16 or 32, learning rate 2e-5 to 5e-5, 2 to 4 epochs. The paper says every result replicates in at most an hour on a Cloud TPU. Our own BERT-base runs on a laptop GPU: **92.55%** SST-2 dev accuracy in 48 minutes (paper: 92.7), and **87.0%** on MRPC dev with the best of the four learning rates (paper: 86.7).
> - **GLUE:** BERT-base averages 79.6 and BERT-large 82.1 on Table 1, against 75.1 for GPT (+4.5 and +7.0 points); 80.5 on the official leaderboard against 72.8.
> - **SQuAD v1.1:** answer = the span maximising $$S \cdot T_i + E \cdot T_j$$ with $$j \ge i$$. Best BERT: 93.2 test F1. A public BERT-base checkpoint, decoded with our own span search, scored **EM 80.79 / F1 88.08** on the SQuAD v1.1 dev set (paper's BERT-base: 80.8 / 88.5).
> - **SQuAD v2.0:** no new weights; "no answer" is the span at `[CLS]`, and a dev-tuned threshold $$\tau$$ decides. BERT-large: 83.1 test F1, +5.1 over the previous best.
> - **SWAG:** four sequences, one score each, softmax across them: 86.3 test accuracy.
> - **Honest details:** the released models feed a pooled `[CLS]` vector (dense + tanh) to the classifier, not the raw $$C$$ of the formula; and GLUE scores CoLA with the Matthews correlation, not accuracy.

**Next, in Part 5:** the ablation studies. Is next sentence prediction really needed? How much of the gain is reading both ways? Do bigger models keep helping? And can BERT be used without fine-tuning at all, as a feature extractor?

<details>
<summary>Run it yourself</summary>

Three scripts produce every number in this part, in [`code/papers/bert/`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/):

```bash
pip install torch transformers datasets
python bert_part4_heads.py            # the GLUE and SWAG heads checked by hand (CPU, seconds)
python bert_part4_finetune.py sst2    # fine-tune BERT-base on SST-2 (GPU recommended)
python bert_part4_finetune.py mrpc 2e-5   # MRPC with one learning rate; try 5e-5, 4e-5, 3e-5 too
python bert_part4_squad.py            # span search, SQuAD v1.1 and v2.0 dev sets (public checkpoints)
```

<figure><img src="/img/papers/bert/part4-heads-run.png" alt="Terminal output of bert_part4_heads.py: the GLUE classifier logits and loss computed by hand match the library, and the SWAG scores match" loading="lazy" /><figcaption>The real output of bert_part4_heads.py.</figcaption></figure>

<figure><img src="/img/papers/bert/part4-sst2-run.png" alt="Terminal output of the SST-2 fine-tuning run: dev accuracy every few hundred steps and the final result" loading="lazy" /><figcaption>The real log of the SST-2 fine-tuning run (bert_part4_finetune.py sst2).</figcaption></figure>

<figure><img src="/img/papers/bert/part4-mrpc-run.png" alt="Terminal output of the MRPC fine-tuning run with learning rate 5e-5" loading="lazy" /><figcaption>The real log of the best MRPC run (bert_part4_finetune.py mrpc 5e-5).</figcaption></figure>

<figure><img src="/img/papers/bert/part4-squad-run.png" alt="Terminal output of bert_part4_squad.py: the span search on the abstract, SQuAD v1.1 dev EM and F1, the SQuAD 2.0 null scores and the threshold search" loading="lazy" /><figcaption>The real output of bert_part4_squad.py.</figcaption></figure>

</details>

## References

1. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019.
2. A. Wang, A. Singh, J. Michael, F. Hill, O. Levy, S. Bowman. [*GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*](https://arxiv.org/abs/1804.07461). ICLR 2019.
3. P. Rajpurkar, J. Zhang, K. Lopyrev, P. Liang. [*SQuAD: 100,000+ Questions for Machine Comprehension of Text*](https://arxiv.org/abs/1606.05250). EMNLP 2016.
4. P. Rajpurkar, R. Jia, P. Liang. [*Know What You Don't Know: Unanswerable Questions for SQuAD*](https://arxiv.org/abs/1806.03822) (SQuAD 2.0). ACL 2018.
5. R. Zellers, Y. Bisk, R. Schwartz, Y. Choi. [*SWAG: A Large-Scale Adversarial Dataset for Grounded Commonsense Inference*](https://arxiv.org/abs/1808.05326). EMNLP 2018.
6. Google Research. [BERT code](https://github.com/google-research/bert) (`run_classifier.py`, `run_squad.py`).
7. Public checkpoints used here: [csarron/bert-base-uncased-squad-v1](https://huggingface.co/csarron/bert-base-uncased-squad-v1) and [deepset/bert-base-uncased-squad2](https://huggingface.co/deepset/bert-base-uncased-squad2).
