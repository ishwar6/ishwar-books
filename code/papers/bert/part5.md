---
title: "Ablations: What Really Matters"
description: "Section 5 and Appendix C of the BERT paper, line by line: which parts of BERT actually cause its gains. Next sentence prediction, bidirectionality, model size, frozen features versus fine-tuning, training length and the 80/10/10 masking recipe, every table row explained, with real parameter counts, a real perplexity and a real re-run of the feature-based NER experiment."
part: 5
covers: "§5, A.4, C.1, C.2"
date: 2026-10-04
tags: [bert, nlp, ablation]
---

Part 4 showed that BERT wins almost everywhere. But a win does not tell you *why*. BERT changed many things at once compared with OpenAI GPT: two new pre-training tasks, bidirectional attention, more data, bigger batches, tuned learning rates. Which of these actually mattered?

This part reads the paper's answer: Section 5 (the ablation studies), plus Appendix A.4 and Appendix C, which hold the rest of the evidence. Every table row gets explained, and I check what can be checked in code.

## What an ablation study is {§5}

> [!PAPER] Devlin et al. (2018), BERT · Section 5 · page 7
> [![The opening of Section 5, Ablation Studies: in this section, we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance](/img/papers/bert/p5-intro.png)](/img/papers/bert/p5-intro.png)
>
> **Context:** right after the experiments of Section 4 (GLUE, SQuAD, SWAG), where BERT set new records.
>
> **What it says:** "we perform ablation experiments over a number of facets of BERT in order to better understand their relative importance." (The sentence continues: more ablations are in Appendix C.)
>
> **Why it matters:** this section turns "BERT is good" into "BERT is good *because of* these specific choices".

> [!DEFINITION] Ablation study
> An experiment where you remove (or replace) one part of a system, keep everything else the same, and measure how much worse it gets. The word comes from medicine, where ablation means removing tissue. If removing a part costs a lot, that part mattered.

> [!DEFINITION] Dev set (development set)
> A set of labeled examples kept aside from training, used to compare versions of a model and pick settings. Almost every number in this part is a Dev set score, not an official Test score like those of Part 4 (Table 7 is the one exception, and it shows both). Ablations use Dev scores partly because official test servers limit submissions: the paper made only one GLUE test submission per model (its footnote 9).

The key rule of a good ablation is in the next screenshot: change **one** thing at a time.

## Effect of the pre-training tasks {§5.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 5.1 · page 8
> [![Section 5.1: two pre-training objectives evaluated using exactly the same pre-training data, fine-tuning scheme, and hyperparameters as BERT-base. No NSP is a bidirectional model trained with the masked LM but without next sentence prediction. LTR and No NSP is a left-context-only model trained as a standard left-to-right LM, with the left-only constraint also applied at fine-tuning, and no NSP. This is directly comparable to OpenAI GPT, but using a larger training dataset, the BERT input representation and the BERT fine-tuning scheme](/img/papers/bert/p5-setup.png)](/img/papers/bert/p5-setup.png)
>
> **Context:** the first ablation, about the two pre-training tasks of Part 3 (masked LM and next sentence prediction).
>
> **What it says:** two new models are pre-trained "using exactly the same pre-training data, fine-tuning scheme, and hyperparameters as BERT-base". **No NSP** keeps the masked LM but drops next sentence prediction. **LTR & No NSP** is a plain left-to-right language model with no NSP, and the left-only rule is kept during fine-tuning too. That second model "is directly comparable to OpenAI GPT", except for BERT's larger data, input format and fine-tuning recipe.
>
> **Why it matters:** these models are expensive (each is a full pre-training run), and they are designed so that each one differs from the previous by exactly one change.

> [!DEFINITION] Hyperparameters
> The settings you choose before training rather than learn: batch size, learning rate, number of steps, dropout rate. "Same hyperparameters" means the only difference between two runs is the one thing being tested.

> [!DEFINITION] LTR and RTL
> **LTR** means left-to-right: a language model that predicts each token from the tokens before it (Part 1). **RTL** means right-to-left: the same thing reading backwards.

One detail in the screenshot deserves attention: for **LTR & No NSP**, "the left-only constraint was also applied at fine-tuning, because removing it introduced a pre-train/fine-tune mismatch that degraded downstream performance". In other words, you cannot take a model trained to look only left and suddenly let it look right during fine-tuning: it never learned what to do with that information, and it got worse, not better.

{{FIG:p5_ladder|The four models of Table 5. Each row changes exactly one thing from the row before, so each difference in scores can be blamed on that one change.}}

### Table 5, row by row

> [!PAPER] Devlin et al. (2018), BERT · Section 5.1, Table 5 · page 8
> [![Table 5 of the BERT paper: Dev set results on MNLI-m, QNLI, MRPC, SST-2 (accuracy) and SQuAD (F1). BERT-base 84.4, 88.4, 86.7, 92.7, 88.5. No NSP 83.9, 84.9, 86.5, 92.6, 87.9. LTR and No NSP 82.1, 84.3, 77.5, 92.1, 77.8. Plus BiLSTM 82.1, 84.1, 75.7, 91.6, 84.9](/img/papers/bert/p5-table5.png)](/img/papers/bert/p5-table5.png)
>
> **Context:** the results of the models just described, all with the BERT-base architecture.
>
> **What it says:** four rows (BERT-base, No NSP, LTR & No NSP, + BiLSTM) and five tasks. "+ BiLSTM" adds "a randomly initialized BiLSTM on top of the 'LTR + No NSP' model during fine-tuning".
>
> **Why it matters:** this table is the main evidence for the paper's central claim, that bidirectionality is what makes BERT work.

The five columns are tasks from Part 4:

| Column | Task | Score |
|---|---|---|
| MNLI-m | does sentence B follow from A, contradict it, or neither? ("m" = matched: test sentences from the same genres as training) | accuracy |
| QNLI | does this sentence contain the answer to this question? | accuracy |
| MRPC | do these two news sentences mean the same thing? | accuracy |
| SST-2 | is this movie-review sentence positive or negative? | accuracy |
| SQuAD | find the answer span in a paragraph (v1.1) | F1 |

> [!DEFINITION] BiLSTM
> A bidirectional LSTM: two LSTMs (Part 1), one reading the tokens left to right and one right to left, with their outputs joined for every token. Placed on top of another model, it adds some right-side context, but only in its own small layer.

Numbers in a table are easier to compare as differences. My script `bert_part5.py` subtracts each row from the row before (the Table 5 values are typed in from the paper):

```text
Table 5 (Dev set): change in points
                                                MNLI-m    QNLI    MRPC   SST-2   SQuAD
cost of removing NSP                              -0.5    -3.5    -0.2    -0.1    -0.6
cost of left-to-right instead of MLM              -1.8    -0.6    -9.0    -0.5   -10.1
effect of adding a BiLSTM to the LTR model        +0.0    -0.2    -1.8    -0.5    +7.1
```

{{FIG:p5_table5|Table 5 as a chart (Dev set; the axis starts at 70 so the gaps are visible). Removing NSP mostly hurts QNLI. Switching to left-to-right crashes MRPC and SQuAD. The BiLSTM rescues part of SQuAD but nothing else.}}

**Row 1 to row 2: removing NSP.**

> [!PAPER] Devlin et al. (2018), BERT · Section 5.1 · page 8
> [![We first examine the impact brought by the NSP task. In Table 5, we show that removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1. The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD](/img/papers/bert/p5-nsp-ltr.png)](/img/papers/bert/p5-nsp-ltr.png)
>
> **Context:** the paper's reading of Table 5.
>
> **What it says:** "removing NSP hurts performance significantly on QNLI, MNLI, and SQuAD 1.1." And comparing No NSP with LTR & No NSP: "The LTR model performs worse than the MLM model on all tasks, with large drops on MRPC and SQuAD."
>
> **Why it matters:** two separate claims: NSP helps, and bidirectionality helps a lot.

Looking at the computed differences, the NSP claim is weaker than the word "significantly" suggests. QNLI really drops (**3.5** points), but MNLI drops **0.5** and SQuAD **0.6**, and MRPC and SST-2 barely move (0.2 and 0.1). The paper does not report how much these Dev scores vary between runs, so we cannot tell how much of a 0.5-point gap is noise. Keep this in mind: in Part 6 we will see that later work (RoBERTa) questioned whether NSP is needed at all.

> [!NOTE] Why QNLI might care about NSP
> QNLI asks whether a sentence answers a question: a judgement about how two pieces of text relate, which is exactly what NSP practises (Part 3). That fits the drop, but the paper does not test this explanation directly.

**Row 2 to row 3: one direction instead of two.** This is the big one. With everything else equal, going from the masked LM to a left-to-right LM costs **9.0** points on MRPC and **10.1** points of F1 on SQuAD, and loses something on every task. This is the cleanest evidence in the paper that the bidirectional design, not just more data or a bigger batch, explains much of BERT's gain.

> [!PAPER] Devlin et al. (2018), BERT · Section 5.1 · page 8
> [![For SQuAD it is intuitively clear that a LTR model will perform poorly at token predictions, since the token-level hidden states have no right-side context. In order to make a good faith attempt at strengthening the LTR system, we added a randomly initialized BiLSTM on top](/img/papers/bert/p5-bilstm.png)](/img/papers/bert/p5-bilstm.png)
> [![The paragraph ends at the top of the next column: results are still far worse than those of the pre-trained bidirectional models. The BiLSTM hurts performance on the GLUE tasks](/img/papers/bert/p5-bilstm2.png)](/img/papers/bert/p5-bilstm2.png)
>
> **Context:** the paragraph runs from the bottom of the left column to the top of the right column of page 8.
>
> **What it says:** an LTR model does badly on SQuAD "since the token-level hidden states have no right-side context". As "a good faith attempt" to help it, they add "a randomly initialized BiLSTM on top". It "does significantly improve results on SQuAD", but they stay "far worse" than the bidirectional models, and "the BiLSTM hurts performance on the GLUE tasks".
>
> **Why it matters:** it answers the obvious objection, "just add some right context on top", and shows it is not enough.

Why does SQuAD suffer most? SQuAD needs a start and an end position for the answer (Part 4). Whether a word is the *end* of an answer depends heavily on the words after it. In an LTR model, the vector for each token was built without ever seeing those words. The BiLSTM adds right-side context, which is why SQuAD jumps by **7.1** points (77.8 to 84.9). But it is one small layer trained only on the task data, while BERT mixes both sides in all 12 layers during pre-training. The result is still **3.6** points below BERT-base (84.9 versus 88.5).

On the GLUE tasks the BiLSTM does not help at all (MNLI unchanged, the others down by 0.2 to 1.8 points). A plausible reason, not tested in the paper: a randomly initialized layer has to be learned from small task datasets, and MRPC has only a few thousand training pairs (3,600, according to Section 5.2).

### Why not just do what ELMo does?

> [!PAPER] Devlin et al. (2018), BERT · Section 5.1 · page 8
> [![We recognize that it would also be possible to train separate LTR and RTL models and represent each token as the concatenation of the two models, as ELMo does. However: (a) this is twice as expensive as a single bidirectional model; (b) this is non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question; (c) this it is strictly less powerful than a deep bidirectional model, since it can use both left and right context at every layer](/img/papers/bert/p5-elmo-args.png)](/img/papers/bert/p5-elmo-args.png)
>
> **Context:** the last paragraph of Section 5.1. The paper did not run this experiment; it argues instead.
>
> **What it says:** you could train a separate LTR and RTL model and glue their vectors together, "as ELMo does". Three reasons against: (a) "twice as expensive"; (b) "non-intuitive for tasks like QA, since the RTL model would not be able to condition the answer on the question"; (c) "strictly less powerful than a deep bidirectional model, since it can use both left and right context at every layer".
>
> **Why it matters:** it explains why the paper did not simply build a bigger ELMo.

Each argument in plain words:

- **(a) Cost.** Two full models to pre-train and to run, instead of one.
- **(b) Question answering.** In BERT's input the question comes first and the paragraph second (Part 4). A right-to-left model reading the paragraph has not reached the question yet (the question is to its left), so its vectors for the paragraph words know nothing about what is being asked.
- **(c) Depth.** In ELMo the two directions meet only at the very end. In BERT, every layer mixes both sides, so later layers can build on information that already combines left and right. A small grammar slip in the paper can confuse readers here: in "this it is strictly less powerful ... since **it** can use both left and right context at every layer", the second "it" means the deep bidirectional model, not the concatenation.

> [!NOTE] An argument, not a measurement
> Point (c) says "strictly less powerful". That is a statement about what the model can represent in principle. The paper gives no experiment comparing BERT with an LTR+RTL concatenation of the same size.

## The other differences between BERT and GPT {§A.4}

Table 5's LTR & No NSP row matters for one more reason, which the paper explains in Appendix A.4.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.4 · page 14
> [![Appendix A.4: OpenAI GPT is the most comparable method. Many design decisions in BERT were made to be as close to GPT as possible. Other differences: GPT is trained on the BooksCorpus (800M words), BERT on BooksCorpus and Wikipedia (2,500M words); GPT uses [SEP] and [CLS] only at fine-tuning time, BERT learns them and the sentence A and B embeddings during pre-training; GPT was trained for 1M steps with a batch size of 32,000 words, BERT for 1M steps with 128,000 words; GPT used the same learning rate of 5e-5 for all fine-tuning experiments, BERT chooses a task-specific fine-tuning learning rate](/img/papers/bert/p5-a4.png)](/img/papers/bert/p5-a4.png)
>
> **Context:** Appendix A.4 compares BERT, ELMo and OpenAI GPT (Figure 3 of the paper, which we read in Part 1).
>
> **What it says:** BERT was designed to be as close to GPT as possible, "so that the two methods could be minimally compared". But four other differences remain: training data, when [SEP], [CLS] and the A/B embeddings are learned, batch size, and the fine-tuning learning rate.
>
> **Why it matters:** any of these four could explain part of BERT's advantage over GPT. A fair comparison has to rule them out.

| | OpenAI GPT | BERT |
|---|---|---|
| Pre-training data | BooksCorpus, 800M words | BooksCorpus 800M + Wikipedia 2,500M words |
| [SEP], [CLS], segment A/B | added only at fine-tuning | learned during pre-training |
| Batch size | 32,000 words, 1M steps | 128,000 words, 1M steps |
| Fine-tuning learning rate | always 5e-5 | best of several, chosen per task on Dev |

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.4 · page 14
> [![To isolate the effect of these differences, we perform ablation experiments in Section 5.1 which demonstrate that the majority of the improvements are in fact coming from the two pre-training tasks and the bidirectionality they enable](/img/papers/bert/p5-a4-end.png)](/img/papers/bert/p5-a4-end.png)
>
> **Context:** the end of Appendix A.4, pointing back to Section 5.1.
>
> **What it says:** the Section 5.1 ablations show "that the majority of the improvements are in fact coming from the two pre-training tasks and the bidirectionality they enable".
>
> **Why it matters:** this is how the four differences are ruled out.

The logic: the LTR & No NSP model is GPT's recipe (left to right, no NSP) but trained with **all four of BERT's advantages** (BERT's data, BERT's input format, BERT's batch size, BERT's fine-tuning). So any gap between LTR & No NSP and BERT-base cannot come from those four things. It must come from the masked LM, which gives bidirectionality, and NSP. And that gap is large: up to 9.2 points on MRPC (77.5 versus 86.7) and 10.7 on SQuAD (77.8 versus 88.5).

## Effect of model size {§5.2}

> [!PAPER] Devlin et al. (2018), BERT · Section 5.2 · page 8
> [![Section 5.2: we trained BERT models with a differing number of layers, hidden units and attention heads, otherwise using the same hyperparameters and training procedure. Table 6 reports the average Dev Set accuracy from 5 random restarts of fine-tuning. Larger models lead to a strict accuracy improvement across all four datasets, even for MRPC which only has 3,600 labeled training examples](/img/papers/bert/p5-size-intro.png)](/img/papers/bert/p5-size-intro.png)
>
> **Context:** the second ablation: not *what* BERT learns, but *how big* it is.
>
> **What it says:** several BERT models with different numbers of layers, hidden sizes and heads, all trained the same way. Table 6 reports "the average Dev Set accuracy from 5 random restarts of fine-tuning". "Larger models lead to a strict accuracy improvement", "even for MRPC which only has 3,600 labeled training examples".
>
> **Why it matters:** in 2018 many people expected a huge model to overfit a tiny dataset. Here bigger was better even on the smallest task.

> [!DEFINITION] Random restarts
> Fine-tuning the same pre-trained model several times with different random choices (the order of the training examples and the starting values of the new output layer). Results vary a little between restarts, so averaging 5 of them gives a steadier number.

> [!DEFINITION] Overfitting
> When a model memorises its training examples instead of learning the general pattern, so it does well on training data but badly on new data. Big models with few examples are the classic risk.

A small inconsistency: the text says "all four datasets", but Table 6 shows three tasks (MNLI-m, MRPC, SST-2). The paper does not say what the fourth one is.

> [!PAPER] Devlin et al. (2018), BERT · Section 5.2, Table 6 · page 9
> [![Table 6: ablation over BERT model size. Columns: number of layers, hidden size, number of attention heads, masked LM perplexity of held-out training data, and Dev accuracy on MNLI-m, MRPC and SST-2. Rows: 3, 768, 12, 5.84, 77.9, 79.8, 88.4. 6, 768, 3, 5.24, 80.6, 82.2, 90.7. 6, 768, 12, 4.68, 81.9, 84.8, 91.3. 12, 768, 12, 3.99, 84.4, 86.7, 92.9. 12, 1024, 16, 3.54, 85.7, 86.9, 93.3. 24, 1024, 16, 3.23, 86.6, 87.8, 93.7](/img/papers/bert/p5-table6.png)](/img/papers/bert/p5-table6.png)
>
> **Context:** the table for Section 5.2. The fourth row (12, 768, 12) is BERT-base and the last (24, 1024, 16) is BERT-large.
>
> **What it says:** six model sizes. For each: #L (layers), #H (hidden size), #A (attention heads), "LM (ppl)", "the masked LM perplexity of held-out training data", and Dev accuracy on three tasks.
>
> **Why it matters:** every column improves as you go down the table: perplexity falls, and all three accuracies rise.

You met L, H and A in Part 2. The new column is perplexity.

> [!DEFINITION] Perplexity (ppl)
> A measure of how surprised a language model is by real text. A perplexity of 4 means the model is, on average, as unsure as if it were choosing between 4 equally likely words. Lower is better; a perfect model would have perplexity 1.

For a masked LM, the perplexity is computed over the positions the model has to predict:

$$
\text{ppl} = \exp\!\left( -\frac{1}{N} \sum_{i=1}^{N} \log p(x_i \mid \text{context}) \right)
$$

where:

- $$N$$ is the number of predicted (masked) positions;
- $$x_i$$ is the true token at the $$i$$-th predicted position;
- $$p(x_i \mid \text{context})$$ is the probability the model gave to that true token, from its softmax over the vocabulary;
- $$\log$$ is the natural logarithm, and the minus sign makes the sum positive: the inside of the bracket is the average **cross-entropy** loss (Part 3);
- $$\exp$$ ("e to the power of") undoes the logarithm, turning an average loss into a number of "equally likely choices".

So BERT-base's 3.99 means: on its held-out training text, when it predicts a masked token, it is on average about as unsure as a choice between 4 words.

### How many parameters is each row?

The paper gives parameter counts only for BERT-base and BERT-large. I built every model of Table 6 with the Hugging Face `BertConfig` (30,522-token vocabulary, 512 positions, feed-forward size 4H, as in Part 2) on PyTorch's `meta` device, which creates the shapes without allocating any memory, and counted:

```python
import torch
from transformers import BertConfig, BertModel

for L, H, A in [(3, 768, 12), (6, 768, 3), (6, 768, 12), (12, 768, 12), (12, 1024, 16), (24, 1024, 16)]:
    cfg = BertConfig(vocab_size=30522, hidden_size=H, num_hidden_layers=L, num_attention_heads=A,
                     intermediate_size=4 * H, max_position_embeddings=512, type_vocab_size=2)
    with torch.device("meta"):              # shapes only, no memory
        model = BertModel(cfg)
    print(L, H, A, sum(p.numel() for p in model.parameters()))
```

```text
 #L    #H  #A        params       formula  embeddings   ppl  MNLI-m  MRPC  SST-2
  3   768  12    45,691,392    45,691,392  23,837,184  5.84    77.9  79.8   88.4
  6   768   3    66,955,008    66,955,008  23,837,184  5.24    80.6  82.2   90.7
  6   768  12    66,955,008    66,955,008  23,837,184  4.68    81.9  84.8   91.3
 12   768  12   109,482,240   109,482,240  23,837,184  3.99    84.4  86.7   92.9
 12  1024  16   183,987,200   183,987,200  31,782,912  3.54    85.7  86.9   93.3
 24  1024  16   335,141,888   335,141,888  31,782,912  3.23    86.6  87.8   93.7
```

The "formula" column is the count from the equation of Part 2 (embeddings, plus $$12H^2 + 13H$$ per layer, plus the $$H \times H$$ pooler), and it matches the real model exactly in every row. Three things stand out:

- **BERT-base is 109.5M** and **BERT-large is 335.1M** parameters, counted this way. The paper rounds them to "110M" and "340M". (Part 2 discusses the 340M.)
- **The two 6-layer models have exactly the same number of parameters** (66,955,008), yet 12 heads beat 3 heads on all three tasks (MNLI-m 81.9 versus 80.6). The number of heads only changes how the same $$H \times H$$ matrices are split up, not how many numbers they hold.
- **The embeddings are a big share of small models.** In the 3-layer model, 23.8M of its 45.7M parameters (52%) are embeddings, the word-lookup table alone.

{{FIG:p5_size|Table 6 as a chart. Accuracy rises with every step up in size, for all three tasks, including MRPC with its few thousand training examples. Parameter counts are my measurements; everything else is from the paper.}}

### Was this big for 2018?

> [!PAPER] Devlin et al. (2018), BERT · Section 5.2 · page 8
> [![For example, the largest Transformer explored in Vaswani et al. (2017) is (L=6, H=1024, A=16) with 100M parameters for the encoder, and the largest Transformer we have found in the literature is (L=64, H=512, A=2) with 235M parameters (Al-Rfou et al., 2018). By contrast, BERT-base contains 110M parameters and BERT-large contains 340M parameters](/img/papers/bert/p5-size-prior.png)](/img/papers/bert/p5-size-prior.png)
>
> **Context:** the paper places its model sizes in the literature of the time.
>
> **What it says:** the largest Transformer in *Attention Is All You Need* is (L=6, H=1024, A=16) with "100M parameters for the encoder"; the largest they found anywhere is (L=64, H=512, A=2) with 235M (Al-Rfou et al., 2018). BERT-base has 110M and BERT-large 340M.
>
> **Why it matters:** BERT-large was bigger than any Transformer the authors knew of. The gains in Table 6 came on top of models that were "already quite large".

> [!PAPER] Devlin et al. (2018), BERT · Section 5.2 · pages 8 and 9
> [![It has long been known that increasing the model size will lead to continual improvements on large-scale tasks such as machine translation and language modeling. However, we believe that this is the first work to demonstrate convincingly that scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained](/img/papers/bert/p5-size-claim.png)](/img/papers/bert/p5-size-claim.png)
> [![Peters et al. presented mixed results on increasing the pre-trained bi-LM size from two to four layers, and Melamud et al. found 200 to 600 hidden units helped but 1,000 did not. Both used a feature-based approach. We hypothesize that when the model is fine-tuned directly and uses only a very small number of randomly initialized additional parameters, the task-specific models can benefit from the larger, more expressive pre-trained representations even when downstream task data is very small](/img/papers/bert/p5-size-claim2.png)](/img/papers/bert/p5-size-claim2.png)
>
> **Context:** the end of Section 5.2, running onto page 9.
>
> **What it says:** bigger models were already known to help large tasks like translation. The new claim: "scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained". Earlier feature-based work found mixed results from making models bigger. The authors "hypothesize" the difference is fine-tuning: with only a few new random parameters, even a small task can use a large pre-trained model.
>
> **Why it matters:** this is the seed of the idea that dominated the following years: pre-train a bigger model and every task benefits.

Note the careful word "hypothesize": the paper offers an explanation for why fine-tuning scales where feature-based approaches did not, but does not test it directly. The measured fact is Table 6 itself.

### A real perplexity, for scale

What does a masked-LM perplexity of about 4 look like on real text? I measured the released `bert-base-uncased` on the test split of WikiText-2, a public set of Wikipedia articles, using the paper's masking recipe (15% of tokens chosen; of those 80% `[MASK]`, 10% random, 10% unchanged), sequences of 512 tokens, and a fixed random seed:

```python
chosen = torch.rand(x.shape) < 0.15                    # pick 15% of positions
chosen[:, 0] = chosen[:, -1] = False                   # never [CLS] or [SEP]
r = torch.rand(x.shape)
inp = x.clone()
inp[chosen & (r < 0.8)] = tok.mask_token_id            # 80%: [MASK]
rnd = chosen & (r >= 0.8) & (r < 0.9)                  # 10%: a random token
inp[rnd] = torch.randint(len(tok), x.shape)[rnd]       # the other 10% stay unchanged
logp = torch.log_softmax(model(input_ids=inp).logits[chosen], -1)
nll += -logp.gather(1, x[chosen][:, None]).sum()       # cross-entropy on the chosen positions only
```

```text
WikiText-2 test: 261,428 WordPiece tokens -> 512 sequences of 512 ([CLS] + 510 + [SEP])
predicted positions: 38,984 of 261,120 (14.93%): 31,282 [MASK], 3,947 random, 3,755 unchanged
mean cross-entropy = 1.9967 nats -> masked-LM perplexity = 7.36   (top-1 accuracy 63.6%)
```

So on this text the released model has a perplexity of **7.36**, and its first guess is right **63.6%** of the time. That is higher (worse) than the **3.99** in Table 6. I cannot reproduce the paper's number, and the comparison is not like for like:

- The paper measured "held-out training data": text from the same BooksCorpus and Wikipedia mix BERT was trained on. WikiText-2 is a different sample, with its own formatting (I undid its `@-@` style marks and dropped headings, but other differences remain).
- Table 6's models are the ablation models, trained with the same procedure; the paper does not say the 3.99 was measured on the released checkpoint.
- My random replacements draw from the whole vocabulary, including rarely used tokens.

What the measurement does show: the 15% / 80-10-10 recipe behaves as described (14.93% of positions chosen, split 80.2% / 10.1% / 9.6%), and a single-digit perplexity on unseen text is in the same range as Table 6's numbers.

## Feature-based approach with BERT {§5.3}

Part 1 introduced two ways to reuse a pre-trained model: feature-based (freeze it, use its vectors) and fine-tuning (train everything). Every BERT result so far was fine-tuned. Section 5.3 asks whether frozen BERT features are good too.

> [!PAPER] Devlin et al. (2018), BERT · Section 5.3 · page 9
> [![Section 5.3: all results so far used fine-tuning. However, the feature-based approach, where fixed features are extracted from the pre-trained model, has certain advantages. First, not all tasks can be easily represented by a Transformer encoder architecture. Second, there are major computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top](/img/papers/bert/p5-feature-why.png)](/img/papers/bert/p5-feature-why.png)
>
> **Context:** the third ablation: not what BERT learns or how big it is, but how you use it.
>
> **What it says:** the feature-based approach "has certain advantages". First, "not all tasks can be easily represented by a Transformer encoder architecture". Second, "major computational benefits": compute the expensive BERT vectors once, then run many experiments "with cheaper models on top".
>
> **Why it matters:** in practice, frozen features are often what people use: for search indexes, for clustering, or when the same text is reused by many small models.

**Use case.** A team wants to try twenty different classifiers on a million documents. Fine-tuning means twenty full BERT training runs. Feature-based means running BERT once over the million documents, saving the vectors, and training twenty small models on the saved vectors, each in minutes.

> [!PAPER] Devlin et al. (2018), BERT · Section 5.3 · page 9
> [![The two approaches are compared on the CoNLL-2003 Named Entity Recognition task. The input uses a case-preserving WordPiece model and includes the maximal document context provided by the data. Following standard practice it is a tagging task, but no CRF layer is used](/img/papers/bert/p5-feature-setup.png)](/img/papers/bert/p5-feature-setup.png)
> [![We use the representation of the first sub-token as the input to the token-level classifier over the NER label set](/img/papers/bert/p5-feature-setup-b.png)](/img/papers/bert/p5-feature-setup-b.png)
> [![To ablate the fine-tuning approach, we apply the feature-based approach by extracting the activations from one or more layers without fine-tuning any parameters of BERT. These contextual embeddings are used as input to a randomly initialized two-layer 768-dimensional BiLSTM before the classification layer](/img/papers/bert/p5-feature-setup2.png)](/img/papers/bert/p5-feature-setup2.png)
>
> **Context:** the experimental setup, split across the two columns of page 9.
>
> **What it says:** the task is CoNLL-2003 named entity recognition. BERT reads text with a "case-preserving WordPiece model" and "the maximal document context". There is no CRF layer, and each word is represented by its **first sub-token**. For the feature-based runs, activations are extracted "without fine-tuning any parameters of BERT" and fed to "a randomly initialized two-layer 768-dimensional BiLSTM".
>
> **Why it matters:** these details are what you would need to copy to reproduce the result.

> [!DEFINITION] CoNLL-2003
> A standard named entity recognition dataset of English news stories (Reuters), from a 2003 shared task. Every word is tagged as part of a person, organisation, location, miscellaneous name, or none.

> [!DEFINITION] Case-preserving (cased)
> A tokenizer that keeps capital letters, so "Apple" and "apple" are different tokens. The `bert-base-uncased` model of earlier parts lowercases everything; for names, capitals are a strong clue, so NER uses the cased model.

> [!DEFINITION] CRF (conditional random field)
> An output layer often put on top of taggers. It scores whole sequences of tags, so it can learn rules like "I-PER cannot follow B-ORG". BERT leaves it out and predicts every tag on its own.

> [!DEFINITION] First sub-token
> WordPiece can split one word into several pieces (Part 2): "Lovelace" might become `Love` `##lace`. A tagger needs one label per word, so BERT uses only the vector of the first piece and ignores the rest.

> [!DEFINITION] Hidden state (activation)
> The vector a layer outputs for each token. BERT-base has 13 of them per token: the embedding output, then one after each of the 12 layers. "Extracting the activations" means saving some of these vectors.

The six feature choices in Table 7 read different hidden states:

{{FIG:p5_layers|The six feature choices of Table 7. A weighted sum learns one weight per layer and adds the layers up; concatenation glues the last four 768-number vectors into one of 3,072 numbers.}}

> [!PAPER] Devlin et al. (2018), BERT · Section 5.3, Table 7 · page 9
> [![Table 7: CoNLL-2003 NER results. ELMo Dev F1 95.7, Test F1 92.2. CVT Test 92.6. CSE Test 93.1. Fine-tuning: BERT-large Dev 96.6, Test 92.8; BERT-base Dev 96.4, Test 92.4. Feature-based BERT-base, Dev F1: Embeddings 91.0, Second-to-Last Hidden 95.6, Last Hidden 94.9, Weighted Sum Last Four Hidden 95.9, Concat Last Four Hidden 96.1, Weighted Sum All 12 Layers 95.5. Scores are averaged over 5 random restarts](/img/papers/bert/p5-table7.png)](/img/papers/bert/p5-table7.png)
>
> **Context:** the results of Section 5.3.
>
> **What it says:** three earlier systems (ELMo, CVT, CSE), BERT fine-tuned, and six feature-based BERT-base variants. Scores are entity-level F1, "averaged over 5 random restarts".
>
> **Why it matters:** it shows how close frozen BERT gets to fine-tuned BERT.

> [!PAPER] Devlin et al. (2018), BERT · Section 5.3 · page 9
> [![Results are presented in Table 7. BERT-large performs competitively with state-of-the-art methods. The best performing method concatenates the token representations from the top four hidden layers of the pre-trained Transformer, which is only 0.3 F1 behind fine-tuning the entire model. This demonstrates that BERT is effective for both fine-tuning and feature-based approaches](/img/papers/bert/p5-feature-result.png)](/img/papers/bert/p5-feature-result.png)
>
> **Context:** the paper's reading of Table 7.
>
> **What it says:** BERT-large "performs competitively with state-of-the-art methods". The best feature-based variant, concatenating the top four layers, is "only 0.3 F1 behind fine-tuning the entire model".
>
> **Why it matters:** you can freeze BERT and lose very little, at least on this task.

Reading Table 7 carefully:

- **Fine-tuning:** BERT-large 96.6 Dev / 92.8 Test, BERT-base 96.4 / 92.4. Note the word "competitively": on the **Test** set, CSE (93.1) is higher than BERT-large (92.8). NER is not one of BERT's eleven new records.
- **Embeddings alone: 91.0.** The embedding output is the word-lookup vector plus position and segment, before any attention. It knows nothing about the sentence. Context is worth more than 5 points here.
- **Last hidden 94.9 is worse than second-to-last 95.6.** One common explanation (not tested in the paper): the last layer is shaped most strongly by the pre-training tasks, predicting masked words, so the layer before it holds more general information.
- **Combining layers helps:** weighted sum of the last four 95.9, concatenating them **96.1**. Summing all 12 layers (95.5) is worse than using only the top four.
- **96.1 versus 96.4:** the best frozen choice is 0.3 points behind fine-tuned BERT-base on Dev.

> [!DEFINITION] Weighted sum of layers
> Multiply each layer's vector by a learned number (its weight) and add them up. The result has the same size as one layer (768 numbers). ELMo combined its layers this way. In code: `(softmax(w)[:, None] * layers).sum(0)`.

### A smaller re-run of the feature-based experiment

Can we see the same pattern ourselves? I ran a smaller version of the experiment in `bert_part5_ner.py`:

- **Model:** `bert-base-cased`, frozen (no weight of BERT changes).
- **Data:** CoNLL-2003 from the Hugging Face hub (`eriktks/conll2003`). Training: the first **5,000** of the 14,041 training sentences. Evaluation: the whole Dev set, **3,250** sentences (51,362 words). Each sentence is read on its own: this copy of the dataset has no document boundaries, so there is no "maximal document context".
- **Features:** all 13 hidden states, first sub-token of every word, saved once.
- **Classifier:** a 2-layer BiLSTM with 384 units per direction (768 in total, my reading of "768-dimensional"), then a linear layer over the 9 tags; Adam, learning rate 0.001, batch 32, 4 epochs. Each choice trained with **3 seeds**, averaged (the paper used 5).
- **Score:** entity-level F1 on Dev, computed by my own code (an entity counts as correct only if its type and its exact span match).

The heart of the feature extraction:

```python
enc = tok(words, is_split_into_words=True, return_tensors="pt")
with torch.no_grad():                                                  # BERT is frozen
    hs = torch.stack(bert(**enc, output_hidden_states=True).hidden_states, 2)[0]   # (tokens, 13, 768)
wid = enc.word_ids(0)                                                  # which word each token belongs to
first = [t for t, w in enumerate(wid) if w is not None and (t == 0 or wid[t - 1] != w)]
features = hs[first]                                                   # (words, 13, 768): first sub-token only

concat_last_four = features[:, 9:13].flatten(1)                        # (words, 3072)
weighted_last_four = (torch.softmax(w, 0)[None, :, None] * features[:, 9:13]).sum(1)   # w: 4 learned weights
```

The results:

```text
CoNLL-2003: 5000 training sentences used, 3250 dev sentences; labels: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC']
features: 67634 train words, 51362 dev words, 13 layers x 768 numbers each, 34 s
Embeddings                   dev F1 = 82.66   (seeds: 82.73, 82.42, 82.82; 113 s)
Second-to-last hidden        dev F1 = 89.94   (seeds: 88.72, 90.25, 90.84; 127 s)
Last hidden                  dev F1 = 88.64   (seeds: 88.13, 89.03, 88.77; 161 s)
Weighted sum last four       dev F1 = 90.37   (seeds: 90.08, 90.00, 91.05; 311 s)
Concat last four             dev F1 = 91.17   (seeds: 90.94, 91.41, 91.16; 484 s)
Weighted sum all 12 layers   dev F1 = 90.49   (seeds: 91.28, 89.60, 90.59; 450 s)
```

{{FIG:p5_ner|Dev F1 for each feature choice: the paper's Table 7 and my smaller re-run. The absolute numbers are lower in my run (fewer training sentences, no document context, a smaller training budget), but the shape agrees.}}

Lined up against Table 7:

| Feature choice | Paper, Dev F1 (Table 7) | Our re-run, Dev F1 |
|---|---|---|
| Embeddings | 91.0 | 82.66 |
| Second-to-last hidden | 95.6 | 89.94 |
| Last hidden | 94.9 | 88.64 |
| Weighted sum last four | 95.9 | 90.37 |
| Concat last four | **96.1** | **91.17** |
| Weighted sum all 12 layers | 95.5 | 90.49 |

What agrees with the paper:

- **The embeddings alone are clearly worst** (82.66), about 6 to 8.5 points below every choice that uses BERT's layers. Context is what BERT adds.
- **Concatenating the last four layers is best** (91.17), in both the paper and our run.
- **The second-to-last layer beats the last layer** (89.94 versus 88.64), as in the paper (95.6 versus 94.9).
- **Combining the top four layers beats any single layer**, whether summed or concatenated.

What does not agree: in our run the weighted sum of all 12 layers (90.49) edges out the weighted sum of the last four (90.37), the opposite of the paper's order. The gap is 0.12 points, while the three seeds of the 12-layer choice alone range from 89.60 to 91.28, so this run cannot separate those two.

Our absolute numbers are about 5 to 8.5 points lower than the paper's. That is expected, not a contradiction: we trained on 5,000 of the 14,041 sentences, for 4 epochs, with no hyperparameter search, and without the document context the paper adds to every sentence. We also did not fine-tune BERT, so this experiment says nothing about the 96.4 of fine-tuning; it only checks the shape of the feature-based rows.

## How long to pre-train {§C.1}

The last two ablations are in Appendix C. The first asks whether BERT's long pre-training is necessary.

> [!PAPER] Devlin et al. (2018), BERT · Appendix C.1 · page 16
> [![Appendix C.1: Figure 5 presents MNLI Dev accuracy after fine-tuning from a checkpoint pre-trained for k steps. Question 1: does BERT really need such a large amount of pre-training (128,000 words per batch times 1,000,000 steps)? Answer: yes, BERT-base achieves almost 1.0% additional accuracy on MNLI when trained on 1M steps compared to 500k steps. Question 2: does MLM pre-training converge slower than LTR pre-training, since only 15% of words are predicted in each batch? Answer: the MLM model does converge slightly slower than the LTR model. However, in terms of absolute accuracy the MLM model begins to outperform the LTR model almost immediately](/img/papers/bert/p5-c1.png)](/img/papers/bert/p5-c1.png)
>
> **Context:** Appendix C.1, "Effect of Number of Training Steps".
>
> **What it says:** two questions. Does BERT need all that pre-training? "Yes": "almost 1.0% additional accuracy on MNLI" at 1M steps compared to 500k. Does the masked LM learn more slowly, since it only predicts 15% of the words? It does "converge slightly slower", but "begins to outperform the LTR model almost immediately".
>
> **Why it matters:** it answers the obvious worry about the masked LM: it wastes 85% of each sentence as a training signal.

> [!DEFINITION] Checkpoint
> A saved copy of a model's weights at some point during training. Fine-tuning from the checkpoint at step k shows how good the model was after k steps of pre-training.

> [!DEFINITION] Converge
> To settle: a training run converges when more steps no longer change it much. "Converges slower" means it needs more steps to reach its best.

The total pre-training in Question 1 is the paper's own product: 128,000 words per batch × 1,000,000 steps = **128 billion** words processed (with the paper's rounding of 256 × 512 = 131,072 to 128,000, as in Part 3).

> [!PAPER] Devlin et al. (2018), BERT · Appendix C.1, Figure 5 · page 16
> [![Figure 5: MNLI Dev accuracy after fine-tuning, starting from model parameters pre-trained for k steps, for k from near 0 to 1,000 thousand. Two curves: BERT-base with the masked LM, rising to about 84, and BERT-base left-to-right, flattening near 82](/img/papers/bert/p5-figure5.png)](/img/papers/bert/p5-figure5.png)
>
> **Context:** the figure for Appendix C.1. The x-axis is k, the pre-training steps in thousands; the y-axis is MNLI Dev accuracy after fine-tuning.
>
> **What it says:** two curves, BERT-base with the masked LM and BERT-base trained left to right, measured at several checkpoints.
>
> **Why it matters:** it shows both answers at once: the MLM curve is still rising at 1M steps, and it sits above the LTR curve almost the whole way.

How to read it:

- **Both curves rise**, then flatten. More pre-training always helped, but with smaller gains later on.
- **The MLM curve is still climbing between 500k and 1M steps**: that is the "almost 1.0%" of Question 1.
- **At the very first checkpoint the left-to-right model is ahead**, which fits "converges slightly slower": it gets a training signal from every word, the MLM from only 15%. But the MLM curve passes it straight away and the gap then widens to about 2 points by the end (reading the figure, not a number printed in the paper).

The trade-off is worth it. The MLM sees fewer training signals per batch, but each one uses context from both sides.

## Masking strategies {§C.2}

Part 3 explained the 80/10/10 masking recipe and *why* the paper uses it. Appendix C.2 tests whether it was a good choice.

> [!PAPER] Devlin et al. (2018), BERT · Appendix C.2 · page 16
> [![Appendix C.2: In Section 3.1, we mention that BERT uses a mixed strategy for masking the target tokens when pre-training with the masked language model objective. The following is an ablation study to evaluate the effect of different masking strategies](/img/papers/bert/p5-c2.png)](/img/papers/bert/p5-c2.png)
> [![Note that the purpose of the masking strategies is to reduce the mismatch between pre-training and fine-tuning, as the [MASK] symbol never appears during the fine-tuning stage. We report the Dev results for both MNLI and NER. For NER, we report both fine-tuning and feature-based approaches, as we expect the mismatch will be amplified for the feature-based approach as the model will not have the chance to adjust the representations](/img/papers/bert/p5-c2-why.png)](/img/papers/bert/p5-c2-why.png)
>
> **Context:** Appendix C.2, "Ablation for Different Masking Procedures".
>
> **What it says:** the mixed strategy exists "to reduce the mismatch between pre-training and fine-tuning, as the [MASK] symbol never appears during the fine-tuning stage". They test MNLI and NER, and for NER also the feature-based approach, because "the mismatch will be amplified for the feature-based approach as the model will not have the chance to adjust the representations".
>
> **Why it matters:** it is a test of a design decision that the main paper only argues for.

> [!DEFINITION] Pre-train / fine-tune mismatch
> A difference between what the model sees during pre-training and what it sees later. During pre-training, inputs contain `[MASK]` tokens; real sentences in a task never do. A model that learned to do its "real work" only at `[MASK]` positions might handle normal text less well.

Why would the mismatch be worse for frozen features? With fine-tuning, every weight of BERT is adjusted on the task's real sentences, so the model can unlearn any habit tied to `[MASK]`. With frozen features, BERT is used exactly as pre-trained. If its vectors for ordinary, unmasked words are not very informative (because it was never asked to predict an unmasked word), nothing can fix that afterwards.

> [!PAPER] Devlin et al. (2018), BERT · Appendix C.2, Table 8 · page 16
> [![Table 8: ablation over different masking strategies. Masking rates MASK, SAME, RND and Dev set results MNLI fine-tune, NER fine-tune, NER feature-based. 80, 10, 10: 84.2, 95.4, 94.9. 100, 0, 0: 84.3, 94.9, 94.0. 80, 0, 20: 84.1, 95.2, 94.6. 80, 20, 0: 84.4, 95.2, 94.7. 0, 20, 80: 83.7, 94.8, 94.6. 0, 0, 100: 83.6, 94.9, 94.6](/img/papers/bert/p5-table8.png)](/img/papers/bert/p5-table8.png)
> [![The results are presented in Table 8. MASK means that we replace the target token with the [MASK] symbol for MLM; SAME means that we keep the target token as is; RND means that we replace the target token with another random token. The left part of the table shows the probabilities used during MLM pre-training (BERT uses 80%, 10%, 10%). For the feature-based approach, we concatenate the last 4 layers of BERT as the features](/img/papers/bert/p5-c2-legend.png)](/img/papers/bert/p5-c2-legend.png)
>
> **Context:** Table 8 and the paragraph that explains its columns.
>
> **What it says:** for the 15% of tokens chosen for prediction, MASK = replace with `[MASK]`, SAME = keep the token, RND = replace with a random token. Each row is a different split. "BERT uses 80%, 10%, 10%." The feature-based column uses the concatenation of the last four layers, the winner of Table 7.
>
> **Why it matters:** six pre-training recipes, three ways of measuring them.

{{FIG:p5_table8|Table 8 as a picture. Each bar shows what happens to the tokens chosen for prediction. In each results column, the best value is drawn in the accent colour and the worst in red.}}

Row by row:

1. **80 / 10 / 10 (BERT's recipe):** 84.2, 95.4, 94.9. The best NER scores in both columns.
2. **100 / 0 / 0 (always `[MASK]`):** MNLI 84.3 is fine, but feature-based NER drops to **94.0**, the worst value in that column and 0.9 below BERT's recipe. This is the predicted mismatch: the model only ever had to predict at `[MASK]`, and its frozen vectors for normal words are less useful.
3. **80 / 0 / 20:** 84.1, 95.2, 94.6. Random replacement without any "keep" cases.
4. **80 / 20 / 0:** 84.4, 95.2, 94.7. Interestingly the best MNLI score in the table, 0.2 above BERT's recipe.
5. **0 / 20 / 80 (no `[MASK]` at all):** 83.7, 94.8, 94.6.
6. **0 / 0 / 100 (always random):** **83.6** on MNLI, the worst. Here the model is trained to find and fix wrong words, and 15% of every training input is noise.

> [!PAPER] Devlin et al. (2018), BERT · Appendix C.2 · page 16
> [![From the table it can be seen that fine-tuning is surprisingly robust to different masking strategies. However, as expected, using only the MASK strategy was problematic when applying the feature-based approach to NER. Interestingly, using only the RND strategy performs much worse than our strategy as well](/img/papers/bert/p5-c2-result.png)](/img/papers/bert/p5-c2-result.png)
>
> **Context:** the last paragraph of the paper's appendix.
>
> **What it says:** "fine-tuning is surprisingly robust to different masking strategies." But "using only the MASK strategy was problematic when applying the feature-based approach to NER", and "using only the RND strategy performs much worse than our strategy as well".
>
> **Why it matters:** the 80/10/10 recipe is a safe middle ground, mainly protecting frozen-feature use.

Putting numbers on "robust": across all six recipes, MNLI moves within **0.8** points (83.6 to 84.4) and fine-tuned NER within **0.6** (94.8 to 95.4). Feature-based NER moves within **0.9** (94.0 to 94.9), with always-`[MASK]` at the bottom. These are small differences, single numbers without error bars, so the honest summary is: the masking recipe matters little when you fine-tune, and a bit more when you freeze.

One more detail, which the paper does not explain: BERT's own row here scores 84.2 on MNLI, while the BERT-base row of Table 5 scores 84.4. They are probably different training runs, but the paper does not say.

> [!TAKEAWAYS] Key takeaways
> - An **ablation** removes one part, keeps everything else equal, and measures the loss. Section 5 and Appendix C do this for BERT's main design choices.
> - **Bidirectionality is the big one.** Going from the masked LM to left-to-right (Table 5) costs **9.0** points on MRPC and **10.1** F1 on SQuAD. A BiLSTM on top wins back 7.1 on SQuAD but stays 3.6 below BERT-base, and hurts GLUE.
> - **NSP's measured effect is mixed:** 3.5 points on QNLI, but only 0.1 to 0.6 on the other four tasks.
> - The **LTR & No NSP** model has all of BERT's other advantages over GPT (data, batch size, input format, tuned fine-tuning), so its gap to BERT isolates the effect of the pre-training tasks (Appendix A.4).
> - **Bigger is better**, even on small tasks (Table 6). Measured sizes: 45.7M to 335.1M parameters; BERT-base is 109.5M. The two 6-layer models have identical parameter counts, yet 12 heads beat 3.
> - **Frozen BERT features work almost as well as fine-tuning** for NER: concatenating the last four layers gives 96.1 Dev F1 versus 96.4 fine-tuned (Table 7). Our smaller frozen-feature re-run agreed on the shape: embeddings alone worst (82.66), concat last four best (91.17).
> - **Longer pre-training helps** (almost 1.0 MNLI point from 500k to 1M steps), and the masked LM overtakes left-to-right almost immediately despite learning from only 15% of tokens (Figure 5).
> - **Masking recipes barely matter for fine-tuning** (within 0.8 points), but always-`[MASK]` hurts frozen features most (Table 8). 80/10/10 is a safe compromise.

**Next, in Part 6:** what happened after BERT. Later work revisited exactly these ablations (RoBERTa dropped NSP; ALBERT, DistilBERT and ELECTRA changed size and the masked LM itself), BERT's limits, how it is used today, and a summary of the whole paper.

<details>
<summary>Run it yourself</summary>

The scripts behind every measured number in this part:

- [`code/papers/bert/bert_part5.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part5.py): the Table 5 differences, the Table 6 parameter counts, and the WikiText-2 perplexity. About 6 minutes on a laptop CPU.
- [`code/papers/bert/bert_part5_ner.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part5_ner.py): the frozen-feature NER experiment. It took about 28 minutes on an Apple M5 Pro GPU that was shared with other jobs at the time.

```bash
pip install torch transformers datasets
python bert_part5.py        # writes results/part5.json
python bert_part5_ner.py    # writes results/part5_ner.json
```

<figure><img src="/img/papers/bert/part5-run.png" alt="Terminal output of bert_part5.py: Table 5 differences, Table 6 parameter counts matching the formula, and the WikiText-2 masked-LM perplexity of 7.36" loading="lazy" /><figcaption>The real output of bert_part5.py.</figcaption></figure>

<figure><img src="/img/papers/bert/part5-ner-run.png" alt="Terminal output of bert_part5_ner.py: dataset sizes, feature extraction, and Dev F1 for six frozen-feature choices, each averaged over three seeds" loading="lazy" /><figcaption>The real output of bert_part5_ner.py.</figcaption></figure>

</details>

## References

1. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019. Section 5, Appendix A.4, Appendix C.
2. E. F. Tjong Kim Sang, F. De Meulder. [*Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition*](https://aclanthology.org/W03-0419/). CoNLL 2003.
3. M. Peters et al. [*Deep contextualized word representations*](https://arxiv.org/abs/1802.05365) (ELMo). NAACL 2018.
4. R. Al-Rfou, D. Choe, N. Constant, M. Guo, L. Jones. [*Character-Level Language Modeling with Deeper Self-Attention*](https://arxiv.org/abs/1808.04444). 2018.
5. S. Merity, C. Xiong, J. Bradbury, R. Socher. [*Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843) (the WikiText datasets). 2016.
6. [CoNLL-2003 on the Hugging Face hub](https://huggingface.co/datasets/eriktks/conll2003) and [WikiText on the Hugging Face hub](https://huggingface.co/datasets/Salesforce/wikitext).
