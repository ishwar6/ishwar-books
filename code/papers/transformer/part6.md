---
title: "Results, Variations and Impact"
description: "What the Transformer scored, what it cost to train, which parts of it matter, and how it did on a second task. We check the cost and the size numbers by hand and look inside a released translator."
part: 6
covers: "§6.1, Table 2, §6.2, Table 3, §6.3, Table 4, §7, Figures 3 to 5"
date: 2026-10-18
tags: [transformer, attention, machine-translation, bleu, nlp]
---

The first five parts built the Transformer and trained it. This part asks two questions: how well does it work, and what does it cost?

## Translation results {§6.1}

The paper measures translation quality with one number, BLEU. Every result below uses it, so we start there.

> [!DEFINITION] BLEU
> A score from 0 to 100 that compares a machine translation with a human translation of the same sentence. It counts how many short word sequences (of 1, 2, 3 and 4 words) of the machine output also appear in the human one. A higher score means more overlap. A difference of 1 BLEU point is a clear gain in translation research.

Think of BLEU as a teacher who grades a translation with a highlighter. The teacher has the human translation beside the student's text. She marks every single word that also appears in the human text. Then she marks every pair of words, every group of three and every group of four. Long matches are hard to get by luck, so they show that the word order is right too.

$$
\text{BLEU} = \text{BP} \cdot \exp\!\left( \sum_{n=1}^{4} \tfrac{1}{4} \log p_n \right), \qquad \text{BP} = \min\!\left(1,\; e^{\,1 - r/c}\right)
$$

Read it from left to right:

- $$p_n$$ is the **precision** for groups of $$n$$ words: the fraction of the output's $$n$$-word groups that also appear in the human translation.
- The sum of $$\tfrac{1}{4} \log p_n$$ inside $$\exp$$ is the geometric mean of the four precisions. If any one of them is very low, the score falls.
- $$\text{BP}$$ is the **brevity penalty**. $$c$$ is the length of the output and $$r$$ the length of the human translation. An output that is too short loses points, because a short output can reach high precision by saying almost nothing.
- The result is multiplied by 100 for the tables.

A tiny example. The human text is "the cat sat on the mat" and the machine writes "the cat sat on a mat". Count the matches:

| group size $$n$$ | groups in the output | groups that match | $$p_n$$ |
|---|---|---|---|
| 1 | 6 | 5 (all but "a") | 5/6 = 0.833 |
| 2 | 5 | 3 ("the cat", "cat sat", "sat on") | 3/5 = 0.600 |
| 3 | 4 | 2 ("the cat sat", "cat sat on") | 2/4 = 0.500 |
| 4 | 3 | 1 ("the cat sat on") | 1/3 = 0.333 |

Both texts have 6 words, so $$\text{BP} = 1$$. The geometric mean is $$(0.833 \times 0.600 \times 0.500 \times 0.333)^{1/4} = 0.0833^{1/4} = 0.537$$. The score is 53.7. One wrong word took away almost half of the points, because it broke three of the longer groups. Real scores are computed over a whole test set of about 3,000 sentences, not one sentence.

> [!PAPER] · Table 2
> [![Table 2 of the paper: BLEU scores on English-to-German and English-to-French newstest2014 and the training cost in FLOPs of ByteNet, Deep-Att + PosUnk, GNMT + RL, ConvS2S, MoE, three ensembles, and the two Transformer models. The Transformer base and big rows are highlighted](/img/papers/transformer/p6-table2.png)](/img/papers/transformer/p6-table2.png)
>
> **What it says:** "The Transformer achieves better BLEU scores than previous state-of-the-art models on the English-to-German and English-to-French newstest2014 tests at a fraction of the training cost."
>
> **What it means:**
>
> The table has two halves. On the left is quality: BLEU on two test sets. On the right is the price: the number of computer operations that training needed. A good model is high on the left and low on the right. Before this paper, models paid for each BLEU point with more compute.
>
> Read the bottom two rows. The big Transformer gets 28.4 on English-to-German. The best older entry is an **ensemble** of ConvS2S models at 26.36, and it needed $$7.7 \cdot 10^{19}$$ operations. The big Transformer needed $$2.3 \cdot 10^{19}$$, about 3.3 times less, and it is one model, not several.
>
> The base model is the surprise. At 27.3 BLEU it already beats every older entry on English-to-German, ensembles included. Its cost is $$3.3 \cdot 10^{18}$$, the lowest number in the cost column. It is about 23 times cheaper than the ConvS2S ensemble that it beats.

> [!DEFINITION] Ensemble
> Several models trained apart, whose predictions are averaged at test time. An ensemble of 8 models is usually better than one model, but it costs 8 times as much to train and to run. In Table 2, the rows with "Ensemble" are such groups.

> [!PAPER] · §6.1
> [![The first paragraph of Section 6.1, with by more than 2.0, BLEU score of 28.4, 3.5 days on 8 P100 GPUs and Even our base model highlighted](/img/papers/transformer/p6-results-ende.png)](/img/papers/transformer/p6-results-ende.png)
>
> **What it says:** the big model "outperforms the best previously reported models (including ensembles) by more than 2.0 BLEU", with "a new state-of-the-art BLEU score of 28.4". "Training took 3.5 days on 8 P100 GPUs."
>
> **What it means:**
>
> A gain of 2 BLEU points was large. Table 2 shows it: on English-to-German, the steps between the older rows are 0.06 to 0.87 points. The gain here comes with a new kind of network, not with more of the old kind.
>
> The second number matters as much: 3.5 days on one machine with 8 graphics cards. An academic lab could repeat this. The older record holders had used far larger budgets. The last sentence makes the same point for the small model: the base model alone beats them all.

> [!PAPER] · §6.1
> [![The English-to-French paragraph of Section 6.1, with BLEU score of 41.0 and less than 1/4 the training cost highlighted](/img/papers/transformer/p6-results-enfr.png)](/img/papers/transformer/p6-results-enfr.png)
>
> **What it says:** on English-to-French, the big model gets "a BLEU score of 41.0", better than all earlier single models, "at less than 1/4 the training cost of the previous state-of-the-art model".
>
> **What it means:**
>
> English-to-French has more training data: 36 million sentence pairs against 4.5 million for German (Part 5). So all scores are higher, and the race is closer. Here the big Transformer beats every single model, but the ConvS2S ensemble (41.29) stays close.
>
> Look at the numbers with care. The text says 41.0 and Table 2 says 41.8 for the same model. This version of the paper prints both numbers. We cannot tell from the paper which one is the final result, so we quote both. The cost claim holds with either: MoE, the best single model before, needed $$1.2 \cdot 10^{20}$$ operations, and $$2.3 \cdot 10^{19} / 1.2 \cdot 10^{20} = 0.19$$, less than a quarter.
>
> The last sentence changes one setting: dropout 0.1 in place of 0.3 for French. With more data, the model needs less protection against memorising it.

### How the training cost was counted

> [!PAPER] · §6.1
> [![The paragraph that explains how the training cost was estimated, with multiplying the training time, the number of GPUs used highlighted, and footnote 5 with the TFLOPS of K80, K40, M40 and P100 GPUs](/img/papers/transformer/p6-flops.png)](/img/papers/transformer/p6-flops.png)
>
> **What it says:** the number of operations is estimated "by multiplying the training time, the number of GPUs used, and an estimate of the sustained single-precision floating-point capacity of each GPU". Footnote 5 gives 9.5 TFLOPS for the P100.
>
> **What it means:**
>
> This is the bill of a taxi: minutes times the price per minute. The authors do not count the operations of the network. They count the time that the cards were busy and multiply by the speed of a card. For older models in the table, they use the same rule with the speed of the card that each model used.
>
> So the cost column is an estimate with the same method for every row, not an exact count. It assumes that every card works at its full speed all the time, which is never quite true. The method is fair for comparison, because it makes the same error for every model.

> [!DEFINITION] FLOP and TFLOPS
> A FLOP is one floating-point operation: one addition or one multiplication of two decimal numbers. TFLOPS means $$10^{12}$$ such operations each second, the speed of a processor.

$$
\text{FLOPs} = t \times n_{\text{GPU}} \times s
$$

Here $$t$$ is the training time in seconds, $$n_{\text{GPU}}$$ the number of cards, and $$s$$ the speed of one card in operations per second. Part 5 gave the times: 12 hours for the base model and 3.5 days for the big model, both on 8 P100 cards. Our script does the multiplication:

```text
base :  12.0 h x 3600 s x 8 GPUs x 9.5e12 FLOP/s = 3.28e+18 FLOPs
big  :  84.0 h x 3600 s x 8 GPUs x 9.5e12 FLOP/s = 2.30e+19 FLOPs
```

Both numbers match Table 2: $$3.3 \cdot 10^{18}$$ and $$2.3 \cdot 10^{19}$$. The big model costs 7 times as much as the base model: 3 times the steps, and each step more than twice as slow.

### Averaged checkpoints and beam search

> [!PAPER] · §6.1
> [![The paragraph on checkpoint averaging and decoding, with averaging the last 5 checkpoints, beam size of 4, length penalty alpha = 0.6 and input length + 50 highlighted](/img/papers/transformer/p6-averaging.png)](/img/papers/transformer/p6-averaging.png)
>
> **What it says:** "For the base models, we used a single model obtained by averaging the last 5 checkpoints, which were written at 10-minute intervals." The translations use "beam search with a beam size of 4 and length penalty α = 0.6", and the output may be at most "input length + 50" tokens long.
>
> **What it means:**
>
> A **checkpoint** is a copy of all the weights, saved during training. Near the end of training, the weights still move a little from step to step. They are like the needle of a scale that shakes around the true weight. Average five readings of the needle and you get closer to the true value. Here the "readings" are five copies of the network, saved over the last 50 minutes, and the average is taken weight by weight.
>
> A tiny example: one weight has the values 0.50, 0.54, 0.46, 0.52 and 0.48 in the five checkpoints. The averaged model uses 0.50. This costs nothing at test time: the result is still one network of the same size. The big models average the last 20 checkpoints.
>
> The second half of the paragraph is about how the trained model writes a sentence. The next definition explains beam search; the length penalty follows below.

> [!DEFINITION] Beam search
> A way to pick the output sentence one token at a time. **Greedy** decoding keeps only the single best word at each step. Beam search with a beam size of 4 keeps the 4 best partial sentences at each step. It extends each of them and again keeps the best 4. At the end it returns the best complete sentence.

Beam search compares sentences by their log probability: the sum of the logs of the probability of each word. Each word adds a negative number, so a long sentence always has a lower score than a short one. Without a fix, beam search prefers short translations. The length penalty from the GNMT paper divides the score by a factor that grows with the length $$|Y|$$:

$$
\text{score}(Y) = \frac{\log P(Y \mid X)}{\text{lp}(Y)}, \qquad \text{lp}(Y) = \frac{(5 + |Y|)^{\alpha}}{(5 + 1)^{\alpha}}
$$

Here $$X$$ is the source sentence, $$Y$$ a candidate translation, and $$\alpha$$ sets the strength of the penalty. With $$\alpha = 0$$ there is no correction. A tiny example with $$\alpha = 0.6$$: a short candidate of 4 tokens has $$\log P = -2.0$$, and a long one of 8 tokens has $$\log P = -2.4$$. The factors are $$(9/6)^{0.6} = 1.275$$ and $$(13/6)^{0.6} = 1.590$$. The scores become $$-2.0 / 1.275 = -1.568$$ and $$-2.4 / 1.590 = -1.509$$. The long candidate now wins, although its raw log probability was lower.

We ran a released translator with these settings. Helsinki-NLP/opus-mt-en-de is a Marian model with the paper's base shapes: 6 + 6 layers, $$d_{\text{model}} = 512$$, 8 heads, $$d_{ff} = 2048$$ and sinusoidal positions. It was trained on other data with another vocabulary, so it is not the paper's model.

```text
layers 6+6, d_model 512, heads 8, d_ff 2048, vocabulary 58101, parameters 74.4 M
source (15 tokens): The law will never be perfect, but its application should be just.
greedy          : Das Gesetz wird nie perfekt sein, aber seine Anwendung sollte gerecht sein.
beam 4, alpha .6: Das Gesetz wird nie perfekt sein, aber seine Anwendung sollte gerecht sein.
output limit = input length + 50 = 65 tokens; used 15
```

For this sentence, greedy decoding and beam search give the same correct German. That is common for short, clear sentences. Beam search helps on hard sentences, where the best first word is not the best start for the whole sentence. We do not measure BLEU here: we have no test set and no BLEU tool installed, so any score from us would be a guess.

## Model variations {§6.2}

Section 6.2 is an **ablation study**: change one part of the base model, train it again, and see what the change costs.

> [!DEFINITION] Ablation study
> An experiment that changes or removes one part of a system and keeps everything else the same. If the result gets worse, that part was useful. The name comes from surgery, where "ablation" means removing tissue.

> [!PAPER] · Table 3
> [![Table 3 of the paper: variations of the base model in rows A to E and the big model, with columns N, d_model, d_ff, h, d_k, d_v, P_drop, epsilon_ls, train steps, perplexity, BLEU and parameters in millions. The phrase unlisted values are identical to those of the base model is highlighted](/img/papers/transformer/p6-table3.png)](/img/papers/transformer/p6-table3.png)
>
> **What it says:** "Unlisted values are identical to those of the base model. All metrics are on the English-to-German translation development set, newstest2013." The perplexities are "per-wordpiece".
>
> **What it means:**
>
> Read the table like a list of recipes that differ from a base recipe. The first row is the base model with all its settings. Each later row writes only the setting that changed; an empty cell means "same as base". So row (C) with "2" under $$N$$ is the base model with 2 layers in place of 6.
>
> The scores come from a different test set than Table 2: newstest2013, the **development set**. Researchers use it to pick settings, and they keep the 2014 set for the final result. This is why the base model has 25.8 here and 27.3 in Table 2. Also, there is no checkpoint averaging in this table.
>
> The two quality columns are perplexity (lower is better) and BLEU (higher is better). The last column is the size of each model in millions of weights.

> [!DEFINITION] Perplexity (PPL)
> A measure of how unsure a language model is about the next token. It is $$e$$ to the power of the average loss. A perplexity of 4.92 means that the model is as unsure as if it had to choose among about 4.92 equally likely tokens. Lower is better.

The base model has a perplexity of 4.92, so its average loss is $$\ln 4.92 = 1.59$$ per token. The big model has 4.33, a loss of $$\ln 4.33 = 1.47$$. The perplexity is per **wordpiece**, the sub-word units of Part 5, so it cannot be compared with numbers per whole word.

### Counting the parameters by hand

The last column of Table 3 follows from the shapes alone. Each encoder layer has one attention block (four matrices: $$W^Q$$, $$W^K$$, $$W^V$$ and $$W^O$$), one feed-forward network (two matrices) and two layer norms. Each decoder layer has two attention blocks. The embedding matrix is shared by the encoder input, the decoder input and the output layer (Part 2). We count each matrix with its bias, and use the shared vocabulary of 37,000 tokens from Part 5:

$$
\underbrace{4 \cdot 512^2}_{\text{attention}} + \underbrace{2 \cdot 512 \cdot 2048}_{\text{feed-forward}} = 1{,}048{,}576 + 2{,}097{,}152 = 3{,}145{,}728
$$

That is the main part of one encoder layer, without the small biases. Our script adds up all the pieces for each row:

```text
row                             ours (M) paper (M)
base                                63.1        65
(A) h=1, dk=dv=512                  63.1      same
(A) h=16, dk=dv=32                  63.1      same
(B) dk=16                           56.0        58
(C) N=2                             33.7        36
(C) N=8                             77.8        80
(C) d_model=256, dk=dv=32           26.8        28
(C) d_model=1024, dk=dv=128        163.9       168
(C) d_ff=4096                       88.3        90
big                                214.2       213
base split: encoder 18.9 M, decoder 25.2 M, shared embedding 18.9 M
```

Our counts are within about 2 million of the paper's for the base rows, and within 1.2 million for the big model. The paper does not give the exact vocabulary size or how it counted, so a small gap is expected. The shape of the numbers matches well: each change of a row moves our count by the same amount as the paper's.

The split is worth a look. The shared embedding alone is 18.9 million weights, as much as the whole encoder stack. The released Marian model has 74.4 million weights mostly for this reason: its vocabulary of 58,101 tokens adds $$21{,}101 \times 512 = 10.8$$ million weights to the embedding.

### More heads, or bigger heads? {§6.2}

> [!PAPER] · §6.2
> [![The rows (A) paragraph of Section 6.2, with single-head and quality also drops off with too many heads highlighted](/img/papers/transformer/p6-rows-a.png)](/img/papers/transformer/p6-rows-a.png)
>
> **What it says:** rows (A) change the number of heads and the size of each head, "keeping the amount of computation constant". "While single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads."
>
> **What it means:**
>
> Think of a budget of 512 numbers for each token. You can give it to 1 head of 512, 8 heads of 64, or 32 heads of 16. The product $$h \times d_k$$ stays 512, so the work and the weights stay the same. Our count shows it: all rows (A) have 63.1 million weights, and the paper leaves the column empty for them.
>
> The result is a sweet spot. One head gets 24.9 BLEU, eight and sixteen heads get 25.8, and thirty-two heads drop back to 25.4. One head can look at only one mix of places at a time; Part 3 showed why several heads help. But with 32 heads, each head has only 16 numbers for its queries and keys, and that is too few to compare tokens well.

> [!PAPER] · §6.2
> [![The paragraph about rows B to E of Table 3, with reducing the attention key size d_k hurts model quality, bigger models are better, dropout is very helpful and nearly identical highlighted](/img/papers/transformer/p6-rows-bcde.png)](/img/papers/transformer/p6-rows-bcde.png)
>
> **What it says:** "reducing the attention key size $$d_k$$ hurts model quality". In rows (C) and (D), "bigger models are better, and dropout is very helpful in avoiding over-fitting". Learned positional embeddings give "nearly identical results to the base model".
>
> **What it means:**
>
> Rows (B) confirm the lesson of rows (A) from another side. With keys of 16 numbers, BLEU falls to 25.1. The authors read this as a hint that the dot product is a weak judge of "how well do these two tokens match", and that a better match function may help. Later work mostly kept the dot product and made the models bigger.
>
> Rows (C) show the rule that later defined the field: more weights, better scores. Two layers give 23.7 BLEU; $$d_{\text{model}} = 1024$$ gives 26.0; a wider feed-forward network gives 26.2. Rows (D) show the cost of removing regularisation: with no dropout, BLEU falls from 25.8 to 24.6.
>
> Row (E) is the most surprising. The sinusoids of Part 2 can be replaced by position vectors that the model learns, and nothing changes: 25.7 against 25.8. The authors kept the sinusoids because they may work for sentences longer than any seen in training. BERT and GPT later chose the learned version.

## A second task: parsing {§6.3}

Translation is the task the Transformer was designed for. To show that the design is general, the authors try a task with a very different output.

> [!DEFINITION] Constituency parsing
> Splitting a sentence into nested phrases and naming each phrase. For "The cat sat", the parse is a tree: a sentence (S) made of a noun phrase (NP) "The cat" and a verb phrase (VP) "sat". It can be written as a bracketed string: (S (NP The cat) (VP sat)).

> [!PAPER] · §6.3
> [![The first two paragraphs of Section 6.3, with strong structural, significantly longer than the input, 4-layer transformer with d_model = 1024, about 40K training sentences and approximately 17M sentences highlighted](/img/papers/transformer/p6-parsing.png)](/img/papers/transformer/p6-parsing.png)
>
> **What it says:** in parsing, "the output is subject to strong structural constraints and is significantly longer than the input". The authors train "a 4-layer transformer with $$d_{model} = 1024$$" on "about 40K training sentences", and also in a setting with about 17 million sentences.
>
> **What it means:**
>
> The trick is to treat the tree as a sentence in a foreign language. The input is "The cat sat"; the output is "(S (NP The cat) (VP sat))". The same encoder and decoder that translate English to German now translate English to brackets.
>
> This is harder than it looks, for two reasons. First, the output must be well formed: every opened bracket must close, in the right order. One missing bracket breaks the whole tree. Second, the output is much longer than the input. Our three words became 12 symbols if you count each bracket, label and word. The decoder must keep track of deep nesting over long distances.
>
> The data is also small. 40,000 sentences is about 1% of the German translation data. Earlier RNN translation models had failed in this small-data setting.

> [!DEFINITION] F1 score
> One number that combines two: **precision** (what fraction of the predicted phrases are correct) and **recall** (what fraction of the true phrases were found). F1 is their harmonic mean, $$F_1 = 2PR / (P + R)$$. It is high only when both are high.

A tiny example. The true tree of a sentence has 10 phrases. A parser predicts 8 phrases, and 7 of them are correct. Then $$P = 7/8 = 0.875$$ and $$R = 7/10 = 0.700$$. The F1 score is $$2 \times 0.875 \times 0.700 / (0.875 + 0.700) = 1.225 / 1.575 = 0.778$$, or 77.8 in the units of Table 4.

> [!PAPER] · Table 4
> [![Table 4 of the paper: F1 scores on Section 23 of the Wall Street Journal for twelve parsers, with the two Transformer rows highlighted: 91.3 when trained on WSJ only and 92.7 in the semi-supervised setting](/img/papers/transformer/p6-table4.png)](/img/papers/transformer/p6-table4.png)
>
> **What it says:** "The Transformer generalizes well to English constituency parsing." It gets 91.3 F1 with the WSJ data only, and 92.7 in the semi-supervised setting.
>
> **What it means:**
>
> The table has three groups, by training data. In the first group, with only the 40,000 WSJ sentences, the Transformer gets 91.3. Only one parser there scores higher: Dyer et al. with 91.7, a model built for parsing. The RNN translation model of Vinyals and Kaiser gets 88.3 on the same data, 3 points lower.
>
> In the second group, with millions of extra sentences, the Transformer gets 92.7, the best of its group. Two parsers in the last rows score higher. They use other kinds of training: several tasks at once, or a model that generates trees with a grammar.

> [!PAPER] · §6.3
> [![The result paragraphs of Section 6.3, with despite the lack of task-specific tuning and even when training only on the WSJ training set of 40K sentences highlighted](/img/papers/transformer/p6-parsing-result.png)](/img/papers/transformer/p6-parsing-result.png)
>
> **What it says:** "despite the lack of task-specific tuning our model performs surprisingly well". Unlike RNN sequence-to-sequence models, "the Transformer outperforms the BerkeleyParser even when training only on the WSJ training set of 40K sentences."
>
> **What it means:**
>
> The key phrase is "lack of task-specific tuning". The authors tried only a few settings: dropout, learning rate and beam size. Everything else stayed as in the English-to-German base model. A network designed for translation, moved almost unchanged to a new task, came close to the best special-purpose parsers.
>
> This is the first sign in the paper of the idea that would define the next years: one architecture for many tasks. The authors do not say it in these words. But a model that translates and parses with the same parts is a general tool, not a translation trick.

## The conclusion {§7}

> [!PAPER] · §7
> [![Section 7 of the paper, with the first sequence transduction model based entirely on attention, trained significantly faster, other than text, images, audio and video, and Making generation less sequential highlighted](/img/papers/transformer/p6-conclusion.png)](/img/papers/transformer/p6-conclusion.png)
>
> **What it says:** the Transformer is "the first sequence transduction model based entirely on attention". It "can be trained significantly faster" than recurrent or convolutional models. The authors plan to apply it to "modalities other than text", such as "images, audio and video", to study local, restricted attention for large inputs, and to make "generation less sequential".
>
> **What it means:**
>
> The first paragraph repeats the claim of the title: no recurrence, only attention. The second gives the evidence in two words: faster and better. Parts 4 and 5 showed why it is faster. Every position of a sentence is computed at the same time, so the graphics cards stay busy.
>
> The third paragraph is a plan, and almost every item came true. "Images" became the Vision Transformer. "Local, restricted attention" became the windowed attention of later image models. "Audio" became speech models that are Transformers. "Less sequential generation" is the one item that stayed hard: the decoder still writes one token at a time, as in Part 3.

## Inside the heads: the appendix {§Appendix}

The appendix has no text, only three figures. Each shows the attention of one head in layer 5 of the 6 encoder layers. They answer a question that the numbers cannot: what do the heads actually look at?

> [!PAPER] · Figure 3
> [![Figure 3 of the paper: the sentence It is in this spirit that a majority of American governments have passed new laws since 2009 making the registration or voting process more difficult, written twice. Coloured lines go from the word making to more and difficult. The caption says that many heads attend to a distant dependency of the verb making, completing the phrase making more difficult](/img/papers/transformer/p6-figure3.png)](/img/papers/transformer/p6-figure3.png)
>
> **What it says:** "Many of the attention heads attend to a distant dependency of the verb 'making', completing the phrase 'making...more difficult'." Only the attention of the word "making" is shown, and each colour is one head.
>
> **What it means:**
>
> Read the figure as two copies of the same sentence, one above the other. A line goes from a word on top to a word below when the top word looks at the bottom word. The thicker the line, the larger the attention weight. All the thick lines from "making" end on "more" and "difficult".
>
> The point is the grammar. "Making ... more difficult" is one phrase, cut apart by four words. To translate "making", the model must know what is being made, and how. An RNN would have to carry "making" through four steps to meet "difficult". Here, one head connects the two words in one step, at any distance.

We gave the same sentence to the released Marian model and printed where "making" looks in each head of encoder layer 5:

```text
27 tokens; query token "▁making" at position 17
  head 1: ▁making 0.60, ▁difficult 0.19, ▁passed 0.07   | weight on tokens 5 or more away: 0.34
  head 2: ▁laws 0.18, </s> 0.13, ▁passed 0.12   | weight on tokens 5 or more away: 0.45
  head 3: ▁difficult 0.71, ▁more 0.29, ▁process 0.00   | weight on tokens 5 or more away: 1.00
  head 4: ▁making 0.19, </s> 0.14, ▁passed 0.08   | weight on tokens 5 or more away: 0.47
  head 5: ▁more 0.29, . 0.18, </s> 0.13   | weight on tokens 5 or more away: 0.77
  head 6: ▁difficult 0.69, ▁more 0.12, </s> 0.04   | weight on tokens 5 or more away: 0.91
  head 7: ▁making 0.99, ▁difficult 0.01, ▁the 0.00   | weight on tokens 5 or more away: 0.01
  head 8: </s> 0.36, . 0.22, ▁making 0.07   | weight on tokens 5 or more away: 0.77
```

{{FIG:p6_making|Attention from the word "making" in the 8 heads of encoder layer 5 of the released Marian model. Darker cells mean more weight. Head 3 puts 0.71 on "difficult" and 0.29 on "more"; head 6 puts 0.69 on "difficult". Head 7 looks only at "making" itself.}}

The result is close to Figure 3. Head 3 puts all its weight on "difficult" (0.71) and "more" (0.29), five and six tokens away. Head 6 does almost the same. This is a different model, trained by other people on other data, and it found the same link. Other heads do other things. Head 7 looks only at the word itself, and heads 2 and 8 spread their weight or rest on the end token.

> [!PAPER] · Figure 4
> [![Figure 4 of the paper: the sentence The Law will never be perfect, but its application should be just, this is what we are missing, in my opinion. Top: all the attention lines of head 5. Bottom: the attention of the word its alone in heads 5 and 6, with sharp lines to Law and application](/img/papers/transformer/p6-figure4.png)](/img/papers/transformer/p6-figure4.png)
>
> **What it says:** two heads are "apparently involved in anaphora resolution". For the word "its", "the attentions are very sharp".
>
> **What it means:**
>
> In the sentence "The Law will never be perfect, but its application should be just", the word "its" points back to "Law". Finding the word that a pronoun refers to is called **anaphora resolution**. A translator needs it: in German, "its" becomes "seine" because "Gesetz" (law) is a neuter noun.
>
> In the bottom picture, nearly all the attention of "its" goes to two words: "Law" and "application". One tells what "its" refers to; the other tells what it owns. The word "apparently" in the caption is honest. A picture of one sentence suggests what a head does. It does not prove it.

```text
26 tokens; query token "▁its" at position 8
  head 1: ▁its 0.82, ▁The 0.14, ▁Law 0.02   | weight on tokens 5 or more away: 0.17
  head 2: ▁but 0.32, , 0.27, </s> 0.12   | weight on tokens 5 or more away: 0.22
  head 3: ▁application 0.88, </s> 0.04, ▁just 0.03   | weight on tokens 5 or more away: 0.05
  head 4: </s> 0.27, ▁its 0.14, ▁be 0.09   | weight on tokens 5 or more away: 0.60
  head 5: ▁be 0.46, ▁- 0.23, ▁should 0.11   | weight on tokens 5 or more away: 0.32
  head 6: ▁its 0.29, </s> 0.27, ▁application 0.27   | weight on tokens 5 or more away: 0.34
  head 7: ▁its 0.55, ▁be 0.17, ▁is 0.10   | weight on tokens 5 or more away: 0.21
  head 8: </s> 0.34, ▁but 0.14, , 0.07   | weight on tokens 5 or more away: 0.50
```

{{FIG:p6_its|Attention from the word "its" in the 8 heads of encoder layer 5 of the released Marian model. Head 3 puts 0.88 on "application". No head of this layer puts much weight on "Law": the largest is 0.02, in head 1.}}

Here our model does not repeat the paper. Head 3 puts 0.88 on "application", the word that "its" owns, as one head of the paper does. But no head in this layer puts much weight on "Law": the largest value is 0.02. The Marian model still translates "its" as "seine", the correct form, so it must find the link somewhere. It may do so in another layer, or in the decoder. We looked at one layer of one model, so the comparison shows what is possible, not a rule.

> [!PAPER] · Figure 5
> [![Figure 5 of the paper: the same Law sentence twice, with the full attention of two different heads of layer 5, one drawn in green and one in red. The patterns of the two heads differ](/img/papers/transformer/p6-figure5.png)](/img/papers/transformer/p6-figure5.png)
>
> **What it says:** "Many of the attention heads exhibit behaviour that seems related to the structure of the sentence." "The heads clearly learned to perform different tasks."
>
> **What it means:**
>
> Compare the two pictures. In the red head, most words send one strong line to a word close by, often the next word or the one before. It works like a reader who checks the neighbours of each word. In the green head, many lines cross the sentence and gather on a few words, such as "missing" and "application". That head collects information from far away.
>
> No one told the heads to split the work in this way. The training only asked for good translations. The division of labour came from the training itself, as Part 3 suggested when it explained why multi-head attention helps. Our Marian heads show the same variety. One head looks only at the word itself, one jumps five words ahead, and one rests on the end token.

> [!DEFINITION] Anaphora
> A word that points back to an earlier word for its meaning, like "its", "she" or "this". The earlier word is the **antecedent**. In "The Law ... its application", "its" is the anaphor and "Law" is the antecedent.

## The series in one page, and what came after

This series read the paper from its title to its appendix. The [Background](background-the-road-to-the-transformer.md) showed the problem: RNNs read a sentence one word at a time, and attention was only a helper. [Part 1](part-1-the-big-idea.md) gave the idea: drop the recurrence and keep only attention.

[Part 2](part-2-the-architecture.md) built the stacks, the feed-forward networks, the embeddings and the sine-wave positions. [Part 3](part-3-attention.md) opened the attention equation and its heads. [Part 4](part-4-why-self-attention.md) showed why attention wins: few sequential steps and short paths between any two words. [Part 5](part-5-training.md) trained the model in 12 hours. This part showed the scores, the cost and the ablations.

The impact came fast and went far beyond translation:

- **Encoder only.** BERT (2018) kept only the encoder stack and trained it to fill in hidden words. It became the standard model for understanding text. The [BERT series](/papers/bert/) explains it.
- **Decoder only.** GPT (2018) kept only the decoder stack with its causal mask, and trained it to predict the next token. Its descendants are today's large language models. They still use the same block: attention, feed-forward network, residual connections and layer norm.
- **Other inputs.** The Vision Transformer (2020) cut images into patches and treated each patch as a word, as §7 had planned. The [ViT series](/papers/vit/) explains it. The [SAM series](/papers/sam/) shows a large image model built on it, with the "local, restricted attention" of §7 in the form of windows.
- **Bigger.** Rows (C) of Table 3 said "bigger models are better". Later work turned that remark into a rule: scale the weights, the data and the compute together, and the scores improve in a predictable way.

The base model of this paper has 65 million weights. Its block design is still at the core of the largest models in use today.

> [!TAKEAWAYS] Key takeaways
> - **BLEU counts shared word groups.** It is the geometric mean of the precisions of 1- to 4-word groups, times a penalty for short outputs. In our toy example, one wrong word cut the score from 100 to 53.7.
> - **Better and cheaper at once.** The big model got 28.4 BLEU on English-to-German, more than 2 points above every earlier result, ensembles included. The base model (27.3) beat them all at $$3.3 \cdot 10^{18}$$ operations.
> - **The cost is time × cards × speed.** 12 hours × 8 cards × 9.5 TFLOPS gives $$3.28 \cdot 10^{18}$$; 3.5 days gives $$2.30 \cdot 10^{19}$$. Both match Table 2.
> - **English-to-French: 41.0 or 41.8.** The text and Table 2 disagree. Either way, the cost is less than a quarter of the best earlier single model.
> - **Decoding settings matter.** The last 5 or 20 checkpoints are averaged, and beam search uses 4 beams and a length penalty with $$\alpha = 0.6$$.
> - **Eight heads is a sweet spot.** With the same budget of 512 numbers, 1 head loses 0.9 BLEU and 32 heads lose 0.4. Keys that are too small hurt.
> - **Bigger is better, dropout helps, and learned positions work as well as sinusoids.** 2 layers give 23.7 BLEU; no dropout gives 24.6; learned positions give 25.7 against 25.8.
> - **The size follows from the shapes.** Our count from the shapes gives 63.1 million weights for the base model (paper: 65) and 214.2 million for the big model (paper: 213).
> - **One design, a second task.** With little tuning, a 4-layer Transformer parsed English at 91.3 F1 from 40,000 sentences, close to parsers built for the task.
> - **Heads specialise.** In a released Marian model, one head links "making" to "more difficult" with all its weight, as in Figure 3. The "its" to "Law" link of Figure 4 did not show in the same layer.

<figure><img src="/img/papers/transformer/part6-run.png" alt="Terminal output of tx_part6.py: the training cost in FLOPs, the parameter counts of the Table 3 rows, the Marian translation with greedy and beam search, and the attention of the words making and its in encoder layer 5" loading="lazy" /><figcaption>The output of tx_part6.py, as it ran for this article.</figcaption></figure>

## The paper and the code

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017) · [official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
