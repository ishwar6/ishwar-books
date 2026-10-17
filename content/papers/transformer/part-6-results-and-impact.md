---
title: "Results, Variations and Impact"
description: "What the Transformer scored, what it cost to train, which parts of it matter, and how it did on a second task. We write BLEU, beam search, checkpoint averaging and the size count in Python, and check each one against a library."
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

### BLEU in code

**Method 1: from scratch.** BLEU needs four small functions. The first counts word groups. The second compares one output with its reference. The third turns the counts into a score, and the fourth does it for a whole test set.

```python
def ngrams(words, n):
    """Count every group of n words that follow each other."""
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))


def bleu_stats(hyp, ref, max_n=4):
    """For one sentence: matches and totals for n = 1..4, and the two lengths."""
    hyp, ref = hyp.split(), ref.split()
    matches, totals = [], []
    for n in range(1, max_n + 1):
        h, r = ngrams(hyp, n), ngrams(ref, n)
        # clipping: a group counts at most as often as it occurs in the reference
        matches.append(sum(min(count, r[g]) for g, count in h.items()))
        totals.append(max(len(hyp) - n + 1, 0))
    return matches, totals, len(hyp), len(ref)


def bleu_from_stats(matches, totals, c, r):
    if min(matches) == 0:
        return 0.0                                   # log(0): one empty precision makes the score 0
    log_mean = sum(math.log(m / t) for m, t in zip(matches, totals)) / len(matches)   # geometric mean, in logs
    bp = 1.0 if c > r else math.exp(1 - r / c)       # brevity penalty: only short outputs lose points
    return 100 * bp * math.exp(log_mean)


def corpus_bleu(hyps, refs):
    """Corpus BLEU adds up the counts of all sentences first, then computes one score."""
    M, T, C, Rl = [0] * 4, [0] * 4, 0, 0
    for h, r in zip(hyps, refs):
        m, t, c, rl = bleu_stats(h, r)
        M = [a + b for a, b in zip(M, m)]
        T = [a + b for a, b in zip(T, t)]
        C, Rl = C + c, Rl + rl
    return bleu_from_stats(M, T, C, Rl)
```

Read `ngrams` first. For "the cat sat" and $$n = 2$$ it slides a window of two words along the sentence and counts ("the", "cat") and ("cat", "sat"). A `Counter` is a dictionary that counts, so a group that occurs twice gets the value 2.

`bleu_stats` holds the one idea of BLEU that the formula hides: **clipping**. Without it, the output "the the the the the the" would have six matching words, because "the" is in the reference. With clipping, "the" counts at most as often as the reference has it, which is twice. The line `min(count, r[g])` does this. `totals` is the number of groups in the output. A sentence of 6 words has 6 single words, 5 pairs, 4 triples and 3 groups of four.

`bleu_from_stats` follows the formula. It takes the mean of the four log precisions, which is the log of their geometric mean. The brevity penalty is 1 when the output is longer than the reference, and $$e^{1 - r/c}$$ when it is shorter. If any precision is 0, its log is minus infinity and the score is 0.

**Method 2: with the library.** The standard tool is sacreBLEU. It fixes the tokenization and the details, so that teams can compare scores. It is not in our Python environment, so we installed it into a separate folder only for this check. We ran both on four outputs against "the cat sat on the mat":

```text
sacrebleu: 2.6.0
"the cat sat on a mat": p1..p4 = 5/6 3/5 2/4 1/3, c = 6, r = 6, BLEU ours 53.73, sacrebleu 53.73
"the cat sat on the mat": p1..p4 = 6/6 5/5 4/4 3/3, c = 6, r = 6, BLEU ours 100.00, sacrebleu 100.00
"the cat sat on": p1..p4 = 4/4 3/3 2/2 1/1, c = 4, r = 6, BLEU ours 60.65, sacrebleu 60.65
"the the the the the the": p1..p4 = 2/6 0/5 0/4 0/3, c = 6, r = 6, BLEU ours 0.00, sacrebleu 0.00
sacrebleu with its default smoothing on the last one: 9.65
unclipped: "the" counted 6 times; clipped: min(6, 2) = 2
two sentences: sentence BLEU 53.73 and 41.11 (mean 47.42); corpus BLEU ours 46.27, sacrebleu 46.27
```

Our 53.73 is the 53.7 of the table above, and sacreBLEU agrees on every line. The third output shows the brevity penalty at work. "the cat sat on" has perfect precisions, but it has 4 words where the reference has 6. So $$\text{BP} = e^{1 - 6/4} = 0.607$$, and the score is 60.65, not 100. The fourth output gets only 2 matches out of 6 words after clipping, and no matching pairs, so its score is 0.

Two details matter for real use. First, sacreBLEU's `sentence_bleu` smooths empty precisions by default, so the hopeless fourth output gets 9.65 instead of 0. We turned that smoothing off for the comparison. Second, corpus BLEU is not the mean of the sentence scores. It adds up the matches and totals of all sentences, then computes one score: 46.27 for our two sentences, not their mean of 47.42. The scores in the paper's tables are corpus scores of this kind.

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

For this sentence, greedy decoding and beam search give the same correct German. That is common for short, clear sentences. Beam search helps on hard sentences, where the best first word is not the best start for the whole sentence. We do not measure BLEU on the model here: we have no WMT test set, so any score from us would be a guess.

### Checkpoint averaging in code

Checkpoint averaging is the shortest algorithm on this page. A checkpoint is a dictionary from weight names to tensors, so the average is one dictionary of means.

**Method 1: from scratch.**

```python
def average_by_hand(state_dicts):
    """Weight by weight: the mean of the same tensor over all checkpoints."""
    return {k: sum(sd[k] for sd in state_dicts) / len(state_dicts) for k in state_dicts[0]}
```

To see it work, we need weights that still move at the end of training. We trained a tiny network (8 inputs, 32 hidden units, 1 output) on a noisy curve, with plain gradient descent and a large step. We saved a checkpoint every 100 steps, our stand-in for the paper's "10-minute intervals", and averaged the last 5.

**Method 2: with the library.** PyTorch has `torch.optim.swa_utils.AveragedModel`, which keeps a running mean of the weights it is given. "SWA" means stochastic weight averaging, a later method built on the same idea.

```python
last5 = checkpoints[-5:]
avg = average_by_hand(last5)
swa = AveragedModel(net)                  # Method 2: the library keeps a running mean
for sd in last5:
    net.load_state_dict(sd)
    swa.update_parameters(net)
```

```text
one weight in the last 5 checkpoints: -0.4073, -0.3782, -0.4102, -0.4517, -0.4189  -> mean -0.4133
validation loss of the last 5 checkpoints: 0.8701, 0.9201, 0.9074, 0.8006, 0.8326
validation loss of their average: 0.7449
average by hand vs AveragedModel: max abs difference 1.2e-07
```

The first line is the needle of the scale from the box above. One weight jumps between −0.45 and −0.38, and the average sits in the middle. The validation loss is measured on new data that the network never trained on. Every single checkpoint scores between 0.80 and 0.92, and their average scores 0.7449, better than all five. The noise of the last steps pushes each checkpoint a little off in a random direction, and the average cancels much of it. The library's running mean agrees with ours to 0.00000012, which is rounding.

One condition hides in this trick. The five checkpoints must be close to each other, in the same valley of the loss. Averaging two networks trained from different random starts usually gives a poor network, because their hidden units can be in a different order. The paper averages checkpoints only 10 minutes apart, so the condition holds.

### Greedy decoding in code

`model.generate` hides the decoding loop. Here we write it ourselves around the released Marian model. We use the model only for one thing: given the source and the tokens so far, give the log probability of every possible next token.

```python
@torch.no_grad()
def encode(text):
    x = tok([text], return_tensors='pt')
    memory = model.get_encoder()(**x).last_hidden_state     # (1, n, 512), computed once
    return x, memory


@torch.no_grad()
def next_log_probs(x, memory, prefixes):
    """log P(next token | source, prefix) for each prefix. prefixes: (beams, t) -> (beams, V)."""
    b = prefixes.shape[0]
    out = model(encoder_outputs=BaseModelOutput(last_hidden_state=memory.expand(b, -1, -1)),
                attention_mask=x['attention_mask'].expand(b, -1), decoder_input_ids=prefixes)
    logits = out.logits[:, -1]                    # scores for the token after the prefix
    logits[:, PAD] = -float('inf')                # the padding token is never a word
    return torch.log_softmax(logits, dim=-1)
```

The encoder runs once per sentence; its output is the "memory" that every decoder step reads. The decoder runs again at each step on the whole prefix. A faster program would keep the keys and values of earlier steps in a cache, but the result is the same. Marian's settings forbid the padding token as an output, so we set its score to minus infinity, as the library does.

**Method 1: from scratch.** Greedy decoding takes the best token, appends it, and repeats until the end token or the paper's limit of input length + 50.

```python
def greedy_decode(text, extra=50):
    x, memory = encode(text)
    ys = [START]
    for _ in range(x['input_ids'].shape[1] + extra):           # the paper's limit: input length + 50
        lp = next_log_probs(x, memory, torch.tensor([ys]))[0]
        best = int(lp.argmax())                                # the single best next token
        ys.append(best)
        if best == EOS:
            break
    return ys
```

**Method 2: with the library.** `model.generate(**x, num_beams=1, do_sample=False, max_new_tokens=n + 50)` is greedy decoding. We compared the token ids, not only the text, on two sentences:

```text
The law will never be perfect, but its application should be just.
  ours   : Das Gesetz wird nie perfekt sein, aber seine Anwendung sollte gerecht sein.
  library: same token ids: True
We trained the model on eight GPUs for twelve hours, and it still beat the ensembles.
  ours   : Wir trainierten das Modell zwölf Stunden lang an acht GPUs, und es schlug die Ensembles noch.
  library: same token ids: True
```

Our loop and the library choose the same tokens. Look at the end of the second translation: "schlug die Ensembles noch". A German reader expects "immer noch" (still) here; "noch" alone is less clear. Greedy decoding made one locally good choice and could not go back. Beam search can.

### Beam search in code

**Method 1: from scratch.** Beam search keeps $$k$$ open sentences, here 4. At each step it extends every open sentence by every token, adds the log probabilities, and keeps the best. A sentence that ends with the end token moves to a list of finished sentences. There, its score is divided by the length penalty.

```python
def lp_hf(n, alpha):
    return n ** alpha                             # the library: length to the power alpha


def lp_gnmt(n, alpha):
    return ((5 + n) / 6) ** alpha                 # the paper (from GNMT): ((5 + |Y|) / 6) ^ alpha


def beam_search(text, k=4, alpha=0.6, lp=lp_hf, extra=50):
    x, memory = encode(text)
    max_new = x['input_ids'].shape[1] + extra
    running = [([START], 0.0)]                    # (tokens, sum of log probabilities)
    finished = []                                 # (tokens, score after the length penalty)
    for t in range(1, max_new + 1):
        lp_next = next_log_probs(x, memory, torch.tensor([ys for ys, _ in running]))   # (beams, V)
        total = torch.tensor([s for _, s in running])[:, None] + lp_next               # score of every extension
        top_scores, top_idx = total.flatten().topk(2 * k)     # keep 2k, in case some of them end here
        V = lp_next.shape[1]
        new_running = []
        for rank, (score, idx) in enumerate(zip(top_scores.tolist(), top_idx.tolist())):
            beam, token = divmod(idx, V)
            ys = running[beam][0] + [token]
            if token == EOS:
                if rank < k:                      # only a top-k candidate may finish
                    finished.append((ys, score / lp(t, alpha)))
            elif len(new_running) < k:
                new_running.append((ys, score))
        finished = sorted(finished, key=lambda f: -f[1])[:k]
        running = new_running
        # stop when k sentences are finished and the best open one cannot beat the worst finished one
        if len(finished) == k and running[0][1] / lp(t, alpha) <= finished[-1][1]:
            break
    return finished[0]
```

Walk through one step. `total` is a table with one row per open sentence and one column per token: 4 × 58,101 scores. `flatten().topk(2 * k)` takes the best 8 of all of them at once, so two candidates may come from the same sentence. `divmod(idx, V)` turns a position in the flat list back into (sentence, token). We take 8 and not 4 because some of the best may be end tokens. Those leave the beam, and we still want 4 open sentences for the next step.

The stop rule needs care. A finished sentence can still lose to an open one that ends later with a better score. So the search stops only when 4 sentences are finished, and the best open sentence cannot beat the worst finished one. The open sentence is scored as if it ended now. This is the library's rule, and we copied it to get the same output.

There are two length penalties in the code. The paper uses GNMT's $$((5 + n)/6)^{\alpha}$$, the formula above. Hugging Face's `length_penalty=0.6` uses $$n^{\alpha}$$ instead, where $$n$$ counts the new tokens including the end token. Both grow with the length, so both work against short outputs, but they give different numbers.

**Method 2: with the library.** `model.generate(**x, num_beams=4, length_penalty=0.6, max_new_tokens=n + 50)`. We compared the tokens and the final score, then ran our search again with the paper's penalty:

```text
The law will never be perfect, but its application should be just.
  ours   : Das Gesetz wird nie perfekt sein, aber seine Anwendung sollte gerecht sein.  (score -0.7540)
  library: same token ids: True, score -0.7540
  with the paper's penalty ((5 + n) / 6)^0.6: Das Gesetz wird nie perfekt sein, aber seine Anwendung sollte gerecht sein.  (score -1.8590)
We trained the model on eight GPUs for twelve hours, and it still beat the ensembles.
  ours   : Wir trainierten das Modell zwölf Stunden lang an acht GPUs, und es schlug die Ensembles immer noch.  (score -1.6315)
  library: same token ids: True, score -1.6315
  with the paper's penalty ((5 + n) / 6)^0.6: Wir trainierten das Modell zwölf Stunden lang an acht GPUs, und es schlug die Ensembles immer noch.  (score -4.2675)
length 4, log P -2.0: GNMT factor 1.275 -> -1.568;  library factor 2.297 -> -0.871
length 8, log P -2.4: GNMT factor 1.590 -> -1.509;  library factor 3.482 -> -0.689
```

Our beam search gives the library's tokens and the library's score to four decimals. On the second sentence, it finds "immer noch", the better German that greedy decoding missed. The paper's penalty picks the same translations here, with other score values. The last two lines redo the worked example above with both penalties. Both prefer the longer candidate, and the library's factor grows faster.

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

### The count in code

**Method 1: from scratch.** The count is a function of the shapes in Table 3's columns. Each line of it names one matrix or bias of the network. Note $$d_k$$ and $$d_v$$. The query and key projections make $$h$$ heads of $$d_k$$ numbers each. $$W^O$$ takes the $$h \times d_v$$ joined head outputs back to $$d_{\text{model}}$$.

```python
def count_params(N=6, d_model=512, d_ff=2048, h=8, d_k=None, d_v=None, vocab=37000):
    d_k = d_k or d_model // h
    d_v = d_v or d_model // h
    attention = (d_model * h * d_k + h * d_k        # W_Q and its bias: d_model -> h heads of d_k
                 + d_model * h * d_k + h * d_k      # W_K
                 + d_model * h * d_v + h * d_v      # W_V
                 + h * d_v * d_model + d_model)     # W_O: the joined heads -> d_model
    ffn = d_model * d_ff + d_ff + d_ff * d_model + d_model   # two linear layers with biases
    norm = 2 * d_model                                        # LayerNorm: a gain and a bias
    encoder_layer = attention + ffn + 2 * norm                # self-attention, FFN, 2 norms
    decoder_layer = 2 * attention + ffn + 3 * norm            # + attention over the encoder, 3 norms
    embedding = vocab * d_model                               # one matrix, shared 3 ways (Part 2)
    return N * encoder_layer + N * decoder_layer + embedding
```

**Method 2: with the library.** PyTorch has ready-made layers with the paper's structure: `nn.TransformerEncoderLayer` and `nn.TransformerDecoderLayer`. We build $$N$$ of each with the shapes of a row and let PyTorch count. The `meta` device creates the shapes without real numbers, so even the big model takes no memory.

```python
def count_with_torch(N=6, d_model=512, d_ff=2048, h=8, vocab=37000):
    """The same shapes built from PyTorch layers on the 'meta' device: no memory, only shapes."""
    with torch.device('meta'):
        enc = nn.ModuleList([nn.TransformerEncoderLayer(d_model, h, d_ff) for _ in range(N)])
        dec = nn.ModuleList([nn.TransformerDecoderLayer(d_model, h, d_ff) for _ in range(N)])
        emb = nn.Embedding(vocab, d_model)
    return sum(p.numel() for m in (enc, dec, emb) for p in m.parameters())
```

```text
row                  shapes (M) PyTorch (M) paper (M)
base                      63.08       63.08 =        65
(A) h=1, d_k=512          63.08       63.08 =         -
(A) h=4, d_k=128          63.08       63.08 =         -
(A) h=16, d_k=32          63.08       63.08 =         -
(A) h=32, d_k=16          63.08       63.08 =         -
(B) d_k=16                55.99    (cannot)        58
(B) d_k=32                58.35    (cannot)        60
(C) N=2                   33.66       33.66 =        36
(C) N=4                   48.37       48.37 =        50
(C) N=8                   77.80       77.80 =        80
(C) d_model=256           26.83       26.83 =        28
(C) d_model=1024         163.89      163.89 =       168
(C) d_ff=1024             50.49       50.49 =        53
(C) d_ff=4096             88.27       88.27 =        90
big                      214.25      214.25 =       213
Marian: count_params(vocab=58,101) = 73,886,208;  sum of the model's own tensors = 74,410,496;  difference 524,288
  the difference is the two stored tables of sinusoidal positions, 2 x 512 x 512 = 524,288
```

Our function and PyTorch agree to the last weight on every row they can both build ("="). PyTorch's attention cannot build rows (B). It always splits $$d_{\text{model}}$$ into $$h$$ equal heads, so $$d_k = d_{\text{model}}/h = 64$$; a key size of 16 or 32 with 8 heads needs our own code. This table also has the rows that the first one left out. Each follows the same pattern: our count is 1 to 4 million below the paper's, except for the big model.

The last two lines test the function on a real network. With Marian's vocabulary, it gives 73,886,208 weights. Marian itself holds 524,288 more, exactly $$2 \times 512 \times 512$$. Marian stores its sine-wave positions as two tables of 512 positions × 512 numbers, one for the encoder and one for the decoder. The paper computes those values with a formula, so our function does not count them. Everything else matches to the last weight.

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

<figure class="fig"><svg viewBox="0 0 640 304" role="img" aria-label="Attention weights from the word making to every token of the sentence, for the 8 heads of encoder layer 5 of the released Marian English-to-German model."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="350.0" y="22.0" text-anchor="middle">Where "making" looks, encoder layer 5 of 6</text><text class="t-tick" x="72.0" y="48.0" text-anchor="end">head 1</text><g class="mark"><title>head 1 → It: 0.00</title><rect class="cell" x="81.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → is: 0.00</title><rect class="cell" x="101.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → in: 0.00</title><rect class="cell" x="121.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → this: 0.00</title><rect class="cell" x="141.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → spirit: 0.00</title><rect class="cell" x="161.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → that: 0.00</title><rect class="cell" x="181.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → a: 0.00</title><rect class="cell" x="201.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → majority: 0.00</title><rect class="cell" x="221.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → of: 0.00</title><rect class="cell" x="241.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → American: 0.00</title><rect class="cell" x="261.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → governments: 0.00</title><rect class="cell" x="281.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → have: 0.01</title><rect class="cell" x="301.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → passed: 0.07</title><rect class="cell" x="321.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.069"/></g><g class="mark"><title>head 1 → new: 0.03</title><rect class="cell" x="341.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → laws: 0.00</title><rect class="cell" x="361.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → since: 0.00</title><rect class="cell" x="381.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → 2009: 0.00</title><rect class="cell" x="401.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → making: 0.60</title><rect class="cell" x="421.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.599"/></g><g class="mark"><title>head 1 → the: 0.01</title><rect class="cell" x="441.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → registration: 0.02</title><rect class="cell" x="461.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → or: 0.00</title><rect class="cell" x="481.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → voting: 0.00</title><rect class="cell" x="501.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → process: 0.01</title><rect class="cell" x="521.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → more: 0.05</title><rect class="cell" x="541.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.053"/></g><g class="mark"><title>head 1 → difficult: 0.19</title><rect class="cell" x="561.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.186"/></g><g class="mark"><title>head 1 → .: 0.00</title><rect class="cell" x="581.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → &lt;/s&gt;: 0.00</title><rect class="cell" x="601.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="68.0" text-anchor="end">head 2</text><g class="mark"><title>head 2 → It: 0.01</title><rect class="cell" x="81.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → is: 0.01</title><rect class="cell" x="101.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → in: 0.00</title><rect class="cell" x="121.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → this: 0.00</title><rect class="cell" x="141.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → spirit: 0.00</title><rect class="cell" x="161.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → that: 0.04</title><rect class="cell" x="181.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.042"/></g><g class="mark"><title>head 2 → a: 0.02</title><rect class="cell" x="201.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → majority: 0.01</title><rect class="cell" x="221.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → of: 0.00</title><rect class="cell" x="241.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → American: 0.00</title><rect class="cell" x="261.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → governments: 0.02</title><rect class="cell" x="281.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → have: 0.02</title><rect class="cell" x="301.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → passed: 0.12</title><rect class="cell" x="321.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.122"/></g><g class="mark"><title>head 2 → new: 0.05</title><rect class="cell" x="341.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.048"/></g><g class="mark"><title>head 2 → laws: 0.18</title><rect class="cell" x="361.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.180"/></g><g class="mark"><title>head 2 → since: 0.04</title><rect class="cell" x="381.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.042"/></g><g class="mark"><title>head 2 → 2009: 0.10</title><rect class="cell" x="401.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.101"/></g><g class="mark"><title>head 2 → making: 0.08</title><rect class="cell" x="421.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.080"/></g><g class="mark"><title>head 2 → the: 0.05</title><rect class="cell" x="441.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.050"/></g><g class="mark"><title>head 2 → registration: 0.03</title><rect class="cell" x="461.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → or: 0.00</title><rect class="cell" x="481.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → voting: 0.02</title><rect class="cell" x="501.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → process: 0.00</title><rect class="cell" x="521.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → more: 0.00</title><rect class="cell" x="541.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → difficult: 0.01</title><rect class="cell" x="561.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → .: 0.03</title><rect class="cell" x="581.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → &lt;/s&gt;: 0.13</title><rect class="cell" x="601.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.132"/></g><text class="t-tick" x="72.0" y="88.0" text-anchor="end">head 3</text><g class="mark"><title>head 3 → It: 0.00</title><rect class="cell" x="81.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → is: 0.00</title><rect class="cell" x="101.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → in: 0.00</title><rect class="cell" x="121.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → this: 0.00</title><rect class="cell" x="141.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → spirit: 0.00</title><rect class="cell" x="161.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → that: 0.00</title><rect class="cell" x="181.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → a: 0.00</title><rect class="cell" x="201.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → majority: 0.00</title><rect class="cell" x="221.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → of: 0.00</title><rect class="cell" x="241.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → American: 0.00</title><rect class="cell" x="261.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → governments: 0.00</title><rect class="cell" x="281.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → have: 0.00</title><rect class="cell" x="301.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → passed: 0.00</title><rect class="cell" x="321.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → new: 0.00</title><rect class="cell" x="341.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → laws: 0.00</title><rect class="cell" x="361.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → since: 0.00</title><rect class="cell" x="381.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → 2009: 0.00</title><rect class="cell" x="401.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → making: 0.00</title><rect class="cell" x="421.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → the: 0.00</title><rect class="cell" x="441.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → registration: 0.00</title><rect class="cell" x="461.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → or: 0.00</title><rect class="cell" x="481.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → voting: 0.00</title><rect class="cell" x="501.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → process: 0.00</title><rect class="cell" x="521.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → more: 0.29</title><rect class="cell" x="541.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.289"/></g><g class="mark"><title>head 3 → difficult: 0.71</title><rect class="cell" x="561.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.709"/></g><g class="mark"><title>head 3 → .: 0.00</title><rect class="cell" x="581.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → &lt;/s&gt;: 0.00</title><rect class="cell" x="601.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="108.0" text-anchor="end">head 4</text><g class="mark"><title>head 4 → It: 0.01</title><rect class="cell" x="81.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → is: 0.01</title><rect class="cell" x="101.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → in: 0.01</title><rect class="cell" x="121.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → this: 0.01</title><rect class="cell" x="141.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → spirit: 0.00</title><rect class="cell" x="161.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → that: 0.02</title><rect class="cell" x="181.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → a: 0.01</title><rect class="cell" x="201.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → majority: 0.01</title><rect class="cell" x="221.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → of: 0.02</title><rect class="cell" x="241.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → American: 0.00</title><rect class="cell" x="261.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → governments: 0.01</title><rect class="cell" x="281.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → have: 0.03</title><rect class="cell" x="301.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → passed: 0.08</title><rect class="cell" x="321.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.082"/></g><g class="mark"><title>head 4 → new: 0.03</title><rect class="cell" x="341.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → laws: 0.05</title><rect class="cell" x="361.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.049"/></g><g class="mark"><title>head 4 → since: 0.03</title><rect class="cell" x="381.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → 2009: 0.06</title><rect class="cell" x="401.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.061"/></g><g class="mark"><title>head 4 → making: 0.19</title><rect class="cell" x="421.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.187"/></g><g class="mark"><title>head 4 → the: 0.06</title><rect class="cell" x="441.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.059"/></g><g class="mark"><title>head 4 → registration: 0.05</title><rect class="cell" x="461.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.051"/></g><g class="mark"><title>head 4 → or: 0.01</title><rect class="cell" x="481.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → voting: 0.04</title><rect class="cell" x="501.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.042"/></g><g class="mark"><title>head 4 → process: 0.03</title><rect class="cell" x="521.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → more: 0.02</title><rect class="cell" x="541.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → difficult: 0.02</title><rect class="cell" x="561.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → .: 0.04</title><rect class="cell" x="581.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → &lt;/s&gt;: 0.14</title><rect class="cell" x="601.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.140"/></g><text class="t-tick" x="72.0" y="128.0" text-anchor="end">head 5</text><g class="mark"><title>head 5 → It: 0.00</title><rect class="cell" x="81.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → is: 0.00</title><rect class="cell" x="101.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → in: 0.00</title><rect class="cell" x="121.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → this: 0.00</title><rect class="cell" x="141.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → spirit: 0.00</title><rect class="cell" x="161.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → that: 0.01</title><rect class="cell" x="181.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → a: 0.00</title><rect class="cell" x="201.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → majority: 0.00</title><rect class="cell" x="221.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → of: 0.00</title><rect class="cell" x="241.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → American: 0.00</title><rect class="cell" x="261.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → governments: 0.00</title><rect class="cell" x="281.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → have: 0.00</title><rect class="cell" x="301.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → passed: 0.00</title><rect class="cell" x="321.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → new: 0.00</title><rect class="cell" x="341.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → laws: 0.00</title><rect class="cell" x="361.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → since: 0.00</title><rect class="cell" x="381.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → 2009: 0.01</title><rect class="cell" x="401.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → making: 0.03</title><rect class="cell" x="421.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → the: 0.04</title><rect class="cell" x="441.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.042"/></g><g class="mark"><title>head 5 → registration: 0.04</title><rect class="cell" x="461.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → or: 0.09</title><rect class="cell" x="481.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.090"/></g><g class="mark"><title>head 5 → voting: 0.02</title><rect class="cell" x="501.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → process: 0.10</title><rect class="cell" x="521.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.099"/></g><g class="mark"><title>head 5 → more: 0.29</title><rect class="cell" x="541.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.293"/></g><g class="mark"><title>head 5 → difficult: 0.05</title><rect class="cell" x="561.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.048"/></g><g class="mark"><title>head 5 → .: 0.18</title><rect class="cell" x="581.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.181"/></g><g class="mark"><title>head 5 → &lt;/s&gt;: 0.13</title><rect class="cell" x="601.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.126"/></g><text class="t-tick" x="72.0" y="148.0" text-anchor="end">head 6</text><g class="mark"><title>head 6 → It: 0.00</title><rect class="cell" x="81.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → is: 0.00</title><rect class="cell" x="101.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → in: 0.00</title><rect class="cell" x="121.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → this: 0.00</title><rect class="cell" x="141.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → spirit: 0.00</title><rect class="cell" x="161.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → that: 0.00</title><rect class="cell" x="181.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → a: 0.00</title><rect class="cell" x="201.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → majority: 0.00</title><rect class="cell" x="221.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → of: 0.00</title><rect class="cell" x="241.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → American: 0.00</title><rect class="cell" x="261.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → governments: 0.00</title><rect class="cell" x="281.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → have: 0.00</title><rect class="cell" x="301.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → passed: 0.01</title><rect class="cell" x="321.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → new: 0.01</title><rect class="cell" x="341.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → laws: 0.01</title><rect class="cell" x="361.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → since: 0.00</title><rect class="cell" x="381.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → 2009: 0.00</title><rect class="cell" x="401.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → making: 0.03</title><rect class="cell" x="421.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → the: 0.01</title><rect class="cell" x="441.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → registration: 0.01</title><rect class="cell" x="461.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → or: 0.01</title><rect class="cell" x="481.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → voting: 0.00</title><rect class="cell" x="501.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → process: 0.03</title><rect class="cell" x="521.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → more: 0.12</title><rect class="cell" x="541.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.122"/></g><g class="mark"><title>head 6 → difficult: 0.69</title><rect class="cell" x="561.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.692"/></g><g class="mark"><title>head 6 → .: 0.00</title><rect class="cell" x="581.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → &lt;/s&gt;: 0.04</title><rect class="cell" x="601.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="168.0" text-anchor="end">head 7</text><g class="mark"><title>head 7 → It: 0.00</title><rect class="cell" x="81.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → is: 0.00</title><rect class="cell" x="101.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → in: 0.00</title><rect class="cell" x="121.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → this: 0.00</title><rect class="cell" x="141.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → spirit: 0.00</title><rect class="cell" x="161.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → that: 0.00</title><rect class="cell" x="181.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → a: 0.00</title><rect class="cell" x="201.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → majority: 0.00</title><rect class="cell" x="221.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → of: 0.00</title><rect class="cell" x="241.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → American: 0.00</title><rect class="cell" x="261.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → governments: 0.00</title><rect class="cell" x="281.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → have: 0.00</title><rect class="cell" x="301.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → passed: 0.00</title><rect class="cell" x="321.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → new: 0.00</title><rect class="cell" x="341.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → laws: 0.00</title><rect class="cell" x="361.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → since: 0.00</title><rect class="cell" x="381.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → 2009: 0.00</title><rect class="cell" x="401.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → making: 0.99</title><rect class="cell" x="421.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.988"/></g><g class="mark"><title>head 7 → the: 0.00</title><rect class="cell" x="441.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → registration: 0.00</title><rect class="cell" x="461.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → or: 0.00</title><rect class="cell" x="481.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → voting: 0.00</title><rect class="cell" x="501.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → process: 0.00</title><rect class="cell" x="521.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → more: 0.00</title><rect class="cell" x="541.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → difficult: 0.01</title><rect class="cell" x="561.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → .: 0.00</title><rect class="cell" x="581.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → &lt;/s&gt;: 0.00</title><rect class="cell" x="601.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="188.0" text-anchor="end">head 8</text><g class="mark"><title>head 8 → It: 0.02</title><rect class="cell" x="81.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → is: 0.02</title><rect class="cell" x="101.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → in: 0.00</title><rect class="cell" x="121.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → this: 0.00</title><rect class="cell" x="141.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → spirit: 0.00</title><rect class="cell" x="161.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → that: 0.02</title><rect class="cell" x="181.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → a: 0.00</title><rect class="cell" x="201.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → majority: 0.00</title><rect class="cell" x="221.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → of: 0.00</title><rect class="cell" x="241.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → American: 0.00</title><rect class="cell" x="261.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → governments: 0.00</title><rect class="cell" x="281.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → have: 0.00</title><rect class="cell" x="301.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → passed: 0.00</title><rect class="cell" x="321.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → new: 0.01</title><rect class="cell" x="341.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → laws: 0.01</title><rect class="cell" x="361.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → since: 0.01</title><rect class="cell" x="381.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → 2009: 0.05</title><rect class="cell" x="401.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.054"/></g><g class="mark"><title>head 8 → making: 0.07</title><rect class="cell" x="421.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.074"/></g><g class="mark"><title>head 8 → the: 0.04</title><rect class="cell" x="441.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.042"/></g><g class="mark"><title>head 8 → registration: 0.02</title><rect class="cell" x="461.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → or: 0.01</title><rect class="cell" x="481.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → voting: 0.01</title><rect class="cell" x="501.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → process: 0.05</title><rect class="cell" x="521.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.048"/></g><g class="mark"><title>head 8 → more: 0.01</title><rect class="cell" x="541.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → difficult: 0.04</title><rect class="cell" x="561.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.041"/></g><g class="mark"><title>head 8 → .: 0.22</title><rect class="cell" x="581.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.223"/></g><g class="mark"><title>head 8 → &lt;/s&gt;: 0.36</title><rect class="cell" x="601.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.363"/></g><text class="t-tick" x="90.0" y="204.0" text-anchor="end" transform="rotate(-50 90.0 204.0)">It</text><text class="t-tick" x="110.0" y="204.0" text-anchor="end" transform="rotate(-50 110.0 204.0)">is</text><text class="t-tick" x="130.0" y="204.0" text-anchor="end" transform="rotate(-50 130.0 204.0)">in</text><text class="t-tick" x="150.0" y="204.0" text-anchor="end" transform="rotate(-50 150.0 204.0)">this</text><text class="t-tick" x="170.0" y="204.0" text-anchor="end" transform="rotate(-50 170.0 204.0)">spirit</text><text class="t-tick" x="190.0" y="204.0" text-anchor="end" transform="rotate(-50 190.0 204.0)">that</text><text class="t-tick" x="210.0" y="204.0" text-anchor="end" transform="rotate(-50 210.0 204.0)">a</text><text class="t-tick" x="230.0" y="204.0" text-anchor="end" transform="rotate(-50 230.0 204.0)">majority</text><text class="t-tick" x="250.0" y="204.0" text-anchor="end" transform="rotate(-50 250.0 204.0)">of</text><text class="t-tick" x="270.0" y="204.0" text-anchor="end" transform="rotate(-50 270.0 204.0)">American</text><text class="t-tick" x="290.0" y="204.0" text-anchor="end" transform="rotate(-50 290.0 204.0)">governments</text><text class="t-tick" x="310.0" y="204.0" text-anchor="end" transform="rotate(-50 310.0 204.0)">have</text><text class="t-tick" x="330.0" y="204.0" text-anchor="end" transform="rotate(-50 330.0 204.0)">passed</text><text class="t-tick" x="350.0" y="204.0" text-anchor="end" transform="rotate(-50 350.0 204.0)">new</text><text class="t-tick" x="370.0" y="204.0" text-anchor="end" transform="rotate(-50 370.0 204.0)">laws</text><text class="t-tick" x="390.0" y="204.0" text-anchor="end" transform="rotate(-50 390.0 204.0)">since</text><text class="t-tick" x="410.0" y="204.0" text-anchor="end" transform="rotate(-50 410.0 204.0)">2009</text><text class="t-tick" x="430.0" y="204.0" text-anchor="end" transform="rotate(-50 430.0 204.0)">making</text><text class="t-tick" x="450.0" y="204.0" text-anchor="end" transform="rotate(-50 450.0 204.0)">the</text><text class="t-tick" x="470.0" y="204.0" text-anchor="end" transform="rotate(-50 470.0 204.0)">registration</text><text class="t-tick" x="490.0" y="204.0" text-anchor="end" transform="rotate(-50 490.0 204.0)">or</text><text class="t-tick" x="510.0" y="204.0" text-anchor="end" transform="rotate(-50 510.0 204.0)">voting</text><text class="t-tick" x="530.0" y="204.0" text-anchor="end" transform="rotate(-50 530.0 204.0)">process</text><text class="t-tick" x="550.0" y="204.0" text-anchor="end" transform="rotate(-50 550.0 204.0)">more</text><text class="t-tick" x="570.0" y="204.0" text-anchor="end" transform="rotate(-50 570.0 204.0)">difficult</text><text class="t-tick" x="590.0" y="204.0" text-anchor="end" transform="rotate(-50 590.0 204.0)">.</text><text class="t-tick" x="610.0" y="204.0" text-anchor="end" transform="rotate(-50 610.0 204.0)">&lt;/s&gt;</text></svg><figcaption>Attention from the word "making" in the 8 heads of encoder layer 5 of the released Marian model. Darker cells mean more weight. Head 3 puts 0.71 on "difficult" and 0.29 on "more"; head 6 puts 0.69 on "difficult". Head 7 looks only at "making" itself.</figcaption></figure>

The result is close to Figure 3. Head 3 puts all its weight on "difficult" (0.71) and "more" (0.29), five and six tokens away. Head 6 does almost the same. This is a different model, trained by other people on other data, and it found the same link. Other heads do other things. Head 7 looks only at the word itself, and heads 2 and 8 spread their weight or rest on the end token.

### The heatmaps in code

Where do these numbers come from? The library returns them if you ask with `output_attentions=True`. But they are only the softmax of Part 3, and we can compute them ourselves from the model's own weights.

**Method 1: from scratch.** We take the vectors that enter encoder layer 5 and apply the layer's query and key matrices. Then we split them into 8 heads and compute $$\text{softmax}(QK^\top / \sqrt{d_k})$$ for each head.

```python
@torch.no_grad()
def attention_weights_by_hand(layer, hidden, h=8):
    """softmax(Q K^T / sqrt(d_k)) for every head of one self-attention layer. hidden: (1, n, d_model)."""
    n, d_model = hidden.shape[1], hidden.shape[2]
    d_k = d_model // h
    Q = layer.self_attn.q_proj(hidden)                       # (1, n, d_model)
    K = layer.self_attn.k_proj(hidden)
    Q = Q.view(1, n, h, d_k).transpose(1, 2)                 # (1, h, n, d_k): split into heads
    K = K.view(1, n, h, d_k).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)        # (1, h, n, n): every query against every key
    return torch.softmax(scores, dim=-1)                     # each row sums to 1


L = 4                                                        # layer 5 of 6, counted from 0
ours = attention_weights_by_hand(model.get_encoder().layers[L], enc.hidden_states[L])   # input of layer 5
lib = enc.attentions[L]
```

`enc.hidden_states` is the list of vectors between the layers. Entry 0 is the embedding, and entry 4 is the output of layer 4, the input of layer 5. The values $$V$$ are not needed. The weights depend only on queries and keys; the values come in after, when the weights mix them.

**Method 2: with the library.** `enc.attentions[4]` holds the same weights, from the same forward pass.

```text
27 tokens; shape (1, 8, 27, 27); by hand vs output_attentions: max abs difference 0.0e+00
```

The two agree exactly: 8 heads, each a table of 27 queries × 27 keys. The heatmap of "making" is one row of each table, row 18, for each of the 8 heads.

The last step is the picture. A heatmap maps each weight to a shade: 0 is blank, 1 is full ink. Our figure scripts draw SVG squares with an opacity equal to the weight. In a terminal, the same idea fits in a few lines, with ten characters from light to dark:

```python
SHADES = ' .:-=+*#%@'                                       # 10 levels of ink, light to dark


def text_heatmap(rows, labels):
    """One line per head; one character per token; darker = more weight."""
    lines = []
    for name, row in zip(labels, rows):
        lines.append(f'{name:>7s} |' + ''.join(SHADES[min(9, int(w * 10))] for w in row.tolist()) + '|')
    return lines
```

```text
 head 1 |                 +      .  |
 head 2 |            . . .         .|
 head 3 |                       :#  |
 head 4 |                 .        .|
 head 5 |                       : ..|
 head 6 |                       .*  |
 head 7 |                 @         |
 head 8 |                         :-|
columns: 27 tokens; "making" is column 18, "more" is 24, "difficult" is 25
```

Read it like the figure. Head 7's "@" sits in column 18: "making" looks at itself with weight 0.99. Heads 3 and 6 have their dark marks in columns 24 and 25, "more" and "difficult". Head 8 puts its marks at the right edge, on the full stop and the end token. The SVG figure above is the same table, drawn with more shades and the words as labels.

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

<figure class="fig"><svg viewBox="0 0 620 304" role="img" aria-label="Attention weights from the word its to every token of the sentence, for the 8 heads of encoder layer 5 of the released Marian English-to-German model."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="340.0" y="22.0" text-anchor="middle">Where "its" looks, encoder layer 5 of 6</text><text class="t-tick" x="72.0" y="48.0" text-anchor="end">head 1</text><g class="mark"><title>head 1 → The: 0.14</title><rect class="cell" x="81.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.145"/></g><g class="mark"><title>head 1 → Law: 0.02</title><rect class="cell" x="101.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → will: 0.00</title><rect class="cell" x="121.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → never: 0.00</title><rect class="cell" x="141.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → be: 0.00</title><rect class="cell" x="161.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → perfect: 0.00</title><rect class="cell" x="181.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → ,: 0.00</title><rect class="cell" x="201.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → but: 0.00</title><rect class="cell" x="221.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → its: 0.82</title><rect class="cell" x="241.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.819"/></g><g class="mark"><title>head 1 → application: 0.02</title><rect class="cell" x="261.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → should: 0.00</title><rect class="cell" x="281.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → be: 0.00</title><rect class="cell" x="301.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → just: 0.00</title><rect class="cell" x="321.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → -: 0.00</title><rect class="cell" x="341.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → this: 0.00</title><rect class="cell" x="361.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → is: 0.00</title><rect class="cell" x="381.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → what: 0.00</title><rect class="cell" x="401.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → we: 0.00</title><rect class="cell" x="421.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → are: 0.00</title><rect class="cell" x="441.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → missing: 0.00</title><rect class="cell" x="461.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → ,: 0.00</title><rect class="cell" x="481.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → in: 0.00</title><rect class="cell" x="501.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → my: 0.00</title><rect class="cell" x="521.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → opinion: 0.00</title><rect class="cell" x="541.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → .: 0.00</title><rect class="cell" x="561.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 1 → &lt;/s&gt;: 0.00</title><rect class="cell" x="581.0" y="35.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="68.0" text-anchor="end">head 2</text><g class="mark"><title>head 2 → The: 0.01</title><rect class="cell" x="81.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → Law: 0.02</title><rect class="cell" x="101.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → will: 0.02</title><rect class="cell" x="121.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → never: 0.01</title><rect class="cell" x="141.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → be: 0.02</title><rect class="cell" x="161.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → perfect: 0.02</title><rect class="cell" x="181.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → ,: 0.27</title><rect class="cell" x="201.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.267"/></g><g class="mark"><title>head 2 → but: 0.32</title><rect class="cell" x="221.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.319"/></g><g class="mark"><title>head 2 → its: 0.08</title><rect class="cell" x="241.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.076"/></g><g class="mark"><title>head 2 → application: 0.04</title><rect class="cell" x="261.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.043"/></g><g class="mark"><title>head 2 → should: 0.02</title><rect class="cell" x="281.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → be: 0.01</title><rect class="cell" x="301.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → just: 0.00</title><rect class="cell" x="321.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → -: 0.02</title><rect class="cell" x="341.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → this: 0.00</title><rect class="cell" x="361.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → is: 0.00</title><rect class="cell" x="381.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → what: 0.00</title><rect class="cell" x="401.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → we: 0.00</title><rect class="cell" x="421.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → are: 0.00</title><rect class="cell" x="441.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → missing: 0.00</title><rect class="cell" x="461.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → ,: 0.00</title><rect class="cell" x="481.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → in: 0.00</title><rect class="cell" x="501.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → my: 0.00</title><rect class="cell" x="521.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → opinion: 0.00</title><rect class="cell" x="541.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → .: 0.01</title><rect class="cell" x="561.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 2 → &lt;/s&gt;: 0.12</title><rect class="cell" x="581.0" y="55.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.124"/></g><text class="t-tick" x="72.0" y="88.0" text-anchor="end">head 3</text><g class="mark"><title>head 3 → The: 0.00</title><rect class="cell" x="81.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → Law: 0.00</title><rect class="cell" x="101.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → will: 0.00</title><rect class="cell" x="121.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → never: 0.00</title><rect class="cell" x="141.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → be: 0.00</title><rect class="cell" x="161.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → perfect: 0.00</title><rect class="cell" x="181.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → ,: 0.00</title><rect class="cell" x="201.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → but: 0.00</title><rect class="cell" x="221.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → its: 0.01</title><rect class="cell" x="241.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → application: 0.88</title><rect class="cell" x="261.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.881"/></g><g class="mark"><title>head 3 → should: 0.01</title><rect class="cell" x="281.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → be: 0.02</title><rect class="cell" x="301.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → just: 0.03</title><rect class="cell" x="321.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → -: 0.00</title><rect class="cell" x="341.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → this: 0.00</title><rect class="cell" x="361.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → is: 0.00</title><rect class="cell" x="381.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → what: 0.00</title><rect class="cell" x="401.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → we: 0.00</title><rect class="cell" x="421.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → are: 0.00</title><rect class="cell" x="441.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → missing: 0.00</title><rect class="cell" x="461.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → ,: 0.00</title><rect class="cell" x="481.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → in: 0.00</title><rect class="cell" x="501.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → my: 0.00</title><rect class="cell" x="521.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → opinion: 0.00</title><rect class="cell" x="541.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → .: 0.00</title><rect class="cell" x="561.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 3 → &lt;/s&gt;: 0.04</title><rect class="cell" x="581.0" y="75.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="108.0" text-anchor="end">head 4</text><g class="mark"><title>head 4 → The: 0.05</title><rect class="cell" x="81.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.048"/></g><g class="mark"><title>head 4 → Law: 0.04</title><rect class="cell" x="101.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → will: 0.00</title><rect class="cell" x="121.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → never: 0.03</title><rect class="cell" x="141.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → be: 0.03</title><rect class="cell" x="161.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → perfect: 0.04</title><rect class="cell" x="181.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.041"/></g><g class="mark"><title>head 4 → ,: 0.01</title><rect class="cell" x="201.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → but: 0.02</title><rect class="cell" x="221.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → its: 0.14</title><rect class="cell" x="241.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.135"/></g><g class="mark"><title>head 4 → application: 0.04</title><rect class="cell" x="261.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → should: 0.01</title><rect class="cell" x="281.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → be: 0.09</title><rect class="cell" x="301.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.094"/></g><g class="mark"><title>head 4 → just: 0.02</title><rect class="cell" x="321.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → -: 0.01</title><rect class="cell" x="341.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → this: 0.03</title><rect class="cell" x="361.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → is: 0.01</title><rect class="cell" x="381.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → what: 0.05</title><rect class="cell" x="401.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.049"/></g><g class="mark"><title>head 4 → we: 0.05</title><rect class="cell" x="421.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.052"/></g><g class="mark"><title>head 4 → are: 0.00</title><rect class="cell" x="441.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → missing: 0.02</title><rect class="cell" x="461.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → ,: 0.00</title><rect class="cell" x="481.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → in: 0.00</title><rect class="cell" x="501.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → my: 0.02</title><rect class="cell" x="521.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → opinion: 0.00</title><rect class="cell" x="541.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → .: 0.02</title><rect class="cell" x="561.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 4 → &lt;/s&gt;: 0.27</title><rect class="cell" x="581.0" y="95.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.271"/></g><text class="t-tick" x="72.0" y="128.0" text-anchor="end">head 5</text><g class="mark"><title>head 5 → The: 0.00</title><rect class="cell" x="81.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → Law: 0.00</title><rect class="cell" x="101.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → will: 0.00</title><rect class="cell" x="121.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → never: 0.00</title><rect class="cell" x="141.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → be: 0.00</title><rect class="cell" x="161.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → perfect: 0.00</title><rect class="cell" x="181.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → ,: 0.00</title><rect class="cell" x="201.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → but: 0.02</title><rect class="cell" x="221.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → its: 0.03</title><rect class="cell" x="241.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → application: 0.02</title><rect class="cell" x="261.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → should: 0.11</title><rect class="cell" x="281.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.112"/></g><g class="mark"><title>head 5 → be: 0.46</title><rect class="cell" x="301.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.462"/></g><g class="mark"><title>head 5 → just: 0.03</title><rect class="cell" x="321.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → -: 0.23</title><rect class="cell" x="341.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.228"/></g><g class="mark"><title>head 5 → this: 0.02</title><rect class="cell" x="361.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → is: 0.01</title><rect class="cell" x="381.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → what: 0.01</title><rect class="cell" x="401.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → we: 0.01</title><rect class="cell" x="421.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → are: 0.00</title><rect class="cell" x="441.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → missing: 0.00</title><rect class="cell" x="461.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → ,: 0.00</title><rect class="cell" x="481.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → in: 0.00</title><rect class="cell" x="501.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → my: 0.00</title><rect class="cell" x="521.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → opinion: 0.00</title><rect class="cell" x="541.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → .: 0.00</title><rect class="cell" x="561.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 5 → &lt;/s&gt;: 0.03</title><rect class="cell" x="581.0" y="115.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="148.0" text-anchor="end">head 6</text><g class="mark"><title>head 6 → The: 0.00</title><rect class="cell" x="81.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → Law: 0.00</title><rect class="cell" x="101.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → will: 0.00</title><rect class="cell" x="121.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → never: 0.00</title><rect class="cell" x="141.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → be: 0.00</title><rect class="cell" x="161.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → perfect: 0.01</title><rect class="cell" x="181.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → ,: 0.01</title><rect class="cell" x="201.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → but: 0.02</title><rect class="cell" x="221.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → its: 0.29</title><rect class="cell" x="241.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.292"/></g><g class="mark"><title>head 6 → application: 0.27</title><rect class="cell" x="261.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.270"/></g><g class="mark"><title>head 6 → should: 0.01</title><rect class="cell" x="281.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → be: 0.02</title><rect class="cell" x="301.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → just: 0.03</title><rect class="cell" x="321.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → -: 0.01</title><rect class="cell" x="341.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → this: 0.00</title><rect class="cell" x="361.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → is: 0.00</title><rect class="cell" x="381.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → what: 0.01</title><rect class="cell" x="401.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → we: 0.00</title><rect class="cell" x="421.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → are: 0.00</title><rect class="cell" x="441.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → missing: 0.01</title><rect class="cell" x="461.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → ,: 0.00</title><rect class="cell" x="481.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → in: 0.00</title><rect class="cell" x="501.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → my: 0.00</title><rect class="cell" x="521.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → opinion: 0.00</title><rect class="cell" x="541.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → .: 0.01</title><rect class="cell" x="561.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 6 → &lt;/s&gt;: 0.27</title><rect class="cell" x="581.0" y="135.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.272"/></g><text class="t-tick" x="72.0" y="168.0" text-anchor="end">head 7</text><g class="mark"><title>head 7 → The: 0.00</title><rect class="cell" x="81.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → Law: 0.00</title><rect class="cell" x="101.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → will: 0.00</title><rect class="cell" x="121.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → never: 0.00</title><rect class="cell" x="141.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → be: 0.02</title><rect class="cell" x="161.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → perfect: 0.00</title><rect class="cell" x="181.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → ,: 0.00</title><rect class="cell" x="201.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → but: 0.01</title><rect class="cell" x="221.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → its: 0.55</title><rect class="cell" x="241.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.551"/></g><g class="mark"><title>head 7 → application: 0.01</title><rect class="cell" x="261.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → should: 0.02</title><rect class="cell" x="281.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → be: 0.17</title><rect class="cell" x="301.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.165"/></g><g class="mark"><title>head 7 → just: 0.00</title><rect class="cell" x="321.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → -: 0.03</title><rect class="cell" x="341.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → this: 0.00</title><rect class="cell" x="361.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → is: 0.10</title><rect class="cell" x="381.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.104"/></g><g class="mark"><title>head 7 → what: 0.01</title><rect class="cell" x="401.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → we: 0.00</title><rect class="cell" x="421.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → are: 0.00</title><rect class="cell" x="441.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → missing: 0.06</title><rect class="cell" x="461.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.060"/></g><g class="mark"><title>head 7 → ,: 0.00</title><rect class="cell" x="481.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → in: 0.00</title><rect class="cell" x="501.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → my: 0.00</title><rect class="cell" x="521.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → opinion: 0.00</title><rect class="cell" x="541.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → .: 0.00</title><rect class="cell" x="561.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 7 → &lt;/s&gt;: 0.00</title><rect class="cell" x="581.0" y="155.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><text class="t-tick" x="72.0" y="188.0" text-anchor="end">head 8</text><g class="mark"><title>head 8 → The: 0.00</title><rect class="cell" x="81.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → Law: 0.01</title><rect class="cell" x="101.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → will: 0.00</title><rect class="cell" x="121.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → never: 0.00</title><rect class="cell" x="141.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → be: 0.01</title><rect class="cell" x="161.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → perfect: 0.04</title><rect class="cell" x="181.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → ,: 0.07</title><rect class="cell" x="201.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.074"/></g><g class="mark"><title>head 8 → but: 0.14</title><rect class="cell" x="221.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.139"/></g><g class="mark"><title>head 8 → its: 0.04</title><rect class="cell" x="241.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.042"/></g><g class="mark"><title>head 8 → application: 0.06</title><rect class="cell" x="261.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.062"/></g><g class="mark"><title>head 8 → should: 0.01</title><rect class="cell" x="281.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → be: 0.07</title><rect class="cell" x="301.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.068"/></g><g class="mark"><title>head 8 → just: 0.06</title><rect class="cell" x="321.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.059"/></g><g class="mark"><title>head 8 → -: 0.05</title><rect class="cell" x="341.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.048"/></g><g class="mark"><title>head 8 → this: 0.00</title><rect class="cell" x="361.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → is: 0.01</title><rect class="cell" x="381.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → what: 0.00</title><rect class="cell" x="401.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → we: 0.00</title><rect class="cell" x="421.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → are: 0.00</title><rect class="cell" x="441.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → missing: 0.06</title><rect class="cell" x="461.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.060"/></g><g class="mark"><title>head 8 → ,: 0.00</title><rect class="cell" x="481.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → in: 0.01</title><rect class="cell" x="501.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → my: 0.00</title><rect class="cell" x="521.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → opinion: 0.00</title><rect class="cell" x="541.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → .: 0.01</title><rect class="cell" x="561.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.040"/></g><g class="mark"><title>head 8 → &lt;/s&gt;: 0.34</title><rect class="cell" x="581.0" y="175.0" width="18.0" height="18.0" rx="2" style="fill-opacity:0.340"/></g><text class="t-tick" x="90.0" y="204.0" text-anchor="end" transform="rotate(-50 90.0 204.0)">The</text><text class="t-tick" x="110.0" y="204.0" text-anchor="end" transform="rotate(-50 110.0 204.0)">Law</text><text class="t-tick" x="130.0" y="204.0" text-anchor="end" transform="rotate(-50 130.0 204.0)">will</text><text class="t-tick" x="150.0" y="204.0" text-anchor="end" transform="rotate(-50 150.0 204.0)">never</text><text class="t-tick" x="170.0" y="204.0" text-anchor="end" transform="rotate(-50 170.0 204.0)">be</text><text class="t-tick" x="190.0" y="204.0" text-anchor="end" transform="rotate(-50 190.0 204.0)">perfect</text><text class="t-tick" x="210.0" y="204.0" text-anchor="end" transform="rotate(-50 210.0 204.0)">,</text><text class="t-tick" x="230.0" y="204.0" text-anchor="end" transform="rotate(-50 230.0 204.0)">but</text><text class="t-tick" x="250.0" y="204.0" text-anchor="end" transform="rotate(-50 250.0 204.0)">its</text><text class="t-tick" x="270.0" y="204.0" text-anchor="end" transform="rotate(-50 270.0 204.0)">application</text><text class="t-tick" x="290.0" y="204.0" text-anchor="end" transform="rotate(-50 290.0 204.0)">should</text><text class="t-tick" x="310.0" y="204.0" text-anchor="end" transform="rotate(-50 310.0 204.0)">be</text><text class="t-tick" x="330.0" y="204.0" text-anchor="end" transform="rotate(-50 330.0 204.0)">just</text><text class="t-tick" x="350.0" y="204.0" text-anchor="end" transform="rotate(-50 350.0 204.0)">-</text><text class="t-tick" x="370.0" y="204.0" text-anchor="end" transform="rotate(-50 370.0 204.0)">this</text><text class="t-tick" x="390.0" y="204.0" text-anchor="end" transform="rotate(-50 390.0 204.0)">is</text><text class="t-tick" x="410.0" y="204.0" text-anchor="end" transform="rotate(-50 410.0 204.0)">what</text><text class="t-tick" x="430.0" y="204.0" text-anchor="end" transform="rotate(-50 430.0 204.0)">we</text><text class="t-tick" x="450.0" y="204.0" text-anchor="end" transform="rotate(-50 450.0 204.0)">are</text><text class="t-tick" x="470.0" y="204.0" text-anchor="end" transform="rotate(-50 470.0 204.0)">missing</text><text class="t-tick" x="490.0" y="204.0" text-anchor="end" transform="rotate(-50 490.0 204.0)">,</text><text class="t-tick" x="510.0" y="204.0" text-anchor="end" transform="rotate(-50 510.0 204.0)">in</text><text class="t-tick" x="530.0" y="204.0" text-anchor="end" transform="rotate(-50 530.0 204.0)">my</text><text class="t-tick" x="550.0" y="204.0" text-anchor="end" transform="rotate(-50 550.0 204.0)">opinion</text><text class="t-tick" x="570.0" y="204.0" text-anchor="end" transform="rotate(-50 570.0 204.0)">.</text><text class="t-tick" x="590.0" y="204.0" text-anchor="end" transform="rotate(-50 590.0 204.0)">&lt;/s&gt;</text></svg><figcaption>Attention from the word "its" in the 8 heads of encoder layer 5 of the released Marian model. Head 3 puts 0.88 on "application". No head of this layer puts much weight on "Law": the largest is 0.02, in head 1.</figcaption></figure>

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
> - **BLEU counts shared word groups.** It is the geometric mean of the precisions of 1- to 4-word groups, times a penalty for short outputs. In our toy example, one wrong word cut the score from 100 to 53.7. Our BLEU in Python gives sacreBLEU's numbers.
> - **Better and cheaper at once.** The big model got 28.4 BLEU on English-to-German, more than 2 points above every earlier result, ensembles included. The base model (27.3) beat them all at $$3.3 \cdot 10^{18}$$ operations.
> - **The cost is time × cards × speed.** 12 hours × 8 cards × 9.5 TFLOPS gives $$3.28 \cdot 10^{18}$$; 3.5 days gives $$2.30 \cdot 10^{19}$$. Both match Table 2.
> - **English-to-French: 41.0 or 41.8.** The text and Table 2 disagree. Either way, the cost is less than a quarter of the best earlier single model.
> - **Decoding settings matter.** The last 5 or 20 checkpoints are averaged, and beam search uses 4 beams and a length penalty with $$\alpha = 0.6$$. Our own beam search gives the same tokens as `model.generate`, and it fixed a greedy "noch" into "immer noch".
> - **Eight heads is a sweet spot.** With the same budget of 512 numbers, 1 head loses 0.9 BLEU and 32 heads lose 0.4. Keys that are too small hurt.
> - **Bigger is better, dropout helps, and learned positions work as well as sinusoids.** 2 layers give 23.7 BLEU; no dropout gives 24.6; learned positions give 25.7 against 25.8.
> - **The size follows from the shapes.** Our count from the shapes gives 63.1 million weights for the base model (paper: 65) and 214.2 million for the big model (paper: 213).
> - **One design, a second task.** With little tuning, a 4-layer Transformer parsed English at 91.3 F1 from 40,000 sentences, close to parsers built for the task.
> - **Heads specialise.** In a released Marian model, one head links "making" to "more difficult" with all its weight, as in Figure 3. The "its" to "Law" link of Figure 4 did not show in the same layer.

The code of the sections "in code" is in [`tx_part6_code.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/transformer/tx_part6_code.py), and its full output is in `results/part6_code_stdout.txt`. [Part 7](part-7-build-it-from-scratch.md) builds the whole Transformer from scratch.

<figure><img src="/img/papers/transformer/part6-run.png" alt="Terminal output of tx_part6.py: the training cost in FLOPs, the parameter counts of the Table 3 rows, the Marian translation with greedy and beam search, and the attention of the words making and its in encoder layer 5" loading="lazy" /><figcaption>The output of tx_part6.py, as it ran for this article.</figcaption></figure>

## The paper and the code

[Attention Is All You Need](https://arxiv.org/abs/1706.03762) (NeurIPS 2017) · [official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
