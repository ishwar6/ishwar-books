---
title: "Pre-training: Masked LM and Next Sentence Prediction"
description: "Section 3.1 and Appendices A.1 and A.2 of the BERT paper, line by line: why a normal language model cannot look both ways, the masked language model and its 80/10/10 rule, next sentence prediction, the training data and the full training recipe. Every claim run on the real model or measured in code."
part: 3
covers: "§3.1, A.1, A.2"
date: 2026-10-04
tags: [bert, nlp, pre-training]
---

Part 2 built the model: a stack of Transformer layers that turns tokens into vectors. But a freshly built model knows nothing. Its millions of weights are random numbers.

This part is about how BERT learns language from plain text, before any real task. The paper uses two training games: **fill in the blanks** (the masked language model) and **does this sentence come next?** (next sentence prediction). We go through Section 3.1 paragraph by paragraph, then the two appendices that give the exact recipe, and we run every piece on the real model.

## Not a normal language model {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1 · page 4
> [![The opening of Section 3.1, Pre-training BERT: unlike Peters et al. and Radford et al., we do not use traditional left-to-right or right-to-left language models to pre-train BERT. Instead, we pre-train BERT using two unsupervised tasks, presented in the left part of Figure 1](/img/papers/bert/p3-intro.png)](/img/papers/bert/p3-intro.png)
>
> **Context:** the start of Section 3.1. Section 3 has just described the model and its input (Part 2).
>
> **What it says:** ELMo (Peters et al.) and OpenAI GPT (Radford et al.) were pre-trained as normal language models, reading left to right or right to left. BERT is not. It uses "two unsupervised tasks" instead.
>
> **Why it matters:** this is where BERT breaks from the usual recipe. The rest of the section explains the two tasks.

> [!DEFINITION] Unsupervised task
> A training task whose answers come from the text itself, so no person has to label anything. "Predict the hidden word" is unsupervised: the hidden word is the answer. (Some people call this **self-supervised**.)

## Why not just look both ways? {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1, Task #1 · page 4
> [![Task 1, Masked LM: a deep bidirectional model is strictly more powerful than a left-to-right model or a shallow concatenation. Unfortunately, standard conditional language models can only be trained left-to-right or right-to-left, since bidirectional conditioning would allow each word to indirectly see itself, and the model could trivially predict the target word in a multi-layered context](/img/papers/bert/p3-mlm-why.png)](/img/papers/bert/p3-mlm-why.png)
>
> **Context:** the first paragraph of Task #1. Part 1 showed why both directions help. Now the paper explains why you cannot simply switch them on.
>
> **What it says:** a deep bidirectional model is "strictly more powerful". But a standard language model cannot be trained that way, because "bidirectional conditioning would allow each word to indirectly 'see itself'", and then the model "could trivially predict the target word in a multi-layered context".
>
> **Why it matters:** this one sentence is the reason the masked language model exists. It is worth understanding exactly.

> [!DEFINITION] Conditional language model
> A model that predicts a word **given** (conditioned on) some other words. A left-to-right model conditions on the words to the left. "Bidirectional conditioning" would mean predicting each word from the words on both sides.

Here is the problem. Say we want every position to predict its own word from all the *other* words. In one layer, that is fine: we can simply forbid each position from looking at itself. But BERT has 12 layers, and information moves sideways at every layer.

{{FIG:p3_see_itself|How a word sees itself through two layers. Position 2 must predict "cat" and never looks at "cat" directly. But in layer 1, the neighbour "sat" reads "cat". In layer 2, position 2 reads "sat", and the answer comes back in.}}

So with two or more layers, every word can find its own answer by going through a neighbour. Training would then learn to copy, not to understand. That is what "trivially predict the target word in a multi-layered context" means.

### A real experiment: watch a model cheat

I trained three tiny Transformers on the same real text (WikiText-2, a public set of Wikipedia articles) and measured how well each predicts words. Each has 128 numbers per token and 4 attention heads, and trains for 3,000 steps on the Apple GPU of a laptop.

- **A. Left-to-right, 2 layers.** A normal language model: predict the next token from the tokens before it.
- **B. Both sides, 1 layer.** Predict each token from every **other** token. Position $$t$$ never sees token $$t$$.
- **C. Both sides, 2 layers.** Exactly like B, with one more layer.

> [!DEFINITION] Loss
> A number that measures how wrong the model's predictions are; training pushes it down. For word prediction, a loss of 0 means "always 100% sure of the right word". With 8,192 possible tokens, guessing evenly gives $$\ln 8192 = 9.01$$.

```python
# simplified from bert_part3_seeitself.py
# B and C: position t may look at every position except itself
not_self = torch.arange(T)[None, :] != torch.arange(T)[:, None]
# in layer 1, the query at position t is built from "position t" only, never from token t
x = blocks[0](x, not_self, query_from=position_embedding, residual=False)
for block in blocks[1:]:          # model C has one more layer here
    x = block(x, not_self)
```

The real output of [`bert_part3_seeitself.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part3_seeitself.py):

```text
A: left-to-right, 2 layers   training loss: step 1 9.30 -> step 3000 4.65   validation loss 5.05 (perplexity 156.1)
B: both sides, 1 layer       training loss: step 1 9.20 -> step 3000 3.95   validation loss 4.37 (perplexity 79.1)
C: both sides, 2 layers      training loss: step 1 9.17 -> step 3000 0.87   validation loss 0.90 (perplexity 2.5)
```

{{FIG:p3_see_chart|Training loss of the three tiny models. A and B learn slowly and honestly. C's loss drops far lower, and stays low on text it never trained on, because it reads its own answer back through a neighbour.}}

> [!DEFINITION] Validation loss and perplexity
> **Validation loss** is the loss on text the model never trained on. It shows whether a model really learned or only memorised. **Perplexity** is $$e^{\text{loss}}$$: roughly, "how many words the model is still choosing between". Lower is better.

Three things to notice:

- **C is cheating.** Its loss on unseen text is 0.90 (perplexity 2.5). No two-layer model with 128 numbers per token can predict real Wikipedia text that well. It is reading the answer.
- **B is honest, and it beats A** (4.37 against 5.05 on unseen text). With one layer there is no way back to the answer, so B really predicts from both sides. Seeing both sides genuinely helps, which is the paper's point from Part 1.
- **The problem appears exactly at the second layer.** That is the "multi-layered context" of the paper. BERT has 12 layers, so it needs another way.

## The fix: hide the words you predict {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1, Task #1 · page 4
> [![To train a deep bidirectional representation, we simply mask some percentage of the input tokens at random, and then predict those masked tokens: a masked LM, also called a Cloze task. The final hidden vectors of the mask tokens are fed into an output softmax over the vocabulary. We mask 15 percent of all WordPiece tokens in each sequence at random. Unlike denoising auto-encoders, we only predict the masked words rather than reconstructing the entire input](/img/papers/bert/p3-mlm-how.png)](/img/papers/bert/p3-mlm-how.png)
>
> **Context:** the second paragraph of Task #1: the solution to the problem above.
>
> **What it says:** "we simply mask some percentage of the input tokens at random, and then predict those masked tokens." The final vectors at the masked positions go into "an output softmax over the vocabulary". They mask "15% of all WordPiece tokens in each sequence at random", and "only predict the masked words rather than reconstructing the entire input".
>
> **Why it matters:** if a word is replaced by `[MASK]` in the input, it is not in the input at all, so no path through any number of layers can bring it back. Now every layer can look both ways safely.

> [!DEFINITION] Softmax over the vocabulary
> For a masked position, the model outputs one score for every token in its vocabulary (30,522 scores for `bert-base-uncased`). **Softmax** turns these scores into probabilities that add up to 1. The highest probability is the model's guess.

{{FIG:p3_mlm|The masked language model. The whole sentence goes in, with the chosen token hidden. Only the output vector of the hidden position (here T₄) goes through a softmax over the vocabulary and counts in the loss.}}

The model is trained with **cross-entropy loss** on the masked positions only:

$$
\mathcal{L}_{\text{MLM}} = -\frac{1}{|M|} \sum_{i \in M} \log P(x_i \mid \tilde{x})
$$

where:

- $$x$$ is the original sequence of tokens, and $$\tilde{x}$$ (read "x tilde") is the same sequence after masking;
- $$M$$ is the set of positions chosen for prediction, and $$|M|$$ is how many there are;
- $$x_i$$ is the original token at position $$i$$, the answer;
- $$P(x_i \mid \tilde{x})$$ is the probability the model gives to that answer, from the softmax at position $$i$$, having seen only $$\tilde{x}$$;
- $$\log$$ is the natural logarithm. If the model is sure and right, $$P = 1$$ and $$-\log P = 0$$. If it gives the answer a probability of 0.01, $$-\log P = 4.6$$.

> [!DEFINITION] Cross-entropy loss
> The standard loss for picking one answer out of many: minus the log of the probability given to the right answer, averaged over all predictions. It punishes confident wrong answers very hard.

How is $$P$$ computed from the final vector $$T_i$$? The paper only says "an output softmax over the vocabulary, as in a standard LM". In the released model it is a small head: one more dense layer with GELU and LayerNorm, then a multiplication by the token embedding matrix (the same matrix used at the input, shared to save weights) plus a bias, then softmax.

> [!DEFINITION] Denoising auto-encoder
> A model that gets a damaged input and must rebuild the **whole** clean input. BERT differs: it only predicts the damaged positions. The other 85% of positions produce no loss at all.

**Use case.** This is the same "fill in the blank" you saw in Part 1, where BERT put 0.901 on "bank". The same masked-LM head is still a quick way to probe what a model knows: give it "The capital of France is [MASK]." and read its guesses.

## The [MASK] problem and the 80/10/10 rule {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1, Task #1 · page 4
> [![A downside is a mismatch between pre-training and fine-tuning, since the MASK token does not appear during fine-tuning. The training data generator chooses 15 percent of the token positions at random for prediction. If the i-th token is chosen, it is replaced by the MASK token 80 percent of the time, a random token 10 percent of the time, and the unchanged i-th token 10 percent of the time. Then T_i is used to predict the original token with cross entropy loss](/img/papers/bert/p3-mlm-8010.png)](/img/papers/bert/p3-mlm-8010.png)
>
> **Context:** the last paragraph of Task #1. Masking works, but it creates a new problem.
>
> **What it says:** "the [MASK] token does not appear during fine-tuning", so there is "a mismatch between pre-training and fine-tuning". To soften it, the chosen 15% of positions are not always masked: "(1) the [MASK] token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time". In every case, $$T_i$$ must predict the original token.
>
> **Why it matters:** this small rule is one of the most copied details in the paper. Part 5 shows the paper's own test of other ratios (Table 8).

> [!DEFINITION] Pre-train/fine-tune mismatch
> When the input during pre-training looks different from the input during real use. BERT pre-trains with `[MASK]` tokens everywhere, but a sentiment classifier never sees `[MASK]`. A model that only learned "work hard at [MASK] positions" might put less effort into normal words.

Appendix A.1 walks through the rule on one sentence, and explains the reasons.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.1 · page 12
> [![Appendix A.1, Masked LM and the Masking Procedure: assuming the sentence is my dog is hairy and the 4th token is chosen: 80 percent of the time replace it with MASK, giving my dog is MASK; 10 percent of the time replace it with a random word, giving my dog is apple; 10 percent of the time keep it unchanged, my dog is hairy. The purpose of this is to bias the representation towards the actual observed word](/img/papers/bert/p3-a1-mlm.png)](/img/papers/bert/p3-a1-mlm.png)
>
> **Context:** Appendix A.1 gives examples of both pre-training tasks. This is the masked LM example.
>
> **What it says:** with "my dog is hairy" and the 4th token chosen: 80% of the time "my dog is [MASK]", 10% "my dog is apple", 10% "my dog is hairy". Keeping the word unchanged is meant "to bias the representation towards the actual observed word".
>
> **Why it matters:** each of the three cases teaches the model something different. The next paragraph of the appendix says what.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.1 · page 12
> [![The advantage of this procedure is that the Transformer encoder does not know which words it will be asked to predict or which have been replaced by random words, so it is forced to keep a distributional contextual representation of every input token. Because random replacement only occurs for 1.5 percent of all tokens, 10 percent of 15 percent, this does not seem to harm the model's language understanding capability](/img/papers/bert/p3-a1-why.png)](/img/papers/bert/p3-a1-why.png)
>
> **Context:** right after the example.
>
> **What it says:** the encoder "does not know which words it will be asked to predict or which have been replaced by random words", so it must keep "a distributional contextual representation of every input token". And random replacement happens to only "1.5% of all tokens (i.e., 10% of 15%)", too rarely to hurt.
>
> **Why it matters:** this is the real reason for the rule: the model can never relax about any token, because any token might be a test.

{{FIG:p3_8010|The 80/10/10 rule. Each chosen token becomes [MASK] 80% of the time, a random word 10% of the time, and stays itself 10% of the time. The target is always the original word.}}

In plain words, the three cases do three jobs:

- **[MASK] (80%)** is the main exercise: fill in a blank from both sides.
- **A random word (10%)** teaches the model that a visible word might be wrong, so it must check every word against its context.
- **Unchanged (10%)** teaches the model that a visible word is usually right, so its vector for a normal word should stay close to that word. This is the case that keeps the model useful when there are no `[MASK]` tokens at all, as in fine-tuning.

### The procedure in code, measured on real text

Here is the rule exactly as Section 3.1 states it:

```python
def mask_tokens(ids, rng, rate=0.15):
    """Choose 15% of the positions. Each chosen one: 80% [MASK], 10% random token, 10% unchanged."""
    cand = [i for i, t in enumerate(ids) if t not in SPECIAL]     # never [CLS], [SEP] or padding
    chosen = sorted(rng.sample(cand, max(1, round(len(cand) * rate))))
    out, case = list(ids), {}
    for i in chosen:
        r = rng.random()
        if r < 0.8:
            out[i], case[i] = tok.mask_token_id, "mask"
        elif r < 0.9:
            out[i], case[i] = rng.randrange(VOCAB_SIZE), "random"   # any id in the vocabulary
        else:
            case[i] = "same"                                        # left as it is, but still predicted
    return out, chosen, case
```

I ran it over the whole test split of WikiText-103 (62 Wikipedia articles), cut into sequences of 128 WordPiece tokens, with a fixed random seed:

```text
1. the masking procedure (paper version), seed 0
   word-piece tokens (not counting [CLS]/[SEP]): 261,324
   chosen for prediction: 39,406 = 15.08% of tokens
   of the chosen: [MASK] 31,548 (80.06%), random 3,933 (9.98%), unchanged 3,925 (9.96%)
   random replacements as a share of ALL tokens: 1.51%   (paper: 10% of 15% = 1.5%)
```

{{FIG:p3_stats|The procedure, measured. Every share lands within 0.1 points of the paper's numbers (the dashed lines): 15% chosen; of those 80% [MASK], 10% random and 10% unchanged; 1.5% of all tokens replaced at random.}}

How does this compare with Google's released code (`create_pretraining_data.py` in the BERT repository)? I read it line by line. It does the same thing, with three small differences that the paper does not mention:

- It picks an **exact count** per sequence, $$\text{round}(0.15 \times \text{length})$$, which for 128 tokens is 19. Re-running its logic on the same text chose exactly the same number of tokens as above, 39,406.
- It has a **cap**, `max_predictions_per_seq` (20 in the README's example for length 128). The README says to set it to about length × 15%, so the cap and the rate agree.
- It **never** masks `[CLS]` or `[SEP]`, and the random replacement is drawn from the **whole** vocabulary file, special tokens included.

### "my dog is hairy" on the real model

Now the three cases of Appendix A.1, on the real `bert-base-uncased`. One honest change: the paper writes the sentence without a full stop. Without it, the model spends its guess at position 4 on the missing full stop (real sentences end with punctuation), so I added one.

```text
3. "my dog is hairy." with token 4 chosen (Appendix A.1). What the model predicts at position 4:
   input: [CLS] my dog is [MASK] . [SEP]
     P(hairy) = 1.7e-05   P(apple) = 1.5e-06   top 5: dead 0.131, here 0.096, fine 0.080, gone 0.069, missing 0.036
   input: [CLS] my dog is apple . [SEP]
     P(hairy) = 2.3e-10   P(apple) = 0.9997   top 5: apple 1.000, apples 0.000, oak 0.000, orchard 0.000, orange 0.000
   input: [CLS] my dog is hairy . [SEP]
     P(hairy) = 0.9999   P(apple) = 9.0e-10   top 5: hairy 1.000, furry 0.000, shaggy 0.000, ugly 0.000, covered 0.000
```

This short example teaches more than it seems:

- **[MASK]:** "hairy" gets a probability of only 0.000017. That is correct behaviour: nothing in "my dog is ___." points to "hairy". The model offers sensible guesses ("dead", "here", "fine").
- **Random word:** the model believes "apple" (0.9997). In a four-word sentence there is no context to overrule it, and in pre-training only 1.5% of all tokens were random replacements, so a visible word is almost always the real one.
- **Unchanged:** "hairy" gets 0.9999. This is the "bias towards the actual observed word" that the appendix describes.

### How often does BERT recover the original word?

On a short sentence the model has little to go on. On real paragraphs it has a lot. I masked 400 of the 128-token sequences above with the same procedure and checked how often the real model's top guess is the original token:

```text
4. top-1 accuracy of bert-base-uncased on the chosen positions of 400 sequences (7,600 predictions)
   all chosen positions: 61.1%
   [MASK]        58.3%  (3,555 of 6,095)
   random token  47.7%  (368 of 771)
   unchanged     97.7%  (717 of 734)
```

{{FIG:p3_accuracy|Top-1 accuracy of the real model on 7,600 chosen positions, split by what the position showed.}}

With a whole paragraph of context, BERT finds the hidden word 58.3% of the time, out of 30,522 possible tokens. And it corrects almost half of the random replacements (47.7%), which is exactly the skill the 10% random case trains. One caution: WikiText is made of Wikipedia articles, and BERT was pre-trained on Wikipedia, so the model may have seen some of this text. These numbers show the mechanism, not a clean test score.

### The cost: only 15% of tokens teach anything

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.1 · pages 12 and 13
> [![Compared to standard language model training, the masked LM only makes predictions on 15 percent of tokens in each batch, which suggests that more pre-training steps may be required for the model to converge](/img/papers/bert/p3-a1-cost.png)](/img/papers/bert/p3-a1-cost.png)
> [![In Section C.1 we demonstrate that MLM does converge marginally slower than a left-to-right model, which predicts every token, but the empirical improvements of the MLM model far outweigh the increased training cost](/img/papers/bert/p3-a1-cost2.png)](/img/papers/bert/p3-a1-cost2.png)
>
> **Context:** the end of the masked LM part of Appendix A.1, running onto page 13.
>
> **What it says:** a normal language model learns from every token; the masked LM "only make[s] predictions on 15% of tokens in each batch", so it may need more training. It does "converge marginally slower", but "the empirical improvements of the MLM model far outweigh the increased training cost".
>
> **Why it matters:** this is the price of bidirectionality. Part 5 shows the measurement (Figure 5), and Part 6 shows that later work (ELECTRA) attacked exactly this waste.

> [!DEFINITION] Converge
> A model has converged when more training stops improving it much. "Converges slower" means it needs more steps to reach the same quality.

## Task 2: next sentence prediction {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1, Task #2 · page 4
> [![Task 2, Next Sentence Prediction: question answering and natural language inference are based on understanding the relationship between two sentences, which is not directly captured by language modeling. When choosing sentences A and B, 50 percent of the time B is the actual next sentence that follows A, labeled IsNext, and 50 percent of the time it is a random sentence from the corpus, labeled NotNext. C is used for next sentence prediction](/img/papers/bert/p3-nsp.png)](/img/papers/bert/p3-nsp.png)
>
> **Context:** the second pre-training task, right after the masked LM.
>
> **What it says:** many tasks, like question answering and natural language inference, depend on "the relationship between two sentences", which language modelling does not teach. So BERT also learns a yes/no task: "50% of the time B is the actual next sentence that follows A (labeled as IsNext), and 50% of the time it is a random sentence from the corpus (labeled as NotNext)". The vector $$C$$ makes the decision.
>
> **Why it matters:** it gives the `[CLS]` position a job during pre-training, and it practises the "two pieces of text" format that many tasks use.

> [!DEFINITION] Binary classification
> Picking one of two answers. The paper calls NSP "binarized": IsNext or NotNext.

> [!DEFINITION] The vector C
> The final output vector of the `[CLS]` token, the first token of every input (Part 2). BERT uses it as a summary of the whole input for classification.

{{FIG:p3_nsp|How one next-sentence example is made. Sentence A comes from a document. Half of the time B is the real next sentence (IsNext); half of the time it is a sentence from a random document (NotNext). The vector C of the [CLS] token makes the two-way decision.}}

The NSP loss is cross-entropy over two classes:

$$
\mathcal{L}_{\text{NSP}} = -\log P(y \mid C), \qquad P(\cdot \mid C) = \operatorname{softmax}(W_{\text{NSP}}\, C)
$$

where:

- $$y$$ is the true label, IsNext or NotNext;
- $$C$$ is the final vector of `[CLS]`, with 768 numbers in BERT-base;
- $$W_{\text{NSP}}$$ is a small learned matrix with 2 rows (one score per class) and 768 columns.

(In the released model, $$C$$ first passes through one extra layer, called the **pooler**, a 768-by-768 dense layer with a tanh function, and the classifier has a bias. The idea is the same.)

Appendix A.1 shows two examples:

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.1 · page 13
> [![Two next sentence prediction examples. Input: CLS the man went to MASK store SEP he bought a gallon MASK milk SEP, Label IsNext. Input: CLS the man MASK to the store SEP penguin MASK are flight less birds SEP, Label NotNext](/img/papers/bert/p3-a1-nsp.png)](/img/papers/bert/p3-a1-nsp.png)
>
> **Context:** the second half of Appendix A.1.
>
> **What it says:** "the man went to [MASK] store" followed by "he bought a gallon [MASK] milk" is IsNext. "the man [MASK] to the store" followed by "penguin [MASK] are flight ##less birds" is NotNext.
>
> **Why it matters:** notice that both tasks run on the same input at the same time: the sentences are already masked. `##less` is a WordPiece fragment (Part 2): "flightless" was split into "flight" and "##less".

### The real NSP head

The released `bert-base-uncased` still contains its trained NSP layer, so we can ask it. Here are the paper's two examples and four pairs of my own:

```text
1. the pre-trained NSP head on single pairs: probability of IsNext
    0.9999  [paper A.1, labelled IsNext]  A: the man went to [MASK] store  |  B: he bought a gallon [MASK] milk
    0.0011  [paper A.1, labelled NotNext]  A: the man [MASK] to the store  |  B: penguin [MASK] are flight ##less birds
    1.0000  [ours, true next]  A: she opened the fridge.  |  B: there was nothing left but an old lemon.
   3.5e-06  [ours, random]  A: she opened the fridge.  |  B: the treaty was signed in 1648 by both parties.
    1.0000  [ours, true next]  A: the match was delayed by rain.  |  B: play finally started two hours late.
   1.1e-05  [ours, random]  A: the match was delayed by rain.  |  B: photosynthesis turns light into chemical energy.
```

All six are right, and very confident. The paper reports how good the final model is in a footnote:

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1, footnotes 5 and 6 · page 4
> [![Footnote 5: the final model achieves 97 to 98 percent accuracy on NSP. Footnote 6: the vector C is not a meaningful sentence representation without fine-tuning, since it was trained with NSP](/img/papers/bert/p3-footnotes.png)](/img/papers/bert/p3-footnotes.png)
>
> **Context:** two footnotes attached to the NSP paragraph.
>
> **What it says:** footnote 5: "The final model achieves 97%-98% accuracy on NSP." Footnote 6: "The vector C is not a meaningful sentence representation without fine-tuning, since it was trained with NSP."
>
> **Why it matters:** footnote 5 says the task is easy for the model. Footnote 6 is a warning that many people later ignored: do not use raw $$C$$ as a sentence embedding.

**Checking footnote 5.** I built 1,000 pairs from real Wikipedia text: for 500, B is the true next sentence; for 500, B is a random sentence from a different article. One sentence on each side, no masks.

```text
2. NSP accuracy on 1000 real sentence pairs from 61 WikiText-103 test articles (one sentence each side, no masks)
   IsNext pairs:  486 of 500 right (97.2%)
   NotNext pairs: 478 of 500 right (95.6%)
   overall:       96.4%
```

96.4% is close to the paper's 97% to 98%, but the two numbers are not the same test. The paper does not say what data its figure comes from, and its "sentences" are long spans of text, while mine are single sentences cut by a simple splitter. Treat it as "the same ballpark", nothing more. Note also that a random sentence from another article is usually about a different topic, which makes NotNext easy to spot. Part 6 comes back to this.

**Checking footnote 6.** If $$C$$ were a good summary of a sentence's meaning, similar sentences would get similar vectors. I compared the raw $$C$$ vectors with **cosine similarity**:

> [!DEFINITION] Cosine similarity
> A number between -1 and 1 that says how much two vectors point the same way. 1 means the same direction. It is the standard way to compare sentence embeddings.

```text
3. cosine similarity of the raw final [CLS] vectors (no fine-tuning)
   0.923  related    A man is playing a guitar on stage.  |  A musician performs a song for the crowd.
   0.909  related    The stock market fell sharply today.  |  Share prices dropped a lot this afternoon.
   0.766  unrelated  A man is playing a guitar on stage.  |  The stock market fell sharply today.
   0.807  unrelated  Penguins cannot fly.  |  The invoice is due next Monday.
   0.977  opposite   I loved this movie.  |  I hated this movie.
```

{{FIG:p3_cls_cos|Footnote 6, measured. Raw [CLS] vectors of any two sentences point in nearly the same direction. "I loved this movie" and "I hated this movie" get the highest similarity of all.}}

Every pair scores above 0.76, and the two sentences with **opposite** meanings score highest of all (0.977). Related pairs do score a bit higher than unrelated ones, but the gaps are small and unreliable. Footnote 6 is right: without fine-tuning, $$C$$ is not a meaning vector. (Part 6 shows models fine-tuned specially to fix this.)

### NSP and earlier work {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1 · page 5
> [![The NSP task is closely related to representation-learning objectives used in Jernite et al. 2017 and Logeswaran and Lee 2018. However, in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters](/img/papers/bert/p3-nsp-prior.png)](/img/papers/bert/p3-nsp-prior.png)
>
> **Context:** the paragraph after Task #2, at the top of page 5.
>
> **What it says:** the idea of learning from "which sentence comes next" is not new (Jernite et al., 2017; Logeswaran and Lee, 2018). The difference: earlier work transferred "only sentence embeddings" to new tasks, while "BERT transfers all parameters".
>
> **Why it matters:** this is the fine-tuning idea from Part 1 again. The whole network is reused, not just one vector per sentence.

## The pre-training data {§3.1}

> [!PAPER] Devlin et al. (2018), BERT · Section 3.1 · page 5
> [![Pre-training data: the BooksCorpus, 800 million words, and English Wikipedia, 2,500 million words. For Wikipedia only the text passages are extracted, ignoring lists, tables and headers. It is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark in order to extract long contiguous sequences](/img/papers/bert/p3-data.png)](/img/papers/bert/p3-data.png)
>
> **Context:** the last paragraph of Section 3.1.
>
> **What it says:** two sources: "the BooksCorpus (800M words)" and "English Wikipedia (2,500M words)". From Wikipedia, "only the text passages", without "lists, tables, and headers". And "it is critical to use a document-level corpus rather than a shuffled sentence-level corpus such as the Billion Word Benchmark".
>
> **Why it matters:** 800 million plus 2,500 million is the 3.3 billion words in the training arithmetic below. And the last sentence explains why the *kind* of text matters, not just the amount.

> [!DEFINITION] Corpus
> A large collection of text used for training. Plural: corpora.

> [!DEFINITION] BooksCorpus
> A collection of books put together for research by Zhu et al. (2015). OpenAI GPT was trained on it too (Appendix A.4).

Why must the text be **document-level**? The Billion Word Benchmark is a big set of single sentences in **shuffled** order. In it, the sentence after "She opened the fridge." is some unrelated sentence from somewhere else. That makes next sentence prediction impossible to learn, and it stops the model from ever seeing long stretches of connected text. BERT needs sequences of up to 512 tokens that really belong together, so it needs whole documents.

## The pre-training procedure {§A.2}

Appendix A.2 gives the exact recipe. First, how one training example is built.

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.2 · page 13
> [![A.2 Pre-training Procedure: to generate each training input sequence, sample two spans of text from the corpus, called sentences even though they are typically much longer. The first receives the A embedding, the second the B embedding. 50 percent of the time B is the actual next sentence. The combined length is at most 512 tokens. The LM masking is applied after WordPiece tokenization with a uniform masking rate of 15 percent, and no special consideration given to partial word pieces](/img/papers/bert/p3-a2-data.png)](/img/papers/bert/p3-a2-data.png)
>
> **Context:** the start of Appendix A.2.
>
> **What it says:** each example has two "sentences" that are really spans of text, "typically much longer than single sentences". Together they are at most 512 tokens. Masking is done "after WordPiece tokenization", at 15%, with "no special consideration given to partial word pieces".
>
> **Why it matters:** the last detail means a word like "flightless" can be half-masked ("flight [MASK]"). Then predicting "##less" from "flight" is easy, which is a weaker exercise.

Google noticed this too. In May 2019 the BERT repository added **Whole Word Masking** models (its README: "New May 31st, 2019: Whole Word Masking Models"), which always mask every piece of a word together. The released masking code has a switch for it, and its comment notes that the training itself does not change: each piece is still predicted on its own.

> [!DEFINITION] Span
> A stretch of consecutive text: a few words, a sentence or several sentences.

Next, the training run itself. The sentence starts at the bottom of the left column:

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.2 · page 13
> [![We train with batch size of 256 sequences, 256 sequences times 512 tokens equals 128,000 tokens per batch, for 1,000,000 steps, which is approximately 40](/img/papers/bert/p3-a2-train0.png)](/img/papers/bert/p3-a2-train0.png)
> [![epochs over the 3.3 billion word corpus. Adam with learning rate 1e-4, beta1 0.9, beta2 0.999, L2 weight decay of 0.01, learning rate warmup over the first 10,000 steps, and linear decay of the learning rate. Dropout probability of 0.1 on all layers. A gelu activation rather than the standard relu, following OpenAI GPT. The training loss is the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood](/img/papers/bert/p3-a2-train.png)](/img/papers/bert/p3-a2-train.png)
>
> **Context:** the second paragraph of Appendix A.2, which continues at the top of the right column.
>
> **What it says:** 256 sequences of 512 tokens per batch ("128,000 tokens/batch"), 1,000,000 steps, "approximately 40 epochs over the 3.3 billion word corpus". Adam with learning rate 1e-4, β₁ = 0.9, β₂ = 0.999, "L2 weight decay of 0.01", warmup over 10,000 steps then linear decay, dropout 0.1, GELU. The loss is "the sum of the mean masked LM likelihood and the mean next sentence prediction likelihood".
>
> **Why it matters:** this is the whole training recipe in a few lines. Let us go through it one item at a time.

> [!DEFINITION] Batch, step and epoch
> A **batch** is the group of examples the model looks at before one update of its weights. One update is a **step**. An **epoch** is one full pass over the whole training data.

### The batch and the 40 epochs

Let us check the arithmetic (real output of [`bert_part3_schedule.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/bert/bert_part3_schedule.py)):

```text
batch: 256 sequences x 512 tokens = 131,072 tokens (the paper rounds this to 128,000)
1,000,000 steps x 128,000 tokens = 128,000,000,000 tokens
divided by the 3.3 billion words of BooksCorpus + Wikipedia = 38.8 passes  ("approximately 40 epochs")
with the exact 131,072 tokens per batch: 39.7 passes
```

So "approximately 40 epochs" checks out. Two honest caveats about this calculation, both from the paper itself. It divides **tokens** by **words**, and WordPiece makes slightly more tokens than words. And it assumes every step uses 512 tokens, but (as we will see in a moment) 90% of the steps used sequences of 128 tokens, and the paper does not say what batch size was used then. So the true number of tokens seen is not stated anywhere in the paper; "about 40 epochs" is the paper's own rough figure.

### Adam, warmup and decay

> [!DEFINITION] Adam
> The most common method for updating weights. For each weight, it keeps a running average of recent gradients ($$\beta_1 = 0.9$$ sets how long that memory is) and of their squares ($$\beta_2 = 0.999$$), and uses both to choose a sensible step size for that weight.

> [!DEFINITION] Learning rate, warmup and decay
> The **learning rate** scales how big each update is. **Warmup** starts it at 0 and raises it slowly, because big updates on a random, untrained model can wreck it. **Decay** lowers it again towards the end, so training settles down.

The released code (`optimization.py`) computes the learning rate at step $$s$$ like this:

$$
\eta(s) = \begin{cases} \eta_{\max} \cdot \dfrac{s}{10{,}000} & \text{if } s < 10{,}000 \\[1.2ex] \eta_{\max} \cdot \left(1 - \dfrac{s}{1{,}000{,}000}\right) & \text{otherwise} \end{cases}
$$

where $$\eta$$ (the Greek letter "eta") is the learning rate, $$\eta_{\max} = 10^{-4}$$ is the peak from the paper, and $$s$$ is the step number.

{{FIG:p3_lr|The learning-rate schedule as the released code computes it: a straight climb from 0 over the first 10,000 steps, then a straight fall to 0 at step 1,000,000.}}

A small detail you only see in code: the decay is counted from step 0, so when warmup ends at step 10,000 the rate is already $$10^{-4} \times 0.99 = 9.9 \times 10^{-5}$$, not quite the peak.

> [!DEFINITION] Weight decay
> A gentle pull of every weight towards 0 at each step, which discourages the model from relying on a few huge weights. 0.01 sets how strong the pull is.

The paper calls it "L2 weight decay". The released optimizer does not add a squared-weight penalty to the loss (a code comment explains that this interacts badly with Adam). Instead it subtracts a small fraction of each weight directly at every update, and it skips LayerNorm weights and biases. This later became known as decoupled weight decay, the "W" in the name of the AdamW optimizer.

> [!DEFINITION] Dropout
> During training, randomly switch off a share of the numbers inside the network (here 10%) at every step. The model cannot rely on any single path, so it generalises better. Dropout is turned off when the model is used.

### GELU instead of ReLU

> [!DEFINITION] Activation function
> A simple function applied to every number inside the network's feed-forward layers. Without one, a stack of layers would collapse into a single matrix multiplication and could learn much less.

The usual choice was **ReLU**, which keeps positive numbers and turns negative ones into 0. BERT uses **GELU** (Hendrycks and Gimpel, 2016), as OpenAI GPT did:

$$
\text{ReLU}(x) = \max(0, x), \qquad \text{GELU}(x) = x \cdot \Phi(x)
$$

where:

- $$x$$ is the input number;
- $$\Phi(x)$$ (the Greek letter "phi") is the probability that a random number from a standard bell curve (a normal distribution with mean 0 and spread 1) is smaller than $$x$$. It goes smoothly from 0 (for very negative $$x$$) to 1 (for very positive $$x$$).

So GELU keeps $$x$$ in proportion to how large $$x$$ is. Big positive inputs pass almost unchanged, big negative inputs become almost 0, and small inputs are scaled down smoothly.

{{FIG:p3_gelu|ReLU and GELU. For large positive inputs they agree. GELU bends smoothly through zero and lets small negative inputs out as small negative numbers.}}

```text
GELU: largest gap between the exact form x*Phi(x) and the tanh formula of the released code: 4.7e-04
   x = -3.0: ReLU +0.000   GELU -0.0040
   x = -1.0: ReLU +0.000   GELU -0.1587
   x = -0.5: ReLU +0.000   GELU -0.1543
   x = +0.0: ReLU +0.000   GELU +0.0000
   x = +0.5: ReLU +0.500   GELU +0.3457
   x = +1.0: ReLU +1.000   GELU +0.8413
   x = +3.0: ReLU +3.000   GELU +2.9960
```

The released code computes $$\Phi$$ with a fast formula based on tanh. It differs from the exact GELU by at most 0.00047.

### One loss for both tasks

The training loss is the sum of the two tasks' average losses:

$$
\mathcal{L} = \mathcal{L}_{\text{MLM}} + \mathcal{L}_{\text{NSP}}
$$

where $$\mathcal{L}_{\text{MLM}}$$ is the mean cross-entropy over all masked positions in the batch and $$\mathcal{L}_{\text{NSP}}$$ is the mean cross-entropy over all sentence pairs. (The paper says "likelihood"; training minimises the negative log-likelihood, which is the same cross-entropy.)

I checked this on a real batch: 8 sentence pairs from Wikipedia (half IsNext, half NotNext), masked with the procedure above, through `BertForPreTraining`:

```text
5. pre-training loss on one real batch of 8 sentence pairs (58 masked positions)
   mean masked-LM loss       2.1302
   mean next-sentence loss   0.0001
   sum                       2.1303
   loss computed by BertForPreTraining: 2.1303
```

The two parts add up exactly to the loss the library computes. Notice how lopsided they are: next sentence prediction is nearly solved (0.0001), while filling in blanks is still hard (2.1302). After pre-training, almost all of the remaining learning signal comes from the masked LM.

### Hardware and the two sequence lengths

> [!PAPER] Devlin et al. (2018), BERT · Appendix A.2 · page 13
> [![Training of BERT-base was performed on 4 Cloud TPUs in Pod configuration, 16 TPU chips total; BERT-large on 16 Cloud TPUs, 64 TPU chips total. Each pre-training took 4 days. Longer sequences are disproportionately expensive because attention is quadratic to the sequence length. We pre-train with sequence length of 128 for 90 percent of the steps, then the rest 10 percent of the steps with sequence of 512 to learn the positional embeddings](/img/papers/bert/p3-a2-tpu.png)](/img/papers/bert/p3-a2-tpu.png)
>
> **Context:** the last two paragraphs of Appendix A.2.
>
> **What it says:** BERT-base trained on "16 TPU chips total", BERT-large on "64 TPU chips total", and "each pre-training took 4 days". Because "attention is quadratic to the sequence length", they used "sequence length of 128 for 90% of the steps", and length 512 only for the last 10%, "to learn the positional embeddings".
>
> **Why it matters:** a practical trick that most later models copied: train short first, long at the end.

> [!DEFINITION] TPU
> Tensor Processing Unit: a chip designed by Google to run neural-network maths fast, similar in purpose to a GPU. A "pod" links many of them together.

> [!DEFINITION] Quadratic
> Growing with the square. If the sequence is 4 times longer, attention compares every token with every other token, so there are $$4^2 = 16$$ times as many comparisons.

```text
attention scores per sequence: 128^2 = 16,384, 512^2 = 262,144 -> 16x more for a 4x longer sequence
per token: 4x more attention work at length 512
```

The other parts of the model (the feed-forward layers) cost the same per token at any length, so attention is the part that blows up. Training mostly at length 128 saves a lot. But the position embeddings for positions 128 to 511 (Part 2) are only trained in the final 10% of steps, which is the reason for that last phase.

> [!TAKEAWAYS] Key takeaways
> - A normal language model cannot simply look both ways: with two or more layers, **each word can see itself through a neighbour**. In a real experiment, a two-layer "both sides" model reached a loss of 0.90 on unseen text by cheating, against 4.37 for the honest one-layer version.
> - **Masked LM:** choose 15% of WordPiece tokens, hide them, and predict only them with cross-entropy over the whole vocabulary. Hidden words cannot leak, so every layer can use both sides.
> - **80/10/10:** a chosen token becomes [MASK] 80% of the time, a random token 10%, and stays the same 10%. Measured on 261,324 real tokens: 15.08% chosen, 80.06 / 9.98 / 9.96 split, 1.51% random overall.
> - The real model recovers 58.3% of masked words and 47.7% of randomly replaced ones on Wikipedia text, and copies unchanged words 97.7% of the time.
> - **Next sentence prediction:** half real next sentences, half random ones, decided from the [CLS] vector C. The released model scored 96.4% on 1,000 real pairs (paper: 97% to 98%). Raw C is **not** a sentence meaning vector: "I loved this movie" and "I hated this movie" scored 0.977.
> - **Data:** BooksCorpus (800M words) and English Wikipedia (2,500M words), as whole documents.
> - **Recipe:** batch 256 × 512 tokens, 1,000,000 steps (about 40 epochs), Adam at 1e-4 with 10,000 warmup steps and linear decay, weight decay 0.01, dropout 0.1, GELU; length 128 for 90% of steps and 512 for the rest; 4 days on 16 TPU chips (base) or 64 (large). The loss is MLM plus NSP.

**Next, in Part 4:** fine-tuning. How the same pre-trained BERT becomes a classifier, a question-answering system and a multiple-choice solver, the paper's results tables, and a real fine-tuning run.

<details>
<summary>Run it yourself</summary>

The scripts behind every number in this part are in [`code/papers/bert/`](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/bert). They need Python with `torch`, `transformers` and `datasets`, and download `bert-base-uncased` (about 440 MB) and the WikiText data the first time.

```bash
pip install torch transformers datasets
python bert_part3_seeitself.py   # the three tiny models (about 6 minutes per model on a laptop GPU)
python bert_part3_mlm.py         # masking statistics, "my dog is hairy", MLM accuracy, the loss on a batch
python bert_part3_nsp.py         # the NSP head, 1,000 real pairs, raw [CLS] similarities
python bert_part3_schedule.py    # batch arithmetic, learning rate, GELU, attention cost
```

<figure><img src="/img/papers/bert/part3-seeitself-run.png" alt="Terminal output of bert_part3_seeitself.py: training and validation losses of the left-to-right model, the one-layer both-sides model and the two-layer both-sides model" loading="lazy" /><figcaption>The real output of bert_part3_seeitself.py. The run times depend on what else the GPU is doing.</figcaption></figure>

<figure><img src="/img/papers/bert/part3-mlm-run.png" alt="Terminal output of bert_part3_mlm.py: masking statistics, the released code comparison, the three my dog is hairy cases, top-1 accuracy by case and the two parts of the pre-training loss" loading="lazy" /><figcaption>The real output of bert_part3_mlm.py.</figcaption></figure>

<figure><img src="/img/papers/bert/part3-nsp-run.png" alt="Terminal output of bert_part3_nsp.py: IsNext probabilities for six pairs, accuracy on 1,000 real pairs, and cosine similarities of raw CLS vectors" loading="lazy" /><figcaption>The real output of bert_part3_nsp.py.</figcaption></figure>

<figure><img src="/img/papers/bert/part3-schedule-run.png" alt="Terminal output of bert_part3_schedule.py: batch and epoch arithmetic, the learning rate at several steps, GELU and ReLU values and the attention cost ratio" loading="lazy" /><figcaption>The real output of bert_part3_schedule.py.</figcaption></figure>

</details>

## References

1. J. Devlin, M.-W. Chang, K. Lee, K. Toutanova. [*BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*](https://arxiv.org/abs/1810.04805). NAACL 2019.
2. Google Research. [BERT code and models](https://github.com/google-research/bert), including `create_pretraining_data.py`, `optimization.py` and the Whole Word Masking models.
3. W. L. Taylor. *Cloze procedure: A new tool for measuring readability*. Journalism Bulletin, 1953.
4. P. Vincent, H. Larochelle, Y. Bengio, P.-A. Manzagol. *Extracting and composing robust features with denoising autoencoders*. ICML 2008.
5. Y. Jernite, S. R. Bowman, D. Sontag. [*Discourse-based objectives for fast unsupervised sentence representation learning*](https://arxiv.org/abs/1705.00557). 2017.
6. L. Logeswaran, H. Lee. [*An efficient framework for learning sentence representations*](https://arxiv.org/abs/1803.02893). ICLR 2018.
7. Y. Zhu et al. *Aligning books and movies: Towards story-like visual explanations by watching movies and reading books* (BooksCorpus). ICCV 2015.
8. D. Hendrycks, K. Gimpel. [*Gaussian Error Linear Units (GELUs)*](https://arxiv.org/abs/1606.08415). 2016.
9. S. Merity, C. Xiong, J. Bradbury, R. Socher. [*Pointer Sentinel Mixture Models*](https://arxiv.org/abs/1609.07843) (the WikiText data). 2016.
