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

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="Why a deep model cannot simply condition on both sides. Even if position 2 never looks at its own word directly, in layer 1 a neighbour reads the word, and in layer 2 position 2 reads the neighbour. With two or more layers, every word can indirectly see itself."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Predict "cat" (position 2) from every other word, with two layers</text><text class="t-tick" x="20.0" y="270.0" text-anchor="start">input tokens</text><text class="t-tick" x="20.0" y="180.0" text-anchor="start">layer 1</text><text class="t-tick" x="20.0" y="90.0" text-anchor="start">layer 2</text><rect class="box" x="160.0" y="250.0" width="80.0" height="32.0" rx="6"/><text class="t-note" x="200.0" y="270.5" text-anchor="middle">the</text><rect class="box" x="160.0" y="160.0" width="80.0" height="32.0" rx="8"/><rect class="box" x="160.0" y="70.0" width="80.0" height="32.0" rx="8"/><rect class="box-mask" x="290.0" y="250.0" width="80.0" height="32.0" rx="6"/><text class="t-note" x="330.0" y="270.5" text-anchor="middle">cat</text><rect class="box" x="290.0" y="160.0" width="80.0" height="32.0" rx="8"/><rect class="box-on" x="290.0" y="70.0" width="80.0" height="32.0" rx="8"/><rect class="box" x="420.0" y="250.0" width="80.0" height="32.0" rx="6"/><text class="t-note" x="460.0" y="270.5" text-anchor="middle">sat</text><rect class="box" x="420.0" y="160.0" width="80.0" height="32.0" rx="8"/><rect class="box" x="420.0" y="70.0" width="80.0" height="32.0" rx="8"/><rect class="box" x="550.0" y="250.0" width="80.0" height="32.0" rx="6"/><text class="t-note" x="590.0" y="270.5" text-anchor="middle">down</text><rect class="box" x="550.0" y="160.0" width="80.0" height="32.0" rx="8"/><rect class="box" x="550.0" y="70.0" width="80.0" height="32.0" rx="8"/><text class="t-tick" x="330.0" y="91.0" text-anchor="middle">predict ?</text><line class="bad-line" x1="340.0" y1="250.0" x2="450.0" y2="192.0"/><text class="t-bad" x="506.0" y="216.0" text-anchor="start">step 1: "sat" reads "cat"</text><text class="t-muted" x="506.0" y="232.0" text-anchor="start">(allowed: it is not its own word)</text><line class="bad-line" x1="450.0" y1="160.0" x2="340.0" y2="102.0"/><text class="t-bad" x="506.0" y="116.0" text-anchor="start">step 2: position 2 reads "sat",</text><text class="t-bad" x="506.0" y="132.0" text-anchor="start">which already contains "cat"</text><line class="edge-dim" x1="200.0" y1="250.0" x2="330.0" y2="192.0"/><line class="edge-dim" x1="590.0" y1="250.0" x2="330.0" y2="192.0"/><text class="t-muted" x="330.0" y="300.0" text-anchor="middle">the answer leaks back in through a neighbour</text></svg><figcaption>How a word sees itself through two layers. Position 2 must predict "cat" and never looks at "cat" directly. But in layer 1, the neighbour "sat" reads "cat". In layer 2, position 2 reads "sat", and the answer comes back in.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="Training loss of the three tiny models. The left-to-right model and the one-layer both-sides model learn slowly and honestly. The two-layer both-sides model drops far lower, on unseen text too, because it reads its own answer back."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="260.0" x2="534" y2="260.0"/><text class="t-tick" x="56.0" y="264.0" text-anchor="end">0</text><line class="grid" x1="64" y1="214.0" x2="534" y2="214.0"/><text class="t-tick" x="56.0" y="218.0" text-anchor="end">2</text><line class="grid" x1="64" y1="168.0" x2="534" y2="168.0"/><text class="t-tick" x="56.0" y="172.0" text-anchor="end">4</text><line class="grid" x1="64" y1="122.0" x2="534" y2="122.0"/><text class="t-tick" x="56.0" y="126.0" text-anchor="end">6</text><line class="grid" x1="64" y1="76.0" x2="534" y2="76.0"/><text class="t-tick" x="56.0" y="80.0" text-anchor="end">8</text><line class="grid" x1="64" y1="30.0" x2="534" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">10</text><text class="t-tick" x="64.0" y="278.0" text-anchor="middle">0</text><text class="t-tick" x="181.5" y="278.0" text-anchor="middle">750</text><text class="t-tick" x="299.0" y="278.0" text-anchor="middle">1,500</text><text class="t-tick" x="416.5" y="278.0" text-anchor="middle">2,250</text><text class="t-tick" x="534.0" y="278.0" text-anchor="middle">3,000</text><line class="axis" x1="64" y1="260" x2="534" y2="260"/><polyline class="l2" points="64.2,46.0 71.8,103.9 79.7,111.9 87.5,115.3 95.3,121.8 103.2,123.1 111.0,124.2 118.8,127.2 126.7,126.4 134.5,130.5 142.3,132.2 150.2,131.4 158.0,134.9 165.8,134.1 173.7,137.2 181.5,138.3 189.3,137.2 197.2,136.0 205.0,140.5 212.8,140.4 220.7,141.2 228.5,139.2 236.3,142.4 244.2,143.0 252.0,142.6 259.8,142.4 267.7,144.4 275.5,144.1 283.3,143.6 291.2,146.5 299.0,146.7 306.8,147.7 314.7,146.9 322.5,145.8 330.3,145.6 338.2,149.2 346.0,147.7 353.8,146.8 361.7,150.4 369.5,147.3 377.3,149.4 385.2,150.9 393.0,149.6 400.8,150.7 408.7,151.4 416.5,149.9 424.3,151.4 432.2,150.0 440.0,152.6 447.8,152.4 455.7,151.1 463.5,152.8 471.3,152.3 479.2,151.2 487.0,152.5 494.8,150.3 502.7,153.8 510.5,151.9 518.3,152.4 526.2,153.0 534.0,153.2"/><polyline class="l3" points="64.2,48.4 71.8,99.3 79.7,107.6 87.5,107.2 95.3,111.7 103.2,109.5 111.0,110.3 118.8,110.9 126.7,112.6 134.5,121.7 142.3,127.0 150.2,128.7 158.0,135.1 165.8,134.5 173.7,139.8 181.5,141.8 189.3,141.8 197.2,141.3 205.0,147.8 212.8,147.3 220.7,147.4 228.5,147.0 236.3,151.5 244.2,151.9 252.0,151.4 259.8,152.0 267.7,153.9 275.5,154.8 283.3,153.9 291.2,156.4 299.0,157.7 306.8,159.5 314.7,158.2 322.5,156.9 330.3,158.3 338.2,162.2 346.0,160.4 353.8,159.7 361.7,163.3 369.5,159.8 377.3,162.5 385.2,165.4 393.0,163.7 400.8,165.3 408.7,166.1 416.5,164.2 424.3,165.6 432.2,165.4 440.0,168.1 447.8,166.8 455.7,164.4 463.5,167.8 471.3,166.9 479.2,166.2 487.0,167.6 494.8,164.6 502.7,170.0 510.5,166.8 518.3,168.2 526.2,169.1 534.0,169.3"/><polyline class="l1" points="64.2,49.1 71.8,105.5 79.7,107.8 87.5,107.8 95.3,113.9 103.2,119.3 111.0,123.2 118.8,138.0 126.7,143.3 134.5,162.7 142.3,174.2 150.2,184.6 158.0,193.8 165.8,195.4 173.7,201.5 181.5,202.5 189.3,206.0 197.2,199.4 205.0,173.9 212.8,204.0 220.7,212.1 228.5,211.3 236.3,216.4 244.2,218.4 252.0,217.1 259.8,222.0 267.7,221.8 275.5,224.8 283.3,225.1 291.2,219.7 299.0,217.8 306.8,228.9 314.7,230.1 322.5,232.0 330.3,233.0 338.2,233.8 346.0,234.8 353.8,235.8 361.7,236.5 369.5,234.2 377.3,236.3 385.2,235.8 393.0,236.5 400.8,239.9 408.7,239.7 416.5,239.8 424.3,239.4 432.2,240.8 440.0,241.0 447.8,234.0 455.7,240.9 463.5,244.1 471.3,244.1 479.2,243.3 487.0,244.3 494.8,218.5 502.7,183.5 510.5,221.3 518.3,230.4 526.2,238.0 534.0,239.9"/><rect class="s2" x="548" y="147.2" width="10" height="10" rx="2"/><text class="t-note" x="564.0" y="156.2" text-anchor="start">left-to-right, 2 layers</text><text class="t-tick" x="564.0" y="172.2" text-anchor="start">train 4.65, unseen text 5.05</text><rect class="s3" x="548" y="181.2" width="10" height="10" rx="2"/><text class="t-note" x="564.0" y="190.2" text-anchor="start">both sides, 1 layer</text><text class="t-tick" x="564.0" y="206.2" text-anchor="start">train 3.95, unseen text 4.37</text><rect class="s1" x="548" y="233.9" width="10" height="10" rx="2"/><text class="t-note" x="564.0" y="242.9" text-anchor="start">both sides, 2 layers</text><text class="t-tick" x="564.0" y="258.9" text-anchor="start">train 0.87, unseen text 0.90</text><text class="t-tick" x="299.0" y="300.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">loss (lower is better)</text></svg><figcaption>Training loss of the three tiny models. A and B learn slowly and honestly. C's loss drops far lower, and stays low on text it never trained on, because it reads its own answer back through a neighbour.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="The masked language model. The whole sentence goes in, with the chosen token hidden. Only the output vectors of the chosen positions (here T4) go through a softmax over the vocabulary, and only they count in the loss."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Masked LM: predict only the chosen positions</text><rect class="box-4" x="40.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="82.0" y="269.5" text-anchor="middle">[CLS]</text><rect class="box" x="134.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="176.0" y="269.5" text-anchor="middle">my</text><rect class="box" x="228.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="270.0" y="269.5" text-anchor="middle">dog</text><rect class="box" x="322.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="364.0" y="269.5" text-anchor="middle">is</text><rect class="box-mask" x="416.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="458.0" y="269.5" text-anchor="middle">[MASK]</text><rect class="box" x="510.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="552.0" y="269.5" text-anchor="middle">.</text><rect class="box-4" x="604.0" y="250.0" width="84.0" height="30.0" rx="6"/><text class="t-tick" x="646.0" y="269.5" text-anchor="middle">[SEP]</text><rect class="box-1" x="40.0" y="160.0" width="648.0" height="50.0" rx="12"/><text class="t-note" x="364.0" y="190.0" text-anchor="middle">BERT: 12 layers, every token sees every token</text><line class="edge" x1="82.0" y1="248.0" x2="82.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="176.0" y1="248.0" x2="176.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="270.0" y1="248.0" x2="270.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="364.0" y1="248.0" x2="364.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="458.0" y1="248.0" x2="458.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="552.0" y1="248.0" x2="552.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="646.0" y1="248.0" x2="646.0" y2="212.0" marker-end="url(#ah)"/><line class="edge" x1="458.0" y1="158.0" x2="458.0" y2="124.0" marker-end="url(#ah)"/><rect class="box-on" x="418.0" y="92.0" width="80.0" height="30.0" rx="8"/><text class="t-math" x="458.0" y="112.0" text-anchor="middle">T₄</text><line class="edge" x1="500.0" y1="107.0" x2="530.0" y2="86.0" marker-end="url(#ah)"/><rect class="box" x="534.0" y="40.0" width="200.0" height="92.0" rx="10"/><text class="t-tick" x="634.0" y="62.0" text-anchor="middle">softmax over all</text><text class="t-tick" x="634.0" y="80.0" text-anchor="middle">30,522 vocabulary ids</text><text class="t-muted" x="634.0" y="106.0" text-anchor="middle">cross-entropy against</text><text class="t-muted" x="634.0" y="122.0" text-anchor="middle">the original word</text><text class="t-muted" x="176.0" y="150.0" text-anchor="middle">no loss</text><text class="t-muted" x="270.0" y="150.0" text-anchor="middle">no loss</text><text class="t-muted" x="364.0" y="150.0" text-anchor="middle">no loss</text><text class="t-muted" x="552.0" y="150.0" text-anchor="middle">no loss</text></svg><figcaption>The masked language model. The whole sentence goes in, with the chosen token hidden. Only the output vector of the hidden position (here T₄) goes through a softmax over the vocabulary and counts in the loss.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 290" role="img" aria-label="The 80/10/10 rule of Section 3.1 and Appendix A.1. A chosen token becomes [MASK] 80% of the time, a random word 10% of the time, and stays itself 10% of the time. The model must predict the original word in every case."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">For each chosen position (15% of tokens), roll the dice once</text><rect class="box-on" x="30.0" y="120.0" width="150.0" height="36.0" rx="6"/><text class="t-note" x="105.0" y="142.5" text-anchor="middle">hairy (chosen)</text><line class="edge" x1="182.0" y1="138.0" x2="236.0" y2="72.0" marker-end="url(#ah)"/><text class="t-val" x="244.0" y="64.0" text-anchor="start">80%</text><text class="t-note" x="286.0" y="64.0" text-anchor="start">replace with [MASK]</text><rect class="box-mask" x="244.0" y="72.0" width="190.0" height="30.0" rx="6"/><text class="t-mono" x="339.0" y="92.0" text-anchor="middle">my dog is [MASK]</text><text class="t-tick" x="452.0" y="92.0" text-anchor="start">learn to fill blanks</text><line class="edge" x1="182.0" y1="138.0" x2="236.0" y2="146.0" marker-end="url(#ah)"/><text class="t-val" x="244.0" y="138.0" text-anchor="start">10%</text><text class="t-note" x="286.0" y="138.0" text-anchor="start">replace with a random word</text><rect class="box-2" x="244.0" y="146.0" width="190.0" height="30.0" rx="6"/><text class="t-mono" x="339.0" y="166.0" text-anchor="middle">my dog is apple</text><text class="t-tick" x="452.0" y="166.0" text-anchor="start">never fully trust the input</text><line class="edge" x1="182.0" y1="138.0" x2="236.0" y2="220.0" marker-end="url(#ah)"/><text class="t-val" x="244.0" y="212.0" text-anchor="start">10%</text><text class="t-note" x="286.0" y="212.0" text-anchor="start">keep the word unchanged</text><rect class="box-3" x="244.0" y="220.0" width="190.0" height="30.0" rx="6"/><text class="t-mono" x="339.0" y="240.0" text-anchor="middle">my dog is hairy</text><text class="t-tick" x="452.0" y="240.0" text-anchor="start">learn about real, unmasked words</text><text class="t-tick" x="30.0" y="270.0" text-anchor="start">In all three cases the target is the same: predict "hairy" at position 4.</text></svg><figcaption>The 80/10/10 rule. Each chosen token becomes [MASK] 80% of the time, a random word 10% of the time, and stays itself 10% of the time. The target is always the original word.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 260" role="img" aria-label="The masking procedure, measured. Every share lands within 0.1 points of the paper: 15% of tokens are chosen; of those, 80% become [MASK], 10% a random token and 10% stay the same; so 1.5% of all tokens are random replacements. The dashed lines mark the paper's numbers."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Measured over 261,324 word pieces of real Wikipedia text (seed 0)</text><text class="t-tick" x="160.0" y="63.0" text-anchor="end">chosen / all tokens</text><g class="mark"><title>chosen / all tokens: 15.08% (paper: 15%)</title><rect class="s1" x="170" y="50" width="72.4" height="20" rx="3"/></g><line class="base-line" x1="242.0" y1="46" x2="242.0" y2="74"/><text class="t-val" x="250.4" y="64.0" text-anchor="start">15.08%</text><text class="t-tick" x="312.4" y="64.0" text-anchor="start">paper: 15%</text><text class="t-tick" x="160.0" y="103.0" text-anchor="end">[MASK] / chosen</text><g class="mark"><title>[MASK] / chosen: 80.06% (paper: 80%)</title><rect class="s1" x="170" y="90" width="384.3" height="20" rx="3"/></g><line class="base-line" x1="554.0" y1="86" x2="554.0" y2="114"/><text class="t-val" x="562.3" y="104.0" text-anchor="start">80.06%</text><text class="t-tick" x="624.3" y="104.0" text-anchor="start">paper: 80%</text><text class="t-tick" x="160.0" y="143.0" text-anchor="end">random / chosen</text><g class="mark"><title>random / chosen: 9.98% (paper: 10%)</title><rect class="s1" x="170" y="130" width="47.9" height="20" rx="3"/></g><line class="base-line" x1="218.0" y1="126" x2="218.0" y2="154"/><text class="t-val" x="225.9" y="144.0" text-anchor="start">9.98%</text><text class="t-tick" x="287.9" y="144.0" text-anchor="start">paper: 10%</text><text class="t-tick" x="160.0" y="183.0" text-anchor="end">unchanged / chosen</text><g class="mark"><title>unchanged / chosen: 9.96% (paper: 10%)</title><rect class="s1" x="170" y="170" width="47.8" height="20" rx="3"/></g><line class="base-line" x1="218.0" y1="166" x2="218.0" y2="194"/><text class="t-val" x="225.8" y="184.0" text-anchor="start">9.96%</text><text class="t-tick" x="287.8" y="184.0" text-anchor="start">paper: 10%</text><text class="t-tick" x="160.0" y="223.0" text-anchor="end">random / all tokens</text><g class="mark"><title>random / all tokens: 1.51% (paper: 1.5%)</title><rect class="s1" x="170" y="210" width="7.2" height="20" rx="3"/></g><line class="base-line" x1="177.2" y1="206" x2="177.2" y2="234"/><text class="t-val" x="185.2" y="224.0" text-anchor="start">1.51%</text><text class="t-tick" x="247.2" y="224.0" text-anchor="start">paper: 1.5%</text></svg><figcaption>The procedure, measured. Every share lands within 0.1 points of the paper's numbers (the dashed lines): 15% chosen; of those 80% [MASK], 10% random and 10% unchanged; 1.5% of all tokens replaced at random.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 200" role="img" aria-label="How often the real model's top guess is the original word, split by what the chosen position showed. It recovers words hidden by [MASK] most of the time, nearly half of the words replaced by a random token, and almost all unchanged words."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">bert-base-uncased, top-1 accuracy on 7,600 chosen positions</text><text class="t-tick" x="152.0" y="71.0" text-anchor="end">[MASK]</text><g class="mark"><title>[MASK]: 58.3%</title><rect class="s1" x="160.0" y="53.0" width="256.6" height="26.0" rx="3"/></g><text class="t-val" x="422.6" y="71.0" text-anchor="start">58.3%</text><text class="t-tick" x="152.0" y="105.0" text-anchor="end">random token</text><g class="mark"><title>random token: 47.7%</title><rect class="s1" x="160.0" y="87.0" width="210.0" height="26.0" rx="3"/></g><text class="t-val" x="376.0" y="105.0" text-anchor="start">47.7%</text><text class="t-tick" x="152.0" y="139.0" text-anchor="end">unchanged</text><g class="mark"><title>unchanged: 97.7%</title><rect class="s1" x="160.0" y="121.0" width="429.8" height="26.0" rx="3"/></g><text class="t-val" x="595.8" y="139.0" text-anchor="start">97.7%</text><text class="t-tick" x="152.0" y="173.0" text-anchor="end">all chosen</text><g class="mark"><title>all chosen: 61.1%</title><rect class="s1" x="160.0" y="155.0" width="268.6" height="26.0" rx="3"/></g><text class="t-val" x="434.6" y="173.0" text-anchor="start">61.1%</text></svg><figcaption>Top-1 accuracy of the real model on 7,600 chosen positions, split by what the position showed.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="How next sentence prediction examples are made. Sentence A comes from a document; half of the time B is the real next sentence (IsNext), half of the time a sentence from a random document (NotNext). The final vector C of the [CLS] token makes the two-way decision."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Building one next-sentence example</text><rect class="box" x="20.0" y="44.0" width="200.0" height="120.0" rx="10"/><text class="t-tick" x="120.0" y="64.0" text-anchor="middle">document 1</text><rect class="box-1" x="34.0" y="76.0" width="172.0" height="26.0" rx="5"/><text class="t-tick" x="120.0" y="94.0" text-anchor="middle">sentence A</text><rect class="box-3" x="34.0" y="108.0" width="172.0" height="26.0" rx="5"/><text class="t-tick" x="120.0" y="126.0" text-anchor="middle">the real next sentence</text><rect class="box" x="20.0" y="186.0" width="200.0" height="70.0" rx="10"/><text class="t-tick" x="120.0" y="206.0" text-anchor="middle">document 2 (random)</text><rect class="box-2" x="34.0" y="218.0" width="172.0" height="26.0" rx="5"/><text class="t-tick" x="120.0" y="236.0" text-anchor="middle">some other sentence</text><line class="edge-3" x1="208.0" y1="121.0" x2="236.0" y2="146.0"/><text class="t-val" x="226.0" y="112.0" text-anchor="middle">50%</text><line class="edge-2" x1="208.0" y1="231.0" x2="236.0" y2="166.0"/><text class="t-val" x="226.0" y="226.0" text-anchor="middle">50%</text><rect class="box-on" x="236.0" y="140.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="264.0" y="160.0" text-anchor="middle">pick B</text><line class="edge" x1="294.0" y1="140.0" x2="304.0" y2="134.0" marker-end="url(#ah)"/><rect class="box-4" x="306.0" y="116.0" width="56.0" height="30.0" rx="6"/><text class="t-tick" x="334.0" y="135.5" text-anchor="middle">[CLS]</text><rect class="box-1" x="368.0" y="116.0" width="56.0" height="30.0" rx="6"/><text class="t-tick" x="396.0" y="135.5" text-anchor="middle">A</text><rect class="box-4" x="430.0" y="116.0" width="56.0" height="30.0" rx="6"/><text class="t-tick" x="458.0" y="135.5" text-anchor="middle">[SEP]</text><rect class="box" x="492.0" y="116.0" width="56.0" height="30.0" rx="6"/><text class="t-tick" x="520.0" y="135.5" text-anchor="middle">B</text><rect class="box-4" x="554.0" y="116.0" width="56.0" height="30.0" rx="6"/><text class="t-tick" x="582.0" y="135.5" text-anchor="middle">[SEP]</text><rect class="box-1" x="306.0" y="186.0" width="302.0" height="40.0" rx="10"/><text class="t-note" x="457.0" y="211.0" text-anchor="middle">BERT</text><line class="edge" x1="334.0" y1="148.0" x2="334.0" y2="184.0" marker-end="url(#ah)"/><line class="edge" x1="396.0" y1="148.0" x2="396.0" y2="184.0" marker-end="url(#ah)"/><line class="edge" x1="458.0" y1="148.0" x2="458.0" y2="184.0" marker-end="url(#ah)"/><line class="edge" x1="520.0" y1="148.0" x2="520.0" y2="184.0" marker-end="url(#ah)"/><line class="edge" x1="582.0" y1="148.0" x2="582.0" y2="184.0" marker-end="url(#ah)"/><line class="edge" x1="334.0" y1="228.0" x2="334.0" y2="262.0" marker-end="url(#ah)"/><rect class="box-on" x="312.0" y="264.0" width="44.0" height="26.0" rx="6"/><text class="t-math" x="334.0" y="282.0" text-anchor="middle">C</text><line class="edge" x1="358.0" y1="277.0" x2="420.0" y2="277.0" marker-end="url(#ah)"/><rect class="box" x="424.0" y="262.0" width="300.0" height="30.0" rx="8"/><text class="t-tick" x="574.0" y="282.0" text-anchor="middle">IsNext or NotNext? (2 classes)</text></svg><figcaption>How one next-sentence example is made. Sentence A comes from a document. Half of the time B is the real next sentence (IsNext); half of the time it is a sentence from a random document (NotNext). The vector C of the [CLS] token makes the two-way decision.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 240" role="img" aria-label="Footnote 6, measured. Without fine-tuning, the [CLS] vectors of any two sentences point in nearly the same direction. "I loved this movie" and "I hated this movie" score the highest similarity of all."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Cosine similarity of raw [CLS] vectors (1 = same direction)</text><text class="t-tick" x="312.0" y="69.0" text-anchor="end">related: guitar on stage / musician</text><g class="mark"><title>related: guitar on stage / musician: 0.923</title><rect class="s2" x="320.0" y="51.0" width="304.7" height="26.0" rx="3"/></g><text class="t-val" x="630.7" y="69.0" text-anchor="start">0.923</text><text class="t-tick" x="312.0" y="103.0" text-anchor="end">related: stocks fell / prices dropped</text><g class="mark"><title>related: stocks fell / prices dropped: 0.909</title><rect class="s2" x="320.0" y="85.0" width="300.1" height="26.0" rx="3"/></g><text class="t-val" x="626.1" y="103.0" text-anchor="start">0.909</text><text class="t-tick" x="312.0" y="137.0" text-anchor="end">unrelated: guitar / stock market</text><g class="mark"><title>unrelated: guitar / stock market: 0.766</title><rect class="s2" x="320.0" y="119.0" width="252.7" height="26.0" rx="3"/></g><text class="t-val" x="578.7" y="137.0" text-anchor="start">0.766</text><text class="t-tick" x="312.0" y="171.0" text-anchor="end">unrelated: penguins / invoice</text><g class="mark"><title>unrelated: penguins / invoice: 0.807</title><rect class="s2" x="320.0" y="153.0" width="266.2" height="26.0" rx="3"/></g><text class="t-val" x="592.2" y="171.0" text-anchor="start">0.807</text><text class="t-tick" x="312.0" y="205.0" text-anchor="end">opposite: loved it / hated it</text><g class="mark"><title>opposite: loved it / hated it: 0.977</title><rect class="s2" x="320.0" y="187.0" width="322.4" height="26.0" rx="3"/></g><text class="t-val" x="648.4" y="205.0" text-anchor="start">0.977</text></svg><figcaption>Footnote 6, measured. Raw [CLS] vectors of any two sentences point in nearly the same direction. "I loved this movie" and "I hated this movie" get the highest similarity of all.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 285" role="img" aria-label="The learning-rate schedule of Appendix A.2, as the released code computes it: it climbs linearly from 0 over the first 10,000 steps, then falls linearly to 0 at step 1,000,000."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Learning rate over pre-training</text><text class="t-title" x="420.0" y="24.0" text-anchor="start">Zoom: the first 20,000 steps</text><line class="grid" x1="60" y1="214.0" x2="360" y2="214.0"/><text class="t-tick" x="52.0" y="218.0" text-anchor="end">0</text><line class="grid" x1="60" y1="171.5" x2="360" y2="171.5"/><text class="t-tick" x="52.0" y="175.5" text-anchor="end">2.5e-5</text><line class="grid" x1="60" y1="129.0" x2="360" y2="129.0"/><text class="t-tick" x="52.0" y="133.0" text-anchor="end">5.0e-5</text><line class="grid" x1="60" y1="86.5" x2="360" y2="86.5"/><text class="t-tick" x="52.0" y="90.5" text-anchor="end">7.5e-5</text><line class="grid" x1="60" y1="44.0" x2="360" y2="44.0"/><text class="t-tick" x="52.0" y="48.0" text-anchor="end">1.0e-4</text><text class="t-tick" x="60.0" y="232.0" text-anchor="middle">0k</text><text class="t-tick" x="135.0" y="232.0" text-anchor="middle">250k</text><text class="t-tick" x="210.0" y="232.0" text-anchor="middle">500k</text><text class="t-tick" x="285.0" y="232.0" text-anchor="middle">750k</text><text class="t-tick" x="360.0" y="232.0" text-anchor="middle">1,000k</text><line class="axis" x1="60" y1="214" x2="360" y2="214"/><polyline class="l1" points="60.0,214.0 60.1,205.5 60.3,197.0 60.5,188.5 60.6,180.0 60.8,171.5 60.9,163.0 61.0,154.5 61.2,146.0 61.4,137.5 61.5,129.0 61.6,120.5 61.8,112.0 62.0,103.5 62.1,95.0 62.2,86.5 62.4,78.0 62.5,69.5 62.7,61.0 62.9,52.5 63.0,45.7 63.1,45.8 63.3,45.9 63.5,46.0 63.6,46.0 63.8,46.1 63.9,46.2 64.0,46.3 64.2,46.4 64.3,46.5 64.5,46.5 64.7,46.6 64.8,46.7 65.0,46.8 65.1,46.9 65.2,47.0 65.4,47.1 65.5,47.1 65.7,47.2 65.8,47.3 66.0,47.4 67.5,48.2 69.0,49.1 70.5,50.0 72.0,50.8 73.5,51.7 75.0,52.5 76.5,53.4 78.0,54.2 79.5,55.0 81.0,55.9 82.5,56.8 84.0,57.6 85.5,58.5 87.0,59.3 88.5,60.2 90.0,61.0 91.5,61.8 93.0,62.7 94.5,63.6 96.0,64.4 97.5,65.2 99.0,66.1 100.5,66.9 102.0,67.8 103.5,68.7 105.0,69.5 106.5,70.3 108.0,71.2 109.5,72.1 111.0,72.9 112.5,73.8 114.0,74.6 115.5,75.5 117.0,76.3 118.5,77.2 120.0,78.0 121.5,78.8 123.0,79.7 124.5,80.5 126.0,81.4 127.5,82.3 129.0,83.1 130.5,83.9 132.0,84.8 133.5,85.7 135.0,86.5 136.5,87.3 138.0,88.2 139.5,89.0 141.0,89.9 142.5,90.8 144.0,91.6 145.5,92.4 147.0,93.3 148.5,94.1 150.0,95.0 151.5,95.8 153.0,96.7 154.5,97.5 156.0,98.4 157.5,99.2 159.0,100.1 160.5,100.9 162.0,101.8 163.5,102.6 165.0,103.5 166.5,104.3 168.0,105.2 169.5,106.0 171.0,106.9 172.5,107.8 174.0,108.6 175.5,109.5 177.0,110.3 178.5,111.2 180.0,112.0 181.5,112.9 183.0,113.7 184.5,114.6 186.0,115.4 187.5,116.3 189.0,117.1 190.5,118.0 192.0,118.8 193.5,119.7 195.0,120.5 196.5,121.4 198.0,122.2 199.5,123.1 201.0,123.9 202.5,124.8 204.0,125.6 205.5,126.5 207.0,127.3 208.5,128.2 210.0,129.0 211.5,129.8 213.0,130.7 214.5,131.6 216.0,132.4 217.5,133.2 219.0,134.1 220.5,134.9 222.0,135.8 223.5,136.6 225.0,137.5 226.5,138.4 228.0,139.2 229.5,140.1 231.0,140.9 232.5,141.8 234.0,142.6 235.5,143.4 237.0,144.3 238.5,145.1 240.0,146.0 241.5,146.8 243.0,147.7 244.5,148.6 246.0,149.4 247.5,150.2 249.0,151.1 250.5,151.9 252.0,152.8 253.5,153.7 255.0,154.5 256.5,155.3 258.0,156.2 259.5,157.1 261.0,157.9 262.5,158.8 264.0,159.6 265.5,160.5 267.0,161.3 268.5,162.1 270.0,163.0 271.5,163.8 273.0,164.7 274.5,165.6 276.0,166.4 277.5,167.2 279.0,168.1 280.5,168.9 282.0,169.8 283.5,170.7 285.0,171.5 286.5,172.4 288.0,173.2 289.5,174.1 291.0,174.9 292.5,175.8 294.0,176.6 295.5,177.4 297.0,178.3 298.5,179.2 300.0,180.0 301.5,180.9 303.0,181.7 304.5,182.5 306.0,183.4 307.5,184.2 309.0,185.1 310.5,185.9 312.0,186.8 313.5,187.7 315.0,188.5 316.5,189.3 318.0,190.2 319.5,191.1 321.0,191.9 322.5,192.8 324.0,193.6 325.5,194.4 327.0,195.3 328.5,196.2 330.0,197.0 331.5,197.8 333.0,198.7 334.5,199.6 336.0,200.4 337.5,201.2 339.0,202.1 340.5,203.0 342.0,203.8 343.5,204.6 345.0,205.5 346.5,206.3 348.0,207.2 349.5,208.0 351.0,208.9 352.5,209.8 354.0,210.6 355.5,211.4 357.0,212.3 358.5,213.2 360.0,214.0"/><line class="grid" x1="460" y1="214.0" x2="720" y2="214.0"/><text class="t-tick" x="452.0" y="218.0" text-anchor="end">0</text><line class="grid" x1="460" y1="171.5" x2="720" y2="171.5"/><text class="t-tick" x="452.0" y="175.5" text-anchor="end">2.5e-5</text><line class="grid" x1="460" y1="129.0" x2="720" y2="129.0"/><text class="t-tick" x="452.0" y="133.0" text-anchor="end">5.0e-5</text><line class="grid" x1="460" y1="86.5" x2="720" y2="86.5"/><text class="t-tick" x="452.0" y="90.5" text-anchor="end">7.5e-5</text><line class="grid" x1="460" y1="44.0" x2="720" y2="44.0"/><text class="t-tick" x="452.0" y="48.0" text-anchor="end">1.0e-4</text><text class="t-tick" x="460.0" y="232.0" text-anchor="middle">0k</text><text class="t-tick" x="525.0" y="232.0" text-anchor="middle">5k</text><text class="t-tick" x="590.0" y="232.0" text-anchor="middle">10k</text><text class="t-tick" x="655.0" y="232.0" text-anchor="middle">15k</text><text class="t-tick" x="720.0" y="232.0" text-anchor="middle">20k</text><line class="axis" x1="460" y1="214" x2="720" y2="214"/><polyline class="l1" points="460.0,214.0 466.5,205.5 473.0,197.0 479.5,188.5 486.0,180.0 492.5,171.5 499.0,163.0 505.5,154.5 512.0,146.0 518.5,137.5 525.0,129.0 531.5,120.5 538.0,112.0 544.5,103.5 551.0,95.0 557.5,86.5 564.0,78.0 570.5,69.5 577.0,61.0 583.5,52.5 590.0,45.7 596.5,45.8 603.0,45.9 609.5,46.0 616.0,46.0 622.5,46.1 629.0,46.2 635.5,46.3 642.0,46.4 648.5,46.5 655.0,46.5 661.5,46.6 668.0,46.7 674.5,46.8 681.0,46.9 687.5,47.0 694.0,47.1 700.5,47.1 707.0,47.2 713.5,47.3 720.0,47.4"/><text class="t-tick" x="610.0" y="150.0" text-anchor="start">warmup: 0 → 1e-4</text><text class="t-tick" x="610.0" y="166.0" text-anchor="start">over 10,000 steps</text><text class="t-tick" x="200.0" y="270.0" text-anchor="middle">step</text><text class="t-tick" x="590.0" y="270.0" text-anchor="middle">step</text></svg><figcaption>The learning-rate schedule as the released code computes it: a straight climb from 0 over the first 10,000 steps, then a straight fall to 0 at step 1,000,000.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="ReLU and GELU. For large positive inputs they agree. GELU bends smoothly through zero and lets small negative inputs pass as small negative outputs, instead of cutting them to exactly 0."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Two activation functions</text><line class="grid" x1="70" y1="260.0" x2="490" y2="260.0"/><text class="t-tick" x="62.0" y="264.0" text-anchor="end">-1</text><line class="grid" x1="70" y1="216.0" x2="490" y2="216.0"/><text class="t-tick" x="62.0" y="220.0" text-anchor="end">0</text><line class="grid" x1="70" y1="172.0" x2="490" y2="172.0"/><text class="t-tick" x="62.0" y="176.0" text-anchor="end">1</text><line class="grid" x1="70" y1="128.0" x2="490" y2="128.0"/><text class="t-tick" x="62.0" y="132.0" text-anchor="end">2</text><line class="grid" x1="70" y1="84.0" x2="490" y2="84.0"/><text class="t-tick" x="62.0" y="88.0" text-anchor="end">3</text><line class="grid" x1="70" y1="40.0" x2="490" y2="40.0"/><text class="t-tick" x="62.0" y="44.0" text-anchor="end">4</text><text class="t-tick" x="70.0" y="278.0" text-anchor="middle">-4</text><text class="t-tick" x="175.0" y="278.0" text-anchor="middle">-2</text><text class="t-tick" x="280.0" y="278.0" text-anchor="middle">0</text><text class="t-tick" x="385.0" y="278.0" text-anchor="middle">2</text><text class="t-tick" x="490.0" y="278.0" text-anchor="middle">4</text><line class="axis" x1="70" y1="260" x2="490" y2="260"/><polyline class="l2" points="70.0,216.0 72.6,216.0 75.2,216.0 77.9,216.0 80.5,216.0 83.1,216.0 85.7,216.0 88.4,216.0 91.0,216.0 93.6,216.0 96.2,216.0 98.9,216.0 101.5,216.0 104.1,216.0 106.8,216.0 109.4,216.0 112.0,216.0 114.6,216.0 117.3,216.0 119.9,216.0 122.5,216.0 125.1,216.0 127.7,216.0 130.4,216.0 133.0,216.0 135.6,216.0 138.2,216.0 140.9,216.0 143.5,216.0 146.1,216.0 148.8,216.0 151.4,216.0 154.0,216.0 156.6,216.0 159.3,216.0 161.9,216.0 164.5,216.0 167.1,216.0 169.8,216.0 172.4,216.0 175.0,216.0 177.6,216.0 180.3,216.0 182.9,216.0 185.5,216.0 188.1,216.0 190.8,216.0 193.4,216.0 196.0,216.0 198.6,216.0 201.2,216.0 203.9,216.0 206.5,216.0 209.1,216.0 211.8,216.0 214.4,216.0 217.0,216.0 219.6,216.0 222.3,216.0 224.9,216.0 227.5,216.0 230.1,216.0 232.8,216.0 235.4,216.0 238.0,216.0 240.6,216.0 243.3,216.0 245.9,216.0 248.5,216.0 251.1,216.0 253.8,216.0 256.4,216.0 259.0,216.0 261.6,216.0 264.3,216.0 266.9,216.0 269.5,216.0 272.1,216.0 274.8,216.0 277.4,216.0 280.0,216.0 282.6,213.8 285.2,211.6 287.9,209.4 290.5,207.2 293.1,205.0 295.7,202.8 298.4,200.6 301.0,198.4 303.6,196.2 306.2,194.0 308.9,191.8 311.5,189.6 314.1,187.4 316.7,185.2 319.4,183.0 322.0,180.8 324.6,178.6 327.2,176.4 329.9,174.2 332.5,172.0 335.1,169.8 337.7,167.6 340.4,165.4 343.0,163.2 345.6,161.0 348.2,158.8 350.9,156.6 353.5,154.4 356.1,152.2 358.8,150.0 361.4,147.8 364.0,145.6 366.6,143.4 369.2,141.2 371.9,139.0 374.5,136.8 377.1,134.6 379.7,132.4 382.4,130.2 385.0,128.0 387.6,125.8 390.2,123.6 392.9,121.4 395.5,119.2 398.1,117.0 400.7,114.8 403.4,112.6 406.0,110.4 408.6,108.2 411.2,106.0 413.9,103.8 416.5,101.6 419.1,99.4 421.8,97.2 424.4,95.0 427.0,92.8 429.6,90.6 432.3,88.4 434.9,86.2 437.5,84.0 440.1,81.8 442.7,79.6 445.4,77.4 448.0,75.2 450.6,73.0 453.2,70.8 455.9,68.6 458.5,66.4 461.1,64.2 463.8,62.0 466.4,59.8 469.0,57.6 471.6,55.4 474.3,53.2 476.9,51.0 479.5,48.8 482.1,46.6 484.8,44.4 487.4,42.2 490.0,40.0"/><polyline class="l1" points="70.0,216.0 72.6,216.0 75.2,216.0 77.9,216.0 80.5,216.0 83.1,216.0 85.7,216.0 88.4,216.0 91.0,216.0 93.6,216.0 96.2,216.0 98.9,216.0 101.5,216.1 104.1,216.1 106.8,216.1 109.4,216.1 112.0,216.1 114.6,216.1 117.3,216.1 119.9,216.2 122.5,216.2 125.1,216.2 127.7,216.2 130.4,216.3 133.0,216.3 135.6,216.4 138.2,216.4 140.9,216.5 143.5,216.5 146.1,216.6 148.8,216.7 151.4,216.8 154.0,216.9 156.6,217.0 159.3,217.1 161.9,217.2 164.5,217.3 167.1,217.5 169.8,217.7 172.4,217.8 175.0,218.0 177.6,218.2 180.3,218.4 182.9,218.6 185.5,218.8 188.1,219.1 190.8,219.3 193.4,219.6 196.0,219.9 198.6,220.1 201.2,220.4 203.9,220.7 206.5,221.0 209.1,221.3 211.8,221.5 214.4,221.8 217.0,222.1 219.6,222.3 222.3,222.6 224.9,222.8 227.5,223.0 230.1,223.2 232.8,223.3 235.4,223.4 238.0,223.5 240.6,223.5 243.3,223.5 245.9,223.4 248.5,223.2 251.1,223.0 253.8,222.8 256.4,222.5 259.0,222.1 261.6,221.6 264.3,221.0 266.9,220.4 269.5,219.7 272.1,218.9 274.8,218.0 277.4,217.1 280.0,216.0 282.6,214.9 285.2,213.6 287.9,212.3 290.5,210.9 293.1,209.4 295.7,207.8 298.4,206.2 301.0,204.5 303.6,202.7 306.2,200.8 308.9,198.8 311.5,196.8 314.1,194.8 316.7,192.7 319.4,190.5 322.0,188.3 324.6,186.0 327.2,183.7 329.9,181.4 332.5,179.0 335.1,176.6 337.7,174.2 340.4,171.7 343.0,169.3 345.6,166.8 348.2,164.3 350.9,161.9 353.5,159.4 356.1,156.9 358.8,154.4 361.4,151.9 364.0,149.5 366.6,147.0 369.2,144.5 371.9,142.1 374.5,139.6 377.1,137.2 379.7,134.8 382.4,132.4 385.0,130.0 387.6,127.6 390.2,125.3 392.9,122.9 395.5,120.5 398.1,118.2 400.7,115.9 403.4,113.6 406.0,111.3 408.6,109.0 411.2,106.7 413.9,104.4 416.5,102.1 419.1,99.9 421.8,97.6 424.4,95.4 427.0,93.1 429.6,90.9 432.3,88.6 434.9,86.4 437.5,84.2 440.1,82.0 442.7,79.7 445.4,77.5 448.0,75.3 450.6,73.1 453.2,70.9 455.9,68.7 458.5,66.5 461.1,64.2 463.8,62.0 466.4,59.8 469.0,57.6 471.6,55.4 474.3,53.2 476.9,51.0 479.5,48.8 482.1,46.6 484.8,44.4 487.4,42.2 490.0,40.0"/><rect class="s2" x="514" y="70" width="10" height="10" rx="2"/><text class="t-note" x="530.0" y="79.0" text-anchor="start">ReLU: max(0, x)</text><text class="t-tick" x="530.0" y="96.0" text-anchor="start">a sharp corner at 0</text><rect class="s1" x="514" y="126" width="10" height="10" rx="2"/><text class="t-note" x="530.0" y="135.0" text-anchor="start">GELU: x · Φ(x)</text><text class="t-tick" x="530.0" y="152.0" text-anchor="start">smooth; slightly negative</text><text class="t-tick" x="530.0" y="168.0" text-anchor="start">for x &lt; 0 (lowest -0.17)</text><text class="t-tick" x="280.0" y="300.0" text-anchor="middle">input x</text></svg><figcaption>ReLU and GELU. For large positive inputs they agree. GELU bends smoothly through zero and lets small negative inputs out as small negative numbers.</figcaption></figure>

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
