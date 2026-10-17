---
title: "Training: Data, Schedule, Adam and Regularization"
description: "How the Transformer was trained: the data, the batches, eight GPUs for twelve hours, Adam with a warm-up learning rate, dropout and label smoothing. We compute each rule by hand and test it on a released translation model."
part: 5
covers: "§5, Eq. 3"
date: 2026-10-18
tags: [transformer, attention, training, adam, learning-rate, label-smoothing, dropout]
---

A good network can still fail if you train it badly. Section 5 of the paper is one page long, and each line of it is a choice that made the Transformer train fast and well.

## The training regime {§5}

Parts 2 to 4 described the machine. This part describes how the machine learned. The section has four short pieces: the data, the hardware, the optimizer and the regularization. We read them in order.

> [!DEFINITION] Training step
> One update of the weights. The model reads one batch of examples, measures its error with a loss, computes the gradient of that loss, and changes every weight a little. A training run is many thousands of steps.

## The data {§5.1}

> [!PAPER] · §5.1
> [![Section 5.1 of the paper: training on WMT 2014 English-German with about 4.5 million sentence pairs, byte-pair encoding with a shared vocabulary of about 37000 tokens, and WMT 2014 English-French with 36M sentences and a 32000 word-piece vocabulary](/img/papers/transformer/p5-data.png)](/img/papers/transformer/p5-data.png)
>
> **What it says:** "We trained on the standard WMT 2014 English-German dataset consisting of about 4.5 million sentence pairs. Sentences were encoded using byte-pair encoding, which has a shared source-target vocabulary of about 37000 tokens." For English-French they used "36M sentences" and "a 32000 word-piece vocabulary".
>
> **What it means:**
>
> A sentence pair is one English sentence and its German translation, written by a human. WMT is a yearly translation competition, and its 2014 data is a fixed test bench. Every team trains and tests on the same sentences, so the scores in Part 6 compare like with like.
>
> The model does not read words. It reads tokens, and byte-pair encoding decides what a token is. Frequent words stay whole. Rare words break into frequent pieces, so the model never meets a word that it cannot write. The box below shows how.
>
> The key word is "shared". English and German use one vocabulary of about 37,000 tokens. Many pieces are the same in both languages: names, numbers, "Hotel", "Computer". With one shared list, the encoder input and the decoder output can use the same embedding table. Part 2 showed that the paper ties these weights; this sentence is what makes that possible.

> [!DEFINITION] Byte-pair encoding (BPE)
> A way to build a vocabulary from data. Start with single characters. Find the pair of symbols that occurs most often, for example "t" + "h", and join it into one new symbol "th". Repeat until the vocabulary has the size you want. A frequent word ends as one token; a rare word stays as a few pieces. Word-piece, used for French, is a close cousin of the same idea.

The released model that we use in this series, Helsinki-NLP/opus-mt-en-de, has the paper's base shapes but a different vocabulary. It splits text with SentencePiece into 58,101 tokens. Our test sentence becomes these German pieces: `▁Matt` + `e` for "Matte" (mat), and whole tokens for "Katze", "saß" and "müde". The `▁` sign marks the start of a word.

## Batches of similar length {§5.1}

> [!PAPER] · §5.1
> [![The end of Section 5.1: sentence pairs were batched together by approximate sequence length, and each batch had approximately 25000 source tokens and 25000 target tokens](/img/papers/transformer/p5-batching.png)](/img/papers/transformer/p5-batching.png)
>
> **What it says:** "Sentence pairs were batched together by approximate sequence length. Each training batch contained a set of sentence pairs containing approximately 25000 source tokens and 25000 target tokens."
>
> **What it means:**
>
> A GPU works on a rectangle of numbers. A batch is a stack of sentences, one per row, and every row must have the same length. Short rows are filled with a padding token that carries no meaning. The GPU still computes on the padding, so each padding token is wasted work.
>
> Think of packing books on shelves of equal height. If you mix tall atlases with small paperbacks, much of each shelf is air. If you put the atlases together and the paperbacks together, the shelves are almost full. Sorting sentences "by approximate sequence length" does the same thing.
>
> Note the unit of the batch size. It counts tokens, not sentences. A batch of short sentences holds many of them, and a batch of long sentences holds few. Each step then sees about the same amount of text, so each step costs about the same time.

We tested this with the Marian tokenizer on the text of the paper itself. Its first ten pages give 213 sentences, with 6,722 tokens. A sentence has 31.6 tokens on average, from 8 to 102. We packed them into batches of at most 600 slots (rows × longest row), once in random order and once sorted by length.

```text
random order     :  23 batches of <= 600 slots, real tokens 6722, slots 12456, padding 46.0%
sorted by length :  13 batches of <= 600 slots, real tokens 6722, slots 7280, padding 7.7%
```

In random order, almost half of the GPU's work is padding. Sorted by length, the waste falls to 7.7%, and the same text needs 13 batches instead of 23. At our mean length, a batch of 25,000 tokens would hold about 792 sentences. This is our own corpus and our own batch size. On WMT data the exact numbers will differ, but the gap will stay large.

## Eight GPUs, twelve hours {§5.2}

> [!PAPER] · §5.2
> [![Section 5.2 of the paper: one machine with 8 NVIDIA P100 GPUs, about 0.4 seconds per step for the base model, 100,000 steps or 12 hours, and 300,000 steps or 3.5 days for the big model at 1.0 seconds per step](/img/papers/transformer/p5-hardware.png)](/img/papers/transformer/p5-hardware.png)
>
> **What it says:** "We trained our models on one machine with 8 NVIDIA P100 GPUs." The base model took "about 0.4 seconds" per step, for "100,000 steps or 12 hours". The big model took 1.0 seconds per step, for "300,000 steps (3.5 days)".
>
> **What it means:**
>
> This is the headline of the paper in numbers. In 2017, the best translation systems trained for many days or weeks on many GPUs. The base Transformer beat most of them after half a day on one machine. Part 4 explained why. Attention has no chain of steps through time, so the GPU works on all the tokens of a batch at once.
>
> Check the arithmetic. 100,000 steps × 0.4 seconds = 40,000 seconds, which is 11.1 hours. The paper rounds this to 12 hours; the extra time is likely spent on saving checkpoints and other work. For the big model, 300,000 × 1.0 seconds = 300,000 seconds = 3.47 days, the paper's 3.5 days.
>
> Each step reads about 25,000 source tokens, split over 8 GPUs, so each GPU gets about 3,125. Over 100,000 steps, the base model reads 2.5 billion source tokens. The big model reads 7.5 billion. The data set has 4.5 million sentence pairs, so the model reads the same sentences many times; each full pass is called an epoch.

The "big" model is the bottom line of Table 3, which Part 6 covers. It has d_model = 1024 instead of 512, 16 heads instead of 8, and a feed-forward size of 4096. Each step is 2.5 times slower, and it trains 3 times longer.

## The optimizer: Adam {§5.3}

> [!PAPER] · §5.3
> [![The first sentence of Section 5.3: the Adam optimizer with beta1 = 0.9, beta2 = 0.98 and epsilon = 10 to the minus 9](/img/papers/transformer/p5-adam.png)](/img/papers/transformer/p5-adam.png)
>
> **What it says:** "We used the Adam optimizer with β1 = 0.9, β2 = 0.98 and ϵ = 10⁻⁹."
>
> **What it means:**
>
> The gradient tells each weight which way is downhill. Plain gradient descent takes a step of the same scale for every weight: learning rate × gradient. That is a problem when some weights get huge gradients and others get tiny ones. One learning rate is too large for the first group and too small for the second.
>
> Adam gives each weight its own step size. It keeps two running averages for each weight: the mean of its recent gradients, and the mean of their squares. It divides the first by the square root of the second. A weight with steady large gradients and a weight with steady small gradients then both move by about one learning rate per step.
>
> The paper changes one default. Most code uses β2 = 0.999; the paper uses 0.98. We show below why that matters.

> [!DEFINITION] Learning rate
> The number that scales each weight update. Too small, and training is slow. Too large, and the updates overshoot and the loss jumps up or becomes "NaN" (not a number). Most training runs change the learning rate over time; that plan is the learning-rate schedule.

Here are the Adam rules for one weight θ with gradient g at step t:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\,g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2)\,g_t^2
$$

$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t = \frac{v_t}{1-\beta_2^t}, \qquad \theta_t = \theta_{t-1} - \text{lr}\cdot\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

Read it left to right. The first line keeps two running means: $$m_t$$ for the gradient and $$v_t$$ for its square. The numbers $$\beta_1$$ and $$\beta_2$$ say how much of the old average to keep at each step. Both averages start at 0, so they are too small at first; dividing by $$1-\beta^t$$ corrects this ("bias correction"). The update divides the mean by the root of the mean square. $$\epsilon$$ only guards against a division by zero.

A worked step. Take lr = 0.001, start from $$m = v = 0$$, and let the first gradient be $$g = 0.5$$. Then $$m_1 = 0.1 \times 0.5 = 0.05$$ and $$v_1 = 0.02 \times 0.25 = 0.005$$. Bias correction gives $$\hat m = 0.05 / 0.1 = 0.5$$ and $$\hat v = 0.005 / 0.02 = 0.25$$. The step is $$0.001 \times 0.5 / \sqrt{0.25} = 0.001$$. Our script prints the same:

```text
gradient g = 0.5: m = 0.0500  v = 0.005000  m_hat = 0.5000  v_hat = 0.2500  step = 0.001000  new w = 0.999000
same first step with g = 50.0: step = 0.001000 (Adam steps do not grow with the gradient)
```

A gradient 100 times larger gives the same step. That is the point of Adam: the step size is set by the learning rate, not by the scale of the gradient.

Now β2. A running average with factor β remembers about $$1/(1-\beta)$$ steps. With 0.999, that is about 1,000 steps; with 0.98, about 50. We fed one weight a gradient of 1.0 for 200 steps, then a sudden jump to 10.0. We measured each step in units of the learning rate.

<figure class="fig"><svg viewBox="0 0 780 300" role="img" aria-label="The size of one Adam step, in units of the learning rate, when the gradient jumps from 1 to 10 at step 200. With beta2 = 0.98 the step grows to about 1.56 and is back near 1 by step 300. With beta2 = 0.999 it grows to about 2.76 and is still 1.34 at step 400."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="540" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0</text><line class="grid" x1="64" y1="176.7" x2="540" y2="176.7"/><text class="t-tick" x="56.0" y="180.7" text-anchor="end">1</text><line class="grid" x1="64" y1="103.3" x2="540" y2="103.3"/><text class="t-tick" x="56.0" y="107.3" text-anchor="end">2</text><line class="grid" x1="64" y1="30.0" x2="540" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">3</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">200</text><text class="t-tick" x="183.0" y="270.0" text-anchor="middle">250</text><text class="t-tick" x="302.0" y="270.0" text-anchor="middle">300</text><text class="t-tick" x="421.0" y="270.0" text-anchor="middle">350</text><text class="t-tick" x="540.0" y="270.0" text-anchor="middle">400</text><line class="axis" x1="64" y1="250" x2="540" y2="250"/><polyline class="l2" points="64.0,176.7 66.4,169.8 68.8,161.0 75.9,144.8 87.8,135.7 183.0,159.0 302.0,171.4 540.0,176.0"/><g class="mark"><title>β2 = 0.98 (paper), 200: 1.00</title><circle class="s2 ring" cx="64.0" cy="176.7" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 201: 1.09</title><circle class="s2 ring" cx="66.4" cy="169.8" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 202: 1.21</title><circle class="s2 ring" cx="68.8" cy="161.0" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 205: 1.44</title><circle class="s2 ring" cx="75.9" cy="144.8" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 210: 1.56</title><circle class="s2 ring" cx="87.8" cy="135.7" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 250: 1.24</title><circle class="s2 ring" cx="183.0" cy="159.0" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 300: 1.07</title><circle class="s2 ring" cx="302.0" cy="171.4" r="5"/></g><g class="mark"><title>β2 = 0.98 (paper), 400: 1.01</title><circle class="s2 ring" cx="540.0" cy="176.0" r="5"/></g><polyline class="l3" points="64.0,176.7 66.4,137.9 68.8,112.3 75.9,70.5 87.8,47.9 183.0,97.2 302.0,130.0 540.0,151.5"/><g class="mark"><title>β2 = 0.999 (common default), 200: 1.00</title><circle class="s3 ring" cx="64.0" cy="176.7" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 201: 1.53</title><circle class="s3 ring" cx="66.4" cy="137.9" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 202: 1.88</title><circle class="s3 ring" cx="68.8" cy="112.3" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 205: 2.45</title><circle class="s3 ring" cx="75.9" cy="70.5" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 210: 2.76</title><circle class="s3 ring" cx="87.8" cy="47.9" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 250: 2.08</title><circle class="s3 ring" cx="183.0" cy="97.2" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 300: 1.64</title><circle class="s3 ring" cx="302.0" cy="130.0" r="5"/></g><g class="mark"><title>β2 = 0.999 (common default), 400: 1.34</title><circle class="s3 ring" cx="540.0" cy="151.5" r="5"/></g><line class="grid" x1="546.0" y1="151.5" x2="556.0" y2="151.5"/><rect class="s3" x="560.0" y="146.5" width="10" height="10" rx="2"/><text class="t-note" x="576.0" y="155.5" text-anchor="start">β2 = 0.999 (common default): 1.34</text><line class="grid" x1="546.0" y1="176.0" x2="556.0" y2="176.0"/><rect class="s2" x="560.0" y="171.0" width="10" height="10" rx="2"/><text class="t-note" x="576.0" y="180.0" text-anchor="start">β2 = 0.98 (paper): 1.01</text><text class="t-tick" x="302.0" y="290.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">step size ÷ learning rate</text></svg><figcaption>After the gradient jumps from 1 to 10 at step 200, the Adam step grows for a while. With the paper's β2 = 0.98, it peaks at about 1.56 learning rates and is back to 1.07 by step 300. With β2 = 0.999, it peaks at about 2.76 and is still 1.34 at step 400.</figcaption></figure>

With 0.999, the mean of squares still remembers the small old gradients. So it is too small, and the step grows to almost three learning rates. With 0.98, the memory is short and the step stays near its normal size. In a Transformer, gradient sizes change fast early in training, so a short memory keeps the steps under control. The paper does not give this reason in words; this test shows the effect.

## The warm-up schedule {§5.3, Eq. 3}

> [!PAPER] · §5.3 · Eq. 3
> [![Equation 3 of the paper, lrate = d_model to the minus 0.5 times the minimum of step_num to the minus 0.5 and step_num times warmup_steps to the minus 1.5, followed by the sentence that it increases the learning rate linearly for the first warmup_steps steps and then decreases it with the inverse square root of the step number, with warmup_steps = 4000](/img/papers/transformer/p5-eq3.png)](/img/papers/transformer/p5-eq3.png)
>
> **What it says:** "This corresponds to increasing the learning rate linearly for the first warmup_steps training steps, and decreasing it thereafter proportionally to the inverse square root of the step number. We used warmup_steps = 4000."
>
> **What it means:**
>
> Think of a new driver on an icy road. At the start they go slowly, because they do not yet know how the car responds. Once they have a feel for it, they speed up. Near the end of the trip they slow down again to park exactly. The schedule does the same: a slow start, a peak, then a long, gentle slow-down.
>
> Why a slow start? At step 1, the weights are random, and so are Adam's running averages. Its estimate of the gradient size comes from only a few steps, so it can be badly wrong. A large learning rate then makes large, wrong updates, and training can break before it begins. 4,000 small steps give Adam time to learn the scale of the gradients.
>
> Why the slow-down? Late in training, the model is near a good solution. Large steps would make it jump around that point. Smaller steps let it settle in. The square root makes the fall gentle: at four times the step number, the rate is half as large.

Here is Eq. 3 in standard notation, with $$s$$ for the step number and $$w$$ for the number of warm-up steps:

$$
\text{lr}(s) = d_{\text{model}}^{-0.5}\cdot\min\!\left(s^{-0.5},\; s\cdot w^{-1.5}\right)
$$

Read it from the right. The second term in the brackets, $$s \cdot w^{-1.5}$$, grows in a straight line with $$s$$. The first term, $$s^{-0.5}$$, falls with the step number. The min picks the smaller one, so the rate first rises with the line and then follows the falling curve. The two terms are equal at the peak, $$s = w$$, where both are $$w^{-0.5}$$. The factor $$d_{\text{model}}^{-0.5}$$ in front sets the overall height: a wider model gets a lower rate.

A worked example for the base model, $$d_{\text{model}} = 512$$ and $$w = 4000$$. The front factor is $$512^{-0.5} = 0.04419$$. At the peak, $$4000^{-0.5} = 0.01581$$, so the rate is $$0.04419 \times 0.01581 = 0.000699$$. At step 1,000, the line gives $$1000 \times 4000^{-1.5} = 0.00395$$, so the rate is $$0.000175$$, a quarter of the peak. At step 16,000, four times the peak step, the rate is $$0.04419 \times 16000^{-0.5} = 0.000349$$, half the peak.

```text
step   1,000: d=512 -> 0.0001747   d=1024 -> 0.0001235
step   4,000: d=512 -> 0.0006988   d=1024 -> 0.0004941
step  16,000: d=512 -> 0.0003494   d=1024 -> 0.0002471
step 100,000: d=512 -> 0.0001398   d=1024 -> 0.0000988
step 300,000: d=512 -> 0.0000807   d=1024 -> 0.0000571
```

<figure class="fig"><svg viewBox="0 0 780 320" role="img" aria-label="The learning rate of Eq. 3 over the first 20,000 steps. It rises in a straight line to 7.0 in units of 0.0001 at step 4,000 for d_model 512, and to 4.9 for d_model 1024, then falls with one over the square root of the step."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="270.0" x2="580" y2="270.0"/><text class="t-tick" x="56.0" y="274.0" text-anchor="end">0</text><line class="grid" x1="64" y1="210.0" x2="580" y2="210.0"/><text class="t-tick" x="56.0" y="214.0" text-anchor="end">2</text><line class="grid" x1="64" y1="150.0" x2="580" y2="150.0"/><text class="t-tick" x="56.0" y="154.0" text-anchor="end">4</text><line class="grid" x1="64" y1="90.0" x2="580" y2="90.0"/><text class="t-tick" x="56.0" y="94.0" text-anchor="end">6</text><line class="grid" x1="64" y1="30.0" x2="580" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">8</text><text class="t-tick" x="64.0" y="290.0" text-anchor="middle">0</text><text class="t-tick" x="167.2" y="290.0" text-anchor="middle">4,000</text><text class="t-tick" x="270.4" y="290.0" text-anchor="middle">8,000</text><text class="t-tick" x="373.6" y="290.0" text-anchor="middle">12,000</text><text class="t-tick" x="476.8" y="290.0" text-anchor="middle">16,000</text><text class="t-tick" x="580.0" y="290.0" text-anchor="middle">20,000</text><line class="axis" x1="64" y1="270" x2="580" y2="270"/><polyline class="l2" points="64.0,270.0 76.9,243.8 89.8,217.6 102.7,191.4 115.6,165.2 128.5,139.0 141.4,112.8 154.3,86.6 167.2,60.4 180.1,72.4 193.0,82.5 205.9,91.2 218.8,98.8 231.7,105.6 244.6,111.5 257.5,116.9 270.4,121.8 283.3,126.2 296.2,130.2 309.1,134.0 322.0,137.4 334.9,140.6 347.8,143.6 360.7,146.4 373.6,149.0 386.5,151.4 399.4,153.7 412.3,155.9 425.2,157.9 438.1,159.9 451.0,161.7 463.9,163.5 476.8,165.2 489.7,166.8 502.6,168.3 515.5,169.8 528.4,171.2 541.3,172.5 554.2,173.8 567.1,175.1 580.0,176.2"/><g class="mark"><title>d_model 512 (base), 0: 0</title><circle class="s2 ring" cx="64.0" cy="270.0" r="5"/></g><g class="mark"><title>d_model 512 (base), 500: 0.9</title><circle class="s2 ring" cx="76.9" cy="243.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 1,000: 1.7</title><circle class="s2 ring" cx="89.8" cy="217.6" r="5"/></g><g class="mark"><title>d_model 512 (base), 1,500: 2.6</title><circle class="s2 ring" cx="102.7" cy="191.4" r="5"/></g><g class="mark"><title>d_model 512 (base), 2,000: 3.5</title><circle class="s2 ring" cx="115.6" cy="165.2" r="5"/></g><g class="mark"><title>d_model 512 (base), 2,500: 4.4</title><circle class="s2 ring" cx="128.5" cy="139.0" r="5"/></g><g class="mark"><title>d_model 512 (base), 3,000: 5.2</title><circle class="s2 ring" cx="141.4" cy="112.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 3,500: 6.1</title><circle class="s2 ring" cx="154.3" cy="86.6" r="5"/></g><g class="mark"><title>d_model 512 (base), 4,000: 7.0</title><circle class="s2 ring" cx="167.2" cy="60.4" r="5"/></g><g class="mark"><title>d_model 512 (base), 4,500: 6.6</title><circle class="s2 ring" cx="180.1" cy="72.4" r="5"/></g><g class="mark"><title>d_model 512 (base), 5,000: 6.2</title><circle class="s2 ring" cx="193.0" cy="82.5" r="5"/></g><g class="mark"><title>d_model 512 (base), 5,500: 6.0</title><circle class="s2 ring" cx="205.9" cy="91.2" r="5"/></g><g class="mark"><title>d_model 512 (base), 6,000: 5.7</title><circle class="s2 ring" cx="218.8" cy="98.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 6,500: 5.5</title><circle class="s2 ring" cx="231.7" cy="105.6" r="5"/></g><g class="mark"><title>d_model 512 (base), 7,000: 5.3</title><circle class="s2 ring" cx="244.6" cy="111.5" r="5"/></g><g class="mark"><title>d_model 512 (base), 7,500: 5.1</title><circle class="s2 ring" cx="257.5" cy="116.9" r="5"/></g><g class="mark"><title>d_model 512 (base), 8,000: 4.9</title><circle class="s2 ring" cx="270.4" cy="121.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 8,500: 4.8</title><circle class="s2 ring" cx="283.3" cy="126.2" r="5"/></g><g class="mark"><title>d_model 512 (base), 9,000: 4.7</title><circle class="s2 ring" cx="296.2" cy="130.2" r="5"/></g><g class="mark"><title>d_model 512 (base), 9,500: 4.5</title><circle class="s2 ring" cx="309.1" cy="134.0" r="5"/></g><g class="mark"><title>d_model 512 (base), 10,000: 4.4</title><circle class="s2 ring" cx="322.0" cy="137.4" r="5"/></g><g class="mark"><title>d_model 512 (base), 10,500: 4.3</title><circle class="s2 ring" cx="334.9" cy="140.6" r="5"/></g><g class="mark"><title>d_model 512 (base), 11,000: 4.2</title><circle class="s2 ring" cx="347.8" cy="143.6" r="5"/></g><g class="mark"><title>d_model 512 (base), 11,500: 4.1</title><circle class="s2 ring" cx="360.7" cy="146.4" r="5"/></g><g class="mark"><title>d_model 512 (base), 12,000: 4.0</title><circle class="s2 ring" cx="373.6" cy="149.0" r="5"/></g><g class="mark"><title>d_model 512 (base), 12,500: 4.0</title><circle class="s2 ring" cx="386.5" cy="151.4" r="5"/></g><g class="mark"><title>d_model 512 (base), 13,000: 3.9</title><circle class="s2 ring" cx="399.4" cy="153.7" r="5"/></g><g class="mark"><title>d_model 512 (base), 13,500: 3.8</title><circle class="s2 ring" cx="412.3" cy="155.9" r="5"/></g><g class="mark"><title>d_model 512 (base), 14,000: 3.7</title><circle class="s2 ring" cx="425.2" cy="157.9" r="5"/></g><g class="mark"><title>d_model 512 (base), 14,500: 3.7</title><circle class="s2 ring" cx="438.1" cy="159.9" r="5"/></g><g class="mark"><title>d_model 512 (base), 15,000: 3.6</title><circle class="s2 ring" cx="451.0" cy="161.7" r="5"/></g><g class="mark"><title>d_model 512 (base), 15,500: 3.5</title><circle class="s2 ring" cx="463.9" cy="163.5" r="5"/></g><g class="mark"><title>d_model 512 (base), 16,000: 3.5</title><circle class="s2 ring" cx="476.8" cy="165.2" r="5"/></g><g class="mark"><title>d_model 512 (base), 16,500: 3.4</title><circle class="s2 ring" cx="489.7" cy="166.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 17,000: 3.4</title><circle class="s2 ring" cx="502.6" cy="168.3" r="5"/></g><g class="mark"><title>d_model 512 (base), 17,500: 3.3</title><circle class="s2 ring" cx="515.5" cy="169.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 18,000: 3.3</title><circle class="s2 ring" cx="528.4" cy="171.2" r="5"/></g><g class="mark"><title>d_model 512 (base), 18,500: 3.2</title><circle class="s2 ring" cx="541.3" cy="172.5" r="5"/></g><g class="mark"><title>d_model 512 (base), 19,000: 3.2</title><circle class="s2 ring" cx="554.2" cy="173.8" r="5"/></g><g class="mark"><title>d_model 512 (base), 19,500: 3.2</title><circle class="s2 ring" cx="567.1" cy="175.1" r="5"/></g><g class="mark"><title>d_model 512 (base), 20,000: 3.1</title><circle class="s2 ring" cx="580.0" cy="176.2" r="5"/></g><polyline class="l3" points="64.0,270.0 76.9,251.5 89.8,232.9 102.7,214.4 115.6,195.9 128.5,177.4 141.4,158.8 154.3,140.3 167.2,121.8 180.1,130.2 193.0,137.4 205.9,143.6 218.8,149.0 231.7,153.7 244.6,157.9 257.5,161.7 270.4,165.2 283.3,168.3 296.2,171.2 309.1,173.8 322.0,176.2 334.9,178.5 347.8,180.6 360.7,182.6 373.6,184.4 386.5,186.1 399.4,187.8 412.3,189.3 425.2,190.8 438.1,192.1 451.0,193.5 463.9,194.7 476.8,195.9 489.7,197.0 502.6,198.1 515.5,199.1 528.4,200.1 541.3,201.1 554.2,202.0 567.1,202.9 580.0,203.7"/><g class="mark"><title>d_model 1024 (big), 0: 0</title><circle class="s3 ring" cx="64.0" cy="270.0" r="5"/></g><g class="mark"><title>d_model 1024 (big), 500: 0.6</title><circle class="s3 ring" cx="76.9" cy="251.5" r="5"/></g><g class="mark"><title>d_model 1024 (big), 1,000: 1.2</title><circle class="s3 ring" cx="89.8" cy="232.9" r="5"/></g><g class="mark"><title>d_model 1024 (big), 1,500: 1.9</title><circle class="s3 ring" cx="102.7" cy="214.4" r="5"/></g><g class="mark"><title>d_model 1024 (big), 2,000: 2.5</title><circle class="s3 ring" cx="115.6" cy="195.9" r="5"/></g><g class="mark"><title>d_model 1024 (big), 2,500: 3.1</title><circle class="s3 ring" cx="128.5" cy="177.4" r="5"/></g><g class="mark"><title>d_model 1024 (big), 3,000: 3.7</title><circle class="s3 ring" cx="141.4" cy="158.8" r="5"/></g><g class="mark"><title>d_model 1024 (big), 3,500: 4.3</title><circle class="s3 ring" cx="154.3" cy="140.3" r="5"/></g><g class="mark"><title>d_model 1024 (big), 4,000: 4.9</title><circle class="s3 ring" cx="167.2" cy="121.8" r="5"/></g><g class="mark"><title>d_model 1024 (big), 4,500: 4.7</title><circle class="s3 ring" cx="180.1" cy="130.2" r="5"/></g><g class="mark"><title>d_model 1024 (big), 5,000: 4.4</title><circle class="s3 ring" cx="193.0" cy="137.4" r="5"/></g><g class="mark"><title>d_model 1024 (big), 5,500: 4.2</title><circle class="s3 ring" cx="205.9" cy="143.6" r="5"/></g><g class="mark"><title>d_model 1024 (big), 6,000: 4.0</title><circle class="s3 ring" cx="218.8" cy="149.0" r="5"/></g><g class="mark"><title>d_model 1024 (big), 6,500: 3.9</title><circle class="s3 ring" cx="231.7" cy="153.7" r="5"/></g><g class="mark"><title>d_model 1024 (big), 7,000: 3.7</title><circle class="s3 ring" cx="244.6" cy="157.9" r="5"/></g><g class="mark"><title>d_model 1024 (big), 7,500: 3.6</title><circle class="s3 ring" cx="257.5" cy="161.7" r="5"/></g><g class="mark"><title>d_model 1024 (big), 8,000: 3.5</title><circle class="s3 ring" cx="270.4" cy="165.2" r="5"/></g><g class="mark"><title>d_model 1024 (big), 8,500: 3.4</title><circle class="s3 ring" cx="283.3" cy="168.3" r="5"/></g><g class="mark"><title>d_model 1024 (big), 9,000: 3.3</title><circle class="s3 ring" cx="296.2" cy="171.2" r="5"/></g><g class="mark"><title>d_model 1024 (big), 9,500: 3.2</title><circle class="s3 ring" cx="309.1" cy="173.8" r="5"/></g><g class="mark"><title>d_model 1024 (big), 10,000: 3.1</title><circle class="s3 ring" cx="322.0" cy="176.2" r="5"/></g><g class="mark"><title>d_model 1024 (big), 10,500: 3.0</title><circle class="s3 ring" cx="334.9" cy="178.5" r="5"/></g><g class="mark"><title>d_model 1024 (big), 11,000: 3.0</title><circle class="s3 ring" cx="347.8" cy="180.6" r="5"/></g><g class="mark"><title>d_model 1024 (big), 11,500: 2.9</title><circle class="s3 ring" cx="360.7" cy="182.6" r="5"/></g><g class="mark"><title>d_model 1024 (big), 12,000: 2.9</title><circle class="s3 ring" cx="373.6" cy="184.4" r="5"/></g><g class="mark"><title>d_model 1024 (big), 12,500: 2.8</title><circle class="s3 ring" cx="386.5" cy="186.1" r="5"/></g><g class="mark"><title>d_model 1024 (big), 13,000: 2.7</title><circle class="s3 ring" cx="399.4" cy="187.8" r="5"/></g><g class="mark"><title>d_model 1024 (big), 13,500: 2.7</title><circle class="s3 ring" cx="412.3" cy="189.3" r="5"/></g><g class="mark"><title>d_model 1024 (big), 14,000: 2.6</title><circle class="s3 ring" cx="425.2" cy="190.8" r="5"/></g><g class="mark"><title>d_model 1024 (big), 14,500: 2.6</title><circle class="s3 ring" cx="438.1" cy="192.1" r="5"/></g><g class="mark"><title>d_model 1024 (big), 15,000: 2.6</title><circle class="s3 ring" cx="451.0" cy="193.5" r="5"/></g><g class="mark"><title>d_model 1024 (big), 15,500: 2.5</title><circle class="s3 ring" cx="463.9" cy="194.7" r="5"/></g><g class="mark"><title>d_model 1024 (big), 16,000: 2.5</title><circle class="s3 ring" cx="476.8" cy="195.9" r="5"/></g><g class="mark"><title>d_model 1024 (big), 16,500: 2.4</title><circle class="s3 ring" cx="489.7" cy="197.0" r="5"/></g><g class="mark"><title>d_model 1024 (big), 17,000: 2.4</title><circle class="s3 ring" cx="502.6" cy="198.1" r="5"/></g><g class="mark"><title>d_model 1024 (big), 17,500: 2.4</title><circle class="s3 ring" cx="515.5" cy="199.1" r="5"/></g><g class="mark"><title>d_model 1024 (big), 18,000: 2.3</title><circle class="s3 ring" cx="528.4" cy="200.1" r="5"/></g><g class="mark"><title>d_model 1024 (big), 18,500: 2.3</title><circle class="s3 ring" cx="541.3" cy="201.1" r="5"/></g><g class="mark"><title>d_model 1024 (big), 19,000: 2.3</title><circle class="s3 ring" cx="554.2" cy="202.0" r="5"/></g><g class="mark"><title>d_model 1024 (big), 19,500: 2.2</title><circle class="s3 ring" cx="567.1" cy="202.9" r="5"/></g><g class="mark"><title>d_model 1024 (big), 20,000: 2.2</title><circle class="s3 ring" cx="580.0" cy="203.7" r="5"/></g><line class="grid" x1="586.0" y1="176.2" x2="596.0" y2="176.2"/><rect class="s2" x="600.0" y="171.2" width="10" height="10" rx="2"/><text class="t-note" x="616.0" y="180.2" text-anchor="start">d_model 512 (base): 3.1</text><line class="grid" x1="586.0" y1="203.7" x2="596.0" y2="203.7"/><rect class="s3" x="600.0" y="198.7" width="10" height="10" rx="2"/><text class="t-note" x="616.0" y="207.7" text-anchor="start">d_model 1024 (big): 2.2</text><text class="t-tick" x="322.0" y="310.0" text-anchor="middle">training step</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">learning rate (× 0.0001)</text></svg><figcaption>The learning rate of Eq. 3 over the first 20,000 steps. Both curves rise in a straight line until step 4,000, then fall with one over the square root of the step. The base model (d_model = 512) peaks at 0.000699; the big model (d_model = 1024) peaks at 0.000494.</figcaption></figure>

At the last step of the base run, step 100,000, the rate is 0.000140, one fifth of the peak. The big model never uses a higher rate than 0.000494, because its front factor is $$1024^{-0.5} = 0.03125$$. The paper does not say why it ties the rate to $$d_{\text{model}}$$. A fair reading: wider layers sum more numbers in each product. So the same change to each weight moves the output more, and a smaller rate balances that.

Warm-up became a standard part of Transformer training after this paper. Later work showed one main reason. In the paper's design, layer normalization comes after the residual sum ("post-norm", Part 2). Such networks can fail without warm-up. Moving the normalization before each sub-layer ("pre-norm") makes training less sensitive to it.

## Regularization {§5.4}

A large model can learn its training sentences by heart and then translate new sentences badly. This is overfitting. The paper uses "three types of regularization" against it. The page break splits the list: the two named types on page 8 are residual dropout and label smoothing.

> [!DEFINITION] Regularization
> Any method that stops a model from fitting its training data too exactly, so that it does better on new data. It often adds noise or a penalty during training. It is turned off when the model is used.

### Residual dropout

> [!PAPER] · §5.4
> [![The Residual Dropout paragraph of Section 5.4: dropout on the output of each sub-layer, before it is added to the sub-layer input and normalized, and on the sums of the embeddings and the positional encodings, with a rate of P_drop = 0.1 for the base model](/img/papers/transformer/p5-dropout.png)](/img/papers/transformer/p5-dropout.png)
>
> **What it says:** "We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks. For the base model, we use a rate of Pdrop = 0.1."
>
> **What it means:**
>
> Picture a team where any member may be absent on a given day. No one can rely on one colleague to do a job, so the team learns to spread each skill over several people. Dropout does this to the numbers of a network. At each step, it sets a random 10% of them to zero, and the network must still give the right answer.
>
> The place matters. Recall the shape of each sub-layer from Part 2: output = LayerNorm(x + Sublayer(x)). Dropout acts on Sublayer(x) only, before the sum. The residual path x always passes through whole. So the noise hits what each layer adds, but never cuts the main road through the stack.
>
> The second place is the input: the token embedding plus the position code. Here dropout hides parts of each input vector, so the model cannot depend on a few exact numbers of one word.

> [!DEFINITION] Dropout
> During training, each number of a vector is set to 0 with probability p, and the numbers that stay are divided by 1 − p. The division keeps the mean of the vector unchanged. At inference, dropout does nothing.

We ran a PyTorch dropout layer with p = 0.1 on ten ones, and then on one million:

```text
training : [1.111, 1.111, 1.111, 1.111, 0.0, 1.111, 1.111, 1.111, 1.111, 1.111]
inference: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
on 1,000,000 ones: zeros 100,232, mean 0.9997
```

One of the ten ones became 0, and the other nine became 1/0.9 = 1.111. On a million values, 100,232 were dropped, about 10%, and the mean stayed at 0.9997. So the next layer sees the same average signal in training and at inference.

Count the dropout places in the base model. Each of the 6 encoder layers has 2 sub-layers, so 12. Each of the 6 decoder layers has 3, so 18. Add the 2 embedding sums, and the total is 32. The released Marian model uses the same rate: its configuration has `dropout 0.1`, with no extra dropout inside attention or the feed-forward layer.

### Label smoothing

> [!PAPER] · §5.4
> [![The Label Smoothing paragraph of Section 5.4: label smoothing of value epsilon_ls = 0.1, which hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score](/img/papers/transformer/p5-smoothing.png)](/img/papers/transformer/p5-smoothing.png)
>
> **What it says:** "During training, we employed label smoothing of value ϵls = 0.1. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score."
>
> **What it means:**
>
> At each position, the decoder gives a probability to every token of the vocabulary. The normal target says: the correct token gets 100%, all others get 0%. To reach that target, the model pushes its scores toward infinity and becomes more and more certain. A model that is always certain is often wrong with full confidence, and small differences between good choices are lost.
>
> Label smoothing changes the target. The correct token gets 90%, and the other 10% is spread over the whole vocabulary. Now the best answer is no longer "100% sure". The model is rewarded for a high but not total confidence, and it keeps some probability for other tokens.
>
> The trade in the last sentence is honest. Perplexity measures how surprised the model is by the correct text, so a less certain model scores worse on it. But the model picks the correct word more often, and its translations score higher. The paper prefers the better translations.

> [!DEFINITION] Perplexity
> The exponential of the mean cross-entropy loss per token. A perplexity of 4 means the model is as unsure as if it chose fairly between 4 tokens at each position. Lower is better, and 1 means the model is always certain and right. BLEU, the translation score, is defined in Part 6.

The smoothed target for token $$k$$, when the correct token is $$y$$ and the vocabulary has $$V$$ tokens, is:

$$
q_k = (1-\epsilon)\cdot[k = y] + \frac{\epsilon}{V}, \qquad \mathcal{L} = -\sum_{k=1}^{V} q_k \log p_k
$$

Read it left to right. $$[k = y]$$ is 1 for the correct token and 0 for the others. So the correct token gets $$1 - \epsilon$$ plus its share of the spread, and every token gets $$\epsilon / V$$. The loss is the cross-entropy between this target $$q$$ and the model's prediction $$p$$. With $$\epsilon = 0$$ it is the normal loss, $$-\log p_y$$.

This is the form of the method's original paper. Some code spreads $$\epsilon$$ over the $$V - 1$$ wrong tokens only; for a large $$V$$ this changes almost nothing.

A tiny example with $$V = 5$$ words, where word 0 is correct. The target becomes $$0.9 + 0.1/5 = 0.92$$ for word 0, and $$0.1/5 = 0.02$$ for each other word.

<figure class="fig"><svg viewBox="0 0 780 180" role="img" aria-label="Two target distributions over a vocabulary of five words. Without smoothing, the correct word gets 1.00 and the others 0. With label smoothing of 0.1, the correct word gets 0.92 and each other word gets 0.02."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="34.0" text-anchor="start">Target without smoothing</text><text class="t-tick t-strong" x="92.0" y="60.0" text-anchor="end">word 0 ✓</text><g class="mark"><title>word 0 ✓: 1.00</title><rect class="s1" x="100.0" y="47.0" width="220.0" height="16.0" rx="3"/></g><text class="t-val" x="326.0" y="60.0" text-anchor="start">1.00</text><text class="t-tick" x="92.0" y="84.0" text-anchor="end">word 1</text><g class="mark"><title>word 1: 0.00</title><rect class="s1" x="100.0" y="71.0" width="2.0" height="16.0" rx="3"/></g><text class="t-val" x="108.0" y="84.0" text-anchor="start">0.00</text><text class="t-tick" x="92.0" y="108.0" text-anchor="end">word 2</text><g class="mark"><title>word 2: 0.00</title><rect class="s1" x="100.0" y="95.0" width="2.0" height="16.0" rx="3"/></g><text class="t-val" x="108.0" y="108.0" text-anchor="start">0.00</text><text class="t-tick" x="92.0" y="132.0" text-anchor="end">word 3</text><g class="mark"><title>word 3: 0.00</title><rect class="s1" x="100.0" y="119.0" width="2.0" height="16.0" rx="3"/></g><text class="t-val" x="108.0" y="132.0" text-anchor="start">0.00</text><text class="t-tick" x="92.0" y="156.0" text-anchor="end">word 4</text><g class="mark"><title>word 4: 0.00</title><rect class="s1" x="100.0" y="143.0" width="2.0" height="16.0" rx="3"/></g><text class="t-val" x="108.0" y="156.0" text-anchor="start">0.00</text><text class="t-title" x="410.0" y="34.0" text-anchor="start">Target with ε = 0.1</text><text class="t-tick t-strong" x="482.0" y="60.0" text-anchor="end">word 0 ✓</text><g class="mark"><title>word 0 ✓: 0.92</title><rect class="s2" x="490.0" y="47.0" width="202.4" height="16.0" rx="3"/></g><text class="t-val" x="698.4" y="60.0" text-anchor="start">0.92</text><text class="t-tick" x="482.0" y="84.0" text-anchor="end">word 1</text><g class="mark"><title>word 1: 0.02</title><rect class="s2" x="490.0" y="71.0" width="4.4" height="16.0" rx="3"/></g><text class="t-val" x="500.4" y="84.0" text-anchor="start">0.02</text><text class="t-tick" x="482.0" y="108.0" text-anchor="end">word 2</text><g class="mark"><title>word 2: 0.02</title><rect class="s2" x="490.0" y="95.0" width="4.4" height="16.0" rx="3"/></g><text class="t-val" x="500.4" y="108.0" text-anchor="start">0.02</text><text class="t-tick" x="482.0" y="132.0" text-anchor="end">word 3</text><g class="mark"><title>word 3: 0.02</title><rect class="s2" x="490.0" y="119.0" width="4.4" height="16.0" rx="3"/></g><text class="t-val" x="500.4" y="132.0" text-anchor="start">0.02</text><text class="t-tick" x="482.0" y="156.0" text-anchor="end">word 4</text><g class="mark"><title>word 4: 0.02</title><rect class="s2" x="490.0" y="143.0" width="4.4" height="16.0" rx="3"/></g><text class="t-val" x="500.4" y="156.0" text-anchor="start">0.02</text></svg><figcaption>Two targets for the same position in a vocabulary of five words. Without smoothing, the correct word gets 1.00 and every other word gets 0. With label smoothing of 0.1, the correct word gets 0.92 and each other word gets 0.02.</figcaption></figure>

Now compare three predictions. One is almost certain (0.999 on word 0), one equals the target, and one is unsure (0.6).

```text
confident 0.999 : plain loss 0.0010  smoothed loss 0.6644
model = target  : plain loss 0.0834  smoothed loss 0.3897
unsure 0.60     : plain loss 0.5108  smoothed loss 0.6542
```

The normal loss loves the confident prediction: 0.0010. The smoothed loss punishes it: 0.6644, worse than the unsure one. The lowest smoothed loss, 0.3897, is at the target itself. That is the whole effect in three lines: the model is pulled toward 92%, not toward 100%.

We ran the same test on the released Marian model. It translated "The cat sat on the mat because it was tired." as "Die Katze saß auf der Matte, weil sie müde war." We then measured its probability for each token of that translation.

```text
       token  p(token)   plain  smoothed
        ▁Die     0.768   0.263     1.578
      ▁Katze     0.944   0.058     1.524
       ▁müde     0.834   0.182     1.568
mean plain loss 0.184 (perplexity 1.20),  mean smoothed loss 1.545
lowest possible smoothed loss with 58,101 tokens (prediction = target): 1.422; each other token gets 1.72e-06
```

The model gives its chosen tokens 0.834 on average, never close to 1. The smoothed loss looks large, 1.545, but even a perfect prediction cannot go below 1.422 here. The reason is the spread: each of the 58,100 other tokens should get 0.1/58,101 = 0.0000017, and that costs about 1.33 alone. We did not check how Marian was trained. So this test shows the size of the two losses, not the effect of smoothing on this model. The perplexity of 1.20 is on the model's own output, which is an easy test; the paper's perplexities in Part 6 are on human translations.

## What these choices bought {§5}

None of these rules is new on its own. Adam, dropout, label smoothing and length-sorted batches all existed before 2017. The new parts are the warm-up schedule tied to $$d_{\text{model}}$$, and a recipe that worked for a network with no recurrence. Part 6 shows the result. The base model beat most earlier systems after 12 hours, and Table 3 tests other values of dropout and label smoothing.

What our tests cannot show: we did not train a Transformer here. The batching test uses the paper's own text, not WMT data. The Adam test uses one weight with a made-up gradient. The Marian model has the base shapes, but its data, vocabulary and training settings are not the paper's.

<figure><img src="/img/papers/transformer/part5-run.png" alt="Terminal output of tx_part5.py: the batching test, the schedule arithmetic, Adam worked by hand, the learning rate of Eq. 3 at chosen steps, the dropout test, and label smoothing on five words and on the Marian model" loading="lazy" /><figcaption>The output of tx_part5.py, as it ran for this article.</figcaption></figure>

> [!TAKEAWAYS] Key takeaways
> - The base model trained on 4.5 million English-German sentence pairs, with one shared BPE vocabulary of about 37,000 tokens.
> - Batches hold about 25,000 tokens, not a fixed number of sentences. Sorting by length cut padding from 46.0% to 7.7% in our test.
> - 8 P100 GPUs, 100,000 steps of 0.4 seconds: 11.1 hours of steps, about 12 hours in total. The big model: 300,000 steps, 3.5 days.
> - Adam gives each weight its own step size. With β2 = 0.98, its memory is about 50 steps, so steps stay near one learning rate when gradients jump.
> - The rate of Eq. 3 rises for 4,000 steps and peaks at 0.000699 for d_model = 512. Then it falls with one over the square root of the step.
> - Warm-up protects the first steps, when Adam's averages are still unreliable.
> - Dropout of 0.1 acts at 32 places in the base model: on each sub-layer output before the residual sum, and on the input embeddings.
> - Label smoothing with ε = 0.1 sets the target to 0.92 instead of 1.00 in a five-word example. It hurts perplexity but improves accuracy and BLEU.

## The paper and the code

[Paper (arXiv 1706.03762)](https://arxiv.org/abs/1706.03762) · [Official code (tensor2tensor)](https://github.com/tensorflow/tensor2tensor) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/transformer)
