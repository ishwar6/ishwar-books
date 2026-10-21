---
title: "Inside a Modern Language Model: Data, Scale and Architecture"
description: "How a frontier language model is built before any chat training: the web data, the scaling rules, the layer design, experts, and the machines that train it."
date: 2026-10-22
tags: [history, llm, deep-learning]
series: "How AI Got Here"
series_part: 5
motif: blocks
accent: "#f27aa0"
---

In 2024, Meta trained a language model with 405 billion parameters on 15.6 trillion tokens of text. The run used up to 16,384 GPUs. In one 54-day stretch of it, the job stopped 466 times, most often because a piece of hardware failed. Each time, the system loaded the last saved copy of the model and carried on.

At the end there was a file of numbers that could continue almost any text. No chat training had happened yet. That file is the **base model**, and it is the subject of this part. What goes into it? How big should it be? What does each layer look like inside, and what does it cost to make?

This is Part 5 of How AI Got Here. [Part 1](from-neuron-to-opus.md) told the story of neural networks up to the Transformer and the first scaling laws. [Part 2](classical-machine-learning.md) covered classical machine learning, and [Part 3](reinforcement-learning-history.md) covered reinforcement learning. [Part 4](how-models-are-trained-today.md) showed how a base model becomes an assistant. This part goes back one step and opens the base model itself, as it is built in October 2026.

> [!TIP] The whole idea in one minute
> A base model learns one thing: to predict the next token. How well it learns depends on three numbers: its size, its data and its compute. **Scaling laws** say how to split a budget between them. **Chinchilla** showed in 2022 that models were too big for their data. Today small models are trained on far more data than that rule says, because they are cheaper to run.
>
> The data is mostly the public web, filtered hard and deduplicated, plus code, maths and some text written by other models. The layer is still the 2017 Transformer, with new parts: **RMSNorm**, **SwiGLU**, **RoPE** and shared keys and values. Many frontier models are **mixtures of experts**, where each token uses only a small part of the network. Training spreads one model over thousands of chips, in 16-bit or even 8-bit numbers.

<figure class="fig"><svg viewBox="0 0 720 364" role="img" aria-label="A timeline from 2017 to 2026 with seven threads: the Transformer, scaling laws, changes to the block, mixture of experts, open web data sets, ways to shrink the KV cache, and long context with FP8 training."><text class="t-title" x="10" y="22">How the pre-trained model took its 2026 shape, 2017 to 2026</text><line class="grid" x1="271.6" y1="34" x2="271.6" y2="326"/><line class="grid" x1="358.2" y1="34" x2="358.2" y2="326"/><line class="grid" x1="444.7" y1="34" x2="444.7" y2="326"/><line class="grid" x1="531.2" y1="34" x2="531.2" y2="326"/><line class="grid" x1="617.8" y1="34" x2="617.8" y2="326"/><text class="t-note" x="10" y="52">Transformer, GPT, BERT</text><text class="t-tick" x="10" y="68">2017 to 2020</text><rect class="box-1" x="291.1" y="42" width="127.6" height="22" rx="4"/><text class="t-note" x="10" y="92">Scaling laws: Kaplan, Chinchilla</text><text class="t-tick" x="10" y="108">2020 to 2022</text><rect class="box-2" x="403.6" y="82" width="95.2" height="22" rx="4"/><text class="t-note" x="10" y="132">Block: RMSNorm, SwiGLU, RoPE</text><text class="t-tick" x="10" y="148">2019 to 2021</text><rect class="box-3" x="392.8" y="122" width="64.9" height="22" rx="4"/><text class="t-note" x="10" y="172">Mixture of experts at scale</text><text class="t-tick" x="10" y="188">2017 to 2026</text><rect class="box-4" x="273.8" y="162" width="421.9" height="22" rx="4"/><text class="t-note" x="10" y="212">Open web data: C4 to FineWeb</text><text class="t-tick" x="10" y="228">2019 to 2024</text><rect class="box-1" x="392.8" y="202" width="203.4" height="22" rx="4"/><text class="t-note" x="10" y="252">KV savings: MQA, GQA, MLA</text><text class="t-tick" x="10" y="268">2019 to 2024</text><rect class="box-2" x="395.0" y="242" width="196.9" height="22" rx="4"/><text class="t-note" x="10" y="292">Long context, FP8, multi-token</text><text class="t-tick" x="10" y="308">2023 to 2026</text><rect class="box-3" x="552.9" y="282" width="142.8" height="22" rx="4"/><line class="axis" x1="250" y1="328" x2="700" y2="328"/><text class="t-tick" x="271.635" y="348" text-anchor="middle">2017</text><text class="t-tick" x="358.173" y="348" text-anchor="middle">2019</text><text class="t-tick" x="444.712" y="348" text-anchor="middle">2021</text><text class="t-tick" x="531.25" y="348" text-anchor="middle">2023</text><text class="t-tick" x="617.788" y="348" text-anchor="middle">2025</text></svg><figcaption>The core of the 2017 design is still there. Most changes after 2020 make the same model cheaper to train, cheaper to serve, or able to read longer inputs.</figcaption></figure>

## What pre-training optimises

### One loss, read off every token

Part 1 introduced the **next-token loss**. The model reads a text, and at every position it gives a probability to each possible next token. The loss is the negative log of the probability it gave to the token that really came next. Averaged over trillions of positions, that one number is the whole goal of pre-training.

> [!DEFINITION] Cross-entropy loss
> The average, over all positions, of $$-\log p$$, where $$p$$ is the probability the model gave to the true next token. It is measured in nats when the natural logarithm is used. Lower is better.

The loss is hard to feel, so people often turn it into **perplexity**. Think of a quiz where the model must guess the next word. A perplexity of 7 means the model is as unsure as if it had to pick fairly between 7 equal choices at every step.

$$
\text{PPL} = \exp\!\Big( -\frac{1}{T} \sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}) \Big) = e^{L}
$$

$$T$$ is the number of tokens, $$x_t$$ is the token at position $$t$$ and $$x_{<t}$$ are the tokens before it. $$p_\theta$$ is the model with parameters $$\theta$$. The part inside the brackets is the average loss $$L$$, so perplexity is just $$e$$ to the power of the loss.

> [!DEFINITION] Perplexity
> The exponential of the average cross-entropy loss. It reads as "the number of equally likely choices the model is torn between". A model that guesses at random over $$V$$ tokens has perplexity $$V$$.

Our script gives the scale. A new model that guesses at random over a vocabulary of 128,000 tokens has a loss of $$\ln 128{,}000 = 11.76$$ and a perplexity of 128,000. A loss of 4.0 is a perplexity of 54.60, and a loss of 2.0 is 7.39. A single token with probability 0.9 costs 0.105; one with probability 0.01 costs 4.605.

> [!WARNING] Loss is not ability
> Two losses can be compared only on the same test text with the same tokenizer. A tokenizer with longer tokens makes each prediction harder, so the loss per token rises even if the model is better. Labs report losses for planning, and benchmarks for ability.

## Scaling laws: how big, how long

### Counting the compute

A **FLOP** is one floating-point operation, such as one multiply or one add. Training cost is counted in FLOPs, and there is a famous shortcut for it.

> [!DEFINITION] FLOP
> One floating-point operation, such as $$a \times b$$ or $$a + b$$ on real numbers. Training budgets are given as a total count of FLOPs; chip speed is given in FLOPs per second.

$$
C \approx 6 N D
$$

$$C$$ is the total training compute in FLOPs, $$N$$ is the number of parameters and $$D$$ is the number of training tokens. Each parameter takes part in one multiply and one add for each token on the forward pass, which is 2 FLOPs. The backward pass costs about twice the forward pass, which adds 4. So every parameter costs about 6 FLOPs per token.

A model with 1 billion parameters trained on 20 billion tokens costs $$6 \times 10^9 \times 2 \times 10^{10} = 1.2 \times 10^{20}$$ FLOPs. The rule leaves out the attention scores, which matter only for long texts. Jared Kaplan and colleagues used it in their 2020 paper, and almost every lab uses it today.

Here are some real runs, from our script:

| Model | Parameters $$N$$ | Tokens $$D$$ | $$6ND$$ (FLOPs) | Tokens per parameter |
|---|---|---|---|---|
| GPT-3 (2020) | 175B | 300B | $$3.15 \times 10^{23}$$ | 1.7 |
| Gopher (2021) | 280B | 300B | $$5.04 \times 10^{23}$$ | 1.1 |
| Chinchilla (2022) | 70B | 1.4T | $$5.88 \times 10^{23}$$ | 20 |
| Llama 3 8B (2024) | 8B | 15T | $$7.2 \times 10^{23}$$ | 1,875 |
| Llama 3 405B (2024) | 405B | 15.6T | $$3.79 \times 10^{25}$$ | 38.5 |
| DeepSeek-V3 (2024) | 37B active | 14.8T | $$3.29 \times 10^{24}$$ | 400 |

The Llama 3 paper itself reports $$3.8 \times 10^{25}$$ FLOPs for the 405B model, so the shortcut agrees. For DeepSeek-V3, $$N$$ is the number of parameters each token uses, not the total; the section on experts explains why.

### Kaplan 2020: smooth power laws

Part 1 introduced the first scaling-law paper. Its main picture shows why it changed how labs plan.

> [!PAPER] Kaplan et al. (2020), Scaling Laws for Neural Language Models · Figure 1 · page 3
> [![Figure 1 of the Kaplan scaling-law paper: three log-log plots of test loss against compute, dataset size and parameters, each a straight line, with fitted power laws of exponent 0.050, 0.095 and 0.076](/img/writings/history/h-kaplan-fig.png)](/img/writings/history/h-kaplan-fig.png)
>
> **What it says:** "Language modeling performance improves smoothly as we increase the model size, datasetset size, and amount of compute used for training. For optimal performance all three factors must be scaled up in tandem."
>
> **What it means:**
>
> On log-log axes, a power law is a straight line, and all three panels are straight lines. That is what made the result useful. A lab could train small models for a few days and draw the line. Then it could read off the loss of a model a thousand times larger, before it spent the money. The exponents are small, 0.076 for parameters, so each tenfold increase buys a modest but reliable drop.

Kaplan's team also asked how to split a growing budget. Their answer was to put most of it into parameters: the best model size grew as $$N \propto C^{0.73}$$. GPT-3, Gopher and Megatron-Turing NLG followed that advice. They were very large and trained on 300 billion tokens or fewer.

### Chinchilla 2022: the balance was wrong

> [!DEFINITION] Compute-optimal
> For a fixed compute budget $$C$$, the model size $$N$$ and number of tokens $$D$$ that give the lowest loss. Spending the same budget on a bigger model with fewer tokens, or the reverse, gives a higher loss.

In March 2022, Jordan Hoffmann and colleagues at DeepMind repeated the study with a different method. They trained over 400 models, and changed the learning-rate schedule to match each run's length, which Kaplan's runs had not done.

> [!PAPER] Hoffmann et al. (2022), Training Compute-Optimal Large Language Models · Abstract · page 1
> [![The Chinchilla abstract, with current large language models are significantly undertrained, the model size and the number of training tokens should be scaled equally, with 70B parameters and 4x more more data highlighted](/img/writings/history/h-chinchilla-abs.png)](/img/writings/history/h-chinchilla-abs.png)
>
> **What it says:** "current large language models are significantly under-trained". "For compute-optimal training, the model size and the number of training tokens should be scaled equally: for every doubling of model size the number of training tokens should also be doubled." Chinchilla used "the same compute budget as Gopher but with 70B parameters and 4× more more data".
>
> **What it means:**
>
> "Under-trained" is the key word. Gopher had 280 billion parameters but saw only 300 billion tokens, about one token per parameter. Chinchilla was four times smaller, read 1.4 trillion tokens, and beat it on almost every test. A smaller model is also cheaper to run for every user, so this was a win twice over. (The doubled "more more" is in the original.)

The paper fits one formula to all its runs. It splits the loss into three parts. One is a floor that no model can beat, one falls as the model grows, and one falls as the data grows.

$$
L(N, D) = E + \frac{A}{N^{\alpha}} + \frac{B}{D^{\beta}}
$$

$$E$$ is the **irreducible loss**, the randomness of language itself; no model can predict a coin toss in a story. $$A / N^{\alpha}$$ is the penalty for a finite model and $$B / D^{\beta}$$ the penalty for finite data. $$\alpha$$ and $$\beta$$ say how fast each penalty shrinks. The fitted values in the paper are $$E = 1.69$$, $$A = 406.4$$, $$B = 410.7$$, $$\alpha = 0.34$$ and $$\beta = 0.28$$.

Put in Gopher, with $$N = 280 \times 10^9$$ and $$D = 300 \times 10^9$$. The model term is 0.0521 and the data term is 0.2511, for a loss of 1.9933. Gopher's loss was mostly a data problem. For Chinchilla, with $$N = 70 \times 10^9$$ and $$D = 1.4 \times 10^{12}$$, the terms are 0.0835 and 0.1632, for a loss of 1.9366. The smaller model wins because its two penalties are better balanced.

To find the best split for a budget, put $$D = C / (6N)$$ into the formula and find the $$N$$ with the lowest loss. Each budget gives a U-shaped curve. Too small a model wastes the budget on tokens it cannot use; too large a model sees too few tokens.

<figure class="fig"><svg viewBox="0 0 720 352" role="img" aria-label="Four U-shaped curves of loss against model size, one per compute budget from 10 to the 21 to 5.76 times 10 to the 23 FLOPs. The minimum moves to larger models as the budget grows: 1.84 billion, 5.2 billion, 14.7 billion and 32.2 billion parameters."><text class="t-title" x="10" y="22">Iso-FLOP curves from the Chinchilla fit: each budget has a best model size</text><line class="grid" x1="70" y1="285.8" x2="690" y2="285.8"/><text class="t-tick" x="62" y="289.778" text-anchor="end">1.9</text><line class="grid" x1="70" y1="228.9" x2="690" y2="228.9"/><text class="t-tick" x="62" y="232.889" text-anchor="end">2.1</text><line class="grid" x1="70" y1="172.0" x2="690" y2="172.0"/><text class="t-tick" x="62" y="176" text-anchor="end">2.3</text><line class="grid" x1="70" y1="115.1" x2="690" y2="115.1"/><text class="t-tick" x="62" y="119.111" text-anchor="end">2.5</text><line class="grid" x1="70" y1="58.2" x2="690" y2="58.2"/><text class="t-tick" x="62" y="62.2222" text-anchor="end">2.7</text><line class="grid" x1="138.9" y1="44" x2="138.9" y2="300"/><text class="t-tick" x="138.889" y="318" text-anchor="middle">100M</text><line class="grid" x1="276.7" y1="44" x2="276.7" y2="300"/><text class="t-tick" x="276.667" y="318" text-anchor="middle">1B</text><line class="grid" x1="414.4" y1="44" x2="414.4" y2="300"/><text class="t-tick" x="414.444" y="318" text-anchor="middle">10B</text><line class="grid" x1="552.2" y1="44" x2="552.2" y2="300"/><text class="t-tick" x="552.222" y="318" text-anchor="middle">100B</text><line class="grid" x1="690.0" y1="44" x2="690.0" y2="300"/><text class="t-tick" x="690" y="318" text-anchor="middle">1T</text><line class="axis" x1="70" y1="300" x2="690" y2="300"/><line class="axis" x1="70" y1="44" x2="70" y2="300"/><text class="t-muted" x="690" y="338" text-anchor="end">parameters N (log scale); tokens D = C / (6N)</text><text class="t-muted" x="14" y="38">loss</text><polyline class="l3" points="110.5,48.0 113.6,52.0 116.7,55.8 119.8,59.6 123.0,63.4 126.1,67.0 129.2,70.5 132.3,74.0 135.4,77.4 138.5,80.7 141.7,83.9 144.8,87.1 147.9,90.1 151.0,93.1 154.1,96.0 157.2,98.9 160.4,101.7 163.5,104.4 166.6,107.0 169.7,109.6 172.8,112.1 175.9,114.5 179.0,116.8 182.2,119.1 185.3,121.4 188.4,123.5 191.5,125.6 194.6,127.7 197.7,129.6 200.9,131.6 204.0,133.4 207.1,135.2 210.2,136.9 213.3,138.6 216.4,140.2 219.5,141.8 222.7,143.3 225.8,144.7 228.9,146.1 232.0,147.4 235.1,148.7 238.2,149.9 241.4,151.1 244.5,152.2 247.6,153.2 250.7,154.2 253.8,155.2 256.9,156.1 260.1,156.9 263.2,157.7 266.3,158.5 269.4,159.2 272.5,159.8 275.6,160.4 278.7,161.0 281.9,161.5 285.0,161.9 288.1,162.3 291.2,162.7 294.3,163.0 297.4,163.2 300.6,163.4 303.7,163.6 306.8,163.7 309.9,163.8 313.0,163.8 316.1,163.8 319.2,163.7 322.4,163.6 325.5,163.4 328.6,163.2 331.7,162.9 334.8,162.6 337.9,162.2 341.1,161.8 344.2,161.4 347.3,160.9 350.4,160.4 353.5,159.8 356.6,159.2 359.7,158.5 362.9,157.8 366.0,157.0 369.1,156.2 372.2,155.3 375.3,154.4 378.4,153.5 381.6,152.5 384.7,151.4 387.8,150.3 390.9,149.2 394.0,148.0 397.1,146.7 400.3,145.5 403.4,144.1 406.5,142.8 409.6,141.3 412.7,139.9 415.8,138.3 418.9,136.8 422.1,135.1 425.2,133.5 428.3,131.8 431.4,130.0 434.5,128.2 437.6,126.3 440.8,124.4 443.9,122.4 447.0,120.4 450.1,118.3 453.2,116.2 456.3,114.0 459.4,111.8 462.6,109.5 465.7,107.2 468.8,104.8 471.9,102.3 475.0,99.8 478.1,97.3 481.3,94.7 484.4,92.0 487.5,89.3 490.6,86.5 493.7,83.6 496.8,80.7 499.9,77.8 503.1,74.8 506.2,71.7 509.3,68.5 512.4,65.3 515.5,62.1 518.6,58.7 521.8,55.3 524.9,51.9 528.0,48.4 531.1,44.8"/><circle class="s3" cx="313.0" cy="163.8" r="5"/><text class="t-s3" x="321.015" y="181.784">10²¹ FLOPs</text><polyline class="l1" points="98.0,48.5 101.2,53.1 104.3,57.6 107.4,62.0 110.5,66.4 113.6,70.6 116.7,74.8 119.8,78.9 123.0,82.9 126.1,86.8 129.2,90.6 132.3,94.4 135.4,98.0 138.5,101.7 141.7,105.2 144.8,108.6 147.9,112.0 151.0,115.3 154.1,118.6 157.2,121.8 160.4,124.9 163.5,127.9 166.6,130.9 169.7,133.8 172.8,136.7 175.9,139.5 179.0,142.2 182.2,144.9 185.3,147.5 188.4,150.0 191.5,152.5 194.6,154.9 197.7,157.3 200.9,159.6 204.0,161.9 207.1,164.1 210.2,166.2 213.3,168.3 216.4,170.4 219.5,172.4 222.7,174.3 225.8,176.2 228.9,178.1 232.0,179.9 235.1,181.6 238.2,183.3 241.4,185.0 244.5,186.6 247.6,188.2 250.7,189.7 253.8,191.2 256.9,192.6 260.1,194.0 263.2,195.3 266.3,196.6 269.4,197.9 272.5,199.1 275.6,200.2 278.7,201.4 281.9,202.5 285.0,203.5 288.1,204.5 291.2,205.5 294.3,206.4 297.4,207.3 300.6,208.2 303.7,209.0 306.8,209.8 309.9,210.5 313.0,211.2 316.1,211.9 319.2,212.5 322.4,213.1 325.5,213.7 328.6,214.2 331.7,214.7 334.8,215.1 337.9,215.6 341.1,215.9 344.2,216.3 347.3,216.6 350.4,216.9 353.5,217.1 356.6,217.3 359.7,217.5 362.9,217.7 366.0,217.8 369.1,217.8 372.2,217.9 375.3,217.9 378.4,217.9 381.6,217.8 384.7,217.7 387.8,217.6 390.9,217.5 394.0,217.3 397.1,217.1 400.3,216.8 403.4,216.5 406.5,216.2 409.6,215.9 412.7,215.5 415.8,215.1 418.9,214.6 422.1,214.2 425.2,213.7 428.3,213.1 431.4,212.5 434.5,211.9 437.6,211.3 440.8,210.6 443.9,209.9 447.0,209.2 450.1,208.4 453.2,207.6 456.3,206.8 459.4,205.9 462.6,205.0 465.7,204.1 468.8,203.1 471.9,202.1 475.0,201.1 478.1,200.0 481.3,198.9 484.4,197.8 487.5,196.6 490.6,195.4 493.7,194.1 496.8,192.9 499.9,191.5 503.1,190.2 506.2,188.8 509.3,187.4 512.4,185.9 515.5,184.4 518.6,182.9 521.8,181.3 524.9,179.7 528.0,178.1 531.1,176.4 534.2,174.7 537.3,172.9 540.5,171.1 543.6,169.3 546.7,167.4 549.8,165.5 552.9,163.6 556.0,161.6 559.1,159.5 562.3,157.4 565.4,155.3 568.5,153.2 571.6,151.0 574.7,148.7 577.8,146.4 581.0,144.1 584.1,141.7 587.2,139.3 590.3,136.8 593.4,134.3 596.5,131.7 599.6,129.1 602.8,126.4 605.9,123.7 609.0,121.0 612.1,118.2 615.2,115.3 618.3,112.4 621.5,109.4 624.6,106.4 627.7,103.4 630.8,100.3 633.9,97.1 637.0,93.9 640.2,90.6 643.3,87.3 646.4,83.9 649.5,80.4 652.6,76.9 655.7,73.4 658.8,69.7 662.0,66.1 665.1,62.3 668.2,58.5 671.3,54.7 674.4,50.7 677.5,46.7"/><circle class="s1" cx="375.3" cy="217.9" r="5"/><text class="t-s1" x="383.327" y="235.905">10²²</text><polyline class="l4" points="91.8,47.9 94.9,52.8 98.0,57.6 101.2,62.4 104.3,67.0 107.4,71.6 110.5,76.0 113.6,80.4 116.7,84.7 119.8,88.9 123.0,93.1 126.1,97.1 129.2,101.1 132.3,105.1 135.4,108.9 138.5,112.7 141.7,116.4 144.8,120.0 147.9,123.5 151.0,127.0 154.1,130.4 157.2,133.8 160.4,137.1 163.5,140.3 166.6,143.5 169.7,146.6 172.8,149.6 175.9,152.6 179.0,155.5 182.2,158.4 185.3,161.2 188.4,163.9 191.5,166.6 194.6,169.2 197.7,171.8 200.9,174.3 204.0,176.8 207.1,179.3 210.2,181.6 213.3,184.0 216.4,186.2 219.5,188.5 222.7,190.7 225.8,192.8 228.9,194.9 232.0,196.9 235.1,198.9 238.2,200.9 241.4,202.8 244.5,204.7 247.6,206.5 250.7,208.3 253.8,210.0 256.9,211.7 260.1,213.4 263.2,215.0 266.3,216.6 269.4,218.2 272.5,219.7 275.6,221.2 278.7,222.6 281.9,224.0 285.0,225.4 288.1,226.7 291.2,228.0 294.3,229.2 297.4,230.5 300.6,231.7 303.7,232.8 306.8,234.0 309.9,235.1 313.0,236.1 316.1,237.2 319.2,238.2 322.4,239.1 325.5,240.1 328.6,241.0 331.7,241.9 334.8,242.7 337.9,243.5 341.1,244.3 344.2,245.1 347.3,245.8 350.4,246.5 353.5,247.2 356.6,247.9 359.7,248.5 362.9,249.1 366.0,249.7 369.1,250.2 372.2,250.7 375.3,251.2 378.4,251.7 381.6,252.1 384.7,252.6 387.8,252.9 390.9,253.3 394.0,253.7 397.1,254.0 400.3,254.3 403.4,254.5 406.5,254.8 409.6,255.0 412.7,255.2 415.8,255.4 418.9,255.5 422.1,255.6 425.2,255.7 428.3,255.8 431.4,255.9 434.5,255.9 437.6,255.9 440.8,255.9 443.9,255.9 447.0,255.8 450.1,255.7 453.2,255.6 456.3,255.5 459.4,255.3 462.6,255.1 465.7,254.9 468.8,254.7 471.9,254.5 475.0,254.2 478.1,253.9 481.3,253.6 484.4,253.3 487.5,252.9 490.6,252.5 493.7,252.1 496.8,251.7 499.9,251.3 503.1,250.8 506.2,250.3 509.3,249.8 512.4,249.2 515.5,248.7 518.6,248.1 521.8,247.5 524.9,246.8 528.0,246.2 531.1,245.5 534.2,244.8 537.3,244.1 540.5,243.3 543.6,242.5 546.7,241.7 549.8,240.9 552.9,240.1 556.0,239.2 559.1,238.3 562.3,237.4 565.4,236.4 568.5,235.5 571.6,234.5 574.7,233.4 577.8,232.4 581.0,231.3 584.1,230.2 587.2,229.1 590.3,227.9 593.4,226.7 596.5,225.5 599.6,224.3 602.8,223.0 605.9,221.7 609.0,220.4 612.1,219.1 615.2,217.7 618.3,216.3 621.5,214.9 624.6,213.4 627.7,211.9 630.8,210.4 633.9,208.9 637.0,207.3 640.2,205.7 643.3,204.0 646.4,202.3 649.5,200.6 652.6,198.9 655.7,197.1 658.8,195.3 662.0,193.5 665.1,191.6 668.2,189.7 671.3,187.8 674.4,185.8 677.5,183.8 680.7,181.8 683.8,179.7 686.9,177.6 690.0,175.4"/><circle class="s4" cx="437.6" cy="255.9" r="5"/><text class="t-s4" x="445.638" y="273.908">10²³</text><polyline class="key-line" points="88.7,46.6 91.8,51.7 94.9,56.7 98.0,61.5 101.2,66.3 104.3,71.0 107.4,75.6 110.5,80.2 113.6,84.6 116.7,89.0 119.8,93.3 123.0,97.5 126.1,101.6 129.2,105.6 132.3,109.6 135.4,113.5 138.5,117.4 141.7,121.1 144.8,124.8 147.9,128.4 151.0,132.0 154.1,135.5 157.2,138.9 160.4,142.3 163.5,145.6 166.6,148.8 169.7,152.0 172.8,155.1 175.9,158.2 179.0,161.2 182.2,164.1 185.3,167.0 188.4,169.9 191.5,172.6 194.6,175.4 197.7,178.0 200.9,180.7 204.0,183.2 207.1,185.7 210.2,188.2 213.3,190.6 216.4,193.0 219.5,195.4 222.7,197.6 225.8,199.9 228.9,202.1 232.0,204.2 235.1,206.3 238.2,208.4 241.4,210.4 244.5,212.4 247.6,214.3 250.7,216.2 253.8,218.1 256.9,219.9 260.1,221.7 263.2,223.5 266.3,225.2 269.4,226.9 272.5,228.5 275.6,230.1 278.7,231.7 281.9,233.2 285.0,234.7 288.1,236.2 291.2,237.6 294.3,239.0 297.4,240.4 300.6,241.7 303.7,243.0 306.8,244.3 309.9,245.6 313.0,246.8 316.1,248.0 319.2,249.1 322.4,250.3 325.5,251.4 328.6,252.4 331.7,253.5 334.8,254.5 337.9,255.5 341.1,256.5 344.2,257.4 347.3,258.3 350.4,259.2 353.5,260.1 356.6,260.9 359.7,261.8 362.9,262.6 366.0,263.3 369.1,264.1 372.2,264.8 375.3,265.5 378.4,266.2 381.6,266.8 384.7,267.5 387.8,268.1 390.9,268.7 394.0,269.2 397.1,269.8 400.3,270.3 403.4,270.8 406.5,271.3 409.6,271.7 412.7,272.2 415.8,272.6 418.9,273.0 422.1,273.4 425.2,273.7 428.3,274.1 431.4,274.4 434.5,274.7 437.6,275.0 440.8,275.3 443.9,275.5 447.0,275.7 450.1,275.9 453.2,276.1 456.3,276.3 459.4,276.5 462.6,276.6 465.7,276.7 468.8,276.8 471.9,276.9 475.0,277.0 478.1,277.0 481.3,277.0 484.4,277.0 487.5,277.0 490.6,277.0 493.7,277.0 496.8,276.9 499.9,276.8 503.1,276.7 506.2,276.6 509.3,276.5 512.4,276.3 515.5,276.2 518.6,276.0 521.8,275.8 524.9,275.6 528.0,275.3 531.1,275.1 534.2,274.8 537.3,274.5 540.5,274.2 543.6,273.9 546.7,273.5 549.8,273.2 552.9,272.8 556.0,272.4 559.1,272.0 562.3,271.6 565.4,271.1 568.5,270.7 571.6,270.2 574.7,269.7 577.8,269.2 581.0,268.6 584.1,268.1 587.2,267.5 590.3,266.9 593.4,266.3 596.5,265.7 599.6,265.0 602.8,264.4 605.9,263.7 609.0,263.0 612.1,262.3 615.2,261.5 618.3,260.8 621.5,260.0 624.6,259.2 627.7,258.4 630.8,257.5 633.9,256.7 637.0,255.8 640.2,254.9 643.3,254.0 646.4,253.1 649.5,252.1 652.6,251.1 655.7,250.1 658.8,249.1 662.0,248.0 665.1,247.0 668.2,245.9 671.3,244.8 674.4,243.6 677.5,242.5 680.7,241.3 683.8,240.1 686.9,238.9 690.0,237.6"/><circle class="s2" cx="484.4" cy="277.0" r="5"/><text class="t-s2" x="492.372" y="295.032">5.76 × 10²³ (Gopher, Chinchilla)</text><text class="t-muted" x="80" y="288">dot = lowest loss for that budget</text></svg><figcaption>Drawn from the paper's fitted formula L(N, D) = 1.69 + 406.4/N^0.34 + 410.7/D^0.28, not from its measured runs. Too small a model starves on parameters; too large a model starves on tokens.</figcaption></figure>

Calculus gives the bottom of each U in closed form:

$$
N_{\text{opt}}(C) = G \left( \frac{C}{6} \right)^{\frac{\beta}{\alpha + \beta}}, \qquad D_{\text{opt}}(C) = \frac{C}{6 \, N_{\text{opt}}(C)}, \qquad G = \left( \frac{\alpha A}{\beta B} \right)^{\frac{1}{\alpha + \beta}}
$$

With the fitted values, the exponent is 0.452 for $$N$$ and 0.548 for $$D$$. So as the budget grows, the best model grows a little slower than its data, and the two grow at almost the same rate. That is the paper's main finding, and it is very different from Kaplan's 0.73.

> [!WARNING] The famous "20 tokens per parameter"
> The rule of thumb comes from the paper's Table 3 (its first method): 67B parameters and 1.5 trillion tokens for Gopher's budget. The fitted formula above, the paper's third method, gives a smaller model with more tokens: 32.2B parameters and about 93 tokens per parameter. In 2024, Tamay Besiroglu and colleagues at Epoch AI refitted the formula to the paper's own data. Their values, $$E = 1.8172$$, $$A = 482.01$$, $$B = 2085.43$$, $$\alpha = 0.3478$$ and $$\beta = 0.3658$$, give 18.4 tokens per parameter for that budget. That agrees with the 20 of the first two methods.

### Over-training on purpose

Chinchilla answers one question: the lowest loss for a training budget. But a model is trained once and then used millions of times. Every user's token costs about $$2N$$ FLOPs to produce, so a smaller model is cheaper forever. If you expect heavy use, you should train a smaller model for longer than "optimal".

> [!PAPER] Llama Team, Meta (2024), The Llama 3 Herd of Models · §1 · pages 1 and 2
> [![Two passages from the Llama 3 paper: the flagship model was pre-trained using 3.8 x 10^25 FLOPs with 405B trainable parameters on 15.6T text tokens; and we also train our smaller models for much longer than is compute-optimal, and they perform better than compute-optimal models at the same inference budget](/img/writings/history/h-llama3-scale.png)](/img/writings/history/h-llama3-scale.png)
>
> **What it says:** the flagship was "pre-trained using $$3.8 \times 10^{25}$$ FLOPs" with "405B trainable parameters on 15.6T text tokens". Its size is "approximately compute-optimal", but "we also train our smaller models for much longer than is compute-optimal. The resulting models perform better than compute-optimal models at the same inference budget."
>
> **What it means:**
>
> "Inference budget" is the cost of using the model. The 8B model was trained on about 15 trillion tokens too: almost two thousand per parameter, about a hundred times the Chinchilla ratio. Meta fitted its own scaling laws, on its own data, to pick 405B for the flagship. Then it broke the rule for the small models on purpose, because they would serve the most users.

Our script puts numbers on that trade, with the Chinchilla fit as a rough guide (Meta's data and tokenizer differ). An 8B model on 15T tokens uses $$7.2 \times 10^{23}$$ FLOPs and reaches a fitted loss of 1.9485. The compute-optimal model for the same budget has 35.6B parameters and a loss of 1.9226. The small model gives up only 0.0259 of loss, and it is 4.5 times cheaper per token to serve. The same 8B model with 20 tokens per parameter would reach only 2.1640.

### The data wall, and a second axis

If models keep growing, and small ones keep reading more, the data must grow too. Epoch AI estimated in 2024 that the stock of good public human text is on the order of 300 trillion tokens. Their paper's figure is 320 trillion, with a wide range. They expect frontier runs to reach that size "between 2026 and 2032", and sooner if models are over-trained.

> [!DEFINITION] Data wall
> The point where the training data a model needs exceeds the useful public text that exists. Ways around it include repeating data, using more languages and kinds of data, and writing synthetic data with models. Another is to spend more compute at answer time.

Llama 3 405B already read 15.6 trillion tokens, so the wall is not far off for text alone. That is one reason why labs now mix in images, audio and video, and why they write data with models. It is also why the second axis in [Part 4](how-models-are-trained-today.md), **test-time compute**, matters so much. A model that thinks longer can get better without any new pre-training data.

## The data

### Starting from the web

> [!DEFINITION] Common Crawl
> A non-profit that has crawled the public web since 2008 and gives the data away. Its archive holds over 300 billion pages, and a new crawl is published about once a month. Most open pre-training sets start from it.

A crawl is raw HTML: menus, cookie banners, adverts, spam, copies of copies, and pages in hundreds of languages. Maybe a small share is text you would want a model to learn from. The art of pre-training data is throwing away the rest without throwing away the good.

<figure class="fig"><svg viewBox="0 0 720 306" role="img" aria-label="Eight steps in two rows: Common Crawl, text extraction, language identification, heuristic filters, deduplication, a quality classifier, mixing, and tokenisation."><text class="t-title" x="10" y="22">From raw web pages to training tokens</text><rect class="box-2" x="20" y="46" width="155" height="62" rx="6"/><text class="t-note" x="97.5" y="73" text-anchor="middle">Common Crawl</text><text class="t-muted" x="97.5" y="90" text-anchor="middle">raw HTML, monthly</text><polyline class="edge" points="175,77 195,77"/><polygon class="arrow" points="195.0,77.0 187.0,81.0 187.0,73.0"/><rect class="box" x="195" y="46" width="155" height="62" rx="6"/><text class="t-note" x="272.5" y="73" text-anchor="middle">Extract text</text><text class="t-muted" x="272.5" y="90" text-anchor="middle">drop menus, ads, code</text><polyline class="edge" points="350,77 370,77"/><polygon class="arrow" points="370.0,77.0 362.0,81.0 362.0,73.0"/><rect class="box" x="370" y="46" width="155" height="62" rx="6"/><text class="t-note" x="447.5" y="73" text-anchor="middle">Language ID</text><text class="t-muted" x="447.5" y="90" text-anchor="middle">keep the target languages</text><polyline class="edge" points="525,77 545,77"/><polygon class="arrow" points="545.0,77.0 537.0,81.0 537.0,73.0"/><rect class="box" x="545" y="46" width="155" height="62" rx="6"/><text class="t-note" x="622.5" y="73" text-anchor="middle">Heuristic filters</text><text class="t-muted" x="622.5" y="90" text-anchor="middle">length, repeats, spam</text><rect class="box-on" x="20" y="166" width="155" height="62" rx="6"/><text class="t-note" x="97.5" y="193" text-anchor="middle">Deduplicate</text><text class="t-muted" x="97.5" y="210" text-anchor="middle">URL, MinHash, lines</text><polyline class="edge" points="175,197 195,197"/><polygon class="arrow" points="195.0,197.0 187.0,201.0 187.0,193.0"/><rect class="box-on" x="195" y="166" width="155" height="62" rx="6"/><text class="t-note" x="272.5" y="193" text-anchor="middle">Quality classifier</text><text class="t-muted" x="272.5" y="210" text-anchor="middle">keep the best pages</text><polyline class="edge" points="350,197 370,197"/><polygon class="arrow" points="370.0,197.0 362.0,201.0 362.0,193.0"/><rect class="box" x="370" y="166" width="155" height="62" rx="6"/><text class="t-note" x="447.5" y="193" text-anchor="middle">Mix and weight</text><text class="t-muted" x="447.5" y="210" text-anchor="middle">web, code, maths</text><polyline class="edge" points="525,197 545,197"/><polygon class="arrow" points="545.0,197.0 537.0,201.0 537.0,193.0"/><rect class="box-1" x="545" y="166" width="155" height="62" rx="6"/><text class="t-note" x="622.5" y="193" text-anchor="middle">Tokenise</text><text class="t-muted" x="622.5" y="210" text-anchor="middle">text becomes integer IDs</text><polyline class="edge" points="637,108 637,136 97,136 97,166"/><polygon class="arrow" points="97.0,166.0 93.0,158.0 101.0,158.0"/><text class="t-muted" x="20" y="270">FineWeb (2024): 96 Common Crawl snapshots in, 15 trillion tokens out. FineWeb-Edu keeps 1.3 trillion of them.</text><text class="t-muted" x="20" y="290">Llama 3 mix (2024): about 50% general knowledge, 25% maths and reasoning, 17% code, 8% other languages.</text></svg><figcaption>Each step throws data away. Deduplication and the quality classifier (highlighted) changed results the most in the open studies of 2023 and 2024.</figcaption></figure>

The first steps are mechanical. Text is pulled out of the HTML, and a small classifier guesses the language of each page. Simple rules then drop pages that are too short, too repetitive or full of bad words. C4, the "Colossal Clean Crawled Corpus" that Google built for its T5 model in 2019, used rules such as "keep only lines that end with a punctuation mark".

### Deduplication

The web repeats itself. The same news story, licence text or product page appears thousands of times. A model trained on such copies memorises them and wastes compute on them. So every pipeline removes near-duplicates.

> [!DEFINITION] Deduplication
> Removing documents, or lines, that are exact or near copies of others. Exact copies are found with hashes; near copies with methods like MinHash that estimate how much two documents overlap.

Near copies are measured by overlap. Cut each document into **shingles**, runs of a few words in a row, and compare the two sets:

$$
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
$$

$$A$$ and $$B$$ are the sets of shingles of two documents. $$|A \cap B|$$ counts the shingles they share, and $$|A \cup B|$$ counts all distinct shingles of both. $$J$$ is the **Jaccard similarity**: 1 for identical sets, 0 for no overlap.

Take "the cat sat on the mat and looked out of the window" and the same sentence ending in "door". With 3-word shingles, each has 10. They share 9, and the union has 11, so $$J = 9/11 = 0.8182$$. They are near duplicates.

Comparing every pair of billions of pages is impossible. **MinHash** gives each document a short signature of hash values, so that two signatures agree at one spot with probability $$J$$. FineWeb uses 112 hashes in 14 buckets of 8. Two documents are flagged if all 8 hashes agree in any one bucket. The chance of that is $$1 - (1 - J^8)^{14}$$. From our script, it is 0.0533 at $$J = 0.5$$, 0.7716 at 0.75 and 0.9996 at 0.9: a soft threshold near 75%.

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale · Abstract · page 1
> [![The FineWeb abstract, with 15-trillion token dataset derived from 96 Common Crawl snapshots and 1.3-trillion token collection of educational text highlighted](/img/writings/history/h-fineweb-abs.png)](/img/writings/history/h-fineweb-abs.png)
>
> **What it says:** the data sets of the best open models "are not publicly available and very little is known about how they were created". FineWeb is "a 15-trillion token dataset derived from 96 Common Crawl snapshots", and FineWeb-Edu is "a 1.3-trillion token collection of educational text filtered from FineWeb".
>
> **What it means:**
>
> In 2024, data recipes were the most secret part of an open model: weights were public, data was not. The Hugging Face team rebuilt a full pipeline in the open and tested every step by training small models. A snapshot is one monthly crawl, so 96 snapshots cover years of the web. The paper is the best public guide to how such pipelines behave.

One of its results surprised the authors. They first deduplicated across all 96 snapshots at once, as you might expect to do.

> [!PAPER] Penedo et al. (2024), The FineWeb Datasets · §3.4 · page 5
> [![A passage of the FineWeb paper: for an older crawl the data that was kept, 10% of the original, was actually of worse quality than the 90% that was removed; individually deduplicating each snapshot matched RefinedWeb](/img/writings/history/h-fineweb-dedup.png)](/img/writings/history/h-fineweb-dedup.png)
>
> **What it says:** for an older crawl, the data "that was kept (10% of the original data) was actually of worse quality than the 90% of data that was removed". So they tried "individually deduplicating each snapshot", and that matched the RefinedWeb baseline.
>
> **What it means:**
>
> Good pages get copied and re-crawled year after year, so global deduplication removed them from old snapshots. What survived was the junk that nobody copied: adverts and keyword lists. Removing duplicates is good, but removing too many can tilt the data toward what is rare, and rare is not the same as good. Each step of a pipeline has to be tested, not assumed.

### Quality classifiers

Rules can remove obvious junk. They cannot tell a clear explanation from a dull one. For that, labs train a **quality classifier**: a small model that scores each page, so the pipeline keeps the best ones.

> [!DEFINITION] Quality classifier
> A small, fast model that gives each document a score for how useful it is for training, such as "how educational is this page?". It is trained on labels from people or from a large language model, then run over the whole crawl.

For FineWeb-Edu, Llama-3-70B-Instruct scored 460,000 web pages from 0 to 5 for educational value. A small classifier learned those scores and ran over all 15 trillion tokens, which took 6,000 H100 GPU hours. Pages that scored 3 or more were kept: 1.3 trillion tokens. A model trained on them matched the final MMLU score of another open set, Matrix, with almost ten times fewer tokens.

DataComp-LM (DCLM), from a large team of university and industry researchers in June 2024, ran the same kind of test as a contest. It released a pool of 240 trillion tokens from Common Crawl and fixed the training code, so only the data changed. Its best filter, also a small classifier, trained a 7B model to 64% on MMLU with 2.6 trillion tokens. Llama 3's paper describes classifiers of the same kind, trained on Llama 2's judgements.

### Code, maths and data written by models

Code and maths pages are rare on the web, but they teach skills that help with reasoning in general. Pipelines use separate classifiers to find them. Llama 3's final mix had about 50% general knowledge, 25% maths and reasoning, 17% code and 8% text in other languages.

> [!DEFINITION] Synthetic data
> Training text written by a model rather than by people: rewritten web pages, textbook-style explanations, exercises with worked answers, or code with tests. It fills gaps in the web, but it can copy the writer model's errors and narrow the style.

In June 2023, Microsoft's phi-1 paper, "Textbooks Are All You Need", made the case in its title. Its model had only 1.3 billion parameters. It was trained on 6 billion tokens of "textbook quality" web data and 1 billion tokens of textbooks and exercises written by GPT-3.5. It reached 50.6% on the HumanEval coding test, far above much larger models of the time. By 2025, rewriting web text with a model was a standard step, for example in the Kimi K2 report.

### Mixtures, annealing and mid-training

> [!DEFINITION] Data mixture
> The share of each source in the training data: web, code, maths, books, papers, other languages. The mix, not only the total size, sets what the model is good at.

Choosing the mix is itself an experiment. Llama 3 trained small models on candidate mixes, fitted scaling laws to them, and predicted how a large model would do. The end of training gets special care too, because the model learns most from what it reads last, while its learning rate falls.

> [!DEFINITION] Annealing
> The last phase of pre-training, when the learning rate falls toward zero. Labs often switch to a smaller set of very high-quality data here, because these final tokens leave a strong mark on the model.

Llama 3's paper says that "during pre-training on the final 40M tokens, we linearly annealed the learning rate to 0". During that phase, the mix was changed "to upsample data sources of very high quality". The final model is an average of several checkpoints from the annealing phase. Meta also used annealing as a test: train briefly on a new small data set while the rate falls, and see if scores rise.

This end phase has grown into a stage of its own, often called **mid-training**. As [Part 4](how-models-are-trained-today.md) described, Olmo 3 runs 100 billion tokens of maths, code and question answering after its main pre-training. Long-context training usually follows.

## Tokenizers: from text to integers

### Byte-pair encoding

A model reads integers, not letters. A **tokenizer** turns text into a list of token IDs from a fixed **vocabulary**. Whole words would make the vocabulary endless, and single letters would make texts very long. The standard answer sits between them, and it came from data compression.

> [!DEFINITION] Tokenizer and vocabulary
> A tokenizer splits text into tokens and maps each one to an integer ID. The vocabulary is the fixed list of all tokens it knows. Every model is tied to its tokenizer: the same text gives different token counts under different tokenizers.

> [!DEFINITION] Byte-pair encoding (BPE)
> A method that builds a vocabulary from data. It starts with single characters (or bytes). Then it merges the most frequent pair of neighbouring symbols into a new symbol, again and again, until the vocabulary is big enough.

Rico Sennrich, Barry Haddow and Alexandra Birch brought BPE to neural translation in 2016, and printed the whole algorithm in their paper.

> [!PAPER] Sennrich, Haddow and Birch (2016), Neural Machine Translation of Rare Words with Subword Units · Algorithm 1 · page 4
> [![Algorithm 1 of the BPE paper: about twenty lines of Python that count symbol pairs over a word list, merge the most frequent pair, and repeat ten times; the line best = max(pairs, key=pairs.get) is highlighted](/img/writings/history/h-bpe-alg.png)](/img/writings/history/h-bpe-alg.png)
>
> **What it says:** "Algorithm 1 Learn BPE operations". The toy vocabulary is low (5 times), lower (2), newest (6) and widest (3), and the loop runs 10 merges.
>
> **What it means:**
>
> The highlighted line is the whole idea: find the most frequent pair. `get_stats` counts every pair of neighbouring symbols, weighted by how often each word appears. `merge_vocab` glues the winning pair together everywhere. The `</w>` marks the end of a word. So "est" at the end of a word can become a token of its own, apart from "est" in the middle.

We ran the same toy example. The first merge is "e" + "s", with a count of 9 (6 from newest, 3 from widest). Then "es" + "t", then "est" + "</w>", all with 9. Then "l" + "o" and "lo" + "w", with 7. After 10 merges, the word "lowest", which was never in the data, splits into "low" and "est</w>".

<figure class="fig"><svg viewBox="0 0 720 368" role="img" aria-label="A table of byte-pair merges on four words. e and s merge first, then es and t, then est and the end marker, then l and o, then lo and w. Later, new and est end-of-word join into newest."><text class="t-title" x="10" y="22">Byte-pair encoding: merge the most frequent pair, again and again</text><text class="t-muted" x="20" y="50">words and counts:  low ×5,  lower ×2,  newest ×6,  widest ×3       (&lt;/w&gt; marks the end of a word)</text><text class="t-label" x="20" y="78">merge (count)</text><text class="t-label" x="172" y="78">low</text><text class="t-label" x="292" y="78">lower</text><text class="t-label" x="432" y="78">newest</text><text class="t-label" x="574" y="78">widest</text><rect class="box" x="14" y="86" width="692" height="26" rx="4"/><text class="t-note" x="20" y="104">start</text><text class="t-mono" x="172" y="104">l o w &lt;/w&gt;</text><text class="t-mono" x="292" y="104">l o w e r &lt;/w&gt;</text><text class="t-mono" x="432" y="104">n e w e s t &lt;/w&gt;</text><text class="t-mono" x="574" y="104">w i d e s t &lt;/w&gt;</text><rect class="box" x="14" y="116" width="692" height="26" rx="4"/><text class="t-note" x="20" y="134">1: e + s (9)</text><text class="t-mono" x="172" y="134">l o w &lt;/w&gt;</text><text class="t-mono" x="292" y="134">l o w e r &lt;/w&gt;</text><text class="t-mono" x="432" y="134">n e w es t &lt;/w&gt;</text><text class="t-mono" x="574" y="134">w i d es t &lt;/w&gt;</text><rect class="box" x="14" y="146" width="692" height="26" rx="4"/><text class="t-note" x="20" y="164">2: es + t (9)</text><text class="t-mono" x="172" y="164">l o w &lt;/w&gt;</text><text class="t-mono" x="292" y="164">l o w e r &lt;/w&gt;</text><text class="t-mono" x="432" y="164">n e w est &lt;/w&gt;</text><text class="t-mono" x="574" y="164">w i d est &lt;/w&gt;</text><rect class="box-on" x="14" y="176" width="692" height="26" rx="4"/><text class="t-note" x="20" y="194">3: est + &lt;/w&gt; (9)</text><text class="t-mono" x="172" y="194">l o w &lt;/w&gt;</text><text class="t-mono" x="292" y="194">l o w e r &lt;/w&gt;</text><text class="t-mono" x="432" y="194">n e w est&lt;/w&gt;</text><text class="t-mono" x="574" y="194">w i d est&lt;/w&gt;</text><rect class="box" x="14" y="206" width="692" height="26" rx="4"/><text class="t-note" x="20" y="224">4: l + o (7)</text><text class="t-mono" x="172" y="224">lo w &lt;/w&gt;</text><text class="t-mono" x="292" y="224">lo w e r &lt;/w&gt;</text><text class="t-mono" x="432" y="224">n e w est&lt;/w&gt;</text><text class="t-mono" x="574" y="224">w i d est&lt;/w&gt;</text><rect class="box" x="14" y="236" width="692" height="26" rx="4"/><text class="t-note" x="20" y="254">5: lo + w (7)</text><text class="t-mono" x="172" y="254">low &lt;/w&gt;</text><text class="t-mono" x="292" y="254">low e r &lt;/w&gt;</text><text class="t-mono" x="432" y="254">n e w est&lt;/w&gt;</text><text class="t-mono" x="574" y="254">w i d est&lt;/w&gt;</text><text class="t-muted" x="20" y="280">... merges 6 and 7: n + e, ne + w ...</text><rect class="box-on" x="14" y="292" width="692" height="26" rx="4"/><text class="t-note" x="20" y="310">8: new + est&lt;/w&gt; (6)</text><text class="t-mono" x="172" y="310">low &lt;/w&gt;</text><text class="t-mono" x="292" y="310">low e r &lt;/w&gt;</text><text class="t-mono" x="432" y="310">newest&lt;/w&gt;</text><text class="t-mono" x="574" y="310">w i d est&lt;/w&gt;</text><text class="t-muted" x="20" y="352">After 10 merges, "lowest" (never seen) becomes  low + est&lt;/w&gt;.  The vocabulary grows by one symbol per merge.</text></svg><figcaption>The toy vocabulary of Sennrich, Haddow and Birch (2016), run by our script with the paper's own algorithm. Real vocabularies, built on bytes instead of letters, hold 32,000 to 262,000 tokens.</figcaption></figure>

### Vocabulary sizes, then and now

GPT-2 in 2019 ran BPE on bytes, not letters, so any text, in any script, can be encoded. Its vocabulary had 50,257 tokens. Llama 2 in 2023 used 32,000. Since then, vocabularies have grown: Llama 3 has 128,000 tokens (100,000 from OpenAI's tiktoken plus 28,000 for other languages). DeepSeek-V3 has 128,000, and Gemma 3 uses Gemini 2.0's tokenizer with 262,000 entries.

A bigger vocabulary means fewer tokens per text, so the model reads more text for the same compute. Meta reports that its new tokenizer raised the average from 3.17 to 3.94 English characters per token, compared with Llama 2's. The cost is a bigger embedding table and output layer, and rare tokens that the model sees too seldom to learn well. Languages that were poorly served by older tokenizers, where one word could cost five or more tokens, gain the most.

## The 2026 decoder block

The Transformer of 2017 had an encoder and a decoder for translation. Almost every large language model since GPT has kept only the decoder: a stack of identical layers, each with attention and a feed-forward network. The layer of 2026 looks like the old one from far away. Up close, almost every part has been swapped.

<figure class="fig"><svg viewBox="0 0 720 528" role="img" aria-label="Two stacks side by side. The 2017 layer adds sine positions, runs multi-head attention, then add and LayerNorm, a ReLU feed-forward, then add and LayerNorm. The 2026 layer runs RMSNorm, attention with RoPE and shared keys and values, a residual add, RMSNorm, a SwiGLU or mixture-of-experts feed-forward, and another residual add."><text class="t-title" x="10" y="22">One Transformer layer, 2017 and 2026</text><text class="t-note" x="170" y="44" text-anchor="middle">2017: "Attention Is All You Need"</text><text class="t-muted" x="170" y="388" text-anchor="middle">tokens in</text><rect class="box" x="70" y="310" width="200" height="40" rx="6"/><text class="t-note" x="170" y="326" text-anchor="middle">+ sine positions</text><text class="t-muted" x="170" y="343" text-anchor="middle">(added once, at the input)</text><polyline class="edge" points="170,366 170,350"/><polygon class="arrow" points="170.0,350.0 174.0,358.0 166.0,358.0"/><rect class="box-2" x="70" y="254" width="200" height="40" rx="6"/><text class="t-note" x="170" y="270" text-anchor="middle">Multi-head attention</text><text class="t-muted" x="170" y="287" text-anchor="middle">every head has its own K, V</text><polyline class="edge" points="170,310 170,294"/><polygon class="arrow" points="170.0,294.0 174.0,302.0 166.0,302.0"/><rect class="box" x="70" y="198" width="200" height="40" rx="6"/><text class="t-note" x="170" y="214" text-anchor="middle">Add, then LayerNorm</text><text class="t-muted" x="170" y="231" text-anchor="middle">(post-norm)</text><polyline class="edge" points="170,254 170,238"/><polygon class="arrow" points="170.0,238.0 174.0,246.0 166.0,246.0"/><rect class="box-3" x="70" y="142" width="200" height="40" rx="6"/><text class="t-note" x="170" y="158" text-anchor="middle">Feed-forward</text><text class="t-muted" x="170" y="175" text-anchor="middle">ReLU, 4d wide</text><polyline class="edge" points="170,198 170,182"/><polygon class="arrow" points="170.0,182.0 174.0,190.0 166.0,190.0"/><rect class="box" x="70" y="96" width="200" height="30" rx="6"/><text class="t-note" x="170" y="115.5" text-anchor="middle">Add, then LayerNorm</text><polyline class="edge" points="170,142 170,126"/><polygon class="arrow" points="170.0,126.0 174.0,134.0 166.0,134.0"/><polyline class="edge" points="170,96 170,80"/><polygon class="arrow" points="170.0,80.0 174.0,88.0 166.0,88.0"/><text class="t-muted" x="170" y="74" text-anchor="middle">next layer</text><text class="t-muted" x="30" y="462">Dense: every token uses every weight.</text><text class="t-muted" x="30" y="480">Encoder and decoder, for translation.</text><text class="t-muted" x="30" y="498">Base model: 65M parameters.</text><text class="t-note" x="530" y="44" text-anchor="middle">2026: a typical decoder (Llama 3, DeepSeek-V3)</text><text class="t-muted" x="530" y="424" text-anchor="middle">tokens in</text><rect class="box-on" x="430" y="346" width="200" height="40" rx="6"/><text class="t-note" x="530" y="362" text-anchor="middle">RMSNorm</text><text class="t-muted" x="530" y="379" text-anchor="middle">(pre-norm, before the block)</text><polyline class="edge" points="530,402 530,386"/><polygon class="arrow" points="530.0,386.0 534.0,394.0 526.0,394.0"/><rect class="box-2" x="430" y="290" width="200" height="40" rx="6"/><text class="t-note" x="530" y="306" text-anchor="middle">Attention with RoPE</text><text class="t-muted" x="530" y="323" text-anchor="middle">GQA or MLA: shared K, V</text><polyline class="edge" points="530,346 530,330"/><polygon class="arrow" points="530.0,330.0 534.0,338.0 526.0,338.0"/><rect class="box" x="430" y="244" width="200" height="30" rx="6"/><text class="t-note" x="530" y="263.5" text-anchor="middle">Add (residual)</text><polyline class="edge" points="530,290 530,274"/><polygon class="arrow" points="530.0,274.0 534.0,282.0 526.0,282.0"/><rect class="box-on" x="430" y="198" width="200" height="30" rx="6"/><text class="t-note" x="530" y="217.5" text-anchor="middle">RMSNorm</text><polyline class="edge" points="530,244 530,228"/><polygon class="arrow" points="530.0,228.0 534.0,236.0 526.0,236.0"/><rect class="box-3" x="430" y="142" width="200" height="40" rx="6"/><text class="t-note" x="530" y="158" text-anchor="middle">SwiGLU feed-forward,</text><text class="t-muted" x="530" y="175" text-anchor="middle">often a mixture of experts</text><polyline class="edge" points="530,198 530,182"/><polygon class="arrow" points="530.0,182.0 534.0,190.0 526.0,190.0"/><rect class="box" x="430" y="96" width="200" height="30" rx="6"/><text class="t-note" x="530" y="115.5" text-anchor="middle">Add (residual)</text><polyline class="edge" points="530,142 530,126"/><polygon class="arrow" points="530.0,126.0 534.0,134.0 526.0,134.0"/><polyline class="edge" points="530,96 530,80"/><polygon class="arrow" points="530.0,80.0 534.0,88.0 526.0,88.0"/><text class="t-muted" x="530" y="74" text-anchor="middle">next layer</text><text class="t-muted" x="390" y="462">Decoder only; 128K to 1M+ tokens.</text><text class="t-muted" x="390" y="480">Often sparse: a few experts per token.</text><text class="t-muted" x="390" y="498">Llama 3 405B: 126 layers deep.</text><line class="edge-dim" x1="360" y1="36" x2="360" y2="510"/></svg><figcaption>Same skeleton, new parts. The norm moved before each block, the positions moved into attention, the keys and values are shared, and the feed-forward got a gate and often a router.</figcaption></figure>

### Pre-norm and RMSNorm

Deep networks need their numbers kept at a steady size from layer to layer, or training blows up or stalls. **Normalisation** does this. The 2017 Transformer normalised after adding each block's output to its input ("post-norm"). GPT-2 moved the normalisation to the input of each block ("pre-norm"). In 2020, Ruibin Xiong and colleagues showed why that helps. With pre-norm, the gradients are well behaved at the start, so training needs less of a careful warm-up.

> [!DEFINITION] Normalisation layer
> A layer that rescales a vector to a standard size before the next block uses it. It keeps signals and gradients from growing or shrinking as they pass through many layers.

In 2019, Biao Zhang and Rico Sennrich proposed a cheaper normaliser. LayerNorm subtracts the mean and divides by the standard deviation. **RMSNorm** skips the mean and divides by the root mean square.

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\frac{1}{d} \sum_{j=1}^{d} x_j^2 + \epsilon}} \; g_i
$$

$$x$$ is the vector for one token, with $$d$$ numbers. The square root is the **root mean square**: square every number, take the average, take the root. $$\epsilon$$ (epsilon) is a tiny number that prevents division by zero, and $$g_i$$ is a learned scale for each position.

Take $$x = [1, 2, 3, 4]$$, with $$g = 1$$. The mean of the squares is $$(1 + 4 + 9 + 16)/4 = 7.5$$, and its root is 2.7386. RMSNorm gives $$[0.3651, 0.7303, 1.0954, 1.4606]$$. LayerNorm would first subtract the mean 2.5 and give $$[-1.3416, -0.4472, 0.4472, 1.3416]$$. RMSNorm keeps the direction of the vector and only fixes its size. The paper reports the same quality as LayerNorm and running times 7% to 64% shorter.

### SwiGLU: a feed-forward network with a gate

The feed-forward network is where most of a dense model's parameters live. In 2017 it was two matrices with a ReLU between them. In 2020, Noam Shazeer tested a family of **gated** versions, and one, SwiGLU, became the standard.

> [!DEFINITION] Gated linear unit (GLU)
> A layer that computes two projections of the input and multiplies them element by element. One projection is the signal; the other, passed through an activation function, is a gate that decides how much of the signal goes through.

$$
\text{FFN}_{\text{SwiGLU}}(x) = \big( \text{SiLU}(x W) \odot x V \big) W_2, \qquad \text{SiLU}(z) = \frac{z}{1 + e^{-z}}
$$

$$x$$ is a token's vector. $$W$$ and $$V$$ are two matrices that widen it to $$d_{\text{ff}}$$ numbers, and $$W_2$$ brings it back to $$d$$. $$\odot$$ means multiply element by element. **SiLU** (also called Swish) is a smooth version of ReLU. It is close to $$z$$ for large $$z$$ and close to 0 for very negative $$z$$.

Take one element with $$xW = 2$$ and $$xV = 3$$. SiLU(2) is 1.7616, so the output is $$1.7616 \times 3 = 5.2848$$. If $$xV$$ were 0, the output would be 0 whatever the gate said. The two paths learn together which features to pass on.

SwiGLU has three matrices instead of two. To keep the parameter count the same, $$d_{\text{ff}}$$ is set near $$\frac{8}{3} d$$ instead of $$4d$$, since $$3 \times d \times \frac{8}{3}d = 8d^2$$ is what two matrices of $$d \times 4d$$ cost. Llama 3 8B uses $$d = 4{,}096$$ and $$d_{\text{ff}} = 14{,}336$$, which is 3.5 times wider. Its feed-forward weights alone come to 176,160,768 per layer.

> [!PAPER] Shazeer (2020), GLU Variants Improve Transformer · §4 Conclusions · page 3
> [![The conclusion of the GLU variants paper, with We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence highlighted](/img/writings/history/h-swiglu.png)](/img/writings/history/h-swiglu.png)
>
> **What it says:** the new variants give "better perplexities" and better results on many tasks, are "simple to implement, and have no apparent computational drawbacks". Then: "We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence."
>
> **What it means:**
>
> It is a joke, and an honest one. Much of a modern block was found by trying variants and keeping the one with the lower loss. The theory came later, or never. SwiGLU is now in Llama, Qwen, DeepSeek, Mistral and many other models, still without a full explanation.

### RoPE: positions as rotations

Attention, on its own, sees a set of tokens with no order. The 2017 Transformer added a fixed wave pattern to each token's vector at the input. That works, but the model must learn to pull position out of the mixture again, and it handles texts longer than its training texts poorly.

> [!DEFINITION] Rotary position embedding (RoPE)
> A way to give attention the order of tokens. It turns each pair of numbers in a query or key vector by an angle proportional to the token's position. The dot product of a turned query and key then depends only on how far apart the two tokens are.

In 2021, Jianlin Su and colleagues at Zhuiyi Technology proposed rotating the queries and keys instead. Think of a clock hand. A token at position $$m$$ has its hand turned by $$m$$ steps. Two hands turned by 3 and 1 steps make the same angle as two hands turned by 7 and 5 steps. Attention compares angles, so it sees only the distance.

For one pair of numbers, the rotation is a 2-by-2 matrix:

$$
R(m\theta) = \begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}, \qquad \big( R(m\theta)\, q \big)^{\top} \big( R(n\theta)\, k \big) = q^{\top} R\big( (n - m)\theta \big)\, k
$$

$$q$$ and $$k$$ are a pair of numbers from a query and a key. $$m$$ and $$n$$ are the positions of their tokens, and $$\theta$$ (theta) is the turning speed for this pair. The right side shows that the score depends on $$m$$ and $$n$$ only through $$n - m$$. That works because turning one vector by $$a$$ and another by $$b$$ changes the angle between them by $$b - a$$.

Our script checks it with $$q = (1, 0.5)$$, $$k = (0.8, -0.2)$$ and $$\theta = 0.3$$. Without rotation, $$q \cdot k = 0.7$$. With the query at position 3 and the key at 1, the score is 0.2389. At positions 7 and 5 it is 0.2389, and at 102 and 100 it is again 0.2389. At positions 5 and 1, four apart, it is $$-0.3056$$.

<figure class="fig"><svg viewBox="0 0 720 380" role="img" aria-label="Two circles. On the left, a query rotated for position 3 and a key rotated for position 1. On the right, the same vectors rotated for positions 7 and 5. The angle between them is the same in both, so the dot product is 0.2389 in both."><text class="t-title" x="10" y="22">RoPE: turn each pair of numbers by an angle that grows with position</text><text class="t-note" x="190" y="50" text-anchor="middle">positions 3 and 1</text><line class="axis" x1="75" y1="190" x2="305" y2="190"/><line class="axis" x1="190" y1="305" x2="190" y2="75"/><circle class="box-ghost" cx="190" cy="190" r="105"/><line class="edge-on" x1="190" y1="190" x2="211.6" y2="87.2"/><polygon class="arrow-on" points="211.6,87.2 213.9,95.9 206.0,94.3"/><text class="t-note" x="213.652" y="81.4585" text-anchor="start">q at position 3</text><line class="edge-2" x1="190" y1="190" x2="294.8" y2="184.2"/><polygon class="arrow-2" points="294.8,184.2 287.1,188.7 286.6,180.7"/><text class="t-note" x="304.826" y="187.676" text-anchor="start">k at position 1</text><text class="t-note" x="530" y="50" text-anchor="middle">positions 7 and 5: both turned 4θ more</text><line class="axis" x1="415" y1="190" x2="645" y2="190"/><line class="axis" x1="530" y1="305" x2="530" y2="75"/><circle class="box-ghost" cx="530" cy="190" r="105"/><line class="edge-on" x1="530" y1="190" x2="442.1" y2="132.6"/><polygon class="arrow-on" points="442.1,132.6 450.9,133.7 446.6,140.4"/><text class="t-note" x="433.677" y="131.175" text-anchor="end">q at 7</text><line class="edge-2" x1="530" y1="190" x2="562.6" y2="90.2"/><polygon class="arrow-2" points="562.6,90.2 563.9,99.0 556.3,96.6"/><text class="t-note" x="565.714" y="84.6861" text-anchor="start">k at 5</text><text class="t-note" x="360" y="340" text-anchor="middle">The angle between q and k is the same in both pictures, so the dot product is the same: 0.2389.</text><text class="t-muted" x="360" y="362" text-anchor="middle">Attention sees only the distance, 2 tokens, not where the pair sits. (θ = 0.3, from our script.)</text></svg><figcaption>A real head has 64 such pairs, each turning at its own speed, from one radian per token down to a full turn every 54,000 tokens (base 10,000) or 2.6 million tokens (base 500,000).</figcaption></figure>

A real head has many pairs, each with its own speed: $$\theta_i = b^{-2i/d}$$ for pair $$i$$ of a head with $$d$$ numbers, with a **base** $$b$$. Fast pairs track nearby words; slow pairs track far ones. With $$d = 128$$ and the usual base of 10,000, the slowest pair turns once every 54,410 tokens. Llama 3 raised the base to 500,000, which makes the slowest turn last 2,559,196 tokens and helps with long texts.

> [!PAPER] Su et al. (2021), RoFormer: Enhanced Transformer with Rotary Position Embedding · Figure 1 · page 5
> [![Figure 1 of the RoFormer paper: a query or key pair x1, x2 at position m is rotated by m theta to give the position-encoded pair; below, each pair of the vector gets its own theta, for positions 1 to 6](/img/writings/history/h-rope-fig.png)](/img/writings/history/h-rope-fig.png)
>
> **What it says:** "Implementation of Rotary Position Embedding (RoPE)."
>
> **What it means:**
>
> The top box is the clock hand: a pair $$(x_1, x_2)$$ is turned by $$m\theta_1$$ to give $$(x'_1, x'_2)$$. The bottom box shows the whole vector, cut into $$d/2$$ pairs, each with its own $$\theta$$, for tokens at positions 1 to 6. Nothing is added to the token's vector, and nothing is learned: the rotation is applied to queries and keys inside every attention layer.

### Attention cost and FlashAttention

Attention compares every token with every other token. For $$n$$ tokens, that is a table of $$n \times n$$ scores per head and per layer.

$$
\text{FLOPs}_{\text{scores}} \approx 2\, n^2 d \quad \text{per layer}, \qquad \text{memory}_{\text{scores}} = n^2 \text{ numbers per head}
$$

$$n$$ is the sequence length and $$d$$ the model width. Our script shows the memory side for one head in 16-bit numbers. At 2,048 tokens the score table is 8 MiB. At 8,192 it is 128 MiB. At 131,072 tokens it is 32,768 MiB, or 32 GiB, for a single head of a single layer.

> [!DEFINITION] HBM and SRAM
> The two kinds of GPU memory that matter here. **HBM** (high-bandwidth memory) is large, tens of gigabytes, but slow to reach. **SRAM** is the small memory on the chip itself, a few megabytes in total, but about ten times faster.

Most of the time of plain attention goes not into arithmetic but into writing the big score table to HBM and reading it back. In May 2022, Tri Dao and colleagues at Stanford fixed that without changing the result.

> [!PAPER] Dao et al. (2022), FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness · Figure 1 · page 2
> [![Figure 1 of FlashAttention: left, the GPU memory pyramid of SRAM at 19 TB/s and 20 MB, HBM at 1.5 TB/s and 40 GB and CPU DRAM; middle, blocks of K, V and Q copied to SRAM in an outer and inner loop; right, attention on GPT-2 taking about 15 ms in PyTorch and about 2 ms as one fused kernel](/img/writings/history/h-flash-fig.png)](/img/writings/history/h-flash-fig.png)
>
> **What it says:** FlashAttention "uses tiling to prevent materialization of the large $$N \times N$$ attention matrix (dotted box) on (relatively) slow GPU HBM". The right panel shows "a 7.6× speedup on the attention computation" for GPT-2.
>
> **What it means:**
>
> The pyramid on the left is the whole story. SRAM runs at 19 TB/s but holds only 20 MB; HBM runs at 1.5 TB/s and holds 40 GB. FlashAttention cuts Q, K and V into tiles that fit in SRAM. It computes the scores for one tile and folds them into the output with a running softmax.
>
> The full table never exists in HBM. The answer is exact, the same as plain attention, only faster and with memory that grows like $$n$$, not $$n^2$$.

> [!DEFINITION] IO-aware algorithm
> An algorithm designed around the cost of moving data between slow and fast memory, not only the cost of the arithmetic. On modern GPUs, moving data is often the slower part.

The paper reports a 3× speedup on GPT-2 training at 1K tokens. FlashAttention-2 and -3 followed for newer GPUs, and some version of it now runs inside nearly every training and serving system. The arithmetic is still $$n^2$$, which is why long-context models also use the sparser patterns of the [attention series](attention-3-sliding-window-sparse.md), such as Gemma 3's local layers.

### Sharing keys and values: MQA, GQA and MLA

When a model writes text, it keeps the keys and values of every earlier token, so it does not compute them again. This store is the **KV cache**, and the [inference series](llm-inference-2-kv-cache.md) covers it in depth. Its size often decides how many users one GPU can serve.

> [!DEFINITION] KV cache
> The stored keys and values of all earlier tokens, for every layer and every key-value head. Each new token reads the whole cache, so its size and the speed of reading it set the cost of generation.

$$
\text{KV bytes per token} = 2 \times L \times H_{kv} \times d_h \times b
$$

The 2 counts keys and values. $$L$$ is the number of layers and $$H_{kv}$$ the number of key-value heads. $$d_h$$ is the size of each head and $$b$$ the bytes per number (2 for 16-bit). Plain **multi-head attention** (MHA) has one key-value head per query head.

Take the shape of Llama 3 70B: 80 layers, 64 query heads of size 128. With MHA, one token would need $$2 \times 80 \times 64 \times 128 \times 2 = 2{,}621{,}440$$ bytes, or 2,560 KiB. A 128K-token chat would need 320 GiB, four times an 80 GB GPU. That is why nobody builds it that way.

> [!DEFINITION] Multi-query and grouped-query attention (MQA, GQA)
> **MQA** gives all query heads one shared key head and one value head. **GQA** puts query heads in groups, and each group shares one key-value head. GQA sits between MHA and MQA.

Noam Shazeer proposed MQA in 2019. It shrinks the cache by the number of heads, but it can lose quality. In 2023, Joshua Ainslie and colleagues at Google proposed GQA as the middle way. They also showed that an existing MHA model can be converted with about 5% of its original training compute. Llama 3 70B uses 8 key-value heads for its 64 query heads: 320 KiB per token, and 40 GiB for 128K tokens. MQA, with one head, would need 40 KiB.

DeepSeek went further in May 2024 with **multi-head latent attention** in DeepSeek-V2.

> [!DEFINITION] Multi-head latent attention (MLA)
> An attention design that stores one small compressed vector per token and layer, the **latent**, instead of full keys and values. The keys and values of all heads are rebuilt from the latent with learned matrices when they are needed.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model · Figure 3 · page 7
> [![Figure 3 of DeepSeek-V2: four panels comparing multi-head, grouped-query, multi-query and multi-head latent attention; striped boxes mark what is cached during inference; in MLA only one compressed latent KV vector is cached and projected into keys and values](/img/writings/history/h-mla-fig.png)](/img/writings/history/h-mla-fig.png)
>
> **What it says:** a "simplified illustration" of MHA, GQA, MQA and MLA. "Through jointly compressing the keys and values into a latent vector, MLA significantly reduces the KV cache during inference."
>
> **What it means:**
>
> The striped boxes are what stays in memory. MHA keeps every head's keys and values; GQA keeps one pair per group; MQA keeps one pair. MLA keeps a single compressed vector, yet every head still gets its own keys and values, rebuilt by the "projection". The paper reports that DeepSeek-V2 cut its KV cache by 93.3% against its earlier dense model, and raised its top generation speed 5.76 times.

DeepSeek-V3 stores 512 numbers for the latent plus 64 for a small RoPE key, per layer. With 61 layers, that is $$61 \times 576 \times 2 = 70{,}272$$ bytes, or 68.6 KiB per token. The same shape with full MHA, 128 heads of size 128, would need 3,904 KiB. MLA is 56.9 times smaller.

<figure class="fig"><svg viewBox="0 0 720 334" role="img" aria-label="A bar chart of KV-cache size per token. Llama 3 70B shape: 2,560 KiB with multi-head attention, 320 KiB with grouped-query attention, 40 KiB with multi-query attention. DeepSeek-V3 shape: 3,904 KiB with multi-head attention and 68.6 KiB with multi-head latent attention."><text class="t-title" x="10" y="22">KV cache for one token, all layers, 16-bit numbers</text><text class="t-note" x="10" y="64">Llama 3 70B shape, MHA (64 KV heads)</text><rect class="box-4" x="330" y="44" width="150.8" height="28" rx="4"/><text class="t-val" x="488.82" y="63">2,560 KiB</text><text class="t-muted" x="712" y="63" text-anchor="end">320 GiB</text><text class="t-note" x="10" y="108">Llama 3 70B, GQA (8 KV heads, as built)</text><rect class="box-1" x="330" y="88" width="18.9" height="28" rx="4"/><text class="t-val" x="356.852" y="107">320 KiB</text><text class="t-muted" x="712" y="107" text-anchor="end">40 GiB</text><text class="t-note" x="10" y="152">Llama 3 70B shape, MQA (1 KV head)</text><rect class="box" x="330" y="132" width="3.0" height="28" rx="4"/><text class="t-val" x="341" y="151">40 KiB</text><text class="t-muted" x="712" y="151" text-anchor="end">5 GiB</text><text class="t-note" x="10" y="210">DeepSeek-V3 shape, MHA (128 heads)</text><rect class="box-4" x="330" y="190" width="230.0" height="28" rx="4"/><text class="t-val" x="568" y="209">3,904 KiB</text><text class="t-muted" x="712" y="209" text-anchor="end">488 GiB</text><text class="t-note" x="10" y="254">DeepSeek-V3, MLA (576 numbers a layer, as built)</text><rect class="box-1" x="330" y="234" width="4.0" height="28" rx="4"/><text class="t-val" x="342.041" y="253">68.6 KiB</text><text class="t-muted" x="712" y="253" text-anchor="end">8.6 GiB</text><text class="t-muted" x="712" y="40" text-anchor="end">at 128K tokens</text><text class="t-muted" x="10" y="300">GQA cuts the Llama cache 8 times. MLA cuts the DeepSeek cache 56.9 times, because it stores one small latent</text><text class="t-muted" x="10" y="318">vector per layer and rebuilds the keys and values from it. An 80 GB GPU cannot even hold the MHA cache for one long chat.</text></svg><figcaption>Bytes per token = 2 (keys and values) × layers × KV heads × head size × 2 bytes. The MHA rows are what the same shapes would need without sharing. Numbers from our script.</figcaption></figure>

### Long context

A model trained on 8,000-token texts does not simply work on 128,000. Its slow RoPE pairs never turned that far, so long distances look like nothing it has seen. The fixes stretch the rotations.

> [!DEFINITION] Context extension
> Training a model, after its main pre-training, to handle longer inputs than it saw before. It usually changes the RoPE frequencies and continues training on long documents for a short time.

In 2023, Shouyuan Chen and colleagues at Meta proposed **position interpolation**. It divides every position by a scale $$s$$, so 128K positions fit into the old 8K range. From 8K to 128K, $$s = 16$$. Later methods scale different pairs by different amounts, so fast pairs keep their precision. YaRN, from Bowen Peng and colleagues in 2023, needed "10x less tokens and 2.5x less training steps than previous methods".

Labs now do this as a stage of pre-training. Llama 3 went from 8K to 128K "in six stages". DeepSeek-V3 used YaRN to go from 4K to 32K and then to 128K. In February 2024, Google's Gemini 1.5 report showed "near-perfect retrieval (>99%) up to at least 10M tokens" in research tests. The models it shipped took one or two million tokens. Google has not published how it does this.

## Mixture of experts

### More parameters, same cost per token

A dense model uses every weight for every token. That ties two things together: what the model can know (its parameters) and what each token costs (its compute). A **mixture of experts** cuts that tie. The feed-forward block is replaced by many smaller blocks, the experts, and a small network chooses a few of them for each token.

> [!DEFINITION] Mixture of experts (MoE)
> A layer with many feed-forward networks, the **experts**, and a **router** that sends each token to only a few of them. The model has many parameters in total but uses only a fraction for each token. Such a model is called **sparse**; one that uses all its weights is **dense**.

Think of a hospital. It has many specialists, but each patient sees only two or three, chosen at the front desk. The hospital knows a great deal, and each visit stays short. The front desk is the router, and it learns its choices during training.

$$
p = \text{softmax}(x W_r), \qquad y = \sum_{i \in \text{TopK}(p)} \frac{p_i}{\sum_{j \in \text{TopK}(p)} p_j} \, E_i(x)
$$

$$x$$ is a token's vector and $$W_r$$ the router's weight matrix, with one column per expert. $$p$$ holds one probability per expert. $$\text{TopK}(p)$$ is the set of the $$k$$ experts with the highest probabilities. $$E_i(x)$$ is the output of expert $$i$$, and the fraction is its **gate**: its probability, rescaled so the chosen gates add up to 1.

Take 8 experts and a token whose router scores are 2.0, 1.0, 0.5, 0.1, −0.3, −1.0, 0.0 and 0.3. Softmax turns them into 0.4528, 0.1666, 0.1010, 0.0677, 0.0454, 0.0225, 0.0613 and 0.0827. With $$k = 2$$, experts 0 and 1 win. Their gates are $$0.4528 / 0.6194 = 0.7311$$ and 0.2689. The other six experts do no work for this token.

<figure class="fig"><svg viewBox="0 0 720 374" role="img" aria-label="A token goes to a router, which gives eight expert probabilities. Experts 0 and 1 have the highest, 0.4528 and 0.1666. They get gate weights 0.7311 and 0.2689, and the output is their weighted sum."><text class="t-title" x="10" y="22">A mixture-of-experts layer: a router picks 2 of 8 experts for each token</text><rect class="box-2" x="20" y="140" width="110" height="56" rx="6"/><text class="t-note" x="75" y="164" text-anchor="middle">token x</text><text class="t-muted" x="75" y="181" text-anchor="middle">hidden vector</text><rect class="box-on" x="160" y="140" width="110" height="56" rx="6"/><text class="t-note" x="215" y="164" text-anchor="middle">router</text><text class="t-muted" x="215" y="181" text-anchor="middle">softmax(x W_r)</text><polyline class="edge" points="130,168 160,168"/><polygon class="arrow" points="160.0,168.0 152.0,172.0 152.0,164.0"/><rect class="box-on" x="330" y="44" width="113.2" height="24" rx="3"/><text class="t-note" x="322" y="61" text-anchor="end">E0</text><text class="t-val" x="449.2" y="61">0.4528</text><line class="edge-on" x1="270" y1="168" x2="300" y2="56"/><rect class="box-on" x="330" y="80" width="41.6" height="24" rx="3"/><text class="t-note" x="322" y="97" text-anchor="end">E1</text><text class="t-val" x="377.65" y="97">0.1666</text><line class="edge-on" x1="270" y1="168" x2="300" y2="92"/><rect class="box" x="330" y="116" width="25.2" height="24" rx="3"/><text class="t-note" x="322" y="133" text-anchor="end">E2</text><text class="t-muted" x="361.25" y="133">0.1010</text><line class="edge-dim" x1="270" y1="168" x2="300" y2="128"/><rect class="box" x="330" y="152" width="16.9" height="24" rx="3"/><text class="t-note" x="322" y="169" text-anchor="end">E3</text><text class="t-muted" x="352.925" y="169">0.0677</text><line class="edge-dim" x1="270" y1="168" x2="300" y2="164"/><rect class="box" x="330" y="188" width="11.4" height="24" rx="3"/><text class="t-note" x="322" y="205" text-anchor="end">E4</text><text class="t-muted" x="347.35" y="205">0.0454</text><line class="edge-dim" x1="270" y1="168" x2="300" y2="200"/><rect class="box" x="330" y="224" width="5.6" height="24" rx="3"/><text class="t-note" x="322" y="241" text-anchor="end">E5</text><text class="t-muted" x="341.625" y="241">0.0225</text><line class="edge-dim" x1="270" y1="168" x2="300" y2="236"/><rect class="box" x="330" y="260" width="15.3" height="24" rx="3"/><text class="t-note" x="322" y="277" text-anchor="end">E6</text><text class="t-muted" x="351.325" y="277">0.0613</text><line class="edge-dim" x1="270" y1="168" x2="300" y2="272"/><rect class="box" x="330" y="296" width="20.7" height="24" rx="3"/><text class="t-note" x="322" y="313" text-anchor="end">E7</text><text class="t-muted" x="356.675" y="313">0.0827</text><line class="edge-dim" x1="270" y1="168" x2="300" y2="308"/><rect class="box-3" x="560" y="50" width="140" height="50" rx="6"/><text class="t-note" x="630" y="71" text-anchor="middle">FFN expert 0</text><text class="t-muted" x="630" y="88" text-anchor="middle">gate 0.7311</text><rect class="box-3" x="560" y="120" width="140" height="50" rx="6"/><text class="t-note" x="630" y="141" text-anchor="middle">FFN expert 1</text><text class="t-muted" x="630" y="158" text-anchor="middle">gate 0.2689</text><polyline class="edge-on" points="506,56 560,70"/><polygon class="arrow-on" points="560.0,70.0 551.3,71.9 553.3,64.1"/><polyline class="edge-on" points="434,92 560,140"/><polygon class="arrow-on" points="560.0,140.0 551.1,140.9 553.9,133.4"/><rect class="box-1" x="522" y="220" width="194" height="56" rx="6"/><text class="t-note" x="619" y="244" text-anchor="middle">output</text><text class="t-muted" x="619" y="261" text-anchor="middle">0.7311 E0(x) + 0.2689 E1(x)</text><polyline class="edge" points="700,75 710,75 710,220"/><polygon class="arrow" points="710.0,220.0 706.0,212.0 714.0,212.0"/><polyline class="edge" points="630,170 630,220"/><polygon class="arrow" points="630.0,220.0 626.0,212.0 634.0,212.0"/><text class="t-muted" x="20" y="340">The six other experts do no work for this token. The two chosen scores, 0.4528 and 0.1666, are rescaled to add up to 1.</text><text class="t-muted" x="20" y="358">Each token picks its own pair, so all experts are busy over a batch, but each token pays for only two.</text></svg><figcaption>One token of our script's router. Mixtral uses exactly this shape: 8 experts, top 2. DeepSeek-V3 picks 8 of 256 routed experts, plus 1 shared expert that every token uses.</figcaption></figure>

Noam Shazeer and colleagues at Google made this work for language models in 2017, with a paper titled "Outrageously Large Neural Networks". It used thousands of experts between LSTM layers. The idea moved into Transformers with Google's GShard in 2020, then into the Switch Transformer.

### Keeping the experts busy

A router left alone tends to collapse. A few experts get slightly better early, so the router sends them more tokens, so they get better still. The others starve. And in practice each expert lives on a different GPU, so an overloaded expert makes the others wait.

> [!PAPER] Fedus, Zoph and Shazeer (2021), Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity · Figure 2 · page 5
> [![Figure 2 of the Switch Transformer paper: the dense feed-forward layer of a Transformer block is replaced by a switching layer of four experts; two tokens, More and Parameters, each go through a router that picks one expert, FFN 2 with probability 0.65 and FFN 1 with probability 0.8](/img/writings/history/h-switch-fig.png)](/img/writings/history/h-switch-fig.png)
>
> **What it says:** "We replace the dense feed forward network (FFN) layer present in the Transformer with a sparse Switch FFN layer". The router "independently routes each token", and the layer returns the chosen expert's output "multiplied by the router gate value".
>
> **What it means:**
>
> The Switch Transformer went to the extreme: one expert per token, $$k = 1$$. That halves the routing work and the communication, and it worked. The paper reports pre-training up to 7 times faster than a dense T5 model with the same compute per token. It also trained models with over a trillion parameters. The two tokens here, "More" and "Parameters", go to different experts in the same layer.

To keep the load even, Switch adds a small **load-balancing loss** to the training loss:

$$
\mathcal{L}_{\text{aux}} = \alpha \, N \sum_{i=1}^{N} f_i \, P_i
$$

$$N$$ is the number of experts. $$f_i$$ is the share of tokens in the batch that were actually sent to expert $$i$$. $$P_i$$ is the average router probability for expert $$i$$ over the batch. $$\alpha$$ is a small weight; the paper uses $$10^{-2}$$. The sum is smallest when both shares are spread evenly.

With 8 experts and perfect balance, every $$f_i$$ and $$P_i$$ is $$1/8$$, so $$N \sum f_i P_i = 8 \times 8 \times \frac{1}{64} = 1$$. If every token goes to expert 0 with probability 1, the sum is 8, the worst case. Our script also ran a random router that favours two experts, on 4,096 tokens. It sent 46.8% and 24.8% of tokens to them, and scored 1.7124, so the loss pushes it back toward 1.

> [!DEFINITION] Load balancing
> Keeping the number of tokens sent to each expert about equal. It matters for quality, because starved experts never learn, and for speed, because the busiest expert sets the pace of the whole layer.

### Mixtral: an open MoE

In December 2023, Mistral AI released Mixtral 8x7B with open weights, and its paper followed in January 2024.

> [!PAPER] Jiang et al. (2024), Mixtral of Experts · Figure 1 · page 2
> [![Figure 1 of the Mixtral paper: inputs go to a router, which sends them to two of several expert boxes; the experts' outputs are summed with the gating weights](/img/writings/history/h-mixtral-fig.png)](/img/writings/history/h-mixtral-fig.png)
>
> **What it says:** "Each input vector is assigned to 2 of the 8 experts by a router. The layer's output is the weighted sum of the outputs of the two selected experts." In Mixtral, "an expert is a standard feedforward block as in a vanilla transformer architecture."
>
> **What it means:**
>
> This is the equation above, drawn. Each token "has access to 47B parameters, but only uses 13B active parameters", the paper says: 27.7% of the model. The attention layers are shared by all tokens; only the feed-forward part is split. The paper reports that Mixtral "outperforms or matches Llama 2 70B" on its benchmarks, with far less compute per token.

### DeepSeek-V3: many small experts and one shared one

DeepSeek pushed the design in a different direction: many small experts instead of a few large ones, plus experts that every token uses.

> [!DEFINITION] Shared expert
> An expert that processes every token, alongside the routed ones. It learns common knowledge that every token needs, so the routed experts are free to specialise.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Figure 2 · page 7
> [![Figure 2 of the DeepSeek-V3 report: a Transformer block of RMSNorm, attention, RMSNorm and feed-forward, repeated L times; the feed-forward is DeepSeekMoE, with shared experts and routed experts chosen by a top-K router; the attention is multi-head latent attention, with a cached latent for keys and values and RoPE applied to separate parts](/img/writings/history/h-dsv3-fig.png)](/img/writings/history/h-dsv3-fig.png)
>
> **What it says:** an "illustration of the basic architecture of DeepSeek-V3. Following DeepSeek-V2, we adopt MLA and DeepSeekMoE for efficient inference and economical training."
>
> **What it means:**
>
> The left column is the 2026 block from our figure: RMSNorm, attention, RMSNorm, feed-forward, with residual paths. The top box is the MoE layer. Green shared experts sit on the left, blue routed experts on the right, and a router picks the top $$K_r$$. The bottom box is MLA: only the latent $$c^{KV}$$ and a small RoPE key, the striped boxes, are cached. Almost every idea in this part is in this one figure.

DeepSeek-V3 has 61 layers. All but the first three use MoE, with 1 shared expert and 256 routed experts, of which each token uses 8. So each token passes through 9 of 257 expert networks. The model has 671 billion parameters, and 37 billion, 5.5%, are active for each token.

It also changed how balance is kept. A large auxiliary loss can hurt quality. So DeepSeek-V3 gives each expert a bias, which is added to its score only when the top 8 are chosen. After each step, the bias of an overloaded expert goes down and that of an idle one goes up. The report calls this "auxiliary-loss-free" balancing, and keeps only a tiny loss of the Switch kind as a backstop.

### What the closed labs say

Google has said the most. The Gemini 1.5 report of 2024 calls Gemini 1.5 Pro "a sparse mixture-of-expert (MoE) Transformer-based model". The 2025 report on Gemini 2.5 says the same of the whole family, and adds what it is for.

> [!PAPER] Gemini Team, Google (2025), Gemini 2.5: Pushing the Frontier with Advanced Reasoning, Multimodality, Long Context, and Next Generation Agentic Capabilities · §2.1 · page 2
> [![Section 2.1 of the Gemini 2.5 report, with sparse mixture-of-experts, native multimodal support for text, vision, and audio inputs, decouple total model capacity from computation and serving cost per token, and training instabilities highlighted](/img/writings/history/h-gemini25-arch.png)](/img/writings/history/h-gemini25-arch.png)
>
> **What it says:** "The Gemini 2.5 models are sparse mixture-of-experts (MoE) transformers with native multimodal support for text, vision, and audio inputs." MoE lets them "decouple total model capacity from computation and serving cost per token". Large MoE models "are known to suffer from training instabilities", and the series made "considerable progress in enhancing large-scale training stability".
>
> **What it means:**
>
> This is nearly all Google publishes about the inside of Gemini. It confirms the two big design choices of this part, sparse experts and native multimodality. It gives no number of experts, no size, no layer count and no data size. "Training instabilities" is a polite phrase for loss spikes and runs that diverge; the section on training returns to them.

The [Gemini 3 Pro model card](https://storage.googleapis.com/deepmind-media/Model-Cards/Gemini-3-Pro-Model-Card.pdf) says the same: a "sparse mixture-of-experts (MoE)" Transformer "with native multimodal support for text, vision, and audio inputs", trained on Google's TPUs. Meta's [Llama 4](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) models of April 2025 are MoE too. Llama 4 Scout has 17 billion active parameters, 16 experts and 109 billion in total; Maverick has 17 billion active, 128 experts and 400 billion in total. Kimi K2 has about a trillion parameters with 32 billion active.

OpenAI and Anthropic do not publish the architecture of their frontier models. Press reports about their sizes and designs exist, but none comes from the labs, so this part does not repeat them.

## Native multimodality

Part 1 showed the Vision Transformer of 2020, which treats an image as a sequence of patches. That one idea is what lets a language model see.

> [!PAPER] Dosovitskiy et al. (2020), An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale · Figure 1 · page 3
> [![Figure 1 of the ViT paper: an image is split into nine patches, each flattened and linearly projected, given a position embedding and fed with an extra class token into a Transformer encoder of norm, multi-head attention and MLP blocks, with an MLP head that outputs the class](/img/writings/history/h-vit-fig.png)](/img/writings/history/h-vit-fig.png)
>
> **What it says:** "We split an image into fixed-size patches, linearly embed each of them, add position embeddings, and feed the resulting sequence of vectors to a standard Transformer encoder."
>
> **What it means:**
>
> Each patch becomes one vector, just like a word's embedding. After that line, the Transformer cannot tell an image from a sentence. A 224 × 224 image with 16 × 16 patches gives $$(224/16)^2 = 196$$ tokens, each made from $$16 \times 16 \times 3 = 768$$ pixel values. A 1024 × 1024 image gives 4,096 tokens, so modern models compress or pool the patch tokens.

> [!DEFINITION] Native multimodality (early fusion)
> Training one model on text, images, audio and video together from the start of pre-training, with all of them turned into tokens in one sequence. The alternative, **late fusion**, connects a separate image encoder to a finished language model with a small adapter.

Early open models used late fusion: a frozen image encoder, a small adapter and a language model, trained together only at the end. Google's first Gemini report, in December 2023, described its models as multimodal from the start of training. Meta's Chameleon paper of May 2024 described "early-fusion token-based mixed-modal models", and Llama 4 uses "early fusion to seamlessly integrate text and vision tokens".

<figure class="fig"><svg viewBox="0 0 720 346" role="img" aria-label="Images are cut into patches, audio into frames and text into BPE tokens. Each becomes a vector, and the vectors are interleaved into one sequence that a single Transformer reads."><text class="t-title" x="10" y="22">One sequence, many kinds of token</text><rect class="box-2" x="20" y="50" width="170" height="70" rx="6"/><text class="t-note" x="105" y="81" text-anchor="middle">image 224 × 224</text><text class="t-muted" x="105" y="98" text-anchor="middle">cut into 16 × 16 patches</text><rect class="box-3" x="20" y="140" width="170" height="56" rx="6"/><text class="t-note" x="105" y="164" text-anchor="middle">audio</text><text class="t-muted" x="105" y="181" text-anchor="middle">short frames → vectors</text><rect class="box-1" x="20" y="216" width="170" height="56" rx="6"/><text class="t-note" x="105" y="240" text-anchor="middle">text</text><text class="t-muted" x="105" y="257" text-anchor="middle">BPE tokens → vectors</text><rect class="box" x="230" y="50" width="150" height="70" rx="6"/><text class="t-note" x="305" y="81" text-anchor="middle">196 patch tokens</text><text class="t-muted" x="305" y="98" text-anchor="middle">linear projection</text><polyline class="edge" points="190,85 230,85"/><polygon class="arrow" points="230.0,85.0 222.0,89.0 222.0,81.0"/><rect class="box" x="230" y="140" width="150" height="56" rx="6"/><text class="t-note" x="305" y="164" text-anchor="middle">audio tokens</text><text class="t-muted" x="305" y="181" text-anchor="middle">an audio encoder</text><polyline class="edge" points="190,168 230,168"/><polygon class="arrow" points="230.0,168.0 222.0,172.0 222.0,164.0"/><rect class="box" x="230" y="216" width="150" height="56" rx="6"/><text class="t-note" x="305" y="240" text-anchor="middle">text tokens</text><text class="t-muted" x="305" y="257" text-anchor="middle">embedding table</text><polyline class="edge" points="190,244 230,244"/><polygon class="arrow" points="230.0,244.0 222.0,248.0 222.0,240.0"/><rect class="box-1" x="420" y="150" width="17" height="36" rx="3"/><rect class="box-1" x="440" y="150" width="17" height="36" rx="3"/><rect class="box-2" x="460" y="150" width="17" height="36" rx="3"/><rect class="box-2" x="480" y="150" width="17" height="36" rx="3"/><rect class="box-2" x="500" y="150" width="17" height="36" rx="3"/><rect class="box-2" x="520" y="150" width="17" height="36" rx="3"/><rect class="box-2" x="540" y="150" width="17" height="36" rx="3"/><rect class="box-1" x="560" y="150" width="17" height="36" rx="3"/><rect class="box-1" x="580" y="150" width="17" height="36" rx="3"/><rect class="box-3" x="600" y="150" width="17" height="36" rx="3"/><rect class="box-3" x="620" y="150" width="17" height="36" rx="3"/><rect class="box-3" x="640" y="150" width="17" height="36" rx="3"/><rect class="box-1" x="660" y="150" width="17" height="36" rx="3"/><rect class="box-1" x="680" y="150" width="17" height="36" rx="3"/><polyline class="edge" points="380,85 440,146"/><polygon class="arrow" points="440.0,146.0 431.5,143.1 437.2,137.5"/><polyline class="edge" points="380,168 420,168"/><polygon class="arrow" points="420.0,168.0 412.0,172.0 412.0,164.0"/><polyline class="edge" points="380,244 440,190"/><polygon class="arrow" points="440.0,190.0 436.7,198.3 431.4,192.4"/><rect class="box-on" x="420" y="230" width="280" height="50" rx="6"/><text class="t-note" x="560" y="251" text-anchor="middle">one Transformer reads it all</text><text class="t-muted" x="560" y="268" text-anchor="middle">same attention, same layers</text><polyline class="edge" points="560,186 560,230"/><polygon class="arrow" points="560.0,230.0 556.0,222.0 564.0,222.0"/><text class="t-note" x="470" y="136">the token stream</text><text class="t-muted" x="20" y="312">Gemini and Llama 4 are described as natively multimodal: they meet images (and, for Gemini, sound) in pre-training,</text><text class="t-muted" x="20" y="330">not only through an adapter bolted on afterwards. The exact encoders of the closed models are not public.</text></svg><figcaption>A 224 × 224 image with 16 × 16 patches gives 196 tokens, the setting of the 2020 Vision Transformer. Modern models use bigger images and compress the patch tokens.</figcaption></figure>

Audio and video follow the same pattern. Sound is cut into short frames, an encoder turns them into vectors, and video is a stream of image patches plus its sound track. The Gemini 1.5 report tested retrieval on up to 9.7 million audio tokens. The exact encoders inside Gemini and other closed models are not public.

## Training at scale

### The machines

Training a frontier model needs thousands of accelerators working as one. Two families dominate: NVIDIA's GPUs and Google's TPUs.

> [!DEFINITION] Accelerator
> A chip built for the matrix arithmetic of neural networks, such as a GPU or Google's TPU (tensor processing unit). Its speed is quoted as peak FLOPs per second at a given number format.

An NVIDIA H100 does 989.4 trillion 16-bit FLOPs per second at its peak, on dense matrices. It carries 80 GB of HBM3 memory at 3.35 TB/s. Meta trained Llama 3 405B on up to 16,384 of them.

Google builds its own chips and joins them into **pods**. Gemini 1.5 was trained on several 4,096-chip pods of TPU v4, the report says. A TPU v5p chip does 459 trillion 16-bit FLOPs per second, and one pod holds 8,960 chips. Ironwood, announced in April 2025, puts 9,216 chips in a pod, with 4,614 trillion 8-bit FLOPs per second for each chip.

<figure><img src="/img/writings/history/img-tpu3.jpg" alt="A blue circuit board with four large chips under copper cold plates, joined by grey liquid-cooling tubes" loading="lazy" /><figcaption>A TPU v3 board: four chips, each under a copper plate, cooled by liquid. Photo: Zinskauf, CC BY-SA 4.0, via Wikimedia Commons.</figcaption></figure>

<figure><img src="/img/writings/history/img-tpu4.jpg" alt="A green circuit board with four TPU v4 chips, joined by coloured liquid-cooling tubes" loading="lazy" /><figcaption>A TPU v4 board, the chip generation that trained Gemini 1.5. Image: Norman P. Jouppi and colleagues, from their 2023 paper on TPU v4, CC BY 4.0, via Wikimedia Commons.</figcaption></figure>

> [!DEFINITION] Model FLOPs utilisation (MFU)
> The share of the hardware's peak speed that goes into the model's useful arithmetic. Time spent waiting for data, talking between chips or recovering from failures lowers it.

No run reaches the peak. Llama 3 reports an MFU of 38% to 43% in 16-bit numbers. With those figures, our script estimates that $$3.8 \times 10^{25}$$ FLOPs on 16,384 H100s would take 63 to 71 days of pure training. That is our estimate, not a number Meta published.

### Splitting the work

A 405B model does not fit on one GPU, and one GPU would take centuries anyway. So the work is split in several ways at once.

> [!DEFINITION] Data, tensor and pipeline parallelism
> **Data parallelism** gives each GPU a full copy of the model and a different slice of each batch, then averages the gradients. **Tensor parallelism** splits each weight matrix across GPUs. **Pipeline parallelism** gives each GPU a block of consecutive layers, and passes activations along the chain.

<figure class="fig"><svg viewBox="0 0 720 362" role="img" aria-label="Three panels. Data parallel: three GPUs each hold the full model and see a different slice of the batch, then average gradients. Tensor parallel: one weight matrix is cut into four column slices on four GPUs. Pipeline parallel: four GPUs hold four stages of layers, and four micro-batches flow through them in a staircase."><text class="t-title" x="10" y="22">Three ways to split one training run across many GPUs</text><text class="t-note" x="120" y="50" text-anchor="middle">Data parallel</text><text class="t-muted" x="120" y="66" text-anchor="middle">same model, different data</text><rect class="box-1" x="20" y="82" width="200" height="46" rx="6"/><text class="t-note" x="120" y="101" text-anchor="middle">GPU 0: full model</text><text class="t-muted" x="120" y="118" text-anchor="middle">batch slice 0</text><rect class="box-1" x="20" y="144" width="200" height="46" rx="6"/><text class="t-note" x="120" y="163" text-anchor="middle">GPU 1: full model</text><text class="t-muted" x="120" y="180" text-anchor="middle">batch slice 1</text><rect class="box-1" x="20" y="206" width="200" height="46" rx="6"/><text class="t-note" x="120" y="225" text-anchor="middle">GPU 2: full model</text><text class="t-muted" x="120" y="242" text-anchor="middle">batch slice 2</text><rect class="box-on" x="14" y="274" width="212" height="40" rx="6"/><text class="t-note" x="120" y="298.5" text-anchor="middle">all-reduce: average gradients</text><text class="t-note" x="360" y="50" text-anchor="middle">Tensor parallel</text><text class="t-muted" x="360" y="66" text-anchor="middle">split each weight matrix</text><rect class="box-2" x="260" y="90" width="46" height="120" rx="4"/><text class="t-muted" x="283" y="154" text-anchor="middle">GPU 0</text><rect class="box-2" x="310" y="90" width="46" height="120" rx="4"/><text class="t-muted" x="333" y="154" text-anchor="middle">GPU 1</text><rect class="box-2" x="360" y="90" width="46" height="120" rx="4"/><text class="t-muted" x="383" y="154" text-anchor="middle">GPU 2</text><rect class="box-2" x="410" y="90" width="46" height="120" rx="4"/><text class="t-muted" x="433" y="154" text-anchor="middle">GPU 3</text><text class="t-muted" x="360" y="230" text-anchor="middle">one 4096 × 14336 matrix</text><text class="t-muted" x="360" y="248" text-anchor="middle">→ four 4096 × 3584 slices</text><rect class="box-on" x="270" y="274" width="180" height="40" rx="6"/><text class="t-note" x="360" y="298.5" text-anchor="middle">exchange after each layer</text><text class="t-note" x="600" y="50" text-anchor="middle">Pipeline parallel</text><text class="t-muted" x="600" y="66" text-anchor="middle">split the layers into stages</text><text class="t-muted" x="500" y="104">GPU 0</text><rect class="box-3" x="538" y="88" width="21" height="24" rx="3"/><text class="t-tick" x="548.5" y="105" text-anchor="middle">1</text><rect class="box-3" x="562" y="88" width="21" height="24" rx="3"/><text class="t-tick" x="572.5" y="105" text-anchor="middle">2</text><rect class="box-3" x="586" y="88" width="21" height="24" rx="3"/><text class="t-tick" x="596.5" y="105" text-anchor="middle">3</text><rect class="box-3" x="610" y="88" width="21" height="24" rx="3"/><text class="t-tick" x="620.5" y="105" text-anchor="middle">4</text><text class="t-muted" x="500" y="150">GPU 1</text><rect class="box-3" x="562" y="134" width="21" height="24" rx="3"/><text class="t-tick" x="572.5" y="151" text-anchor="middle">1</text><rect class="box-3" x="586" y="134" width="21" height="24" rx="3"/><text class="t-tick" x="596.5" y="151" text-anchor="middle">2</text><rect class="box-3" x="610" y="134" width="21" height="24" rx="3"/><text class="t-tick" x="620.5" y="151" text-anchor="middle">3</text><rect class="box-3" x="634" y="134" width="21" height="24" rx="3"/><text class="t-tick" x="644.5" y="151" text-anchor="middle">4</text><text class="t-muted" x="500" y="196">GPU 2</text><rect class="box-3" x="586" y="180" width="21" height="24" rx="3"/><text class="t-tick" x="596.5" y="197" text-anchor="middle">1</text><rect class="box-3" x="610" y="180" width="21" height="24" rx="3"/><text class="t-tick" x="620.5" y="197" text-anchor="middle">2</text><rect class="box-3" x="634" y="180" width="21" height="24" rx="3"/><text class="t-tick" x="644.5" y="197" text-anchor="middle">3</text><rect class="box-3" x="658" y="180" width="21" height="24" rx="3"/><text class="t-tick" x="668.5" y="197" text-anchor="middle">4</text><text class="t-muted" x="500" y="242">GPU 3</text><rect class="box-3" x="610" y="226" width="21" height="24" rx="3"/><text class="t-tick" x="620.5" y="243" text-anchor="middle">1</text><rect class="box-3" x="634" y="226" width="21" height="24" rx="3"/><text class="t-tick" x="644.5" y="243" text-anchor="middle">2</text><rect class="box-3" x="658" y="226" width="21" height="24" rx="3"/><text class="t-tick" x="668.5" y="243" text-anchor="middle">3</text><rect class="box-3" x="682" y="226" width="21" height="24" rx="3"/><text class="t-tick" x="692.5" y="243" text-anchor="middle">4</text><text class="t-muted" x="600" y="280" text-anchor="middle">blank slots = idle time,</text><text class="t-muted" x="600" y="298" text-anchor="middle">the "bubble": 3/7 here</text><line class="edge-dim" x1="240" y1="40" x2="240" y2="320"/><line class="edge-dim" x1="480" y1="40" x2="480" y2="320"/><text class="t-muted" x="10" y="346">Large runs use all three at once, plus expert parallelism for MoE layers and ZeRO / FSDP to shard optimizer states.</text></svg><figcaption>With 4 stages and 4 micro-batches the pipeline is idle 3/7 of the time; with 32 micro-batches only 3/35 (8.6%). Numbers from our script.</figcaption></figure>

Tensor parallelism comes from NVIDIA's Megatron-LM paper of 2019. It needs fast links, because GPUs must swap partial results inside every layer. So it usually stays within one server, where 8 GPUs are joined by NVLink.

> [!PAPER] Shoeybi et al. (2019), Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism · Figure 3 · page 4
> [![Figure 3 of Megatron-LM: (a) the MLP block with the first weight matrix A split by columns into A1 and A2 and the second matrix B split by rows, with operators f and g at the ends; (b) self-attention with the heads split across two GPUs in the same way](/img/writings/history/h-megatron-fig.png)](/img/writings/history/h-megatron-fig.png)
>
> **What it says:** "Blocks of Transformer with Model Parallelism. f and g are conjugate. f is an identity operator in the forward pass and all reduce in the backward pass while g is an all reduce in the forward pass and identity in the backward pass."
>
> **What it means:**
>
> Split the first matrix by columns and the second by rows. Then each GPU can run its half of the feed-forward block with no talking in between. Only at the end, at $$g$$, do the two halves add up their results. Attention splits the same way, by heads. The paper trained an 8.3B model on 512 GPUs with 76% scaling efficiency, which was huge for 2019.

Take Llama 3 8B's feed-forward matrix of 4,096 × 14,336. Four GPUs would each hold a 4,096 × 3,584 slice; eight would each hold 4,096 × 1,792. Pipeline parallelism has a different cost: at the start and end of each batch, some stages sit idle while they wait. Cut the batch into $$m$$ **micro-batches** that flow through $$p$$ stages, and the idle share is

$$
\text{bubble} = \frac{p - 1}{m + p - 1}
$$

With 4 stages and 4 micro-batches, $$3/7 = 0.429$$ of the time is wasted. With 32 micro-batches it falls to $$3/35 = 0.086$$. With 16 stages and 64 micro-batches it is 0.190. More micro-batches shrink the bubble but need more memory. MoE models add a fourth kind, **expert parallelism**: different experts on different GPUs, with tokens sent to wherever their experts live. DeepSeek-V3 spread each layer's routed experts over 64 GPUs.

The [inference series](llm-inference-7-beyond-one-gpu.md) shows the same splits used for serving.

### Memory: sharding the optimizer

Training needs far more memory than the weights alone. In mixed-precision training with the Adam optimizer, each parameter needs 2 bytes for its 16-bit weight, and 2 for its 16-bit gradient. It also needs 12 bytes of 32-bit optimizer state: a master copy of the weight and Adam's two running averages. That is 16 bytes per parameter.

> [!DEFINITION] ZeRO and FSDP
> Methods that shard the optimizer states, gradients and even the weights across the GPUs of a data-parallel group, instead of copying them to every GPU. Each GPU keeps only its share and fetches the rest when needed. ZeRO is Microsoft's name; FSDP (fully sharded data parallel) is PyTorch's version.

For a 70B model, 16 bytes per parameter is 1,120 GB: the full memory of 14 GPUs with 80 GB each, before any activations. For 405B it is 6,480 GB. The ZeRO paper of 2019, from Samyam Rajbhandari and colleagues at Microsoft, gave the fix. Its example is a 7.5B model, which needs 120 GB of states. Sharded over 64 GPUs, each holds only 1.875 GB.

### Fewer bits per number

> [!DEFINITION] Mixed precision, bf16 and FP8
> Doing most arithmetic in short number formats while keeping a master copy of the weights in 32-bit. The **bf16** (brain floating point) format has 16 bits, with the range of 32-bit numbers but less precision. **FP8** uses only 8 bits, and needs careful scaling to avoid overflow.

Halving the bits halves the memory traffic and roughly doubles the speed of the matrix units. In 2017, Paulius Micikevicius and colleagues at NVIDIA and Baidu showed that 16-bit training works. You keep a 32-bit master copy and scale the loss. The bf16 format, with its wide range, made the scaling unnecessary and became the default.

DeepSeek-V3 was the first very large open model trained mostly in FP8. Its report says it was the first to "validate its effectiveness on an extremely large-scale model". The trick is to scale small blocks of numbers separately, so one large value does not ruin the precision of its neighbours. The most sensitive parts, such as the embedding, the output head and the normalisation, stay in higher precision.

### The optimizer and the schedule

Almost every large model is trained with **AdamW**, from Ilya Loshchilov and Frank Hutter (2017). It is like a hiker who keeps a running average of the slope and of how bumpy the slope has been. Steps are large where the ground is steady and small where it is noisy.

> [!DEFINITION] Learning rate
> The size of each training step: how far the weights move in the direction the gradient suggests. Too large and training becomes unstable; too small and it is slow.

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad w_t = w_{t-1} - \eta \left( \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon} + \lambda w_{t-1} \right)
$$

$$g_t$$ is the gradient at step $$t$$. $$m_t$$ is the running average of gradients and $$v_t$$ of squared gradients, with decay rates $$\beta_1$$ and $$\beta_2$$. $$\hat m_t$$ and $$\hat v_t$$ are the same averages, corrected for starting at zero: divide by $$1 - \beta_1^t$$ and $$1 - \beta_2^t$$. $$\eta$$ (eta) is the learning rate and $$\lambda$$ (lambda) the **weight decay**, which pulls every weight a little toward 0.

Take one weight $$w = 1.0$$ with gradient $$g = 0.5$$, $$\eta = 0.001$$, $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$ and $$\lambda = 0.1$$. Then $$m = 0.05$$, $$v = 0.0125$$, $$\hat m = 0.5$$ and $$\hat v = 0.25$$. The Adam step is $$0.001 \times 0.5 / 0.5 = 0.001$$, the decay adds 0.0001, and the new weight is 0.9989. With $$g = 50$$, the step is still 0.001. Adam's first step has the size of the learning rate whatever the gradient's size, which is part of why it is stable.

DeepSeek-V3 used $$\beta_1 = 0.9$$, $$\beta_2 = 0.95$$ and weight decay 0.1, the same values as our example. The learning rate also changes during training. It starts at zero, rises in a straight line during a **warm-up**, and then falls.

$$
\eta(t) = \eta_{\min} + \tfrac{1}{2} (\eta_{\max} - \eta_{\min}) \left( 1 + \cos \frac{\pi (t - t_w)}{T - t_w} \right) \quad \text{for } t \ge t_w
$$

$$t$$ is the step, $$t_w$$ the length of the warm-up and $$T$$ the last step. $$\eta_{\max}$$ is the peak rate and $$\eta_{\min}$$ the final one. The cosine falls from 1 to $$-1$$, so the rate glides from the peak to the floor, slowly at first, then faster, then slowly again.

Llama 3 405B used a peak of $$8 \times 10^{-5}$$, 8,000 warm-up steps, and a cosine to $$8 \times 10^{-7}$$ over 1,200,000 steps. Our script assumes that the 1.2 million steps include the warm-up. Then the rate is $$4 \times 10^{-5}$$ at step 4,000 and $$4.04 \times 10^{-5}$$ half-way through, at step 604,000.

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="A curve that rises in a straight line to 8 times 10 to the minus 5 at step 8,000, then falls along a cosine to 8 times 10 to the minus 7 at step 1.2 million."><text class="t-title" x="10" y="22">Learning rate: linear warm-up, then cosine decay (Llama 3 405B settings)</text><line class="grid" x1="80" y1="250.0" x2="690" y2="250.0"/><text class="t-tick" x="72" y="254" text-anchor="end">0</text><line class="grid" x1="80" y1="201.0" x2="690" y2="201.0"/><text class="t-tick" x="72" y="204.952" text-anchor="end">2e-5</text><line class="grid" x1="80" y1="151.9" x2="690" y2="151.9"/><text class="t-tick" x="72" y="155.905" text-anchor="end">4e-5</text><line class="grid" x1="80" y1="102.9" x2="690" y2="102.9"/><text class="t-tick" x="72" y="106.857" text-anchor="end">6e-5</text><line class="grid" x1="80" y1="53.8" x2="690" y2="53.8"/><text class="t-tick" x="72" y="57.8095" text-anchor="end">8e-5</text><text class="t-tick" x="80" y="268" text-anchor="middle">0</text><text class="t-tick" x="232.5" y="268" text-anchor="middle">300K</text><text class="t-tick" x="385" y="268" text-anchor="middle">600K</text><text class="t-tick" x="537.5" y="268" text-anchor="middle">900K</text><text class="t-tick" x="690" y="268" text-anchor="middle">1,200K</text><polyline class="l1" points="80.0,250.0 80.0,250.0 80.3,237.7 80.5,225.5 80.8,213.2 81.0,201.0 81.3,188.7 81.5,176.4 81.8,164.2 82.0,151.9 82.3,139.6 82.5,127.4 82.8,115.1 83.0,102.9 83.3,90.6 83.6,78.3 83.8,66.1 84.1,53.8 85.1,53.8 90.2,53.9 95.2,54.0 100.3,54.2 105.4,54.4 110.5,54.7 115.6,55.1 120.7,55.6 125.8,56.1 130.8,56.7 135.9,57.3 141.0,58.0 146.1,58.8 151.2,59.6 156.2,60.5 161.3,61.5 166.4,62.5 171.5,63.6 176.6,64.8 181.7,66.0 186.8,67.3 191.8,68.6 196.9,70.0 202.0,71.4 207.1,72.9 212.2,74.5 217.2,76.1 222.3,77.7 227.4,79.4 232.5,81.2 237.6,83.0 242.7,84.8 247.8,86.7 252.8,88.7 257.9,90.7 263.0,92.7 268.1,94.8 273.2,96.9 278.2,99.0 283.3,101.2 288.4,103.4 293.5,105.7 298.6,107.9 303.7,110.2 308.8,112.6 313.8,114.9 318.9,117.3 324.0,119.7 329.1,122.2 334.2,124.6 339.2,127.1 344.3,129.6 349.4,132.1 354.5,134.6 359.6,137.1 364.7,139.7 369.8,142.2 374.8,144.8 379.9,147.3 385.0,149.9 390.1,152.5 395.2,155.0 400.2,157.6 405.3,160.1 410.4,162.7 415.5,165.2 420.6,167.7 425.7,170.2 430.8,172.7 435.8,175.2 440.9,177.7 446.0,180.2 451.1,182.6 456.2,185.0 461.2,187.4 466.3,189.7 471.4,192.1 476.5,194.4 481.6,196.6 486.7,198.9 491.8,201.1 496.8,203.3 501.9,205.4 507.0,207.5 512.1,209.6 517.2,211.6 522.2,213.6 527.3,215.5 532.4,217.4 537.5,219.2 542.6,221.0 547.7,222.8 552.8,224.5 557.8,226.1 562.9,227.7 568.0,229.2 573.1,230.7 578.2,232.2 583.2,233.5 588.3,234.9 593.4,236.1 598.5,237.3 603.6,238.5 608.7,239.5 613.8,240.5 618.8,241.5 623.9,242.4 629.0,243.2 634.1,244.0 639.2,244.7 644.2,245.3 649.3,245.9 654.4,246.4 659.5,246.8 664.6,247.2 669.7,247.5 674.8,247.7 679.8,247.9 684.9,248.0 690.0,248.0"/><circle class="s1" cx="84.1" cy="53.8" r="4"/><text class="t-s1" x="94.0667" y="43.8095" text-anchor="start">peak 8e-5 after 8,000 warm-up steps</text><circle class="s1" cx="387.0" cy="150.9" r="4"/><text class="t-s1" x="397.033" y="140.924" text-anchor="start">half-way: 4.04e-5</text><circle class="s1" cx="690.0" cy="248.0" r="4"/><text class="t-s1" x="680" y="238.038" text-anchor="end">end: 8e-7</text><line class="axis" x1="80" y1="250" x2="690" y2="250"/><text class="t-muted" x="690" y="288" text-anchor="end">training step</text></svg><figcaption>The warm-up is too short to see at this scale: 8,000 of 1,200,000 steps. Llama 3 then annealed the rate to 0 over the last 40 million tokens. Curve from our script.</figcaption></figure>

Other shapes are common. DeepSeek-V3 held its peak rate flat until 10 trillion tokens and decayed it along a cosine over the next 4.3 trillion. It ended with two short flat steps. Chinchilla's lesson about schedules still holds: the decay must match the planned length of the run.

### When training breaks

> [!DEFINITION] Loss spike
> A sudden jump in the training loss, sometimes followed by recovery, sometimes by **divergence**, where the loss never comes back. Spikes become more common as models grow, and their causes are often unclear.

Large runs are fragile. Google's PaLM paper of 2022 is unusually frank about it. For its 540B model, the team "observed spikes in the loss roughly 20 times during training", although gradient clipping was on. Their fix was to restart "from a checkpoint roughly 100 steps before the spike started" and skip "roughly 200–500 data batches". The same batches, given to an earlier checkpoint, caused no spike. So the cause was the data and the model's state together, not bad data alone.

Since then, labs have added tricks such as extra normalisation inside attention and limits on the size of attention scores. DeepSeek-V3 reports that "we did not experience any irrecoverable loss spikes or perform any rollbacks". Kimi K2 used a new optimizer, MuonClip, and reports pre-training on 15.5 trillion tokens "with zero loss spike".

Hardware fails too. In one 54-day stretch of Llama 3 training, there were 466 interruptions. Of these, 419 were unexpected, and about 78% of those were traced to confirmed or suspected hardware problems. Google's Gemini 2.5 report describes the same problem at TPU scale.

> [!PAPER] Gemini Team, Google (2025), Gemini 2.5 · §2.3 · page 4
> [![Section 2.3 of the Gemini 2.5 report, with multiple 8960-chip pods of Google's TPUv5p accelerators and interruptions from hardware failures multiple times per hour highlighted](/img/writings/history/h-gemini25-tpu.png)](/img/writings/history/h-gemini25-tpu.png)
>
> **What it says:** Gemini 2.5 is "the first to be trained on TPUv5p architecture", with "synchronous data-parallel training" over "multiple 8960-chip pods of Google's TPUv5p accelerators, distributed across multiple datacenters". "At the scale of this training run we see interruptions from hardware failures multiple times per hour."
>
> **What it means:**
>
> Several pods, in several buildings, act as one machine, and something breaks several times an hour. Google's answer is to keep going with fewer chips while a failed group is swapped out, at about 97% speed. The report adds that 93.4% of the run's time went into TPU work. At this scale, the engineering of failure is as important as the design of the model.

### What it costs

The most quoted cost in recent years comes from DeepSeek-V3, and it is often quoted wrongly.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · Table 1 and §1 · page 5
> [![Table 1 of the DeepSeek-V3 report: training costs of 2664K H800 GPU hours for pre-training, 119K for context extension and 5K for post-training, 2788K in total, or 5.576 million US dollars at 2 dollars per GPU hour; with the passage that the costs exclude prior research and ablation experiments highlighted](/img/writings/history/h-dsv3-cost.png)](/img/writings/history/h-dsv3-cost.png)
>
> **What it says:** pre-training took 2,664K H800 GPU hours, context extension 119K and post-training 5K, for 2,788K in total, or "$5.576M" at "$2 per GPU hour". Each trillion tokens took "180K H800 GPU hours, i.e., 3.7 days on our cluster with 2048 H800 GPUs". The costs cover "only the official training", "excluding the costs associated with prior research and ablation experiments".
>
> **What it means:**
>
> The table is the rent for the final run, at an assumed price. It is not the cost of the company, its chips, its staff or the many smaller runs that found the recipe. The numbers check out: 180K hours times 14.8 trillion tokens is 2,664K. Spread over 2,048 GPUs, 2,664K hours is about 54 days. It was still a striking result for a model of its quality.

Our script adds one more check. With $$C = 6 \times 37 \times 10^9 \times 14.8 \times 10^{12} = 3.29 \times 10^{24}$$ FLOPs over 2,664K GPU hours, each GPU did about 343 trillion useful FLOPs per second. That leaves out attention, and it is our estimate, not DeepSeek's. Closed labs do not publish their training costs at all.

### Predicting more than one token

DeepSeek-V3 also trained with **multi-token prediction**. Besides the next token, a small extra module predicts the token after it.

> [!DEFINITION] Multi-token prediction (MTP)
> A training objective where the model predicts several future tokens at each position, not only the next one. It gives more training signal per token, and the extra predictions can later speed up generation.

Fabian Gloeckle and colleagues at Meta studied the idea in 2024. Their models trained with 4-token prediction were "up to 3× faster at inference". DeepSeek-V3 used a depth of 1, one extra token. It reports that the extra prediction is accepted 85% to 90% of the time, which gives about 1.8 times more tokens per second. The next section shows where that number comes from.

## Built for serving

A model is trained once and served for months to millions of people, so its design is shaped by serving costs. GQA, MLA and MoE are all, in part, serving decisions. Three more tricks matter.

### Speculative decoding

> [!DEFINITION] Speculative decoding
> A way to generate faster with no change to the output. A small, fast **draft** model guesses several tokens. The large model checks them all in one pass. It keeps the guesses that match its own choices and fixes the first wrong one.

Writing one token at a time leaves a GPU mostly idle, because each step reads all the weights to make one token. Checking five guessed tokens costs about the same as making one. In 2022, Yaniv Leviathan and colleagues at Google showed a 2X-3X speed-up on T5-XXL "with identical outputs". If each guess is accepted with probability $$a$$ and the draft guesses $$\gamma$$ tokens, each pass of the large model yields on average

$$
\mathbb{E}[\text{tokens per pass}] = \frac{1 - a^{\gamma + 1}}{1 - a}
$$

$$a$$ is the acceptance rate and $$\gamma$$ (gamma) is the draft length. The formula counts the accepted guesses plus the one token the large model always adds. With $$a = 0.8$$ and $$\gamma = 4$$, that is 3.362 tokens per pass; with $$a = 0.6$$, 2.306.

DeepSeek-V3 uses its MTP module as the draft, with $$\gamma = 1$$. At $$a = 0.85$$ the formula gives 1.850 tokens per pass, and at 0.9 it gives 1.900. That agrees with the 1.8 times speed-up the report gives. The [inference series](llm-inference-4-speculative-decoding.md) covers the method in depth.

### Quantisation

Part 1 introduced **quantisation**: storing weights in fewer bits. A 70B model needs 140 GB at 16 bits, 70 GB at 8 bits and 35 GB at 4 bits. DeepSeek-V3's 671B parameters take about 671 GB in FP8, the format it was trained in. Many open models are now released in 8-bit or 4-bit versions, and serving systems run them with special matrix kernels.

### Distillation into small models

Most users talk to small models. Labs make them by **distillation**, defined in [Part 4](how-models-are-trained-today.md): a small student learns from a big teacher's outputs. In pre-training, the student learns from the teacher's full probability over the next token, not only from the one token in the text. That is a far richer signal: it says that "cat" was likely, "dog" was nearly as likely, and "car" was not.

> [!PAPER] Gemini Team, Google (2025), Gemini 2.5 · §2.1 · page 3
> [![A passage of the Gemini 2.5 report: the smaller models in the Gemini 2.5 series, Flash size and below, use distillation; the teacher's next token prediction distribution is approximated with a k-sparse distribution over the vocabulary](/img/writings/history/h-gemini25-distill.png)](/img/writings/history/h-gemini25-distill.png)
>
> **What it says:** the smaller Gemini 2.5 models, Flash size and below, "use distillation", "as was done in the Gemini 1.5 series". To save storage, the teacher's next-token distribution is approximated "using a k-sparse distribution over the vocabulary".
>
> **What it means:**
>
> "k-sparse" means that only the teacher's top $$k$$ tokens and their probabilities are stored for each position, not all of the vocabulary. Even so, the data grows by a factor of $$k$$, and Google says the quality gain is worth it. The Gemini 1.5 report says that 1.5 Flash was "online distilled" from 1.5 Pro, meaning it learned while the teacher ran alongside it.

Google's open Gemma models follow the same path: the Gemma 3 report says that "all Gemma 3 models are trained with knowledge distillation". Meta used the 405B model to improve its smaller Llama 3 models in post-training, and Qwen3 describes "strong-to-weak distillation". The big model's main job in a family is often to teach the small ones.

## What labs publish, and what they do not

Everything in this part comes from papers, technical reports and model cards. They differ widely in how much they say.

> [!DEFINITION] Model card and technical report
> A **model card** is a short document that a lab publishes with a model. It says what the model is for, how it was evaluated, and its limits and safety tests. A **technical report** is longer and may describe the architecture, data and training, but labs choose what to put in it.

The open-weight labs publish the most: DeepSeek, Meta's Llama papers, Qwen, Kimi and above all the Allen Institute's Olmo, which also releases its data. Google's Gemini reports confirm the design in outline: sparse MoE, natively multimodal, long context, TPUs, distillation for the small models. OpenAI's GPT-4 report of 2023 set the closed pattern. It "contains no further details about the architecture (including model size), hardware, training compute, dataset construction, training method, or similar."

| Question | Open-weight reports (DeepSeek, Llama, Olmo) | Closed frontier labs |
|---|---|---|
| Layer design | Full: norms, attention type, sizes | Google: "sparse MoE", little more; others: nothing |
| Parameter count | Exact, total and active | Not published |
| Training tokens | Exact or close | Not published |
| Data sources | Described; Olmo releases the data | Categories only ("web documents, code, images") |
| Hardware | Chip type and count | Google names TPU generations; others: little |
| Cost | DeepSeek: GPU hours for the final run | Not published |

So, when you read a claim about a closed model's size or cost, ask where it comes from. If it is not in a lab's own report or model card, it is a guess or a leak, however often it is repeated.

## How the five parts fit together

Put the parts of this series side by side and you get the whole build of a frontier model in 2026.

<figure class="fig"><svg viewBox="0 0 720 390" role="img" aria-label="Eight stages in two rows: data, pre-training, mid-training, long context, post-training, distillation, serving, and the model card, with hardware and parallelism under all of them."><text class="t-title" x="10" y="22">How a frontier model is built, end to end</text><rect class="box-2" x="20" y="50" width="155" height="86" rx="6"/><text class="t-note" x="97.5" y="80.5" text-anchor="middle">Data</text><text class="t-muted" x="97.5" y="97.5" text-anchor="middle">crawl, filter, dedup,</text><text class="t-muted" x="97.5" y="114.5" text-anchor="middle">mix, tokenise</text><polyline class="edge" points="175,93 195,93"/><polygon class="arrow" points="195.0,93.0 187.0,97.0 187.0,89.0"/><rect class="box-1" x="195" y="50" width="155" height="86" rx="6"/><text class="t-note" x="272.5" y="80.5" text-anchor="middle">Pre-training</text><text class="t-muted" x="272.5" y="97.5" text-anchor="middle">trillions of tokens,</text><text class="t-muted" x="272.5" y="114.5" text-anchor="middle">next-token loss</text><polyline class="edge" points="350,93 370,93"/><polygon class="arrow" points="370.0,93.0 362.0,97.0 362.0,89.0"/><rect class="box-1" x="370" y="50" width="155" height="86" rx="6"/><text class="t-note" x="447.5" y="80.5" text-anchor="middle">Mid-training</text><text class="t-muted" x="447.5" y="97.5" text-anchor="middle">annealing on</text><text class="t-muted" x="447.5" y="114.5" text-anchor="middle">high-quality data</text><polyline class="edge" points="525,93 545,93"/><polygon class="arrow" points="545.0,93.0 537.0,97.0 537.0,89.0"/><rect class="box-1" x="545" y="50" width="155" height="86" rx="6"/><text class="t-note" x="622.5" y="80.5" text-anchor="middle">Long context</text><text class="t-muted" x="622.5" y="97.5" text-anchor="middle">RoPE scaling,</text><text class="t-muted" x="622.5" y="114.5" text-anchor="middle">8K → 128K → 1M</text><rect class="box-3" x="20" y="180" width="155" height="86" rx="6"/><text class="t-note" x="97.5" y="210.5" text-anchor="middle">Post-training</text><text class="t-muted" x="97.5" y="227.5" text-anchor="middle">SFT, preferences,</text><text class="t-muted" x="97.5" y="244.5" text-anchor="middle">reasoning RL (Part 4)</text><polyline class="edge" points="175,223 195,223"/><polygon class="arrow" points="195.0,223.0 187.0,227.0 187.0,219.0"/><rect class="box-3" x="195" y="180" width="155" height="86" rx="6"/><text class="t-note" x="272.5" y="210.5" text-anchor="middle">Distillation</text><text class="t-muted" x="272.5" y="227.5" text-anchor="middle">small models learn</text><text class="t-muted" x="272.5" y="244.5" text-anchor="middle">from the big one</text><polyline class="edge" points="350,223 370,223"/><polygon class="arrow" points="370.0,223.0 362.0,227.0 362.0,219.0"/><rect class="box-4" x="370" y="180" width="155" height="86" rx="6"/><text class="t-note" x="447.5" y="210.5" text-anchor="middle">Serving</text><text class="t-muted" x="447.5" y="227.5" text-anchor="middle">quantise, KV cache,</text><text class="t-muted" x="447.5" y="244.5" text-anchor="middle">speculative decoding</text><polyline class="edge" points="525,223 545,223"/><polygon class="arrow" points="545.0,223.0 537.0,227.0 537.0,219.0"/><rect class="box" x="545" y="180" width="155" height="86" rx="6"/><text class="t-note" x="622.5" y="210.5" text-anchor="middle">Model card</text><text class="t-muted" x="622.5" y="227.5" text-anchor="middle">what the lab</text><text class="t-muted" x="622.5" y="244.5" text-anchor="middle">chooses to publish</text><polyline class="edge" points="637,136 637,158 97,158 97,180"/><polygon class="arrow" points="97.0,180.0 93.0,172.0 101.0,172.0"/><rect class="box-ghost" x="20" y="300" width="680" height="44" rx="6"/><text class="t-note" x="360" y="327" text-anchor="middle">Under it all: thousands of GPUs or TPUs, parallelism, bf16 or FP8, checkpoints and restarts</text><text class="t-muted" x="20" y="372">Parts 1 and 5 cover the first four boxes, Part 4 the post-training box. Labs differ in order and detail.</text></svg><figcaption>A composite of the Llama 3, DeepSeek-V3, Gemini 2.5 and Olmo 3 reports. Closed labs confirm only parts of it.</figcaption></figure>

[Part 1](from-neuron-to-opus.md) gave the parts list: the neuron, backpropagation, attention, the Transformer, the next-token loss. This part showed what they look like at scale. Part 2's classical methods live on inside the pipeline, as the small fast classifiers that filter data. [Part 3](reinforcement-learning-history.md) and [Part 4](how-models-are-trained-today.md) take over once the base model exists, and turn it into an assistant that reasons and uses tools.

The lessons repeat across all five parts. A good loss is not the same as a good model. Goodhart's law in Part 4 and the warning about loss in this part both show it. Scale wins, but only when the bottleneck moves: from compute to data, from data to memory, from memory to the cost of every answer. And most progress is a chain of fixes, each solving the limit the step before it left behind.

## What is still open

Data is the hardest limit. Good public text is close to used up, so the future depends on synthetic data, other kinds of data and better reuse. Whether a model can learn as much from text written by models as from text written by people is still an open question.

Architecture is less settled than it looks. Many 2025 and 2026 models mix in other layer types, such as linear attention, state-space layers and sparse attention, to make long contexts cheaper. The [attention series](attention-4-linear-hybrid.md) covers them. DeepSeek-V3.2 added its own sparse attention. The SwiGLU paper's "divine benevolence" is still a fair summary of how many of these choices are made.

And scaling itself is changing shape. Pre-training compute keeps growing, but more of the total now goes to post-training and to thinking at answer time, as Part 4 showed. The next part of the curve may be drawn on that second axis.

| Step | What it fixed | What it left open |
|---|---|---|
| Next-token loss, perplexity | One clear goal for raw text | Loss is not ability |
| Kaplan scaling laws (2020) | Predictable gains from small runs | Wrong split between size and data |
| Chinchilla (2022) | Compute-optimal size and data | Ignores the cost of serving |
| Over-training (Llama 3, 2024) | Small models that are cheap to serve | Needs ever more data |
| Filtering, dedup, quality classifiers | Better data from the same web | Hard to test; can tilt the data |
| Byte-level BPE | Any text in a fixed vocabulary | Uneven cost across languages |
| Pre-norm, RMSNorm, SwiGLU | Stable, cheaper deep stacks | Found by trial, not theory |
| RoPE and its scaling | Relative positions; longer contexts | Quality falls far past the training length |
| FlashAttention | Exact attention without the $$n^2$$ memory | Still $$n^2$$ arithmetic |
| MQA, GQA, MLA | A much smaller KV cache | Extra design and tuning |
| Mixture of experts | More knowledge per FLOP | Load balance, memory, communication |
| Parallelism, ZeRO, FP8 | Training across thousands of chips | Failures, loss spikes, cost |
| Speculative decoding, quantisation, distillation | Cheaper, faster serving | Small models trail the big ones |

> [!TAKEAWAYS]
> - Pre-training minimises one number, the next-token cross-entropy; perplexity is its exponential.
> - Training compute is about $$6ND$$. Kaplan showed smooth power laws; Chinchilla showed that size and data should grow together, near 20 tokens per parameter.
> - Labs now over-train small models on far more tokens than that, because a smaller model is cheaper for every user. The limit is now the supply of good data.
> - Pre-training data is the filtered web: extraction, language checks, rules, deduplication and quality classifiers. It is mixed with code, maths, other languages and synthetic text, and ends with an annealing phase.
> - The 2026 decoder layer keeps the Transformer's skeleton but uses pre-norm RMSNorm, SwiGLU, RoPE and shared keys and values (GQA or MLA); FlashAttention makes attention fit the memory of the chip.
> - Mixture-of-experts layers send each token to a few experts chosen by a router, so total size and cost per token come apart; DeepSeek-V3 uses 37B of 671B parameters per token, and Gemini is a sparse MoE.
> - Training spreads one model over thousands of GPUs or TPUs with data, tensor, pipeline and expert parallelism. It shards the optimizer states, uses 16-bit or 8-bit numbers, and must survive constant hardware failures.
> - Open-weight labs publish their designs in detail; closed labs publish outlines, so claims about closed models need a source.

Key papers: [Kaplan et al. 2020](https://arxiv.org/abs/2001.08361), [Chinchilla 2022](https://arxiv.org/abs/2203.15556), [Llama 3 2024](https://arxiv.org/abs/2407.21783). Data and architecture: [FineWeb 2024](https://arxiv.org/abs/2406.17557), [BPE 2016](https://arxiv.org/abs/1508.07909), [RoFormer 2021](https://arxiv.org/abs/2104.09864), [FlashAttention 2022](https://arxiv.org/abs/2205.14135). Experts and scale: [Switch Transformer 2021](https://arxiv.org/abs/2101.03961), [DeepSeek-V3 2024](https://arxiv.org/abs/2412.19437), [Gemini 2.5 2025](https://arxiv.org/abs/2507.06261).

Back to [Part 4, From RLHF to Reasoning](how-models-are-trained-today.md), or to the start of the series, [Part 1, From One Neuron to Opus](from-neuron-to-opus.md).
