---
description: "How language models are measured, and how far to trust the numbers: held-out loss, perplexity and bits per byte; the history of benchmarks from GLUE to GPQA; how a multiple-choice question is really scored; confidence intervals; pass@k; contamination; human preference, Elo and Bradley-Terry ratings; model judges and their biases; reward-model evaluation and reward hacking. With real evaluations of Qwen2.5-0.5B base and Instruct, and the papers' own tables and figures."
---
# Chapter 3 · How we measure a model

> **Goal:** by the end of this chapter you can explain what each common kind of evaluation measures and what it misses, compute perplexity and bits per byte for a real model, score a multiple-choice benchmark three different ways and explain why the numbers differ, put a 95% confidence interval on any accuracy with the bootstrap, compute pass@k with the unbiased estimator, check a benchmark for contamination with n-grams, turn pairwise votes into Elo and Bradley-Terry ratings, and spot the biases of a model used as a judge. Every number in this chapter that is not quoted from a paper comes from a script you can run on a laptop.

---

## 3.1 Why measuring a model is hard

Chapter 1 ended with a small test and a warning: the prompt format, the token budget and the matching rule changed the score as much as the model did. This chapter takes that warning seriously. Every later chapter trains something, and each needs an answer to the same question: *did it get better?* If the measurement is wrong, every decision built on it is wrong too.

Measuring a calculator is easy: give it sums with known answers and count. Measuring a language model is hard for four reasons.

**It does many things.** One model writes poems, answers trivia, solves maths, writes code and chats, and can be excellent at one and poor at another. No single number describes all of them.

**Many tasks have no single right answer.** "What is 17 times 23?" has one answer; "write a polite email declining a meeting" has thousands of good ones. Open-ended tasks need a judge, a person or another model, and judges disagree and have biases.

**The answer depends on how you ask.** The same model, with the same knowledge, gets different scores when the question is phrased differently, when the options are reordered, or when the score is read from probabilities instead of from generated text. Section 3.4 shows this on a real model: the same model on the same 400 questions scores anywhere from 32.5% to 49.0% depending only on the scoring method.

**The test can leak.** Models are trained on trillions of words scraped from the web. Benchmarks are published on the web. If the test questions (or their answers) are in the training data, a high score may mean "memorised", not "can do". Section 3.8 is about this.

> [!DEFINITION] Evaluation
> Any procedure that turns a model's behaviour into a number (or a ranking) so that models, or versions of one model, can be compared. An evaluation has three parts: the data (which questions), the protocol (how the model is asked and how its answer is read), and the metric (how answers become a score). Change any of the three and the number changes.

> [!DEFINITION] Benchmark
> A fixed, public evaluation that many people run the same way, so their numbers can be compared: a set of test items, an answer key or a grading rule, and usually a standard prompt format. MMLU, GSM8K and HumanEval are benchmarks.

> [!DEFINITION] Metric
> The rule that turns answers into a score: accuracy (fraction right), exact match, pass@k, win rate, perplexity, and so on.

> [!DEFINITION] Held-out data
> Data the model never saw during training, kept aside only for evaluation. A test set is useful only while it stays held out.

### Six families of evaluation

Evaluations fall into a few families that differ in cost, in how close they are to what users experience, and in how they fail.

<figure class="fig"><svg viewBox="0 0 760 374" role="img" aria-label="Six boxes in two rows: held-out loss, benchmarks with an answer key, human preference, a model as the judge, reward-model tests, and safety and behaviour tests, each with the question it asks, an example and its main weakness."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Six families of evaluation, from cheapest to closest to what users feel</text><rect class="box-1" x="20.0" y="44.0" width="232.0" height="150.0" rx="10"/><text class="t-note" x="32.0" y="68.0" text-anchor="start">Held-out loss, perplexity</text><text class="t-tick" x="32.0" y="90.0" text-anchor="start">How well does it predict</text><text class="t-tick" x="32.0" y="107.0" text-anchor="start">text it has not seen?</text><text class="t-muted" x="32.0" y="134.0" text-anchor="start">e.g. WikiText-2, a validation split</text><text class="t-muted" x="32.0" y="158.0" text-anchor="start">cheap and smooth, but says</text><text class="t-muted" x="32.0" y="175.0" text-anchor="start">nothing about answers</text><rect class="box-2" x="266.0" y="44.0" width="232.0" height="150.0" rx="10"/><text class="t-note" x="278.0" y="68.0" text-anchor="start">Benchmarks with a key</text><text class="t-tick" x="278.0" y="90.0" text-anchor="start">Does it reach the known</text><text class="t-tick" x="278.0" y="107.0" text-anchor="start">right answer?</text><text class="t-muted" x="278.0" y="134.0" text-anchor="start">e.g. MMLU, GSM8K, HumanEval</text><text class="t-muted" x="278.0" y="158.0" text-anchor="start">leaks into training data,</text><text class="t-muted" x="278.0" y="175.0" text-anchor="start">saturates, format-sensitive</text><rect class="box-3" x="512.0" y="44.0" width="232.0" height="150.0" rx="10"/><text class="t-note" x="524.0" y="68.0" text-anchor="start">Human preference</text><text class="t-tick" x="524.0" y="90.0" text-anchor="start">Which of two answers do</text><text class="t-tick" x="524.0" y="107.0" text-anchor="start">people like more?</text><text class="t-muted" x="524.0" y="134.0" text-anchor="start">e.g. Chatbot Arena, InstructGPT</text><text class="t-muted" x="524.0" y="158.0" text-anchor="start">slow, costly, noisy;</text><text class="t-muted" x="524.0" y="175.0" text-anchor="start">style can beat substance</text><rect class="box-4" x="20.0" y="210.0" width="232.0" height="150.0" rx="10"/><text class="t-note" x="32.0" y="234.0" text-anchor="start">A model as the judge</text><text class="t-tick" x="32.0" y="256.0" text-anchor="start">Which answer does a strong</text><text class="t-tick" x="32.0" y="273.0" text-anchor="start">model prefer?</text><text class="t-muted" x="32.0" y="300.0" text-anchor="start">e.g. MT-Bench, AlpacaEval</text><text class="t-muted" x="32.0" y="324.0" text-anchor="start">position, length and</text><text class="t-muted" x="32.0" y="341.0" text-anchor="start">self-preference biases</text><rect class="box-1" x="266.0" y="210.0" width="232.0" height="150.0" rx="10"/><text class="t-note" x="278.0" y="234.0" text-anchor="start">Reward-model tests</text><text class="t-tick" x="278.0" y="256.0" text-anchor="start">Does the reward model</text><text class="t-tick" x="278.0" y="273.0" text-anchor="start">rank answer pairs right?</text><text class="t-muted" x="278.0" y="300.0" text-anchor="start">e.g. RewardBench</text><text class="t-muted" x="278.0" y="324.0" text-anchor="start">a high score does not mean</text><text class="t-muted" x="278.0" y="341.0" text-anchor="start">safe to optimise against</text><rect class="box-2" x="512.0" y="210.0" width="232.0" height="150.0" rx="10"/><text class="t-note" x="524.0" y="234.0" text-anchor="start">Safety and behaviour</text><text class="t-tick" x="524.0" y="256.0" text-anchor="start">Does it refuse, invent or</text><text class="t-tick" x="524.0" y="273.0" text-anchor="start">comply when it should not?</text><text class="t-muted" x="524.0" y="300.0" text-anchor="start">e.g. refusal sets, red-teaming</text><text class="t-muted" x="524.0" y="324.0" text-anchor="start">endless cases; passing a</text><text class="t-muted" x="524.0" y="341.0" text-anchor="start">test is not proof of safety</text></svg><figcaption>Six families of evaluation. The top-left family is the cheapest and runs constantly during training; the families on the right and below are slower and more expensive, and closer to "is this model good for people?". The last lines of each box name its typical way of failing.</figcaption></figure>

This chapter goes through them in roughly that order: loss and perplexity (Section 3.2), benchmarks with an answer key (Sections 3.3 to 3.8), human preference (3.9), model judges (3.10), reward-model tests (3.11), safety and behaviour (3.12), and finally how all of this fits into a training run (3.13).

### Goodhart's law

There is one idea that runs through the whole chapter. It is usually quoted as **Goodhart's law**, in the form the anthropologist Marilyn Strathern gave it in 1997: *"When a measure becomes a target, it ceases to be a good measure."* The economist Charles Goodhart had made the original point about monetary statistics in 1975.

> [!DEFINITION] Goodhart's law
> When people (or optimisers) push hard on a number that was chosen because it tracked a goal, they find ways to raise the number that do not serve the goal, and the number stops tracking it.

Machine learning is an optimiser by nature, so it meets Goodhart's law everywhere. A benchmark score tracks knowledge, until test questions leak into the training data. A judge model's preference tracks answer quality, until models learn that judges like long answers. A reward model's score tracks what people want, until a policy trained against it finds outputs the reward model over-rates. You will see all three happen, with real numbers, later in this chapter.

<figure class="fig"><svg viewBox="0 0 760 274" role="img" aria-label="Three rows. Knowledge is measured by a benchmark score, which breaks when test questions leak into training. Answer quality is measured by a judge model, which breaks because judges favour long answers and the first position. Human wishes are measured by a reward model, which breaks through reward hacking."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Goodhart's law: once a measure becomes the target, it stops tracking the goal</text><text class="t-muted" x="20.0" y="52.0" text-anchor="start">what we want</text><text class="t-muted" x="230.0" y="52.0" text-anchor="start">what we can count</text><text class="t-muted" x="440.0" y="52.0" text-anchor="start">how optimising the count breaks the link</text><rect class="box-3" x="20.0" y="64.0" width="180.0" height="52.0" rx="10"/><text class="t-note" x="110.0" y="95.0" text-anchor="middle">knowledge of the subject</text><line class="edge" x1="204.0" y1="90.0" x2="226.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-2" x="230.0" y="64.0" width="180.0" height="52.0" rx="10"/><text class="t-note" x="320.0" y="95.0" text-anchor="middle">benchmark score</text><line class="edge" x1="414.0" y1="90.0" x2="436.0" y2="90.0" marker-end="url(#ah)"/><rect class="box-ghost" x="440.0" y="64.0" width="300.0" height="52.0" rx="10"/><text class="t-tick" x="452.0" y="86.0" text-anchor="start">test questions leak into the training data;</text><text class="t-tick" x="452.0" y="103.0" text-anchor="start">the score rises, the knowledge does not</text><rect class="box-3" x="20.0" y="134.0" width="180.0" height="52.0" rx="10"/><text class="t-note" x="110.0" y="165.0" text-anchor="middle">a better answer</text><line class="edge" x1="204.0" y1="160.0" x2="226.0" y2="160.0" marker-end="url(#ah)"/><rect class="box-2" x="230.0" y="134.0" width="180.0" height="52.0" rx="10"/><text class="t-note" x="320.0" y="165.0" text-anchor="middle">a judge model prefers it</text><line class="edge" x1="414.0" y1="160.0" x2="436.0" y2="160.0" marker-end="url(#ah)"/><rect class="box-ghost" x="440.0" y="134.0" width="300.0" height="52.0" rx="10"/><text class="t-tick" x="452.0" y="156.0" text-anchor="start">the judge likes long answers and the first</text><text class="t-tick" x="452.0" y="173.0" text-anchor="start">position; padding wins votes</text><rect class="box-3" x="20.0" y="204.0" width="180.0" height="52.0" rx="10"/><text class="t-note" x="110.0" y="235.0" text-anchor="middle">what people actually want</text><line class="edge" x1="204.0" y1="230.0" x2="226.0" y2="230.0" marker-end="url(#ah)"/><rect class="box-2" x="230.0" y="204.0" width="180.0" height="52.0" rx="10"/><text class="t-note" x="320.0" y="235.0" text-anchor="middle">reward model score</text><line class="edge" x1="414.0" y1="230.0" x2="436.0" y2="230.0" marker-end="url(#ah)"/><rect class="box-ghost" x="440.0" y="204.0" width="300.0" height="52.0" rx="10"/><text class="t-tick" x="452.0" y="226.0" text-anchor="start">the policy finds outputs the reward model</text><text class="t-tick" x="452.0" y="243.0" text-anchor="start">over-rates: reward hacking</text></svg><figcaption>Three places where a measure is turned into a target. In each row the middle box is what we can count, and the right box is how optimising that count breaks its link to what we actually want.</figcaption></figure>

So for every number in this chapter, and in every model report you read, ask three questions:

1. **What exactly was measured?** Which data, which prompt, which way of reading the answer.
2. **How sure is it?** How many test items, and how wide is the confidence interval (Section 3.6)?
3. **Could the model have been optimised for this number, directly or by accident?** Leaked test data, tuning on the test set, or training against the judge.

## 3.2 Intrinsic metrics: loss, perplexity and bits per byte

The cheapest evaluation needs only text the model has not seen: how well does the model predict it? Chapter 1 (Section 1.5) defined the two numbers, the average cross-entropy loss and its exponential, perplexity.

> [!DEFINITION] Intrinsic and extrinsic evaluation
> An intrinsic metric measures how well the model does the thing it was trained to do (predict the next token). An extrinsic metric measures how well it does a downstream task (answer a question, write working code). Intrinsic metrics are cheap and smooth; extrinsic ones are what users feel.

A quick reminder of the two formulas, for a held-out text of $$T$$ tokens:

$$\mathcal{L} = -\frac{1}{T-1}\sum_{t=2}^{T} \ln P_\theta(x_t \mid x_{<t}), \qquad \text{PPL} = e^{\mathcal{L}}$$

where:

- $$x_t$$ is the $$t$$-th token of the text and $$x_{<t}$$ all the tokens before it;
- $$P_\theta(x_t \mid x_{<t})$$ is the probability the model gave to the token that actually came next;
- $$\ln$$ is the natural logarithm, so $$\mathcal{L}$$ is in nats per token;
- $$\text{PPL}$$, the perplexity, is "as unsure as choosing uniformly among this many tokens".

### Measuring perplexity on a real benchmark: WikiText-2

**WikiText-2** (Merity et al., 2016) is a set of good-quality Wikipedia articles split into train, validation and test sets. Its test split, about 1.3 million characters, has been the standard text for reporting perplexity since before GPT-2. We score three models on it: GPT-2 (124M parameters, 2019), Qwen2.5-0.5B base and Qwen2.5-0.5B-Instruct.

The text is far longer than a model can read at once, so it is scored in overlapping windows.

<figure class="fig"><svg viewBox="0 0 760 354" role="img" aria-label="A long strip of tokens with four overlapping windows of 1024 tokens, each shifted by 512. In the first window all tokens are scored; in the later windows only the last 512 tokens are scored and the first 512 serve as context."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Scoring a long text with a fixed context: window 1024 tokens, stride 512</text><text class="t-muted" x="20.0" y="44.0" text-anchor="start">WikiText-2 test is 299,078 Qwen tokens long, far more than one forward pass can take</text><rect class="box-ghost" x="40.0" y="64.0" width="640.0" height="26.0" rx="4"/><text class="t-tick" x="40.0" y="108.0" text-anchor="middle">0</text><text class="t-tick" x="104.0" y="108.0" text-anchor="middle">256</text><text class="t-tick" x="168.0" y="108.0" text-anchor="middle">512</text><text class="t-tick" x="232.0" y="108.0" text-anchor="middle">768</text><text class="t-tick" x="296.0" y="108.0" text-anchor="middle">1,024</text><text class="t-tick" x="360.0" y="108.0" text-anchor="middle">1,280</text><text class="t-tick" x="424.0" y="108.0" text-anchor="middle">1,536</text><text class="t-tick" x="488.0" y="108.0" text-anchor="middle">1,792</text><text class="t-tick" x="552.0" y="108.0" text-anchor="middle">2,048</text><text class="t-tick" x="616.0" y="108.0" text-anchor="middle">2,304</text><text class="t-tick" x="680.0" y="108.0" text-anchor="middle">2,560</text><text class="t-tick" x="690.0" y="82.0" text-anchor="start">tokens ...</text><text class="t-note" x="20.0" y="145.0" text-anchor="middle">1</text><rect class="box" x="40.0" y="128.0" width="256.0" height="26.0" rx="4"/><rect class="s2" x="40.0" y="128.0" width="256.0" height="26.0" rx="4"/><text class="t-tick" x="48.0" y="146.0" text-anchor="start"></text><text class="t-tick" x="168.0" y="146.0" text-anchor="middle">scored</text><text class="t-note" x="20.0" y="189.0" text-anchor="middle">2</text><rect class="box" x="168.0" y="172.0" width="256.0" height="26.0" rx="4"/><rect class="s2" x="296.0" y="172.0" width="128.0" height="26.0" rx="4"/><text class="t-tick" x="176.0" y="190.0" text-anchor="start">context only</text><text class="t-tick" x="360.0" y="190.0" text-anchor="middle">scored</text><text class="t-note" x="20.0" y="233.0" text-anchor="middle">3</text><rect class="box" x="296.0" y="216.0" width="256.0" height="26.0" rx="4"/><rect class="s2" x="424.0" y="216.0" width="128.0" height="26.0" rx="4"/><text class="t-tick" x="304.0" y="234.0" text-anchor="start">context only</text><text class="t-tick" x="488.0" y="234.0" text-anchor="middle">scored</text><text class="t-note" x="20.0" y="277.0" text-anchor="middle">4</text><rect class="box" x="424.0" y="260.0" width="256.0" height="26.0" rx="4"/><rect class="s2" x="552.0" y="260.0" width="128.0" height="26.0" rx="4"/><text class="t-tick" x="432.0" y="278.0" text-anchor="start">context only</text><text class="t-tick" x="616.0" y="278.0" text-anchor="middle">scored</text><text class="t-muted" x="20.0" y="314.0" text-anchor="start">Each window re-reads the previous 512 tokens as context but only scores its new 512, so every token is scored once</text><text class="t-muted" x="20.0" y="331.0" text-anchor="start">and (after the first window) always with at least 512 tokens of history. Smaller stride = more context, more compute.</text></svg><figcaption>Sliding-window scoring. Each window holds 1,024 tokens and starts 512 tokens after the previous one. Only the new half of each window is scored, so every token is scored once and (after the first window) always sees at least 512 tokens of history.</figcaption></figure>

```python
# simplified from ch3_perplexity.py
def sliding_nll(model, ids, window=1024, stride=512):
    total, count, prev_end = 0.0, 0, 0
    for begin in range(0, len(ids), stride):
        end = min(begin + window, len(ids))
        x = torch.tensor([ids[begin:end]], device=DEV)
        logits = model(x).logits[0, :-1].float()               # position t predicts token t+1
        nll = Fn.cross_entropy(logits, x[0, 1:], reduction='none')
        new = end - prev_end                                   # tokens no earlier window has scored
        nll = nll[-new:] if begin > 0 else nll
        total += nll.sum().item(); count += nll.numel()
        prev_end = end
        if end == len(ids):
            break
    return total, count                                        # summed nats, number of scored tokens
```

The loop moves a window of `window` tokens forward by `stride`. One forward pass gives the loss of every token (`reduction='none'`); the line with `new` keeps only the tokens this window adds, so overlapping tokens serve as context but are not counted twice. Total loss divided by tokens scored is $$\mathcal{L}$$.

```text
== A. WikiText-2 test set ==
characters: 1,294,336   UTF-8 bytes: 1,296,370
GPT-2 (124M)             tokens 287,644  bytes/token 4.51  loss 3.2257 nats/token  PPL   25.17  bits/byte 1.0326
Qwen2.5-0.5B base        tokens 299,078  bytes/token 4.33  loss 2.5258 nats/token  PPL   12.50  bits/byte 0.8407
Qwen2.5-0.5B-Instruct    tokens 299,078  bytes/token 4.33  loss 2.6140 nats/token  PPL   13.65  bits/byte 0.8700
```

Qwen2.5-0.5B base reaches 12.50: on average it is as unsure as if it chose among about 12 tokens, half as many as GPT-2.

### Perplexity depends on the tokenizer: bits per byte

Can we conclude that Qwen predicts Wikipedia twice as well as GPT-2? Not from perplexity. Perplexity is *per token*, and the two models cut the same text into different tokens: GPT-2 into 287,644 tokens (4.51 bytes each on average), Qwen into 299,078 (4.33 bytes each). A model with a tokenizer that makes longer tokens has fewer, harder predictions to make, so its per-token loss is naturally higher. Perplexities of models with different tokenizers are not comparable.

The fix is to divide by something both models share: the text itself. **Bits per byte** is the total loss of the whole text, in bits, divided by its length in bytes.

$$\text{BPB} = \frac{\sum_t -\ln P_\theta(x_t \mid x_{<t})}{\ln 2 \times N_{\text{bytes}}} = \frac{\mathcal{L}}{\ln 2} \times \frac{N_{\text{tokens}}}{N_{\text{bytes}}}$$

where:

- the numerator is the total loss over all scored tokens, in nats;
- dividing by $$\ln 2 \approx 0.693$$ turns nats into **bits** (a bit is the information in one fair coin flip);
- $$N_{\text{bytes}}$$ is the length of the text in bytes of UTF-8 (1,296,370 here, the same for every model);
- $$N_{\text{tokens}} / N_{\text{bytes}}$$ is one over the average bytes per token, so the second form says "bits per token, divided by bytes per token".

> [!DEFINITION] Bits per byte (BPB)
> The number of bits a model needs, on average, to encode each byte of a text using its predicted probabilities. It does not depend on the tokenizer, so models with different vocabularies can be compared. For comparison, `ch3_gzip.py` compresses the same WikiText-2 text with ordinary compressors: gzip needs 2.75 bits per byte, bzip2 2.20 and xz 2.27. Even GPT-2, at 1.03, is a much better compressor of English, because predicting text well and compressing it are the same problem.

**Worked example.** Qwen2.5-0.5B base: 2.5258 nats per token is $$2.5258 / 0.6931 = 3.644$$ bits per token. Each token is 4.33 bytes on average, so BPB $$= 3.644 / 4.33 = 0.841$$. GPT-2: $$3.2257 / 0.6931 = 4.654$$ bits per token, over 4.51 bytes, gives 1.033. The printed values (0.8407 and 1.0326) are the same up to rounding.

<figure class="fig"><svg viewBox="0 0 760 226" role="img" aria-label="Two bar panels. Left: perplexity of GPT-2, Qwen2.5-0.5B base and Qwen2.5-0.5B-Instruct on WikiText-2. Right: bits per byte for the same three."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">WikiText-2 test set: the same text, three models</text><text class="t-muted" x="20.0" y="52.0" text-anchor="start">perplexity (per token: depends on the tokenizer)</text><text class="t-tick" x="180.0" y="78.0" text-anchor="end">GPT-2 (124M)</text><g class="mark"><title>GPT-2 (124M): 25.17</title><rect class="s4" x="190.0" y="64.0" width="151.0" height="18.0" rx="3"/></g><text class="t-val" x="347.0" y="78.0" text-anchor="start">25.17</text><text class="t-tick" x="180.0" y="108.0" text-anchor="end">Qwen2.5-0.5B base</text><g class="mark"><title>Qwen2.5-0.5B base: 12.50</title><rect class="s1" x="190.0" y="94.0" width="75.0" height="18.0" rx="3"/></g><text class="t-val" x="271.0" y="108.0" text-anchor="start">12.50</text><text class="t-tick" x="180.0" y="138.0" text-anchor="end">Qwen2.5-0.5B-Instruct</text><g class="mark"><title>Qwen2.5-0.5B-Instruct: 13.65</title><rect class="s3" x="190.0" y="124.0" width="81.9" height="18.0" rx="3"/></g><text class="t-val" x="277.9" y="138.0" text-anchor="start">13.65</text><text class="t-muted" x="400.0" y="52.0" text-anchor="start">bits per byte (comparable across tokenizers)</text><text class="t-tick" x="400.0" y="78.0" text-anchor="end"></text><g class="mark"><title>: 1.033</title><rect class="s4" x="410.0" y="64.0" width="268.5" height="18.0" rx="3"/></g><text class="t-val" x="684.5" y="78.0" text-anchor="start">1.033</text><text class="t-tick" x="400.0" y="108.0" text-anchor="end"></text><g class="mark"><title>: 0.841</title><rect class="s1" x="410.0" y="94.0" width="218.6" height="18.0" rx="3"/></g><text class="t-val" x="634.6" y="108.0" text-anchor="start">0.841</text><text class="t-tick" x="400.0" y="138.0" text-anchor="end"></text><g class="mark"><title>: 0.870</title><rect class="s3" x="410.0" y="124.0" width="226.2" height="18.0" rx="3"/></g><text class="t-val" x="642.2" y="138.0" text-anchor="start">0.870</text><text class="t-muted" x="20.0" y="170.0" text-anchor="start">tokens: GPT-2 287,644 (4.51 bytes each), Qwen 299,078 (4.33 bytes each).</text><text class="t-muted" x="20.0" y="187.0" text-anchor="start">Lower is better on both. Perplexity divides by a different number of tokens for each tokenizer,</text><text class="t-muted" x="20.0" y="204.0" text-anchor="start">so only bits per byte can say which model predicts this text better.</text></svg><figcaption>The same text and three models. Left: perplexity, which mixes model quality with tokenizer choice. Right: bits per byte, which compares only the models. Qwen2.5-0.5B base needs 0.84 bits for each byte of Wikipedia, GPT-2 1.03.</figcaption></figure>

So Qwen base is better than GPT-2 by about 19% in bits per byte, not by a factor of two. Here the ranking survives and only the gap shrinks; between tokenizers that differ more (for example on code or other languages), the ranking itself can flip.

### The same model, different numbers

Here is the first example of a theme of this chapter: the protocol is part of the number. The script scores Qwen2.5-0.5B base four more times on the same text, changing only the window and stride.

```text
== B. same model, different measuring choices (Qwen2.5-0.5B base) ==
window  1024 stride  1024:  loss 2.6848  PPL  14.66  bits/byte 0.8927
window  1024 stride   512:  loss 2.5258  PPL  12.50  bits/byte 0.8407
window   256 stride   256:  loss 3.0588  PPL  21.30  bits/byte 1.0141
window  2048 stride  1024:  loss 2.4571  PPL  11.67  bits/byte 0.8178
```

The same model on the same text has a perplexity anywhere from 11.67 to 21.30. With short windows (256 tokens, no overlap) many tokens are predicted with almost no context, and each window's first tokens are nearly guesses. With non-overlapping 1,024-token windows the start of each window is still blind. Overlap and longer windows give more context and a lower number. None of these is "the" perplexity; a number is only comparable to another computed the same way.

### Perplexity is not usefulness

Now compare the base model with the Instruct model. On Wikipedia the base model wins: 12.50 against 13.65, or 0.841 against 0.870 bits per byte. By this metric, post-training made the model *worse*. Of course it did: SFT and preference tuning moved its probabilities away from "continue a Wikipedia article" and toward "answer like an assistant" (Chapter 1, Section 1.9 measured that shift token by token). The Instruct model is not worse at its job; its job changed.

To check, the third part of the script scores both models on assistant-style text: 200 GSM8K reference solutions, written as the assistant's reply to the question in the chat template, with the loss counted only on the answer tokens.

```text
== C. assistant-style text: 200 GSM8K test solutions in the chat template, loss on answer tokens only ==
Qwen2.5-0.5B base        answer tokens 17,715  loss 1.2056  PPL 3.34
Qwen2.5-0.5B-Instruct    answer tokens 17,715  loss 1.0454  PPL 2.84
```

Now the order flips: the Instruct model predicts assistant answers better (perplexity 2.84 against 3.34). The two models are not "better" and "worse"; each fits a different kind of text. And neither perplexity tells you which one *answers maths questions correctly* more often. That needs a task metric, which is where the rest of the chapter goes.

<figure class="fig"><svg viewBox="0 0 760 274" role="img" aria-label="A table of four measurements for the base and Instruct models: WikiText-2 perplexity, perplexity of chat answers, MMLU accuracy and GSM8K accuracy, with the winner of each row highlighted."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Base or Instruct: which is "better"? It depends on the measurement</text><text class="t-note" x="330.0" y="52.0" text-anchor="middle">base</text><text class="t-note" x="470.0" y="52.0" text-anchor="middle">Instruct</text><text class="t-muted" x="620.0" y="52.0" text-anchor="middle">better is</text><text class="t-tick" x="20.0" y="85.0" text-anchor="start">WikiText-2 perplexity</text><rect class="box-on" x="270.0" y="66.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="330.0" y="85.0" text-anchor="middle">12.50</text><rect class="box" x="410.0" y="66.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="470.0" y="85.0" text-anchor="middle">13.65</text><text class="t-muted" x="620.0" y="85.0" text-anchor="middle">lower</text><text class="t-tick" x="20.0" y="125.0" text-anchor="start">perplexity of chat answers</text><rect class="box" x="270.0" y="106.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="330.0" y="125.0" text-anchor="middle">3.34</text><rect class="box-on" x="410.0" y="106.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="470.0" y="125.0" text-anchor="middle">2.84</text><text class="t-muted" x="620.0" y="125.0" text-anchor="middle">lower</text><text class="t-tick" x="20.0" y="165.0" text-anchor="start">MMLU, 5-shot letter</text><rect class="box" x="270.0" y="146.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="330.0" y="165.0" text-anchor="middle">49.0%</text><rect class="box-on" x="410.0" y="146.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="470.0" y="165.0" text-anchor="middle">49.2%</text><text class="t-muted" x="620.0" y="165.0" text-anchor="middle">higher</text><text class="t-tick" x="20.0" y="205.0" text-anchor="start">GSM8K, as each is meant to be used</text><rect class="box" x="270.0" y="186.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="330.0" y="205.0" text-anchor="middle">36.0%</text><rect class="box-on" x="410.0" y="186.0" width="120.0" height="28.0" rx="6"/><text class="t-val" x="470.0" y="205.0" text-anchor="middle">42.3%</text><text class="t-muted" x="620.0" y="205.0" text-anchor="middle">higher</text><text class="t-muted" x="20.0" y="234.0" text-anchor="start">Highlighted: the winner on that row. The base model predicts Wikipedia text better; the Instruct model predicts</text><text class="t-muted" x="20.0" y="251.0" text-anchor="start">assistant answers better. Neither number says which one is more useful: that needs task metrics.</text></svg><figcaption>Four measurements of the same two models. Each model wins on the text that looks like its training target. The bottom two rows, real task accuracy, come from Sections 3.5 and later; neither perplexity row predicts them.</figcaption></figure>

So perplexity measures fit to a distribution of text. It is excellent for comparing checkpoints of the *same* model on the *same* data, which is why pretraining runs watch it constantly (Chapter 4), but it says nothing about helpfulness, correctness or safety, and it is comparable only within one protocol and one tokenizer.

## 3.3 A short history of benchmarks

Before looking at how to score a benchmark, it helps to know where today's benchmarks came from. The story repeats one pattern again and again: a benchmark is built to be hard, models catch up within a few years, the benchmark **saturates**, and a harder one replaces it.

> [!DEFINITION] Saturation
> A benchmark is saturated when the best models score close to its ceiling (often the human baseline or 100%), so it can no longer tell strong models apart. Small differences near the top are then mostly noise, label errors and contamination.

### GLUE and SuperGLUE (2018 to 2019): many tasks, one number

**GLUE** (Wang et al., 2018) averaged nine sentence-understanding tasks (is this sentence grammatical, does A imply B, are these questions duplicates) into one "GLUE score". In the BERT era a model was fine-tuned separately on each task and then tested. One leaderboard number made progress easy to compare, and easy to saturate.

Within about a year the best GLUE score (88.4, July 2019) passed the human baseline (87.1), as the SuperGLUE paper noted when it introduced a "stickier" replacement (Wang et al., 2019). SuperGLUE lasted until January 2021, when DeBERTa scored 89.9 against a human baseline of 89.8 (He et al., 2021).

### HellaSwag (2019): adversarial filtering

HellaSwag (Zellers et al., 2019) asks a model to pick the most sensible ending of a short description of an everyday activity, from four options. The trick was in how the wrong endings were made: they were generated by a language model and then filtered so that the *models of the time* found them hard, while people did not.

> [!PAPER] Zellers et al. (2019), HellaSwag · Abstract · page 1
> [![The HellaSwag abstract, highlighted: its questions are trivial for humans, over 95% accuracy, while state-of-the-art models struggle, under 48%, achieved by Adversarial Filtering](/img/training/ch3-hellaswag-abstract.png)](/img/training/ch3-hellaswag-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** the questions are "trivial for humans" (above 95%) but "state-of-the-art models struggle" (below 48%), thanks to Adversarial Filtering.
>
> **Why it matters:** a benchmark built against one year's models measures that year's weaknesses. The Qwen2.5 report lists 52.1% for the tiny 0.5B base model we test in this chapter, above the best model of 2019.

### MMLU (2020): knowledge across 57 subjects

With GPT-3, models stopped being fine-tuned for each test. A single model was *prompted*, often with a few solved examples in the prompt (**few-shot**). Benchmarks changed to match. **MMLU** (Massive Multitask Language Understanding, Hendrycks et al., 2020) is a set of four-option multiple-choice questions from 57 subjects, from elementary mathematics to professional law and medicine, collected from practice exams. It has 5 example questions per subject for few-shot prompts, a validation set, and about 14,000 test questions (14,042 in the copy on the Hugging Face Hub that we use).

> [!PAPER] Hendrycks et al. (2020), MMLU · Figure 1 · page 2
> [![Figure 1 of the MMLU paper. Left: a few-shot prompt with two solved high school mathematics questions and a third whose answer, C, was completed by GPT-3. Right: bar chart of GPT-3 few-shot performance by model size on a commonsense benchmark, a linguistic benchmark and the new knowledge test; only the largest model rises clearly above 25% on the knowledge test](/img/training/ch3-mmlu-fig1.png)](/img/training/ch3-mmlu-fig1.png)
>
> **Context:** the first figure of the paper.
>
> **What it says:** left, the exact prompt format, ending in "Answer:"; right, GPT-3 of every size is far above chance on HellaSwag and SuperGLUE, but on the new test only the largest model clearly beats the 25% of random guessing.
>
> **Why it matters:** MMLU was designed to have headroom. The 175B GPT-3 scored 43.9%, unspecialised crowd workers 34.5%, and the authors estimated expert-level accuracy at about 89.8%. Three years later GPT-4 reported 86.4% (OpenAI, 2023). MMLU became the most quoted number in model reports, and then it saturated too.

### TruthfulQA, GSM8K and HumanEval (2021): new kinds of questions

2021 brought benchmarks that test something other than recognising the right option.

**TruthfulQA** (Lin et al., 2021) has 817 questions that some people answer falsely because of a common misconception ("What happens if you crack your knuckles a lot?"). A model that imitates the internet will repeat the misconception. The best model in the paper was truthful on 58% of questions, against 94% for people.

> [!PAPER] Lin et al. (2021), TruthfulQA · Figure 2 · page 3
> [![Two bar charts. Top: average truthfulness on TruthfulQA for GPT-3, GPT-Neo/J, GPT-2 and UnifiedQA at several sizes; within each family the larger models are less truthful. Bottom: on control trivia questions the larger models are more accurate](/img/training/ch3-truthfulqa-fig2.png)](/img/training/ch3-truthfulqa-fig2.png)
>
> **Context:** the main result.
>
> **What it says:** "Larger models are less truthful." On ordinary trivia (bottom) bigger is better; on questions that probe misconceptions (top) bigger is worse.
>
> **Why it matters:** predicting human text well includes predicting human mistakes, one reason post-training (Chapter 5 and later) is needed.

**GSM8K** (Cobbe et al., 2021) is 8,500 grade-school maths word problems (7,500 for training, 1,000 for testing; 1,319 test problems in the released file), each needing 2 to 8 steps of basic arithmetic. The answer is a single number, so grading is simple: extract the final number and compare. The paper's own fine-tuned 175B GPT-3 solved only about a third of the test set; training a separate **verifier** to pick among many sampled solutions helped about as much as making the model 30 times bigger.

> [!PAPER] Cobbe et al. (2021), GSM8K · Figure 1 · page 2
> [![Three example GSM8K problems with step-by-step solutions. Calculation annotations such as <<4*2=8>> are shown in red, and each solution ends with a final numeric answer](/img/training/ch3-gsm8k-fig1.png)](/img/training/ch3-gsm8k-fig1.png)
>
> **Context:** examples from the dataset.
>
> **What it says:** each solution is natural-language steps with "calculation annotations" in double angle brackets, and a final answer (written after `####` in the data file).
>
> **Why it matters:** a numeric final answer can be graded automatically, which made GSM8K the standard maths benchmark and later a source of *verifiable rewards* for RL (Chapter 2, Section 2.9).

**HumanEval** (Chen et al., 2021) is 164 hand-written Python problems. The model gets a function signature and a docstring and must write the body; the answer is right if it passes the problem's unit tests. Section 3.7 looks at it closely, because it introduced the **pass@k** metric.

### BIG-bench (2022): crowd-sourced breadth

**BIG-bench** (Srivastava et al., 2022) went for breadth: 204 tasks from 450 authors at 132 institutions, with human expert raters as a baseline. Its harder subset, BIG-Bench Hard (BBH), became a standard reasoning test.

### Saturation, and the next generation (2023 to 2024)

By 2023 the strongest models scored around 86% on MMLU, 92% on GSM8K (with chain-of-thought prompting) and 67% on HumanEval (GPT-4 technical report). New benchmarks were made harder in two ways.

**MMLU-Pro** (Wang et al., 2024) keeps the MMLU format but uses ten options instead of four, removes trivial and noisy questions, and adds questions that need reasoning.

> [!PAPER] Wang et al. (2024), MMLU-Pro · Abstract · page 1
> [![The MMLU-Pro abstract, highlighted: expanding the choice set from four to ten options; with 24 prompt styles tested, the sensitivity of model scores to prompt variations decreased from 4-5% in MMLU to just 2% in MMLU-Pro](/img/training/ch3-mmlupro-abstract.png)](/img/training/ch3-mmlupro-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** ten options instead of four lower the chance of a lucky guess from 25% to 10%, and scores move less when the prompt changes ("from 4-5% in MMLU to just 2%").
>
> **Why it matters:** prompt sensitivity is a property of the benchmark as well as of the model. We measure it ourselves in Section 3.4.

**GPQA** (Rein et al., 2023) goes the other way: 448 questions so hard that even experts with web access struggle.

> [!PAPER] Rein et al. (2023), GPQA · Abstract · page 1
> [![The GPQA abstract, highlighted: experts with PhDs reach 65% accuracy, while highly skilled non-expert validators only reach 34% accuracy despite over 30 minutes with unrestricted access to the web](/img/training/ch3-gpqa-abstract.png)](/img/training/ch3-gpqa-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** PhD-level experts "reach 65% accuracy", skilled non-experts "only reach 34%" even with "unrestricted access to the web", so the questions are "Google-proof".
>
> **Why it matters:** at expert level, the hard part is writing questions with certain answers that cannot be looked up. With only 448 questions, each is worth 0.22 points, so intervals are wide (Section 3.6).

Here is the history in one table. The "then" numbers are the best results reported in the benchmark's own paper; the "later" numbers show how fast the gap closed.

| Benchmark | Year | What it tests | Size (test) | Metric | Then | Later |
|---|---|---|---|---|---|---|
| GLUE | 2018 | sentence understanding, 9 tasks | varies | average score | | 88.4 vs human 87.1 (2019) |
| SuperGLUE | 2019 | harder understanding, 8 tasks | varies | average score | | 89.9 vs human 89.8 (Jan 2021) |
| HellaSwag | 2019 | commonsense sentence endings | 10,042 (validation) | accuracy (4 options) | below 48% | 52.1% for a 0.5B model (2024) |
| MMLU | 2020 | knowledge, 57 subjects | 14,042 | accuracy (4 options) | 43.9% (GPT-3) | 86.4% (GPT-4, 2023) |
| TruthfulQA | 2021 | avoiding common misconceptions | 817 | % truthful | 58% (human 94%) | |
| GSM8K | 2021 | multi-step arithmetic | 1,319 | exact match | about 33% (fine-tuned) | 92.0% (GPT-4, 2023) |
| HumanEval | 2021 | Python functions | 164 | pass@k | 28.8% pass@1 | 67.0% (GPT-4, 2023) |
| BIG-bench | 2022 | 204 varied tasks | varies | per task | | |
| GPQA | 2023 | expert science, "Google-proof" | 448 | accuracy (4 options) | experts 65% | |
| MMLU-Pro | 2024 | harder MMLU, 10 options | 12,032 | accuracy | | |

One thing the table does not show: none of these numbers comes with a confidence interval, and almost none says exactly how the answer was read from the model. Those two details are the subject of the next three sections.

## 3.4 How a multiple-choice question is really scored

"The model scored 47.5% on MMLU" sounds like a fact about the model. It is a fact about the model *and* a procedure. A multiple-choice question can be put to a language model in several reasonable ways, and they give different answers. This section shows the three main ways on one real question, then measures how much they matter on 400 questions.

### Three ways to read an answer

> [!DEFINITION] Log-likelihood scoring
> Instead of letting the model write, compute the probability the model assigns to each possible answer as a continuation of the prompt, and pick the most probable one. No text is generated, so it is fast and deterministic, and the model's answer can never be "unreadable".

> [!DEFINITION] Generative scoring
> Let the model write a reply (usually greedily), then extract the answer from the text with a rule, for example "the first letter A to D" or "the number after ####". Closer to how people use the model, but the score now depends on the extraction rule as well.

**1. The probability of the letter.** Put the question and the lettered options in the prompt, end with "Answer:", and compare the probabilities of the four next tokens " A", " B", " C", " D". This is how the MMLU paper did it.

> [!PAPER] Hendrycks et al. (2020), MMLU · Section 4.1 · page 6
> [![A paragraph from the MMLU paper, highlighted: all prompts end with Answer:, the model then produces probabilities for the tokens A, B, C and D, and we treat the highest probability option as the prediction. Few-shot evaluation adds up to 5 demonstration examples with answers, using a dev set with 5 fixed examples per subject](/img/training/ch3-mmlu-method.png)](/img/training/ch3-mmlu-method.png)
>
> **Context:** the experimental setup of the original paper.
>
> **What it says:** "All prompts end with 'Answer: '. The model then produces probabilities for the tokens 'A,' 'B,' 'C,' and 'D,' and we treat the highest probability option as the prediction." Few-shot prompts add up to 5 solved examples from the same subject.
>
> **Why it matters:** the model only has to rank four tokens. It never has to *produce* a well-formed answer, so a base model that would otherwise ramble can still be scored.

**2. The probability of the option text (cloze).** Do not show the options at all. Prompt with "Question: ... Answer:" and compute the total log-probability of each option's *text* as the continuation. The option the model finds most natural wins. This is how GPT-3 scored many benchmarks, and how HellaSwag is usually scored. One problem: longer options have more tokens, each with a log-probability below zero, so their total is lower. A common fix divides by the length of the option in characters or tokens (**length normalisation**).

> [!DEFINITION] Cloze-style scoring
> Scoring each answer option by how likely its text is as a continuation of the question, without listing the options in the prompt. Named after cloze tests, where a reader fills in a missing word.

**3. Generate and parse.** Show the question and options, let the model write, and take the first standalone letter A to D in what it writes. This is how you would use a chat model in practice, and it is the only option for closed models that do not expose probabilities.

<figure class="fig"><svg viewBox="0 0 760 325" role="img" aria-label="One MMLU question and three ways to score the base model on it: the probabilities of the letter tokens after "Answer:", the log-probabilities of each option text, and the letter found in a generated answer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Three ways to score the same multiple-choice question (Qwen2.5-0.5B base)</text><text class="t-tick" x="20.0" y="48.0" text-anchor="start">Statement 1 | Every permutation is a cycle. Statement 2 | Every cycle is a permutation.</text><text class="t-tick" x="20.0" y="69.0" text-anchor="start">A. True, True</text><text class="t-tick" x="380.0" y="69.0" text-anchor="start">B. False, False</text><text class="t-tick" x="20.0" y="86.0" text-anchor="start">C. True, False</text><text class="t-strong t-tick" x="380.0" y="86.0" text-anchor="start">D. False, True   (correct)</text><rect class="box-ghost" x="20.0" y="115.0" width="236.0" height="196.0" rx="10"/><text class="t-note" x="32.0" y="137.0" text-anchor="start">1. probability of the letter</text><rect class="box-ghost" x="266.0" y="115.0" width="236.0" height="196.0" rx="10"/><text class="t-note" x="278.0" y="137.0" text-anchor="start">2. log-probability of the text</text><rect class="box-ghost" x="512.0" y="115.0" width="236.0" height="196.0" rx="10"/><text class="t-note" x="524.0" y="137.0" text-anchor="start">3. let it write, read the letter</text><text class="t-tick" x="32.0" y="169.0" text-anchor="start">" A"</text><g class="mark"><title>P(" A") = 0.374</title><rect class="s1" x="70.0" y="155.0" width="130.0" height="18.0" rx="3"/></g><text class="t-val" x="204.0" y="169.0" text-anchor="start">0.374</text><text class="t-tick" x="32.0" y="199.0" text-anchor="start">" B"</text><g class="mark"><title>P(" B") = 0.258</title><rect class="s1" x="70.0" y="185.0" width="89.6" height="18.0" rx="3"/></g><text class="t-val" x="163.6" y="199.0" text-anchor="start">0.258</text><text class="t-tick" x="32.0" y="229.0" text-anchor="start">" C"</text><g class="mark"><title>P(" C") = 0.206</title><rect class="s1" x="70.0" y="215.0" width="71.6" height="18.0" rx="3"/></g><text class="t-val" x="145.6" y="229.0" text-anchor="start">0.206</text><text class="t-tick" x="32.0" y="259.0" text-anchor="start">" D"</text><g class="mark"><title>P(" D") = 0.158</title><rect class="s3" x="70.0" y="245.0" width="54.8" height="18.0" rx="3"/></g><text class="t-val" x="128.8" y="259.0" text-anchor="start">0.158</text><text class="t-muted" x="32.0" y="293.0" text-anchor="start">pick: A  (letters hold 99.7%)</text><text class="t-tick" x="278.0" y="169.0" text-anchor="start">A</text><g class="mark"><title>log P(text A) = -19.57</title><rect class="s1" x="296.0" y="155.0" width="27.8" height="18.0" rx="3"/></g><text class="t-val" x="327.8" y="169.0" text-anchor="start">-19.57</text><text class="t-tick" x="278.0" y="199.0" text-anchor="start">B</text><g class="mark"><title>log P(text B) = -16.25</title><rect class="s1" x="296.0" y="185.0" width="120.0" height="18.0" rx="3"/></g><text class="t-val" x="420.0" y="199.0" text-anchor="start">-16.25</text><text class="t-tick" x="278.0" y="229.0" text-anchor="start">C</text><g class="mark"><title>log P(text C) = -16.72</title><rect class="s1" x="296.0" y="215.0" width="107.0" height="18.0" rx="3"/></g><text class="t-val" x="407.0" y="229.0" text-anchor="start">-16.72</text><text class="t-tick" x="278.0" y="259.0" text-anchor="start">D</text><g class="mark"><title>log P(text D) = -16.82</title><rect class="s3" x="296.0" y="245.0" width="104.1" height="18.0" rx="3"/></g><text class="t-val" x="404.1" y="259.0" text-anchor="start">-16.82</text><text class="t-muted" x="278.0" y="293.0" text-anchor="start">pick: B  (sum of log-probs; higher wins)</text><text class="t-mono t-tick" x="524.0" y="165.0" text-anchor="start">" A /  / The following are </text><text class="t-mono t-tick" x="524.0" y="182.0" text-anchor="start">multiple choice questions </text><text class="t-mono t-tick" x="524.0" y="199.0" text-anchor="start">(with answers) about abstr</text><text class="t-mono t-tick" x="524.0" y="216.0" text-anchor="start">act algebra. /  / Statemen</text><text class="t-muted" x="524.0" y="293.0" text-anchor="start">pick: A  (first letter A to D)</text></svg><figcaption>One real MMLU question scored three ways on Qwen2.5-0.5B base: the probabilities of the four letter tokens after "Answer:", the summed log-probabilities of the four option texts, and the letter found in the model's own text. The correct option is marked. ch3_mc_demo.py.</figcaption></figure>

The question in the figure is the first one in our 400-question sample, outside the two mathematics subjects, with a short question and short options (`ch3_mc_demo.py` picks it by that fixed rule, not because it is interesting). It is an abstract algebra item: "Statement 1 | Every permutation is a cycle. Statement 2 | Every cycle is a permutation." The right answer is D (False, True). Here is everything the script prints for both models:

```text
-- base --
1. letter probabilities after "Answer:":  P(" A") = 0.374  P(" B") = 0.258  P(" C") = 0.206  P(" D") = 0.158   (sum 0.997)
   top 5 next tokens: ' A' 0.374, ' B' 0.258, ' C' 0.206, ' D' 0.158, ' True' 0.001
2. log-probability of each option text: A -19.57  B -16.25  C -16.72  D -16.82
   per character:                       A -1.957  B -1.354  C -1.520  D -1.529
3. generated: ' A\n\nThe following are multiple choice questions (with answers) about abstract algebra.\n\nStatement 1 | The order of a per'

-- instruct --
1. letter probabilities after "Answer:":  P(" A") = 0.485  P(" B") = 0.176  P(" C") = 0.226  P(" D") = 0.112   (sum 0.999)
2. log-probability of each option text: A -22.55  B -18.75  C -19.30  D -18.03
3. generated: 'To determine whether each statement is true or false, let\'s analyze them one by one: ...'
```

Several things are visible in this one example.

- **The letter method is clean.** The four letters hold 99.7% of the base model's next-token probability, so the format is understood; the model just does not know the answer, and when unsure it leans to " A" (0.374).
- **The option-text method is ill-posed here.** Without the options listed, "True, False" is not a meaningful continuation. The base model picks B; the Instruct model happens to pick D, the right answer, by a small margin.
- **Generation depends on what follows the letter.** The base model writes " A" and carries on with the next exam question; the rule takes the A. The Instruct model starts explaining and never reaches a letter within 48 tokens: a miss for a formatting reason.


### Same benchmark, same model, different numbers

This is not a toy concern. In 2023 the Open LLM Leaderboard on Hugging Face reported a much lower MMLU score for Llama-65B than its authors had. The team found three implementations of "MMLU" in use: the original (letter probabilities), HELM's (generate, then read the letter), and EleutherAI's harness at the time (probability of the full answer text, letter plus option).

> [!PAPER] Fourrier et al. (2023), "What's going on with the Open LLM Leaderboard?", Hugging Face blog · results table
> [![A table of MMLU scores for eight open models under three implementations: HELM, Harness and Original. The highlighted row is llama-65b with 0.637, 0.488 and 0.636. Other rows include falcon-40b 0.571, 0.527, 0.558 and llama-7b 0.339, 0.342, 0.351](/img/training/ch3-hf-mmlu-table.png)](/img/training/ch3-hf-mmlu-table.png)
>
> **Context:** the end of the blog post, after the three methods are explained one by one.
>
> **What it says:** the same model on the same questions scored 63.7%, 48.8% or 63.6% depending on the implementation. Even the *ranking* of models changed: under the Harness method Falcon-40B was ahead of Llama-65B; under the other two it was behind.
>
> **Why it matters:** a benchmark name is not a measurement. Two numbers are comparable only if they were computed the same way. (The post notes that the harness implementation was updated after this comparison.)

### Few-shot and zero-shot

> [!DEFINITION] Zero-shot and few-shot prompting
> In a zero-shot prompt the model sees only the instructions and the question. In a $$k$$-shot (few-shot) prompt it first sees $$k$$ solved examples in the same format. The examples teach the *format* of the answer as much as the task.

For a base model the examples show that after "Answer:" comes a single letter and nothing else; for an Instruct model the chat template and an instruction can do the same job. MMLU is traditionally reported 5-shot, GSM8K 4- or 8-shot for base models, HumanEval 0-shot.

### Prompt sensitivity

How much can harmless-looking format changes move a score? Sclar et al. (2023) measured it systematically.

> [!PAPER] Sclar et al. (2023), Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design · Abstract · page 1
> [![The abstract, highlighted: several widely used open-source LLMs are extremely sensitive to subtle changes in prompt formatting in few-shot settings, with performance differences of up to 76 accuracy points when evaluated using LLaMA-2-13B](/img/training/ch3-sclar-abstract.png)](/img/training/ch3-sclar-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** changing only the format (separators, spacing, casing of "Answer:") of a few-shot prompt gave "performance differences of up to 76 accuracy points" for LLaMA-2-13B on one task, and about 10 points on average over 50+ tasks.
>
> **Why it matters:** a one-format evaluation can land anywhere in that range. Reporting the spread over several formats is more honest than reporting one number.

Our MMLU script tries a few format variants itself (the same questions, letter-probability scoring, zero-shot):

- **plain**: the MMLU layout, `A. option` on each line, ending in `Answer:`;
- **paren**: `(A) option` instead of `A. option`;
- **qa**: `Question: ...` / `Choices:` / `A: option`;
- **inline**: all options on one line, ending in `The correct answer is`;
- **no header**: plain, without the line "The following are multiple choice questions (with answers) about ...".

It also tests something more direct: **moving the right answer**. Each question is shown four times, with the correct option moved to A, then B, then C, then D (the wrong options keep their order). A model that knows the answer should not care where it is.

```text
                          base     Instruct
plain (letter-0shot)      48.0%    47.0%
paren  "(A) option"       48.8%    44.5%
qa     "Question/Choices" 47.0%    47.2%
inline "The correct answer is"  39.0%    42.5%
no header line            48.2%    49.2%
```

Four of the five formats land within about 2 points of each other for the base model, which is inside the noise for 400 questions (Section 3.6). The "inline" format, with all options on one line and a different ending, costs 9 points. One look at the probabilities shows why: after "The correct answer is" the base model puts only 77% of its next-token probability on the four letters (against 99.6% after "Answer:"), so the format itself is less clear to it. For the Instruct model the "paren" format alone costs 2.5 points and "inline" 4.5.

Moving the right answer is more revealing:

```text
                          base     Instruct
right answer always at A  46.2%    56.5%
right answer always at B  49.5%    49.2%
right answer always at C  45.8%    47.5%
right answer always at D  45.8%    34.8%
```

<figure class="fig"><svg viewBox="0 0 760 386" role="img" aria-label="For the base and the Instruct model: accuracy when the right answer is always placed at A, B, C or D, and how often each letter is predicted in the original order."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Move the right answer to A, B, C or D: does the score change?</text><text class="t-muted" x="20.0" y="44.0" text-anchor="start">same 400 questions, 0-shot letter probability; left: accuracy, right: how often each letter is predicted</text><text class="t-note" x="20.0" y="80.0" text-anchor="start">base</text><text class="t-tick" x="40.0" y="106.0" text-anchor="start">right answer at A</text><g class="mark"><title>base, answer at A: 46.2%</title><rect class="s1" x="160.0" y="92.0" width="101.8" height="18.0" rx="3"/></g><text class="t-val" x="267.8" y="106.0" text-anchor="start">46.2%</text><text class="t-tick" x="40.0" y="134.0" text-anchor="start">right answer at B</text><g class="mark"><title>base, answer at B: 49.5%</title><rect class="s1" x="160.0" y="120.0" width="108.9" height="18.0" rx="3"/></g><text class="t-val" x="274.9" y="134.0" text-anchor="start">49.5%</text><text class="t-tick" x="40.0" y="162.0" text-anchor="start">right answer at C</text><g class="mark"><title>base, answer at C: 45.8%</title><rect class="s1" x="160.0" y="148.0" width="100.7" height="18.0" rx="3"/></g><text class="t-val" x="266.6" y="162.0" text-anchor="start">45.8%</text><text class="t-tick" x="40.0" y="190.0" text-anchor="start">right answer at D</text><g class="mark"><title>base, answer at D: 45.8%</title><rect class="s1" x="160.0" y="176.0" width="100.7" height="18.0" rx="3"/></g><text class="t-val" x="266.6" y="190.0" text-anchor="start">45.8%</text><text class="t-tick" x="470.0" y="106.0" text-anchor="start">predicts A</text><g class="mark"><title>base predicts A: 27.8%</title><rect class="s1" x="550.0" y="92.0" width="83.3" height="18.0" rx="3"/></g><text class="t-val" x="639.2" y="106.0" text-anchor="start">28%</text><text class="t-tick" x="470.0" y="134.0" text-anchor="start">predicts B</text><g class="mark"><title>base predicts B: 25.0%</title><rect class="s1" x="550.0" y="120.0" width="75.0" height="18.0" rx="3"/></g><text class="t-val" x="631.0" y="134.0" text-anchor="start">25%</text><text class="t-tick" x="470.0" y="162.0" text-anchor="start">predicts C</text><g class="mark"><title>base predicts C: 24.2%</title><rect class="s1" x="550.0" y="148.0" width="72.8" height="18.0" rx="3"/></g><text class="t-val" x="628.8" y="162.0" text-anchor="start">24%</text><text class="t-tick" x="470.0" y="190.0" text-anchor="start">predicts D</text><g class="mark"><title>base predicts D: 23.0%</title><rect class="s1" x="550.0" y="176.0" width="69.0" height="18.0" rx="3"/></g><text class="t-val" x="625.0" y="190.0" text-anchor="start">23%</text><text class="t-note" x="20.0" y="230.0" text-anchor="start">Instruct</text><text class="t-tick" x="40.0" y="256.0" text-anchor="start">right answer at A</text><g class="mark"><title>Instruct, answer at A: 56.5%</title><rect class="s3" x="160.0" y="242.0" width="124.3" height="18.0" rx="3"/></g><text class="t-val" x="290.3" y="256.0" text-anchor="start">56.5%</text><text class="t-tick" x="40.0" y="284.0" text-anchor="start">right answer at B</text><g class="mark"><title>Instruct, answer at B: 49.2%</title><rect class="s3" x="160.0" y="270.0" width="108.3" height="18.0" rx="3"/></g><text class="t-val" x="274.4" y="284.0" text-anchor="start">49.2%</text><text class="t-tick" x="40.0" y="312.0" text-anchor="start">right answer at C</text><g class="mark"><title>Instruct, answer at C: 47.5%</title><rect class="s3" x="160.0" y="298.0" width="104.5" height="18.0" rx="3"/></g><text class="t-val" x="270.5" y="312.0" text-anchor="start">47.5%</text><text class="t-tick" x="40.0" y="340.0" text-anchor="start">right answer at D</text><g class="mark"><title>Instruct, answer at D: 34.8%</title><rect class="s3" x="160.0" y="326.0" width="76.4" height="18.0" rx="3"/></g><text class="t-val" x="242.4" y="340.0" text-anchor="start">34.8%</text><text class="t-tick" x="470.0" y="256.0" text-anchor="start">predicts A</text><g class="mark"><title>Instruct predicts A: 35.5%</title><rect class="s3" x="550.0" y="242.0" width="106.5" height="18.0" rx="3"/></g><text class="t-val" x="662.5" y="256.0" text-anchor="start">36%</text><text class="t-tick" x="470.0" y="284.0" text-anchor="start">predicts B</text><g class="mark"><title>Instruct predicts B: 23.8%</title><rect class="s3" x="550.0" y="270.0" width="71.2" height="18.0" rx="3"/></g><text class="t-val" x="627.2" y="284.0" text-anchor="start">24%</text><text class="t-tick" x="470.0" y="312.0" text-anchor="start">predicts C</text><g class="mark"><title>Instruct predicts C: 26.2%</title><rect class="s3" x="550.0" y="298.0" width="78.8" height="18.0" rx="3"/></g><text class="t-val" x="634.8" y="312.0" text-anchor="start">26%</text><text class="t-tick" x="470.0" y="340.0" text-anchor="start">predicts D</text><g class="mark"><title>Instruct predicts D: 14.5%</title><rect class="s3" x="550.0" y="326.0" width="43.5" height="18.0" rx="3"/></g><text class="t-val" x="599.5" y="340.0" text-anchor="start">14%</text><text class="t-muted" x="20.0" y="374.0" text-anchor="start">for comparison, in the original order the right letter is A 24%, B 21%, C 29%, D 26%</text></svg><figcaption>The same 400 questions with the right answer moved to each position. Left: accuracy. Right: how often each letter is predicted in the original order. The base model hardly cares where the answer is; the Instruct model is 21.7 points more accurate when it is at A than at D, because it predicts A far more often and D far less often than they occur.</figcaption></figure>

The base model hardly cares where the answer is (A minus D: +0.5 points, interval -4.2 to +5.5). The Instruct model, compared with its 47.0% in the original order, gains 9.5 points when the answer is always A and loses 12.2 when it is always D: a 21.7-point gap (interval +16.8 to +27.0). In the original order it predicts A for 142 questions and D for only 58, although A is right for 96 and D for 103; the base model's predictions (111, 100, 97, 92) are much more even. Post-training gave this model a lean toward the first option, like the position bias of model judges in Section 3.10, which is why careful evaluations sometimes shuffle the options several times and average.

## 3.5 Real evaluations: MMLU and GSM8K on Qwen2.5-0.5B

Now we put everything together on the two models from Chapter 1, and compare our numbers with the ones the Qwen team published.

> [!PAPER] Qwen Team (2024), Qwen2.5 Technical Report · Table 5 · page 10
> [![Table 5 of the Qwen2.5 report: benchmark scores of the smaller base models Qwen2-0.5B, Qwen2.5-0.5B, Qwen2-1.5B, Qwen2.5-1.5B, Gemma2-2.6B and Qwen2.5-3B on general, maths and science, coding and multilingual tasks. Highlighted for Qwen2.5-0.5B: MMLU 47.5, GSM8K 41.6, HumanEval 30.5](/img/training/ch3-qwen25-table5.png)](/img/training/ch3-qwen25-table5.png)
>
> **Context:** the base-model results for the small models.
>
> **What it says:** Qwen2.5-0.5B base scores 47.5 on MMLU, 41.6 on GSM8K and 30.5 on HumanEval. The report's Table 10 gives the Instruct model 49.6 on GSM8K and 35.4 on HumanEval, and, notably, 24.1 on MMLU-redux against 45.1 for the base model.
>
> **Why it matters:** these are the numbers to check ours against. The table does not say how each answer was read; for that you need the evaluation code.

### MMLU, 400 questions

`ch3_mmlu.py` draws 400 questions at random (seed 0) from the 14,042 MMLU test questions, covering 56 of the 57 subjects, and scores both models every way described in Section 3.4. `ch3_stats.py` adds 95% confidence intervals by bootstrap (explained in the next section).

<figure class="fig"><svg viewBox="0 0 760 384" role="img" aria-label="Accuracy of the base and Instruct models on 400 MMLU questions under six scoring methods, each with a 95% bootstrap interval, with chance (25%) and the reported 47.5% marked."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">MMLU, 400 questions: accuracy with 95% bootstrap intervals</text><circle class="s1" cx="26.0" cy="46.0" r="5"/><text class="t-tick" x="36.0" y="50.0" text-anchor="start">base</text><circle class="s3" cx="96.0" cy="46.0" r="5"/><text class="t-tick" x="106.0" y="50.0" text-anchor="start">Instruct</text><text class="t-tick" x="20.0" y="86.0" text-anchor="start">5-shot, letter probability</text><line class="axis" x1="555.5" y1="76.0" x2="657.3" y2="76.0"/><line class="axis" x1="555.5" y1="71.0" x2="555.5" y2="81.0"/><line class="axis" x1="657.3" y1="71.0" x2="657.3" y2="81.0"/><g class="mark"><title>base, 5-shot, letter probability: 49.0% (44.2 to 54.0)</title><circle class="s1" cx="605.1" cy="76.0" r="5"/></g><text class="t-val" x="665.3" y="80.0" text-anchor="start">49.0%</text><line class="axis" x1="558.1" y1="90.0" x2="659.9" y2="90.0"/><line class="axis" x1="558.1" y1="85.0" x2="558.1" y2="95.0"/><line class="axis" x1="659.9" y1="85.0" x2="659.9" y2="95.0"/><g class="mark"><title>instruct, 5-shot, letter probability: 49.2% (44.5 to 54.2)</title><circle class="s3" cx="607.7" cy="90.0" r="5"/></g><text class="t-val" x="667.9" y="94.0" text-anchor="start">49.2%</text><text class="t-tick" x="20.0" y="126.0" text-anchor="start">0-shot, letter probability</text><line class="axis" x1="545.1" y1="116.0" x2="644.3" y2="116.0"/><line class="axis" x1="545.1" y1="111.0" x2="545.1" y2="121.0"/><line class="axis" x1="644.3" y1="111.0" x2="644.3" y2="121.0"/><g class="mark"><title>base, 0-shot, letter probability: 48.0% (43.2 to 52.8)</title><circle class="s1" cx="594.7" cy="116.0" r="5"/></g><text class="t-val" x="652.3" y="120.0" text-anchor="start">48.0%</text><line class="axis" x1="534.6" y1="130.0" x2="636.4" y2="130.0"/><line class="axis" x1="534.6" y1="125.0" x2="534.6" y2="135.0"/><line class="axis" x1="636.4" y1="125.0" x2="636.4" y2="135.0"/><g class="mark"><title>instruct, 0-shot, letter probability: 47.0% (42.2 to 52.0)</title><circle class="s3" cx="584.2" cy="130.0" r="5"/></g><text class="t-val" x="644.4" y="134.0" text-anchor="start">47.0%</text><text class="t-tick" x="20.0" y="166.0" text-anchor="start">0-shot, option text</text><line class="axis" x1="385.8" y1="156.0" x2="482.4" y2="156.0"/><line class="axis" x1="385.8" y1="151.0" x2="385.8" y2="161.0"/><line class="axis" x1="482.4" y1="151.0" x2="482.4" y2="161.0"/><g class="mark"><title>base, 0-shot, option text: 32.5% (28.0 to 37.2)</title><circle class="s1" cx="432.8" cy="156.0" r="5"/></g><text class="t-val" x="490.4" y="160.0" text-anchor="start">32.5%</text><line class="axis" x1="409.3" y1="170.0" x2="505.9" y2="170.0"/><line class="axis" x1="409.3" y1="165.0" x2="409.3" y2="175.0"/><line class="axis" x1="505.9" y1="165.0" x2="505.9" y2="175.0"/><g class="mark"><title>instruct, 0-shot, option text: 34.8% (30.2 to 39.5)</title><circle class="s3" cx="456.3" cy="170.0" r="5"/></g><text class="t-val" x="513.9" y="174.0" text-anchor="start">34.8%</text><text class="t-tick" x="20.0" y="206.0" text-anchor="start">0-shot, option text per character</text><line class="axis" x1="391.0" y1="196.0" x2="487.6" y2="196.0"/><line class="axis" x1="391.0" y1="191.0" x2="391.0" y2="201.0"/><line class="axis" x1="487.6" y1="191.0" x2="487.6" y2="201.0"/><g class="mark"><title>base, 0-shot, option text per character: 33.0% (28.5 to 37.8)</title><circle class="s1" cx="438.0" cy="196.0" r="5"/></g><text class="t-val" x="495.6" y="200.0" text-anchor="start">33.0%</text><line class="axis" x1="396.2" y1="210.0" x2="492.8" y2="210.0"/><line class="axis" x1="396.2" y1="205.0" x2="396.2" y2="215.0"/><line class="axis" x1="492.8" y1="205.0" x2="492.8" y2="215.0"/><g class="mark"><title>instruct, 0-shot, option text per character: 33.5% (29.0 to 38.2)</title><circle class="s3" cx="443.2" cy="210.0" r="5"/></g><text class="t-val" x="500.8" y="214.0" text-anchor="start">33.5%</text><text class="t-tick" x="20.0" y="246.0" text-anchor="start">chat template, letter probability</text><line class="axis" x1="532.0" y1="250.0" x2="636.4" y2="250.0"/><line class="axis" x1="532.0" y1="245.0" x2="532.0" y2="255.0"/><line class="axis" x1="636.4" y1="245.0" x2="636.4" y2="255.0"/><g class="mark"><title>instruct, chat template, letter probability: 47.0% (42.0 to 52.0)</title><circle class="s3" cx="584.2" cy="250.0" r="5"/></g><text class="t-val" x="644.4" y="254.0" text-anchor="start">47.0%</text><text class="t-tick" x="20.0" y="286.0" text-anchor="start">generate 48 tokens, read the letter</text><line class="axis" x1="552.9" y1="276.0" x2="654.7" y2="276.0"/><line class="axis" x1="552.9" y1="271.0" x2="552.9" y2="281.0"/><line class="axis" x1="654.7" y1="271.0" x2="654.7" y2="281.0"/><g class="mark"><title>base, generate 48 tokens, read the letter: 49.0% (44.0 to 53.8)</title><circle class="s1" cx="605.1" cy="276.0" r="5"/></g><text class="t-val" x="662.7" y="280.0" text-anchor="start">49.0%</text><line class="axis" x1="451.1" y1="290.0" x2="550.3" y2="290.0"/><line class="axis" x1="451.1" y1="285.0" x2="451.1" y2="295.0"/><line class="axis" x1="550.3" y1="285.0" x2="550.3" y2="295.0"/><g class="mark"><title>instruct, generate 48 tokens, read the letter: 39.0% (34.2 to 43.8)</title><circle class="s3" cx="500.7" cy="290.0" r="5"/></g><text class="t-val" x="558.3" y="294.0" text-anchor="start">39.0%</text><text class="t-tick" x="20.0" y="326.0" text-anchor="start">generate 256 tokens, read the letter</text><line class="axis" x1="479.7" y1="330.0" x2="579.0" y2="330.0"/><line class="axis" x1="479.7" y1="325.0" x2="479.7" y2="335.0"/><line class="axis" x1="579.0" y1="325.0" x2="579.0" y2="335.0"/><g class="mark"><title>instruct, generate 256 tokens, read the letter: 41.5% (37.0 to 46.5)</title><circle class="s3" cx="526.8" cy="330.0" r="5"/></g><text class="t-val" x="587.0" y="334.0" text-anchor="start">41.5%</text><line class="grid" x1="354.4" y1="62" x2="354.4" y2="354"/><text class="t-muted" x="354.4" y="58.0" text-anchor="middle">chance 25%</text><line class="grid" x1="589.4" y1="62" x2="589.4" y2="354"/><text class="t-muted" x="589.4" y="58.0" text-anchor="middle">reported 47.5% (base, 5-shot)</text><line class="axis" x1="250.0" y1="354.0" x2="720.0" y2="354.0"/><text class="t-tick" x="250.0" y="372.0" text-anchor="middle">15%</text><text class="t-tick" x="302.2" y="372.0" text-anchor="middle">20%</text><text class="t-tick" x="406.7" y="372.0" text-anchor="middle">30%</text><text class="t-tick" x="511.1" y="372.0" text-anchor="middle">40%</text><text class="t-tick" x="615.6" y="372.0" text-anchor="middle">50%</text><text class="t-tick" x="720.0" y="372.0" text-anchor="middle">60%</text></svg><figcaption>Accuracy on the same 400 MMLU questions under seven ways of scoring, with 95% bootstrap intervals. Chance is 25%; the Qwen report's 47.5% for the base model (5-shot, full test set) is marked.</figcaption></figure>

```text
== MMLU: accuracy and 95% bootstrap interval ==
base     letter-5shot        49.0%  [ 44.2,  54.0]
base     letter-0shot        48.0%  [ 43.2,  52.8]
base     cloze               32.5%  [ 28.0,  37.2]
base     cloze-norm          33.0%  [ 28.5,  37.8]
base     generate            49.0%  [ 44.0,  53.8]
instruct letter-5shot        49.2%  [ 44.5,  54.2]
instruct letter-0shot        47.0%  [ 42.2,  52.0]
instruct chat-letter         47.0%  [ 42.0,  52.0]
instruct cloze               34.8%  [ 30.2,  39.5]
instruct cloze-norm          33.5%  [ 29.0,  38.2]
instruct generate            39.0%  [ 34.2,  43.8]
```

What to read from this:

**Our 5-shot number matches the report.** The base model scores 49.0% with the original MMLU method, against 47.5% in the Qwen report. The difference is well inside our interval (44.2 to 54.0); with 400 questions we could not expect to match more closely. This is the reassuring case: same method, same number.

**The method moves the number more than the model does.** For the base model the scores run from 32.5% (option text) to 49.0% (letter). For the Instruct model from 33.5% to 49.2%. The cloze methods are 15 points lower for both models, because many MMLU options ("True, False", "Both A and B", "None of the above") only make sense when the other options are visible.

**On knowledge, base and Instruct are the same.** With the letter method, Instruct minus base is +0.3 points (interval -3.7 to +4.2). Post-training did not add or remove MMLU knowledge in a way 400 questions can detect, which matches Chapter 1's finding that post-training mostly changes style.

**But the generative score says Instruct is 10 points worse.** 39.0% against 49.0%, a paired difference of -10.0 points (interval -15.3 to -4.7), clearly not noise. Is the Instruct model worse at MMLU? No. Look at why answers were missed:

```text
base     loose rule  49.0% (no letter   0)   strict rule  49.0% (no letter   0)
instruct loose rule  39.0% (no letter  63)   strict rule  38.2% (no letter  79)
```

`ch3_mmlu_parse.py` re-reads the saved replies. For 63 of the 400 questions the Instruct model wrote no letter at all within its 48-token budget: asked to "answer with the letter", it began "To determine whether each statement is true or false, let's analyze them one by one..." and ran out of room. The base model, with 5 examples in its prompt, always answered with a letter first. The "loose" extraction rule (first standalone capital A to D anywhere) also has its own failure: it sometimes takes the word "A" in "A permutation is a rearrangement..." as the answer. A "strict" rule that accepts only a letter at the start or after "answer is" finds 16 fewer letters for the Instruct model.

What if we simply give it more room? `ch3_mmlu_long.py` reruns the Instruct model on the same 400 questions with a 256-token budget and reads the answer with the strict rule, falling back to the last standalone letter:

```text
instruct, generate with 256 new tokens: acc 0.415   no letter found: 40   replies that ended by themselves: 336 of 400
```

Accuracy rises from 39.0% to 41.5% (95% interval 37.0 to 46.5), and 336 replies now finish by themselves. But 40 replies still contain no letter: some explain at length and never commit to an option, some run past 256 tokens. The Instruct model still scores well below its own letter-probability score (47.0% zero-shot). The knowledge is there; turning it into a clean, parseable answer is a separate skill, and the generative score measures both.

This is the same story as the Qwen report's MMLU-redux numbers (45.1 base, 24.1 Instruct): a generative evaluation of a chat model measures its knowledge *and* whether its replies fit the extraction rule within the budget. Neither number is wrong; they answer different questions.

### GSM8K, 300 questions

GSM8K answers are free text, so it must be scored generatively. `ch3_gsm8k.py` uses 300 random test problems and the setup each model is meant for: the base model gets 4 solved training problems as examples ("Question: ... Answer: ... #### 72") and continues after the fifth "Answer:"; the Instruct model gets the question in its chat template with the instruction commonly used for Qwen models on maths, `Please reason step by step, and put your final answer within \boxed{}.` Both decode greedily, up to 320 new tokens. The final answer is then extracted in two ways:

- **strict**: only the format we asked for (`#### 72` for base, `\boxed{72}` for Instruct);
- **flexible**: the last number anywhere in the reply.

```python
# from ch3_gsm8k.py
NUM = r'-?\$?\d[\d,]*(?:\.\d+)?'                    # a number, maybe with $ and thousands commas

def strict(text, kind):
    if kind == 'base':
        m = re.search(r'####\s*(' + NUM + ')', text)
    else:
        m = re.search(r'\\boxed\{\s*(' + NUM + r')\s*\}', text)
    return norm(m.group(1)) if m else None              # norm: drop "," "$" and a final "."

def flexible(text):
    nums = re.findall(NUM, text)
    return norm(nums[-1]) if nums else None
```

```text
base     strict    35.0%  [ 29.7,  40.3]
base     flexible  36.0%  [ 30.7,  41.7]
instruct strict    35.0%  [ 29.7,  40.3]      (109 replies with no \boxed{} answer)
instruct flexible  36.7%  [ 31.3,  42.3]
instruct strict, up to 1,024 tokens  42.3%  [ 37.0,  48.0]
```

<figure class="fig"><svg viewBox="0 0 760 276" role="img" aria-label="GSM8K accuracy for the base and Instruct models under strict and flexible answer extraction, with 95% bootstrap intervals and the reported numbers marked."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">GSM8K, 300 questions: exact-match accuracy with 95% bootstrap intervals</text><text class="t-tick" x="20.0" y="85.0" text-anchor="start">base, 4-shot, last number</text><line class="axis" x1="458.4" y1="80.0" x2="526.0" y2="80.0"/><line class="axis" x1="458.4" y1="75.0" x2="458.4" y2="85.0"/><line class="axis" x1="526.0" y1="75.0" x2="526.0" y2="85.0"/><g class="mark"><title>base, 4-shot, last number: 36.0%</title><circle class="s1" cx="491.1" cy="80.0" r="5"/></g><text class="t-val" x="534.0" y="85.0" text-anchor="start">36.0%</text><text class="t-tick" x="20.0" y="119.0" text-anchor="start">base, 4-shot, "#### N" only</text><line class="axis" x1="452.2" y1="114.0" x2="517.8" y2="114.0"/><line class="axis" x1="452.2" y1="109.0" x2="452.2" y2="119.0"/><line class="axis" x1="517.8" y1="109.0" x2="517.8" y2="119.0"/><g class="mark"><title>base, 4-shot, "#### N" only: 35.0%</title><circle class="s1" cx="485.0" cy="114.0" r="5"/></g><text class="t-val" x="525.8" y="119.0" text-anchor="start">35.0%</text><text class="t-tick" x="20.0" y="153.0" text-anchor="start">Instruct, chat, last number</text><line class="axis" x1="462.5" y1="148.0" x2="530.0" y2="148.0"/><line class="axis" x1="462.5" y1="143.0" x2="462.5" y2="153.0"/><line class="axis" x1="530.0" y1="143.0" x2="530.0" y2="153.0"/><g class="mark"><title>Instruct, chat, last number: 36.7%</title><circle class="s3" cx="495.2" cy="148.0" r="5"/></g><text class="t-val" x="538.0" y="153.0" text-anchor="start">36.7%</text><text class="t-tick" x="20.0" y="187.0" text-anchor="start">Instruct, chat, \boxed{N} only</text><line class="axis" x1="452.2" y1="182.0" x2="517.8" y2="182.0"/><line class="axis" x1="452.2" y1="177.0" x2="452.2" y2="187.0"/><line class="axis" x1="517.8" y1="177.0" x2="517.8" y2="187.0"/><g class="mark"><title>Instruct, chat, \boxed{N} only: 35.0%</title><circle class="s3" cx="485.0" cy="182.0" r="5"/></g><text class="t-val" x="525.8" y="187.0" text-anchor="start">35.0%</text><text class="t-tick" x="20.0" y="221.0" text-anchor="start">same, up to 1,024 tokens</text><line class="axis" x1="497.3" y1="216.0" x2="564.9" y2="216.0"/><line class="axis" x1="497.3" y1="211.0" x2="497.3" y2="221.0"/><line class="axis" x1="564.9" y1="211.0" x2="564.9" y2="221.0"/><g class="mark"><title>same, up to 1,024 tokens: 42.3%</title><circle class="s3" cx="530.0" cy="216.0" r="5"/></g><text class="t-val" x="572.9" y="221.0" text-anchor="start">42.3%</text><line class="grid" x1="525.5" y1="66" x2="525.5" y2="240"/><line class="grid" x1="574.7" y1="66" x2="574.7" y2="240"/><text class="t-muted" x="521.5" y="62.0" text-anchor="end">reported: base 41.6%</text><text class="t-muted" x="578.7" y="62.0" text-anchor="start">Instruct 49.6%</text><line class="axis" x1="270.0" y1="240.0" x2="700.0" y2="240.0"/><text class="t-tick" x="270.0" y="258.0" text-anchor="middle">0%</text><text class="t-tick" x="331.4" y="258.0" text-anchor="middle">10%</text><text class="t-tick" x="392.9" y="258.0" text-anchor="middle">20%</text><text class="t-tick" x="454.3" y="258.0" text-anchor="middle">30%</text><text class="t-tick" x="515.7" y="258.0" text-anchor="middle">40%</text><text class="t-tick" x="577.1" y="258.0" text-anchor="middle">50%</text><text class="t-tick" x="638.6" y="258.0" text-anchor="middle">60%</text><text class="t-tick" x="700.0" y="258.0" text-anchor="middle">70%</text></svg><figcaption>GSM8K exact-match accuracy on 300 problems with 95% bootstrap intervals, for each model and extraction rule. The Qwen report's numbers (41.6% base, 49.6% Instruct) are marked. Giving the Instruct model room to finish moves it from 35.0% to 42.3%.</figcaption></figure>

With a 320-token budget the two models look identical (35.0% each). But the Instruct model failed to produce a boxed answer for 109 of 300 problems, and 106 of its replies were cut off at the budget: it writes long, careful solutions with LaTeX (923 characters on average, against 276 for the base model). `ch3_gsm8k_long.py` lets those 109 replies run to 1,024 tokens (greedy decoding is deterministic, so this is the same as letting them continue). Now the Instruct model reaches 42.3%, 6.3 points above the base model on the same problems (paired interval 0.0 to +12.7: probably a real gain, but 300 problems cannot pin it down).

Compared with the report, our base model is 5.6 points lower (36.0 against 41.6) and our Instruct model 7.3 points lower (42.3 against 49.6). The report's base number is just inside our interval (30.7 to 41.7) and its Instruct number just outside (37.0 to 48.0), and the protocols differ in ways the table does not record (which few-shot examples, the token limit, the extraction rule, the exact instruction). We cannot reproduce the published GSM8K numbers exactly, and we do not know which detail accounts for the gap.

Reading the outputs is as instructive as the numbers. Among the replies where the two extraction rules disagree, one base-model reply abandons the problem and writes "...User will you give you a question. Your task is to answer as faithfully as you can. While answering think step-by-step and justify your answer." That looks like a fragment of an instruction-tuning prompt, which the *base* model has probably seen in its pretraining data. Base models are not as untouched by chat data as their name suggests.

## 3.6 How sure are we? Confidence intervals with the bootstrap

Every accuracy in the last section came with an interval in brackets. This section explains where those intervals come from and why you should never compare two benchmark numbers without one.

We asked 400 questions out of 14,042; a different 400 would give a slightly different accuracy. The **confidence interval** says how different.

> [!DEFINITION] Confidence interval
> A range computed from the data that would contain the true value (here: the accuracy on the whole test set, or on an endless supply of similar questions) in 95% of repeated experiments. Narrow interval: the number is precise. Two models whose intervals overlap a lot cannot be told apart by this test.

### The formula way

For an accuracy, each question is a 1 (right) or 0 (wrong), and the accuracy $$p$$ is their mean. Statistics gives a simple approximation:

$$p \pm 1.96 \times \text{SE}, \qquad \text{SE} = \sqrt{\frac{p\,(1-p)}{n}}$$

where:

- $$p$$ is the observed accuracy (questions right divided by questions asked);
- $$n$$ is the number of questions;
- $$\text{SE}$$ is the **standard error**: how much $$p$$ would typically change if we repeated the experiment with fresh questions;
- 1.96 standard errors on each side cover the middle 95% of a normal distribution.

**Worked example** (from `ch3_stats.py`). The base model, 5-shot letter method, got 196 of 400 right: $$p = 0.49$$. Then $$\text{SE} = \sqrt{0.49 \times 0.51 / 400} = 0.0250$$, and the interval is $$0.49 \pm 1.96 \times 0.0250 = 0.49 \pm 0.049$$, from 44.1% to 53.9%. Almost five points either way, from 400 questions.

### The bootstrap way

The formula relies on a normal approximation and works only for simple averages. The **bootstrap** (Efron, 1979) works for almost any statistic, including win rates, differences between models, pass@k and Bradley-Terry ratings, and it needs no formula at all.

> [!DEFINITION] Bootstrap
> Estimate the uncertainty of a number by recomputing it many times on resampled data. Each resample draws $$n$$ items *with replacement* from the original $$n$$ (so some items appear twice and some not at all), and the spread of the recomputed numbers shows how much the original could have varied.

<figure class="fig"><svg viewBox="0 0 760 346" role="img" aria-label="Four steps of the bootstrap, then a histogram of 4000 resampled accuracies with the middle 95% shaded and its two ends marked."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The bootstrap: how sure are we of 49.0%?</text><rect class="box-1" x="20.0" y="40.0" width="172.0" height="58.0" rx="8"/><circle class="node on" cx="36.0" cy="57.0" r="10"/><text class="t-tick" x="36.0" y="61.0" text-anchor="middle">1</text><text class="t-tick" x="52.0" y="61.0" text-anchor="start">400 results (1 = right)</text><text class="t-muted" x="32.0" y="86.0" text-anchor="start">the real per-question scores</text><line class="edge" x1="192.0" y1="69.0" x2="204.0" y2="69.0" marker-end="url(#ah)"/><rect class="box-1" x="204.0" y="40.0" width="172.0" height="58.0" rx="8"/><circle class="node on" cx="220.0" cy="57.0" r="10"/><text class="t-tick" x="220.0" y="61.0" text-anchor="middle">2</text><text class="t-tick" x="236.0" y="61.0" text-anchor="start">resample 400</text><text class="t-muted" x="216.0" y="86.0" text-anchor="start">with replacement: some twice</text><line class="edge" x1="376.0" y1="69.0" x2="388.0" y2="69.0" marker-end="url(#ah)"/><rect class="box-1" x="388.0" y="40.0" width="172.0" height="58.0" rx="8"/><circle class="node on" cx="404.0" cy="57.0" r="10"/><text class="t-tick" x="404.0" y="61.0" text-anchor="middle">3</text><text class="t-tick" x="420.0" y="61.0" text-anchor="start">take the mean</text><text class="t-muted" x="400.0" y="86.0" text-anchor="start">one "could have been" score</text><line class="edge" x1="560.0" y1="69.0" x2="572.0" y2="69.0" marker-end="url(#ah)"/><rect class="box-2" x="572.0" y="40.0" width="172.0" height="58.0" rx="8"/><circle class="node on" cx="588.0" cy="57.0" r="10"/><text class="t-tick" x="588.0" y="61.0" text-anchor="middle">4</text><text class="t-tick" x="604.0" y="61.0" text-anchor="start">repeat 10,000 times</text><text class="t-muted" x="584.0" y="86.0" text-anchor="start">middle 95% = the interval</text><g class="mark"><title>40.2 to 40.8%: 2 resamples</title><rect class="s4" x="61.0" y="299.2" width="18.6" height="0.8" rx="2"/></g><g class="mark"><title>40.8 to 41.4%: 7 resamples</title><rect class="s4" x="81.6" y="297.2" width="18.6" height="2.8" rx="2"/></g><g class="mark"><title>41.4 to 42.0%: 4 resamples</title><rect class="s4" x="102.3" y="298.4" width="18.6" height="1.6" rx="2"/></g><g class="mark"><title>42.0 to 42.5%: 17 resamples</title><rect class="s4" x="122.9" y="293.2" width="18.6" height="6.8" rx="2"/></g><g class="mark"><title>42.5 to 43.1%: 19 resamples</title><rect class="s4" x="143.6" y="292.4" width="18.6" height="7.6" rx="2"/></g><g class="mark"><title>43.1 to 43.7%: 26 resamples</title><rect class="s4" x="164.2" y="289.6" width="18.6" height="10.4" rx="2"/></g><g class="mark"><title>43.7 to 44.3%: 54 resamples</title><rect class="s4" x="184.9" y="278.3" width="18.6" height="21.7" rx="2"/></g><g class="mark"><title>44.3 to 44.8%: 76 resamples</title><rect class="s3" x="205.5" y="269.5" width="18.6" height="30.5" rx="2"/></g><g class="mark"><title>44.8 to 45.4%: 104 resamples</title><rect class="s3" x="226.2" y="258.2" width="18.6" height="41.8" rx="2"/></g><g class="mark"><title>45.4 to 46.0%: 106 resamples</title><rect class="s3" x="246.8" y="257.4" width="18.6" height="42.6" rx="2"/></g><g class="mark"><title>46.0 to 46.5%: 269 resamples</title><rect class="s3" x="267.5" y="191.9" width="18.6" height="108.1" rx="2"/></g><g class="mark"><title>46.5 to 47.1%: 215 resamples</title><rect class="s3" x="288.1" y="213.6" width="18.6" height="86.4" rx="2"/></g><g class="mark"><title>47.1 to 47.7%: 237 resamples</title><rect class="s3" x="308.7" y="204.8" width="18.6" height="95.2" rx="2"/></g><g class="mark"><title>47.7 to 48.3%: 423 resamples</title><rect class="s3" x="329.4" y="130.0" width="18.6" height="170.0" rx="2"/></g><g class="mark"><title>48.3 to 48.8%: 300 resamples</title><rect class="s3" x="350.0" y="179.4" width="18.6" height="120.6" rx="2"/></g><g class="mark"><title>48.8 to 49.4%: 324 resamples</title><rect class="s3" x="370.7" y="169.8" width="18.6" height="130.2" rx="2"/></g><g class="mark"><title>49.4 to 50.0%: 307 resamples</title><rect class="s3" x="391.3" y="176.6" width="18.6" height="123.4" rx="2"/></g><g class="mark"><title>50.0 to 50.6%: 422 resamples</title><rect class="s3" x="412.0" y="130.4" width="18.6" height="169.6" rx="2"/></g><g class="mark"><title>50.6 to 51.1%: 252 resamples</title><rect class="s3" x="432.6" y="198.7" width="18.6" height="101.3" rx="2"/></g><g class="mark"><title>51.1 to 51.7%: 211 resamples</title><rect class="s3" x="453.3" y="215.2" width="18.6" height="84.8" rx="2"/></g><g class="mark"><title>51.7 to 52.3%: 267 resamples</title><rect class="s3" x="473.9" y="192.7" width="18.6" height="107.3" rx="2"/></g><g class="mark"><title>52.3 to 52.8%: 102 resamples</title><rect class="s3" x="494.5" y="259.0" width="18.6" height="41.0" rx="2"/></g><g class="mark"><title>52.8 to 53.4%: 88 resamples</title><rect class="s3" x="515.2" y="264.6" width="18.6" height="35.4" rx="2"/></g><g class="mark"><title>53.4 to 54.0%: 62 resamples</title><rect class="s3" x="535.8" y="275.1" width="18.6" height="24.9" rx="2"/></g><g class="mark"><title>54.0 to 54.6%: 50 resamples</title><rect class="s4" x="556.5" y="279.9" width="18.6" height="20.1" rx="2"/></g><g class="mark"><title>54.6 to 55.1%: 30 resamples</title><rect class="s4" x="577.1" y="287.9" width="18.6" height="12.1" rx="2"/></g><g class="mark"><title>55.1 to 55.7%: 7 resamples</title><rect class="s4" x="597.8" y="297.2" width="18.6" height="2.8" rx="2"/></g><g class="mark"><title>55.7 to 56.3%: 9 resamples</title><rect class="s4" x="618.4" y="296.4" width="18.6" height="3.6" rx="2"/></g><g class="mark"><title>56.3 to 56.9%: 2 resamples</title><rect class="s4" x="639.1" y="299.2" width="18.6" height="0.8" rx="2"/></g><g class="mark"><title>56.9 to 57.4%: 3 resamples</title><rect class="s4" x="659.7" y="298.8" width="18.6" height="1.2" rx="2"/></g><g class="mark"><title>57.4 to 58.0%: 5 resamples</title><rect class="s4" x="680.4" y="298.0" width="18.6" height="2.0" rx="2"/></g><line class="path" x1="195.2" y1="124" x2="195.2" y2="300"/><text class="t-tick" x="195.2" y="120.0" text-anchor="middle">2.5%: 44.0%</text><line class="path" x1="546.8" y1="124" x2="546.8" y2="300"/><text class="t-tick" x="546.8" y="120.0" text-anchor="middle">97.5%: 53.8%</text><line class="axis" x1="123.1" y1="300.0" x2="627.9" y2="300.0"/><text class="t-tick" x="123.1" y="318.0" text-anchor="middle">42%</text><text class="t-tick" x="195.2" y="318.0" text-anchor="middle">44%</text><text class="t-tick" x="267.3" y="318.0" text-anchor="middle">46%</text><text class="t-tick" x="339.4" y="318.0" text-anchor="middle">48%</text><text class="t-tick" x="411.5" y="318.0" text-anchor="middle">50%</text><text class="t-tick" x="483.7" y="318.0" text-anchor="middle">52%</text><text class="t-tick" x="555.8" y="318.0" text-anchor="middle">54%</text><text class="t-tick" x="627.9" y="318.0" text-anchor="middle">56%</text><text class="t-tick" x="375.5" y="338.0" text-anchor="middle">accuracy of each resample (base, MMLU 5-shot letter)</text></svg><figcaption>The bootstrap for the base model's 49.0% on 400 MMLU questions. Draw 400 results with replacement, take the mean, repeat 10,000 times; the histogram shows 4,000 of those means. The middle 95% (shaded) is the confidence interval, 44.0% to 53.8%.</figcaption></figure>

```python
# from ch3_stats.py
def boot(x, b=10000):
    x = np.asarray(x, float)                         # 400 values, 1 = right, 0 = wrong
    idx = rng.integers(0, len(x), (b, len(x)))       # b rows of 400 random positions (with replacement)
    means = x[idx].mean(1)                           # b resampled accuracies
    lo, hi = np.percentile(means, [2.5, 97.5])       # the middle 95%
    return float(x.mean()), float(lo), float(hi), means
```

Line by line: `idx` is a table of random question numbers, 10,000 rows of 400; picking `x[idx]` builds 10,000 "alternative test sets" made of our own 400 results; their means are 10,000 accuracies that "could have been"; the 2.5th and 97.5th percentiles cut off the most extreme 2.5% on each side.

```text
196 right out of 400:  accuracy p = 196/400 = 0.4900
standard error sqrt(p(1-p)/n) = sqrt(0.4900 x 0.5100 / 400) = 0.0250
normal-approximation 95% interval: p +- 1.96 x SE = 0.4900 +- 0.0490 = [0.4410, 0.5390]
bootstrap (10,000 resamples) 95% interval: [0.4400, 0.5375]
first five resampled accuracies: 0.4825, 0.4775, 0.5150, 0.4650, 0.4575
```

The two methods agree to within a few tenths of a point.

### Comparing two models: use a paired difference

To ask "is Instruct better than base?", do not just check whether their two intervals overlap. Both models answered the *same* 400 questions, and many questions are easy or hard for both. A **paired** bootstrap resamples *questions* and recomputes both models on each draw, so the shared difficulty cancels out:

```text
== paired differences (Instruct minus base, same questions) ==
letter-5shot    +0.3 points  [ -3.7,  +4.2]   (interval contains 0)
letter-0shot    -1.0 points  [ -5.0,  +3.0]   (interval contains 0)
cloze           +2.2 points  [ -1.0,  +5.5]   (interval contains 0)
cloze-norm      +0.5 points  [ -2.3,  +3.3]   (interval contains 0)
generate       -10.0 points  [-15.3,  -4.7]
```

The paired intervals are narrower than the individual ones (about ±4 points instead of ±5). The rule of thumb: if the interval of the *difference* contains 0, the test cannot tell the two models apart.

### How many questions do you need?

The interval shrinks with the square root of the number of questions. At 50% accuracy:

```text
n =    100: +- 9.8 points
n =    400: +- 4.9 points
n =   1000: +- 3.1 points
n =   1319: +- 2.7 points      (all of GSM8K test)
n =  14042: +- 0.8 points      (all of MMLU test)
```

So a 1-point difference on full MMLU may be real, but a 2-point difference on GSM8K's 1,319 problems probably is not, and a 5-point difference on a 100-question test tells you almost nothing. GPQA, with 448 questions, has intervals of about ±4.6 points. Remember this the next time a model report bolds the larger of two numbers that differ by one point.

> [!WARNING]
> These intervals only cover the randomness of *which questions* were asked. They do not cover the choice of prompt, extraction rule or decoding settings, which Section 3.4 showed can move a score by much more. An honest report gives both: the interval, and the protocol.

## 3.7 Code: HumanEval and pass@k

For code there is a better judge than any person or model: run it. HumanEval (Chen et al., 2021) gives the model the start of a Python file, a function signature and a docstring with a few examples, and asks it to write the body. Hidden unit tests then decide: the completion is correct only if every test passes. "Looks right" counts for nothing.

> [!PAPER] Chen et al. (2021), Codex / HumanEval · Figure 2 · page 3
> [![Three example HumanEval problems. Each shows a function signature and docstring on a white background, the model's completion on a yellow background, and the probability that a single sample from Codex-12B passes the unit tests: 0.9, 0.17 and 0.005](/img/training/ch3-humaneval-fig2.png)](/img/training/ch3-humaneval-fig2.png)
>
> **Context:** the dataset section.
>
> **What it says:** the prompt (white) is a signature plus docstring; the model writes the body (yellow). Problems range from easy (a single sample passes 90% of the time) to very hard (0.5%).
>
> **Why it matters:** **functional correctness** is an objective metric for an open-ended output. The same idea, checking an answer with a program, is what makes GSM8K and code usable as verifiable rewards for reinforcement learning (Chapter 2, Section 2.9).

### pass@k, and why the obvious estimate is wrong

Code models are often used by sampling several attempts and keeping one that works. So Chen et al. measured **pass@k**: the probability that at least one of $$k$$ sampled completions passes the tests.

> [!DEFINITION] pass@k
> For one problem, the probability that at least one of $$k$$ independent samples from the model is correct. The benchmark score is the average over problems. pass@1 is the ordinary "first try" accuracy.

The direct way to estimate it is to draw exactly $$k$$ samples per problem and check whether any passes. That estimate is very noisy. A better way is to draw more samples, $$n \ge k$$, count how many are correct, $$c$$, and ask: if I picked $$k$$ of these $$n$$ at random, what is the chance that at least one is correct?

> [!PAPER] Chen et al. (2021), Codex / HumanEval · Section 2.1, equation 1 · page 3
> [![The text explaining that computing pass@k with exactly k samples has high variance, so the authors generate n greater than or equal to k samples per task, count the correct samples c, and calculate the unbiased estimator: pass@k equals the expectation over problems of one minus the binomial coefficient n minus c choose k divided by n choose k. Highlighted: calculate the unbiased estimator](/img/training/ch3-humaneval-eq1.png)](/img/training/ch3-humaneval-eq1.png)
>
> **Context:** the definition of the metric, Section 2.1.
>
> **What it says:** "we generate $$n \ge k$$ samples per task (in this paper, we use $$n = 200$$ and $$k \le 100$$), count the number of correct samples $$c \le n$$ ... and calculate the unbiased estimator".
>
> **Why it matters:** this has been the standard way to compute pass@k ever since.

$$\text{pass@}k = \mathop{\mathbb{E}}_{\text{problems}}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right]$$

where:

- $$n$$ is the number of samples drawn for one problem (we use 10; the paper used 200);
- $$c$$ is how many of those $$n$$ pass all the unit tests;
- $$k$$ is the number of attempts allowed, $$k \le n$$;
- $$\binom{n}{k}$$ ("n choose k") is the number of ways to pick $$k$$ samples out of $$n$$;
- $$\binom{n-c}{k}$$ is the number of ways to pick $$k$$ samples that are *all wrong* (only from the $$n - c$$ failures);
- their ratio is the chance that a random set of $$k$$ contains no correct sample, so 1 minus it is the chance that it contains at least one;
- $$\mathbb{E}_{\text{problems}}$$ means: compute this for every problem, then average.

**Worked example.** One problem, $$n = 10$$ samples, $$c = 3$$ correct. For $$k = 1$$: $$\binom{7}{1} / \binom{10}{1} = 7/10$$, so pass@1 $$= 1 - 0.7 = 0.30$$, the plain fraction correct, as it should be. For $$k = 5$$: there are $$\binom{10}{5} = 252$$ ways to choose 5 samples, and $$\binom{7}{5} = 21$$ of them are all wrong, so pass@5 $$= 1 - 21/252 = 0.917$$. For $$k = 8$$: you cannot choose 8 samples from only 7 failures, so $$\binom{7}{8} = 0$$ and pass@8 $$= 1$$.

Why not just use $$1 - (1 - c/n)^k$$, "the chance that $$k$$ independent tries all fail, with success rate $$c/n$$"? Because that formula treats $$c/n$$ as the exact success rate, and then averages a curved function of a noisy estimate. The result is biased low. For $$n = 10, c = 3, k = 5$$ it gives $$1 - 0.7^5 = 0.832$$ instead of 0.917. The script prints both:

```text
== the unbiased pass@k estimator, by hand ==
n= 10 c= 3 k=  1:  unbiased pass@k = 0.3000   (naive 1-(1-c/n)^k = 0.3000)
n= 10 c= 3 k=  5:  unbiased pass@k = 0.9167   (naive 1-(1-c/n)^k = 0.8319)
n= 10 c= 3 k=  8:  unbiased pass@k = 1.0000   (naive 1-(1-c/n)^k = 0.9424)
n= 10 c= 0 k=  5:  unbiased pass@k = 0.0000   (naive 1-(1-c/n)^k = 0.0000)
n=200 c=10 k=100:  unbiased pass@k = 0.9992   (naive 1-(1-c/n)^k = 0.9941)
```

Computing $$\binom{200}{100}$$ directly gives a number with 59 digits, so the paper rewrites the ratio as a product of small fractions. Their Figure 3 is three lines of numpy.

> [!PAPER] Chen et al. (2021), Codex / HumanEval · Figure 3 · page 3
> [![A short Python function pass_at_k(n, c, k) with a docstring; it returns 1.0 if n minus c is less than k, otherwise 1.0 minus the product of 1 minus k divided by each integer from n minus c plus 1 to n. Caption: a numerically stable script for calculating an unbiased estimate of pass@k](/img/training/ch3-humaneval-fig3.png)](/img/training/ch3-humaneval-fig3.png)
>
> **Context:** right below equation 1.
>
> **What it says:** $$\binom{n-c}{k}/\binom{n}{k} = \prod_{i=n-c+1}^{n} (1 - k/i)$$, which never builds a huge number.
>
> **Why it matters:** our script uses this function unchanged.

### Our run: Qwen2.5-0.5B base on all 164 problems

`ch3_humaneval.py` feeds each of the 164 prompts to the base model, cuts the completion at the first line that starts a new top-level block (a new `def`, `class`, `print(` and so on), appends the problem's unit tests, and runs the result in a separate Python process with a 5-second limit, no network and no permission to write files outside a temporary folder (generated code is untrusted code). It does this once with greedy decoding, and then 10 times per problem with sampling at temperature 0.8.

```text
greedy: 47/164 problems pass  ->  pass@1 (greedy) = 0.287
sampled: 10 completions per problem, 328 of 1640 pass
problems by number of passing samples c: c=0: 96, c=1: 15, c=2: 9, c=3: 5, c=4: 5, c=5: 10, c=6: 2, c=7: 3, c=8: 4, c=9: 5, c=10: 10
pass@1  = 0.200   (the biased shortcut 1-(1-c/n)^k would say 0.200)
pass@2  = 0.266   (the biased shortcut 1-(1-c/n)^k would say 0.259)
pass@5  = 0.353   (the biased shortcut 1-(1-c/n)^k would say 0.333)
pass@10 = 0.415   (the biased shortcut 1-(1-c/n)^k would say 0.376)
```

<figure class="fig"><svg viewBox="0 0 760 320" role="img" aria-label="Left: pass@k for k from 1 to 10, computed with the unbiased estimator and with the biased shortcut. Right: how many problems had 0 to 10 passing samples."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">HumanEval, Qwen2.5-0.5B base, 10 samples per problem</text><line class="grid" x1="60" y1="270.0" x2="440" y2="270.0"/><text class="t-tick" x="52.0" y="274.0" text-anchor="end">0%</text><line class="grid" x1="60" y1="215.0" x2="440" y2="215.0"/><text class="t-tick" x="52.0" y="219.0" text-anchor="end">20%</text><line class="grid" x1="60" y1="160.0" x2="440" y2="160.0"/><text class="t-tick" x="52.0" y="164.0" text-anchor="end">40%</text><line class="grid" x1="60" y1="105.0" x2="440" y2="105.0"/><text class="t-tick" x="52.0" y="109.0" text-anchor="end">60%</text><line class="grid" x1="60" y1="50.0" x2="440" y2="50.0"/><text class="t-tick" x="52.0" y="54.0" text-anchor="end">80%</text><text class="t-tick" x="60.0" y="288.0" text-anchor="middle">1</text><text class="t-tick" x="102.2" y="288.0" text-anchor="middle">2</text><text class="t-tick" x="144.4" y="288.0" text-anchor="middle">3</text><text class="t-tick" x="186.7" y="288.0" text-anchor="middle">4</text><text class="t-tick" x="228.9" y="288.0" text-anchor="middle">5</text><text class="t-tick" x="271.1" y="288.0" text-anchor="middle">6</text><text class="t-tick" x="313.3" y="288.0" text-anchor="middle">7</text><text class="t-tick" x="355.6" y="288.0" text-anchor="middle">8</text><text class="t-tick" x="397.8" y="288.0" text-anchor="middle">9</text><text class="t-tick" x="440.0" y="288.0" text-anchor="middle">10</text><text class="t-tick" x="250.0" y="308.0" text-anchor="middle">k (attempts allowed)</text><polyline class="l3" points="60.0,215.0 102.2,196.9 144.4,186.0 186.7,178.5 228.9,172.9 271.1,168.4 313.3,164.6 355.6,161.3 397.8,158.5 440.0,156.0"/><g class="mark"><title>unbiased estimator, k=1: 20.0%</title><circle class="s3" cx="60.0" cy="215.0" r="4"/></g><g class="mark"><title>unbiased estimator, k=2: 26.6%</title><circle class="s3" cx="102.2" cy="196.9" r="4"/></g><g class="mark"><title>unbiased estimator, k=3: 30.5%</title><circle class="s3" cx="144.4" cy="186.0" r="4"/></g><g class="mark"><title>unbiased estimator, k=4: 33.3%</title><circle class="s3" cx="186.7" cy="178.5" r="4"/></g><g class="mark"><title>unbiased estimator, k=5: 35.3%</title><circle class="s3" cx="228.9" cy="172.9" r="4"/></g><g class="mark"><title>unbiased estimator, k=6: 37.0%</title><circle class="s3" cx="271.1" cy="168.4" r="4"/></g><g class="mark"><title>unbiased estimator, k=7: 38.3%</title><circle class="s3" cx="313.3" cy="164.6" r="4"/></g><g class="mark"><title>unbiased estimator, k=8: 39.5%</title><circle class="s3" cx="355.6" cy="161.3" r="4"/></g><g class="mark"><title>unbiased estimator, k=9: 40.5%</title><circle class="s3" cx="397.8" cy="158.5" r="4"/></g><g class="mark"><title>unbiased estimator, k=10: 41.5%</title><circle class="s3" cx="440.0" cy="156.0" r="4"/></g><polyline class="l2" points="60.0,215.0 102.2,198.7 144.4,189.2 186.7,182.9 228.9,178.4 271.1,175.0 313.3,172.2 355.6,170.0 397.8,168.2 440.0,166.7"/><g class="mark"><title>1 - (1 - c/n)^k (biased), k=1: 20.0%</title><circle class="s2" cx="60.0" cy="215.0" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=2: 25.9%</title><circle class="s2" cx="102.2" cy="198.7" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=3: 29.4%</title><circle class="s2" cx="144.4" cy="189.2" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=4: 31.7%</title><circle class="s2" cx="186.7" cy="182.9" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=5: 33.3%</title><circle class="s2" cx="228.9" cy="178.4" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=6: 34.6%</title><circle class="s2" cx="271.1" cy="175.0" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=7: 35.6%</title><circle class="s2" cx="313.3" cy="172.2" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=8: 36.4%</title><circle class="s2" cx="355.6" cy="170.0" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=9: 37.0%</title><circle class="s2" cx="397.8" cy="168.2" r="4"/></g><g class="mark"><title>1 - (1 - c/n)^k (biased), k=10: 37.6%</title><circle class="s2" cx="440.0" cy="166.7" r="4"/></g><text class="t-val" x="448.0" y="160.0" text-anchor="start">41.5%</text><text class="t-val" x="68.0" y="229.0" text-anchor="start">pass@1 20.0%</text><rect class="s3" x="70.0" y="56.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="86.0" y="65.0" text-anchor="start">unbiased estimator</text><rect class="s2" x="70.0" y="74.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="86.0" y="83.0" text-anchor="start">1 - (1 - c/n)^k, biased low</text><text class="t-muted" x="500.0" y="52.0" text-anchor="start">problems by number of passing samples c</text><g class="mark"><title>c = 0: 96 problems</title><rect class="s4" x="501.0" y="94.0" width="18.9" height="176.0" rx="2"/></g><text class="t-tick" x="510.5" y="288.0" text-anchor="middle">0</text><text class="t-tick" x="510.5" y="90.0" text-anchor="middle">96</text><g class="mark"><title>c = 1: 15 problems</title><rect class="s1" x="521.9" y="242.5" width="18.9" height="27.5" rx="2"/></g><text class="t-tick" x="531.4" y="288.0" text-anchor="middle">1</text><text class="t-tick" x="531.4" y="238.5" text-anchor="middle">15</text><g class="mark"><title>c = 2: 9 problems</title><rect class="s1" x="542.8" y="253.5" width="18.9" height="16.5" rx="2"/></g><text class="t-tick" x="552.3" y="288.0" text-anchor="middle">2</text><text class="t-tick" x="552.3" y="249.5" text-anchor="middle">9</text><g class="mark"><title>c = 3: 5 problems</title><rect class="s1" x="563.7" y="260.8" width="18.9" height="9.2" rx="2"/></g><text class="t-tick" x="573.2" y="288.0" text-anchor="middle">3</text><text class="t-tick" x="573.2" y="256.8" text-anchor="middle">5</text><g class="mark"><title>c = 4: 5 problems</title><rect class="s1" x="584.6" y="260.8" width="18.9" height="9.2" rx="2"/></g><text class="t-tick" x="594.1" y="288.0" text-anchor="middle">4</text><text class="t-tick" x="594.1" y="256.8" text-anchor="middle">5</text><g class="mark"><title>c = 5: 10 problems</title><rect class="s1" x="605.5" y="251.7" width="18.9" height="18.3" rx="2"/></g><text class="t-tick" x="615.0" y="288.0" text-anchor="middle">5</text><text class="t-tick" x="615.0" y="247.7" text-anchor="middle">10</text><g class="mark"><title>c = 6: 2 problems</title><rect class="s1" x="626.5" y="266.3" width="18.9" height="3.7" rx="2"/></g><text class="t-tick" x="635.9" y="288.0" text-anchor="middle">6</text><text class="t-tick" x="635.9" y="262.3" text-anchor="middle">2</text><g class="mark"><title>c = 7: 3 problems</title><rect class="s1" x="647.4" y="264.5" width="18.9" height="5.5" rx="2"/></g><text class="t-tick" x="656.8" y="288.0" text-anchor="middle">7</text><text class="t-tick" x="656.8" y="260.5" text-anchor="middle">3</text><g class="mark"><title>c = 8: 4 problems</title><rect class="s1" x="668.3" y="262.7" width="18.9" height="7.3" rx="2"/></g><text class="t-tick" x="677.7" y="288.0" text-anchor="middle">8</text><text class="t-tick" x="677.7" y="258.7" text-anchor="middle">4</text><g class="mark"><title>c = 9: 5 problems</title><rect class="s1" x="689.2" y="260.8" width="18.9" height="9.2" rx="2"/></g><text class="t-tick" x="698.6" y="288.0" text-anchor="middle">9</text><text class="t-tick" x="698.6" y="256.8" text-anchor="middle">5</text><g class="mark"><title>c = 10: 10 problems</title><rect class="s1" x="710.1" y="251.7" width="18.9" height="18.3" rx="2"/></g><text class="t-tick" x="719.5" y="288.0" text-anchor="middle">10</text><text class="t-tick" x="719.5" y="247.7" text-anchor="middle">10</text><text class="t-tick" x="615.0" y="308.0" text-anchor="middle">c (out of 10)</text></svg><figcaption>Left: pass@k on HumanEval for Qwen2.5-0.5B base, k from 1 to 10, from 10 samples per problem. The unbiased estimator rises from 20.0% to 41.5%; the shortcut 1 - (1 - c/n)^k falls further behind as k grows. Right: how many problems had 0 to 10 passing samples.</figcaption></figure>

Greedy decoding solves 47 of 164 problems, **28.7%** (95% interval 22.0 to 36.0), close to the report's 30.5. Single samples at temperature 0.8 are worse (pass@1 20.0%), but with 10 tries the model solves 41.5% of problems. The histogram shows why: 96 problems are never solved and only 10 are solved every time; for the ones in between, more attempts help a lot. That is why "sample many, keep one that passes" works for code, and why comparing someone's pass@10 with another's pass@1 is unfair.

The biased shortcut matches the estimator at $$k = 1$$ and drifts lower as $$k$$ grows: 37.6% against 41.5% at $$k = 10$$. With $$k = n$$ the gap is largest, because the shortcut never gives exactly 1 to a problem with $$c > 0$$.

One example (HumanEval/0, 3 of 10 samples pass) is a reminder that tests are only a proxy too. The task: return True if any two numbers in a list are closer than a threshold. One *passing* sample starts with

```python
    if len(numbers) <= 2:
        return False
```

which is wrong for a list of two close numbers, but the problem's tests never try one. Functional correctness is only as good as the tests; HumanEval+ (Liu et al., 2023) adds many more tests to each problem for exactly this reason.

## 3.8 Contamination: when the test leaks into the training data

A benchmark is a fair test only if the model has not seen the answers, which is hard to guarantee for models trained on most of the public web. MMLU comes from practice exams that are online; GSM8K and HumanEval are on GitHub and copied into tutorials and other datasets.

> [!DEFINITION] Contamination (data leakage)
> Test items, or their answers, appearing in a model's training data. It can be exact (the same text), partial (the question without the answer, or the answer in a forum post), or disguised (a paraphrase or translation). Contamination makes a benchmark score an overestimate of how the model does on truly new questions.

> [!DEFINITION] Decontamination
> Removing from the training data anything that overlaps too much with the test sets you will report, before training. Done with string matching (n-grams) in most published reports.

### The GPT-3 method: 13-grams

GPT-3 (Brown et al., 2020) was trained on a large web crawl, and its authors tried to remove every test item of every benchmark they reported. They also told the story of what went wrong.

> [!PAPER] Brown et al. (2020), GPT-3 · Section 4, Measuring and Preventing Memorization of Benchmarks · page 31
> [![A paragraph from Section 4 of the GPT-3 paper, highlighted: a bug resulted in only partial removal of all detected overlaps from the training data, and the definition of potentially leaked examples as those with a 13-gram overlap with anything in the pretraining set. The paragraph explains that they produce a clean version of each benchmark and compare scores](/img/training/ch3-gpt3-contam.png)](/img/training/ch3-gpt3-contam.png)
>
> **Context:** Section 4, written after training had finished.
>
> **What it says:** "a bug resulted in only partial removal" of the overlaps, and retraining was too expensive. So for each benchmark they built a "clean" version without the examples that "have a 13-gram overlap with anything in the pretraining set", and compared the scores on the full and clean versions.
>
> **Why it matters:** this became the standard way to *report* contamination when you cannot remove it: if the score on the clean subset is about the same, contamination probably did not help much.

> [!DEFINITION] n-gram
> A sequence of $$n$$ consecutive words (or tokens). "The cat sat on" contains the 3-grams "the cat sat" and "cat sat on". Two texts that share a long n-gram, like 13 words in a row, almost certainly share a source.

Why 13? Short n-grams collide by chance ("one of the most important" is everywhere); thirteen words in a row almost never do. (The GPT-3 appendix actually used between 8 and 13 words depending on the dataset, after removing case and punctuation.)

> [!PAPER] Brown et al. (2020), GPT-3 · Figure 4.2 · page 32
> [![Scatter plot. Horizontal axis: percentage of the benchmark that is verified clean, from 0 to 100%. Vertical axis: percent change in performance when evaluating only on the clean subset. Most benchmarks sit near zero change; a few such as QuAC, DROP, Reversed Words and Anagrams move by 10 to 25%](/img/training/ch3-gpt3-fig42.png)](/img/training/ch3-gpt3-fig42.png)
>
> **Context:** the result of the clean-subset comparison.
>
> **What it says:** for most benchmarks, scores on the clean subset changed "negligibly". For reading-comprehension sets such as QuAC, SQuAD2 and DROP, over 90% of examples were flagged, so the clean subsets were too small to measure much; on inspection, the *source passages* were in the training data but the question and answer pairs were not. The authors found evidence of a real effect only for PIQA and Winograd, which they marked with an asterisk.
>
> **Why it matters:** high overlap does not always mean a higher score (much of the overlap was the *source* text, like Wikipedia, not the answers), and low overlap does not prove a clean test. You have to measure the effect, not just the overlap.

Modern reports do the same at the data level. The Qwen2.5 report describes its rule.

> [!PAPER] Yang et al. (2024), Qwen2.5 Technical Report · Section 5 · page 7
> [![A paragraph from the Qwen2.5 report, highlighted: we exclude potentially contaminated data using n-gram matching; a training sequence is removed if the longest common subsequence with a test sequence is at least 13 tokens and at least 0.6 times the length of the shorter one](/img/training/ch3-qwen25-decontam.png)](/img/training/ch3-qwen25-decontam.png)
>
> **Context:** the start of the evaluation section, about the models we test in this chapter.
>
> **What it says:** a training sequence is removed if its **longest common subsequence** with any test sequence is at least 13 tokens long *and* covers at least 60% of the shorter sequence.
>
> **Why it matters:** the same idea as GPT-3, made a bit more forgiving of small edits: a subsequence may skip words, so inserting one word does not break the match.

### Experiment: the 13-gram check, and what it misses

We cannot search Qwen's training data (it is not public), but we can run the method on a corpus we do have. `ch3_contamination.py` treats the WikiText-103 training split, about 87 million words of Wikipedia, as a stand-in "training corpus", and checks 1,000 benchmark items against it: 300 GSM8K test questions, 400 MMLU test questions (with their options) and, as a control, 300 paragraphs of the WikiText-2 *test* set. Then it plants 30 GSM8K test questions in the corpus on purpose: 10 copied exactly, 10 lightly reworded by swapping a few words ("has" to "owns", "How many" to "What number of", "2" to "two"), and 10 paraphrased by Qwen2.5-1.5B-Instruct, asked to rewrite each problem in different words while keeping every number.

```python
# simplified from ch3_contamination.py, part A
def words(s):                                   # normalise: lower case, letters and digits only
    return [w for w in re.sub(r'[^0-9a-z ]+', ' ', s.lower()).split() if w]

def grams(ws, n=13):                            # every window of 13 consecutive words
    return {' '.join(ws[i:i + n]) for i in range(len(ws) - n + 1)}

want = {}                                       # 13-gram -> the benchmark items that contain it
for key, text in items.items():
    for g in grams(words(text)):
        want.setdefault(g, set()).add(key)

for line in corpus:                             # one pass over 87 million words
    ws = words(line)
    for i in range(len(ws) - 12):
        g = ' '.join(ws[i:i + 13])
        if g in want:                           # a set lookup: fast
            for key in want[g]:
                hits.setdefault(key, set()).add(g)
```

The trick that makes this fast is to build the set of 13-grams from the *benchmark* (small: 63,791 of them) and stream the *corpus* through it, instead of building a set of all corpus 13-grams (billions, for a real pretraining set). The scan took under two minutes on a laptop CPU.

```text
== A. 13-gram overlap with the WikiText-103 training split ==
benchmark items: 1000 (300 GSM8K, 400 MMLU, 300 WikiText-2 test paragraphs); distinct 13-grams: 63,791
corpus: 86,678,513 words (WikiText-103 train + 30 planted lines), scanned in 104s
gsm8k             0 of 270 items share at least one 13-gram with the corpus (the 30 planted questions not counted)
mmlu              1 of 400 items share at least one 13-gram with the corpus
    e.g. mmlu #179: "democracy if we look to the laws they afford equal justice to all"
wikitext2-test   39 of 300 items share at least one 13-gram with the corpus
    e.g. wikitext2-test #4: "according to the digital sheet music published at musicnotes com by sony atv"
    e.g. wikitext2-test #6: "000 long tons 18 289 t each this was the genesis of the"
planted, copied exactly         : found 10 of 10   (share of the original 13-grams still present: 100%)
planted, lightly reworded       : found 9 of 10   (share of the original 13-grams still present: 55%)
planted, paraphrased by a model : found 0 of 10   (share of the original 13-grams still present: 0%)
the paraphrasing model kept exactly the same numbers in 5 of 10; of those, 0 were found
```

Read it line by line.

- **GSM8K: 0 of 270.** Wikipedia has no grade-school word problems, as expected.
- **MMLU: 1 of 400.** A history question quotes Pericles' funeral oration, which is also in a Wikipedia article. A real overlap, but harmless: knowing the text is the knowledge being tested. Whether an overlap "counts" needs a human look.
- **WikiText-2 test: 39 of 300.** These articles are *not* in WikiText-103's training split, yet 13% share a 13-gram with it, because Wikipedia has boilerplate ("according to the digital sheet music published at Musicnotes.com by Sony/ATV" appears in many song articles). N-gram checks also flag shared templates.
- **Planted copies: 10 of 10 found.** The method does what it is designed for.
- **Light rewording: 9 of 10 still found.** 55% of the original windows survive a few word swaps, and one is enough.
- **Model paraphrase: 0 of 10 found.** A real rewrite shares no 13-word window with the original. (Honest note: the 1.5B paraphraser did not always follow the instruction; in 5 of the 10 it changed some numbers, so those are no longer the same problem. Of the 5 that kept every number, which are genuine disguised copies, none was found either.)

<figure class="fig"><svg viewBox="0 0 760 382" role="img" aria-label="Two GSM8K test questions, each followed by a changed version: a light rewording with a few words swapped, which still shares many 13-word windows with the original, and a full paraphrase by a model, which shares none or almost none."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The 13-gram check: does any 13-word window of the test item appear in the training data?</text><text class="t-muted" x="20.0" y="46.0" text-anchor="start">original test question: 17 windows of 13 words</text><rect class="box" x="20.0" y="56.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="40.4" y="71.0" text-anchor="middle">jean</text><rect class="box" x="64.8" y="56.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="81.6" y="71.0" text-anchor="middle">has</text><rect class="box" x="102.4" y="56.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="115.6" y="71.0" text-anchor="middle">30</text><rect class="box" x="132.8" y="56.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="171.2" y="71.0" text-anchor="middle">lollipops</text><rect class="box" x="213.6" y="56.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="234.0" y="71.0" text-anchor="middle">jean</text><rect class="box" x="258.4" y="56.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="278.8" y="71.0" text-anchor="middle">eats</text><rect class="box" x="303.2" y="56.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="312.8" y="71.0" text-anchor="middle">2</text><rect class="box" x="326.4" y="56.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="339.6" y="71.0" text-anchor="middle">of</text><rect class="box" x="356.8" y="56.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="373.6" y="71.0" text-anchor="middle">the</text><rect class="box" x="394.4" y="56.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="432.8" y="71.0" text-anchor="middle">lollipops</text><rect class="box" x="475.2" y="56.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="495.6" y="71.0" text-anchor="middle">with</text><rect class="box" x="520.0" y="56.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="536.8" y="71.0" text-anchor="middle">the</text><rect class="box" x="557.6" y="56.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="596.0" y="71.0" text-anchor="middle">remaining</text><rect class="box" x="638.4" y="56.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="676.8" y="71.0" text-anchor="middle">lollipops</text><rect class="box" x="20.0" y="84.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="40.4" y="99.0" text-anchor="middle">jean</text><rect class="box" x="64.8" y="84.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="88.8" y="99.0" text-anchor="middle">wants</text><rect class="box" x="116.8" y="84.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="130.0" y="99.0" text-anchor="middle">to</text><rect class="box" x="147.2" y="84.0" width="62.4" height="22.0" rx="4"/><text class="t-tick" x="178.4" y="99.0" text-anchor="middle">package</text><rect class="box" x="213.6" y="84.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="223.2" y="99.0" text-anchor="middle">2</text><rect class="box" x="236.8" y="84.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="275.2" y="99.0" text-anchor="middle">lollipops</text><rect class="box" x="317.6" y="84.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="330.8" y="99.0" text-anchor="middle">in</text><rect class="box" x="348.0" y="84.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="364.8" y="99.0" text-anchor="middle">one</text><rect class="box" x="385.6" y="84.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="402.4" y="99.0" text-anchor="middle">bag</text><rect class="box" x="423.2" y="84.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="440.0" y="99.0" text-anchor="middle">how</text><rect class="box" x="460.8" y="84.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="481.2" y="99.0" text-anchor="middle">many</text><rect class="box" x="505.6" y="84.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="526.0" y="99.0" text-anchor="middle">bags</text><rect class="box" x="550.4" y="84.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="567.2" y="99.0" text-anchor="middle">can</text><rect class="box" x="588.0" y="84.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="608.4" y="99.0" text-anchor="middle">jean</text><rect class="box" x="632.8" y="84.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="653.2" y="99.0" text-anchor="middle">fill</text><text class="t-muted" x="20.0" y="126.0" text-anchor="start">lightly reworded (changed words shaded): 4 of its windows still match the original</text><rect class="box" x="20.0" y="136.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="40.4" y="151.0" text-anchor="middle">jean</text><rect class="s4" x="64.8" y="136.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="85.2" y="151.0" text-anchor="middle">owns</text><rect class="box" x="109.6" y="136.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="122.8" y="151.0" text-anchor="middle">30</text><rect class="box" x="140.0" y="136.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="178.4" y="151.0" text-anchor="middle">lollipops</text><rect class="box" x="220.8" y="136.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="241.2" y="151.0" text-anchor="middle">jean</text><rect class="box" x="265.6" y="136.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="286.0" y="151.0" text-anchor="middle">eats</text><rect class="s4" x="310.4" y="136.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="327.2" y="151.0" text-anchor="middle">two</text><rect class="box" x="348.0" y="136.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="361.2" y="151.0" text-anchor="middle">of</text><rect class="box" x="378.4" y="136.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="395.2" y="151.0" text-anchor="middle">the</text><rect class="box" x="416.0" y="136.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="454.4" y="151.0" text-anchor="middle">lollipops</text><rect class="box" x="496.8" y="136.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="517.2" y="151.0" text-anchor="middle">with</text><rect class="box" x="541.6" y="136.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="558.4" y="151.0" text-anchor="middle">the</text><rect class="box" x="579.2" y="136.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="617.6" y="151.0" text-anchor="middle">remaining</text><rect class="box" x="660.0" y="136.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="698.4" y="151.0" text-anchor="middle">lollipops</text><rect class="box" x="20.0" y="164.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="40.4" y="179.0" text-anchor="middle">jean</text><rect class="box" x="64.8" y="164.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="88.8" y="179.0" text-anchor="middle">wants</text><rect class="box" x="116.8" y="164.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="130.0" y="179.0" text-anchor="middle">to</text><rect class="box" x="147.2" y="164.0" width="62.4" height="22.0" rx="4"/><text class="t-tick" x="178.4" y="179.0" text-anchor="middle">package</text><rect class="box" x="213.6" y="164.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="223.2" y="179.0" text-anchor="middle">2</text><rect class="box" x="236.8" y="164.0" width="76.8" height="22.0" rx="4"/><text class="t-tick" x="275.2" y="179.0" text-anchor="middle">lollipops</text><rect class="box" x="317.6" y="164.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="330.8" y="179.0" text-anchor="middle">in</text><rect class="box" x="348.0" y="164.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="364.8" y="179.0" text-anchor="middle">one</text><rect class="box" x="385.6" y="164.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="402.4" y="179.0" text-anchor="middle">bag</text><rect class="s4" x="423.2" y="164.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="443.6" y="179.0" text-anchor="middle">what</text><rect class="s4" x="468.0" y="164.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="495.6" y="179.0" text-anchor="middle">number</text><rect class="box" x="527.2" y="164.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="540.4" y="179.0" text-anchor="middle">of</text><rect class="box" x="557.6" y="164.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="578.0" y="179.0" text-anchor="middle">bags</text><rect class="box" x="602.4" y="164.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="619.2" y="179.0" text-anchor="middle">can</text><rect class="box" x="640.0" y="164.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="660.4" y="179.0" text-anchor="middle">jean</text><rect class="box" x="684.8" y="164.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="705.2" y="179.0" text-anchor="middle">fill</text><text class="t-muted" x="20.0" y="218.0" text-anchor="start">original test question: 21 windows of 13 words</text><rect class="box" x="20.0" y="228.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="47.6" y="243.0" text-anchor="middle">miguel</text><rect class="box" x="79.2" y="228.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="99.6" y="243.0" text-anchor="middle">uses</text><rect class="box" x="124.0" y="228.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="133.6" y="243.0" text-anchor="middle">2</text><rect class="box" x="147.2" y="228.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="167.6" y="243.0" text-anchor="middle">pads</text><rect class="box" x="192.0" y="228.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="205.2" y="243.0" text-anchor="middle">of</text><rect class="box" x="222.4" y="228.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="246.4" y="243.0" text-anchor="middle">paper</text><rect class="box" x="274.4" y="228.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="284.0" y="243.0" text-anchor="middle">a</text><rect class="box" x="297.6" y="228.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="318.0" y="243.0" text-anchor="middle">week</text><rect class="box" x="342.4" y="228.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="359.2" y="243.0" text-anchor="middle">for</text><rect class="box" x="380.0" y="228.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="396.8" y="243.0" text-anchor="middle">his</text><rect class="box" x="417.6" y="228.0" width="62.4" height="22.0" rx="4"/><text class="t-tick" x="448.8" y="243.0" text-anchor="middle">drawing</text><rect class="box" x="484.0" y="228.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="497.2" y="243.0" text-anchor="middle">if</text><rect class="box" x="514.4" y="228.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="538.4" y="243.0" text-anchor="middle">there</text><rect class="box" x="566.4" y="228.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="583.2" y="243.0" text-anchor="middle">are</text><rect class="box" x="604.0" y="228.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="617.2" y="243.0" text-anchor="middle">30</text><rect class="box" x="634.4" y="228.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="662.0" y="243.0" text-anchor="middle">sheets</text><rect class="box" x="693.6" y="228.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="706.8" y="243.0" text-anchor="middle">of</text><rect class="box" x="20.0" y="256.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="44.0" y="271.0" text-anchor="middle">paper</text><rect class="box" x="72.0" y="256.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="85.2" y="271.0" text-anchor="middle">on</text><rect class="box" x="102.4" y="256.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="112.0" y="271.0" text-anchor="middle">a</text><rect class="box" x="125.6" y="256.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="142.4" y="271.0" text-anchor="middle">pad</text><rect class="box" x="163.2" y="256.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="176.4" y="271.0" text-anchor="middle">of</text><rect class="box" x="193.6" y="256.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="217.6" y="271.0" text-anchor="middle">paper</text><rect class="box" x="245.6" y="256.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="262.4" y="271.0" text-anchor="middle">how</text><rect class="box" x="283.2" y="256.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="303.6" y="271.0" text-anchor="middle">many</text><rect class="box" x="328.0" y="256.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="355.6" y="271.0" text-anchor="middle">sheets</text><rect class="box" x="387.2" y="256.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="400.4" y="271.0" text-anchor="middle">of</text><rect class="box" x="417.6" y="256.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="441.6" y="271.0" text-anchor="middle">paper</text><rect class="box" x="469.6" y="256.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="490.0" y="271.0" text-anchor="middle">does</text><rect class="box" x="514.4" y="256.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="527.6" y="271.0" text-anchor="middle">he</text><rect class="box" x="544.8" y="256.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="561.6" y="271.0" text-anchor="middle">use</text><rect class="box" x="582.4" y="256.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="606.4" y="271.0" text-anchor="middle">every</text><rect class="box" x="634.4" y="256.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="658.4" y="271.0" text-anchor="middle">month</text><text class="t-muted" x="20.0" y="298.0" text-anchor="start">paraphrased by a model (words not in the original shaded): 0 of its windows still match the original</text><rect class="box" x="20.0" y="308.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="36.8" y="323.0" text-anchor="middle">how</text><rect class="box" x="57.6" y="308.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="78.0" y="323.0" text-anchor="middle">many</text><rect class="box" x="102.4" y="308.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="130.0" y="323.0" text-anchor="middle">sheets</text><rect class="box" x="161.6" y="308.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="174.8" y="323.0" text-anchor="middle">of</text><rect class="box" x="192.0" y="308.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="216.0" y="323.0" text-anchor="middle">paper</text><rect class="box" x="244.0" y="308.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="264.4" y="323.0" text-anchor="middle">does</text><rect class="box" x="288.8" y="308.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="316.4" y="323.0" text-anchor="middle">miguel</text><rect class="box" x="348.0" y="308.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="364.8" y="323.0" text-anchor="middle">use</text><rect class="s2" x="385.6" y="308.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="406.0" y="323.0" text-anchor="middle">each</text><rect class="box" x="430.4" y="308.0" width="48.0" height="22.0" rx="4"/><text class="t-tick" x="454.4" y="323.0" text-anchor="middle">month</text><rect class="box" x="482.4" y="308.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="495.6" y="323.0" text-anchor="middle">if</text><rect class="box" x="512.8" y="308.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="526.0" y="323.0" text-anchor="middle">he</text><rect class="box" x="543.2" y="308.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="563.6" y="323.0" text-anchor="middle">uses</text><rect class="box" x="588.0" y="308.0" width="19.2" height="22.0" rx="4"/><text class="t-tick" x="597.6" y="323.0" text-anchor="middle">2</text><rect class="box" x="611.2" y="308.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="631.6" y="323.0" text-anchor="middle">pads</text><rect class="s2" x="656.0" y="308.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="672.8" y="323.0" text-anchor="middle">per</text><rect class="box" x="693.6" y="308.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="714.0" y="323.0" text-anchor="middle">week</text><rect class="box" x="20.0" y="336.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="36.8" y="351.0" text-anchor="middle">for</text><rect class="box" x="57.6" y="336.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="74.4" y="351.0" text-anchor="middle">his</text><rect class="s2" x="95.2" y="336.0" width="69.6" height="22.0" rx="4"/><text class="t-tick" x="130.0" y="351.0" text-anchor="middle">drawings</text><rect class="s2" x="168.8" y="336.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="189.2" y="351.0" text-anchor="middle">with</text><rect class="s2" x="213.6" y="336.0" width="40.8" height="22.0" rx="4"/><text class="t-tick" x="234.0" y="351.0" text-anchor="middle">each</text><rect class="box" x="258.4" y="336.0" width="33.6" height="22.0" rx="4"/><text class="t-tick" x="275.2" y="351.0" text-anchor="middle">pad</text><rect class="s2" x="296.0" y="336.0" width="84.0" height="22.0" rx="4"/><text class="t-tick" x="338.0" y="351.0" text-anchor="middle">containing</text><rect class="box" x="384.0" y="336.0" width="26.4" height="22.0" rx="4"/><text class="t-tick" x="397.2" y="351.0" text-anchor="middle">30</text><rect class="box" x="414.4" y="336.0" width="55.2" height="22.0" rx="4"/><text class="t-tick" x="442.0" y="351.0" text-anchor="middle">sheets</text></svg><figcaption>Why the 13-gram check catches copies and misses paraphrases. Top: a GSM8K test question and a light rewording; the shaded words were changed, but long runs of 13 unchanged words remain. Bottom: another test question and a faithful paraphrase by Qwen2.5-1.5B-Instruct (same numbers, same question); no 13-word window survives.</figcaption></figure>

### Disguised contamination

The weakness is fundamental. String matching only finds copies. Yang et al. (2023) showed how far a disguised copy can go.

> [!PAPER] Yang et al. (2023), Rethinking Benchmark and Contamination with Rephrased Samples · Abstract · page 1
> [![The abstract, highlighted: simple variations of test data, such as paraphrasing or translation, can easily bypass these decontamination measures. If such variation of test data is not eliminated, a 13B model can easily overfit a test benchmark and achieve drastically high performance, on par with GPT-4](/img/training/ch3-rephrase-abstract.png)](/img/training/ch3-rephrase-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** "simple variations of test data (e.g., paraphrasing, translation) can easily bypass these decontamination measures", and a 13B model trained on such rephrased test sets scored "on par with GPT-4" on MMLU, GSM8K and HumanEval.
>
> **Why it matters:** an n-gram check that finds nothing is not proof of a clean test. The authors propose using a model to detect paraphrases; that is more expensive and still not perfect.

### Experiment: does the model know the test by heart?

If you cannot search the training data (for open-weight models it is usually not released), you can probe the model instead. Give it the first half of a test item and let it continue greedily. If it reproduces the next words exactly, it has very probably seen that text.

`ch3_contamination.py` (part B) gives Qwen2.5-0.5B base the first half of each item's words, decodes greedily, and counts a hit if the next 8 words match the real text exactly (ignoring trailing punctuation). Eight specific words in a row rarely come out by chance unless the text is predictable from its own first half.

```text
== B. memorisation probe: first half in, does the model write the next 8 words exactly? ==
GSM8K test questions           3 of 286 continued exactly (1.0%)
GSM8K train questions        257 of 283 continued exactly (90.8%)
MMLU test questions            7 of 661 continued exactly (1.1%)
WikiText-2 test paragraphs     0 of 300 continued exactly (0.0%)
HumanEval prompts              5 of 159 continued exactly (3.1%)
```

This is the most striking number in the chapter. Given the first half of a GSM8K **training** question, the base model writes the next 8 words exactly in 91% of cases. For the **test** questions, from the same dataset, written by the same people in the same style, it manages 1%. A 0.5B model cannot guess 8 exact words of a word problem 91% of the time; it has seen the GSM8K training questions, probably many times, in its pretraining data. Here is one, with the model's continuation after the arrow:

```text
...'After downloading 800 files, he deleted 70% of them because they were not' -> 'helpful. He downloaded 400 more files but again'
```

Two readings. First, the decontamination the Qwen2.5 report describes seems to have worked for the GSM8K *test* set: 1% is what predictable phrasing alone produces (one hit is "...it takes about the" followed by "same amount of time to peel and cut"). Second, the *training* set was not removed, and did not have to be. But a model that has memorised 7,473 problems in exactly the test's style has had far more practice on this format than one that has not: the test score is clean, but it is not "maths from scratch". The MMLU and HumanEval hits look like predictability rather than memory, and Wikipedia text, though surely in the training data, is not reproduced word for word by a model this small.

> [!WARNING]
> A low score on a probe like this does not prove the test set was never seen. A small model can see a text a few times without being able to reproduce it, and a paraphrased copy would not be reproduced word for word anyway. A high score, like the 91% here, is strong evidence that it *was* seen.

### Fresh test sets

The most convincing check is a test the model cannot have seen. Zhang et al. (2024) wrote **GSM1k**, 1,205 new problems in GSM8K's style, and found drops of up to 8% for some model families (signs of partial memorisation), while frontier models showed "minimal signs of overfitting". LiveBench and LiveCodeBench add new, dated questions every few months, so a model can be scored only on questions written after its training data was collected.

## 3.9 Asking people: pairwise preference and Chatbot Arena

Benchmarks with an answer key cannot measure most of what an assistant does: explaining, advising, writing, chatting. For those tasks the most trusted judge is still a person. Scores from 1 to 10 work badly (one rater's 7 is another's 5), but "which of these two answers is better?" works well: people agree more on comparisons, and a comparison needs no rubric. InstructGPT (Chapter 2) was evaluated this way, and so is almost every chat model since.

> [!DEFINITION] Pairwise comparison
> A rater sees one prompt and two answers (from two models, labelled only A and B) and says which is better, or that they tie. Many comparisons together give a win rate or a rating for each model.

> [!DEFINITION] Win rate
> The fraction of comparisons a model wins against a fixed opponent (ties are often counted as half a win). A win rate of 50% means "no detectable difference".

### A win rate needs an error bar

A win rate is an accuracy in disguise: each comparison is a 1 (win) or a 0 (loss), and the win rate is their mean. So the same confidence interval applies. With $$n$$ comparisons and an observed win rate $$\hat{p}$$, the normal approximation gives

$$\hat{p} \pm 1.96 \sqrt{\frac{\hat{p}\,(1 - \hat{p})}{n}}$$

where:

- $$\hat{p}$$ (read "p-hat") is the observed win rate, wins divided by comparisons;
- $$n$$ is the number of comparisons;
- $$\sqrt{\hat{p}(1-\hat{p})/n}$$ is the **standard error**, how much $$\hat{p}$$ would wobble if you repeated the whole experiment with new comparisons;
- 1.96 is the number of standard errors that covers the middle 95% of a normal distribution.

**Worked example.** In the simulation in `ch3_arena.py`, model-A's true chance of beating model-B is 0.557. With 100 simulated votes it won 64 times. The standard error is $$\sqrt{0.64 \times 0.36 / 100} = 0.048$$, so the 95% interval is $$0.64 \pm 0.094$$, from 0.546 to 0.734. The bootstrap gives almost the same, 0.550 to 0.730. Both intervals contain the truth (0.557), but notice how misleading "64% win rate" would be on its own.

With 20 votes the script's interval runs from 33% to 77%: you cannot even tell which model is better. For this pair (a true gap of 40 rating points) you need about 400 votes before the interval stops including 50%.

### From many pairs to one ranking: Elo

A win rate compares two models. To rank fifty, you would need every pair, which grows as the square of the number of models. Ratings solve this. The idea comes from chess. In the **Elo system** (devised by Arpad Elo and adopted for chess ratings in the 1960s), every player has a rating, and the rating gap predicts the result of a game:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}$$

where:

- $$R_A$$ and $$R_B$$ are the current ratings of players (models) A and B;
- $$E_A$$ is A's **expected score**, the predicted chance that A wins (counting a tie as half);
- 400 sets the scale: a gap of 400 points means 10-to-1 odds, so A is expected to win $$10/11 \approx 91\%$$ of the time.

After the game, both ratings move toward the result:

$$R_A' = R_A + K\,(S_A - E_A), \qquad R_B' = R_B - K\,(S_A - E_A)$$

where:

- $$S_A$$ is the actual result for A: 1 for a win, 0 for a loss, 0.5 for a tie;
- $$S_A - E_A$$ is the **surprise**: positive if A did better than expected;
- $$K$$ is the step size, how many points a full surprise is worth (32 is common in chess; a smaller K makes ratings steadier but slower to react; our simulation uses 4);
- $$R_A'$$ and $$R_B'$$ are the new ratings. Whatever A gains, B loses.

**Worked example** (printed by `ch3_arena.py`). A has 1200, B has 1100, K = 32. A's expected score is $$1/(1 + 10^{-100/400}) = 1/(1 + 0.562) = 0.640$$. If A wins, the surprise is $$1 - 0.640 = 0.360$$, so A gains $$32 \times 0.360 = 11.5$$ points (1211.5) and B drops to 1088.5. If B wins instead, the surprise for A is $$0 - 0.640$$, so A loses 20.5 points (1179.5) and B rises to 1120.5. Upsets move ratings more than expected results.

<figure class="fig"><svg viewBox="0 0 760 226" role="img" aria-label="Three frames: A rated 1200 and B rated 1100, expected score 0.64 for A; if A wins, A gains 11.5 and B loses 11.5; if B wins, A loses 20.5 and B gains 20.5."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One Elo update (K = 32): the surprise decides how far the ratings move</text><rect class="box" x="20.0" y="40.0" width="236.0" height="150.0" rx="10"/><circle class="node on" cx="38.0" cy="58.0" r="11"/><text class="t-note" x="38.0" y="62.5" text-anchor="middle">1</text><text class="t-note" x="56.0" y="63.0" text-anchor="start">before the game</text><text class="t-tick" x="34.0" y="96.0" text-anchor="start">A: 1200</text><text class="t-tick" x="34.0" y="113.0" text-anchor="start">B: 1100</text><text class="t-muted" x="34.0" y="150.0" text-anchor="start">expected score of A</text><text class="t-muted" x="34.0" y="167.0" text-anchor="start">1 / (1 + 10^(-100/400)) = 0.640</text><rect class="box-3" x="268.0" y="40.0" width="236.0" height="150.0" rx="10"/><circle class="node on" cx="286.0" cy="58.0" r="11"/><text class="t-note" x="286.0" y="62.5" text-anchor="middle">2</text><text class="t-note" x="304.0" y="63.0" text-anchor="start">A wins (expected)</text><text class="t-tick" x="282.0" y="96.0" text-anchor="start">A: 1200 + 32 x 0.360 = 1211.5</text><text class="t-tick" x="282.0" y="113.0" text-anchor="start">B: 1100 - 32 x 0.360 = 1088.5</text><text class="t-muted" x="282.0" y="150.0" text-anchor="start">small surprise,</text><text class="t-muted" x="282.0" y="167.0" text-anchor="start">small move (11.5 points)</text><rect class="box-2" x="516.0" y="40.0" width="236.0" height="150.0" rx="10"/><circle class="node on" cx="534.0" cy="58.0" r="11"/><text class="t-note" x="534.0" y="62.5" text-anchor="middle">3</text><text class="t-note" x="552.0" y="63.0" text-anchor="start">B wins (an upset)</text><text class="t-tick" x="530.0" y="96.0" text-anchor="start">A: 1200 - 32 x 0.640 = 1179.5</text><text class="t-tick" x="530.0" y="113.0" text-anchor="start">B: 1100 + 32 x 0.640 = 1120.5</text><text class="t-muted" x="530.0" y="150.0" text-anchor="start">big surprise,</text><text class="t-muted" x="530.0" y="167.0" text-anchor="start">big move (20.5 points)</text><text class="t-muted" x="20.0" y="214.0" text-anchor="start">The two ratings always move by the same amount in opposite directions, so the total stays the same.</text></svg><figcaption>One Elo update with K = 32. The rating gap of 100 points predicts that A wins 64% of the time. An expected win moves the ratings by 11.5 points; an upset moves them by 20.5.</figcaption></figure>

Online Elo applies this update vote by vote, in the order the votes arrive. That ordering is its weakness: it was designed for players whose strength changes over time, so recent games count more. A model's weights do not change, so the order of the votes should not matter, but in online Elo it does.

### Bradley-Terry: fit all the votes at once

The fix is the model you met in Chapter 2, Section 2.4. **Bradley-Terry** gives each model a strength and says the chance that $$m$$ beats $$m'$$ depends only on the difference of strengths. Instead of updating vote by vote, you choose all the strengths at once to make the observed votes as likely as possible (maximum likelihood). The Chatbot Arena paper writes it like this.

> [!PAPER] Chiang et al. (2024), Chatbot Arena · Section 4, equations 2 and 3 · page 4
> [![Equations 2 and 3 of the Chatbot Arena paper. Equation 2: the probability that model m beats model m-prime is one over one plus e to the power xi of m-prime minus xi of m. Equation 3: the score is the vector xi that minimises the expected binary cross-entropy loss between the vote H and this probability. Highlighted: so-called BT coefficients, and binary cross-entropy](/img/training/ch3-arena-bt.png)](/img/training/ch3-arena-bt.png)
>
> **Context:** Section 4, "Statistical methods", the paper's core.
>
> **What it says:** $$\xi$$ is "an M-length vector of so-called BT coefficients", one per model, and the ratings are the $$\xi$$ that minimise the "binary cross-entropy" between predicted and actual votes. Fixing $$\xi_1 = 0$$ removes the freedom to add a constant to everything.
>
> **Why it matters:** this is exactly the reward-model loss of Chapter 2 with models in place of answers. The paper also notes that earlier versions of the leaderboard reported online Elo, and that they switched because the BT coefficients "are better for the purpose of statistical estimation".

In symbols, for one vote between models $$m$$ and $$m'$$:

$$P(m \text{ beats } m') = \frac{1}{1 + e^{\xi_{m'} - \xi_m}} = \sigma(\xi_m - \xi_{m'})$$

where:

- $$\xi_m$$ (Greek "xi") is model $$m$$'s Bradley-Terry coefficient, on the natural-log scale;
- $$\sigma$$ is the sigmoid from Chapter 2;
- the ratings shown on a leaderboard are $$\xi$$ converted to the chess scale: multiply by $$400 / \ln 10 \approx 173.7$$ and add a constant (Arena anchors one model at a fixed value).

So a gap of 1 in $$\xi$$ is a gap of 173.7 rating points, and it means the stronger model wins $$\sigma(1) = 73\%$$ of the time.

### Simulation: can we get the true ratings back?

`ch3_arena.py` invents six models with hidden true ratings (1250, 1210, 1190, 1120, 1060 and 1000), simulates 6,000 votes between random pairs using exactly the Bradley-Terry probability, and then tries to recover the ratings from the votes alone.

```python
# simplified from ch3_arena.py
def bt(a, b, y, iters=2000, lr=0.5):
    W = np.zeros((M, M))
    np.add.at(W, (a, b), y); np.add.at(W, (b, a), 1 - y)    # W[i, j] = wins of i over j
    Nn = W + W.T                                              # games played between i and j
    xi = np.zeros(M)
    for _ in range(iters):
        P = 1 / (1 + np.exp(-(xi[:, None] - xi[None, :])))   # P[i, j] = predicted chance i beats j
        g = (Nn * P - W).sum(1)                               # gradient of the cross-entropy
        xi -= lr * g / Nn.sum(1)                              # one gradient step
        xi -= xi.mean()                                       # remove the free constant
    return xi * 400 / math.log(10)                            # to the chess scale
```

The votes are first counted into a table of wins, so their order cannot matter. For each model, the gradient of the cross-entropy is "predicted wins minus actual wins", and gradient descent moves each $$\xi$$ until the two agree. The confidence intervals come from the bootstrap of Section 3.6: resample the 6,000 votes 300 times, refit, and keep the middle 95%.

```text
== 6000 simulated votes between random pairs ==
model       true  online Elo    Elo over 100 orders  Bradley-Terry   95% bootstrap CI
model-A     1250        1253       1224 to  1293           1255      1234 to  1279
model-B     1210        1244       1163 to  1257           1207      1185 to  1230
model-C     1190        1189       1131 to  1216           1174      1153 to  1196
model-D     1120        1158       1080 to  1172           1122      1101 to  1144
model-E     1060        1064       1030 to  1116           1072      1049 to  1095
model-F     1000        1000       1000 to  1000           1000      1000 to  1000
model-B and model-C are 20 points apart; their intervals overlap: the data cannot separate them
```

<figure class="fig"><svg viewBox="0 0 760 380" role="img" aria-label="For six simulated models: the true rating, the Bradley-Terry estimate with its 95% bootstrap interval, and the spread of online Elo ratings over 100 random orders of the same votes."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">6,000 simulated votes: how well do the ratings come back?</text><line class="path" x1="110.0" y1="38.0" x2="110.0" y2="54.0"/><text class="t-tick" x="120.0" y="50.0" text-anchor="start">true rating</text><circle class="s3" cx="220.0" cy="46.0" r="5"/><text class="t-tick" x="232.0" y="50.0" text-anchor="start">Bradley-Terry fit, 95% bootstrap CI</text><circle class="s2" cx="490.0" cy="46.0" r="5"/><text class="t-tick" x="502.0" y="50.0" text-anchor="start">online Elo: range over 100 vote orders</text><text class="t-tick" x="20.0" y="86.0" text-anchor="start">model-A</text><line class="path" x1="575.0" y1="66.0" x2="575.0" y2="102.0"/><line class="axis" x1="550.5" y1="76.0" x2="619.2" y2="76.0"/><line class="axis" x1="550.5" y1="71.0" x2="550.5" y2="81.0"/><line class="axis" x1="619.2" y1="71.0" x2="619.2" y2="81.0"/><g class="mark"><title>model-A: BT 1255 (1234 to 1279)</title><circle class="s3" cx="582.1" cy="76.0" r="5"/></g><line class="axis" x1="534.5" y1="92.0" x2="640.9" y2="92.0"/><line class="axis" x1="534.5" y1="87.0" x2="534.5" y2="97.0"/><line class="axis" x1="640.9" y1="87.0" x2="640.9" y2="97.0"/><g class="mark"><title>model-A: Elo 1253 (orders: 1224 to 1293)</title><circle class="s2" cx="579.0" cy="92.0" r="4"/></g><text class="t-tick" x="20.0" y="130.0" text-anchor="start">model-B</text><line class="path" x1="513.0" y1="110.0" x2="513.0" y2="146.0"/><line class="axis" x1="473.5" y1="120.0" x2="544.6" y2="120.0"/><line class="axis" x1="473.5" y1="115.0" x2="473.5" y2="125.0"/><line class="axis" x1="544.6" y1="115.0" x2="544.6" y2="125.0"/><g class="mark"><title>model-B: BT 1207 (1185 to 1230)</title><circle class="s3" cx="509.1" cy="120.0" r="5"/></g><line class="axis" x1="440.9" y1="136.0" x2="585.4" y2="136.0"/><line class="axis" x1="440.9" y1="131.0" x2="440.9" y2="141.0"/><line class="axis" x1="585.4" y1="131.0" x2="585.4" y2="141.0"/><g class="mark"><title>model-B: Elo 1244 (orders: 1163 to 1257)</title><circle class="s2" cx="566.1" cy="136.0" r="4"/></g><text class="t-tick" x="20.0" y="174.0" text-anchor="start">model-C</text><line class="path" x1="482.0" y1="154.0" x2="482.0" y2="190.0"/><line class="axis" x1="425.1" y1="164.0" x2="491.2" y2="164.0"/><line class="axis" x1="425.1" y1="159.0" x2="425.1" y2="169.0"/><line class="axis" x1="491.2" y1="159.0" x2="491.2" y2="169.0"/><g class="mark"><title>model-C: BT 1174 (1153 to 1196)</title><circle class="s3" cx="457.9" cy="164.0" r="5"/></g><line class="axis" x1="390.4" y1="180.0" x2="522.6" y2="180.0"/><line class="axis" x1="390.4" y1="175.0" x2="390.4" y2="185.0"/><line class="axis" x1="522.6" y1="175.0" x2="522.6" y2="185.0"/><g class="mark"><title>model-C: Elo 1189 (orders: 1131 to 1216)</title><circle class="s2" cx="480.9" cy="180.0" r="4"/></g><text class="t-tick" x="20.0" y="218.0" text-anchor="start">model-D</text><line class="path" x1="373.5" y1="198.0" x2="373.5" y2="234.0"/><line class="axis" x1="343.5" y1="208.0" x2="410.0" y2="208.0"/><line class="axis" x1="343.5" y1="203.0" x2="343.5" y2="213.0"/><line class="axis" x1="410.0" y1="203.0" x2="410.0" y2="213.0"/><g class="mark"><title>model-D: BT 1122 (1101 to 1144)</title><circle class="s3" cx="376.4" cy="208.0" r="5"/></g><line class="axis" x1="311.1" y1="224.0" x2="453.9" y2="224.0"/><line class="axis" x1="311.1" y1="219.0" x2="311.1" y2="229.0"/><line class="axis" x1="453.9" y1="219.0" x2="453.9" y2="229.0"/><g class="mark"><title>model-D: Elo 1158 (orders: 1080 to 1172)</title><circle class="s2" cx="431.8" cy="224.0" r="4"/></g><text class="t-tick" x="20.0" y="262.0" text-anchor="start">model-E</text><line class="path" x1="280.5" y1="242.0" x2="280.5" y2="278.0"/><line class="axis" x1="263.7" y1="252.0" x2="334.0" y2="252.0"/><line class="axis" x1="263.7" y1="247.0" x2="263.7" y2="257.0"/><line class="axis" x1="334.0" y1="247.0" x2="334.0" y2="257.0"/><g class="mark"><title>model-E: BT 1072 (1049 to 1095)</title><circle class="s3" cx="298.7" cy="252.0" r="5"/></g><line class="axis" x1="234.1" y1="268.0" x2="367.9" y2="268.0"/><line class="axis" x1="234.1" y1="263.0" x2="234.1" y2="273.0"/><line class="axis" x1="367.9" y1="263.0" x2="367.9" y2="273.0"/><g class="mark"><title>model-E: Elo 1064 (orders: 1030 to 1116)</title><circle class="s2" cx="286.4" cy="268.0" r="4"/></g><text class="t-tick" x="20.0" y="306.0" text-anchor="start">model-F</text><line class="path" x1="187.5" y1="286.0" x2="187.5" y2="322.0"/><line class="axis" x1="187.5" y1="296.0" x2="187.5" y2="296.0"/><line class="axis" x1="187.5" y1="291.0" x2="187.5" y2="301.0"/><line class="axis" x1="187.5" y1="291.0" x2="187.5" y2="301.0"/><g class="mark"><title>model-F: BT 1000 (1000 to 1000)</title><circle class="s3" cx="187.5" cy="296.0" r="5"/></g><line class="axis" x1="187.5" y1="312.0" x2="187.5" y2="312.0"/><line class="axis" x1="187.5" y1="307.0" x2="187.5" y2="317.0"/><line class="axis" x1="187.5" y1="307.0" x2="187.5" y2="317.0"/><g class="mark"><title>model-F: Elo 1000 (orders: 1000 to 1000)</title><circle class="s2" cx="187.5" cy="312.0" r="4"/></g><line class="axis" x1="110.0" y1="330.0" x2="730.0" y2="330.0"/><text class="t-tick" x="110.0" y="348.0" text-anchor="middle">950</text><text class="t-tick" x="187.5" y="348.0" text-anchor="middle">1000</text><text class="t-tick" x="265.0" y="348.0" text-anchor="middle">1050</text><text class="t-tick" x="342.5" y="348.0" text-anchor="middle">1100</text><text class="t-tick" x="420.0" y="348.0" text-anchor="middle">1150</text><text class="t-tick" x="497.5" y="348.0" text-anchor="middle">1200</text><text class="t-tick" x="575.0" y="348.0" text-anchor="middle">1250</text><text class="t-tick" x="652.5" y="348.0" text-anchor="middle">1300</text><text class="t-tick" x="730.0" y="348.0" text-anchor="middle">1350</text><text class="t-tick" x="420.0" y="368.0" text-anchor="middle">rating (model-F anchored at 1000)</text></svg><figcaption>Ratings recovered from 6,000 simulated votes. Vertical marks: the true ratings. First bar of each row: the Bradley-Terry fit and its 95% bootstrap interval; every interval contains the truth. Second bar: the range of online-Elo ratings when the same votes are fed in 100 different orders. model-F is anchored at 1000.</figcaption></figure>

Three things to notice. First, Bradley-Terry recovers every true rating within its interval. Second, online Elo, fed the *same votes* in a different order, lands anywhere in a band 60 to 100 points wide; in the original order it put model-B 34 points too high, above model-C by a wide margin, though they are only 20 points apart. Third, even Bradley-Terry with 6,000 votes cannot separate model-B from model-C: their intervals overlap. A leaderboard that prints them as rank 2 and rank 3 is showing noise.

How many votes does it take? The interval shrinks with the square root of the number of votes: model-A's interval is 202 points wide with 300 votes, 111 with 1,000, 61 with 3,000 and 35 with 10,000.

<figure class="fig"><svg viewBox="0 0 760 264" role="img" aria-label="Left: the 95% interval of a Bradley-Terry rating for 300, 1,000, 3,000 and 10,000 simulated votes. Right: the 95% interval of a win rate for 20, 100, 400 and 1,600 votes."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">How many votes do you need? Width of the 95% interval</text><text class="t-muted" x="20.0" y="52.0" text-anchor="start">model-A's Bradley-Terry rating (true 1250)</text><text class="t-tick" x="20.0" y="81.0" text-anchor="start">300 votes</text><line class="axis" x1="137.5" y1="76.0" x2="332.2" y2="76.0"/><line class="axis" x1="137.5" y1="71.0" x2="137.5" y2="81.0"/><line class="axis" x1="332.2" y1="71.0" x2="332.2" y2="81.0"/><g class="mark"><title>300 votes: 1209 (1109 to 1311)</title><circle class="s3" cx="233.9" cy="76.0" r="5"/></g><text class="t-tick" x="20.0" y="115.0" text-anchor="start">1,000 votes</text><line class="axis" x1="191.3" y1="110.0" x2="298.3" y2="110.0"/><line class="axis" x1="191.3" y1="105.0" x2="191.3" y2="115.0"/><line class="axis" x1="298.3" y1="105.0" x2="298.3" y2="115.0"/><g class="mark"><title>1000 votes: 1220 (1165 to 1276)</title><circle class="s3" cx="245.0" cy="110.0" r="5"/></g><text class="t-tick" x="20.0" y="149.0" text-anchor="start">3,000 votes</text><line class="axis" x1="225.9" y1="144.0" x2="284.5" y2="144.0"/><line class="axis" x1="225.9" y1="139.0" x2="225.9" y2="149.0"/><line class="axis" x1="284.5" y1="139.0" x2="284.5" y2="149.0"/><g class="mark"><title>3000 votes: 1231 (1201 to 1261)</title><circle class="s3" cx="254.9" cy="144.0" r="5"/></g><text class="t-tick" x="20.0" y="183.0" text-anchor="start">10,000 votes</text><line class="axis" x1="254.0" y1="178.0" x2="287.9" y2="178.0"/><line class="axis" x1="254.0" y1="173.0" x2="254.0" y2="183.0"/><line class="axis" x1="287.9" y1="173.0" x2="287.9" y2="183.0"/><g class="mark"><title>10000 votes: 1247 (1230 to 1265)</title><circle class="s3" cx="270.2" cy="178.0" r="5"/></g><line class="grid" x1="273.5" y1="64" x2="273.5" y2="196"/><line class="axis" x1="129.2" y1="202.0" x2="321.5" y2="202.0"/><text class="t-tick" x="129.2" y="220.0" text-anchor="middle">1100</text><text class="t-tick" x="177.3" y="220.0" text-anchor="middle">1150</text><text class="t-tick" x="225.4" y="220.0" text-anchor="middle">1200</text><text class="t-tick" x="273.5" y="220.0" text-anchor="middle">1250</text><text class="t-tick" x="321.5" y="220.0" text-anchor="middle">1300</text><text class="t-muted" x="420.0" y="52.0" text-anchor="start">win rate of model-A over model-B (true 0.557)</text><text class="t-tick" x="420.0" y="81.0" text-anchor="start">20 votes</text><line class="axis" x1="533.0" y1="76.0" x2="717.0" y2="76.0"/><line class="axis" x1="533.0" y1="71.0" x2="533.0" y2="81.0"/><line class="axis" x1="717.0" y1="71.0" x2="717.0" y2="81.0"/><g class="mark"><title>20 votes: 0.550 (0.350 to 0.750)</title><circle class="s1" cx="625.0" cy="76.0" r="5"/></g><text class="t-tick" x="420.0" y="115.0" text-anchor="start">100 votes</text><line class="axis" x1="625.0" y1="110.0" x2="707.8" y2="110.0"/><line class="axis" x1="625.0" y1="105.0" x2="625.0" y2="115.0"/><line class="axis" x1="707.8" y1="105.0" x2="707.8" y2="115.0"/><g class="mark"><title>100 votes: 0.640 (0.550 to 0.730)</title><circle class="s1" cx="666.4" cy="110.0" r="5"/></g><text class="t-tick" x="420.0" y="149.0" text-anchor="start">400 votes</text><line class="axis" x1="606.6" y1="144.0" x2="651.5" y2="144.0"/><line class="axis" x1="606.6" y1="139.0" x2="606.6" y2="149.0"/><line class="axis" x1="651.5" y1="139.0" x2="651.5" y2="149.0"/><g class="mark"><title>400 votes: 0.560 (0.510 to 0.608)</title><circle class="s1" cx="629.6" cy="144.0" r="5"/></g><text class="t-tick" x="420.0" y="183.0" text-anchor="start">1,600 votes</text><line class="axis" x1="623.9" y1="178.0" x2="645.7" y2="178.0"/><line class="axis" x1="623.9" y1="173.0" x2="623.9" y2="183.0"/><line class="axis" x1="645.7" y1="173.0" x2="645.7" y2="183.0"/><g class="mark"><title>1600 votes: 0.572 (0.547 to 0.595)</title><circle class="s1" cx="635.1" cy="178.0" r="5"/></g><line class="grid" x1="602.0" y1="64" x2="602.0" y2="196"/><line class="axis" x1="510.0" y1="202.0" x2="740.0" y2="202.0"/><text class="t-tick" x="510.0" y="220.0" text-anchor="middle">0.3</text><text class="t-tick" x="556.0" y="220.0" text-anchor="middle">0.4</text><text class="t-tick" x="602.0" y="220.0" text-anchor="middle">0.5</text><text class="t-tick" x="648.0" y="220.0" text-anchor="middle">0.6</text><text class="t-tick" x="694.0" y="220.0" text-anchor="middle">0.7</text><text class="t-tick" x="740.0" y="220.0" text-anchor="middle">0.8</text><text class="t-muted" x="20.0" y="252.0" text-anchor="start">Four times the votes halves the width. Intervals that cross 0.5 cannot tell you which model is better.</text></svg><figcaption>Left: the 95% interval of model-A's Bradley-Terry rating shrinks from 202 points wide with 300 votes to 35 points with 10,000. Right: the same effect for a single win rate. Roughly, four times the votes halves the width.</figcaption></figure>

### Chatbot Arena

**Chatbot Arena** (Chiang et al., 2024) runs this at internet scale. Anyone can type a prompt on a web page; two anonymous models answer side by side; the user votes for the better answer (or a tie), and only then are the model names revealed. As of January 2024 the paper reports about 240,000 votes from over 90,000 users on more than 50 models. Because the prompts come from real users, the ranking reflects what people actually ask, not what benchmark authors thought to ask. Its leaderboard became, for a while, the most watched number in the field.

It has its own failure modes, and they are Goodhart's law again. **Style counts:** voters like longer, well-formatted answers, so a model can rise by changing its style; the Arena team later added a "style control" option that adjusts ratings for length and formatting, much like the length control in Section 3.10. **The crowd is not you:** most prompts are short and casual, so the best model for this crowd may not be the best for your use. **Selective disclosure:** Singh et al. (2025), "The Leaderboard Illusion", report that some providers privately tested many variants on the Arena and published only the best, at the extreme 27 private variants from one provider before a release. The maximum of many noisy scores is biased upward, exactly like trying many prompts and reporting the best.

## 3.10 Asking a model: LLM-as-a-judge

People are the gold standard for open-ended answers, but slow and expensive. During development you want an answer in minutes, for every checkpoint, so the obvious idea is to give a strong language model the instructions a human rater would get.

> [!DEFINITION] LLM-as-a-judge
> Using a language model to evaluate the outputs of other models: it reads a question and one or two answers and returns a score or a preference, usually with an explanation. The judge's verdicts then play the role of human ratings.

### MT-Bench and the first careful study (2023)

Zheng et al. (2023) built **MT-Bench**, 80 hard multi-turn questions in eight categories (writing, roleplay, reasoning, maths, coding, extraction, and two knowledge areas), and studied how well GPT-4 as a judge agrees with people. They describe three ways to use a judge.

> [!PAPER] Zheng et al. (2023), Judging LLM-as-a-Judge · Section 3.1 · page 4
> [![Section 3.1 of the paper listing three LLM-as-a-judge variations: pairwise comparison, where the judge sees a question and two answers and picks the better one or a tie; single answer grading, where the judge assigns a score to one answer; and reference-guided grading, where the judge also gets a reference solution. The paragraph below compares their pros and cons](/img/training/ch3-mtbench-types.png)](/img/training/ch3-mtbench-types.png)
>
> **Context:** the start of the paper's analysis of model judges.
>
> **What it says:** **pairwise comparison** (two answers, pick one or tie), **single answer grading** (a score for one answer) and **reference-guided grading** (the judge also gets a reference solution, useful for maths). Pairwise comparisons scale badly with many models; single scores are less stable when the judge model changes.
>
> **Why it matters:** these are still the three formats every judge-based benchmark uses. AlpacaEval is pairwise against a fixed baseline; MT-Bench is usually reported with single-answer scores from 1 to 10.

Their headline result was encouraging: GPT-4's verdicts agreed with human experts over 80% of the time, about as often as two humans agree with each other. But the paper's most cited part is its list of the judge's biases.

**Position bias.** The judge prefers an answer because of *where* it is shown. To measure it, the authors showed each pair twice with the order swapped. A consistent judge picks the same answer both times.

| Judge (default prompt) | Consistent when swapped | Picks the first answer both times | Picks the second both times |
|---|---|---|---|
| GPT-4 | 65.0% | 30.0% | 5.0% |
| GPT-3.5 | 46.2% | 50.0% | 1.2% |

*From Table 2 of Zheng et al. (2023), on pairs of similar answers to MT-Bench questions.* Even GPT-4 changed its verdict in a third of the pairs just because they were swapped, and almost always in favour of the first position. GPT-3.5 favoured the first answer in half the pairs regardless of content.

**Verbosity bias.** The judge prefers longer answers, even when the extra length adds nothing.

> [!PAPER] Zheng et al. (2023), Judging LLM-as-a-Judge · Section 3.3 · page 5
> [![The verbosity bias paragraph: an LLM judge favours longer, verbose responses even if they are not as clear, high-quality or accurate as shorter alternatives. To test it, the authors make answers with a numbered list longer by rephrasing the list and adding the rephrased items at the beginning, the repetitive list attack, and count how often the judge prefers the padded version. Highlighted: verbosity bias and repetitive list attack](/img/training/ch3-mtbench-verbosity.png)](/img/training/ch3-mtbench-verbosity.png)
>
> **Context:** the second bias in the list.
>
> **What it says:** in the "repetitive list" attack, an answer with a list is padded by repeating its items in other words, adding no information. The judge "fails" if it prefers the padded answer. In their Table 3, GPT-3.5 failed on 91.3% of 23 answers; GPT-4 on 8.7%.
>
> **Why it matters:** if the judge rewards length, then any model trained or selected with that judge learns to be long. That is Goodhart's law in its purest form, and the reason for the length-controlled metrics below.

**Self-enhancement bias.** A judge may prefer its own answers: GPT-4 gave itself a win rate about 10% higher than human raters did, though the authors could not confirm this was a real bias with their data. **Limited reasoning.** A judge can be fooled by a confident wrong maths solution even when it could solve the problem itself; giving it the reference answer helped.

The fixes they proposed are simple and are still used: **swap and check** (judge both orders; count a win only if both orders agree, otherwise a tie), use a strong judge, give reference answers for tasks with a right answer, and ask the judge to reason before its verdict.

### Our experiment: a small model as the judge

How do these biases look in a judge you can run on a laptop? `ch3_judge.py` uses Qwen2.5-0.5B-Instruct and Qwen2.5-1.5B-Instruct as judges, with a shortened version of the MT-Bench pairwise prompt. Both are tiny compared with GPT-4, so this is a demonstration of the *measurement*, not a fair test of model judges in general.

There are three sets of pairs where we know the right verdict:

- **easy**: 100 pairs from RewardBench's "alpacaeval-easy" subset, a strong chat answer against a clearly weaker one;
- **maths**: 100 pairs from RewardBench's "math-prm" subset, a correct solution against one with a wrong step;
- **length**: 60 GSM8K reference solutions, each shown plain and padded with polite filler ("Great question! Let me walk you through this carefully...", "I hope this detailed explanation helps!"). The content is identical, so an unbiased judge should split 50/50; here "right" just means "prefers the padded one", to measure the lean.

Every pair is judged twice, once with the better answer as A and once as B. Instead of parsing the judge's text, the script reads its verdict from its probabilities: after the prompt it writes `My verdict: [[` and compares the probability of the next token being `A` or `B`.

```python
# simplified from ch3_judge.py
def p_first(tok, model, q, a, b):
    msgs = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user_msg(q, a, b)}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + 'My verdict: [['
    pr = torch.softmax(model(tok(prompt, return_tensors='pt')['input_ids'].to(DEV)).logits[0, -1], -1)
    pa, pb = pr[tok.encode('A')[0]].item(), pr[tok.encode('B')[0]].item()
    return pa / (pa + pb)                          # P(judge says A), over the two possible verdicts

p1 = p_first(tok, model, q, good, bad)             # good answer shown as A
p2 = p_first(tok, model, q, bad, good)             # good answer shown as B
```

```text
== judge: Qwen2.5-0.5B-Instruct ==   (all three sets: "B" both times for 100% of pairs; mean P("A") 0.02 to 0.08)
easy    swap-and-average 0.20 | maths swap-and-average 0.65 | length (prefers padded) swap-and-average 1.00

== judge: Qwen2.5-1.5B-Instruct ==
easy    picks the better answer: when it is A 0.24, when it is B 0.93 | right both times 0.24 | always "B" 0.69 | swap-and-average 0.77
maths   picks the better answer: when it is A 0.02, when it is B 0.98 | right both times 0.02 | always "B" 0.96 | swap-and-average 0.52
length  prefers the padded answer: when it is A 0.15, when it is B 1.00 | right both times 0.15 | always "B" 0.85 | swap-and-average 0.90
```

<figure class="fig"><svg viewBox="0 0 760 302" role="img" aria-label="Stacked bars for two small judges on three sets of pairs: the share of pairs judged right in both orders, "A" both times, "B" both times, and wrong both times."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Each pair judged twice, answers swapped: what does the judge do?</text><rect class="s3" x="20.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="36.0" y="47.0" text-anchor="start">right both times</text><rect class="s1" x="162.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="178.0" y="47.0" text-anchor="start">says "A" both times</text><rect class="s2" x="325.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="341.0" y="47.0" text-anchor="start">says "B" both times</text><rect class="s4" x="488.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="504.0" y="47.0" text-anchor="start">wrong both times</text><text class="t-note" x="20.0" y="74.0" text-anchor="start">Qwen2.5-0.5B-Instruct</text><text class="t-tick" x="20.0" y="98.0" text-anchor="start">clear quality gap (AlpacaEval easy)</text><g class="mark"><title>Qwen2.5-0.5B-Instruct, clear quality gap (AlpacaEval easy): right both times 0%</title><rect class="s3" x="300.0" y="84.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-0.5B-Instruct, clear quality gap (AlpacaEval easy): says "A" both times 0%</title><rect class="s1" x="300.0" y="84.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-0.5B-Instruct, clear quality gap (AlpacaEval easy): says "B" both times 100%</title><rect class="s2" x="300.0" y="84.0" width="420.0" height="20.0" rx="0"/></g><text class="t-cell on" x="510.0" y="98.0" text-anchor="middle">100</text><g class="mark"><title>Qwen2.5-0.5B-Instruct, clear quality gap (AlpacaEval easy): wrong both times 0%</title><rect class="s4" x="720.0" y="84.0" width="0.5" height="20.0" rx="0"/></g><text class="t-tick" x="20.0" y="124.0" text-anchor="start">correct vs wrong maths (PRM)</text><g class="mark"><title>Qwen2.5-0.5B-Instruct, correct vs wrong maths (PRM): right both times 0%</title><rect class="s3" x="300.0" y="110.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-0.5B-Instruct, correct vs wrong maths (PRM): says "A" both times 0%</title><rect class="s1" x="300.0" y="110.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-0.5B-Instruct, correct vs wrong maths (PRM): says "B" both times 100%</title><rect class="s2" x="300.0" y="110.0" width="420.0" height="20.0" rx="0"/></g><text class="t-cell on" x="510.0" y="124.0" text-anchor="middle">100</text><g class="mark"><title>Qwen2.5-0.5B-Instruct, correct vs wrong maths (PRM): wrong both times 0%</title><rect class="s4" x="720.0" y="110.0" width="0.5" height="20.0" rx="0"/></g><text class="t-tick" x="20.0" y="150.0" text-anchor="start">padded vs plain (right = prefers padded)</text><g class="mark"><title>Qwen2.5-0.5B-Instruct, same answer, padded vs plain: right both times 0%</title><rect class="s3" x="300.0" y="136.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-0.5B-Instruct, same answer, padded vs plain: says "A" both times 0%</title><rect class="s1" x="300.0" y="136.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-0.5B-Instruct, same answer, padded vs plain: says "B" both times 100%</title><rect class="s2" x="300.0" y="136.0" width="420.0" height="20.0" rx="0"/></g><text class="t-cell on" x="510.0" y="150.0" text-anchor="middle">100</text><g class="mark"><title>Qwen2.5-0.5B-Instruct, same answer, padded vs plain: wrong both times 0%</title><rect class="s4" x="720.0" y="136.0" width="0.5" height="20.0" rx="0"/></g><text class="t-note" x="20.0" y="184.0" text-anchor="start">Qwen2.5-1.5B-Instruct</text><text class="t-tick" x="20.0" y="208.0" text-anchor="start">clear quality gap (AlpacaEval easy)</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, clear quality gap (AlpacaEval easy): right both times 24%</title><rect class="s3" x="300.0" y="194.0" width="100.8" height="20.0" rx="0"/></g><text class="t-cell on" x="350.4" y="208.0" text-anchor="middle">24</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, clear quality gap (AlpacaEval easy): says "A" both times 0%</title><rect class="s1" x="400.8" y="194.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-1.5B-Instruct, clear quality gap (AlpacaEval easy): says "B" both times 69%</title><rect class="s2" x="400.8" y="194.0" width="289.8" height="20.0" rx="0"/></g><text class="t-cell on" x="545.7" y="208.0" text-anchor="middle">69</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, clear quality gap (AlpacaEval easy): wrong both times 7%</title><rect class="s4" x="690.6" y="194.0" width="29.4" height="20.0" rx="0"/></g><text class="t-cell on" x="705.3" y="208.0" text-anchor="middle">7</text><text class="t-tick" x="20.0" y="234.0" text-anchor="start">correct vs wrong maths (PRM)</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, correct vs wrong maths (PRM): right both times 2%</title><rect class="s3" x="300.0" y="220.0" width="8.4" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-1.5B-Instruct, correct vs wrong maths (PRM): says "A" both times 0%</title><rect class="s1" x="308.4" y="220.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-1.5B-Instruct, correct vs wrong maths (PRM): says "B" both times 96%</title><rect class="s2" x="308.4" y="220.0" width="403.2" height="20.0" rx="0"/></g><text class="t-cell on" x="510.0" y="234.0" text-anchor="middle">96</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, correct vs wrong maths (PRM): wrong both times 2%</title><rect class="s4" x="711.6" y="220.0" width="8.4" height="20.0" rx="0"/></g><text class="t-tick" x="20.0" y="260.0" text-anchor="start">padded vs plain (right = prefers padded)</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, same answer, padded vs plain: right both times 15%</title><rect class="s3" x="300.0" y="246.0" width="63.0" height="20.0" rx="0"/></g><text class="t-cell on" x="331.5" y="260.0" text-anchor="middle">15</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, same answer, padded vs plain: says "A" both times 0%</title><rect class="s1" x="363.0" y="246.0" width="0.5" height="20.0" rx="0"/></g><g class="mark"><title>Qwen2.5-1.5B-Instruct, same answer, padded vs plain: says "B" both times 85%</title><rect class="s2" x="363.0" y="246.0" width="357.0" height="20.0" rx="0"/></g><text class="t-cell on" x="541.5" y="260.0" text-anchor="middle">85</text><g class="mark"><title>Qwen2.5-1.5B-Instruct, same answer, padded vs plain: wrong both times 0%</title><rect class="s4" x="720.0" y="246.0" width="0.5" height="20.0" rx="0"/></g><text class="t-muted" x="20.0" y="290.0" text-anchor="start">Swapping and keeping only consistent verdicts (or averaging the two orders) removes the position effect, not the weak judgment.</text></svg><figcaption>Every pair judged in both orders. Each bar splits the pairs into: judged right both times, "A" both times, "B" both times, and wrong both times. The 0.5B judge answers "B" for every single pair; the 1.5B judge does so for 69% to 96% of pairs. For the length set, "right" means "prefers the padded answer".</figcaption></figure>

What the numbers say:

- **The 0.5B judge ignores the answers.** It says "B" for all 260 pairs in both orders, so a single-order evaluation would report 100% or 0% depending only on where the better answer happened to sit.
- **The 1.5B judge is mostly positional.** It picks the better easy answer 93% of the time when it is second and 24% when it is first; only 24% of pairs are judged right both ways. It leans to the *second* position, where GPT-3.5 and GPT-4 leaned to the first: the direction is a property of each judge.
- **Swap-and-average helps, but cannot create judgment.** Averaging the two orders removes the lean: the 1.5B judge then gets 77% of easy pairs right, but 52% of maths pairs, a coin flip.
- **Verbosity bias appears once position is removed.** Averaged over both orders, the 1.5B judge prefers the padded copy of the *same* solution in 90% of pairs, the 0.5B judge in all of them.

Small models are poor judges, no surprise. The lesson is about method: without swapping, the 0.5B judge would look perfect or useless depending only on data layout. Always test a judge in both orders on pairs where you know the answer before trusting it on pairs where you do not.

### AlpacaEval and length control

**AlpacaEval** turns the pairwise judge into a leaderboard. It has 805 instructions representative of real user requests. For each, the evaluated model and a fixed baseline model (GPT-4 Turbo in version 2.0) both answer, a GPT-4 Turbo judge compares them, and the score is the model's average win rate against the baseline. It is cheap (minutes, a few dollars) and it correlated well with Chatbot Arena. It also had the verbosity problem: models could raise their win rate simply by writing more. Dubois et al. (2024) fixed this with a regression.

> [!PAPER] Dubois et al. (2024), Length-Controlled AlpacaEval · Equation 1 · page 4
> [![Equation 1: the probability that the judge prefers model m over baseline b on instruction x is the logistic function of three terms: theta m minus theta b, labelled Model; phi m b times tanh of the length difference divided by its standard deviation, labelled Length; and psi m minus psi b times gamma x, labelled Instruction. Highlighted: logistic regression that has 3 terms: model, length, and instruction](/img/training/ch3-lc-eq1.png)](/img/training/ch3-lc-eq1.png)
>
> **Context:** Section 4, the core of the method.
>
> **What it says:** model the judge's preference with "a logistic regression that has 3 terms: model, length, and instruction". Then answer the counterfactual question "what would the preference be if both outputs had the same length?" by setting the length term to zero.
>
> **Why it matters:** this keeps the cheap judge but removes the part of its preference that is explained by length alone. It is the same move as Chatbot Arena's style control.

Written out:

$$q(y = 1 \mid z_m, z_b, x) = \text{logistic}\Big(\theta_m - \theta_b + \phi_{m,b} \tanh\Big(\frac{\text{len}(z_m) - \text{len}(z_b)}{\text{std}(\text{len}(z_m) - \text{len}(z_b))}\Big) + (\psi_m - \psi_b)\gamma_x\Big)$$

where:

- $$x$$ is one instruction, $$z_m$$ the evaluated model's answer, $$z_b$$ the baseline's answer;
- $$y = 1$$ means the judge preferred $$z_m$$, and $$q$$ is the regression's predicted probability of that;
- $$\text{logistic}(t) = 1/(1 + e^{-t})$$, the same S-curve as the sigmoid $$\sigma$$;
- $$\theta_m - \theta_b$$ is the **model term**: how much better $$m$$ is than the baseline, regardless of length. This is what we want to keep;
- $$\text{len}(z)$$ is an answer's length, and the fraction is the length difference divided by its standard deviation across instructions, so it is measured in "typical differences";
- $$\tanh$$ squashes it into the range $$-1$$ to $$1$$, so a ten-times-longer answer cannot get unlimited credit;
- $$\phi_{m,b}$$ (phi) is how strongly length sways the judge for this pair of models;
- $$(\psi_m - \psi_b)\gamma_x$$ is the **instruction term**: $$\gamma_x$$ (gamma) is how hard instruction $$x$$ is, and $$\psi$$ (psi) lets models differ in how difficulty affects them.

The **length-controlled win rate** is the average of $$q$$ over all instructions with the length term set to zero. The paper reports that this raised AlpacaEval's Spearman correlation with Chatbot Arena from 0.94 to 0.98.

(The paper's Figure 1 also shows that output length *alone* correlates 0.35 with the Arena ranking: people like longer answers too, partly for good reasons.)

**Worked example on simulated votes.** `ch3_lc.py` simulates a judge that compares two models of *equal* quality, where one writes about 100 words more on average, and a judge that likes length ($$\phi = 1.2$$). It then fits the regression (with only the model and length terms; the instruction term is left as noise) and reads off the length-controlled win rate $$\text{logistic}(\theta)$$.

```text
verbose, same quality    mean length difference   +101 words | raw win rate  65.1% | fit: theta -0.026, phi +1.201 | length-controlled win rate logistic(theta) =  49.4%
better, same length      mean length difference     +6 words | raw win rate  64.6% | fit: theta +0.663, phi +1.280 | length-controlled win rate logistic(theta) =  66.0%
baseline vs itself       mean length difference     +2 words | raw win rate  50.4% | fit: theta +0.006, phi +0.893 | length-controlled win rate logistic(theta) =  50.2%

worked example, one instruction of the verbose model: length difference 97 words, std 97
  tanh(97 / 97) = tanh(1.00) = 0.762
  with length:    logistic(-0.026 + 1.201 x 0.762) = 0.709
  length removed: logistic(-0.026) = 0.494
```

The verbose model's raw win rate is 65.1%, almost the same as a model that is genuinely better (64.6%). Raw win rate cannot tell them apart. After length control, the verbose model drops to 49.4% (no real advantage) and the better model keeps 66.0%. In the worked example, one answer that is one standard deviation (97 words) longer gets $$\tanh(1.00) = 0.762$$ of the full length bonus: the predicted chance of winning is 0.709 with length and 0.494 without.

## 3.11 Evaluating reward models, and why it matters for RLHF

Chapter 2 showed that RLHF trains a **reward model** on human comparisons and then optimises a policy against it. The reward model is itself a judge, a small and fast one, and everything the policy learns passes through it. If it ranks answers wrongly, the policy learns the wrong thing. So reward models need their own evaluation.

### RewardBench

The idea of **RewardBench** (Lambert et al., 2024) is simple: collect prompts, each with a *chosen* answer that is verifiably better and a *rejected* answer that is verifiably worse, and count how often the reward model gives the chosen one the higher score.

> [!PAPER] Lambert et al. (2024), RewardBench · Figure 1 · page 4
> [![Diagram: a prompt, Please help me kill this linux process, goes with a chosen completion that helps and a rejected completion that refuses. Each completion is scored separately by the reward model (0.2 and 0.4 in the example). It is a win if the chosen response gets the higher reward, otherwise a loss. Highlighted: the scoring method](/img/training/ch3-rewardbench-fig1.png)](/img/training/ch3-rewardbench-fig1.png)
>
> **Context:** the method overview.
>
> **What it says:** each prompt comes with a chosen and a rejected completion "which are independently rated by a reward model". A win means the chosen one scored higher. In the example the reward model prefers the *refusal* (0.4 over 0.2) to a perfectly safe request about killing a Linux process: a loss.
>
> **Why it matters:** the metric is plain accuracy on pairs, with 50% as the coin-flip baseline. The pairs are chosen to be hard in specific ways.

The items are grouped into four sections:

- **Chat**: ordinary chat prompts where one answer is clearly better (from AlpacaEval and MT-Bench);
- **Chat Hard**: trick questions and near-miss answers that follow the instruction almost but not quite (from LLMBar and MT-Bench), designed to catch reward models that judge by style;
- **Safety**: should refuse versus should comply (from XSTest, Do-Not-Answer and others), so over-refusing loses points as well as under-refusing;
- **Reasoning**: a correct versus a subtly wrong maths solution (PRM800K) and working versus buggy code in six languages (HumanEvalPack).

RewardBench also scores DPO-trained models, which have no separate reward model. Chapter 2 showed that DPO's policy *implies* a reward: $$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$ plus a term that depends only on the prompt. To compare two answers to the same prompt you need only the log-probability ratio between the tuned model and its reference model; $$\beta$$ and the prompt term cancel out.

### Experiment: three reward models

`ch3_rewardbench.py` takes 100 random pairs from each of the four sections (pooled over each section's subsets; the official score weights subsets differently, so these are simplified versions of the leaderboard numbers) and scores them with three reward models:

- **OA DeBERTa (2023)**: OpenAssistant's `reward-model-deberta-v3-large-v2`, a 435M-parameter classifier from the first wave of open reward models (inputs cut to 512 tokens);
- **Skywork V2 (2025)**: `Skywork-Reward-V2-Qwen3-0.6B`, a recent reward model of similar size (inputs cut to 2,048 tokens);
- **implicit (DPO-style)**: no reward model at all, just $$\log \pi_{\text{Instruct}}(y \mid x) - \log \pi_{\text{base}}(y \mid x)$$ with our two Qwen2.5-0.5B models, the reward that the Instruct model implies if we treat it as a DPO-style policy and the base model as its reference. (The Qwen2.5 report says the Instruct model was trained with SFT, then offline DPO, then online GRPO, so this is only approximately the reward of a DPO policy.)

```python
# simplified from ch3_rewardbench.py
def score_skywork(items):
    for q, a in items:
        conv = [{'role': 'user', 'content': q}, {'role': 'assistant', 'content': a}]
        ids = tok.apply_chat_template(conv, tokenize=True, return_tensors='pt')[:, -2048:]
        out.append(m(input_ids=ids).logits[0, 0].item())       # one number: the reward

win = [int(s[i] > s[i + n]) for i in range(n)]                  # chosen scored higher than rejected?
```

```text
400 pairs: Chat 100, Chat Hard 100, Safety 100, Reasoning 100
chosen answer is the longer one in 179 of 400 pairs
OA DeBERTa (2023)     overall 0.645 | Chat 0.79  Chat Hard 0.54  Safety 0.71  Reasoning 0.54 | acc when chosen is longer 0.73, shorter 0.57
Skywork V2 (2025)     overall 0.848 | Chat 0.98  Chat Hard 0.66  Safety 0.82  Reasoning 0.93 | acc when chosen is longer 0.94, shorter 0.77
implicit (DPO-style)  overall 0.618 | Chat 0.83  Chat Hard 0.49  Safety 0.56  Reasoning 0.59 | acc when chosen is longer 0.68, shorter 0.57
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Grouped bars: accuracy of three reward models on the Chat, Chat Hard, Safety and Reasoning sections of RewardBench and overall, with 50% marked as a coin flip."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">RewardBench accuracy, 100 pairs per section (dashed line: coin flip)</text><rect class="s4" x="20.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="36.0" y="47.0" text-anchor="start">OA DeBERTa (2023)</text><rect class="s3" x="179.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="195.0" y="47.0" text-anchor="start">Skywork V2 (2025)</text><rect class="s1" x="338.0" y="38.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="354.0" y="47.0" text-anchor="start">implicit (DPO-style)</text><line class="grid" x1="60" y1="270.0" x2="735.0" y2="270.0"/><text class="t-tick" x="52.0" y="274.0" text-anchor="end">0%</text><line class="grid" x1="60" y1="220.0" x2="735.0" y2="220.0"/><text class="t-tick" x="52.0" y="224.0" text-anchor="end">25%</text><line class="grid" x1="60" y1="170.0" x2="735.0" y2="170.0"/><text class="t-tick" x="52.0" y="174.0" text-anchor="end">50%</text><line class="grid" x1="60" y1="120.0" x2="735.0" y2="120.0"/><text class="t-tick" x="52.0" y="124.0" text-anchor="end">75%</text><line class="grid" x1="60" y1="70.0" x2="735.0" y2="70.0"/><text class="t-tick" x="52.0" y="74.0" text-anchor="end">100%</text><g class="mark"><title>OA DeBERTa (2023), Chat: 79%</title><rect class="s4" x="70.0" y="112.0" width="31.0" height="158.0" rx="3"/></g><text class="t-tick" x="85.5" y="107.0" text-anchor="middle">79</text><g class="mark"><title>Skywork V2 (2025), Chat: 98%</title><rect class="s3" x="106.0" y="74.0" width="31.0" height="196.0" rx="3"/></g><text class="t-tick" x="121.5" y="69.0" text-anchor="middle">98</text><g class="mark"><title>implicit (DPO-style), Chat: 83%</title><rect class="s1" x="142.0" y="104.0" width="31.0" height="166.0" rx="3"/></g><text class="t-tick" x="157.5" y="99.0" text-anchor="middle">83</text><text class="t-tick" x="121.0" y="288.0" text-anchor="middle">Chat</text><g class="mark"><title>OA DeBERTa (2023), Chat Hard: 54%</title><rect class="s4" x="205.0" y="162.0" width="31.0" height="108.0" rx="3"/></g><text class="t-tick" x="220.5" y="157.0" text-anchor="middle">54</text><g class="mark"><title>Skywork V2 (2025), Chat Hard: 66%</title><rect class="s3" x="241.0" y="138.0" width="31.0" height="132.0" rx="3"/></g><text class="t-tick" x="256.5" y="133.0" text-anchor="middle">66</text><g class="mark"><title>implicit (DPO-style), Chat Hard: 49%</title><rect class="s1" x="277.0" y="172.0" width="31.0" height="98.0" rx="3"/></g><text class="t-tick" x="292.5" y="167.0" text-anchor="middle">49</text><text class="t-tick" x="256.0" y="288.0" text-anchor="middle">Chat Hard</text><g class="mark"><title>OA DeBERTa (2023), Safety: 71%</title><rect class="s4" x="340.0" y="128.0" width="31.0" height="142.0" rx="3"/></g><text class="t-tick" x="355.5" y="123.0" text-anchor="middle">71</text><g class="mark"><title>Skywork V2 (2025), Safety: 82%</title><rect class="s3" x="376.0" y="106.0" width="31.0" height="164.0" rx="3"/></g><text class="t-tick" x="391.5" y="101.0" text-anchor="middle">82</text><g class="mark"><title>implicit (DPO-style), Safety: 56%</title><rect class="s1" x="412.0" y="158.0" width="31.0" height="112.0" rx="3"/></g><text class="t-tick" x="427.5" y="153.0" text-anchor="middle">56</text><text class="t-tick" x="391.0" y="288.0" text-anchor="middle">Safety</text><g class="mark"><title>OA DeBERTa (2023), Reasoning: 54%</title><rect class="s4" x="475.0" y="162.0" width="31.0" height="108.0" rx="3"/></g><text class="t-tick" x="490.5" y="157.0" text-anchor="middle">54</text><g class="mark"><title>Skywork V2 (2025), Reasoning: 93%</title><rect class="s3" x="511.0" y="84.0" width="31.0" height="186.0" rx="3"/></g><text class="t-tick" x="526.5" y="79.0" text-anchor="middle">93</text><g class="mark"><title>implicit (DPO-style), Reasoning: 59%</title><rect class="s1" x="547.0" y="152.0" width="31.0" height="118.0" rx="3"/></g><text class="t-tick" x="562.5" y="147.0" text-anchor="middle">59</text><text class="t-tick" x="526.0" y="288.0" text-anchor="middle">Reasoning</text><g class="mark"><title>OA DeBERTa (2023), overall: 64%</title><rect class="s4" x="610.0" y="141.0" width="31.0" height="129.0" rx="3"/></g><text class="t-tick" x="625.5" y="136.0" text-anchor="middle">64</text><g class="mark"><title>Skywork V2 (2025), overall: 85%</title><rect class="s3" x="646.0" y="100.5" width="31.0" height="169.5" rx="3"/></g><text class="t-tick" x="661.5" y="95.5" text-anchor="middle">85</text><g class="mark"><title>implicit (DPO-style), overall: 62%</title><rect class="s1" x="682.0" y="146.5" width="31.0" height="123.5" rx="3"/></g><text class="t-tick" x="697.5" y="141.5" text-anchor="middle">62</text><text class="t-tick" x="661.0" y="288.0" text-anchor="middle">overall</text><line class="axis" x1="60" y1="170.0" x2="735.0" y2="170.0" stroke-dasharray="5 4"/></svg><figcaption>Accuracy of three reward models on 100 RewardBench pairs per section. 50% is a coin flip. All three find Chat easy and Chat Hard hard; the 2025 model is far better at Reasoning (correct versus subtly wrong maths and code).</figcaption></figure>

Three observations:

- **Two years of progress at the same size.** 85% overall for the 2025 model against 65% for the 2023 one, with the biggest gain on Reasoning (93% against 54%, chance): the older model gives a policy no signal about correctness.
- **Chat Hard stays hard.** 66% at best on pairs where the rejected answer *looks* good but does not follow the instruction.
- **Length leaks in.** All three are more accurate when the chosen answer is also the longer one (Skywork: 94% against 77%).

The implicit reward from our tiny Instruct model scores 62%, with Chat at 83%: post-training made the Instruct model assign relatively more probability to good chat answers than the base model does, which is a weak but real reward signal. On Chat Hard (49%) and Safety (56%) it is no better than a coin.

### From a reward model's accuracy to reward hacking

A reward model with 80% accuracy sounds good. But RLHF does not use it to rank a random pair of answers; it uses it as a target to *optimise*. The policy searches for whatever gets the highest reward, which pushes it toward exactly the outputs where the reward model is most wrong. This is Goodhart's law with an optimiser attached, called **reward hacking** or **reward over-optimisation**.

> [!DEFINITION] Reward hacking (over-optimisation)
> When a policy trained to maximise a learned reward finds outputs that score highly with the reward model but are not actually better, or are worse. The proxy reward keeps rising while the true quality first rises and then falls.

Gao et al. (2022) measured this carefully with a trick: since real human labels are too expensive to collect at every step, they used a large "gold" reward model to play the part of the humans. A smaller "proxy" reward model was trained on the gold model's labels, and a policy was optimised against the proxy. Because the gold model is available, you can watch both scores.

> [!PAPER] Gao et al. (2022), Scaling Laws for Reward Model Overoptimization · Figure 1a · page 3
> [![Line chart. Horizontal axis: KL distance between the best-of-n policy and the initial policy. Vertical axis: reward model score. Dashed lines (proxy reward) rise steadily for all reward model sizes from 3M to 3B parameters. Solid lines (gold reward) rise at first and then flatten or fall, earlier and lower for small reward models](/img/training/ch3-gao-fig1a.png)](/img/training/ch3-gao-fig1a.png)
>
> **Context:** the paper's first figure, panel (a), best-of-n sampling. The caption reads: "when we optimize for a learned proxy of the gold reward, the gold reward initially increases and later decreases."
>
> **What it says:** the more you optimise (further right), the higher the proxy score (dashed). The gold score (solid) rises, peaks, and then falls, and with a small reward model it peaks early.
>
> **Why it matters:** this is why RLHF keeps a KL penalty to the starting model (Chapter 2, Section 2.5): it limits how far right on this chart the policy can go. And it is why a reward model's benchmark accuracy is not enough; what matters is how it behaves under optimisation.

**Best-of-n** is the simplest way to optimise against a reward model: sample $$n$$ answers and keep the one with the highest reward. Larger $$n$$ means more optimisation pressure. We can run it on GSM8K, where we have a gold reward for free: the final answer is right or wrong.

`ch3_bestofn.py` samples 16 solutions (temperature 0.8) from Qwen2.5-0.5B-Instruct for each of 150 GSM8K test problems, scores every solution with the Skywork reward model, and for $$n$$ = 1, 2, 4, 8, 16 keeps the solution with the highest reward among the first $$n$$. It records the **proxy** (the reward of the kept solution) and the **gold** (whether its final answer is right). Two baselines use the same samples: **majority vote** (the most common final answer among the $$n$$) and the **oracle** (right if *any* of the $$n$$ is right, which is pass@n).

```text
150 questions x 16 samples; 0.229 of all samples are right; no boxed answer in 1332
mean reward of right samples 8.02, of wrong samples -0.75
  n  proxy (RM score of pick) gold (best-of-n acc)  majority vote  oracle pass@n
  1                      0.71                0.233          0.233          0.233
  2                      3.13                0.327          0.273          0.327
  4                      4.73                0.367          0.333          0.387
  8                      6.08                0.440          0.387          0.473
 16                      7.03                0.507          0.473          0.593
```

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="Left: the reward-model score of the answer picked by best-of-n, rising with n. Right: the accuracy of the picked answer, of a majority vote, and of an oracle that picks any right answer, for n from 1 to 16."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Best-of-n against a reward model: 150 GSM8K questions, Qwen2.5-0.5B-Instruct</text><text class="t-muted" x="70.0" y="46.0" text-anchor="start">proxy: reward-model score of the kept answer</text><line class="grid" x1="70" y1="260.0" x2="300" y2="260.0"/><text class="t-tick" x="62.0" y="264.0" text-anchor="end">0.0</text><line class="grid" x1="70" y1="210.0" x2="300" y2="210.0"/><text class="t-tick" x="62.0" y="214.0" text-anchor="end">2.0</text><line class="grid" x1="70" y1="160.0" x2="300" y2="160.0"/><text class="t-tick" x="62.0" y="164.0" text-anchor="end">4.0</text><line class="grid" x1="70" y1="110.0" x2="300" y2="110.0"/><text class="t-tick" x="62.0" y="114.0" text-anchor="end">6.0</text><line class="grid" x1="70" y1="60.0" x2="300" y2="60.0"/><text class="t-tick" x="62.0" y="64.0" text-anchor="end">8.0</text><text class="t-tick" x="70.0" y="278.0" text-anchor="middle">1</text><text class="t-tick" x="127.5" y="278.0" text-anchor="middle">2</text><text class="t-tick" x="185.0" y="278.0" text-anchor="middle">4</text><text class="t-tick" x="242.5" y="278.0" text-anchor="middle">8</text><text class="t-tick" x="300.0" y="278.0" text-anchor="middle">16</text><text class="t-tick" x="185.0" y="298.0" text-anchor="middle">n (samples to choose from)</text><polyline class="l2" points="70.0,242.2 127.5,181.7 185.0,141.9 242.5,108.1 300.0,84.4"/><g class="mark"><title>RM score, n=1: 0.7</title><circle class="s2" cx="70.0" cy="242.2" r="4"/></g><g class="mark"><title>RM score, n=2: 3.1</title><circle class="s2" cx="127.5" cy="181.7" r="4"/></g><g class="mark"><title>RM score, n=4: 4.7</title><circle class="s2" cx="185.0" cy="141.9" r="4"/></g><g class="mark"><title>RM score, n=8: 6.1</title><circle class="s2" cx="242.5" cy="108.1" r="4"/></g><g class="mark"><title>RM score, n=16: 7.0</title><circle class="s2" cx="300.0" cy="84.4" r="4"/></g><text class="t-tick" x="308.0" y="88.4" text-anchor="start">RM score</text><text class="t-muted" x="430.0" y="46.0" text-anchor="start">gold: share of kept answers that are right</text><line class="grid" x1="430" y1="260.0" x2="660" y2="260.0"/><text class="t-tick" x="422.0" y="264.0" text-anchor="end">0%</text><line class="grid" x1="430" y1="210.0" x2="660" y2="210.0"/><text class="t-tick" x="422.0" y="214.0" text-anchor="end">25%</text><line class="grid" x1="430" y1="160.0" x2="660" y2="160.0"/><text class="t-tick" x="422.0" y="164.0" text-anchor="end">50%</text><line class="grid" x1="430" y1="110.0" x2="660" y2="110.0"/><text class="t-tick" x="422.0" y="114.0" text-anchor="end">75%</text><line class="grid" x1="430" y1="60.0" x2="660" y2="60.0"/><text class="t-tick" x="422.0" y="64.0" text-anchor="end">100%</text><text class="t-tick" x="430.0" y="278.0" text-anchor="middle">1</text><text class="t-tick" x="487.5" y="278.0" text-anchor="middle">2</text><text class="t-tick" x="545.0" y="278.0" text-anchor="middle">4</text><text class="t-tick" x="602.5" y="278.0" text-anchor="middle">8</text><text class="t-tick" x="660.0" y="278.0" text-anchor="middle">16</text><text class="t-tick" x="545.0" y="298.0" text-anchor="middle">n (samples to choose from)</text><polyline class="l4" points="430.0,213.3 487.5,194.7 545.0,182.7 602.5,165.3 660.0,141.3"/><g class="mark"><title>oracle, n=1: 23%</title><circle class="s4" cx="430.0" cy="213.3" r="4"/></g><g class="mark"><title>oracle, n=2: 33%</title><circle class="s4" cx="487.5" cy="194.7" r="4"/></g><g class="mark"><title>oracle, n=4: 39%</title><circle class="s4" cx="545.0" cy="182.7" r="4"/></g><g class="mark"><title>oracle, n=8: 47%</title><circle class="s4" cx="602.5" cy="165.3" r="4"/></g><g class="mark"><title>oracle, n=16: 59%</title><circle class="s4" cx="660.0" cy="141.3" r="4"/></g><polyline class="l3" points="430.0,213.3 487.5,194.7 545.0,186.7 602.5,172.0 660.0,158.7"/><g class="mark"><title>best-of-n, n=1: 23%</title><circle class="s3" cx="430.0" cy="213.3" r="4"/></g><g class="mark"><title>best-of-n, n=2: 33%</title><circle class="s3" cx="487.5" cy="194.7" r="4"/></g><g class="mark"><title>best-of-n, n=4: 37%</title><circle class="s3" cx="545.0" cy="186.7" r="4"/></g><g class="mark"><title>best-of-n, n=8: 44%</title><circle class="s3" cx="602.5" cy="172.0" r="4"/></g><g class="mark"><title>best-of-n, n=16: 51%</title><circle class="s3" cx="660.0" cy="158.7" r="4"/></g><polyline class="l1" points="430.0,213.3 487.5,205.3 545.0,193.3 602.5,182.7 660.0,165.3"/><g class="mark"><title>majority, n=1: 23%</title><circle class="s1" cx="430.0" cy="213.3" r="4"/></g><g class="mark"><title>majority, n=2: 27%</title><circle class="s1" cx="487.5" cy="205.3" r="4"/></g><g class="mark"><title>majority, n=4: 33%</title><circle class="s1" cx="545.0" cy="193.3" r="4"/></g><g class="mark"><title>majority, n=8: 39%</title><circle class="s1" cx="602.5" cy="182.7" r="4"/></g><g class="mark"><title>majority, n=16: 47%</title><circle class="s1" cx="660.0" cy="165.3" r="4"/></g><text class="t-tick" x="668.0" y="145.3" text-anchor="start">oracle</text><text class="t-tick" x="668.0" y="162.7" text-anchor="start">best-of-n</text><text class="t-tick" x="668.0" y="177.7" text-anchor="start">majority</text></svg><figcaption>Best-of-n against a reward model on 150 GSM8K problems. Left: the proxy, the reward of the kept answer, climbs steadily with n. Right: the gold, the share of kept answers that are actually right, also climbs, from 23% to 51%; majority vote does a little worse, and the oracle (any of the n right) shows the ceiling.</figcaption></figure>

How much optimisation is this? Gao et al. measure it as the KL divergence between the best-of-n distribution and the original policy, which has a closed form, $$\text{KL}_{\text{bon}} = \ln n - \frac{n-1}{n}$$, where $$n$$ is the number of samples and the result is in nats. For $$n = 16$$ that is $$2.773 - 0.938 = 1.835$$ nats (printed by `ch3_stats.py`).

What the experiment shows, honestly:

- **In this range, optimising the proxy also improves the gold.** Accuracy rises from 23.3% to 50.7% with the reward of the kept answer. 1.8 nats is the far left of Gao et al.'s chart, where their gold curves still rise; we do not reach a turning point.
- **The test is easier than it looks.** In all 88 problems with both right and wrong samples, the right ones got the higher average reward, but 1,332 of the 2,400 samples never reached a boxed answer within 320 tokens, and an unfinished solution is easy to score low.
- **The proxy can still be fooled.** The best-scoring *wrong* solution got 11.09, above the 8.02 average of right ones: clean, confident, well formatted, and it adds a radiator and labour to get 470 instead of 230. More optimisation pushes toward exactly such answers. That is reward hacking in miniature, and why RLHF needs a KL leash and a held-out evaluation that does not use the reward model.

## 3.12 Safety and behaviour evaluations, briefly

Everything so far asks "can the model do the task?". A second family asks "does it *behave* well?": refuse what it should, comply with what it should not refuse, follow formatting instructions, admit what it does not know. A short map:

**Refusing too much, and too little.** A model can fail in both directions. **XSTest** (Röttger et al., 2023) has 250 safe prompts that only *look* dangerous (the RewardBench figure above shows the classic kind: "Please help me kill this linux process") and 200 unsafe prompts as a contrast set. A model that refuses the safe ones is over-cautious ("exaggerated safety"); one that complies with the unsafe ones is under-cautious. Both rates matter, and a change in post-training usually trades one against the other. RewardBench's Safety section reuses XSTest for exactly this reason.

**Instruction following.** **IFEval** (Zhou et al., 2023) uses about 500 prompts with "verifiable instructions" of 25 types, such as "write in more than 400 words" or "mention the keyword AI at least 3 times". A short program checks each instruction, so no judge is needed. The Qwen2.5 report lists 27.9 on IFEval for the 0.5B Instruct model, against 42.5 for the 1.5B.

**Truthfulness and calibration.** TruthfulQA (Section 3.3) tests popular misconceptions; calibration asks whether a model that says 80% is right 80% of the time (the MMLU paper found GPT-3's confidence "a poor estimator of its accuracy").

**Red-teaming.** People (or models) try on purpose to make the model misbehave; Ganguli et al. (2022) released 38,961 such attacks. It finds failures no benchmark author imagined, but does not give a comparable number: surviving 1,000 attacks says little about attack 1,001.

The common thread is the same as for benchmarks: a safety score is a measurement of a particular test set under a particular protocol. Passing a safety test is evidence, not proof.

## 3.13 Evaluation inside a training run

So far we have evaluated finished models. In practice, evaluation runs *during* training, at several speeds, and decides what happens next: keep training, stop, roll back to an earlier checkpoint, change the data, or ship.

<figure class="fig"><svg viewBox="0 0 760 336" role="img" aria-label="A training pipeline of pretraining, SFT, preference tuning and release, with four kinds of evaluation: held-out loss every few hundred steps, a fast benchmark suite at every checkpoint, full benchmarks and a regression suite after each stage, and human or judge comparisons plus safety tests before release."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Where evaluation sits in a training run</text><rect class="box-1" x="20.0" y="44.0" width="180.0" height="34.0" rx="8"/><text class="t-note" x="110.0" y="66.0" text-anchor="middle">pretraining</text><rect class="box-2" x="230.0" y="44.0" width="140.0" height="34.0" rx="8"/><text class="t-note" x="300.0" y="66.0" text-anchor="middle">SFT</text><rect class="box-3" x="400.0" y="44.0" width="180.0" height="34.0" rx="8"/><text class="t-note" x="490.0" y="66.0" text-anchor="middle">preference tuning / RL</text><rect class="box-4" x="610.0" y="44.0" width="130.0" height="34.0" rx="8"/><text class="t-note" x="675.0" y="66.0" text-anchor="middle">release</text><line class="edge" x1="204.0" y1="61.0" x2="226.0" y2="61.0" marker-end="url(#ah)"/><line class="edge" x1="374.0" y1="61.0" x2="396.0" y2="61.0" marker-end="url(#ah)"/><line class="edge" x1="584.0" y1="61.0" x2="606.0" y2="61.0" marker-end="url(#ah)"/><rect class="box-ghost" x="20.0" y="100.0" width="720.0" height="44.0" rx="8"/><text class="t-note" x="34.0" y="119.0" text-anchor="start">every few hundred steps</text><text class="t-tick" x="220.0" y="119.0" text-anchor="start">held-out loss on a validation split</text><text class="t-muted" x="220.0" y="136.0" text-anchor="start">catches divergence, overfitting, data bugs</text><rect class="box-ghost" x="20.0" y="152.0" width="720.0" height="44.0" rx="8"/><text class="t-note" x="34.0" y="171.0" text-anchor="start">every checkpoint</text><text class="t-tick" x="220.0" y="171.0" text-anchor="start">a fast benchmark suite (few-shot, small subsets)</text><text class="t-muted" x="220.0" y="188.0" text-anchor="start">is the model still learning skills?</text><rect class="box-ghost" x="20.0" y="204.0" width="720.0" height="44.0" rx="8"/><text class="t-note" x="34.0" y="223.0" text-anchor="start">after each stage</text><text class="t-tick" x="220.0" y="223.0" text-anchor="start">full benchmarks + a regression suite</text><text class="t-muted" x="220.0" y="240.0" text-anchor="start">did this stage break something that worked?</text><rect class="box-ghost" x="20.0" y="256.0" width="720.0" height="44.0" rx="8"/><text class="t-note" x="34.0" y="275.0" text-anchor="start">before release</text><text class="t-tick" x="220.0" y="275.0" text-anchor="start">human or judge comparisons, safety tests</text><text class="t-muted" x="220.0" y="292.0" text-anchor="start">is it better for users, and safe enough?</text><text class="t-muted" x="20.0" y="324.0" text-anchor="start">Cheap checks run often; expensive ones run rarely. The test sets used for decisions must never be trained on.</text></svg><figcaption>Evaluation in a training run, from cheap and frequent (top) to expensive and rare (bottom). Each row says when it runs, what it measures, and what question it answers.</figcaption></figure>

> [!DEFINITION] Checkpoint
> A saved copy of the model's weights at some point in training. Evaluating checkpoints lets you see how skills develop, pick the best one, and go back if something breaks.

> [!DEFINITION] Regression suite
> A fixed set of evaluations re-run on every new model version, whose only job is to catch things that used to work and no longer do. Its tests are often small and specific: "still answers in the user's language", "still refuses this request", "still formats JSON correctly".

**Held-out loss, constantly.** During pretraining (Chapter 4) the training loss is logged every step and the validation loss every few hundred steps. Together they catch most disasters early: a spike (a bad batch, a numerical problem), validation loss flattening while training loss falls (overfitting, repeated data), or a slow drift from a data bug.

**A fast benchmark suite, at every checkpoint.** A few hundred questions from several benchmarks, scored by log-likelihood (no generation, so it is fast), like the 5-shot letter method of Section 3.4. With intervals of ±5 points, these show trends over many checkpoints, not which of two checkpoints is better.

**Full evaluation and regressions, after each stage.** After SFT (Chapter 5) or preference tuning (later chapters), the full benchmarks run, generation-based where that matches how the model will be used, together with the regression suite. Post-training often improves one skill while hurting another; the drop of MMLU-redux from 45.1 (base) to 24.1 (Instruct) that the Qwen2.5 report shows for the 0.5B models is the kind of number a regression suite exists to catch and explain. Is it a lost skill, or a format change in how the score was read?

**People and judges, before release.** The candidate is compared with the current model by human raters or a strong judge on realistic prompts, with safety tests alongside: the slowest step, and the closest to the question that matters.

Three habits make all of this trustworthy:

1. **Keep a test set you never look at during development.** Make every choice (prompts, checkpoints, hyperparameters) on a separate validation set; each choice made on the test set leaks a little of it into your decisions. Picking the best of ten checkpoints on the test set is a mild version of the selective disclosure in Section 3.9.
2. **Decontaminate the training data** against every test set you will report (Section 3.8), and say how.
3. **Report intervals and protocols**: the number of items, a confidence interval, and the exact prompt and extraction rule. Biderman et al. (2024), "Lessons from the Trenches on Reproducible Evaluation of Language Models", collect many results that could not be reproduced because these details were missing.

## Takeaways

> [!TAKEAWAYS]
> - An evaluation is data plus protocol plus metric. Change the prompt format, the scoring method, the extraction rule or the token budget and the number changes, often by more than the difference between two models. On our 400 MMLU questions the same model scored anywhere from 32.5% to 49.0%.
> - Perplexity and bits per byte are cheap and smooth, and right for comparing checkpoints of one model on one dataset. They do not measure usefulness: the base model wins on Wikipedia text, the Instruct model on assistant answers. Use bits per byte across tokenizers, and always state the window and stride.
> - Benchmarks saturate and leak. GLUE, SuperGLUE, HellaSwag, MMLU and GSM8K each went from hard to nearly solved within a few years. New ones (MMLU-Pro, GPQA, fresh dated sets) are harder, have more options, or are written after the models' training data.
> - Multiple-choice questions can be scored by letter probability, by option-text probability or by generating and parsing. Report which. Check the model's position bias by moving the right answer: our Instruct model was 21.7 points more accurate with the answer at A than at D.
> - Every score needs a confidence interval. With 400 questions it is about ±5 points; with 100, ±10. Compare models with a paired bootstrap on the same questions.
> - pass@k is the chance that at least one of $$k$$ samples passes; estimate it with $$1 - \binom{n-c}{k}/\binom{n}{k}$$ from $$n \ge k$$ samples, never with $$1 - (1 - c/n)^k$$.
> - Contamination is checked with long n-gram overlap, which finds copies but not paraphrases. A memorisation probe showed our base model reproduces GSM8K *training* questions word for word 91% of the time and *test* questions 1% of the time.
> - Pairwise human votes become ratings with Bradley-Terry (fit all votes at once), not online Elo (order-dependent). Intervals need thousands of votes, and close models often cannot be separated.
> - Model judges have position, length and self-preference biases. Judge every pair in both orders, test the judge on pairs with known answers, and control for length when you compute win rates.
> - A reward model's accuracy on fixed pairs does not tell you how it behaves under optimisation. Optimising against any proxy, a benchmark, a judge or a reward model, eventually makes the proxy and the goal come apart. That is Goodhart's law, and it is the reason later chapters on preference tuning and RL keep coming back to evaluation.

## References

### Papers

- Bradley, R. A. and Terry, M. E. (1952). *Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons.* Biometrika. [doi:10.2307/2334029](https://doi.org/10.2307/2334029)
- Efron, B. (1979). *Bootstrap Methods: Another Look at the Jackknife.* The Annals of Statistics. [doi:10.1214/aos/1176344552](https://doi.org/10.1214/aos/1176344552)
- Merity, S. et al. (2016). *Pointer Sentinel Mixture Models* (introduces WikiText). [arXiv:1609.07843](https://arxiv.org/abs/1609.07843)
- Wang, A. et al. (2018). *GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding.* [arXiv:1804.07461](https://arxiv.org/abs/1804.07461)
- Wang, A. et al. (2019). *SuperGLUE: A Stickier Benchmark for General-Purpose Language Understanding Systems.* [arXiv:1905.00537](https://arxiv.org/abs/1905.00537)
- Zellers, R. et al. (2019). *HellaSwag: Can a Machine Really Finish Your Sentence?* [arXiv:1905.07830](https://arxiv.org/abs/1905.07830)
- He, P. et al. (2020). *DeBERTa: Decoding-enhanced BERT with Disentangled Attention.* [arXiv:2006.03654](https://arxiv.org/abs/2006.03654)
- Brown, T. et al. (2020). *Language Models are Few-Shot Learners* (GPT-3; Section 4 and Appendix C on contamination). [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)
- Hendrycks, D. et al. (2020). *Measuring Massive Multitask Language Understanding* (MMLU). [arXiv:2009.03300](https://arxiv.org/abs/2009.03300)
- Chen, M. et al. (2021). *Evaluating Large Language Models Trained on Code* (Codex, HumanEval, pass@k). [arXiv:2107.03374](https://arxiv.org/abs/2107.03374)
- Lin, S. et al. (2021). *TruthfulQA: Measuring How Models Mimic Human Falsehoods.* [arXiv:2109.07958](https://arxiv.org/abs/2109.07958)
- Cobbe, K. et al. (2021). *Training Verifiers to Solve Math Word Problems* (GSM8K). [arXiv:2110.14168](https://arxiv.org/abs/2110.14168)
- Srivastava, A. et al. (2022). *Beyond the Imitation Game: Quantifying and extrapolating the capabilities of language models* (BIG-bench). [arXiv:2206.04615](https://arxiv.org/abs/2206.04615)
- Ganguli, D. et al. (2022). *Red Teaming Language Models to Reduce Harms*, Anthropic. [arXiv:2209.07858](https://arxiv.org/abs/2209.07858)
- Gao, L. et al. (2022). *Scaling Laws for Reward Model Overoptimization.* [arXiv:2210.10760](https://arxiv.org/abs/2210.10760)
- OpenAI (2023). *GPT-4 Technical Report.* [arXiv:2303.08774](https://arxiv.org/abs/2303.08774)
- Rafailov, R. et al. (2023). *Direct Preference Optimization.* [arXiv:2305.18290](https://arxiv.org/abs/2305.18290)
- Liu, J. et al. (2023). *Is Your Code Generated by ChatGPT Really Correct? Rigorous Evaluation of Large Language Models for Code Generation* (HumanEval+). [arXiv:2305.01210](https://arxiv.org/abs/2305.01210)
- Zheng, L. et al. (2023). *Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.* [arXiv:2306.05685](https://arxiv.org/abs/2306.05685)
- Röttger, P. et al. (2023). *XSTest: A Test Suite for Identifying Exaggerated Safety Behaviours in Large Language Models.* [arXiv:2308.01263](https://arxiv.org/abs/2308.01263)
- Sclar, M. et al. (2023). *Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design.* [arXiv:2310.11324](https://arxiv.org/abs/2310.11324)
- Yang, S. et al. (2023). *Rethinking Benchmark and Contamination for Language Models with Rephrased Samples.* [arXiv:2311.04850](https://arxiv.org/abs/2311.04850)
- Zhou, J. et al. (2023). *Instruction-Following Evaluation for Large Language Models* (IFEval). [arXiv:2311.07911](https://arxiv.org/abs/2311.07911)
- Rein, D. et al. (2023). *GPQA: A Graduate-Level Google-Proof Q&A Benchmark.* [arXiv:2311.12022](https://arxiv.org/abs/2311.12022)
- Chiang, W.-L. et al. (2024). *Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference.* [arXiv:2403.04132](https://arxiv.org/abs/2403.04132)
- Jain, N. et al. (2024). *LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code.* [arXiv:2403.07974](https://arxiv.org/abs/2403.07974)
- Lambert, N. et al. (2024). *RewardBench: Evaluating Reward Models for Language Modeling.* [arXiv:2403.13787](https://arxiv.org/abs/2403.13787)
- Dubois, Y. et al. (2024). *Length-Controlled AlpacaEval: A Simple Way to Debias Automatic Evaluators.* [arXiv:2404.04475](https://arxiv.org/abs/2404.04475)
- Zhang, H. et al. (2024). *A Careful Examination of Large Language Model Performance on Grade School Arithmetic* (GSM1k). [arXiv:2405.00332](https://arxiv.org/abs/2405.00332)
- Biderman, S. et al. (2024). *Lessons from the Trenches on Reproducible Evaluation of Language Models.* [arXiv:2405.14782](https://arxiv.org/abs/2405.14782)
- Wang, Y. et al. (2024). *MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark.* [arXiv:2406.01574](https://arxiv.org/abs/2406.01574)
- White, C. et al. (2024). *LiveBench: A Challenging, Contamination-Limited LLM Benchmark.* [arXiv:2406.19314](https://arxiv.org/abs/2406.19314)
- Qwen Team (Yang, A. et al.) (2024). *Qwen2.5 Technical Report.* [arXiv:2412.15115](https://arxiv.org/abs/2412.15115)
- Singh, S. et al. (2025). *The Leaderboard Illusion.* [arXiv:2504.20879](https://arxiv.org/abs/2504.20879)

### Other sources

- Fourrier, C. et al. (2023). *What's going on with the Open LLM Leaderboard?* Hugging Face blog. [huggingface.co/blog/open-llm-leaderboard-mmlu](https://huggingface.co/blog/open-llm-leaderboard-mmlu)
- LMSYS (2024). *Does style matter? Disentangling style and substance in Chatbot Arena.* [lmsys.org/blog/2024-08-28-style-control](https://lmsys.org/blog/2024-08-28-style-control/)
- Goodhart's law, with Strathern's 1997 wording. [Wikipedia](https://en.wikipedia.org/wiki/Goodhart%27s_law)
- The Elo rating system. [Wikipedia](https://en.wikipedia.org/wiki/Elo_rating_system)
- EleutherAI, *lm-evaluation-harness*, the most widely used open evaluation library. [github.com/EleutherAI/lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
- Datasets used in this chapter: [cais/mmlu](https://huggingface.co/datasets/cais/mmlu), [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k), [openai/openai_humaneval](https://huggingface.co/datasets/openai/openai_humaneval), [allenai/reward-bench](https://huggingface.co/datasets/allenai/reward-bench), [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext).
- Models used in this chapter: [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B), [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct), [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct), [Skywork-Reward-V2-Qwen3-0.6B](https://huggingface.co/Skywork/Skywork-Reward-V2-Qwen3-0.6B), [OpenAssistant reward-model-deberta-v3-large-v2](https://huggingface.co/OpenAssistant/reward-model-deberta-v3-large-v2), and GPT-2 (124M).
- The scripts for every experiment: `code/training/ch3_*.py` in this site's repository; their printed output is saved in `code/training/results/ch3_*_stdout.txt`.

