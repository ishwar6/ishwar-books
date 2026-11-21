---
title: "Distillation: Small Models That Reason"
description: "How 800k answers written by R1 teach six small open models to reason with plain fine-tuning, and why this beat RL on the same model. We write the loss by hand and test a distilled 1.5B model."
part: 6
covers: "§2.4, §3.2, Table 5, §4.1, Table 6"
date: 2026-11-21
tags: [deepseek, distillation, knowledge-distillation, sft, reasoning]
---

DeepSeek-R1 has 671 billion parameters, and few people can run it. Yet its reasoning can be copied into a 1.5-billion-parameter model that runs on a laptop. The small model needs no reinforcement learning: it only reads R1's written answers.

## The promise {§1}

> [!PAPER] · §1 · page 3
> [![The paragraph of the introduction about distillation, with direct distillation from DeepSeek-R1 outperforms applying RL on it, reasoning patterns discovered by larger base models are crucial, and distilled 14B model outperforms state-of-the-art open-source QwQ-32B-Preview highlighted](/img/papers/deepseek-r1/p6-intro.png)](/img/papers/deepseek-r1/p6-intro.png)
>
> **What it says:** on Qwen2.5-32B, "direct distillation from DeepSeek-R1 outperforms applying RL on it". The authors conclude that "the reasoning patterns discovered by larger base models are crucial". The distilled 14B model beats QwQ-32B-Preview, and the distilled 32B and 70B models "set a new record" among dense models.
>
> **What it means:**
>
> The introduction gives the whole result of this part in four sentences. There are two ways to give a 32-billion-parameter model the skill to reason. The first way is the R1-Zero way: reinforcement learning on the model itself. The second way is to let it copy the answers of a bigger model that already reasons. The second way won.
>
> The reason the authors give is in the phrase "reasoning patterns discovered by larger base models". A pattern here is a habit of thought: try a method, check the result, notice an error, go back. R1 found these habits through RL on a 671B model. A small model that copies them does not have to find them again.
>
> "Dense" means that every parameter works on every token. R1 is a mixture-of-experts model, so only a part of it runs for each token. The students are ordinary dense models from the Qwen2.5 and Llama 3 families.

> [!PAPER] · §1.1 · page 4
> [![The Distillation contribution bullets, with reasoning patterns of larger models can be distilled into smaller, better performance compared to the reasoning patterns discovered, and DeepSeek-R1-Distill-Qwen-7B achieves 55.5% on AIME 2024 highlighted](/img/papers/deepseek-r1/p6-contrib.png)](/img/papers/deepseek-r1/p6-contrib.png)
>
> **What it says:** "the reasoning patterns of larger models can be distilled into smaller models, resulting in better performance compared to the reasoning patterns discovered through RL on small models." The distilled 7B gets 55.5% on AIME 2024; the distilled 32B gets 72.6% on AIME 2024 and 94.3% on MATH-500. Six checkpoints are released: 1.5B, 7B, 8B, 14B, 32B and 70B.
>
> **What it means:**
>
> The first bullet is a comparison, not only a claim that distillation works. It puts two sources of reasoning side by side: patterns that a large model found, and patterns that a small model finds by itself. The large model's patterns win. Section 4.1 is the experiment behind this sentence, and the end of this page reads it.
>
> The second bullet is about weights you can download. The six models are on Hugging Face under names such as DeepSeek-R1-Distill-Qwen-1.5B. That is the model we run on every page of this series, and it is the reason a laptop can show what R1 does.

## Background: what distillation is, and where it came from

The word comes from chemistry. You boil a liquid, keep the vapour and cool it again, and you get the essence without the bulk. In machine learning, the "liquid" is a large model, and the "essence" is the behaviour we want a small model to copy.

> [!DEFINITION] Knowledge distillation
> Training a small model (the **student**) to behave like a large trained model (the **teacher**). The student learns from the teacher's outputs, not only from the original labels. The teacher stays frozen; only the student changes.

### Soft labels: the first form

> [!PAPER] Hinton et al. (2015), Distilling the Knowledge in a Neural Network · §1 · page 2
> [![A paragraph of Hinton, Vinyals and Dean, with use the class probabilities produced by the cumbersome model as soft targets highlighted](/img/papers/deepseek-r1/p6-hinton.png)](/img/papers/deepseek-r1/p6-hinton.png)
>
> **What it says:** to move what a large ("cumbersome") model knows into a small one, "use the class probabilities produced by the cumbersome model as 'soft targets'" for the small model. Soft targets carry "much more information per training case" than the hard labels.
>
> **What it means:**
>
> A classifier ends with a list of probabilities, one for each class. The normal label of a photo of a dog says "dog: 1, everything else: 0". That is a **hard label**. The teacher's output says something like "dog: 0.9, wolf: 0.08, cat: 0.02". That is a **soft label**, and the student is trained to match the whole list.
>
> The paper also uses a temperature $$T$$ in the softmax to make the list flatter, so that the small numbers become visible. We do not need the temperature here. The key idea is that the student copies a distribution, not a single answer.

> [!PAPER] Hinton et al. (2015), Distilling the Knowledge in a Neural Network · §1 · page 2
> [![The BMW example of Hinton, Vinyals and Dean, with relative probabilities of incorrect answers, An image of a BMW and mistaken for a garbage truck highlighted](/img/papers/deepseek-r1/p6-hinton-bmw.png)](/img/papers/deepseek-r1/p6-hinton-bmw.png)
>
> **What it says:** "The relative probabilities of incorrect answers" show how the teacher generalises. "An image of a BMW" has a very small chance of being "mistaken for a garbage truck", but that mistake is many times more probable than a mistake for a carrot.
>
> **What it means:**
>
> This is why soft labels help. A hard label says only "BMW". The teacher's probabilities also say "this looks a little like a truck, and not at all like a vegetable". That is knowledge about how the classes relate, and the student gets it for free.
>
> Keep this picture in mind. In R1's case the "classes" are the next tokens of a text. The question is whether the student copies the teacher's full list of probabilities at each token, or only the text that the teacher wrote.

### Sequence-level distillation: copy the text

> [!PAPER] Kim and Rush (2016), Sequence-Level Knowledge Distillation · §1 · page 2
> [![A paragraph of Kim and Rush, with sequence-level (as opposed to word-level) and trained on a newly generated dataset that is the result of running beam search with the teacher network highlighted](/img/papers/deepseek-r1/p6-kimrush.png)](/img/papers/deepseek-r1/p6-kimrush.png)
>
> **What it says:** for translation, the student can match the teacher at the "sequence-level (as opposed to word-level)". In practice the student is "trained on a newly generated dataset that is the result of running beam search with the teacher network".
>
> **What it means:**
>
> A translation model makes a sentence one word at a time. Word-level distillation applies Hinton's idea at every word: the student matches the teacher's probabilities for each next word. Sequence-level distillation is simpler. The teacher writes whole translations, and the student is trained on them as if they were the correct answers.
>
> This turns distillation into ordinary fine-tuning on a new dataset. The student never sees the teacher's probabilities, only its text. Kim and Rush found that this works well, and that the student even needs less search when it generates.

> [!DEFINITION] Sequence-level distillation
> The teacher generates complete outputs for many inputs, and the student is fine-tuned on these outputs with the normal next-token loss. The teacher's text replaces the human-written answers. R1's distillation is of this kind, with sampling and a correctness filter in place of beam search.

### The two losses, side by side

Look at one position of a text. Let $$p_T(v)$$ be the teacher's probability that the next token is $$v$$, and $$p_S(v)$$ the student's. Word-level distillation uses the cross-entropy between the two lists:

$$
\mathcal{L}_{\text{word}} = - \sum_{v} p_T(v) \, \log p_S(v)
$$

Sequence-level distillation keeps only the token $$\hat{y}$$ that the teacher actually wrote at that position:

$$
\mathcal{L}_{\text{seq}} = - \log p_S(\hat{y})
$$

Read them from left to right. The sum runs over every token of the vocabulary. Each term is the teacher's probability for that token times the student's log-probability for it. The second loss has no sum: one token, one log-probability. Both are summed (or averaged) over all positions of the text.

A tiny example. Take a vocabulary of three tokens, A, B and C. The teacher gives (0.7, 0.2, 0.1) and the student gives (0.5, 0.3, 0.2). The word-level loss is

$$
-(0.7 \ln 0.5 + 0.2 \ln 0.3 + 0.1 \ln 0.2) = 0.485 + 0.241 + 0.161 = 0.887
$$

Suppose the teacher sampled A. The sequence-level loss is $$-\ln 0.5 = 0.693$$. If it had sampled C, the loss would be $$-\ln 0.2 = 1.609$$. Now sample many times from the teacher: A comes 70% of the time, B 20% and C 10%. The average sequence-level loss is then exactly 0.887, the word-level loss. Sampled text is a noisy but honest copy of the teacher's distribution.

R1's data is not plain samples either. For each prompt the authors sample several answers and keep only the correct ones ([Part 5](part-5-the-r1-pipeline.md) runs this rejection sampling). Version 1 does not give the sampling temperature for this step; the 0.6 of §3 is for evaluation. So the student copies the teacher's good answers, not its full distribution. That is a choice, and it fits the goal: we want the student to reason well, not to copy R1's mistakes.

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="Word-level distillation trains the student on the teacher's whole probability distribution for the next token. Sequence-level distillation, used by R1, trains it on one text the teacher wrote, so each position has a single target token."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Two ways to copy a teacher</text><text class="t-note" x="20.0" y="52.0" text-anchor="start">Word-level (Hinton et al., 2015)</text><text class="t-muted" x="20.0" y="70.0" text-anchor="start">target = the teacher's whole next-token distribution</text><text class="t-tick" x="20.0" y="94.0" text-anchor="start">after "...ay at the farmers' market." (blank line)</text><text class="t-mono" x="110.0" y="123.0" text-anchor="end">'Wait'</text><g class="mark"><title>'Wait': p = 0.900</title><path class="s1" d="M118,111H294.0Q298.0,111 298.0,115V123Q298.0,127 294.0,127H118Z"/></g><text class="t-val" x="304.0" y="124.0" text-anchor="start">0.900</text><text class="t-mono" x="110.0" y="149.0" text-anchor="end">'Let'</text><g class="mark"><title>'Let': p = 0.074</title><path class="s1" d="M118,137H128.8Q132.8,137 132.8,141V149Q132.8,153 128.8,153H118Z"/></g><text class="t-val" x="138.8" y="150.0" text-anchor="start">0.074</text><text class="t-mono" x="110.0" y="175.0" text-anchor="end">'Hmm'</text><g class="mark"><title>'Hmm': p = 0.011</title><path class="s1" d="M118,163H118.0Q120.2,163 120.2,165.2V176.8Q120.2,179 118.0,179H118Z"/></g><text class="t-val" x="126.2" y="176.0" text-anchor="start">0.011</text><text class="t-mono" x="110.0" y="201.0" text-anchor="end">'But'</text><g class="mark"><title>'But': p = 0.005</title><path class="s1" d="M118,189H118Q120,189 120,191V203Q120,205 118,205H118Z"/></g><text class="t-val" x="125.0" y="202.0" text-anchor="start">0.005</text><text class="t-mono" x="110.0" y="227.0" text-anchor="end">'Hold'</text><g class="mark"><title>'Hold': p = 0.005</title><path class="s1" d="M118,215H118Q120,215 120,217V229Q120,231 118,231H118Z"/></g><text class="t-val" x="125.0" y="228.0" text-anchor="start">0.005</text><text class="t-muted" x="20.0" y="250.0" text-anchor="start">the student matches every bar,</text><text class="t-muted" x="20.0" y="266.0" text-anchor="start">at every position of the text</text><text class="t-note" x="380.0" y="52.0" text-anchor="start">Sequence-level (Kim and Rush, 2016; R1)</text><text class="t-muted" x="380.0" y="70.0" text-anchor="start">target = one text the teacher wrote</text><text class="t-tick" x="380.0" y="94.0" text-anchor="start">the same position in R1-style training data:</text><text class="t-mono" x="470.0" y="123.0" text-anchor="end">'Wait'</text><g class="mark"><title>'Wait': target probability 1</title><path class="s2" d="M478,111H674Q678,111 678,115V123Q678,127 674,127H478Z"/></g><text class="t-val" x="684.0" y="124.0" text-anchor="start">1</text><text class="t-muted" x="380.0" y="160.0" text-anchor="start">The teacher gave 'Wait' p = 0.900; the sampled</text><text class="t-muted" x="380.0" y="176.0" text-anchor="start">text keeps only that token, as a hard label.</text><text class="t-note" x="380.0" y="214.0" text-anchor="start">needs: the teacher's text only</text><text class="t-muted" x="380.0" y="232.0" text-anchor="start">(no logits, any tokenizer, any API)</text><text class="t-muted" x="20.0" y="292.0" text-anchor="start">Left: what a soft-label student sees. Right: what an R1 student sees. Numbers from our run.</text></svg><figcaption>One real position of our duck trace, with R1-Distill-Qwen-1.5B as the teacher, seen both ways. On the left, the teacher's top five next tokens and their probabilities, which a word-level student would match. On the right, the one token that the sampled text keeps, which is all that an R1 student sees.</figcaption></figure>

Why did DeepSeek choose the text and not the probabilities? Three reasons follow from the setup, although the paper does not list them. None of the six students uses R1's tokenizer, so R1's probabilities over its own tokens do not line up with a student's tokens. Storing a full probability list for every token of 800k long answers would be huge, while text is small. And the 800k samples already existed: they were made for R1's own training.

## The data: 800k samples {§2.3.3}

> [!PAPER] · §2.3.3 · page 11
> [![The end of the Non-Reasoning data paragraph, with approximately 200k training samples and about 800k samples highlighted](/img/papers/deepseek-r1/p6-800k.png)](/img/papers/deepseek-r1/p6-800k.png)
>
> **What it says:** besides about 600k reasoning samples, the authors collect "approximately 200k training samples" for writing, factual questions, self-cognition and translation. R1's stage-3 SFT uses the whole set of "about 800k samples".
>
> **What it means:**
>
> [Part 5](part-5-the-r1-pipeline.md) explains how this set was built. In short: an RL checkpoint of R1 wrote many answers for each reasoning prompt, and only the correct and readable ones were kept. The other 200k are a mixture. Some are reused portions of DeepSeek-V3's own fine-tuning data. For some non-reasoning tasks, DeepSeek-V3 was prompted to write new answers, with a short chain of thought first.
>
> The important fact for this part is that the data was made once and used twice. It fine-tunes DeepSeek-V3-Base on the way to R1, and it fine-tunes the six small students. The same text teaches a 671B model and a 1.5B model.

## Distillation: fine-tune small models on R1's answers {§2.4}

<figure class="fig"><svg viewBox="0 0 720 440" role="img" aria-label="Pipeline diagram. DeepSeek-V3-Base goes through the four R1 stages to become DeepSeek-R1. The 800k samples of the rejection sampling stage then train six smaller open models with SFT only: four Qwen2.5 models and two Llama models."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Where distillation sits: R1 writes the data, six small models copy it</text><rect class="box" x="20.0" y="46.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="105.0" y="67.5" text-anchor="middle">DeepSeek-V3-Base</text><line class="edge" x1="105.0" y1="80.0" x2="105.0" y2="98.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="100.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="105.0" y="121.5" text-anchor="middle">cold start SFT</text><line class="edge" x1="105.0" y1="134.0" x2="105.0" y2="152.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="154.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="105.0" y="175.5" text-anchor="middle">reasoning RL</text><line class="edge" x1="105.0" y1="188.0" x2="105.0" y2="206.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="208.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="105.0" y="229.5" text-anchor="middle">rejection sampling</text><line class="edge" x1="105.0" y1="242.0" x2="105.0" y2="260.0" marker-end="url(#ah)"/><rect class="box-1" x="20.0" y="262.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="105.0" y="283.5" text-anchor="middle">SFT + RL for all</text><line class="edge" x1="105.0" y1="296.0" x2="105.0" y2="314.0" marker-end="url(#ah)"/><rect class="box-on" x="20.0" y="316.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="105.0" y="337.5" text-anchor="middle">DeepSeek-R1</text><text class="t-muted" x="105.0" y="384.0" text-anchor="middle">Part 5 (§2.3)</text><rect class="box-on" x="250.0" y="156.0" width="170.0" height="34.0" rx="6"/><text class="t-note" x="335.0" y="177.5" text-anchor="middle">800k samples</text><text class="t-muted" x="335.0" y="210.0" text-anchor="middle">600k reasoning traces</text><text class="t-muted" x="335.0" y="226.0" text-anchor="middle">(rejection sampled)</text><text class="t-muted" x="335.0" y="246.0" text-anchor="middle">200k other: writing,</text><text class="t-muted" x="335.0" y="262.0" text-anchor="middle">factual QA, translation</text><line class="edge-on" x1="190.0" y1="173.0" x2="246.0" y2="173.0" marker-end="url(#ah-on)"/><text class="t-tick" x="218.0" y="165.0" text-anchor="middle">§2.3.3</text><rect class="box-1" x="460.0" y="156.0" width="90.0" height="34.0" rx="6"/><text class="t-note" x="505.0" y="177.5" text-anchor="middle">SFT only</text><line class="edge-on" x1="420.0" y1="173.0" x2="456.0" y2="173.0" marker-end="url(#ah-on)"/><text class="t-muted" x="505.0" y="210.0" text-anchor="middle">no RL stage</text><text class="t-muted" x="505.0" y="226.0" text-anchor="middle">(§2.4)</text><g class="mark"><title>Qwen2.5-Math-1.5B fine-tuned on the 800k samples gives R1-Distill-Qwen-1.5B</title><rect class="box" x="575.0" y="52.0" width="130.0" height="22.0" rx="6"/><text class="t-tick" x="640.0" y="67.5" text-anchor="middle">Qwen2.5-Math-1.5B</text><text class="t-tick" x="640.0" y="88.0" text-anchor="middle">R1-Distill-Qwen-1.5B</text></g><line class="edge" x1="550.0" y1="173.0" x2="573.0" y2="63.0" marker-end="url(#ah)"/><g class="mark"><title>Qwen2.5-Math-7B fine-tuned on the 800k samples gives R1-Distill-Qwen-7B</title><rect class="box" x="575.0" y="104.0" width="130.0" height="22.0" rx="6"/><text class="t-tick" x="640.0" y="119.5" text-anchor="middle">Qwen2.5-Math-7B</text><text class="t-tick" x="640.0" y="140.0" text-anchor="middle">R1-Distill-Qwen-7B</text></g><line class="edge" x1="550.0" y1="173.0" x2="573.0" y2="115.0" marker-end="url(#ah)"/><g class="mark"><title>Qwen2.5-14B fine-tuned on the 800k samples gives R1-Distill-Qwen-14B</title><rect class="box" x="575.0" y="156.0" width="130.0" height="22.0" rx="6"/><text class="t-tick" x="640.0" y="171.5" text-anchor="middle">Qwen2.5-14B</text><text class="t-tick" x="640.0" y="192.0" text-anchor="middle">R1-Distill-Qwen-14B</text></g><line class="edge" x1="550.0" y1="173.0" x2="573.0" y2="167.0" marker-end="url(#ah)"/><g class="mark"><title>Qwen2.5-32B fine-tuned on the 800k samples gives R1-Distill-Qwen-32B</title><rect class="box" x="575.0" y="208.0" width="130.0" height="22.0" rx="6"/><text class="t-tick" x="640.0" y="223.5" text-anchor="middle">Qwen2.5-32B</text><text class="t-tick" x="640.0" y="244.0" text-anchor="middle">R1-Distill-Qwen-32B</text></g><line class="edge" x1="550.0" y1="173.0" x2="573.0" y2="219.0" marker-end="url(#ah)"/><g class="mark"><title>Llama-3.1-8B fine-tuned on the 800k samples gives R1-Distill-Llama-8B</title><rect class="box" x="575.0" y="260.0" width="130.0" height="22.0" rx="6"/><text class="t-tick" x="640.0" y="275.5" text-anchor="middle">Llama-3.1-8B</text><text class="t-tick" x="640.0" y="296.0" text-anchor="middle">R1-Distill-Llama-8B</text></g><line class="edge" x1="550.0" y1="173.0" x2="573.0" y2="271.0" marker-end="url(#ah)"/><g class="mark"><title>Llama-3.3-70B-Instruct fine-tuned on the 800k samples gives R1-Distill-Llama-70B</title><rect class="box" x="575.0" y="312.0" width="130.0" height="22.0" rx="6"/><text class="t-tick" x="640.0" y="327.5" text-anchor="middle">Llama-3.3-70B-Inst.</text><text class="t-tick" x="640.0" y="348.0" text-anchor="middle">R1-Distill-Llama-70B</text></g><line class="edge" x1="550.0" y1="173.0" x2="573.0" y2="323.0" marker-end="url(#ah)"/><text class="t-muted" x="575.0" y="382.0" text-anchor="start">student base model, and the</text><text class="t-muted" x="575.0" y="398.0" text-anchor="start">released model it becomes</text></svg><figcaption>You are here. Part 5 turns DeepSeek-V3-Base into DeepSeek-R1, and its stage 3 builds 800k samples. Section 2.4 fine-tunes six open models on those samples, with no RL stage.</figcaption></figure>

> [!PAPER] · §2.4 · page 11
> [![The first lines of Section 2.4, with directly fine-tuned open-source models like Qwen, the 800k samples curated with DeepSeek-R1 and this straightforward distillation method significantly enhances the reasoning abilities of smaller highlighted](/img/papers/deepseek-r1/p6-s24-a.png)](/img/papers/deepseek-r1/p6-s24-a.png)
>
> **What it says:** the authors "directly fine-tuned open-source models like Qwen and Llama using the 800k samples curated with DeepSeek-R1". They find that "this straightforward distillation method significantly enhances the reasoning abilities of smaller models".
>
> **What it means:**
>
> "Directly" is the important word. There is no cold start, no reward and no RL here. The recipe is the most ordinary one in machine learning: take a pretrained model and fine-tune it on examples of the behaviour you want. The only unusual part is where the examples came from.
>
> Each example is a prompt and an answer. For a maths problem, the answer is R1's full reasoning between the thinking tags, then a short summary with the final result. The student learns to write that whole text, token by token.

> [!PAPER] · §2.4 · page 11
> [![The list of base models, with Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B and Llama-3.3-70B-Instruct highlighted, and the reason for Llama-3.3](/img/papers/deepseek-r1/p6-s24-b.png)](/img/papers/deepseek-r1/p6-s24-b.png)
>
> **What it says:** the students are "Qwen2.5-Math-1.5B, Qwen2.5-Math-7B, Qwen2.5-14B, Qwen2.5-32B, Llama-3.1-8B, and Llama-3.3-70B-Instruct". Llama-3.3 was chosen because "its reasoning capability is slightly better than that of Llama-3.1".
>
> **What it means:**
>
> Look at the list closely, because it is not uniform. The two smallest Qwen students are maths specialists: Qwen2.5-Math models were trained on extra maths text before any of this. The 14B and 32B Qwen students are general base models. The 70B Llama student is an instruct model, already fine-tuned to chat, while the 8B Llama is a base model.
>
> So the students start from different places. Keep this in mind when you read Table 5. The 1.5B student had a head start in maths. The 70B student had a head start in following instructions.

> [!PAPER] · §2.4 · page 11
> [![The second paragraph of Section 2.4, with we apply only SFT and do not include an RL stage, could substantially boost model performance and demonstrate the effectiveness of the distillation technique highlighted](/img/papers/deepseek-r1/p6-s24-c.png)](/img/papers/deepseek-r1/p6-s24-c.png)
>
> **What it says:** "we apply only SFT and do not include an RL stage", even though RL "could substantially boost model performance". The goal is to "demonstrate the effectiveness of the distillation technique".
>
> **What it means:**
>
> This is a choice made for science, not for the best scores. If the students also got RL, a reader could not tell which gains came from R1's text and which from the RL. With SFT only, every point in Table 5 comes from the copied answers.
>
> It also makes the recipe cheap to repeat. Anyone with the data and a few GPUs can run SFT. RL needs a reward, many samples per prompt and a careful setup, as Parts 2 to 4 showed.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-R1, later long version · Appendix B.4.3 · page 35
> [![Appendix B.4.3 of the later version, with 2 to 3 epochs, cosine decay learning rate scheduler, one-tenth of its initial value, batch size is 64, Qwen2.5-Math-1.5B and 1 times 10 to the minus 4 highlighted](/img/papers/deepseek-r1/p6-hparams.png)](/img/papers/deepseek-r1/p6-hparams.png)
>
> **What it says:** each base model is fine-tuned "for 2–3 epochs" on the 800k data, with a "cosine decay learning rate scheduler" that ends at "one-tenth of its initial value". The maximum context is 32,768 tokens and the "batch size is 64". The 1.5B student starts at a learning rate of $$1 \times 10^{-4}$$; the 70B student starts at $$2 \times 10^{-5}$$.
>
> **What it means:**
>
> Version 1 of the paper gives none of these numbers; the later long version adds them. They are ordinary fine-tuning settings. The smaller the model, the larger its learning rate, which is the usual rule.
>
> A quick count shows the size of the job. With 800k samples and a batch of 64, one epoch is 800,000 / 64 = 12,500 steps. Two to three epochs make 25,000 to 37,500 steps. Each step is one forward and one backward pass, with no sampling.

### The loss of distillation, written out

Every step of this SFT uses one loss. The intuition is a teacher who reads a student's essay aloud and asks, at each word, "how surprised were you by this word?". The student is graded only on the answer part, never on the question, because the question is given.

$$
\mathcal{L}_{\text{SFT}} = - \frac{1}{N} \sum_{t \in \text{response}} \log p_\theta\left(y_t \mid y_{<t}\right)
$$

Read it from left to right. $$N$$ is the number of response tokens. The sum runs only over positions in the response, the teacher's answer. $$y_t$$ is the token that the teacher wrote at position $$t$$, and $$y_{<t}$$ is everything before it, the prompt included. $$p_\theta$$ is the student's probability, with its weights $$\theta$$. The minus sign turns "high probability" into "low loss".

This is exactly the sequence-level loss of the last section, averaged over a whole answer. A tiny example: a response of three tokens that the student gives probabilities 0.5, 0.25 and 0.8. The loss is $$-(\ln 0.5 + \ln 0.25 + \ln 0.8)/3 = (0.693 + 1.386 + 0.223)/3 = 0.767$$. The prompt tokens add nothing, however unlikely they are.

> [!DEFINITION] Perplexity
> $$e^{\mathcal{L}}$$, the exponential of the mean loss per token. It reads as "how many tokens the model is choosing between, on average". A perplexity of 2 means the model was as unsure as a fair coin at each token. Our three-token example has a perplexity of $$e^{0.767} = 2.15$$.

How does a program know which tokens are the response? Libraries mark every prompt token with the label $$-100$$, a number that no real token has, and skip those positions in the sum. The student must also predict token $$t+1$$ from position $$t$$, so the labels are shifted by one. These two details are where SFT code usually goes wrong, so we write them ourselves.

#### Method 1: the loss from scratch

We use a real R1-style example. The distilled 1.5B model solved the first problem of the GSM8K test set, about a farmer's ducks and eggs:

```text
question: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?
gold answer: 18
generated tokens: 872, finished with end token: True
trace starts: "Okay, so I have this problem about Janet and her ducks. Let me try to break it down step by step. I need to figure out how much money Janet makes every day at the farmers' market. Hmm, let's see.\n\nFirst, the problem says"
trace ends:   "ells for $2. Therefore, the total money she makes per day is:\n\n\\[\n9 \\times 2 = 18\n\\]\n\nThus, Janet makes \\(\\boxed{18}\\) dollars every day at the farmers' market."
```

Then we tokenise it the way SFT does. The prompt comes from the chat template and ends with the thinking tag; the response is the teacher's text plus the end token.

```python
def build_example(tok, question, response):
    """Tokenise one distillation example the way SFT sees it.
    prompt: the chat template up to where the assistant starts (it ends with '<think>\\n' for the R1 template).
    response: the teacher's text plus the end-of-sequence token. Only the response is trained on."""
    msgs = [{'role': 'user', 'content': question + SUFFIX}]
    prompt_ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)   # list of P ids
    if not isinstance(prompt_ids, list):
        prompt_ids = prompt_ids['input_ids']
    response_ids = tok(response + tok.eos_token, add_special_tokens=False)['input_ids']  # list of N ids
    input_ids = prompt_ids + response_ids                                                 # P + N ids
    labels = [-100] * len(prompt_ids) + response_ids                                      # -100 = "no loss here"
    return input_ids, labels, len(prompt_ids)
```

<figure class="fig"><svg viewBox="0 0 720 170" role="img" aria-label="The tokens around the boundary between prompt and response in our real example, with their labels. Prompt tokens get the label minus 100 and add nothing to the loss; response tokens keep their own id."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One SFT example: the prompt is masked, the response is trained on</text><rect class="box" x="20.0" y="50.0" width="46.0" height="30.0" rx="6"/><text class="t-mono" x="43.0" y="69.5" text-anchor="middle">{}.</text><text class="t-tick" x="43.0" y="100.0" text-anchor="middle">-100</text><rect class="box" x="72.0" y="50.0" width="122.6" height="30.0" rx="6"/><text class="t-mono" x="133.3" y="69.5" text-anchor="middle">&lt;｜Assistant｜&gt;</text><text class="t-tick" x="133.3" y="100.0" text-anchor="middle">-100</text><rect class="box" x="200.6" y="50.0" width="73.4" height="30.0" rx="6"/><text class="t-mono" x="237.3" y="69.5" text-anchor="middle">&lt;think&gt;</text><text class="t-tick" x="237.3" y="100.0" text-anchor="middle">-100</text><rect class="box" x="280.0" y="50.0" width="46.0" height="30.0" rx="6"/><text class="t-mono" x="303.0" y="69.5" text-anchor="middle">\n</text><text class="t-tick" x="303.0" y="100.0" text-anchor="middle">-100</text><rect class="box-on" x="332.0" y="50.0" width="48.8" height="30.0" rx="6"/><text class="t-mono" x="356.4" y="69.5" text-anchor="middle">Okay</text><text class="t-tick" x="356.4" y="100.0" text-anchor="middle">32313</text><rect class="box-on" x="386.8" y="50.0" width="46.0" height="30.0" rx="6"/><text class="t-mono" x="409.8" y="69.5" text-anchor="middle">,</text><text class="t-tick" x="409.8" y="100.0" text-anchor="middle">11</text><rect class="box-on" x="438.8" y="50.0" width="46.0" height="30.0" rx="6"/><text class="t-mono" x="461.8" y="69.5" text-anchor="middle"> so</text><text class="t-tick" x="461.8" y="100.0" text-anchor="middle">773</text><rect class="box-on" x="490.8" y="50.0" width="46.0" height="30.0" rx="6"/><text class="t-mono" x="513.8" y="69.5" text-anchor="middle"> I</text><text class="t-tick" x="513.8" y="100.0" text-anchor="middle">358</text><rect class="box-on" x="542.8" y="50.0" width="57.0" height="30.0" rx="6"/><text class="t-mono" x="571.3" y="69.5" text-anchor="middle"> have</text><text class="t-tick" x="571.3" y="100.0" text-anchor="middle">614</text><text class="t-muted" x="20.0" y="122.0" text-anchor="start">label</text><text class="t-muted" x="20.0" y="148.0" text-anchor="start">Plain boxes: prompt tokens, label -100, no loss.  Outlined boxes: response tokens, label = the token id, trained on.</text></svg><figcaption>The tokens of our real example around the boundary between prompt and response. Prompt tokens carry the label minus 100; response tokens carry their own token id.</figcaption></figure>

```text
tokens: 957 in total, 85 prompt tokens (label -100), 872 response tokens
  position  82  token '<｜Assistant｜>'        label -100
  position  83  token '<think>'              label -100
  position  84  token '\n'                   label -100
  position  85  token 'Okay'                 label 32313
  position  86  token ','                    label 11
  position  87  token ' so'                  label 773
```

Now the loss itself, with the log-softmax written by hand. A model returns one row of scores (logits) for each position. The softmax turns a row into probabilities, and we need their log.

```python
def sft_loss_from_scratch(logits, labels):
    """Method 1. logits: (T, V) floats, one row per position. labels: (T,) ids, -100 where there is no loss.
    The model at position t predicts the token at t + 1, so we shift by one."""
    logits = logits[:-1]                                      # (T-1, V): predictions made at positions 0 .. T-2
    target = labels[1:]                                       # (T-1,):   the tokens at positions 1 .. T-1
    keep = target != -100                                     # (T-1,):   True only on response tokens
    m = logits.max(dim=-1, keepdim=True).values               # (T-1, 1): row maximum, for a stable log-sum-exp
    logsumexp = m + torch.log(torch.exp(logits - m).sum(dim=-1, keepdim=True))   # (T-1, 1)
    logprobs = logits - logsumexp                             # (T-1, V): log-softmax by hand
    picked = logprobs[keep].gather(1, target[keep].unsqueeze(1)).squeeze(1)      # (N,): log p(correct token)
    nll = -picked                                             # (N,): loss of each response token
    return nll.mean(), nll                                    # mean over response tokens only
```

#### Method 2: the same loss from the libraries

The student here is Qwen2.5-Math-1.5B, the base model that became R1-Distill-Qwen-1.5B. We give it the same example three more ways. transformers computes the loss itself when it gets `labels`. PyTorch's `cross_entropy` skips the $$-100$$ labels with `ignore_index`. And TRL's `SFTTrainer`, the trainer most people use for this job, tokenises the prompt and completion, builds its own labels and computes its own loss.

```python
loss1, nll1 = sft_loss_from_scratch(logits, lab[0])                  # Method 1, our function

# Method 2a: transformers computes the loss itself when we pass labels
loss2 = bmodel(ids, labels=lab).loss.float()
# Method 2b: PyTorch's cross_entropy with ignore_index=-100
loss2b = F.cross_entropy(logits[:-1], lab[0, 1:], ignore_index=-100)
# Method 2c: TRL's SFTTrainer builds the example (tokenise, mask the prompt) and computes the loss
ds = Dataset.from_list([{'prompt': prompt_text, 'completion': trace + dtok.eos_token}])
cfg = SFTConfig(output_dir=os.path.join(RESULTS, '.part6_trl_tmp'), report_to=[], completion_only_loss=True,
                max_length=None, per_device_eval_batch_size=1)
trainer = SFTTrainer(model=bmodel, args=cfg, train_dataset=ds, processing_class=dtok)
ex = trainer.train_dataset[0]
same_ids = list(ex['input_ids']) == input_ids
same_labels = list(ex['labels']) == labels
batch = trainer.data_collator([ex])
loss3 = trainer.compute_loss(bmodel, {k: v.to(DEV) for k, v in batch.items()}).float()
```

```text
logits shape: (957, 151936)
Method 1 (from scratch)        loss = 0.927018  over 872 tokens
Method 2a (model(labels=...))  loss = 0.927019
Method 2b (F.cross_entropy)    loss = 0.927019
Method 2c (TRL SFTTrainer.compute_loss) loss = 0.927019
TRL tokenised example: same input ids as ours: True, same labels (prompt masked): True
largest difference between methods: 1.25e-06  (agree to 1e-5: True)
first 12 response tokens and their loss under the student (nats):
  'Okay':12.07  ',':0.17  ' so':3.67  ' I':4.01  ' have':2.57  ' this':3.09  ' problem':0.67  ' about':3.20  ' Janet':0.93  ' and':1.18  ' her':0.08  ' ducks':0.14
```

All four numbers are 0.927 nats per token. Our function and the libraries differ by $$1.25 \times 10^{-6}$$, which is rounding in 32-bit floats, well inside $$10^{-5}$$. TRL also built exactly our token ids and exactly our labels, with the 85 prompt tokens masked. So the whole of SFT distillation fits in the two short functions above: build the labels, then average the log-probabilities of the response.

The per-token losses are worth a look. The base model gives the first word, "Okay", a loss of 12.07 nats: it almost never starts an answer like that. A few tokens later, " her" and " ducks" cost under 0.15 nats, because the question already said them. Distillation must teach the student the first kind of token, the voice and habits of R1. The second kind it already knows.

[![Terminal output of r1_part6.py: the trace, the mask, the four losses and the base against distilled comparison](/img/papers/deepseek-r1/part6_loss-run.png)](/img/papers/deepseek-r1/part6_loss-run.png)

## How the distilled models are tested {§3}

> [!PAPER] · §3 · page 12
> [![The sentence on distilled models in the Benchmarks paragraph, with report representative results on AIME 2024, MATH-500, GPQA Diamond, Codeforces, and LiveCodeBench highlighted](/img/papers/deepseek-r1/p6-bench.png)](/img/papers/deepseek-r1/p6-bench.png)
>
> **What it says:** "For distilled models, we report representative results on AIME 2024, MATH-500, GPQA Diamond, Codeforces, and LiveCodeBench."
>
> **What it means:**
>
> Five tests, in three groups. AIME 2024 and MATH-500 are maths: 30 hard competition problems and 500 school and competition problems. GPQA Diamond is science: graduate-level multiple-choice questions in biology, physics and chemistry. LiveCodeBench and Codeforces are code: write a program that passes hidden tests.
>
> Remember these groups. The headline about the smallest model is true for one group and false for another.

> [!PAPER] · §3 · page 12
> [![The Baselines paragraph, with GPT-4o-0513, OpenAI-o1-mini and For distilled models, we also compare the open-source model QwQ-32B-Preview highlighted](/img/papers/deepseek-r1/p6-baselines.png)](/img/papers/deepseek-r1/p6-baselines.png)
>
> **What it says:** the baselines include GPT-4o-0513, Anthropic's Sonnet 3.5 and OpenAI-o1-mini. "For distilled models, we also compare the open-source model QwQ-32B-Preview."
>
> **What it means:**
>
> The four comparison models have two kinds. GPT-4o and Sonnet 3.5 are strong general chat models. They answer without a long phase of reasoning first. o1-mini and QwQ-32B-Preview are reasoning models: they think at length before they answer. QwQ-32B-Preview was the best open reasoning model of late 2024, from the same team that makes Qwen.
>
> So Table 5 asks two questions. Can a small distilled model beat big models that do not reason? And can it match models that were built to reason?

> [!PAPER] · §3 · page 12
> [![The Evaluation Setup paragraph, with maximum generation length to 32,768 tokens, sampling temperature of 0.6, top-p value of 0.95, consensus (majority vote) results and cons@64 highlighted](/img/papers/deepseek-r1/p6-setup.png)](/img/papers/deepseek-r1/p6-setup.png)
>
> **What it says:** answers may be up to "32,768 tokens" long. Each question gets $$k$$ samples with a "sampling temperature of 0.6" and top-p 0.95, and pass@1 is the mean correctness of those samples. For AIME there is also a majority vote over 64 samples, "cons@64".
>
> **What it means:**
>
> [Part 4](part-4-r1-zero-and-the-aha-moment.md) met these two scores in Table 2, and [Part 7](part-7-results-failures-and-limits.md) reads the whole setup. In one line: pass@1 is "how often is one answer right", and cons@64 is "is the most common answer of 64 right".
>
> One number matters for our own run below: 32,768 tokens. A distilled model often thinks for thousands of tokens. On a laptop we must stop much earlier, and that costs accuracy.

## Table 5, row by row {§3.2}

> [!PAPER] · Table 5 · page 14
> [![Table 5 of the paper: GPT-4o-0513, Anthropic's Sonnet 3.5, OpenAI-o1-mini and QwQ-32B-Preview above a line, then the six DeepSeek-R1-Distill models, with AIME 2024 pass@1 and cons@64, MATH-500, GPQA Diamond, LiveCodeBench and CodeForces rating](/img/papers/deepseek-r1/p6-table5.png)](/img/papers/deepseek-r1/p6-table5.png)
>
> **What it says:** ten models, five benchmarks. Above the line are four models that did not learn from R1. Below the line are the six distilled students. The bold numbers are the best of each column.
>
> **What it means:**
>
> The table is dense, so the figure below draws its two maths columns. The sections after it read the table row by row.

<figure class="fig"><svg viewBox="0 0 720 524" role="img" aria-label="Grouped bar chart of Table 5. For each of ten models, a bar for AIME 2024 pass@1 and a bar for MATH-500 pass@1. Every distilled model except the 1.5B beats o1-mini on MATH-500; the 1.5B still beats GPT-4o and Sonnet 3.5 on both maths tests."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Table 5: the six distilled models against four other models (pass@1, %)</text><rect class="s1" x="190" y="38" width="12" height="12" rx="2"/><text class="t-note" x="208.0" y="48.0" text-anchor="start">AIME 2024</text><rect class="s2" x="284.8" y="38" width="12" height="12" rx="2"/><text class="t-note" x="302.8" y="48.0" text-anchor="start">MATH-500</text><line class="grid" x1="190.0" y1="60" x2="190.0" y2="464"/><text class="t-tick" x="190.0" y="480.0" text-anchor="middle">0</text><line class="grid" x1="305.0" y1="60" x2="305.0" y2="464"/><text class="t-tick" x="305.0" y="480.0" text-anchor="middle">25</text><line class="grid" x1="420.0" y1="60" x2="420.0" y2="464"/><text class="t-tick" x="420.0" y="480.0" text-anchor="middle">50</text><line class="grid" x1="535.0" y1="60" x2="535.0" y2="464"/><text class="t-tick" x="535.0" y="480.0" text-anchor="middle">75</text><line class="grid" x1="650.0" y1="60" x2="650.0" y2="464"/><text class="t-tick" x="650.0" y="480.0" text-anchor="middle">100</text><text class="t-tick" x="180.0" y="84.0" text-anchor="end">GPT-4o-0513</text><g class="mark"><title>GPT-4o-0513, AIME 2024: 9.3</title><path class="s1" d="M190,68H228.78Q232.78,68 232.78,72V77Q232.78,81 228.78,81H190Z"/></g><text class="t-val" x="237.8" y="79.0" text-anchor="start">9.3</text><g class="mark"><title>GPT-4o-0513, MATH-500: 74.6</title><path class="s2" d="M190,84H529.1600000000001Q533.1600000000001,84 533.1600000000001,88V93Q533.1600000000001,97 529.1600000000001,97H190Z"/></g><text class="t-val" x="538.2" y="95.0" text-anchor="start">74.6</text><text class="t-tick" x="180.0" y="124.0" text-anchor="end">Sonnet 3.5 (Anthropic)</text><g class="mark"><title>Sonnet 3.5 (Anthropic), AIME 2024: 16.0</title><path class="s1" d="M190,108H259.6Q263.6,108 263.6,112V117Q263.6,121 259.6,121H190Z"/></g><text class="t-val" x="268.6" y="119.0" text-anchor="start">16.0</text><g class="mark"><title>Sonnet 3.5 (Anthropic), MATH-500: 78.3</title><path class="s2" d="M190,124H546.1800000000001Q550.1800000000001,124 550.1800000000001,128V133Q550.1800000000001,137 546.1800000000001,137H190Z"/></g><text class="t-val" x="555.2" y="135.0" text-anchor="start">78.3</text><text class="t-tick" x="180.0" y="164.0" text-anchor="end">OpenAI-o1-mini</text><g class="mark"><title>OpenAI-o1-mini, AIME 2024: 63.6</title><path class="s1" d="M190,148H478.56Q482.56,148 482.56,152V157Q482.56,161 478.56,161H190Z"/></g><text class="t-val" x="487.6" y="159.0" text-anchor="start">63.6</text><g class="mark"><title>OpenAI-o1-mini, MATH-500: 90.0</title><path class="s2" d="M190,164H600.0Q604.0,164 604.0,168V173Q604.0,177 600.0,177H190Z"/></g><text class="t-val" x="609.0" y="175.0" text-anchor="start">90.0</text><text class="t-tick" x="180.0" y="204.0" text-anchor="end">QwQ-32B-Preview</text><g class="mark"><title>QwQ-32B-Preview, AIME 2024: 50.0</title><path class="s1" d="M190,188H416.0Q420.0,188 420.0,192V197Q420.0,201 416.0,201H190Z"/></g><text class="t-val" x="425.0" y="199.0" text-anchor="start">50.0</text><g class="mark"><title>QwQ-32B-Preview, MATH-500: 90.6</title><path class="s2" d="M190,204H602.76Q606.76,204 606.76,208V213Q606.76,217 602.76,217H190Z"/></g><text class="t-val" x="611.8" y="215.0" text-anchor="start">90.6</text><line class="axis" x1="20" y1="220" x2="700" y2="220"/><text class="t-tick t-strong" x="180.0" y="244.0" text-anchor="end">R1-Distill-Qwen-1.5B</text><g class="mark"><title>R1-Distill-Qwen-1.5B, AIME 2024: 28.9</title><path class="s1" d="M190,228H318.94Q322.94,228 322.94,232V237Q322.94,241 318.94,241H190Z"/></g><text class="t-val" x="327.9" y="239.0" text-anchor="start">28.9</text><g class="mark"><title>R1-Distill-Qwen-1.5B, MATH-500: 83.9</title><path class="s2" d="M190,244H571.94Q575.94,244 575.94,248V253Q575.94,257 571.94,257H190Z"/></g><text class="t-val" x="580.9" y="255.0" text-anchor="start">83.9</text><text class="t-tick t-strong" x="180.0" y="284.0" text-anchor="end">R1-Distill-Qwen-7B</text><g class="mark"><title>R1-Distill-Qwen-7B, AIME 2024: 55.5</title><path class="s1" d="M190,268H441.3Q445.3,268 445.3,272V277Q445.3,281 441.3,281H190Z"/></g><text class="t-val" x="450.3" y="279.0" text-anchor="start">55.5</text><g class="mark"><title>R1-Distill-Qwen-7B, MATH-500: 92.8</title><path class="s2" d="M190,284H612.88Q616.88,284 616.88,288V293Q616.88,297 612.88,297H190Z"/></g><text class="t-val" x="621.9" y="295.0" text-anchor="start">92.8</text><text class="t-tick t-strong" x="180.0" y="324.0" text-anchor="end">R1-Distill-Qwen-14B</text><g class="mark"><title>R1-Distill-Qwen-14B, AIME 2024: 69.7</title><path class="s1" d="M190,308H506.62Q510.62,308 510.62,312V317Q510.62,321 506.62,321H190Z"/></g><text class="t-val" x="515.6" y="319.0" text-anchor="start">69.7</text><g class="mark"><title>R1-Distill-Qwen-14B, MATH-500: 93.9</title><path class="s2" d="M190,324H617.94Q621.94,324 621.94,328V333Q621.94,337 617.94,337H190Z"/></g><text class="t-val" x="626.9" y="335.0" text-anchor="start">93.9</text><text class="t-tick t-strong" x="180.0" y="364.0" text-anchor="end">R1-Distill-Qwen-32B</text><g class="mark"><title>R1-Distill-Qwen-32B, AIME 2024: 72.6</title><path class="s1" d="M190,348H519.96Q523.96,348 523.96,352V357Q523.96,361 519.96,361H190Z"/></g><text class="t-val" x="529.0" y="359.0" text-anchor="start">72.6</text><g class="mark"><title>R1-Distill-Qwen-32B, MATH-500: 94.3</title><path class="s2" d="M190,364H619.78Q623.78,364 623.78,368V373Q623.78,377 619.78,377H190Z"/></g><text class="t-val" x="628.8" y="375.0" text-anchor="start">94.3</text><text class="t-tick t-strong" x="180.0" y="404.0" text-anchor="end">R1-Distill-Llama-8B</text><g class="mark"><title>R1-Distill-Llama-8B, AIME 2024: 50.4</title><path class="s1" d="M190,388H417.84000000000003Q421.84000000000003,388 421.84000000000003,392V397Q421.84000000000003,401 417.84000000000003,401H190Z"/></g><text class="t-val" x="426.8" y="399.0" text-anchor="start">50.4</text><g class="mark"><title>R1-Distill-Llama-8B, MATH-500: 89.1</title><path class="s2" d="M190,404H595.86Q599.86,404 599.86,408V413Q599.86,417 595.86,417H190Z"/></g><text class="t-val" x="604.9" y="415.0" text-anchor="start">89.1</text><text class="t-tick t-strong" x="180.0" y="444.0" text-anchor="end">R1-Distill-Llama-70B</text><g class="mark"><title>R1-Distill-Llama-70B, AIME 2024: 70.0</title><path class="s1" d="M190,428H508.0Q512.0,428 512.0,432V437Q512.0,441 508.0,441H190Z"/></g><text class="t-val" x="517.0" y="439.0" text-anchor="start">70.0</text><g class="mark"><title>R1-Distill-Llama-70B, MATH-500: 94.5</title><path class="s2" d="M190,444H620.7Q624.7,444 624.7,448V453Q624.7,457 620.7,457H190Z"/></g><text class="t-val" x="629.7" y="455.0" text-anchor="start">94.5</text><text class="t-muted" x="20.0" y="506.0" text-anchor="start">Above the line: models that were not distilled from R1. Below: the six students. Source: arXiv v1, Table 5.</text></svg><figcaption>Table 5 drawn as bars: AIME 2024 pass@1 and MATH-500 pass@1 for the four comparison models (top) and the six distilled students (bottom).</figcaption></figure>

**R1-Distill-Qwen-1.5B.** AIME 28.9, MATH-500 83.9, GPQA 33.8, LiveCodeBench 16.9, Codeforces 954. On the two maths tests it beats GPT-4o (9.3 and 74.6) and Sonnet 3.5 (16.0 and 78.3). On GPQA and LiveCodeBench it is far below both. GPQA Diamond has four choices per question, so 33.8% is not far above a guess.

On the Codeforces rating, the 1.5B student is above both chat models: 954 against 759 and 717. But that column measures competition problems, which are close to maths puzzles.

**R1-Distill-Qwen-7B.** AIME 55.5, MATH-500 92.8. On maths it is already near o1-mini (63.6 and 90.0) and above QwQ-32B-Preview on AIME (50.0). It beats GPT-4o on every column but one: GPQA, 49.1 against 49.9. The paper's words "across the board" round that away.

**R1-Distill-Qwen-14B.** AIME 69.7, cons@64 80.0, MATH-500 93.9, GPQA 59.1, LiveCodeBench 53.1, Codeforces 1481. Every one of these numbers is above QwQ-32B-Preview's, a model more than twice its size that was built to reason.

**R1-Distill-Qwen-32B.** AIME 72.6, MATH-500 94.3, GPQA 62.1, LiveCodeBench 57.2. It beats o1-mini on five of six columns. The exception is the Codeforces rating, 1691 against 1820.

**R1-Distill-Llama-8B.** AIME 50.4, MATH-500 89.1. It is a little weaker on maths than the Qwen 7B student, which started from a maths model. On code it is a little stronger: LiveCodeBench 39.6 against 37.6.

**R1-Distill-Llama-70B.** AIME 70.0, cons@64 86.7, MATH-500 94.5, GPQA 65.2, LiveCodeBench 57.5. It has the best value in four columns. Yet it is more than twice the size of the Qwen 32B student, and only level with it. AIME is 70.0 against 72.6, and MATH-500 is 94.5 against 94.3.

Two patterns run through the rows. First, size helps, but less and less: AIME goes 28.9, 55.5, 69.7, 72.6 from 1.5B to 32B. Second, the start matters. The maths models are strong on maths for their size. The Llama students are not better per parameter than the Qwen students.

> [!PAPER] · §5 · page 16
> [![The distillation paragraph of the conclusion, with 800K training samples and on math benchmarks with 28.9% on AIME and 83.9% on MATH highlighted](/img/papers/deepseek-r1/p6-conclusion.png)](/img/papers/deepseek-r1/p6-conclusion.png)
>
> **What it says:** R1 is the teacher for "800K training samples". The 1.5B student outperforms GPT-4o and Anthropic's Sonnet 3.5 "on math benchmarks with 28.9% on AIME and 83.9% on MATH".
>
> **What it means:**
>
> Read the words "on math benchmarks" with care. This is the sentence most often repeated about the paper, usually without them. A 1.5B model is not better than GPT-4o in general. It is better at two maths tests, after it learned from 800k answers that are mostly maths and code reasoning.
>
> The next figure shows the same three models on the other two tests. The order turns upside down.

<figure class="fig"><svg viewBox="0 0 720 320" role="img" aria-label="Bars for GPT-4o, Sonnet 3.5 and R1-Distill-Qwen-1.5B on four benchmarks. The 1.5B model is best on AIME 2024 and MATH-500, and worst on GPQA Diamond and LiveCodeBench."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">The 1.5B student wins on maths and loses on science and code</text><rect class="s3" x="20" y="40" width="12" height="12" rx="2"/><text class="t-note" x="38.0" y="50.0" text-anchor="start">GPT-4o-0513</text><rect class="s4" x="129.2" y="40" width="12" height="12" rx="2"/><text class="t-note" x="147.2" y="50.0" text-anchor="start">Sonnet 3.5 (Anthropic)</text><rect class="s1" x="317.6" y="40" width="12" height="12" rx="2"/><text class="t-note" x="335.6" y="50.0" text-anchor="start">R1-Distill-Qwen-1.5B</text><text class="t-tick" x="100.0" y="272.0" text-anchor="middle">AIME 2024</text><line class="axis" x1="30" y1="250" x2="170" y2="250"/><g class="mark"><title>GPT-4o-0513, AIME 2024: 9.3</title><path class="s3" d="M40,250.0V238.19Q40,234.19 44,234.19H72Q76,234.19 76,238.19V250.0Z"/></g><text class="t-val" x="58.0" y="228.2" text-anchor="middle">9.3</text><g class="mark"><title>Sonnet 3.5 (Anthropic), AIME 2024: 16.0</title><path class="s4" d="M82,250.0V226.8Q82,222.8 86,222.8H114Q118,222.8 118,226.8V250.0Z"/></g><text class="t-val" x="100.0" y="216.8" text-anchor="middle">16.0</text><g class="mark"><title>R1-Distill-Qwen-1.5B, AIME 2024: 28.9</title><path class="s1" d="M124,250.0V204.87Q124,200.87 128,200.87H156Q160,200.87 160,204.87V250.0Z"/></g><text class="t-val" x="142.0" y="194.9" text-anchor="middle">28.9</text><text class="t-tick" x="272.0" y="272.0" text-anchor="middle">MATH-500</text><line class="axis" x1="202" y1="250" x2="342" y2="250"/><g class="mark"><title>GPT-4o-0513, MATH-500: 74.6</title><path class="s3" d="M212,250.0V127.18000000000002Q212,123.18000000000002 216,123.18000000000002H244Q248,123.18000000000002 248,127.18000000000002V250.0Z"/></g><text class="t-val" x="230.0" y="117.2" text-anchor="middle">74.6</text><g class="mark"><title>Sonnet 3.5 (Anthropic), MATH-500: 78.3</title><path class="s4" d="M254,250.0V120.88999999999999Q254,116.88999999999999 258,116.88999999999999H286Q290,116.88999999999999 290,120.88999999999999V250.0Z"/></g><text class="t-val" x="272.0" y="110.9" text-anchor="middle">78.3</text><g class="mark"><title>R1-Distill-Qwen-1.5B, MATH-500: 83.9</title><path class="s1" d="M296,250.0V111.36999999999998Q296,107.36999999999998 300,107.36999999999998H328Q332,107.36999999999998 332,111.36999999999998V250.0Z"/></g><text class="t-val" x="314.0" y="101.4" text-anchor="middle">83.9</text><line class="grid" x1="358" y1="70" x2="358" y2="280"/><text class="t-tick" x="444.0" y="272.0" text-anchor="middle">GPQA Diamond</text><line class="axis" x1="374" y1="250" x2="514" y2="250"/><g class="mark"><title>GPT-4o-0513, GPQA Diamond: 49.9</title><path class="s3" d="M384,250.0V169.17000000000002Q384,165.17000000000002 388,165.17000000000002H416Q420,165.17000000000002 420,169.17000000000002V250.0Z"/></g><text class="t-val" x="402.0" y="159.2" text-anchor="middle">49.9</text><g class="mark"><title>Sonnet 3.5 (Anthropic), GPQA Diamond: 65.0</title><path class="s4" d="M426,250.0V143.5Q426,139.5 430,139.5H458Q462,139.5 462,143.5V250.0Z"/></g><text class="t-val" x="444.0" y="133.5" text-anchor="middle">65.0</text><g class="mark"><title>R1-Distill-Qwen-1.5B, GPQA Diamond: 33.8</title><path class="s1" d="M468,250.0V196.54000000000002Q468,192.54000000000002 472,192.54000000000002H500Q504,192.54000000000002 504,196.54000000000002V250.0Z"/></g><text class="t-val" x="486.0" y="186.5" text-anchor="middle">33.8</text><text class="t-tick" x="616.0" y="272.0" text-anchor="middle">LiveCodeBench</text><line class="axis" x1="546" y1="250" x2="686" y2="250"/><g class="mark"><title>GPT-4o-0513, LiveCodeBench: 32.9</title><path class="s3" d="M556,250.0V198.07Q556,194.07 560,194.07H588Q592,194.07 592,198.07V250.0Z"/></g><text class="t-val" x="574.0" y="188.1" text-anchor="middle">32.9</text><g class="mark"><title>Sonnet 3.5 (Anthropic), LiveCodeBench: 38.9</title><path class="s4" d="M598,250.0V187.87Q598,183.87 602,183.87H630Q634,183.87 634,187.87V250.0Z"/></g><text class="t-val" x="616.0" y="177.9" text-anchor="middle">38.9</text><g class="mark"><title>R1-Distill-Qwen-1.5B, LiveCodeBench: 16.9</title><path class="s1" d="M640,250.0V225.27Q640,221.27 644,221.27H672Q676,221.27 676,225.27V250.0Z"/></g><text class="t-val" x="658.0" y="215.3" text-anchor="middle">16.9</text><text class="t-muted" x="30.0" y="308.0" text-anchor="start">Left of the line: maths. Right: science questions and code. Pass@1 in %, arXiv v1 Table 5.</text></svg><figcaption>GPT-4o, Sonnet 3.5 and the distilled 1.5B model on four benchmarks of Table 5. The 1.5B model is first on both maths tests and last on science and code.</figcaption></figure>

Why the split? A small model has little room for facts. GPQA asks for deep knowledge of biology, physics and chemistry, and reasoning cannot make up for facts the model does not know. Maths problems need fewer facts and more steps, and steps are what R1's text teaches.

> [!PAPER] · §3.2 · page 14
> [![The paragraph after Table 5, with outperform non-reasoning models like GPT-4o-0513 across the board, DeepSeek-R1-14B surpasses QwQ-32B-Preview on all evaluation metrics and exceed o1-mini on most benchmarks highlighted](/img/papers/deepseek-r1/p6-t5-text.png)](/img/papers/deepseek-r1/p6-t5-text.png)
>
> **What it says:** distillation lets the 7B student "outperform non-reasoning models like GPT-4o-0513 across the board". The 14B student "surpasses QwQ-32B-Preview on all evaluation metrics", and the 32B and 70B students "exceed o1-mini on most benchmarks".
>
> **What it means:**
>
> We checked each claim against the table above. The 14B claim holds on all six columns. The 32B and 70B claims hold on five of six; both lose on the Codeforces rating. The 7B claim has one small exception, GPQA, where it trails GPT-4o by 0.8 points.
>
> None of these exceptions changes the story. They are a reminder to read a table, not only the sentence about it.

> [!PAPER] · §3.2 · page 14
> [![The last sentences after Table 5, with applying RL to these distilled models yields significant further gains and only the results of the simple SFT-distilled models highlighted](/img/papers/deepseek-r1/p6-t5-rl.png)](/img/papers/deepseek-r1/p6-t5-rl.png)
>
> **What it says:** "applying RL to these distilled models yields significant further gains", but the paper presents "only the results of the simple SFT-distilled models".
>
> **What it means:**
>
> Distillation and RL are not rivals here. They stack: first copy a strong teacher, then let RL polish the student. The paper gives no numbers for this, so we cannot say how large the gain is.
>
> Many later open reasoning models followed exactly this order. They start from a distilled R1 model and add RL on top.

## Why the copy works: R1's text through two models

Section 1 says that the student copies R1's "reasoning patterns". We can measure that idea directly. Take text that R1 wrote, and ask two models how surprised they are by each token. One is the base model Qwen2.5-Math-1.5B, before distillation. The other is R1-Distill-Qwen-1.5B, after it. The loss is the same $$\mathcal{L}_{\text{SFT}}$$ as above, so a lower number means "this is the kind of text I write".

We use five texts. One is the trace from above. Three are real traces of the full DeepSeek-R1, from the public s1K-1.1 dataset. We took the three shortest correct ones among its first 300 rows, cut at 1,024 tokens. The last one is a control: an answer that the base model wrote itself to the duck problem.

```text
our trace (R1-Distill-1.5B)      732 tokens | base: loss 0.973, perplexity   2.64 | distilled: loss 0.361, perplexity   1.43
real R1 trace 1 (s1K-1.1)       1024 tokens | base: loss 0.480, perplexity   1.62 | distilled: loss 0.259, perplexity   1.30
real R1 trace 2 (s1K-1.1)       1024 tokens | base: loss 1.476, perplexity   4.38 | distilled: loss 0.998, perplexity   2.71
real R1 trace 3 (s1K-1.1)       1024 tokens | base: loss 0.701, perplexity   2.02 | distilled: loss 0.458, perplexity   1.58
control: base model answer       107 tokens | base: loss 0.250, perplexity   1.28 | distilled: loss 0.809, perplexity   2.25
same response token ids under both tokenizers: True (checked for every text)
```

<figure class="fig"><svg viewBox="0 0 720 350" role="img" aria-label="Bars of the mean per-token loss of five texts under the base model and the distilled model. On the four R1-style texts the distilled model has a much lower loss; on the base model's own answer the base model is lower."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Mean loss per response token (nats, lower = the model finds the text more likely)</text><rect class="s3" x="230" y="40" width="12" height="12" rx="2"/><text class="t-note" x="248.0" y="50.0" text-anchor="start">Qwen2.5-Math-1.5B (base)</text><rect class="s1" x="432.8" y="40" width="12" height="12" rx="2"/><text class="t-note" x="450.8" y="50.0" text-anchor="start">R1-Distill-Qwen-1.5B</text><text class="t-tick" x="220.0" y="94.0" text-anchor="end">our trace (R1-Distill-1.5B)</text><g class="mark"><title>our trace (R1-Distill-1.5B), base: loss 0.973, perplexity 2.64</title><path class="s3" d="M230,76H460.92344499413934Q464.92344499413934,76 464.92344499413934,80V86Q464.92344499413934,90 460.92344499413934,90H230Z"/></g><text class="t-val" x="469.9" y="88.0" text-anchor="start">0.973</text><g class="mark"><title>our trace (R1-Distill-1.5B), distilled: loss 0.361, perplexity 1.43</title><path class="s1" d="M230,93H313.21035623196775Q317.21035623196775,93 317.21035623196775,97V103Q317.21035623196775,107 313.21035623196775,107H230Z"/></g><text class="t-val" x="322.2" y="105.0" text-anchor="start">0.361</text><text class="t-tick" x="220.0" y="140.0" text-anchor="end">real R1 trace 1 (s1K-1.1)</text><g class="mark"><title>real R1 trace 1 (s1K-1.1), base: loss 0.480, perplexity 1.62</title><path class="s3" d="M230,122H341.95190458878426Q345.95190458878426,122 345.95190458878426,126V132Q345.95190458878426,136 341.95190458878426,136H230Z"/></g><text class="t-val" x="351.0" y="134.0" text-anchor="start">0.480</text><g class="mark"><title>real R1 trace 1 (s1K-1.1), distilled: loss 0.259, perplexity 1.30</title><path class="s1" d="M230,139H288.6168260368264Q292.6168260368264,139 292.6168260368264,143V149Q292.6168260368264,153 288.6168260368264,153H230Z"/></g><text class="t-val" x="297.6" y="151.0" text-anchor="start">0.259</text><text class="t-tick" x="220.0" y="186.0" text-anchor="end">real R1 trace 2 (s1K-1.1)</text><g class="mark"><title>real R1 trace 2 (s1K-1.1), base: loss 1.476, perplexity 4.38</title><path class="s3" d="M230,168H582.5217391304348Q586.5217391304348,168 586.5217391304348,172V178Q586.5217391304348,182 582.5217391304348,182H230Z"/></g><text class="t-val" x="591.5" y="180.0" text-anchor="start">1.476</text><g class="mark"><title>real R1 trace 2 (s1K-1.1), distilled: loss 0.998, perplexity 2.71</title><path class="s1" d="M230,185H467.17511800988893Q471.17511800988893,185 471.17511800988893,189V195Q471.17511800988893,199 467.17511800988893,199H230Z"/></g><text class="t-val" x="476.2" y="197.0" text-anchor="start">0.998</text><text class="t-tick" x="220.0" y="232.0" text-anchor="end">real R1 trace 3 (s1K-1.1)</text><g class="mark"><title>real R1 trace 3 (s1K-1.1), base: loss 0.701, perplexity 2.02</title><path class="s3" d="M230,214H395.36661926107865Q399.36661926107865,214 399.36661926107865,218V224Q399.36661926107865,228 395.36661926107865,228H230Z"/></g><text class="t-val" x="404.4" y="226.0" text-anchor="start">0.701</text><g class="mark"><title>real R1 trace 3 (s1K-1.1), distilled: loss 0.458, perplexity 1.58</title><path class="s1" d="M230,231H336.5746738974601Q340.5746738974601,231 340.5746738974601,235V241Q340.5746738974601,245 336.5746738974601,245H230Z"/></g><text class="t-val" x="345.6" y="243.0" text-anchor="start">0.458</text><text class="t-tick" x="220.0" y="278.0" text-anchor="end">control: base model answer</text><g class="mark"><title>control: base model answer, base: loss 0.250, perplexity 1.28</title><path class="s3" d="M230,260H286.4123913917024Q290.4123913917024,260 290.4123913917024,264V270Q290.4123913917024,274 286.4123913917024,274H230Z"/></g><text class="t-val" x="295.4" y="272.0" text-anchor="start">0.250</text><g class="mark"><title>control: base model answer, distilled: loss 0.809, perplexity 2.25</title><path class="s1" d="M230,277H421.35584049633553Q425.35584049633553,277 425.35584049633553,281V287Q425.35584049633553,291 421.35584049633553,291H230Z"/></g><text class="t-val" x="430.4" y="289.0" text-anchor="start">0.809</text><text class="t-muted" x="20.0" y="336.0" text-anchor="start">Each text is at most 1,024 tokens. The last row is an answer the base model wrote itself.</text></svg><figcaption>The mean loss per token of five texts under the base model and the distilled model. Lower means the model finds the text more likely.</figcaption></figure>

The pattern is clear. On every R1-style text, the distilled model has the lower loss. On our trace it falls from 0.973 to 0.361 nats per token, so the perplexity falls from 2.64 to 1.43. On the three real R1 traces it falls from 0.480 to 0.259, from 1.476 to 0.998, and from 0.701 to 0.458.

The control row turns the other way. On the base model's own answer, the base model has a loss of 0.250 and the distilled model 0.809. So the distilled model is not simply "better at text". It has moved toward one style of text, R1's, and away from the plain answers the base model used to write.

Two notes keep the test honest. Our trace was written by the distilled model itself, so this model must find it likely. That is why the three real R1 traces matter. Also, we used only the thinking part of our trace, 732 tokens. The end-of-thinking tag is one token for the R1 tokenizer and several for the base one. Every text gave identical token ids under both tokenizers, which the script checks.

<figure class="fig"><svg viewBox="0 0 720 280" role="img" aria-label="Two lines of per-token loss along the first real R1 trace. The base model line stays well above the distilled model line from start to end."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">Loss along a real R1 trace (running mean over 32 tokens)</text><line class="grid" x1="50" y1="234.0" x2="570" y2="234.0"/><text class="t-tick" x="42.0" y="238.0" text-anchor="end">0.0</text><line class="grid" x1="50" y1="201.7" x2="570" y2="201.7"/><text class="t-tick" x="42.0" y="205.7" text-anchor="end">0.5</text><line class="grid" x1="50" y1="169.3" x2="570" y2="169.3"/><text class="t-tick" x="42.0" y="173.3" text-anchor="end">1.0</text><line class="grid" x1="50" y1="137.0" x2="570" y2="137.0"/><text class="t-tick" x="42.0" y="141.0" text-anchor="end">1.5</text><line class="grid" x1="50" y1="104.7" x2="570" y2="104.7"/><text class="t-tick" x="42.0" y="108.7" text-anchor="end">2.0</text><line class="grid" x1="50" y1="72.3" x2="570" y2="72.3"/><text class="t-tick" x="42.0" y="76.3" text-anchor="end">2.5</text><line class="grid" x1="50" y1="40.0" x2="570" y2="40.0"/><text class="t-tick" x="42.0" y="44.0" text-anchor="end">3.0</text><text class="t-tick" x="50.0" y="252.0" text-anchor="middle">0</text><text class="t-tick" x="180.1" y="252.0" text-anchor="middle">256</text><text class="t-tick" x="310.3" y="252.0" text-anchor="middle">512</text><text class="t-tick" x="440.4" y="252.0" text-anchor="middle">768</text><text class="t-tick" x="310.0" y="272.0" text-anchor="middle">token position in the trace</text><polyline class="l3" points="50.0,40.0 51.5,60.8 53.0,84.1 54.6,126.4 56.1,136.7 57.6,149.4 59.1,149.0 60.7,149.1 62.2,147.1 63.7,151.4 65.2,155.3 66.8,171.5 68.3,184.6 69.8,188.0 71.3,191.4 72.9,196.0 74.4,195.7 75.9,205.8 77.4,214.4 79.0,224.7 80.5,192.8 82.0,171.0 83.5,160.0 85.1,139.3 86.6,137.0 88.1,136.9 89.6,132.9 91.2,131.0 92.7,130.9 94.2,130.8 95.7,152.9 97.3,158.2 98.8,184.1 100.3,187.6 101.8,209.3 103.4,203.9 104.9,200.2 106.4,196.9 107.9,197.5 109.5,195.6 111.0,181.7 112.5,187.4 114.0,190.6 115.6,193.4 117.1,187.5 118.6,193.6 120.1,196.5 121.7,200.4 123.2,180.6 124.7,173.0 126.2,183.0 127.8,188.4 129.3,184.5 130.8,182.7 132.3,186.8 133.9,187.9 135.4,185.2 136.9,180.7 138.4,176.8 140.0,201.9 141.5,206.2 143.0,206.6 144.5,207.3 146.1,214.1 147.6,213.8 149.1,214.6 150.6,215.5 152.2,218.7 153.7,227.6 155.2,227.6 156.7,227.6 158.3,228.2 159.8,223.1 161.3,203.9 162.8,189.6 164.4,187.9 165.9,188.5 167.4,188.6 168.9,188.8 170.5,189.2 172.0,189.3 173.5,189.3 175.0,189.8 176.6,203.7 178.1,204.0 179.6,217.4 181.1,214.7 182.7,209.1 184.2,209.1 185.7,203.6 187.2,204.5 188.8,204.5 190.3,206.5 191.8,215.8 193.3,213.5 194.9,213.6 196.4,218.4 197.9,224.5 199.4,224.6 201.0,211.0 202.5,213.8 204.0,213.8 205.5,210.5 207.1,211.2 208.6,214.2 210.1,214.3 211.6,214.3 213.2,214.3 214.7,214.3 216.2,224.3 217.7,230.6 219.3,214.5 220.8,212.2 222.3,208.6 223.8,207.7 225.4,207.7 226.9,207.6 228.4,207.6 229.9,207.4 231.5,207.4 233.0,207.4 234.5,219.4 236.0,223.4 237.6,224.3 239.1,231.6 240.6,232.1 242.1,232.1 243.7,232.1 245.2,232.1 246.7,232.3 248.2,224.9 249.8,224.8 251.3,222.6 252.8,221.7 254.3,214.5 255.9,202.6 257.4,194.3 258.9,178.0 260.4,176.6 262.0,161.9 263.5,160.6 265.0,160.7 266.5,159.2 268.1,161.4 269.6,138.7 271.1,137.6 272.6,136.0 274.2,139.8 275.7,149.3 277.2,152.7 278.7,153.8 280.3,159.1 281.8,163.8 283.3,163.8 284.8,187.8 286.4,195.1 287.9,201.0 289.4,192.4 290.9,196.4 292.5,203.3 294.0,206.4 295.5,204.1 297.0,204.3 298.6,204.3 300.1,204.3 301.6,204.3 303.1,205.6 304.7,209.2 306.2,217.8 307.7,219.3 309.2,222.5 310.8,225.6 312.3,228.5 313.8,228.5 315.3,228.5 316.9,228.5 318.4,228.5 319.9,214.9 321.4,218.0 323.0,219.6 324.5,219.3 326.0,219.5 327.5,219.5 329.1,219.5 330.6,219.5 332.1,219.5 333.6,219.5 335.2,228.0 336.7,224.9 338.2,222.8 339.7,210.5 341.3,204.9 342.8,191.5 344.3,179.2 345.8,176.0 347.4,176.0 348.9,175.9 350.4,175.9 351.9,182.5 353.5,184.7 355.0,187.1 356.5,193.1 358.0,197.9 359.6,213.1 361.1,224.7 362.6,226.2 364.1,225.9 365.7,225.8 367.2,225.8 368.7,225.8 370.2,223.6 371.8,229.8 373.3,229.9 374.8,230.6 376.3,230.6 377.9,221.9 379.4,221.9 380.9,222.3 382.4,222.4 384.0,219.3 385.5,214.1 387.0,216.2 388.5,216.7 390.1,216.7 391.6,216.7 393.1,225.7 394.6,225.7 396.2,225.7 397.7,218.7 399.2,213.3 400.7,216.4 402.3,221.5 403.8,221.5 405.3,221.5 406.8,201.8 408.4,201.1 409.9,197.2 411.4,193.7 412.9,187.4 414.5,199.4 416.0,193.1 417.5,190.6 419.0,186.8 420.6,186.2 422.1,184.4 423.6,192.8 425.1,176.3 426.7,171.1 428.2,173.7 429.7,171.0 431.2,166.6 432.8,169.2 434.3,171.0 435.8,174.7 437.3,175.3 438.9,185.9 440.4,205.2 441.9,210.0 443.4,215.6 445.0,215.6 446.5,224.5 448.0,221.9 449.5,225.3 451.1,225.4 452.6,225.5 454.1,225.5 455.6,225.5 457.2,225.5 458.7,225.4 460.2,225.4 461.7,225.4 463.3,231.3 464.8,233.6 466.3,233.6 467.8,233.6 469.4,233.6 470.9,233.6 472.4,233.2 473.9,233.3 475.5,233.3 477.0,233.3 478.5,233.3 480.0,232.9 481.6,228.4 483.1,228.4 484.6,228.4 486.1,228.4 487.7,228.4 489.2,228.8 490.7,228.7 492.2,228.7 493.8,226.4 495.3,226.8 496.8,231.6 498.3,231.6 499.9,231.6 501.4,231.6 502.9,231.4 504.4,231.4 506.0,229.7 507.5,229.4 509.0,229.8 510.5,230.5 512.1,223.2 513.6,221.8 515.1,221.3 516.6,220.8 518.2,220.8 519.7,219.6 521.2,220.6 522.7,198.5 524.3,191.8 525.8,187.4 527.3,174.1 528.8,179.7 530.4,178.8 531.9,178.9 533.4,178.9 534.9,175.3 536.5,174.1 538.0,176.5 539.5,188.2 541.0,195.1 542.6,210.3 544.1,202.2 545.6,205.3 547.1,205.7 548.7,194.0 550.2,196.4 551.7,196.4 553.2,196.8 554.8,203.5 556.3,205.3 557.8,206.2 559.3,214.4 560.9,213.6 562.4,213.8 563.9,218.9 565.4,224.2 567.0,219.9 568.5,221.6 570.0,216.9"/><rect class="s3" x="566.0" y="51" width="10" height="10" rx="2"/><text class="t-note" x="580.0" y="60.0" text-anchor="start">base: mean 0.480</text><polyline class="l1" points="50.0,207.6 51.5,223.5 53.0,199.4 54.6,209.8 56.1,213.2 57.6,212.5 59.1,201.5 60.7,205.2 62.2,193.2 63.7,197.3 65.2,200.3 66.8,201.0 68.3,205.6 69.8,207.7 71.3,207.9 72.9,209.7 74.4,203.4 75.9,212.0 77.4,217.4 79.0,224.7 80.5,221.2 82.0,211.4 83.5,206.1 85.1,202.9 86.6,202.7 88.1,203.3 89.6,203.2 91.2,208.9 92.7,208.8 94.2,208.8 95.7,207.7 97.3,210.8 98.8,225.4 100.3,223.5 101.8,226.0 103.4,224.6 104.9,222.6 106.4,218.4 107.9,215.6 109.5,215.6 111.0,211.3 112.5,212.3 114.0,210.8 115.6,213.0 117.1,205.0 118.6,204.5 120.1,206.4 121.7,203.2 123.2,194.4 124.7,193.5 126.2,195.7 127.8,196.5 129.3,195.1 130.8,196.1 132.3,196.1 133.9,205.9 135.4,201.5 136.9,196.7 138.4,196.1 140.0,209.7 141.5,212.3 143.0,213.3 144.5,213.1 146.1,216.4 147.6,216.4 149.1,217.7 150.6,217.5 152.2,222.1 153.7,230.4 155.2,230.4 156.7,230.4 158.3,230.9 159.8,229.8 161.3,228.1 162.8,223.5 164.4,223.4 165.9,223.5 167.4,223.8 168.9,223.8 170.5,225.1 172.0,225.0 173.5,225.0 175.0,224.5 176.6,221.9 178.1,222.0 179.6,225.8 181.1,221.3 182.7,218.0 184.2,218.0 185.7,215.8 187.2,215.9 188.8,215.9 190.3,216.7 191.8,220.9 193.3,219.9 194.9,219.9 196.4,224.4 197.9,228.4 199.4,228.6 201.0,215.2 202.5,215.9 204.0,215.9 205.5,212.7 207.1,213.3 208.6,215.9 210.1,215.9 211.6,215.9 213.2,215.9 214.7,215.9 216.2,224.3 217.7,230.8 219.3,226.0 220.8,220.4 222.3,221.0 223.8,220.0 225.4,220.0 226.9,220.0 228.4,220.0 229.9,219.2 231.5,219.1 233.0,219.1 234.5,222.1 236.0,227.8 237.6,228.0 239.1,230.5 240.6,231.5 242.1,231.5 243.7,231.5 245.2,231.5 246.7,232.3 248.2,223.8 249.8,221.0 251.3,218.3 252.8,215.1 254.3,210.6 255.9,197.5 257.4,195.8 258.9,184.1 260.4,176.8 262.0,174.1 263.5,173.4 265.0,179.2 266.5,181.3 268.1,182.8 269.6,174.8 271.1,177.4 272.6,172.6 274.2,168.7 275.7,181.8 277.2,174.4 278.7,176.4 280.3,176.1 281.8,181.3 283.3,181.3 284.8,191.6 286.4,196.9 287.9,202.7 289.4,211.5 290.9,218.8 292.5,216.0 294.0,225.4 295.5,225.0 297.0,225.4 298.6,225.4 300.1,225.4 301.6,225.4 303.1,225.5 304.7,226.4 306.2,224.6 307.7,228.2 309.2,229.6 310.8,231.1 312.3,231.3 313.8,231.3 315.3,231.3 316.9,231.3 318.4,231.3 319.9,225.4 321.4,224.2 323.0,226.2 324.5,226.1 326.0,226.2 327.5,226.2 329.1,226.2 330.6,226.2 332.1,226.2 333.6,226.2 335.2,230.7 336.7,231.4 338.2,231.0 339.7,223.0 341.3,221.1 342.8,213.8 344.3,211.5 345.8,210.0 347.4,210.0 348.9,210.0 350.4,210.0 351.9,210.9 353.5,212.3 355.0,212.9 356.5,211.2 358.0,213.3 359.6,220.8 361.1,222.9 362.6,223.8 364.1,223.8 365.7,223.8 367.2,223.8 368.7,223.8 370.2,223.1 371.8,232.8 373.3,231.7 374.8,231.7 376.3,231.6 377.9,231.8 379.4,231.8 380.9,231.7 382.4,231.7 384.0,228.4 385.5,216.9 387.0,217.5 388.5,218.4 390.1,218.5 391.6,218.5 393.1,218.7 394.6,218.7 396.2,218.6 397.7,216.8 399.2,211.3 400.7,222.0 402.3,226.1 403.8,226.3 405.3,226.4 406.8,221.7 408.4,217.2 409.9,215.7 411.4,213.3 412.9,211.6 414.5,218.8 416.0,217.0 417.5,212.0 419.0,211.2 420.6,206.7 422.1,205.8 423.6,208.3 425.1,207.4 426.7,205.7 428.2,207.4 429.7,208.0 431.2,206.7 432.8,206.0 434.3,208.9 435.8,209.3 437.3,213.9 438.9,215.9 440.4,220.6 441.9,222.6 443.4,226.0 445.0,225.7 446.5,227.1 448.0,224.2 449.5,226.8 451.1,228.3 452.6,227.7 454.1,227.8 455.6,227.8 457.2,227.8 458.7,227.8 460.2,227.5 461.7,227.7 463.3,232.1 464.8,232.4 466.3,232.4 467.8,233.0 469.4,233.0 470.9,233.0 472.4,232.9 473.9,232.7 475.5,232.5 477.0,232.5 478.5,232.3 480.0,227.6 481.6,225.8 483.1,225.8 484.6,225.9 486.1,225.9 487.7,225.9 489.2,226.2 490.7,225.8 492.2,226.3 493.8,226.1 495.3,226.6 496.8,232.8 498.3,232.8 499.9,232.8 501.4,232.7 502.9,231.8 504.4,232.0 506.0,230.6 507.5,230.6 509.0,230.4 510.5,230.9 512.1,228.6 513.6,228.5 515.1,227.7 516.6,227.3 518.2,227.3 519.7,223.5 521.2,224.8 522.7,219.3 524.3,218.3 525.8,215.8 527.3,214.7 528.8,216.9 530.4,216.3 531.9,216.6 533.4,215.7 534.9,215.4 536.5,220.1 538.0,218.6 539.5,215.4 541.0,216.0 542.6,219.3 544.1,217.5 545.6,218.1 547.1,218.2 548.7,208.1 550.2,209.3 551.7,209.3 553.2,208.8 554.8,213.4 556.3,220.2 557.8,220.3 559.3,222.1 560.9,221.3 562.4,221.4 563.9,226.2 565.4,229.8 567.0,229.7 568.5,226.2 570.0,224.2"/><rect class="s1" x="566.0" y="71" width="10" height="10" rx="2"/><text class="t-note" x="580.0" y="80.0" text-anchor="start">distilled: mean 0.259</text></svg><figcaption>The loss of every token of a real R1 trace, smoothed, under the base model and the distilled model.</figcaption></figure>

Which tokens moved most? The script averages the loss of each token over the three real R1 traces. It keeps tokens seen at least five times and sorts them by the drop:

```text
tokens of the real R1 traces with the largest gap (seen at least 5 times; mean loss base -> distilled):
  'Okay'         x  5    6.36 ->  0.90
  'First'        x  5    4.04 ->  0.34
  ' have'        x  6    3.82 ->  0.97
  ' right'       x  6    3.47 ->  1.13
  ' Let'         x 10    3.39 ->  1.09
  ' me'          x  8    2.74 ->  0.50
  ' susceptible' x  6    2.22 ->  0.45
  ' here'        x  5    3.70 ->  2.35
  ' so'          x 10    1.90 ->  0.59
  ' *'           x 11    1.38 ->  0.08
  ' \n\n'        x  5    2.30 ->  1.00
  ' that'        x 15    1.85 ->  0.61
  word 'Wait': x2, base 13.71, distilled 4.03 nats
  word ' wait': x1, base 4.02, distilled 0.79 nats
  word ' Hmm': x3, base 11.75, distilled 1.00 nats
  word 'Okay': x5, base 6.36, distilled 0.90 nats
  word ' Let': x10, base 3.39, distilled 1.09 nats
```

The top of the list is not maths. It is the voice of a thinker talking to itself: "Okay", "First", " Let", " me", " right". The words that mark R1's self-checks move the most of all. "Wait" costs the base model 13.71 nats, which is a probability of about one in a million, and the distilled model 4.03. " Hmm" goes from 11.75 to 1.00. These are the "reasoning patterns" of §1, visible as numbers: the student learned when to stop, doubt and look again.

## Our Table 5 on a laptop

The paper scores six students on five benchmarks with up to 32,768 tokens per answer. We can do a small part of that: the same base model and its released student on the same maths problems. We drew 16 problems of levels 1 to 3 from MATH-500, with a fixed seed. Each model wrote 4 answers per problem, with temperature 0.6 and top-p 0.95 as in §3. Each answer may use at most 2,048 new tokens, 16 times less than the paper allows.

We grade with one policy, the same as in [Part 4](part-4-r1-zero-and-the-aha-moment.md). An answer that hits the 2,048-token limit has no final answer, so it counts as wrong. For the distilled model, we grade only the text after `</think>`: the summary it writes once it has finished thinking. An answer that never closes `</think>` counts as wrong. The base model writes no thinking block, so we grade its whole finished text. math-verify then compares the last boxed answer with the reference.

The generation script saved every answer, and `r1_part6_regrade.py` grades the saved answers with this policy. No model runs again:

```text
$ python r1_part6_regrade.py
== 128 saved answers (16 problems x 4 samples x 2 models), max 2048 new tokens
== policy: cut off at the limit = wrong; distilled: grade only the text after </think>, no </think> = wrong
base      pass@1 0.328 (21/64) | problems solved at least once 10/16 | level 1: 0.25, level 2: 0.29, level 3: 0.45
          finished inside the limit 34/64 | closed </think> 0/64 | mean tokens 1263 (correct answers only: 538)
          old grading (whole text, cut-off answers included): 24/64 = 0.375
          kinder: finished answers graded at their first box: 21/64 = 0.328
distilled pass@1 0.484 (31/64) | problems solved at least once 11/16 | level 1: 0.70, level 2: 0.38, level 3: 0.40
          finished inside the limit 31/64 | closed </think> 36/64 | mean tokens 1693 (correct answers only: 1316)
          old grading (whole text, cut-off answers included): 38/64 = 0.594
distilled answers that finished: 31, of which correct: 31/31
```

[![Terminal output of r1_part6_regrade.py: the saved answers of both models regraded with one policy](/img/papers/deepseek-r1/part6_regrade-run.png)](/img/papers/deepseek-r1/part6_regrade-run.png)

<figure class="fig"><svg viewBox="0 0 720 250" role="img" aria-label="Bars comparing the base model and the distilled model in our run: accuracy, mean answer length, and how many answers ran into the token limit."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Our run: 16 MATH-500 problems (levels 1 to 3), 4 samples each, max 2048 new tokens</text><rect class="s3" x="20" y="40" width="12" height="12" rx="2"/><text class="t-note" x="38.0" y="50.0" text-anchor="start">Qwen2.5-Math-1.5B (base)</text><rect class="s1" x="222.8" y="40" width="12" height="12" rx="2"/><text class="t-note" x="240.8" y="50.0" text-anchor="start">R1-Distill-Qwen-1.5B</text><text class="t-tick" x="210.0" y="98.0" text-anchor="end">pass@1 (%)</text><g class="mark"><title>base, pass@1 (%): 32.8</title><path class="s3" d="M220,80H347.25Q351.25,80 351.25,84V90Q351.25,94 347.25,94H220Z"/></g><text class="t-val" x="356.2" y="92.0" text-anchor="start">32.8</text><g class="mark"><title>distilled, pass@1 (%): 48.4</title><path class="s1" d="M220,98H409.75Q413.75,98 413.75,102V108Q413.75,112 409.75,112H220Z"/></g><text class="t-val" x="418.8" y="110.0" text-anchor="start">48.4</text><text class="t-tick" x="210.0" y="154.0" text-anchor="end">mean answer length (tokens)</text><g class="mark"><title>base, mean answer length (tokens): 1263</title><path class="s3" d="M220,136H462.722412109375Q466.722412109375,136 466.722412109375,140V146Q466.722412109375,150 462.722412109375,150H220Z"/></g><text class="t-val" x="471.7" y="148.0" text-anchor="start">1263</text><g class="mark"><title>distilled, mean answer length (tokens): 1693</title><path class="s1" d="M220,154H546.7403564453125Q550.7403564453125,154 550.7403564453125,158V164Q550.7403564453125,168 546.7403564453125,168H220Z"/></g><text class="t-val" x="555.7" y="166.0" text-anchor="start">1693</text><text class="t-tick" x="210.0" y="210.0" text-anchor="end">answers that hit the limit</text><g class="mark"><title>base, answers that hit the limit: 30</title><path class="s3" d="M220,192H403.5Q407.5,192 407.5,196V202Q407.5,206 403.5,206H220Z"/></g><text class="t-val" x="412.5" y="204.0" text-anchor="start">30</text><g class="mark"><title>distilled, answers that hit the limit: 33</title><path class="s1" d="M220,210H422.25Q426.25,210 426.25,214V220Q426.25,224 422.25,224H220Z"/></g><text class="t-val" x="431.2" y="222.0" text-anchor="start">33</text><text class="t-muted" x="20.0" y="238.0" text-anchor="start">A cut-off answer counts as wrong; the distilled model is graded after &lt;/think&gt;. MacBook (mps), bf16.</text></svg><figcaption>Our run on 16 MATH-500 problems: accuracy under one grading policy (a cut-off answer is wrong, the distilled model is graded after its thinking), mean answer length, and answers that hit the 2,048-token limit, for the base model and the distilled model.</figcaption></figure>

The distilled model is right on 31 of 64 answers, a pass@1 of 0.484. The base model is right on 21 of 64, or 0.328. The distilled model solves 11 of the 16 problems at least once; the base model solves 10. That is the direction of Table 5, on a much smaller test.

Look at the "old grading" lines. Our first grading read the whole text of every answer, cut off or not, and gave 38 and 24. That was too kind. A cut-off answer can hold a boxed number from the middle of its thinking, a step the model had not yet checked. The old grader counted it. The policy above does not.

The lengths tell the other half. The distilled model's correct answers average 1,316 tokens; the base model's correct answers average 538. The student has learned to think for a long time, as R1 does. With a cap of 2,048 tokens, 33 of its 64 answers were cut off before they ended.

Every one of the 31 answers that did finish was right. So the cap, not the model, decides most of this score. With the paper's 32,768 tokens, this model would likely score higher than 0.484.

The base model also hit the limit, 30 times, but for a different reason. We read its long answers. It often gives an answer and then keeps going. It writes a new problem of its own and solves that, or it repeats one phrase until the cap. A base model was never taught where an answer ends. Under our policy these answers are wrong, because they never end.

The "kinder" line grades each finished base answer at its first box instead of its last; it changes nothing (21 either way).

Sixteen problems with four samples each is a small test, so read these numbers as a direction, not as a benchmark score.

## Distillation or RL? {§4.1}

> [!PAPER] · §4.1 · page 14
> [![The start of Section 4.1, with can the model achieve comparable performance through the large-scale RL training without distillation, large-scale RL training on Qwen-32B-Base using math, code, and STEM data, training for over 10K steps, and DeepSeek-R1-Zero-Qwen-32B highlighted](/img/papers/deepseek-r1/p6-s41-q.png)](/img/papers/deepseek-r1/p6-s41-q.png)
>
> **What it says:** the question is whether a small model "can achieve comparable performance through the large-scale RL training discussed in the paper without distillation". To answer it, the authors run "large-scale RL training on Qwen-32B-Base using math, code, and STEM data, training for over 10K steps". The result is DeepSeek-R1-Zero-Qwen-32B.
>
> **What it means:**
>
> This is the control experiment that makes the distillation result mean something. Take the same base model as the 32B student, Qwen2.5-32B. Give it the R1-Zero treatment of [Part 4](part-4-r1-zero-and-the-aha-moment.md): RL with rule-based rewards, from the base model, with no copied answers. Then compare.
>
> "Over 10K steps" is a lot of RL. Each RL step samples a group of answers for every prompt in the batch, scores them and updates the model. The paper does not give the batch size or the group size for this run.

> [!PAPER] · Table 6 · page 15
> [![Table 6 of the paper: QwQ-32B-Preview, DeepSeek-R1-Zero-Qwen-32B and DeepSeek-R1-Distill-Qwen-32B on AIME 2024 pass@1 and cons@64, MATH-500, GPQA Diamond and LiveCodeBench](/img/papers/deepseek-r1/p6-table6.png)](/img/papers/deepseek-r1/p6-table6.png)
>
> **What it says:** QwQ-32B-Preview gets 50.0, 60.0, 90.6, 54.5 and 41.9. R1-Zero-Qwen-32B gets 47.0, 60.0, 91.6, 55.0 and 40.2. R1-Distill-Qwen-32B gets 72.6, 83.3, 94.3, 62.1 and 57.2.
>
> **What it means:**
>
> The RL model and QwQ-32B-Preview are almost twins: within 3 points on every column, and equal on cons@64. So RL on a 32B base works. It produces a real reasoning model, as good as the best open one of its time.
>
> The distilled model is in another class. It is 25.6 points higher on AIME pass@1, 23.3 higher on cons@64, 7.1 higher on GPQA and 17.0 higher on LiveCodeBench. Only MATH-500 is close, 94.3 against 91.6, because both are near the top of that test.

<figure class="fig"><svg viewBox="0 0 720 320" role="img" aria-label="Bars for QwQ-32B-Preview, R1-Zero-Qwen-32B and R1-Distill-Qwen-32B on five columns. The RL model is about level with QwQ-32B-Preview; the distilled model is clearly higher on every column."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Table 6: the same 32B base, RL alone against distillation (%)</text><rect class="s3" x="20" y="40" width="12" height="12" rx="2"/><text class="t-note" x="38.0" y="50.0" text-anchor="start">QwQ-32B-Preview</text><rect class="s4" x="158.0" y="40" width="12" height="12" rx="2"/><text class="t-note" x="176.0" y="50.0" text-anchor="start">R1-Zero-Qwen-32B (RL)</text><rect class="s1" x="339.20000000000005" y="40" width="12" height="12" rx="2"/><text class="t-note" x="357.2" y="50.0" text-anchor="start">R1-Distill-Qwen-32B (SFT)</text><text class="t-tick" x="84.0" y="272.0" text-anchor="middle">AIME pass@1</text><line class="axis" x1="20" y1="250" x2="148" y2="250"/><g class="mark"><title>QwQ-32B-Preview, AIME pass@1: 50.0</title><path class="s3" d="M26,250.0V169.0Q26,165.0 30,165.0H56Q60,165.0 60,169.0V250.0Z"/></g><text class="t-val" x="43.0" y="159.0" text-anchor="middle">50.0</text><g class="mark"><title>R1-Zero-Qwen-32B (RL), AIME pass@1: 47.0</title><path class="s4" d="M66,250.0V174.1Q66,170.1 70,170.1H96Q100,170.1 100,174.1V250.0Z"/></g><text class="t-val" x="83.0" y="164.1" text-anchor="middle">47.0</text><g class="mark"><title>R1-Distill-Qwen-32B (SFT), AIME pass@1: 72.6</title><path class="s1" d="M106,250.0V130.58Q106,126.58000000000001 110,126.58000000000001H136Q140,126.58000000000001 140,130.58V250.0Z"/></g><text class="t-val" x="123.0" y="120.6" text-anchor="middle">72.6</text><text class="t-tick" x="222.0" y="272.0" text-anchor="middle">AIME cons@64</text><line class="axis" x1="158" y1="250" x2="286" y2="250"/><g class="mark"><title>QwQ-32B-Preview, AIME cons@64: 60.0</title><path class="s3" d="M164,250.0V152.0Q164,148.0 168,148.0H194Q198,148.0 198,152.0V250.0Z"/></g><text class="t-val" x="181.0" y="142.0" text-anchor="middle">60.0</text><g class="mark"><title>R1-Zero-Qwen-32B (RL), AIME cons@64: 60.0</title><path class="s4" d="M204,250.0V152.0Q204,148.0 208,148.0H234Q238,148.0 238,152.0V250.0Z"/></g><text class="t-val" x="221.0" y="142.0" text-anchor="middle">60.0</text><g class="mark"><title>R1-Distill-Qwen-32B (SFT), AIME cons@64: 83.3</title><path class="s1" d="M244,250.0V112.38999999999999Q244,108.38999999999999 248,108.38999999999999H274Q278,108.38999999999999 278,112.38999999999999V250.0Z"/></g><text class="t-val" x="261.0" y="102.4" text-anchor="middle">83.3</text><text class="t-tick" x="360.0" y="272.0" text-anchor="middle">MATH-500</text><line class="axis" x1="296" y1="250" x2="424" y2="250"/><g class="mark"><title>QwQ-32B-Preview, MATH-500: 90.6</title><path class="s3" d="M302,250.0V99.98000000000002Q302,95.98000000000002 306,95.98000000000002H332Q336,95.98000000000002 336,99.98000000000002V250.0Z"/></g><text class="t-val" x="319.0" y="90.0" text-anchor="middle">90.6</text><g class="mark"><title>R1-Zero-Qwen-32B (RL), MATH-500: 91.6</title><path class="s4" d="M342,250.0V98.28000000000003Q342,94.28000000000003 346,94.28000000000003H372Q376,94.28000000000003 376,98.28000000000003V250.0Z"/></g><text class="t-val" x="359.0" y="88.3" text-anchor="middle">91.6</text><g class="mark"><title>R1-Distill-Qwen-32B (SFT), MATH-500: 94.3</title><path class="s1" d="M382,250.0V93.69Q382,89.69 386,89.69H412Q416,89.69 416,93.69V250.0Z"/></g><text class="t-val" x="399.0" y="83.7" text-anchor="middle">94.3</text><text class="t-tick" x="498.0" y="272.0" text-anchor="middle">GPQA Diamond</text><line class="axis" x1="434" y1="250" x2="562" y2="250"/><g class="mark"><title>QwQ-32B-Preview, GPQA Diamond: 54.5</title><path class="s3" d="M440,250.0V161.35Q440,157.35 444,157.35H470Q474,157.35 474,161.35V250.0Z"/></g><text class="t-val" x="457.0" y="151.3" text-anchor="middle">54.5</text><g class="mark"><title>R1-Zero-Qwen-32B (RL), GPQA Diamond: 55.0</title><path class="s4" d="M480,250.0V160.5Q480,156.5 484,156.5H510Q514,156.5 514,160.5V250.0Z"/></g><text class="t-val" x="497.0" y="150.5" text-anchor="middle">55.0</text><g class="mark"><title>R1-Distill-Qwen-32B (SFT), GPQA Diamond: 62.1</title><path class="s1" d="M520,250.0V148.43Q520,144.43 524,144.43H550Q554,144.43 554,148.43V250.0Z"/></g><text class="t-val" x="537.0" y="138.4" text-anchor="middle">62.1</text><text class="t-tick" x="636.0" y="272.0" text-anchor="middle">LiveCodeBench</text><line class="axis" x1="572" y1="250" x2="700" y2="250"/><g class="mark"><title>QwQ-32B-Preview, LiveCodeBench: 41.9</title><path class="s3" d="M578,250.0V182.76999999999998Q578,178.76999999999998 582,178.76999999999998H608Q612,178.76999999999998 612,182.76999999999998V250.0Z"/></g><text class="t-val" x="595.0" y="172.8" text-anchor="middle">41.9</text><g class="mark"><title>R1-Zero-Qwen-32B (RL), LiveCodeBench: 40.2</title><path class="s4" d="M618,250.0V185.66Q618,181.66 622,181.66H648Q652,181.66 652,185.66V250.0Z"/></g><text class="t-val" x="635.0" y="175.7" text-anchor="middle">40.2</text><g class="mark"><title>R1-Distill-Qwen-32B (SFT), LiveCodeBench: 57.2</title><path class="s1" d="M658,250.0V156.76Q658,152.76 662,152.76H688Q692,152.76 692,156.76V250.0Z"/></g><text class="t-val" x="675.0" y="146.8" text-anchor="middle">57.2</text><text class="t-muted" x="20.0" y="308.0" text-anchor="start">R1-Zero-Qwen-32B: over 10K steps of RL on Qwen2.5-32B. R1-Distill-Qwen-32B: SFT on the 800k samples. arXiv v1 Table 6.</text></svg><figcaption>Table 6 as bars. On the same Qwen2.5-32B base, over 10K steps of RL give about the level of QwQ-32B-Preview; SFT on R1's 800k samples gives much more on every column.</figcaption></figure>

> [!PAPER] · §4.1 · page 15
> [![The text after Table 6, with achieves performance on par with QwQ-32B-Preview, performs significantly better than and across all benchmarks highlighted](/img/papers/deepseek-r1/p6-t6-text.png)](/img/papers/deepseek-r1/p6-t6-text.png)
>
> **What it says:** the RL model "achieves performance on par with QwQ-32B-Preview", while the distilled model "performs significantly better than" the RL model "across all benchmarks".
>
> **What it means:**
>
> Why should copying beat learning by trial and error on the same model? Think about how RL learns. It can only make more likely the answers that the model already writes now and then. If a good line of thought is very rare for the model, RL almost never sees it, and so it almost never gets rewarded.
>
> R1's patterns came from RL on a 671B model, which could find them because it was much stronger at the start. The 32B model alone finds weaker patterns. Distillation skips the search: it hands the student the patterns that the bigger model found. Our loss experiment above showed the same gap from the other side. The base 1.5B model finds R1's text very unlikely, so it would rarely write such text on its own.

> [!PAPER] · §4.1 · page 15
> [![The two conclusions of Section 4.1, with distilling more powerful models into smaller ones yields excellent results, require enormous computational power and may not even achieve the performance of distillation, and advancing beyond the boundaries of intelligence may still require more powerful base models and larger-scale reinforcement learning highlighted](/img/papers/deepseek-r1/p6-conclusions.png)](/img/papers/deepseek-r1/p6-conclusions.png)
>
> **What it says:** first, "distilling more powerful models into smaller ones yields excellent results", while small models that rely on large-scale RL "require enormous computational power and may not even achieve the performance of distillation". Second, "advancing beyond the boundaries of intelligence may still require more powerful base models and larger-scale reinforcement learning."
>
> **What it means:**
>
> The first conclusion is practical. If a strong teacher exists, copy it: it is cheaper and it works better. SFT on 800k samples costs one pass of the data two or three times. RL costs sampling, scoring and updating, for over 10K steps, and still ends lower.
>
> The second conclusion is the limit of the first. In this paper, every student scores below its teacher, R1, and the students learn only from text that R1 wrote and that passed a filter. That is an observation about these experiments, not a law: in other work, students have sometimes beaten their teachers. Still, someone must build a strong teacher first, and the paper says that pushing further "may still require" a bigger base model and more RL. In this paper, distillation spreads what the frontier model can do, and RL on a large model is what moved the frontier.

## A tiny distillation of our own

Can we watch the recipe work at toy size? We tried §2.3.3 and §2.4 in miniature. The teacher was R1-Distill-Qwen-1.5B, standing in for R1. The student was Qwen2.5-0.5B-Instruct.

The teacher wrote one answer for each of 128 GSM8K training problems, with at most 1,024 new tokens. We kept only the answers that ended and were right. Then the student got SFT on the kept answers with our Method 1 loss. We used 3 epochs, batch 1, AdamW, and a learning rate of $$10^{-5}$$ with cosine decay to one tenth.

```text
== 1. teacher deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B writes one trace per problem (128 GSM8K train problems, max 1024 tokens)
== 0. teacher distribution at response position 444 (after " $18 every day at the farmers' market.\n\n"):
   'Wait' 0.900, 'Let' 0.074, 'Hmm' 0.011, 'But' 0.005, 'Hold' 0.005; the trace has 'Wait' (p = 0.900)
traces written 128, finished and correct (kept) 35, mean length of kept 835 tokens, 391 s
== 3a. student Qwen/Qwen2.5-0.5B-Instruct before distillation (30 held-out GSM8K test problems x 2 samples)
before  accuracy 0.450 (27/60) | mean tokens 328 | with </think> 0/60 | with "Wait" 0/60 | hit the limit 0/60
== 2. SFT on the kept traces: 3 epochs, batch 1, AdamW, lr 1e-05 cosine to 1.0000000000000002e-06
  step    1/105  loss 0.8382  (mean of last 1)  3 s
  step   25/105  loss 0.7023  (mean of last 25)  52 s
  step   50/105  loss 0.4550  (mean of last 25)  162 s
  step   75/105  loss 0.3267  (mean of last 25)  476 s
  step  100/105  loss 0.1719  (mean of last 25)  712 s
```

The run did not finish. Our session was restarted at step 100 of 105, before the last evaluation, and the shared machine was then busy with a long training job. So we have no "after" numbers, and we do not guess them. What the log does show is still worth reading.

First, rejection sampling is expensive. Of 128 answers, only 35 both ended within 1,024 tokens and were right. Most of the rest were cut off while still thinking. The kept answers average 835 tokens, against 328 tokens for the student's own answers before training.

Second, the student starts far from the teacher's style. Before training it is right on 27 of 60 answers, and none of its 60 answers contains "Wait" or a thinking tag. Its training loss on the teacher's text then falls from 0.84 at the first step to 0.17 over steps 76 to 100. That fall is the same effect as in our comparison of the two 1.5B models: the student learns to find the teacher's text likely.

Third, the probe at the top of the log is the left half of the figure of the two losses. At one position of our duck trace, after the final answer, the teacher gives "Wait" a probability of 0.900, "Let" 0.074 and "Hmm" 0.011. The model checks its work even after it has the answer.

> [!TAKEAWAYS] Key takeaways
> - Distillation in R1 is plain SFT: six open models are fine-tuned on the 800k samples of R1's stage 3, with no RL stage, on purpose.
> - It is sequence-level distillation (Kim and Rush, 2016): the student learns R1's text, not R1's probabilities as in Hinton et al. (2015). On average over samples, the two losses are the same.
> - The loss is the next-token cross-entropy on the response tokens only; prompt tokens get the label -100. Our from-scratch loss matched transformers, PyTorch and TRL to $$1.25 \times 10^{-6}$$.
> - The students are Qwen2.5-Math-1.5B and 7B, Qwen2.5-14B and 32B, Llama-3.1-8B and Llama-3.3-70B-Instruct. The later version adds 2 to 3 epochs, batch 64 and a learning rate of 1e-4 for the 1.5B.
> - The 1.5B student beats GPT-4o and Sonnet 3.5 on maths only. AIME is 28.9 against 9.3 and 16.0; GPQA Diamond is 33.8 against 49.9 and 65.0.
> - The 14B student beats QwQ-32B-Preview on every column; the 32B and 70B students beat o1-mini on five of six.
> - On Qwen2.5-32B, over 10K steps of RL reach 47.0 on AIME, level with QwQ-32B-Preview; distillation reaches 72.6.
> - R1-Distill-Qwen-1.5B finds real R1 text far more likely than its base model does (loss 0.259 against 0.480 on one trace); "Wait" drops from 13.71 to 4.03 nats.
> - On 16 MATH-500 problems with a 2,048-token cap (cut-off = wrong), the distilled 1.5B scored 0.484 pass@1 and its base 0.328. All 31 distilled answers that finished were right, and its correct answers were more than twice as long.
> - Copying a stronger teacher is cheaper and better for small models; moving the frontier still needs bigger base models and more RL.

## The paper and the code

[Paper (arXiv 2501.12948)](https://arxiv.org/abs/2501.12948) · [Official repository](https://github.com/deepseek-ai/DeepSeek-R1) · [Our scripts for this series](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/deepseek-r1)
