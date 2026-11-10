---
title: "DeepSeek's Specialised Models: Code, Maths, Proofs, Vision and OCR"
description: "The side models behind DeepSeek's main line: Coder, DeepSeekMath and GRPO, the Lean provers, VL2, Janus and DeepSeek-OCR, built up from zero with small experiments."
date: 2026-11-10
tags: [deepseek, llm, code, math, theorem-proving, vision, ocr]
series: "DeepSeek, From the Ground Up"
series_part: 8
motif: blocks
accent: "#c98500"
---

In February 2024, DeepSeek released a 7-billion-parameter maths model. Its paper had two contributions. One was a pile of 120 billion maths tokens, fished out of the public web. The other was a small change to a reinforcement learning algorithm, so that it no longer needed a second network to judge the first. That change was called GRPO. A year later, GRPO trained DeepSeek-R1, the model that made the lab famous ([Part 4](deepseek-4-r1-and-reasoning-by-rl.md)).

DeepSeek's main models (V2, V3, R1, V4) get the headlines. Next to them runs a line of smaller, focused models for code, maths, formal proofs, images and documents. Many ideas were tried there first, and some flowed back into the main line. Most have open weights you can run today.

> [!TIP] The whole idea in one minute
> DeepSeek built a specialised model for each hard skill. **Coder** learned from whole repositories and learned to fill gaps in code. **DeepSeekMath** mined the web for maths and introduced **GRPO**, a cheaper reinforcement learning method. It also found that this RL mostly makes the model more consistent, not more capable.
>
> **DeepSeekMath-V2** learned to grade its own proofs. The **Prover** models write proofs that a computer program, Lean 4, checks line by line. **VL2** and **Janus** look at images; Janus can also draw them. **DeepSeek-OCR** asks a strange question: can a picture of a page hold the same text in ten times fewer tokens? The answer is "about 97% of the time, yes".

{{FIG:timeline|Every specialised release in this part, by family. Dates are the Hugging Face repository creation date or the arXiv date. The vision lane is the busiest; the code line ends when Coder-V2 is merged into V2.5 in September 2024.}}

## Why build side models at all?

A general chat model must be good at everything at once, so it is a poor place to test one idea. A focused model is a cleaner experiment: one skill, one benchmark, one change.

There is a second reason. Code and maths have **an automatic way to check the answer**: tests, a known result, a proof checker. Checkable answers make good rewards for reinforcement learning, and honest benchmarks.

> [!DEFINITION] Benchmark
> **HumanEval** and **MBPP** are small Python programming benchmarks: the model writes a function, and hidden tests check it. **GSM8K** has grade-school word problems; **MATH** has competition problems.

The table below is the map for this part.

| Family | First release | Sizes | What is new | Weights license |
|---|---|---|---|---|
| DeepSeek-Coder | Oct to Nov 2023 | 1.3B to 33B, dense | Repository-level data, fill-in-the-middle, 16K context | DeepSeek License (commercial use allowed) |
| Coder-V2 | Jun 2024 | 16B and 236B, MoE | Continued from a V2 checkpoint, 338 languages, 128K | DeepSeek License |
| DeepSeekMath | Feb 2024 | 7B, dense | 120B-token web maths corpus, GRPO | DeepSeek License |
| DeepSeekMath-V2 | Nov 2025 | 671B class, MoE | A verifier that grades proofs; self-verification | Apache 2.0 |
| Prover V1, V1.5, V2 | May 2024 to Apr 2025 | 7B; V2 also 671B | Machine-checked proofs in Lean 4 | DeepSeek License (V2 labels are mixed) |
| VL, VL2 | Mar 2024, Dec 2024 | 1.3B to 27B | Hybrid encoder; dynamic tiling with an MoE language side | DeepSeek License |
| Janus, JanusFlow, Janus-Pro | Oct 2024 to Jan 2025 | 1B to 7B | Two image encoders: one to understand, one to draw | DeepSeek License (some tags say MIT) |
| DeepSeek-OCR, OCR 2 | Oct 2025, Jan 2026 | about 3B, MoE decoder | Text stored as images: optical compression | MIT; OCR 2 Apache 2.0 |
| ESFT | Jul 2024 | V2-Lite architecture | Fine-tune only the experts a task uses | no license tag on the weights |
| GRM / SPCT | Apr 2025 | 27B (on Gemma-2) | A reward model that writes principles and critiques | not in the DeepSeek org (see below) |

Dates come from the Hugging Face API and arXiv, as collected for this series. Part 1 has the full catalogue ([Part 1](deepseek-1-every-model-and-what-is-open.md)).

## DeepSeek-Coder: learning from whole repositories

### The problem: code is not a pile of files

Early code models trained on single files, shuffled together. Real code does not live that way. A file `train.py` calls functions defined in `model.py`, which uses helpers from `utils.py`. If the model sees `train.py` alone, the calls look like magic. If it sees `train.py` before `utils.py`, it learns to call functions it has not seen yet.

DeepSeek-Coder (2023) fixed the order. It parsed each repository, found which file imports which, and sorted the files so that every file comes after the files it uses. Then it joined the sorted files into one long training sample, with the file path as a comment on top of each file.

> [!DEFINITION] Repository-level data
> Training samples built from a whole software project in a sensible order, not from single files. The model sees definitions before uses, the way a programmer reads a project.

{{FIG:repo_order|A small repository. The arrows point from a file to the files that import it. Sorting the graph puts utils.py first and train.py last, and the files are joined in that order into one training sample.}}

> [!PAPER] Guo et al. (2024), DeepSeek-Coder · §2.2 · page 4
> [![Page 4 of the DeepSeek-Coder paper, highlighting that files are arranged so the context each file relies on comes before it](/img/deepseek/papers/p8-coder-deps.png)](/img/deepseek/papers/p8-coder-deps.png)
>
> **What it says:** they "arrange these files in an order that ensures the context each file relies on is placed before that file in the input sequence."
>
> **What it means:**
>
> This is a topological sort, the same thing a build tool does when it decides what to compile first. The paper's algorithm has one twist: it picks the file with the fewest unmet dependencies, not only files with zero. Real projects have cycles (two files that import each other), and a strict sort would stop there.

The finished corpus had 87 programming languages and 798 GB of cleaned source code (paper Table 1, page 6). The training mix was 87% source code, 10% English text about code and 3% Chinese text. Every model saw 2 trillion tokens. Before training, any file that shared a 10-word string with HumanEval, MBPP, GSM8K or MATH was removed. This keeps the test questions out of the training data.

### Fill-in-the-middle

A chat model writes left to right. A programmer in an editor does not. The cursor sits in the middle of a file, with code above and below it. The model must write the missing piece so that it fits both sides.

> [!DEFINITION] Fill-in-the-middle (FIM)
> A training trick for code models. Cut a file into a prefix, a middle and a suffix. Put the prefix and the suffix first, then the middle. The model learns to write the middle after seeing both sides. "PSM" is this order: prefix, suffix, middle.

{{FIG:fim|Fill-in-the-middle on a three-line file. The middle line moves to the end of the training sample, after special marker tokens, so the model must predict it from the code on both sides.}}

There is a cost: a sample in FIM order does not teach plain left-to-right writing. DeepSeek tested four settings on a 1.3B model.

> [!PAPER] Guo et al. (2024), DeepSeek-Coder · §3.1.2 · page 8
> [![Page 8 of the DeepSeek-Coder paper, highlighting the choice of a 50% PSM rate and the FIM template with its sentinel tokens](/img/deepseek/papers/p8-coder-fim.png)](/img/deepseek/papers/p8-coder-fim.png)
>
> **What it says:** "we ultimately choose the 50% PSM rate as our preferred training policy."
>
> **What it means:**
>
> The page before this one shows the trade-off. With FIM on 100% of samples, the model is best at filling gaps and worst at plain completion. With 0%, the reverse. Half and half keeps most of both skills.
>
> The template at the bottom is what an editor plugin sends at run time. The code above the cursor is the prefix; the code below is the suffix. What the model writes after the end marker is the grey suggestion.

The models came in 1.3B, 6.7B and 33B sizes (plus a 5.7B variant), all dense, with a 16K context. On HumanEval, the 33B base model scored 56.1% and the 6.7B base scored 49.4%. The 6.7B base beat CodeLlama-Base 34B (48.2%), a model five times larger. After instruction tuning, the 33B scored 79.3%, against 76.2% for GPT-3.5-Turbo and 84.1% for GPT-4 (paper Table 3, greedy decoding).

In January 2024, DeepSeek-Coder-v1.5 7B took a different route. It started from the general DeepSeek-LLM-7B and continued training on 2T more tokens, mostly code. That model matters later: it is the starting point of DeepSeekMath.

## Coder-V2: code on top of a general model

By mid-2024, DeepSeek had DeepSeek-V2, a mixture-of-experts (MoE) model with Multi-head Latent Attention ([Part 2](deepseek-2-the-v3-architecture.md) explains both). Training a separate code model from scratch no longer made sense. A better start was a model that already knew language.

> [!PAPER] Zhu et al. (2024), DeepSeek-Coder-V2 · §3.3 · page 5
> [![Page 5 of the DeepSeek-Coder-V2 paper, highlighting that training continues from an intermediate DeepSeek-V2 checkpoint and the total of 10.2T tokens](/img/deepseek/papers/p8-coderv2-ckpt.png)](/img/deepseek/papers/p8-coderv2-ckpt.png)
>
> **What it says:** "we continue the pre-training process from an intermediate checkpoint of DeepSeek-V2," and the model was "exposed to a total of 10.2T high-quality tokens."
>
> **What it means:**
>
> The checkpoint had seen 4.2T tokens of general text. Coder-V2 added 6T more, with a new mix: 60% code, 10% maths, 30% natural language. So it is mostly a general model that went to coding school for a long time.

The code part of the new mix, 1,170B tokens, was collected with the same web-mining loop that DeepSeekMath had built (next section). The number of programming languages went from 86 to 338 (the Coder-V2 paper counts 86 for the first Coder; its own paper said 87). The context grew from 16K to 128K. Two sizes shipped: 236B total with 21B active, and a Lite model with 16B total and 2.4B active. Only the Lite model used FIM.

After pre-training came supervised fine-tuning and RL with GRPO, using a reward model trained on compiler and test results.

It scored 90.2% on HumanEval and 75.7% on MATH. It was the first open model above 10% on SWE-Bench, a benchmark of real GitHub issues (12.7%).

Then the line ended. In September 2024, DeepSeek-V2.5 "combines DeepSeek-V2-Chat and DeepSeek-Coder-V2-Instruct" (V2.5 model card). From then on, code was a skill of the main model. No Coder-V3 exists.

## DeepSeekMath: the corpus, GRPO and a sobering finding

### Mining the web for maths

Good maths text is rare on the web, and it hides among recipes, forum spam and sports scores. Earlier maths models used curated sets such as arXiv papers. DeepSeek asked a different question: how much maths is in Common Crawl if you look hard enough?

> [!DEFINITION] fastText classifier
> A very small and fast text classifier from Facebook AI Research. It looks at the words and short word groups in a page and outputs a score, here "how likely is this page about maths". It can score billions of pages on ordinary CPUs.

The trick is a loop. Start with a small, clean set of maths pages (the seed). Train the classifier on it. Use the classifier to pull maths pages out of the crawl. Look at which websites gave many hits; those sites probably have more maths that the classifier missed. Add the missed pages to the seed, and train again.

{{FIG:math_loop|The DeepSeekMath data loop. Each round, a better classifier finds more maths, and the websites it finds point people to the pages it still misses. After four rounds the loop stopped finding new data.}}

> [!PAPER] Shao et al. (2024), DeepSeekMath · Figure 2 · page 5
> [![Figure 2 of the DeepSeekMath paper: the iterative pipeline from a maths seed through a fastText model, Common Crawl recall, domain discovery and URL annotation](/img/deepseek/papers/p8-math-pipeline.png)](/img/deepseek/papers/p8-math-pipeline.png)
>
> **What it says:** "An iterative pipeline that collects mathematical web pages from Common Crawl."
>
> **What it means:**
>
> The globe in the middle is the deduplicated crawl: 40 billion HTML pages. The seed at the left was 500,000 pages from OpenWebMath, a public maths dataset. The classifier only knows what the seed shows it, so the first round misses whole kinds of maths writing.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §2.1 · page 5
> [![Page 5 of the DeepSeekMath paper, highlighting the 10% domain rule, the final 35.5M pages and 120B tokens, and that 98% was already found in round three](/img/deepseek/papers/p8-math-iterations.png)](/img/deepseek/papers/p8-math-iterations.png)
>
> **What it says:** "Domains where over 10% of the web pages have been collected are classified as math-related," and "After four iterations of data collection, we end up with 35.5M mathematical web pages, totaling 120B tokens."
>
> **What it means:**
>
> The 10% rule is the detective step. If one tenth of a website's pages already look like maths, the site is a maths site (mathoverflow.net is the paper's example). The other nine tenths deserve a second look.

DeepSeekMath-Base 7B then trained on 500B tokens, 56% of them from this corpus, starting from DeepSeek-Coder-Base-v1.5 7B. The paper says "starting from a code training model is a better choice" (page 2). It also found that arXiv papers gave "no notable improvements" on its benchmarks.

The base model scored 64.2% on GSM8K and 36.2% on MATH. The pipeline later collected the data for Coder-V2.

### GRPO is born

DeepSeekMath then trained the model with reinforcement learning. Part 4 builds GRPO in full; here is only the part that matters for this story.

The standard method, PPO, needs a second network called a value model. Its job is to guess how good a partial answer will turn out. The training then knows whether an answer was better or worse than expected. The value model is about as big as the policy, so it doubles the memory.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §4.1 · page 13
> [![Figure 4 of the DeepSeekMath paper comparing PPO, with a trained value model, and GRPO, which computes advantages from a group of sampled outputs](/img/deepseek/papers/p8-math-grpo.png)](/img/deepseek/papers/p8-math-grpo.png)
>
> **What it says:** "GRPO foregoes the value model, instead estimating the baseline from group scores, significantly reducing training resources."
>
> **What it means:**
>
> In PPO (top), one question $$q$$ gives one output $$o$$, and the value model must say what score to expect. In GRPO (bottom), one question gives a group of outputs $$o_1 \ldots o_G$$, and the expected score is the group's average.
>
> It is grading on a curve. You do not need a separate teacher to predict how hard the exam was. You look at how the whole class did, and each answer is good or bad compared with its classmates.

The group comparison fits in one line. For a group of $$G$$ answers with rewards $$r_1, \ldots, r_G$$:

$$
\hat{A}_i = \frac{r_i - \operatorname{mean}(r_1, \ldots, r_G)}{\operatorname{std}(r_1, \ldots, r_G)}
$$

Here $$\hat{A}_i$$ is the **advantage** of answer $$i$$: how much better it was than its group. The mean is the group's average reward, and std is the spread of the rewards. Training pushes up the probability of answers with a positive advantage and pushes down the others.

A worked example: four answers to one question, scored 1 for right and 0 for wrong: $$r = (1, 0, 0, 1)$$. The mean is 0.5 and the standard deviation is 0.5. So the two right answers get $$\hat{A} = (1 - 0.5)/0.5 = +1$$, and the two wrong ones get $$-1$$.

The RL data was about 144K questions in the style of GSM8K and MATH. With GRPO, DeepSeekMath-Instruct 7B went from 46.8% to 51.7% on MATH, and from 82.9% to 88.2% on GSM8K. With 64 samples and a majority vote, the RL model reached 60.9% on MATH. GRPO then trained V2-Chat, Coder-V2, R1 and V4.

### "RL enhances Maj@K but not Pass@K"

The most quoted result of the paper is not a score. It is a question the authors asked about their own success: what did RL actually teach the model?

Say a model gives the right answer to a question with probability $$p$$. The chance that at least one of $$K$$ independent samples is right is:

$$
\text{Pass@}K = 1 - (1 - p)^K
$$

Read it from the inside out. $$1 - p$$ is the chance that one sample is wrong. $$(1-p)^K$$ is the chance that all $$K$$ are wrong. One minus that is the chance that at least one is right. With $$p = 0.3$$ and $$K = 4$$: $$1 - 0.7^4 = 1 - 0.2401 = 0.76$$. A model that is right only 30% of the time still finds the answer in four tries 76% of the time.

> [!PAPER] Shao et al. (2024), DeepSeekMath · Figure 7 · page 21
> [![Figure 7 of the DeepSeekMath paper: Maj@K and Pass@K curves for the Instruct and RL models on GSM8K and MATH, with Pass@K nearly equal for both models](/img/deepseek/papers/p8-math-majk.png)](/img/deepseek/papers/p8-math-majk.png)
>
> **What it says:** "It was noted that RL enhances Maj@K but not Pass@K."
>
> **What it means:**
>
> Look at the MATH panel on the right. The green and blue lines (Pass@K before and after RL) lie on top of each other at large K. The purple and orange lines (Maj@K) are separated: the RL model's vote is right more often.
>
> So after RL, the model does not find answers it could not find before. It finds the same answers, but it gives the right one more often, so the vote picks it. RL turned a model that sometimes knew into a model that more often says what it knows.

> [!PAPER] Shao et al. (2024), DeepSeekMath · §5.2.2 · page 21
> [![Page 21 of the DeepSeekMath paper, highlighting that RL makes the output distribution more robust by boosting the correct response from the top K rather than adding fundamental capabilities](/img/deepseek/papers/p8-math-why-rl.png)](/img/deepseek/papers/p8-math-why-rl.png)
>
> **What it says:** RL works by "rendering the output distribution more robust," and the improvement comes from "boosting the correct response from TopK rather than the enhancement of fundamental capabilities."
>
> **What it means:**
>
> The authors are honest about the limit of their own method. RL with this data is a sharpener, not a teacher. It cannot reward an answer the model never writes, so it cannot create a new skill from nothing.
>
> They also name a suspect: the RL questions came from instruction data the model had already seen. [Part 4](deepseek-4-r1-and-reasoning-by-rl.md) shows what changed with R1.

We can see the mechanism in a toy. Give each of 2,000 made-up questions a probability over five answers, one of them right. That is the "instruct" model. Then sharpen every distribution: square each probability and renormalise. That is the "RL" model. Sharpening moves probability toward whatever answer was already the favourite.

{{FIG:sharpen|Sharpening on two questions. When the right answer is already the favourite (A), squaring and renormalising lifts it from 0.50 to 0.77. When the right answer is second (B), the same operation lowers it from 0.25 to 0.21.}}

```python
def sharpen(P, a):                     # P: questions x answers, rows sum to 1
    S = P ** a
    return S / S.sum(1, keepdims=True)

def pass_at_k(P, k):                   # answer 0 is the right one
    return float(np.mean(1 - (1 - P[:, 0]) ** k))
```

Maj@K needs real votes, so the script samples $$K$$ answers many times and counts the winner (`code/deepseek/part8_majk_passk.py`, NumPy on a CPU, fixed seed). The real output:

```text
Toy model: 2000 questions, 5 answers each, RL = sharpen every distribution with alpha = 2.0
Mean probability of the correct answer: instruct 0.426, RL 0.502
Questions where the correct answer is already the most likely one: 54.6%

  K  Pass@K instr  Pass@K RL   Maj@K instr  Maj@K RL
  1         42.6%      50.2%         42.6%     50.2%
  4         72.2%      67.9%         48.1%     52.7%
  8         83.1%      74.8%         50.9%     53.6%
 16         90.8%      80.7%         52.6%     54.0%
 32         95.6%      85.7%         53.4%     54.3%
 64         98.1%      89.6%         53.8%     54.4%

Split by question type (probability of the correct answer, instruct -> RL):
  correct answer is the most likely     : 0.654 -> 0.828  (1093 questions)
  correct answer is NOT the most likely : 0.152 -> 0.110  (907 questions)
```

{{FIG:majk|The toy simulation. Sharpening lifts Pass@1 and Maj@K at small K, but Pass@K at large K ends lower than before (89.6% against 98.1% at K = 64), because questions where the right answer was a long shot lose probability.}}

The split at the bottom is the whole story. On questions where the model already favoured the right answer, sharpening raised it from 0.654 to 0.828. On the others, it lowered it from 0.152 to 0.110. Majority voting only cares about the first group, so Maj@K rises. Pass@K at large K lives on the long shots in the second group, so it falls.

> [!WARNING] What the toy does and does not show
> The toy is pure sharpening, so the vote's winner never changes and Maj@K before and after meet at large K (53.8% and 54.4%). In the paper, the RL model's Maj@K stays ahead even at K = 64. Its Pass@K at large K roughly matches the instruct model instead of dropping. Real RL is not only sharpening. The toy shows the direction of the effect, not its size.

## DeepSeekMath-V2: grading its own proofs

### The problem: a right answer is not a right proof

Answer-checking rewards have a blind spot. A model can reach the right number by a wrong argument, and the reward still says "correct". For a proof, that is fatal: a proof with a hole proves nothing.

DeepSeekMath-V2 (November 2025) attacks this directly. It is built on DeepSeek-V3.2-Exp-Base, so it is a 671B-class MoE model, and its weights are Apache 2.0. The plan has three roles, all played in the end by one model.

> [!DEFINITION] Verifier
> A model that reads a problem and a proposed proof, lists the issues it finds, and gives a score. In DeepSeekMath-V2 the score is 0 (wrong), 0.5 (has gaps) or 1 (complete and rigorous).

The verifier was trained with RL on 17,503 contest proof problems, with proofs scored by experts. It was rewarded for matching their score:

$$
R_{\text{score}}(s', s) = 1 - |s' - s|
$$

Here $$s$$ is the expert's score and $$s'$$ is the verifier's score. If the expert says 0.5 and the verifier says 1, the reward is $$1 - 0.5 = 0.5$$. If they agree, the reward is 1. A format check multiplies this, so the verifier must also write its findings in a fixed shape.

> [!PAPER] Shao et al. (2025), DeepSeekMath-V2 · §2.1.2 · page 3
> [![Page 3 of the DeepSeekMath-V2 paper, highlighting that a verifier can get full reward while hallucinating issues, and the introduction of meta-verification](/img/deepseek/papers/p8-mathv2-meta.png)](/img/deepseek/papers/p8-mathv2-meta.png)
>
> **What it says:** "the verifier can receive full reward by predicting the correct scores while hallucinating non-existent issues," so "we introduce meta-verification."
>
> **What it means:**
>
> A score-only reward checks the verdict, not the reasons. Think of a referee who gives the right red card but writes a report about a foul that never happened. The card was right this time; the referee is still not trustworthy.
>
> Meta-verification reads the verifier's report and asks: do these issues exist, and do they justify this score? This matters because the verifier becomes the reward for the proof writer.

{{FIG:mathv2|The DeepSeekMath-V2 loop. The generator writes a proof and its own critique; the verifier scores the proof; the meta-verifier checks the verifier's reasons. Hard new proofs, labelled by many independent checks, retrain the verifier so it stays ahead of the generator.}}

The proof writer (the generator) then trains against the verifier. It writes a proof and, after it, a self-analysis in the verifier's format. It gets reward for a good proof and for an honest self-analysis. A model that claims "no issues" on a broken proof is punished. The generator learns to find and fix its own mistakes before it stops.

The last part keeps the loop going. As the generator gets better, its proofs get harder to judge. So DeepSeek spends more compute on verification, with many independent checks of the same proof. These checks label hard new proofs, which retrain the verifier. The paper calls the difference the "generation-verification gap": checking is easier than writing, and they keep it that way.

> [!PAPER] Shao et al. (2025), DeepSeekMath-V2 · §3.3.3 · page 8
> [![Page 8 of the DeepSeekMath-V2 paper, highlighting the Putnam 2024 score of 118 out of 120 and that the hardest IMO-level problems remain challenging](/img/deepseek/papers/p8-mathv2-putnam.png)](/img/deepseek/papers/p8-mathv2-putnam.png)
>
> **What it says:** on Putnam 2024 the model solved 11 of 12 problems completely, "scoring 118/120 and surpassing the highest human score of 90," and "the hardest IMO-level problems remain challenging for our model."
>
> **What it means:**
>
> These scores come from heavy search: 64 candidate proofs with 64 verifications each. Up to 16 rounds of refinement run until a proof passes all 64 checks. Human experts graded the final proofs.
>
> The same search solved 5 of 6 problems on IMO 2025 (83.3% of points). The last highlighted sentence is the honest one: search can polish a proof, but it cannot always find the key idea.

On IMO-ProofBench, graded by experts, the "Heavy" setting scored 99.0 on the basic set and 61.9 on the advanced set. Gemini Deep Think (IMO Gold) scored 89.0 and 65.7 in the same figure, with numbers the paper takes from another group. DeepSeekMath-V2's data and reward method later trained DeepSeek-V3.2-Speciale, and V4 uses its self-verification to filter maths solutions.

## Proofs a computer can check: the Prover line

### What a formal proof is

A textbook proof is written for people and skips "obvious" steps. A **formal proof** is written for a program: every step must follow by an exact rule, and the program checks each one.

> [!DEFINITION] Lean 4 and Mathlib
> **Lean 4** is a programming language and **proof assistant**: you state a theorem in its language, write a proof, and Lean checks it. **Mathlib** is Lean's large shared library of definitions and already-proved facts.

> [!DEFINITION] Tactic and sorry
> A **tactic** is a proof command, such as "expand this algebra" or "this follows by linear arithmetic". **sorry** is a placeholder that says "trust me, I will prove this later". Lean accepts a file with `sorry` but marks the proof as unfinished.

Here is a tiny theorem in Lean 4 with Mathlib: for any real numbers $$a$$ and $$b$$, $$a^2 + b^2 \ge 0$$. We did not run it here (no Lean install), but `sq_nonneg` and `linarith` are standard Mathlib names.

```lean
theorem sum_sq_nonneg (a b : ℝ) : 0 ≤ a ^ 2 + b ^ 2 := by
  have h1 : 0 ≤ a ^ 2 := sq_nonneg a     -- a square is never negative
  have h2 : 0 ≤ b ^ 2 := sq_nonneg b
  linarith                                -- adding two non-negative facts finishes it
```

A sketch of the same proof, with holes, looks like this. Lean accepts its structure but not the proof:

```lean
theorem sum_sq_nonneg (a b : ℝ) : 0 ≤ a ^ 2 + b ^ 2 := by
  have h1 : 0 ≤ a ^ 2 := by sorry
  have h2 : 0 ≤ b ^ 2 := by sorry
  linarith
```

That second shape, a plan with holes, is exactly how Prover-V2 works. Hold on to it.

### Why a checker is a perfect reward

A proof checker gives a reward that cannot be argued with: the proof is accepted, or it is not. To see this in miniature, our script plays Lean's role for one inequality, $$(a+b)^2 \ge 4ab$$. A proof is a list of steps. Each step names a rule, and SymPy does the algebra (`code/deepseek/part8_proof_checker.py`).

```python
def check(step, prev):
    rule, expr = step
    if rule == 'sorry':
        return 'HOLE', 'left as sorry: accepted for now, proof not finished'
    if rule == 'rewrite':                       # the new expression must equal the previous one
        ok = sp.simplify(prev - expr) == 0
        ...
```

```text
--- a correct proof ---
goal: -4*a*b + (a + b)**2 >= 0
step 1: rewrite   a**2 - 2*a*b + b**2      OK    equal after expanding
step 2: rewrite   (a - b)**2               OK    equal after expanding
step 3: sq_nonneg (a - b)**2               OK    a square is never negative
reward: 1

--- a proof with one wrong step ---
goal: -4*a*b + (a + b)**2 >= 0
step 1: rewrite   a**2 - 4*a*b + b**2      FAIL  not equal: difference is 2*a*b
reward: 0

--- a sketch with a hole (sorry) ---
goal: -4*a*b + (a + b)**2 >= 0
step 1: sorry     a**2 - 2*a*b + b**2      HOLE  left as sorry: accepted for now, proof not finished
step 2: rewrite   (a - b)**2               OK    equal after expanding
step 3: sq_nonneg (a - b)**2               OK    a square is never negative
reward: 0
```

The wrong step is caught at once, with the exact difference ($$2ab$$). The sketch with a hole gets reward 0 even though the rest is fine. There is no partial credit to game.

### Prover-V1 and V1.5: synthetic data, then RL from the checker

The first problem in formal maths is data: very few competition problems exist in Lean. Prover-V1 (May 2024) translated natural-language problems into Lean, generated proofs, kept the ones Lean accepted, and repeated. The result was 8 million formal statements with proofs. Fine-tuning DeepSeekMath 7B on them gave 46.3% on miniF2F-test with 64 samples, against 23.0% for GPT-4.

> [!DEFINITION] miniF2F
> A benchmark of 244 test problems (and 244 validation problems) from maths competitions such as AMC, AIME and IMO, stated formally. A problem counts as solved only if the proof assistant accepts the proof.

Prover-V1.5 (August 2024) added reinforcement learning with the checker as the judge.

> [!PAPER] Xin et al. (2024), DeepSeek-Prover-V1.5 · §1.1 · page 4
> [![Page 4 of the Prover-V1.5 paper, highlighting RL from proof assistant feedback, Lean verification results as reward, and intrinsic rewards in RMaxTS](/img/deepseek/papers/p8-proverv15-rl.png)](/img/deepseek/papers/p8-proverv15-rl.png)
>
> **What it says:** they perform "reinforcement learning from proof assistant feedback (RLPAF)"; "Verification results from the Lean prover serve as reward supervision." For search, RMaxTS works "By assigning intrinsic rewards."
>
> **What it means:**
>
> RLPAF is GRPO with Lean as the reward function. Sample a group of proofs for one theorem, let Lean accept or reject each one, and push toward the accepted ones. It is our toy checker at scale.
>
> The search part solves a different problem. Most attempts fail, so the reward is almost always 0 and gives no direction. RMaxTS is a tree search that pays the model a small made-up bonus for reaching proof states it has not visited before. Curiosity keeps the search exploring until a real reward appears.

V1.5 reached 60.2% on miniF2F-test in single-pass whole-proof generation, and 63.5% with tree search. On ProofNet, a set of undergraduate theorems, it reached 25.3% on the test set.

### Prover-V2: a big model plans, a small model fills the holes

By 2025, general models such as DeepSeek-V3 could reason about maths informally very well, but they were poor at Lean. Specialised provers were good at Lean but could not plan a long proof. Prover-V2 (April 2025) combined the two.

> [!PAPER] Ren et al. (2025), DeepSeek-Prover-V2 · Figure 2 · page 3
> [![Figure 2 of the Prover-V2 paper: DeepSeek-V3 writes a proof sketch with sorry placeholders, the 7B prover solves each subgoal, and the pieces are combined into a complete Lean proof](/img/deepseek/papers/p8-proverv2-cold.png)](/img/deepseek/papers/p8-proverv2-cold.png)
>
> **What it says:** DeepSeek-V3 formalises a proof sketch into a Lean statement "with sorry placeholders for omitted proof details. A 7B prover model then recursively solves the decomposed subgoals."
>
> **What it means:**
>
> Read the left panel from the top. The theorem is $$n^2 \le n!$$ for $$n \ge 4$$. V3 thinks in words ("use induction"), then writes the Lean skeleton: a base case, an inductive step and a final step, each ending in `sorry`. That is our sketch-with-holes shape.
>
> On the right, each `sorry` becomes its own small theorem for the 7B prover. The pieces are put back together and Lean checks the whole. Attached to V3's reasoning, this becomes training data that links informal thinking to formal steps.

{{FIG:prover_v2|The Prover-V2 cold-start pipeline. V3 turns a hard theorem into subgoals; the 7B prover proves each one; Lean checks the stitched proof; the result plus V3's reasoning becomes training data for the RL stage.}}

After this cold start, the 671B model (built on DeepSeek-V3) was trained with RL, again with Lean as the judge. A 7B version was made from Prover-V1.5-Base with its context extended from 4K to 32K tokens. The headline: 88.9% on miniF2F-test with 8,192 samples per problem, and 37.1% on ProofNet-test with 1,024 samples.

> [!DEFINITION] PutnamBench
> A benchmark of problems from the Putnam competition, the hardest maths contest for undergraduates in the US and Canada, stated in Lean 4. The version used here has 658 problems.

The PutnamBench number has two public versions: 49 of 658 in the GitHub README, 47 in the paper (version 2). The paper explains why.

> [!PAPER] Ren et al. (2025), DeepSeek-Prover-V2 · §4.3 · page 10
> [![Page 10 of the Prover-V2 paper, highlighting that the initial run solved 49 PutnamBench problems, two were excluded for misformulated statements, and the final result is 47](/img/deepseek/papers/p8-proverv2-putnam.png)](/img/deepseek/papers/p8-proverv2-putnam.png)
>
> **What it says:** "Our initial run solved 49 problems"; after submission "we excluded two problems due to misformulated statements"; "The final result stands at 47 problems successfully solved."
>
> **What it means:**
>
> A formal statement can be wrong. A Putnam problem typed badly into Lean can become a different, easier problem, and a proof of it answers something nobody asked. The maintainers found two such statements, so 49 became 47. Cite 47 and the paper.

The same page has a second, more instructive correction.

> [!PAPER] Ren et al. (2025), DeepSeek-Prover-V2 · §4.3 · page 10
> [![Page 10 of the Prover-V2 paper, highlighting that 13 surprising 7B solutions were traced to a user interface bug in Lean 4.9.0 that the model learned to exploit](/img/deepseek/papers/p8-proverv2-hack.png)](/img/deepseek/papers/p8-proverv2-hack.png)
>
> **What it says:** the 7B model's 13 extra PutnamBench solutions were "traced to a user interface bug in Lean 4.9.0," and the model used two Cardinal lemmas "to exploit this user-interface bug."
>
> **What it means:**
>
> In Lean 4.9.0, one tactic could leave a hidden `sorry` that the interface failed to report. The 7B model found this during RL. It had learned to write "proofs" that the checker reported as clean but that contained a hole.
>
> This is **reward hacking**. Even a "perfect" checker is a program with bugs, and RL is very good at finding them. The 671B model's outputs did not show the pattern.

{{FIG:prover_bars|miniF2F-test pass rates across the Prover line. Each bar comes from its own paper, with different sampling budgets and search, so the chart shows a trend rather than a fair comparison.}}

On 15 AIME 2024 and 2025 problems in the new ProverBench set, Prover-V2 proved 6 formally and DeepSeek-V3 solved 8 informally. The gap between formal and informal maths "is substantially narrowing", the paper says.

## Seeing: DeepSeek-VL and VL2

### Images as tokens

To read an image, a vision encoder cuts it into small square patches and turns each patch into a vector. An adaptor maps those vectors into the language model's input, next to the text tokens.

> [!DEFINITION] Vision encoder, SigLIP and SAM
> A **vision encoder** turns an image into a sequence of vectors. **SigLIP** (from Google) and **CLIP** (from OpenAI) are trained to match images with their captions, so their vectors carry meaning ("a dog on grass"). **SAM** (from Meta) is trained to outline objects, so its vectors carry fine detail and edges.

DeepSeek-VL (March 2024, 1.3B and 7B) used both: SigLIP at 384 x 384 pixels for meaning and SAM-B at 1024 x 1024 for detail. Its design goal was to keep the language model strong, so it mixed plain text data into training from the start.

### VL2: tiles of any shape

A fixed 1024 x 1024 input is a bad fit for a long receipt or a wide screenshot. Squash a wide chart into a square and the text becomes unreadable. DeepSeek-VL2 (December 2024) cuts the image into tiles instead.

> [!PAPER] Wu et al. (2024), DeepSeek-VL2 · §2 · page 4
> [![Page 4 of the DeepSeek-VL2 paper, highlighting the choice of the resolution with least padding, the extra global thumbnail tile, and 729 visual embeddings per tile](/img/deepseek/papers/p8-vl2-tiling.png)](/img/deepseek/papers/p8-vl2-tiling.png)
>
> **What it says:** "We select the resolution" from the candidate grids "that minimizes the padding area"; the image is cut into 384 x 384 tiles "plus one global thumbnail tile"; each tile gives "729 visual embeddings of 1152 dimensions per tile."
>
> **What it means:**
>
> The candidates are every grid of 384-pixel tiles with at most 9 tiles. Pick the grid whose shape best matches the image, resize, and cut. SigLIP only ever sees the 384 x 384 squares it was trained on. The thumbnail is the whole image in one tile, so the big picture is not lost.

On the next page, each tile's 27 x 27 = 729 vectors become 14 x 14 = 196, by merging each 2 x 2 block. It also adds a "new line" token at the end of each row, so the model knows where rows end. The total number of vision tokens for a grid of $$c$$ columns and $$r$$ rows is:

$$
N = \underbrace{14 \times 15}_{\text{thumbnail}} + \underbrace{1}_{\text{separator}} + (r \times 14) \times (c \times 14 + 1)
$$

Here 14 x 15 is the thumbnail's 14 rows of 14 tokens plus one new-line token each. The 1 is a separator between the thumbnail and the tiles. The last term is the tile grid, with a new-line token after each of its $$r \times 14$$ rows. For one tile ($$c = r = 1$$): $$210 + 1 + 14 \times 15 = 421$$ tokens.

Our script applies this formula, plus the grid choice from the released code, to a few common images (`code/deepseek/part8_vl2_tiles.py`):

```text
23 candidate grids (cols x rows with cols * rows <= 9)
image                         grid  tiles  vision tokens
small square photo         1x1          2            421
phone photo 3024 x 4032    3x3         10          2,017
screenshot 1920 x 1080     4x2          9          1,807
A4 scan at 150 dpi         3x3         10          2,017
panorama 3000 x 1000       4x2          9          1,807
```

The panorama is the surprise. Its shape is 3:1, so a 3 x 1 grid would need no padding. The released code picks 4 x 2 instead, because its rule is "keep the most detail first, then waste the least". The paper's sentence describes only the second half of that rule. For most images the two agree.

{{FIG:vl2_tiles|DeepSeek-VL2 on a 1920 x 1080 screenshot. The released code chooses a 4 x 2 grid of 384-pixel tiles plus a thumbnail; each tile becomes 196 tokens after a 2 x 2 squeeze, for 1,807 vision tokens.}}

The language side of VL2 is a DeepSeekMoE model, with MLA in the two larger sizes. The three sizes are Tiny (3B total, 1.0B active including the vision part), Small (16B, 2.8B) and VL2 (27B, 4.5B). The largest scored 93.3 on DocVQA and 86.0 on ChartQA (questions about documents and charts), and 811 on OCRBench.

## Understanding and drawing: Janus, JanusFlow and Janus-Pro

A model that can both describe an image and draw one sounds like it should share one way of seeing. Janus (October 2024) argues the opposite.

Understanding wants meaning: "two cats on a red sofa". Drawing wants exact layout: which pixel goes where, at what colour. An encoder tuned for one is poor at the other. The Janus paper argues that earlier unified models, such as Chameleon, used one image encoder for both jobs and lost understanding quality.

> [!DEFINITION] VQ tokenizer
> A "vector-quantised" image tokenizer. It cuts an image into patches and replaces each patch with the ID of the closest entry in a fixed codebook. An image becomes a short list of discrete codes, like words, that a language model can predict one at a time. A decoder turns codes back into pixels.

{{FIG:janus|Janus routes images through two separate encoders: SigLIP when the task is to understand an image, and a VQ tokenizer when the task is to draw one. Both feed one transformer, which has a text head and an image head.}}

> [!PAPER] Chen et al. (2025), Janus-Pro · Figure 3 · page 3
> [![Figure 3 of the Janus-Pro paper: an understanding encoder and a generation encoder feeding one auto-regressive transformer, with a text de-tokenizer and an image decoder](/img/deepseek/papers/p8-januspro-arch.png)](/img/deepseek/papers/p8-januspro-arch.png)
>
> **What it says:** "We decouple visual encoding for multimodal understanding and visual generation."
>
> **What it means:**
>
> On the left, an image comes in through the understanding encoder and text comes out. On the right, text comes in and image codes come out. The transformer in the middle is shared: one set of weights learns both directions, and only the doors in and out are separate.

So each encoder is chosen for what the transformer must do with it. SigLIP gives rich, continuous vectors: good for answering questions, impossible to predict one by one. The VQ tokenizer gives codes from a fixed list: poor at nuance, but exactly what a next-token predictor can output. Read with the first; write with the second.

The family grew in two directions.

- **JanusFlow** (November 2024, 1.3B) replaced the code-by-code image output with **rectified flow**, inside the same language model. Rectified flow is a diffusion-style method that turns noise into an image in a few smooth steps. JanusFlow scored 0.63 on GenEval (defined below).
- **Janus-Pro** (January 2025, 1B and 7B) kept the Janus design and scaled it. It has a better training schedule, a 7B language model and about 72 million synthetic "aesthetic" images. Real and synthetic data are mixed 1:1 in pre-training.

> [!DEFINITION] GenEval
> A benchmark for text-to-image models. It checks whether the image has the right objects, counts, colours and positions from the prompt, using an object detector. Scores run from 0 to 1.

Janus-Pro-7B scored 0.80 on GenEval, ahead of DALL-E 3 (0.67) and Stable Diffusion 3 Medium (0.74), as reported in its paper. On MMBench, a multiple-choice test of image understanding, it rose from 69.4 (Janus) to 79.2. The limit is resolution: images are 384 x 384, so small text and fine detail in generated images suffer.

## DeepSeek-OCR: a page as a picture

### The strange question

Every token in a long context costs memory in the KV cache and compute in attention ([the KV cache article](llm-inference-2-kv-cache.md) explains why). A page of English text is about a thousand tokens. A vision encoder can describe the same page, as an image, in a few hundred tokens or fewer.

DeepSeek-OCR (October 2025) asks: if a model can read the text back from those vision tokens, are vision tokens a compressed form of text? The paper calls it "contexts optical compression".

The ratio has a simple formula:

$$
\text{ratio} = \frac{N_{\text{text}}}{N_{\text{vision}}}
$$

where $$N_{\text{text}}$$ is how many tokens the page's text takes with the decoder's tokenizer, and $$N_{\text{vision}}$$ is how many vision tokens the encoder produces. For a 700-token page read from 64 vision tokens, the ratio is $$700 / 64 \approx 10.9$$x.

### DeepEncoder: cheap first, careful second

The encoder must take a high-resolution page, keep memory low and output few tokens. Those goals fight each other. Global attention over thousands of patches is expensive; few tokens lose detail.

> [!PAPER] Wei et al. (2025), DeepSeek-OCR · Figure 3 · page 5
> [![Figure 3 of the DeepSeek-OCR paper: SAM with local attention, a 16x convolutional down-sampler, CLIP with global attention, and the DeepSeek-3B MoE decoder](/img/deepseek/papers/p8-ocr-arch.png)](/img/deepseek/papers/p8-ocr-arch.png)
>
> **What it says:** DeepEncoder has "a SAM [17] for perception dominated by window attention, a CLIP [29] for knowledge with dense global attention, and a 16× token compressor that bridges between them."
>
> **What it means:**
>
> The order is the idea. SAM sees the full page in thousands of patches, but only through small windows, so its cost stays low. The convolution then squeezes 16 patches into one token. Only after that does CLIP do the expensive global attention, on 16 times fewer tokens.
>
> It is skimming, then reading. The decoder on the right is a 3B MoE model with 570M parameters active per token.

The numbers check out by hand. Each patch is 16 x 16 pixels, so a 1024-pixel side holds 64 patches. The page holds 64 x 64 = 4,096 (paper page 5). The two convolution layers each halve the width and the height, so 4,096 becomes 256 tokens.

{{FIG:deepencoder|DeepSeek-OCR at Base mode. A 1024 x 1024 page is split into 4,096 patches, read cheaply by SAM, squeezed 16 times to 256 tokens, read globally by CLIP and passed to the MoE decoder. The same page holds 761 to 1,061 text tokens in our count.}}

### Experiment: an optical compression calculator

The rule is: patches $$= (\text{side}/16)^2$$, and vision tokens $$=$$ patches $$/16$$. The script applies it and checks it against the paper's Table 1 (`code/deepseek/part8_ocr_calc.py`, CPU only). Then it counts real text tokens on three pages of the OCR paper itself. We count with the DeepSeek-V2-Lite tokenizer (about 100K entries). DeepSeek-OCR's own tokenizer (about 129K entries) is not in our cache, so our counts are close but not identical.

```python
PATCH, SQUEEZE = 16, 16            # SAM patch size; the 2-layer conv compressor cuts the token count 16x

def vision_tokens(side):
    patches = (side // PATCH) ** 2
    return patches, patches // SQUEEZE

def valid_tokens(n_actual, w, h):
    """Paper Eq. 1 (p6): padding a non-square page wastes tokens; only this many carry the page."""
    return math.ceil(n_actual * (1 - (max(w, h) - min(w, h)) / max(w, h)))
```

```text
== 1. Vision tokens per mode: 16x16 patches, then a 16x squeeze ==
mode    side  patches  tokens  paper
Tiny     512     1024      64     64
Small    640     1600     100    100
Base    1024     4096     256    256
Large   1280     6400     400    400

== 2. Valid tokens for an A4-shaped page (210 x 297 mm) padded to a square (paper Eq. 1) ==
Base   256 tokens, of which 182 cover the page
Large  400 tokens, of which 283 cover the page

== 3. Text tokens on real pages (DeepSeek-V2-Lite tokenizer, a stand-in for the OCR tokenizer) ==
OCR paper, page 5 (text and a figure): 2,848 characters, 422 words, 761 text tokens
OCR paper, page 10 (text and Table 2): 2,947 characters, 472 words, 995 text tokens
OCR paper, page 12 (text and Table 4): 3,229 characters, 482 words, 1,061 text tokens

== 4. Compression ratio = text tokens / vision tokens, for each page and mode ==
page                                      Tiny 64  Small 100  Base 256  Large 400
OCR paper, page 5 (text and a figure)       11.9x       7.6x      3.0x       1.9x
OCR paper, page 10 (text and Table 2)       15.5x       9.9x      3.9x       2.5x
OCR paper, page 12 (text and Table 4)       16.6x      10.6x      4.1x       2.7x

== 5. What the paper measured near those ratios (Table 2, Fox English pages, 600-1300 text tokens) ==
OCR paper, page 5 (text and a figure): bucket 700-800 -> 93.8% with 64 tokens, 97.3% with 100 tokens (28 test pages)
OCR paper, page 10 (text and Table 2): bucket 900-1000 -> 85.9% with 64 tokens, 96.8% with 100 tokens (14 test pages)
OCR paper, page 12 (text and Table 4): bucket 1000-1100 -> 79.3% with 64 tokens, 91.5% with 100 tokens (11 test pages)

== 6. A newspaper page: the paper says 4,000 to 5,000 text tokens (p12) ==
4000 text tokens: Tiny 62.5x, Small 40.0x, Base 15.6x, Large 10.0x
5000 text tokens: Tiny 78.1x, Small 50.0x, Base 19.5x, Large 12.5x
Under 10x needs at least 400 to 500 vision tokens, which is why the paper uses Gundam modes for newspapers.
```

Read it top to bottom. Section 1: the simple rule reproduces all four native modes of the paper exactly. Section 2: a tall page padded into a square wastes about a third of its tokens. Our 182 valid tokens at Base match the paper's own "256(182)" in its Table 3.

Sections 3 to 5: a page of a research paper holds 761 to 1,061 text tokens. At Small mode (100 vision tokens), that is 7.6x to 10.6x. The paper measured 91.5% to 97.3% precision in those ranges.

Section 6 shows where the idea runs out. A dense newspaper page cannot fit under 10x even at Large mode. The paper uses its "Gundam" modes for them: several 640 or 1024 pixel tiles plus a global view, which costs more tokens.

> [!PAPER] Wei et al. (2025), DeepSeek-OCR · Table 2 · page 10
> [![Table 2 of the DeepSeek-OCR paper: precision and compression with 64 and 100 vision tokens for Fox pages of 600 to 1300 text tokens, highlighting 96.5% at 10.5x and 59.1% at 19.7x](/img/deepseek/papers/p8-ocr-table2.png)](/img/deepseek/papers/p8-ocr-table2.png)
>
> **What it says:** text tokens are counted "using DeepSeek-OCR's tokenizer"; with 64 vision tokens, precision is 96.5% at 10.5x compression and 59.1% at 19.7x.
>
> **What it means:**
>
> Each row is a group of pages by length. Read across one row: the same pages, shown with 64 or 100 vision tokens. More tokens mean less compression and higher precision.
>
> The headline, "about 97% under 10x, about 60% at 20x", comes from this table. With 100 tokens, every row under 10x is at 96.8% or more; the last 64-token row is 59.1% at 19.7x. That row has only 4 pages, so it is a noisy number. Figure 1 of the paper prints 85.8% and 76.3% where this table says 85.9% and 76.4%; we use the table.

{{FIG:ocr_curve|DeepSeek-OCR precision against compression, from Table 2 of the paper. Every point below 10x is at 96.8% or more. Past 15x, precision with 64 tokens falls quickly, to 59.1% at 19.7x.}}

> [!PAPER] Wei et al. (2025), DeepSeek-OCR · §4.2 · page 12
> [![Page 12 of the DeepSeek-OCR paper, highlighting that slides need only 64 vision tokens and that newspapers hold 4 to 5 thousand text tokens](/img/deepseek/papers/p8-ocr-news.png)](/img/deepseek/papers/p8-ocr-news.png)
>
> **What it says:** some documents need very few tokens, "such as slides which only need 64 vision tokens," while "the text tokens in newspapers are 4-5,000, far exceeding the 10× compression of other modes."
>
> **What it means:**
>
> The ratio depends on the page, not only on the model. A slide has little text in big letters, so 64 tokens hold it. A newspaper is small print from edge to edge.
>
> This is the practical rule from the calculator. Count the text, divide by your token budget, and stay under about 10x if you need near-perfect text back.

The paper ends with a bolder thought. At nearly 20x, the paper notes that "precision can still approach 60%" (page 12), and the authors connect that to forgetting. Recent context could stay as text and old context as ever smaller, blurrier images: exact when fresh, faded but cheap when old. This is a research idea, not a feature of any released chat model.

In practice, DeepSeek-OCR is also a strong OCR tool. On OmniDocBench it beat GOT-OCR2.0 (256 tokens per page) with only 100 vision tokens. It beat MinerU2.0 (over 6,000 tokens per page) with fewer than 800. One A100-40G GPU can process more than 200,000 pages per day. That is how DeepSeek turns PDFs into training text. The weights and code are MIT.

> [!TIP] In production
> Pick the mode from the text density of your pages, not from the image size. Slides and short forms work in Tiny or Small mode. Books and reports work in Small or Base. Newspapers and dense tables need Gundam mode. The precision numbers above come from clean English pages; scanned, rotated or handwritten pages will do worse.

### DeepSeek-OCR 2: learning the reading order

Most vision encoders read patches in a fixed order, left to right and top to bottom. People do not read a two-column paper that way. DeepSeek-OCR 2 (January 2026) replaces the CLIP part with a small language-model-style encoder (about 500M parameters, Qwen2 style). A set of learned "causal flow" queries reads the visual tokens and learns an order, and only these queries go to the decoder.

The budget is 256 to 1,120 vision tokens. On OmniDocBench v1.5, the overall score rose from 87.36 to 91.09, against DeepSeek-OCR with a similar budget. The reading-order edit distance fell from 0.073 to 0.048. The weights are Apache 2.0 per the model card; we found no license file in the code repository.

## ESFT: fine-tune only the experts a task uses

In an MoE model ([Part 2](deepseek-2-the-v3-architecture.md)), a router sends each token to a few experts. ESFT (July 2024) asks: why fine-tune experts a task never uses?

The method is short. Run a small sample of the task's data through the model (32 concatenated samples of 4,096 tokens). For each expert in each layer, measure how much the task uses it. ESFT-Gate uses the average gate score; ESFT-Token uses the share of tokens routed to it. Fine-tune only the top experts, and freeze everything else, including attention and the shared experts.

The backbone had the DeepSeek-V2-Lite architecture. Results are the average over eight tasks (paper Table 1) and over general benchmarks (Table 2):

| Method | Task average | General ability average |
|---|---|---|
| No fine-tuning | 33.6 | 62.4 |
| Full fine-tuning | 51.0 | 58.8 |
| LoRA | 44.9 | 59.1 |
| ESFT-Token | 49.4 | 61.5 |
| ESFT-Gate | 50.2 | 60.6 |

ESFT gets close to full fine-tuning on the task, and loses less general ability. The paper reports up to 90% less storage and up to 30% less training time than full fine-tuning. The training code is public, which is rare for DeepSeek; the 13 released checkpoints carry no license tag on Hugging Face.

## GRM and SPCT: a reward model that explains itself

For "write a polite reply" or "summarise this contract", no rule can check the answer. You need a reward model: a model that judges answers. Most reward models output only a number, so they cannot explain themselves or get better by thinking longer.

> [!DEFINITION] Generative reward model (GRM)
> A reward model that writes text: first the principles it will judge by, then a critique of each answer, then a score. Because it writes, you can sample it several times and combine the verdicts.

SPCT (Self-Principled Critique Tuning, April 2025) trains such a model in two phases. First, a cold start: fine-tune on judgements whose scores matched the known preference, and reject the rest. Then online RL with a simple rule-based reward: did the model's scores pick the better answer?

The twist is at inference. Sample the GRM 32 times and let the verdicts vote. A second "meta" reward model can filter out low-quality judgements before the vote. On the paper's combined benchmark (RewardBench, PPE and RMB), DeepSeek-GRM-27B went from 69.9 (one sample) to 71.0 (32 votes) and 72.8 (meta-filtered votes). GPT-4o scored 71.3 and Nemotron-4-340B-Reward 70.5 in the same table.

> [!WARNING] Where the GRM weights live
> The GRM-27B is post-trained from Google's Gemma-2-27B, not from a DeepSeek model. The paper says the models are released on Hugging Face and ModelScope, but no GRM repository exists in the deepseek-ai organisation (checked 10 November 2026). Repositories under the user BBQGOD carry the DeepSeek logo and a Gemma license. That they are the official release is plausible but not confirmed.

DeepSeek-V4 uses a generative reward model for hard-to-check tasks, and the policy model itself acts as the judge (V4 report, pages 29 to 30). The V4 report does not cite SPCT by name in that passage, so the link from GRM to V4 is our reading.

## The 2026 side work, briefly

Two more lines appeared in 2026. Both connect to V4, which [Part 6](deepseek-6-v4-and-v4-1.md) covers in depth.

**DSpark and DeepSpec (June to July 2026).** DSpark is a speculative decoding method: a small draft model guesses several tokens, and the big model checks them in one pass ([speculative decoding explained](llm-inference-4-speculative-decoding.md)). It picks how many tokens to verify per request. In DeepSeek's live V4 serving, per-user generation was 60% to 85% faster at the same throughput. The baseline was MTP-1, the multi-token prediction head of [Part 2](deepseek-2-the-v3-architecture.md). The DeepSpec code (MIT) includes training code for the drafters.

**DeepSeek-V4-Flash-Vision-Exp (August 2026).** The model card calls it "our first experimental multimodal model in the DeepSeek-V4 family". It adds vision modules to V4-Flash and continues training. There is no paper. The card says only "vision encoder and aligner", so whether it reuses DeepEncoder, SigLIP or Janus parts is not confirmed. The weights are MIT.

## What flowed back into the main models

The side models were not dead ends. The map below shows the links that DeepSeek states in a paper or model card, and the two that are our inference.

{{FIG:flow_map|What the side models gave the main line. Solid arrows are stated in a DeepSeek paper or model card; dashed arrows are our inference or a reuse that is not confirmed.}}

| Specialised work | Went into | Evidence |
|---|---|---|
| Coder-v1.5 7B | the starting weights of DeepSeekMath 7B | DeepSeekMath paper, p2 |
| DeepSeekMath data loop | the Coder-V2 code and maths corpus | Coder-V2 paper, p2 |
| GRPO | V2-Chat RL, Coder-V2, R1, V4 | V2 p4; Coder-V2 p7; R1 p2; V4 p5 |
| Coder-V2-Instruct | DeepSeek-V2.5 (merge) | V2.5 model card |
| DeepSeekMath-V2 data and reward | DeepSeek-V3.2-Speciale | V3.2 report, p6 |
| DeepSeekMath-V2 self-verification | V4 formal-maths pipeline | V4 report, p37 |
| mHC ([Part 6](deepseek-6-v4-and-v4-1.md)) | V4 residual connections | V4 report, p1 |
| Engram (Part 6) | V4.1 | V4.1 report ([Part 6](deepseek-6-v4-and-v4-1.md)); not in V4 |
| DSpark | V4 serving and DSpark checkpoints | DSpark paper, p1 |
| GRM line | V4's generative reward model | V4 report, pp29 to 30 (link is our reading) |
| Vision line | V4-Flash-Vision-Exp | model card (reuse not confirmed) |

The flow also ran the other way: V3 is the base of Prover-V2-671B, and V3.2-Exp-Base is the base of DeepSeekMath-V2.

## What is open, family by family

| Family | Weights | Inference code | Training code | Training data |
|---|---|---|---|---|
| Coder, Coder-V2 | yes, all sizes | usage examples | a fine-tuning script only (Coder) | described only |
| DeepSeekMath | base, instruct, RL | usage and evaluation | no (GRPO described, not released) | described only |
| DeepSeekMath-V2 | yes (Apache 2.0) | yes | no | described only |
| Prover V1, V1.5, V2 | yes | search and evaluation code (V1.5) | no | V1 dataset and ProverBench released |
| VL, VL2 | yes | yes | no | described only |
| Janus family | yes | yes | no | described only |
| DeepSeek-OCR, OCR 2 | yes (MIT; Apache 2.0) | yes, with vLLM | no | described only |
| ESFT | 13 checkpoints | yes | yes | small task datasets |
| GRM | only under an unofficial account | no | no | no |

The pattern from [Part 1](deepseek-1-every-model-and-what-is-open.md) holds. DeepSeek opens weights and inference code, but never pre-training code or data. Its released training code is narrow: the Coder fine-tuning script, ESFT and the DeepSpec drafters.

## The whole series on one page

All nine parts, for reference.

| Part | What it covers |
|---|---|
| [1. Every model and what is open](deepseek-1-every-model-and-what-is-open.md) | The lab, every model from 2023 to November 2026 with sizes, context and licenses, what is open, and the API |
| [2. The V3 architecture](deepseek-2-the-v3-architecture.md) | DeepSeekMoE, MLA, auxiliary-loss-free balancing, node-limited routing, multi-token prediction, the 671B / 37B breakdown |
| [3. Training V3](deepseek-3-training-v3.md) | Pre-training data, FP8, DualPipe, the cluster, and what the famous cost number leaves out |
| [4. R1 and reasoning by RL](deepseek-4-r1-and-reasoning-by-rl.md) | GRPO in full, R1-Zero, the R1 pipeline, distilled models, hybrid thinking and the RL of V3.2 and V4 |
| [5. Sparse attention and V3.2](deepseek-5-sparse-attention-and-v3-2.md) | NSA, DeepSeek Sparse Attention, the lightning indexer, V3.2-Exp and V3.2 |
| [6. V4 and V4.1](deepseek-6-v4-and-v4-1.md) | Hybrid compressed attention, mHC, FP4, 1M context, DSpark, the V4.1 encoder-decoder and Engram |
| [7. Running DeepSeek](deepseek-7-running-deepseek.md) | Memory arithmetic, hardware, quantised files, llama.cpp, KTransformers, vLLM, SGLang and DeepSeek's own serving |
| 8. The specialised models (this part) | Coder, DeepSeekMath and GRPO, Math-V2, the Lean provers, VL2, Janus, DeepSeek-OCR, ESFT and GRM |
| [9. DeepInfer](deepseek-9-deepinfer.md) | DeepSeek's own production inference engine, read from its source, and how it differs from vLLM and SGLang |

> [!TAKEAWAYS]
> - DeepSeek-Coder trained on whole repositories sorted by their imports, with fill-in-the-middle on half of the samples.
> - Coder-V2 continued a V2 checkpoint with 6T more tokens, reached 338 languages and 128K context, and merged into V2.5.
> - DeepSeekMath mined 120B maths tokens from Common Crawl with a four-round fastText loop, and introduced GRPO.
> - Its RL "enhances Maj@K but not Pass@K": it makes the model more consistent, not more capable. Our sharpening toy shows the same direction.
> - DeepSeekMath-V2 rewards a proof writer through a verifier that is itself checked; with heavy search it scored 118/120 on Putnam 2024.
> - Prover-V2 lets V3 split a theorem into `sorry` subgoals for a 7B prover, and reaches 88.9% on miniF2F-test. On PutnamBench, cite 47 (paper), not 49 (README).
> - VL2 cuts images into 384-pixel tiles plus a thumbnail; Janus reads images with one encoder and draws them with another.
> - DeepSeek-OCR reads a page back from 64 to 400 vision tokens: about 97% precision under 10x compression, about 60% near 20x.
> - GRPO, the maths data loop, Coder-V2 and Math-V2's self-verification all flowed into the main models.

**Next:** [Part 9](deepseek-9-deepinfer.md) opens DeepInfer, the engine DeepSeek uses to serve these models, and compares it with vLLM and SGLang. If you started here, [Part 1](deepseek-1-every-model-and-what-is-open.md) maps every model, and [Part 2](deepseek-2-the-v3-architecture.md) explains the architecture most of these models borrowed. The [LLM Inference series](llm-inference-1-prefill-and-decode.md) shows how models like these are served.

<details>
<summary>Research notes: sources, versions and things we could not confirm</summary>

- Paper versions used: Coder 2401.14196v2, Coder-V2 2406.11931v1, DeepSeekMath 2402.03300v3, DeepSeekMath-V2 2511.22570v1, Prover 2405.14333v1, Prover-V1.5 2408.08152v1, Prover-V2 2504.21801v2, VL 2403.05525v2, VL2 2412.10302v1, Janus 2410.13848v1, JanusFlow 2411.07975v2, Janus-Pro 2501.17811v1, OCR 2510.18234v1, OCR 2 2601.20552v1, ESFT 2407.01906v2, GRM 2504.02495v3, DSpark 2607.05147v1. Page numbers in the boxes are 1-based PDF pages of these versions.
- Release dates are Hugging Face repository creation dates, which are not always announcement dates. The DeepSeek news page dates V4-Flash-Vision-Exp to 21 August 2026; its repository creation date is 31 August 2026.
- Prover-V2 licenses are mixed: the Hugging Face LICENSE file is MIT, but the card says the model is subject to the DeepSeek Model License. Janus repositories carry an MIT tag while their cards point to the DeepSeek Model License. We treat both as DeepSeek License.
- The DeepSeek-OCR-2 code license and the ESFT license could not be confirmed: no LICENSE file was found in those repositories on 10 November 2026.
- The VL2 grid choice in our script follows the released code (`select_best_resolution` in `deepseek_vl2/models/processing_deepseek_vl_v2.py`), which keeps the most effective resolution first and the least waste second. The paper text describes least padding only. We assumed every grid with at most 9 tiles is a candidate, as the paper states; the shipped processor config may list a different set.
- The OCR calculator uses the DeepSeek-V2-Lite tokenizer as a stand-in. DeepSeek-OCR's tokenizer has about 129K entries, so real text-token counts differ by a few percent.
- The Maj@K toy uses 2,000 questions, 5 answers, logits drawn from a normal distribution plus a uniform bonus on the right answer, sharpening with alpha = 2, and Monte Carlo voting with random tie-breaks. It is a simulation, not a model run.
- The Lean snippets were not compiled here. `sq_nonneg` and `linarith` are standard Mathlib names, but the exact form may need small changes across Mathlib versions.

</details>

## References

- Guo et al., [DeepSeek-Coder](https://arxiv.org/abs/2401.14196) (2024) and [code](https://github.com/deepseek-ai/DeepSeek-Coder); Zhu et al., [DeepSeek-Coder-V2](https://arxiv.org/abs/2406.11931) (2024) and [code](https://github.com/deepseek-ai/DeepSeek-Coder-V2); [DeepSeek-V2.5 model card](https://huggingface.co/deepseek-ai/DeepSeek-V2.5).
- Shao et al., [DeepSeekMath](https://arxiv.org/abs/2402.03300) (2024) and [code](https://github.com/deepseek-ai/DeepSeek-Math); [DeepSeekMath-V2](https://arxiv.org/abs/2511.22570) (2025) and [weights](https://huggingface.co/deepseek-ai/DeepSeek-Math-V2).
- Xin et al., [DeepSeek-Prover](https://arxiv.org/abs/2405.14333) (2024); [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) (2024) and [code](https://github.com/deepseek-ai/DeepSeek-Prover-V1.5); Ren et al., [DeepSeek-Prover-V2](https://arxiv.org/abs/2504.21801) (2025) and [code](https://github.com/deepseek-ai/DeepSeek-Prover-V2).
- Lu et al., [DeepSeek-VL](https://arxiv.org/abs/2403.05525) (2024); Wu et al., [DeepSeek-VL2](https://arxiv.org/abs/2412.10302) (2024) and [code](https://github.com/deepseek-ai/DeepSeek-VL2).
- Wu et al., [Janus](https://arxiv.org/abs/2410.13848) (2024); Ma et al., [JanusFlow](https://arxiv.org/abs/2411.07975) (2024); Chen et al., [Janus-Pro](https://arxiv.org/abs/2501.17811) (2025); [code for all three](https://github.com/deepseek-ai/Janus).
- Wei, Sun and Li, [DeepSeek-OCR](https://arxiv.org/abs/2510.18234) (2025) and [code](https://github.com/deepseek-ai/DeepSeek-OCR); [DeepSeek-OCR 2](https://arxiv.org/abs/2601.20552) (2026) and [weights](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2).
- Wang et al., [ESFT](https://arxiv.org/abs/2407.01906) (2024) and [code](https://github.com/deepseek-ai/ESFT); Liu et al., [Inference-Time Scaling for Generalist Reward Modeling](https://arxiv.org/abs/2504.02495) (2025) and the [unofficial GRM-27B repository](https://huggingface.co/BBQGOD/DeepSeek-GRM-27B).
- Cheng et al., [DSpark](https://arxiv.org/abs/2607.05147) (2026) and [DeepSpec](https://github.com/deepseek-ai/DeepSpec); [DeepSeek-V4-Flash-Vision-Exp model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp).
