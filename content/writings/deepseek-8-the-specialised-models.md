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

<figure class="fig"><svg viewBox="0 0 780 408" role="img" aria-label="A timeline of DeepSeek specialised releases from 2023 to 2026 in five lanes: code, maths, proofs, vision and other."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Specialised releases by family (dates: Hugging Face repo creation or arXiv)</text><line class="grid" x1="149.5" y1="64.0" x2="149.5" y2="394.0"/><text class="t-tick" x="149.5" y="58.0" text-anchor="middle">2024</text><line class="grid" x1="363.7" y1="64.0" x2="363.7" y2="394.0"/><text class="t-tick" x="363.7" y="58.0" text-anchor="middle">2025</text><line class="grid" x1="577.9" y1="64.0" x2="577.9" y2="394.0"/><text class="t-tick" x="577.9" y="58.0" text-anchor="middle">2026</text><text class="t-note" x="20.0" y="100.0" text-anchor="start">Code</text><line class="axis" x1="96.0" y1="96.0" x2="760.0" y2="96.0"/><circle class="s1 ring" cx="113.1" cy="96.0" r="5"/><text class="t-tick" x="113.1" y="85.0" text-anchor="middle">Coder 1.3B-33B</text><circle class="s1 ring" cx="164.5" cy="96.0" r="5"/><text class="t-tick" x="164.5" y="117.0" text-anchor="middle">Coder-v1.5</text><circle class="s1 ring" cx="245.9" cy="96.0" r="5"/><text class="t-tick" x="245.9" y="85.0" text-anchor="middle">Coder-V2</text><circle class="s1 ring" cx="295.2" cy="96.0" r="5"/><text class="t-tick" x="295.2" y="117.0" text-anchor="middle">merged into V2.5</text><text class="t-note" x="20.0" y="166.0" text-anchor="start">Maths</text><line class="axis" x1="96.0" y1="162.0" x2="760.0" y2="162.0"/><circle class="s2 ring" cx="171.0" cy="162.0" r="5"/><text class="t-tick" x="171.0" y="151.0" text-anchor="middle">DeepSeekMath 7B + GRPO</text><circle class="s2 ring" cx="558.7" cy="162.0" r="5"/><text class="t-tick" x="558.7" y="151.0" text-anchor="middle">Math-V2</text><text class="t-note" x="20.0" y="232.0" text-anchor="start">Proofs</text><line class="axis" x1="96.0" y1="228.0" x2="760.0" y2="228.0"/><circle class="s3 ring" cx="233.1" cy="228.0" r="5"/><text class="t-tick" x="233.1" y="217.0" text-anchor="middle">Prover-V1</text><circle class="s3 ring" cx="282.3" cy="228.0" r="5"/><text class="t-tick" x="282.3" y="249.0" text-anchor="middle">V1.5</text><circle class="s3 ring" cx="434.4" cy="228.0" r="5"/><text class="t-tick" x="434.4" y="217.0" text-anchor="middle">Prover-V2</text><text class="t-note" x="20.0" y="298.0" text-anchor="start">Vision</text><line class="axis" x1="96.0" y1="294.0" x2="760.0" y2="294.0"/><circle class="s1 ring" cx="188.1" cy="294.0" r="5"/><text class="t-tick" x="188.1" y="283.0" text-anchor="middle">VL</text><circle class="s1 ring" cx="320.9" cy="294.0" r="5"/><text class="t-tick" x="320.9" y="283.0" text-anchor="middle">Janus</text><circle class="s1 ring" cx="335.9" cy="294.0" r="5"/><text class="t-tick" x="335.9" y="315.0" text-anchor="middle">JanusFlow</text><circle class="s1 ring" cx="353.0" cy="294.0" r="5"/><text class="t-tick" x="353.0" y="283.0" text-anchor="middle">VL2</text><circle class="s1 ring" cx="380.9" cy="294.0" r="5"/><text class="t-tick" x="380.9" y="330.0" text-anchor="middle">Janus-Pro</text><circle class="s1 ring" cx="535.1" cy="294.0" r="5"/><text class="t-tick" x="535.1" y="283.0" text-anchor="middle">OCR</text><circle class="s1 ring" cx="595.1" cy="294.0" r="5"/><text class="t-tick" x="595.1" y="315.0" text-anchor="middle">OCR 2</text><circle class="s1 ring" cx="719.3" cy="294.0" r="5"/><text class="t-tick" x="719.3" y="283.0" text-anchor="middle">V4-Flash-Vision-Exp</text><text class="t-note" x="20.0" y="364.0" text-anchor="start">Other</text><line class="axis" x1="96.0" y1="360.0" x2="760.0" y2="360.0"/><circle class="s2 ring" cx="258.8" cy="360.0" r="5"/><text class="t-tick" x="258.8" y="349.0" text-anchor="middle">ESFT</text><circle class="s2 ring" cx="419.4" cy="360.0" r="5"/><text class="t-tick" x="419.4" y="381.0" text-anchor="middle">GRM / SPCT</text><circle class="s2 ring" cx="682.9" cy="360.0" r="5"/><text class="t-tick" x="682.9" y="349.0" text-anchor="middle">DSpark</text></svg><figcaption>Every specialised release in this part, by family. Dates are the Hugging Face repository creation date or the arXiv date. The vision lane is the busiest; the code line ends when Coder-V2 is merged into V2.5 in September 2024.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 270" role="img" aria-label="Four files with import arrows are sorted so that every file comes after the files it uses, then joined into one training sample with a path comment on top of each file."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">One repository, ordered so each file comes after the files it imports</text><rect class="box" x="60.0" y="70.0" width="140.0" height="46.0" rx="10"/><text class="t-note" x="130.0" y="97.0" text-anchor="middle">utils.py</text><rect class="box" x="60.0" y="160.0" width="140.0" height="46.0" rx="10"/><text class="t-note" x="130.0" y="187.0" text-anchor="middle">model.py</text><rect class="box" x="250.0" y="70.0" width="140.0" height="46.0" rx="10"/><text class="t-note" x="320.0" y="97.0" text-anchor="middle">data.py</text><rect class="box" x="250.0" y="160.0" width="140.0" height="46.0" rx="10"/><text class="t-note" x="320.0" y="187.0" text-anchor="middle">train.py</text><line class="edge" x1="130.0" y1="116.0" x2="130.0" y2="160.0" marker-end="url(#ah)"/><line class="edge" x1="200.0" y1="93.0" x2="250.0" y2="93.0" marker-end="url(#ah)"/><line class="edge" x1="320.0" y1="116.0" x2="320.0" y2="160.0" marker-end="url(#ah)"/><line class="edge" x1="200.0" y1="183.0" x2="250.0" y2="183.0" marker-end="url(#ah)"/><text class="t-tick" x="60.0" y="236.0" text-anchor="start">Arrows: "is imported by". Found with regular expressions</text><text class="t-tick" x="60.0" y="254.0" text-anchor="start">for import (Python), using (C#), include (C).</text><line class="path" x1="410.0" y1="140.0" x2="460.0" y2="140.0" marker-end="url(#ah-on)"/><rect class="code-bg" x="470.0" y="50.0" width="290.0" height="200.0" rx="8"/><text class="t-label" x="484.0" y="72.0" text-anchor="start">one training sample</text><text class="t-code-key" x="490.0" y="96.0" text-anchor="start"># utils.py</text><text class="t-code" x="490.0" y="115.0" text-anchor="start">...code...</text><text class="t-code-key" x="490.0" y="134.0" text-anchor="start"># data.py</text><text class="t-code" x="490.0" y="153.0" text-anchor="start">...code...</text><text class="t-code-key" x="490.0" y="172.0" text-anchor="start"># model.py</text><text class="t-code" x="490.0" y="191.0" text-anchor="start">...code...</text><text class="t-code-key" x="490.0" y="210.0" text-anchor="start"># train.py</text><text class="t-code" x="490.0" y="229.0" text-anchor="start">...code...</text></svg><figcaption>A small repository. The arrows point from a file to the files that import it. Sorting the graph puts utils.py first and train.py last, and the files are joined in that order into one training sample.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 256" role="img" aria-label="A small Python file is cut into a prefix, a middle and a suffix; the training sample puts the prefix, then the suffix, then the middle, separated by special tokens."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Fill-in-the-middle: cut a file in three, then move the middle to the end</text><text class="t-label" x="20.0" y="56.0" text-anchor="start">The file as written</text><rect class="box-1" x="20.0" y="66.0" width="330.0" height="28.0" rx="6"/><text class="t-code" x="30.0" y="85.0" text-anchor="start">def area(r):</text><text class="t-tick" x="360.0" y="85.0" text-anchor="start">prefix</text><rect class="box-2" x="20.0" y="98.0" width="330.0" height="28.0" rx="6"/><text class="t-code" x="30.0" y="117.0" text-anchor="start">    return 3.14159 * r * r</text><text class="t-tick" x="360.0" y="117.0" text-anchor="start">middle</text><rect class="box-3" x="20.0" y="130.0" width="330.0" height="28.0" rx="6"/><text class="t-code" x="30.0" y="149.0" text-anchor="start">print(area(2))</text><text class="t-tick" x="360.0" y="149.0" text-anchor="start">suffix</text><line class="path" x1="420.0" y1="112.0" x2="460.0" y2="112.0" marker-end="url(#ah-on)"/><text class="t-label" x="470.0" y="56.0" text-anchor="start">The training sample (PSM order)</text><rect class="box" x="470.0" y="66.0" width="150.0" height="22.0" rx="5"/><text class="t-tick" x="545.0" y="82.0" text-anchor="middle">FIM start</text><rect class="box-1" x="470.0" y="92.0" width="150.0" height="22.0" rx="5"/><text class="t-tick" x="545.0" y="108.0" text-anchor="middle">prefix</text><rect class="box" x="470.0" y="118.0" width="150.0" height="22.0" rx="5"/><text class="t-tick" x="545.0" y="134.0" text-anchor="middle">hole</text><rect class="box-3" x="470.0" y="144.0" width="150.0" height="22.0" rx="5"/><text class="t-tick" x="545.0" y="160.0" text-anchor="middle">suffix</text><rect class="box" x="470.0" y="170.0" width="150.0" height="22.0" rx="5"/><text class="t-tick" x="545.0" y="186.0" text-anchor="middle">FIM end</text><rect class="box-2" x="470.0" y="196.0" width="150.0" height="22.0" rx="5"/><text class="t-tick" x="545.0" y="212.0" text-anchor="middle">middle</text><text class="t-tick" x="636.0" y="145.0" text-anchor="start">the model learns</text><text class="t-tick" x="636.0" y="163.0" text-anchor="start">to write the middle</text><text class="t-tick" x="636.0" y="181.0" text-anchor="start">after seeing both sides</text><text class="t-tick" x="20.0" y="240.0" text-anchor="start">In an editor: the code above and below the cursor become prefix and suffix; the model writes the middle.</text></svg><figcaption>Fill-in-the-middle on a three-line file. The middle line moves to the end of the training sample, after special marker tokens, so the model must predict it from the code on both sides.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 326" role="img" aria-label="The DeepSeekMath data loop: a seed set trains a fastText classifier, which scores Common Crawl; maths-heavy domains are found, people mark maths URL paths, missed pages join the seed, and the loop repeats four times to reach 35.5 million pages and 120 billion tokens."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Building a 120B-token maths corpus from Common Crawl: four rounds of the same loop</text><rect class="box" x="30.0" y="50.0" width="200.0" height="56.0" rx="10"/><text class="t-note" x="130.0" y="73.0" text-anchor="middle">Seed: 500K pages</text><text class="t-tick" x="130.0" y="91.0" text-anchor="middle">from OpenWebMath</text><rect class="box" x="290.0" y="50.0" width="200.0" height="56.0" rx="10"/><text class="t-note" x="390.0" y="73.0" text-anchor="middle">Train fastText</text><text class="t-tick" x="390.0" y="91.0" text-anchor="middle">math vs not math</text><rect class="box" x="550.0" y="50.0" width="200.0" height="56.0" rx="10"/><text class="t-note" x="650.0" y="73.0" text-anchor="middle">Score 40B pages</text><text class="t-tick" x="650.0" y="91.0" text-anchor="middle">keep the top ones</text><rect class="box" x="550.0" y="170.0" width="200.0" height="56.0" rx="10"/><text class="t-note" x="650.0" y="193.0" text-anchor="middle">Find maths domains</text><text class="t-tick" x="650.0" y="211.0" text-anchor="middle">&gt;10% pages collected</text><rect class="box" x="290.0" y="170.0" width="200.0" height="56.0" rx="10"/><text class="t-note" x="390.0" y="193.0" text-anchor="middle">People mark maths</text><text class="t-tick" x="390.0" y="211.0" text-anchor="middle">URL paths</text><rect class="box" x="30.0" y="170.0" width="200.0" height="56.0" rx="10"/><text class="t-note" x="130.0" y="193.0" text-anchor="middle">Add missed pages</text><text class="t-tick" x="130.0" y="211.0" text-anchor="middle">to the seed</text><line class="edge" x1="230.0" y1="78.0" x2="290.0" y2="78.0" marker-end="url(#ah)"/><line class="edge" x1="490.0" y1="78.0" x2="550.0" y2="78.0" marker-end="url(#ah)"/><line class="edge" x1="650.0" y1="106.0" x2="650.0" y2="170.0" marker-end="url(#ah)"/><line class="edge" x1="550.0" y1="198.0" x2="490.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="290.0" y1="198.0" x2="230.0" y2="198.0" marker-end="url(#ah)"/><line class="edge" x1="130.0" y1="170.0" x2="130.0" y2="106.0" marker-end="url(#ah)"/><rect class="box-on" x="200.0" y="252.0" width="380.0" height="40.0" rx="8"/><text class="t-strong" x="390.0" y="277.0" text-anchor="middle">After 4 rounds: 35.5M pages, 120B tokens</text><text class="t-tick" x="390.0" y="312.0" text-anchor="middle">Round 4 found almost nothing new: nearly 98% was already in round 3, so they stopped.</text></svg><figcaption>The DeepSeekMath data loop. Each round, a better classifier finds more maths, and the websites it finds point people to the pages it still misses. After four rounds the loop stopped finding new data.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 344" role="img" aria-label="Two answer distributions before and after squaring and renormalising: the right answer grows from 0.50 to 0.77 when it is already the favourite, and shrinks from 0.25 to 0.21 when it is second."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">RL as sharpening: squaring each probability and renormalising (alpha = 2)</text><text class="t-note" x="20.0" y="54.0" text-anchor="start">Question A: the right answer is already the favourite</text><text class="t-label" x="20.0" y="82.0" text-anchor="start">before RL</text><rect class="box-on" x="90.0" y="126.0" width="40.0" height="40.0" rx="3"/><text class="t-tick" x="110.0" y="120.0" text-anchor="middle">0.50</text><text class="t-tick" x="110.0" y="182.0" text-anchor="middle">right</text><rect class="box" x="148.0" y="150.0" width="40.0" height="16.0" rx="3"/><text class="t-tick" x="168.0" y="144.0" text-anchor="middle">0.20</text><text class="t-tick" x="168.0" y="182.0" text-anchor="middle">w1</text><rect class="box" x="206.0" y="154.0" width="40.0" height="12.0" rx="3"/><text class="t-tick" x="226.0" y="148.0" text-anchor="middle">0.15</text><text class="t-tick" x="226.0" y="182.0" text-anchor="middle">w2</text><rect class="box" x="264.0" y="158.0" width="40.0" height="8.0" rx="3"/><text class="t-tick" x="284.0" y="152.0" text-anchor="middle">0.10</text><text class="t-tick" x="284.0" y="182.0" text-anchor="middle">w3</text><rect class="box" x="322.0" y="162.0" width="40.0" height="4.0" rx="3"/><text class="t-tick" x="342.0" y="156.0" text-anchor="middle">0.05</text><text class="t-tick" x="342.0" y="182.0" text-anchor="middle">w4</text><text class="t-label" x="20.0" y="202.0" text-anchor="start">after RL</text><rect class="box-on" x="90.0" y="224.5" width="40.0" height="61.5" rx="3"/><text class="t-tick" x="110.0" y="218.5" text-anchor="middle">0.77</text><text class="t-tick" x="110.0" y="302.0" text-anchor="middle">right</text><rect class="box-1" x="148.0" y="276.2" width="40.0" height="9.8" rx="3"/><text class="t-tick" x="168.0" y="270.2" text-anchor="middle">0.12</text><text class="t-tick" x="168.0" y="302.0" text-anchor="middle">w1</text><rect class="box-1" x="206.0" y="280.5" width="40.0" height="5.5" rx="3"/><text class="t-tick" x="226.0" y="274.5" text-anchor="middle">0.07</text><text class="t-tick" x="226.0" y="302.0" text-anchor="middle">w2</text><rect class="box-1" x="264.0" y="283.5" width="40.0" height="2.5" rx="3"/><text class="t-tick" x="284.0" y="277.5" text-anchor="middle">0.03</text><text class="t-tick" x="284.0" y="302.0" text-anchor="middle">w3</text><rect class="box-1" x="322.0" y="285.4" width="40.0" height="0.6" rx="3"/><text class="t-tick" x="342.0" y="279.4" text-anchor="middle">0.01</text><text class="t-tick" x="342.0" y="302.0" text-anchor="middle">w4</text><text class="t-note" x="410.0" y="54.0" text-anchor="start">Question B: the right answer is second</text><text class="t-label" x="410.0" y="82.0" text-anchor="start">before RL</text><rect class="box-on" x="480.0" y="146.0" width="40.0" height="20.0" rx="3"/><text class="t-tick" x="500.0" y="140.0" text-anchor="middle">0.25</text><text class="t-tick" x="500.0" y="182.0" text-anchor="middle">right</text><rect class="box" x="538.0" y="130.0" width="40.0" height="36.0" rx="3"/><text class="t-tick" x="558.0" y="124.0" text-anchor="middle">0.45</text><text class="t-tick" x="558.0" y="182.0" text-anchor="middle">w1</text><rect class="box" x="596.0" y="154.0" width="40.0" height="12.0" rx="3"/><text class="t-tick" x="616.0" y="148.0" text-anchor="middle">0.15</text><text class="t-tick" x="616.0" y="182.0" text-anchor="middle">w2</text><rect class="box" x="654.0" y="158.0" width="40.0" height="8.0" rx="3"/><text class="t-tick" x="674.0" y="152.0" text-anchor="middle">0.10</text><text class="t-tick" x="674.0" y="182.0" text-anchor="middle">w3</text><rect class="box" x="712.0" y="162.0" width="40.0" height="4.0" rx="3"/><text class="t-tick" x="732.0" y="156.0" text-anchor="middle">0.05</text><text class="t-tick" x="732.0" y="182.0" text-anchor="middle">w4</text><text class="t-label" x="410.0" y="202.0" text-anchor="start">after RL</text><rect class="box-on" x="480.0" y="269.3" width="40.0" height="16.7" rx="3"/><text class="t-tick" x="500.0" y="263.3" text-anchor="middle">0.21</text><text class="t-tick" x="500.0" y="302.0" text-anchor="middle">right</text><rect class="box-1" x="538.0" y="232.0" width="40.0" height="54.0" rx="3"/><text class="t-tick" x="558.0" y="226.0" text-anchor="middle">0.67</text><text class="t-tick" x="558.0" y="302.0" text-anchor="middle">w1</text><rect class="box-1" x="596.0" y="280.0" width="40.0" height="6.0" rx="3"/><text class="t-tick" x="616.0" y="274.0" text-anchor="middle">0.07</text><text class="t-tick" x="616.0" y="302.0" text-anchor="middle">w2</text><rect class="box-1" x="654.0" y="283.3" width="40.0" height="2.7" rx="3"/><text class="t-tick" x="674.0" y="277.3" text-anchor="middle">0.03</text><text class="t-tick" x="674.0" y="302.0" text-anchor="middle">w3</text><rect class="box-1" x="712.0" y="285.3" width="40.0" height="0.7" rx="3"/><text class="t-tick" x="732.0" y="279.3" text-anchor="middle">0.01</text><text class="t-tick" x="732.0" y="302.0" text-anchor="middle">w4</text><text class="t-tick" x="20.0" y="330.0" text-anchor="start">Highlighted bar: the right answer. Sharpening helps A (0.50 to 0.77) and hurts B (0.25 to 0.21).</text></svg><figcaption>Sharpening on two questions. When the right answer is already the favourite (A), squaring and renormalising lifts it from 0.50 to 0.77. When the right answer is second (B), the same operation lowers it from 0.25 to 0.21.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 330" role="img" aria-label="Toy simulation: Pass@K and Maj@K for an instruct model and its sharpened RL version, K from 1 to 64."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="280.0" x2="590" y2="280.0"/><text class="t-tick" x="56.0" y="284.0" text-anchor="end">40</text><line class="grid" x1="64" y1="238.3" x2="590" y2="238.3"/><text class="t-tick" x="56.0" y="242.3" text-anchor="end">50</text><line class="grid" x1="64" y1="196.7" x2="590" y2="196.7"/><text class="t-tick" x="56.0" y="200.7" text-anchor="end">60</text><line class="grid" x1="64" y1="155.0" x2="590" y2="155.0"/><text class="t-tick" x="56.0" y="159.0" text-anchor="end">70</text><line class="grid" x1="64" y1="113.3" x2="590" y2="113.3"/><text class="t-tick" x="56.0" y="117.3" text-anchor="end">80</text><line class="grid" x1="64" y1="71.7" x2="590" y2="71.7"/><text class="t-tick" x="56.0" y="75.7" text-anchor="end">90</text><line class="grid" x1="64" y1="30.0" x2="590" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">100</text><text class="t-tick" x="64.0" y="300.0" text-anchor="middle">1</text><text class="t-tick" x="239.3" y="300.0" text-anchor="middle">4</text><text class="t-tick" x="327.0" y="300.0" text-anchor="middle">8</text><text class="t-tick" x="414.7" y="300.0" text-anchor="middle">16</text><text class="t-tick" x="502.3" y="300.0" text-anchor="middle">32</text><text class="t-tick" x="590.0" y="300.0" text-anchor="middle">64</text><line class="axis" x1="64" y1="280" x2="590" y2="280"/><polyline class="l1" points="64.0,269.1 239.3,145.9 327.0,100.2 414.7,68.2 502.3,48.5 590.0,37.8"/><g class="mark"><title>Pass@K instruct, 1: 43</title><circle class="s1 ring" cx="64.0" cy="269.1" r="5"/></g><g class="mark"><title>Pass@K instruct, 4: 72</title><circle class="s1 ring" cx="239.3" cy="145.9" r="5"/></g><g class="mark"><title>Pass@K instruct, 8: 83</title><circle class="s1 ring" cx="327.0" cy="100.2" r="5"/></g><g class="mark"><title>Pass@K instruct, 16: 91</title><circle class="s1 ring" cx="414.7" cy="68.2" r="5"/></g><g class="mark"><title>Pass@K instruct, 32: 96</title><circle class="s1 ring" cx="502.3" cy="48.5" r="5"/></g><g class="mark"><title>Pass@K instruct, 64: 98</title><circle class="s1 ring" cx="590.0" cy="37.8" r="5"/></g><polyline class="l2" points="64.0,237.3 239.3,163.6 327.0,135.0 414.7,110.4 502.3,89.8 590.0,73.2"/><g class="mark"><title>Pass@K RL, 1: 50</title><circle class="s2 ring" cx="64.0" cy="237.3" r="5"/></g><g class="mark"><title>Pass@K RL, 4: 68</title><circle class="s2 ring" cx="239.3" cy="163.6" r="5"/></g><g class="mark"><title>Pass@K RL, 8: 75</title><circle class="s2 ring" cx="327.0" cy="135.0" r="5"/></g><g class="mark"><title>Pass@K RL, 16: 81</title><circle class="s2 ring" cx="414.7" cy="110.4" r="5"/></g><g class="mark"><title>Pass@K RL, 32: 86</title><circle class="s2 ring" cx="502.3" cy="89.8" r="5"/></g><g class="mark"><title>Pass@K RL, 64: 90</title><circle class="s2 ring" cx="590.0" cy="73.2" r="5"/></g><polyline class="l3" points="64.0,269.1 239.3,246.4 327.0,234.6 414.7,227.6 502.3,224.2 590.0,222.4"/><g class="mark"><title>Maj@K instruct, 1: 43</title><circle class="s3 ring" cx="64.0" cy="269.1" r="5"/></g><g class="mark"><title>Maj@K instruct, 4: 48</title><circle class="s3 ring" cx="239.3" cy="246.4" r="5"/></g><g class="mark"><title>Maj@K instruct, 8: 51</title><circle class="s3 ring" cx="327.0" cy="234.6" r="5"/></g><g class="mark"><title>Maj@K instruct, 16: 53</title><circle class="s3 ring" cx="414.7" cy="227.6" r="5"/></g><g class="mark"><title>Maj@K instruct, 32: 53</title><circle class="s3 ring" cx="502.3" cy="224.2" r="5"/></g><g class="mark"><title>Maj@K instruct, 64: 54</title><circle class="s3 ring" cx="590.0" cy="222.4" r="5"/></g><polyline class="l4" points="64.0,237.4 239.3,227.2 327.0,223.5 414.7,221.7 502.3,220.6 590.0,220.0"/><g class="mark"><title>Maj@K RL, 1: 50</title><circle class="s4 ring" cx="64.0" cy="237.4" r="5"/></g><g class="mark"><title>Maj@K RL, 4: 53</title><circle class="s4 ring" cx="239.3" cy="227.2" r="5"/></g><g class="mark"><title>Maj@K RL, 8: 54</title><circle class="s4 ring" cx="327.0" cy="223.5" r="5"/></g><g class="mark"><title>Maj@K RL, 16: 54</title><circle class="s4 ring" cx="414.7" cy="221.7" r="5"/></g><g class="mark"><title>Maj@K RL, 32: 54</title><circle class="s4 ring" cx="502.3" cy="220.6" r="5"/></g><g class="mark"><title>Maj@K RL, 64: 54</title><circle class="s4 ring" cx="590.0" cy="220.0" r="5"/></g><line class="grid" x1="596.0" y1="37.8" x2="606.0" y2="37.8"/><rect class="s1" x="610.0" y="32.8" width="10" height="10" rx="2"/><text class="t-note" x="626.0" y="41.8" text-anchor="start">Pass@K instruct: 98</text><line class="grid" x1="596.0" y1="73.2" x2="606.0" y2="73.2"/><rect class="s2" x="610.0" y="68.2" width="10" height="10" rx="2"/><text class="t-note" x="626.0" y="77.2" text-anchor="start">Pass@K RL: 90</text><line class="grid" x1="596.0" y1="220.0" x2="606.0" y2="220.0"/><rect class="s4" x="610.0" y="215.0" width="10" height="10" rx="2"/><text class="t-note" x="626.0" y="224.0" text-anchor="start">Maj@K RL: 54</text><line class="grid" x1="596.0" y1="222.4" x2="606.0" y2="237.0"/><rect class="s3" x="610.0" y="232.0" width="10" height="10" rx="2"/><text class="t-note" x="626.0" y="241.0" text-anchor="start">Maj@K instruct: 54</text><text class="t-tick" x="327.0" y="320.0" text-anchor="middle">K: number of samples per question (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">accuracy (%)</text></svg><figcaption>The toy simulation. Sharpening lifts Pass@1 and Maj@K at small K, but Pass@K at large K ends lower than before (89.6% against 98.1% at K = 64), because questions where the right answer was a long shot lose probability.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 358" role="img" aria-label="DeepSeekMath-V2 training loop: a generator writes a proof and a self-critique, a verifier lists issues and scores it, a meta-verifier checks that the listed issues are real, and the resulting reward updates the generator; heavy verification labels hard new proofs to retrain the verifier."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">DeepSeekMath-V2: one model learns to write proofs and to grade them</text><rect class="box" x="300.0" y="40.0" width="180.0" height="50.0" rx="10"/><text class="t-note" x="390.0" y="60.0" text-anchor="middle">Hard new proofs</text><text class="t-tick" x="390.0" y="78.0" text-anchor="middle">labelled by many checks</text><line class="edge" x1="390.0" y1="90.0" x2="390.0" y2="120.0" marker-end="url(#ah)"/><text class="t-tick" x="500.0" y="70.0" text-anchor="start">retrain the verifier, so it stays ahead</text><rect class="box-1" x="20.0" y="120.0" width="180.0" height="64.0" rx="10"/><text class="t-note" x="110.0" y="138.0" text-anchor="middle">Generator</text><text class="t-tick" x="110.0" y="156.0" text-anchor="middle">writes a proof +</text><text class="t-tick" x="110.0" y="174.0" text-anchor="middle">its own critique</text><rect class="box-2" x="300.0" y="120.0" width="180.0" height="64.0" rx="10"/><text class="t-note" x="390.0" y="138.0" text-anchor="middle">Verifier</text><text class="t-tick" x="390.0" y="156.0" text-anchor="middle">lists issues,</text><text class="t-tick" x="390.0" y="174.0" text-anchor="middle">scores 0, 0.5 or 1</text><rect class="box-3" x="580.0" y="120.0" width="180.0" height="64.0" rx="10"/><text class="t-note" x="670.0" y="138.0" text-anchor="middle">Meta-verifier</text><text class="t-tick" x="670.0" y="156.0" text-anchor="middle">are those issues</text><text class="t-tick" x="670.0" y="174.0" text-anchor="middle">real and fair?</text><line class="edge" x1="200.0" y1="152.0" x2="300.0" y2="152.0" marker-end="url(#ah)"/><line class="edge" x1="480.0" y1="152.0" x2="580.0" y2="152.0" marker-end="url(#ah)"/><rect class="box-on" x="300.0" y="240.0" width="180.0" height="56.0" rx="10"/><text class="t-note" x="390.0" y="263.0" text-anchor="middle">Reward</text><text class="t-tick" x="390.0" y="281.0" text-anchor="middle">proof score x honesty</text><line class="edge" x1="390.0" y1="184.0" x2="390.0" y2="240.0" marker-end="url(#ah)"/><line class="edge" x1="670.0" y1="184.0" x2="480.0" y2="260.0" marker-end="url(#ah)"/><line class="path" x1="300.0" y1="268.0" x2="110.0" y2="184.0" marker-end="url(#ah-on)"/><text class="t-tick" x="150.0" y="260.0" text-anchor="middle">RL update</text><text class="t-tick" x="20.0" y="326.0" text-anchor="start">The generator is paid for a good proof and for an honest self-check:</text><text class="t-tick" x="20.0" y="344.0" text-anchor="start">claiming "no issues" on a flawed proof costs reward.</text></svg><figcaption>The DeepSeekMath-V2 loop. The generator writes a proof and its own critique; the verifier scores the proof; the meta-verifier checks the verifier's reasons. Hard new proofs, labelled by many independent checks, retrain the verifier so it stays ahead of the generator.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 312" role="img" aria-label="Prover-V2 cold-start pipeline: DeepSeek-V3 writes a proof sketch with subgoals left as sorry, a 7B prover proves each subgoal, Lean checks the stitched proof, and the checked proof plus V3 reasoning becomes training data for reinforcement learning."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Prover-V2 cold start: a big model plans, a small model fills the holes, Lean checks</text><rect class="box" x="20.0" y="56.0" width="160.0" height="60.0" rx="10"/><text class="t-note" x="100.0" y="81.0" text-anchor="middle">Hard theorem</text><text class="t-tick" x="100.0" y="99.0" text-anchor="middle">in Lean 4</text><rect class="box-1" x="220.0" y="56.0" width="200.0" height="60.0" rx="10"/><text class="t-note" x="320.0" y="72.0" text-anchor="middle">DeepSeek-V3</text><text class="t-tick" x="320.0" y="90.0" text-anchor="middle">sketch + subgoals</text><text class="t-tick" x="320.0" y="108.0" text-anchor="middle">each "by sorry"</text><line class="edge" x1="180.0" y1="86.0" x2="220.0" y2="86.0" marker-end="url(#ah)"/><line class="edge" x1="420.0" y1="86.0" x2="460.0" y2="86.0" marker-end="url(#ah)"/><rect class="box-2" x="460.0" y="40.0" width="140.0" height="36.0" rx="10"/><text class="t-note" x="530.0" y="62.0" text-anchor="middle">subgoal 1</text><rect class="box-2" x="460.0" y="84.0" width="140.0" height="36.0" rx="10"/><text class="t-note" x="530.0" y="106.0" text-anchor="middle">subgoal 2</text><rect class="box-2" x="460.0" y="128.0" width="140.0" height="36.0" rx="10"/><text class="t-note" x="530.0" y="150.0" text-anchor="middle">subgoal 3</text><rect class="box-2" x="630.0" y="56.0" width="130.0" height="60.0" rx="10"/><text class="t-note" x="695.0" y="72.0" text-anchor="middle">Prover 7B</text><text class="t-tick" x="695.0" y="90.0" text-anchor="middle">proves each</text><text class="t-tick" x="695.0" y="108.0" text-anchor="middle">subgoal</text><line class="edge" x1="600.0" y1="86.0" x2="630.0" y2="86.0" marker-end="url(#ah)"/><rect class="box-on" x="460.0" y="200.0" width="300.0" height="56.0" rx="10"/><text class="t-note" x="610.0" y="223.0" text-anchor="middle">Lean 4 checks the stitched proof</text><text class="t-tick" x="610.0" y="241.0" text-anchor="middle">accepted or rejected, nothing in between</text><line class="edge" x1="695.0" y1="116.0" x2="695.0" y2="200.0" marker-end="url(#ah)"/><rect class="box-3" x="20.0" y="200.0" width="360.0" height="56.0" rx="10"/><text class="t-note" x="200.0" y="223.0" text-anchor="middle">Training data</text><text class="t-tick" x="200.0" y="241.0" text-anchor="middle">V3's reasoning + the checked formal proof</text><line class="edge" x1="460.0" y1="228.0" x2="380.0" y2="228.0" marker-end="url(#ah)"/><text class="t-tick" x="20.0" y="296.0" text-anchor="start">Then RL (GRPO) with the Lean verdict as reward. The released 671B model starts from DeepSeek-V3-Base.</text></svg><figcaption>The Prover-V2 cold-start pipeline. V3 turns a hard theorem into subgoals; the 7B prover proves each one; Lean checks the stitched proof; the result plus V3's reasoning becomes training data for the RL stage.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 318" role="img" aria-label="Bar chart of miniF2F-test pass rates: Prover-V1 46.3 and 52.0, Prover-V1.5 60.2 and 63.5 with tree search, Prover-V2 7B 82.0 and 671B 88.9 at 8192 samples."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">miniF2F-test, share of 244 competition problems proved and checked by Lean (%)</text><text class="t-tick" x="220.0" y="68.0" text-anchor="end">Prover-V1, 64 samples</text><rect class="box" x="230.0" y="52.0" width="217.6" height="26.0" rx="4"/><text class="t-val" x="455.6" y="70.0" text-anchor="start">46.3</text><text class="t-tick" x="220.0" y="108.0" text-anchor="end">Prover-V1, cumulative</text><rect class="box" x="230.0" y="92.0" width="244.4" height="26.0" rx="4"/><text class="t-val" x="482.4" y="110.0" text-anchor="start">52.0</text><text class="t-tick" x="220.0" y="148.0" text-anchor="end">Prover-V1.5, whole proof</text><rect class="box-1" x="230.0" y="132.0" width="282.9" height="26.0" rx="4"/><text class="t-val" x="520.9" y="150.0" text-anchor="start">60.2</text><text class="t-tick" x="220.0" y="188.0" text-anchor="end">Prover-V1.5 + RMaxTS</text><rect class="box-1" x="230.0" y="172.0" width="298.4" height="26.0" rx="4"/><text class="t-val" x="536.5" y="190.0" text-anchor="start">63.5</text><text class="t-tick" x="220.0" y="228.0" text-anchor="end">Prover-V2 7B, Pass@8192</text><rect class="box-2" x="230.0" y="212.0" width="385.4" height="26.0" rx="4"/><text class="t-val" x="623.4" y="230.0" text-anchor="start">82.0</text><text class="t-tick" x="220.0" y="268.0" text-anchor="end">Prover-V2 671B, Pass@8192</text><rect class="box-on" x="230.0" y="252.0" width="417.8" height="26.0" rx="4"/><text class="t-val" x="655.8" y="270.0" text-anchor="start">88.9</text><text class="t-tick" x="20.0" y="304.0" text-anchor="start">Settings differ between rows (samples, search), so read it as a trend, not a fair race. Sources: each paper.</text></svg><figcaption>miniF2F-test pass rates across the Prover line. Each bar comes from its own paper, with different sampling budgets and search, so the chart shows a trend rather than a fair comparison.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 244" role="img" aria-label="DeepSeek-VL2 cuts a 1920 by 1080 screenshot into 4 by 2 tiles of 384 pixels plus one thumbnail; each tile becomes 196 tokens, for 1,807 vision tokens in total."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">DeepSeek-VL2 dynamic tiling: a 1920 x 1080 screenshot</text><rect class="box" x="20.0" y="50.0" width="192.0" height="108.0" rx="4"/><text class="t-tick" x="116.0" y="108.0" text-anchor="middle">1920 x 1080</text><line class="path" x1="220.0" y1="104.0" x2="260.0" y2="104.0" marker-end="url(#ah-on)"/><rect class="box-1" x="270.0" y="52.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="322.0" y="52.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="374.0" y="52.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="426.0" y="52.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="270.0" y="104.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="322.0" y="104.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="374.0" y="104.0" width="48.0" height="48.0" rx="3"/><rect class="box-1" x="426.0" y="104.0" width="48.0" height="48.0" rx="3"/><text class="t-tick" x="374.0" y="174.0" text-anchor="middle">4 x 2 local tiles of 384 x 384</text><rect class="box-2" x="510.0" y="66.0" width="70.0" height="70.0" rx="3"/><text class="t-tick" x="545.0" y="156.0" text-anchor="middle">+ 1 thumbnail</text><line class="path" x1="590.0" y1="104.0" x2="620.0" y2="104.0" marker-end="url(#ah-on)"/><rect class="box" x="626.0" y="60.0" width="140.0" height="90.0" rx="10"/><text class="t-note" x="696.0" y="82.0" text-anchor="middle">SigLIP per tile</text><text class="t-tick" x="696.0" y="100.0" text-anchor="middle">27 x 27 = 729</text><text class="t-tick" x="696.0" y="118.0" text-anchor="middle">2x2 squeeze: 196</text><text class="t-tick" x="696.0" y="136.0" text-anchor="middle">total 1,807</text><text class="t-note" x="20.0" y="206.0" text-anchor="start">Total vision tokens = 14 x 15 (thumbnail) + 1 + (2 x 14) x (4 x 14 + 1) = 1,807.</text><text class="t-tick" x="20.0" y="228.0" text-anchor="start">The grid is chosen from all cols x rows &lt;= 9 to keep the most detail; the language side is a DeepSeekMoE model.</text></svg><figcaption>DeepSeek-VL2 on a 1920 x 1080 screenshot. The released code chooses a 4 x 2 grid of 384-pixel tiles plus a thumbnail; each tile becomes 196 tokens after a 2 x 2 squeeze, for 1,807 vision tokens.</figcaption></figure>

The language side of VL2 is a DeepSeekMoE model, with MLA in the two larger sizes. The three sizes are Tiny (3B total, 1.0B active including the vision part), Small (16B, 2.8B) and VL2 (27B, 4.5B). The largest scored 93.3 on DocVQA and 86.0 on ChartQA (questions about documents and charts), and 811 on OCRBench.

## Understanding and drawing: Janus, JanusFlow and Janus-Pro

A model that can both describe an image and draw one sounds like it should share one way of seeing. Janus (October 2024) argues the opposite.

Understanding wants meaning: "two cats on a red sofa". Drawing wants exact layout: which pixel goes where, at what colour. An encoder tuned for one is poor at the other. The Janus paper argues that earlier unified models, such as Chameleon, used one image encoder for both jobs and lost understanding quality.

> [!DEFINITION] VQ tokenizer
> A "vector-quantised" image tokenizer. It cuts an image into patches and replaces each patch with the ID of the closest entry in a fixed codebook. An image becomes a short list of discrete codes, like words, that a language model can predict one at a time. A decoder turns codes back into pixels.

<figure class="fig"><svg viewBox="0 0 780 318" role="img" aria-label="Janus architecture: images for understanding enter through a SigLIP encoder and images for generation through a VQ tokenizer, both feeding one autoregressive transformer with a text head and an image head."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Janus: one transformer, two ways in for images</text><rect class="box-1" x="20.0" y="220.0" width="170.0" height="56.0" rx="10"/><text class="t-note" x="105.0" y="243.0" text-anchor="middle">Understanding</text><text class="t-tick" x="105.0" y="261.0" text-anchor="middle">SigLIP: meaning</text><rect class="box" x="210.0" y="220.0" width="150.0" height="56.0" rx="10"/><text class="t-note" x="285.0" y="243.0" text-anchor="middle">Text tokenizer</text><text class="t-tick" x="285.0" y="261.0" text-anchor="middle">the question</text><rect class="box" x="420.0" y="220.0" width="150.0" height="56.0" rx="10"/><text class="t-note" x="495.0" y="243.0" text-anchor="middle">Text tokenizer</text><text class="t-tick" x="495.0" y="261.0" text-anchor="middle">the prompt</text><rect class="box-2" x="590.0" y="220.0" width="170.0" height="56.0" rx="10"/><text class="t-note" x="675.0" y="243.0" text-anchor="middle">Generation</text><text class="t-tick" x="675.0" y="261.0" text-anchor="middle">VQ: image codes</text><rect class="box-on" x="20.0" y="120.0" width="740.0" height="50.0" rx="10"/><text class="t-note" x="390.0" y="149.0" text-anchor="middle">One autoregressive transformer (DeepSeek-LLM 1.3B to 7B)</text><line class="edge" x1="105.0" y1="220.0" x2="105.0" y2="170.0" marker-end="url(#ah)"/><line class="edge" x1="285.0" y1="220.0" x2="285.0" y2="170.0" marker-end="url(#ah)"/><line class="edge" x1="495.0" y1="220.0" x2="495.0" y2="170.0" marker-end="url(#ah)"/><line class="edge" x1="675.0" y1="220.0" x2="675.0" y2="170.0" marker-end="url(#ah)"/><rect class="box-1" x="110.0" y="46.0" width="200.0" height="48.0" rx="10"/><text class="t-note" x="210.0" y="65.0" text-anchor="middle">Text head</text><text class="t-tick" x="210.0" y="83.0" text-anchor="middle">writes the answer</text><rect class="box-2" x="470.0" y="46.0" width="200.0" height="48.0" rx="10"/><text class="t-note" x="570.0" y="65.0" text-anchor="middle">Image head</text><text class="t-tick" x="570.0" y="83.0" text-anchor="middle">next image code</text><line class="edge" x1="210.0" y1="120.0" x2="210.0" y2="94.0" marker-end="url(#ah)"/><line class="edge" x1="570.0" y1="120.0" x2="570.0" y2="94.0" marker-end="url(#ah)"/><text class="t-tick" x="20.0" y="304.0" text-anchor="start">Understanding needs "what is in the picture"; drawing needs "which pixels go where". One encoder for both made each worse.</text></svg><figcaption>Janus routes images through two separate encoders: SigLIP when the task is to understand an image, and a VQ tokenizer when the task is to draw one. Both feed one transformer, which has a text head and an image head.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 218" role="img" aria-label="DeepSeek-OCR pipeline: a 1024 by 1024 page is cut into 4,096 patches, passed through SAM with window attention, squeezed 16 times to 256 tokens by convolutions, passed through CLIP with global attention, and read by a 3B mixture-of-experts decoder."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">DeepSeek-OCR at Base mode: a 1024 x 1024 page becomes 256 vision tokens</text><rect class="box" x="20.0" y="56.0" width="112.0" height="64.0" rx="10"/><text class="t-note" x="76.0" y="83.0" text-anchor="middle">page image</text><text class="t-tick" x="76.0" y="101.0" text-anchor="middle">1024 x 1024</text><rect class="box" x="146.0" y="56.0" width="112.0" height="64.0" rx="10"/><text class="t-note" x="202.0" y="83.0" text-anchor="middle">16 x 16 patches</text><text class="t-tick" x="202.0" y="101.0" text-anchor="middle">4,096 tokens</text><line class="edge" x1="132.0" y1="88.0" x2="146.0" y2="88.0" marker-end="url(#ah)"/><rect class="box-1" x="272.0" y="56.0" width="112.0" height="64.0" rx="10"/><text class="t-note" x="328.0" y="83.0" text-anchor="middle">SAM-base 80M</text><text class="t-tick" x="328.0" y="101.0" text-anchor="middle">window attention</text><line class="edge" x1="258.0" y1="88.0" x2="272.0" y2="88.0" marker-end="url(#ah)"/><rect class="box-on" x="398.0" y="56.0" width="112.0" height="64.0" rx="10"/><text class="t-note" x="454.0" y="83.0" text-anchor="middle">conv squeeze</text><text class="t-tick" x="454.0" y="101.0" text-anchor="middle">16x: 256 tokens</text><line class="edge" x1="384.0" y1="88.0" x2="398.0" y2="88.0" marker-end="url(#ah)"/><rect class="box-2" x="524.0" y="56.0" width="112.0" height="64.0" rx="10"/><text class="t-note" x="580.0" y="83.0" text-anchor="middle">CLIP-large 300M</text><text class="t-tick" x="580.0" y="101.0" text-anchor="middle">global attention</text><line class="edge" x1="510.0" y1="88.0" x2="524.0" y2="88.0" marker-end="url(#ah)"/><rect class="box-3" x="650.0" y="56.0" width="112.0" height="64.0" rx="10"/><text class="t-note" x="706.0" y="83.0" text-anchor="middle">MoE decoder</text><text class="t-tick" x="706.0" y="101.0" text-anchor="middle">3B, 570M active</text><line class="edge" x1="636.0" y1="88.0" x2="650.0" y2="88.0" marker-end="url(#ah)"/><text class="t-note" x="20.0" y="160.0" text-anchor="start">The same page as text: 761 to 1,061 text tokens (three pages of the OCR paper, our count).</text><text class="t-tick" x="20.0" y="182.0" text-anchor="start">Compression = text tokens / vision tokens: about 3.0x to 4.1x at Base, 11.9x to 16.6x at Tiny (64 tokens).</text><text class="t-tick" x="20.0" y="202.0" text-anchor="start">Window attention keeps memory low on 4,096 tokens; global attention runs only after the squeeze.</text></svg><figcaption>DeepSeek-OCR at Base mode. A 1024 x 1024 page is split into 4,096 patches, read cheaply by SAM, squeezed 16 times to 256 tokens, read globally by CLIP and passed to the MoE decoder. The same page holds 761 to 1,061 text tokens in our count.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 364" role="img" aria-label="Line chart of DeepSeek-OCR precision against compression ratio: with 100 vision tokens precision stays between 87.1 and 98.5 percent up to 12.6 times; with 64 tokens it falls from 96.5 percent at 10.5 times to 59.1 percent at 19.7 times."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">DeepSeek-OCR: decoding precision against compression (Fox benchmark, paper Table 2)</text><line class="grid" x1="70.0" y1="270.0" x2="630.0" y2="270.0"/><text class="t-tick" x="62.0" y="274.0" text-anchor="end">50%</text><line class="grid" x1="70.0" y1="224.0" x2="630.0" y2="224.0"/><text class="t-tick" x="62.0" y="228.0" text-anchor="end">60%</text><line class="grid" x1="70.0" y1="178.0" x2="630.0" y2="178.0"/><text class="t-tick" x="62.0" y="182.0" text-anchor="end">70%</text><line class="grid" x1="70.0" y1="132.0" x2="630.0" y2="132.0"/><text class="t-tick" x="62.0" y="136.0" text-anchor="end">80%</text><line class="grid" x1="70.0" y1="86.0" x2="630.0" y2="86.0"/><text class="t-tick" x="62.0" y="90.0" text-anchor="end">90%</text><line class="grid" x1="70.0" y1="40.0" x2="630.0" y2="40.0"/><text class="t-tick" x="62.0" y="44.0" text-anchor="end">100%</text><text class="t-tick" x="70.0" y="290.0" text-anchor="middle">5x</text><text class="t-tick" x="245.0" y="290.0" text-anchor="middle">10x</text><text class="t-tick" x="420.0" y="290.0" text-anchor="middle">15x</text><text class="t-tick" x="595.0" y="290.0" text-anchor="middle">20x</text><line class="base-line" x1="245.0" y1="40.0" x2="245.0" y2="270.0"/><text class="t-tick" x="251.0" y="262.0" text-anchor="start">10x line</text><polyline class="l2" points="129.5,46.9 157.5,52.4 192.5,54.7 234.5,54.7 266.0,79.1 290.5,86.9 336.0,99.3"/><g class="mark"><title>100 vision tokens, 600-700 text tokens: 98.5% at 6.7x</title><circle class="s2 ring" cx="129.5" cy="46.9" r="5"/></g><g class="mark"><title>100 vision tokens, 700-800 text tokens: 97.3% at 7.5x</title><circle class="s2 ring" cx="157.5" cy="52.4" r="5"/></g><g class="mark"><title>100 vision tokens, 800-900 text tokens: 96.8% at 8.5x</title><circle class="s2 ring" cx="192.5" cy="54.7" r="5"/></g><g class="mark"><title>100 vision tokens, 900-1000 text tokens: 96.8% at 9.7x</title><circle class="s2 ring" cx="234.5" cy="54.7" r="5"/></g><g class="mark"><title>100 vision tokens, 1000-1100 text tokens: 91.5% at 10.6x</title><circle class="s2 ring" cx="266.0" cy="79.1" r="5"/></g><g class="mark"><title>100 vision tokens, 1100-1200 text tokens: 89.8% at 11.3x</title><circle class="s2 ring" cx="290.5" cy="86.9" r="5"/></g><g class="mark"><title>100 vision tokens, 1200-1300 text tokens: 87.1% at 12.6x</title><circle class="s2 ring" cx="336.0" cy="99.3" r="5"/></g><circle class="s2 ring" cx="76.0" cy="330.0" r="5"/><text class="t-note" x="88.0" y="334.0" text-anchor="start">100 vision tokens: 98.5% at 6.7x, falling to 87.1% at 12.6x</text><polyline class="l1" points="262.5,56.1 308.0,68.5 357.0,114.5 423.5,104.9 472.5,135.2 514.5,148.6 584.5,228.1"/><g class="mark"><title>64 vision tokens, 600-700 text tokens: 96.5% at 10.5x</title><circle class="s1 ring" cx="262.5" cy="56.1" r="5"/></g><g class="mark"><title>64 vision tokens, 700-800 text tokens: 93.8% at 11.8x</title><circle class="s1 ring" cx="308.0" cy="68.5" r="5"/></g><g class="mark"><title>64 vision tokens, 800-900 text tokens: 83.8% at 13.2x</title><circle class="s1 ring" cx="357.0" cy="114.5" r="5"/></g><g class="mark"><title>64 vision tokens, 900-1000 text tokens: 85.9% at 15.1x</title><circle class="s1 ring" cx="423.5" cy="104.9" r="5"/></g><g class="mark"><title>64 vision tokens, 1000-1100 text tokens: 79.3% at 16.5x</title><circle class="s1 ring" cx="472.5" cy="135.2" r="5"/></g><g class="mark"><title>64 vision tokens, 1100-1200 text tokens: 76.4% at 17.7x</title><circle class="s1 ring" cx="514.5" cy="148.6" r="5"/></g><g class="mark"><title>64 vision tokens, 1200-1300 text tokens: 59.1% at 19.7x</title><circle class="s1 ring" cx="584.5" cy="228.1" r="5"/></g><circle class="s1 ring" cx="76.0" cy="350.0" r="5"/><text class="t-note" x="88.0" y="354.0" text-anchor="start">64 vision tokens: 96.5% at 10.5x, falling to 59.1% at 19.7x</text><text class="t-tick" x="350.0" y="310.0" text-anchor="middle">compression = text tokens on the page / vision tokens</text></svg><figcaption>DeepSeek-OCR precision against compression, from Table 2 of the paper. Every point below 10x is at 96.8% or more. Past 15x, precision with 64 tokens falls quickly, to 59.1% at 19.7x.</figcaption></figure>

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

<figure class="fig"><svg viewBox="0 0 780 502" role="img" aria-label="Map of influence: Coder-v1.5 started DeepSeekMath; DeepSeekMath gave GRPO to V2, R1, V3.2 and V4 and its data loop to Coder-V2; Coder-V2 merged into V2.5; DeepSeekMath-V2 fed V3.2-Speciale and the V4 formal-maths pipeline; reward-model work, the vision line and DSpark connect to V4."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">What flowed from the side models into the main line</text><text class="t-label" x="20.0" y="46.0" text-anchor="start">Specialised work</text><text class="t-label" x="470.0" y="46.0" text-anchor="start">Main models</text><rect class="box-1" x="20.0" y="58.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="79.0" text-anchor="start">Coder-v1.5 7B</text><rect class="box-1" x="20.0" y="98.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="119.0" text-anchor="start">DeepSeekMath 7B (GRPO, data loop)</text><rect class="box-1" x="20.0" y="138.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="159.0" text-anchor="start">Coder-V2</text><rect class="box" x="20.0" y="178.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="199.0" text-anchor="start">mHC (residual research)</text><rect class="box-2" x="20.0" y="218.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="239.0" text-anchor="start">DeepSeekMath-V2 (self-verification)</text><rect class="box-3" x="20.0" y="298.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="319.0" text-anchor="start">GRM / SPCT reward models</text><rect class="box-3" x="20.0" y="338.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="359.0" text-anchor="start">VL, VL2, Janus, OCR</text><rect class="box-3" x="20.0" y="378.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="399.0" text-anchor="start">DSpark (speculative decoding)</text><rect class="box" x="20.0" y="418.0" width="300.0" height="32.0" rx="6"/><text class="t-note" x="32.0" y="439.0" text-anchor="start">Engram (conditional memory)</text><rect class="box" x="470.0" y="58.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="79.0" text-anchor="start">DeepSeekMath 7B start</text><rect class="box" x="470.0" y="98.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="119.0" text-anchor="start">V2-Chat, Coder-V2, R1, V4 (GRPO)</text><rect class="box" x="470.0" y="138.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="159.0" text-anchor="start">V2.5 (merge, Sept 2024)</text><rect class="box" x="470.0" y="178.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="199.0" text-anchor="start">V4 residual stream (mHC)</text><rect class="box" x="470.0" y="218.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="239.0" text-anchor="start">V3.2-Speciale</text><rect class="box" x="470.0" y="258.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="279.0" text-anchor="start">V4 formal-maths pipeline</text><rect class="box" x="470.0" y="298.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="319.0" text-anchor="start">V4 post-training GRM</text><rect class="box" x="470.0" y="338.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="359.0" text-anchor="start">V4-Flash-Vision-Exp</text><rect class="box" x="470.0" y="378.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="399.0" text-anchor="start">V4 serving, V4-*-DSpark</text><rect class="box" x="470.0" y="418.0" width="290.0" height="32.0" rx="6"/><text class="t-note" x="482.0" y="439.0" text-anchor="start">V4.1 (Engram memory)</text><line class="edge" x1="320" y1="74" x2="466" y2="74" marker-end="url(#ah)"/><line class="edge" x1="320" y1="114" x2="466" y2="114" marker-end="url(#ah)"/><line class="edge" x1="320" y1="154" x2="466" y2="154" marker-end="url(#ah)"/><line class="edge" x1="320" y1="194" x2="466" y2="194" marker-end="url(#ah)"/><line class="edge" x1="320" y1="234" x2="466" y2="234" marker-end="url(#ah)"/><line class="edge" x1="320" y1="234" x2="466" y2="274" marker-end="url(#ah)"/><line class="edge" x1="320" y1="314" x2="466" y2="314" stroke-dasharray="5 4" marker-end="url(#ah)"/><line class="edge" x1="320" y1="354" x2="466" y2="354" stroke-dasharray="5 4" marker-end="url(#ah)"/><line class="edge" x1="320" y1="394" x2="466" y2="394" marker-end="url(#ah)"/><line class="edge" x1="320" y1="434" x2="466" y2="434" marker-end="url(#ah)"/><text class="t-tick" x="20.0" y="472.0" text-anchor="start">Solid: stated in a DeepSeek paper or model card. Dashed: our inference, or reuse not confirmed.</text><text class="t-tick" x="20.0" y="490.0" text-anchor="start">Not drawn: the DeepSeekMath data loop also built the Coder-V2 corpus; V3 and V3.2 were the bases of Prover-V2 and Math-V2.</text></svg><figcaption>What the side models gave the main line. Solid arrows are stated in a DeepSeek paper or model card; dashed arrows are our inference or a reuse that is not confirmed.</figcaption></figure>

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
