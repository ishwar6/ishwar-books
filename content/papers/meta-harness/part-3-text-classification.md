---
title: "Text Classification: What the Search Found"
description: "The first test of Meta-Harness: a model sorts short texts into classes, and the search writes the code that picks its examples. We read the tables, the accuracy and context trade-off, and the two discovered harnesses, then rebuild both around a small local model."
part: 3
covers: "§4.1, Tables 2 to 5, Figures 3 to 7, Table 9, Appendix B.1"
date: 2026-10-18
tags: [meta-harness, harness-engineering, text-classification, retrieval, llm, pareto]
---

A patient writes "I have a burning sensation in my stomach, and it is worse when I lie down". Which disease is it? A language model can guess, but its guess depends a lot on which solved cases you show it first. Section 4.1 lets Meta-Harness write the code that picks those cases. Each piece below shows the exact lines of the paper, then explains them. At the end, we rebuild the two harnesses it found and run them on a laptop.

## The task: online text classification {§4.1}

> [!PAPER] · §4.1
> [![The start of Section 4 and of Section 4.1, with Human-designed strategies, Program-search methods, receives labeled examples one at a time, updates its memory and GPT-OSS-120B highlighted](/img/papers/meta-harness/p3-setup.png)](/img/papers/meta-harness/p3-setup.png)
>
> **What it says:** each domain compares two kinds of rivals: "Human-designed strategies" and "Program-search methods". In text classification, "an LLM receives labeled examples one at a time, updates its memory, and is evaluated on a held-out test set". The classifier is "GPT-OSS-120B".
>
> **What it means:**
>
> Think of a new nurse at a clinic. Each day a few patients arrive, and a senior doctor tells her the right diagnosis for each one. She keeps a notebook of these cases. Later she must name the disease for new patients alone, and she may look in her notebook. That is the task here. The model is the nurse, and the harness is her way of using the notebook.
>
> The model itself never changes. GPT-OSS-120B is an open-weight model with about 120 billion parameters, and its weights stay frozen. Only the harness changes. It decides what goes into the notebook, which cases come out for a new patient, and how the prompt puts them together.
>
> The two kinds of rivals answer two different questions. The human-designed harnesses ask "is the search better than the best people?". The program-search methods ask "is this search better than other searches?". The sections below take them one at a time.

> [!DEFINITION] Text classification
> A task where a model reads a piece of text and picks one name from a fixed list. Each name is a **class**, and the right class of a text is its **label**. Sorting e-mails into "spam" and "not spam" is text classification with two classes.

> [!DEFINITION] Online classification
> The labeled examples do not come as one big set before the test. They arrive one at a time, and the system may store each one in a **memory** before the next one arrives. The model is not trained on them; the harness only keeps them and shows some of them to the model later.

<figure class="fig"><svg viewBox="0 0 660 375" role="img" aria-label="Online text classification. Labeled examples arrive one at a time and go into a memory. For a new description, the harness decides what to take from the memory and what prompt to send to a frozen model, which returns one label."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">labeled examples arrive one at a time</text><rect class="box-2" x="20.0" y="40.0" width="175.0" height="50.0" rx="10"/><text class="t-tick" x="30.0" y="60.0" text-anchor="start">"burning in my stomach..."</text><text class="t-note" x="30.0" y="80.0" text-anchor="start">-&gt; peptic ulcer</text><rect class="box-2" x="210.0" y="40.0" width="175.0" height="50.0" rx="10"/><text class="t-tick" x="220.0" y="60.0" text-anchor="start">"itchy red rash..."</text><text class="t-note" x="220.0" y="80.0" text-anchor="start">-&gt; fungal infection</text><rect class="box-2" x="400.0" y="40.0" width="175.0" height="50.0" rx="10"/><text class="t-tick" x="410.0" y="60.0" text-anchor="start">"high fever, chills..."</text><text class="t-note" x="410.0" y="80.0" text-anchor="start">-&gt; malaria</text><line class="edge" x1="300.0" y1="92.0" x2="300.0" y2="128.0" marker-end="url(#ah)"/><rect class="box" x="150.0" y="130.0" width="300.0" height="46.0" rx="10"/><text class="t-note" x="300.0" y="158.0" text-anchor="middle">memory: every labeled example so far</text><rect class="box-3" x="20.0" y="220.0" width="170.0" height="60.0" rx="10"/><text class="t-note" x="105.0" y="245.0" text-anchor="middle">new description</text><text class="t-tick" x="105.0" y="265.0" text-anchor="middle">(no label)</text><rect class="box-1" x="240.0" y="210.0" width="220.0" height="80.0" rx="10"/><text class="t-title" x="350.0" y="238.0" text-anchor="middle">HARNESS (the code)</text><text class="t-tick" x="350.0" y="258.0" text-anchor="middle">what to retrieve, what to show,</text><text class="t-tick" x="350.0" y="276.0" text-anchor="middle">how many calls</text><line class="edge" x1="190.0" y1="250.0" x2="238.0" y2="250.0" marker-end="url(#ah)"/><line class="edge" x1="320.0" y1="178.0" x2="320.0" y2="208.0" marker-end="url(#ah)"/><rect class="box" x="510.0" y="220.0" width="130.0" height="60.0" rx="10"/><text class="t-note" x="575.0" y="245.0" text-anchor="middle">frozen model</text><text class="t-tick" x="575.0" y="265.0" text-anchor="middle">GPT-OSS-120B</text><line class="edge" x1="462.0" y1="240.0" x2="508.0" y2="240.0" marker-end="url(#ah)"/><line class="edge" x1="508.0" y1="262.0" x2="462.0" y2="262.0" marker-end="url(#ah)"/><rect class="box-2" x="510.0" y="320.0" width="130.0" height="40.0" rx="10"/><text class="t-note" x="575.0" y="345.0" text-anchor="middle">one label</text><line class="edge" x1="400.0" y1="292.0" x2="520.0" y2="318.0" marker-end="url(#ah)"/><text class="t-note" x="20.0" y="330.0" text-anchor="start">Meta-Harness rewrites only the HARNESS box.</text><text class="t-tick" x="20.0" y="350.0" text-anchor="start">The model and the data stay the same.</text></svg><figcaption>The online setup. Labeled examples go into a memory. For a new description, the harness chooses what to take from the memory and what prompt to send. The model stays frozen; the search changes only the harness.</figcaption></figure>

> [!PAPER] · §4.1
> [![The paragraph about the three datasets, with LawBench, Symptom2Disease, USPTO-50k, the class counts, zero-shot, few-shot, ACE, and MCE and producing 40 candidate harnesses highlighted](/img/papers/meta-harness/p3-datasets.png)](/img/papers/meta-harness/p3-datasets.png)
>
> **What it says:** three datasets, "chosen for difficulty and domain diversity": LawBench (criminal charges from case descriptions, 215 classes), Symptom2Disease (diseases from symptom descriptions, 22 classes) and USPTO-50k (precursor reactants from product molecules, 180 classes). The search starts from "zero-shot, few-shot, ACE, and MCE", and runs 20 iterations with two candidates each, "producing 40 candidate harnesses".
>
> **What it means:**
>
> The three datasets are far apart on purpose. LawBench is Chinese legal text with 215 possible charges. Symptom2Disease is short English text from patients, with 22 diseases. USPTO-50k is chemistry: the input is a molecule written as a string, and the label is the set of molecules that make it. A harness trick that only works on one kind of text will not help on all three.
>
> The search does not start from nothing. Its first population is four known harnesses, so the proposer can read their code and their failures before it writes anything new. Then it runs 20 rounds, and each round proposes two harnesses: 40 new candidates in total. [Part 2](part-2-the-method.md) shows this loop as Algorithm 1.
>
> We use the middle dataset, Symptom2Disease, for our own runs below. It is public on Hugging Face, it has short texts, and its 22 classes fit in one prompt for a small model.

> [!DEFINITION] Search set and test set
> The **search set** is the data that scores each candidate during the search; the proposer sees these scores and traces. The **test set** is kept apart and is used only once, at the end, to report results. A harness that is good only on the search set has **overfit** to it.

## The baselines and Table 2 {§4.1}

> [!PAPER] · §4.1, Table 4
> [![The paragraph that compares Meta-Harness with ACE and MCE, next to Table 4, with reflective memory curation, a library of natural-language skills, reaches 48.6% accuracy, the context token counts and search set highlighted](/img/papers/meta-harness/p3-sota.png)](/img/papers/meta-harness/p3-sota.png)
> [![Footnote 2, with denotes the best discovered harness highlighted](/img/papers/meta-harness/p3-footnote2.png)](/img/papers/meta-harness/p3-footnote2.png)
>
> **What it says:** ACE "uses reflective memory curation to build context over time"; MCE "maintains and evolves a library of natural-language skills". The paper also runs zero-shot and few-shot prompts with 4 to all examples. The selected harness "reaches 48.6% accuracy, outperforming ACE by 7.7 points and MCE by 8.6 points", and "uses only 11.4K context tokens, versus 50.8K for ACE and 28.5K for MCE". The footnote says that in tables, the name Meta-Harness "denotes the best discovered harness".
>
> **What it means:**
>
> The two hand-built rivals are strong systems from 2025. ACE (Agentic Context Engineering) keeps a "playbook" of lessons: after each example, the model reflects on what went right or wrong and edits the playbook. MCE (Meta Context Engineering) keeps a library of written skills and improves that library over time. Both put a lot of text in front of the model. The [Background](background-what-a-harness-is.md) page says more about them.
>
> The headline is two numbers together. The best found harness is 7.7 points more accurate than ACE, and it shows the model less than a quarter of ACE's extra text. A better score that costs more text would be a weaker claim; this one costs less.
>
> The footnote matters when you read the tables. "Meta-Harness" in a table row is not the search; it is the one harness file that the search picked. In this part, that file is the one Appendix B.1 calls "Label-Primed Query".

> [!DEFINITION] Zero-shot and few-shot
> A **zero-shot** prompt gives the model the task and the input, and no solved examples. A **few-shot** prompt also gives some solved examples (input and right label). "Few-shot (8)" means 8 examples in the prompt.

> [!DEFINITION] Token and context
> A **token** is a piece of text that the model reads as one unit, often a word or part of a word. In English, one token is about four characters. The **context** is all the text in the prompt. The tables count the extra context a harness adds: zero-shot adds none.

> [!PAPER] · Table 2
> [![Table 2 and Figure 3 side by side. Table 2 lists USPTO, S2D and Law accuracy, average accuracy and context for Zero-Shot, Few-Shot (8), Few-Shot (32), Few-Shot (all), MCE, ACE and Meta-Harness, with the definition of Ctx highlighted](/img/papers/meta-harness/p3-table2.png)](/img/papers/meta-harness/p3-table2.png)
>
> **What it says:** "Test-set metrics for all harnesses on the three datasets. Ctx denotes additional input tokens in context (thousands)." Meta-Harness: 14.0 on USPTO, 86.8 on S2D, 45.0 on Law, 48.6 on average, with 11.4 thousand context tokens.
>
> **What it means:**
>
> Read the table one row at a time, from top to bottom. Each row adds more help to the same frozen model. Zero-shot gets 27.4% on average. Few-shot with all examples gets 40.8%, at 12.3 thousand extra tokens. ACE gets 40.9%, but needs 50.8 thousand. Meta-Harness gets 48.6% at 11.4 thousand: more accuracy than any row, and less context than Few-Shot (all).
>
> Now read the columns. On Symptom2Disease, accuracy goes from 63.2% (zero-shot) to 86.8%. On LawBench, it goes from 7.0% to 45.0%, more than six times higher. USPTO stays hard for everyone: the best row is ACE at 16.0%, and Meta-Harness gets 14.0%. The found harness does not win every column; it wins the average.
>
> One detail is easy to miss. The model is the same in every row. So all of these differences come from the code around the model, which is the paper's whole claim in one table.

## Two goals at once: the Pareto frontier {§4.1, Figure 3}

Before Figure 3, we need one idea from economics. It answers the question "which of these options is best?" when there are two goals that pull in different directions.

> [!DEFINITION] Pareto frontier
> Option A **dominates** option B if A is at least as good as B on every goal, and better on at least one. The **Pareto frontier** is the set of options that no other option dominates. For a harness, the goals are high accuracy and little context.

A small example makes it concrete. Take four harnesses, each written as (extra context, accuracy): A = (0, 30%), B = (2,000, 35%), C = (3,000, 33%), D = (5,000, 45%). C costs more than B and is less accurate, so B dominates C, and C is off the frontier. A, B and D each win on something: A is the cheapest, D is the most accurate, and B sits between them. So the frontier is A, B and D. No single winner exists; the frontier is the menu of good trade-offs, and the user picks from it.

> [!PAPER] · Figure 3, §4.1
> [![Figure 3: a scatter plot of test accuracy against additional context in characters. Red stars for the Pareto-optimal Meta-Harness harnesses form a curve above the purple few-shot points, the zero-shot cross and the MCE and ACE points far to the right](/img/papers/meta-harness/p3-figure3.png)](/img/papers/meta-harness/p3-figure3.png)
> [![The Accuracy-Context Tradeoffs paragraph, with joint preference for both accuracy and context cost and smooth accuracy-context Pareto curve in Figure 3 highlighted](/img/papers/meta-harness/p3-tradeoff.png)](/img/papers/meta-harness/p3-tradeoff.png)
>
> **What it says:** Figure 3 is the "Pareto frontier of accuracy vs. context tokens". Because the search writes free-form code, the authors "can express a joint preference for both accuracy and context cost", and the proposer finds harnesses "across a broad range of the frontier, yielding a smooth accuracy–context Pareto curve in Figure 3".
>
> **What it means:**
>
> In the plot, up is better and left is cheaper. The red stars are found harnesses on the frontier, joined by a dashed line. The purple dots are few-shot prompts with more and more examples. At every amount of context, a red star sits above the purple line: for the same cost, the found harnesses are more accurate.
>
> Look at the right edge. MCE and ACE sit at about 115 thousand and 200 thousand characters, past a break in the axis. They are both more expensive and less accurate than the top red stars, so they are dominated.
>
> The paragraph says how this happened. Most optimizers need one number to maximize, so a person must first decide how much one point of accuracy is worth in tokens. Here the proposer gets the scores and a sentence that says both goals matter. It then writes some cheap harnesses and some expensive ones, and the frontier fills in.

> [!PAPER] · Table 9
> [![Table 9: eight Pareto-optimal discovered variants from Draft Verification at 5.4 thousand characters and 40.1% average to Label-Primed Query at 45.5 thousand characters and 48.6%, with the definition of Ctx in characters highlighted](/img/papers/meta-harness/p3-table9.png)](/img/papers/meta-harness/p3-table9.png)
>
> **What it says:** "Pareto-optimal discovered variants from the main text-classification search". The cheapest, Draft Verification, gets 40.1% on average with 5.4 thousand characters. The most accurate, Label-Primed Query, gets 48.6% with 45.5 thousand. "Ctx denotes average additional characters in input context (thousands)."
>
> **What it means:**
>
> These are the red stars of Figure 3, with names. Read down the "Avg" column and the "Ctx" column together: both go up on every row. That is what a frontier looks like as a table. Each step buys some accuracy with more context.
>
> The units changed. Table 2 counts tokens, and Table 9 counts characters. Label-Primed Query has 45.5 thousand characters here and 11.4 thousand tokens in Table 2. That is about four characters per token, the usual rate for English text, so the two tables agree.
>
> The names are worth reading too. "Contrastive", "Cluster Coverage", "Cascade Retrieval" and "Error-Annotated" are all ways to choose which past examples go into the prompt. The search did not find one trick; it found a family of retrieval rules, and the frontier shows their prices.

## Against other optimizers {§4.1, Table 4, Figure 4}

> [!PAPER] · §4.1
> [![The paragraph Comparison vs text optimizers and its three bullets, with same proposer configuration, select candidates solely based on search-set performance, the same budget, independent samples from the seed, evolutionary search over programs with LLM mutation and proposal selection via the PUCT reuse rule highlighted](/img/papers/meta-harness/p3-optimizers.png)](/img/papers/meta-harness/p3-optimizers.png)
>
> **What it says:** for a fair test, every method uses the "same proposer configuration", they "select candidates solely based on search-set performance", and each gets "the same budget of proposal harness evaluations". The rivals are Best-of-N ("independent samples from the seed"), OpenEvolve ("evolutionary search over programs with LLM mutation") and TTT-Discover ("proposal selection via the PUCT reuse rule").
>
> **What it means:**
>
> This is a race with the same car and the same fuel. Every method uses the same strong model to write harnesses. Every method gets the same number of evaluations, because running a harness on the data is the slow and expensive step. Only the loop around the proposer changes: what it sees, and which earlier harness it starts from.
>
> Best-of-N is the control. It asks the proposer N times, each time from the same starting harness, and keeps the best. If a clever search cannot beat this, the search adds nothing. OpenEvolve keeps a database of programs and asks the model to mutate good ones. TTT-Discover uses a rule called PUCT to choose which earlier candidate to build on next.

> [!DEFINITION] Evolutionary search
> A search that keeps a population of candidates, picks good ones as parents, and makes children by changing them (mutation) or mixing them. In OpenEvolve and AlphaEvolve, a language model writes the changes to a parent program.

> [!DEFINITION] PUCT
> A selection rule from game-playing tree search, used in AlphaGo and AlphaZero. It gives each candidate a value: its score, plus a bonus that is large when the candidate has been tried few times. The next step builds on the candidate with the largest value, so the search balances "use what works" with "try what is new".

> [!PAPER] · §4.1, Figure 4
> [![The paragraph that says Meta-Harness matches the best prior text optimizers in 0.1 times the evaluations, with surpasses theirs by more than 10 points, full experience history using a filesystem and smallest-context setting we study highlighted](/img/papers/meta-harness/p3-10x.png)](/img/papers/meta-harness/p3-10x.png)
> [![Figure 4: best search-set accuracy against the number of harness evaluations, 0 to 40, for Meta-Harness, TTT-Discover, Best-of-N, OpenEvolve and GEPA, with ACE, few-shot and zero-shot as dashed lines. The Meta-Harness line rises past 50% within 8 evaluations and ends near 57%](/img/papers/meta-harness/p3-figure4.png)](/img/papers/meta-harness/p3-figure4.png)
>
> **What it says:** Meta-Harness "matches the best prior text optimizers (OpenEvolve, TTT-Discover) in 0.1× the evaluations, and its final accuracy surpasses theirs by more than 10 points". The authors credit "full experience history using a filesystem". They also note that this is the "smallest-context setting we study". In Figure 4, "lines track the best-so-far", and Meta-Harness reaches the others' final accuracy "within the first 4 evaluations".
>
> **What it means:**
>
> Figure 4 is a race chart. The x axis counts harness evaluations, from 0 to 40. Each dot is one candidate's search-set accuracy. Each line is the best score found so far, so a line can only go up or stay flat. All lines start near 33%, where the starting harnesses score.
>
> The red line climbs fast. By about the fourth evaluation it is near where OpenEvolve and TTT-Discover end after 40, which is the "0.1×" claim. It keeps climbing to 56.7%, while the best rival, TTT-Discover, ends at 45.6%. Best-of-N, the control with no search at all, ends at 44.2%, above OpenEvolve (43.3%) and GEPA (40.2%). That says something uncomfortable about the older methods: in this setting their structure helped little over asking many times.
>
> The last sentence of the paragraph is a prediction, not a result. Text classification has the shortest traces of the three domains. If the gap is already this large here, the authors expect it to grow on tasks with longer traces.

Table 4 sits beside the ACE paragraph, in the first screenshot of the Table 2 section. It gives the same race as numbers, on the search set: the median and best harness of each optimizer. Meta-Harness has a median of 50.0 and a best of 56.7. The best rival is TTT-Discover with a best of 45.6, and GEPA is lowest with 40.2.

> [!DEFINITION] Median
> The middle value when you sort a list. For the scores 30, 34, 50, 52 and 57, the median is 50. Half of the candidates are better and half are worse. So the median shows a typical candidate, not only the lucky best one.

## What the proposer must see: Table 3 {§4.1, Table 3}

> [!PAPER] · §4.1
> [![The paragraph that starts To isolate which parts of the proposer interface matter most, with a scores-only condition, LLM-generated summaries but no raw traces and access to execution traces highlighted](/img/papers/meta-harness/p3-ablation.png)](/img/papers/meta-harness/p3-ablation.png)
> [![The rest of the paragraph on page 7, with even its median candidate outperforms the best candidate found under either ablation and summaries do not recover the missing signal highlighted](/img/papers/meta-harness/p3-ablation-2.png)](/img/papers/meta-harness/p3-ablation-2.png)
>
> **What it says:** three proposers see different things: "a scores-only condition", a condition with "LLM-generated summaries but no raw traces", and the full interface "with access to execution traces". With full access, "even its median candidate outperforms the best candidate found under either ablation". The authors conclude that "summaries do not recover the missing signal".
>
> **What it means:**
>
> Imagine you fix cars, and a customer says only "it scored 6 out of 10 on the test drive". That is the scores-only proposer. Now a helper adds a short note, "it seemed slow on hills": that is the summary. Then you get the full recording of the drive, with every gear change and warning light: that is the execution trace. Only the last one lets you find the loose cable.
>
> The result is not close. Scores only gives a median of 34.6 and a best of 41.3. Scores plus summaries gives 34.9 and 38.7, so the summaries did not help, and the best even went down. The full interface gives 50.0 and 56.7. A typical full-access candidate beats the luckiest candidate of the other two.
>
> Why would a summary hurt? A summary is a guess, made in advance, about which details matter. In harness search, the useful detail is often small: one wrong label that keeps coming back, or one retrieved example from the wrong class. A summary writer does not know which detail the proposer will need, so it drops the very thing that would explain the failure.

> [!DEFINITION] Ablation
> An experiment that removes one part of a system and measures the change. If the score drops a lot, that part mattered. The name comes from medicine, where ablation means removing tissue.

> [!DEFINITION] Execution trace
> The full record of a harness run. For each example, it holds the exact prompt, the model's raw answer, the parsed label, the right label and the retrieved examples. Meta-Harness stores one trace per evaluated harness in its filesystem.

> [!PAPER] · Table 3
> [![Table 3: Scores Only, Scores + Summary and Meta-Harness full, with columns for scores, code, summaries and traces, median and best accuracy, and runs above zero-shot: 26, 23 and 39, with the meaning of > ZS highlighted](/img/papers/meta-harness/p3-table3.png)](/img/papers/meta-harness/p3-table3.png)
>
> **What it says:** all three conditions see scores and code; only the full one sees traces. The last column, "> ZS", is the "number of runs whose accuracy exceeded the zero-shot baseline": 26, 23 and 39.
>
> **What it means:**
>
> The last column is a second view of the same story. A candidate that cannot beat zero-shot is worse than doing nothing clever at all. With traces, 39 runs beat zero-shot; without them, only 26 and 23. Traces make the proposer fail less often, not only win bigger.
>
> Look at which boxes are ticked. Every row has the code of the earlier harnesses. So the gain does not come from seeing the code; it comes from seeing what the code did on each example. That is the core design choice of the paper, and this table is its most direct test.

## New datasets: Table 5 {§4.1, Table 5}

> [!PAPER] · §4.1, Table 5
> [![The out-of-distribution paragraph, with entirely new datasets unseen during search, 73.1%, ACE 70.2%, beyond 32 hurts performance in 7/9 tasks and 6/9 datasets highlighted](/img/papers/meta-harness/p3-ood.png)](/img/papers/meta-harness/p3-ood.png)
> [![Table 5: accuracy of zero-shot, few-shot with 8, 32 and all examples, ACE and Meta-Harness on nine new datasets, SciCite, FiNER, Amazon, FPB, GoEmotions, Banking77, AG News, SciTail and TweetEval Hate, with averages 67.0, 68.9, 69.6, 68.2, 70.2 and 73.1](/img/papers/meta-harness/p3-table5.png)](/img/papers/meta-harness/p3-table5.png)
>
> **What it says:** the selected harness is run on nine "entirely new datasets unseen during search". It gets the best average accuracy "(73.1%)", ahead of "ACE (70.2%)". More few-shot examples "beyond 32 hurts performance in 7/9 tasks". Meta-Harness is best on "6/9 datasets". It "outperforms the next best method by 2.9 points".
>
> **What it means:**
>
> This is the overfitting test. The harness was chosen on three datasets. If it only learned tricks for chemistry, law and symptoms, it would fail on bank questions or tweets. It does not fail: it wins on average and on six of nine datasets, with 7.3 thousand extra tokens, less than ACE's 11.7 thousand.
>
> The few-shot rows hide a useful lesson. Few-shot (32) has an average of 69.6%, and Few-shot (all) drops to 68.2%. More examples in the prompt is not always better: past some point, the extra text distracts the model. A good harness must choose examples, not only add them.
>
> The gains are not even. On SciCite, accuracy goes from 40.7% (ACE) to 53.3%. On FiNER, ACE is better (74.0% against 67.0%), and on Banking77, few-shot (32) is better. A general strategy helps on average; it does not win every dataset.

> [!DEFINITION] Out-of-distribution (OOD)
> Data that differs from the data a system was built or tuned on: a new topic, a new style or a new set of classes. A method that does well out of distribution has learned something general, not a trick for one dataset.

## The two harnesses it found {§Appendix B.1}

> [!PAPER] · Appendix B.1
> [![The start of Appendix B.1, with a family of memory-based harnesses, the lowest-context frontier point, the highest-accuracy frontier point and What differs is the control flow highlighted](/img/papers/meta-harness/p3-b1.png)](/img/papers/meta-harness/p3-b1.png)
>
> **What it says:** the search finds "a family of memory-based harnesses". Two are described: Draft Verification, "the lowest-context frontier point", and Label-Primed Query, "the highest-accuracy frontier point". Both keep a memory of past labeled examples; "What differs is the control flow".
>
> **What it means:**
>
> Both harnesses start from the same notebook of solved cases. They differ in how they use it. Draft Verification asks the model twice, with a short prompt each time. Label-Primed Query asks once, with a long, carefully built prompt.
>
> "Control flow" is a programmer's word for the order of steps and the decisions between them: which call comes first, what the second call depends on. The paper's point is that the search changed the program, not only the words of the prompt. A prompt optimizer can rewrite text; it cannot add a second model call that depends on the first answer.

### Retrieval: finding the closest solved cases

Both harnesses need one tool: given a new description, find the most similar descriptions in the memory. The paper's Label-Primed Query uses TF-IDF for this, so we use TF-IDF in both.

> [!DEFINITION] TF-IDF
> A way to turn a text into a list of numbers, one per word of the vocabulary. **TF** (term frequency) counts how often a word appears in the text. **IDF** (inverse document frequency) is large for words that few texts use and small for words that many texts use. Each count is multiplied by the word's IDF, so rare, telling words weigh more than "and" or "the".

> [!DEFINITION] Cosine similarity
> A score for how much two lists of numbers point in the same direction: 1 for the same direction, 0 for nothing in common. For TF-IDF vectors of length 1, it is the dot product: multiply the two lists number by number and add.

The IDF of a word is

$$
\text{idf}(w) = \ln\frac{1 + n}{1 + \text{df}(w)} + 1
$$

Read it from left to right. $$n$$ is the number of texts in the memory. $$\text{df}(w)$$, the document frequency, is the number of texts that contain the word $$w$$. The fraction is large when few texts contain the word. The logarithm $$\ln$$ shrinks large ratios, and the $$+1$$ terms stop a division by zero and keep every weight above zero. This is the exact form that scikit-learn uses by default.

A tiny example: with $$n = 751$$ texts, a word in 1 text gets $$\ln(752 / 2) + 1 = 6.93$$. A word in all 751 texts gets $$\ln(752 / 752) + 1 = 1$$. Our real memory gives these values:

```text
vocabulary: ours 988 words, scikit-learn 988 words
largest difference between the two matrices: 1.11e-16
  word 'and'        in 728 of 751 texts, idf = 1.031
  word 'stomach'    in  69 of 751 texts, idf = 3.374
  word 'burning'    in  25 of 751 texts, idf = 4.365
  word 'heartburn'  in  20 of 751 texts, idf = 4.578
```

"and" is in most texts and gets a weight near 1. "heartburn" is in few texts and gets a weight about four times larger. So two descriptions that share "heartburn" count as much more similar than two that share "and".

**Method 1, from scratch.** The functions below build the vocabulary, count document frequencies, compute the IDF, and make one unit-length row per text. `Memory.top` returns the indices of the `k` most similar memory texts; its `keep` argument lets a harness ask for "the closest texts with this label" or "with any other label".

```python
def tokenize(text):
    return TOKEN.findall(text.lower())


def tfidf_fit(docs):
    """Method 1: TF-IDF from scratch. Returns the vocabulary, the idf weights and the document matrix."""
    vocab = {w: i for i, w in enumerate(sorted({w for d in docs for w in tokenize(d)}))}
    df = np.zeros(len(vocab))                               # in how many documents each word appears
    for d in docs:
        for w in set(tokenize(d)):
            df[vocab[w]] += 1
    n = len(docs)
    idf = np.log((1 + n) / (1 + df)) + 1                    # rare words get a large weight, common words a small one
    return vocab, idf, tfidf_transform(docs, vocab, idf)


def tfidf_transform(docs, vocab, idf):
    X = np.zeros((len(docs), len(vocab)))                  # one row per document, one column per word
    for r, d in enumerate(docs):
        for w in tokenize(d):
            if w in vocab:
                X[r, vocab[w]] += 1                        # tf: how often the word appears in this document
    X *= idf                                               # tf x idf
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(norms == 0, 1, norms)


class Memory:
    """The labeled examples a harness has seen, with TF-IDF retrieval over their texts."""
    def __init__(self, examples):
        self.texts = [t for t, _ in examples]
        self.labels = [y for _, y in examples]
        self.vocab, self.idf, self.X = tfidf_fit(self.texts)

    def sims(self, query):
        q = tfidf_transform([query], self.vocab, self.idf)[0]
        return self.X @ q                                  # cosine similarity of the query with every memory text

    def top(self, query, k, keep=lambda i: True):
        order = np.argsort(-self.sims(query), kind='stable')
        return [int(i) for i in order if keep(i)][:k]
```

**Method 2, the library.** scikit-learn's `TfidfVectorizer` does the same in one line. The script fits both on the 751 memory texts, compares the two matrices, and checks that every test query gets the same five neighbours:

```python
T = Log('part3_tfidf')
vec = TfidfVectorizer()
Xs = vec.fit_transform(mem.texts).toarray()                     # Method 2: the library
cols = [vec.vocabulary_[w] for w in sorted(mem.vocab, key=mem.vocab.get)]
T(f'vocabulary: ours {len(mem.vocab)} words, scikit-learn {len(vec.vocabulary_)} words')
T(f'largest difference between the two matrices: {np.abs(mem.X - Xs[:, cols]).max():.2e}')
for w in ['and', 'stomach', 'burning', 'heartburn']:
    df = sum(w in set(tokenize(t)) for t in mem.texts)
    T(f'  word {w!r:<12} in {df:>3} of {len(mem.texts)} texts, idf = {mem.idf[mem.vocab[w]]:.3f}')
agree = 0
for q, _ in test:
    ours = mem.top(q, 5)
    lib = list(np.argsort(-(Xs @ vec.transform([q]).toarray()[0]), kind='stable')[:5])
    agree += ours == [int(i) for i in lib]
T(f'test queries with the same top-5 neighbours: {agree} of {len(test)}')
```

```text
vocabulary: ours 988 words, scikit-learn 988 words
largest difference between the two matrices: 1.11e-16
  word 'and'        in 728 of 751 texts, idf = 1.031
  word 'stomach'    in  69 of 751 texts, idf = 3.374
  word 'burning'    in  25 of 751 texts, idf = 4.365
  word 'heartburn'  in  20 of 751 texts, idf = 4.578
test queries with the same top-5 neighbours: 212 of 212
query: "I have a burning sensation in my stomach that comes and goes. It's worse when I eat and wh..."
  0.673  peptic ulcer disease     "I have a burning sensation in my upper stomach that comes an..."
  0.566  peptic ulcer disease     "I have a burning feeling in my stomach. It's worse when I ea..."
  0.561  gastroesophageal reflux disease "I have a burning sensation in my chest and throat. It feels ..."
  0.505  gastroesophageal reflux disease "I have a burning sensation in my throat that comes and goes,..."
  0.498  gastroesophageal reflux disease "I get heartburn and indigestion a lot. It's worse when I eat..."
```

The two matrices agree to $$10^{-16}$$, the size of a rounding error, and all 212 test queries get the same top 5. The last lines show the retrieval for our first test description. Its two nearest cases are peptic ulcer, the right answer. The next three are acid reflux (gastroesophageal reflux disease), which has very similar symptoms. A majority vote of the five would pick the wrong disease, three to two. Keep this example in mind.

### Draft Verification

> [!PAPER] · Figure 5, Appendix B.1
> [![Figure 5: a flowchart. Query plus memory goes to Retrieve top-5 similar examples, then Draft call initial label D. D goes to Retrieve confirmers equal to D and challengers not equal to D, then Verification call keep or revise D, then Final label](/img/papers/meta-harness/p3-figure5.png)](/img/papers/meta-harness/p3-figure5.png)
> [![The Draft Verification paragraph and its four bullets, with 5 confirmers with the same label, 5 challengers with different labels, depends on both the query and the draft prediction and falls back to a standard single-call few-shot prompt highlighted](/img/papers/meta-harness/p3-dv.png)](/img/papers/meta-harness/p3-dv.png)
>
> **What it says:** the first call makes a draft label from the 5 most similar examples. The second retrieval fetches "5 confirmers with the same label" and "5 challengers with different labels", and asks the model to keep or revise. "The key discovered behavior is that the second retrieval depends on both the query and the draft prediction". With too few examples, it "falls back to a standard single-call few-shot prompt".
>
> **What it means:**
>
> This is what a careful doctor does. First a quick guess: "it looks like acid reflux". Then a check: "show me the closest cases that really were acid reflux, and the closest cases that were something else". If a case from the second group looks more like this patient, change the guess.
>
> The clever part is the second search. A normal few-shot harness retrieves once, using only the patient's text. Here the second retrieval also uses the draft. The challengers are the cases that are most likely to prove the draft wrong. The model gets exactly the evidence it needs to catch its own mistake.
>
> It is cheap because both prompts are short: 5 examples, then 10. In Table 9 it is the cheapest frontier point, at 5.4 thousand characters. The cold-start rule is a sign of real code. With fewer than 5 examples in memory, the program cannot fill the second prompt, so it uses a plain few-shot prompt.

The authors' public repository ([stanford-iris-lab/meta-harness](https://github.com/stanford-iris-lab/meta-harness), folder `reference_examples/text_classification`) has the task, the data and the baseline harnesses. Every harness there is a class with two methods: `predict(input)` makes a label before the right answer is known, and `learn_from_batch(batch_results)` updates the memory afterwards. The two discovered files, `draft_verification.py` and `label_primed_query_anchored.py`, are not in it. So our version follows the description above and Figure 5, and our prompts and exact retrieval are our own reading. `call` is the model call: it takes a prompt and returns a parsed label.

```python
def shots(mem, idx):
    return '\n\n'.join(f'Description: {mem.texts[i]}\nDisease: {mem.labels[i]}' for i in idx)


def draft_verification(q, mem, labels, call, k=5):
    """Figure 5. Call 1 drafts a label from the 5 nearest examples. Call 2 shows 5 examples that agree with the draft
    (confirmers) and 5 that do not (challengers), and keeps or revises the draft."""
    if len(mem.texts) < k:                                          # cold start: a single few-shot call
        return few_shot(q, mem, labels, call, n=len(mem.texts))
    near = mem.top(q, k)
    p1 = (f'Classify the patient description as one of these diseases: {label_list(labels)}.\n\n'
          f'Similar solved cases:\n\n{shots(mem, near)}\n\nDescription: {q}\n{ASK}')
    draft = call(p1)
    if draft is None:
        return None, [p1]
    confirm = mem.top(q, k, keep=lambda i: mem.labels[i] == draft)  # the second retrieval depends on the draft
    challenge = mem.top(q, k, keep=lambda i: mem.labels[i] != draft)
    p2 = (f'Classify the patient description as one of these diseases: {label_list(labels)}.\n\n'
          f'A first guess was: {draft}.\n\nCases WITH the label {draft}:\n\n{shots(mem, confirm)}\n\n'
          f'Similar cases with OTHER labels:\n\n{shots(mem, challenge)}\n\n'
          f'Description: {q}\nKeep the first guess or change it. {ASK}')
    return call(p2), [p1, p2]
```

The line with `confirm = mem.top(q, k, keep=lambda i: mem.labels[i] == draft)` is the "key discovered behavior": the second retrieval depends on the draft. Here is the second prompt for our first test description, as the script printed it. The draft was acid reflux, which is wrong:

```text
Classify the patient description as one of these diseases: allergy, arthritis, bronchial asthma, cervical spondylosis, chicken pox, common cold, dengue, diabetes, drug reaction, fungal infection, gastroesophageal reflux disease, hypertension, impetigo, jaundice, malaria, migraine, peptic ulcer disease, pneumonia, psoriasis, typhoid, urinary tract infection, varicose veins.

A first guess was: gastroesophageal reflux disease.

Cases WITH the label gastroesophageal reflux disease:

Description: I have a burning sensation in my chest and throat. It feels like acid or food is coming up into my throat. It's worse when I lie down. I also have heartburn and indigestion after I eat, and I sometimes throw up.
Disease: gastroesophageal reflux disease

Description: I have a burning sensation in my throat that comes and goes, and it's worse after I eat. I also have a bitter or sour aftertaste in my mouth sometimes.
Disease: gastroesophageal reflux disease

[... 3 more cases with this label ...]

Similar cases with OTHER labels:

Description: I have a burning sensation in my upper stomach that comes and goes. I also have heartburn, indigestion, and nausea.
Disease: peptic ulcer disease

Description: I have a burning feeling in my stomach. It's worse when I eat or drink. I also get heartburn, indigestion, and nausea.
Disease: peptic ulcer disease

[... 3 more cases with other labels ...]

Description: I have a burning sensation in my stomach that comes and goes. It's worse when I eat and when I lie down. I also have heartburn and indigestion.
Keep the first guess or change it. Answer with the disease name only.
----- model answer: 'gastroesophageal reflux disease'
```

### Label-Primed Query

> [!PAPER] · Figure 6, Appendix B.1
> [![Figure 6: a flowchart. Query plus memory goes to Label primer all valid labels and to TF-IDF retrieval query-anchored pairing. Retrieval feeds a Coverage block best example per label and Contrastive pairs similar examples different labels. All three go into Assemble one prompt, then Final label](/img/papers/meta-harness/p3-figure6.png)](/img/papers/meta-harness/p3-figure6.png)
> [![The Label-Primed Query paragraph and its four bullets, with a label primer listing the valid output labels, one query-relevant example per label, place highly similar examples with different labels side by side and TF-IDF retrieval highlighted](/img/papers/meta-harness/p3-lpq.png)](/img/papers/meta-harness/p3-lpq.png)
>
> **What it says:** one larger call built from three parts: "a label primer listing the valid output labels", a coverage section "with one query-relevant example per label", and contrastive pairs that "place highly similar examples with different labels side by side". Retrieval uses TF-IDF and "a query-anchored pairing rule".
>
> **What it means:**
>
> Think of a student before a multiple-choice exam. First she reads all the possible answers: that is the label primer. Then she looks at one solved question for every answer, chosen to be close to her own question: that is the coverage block. Last, she studies pairs of questions that look almost the same but have different answers: those are the contrastive pairs. The pairs teach her where the line between two answers is.
>
> Each block targets a different failure. Without the primer, the model may answer with a name that is not on the list. Without coverage, a rare class never shows up in the prompt, so the model forgets it exists. Without pairs, the model sees many similar cases but never the small detail that separates two close classes.
>
> This is the harness of the main text, the 48.6% row of Table 2. It is the most expensive frontier point in Table 9, at 45.5 thousand characters. The paper says only "query-anchored pairing rule that chooses contrasting examples from the same local neighborhood", so our pairing rule below is one reading of that sentence.

```python
def label_primed_query(q, mem, labels, call, n_pairs=4, pool=20):
    """Figure 6. One prompt in three blocks: the label primer, one query-relevant example per label (coverage),
    and pairs of close examples with different labels taken from the query's neighbourhood (contrastive pairs)."""
    s = mem.sims(q)
    coverage = []
    for lab in labels:                                              # the best example of every label
        own = [i for i in range(len(mem.texts)) if mem.labels[i] == lab]
        coverage.append(max(own, key=lambda i: s[i]))
    near = [int(i) for i in np.argsort(-s, kind='stable')[:pool]]  # the query's neighbourhood
    pairs, used = [], set()
    for a in near:                                                 # pair each close example with the closest one
        if a in used:                                              # that carries a different label
            continue
        b = next((j for j in near if j not in used and j != a and mem.labels[j] != mem.labels[a]), None)
        if b is not None:
            pairs.append((a, b)); used |= {a, b}
        if len(pairs) == n_pairs:
            break
    pair_text = '\n\n'.join(f'Pair {n + 1}:\n{shots(mem, [a])}\n{shots(mem, [b])}' for n, (a, b) in enumerate(pairs))
    p = (f'Valid labels: {label_list(labels)}.\n\n'
         f'One example for each label:\n\n{shots(mem, coverage)}\n\n'
         f'Close cases with different labels:\n\n{pair_text}\n\n'
         f'Description: {q}\nClassify the description as one of the valid labels. {ASK}')
    return call(p), [p]
```

The coverage loop picks, for every label, the memory example closest to the query. The pairing loop walks down the 20 nearest examples and pairs each with the next close example that has a different label. Here is the prompt for our first test description, cut to its three blocks (the full prompt is 22 coverage examples and 4 pairs):

```text
Valid labels: allergy, arthritis, bronchial asthma, cervical spondylosis, chicken pox, common cold, dengue, diabetes, drug reaction, fungal infection, gastroesophageal reflux disease, hypertension, impetigo, jaundice, malaria, migraine, peptic ulcer disease, pneumonia, psoriasis, typhoid, urinary tract infection, varicose veins.

One example for each label:

Description: I have a sore throat and a lot of sneezing. There are times when the skin around my eyes and my lips swell. I find that once I start, I just cannot stop.
Disease: allergy

Description: I have a lot of pain in my neck and my muscles feel weak. My joints are swollen and I have a lot of stiffness when I walk. Walking is also very uncomfortable.
Disease: arthritis

Description: I have a bad cough and I'm really tired. I have a high fever and my breathing is shallow. When I cough, a lot of saliva comes out.
Disease: bronchial asthma

[... 19 more examples, one per label ...]

Close cases with different labels:

Pair 1:
Description: I have a burning sensation in my upper stomach that comes and goes. I also have heartburn, indigestion, and nausea.
Disease: peptic ulcer disease
Description: I have a burning sensation in my chest and throat. It feels like acid or food is coming up into my throat. It's worse when I lie down. I also have heartburn and indigestion after I eat, and I sometimes throw up.
Disease: gastroesophageal reflux disease

Pair 2:
Description: I have a burning feeling in my stomach. It's worse when I eat or drink. I also get heartburn, indigestion, and nausea.
Disease: peptic ulcer disease
Description: I have a burning sensation in my throat that comes and goes, and it's worse after I eat. I also have a bitter or sour aftertaste in my mouth sometimes.
Disease: gastroesophageal reflux disease

[... 2 more pairs ...]

Description: I have a burning sensation in my stomach that comes and goes. It's worse when I eat and when I lie down. I also have heartburn and indigestion.
Classify the description as one of the valid labels. Answer with the disease name only.
----- model answer: 'gastroesophageal reflux disease'
```

## Our run: the two harnesses on a small model

Now we run everything. The task model is Qwen2.5-1.5B-Instruct, an open model with 1.5 billion parameters, on the laptop's Apple GPU. It stays frozen, as in the paper. The data is Symptom2Disease from Hugging Face (`gretelai/symptom_to_diagnosis`):

```text
Symptom2Disease (gretelai/symptom_to_diagnosis): 22 labels
memory: 751 labeled examples the harness may read
search: 100 held-out training examples (to pick a harness)
test:   212 examples of the official test split, never used to choose anything
training texts dropped because the same text is in the test split: 2
labels: allergy, arthritis, bronchial asthma, cervical spondylosis, chicken pox, common cold, dengue, diabetes, drug reaction, fungal infection, gastroesophageal reflux disease, hypertension, impetigo, jaundice, malaria, migraine, peptic ulcer disease, pneumonia, psoriasis, typhoid, urinary tract infection, varicose veins
example: "I have a burning sensation in my stomach that comes and goes. It's worse when I eat and when I lie down. I also have heartburn and indigestion." -> peptic ulcer disease
```

The memory is the notebook: 751 solved cases that every harness may read. We fill it before the test, which is where the paper's online memory ends up after the last labeled example arrives. The search split is 100 more training cases that play the role of the search set. The test split is never used to choose anything. We do not run a search here (that is [Part 6](part-6-build-it-from-scratch.md)); we only compare fixed harnesses on both splits.

All our harnesses give the model the list of 22 diseases, even zero-shot. A model that does not know the names cannot match them. Even with the list, zero-shot gives 20 answers that match no label (see the results). Because every harness has the list, it is not counted as extra context.

### One model call, two ways

> [!DEFINITION] Greedy decoding
> The model gives a score to every possible next token. Greedy decoding takes the highest-scoring token, adds it to the text, and repeats. It has no randomness, so the same prompt always gives the same answer.

**Method 1** is our own loop: run the model, take the top token, feed it back, and stop at the end-of-turn token. It keeps the past keys and values (`past_key_values`), so each new step reads only one new token. **Method 2** is the library's `model.generate` with sampling turned off. `parse` then maps the answer to one of the 22 labels.

```python
class Model:
    def __init__(self, name=MODEL, dtype='bfloat16'):   # 16-bit numbers: half the memory of float32
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.tok = AutoTokenizer.from_pretrained(name)
        self.lm = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(self.dev).eval()

    def ids(self, user):
        msgs = [{'role': 'user', 'content': user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return self.tok(text, return_tensors='pt').input_ids.to(self.dev)

    def greedy(self, user, max_new=16):
        """Method 1: our own greedy decoding loop. Run the model, take the most likely next token, append, repeat."""
        torch = self.torch
        ids = self.ids(user)
        out, past = [], None
        with torch.no_grad():
            x = ids
            for _ in range(max_new):
                r = self.lm(input_ids=x, past_key_values=past, use_cache=True)
                past = r.past_key_values                   # keep the keys and values, so each step reads one token
                nxt = int(r.logits[0, -1].argmax())        # the most likely next token
                if nxt == self.tok.eos_token_id or nxt == self.tok.convert_tokens_to_ids('<|im_end|>'):
                    break
                out.append(nxt)
                x = torch.tensor([[nxt]], device=self.dev)
        return self.tok.decode(out, skip_special_tokens=True)

    def generate(self, user, max_new=16, repetition_penalty=1.0):
        """Method 2: the library way, model.generate with sampling turned off.
        Qwen's generation_config.json also sets repetition_penalty=1.1; 1.0 turns it off, as in our loop."""
        ids = self.ids(user)
        with self.torch.no_grad():
            g = self.lm.generate(ids, max_new_tokens=max_new, do_sample=False, top_p=None, top_k=None,
                                 temperature=None, repetition_penalty=repetition_penalty,
                                 pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(g[0, ids.shape[1]:], skip_special_tokens=True)


def parse(answer, labels):
    """Map the model's text to one valid label: an exact label inside the answer, else the closest spelling."""
    a = answer.lower().strip()
    hits = [l for l in labels if l in a]
    if hits:
        return max(hits, key=len)
    close = difflib.get_close_matches(a.split('\n')[0], labels, n=1, cutoff=0.6)
    return close[0] if close else None
```

```text
--- model.generate with repetition_penalty=1.1
ours: 'Gastroesophageal reflux disease (GERD)'   generate: 'Gastroesophageal Reflux Disease (GERD)'
ours: 'Gastroesophageal reflux disease (GERD)'   generate: 'Gastroesophageal Reflux Disease (GERD)'
ours: 'Migraine'                                 generate: 'Migraine'
ours: 'Pneumonia'                                generate: 'Pneumonia'
ours: 'psoriasis'                                generate: 'psoriasis'
same text: 3 of 5, same label after parse(): 5 of 5
--- model.generate with repetition_penalty=1.0
ours: 'Gastroesophageal reflux disease (GERD)'   generate: 'Gastroesophageal reflux disease (GERD)'
ours: 'Gastroesophageal reflux disease (GERD)'   generate: 'Gastroesophageal reflux disease (GERD)'
ours: 'Migraine'                                 generate: 'Migraine'
ours: 'Pneumonia'                                generate: 'Pneumonia'
ours: 'psoriasis'                                generate: 'psoriasis'
same text: 5 of 5, same label after parse(): 5 of 5
```

The first block is a surprise: with the library's defaults, two of five answers differ, "reflux disease" against "Reflux Disease". Both methods are greedy, so where does the difference come from? `model.generate` reads Qwen's `generation_config.json`, and that file sets `repetition_penalty` to 1.1. The penalty lowers the score of every token that is already in the text. "reflux disease" in lower case is already in the prompt, so the library pushed the model towards capital letters.

With the penalty set to 1.0, both methods give the same text on all five prompts. The lesson is useful for any harness: a library call can carry settings that you never wrote. Note also that the model answers "Gastroesophageal reflux disease (GERD)", which is not exactly a label. `parse` maps it to "gastroesophageal reflux disease"; an answer that matches no label counts as wrong.

### The results

`run` (in the script) sends every example of a split through a harness. It stores the raw answers and the parsed label, and counts the characters of every prompt. "Extra chars" is the prompt length minus the zero-shot prompt length for the same description, as in the paper's Ctx column. The last row uses no model at all: it takes the most common label among the 5 nearest memory examples.

```text
model: Qwen/Qwen2.5-1.5B-Instruct, greedy decoding, device mps
harness                        search acc  test acc  calls  extra chars  no label
zero-shot                           36.0%     40.6%      1            0        20
few-shot (8)                        34.0%     40.1%      1        1,336         4
draft-verification                  84.0%     86.8%      2        3,553         0
label-primed query                  76.0%     79.7%      1        5,504         0
TF-IDF 5-NN vote (no model)         90.0%     89.6%      0            0         0
check with scikit-learn accuracy_score (label-primed query, test): 79.7%
one standard error of a test accuracy near 80% on 212 examples: about 2.8 points
```

Read the test column first. Zero-shot gets 40.6%, and 20 of its 212 answers match no label at all. Few-shot with 8 random examples gets 40.1%, no better. Eight random examples show at most 8 of the 22 diseases, and they are rarely close to the patient. The two discovered designs more than double the score: 86.8% for draft-verification and 79.7% for label-primed query. Both retrieve close cases from the memory, and that is the step that matters here.

How large is the noise? With 212 test examples, one standard error is about 2.8 points. So the 7.1-point gap between draft-verification and label-primed query is larger than the noise. The 0.5-point gap between zero-shot and few-shot is not.

The last row is the honest surprise. A vote of the 5 nearest cases, with no model at all, gets 89.6% on test and 90.0% on search. It beats every harness that calls the model. On this dataset, the words of a description almost give away the disease, so close cases nearly always share its label. Our reading is that the small model mostly copies its retrieved neighbours, and sometimes copies them badly. The paper does not report a no-model retrieval baseline, so we cannot say whether the same holds for GPT-OSS-120B.

The ranking of the two discovered harnesses is the opposite of the paper's. In Table 9, Label-Primed Query is the better of the two on Symptom2Disease (86.8 against 85.4). Our guess, which we did not test: a 1.5B model cannot use a prompt of 30 examples as well as a 120B model can. Our 86.8% for draft-verification equals the paper's Label-Primed Query number only by chance: the model, the split and the harness are all different.

The per-disease picture shows where retrieval helps. Zero-shot gets 0% on six diseases, among them malaria, peptic ulcer and dengue; label-primed query gets 50% or more on all six. Only bronchial asthma drops, from 90% to 70%.

```text
disease                           n  zero-shot  label-primed
malaria                          10         0%          100%
peptic ulcer disease             10         0%           90%
drug reaction                     8         0%           75%
jaundice                          7        14%           86%
allergy                          10        30%           90%
diabetes                         10        20%           80%
hypertension                     10         0%           60%
fungal infection                  9         0%           56%
dengue                           10         0%           50%
arthritis                        10        60%          100%
pneumonia                        10        50%           90%
varicose veins                   10        60%          100%
typhoid                           9        11%           44%
chicken pox                      10        20%           50%
common cold                      10        60%           80%
psoriasis                        10        80%          100%
urinary tract infection           9        67%           78%
cervical spondylosis             10        70%           80%
gastroesophageal reflux disease  10        80%           90%
impetigo                         10        80%           90%
migraine                         10        80%           90%
bronchial asthma                 10        90%           70%
```

<figure class="fig"><svg viewBox="0 0 610 564" role="img" aria-label="Test accuracy for each of the 22 diseases, zero-shot against label-primed query, sorted by the gain."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">test accuracy per disease (%)</text><rect class="s3" x="360" y="12" width="12" height="12" rx="2"/><text class="t-tick" x="378.0" y="22.0" text-anchor="start">zero-shot</text><rect class="s1" x="450" y="12" width="12" height="12" rx="2"/><text class="t-tick" x="468.0" y="22.0" text-anchor="start">label-primed query</text><line class="grid" x1="210.0" y1="38" x2="210.0" y2="538"/><text class="t-tick" x="210.0" y="554.0" text-anchor="middle">0</text><line class="grid" x1="390.0" y1="38" x2="390.0" y2="538"/><text class="t-tick" x="390.0" y="554.0" text-anchor="middle">50</text><line class="grid" x1="570.0" y1="38" x2="570.0" y2="538"/><text class="t-tick" x="570.0" y="554.0" text-anchor="middle">100</text><text class="t-tick" x="202.0" y="55.0" text-anchor="end">malaria</text><g class="mark"><title>malaria: zero-shot 0%, label-primed query 100%</title><rect class="s3" x="210" y="44.0" width="0.0" height="8" rx="2"/><rect class="s1" x="210" y="53.0" width="360.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="77.0" text-anchor="end">peptic ulcer disease</text><g class="mark"><title>peptic ulcer disease: zero-shot 0%, label-primed query 90%</title><rect class="s3" x="210" y="66.0" width="0.0" height="8" rx="2"/><rect class="s1" x="210" y="75.0" width="324.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="99.0" text-anchor="end">drug reaction</text><g class="mark"><title>drug reaction: zero-shot 0%, label-primed query 75%</title><rect class="s3" x="210" y="88.0" width="0.0" height="8" rx="2"/><rect class="s1" x="210" y="97.0" width="270.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="121.0" text-anchor="end">jaundice</text><g class="mark"><title>jaundice: zero-shot 14%, label-primed query 86%</title><rect class="s3" x="210" y="110.0" width="51.4" height="8" rx="2"/><rect class="s1" x="210" y="119.0" width="308.6" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="143.0" text-anchor="end">diabetes</text><g class="mark"><title>diabetes: zero-shot 20%, label-primed query 80%</title><rect class="s3" x="210" y="132.0" width="72.0" height="8" rx="2"/><rect class="s1" x="210" y="141.0" width="288.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="165.0" text-anchor="end">hypertension</text><g class="mark"><title>hypertension: zero-shot 0%, label-primed query 60%</title><rect class="s3" x="210" y="154.0" width="0.0" height="8" rx="2"/><rect class="s1" x="210" y="163.0" width="216.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="187.0" text-anchor="end">allergy</text><g class="mark"><title>allergy: zero-shot 30%, label-primed query 90%</title><rect class="s3" x="210" y="176.0" width="108.0" height="8" rx="2"/><rect class="s1" x="210" y="185.0" width="324.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="209.0" text-anchor="end">fungal infection</text><g class="mark"><title>fungal infection: zero-shot 0%, label-primed query 56%</title><rect class="s3" x="210" y="198.0" width="0.0" height="8" rx="2"/><rect class="s1" x="210" y="207.0" width="200.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="231.0" text-anchor="end">dengue</text><g class="mark"><title>dengue: zero-shot 0%, label-primed query 50%</title><rect class="s3" x="210" y="220.0" width="0.0" height="8" rx="2"/><rect class="s1" x="210" y="229.0" width="180.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="253.0" text-anchor="end">pneumonia</text><g class="mark"><title>pneumonia: zero-shot 50%, label-primed query 90%</title><rect class="s3" x="210" y="242.0" width="180.0" height="8" rx="2"/><rect class="s1" x="210" y="251.0" width="324.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="275.0" text-anchor="end">arthritis</text><g class="mark"><title>arthritis: zero-shot 60%, label-primed query 100%</title><rect class="s3" x="210" y="264.0" width="216.0" height="8" rx="2"/><rect class="s1" x="210" y="273.0" width="360.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="297.0" text-anchor="end">varicose veins</text><g class="mark"><title>varicose veins: zero-shot 60%, label-primed query 100%</title><rect class="s3" x="210" y="286.0" width="216.0" height="8" rx="2"/><rect class="s1" x="210" y="295.0" width="360.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="319.0" text-anchor="end">typhoid</text><g class="mark"><title>typhoid: zero-shot 11%, label-primed query 44%</title><rect class="s3" x="210" y="308.0" width="40.0" height="8" rx="2"/><rect class="s1" x="210" y="317.0" width="160.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="341.0" text-anchor="end">chicken pox</text><g class="mark"><title>chicken pox: zero-shot 20%, label-primed query 50%</title><rect class="s3" x="210" y="330.0" width="72.0" height="8" rx="2"/><rect class="s1" x="210" y="339.0" width="180.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="363.0" text-anchor="end">psoriasis</text><g class="mark"><title>psoriasis: zero-shot 80%, label-primed query 100%</title><rect class="s3" x="210" y="352.0" width="288.0" height="8" rx="2"/><rect class="s1" x="210" y="361.0" width="360.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="385.0" text-anchor="end">common cold</text><g class="mark"><title>common cold: zero-shot 60%, label-primed query 80%</title><rect class="s3" x="210" y="374.0" width="216.0" height="8" rx="2"/><rect class="s1" x="210" y="383.0" width="288.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="407.0" text-anchor="end">urinary tract infection</text><g class="mark"><title>urinary tract infection: zero-shot 67%, label-primed query 78%</title><rect class="s3" x="210" y="396.0" width="240.0" height="8" rx="2"/><rect class="s1" x="210" y="405.0" width="280.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="429.0" text-anchor="end">cervical spondylosis</text><g class="mark"><title>cervical spondylosis: zero-shot 70%, label-primed query 80%</title><rect class="s3" x="210" y="418.0" width="252.0" height="8" rx="2"/><rect class="s1" x="210" y="427.0" width="288.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="451.0" text-anchor="end">gastroesophageal reflux disease</text><g class="mark"><title>gastroesophageal reflux disease: zero-shot 80%, label-primed query 90%</title><rect class="s3" x="210" y="440.0" width="288.0" height="8" rx="2"/><rect class="s1" x="210" y="449.0" width="324.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="473.0" text-anchor="end">migraine</text><g class="mark"><title>migraine: zero-shot 80%, label-primed query 90%</title><rect class="s3" x="210" y="462.0" width="288.0" height="8" rx="2"/><rect class="s1" x="210" y="471.0" width="324.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="495.0" text-anchor="end">impetigo</text><g class="mark"><title>impetigo: zero-shot 80%, label-primed query 90%</title><rect class="s3" x="210" y="484.0" width="288.0" height="8" rx="2"/><rect class="s1" x="210" y="493.0" width="324.0" height="8" rx="2"/></g><text class="t-tick" x="202.0" y="517.0" text-anchor="end">bronchial asthma</text><g class="mark"><title>bronchial asthma: zero-shot 90%, label-primed query 70%</title><rect class="s3" x="210" y="506.0" width="324.0" height="8" rx="2"/><rect class="s1" x="210" y="515.0" width="252.0" height="8" rx="2"/></g></svg><figcaption>Test accuracy for each disease, zero-shot (grey) against label-primed query (teal), sorted by the gain. Our run, Qwen2.5-1.5B-Instruct, 212 test examples, about 10 per disease.</figcaption></figure>

### What the verification call changed

```text
search: draft accuracy 85.0%, final 84.0%; kept 99 of 100, changed 1: fixed 0, broke 1
test: draft accuracy 87.7%, final 86.8%; kept 210 of 212, changed 2: fixed 0, broke 2
broke: "I've been scratching myself a lot lately, and my skin is covered in rashy places. I also have a few ..."
  draft: fungal infection   final: psoriasis   true: fungal infection
```

The verification call almost never changes anything. On the test split it kept the draft 210 times of 212. Both changes went from a right answer to a wrong one, so the draft alone (87.7%) is better than the final answer (86.8%). The search split shows the same: one change, from right to wrong.

The prompt above shows why this is a model problem, not a retrieval problem. The challengers are exactly the right evidence: the closest cases with another label are peptic ulcer cases, and peptic ulcer is the right answer. A model that can weigh evidence would switch. Our 1.5B model repeats its first answer. Draft-verification is a design for a model that can check its own work. A small model does not get that benefit, but still pays for a second call.

### Our Pareto frontier, two ways

**Method 1** sorts the harnesses by context, cheapest first, and keeps a harness only if it is more accurate than every cheaper one. **Method 2** is the definition itself: drop a harness if another one is at least as good on both goals and better on one. Both must give the same frontier.

```python
def pareto_sweep(points):
    """Method 1: sort by context (small first); a point is on the frontier if it beats every cheaper point."""
    best, front = -1.0, []
    for name, ctx, acc in sorted(points, key=lambda p: (p[1], -p[2])):
        if acc > best:
            front.append(name); best = acc
    return front


def pareto_brute(points):
    """Method 2: the definition itself. Drop a point if another point is at least as good on both and better on one."""
    def dominates(p, q):
        return p[1] <= q[1] and p[2] >= q[2] and (p[1] < q[1] or p[2] > q[2])
    return [q[0] for q in points if not any(dominates(p, q) for p in points)]
```

```text
zero-shot              extra context       0 chars   test accuracy 40.6%
few-shot (8)           extra context   1,336 chars   test accuracy 40.1%
draft-verification     extra context   3,553 chars   test accuracy 86.8%
label-primed query     extra context   5,504 chars   test accuracy 79.7%
Method 1 (sweep):       frontier = ['zero-shot', 'draft-verification']
Method 2 (definition):  frontier = ['zero-shot', 'draft-verification']
same frontier: True
```

<figure class="fig"><svg viewBox="0 0 680 380" role="img" aria-label="Our four harnesses on the Symptom2Disease test split: extra prompt characters on the x axis, test accuracy on the y axis. The dashed line joins the points on the Pareto frontier."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="326.0" x2="650" y2="326.0"/><text class="t-tick" x="56.0" y="330.0" text-anchor="end">30</text><line class="grid" x1="64" y1="275.7" x2="650" y2="275.7"/><text class="t-tick" x="56.0" y="279.7" text-anchor="end">40</text><line class="grid" x1="64" y1="225.3" x2="650" y2="225.3"/><text class="t-tick" x="56.0" y="229.3" text-anchor="end">50</text><line class="grid" x1="64" y1="175.0" x2="650" y2="175.0"/><text class="t-tick" x="56.0" y="179.0" text-anchor="end">60</text><line class="grid" x1="64" y1="124.7" x2="650" y2="124.7"/><text class="t-tick" x="56.0" y="128.7" text-anchor="end">70</text><line class="grid" x1="64" y1="74.3" x2="650" y2="74.3"/><text class="t-tick" x="56.0" y="78.3" text-anchor="end">80</text><line class="grid" x1="64" y1="24.0" x2="650" y2="24.0"/><text class="t-tick" x="56.0" y="28.0" text-anchor="end">90</text><text class="t-tick" x="64.0" y="346.0" text-anchor="middle">0</text><text class="t-tick" x="161.7" y="346.0" text-anchor="middle">1,000</text><text class="t-tick" x="259.3" y="346.0" text-anchor="middle">2,000</text><text class="t-tick" x="357.0" y="346.0" text-anchor="middle">3,000</text><text class="t-tick" x="454.7" y="346.0" text-anchor="middle">4,000</text><text class="t-tick" x="552.3" y="346.0" text-anchor="middle">5,000</text><text class="t-tick" x="650.0" y="346.0" text-anchor="middle">6,000</text><line class="axis" x1="64" y1="326" x2="650" y2="326"/><polyline class="l1" points="64.0,272.8 411.0,40.1" style="stroke-dasharray:6 4"/><g class="mark"><title>zero-shot: 0.0, 40.6</title><circle class="s1 ring" cx="64.0" cy="272.8" r="7"/></g><text class="t-note" x="76.0" y="262.8" text-anchor="start">zero-shot (40.6)</text><g class="mark"><title>few-shot (8): 1,336.0, 40.1</title><circle class="s3 ring" cx="194.5" cy="275.2" r="7"/></g><text class="t-note" x="204.5" y="297.2" text-anchor="start">few-shot (8) (40.1)</text><g class="mark"><title>draft-verification: 3,552.7, 86.8</title><circle class="s1 ring" cx="411.0" cy="40.1" r="7"/></g><text class="t-note" x="411.0" y="26.1" text-anchor="middle">draft-verification (86.8)</text><g class="mark"><title>label-primed query: 5,503.6, 79.7</title><circle class="s3 ring" cx="601.5" cy="75.8" r="7"/></g><text class="t-note" x="591.5" y="99.8" text-anchor="end">label-primed query (79.7)</text><text class="t-tick" x="357.0" y="368.0" text-anchor="middle">extra characters in the prompt, compared with zero-shot (average per test example)</text><text class="t-tick" x="14.0" y="16.0" text-anchor="start">test accuracy (%)</text></svg><figcaption>Our four model harnesses on the Symptom2Disease test split. Right means more extra prompt characters; up means higher test accuracy. The dashed line joins the Pareto frontier.</figcaption></figure>

Both methods give the same frontier: zero-shot and draft-verification. Draft-verification dominates label-primed query: it uses less context and is more accurate. Zero-shot dominates few-shot (8), although the 0.5-point gap is inside the noise. The 5-NN vote is left out because it never calls the model. With no extra prompt and 89.6%, it would dominate every point on this chart.

[![Terminal output of mh_part3.py: the evaluation table of the five harnesses on the search and test splits](/img/papers/meta-harness/part3_eval-run.png)](/img/papers/meta-harness/part3_eval-run.png)

### What a small model cannot show

Our run checks the two harness designs, not the paper's claims. Several things are out of its reach:

- **The model.** Qwen2.5-1.5B is about 80 times smaller than GPT-OSS-120B. The verification step needs a model that can change its mind on evidence, and ours does not.
- **The data.** We use one of the three datasets, from a public copy on Hugging Face. The authors' repository ships its own split, which we did not use. LawBench and USPTO are much harder and have 180 to 215 classes.
- **The search.** We wrote both harnesses by hand from the paper's description. No proposer read traces, so Table 3 (traces against summaries) and Table 4 (other optimizers) cannot be tested here. [Part 6](part-6-build-it-from-scratch.md) builds a small search loop.
- **The online memory.** Our memory is full before the test. The paper's harnesses can also learn from their own mistakes as examples arrive; ours only store and retrieve.
- **The noise.** 212 test examples give a standard error of about 2.8 points, so gaps smaller than about 5 points can be chance.

## Search set against test set: Figure 7 {§Appendix B.1, Figure 7}

> [!PAPER] · Figure 7
> [![Figure 7: three scatter plots, LawBench, Symptom2Disease and USPTO, of test accuracy against search-set accuracy, with pink dots for discovered strategies, labeled baselines and a dashed diagonal y = x](/img/papers/meta-harness/p3-figure7.png)](/img/papers/meta-harness/p3-figure7.png)
>
> **What it says:** "Search-set vs. test accuracy per dataset for discovered text-classification strategies. Each pink dot is a discovered strategy; baselines are labeled. The dashed diagonal is y=x."
>
> **What it means:**
>
> Each dot is one harness, placed by two scores: search-set accuracy across, test accuracy up. A dot on the diagonal scores the same on both. A dot far below the diagonal did well on the search set but not on new data: that is the picture of overfitting.
>
> In the middle panel, Symptom2Disease, most pink dots sit above the diagonal. Most found harnesses score between about 75 and 90 on the search set and between about 85 and 90 on the test set. They did not lose accuracy on new data. In LawBench and USPTO, the dots spread more widely on both sides of the line.
>
> The trend matters as much as the place. In LawBench, the cloud of dots climbs along the diagonal: a better search score mostly means a better test score. That is what makes search possible at all. If the two scores did not move together, picking by search score would be picking at random. [Part 5](part-5-limits-and-what-came-after.md) returns to this figure and to overfitting in code space.

<figure class="fig"><svg viewBox="0 0 680 380" role="img" aria-label="Search-set accuracy against test accuracy for our five harnesses. Points on the dashed line y = x score the same on both sets."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="326.0" x2="650" y2="326.0"/><text class="t-tick" x="56.0" y="330.0" text-anchor="end">30</text><line class="grid" x1="64" y1="282.9" x2="650" y2="282.9"/><text class="t-tick" x="56.0" y="286.9" text-anchor="end">40</text><line class="grid" x1="64" y1="239.7" x2="650" y2="239.7"/><text class="t-tick" x="56.0" y="243.7" text-anchor="end">50</text><line class="grid" x1="64" y1="196.6" x2="650" y2="196.6"/><text class="t-tick" x="56.0" y="200.6" text-anchor="end">60</text><line class="grid" x1="64" y1="153.4" x2="650" y2="153.4"/><text class="t-tick" x="56.0" y="157.4" text-anchor="end">70</text><line class="grid" x1="64" y1="110.3" x2="650" y2="110.3"/><text class="t-tick" x="56.0" y="114.3" text-anchor="end">80</text><line class="grid" x1="64" y1="67.1" x2="650" y2="67.1"/><text class="t-tick" x="56.0" y="71.1" text-anchor="end">90</text><line class="grid" x1="64" y1="24.0" x2="650" y2="24.0"/><text class="t-tick" x="56.0" y="28.0" text-anchor="end">100</text><text class="t-tick" x="64.0" y="346.0" text-anchor="middle">30</text><text class="t-tick" x="147.7" y="346.0" text-anchor="middle">40</text><text class="t-tick" x="231.4" y="346.0" text-anchor="middle">50</text><text class="t-tick" x="315.1" y="346.0" text-anchor="middle">60</text><text class="t-tick" x="398.9" y="346.0" text-anchor="middle">70</text><text class="t-tick" x="482.6" y="346.0" text-anchor="middle">80</text><text class="t-tick" x="566.3" y="346.0" text-anchor="middle">90</text><text class="t-tick" x="650.0" y="346.0" text-anchor="middle">100</text><line class="axis" x1="64" y1="326" x2="650" y2="326"/><line class="base-line" x1="64.0" y1="326.0" x2="650.0" y2="24.0"/><text class="t-tick" x="644.0" y="40.0" text-anchor="end">y = x</text><g class="mark"><title>zero-shot: 36, 40.6</title><circle class="s1 ring" cx="114.2" cy="280.4" r="7"/></g><text class="t-note" x="126.2" y="272.4" text-anchor="start">zero-shot (40.6)</text><g class="mark"><title>few-shot (8): 34, 40.1</title><circle class="s1 ring" cx="97.5" cy="282.5" r="7"/></g><text class="t-note" x="109.5" y="300.5" text-anchor="start">few-shot (8) (40.1)</text><g class="mark"><title>draft-verification: 84, 86.8</title><circle class="s1 ring" cx="516.1" cy="81.0" r="7"/></g><text class="t-note" x="504.1" y="85.0" text-anchor="end">draft-verification (86.8)</text><g class="mark"><title>label-primed query: 76, 79.7</title><circle class="s1 ring" cx="449.1" cy="111.5" r="7"/></g><text class="t-note" x="437.1" y="103.5" text-anchor="end">label-primed query (79.7)</text><g class="mark"><title>5-NN vote, no model: 90, 89.6</title><circle class="s2 ring" cx="566.3" cy="68.8" r="7"/></g><text class="t-note" x="554.3" y="56.8" text-anchor="end">5-NN vote, no model (89.6)</text><text class="t-tick" x="357.0" y="368.0" text-anchor="middle">search accuracy (%), 100 held-out training examples</text><text class="t-tick" x="14.0" y="16.0" text-anchor="start">test accuracy (%), 212 test examples</text></svg><figcaption>Our five harnesses: search accuracy (100 held-out training examples) across, test accuracy (212 test examples) up. The dashed line is y = x.</figcaption></figure>

Our five harnesses sit on or near the diagonal. Each scores about the same, or a little better, on the test split than on the search split. The order is also the same on both splits: 5-NN vote, then draft-verification, label-primed query, zero-shot and few-shot. That is the property a search needs, because the search set then gives a fair preview of the test set. We did not search, so we could not overfit; in a real search, the gap between the two scores is the first thing to watch.

> [!TAKEAWAYS] Key takeaways
> - In online text classification, labeled examples arrive one at a time, a harness stores them, and a frozen model labels new texts. Meta-Harness changes only the harness.
> - The selected harness gets 48.6% average test accuracy on three datasets, 7.7 points above ACE, with 11.4 thousand context tokens against ACE's 50.8 thousand.
> - With two goals, there is no single best harness. The Pareto frontier is the set of harnesses that no other harness beats on both accuracy and context.
> - With the same proposer and budget, Meta-Harness reaches the final search score of OpenEvolve and TTT-Discover in about 4 evaluations. It ends more than 10 points above them.
> - Raw execution traces are the key input: with only scores or summaries, the best candidate (41.3, 38.7) is worse than the median full-access candidate (50.0).
> - On nine new datasets, the selected harness gets 73.1% on average against 70.2% for ACE. More than 32 few-shot examples hurt on 7 of 9.
> - The search found a family of retrieval harnesses. Draft Verification checks a first guess against confirmers and challengers; Label-Primed Query shows all labels, one close example per label, and contrastive pairs.
> - Our TF-IDF from scratch matches scikit-learn to $$10^{-16}$$, and our greedy loop matches `model.generate` once Qwen's default repetition penalty is turned off.
> - On Symptom2Disease with Qwen2.5-1.5B, draft-verification gets 86.8% and label-primed query 79.7%, against 40.6% for zero-shot. A 5-nearest-neighbour vote with no model gets 89.6%.
> - A small model does not use the verification step: it kept its draft on 210 of 212 test cases. The designs are general; how much they help depends on the model.

## The paper and the code

[Paper (arXiv 2603.28052)](https://arxiv.org/abs/2603.28052) · [Project page](https://yoonholee.com/meta-harness/) · [Our scripts](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness)
