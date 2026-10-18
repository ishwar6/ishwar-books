---
title: "The Big Idea: Let an Agent Write the Harness"
description: "Why the code around a language model can matter as much as the model, and the paper's proposal: a coding agent that reads every earlier attempt and writes a better one. We read the title, Figure 1, the abstract and §1, and we run one small model with four different harnesses."
part: 1
covers: "Title, Figure 1, Abstract, §1"
date: 2026-10-18
tags: [harness-engineering, agents, llm, optimization, coding-agents]
---

A language model answers what it is shown. Someone writes the code that decides what it is shown, and that code can move the score as much as a better model can. This paper asks a coding agent to write that code, after it reads every earlier attempt in full.

Each piece below shows the exact lines of the paper, then explains them.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## The title {§Title}

> [!PAPER] · Title
> [![The title of the paper: Meta-Harness, End-to-End Optimization of Model Harnesses](/img/papers/meta-harness/p1-title.png)](/img/papers/meta-harness/p1-title.png)
>
> **What it says:** "Meta-Harness: End-to-End Optimization of Model Harnesses."
>
> **What it means:**
>
> Think of a horse and its harness. The horse gives the power. The harness decides where that power goes: which way the cart moves, how the load sits, when the horse stops. A strong horse in a bad harness pulls the cart into a ditch. The paper uses the word in the same way. The model gives the power, and the harness is the code that points it at a task.
>
> "Optimization of model harnesses" says what the paper improves. It does not train the model. The model weights stay frozen for the whole paper. Only the code around the model changes.
>
> "End-to-end" says how much of that code is open to change. Earlier methods change one part, such as the text of a prompt. Meta-Harness lets the optimizer rewrite the whole program. That covers what it stores, what it looks up, how it builds the prompt and how it reads the answer. "Meta" says that the optimizer is itself a harness, one level up: it is code around a model that writes code around a model.

> [!DEFINITION] Language model (LLM)
> A neural network that reads text and writes the next words. A large language model (LLM) has billions of numbers inside it, its **weights**. The paper never changes the weights. It treats each model as a fixed box: text in, text out.

> [!DEFINITION] Harness
> The program around a language model that decides what the model sees and what happens with its answer. It can keep a memory of past cases and look up related ones. It builds the prompt, calls tools, and turns the model's text into a final answer. The paper's words: "the code that determines what to store, retrieve, and show to the model".

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="A cycle of five boxes. Store, retrieve and present are code that builds the prompt. The prompt goes into box 4, the model with frozen weights. Its text goes to the read step, which turns it into an answer, and the answer can be stored for later cases."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="10.0" y="10.0" width="700.0" height="280.0" rx="14"/><text class="t-title" x="30.0" y="38.0" text-anchor="start">The harness: the code around the model</text><rect class="box-1" x="40.0" y="62.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="130.0" y="86.0" text-anchor="middle">1. Store</text><text class="t-tick" x="130.0" y="106.0" text-anchor="middle">keep past cases</text><rect class="box-1" x="270.0" y="62.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="360.0" y="86.0" text-anchor="middle">2. Retrieve</text><text class="t-tick" x="360.0" y="106.0" text-anchor="middle">find related ones</text><rect class="box-1" x="500.0" y="62.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="590.0" y="86.0" text-anchor="middle">3. Present</text><text class="t-tick" x="590.0" y="106.0" text-anchor="middle">build the prompt</text><rect class="box-on" x="500.0" y="178.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="590.0" y="202.0" text-anchor="middle">4. The model</text><text class="t-tick" x="590.0" y="222.0" text-anchor="middle">frozen weights</text><rect class="box-1" x="270.0" y="178.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="360.0" y="202.0" text-anchor="middle">5. Read</text><text class="t-tick" x="360.0" y="222.0" text-anchor="middle">text to an answer</text><line class="edge" x1="220.0" y1="91.0" x2="266.0" y2="91.0" marker-end="url(#ah)"/><line class="edge" x1="450.0" y1="91.0" x2="496.0" y2="91.0" marker-end="url(#ah)"/><line class="path" x1="590.0" y1="120.0" x2="590.0" y2="174.0" marker-end="url(#ah-on)"/><line class="path" x1="500.0" y1="207.0" x2="454.0" y2="207.0" marker-end="url(#ah-on)"/><line class="edge" x1="270.0" y1="207.0" x2="132.0" y2="207.0" marker-end="url(#ah)"/><line class="edge" x1="130.0" y1="207.0" x2="130.0" y2="124.0" marker-end="url(#ah)"/><text class="t-tick" x="40.0" y="268.0" text-anchor="start">Only box 4 is the model. Boxes 1, 2, 3 and 5 are code: change them, and the score changes.</text></svg><figcaption>A harness as five steps. Four of them are code that you can change. Only step 4 is the model, and the paper keeps it frozen.</figcaption></figure>

## Figure 1: the results at a glance {§Figure 1}

> [!PAPER] · Figure 1
> [![Figure 1 of the paper: left, the best accuracy so far against the number of harness evaluations for Meta-Harness, TTT-Discover, OpenEvolve, ACE, few-shot and zero-shot; right, TerminalBench-2 pass rates of six harnesses, with Meta-Harness first at 37.6](/img/papers/meta-harness/p1-figure1.png)](/img/papers/meta-harness/p1-figure1.png)
>
> **What it says:** on text classification, Meta-Harness beats the best hand-designed harness (ACE) and the text optimizers TTT-Discover and OpenEvolve, "matching the next-best method's final accuracy after just 4 evaluations". On TerminalBench-2, it "outperforms all reported" harnesses for the model Haiku 4.5.
>
> **What it means:**
>
> The left chart is a race between optimizers. Each one proposes harnesses, and each proposal is run and scored once. That run is one **evaluation**. The line shows the best score found so far. The orange line of Meta-Harness passes the final level of TTT-Discover and OpenEvolve after 4 evaluations, and then keeps going up to about 57%. The flat dashed lines are fixed harnesses that nobody optimizes: zero-shot, few-shot and ACE.
>
> Evaluations are the right unit for the race. Each one means running a language model on many tasks, which costs time and money. An optimizer that needs fewer evaluations is cheaper to use. The left chart shows the accuracy on the tasks that the optimizer uses during its search. The appendix draws the same curves as Figure 4, and [Part 3](part-3-text-classification.md) reads them closely.
>
> The right chart is a different task: an agent works in a computer terminal to finish real jobs. Every bar uses the same model, Haiku 4.5. Five bars are harnesses that people wrote, including Anthropic's own coding agent at 27.5%. The orange bar is the harness that Meta-Harness found, at 37.6%. Same model, and the pass rate goes from 27.5% to 37.6% only because the code around it changed.

> [!DEFINITION] Evaluation
> One complete test of one harness: run it on a fixed set of tasks and compute its score. In this paper, the evaluation is the expensive step, so the paper counts cost in evaluations.

> [!DEFINITION] TerminalBench-2 and pass rate
> A benchmark of 89 hard tasks that an agent must finish on its own in a command-line terminal. Examples are configuring a git web server or assembling a protein sequence. The **pass rate** is the percentage of tasks that the agent finishes correctly. [Part 4](part-4-math-and-terminalbench.md) covers it.

> [!DEFINITION] Zero-shot and few-shot
> Two basic harnesses. **Zero-shot** shows the model only the question. **Few-shot** also shows a few solved examples first, so the model can copy their form. The [Background page](background-what-a-harness-is.md) builds both.

## The abstract {§Abstract}

> [!PAPER] · Abstract
> [![The first sentence of the abstract, with not only on model weights, but also on their harness and what information to store, retrieve, and present to the model highlighted](/img/papers/meta-harness/p1-abstract-1.png)](/img/papers/meta-harness/p1-abstract-1.png)
>
> **What it says:** the performance of LLM systems depends "not only on model weights, but also on their harness": the code that determines "what information to store, retrieve, and present to the model".
>
> **What it means:**
>
> Most news about AI is about weights: a new model, a bigger model, a model trained on more data. This sentence points at the other half of every product. When you ask a chatbot a question, the model does not see only your question. It also sees a system message, maybe some earlier messages, maybe some search results. Code chose all of it.
>
> The three verbs name three jobs. "Store" decides what the system keeps from earlier cases. "Retrieve" decides which stored things to bring back for the case at hand. "Present" decides how to put them into the prompt, and in which order. Each job is a design choice, and each choice changes the answer.

> [!PAPER] · Abstract
> [![The second and third sentences of the abstract, with designed largely by hand, compress feedback too aggressively, they are memoryless, condition only on scalar scores, and short templates or summaries highlighted](/img/papers/meta-harness/p1-abstract-2.png)](/img/papers/meta-harness/p1-abstract-2.png)
>
> **What it says:** harnesses are "designed largely by hand", and existing text optimizers "compress feedback too aggressively: they are memoryless, condition only on scalar scores, or restrict feedback to short templates or summaries".
>
> **What it means:**
>
> Imagine a cooking teacher who tastes your soup and says only "6 out of 10". You know that the soup is not good. You do not know if it needs salt, more time, or a different pot. Now imagine a teacher who lets you read the notes of every soup that the class made, with the recipe, each step, and each score. You can find the pattern yourself.
>
> The paper names three ways in which earlier optimizers act like the first teacher. A **memoryless** optimizer sees only the current attempt and forgets the ones before it. An optimizer that sees only **scalar scores** gets one number per attempt. An optimizer that sees **summaries** gets a short text that someone, or some model, wrote about the attempt. In each case, most of what happened during the run is lost before the optimizer can look at it.

> [!DEFINITION] Text optimizer
> A program that improves a piece of text, such as a prompt, by trial and error. It proposes a new version, scores it, and uses the result to propose the next version. Usually a language model writes the proposals. The [Background page](background-what-a-harness-is.md) describes the main ones.

> [!DEFINITION] Feedback
> What the optimizer learns about an attempt, which it uses for the next attempt. It can be one number (a score), a short text (a summary), or the full record of the run.

> [!PAPER] · Abstract
> [![The fourth and fifth sentences of the abstract, with outer-loop, searches over harness code, agentic proposer, source code, scores, and execution traces and through a filesystem highlighted](/img/papers/meta-harness/p1-abstract-3.png)](/img/papers/meta-harness/p1-abstract-3.png)
>
> **What it says:** Meta-Harness is "an outer-loop system that searches over harness code". It uses "an agentic proposer that accesses the source code, scores, and execution traces of all prior candidates through a filesystem".
>
> **What it means:**
>
> This is the whole method in two sentences. There are two loops. The **inner loop** is the harness at work: it reads a task, calls the model, and gives an answer. The **outer loop** improves the harness: it proposes a new harness, runs it, and keeps the result. Meta-Harness is the outer loop.
>
> The new part is what the proposer can read. It is not given one number or one summary. It gets a folder that holds, for every earlier harness, the code, the score and the full trace of the run. The proposer is an agent, so it decides which files to open. It can open ten traces of one failed harness, or compare the code of two harnesses line by line.

> [!DEFINITION] Execution trace
> The full record of one run of a harness. It holds every prompt that the harness built, every model answer, every tool call and every change to its memory. A trace shows *how* a harness got its score, not only the score.

> [!DEFINITION] Proposer
> The part of an optimizer that writes the next candidate. In Meta-Harness, the proposer is a coding agent (defined below). The paper's proposer is Anthropic's coding agent running the model Opus 4.6.

<figure class="fig"><svg viewBox="0 0 720 210" role="img" aria-label="A loop of three boxes. The proposer agent reads the filesystem and writes a new harness. The harness is evaluated on search tasks, and its code, scores and traces go into a new folder of the filesystem. Then the loop repeats."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The Meta-Harness loop, in one picture</text><rect class="box-1" x="30.0" y="50.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="120.0" y="74.0" text-anchor="middle">Proposer agent</text><text class="t-tick" x="120.0" y="94.0" text-anchor="middle">reads any files it wants</text><rect class="box-1" x="270.0" y="50.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="360.0" y="74.0" text-anchor="middle">New harness</text><text class="t-tick" x="360.0" y="94.0" text-anchor="middle">one Python file</text><rect class="box-1" x="510.0" y="50.0" width="180.0" height="58.0" rx="8"/><text class="t-tick t-strong" x="600.0" y="74.0" text-anchor="middle">Evaluate</text><text class="t-tick" x="600.0" y="94.0" text-anchor="middle">run it on search tasks</text><line class="path" x1="210.0" y1="79.0" x2="266.0" y2="79.0" marker-end="url(#ah-on)"/><line class="path" x1="450.0" y1="79.0" x2="506.0" y2="79.0" marker-end="url(#ah-on)"/><rect class="box-on" x="150.0" y="140.0" width="420.0" height="50.0" rx="8"/><text class="t-tick t-strong" x="360.0" y="162.0" text-anchor="middle">Filesystem: one folder per harness</text><text class="t-tick" x="360.0" y="181.0" text-anchor="middle">its code, its scores, its full execution traces</text><line class="path" x1="600.0" y1="108.0" x2="520.0" y2="138.0" marker-end="url(#ah-on)"/><line class="path" x1="200.0" y1="138.0" x2="120.0" y2="110.0" marker-end="url(#ah-on)"/></svg><figcaption>The outer loop of Meta-Harness. The proposer reads the filesystem, writes a harness, the harness is scored on search tasks, and everything goes into a new folder. Part 2 reads the paper's own picture of it, Figure 2.</figcaption></figure>

> [!PAPER] · Abstract
> [![The results sentences of the abstract, with by 7.7 points while using 4× fewer context tokens, by 4.7 points on average across five held-out models, surpass the best hand-engineered, and richer access to prior experience highlighted](/img/papers/meta-harness/p1-abstract-4.png)](/img/papers/meta-harness/p1-abstract-4.png)
>
> **What it says:** on text classification, Meta-Harness beats a state-of-the-art system "by 7.7 points while using 4× fewer context tokens". On math, one discovered harness improves accuracy on 200 hard problems "by 4.7 points on average across five held-out models". On TerminalBench-2, discovered harnesses "surpass the best hand-engineered baselines". So "richer access to prior experience can enable automated harness engineering".
>
> **What it means:**
>
> Three tasks, and they are very different. Text classification puts each text into one class. Math needs long reasoning, helped by solved problems that the harness looks up. TerminalBench-2 needs an agent that acts in a terminal for many steps. A method that helps on all three is not a trick for one task.
>
> Two details make the claims stronger. On text classification, the better harness uses *less* context, so the gain does not come from a bigger prompt. On math, the harness was found with one model and then tested on five models, four of which the search never used. A harness that helps models it never saw has learned something general about the task.
>
> The last sentence is the thesis of the paper. The gain does not come from a cleverer search rule. It comes from what the proposer can read.

> [!DEFINITION] Points
> Short for **percentage points**: the plain difference between two percentages. From 40.9% to 48.6% is a gain of 7.7 points. In relative terms it is 18.8% more, but the paper reports points.

> [!DEFINITION] Context tokens
> The tokens that a harness puts into the prompt besides the question itself, such as examples and notes. A **token** is a piece of text, often a word or part of a word. More context costs more time and money on every model call.

> [!DEFINITION] Held-out model
> A model that was not used to build the thing under test. Testing a harness on held-out models checks that it was not fitted to one model. In §4.2, the math harness was selected with GPT-OSS-20B. The paper still counts it among the five "held-out" models. The other four were never used in the search.

## The 6× gap {§1}

> [!PAPER] · §1
> [![The first lines of the introduction, with 6×, performance gap on the same benchmark [47], often matters as much as the model itself, and harness engineering highlighted](/img/papers/meta-harness/p1-intro-6x.png)](/img/papers/meta-harness/p1-intro-6x.png)
>
> **What it says:** "Changing the harness around a fixed large language model (LLM) can produce a 6× performance gap on the same benchmark [47]." The harness "often matters as much as the model itself". This has led to interest in "harness engineering", the practice of refining the code around an LLM.
>
> **What it means:**
>
> The paper opens with its strongest reason to care. Keep the model fixed. Keep the test fixed. Change only the harness, and one version can score six times higher than another. Few model upgrades give a factor of six.
>
> "Harness engineering" is a new name for an old job. People who build products on language models spend much of their time on this code. They decide which documents to retrieve, how to format tool output, and when to stop. The paper cites blog posts from 2025 and 2026 that use the name. The term is new because the job became large.
>
> The number 6× comes from another paper, reference [47]. A claim this strong is worth a look at its source, so the next box opens it.

> [!DEFINITION] Benchmark
> A fixed set of tasks with a fixed way to score answers. Everybody who reports on the same benchmark is measured on the same tasks, so their numbers can be compared.

> [!PAPER] Tian et al. (2026), SWE-Bench Mobile · §3.2
> [![A passage of SWE-Bench Mobile, with achieves 12% on Cursor but only 2% on OpenCode and 6× difference highlighted](/img/papers/meta-harness/p1-swebm-6x.png)](/img/papers/meta-harness/p1-swebm-6x.png)
>
> **What it says:** "the same model (Opus 4.5) achieves 12% on Cursor but only 2% on OpenCode, a 6× difference".
>
> **What it means:**
>
> SWE-Bench Mobile (arXiv 2602.09540) tests coding agents on real tasks from the code of an iPhone app. Cursor and OpenCode are two coding agents, which are two harnesses. Both ran the same model, Opus 4.5. One solved 12% of the tasks and the other solved 2%.
>
> The source also says how many tasks there are: a "fixed denominator of 50 tasks". So 12% is 6 tasks and 2% is 1 task. The ratio is six, but it rests on small counts, and one more task solved by OpenCode would make it three. The same table shows smaller gaps for other models: 3× for Sonnet 4.5 and 1.5× for GLM-4.6. The direction of the claim is solid. The exact factor is the extreme case.

### The same gap, on a laptop

We can see this effect ourselves with a small open model. The model is Qwen2.5-1.5B-Instruct, and its weights never change. The task is Symptom2Disease, one of the paper's three text-classification datasets: read a patient's description of symptoms and name one of 22 diseases. We score 60 fixed cases from its test split.

There is one function that calls the model. Every harness goes through it, so the only thing that can change is what the harness puts in and how it reads what comes out:

```python
def call_model(messages, max_new_tokens=12):
    """The only place where the model runs. A harness decides what goes into `messages`."""
    prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    ids = tok(prompt, return_tensors='pt')['input_ids'].to(DEV)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip(), ids.shape[1]
```

`do_sample=False` means greedy decoding: the model always takes its most likely next token, so a run gives the same output each time. The four harnesses are four small functions. Each returns the messages for the model and the function that reads the answer:

```python
def parse_strict(output):
    """Accept the answer only if the whole output is exactly one label."""
    return norm(output) if norm(output) in LABELS else None


def parse_lenient(output):
    """Accept the first label that occurs anywhere in the output (longest labels first)."""
    o = norm(output)
    for lab in sorted(LABELS, key=len, reverse=True):
        if lab in o:
            return lab
    return None
```

```python
def h_bare(text):
    return [{'role': 'user', 'content': f'Patient: {text}\nWhat is the diagnosis?'}], parse_strict


def h_labels(text):
    return [{'role': 'system', 'content': LABEL_LIST + ' Answer with one diagnosis from the list only.'},
            {'role': 'user', 'content': f'Patient: {text}\nDiagnosis:'}], parse_strict


def h_labels_lenient(text):
    msgs, _ = h_labels(text)
    return msgs, parse_lenient


def h_retrieval(text):
    shots = ''.join(f'Patient: {t}\nDiagnosis: {y}\n\n' for t, y in retrieve(text, K_SHOTS))
    return [{'role': 'system', 'content': LABEL_LIST + ' Answer with one diagnosis from the list only.'},
            {'role': 'user', 'content': f'{shots}Patient: {text}\nDiagnosis:'}], parse_lenient
```

Harness A asks the bare question. Harness B also lists the 22 diseases and asks for one of them. Harness C has the same prompt as B, but its reader accepts a disease name inside a longer answer. Harness D adds the 5 most similar solved cases from the 853 training cases, found by TF-IDF word overlap. This is retrieval, the "retrieve" job of the abstract. The results:

```text
== 1. One frozen model, one task, four harnesses
model: Qwen/Qwen2.5-1.5B-Instruct on mps, greedy decoding, weights never change
task: Symptom2Disease, 22 labels, memory 853 labeled examples, eval 60 test examples
harness               accuracy  right  input tokens
A_bare                   0.000   0/60          72.4
B_labels                 0.433  26/60         145.4
C_labels_lenient         0.517  31/60         145.4
D_labels_retrieval       0.767  46/60         368.0
time for 4 x 60 model calls: 350 s
the first outputs of harness A (12 new tokens at most):
  'Based on your symptoms of skin irritations, rashes,'
  'Based on your symptoms of frequent coughing, difficulty breathing,'
best 0.767, worst 0.000, gap 76.7 points, ratio undefined (worst is 0)
```

<figure class="fig"><svg viewBox="0 0 720 226" role="img" aria-label="Horizontal bars of accuracy for four harnesses around the same frozen model: A: bare question 0.0 percent, B: + label list 43.3 percent, C: + lenient reader 51.7 percent, D: + 5 retrieved cases 76.7 percent."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="40.0" text-anchor="start">Same model, same 60 test cases, four harnesses</text><text class="t-tick" x="212.0" y="71.0" text-anchor="end">A: bare question</text><g class="mark"><title>A: bare question: 0.0%</title><rect class="s1" x="220.0" y="53.0" width="2.0" height="26.0" rx="3"/></g><text class="t-val" x="228.0" y="71.0" text-anchor="start">0.0%</text><text class="t-tick" x="212.0" y="105.0" text-anchor="end">B: + label list</text><g class="mark"><title>B: + label list: 43.3%</title><rect class="s1" x="220.0" y="87.0" width="164.7" height="26.0" rx="3"/></g><text class="t-val" x="390.7" y="105.0" text-anchor="start">43.3%</text><text class="t-tick" x="212.0" y="139.0" text-anchor="end">C: + lenient reader</text><g class="mark"><title>C: + lenient reader: 51.7%</title><rect class="s1" x="220.0" y="121.0" width="196.3" height="26.0" rx="3"/></g><text class="t-val" x="422.3" y="139.0" text-anchor="start">51.7%</text><text class="t-tick t-strong" x="212.0" y="173.0" text-anchor="end">D: + 5 retrieved cases</text><g class="mark"><title>D: + 5 retrieved cases: 76.7%</title><rect class="s1" x="220.0" y="155.0" width="291.3" height="26.0" rx="3"/></g><text class="t-val" x="517.3" y="173.0" text-anchor="start">76.7%</text><text class="t-note" x="20.0" y="210.0" text-anchor="start">Qwen2.5-1.5B-Instruct on Symptom2Disease (22 diseases). Accuracy on 60 fixed test cases.</text></svg><figcaption>Our run: one frozen model, four harnesses, 60 test cases. The accuracy goes from 0% to 76.7%, and the weights never changed.</figcaption></figure>

Harness A scores zero, and the outputs show why. The model starts politely, "Based on your symptoms of...", and the 12-token limit cuts it off before it names a disease. The strict reader then finds no disease name. The model may know the answer; the harness never lets it say it. The ratio of best to worst is not even defined, because the worst is 0.

Each later harness fixes one thing. The label list (B) tells the model what form the answer must have: 26 right. The lenient reader (C) accepts "Urinary tract infection (UTI)" as "urinary tract infection": 31 right. Five retrieved cases (D) show the model how similar patients were labeled: 46 right. D uses about 2.5 times the input tokens of B, which is the price of the examples.

This is the 6× claim at toy scale, and it has the same lesson. Nothing about the model changed. Small code choices, such as an output limit and a reader, decided between 0% and 77%. Be careful with the size of these numbers: 60 cases is a small sample, and one case is 1.7 points.

### What the model really sees, two ways

A harness builds messages, but the model reads one long string. Our first method writes that string by hand in Qwen's chat format. The second method asks the tokenizer's own chat template, which is the library's way. If our understanding is right, the two strings are equal, character for character:

```python
def chatml_by_hand(messages):
    """Qwen2.5's chat format: each turn is <|im_start|>role, newline, text, <|im_end|>, newline."""
    s = ''
    if messages[0]['role'] != 'system':                                  # Qwen adds its default system turn
        s += '<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n'
    for m in messages:
        s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    return s + '<|im_start|>assistant\n'                                  # the model writes from here


ours = chatml_by_hand(msgs)
lib = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
log(f'Method 1 length {len(ours)} chars, Method 2 length {len(lib)} chars, identical: {ours == lib}')
```

```text
== 2. The prompt the model sees: built by hand (Method 1) and by the chat template (Method 2)
Method 1 length 611 chars, Method 2 length 611 chars, identical: True
harness A (no system turn): identical: True
the start of the prompt of harness B, as the model reads it:
  | <|im_start|>system
  | Possible diagnoses: allergy, arthritis, bronchial asthma, cervical spondylosis, chicken pox, common cold, deng ...
  | <|im_start|>user
```

Both methods give the same 611 characters. One detail is a harness choice that nobody wrote: when harness A sends no system message, the template adds Qwen's default one. A harness can put text in front of the model that its author never typed. Only a full trace of the prompt shows it.

We also count the accuracy in two ways, by hand and with scikit-learn's `accuracy_score`. They agree for all four harnesses:

```text
== 3. Accuracy counted by hand (Method 1) and by scikit-learn (Method 2)
A_bare               ours 0.0000  sklearn 0.0000  agree: True
B_labels             ours 0.4333  sklearn 0.4333  agree: True
C_labels_lenient     ours 0.5167  sklearn 0.5167  agree: True
D_labels_retrieval   ours 0.7667  sklearn 0.7667  agree: True
```

## Harness engineering is done by hand {§1}

> [!PAPER] · §1
> [![The end of the first paragraph of the introduction, with adjust heuristics, and iterate on a small number of designs, and whether this process itself can be automated highlighted](/img/papers/meta-harness/p1-intro-manual.png)](/img/papers/meta-harness/p1-intro-manual.png)
>
> **What it says:** harness engineering "remains largely manual": practitioners inspect failures, "adjust heuristics, and iterate on a small number of designs". The paper asks "whether this process itself can be automated".
>
> **What it means:**
>
> Our four harnesses are a small picture of this manual process. We looked at the failures of A, saw cut-off sentences, and added a label list. We looked at the failures of B, saw names with an abbreviation in brackets, and made the reader lenient. Each step was: read the failures, guess the cause, change the code, run again.
>
> A person can do this for a few designs before time runs out. Note what the person reads at each step: the failed cases themselves, not only the score. The question of the paper is whether an agent can take the place of the person in this loop, and do many more rounds.

> [!DEFINITION] Heuristic
> A rule of thumb in code that usually works but has no proof, such as "show 5 examples" or "stop after 30 steps". A harness is full of heuristics, and each one is a choice that an optimizer could change.

## Why earlier optimizers are a poor fit {§1}

> [!PAPER] · §1
> [![The second paragraph of the introduction, with condition only on the current candidate, rely primarily on scalar scores, short templates or LLM-generated summaries, and pragmatic scalability choice highlighted](/img/papers/meta-harness/p1-intro-textopt.png)](/img/papers/meta-harness/p1-intro-textopt.png)
>
> **What it says:** text optimizers are a natural start, but they use "short-horizon or heavily compressed feedback": some "condition only on the current candidate", others "rely primarily on scalar scores", and others restrict feedback to "short templates or LLM-generated summaries". The paper calls this "a pragmatic scalability choice, not evidence that longer-range dependencies are uninformative".
>
> **What it means:**
>
> The three kinds match the three flaws of the abstract. Their references show which methods the authors mean. Self-Refine, OPRO and TextGrad mostly look at the current attempt. AlphaEvolve and AdaEvolve mostly use scores. GEPA and Feedback Descent use templates or summaries. The [Background page](background-what-a-harness-is.md) explains each one, and [Part 2](part-2-the-method.md) reads the paper's Table 1, which compares them.
>
> The last sentence is fair to these methods. They compress the feedback because a model can read only so much text at once, not because the full record is useless. The designers had a good reason. The paper's claim is that this reason no longer holds when the proposer is an agent that reads files one at a time.

> [!DEFINITION] Context window
> The largest amount of text that a model can read in one call, counted in tokens. Text beyond it does not fit. A feedback record larger than the context window cannot go into one prompt, so older methods had to shorten it.

> [!PAPER] · §1
> [![The third paragraph of the introduction, with affect behavior many reasoning steps later, trace downstream failures to earlier harness decisions, ranges from only 100 to 30,000 tokens, and accessed adaptively highlighted](/img/papers/meta-harness/p1-intro-horizon.png)](/img/papers/meta-harness/p1-intro-horizon.png)
>
> **What it says:** "Harnesses act over long horizons": one choice about what to store or retrieve "can affect behavior many reasoning steps later". Compressed feedback removes the information needed to "trace downstream failures to earlier harness decisions". The context per optimization step of earlier optimizers "ranges from only 100 to 30,000 tokens". Useful context should often be "accessed adaptively" instead of packed into one prompt.
>
> **What it means:**
>
> A harness failure often shows up far from its cause. Suppose that a harness stores a wrong label for case 12. Nothing goes wrong at case 12. At case 80, the harness retrieves that stored case as an example, and the model copies the wrong label. The score drops at case 80, but the cause is the storage rule that ran at case 12. Only a trace that holds both moments lets you connect them.
>
> The numbers show the size of the problem. Earlier optimizers see at most 100 to 30,000 tokens per step. A harness run on many tasks writes far more than that. So the old methods must throw most of the record away. The link from case 80 back to case 12 is one of the things that gets lost.
>
> The last sentence names the way out. Do not pack everything into one prompt. Leave the record outside and let the reader fetch the parts it needs, as a search engine or a database does. "Adaptively" means that the next thing to read depends on what the reader found so far.

### Feedback three ways, in our run

Our run can measure this compression. Take harness B and write its feedback in three forms: a score, a short summary, and the full traces. Then count the tokens of each with the model's own tokenizer:

```text
== 4. Feedback about harness B, three ways
a score:               10 tokens  -> B_labels: accuracy 0.433
a short summary:       32 tokens
the full traces:    11729 tokens  (60 prompts, outputs and verdicts)
traces / score = 1173x
what only the traces show: 34 wrong answers, and 13 of them are outputs that match no label
  output 'gastroesophageal reflux disease (GERD)' gold 'gastroesophageal reflux disease'
  output 'gastroesophageal reflux disease (GERD)' gold 'gastroesophageal reflux disease'
  output 'Urinary tract infection (UTI)'    gold 'urinary tract infection'
5 of the 13 rejected outputs name the right disease inside a longer answer
```

The traces are 1,173 times longer than the score, for only 60 short cases. They also hold the one fact that tells you what to fix. Five answers were right but had an abbreviation in brackets, and the strict reader threw them away. The score cannot say this. Our summary, "The model often gives answers that do not match a label", comes close, but it does not say which answers or in what form.

The fix that the traces suggest is exactly harness C. C went from 26 to 31 right answers, a gain of 5, the same 5 cases that the traces showed. This is the paper's argument on a small scale: the diagnosis was in the raw record, and compression would have removed it.

<figure class="fig"><svg viewBox="0 0 720 256" role="img" aria-label="Bars on a log scale. Our run: a score is 10 tokens, a summary 32 tokens, the full traces 11,729 tokens. The paper: earlier optimizers see at most about 30,000 tokens per step; one Meta-Harness evaluation can produce up to 10,000,000 tokens."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">How much feedback the optimizer can read (tokens, log scale)</text><line class="axis" x1="290.0" y1="50" x2="290.0" y2="226"/><text class="t-tick" x="290.0" y="242.0" text-anchor="middle">10^0</text><line class="axis" x1="341.4" y1="50" x2="341.4" y2="226"/><text class="t-tick" x="341.4" y="242.0" text-anchor="middle">10^1</text><line class="axis" x1="392.9" y1="50" x2="392.9" y2="226"/><text class="t-tick" x="392.9" y="242.0" text-anchor="middle">10^2</text><line class="axis" x1="444.3" y1="50" x2="444.3" y2="226"/><text class="t-tick" x="444.3" y="242.0" text-anchor="middle">10^3</text><line class="axis" x1="495.7" y1="50" x2="495.7" y2="226"/><text class="t-tick" x="495.7" y="242.0" text-anchor="middle">10^4</text><line class="axis" x1="547.1" y1="50" x2="547.1" y2="226"/><text class="t-tick" x="547.1" y="242.0" text-anchor="middle">10^5</text><line class="axis" x1="598.6" y1="50" x2="598.6" y2="226"/><text class="t-tick" x="598.6" y="242.0" text-anchor="middle">10^6</text><line class="axis" x1="650.0" y1="50" x2="650.0" y2="226"/><text class="t-tick" x="650.0" y="242.0" text-anchor="middle">10^7</text><text class="t-tick" x="282.0" y="77.0" text-anchor="end">our run: a score</text><rect class="s2" x="290.0" y="61.0" width="51.4" height="22.0" rx="3"/><text class="t-val" x="347.4" y="77.0" text-anchor="start">10</text><text class="t-tick" x="282.0" y="111.0" text-anchor="end">our run: a summary</text><rect class="s2" x="290.0" y="95.0" width="77.4" height="22.0" rx="3"/><text class="t-val" x="373.4" y="111.0" text-anchor="start">32</text><text class="t-tick" x="282.0" y="145.0" text-anchor="end">our run: full traces</text><rect class="s2" x="290.0" y="129.0" width="209.3" height="22.0" rx="3"/><text class="t-val" x="505.3" y="145.0" text-anchor="start">11,729</text><text class="t-tick" x="282.0" y="179.0" text-anchor="end">paper: prior optimizers, top of range</text><rect class="s3" x="290.0" y="163.0" width="230.3" height="22.0" rx="3"/><text class="t-val" x="526.3" y="179.0" text-anchor="start">30,000</text><text class="t-tick" x="282.0" y="213.0" text-anchor="end">paper: one evaluation, up to</text><rect class="s1" x="290.0" y="197.0" width="360.0" height="22.0" rx="3"/><text class="t-val" x="656.0" y="213.0" text-anchor="start">10,000,000</text></svg><figcaption>Feedback sizes on a log scale. Our three forms of feedback for one harness, and the two sizes that §1 of the paper gives: at most about 30,000 tokens per step for earlier optimizers, and up to 10,000,000 tokens for one Meta-Harness evaluation.</figcaption></figure>

## The answer: a coding agent and a filesystem {§1}

> [!PAPER] · §1
> [![The fourth paragraph of the introduction, with coding agent, the amount of experience quickly exceeds context limits, expose full history through a filesystem, and grep and cat highlighted](/img/papers/meta-harness/p1-intro-agent.png)](/img/papers/meta-harness/p1-intro-agent.png)
>
> **What it says:** the proposer of Meta-Harness is a "coding agent", a system built on a language model "that can invoke developer tools and modify code". This matters because "the amount of experience quickly exceeds context limits". The key design choice is to "expose full history through a filesystem", which the proposer reads with "grep and cat".
>
> **What it means:**
>
> A plain language model gets one prompt and writes one answer. If the history does not fit in the prompt, the model never sees it. A coding agent works in steps. It can list a folder, search the files for a word, open one file, and then decide what to open next. A person who debugs a large codebase works the same way: nobody reads all of it, but anyone can find the right part.
>
> The filesystem is the store of experience. Each earlier harness has a folder with its code, its scores and its traces. The proposer uses ordinary tools on it. `grep` finds lines that contain a pattern, in many files at once. `cat` prints one file. With these two tools, a history of millions of tokens is no problem, because the agent reads only the parts it chooses.
>
> This is also why the paper could not have worked a few years earlier. The method needs an agent that is good at exactly this: searching a codebase, forming a theory, and editing code. In a footnote of §3, the authors say they think this workflow became practical only after coding agents improved around early 2026.

> [!DEFINITION] Coding agent
> A system built on a language model that can run developer tools in a loop. It can list, read and search files, run programs and edit code. After each tool call it sees the result and decides the next step.

> [!DEFINITION] Filesystem
> The folders and files on a computer's disk. Here it is the store of experience: one folder per harness, with its code, scores and traces as ordinary files.

> [!PAPER] · §1
> [![An excerpt of the introduction, with median of 82 files per iteration and over 20 prior candidates per step highlighted](/img/papers/meta-harness/p1-intro-82.png)](/img/papers/meta-harness/p1-intro-82.png)
> [![The next lines of the introduction, with up to 10,000,000 tokens of diagnostic information and roughly three orders of magnitude highlighted](/img/papers/meta-harness/p1-intro-10m.png)](/img/papers/meta-harness/p1-intro-10m.png)
>
> **What it says:** the proposer reads "a median of 82 files per iteration" in the most demanding setting, "referencing over 20 prior candidates per step". One evaluation can produce "up to 10,000,000 tokens of diagnostic information", "roughly three orders of magnitude" beyond the largest feedback budgets of earlier text optimizers.
>
> **What it means:**
>
> These numbers show that the proposer really uses the history. It does not read only the last harness and tweak it. In each round it opens about 82 files and looks at more than 20 earlier harnesses. Appendix A gives the split: about as many files of harness code as files of traces.
>
> "Three orders of magnitude" means a factor of about 1,000. Ten million tokens against at most 30,000 tokens is a factor of about 330, and against the typical budgets the gap is larger. No prompt can hold ten million tokens. The only way to use that much feedback is to search it and read parts, which is what the agent does. [Part 5](part-5-limits-and-what-came-after.md) reads Appendix A, where the proposer's own notes show this reading at work.

### grep, two ways

Our run wrote its own small filesystem: one folder per harness, with `scores.json` and `traces.jsonl`. A proposer would search it. Here we count the wrong answers in each folder in two ways. Method 1 is our own Python. Method 2 is the real `grep` program that a coding agent would run:

```python
def grep_count(root, pattern):
    """Count matching lines in every traces.jsonl under root, like grep -rc."""
    counts = {}
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn == 'traces.jsonl':
                p = os.path.join(dirpath, fn)
                counts[os.path.relpath(p, root)] = sum(pattern in line for line in open(p))
    return dict(sorted(counts.items()))


ours = grep_count(FS, PATTERN)
out = subprocess.run(['grep', '-rc', '--include=traces.jsonl', PATTERN, '.'], cwd=FS, capture_output=True, text=True).stdout
```

```text
== 5. Search the trace files: our Python (Method 1) and grep (Method 2)
A_bare/traces.jsonl              Method 1  60   grep  60
B_labels/traces.jsonl            Method 1  34   grep  34
C_labels_lenient/traces.jsonl    Method 1  29   grep  29
D_labels_retrieval/traces.jsonl  Method 1  14   grep  14
Method 1 and grep agree: True
```

The pattern is `"correct": false`, so each count is the number of wrong answers of one harness. They match the accuracy table: 60 minus 26 right is 34 wrong for B. One command answers a question about every harness at once, and its output is a few lines, not 11,729 tokens. That is why an agent with `grep` can work with a history far larger than its context window.

## The results, in one paragraph {§1}

> [!PAPER] · §1
> [![The results paragraph of the introduction, with by 7.7 points while using 4× fewer, after 60 proposals, with only four, 4.7 points, and ranks #1 among all Haiku 4.5 agents highlighted](/img/papers/meta-harness/p1-intro-results.png)](/img/papers/meta-harness/p1-intro-results.png)
>
> **What it says:** on text classification, discovered harnesses improve over Agentic Context Engineering (ACE) "by 7.7 points while using 4× fewer" context tokens. They match the next-best text optimizer's final performance "after 60 proposals with only four". On math, one harness gains "4.7 points" on 200 IMO-level problems. On TerminalBench-2, the harness surpasses Terminus-KIRA and "ranks #1 among all Haiku 4.5 agents".
>
> **What it means:**
>
> ACE is the strongest hand-designed harness for this kind of text classification. It builds up notes about the task as it sees labeled examples (the definition box below). Meta-Harness found a harness that is more accurate and also uses a smaller prompt. That is a win on both axes at once.
>
> "After 60 proposals with only four" is the speed claim of Figure 1. The other optimizers needed their whole budget to reach a level that Meta-Harness reached in its first four tries. Figure 1's axis stops at 40 evaluations, so the paper's counts need care; [Part 3](part-3-text-classification.md) reads the exact budgets of each method.
>
> IMO-level problems are at the level of the International Mathematical Olympiad, the hardest contest for school students. Terminus-KIRA is a strong harness that people built for TerminalBench. Beating it, and placing first among all published Haiku 4.5 agents, means beating teams who tuned their harness for this exact benchmark.

> [!DEFINITION] ACE (Agentic Context Engineering)
> A hand-designed harness from 2025. The paper describes it as using "reflective memory curation to build context over time": as it sees labeled examples, it keeps and edits notes that go into the prompt. It is the paper's main baseline for text classification.

### The headline numbers, checked

The arithmetic behind the claims is short, so our script checks it:

```text
== 6. The paper's headline numbers, checked
text classification: Meta-Harness 48.6 vs ACE 40.9 (points)        7.7
context: ACE 50.8K vs Meta-Harness 11.4K tokens (times fewer)      4.46
TerminalBench-2, Opus 4.6: 76.4 vs Terminus-KIRA 74.7 (points)     1.7
TerminalBench-2, Haiku 4.5: 37.6 vs Goose 35.5 (points)            2.1
the 6x source: same model, 12% vs 2% of 50 tasks (ratio)           6.0
12% and 2% of 50 tasks, as task counts                             (6.0, 1.0)
```

The accuracies and context sizes come from the paper's Table 2 and Table 7. "4× fewer" is 4.46 when you divide, so the paper rounds down. On TerminalBench-2, the gains are smaller in points: 1.7 on Opus 4.6 and 2.1 on Haiku 4.5. On a benchmark of 89 tasks, one task is worth about 1.1 points, so each of these gains is worth one or two tasks. Part 4 reads these tables in full.

## What our small run cannot show

Our run shows that a harness can move the score of a frozen model a lot. It does not show Meta-Harness itself. Here, a person wrote the four harnesses, with a toy model and 60 cases. The paper's proposer is Anthropic's coding agent with Opus 4.6, and it ran about 60 harnesses per search on far larger tasks. We have no API key and pay for no API, so we cannot run that proposer.

Our task is also easy for retrieval: many Symptom2Disease cases use similar words for the same disease. The paper's model, GPT-OSS-120B, is much larger than ours. Our 76.7% is on 60 cases of one dataset, while the paper's 48.6% is the average over three datasets and is not comparable. [Part 6](part-6-build-it-from-scratch.md) builds a small Meta-Harness loop that does propose harnesses, with a local model as the proposer. It also says what a small proposer can and cannot do.

> [!TAKEAWAYS] Key takeaways
> - **A harness** is the code around a model: what to store, retrieve and present, and how to read the answer. The paper never changes the model's weights, only this code.
> - **The 6× claim** comes from SWE-Bench Mobile. Opus 4.5 solved 12% of 50 tasks in one agent and 2% in another: 6 tasks against 1.
> - **In our run,** one frozen Qwen2.5-1.5B model scored 0%, 43.3%, 51.7% and 76.7% on the same 60 cases with four harnesses.
> - **Earlier optimizers compress feedback:** they forget past attempts, see only a score, or read a summary. They see at most 100 to 30,000 tokens per step.
> - **Meta-Harness** gives a coding agent a filesystem with the code, scores and full traces of every earlier harness. The agent reads it with tools like `grep` and `cat`, a median of 82 files per round.
> - **In our run,** the full traces of one harness were 1,173 times longer than its score. They also showed the exact fix: 5 right answers that a strict reader rejected.
> - **The results:** +7.7 points over ACE with 4× less context on text classification, and +4.7 points on 200 IMO-level problems across five models. On TerminalBench-2, first place among Haiku 4.5 agents (37.6%).

**Next, in Part 2:** §2 and §3 of the paper, the method. Part 2 reads the objective as an equation, and the search loop of Figure 2 and Algorithm 1. It also shows how the paper keeps a search set apart from a test set.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/meta-harness/mh_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/meta-harness/mh_part1.py). It loads Qwen2.5-1.5B-Instruct and Symptom2Disease, runs the four harnesses on 60 test cases, and writes one folder per harness to `results/part1_fs/`. It then checks the prompt and the accuracy two ways, counts the feedback tokens, and runs `grep` on the folders. Every number is saved in `results/part1.json`. On an Apple M-series laptop it takes a few minutes.

```bash
pip install torch transformers datasets scikit-learn
python mh_part1.py   # downloads Qwen/Qwen2.5-1.5B-Instruct (about 3 GB) and gretelai/symptom_to_diagnosis
```

<figure><img src="/img/papers/meta-harness/part1-run.png" alt="Terminal output of mh_part1.py: the accuracy table of the four harnesses, the prompt check, the accuracy check, the feedback sizes, the grep check and the paper's numbers" loading="lazy" /><figcaption>The full output of mh_part1.py, as it ran for this article.</figcaption></figure>

</details>

> [!TIP] Try it
> Open a real harness search in your browser, with every candidate's code, scores and traces: [the interactive demo](/papers/meta-harness/demo).

## The paper and the code

[Meta-Harness on arXiv](https://arxiv.org/abs/2603.28052) · [project page](https://yoonholee.com/meta-harness/) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness)
