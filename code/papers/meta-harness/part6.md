---
title: "Build It from Scratch: A Small Meta-Harness in Python"
description: "We build the whole Meta-Harness loop in about a thousand lines of Python: a task, harness files, an evaluator that logs every prompt, a folder per candidate, and two proposers. Then we run it on a laptop with small local models and measure the gap between the search set and the test set."
part: 6
covers: "Figure 2, §1, §3, Algorithm 1, §4.1, Table 8, Appendix B.1, Figure 7, Appendix C.1, Appendix D"
date: 2026-10-18
tags: [meta-harness, harness-engineering, from-scratch, python, llm, agents]
---

Parts 1 to 5 read the paper. In this part we build its loop ourselves, run it on a laptop, and look at what it finds and where it fails.

Everything runs on local models, with no API and no cost. The model that answers is Qwen2.5-0.5B-Instruct. The proposer that writes new harnesses is either a set of fixed rules or Qwen2.5-1.5B-Instruct. The paper used much stronger models, and this page says at each step what that changes.

## The plan: Figure 2 as files {§3}

> [!PAPER] · Figure 2
> [![Figure 2 of the paper: a filesystem of all experience on the left, an agent that proposes harness code, the harness with the language model, an evaluation on tasks, and an arrow that stores all logs back into a new folder of the filesystem](/img/papers/meta-harness/p6-figure2.png)](/img/papers/meta-harness/p6-figure2.png)
>
> **What it says:** an agent "reads a filesystem containing all prior candidates" (their code, traces and scores) and proposes a new harness. The loop evaluates it, and all its logs are "stored in the filesystem in a new directory". Then the loop repeats.
>
> **What it means:**
>
> Figure 2 has three arrows, and each arrow is a piece of code we must write. Arrow 1 is the proposer: it reads old folders and writes a new harness file. Arrow 2 is the evaluator: it runs the harness with the model on some tasks. Arrow 3 is the store: it saves everything in a new folder.
>
> The figure also hides two pieces that the text names. A check rejects broken harness files before any evaluation. And a test set stays outside the loop, so the final score is honest. [Part 2](part-2-the-method.md) explains the loop in depth. Here we only build it.

> [!DEFINITION] Harness
> The code around a language model that decides what the model sees: which instructions, which examples, which memory. The model's weights do not change. In our build, a harness is one Python file.

> [!DEFINITION] Proposer
> The part of the loop that writes new harnesses. In the paper it is a coding agent: a language model that can read files, run commands and edit code. In our build, it is either fixed rules or a small local model.

Our package is in [code/papers/meta-harness/scratch](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness/scratch). The picture below puts our file names on the pieces of Figure 2.

{{FIG:p6_map|The loop of Figure 2 with our files. The proposer reads the folders of earlier candidates and writes a new harness file. validate() rejects broken files. evaluate() runs the frozen model on the search set and writes a new folder. The test set sits outside the loop and gets used once, at the end.}}

| File | Piece of the paper | What it does |
|---|---|---|
| `task.py` | the tasks $$\mathcal{X}$$ | 10 Banking77 intents; memory 60, search set 50, test set 100 |
| `llm.py` | the frozen model $$M$$ | Qwen2.5-0.5B-Instruct, greedy decoding |
| `harness_api.py` | interface validation | the `Harness` interface and `validate()` |
| `design_space.py` | the seeds | a harness template with seven knobs |
| `evaluate.py` | Evaluate($$H, M, \mathcal{X}$$) | runs a harness and records every prompt and reply |
| `store.py` | the filesystem $$\mathcal{D}$$ | one folder per candidate, and the Pareto frontier |
| `proposers.py` | the proposer $$P$$ | scripted rules, a local LLM, and two that need a key |
| `loop.py` | Algorithm 1 | the outer loop, line by line |
| `run.py`, `mh.py` | | one full run; a small command-line tool over a run folder |

## The task {§4.1}

> [!PAPER] · §4.1
> [![Section 4.1: the online text classification setup, where an LLM receives labeled examples one at a time, updates its memory, and is evaluated on a held-out test set](/img/papers/meta-harness/p6-online.png)](/img/papers/meta-harness/p6-online.png)
>
> **What it says:** in online text classification, "an LLM receives labeled examples one at a time, updates its memory, and is evaluated on a held-out test set".
>
> **What it means:**
>
> Picture a new employee at a bank's help desk. A senior colleague shows them old customer messages, one at a time, each with the right category. The employee keeps notes. Then real messages arrive, and the employee must label each one alone.
>
> The model is the employee, and it never learns in the usual sense: its weights stay fixed. The harness is the notebook and the habits. It decides what to write down from each example, and what to look up before it answers. Two harnesses with the same model can give very different scores.

We use the same setup. Our harness gets 60 labeled messages through a method `update(text, label)`, one at a time. Then it must label new messages through `predict(text)`.

> [!PAPER] · Appendix C.1
> [![Appendix C.1: Banking77 is a fine-grained intent classification benchmark with online banking user utterances labeled with 77 intents](/img/papers/meta-harness/p6-banking77.png)](/img/papers/meta-harness/p6-banking77.png)
>
> **What it says:** Banking77 contains "online banking user utterances labeled with 77 intents". It tests intent detection "with a large label space".
>
> **What it means:**
>
> Banking77 is one of the nine datasets that the paper uses to test its discovered harness on new tasks (Table 5). Each message is a short question from a bank customer, and the label is what the customer wants.
>
> We keep only 10 of the 77 intents, all about cards. They are close to each other on purpose: "my card has not arrived" and "my card payment is still pending" share many words. A small model makes many mistakes here, so a better harness has room to help.

> [!DEFINITION] Intent
> What a customer wants, as one fixed label. "When will my card come?" has the intent `card_delivery_estimate`. The task is to pick the right intent from a fixed list.

> [!PAPER] · Appendix D
> [![Appendix D: start with a baseline harness and a search set that is hard for it; the search has little to optimize if the baseline already saturates the evaluation; keep the search set small enough for roughly 50 full evaluations per run](/img/papers/meta-harness/p6-search-set.png)](/img/papers/meta-harness/p6-search-set.png)
>
> **What it says:** start with a baseline and "a search set that is hard for it", because "the search has little to optimize if the baseline already saturates the evaluation". Keep the search set small enough for "roughly 50 full evaluations per run".
>
> **What it means:**
>
> A search set is the exam that the loop uses to grade each new harness. If the first harness already gets 98%, no new idea can show itself. So the authors pick hard examples. They also keep the exam short, because the loop sits the exam many times.

> [!DEFINITION] Search set and test set
> The **search set** scores each candidate during the search, and the proposer sees those scores. The **test set** is a separate group of tasks. It gets used once, after the search, and the proposer never sees it. The gap between the two scores shows how much the search fitted its own exam.

`task.py` follows this advice. It takes 6 memory messages and 5 search messages per intent from Banking77's training file, and 10 test messages per intent from its test file. The first section of our check script prints the sizes:

```text
== 1. The task
10 labels: activate_my_card, card_arrival, card_delivery_estimate, card_not_working, ...
memory 60 (labeled stream), search 50, test 100
example: 'I made a purchase with my card and I was charged a fee for using it' -> card_payment_fee_charged
```

Our first harness, zero-shot, scores 32.0% on the search set, against 10% for a random guess. So the search set is hard for the baseline, and there is room to improve.

## The frozen model {§3}

The model is the part that the search never changes. `llm.py` wraps Qwen2.5-0.5B-Instruct behind one call: `llm(prompt)` returns the reply. It decodes greedily: at each step, it takes the most likely next token. So the same prompt always gives the same reply, and two scores differ only because two harnesses differ.

> [!DEFINITION] Greedy decoding
> A way to make a model write text. At each step the model gives a score to every possible next token, and we take the token with the highest score. We repeat until the model writes its end token.

The library does this inside `model.generate()`. To see what it does, `llm.py` also writes the loop out by hand (Method 1). The library way is `LocalLLM.__call__` (Method 2):

```python
@torch.no_grad()
def greedy_from_scratch(llm, prompt, max_new_tokens=16):
    """Method 1: greedy decoding written out. No KV cache, so it is slower, but every step is visible."""
    ids = llm.chat_ids(prompt)                                   # (1, n) the prompt tokens
    start = ids.shape[1]
    for _ in range(max_new_tokens):
        logits = llm.model(ids).logits                           # (1, n, vocab): a score for every next token
        next_id = logits[0, -1].argmax()                         # the most likely token after the last position
        if next_id.item() == llm.tok.eos_token_id or next_id.item() in llm.tok.all_special_ids:
            break                                                # the model says "I am done"
        ids = torch.cat([ids, next_id.view(1, 1)], dim=1)        # (1, n + 1): append and go again
    return llm.tok.decode(ids[0, start:], skip_special_tokens=True).strip()
```

The check sends three zero-shot prompts both ways:

```text
== 2. The frozen model: greedy decoding two ways
from scratch: 'card_payment_not_recognised'  generate(): 'card_payment_not_recognised'  gold: card_payment_fee_charged
from scratch: 'pending_card_payment'         generate(): 'pending_card_payment'         gold: card_payment_not_recognised
from scratch: 'declined_card_payment'        generate(): 'declined_card_payment'        gold: card_payment_fee_charged
same reply in 3 of 3 prompts
```

The two ways agree on all three replies. All three replies are also wrong. That is the zero-shot harness at work: the model sees only the list of labels, and it guesses.

## A harness is one Python file {§3}

> [!PAPER] · §3
> [![The Practical implementation paragraph of Section 3: each harness is a single-file Python program, the proposer is guided by a minimal domain-specific skill, the base model is always frozen, and a typical run evaluates roughly 60 harnesses over 20 iterations](/img/papers/meta-harness/p6-practical.png)](/img/papers/meta-harness/p6-practical.png)
>
> **What it says:** each harness is "a single-file Python program" that changes prompting, retrieval, memory and orchestration. The proposer is Anthropic's coding agent with Opus 4.6, guided by "a minimal domain-specific skill". The base model is "always frozen". A typical run evaluates "roughly 60 harnesses over 20 iterations".
>
> **What it means:**
>
> One file per harness makes the search space concrete. A candidate is not a vague idea; it is a file you can open, run, compare and copy. Every tool that works on code now works on harnesses: `diff`, `grep`, a test.
>
> The last number sets the scale. The paper's run is about 60 evaluations. Ours, with a shared laptop GPU, is 13. The shape of the loop is the same; the budget is about five times smaller.

Every harness must follow one interface. Here is the smallest useful harness, from the package's README:

```python
from hlib import TfIdf, parse_label

class Harness:
    def __init__(self, llm, labels):
        self.llm, self.labels, self.memory = llm, labels, []

    def update(self, text, label):
        self.memory.append((text, label))

    def predict(self, text):
        index = TfIdf([t for t, _ in self.memory])
        shots = [self.memory[i] for i in index.top(text, 4)]
        prompt = 'Labels: ' + ', '.join(self.labels) + '\n\n'
        prompt += ''.join(f'Message: {t}\nLabel: {l}\n\n' for t, l in shots)
        prompt += f'Message: {text}\nLabel:'
        return parse_label(self.llm(prompt), self.labels) or self.labels[0]
```

`update()` writes the example into memory. `predict()` finds the 4 stored messages most similar to the new one, shows them with their labels, and asks the model. `parse_label()` turns a reply like "Label: card arrival." into a valid label.

One rule matters for honesty: a harness may not open files, import `os`, run programs or use the network. Without the rule, a harness could read the test set from disk. With it, everything the harness knows comes from `update()` and from the model.

## Interface validation {Algorithm 1}

> [!PAPER] · Appendix D
> [![Appendix D: lightweight validation before expensive benchmarks, a test that imports the module, instantiates the class, and calls both methods; and automate evaluation outside the proposer](/img/papers/meta-harness/p6-validation.png)](/img/papers/meta-harness/p6-validation.png)
>
> **What it says:** write a small test that "imports the module, instantiates the class, and calls both methods" on a few examples. It catches most broken candidates "in seconds". And "Automate evaluation outside the proposer".
>
> **What it means:**
>
> A full evaluation costs real time: in our run, from 37 to 157 seconds per candidate. A harness with a typo would waste all of it and then score zero. A test that takes a fraction of a second throws such a file away first.
>
> The second tip splits the work. The proposer only writes files. A separate program scores them and writes the results. So the proposer cannot run the exam itself, and it cannot change the scores.

`validate()` in `harness_api.py` does exactly the three steps of the tip. It loads the file, makes a `Harness` with a fake model that always answers with the first label, calls `update()` six times and `predict()` twice. It also searches the source for the forbidden patterns. The check feeds it four files:

```text
== 4. Interface validation (Algorithm 1, line 11)
the README harness         passed in 2.3 ms  
reads a file               rejected in 0.0 ms  forbidden pattern '\\bopen\\s*\\(': a harness may not touch files, pro
returns a made-up label    rejected in 0.9 ms  predict() returned 'other', which is not one of the labels
has no Harness class       rejected in 1.0 ms  no class named Harness
```

Each check takes at most 2.3 milliseconds. A full evaluation takes at least 37 seconds, more than ten thousand times longer. Later, the local LLM proposer gives this check real work.

## Retrieval inside a harness: TF-IDF two ways {Appendix B.1}

> [!PAPER] · Appendix B.1
> [![Appendix B.1: the label-primed query-anchored harness, with a label primer, one query-relevant example per label, query-anchored contrastive pairs, and TF-IDF retrieval over past labeled examples](/img/papers/meta-harness/p6-tfidf.png)](/img/papers/meta-harness/p6-tfidf.png)
>
> **What it says:** the strongest discovered harness shows the labels, then "one query-relevant example per label", then "query-anchored contrastive pairs". In code, it uses "TF-IDF retrieval over past labeled examples".
>
> **What it means:**
>
> The best harness of the paper's text classification search does not use a neural search engine. It uses TF-IDF, a word-counting method from the 1970s. What the search discovered is not a new tool. It is a good way to use an old tool: which examples to fetch, and in which order to show them.
>
> So our harnesses need TF-IDF too. We write it ourselves in `hlib.py`, and we check it against scikit-learn.

> [!DEFINITION] TF-IDF
> A way to turn a text into a list of word weights. A word gets more weight when it occurs often in this text (term frequency, TF). It also gets more weight when it is rare in the other texts (inverse document frequency, IDF). Two texts are similar when their weights point the same way.

The weight of a word $$w$$ in a text $$d$$ is a product of two numbers:

$$
\text{tfidf}(w, d) = \text{tf}(w, d) \times \text{idf}(w), \qquad \text{idf}(w) = \ln\frac{1 + N}{1 + \text{df}(w)} + 1
$$

Read it left to right. $$\text{tf}(w, d)$$ counts how many times the word occurs in the text. $$N$$ is the number of stored texts, and $$\text{df}(w)$$ counts how many of them contain the word. A rare word has a small $$\text{df}$$, so a large IDF.

A tiny example with $$N = 3$$ texts. The word "card" is in all three: $$\text{idf} = \ln(4/4) + 1 = 1$$. The word "fee" is in one: $$\text{idf} = \ln(4/2) + 1 = 1.69$$. So "fee" counts 1.69 times more than "card", which tells almost nothing in a set of card messages. Last, each vector is divided by its length, and the similarity of two texts is the dot product of their vectors: the **cosine**.

```python
class TfIdf:
    def __init__(self, docs):
        self.docs = list(docs)
        self.n = len(self.docs)
        df = Counter(w for d in self.docs for w in set(words(d)))
        self.idf = {w: math.log((1 + self.n) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = [self.vector(d) for d in self.docs]

    def vector(self, text):
        tf = Counter(w for w in words(text) if w in self.idf)        # words never seen in memory are dropped
        v = {w: c * self.idf[w] for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def scores(self, text):
        q = self.vector(text)
        return [sum(q[w] * dv.get(w, 0.0) for w in q) for dv in self.vecs]   # cosine with every document

    def top(self, text, k):
        """Indices of the k most similar documents, best first."""
        s = self.scores(text)
        return sorted(range(self.n), key=lambda i: -s[i])[:k]
```

Method 2 is scikit-learn's `TfidfVectorizer`, given the same word splitter. Its default settings use the same IDF formula and the same length division:

```python
sk = TfidfVectorizer(tokenizer=hlib.words, lowercase=False, token_pattern=None)   # same words as ours
D = sk.fit_transform(docs)
```

```text
== 3. TF-IDF retrieval two ways
vocabulary: ours 198 words, sklearn 198 words
largest difference in cosine scores over 5 queries x 60 documents: 1.7e-16
same top-5 neighbours in 5 of 5 queries
query: 'I made a purchase with my card and I was charged a fee for using it'
  0.531  card_payment_fee_charged   'What is the fee charged for using my card?'
  0.357  card_payment_fee_charged   'Am I getting charged for making a payment?'
  0.299  card_payment_fee_charged   'I paid with a card and there is an extra charge. Why?'
```

The two methods agree to $$1.7 \times 10^{-16}$$, the rounding limit of the computer. The three nearest stored messages for a "fee" question all carry the right label. That is why retrieval helps: the model sees the right answer next to a similar question.

## The design space and the seeds {Appendix B.1}

> [!PAPER] · Appendix B.1
> [![Appendix B.1: the draft verification harness retrieves the 5 most similar examples for a draft, then 5 confirmers with the same label and 5 challengers with different labels, and falls back to a single-call few-shot prompt when memory is small](/img/papers/meta-harness/p6-draft-verify.png)](/img/papers/meta-harness/p6-draft-verify.png)
>
> **What it says:** the draft-verification harness makes two calls. The first gives a draft label. The second shows "5 confirmers with the same label and 5 challengers with different labels" and asks whether to keep the draft. With too little memory, it "falls back to a standard single-call few-shot prompt".
>
> **What it means:**
>
> This is how a careful person checks an answer. First a quick guess. Then two questions: what do real examples of my guess look like, and what do the close alternatives look like? If the message looks more like a challenger, change the answer.
>
> The fallback matters too. A harness must work at the start of the stream, when memory is almost empty. Real code needs such edge cases, and a code search finds them where a prompt search cannot.

`design_space.py` turns the strategies of Appendix B.1 into one harness template with seven knobs:

| Knob | Values | From the paper |
|---|---|---|
| `shots` | 0, 4, 8, 16, 60 | zero-shot and few-shot baselines |
| `select` | first, random, tfidf | TF-IDF retrieval |
| `coverage` | off, on | one example per label |
| `contrastive` | off, on | close examples with different labels |
| `verify` | off, on | draft verification (2 calls, 3 confirmers, 3 challengers) |
| `label_style` | `card_arrival`, `card arrival` | |
| `instruction` | short, careful | |

Each point of the space renders as a complete Python file with a `CONFIG` dict at the top. The three seeds are points too: zero-shot, 8 random examples (few-shot), and all 60 examples. These are the paper's own kinds of baseline (Table 2).

## Evaluate, and keep everything {§3}

> [!PAPER] · §3
> [![Section 3: each evaluated harness contributes a directory containing its source code, scores, and execution traces; the proposer queries it through terminal tools such as grep and cat](/img/papers/meta-harness/p6-directory.png)](/img/papers/meta-harness/p6-directory.png)
>
> **What it says:** each evaluated harness "contributes a directory containing its source code, scores, and execution traces". The proposer queries this folder "through terminal tools such as grep and cat".
>
> **What it means:**
>
> A score says that a harness failed. A trace says why. The trace is the record of the run: each prompt the harness built, each reply the model gave, and the final answer. With the trace, you can open one wrong answer and see what the model was shown.
>
> This is the heart of the paper's ablation (Table 3). With scores only, the best harness reached 41.3%. With full traces, it reached 56.7%. So our evaluator must save traces, not only numbers.

> [!DEFINITION] Execution trace
> A complete record of one run of a harness on one task. It holds every prompt sent to the model, every reply, and the final answer next to the right one. In our build, a trace is one line of JSON.

`evaluate.py` wraps the model in a small recorder, `TracingLLM`. Each call to the model appends the prompt and the reply to a list. The evaluator first feeds the memory stream, then predicts each message of the set:

```python
def evaluate(harness_path, llm, labels, memory, eval_set):
    """Return (scores, traces). Accuracy is the share of messages whose predicted label equals the gold label."""
    t0 = time.time()
    tracer = TracingLLM(llm)
    harness = load_module(harness_path).Harness(tracer, list(labels))
    for ex in memory:
        harness.update(ex['text'], ex['label'])
    traces, correct, invalid, errors = [], 0, 0, 0
    per_label = collections.defaultdict(lambda: [0, 0])
    confusions = collections.Counter()
    for i, ex in enumerate(eval_set):
        tracer.calls = []
        try:
            pred = harness.predict(ex['text'])
            err = None
        except Exception as e:                                  # a crash on one message counts as a wrong answer
            pred, err = None, f'{type(e).__name__}: {e}'
            errors += 1
        if pred not in labels:
            invalid += 1
        ok = pred == ex['label']
        correct += ok
        per_label[ex['label']][0] += ok
        per_label[ex['label']][1] += 1
        if not ok:
            confusions[f"{ex['label']} -> {pred}"] += 1
        traces.append({'id': i, 'text': ex['text'], 'gold': ex['label'], 'pred': pred, 'correct': ok,
                       'calls': tracer.calls, 'prompt_chars': sum(len(c['prompt']) for c in tracer.calls),
                       **({'error': err} if err else {})})
```

The score is the accuracy on the search set. This is the paper's objective in its most basic form. The reward $$r$$ of one message is 1 if the label is right and 0 if not. The expectation in the objective becomes a mean over the 50 messages. [Part 2](part-2-the-method.md) reads the formula symbol by symbol. The check confirms the mean on one saved candidate:

```text
== 5. The objective: the mean reward over the search set
tiny example: rewards [1, 0, 1, 1, 0] -> mean 0.60
012_60_tfidf_examples_verify: 29 rewards of 1 out of 50 -> 58.0%  (scores.json: 58.0%)
```

Next to the accuracy, `scores.json` stores the prompt size (characters per message), the accuracy per label, and the five most common confusions. Each candidate gets a folder like this one:

```text
runs/scripted-1/candidates/006_8_tfidf_examples_verify/
    harness.py      the code
    scores.json     accuracy 56.0, ctx_chars 2318, calls 2.0, accuracy by label, top confusions
    traces.jsonl    50 lines: the text, gold, pred, and every prompt and reply
    proposal.md     the parent, the diagnosis, the edit and the reason
```

The first trace line of that folder shows why traces matter. The message is "I was charged for using my card?", and the right label is `card_payment_fee_charged`. The first call replied `declined_card_payment`, a wrong draft. The second call showed confirmers and challengers, and the final answer was right. Only the trace shows that the second call fixed the first.

> [!PAPER] · Appendix D
> [![Appendix D: log everything in a format that is easy to navigate, using machine-readable formats such as JSON; and make logs queryable through a small CLI that lists the Pareto frontier, shows top-k harnesses, and diffs code and results](/img/papers/meta-harness/p6-logs.png)](/img/papers/meta-harness/p6-logs.png)
>
> **What it says:** log code, scores and traces in "machine-readable formats such as JSON", with consistent names. A small command-line tool that "lists the Pareto frontier, shows top-k harnesses, and diffs" code saves the proposer effort.
>
> **What it means:**
>
> A proposer reads the folder the way you read a codebase you do not know. Clear names and JSON make `grep` work. A summary command answers the first question ("which harnesses are best?") without opening 50 files.
>
> Our `mh.py` has five commands: `list`, `frontier`, `errors`, `diff` and `test`. The `test` command only works after the search ends.

## The Pareto frontier {§3}

> [!PAPER] · §3
> [![Section 3: when multiple objectives are relevant, such as accuracy and context cost, candidates are evaluated under Pareto dominance and the resulting frontier is reported](/img/papers/meta-harness/p6-pareto.png)](/img/papers/meta-harness/p6-pareto.png)
>
> **What it says:** when there is more than one goal, such as accuracy and context cost, the paper evaluates candidates "under Pareto dominance" and reports "the resulting frontier".
>
> **What it means:**
>
> A longer prompt often gives a better answer, but it costs more. So "the best harness" has no single meaning. Think of laptops: one is lighter, another is faster. A laptop is a bad buy only when another one is lighter *and* faster.
>
> The frontier is the list of good buys. Algorithm 1 returns it, and the paper shows its frontier in Figure 3.

> [!DEFINITION] Pareto frontier
> Candidate A **dominates** B when A is at least as good on every goal and strictly better on one. The Pareto frontier is the set of candidates that no other candidate dominates. Here the goals are high accuracy and a short prompt.

`store.py` computes the frontier two ways. Method 1 follows the definition: compare each candidate with every other one. Method 2 sorts by prompt size, cheapest first, and keeps each candidate that beats the best accuracy so far:

```python
def pareto_front(points):
    """Method 1, from the definition. points: list of (accuracy, context). Higher accuracy and lower context are better.
    A point is on the frontier if no other point is at least as good on both and strictly better on one."""
    front = []
    for i, (a, c) in enumerate(points):
        dominated = any((a2 >= a and c2 <= c) and (a2 > a or c2 < c) for j, (a2, c2) in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front


def pareto_front_sweep(points):
    """Method 2, the usual fast way: sort by context (cheapest first), keep each point that beats the best accuracy so far."""
    order = sorted(range(len(points)), key=lambda i: (points[i][1], -points[i][0]))
    front, best = [], float('-inf')
    for i in order:
        if points[i][0] > best:
            front.append(i)
            best = points[i][0]
        elif front and points[i] == points[front[-1]]:      # an exact tie with a kept point: neither dominates
            front.append(i)
    return sorted(front)
```

The first version of the sweep had no tie rule, and the check caught it: the sweep dropped two candidates that the definition kept. Two candidates with exactly the same accuracy and the same prompt size do not dominate each other, so both belong on the frontier. After the fix, the two methods agree on the 13 candidates of our scripted run:

```text
== 6. The Pareto frontier two ways
13 candidates; by definition: [0, 1, 3, 5, 6, 7, 12]; by sort and sweep: [0, 1, 3, 5, 6, 7, 12]; same: True
  000_zero_shot                                 32.0%    402 chars
  001_few_shot_8                                42.0%   1139 chars
  003_8_tfidf_examples                          54.0%   1177 chars
  005_8_tfidf_examples_contrastive              54.0%   1177 chars
  006_8_tfidf_examples_verify                   56.0%   2318 chars
  007_8_tfidf_examples_contrastive_verify       56.0%   2318 chars
  012_60_tfidf_examples_verify                  58.0%   6956 chars
```

[![The terminal output of mh_part6.py: the six checks, from the task sizes to the Pareto frontier](/img/papers/meta-harness/part6_checks-run.png)](/img/papers/meta-harness/part6_checks-run.png)

## The outer loop {Algorithm 1}

> [!PAPER] · Algorithm 1
> [![Algorithm 1: initialize a population of harnesses and an empty filesystem, evaluate each seed, then for N iterations let the proposer query the filesystem and propose k new harnesses, evaluate each one that passes interface validation, and return the Pareto frontier](/img/papers/meta-harness/p6-alg1.png)](/img/papers/meta-harness/p6-alg1.png)
>
> **What it says:** evaluate the seed harnesses and store them. Then, for $$N$$ iterations, the proposer queries the filesystem and proposes $$k$$ new harnesses. Each one that "passes interface validation" gets evaluated and stored. At the end, "return Pareto frontier of harnesses stored in D".
>
> **What it means:**
>
> The algorithm is short because the proposer does the hard part. The loop does not choose a parent, does not mutate code and does not rank ideas. It only keeps the books: evaluate, store, repeat.
>
> So once the pieces exist, the loop is about 25 lines. Each comment below names the line of Algorithm 1 that the code implements.

```python
def meta_harness(run_dir, seeds, proposer, llm, labels, memory, search, iterations, k, log=print):
    store = Store(run_dir)                                                   # line 3: D <- empty
    log(f'== seeds ({len(seeds)})')
    for name, source, proposal in seeds:                                     # lines 4 to 6
        evaluate_into(store, name, source, proposal, llm, labels, memory, search, log)
    stats = {'proposed': 0, 'rejected': 0, 'duplicates': 0, 'evaluated': 0}
    for t in range(1, iterations + 1):                                       # line 7
        t0 = time.time()
        proposals = proposer.propose(store, t, k)                            # lines 8 and 9
        log(f'== iteration {t}: {len(proposals)} proposals ({time.time() - t0:.0f}s to propose)')
        seen = {normalise(c['source']): c['folder'] for c in store.candidates()}
        for p in proposals:                                                  # line 10
            stats['proposed'] += 1
            ok, err = validate(p['source'], labels, memory)                  # line 11
            if ok and normalise(p['source']) in seen:
                ok, err = False, f"duplicate: same code as {seen[normalise(p['source'])]}"
                stats['duplicates'] += 1
            if not ok:
                stats['rejected'] += 1
                d = store.reject(p['name'], p['source'], err, p['proposal'])
                log(f'  x {os.path.basename(d):44s} rejected: {err[:90]}')
                continue
            folder, _ = evaluate_into(store, p['name'], p['source'], p['proposal'], llm, labels, memory, search, log)
            seen[normalise(p['source'])] = folder
            stats['evaluated'] += 1
    return store, store.frontier(), stats                                    # line 13
```

We add one thing that the algorithm does not list: a duplicate check. If a proposal has the same code as an earlier candidate (ignoring spaces and comments), the loop rejects it. Our deterministic evaluator would only give the same score again.

## Proposer 1: fixed rules {§3}

A proposer needs to read traces and turn them into an edit. The paper hands this job to a strong coding agent. We start with something we can fully understand: hand-written rules. This is the honest baseline. If a smart proposer cannot beat fixed rules, it adds nothing.

`ScriptedProposer` starts from the best candidate on the search set. It reads that candidate's traces and counts three things. The first is replies that are not a valid label. The second is wrong answers whose prompt showed no example of the right label. The third is the most common confusion.

Then fixed if-then rules turn the counts into edits. This is a slice of them:

```python
        if cfg['shots'] > 0 and cfg['select'] != 'tfidf':
            out.append(({'select': 'tfidf'},
                        f"{d['gold_not_shown']} of {d['wrong']} wrong answers had no example of the gold label in the "
                        'prompt; pick the examples by similarity to the message'))
        if d['gold_not_shown'] > 0.3 * w and not cfg['coverage']:
            out.append(({'coverage': True},
                        f"{d['gold_not_shown']} of {d['wrong']} wrong answers never saw the gold label in an example; "
                        'add one example of every label'))
        if d['top_pair_count'] >= 2 and not cfg['contrastive']:
            g, p = d['top_pair']
            out.append(({'contrastive': True},
                        f"the most common mistake is {g} -> {p} ({d['top_pair_count']} times); show close examples "
                        'with different labels side by side'))
```

Each proposal writes its reasons to `proposal.md`. This is the one behind candidate 006:

```text
Parent: 003_8_tfidf_examples (search accuracy 54.0%, 1177 prompt characters)

Diagnosis from the parent's traces: {"n": 50, "wrong": 23, "invalid": 0, "gold_not_shown": 4, "top_pair": ["card_arrival", "pending_card_payment"], "top_pair_count": 4, "ctx_chars": 1177}

Edit: mistakes repeat between the same labels; check the first answer against confirmers and challengers
```

The rules are the weak point, and we know it. They can only find what we thought of in advance. They cannot invent a new strategy, and they cannot notice a bug in their own code. Keep that in mind for the results.

## Run 1: the scripted proposer

The run has 3 seeds and 5 iterations of 2 proposals, so 13 candidates. It ran on an Apple laptop GPU (`mps`) that other jobs shared, so the times are slower than an idle machine would give.

```text
run scripted-1: proposer scripted, 5 iterations x 2 proposals
task model Qwen/Qwen2.5-0.5B-Instruct on mps; 10 labels; memory 60, search 50, test 100 (test is never shown to the proposer)
== seeds (3)
  + 000_zero_shot                                search acc  32.0%   ctx   402 chars   calls 1.0   (37.3s)
  + 001_few_shot_8                               search acc  42.0%   ctx  1139 chars   calls 1.0   (47.3s)
  + 002_few_shot_all                             search acc  26.0%   ctx  5803 chars   calls 1.0   (42.7s)
== iteration 1: 2 proposals (0s to propose)
  + 003_8_tfidf_examples                         search acc  54.0%   ctx  1177 chars   calls 1.0   (37.0s)
  + 004_8_random_examples_coverage               search acc  34.0%   ctx  2022 chars   calls 1.0   (43.0s)
== iteration 2: 2 proposals (0s to propose)
  + 005_8_tfidf_examples_contrastive             search acc  54.0%   ctx  1177 chars   calls 1.0   (60.4s)
  + 006_8_tfidf_examples_verify                  search acc  56.0%   ctx  2318 chars   calls 2.0   (127.2s)
== iteration 3: 2 proposals (0s to propose)
  + 007_8_tfidf_examples_contrastive_verify      search acc  56.0%   ctx  2318 chars   calls 2.0   (122.3s)
  + 008_16_tfidf_examples_verify                 search acc  50.0%   ctx  3093 chars   calls 2.0   (157.1s)
== iteration 4: 2 proposals (0s to propose)
  + 009_4_tfidf_examples_verify                  search acc  50.0%   ctx  1923 chars   calls 2.0   (95.7s)
  + 010_8_tfidf_examples_verify_careful_instruct search acc  56.0%   ctx  2480 chars   calls 2.0   (81.8s)
== iteration 5: 2 proposals (0s to propose)
  + 011_8_first_examples_verify                  search acc  54.0%   ctx  2342 chars   calls 2.0   (70.2s)
  + 012_60_tfidf_examples_verify                 search acc  58.0%   ctx  6956 chars   calls 2.0   (72.0s)
== search done in 16.8 min: {'proposed': 10, 'rejected': 0, 'duplicates': 0, 'evaluated': 10}
```

Read it from the top. Zero-shot gets 32.0%. Eight random examples get 42.0%. All 60 examples get only 26.0%: the 0.5B model gets lost in a 5,803-character prompt. More context is not always better, and the paper saw the same thing: past 32 examples, few-shot hurt on 7 of its 9 new datasets.

The first rule gives the largest jump. Choosing the 8 examples by TF-IDF instead of at random takes the score from 42.0% to 54.0%, at almost the same prompt size. Draft verification adds 2 points (56.0%) but doubles the calls. The last explore step, all 60 examples ranked by TF-IDF with verification, reaches 58.0%.

Candidate 005 is a lesson in itself. The contrastive knob changed the code, but not one prompt: the same 1,177 characters and the same 54.0%. Its block takes the nearest example of each label among the 8 nearest messages, and the 8 TF-IDF examples already contain them. A score alone hides this. The traces show it at once, because the prompts are the same.

{{FIG:p6_progress|Search accuracy of each candidate (dots) and the best so far (line), for the two proposers. The first three candidates are the same seeds in both runs. The scripted proposer climbs to 58.0%. The local LLM proposer stops at 48.0%.}}

{{FIG:p6_pareto|Each candidate of the scripted run as a dot: prompt size against search accuracy. The orange step line is the Pareto frontier. Moving right along it buys a few points of accuracy with a much longer prompt: from 1,177 characters at 54.0% to 6,956 characters at 58.0%.}}

## The test set, at last {§3}

> [!PAPER] · §3
> [![Section 3: the proposer never sees test-set results; its only feedback comes from the search set and from execution traces logged during those search runs](/img/papers/meta-harness/p6-never-sees.png)](/img/papers/meta-harness/p6-never-sees.png)
>
> **What it says:** "The proposer never sees test-set results". Its only feedback comes from the "search set" and from the traces of the search runs.
>
> **What it means:**
>
> A student who practises on last year's exam can learn the answers by heart instead of the subject. The only fair grade is a new exam. The search set is the practice exam, and the test set is the new one.
>
> In our package the rule is a matter of folders. Proposers read only `candidates/`. Test results go to `heldout/`, and `run.py` writes them only after the last iteration.

After the search, `run.py` scores every candidate of the scripted run on the 100 test messages:

```text
== test set (100 messages) for 13 candidates
  000_zero_shot                                search  32.0%  test  38.0%  gap  -6.0   (frontier)
  001_few_shot_8                               search  42.0%  test  42.0%  gap  +0.0   (frontier)
  002_few_shot_all                             search  26.0%  test  29.0%  gap  -3.0
  003_8_tfidf_examples                         search  54.0%  test  53.0%  gap  +1.0   (frontier)
  004_8_random_examples_coverage               search  34.0%  test  36.0%  gap  -2.0
  005_8_tfidf_examples_contrastive             search  54.0%  test  53.0%  gap  +1.0   (frontier)
  006_8_tfidf_examples_verify                  search  56.0%  test  55.0%  gap  +1.0   (frontier)
  007_8_tfidf_examples_contrastive_verify      search  56.0%  test  55.0%  gap  +1.0   (frontier)
  008_16_tfidf_examples_verify                 search  50.0%  test  52.0%  gap  -2.0
  009_4_tfidf_examples_verify                  search  50.0%  test  52.0%  gap  -2.0
  010_8_tfidf_examples_verify_careful_instruct search  56.0%  test  53.0%  gap  +3.0
  011_8_first_examples_verify                  search  54.0%  test  50.0%  gap  +4.0
  012_60_tfidf_examples_verify                 search  58.0%  test  50.0%  gap  +8.0   (frontier)
selected by search accuracy: 012_60_tfidf_examples_verify search 58.0% test 50.0%
total 35.2 min, 3000 task-model calls
```

The candidate with the best search score, 012, is not the best on the test set. It drops from 58.0% to 50.0%, a gap of 8 points. Candidate 006 scores 2 points lower on the search set, and it holds up: 56.0% on search, 55.0% on test.

Why? One step on our search set is one message, so 2 points are one message. The difference between 006 and 012 is one more right answer out of 50. With so few messages, a lucky candidate can rise to the top, and 012 looks like such a case. The fix is not a cleverer proposer. It is a larger search set, or a rule that prefers the cheaper candidate when two scores are within noise.

> [!PAPER] · Figure 7
> [![Figure 7 of the paper: search-set accuracy against test accuracy for discovered strategies on LawBench, Symptom2Disease and USPTO, with the dashed diagonal y=x](/img/papers/meta-harness/p6-figure7.png)](/img/papers/meta-harness/p6-figure7.png)
>
> **What it says:** each pink dot is one discovered strategy, with its search accuracy across and its test accuracy up. "The dashed diagonal is y=x".
>
> **What it means:**
>
> A dot on the diagonal scores the same on both sets. A dot below it did better on the search set than on the test set: it fitted the practice exam. In the paper's three panels, the cloud of discovered strategies mostly follows the diagonal, with spread around it.
>
> Our run gives the same kind of picture at a smaller scale, below. Most of our dots sit near the line. The best search candidate sits clearly below it.

{{FIG:p6_gap|Search accuracy against test accuracy for all 13 candidates of the scripted run, drawn like the paper's Figure 7. The seeds (green) sit on or above the line. The candidate with the highest search score (orange, far right) sits 8 points below it.}}

## Proposer 2: a small local model {§1}

Now the real idea of the paper: a language model reads the folder and writes the code. We have no key for a strong model, so we use the largest model that fits next to the task model: Qwen2.5-1.5B-Instruct.

> [!PAPER] · Appendix D
> [![Appendix D: write a good skill; the skill text is the primary interface for steering the search, and it should constrain outputs and safety-relevant behavior, not the proposer's diagnosis procedure](/img/papers/meta-harness/p6-skill.png)](/img/papers/meta-harness/p6-skill.png)
>
> **What it says:** the skill text is "the primary interface for steering the search". It should "constrain outputs and safety-relevant behavior, not the proposer's diagnosis procedure". The authors tune it with a few short runs of 3 to 5 iterations.
>
> **What it means:**
>
> A skill is a short instruction sheet for the agent: your job, where the files are, what you may not touch, what to hand in. It does not say how to think. The paper's advice is to say what to deliver and leave the diagnosis to the agent.
>
> That advice assumes an agent that can diagnose. We follow it anyway, and we see below what happens when the agent is small.

> [!DEFINITION] Skill
> A text file of instructions that an agent reads before it starts work. Ours is `skill.md`. It gives the harness interface, the folder layout and the rules (no files, no network). It also fixes the output: one `IDEA:` line, then the whole file.

`LocalLLMProposer` works in two steps per proposal. First, it sees the list of candidates and names up to 3 files it wants to read. Second, it gets those files and writes a complete harness. We also always show it the source of the best harness, with the instruction to start from it. Without that help, both proposals of a first test crashed: one used a class that did not exist, and the other imported a made-up module. If its file fails validation, it gets the error and one more try.

```text
run local-llm-1: proposer local-llm, 5 iterations x 2 proposals
task model Qwen/Qwen2.5-0.5B-Instruct on mps; 10 labels; memory 60, search 50, test 100 (test is never shown to the proposer)
== seeds (3)
  + 000_zero_shot                                search acc  32.0%   ctx   402 chars   calls 1.0   (75.6s)
  + 001_few_shot_8                               search acc  42.0%   ctx  1139 chars   calls 1.0   (68.2s)
  + 002_few_shot_all                             search acc  26.0%   ctx  5803 chars   calls 1.0   (79.1s)
== iteration 1: 2 proposals (413s to propose)
  x 003_llm_t1_0                                 rejected: ImportError: cannot import name 'log' from 'hlib' (/Users/admin/Desktop/ishwar-books/code/
  + 004_llm_t1_1                                 search acc  48.0%   ctx  1848 chars   calls 1.0   (40.7s)
== iteration 2: 2 proposals (237s to propose)
  + 005_llm_t2_0                                 search acc  36.0%   ctx  1870 chars   calls 1.0   (36.9s)
  + 006_llm_t2_1                                 search acc  48.0%   ctx  1848 chars   calls 1.0   (31.9s)
== iteration 3: 2 proposals (376s to propose)
  x 007_llm_t3_0                                 rejected: duplicate: same code as 004_llm_t1_1
  x 008_llm_t3_1                                 rejected: duplicate: same code as 004_llm_t1_1
== iteration 4: 2 proposals (239s to propose)
  x 009_llm_t4_0                                 rejected: NameError: name 'aspect' is not defined | draft = parse_label(self.llm(self.prompt(text, s
  + 010_llm_t4_1                                 search acc  48.0%   ctx  1848 chars   calls 1.0   (12.9s)
== iteration 5: 2 proposals (111s to propose)
  x 011_llm_t5_0                                 rejected: duplicate: same code as 005_llm_t2_0
  + 012_llm_t5_1                                 search acc  48.0%   ctx  1848 chars   calls 1.0   (8.6s)
== search done in 29.8 min: {'proposed': 10, 'rejected': 5, 'duplicates': 3, 'evaluated': 5}
```

Of 10 proposals, 2 crashed validation, 3 were exact copies of earlier code, and 5 ran. The best search score is 48.0%, 6 points above the best seed and 10 points below the scripted proposer. But the score does not tell the whole story. The files do.

Candidate 004 changed the `CONFIG` line of its parent: 5 random examples instead of 8, labels as words, and a longer instruction. It also turned on the coverage block, and it rewrote one line of it:

```text
-        return [self.nearest(text, 1, keep=lambda l, x=label: l == x)[0]
+        return [self.nearest(text, 1, keep=lambda l: True)[0]
```

The old line fetched the nearest example *of each label*. The new line fetches the nearest example of *any* label, ten times. So every prompt now starts with the same message, repeated ten times:

```text
Examples:
Message: What is the fee charged for using my card?
Label: card payment fee charged
Message: What is the fee charged for using my card?
Label: card payment fee charged
Message: What is the fee charged for using my card?
Label: card payment fee charged
```

This is a bug, and it still raised the score from 42.0% to 48.0%. By accident, it puts the single most similar stored message into the prompt, with a lot of weight. That is a crude form of the retrieval that the scripted proposer found on purpose. The validation check cannot catch it, because the code runs and returns valid labels.

The other proposals show the same pattern. Candidate 005 passes a label name, `pending_card_payment`, into every prompt as an extra line, and it drops to 36.0%. Candidates 006, 010 and 012 differ from 004 only in comments and in an unused test line at the end of the file. So they give exactly the same 48.0%. Our duplicate check ignores comments, but not code that never runs.

> [!PAPER] · §1
> [![Section 1: the choice of coding agent rather than raw LLM matters because the amount of experience quickly exceeds context limits, so the proposer must decide what to inspect; it retrieves files via standard operations such as grep and cat](/img/papers/meta-harness/p6-coding-agent.png)](/img/papers/meta-harness/p6-coding-agent.png)
>
> **What it says:** a coding agent, not a raw LLM, matters because "the amount of experience quickly exceeds context limits". The proposer must "decide what to inspect" and gets files "via standard operations such as grep and cat".
>
> **What it means:**
>
> Our small proposer is exactly the raw LLM that this sentence warns about. It reads what we paste into its prompt, and it cannot run its own code to test an idea. It cannot `grep` the traces for a pattern. It writes the file in one pass and hopes.
>
> So the result above is not a test of Meta-Harness. It is a test of the weakest proposer that still fits the loop. The loop handled it as designed: it rejected the broken files and kept the scores honest.

> [!PAPER] · Appendix A.1
> [![Appendix A.1 and Table 8: the proposer reads a median of 82 files per iteration, range 69 to 99, roughly evenly split between harness source code and execution traces](/img/papers/meta-harness/p6-table8.png)](/img/papers/meta-harness/p6-table8.png)
>
> **What it says:** in the TerminalBench-2 run, the proposer reads "a median of 82 files per iteration" (range 69 to 99). About 41% are harness source code and 40% are execution traces. Its access is "non-Markovian": it reads far back in the history, not only the latest parent.
>
> **What it means:**
>
> Our small proposer asked for 1 to 3 files per proposal, plus the best harness that we always show. Once it named no real file at all. That is more than 20 times fewer files than the paper's median. It also used little of what it read. Its `IDEA:` lines are general ("Add more diverse prompts"). Only one names a confusion from the traces, and its file was a copy of earlier code.
>
> The scripted proposer read less, only the best candidate's traces. But it turned three counts from them into a targeted edit. Reading is only useful when the reader can draw a conclusion.

> [!PAPER] · §3
> [![Footnote 1 of Section 3: based on earlier exploration, the authors think this workflow only became practical recently, following major improvements in coding-agent capabilities around early 2026](/img/papers/meta-harness/p6-footnote.png)](/img/papers/meta-harness/p6-footnote.png)
>
> **What it says:** the authors think the workflow "only became practical recently", after major gains in coding agents around early 2026.
>
> **What it means:**
>
> This footnote predicts our result. The loop is short; the proposer does the real work. With a 1.5B model, the loop does not find a better harness than fixed rules. The footnote suggests that the authors' own earlier exploration, with weaker agents, went the same way.

On the test set, the LLM run's best candidate also scores below the scripted one:

```text
== test set (100 messages) for 7 candidates
  000_zero_shot                                search  32.0%  test  38.0%  gap  -6.0   (frontier)
  001_few_shot_8                               search  42.0%  test  42.0%  gap  +0.0   (frontier)
  002_few_shot_all                             search  26.0%  test  29.0%  gap  -3.0
  004_llm_t1_1                                 search  48.0%  test  46.0%  gap  +2.0   (frontier)
  006_llm_t2_1                                 search  48.0%  test  46.0%  gap  +2.0   (frontier)
  010_llm_t4_1                                 search  48.0%  test  46.0%  gap  +2.0   (frontier)
  012_llm_t5_1                                 search  48.0%  test  46.0%  gap  +2.0   (frontier)
selected by search accuracy: 004_llm_t1_1 search 48.0% test 46.0%
total 31.8 min, 1100 task-model calls
```

| Proposer | Candidates evaluated | Best search | Its test score | Proposal time |
|---|---|---|---|---|
| none (best seed, few-shot 8) | 3 | 42.0% | 42.0% | |
| scripted rules | 10 | 58.0% (012) | 50.0% | under 1 second |
| Qwen2.5-1.5B-Instruct | 5 of 10 | 48.0% (004) | 46.0% | 111 to 413 seconds per iteration |
| best scripted candidate on test (006) | | 56.0% | 55.0% | |

## The API proposer, for readers with a key

The package has two more proposers that we did not run. Both need something that costs money or that we do not have here.

`ChatAPIProposer` does the same two steps as the local proposer, but it sends them to any chat endpoint that speaks the common `/chat/completions` protocol. Only the `chat` method changes:

```python
    def chat(self, prompt, max_new_tokens, seed):
        body = {'model': self.model, 'temperature': self.temperature, 'max_tokens': max_new_tokens, 'seed': seed,
                'messages': [{'role': 'system', 'content': SKILL}, {'role': 'user', 'content': prompt}]}
        req = urllib.request.Request(f'{self.base}/chat/completions', data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json',
                                              'Authorization': f'Bearer {self.key}'})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)['choices'][0]['message']['content']
```

`AgentCLIProposer` is the paper's own setup. It writes `SKILL.md` into the run folder and starts a coding agent there, with its own file tools. The agent writes its harnesses into `pending/`, and the loop collects, validates and evaluates them. To use either one:

```bash
export MH_API_BASE=https://<provider>/v1 MH_API_KEY=... MH_API_MODEL=<model>
python run.py --proposer api --iters 10 --k 2 --name api-1

export MH_AGENT_CMD='<your-agent-command> --print {prompt}'
python run.py --proposer agent-cli --iters 10 --k 2 --name agent-1
```

Our results do not predict what such a run would find. The paper's proposer is Anthropic's coding agent with Opus 4.6, and its proposals read about 82 files each. We expect a strong proposer to beat our fixed rules, because it can do what the rules cannot. It can notice the coverage bug, invent a strategy that is not a knob, and test a change before it hands it in. That is a claim of the paper, not something this page measured.

## What our small version cannot show

- **The scale.** The paper evaluates about 60 harnesses per run and up to 10,000,000 tokens of diagnostic information per evaluation. Our runs evaluate 13 and 8 candidates, and each trace file holds 50 records that fit in a text editor.
- **A strong proposer.** Our only LLM proposer has 1.5B parameters and no tools. It cannot test the result of its own edits, the ability that the paper's footnote calls essential.
- **Statistical confidence.** One search-set message is 2 points and one test message is 1 point. Most of our differences are a few messages. We ran each search once, with one seed.
- **The paper's numbers.** We use a different model (Qwen2.5-0.5B, not GPT-OSS-120B) and 10 of Banking77's 77 intents. None of our accuracies are comparable with Tables 2 to 5.

What it does show is the machinery. Each piece of Figure 2 and Algorithm 1 is a small, testable function. Traces explain results that scores hide, such as candidate 005 and the coverage bug of candidate 004. And the search-test gap appears even in a loop of 13 candidates.

> [!TAKEAWAYS] Key takeaways
> - The whole loop fits in about a thousand lines of Python, comments included. It has a task, a harness interface, a validator, an evaluator with traces, a folder per candidate, the Pareto frontier and proposers.
> - Our own TF-IDF, greedy decoding and Pareto frontier match scikit-learn, `generate()` and a second method exactly. The Pareto check caught a tie bug in our own sweep.
> - With fixed rules as the proposer, the search climbs from 42.0% (best seed) to 58.0% on the search set. The largest single gain is TF-IDF retrieval of examples: 42.0% to 54.0%.
> - The best search candidate scores 50.0% on the test set, 8 points lower. A runner-up with a shorter prompt scores 55.0%. With 50 search messages, small differences are noise.
> - A 1.5B model as the proposer reaches 48.0% on search and 46.0% on test. Half its proposals crash or repeat old code, and its best file only works because of a bug.
> - Traces explain what scores hide: an edit that left every prompt the same, and a bug that repeats one example ten times.
> - The paper's footnote fits our result: the loop is short, and the strength of the proposer decides what the loop can find.

## The paper and the code

[arXiv 2603.28052](https://arxiv.org/abs/2603.28052) · [Project page](https://yoonholee.com/meta-harness/) · [Our code](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/meta-harness)
