---
description: "Workflow patterns: prompt chains, routers, parallel calls, orchestrators and evaluators; what each costs, how each fails and how to test it."
---
# Chapter 4 · Workflow patterns: chaining, routing, parallel work, orchestrators and evaluators

> **Goal:** by the end of this chapter you can draw any LLM system as a graph of steps, say which steps your code decides and which the model decides, and choose among the five workflow patterns (prompt chaining, routing, parallelisation, orchestrator-workers, evaluator-optimiser) by what each costs, how it fails and how you will test it. One running example carries you through all five, four small experiments against a real model show where each pattern earns its keep, and one complete program puts them together with the plumbing production needs: typed state, schema checks, budgets, retries, checkpoints, a human approval step and a trace.

---

## 4.1 Workflows versus agents: who holds the control flow

### The running example: Northwind Home's support inbox

Northwind Home is a fictional online shop that sells furniture and kitchenware in the UK. Its support inbox receives a few thousand messages a week: "order #48213 arrived with the screen cracked", "I was charged twice for ORD-10988", "do you ship to the Isle of Man?", "the sofa from order 61234 has a torn seam, I want my money back, £649". The team wants a model to read each message, work out what it is about, draft a reply, check the reply against company policy, and send it, with a person approving anything that costs money.

The first version is one prompt: "here is the policy, here is the message, write the reply". It mostly works, and it fails in ways nobody can see from outside: the order number is quoted wrongly, the policy's ban on promising delivery dates is ignored, a refund is offered on a question about shipping zones. Every fix is another sentence in a prompt that keeps growing, and every new sentence can break a case that used to work.

This chapter rebuilds that system five times, once per pattern, and asks the same four questions each time: what changes, what it costs, how it fails, and how you test it. Underneath all five is one question: **who decides what happens next?**

> [!DEFINITION] Control flow
> The order in which the steps of a program run, and the rule that picks the next step. In an LLM system it can be written by the engineer in code, chosen by the model at run time (its output names the next step), or a mixture of the two.

> [!DEFINITION] Workflow (LLM workflow)
> A system in which model calls and tools are arranged by control flow the engineer wrote in advance: the steps, their order and the conditions for branching are code. A model fills in content at each step and may choose between branches the code offers, but cannot invent a step the code does not contain. An **agent**, by contrast, lets the model decide which step comes next (Chapters 1 and 3).

Anthropic's December 2024 post "Building effective agents" drew this line, and the industry adopted it. It names five workflow patterns, and Sections 4.2 to 4.6 take them in turn.

> [!PAPER] Anthropic, "Building effective agents" (Engineering blog, December 2024)
> [![Two bullets from the post: Workflows are systems where LLMs and tools are orchestrated through predefined code paths; Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks](/img/agents/ch4-anthropic-workflows.png)](/img/agents/ch4-anthropic-workflows.png)
>
> **Context:** a post by Erik Schluntz and Barry Zhang, published 19 December 2024, drawing on Anthropic's work with customers. It groups all of these under "agentic systems" and then splits the group in two.
>
> **What it says:** workflows are "orchestrated through predefined code paths"; agents "dynamically direct their own processes and tool usage". It then describes prompt chaining, routing, parallelisation (sectioning and voting), orchestrator-workers and evaluator-optimiser, and recommends "finding the simplest solution possible, and only increasing complexity when needed".
>
> **Why it matters:** the distinction is about the control flow, not about how capable the model is. The same model can sit inside either. What changes is what you can test: in a workflow you can test each path, because you wrote them all; in an agent you mostly test outcomes, because the paths are generated.

{{FIG:ch4_spectrum|Six arrangements of model calls, ordered by who decides the next step. On the left the engineer writes every path, and the system is predictable, easy to test, cheap and fast, but brittle on inputs nobody foresaw; on the right the model decides every step and the trade reverses. The five workflow patterns sit in between, each handing the model one more kind of decision. The bars are a qualitative summary, not a measurement.}}

Read the figure as a sequence of hand-overs. A **chain** hands the model no decisions about flow. A **router** hands it one, made once: which branch should this input take? **Parallel** sections hand it none; code fans the work out and merges it. An **orchestrator** hands it the shape of the work: how many subtasks, and which. An **evaluator loop** hands it the decision to stop. An **agent** hands it every step. Each hand-over buys adaptability and usually costs predictability, testability, money and time. So a useful default is to hand the model a decision only when you cannot write that decision in code, and to know which decisions you have handed over.

### Graphs: the vocabulary of workflows

Every workflow can be drawn as a graph, and the frameworks that run them (LangGraph, the OpenAI Agents SDK, Temporal, AWS Step Functions) all use graph words.

> [!DEFINITION] Node, edge and conditional edge
> A **node** is one step: a model call, a tool call or plain code. It reads the current state (the data the workflow carries) and returns an update. An **edge** is a fixed transition: after A, always run B. A **conditional edge** is a function of the state that chooses the next node: after the classifier, run the refund flow if the label is "refund", otherwise the reply flow. The function can be ordinary code reading a field the model filled in, which is how a router keeps the model's choice inside paths the engineer wrote.

> [!DEFINITION] DAG (directed acyclic graph)
> A graph whose edges have a direction and contain no cycles. A chain and a fan-out followed by a merge are DAGs. A retry or an evaluator loop is a cycle, so a workflow with loops is a directed graph but not a DAG, and every cycle needs a stop rule.

> [!PAPER] LangChain, "Graph API overview" (LangGraph documentation, accessed October 2026)
> [![A passage from the LangGraph docs: State, a shared data structure that represents the current snapshot of the application; Nodes, functions that encode the logic of your agents, which receive the current state and return an updated state; Edges, functions that determine which Node to execute next based on the current state, which can be conditional branches or fixed transitions; Nodes and Edges are nothing more than functions, they can contain an LLM or just good ol' code; In short: nodes do the work, edges tell what to do next](/img/agents/ch4-langgraph-nodes-edges.png)](/img/agents/ch4-langgraph-nodes-edges.png)
>
> **Context:** the conceptual page of LangGraph, LangChain's graph-based orchestration library. Uber's Genie team built its pipeline on it (Section 4.8).
>
> **What it says:** a graph has state, nodes and edges; edges "can be conditional branches or fixed transitions"; nodes and edges "can contain an LLM or just good ol' code". In short: "nodes do the work, edges tell what to do next".
>
> **Why it matters:** "an LLM or just good ol' code" is the design space of this chapter in a phrase. Every node you can make plain code is free, fast and deterministic, and every edge you can make plain code is a decision that cannot go wrong in a new way tomorrow.

{{FIG:ch4_graph_vocab|Northwind's ticket flow drawn as a graph. An extract node feeds a gate that checks the output's schema and retries on failure; a classify node feeds a conditional edge that sends refunds to a flow with human approval and everything else to a reply flow. Squares mark checkpoints, where the state is saved so the run can pause or resume.}}

> [!DEFINITION] Gate
> A programmatic check between two steps that decides whether the workflow continues: the output parses, required fields exist, a value matches a pattern, a number is in range. A failed gate usually retries the step with the error in the prompt, or stops the run and hands it to a person. The word is Anthropic's.

> [!DEFINITION] Checkpoint
> A copy of the workflow's state written to durable storage after a node finishes, so the run can be paused for a person, resumed after a crash, or replayed for debugging (Section 4.7).

<details>
<summary>Research notes: OpenAI's view from the agent end (optional reading)</summary>

OpenAI's practical guide of April 2025 frames the same choice from the other side: start with one agent and add structure only when it fails.

> [!PAPER] OpenAI, "A practical guide to building agents" (April 2025) · page 16
> [![A page of the guide: When to consider creating multiple agents; our general recommendation is to maximize a single agent's capabilities first; when your agents fail to follow complicated instructions or consistently select incorrect tools, you may need to further divide your system; complex logic, when prompts contain many conditional statements; tool overload, some implementations manage more than 15 well-defined tools while others struggle with fewer than 10 overlapping tools](/img/agents/ch4-openai-guide-multi.png)](/img/agents/ch4-openai-guide-multi.png)
>
> **Context:** a 34-page PDF for teams building their first agents. It describes single-agent systems and two multi-agent shapes: a "manager" that calls other agents as tools, and peers that hand off to one another.
>
> **What it says:** "maximize a single agent's capabilities first"; split when an agent fails "to follow complicated instructions or consistently select incorrect tools", in particular when prompts contain "many conditional statements (multiple if-then-else branches)".
>
> **Why it matters:** an if-then-else inside a prompt is a branch the model has to choose; moving that branch into code is a router (Section 4.3). The guide also draws multi-agent systems as graphs with "agents represented as nodes", which is the vocabulary above.

The two vendors start at opposite ends and meet in the middle: add a moving part when a measurement asks for it.

</details>

### Discussion

1. **Is a router an agent?** The model makes a decision about flow. My view: no, as long as the branches are fixed in code and the router can be tested against labels like any classifier; it becomes agent-like when the model can name a branch you did not write.
2. **Does the distinction matter to users?** They see an answer, not a graph. But latency ceilings, cost ceilings and testability are properties of the graph, and those are what break in production.
3. **Will better models make workflows obsolete?** Better models move the line towards agents for open-ended tasks. For high-volume, well-understood tasks, a fixed path's predictability and cost per call are part of the product, so the line moves more slowly there.

## 4.2 Prompt chaining: fixed steps, with gates in between

### The example: split Northwind's one prompt into three

The one-prompt system drops requirements because it is asked to do four things at once: understand the ticket, choose a response, write it, and obey the policy. A **prompt chain** gives each job its own call:

1. **Extract**: read the ticket and return JSON with the issue type (one of five), the order id in the form `ORD-NNNNN`, and the urgency.
2. **Draft**: from those facts, choose the reply category from a fixed table and write the reply.
3. **Check**: test the reply against the policy (at most 70 words, quote the order id, no promises of timing such as "as soon as possible", no talk of refunds outside billing).

Between the steps sit **gates**: code that checks the previous step's output before the next one runs.

> [!DEFINITION] Prompt chain
> A workflow of a fixed sequence of model calls in which each call's output becomes part of the next call's input. Each step has its own prompt, can use its own model, and can be tested on its own. Anthropic describes it as trading "latency for higher accuracy, by making each LLM call an easier task".

> [!DEFINITION] Structured output
> Asking the model to return data in a fixed machine-readable shape, usually JSON that matches a schema (named keys, allowed values, types). Many APIs can enforce the shape during generation. Structured output is what makes gates possible: you cannot check a field buried in prose.

{{FIG:ch4_chain|Northwind's three-step chain: extract facts as JSON, draft a reply, check it against the policy. A gate after each step checks the output in code; a failed gate retries the step once with the error in the prompt, and a second failure hands the ticket to a person. Gates catch broken formats and impossible values; they cannot catch a plausible value that is wrong.}}

### Why chains need gates: the arithmetic

A chain multiplies probabilities. If each step returns a usable, correct output with probability $$r$$, the whole chain of $$n$$ steps succeeds with probability

$$P(\text{chain}) = r^{\,n}.$$

**Assumptions:** steps fail independently, a bad output anywhere ruins the result, and no later step repairs an earlier mistake. Real chains break all three in both directions: one confusing ticket makes several steps fail together, and a good drafting step can paper over a slightly wrong extraction. Treat the formula as a way to see the shape, not a forecast. The shape is unforgiving: five steps at 90% succeed 59% of the time; ten succeed 35%.

Now add a gate that detects a fraction $$d$$ of bad outputs and retries the step once. The step succeeds if the first attempt is good, or if it is bad, the gate notices and the retry is good:

$$r' = r + (1 - r)\, d\, r .$$

**Assumptions:** the retry succeeds with the same probability $$r$$ as the first attempt (in practice the error message often helps, so this is a floor), and the gate never rejects a good output. With $$r = 0.9$$ and $$d = 0.8$$, $$r' = 0.972$$, and five steps succeed 87% of the time instead of 59%, for about 8% more calls. The large condition is $$d$$: **the share of bad outputs the gate can see**. A gate sees a missing key, an unknown label, a malformed id, a reply over the word limit. It cannot see a well-formed answer that is wrong.

{{FIG:ch4_chain_reliability|End-to-end success of an n-step chain, from ch4_cost.py. At 99% per step a ten-step chain succeeds 90% of the time; at 90% per step, 35%. A gate that detects 80% of bad outputs and retries once lifts a 90% step to 97%, so a five-step chain goes from 59% to 87%. All three curves assume independent steps.}}

[![Terminal output of ch4_cost.py, the first parts: the chain-reliability table for per-step reliability 0.90, 0.95 and 0.99 with and without a gate that detects 80 percent of bad outputs, five steps at 90 percent giving 59 percent and 87 percent with the gate, with the assumptions printed; then routing cost against the share of hard inputs and the router's recall and false-escalation rate; then sectioning latency and the plurality-voting table with correlated errors](/img/agents/ch4_cost-run.png)](/img/agents/ch4_cost-run.png)

> [!DEFINITION] Step-level retry
> Re-running one step, not the whole workflow, when its gate fails, with the failure added to the prompt ("order_id must match ORD-NNNNN, got '48213'"). It costs one extra call for that step and works in proportion to how specific the failure message is.

### Our experiment: Northwind's chain, with and without gates

The script `ch4_chain.py` runs this chain on twenty short tickets with a real model (gpt-5-mini, reasoning effort set to minimal). The tickets are deliberately messy: order numbers written five ways ("order #48213", "ORD 55501", "order no. 30377"), one in Spanish, one mentioning two orders, several with no order at all. Each ticket has gold labels: its issue type and its canonical order id.

There are four conditions, varying one factor at a time. The step-1 prompt is either **strict** (it names the keys, the five allowed values and the `ORD-NNNNN` format) or **loose** ("extract the issue type, the order id and the urgency ... reply in JSON", the kind of prompt a first version often has). Each runs **without gates** or **with gates** (a schema check after step 1, a schema and policy check after step 2, one retry each). Success has two parts, reported separately because they differ in independence:

- **Gold correct**: the right reply category and the gold order id quoted in the reply. No gate ever sees the gold labels, so this is an independent measure.
- **Policy correct**: word limit, no timing promises, no refund talk outside billing. Gate 2 checks the same rules, so in the gated conditions this part is graded by the same code that gave the feedback. It shows compliance, not independent quality.

The whole run cost about four US cents; token counts come from the API's usage fields.

```python
# code/agents/ch4_chain.py (abridged): one chain step with a gate. Without gates, whatever came back flows on.
def step(prompt, gate, gated, use):
    text, u = llm.ask(prompt, max_tokens=500)                 # one model call; usage recorded from the API
    use.append(u)
    d = llm.parse_json(text)
    if not gated:
        return d, text, 0
    problems = gate(d)                                         # e.g. "order_id must match ORD-NNNNN or be null, got '48213'"
    if not problems:
        return d, text, 0
    retry = prompt + '\n\nYour previous answer failed these checks:\n- ' + '\n- '.join(problems) + \
            '\nReply again with a corrected JSON object only.'
    text, u = llm.ask(retry, max_tokens=500)                   # one retry, with the exact failures in the prompt
    use.append(u)
    d = llm.parse_json(text)
    return (d if not gate(d) else None), text, 1               # None stops the chain: hand the ticket to a person
```

[![Terminal output of ch4_chain.py: the loose prompt without gates, where tickets fail at step 1 because the model returned free-text issue types such as Damaged item, cracked screen, and bare order numbers such as 48213; then the summary table: strict prompts 19 of 20 with and without gates, loose without gates 12 of 20, loose with gates 16 of 20 at 78 calls instead of 60, policy correct 19 or 20 of 20 in every condition, and the step-3 model checker flagging policy P1 71 times](/img/agents/ch4_chain_summary-run.png)](/img/agents/ch4_chain_summary-run.png)

{{FIG:ch4_chain_results|Results of the final run of ch4_chain.py. With a strict first-step prompt the gates had nothing to catch: 19 of 20 either way. With a loose prompt, 8 of 20 chains broke on free-form labels and bare order numbers; gates with one retry brought success to 16 of 20 for 18 extra calls. The model checker in step 3 disagreed with a plain word count on most tickets.}}

**When the prompt is specific, the gates are idle.** With the strict prompt the model returned valid labels and canonical ids on every ticket; the gates never fired, and both conditions scored 19 of 20. The one failure was "Do you ship to the Isle of Man?", labelled "delivery" instead of "other". That is a valid label, so no gate could catch it; it needs a labelled set to find.

**When the prompt is loose, the gates do the work.** With the loose prompt every step-1 output parsed as JSON and almost none was usable: issue types came back as "Damaged item / cracked screen" and "Incorrect VAT rate on invoice", order ids as "48213" and "ORD 55501". Without gates these flowed on and 8 of 20 replies went to the wrong category or failed to quote the order id. With gates, all twenty tickets failed their first check, were retried once with the exact problems listed, and 16 of 20 succeeded. Of the four failures, one was the Isle of Man label, one was a ticket the gate correctly stopped and handed to a person (it mentioned two orders), and two are the instructive ones: the retry "fixed" the order id by setting it to `null`, which the schema allows for tickets with no order. **A retry optimises for the gate**, so a gate with an easy wrong exit gets that exit.

**A model is a poor checker of mechanical rules.** Step 3, the model asked whether the reply broke the policy, disagreed with a plain-code check on most tickets. Across the four conditions it flagged "over 70 words" 71 times on replies that a word count passed. A rule you can write in code (a word count, a list of banned phrases, a regular expression) is better checked in code, and a model checker belongs where code cannot reach, such as tone, after you have measured it (Section 4.6).

> [!WARNING]
> What this experiment does and does not show. Twenty tickets is a demonstration; the counts would move by one or two on another run. The strict and loose prompts differ in one thing (how precisely step 1 is specified), and the gated and ungated runs in one thing (the gates), so each comparison isolates one factor. It does not show how often a strict prompt fails on real traffic, which needs your own labelled tickets. The mechanisms are the findings: gates rescue loose interfaces between steps, cannot see valid-but-wrong values, and shape what the retry produces.

### The research behind chaining

The idea was first studied as a question of human control. In October 2021 Tongshuang Wu, Michael Terry and Carrie Cai at Google posted "AI Chains" (CHI 2022). Large models of the time did well on single operations and poorly on tasks with several parts, and users could not see or fix what went wrong inside one big prompt.

> [!PAPER] Wu et al. (2021), AI Chains · Figure 1 · page 2
> [![Figure 1 of the AI Chains paper: the same task, rewriting peer feedback about a presentation into a friendly paragraph, done by one prompt in panel A, which produces a generic paragraph, and by a chain in panel B: a Split points step extracts the three problems, an Ideation step brainstorms suggestions for each, and a Compose points step writes a paragraph that addresses all three](/img/agents/ch4-aichains-figure1.png)](/img/agents/ch4-aichains-figure1.png)
>
> **Context:** arXiv 2110.01691, posted 4 October 2021. The authors define eight "primitive operations" a single model call handles well (classification, factual query, generation, ideation, information extraction, rewriting, split points, compose points), build an interface for chaining them, and run a 20-person user study.
>
> **What it says:** one prompt (A) "remains mostly impersonal and does not provide concrete suggestions for all 3 of Alex's presentation problems"; the chain (B) splits the feedback into problems, ideates per problem and composes, and covers each one. In the study, chaining "improved the quality of task outcomes" and "significantly enhanced system transparency, controllability, and sense of collaboration", and users "debugged unexpected model outputs by 'unit-testing' sub-components of a Chain" (abstract).
>
> **Why it matters:** the failure in (A) is a dropped requirement, the same failure as Northwind's one prompt. Splitting the work makes each requirement the whole job of one step, and "unit-testing sub-components" is the production argument for chains, made in 2021.

The study's measures were participants' ratings, not accuracy; the authors note the limits (twenty people, two tasks, a 2021 model). In 2022 two papers turned decomposition into a reasoning method: Least-to-Most prompting (Zhou et al., Google) and Decomposed Prompting (Khot et al., Allen Institute for AI). Both found the largest gains on problems with many steps, and both point at the same practical conclusion: the decomposition is the hard part, and when an engineer can write it, it can live in code.

<details>
<summary>Research notes: Least-to-Most, Decomposed Prompting and the AI Chains abstract (optional reading)</summary>

> [!PAPER] Wu et al. (2021), AI Chains · Abstract · page 1
> [![Abstract of the AI Chains paper with highlighted phrases: the output of one step becomes the input for the next; Chaining not only improved the quality of task outcomes; debugged unexpected model outputs](/img/agents/ch4-aichains-abstract.png)](/img/agents/ch4-aichains-abstract.png)
>
> **Context:** the abstract of the CHI 2022 paper.
>
> **What it says:** in chaining "the output of one step becomes the input for the next, thus aggregating the gains per step".
>
> **Why it matters:** "aggregating the gains per step" is the optimistic reading of the product formula above; the reliability curve is the pessimistic one. Both are true, and gates decide which one you get.

> [!PAPER] Zhou et al. (2022), Least-to-Most · Figure 1 · page 2
> [![Figure 1 of the Least-to-Most paper: stage 1 decomposes a word problem about Amy and a water slide into the sub-question how long does each trip take; stage 2 solves sub-question 1, appends the answer, and then solves sub-question 2, Amy can slide 15 divided by 5 equals 3 times](/img/agents/ch4-l2m-figure1.png)](/img/agents/ch4-l2m-figure1.png)
>
> **Context:** arXiv 2205.10625 by Denny Zhou and colleagues at Google, posted 21 May 2022, published at ICLR 2023. The motivation is "easy-to-hard generalization": chain of thought did poorly on problems harder than its examples.
>
> **What it says:** first "decompose the problem into subproblems", then "sequentially solve the subproblems", each answer appended to the next call's context. On SCAN, a compositional-generalisation benchmark, the abstract reports at least 99% accuracy with 14 exemplars against 16% for chain of thought.
>
> **Why it matters:** this is a two-step chain whose first output is a plan.

> [!PAPER] Zhou et al. (2022), Least-to-Most · Section 3.3 and Table 11 · page 7
> [![A paragraph and Table 11 of the Least-to-Most paper: on GSM8K least-to-most improves chain of thought only slightly, from 60.97 to 62.39 percent, but on problems needing at least five steps from 39.07 to 45.23 percent; on the DROP subsets least-to-most reaches 82.45 and 73.42 against 74.77 and 59.56](/img/agents/ch4-l2m-table11.png)](/img/agents/ch4-l2m-table11.png)
>
> **Context:** maths word problems with code-davinci-002, decomposition and solving in one prompt.
>
> **What it says:** on all of GSM8K the gain is small, "from 60.97% to 62.39%"; on problems "which need at least 5 steps" it is "from 39.07% to 45.23%"; and "almost every problem in GSM8K that least-to-most prompting fails to solve can be eventually solved by using a manually crafted decomposition".
>
> **Why it matters:** gains concentrate where tasks have many steps, and a human-written decomposition beats the model's. When you can write it, write it in code.

> [!PAPER] Khot et al. (2022), Decomposed Prompting · Figure 1 · page 2
> [![Figure 1 of the Decomposed Prompting paper: standard prompting maps input to label; chain of thought shows reasoning steps; decomposed prompting has a decomposer prompt that calls sub-tasks A, B and C, handled by a standard prompt, a further decomposed prompt, or a symbolic function such as retrieval](/img/agents/ch4-decomp-figure1.png)](/img/agents/ch4-decomp-figure1.png)
>
> **Context:** arXiv 2210.02406 by Tushar Khot and colleagues at the Allen Institute for AI, posted 5 October 2022, published at ICLR 2023.
>
> **What it says:** a decomposer prompt describes the procedure; each sub-task "is handled by sub-task specific handlers which can vary from a standard prompt ... a further decomposed prompt ... or a symbolic function such as retrieval".
>
> **Why it matters:** a chain step does not have to be a model. If a step can be a regular expression, a database query or a search call, it can be swapped in without touching the other steps.

> [!PAPER] Khot et al. (2022), Decomposed Prompting · Abstract · page 1
> [![Abstract of the Decomposed Prompting paper with highlighted phrases: shared library of prompting-based LLMs dedicated to these sub-tasks; trained models, or symbolic functions if desired](/img/agents/ch4-decomp-abstract.png)](/img/agents/ch4-decomp-abstract.png)
>
> **Context:** the first half of the abstract.
>
> **What it says:** sub-tasks are delegated "to a shared library of prompting-based LLMs dedicated to these sub-tasks", each of which can be "replaced with more effective prompts, trained models, or symbolic functions if desired".
>
> **Why it matters:** replaceability is the engineering benefit: with stable interfaces between steps, you can swap a prompt for a small fine-tuned model, or a model for a rule, one step at a time, re-running only that step's eval.

</details>

### What a production team learned about gates

> [!PAPER] LinkedIn, "Musings on building a Generative AI product" (Engineering blog, April 2024)
> [![A paragraph from the post: while about 90 percent of the time the LLM responses contained the parameters in the right format, about 10 percent of the time the LLM would make mistakes and often output data that was invalid as per the schema supplied, or worse not even valid YAML; these mistakes caused the code parsing them to barf; 10 percent was a high enough number for us to not ignore trivially](/img/agents/ch4-linkedin-yaml.png)](/img/agents/ch4-linkedin-yaml.png)
>
> **Context:** a post by Juan Pablo Bottaro and Karthik Ramgopal on the assistant in LinkedIn Premium, a fixed pipeline in which the model chose an internal API ("skill") and wrote its parameters in YAML, chosen because it "consumes fewer tokens than JSON".
>
> **What it says (reported by LinkedIn):** about 90% of responses had parameters "in the right format" and about 10% were "invalid as per the schema supplied, or worse not even valid YAML". Re-prompting with the error worked but "adds a non-trivial amount of latency and also consumes precious GPU capacity", so the team "ended up writing an in-house defensive YAML parser".
>
> **Why it matters (our reading):** ten per cent is the per-step failure rate of the reliability curve, and it was enough to break a product. A gate has two responses: retry (one more call) or repair in code (no call). A parser that fixes the common mistakes is a gate whose retry is free.

### Prompt chaining in practice

| | One big prompt | Prompt chain with gates |
|---|---|---|
| Pros | one call, lowest latency; nothing to wire | each step easy for the model and testable alone; different models per step; format failures caught in code; one step changes without touching the others |
| Cons | requirements dropped silently; nothing to test between input and output | latency adds up; errors propagate; interfaces between steps must be designed and versioned; a gate can push a retry into a wrong-but-valid answer |
| When to pick it | the task is one operation (classify, extract, rewrite) | the task has distinct operations in a known order, especially if some can be code |
| Production failure | a missed requirement nobody notices until a user does | an upstream change (new input formats) degrades every step downstream |

**What to measure before you decide**

| Measurement | How | What it tells you |
|---|---|---|
| Per-step success rate | label each step's output on a hundred inputs | the chain's ceiling; which step to improve first |
| Gate detection rate $$d$$ | label the failures the gate missed | how much a gate can recover; valid-but-wrong is the rest |
| First fault position | for each failed run, the first step whose output was wrong | whether to fix extraction or drafting |
| Retry outcome | for each retry: fixed, still wrong, wrong in a new valid way | whether the gate's message helps or opens an easy exit (our `null` ids) |
| Latency per step | wall clock per node from traces | which step to shrink, parallelise or give a smaller model |

> [!TIP] In production
> Version the interface between steps like an API: a schema per step, validated in code, with the schema version in the trace. Write the gate before the prompt and make its messages specific ("got '48213', expected ORD-NNNNN"). Repair what you can predict in code (normalise "ORD 55501" to "ORD-55501"), retry what you cannot, and close easy exits: if `null` is allowed, check `null` against the input.

### Discussion

1. **How many steps should a chain have?** One per distinct operation, and no more. Merge two model steps whose combined prompt is still a single operation, and replace any step that can be code.
2. **Should a step see the original input or only the previous output?** Passing only structured outputs keeps contexts small but loses information. A reasonable rule: steps that write for a person see the original input; steps that classify or check see only what they need.
3. **Retry or repair?** LinkedIn chose a parser for latency. My view: repair what you can predict, retry what you cannot, and count both in the trace, because a rising repair rate is often the first sign that an upstream prompt or model has changed.

## 4.3 Routing: classify, then dispatch

### The example: not every Northwind question needs the expensive model

Northwind's team also runs an internal help desk where staff ask the model quick questions. Most are easy ("what is the plural of criterion?", "what is 15% of 240?"); some are multi-step calculations of the kind that price disputes produce ("3 pens for £2.40 and notebooks at £1.75; what do 9 pens and 4 notebooks cost?"). Sending everything to the strongest model is accurate and expensive; sending everything to the cheapest is cheap and wrong on the hard ones. A **router** looks at each input and sends it to the handler that suits it.

> [!DEFINITION] Router
> A step that classifies an input and dispatches it to one of several fixed handlers: a specialised prompt, a different model, a tool, or a person. The router can be a rule, a trained classifier, an embedding similarity check ("semantic routing"), or a model call. Its decision is the only one handed to the model, and it is made once.

> [!DEFINITION] Cascade
> A routing design that tries the cheapest handler first and escalates to a stronger one only when a confidence check on the cheap answer fails. The check can be a scorer, a verifier, or agreement between two cheap samples. Unlike a classifier router, a cascade decides after seeing an answer, so every input pays for at least one cheap call.

{{FIG:ch4_router|Two routing designs. A classifier router decides up front and sends the input to a specialised prompt, a stronger model or a person (the "unknown" bucket). A cheap-first cascade answers with the cheap model and escalates only when a confidence check fails.}}

### The arithmetic: a router's confusion matrix is its cost model

A router makes two kinds of mistake. It sends a hard input to the cheap handler (accuracy is lost), or an easy input to the expensive one (money is lost). Write $$h$$ for the share of hard inputs, $$s$$ for the router's recall on hard inputs (the share of hard inputs it escalates) and $$e$$ for its false-escalation rate (the share of easy inputs it escalates). Then the share of traffic that reaches the strong model is

$$\text{strong share} = h\,s + (1 - h)\,e ,$$

and the cost per question is the router's own cost plus the cheap and strong costs weighted by their shares. **Assumptions:** the two error rates do not depend on the input, and each model's accuracy on easy and hard inputs is fixed. `ch4_cost.py` tabulates this with illustrative accuracies; the point it makes is that every point of lost recall costs accuracy and every point of false escalation costs money, so you cannot judge a router by "accuracy" alone. You need its confusion matrix.

> [!DEFINITION] Confusion matrix (of a router)
> A table that counts, for a labelled set of inputs, where each kind of input was sent: hard inputs sent to the strong model (right), hard inputs sent to the cheap one (lost accuracy), easy inputs sent to the strong one (wasted money), easy inputs sent to the cheap one (right). "Hard" is best defined by outcome: the cheap handler gets it wrong.

### Our experiment: a cascade and a classifier router

`ch4_router.py` answers 24 questions with exact answers (a mixed set standing in for the help desk: eight lookups, eight multi-step calculations, eight trickier counting and date problems) in five ways: always-cheap, always-strong, a cascade (two cheap samples; if they agree, accept, otherwise ask the strong model), a classifier router (one cheap call labels the question easy or hard), and an oracle router that knows which questions the cheap model gets right. "Cheap" is gpt-5-mini at minimal reasoning effort; "strong" is **the same model at medium effort**, so the price per token is identical and the difference is the hidden reasoning tokens the medium setting spends. That keeps the comparison on one verified price, and it means our quality gap is smaller than two different models would show. The five conditions differ only in the routing policy.

Grading is typed and exact. An earlier draft of this book graded by substring, which gives "184" credit for 84; `ch4_grade.py` instead requires exactly one number and compares it numerically, compares times as minutes, and compares words after explicit normalisation. It prints its own test on adversarial strings:

[![Terminal output of ch4_grade.py: nineteen adversarial cases, including 184 against gold 84 graded False, 13 against 3 False, It is not 84; the answer is 74 against 84 False because it contains two numbers, The answer is 84 True, 84.0 True, 1,200 g against 1200 True, 14.25 against 14.20 False, 2496 with working shown False, Saturday or Sunday False, Lisbon, Portugal False; 19 of 19 cases behave as specified](/img/agents/ch4_grade-run.png)](/img/agents/ch4_grade-run.png)

```python
# code/agents/ch4_router.py (abridged): the two routing policies, computed from the same samples so they differ only in the policy.
def cascade(r):                      # two cheap samples; accept if they agree, otherwise pay for the strong model
    if r['agree']:
        return r['cheap_ok'], [r['u_cheap'], r['u_cheap2']]
    return r['strong_ok'], [r['u_cheap'], r['u_cheap2'], r['u_strong']]

def router(r):                       # one cheap classification call decides before anything is answered
    if r['route'] == 'easy':
        return r['cheap_ok'], [r['u_router'], r['u_cheap']]
    return r['strong_ok'], [r['u_router'], r['u_strong']]
```

[![Terminal output of ch4_router.py: a per-question table of 24 rows with the cheap and strong answers, whether two cheap samples agreed and where the router sent each; then the summary: always-cheap 17 of 24, 71 percent, 0.048 dollars per thousand correct answers; always-strong 24 of 24 at 0.194; cascade 21 of 24, 88 percent, at 0.135; classifier router 17 of 24 at 0.125; oracle router 24 of 24 at 0.092; the cascade escalated 5 of 24 and the router sent 1 of 24 to the strong model; the confusion matrix against the oracle shows 7 questions that needed the strong model all sent to the cheap one](/img/agents/ch4_router-run.png)](/img/agents/ch4_router-run.png)

{{FIG:ch4_router_results|Results of the final run of ch4_router.py. The cascade recovered most of the strong model's accuracy at about two-thirds of its cost per correct answer; the classifier router sent only one question to the strong model, caught none of the seven that needed it, and ended with the cheap model's accuracy at more than twice its cost per correct answer.}}

Three results, each a general point.

**The classifier router failed in the way routers usually fail: it was overconfident.** Asked "would a small model handle this?", the cheap model said yes to 23 of 24 questions, including all seven it then got wrong. Its recall on the hard questions was zero, so it matched always-cheap's 71% while paying for an extra call on every question. A router's error is invisible in end-to-end accuracy until you build the confusion matrix, and here the matrix is the whole story.

**The cascade worked, and its check was weaker than it looked.** It escalated 5 of 24 questions and reached 88% (21 of 24) at $0.135 per thousand correct answers, against $0.194 for always-strong. But read its agreement check as a verifier, which `ch4_router_stats.py` does from the saved results. The check accepted 16 of the 17 right cheap answers ($$r$$ = 94%) and also 3 of the 7 wrong ones ($$f$$ = 43%): on "what is the sum of the digits of 2 to the power 20?", both cheap samples said 7 (the answer is 31). When a model is confidently wrong, it is wrong twice.

[![Terminal output of ch4_router_stats.py: p, the chance the cheap answer is right, is 17 of 24, 71 percent; r, the chance two samples agree given a right answer, 16 of 17, 94 percent; f, the false-accept rate, 3 of 7, 43 percent; precision, the chance an accepted answer is right, 16 of 19, 84 percent, matching the Bayes formula; three accepted but wrong answers listed](/img/agents/ch4_router_stats-run.png)](/img/agents/ch4_router_stats-run.png)

**The false-accept rate is not one minus precision.** Precision, the share of accepted answers that are right, depends on how often the cheap model is right in the first place. With $$p$$ the share of right cheap answers, Bayes' rule gives

$$P(\text{right} \mid \text{accepted}) = \frac{p\,r}{p\,r + (1-p)\,f}.$$

For our cascade, $$p = 0.71$$, $$r = 0.94$$ and $$f = 0.43$$ give 84%: 16 of the 19 accepted answers were right. The same check on a task where the cheap model is right 30% of the time would give much lower precision. **Assumptions:** $$r$$ and $$f$$ are properties of the check that do not change with the mix of questions, which is only roughly true.

> [!DEFINITION] False-accept rate and precision
> For any check that accepts or rejects an answer: the **false-accept rate** $$f$$ is the share of wrong answers it accepts; **precision** is the share of accepted answers that are right. They are different numbers, connected by Bayes' rule through the base rate $$p$$ of right answers. A check with a 20% false-accept rate gives 98% precision when 90% of answers are right and 68% when 30% are.

> [!WARNING]
> Twenty-four questions, one run, one model at two effort settings. The counts would move by one or two on another run (they did between our development runs). What generalises: build the confusion matrix, measure the check's $$r$$ and $$f$$ separately, and do not let the cheap model judge whether it is good enough without measuring how often it is right about that.

### The research behind routing

Routing between models of different price became a research topic in 2023, when API prices differed by two orders of magnitude. Lingjiao Chen, Matei Zaharia and James Zou at Stanford posted **FrugalGPT** in May 2023 and made the cascade the reference design.

> [!PAPER] Chen, Zaharia and Zou (2023), FrugalGPT · Figure 3 · page 7
> [![Figure 3 of the FrugalGPT paper: the learned cascade on the HEADLINES dataset sends financial news first to GPT-J, then if a scorer gives less than 0.96 to J1-L, then if the score is under 0.37 to GPT-4; an example where GPT-4 answers price up wrongly and the cascade answers price down correctly; a table where GPT-4 scores 0.857 accuracy at 33.1 dollars and FrugalGPT 0.872 at 6.5 dollars](/img/agents/ch4-frugal-figure3.png)](/img/agents/ch4-frugal-figure3.png)
>
> **Context:** arXiv 2305.05176, posted 9 May 2023. Twelve commercial APIs priced in March 2023, three classification and reading datasets. The cascade's acceptance check is a small trained scorer (a DistilBERT model), learned per dataset together with the order of models and the thresholds.
>
> **What it says:** on HEADLINES the learned cascade asks GPT-J first, escalates to J1-L if the score is below 0.96 and to GPT-4 if below 0.37; "FrugalGPT reduces the cost by 80%, while improves the accuracy by 1.5% compared to GPT-4". The paper's Table 3 reports cost savings of 59% to 98% at matched accuracy across the three datasets.
>
> **Why it matters:** the check is the design. FrugalGPT's scorer was trained on labelled examples of each task, which is why its acceptance is trustworthy; our two-sample agreement was not trained on anything, and it let through 3 of 7 wrong answers. A cascade is only as cheap as its check is accurate.

Two later papers trained routers that decide before any answer is generated. Microsoft's **Hybrid LLM** (Ding et al., ICLR 2024) routes between a small and a large model by predicted difficulty and reports "up to 40% fewer calls to the large model, with no drop in response quality". **RouteLLM** (Ong et al., UC Berkeley, Anyscale and Canva, June 2024) trains routers on human preference data from Chatbot Arena and reports cost reductions of "over 2 times" at matched quality. Both report their gains as curves of quality against the share of calls sent to the strong model, which is the honest way to show a router: a single accuracy number hides the trade.

<details>
<summary>Research notes: RouteLLM, Hybrid LLM and the FrugalGPT savings table (optional reading)</summary>

> [!PAPER] Ong et al. (2024), RouteLLM · Figure 1 · page 2
> [![Figure 1 of the RouteLLM paper: three plots of performance against the percentage of calls to GPT-4, from Mixtral at zero to GPT-4 at a hundred; several trained routers sit above the random baseline on GSM8K and MT Bench; on MMLU the call-performance threshold CPT 50 percent is marked at about 37 percent of calls, with the area under the curve shaded as average performance gain recovered](/img/agents/ch4-routellm-figure1.png)](/img/agents/ch4-routellm-figure1.png)
>
> **Context:** arXiv 2406.18665 by Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph Gonzalez, M Waleed Kadous and Ion Stoica, posted 26 June 2024 (ICLR 2025 version). Routers (matrix factorisation, a BERT classifier, a causal model) are trained to predict whether the strong model's answer would be preferred.
>
> **What it says:** each curve plots quality against the share of calls sent to GPT-4; a good router rises faster than the diagonal of random routing. CPT(50%) is the share of strong calls needed to recover half the quality gap.
>
> **Why it matters:** this is the right picture for any router: quality against strong-model share, with random routing as the baseline.

> [!PAPER] Ong et al. (2024), RouteLLM · Table 6 · page 9
> [![Table 6 of the RouteLLM paper: cost saving ratios of the best routers over GPT-4; MT Bench 3.66 at 95 percent of GPT-4 quality and 2.49 at CPT 80 percent; MMLU 1.41 and 1.14; GSM8K 1.49 and 1.27; the text estimates GPT-4 at 24.7 dollars and Mixtral at 0.24 dollars per million tokens](/img/agents/ch4-routellm-table6.png)](/img/agents/ch4-routellm-table6.png)
>
> **Context:** cost savings computed from the share of GPT-4 calls relative to random routing, at early-2024 prices.
>
> **What it says:** "cost savings of up to 3.66x" on MT Bench; on MMLU and GSM8K, 1.14x to 1.49x.
>
> **Why it matters:** the savings depend heavily on how well the training data matches the traffic (the paper's Table 5 measures that match). On the maths benchmark, the saving at 80% of GPT-4's quality was 1.27x.

> [!PAPER] Chen, Zaharia and Zou (2023), FrugalGPT · Table 3 · page 8
> [![Table 3 of the FrugalGPT paper: cost to reach the best individual model's accuracy; HEADLINES GPT-4 33.1 dollars against FrugalGPT 0.6, a 98.3 percent saving; OVERRULING GPT-4 9.7 against 2.6, 73.3 percent; COQA GPT-3 72.5 against 29.6, 59.2 percent; the text says savings range from 50 to 98 percent](/img/agents/ch4-frugal-table3.png)](/img/agents/ch4-frugal-table3.png)
>
> **Context:** cost to match the best single model's accuracy on each dataset.
>
> **What it says:** savings of 98.3%, 73.3% and 59.2%, "which range from 50% to 98%".
>
> **Why it matters:** the largest saving is on HEADLINES, a classification task where small models are often right; the smallest is on reading comprehension. Savings track how often the cheap model is good enough, which is the $$p$$ in the Bayes formula above.

> [!PAPER] Ding et al. (2024), Hybrid LLM · Abstract · page 1
> [![Abstract of the Hybrid LLM paper with highlighted phrases: router that assigns queries to the small or large model; up to 40 percent fewer calls to the large model, with no drop in response quality](/img/agents/ch4-hybrid-abstract.png)](/img/agents/ch4-hybrid-abstract.png)
>
> **Context:** arXiv 2404.14618 by Dujian Ding and colleagues at the University of British Columbia and Microsoft, posted 22 April 2024, ICLR 2024. The router is a DeBERTa model trained on predicted quality gaps; the quality threshold can be changed at run time.
>
> **What it says:** a "router that assigns queries to the small or large model based on the predicted query difficulty"; "up to 40% fewer calls to the large model, with no drop in response quality".
>
> **Why it matters:** the tunable threshold turns the router into a dial between cost and quality that can be moved per customer or per hour.

> [!PAPER] Ding et al. (2024), Hybrid LLM · Table 1 · page 9
> [![Table 1 of the Hybrid LLM paper: quality drop, as a percentage of BART score against sending everything to the large model, at 10, 20 and 40 percent cost advantage for three model pairs; for Llama-2 7b and 13b the drop is 0.0 to 0.2; for Llama-2 13b and GPT-3.5-turbo 0.8 to 3.5; for FLAN-t5 800m and Llama-2 13b 2.1 to 13.8](/img/agents/ch4-hybrid-table1.png)](/img/agents/ch4-hybrid-table1.png)
>
> **Context:** "cost advantage" is the share of queries routed to the small model.
>
> **What it says:** when the two models are close (Llama-2 7b and 13b), routing 40% to the small one costs almost nothing (0.0 to 0.2%); when they are far apart (an 800-million-parameter FLAN-T5 against Llama-2 13b), the same 40% costs 10 to 14%.
>
> **Why it matters:** routing saves most when the cheap model is nearly as good on most inputs. If the gap is large, the router must be very good to save anything.

AWS's prescriptive guidance on agentic patterns (July 2025) lists the same use cases from the practitioner's side: "triaging requests across a variety of tasks", inputs that "must be preprocessed or normalized before entering more specialized workflows", and an agent "acting as a conversational switchboard".

</details>

### Routing in practice

| | Classifier router (decide first) | Cascade (answer cheaply, then check) |
|---|---|---|
| Pros | one decision, then a specialised handler with its own prompt, model and eval set; can send "unknown" to a person | needs no labelled routing data to start; the strong model is paid for only on escalations |
| Cons | its mistakes are silent unless you build its confusion matrix; a model asked "is this hard?" tends to say no | every input pays for at least one cheap call; a weak check accepts confident wrong answers |
| When to pick it | inputs fall into distinct kinds with different prompts or tools | one kind of task, cheap and strong models of the same family, and a check you have measured |
| Production failure | a new kind of input is forced into the nearest branch | the cheap model drifts and the check keeps accepting |

**What to measure before you decide**

| Measurement | How | What it tells you |
|---|---|---|
| Router confusion matrix | label a few hundred inputs by outcome (did the cheap handler get it right?) | lost accuracy and wasted cost, separately |
| Check $$r$$ and $$f$$ | for a cascade, how often the check accepts right and wrong cheap answers | precision via Bayes; whether the cascade is safe |
| Strong-model share | the share of traffic escalated | cost per thousand requests, and how it moves as traffic changes |
| Unknown-bucket rate | inputs the router sends to a person or a fallback | coverage; a rising rate is new kinds of input arriving |

> [!TIP] In production
> Define "hard" by outcome, not by how a question looks: label inputs by whether the cheap handler got them right, and train or prompt the router on those labels. Keep an explicit "unknown" branch that goes to a person or a safe default, and route by code wherever a field already decides (a ticket from the billing page is a billing ticket).

> [!DEFINITION] Unknown bucket
> An explicit branch for inputs the router cannot place with confidence, handled by a person or a safe default. Without it, every new kind of input is forced into the nearest existing branch.

### Discussion

1. **Rule, classifier or model?** A rule is free and exact where a field decides; a trained classifier is cheap and measurable; a model handles the long tail. Most production routers are all three in that order.
2. **Can the cheap model be its own router?** Ours said yes to 23 of 24 questions. My view: only if you have measured its self-assessment against outcomes; a separately trained scorer, as in FrugalGPT, is usually better.
3. **Where does a router go in a chain?** Usually first, so each branch can be a simpler chain. A second router deep in a chain is often a sign the first one is too coarse.

## 4.4 Parallelisation: split the work, or do it several times

### The example: two different reasons to make several calls at once

Before a refund is approved, Northwind wants three checks on the customer's message: does the refund fit the returns policy, is there a fraud signal, is the message abusive? They are independent, so there is no reason to run them one after another. That is **sectioning**. Separately, the shop's price calculations ("9 pens at 3 for £2.40 and 4 notebooks at £1.75, paid with a £20 note: what change?") are sometimes wrong; asking several times and taking the most common answer might help. That is **voting**.

> [!DEFINITION] Sectioning and voting
> Two forms of parallelisation in Anthropic's post. **Sectioning** splits a task into independent parts handled by different prompts at the same time and merges the results. **Voting** runs the same task several times and aggregates the answers: by the most frequent answer (a plurality vote), by a verifier that checks each candidate, or by a judge that picks the best.

> [!DEFINITION] Fan-out and fan-in
> Starting several calls at once from one point in the workflow (fan-out), then waiting for all of them and combining their results (fan-in). Latency becomes the slowest branch plus the merge; cost stays the sum of every branch.

{{FIG:ch4_parallel|Sectioning against voting. Sectioning splits a task into independent parts with different prompts and merges them; voting runs the same prompt several times and aggregates. Both run concurrently, so they buy time, not tokens.}}

### The arithmetic: what voting can and cannot do

Suppose each sample is right with probability $$p$$, and when it is wrong it picks one of $$m$$ different wrong answers equally often. A plurality vote returns the most frequent answer. It does not need a majority: a right answer given 40% of the time beats two wrong answers given 30% each. `ch4_cost.py` computes the exact probability that the plurality is right:

| right answer $$p$$ | distinct wrong answers $$m$$ | n = 1 | n = 5 | n = 11 |
|---|---|---|---|---|
| 0.4 | 1 (binary) | 40% | 32% | 25% |
| 0.4 | 2 | 40% | 45% | 50% |
| 0.4 | 5 | 40% | 58% | 76% |
| 0.6 | 2 | 60% | 77% | 90% |

**Assumptions:** samples are independent, $$p$$ is constant, and wrong answers spread evenly. Two consequences follow. Voting helps when the right answer is the most frequent one, and it helps more when wrong answers scatter; it hurts when one wrong answer is more common than the right one (the binary row). And real samples are not independent: if, with probability $$\rho$$, all samples copy one shared misreading of the problem, accuracy becomes $$\rho\,p + (1-\rho)\,\text{plurality}$$, which for $$p = 0.4$$, $$m = 2$$, $$n = 5$$ falls from 45% at $$\rho = 0$$ to 42% at $$\rho = 0.6$$. Correlated errors are why voting gains are usually smaller than the formula promises.

With a **verifier** that can check each candidate (a test, a calculation, a schema), you do not need the most frequent answer, only one that passes. If each sample passes with probability $$p$$, at least one of $$n$$ passes with probability $$1 - (1-p)^n$$, under the same independence assumption and a verifier that never accepts a wrong answer.

### Our experiment: voting, best-of-n with a verifier, and wall-clock time

`ch4_parallel.py` measures all three with the real model. **Voting**: ten multi-step word problems with one exact answer (Northwind's price-calculation kind), five samples each, plurality vote over the first 1, 3 and 5. **Best-of-n with a verifier**: ten "Game of 24" puzzles (combine four numbers with + − × ÷ into 24), chosen because a program can check any candidate exactly; a puzzle counts as solved if any of the first $$n$$ candidates verifies. **Latency**: five samples run one after another, then on five threads. The run cost under a cent.

```python
# code/agents/ch4_parallel.py (abridged): the same five calls, one after another and then fanned out on threads.
t0 = time.time()
for _ in range(N):
    llm.ask(P24_PROMPT.format(nums=nums), max_tokens=200)          # sequential: latency is the sum
seq = time.time() - t0
t0 = time.time()
with ThreadPoolExecutor(max_workers=N) as ex:                      # fan-out: latency is the slowest call
    list(ex.map(lambda _: llm.ask(P24_PROMPT.format(nums=nums), max_tokens=200), range(N)))
par = time.time() - t0                                             # the token bill is identical in both cases
```

[![Terminal output of ch4_parallel.py: ten word problems with five samples each and the plurality vote at n of 1, 3 and 5, accuracy 40 percent at every n while cost rises to 5 times; ten Game of 24 puzzles with the verified samples marked, solved 10, 30 and 30 percent at n of 1, 3 and 5 against a per-sample pass rate of 18 percent, which would predict 45 and 63 percent if samples were independent; wall-clock time of five samples, 4.6 seconds one by one against 1.0 seconds on five threads, a 4.5 times speedup](/img/agents/ch4_parallel-run.png)](/img/agents/ch4_parallel-run.png)

{{FIG:ch4_parallel_results|Results of the final run of ch4_parallel.py. The plurality vote stayed at 40% from one sample to five; best-of-n with a verifier rose from 10% to 30%, well below the 63% that independent samples would give. Fanning five calls out on threads cut wall-clock time from 4.6 to 1.0 seconds for the same tokens.}}

**Voting did not help, and the samples show why.** On the four problems the model solved, all five samples agreed (or four of five); on the six it missed, the samples scattered over wrong answers ("3.2", "6.15", "3.25", "5.2", "5.6" for a right answer of 5.80) or agreed on the same wrong one ("80" four times out of five for a right answer of 60). In neither case was the right answer the most frequent. Voting amplifies a model that is usually right on that question; it cannot create an answer the model rarely produces.

**Best-of-n with a verifier helped, by less than independence predicts.** Across all fifty candidates, 18% verified. If samples were independent, three would solve 45% of puzzles and five would solve 63%; we measured 30% and 30%. The verified candidates clustered on the same three puzzles, and on seven puzzles no candidate verified at all. Samples from one model at one temperature share their blind spots.

**Fan-out bought time, not tokens.** Five calls took 4.6 seconds one by one and 1.0 second on five threads, a 4.5 times speedup, for the same bill. For sectioning that is the whole point; for voting it means the latency of $$n$$ samples is about the latency of one.

> [!WARNING]
> Ten problems and ten puzzles, at minimal reasoning effort, where the model is often wrong; a stronger setting would change the absolute numbers. The conditions within each part differ only in $$n$$. What this run supports: measure agreement and per-sample pass rates before paying for votes, and expect correlated failures to cut the gain.

### The research behind parallel sampling

The research line runs from self-consistency (Chapter 3) through two 2024 papers that scaled sampling far further. Bradley Brown and colleagues at Stanford, Oxford and Google DeepMind asked in "Large Language Monkeys" (July 2024) what happens with hundreds or thousands of samples per problem.

> [!PAPER] Brown et al. (2024), Large Language Monkeys · Figure 7 · page 10
> [![Figure 7 of the Large Language Monkeys paper: success rate against number of samples from one to ten thousand for Llama-3 8B and 70B on GSM8K and MATH; coverage, the share of problems solved by at least one sample, rises towards 1.0, while majority vote, reward model best-of-N and reward model majority vote flatten between about 0.4 and 0.9 and stop improving before a hundred samples](/img/agents/ch4-monkeys-figure7.png)](/img/agents/ch4-monkeys-figure7.png)
>
> **Context:** arXiv 2407.21787, posted 31 July 2024. "Coverage" is the share of problems solved by at least one sample, which is what you get with a perfect verifier.
>
> **What it says:** coverage keeps rising with samples, towards nearly 100%, but "all sample selection methods fail to reach the coverage upper bound and saturate before reaching 100 samples". On MATH with Llama-3-8B, coverage approaches 1.0 while the selection methods stay near 0.4.
>
> **Why it matters:** the gap between the top line and the others is the value of a real verifier. Where answers can be checked automatically (code against tests, proofs, our Game of 24), more samples keep paying; where they cannot, voting and learned scorers stop improving early. The abstract reports that on SWE-bench Lite, issues solved rose "from 15.9% with one sample to 56% with 250 samples" because tests could verify candidates.

Two other papers fill in the picture. "More Agents Is All You Need" (Li et al., Tencent, TMLR 2024) samples up to forty answers and votes, and finds gains that grow with task difficulty: with Llama2-13B, GSM8K accuracy went from 0.35 with one sample to 0.59 with forty, above Llama2-70B's single sample (0.54). **Universal Self-Consistency** (Chen et al., Google, November 2023) handles free-form answers, where exact voting is impossible, by asking a model to pick "the most consistent response based on majority consensus"; it matched exact-match voting on maths and execution-based voting on SQL without running the code.

<details>
<summary>Research notes: More Agents, Universal Self-Consistency and the Monkeys abstract (optional reading)</summary>

> [!PAPER] Brown et al. (2024), Large Language Monkeys · Abstract · page 1
> [![Abstract of the Large Language Monkeys paper with highlighted phrases: from 15.9 percent with one sample to 56 percent with 250 samples; plateau beyond several hundred samples](/img/agents/ch4-monkeys-abstract.png)](/img/agents/ch4-monkeys-abstract.png)
>
> **Context:** authors Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc Le, Christopher Ré and Azalia Mirhoseini.
>
> **What it says:** coverage "scales with the number of samples over four orders of magnitude"; in domains "where answers can be automatically verified, these increases in coverage directly translate into improved performance"; without verifiers, majority voting and reward models "plateau beyond several hundred samples".
>
> **Why it matters:** the same conclusion as our run, at a thousand times the scale: the verifier, not the vote, turns samples into accuracy.

> [!PAPER] Li et al. (2024), More Agents Is All You Need · Figure 2 · page 4
> [![Figure 2 of the More Agents paper: a query, alone or with prompts, is sent to several LLM agents in a sampling phase; their answers are combined by majority voting in a voting phase](/img/agents/ch4-moreagents-figure2.png)](/img/agents/ch4-moreagents-figure2.png)
>
> **Context:** arXiv 2402.05120 by Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu and Deheng Ye (Tencent), posted 3 February 2024, published in TMLR. The method, "Agent Forest", is sampling and voting, optionally on top of other prompting methods.
>
> **What it says:** two phases, sampling then voting; "an LLM agent refers to a single LLM or a multiple LLM-Agents collaboration framework".
>
> **Why it matters:** the "agents" in the title are independent samples; there is no interaction between them. It is voting, measured at scale.

> [!PAPER] Li et al. (2024), More Agents Is All You Need · Table 2 · page 6
> [![Table 2 of the More Agents paper: single-query against ensemble-of-40 accuracy for Llama2-13B, Llama2-70B, GPT-3.5-Turbo and GPT-4 on GSM8K, MATH, Chess, MMLU and HumanEval; Llama2-13B on GSM8K rises from 0.35 to 0.59, above Llama2-70B's single 0.54; the text reports gains of 12 to 24 percent on GSM8K and 1 to 4 percent on Chess](/img/agents/ch4-moreagents-table2.png)](/img/agents/ch4-moreagents-table2.png)
>
> **Context:** ensembles of 40 samples, averaged over ten runs.
>
> **What it says:** gains "range from 12% to 24% on the GSM8K", "from 6% to 10% on the MATH", but only "from 1% to 4% on the Chess".
>
> **Why it matters:** forty times the tokens for one to twenty-four points. The paper's own analysis finds gains largest at moderate difficulty, which matches the plurality table: voting needs the right answer to be the most common one.

> [!PAPER] Chen et al. (2023), Universal Self-Consistency · Figure 1 · page 2
> [![Figure 1 of the Universal Self-Consistency paper: a question is sent to an LLM several times; the responses are put into a USC prompt that asks the LLM to select the most consistent response based on majority consensus; the LLM returns the selected response](/img/agents/ch4-usc-figure1.png)](/img/agents/ch4-usc-figure1.png)
>
> **Context:** arXiv 2311.17311 by Xinyun Chen and colleagues at Google, posted 29 November 2023.
>
> **What it says:** the selector prompt asks a model to "select the most consistent response based on majority consensus" among the samples.
>
> **Why it matters:** a judge as the aggregator. It extends voting to free-form text (summaries, open answers), where no two answers are character-identical.

> [!PAPER] Chen et al. (2023), Universal Self-Consistency · Tables 1 and 2 · page 5
> [![Tables 1 and 2 of the Universal Self-Consistency paper: on GSM8K with PaLM 2-L, greedy 85.7, random selection 82.9, self-consistency 90.4 and USC 90.2; with gpt-3.5-turbo 73.4, 68.5, 78.5 and 77.8; on BIRD-SQL execution accuracy greedy 42.4, execution-based self-consistency 45.6 and USC 45.5](/img/agents/ch4-usc-tables.png)](/img/agents/ch4-usc-tables.png)
>
> **Context:** maths and text-to-SQL.
>
> **What it says:** USC is within about a point of exact voting on maths (90.2 against 90.4) and of execution-based voting on SQL (45.5 against 45.6), "without access to execution results".
>
> **Why it matters:** where a verifier exists (running the SQL), it is still at least as good and cheaper to trust; the judge earns its place where no verifier exists.

</details>

### Parallelisation in practice

| | Sectioning | Voting or best-of-n |
|---|---|---|
| Pros | latency of the slowest section, not the sum; each section has a focused prompt and its own eval | can lift accuracy when samples vary and the right answer is common, or when a verifier exists |
| Cons | sections must really be independent, or the merge has to reconcile them; cost is the sum | $$n$$ times the tokens; correlated errors cut the gain; a weak aggregator picks wrong |
| When to pick it | separate checks or separate parts of a document | exact answers with high disagreement between samples, or any task with a cheap verifier |
| Production failure | two sections disagree and the merge silently picks one | a vote that is confidently wrong because every sample made the same mistake |

**What to measure before you decide**

| Measurement | How | What it tells you |
|---|---|---|
| Section independence | does any section's output change another's? | whether you can fan out at all |
| Agreement rate | sample the same input five times | if samples agree, one is enough; if the right answer is rare, voting cannot help |
| Per-sample pass rate with a verifier | verify every candidate on a labelled set | the ceiling for best-of-n, and how far correlation pulls you below $$1-(1-p)^n$$ |
| Wall-clock time, sequential against concurrent | time both on real traffic | the speedup your provider's rate limits actually allow |

> [!TIP] In production
> Fan out with a concurrency limit and a timeout per branch, and decide in advance what the merge does when one branch fails (wait, use partial results, or stop). For voting, log every sample, not only the winner: the agreement rate is a free confidence signal, and a falling agreement rate is an early warning of drift.

### Discussion

1. **Is sectioning just a chain with the arrows removed?** Only when the sections do not depend on each other. If the fraud check needs the policy check's result, you have a chain, and running it in parallel creates a race.
2. **When is voting worth five times the tokens?** When samples disagree often and the right answer is usually the most common one. Measure both on a hundred inputs first; our run would have saved 80% of its voting cost by checking agreement first.
3. **Different prompts or the same prompt?** Varying the prompt or the model across samples reduces correlated errors, at the cost of comparability. Uber's code review (Section 4.8) uses different specialised assistants, which is sectioning, not voting.

## 4.5 Orchestrator-workers: when the split cannot be written in advance

### The example: a question nobody can decompose in advance

Northwind's head of operations asks: "Delivery complaints rose by about forty per cent last quarter. Why, and what should we change?" No fixed chain answers this. The work depends on what the first look finds: if complaints cluster in two regions, someone has to read those regions' courier notices; if they cluster on one product, someone has to read its returns data. The number and kind of subtasks are decided by the input and by early findings. That is the case for an **orchestrator**: a model that plans the split, hands each part to a worker, and combines what comes back.

> [!DEFINITION] Orchestrator and worker
> In the orchestrator-workers pattern, an **orchestrator** is a model call (or loop) that reads the task, decides how many subtasks there are and what each is, writes a brief for each, and later synthesises the results. A **worker** is a model call or a small agent that carries out one brief with its own context and tools and returns a result. The difference from sectioning is who decides the split: code there, the orchestrator here.

> [!DEFINITION] Brief (hand-off)
> The message an orchestrator sends a worker: the objective, the output format, the tools and sources to use, and the boundaries of the task. It is the only thing the worker knows about the larger job, so its quality decides whether workers duplicate each other or leave gaps.

> [!DEFINITION] Context isolation
> Giving each worker a fresh context that contains only its brief and its own tool results, not the orchestrator's history or other workers' work. It keeps each context small and focused, and it is also why workers cannot see each other's mistakes or findings unless the orchestrator passes them on.

{{FIG:ch4_orchestrator|Orchestrator-workers. The orchestrator plans and writes a brief per worker; workers run with their own contexts and tools, usually in parallel; a synthesis step combines their results, and the orchestrator can plan again and spawn more workers. Typical failures are vague briefs (duplicated work), too many workers for an easy task, and a synthesis that drops or contradicts findings.}}

### What the production evidence says

The clearest public account is Anthropic's description of the multi-agent system behind Claude's Research feature, published in June 2025.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025)
> [![A paragraph from the post: our internal evaluations show that multi-agent research systems excel especially for breadth-first queries that involve pursuing multiple independent directions simultaneously; a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2 percent on our internal research eval; for example, when asked to identify all the board members of the companies in the Information Technology S&P 500, the multi-agent system found the correct answers by decomposing this into tasks for subagents, while the single agent system failed with slow, sequential searches](/img/agents/ch4-anthropic-research-90.png)](/img/agents/ch4-anthropic-research-90.png)
>
> **Context:** published 13 June 2025 by Jeremy Hadfield, Barry Zhang and colleagues. A lead agent plans, saves its plan to memory, spawns subagents that search in parallel with their own context windows, and synthesises; a separate citation agent attributes claims to sources.
>
> **What it says (reported by Anthropic):** the multi-agent system "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval", and it excels at "breadth-first queries that involve pursuing multiple independent directions simultaneously". The post also reports that in their analysis of the BrowseComp benchmark, "token usage by itself explains 80% of the variance" in performance.
>
> **Why it matters (our reading):** the gain is on breadth-first research, where subtasks are independent and parallel; the internal eval is not public, so the 90.2% cannot be checked or compared across tasks. The 80% finding suggests that much of the benefit comes from spending more tokens in parallel, which leads directly to the cost.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on cost
> [![A paragraph from the post: there is a downside, in practice these architectures burn through tokens fast; in our data, agents typically use about 4 times more tokens than chat interactions, and multi-agent systems use about 15 times more tokens than chats; for economic viability, multi-agent systems require tasks where the value of the task is high enough to pay for the increased performance; domains that require all agents to share the same context or involve many dependencies between agents are not a good fit; most coding tasks involve fewer truly parallelizable tasks than research](/img/agents/ch4-anthropic-research-tokens.png)](/img/agents/ch4-anthropic-research-tokens.png)
>
> **Context:** the paragraph right after the performance claims.
>
> **What it says (reported by Anthropic):** "agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats"; they need "tasks where the value of the task is high enough"; and tasks that "require all agents to share the same context or involve many dependencies between agents are not a good fit", which includes "most coding tasks".
>
> **Why it matters (our reading):** this is the cost side stated by the people selling the tokens. Use the pattern where subtasks are independent and the answer is worth fifteen chats, not as a default.

The same post lists what went wrong early, in terms every orchestrator builder will recognise: agents "spawning 50 subagents for simple queries", "scouring the web endlessly for nonexistent sources", and subagents that "duplicated work" because the lead agent's briefs were short ("research the semiconductor shortage"). The fixes were in the orchestrator's prompt: briefs with "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries", and explicit effort rules ("simple fact-finding requires just 1 agent with 3-10 tool calls ... complex research might use more than 10 subagents"). Running 3 to 5 subagents in parallel, each making several tool calls in parallel, "cut research time by up to 90% for complex queries".

### The cost arithmetic, with its assumptions

`ch4_cost.py` compares one agent doing $$k$$ subtasks in sequence with an orchestrator that writes $$k$$ briefs, runs $$k$$ workers concurrently and synthesises. With a 1,500-token prompt, 300-token results and a 400-token brief, the orchestrator sends 20,600 input tokens at $$k = 8$$ against 29,700 for the single agent, and finishes in 25 seconds against 72.

{{FIG:ch4_orch_cost|Input tokens and latency per task for one agent doing k subtasks in sequence and for an orchestrator with k concurrent workers, from ch4_cost.py. Under these assumptions the tokens are similar and the orchestrator is faster by the width of the fan-out; real workers that run their own tool loops multiply the token count.}}

**Assumptions, and why they matter:** the single agent replays its whole growing history at every step (no compaction, no prompt caching, both of which cut its cost a lot); each worker makes exactly one call; workers run concurrently; the orchestrator does not re-plan. Real workers usually run a tool-using loop of their own, which is where Anthropic's "about 15×" comes from. So the arithmetic says something narrower than "orchestrators are cheap": orchestration itself adds little; what costs is many workers each doing real work, and what you buy is time and breadth.

### The research behind orchestrators

The pattern appeared in research in 2023. **HuggingGPT** (Shen et al., Zhejiang University and Microsoft Research Asia, March 2023) used a language model as a controller over expert models from the Hugging Face hub in four stages: task planning, model selection, task execution and response generation. **AutoGen** (Wu et al., Microsoft Research and three universities, August 2023) made multi-agent conversation a programming framework. **Magentic-One** (Fourney et al., Microsoft Research, November 2024) is the most carefully engineered of the three, and its orchestrator keeps two explicit records.

> [!PAPER] Fourney et al. (2024), Magentic-One · Figure 2 · page 5
> [![Figure 2 of the Magentic-One paper: an Orchestrator with an outer loop that creates or updates a Task Ledger of given or verified facts, facts to look up, facts to derive, educated guesses and a task plan, and an inner loop that updates a Progress Ledger asking whether the task is complete, whether there are unproductive loops, whether progress is being made, who speaks next and with what instruction; if progress stalls more than twice, the outer loop re-plans; the orchestrator directs four agents: Coder, ComputerTerminal, WebSurfer and FileSurfer](/img/agents/ch4-magentic-figure2.png)](/img/agents/ch4-magentic-figure2.png)
>
> **Context:** arXiv 2411.04468 by Adam Fourney and colleagues at Microsoft Research, posted 7 November 2024. The workers are a coder, a terminal that runs the coder's code, a web browser and a file reader.
>
> **What it says:** the outer loop maintains a **task ledger** (facts, guesses, plan); the inner loop maintains a **progress ledger** that asks, every step, whether the task is complete, whether the team is looping, whether progress is being made, and who acts next. If progress stalls more than twice, the orchestrator re-plans.
>
> **Why it matters:** the progress ledger is a set of stop and stall checks made explicit, which is what Anthropic's early failures (endless searching, too many subagents) lacked. The paper's Table 1 reports 38.0% on the GAIA test set with GPT-4o and o1 against 92.0% for people, so this is a careful design on hard tasks, not a solved problem.

<details>
<summary>Research notes: HuggingGPT, Magentic-One's ledgers and results, AutoGen, and Anthropic's prompting rules (optional reading)</summary>

> [!PAPER] Shen et al. (2023), HuggingGPT · Figure 2 · page 3
> [![Figure 2 of the HuggingGPT paper: a request to generate an image of a girl reading a book in the pose of a boy in an example image and describe it aloud goes through four stages: task planning produces six tasks with dependencies, pose detection, pose-to-image, image classification, object detection, image-to-text and text-to-speech; model selection picks a Hugging Face model for each; task execution runs them on hybrid endpoints; response generation summarises the results](/img/agents/ch4-hugginggpt-figure2.png)](/img/agents/ch4-hugginggpt-figure2.png)
>
> **Context:** arXiv 2303.17580 by Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu and Yueting Zhuang, posted 30 March 2023, NeurIPS 2023.
>
> **What it says:** four stages: "Task planning: LLM parses the user request into a task list and determines the execution order and resource dependencies among tasks"; model selection; task execution; response generation.
>
> **Why it matters:** the plan is a task graph with dependencies (`<resource-2>`), so independent tasks can run in parallel, the same idea as Chapter 3's LLMCompiler.

> [!PAPER] Shen et al. (2023), HuggingGPT · Abstract · page 1
> [![Abstract of the HuggingGPT paper with highlighted phrases: act as a controller to manage existing AI models; conduct task planning when receiving a user request](/img/agents/ch4-hugginggpt-abstract.png)](/img/agents/ch4-hugginggpt-abstract.png)
>
> **Context:** the abstract.
>
> **What it says:** a language model can "act as a controller to manage existing AI models", and the system will "conduct task planning when receiving a user request".
>
> **Why it matters:** "controller" is the orchestrator; the workers here are specialised models, not language models. The authors evaluate planning quality separately from execution (their Tables 2 to 8), which is the right instinct: an orchestrator's plan can be graded on its own.

> [!PAPER] Fourney et al. (2024), Magentic-One · Section 4.1 · page 6
> [![A passage from the Magentic-One paper: the outer loop maintains the task ledger, which contains the overall plan, while the inner loop maintains the progress ledger; the orchestrator pre-populates the task ledger with given or verified facts, facts to look up, facts to derive and educated guesses; the plan is used like chain of thought, as a hint; since the plan may be revisited, we force all agents to clear their contexts and reset their states after each plan update](/img/agents/ch4-magentic-ledgers.png)](/img/agents/ch4-magentic-ledgers.png)
>
> **Context:** the description of the two loops.
>
> **What it says:** the task ledger holds "given or verified facts, facts to look up ... facts to derive ... and educated guesses"; after each plan update "we force all agents to clear their contexts and reset their states".
>
> **Why it matters:** clearing worker contexts on re-plan is context isolation used as error recovery: a worker that went down a wrong path does not carry it into the new plan.

> [!PAPER] Fourney et al. (2024), Magentic-One · Table 1 · page 12
> [![Table 1 of the Magentic-One paper: task completion rates on GAIA, AssistantBench and WebArena for leaderboard systems, GPT-4, humans and Magentic-One; humans 92.00 on GAIA and 78.2 on WebArena; Magentic-One with GPT-4o 32.33 on GAIA and 32.8 on WebArena; with GPT-4o and o1, 38.00 on GAIA and 27.7 accuracy on AssistantBench](/img/agents/ch4-magentic-table1.png)](/img/agents/ch4-magentic-table1.png)
>
> **Context:** test-set results with 95% error bars; underlined entries are statistically comparable to Magentic-One.
>
> **What it says:** 38.00 ± 5.5 on GAIA with GPT-4o and o1, statistically comparable to the leaderboard leaders of the time; humans score 92.00.
>
> **Why it matters:** competitive and far from human. The paper's ablations (its Figure 3) show that replacing the ledger-based orchestrator with a simple "who speaks next" selector lowers performance, which is evidence that the explicit progress checks matter.

> [!PAPER] Wu et al. (2023), AutoGen · Tables 5 and 6 · page 28
> [![Tables 5 and 6 of the AutoGen paper: on 12 manually crafted tasks, successes for a two-agent system, a group chat with role-play speaker selection and a group chat with task-based selection; with GPT-4, 9, 11 and 8; average LLM calls and termination failures with GPT-3.5-turbo, 9.9 calls and 9 failures for two agents against 5.3 and 0 for the group chat](/img/agents/ch4-autogen-tables.png)](/img/agents/ch4-autogen-tables.png)
>
> **Context:** arXiv 2308.08155 by Qingyun Wu and colleagues at Microsoft Research, Penn State, the University of Washington and Xidian University, posted 16 August 2023. A pilot study of a "dynamic group chat" in which a manager agent selects the next speaker.
>
> **What it says:** with GPT-4, the group chat solved 11 of 12 tasks against 9 for two agents; with GPT-3.5-turbo, the two-agent design failed to terminate 9 times in 12, the group chat none.
>
> **Why it matters:** twelve tasks is a pilot, and the authors present it as one. The termination failures are the interesting number: a coordination design without an explicit stop check can fail to end, which is what Magentic-One's progress ledger addresses.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on briefs
> [![A paragraph from the post: teach the orchestrator how to delegate; each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries; without detailed task descriptions, agents duplicate work, leave gaps, or fail to find necessary information; short instructions like research the semiconductor shortage were vague enough that subagents misinterpreted the task or performed the exact same searches; one subagent explored the 2021 automotive chip crisis while 2 others duplicated work investigating current 2025 supply chains](/img/agents/ch4-anthropic-research-delegate.png)](/img/agents/ch4-anthropic-research-delegate.png)
>
> **Context:** the second of the post's principles for prompting agents.
>
> **What it says:** "each subagent needs an objective, an output format, guidance on the tools and sources to use, and clear task boundaries"; without them "agents duplicate work, leave gaps, or fail to find necessary information".
>
> **Why it matters:** the brief is the interface between the orchestrator and a worker, and it deserves the same care as a schema between chain steps. The same post notes that the lead agent waits for each set of subagents ("synchronous execution creates bottlenecks") and that deployments must not break agents mid-run, which Section 4.7 returns to.

</details>

### Orchestrator-workers in practice

| | Fixed sectioning | Orchestrator-workers |
|---|---|---|
| Pros | predictable cost and latency; each section testable | handles tasks whose split depends on the input or on early findings; breadth in parallel |
| Cons | cannot adapt the split | cost is the sum of every worker's loop (Anthropic reports about 15 times a chat); plans and briefs can be wrong; harder to test |
| When to pick it | the parts are known in advance | open-ended, breadth-first, high-value tasks with independent sub-questions |
| Production failure | a part nobody planned for | duplicated or contradictory workers, runaway spawning, a synthesis that drops findings |

**What to measure before you decide**

| Measurement | How | What it tells you |
|---|---|---|
| Plan quality | grade a sample of plans before execution | whether the orchestrator decomposes well, separately from execution |
| Worker overlap | compare workers' queries or sources per task | duplicated work from vague briefs |
| Workers and tokens per task | from traces, against task difficulty | whether effort scales with the question (Anthropic's early failure) |
| Synthesis fidelity | check that each worker's key finding appears in the answer | lost or contradicted findings |
| Value per task | what a good answer is worth against its token cost | whether the pattern pays at all |

> [!TIP] In production
> Cap workers per task and tool calls per worker in code, not only in the prompt. Give the orchestrator effort rules tied to the question type. Make workers write results to storage and return references, so the synthesis reads the originals rather than a summary of a summary (Anthropic's post calls the alternative "a game of telephone").

### Discussion

1. **Is orchestrator-workers a workflow or an agent?** The orchestrator decides the split, so it is the most agent-like of the five. In practice the outer frame (plan, fan out, synthesise, stop) is code and the inside is the model's, which is where the testing burden lands.
2. **Should workers share context?** Isolation keeps contexts small and stops one worker's error spreading; sharing lets workers build on each other. My view: isolate by default and pass findings explicitly through the orchestrator, because that path is traceable.
3. **When is the 15-times cost worth it?** When a wrong or slow answer is expensive and the question is breadth-first. A monthly operations question at Northwind qualifies; answering each support ticket does not.

## 4.6 Evaluator-optimiser: generate, check, refine, stop

### The example: rewriting Northwind's weekly update

Every week Northwind's team leads write paragraphs about their work, and the internal-communications editor wants them rewritten for the all-staff email under strict rules: 55 to 75 words, exactly four sentences, no sentence over 20 words, three named keywords, and none of ten banned words ("very", "basically", "leverage" and so on). A single call often misses one rule. The **evaluator-optimiser** pattern adds a second role: something that checks the draft and sends back what is wrong, in a loop that stops by a rule.

> [!DEFINITION] Evaluator-optimiser
> A loop of two roles: a generator writes a draft; an evaluator scores it or lists its failures; the failures go back to the generator; the loop ends by a **stop rule**. The evaluator can be code (tests, a schema, a word count), a model (an "LLM judge"), a person, or a combination. Chapter 3 called the same structure reflection; here the point is that the evaluator is a separate step you can design and measure.

> [!DEFINITION] Stop rule
> The condition that ends a loop: the draft passes every check, a score clears a threshold, a maximum number of rounds is reached, or the same failure repeats (no progress). Without one, a loop that cannot succeed runs until the budget does.

> [!DEFINITION] LLM-as-a-judge
> Using a model to evaluate outputs: grading one answer against a rubric, or choosing the better of two. It can judge what code cannot (tone, helpfulness, faithfulness to a source), and it has known biases that must be measured before it is trusted to stop a loop.

{{FIG:ch4_evaluator|The evaluator-optimiser loop. The generator writes a draft; the evaluator runs code checks first and a judge model second; a stop rule decides; on failure the exact failures go back to the generator. Below, the stop rules in the order to apply them.}}

### The arithmetic of the stop, with an imperfect evaluator

If each round passes with probability $$p$$ and the evaluator is perfect, the chance of passing within $$k$$ rounds is $$1-(1-p)^k$$ and the cost per accepted draft is one round's cost divided by $$p$$, whatever $$k$$ is (Chapter 3 derived this). **Assumptions:** rounds are independent with the same $$p$$, each round costs the same, and the evaluator never errs. Feedback-driven rounds break the first assumption (feedback usually raises $$p$$ after the first round), so this is a floor on the benefit, not a forecast.

The evaluator is never perfect. Write $$r$$ for the chance it accepts a good draft and $$f$$ for the chance it accepts a bad one (its false-accept rate). The share of drafts the loop stops on that are actually good is, by Bayes' rule,

$$P(\text{good} \mid \text{stopped}) = \frac{p\,r}{p\,r + (1-p)\,f},$$

the same formula as the cascade's check in Section 4.3. With $$p = 0.9$$, $$r = 1$$ and $$f = 0.2$$ it gives 97.8%; with $$p = 0.3$$ and the same evaluator, 68%. **The harder the task, the more the evaluator's false-accept rate matters.** And a low $$r$$ (rejecting good drafts) has its own cost: more rounds, and rewrites of drafts that were already fine.

[![Terminal output of part 5 of ch4_cost.py: with a perfect critic, the chance of passing within k rounds and the expected cost, where cost per accepted draft equals one round's cost divided by p for every k; then the imperfect critic table: p good 0.9, r 1.00, f 0.20 gives 92 percent accepted and 97.8 percent precision; p 0.5 gives 83.3 percent; p 0.3 gives 68.2 percent; p 0.5 with r 0.9 and f 0.1 gives 90.0 percent](/img/agents/ch4_cost_stop-run.png)](/img/agents/ch4_cost_stop-run.png)

### Our experiment: a code checker against a judge model

`ch4_evaluator.py` rewrites eight source paragraphs under the five rules, in two loops from the same start, each allowed up to four rounds. In the **programmatic** loop a Python checker lists the rules each draft breaks, with measured values ("C3: a sentence of 24 words, limit 20"), and the loop stops when the list is empty. In the **judge** loop a separate model call marks each rule pass or fail, its verdict is fed back, and the loop stops when it says everything passes. Every draft in both loops is also scored by the other evaluator, so we can count how often the judge and the code disagree. Because the programmatic loop is graded by the same code that gives it feedback, we add a **held-out check** that no loop ever sees: two key facts per source paragraph (for example "staging" and "audit log" in the incident report) must survive the rewrite. The run cost about two cents.

```python
# code/agents/ch4_evaluator.py (abridged): one loop; arm decides who the critic is. The held-out fact check is applied once, at the end.
for rnd in range(1, MAX_ROUNDS + 1):
    text, ug = llm.ask(msgs, max_tokens=400)                  # generator
    viol = check(text, keywords)                              # programmatic checker: exact counts, never wrong about counting
    jfails, notes, uj = judge(text, keywords)                 # judge model: pass/fail per constraint
    critic_fails = viol if arm == 'programmatic' else jfails  # who decides whether to stop
    if not critic_fails:
        break
    feedback = viol if arm == 'programmatic' else [f'{c}: the reviewer marked this constraint as failed. {notes}' for c in jfails]
    msgs += [{'role': 'assistant', 'content': text}, {'role': 'user', 'content': REGEN.format(viol='\n- '.join(feedback))}]
```

[![Terminal output of ch4_evaluator.py: per-paragraph trails for both loops; then over 40 drafts the judge agreed with the checker 40 percent of the time, with 2 false passes and 22 false fails; agreement by constraint 32 percent on the word count, 90 on four sentences, 62 on the sentence length limit, 90 on keywords and 80 on banned words; summary: the programmatic loop stopped on 7 of 8, all 7 truly passing, mean 1.88 rounds; the judge loop stopped on only 4 of 8 after a mean of 3.12 rounds, although 7 of its final drafts passed every constraint; held-out facts kept 8 of 8 and 7 of 8; cost per round 0.0002 against 0.0005 dollars](/img/agents/ch4_evaluator-run.png)](/img/agents/ch4_evaluator-run.png)

{{FIG:ch4_evaluator_results|Results of the final run of ch4_evaluator.py. The judge agreed with the code checker on only 32% of word-count verdicts. In this run its errors were mostly false fails, so its loop ran 3.1 rounds on average against 1.9 and cost more than twice as much per round, and rewriting drafts that already passed cost one paragraph a key fact.}}

**The judge cannot count.** Over 40 drafts it agreed with the checker's overall verdict 40% of the time, with 2 false passes and 22 false fails. It was worst on the word count (32% agreement) and on the sentence-length rule (62%), the two rules that require counting, and best on keywords (90%), which require only looking.

**False fails are not harmless.** In this run the judge mostly rejected drafts that were fine. Its loop averaged 3.1 rounds against 1.9, five of its eight paragraphs hit the four-round cap, and with the judge's own call in every round it cost $0.0005 per round against $0.0002. Worse, rewriting a passing draft can damage it: on paragraph 5 the first draft met every rule, the judge rejected it three times, and the final draft had dropped one of the held-out facts (the hourly queries that drive the warehouse costs). In a development run the judge's errors leaned the other way, towards false passes; the direction varies, the unreliability does not.

**A false pass ships a broken draft.** On paragraph 1 the judge accepted a 52-word draft against a 55-word minimum after one round. The programmatic loop, whose checker cannot miscount, stopped on 7 of 8 paragraphs, and all 7 truly passed and kept their held-out facts. Its one failure was a paragraph about app crashes where the model kept writing one 24-word sentence across four rounds, the "no progress" case for which a stop rule should hand the draft to a person.

> [!WARNING]
> What this experiment isolates: the two loops differ only in who evaluates, from the same prompts and sources. It does not show that judges are useless. They were asked to count, which is the task where models are weakest and code is perfect. The finding is narrower and practical: do not use a model for checks code can make, and measure a judge's $$r$$ and $$f$$ on labelled drafts before letting it stop a loop. Eight paragraphs and one run; the judge's error direction changed between runs.

### The research behind judges

The reference study of model judges is Lianmin Zheng and colleagues' "Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena" (the LMSYS group at Berkeley and others, June 2023). It found strong judges agreeing with human preferences about as often as humans agree with each other, "over 80% agreement", and it catalogued the biases.

> [!PAPER] Zheng et al. (2023), Judging LLM-as-a-Judge · Table 2 · page 5
> [![Table 2 of the LLM-as-a-judge paper: position bias of three judges when the order of two answers is swapped; Claude-v1 with the default prompt is consistent in 23.8 percent of cases and biased toward the first answer in 75.0 percent; GPT-3.5 consistent in 46.2 percent and biased toward the first in 50.0; GPT-4 consistent in 65.0 percent and biased toward the first in 30.0](/img/agents/ch4-judge-table2.png)](/img/agents/ch4-judge-table2.png)
>
> **Context:** arXiv 2306.05685, posted 9 June 2023, NeurIPS 2023 Datasets and Benchmarks track. Each judge compares two similar answers, then the same two with their order swapped.
>
> **What it says:** a consistent judge gives the same verdict both ways. GPT-4 did so in 65.0% of cases, GPT-3.5 in 46.2% and Claude-v1 in 23.8%, which favoured whichever answer came first 75.0% of the time. The same section reports **verbosity bias** (a "repetitive list" padding attack fooled Claude-v1 and GPT-3.5 in 91.3% of cases and GPT-4 in 8.7%) and limited ability to grade maths without a reference (GPT-4 judged a wrong answer correct in 14 of 20 cases with the default prompt, 3 of 20 with a reference answer).
>
> **Why it matters:** each bias has a cheap mitigation the paper tests: swap positions and count only consistent verdicts, give the judge a reference answer, prefer single-answer grading against a rubric to vague comparisons. And grading maths is the counting problem in our experiment: a judge with a reference, or code, does it better.

> [!DEFINITION] Position, verbosity and self-enhancement bias
> Three measured tendencies of model judges: favouring the answer shown first (**position**), favouring the longer answer whatever its quality (**verbosity**), and favouring answers written by the same model (**self-enhancement**, which Zheng et al. found hints of but could not confirm). Each is tested by changing only the suspected cause and checking whether the verdict changes.

G-Eval (Liu et al., Microsoft, March 2023) showed how to make a judge more consistent: have the model write evaluation steps from the criteria, fill in a form, and weight the score by the probabilities of each grade. Its Spearman correlation with human ratings of summaries was 0.514 with GPT-4, ahead of every earlier automatic metric; the authors also warn that model evaluators may prefer model-written text. Chapter 3's caution applies here too: Huang et al. (2023) found that asking a model to correct its own reasoning without outside feedback often made it worse. A judge is outside feedback only to the extent that it sees something the generator did not, or is better than the generator at the specific check.

<details>
<summary>Research notes: the judge paper's abstract, G-Eval, and Anthropic's conditions for the pattern (optional reading)</summary>

> [!PAPER] Zheng et al. (2023), Judging LLM-as-a-Judge · Abstract · page 1
> [![Abstract of the LLM-as-a-judge paper with highlighted phrases: position, verbosity, and self-enhancement biases; achieving over 80 percent agreement](/img/agents/ch4-judge-abstract.png)](/img/agents/ch4-judge-abstract.png)
>
> **Context:** authors include Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Hao Zhang, Joseph Gonzalez and Ion Stoica, with colleagues at Berkeley, UC San Diego, Carnegie Mellon, Stanford and MBZUAI.
>
> **What it says:** strong judges "can match both controlled and crowdsourced human preferences well, achieving over 80% agreement, the same level of agreement between humans", and the paper examines "position, verbosity, and self-enhancement biases, as well as limited reasoning ability".
>
> **Why it matters:** 80% agreement on open-ended preference is a good result for preference; it says nothing about counting or checking facts, where our judge fell to 32%. Measure your judge on your check.

> [!PAPER] Liu et al. (2023), G-Eval · Figure 1 · page 2
> [![Figure 1 of the G-Eval paper: a task introduction and evaluation criteria are given to the model, which generates evaluation steps by chain of thought; the steps, the input context and the target summary are then scored in a form; the final score is the probability-weighted sum of the scores 1 to 5, here 2.59](/img/agents/ch4-geval-figure1.png)](/img/agents/ch4-geval-figure1.png)
>
> **Context:** arXiv 2303.16634 by Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu and Chenguang Zhu (Microsoft), posted 29 March 2023, EMNLP 2023.
>
> **What it says:** criteria become written evaluation steps; scoring is a form; the score is "the probability-weighted summation of the output scores".
>
> **Why it matters:** weighting by probability turns a judge's coarse integer grades into a continuous score with fewer ties, which makes thresholds in a stop rule less jumpy.

> [!PAPER] Liu et al. (2023), G-Eval · Table 1 · page 4
> [![Table 1 of the G-Eval paper: Spearman and Kendall-Tau correlations with human judgements on SummEval; ROUGE-1 averages 0.192, BERTScore 0.225, BARTScore 0.385, UniEval 0.474, G-Eval with GPT-3.5 0.401 and G-Eval with GPT-4 0.514](/img/agents/ch4-geval-table1.png)](/img/agents/ch4-geval-table1.png)
>
> **Context:** summary-level correlations with human ratings on four dimensions.
>
> **What it says:** G-Eval-4 averages 0.514 Spearman against 0.474 for UniEval, the best earlier metric, and 0.192 for ROUGE-1.
>
> **Why it matters:** a correlation of about 0.5 with humans is the best of its kind and still far from agreement on each item. A judge built this way is good for ranking systems on average; it is weaker as a gate on one draft.

> [!PAPER] Anthropic, "Building effective agents" (Engineering blog, December 2024), on evaluator-optimiser
> [![A paragraph from the post: When to use this workflow: this workflow is particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value; the two signs of good fit are, first, that LLM responses can be demonstrably improved when a human articulates their feedback; and second, that the LLM can provide such feedback](/img/agents/ch4-anthropic-evaluator.png)](/img/agents/ch4-anthropic-evaluator.png)
>
> **Context:** the post's description of the fifth workflow.
>
> **What it says:** the pattern is "particularly effective when we have clear evaluation criteria, and when iterative refinement provides measurable value"; the second sign of fit is "that the LLM can provide such feedback".
>
> **Why it matters:** our experiment is a case where the second sign fails for two of five criteria: the model cannot reliably tell whether a paragraph has 55 words. Code can, so the evaluator for those criteria should be code.

</details>

### Evaluator-optimiser in practice

| | Code evaluator (tests, schema, counts) | Model judge |
|---|---|---|
| Pros | exact, free, fast; never miscounts; specific feedback | judges what code cannot: tone, helpfulness, faithfulness to a source |
| Cons | only checks what you can write down | false passes and false fails; position and verbosity bias; one extra call per round |
| When to pick it | any rule you can express in code | qualities with no programmatic test, after measuring its $$r$$ and $$f$$ |
| Production failure | a check that tests the wrong thing, passed reliably | a judge that drifts with a model update and nobody re-measures it |

**What to measure before you decide**

| Measurement | How | What it tells you |
|---|---|---|
| Judge $$r$$ and $$f$$ | run the judge on a few hundred labelled drafts | the precision of the stop, via Bayes |
| Position and verbosity sensitivity | swap order; pad an answer without adding content | whether the judge's verdicts survive changes that should not matter |
| Rounds to pass, and the no-progress rate | from traces | where to cap rounds; how often to hand off to a person |
| Damage from extra rounds | held-out checks on drafts that passed early | whether false fails are degrading good work |
| Cost per round and per accepted draft | API usage per round | the price of each extra round |

> [!TIP] In production
> Put code checks first and the judge second, and send the judge only what code cannot check. Give the judge a reference answer or a rubric with examples, ask for a verdict per criterion, and log its reasons. Stop on pass, on a round cap of two to four, or on the same failure twice, and hand the best draft so far to a person with its failures attached.

### Discussion

1. **Is a judge outside feedback?** Partly. It sees the draft fresh, without the generator's reasoning, but it shares the model's blind spots if it is the same model. Using a different model as judge reduces shared blind spots; it does not remove the need to measure the judge.
2. **What should the loop optimise?** Whatever the evaluator checks, exactly as our gated chain did. If the evaluator is narrow, add held-out checks the loop never sees, as we did with the key facts.
3. **How many rounds?** Our programmatic loop passed most paragraphs by round two; most of the judge loop's extra rounds were spent rewriting drafts that already passed. A cap of two to four rounds, with a no-progress stop, is a reasonable start, then adjust from traces.

## 4.7 Composing patterns: the workflow as a state machine

### The example: Northwind's support workflow, all together

Real systems combine the patterns. Northwind's support workflow, built in full in `ch4_workflow.py`, puts a **router** in front (code rules first, a model call only when the rules do not decide), a **gated chain** behind it (extract, then draft, each schema-validated with one retry), a **policy check** in code, a **human approval** step for refunds over £50, and an idempotent **send**. Around those nodes sits the plumbing every production workflow needs: typed state, budgets, checkpoints, resume, and a trace. A larger version could hang an orchestrator off one branch (for "why are complaints up?" questions) and an evaluator loop before the send; the plumbing would not change.

{{FIG:ch4_compose|Patterns compose. A router sends known requests to a gated chain and open-ended ones to an orchestrator with parallel workers; an evaluator checks the result and a human-approval interrupt pauses the run. Below, durable execution: state is checkpointed after each node, a crash resumes from the last checkpoint, and the re-run step must be idempotent so that it cannot send or charge twice.}}

> [!DEFINITION] State machine
> A program described as a set of states (here: the node to run next, plus the data gathered so far) and the transitions between them. Writing a workflow this way means that "where is this ticket?" always has a one-word answer (`draft`, `awaiting_approval`, `done`), which is what makes pausing, resuming and tracing straightforward. Stripe's coding agents and Airbnb's test migration both describe their workflows this way (Section 4.8).

> [!DEFINITION] Durable execution
> Running a workflow so that its progress survives crashes and restarts: the state after each step is saved, and a restarted process resumes from the last saved step instead of starting again. Temporal is a service built for this; LangGraph's checkpointers and our checkpoint files are lighter versions of the same idea.

> [!DEFINITION] Idempotent step
> A step that has the same effect whether it runs once or several times. Resuming after a crash can re-run the step that was in progress, so any step with a side effect (sending an email, issuing a refund) must be idempotent, usually by attaching an **idempotency key** that the receiving system remembers and refuses to process twice.

> [!DEFINITION] Human-in-the-loop interrupt
> A node that pauses the workflow, saves its state, and waits for a person to approve, edit or reject before continuing. The run can wait for minutes or days, because the state lives in storage, not in a running process.

> [!PAPER] Temporal, "Understanding Temporal" (Temporal documentation, accessed October 2026)
> [![A paragraph from the Temporal docs: Temporal runs your application code as a Durable Execution; once a Workflow starts, it runs to completion, whether that takes a second or a year; if the process running it crashes, the Temporal Service hands the work to another process, which rebuilds the state of the execution and resumes at the point where it stopped, with local variables and progress intact](/img/agents/ch4-temporal-durable.png)](/img/agents/ch4-temporal-durable.png)
>
> **Context:** the overview page of Temporal, an open-source durable-execution engine. Workflows are ordinary code; each external call (an "Activity") is retried with a policy, and every step is recorded in an event history.
>
> **What it says:** "Once a Workflow starts, it runs to completion, whether that takes a second or a year. If the process running it crashes, the Temporal Service hands the work to another process, which rebuilds the state of the execution and resumes at the point where it stopped." The page notes that workflow code "must be deterministic" because it is replayed to rebuild state.
>
> **Why it matters:** an LLM workflow that waits for a human approval, or calls a model that sometimes times out, has the same needs as a payment workflow. The determinism rule has a direct consequence for LLM nodes: a model call is not deterministic, so its result must be recorded as a step's output and replayed from the record, never called again during replay.

### The complete program

The core of the runner is a loop over the current node, with a checkpoint after every node:

```python
# code/agents/ch4_workflow.py (abridged): the runner. Every node updates the typed State and names the next node.
def run(st, crash_after=None):
    try:
        while st.node not in ('done', 'needs_human', 'awaiting_approval'):
            t0 = time.time()
            if st.node == 'extract':
                obj, use, retried = extract(st)                      # schema-validated (pydantic), one retry with the errors
                if obj is None:
                    span(st, 'extract', 'gate failed twice -> person', t0, use); st.node = 'needs_human'
                else:
                    st.facts = obj.model_dump(); span(st, 'extract', 'ok', t0, use); st.node = 'draft'
            elif st.node == 'review':
                amt = st.facts.get('refund_amount_gbp') or 0
                if amt > REFUND_LIMIT and st.approved is None:
                    span(st, 'review', 'PAUSE for approval', t0); st.node = 'awaiting_approval'   # human-in-the-loop interrupt
                else:
                    span(st, 'review', 'policy ok', t0); st.node = 'send'
            ...                                                      # route, faq, draft and send follow the same shape
            checkpoint(st)                                           # state saved after every node: resume starts here
    except BudgetExceeded as e:                                      # at most 6 model calls and 8,000 tokens per ticket
        span(st, st.node, f'STOP: {e}', time.time()); st.node = 'needs_human'; checkpoint(st)
    return st
```

The demonstration runs six tickets, kills the process on purpose after ticket T3's draft is checkpointed, resumes every unfinished ticket from its checkpoint, has a person approve the paused refund, and then replays a send to show that it cannot happen twice. It made 12 model calls and cost $0.0015.

[![Terminal output of ch4_workflow.py: one trace line per node with a trace id, the node, the outcome, tokens and milliseconds; T1 is routed by rule, extracted, drafted, reviewed and sent; the run is killed after T3's draft is checkpointed; T4 is routed by the model to the FAQ workflow; T5's refund of 649 pounds pauses for approval; T6, in Spanish, is routed by the model; on resume T3 continues at the review node without re-running extract or draft; a person approves T5, which is then sent; replaying the send for T1 reports already sent, idempotency key seen; the final table shows every ticket done with two or three calls each, five replies in the outbox and 29 spans in the trace](/img/agents/ch4_workflow-run.png)](/img/agents/ch4_workflow-run.png)

Four behaviours in that output are the reasons the plumbing exists. **Resume skips finished work**: T3 continued at `review` with its two earlier calls already counted, and extract and draft were not run again. **The interrupt waits**: T5's £649 refund stopped at `awaiting_approval` with its state on disk, and continued from there after approval. **The side effect is idempotent**: replaying T1's send found its key in the outbox and did nothing. **The router used the model only when needed**: four tickets were routed by keyword rules at no cost; the Spanish ticket (T6) and the shipping question (T4) fell through the rules and cost one small call each.

<details>
<summary>The full program: code/agents/ch4_workflow.py (about 260 lines; optional reading)</summary>

```python
"""Chapter 4, Section 4.7: one complete, runnable composed workflow for Northwind Home's support tickets, with the real API
(gpt-5-mini, reasoning_effort=minimal). Everything the chapter recommends, in about 250 lines:
  typed state (a dataclass) and typed step outputs (pydantic schemas, validated in code);
  a router that is code first and asks the model only when the rules do not decide;
  a gated chain: extract -> draft, each step schema-validated, retried once with the exact errors, then handed to a person;
  a policy check in code (word limit, banned phrases, order id quoted) after the draft;
  a human-in-the-loop interrupt: refunds over GBP 50 pause the run until someone approves;
  an idempotent side effect: replies go to an outbox keyed by an idempotency key, so a resumed run cannot send twice;
  budgets per ticket (model calls, tokens) that stop the run instead of letting it spin;
  a checkpoint after every node, and resume from the last checkpoint after a crash;
  a trace: one JSON line per node (trace id, node, outcome, tokens, milliseconds) in results/ch4_workflow_trace.jsonl.
The demo processes six tickets, injects one crash after the draft step of ticket T3, resumes it, and approves the paused refund."""
import hashlib, json, os, re, time, uuid
from dataclasses import dataclass, field, asdict
from typing import Literal, Optional
from pydantic import BaseModel, Field, ValidationError
from common import Log, save, RESULTS
import ch4_llm as llm

log = Log('ch4_workflow')
llm.set_log(log)
CKPT = os.path.join(RESULTS, 'ch4_workflow_ckpt')
OUTBOX = os.path.join(RESULTS, 'ch4_workflow_outbox.json')
TRACE = os.path.join(RESULTS, 'ch4_workflow_trace.jsonl')
MAX_CALLS, MAX_TOKENS, REFUND_LIMIT = 6, 8000, 50.0
BANNED = ['24 hours', 'immediately', 'guarantee', 'today', 'right away', 'as soon as possible', 'shortly']


# ---------------------------------------------------------------- typed step outputs (the schemas the gates enforce)
class Facts(BaseModel):
    issue_type: Literal['billing', 'delivery', 'damaged_or_faulty', 'account', 'other']
    order_id: Optional[str] = Field(default=None, pattern=r'^ORD-\d{5}$')
    refund_amount_gbp: Optional[float] = Field(default=None, ge=0)


class Draft(BaseModel):
    reply: str = Field(min_length=20)


# ---------------------------------------------------------------- typed workflow state
@dataclass
class State:
    ticket_id: str
    text: str
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    node: str = 'route'                 # the next node to run; 'done' and 'needs_human' and 'awaiting_approval' are terminal or paused
    route: Optional[str] = None
    facts: Optional[dict] = None
    reply: Optional[str] = None
    approved: Optional[bool] = None
    calls: int = 0
    tokens: int = 0
    notes: list = field(default_factory=list)


class BudgetExceeded(Exception):
    pass


class InjectedCrash(Exception):
    pass


# ---------------------------------------------------------------- infrastructure: checkpoints, trace, model calls with budgets
def checkpoint(st):
    os.makedirs(CKPT, exist_ok=True)
    json.dump(asdict(st), open(os.path.join(CKPT, st.ticket_id + '.json'), 'w'), indent=1)


def load(ticket_id):
    p = os.path.join(CKPT, ticket_id + '.json')
    return State(**json.load(open(p))) if os.path.exists(p) else None


def span(st, node, outcome, t0, use=None):
    rec = {'trace_id': st.trace_id, 'ticket': st.ticket_id, 'node': node, 'outcome': outcome, 'ms': round((time.time() - t0) * 1000),
           'tokens': (use or {}).get('input', 0) + (use or {}).get('output', 0)}
    open(TRACE, 'a').write(json.dumps(rec) + '\n')
    log(f'  [{st.trace_id}] {st.ticket_id} {node:<9} {outcome:<34} {rec["tokens"]:>5} tok {rec["ms"]:>6} ms')


def call(st, prompt):
    if st.calls >= MAX_CALLS or st.tokens >= MAX_TOKENS:
        raise BudgetExceeded(f'budget: {st.calls} calls, {st.tokens} tokens')
    text, use = llm.ask(prompt, max_tokens=600)
    st.calls += 1
    st.tokens += use['input'] + use['output']
    return text, use


def validated(st, prompt, model_cls, extra_check=None):
    """A gated step: call, parse, validate against the schema (and an optional extra check), retry once with the exact errors."""
    total = {'input': 0, 'output': 0}
    for attempt in range(2):
        text, use = call(st, prompt)
        for k in total:
            total[k] += use[k]
        try:
            obj = model_cls.model_validate(llm.parse_json(text) or {})
            problems = extra_check(obj) if extra_check else []
        except ValidationError as e:
            obj, problems = None, [f'{".".join(str(x) for x in err["loc"])}: {err["msg"]}' for err in e.errors()]
        if not problems:
            return obj, total, attempt
        prompt = prompt + '\n\nYour previous answer failed these checks:\n- ' + '\n- '.join(problems) + '\nReply with a corrected JSON object only.'
    return None, total, 1


# ---------------------------------------------------------------- the nodes
def route(st):
    t = st.text.lower()
    if re.search(r'\b(refund|charged|invoice|broken|cracked|damaged|parcel|delivery|order)\b', t):     # code decides when it can
        return 'support', None
    text, use = call(st, 'Is this message a support request about an order or account (answer "support") or a general question '
                         f'about products, shipping zones or opening hours (answer "faq")? Reply with one word.\n\nMessage: {st.text}')
    return ('faq' if 'faq' in text.lower() else 'support'), use


def extract(st):
    prompt = ('Extract facts from this support ticket. Reply with ONE JSON object with keys: "issue_type" (one of billing, delivery, '
              'damaged_or_faulty, account, other), "order_id" (the order number as ORD-NNNNN, five digits, or null), '
              f'"refund_amount_gbp" (a number if the customer asks for money back and states an amount, else null).\n\nTicket: {st.text}')
    return validated(st, prompt, Facts)


def policy_problems(reply, order_id):
    p = []
    if len(reply.split()) > 80:
        p.append(f'the reply has {len(reply.split())} words; the limit is 80')
    hits = [b for b in BANNED if b in reply.lower()]
    if hits:
        p.append(f'the reply promises timing: {hits}; remove these phrases')
    if order_id and order_id not in reply:
        p.append(f'the reply must quote the order id {order_id}')
    return p


def draft(st):
    f = st.facts
    prompt = ('Write a short, polite reply (at most 80 words) to this customer, for Northwind Home. Do not promise any timing. '
              + (f'Quote the order id {f["order_id"]}. ' if f.get('order_id') else '')
              + ('Say that the refund request has been passed for approval. ' if f.get('refund_amount_gbp') else '')
              + 'Reply with ONE JSON object: ' + json.dumps({'reply': '...'}) + f'\n\nTicket: {st.text}\nFacts: {json.dumps(f)}')
    return validated(st, prompt, Draft, lambda d: policy_problems(d.reply, f.get('order_id')))


def send(st):
    """The side effect. Idempotent: the key is derived from the ticket and the reply, and the outbox ignores a key it has seen."""
    key = f'{st.ticket_id}:' + hashlib.sha256(st.reply.encode()).hexdigest()[:12]   # stable across processes, unlike hash()
    box = json.load(open(OUTBOX)) if os.path.exists(OUTBOX) else {}
    if key in box:
        return 'already sent (idempotency key seen)'
    box[key] = {'ticket': st.ticket_id, 'reply': st.reply, 'at': time.strftime('%H:%M:%S')}
    json.dump(box, open(OUTBOX, 'w'), indent=1)
    return 'sent'


# ---------------------------------------------------------------- the graph runner
def run(st, crash_after=None):
    """Run from st.node until the workflow finishes, pauses or fails; checkpoint after every node."""
    try:
        while st.node not in ('done', 'needs_human', 'awaiting_approval'):
            t0 = time.time()
            if st.node == 'route':
                st.route, use = route(st)
                span(st, 'route', f'-> {st.route}' + (' (model)' if use else ' (rule)'), t0, use)
                st.node = 'extract' if st.route == 'support' else 'faq'
            elif st.node == 'faq':
                st.notes.append('sent to the FAQ answerer (out of scope for this demo)')
                span(st, 'faq', 'handed to the FAQ workflow', t0)
                st.node = 'done'
            elif st.node == 'extract':
                obj, use, retried = extract(st)
                if obj is None:
                    span(st, 'extract', 'gate failed twice -> person', t0, use); st.node = 'needs_human'
                else:
                    st.facts = obj.model_dump()
                    span(st, 'extract', f'ok{" after retry" if retried else ""} {st.facts["issue_type"]} {st.facts["order_id"]}', t0, use)
                    st.node = 'draft'
            elif st.node == 'draft':
                obj, use, retried = draft(st)
                if obj is None:
                    span(st, 'draft', 'gate failed twice -> person', t0, use); st.node = 'needs_human'
                else:
                    st.reply = obj.reply
                    span(st, 'draft', f'ok{" after retry" if retried else ""} ({len(st.reply.split())} words)', t0, use)
                    st.node = 'review'
            elif st.node == 'review':
                amt = st.facts.get('refund_amount_gbp') or 0
                if amt > REFUND_LIMIT and st.approved is None:
                    span(st, 'review', f'refund GBP {amt:.2f} > {REFUND_LIMIT:.0f}: PAUSE', t0); st.node = 'awaiting_approval'
                elif st.approved is False:
                    span(st, 'review', 'refund rejected by a person', t0); st.node = 'needs_human'
                else:
                    span(st, 'review', 'policy ok' + (' (approved)' if st.approved else ''), t0); st.node = 'send'
            elif st.node == 'send':
                span(st, 'send', send(st), t0); st.node = 'done'
            checkpoint(st)
            if crash_after and st.node == crash_after[1] and st.ticket_id == crash_after[0]:
                raise InjectedCrash(f'process killed after checkpointing, before running {st.node}')
    except BudgetExceeded as e:
        span(st, st.node, f'STOP: {e}', time.time()); st.node = 'needs_human'; checkpoint(st)
    return st


TICKETS = [('T1', 'Order #48213 arrived with the screen cracked. Please send a replacement.'),
           ('T2', 'I was charged twice for ORD-10988, please refund the extra GBP 39.99.'),
           ('T3', 'Where is my parcel? Tracking has not moved in 6 days. Order 77120.'),
           ('T4', 'Do you ship to the Isle of Man?'),
           ('T5', 'The sofa from order 61234 has a torn seam. I want my money back, GBP 649.'),
           ('T6', 'Mi pedido 33310 llego roto.')]

if __name__ == '__main__':
    import shutil
    for p in [CKPT]:
        shutil.rmtree(p, ignore_errors=True)
    for p in [OUTBOX, TRACE]:
        if os.path.exists(p):
            os.remove(p)
    log(f'== A composed workflow for Northwind Home support ({llm.MODEL}, minimal effort): router -> gated chain -> policy -> approval -> send ==')
    log(f'budgets per ticket: {MAX_CALLS} model calls, {MAX_TOKENS} tokens; refunds over GBP {REFUND_LIMIT:.0f} need a person')
    log('')
    log('-- first run (a crash is injected in T3 after the draft is checkpointed) --')
    for tid, text in TICKETS:
        st = State(tid, text)
        try:
            run(st, crash_after=('T3', 'review'))
        except InjectedCrash as e:
            log(f'  !! {tid}: {e}')
    log('')
    log('-- resume: load every checkpoint that is not finished and continue from its saved node --')
    for tid, _ in TICKETS:
        st = load(tid)
        if st.node not in ('done', 'needs_human', 'awaiting_approval'):
            log(f'  {tid}: resuming at node "{st.node}" (calls so far {st.calls}; extract and draft are NOT re-run)')
            run(st)
    log('')
    log('-- a person approves the paused refund, and the run continues from the checkpoint --')
    for tid, _ in TICKETS:
        st = load(tid)
        if st.node == 'awaiting_approval':
            st.approved = True
            st.node = 'review'
            log(f'  {tid}: approved by a person')
            run(st)
    log('')
    log('-- replaying "send" for T1 to show idempotency --')
    st = load('T1'); t0 = time.time(); span(st, 'send', send(st), t0)
    log('')
    final = [load(t) for t, _ in TICKETS]
    log(f'{"ticket":<7} {"route":<8} {"final node":<12} {"calls":>5} {"tokens":>7}  facts')
    for st in final:
        f = st.facts or {}
        log(f'{st.ticket_id:<7} {str(st.route):<8} {st.node:<12} {st.calls:>5} {st.tokens:>7}  {f.get("issue_type", "-")} {f.get("order_id", "-")} '
            f'{("refund " + str(f.get("refund_amount_gbp"))) if f.get("refund_amount_gbp") else ""}')
    box = json.load(open(OUTBOX))
    spans = [json.loads(l) for l in open(TRACE)]
    U = llm.snapshot()
    log(f'outbox: {len(box)} replies sent for {sum(1 for s in final if s.node == "done" and s.route == "support")} finished support tickets; '
        f'trace: {len(spans)} spans in results/ch4_workflow_trace.jsonl')
    log(f'total spend: ${llm.cost(U):.4f} ({U["calls"]} calls); stub used: ' + ('YES' if llm.stub_used() else 'no'))
    save('ch4_workflow', {'states': [asdict(s) for s in final], 'spans': spans, 'outbox': box, 'usage': U, 'stub': llm.stub_used()})
```

</details>

<details>
<summary>Research notes: LangGraph's interrupt (optional reading)</summary>

> [!PAPER] LangChain, "Interrupts" (LangGraph documentation, accessed October 2026)
> [![A paragraph from the LangGraph docs: Interrupts allow you to pause graph execution at specific points and wait for external input before continuing; this enables human-in-the-loop patterns where you need external input to proceed; when an interrupt is triggered, LangGraph saves the graph state using its persistence layer and waits indefinitely until you resume execution](/img/agents/ch4-langgraph-interrupts.png)](/img/agents/ch4-langgraph-interrupts.png)
>
> **Context:** the documentation for `interrupt()`, which a node calls to pause; the run is resumed with `Command(resume=...)` and the same thread id.
>
> **What it says:** "LangGraph saves the graph state using its persistence layer and waits indefinitely until you resume execution." The page also warns that on resume "the node restarts from the beginning of the node where the interrupt was called ... so any code before the interrupt runs again".
>
> **Why it matters:** that warning is the idempotency rule again. Anything with a side effect that runs before an interrupt inside the same node will run twice; put side effects in their own node after the approval.

</details>

### Observability to design in now

Every node in the program writes one line to a trace file: the trace id of the run, the ticket, the node, the outcome, the tokens and the milliseconds. That is the minimum. With it you can answer the questions this chapter has asked of every pattern: which step fails first, how often each gate retries, how many tickets the router sends to the model, how long approvals wait, what each ticket costs. Chapter 8 builds this into proper tracing with spans, parent ids and dashboards; the decision to make now is that every node emits a record with the same trace id.

> [!DEFINITION] Trace and span
> A **trace** is the record of one run through the workflow, identified by a trace id. A **span** is the record of one step within it: which node, when it started and ended, what it consumed, and how it ended. Chapter 8 treats both in depth.

### Composition in practice

| Plumbing | What it prevents | How the program does it |
|---|---|---|
| Typed state and schemas | a step reading a field that is missing or malformed | a dataclass for state; pydantic models for step outputs |
| Gates with one retry | format failures flowing downstream | `validated()`: parse, validate, retry once with the errors |
| Budgets per run | a loop or retry storm burning money | a call and token cap that stops the run and hands it to a person |
| Checkpoints and resume | a crash losing work, or redoing paid calls | state written after every node; `load()` continues from `node` |
| Idempotent side effects | duplicate emails or refunds after a resume | an outbox keyed by a stable hash of ticket and reply |
| Human interrupt | money moving without approval | `awaiting_approval` state, resumed after a person decides |
| Trace | not knowing where or why runs fail | one JSON line per node with the trace id |

> [!TIP] In production
> Record each model call's output in the state before the next node runs, so a resume never calls the model again for a finished step (this is also what makes a replay deterministic). Deploy new versions without breaking runs in flight: Anthropic's research post describes "rainbow deployments" that keep old and new versions running side by side for this reason.

### Discussion

1. **Framework or hand-written?** Our runner is about thirty lines; LangGraph, Temporal and the vendor SDKs add persistence, retries, visualisation and hosting. A framework pays for itself when you need durable waits of hours or days and many workflows; the concepts are the same either way.
2. **Where should budgets live?** In code, per run, enforced before each call, as in the program. A budget written only in a prompt is a suggestion.
3. **What is the first thing to break when the workflow grows?** Usually the interfaces: a schema changes in one step and the next step still expects the old one. Version schemas and put the version in the trace.

## 4.8 Workflows in production: what companies built and what they report

This section reads engineering write-ups for the same four things each time: which pattern, why, what went wrong, and what they report. Every quotation was checked against the live page. Each case separates **what the company reports** from **our reading** of it, and attributes numbers to the exact system they describe. Where a page gives no number, none is given here. Pages that block automated browsers are quoted with a link rather than screenshotted.

{{FIG:ch4_companies|Workflow patterns stated in the production write-ups of this section, one row per system. Chains are the most common shape, usually with a router in front or an evaluator behind; orchestrator-workers appear for open-ended research. Filled cells are patterns the page describes, not our inference.}}

### Uber: a chain of filters, and a router of intents

> [!PAPER] Uber, "uReview: Scalable, Trustworthy GenAI for Code Review at Uber" (Engineering blog, August 2025)
> [![A paragraph from the post: uReview is a modular, multi-stage GenAI system designed to automate and enhance code reviews across Uber's engineering platforms; its prompt-chaining-based architecture breaks down the code-review task into four simpler sub-tasks, and allows each sub-task, comment generation, filtering, validation, and deduplication, to evolve independently](/img/agents/ch4-uber-ureview.png)](/img/agents/ch4-uber-ureview.png)
>
> **Context:** uReview reviews code changes at Uber. Comments come from three specialised assistants (bugs, Uber's best practices, application security), then pass through filters.
>
> **What Uber reports:** a "prompt-chaining-based architecture" with four sub-tasks: "comment generation, filtering, validation, and deduplication". A "secondary prompt evaluates each comment's quality and assigns a confidence score", "a semantic similarity filter merges overlapping suggestions", and a category classifier suppresses categories "with historically low developer value". The system "analyzes over 90% of the weekly ~65,000 diffs"; engineers "mark 75% of its comments as useful"; and "over 65% of its posted comments" are addressed, against 51% of human-written comments in Uber's internal audit. The stated reason for the design: "a simple standalone prompt results in many false-positive comments and many low-value true-positive comments".
>
> **Our reading:** a chain whose later steps exist to remove output: sectioning by issue type (three assistants), then an evaluator (the confidence grader), then filters. Thresholds are set "per assistant, per language, and comment category", which is possible only because each step is separate and measured. Uber also evaluates whether a comment was addressed by re-running the review five times on the final code, because a single re-run is too noisy: voting used for evaluation, not for answers.

> [!PAPER] Uber, "QueryGPT: Natural Language to SQL Using Generative AI" (Engineering blog, September 2024)
> [![A paragraph from the post: every incoming prompt from the user now first runs through an intent agent; the purpose of this intent agent is to map the user's question to one or more business domains or workspaces, and by extension a set of SQL samples and tables mapped to the domain; we use an LLM call to infer the intent from the user question and map these to system workspaces or custom workspaces](/img/agents/ch4-uber-querygpt.png)](/img/agents/ch4-uber-querygpt.png)
>
> **Context:** QueryGPT writes SQL for Uber's internal data platform. Its first version retrieved tables and examples directly from the question; accuracy fell "as we started to onboard more tables".
>
> **What Uber reports:** a chain of small model steps: an **intent agent** (a router to one of a dozen business "workspaces"), a **table agent** whose chosen tables the user can accept or edit, and a **column prune agent** that removes irrelevant columns from the prompt. They evaluate in two modes: "vanilla", end to end, and "decoupled", which feeds each step the correct output of the previous one so that each step is measured on its own. Their stated learnings: "LLMs are excellent classifiers", because the agents "were asked to work on a single unit of work rather than a broad generalized task"; hallucinated tables and columns remain unsolved; and repeated evaluation runs vary, so they "do not over-index decisions based on ~5% run-to-run changes".
>
> **Our reading:** routing in front of a chain, a human gate on the table choice, and the clearest public example of per-step evaluation, which is exactly the "first fault position" measurement of Section 4.2.

### Anthropic: orchestrator-workers for research

Section 4.5 read the post in detail. **What Anthropic reports:** a lead agent with parallel subagents "outperformed single-agent Claude Opus 4 by 90.2%" on an internal research eval; multi-agent systems "use about 15× more tokens than chats"; early versions spawned "50 subagents for simple queries"; parallelism "cut research time by up to 90% for complex queries"; the lead agent waits for each set of subagents, which "creates bottlenecks". For evaluation they use a single model judge with a rubric (factual accuracy, citation accuracy, completeness, source quality, tool efficiency) scoring 0.0 to 1.0 with a pass-fail grade, plus human testing, starting from "a set of about 20 queries". **Our reading:** the pattern paid where subtasks were independent and answers valuable; the failures were orchestration failures (effort, briefs, stopping) fixed in the orchestrator's prompt and in code limits.

### DoorDash: a guarded pipeline for support

DoorDash's September 2024 post "Path to high-quality LLM-based Dasher support automation" (DoorDash's site blocks automated browsers, so it is quoted here) describes a support system for delivery drivers. **What DoorDash reports:** a fixed retrieval pipeline with a two-tier guardrail (a cheap semantic-similarity check, then a model-based evaluator) that can retry or hand the conversation to a person, plus an offline model judge on five quality dimensions. "This guardrail system has successfully reduced overall hallucinations by 90% and cut down potentially severe compliance issues by 99%"; the guardrail's latency "is a notable drawback", and a more sophisticated guardrail model was dropped because "increased response times and heavy usage of model tokens made it prohibitively expensive". A November 2025 post from the same company ("Beyond Single Agents") argues for starting with "deterministic workflows" and warns that "you can't jump straight to sophisticated, multi-agent collaboration". **Our reading:** a chain with an evaluator at the end, ordered cheap check first and model check second, with a human fallback rather than a long retry loop; the cost of the evaluator shaped the design.

<details>
<summary>More production cases: Stripe, Airbnb, LinkedIn, Uber Genie, Spotify, Microsoft and Amazon (optional reading)</summary>

**Stripe and Airbnb: workflows written as state machines.** Two coding pipelines describe themselves in this chapter's vocabulary. **Stripe** (February 2026, "Minions") reports that its background coding agents follow "blueprints", workflows "defined in code" that look like "a state machine that intermixes deterministic code nodes and free-flowing agent nodes", with at most two rounds of continuous integration "since CI runs cost tokens, compute, and time". **Airbnb** (March 2025, "Accelerating Large-Scale Test Migration with LLMs", quoted because the page blocks automated browsers) migrated about 3,500 test files with a per-file pipeline "modeled ... like a state machine", retrying steps with the validation errors "until they passed or we reached a limit"; it reports 75% of files migrated "in just four hours", 97% after four days of tuning, and the remaining 3% finished by hand. **Our reading:** both are gated chains with code evaluators (tests, linters, CI) and capped retries, with a model inside some nodes and people at the end of the tail.

> [!PAPER] LinkedIn, "Musings on building a Generative AI product" (Engineering blog, April 2024), on the pipeline
> [![A list from the post: Routing decides if the query is in scope or not, and which AI agent to forward it to, with examples such as job assessment, company understanding and takeaways for posts; Retrieval, a recall-oriented step where the AI agent decides which services to call and how; Generation, a precision-oriented step that sieves through the noisy data retrieved, filters it and produces the final response](/img/agents/ch4-linkedin-pipeline.png)](/img/agents/ch4-linkedin-pipeline.png)
>
> **Context:** the Premium assistant's architecture, the same post as Section 4.2's gate example.
>
> **What LinkedIn reports:** a "fixed 3-step pipeline" of routing, retrieval and generation; "small models for routing/retrieval, bigger models for generation"; "per-step specific evaluation pipelines, particularly for routing/retrieval". Routing and retrieval "felt more natural given their classification nature"; generation "followed the 80/20 rule". The team reached "80% of the basic experience" in the first month and then "spent an additional four months attempting to surpass 95%".
>
> **Our reading:** a router in front of a two-step chain, model sizes chosen per step, and evals per step: the composition of Sections 4.2 and 4.3, with an honest account of where the time went.

> [!PAPER] Uber, "Enhanced Agentic-RAG: What If Chatbots Could Deliver Near-Human Precision?" (Engineering blog, May 2025)
> [![A paragraph from the post: in the pre-processing step, we use two agents, Query Optimizer and Source Identifier; Query Optimizer refines the query when it lacks context or is ambiguous and breaks down complex queries into multiple simpler queries for better retrieval; Source Identifier then processes the optimized query to narrow down the subset of policy documents most likely to contain relevant answers](/img/agents/ch4-uber-genie.png)](/img/agents/ch4-uber-genie.png)
>
> **Context:** Genie, Uber's on-call assistant, in its security and privacy channels.
>
> **What Uber reports:** pre-processing agents (a query optimiser and a source identifier that narrows the documents), hybrid retrieval (the union of vector and keyword results), post-processing, and an automated model judge scoring answers 0 to 5 against expert-written answers. The post reports "increasing the percentage of acceptable answers by a relative 27% and reducing incorrect advice by a relative 60%", and says "our current implementation follows a sequential flow", built on LangGraph.
>
> **Our reading:** "agentic RAG" here is a fixed chain of model steps with a router-like source selection and an offline evaluator. The word "agentic" describes the steps, not the control flow.

**Spotify, Honk (December 2025).** Spotify's background coding agent ends with deterministic verifiers (build, tests, formatting) and then a model judge that compares the diff with the original prompt: "The judge is simple. It uses the diff of the proposed change and the original prompt, and sends them to an LLM for evaluation." The post reports that "the judge vetoes about a quarter" of thousands of sessions and that "the agent is able to course correct half the time", and states plainly: "We have yet to invest in evals for our judge." Our reading: an evaluator placed after the code checks, in the order this chapter recommends, whose own $$r$$ and $$f$$ are not yet measured.

**LinkedIn, SQL Bot (December 2024)** checks generated queries with validators that "access new information not available to the query writer" (tables and fields exist; `EXPLAIN` runs) and feeds errors to a correction step (Chapter 3 reads it in detail). **Microsoft's Magentic-One** (Section 4.5) is the research orchestrator with explicit task and progress ledgers. **Amazon's** code-transformation agent in Q Developer has developers "review and iterate on the plan before the agent implements it", then builds and tests the result (Chapter 3). We could not verify public engineering write-ups with workflow details for Netflix, Instacart or Klarna, so they are not included.

</details>

### The table

| Company, system | Pattern | Why (as stated) | Reported problem or limit | Reported result |
|---|---|---|---|---|
| Uber, uReview | sectioned generation, then a grader and filters | single prompts gave false positives and low-value comments | noise; thresholds need per-language tuning | 75% of comments marked useful; 65% addressed |
| Uber, QueryGPT | intent router, then a chain; human edits the tables | accuracy fell as tables grew | hallucinated tables and columns; ~5% run-to-run variance | not given as a single number |
| Uber, Genie (EAg-RAG) | chain of pre- and post-processing agents; offline judge | incomplete or wrong answers; slow expert evaluation | evaluation took experts weeks | +27% relative acceptable answers, -60% relative incorrect advice |
| LinkedIn, Premium assistant | router, retrieval, generation | different difficulty per step | ~10% of structured outputs invalid; last 15% of quality slow | 80% of target in one month; four more months towards 95% |
| DoorDash, Dasher support | pipeline with a two-tier guardrail and human fallback | hallucination and compliance risk | guardrail latency; a stronger guardrail too costly | -90% hallucinations, -99% severe compliance issues |
| Spotify, Honk | agent with code verifiers, then a model judge | agents straying outside the prompt | judge not yet evaluated | judge vetoes about a quarter; half recover |
| Stripe, Minions | state machine of code nodes and agent nodes; CI as evaluator | CI rounds cost tokens and time | diminishing returns from more CI rounds | capped at two CI rounds |
| Airbnb, test migration | per-file state machine with capped retries | a known transformation over 3,500 files | a long tail automation could not fix | 75% in four hours; 97% in four days; rest by hand |
| Anthropic, Research | orchestrator-workers with parallel subagents | breadth-first questions | 15x chat tokens; over-spawning; synchronous bottleneck | +90.2% on an internal eval |

Three things hold across the table, as patterns rather than laws. Most systems are chains, with a router in front or an evaluator behind; the orchestrator appears where the question is open-ended and valuable. Evaluators are usually placed after cheap code checks, and their cost and latency shaped the designs (DoorDash dropped a stronger guardrail; Stripe capped CI rounds). And the teams that report the most progress also report how they measured each step, not only the end result.

### Discussion

1. **Why are most production systems chains?** The tasks automated first are the ones whose steps are known, and a known chain is cheaper to run, test and explain. That is a reasonable order of work, not a lack of ambition.
2. **Which reported numbers can you compare?** Few. Each company measures its own system on its own data with its own definition of success (useful comments, acceptable answers, vetoed sessions). Read them as evidence that the pattern was worth keeping, not as benchmarks.
3. **What is missing from the write-ups?** Mostly the evaluators' own error rates: how often the grader, guardrail or judge is wrong. Spotify says so openly; the rest are silent. That is the measurement to add first in your own system.

## 4.9 Choosing and evolving a workflow

The decision is not a single choice but an order of additions, each justified by a measured problem. The questions below map task properties to patterns.

| Task property | How to measure it | What it argues for |
|---|---|---|
| Variability of inputs | cluster a sample of real inputs; count the kinds | one kind: a chain; several kinds with different handling: a router in front |
| Independence of subtasks | for each pair of subtasks, does one need the other's output? | independent: parallel sections; dependent: a chain; unknowable in advance: an orchestrator |
| Verifiability | list the checks you can write in code, and the ones that need judgement | code checks: gates and an evaluator loop; judgement only: a measured judge, or a person |
| Latency budget | the product's p95 target against the chain's sequential depth | tight: parallelise, use smaller models per step, cut steps |
| Cost per task | tokens per run from traces, times traffic | high volume: routers and cascades; high value per task: orchestrators may pay |
| Blast radius | what happens if a wrong output ships: a reply, a refund, a deleted record | high: a human interrupt before the side effect, and idempotent steps |

{{FIG:ch4_decision|A migration path, as a starting heuristic rather than a fixed sequence. Start with a gated chain; add a router when inputs diverge, parallel sections when latency hurts, an orchestrator when the split cannot be written in advance, an evaluator where quality can be measured, and an agent loop only when the steps themselves cannot be known. Re-measure after every addition.}}

The migration path in the figure is a heuristic: the patterns compose, so a system may need an evaluator before it needs a router, and some tasks start as orchestrators. What does hold is the order of evidence. Each addition should answer a failure you have seen in traces and labelled data, and each should be re-measured on accuracy, cost per correct answer and latency once it is in.

The one-rule summary of the chapter: **hand the model a decision only when you cannot write it in code, and measure every decision you hand over.** A chain hands over none; a router hands over one, measurable with a confusion matrix; parallel calls hand over none and buy time; an orchestrator hands over the shape of the work, measurable by grading plans and counting duplicated effort; an evaluator hands over the decision to stop, measurable by its false-accept and false-reject rates. The experiments in this chapter found the same thing from four directions: the model was good at the content of each step and unreliable at the decisions around it (whether its answer was good enough, whether a draft had 55 words, whether a question was hard), and code made those decisions better wherever code could make them.

### Discussion

1. **Where should Northwind start?** With the gated chain of Section 4.2, a labelled set of a few hundred tickets, and per-step evals. Everything else waits for a measured reason.
2. **When is it right to skip a step on the path?** When the task's structure is obvious: code generation against tests goes straight to an evaluator loop with the tests as the evaluator; a research question goes straight to an orchestrator. Skipping is fine if you still take the measurement that would have justified the step.
3. **What should be re-measured when the model is upgraded?** Every model decision: the router's confusion matrix, the judge's error rates, the per-step success rates. A new model can improve the content and change the decisions in either direction.

## Exercises

1. Northwind's chain has three steps at 95%, 92% and 98% per step on a labelled set. Compute the end-to-end success under the independence assumption, say which step a gate would help most, and describe one way the independence assumption could fail on real tickets.
2. A cascade's check accepts 90% of right cheap answers and 25% of wrong ones. Compute the precision of accepted answers when the cheap model is right 80% of the time and when it is right 40% of the time, and decide in which case you would ship the cascade.
3. Design the brief an orchestrator should give one worker for "why did delivery complaints rise last quarter?": objective, output format, tools and sources, boundaries. Then list two measurements that would tell you whether workers are duplicating each other.
4. Your evaluator loop uses a model judge for tone and a word count in code. Traces show the loop runs four rounds on 30% of drafts. Describe how you would find out whether the judge is producing false fails, and what you would change if it is.
5. In `ch4_workflow.py`, the send step is idempotent because of its outbox key. Name two other side effects a support workflow might have, and how you would make each idempotent.

## Key takeaways

- A **workflow** is model calls arranged by control flow the engineer wrote; an **agent** lets the model choose the next step. Most production systems are workflows with a model inside some nodes.
- Draw the system as a **graph** of nodes and edges, and mark which edges code decides and which the model decides. Each decision handed to the model needs its own measurement.
- **Prompt chains** make each step easy and testable. End-to-end success is roughly the product of per-step success, so **gates** between steps matter: in our run, gates with one retry lifted a loosely specified chain from 12 to 16 of 20, and could not catch valid-but-wrong values.
- **Routers** are judged by their confusion matrix, not their accuracy. Our model router sent one of 24 questions to the strong model and caught none of the seven that needed it; a cascade reached 88% at about two-thirds of the strong model's cost per correct answer.
- The **false-accept rate** of a check is not one minus its precision: precision depends on the base rate, $$p\,r / (p\,r + (1-p)\,f)$$.
- **Parallel** calls buy time, not tokens (4.5 times faster in our run). A **plurality vote** needs the right answer to be the most frequent, not a majority; correlated errors cut the gain, and our vote gained nothing. Best-of-n with a **verifier** helped, by less than independence predicts.
- **Orchestrator-workers** fits open-ended, breadth-first, valuable tasks. Anthropic reports about 15 times the tokens of a chat; the failures they report are about briefs, effort and stopping.
- In an **evaluator-optimiser** loop, put code checks first and measure any judge before it stops a loop. Our judge agreed with a word count 32% of the time, and its false fails both cost rounds and damaged a good draft.
- Production plumbing is part of the pattern: typed state, schema checks, budgets, checkpoints, idempotent side effects, human interrupts and one trace line per node, all shown working in `ch4_workflow.py`.
- Hand the model a decision only when you cannot write it in code, and re-measure every handed-over decision after each change.

## References

**Papers**

1. Tongshuang Wu, Michael Terry, Carrie J. Cai. [*AI Chains: Transparent and Controllable Human-AI Interaction by Chaining Large Language Model Prompts*](https://arxiv.org/abs/2110.01691). CHI 2022 (arXiv October 2021).
2. Denny Zhou, Nathanael Schärli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc Le, Ed Chi. [*Least-to-Most Prompting Enables Complex Reasoning in Large Language Models*](https://arxiv.org/abs/2205.10625). ICLR 2023 (arXiv May 2022).
3. Tushar Khot, Harsh Trivedi, Matthew Finlayson, Yao Fu, Kyle Richardson, Peter Clark, Ashish Sabharwal. [*Decomposed Prompting: A Modular Approach for Solving Complex Tasks*](https://arxiv.org/abs/2210.02406). ICLR 2023 (arXiv October 2022).
4. Lingjiao Chen, Matei Zaharia, James Zou. [*FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance*](https://arxiv.org/abs/2305.05176). arXiv May 2023.
5. Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks V. S. Lakshmanan, Ahmed Hassan Awadallah. [*Hybrid LLM: Cost-Efficient and Quality-Aware Query Routing*](https://arxiv.org/abs/2404.14618). ICLR 2024 (arXiv April 2024).
6. Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, Ion Stoica. [*RouteLLM: Learning to Route LLMs with Preference Data*](https://arxiv.org/abs/2406.18665). ICLR 2025 (arXiv June 2024).
7. Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, Deheng Ye. [*More Agents Is All You Need*](https://arxiv.org/abs/2402.05120). TMLR 2024 (arXiv February 2024).
8. Xinyun Chen, Renat Aksitov, Uri Alon, Jie Ren, Kefan Xiao, Pengcheng Yin, Sushant Prakash, Charles Sutton, Xuezhi Wang, Denny Zhou. [*Universal Self-Consistency for Large Language Model Generation*](https://arxiv.org/abs/2311.17311). arXiv November 2023.
9. Bradley Brown, Jordan Juravsky, Ryan Ehrlich, Ronald Clark, Quoc V. Le, Christopher Ré, Azalia Mirhoseini. [*Large Language Monkeys: Scaling Inference Compute with Repeated Sampling*](https://arxiv.org/abs/2407.21787). arXiv July 2024.
10. Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, Yueting Zhuang. [*HuggingGPT*](https://arxiv.org/abs/2303.17580) (full title on arXiv). NeurIPS 2023 (arXiv March 2023).
11. Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W. White, Doug Burger, Chi Wang. [*AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation*](https://arxiv.org/abs/2308.08155). arXiv August 2023.
12. Adam Fourney, Gagan Bansal, Hussein Mozannar, Cheng Tan, Eduardo Salinas, Erkang Zhu, Friederike Niedtner, Grace Proebsting, Griffin Bassman, Jack Gerrits, Jacob Alber, Peter Chang, Ricky Loynd, Robert West, Victor Dibia, Ahmed Awadallah, Ece Kamar, Rafah Hosn, Saleema Amershi. [*Magentic-One: A Generalist Multi-Agent System for Solving Complex Tasks*](https://arxiv.org/abs/2411.04468). arXiv November 2024.
13. Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, Ion Stoica. [*Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena*](https://arxiv.org/abs/2306.05685). NeurIPS 2023 Datasets and Benchmarks (arXiv June 2023).
14. Yang Liu, Dan Iter, Yichong Xu, Shuohang Wang, Ruochen Xu, Chenguang Zhu. [*G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment*](https://arxiv.org/abs/2303.16634). EMNLP 2023 (arXiv March 2023).
15. Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, Denny Zhou. [*Large Language Models Cannot Self-Correct Reasoning Yet*](https://arxiv.org/abs/2310.01798). ICLR 2024 (arXiv October 2023).

**Engineering blogs and docs**

16. Anthropic (Erik Schluntz, Barry Zhang). [*Building effective agents*](https://www.anthropic.com/research/building-effective-agents). 19 December 2024.
17. Anthropic (Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, Daniel Ford). [*How we built our multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system). Engineering blog, 13 June 2025.
18. OpenAI. [*A practical guide to building agents*](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf). PDF, April 2025.
19. LangChain. [*Graph API overview*](https://docs.langchain.com/oss/python/langgraph/graph-api) and [*Interrupts*](https://docs.langchain.com/oss/python/langgraph/interrupts). LangGraph documentation, accessed October 2026.
20. Temporal. [*Understanding Temporal*](https://docs.temporal.io/evaluate/understanding-temporal). Temporal documentation, accessed October 2026.
21. AWS (Aaron Sempf, Andrew Hooker). [*Agentic AI patterns and workflows on AWS: Workflow for routing*](https://docs.aws.amazon.com/prescriptive-guidance/latest/agentic-ai-patterns/workflow-for-routing.html). AWS Prescriptive Guidance, July 2025.
22. Uber. [*uReview: Scalable, Trustworthy GenAI for Code Review at Uber*](https://www.uber.com/us/en/blog/ureview/). Engineering blog, 12 August 2025.
23. Uber. [*QueryGPT: Natural Language to SQL Using Generative AI*](https://www.uber.com/us/en/blog/query-gpt/). Engineering blog, 19 September 2024.
24. Uber. [*Enhanced Agentic-RAG: What If Chatbots Could Deliver Near-Human Precision?*](https://www.uber.com/us/en/blog/enhanced-agentic-rag/). Engineering blog, 29 May 2025.
25. LinkedIn (Juan Pablo Bottaro, Karthik Ramgopal). [*Musings on building a Generative AI product*](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product). Engineering blog, 25 April 2024.
26. LinkedIn (Albert Chen and colleagues). [*Practical text-to-SQL for data analytics*](https://www.linkedin.com/blog/engineering/ai/practical-text-to-sql-for-data-analytics). Engineering blog, 9 December 2024.
27. DoorDash. [*Path to high-quality LLM-based Dasher support automation*](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/). Engineering blog, 17 September 2024; and [*Beyond Single Agents: How DoorDash is building a collaborative AI ecosystem*](https://careersatdoordash.com/blog/beyond-single-agents-doordash-building-collaborative-ai-ecosystem/). 11 November 2025.
28. Spotify (Max Charas, Marc Bruggmann). [*Background Coding Agents: Predictable Results Through Strong Feedback Loops (Honk, Part 3)*](https://engineering.atspotify.com/2025/12/feedback-loops-background-coding-agents-part-3). Engineering blog, 9 December 2025.
29. Stripe (Alistair Gray). [*Minions: Stripe's one-shot, end-to-end coding agents*](https://stripe.dev/blog/minions-stripes-one-shot-end-to-end-coding-agents). stripe.dev blog, 9 February 2026.
30. Airbnb (Charles Covey-Brandt). [*Accelerating Large-Scale Test Migration with LLMs*](https://medium.com/airbnb-engineering/accelerating-large-scale-test-migration-with-llms-9565c208023b). Airbnb Tech Blog, March 2025.
31. AWS (Aytul Arisoy Cholkar). [*Amazon Q Developer just reached a $260 million dollar milestone*](https://aws.amazon.com/blogs/devops/amazon-q-developer-just-reached-a-260-million-dollar-milestone). AWS DevOps blog, 1 August 2024.

**Code for this chapter**

32. [`code/agents/ch4_chain.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_chain.py): the three-step ticket chain under four conditions (strict or loose first step, with or without gates) against gpt-5-mini; results in `results/ch4_chain.json`.
33. [`code/agents/ch4_router.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_router.py), [`ch4_router_stats.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_router_stats.py) and [`ch4_grade.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_grade.py): always-cheap, always-strong, cascade and classifier router on 24 questions, the cascade check read as a verifier, and the typed grader with its adversarial self-test.
34. [`code/agents/ch4_parallel.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_parallel.py): plurality voting, best-of-n with a programmatic verifier, and sequential against concurrent wall-clock time.
35. [`code/agents/ch4_evaluator.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_evaluator.py): the evaluator-optimiser loop with a code checker or a model judge, judge-checker agreement, and a held-out fact check.
36. [`code/agents/ch4_cost.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_cost.py): chain reliability, routing cost, plurality voting with correlated errors, orchestrator cost, and the precision of a stop with an imperfect evaluator, each with its assumptions printed.
37. [`code/agents/ch4_workflow.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_workflow.py): the complete composed workflow with typed state, schema-validated steps, budgets, checkpoints, resume, a human interrupt, an idempotent send and a trace; [`ch4_llm.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch4_llm.py) is the shared API helper.
38. [`code/agents/figs_ch4.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/figs_ch4.py) and [`code/agents/shots_ch4.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/shots_ch4.py): the figures (with an automatic text-overlap check) and the paper excerpts.

## Next

→ [Chapter 5: Multi-agent systems](./05-multi-agent-systems.md)

This chapter kept the control flow in code wherever it could and handed the model one decision at a time. Chapter 5 follows the orchestrator-workers pattern to its end: systems of several agents, each with its own loop, that hand work to one another, debate or vote. It asks the questions this chapter asked of every pattern (what changes, what it costs, how it fails and how you test it) of systems where the graph itself is partly written by the models, and it reads the evidence on when several agents beat one and when they only multiply the bill.
