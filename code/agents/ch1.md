---
description: "What an agent is and when not to build one: the perceive-reason-act loop, levels of autonomy, reliability arithmetic, and what three companies learned."
---
# Chapter 1 · What an agent is, and when not to build one

> **Goal:** by the end of this chapter you can say precisely what makes a system an *agent* rather than a model or a workflow, draw the loop that every agent runs, place any design on the spectrum from script to multi-agent system, do the arithmetic that tells you how many steps an agent can afford, and decide, for a given task, whether an agent is the right answer at all. You will also know where the idea came from and what three companies learned by running agents in production.

---

## 1.1 From one LLM call to a loop

You already know the basic move. You write a prompt, send it to a large language model, and get text back. One request, one response. The model does not remember the previous request unless you paste it in, it cannot look anything up, and it cannot do anything except produce text. This is a **model call**, and most useful LLM products are built from exactly this.

> [!DEFINITION] Model call
> One request to a language model: a prompt (instructions, context, a question) goes in, and a piece of text comes out. The model keeps no state between calls. Everything it "knows" about the current situation must be inside the prompt.

Now imagine a different kind of system. An alert fires at 14:06: the error rate of `checkout-api` has jumped to 12%. The system reads the alert. It decides that it needs more information, so it queries the log store for recent errors. The logs come back: 212 errors, almost all "connection pool exhausted", starting at 14:02, four minutes after a deploy of version 2.31. The system reads that, concludes the deploy is the likely cause, and opens a ticket proposing a rollback. Then it reports to the on-call engineer and stops.

Nothing in that story is a single call. The system took an **observation** (the alert), **reasoned** about it, chose an **action** (query the logs), observed the result, reasoned again, chose another action (open a ticket), and finally decided it was done. The same model was called three times, and each call saw everything that had happened so far. That repeating cycle is what makes the system an **agent**.

> [!DEFINITION] Agent
> A system that pursues a goal by repeatedly observing its situation, deciding what to do next, and acting, where the decisions are made by a language model. The model decides *which* step comes next; the surrounding code runs the step and feeds the result back. An agent is defined by the loop, not by the model inside it.

> [!DEFINITION] Environment
> Everything outside the agent that it can observe or change: the log store, the ticket system, a web page, a file system, a database, the person it is talking to. The agent reaches the environment only through its tools.

> [!DEFINITION] Observation
> A piece of information that arrives from the environment: the text of an alert, the rows a query returned, the error a command printed, a message from a user. Observations are appended to what the model sees, so the next decision can use them.

> [!DEFINITION] Action
> Something the agent does to the environment: call an API, run a query, write a file, send a message, or give a final answer. In an LLM agent an action is almost always a **tool call**.

> [!DEFINITION] Tool
> A function the model is allowed to ask for, described to it by name, purpose and parameters. The model does not run the tool; it emits a request such as `search_logs(service="checkout-api", minutes=15)`, and the surrounding code runs it and returns the result as an observation. Chapter 2 is about designing tools well.

> [!DEFINITION] Loop
> The cycle *perceive, reason, act, observe*, run until the model declares the task finished or a budget (steps, time, money) runs out. The loop is ordinary code; the model is called once per turn of it.

> [!DEFINITION] Autonomy
> How much the system decides and does on its own before a person is involved. An agent that only drafts a message has low autonomy; an agent that rolls back a deployment without asking has high autonomy. Section 1.5 gives a ladder of levels.

The idea is much older than language models. In the standard textbook on artificial intelligence, *Artificial Intelligence: A Modern Approach*, Stuart Russell and Peter Norvig define an agent as anything that perceives its environment through sensors and acts on it through actuators, and they describe a "rational agent" as one that chooses the action expected to do best given what it has perceived so far. A thermostat is an agent in that sense; so is a chess program; so is a robot. What is new since 2022 is that the "decide what to do next" part can be a general-purpose language model that reads the observations as text and writes its chosen action as text. That one substitution turned a forty-year-old concept into something you can build in an afternoon, and that is also why it is so easy to build badly.

{{FIG:ch1_loop|The agent loop. The context (everything observed so far) is given to the model; the model picks a tool call; the tool acts on the environment; the result is appended to the context, and the loop runs again until the model gives a final answer or the step budget runs out. The grey text above each block shows step 1 of the on-call example.}}

Here is the loop in code. The whole thing is about fifty lines, and it runs with no API key because the "model" is a **stub**: a few `if` statements that return exactly the tool calls a real model would be expected to make for this one scenario. That is deliberate. The point of the script is the *shape* of the loop (where the model sits, what it sees, what the code does with its reply), not intelligence. Everything a real agent adds later (a real model, real tools, memory, tracing) plugs into one of these slots.

```python
# code/agents/ch1_loop.py (the model is a stub; a real agent sends `messages` to an LLM here)
def model(messages):
    last = messages[-1]
    if last['role'] == 'user':
        return {'thought': 'An alert needs context first. Check recent errors for the service.',
                'tool': 'search_logs', 'args': {'service': 'checkout-api', 'minutes': 15}}
    if last['tool'] == 'search_logs':
        obs = last['content']
        return {'thought': f"{obs['errors']} errors, pool exhaustion, started 4 min after deploy ...",
                'tool': 'open_ticket', 'args': {'title': 'checkout-api: pool exhausted after v2.31', ...}}
    return {'thought': 'Ticket filed. Report back to the on-call engineer.', 'final': 'Opened OPS-4821 ...'}

def run_agent(task, max_steps=5):
    messages = [{'role': 'user', 'content': task}]
    for step in range(1, max_steps + 1):
        reply = model(messages)                                  # reason
        if 'final' in reply:                                     # the model decides it is done
            return reply['final']
        result = TOOLS[reply['tool']](**reply['args'])           # act
        messages.append({'role': 'tool', 'tool': reply['tool'], 'content': result})   # observe
    log('stopped: step budget exhausted')                        # a real loop always has a budget
```

Running it prints the **trace**: every thought, action and observation in order.

[![Terminal output of ch1_loop.py: the task, then three numbered steps, each with a THOUGHT line, an ACTION line calling search_logs or open_ticket, and an OBSERVE line with the tool result, ending with a FINAL line that reports the ticket](/img/agents/ch1_loop-run.png)](/img/agents/ch1_loop-run.png)

Three things in that output matter for the rest of the book. First, the model never touched the logs or the ticket system; it *asked* for things, and the loop did them. That is the boundary where permissions and guardrails live (Chapter 9). Second, every observation was appended to `messages`, so the second model call saw the alert *and* the log result. This growing list is the agent's **context**, and managing it is most of Chapter 2. Third, the loop had `max_steps=5`. A real model can loop forever on a task it cannot finish; a budget is not optional.

> [!DEFINITION] Context
> The full input the model receives on one turn of the loop: the instructions, the task, every tool call and observation so far, and whatever memory has been retrieved. The model's decision quality is bounded by what is in the context, and its cost is proportional to the context's length.

> [!DEFINITION] Trace
> The ordered record of one run: each model call, what it saw, what it decided, each tool call and its result, with timings and token counts. Chapter 8 makes traces first-class; in this chapter they are the printed log.

## 1.2 Model versus agent

Engineers often say "the agent hallucinated" or "the model did X" as if they were the same thing. They are not, and the difference decides where bugs live. A **model** is a function: text in, text out, stateless, with no goal and no ability to act. An **agent** is a system with a goal that runs over time; it has a loop, tools, some form of memory and the initiative to choose its next step. The model sits inside the agent as its decision-maker, the way a chess engine's evaluation function sits inside the engine.

> [!DEFINITION] Model (LLM)
> The trained network that turns an input text into a probability over the next token and, by sampling repeatedly, into an output text. Within an agent it is called once per turn and asked, in effect, "given all this, what should happen next?".

| | A model | An agent |
|---|---|---|
| Unit of work | one call | a task of many steps |
| State | none between calls | context, memory, tool results carried forward |
| Can act on the world | no, it only writes text | yes, through tools |
| Who chooses the next step | the caller | the model, inside the loop |
| Stops when | the response ends | the model says done, or a budget runs out |
| Failure looks like | a wrong or made-up answer | a wrong *action*, a loop that never ends, a correct answer to the wrong sub-task |
| How you test it | input, expected output | the whole trajectory: steps taken, tools used, final state |
| Cost | one prompt plus one completion | the sum over every turn, with a context that grows each turn |

An analogy helps. A model is like a brilliant consultant on the phone who has amnesia between calls: ask a question, get a sharp answer, hang up, and the next call starts from nothing. An agent is the same consultant with a notebook, a phone directory and a to-do list, allowed to make calls, read the replies, write them down and keep going until the job is done. The notebook is memory. The directory is the tool set. The to-do list is the goal. The consultant's intelligence did not change; what changed is that it can now *do* things and can now do them *wrong* in ways that persist.

{{FIG:ch1_model_vs_agent|Left: a model computes one output from one prompt. Right: an agent wraps the same model in a loop with a goal, a tool set and memory, and pursues the goal over many steps; the model is one component, not the whole.}}

> [!DEFINITION] Memory (state)
> Information kept between turns or between runs: the current context, scratch notes the agent writes for itself, and anything stored outside the context (a database, a file) and retrieved later. The context *is* short-term memory; everything else is a design choice (Chapter 2).

> [!DEFINITION] Trajectory
> The sequence of states, actions and observations of one run, from task to final answer. "Evaluating an agent" almost always means evaluating trajectories, not just the last message (Chapters 6 and 7).

The practical consequence: when an agent fails, ask *where in the loop* it failed. Did the model misread an observation (a model problem)? Was the right tool missing or badly described (a tool problem)? Did the loop keep going after the task was done (a loop problem)? Was the needed fact pushed out of a too-long context (a memory problem)? The same symptom, a wrong final answer, can come from any of them, and only the trace tells you which.

## 1.3 Workflows versus agents

Here is the most useful distinction in the field, and the one most often blurred in product pitches. Many systems call a language model several times, pass data between the calls and use tools, yet are not agents, because *the code decides the sequence*. Anthropic's December 2024 post "Building effective agents" draws the line clearly.

> [!PAPER] Anthropic, "Building effective agents" (Research blog, December 2024)
> [![Two bullet points from the post: Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks](/img/agents/ch1-anthropic-workflows-agents.png)](/img/agents/ch1-anthropic-workflows-agents.png)
>
> **Context:** the opening section, "What are agents?", after the authors note that "agent" means different things to different teams.
>
> **What it says:** workflows are "systems where LLMs and tools are orchestrated through predefined code paths"; agents are "systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks".
>
> **Why it matters:** the test is not "does it use tools" or "does it call the model more than once". The test is *who decides the path*. If you can draw the steps as a flowchart before the run starts, it is a workflow. If the model picks the next step at run time, it is an agent.

> [!DEFINITION] Workflow
> A system in which code fixes the sequence of steps, and one or more of the steps is a model call. Prompt chaining (call A feeds call B), routing (a model classifies, code dispatches), and parallel fan-out are all workflows. Chapter 4 covers the five standard workflow patterns.

The same post makes a second point that this whole chapter is built on: start simple.

> [!PAPER] Anthropic, "Building effective agents" (Research blog, December 2024)
> [![A paragraph from the post: When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense](/img/agents/ch1-anthropic-simplest.png)](/img/agents/ch1-anthropic-simplest.png)
>
> **Context:** the section "When (and when not) to use agents", immediately after the definitions.
>
> **What it says:** find "the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all." Agents "trade latency and cost for better task performance".
>
> **Why it matters:** this is a company that sells model calls telling you to make fewer of them. The reason is the arithmetic in Section 1.5: every extra step multiplies cost and divides reliability. The post adds that agents suit "open-ended problems where it's difficult or impossible to predict the required number of steps", and that you "must have some level of trust in its decision-making".

OpenAI's practical guide to building agents (a PDF published in April 2025) reaches the same place from the other side. It defines agents as systems that "independently accomplish tasks on your behalf", says plainly that applications which use an LLM but do not let it control the workflow ("simple chatbots, single-turn LLMs, or sentiment classifiers") are not agents, and then gives three criteria for when an agent is worth it.

> [!PAPER] OpenAI, "A practical guide to building agents" (PDF guide, April 2025) · page 6
> [![Page 6 of the guide: prioritize workflows that have previously resisted automation; three numbered criteria, complex decision-making, difficult-to-maintain rules, heavy reliance on unstructured data, each with an example; and the closing line that otherwise a deterministic solution may suffice](/img/agents/ch1-openai-when-to-build.png)](/img/agents/ch1-openai-when-to-build.png)
>
> **Context:** the section "When should you build an agent?", pages 5 and 6. The previous page contrasts a rules engine that "works like a checklist" with an agent that "functions more like a seasoned investigator".
>
> **What it says:** prioritise workflows "that have previously resisted automation", in particular those with "complex decision-making" (nuanced judgement, exceptions), "difficult-to-maintain rules" (rule sets that have become unwieldy), and "heavy reliance on unstructured data" (natural language, documents, conversation). "Otherwise, a deterministic solution may suffice."
>
> **Why it matters:** both vendors say the same thing in different words. An agent earns its cost only where the path cannot be written down in advance. If you can write it down, write it down.

Put the two posts together and you get a spectrum rather than a binary.

{{FIG:ch1_spectrum|The spectrum from a plain script to a multi-agent system. Moving right, the model takes over more of the decision about which step comes next; cost, variance between runs and testing effort all rise with it.}}

- **Script.** Code decides everything. A cron job, a rules engine, a SQL report. Deterministic, cheap, testable, and the right answer far more often than engineers who have just discovered agents want to admit.
- **Workflow with LLM steps.** Code decides the path; the model fills in steps that need language: classify this email, summarise this document, extract these fields. Most production "AI features" live here, and they should.
- **Agent.** The model decides the next step, including when to stop. Needed when the number and order of steps depends on what is discovered along the way: debugging, research, open-ended support, operating a browser.
- **Multi-agent system.** Several agents, usually one coordinating the others. Needed rarely, and mostly when the work parallelises or exceeds one context window. Chapter 5 is about when this helps and the many ways it fails.

> [!DEFINITION] Multi-agent system
> Two or more agents that exchange messages to complete one task, typically an orchestrator that splits the work and workers that each run their own loop. Each agent has its own context, which is both the benefit (isolation, parallelism) and the cost (coordination failures).

{{FIG:ch1_workflow_vs_agent|Left: a workflow runs the same code path on every input and calls the model inside fixed steps, so each step can be tested on its own. Right: an agent is given a set of tools and chooses the order, the number of steps and the stopping point at run time, so the whole trajectory must be tested.}}

The figure's right side explains why the rest of this book spends four chapters on evaluation and observability. In the workflow, you can unit-test "classify the topic" with a hundred labelled emails. In the agent, there is no fixed step to test: on one input it reads the email then looks up the account; on another it searches the knowledge base first; on a third it loops twice. You can only test the *behaviour*, over many runs, and you can only debug it from the trace.

> [!TIP] In production
> Before you call a design an agent, draw its flowchart. If every box and arrow is known before the run, you have a workflow, and you should build it as one: cheaper, faster, testable step by step. Reach for an agent only for the part of the diagram you genuinely cannot draw.

## 1.4 Where the idea came from

Five papers and two engineering posts, from October 2022 to June 2025, cover the whole arc from "a model can reason and act in turns" to "here is how multi-agent systems fail in production". Later chapters read several of them closely; this section gives one paragraph each so you know the names.

{{FIG:ch1_timeline|Timeline. ReAct (October 2022) made the reason-act-observe loop explicit; Toolformer, Reflexion and Generative Agents followed within six months in 2023; the production write-ups, workflows versus agents, the failure taxonomy and the multi-agent research system, arrived in 2024 and 2025.}}

**ReAct (October 2022).** Shunyu Yao and colleagues at Princeton and Google Research noticed that two lines of work had been kept apart: prompting a model to reason step by step (chain of thought), and prompting it to emit actions. ReAct interleaves them. The model writes a thought, then an action, then reads the observation, then writes the next thought. This is the loop of Section 1.1 written down as a prompting format, and it is the direct ancestor of every tool-using agent since. Chapter 3 reads the paper in full.

> [!PAPER] Yao et al. (2022), ReAct · Abstract · page 1
> [![Abstract of the ReAct paper with highlighted phrases: generate both reasoning traces and task-specific actions in an interleaved manner; reasoning traces help the model induce, track, and update action plans; actions allow it to interface with and gather additional information from external sources](/img/agents/ch1-react-abstract.png)](/img/agents/ch1-react-abstract.png)
>
> **Context:** the abstract, arXiv 2210.03629, first posted 6 October 2022; published at ICLR 2023.
>
> **What it says:** the model generates "both reasoning traces and task-specific actions in an interleaved manner". Reasoning "help[s] the model induce, track, and update action plans as well as handle exceptions", while actions "allow it to interface with and gather additional information from external sources such as knowledge bases or environments".
>
> **Why it matters:** this single sentence is the job description of an agent's model: plan, act, read, re-plan. Everything in Section 1.1 is an implementation of it.

> [!PAPER] Yao et al. (2022), ReAct · Figure 1 · page 2
> [![Figure 1 of the ReAct paper: four prompting methods on a HotpotQA question (standard, chain-of-thought, act-only, ReAct) and two on an AlfWorld task (act-only, ReAct), shown as transcripts of Thought, Act and Obs lines; the ReAct columns end with a tick, the others with a cross](/img/agents/ch1-react-figure1.png)](/img/agents/ch1-react-figure1.png)
>
> **Context:** the paper's only overview figure, comparing the methods on a question-answering task (top) and a text game (bottom).
>
> **What it says:** reasoning alone (1b) hallucinates a fact; acting alone (1c) runs searches but cannot put the results together; interleaving the two (1d) reaches the right answer. In the text game (2a versus 2b), the act-only agent repeats a failing action, while the thinking agent plans where to look.
>
> **Why it matters:** look at the labels in the right-hand panels: `Thought`, `Act`, `Obs`. Your production traces in Chapter 8 will have exactly these three kinds of line, and the failure modes in the left-hand panels (made-up facts, repeated actions) are still the top two things traces catch.

**Toolformer (February 2023).** Timo Schick and colleagues at Meta AI asked a different question: instead of *prompting* a model to use tools, could a model *learn* to call them? Toolformer takes a plain language model and teaches it, from a handful of examples per tool and a self-supervised filtering trick, to insert API calls (a calculator, a search engine, a calendar) into its own text where they reduce its prediction error. It matters here because it shows that tool use is not a prompt hack: it can be a trained ability, and the "function calling" features now built into commercial models are the productised descendants of this idea.

> [!PAPER] Schick et al. (2023), Toolformer · Abstract · page 1
> [![Abstract of the Toolformer paper with highlighted phrases: LMs can teach themselves to use external tools via simple APIs; decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results](/img/agents/ch1-toolformer-abstract.png)](/img/agents/ch1-toolformer-abstract.png)
>
> **Context:** the abstract, arXiv 2302.04761, posted 9 February 2023; published at NeurIPS 2023.
>
> **What it says:** language models "can teach themselves to use external tools via simple APIs". The model is "trained to decide which APIs to call, when to call them, what arguments to pass, and how to best incorporate the results into future token prediction", in "a self-supervised way".
>
> **Why it matters:** read the four decisions in that sentence again: which tool, when, with what arguments, and what to do with the result. Those are the four things that can go wrong on every turn of the loop in Section 1.1, and Chapter 2 is about making each of them easy for the model.

**Reflexion (March 2023).** Noah Shinn and colleagues took the loop one level up. When an agent fails a task, Reflexion asks the model to write a short verbal reflection on *why* it failed, stores that text in an episodic memory, and includes it in the next attempt. No weights are updated; the "learning" is in the text. The result was large gains on coding and reasoning benchmarks, and the pattern (try, critique, retry with the critique in context) is now a standard tool in the box. Chapter 3 covers it as the reflection pattern.

> [!PAPER] Shinn et al. (2023), Reflexion · Abstract · page 1
> [![Abstract of the Reflexion paper with highlighted phrases: not by updating weights, but instead through linguistic feedback; verbally reflect on task feedback signals; episodic memory buffer](/img/agents/ch1-reflexion-abstract.png)](/img/agents/ch1-reflexion-abstract.png)
>
> **Context:** the abstract, arXiv 2303.11366, posted 20 March 2023; published at NeurIPS 2023.
>
> **What it says:** Reflexion reinforces agents "not by updating weights, but instead through linguistic feedback". Agents "verbally reflect on task feedback signals, then maintain their own reflective text in an episodic memory buffer to induce better decision-making in subsequent trials".
>
> **Why it matters:** this is the first widely cited example of an agent improving *across runs* by writing to memory, which is a different thing from improving *within a run* by reading observations. It is also a warning: the paper itself notes that the method relies on the model's ability to judge its own work, and that there is no guarantee of success. Self-critique is a tool, not a safety net.

**Generative Agents (April 2023).** Joon Sung Park and colleagues at Stanford and Google put twenty-five agents in a small simulated town, each with a memory stream of everything it had observed, a mechanism to retrieve relevant memories, and a step that periodically turned memories into higher-level reflections and plans. The agents woke up, cooked breakfast, went to work, held conversations, and, when one was told she wanted to throw a party, spread the invitation through the town so that others showed up. It is a research demonstration rather than a production system, but it established the memory architecture (observe, store, retrieve, reflect, plan) that Chapter 2 builds on, and it is the paper that made "agents" a mainstream word.

> [!PAPER] Park et al. (2023), Generative Agents · Figure 1 · page 1
> [![Figure 1 of the Generative Agents paper: a pixel-art map of the town of Smallville with call-outs showing agents taking a walk in the park, joining for coffee at a cafe, arriving at school, sharing news with colleagues and finishing a morning routine, with short dialogue snippets](/img/agents/ch1-genagents-figure1.png)](/img/agents/ch1-genagents-figure1.png)
>
> **Context:** the teaser figure at the top of page 1, arXiv 2304.03442, posted 7 April 2023; published at UIST 2023.
>
> **What it says:** generative agents "populat[e] a sandbox environment, reminiscent of The Sims, with twenty-five agents", and users can "observe and intervene as agents plan their days, share news, form relationships, and coordinate group activities".
>
> **Why it matters:** each little character is a full agent loop with its own memory. The paper's contribution is not the game; it is the finding that believable long-running behaviour needs a *memory architecture*, not just a bigger prompt. That is the problem a production agent hits the first time a task runs longer than one context window.

**"Why Do Multi-Agent LLM Systems Fail?" (March 2025).** Two years after the burst of 2023, Mert Cemri, Melissa Pan, Shuyi Yang and colleagues at UC Berkeley asked the uncomfortable question in their title. They collected execution traces from popular open-source multi-agent frameworks, had experts annotate where each failed, and built a taxonomy (MAST) of fourteen failure modes in three groups. The version of the paper published at NeurIPS 2025 reports over 1,600 annotated traces across seven frameworks, failure rates between 41% and 86.7% on the systems studied, and a split of failures into system design issues (44.2%), misalignment between agents (32.3%) and weak task verification (23.5%).

> [!PAPER] Cemri et al. (2025), MAST · Abstract · page 1
> [![Abstract of the paper with highlighted phrases: 1600+ annotated traces collected across 7 popular MAS frameworks; 14 unique modes, clustered into 3 categories; system design issues; inter-agent misalignment; task verification](/img/agents/ch1-mast-abstract.png)](/img/agents/ch1-mast-abstract.png)
>
> **Context:** the abstract, arXiv 2503.13657, first posted 17 March 2025; the screenshot is from version 3 (October 2025), accepted at the NeurIPS 2025 Datasets and Benchmarks track.
>
> **What it says:** a dataset of "1600+ annotated traces collected across 7 popular MAS frameworks", and a taxonomy of "14 unique modes, clustered into 3 categories: (i) system design issues, (ii) inter-agent misalignment, and (iii) task verification".
>
> **Why it matters:** the first category is the largest. Most failures were not the model being stupid; they were the *system* being badly specified: agents disobeying the task or their role, repeating steps, losing conversation history, not knowing when to stop. Those are engineering problems, and they are this book's subject.

> [!PAPER] Cemri et al. (2025), MAST · Figure 1 · page 2
> [![Figure 1 of the paper: a table of fourteen failure modes grouped into system design issues (44.2%), inter-agent misalignment (32.3%) and task verification (23.5%), each with its share of failures and the stage of the conversation in which it occurs](/img/agents/ch1-mast-figure1.png)](/img/agents/ch1-mast-figure1.png)
>
> **Context:** the taxonomy figure, with the percentage of all observed failures attributed to each mode across 1,642 traces.
>
> **What it says:** the single most common mode is "step repetition" (15.7%), followed by "reasoning-action mismatch" (13.2%), "unaware of termination conditions" (12.4%) and "disobey task specification" (11.8%). Verification failures ("no or incomplete verification", "incorrect verification") together account for 17.3%.
>
> **Why it matters:** three of the top four modes are things a plain loop with a budget, a clear stop condition and a trace would surface immediately. Keep this figure in mind in Chapter 5; it is the checklist for multi-agent designs.

Two engineering posts close the timeline, and Section 1.6 reads both: Anthropic's "Building effective agents" (December 2024), which gave the field the workflow-versus-agent vocabulary, and Anthropic's account of its multi-agent research system (June 2025), which is the most detailed public description of a production orchestrator-and-subagents design, including its evaluation.

## 1.5 Levels of autonomy and the economics of agents

Two questions decide whether an agent design is viable before a single line is written: how much are you letting it do without asking, and what does each step cost in reliability, money and time.

### The autonomy ladder

| Level | The agent... | A person... | Example | What a wrong step costs |
|---|---|---|---|---|
| 1. Suggest | reads, reasons, proposes | reads the proposal and acts | drafts a reply to a support email | a bad draft, caught before sending |
| 2. Act with approval | prepares an action and asks | clicks approve or reject | proposes a rollback and waits | one click of attention per action |
| 3. Act and report | acts, then says what it did | reviews the report, can undo | rolls back, then posts to the channel | a wrong action, visible quickly, reversible if designed so |
| 4. Fully autonomous | acts; nobody looks | is alerted only on failure | triages and closes tickets overnight | wrong actions that compound before anyone sees them |

{{FIG:ch1_autonomy|Four levels of autonomy, from suggesting to acting without review. Each level up saves human time per task and widens the blast radius of a single wrong step.}}

> [!DEFINITION] Human in the loop (approval step)
> A point in the loop where a proposed action is shown to a person and only runs if they approve. It converts a level-3 or level-4 agent into a level-2 one for that action, at the cost of a person's time and attention.

> [!DEFINITION] Blast radius
> How much damage one wrong action can do before it is noticed and undone: one draft email versus every email in a queue, one ticket versus a production database. Autonomy should rise only as blast radius falls or reversibility rises.

The ladder is not a maturity model where level 4 is the goal. Most production agents today, including the three in Section 1.6, run at levels 1 and 2, and that is a design decision, not a shortcoming. The right level for an action depends on two things: how reliable the agent is at that action (measured, not assumed) and how reversible the action is. Reading logs can be level 4 from day one. Deleting records should be level 2 until you have the evidence to argue otherwise.

### The reliability arithmetic

An agent that takes ten steps must get all ten right. If each step succeeds independently with probability $$p$$, the whole task succeeds with probability

$$
P(\text{task}) = p^{n}
$$

for $$n$$ steps. Independence is an approximation (errors often correlate, and a good agent can recover from some), but the shape of the result is what matters, and it is brutal. The script `ch1_math.py` computes the table.

[![Terminal output of ch1_math.py: a table of success probability for p of 0.9, 0.95, 0.99 and 0.999 across 1, 5, 10, 20 and 50 steps; the number of steps each p affords before success falls below 90 percent and 50 percent; a token-cost table for one call, an eight-step agent and a lead agent with four subagents at an example price; and a latency line](/img/agents/ch1_math-run.png)](/img/agents/ch1_math-run.png)

| step reliability $$p$$ | 5 steps | 10 steps | 20 steps | 50 steps | steps that keep success above 90% |
|---|---|---|---|---|---|
| 0.90 | 59.0% | 34.9% | 12.2% | 0.5% | 1 |
| 0.95 | 77.4% | 59.9% | 35.8% | 7.7% | 2 |
| 0.99 | 95.1% | 90.4% | 81.8% | 60.5% | 10 |
| 0.999 | 99.5% | 99.0% | 98.0% | 95.1% | 105 |

{{FIG:ch1_reliability|Probability that a task of n steps succeeds when each step succeeds with probability p. At p equal to 0.90, a ten-step task succeeds about a third of the time; a step reliability of 0.99 buys ten steps above the 90 percent line; only 0.999 stays above it for a hundred steps.}}

Read the table as a budget. A model that is right 90% of the time on each step, which sounds respectable, gives you a ten-step agent that fails two times in three. To run ten steps with 90% task success you need every step at 99%. To run fifty steps you need 99.9%, which is the territory of carefully constrained tools, strict output formats and retries, not of free-form reasoning. This is why so many agent demos work beautifully and so few agent products ship: the demo is one run, and the product is $$p^{n}$$ over ten thousand runs.

> [!DEFINITION] Step reliability
> The probability that one turn of the loop does the right thing: picks a sensible action, calls the tool with valid arguments, and reads the result correctly. It is a measured quantity (from evals and traces), different for different kinds of step, and the single most important number in an agent's design.

Three design moves follow directly from the formula, and each is a later chapter:

1. **Reduce $$n$$.** Fewer, bigger, better-designed tools (Chapter 2); fixed workflow steps for the parts that do not need judgement (Chapter 4).
2. **Raise $$p$$.** Clear instructions, good tool descriptions, structured outputs, and the reasoning patterns of Chapter 3; evals that tell you which kind of step is weak (Chapter 6).
3. **Break the independence.** Verification steps, retries and self-checks turn a chain of must-succeed steps into one with recovery paths, so a single slip no longer sinks the run (Chapters 3 and 9).

> [!WARNING]
> The formula also explains why "it worked in testing" is not evidence. If you ran a 10-step agent five times and it passed, that is consistent with a per-step reliability as low as 0.95 (a 60% task success rate), which would fail four thousand of your first ten thousand users. Chapter 7 is about measuring this before they do.

### Token and latency cost

Every turn of the loop sends the whole context to the model again. An eight-step agent whose context averages 3,000 input tokens per step therefore sends about 24,000 input tokens, against 3,000 for a single call. To make that concrete, `ch1_math.py` prices it at an **example** price of $3 per million input tokens and $15 per million output tokens. These are round numbers chosen for the arithmetic, not a quote from any provider; substitute your own.

| system | model calls | input tokens | output tokens | cost per task | cost per 10,000 tasks |
|---|---|---|---|---|---|
| one LLM call | 1 | 3,000 | 500 | $0.0165 | $165 |
| agent, 8 steps | 8 | 24,000 | 3,200 | $0.1200 | $1,200 |
| lead agent plus 4 subagents of 8 steps | 33 | 120,000 | 16,000 | $0.6000 | $6,000 |

{{FIG:ch1_cost|The same task as one model call, as an eight-step agent, and as a lead agent with four subagents of eight steps each, at an example price. The agent costs about seven times the single call and the multi-agent system about thirty-six times; the sequential agent also takes about eight times as long.}}

The agent costs 7.3 times the single call and the multi-agent system 36 times, before counting retries. Latency stacks the same way: if a model call takes about two seconds, eight sequential steps take sixteen, plus tool time. Anthropic's production numbers, in Section 1.6, are in the same range: agents used about four times the tokens of a chat interaction and multi-agent systems about fifteen times.

None of this says agents are too expensive. It says the task has to be worth it: an agent that saves an engineer twenty minutes of log-reading is worth far more than $0.12. But it does mean that every design decision about steps, context length and model size is a cost decision, and that you need to *see* those numbers per run, which is what tracing (Chapter 8) gives you.

> [!TIP] In production
> Put a budget on every run from the first prototype: a maximum number of steps, a maximum number of tokens, and a wall-clock timeout. Log how much of each budget a run used. Runs that hit the ceiling are the first traces to read; they are where the loop is going wrong.

This arithmetic is why evals and tracing sit at the heart of this book rather than in an appendix. $$p$$ cannot be guessed; it must be measured per kind of step, which is an eval. $$n$$ and the token count cannot be reasoned about from the code; they emerge at run time, which is a trace. An agent without both is a system whose reliability and cost are unknown by construction.

## 1.6 What companies actually run

Three systems, chosen because their teams published enough detail to learn from and because the claims can be checked against the published text. For each: what the agent does, the architecture in one sentence, how it is evaluated, and what the team said went wrong or surprised them. Where a post gives no number, this section says so rather than inventing one.

### (a) Uber's Genie: an on-call copilot in Slack

Uber's engineering teams run Slack channels where internal users ask for help with platforms such as Michelangelo, Uber's machine-learning platform. The October 2024 post "Genie: Uber's Gen AI On-Call Copilot" describes the bot built to answer those questions.

> [!PAPER] Uber, "Genie: Uber's Gen AI On-Call Copilot" (Engineering blog, October 2024)
> [![A paragraph from the post: at Uber, different teams like the Michelangelo team have Slack support channels where their internal users can ask for help; people ask around 45,000 questions on these channels each month; high question volumes and long response wait times reduce productivity for users and on-call engineers](/img/agents/ch1-uber-genie-45k.png)](/img/agents/ch1-uber-genie-45k.png)
>
> **Context:** the introduction, stating the problem.
>
> **What it says:** "People ask around 45,000 questions on these channels each month." Long wait times "reduce productivity for users and on-call engineers".
>
> **Why it matters:** this is the shape of a good agent problem: high volume, repetitive, answerable from existing documents, and with a clear human fallback (the on-call engineer) when the system cannot help.

**What it does.** A user posts a question in a Slack channel; Genie answers in the thread with a response grounded in Uber's internal documentation, with citations, and offers feedback buttons. If the answer does not help, the on-call engineer picks it up as before.

**Architecture in one sentence.** Retrieval-augmented generation wrapped in a Slack bot: documents are chunked and embedded into a vector database; a back-end "Knowledge Service" embeds the incoming question, fetches the most relevant chunks, and sends them with the question to an LLM through Uber's Michelangelo gateway. The post says the team chose RAG over fine-tuning because it needs no training examples to start, which "reduced the time to market".

{{FIG:ch1_genie|Genie as the Uber blog describes it. Internal documents are chunked and embedded into a vector database; a Slack question is embedded, matched to chunks and answered by an LLM with citations; user feedback and LLM-as-a-judge evaluations feed back to the owners of the documents.}}

> [!DEFINITION] Retrieval-augmented generation (RAG)
> Fetching the passages most relevant to a question from a document store and putting them in the model's context before it answers, so that the answer is grounded in your documents rather than in the model's training data. This site's RAG book covers it from first principles.

Be honest about the label. As described in the 2024 post, Genie is closer to a workflow than to an agent in the Section 1.3 sense: the path (embed, retrieve, generate, reply) is fixed in code, and the model fills in the generation step. It is in this chapter because its *evaluation loop* is exactly what agentic systems need, and because it shows the simplest-solution principle in practice: a fixed pipeline answered tens of thousands of questions before anyone needed to let the model choose steps.

**How they evaluate it.** Two mechanisms. First, feedback buttons on every answer.

> [!PAPER] Uber, "Genie: Uber's Gen AI On-Call Copilot" (Engineering blog, October 2024)
> [![A bulleted list from the post: Resolved, the answer completely resolved the issue; Helpful, the answer partially helped but the user needs more help; Not Helpful, the response is wrong or not relevant; Not Relevant, the user needs help from someone on call and Genie cannot assist, like for a code review](/img/agents/ch1-uber-genie-feedback.png)](/img/agents/ch1-uber-genie-feedback.png)
>
> **Context:** the section on user feedback, listing the four options under every answer.
>
> **What it says:** "Resolved", "Helpful", "Not Helpful" and "Not Relevant", with a definition of each, including the honest fourth case where the question was never one the bot could answer.
>
> **Why it matters:** this is an online eval with a four-point label scheme, defined in advance so that the numbers mean something. Note the separation of "wrong" from "not a question for this system"; without it, a helpfulness rate blends two different failures.

Second, channel owners can run custom offline evaluations: the post describes an evaluator that fetches a specified prompt, runs "LLM as a Judge", and extracts the metrics the owner cares about, such as hallucination and answer relevancy, so that teams can improve the documents the bot reads.

> [!DEFINITION] LLM-as-a-judge
> Using a language model, with a rubric, to score the outputs of another model or system: is this answer grounded in the retrieved text, is it relevant, is it complete. Cheap and scalable, and only as good as its calibration against human judgements (Chapter 6).

**The numbers the post reports.**

> [!PAPER] Uber, "Genie: Uber's Gen AI On-Call Copilot" (Engineering blog, October 2024)
> [![A paragraph from the post: since its launch in September 2023, Genie has expanded its presence to 154 Slack channels and has answered over 70,000 questions; Genie boasts a 48.9 percent helpfulness rate; the team estimates it has saved 13,000 engineering hours since launch](/img/agents/ch1-uber-genie-results.png)](/img/agents/ch1-uber-genie-results.png)
>
> **Context:** the results section near the end of the post.
>
> **What it says:** since launch in September 2023, 154 Slack channels, over 70,000 questions answered, a "48.9% helpfulness rate", and an estimated 13,000 engineering hours saved.
>
> **Why it matters:** a 48.9% helpfulness rate is a real production number, not a demo number, and the team published it. Roughly half of answers were judged helpful or better by the person asking; the other half went to a human as before. For a system that costs a few cents per answer and sits in front of a human fallback, that is a success, and it is the kind of honest figure you should expect from your own first agent.

**What they learned.** The post does not give a failure analysis, so this section will not invent one. What it does show is the design pattern worth copying: launch a fixed pipeline, instrument every answer with a cheap label, give the owners of the knowledge a way to measure their own slice, and let the measured helpfulness rate drive what to improve.

### (b) Anthropic's multi-agent research system

In June 2025 Anthropic published a detailed account of how its Research feature works: a system that, given an open question, searches the web and internal sources for several minutes and returns a cited report.

**What it does.** Takes a research question, plans an approach, runs several searches in parallel, reads the results, decides whether more is needed, and writes a report with citations.

**Architecture in one sentence.** An orchestrator-worker design: a lead agent (Claude Opus 4) plans the research, saves its plan to memory, spawns subagents (Claude Sonnet 4) that each search a sub-question in parallel with their own tools and context, synthesises what they return, and hands the draft to a citation agent that attaches sources.

{{FIG:ch1_research|The orchestrator-worker design of the multi-agent research system as the post describes it: a lead agent plans and delegates, several subagents search in parallel with their own tools, the lead synthesises their findings, and a citation agent attaches sources.}}

> [!DEFINITION] Orchestrator (lead agent)
> The agent in a multi-agent system that owns the overall task: it splits the work, assigns pieces to worker agents, collects their results and decides what to do next. Workers (subagents) each run their own loop on their piece and report back.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025)
> [![A paragraph from the post: internal evaluations show that multi-agent research systems excel especially for breadth-first queries that involve pursuing multiple independent directions simultaneously; a multi-agent system with Claude Opus 4 as the lead agent and Claude Sonnet 4 subagents outperformed single-agent Claude Opus 4 by 90.2 percent on the internal research eval; an example about finding all board members of the Information Technology companies in the S&P 500](/img/agents/ch1-anthropic-research-90.png)](/img/agents/ch1-anthropic-research-90.png)
>
> **Context:** the section "Benefits of a multi-agent system", stating the headline result.
>
> **What it says:** the multi-agent system "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval", especially "for breadth-first queries that involve pursuing multiple independent directions simultaneously". The example: finding all board members of the technology companies in the S&P 500, which the single agent failed with "slow, sequential searches".
>
> **Why it matters:** two cautions in reading it. It is an *internal* eval on tasks the system was built for, not a public benchmark; and the comparison is to a single agent, not to a workflow. What the number does show is the one case where multi-agent reliably wins: work that parallelises into independent searches and exceeds one context window.

The post is unusually frank about cost.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025)
> [![A paragraph from the post: there is a downside, in practice these architectures burn through tokens fast; agents typically use about 4 times more tokens than chat interactions, and multi-agent systems about 15 times more tokens than chats; multi-agent systems require tasks where the value is high enough to pay for the increased performance; domains that require all agents to share the same context, such as most coding tasks, are not a good fit; multi-agent systems excel at tasks that involve heavy parallelization, information that exceeds single context windows, and interfacing with numerous complex tools](/img/agents/ch1-anthropic-research-tokens.png)](/img/agents/ch1-anthropic-research-tokens.png)
>
> **Context:** the same section, the paragraph after the 90.2% result.
>
> **What it says:** "agents typically use about 4× more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats." Multi-agent "require[s] tasks where the value of the task is high enough to pay for the increased performance". Domains "that require all agents to share the same context", such as "most coding tasks", "are not a good fit".
>
> **Why it matters:** this is Section 1.5 measured in production: the 15× is the same order as the 36× in the example table, and the "value high enough to pay" sentence is the economics in one line. The post also reports that in its analysis of the BrowseComp benchmark, token usage alone explained 80% of the variance in performance: the systems that did better mostly did so by spending more.

**How they evaluate it.** The post's section on evaluation is the most useful part for this book, and Chapter 7 returns to it.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025)
> [![A paragraph from the post headed Start evaluating immediately with small samples: in early agent development changes have dramatic impacts because there is abundant low-hanging fruit; a prompt tweak might boost success rates from 30 to 80 percent; the team started with a set of about 20 queries representing real usage patterns; it is best to start small-scale testing right away rather than delaying until you can build more thorough evals](/img/agents/ch1-anthropic-research-evals.png)](/img/agents/ch1-anthropic-research-evals.png)
>
> **Context:** the section "Effective evaluation of agents", first principle.
>
> **What it says:** "Start evaluating immediately with small samples." Early on, "a prompt tweak might boost success rates from 30% to 80%", so "you can spot changes with just a few test cases"; the team "started with a set of about 20 queries representing real usage patterns" rather than waiting for hundreds.
>
> **Why it matters:** the most common reason teams have no evals is that they think evals must be big. Twenty real queries and a pass/fail judgement is enough to see a 30-to-80 jump, and that is enough to start.

The same section describes the rest of the method: an LLM judge scoring each answer on a rubric (factual accuracy, citation accuracy, completeness, source quality, tool efficiency), where "a single LLM call with a single prompt outputting scores from 0.0-1.0 and a pass-fail grade was the most consistent"; and human testers, because "people testing agents find edge cases that evals miss", such as the agents' early preference for SEO-optimised content farms over authoritative sources.

**What they learned.** The post lists production problems that any long-running agent will meet: agents are stateful and errors compound, so a minor failure can derail a run; the team added the ability to resume from where an error occurred rather than restart; it used "rainbow deployments" that keep old and new versions running side by side so that an in-flight agent is not disrupted by a deploy; and it found that full tracing of agent decision patterns was needed to diagnose failures without reading users' conversations. Every one of those is a chapter of this book.

### (c) LinkedIn's generative AI product

The third system is older and less agentic than the second, and is here for its honesty about evaluation. In April 2024 two LinkedIn engineers, Juan Pablo Bottaro and Karthik R., published "Musings on Building a Generative AI Product", describing the feature that lets a member ask questions about a job post or a feed item and get a tailored answer.

**What it does.** A member asks, for example, how well they fit a job; the system routes the question, gathers the member's profile and the job through internal APIs, and writes an answer.

**Architecture in one sentence.** A router decides whether the question is in scope and which topic-specific "AI agent" should handle it; that agent runs a retrieval step (recall-oriented: internal APIs and search) and a generation step (precision-oriented: write from what was found), with the internal APIs wrapped as "skills" the model can call.

{{FIG:ch1_linkedin|LinkedIn's design as the blog describes it: a router decides whether a question is in scope and which topic agent handles it; that agent retrieves from internal APIs and search, then generates an answer; a linguist team scores hundreds of conversations a day.}}

In Section 1.3 terms this is a routing workflow whose branches call tools; the authors use the word "agent" for each branch. The pattern (classify, dispatch, retrieve, generate) is Chapter 4's routing pattern, and it is the most common production architecture in the industry.

**How they evaluate it, and what was hard.**

> [!PAPER] LinkedIn, "Musings on Building a Generative AI Product" (Engineering blog, April 2024)
> [![A paragraph from the post: tuning routing and retrieval felt natural given their classification nature, the team built dev sets and fitted them with prompt engineering and in-house models; generation was a different story and followed the 80/20 rule, getting it 80 percent was fast but the last 20 percent took most of the work; when the expectation is that 99 percent or more of answers should be great, even the most advanced models still require a lot of work and creativity to gain every 1 percent](/img/agents/ch1-linkedin-8020.png)](/img/agents/ch1-linkedin-8020.png)
>
> **Context:** the section on development speed, after the authors note that the basic pipeline took only days to stand up.
>
> **What it says:** routing and retrieval were tuned like classifiers, with dev sets. Generation "followed the 80/20 rule; getting it 80% was fast, but that last 20% took most of our work. When the expectation from your product is that 99%+ of your answers should be great, even using the most advanced models available still requires a lot of work and creativity to gain every 1%."
>
> **Why it matters:** this is the single most repeated finding in production LLM write-ups, and it is the reliability arithmetic again. The demo is the 80%. The product is the last 20%, and you cannot find it without measuring.

> [!PAPER] LinkedIn, "Musings on Building a Generative AI Product" (Engineering blog, April 2024)
> [![A bullet from the post headed Scaling annotation: initially everyone in the team chimed in, but a more principled approach with consistent and diverse annotators was needed; the internal linguist team built tooling and processes to evaluate up to 500 daily conversations and get metrics around overall quality score, hallucination rate, Responsible AI violation, coherence, style; this became the main signpost to understand trends, iterate on prompts and decide readiness to go live](/img/agents/ch1-linkedin-evals.png)](/img/agents/ch1-linkedin-evals.png)
>
> **Context:** the section on evaluation, which the authors say was harder than expected: first agreeing on guidelines for what a good answer is, then scaling the annotation.
>
> **What it says:** an internal linguist team built tooling to "evaluate up to 500 daily conversations" with metrics for "overall quality score, hallucination rate, Responsible AI violation, coherence, style". This "became our main signpost to understand trends, iterate on prompts & ensure we were ready to go live".
>
> **Why it matters:** five hundred human-scored conversations a day is an unusual investment, and the authors say it was the thing that told them when they were ready to launch. The next bullet of the post calls automatic evaluation "the holy grail, but still a work in progress", which is an honest statement of where LLM-as-a-judge stood in 2024 and a theme Chapter 6 takes up.

**What went wrong.** The post describes the model producing malformed structured output when calling the internal APIs: about 10% of responses had parameter mistakes, which the team reduced to about 0.01% by analysing the common errors, writing code to detect and patch them before parsing, and adding hints to the prompt.

> [!PAPER] LinkedIn, "Musings on Building a Generative AI Product" (Engineering blog, April 2024)
> [![A paragraph from the post: through an analysis of various payloads the team determined common mistakes made by the LLM and wrote code to detect and patch these appropriately before parsing; they also modified prompts to inject hints around some of these common mistakes; they were ultimately able to reduce occurrences of these errors to about 0.01 percent](/img/agents/ch1-linkedin-yaml.png)](/img/agents/ch1-linkedin-yaml.png)
>
> **Context:** the section on calling internal APIs, where the model emits a structured payload that the system parses to call a skill.
>
> **What it says:** the team analysed the payloads, found the model's common mistakes, "wrote code to detect and patch these appropriately before parsing", added prompt hints, and reduced the errors "to ~0.01%".
>
> **Why it matters:** this is step reliability being raised from 0.90 to 0.9999 on one kind of step, by engineering around the model rather than waiting for a better one. It is also a preview of Chapter 2: the format in which a model emits a tool call is a design surface, and a forgiving parser is part of the tool.

The post also reports a latency trade-off that every agent designer meets: chain-of-thought reasoning improved quality and reduced hallucination but added tokens "that the member never sees", increasing perceived latency; and that watching only time-to-first-token hid a degradation in time between tokens that set off alerts during a public ramp and forced a capacity increase. Measure both.

### What the three have in common

| | Uber Genie | Anthropic Research | LinkedIn |
|---|---|---|---|
| Spectrum position (§1.3) | workflow (RAG pipeline) | multi-agent | routing workflow with tool-calling branches |
| Autonomy level (§1.5) | 1: suggests an answer, human fallback | 1: produces a report | 1: produces an answer |
| Online eval | four-label feedback buttons | user feedback (described, no numbers given) | not described |
| Offline eval | LLM-as-a-judge on owner-chosen metrics | ~20 queries to start, LLM judge with a rubric, human testers | linguist team, up to 500 conversations a day, five metrics |
| Published numbers | 45k questions/month, 154 channels, 70k answered, 48.9% helpful, 13k hours | +90.2% versus single agent (internal), 4× and 15× tokens | ~10% to ~0.01% payload errors; "80% fast, last 20% most of the work" |
| Hardest part, in their words | (not stated) | evaluation, compounding errors, deployment of stateful agents | evaluation guidelines and annotation; the last 20% of quality |

Notice what is absent: none of the three runs at autonomy level 3 or 4, none of the three skipped evaluation, and two of the three are workflows rather than agents by the strict definition. The companies with the most experience chose the simplest design that worked and put their effort into measurement.

## 1.7 When NOT to build an agent

Here is the checklist this chapter has been building towards. Go through it before writing a prompt. An agent is the wrong answer if any of the following holds.

1. **Deterministic logic is enough.** If you can write the steps as code and they are the same for every input, write the code. A model adds cost, latency and variance, and removes testability, for no gain.
2. **The cost or latency budget does not fit $$n$$ steps.** If the task must complete in two seconds or cost under a cent, an eight-step agent is out before you start. Do the Section 1.5 arithmetic with your own numbers.
3. **The accuracy requirement is above what $$p^{n}$$ allows.** If the task must succeed 99% of the time and you have ten steps, you need $$p \approx 0.999$$ per step. If you cannot measure $$p$$ or cannot reach it, reduce $$n$$ or use a workflow.
4. **There is no way to evaluate it.** If you cannot say what a correct outcome looks like for a few dozen representative inputs, you cannot tell whether the agent works, cannot tell whether a change helped, and cannot tell when it has silently got worse. Build the eval set first; it will also tell you whether you needed an agent.
5. **The actions are irreversible and there is no approval step.** Sending money, deleting data, emailing customers, changing production configuration. Either design the approval step (level 2) and the undo path, or do not let the model act.

{{FIG:ch1_decision|A decision flow for whether to build an agent. Most tasks leave at the first two questions; an agent is the answer only when the steps vary between runs, success can be measured, the budget fits the step count, and wrong actions can be undone or approved.}}

> [!DEFINITION] Eval (evaluation set)
> A set of representative inputs with a definition of success for each (an expected answer, a rubric, a checker), run against the system to produce a score. Chapters 6 and 7 build them for agents; here the point is that if one cannot be written, the system cannot be trusted.

> [!DEFINITION] Guardrail
> A check that runs outside the model, on its inputs, outputs or proposed actions, and blocks or alters them when they violate a rule: a permission list for tools, a schema check on outputs, a classifier on inputs, an approval step on dangerous actions. Chapter 9 is about guardrails.

Three worked scenarios, deliberately mundane:

**Scenario 1: turn the office lights off at 7 pm.** Checklist item 1 fails immediately: a cron job and one API call do this perfectly, for free, forever. There is no judgement, no variation between runs and no natural language. Anyone proposing an agent here has confused "uses AI" with "is good". *Answer: a script.*

**Scenario 2: triage incoming customer emails.** Item 1 passes: emails are unstructured, and the decision (billing? bug? refund? spam?) needs language understanding. Item 2 passes: a few seconds and a fraction of a cent per email is fine. Item 3 and 4 need work: you need a few hundred labelled emails to measure the classifier's accuracy, and you need to decide what accuracy is acceptable given that a human reads the result. Item 5 passes if the action is "put in a queue with a suggested label" (level 1) rather than "reply automatically". But look at the path: classify, look up the account, draft a reply, queue for review. It is the same every time. *Answer: a routing workflow with model steps, at autonomy level 1. Not an agent.* Promote the "draft a reply" step to a small agent with tools (search the knowledge base, read the order history) only when the eval shows the fixed path is what limits quality.

**Scenario 3: flag transactions over a threshold.** Item 1 fails: `amount > threshold` is one line of SQL, and a model would be slower, dearer and occasionally wrong at comparing two numbers. *Answer: a rule.* The interesting version is the one OpenAI's guide uses: fraud *investigation*, where the rule has flagged a transaction and someone must now read the account history, look for patterns and decide. That has variable steps, unstructured evidence and judgement, and could justify an agent at level 1 (write up the case for a human analyst) once there is an eval set of past investigations to measure it against.

> [!TIP] In production
> The most reliable way to decide is to build the workflow version first. It gives you a baseline, an eval set and traces for free. If the workflow's quality plateaus and the traces show the fixed path is what limits it, you have the evidence and the harness you need to try an agent. If it does not plateau, you are done, and you saved a great deal of money.

## 1.8 The map of this book

The chapters that follow are organised as the layers of a production agent, from the inside out.

{{FIG:ch1_book_map|The map of the book. The loop, tools, memory and context sit at the centre (Chapters 1 and 2); the reasoning, workflow and multi-agent patterns are arranged around them (3 to 5); evals, tracing and guardrails form the outer ring (6 to 9), which is most of the book; the capstone (10) builds one agent through every layer.}}

At the centre is what an agent is made of: this chapter's loop, and Chapter 2's building blocks (model, tools, instructions, memory, state) with a close look at how to design a tool a model can actually use and how to manage what is in the context. Around that are the patterns: Chapter 3 on how a single agent reasons (ReAct, plan-and-execute, reflection, tree search, and what the papers measured), Chapter 4 on the workflow patterns that fix a path in code (chaining, routing, parallel fan-out, orchestrator-workers, evaluator-optimiser), and Chapter 5 on multi-agent systems and, with the MAST taxonomy in hand, the ways they fail.

The outer ring is the largest, because it is where production systems live or die: Chapters 6 and 7 on evaluation (golden sets, error analysis, LLM-as-a-judge and its calibration, trajectory evals, public benchmarks, and then evals in production: offline suites, online checks, regression gates in CI, and how the companies in this chapter did it), Chapter 8 on observability (traces and spans for agents, what to log, replaying a run, cost and latency dashboards) and Chapter 9 on guardrails and human oversight (input and output checks, tool permissions, sandboxing, prompt injection, approval steps, kill switches). Chapter 10 builds one small production-shaped agent through every layer, with the code in the repository, and ends with a cheatsheet and a glossary.

## Exercises

1. Take a system you have built or used that is described as "AI-powered". Draw its flowchart. Is every box and arrow known before a run starts? Place it on the spectrum of Section 1.3 and give the autonomy level of each action it takes.
2. For the on-call assistant of Section 1.1, list five distinct ways a single turn of the loop could fail (a wrong tool, bad arguments, misread observation, and so on). For each, say whether it is a model, tool, loop or memory problem, and what line in the trace would reveal it.
3. Your agent needs 12 steps and must succeed on 95% of tasks. What per-step reliability do you need? Which of the three design moves in Section 1.5 would you try first to get there, and why?
4. Re-price the cost table in Section 1.5 with your own provider's prices and a context that grows by 1,500 tokens per step rather than staying at 3,000. At what step count does the agent cost more than a human doing the task for ten minutes at your organisation's loaded hourly rate?
5. Pick one of the three company systems in Section 1.6 and write the eval set you would build for it: ten representative inputs, the definition of success for each, and who or what would judge it. Where would an LLM judge be acceptable and where would you insist on a human?

## Key takeaways

- A **model** computes one output from one prompt and is stateless. An **agent** is a loop around a model: observe, reason, act, observe again, until the model says it is done or a budget runs out. The agent is defined by the loop, not the model.
- The loop has four slots, context, model, tool call and environment, and every agent failure lives in one of them. Only the trace tells you which.
- **Workflows** fix the path in code and call the model inside steps; **agents** let the model choose the path at run time. The test is "who decides the next step", not "does it use tools". Most production AI features are, and should be, workflows.
- Both Anthropic and OpenAI say the same thing: find the simplest solution, and build an agent only where the path cannot be written down in advance (variable steps, judgement, unstructured data).
- The idea runs from ReAct (interleaved reasoning and acting, 2022) through Toolformer (learned tool use), Reflexion (verbal self-reflection in memory) and Generative Agents (a memory architecture for long-running behaviour) to the 2025 failure taxonomy, where most multi-agent failures were system design problems, not model problems.
- Autonomy is a ladder (suggest, act with approval, act and report, fully autonomous), and the right rung depends on measured reliability and reversibility. The three production systems in this chapter all run at level 1.
- Reliability compounds: a task of $$n$$ steps at per-step reliability $$p$$ succeeds with probability $$p^{n}$$. At $$p = 0.9$$ a ten-step task fails two times in three; ten steps above 90% needs $$p = 0.99$$ on every step.
- Cost compounds too: an eight-step agent costs about seven times a single call in the example and takes about eight times as long; Anthropic measured 4× tokens for agents and 15× for multi-agent systems in production. The task has to be worth it.
- The companies that run agents put their effort into evaluation: four-label feedback buttons, twenty real queries and an LLM judge to start, five hundred human-scored conversations a day. "80% was fast; the last 20% took most of the work."
- Do not build an agent when code can do it, when the budget does not fit the step count, when the accuracy requirement is above what $$p^{n}$$ allows, when you cannot evaluate it, or when the actions are irreversible with no approval step. Build the workflow first; it gives you the baseline and the eval set.

## References

**Papers and books**

1. Stuart Russell and Peter Norvig. [*Artificial Intelligence: A Modern Approach*](https://aima.cs.berkeley.edu/), 4th edition. Pearson 2020. (Chapter 2, "Intelligent Agents".)
2. Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, Yuan Cao. [*ReAct: Synergizing Reasoning and Acting in Language Models*](https://arxiv.org/abs/2210.03629). ICLR 2023 (arXiv October 2022).
3. Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom. [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761). NeurIPS 2023 (arXiv February 2023).
4. Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, Shunyu Yao. [*Reflexion: Language Agents with Verbal Reinforcement Learning*](https://arxiv.org/abs/2303.11366). NeurIPS 2023 (arXiv March 2023).
5. Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein. [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442). UIST 2023 (arXiv April 2023).
6. Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. [*Why Do Multi-Agent LLM Systems Fail?*](https://arxiv.org/abs/2503.13657). NeurIPS 2025 Datasets and Benchmarks track (arXiv March 2025).

**Engineering blogs and docs**

7. Anthropic. [*Building effective agents*](https://www.anthropic.com/research/building-effective-agents). Research blog, 19 December 2024.
8. OpenAI. [*A practical guide to building agents*](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf). PDF guide, April 2025.
9. Nicholas Marcott, Eduards Sidorovics, Paarth Chothani, Xiyuan Feng, Chun Zhu, Meghana Somasundara, Kailiang Fu, Jonathan Li. [*Genie: Uber's Gen AI On-Call Copilot*](https://www.uber.com/us/en/blog/genie-ubers-gen-ai-on-call-copilot/). Uber Engineering blog, 10 October 2024.
10. Anthropic. [*How we built our multi-agent research system*](https://www.anthropic.com/engineering/built-multi-agent-research-system). Engineering blog, 13 June 2025.
11. Juan Pablo Bottaro and Karthik R.. [*Musings on Building a Generative AI Product*](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product). LinkedIn Engineering blog, 25 April 2024.

**Code for this chapter**

12. [`code/agents/ch1_loop.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch1_loop.py): the fifty-line agent loop with a stubbed model and two fake tools; its output is `results/ch1_loop_stdout.txt`.
13. [`code/agents/ch1_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch1_math.py): the reliability table ($$p^{n}$$), the step budgets, and the token-cost comparison at an example price; results in `results/ch1_math.json`.
14. [`code/agents/figs_ch1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/figs_ch1.py) and [`code/agents/shots_ch1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/shots_ch1.py): the figures and the paper excerpts.

## Next

→ [Chapter 2: The building blocks](02-the-building-blocks.md)

Chapter 2 opens the agent up. It takes the four slots of the loop (model, tools, instructions, memory and state) one at a time, shows how to design a tool so that a model calls it correctly (the "which, when, what arguments, what to do with the result" of Toolformer), and introduces context engineering: deciding what goes into the model's input on each turn, what is summarised, what is stored outside and retrieved, and what is thrown away. It is the chapter that turns the stubbed `model()` of this chapter into something real.
