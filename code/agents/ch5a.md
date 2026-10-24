---
description: "When several AI agents beat one and when they do not, six multi-agent architectures, how agents hand work over, and how they fail."
---
# Chapter 5 · Multi-agent systems: when several agents beat one, how they coordinate, and how they fail

> **Goal:** by the end of this chapter you can say when a task is worth splitting across several agents and when one agent will do better, draw the common multi-agent architectures and choose one for a task, and write a handoff that another agent can act on. You will also know how agents can check each other through debate and voting, the ways multi-agent systems fail, and how to test and run them in production. One research question carries you through, and real experiments measure what splitting the work buys and what it costs.

---

## 5.1 When several agents beat one, and when they do not

### The running example: Northwind's supplier question

Northwind Home, the fictional UK shop of Chapter 4, sells furniture and kitchenware. Its buying team now asks a research question, not a support question: "Which three new kitchenware suppliers should we add for 2027? Compare price, delivery time to the UK, and risks."

This question is hard for one agent in three ways. It is **broad**: there are 24 candidate suppliers, and each one must be checked. It is **long**: every supplier has a catalogue page of about 500 words, and there is a news feed and a shipping table too. It needs **different tools**: the price is on a supplier page, the delivery time needs a shipping lookup, and the risks are in the news.

To make the experiments repeatable and cheap, we built a small fake world for this question. The script `ch5a_data.py` writes it to `data/ch5/a/`: 24 supplier pages, 16 news items, a table of 19 ports, and Northwind's buying rules. The rules say how to convert each currency at fixed planning rates, how to add freight and a 4% import duty to get a **landed cost** per reference basket, and when to reject a supplier: delivery over 60 days, a minimum order over 1,000 units, an open safety recall or a suspended ethical audit. Then the three cheapest suppliers that pass win.

> [!DEFINITION] Landed cost
> The full cost of a product once it is in your warehouse: the price, converted to your currency, plus freight and import duty. Northwind ranks suppliers on the landed cost of one reference basket (a frying pan, a chef's knife and a set of mixing bowls).

The catalogue hides traps a careless researcher falls into. The cheapest supplier (S03) has an open recall that appears only in the news. The second cheapest (S11) needs a minimum order of 2,000 units. The third cheapest (S07) takes 75 days to deliver. One news item recalls kettles from "Lumen Housewares" of Ohio, which is not "Lumen Homeware" of Porto (S08). Because the rules are exact, the answer is known: the same script computes it in code.

[![Terminal output of ch5a_data.py: 24 supplier pages of about 505 words each, 16 news items and 19 ports; then the known answer as a table sorted by landed cost: S03 at 32.99 pounds rejected for an open safety recall, S11 rejected for a minimum order of 2,000, S07 rejected for 75 days of delivery, then S05 at 39.25, S09 at 39.72 and S12 at 39.90 accepted; the gold picks are S05, S09 and S12](/img/agents/ch5a_data-run.png)](/img/agents/ch5a_data-run.png)

The three winners, S05 (£39.25), S09 (£39.72) and S12 (£39.90), sit within 65 pence of each other. The next eligible supplier, S13, costs £40.53. A researcher who skips a supplier, uses a market exchange rate or forgets the duty will choose a different three. That is what makes this a good test: small errors in breadth or in rule-following show up in the answer.

### What a multi-agent system is

Chapter 1 defined an agent as a loop in which a model decides the next step, and a multi-agent system as two or more agents that exchange messages to complete one task. Four more words are needed before we can compare designs.

> [!DEFINITION] Agent (in this chapter)
> One model running its own loop, with its own tools and its own **context**: the messages, tool results and instructions that the model reads at each step. Two agents can use the same model; what makes them two agents is two separate loops and two separate contexts.

> [!DEFINITION] Orchestrator (lead agent)
> The agent that owns the whole task. It splits the work, starts other agents, gives each one its part, collects what they return and writes the final answer. Anthropic calls it the lead agent; frameworks also call it a supervisor or manager.

> [!DEFINITION] Subagent
> An agent started by another agent to do one part of the task. It runs its own loop with a fresh context that holds only what it was given, and it returns a result to the agent that started it. It does not see the lead's conversation unless the lead passes it on.

> [!DEFINITION] Handoff
> The moment work, and the information needed to do it, passes from one agent to another. It can carry the full conversation so far, a short written task description (a **brief**), or a structured result. Whatever the handoff does not carry, the receiving agent does not know.

A newsroom is a good picture. The editor (the lead) reads the story request, decides which reporters to send, and gives each one an assignment. Each reporter (a subagent) goes out with a notebook (a fresh context) and comes back with a short report. The editor never reads the reporters' notebooks, only their reports. This works well when the assignments are clear and the stories are independent. It works badly when two reporters must agree on facts that neither one has.

Chapter 4 drew the same shape as the orchestrator-workers **workflow**, where each worker was one model call. Here each worker is a full agent with its own tool loop, so it can search, read and decide for itself. That makes it more capable, more expensive and harder to predict.

### Three reasons several agents can help

**Context isolation.** A model reads its whole context at every step. As the context grows, each step costs more, and long contexts make it harder for the model to use what is in them (Chapter 2). If four subagents each read six supplier pages, no single context holds all 24 pages. The lead receives four short reports instead. Anthropic describes this as compression.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025) · Benefits of a multi-agent system
> [![A paragraph from the post: the essence of search is compression, distilling insights from a vast corpus; subagents facilitate compression by operating in parallel with their own context windows, exploring different aspects of the question simultaneously before condensing the most important tokens for the lead research agent; each subagent also provides separation of concerns, distinct tools, prompts and exploration trajectories, which reduces path dependency and enables thorough, independent investigations](/img/agents/ch5a-anthropic-compression.png)](/img/agents/ch5a-anthropic-compression.png)
>
> **What it says:** subagents "facilitate compression by operating in parallel with their own context windows", and each one "provides separation of concerns: distinct tools, prompts, and exploration trajectories".
>
> **What it means:**
>
> Think of a library search. One person who reads every book on a topic ends up with a desk covered in open books. Four people who each read a shelf, and each hand back one page of notes, leave the person who writes the summary with four pages. The reading still happened, but the writer's desk stays clear.

**Parallel breadth.** Independent parts can run at the same time. If checking one supplier takes about ten seconds, checking 24 one after another takes four minutes; four subagents with six suppliers each can finish in about one minute. The total work is the same. Only the waiting is shorter.

**Specialised tools and prompts.** Each agent can have its own instructions, its own tools and even its own model. A pricing agent needs the catalogue and the exchange rates; a risk agent needs the news feed. Fewer tools per agent means fewer wrong tool choices (Chapter 2). AutoGen's authors tested this directly, and Section 5.2 reads their result: splitting code writing and safety checking into two agents caught more unsafe code than one agent doing both.

### What it always costs

Every agent pays for its own system prompt, its own brief, its own reading and its own report. The lead pays to plan, to write briefs and to read reports. Chapter 1 and Chapter 4 quoted Anthropic's own measurement: multi-agent systems "use about 15× more tokens than chats". The same post explains why they work anyway.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025) · Benefits of a multi-agent system
> [![A paragraph from the post: multi-agent systems work mainly because they help spend enough tokens to solve the problem; in our analysis, three factors explained 95 percent of the performance variance in the BrowseComp evaluation; token usage by itself explains 80 percent of the variance, with the number of tool calls and the model choice as the two other explanatory factors; multi-agent architectures effectively scale token usage for tasks that exceed the limits of single agents](/img/agents/ch5a-anthropic-tokens.png)](/img/agents/ch5a-anthropic-tokens.png)
>
> **What it says (reported by Anthropic):** "Multi-agent systems work mainly because they help spend enough tokens to solve the problem." On the BrowseComp benchmark, "token usage by itself explains 80% of the variance" in performance.
>
> **What it means:**
>
> This is an honest and slightly surprising claim. The benefit is not that agents talk to each other cleverly. It is that a team can read more than one person, because each member has a fresh context.
>
> The consequence is that a multi-agent system is a way to spend more, not a way to save. If one agent can already read everything it needs, splitting the work adds cost and may add nothing else. Our experiment below tests exactly this case.

Latency has a cost too. The lead must plan before anyone starts, wait for the slowest subagent, and then merge. The third cost is **coordination**: whatever the lead forgets to write in a brief is lost, and two agents working in parallel can make choices that do not fit together.

{{FIG:a_why|What several agents can buy (left) and what they always cost (right). A multi-agent design pays off only when the left column is large for the task.}}

### The evidence, and the disagreement

Two engineering posts appeared one day apart in June 2025 and reached opposite conclusions. On 12 June, Walden Yan of Cognition, the company behind the coding agent Devin, published "Don't Build Multi-Agents". On 13 June, Anthropic published how it built the multi-agent system behind its Research feature. Chapter 4 read Anthropic's headline numbers; here we read the argument against.

> [!PAPER] Cognition, "Don't Build Multi-Agents" (Walden Yan, blog, June 2025)
> [![Two paragraphs from the post: suppose your task is build a Flappy Bird clone; this gets divided into subtask 1, build a moving game background with green pipes and hit boxes, and subtask 2, build a bird that you can move up and down; it turns out subagent 1 actually mistook your subtask and started building a background that looks like Super Mario Bros; subagent 2 built you a bird, but it does not look like a game asset and it moves nothing like the one in Flappy Bird; now the final agent is left with the undesirable task of combining these two miscommunications](/img/agents/ch5a-cognition-flappy.png)](/img/agents/ch5a-cognition-flappy.png)
>
> **What it says:** a task split into two subtasks fails when "subagent 1 actually mistook your subtask" and subagent 2's bird "moves nothing like the one in Flappy Bird". The final agent must combine "these two miscommunications".
>
> **What it means:**
>
> The example is about a handoff. "Build a bird that you can move up and down" is a true description of the subtask, but it leaves out everything the user meant by "Flappy Bird": the art style, the physics, how the bird and the pipes must fit together. The original agent knew this from the conversation. The subagents did not.

The post states two principles from this example.

> [!PAPER] Cognition, "Don't Build Multi-Agents" (Walden Yan, blog, June 2025) · Principle 1
> [![Principle 1 from the post: share context, and share full agent traces, not just individual messages](/img/agents/ch5a-cognition-principle1.png)](/img/agents/ch5a-cognition-principle1.png)
>
> **What it says:** "Share context, and share full agent traces, not just individual messages."
>
> **What it means:**
>
> A **trace** is the full record of what an agent saw and did: every message, tool call and tool result. The principle says a receiving agent should get that whole record, not a summary written by the sender. A summary is the sender's guess about what matters, and the guess can be wrong. The post adds that copying only the original task is not enough, because "any number of details could have consequences on the interpretation of the task".
>
> This is the opposite of Anthropic's compression. Cognition wants the receiver to see everything; Anthropic wants the lead to see only the condensed result. Section 5.3 tests both with the same subagent.

> [!PAPER] Cognition, "Don't Build Multi-Agents" (Walden Yan, blog, June 2025) · Principle 2
> [![Principle 2 from the post: actions carry implicit decisions, and conflicting decisions carry bad results](/img/agents/ch5a-cognition-principle2.png)](/img/agents/ch5a-cognition-principle2.png)
>
> **What it says:** "Actions carry implicit decisions, and conflicting decisions carry bad results."
>
> **What it means:**
>
> When a subagent draws the background, it also decides the art style, the colours and the scale, without anyone asking it to. A second subagent drawing the bird decides the same things differently. Each choice is reasonable; together they clash. Even with full context, parallel agents that each **act** can conflict, because neither sees the other's choices as they are made.
>
> The post's recommendation follows: by default, use one "single-threaded linear agent", and when its context overflows, compress its history with a model trained for that, rather than splitting the work.

Read side by side, the two posts disagree less than their titles suggest. Anthropic's subagents mostly **read**: they search and return findings, and the lead decides. Anthropic's own post also says that tasks where agents "share the same context or involve many dependencies between agents are not a good fit", and that most coding tasks have fewer truly parallel parts than research. Cognition's subagents **write**: each one produces part of a program, and every part carries decisions that must fit together. A useful rule comes out of this: parallel agents are safest when they gather information and one agent makes the decisions. Our supplier question is mostly reading, which is Anthropic's side of the line.

### The cost arithmetic

Why can splitting the work save tokens in one case and waste them in another? The answer is in how an agent pays for its context.

**The intuition.** An agent sends its whole context to the model at every step. Think of a student who rereads all their notes before writing each new line. With one page of notes this is cheap. With 24 pages it is slow, and the cost grows faster than the notes do, because every new page is reread at every later step.

**The equations.** Suppose one agent must fetch $$n$$ tool results of $$d$$ tokens each, one result per model call, and each call also adds $$o$$ output tokens to the history. Its base prompt is $$P$$ tokens. The input tokens over $$n$$ fetch calls and one final answer call are

$$T_1 = (n+1)\,P + (d+o)\,\frac{n(n+1)}{2}.$$

Now split the same work over $$m$$ subagents. Each one fetches $$k = n/m$$ results, with its own prompt plus a brief of $$B$$ tokens. The lead adds $$L$$ tokens of its own: its calls, its briefs and the reports it reads. Then

$$T_m = m\left[(k+1)(P+B) + (d+o)\,\frac{k(k+1)}{2}\right] + L.$$

**Reading them.** In $$T_1$$, the first term is the prompt paid once per call. The second term is the rereading: the result fetched at step $$i$$ is read again at every later step, which adds up to a triangle of size $$n(n+1)/2$$. In $$T_m$$, each subagent has its own small triangle of size $$k(k+1)/2$$, and there are $$m$$ of them. Since $$m \cdot k(k+1)/2$$ is about $$1/m$$ of $$n(n+1)/2$$, splitting cuts the rereading term by about $$m$$. In exchange, every subagent pays $$P+B$$ at each of its calls, and the lead pays $$L$$.

**A tiny example.** Take $$n = 4$$ pages, $$d + o = 1{,}000$$ tokens and $$P = 1{,}000$$. One agent pays $$5 \times 1{,}000 + 1{,}000 \times 10 = 15{,}000$$ input tokens. Two subagents with two pages each, a brief of $$B = 500$$ and $$L = 3{,}000$$ pay $$2 \times [3 \times 1{,}500 + 1{,}000 \times 3] + 3{,}000 = 18{,}000$$. With four pages, splitting costs more. Repeat with $$n = 24$$ and the rereading term dominates, so splitting wins.

**The real numbers.** `ch5a_cost.py` runs this arithmetic with our catalogue's page size, measured from the agent traces of the experiment below: ⟦PAGE⟧ tokens per page. It also covers the escape route that modern APIs offer a single agent: asking for several tool calls in one turn, which cuts the number of steps and so the rereading.

[![Terminal output of ch5a_cost.py: ⟦COSTALT⟧](/img/agents/ch5a_cost-run.png)](/img/agents/ch5a_cost-run.png)

{{FIG:a_cost|Input tokens and time per task under the assumptions printed by ch5a_cost.py. One agent that reads one page per turn pays most, because it rereads a growing context. Splitting the reading over subagents cuts that, and so does letting one agent request several tool calls per turn.}}

**Assumptions.** Tools take no time, every call takes the same time, nothing is cached and nobody compacts the history. Prompt caching (Chapter 2) lowers the price of the rereading term a lot, and compaction removes some of it. The arithmetic shows the shape: splitting wins when one context would have to reread a lot, and loses when the work is small or one agent can batch its reading.

### Our experiment: one agent against a lead with subagents

`ch5a_single_vs_multi.py` asks the supplier question with a real model (gpt-5-mini, reasoning effort low) in three arrangements:

- **one agent**: one loop with all five tools (read the rules, list suppliers, read a supplier page, search the news, look up a port);
- **one agent, one tool call per turn**: the same, but the API is told to allow only one tool call per model turn, the way many older agents worked;
- **lead + subagents**: a lead that can only read the rules and the supplier list, and a `run_subagents` tool that starts up to six subagents in parallel. Each subagent has the three research tools and sees only its brief.

All three get the same task text and must answer in the same JSON shape. Code grades each answer against the known answer: how many of the three gold suppliers it picked, whether any pick breaks a rule, and whether the landed cost and delivery days it reports are right. From the traces we also count how many of the 24 supplier pages the system actually read.

{{FIG:a_setup|The two designs of the experiment. One agent holds every tool and one context. The lead holds the rules and the supplier list, and everything it learns about a supplier comes back in a subagent's report.}}

```python
# code/agents/ch5a_agents.py (abridged): the lead's one special tool. Subagents run on threads; the lead waits for all of them.
def run_subagents(tasks):
    tasks = tasks[:max_subagents]                                    # a cap in code, not only in the prompt
    with ThreadPoolExecutor(len(tasks)) as ex:
        res = list(ex.map(lambda nt: run_subagent(nt[0], nt[1]['brief'], meter, trace), zip(names, tasks)))
    return '\n\n'.join(f'=== report from {n} ({s:.0f} s) ===\n{out}' for n, (out, s) in zip(names, res))

def run_subagent(name, brief, meter, trace=None, max_calls=25, history=None):
    msgs = [{'role': 'system', 'content': SUB_SYSTEM}] + (history or []) + [{'role': 'user', 'content': brief}]
    return run_agent(name, msgs, SUB_TOOLS, SUB_HANDLERS, meter, max_calls=max_calls, trace=trace)   # the Chapter 1 loop
```

[![Terminal output of ch5a_single_vs_multi.py: ⟦E1ALT⟧](/img/agents/ch5a_single_vs_multi-run.png)](/img/agents/ch5a_single_vs_multi-run.png)

{{FIG:a_exp1|Results of the final run of ch5a_single_vs_multi.py, means over ⟦N1⟧ runs per arm. ⟦E1CAP⟧}}

⟦E1TEXT⟧

> [!WARNING]
> What this experiment does and does not show. ⟦N1⟧ runs per arm on one question is a demonstration, not a benchmark; on another run each count can move. The catalogue fits comfortably in one context of this model, which is the case where Cognition's advice should win on cost. It does not show what happens when the material is many times larger than a context window, which is where Anthropic's gains were measured. The finding to keep is the mechanism: what each design reads, what it pays for the reading, and where its errors come from.

### Discussion

1. **Is the extra cost of a lead and subagents worth it here?** ⟦DISC1⟧
2. **Would a better prompt fix the single agent?** Possibly; "check every supplier" was already in its instructions. A plan written as a checklist in code, with one step per supplier, is a cheaper fix than a team of agents (Chapter 4). The lesson is that coverage must be checked, not assumed.
3. **When would the lead win clearly?** When there are hundreds of suppliers, when each page is long, or when the subagents need tools the lead should not have. Then no single context can hold the reading, and the time saved by parallel work grows with the number of subagents.

## 5.2 Architectures: six ways to arrange agents

A multi-agent architecture answers three questions. Who decides what each agent does? What can each agent see? How do results travel back? Six arrangements cover almost every system in use. We take them from the simplest up, and for each one say what problem it solves.

### Single agent with tools (the baseline)

{{FIG:a_arch_single|The baseline: one loop, one context, several tools. Nothing is lost between agents because there is only one.}}

Every comparison starts here. One agent holds the whole conversation, every tool result and every decision. It cannot lose information in a handoff and cannot disagree with itself in parallel. Its limits are the size of its context and the speed of one loop. OpenAI's practical guide (Chapter 4) gives the same default: get the most out of one agent first, and split only when it fails to follow its instructions or keeps choosing the wrong tools.

### Orchestrator with subagents

{{FIG:a_arch_orch|An orchestrator, also called a supervisor or lead agent, plans, briefs subagents, runs them in parallel with their own contexts, and merges their results. It can run more rounds.}}

This is the shape of our experiment and of Anthropic's research system. The lead owns the answer, and subagents never talk to each other. It solves the breadth problem: many independent parts, each too big to keep in the lead's context. Its weak point is the brief, because a subagent knows only what the lead wrote. OpenAI's Agents SDK names this pattern "agents as tools": the manager calls a specialist like any other tool and keeps control.

> [!PAPER] OpenAI, "Agent orchestration" (Agents SDK documentation, accessed October 2026)
> [![A table from the documentation: Agents as tools, a manager agent keeps control of the conversation and calls specialist agents through Agent.as_tool(), best when you want one agent to own the final answer, combine outputs from multiple specialists, or enforce shared guardrails in one place; Handoffs, a triage agent routes the conversation to a specialist, and that specialist becomes the active agent for the rest of the turn, best when you want the specialist to respond directly, keep prompts focused, or have the handoff switch the active instructions](/img/agents/ch5a-sdk-patterns.png)](/img/agents/ch5a-sdk-patterns.png)
>
> **What it says:** in "agents as tools", "a manager agent keeps control of the conversation and calls specialist agents"; with "handoffs", "a triage agent routes the conversation to a specialist, and that specialist becomes the active agent".
>
> **What it means:**
>
> The two rows differ in one thing: who speaks last. With agents as tools, the specialist's answer goes back to the manager, who decides what to do with it. With a handoff, the specialist takes over and answers the user directly.

### Hierarchical teams

{{FIG:a_arch_hier|Hierarchical teams: a top lead delegates to team leads, and each team lead runs its own workers. Each level summarises for the level above.}}

A hierarchy is an orchestrator applied twice. The top lead talks only to team leads, and each team lead runs its own subagents. It solves the problem of a lead with too many reports to read: forty subagents would flood one lead's context, but four team leads with ten subagents each would not. Every level adds a handoff where detail can be lost, and a level of calls to pay for. Use it only when one lead cannot hold the reports of all the workers.

### Peer-to-peer handoffs

{{FIG:a_arch_handoff|Peer-to-peer handoffs. A triage agent transfers the conversation to a specialist through a handoff tool, and the specialist answers the user. No agent sits above the others.}}

In this shape there is no lead. An agent that decides another agent is better placed passes the whole conversation on, and the new agent becomes the active one. OpenAI's Swarm, a library its README calls "experimental, educational", built everything from two ideas, "Agents and handoffs": an agent "encompasses instructions and tools, and can at any point choose to hand off a conversation to another Agent". Its README now points to the Agents SDK as its "production-ready evolution".

> [!PAPER] OpenAI, "Handoffs" (Agents SDK documentation, accessed October 2026)
> [![A sentence from the documentation: handoffs are represented as tools to the LLM; so if there is a handoff to an agent named Refund Agent, the tool would be named transfer_to_refund_agent](/img/agents/ch5a-sdk-handoff-tools.png)](/img/agents/ch5a-sdk-handoff-tools.png)
>
> **What it says:** "Handoffs are represented as tools to the LLM", so a handoff to a Refund Agent is a tool named `transfer_to_refund_agent`.
>
> **What it means:**
>
> A handoff is not a new mechanism. It is a tool call whose effect is to change which agent runs next. The model chooses it the same way it chooses any tool: from the tool's name and description. So a wrong handoff is a tool-selection error (Chapter 2), and you can test it like one, with labelled cases.

Handoffs suit conversations that move between domains: a customer starts with an order question and ends with a refund. They suit research badly, because nobody gathers the results of several specialists into one answer.

### Shared state: a blackboard

{{FIG:a_arch_board|A shared store that every agent reads and writes. Agents coordinate through the state, not through messages to each other.}}

Instead of sending messages, agents can write to one shared place and read from it. The idea is older than language models. Speech-understanding systems of the 1970s, such as Hearsay-II, used a "blackboard": independent specialist programs posted partial results to a shared structure, and each one acted when it saw something it could use. The analogy is a team of detectives working around a board of pinned evidence.

In a modern system the blackboard can be a database table, a file or a typed state object. A price agent writes landed costs, a risk agent writes recall flags, and a writer agent waits until both columns are full. MetaGPT (Section 5.2's research below) uses a form of it called a shared message pool. The danger is that two agents write the same field. A blackboard therefore needs ownership rules: which agent may write which field, and what happens on a conflict.

### Role-play teams

{{FIG:a_arch_roles|Role-play teams in the style of ChatDev: pairs of agents with job titles chat in turns, and each pair passes its result to the next.}}

The last shape gives each agent a job title and a persona, and lets them talk like colleagues: a CEO and a CTO agree on a design, a programmer writes code, a reviewer comments, a tester tests. It was the most visible multi-agent idea of 2023. The roles are prompts on the same model, not different skills, so the gain, when there is one, comes from the structure the roles impose: a fixed sequence of steps and a required document at each step.

### The research behind them

The four papers below appeared on arXiv between March and August 2023, a few months after chat models became widely available. Each one asked whether several chat agents could do together what one could not.

**CAMEL** (Li et al., King Abdullah University of Science and Technology, March 2023) started with two agents. A human gives an idea, a "task specifier" agent makes it concrete, and then an AI user and an AI assistant chat until the task is done.

> [!PAPER] Li et al. (2023), CAMEL · Figure 1 · page 4
> [![Figure 1 of the CAMEL paper: a human user gives the idea develop a trading bot for the stock market and assigns roles, an AI assistant who is a Python programmer and an AI user who is a stock trader; a task specifier turns the idea into a specific task; in the role-playing session the AI user gives instructions such as install the necessary Python libraries, and the AI assistant replies with solutions and code](/img/agents/ch5a-camel-figure1.png)](/img/agents/ch5a-camel-figure1.png)
>
> **What it says:** the human gives an idea and assigns roles; "the task is made more specific using our task specifier agent"; then the AI user and AI assistant "collaboratively communicate by chatting with each other in an instruction-following fashion".
>
> **What it means:**
>
> The design separates two jobs that one chat mixes: deciding what to do next (the AI user, playing the stock trader) and doing it (the AI assistant, playing the programmer). The user agent gives one instruction at a time; the assistant answers each one.

> [!PAPER] Li et al. (2023), CAMEL · Section 4.1 · page 8
> [![A list from the CAMEL paper of four challenges: role flipping, where the assistant and user switch roles during the conversation; the assistant repeating the user's instructions; flake replies, where the assistant says I will and does not follow through; and an infinite loop of meaningless conversation, such as repeatedly thanking each other, where the agents are sometimes aware they are stuck but are unable to break out of it](/img/agents/ch5a-camel-challenges.png)](/img/agents/ch5a-camel-challenges.png)
>
> **What it says:** the authors saw "role flipping", where "the assistant and user switch roles", replies that promise action and do not deliver, and "an infinite loop of meaningless conversation", sometimes with agents "aware that they are stuck in a loop, but are unable to break out of it".
>
> **What it means:**
>
> These are coordination failures, and they appear with only two agents. Role flipping is the subtle one: the model playing the assistant drifts into giving orders, because nothing outside the prompt holds it to its role. Our handoff experiment in Section 5.3 found a modern version of the same failure.
>
> The infinite loop is a missing stop rule. Two polite agents can thank each other forever, because neither one owns the decision to end. A coordinator with a budget, or a code check that ends the run, removes this failure entirely.

**MetaGPT** (Hong et al., August 2023, ICLR 2024) answered CAMEL's chatter with structure. It models a software company: a product manager writes requirements, an architect writes a design, engineers write code, and each role must produce a document in a fixed format.

> [!PAPER] Hong et al. (2023), MetaGPT · Section 1 · page 2
> [![A passage from the MetaGPT paper: in a company simulated by MetaGPT, all employees follow a strict and streamlined workflow, and all their handovers must comply with certain established standards; this reduces the risk of hallucinations caused by idle chatter between LLMs, particularly in role-playing frameworks, like Hi, hello and how are you, from Alice the product manager, and Great, have you had lunch, from Bob the architect](/img/agents/ch5a-metagpt-chatter.png)](/img/agents/ch5a-metagpt-chatter.png)
>
> **What it says:** "all their handovers must comply with certain established standards", which "reduces the risk of hallucinations caused by idle chatter between LLMs".
>
> **What it means:**
>
> The authors noticed that free conversation between agents produces conversation, not work. Their fix is a standard document at every handoff: a requirements document, a design, an interface specification. A document with required sections cannot be "Great! Have you had lunch?".
>
> This is the same move as a schema between chain steps in Chapter 4, applied to agents. MetaGPT reports 85.9% and 87.7% pass@1 on the HumanEval and MBPP coding benchmarks with GPT-4, against 67.0% for GPT-4 alone on HumanEval in its own figure. Those numbers come from the authors' setup and are not ours.

**ChatDev** (Qian et al., Tsinghua University and others, July 2023) also simulates a software company, but as a "chat chain": a fixed sequence of phases, each one a two-agent conversation in CAMEL's style.

> [!PAPER] Qian et al. (2023), ChatDev · Figure 2 · page 3
> [![Figure 2 of the ChatDev paper: three phases, design, coding and testing, split into subtasks, design, coding, code complete, code review and testing; for each subtask an instructor, CEO, CTO, reviewer or tester, chats with an assistant, CTO or programmer; the chat chain passes task to ideas, then to code, and the code on through each subtask](/img/agents/ch5a-chatdev-figure2.png)](/img/agents/ch5a-chatdev-figure2.png)
>
> **What it says:** "software agents engage in multi-turn communication and perform instruction-following along a chain-structured workflow".
>
> **What it means:**
>
> Look at the middle row of the figure: only `{task}`, `{ideas}` and `{code}` travel along the chain. The conversation inside each pair stays inside that pair. That is a handoff design decision, and the paper states it directly in a passage Section 5.3 reads.

**AutoGen** (Wu et al., Microsoft Research and others, August 2023) turned these ideas into a general framework. Its unit is the "conversable agent", which can send and receive messages, and an application is a pattern of conversations between such agents. The paper's Figure 3 draws six such patterns, and they include this section's shapes in miniature: a commander directing a writer and a safeguard (an orchestrator), and a manager that picks the next speaker and broadcasts to all (close to a shared board). Chapter 4 read its group-chat pilot. Here we read its one direct comparison with a single agent.

> [!PAPER] Wu et al. (2023), AutoGen · Section 3, A4 · page 8
> [![A passage from the AutoGen paper: we also conduct an ablation showing that multi-agent abstraction is necessary; we construct a single-agent approach where a single agent conducts both the code-writing and safeguard processes; we tested the single- and multi-agent approaches on a dataset of 100 coding tasks with equal numbers of safe and unsafe tasks; the multi-agent design boosts the F-1 score in identifying unsafe code by 8 percent with GPT-4 and 35 percent with GPT-3.5-turbo](/img/agents/ch5a-autogen-ablation.png)](/img/agents/ch5a-autogen-ablation.png)
>
> **What it says:** compared with "a single agent [that] conducts both the code-writing and safeguard processes", the multi-agent design "boosts the F-1 score in identifying unsafe code by 8% (with GPT-4) and 35% (with GPT-3.5-turbo)", on 100 coding tasks.
>
> **What it means:**
>
> This is the specialisation argument with a number. An agent asked to write code and also judge whether it is safe is grading its own work. A separate safeguard agent, with its own prompt and no stake in the code, catches more.
>
> Two cautions. The test set is 100 tasks built for this test, and the gain is much larger with the weaker model. That suggests the benefit of splitting roles shrinks as models get better at holding several instructions at once.

### When to use which

| Architecture | Use it when | Avoid it when | What breaks first |
|---|---|---|---|
| Single agent | the task fits in one context and one loop is fast enough | the reading is far larger than the context | coverage: it stops early or skips parts |
| Orchestrator + subagents | many independent parts, mostly reading, one place to decide | parts depend on each other, or subagents must act | briefs that leave out rules; a merge that drops findings |
| Hierarchical teams | too many workers for one lead to read their reports | a flat orchestrator still fits | detail lost at each level; cost of each level |
| Peer-to-peer handoffs | a conversation moves between domains, one specialist at a time | several results must be combined | handing off too early or to the wrong agent |
| Shared state | agents work on different fields of one record, at different speeds | ownership of fields is unclear | two agents overwrite the same field |
| Role-play teams | a fixed procedure with a required document at each step | the roles are only personas | chatter, role flipping, loops with no end |

### Discussion

1. **Are role-play teams real architectures?** Without required documents and fixed phases, my view is that they are one model talking to itself with extra cost. MetaGPT's and ChatDev's gains come from the procedure, which you can also build as a workflow.
2. **Orchestrator or handoffs?** Choose by who must speak last. If a single answer must be checked and assembled, keep a manager. If the user should talk to the right specialist, hand off.
3. **Is a blackboard better than messages?** It removes the question of who tells whom, and it leaves a record that is easy to inspect. It adds the question of who may write what. For agents that only read and report, a list of structured results is a simple blackboard already.

## 5.3 How agents coordinate

### What passes between agents

Every handoff carries one of three things, or a mix of them.

{{FIG:a_passes|Three things one agent can hand another: the full history, a task brief or a structured result. Each keeps something and loses something.}}

> [!DEFINITION] Task brief
> A written description of a piece of work, made for the agent that will do it: the objective, the rules and facts it needs, the output format, the tools to use, and the limits. It is the subagent's whole picture of the task.

> [!DEFINITION] Structured result
> A result returned in fields the receiver asked for, such as a JSON object with one entry per supplier. It can be checked and merged in code. It keeps the values and usually drops the reasoning behind them.

**The full history** keeps everything, which is Cognition's first principle. It is large, it carries the sender's role and instructions along with the facts, and the receiver must work out which parts apply to it. The OpenAI Agents SDK makes it the default for a handoff.

> [!PAPER] OpenAI, "Handoffs" (Agents SDK documentation, accessed October 2026) · Input filters
> [![A paragraph from the documentation: when a handoff occurs, it is as though the new agent takes over the conversation, and gets to see the entire previous conversation history; if you want to change this, you can set an input_filter, a function that receives the existing input and must return a new one](/img/agents/ch5a-sdk-handoff-history.png)](/img/agents/ch5a-sdk-handoff-history.png)
>
> **What it says:** "the new agent takes over the conversation, and gets to see the entire previous conversation history", unless you set an `input_filter`.
>
> **What it means:**
>
> For a handoff, this default is right: the specialist becomes the agent talking to the user, so it needs the conversation. The input filter exists because the whole history is not always wanted, for example when it holds tool results the next agent should not see.

**A task brief** is small and aimed at one job. It loses whatever its writer forgot to include, and its writer is a model that may not know which details matter. **A structured result** flows the other way, from subagent to lead. ChatDev chose it on purpose for the step between phases.

> [!PAPER] Qian et al. (2023), ChatDev · Section 3.1 · page 4
> [![A paragraph from the ChatDev paper: by sharing only the solutions of each subtask rather than the entire communication history, ChatDev minimizes the risk of being overwhelmed by too much information, enhancing concentration on each task and encouraging more targeted cooperation, while simultaneously facilitating cross-phase context continuity](/img/agents/ch5a-chatdev-memory.png)](/img/agents/ch5a-chatdev-memory.png)
>
> **What it says:** "sharing only the solutions of each subtask rather than the entire communication history" lowers "the risk of being overwhelmed by too much information".
>
> **What it means:**
>
> Between the coding phase and the review phase, only the code travels. The reviewer does not read the whole argument about how the code came to be. It reviews the code as it is, the way a human reviewer reads a pull request rather than the author's chat log.
>
> It is the opposite choice to Cognition's, made for a different reason. Cognition worries about missing decisions; ChatDev worries about too much noise. Both are right about their own failure, and the experiment below measures both on our task.

### Shared state versus messages

Messages go from one agent to another. Shared state sits in one place that several agents read and write. MetaGPT explains why it chose a pool, and why a pool alone is not enough.

> [!PAPER] Hong et al. (2023), MetaGPT · Section 3.2 · page 6
> [![The publish-subscribe passage of the MetaGPT paper: communicating information one-to-one can complicate the communication topology; a viable approach is to store information in a global message pool that all agents can publish to and read from; but sharing all information with every agent can lead to information overload, so agents use role-specific interests and select information to follow based on their role profiles; an agent acts only after receiving all its prerequisite dependencies](/img/agents/ch5a-metagpt-pool.png)](/img/agents/ch5a-metagpt-pool.png)
>
> **What it says:** one-to-one messages "complicate the communication topology", so agents "store information in a global message pool". But "sharing all information with every agent can lead to information overload", so agents "select information to follow based on their role profiles".
>
> **What it means:**
>
> With five agents that message each other, there are twenty possible one-way channels. With one pool, there are five connections. That is the gain of shared state.
>
> The subscription is the full-history problem solved at the reader's side. Everything is stored, but each role reads only the kinds of message it needs. "An agent activates its action only after receiving all its prerequisite dependencies" is also a coordination rule: the architect waits for the requirements document, and so cannot start on a guess.

Anthropic made a related choice for results that are large. Instead of passing a long report back through the lead, a subagent stores it and returns a pointer.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025) · Appendix
> [![A paragraph from the post: subagent output to a filesystem to minimize the game of telephone; direct subagent outputs can bypass the main coordinator for certain types of results, improving both fidelity and performance; rather than requiring subagents to communicate everything through the lead agent, implement artifact systems where specialized agents can create outputs that persist independently; subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator](/img/agents/ch5a-anthropic-telephone.png)](/img/agents/ch5a-anthropic-telephone.png)
>
> **What it says:** "Subagents call tools to store their work in external systems, then pass lightweight references back to the coordinator", which "prevents information loss during multi-stage processing".
>
> **What it means:**
>
> In the children's game of telephone, a message whispered along a line arrives changed. Each agent that rewrites a result in its own words is one more whisper. Storing the result once and passing its address means that the final writer reads the original.

### How a lead writes a good subagent brief

Anthropic's lesson, read in Chapter 4, was that each subagent needs "an objective, an output format, guidance on the tools and sources to use, and clear task boundaries". Our experiments add one more part. A lead often learns rules before it delegates: in our case, the exchange rates, the duty and the limits. A subagent cannot apply a rule it was never given.

{{FIG:a_brief|The five parts of a subagent brief. The second part, the rules and facts the lead already knows, is the one most easily left out.}}

Here is the full brief used in the experiment below, abridged. Each part answers a question the subagent would otherwise have to guess.

```text
Objective: check six candidate kitchenware suppliers for Northwind Home (a UK shop) and report the facts the lead needs
to choose three suppliers for 2027. Your suppliers: S03, S07, S08, S09, S12, S19. Do not research any other supplier.
Rules to apply (from Northwind's buying rules; apply them exactly):
2. Convert to GBP with these fixed planning rates, not market rates: 1 EUR = 0.85, 1 USD = 0.78, 1 CNY = 0.11, ...
3. Landed cost per basket: if the price is FOB, add the freight per basket for the supplier's port (shipping_lookup), ...
6. Search the news for each supplier by name. Reject for an OPEN safety recall or a suspended ethical audit. ...
   Check that a news item is about the same company: similar names can be different companies.
Tools: read_supplier for each page, shipping_lookup for each port, search_news with each supplier's name. At most 25 tool calls.
Output: JSON only, one object per supplier: {"suppliers": [{"id": "S..", ..., "landed_cost_gbp": 0.00, "delivery_days": 0,
"moq": 0, "verdict": "accept or reject", "reason": "...", "risks": "..."}]}
```

The effort a lead puts into splitting the work also needs rules. Anthropic found that its lead could not judge how many subagents a question deserved, so it wrote the rule into the prompt.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025) · Prompt engineering and evaluations
> [![A list item from the post: scale effort to query complexity; agents struggle to judge appropriate effort for different tasks, so we embedded scaling rules in the prompts; simple fact-finding requires just 1 agent with 3 to 10 tool calls, direct comparisons might need 2 to 4 subagents with 10 to 15 calls each, and complex research might use more than 10 subagents with clearly divided responsibilities; these explicit guidelines help prevent overinvestment in simple queries, a common failure mode in early versions](/img/agents/ch5a-anthropic-effort.png)](/img/agents/ch5a-anthropic-effort.png)
>
> **What it says:** "Simple fact-finding requires just 1 agent with 3-10 tool calls, direct comparisons might need 2-4 subagents with 10-15 calls each, and complex research might use more than 10 subagents."
>
> **What it means:**
>
> A model has no sense of what a question is worth. Asked to research, it may start ten subagents for a fact that one search would settle. A table of question types and budgets gives it that sense from outside.

### Merging results

The lead's last job is to combine reports into one answer. Three habits make this safe. First, ask for structured results, so the merge can happen in code where possible: in our task, sorting accepted suppliers by landed cost is a line of Python, not a judgement. Second, check reports against each other and against the rules before trusting them. A report that accepts a supplier with 75 days of delivery breaks a rule the lead can test. Third, keep the source of every value, so a wrong number in the answer can be traced to the subagent and the page it came from.

### Stopping rules and budgets

Several agents multiply every loop problem of Chapter 1. Each subagent can loop, and the lead can keep starting more of them. CAMEL's agents thanked each other forever; Anthropic's early lead started "50 subagents for simple queries" (Chapter 4). The fixes are limits written in code: a maximum number of subagents per round (our lead's tool cuts the list at six), a maximum number of model calls per agent (25 per subagent, 12 for the lead), and a total token budget for the run. A prompt that says "be efficient" is a request; a cap in code is a rule.

Anthropic's lead also waits for each round of subagents to finish, which is simple and slow.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025) · Production reliability and engineering challenges
> [![A paragraph from the post: synchronous execution creates bottlenecks; currently, our lead agents execute subagents synchronously, waiting for each set of subagents to complete before proceeding; this simplifies coordination, but creates bottlenecks; the lead agent cannot steer subagents, subagents cannot coordinate, and the entire system can be blocked while waiting for a single subagent to finish searching; asynchronous execution would enable additional parallelism, but adds challenges in result coordination, state consistency and error propagation](/img/agents/ch5a-anthropic-sync.png)](/img/agents/ch5a-anthropic-sync.png)
>
> **What it says:** the lead waits "for each set of subagents to complete before proceeding", so "the entire system can be blocked while waiting for a single subagent".
>
> **What it means:**
>
> A round of subagents is as slow as its slowest member. Our experiment shows the same: ⟦SLOWEST⟧.
>
> Running subagents without waiting would be faster, but then the lead must handle results arriving in any order, subagents that are still running when the plan changes, and errors that arrive late. Anthropic chose the simple version and named its cost. For most teams that is the right first choice.

### Protocols: MCP for tools, A2A for agents

So far every agent ran in one program. When agents are built by different teams or companies, they need a shared way to talk. Two open protocols divide this job.

> [!DEFINITION] JSON-RPC 2.0
> A small standard for remote procedure calls written as JSON: a request names a method and its parameters and carries an id, and the response carries the same id with a result or an error. Both MCP and A2A use it.

The **Model Context Protocol (MCP)** was released by Anthropic on 25 November 2024. It connects an AI application to tools and data. Its specification, at version 2026-07-28 when we read it, names three parties: hosts, clients and servers.

> [!PAPER] Model Context Protocol, "Specification" (version 2026-07-28, accessed October 2026) · Overview
> [![A list from the MCP specification: servers offer resources, context and data for the user or the AI model to use; prompts, templated messages and workflows for users; and tools, functions for the AI model to execute](/img/agents/ch5a-mcp-features.png)](/img/agents/ch5a-mcp-features.png)
>
> **What it says:** servers offer "Resources: Context and data, for the user or the AI model to use", "Prompts: Templated messages and workflows for users", and "Tools: Functions for the AI model to execute". The protocol uses "JSON-RPC 2.0 messages" between "Hosts: LLM applications that initiate connections", "Clients: Connectors within the host application" and "Servers: Services that provide context and capabilities".
>
> **What it means:**
>
> MCP is about the edge between an agent and the things it uses. Northwind's news feed could be an MCP server with a `search_news` tool; any MCP-capable agent could then use it without new code.

The **Agent2Agent protocol (A2A)** was announced by Google on 9 April 2025 with more than 50 technology partners, as "an open protocol that complements Anthropic's Model Context Protocol". Its specification, at version 1.0.0 when we read it and now published under the Linux Foundation, defines how one agent finds another, gives it a task and gets results back.

> [!PAPER] A2A Protocol, "Specification" (version 1.0.0, accessed October 2026) · Section 1.2
> [![A list item from the A2A specification: opaque execution, agents collaborate based on declared capabilities and exchanged information, without needing to share their internal thoughts, plans, or tool implementations](/img/agents/ch5a-a2a-intro.png)](/img/agents/ch5a-a2a-intro.png)
>
> **What it says:** "Opaque Execution: Agents collaborate based on declared capabilities and exchanged information, without needing to share their internal thoughts, plans, or tool implementations."
>
> **What it means:**
>
> A2A assumes the other agent is a closed box run by someone else, like a freight broker's quoting agent. You cannot read its trace or its prompt. You can read its **Agent Card**, a JSON document that lists its skills and endpoint, and you can send it a **task**, which moves through a defined lifecycle and returns **artifacts** made of parts (text, files or structured data).
>
> That makes A2A the brief-and-result style of handoff, written into a standard. Cognition's full-trace sharing is impossible across such a boundary by design, so everything the remote agent needs must be in the task message.

The specification's Appendix B calls the two protocols "complementary". MCP connects an agent downward to its tools; A2A connects it sideways to other agents, and an A2A server agent "might use MCP to interact with several underlying tools". Neither protocol makes a multi-agent design a good idea. They make it possible across company boundaries, and the questions of this chapter stay the same.

{{FIG:a_protocols|MCP and A2A on one diagram. MCP connects an agent to tool and data servers; A2A connects it to other agents that it cannot see inside. Both use JSON-RPC 2.0 messages.}}

### Our experiment: a thin brief, a full brief and the full history

`ch5a_handoff.py` holds everything fixed except the handoff. The same subagent (same model, system prompt and three research tools) gets the same six suppliers: S03, S07, S08, S09, S12 and S19. These six contain most of the catalogue's traps: an open recall, a delivery time over the limit, a name that resembles a recalled company, a closed recall priced in rupees, a delivered-duty-paid price and a suspended audit. The subagent receives one of three handoffs:

- **thin**: one line, the kind Anthropic's early lead wrote: "Research suppliers S03, S07, S08, S09, S12, S19 for Northwind Home's 2027 kitchenware sourcing." (14 words)
- **full brief**: the five-part brief shown above (282 words).
- **full history**: the lead's whole conversation up to the handoff (the user's question, the buying rules it read, the supplier list and its plan) followed by the same one line (591 words).

We score twelve facts the lead needs: a verdict for each of the six suppliers, and the landed cost and delivery days of the three that should pass. A JSON report is read in code. A prose report goes through one extra model call that copies out what it states; that call's tokens are not charged to the condition.

```python
# code/agents/ch5a_handoff.py (abridged): the three handoffs differ only in what the subagent receives.
for cond in ('thin', 'brief', 'history'):
    brief = FULL if cond == 'brief' else THIN
    hist = history() if cond == 'history' else None          # the lead's messages up to the handoff
    report, secs = A.run_subagent('sub', brief, meter, history=hist)
    rows = grade(report, ext)                                 # 12 facts against gold.json
```

[![Terminal output of ch5a_handoff.py: three runs each of a one-line brief, a full brief and the full history for suppliers S03, S07, S08, S09, S12 and S19; the one-line brief got 4, 3 and 4 of 12 facts right with 6 tool calls and accepted the recalled supplier S03 every time; the full brief got 12, 12 and 10 facts right, always in the lead's JSON format; the full history got 10, 12 and 11 facts right but acted as the lead in all three runs; means 3.7, 11.3 and 11.0 of 12; total spend 9 US cents](/img/agents/ch5a_handoff-run.png)](/img/agents/ch5a_handoff-run.png)

{{FIG:a_exp2|Results of the final run of ch5a_handoff.py, means over three runs per condition. A one-line brief lost the rules: 3.7 of 12 facts right. The full brief and the full history both carried them (11.3 and 11.0), but in every history run the subagent answered the user's whole question instead of reporting in the lead's format.}}

**A one-line brief loses the rules, and the subagent does not notice.** With the thin brief, the subagent read its six pages (six tool calls) and stopped. It never searched the news and never looked up a port. In all three runs it accepted S03, whose open recall appears only in the news, and S19, whose audit is suspended. In two runs it also accepted S07, which takes 75 days. Its "delivery days" were the production lead times, because nothing told it to add the transit. Then it offered next steps it could take, "price-to-GBP conversions and freight/transit lookups" among them. This is the Flappy Bird failure in our own data: a sensible version of the task it was given, which was not the task the lead had in mind.

**A full brief carries the rules.** The full brief scored 12, 12 and 10 of 12, and every report came back in the requested JSON, ready to merge in code. The weak run accepted S03 and S19 and wrote "No open recalls or suspended audits found in news search" for both, although the news feed holds both items. A good brief does not guarantee a careful subagent, which is why the lead must check reports (below). Input tokens varied from 11,489 to 69,110 between runs, depending on whether the model batched its tool calls into 3 turns or spread them over 14.

**The full history carries the rules, and the lead's role with them.** The history condition scored 10, 12 and 11 of 12, as good as the brief on the facts. But in all three runs the subagent answered the user's original question. It chose "three suppliers" from its six and returned picks and rejections in the user's answer format, not a report in the lead's format. The history told it what the user wanted, and it acted on that, like CAMEL's role flipping. It worked out here only because exactly three of its six suppliers pass. With four passing suppliers, it would have dropped one before the lead could rank it against the other eighteen. It also left S19 out of one report, and twice gave S12 a delivery of 24 days instead of 25.

> [!WARNING]
> What this experiment does and does not show. Three runs per condition with one group of six suppliers is a demonstration. The full brief was written by us, with the rules copied in; a lead model writing its own briefs may do better or worse. The history condition passed the lead's messages as text, not as the original tool-call messages. The mechanisms are the findings: a thin brief loses the rules, the history keeps the rules but also the lead's role, and a full brief keeps the rules and the role, if its writer remembered both.

### Coordination in practice

| Decision | Default | Change it when |
|---|---|---|
| What goes down to a subagent | a full brief with the rules copied in | the subagent must act on a long, changing conversation (then share the trace, or do not split) |
| What comes back up | a structured result, stored, with a pointer | the lead needs the reasoning to judge a conflict |
| Shared state or messages | structured results the lead merges in code | several agents fill fields of one record at different times |
| Waiting | synchronous rounds | the slowest subagent dominates and results are independent |
| Limits | caps in code: subagents per round, calls per agent, tokens per run | never remove them; tune them from traces |
| Protocols | plain function calls inside one program; MCP for shared tools | another team's agent must take tasks from yours (A2A) |

> [!TIP] In production
> Log every brief and every report with the run id, so a wrong answer can be traced to the handoff that lost the fact. Test briefs the way you test prompts: keep a few tasks with known answers and run each new lead prompt against them. Check the output format of every subagent in code, and send failures back once with the error, as in Chapter 4's gates. Measure coverage directly: count the items each run actually read, because an agent's answer does not tell you what it skipped.

### Discussion

1. **Brief or history?** On this task, both carried the rules; only the brief also carried the subagent's role and the lead's format. My view: write a brief by default, with the rules copied in word for word, and share the full trace only when the receiving agent must continue the same conversation, as in a handoff to a specialist.
2. **Who should write the brief, a model or code?** When the split is predictable (one subagent per group of suppliers), a template filled by code cannot forget a rule. When the split depends on the question, the lead must write it, and then a check that every rule appears in every brief is cheap insurance.
3. **Does A2A change any of this?** It makes the brief style mandatory across companies, because a remote agent's trace is closed. It does not write the brief for you.
