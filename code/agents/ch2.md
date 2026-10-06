# Chapter 2 · The building blocks: model, tools, instructions, memory and state

> **Goal:** by the end of this chapter you can take the loop of Chapter 1 apart into its five blocks, say what each one is responsible for and what breaks when it is weak, choose a model for an agent with a measurement rather than a feeling, design a tool a model will actually call correctly, write instructions that behave like architecture rather than like a wish list, give an agent the right kind of memory for the right length of time, keep its state so that a run can be resumed, and decide, token by token, what belongs in the context window. Every choice comes with its trade-offs, a way to test whether it was right, and what tends to go wrong in production.

---

## 2.1 The five blocks and how they connect

Chapter 1 drew the agent as a loop: context in, decision out, tool runs, observation back, repeat. That picture is right but it hides the engineering. When you build a real agent you are making five separate kinds of decision, and when one fails in production you need to know which of the five it was. This chapter calls them the **building blocks**.

> [!DEFINITION] Building block
> One of the five parts every agent is assembled from: the **model** (reasoning), the **tools** (acting), the **instructions** (shaping), the **memory** (remembering) and the **state** (knowing where the loop is). Frameworks name and package them differently; the five responsibilities are the same in all of them.

{{FIG:ch2_blocks|The five building blocks and what flows between them. Instructions and retrieved memory are assembled, with the history, into the context the model reads on every turn; the model emits a tool call; the tool acts on the environment and returns an observation; the loop keeps the state (which step, what plan, which call is pending, how much budget is left, which errors have happened) and decides whether to call the model again.}}

Each block owns one question.

| Block | The question it answers | Made of | Typical failure when it is weak or missing |
|---|---|---|---|
| Model | "Given all this, what should happen next?" | an LLM behind an API or on your own hardware | wrong tool, wrong arguments, misread observation, gives up or loops |
| Tools | "How does the decision become an effect in the world?" | functions with a name, a description and a schema | the model cannot act, acts on the wrong thing, or drowns in the result |
| Instructions | "What is this agent for, and what must it never do?" | the system prompt: role, rules, formats, examples, stop conditions | drifts off task, ignores constraints, output that no parser accepts |
| Memory | "What does the agent know from before this turn, this run, or earlier runs?" | the context window, a scratchpad, a store it reads and writes | repeats work, forgets the user's constraint, contradicts itself |
| State | "Where in the task is the loop, and can it pick up where it left off?" | a small object: step, plan, pending call, budget, errors, status | cannot resume after a crash, retries blindly, never knows it is done |

Two of these are easy to confuse, and the confusion causes real bugs. **Memory** is *knowledge*: what the agent has learned or been told. **State** is *control*: where the loop is in the task. "The user prefers mornings" is memory. "We are at step 4, the email tool just failed once, and there are four steps of budget left" is state. A system that keeps only memory cannot resume a crashed run safely; a system that keeps only state cannot remember what it was told. Section 2.6 draws the line in detail.

The blocks are connected by one artefact: the **context** the model receives on each turn. Instructions go in first and stay fixed. Tool definitions go in so the model knows what it may call. Memory contributes whatever was retrieved. The history of the run so far, including every observation, is appended. The last observation is the newest thing in it. The model reads all of that and writes one decision. This is why Section 2.7 on context engineering sits at the end of the chapter: it is where the other four blocks meet, and it is the budget they all draw on.

Chapter 1 gave the arithmetic of why reliability compounds across steps. This chapter is about the levers that raise per-step reliability, and almost all of them live in these blocks: a model that calls tools correctly, tools that are hard to misuse, instructions that remove ambiguity, memory that puts the right fact in front of the model, and state that makes a failed step recoverable rather than fatal.

### Discussion

A team designing an agent should argue about these before writing a line:

1. **Which block are we weakest at, and how would we know?** Most teams assume the model is the weak block and reach for a bigger one. In the production write-ups of Chapter 1, the weak blocks were tools (LinkedIn's malformed payloads) and instructions and verification (the MAST taxonomy). My view: instrument every block in the trace before you upgrade any of them.
2. **Do we need all five?** A single-turn classifier needs a model and instructions. A workflow with tools needs no agent state. Adding a block you do not need adds a place to fail. My view: start with model, tools and instructions, add memory when the task outlives one context, add explicit state when runs are long enough to crash.
3. **Who owns each block?** In a team, the prompt is often owned by product, the tools by backend engineers, the model choice by a platform team, and memory by nobody. My view: give each block an owner and an eval, or the unowned one will be where the incidents come from.
4. **What is the unit of change?** A change to a tool description changes the model's behaviour as surely as a change to the prompt. My view: version all five blocks together, as one release, and test them together.

## 2.2 The model: choosing the brain

The model is the only block that reasons, and it is also the only block you did not write. You choose it, you cannot debug it, and you can only measure it. Choosing well therefore means knowing what to measure.

### What matters for an agent, as opposed to a chat

A model that is excellent at writing essays can be mediocre inside a loop. The agent asks it a narrower question many times over: read this growing context, pick one of these tools, fill in these arguments exactly, or decide that no tool is needed. The factors that matter are different from the ones on a general benchmark.

| Factor | What it means for an agent | How to measure it on your task |
|---|---|---|
| Capability on the task | can it do the hard reasoning steps at all (plan, diagnose, synthesise)? | a labelled set of the hard steps, scored by a rubric or a checker |
| Tool-call reliability | does it pick the right tool, with valid arguments, when it should, and no tool when it should not? | tool-selection accuracy, argument accuracy and refusal rate on a labelled set (below) |
| Latency | the loop multiplies it: ten steps at 3 s is 30 s before any tool time | p50 and p95 per call, measured with your real prompt length |
| Cost | the loop multiplies it too, and the context grows every step | cost per task over a sample of real runs, not per call |
| Context window | can one run fit, or will you need compaction (Section 2.7)? | the largest context your traces reach, with headroom |
| Structured-output support | can it be made to emit valid JSON every time, so your parser never breaks? | parse-failure rate over a thousand calls |
| Open-weight versus API | control over data, hosting and version pinning against convenience and the frontier of capability | a decision about data residency, cost at volume and operations, not a benchmark |

> [!DEFINITION] Tool-call accuracy
> The share of steps on which the model chose the tool a labelled set says it should have chosen. Measured on your own tool set with your own requests. A model can score well on a public leaderboard and poorly on your ten badly named tools.

> [!DEFINITION] Argument accuracy
> The share of tool calls whose arguments are valid against the schema *and* correct for the request: the right email addresses, the right date, the right units. Schema validity is cheap to check; correctness needs labels.

> [!DEFINITION] Refusal rate (relevance detection)
> How often the model correctly emits *no* tool call when none of the available tools fits the request. A model that always calls something will call the wrong thing when the right one is missing. The Berkeley leaderboard calls this "function relevance detection".

> [!DEFINITION] Structured output
> A model reply constrained to a format your code can parse, usually JSON matching a schema. Some APIs enforce the schema during decoding; others only encourage it in the prompt. For an agent, every tool call is a structured output, so this is not a nice-to-have.

> [!DEFINITION] Open-weight model
> A model whose weights you can download and run yourself (or have hosted for you), as opposed to one reachable only through a vendor's API. Open weights give you version pinning, data control and a fixed cost at volume; the API gives you the frontier of capability and no operations.

{{FIG:ch2_model_factors|The usual shape of the trade-off between three classes of model, sketched qualitatively. Capability and tool-call reliability tend to rise with size; speed, cost and control fall. The sketch is not a measurement: the only numbers that matter are the ones from a per-step evaluation on your own tasks.}}

### What the research says about tool calling

The ability to call a function correctly is a trained skill, and it varies more between models than general fluency does. The Gorilla paper from Berkeley, in May 2023, was the first to measure it carefully.

> [!PAPER] Patil et al. (2023), Gorilla · Abstract · page 1
> [![Abstract of the Gorilla paper with highlighted phrases: their potential to effectively use tools via API calls remains unfulfilled; inability to generate accurate input arguments and their tendency to hallucinate the wrong usage of an API call; adapt to test-time document changes](/img/agents/ch2-gorilla-abstract.png)](/img/agents/ch2-gorilla-abstract.png)
>
> **Context:** the abstract, arXiv 2305.15334, posted 24 May 2023. The paper fine-tunes a 7-billion-parameter model to write API calls for machine-learning hubs and pairs it with a document retriever.
>
> **What it says:** even the best models of the time had an "inability to generate accurate input arguments" and a "tendency to hallucinate the wrong usage of an API call". With retrieval over API documentation, the fine-tuned model could "adapt to test-time document changes".
>
> **Why it matters:** the two failure modes named here, wrong arguments and invented usage, are still the two most common tool-calling errors in traces today. And the retrieval idea foreshadows how large tool sets are handled now: do not put every tool in the context; fetch the ones that fit.

The same group turned the measurement into a public benchmark, the Berkeley Function-Calling Leaderboard.

> [!PAPER] Berkeley (Yan, Mao, Ji et al.), "Berkeley Function-Calling Leaderboard" (Gorilla blog, last updated August 2024)
> [![The opening paragraph of the BFCL blog post: the first comprehensive evaluation on the LLM's ability to call functions and tools; function calls of various forms including parallel and multiple; the model's ability to withhold picking any function when the right function is not available; the leaderboard now also includes cost and latency](/img/agents/ch2-bfcl-blog.png)](/img/agents/ch2-bfcl-blog.png)
>
> **Context:** the first post describing the benchmark, by the Gorilla team at UC Berkeley. Later versions added enterprise-contributed functions (V2), multi-turn interaction (V3) and agentic evaluation (V4).
>
> **What it says:** the benchmark covers "function calls of various forms including parallel ... and multiple", several languages, executed calls, and "the model's ability to withhold picking any function when the right function is not available". The post says the dataset holds about 2,000 question-function-answer pairs, including 100 Java, 50 JavaScript, 70 REST, 100 SQL and 1,680 Python cases.
>
> **Why it matters:** this is the public version of the table above, and it defines the categories you should copy into your own per-step eval: simple calls, choosing one function among several, parallel calls, and correctly calling nothing.

> [!PAPER] Berkeley, "Berkeley Function Calling Leaderboard V4" (live leaderboard page, updated 12 April 2026)
> [![The top of the live leaderboard table: columns for rank, overall accuracy, model, estimated cost in dollars for the whole benchmark, and agentic sub-scores for web search and memory; the first rows show overall accuracies in the high sixties and seventies with benchmark costs ranging from under five dollars to almost three hundred](/img/agents/ch2-bfcl-table.png)](/img/agents/ch2-bfcl-table.png)
>
> **Context:** the live page, which the team updates as models are released; the screenshot is from the version marked "Last Updated: 2026-04-12". Rows and numbers change; read the page, not this picture, for current values.
>
> **What it says:** the top rows at that date had overall accuracies between about 69 and 77 out of 100, and the cost of running the whole benchmark ranged from a few dollars to almost three hundred, with no simple relation between the two. The columns also split accuracy by agentic category (web search, memory, multi-turn).
>
> **Why it matters:** two things to take from the shape of the table, whatever the current rows say. First, even the best models miss a quarter of the cases, so tool calling is a measured risk, not a solved problem. Second, cost and accuracy are separate axes: the expensive model is not reliably the accurate one. This is the empirical ground for the router below.

> [!WARNING]
> A public leaderboard measures a model on *its* tools and *its* requests. Your tool set has different names, your users write different requests, and your hard cases are not in the benchmark. Use the leaderboard to shortlist; use your own labelled set to choose.

### The router: a small model for routine steps, a large one for hard ones

Most steps of most agents are routine. Reading a tool result and deciding to call the obvious next tool, extracting a field, reformatting an answer: a small model does these as well as a large one, several times faster and at a fraction of the cost. A few steps are hard: making the plan, recovering from an unexpected error, choosing between two tools whose purposes overlap. The **router** pattern sends each step to the model that fits it.

> [!DEFINITION] Model router
> A component inside the loop that decides, per step, which model to call. It can be a fixed rule (the planning step always goes to the large model), a cheap classifier, or the small model itself declaring low confidence. The alternative, one model for every step, is simpler and is the right default until measurement says otherwise.

{{FIG:ch2_router|A router inside the loop sends routine steps (formatting, extraction, an obvious tool choice) to a small model and hard steps (planning, recovery, ambiguous tool choice) to a large one, with an escalation path when the small model fails a check. A per-step evaluation set, labelled routine or hard, decides where the line goes.}}

| | One model for every step | Router: small for routine, large for hard |
|---|---|---|
| Pros | one thing to evaluate, version and reason about; no routing errors | most steps cheap and fast; the large model is paid for only where it changes the outcome |
| Cons | every step pays the price of the hardest step | two models to evaluate and version; a wrong route is a silent quality loss; the router itself can be the weak step |
| When to pick it | early, always; and whenever the step mix is mostly hard | when traces show that most steps are routine *and* cost or latency is what blocks shipping |
| How to tell it was right | the per-step eval passes at the required rate | the per-step eval passes at the same rate as the single large model, at lower cost; the escalation rate is low and stable |
| Production failure | cost or latency too high to serve | the small model drifts after a version change and the router does not notice; escalations spike after a prompt edit |

> [!TIP] In production
> Build the **per-step eval set** before you build the router. Take a few hundred steps from real traces, label each one routine or hard, and record what a good decision looked like. Run both models over it. Where the small model matches the large one, route to the small one; where it does not, keep the large one and keep measuring. Re-run the set whenever either model or any prompt changes. Without this set the router is a guess that saves money until the day it does not.

### Discussion

1. **Should we pick the model first or the tools first?** The model is the most visible choice, but tool-call reliability depends on the tool set. My view: design the tools and the eval set first, then let the eval choose the model; otherwise you choose a model for tools you will redesign.
2. **Is a bigger model ever the wrong fix?** Yes: when the failure is a vague tool description or a missing stop condition, a bigger model reads the same ambiguity more cleverly and still guesses. My view: upgrade the model only after the trace shows the step was reasoned well and still wrong.
3. **How do we handle model version changes?** Vendors retire and replace models; open weights do not change under you but need operations. My view: pin versions explicitly, treat an upgrade as a release with the full eval, and never let "latest" into production.
4. **Open weights or API?** The honest answer depends on volume, data rules and the team's operations capacity. My view: start on an API to learn what the task needs; move the routine steps to an open-weight model only when the per-step eval proves parity and the volume pays for the operations.

## 2.3 Tools: giving the model hands

A model can only produce text. Everything an agent *does* happens through a tool, and the mechanism by which text becomes an action is what vendors call **function calling** (or tool use). Because the model never runs anything itself, tools are the boundary where your code, your permissions and your safety checks live. They are also, in the experience of every team that has published about it, the block most worth engineering carefully.

### What a tool is and how the round trip works

> [!DEFINITION] Tool definition
> A description of a function the model may ask for, with three parts: a **name**, a natural-language **description** of what it does and when to use it, and an **input schema** (JSON Schema) that lists the arguments, their types, which are required, and what each means. The definition is put into the model's context; the implementation stays in your code.

> [!DEFINITION] JSON Schema
> A standard way to describe the shape of a JSON value: an object with these properties, of these types, with these ones required. Tool arguments are described in JSON Schema so that both the model and your validator read the same contract.

> [!DEFINITION] Function calling (tool use)
> The protocol by which a model emits a structured request to run a named function with arguments, instead of plain text; your code runs the function and returns the result as a new message; the model continues from there. The model proposes, your code disposes.

{{FIG:ch2_roundtrip|One function-calling round trip. The request carries the system prompt, the tool definitions and the user turn; the model replies with a structured tool call and a stop reason that says so; your code validates the arguments and runs the function; the result goes back as a tool-result turn; the model replies with text or with another call. Steps one to four repeat while the model keeps asking for tools, and the whole exchange is the context the next call sees.}}

The script `ch2_tools.py` prints the exact messages of one round trip for the meeting-scheduling example that runs through this chapter. The model is a **stub** (a function that returns what a real model would be expected to return at each point), so the script runs with no API key; the tool definition and the message shapes are real.

[![Terminal output of ch2_tools.py part one: the tool definitions sent in the request, the full JSON schema of calendar_find_free_slots with attendee_emails, duration_minutes, window_start, window_end and earliest_hour, the user turn, the assistant reply containing a tool_use block with id call_01 and filled-in arguments, the tool_result turn sent back with three slots and a time zone, and the final assistant text listing the three slots](/img/agents/ch2_tools-run.png)](/img/agents/ch2_tools-run.png)

Four things in that output are the whole mechanism. The tool definitions travel *in the request*, so they cost input tokens on every turn. The model's reply is not text but a `tool_use` block with an id, a name and an `input` object that matches the schema. Your code, not the model, runs `find_free_slots`, and it is your code that could refuse, log, rate-limit or ask a human first. The result goes back as a `tool_result` tied to the id, and the model's next reply is ordinary text because it decided it was done.

### Where tool use came from

Chapter 1 introduced Toolformer as the paper that showed tool use can be learned rather than prompted. Its first figure is worth a second look here, because it shows what a tool call *is* from the model's point of view: a piece of text, in a fixed format, inserted exactly where the information is needed.

> [!PAPER] Schick et al. (2023), Toolformer · Figure 1 · page 1
> [![Figure 1 of the Toolformer paper next to its abstract: four example sentences in which the model inserts a call to a question-answering system, a calculator, a machine translation system and a Wikipedia search, each written inline as an API call with its result, such as Calculator of 400 divided by 1400 returning 0.29](/img/agents/ch2-toolformer-figure1.png)](/img/agents/ch2-toolformer-figure1.png)
>
> **Context:** page 1 of arXiv 2302.04761; the figure sits next to the abstract.
>
> **What it says:** the model "autonomously decides to call different APIs" (a question-answering system, a calculator, a translation system, a search engine) and splices the call and its result into the text where it helps the prediction.
>
> **Why it matters:** the call `[Calculator(400 / 1400) → 0.29]` is a tool call written as tokens: a name, arguments, a result. Commercial function calling wraps the same idea in JSON and a stop reason, but the model is still doing exactly this: emitting a formatted request at the point where it needs something it cannot compute itself.

Scale came next. The ToolLLM paper (July 2023) asked what happens when the tool set is not four tools but thousands.

> [!PAPER] Qin et al. (2023), ToolLLM · Figure 1 · page 2
> [![Figure 1 of the ToolLLM paper: three phases, API collection from RapidAPI, instruction generation and solution path annotation into the ToolBench dataset, then training of an API retriever and a fine-tuned ToolLLaMA; at inference the API retriever recommends relevant APIs to ToolLLaMA, which performs multiple rounds of API calls to derive the final answer, evaluated by ToolEval](/img/agents/ch2-toolllm-figure1.png)](/img/agents/ch2-toolllm-figure1.png)
>
> **Context:** the overview figure of arXiv 2307.16789, published at ICLR 2024. The paper reports collecting 16,464 REST APIs from RapidAPI across 49 categories.
>
> **What it says:** at inference "the API retriever recommends relevant APIs to ToolLLaMA, which performs multiple rounds of API calls to derive the final answer".
>
> **Why it matters:** with sixteen thousand tools, no model can be shown them all. The paper's answer is a *retriever in front of the tool set*: select a handful of candidate tools per request, then let the model choose among those. Every production system with a large tool catalogue now does some version of this, and Anthropic's code-execution post below reports the same move by a different route.

### How to design tools a model can use

The most useful engineering guidance on this is Anthropic's September 2025 post "Writing effective tools for agents", which came out of optimising the company's own internal tools against evaluations. Its five principles are a checklist; this section reads them one at a time and adds the trade-offs.

**1. Choose which tools to build, and do not build the rest.**

> [!PAPER] Anthropic, "Writing effective tools for agents" (Engineering blog, September 2025)
> [![A paragraph from the post: we recommend building a few thoughtful tools targeting specific high-impact workflows, which match your evaluation tasks and scaling up from there; in the address book case, you might choose to implement a search_contacts or message_contact tool instead of a list_contacts tool](/img/agents/ch2-anthropic-tools-choosing.png)](/img/agents/ch2-anthropic-tools-choosing.png)
>
> **Context:** the section "Choosing the right tools for agents", after the observation that "more tools don't always lead to better outcomes" and that a common error is "tools that merely wrap existing software functionality or API endpoints".
>
> **What it says:** build "a few thoughtful tools targeting specific high-impact workflows", matched to your evaluation tasks. Prefer `search_contacts` or `message_contact` over `list_contacts`.
>
> **Why it matters:** a model has limited context; a computer has cheap memory. A tool that returns everything and lets the model search through it moves the search into the most expensive component you have. Design the tool around what the agent needs to *decide*, not around what the API happens to expose.

> [!PAPER] Anthropic, "Writing effective tools for agents" (Engineering blog, September 2025)
> [![Three bullet points from the post: instead of list_users, list_events and create_event, consider a schedule_event tool which finds availability and schedules an event; instead of read_logs, a search_logs tool which only returns relevant log lines and some surrounding context; instead of get_customer_by_id, list_transactions and list_notes, a get_customer_context tool which compiles all of a customer's recent and relevant information at once](/img/agents/ch2-anthropic-tools-consolidate.png)](/img/agents/ch2-anthropic-tools-consolidate.png)
>
> **Context:** the same section, on consolidating several API calls into one tool.
>
> **What it says:** replace `list_users`, `list_events` and `create_event` with one `schedule_event`; replace `read_logs` with `search_logs` that returns "relevant log lines and some surrounding context"; replace three customer look-ups with one `get_customer_context`.
>
> **Why it matters:** this is the "reduce n" move from Chapter 1's reliability arithmetic applied to tools. Three calls that each succeed 95% of the time succeed together 86% of the time; one consolidated call that does the same work succeeds 95% of the time and costs two fewer round trips of context. The trade-off is flexibility: a consolidated tool does one workflow well and others not at all.

**2. Namespace the tools.**

> [!PAPER] Anthropic, "Writing effective tools for agents" (Engineering blog, September 2025)
> [![A paragraph from the post: namespacing, grouping related tools under common prefixes, can help delineate boundaries between lots of tools; MCP clients sometimes do this by default; for example, namespacing tools by service such as asana_search and jira_search, and by resource such as asana_projects_search and asana_users_search, can help agents select the right tools at the right time](/img/agents/ch2-anthropic-tools-namespacing.png)](/img/agents/ch2-anthropic-tools-namespacing.png)
>
> **Context:** the section "Namespacing your tools", written for agents that may have "dozens of MCP servers and hundreds of different tools".
>
> **What it says:** group tools "under common prefixes", by service (`asana_search`, `jira_search`) and by resource (`asana_projects_search`, `asana_users_search`). The post adds that the choice between prefix and suffix naming had "non-trivial effects" on their tool-use evaluations, and that the effect varied by model.
>
> **Why it matters:** the name is the first thing the model reads when choosing, and two tools called `search` from different servers are a coin toss. A naming scheme is cheap to adopt and measurably matters; the last sentence is also a reminder that these effects are empirical, so test the scheme on your eval rather than copying one.

> [!DEFINITION] Namespacing (tools)
> Prefixing tool names with the service and the resource they act on (`calendar_find_free_slots`, `crm_get_customer_context`) so that the name alone tells the model which system and which object a tool touches. Essential once the tool set crosses about a dozen tools or spans several servers.

**3. Return meaningful context, not raw payloads.**

> [!PAPER] Anthropic, "Writing effective tools for agents" (Engineering blog, September 2025)
> [![A paragraph from the post under the heading Returning meaningful context from your tools: tool implementations should take care to return only high signal information back to agents; they should prioritize contextual relevance over flexibility, and eschew low-level technical identifiers such as uuid, 256px_image_url, mime_type; fields like name, image_url, and file_type are much more likely to directly inform agents' downstream actions and responses](/img/agents/ch2-anthropic-tools-context.png)](/img/agents/ch2-anthropic-tools-context.png)
>
> **Context:** the section on tool responses. The post goes on to say that resolving opaque UUIDs to names "significantly improves Claude's precision in retrieval tasks by reducing hallucinations", and suggests a `response_format` parameter (concise or detailed) so the agent can ask for identifiers only when it needs them for a later call.
>
> **What it says:** return "only high signal information", prefer "contextual relevance over flexibility", and avoid "low-level technical identifiers".
>
> **Why it matters:** everything a tool returns is read by the model, token by token, on this turn and every later turn until compaction. A response designed for a program (ids, mime types, nested metadata) wastes the budget and gives the model opaque strings to copy wrongly. A response designed for a reader gives it names it can reason about.

**4. Make responses token-efficient.**

> [!PAPER] Anthropic, "Writing effective tools for agents" (Engineering blog, September 2025)
> [![A paragraph from the post: we suggest implementing some combination of pagination, range selection, filtering, and/or truncation with sensible default parameter values for any tool responses that could use up lots of context; for Claude Code, we restrict tool responses to 25,000 tokens by default; we expect the effective context length of agents to grow over time, but the need for context-efficient tools to remain](/img/agents/ch2-anthropic-tools-tokens.png)](/img/agents/ch2-anthropic-tools-tokens.png)
>
> **Context:** the section "Optimizing tool responses for token efficiency". The next paragraphs add that truncation should come with instructions that steer the agent (make smaller, targeted searches) and that error responses should be "specific and actionable" rather than "opaque error codes or tracebacks".
>
> **What it says:** use "pagination, range selection, filtering, and/or truncation with sensible default parameter values"; the company's own coding agent caps tool responses at "25,000 tokens by default".
>
> **Why it matters:** Section 2.7 computes what one oversized response costs: a 25,000-token log dump at step 5 of a thirty-step run, re-sent on every later step, adds about 650,000 input tokens, almost as much as the entire run without it. The cap is not a detail; it is a cost control and an attention control in one line of code.

**5. Prompt-engineer the descriptions.**

> [!PAPER] Anthropic, "Writing effective tools for agents" (Engineering blog, September 2025)
> [![A paragraph from the post: when writing tool descriptions and specs, think of how you would describe your tool to a new hire on your team; consider the context that you might implicitly bring, specialized query formats, definitions of niche terminology, relationships between underlying resources, and make it explicit; avoid ambiguity by clearly describing and enforcing with strict data models expected inputs and outputs; in particular, input parameters should be unambiguously named: instead of a parameter named user, try a parameter named user_id](/img/agents/ch2-anthropic-tools-descriptions.png)](/img/agents/ch2-anthropic-tools-descriptions.png)
>
> **Context:** the section "Prompt-engineering your tool descriptions", which the post calls "one of the most effective methods for improving tools". It reports that "precise refinements to tool descriptions" were part of reaching state-of-the-art results on a public software-engineering benchmark.
>
> **What it says:** describe the tool "to a new hire on your team": make implicit context explicit, remove ambiguity, enforce inputs with strict schemas, and name parameters so they cannot be misread (`user_id`, not `user`).
>
> **Why it matters:** the description is the only training the model gets on your tool. It is a prompt, it is read on every turn, and it is the cheapest lever you have on tool-call accuracy.

Anthropic's December 2024 post "Building effective agents" made the same point from the other direction a year earlier, in an appendix on what it calls the agent-computer interface.

> [!PAPER] Anthropic, "Building effective agents" (Research blog, December 2024)
> [![A sentence from Appendix 2 of the post: one rule of thumb is to think about how much effort goes into human-computer interfaces, HCI, and plan to invest just as much effort in creating good agent-computer interfaces, ACI](/img/agents/ch2-anthropic-aci.png)](/img/agents/ch2-anthropic-aci.png)
>
> **Context:** Appendix 2, "Prompt engineering your tools". The appendix also recommends giving the model "enough tokens to think before it writes itself into a corner", keeping formats "close to what the model has seen naturally occurring in text on the internet", and avoiding formatting overhead such as counting lines or escaping strings. The team reports that for their coding agent they "spent more time optimizing our tools than the overall prompt", and that switching a file tool from relative to absolute paths made the model use it "flawlessly".
>
> **What it says:** invest as much in the "agent-computer interface (ACI)" as teams invest in human-computer interfaces.
>
> **Why it matters:** this reframes tool design as interface design. The same discipline that produces a usable form for a person (clear labels, sensible defaults, errors that say what to fix, no way to enter nonsense) produces a usable tool for a model. The absolute-path story is the whole method in one anecdote: find the mistake in the traces, change the argument so the mistake cannot be made.

{{FIG:ch2_tool_good_bad|A tool the model will misuse next to a tool it can use. The good tool has a namespaced name, a description that says when to use it and what it returns, unambiguous argument names, a bounded return size, an error message that says what to do next, no side effects and safe retries. The bad tool forces the model to guess the argument, read forty thousand tokens to find one row, and cannot tell a failure from an empty answer.}}

| Property | A tool the model will misuse | A tool the model can use | Why it matters in the loop |
|---|---|---|---|
| Name | `get_data`, `fetch`, `do_calendar` | `calendar_find_free_slots` (service, resource, verb) | the name is read first when choosing |
| Description | "Gets data." | what it does, when to use it, what it returns, what it returns when there is nothing | the only training the model gets on the tool |
| Argument count | one vague `q: string`, or twelve optional flags | three to six named, typed arguments with defaults | each argument is a place to be wrong |
| Argument names | `user`, `date`, `id` | `user_id`, `window_start` (ISO date), `event_id` | ambiguity becomes a wrong value |
| Return size | the whole table | at most N items, paginated or filtered, with a cap | every token returned is re-read every later turn |
| Error messages | "Error 500", a stack trace | "window_end is before window_start: swap them or widen the window" | the model can only recover from errors it can read |
| Empty result | raises | returns an empty list and says so | a model that sees an exception may retry forever |
| Idempotency | a retry creates a second event | safe to retry, or takes an idempotency key | the loop *will* retry |
| Side effects | undocumented | declared (read-only, destructive, external) and checked by the loop | permissions live at this boundary |

> [!DEFINITION] Idempotency
> A tool is idempotent if calling it twice with the same arguments has the same effect as calling it once. Reads are naturally idempotent; writes are made so with an idempotency key (`create_event` with the same key returns the same event). Because agent loops retry after errors and resume after crashes (Section 2.6), non-idempotent write tools are a standing source of duplicates.

### MCP: a standard way to expose tools

Until late 2024 every team wired its tools into its agent by hand, and every tool vendor shipped a different integration for every agent product. The **Model Context Protocol** (MCP), released by Anthropic in November 2024 and since adopted widely, standardises the wire: a server exposes tools (and two other things) over a protocol; any host that speaks MCP can use them.

> [!DEFINITION] Model Context Protocol (MCP)
> An open standard for connecting AI applications to external systems. An MCP **server** exposes **tools** (functions the model can call), **resources** (data it can read) and **prompts** (templates it can use) over JSON-RPC, locally over standard input and output or remotely over HTTP. An MCP **host** (the AI application) creates one MCP **client** per server and puts the servers' tool definitions into the model's context.

> [!PAPER] Model Context Protocol, "Architecture overview" (modelcontextprotocol.io documentation, protocol version 2026-07-28)
> [![Three bullet points from the MCP architecture page: MCP Host, the AI application that coordinates and manages one or multiple MCP clients; MCP Client, a component that maintains a connection to an MCP server and obtains context from an MCP server for the MCP host to use; MCP Server, a program that provides context to MCP clients](/img/agents/ch2-mcp-roles.png)](/img/agents/ch2-mcp-roles.png)
>
> **Context:** the "Participants" section of the architecture page. The paragraph above the list says MCP "follows a client-server architecture" in which the host "establishes connections to one or more MCP servers" by "creating one MCP client for each MCP server".
>
> **What it says:** the host "coordinates and manages one or multiple MCP clients"; a client "maintains a connection to an MCP server"; a server is "a program that provides context to MCP clients".
>
> **Why it matters:** the vocabulary is confusing on first contact because "client" is not the user's application; the host is. Your agent is the host. Each tool provider is a server. The clients are plumbing the host creates, one per server.

> [!PAPER] Model Context Protocol, "Architecture overview" (modelcontextprotocol.io documentation, protocol version 2026-07-28)
> [![Three bullet points from the MCP architecture page under Primitives: Tools, executable functions that AI applications can invoke to perform actions such as file operations, API calls, database queries; Resources, data sources that provide contextual information to AI applications such as file contents, database records, API responses; Prompts, reusable templates that help structure interactions with language models such as system prompts, few-shot examples](/img/agents/ch2-mcp-primitives.png)](/img/agents/ch2-mcp-primitives.png)
>
> **Context:** the "Primitives" section, which the page calls "the most important concept within MCP". Each primitive has methods for discovery (`tools/list`), retrieval and, for tools, execution (`tools/call`), so a host can list what a server offers at run time.
>
> **What it says:** servers expose "Tools: executable functions", "Resources: data sources that provide contextual information" and "Prompts: reusable templates".
>
> **Why it matters:** the three primitives map onto three of this chapter's blocks. Tools are the tools block; resources are retrieved memory, served by someone else; prompts are a piece of the instructions block that the tool provider writes because it knows its tools best. A server that ships a good "triage this issue" prompt alongside its `issue_create` tool is doing the ACI work for you.

{{FIG:ch2_mcp|The participants of the Model Context Protocol as its documentation describes them. The host (your agent, an IDE, a desktop application) creates one client per server; each server exposes tools, resources and prompts over JSON-RPC, locally over standard input and output or remotely over HTTP; the host lists the tools from every client and puts their definitions into the model's context.}}

| | Hand-written tool wrappers | MCP servers |
|---|---|---|
| Pros | full control of names, descriptions, return shapes and error text; no protocol overhead; trivial to test | reuse of hundreds of existing servers; one integration per host, not per tool; dynamic discovery; standard annotations for destructive or open-world tools |
| Cons | every tool is bespoke work; every agent product needs its own integration | you inherit the server author's names, descriptions and return sizes; many servers wrap endpoints one-to-one, exactly the anti-pattern above; tool definitions from many servers can flood the context |
| When to pick it | a small, stable tool set you own, where every description is tuned on your eval | many systems to connect, especially third-party ones; or you are shipping tools for other people's agents |
| How to tell it was right | tool-selection and argument accuracy on your eval; return sizes within budget | the same metrics, measured *with the servers connected*; plus the context cost of the loaded definitions |
| Production failure | a wrapper silently drifts from the API it wraps | a server update renames a tool or changes a return shape and your prompt's tool guidance is now wrong; a server with sixty tools pushes your definitions past the attention budget |

The context cost of many servers is real and measured. Anthropic's November 2025 post on code execution with MCP describes an agent that, instead of loading every tool definition, writes code that discovers tools as files and calls them from a sandbox.

> [!PAPER] Anthropic, "Code execution with MCP: Building more efficient agents" (Engineering blog, November 2025)
> [![A paragraph from the post: the agent discovers tools by exploring the filesystem, listing the servers directory to find available servers like google-drive and salesforce, then reading the specific tool files it needs to understand each tool's interface; this lets the agent load only the definitions it needs for the current task; this reduces the token usage from 150,000 tokens to 2,000 tokens, a time and cost saving of 98.7 percent](/img/agents/ch2-anthropic-codeexec.png)](/img/agents/ch2-anthropic-codeexec.png)
>
> **Context:** the post opens with two costs of direct tool calling at scale: "tool definitions overload the context window" and "intermediate tool results consume additional tokens" (a transcript fetched from one system and written into another passes through the model twice). The remedy presents servers as code APIs in a file tree.
>
> **What it says:** the agent "load[s] only the definitions it needs for the current task", which in their example "reduces the token usage from 150,000 tokens to 2,000 tokens".
>
> **Why it matters:** this is ToolLLM's retriever again, implemented as a file system the model explores, and it shows how large the definition cost becomes with hundreds of tools. Whether or not you adopt code execution, the principle holds: do not put every tool in every turn. Load what the step needs.

> [!DEFINITION] Tool annotations
> Metadata on a tool definition that declares its nature: read-only, destructive, idempotent, or reaching the open world. MCP standardises these so a host can, for example, require approval for every destructive tool regardless of who wrote it. They are hints from the server author, not guarantees, so the host's permission checks must not rely on them alone.

### Failure modes

Every one of these appears in traces of real systems, and each has a cheap detector.

| Failure | What it looks like in the trace | Why it happens | Detector |
|---|---|---|---|
| Too many tools | the model picks a plausible but wrong tool; long deliberation; more steps than needed | definitions compete for attention; many are never relevant | tool-selection accuracy falls as tools are added; definition tokens per turn |
| Overlapping tools | `search` and `find` and `lookup` chosen interchangeably, with different results | no clear boundary in names or descriptions | confusion matrix between tools on the eval set |
| Huge return payloads | one observation of tens of thousands of tokens; later steps lose earlier facts | the tool returns what the API returns | observation size per step; p95 of tool result tokens |
| Success on failure | the tool returns `200 OK` with an error string inside; the model carries on as if it worked | the wrapper does not inspect the body | tool results containing error words with a success status |
| Ambiguous arguments | `date: "next Monday"`, `user: "Priya"` where an id was needed | the schema allows free text where it should not | argument validation failures; downstream "not found" errors |
| Non-idempotent retries | two events, two emails, two tickets | the loop retried a write after a timeout | duplicate detection on side effects |
| Missing "no tool" option | the model calls a tool for a question it could answer, or calls a wrong one when none fits | nothing in the instructions says when *not* to call | refusal-rate cases in the eval |

### How to evaluate a tool set

Three numbers, all from a labelled set of real requests.

1. **Tool-selection accuracy**: did the model pick the labelled tool (or correctly pick none)? Report it per tool as well as overall; one badly described tool can hide behind a good average.
2. **Argument correctness**: schema-valid *and* semantically right. Schema validity is free; semantic correctness needs the label to contain the expected arguments.
3. **Trajectory length**: how many tool calls did the task take against the labelled minimum? Extra calls mean the tools do not match how the task decomposes, or the returns do not contain what the next step needs.

The second half of `ch2_tools.py` shows the mechanism of the first metric with a deliberately simple stub: a chooser that picks the tool whose name and description share the most words with the request.

[![Terminal output of ch2_tools.py part two: ten labelled requests run against a set of five bare endpoint wrappers named get_calendar, post_calendar_event, get_crm_record, post_mail and get_search, with seven of ten wrong and 3 of 10 correct, then the same ten requests against five described tools named calendar_find_free_slots, calendar_create_event, crm_get_customer_context, email_send and docs_search, all ten correct; the last line reads selection accuracy wrappers 30 percent, clear 100 percent, a stub, not a model, it shows the mechanism, not a benchmark](/img/agents/ch2_tools_select-run.png)](/img/agents/ch2_tools_select-run.png)

Read the 30% against 100% for what it is: the output of a word-overlap stub, not a model's score. What it demonstrates is *why* descriptions matter. The wrapper set (`GET /calendar`, `POST /mail`) shares almost no vocabulary with how people ask for things, so the chooser has nothing to match and falls back on the first tool. The described set names the service, the verb and the words a user would use. A real model is far better than a word count at bridging that gap, but it bridges it with the same material, and the gap is what your eval measures.

> [!TIP] In production
> Build the tool eval from your traces, not from your imagination. Sample a few hundred real steps, label the right tool and the right arguments, and keep the labelled set under version control next to the tool definitions. Run it on every change to a description, a schema or a model. The post above describes the same loop: prototype, evaluate, read the transcripts, refine the descriptions, repeat, with held-out cases so the descriptions do not overfit the eval.

### Discussion

1. **One tool per endpoint or one tool per task?** Endpoint wrappers are quick to generate and map to documentation you already have; task-shaped tools need design and an eval. My view: generate the wrappers to learn the domain, then replace the ones on the hot path with consolidated, task-shaped tools, and measure the trajectory length drop.
2. **How many tools is too many?** There is no fixed number; the symptom is selection accuracy falling and deliberation rising. My view: if a human engineer cannot say which tool to use in a given situation without thinking, the model cannot either; past a dozen or two, load tools per step rather than all at once.
3. **MCP or in-house?** MCP buys reach and costs control. My view: use MCP for everything you do not own and would otherwise integrate badly; write your own tools for the five that decide whether your agent works, and tune their descriptions on your eval.
4. **Should tools be forgiving or strict?** A forgiving tool (accepts "next Monday") raises step success now; a strict one (requires an ISO date) raises it permanently by making the ambiguity visible. My view: strict schemas with helpful error messages; the model learns from the error on the next turn, and the trace shows you where the schema should change.
5. **Who approves a destructive tool?** The annotation says destructive; something must act on it. My view: the loop, not the model, checks annotations and routes destructive calls to the approval step of Chapter 1's level 2; never trust the model to ask permission on its own.

## 2.4 Instructions: the system prompt as architecture

The system prompt is usually written last, by whoever is closest to the demo, and then grows by accretion as each incident adds a sentence. Treated that way it becomes the least reliable block in the system. Treated as architecture, it is the cheapest place to raise per-step reliability, because every word in it is read on every turn.

> [!DEFINITION] System prompt (instructions)
> The fixed text at the start of the context that tells the model what it is, what it is for, what it must and must not do, how to use its tools, what format to answer in, and when to stop. Unlike the history, it does not change during a run, and unlike the tool definitions, it is written by you.

### The anatomy

{{FIG:ch2_instructions|The six sections of an agent's system prompt, each with an example line and the production smell that appears when it is missing: role and goal (the agent does adjacent tasks nobody asked for), hard constraints (the guardrail lives only in people's heads), tool guidance (the right tool called in the wrong order), output format (a parser that breaks every third run), stop conditions (loops that never end) and canonical examples (edge cases listed as rules instead of shown).}}

**Role and goal.** One or two sentences: what the agent is, who it serves, and what finishing looks like. The goal matters more than the role; "your job ends when the invitation has been sent" prevents more drift than three paragraphs of persona.

**Hard constraints.** The rules that must hold whatever the user says: never book before 10:00, never email outside the company, ask before deleting. These are the first line of guardrails (Chapter 9), and they belong in the prompt *and* in code, because the prompt is a request and code is a guarantee.

**Tool guidance.** When to use which tool and in what order; when to use none. The tool descriptions say what each tool does; the prompt says how they fit together for this task ("find free slots before creating an event; prefer one filtered search to many broad ones").

**Output format.** What the final answer must look like, so that the code that reads it never guesses. For an agent, the final answer is often a structured object, and the format section is a schema.

**Stop conditions.** When the agent is done, how many steps it may take, and what to do when a tool fails repeatedly. Chapter 1 showed "unaware of termination conditions" as one of the most common multi-agent failures; the fix starts here.

> [!DEFINITION] Stop condition
> The rule that ends a run: the goal is reached (the invitation is sent), a budget is exhausted (eight steps, a token limit, a timeout), or a failure is unrecoverable (a tool failed twice). Written in the prompt so the model knows when to say it is done, and enforced in the loop so it stops even if the model does not.

**Canonical examples.** Two or three short transcripts of a good run, chosen for diversity rather than coverage.

> [!DEFINITION] Few-shot examples
> Worked examples inside the prompt that show the model the expected behaviour on representative inputs instead of describing it. A handful of diverse, canonical examples teaches more than a long list of rules, and costs tokens on every turn, so they are chosen and kept short.

### What the vendors advise

OpenAI's practical guide to building agents (April 2025) devotes a page to instructions and lists four practices.

> [!PAPER] OpenAI, "A practical guide to building agents" (PDF guide, April 2025) · page 11
> [![Page 11 of the guide, Configuring instructions: high-quality instructions are essential for any LLM-powered app, but especially critical for agents; clear instructions reduce ambiguity and improve agent decision-making; four best practices, use existing documents, prompt agents to break down tasks, define clear actions, capture edge cases, each with a paragraph of explanation](/img/agents/ch2-openai-instructions.png)](/img/agents/ch2-openai-instructions.png)
>
> **Context:** the "Configuring instructions" page, after the guide has named model, tools and instructions as the three core components of an agent.
>
> **What it says:** "clear instructions reduce ambiguity and improve agent decision-making". Use existing documents (operating procedures, support scripts, policies) as the source; break dense material into smaller steps; make every step correspond to a specific action or output; and capture edge cases such as missing information with conditional steps.
>
> **Why it matters:** the first practice is the one teams skip. Your organisation almost certainly already has a document that says how this task is done for humans; the prompt should be a translation of it, not a fresh invention. The fourth practice is where prompts grow fat, and the next source says how to keep them from doing so.

Anthropic's September 2025 post on context engineering describes the balance to strike.

> [!PAPER] Anthropic, "Effective context engineering for AI agents" (Engineering blog, September 2025)
> [![A paragraph from the post: system prompts should be extremely clear and use simple, direct language that presents ideas at the right altitude for the agent; the right altitude is the Goldilocks zone between two common failure modes; at one extreme, engineers hardcoding complex, brittle logic in their prompts to elicit exact agentic behavior, which creates fragility and increases maintenance complexity over time; at the other extreme, vague, high-level guidance that fails to give the LLM concrete signals or falsely assumes shared context; the optimal altitude is specific enough to guide behavior effectively, yet flexible enough to provide the model with strong heuristics](/img/agents/ch2-anthropic-ctx-altitude.png)](/img/agents/ch2-anthropic-ctx-altitude.png)
>
> **Context:** the section "The anatomy of effective context", on system prompts. The post recommends organising prompts into distinct sections with headers or tags, starting with "a minimal prompt with the best model available", and adding instructions and examples only "based on failure modes found during initial testing". On examples it warns against stuffing "a laundry list of edge cases into a prompt" and recommends "a set of diverse, canonical examples".
>
> **What it says:** write at "the right altitude": not "complex, brittle logic" that hard-codes behaviour, not "vague, high-level guidance" that assumes shared context, but "specific enough to guide behavior effectively, yet flexible enough to provide the model with strong heuristics".
>
> **Why it matters:** the two failure modes are the two ways prompts die in production. The brittle prompt breaks on the first case its author did not foresee; the vague prompt never worked and nobody could say why. "Minimal does not necessarily mean short", the post adds; it means every sentence is there because a failure needed it.

### Long or short instructions

| | Long, detailed instructions | Short, minimal instructions |
|---|---|---|
| Pros | more control; edge cases handled; less dependence on the model's defaults | fewer tokens on every turn; fewer internal contradictions; easier to test and reason about; lets a capable model use its judgement |
| Cons | tokens paid on every step; rules conflict with each other and with tool descriptions; later rules get lost in the middle (Section 2.7); nobody knows which sentence does what | relies on the model doing the sensible thing; edge cases surface in production |
| When to pick it | a regulated task with hard rules; a weaker model that needs them; a stable task whose edge cases are known | early, always; a capable model; a task where the right action depends on judgement |
| How to tell it was right | the eval passes *and* removing any paragraph makes it fail (otherwise the paragraph is dead weight) | the eval passes; the failure analysis finds no cluster that a sentence would fix |
| Production failure | a new rule added after an incident silently contradicts an old one; cost creep | a long tail of unhandled cases, each handled by a hot-fix sentence until the prompt is long anyway |

### Prompt versioning and testing as code

> [!DEFINITION] Prompt versioning
> Keeping every version of the system prompt (and the tool descriptions, which are prompts too) in version control, with the eval results that justified each change, and deploying prompts through the same release process as code. A prompt edit is a behaviour change; it deserves a diff, a review and a test run.

The practical rules are the ones you already use for code. Store the prompt as a file, not a string buried in a function. Review changes as diffs. Run the eval set in CI on every change, and gate the release on it (Chapter 7 builds that gate). Log the prompt version in every trace, so a regression can be traced to the edit that caused it. And remove sentences as readily as you add them: the eval that justified a sentence should be re-run without it from time to time, because models change and the sentence may now be noise.

### Instruction smells

| Smell | Example | What it causes | Fix |
|---|---|---|---|
| Rules that contradict | "always confirm before booking" and "complete the task without asking" | random behaviour per run | one rule, with the exception stated |
| Hidden stop | no sentence says when the task is done | the agent keeps improving its answer until the budget runs out | an explicit done condition and a step budget |
| Tool guidance in the wrong place | the prompt describes what `find_free_slots` returns | the description and the prompt drift apart | describe the tool in the tool; sequence the tools in the prompt |
| Edge cases as rules | twenty "if the user says X then Y" lines | brittle; the twenty-first case breaks it | three canonical examples that show the judgement |
| Shouting | "NEVER EVER", "CRITICAL", "YOU MUST" | emphasis inflation; everything is critical so nothing is | plain sentences; the hard constraints also enforced in code |
| Negatives without alternatives | "do not use markdown" | the model does not know what to do instead | "reply in plain sentences" |
| Persona over purpose | three paragraphs of character, one line of goal | drift and verbosity | one line of role, a clear goal, a clear finish |
| Untested edits | a sentence added after an incident, no eval run | fixes one case, breaks three | prompt in version control, eval in CI |

### Discussion

1. **How much should the prompt know about the tools?** Too little and the model uses them in the wrong order; too much and the prompt duplicates the descriptions and drifts. My view: tool *descriptions* say what; the prompt says *when and in what order*, in one short section.
2. **Should hard constraints be in the prompt if they are also in code?** Both. The prompt tells the model what to avoid so it does not waste steps proposing it; the code stops it when the prompt fails. My view: never only in the prompt.
3. **Who edits the prompt?** Product people understand the task; engineers understand the loop. My view: anyone can propose, in a pull request, with the eval run attached; nobody edits in production.
4. **Can we trust a vendor's prompting guide?** Their advice is tuned for their models and changes with each release. My view: adopt the structure (sections, altitude, canonical examples), and test the specifics on your eval.

## 2.5 Memory: what the agent remembers and for how long

A model remembers nothing between calls. Everything an agent "remembers" is engineering: something your code kept and put back in front of the model. The design questions are *what* to keep, *where*, and *for how long*, and the right answers differ by kind of information.

### Three places memory lives

{{FIG:ch2_memory_ladder|Left: the three places memory lives. The context window is seen directly by the model and paid for on every turn; the working memory is a scratchpad the agent writes for itself during a run; the long-term store lives outside the context, across runs, and is written to and retrieved from. Right: the three kinds of memory, episodic (what happened), semantic (facts) and procedural (how to), each with an example.}}

> [!DEFINITION] Short-term memory (the context window)
> What the model sees on this turn: instructions, tool definitions, the history of the run and the newest observation. It is the only memory the model reads directly, it costs tokens on every call, and it is gone when the run ends. Chapter 1 called it the context; here it is the top of the ladder.

> [!DEFINITION] Working memory (scratchpad)
> Notes the agent writes for itself during a run: the plan, the candidates found so far, the problem just hit. Kept small and structured, re-inserted into the context each turn, and preserved when the history is compacted. The structured note-taking of the Anthropic post below, and the "running plan" of plan-and-execute agents in Chapter 3.

> [!DEFINITION] Long-term memory (a store)
> Information kept outside the context, across runs, and retrieved when relevant: a key-value store, a vector index over text, files in a directory, a database. The model reaches it through tools (`remember`, `recall`) or your code retrieves into the context before the call. Cheap to hold; expensive when the wrong thing is retrieved or the right thing is not.

### Three kinds of thing remembered

The second axis comes from cognitive science and has been adopted by agent frameworks because it tells you what to store where.

> [!DEFINITION] Episodic memory
> Memory of *what happened*: specific events, with their time and outcome. "On Monday the invitation bounced because Tom was out." Past runs, traces, the reflections of Reflexion (Chapter 1). Useful for not repeating a mistake; stale quickly; privacy-sensitive because it is a record of real interactions.

> [!DEFINITION] Semantic memory
> Memory of *facts*: about the user, the domain, the world. "Priya is in Dublin." "This user never meets before 10:00." A profile, a knowledge base, retrieved documents. Useful across many runs; needs a way to be corrected when the fact changes.

> [!DEFINITION] Procedural memory
> Memory of *how to do things*: rules, recipes, the order in which tools should be used. "Always check the room before booking." The instructions block is procedural memory written by you; learned procedural memory is the agent writing its own rules from experience, which is powerful and dangerous in equal measure.

| Memory type | Cost | Staleness | Privacy | Retrieval errors | Typical store |
|---|---|---|---|---|---|
| Short-term (context) | highest: paid on every turn, grows every step | none: it is the present | contained in the run | none: the model sees all of it, though attention fades (Section 2.7) | the message list |
| Working (scratchpad) | low: a few hundred tokens per turn | low: rewritten by the agent as it goes | contained in the run | low: small and structured | a notes field, a plan object, a notes file |
| Long-term episodic | storage cheap; retrieval has a token cost per item | high: events age; outcomes get superseded | highest: records of real people's interactions, with retention obligations | retrieving an irrelevant or outdated episode misleads the agent | vector store over summaries, logs with embeddings |
| Long-term semantic | storage cheap; retrieval modest | medium: facts change and must be corrected, not appended | medium: user profiles are personal data | retrieving a contradicted fact; two versions of the same fact | key-value by entity, a profile table, a knowledge base |
| Long-term procedural | low: a few rules | low to medium: rules outlive the reason for them | low | a learned rule that was wrong propagates to every run | the prompt; a rules file the agent may edit under review |

### The research: a memory stream, and an operating system

The Generative Agents paper (April 2023) is where the modern memory architecture for agents was written down.

> [!PAPER] Park et al. (2023), Generative Agents · Figure 5 · page 8
> [![Figure 5 of the Generative Agents paper: the generative agent architecture; Perceive feeds a Memory Stream inside a box labelled Generative Agent Memory; Retrieve produces Retrieved Memories which lead to Act; arrows from Retrieved Memories loop back through Plan and Reflect into the Memory Stream](/img/agents/ch2-genagents-figure5.png)](/img/agents/ch2-genagents-figure5.png)
>
> **Context:** Section 4 of arXiv 2304.03442, "Generative agent architecture".
>
> **What it says:** "all perceptions are saved in a comprehensive record of the agent's experiences called the memory stream"; the architecture "retrieves relevant memories and uses those retrieved actions to determine an action"; retrieved memories are "also used to form longer-term plans and create higher-level reflections, both of which are entered into the memory stream".
>
> **Why it matters:** this is the memory ladder as a loop. The stream is long-term episodic memory; retrieval selects a few records into the context (short-term); reflection turns many episodes into a few semantic facts ("Klaus is dedicated to his research"); planning writes procedural memory for the day. Every production memory system is a subset of this figure.

The paper's retrieval function is a small formula worth knowing, because every "which memories should go into the prompt" decision ends up looking like it.

> [!PAPER] Park et al. (2023), Generative Agents · Section 4.1 · page 9
> [![A paragraph from the paper: to calculate the final retrieval score, we normalise the recency, relevance and importance scores to the range zero to one using min-max scaling; the retrieval function scores all memories as a weighted combination of the three elements, score equals alpha recency times recency plus alpha importance times importance plus alpha relevance times relevance; in our implementation all alphas are set to one; the top-ranked memories that fit within the language model's context window are included in the prompt](/img/agents/ch2-genagents-retrieval.png)](/img/agents/ch2-genagents-retrieval.png)
>
> **Context:** the "Retrieval" part of Section 4.1. Recency is "an exponential decay function over the number of sandbox game hours since the memory was last retrieved" with a decay factor of 0.995; importance is a score from 1 to 10 that the model assigns when the memory is stored; relevance is the embedding similarity between the memory and the current situation.
>
> **What it says:** the score is "a weighted combination of the three elements", with all weights set to 1, and "the top-ranked memories that fit within the language model's context window are included in the prompt".
>
> **Why it matters:** three signals, not one. Pure similarity search (the default of most retrieval systems) would surface an old, trivial memory that happens to share words with the query; recency and importance keep it out. Notice also the last clause: the budget is the context window, and the ranking exists to spend it well.

> [!DEFINITION] Retrieval score
> The number a memory system assigns to each stored item to decide which items go into the context for the current step. In Generative Agents it is the sum of recency (decayed since last use), importance (assigned at storage) and relevance (similarity to the situation), each scaled to 0 to 1. Your weights and signals may differ; the shape, several signals combined and a budget, is what to copy.

{{FIG:ch2_retrieval_score|The retrieval function of Generative Agents on four illustrative memories. Each gets a recency score, an importance score assigned when it was stored, and a relevance score to the current situation; the paper sums them with equal weights, and the top-ranked memories that fit in the context go into the prompt. The two highlighted memories are selected; the recent but trivial ones are not. Values are illustrative.}}

Six months later, MemGPT (October 2023) gave the problem its most useful analogy.

> [!PAPER] Packer et al. (2023), MemGPT · Abstract · page 1
> [![Abstract of the MemGPT paper with highlighted phrases: virtual context management; paging between physical memory and disk; intelligently manages different storage tiers](/img/agents/ch2-memgpt-abstract.png)](/img/agents/ch2-memgpt-abstract.png)
>
> **Context:** the abstract, arXiv 2310.08560, posted 12 October 2023. The paper's title is "MemGPT: Towards LLMs as Operating Systems"; the authors later built the Letta framework around the idea.
>
> **What it says:** "virtual context management, a technique drawing inspiration from hierarchical memory systems in traditional operating systems which provide the illusion of an extended virtual memory via paging between physical memory and disk"; the system "intelligently manages different storage tiers".
>
> **Why it matters:** an operating system gives every program the illusion of more memory than the machine has by moving pages between RAM and disk. MemGPT gives the model the illusion of a longer context by moving information between the context window and external storage. The twist is who does the moving: the model itself, through tool calls.

> [!PAPER] Packer et al. (2023), MemGPT · Figures 1 and 2 · page 2
> [![Figures 1 and 2 of the MemGPT paper: on the left, a chat in which the user mentions a birthday and a boyfriend named James, a system alert reads Memory Pressure, and the model calls working_context.append with Birthday is February 7 and Boyfriend named James; on the right, a later chat in which the model calls recall_storage.search for six flags, receives three dated results, and replies with a detail from the earlier conversation](/img/agents/ch2-memgpt-figure1.png)](/img/agents/ch2-memgpt-figure1.png)
>
> **Context:** the two example figures on page 2. Figure 1 is captioned "MemGPT (left) writes data to persistent memory after it receives a system alert about limited context space"; Figure 2 shows it searching out-of-context data to bring relevant information into the current context.
>
> **What it says:** when the context nears its limit, the loop sends a "memory pressure" alert; the model responds by calling `working_context.append(...)` to save the facts worth keeping; later, prompted by a question, it calls `recall_storage.search(...)` and the results arrive as an observation.
>
> **Why it matters:** memory management becomes ordinary tool use. The two function calls are tools like any other, with the same design rules from Section 2.3, and the alert is a stop-condition-like signal from the loop. The model decides *what* is worth remembering, which is both the power of the design (it knows what mattered) and its risk (it may be wrong, and nobody reviews it).

{{FIG:ch2_memgpt|The operating-system analogy of MemGPT. The fixed-size context window is main memory, holding the system instructions, a working context of facts the model chose to keep, and a queue of recent messages; recall storage (the full searchable history) and archival storage (filed documents and notes) are disk. The model pages information in and out by calling memory functions when the loop warns it of memory pressure.}}

### Memory in production: compaction, notes, sub-agents

Anthropic's context-engineering post names three techniques its own long-running agents use, and they map exactly onto the ladder.

> [!PAPER] Anthropic, "Effective context engineering for AI agents" (Engineering blog, September 2025)
> [![A paragraph from the post: compaction is the practice of taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary; compaction typically serves as the first lever in context engineering to drive better long-term coherence; at its core, compaction distils the contents of a context window in a high-fidelity manner, enabling the agent to continue with minimal performance degradation](/img/agents/ch2-anthropic-ctx-compaction.png)](/img/agents/ch2-anthropic-ctx-compaction.png)
>
> **Context:** the section "Context engineering for long-horizon tasks". The post describes its coding agent passing the history to the model to summarise, preserving "architectural decisions, unresolved bugs, and implementation details while discarding redundant tool outputs", then continuing with the summary "plus the five most recently accessed files". It recommends tuning the compaction prompt for recall first, then precision, and names "tool result clearing" as the safest first step.
>
> **What it says:** compaction is "taking a conversation nearing the context window limit, summarizing its contents, and reinitiating a new context window with the summary", and it is "the first lever".
>
> **Why it matters:** compaction is lossy by design, and the loss is chosen by a prompt. The post's own warning, that "overly aggressive compaction can result in the loss of subtle but critical context whose importance only becomes apparent later", is the memory failure mode to test for: a fact stated at step 2 that is needed at step 20.

> [!DEFINITION] Compaction
> Replacing the older part of the history with a shorter summary so the context stays within budget, keeping the most recent turns verbatim. The summary is usually written by the model from a prompt that says what to preserve. The cheapest form is clearing old tool results; the riskiest is summarising the user's own words.

> [!PAPER] Anthropic, "Effective context engineering for AI agents" (Engineering blog, September 2025)
> [![A paragraph from the post: structured note-taking, or agentic memory, is a technique where the agent regularly writes notes persisted to memory outside of the context window; these notes get pulled back into the context window at later times](/img/agents/ch2-anthropic-ctx-notes.png)](/img/agents/ch2-anthropic-ctx-notes.png)
>
> **Context:** the same section. The examples are a to-do list the coding agent keeps, a `NOTES.md` file a custom agent maintains, and a game-playing agent that kept "precise tallies across thousands of game steps" and, after context resets, "reads its own notes and continues multi-hour training sequences".
>
> **What it says:** the agent "regularly writes notes persisted to memory outside of the context window", and these "get pulled back into the context window at later times".
>
> **Why it matters:** this is working memory made durable: the scratchpad is a file, so it survives compaction and even a restart. It is also the simplest long-term memory that works: a text file the agent reads at the start and edits as it goes, with no vector store and no retrieval function. Start here before building anything cleverer.

> [!PAPER] Anthropic, "Effective context engineering for AI agents" (Engineering blog, September 2025)
> [![A paragraph from the post: sub-agent architectures provide another way around context limitations; rather than one agent attempting to maintain state across an entire project, specialized sub-agents can handle focused tasks with clean context windows; the main agent coordinates with a high-level plan while subagents perform deep technical work or use tools to find relevant information; each subagent might explore extensively, using tens of thousands of tokens or more, but returns only a condensed, distilled summary of its work, often 1,000 to 2,000 tokens](/img/agents/ch2-anthropic-ctx-subagents.png)](/img/agents/ch2-anthropic-ctx-subagents.png)
>
> **Context:** the third technique, with a pointer to the multi-agent research system of Chapter 1. The post closes the section with a rule of thumb: compaction for tasks with "extensive back-and-forth", note-taking for "iterative development with clear milestones", multi-agent for "complex research and analysis where parallel exploration pays dividends".
>
> **What it says:** sub-agents "handle focused tasks with clean context windows" and each "returns only a condensed, distilled summary of its work (often 1,000-2,000 tokens)".
>
> **Why it matters:** a sub-agent is memory isolation. The lead agent never sees the forty thousand tokens of searching; it sees a two-thousand-token result. That is compaction done by delegation, and Chapter 5 weighs what it costs in coordination.

### A toy memory in fifty lines

The script `ch2_memory.py` implements the ladder in miniature: a history (short-term), a scratchpad (working), a key-value store (long-term), and a **summariser stub** that compacts the history every four steps by keeping the first ninety characters of each old message (a real system would ask the model to summarise). It replays twelve steps of the meeting task and prints the approximate context size per step, with and without compaction. Token counts use the rule of thumb of about four characters per token; they are an approximation, not a tokenizer.

```python
# code/agents/ch2_memory.py (abridged; the summariser is a stub and the token count is chars // 4)
class Memory:
    def __init__(self, compact_every=4, keep_last=2):
        self.history, self.scratch, self.store = [], [], {}  # short-term, working, long-term

    def remember(self, key, value): self.store[key] = value             # long-term: survives compaction
    def recall(self, query): return {k: v for k, v in self.store.items() if any(w in k for w in query.split())}
    def note(self, line): self.scratch.append(line)                      # working memory: the running plan

    def add(self, role, content, step):
        self.history.append({'role': role, 'content': content})
        if step % self.compact_every == 0:                               # compaction: old turns -> one summary
            old, recent = self.history[:-self.keep_last], self.history[-self.keep_last:]
            summary = ' '.join(m['content'][:90] for m in old if m['role'] != 'summary')
            self.history = [{'role': 'summary', 'content': summary}] + recent

    def context(self, query):                                            # what the model would see this turn
        return '\n'.join(['SYSTEM: ...'] + [f'MEMORY {k}: {v}' for k, v in self.recall(query).items()]
                         + ['PLAN: ' + ' | '.join(self.scratch)] + [f'{m["role"]}: {m["content"]}' for m in self.history])
```

[![Terminal output of ch2_memory.py: without compaction the context grows from 79 to 489 approximate tokens over twelve steps, 3,660 in total; with compaction every four steps it peaks at 349 and ends at 296, 2,772 in total; then the compacted context at step 12 showing the system line, two MEMORY lines including the preference never before 10:00, a PLAN line of notes, a SUMMARY line, and the last two turns; finally recall of the word meetings returns the stored preference, with the note that the user turn stating the rule was summarised away at step 4 but survives in the long-term store](/img/agents/ch2_memory-run.png)](/img/agents/ch2_memory-run.png)

{{FIG:ch2_memory_sizes|Approximate context size per step of the twelve-step toy run, with and without compaction every four steps. The compacted run drops at steps 4, 8 and 12 and ends about forty percent smaller; over twelve steps it sends about a quarter fewer tokens. The numbers are from the four-characters-per-token approximation in ch2_memory.py.}}

Two things to notice. The saving is modest at twelve steps (3,660 against 2,772 tokens) because the run is short; Section 2.7 shows what happens at thirty. More important is the last two lines of the output: the user's rule "never before 10:00" was stated at step 1, compacted away at step 4, and is still available at step 12 *because it was also written to the long-term store*. Without that write, the agent would have booked Tue 09:00 at step 10 with a clear conscience. The summariser kept the first ninety characters of each message, which happened to include the rule in this run; a real summariser might not. Memory that matters must not depend on what a summary happens to keep.

### How to evaluate memory

| Metric | What it measures | How |
|---|---|---|
| Recall after N turns | can the agent still use a fact stated N turns ago? | plant facts at step k, ask at step k+N, score the answer; sweep N past the compaction boundary |
| Contradiction rate | does the agent act against a stored fact or against something it said earlier? | a checker over the trace (booked before 10:00 when the store says never) |
| Retrieval precision | of the memories put into the context, how many were used or relevant? | label retrieved items; or ablate them and see whether the answer changes |
| Cost of context per step | what does memory add to each call? | tokens of retrieved memory and notes per step, from the trace |
| Staleness | how often does a retrieved fact turn out to be superseded? | compare retrieved facts to the current truth where you have it |

> [!WARNING]
> Memory failures look like intelligence failures. An agent that books the 9 am slot "forgot" the rule; the trace shows the rule was compacted out two steps earlier. Before you blame the model, check what was in the context at the step that went wrong (Chapter 8 makes this a one-click question).

### Discussion

1. **What is worth storing long-term?** Everything is cheap to store and expensive to retrieve wrongly. My view: store semantic facts about the user by key, episodic summaries only where a past outcome changes a future decision, and procedural rules only under human review.
2. **Should the model decide what to remember, or should code?** MemGPT lets the model decide; a profile table lets code decide. My view: let the model propose, and have code (or a reviewer) accept, for anything that will influence future runs for other people.
3. **How aggressively should we compact?** Aggressive compaction saves money and loses facts. My view: tune for recall first (plant facts and check they survive), then trim; clear tool results before you summarise anything a user said.
4. **What do we owe the user about their memory?** Episodic and semantic memory are records of a person. My view: show them what is stored, let them delete it, and set retention before launch, because the first deletion request will come after.
5. **Is a vector store the default?** It is the default in tutorials, not in production. My view: a key-value profile and a notes file cover most agents; add similarity search when you have many unstructured memories and a measured retrieval problem.

## 2.6 State: where the loop is

Memory is what the agent knows. **State** is where it is: which step, what it planned to do, which call is in flight, how much budget is left, what has gone wrong. Confusing the two produces agents that cannot be resumed, retried or inspected. Keeping state explicitly produces agents that can.

> [!DEFINITION] State (of a run)
> The small object that describes the control position of one run: the current step, the plan and its progress, the pending tool call if any, the budget remaining (steps, tokens, time, money), the errors so far and the status (running, waiting for approval, needs retry, done, failed). It is owned by the loop, not by the model, and it is what a checkpoint saves.

> [!DEFINITION] Checkpoint
> A saved copy of the state (and usually the history) at a well-defined point, typically after every step, so that a run can be resumed from the last checkpoint after a crash, a timeout, a deploy or a human approval, instead of starting again.

{{FIG:ch2_state|The state object of a meeting-scheduling run over four steps: the current step, the plan with progress marks, the pending tool call, the budget left, the errors so far and the status. Each step ends with a checkpoint, so a crashed run resumes from the last one instead of starting again; the pending call is recorded so a retry is deliberate, and the email tool must be idempotent or the second attempt invites everyone twice.}}

Why does the pending call matter? Suppose the process dies between sending the email and recording the result. On resume, the state says "pending: send_email". The loop now has a choice: run it again (safe only if the tool is idempotent), check whether it happened (if the tool offers a way), or ask a person. Without the pending field, the loop does not even know there is a question. Chapter 1's account of Anthropic's research system mentioned exactly this: the team added "the ability to resume from where an error occurred rather than restart", and used deployments that let in-flight agents finish on the old version. Both are state engineering.

### The state machine view

An agent loop can be drawn as a graph: nodes are steps (call the model, run a tool, ask for approval, summarise), edges are transitions, and the state object travels along the edges. Frameworks such as LangGraph make this explicit, and attach persistence to it.

> [!DEFINITION] State machine (graph) agent
> An agent whose possible steps and transitions are declared as a graph in code, with the state object passed from node to node, as opposed to a free-running loop that calls the model and whatever tool it names until it stops. The model still chooses among the allowed transitions; the graph fixes which transitions exist.

> [!PAPER] LangChain, "Persistence" (LangGraph documentation, accessed October 2026)
> [![Two bullet points from the LangGraph persistence page: checkpointers persist a thread's graph state as checkpoints, use them for short-term, thread-scoped memory, including conversation continuity, human-in-the-loop workflows, time travel, and fault tolerance; stores persist application-defined data outside the graph state, use them for long-term, cross-thread memory, including user preferences, facts, and shared knowledge](/img/agents/ch2-langgraph-persistence.png)](/img/agents/ch2-langgraph-persistence.png)
>
> **Context:** the overview of the persistence layer, which the page summarises as giving agents "short-term memory through checkpointers and long-term memory through stores". The page's troubleshooting section warns that in-memory checkpointers are lost on restart and that checkpoints "accumulate" over long conversations and should be pruned.
>
> **What it says:** checkpointers "persist a thread's graph state as checkpoints" for "conversation continuity, human-in-the-loop workflows, time travel, and fault tolerance"; stores "persist application-defined data outside the graph state" for "user preferences, facts, and shared knowledge".
>
> **Why it matters:** the framework draws exactly this chapter's line. The checkpointer is state (plus the history it carries): per run, for resuming and for pausing at an approval. The store is long-term memory: across runs, for facts. The four uses listed for checkpoints are the four reasons to keep state at all.

| | Free-running loop | Explicit state machine (graph) |
|---|---|---|
| Pros | simplest possible code; the model has full freedom; new tools need no graph change | every possible transition is visible and testable; approval steps, retries and compaction are nodes, not special cases; checkpointing falls out of the design |
| Cons | hard to resume; hard to insert an approval step; the only budget is a step counter; behaviour is wherever the model took it | more code and more concepts; the graph can over-constrain a capable model; frameworks bring their own abstractions and upgrades |
| When to pick it | prototypes; short runs (a handful of steps) where restarting is cheap | runs long enough to crash or to need approval; regulated actions; anything that must resume |
| How to tell it was right | runs finish within budget and nobody needs to resume one | resumed runs complete at the same rate as fresh ones; approval steps are hit exactly when the graph says; no duplicate side effects after retries |
| Production failure | a deploy kills in-flight runs; a timeout retries a write; nobody can say what step a stuck run is on | the graph does not allow the step the task needed; checkpoints grow unbounded; two versions of the graph disagree about a checkpoint's shape |

> [!TIP] In production
> Checkpoint after every step from the first prototype, even to a local file. Record the pending call *before* running it and clear it *after* recording the result. Make every write tool idempotent or give it an idempotency key. Log the state (step, budget left, status) on every trace line. These four habits turn "the agent got stuck" from a mystery into a query.

### Discussion

1. **Where does the plan live, memory or state?** It is both: the plan's *content* is working memory, its *progress* is state. My view: keep the plan text in the scratchpad and the progress marks in the state object, so compaction never erases where you are.
2. **Framework or your own loop?** A graph framework gives you checkpointing and approval nodes on day one and a dependency on day two. My view: write the free loop first to understand the task, then adopt a graph (your own or a framework's) the moment you need to resume or to pause for a human.
3. **How much should the model see of the state?** Showing it the budget ("3 steps left") improves stopping; showing it the whole state invites it to reason about plumbing. My view: show the budget and the errors, hide the rest.
4. **Retry or ask?** After a crash with a pending write, retrying is fast and asking is safe. My view: retry only idempotent calls automatically; route everything else to a person, and make the trace show which happened.

## 2.7 Context engineering: the art of what to put in the window

Every block in this chapter ends up as tokens in one place: the context the model reads on this turn. The model sees nothing else. It does not see your database, your notes file, your tool implementations or your intentions; it sees the window. **Context engineering** is deciding what goes into that window at each step, and it is the discipline that ties the five blocks together.

> [!DEFINITION] Context engineering
> Choosing, for every call, which tokens the model will see: which instructions, which tool definitions, which memories, how much history, in what order and at what length. Anthropic's post defines it as "the set of strategies for curating and maintaining the optimal set of tokens (information) during LLM inference". Prompt engineering writes one text; context engineering manages a changing set of texts over a run.

### The budget

{{FIG:ch2_context_budget|The context of one turn as a budget bar at step 10 of a run, using the planning numbers in ch2_math.py: 1,500 tokens of instructions, 3,000 of tool definitions (twenty tools at about 150 each), 1,000 of retrieved memory, 10,800 of history and 800 for the current observation, 17,100 in all. The history already dominates and grows every step; the fixed parts are paid for on every call.}}

The numbers in the figure are planning numbers, not measurements: the point is the shape. At step 10 the history is already 63% of the turn, and it grows by about 1,200 tokens per step (a model turn of about 400 tokens plus an observation of about 800). The instructions and tool definitions, 4,500 tokens here, are paid on every single call whether or not the step needs them. The retrieved memory is small only because somebody made it so.

> [!DEFINITION] Token budget (per turn)
> The number of tokens one call may use, set below the model's window and well below it in practice. It is divided among the fixed parts (instructions, tool definitions), the retrieved parts (memory, documents) and the growing part (history). When the growing part reaches the budget, something must be summarised, cleared or dropped.

### Why the window's hard limit is not the real limit

Models with very long windows exist, and the temptation is to treat the limit as the budget. Two findings say otherwise. The first is from Stanford and Berkeley, in July 2023.

> [!PAPER] Liu et al. (2023), Lost in the Middle · Figure 1 · page 1
> [![Figure 1 of the Lost in the Middle paper: accuracy of a model on multi-document question answering with twenty retrieved documents, plotted against the position of the document containing the answer; accuracy is about 75 when the answer is in the first document, falls to the mid fifties when it is in the middle, and rises to the low sixties when it is last; a dashed line shows the closed-book baseline just above 55; the caption describes a U-shaped performance curve with a primacy bias and a recency bias](/img/agents/ch2-litm-figure1.png)](/img/agents/ch2-litm-figure1.png)
>
> **Context:** arXiv 2307.03172, published in TACL 2024. The task is to answer a question from twenty retrieved documents (about four thousand tokens), one of which contains the answer; the experiment moves that document from first to last.
>
> **What it says:** performance is "a U-shaped performance curve": models "are better at using relevant information that occurs at the very beginning (primacy bias) or end of its input context (recency bias), and performance degrades significantly when models must access and use information located in the middle". In the figure, accuracy in the middle positions falls to about the level of the model answering with no documents at all.
>
> **Why it matters:** the agent's history is the middle of its context. The user's constraint from step 1 sits after the instructions and before twenty tool results; by step 20 it is exactly where this curve says the model reads worst. That is why important facts are re-stated in the scratchpad near the end of the context, and why compaction that keeps the right facts can *improve* accuracy, not only cost. Newer models degrade more gently than the 2023 ones in the figure, but no public result shows the effect gone.

> [!DEFINITION] Lost in the middle
> The finding that models use information at the start and end of a long context better than information in the middle. For an agent it means the order of the context is a design decision: instructions first, the live plan and the newest observation last, and the middle kept short.

The second finding is the one Anthropic's post calls context rot: as the number of tokens in the window increases, the model's ability to recall information from it decreases, so that context "must be treated as a finite resource with diminishing marginal returns". The post's explanation is architectural: attention relates every token to every other, and models have seen far more short sequences than long ones in training, so precision at long range is a "performance gradient rather than a hard cliff".

> [!DEFINITION] Context rot
> The gradual loss of a model's ability to find and use information as its context grows, even well within the window's limit. It is why the practical budget is set by attention, not by the window, and why a tool result of twenty thousand tokens costs more than its token price.

### The arithmetic of a thirty-step run

The script `ch2_math.py` runs the budget forward for thirty steps, with and without compaction every ten steps (the history replaced by a 600-token summary plus the last two turns), at the same example price as Chapter 1: $3 per million input tokens and $15 per million output tokens, round numbers for the arithmetic and not a quote from any provider.

[![Terminal output of ch2_math.py: part one, the budget of one turn at step 10 with instructions 1,500, tool definitions 3,000, retrieved memory 1,000, history 10,800 and observation 800, total 17,100; part two, a table of input tokens per step with and without compaction, 5,500 at step 1 for both, 16,300 at step 10, then 17,500 against 8,500 at step 11 and 40,300 against 19,300 at step 30; totals of 687,000 against 387,000 input tokens, 12,000 output tokens, cost 2.241 against 1.341 dollars, 44 percent fewer input tokens and 40 percent cheaper; part three, a 25,000-token tool result at step 5 re-sent on every later step adds 650,000 input tokens and 1.95 dollars per run, against 20,800 tokens if trimmed to 800](/img/agents/ch2_math-run.png)](/img/agents/ch2_math-run.png)

{{FIG:ch2_compaction_cost|Input tokens sent on each step of a thirty-step run. Without compaction the context grows linearly to 40,300 tokens at step 30 and the run sends 687,000 input tokens in all; with compaction every ten steps the context never exceeds 19,300 and the run sends 387,000. At the example price the run costs $2.24 against $1.34, about forty percent less.}}

| | Without compaction | Compaction every 10 steps |
|---|---|---|
| Context at step 1 | 5,500 | 5,500 |
| Context at step 11 | 17,500 | 8,500 |
| Context at step 30 | 40,300 | 19,300 |
| Input tokens over the run | 687,000 | 387,000 |
| Output tokens over the run | 12,000 | 12,000 |
| Cost at the example price | $2.24 | $1.34 |

Three readings of the table. First, compaction cut input tokens by 44% and cost by 40% in a thirty-step run, and the saving grows with length because the uncompacted cost is quadratic in the number of steps (each step re-sends everything before it). Second, the largest context halved, from 40,300 to 19,300, which matters for attention as much as for money. Third, the output tokens did not change: compaction is about what the model reads, not what it writes.

The third part of the script is the single most useful number in this chapter for a tool designer. One tool that returns 25,000 tokens at step 5 (a log dump, a full table, a whole document) and is then carried in the history for the remaining 26 steps adds 650,000 input tokens to the run, almost as much as the entire uncompacted run without it, and about $1.95 at the example price. The same tool trimmed to 800 tokens of relevant lines adds 20,800. That is the token-efficiency principle of Section 2.3 with a price on it, and it is why a response cap is a cost control.

### Compaction and summarisation: how to do it without losing the plot

Compaction is the first lever and the one most likely to do quiet damage. The rules that keep it safe:

1. **Clear tool results first.** Once a tool result has been read and acted on, the raw payload rarely matters again; the decision it led to is in the next model turn. Clearing old tool results is nearly lossless and often the only compaction you need.
2. **Keep the user's words verbatim as long as possible.** Constraints and goals come from the user; a summary of them is a paraphrase by the model, and the paraphrase is where "never before 10:00" becomes "prefers mornings".
3. **Write the things that must survive into working memory before compacting.** The scratchpad is re-inserted whole after compaction; the history is not.
4. **Give the summariser a recall-first prompt and test it on real traces.** Plant facts, compact, ask. Only trim the prompt once nothing planted goes missing.
5. **Keep the last few turns verbatim.** Recency bias is your friend here: the newest observation and the model's last thought are where the next decision comes from.
6. **Checkpoint before compacting.** If the summary turns out to have dropped something, the checkpoint lets you rebuild from the full history.

> [!TIP] In production
> Log, for every step, the size of each part of the context: instructions, tool definitions, memory, history, observation. Plot it over the run. A step where the observation is ten times the median is a tool that needs a cap; a run where the history crosses your budget before the task is half done is a run that needs compaction or a sub-agent; a fixed part that is a third of every turn is a tool set or a prompt that should be loaded per step. You cannot engineer a budget you cannot see.

### Discussion

1. **How big should the budget be?** Below the window by a wide margin, and set by measured accuracy, not by the vendor's maximum. My view: find the context length at which your per-step eval starts to slip, and set the budget below it.
2. **Compaction, notes or sub-agents?** All three work; they fit different tasks. My view (following the post): compaction for conversational tasks, notes for milestone-driven work, sub-agents for parallel exploration; and tool-result clearing in every case.
3. **Should retrieved documents go at the start or the end?** The U-curve says both ends are read well and the middle is not. My view: instructions at the start, retrieved material and the live plan near the end, just before the newest observation; and test the order on your eval, because models differ.
4. **Who pays for the tool definitions?** They are paid on every turn by every run. My view: load per step when the catalogue is large, and treat definition tokens as a line item on the cost dashboard.

## 2.8 Putting the blocks together: the meeting example

One task, traced block by block. The request: "Set up a 45-minute design review with Priya and Tom next week. I prefer mornings but never before 10:00. Send the invitations."

{{FIG:ch2_meeting_trace|The meeting-scheduling run traced block by block over five steps: what the instructions contributed, what memory recalled or stored, what the model decided, which tool ran and what the state recorded. At step 4 the invitation to Tom bounces; at step 5 the agent recovers, and the 10:00 rule, by then compacted out of the history, is recovered from long-term memory.}}

| Step | Instructions | Memory | Model | Tools | State |
|---|---|---|---|---|---|
| 1 | role: scheduler; constraint: not before 10:00; stop: when invitations are sent | recalls the user's stored preference for mornings; writes the goal to the scratchpad; stores the 10:00 rule | reads the request, writes a plan: find slots, check a room, book, invite | none yet | step 1 of 8, plan set, budget 8 steps |
| 2 | tool guidance: find slots before booking | notes the candidate slots at or after 10:00 | calls `calendar_find_free_slots` with `earliest_hour=10` | returns 14 shared slots in a bounded list | pending call recorded, then cleared with the result |
| 3 | output format: event id in the summary | notes room 4B | calls `find_rooms`, then `calendar_create_event` with an idempotency key | event EVT-1187 created for Monday 10:00 | step 3 of 8; event id stored in state |
| 4 | constraint: only company addresses | stores "Tom out on Monday" as a fact | reads the tool result: invitations sent, but an auto-reply says Tom is out Monday; decides to move | `email_send` returns ok plus the bounce text in the result (not a silent success) | errors: 1; status: needs retry |
| 5 | stop: when the invitations are sent | recalls the 10:00 rule from the store (the user's sentence was compacted at step 4) | chooses Wednesday 10:00 from the remaining slots; calls `update_event` | event moved, updated invitations sent | step 5 of 8; done; final checkpoint |

Every step used all five blocks, and the two moments where the run could have gone wrong were both block failures averted by design. At step 4 a tool that returned "sent" without the bounce text would have ended the run with Tom absent: a tools failure (success on failure). At step 5 an agent that relied on the history alone would have had no 10:00 rule, because compaction had summarised the user's sentence away: a memory failure. The code in `ch2_memory.py` shows the second case literally.

### What would break

| If this block were weak... | Symptom in the run | Where the trace shows it | The fix lives in |
|---|---|---|---|
| Missing memory (no store, compaction only) | books Tuesday 09:00 at step 5 with a clear conscience | the context at step 5 has no 10:00 rule | write constraints to the long-term store at step 1; re-insert the scratchpad after compaction |
| Bad tool description (`do_calendar`: "Calendar stuff.") | calls the wrong calendar tool, or the right one with a free-text date | tool-selection error; argument validation failure | namespaced names, when-to-use descriptions, strict schemas |
| Huge tool return (`find_free_slots` returns every calendar entry) | step 2 observation of twenty thousand tokens; the plan from step 1 is "lost in the middle" by step 4 | observation size spike; later contradictions | a cap and filters in the tool; tool-result clearing |
| Vague instruction (no stop condition) | after the invitations are sent, the agent "improves" the event, adds an agenda, emails again | steps continue after the goal; budget exhausted | an explicit done condition in the prompt and the loop |
| No state (free loop, no checkpoint) | the process restarts after step 3 and books a second event, then sends duplicate invitations | two event ids in the side-effect log; no record of the pending call | checkpoint after every step; record pending calls; idempotency keys |
| Weak model (routine model on the hard step) | at step 4 it reads the bounce as success, or retries the same email | the reasoning line ignores the auto-reply text | route recovery steps to the large model; make the tool put the error first in its result |

Notice what the table does not say: "use a bigger model" appears once, as the last row, and only after five rows that a bigger model would not have fixed.

## Exercises

1. Your agent has 40 tools from four MCP servers, and tool-selection accuracy on your eval fell from 94% to 81% when the fourth server was added. List three remedies (namespacing, per-step tool loading, consolidation), the pros and cons of each, and the measurement that would tell you which one worked.
2. Design the `send_email` tool for the meeting agent as a full definition: name, description, schema, return shape, error messages, idempotency and annotations. Then write the three eval cases you would use to test it, including one where the right answer is not to call it.
3. A team proposes a router: a small model for every step, escalating to a large one when the small model's own confidence is below a threshold. Argue both sides: what does this save, what can go silently wrong, and what per-step eval would you insist on before shipping it?
4. Your compaction prompt summarises the history every 15 steps. Plant five facts at steps 2, 5, 8, 11 and 14 and design the questions you would ask at step 30 to measure recall. Which facts would you expect to survive, which would you move to the long-term store or the scratchpad, and why?
5. Re-run the thirty-step arithmetic of Section 2.7 for your own agent: your instruction length, your tool count, your observed observation size and your provider's prices. At what step does the uncompacted context cross your practical budget, and what is the cost of the single largest tool result in your traces re-sent for the rest of the run?

## Key takeaways

- An agent is assembled from five blocks: the **model** reasons, the **tools** act, the **instructions** shape, the **memory** remembers and the **state** says where the loop is. Each has its own failure modes and its own eval; when a run fails, find the block before you change anything.
- Choose the model by measurement on your own tasks: tool-call accuracy, argument accuracy, refusal rate, latency p50 and p95, cost per task, parse-failure rate. Public leaderboards shortlist; your labelled set decides. Even the best models miss a quarter of the public benchmark's cases.
- A **router** (small model for routine steps, large for hard ones) can cut cost and latency, at the price of two models to evaluate and a silent failure mode when the route is wrong. Build the per-step eval set first.
- A **tool** is a name, a description and a JSON schema; the model emits a structured call, your code runs it and returns the result as a message. Design tools as interfaces: namespaced names, when-to-use descriptions, unambiguous arguments, bounded returns, errors that say what to do, idempotent writes, declared side effects. Consolidate endpoint wrappers into task-shaped tools.
- **MCP** standardises how tools, resources and prompts are exposed to a host; it buys reach and costs control. Loading every tool definition into every turn does not scale; load what the step needs.
- The **system prompt** is architecture: role and goal, hard constraints, tool guidance, output format, stop conditions and a few canonical examples, written at the right altitude, versioned and tested as code.
- **Memory** lives in three places (the context, a scratchpad, a store) and holds three kinds of thing (episodic, semantic, procedural). Generative Agents gave the retrieval formula (recency, importance, relevance); MemGPT gave the paging analogy; production systems use compaction, structured notes and sub-agents.
- **State** is control, not knowledge: step, plan progress, pending call, budget, errors, status. Checkpoint it after every step, record pending calls, and make writes idempotent, or a retry will do the task twice.
- The model sees only the **context**. Budget it: fixed parts are paid every turn, the history grows every step, and attention fades in the middle long before the window is full. In the thirty-step example, compaction every ten steps cut input tokens by 44% and cost by 40%; one uncapped 25,000-token tool result cost almost as much as the whole run.
- In the meeting example, the two near-failures were a tool that could have hidden an error and a memory that could have lost a constraint. "Use a bigger model" fixed neither.

## References

**Papers**

1. Shishir G. Patil, Tianjun Zhang, Xin Wang, Joseph E. Gonzalez. [*Gorilla: Large Language Model Connected with Massive APIs*](https://arxiv.org/abs/2305.15334). NeurIPS 2024 (arXiv May 2023).
2. Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, Thomas Scialom. [*Toolformer: Language Models Can Teach Themselves to Use Tools*](https://arxiv.org/abs/2302.04761). NeurIPS 2023 (arXiv February 2023).
3. Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, Maosong Sun. [*ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs*](https://arxiv.org/abs/2307.16789). ICLR 2024 (arXiv July 2023).
4. Joon Sung Park, Joseph C. O'Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein. [*Generative Agents: Interactive Simulacra of Human Behavior*](https://arxiv.org/abs/2304.03442). UIST 2023 (arXiv April 2023).
5. Charles Packer, Sarah Wooders, Kevin Lin, Vivian Fang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez. [*MemGPT: Towards LLMs as Operating Systems*](https://arxiv.org/abs/2310.08560). arXiv October 2023.
6. Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, Percy Liang. [*Lost in the Middle: How Language Models Use Long Contexts*](https://arxiv.org/abs/2307.03172). Transactions of the ACL, volume 12, 2024 (arXiv July 2023).

**Engineering blogs and docs**

7. Fanjia Yan, Huanzhi Mao, Charlie Cheng-Jie Ji, Tianjun Zhang, Shishir G. Patil, Ion Stoica, Joseph E. Gonzalez. [*Berkeley Function-Calling Leaderboard*](https://gorilla.cs.berkeley.edu/blogs/8_berkeley_function_calling_leaderboard.html). Gorilla blog, UC Berkeley, last updated 19 August 2024; and the [live leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (V4, updated 12 April 2026 at the time of writing).
8. Anthropic (Ken Aizawa and colleagues). [*Writing effective tools for agents, with agents*](https://www.anthropic.com/engineering/writing-tools-for-agents). Engineering blog, 11 September 2025.
9. Anthropic. [*Effective context engineering for AI agents*](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents). Engineering blog, 29 September 2025.
10. Anthropic. [*Code execution with MCP: Building more efficient agents*](https://www.anthropic.com/engineering/code-execution-with-mcp). Engineering blog, 4 November 2025.
11. Anthropic. [*Building effective agents*](https://www.anthropic.com/research/building-effective-agents), Appendix 2, "Prompt engineering your tools". Research blog, 19 December 2024.
12. OpenAI. [*A practical guide to building agents*](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf), "Configuring instructions", page 11. PDF guide, April 2025.
13. Model Context Protocol. [*Architecture overview*](https://modelcontextprotocol.io/docs/learn/architecture). modelcontextprotocol.io documentation, protocol version 2026-07-28.
14. LangChain. [*Persistence*](https://docs.langchain.com/oss/python/langgraph/persistence). LangGraph documentation, accessed October 2026.

**Code for this chapter**

15. [`code/agents/ch2_tools.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch2_tools.py): one function-calling round trip with a real JSON schema and a stubbed model, and the tool-selection comparison on ten labelled requests; outputs in `results/ch2_tools_stdout.txt` and `results/ch2_tools_select_stdout.txt`.
16. [`code/agents/ch2_memory.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch2_memory.py): the fifty-line memory with a scratchpad, a summariser stub and a key-value store, with context sizes per step; results in `results/ch2_memory.json`.
17. [`code/agents/ch2_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch2_math.py): the context budget, the thirty-step run with and without compaction, and the cost of one oversized tool result; results in `results/ch2_math.json`.
18. [`code/agents/figs_ch2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/figs_ch2.py) and [`code/agents/shots_ch2.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/shots_ch2.py): the figures and the paper excerpts.

## Next

→ [Chapter 3: Single-agent reasoning patterns](../part2-patterns/03-reasoning-patterns.md)

With the blocks in hand, Chapter 3 asks how a single agent should *think* between tool calls. It reads ReAct closely (the interleaving of thought, action and observation that Chapter 1 introduced), then plan-and-execute (write the whole plan first, then carry it out), reflection and self-critique (Reflexion's verbal memory of failures), and tree search over possible next steps, and for each pattern it asks the questions this chapter asked of the blocks: what it costs in tokens and steps, when it is the right choice, what the papers measured, and how to tell from a trace that it is working.
