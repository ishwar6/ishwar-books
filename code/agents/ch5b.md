## 5.4 Agents that argue: debate and voting

### The example: three analysts and one landed cost

Northwind's buyers now need one number per candidate: the landed cost of a casserole dish, after currency, duty and a share of the freight. One slip in that sum moves a supplier up or down the ranking. A careful team would ask three analysts to work it out. Then it would either take the most common answer or put the three in a room until they agree.

Those are the two ideas of this section. **Voting**, which Chapter 4 measured, samples several answers independently and takes the most frequent one. **Debate** lets the agents read each other's answers and revise their own over a few rounds. Debate sounds better, because people who talk catch each other's mistakes. It also costs more per call, and agents can agree too easily. So we ask the buyer's question: for the same number of model calls, which gives more right answers?

> [!DEFINITION] Multi-agent debate
> Several copies of a model (the agents) first answer a question independently. In each later round, every agent reads the other agents' latest answers and writes an updated answer. At the end, the final answer is usually a plurality vote over the agents' last answers. The agents often share one model and one prompt; they differ only in what they have read.

> [!DEFINITION] Equal-compute comparison
> A comparison in which two methods get the same budget, usually the same number of model calls or tokens. Three agents that answer and then debate once use six calls. The fair baseline is then a vote over six independent samples, not one call and not a vote of three.

### Where the idea came from

In 1986 Marvin Minsky published *The Society of Mind*. The book describes a mind as a society of many small agents, none of them clever alone. It was about human minds, but the name stuck. In May 2023 a team from MIT and Google Brain used it for a concrete method: run several instances of one model and make them argue.

> [!PAPER] Du et al. (2023), Multiagent Debate · Figure 2 · page 3
> [![Figure 2 of the multiagent debate paper: a question about 175 diamonds, 35 fewer rubies and twice as many emeralds as rubies; in round 1 agent 1 answers 225 (wrong) and agent 2 answers 595 (right); in round 2 each agent receives the other's reply, agent 1 changes to 560 (still wrong) and agent 2 keeps 595; in round 3 both agree on 595](/img/agents/ch5b-du-figure2.png)](/img/agents/ch5b-du-figure2.png)
>
> **What it says:** after the first answers, each agent receives "Using the solutions from other agents as additional information, can you give an updated response" with the other replies copied in. The wrong agent moves twice and ends at 595.
>
> **What it means:**
>
> The mechanism is copy and paste. Nothing is trained and no agent has a special role. Agent 1 made an algebra slip, agent 2's working showed it, and after two rounds agent 1 adopted the right answer.
>
> Notice what this needs. One agent must be right at the start, and its working must be clear enough to persuade. If both had made the same slip, there would be nothing to copy.

The paper ran three agents for two rounds of debate on arithmetic, school maths, chess and biographies, with one chat model. It compared debate with one agent, one agent that reflects on its answer, and a majority vote across agents.

> [!PAPER] Du et al. (2023), Multiagent Debate · Table 1 · page 6
> [![Table 1 of the paper: on arithmetic, single agent 67.0 percent, single agent with reflection 72.1, multi-agent majority 69.0, multi-agent debate 81.8; on grade school maths 77.0, 75.0, 81.0 and 85.0; on chess, change in pawn score 91.4, 102.1, 102.2 and 122.9; multi-agent results use 3 agents and two rounds of debate](/img/agents/ch5b-du-table1.png)](/img/agents/ch5b-du-table1.png)
>
> **What it says:** debate scores 81.8% on arithmetic against 69.0% for a majority vote and 67.0% for one agent. On school maths it scores 85.0% against 81.0% and 77.0%.
>
> **What it means:** this table started the field. Read the two middle rows with one question: how many calls did each spend? Three agents over two rounds spend six or nine calls, depending on whether the first answers count as a round. The table does not show that the vote spent as many, so the comparison may not be at equal compute. Later papers tested exactly that.

The paper also reports two behaviours that matter for every multi-agent system.

> [!PAPER] Du et al. (2023), Multiagent Debate · Section 2.2 · page 4
> [![A paragraph from the paper: prompts that encouraged models to be more stubborn based on their own solutions led to longer debates and better final solutions; overall, we observed that language model agents were relatively agreeable, perhaps as a result of instruction tuning or reinforcement learning based on human feedback](/img/agents/ch5b-du-agreeable.png)](/img/agents/ch5b-du-agreeable.png)
>
> **What it says:** "more 'stubborn'" prompts "led to longer debates and better final solutions"; the agents "were relatively 'agreeable'".
>
> **What it means:** a debate is only as good as its disagreement. If every agent gives way at the first objection, the first confident voice wins. The authors suspect the training that makes chat models polite: a model tuned to please a user also pleases a peer. A jury that always agrees with the loudest juror is one juror.

> [!PAPER] Du et al. (2023), Multiagent Debate · Limitations · page 10
> [![A paragraph from the limitations section: while debates typically converged into single final answers, these answers were not necessarily correct; despite answers being incorrect, language models would confidently affirm that their answer is correct and consistent with all other agent responses](/img/agents/ch5b-du-limits.png)](/img/agents/ch5b-du-limits.png)
>
> **What it says:** converged answers "were not necessarily correct", and the models "would confidently affirm" them.
>
> **What it means:** agreement is not evidence. From outside, three agents agreeing on a wrong answer look exactly like three agreeing on a right one. Measure agreement against labelled answers before you trust it.

### Debate against the reflection trap

A second paper of May 2023 came from Tsinghua, Shanghai Jiao Tong and Tencent AI Lab. It started from self-reflection, which asks a model to critique and fix its own answer (Chapter 3).

> [!PAPER] Liang et al. (2023), Encouraging Divergent Thinking through Multi-Agent Debate · Section 1 · page 1
> [![A paragraph defining Degeneration-of-Thought: once the LLM-based agent has established confidence in its answers, it is unable to generate novel thoughts later through self-reflection even if the initial stance is incorrect](/img/agents/ch5b-liang-dot.png)](/img/agents/ch5b-liang-dot.png)
>
> **What it says:** "Degeneration-of-Thought": once confident, a model is "unable to generate novel thoughts later through self-reflection even if the initial stance is incorrect".
>
> **What it means:**
>
> A student checking their own exam reads what they meant to write, not what they wrote. A model reviewing its own answer starts from that answer and tends to defend it. Chapter 3 met the same finding from Huang et al.: without outside feedback, self-correction of reasoning often fails.
>
> A second agent is outside feedback. It did not write the answer, so it has nothing to defend. Our experiment below tests this with a self-review control.

> [!PAPER] Liang et al. (2023), Encouraging Divergent Thinking through Multi-Agent Debate · Figure 2 · page 3
> [![Figure 2 of the paper: a question about how many times circle A revolves as it rolls around circle B three times its radius; chain of thought answers 3, wrong; self-reflection repeats that it is confident the answer is correct; in multi-agent debate an angel and a devil argue, one says to consider both the rotation around circle B and the rotation of circle A itself, and a judge concludes the answer is 4](/img/agents/ch5b-liang-figure2.png)](/img/agents/ch5b-liang-figure2.png)
>
> **What it says:** two debaters, one told to oppose the other ("tit for tat"), and a judge who ends the debate. Self-reflection stays at 3; the debate reaches 4.
>
> **What it means:** the design forces disagreement through the prompt, and a separate judge decides when to stop. The paper found that forcing extra rounds hurt translation quality. It also found that a judge preferred the debater built on its own model when two different models argued.

> [!PAPER] Liang et al. (2023), Encouraging Divergent Thinking through Multi-Agent Debate · Table 3 · page 5
> [![Table 3 of the paper: accuracy on counter-intuitive arithmetic reasoning; GPT-4 51.0 percent; GPT-3.5-Turbo 26.0; with chain of thought 28.0; with self-consistency 29.5; with self-reflect 27.5; with multi-agent debate 37.0](/img/agents/ch5b-liang-table3.png)](/img/agents/ch5b-liang-table3.png)
>
> **What it says:** on trick arithmetic questions, debate lifts GPT-3.5-Turbo from 26.0% to 37.0%. Self-consistency reaches 29.5%, self-reflection 27.5%, and GPT-4 alone 51.0%.
>
> **What it means:** where the first instinct is usually wrong, an opponent helps more than a vote. A vote can only count the instinct; an opponent can attack it. Debate still closed less than half the gap to the stronger model.

### The later evidence: at equal compute, voting is hard to beat

By 2024 three groups asked the equal-compute question directly. The first was InstaDeep in London, at ICML 2024.

> [!PAPER] Smit et al. (2024), Should we be going MAD? · Abstract · page 1
> [![The abstract of the paper with highlighted phrases: multi-agent debating systems, in their current form, do not reliably outperform other proposed prompting strategies, such as self-consistency and ensembling using multiple reasoning paths; MAD protocols are more sensitive to different hyperparameter settings and difficult to optimize](/img/agents/ch5b-smit-abstract.png)](/img/agents/ch5b-smit-abstract.png)
>
> **What it says:** debate systems "do not reliably outperform" self-consistency and ensembles, and are "more sensitive to different hyperparameter settings". Tuning how much the agents agree made one protocol the best.
>
> **What it means:** a method that wins only after careful tuning is a research result, not a default. For a production team, "sensitive" means a debate that worked last month can stop working after a model upgrade.

> [!PAPER] Smit et al. (2024), Should we be going MAD? · Figure 1 · page 3
> [![Figure 1 of the paper: medical question accuracy against total cost for many configurations of Society of Mind, Ensemble Refinement, ChatEval, Self-Consistency, Single Agent, Multi-Persona and Medprompt; dot size shows API calls; most configurations sit between 0.57 and 0.64 accuracy; Medprompt, which does not use debate, has the highest median at lower cost](/img/agents/ch5b-smit-figure1.png)](/img/agents/ch5b-smit-figure1.png)
>
> **What it says:** on medical questions with GPT-3.5, Medprompt, which "does not use debate", performed best overall and cost less. "Society of Mind" in the legend is Du et al.'s method.
>
> **What it means:** look at the spread inside each colour. Configurations of one method differ by about as much as methods differ from each other. With noise that large, one published comparison says little about your task.

Qineng Wang and colleagues at Zhejiang University, HKUST and UIUC (ACL 2024) used a different control: one agent with a good prompt.

> [!PAPER] Wang et al. (2024), Are Multi-Agent Discussions the Key? · Abstract · page 1
> [![The abstract with highlighted phrases: a single-agent LLM with strong prompts can achieve almost the same performance as the best existing discussion approach; multi-agent discussion performs better than a single agent only when there is no demonstration in the prompt](/img/agents/ch5b-wang-abstract.png)](/img/agents/ch5b-wang-abstract.png)
>
> **What it says:** "a single-agent LLM with strong prompts" almost matches the best discussion method; discussion wins "only when there is no demonstration in the prompt".
>
> **What it means:** a worked example in the prompt gives one agent what a discussion would give it, a second view of the problem. If debate mostly repairs a weak prompt, the cheaper fix is a better prompt.

The most direct test came from the University of Wisconsin-Madison at NeurIPS 2025. Hyeong Kyu Choi, Xiaojin Zhu and Sharon Li split debate into its two parts, several agents and the talking, and measured each.

> [!PAPER] Choi, Zhu and Li (2025), Debate or Vote · Abstract · page 1
> [![The abstract with highlighted phrases: Majority Voting alone accounts for most of the performance gains typically attributed to multi-agent debate; we prove that debate induces a martingale over agents' belief trajectories, implying that debate alone does not improve expected correctness](/img/agents/ch5b-choi-abstract.png)](/img/agents/ch5b-choi-abstract.png)
>
> **What it says:** "Majority Voting alone accounts for most of the performance gains"; debate "induces a martingale", so "debate alone does not improve expected correctness".
>
> **What it means:**
>
> A martingale is a fair game: your expected wealth after the next bet equals your wealth now. The paper models each agent's belief in the right answer this way. Reading peers moves the belief up or down at random, and on average it stays where it started.
>
> What does rise is agreement, and the vote at the end does the useful work. The proof assumes simple, identical agents, so read it as an explanation of the average. The paper also shows that updates biased toward correct answers do help.

> [!PAPER] Choi, Zhu and Li (2025), Debate or Vote · Figures 1 and 2 · page 2
> [![Figures 1 and 2 of the paper: on the left, majority voting sends a question to three agents and combines their answers, while debate passes answers between agents for T rounds first; on the right, bars of accuracy for a single agent, the best debate and majority voting on seven benchmarks, where majority voting is as high as the best debate on most of them](/img/agents/ch5b-choi-figure2.png)](/img/agents/ch5b-choi-figure2.png)
>
> **What it says:** on seven benchmarks with 7B and 8B open models and five agents, the majority-voting bar matches or beats the best debate on most tasks.
>
> **What it means:** this vote uses only the agents' first answers, so it spends fewer calls than the debate. Voting matched debate while spending less, which is the strongest form of the result.

### Our experiment: debate against voting at the same number of calls

The research gives a default: vote first. It also gives two hints about when debate might still help. Liang suggests it helps when single answers are often wrong, and Wang when one agent's prompt is weak. `ch5b_debate.py` tests both hints at equal calls, with a self-review control.

The task is thirty word problems from Northwind's buying desk. They cover tiered prices, stock with a damaged delivery, customs dates around a holiday, landed cost with shared freight, and packers with a break. Each takes four to seven steps. `ch5b_make_data.py` computes every gold answer in Python. The model is gpt-5-mini with minimal reasoning effort, so the working we allow in the answer is most of the thinking it gets.

Two working budgets set the difficulty. **Terse** allows one short sentence before the answer; **brief** allows 40 words of key working. For each problem we make 21 calls:

- 9 independent samples; votes of 3, 6 and 9 use the first 3, 6 or 9.
- A debate of three agents whose round 0 is the first three samples. In each of two rounds every agent reads the other two answers and answers again (6 calls).
- A **self-review** control with the same shape: each agent re-checks only its own answer for two rounds (6 calls).

Every call has the same word budget, so debaters get no extra room to think. Every multi-call method ends in a plurality vote.

{{FIG:b_debate_shapes|Three ways to spend six model calls on one question. A vote counts six independent answers. A debate lets three agents read each other's answers before a vote. The control lets each agent re-check only its own answer. Round 0 is shared, so the comparison is paired problem by problem.}}

```python
# code/agents/ch5b_debate.py (abridged): one debate round. The control is the same loop with REVIEW, which shows only the agent's own answer.
for r in range(2):
    new, newt = [], []
    for i in range(3):
        others = '\n'.join(f'Agent {j + 1}: {rtexts[-1][j]}' for j in range(3) if j != i)
        t, u = llm.ask(DEBATE.format(budget=b, q=q['q'], mine=rtexts[-1][i], others=others), max_tokens=900)
        new.append(answer(t)); newt.append(t)                    # same word budget as the voters
    rounds.append(new); rtexts.append(newt)
row['debate1'] = right(plurality(rounds[1]), gold)              # 6 calls in total, compared with a vote of 6
```

The run made 1,260 calls and cost 41 US cents, from the API's usage fields.

[![Terminal output of the summary of ch5b_debate.py: with the terse budget, one call 42 percent, vote of 3 43, vote of 6 53, debate with 6 calls 73, self-review with 6 calls 67, vote of 9 60, debate with 9 calls 83, self-review with 9 calls 43; with the brief budget 74, 87, 90, 83, 90, 90, 90 and 87; debate used 1.65 and 1.88 times the tokens of the vote but about the same money; answer changes and unanimity counts for both budgets; 1,260 calls for 0.410 dollars](/img/agents/ch5b_debate_summary-run.png)](/img/agents/ch5b_debate_summary-run.png)

{{FIG:b_debate_results|Results of the final run of ch5b_debate.py. With terse answers, debate beat a vote of the same size: 73% against 53% at six calls, 83% against 60% at nine. With 40 words of working, every multi-call method scored 83% to 90% and debate gained nothing.}}

**With terse answers, debate beat voting at equal calls.** Debate was right on 73% of problems at six calls, the vote on 53%. At nine calls the scores were 83% and 60%. Problem by problem, debate alone was right 6 times and the vote alone never (six calls), and 8 against 1 at nine calls. A sign test gives p = 0.031 and p = 0.039. That is unlikely to be noise, but thirty problems cannot pin down the size of the gap.

[![Terminal output of ch5b_debate_stats.py: for the terse budget, debate against a vote of 6, debate alone right on 6 problems and the vote on 0, sign test p 0.031; at 9 calls 8 against 1, p 0.039; debate against self-review 5 against 3 at 6 calls, p 0.727, and 15 against 3 at 9 calls, p 0.008; debate answer changes 44 wrong to right and 13 right to wrong; self-review 23 and 24; for the brief budget no difference between methods](/img/agents/ch5b_debate_stats-run.png)](/img/agents/ch5b_debate_stats-run.png)

**The gain came from reading other agents, not from a second look.** The self-review agents had the same calls and budget. Over two rounds they fixed 23 answers and broke 24, a coin toss, and after two rounds their vote fell back to 43%. The debaters fixed 44 answers and broke 13. This is Liang's degeneration of thought in one line of numbers. A model re-reading its own work moves at random; a model reading a different answer moves toward the right one more often.

{{FIG:b_flips|How individual answers changed between rounds with the terse budget, 180 updates per method. Debate moved 44 answers from wrong to right and 13 from right to wrong; self-review moved 23 and 24.}}

**With 40 words of working, debate added nothing.** Every multi-call method scored between 83% and 90%. Fifteen of thirty problems were already unanimous at round 0, so there was little to argue about. This matches Choi et al. and Wang et al.: once each agent can do the work, the vote does what debate does.

**Why did terse debate help?** With terse answers, the three agents disagreed on 22 of 30 problems at round 0. The right answer was often in the minority. A vote counts a minority answer as a minority. A debater reads another agent's line of working, recomputes and changes its mind. On the seven problems where only one agent was right at the start, the debate never ended wrong. Here the other answers acted as a cheap verifier: they showed each agent where to look.

**The cost was almost the same.** Debate calls read the other answers, so they used 1.65 times the tokens of the vote (terse) and 1.88 times (brief). But input tokens cost an eighth of output tokens at gpt-5-mini's list price, and debaters wrote shorter answers: 77 output tokens per call against 112 for voters. The six-call debate cost $0.0471 and the six-call vote $0.0470. A model with another price ratio, or longer contexts, changes this balance.

> [!WARNING]
> One run, one model at minimal reasoning effort, thirty synthetic problems and one debate prompt. The terse budget is an artificial way to make single answers weak. Debate also raised agreement: unanimous problems went from 8 to 18 (terse) and from 15 to 29 (brief). Of those, 2 and 3 were unanimous on a wrong answer. What this run supports: test debate against a vote of the same size, and a self-review control, on your own task.

### Debate and voting in practice

| | Vote (self-consistency) | Debate |
|---|---|---|
| Calls and latency | n independent calls, all in parallel; latency of one call | agents x rounds; one call of latency per round |
| When it helps | the right answer is usually the most common one | single answers are often wrong, and a peer's working can show the error |
| How it fails | a shared mistake wins the count | agents converge on a confident wrong answer; gains vanish after a model change |
| What to log | every sample and the agreement rate | every round, and which agents changed their answer |

> [!TIP] In production
> Default to a vote, run in parallel, and log the agreement rate as a confidence signal. Try debate only where single answers are weak and no verifier exists. Keep it only if it beats a vote of the same size on your labelled set. If a program can check the answer, use the check: Chapter 4's best-of-n with a verifier beats both.

### Discussion

1. **Is debate a multi-agent system at all?** The agents share one model, one prompt and no tools. My view: treat it as an aggregation method inside one step, not as an architecture.
2. **Why not add a judge, as Liang did?** A judge is one more decision someone must measure; Chapter 4's judge often disagreed with a simple word count. A plurality vote is free and has no bias toward its own model.
3. **Would different models make better debaters?** Different models make different mistakes, which a debate needs. They also make a fair judge harder. It is worth testing, not a safe default.

## 5.5 How multi-agent systems fail

### The example: a recommendation that lost its prices

Pass Northwind's supplier question along a line of five agents. A researcher reads the supplier pages, an analyst ranks the suppliers and a risk reviewer assesses them. A writer drafts the memo and an editor shortens it. Every agent does its own job well, and the memo that reaches the director contains no prices. Nobody decided to drop them. The risk reviewer thought prices were not its job, and later agents could only use what they received.

That is a failure of the system, not of one agent, and it is the most common kind. This section reads the study behind that claim, adds failures the study does not name, does the arithmetic, and then runs the line of five agents.

### MAST, read closely

Chapter 1 introduced MAST, the failure taxonomy of Mert Cemri, Melissa Pan, Shuyi Yang and colleagues at UC Berkeley. They collected 1,642 traces from seven open-source frameworks, including MetaGPT, ChatDev, AG2 and Magentic-One. Six experts built the taxonomy from 150 traces. Annotators applied it consistently (Cohen's kappa 0.88, where 1 is perfect agreement and 0 is chance), and a model annotator, checked against the experts, labelled the rest. MAST has 14 failure modes in three categories.

{{FIG:b_mast_map|The three MAST categories with their most common failure modes and their share of all failures, each paired with the control that targets it. The shares are the paper's; the pairing with controls is ours.}}

**System design issues (44.2%)** start before the run: an unclear task, roles that agents do not keep, and no rule for when to stop. Step repetition (15.7%) and being unaware of termination conditions (12.4%) are both loop failures. In one example, ChatDev was asked for a Wordle game without a fixed word list. It produced a fixed list anyway, even when the prompt said so explicitly.

> [!PAPER] Cemri et al. (2025), Why Do Multi-Agent LLM Systems Fail? · §4, FC2 · page 7
> [![A passage from the paper: inter-agent misalignment, failures arising from a breakdown in critical information flow; insight 2, solutions focused on context or communication protocols are often insufficient; failure modes conversation reset 2.20 percent, fail to ask for clarification 6.80 percent, task derailment 7.40 percent, information withholding 0.85 percent, ignored other agent's input 1.90 percent, reasoning-action mismatch 13.2 percent](/img/agents/ch5b-mast-fc2.png)](/img/agents/ch5b-mast-fc2.png)
>
> **What it says:** inter-agent misalignment is "a breakdown in critical information flow". Its largest mode is "mismatches between reasoning and action (FM-2.6, 13.2%)". Fixes to "context or communication protocols are often insufficient".
>
> **What it means:**
>
> A reasoning-action mismatch is an agent that says one thing and does another. For example, it concludes "the parser is broken" and then edits the README. In a team this is worse than in one agent, because the next agent reads the reasoning and assumes the action matched it.
>
> The warning about protocols matters for Section 5.3. MCP and A2A standardise the envelope of a message. They do not make an agent say what the next agent needs to know. The authors call this a failure of "theory of mind".

> [!PAPER] Cemri et al. (2025), Why Do Multi-Agent LLM Systems Fail? · Figure 3 · page 6
> [![Figure 3 of the paper: a supervisor agent asks a phone agent to log in with username a@mail.com; the phone agent reads the API docs, which say the username is the account phone number, but logs in with the email, gets invalid credentials, and asks the supervisor for the correct username and password without saying that the username should be the phone number](/img/agents/ch5b-mast-figure3.png)](/img/agents/ch5b-mast-figure3.png)
>
> **What it says:** "The Phone Agent fails to communicate API requirements (username format) to the Supervisor Agent, who also fails to seek clarification".
>
> **What it means:** each agent acted sensibly with what it saw. The phone agent read the documentation but did not pass on the one fact that mattered. The supervisor got an error but did not ask why. A structured result with a field for "requirements found" would have carried the fact.

**Task verification (23.5%)** is the last line of defence, and the paper found it weak.

> [!PAPER] Cemri et al. (2025), Why Do Multi-Agent LLM Systems Fail? · §4, FC3 · page 8
> [![A passage from the paper: task verification failures include premature termination 6.20 percent, no or incomplete verification 8.20 percent and incorrect verification 9.10 percent; systems with explicit verifiers show fewer failures but the presence of a verifier is not a silver bullet; a ChatDev chess program passes superficial checks such as compilation but contains runtime bugs; many verifiers perform only superficial checks; adding a high-level task objective verification step to ChatDev yields a 15.6 percent improvement in task success](/img/agents/ch5b-mast-fc3.png)](/img/agents/ch5b-mast-fc3.png)
>
> **What it says:** "the presence of a verifier is not a silver bullet"; many verifiers do "only superficial" checks, such as whether the code compiles. Checking the task objective gave "+15.6%" on ChatDev.
>
> **What it means:** a verifier that checks the format is a gate (Chapter 4). Gates answer "is this well-formed?", not "is this right?". The chess program compiled and did not follow the rules of chess. Verify against the task itself.

> [!PAPER] Cemri et al. (2025), Why Do Multi-Agent LLM Systems Fail? · Table 5 · page 27
> [![Table 5 of the paper: for AG2 on GSM-Plus with GPT-4, baseline 84.75, improved prompt 89.75, new topology 85.50; with GPT-4o 84.25, 89.00 and 88.83; for ChatDev on ProgramDev-v0, baseline 25.0, improved prompt 34.4, new topology 40.6; on HumanEval 89.6, 90.3 and 91.5](/img/agents/ch5b-mast-table5.png)](/img/agents/ch5b-mast-table5.png)
>
> **What it says:** clearer role prompts lifted ChatDev from 25.0% to 34.4% of programming tasks. A new topology, where the CTO agent must confirm all reviews before the run ends, reached 40.6%. On AG2 maths the gains were a few points.
>
> **What it means:** both fixes are cheap design changes, and both helped. Both also leave ChatDev failing most tasks. The authors conclude that such fixes are not universal, and that verification and structure must be designed in.

### Five more ways the system fails

Production write-ups add five shapes of failure that cut across MAST's categories.

**Conflicting decisions.** In Cognition's Flappy Bird story (Section 5.1), two subagents that cannot see each other build a background and a bird in different styles. "Actions carry implicit decisions": every choice a subagent makes is one the others did not see. Shared choices must be made by one agent, or written into the brief before the split.

**Error cascades.** A missing fact early in a chain is not one error. Every later agent builds on the gap, and none of them can see the source. Section 5.3 read Anthropic's fix for this "game of telephone": store results and pass a reference. Our experiment below measures the loss.

**Loops that do not stop.** Step repetition and unawareness of termination are together 28.1% of MAST's failures. CAMEL's agents thanked each other in a loop (Section 5.3).

**Cost blow-ups.** More agents means more contexts. In a group chat, every agent also re-reads what all the others said. Anthropic's early lead started "50 subagents for simple queries" (Section 5.3).

**Agents agreeing too easily.** Du et al. called their agents "agreeable". In our terse debate, unanimous problems rose from 8 to 18, and 2 of them were unanimous on a wrong answer. A reviewer agent that approves whatever it sees is the same failure in another role.

### The arithmetic: what compounds across handoffs

**Facts across a chain.** A photocopy of a photocopy is almost as good as the last one, and after ten copies the text is grey. Suppose each handoff keeps a given fact with probability $$q$$. After $$n$$ handoffs, the share of facts that survive is

$$S(n) = q^{\,n}.$$

$$S(n)$$ is the share still present after $$n$$ hops, and $$q$$ is the chance that one hop keeps one fact. **Assumptions:** hops lose facts independently, a lost fact never comes back, and all facts are equally fragile. With $$q = 0.9$$, five hops keep $$0.9^5 = 0.59$$: about 18 of 30 facts. With $$q = 0.99$$, about 29 survive. It is Chapter 4's chain formula $$r^n$$, applied to facts instead of steps.

**A verifier after every hop.** Add a checker that sees the source and detects a share $$d$$ of the facts a hop lost. If each detected loss is repaired, a fact survives a hop when it was kept, or when it was lost, detected and restored:

$$q' = q + (1 - q)\,d.$$

With $$q = 0.9$$ and $$d = 0.8$$, $$q' = 0.98$$, and five hops keep 90% instead of 59%. **Assumptions:** repairs always work, and the checker never removes a correct fact. Our experiment shows both assumptions failing in small ways.

**Tokens in a group chat.** Let $$N$$ agents speak in turn for $$R$$ rounds, so there are $$T = NR$$ turns. Each message is $$m$$ tokens and each turn reads $$p$$ tokens of instructions. If every turn re-reads all earlier messages, the run reads

$$T_{\text{chat}} = \sum_{t=1}^{T} \big(p + (t-1)\,m\big) = T\,p + m\,\frac{T(T-1)}{2}$$

input tokens. The first term is the instructions, once per turn. The second is the history: turn $$t$$ re-reads $$t-1$$ messages, and $$0 + 1 + \dots + (T-1) = T(T-1)/2$$. If each turn reads only a brief of about one message, $$T_{\text{brief}} = T\,(p + m)$$, a straight line. With $$m = 300$$, $$p = 500$$, eight agents and six rounds ($$T = 48$$), the chat reads $$24{,}000 + 338{,}400 = 362{,}400$$ tokens. The briefs read $$38{,}400$$, so the chat costs 9.4 times as much. **Assumptions:** no caching and no summarising, both of which help the chat in practice (Chapter 2).

{{FIG:b_groupchat|Input tokens over six rounds under the assumptions of ch5b_math.py, part C. A group chat in which every turn re-reads the conversation grows with the square of the number of turns; with eight agents it reads 9.4 times the tokens of a design in which each turn reads only its brief.}}

**A loop with no hard stop.** Suppose the agents decide to stop with probability $$s$$ on each turn. The mean number of turns is $$1/s$$, and a cap of $$K$$ turns gives

$$E[\text{turns}] = \frac{1 - (1-s)^{K}}{s},$$

where $$(1-s)^K$$ is the chance the agents are still talking after $$K$$ turns. With $$s = 0.1$$ the mean is 10 turns, or 8.8 with a cap of 20. The tail is what hurts: with $$s = 0.05$$, 35.8% of runs are still going after 20 turns. A prompt that says "stop when done" sets $$s$$; only a cap in code sets $$K$$.

[![Terminal output of ch5b_math.py: a table of q to the power n for q of 0.99, 0.95, 0.90 and 0.867 and n from 1 to 10; the verifier formula, where q 0.9 and d 0.8 give 0.98 and five hops go from 59 to 90 percent; input tokens of a group chat against briefs for 2 to 8 agents and 3 or 6 rounds, up to 362,400 against 38,400; expected turns of a loop with stop probability s and caps of 5, 10 and 20](/img/agents/ch5b_math-run.png)](/img/agents/ch5b_math-run.png)

{{FIG:b_compound|The share of facts that survive n independent handoffs, q to the power n. The dots mark q = 0.867, the per-hop rate implied by our free-text chain below.}}

### Our experiment: the telephone game on Northwind's suppliers

`ch5b_handoff.py` runs the line of five agents on six short supplier pages (`data/ch5/b/suppliers/`, a small set separate from Section 5.1's catalogue). The pages hold 30 gold facts. For each supplier these are the price per unit delivered to the UK, the delivery days to a UK warehouse, the minimum order and two risks. The pages also hold distractors: delivery days to an EU warehouse, and prices in euros, dollars or rupees.

Only the researcher reads the pages; every later agent sees only the previous handoff. There are four conditions, three runs each:

- **free**: each handoff is prose for the next colleague.
- **free + verifier**: after every hop, a verifier agent compares the handoff with the source pages and lists wrong or missing facts. The hop is redone once with that list.
- **schema**: each handoff is JSON. A supplier table with fixed fields is copied unchanged by later agents, and a "work" field holds each agent's own contribution.
- **schema + verifier**: both.

After every hop, a separate **probe** call reads only that hop's output and extracts what it states about each supplier. Code grades each fact as kept, lost or distorted. On the source pages themselves the probe found all 30 facts.

{{FIG:b_telephone|The handoff experiment. Six pages feed a chain of five agents. In the free condition each handoff is prose; in the schema condition the facts travel as a table each agent copies, beside its own work; with a verifier, a checker compares each hop's output with the source pages and the hop is redone once.}}

```python
# code/agents/ch5b_handoff.py (abridged): one hop. Only hop 0 sees the pages; the verifier always does.
given = f'SUPPLIER PAGES:\n{PAGES}' if h == 0 else f'HAND-OFF YOU RECEIVED:\n{prev}'
text, u = llm.ask(f'{TASK}\n\n{job}\n{fmt}\n\n{given}', max_tokens=4000)
if cond.endswith('+ver'):
    problems = (llm.parse_json(llm.ask(VERIFY.format(pages=PAGES, text=text))[0]) or {}).get('problems') or []
    if problems:                                                  # one redo with the checker's list
        text, u = llm.ask(prompt + REDO.format(problems='\n'.join(problems)), max_tokens=4000)
counts, bad = grade(probe(text, use))                            # kept / lost / distorted against the 30 gold facts
```

The run made 180 calls, probes included, and cost 52 US cents.

[![Terminal output of ch5b_handoff.py, run 1 of three conditions: the probe kept 30 of 30 facts on the source pages; free text kept 30, 29, 13, 15 and 11 facts at the five hops, with handoffs of about 1,600 to 2,000 words and an editor's memo of 264 words; free text with a verifier kept 30, 30, 30, 29 and 27, with the verifier flagging 6 to 20 problems per hop; the schema kept 30 of 30 at every hop with handoffs of about 370 to 430 words](/img/agents/ch5b_handoff_run1-run.png)](/img/agents/ch5b_handoff_run1-run.png)

{{FIG:b_handoff_results|Facts still stated after each hop, mean of three runs. Free text fell to 14.7 of 30 by the editor, most of it at the risk reviewer. The schema kept all 30 in every run; the verifier rescued free text at a cost.}}

**Free text lost half the facts, mostly at one hop.** The researcher and analyst passed on nearly everything. Then the risk reviewer wrote about risks and dropped prices and minimum orders. A supplier's exact price appeared in only 6% of the risk reviewers' handoffs, and later agents could not bring the facts back. The editor's memo kept 11, 16 and 17 of 30 facts in the three runs. Facts disappeared rather than changed: in 60 hop outputs the probe found 2 wrong values, and no distractor was copied.

**The loss was not the smooth decay of the formula.** Fitting $$q^5$$ to 14.7 of 30 gives $$q = 0.867$$ per hop. But the real loss came at the one hop whose role made other facts look irrelevant. For design, the danger is the handoff into an agent with a narrower job.

[![Terminal output of ch5b_handoff.py, summary: mean facts kept after each hop; free 30.0, 29.3, 16.0, 16.7 and 14.7; free with verifier 29.7, 29.7, 30.0, 19.7 and 28.7; schema 30.0 at every hop; schema with verifier 30.0 to 29.7; cost per run without probes 0.0253, 0.0722, 0.0100 and 0.0346 dollars; implied per-hop retention 0.867 for free text](/img/agents/ch5b_handoff_summary-run.png)](/img/agents/ch5b_handoff_summary-run.png)

**The schema kept every fact and cost the least.** All three schema runs ended with 30 of 30 facts. The table was a payload the agents copied, not text they rewrote, and each agent's own work had its own field. It was also the cheapest condition: $0.010 per chain against $0.025 for free text. The schema agents wrote about 400 words per handoff; the free-text agents wrote about 1,800. The schema editors' memos stayed within the 250-word limit (177 to 193 words). The free-text editors wrote 264, 437 and 615 words.

**The verifier rescued free text, noisily.** With a verifier, free-text chains ended with 27, 29 and 30 facts. The cost was almost three times that of free text alone: $0.072 per chain. But the verifier is a model, and the traces show its own failures:

- It flagged 202 problems on free text, and all 15 hops were redone. Of those flags, 25 contain phrases such as "not a factual error" or "no problem": items it had itself found correct.
- In one run, the writer's redo came back empty. The whole experiment's usage shows exactly 4,000 hidden reasoning tokens, one call's full output allowance. The chain went on, and the editor recovered only because its own verifier listed the facts from the source.
- With the schema, the verifier asked for facts the task did not need, such as EU delivery days. In one run the researcher then turned `uk_delivery_days` into a nested object. Five of fifteen schema-and-verifier hops broke the table's shape; with no verifier, none did.

[![Terminal output of ch5b_handoff_stats.py: the share of suppliers whose exact price, days and minimum order appear in each hop's text; free text drops to 6 percent for price at the risk reviewer; editor word counts against the 250 limit; the verifier flagged 202 problems over 15 free-text hops, 25 of which call the item correct or not an error, and 212 over 15 schema hops; schema drift in 5 of 15 schema-with-verifier hops and none without the verifier](/img/agents/ch5b_handoff_stats-run.png)](/img/agents/ch5b_handoff_stats-run.png)

Two more findings come from the saved handoffs. A free-text risk reviewer added "single-site dependency" as a risk for the UK supplier, which no page states: rewriting invites invention as well as loss. And the empty handoff passed silently to the next agent. A one-line check in code would have stopped it.

> [!WARNING]
> Three runs per condition, one chain of five fixed roles, six short pages and one model at minimal reasoning effort. The probe is itself a model call. It found all 30 facts on the source pages, and a separate string check for the exact figures showed the same pattern. What this run supports: rewriting is where facts are lost, and a narrow role is where they are lost fastest. A schema prevents the loss more cheaply than a checker of prose.

### Failures in practice

| Failure | What you see in the trace | Control |
|---|---|---|
| Lost facts in a chain | a field present at hop k and absent at hop k+1 | structured results copied as data; store and pass a reference (Section 5.3) |
| Conflicting decisions | two subagents assume different units, styles or rules | decide shared choices before the split; one owner per decision |
| Loops | the same call or message repeated; no "done" | a turn cap and a token budget in code; repetition detection |
| Cost blow-up | tokens per run growing with the square of turns | briefs instead of a shared chat; caps on subagents per round |
| Easy agreement | unanimity rises, accuracy does not | measure agreement against labels; verifiers that see the source |
| Weak verification | checks that pass broken output (it compiles) | check the task objective and the sources, not only the format |
| Silent empty output | a hop with zero words, and the run goes on | schema and non-empty checks between agents, like Chapter 4's gates |

> [!TIP] In production
> Give every handoff a schema and validate it in code before the next agent runs. Keep facts in fields that agents copy, and their own work in a separate field. Put verifiers where they can see the source, and validate their output too. A checker that asks for more than the task needs will push agents to change the schema.

### Discussion

1. **Is MAST's 44% for "system design" a comfort or a warning?** Both. Most failures are in things you control, but the paper's own fixes left ChatDev failing most tasks.
2. **Why not give every agent the source pages?** Then nothing is lost at handoffs, but every context grows with the source. That is Cognition's "share context" principle, and it works until the sources outgrow one context.
3. **Is a model verifier worth three times the cost?** Here a schema did better for less. A verifier earns its place where the output cannot be a schema, such as an argument or code behaviour, and where it can check against something real.

## 5.6 Building, testing and running multi-agent systems in practice

### The example: Northwind decides how to ship it

Northwind's team now has this chapter's evidence. A lead with subagents covers more suppliers (Section 5.1), and a full brief carries the rules (Section 5.3). Debate helps only where single answers are weak (Section 5.4), and free-text handoffs lose facts (Section 5.5). The team must now decide what to build, how to stop it running away, how to see inside it, and how to know it works.

### Start with one agent, and a checklist for the second

Chapter 4 quoted OpenAI's advice to "maximize a single agent's capabilities first". Anthropic and Cognition disagree about much, but both describe tasks where one agent is the right design. The checklist below turns that into questions with measurable answers.

{{FIG:b_checklist|A checklist from one agent to several. Each step needs a measured "yes": one agent fails on a fixed eval set; the failure is about context, not skill; the work splits into independent parts that do not share decisions; and the value pays for the tokens. Then ship with verifiers, human checkpoints and per-agent evals.}}

The order matters. First, measure that one agent fails, on fixed tasks with traces; the fix is often a better tool or prompt (Chapter 2). Second, check that the failure is about context: too many pages or too many tools for one agent. More agents of the same model do not fix a skill failure such as bad arithmetic. Third comes Cognition's test: if the parts share decisions, splitting creates conflicts. Only then ask about cost, and expect Anthropic's scale of about 15 times the tokens of a chat (Chapter 1).

### Budgets and turn limits, in code

Every agent needs a limit on its own loop, and the run needs a limit on the whole. Frameworks now provide the first as a parameter.

> [!PAPER] OpenAI Agents SDK, "Running agents" (documentation, accessed October 2026)
> [![A list from the documentation: the runner calls the LLM for the current agent; if the output is final, the loop ends; if the LLM requests a handoff, the current agent and input are updated and the loop re-runs; if it produces tool calls, they run and the loop re-runs; if we exceed the max_turns passed, we raise a MaxTurnsExceeded exception; pass max_turns=None to disable this turn limit](/img/agents/ch5b-sdk-maxturns.png)](/img/agents/ch5b-sdk-maxturns.png)
>
> **What it says:** the runner loops over model calls, handoffs and tool calls; "If we exceed the max_turns passed, we raise a MaxTurnsExceeded exception."
>
> **What it means:**
>
> A handoff to another agent is one more turn of the same loop, so one cap bounds a whole chain of agents. It is the cap $$K$$ of the loop formula, written in code.
>
> The cap does not limit tokens, so add a token or money budget too. And an exception is not an answer for the user. Decide in advance what the user receives when the cap is hit; the documentation describes handlers for this.

Set limits at three levels: calls per agent, subagents per round (Section 5.3's lead stopped at six), and tokens or money per run. Log which limit ended each run. A rising share of runs that end at a cap is an early warning.

### Tracing each agent

A multi-agent run is a tree: a lead, its subagents, and their model and tool calls. If each is a span with a parent, "which agent lost the price?" takes minutes to answer.

> [!PAPER] OpenAI Agents SDK, "Tracing" (documentation, accessed October 2026)
> [![A list from the documentation: by default the SDK traces the whole run in a trace; each runner invocation in a task span; each model turn in a turn span; each time an agent runs in an agent span; LLM generations in a generation span; function tool calls in a function span; guardrails in a guardrail span; handoffs in a handoff span; and audio inputs and outputs in their own spans](/img/agents/ch5b-sdk-tracing.png)](/img/agents/ch5b-sdk-tracing.png)
>
> **What it says:** by default each agent run, model generation, tool call, guardrail and handoff is "wrapped in" its own span inside one trace.
>
> **What it means:** the handoff has a span of its own, and that is where to record what passed between agents. The telephone game happens there. Our per-hop probe was a manual version of the same idea.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on debugging
> [![A paragraph from the post: debugging benefits from new approaches; agents make dynamic decisions and are non-deterministic between runs, even with identical prompts; users would report agents not finding obvious information, but we couldn't see why; adding full production tracing let us diagnose why agents failed and fix issues systematically; we monitor agent decision patterns and interaction structures, all without monitoring the contents of individual conversations, to maintain user privacy](/img/agents/ch5b-anthropic-debugging.png)](/img/agents/ch5b-anthropic-debugging.png)
>
> **What it says:** "Adding full production tracing let us diagnose why agents failed"; the team monitors "decision patterns and interaction structures" without reading conversations.
>
> **What it means:** "not finding obvious information" can be a bad query, a poor source or a tool failure, and only a trace tells them apart. You can also learn a lot from the shape of runs (how many subagents, which tools, where loops happen) without reading what users asked. Chapter 8 builds this tracing.

### Evaluating the whole system and each agent

A multi-agent system needs evals at two levels. **End to end** asks whether the final answer met the task. **Per agent** asks whether each agent did its part and handed on what the next one needed. End-to-end evals tell you whether to ship; per-agent evals tell you what to fix. Our handoff experiment needed both: the memo was the end-to-end result, and the per-hop probe found the risk reviewer.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on judges
> [![A paragraph from the post: LLM-as-judge evaluation scales when done well; an LLM judge evaluated each output against a rubric of factual accuracy, citation accuracy, completeness, source quality and tool efficiency; multiple judges for each component were less consistent than a single LLM call with a single prompt outputting scores from 0.0 to 1.0 and a pass-fail grade; especially effective when test cases had a clear answer](/img/agents/ch5b-anthropic-judge.png)](/img/agents/ch5b-anthropic-judge.png)
>
> **What it says:** a rubric of factual accuracy, citation accuracy, completeness, source quality and tool efficiency. "A single LLM call with a single prompt" giving scores and a pass-fail grade "was the most consistent and aligned with human judgements".
>
> **What it means:** one judge with one rubric beat several specialised judges. More parts in the evaluator create the same coordination problems as more agents in the system. The best case was a test with a clear answer, which is how our experiments graded too.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on human testers
> [![A paragraph from the post: human evaluation catches what automation misses; people testing agents find edge cases that evals miss, including hallucinated answers on unusual queries, system failures, or subtle source selection biases; human testers noticed that early agents consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources like academic PDFs or personal blogs; adding source quality heuristics to prompts helped](/img/agents/ch5b-anthropic-human.png)](/img/agents/ch5b-anthropic-human.png)
>
> **What it says:** testers found that agents "consistently chose SEO-optimized content farms over authoritative but less highly-ranked sources".
>
> **What it means:** the judge scored the answers, and they looked fine. A person reading the sources saw a pattern no rubric asked about. Use people to discover what to measure, then add it to the rubric. Chapters 6 and 7 build this loop.

### Human checkpoints

Put a person where a wrong decision is expensive and hard to undo, not at every step. For Northwind that is one checkpoint: the buying director approves the three suppliers before anyone contacts them. While the system is new, a person also reviews a sample of runs and their traces each week. Chapter 4 showed how to pause a workflow for approval and resume it. In a multi-agent system the pause belongs to the lead, after merging and verification, so the person sees one result, not six partial ones.

### Running it: state, deployments and emergent behaviour

Multi-agent runs hold state for minutes or hours, which turns ordinary operations into design questions.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on state
> [![A paragraph from the post: agents are stateful and errors compound; agents can run for long periods, maintaining state across many tool calls; minor system failures can be catastrophic; restarts are expensive and frustrating for users; the team built systems that can resume from where the agent was when the errors occurred, let the agent know when a tool is failing so it can adapt, and combined this with deterministic safeguards like retry logic and regular checkpoints](/img/agents/ch5b-anthropic-stateful.png)](/img/agents/ch5b-anthropic-stateful.png)
>
> **What it says:** "Agents are stateful and errors compound"; the team built systems that "resume from where the agent was", with "retry logic and regular checkpoints".
>
> **What it means:** this is Chapter 4's durable execution, applied to agents. A subagent that crashes after twenty minutes should not force the lead to start again. Checkpoint each agent's state and each handoff, and make side effects idempotent.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on deployment
> [![A paragraph from the post: deployment needs careful coordination; agent systems are highly stateful webs of prompts, tools and execution logic that run almost continuously; whenever updates are deployed, agents might be anywhere in their process; rainbow deployments avoid disrupting running agents by gradually shifting traffic from old to new versions while keeping both running simultaneously](/img/agents/ch5b-anthropic-rainbow.png)](/img/agents/ch5b-anthropic-rainbow.png)
>
> **What it says:** "agents might be anywhere in their process" when code changes, so the team uses "rainbow deployments", shifting traffic gradually while old and new versions both run.
>
> **What it means:** a run that started with version 1 of the lead's prompt should finish with version 1. If a handoff schema changes, a subagent on the new version may not understand an old brief. Version prompts and schemas together.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on emergent behaviour
> [![A paragraph from the post: multi-agent systems have emergent behaviors, which arise without specific programming; small changes to the lead agent can unpredictably change how subagents behave; success requires understanding interaction patterns, not just individual agent behavior; the best prompts are frameworks for collaboration that define the division of labor, problem-solving approaches and effort budgets](/img/agents/ch5b-anthropic-emergent.png)](/img/agents/ch5b-anthropic-emergent.png)
>
> **What it says:** "small changes to the lead agent can unpredictably change how subagents behave".
>
> **What it means:** our verifier is a small example. Adding it changed what the researcher wrote, a nested field the schema did not have, which changed what every later agent received. After any change to one agent, test the whole system.

> [!PAPER] Anthropic, "How we built our multi-agent research system" (Engineering blog, June 2025), on prompting
> [![A paragraph from the post: think like your agents; to iterate on prompts you must understand their effects; the team built simulations with the exact prompts and tools from the system and watched agents work step by step; this revealed failure modes such as agents continuing when they already had sufficient results, using overly verbose search queries, or selecting incorrect tools](/img/agents/ch5b-anthropic-think.png)](/img/agents/ch5b-anthropic-think.png)
>
> **What it says:** watching agents "step-by-step" revealed agents that kept going after they had enough, wrote "overly verbose search queries" or selected "incorrect tools".
>
> **What it means:** each of those is a MAST mode, found by reading traces, not final answers. The cheapest debugging tool for a multi-agent system is a person reading ten traces.

{{FIG:b_ops|Where the controls sit in a running multi-agent system. Each subagent's result passes a schema check in code; a verifier compares the merged result with the sources; a person approves the decision. Budgets, turn limits, a trace per agent and evals at both levels apply to every part.}}

> [!TIP] In production
> Before launch, write down five things. They are the caps on calls and tokens, the schema of every handoff, and the trace fields at each handoff. They are also the eval set, with end-to-end and per-agent checks, and the one decision a person approves. If any is missing, the system is not ready, however good the demo looked.

### Discussion

1. **Should the verifier be an agent or code?** Code wherever the check can be written: schema, ranges, every supplier present. A model only for what code cannot judge, measured against labels like any judge (Chapter 6).
2. **Who owns a multi-agent system's quality?** One team, not one per agent. In our chain each agent could pass its own test while the system failed.
3. **When should a team remove agents?** When traces show an agent mostly passing text through, or when a schema and code can do its job. Every handoff removed is one less place to lose a fact.

## 5.7 Discussion

Splitting work across agents helps when the work is broad, parallel and too large for one context, and when the parts do not share decisions. Every split adds a handoff, and every handoff is a place where facts are lost, decisions conflict and tokens multiply. Talking agents can help where single answers are weak, but at equal compute a vote is usually as good and simpler.

The engineering answer is the same in every section: move what you can out of conversation and into structure. Use briefs instead of one-line requests, schemas instead of prose, and caps in code instead of "be efficient". Use verifiers that read sources, and traces at every handoff. What remains in conversation is the part only a model can do, and that part needs measurement, which is the subject of the next three chapters.

## Exercises

1. Northwind wants a sixth agent, a "negotiation planner", at the end of the chain of Section 5.5. Using the per-hop retention measured for free text and for the schema, estimate the facts that reach the end in each design. State your independence assumption and say which hop you would instrument first.
2. A debate of four agents over three rounds of answers uses how many calls? Design the equal-compute vote and the self-review control, and say which result would make you keep the debate in production.
3. Write the JSON schema for the handoff from Northwind's risk reviewer to the writer. Mark which fields are copied facts and which are the reviewer's judgement, and name the code check that runs before the writer starts.
4. Your multi-agent system ends 8% of runs at the turn cap, and the share is rising. List three hypotheses, the trace field that would test each, and the MAST mode each belongs to.
5. Section 5.1 compared one agent with a lead and subagents; Section 5.3 found that a one-line brief lost the rules. Design an experiment that measures how the lead's own brief-writing changes the end-to-end result, with the metric, the control and the expected cost.

## Key takeaways

- A **multi-agent system** gives each agent its own loop, tools and context. It earns its cost when work is broad, parallel and larger than one context. Anthropic reported a 90.2% gain on its research eval at about 15 times the tokens of a chat; Cognition argued that agents which share decisions should not be split.
- The **architectures** (orchestrator and subagents, hierarchies, peer handoffs, a shared blackboard, role-play teams) differ in who decides and what each agent sees. Choose by what the parts share.
- A **handoff** is all a subagent knows. In Section 5.3 a one-line brief got 3.7 of 12 facts right, a full brief 11.3, and the full history 11.0, with the lead's role leaking into the subagent.
- **MCP** connects an agent to tools and data; **A2A** connects it to other agents. Both standardise the envelope of a message, not what it says.
- **Debate** lets agents read each other's answers before a vote. In our run it beat a vote of the same size when single answers were terse and weak (73% against 53% at six calls). It added nothing when the agents had room to work. Published equal-compute studies mostly favour voting.
- A model **re-checking its own answer** moved from right to wrong as often as the reverse (24 against 23). Reading other answers moved 44 the right way and 13 the wrong way.
- **MAST** found 44.2% of failures in system design, 32.3% in misalignment between agents and 23.5% in verification. Most are in specs, roles, stop rules and checks you control.
- Facts compound across handoffs like steps across a chain, $$q^n$$. In our five-agent chain, free-text handoffs kept 14.7 of 30 facts, mostly lost at one narrow role; a **schema** kept all 30 and cost less.
- A **verifier** agent is only as good as its own measurement. Ours rescued free text at three times the cost, raised false flags, and in one run pushed agents to change the schema.
- Run multi-agent systems with **caps in code** and a **trace** with a span for every agent and handoff. Evaluate **end to end and per agent**, and put one **human checkpoint** where a decision is costly.

## References

**Papers and books**

1. Marvin Minsky. *The Society of Mind*. Simon & Schuster, 1986.
2. Yilun Du, Shuang Li, Antonio Torralba, Joshua B. Tenenbaum, Igor Mordatch. [*Improving Factuality and Reasoning in Language Models through Multiagent Debate*](https://arxiv.org/abs/2305.14325). ICML 2024 (arXiv May 2023).
3. Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Shuming Shi, Zhaopeng Tu. [*Encouraging Divergent Thinking in Large Language Models through Multi-Agent Debate*](https://arxiv.org/abs/2305.19118). EMNLP 2024 (arXiv May 2023).
4. Andries Smit, Nathan Grinsztajn, Paul Duckworth, Thomas D. Barrett, Arnu Pretorius. [*Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs*](https://arxiv.org/abs/2311.17371). ICML 2024 (arXiv November 2023).
5. Qineng Wang, Zihao Wang, Ying Su, Hanghang Tong, Yangqiu Song. [*Rethinking the Bounds of LLM Reasoning: Are Multi-Agent Discussions the Key?*](https://aclanthology.org/2024.acl-long.331). ACL 2024 (arXiv February 2024).
6. Hyeong Kyu Choi, Xiaojin Zhu, Sharon Li. [*Debate or Vote: Which Yields Better Decisions in Multi-Agent Large Language Models?*](https://arxiv.org/abs/2508.17536). NeurIPS 2025 (arXiv August 2025).
7. Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, Ion Stoica. [*Why Do Multi-Agent LLM Systems Fail?*](https://arxiv.org/abs/2503.13657). NeurIPS 2025 Datasets and Benchmarks track (arXiv March 2025).

**Engineering blogs and docs**

8. Anthropic (Jeremy Hadfield, Barry Zhang, Kenneth Lien, Florian Scholz, Jeremy Fox, Daniel Ford). [*How we built our multi-agent research system*](https://www.anthropic.com/engineering/multi-agent-research-system). Engineering blog, 13 June 2025.
9. Cognition (Walden Yan). [*Don't Build Multi-Agents*](https://cognition.ai/blog/dont-build-multi-agents). Blog, 12 June 2025.
10. OpenAI. [*Running agents*](https://openai.github.io/openai-agents-python/running_agents/) and [*Tracing*](https://openai.github.io/openai-agents-python/tracing/). OpenAI Agents SDK documentation, accessed October 2026.

**Code for Sections 5.4 to 5.6**

11. [`code/agents/ch5b_make_data.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_make_data.py): thirty word problems with computed gold answers, and six supplier pages with 30 gold facts, in `data/ch5/b/`.
12. [`code/agents/ch5b_debate.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_debate.py) and [`ch5b_debate_stats.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_debate_stats.py): debate, voting and self-review at equal calls, with paired sign tests and answer-change counts.
13. [`code/agents/ch5b_handoff.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_handoff.py) and [`ch5b_handoff_stats.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_handoff_stats.py): the five-agent chain in four conditions, the per-hop probe, the string check, verifier noise and schema drift.
14. [`code/agents/ch5b_math.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_math.py): facts across handoffs, the verifier formula, group-chat tokens and capped loops, each with its assumptions printed.
15. [`code/agents/figs_ch5b.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/figs_ch5b.py), [`shots_ch5b.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/shots_ch5b.py) and [`ch5b_termshots.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/agents/ch5b_termshots.py): figures, paper and page excerpts, and terminal pictures.

## Next

→ [Chapter 6: Evals, part 1](./06-evals-1-golden-sets-and-judges.md)

Every experiment in this chapter needed three things: tasks with known answers, a grader you can trust, and a way to look inside a run. Chapter 6 builds them properly. It starts from golden sets and error analysis and turns failures into pass/fail criteria. Then it calibrates a model judge against people, evaluates the path an agent took as well as its answer, and reads what the public agent benchmarks measure.
