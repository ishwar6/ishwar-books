---
title: "Agentic Systems"
subtitle: "Patterns, evals and production, from first principles"
description: "How real AI agents are designed, measured and run in production: the agent loop, tools and memory, the reasoning and workflow patterns from the research papers, multi-agent systems, evals that actually catch regressions, tracing, guardrails, and what companies like Uber, DoorDash and Anthropic learned the hard way. Written for engineers who know what an LLM call is and want to build systems around it."
status: in-progress
accent: "#e05c9a"
order: 2
chapters: 10
topics: [Agent patterns, Tools and memory, Multi-agent systems, Evals, Tracing, Guardrails]
---

You know what an LLM call is: a prompt goes in, text comes out. This book is about everything that has to be built around that call before it can be trusted to do a job on its own: the loop that lets a model act, the tools it acts with, the patterns that keep it on track, the evals that tell you whether it works, and the tracing and guardrails that keep it safe when it runs ten thousand times a day.

It is written for engineers, in plain English, in the same style as the research-paper breakdowns on this site: every pattern comes with the paper that introduced it (as highlighted screenshots), a simple picture, a small piece of real code where it helps, and a "what a company learned" box drawn from published engineering blogs. The emphasis is on production: how big agentic systems are evaluated, traced and kept reliable, not on toy demos.

## The ten chapters

**Part 1, Foundations**
1. **What an agent is, and when not to build one.** The perceive-reason-act loop, model versus agent, workflows versus agents, levels of autonomy, and the economics of letting a model take ten steps instead of one.
2. **The building blocks.** Model, tools, instructions, memory and state; how to design a tool a model can actually use; context engineering.

**Part 2, Patterns**
3. **Single-agent reasoning patterns.** ReAct, plan-and-execute, reflection and self-critique, tree search over thoughts, and what the papers measured.
4. **Workflow patterns.** Prompt chaining, routing, parallel fan-out, orchestrator and workers, evaluator and optimiser; a planner-plus-responders case study.
5. **Multi-agent systems.** When several agents beat one, how they coordinate, and the ways they fail.

**Part 3, Evals**
6. **Evals, part 1.** Golden sets, error analysis, pass/fail criteria, LLM-as-judge and how to calibrate it, trajectory evals, and what the public agent benchmarks measure.
7. **Evals, part 2: in production.** Offline suites, online checks, regression gates in CI, and how companies built evals for real agent products.

**Part 4, Production**
8. **Observability and tracing.** Traces and spans for agents, what to log, replaying a run, cost and latency dashboards.
9. **Guardrails, safety and human oversight.** Input and output checks, tool permissions, sandboxing, prompt injection, approval steps and kill switches.

**Part 5, Capstone**
10. **One agent, end to end.** A small production-shaped agent with tools, a loop, tracing, an eval set, a judge and a CI gate, with the code in the repo. Plus a cheatsheet and a glossary.

Chapters are published one at a time; the list above is the plan.
