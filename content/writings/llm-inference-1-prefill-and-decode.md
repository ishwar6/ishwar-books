---
title: "How an LLM Writes: Prefill and Decode"
description: "Why the first word takes a moment and the rest stream steadily. The two phases of LLM inference, built from the ground up and measured on a real model: a 512-token prompt is read in 27 ms, but writing 512 tokens takes 5.2 seconds."
date: 2026-09-28
tags: [inference, llm, vllm]
series: "LLM Inference from the Ground Up"
series_part: 1
motif: timeline
accent: "#7c9cff"
---

You type a question into a chatbot and press enter. There is a short pause. Then the first word appears, and the rest of the answer streams out at a steady pace, word after word.

That pause and that stream are not the same thing happening at two speeds. They are two completely different jobs, with opposite physics, done by the same model. Once you see why, almost everything about how LLMs are served, from vLLM's design to why your GPU bill looks the way it does, starts to make sense.

This series builds that picture from the ground up. No prior knowledge of GPUs needed.

> [!TIP] The whole series in one minute
> **Part 1 (this one):** an LLM answers in two phases. **Prefill** reads your whole prompt at once and is limited by math. **Decode** writes one token at a time and is limited by memory. Decode is where the time goes.
>
> **Part 2:** to write each new token, the model needs notes on every token before it. Those notes are the **KV cache**, and they get big.
>
> **Part 3:** **vLLM** manages those notes the way an operating system manages memory, which lets one GPU serve many more people at once.
>
> **Part 4:** **speculative decoding** uses the compute that decode leaves idle to write several tokens per step, without changing a single word of the output.

Every number in this piece was measured on a real model, Qwen2.5-0.5B, running on an Apple M5 Pro GPU with PyTorch. It is a small model on a laptop, so the absolute numbers are small too. The *shapes* are the same on a data-centre GPU, and where it matters I will show the equivalent numbers for an NVIDIA H100.

## How a model writes one word

A language model does not see words. It sees **tokens**: chunks of text, usually a word or part of a word. "The cat sat on the" is five tokens.

What the model does is simple to state. Give it a list of tokens, and it runs them through a stack of layers (24 in our model) and produces one thing at the end: a probability for every token in its vocabulary of being the **next** one. Pick one, usually a likely one, and you have the next word.

<figure class="fig"><svg viewBox="0 0 760 262" role="img" aria-label="The generation loop: the text so far goes into the model, one new token comes out, it is appended, and the model runs again."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="40" y="70" width="300" height="64" rx="10"/><text class="t-strong" x="190" y="108" text-anchor="middle">The cat sat on the</text><rect class="box-1" x="420" y="52" width="150" height="100" rx="16"/><text class="t-strong" x="495" y="98" text-anchor="middle">the model</text><text class="t-tick" x="495" y="118" text-anchor="middle">24 layers</text><rect class="box-3" x="650" y="76" width="70" height="52" rx="10"/><text class="t-strong" x="685" y="108" text-anchor="middle">mat</text><line class="edge" x1="340" y1="102" x2="416" y2="102" marker-end="url(#ah)"/><line class="edge" x1="570" y1="102" x2="646" y2="102" marker-end="url(#ah)"/><path class="path" d="M685,132 C685,215 190,215 190,138" fill="none" marker-end="url(#ah-on)"/><text class="t-note" x="438" y="246" text-anchor="middle">append the new token, run the whole model again</text><text class="t-tick" x="190" y="58" text-anchor="middle">everything so far</text><text class="t-tick" x="685" y="64" text-anchor="middle">one new token</text></svg><figcaption>The generation loop. Everything so far goes in, one new token comes out, it is appended, and the whole model runs again.</figcaption></figure>

To write a whole answer, you repeat that. Append the new token, run the model again, get the next one. A 500-token answer means 500 trips through the model, one after another. Each trip needs the token from the trip before it, so they cannot run in parallel.

That loop is the entire story of LLM inference. Everything else is making it faster.

## Two phases: reading and writing

The easiest way to see what happens is to follow one real request from start to finish. Say you ask:

> What is the capital of France?

That question is about 7 tokens: `What`, `is`, `the`, `capital`, `of`, `France`, `?`. Here is what the model does with it, step by step.

<figure class="fig"><svg viewBox="0 0 760 388" role="img" aria-label="One request step by step: prefill reads all seven prompt tokens together and produces the first word; each decode step feeds back one word and produces the next."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="90" y="22" text-anchor="middle">step 0 · prefill</text><text class="t-tick" x="90" y="40" text-anchor="middle">all 7 prompt tokens at once</text><rect class="box-1" x="40" y="54" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="69" text-anchor="middle">What</text><rect class="box-1" x="40" y="78" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="93" text-anchor="middle">is</text><rect class="box-1" x="40" y="102" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="117" text-anchor="middle">the</text><rect class="box-1" x="40" y="126" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="141" text-anchor="middle">capital</text><rect class="box-1" x="40" y="150" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="165" text-anchor="middle">of</text><rect class="box-1" x="40" y="174" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="189" text-anchor="middle">France</text><rect class="box-1" x="40" y="198" width="100" height="20" rx="4"/><text class="t-tick" x="90" y="213" text-anchor="middle">?</text><line class="edge" x1="90" y1="224" x2="90" y2="246" marker-end="url(#ah)"/><rect class="box" x="25" y="250" width="130" height="40" rx="8"/><text class="t-strong" x="90" y="275" text-anchor="middle">model</text><line class="edge" x1="90" y1="290" x2="90" y2="314" marker-end="url(#ah)"/><rect class="box-3" x="50" y="318" width="80" height="28" rx="6"/><text class="t-strong" x="90" y="337" text-anchor="middle">The</text><text class="t-title" x="247" y="22" text-anchor="middle">step 1 · decode</text><text class="t-tick" x="247" y="40" text-anchor="middle">1 token in, 1 out</text><rect class="box-2" x="207" y="200" width="80" height="22" rx="4"/><text class="t-tick" x="247" y="215" text-anchor="middle">The</text><line class="edge" x1="247" y1="224" x2="247" y2="246" marker-end="url(#ah)"/><rect class="box" x="192" y="250" width="110" height="40" rx="8"/><text class="t-strong" x="247" y="275" text-anchor="middle">model</text><line class="edge" x1="247" y1="290" x2="247" y2="314" marker-end="url(#ah)"/><rect class="box-3" x="207" y="318" width="80" height="28" rx="6"/><text class="t-strong" x="247" y="337" text-anchor="middle">capital</text><path class="path" d="M130,332 C170,332 182,211 203,211" fill="none" marker-end="url(#ah-on)"/><text class="t-title" x="387" y="22" text-anchor="middle">step 2 · decode</text><text class="t-tick" x="387" y="40" text-anchor="middle">1 token in, 1 out</text><rect class="box-2" x="347" y="200" width="80" height="22" rx="4"/><text class="t-tick" x="387" y="215" text-anchor="middle">capital</text><line class="edge" x1="387" y1="224" x2="387" y2="246" marker-end="url(#ah)"/><rect class="box" x="332" y="250" width="110" height="40" rx="8"/><text class="t-strong" x="387" y="275" text-anchor="middle">model</text><line class="edge" x1="387" y1="290" x2="387" y2="314" marker-end="url(#ah)"/><rect class="box-3" x="347" y="318" width="80" height="28" rx="6"/><text class="t-strong" x="387" y="337" text-anchor="middle">is</text><path class="path" d="M287,332 C327,332 322,211 343,211" fill="none" marker-end="url(#ah-on)"/><text class="t-title" x="527" y="22" text-anchor="middle">step 3 · decode</text><text class="t-tick" x="527" y="40" text-anchor="middle">1 token in, 1 out</text><rect class="box-2" x="487" y="200" width="80" height="22" rx="4"/><text class="t-tick" x="527" y="215" text-anchor="middle">is</text><line class="edge" x1="527" y1="224" x2="527" y2="246" marker-end="url(#ah)"/><rect class="box" x="472" y="250" width="110" height="40" rx="8"/><text class="t-strong" x="527" y="275" text-anchor="middle">model</text><line class="edge" x1="527" y1="290" x2="527" y2="314" marker-end="url(#ah)"/><rect class="box-3" x="487" y="318" width="80" height="28" rx="6"/><text class="t-strong" x="527" y="337" text-anchor="middle">Paris</text><path class="path" d="M427,332 C467,332 462,211 483,211" fill="none" marker-end="url(#ah-on)"/><text class="t-title" x="667" y="22" text-anchor="middle">step 4 · decode</text><text class="t-tick" x="667" y="40" text-anchor="middle">1 token in, 1 out</text><rect class="box-2" x="627" y="200" width="80" height="22" rx="4"/><text class="t-tick" x="667" y="215" text-anchor="middle">Paris</text><line class="edge" x1="667" y1="224" x2="667" y2="246" marker-end="url(#ah)"/><rect class="box" x="612" y="250" width="110" height="40" rx="8"/><text class="t-strong" x="667" y="275" text-anchor="middle">model</text><line class="edge" x1="667" y1="290" x2="667" y2="314" marker-end="url(#ah)"/><rect class="box-3" x="627" y="318" width="80" height="28" rx="6"/><text class="t-strong" x="667" y="337" text-anchor="middle">.</text><path class="path" d="M567,332 C607,332 602,211 623,211" fill="none" marker-end="url(#ah-on)"/><text class="t-tick" x="20" y="374" text-anchor="start">Each new token is fed back in as the next step's input. The answer: The capital is Paris.</text></svg><figcaption>One request, step by step. Step 0 takes in all seven prompt tokens together and produces the first word. Every later step takes in one word and produces the next.</figcaption></figure>

**Step 0 is special.** The whole question already exists. Nothing about it is unknown. So the model can take in **all 7 tokens at the same moment**, the way you take in a short sentence at a glance instead of letter by letter. Inside the GPU, the 7 tokens travel through every layer side by side, as one block of numbers, and each layer processes all of them in a single operation.

At the end of step 0 the model has done two things. It has predicted the first word of the answer, `The`. And, while it read, it wrote down a set of notes about each of the 7 tokens, so it never has to read them again. (Those notes are the KV cache. They are the subject of Part 2.)

**Every step after that is different.** To produce the second word, the model needs to know the first. To produce the third, it needs the second. So each step takes in exactly **one** new token, the one it just wrote, and produces exactly one more:

| Step | Goes in | Comes out | Phase |
|---|---|---|---|
| 0 | `What is the capital of France?` (7 tokens, together) | `The` | prefill |
| 1 | `The` | `capital` | decode |
| 2 | `capital` | `is` | decode |
| 3 | `is` | `Paris` | decode |
| 4 | `Paris` | `.` | decode |

That split has names:

- **Prefill** is step 0: read the whole prompt at once and produce the first token.
- **Decode** is every step after it: produce one token per step, each one depending on the last.

### Why can't the answer be written all at once too?

Because it does not exist yet. The prompt is known the moment you press enter, so all of it can be processed together. The answer is being invented one word at a time, and the model cannot know word 3 until it has chosen word 2. It is like the difference between reading a finished page, which you can scan in one look, and writing a sentence, which you can only write one word after another.

### What the two phases look like in time

Here is the same idea drawn to scale, using real timings for a 512-token prompt:

<figure class="fig"><svg viewBox="0 0 760 210" role="img" aria-label="A request has two phases. Prefill reads the whole prompt in one pass and produces the first token. Decode then produces one token per step."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="80" y="40" text-anchor="start">time</text><line class="edge" x1="114" y1="36" x2="200" y2="36" marker-end="url(#ah)"/><g class="mark"><title>prefill: all 512 prompt tokens in one pass, 27.4 ms</title><rect class="s1" x="80" y="70" width="91.2" height="54" rx="6"/></g><text class="t-val" x="125.60102446843905" y="102" text-anchor="middle" style="fill:#fff">prefill</text><g class="mark"><title>decode step 1: one new token, 10.0 ms</title><rect class="s2" x="174.2" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 2: one new token, 10.0 ms</title><rect class="s2" x="207.7" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 3: one new token, 10.0 ms</title><rect class="s2" x="241.2" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 4: one new token, 10.0 ms</title><rect class="s2" x="274.7" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 5: one new token, 10.0 ms</title><rect class="s2" x="308.1" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 6: one new token, 10.0 ms</title><rect class="s2" x="341.6" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 7: one new token, 10.0 ms</title><rect class="s2" x="375.1" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 8: one new token, 10.0 ms</title><rect class="s2" x="408.6" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 9: one new token, 10.0 ms</title><rect class="s2" x="442.1" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 10: one new token, 10.0 ms</title><rect class="s2" x="475.6" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 11: one new token, 10.0 ms</title><rect class="s2" x="509.1" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 12: one new token, 10.0 ms</title><rect class="s2" x="542.5" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 13: one new token, 10.0 ms</title><rect class="s2" x="576.0" y="70" width="30.5" height="54" rx="4"/></g><g class="mark"><title>decode step 14: one new token, 10.0 ms</title><rect class="s2" x="609.5" y="70" width="30.5" height="54" rx="4"/></g><text class="t-strong" x="650.9999999999998" y="103" text-anchor="start">...</text><line class="axis" x1="80" y1="140" x2="171.2020489368781" y2="140"/><line class="axis" x1="171.2020489368781" y1="134" x2="171.2020489368781" y2="146"/><line class="axis" x1="80" y1="134" x2="80" y2="146"/><text class="t-note" x="80" y="166" text-anchor="start">time to first token (TTFT)</text><line class="axis" x1="308.14432066919863" y1="140" x2="338.62988860227875" y2="140"/><text class="t-note" x="323.8871046357387" y="166" text-anchor="start">time per output token (TPOT)</text><text class="t-tick" x="80" y="194" text-anchor="start">measured: 512-token prompt, Qwen2.5-0.5B on an Apple M5 Pro. Prefill 27 ms, each decode step 10 ms.</text></svg><figcaption>The two phases, drawn to scale from real measurements: one wide prefill step, then a long run of narrow decode steps, one per new token.</figcaption></figure>

These two phases map onto the two numbers every serving dashboard shows, and onto what you feel as a user:

| Metric | What you experience | Set by |
|---|---|---|
| **TTFT**, time to first token | the pause before anything appears | prefill (step 0) |
| **TPOT**, time per output token (also called inter-token latency) | how fast the words stream after that | decode (every later step) |

In the France example, TTFT is how long step 0 takes. TPOT is the average length of steps 1 to 4.

## The measurement that surprises everyone

Here is a simple experiment. Take a 512-token prompt. Measure how long the model takes to **read** it (prefill). Then measure how long it takes to **write** 512 tokens (decode). Same model, same number of tokens.

<figure class="fig"><svg viewBox="0 0 760 150" role="img" aria-label="512 tokens in takes 27 ms, 512 tokens out takes about 5.2 seconds."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><g class="mark"><title>read 512 prompt tokens (prefill): 27.4 ms</title><rect class="hit-area" x="0" y="16.0" width="760" height="59.0"/><path class="s1 bar-mark" d="M260,33.5H260.0Q262.06849100123367,33.5 262.06849100123367,35.56849100123368V55.43150899876632Q262.06849100123367,57.5 260.0,57.5H260Z"/></g><text class="t-note" x="248" y="50.5" text-anchor="end">read 512 prompt tokens (prefill)</text><text class="t-val" x="270.06849100123367" y="50.5" text-anchor="start">27.4 ms</text><g class="mark"><title>write 512 new tokens (decode): 5,158 ms</title><rect class="hit-area" x="0" y="75.0" width="760" height="59.0"/><path class="s2 bar-mark" d="M260,92.5H646.0Q650.0,92.5 650.0,96.5V112.5Q650.0,116.5 646.0,116.5H260Z"/></g><text class="t-note" x="248" y="109.5" text-anchor="end">write 512 new tokens (decode)</text><text class="t-val" x="658.0" y="109.5" text-anchor="start">5,158 ms</text><line class="axis" x1="260" y1="16" x2="260" y2="134"/></svg><figcaption>Reading 512 tokens versus writing 512 tokens, measured on Qwen2.5-0.5B. The prefill bar is so short it is barely visible.</figcaption></figure>

Reading took **27 milliseconds**. Writing took **5.2 seconds**. That is about 190 times longer, for the same number of tokens.

Nothing is broken here. This is how every LLM behaves, on every GPU. And the reason is the most important idea in this whole series.

## The model is heavy, the math is light

Here is an analogy first, then the real numbers.

Picture a chef in a kitchen. To cook anything, the chef needs a huge recipe book, but the book is kept in a storeroom down the hall. And there is a strange rule: every time the chef cooks, they must carry the **entire book** from the storeroom and read every page, even to make a single dish.

- If **512 orders** are all waiting at once, that is fine. One trip to the storeroom, one read of the book, and the chef cooks all 512 dishes while the book is open. That is **prefill**.
- But if each dish can only be started once the previous one is done, the chef has to make the whole trip **once per dish**. The cooking takes a moment; the walking takes most of the time. That is **decode**.

The chef is the GPU's compute cores. The storeroom is the GPU's memory. The recipe book is the model's weights. Now the real numbers.

A model is, physically, a very large pile of numbers: its **weights**. Our small model has 494 million of them, stored in 16-bit format. That is **0.99 GB**. Llama 3.1 8B is 16 GB. The big frontier models are hundreds of gigabytes.

Those weights live in the GPU's memory. The part of the GPU that does arithmetic, the compute cores, sits next to that memory and has to pull the weights in to use them. Every single trip through the model, it has to pull in **all** of them, every layer, every matrix.

<figure class="fig"><svg viewBox="0 0 760 224" role="img" aria-label="Every forward pass reads all the weights from memory. Prefill uses that read for 512 tokens of work, decode for one."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="30" y="40" width="200" height="170" rx="14"/><text class="t-strong" x="130" y="72" text-anchor="middle">GPU memory</text><text class="t-note" x="130" y="94" text-anchor="middle">weights: 0.99 GB</text><rect class="box-1" x="56" y="110" width="148" height="11" rx="3"/><rect class="box-1" x="56" y="126" width="148" height="11" rx="3"/><rect class="box-1" x="56" y="142" width="148" height="11" rx="3"/><rect class="box-1" x="56" y="158" width="148" height="11" rx="3"/><rect class="box-1" x="56" y="174" width="148" height="11" rx="3"/><rect class="box-1" x="56" y="190" width="148" height="11" rx="3"/><rect class="box" x="530" y="40" width="200" height="170" rx="14"/><text class="t-strong" x="630" y="72" text-anchor="middle">compute cores</text><text class="t-note" x="630" y="94" text-anchor="middle">multiply and add</text><line class="edge" x1="230" y1="128" x2="526" y2="128" marker-end="url(#ah)"/><text class="t-tick" x="378" y="120" text-anchor="middle">read all 0.99 GB of weights</text><rect class="box-2" x="560" y="114" width="140" height="26" rx="6"/><text class="t-note" x="630" y="132" text-anchor="middle">decode: 1 token</text><line class="edge" x1="230" y1="176" x2="526" y2="176" marker-end="url(#ah)"/><text class="t-tick" x="378" y="168" text-anchor="middle">read all 0.99 GB of weights</text><rect class="box-1" x="560" y="162" width="140" height="26" rx="6"/><text class="t-note" x="630" y="180" text-anchor="middle">prefill: 512 tokens</text><text class="t-note" x="380" y="30" text-anchor="middle">memory bandwidth, measured: 262 GB/s</text></svg><figcaption>Both phases pay for the same trip through memory. Prefill gets 512 tokens of work out of it. Decode gets one.</figcaption></figure>

Now think about what each phase does with that trip:

- **Prefill** pulls the weights in once and uses them for **all 512 prompt tokens** at the same time.
- **Decode** pulls the weights in once and uses them for **one token**. Then it does it all again for the next token.

So decode spends most of its time not calculating, but *waiting for weights to arrive*, like the chef walking to the storeroom. We can check this with numbers. I measured this GPU's memory speed at **262 GB/s**. Reading 0.99 GB at that speed takes at least **3.8 ms**. That is the floor: no decode step can be faster than the time it takes to read the model once.

We measured a decode step at **10 ms**. So even the floor, the pure weight-reading time, is over a third of each step. Most of the rest is fixed per-step overhead: a small model like this one runs as hundreds of tiny GPU operations launched one after another, and launching each one costs time. You will see the proof of that further down, in the batching measurements: decoding *two* sequences at once costs only half a millisecond more than one.

> [!NOTE] The one idea
> **Decode is limited by how fast you can move the weights, not by how fast you can multiply.** Engineers call this being **memory-bound**. Prefill, which reuses each weight for many tokens, is limited by the multiplication itself: it is **compute-bound**.

## Counting work per byte

There is a clean way to put a number on this. It is called **arithmetic intensity**: how many calculations you do for each byte you read from memory.

Every weight gets used in about two calculations (a multiply and an add) for each token it processes. A weight is two bytes. So:

- **Decode, one sequence:** about 2 calculations per 2 bytes, so roughly **1 calculation per byte**.
- **Prefill, 512 tokens:** the same weights are used for 512 tokens, so roughly **512 calculations per byte**.

Every GPU has a speed limit for each resource. An NVIDIA H100 can do about **989 trillion** 16-bit calculations per second, and read about **3.35 trillion bytes** per second from memory. Divide one by the other and you get about **295**: below that many calculations per byte, the GPU cannot be kept busy, because the memory cannot feed it fast enough.

That boundary is best drawn as a picture called a **roofline**:

<figure class="fig"><svg viewBox="0 0 760 330" role="img" aria-label="Roofline for an H100: decode at small batch sits far down the memory-bound slope; prefill sits on the compute roof."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="70" y1="280.0" x2="740" y2="280.0"/><text class="t-tick" x="62" y="284.0" text-anchor="end">1</text><line class="grid" x1="70" y1="202.4" x2="740" y2="202.4"/><text class="t-tick" x="62" y="206.44844780681638" text-anchor="end">10</text><line class="grid" x1="70" y1="124.9" x2="740" y2="124.9"/><text class="t-tick" x="62" y="128.89689561363272" text-anchor="end">100</text><line class="grid" x1="70" y1="47.3" x2="740" y2="47.3"/><text class="t-tick" x="62" y="51.3453434204491" text-anchor="end">1,000</text><text class="t-tick" x="121.53846153846155" y="300" text-anchor="middle">1</text><text class="t-tick" x="276.1538461538462" y="300" text-anchor="middle">8</text><text class="t-tick" x="430.7692307692308" y="300" text-anchor="middle">64</text><text class="t-tick" x="585.3846153846154" y="300" text-anchor="middle">512</text><text class="t-tick" x="740.0" y="300" text-anchor="middle">4,096</text><line class="axis" x1="70" y1="280" x2="740" y2="280"/><polyline class="l1" points="70.0,262.6 544.4,47.7 740.0,47.7"/><text class="t-tick" x="203.22499042178268" y="188.2804663554529" text-anchor="start" transform="rotate(-33 203.2 188.3)">memory-bound: speed = bandwidth × intensity</text><text class="t-tick" x="740" y="37.71787846337128" text-anchor="end">compute roof: 989 TFLOP/s</text><line class="drop" x1="544.4" y1="47.7" x2="544.4" y2="280"/><text class="t-tick" x="550.4458432443407" y="272" text-anchor="start">ridge ≈ 295</text><g class="mark"><title>decode, batch 1: about 1 FLOP per byte, at most 3 TFLOP/s</title><circle class="s2 ring" cx="121.5" cy="239.3" r="6"/></g><text class="t-note" x="131.53846153846155" y="227.28196024332206" text-anchor="start">decode, batch 1</text><g class="mark"><title>decode, batch 64: about 64 FLOP per byte, at most 214 TFLOP/s</title><circle class="s2 ring" cx="430.8" cy="99.2" r="6"/></g><text class="t-note" x="440.7692307692308" y="121.20989972062759" text-anchor="start">decode, batch 64</text><g class="mark"><title>prefill, 512 tokens: about 512 FLOP per byte, at most 989 TFLOP/s</title><circle class="s1 ring" cx="585.4" cy="47.7" r="6"/></g><text class="t-note" x="595.3846153846154" y="73.71787846337128" text-anchor="start">prefill, 512 tokens</text><text class="t-tick" x="405.0" y="322" text-anchor="middle">arithmetic intensity: FLOPs per byte read from memory (log scale)</text><text class="t-tick" x="10" y="16" text-anchor="start">TFLOP/s</text></svg><figcaption>A roofline for an H100. On the left slope, speed is capped by memory. On the flat roof, it is capped by compute. Decode at batch 1 sits at the bottom left; prefill sits on the roof.</figcaption></figure>

Decode for a single sequence sits at about 1 calculation per byte, far down the slope. It uses well under 1% of what the GPU can compute. Prefill sits on the roof. That is the 190x gap from the measurement above, explained in one picture.

There is one more detail in the prefill numbers worth noticing:

| Prompt length | Prefill time | Tokens per second |
|---|---|---|
| 16 | 10.8 ms | 1,484 |
| 256 | 17.7 ms | 14,483 |
| 1,024 | 47.9 ms | 21,371 |
| 2,048 | 95.0 ms | 21,551 |
| 8,192 | 450.5 ms | 18,185 |

Short prompts are inefficient (16 tokens take about as long as one decode step, because the fixed overhead dominates). Throughput climbs as the prompt gets long enough to keep the GPU busy, then dips slightly at 8,192 tokens. Most of that dip comes from attention: every token looks at every earlier token, so that part of the work grows with the *square* of the length. We will meet attention properly in Part 2.

## The trick: write for many people at once

If decode wastes the weight trip on a single token, the fix almost suggests itself: **use each trip for many sequences**. Instead of one conversation, decode 8, or 64, or 128 at the same time. The weights are pulled in once per step and every sequence in the batch gets its next token.

Here is what that does, measured:

<figure class="fig"><svg viewBox="0 0 760 320" role="img" aria-label="Decode throughput grows almost linearly with batch size because the weight read is shared."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="264.0" x2="740" y2="264.0"/><text class="t-tick" x="56" y="268.0" text-anchor="end">0</text><line class="grid" x1="64" y1="217.2" x2="740" y2="217.2"/><text class="t-tick" x="56" y="221.2" text-anchor="end">1,000</text><line class="grid" x1="64" y1="170.4" x2="740" y2="170.4"/><text class="t-tick" x="56" y="174.39999999999998" text-anchor="end">2,000</text><line class="grid" x1="64" y1="123.6" x2="740" y2="123.6"/><text class="t-tick" x="56" y="127.6" text-anchor="end">3,000</text><line class="grid" x1="64" y1="76.8" x2="740" y2="76.8"/><text class="t-tick" x="56" y="80.79999999999998" text-anchor="end">4,000</text><line class="grid" x1="64" y1="30.0" x2="740" y2="30.0"/><text class="t-tick" x="56" y="34.0" text-anchor="end">5,000</text><line class="axis" x1="64" y1="264" x2="740" y2="264"/><g class="mark"><title>batch 1: 100 tokens/s, 10.0 ms per step</title><rect class="hit-area" x="64.0" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M94.25,264.0V263.34088819824257Q94.25,259.34088819824257 98.25,259.34088819824257H114.25Q118.25,259.34088819824257 118.25,263.34088819824257V264.0Z"/></g><text class="t-val" x="106.25" y="251.34088819824257" text-anchor="middle">99.6</text><text class="t-tick" x="106.25" y="284" text-anchor="middle">1</text><g class="mark"><title>batch 2: 190 tokens/s, 10.6 ms per step</title><rect class="hit-area" x="148.5" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M178.75,264.0V259.12841783897136Q178.75,255.12841783897136 182.75,255.12841783897136H198.75Q202.75,255.12841783897136 202.75,259.12841783897136V264.0Z"/></g><text class="t-val" x="190.75" y="247.12841783897136" text-anchor="middle">190</text><text class="t-tick" x="190.75" y="284" text-anchor="middle">2</text><g class="mark"><title>batch 4: 361 tokens/s, 11.1 ms per step</title><rect class="hit-area" x="233.0" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M263.25,264.0V251.08300029697827Q263.25,247.08300029697827 267.25,247.08300029697827H283.25Q287.25,247.08300029697827 287.25,251.08300029697827V264.0Z"/></g><text class="t-val" x="275.25" y="239.08300029697827" text-anchor="middle">361</text><text class="t-tick" x="275.25" y="284" text-anchor="middle">4</text><g class="mark"><title>batch 8: 670 tokens/s, 11.9 ms per step</title><rect class="hit-area" x="317.5" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M347.75,264.0V236.63128334359791Q347.75,232.63128334359791 351.75,232.63128334359791H367.75Q371.75,232.63128334359791 371.75,236.63128334359791V264.0Z"/></g><text class="t-val" x="359.75" y="224.63128334359791" text-anchor="middle">670</text><text class="t-tick" x="359.75" y="284" text-anchor="middle">8</text><g class="mark"><title>batch 16: 1,247 tokens/s, 12.8 ms per step</title><rect class="hit-area" x="402.0" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M432.25,264.0V209.66274263170567Q432.25,205.66274263170567 436.25,205.66274263170567H452.25Q456.25,205.66274263170567 456.25,209.66274263170567V264.0Z"/></g><text class="t-val" x="444.25" y="197.66274263170567" text-anchor="middle">1,247</text><text class="t-tick" x="444.25" y="284" text-anchor="middle">16</text><g class="mark"><title>batch 32: 2,120 tokens/s, 15.1 ms per step</title><rect class="hit-area" x="486.5" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M516.75,264.0V168.77820471286415Q516.75,164.77820471286415 520.75,164.77820471286415H536.75Q540.75,164.77820471286415 540.75,168.77820471286415V264.0Z"/></g><text class="t-val" x="528.75" y="156.77820471286415" text-anchor="middle">2,120</text><text class="t-tick" x="528.75" y="284" text-anchor="middle">32</text><g class="mark"><title>batch 64: 3,242 tokens/s, 19.7 ms per step</title><rect class="hit-area" x="571.0" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M601.25,264.0V116.27451800295245Q601.25,112.27451800295245 605.25,112.27451800295245H621.25Q625.25,112.27451800295245 625.25,116.27451800295245V264.0Z"/></g><text class="t-val" x="613.25" y="104.27451800295245" text-anchor="middle">3,242</text><text class="t-tick" x="613.25" y="284" text-anchor="middle">64</text><g class="mark"><title>batch 128: 4,367 tokens/s, 29.3 ms per step</title><rect class="hit-area" x="655.5" y="30" width="84.5" height="234"/><path class="s1 bar-mark" d="M685.75,264.0V63.61081165643168Q685.75,59.61081165643168 689.75,59.61081165643168H705.75Q709.75,59.61081165643168 709.75,63.61081165643168V264.0Z"/></g><text class="t-val" x="697.75" y="51.61081165643168" text-anchor="middle">4,367</text><text class="t-tick" x="697.75" y="284" text-anchor="middle">128</text><text class="t-tick" x="402.0" y="310" text-anchor="middle">sequences decoded together (batch size)</text><text class="t-tick" x="14" y="18" text-anchor="start">tokens per second</text></svg><figcaption>Tokens per second as more sequences are decoded together. Hover a bar to see the time per step.</figcaption></figure>

| Sequences at once | Time per step | Tokens per second |
|---|---|---|
| 1 | 10.0 ms | 100 |
| 8 | 11.9 ms | 670 |
| 32 | 15.1 ms | 2,120 |
| 64 | 19.7 ms | 3,242 |
| 128 | 29.3 ms | 4,367 |

Going from 1 sequence to 128 made each step about **3 times slower** but produced about **44 times more tokens**. Each user's answer streams a little slower; the GPU does vastly more useful work. This is **batching**, and it is the single biggest lever in LLM serving.

Why not batch a thousand sequences, then? Two things stop you:

1. **Each step gets slower.** Past a point, you move off the memory slope and onto the compute roof, and every user waits longer for every token.
2. **Each sequence needs its own memory.** Every conversation carries notes about everything said so far, and those notes have to sit in GPU memory next to the weights. On a real server, that memory, not compute, is what usually caps the batch.

Those notes are the **KV cache**. They are what Part 2 is about.

## Latency or throughput: pick your trade

Batching exposes the central trade-off in serving: **latency** (how fast one user gets their answer) against **throughput** (how many tokens the whole GPU produces per second).

| You care about | You want | Which means |
|---|---|---|
| one user, fast | small batches | low TPOT, but an expensive GPU sitting mostly idle |
| many users, cheap | large batches | high throughput per GPU, but each token arrives a bit later |
| a snappy first word | fast prefill, not stuck behind others | a scheduler that does not let long prompts block short ones (Part 3) |

Every serving system, vLLM included, is a machine for managing this trade: packing as many sequences into each step as memory allows, without letting any one user's experience fall apart.

## A hint of a very useful trick

One measurement from earlier deserves a second look. Prefilling **16 tokens took 10.8 ms**. Decoding **1 token took 10.0 ms**. Processing 16 tokens at once cost almost the same as processing one.

That suggests a clever move. What if a cheap helper *guessed* the next several tokens, and the big model *checked* all the guesses in a single pass? If most guesses are right, you get several tokens for the price of one step. This is **speculative decoding**, and it can be done so that the output is exactly what the big model would have written anyway. It gets all of Part 4, and it works for precisely the reason this article is about: decode leaves the GPU's compute idle.

## Summary

- An LLM writes by running the whole model once per token, in a loop.
- **Prefill** reads the whole prompt in one pass. It reuses every weight for many tokens, so it is **compute-bound** and fast per token. It sets **time to first token**.
- **Decode** writes one token per step. Each step reads all the weights for very little work, so it is **memory-bound** and slow per token. It sets **time per output token**, and it is where most of the time goes.
- Measured: reading 512 tokens took **27 ms**; writing 512 took **5.2 s**.
- **Batching** decodes many sequences per weight trip: 128 sequences gave **44x** the throughput for **3x** the step time.
- What limits the batch is usually memory for each sequence's notes, the **KV cache**. That is Part 2.

<details>
<summary>The benchmark script</summary>

```python
"""Prefill vs decode, measured. Qwen2.5-0.5B (BF16) on a GPU via PyTorch.
Random token ids: the content of the text does not change how long the math takes."""
import statistics, time
import torch
from transformers import AutoModelForCausalLM, DynamicCache

DEV = "mps"                      # "cuda" on an NVIDIA GPU
sync = torch.mps.synchronize     # torch.cuda.synchronize on NVIDIA
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16).to(DEV).eval()
V = model.config.vocab_size

def ids(b, n):
    return torch.randint(0, V, (b, n), device=DEV)

@torch.inference_mode()
def prefill(x):
    out = model(input_ids=x, past_key_values=DynamicCache(), logits_to_keep=1)
    return out.logits[:, -1:].argmax(-1), out.past_key_values

@torch.inference_mode()
def decode_step_ms(batch, context, steps=24):
    tok, cache = prefill(ids(batch, context))
    times = []
    for _ in range(steps):
        sync(); t = time.perf_counter()
        out = model(input_ids=tok, past_key_values=cache, logits_to_keep=1)
        tok, cache = out.logits[:, -1:].argmax(-1), out.past_key_values
        sync(); times.append(time.perf_counter() - t)
    return statistics.median(times[3:]) * 1000

x = ids(1, 512)
prefill(x); sync()
t = time.perf_counter(); prefill(x); sync()
print(f"prefill 512 tokens: {(time.perf_counter() - t) * 1000:.1f} ms")
for b in (1, 8, 32, 64, 128):
    ms = decode_step_ms(b, 512)
    print(f"batch {b:3d}: {ms:6.2f} ms per step, {b / ms * 1000:7.0f} tokens/s")
```

</details>

## References

1. NVIDIA, [H100 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/h100/) (3.35 TB/s memory bandwidth; 1,979 TFLOPS BF16 with sparsity, so about 989 dense).
2. S. Williams, A. Waterman, D. Patterson. *Roofline: an insightful visual performance model for multicore architectures.* Communications of the ACM, 2009.
3. A. Agrawal et al. [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310). OSDI 2024. ("Prefill iterations have high latency but saturate GPU compute... decode iterations have low latency but also low compute utilization.")
