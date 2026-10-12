---
title: "The Big Idea and the Residual Stream"
description: "The opening of A Mathematical Framework for Transformer Circuits, read slowly: what mechanistic interpretability is, why the paper studies tiny attention-only models, the whole model written as four equations with every symbol and shape, the residual stream as a shared channel that layers read from and write to, virtual weights, the bandwidth problem, and why a zero-layer transformer can only learn bigrams. Every equation derived by hand, then checked on real models."
part: 1
covers: "Introduction, Model Simplifications, High-Level Architecture, Virtual Weights, Subspaces, Zero-Layer Transformers"
date: 2026-10-12
tags: [interpretability, transformers, residual-stream, circuits]
---

A trained language model is a big pile of numbers. Even the small two-layer model we use in this part has 52 million of them. Nobody chose these numbers by hand: training found them. The model uses them to predict the next word, and it does this well. But if you ask "what rule did it learn?", the numbers do not say.

In December 2021, a team at Anthropic published a long web article, *A Mathematical Framework for Transformer Circuits*, that tried to read the rules out of the numbers. It took the smallest transformers it could, wrote them in a new but exactly equal way, and showed that each one is a **sum of simple pieces** you can read straight from the weights.

This part covers the paper's setup: what "mechanistic interpretability" means, which parts of a transformer the paper removes, the whole model in four equations, the **residual stream** that every layer reads from and writes to, and why a transformer with zero layers can only learn **bigrams** (word pairs).

Each idea follows the same pattern: the paper's own words as a highlighted screenshot in a teal box; a plain-English explanation, with a yellow box for every new word; the maths, derived one step at a time with a small example you can check by hand; and the same idea on a real model, with real code and its real output. The models are small public ones, run with the TransformerLens library on the CPU of an Apple M5 Pro (64 GB).

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word in plain language, the first time it appears. If you already know the word, skip the box.

## The paper {§Title}

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Title and authors
> [![The title, authors, affiliation (Anthropic) and date (Dec 22, 2021) of the paper](/img/papers/circuits/p1-title.png)](/img/papers/circuits/p1-title.png)
>
> **What it says:** the paper has twenty-five authors from Anthropic and is dated 22 December 2021. The marks after the names flag the core research contributors (Elhage, Nanda, Olsson), the core infrastructure contributors and the contact author (Chris Olah).
>
> **Why it matters:** the title promises a "framework", a way of writing transformers down so they can be read, not a new model. The findings (bigrams, skip-trigrams, induction heads) fall out of that new way of writing. It was also the first article of a long series from the same group: induction heads (2022), superposition (2022), sparse autoencoders (2023 and 2024) and circuit tracing (2025) all build on the vocabulary set out here.

A **circuit** is a small part of a network that does one understandable job, such as "after the word *Barack*, predict *Obama*". The word comes from earlier work on image models, the Distill *Circuits* thread, which found such parts inside an image network. A **framework** is a set of equations and names: the paper does not train a new kind of model, it writes the usual one in a different but equal form.

> [!DEFINITION] Token
> One unit of text that a language model reads. A token is a word, a piece of a word, or a symbol. " Barack" (with its leading space) is one token. "Dursley" might be three: "D", "urs", "ley". The model never sees letters, only token numbers.

> [!DEFINITION] Vocabulary
> The fixed list of all tokens a model knows. Each token has a number, its index in the list. The small models in this part have a vocabulary of 48,262 tokens; GPT-2 has 50,257.

## Why reverse-engineer a model? {§Introduction}

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Introduction
> [![The introduction paragraph defining mechanistic interpretability as reverse engineering, like turning binaries into source code](/img/papers/circuits/p1-intro-reverse.png)](/img/papers/circuits/p1-intro-reverse.png)
>
> **What it says:** large language models keep surprising their own builders: new abilities and harmful behaviours are sometimes found years after training. One answer is **mechanistic interpretability**: reverse-engineering "the detailed computations performed by transformers", the way a programmer turns binaries back into source code.
>
> **Why it matters:** it sets the goal. The goal is not a rough story ("the model looks at *Barack*") but the exact computation, in a form a person can read.

A **compiler** turns readable source code into a binary file that a computer runs but a person cannot read. If you only have the binary, you **reverse engineer** it: work back from the bytes to what the program does. A trained transformer is in the same place. Training is the compiler, the weights are the binary, and nobody ever wrote the source code. Mechanistic interpretability tries to recover it.

<figure class="fig"><svg viewBox="0 0 760 372" role="img" aria-label="Two rows. Top: source code is compiled into an unreadable binary file, and reverse engineering goes back from the binary to readable code. Bottom: training turns an unknown algorithm into 52 million weights, and mechanistic interpretability tries to go back from the weights to the algorithm."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="34.0" text-anchor="start">Ordinary software</text><rect class="box-1" x="24.0" y="48.0" width="230.0" height="84.0" rx="10"/><text class="t-note" x="139.0" y="70.0" text-anchor="middle">source code</text><text class="t-code" x="40.0" y="94.0" text-anchor="start">if word == "Barack":</text><text class="t-code" x="40.0" y="112.0" text-anchor="start">    say("Obama")</text><line class="edge" x1="262.0" y1="90.0" x2="466.0" y2="90.0" marker-end="url(#ah)"/><text class="t-tick" x="364.0" y="82.0" text-anchor="middle">compiler</text><rect class="box" x="474.0" y="48.0" width="262.0" height="84.0" rx="10"/><text class="t-note" x="605.0" y="70.0" text-anchor="middle">binary file</text><text class="t-code" x="490.0" y="94.0" text-anchor="start">cf fa ed fe 0c 00 00 01</text><text class="t-code" x="490.0" y="112.0" text-anchor="start">00 00 00 00 02 00 00 00 ...</text><path class="drop" d="M605,136 C605,166 139,166 139,138" marker-end="url(#ah)"/><text class="t-note" x="372.0" y="180.0" text-anchor="middle">reverse engineering</text><text class="t-title" x="24.0" y="204.0" text-anchor="start">A trained transformer</text><rect class="box-3" x="24.0" y="218.0" width="230.0" height="84.0" rx="10"/><text class="t-note" x="139.0" y="240.0" text-anchor="middle">the algorithm</text><text class="t-tick" x="40.0" y="264.0" text-anchor="start">(unknown: this is what</text><text class="t-tick" x="40.0" y="282.0" text-anchor="start"> we want to find)</text><line class="edge" x1="262.0" y1="260.0" x2="466.0" y2="260.0" marker-end="url(#ah)"/><text class="t-tick" x="364.0" y="252.0" text-anchor="middle">training</text><rect class="box" x="474.0" y="218.0" width="262.0" height="84.0" rx="10"/><text class="t-note" x="605.0" y="240.0" text-anchor="middle">weights</text><text class="t-tick" x="490.0" y="264.0" text-anchor="start">52,094,086 numbers:</text><text class="t-tick" x="490.0" y="282.0" text-anchor="start">0.044, -0.074, -0.164 ...</text><path class="drop" d="M605,306 C605,336 139,336 139,308" marker-end="url(#ah)"/><text class="t-note" x="372.0" y="350.0" text-anchor="middle">mechanistic interpretability</text></svg><figcaption>This is the comparison the paper makes. At the top, a compiler turns readable source code into an unreadable binary (the bytes shown are the real first bytes of the Python program), and reverse engineering goes back. At the bottom, training turns an unknown algorithm into weights (the 52 million numbers of the two-layer model we use; the first three are shown), and mechanistic interpretability tries to go back.</figcaption></figure>

> [!DEFINITION] Mechanistic interpretability
> Mechanistic interpretability means explaining a neural network by finding the exact steps it computes, in terms of its weights and its inner signals, the way you would explain a program by reading its code. "Mechanistic" means that we describe the mechanism, not only the behaviour.

> [!DEFINITION] Weights (parameters)
> Weights are the numbers inside a network that training adjusts. A weight matrix is a table of such numbers. "Reading a rule from the weights" means finding the rule by looking at these tables, without running the model on any text.

Why want this? The paper's answer is safety: if we could read the algorithm, we could explain known problems, find new ones, and perhaps foresee the problems of bigger models that do not exist yet.

### Start with the smallest models {§Introduction}

Modern models are huge, so the authors "start with the simplest possible models and work our way up": transformers with **two layers or fewer** and **only attention blocks**. The plan is to find simple patterns in these toy models, then look for them in big ones (the follow-up paper, *In-context Learning and Induction Heads*, did that).

The paper's summary of results has one finding per model size:

- **Zero layers:** the model learns **bigram** statistics, readable straight from the weights (end of this part).
- **One layer:** the model learns bigrams plus **skip-trigrams**, patterns "A … B C" where token A earlier and token B now make C more likely (Parts 2 to 4).
- **Two layers:** heads combine into **induction heads**, which continue a pattern seen earlier in the text (Parts 5 and 6).

<figure class="fig"><svg viewBox="0 0 760 460" role="img" aria-label="Three attention-only models side by side: zero layers (embed then unembed: bigram statistics, covered in Part 1), one layer of eight heads (bigrams plus skip-trigrams, Parts 2 to 4), and two layers of eight heads (heads compose into induction heads, Parts 5 and 6)."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="380.0" y="22.0" text-anchor="middle">The paper climbs a ladder of attention-only models</text><rect class="box-ghost" x="40.0" y="36.0" width="190.0" height="334.0" rx="12"/><text class="t-title" x="135.0" y="62.0" text-anchor="middle">zero layers</text><rect class="box" x="80.0" y="326.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="135.0" y="343.5" text-anchor="middle">tokens</text><line class="edge" x1="135.0" y1="324.0" x2="135.0" y2="310.0" marker-end="url(#ah)"/><rect class="box-1" x="80.0" y="280.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="135.0" y="297.5" text-anchor="middle">embed W<tspan baseline-shift="sub" style="font-size:72%">E</tspan></text><line class="edge" x1="135.0" y1="280.0" x2="135.0" y2="136.0" marker-end="url(#ah)"/><rect class="box-3" x="80.0" y="108.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="135.0" y="125.5" text-anchor="middle">unembed W<tspan baseline-shift="sub" style="font-size:72%">U</tspan></text><line class="edge" x1="135.0" y1="108.0" x2="135.0" y2="96.0" marker-end="url(#ah)"/><text class="t-tick" x="135.0" y="90.0" text-anchor="middle">logits</text><text class="t-note" x="135.0" y="398.0" text-anchor="middle">bigram statistics</text><text class="t-tick" x="135.0" y="416.0" text-anchor="middle">"Barack" → "Obama"</text><text class="t-muted" x="135.0" y="446.0" text-anchor="middle">Part 1 (this part)</text><rect class="box-ghost" x="285.0" y="36.0" width="190.0" height="334.0" rx="12"/><text class="t-title" x="380.0" y="62.0" text-anchor="middle">one layer</text><rect class="box" x="325.0" y="326.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="380.0" y="343.5" text-anchor="middle">tokens</text><line class="edge" x1="380.0" y1="324.0" x2="380.0" y2="310.0" marker-end="url(#ah)"/><rect class="box-1" x="325.0" y="280.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="380.0" y="297.5" text-anchor="middle">embed W<tspan baseline-shift="sub" style="font-size:72%">E</tspan></text><line class="edge" x1="380.0" y1="280.0" x2="380.0" y2="252.0" marker-end="url(#ah)"/><rect class="box-2" x="318.0" y="224.0" width="124.0" height="26.0" rx="6"/><text class="t-tick" x="380.0" y="241.5" text-anchor="middle">layer 0: 8 heads</text><circle class="node" cx="456" cy="237" r="9"/><text class="t-tick" x="456.0" y="241.5" text-anchor="middle">+</text><line class="edge" x1="380.0" y1="224.0" x2="380.0" y2="136.0" marker-end="url(#ah)"/><rect class="box-3" x="325.0" y="108.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="380.0" y="125.5" text-anchor="middle">unembed W<tspan baseline-shift="sub" style="font-size:72%">U</tspan></text><line class="edge" x1="380.0" y1="108.0" x2="380.0" y2="96.0" marker-end="url(#ah)"/><text class="t-tick" x="380.0" y="90.0" text-anchor="middle">logits</text><text class="t-note" x="380.0" y="398.0" text-anchor="middle">bigrams + skip-trigrams</text><text class="t-tick" x="380.0" y="416.0" text-anchor="middle">"A … B C", simple copying</text><text class="t-muted" x="380.0" y="446.0" text-anchor="middle">Parts 2 to 4</text><rect class="box-ghost" x="530.0" y="36.0" width="190.0" height="334.0" rx="12"/><text class="t-title" x="625.0" y="62.0" text-anchor="middle">two layers</text><rect class="box" x="570.0" y="326.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="625.0" y="343.5" text-anchor="middle">tokens</text><line class="edge" x1="625.0" y1="324.0" x2="625.0" y2="310.0" marker-end="url(#ah)"/><rect class="box-1" x="570.0" y="280.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="625.0" y="297.5" text-anchor="middle">embed W<tspan baseline-shift="sub" style="font-size:72%">E</tspan></text><line class="edge" x1="625.0" y1="280.0" x2="625.0" y2="252.0" marker-end="url(#ah)"/><rect class="box-2" x="563.0" y="224.0" width="124.0" height="26.0" rx="6"/><text class="t-tick" x="625.0" y="241.5" text-anchor="middle">layer 0: 8 heads</text><circle class="node" cx="701" cy="237" r="9"/><text class="t-tick" x="701.0" y="241.5" text-anchor="middle">+</text><line class="edge" x1="625.0" y1="224.0" x2="625.0" y2="196.0" marker-end="url(#ah)"/><rect class="box-2" x="563.0" y="168.0" width="124.0" height="26.0" rx="6"/><text class="t-tick" x="625.0" y="185.5" text-anchor="middle">layer 1: 8 heads</text><circle class="node" cx="701" cy="181" r="9"/><text class="t-tick" x="701.0" y="185.5" text-anchor="middle">+</text><line class="edge" x1="625.0" y1="168.0" x2="625.0" y2="136.0" marker-end="url(#ah)"/><rect class="box-3" x="570.0" y="108.0" width="110.0" height="26.0" rx="6"/><text class="t-tick" x="625.0" y="125.5" text-anchor="middle">unembed W<tspan baseline-shift="sub" style="font-size:72%">U</tspan></text><line class="edge" x1="625.0" y1="108.0" x2="625.0" y2="96.0" marker-end="url(#ah)"/><text class="t-tick" x="625.0" y="90.0" text-anchor="middle">logits</text><text class="t-note" x="625.0" y="398.0" text-anchor="middle">heads compose:</text><text class="t-tick" x="625.0" y="416.0" text-anchor="middle">induction heads</text><text class="t-muted" x="625.0" y="446.0" text-anchor="middle">Parts 5 and 6</text></svg><figcaption>This is the ladder the paper climbs. Each model adds one layer of eight attention heads. With zero layers, the model learns only bigrams. With one layer, it learns bigrams plus skip-trigrams. With two layers, heads can work together and form induction heads. This series follows the same ladder.</figcaption></figure>

> [!DEFINITION] Bigram
> A pair of tokens that follow each other, such as "Hong Kong". **Bigram statistics** are the probabilities of the next token given only the current token: how often "Kong" comes after "Hong". A bigram model ignores everything earlier in the text.

> [!DEFINITION] Layer and attention head
> A transformer is a stack of **layers**. Here each layer is a group of **attention heads** running side by side. A head lets each token look at earlier tokens and copy some information from them. Part 2 opens one up; in this part, a head is just "a box that reads the stream and adds something to it".

## The simplifications {§Model Simplifications}

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Model Simplifications
> [![The Model Simplifications paragraphs: attention-only transformers, no biases, no layer normalization](/img/papers/circuits/p1-simplifications.png)](/img/papers/circuits/p1-simplifications.png)
>
> **What it says:** the paper makes one big change: it removes the MLP layers, partly because the authors "had much less success in understanding MLP layers". It also makes two small changes. It removes biases and layer norm, because both fold into the weights.
>
> **Why it matters:** without MLPs and layer norm, almost every step is a matrix multiplication or an addition. That is what lets the model split into a sum of pieces.

A normal transformer layer (GPT-2, for example) has layer normalization, attention heads, an MLP, and biases inside each. The paper keeps only the attention heads.

<figure class="fig"><svg viewBox="0 0 760 410" role="img" aria-label="Left: a standard transformer block with layer norm, attention heads with biases, an addition, another layer norm, an MLP with biases and another addition. Right: the paper's toy block keeps only the attention heads and the addition. The MLP is dropped because it was too hard to read; layer norm and biases are dropped because they can be folded into the weights."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="190.0" y="24.0" text-anchor="middle">A standard block (GPT-2 style)</text><text class="t-title" x="570.0" y="24.0" text-anchor="middle">The paper's toy block</text><line class="edge-dim" x1="380.0" y1="14.0" x2="380.0" y2="384.0"/><line class="edge" x1="60.0" y1="372.0" x2="60.0" y2="44.0" marker-end="url(#ah)"/><text class="t-math" x="52.0" y="372.0" text-anchor="end">x</text><rect class="waste" x="100.0" y="340.0" width="210.0" height="28.0" rx="6"/><text class="t-tick" x="205.0" y="358.5" text-anchor="middle">layer norm</text><line class="edge" x1="60.0" y1="354.0" x2="100.0" y2="354.0"/><rect class="box-2" x="100.0" y="288.0" width="210.0" height="28.0" rx="6"/><text class="t-tick" x="205.0" y="306.5" text-anchor="middle">attention heads + biases</text><line class="edge" x1="60.0" y1="302.0" x2="100.0" y2="302.0"/><circle class="node" cx="60" cy="250" r="10"/><text class="t-tick" x="60.0" y="254.5" text-anchor="middle">+</text><text class="t-muted" x="80.0" y="255.0" text-anchor="start">add to stream</text><rect class="waste" x="100.0" y="184.0" width="210.0" height="28.0" rx="6"/><text class="t-tick" x="205.0" y="202.5" text-anchor="middle">layer norm</text><line class="edge" x1="60.0" y1="198.0" x2="100.0" y2="198.0"/><rect class="waste" x="100.0" y="132.0" width="210.0" height="28.0" rx="6"/><text class="t-tick" x="205.0" y="150.5" text-anchor="middle">MLP (+ biases)</text><line class="edge" x1="60.0" y1="146.0" x2="100.0" y2="146.0"/><circle class="node" cx="60" cy="94" r="10"/><text class="t-tick" x="60.0" y="98.5" text-anchor="middle">+</text><text class="t-muted" x="80.0" y="99.0" text-anchor="start">add to stream</text><line class="edge" x1="440.0" y1="372.0" x2="440.0" y2="44.0" marker-end="url(#ah)"/><rect class="box-2" x="480.0" y="236.0" width="200.0" height="28.0" rx="6"/><text class="t-tick" x="580.0" y="254.5" text-anchor="middle">attention heads</text><line class="edge" x1="440.0" y1="250.0" x2="480.0" y2="250.0"/><circle class="node" cx="440" cy="196" r="10"/><text class="t-tick" x="440.0" y="200.5" text-anchor="middle">+</text><text class="t-muted" x="460.0" y="201.0" text-anchor="start">add to stream</text><rect class="waste" x="470.0" y="287.0" width="12.0" height="12.0" rx="2"/><text class="t-note" x="490.0" y="298.0" text-anchor="start">no MLP: the big change</text><text class="t-muted" x="490.0" y="315.0" text-anchor="start">MLP layers were too hard to read</text><rect class="waste" x="470.0" y="327.0" width="12.0" height="12.0" rx="2"/><text class="t-note" x="490.0" y="338.0" text-anchor="start">no layer norm</text><text class="t-muted" x="490.0" y="355.0" text-anchor="start">folded into nearby weights</text><rect class="waste" x="470.0" y="367.0" width="12.0" height="12.0" rx="2"/><text class="t-note" x="490.0" y="378.0" text-anchor="start">no biases</text><text class="t-muted" x="490.0" y="395.0" text-anchor="start">folded in with an always-1 input</text><text class="t-tick" x="560.0" y="100.0" text-anchor="middle">one equation per layer:</text><text class="t-math" x="560.0" y="126.0" text-anchor="middle">x  ←  x  +  Σ h(x)</text></svg><figcaption>On the left is a standard GPT-2 style block, with layer norm, attention heads with biases, an addition, a second layer norm, an MLP with biases and a second addition. On the right, the paper's toy block keeps only the attention heads and the addition. The dashed red parts are removed.</figcaption></figure>

### 1. No MLP layers: the big change

> [!DEFINITION] MLP layer
> MLP stands for "multi-layer perceptron". In a transformer, an MLP is a block that takes each token's vector, multiplies it by a big matrix (to about 4 times the width), applies a non-linear function such as GeLU to every number, and multiplies by a second matrix to get back to the original width. It works on each token separately. In GPT-2 small, one MLP has 2 × 768 × 3,072 = 4.7 million weights, twice as many as the four 768 × 768 attention matrices of the same layer.

**Why remove it.** Attention heads raise new questions that studying them alone answers cleanly, and, honestly, MLPs were much harder to understand: their non-linear function breaks the "sum of pieces" picture.

**What it costs.** It costs a lot, and the paper calls it "a major weakness". These models are toys: they learn real behaviours (bigrams, copying, induction), but real models have MLPs. Part 7 covers how later work went back to them.

### 2. No biases: a free change

> [!DEFINITION] Bias
> A bias is an extra vector added after a matrix multiplication, as in $$y=Wx+b$$. It lets a layer output something even when its input is zero.

**Why remove it.** A bias can always be turned into one more column of weights. Add an extra input number that is always 1, and put the bias in the matching column:

$$
W x+b\;=\;\begin{bmatrix}W&b\end{bmatrix}\begin{bmatrix}x\\1\end{bmatrix}
$$

Read it in words: multiplying the bigger matrix by the longer vector gives $$W x$$ (from the first columns) plus $$b\times1$$ (from the last column). Here $$W$$ is any weight matrix (say $$m\times n$$), $$x$$ is the input ($$n$$ numbers), $$b$$ is the bias ($$m$$ numbers), and the new matrix $$[W\;b]$$ is $$m\times(n+1)$$.

A tiny check from our toy script, with $$W=\begin{bmatrix}2&1\\0&3\end{bmatrix}$$, $$b=(1,-1)$$ and $$x=(4,5)$$:

$$
\begin{aligned}
W x+b&=(13,15)+(1,-1)\\
&=(14,14)
\end{aligned}
$$

The bigger matrix times the longer vector gives the same answer:

$$
\begin{bmatrix}2&1&1\\0&3&-1\end{bmatrix}\begin{bmatrix}4\\5\\1\end{bmatrix}=\begin{bmatrix}14\\14\end{bmatrix}
$$

**What it costs.** It costs nothing. A model with biases is a model without biases plus one always-1 dimension. The paper adds that in attention-only models the biases mostly act like a fixed bias on the final scores; we will see exactly that in the real model below.

### 3. No layer normalization: almost free

> [!DEFINITION] Layer normalization (layer norm)
> Layer norm is a step that rescales each token's vector before a layer uses it: subtract the vector's mean, divide by its size (its standard deviation), then multiply by learned weights and add a learned bias. It keeps the numbers in a steady range during training.

**Why remove it.** The learned multiply-and-add part is linear, so it can be merged into the next matrix. What is left, "divide by the size", changes a vector's scale but not its direction: layer norm folds into nearby weights "up to a variable scaling".

**What it costs.** It costs very little. TransformerLens does this folding when it loads a model. We loaded `attn-only-2l` folded and unfolded, ran both on the same 31-token prompt, and compared every output log-probability:

```text
== 2. FOLDING LAYER NORM INTO THE WEIGHTS
  prompt has 31 tokens; largest change in any log-probability after folding: 7.82e-05
```

A change of 0.00008 is rounding noise: the folded model is the same model. The division by the size cannot be folded, but it is one positive number per token, so it never changes which token scores highest.

## The model in four equations {§High-Level Architecture}

Now the model itself. The paper studies **autoregressive, decoder-only** transformers like GPT-3: models that read text left to right and predict the next token. Here is the paper's own drawing.

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · High-Level Architecture
> [![The paper's diagram of a transformer with its four equations, from x0 = W_E t to T(t) = W_U x_{-1}](/img/papers/circuits/p1-architecture.png)](/img/papers/circuits/p1-architecture.png)
>
> **What it says:** a transformer "starts with a token embedding, followed by a series of residual blocks, and finally a token unembedding". The drawing writes this as four equations, bottom to top: embed, add every head's output, add the MLP's output, unembed.
>
> **Why it matters:** every block takes its input from the vertical line and puts its output back with a "+". That line is the **residual stream**, and the paper is built on it. Attention-only models have no MLP row.

We now go through the equations one at a time. The real model we use is `attn-only-2l` from TransformerLens: two layers, eight heads per layer, trained on web text and code in the style of the paper. Here are its sizes:

```python
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained('attn-only-2l', device='cpu')
for name in ['W_E', 'W_pos', 'W_U', 'W_Q', 'W_K', 'W_V', 'W_O']:
    print(name, tuple(getattr(model, name).shape))
```

```text
attn-only-2l: layers=2 heads/layer=8 d_model=512 d_head=64 n_vocab=48262 n_ctx=1024 attn_only=True params=52,094,086
  attn-only-2l W_E: (48262, 512)
  attn-only-2l W_pos: (1024, 512)
  attn-only-2l W_U: (512, 48262)
  attn-only-2l W_Q: (2, 8, 512, 64)
  attn-only-2l W_K: (2, 8, 512, 64)
  attn-only-2l W_V: (2, 8, 512, 64)
  attn-only-2l W_O: (2, 8, 64, 512)
  W_E + W_U hold 49,420,288 of 52,094,086 parameters
```

One warning. The paper writes vectors as **columns** ($$W x$$); TransformerLens writes them as **rows** ($$x W$$). So every TransformerLens matrix is the **transpose** of the paper's: the paper's $$W_{\mathrm{E}}$$ is 512 × 48,262, TransformerLens stores 48,262 × 512. In the text we always use the paper's shapes. Notice also that 95% of the weights sit in the embedding and unembedding; the sixteen heads are small.

> [!DEFINITION] Dimension and $$d_\text{model}$$
> A vector's **dimension** is how many numbers it has; each position is also called a dimension ("dimension 3"). $$d_\text{model}$$ is the width of the stream: 512 in our small models, 768 in GPT-2 small.

### Equation 1: the embedding

The model cannot do maths on words, so the first step turns each token into a vector of 512 numbers.

$$
x_0=W_{\mathrm{E}}\,t
$$

Here is what each symbol means:

- $$t$$ is the token, written as a **one-hot vector**. It has 48,262 numbers, all 0 except a single 1 at the token's index. Its shape is $$n_\text{vocab}=48{,}262$$.
- $$W_{\mathrm{E}}$$ is the **embedding matrix**, with one column per token. Its shape is $$d_\text{model}\times n_\text{vocab}=512\times48{,}262$$.
- $$x_0$$ is the result, the token's starting vector in the residual stream. Its shape is $$d_\text{model}=512$$.

> [!DEFINITION] One-hot vector
> A one-hot vector is a vector of zeros with a single 1. It is a way to write "token number $$k$$" so that a matrix can use it: the 1 sits at position $$k$$.

> [!DEFINITION] Embedding
> An embedding is the vector a model uses to stand for a token. The **embedding matrix** $$W_{\mathrm{E}}$$ stores one such vector per token in the vocabulary, learned in training.

**Derivation.** Why does multiplying by a one-hot vector give "the token's vector"? Write out the matrix-vector product for output number $$i$$:

$$
(W_{\mathrm{E}} t)_i=\sum_{j=1}^{n_\text{vocab}}(W_{\mathrm{E}})_{i j}\,t_j
$$

Every $$t_j$$ is 0 except $$t_k=1$$, where $$k$$ is our token. So every term of the sum is 0 except one:

$$
(W_{\mathrm{E}} t)_i=(W_{\mathrm{E}})_{i k}\cdot1=(W_{\mathrm{E}})_{i k}
$$

That is true for every row $$i$$. So $$W_{\mathrm{E}} t$$ is simply **column $$k$$ of $$W_{\mathrm{E}}$$**. The multiplication is a lookup.

**Toy example.** Take a vocabulary of four tokens (the, cat, dog, sat) and a stream of only 3 dimensions. Our toy script uses this $$W_{\mathrm{E}}$$ (3 × 4), with one column per token:

$$
W_{\mathrm{E}}=\begin{bmatrix}1&0&0&0\\0&1&1&0\\0&0&1&1\end{bmatrix}
$$

The token "the" is the one-hot vector

$$
t_\text{the}=\begin{bmatrix}1\\0\\0\\0\end{bmatrix}
$$

So its starting vector is

$$
x_0=W_{\mathrm{E}}\,t_\text{the}=\begin{bmatrix}1\\0\\0\end{bmatrix}
$$

<figure class="fig"><svg viewBox="0 0 760 270" role="img" aria-label="A 3 by 4 embedding matrix with one column per token (the, cat, dog, sat), multiplied by the one-hot vector 1, 0, 0, 0 for "the", gives the first column, 1, 0, 0: the vector of "the"."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="380.0" y="22.0" text-anchor="middle">Multiplying by a one-hot vector picks one column</text><text class="t-note" x="103.0" y="60.0" text-anchor="middle">the</text><text class="t-tick" x="149.0" y="60.0" text-anchor="middle">cat</text><text class="t-tick" x="195.0" y="60.0" text-anchor="middle">dog</text><text class="t-tick" x="241.0" y="60.0" text-anchor="middle">sat</text><rect class="box-1" x="77" y="67" width="48" height="140" rx="6"/><text class="t-muted" x="68.0" y="97.0" text-anchor="end">dim 1</text><rect class="box" x="82.0" y="72.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="102.0" y="97.0" text-anchor="middle">1</text><rect class="box" x="128.0" y="72.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="148.0" y="97.0" text-anchor="middle">0</text><rect class="box" x="174.0" y="72.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="194.0" y="97.0" text-anchor="middle">0</text><rect class="box" x="220.0" y="72.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="240.0" y="97.0" text-anchor="middle">0</text><text class="t-muted" x="68.0" y="143.0" text-anchor="end">dim 2</text><rect class="box" x="82.0" y="118.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="102.0" y="143.0" text-anchor="middle">0</text><rect class="box" x="128.0" y="118.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="148.0" y="143.0" text-anchor="middle">1</text><rect class="box" x="174.0" y="118.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="194.0" y="143.0" text-anchor="middle">1</text><rect class="box" x="220.0" y="118.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="240.0" y="143.0" text-anchor="middle">0</text><text class="t-muted" x="68.0" y="189.0" text-anchor="end">dim 3</text><rect class="box" x="82.0" y="164.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="102.0" y="189.0" text-anchor="middle">0</text><rect class="box" x="128.0" y="164.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="148.0" y="189.0" text-anchor="middle">0</text><rect class="box" x="174.0" y="164.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="194.0" y="189.0" text-anchor="middle">1</text><rect class="box" x="220.0" y="164.0" width="40.0" height="40.0" rx="6"/><text class="t-val" x="240.0" y="189.0" text-anchor="middle">1</text><text class="t-math" x="172.0" y="232.0" text-anchor="middle">W<tspan baseline-shift="sub" style="font-size:72%">E</tspan>   (3 × 4)</text><text class="t-muted" x="172.0" y="252.0" text-anchor="middle">one column per token</text><text class="t-title" x="308.0" y="142.0" text-anchor="middle">×</text><rect class="box-1" x="330.0" y="48.0" width="40.0" height="34.0" rx="6"/><text class="t-val" x="350.0" y="70.0" text-anchor="middle">1</text><text class="t-tick" x="380.0" y="70.0" text-anchor="start">the</text><rect class="box" x="330.0" y="88.0" width="40.0" height="34.0" rx="6"/><text class="t-val" x="350.0" y="110.0" text-anchor="middle">0</text><text class="t-tick" x="380.0" y="110.0" text-anchor="start">cat</text><rect class="box" x="330.0" y="128.0" width="40.0" height="34.0" rx="6"/><text class="t-val" x="350.0" y="150.0" text-anchor="middle">0</text><text class="t-tick" x="380.0" y="150.0" text-anchor="start">dog</text><rect class="box" x="330.0" y="168.0" width="40.0" height="34.0" rx="6"/><text class="t-val" x="350.0" y="190.0" text-anchor="middle">0</text><text class="t-tick" x="380.0" y="190.0" text-anchor="start">sat</text><text class="t-math" x="350.0" y="232.0" text-anchor="middle">t  (4)</text><text class="t-muted" x="350.0" y="252.0" text-anchor="middle">"the" is token 1</text><text class="t-title" x="492.0" y="142.0" text-anchor="middle">=</text><rect class="box-1" x="520.0" y="72.0" width="44.0" height="40.0" rx="6"/><text class="t-val" x="542.0" y="97.0" text-anchor="middle">1</text><rect class="box-1" x="520.0" y="118.0" width="44.0" height="40.0" rx="6"/><text class="t-val" x="542.0" y="143.0" text-anchor="middle">0</text><rect class="box-1" x="520.0" y="164.0" width="44.0" height="40.0" rx="6"/><text class="t-val" x="542.0" y="189.0" text-anchor="middle">0</text><text class="t-math" x="542.0" y="232.0" text-anchor="middle">x₀  (3)</text><text class="t-muted" x="542.0" y="252.0" text-anchor="middle">the vector of "the"</text><text class="t-tick" x="600.0" y="100.0" text-anchor="start">each output number is</text><text class="t-tick" x="600.0" y="118.0" text-anchor="start">row · t = 1 × (column 1)</text><text class="t-tick" x="600.0" y="136.0" text-anchor="start">+ 0 × (everything else)</text></svg><figcaption>This is Equation 1 worked by hand. The one-hot vector for "the" has a 1 in position 1, so the product keeps column 1 of W_E and throws away the other columns. The result (1, 0, 0) is the vector that stands for "the".</figcaption></figure>

**Why write it this way, and what is it useful for?** Code does a lookup, not a multiplication. The paper writes the multiplication on purpose: it makes the embedding a **linear map** like every other step, so it can be multiplied with later matrices. At the end of this part, the product $$W_{\mathrm{U}} W_{\mathrm{E}}$$ turns out to be a bigram table.

> [!DEFINITION] Linear map
> A linear map is a function $$f$$ that respects addition and scaling: $$f(a+b)=f(a)+f(b)$$ and $$f(c\,a)=c\,f(a)$$. Multiplying by a matrix is always a linear map, and every linear map between vectors can be written as a matrix. Linear maps are the easiest functions to take apart, because you can handle each piece of the input separately and add up the results.

The real model also adds a **position embedding** from a 1024 × 512 table `W_pos` ("this is position 31"). The paper leaves it out of its equations, so we treat it as one more piece in our checks.

### Equation 2: each layer adds to the stream

$$
x_{i+1}=x_i+\sum_{h\in H_i}h(x_i)
$$

Here is what each symbol means:

- $$x_i$$ is the residual stream vector before layer $$i$$. Its shape is 512 numbers, and there is one such vector per token position.
- $$H_i$$ is the set of attention heads in layer $$i$$: eight heads in our model.
- $$h(x_i)$$ is what head $$h$$ computes from the stream. Its shape is 512, the same as the stream, so it can be added.
- $$x_{i+1}$$ is the stream after the layer.

> [!DEFINITION] Residual stream
> The residual stream is the running vector, one per token position, that flows from the embedding to the unembedding. Every layer reads it and adds its own output to it. The paper's definition: the residual stream "is simply the sum of the output of all the previous layers and the original embedding".

In words: **every head reads the stream, computes something, and adds its result back**. Nothing is overwritten or multiplied. The plus sign is the **residual connection** from ResNets. (A head also looks at the stream at earlier positions, which is what attention does; Part 2 covers that. Here we only need that its output is added.)

The paper's third equation, $$x_{i+2}=x_{i+1}+m(x_{i+1})$$, is the same "read, compute, add" for an MLP $$m$$. Attention-only models simply do not have it.

### Equation 4: the unembedding

$$
T(t)=W_{\mathrm{U}}\,x_{-1}
$$

Here is what each symbol means:

- $$x_{-1}$$ is the stream after the last layer (index $$-1$$ means "the last one", as in Python). Its shape is 512.
- $$W_{\mathrm{U}}$$ is the **unembedding matrix**, with one row per token. Its shape is $$n_\text{vocab}\times d_\text{model}=48{,}262\times512$$.
- $$T(t)$$ is a vector of **logits**, one score per token in the vocabulary. Its shape is 48,262. ($$T$$ stands for "transformer", because the whole model is the function $$T$$.)

> [!DEFINITION] Logit
> A logit is a raw score for one possible next token, any real number. Higher means "more likely". Logits are turned into probabilities by the **softmax**.

> [!DEFINITION] Softmax
> Softmax is the function that turns a list of scores $$z_1,\dots,z_n$$ into probabilities: $$\operatorname{softmax}(z)_b=e^{z_b}/\sum_c e^{z_c}$$. Each result is between 0 and 1, and they add up to 1. Bigger scores get bigger probabilities.

> [!DEFINITION] Unembedding
> The unembedding is the last matrix of the model, $$W_{\mathrm{U}}$$. Row $$b$$ of $$W_{\mathrm{U}}$$ is a direction in the stream; the logit for token $$b$$ is the dot product of that row with the final stream vector. So "how much the stream points in direction $$b$$" becomes "how likely token $$b$$ is next".

**Toy example.** In our 3-dimensional toy, take this $$W_{\mathrm{U}}$$ (4 × 3), one row per possible next token:

$$
W_{\mathrm{U}}=\begin{bmatrix}0&0&1\\2&0&0\\1&0&0\\0&2&0\end{bmatrix}
\begin{matrix}\leftarrow\text{the}\\\leftarrow\text{cat}\\\leftarrow\text{dog}\\\leftarrow\text{sat}\end{matrix}
$$

With no layers at all, $$x_{-1}=x_0=(1,0,0)$$, the vector of "the". Each logit is one row of $$W_{\mathrm{U}}$$ times that vector:

$$
\begin{aligned}
W_{\mathrm{U}} x_0&=(0\cdot1,\;2\cdot1,\;1\cdot1,\;0\cdot1)\\
&=(0,2,1,0)
\end{aligned}
$$

The softmax turns these scores into probabilities. The four exponentials are $$e^0=1$$, $$e^2=7.389$$, $$e^1=2.718$$ and $$e^0=1$$, which add up to 12.107. Dividing each by 12.107:

```text
  logits = W_U x0 = [0.0, 2.0, 1.0, 0.0]
  softmax(logits) = [0.083, 0.61, 0.225, 0.083]  -> after "the": the 0.083, cat 0.610, dog 0.225, sat 0.083
```

So this toy model says: after "the", the next word is "cat" with probability 0.61.

**What it is useful for.** The unembedding is linear too. So if the stream is a sum of pieces, the logits are the same sum of pieces. The next section uses this.

Here is the whole real model in one picture, with the paper's shapes.

<figure class="fig"><svg viewBox="0 0 760 436" role="img" aria-label="The two-layer attention-only model as a vertical residual stream. The embedding writes x0 = W_E t (W_E is 512 by 48,262). Layer 0's eight heads read x0 and add their outputs to make x1. Layer 1's eight heads read x1 and add to make x2. The unembedding W_U (48,262 by 512) turns x2 into 48,262 logits, and softmax picks the next token."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="380.0" y="22.0" text-anchor="middle">attn-only-2l, with the paper's shapes</text><line class="path" x1="300.0" y1="392.0" x2="300.0" y2="144.0"/><line class="edge" x1="300.0" y1="160.0" x2="300.0" y2="142.0" marker-end="url(#ah)"/><rect class="box-1" x="180.0" y="392.0" width="240.0" height="30.0" rx="6"/><text class="t-tick" x="300.0" y="411.5" text-anchor="middle">embed: x₀ = W<tspan baseline-shift="sub" style="font-size:72%">E</tspan> t</text><rect class="box-3" x="180.0" y="110.0" width="240.0" height="30.0" rx="6"/><text class="t-tick" x="300.0" y="129.5" text-anchor="middle">unembed: logits = W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> x₂</text><rect class="box" x="180.0" y="40.0" width="240.0" height="30.0" rx="6"/><text class="t-tick" x="300.0" y="59.5" text-anchor="middle">softmax → next token</text><line class="edge" x1="300.0" y1="110.0" x2="300.0" y2="72.0" marker-end="url(#ah)"/><rect class="box-2" x="40.0" y="296.0" width="190.0" height="30.0" rx="6"/><text class="t-tick" x="135.0" y="315.5" text-anchor="middle">layer 0: 8 heads</text><line class="edge" x1="300.0" y1="366.0" x2="135.0" y2="366.0"/><line class="edge" x1="135.0" y1="366.0" x2="135.0" y2="328.0" marker-end="url(#ah)"/><path class="edge" d="M230,311 L287,276" marker-end="url(#ah)"/><circle class="node on" cx="300" cy="270" r="11"/><text class="t-tick" x="300.0" y="274.5" text-anchor="middle">+</text><text class="t-math" x="316.0" y="262.0" text-anchor="start">x₁</text><text class="t-muted" x="150.0" y="360.0" text-anchor="start">reads x₀</text><rect class="box-2" x="40.0" y="200.0" width="190.0" height="30.0" rx="6"/><text class="t-tick" x="135.0" y="219.5" text-anchor="middle">layer 1: 8 heads</text><line class="edge" x1="300.0" y1="254.0" x2="135.0" y2="254.0"/><line class="edge" x1="135.0" y1="254.0" x2="135.0" y2="232.0" marker-end="url(#ah)"/><path class="edge" d="M230,215 L287,180" marker-end="url(#ah)"/><circle class="node on" cx="300" cy="174" r="11"/><text class="t-tick" x="300.0" y="178.5" text-anchor="middle">+</text><text class="t-math" x="316.0" y="166.0" text-anchor="start">x₂</text><text class="t-muted" x="150.0" y="248.0" text-anchor="start">reads x₁</text><text class="t-math" x="316.0" y="384.0" text-anchor="start">x₀</text><text class="t-tick" x="440.0" y="411.0" text-anchor="start">W<tspan baseline-shift="sub" style="font-size:72%">E</tspan>: 512 × 48,262.  t: one-hot, 48,262 long.</text><text class="t-tick" x="440.0" y="322.0" text-anchor="start">each head adds a 512-vector to x₀</text><text class="t-tick" x="440.0" y="226.0" text-anchor="start">these heads read x₁, which already holds layer 0</text><text class="t-tick" x="440.0" y="129.0" text-anchor="start">W<tspan baseline-shift="sub" style="font-size:72%">U</tspan>: 48,262 × 512.  48,262 logits out.</text><text class="t-tick" x="440.0" y="59.0" text-anchor="start">on our prompt: "ley" with probability 0.988</text><text class="t-muted" x="440.0" y="270.0" text-anchor="start">the stream: 512 numbers per token,</text><text class="t-muted" x="440.0" y="288.0" text-anchor="start">never transformed, only added to</text></svg><figcaption>This is the two-layer model as the paper sees it. One vertical line, the residual stream, carries 512 numbers per token from the embedding to the unembedding. Each layer's eight heads read the stream and add their outputs at a + node. Nothing on the line itself is ever transformed.</figcaption></figure>

## The residual stream as a communication channel {§Virtual Weights and the Residual Stream}

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Virtual Weights and the Residual Stream as a Communication Channel
> [![The paragraph defining the residual stream as the sum of all previous outputs and a communication channel](/img/papers/circuits/p1-residual-stream.png)](/img/papers/circuits/p1-residual-stream.png)
>
> **What it says:** the residual stream "is simply the sum of the output of all the previous layers and the original embedding", and a "communication channel" that does no processing itself.
>
> **Why it matters:** the usual picture is a pipeline, each layer transforming the data in turn. The paper's picture is a shared whiteboard: every layer reads the same board and writes on it, and what is written stays for every later layer.

### Derivation: the stream is a sum

The paper's claim follows from Equation 2 by writing it out. Take our two-layer model. Layer 0 gives

$$
x_1=x_0+\sum_{h\in H_0}h(x_0)
$$

Layer 1 gives

$$
x_2=x_1+\sum_{h\in H_1}h(x_1)
$$

Now replace $$x_1$$ in the second line by the right side of the first line:

$$
\begin{aligned}
x_2&=\underbrace{x_0}_{\text{embedding}}\\
&\quad+\underbrace{\sum_{h\in H_0}h(x_0)}_{\text{8 heads of layer 0}}\\
&\quad+\underbrace{\sum_{h\in H_1}h(x_1)}_{\text{8 heads of layer 1}}
\end{aligned}
$$

That is the whole proof. The final stream is the embedding plus the output of every head, each added once. In the real model there are also the position embedding and one output bias per layer, so the final vector at one position has $$1+1+8+1+8+1=20$$ pieces.

**Why this is special.** In most networks the output of one layer goes through a non-linear function before the next layer sees it, so the pieces cannot be pulled apart. In the transformer, nothing is ever applied to the stream itself; layers only add to it. Even ResNets apply non-linear functions on their residual path.

**Toy example.** In our 3-dimensional toy, say head 1 writes $$(0,0.5,0)$$ and head 2 writes $$(0.2,0,-0.3)$$:

$$
\begin{aligned}
x_\text{final}&=\begin{bmatrix}1\\0\\0\end{bmatrix}+\begin{bmatrix}0\\0.5\\0\end{bmatrix}+\begin{bmatrix}0.2\\0\\-0.3\end{bmatrix}\\
&=\begin{bmatrix}1.2\\0.5\\-0.3\end{bmatrix}
\end{aligned}
$$

Now apply the unembedding. Because $$W_{\mathrm{U}}$$ is linear, $$W_{\mathrm{U}}(a+b+c)=W_{\mathrm{U}} a+W_{\mathrm{U}} b+W_{\mathrm{U}} c$$, so the logits split into one piece per writer:

$$
\begin{aligned}
W_{\mathrm{U}} x_\text{final}&=\underbrace{(0,2,1,0)}_{W_{\mathrm{U}} x_0}\\
&\quad+\underbrace{(0,0,0,1)}_{W_{\mathrm{U}} h_1}\\
&\quad+\underbrace{(-0.3,0.4,0.2,0)}_{W_{\mathrm{U}} h_2}\\
&=(-0.3,2.4,1.2,1.0)
\end{aligned}
$$

<figure class="fig"><svg viewBox="0 0 760 380" role="img" aria-label="Top: the embedding of "the" (1, 0, 0) plus what head 1 wrote (0, 0.5, 0) plus what head 2 wrote (0.2, 0, -0.3) equals the final vector (1.2, 0.5, -0.3). Bottom: multiplying each piece by the unembedding gives logits (0, 2, 1, 0), (0, 0, 0, 1) and (-0.3, 0.4, 0.2, 0), which add up to the final logits (-0.3, 2.4, 1.2, 1.0) for the, cat, dog and sat."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">1. The stream at the end is the sum of what everyone wrote</text><rect class="s1" x="60.0" y="60.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="83.0" y="77.0" text-anchor="middle">1</text><rect class="s1" x="60.0" y="90.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="83.0" y="107.0" text-anchor="middle">0</text><rect class="s1" x="60.0" y="120.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="83.0" y="137.0" text-anchor="middle">0</text><text class="t-tick" x="83.0" y="160.0" text-anchor="middle">x₀ (embedding)</text><text class="t-title" x="170.0" y="110.0" text-anchor="middle">+</text><rect class="s2" x="230.0" y="60.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="253.0" y="77.0" text-anchor="middle">0</text><rect class="s2" x="230.0" y="90.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="253.0" y="107.0" text-anchor="middle">0.5</text><rect class="s2" x="230.0" y="120.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="253.0" y="137.0" text-anchor="middle">0</text><text class="t-tick" x="253.0" y="160.0" text-anchor="middle">head 1 wrote</text><text class="t-title" x="340.0" y="110.0" text-anchor="middle">+</text><rect class="s3" x="400.0" y="60.0" width="46" height="26" rx="4" style="fill-opacity:0.47"/><text class="t-val" x="423.0" y="77.0" text-anchor="middle">0.2</text><rect class="s3" x="400.0" y="90.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="423.0" y="107.0" text-anchor="middle">0</text><rect class="s3" x="400.0" y="120.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="423.0" y="137.0" text-anchor="middle">-0.3</text><text class="t-tick" x="423.0" y="160.0" text-anchor="middle">head 2 wrote</text><text class="t-title" x="517.5" y="110.0" text-anchor="middle">=</text><rect class="s4" x="600.0" y="60.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="623.0" y="77.0" text-anchor="middle">1.2</text><rect class="s4" x="600.0" y="90.0" width="46" height="26" rx="4" style="fill-opacity:0.33"/><text class="t-val" x="623.0" y="107.0" text-anchor="middle">0.5</text><rect class="s4" x="600.0" y="120.0" width="46" height="26" rx="4" style="fill-opacity:0.24"/><text class="t-val" x="623.0" y="137.0" text-anchor="middle">-0.3</text><text class="t-tick" x="623.0" y="160.0" text-anchor="middle">x (final)</text><text class="t-title" x="24.0" y="206.0" text-anchor="start">2. Multiply by W<tspan baseline-shift="sub" style="font-size:72%">U</tspan>: the logits split into the same three pieces</text><text class="t-tick" x="50.0" y="248.0" text-anchor="end">the</text><text class="t-tick" x="50.0" y="278.0" text-anchor="end">cat</text><text class="t-tick" x="50.0" y="308.0" text-anchor="end">dog</text><text class="t-tick" x="50.0" y="338.0" text-anchor="end">sat</text><rect class="s1" x="60.0" y="232.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="83.0" y="249.0" text-anchor="middle">0</text><rect class="s1" x="60.0" y="262.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="83.0" y="279.0" text-anchor="middle">2</text><rect class="s1" x="60.0" y="292.0" width="46" height="26" rx="4" style="fill-opacity:0.38"/><text class="t-val" x="83.0" y="309.0" text-anchor="middle">1</text><rect class="s1" x="60.0" y="322.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="83.0" y="339.0" text-anchor="middle">0</text><text class="t-math" x="83.0" y="364.0" text-anchor="middle">W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> x₀</text><text class="t-title" x="170.0" y="296.0" text-anchor="middle">+</text><rect class="s2" x="230.0" y="232.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="253.0" y="249.0" text-anchor="middle">0</text><rect class="s2" x="230.0" y="262.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="253.0" y="279.0" text-anchor="middle">0</text><rect class="s2" x="230.0" y="292.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="253.0" y="309.0" text-anchor="middle">0</text><rect class="s2" x="230.0" y="322.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="253.0" y="339.0" text-anchor="middle">1</text><text class="t-math" x="253.0" y="364.0" text-anchor="middle">W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> h₁</text><text class="t-title" x="340.0" y="296.0" text-anchor="middle">+</text><rect class="s3" x="400.0" y="232.0" width="46" height="26" rx="4" style="fill-opacity:0.51"/><text class="t-val" x="423.0" y="249.0" text-anchor="middle">-0.3</text><rect class="s3" x="400.0" y="262.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="423.0" y="279.0" text-anchor="middle">0.4</text><rect class="s3" x="400.0" y="292.0" width="46" height="26" rx="4" style="fill-opacity:0.38"/><text class="t-val" x="423.0" y="309.0" text-anchor="middle">0.2</text><rect class="s3" x="400.0" y="322.0" width="46" height="26" rx="4" style="fill-opacity:0.10"/><text class="t-val" x="423.0" y="339.0" text-anchor="middle">0</text><text class="t-math" x="423.0" y="364.0" text-anchor="middle">W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> h₂</text><text class="t-title" x="517.5" y="296.0" text-anchor="middle">=</text><rect class="s4" x="600.0" y="232.0" width="46" height="26" rx="4" style="fill-opacity:0.17"/><text class="t-val" x="623.0" y="249.0" text-anchor="middle">-0.3</text><rect class="s4" x="600.0" y="262.0" width="46" height="26" rx="4" style="fill-opacity:0.65"/><text class="t-val" x="623.0" y="279.0" text-anchor="middle">2.4</text><rect class="s4" x="600.0" y="292.0" width="46" height="26" rx="4" style="fill-opacity:0.38"/><text class="t-val" x="623.0" y="309.0" text-anchor="middle">1.2</text><rect class="s4" x="600.0" y="322.0" width="46" height="26" rx="4" style="fill-opacity:0.33"/><text class="t-val" x="623.0" y="339.0" text-anchor="middle">1</text><text class="t-tick" x="623.0" y="364.0" text-anchor="middle">W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> x  (logits)</text></svg><figcaption>At the top, the final stream is the sum of what the embedding and the two heads wrote. At the bottom, multiplying each piece by W_U gives each piece's own logits, and they add up to the real logits. "cat" wins with 2.4, and we can see that 2.0 of it came from the embedding and 0.4 from head 2.</figcaption></figure>

**What it is useful for.** If the logits are a sum, you can ask **which piece pushed which token up**. In the toy, "cat" gets 2.4: 2.0 from the token itself, 0.4 from head 2. This is called **direct logit attribution**.

### Check it on the real model

Now test both claims on `attn-only-2l`. The prompt is the opening sentence of a well-known novel, cut off halfway through a name it has already used: *"Mr and Mrs Dursley, of number four, Privet Drive, were proud to say that they were perfectly normal. Mr and Mrs Durs"*. We split the stream at the last position into its 20 pieces and add them up (a shortened version of `circuits_part1.py`):

```python
logits, cache = model.run_with_cache(tokens)       # run once, keep every inner signal
pos = tokens.shape[1] - 1                          # the last position ("urs")
parts = [cache['hook_embed'][0, pos], cache['hook_pos_embed'][0, pos]]
for L in range(2):
    z = cache[f'blocks.{L}.attn.hook_z'][0, pos]   # [8, 64]: each head's result vector
    for h in range(8):
        parts.append(z[h] @ model.W_O[L, h])       # what head L.h writes: 512 numbers
    parts.append(model.b_O[L])                     # the layer's output bias
real = cache['blocks.1.hook_resid_post'][0, pos]   # the real final stream vector
print((sum(parts) - real).abs().max())
```

In words: `run_with_cache` runs the model once and keeps every inner signal. The first two pieces are the token and position embeddings. For each head, `hook_z` holds its 64-number result, and multiplying by its output matrix `W_O` gives the 512 numbers it writes (Part 2 explains `W_O`). Each layer also adds a bias `b_O`. Finally we compare the sum with the real vector.

```text
== 3. THE RESIDUAL STREAM AT THE LAST POSITION, TAKEN APART (attn-only-2l)
  last tokens: ['.', ' Mr', ' and', ' Mrs', ' D', 'urs']  -> model predicts 'ley'
  top 5: 'ley' 0.988, 'leys' 0.005, 'ki' 0.000, 'y' 0.000, 'a' 0.000
  20 parts; |sum of parts - real residual vector| max = 2.38e-06; ||x_final|| = 21.37
  first 4 coordinates of the real vector: [-0.256, -0.06, -0.569, -0.621]
  first 4 coordinates of the sum:         [-0.256, -0.06, -0.569, -0.621]
```

The model finishes "Dursley" ("ley", probability 0.988), and the 20 pieces add up to the real vector with a largest difference of 0.0000024, which is rounding noise. The stream really is a sum.

Now split the logit of "ley". One detail: before the unembedding, what is left of the last layer norm is "subtract the mean, divide by the size". Subtracting the mean is linear, so we apply it to each piece; the size is one number for the whole vector (0.945 here), so we divide every piece by it. Each piece's share is

$$
\begin{aligned}
\text{share of }p&=\frac{1}{0.945}\big(p-\text{mean}(p)\big)\\
&\qquad\cdot W_{\mathrm{U}}[\,:,\text{ley}\,]
\end{aligned}
$$

where $$W_{\mathrm{U}}[\,:,\text{ley}\,]$$ is the unembedding direction of "ley". The shares plus the unembedding bias add up to the real logit:

```text
  sum of all lines = 20.645; the model's real logit = 20.645
```

<figure class="fig"><svg viewBox="0 0 760 527" role="img" aria-label="Horizontal bars for the 21 parts of the residual stream at the last position, showing how much each adds to the logit of "ley". Head 1.6 adds 9.63, the layer 1 output bias 2.78, the unembedding bias 2.73, the token embedding 2.22; most other heads add less than 1.3 either way."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="24.0" text-anchor="start">Who pushed the logit of "ley"? Each part of the stream, one bar</text><text class="t-tick" x="24.0" y="44.0" text-anchor="start">attn-only-2l, last position of the prompt ("… Mrs D|urs"); the bars add up to the real logit 20.65</text><line class="grid" x1="270.0" y1="58.0" x2="270.0" y2="463.0"/><text class="t-muted" x="270.0" y="479.0" text-anchor="middle">-2</text><line class="grid" x1="330.0" y1="58.0" x2="330.0" y2="463.0"/><text class="t-muted" x="330.0" y="479.0" text-anchor="middle">0</text><line class="grid" x1="390.0" y1="58.0" x2="390.0" y2="463.0"/><text class="t-muted" x="390.0" y="479.0" text-anchor="middle">+2</text><line class="grid" x1="450.0" y1="58.0" x2="450.0" y2="463.0"/><text class="t-muted" x="450.0" y="479.0" text-anchor="middle">+4</text><line class="grid" x1="510.0" y1="58.0" x2="510.0" y2="463.0"/><text class="t-muted" x="510.0" y="479.0" text-anchor="middle">+6</text><line class="grid" x1="570.0" y1="58.0" x2="570.0" y2="463.0"/><text class="t-muted" x="570.0" y="479.0" text-anchor="middle">+8</text><line class="grid" x1="630.0" y1="58.0" x2="630.0" y2="463.0"/><text class="t-muted" x="630.0" y="479.0" text-anchor="middle">+10</text><line class="axis" x1="330.0" y1="58.0" x2="330.0" y2="463.0"/><text class="t-tick t-strong" x="256.0" y="77.5" text-anchor="end">token embedding</text><g class="mark"><title>embedding W_E t: +2.218</title><rect class="s1" x="330.0" y="67.0" width="66.5" height="13" rx="2"/></g><text class="t-val" x="402.5" y="77.5" text-anchor="start">+2.22</text><text class="t-tick" x="256.0" y="96.5" text-anchor="end">position embedding</text><g class="mark"><title>position W_pos: -0.102</title><rect class="s2" x="326.9" y="86.0" width="3.1" height="13" rx="2"/></g><text class="t-tick" x="336.0" y="96.5" text-anchor="start">-0.10</text><text class="t-tick" x="256.0" y="115.5" text-anchor="end">head 0.0</text><g class="mark"><title>head 0.0: +0.126</title><rect class="s1" x="330.0" y="105.0" width="3.8" height="13" rx="2"/></g><text class="t-tick" x="339.8" y="115.5" text-anchor="start">+0.13</text><text class="t-tick" x="256.0" y="134.5" text-anchor="end">head 0.1</text><g class="mark"><title>head 0.1: +0.805</title><rect class="s1" x="330.0" y="124.0" width="24.1" height="13" rx="2"/></g><text class="t-tick" x="360.1" y="134.5" text-anchor="start">+0.80</text><text class="t-tick" x="256.0" y="153.5" text-anchor="end">head 0.2</text><g class="mark"><title>head 0.2: +0.264</title><rect class="s1" x="330.0" y="143.0" width="7.9" height="13" rx="2"/></g><text class="t-tick" x="343.9" y="153.5" text-anchor="start">+0.26</text><text class="t-tick" x="256.0" y="172.5" text-anchor="end">head 0.3</text><g class="mark"><title>head 0.3: +0.828</title><rect class="s1" x="330.0" y="162.0" width="24.8" height="13" rx="2"/></g><text class="t-tick" x="360.8" y="172.5" text-anchor="start">+0.83</text><text class="t-tick" x="256.0" y="191.5" text-anchor="end">head 0.4</text><g class="mark"><title>head 0.4: -0.978</title><rect class="s2" x="300.7" y="181.0" width="29.3" height="13" rx="2"/></g><text class="t-tick" x="336.0" y="191.5" text-anchor="start">-0.98</text><text class="t-tick" x="256.0" y="210.5" text-anchor="end">head 0.5</text><g class="mark"><title>head 0.5: +0.795</title><rect class="s1" x="330.0" y="200.0" width="23.9" height="13" rx="2"/></g><text class="t-tick" x="359.9" y="210.5" text-anchor="start">+0.80</text><text class="t-tick" x="256.0" y="229.5" text-anchor="end">head 0.6</text><g class="mark"><title>head 0.6: +0.061</title><rect class="s1" x="330.0" y="219.0" width="1.8" height="13" rx="2"/></g><text class="t-tick" x="337.8" y="229.5" text-anchor="start">+0.06</text><text class="t-tick" x="256.0" y="248.5" text-anchor="end">head 0.7</text><g class="mark"><title>head 0.7: +0.227</title><rect class="s1" x="330.0" y="238.0" width="6.8" height="13" rx="2"/></g><text class="t-tick" x="342.8" y="248.5" text-anchor="start">+0.23</text><text class="t-tick" x="256.0" y="267.5" text-anchor="end">bias b<tspan baseline-shift="sub" style="font-size:72%">O</tspan> layer 0</text><g class="mark"><title>bias b_O layer 0: -0.281</title><rect class="s2" x="321.6" y="257.0" width="8.4" height="13" rx="2"/></g><text class="t-tick" x="336.0" y="267.5" text-anchor="start">-0.28</text><text class="t-tick" x="256.0" y="286.5" text-anchor="end">head 1.0</text><g class="mark"><title>head 1.0: +0.760</title><rect class="s1" x="330.0" y="276.0" width="22.8" height="13" rx="2"/></g><text class="t-tick" x="358.8" y="286.5" text-anchor="start">+0.76</text><text class="t-tick" x="256.0" y="305.5" text-anchor="end">head 1.1</text><g class="mark"><title>head 1.1: -0.356</title><rect class="s2" x="319.3" y="295.0" width="10.7" height="13" rx="2"/></g><text class="t-tick" x="336.0" y="305.5" text-anchor="start">-0.36</text><text class="t-tick" x="256.0" y="324.5" text-anchor="end">head 1.2</text><g class="mark"><title>head 1.2: -0.070</title><rect class="s2" x="327.9" y="314.0" width="2.1" height="13" rx="2"/></g><text class="t-tick" x="336.0" y="324.5" text-anchor="start">-0.07</text><text class="t-tick" x="256.0" y="343.5" text-anchor="end">head 1.3</text><g class="mark"><title>head 1.3: -0.387</title><rect class="s2" x="318.4" y="333.0" width="11.6" height="13" rx="2"/></g><text class="t-tick" x="336.0" y="343.5" text-anchor="start">-0.39</text><text class="t-tick" x="256.0" y="362.5" text-anchor="end">head 1.4</text><g class="mark"><title>head 1.4: +1.245</title><rect class="s1" x="330.0" y="352.0" width="37.3" height="13" rx="2"/></g><text class="t-tick" x="373.3" y="362.5" text-anchor="start">+1.24</text><text class="t-tick" x="256.0" y="381.5" text-anchor="end">head 1.5</text><g class="mark"><title>head 1.5: +0.095</title><rect class="s1" x="330.0" y="371.0" width="2.9" height="13" rx="2"/></g><text class="t-tick" x="338.9" y="381.5" text-anchor="start">+0.10</text><text class="t-tick t-strong" x="256.0" y="400.5" text-anchor="end">head 1.6</text><g class="mark"><title>head 1.6: +9.630</title><rect class="s1" x="330.0" y="390.0" width="288.9" height="13" rx="2"/></g><text class="t-val" x="624.9" y="400.5" text-anchor="start">+9.63</text><text class="t-tick" x="256.0" y="419.5" text-anchor="end">head 1.7</text><g class="mark"><title>head 1.7: +0.255</title><rect class="s1" x="330.0" y="409.0" width="7.7" height="13" rx="2"/></g><text class="t-tick" x="343.7" y="419.5" text-anchor="start">+0.26</text><text class="t-tick t-strong" x="256.0" y="438.5" text-anchor="end">bias b<tspan baseline-shift="sub" style="font-size:72%">O</tspan> layer 1</text><g class="mark"><title>bias b_O layer 1: +2.776</title><rect class="s1" x="330.0" y="428.0" width="83.3" height="13" rx="2"/></g><text class="t-val" x="419.3" y="438.5" text-anchor="start">+2.78</text><text class="t-tick t-strong" x="256.0" y="457.5" text-anchor="end">unembedding bias b<tspan baseline-shift="sub" style="font-size:72%">U</tspan></text><g class="mark"><title>unembedding bias b_U: +2.732</title><rect class="s1" x="330.0" y="447.0" width="82.0" height="13" rx="2"/></g><text class="t-val" x="418.0" y="457.5" text-anchor="start">+2.73</text><text class="t-tick" x="24.0" y="497.0" text-anchor="start">Blue pushes "ley" up, orange pushes it down. One head of the second layer (1.6) does almost half the work;</text><text class="t-tick" x="24.0" y="515.0" text-anchor="start">It looks back at "ley", the token that followed "urs" earlier in the prompt (attention 0.65), and copies it.</text></svg><figcaption>The logit of "ley" (20.65) split into 21 shares. One head of the second layer, head 1.6, adds 9.63, almost half. The token embedding adds 2.22. The output bias of layer 1 and the unembedding bias add about 2.7 each, which is the paper's remark that biases mostly act like a fixed bias on the logits.</figcaption></figure>

Head 1.6 does almost half of the work. Where is it looking?

```text
  head 1.6 at the last position attends to: position 6 'ley' 0.65, position 0 '<|BOS|>' 0.33, position 7 ',' 0.00
```

It looks back at the "ley" that followed "urs" the first time, and pushes "ley" up: a head that continues a pattern seen earlier in the text. That is an induction head, the subject of Part 6, found here with nothing more than addition.

### Linear and additive: the key property

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Virtual Weights and the Residual Stream as a Communication Channel
> [![The paragraph on the deeply linear, additive structure of the residual stream and its lack of a privileged basis](/img/papers/circuits/p1-linear.png)](/img/papers/circuits/p1-linear.png)
>
> **What it says:** every layer reads with one linear map and writes, by adding, with another. One consequence is that the stream "doesn't have a privileged basis".
>
> **Why it matters:** "linear" lets us multiply matrices across layers (virtual weights, next); "additive" lets us split the stream into pieces (what we just did).

What does "no privileged basis" mean? The model never looks at single dimensions of the stream on their own: it only reads through matrices and writes by adding. So we can rotate the whole stream, rotate every matrix that touches it to match, and the output stays the same.

> [!DEFINITION] Basis and privileged basis
> A **basis** is a set of directions used to write a vector as a list of numbers; the usual one is "dimension 1, dimension 2, ...". A basis is **privileged** when the model treats those exact directions specially, for example by applying a function to each number on its own (as an MLP's GeLU does). The residual stream has no such function, so no direction is special: dimension 7 means nothing by itself.

**Derivation.** Let $$R$$ be a rotation: a square matrix with $$R^\top R=I$$ (turning back undoes turning). Rotate the stream: $$x'=R x$$. A reader with matrix $$W$$ is replaced by $$W'=W R^\top$$. Then the reader sees

$$
\begin{aligned}
W'x'&=(W R^\top)(R x)\\
&=W(R^\top R)x\\
&=W I x=W x
\end{aligned}
$$

exactly what it saw before. A writer $$W_{\mathrm{O}}$$ becomes $$R\,W_{\mathrm{O}}$$. Every number in the stream changes; every output stays the same. (In our 3-dimensional toy, a 90-degree turn moves "cat" from $$(0,1,0)$$ to $$(0,0,1)$$, and its logits stay $$(0,0,0,2)$$.)

**Real model.** We rotated all 512 dimensions of `attn-only-1l` at random (keeping fixed the all-ones direction that layer norm uses), together with every matrix that reads or writes the stream:

```text
== 7. ROTATE THE RESIDUAL STREAM (attn-only-1l)
  rotation is orthogonal: max |R R^T - I| = 1.3e-06; keeps ones: 8.6e-08
  every weight that touches the stream changed (mean |change| in W_E: 0.218)
  largest change in any output logit: 9.6e-05
```

**What it is useful for.** It is a warning: "dimension 7 of the stream" has no meaning of its own, since a rotated model with a different dimension 7 behaves identically. Study the **matrices** that read and write the stream, and their products.

## Virtual weights {§Virtual Weights}

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Virtual Weights
> [![The paper's figure of layers reading and writing the residual stream, and virtual weights such as W_I^3 W_O^1](/img/papers/circuits/p1-virtual-weights-fig.png)](/img/papers/circuits/p1-virtual-weights-fig.png)
>
> **What it says:** each layer reads with $$W_{\mathrm{I}}$$ and writes with $$W_{\mathrm{O}}$$. Because both are linear, you can "multiply through" the stream: the product of one layer's output weights with a later layer's input weights, such as $$W_{\mathrm{I}}^2W_{\mathrm{O}}^1$$, is a **virtual weight** that connects the two layers directly. $$W_{\mathrm{I}}^3W_{\mathrm{O}}^1$$ connects layer 1 to layer 3, even with layer 2 in between.
>
> **Why it matters:** it finds which layers talk to which by multiplying two weight matrices, without running the model.

### Reading and writing, as equations

Give every layer two matrices:

- An **input matrix** $$W_{\mathrm{I}}$$ reads from the stream: the layer's input is $$W_{\mathrm{I}} x$$. Its shape is $$d_\text{in}\times d_\text{model}$$.
- An **output matrix** $$W_{\mathrm{O}}$$ writes to the stream: the layer adds $$W_{\mathrm{O}} a$$, where $$a$$ is what the layer computed. Its shape is $$d_\text{model}\times d_\text{out}$$.

(A head has three input matrices, $$W_{\mathrm{Q}}$$, $$W_{\mathrm{K}}$$ and $$W_{\mathrm{V}}$$; for now they are all "input weights", and Part 2 separates them.)

> [!DEFINITION] Projection (reading with a matrix)
> Multiplying the stream by a short, wide matrix, such as $$W_{\mathrm{V}}$$ (64 × 512), keeps only the parts of the stream the matrix "looks at" and drops the rest. The paper calls this reading "by performing a linear projection".

### Derivation: where virtual weights come from

Let layer 1 write $$W_{\mathrm{O}}^1a^1$$ into the stream. Later, layer 2 reads the stream. By the sum rule above, the stream it reads contains layer 1's output as one term:

$$
\begin{aligned}
x={}&x_0+W_{\mathrm{O}}^1a^1\\
&+\text{(everything else written}\\
&\quad\text{before layer 2)}
\end{aligned}
$$

Layer 2 multiplies this by its input matrix. Matrix multiplication spreads over a sum, so:

$$
\begin{aligned}
W_{\mathrm{I}}^2x={}&W_{\mathrm{I}}^2x_0\\
&+\big(W_{\mathrm{I}}^2W_{\mathrm{O}}^1\big)\,a^1\\
&+W_{\mathrm{I}}^2(\text{everything else})
\end{aligned}
$$

Look at the middle term: what layer 2 receives **from layer 1** is $$a^1$$ times one matrix, $$W_{\mathrm{I}}^2W_{\mathrm{O}}^1$$. It acts like a direct wire from layer 1 to layer 2, though no such wire exists in the code. The paper calls it a **virtual weight**.

$$
W_\text{virtual}^{1\to2}=W_{\mathrm{I}}^2\,W_{\mathrm{O}}^1
$$

Its shape is $$(d_\text{in}^2\times d_\text{model})$$ times $$(d_\text{model}\times d_\text{out}^1)$$, which gives $$d_\text{in}^2\times d_\text{out}^1$$. The 512-wide stream drops out of the shape. What is left connects layer 1's outputs straight to layer 2's inputs.

> [!DEFINITION] Virtual weight
> A virtual weight is the product $$W_{\mathrm{I}}^\text{later}W_{\mathrm{O}}^\text{earlier}$$ of a later layer's read matrix and an earlier layer's write matrix. It describes how strongly, and in which way, the later layer reads what the earlier one wrote. It is "virtual" because it is not stored anywhere in the model; it is computed from two real matrices.

**Toy example.** One writer and two readers on our 3-dimensional stream. The writer has $$W_{\mathrm{O}}=(0,0,1)^\top$$: it writes its single number into dimension 3. It sends $$a=0.7$$. Reader A has $$W_{\mathrm{I}}=(0,0,2)$$: it reads dimension 3. Reader B has $$W_{\mathrm{I}}=(1,0,0)$$: it reads dimension 1. The virtual weight of reader A is

$$
\begin{aligned}
W_{\mathrm{I}}^A W_{\mathrm{O}}&=(0,0,2)\cdot(0,0,1)^\top\\
&=2
\end{aligned}
$$

The virtual weight of reader B is

$$
\begin{aligned}
W_{\mathrm{I}}^B W_{\mathrm{O}}&=(1,0,0)\cdot(0,0,1)^\top\\
&=0
\end{aligned}
$$

Reader A receives $$2\times0.7=1.4$$. Reader B receives 0: the virtual weight says these two layers never talk, without running anything.

<figure class="fig"><svg viewBox="0 0 760 400" role="img" aria-label="A residual stream with three dimensions drawn as three lanes. A writer with W_O = (0, 0, 1) adds 0.7 into dimension 3. Reader A with W_I = (0, 0, 2) reads dimension 3 and gets 1.4: their virtual weight is 2. Reader B with W_I = (1, 0, 0) reads dimension 1 and gets 0: their virtual weight is 0."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="24.0" text-anchor="start">A writer and two readers, connected only through the stream</text><rect class="box" x="150" y="117" width="460" height="26" rx="13"/><text class="t-tick" x="140.0" y="134.0" text-anchor="end">dimension 1</text><rect class="box" x="150" y="157" width="460" height="26" rx="13"/><text class="t-tick" x="140.0" y="174.0" text-anchor="end">dimension 2</text><rect class="box-3" x="150" y="197" width="460" height="26" rx="13"/><text class="t-tick" x="140.0" y="214.0" text-anchor="end">dimension 3</text><text class="t-muted" x="150.0" y="100.0" text-anchor="start">residual stream (3 numbers) →</text><rect class="box" x="450.0" y="50.0" width="220.0" height="30.0" rx="6"/><text class="t-tick" x="560.0" y="69.5" text-anchor="middle">reader B: W<tspan baseline-shift="sub" style="font-size:72%">I</tspan> = [1, 0, 0]</text><line class="edge" x1="560.0" y1="117.0" x2="560.0" y2="82.0" marker-end="url(#ah)"/><text class="t-tick" x="548.0" y="106.0" text-anchor="end">reads dim 1: gets 0</text><rect class="box-2" x="200.0" y="270.0" width="200.0" height="30.0" rx="6"/><text class="t-tick" x="300.0" y="289.5" text-anchor="middle">writer: W<tspan baseline-shift="sub" style="font-size:72%">O</tspan> = [0, 0, 1]ᵀ</text><line class="edge" x1="300.0" y1="268.0" x2="300.0" y2="225.0" marker-end="url(#ah)"/><text class="t-tick" x="310.0" y="252.0" text-anchor="start">adds 0.7 to dim 3</text><text class="t-tick" x="300.0" y="320.0" text-anchor="middle">sends a = 0.7</text><rect class="box-1" x="450.0" y="270.0" width="220.0" height="30.0" rx="6"/><text class="t-tick" x="560.0" y="289.5" text-anchor="middle">reader A: W<tspan baseline-shift="sub" style="font-size:72%">I</tspan> = [0, 0, 2]</text><line class="edge" x1="560.0" y1="225.0" x2="560.0" y2="268.0" marker-end="url(#ah)"/><text class="t-note" x="570.0" y="252.0" text-anchor="start">reads 2 × 0.7 = 1.4</text><text class="t-tick" x="24.0" y="362.0" text-anchor="start">virtual weight, writer → A:  W<tspan baseline-shift="sub" style="font-size:72%">I</tspan> W<tspan baseline-shift="sub" style="font-size:72%">O</tspan> = [0, 0, 2] · [0, 0, 1]ᵀ = 2</text><text class="t-tick" x="24.0" y="386.0" text-anchor="start">virtual weight, writer → B:  W<tspan baseline-shift="sub" style="font-size:72%">I</tspan> W<tspan baseline-shift="sub" style="font-size:72%">O</tspan> = [1, 0, 0] · [0, 0, 1]ᵀ = 0   (they never talk)</text></svg><figcaption>This figure works out virtual weights by hand. The writer puts 0.7 into dimension 3. Reader A reads dimension 3 with weight 2 and gets 1.4, so its virtual weight with the writer is 2. Reader B reads only dimension 1 and gets nothing, so its virtual weight is 0.</figcaption></figure>

**Real model.** In `attn-only-2l`, take head 0.0 (layer 0, head 0). In the paper's shapes it reads with $$W_{\mathrm{V}}$$ (64 × 512) and writes with $$W_{\mathrm{O}}$$ (512 × 64). A head in layer 1 reads with its own $$W_{\mathrm{V}}$$ (64 × 512). The virtual weight from a layer-0 head to a layer-1 head is therefore 64 × 64, much smaller than either matrix:

```python
V = model.W_V[1, h2].T @ model.W_O[0, h1].T    # paper's W_V (64 x 512) times paper's W_O (512 x 64)
```

```text
== 5. READING, WRITING AND VIRTUAL WEIGHTS (attn-only-2l, paper notation: W_V is d_head x d_model)
  head 0.0 reads with W_V (64, 512) and writes with W_O (512, 64)
  W_O W_V is (512, 512) = 262,144 numbers, but its rank is only 64
  virtual weight W_V(1.h2) W_O(0.h1): shape (64, 64), one for each of the 64 head pairs
  largest size (Frobenius norm): head 0.2 -> head 1.6: 5.41; smallest: head 0.0 -> head 1.5: 1.16
  (raw sizes are not yet a fair score of how much two heads talk; Part 5 normalises them)
  the embedding as a "layer 0 writer": W_E is (512, 48262); head 0.0 reading it: W_V W_E is (64, 48262)
```

There are three things to see. There is one 64 × 64 virtual weight for each of the 64 pairs of heads; Part 5 turns their sizes into a fair score. The embedding is a writer too: $$W_{\mathrm{V}} W_{\mathrm{E}}$$ says what head 0.0 reads about each of the 48,262 tokens. And a head's own write-after-read matrix $$W_{\mathrm{O}} W_{\mathrm{V}}$$ is 512 × 512 but has **rank** only 64, because everything passes through the head's 64 numbers. That is where Part 2 starts.

> [!DEFINITION] Rank
> The rank of a matrix is the number of independent directions it can output. A 512 × 512 matrix built as (512 × 64) times (64 × 512) can only output vectors from a 64-dimensional space, so its rank is at most 64. A matrix with a rank much smaller than its size is called **low-rank**.

## Subspaces and bandwidth {§Subspaces and Residual Stream Bandwidth}

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Subspaces and Residual Stream Bandwidth
> [![The paper's figure of subspaces of the residual stream: shared, disjoint, and deleting information](/img/papers/circuits/p1-subspaces-fig.png)](/img/papers/circuits/p1-subspaces-fig.png)
>
> **What it says:** the stream has hundreds of dimensions in small models and tens of thousands in big ones, so "layers can send different information to different layers by storing it in different subspaces". Layers interact only through shared subspaces; with separate subspaces they do not interact. A layer can also **delete** information by reading a subspace and writing back the negative.
>
> **Why it matters:** it generalises the toy above: reader B did not hear the writer because they used different subspaces.

> [!DEFINITION] Subspace
> A subspace is a "flat" part of a vector space that passes through zero: all the vectors you can make by adding and scaling a few chosen directions. In our toy, "dimension 3 only" is a 1-dimensional subspace. A head that writes through a 512 × 64 matrix can only write into a subspace of at most 64 dimensions.

Once a layer writes something, it stays "unless another layer actively deletes it". So the stream's dimensions act like **memory**, or **bandwidth**: a limited number of lanes every message must share.

> [!DEFINITION] Bandwidth
> Bandwidth is how much information a channel can carry at once. For the residual stream, the bandwidth is its number of dimensions: 512 numbers per token in our small models.

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Subspaces and Residual Stream Bandwidth
> [![The paragraphs on residual stream bandwidth being in high demand, and memory management](/img/papers/circuits/p1-bandwidth.png)](/img/papers/circuits/p1-bandwidth.png)
>
> **What it says:** "we should expect residual stream bandwidth to be in very high demand": in the middle of a 50-layer model, about 100 times more neurons write before that point than the stream has dimensions. So some neurons and heads may do **"memory management"**, clearing dimensions by writing the negative of what they read.
>
> **Why it matters:** a crowded stream is hard to read directly, which is why the paper studies virtual weights (the channels between layers) instead.

We counted the same numbers for our models. "Head outputs" is the number of heads times 64 (each head computes 64 numbers before writing); "MLP neurons" is 3,072 per layer in GPT-2 small.

<figure class="fig"><svg viewBox="0 0 760 270" role="img" aria-label="Bar chart. attn-only-2l: a residual stream of 512 dimensions and 1,024 head output dimensions. gpt2 small: a residual stream of 768 dimensions, 9,216 head output dimensions and 36,864 MLP neurons, 60 times more than the stream."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="26.0" text-anchor="start">Many writers, one narrow stream (dimensions)</text><text class="t-note" x="24.0" y="71.0" text-anchor="start">attn-only-2l</text><text class="t-tick" x="240.0" y="73.0" text-anchor="end">residual stream</text><g class="mark"><title>attn-only-2l, residual stream: 512</title><rect class="s1" x="250" y="60" width="5.8" height="22" rx="3"/></g><text class="t-val" x="263.8" y="75.0" text-anchor="start">512</text><text class="t-tick" x="240.0" y="107.0" text-anchor="end">all head outputs</text><g class="mark"><title>attn-only-2l, all head outputs: 1,024</title><rect class="s2" x="250" y="94" width="11.6" height="22" rx="3"/></g><text class="t-val" x="269.6" y="109.0" text-anchor="start">1,024</text><text class="t-note" x="24.0" y="155.0" text-anchor="start">gpt2 (small)</text><text class="t-tick" x="240.0" y="157.0" text-anchor="end">residual stream</text><g class="mark"><title>gpt2 (small), residual stream: 768</title><rect class="s1" x="250" y="144" width="8.7" height="22" rx="3"/></g><text class="t-val" x="266.7" y="159.0" text-anchor="start">768</text><text class="t-tick" x="240.0" y="191.0" text-anchor="end">all head outputs</text><g class="mark"><title>gpt2 (small), all head outputs: 9,216</title><rect class="s2" x="250" y="178" width="104.6" height="22" rx="3"/></g><text class="t-val" x="362.6" y="193.0" text-anchor="start">9,216</text><text class="t-tick" x="240.0" y="225.0" text-anchor="end">all MLP neurons</text><g class="mark"><title>gpt2 (small), all MLP neurons: 36,864</title><rect class="s3" x="250" y="212" width="418.5" height="22" rx="3"/></g><text class="t-val" x="676.5" y="227.0" text-anchor="start">36,864</text><text class="t-tick" x="24.0" y="256.0" text-anchor="start">gpt2: 46,080 computed dimensions share 768 stream dimensions, 60 times more.</text></svg><figcaption>How many numbers are computed inside each model, against how wide its residual stream is. In attn-only-2l the 16 heads compute 1,024 numbers and share a 512-wide stream. In GPT-2 small, heads and MLP neurons compute 46,080 numbers and share a 768-wide stream: 60 times more.</figcaption></figure>

```text
== 6. BANDWIDTH
  attn-only-2l: residual stream 512 dims; heads write 1024 dims in total; MLP neurons 0; ratio 2.0x
  gpt2: residual stream 768 dims; heads write 9216 dims in total; MLP neurons 36864; ratio 60.0x
```

The toy model is mildly crowded (2 times). GPT-2 small, a real 12-layer model, is already at 60 times, on the way to the paper's "100 times" for a 50-layer model.

(The paper also notes, in a footnote, that in large models the embedding uses only a fairly small part of the stream. Checking this needs bigger models than ours, so we leave it for Part 7.)

The bandwidth paragraph also says the model is "somehow communicating in superposition". The paper only names the idea; it became the subject of *Toy Models of Superposition* and of Part 7.

> [!DEFINITION] Superposition
> Superposition means storing more separate pieces of information than a vector has dimensions, by giving each piece a direction that is almost, but not exactly, at right angles to the others. It saves space, at the cost of a little interference between pieces.

## The zero-layer transformer {§Zero-Layer Transformers}

Now we can read the first real result of the paper. It is short, and everything above was needed for it.

> [!PAPER] Elhage et al. (2021), A Mathematical Framework for Transformer Circuits · Zero-Layer Transformers
> [![The zero-layer section: T = W_U W_E approximates the bigram log-likelihood; the direct path; Barack followed by Obama](/img/papers/circuits/p1-zero-layer.png)](/img/papers/circuits/p1-zero-layer.png)
>
> **What it says:** a zero-layer model is $$T=W_{\mathrm{U}} W_{\mathrm{E}}$$, and the best it can do is "approximate the bigram log-likelihood". The same term sits inside every transformer as the **direct path**, often storing pairs like "Barack" followed by "Obama".
>
> **Why it matters:** it is the simplest circuit: one matrix product, read straight from the weights.

### Derivation 1: the model is one matrix

With no layers, Equation 2 never runs, so the last stream is the first stream: $$x_{-1}=x_0$$. Put Equation 1 into Equation 4:

$$
\begin{aligned}
T(t)&=W_{\mathrm{U}}\,x_0\\
&=W_{\mathrm{U}}\,(W_{\mathrm{E}}\,t)\\
&=(W_{\mathrm{U}} W_{\mathrm{E}})\,t
\end{aligned}
$$

The last step only moves the brackets, which is allowed for matrix products. So the whole model is a single matrix $$W_{\mathrm{U}} W_{\mathrm{E}}$$ applied to the one-hot token. The paper drops the $$t$$ and writes $$T=W_{\mathrm{U}} W_{\mathrm{E}}$$.

Here are the shapes:

$$
\begin{aligned}
&\underbrace{W_{\mathrm{U}}}_{48{,}262\times512}\;\underbrace{W_{\mathrm{E}}}_{512\times48{,}262}\\
&=\underbrace{W_{\mathrm{U}} W_{\mathrm{E}}}_{48{,}262\times48{,}262}
\end{aligned}
$$

By the one-hot rule, multiplying by $$t$$ picks one **column**: column $$a$$ of $$W_{\mathrm{U}} W_{\mathrm{E}}$$ is the list of next-token logits when the current token is $$a$$. The matrix is a **table** with one row per next token and one column per current token: the shape of a bigram table.

**Toy example.** Our toy's table is $$W_{\mathrm{U}} W_{\mathrm{E}}$$ = (4 × 3)(3 × 4) = 4 × 4. The script computed it:

```text
  the whole table W_U W_E (row = next token, column = current token):
     the: [0.0, 0.0, 1.0, 1.0]
     cat: [2.0, 0.0, 0.0, 0.0]
     dog: [1.0, 0.0, 0.0, 0.0]
     sat: [0.0, 2.0, 2.0, 0.0]
```

Check one column by hand: column "the" is $$W_{\mathrm{U}}(1,0,0)^\top$$, the first column of $$W_{\mathrm{U}}$$, which is $$(0,2,1,0)$$: the logits we computed for Equation 4. This toy table has learned "the → cat", "cat → sat" and "dog → sat".

<figure class="fig"><svg viewBox="0 0 760 310" role="img" aria-label="A 4 by 4 table of logits W_U W_E for the toy model, rows are the next token and columns the current token. The column for "the" is 0, 2, 1, 0; its softmax gives the 0.083, cat 0.610, dog 0.225, sat 0.083."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="24.0" text-anchor="start">The zero-layer model is one table: W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> W<tspan baseline-shift="sub" style="font-size:72%">E</tspan></text><text class="t-muted" x="212.0" y="42.0" text-anchor="middle">current token (column)</text><text class="t-note" x="143.0" y="66.0" text-anchor="middle">the</text><text class="t-tick" x="189.0" y="66.0" text-anchor="middle">cat</text><text class="t-tick" x="235.0" y="66.0" text-anchor="middle">dog</text><text class="t-tick" x="281.0" y="66.0" text-anchor="middle">sat</text><text class="t-muted" x="56.0" y="172.0" text-anchor="end">next</text><text class="t-muted" x="56.0" y="188.0" text-anchor="end">token</text><text class="t-muted" x="56.0" y="204.0" text-anchor="end">(row)</text><text class="t-tick" x="110.0" y="103.0" text-anchor="end">the</text><g class="mark"><title>after the, logit of the: 0</title><rect class="cell" x="122" y="78" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="143.0" y="103.0" text-anchor="middle">0</text><g class="mark"><title>after cat, logit of the: 0</title><rect class="cell" x="168" y="78" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="189.0" y="103.0" text-anchor="middle">0</text><g class="mark"><title>after dog, logit of the: 1</title><rect class="cell" x="214" y="78" width="42" height="42" rx="5" style="fill-opacity:0.28"/></g><text class="t-val" x="235.0" y="103.0" text-anchor="middle">1</text><g class="mark"><title>after sat, logit of the: 1</title><rect class="cell" x="260" y="78" width="42" height="42" rx="5" style="fill-opacity:0.28"/></g><text class="t-val" x="281.0" y="103.0" text-anchor="middle">1</text><text class="t-tick" x="110.0" y="149.0" text-anchor="end">cat</text><g class="mark"><title>after the, logit of cat: 2</title><rect class="cell" x="122" y="124" width="42" height="42" rx="5" style="fill-opacity:0.48"/></g><text class="t-val" x="143.0" y="149.0" text-anchor="middle">2</text><g class="mark"><title>after cat, logit of cat: 0</title><rect class="cell" x="168" y="124" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="189.0" y="149.0" text-anchor="middle">0</text><g class="mark"><title>after dog, logit of cat: 0</title><rect class="cell" x="214" y="124" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="235.0" y="149.0" text-anchor="middle">0</text><g class="mark"><title>after sat, logit of cat: 0</title><rect class="cell" x="260" y="124" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="281.0" y="149.0" text-anchor="middle">0</text><text class="t-tick" x="110.0" y="195.0" text-anchor="end">dog</text><g class="mark"><title>after the, logit of dog: 1</title><rect class="cell" x="122" y="170" width="42" height="42" rx="5" style="fill-opacity:0.28"/></g><text class="t-val" x="143.0" y="195.0" text-anchor="middle">1</text><g class="mark"><title>after cat, logit of dog: 0</title><rect class="cell" x="168" y="170" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="189.0" y="195.0" text-anchor="middle">0</text><g class="mark"><title>after dog, logit of dog: 0</title><rect class="cell" x="214" y="170" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="235.0" y="195.0" text-anchor="middle">0</text><g class="mark"><title>after sat, logit of dog: 0</title><rect class="cell" x="260" y="170" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="281.0" y="195.0" text-anchor="middle">0</text><text class="t-tick" x="110.0" y="241.0" text-anchor="end">sat</text><g class="mark"><title>after the, logit of sat: 0</title><rect class="cell" x="122" y="216" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="143.0" y="241.0" text-anchor="middle">0</text><g class="mark"><title>after cat, logit of sat: 2</title><rect class="cell" x="168" y="216" width="42" height="42" rx="5" style="fill-opacity:0.48"/></g><text class="t-val" x="189.0" y="241.0" text-anchor="middle">2</text><g class="mark"><title>after dog, logit of sat: 2</title><rect class="cell" x="214" y="216" width="42" height="42" rx="5" style="fill-opacity:0.48"/></g><text class="t-val" x="235.0" y="241.0" text-anchor="middle">2</text><g class="mark"><title>after sat, logit of sat: 0</title><rect class="cell" x="260" y="216" width="42" height="42" rx="5" style="fill-opacity:0.08"/></g><text class="t-val" x="281.0" y="241.0" text-anchor="middle">0</text><rect class="box-on" x="120" y="76" width="46" height="184" rx="6" style="fill:none"/><text class="t-tick" x="212.0" y="286.0" text-anchor="middle">the "the" column = logits after "the"</text><text class="t-note" x="470.0" y="66.0" text-anchor="start">softmax of that column</text><text class="t-tick" x="500.0" y="103.0" text-anchor="end">the</text><g class="mark"><title>P(the | the) = 0.083</title><rect class="s1" x="510" y="90" width="16.6" height="20" rx="3"/></g><text class="t-val" x="534.6" y="105.0" text-anchor="start">0.083</text><text class="t-tick" x="500.0" y="149.0" text-anchor="end">cat</text><g class="mark"><title>P(cat | the) = 0.610</title><rect class="s1" x="510" y="136" width="122.0" height="20" rx="3"/></g><text class="t-val" x="640.0" y="151.0" text-anchor="start">0.610</text><text class="t-tick" x="500.0" y="195.0" text-anchor="end">dog</text><g class="mark"><title>P(dog | the) = 0.225</title><rect class="s1" x="510" y="182" width="45.0" height="20" rx="3"/></g><text class="t-val" x="563.0" y="197.0" text-anchor="start">0.225</text><text class="t-tick" x="500.0" y="241.0" text-anchor="end">sat</text><g class="mark"><title>P(sat | the) = 0.083</title><rect class="s1" x="510" y="228" width="16.6" height="20" rx="3"/></g><text class="t-val" x="534.6" y="243.0" text-anchor="start">0.083</text><text class="t-tick" x="470.0" y="286.0" text-anchor="start">probabilities of the next token</text></svg><figcaption>The toy zero-layer model is one 4 by 4 table, W_U W_E. Each column is a current token and lists the scores of every possible next token. The softmax of the "the" column gives the probabilities after "the": cat 0.610, dog 0.225, and 0.083 for the other two.</figcaption></figure>

### Derivation 2: the best table is the log of the bigram probabilities

Why does the paper say the best $$W_{\mathrm{U}} W_{\mathrm{E}}$$ is "the bigram log-likelihood"? Here is the reasoning in three steps.

**Step 1: what training rewards.** A language model is trained to make the real next token likely. For one current token $$a$$, the training loss is the average of $$-\log q(b\mid a)$$ over all the times a token $$b$$ followed $$a$$ in the training text. Here $$q(b\mid a)$$ is the model's probability for $$b$$, the softmax of column $$a$$.

> [!DEFINITION] Cross-entropy loss
> Cross-entropy loss is the usual training loss of a language model. It is the average of $$-\log(\text{probability the model gave to the real next token})$$. If the model gives the real token probability 1, the loss is 0. If it gives it a small probability, the loss is large.

**Step 2: the best possible guess.** Let $$P(b\mid a)$$ be the true fraction of times $$b$$ follows $$a$$ in the data. A standard fact of probability (Gibbs' inequality) says that the average of $$-\log q$$, taken over data drawn from $$P$$, is smallest when $$q=P$$. In words: the loss is lowest when the model's probabilities equal the real frequencies. So the best zero-layer model has

$$
\begin{aligned}
&\operatorname{softmax}\big(\text{column }a\text{ of }W_{\mathrm{U}} W_{\mathrm{E}}\big)_b\\
&=P(b\mid a)
\end{aligned}
$$

**Step 3: undo the softmax.** Which logits give those probabilities? Try $$z_b=\log P(b\mid a)+c$$ for any number $$c$$:

$$
\begin{aligned}
\operatorname{softmax}(z)_b&=\frac{e^{\log P(b\mid a)+c}}{\sum_{b'}e^{\log P(b'\mid a)+c}}\\
&=\frac{e^c\,P(b\mid a)}{e^c\sum_{b'}P(b'\mid a)}\\
&=\frac{P(b\mid a)}{1}\\
&=P(b\mid a)
\end{aligned}
$$

The $$e^c$$ cancels, and the probabilities in the bottom add up to 1. So the best column is the **log of the bigram probabilities**, plus any constant. That is what "bigram log-likelihood" means.

> [!DEFINITION] Log-likelihood
> A log-likelihood is the logarithm of a probability. "Bigram log-likelihood" is $$\log P(b\mid a)$$: the log of the probability that token $$b$$ comes after token $$a$$. Logs turn small probabilities into large negative numbers, which is the natural scale for logits.

**Toy example.** Take the nine-word text "the cat sat . the cat ran . the dog sat .". The word "the" is followed twice by "cat" and once by "dog":

```text
== A. BIGRAMS IN A TINY CORPUS
  corpus: the cat sat . the cat ran . the dog sat .
  words that follow "the": ['cat', 'cat', 'dog']  counts {'cat': 2, 'dog': 1}
  P(next | the): {'cat': 0.667, 'dog': 0.333}
  log P(next | the): {'cat': -0.405, 'dog': -1.099}
  softmax(log P + 0.0) = [0.667, 0.333]  (adding the same number to every logit changes nothing)
  softmax(log P + 5.0) = [0.667, 0.333]  (adding the same number to every logit changes nothing)
```

The best column for "the" holds $$\log(2/3)=-0.405$$ for "cat" and $$\log(1/3)=-1.099$$ for "dog" (and very negative numbers for unseen words). Adding 5 to both changes nothing, as Step 3 promised.

**The catch.** A full bigram table for our vocabulary has $$48{,}262^2\approx2.33$$ billion entries, but $$W_{\mathrm{U}} W_{\mathrm{E}}$$ passes through a 512-wide middle, so its rank is at most 512. It cannot hold an arbitrary table. That is why the paper says "approximate": the model stores the best **low-rank** version it can.

### The direct path in a real model

The term $$W_{\mathrm{U}} W_{\mathrm{E}}$$ is not only the zero-layer model. In every transformer the stream is "embedding plus everything the layers add", so the logits always contain $$W_{\mathrm{U}} x_0=W_{\mathrm{U}} W_{\mathrm{E}} t$$: the token's embedding going straight to the unembedding. The paper calls it the **direct path**.

> [!DEFINITION] Direct path
> The direct path is the term $$W_{\mathrm{U}} W_{\mathrm{E}}$$ inside any transformer. It is the route from the embedding straight to the unembedding, skipping every layer. It can only use the current token, so it can only store bigram-like knowledge.

In a model with layers, heads can predict part of the bigram table, so the direct path holds a kind of "residual": pairs no general rule explains, like "Barack" followed by "Obama". Let us read it from the weights of the one-layer `attn-only-1l`: take one token's embedding, multiply by $$W_{\mathrm{U}}$$, list the five highest of the 48,262 scores.

```python
t = model.to_single_token(' Barack')
row = model.W_E[t] @ model.W_U          # 48,262 scores: "after ' Barack', which token?"
print(model.to_str_tokens(row.topk(5).indices))
```

No text is run through the model; this reads the weights only.

```text
== 4. THE DIRECT PATH W_U W_E (top 5 "next tokens" from the token alone)
  attn-only-1l
       ' Barack' -> ' Obama', ' Hussein', 'lung', 'Obama', 'hurst'   (rank of the token itself: 33)
       ' United' -> ' States', ' Nations', ' Kingdom', ' Methodist', ' Arab'   (rank of the token itself: 2623)
          ' New' -> ' Zealand', ' York', ' Yorker', ' Orleans', ' Testament'   (rank of the token itself: 33197)
         ' Hong' -> ' Kong', 'wei', 'chen', 'qi', 'yang'   (rank of the token itself: 2794)
           ' Mr' -> ' Corbyn', 'unal', ' Modi', ' Putin', ' Trump'   (rank of the token itself: 9452)
    ' according' -> ' to', ' specific', ' logger', ' diligence', ' respective'   (rank of the token itself: 620)
          ' Los' -> ' Angeles', 'artan', ' Santos', 'opian', 'erville'   (rank of the token itself: 15214)
```

<figure class="fig"><svg viewBox="0 0 760 352" role="img" aria-label="For each of seven tokens, the three next tokens the direct path of attn-only-1l scores highest: Barack to Obama, Hussein, -lung; United to States, Nations, Kingdom; New to Zealand, York, Yorker; Hong to Kong, -wei, -chen; Mr to Corbyn, -unal, Modi; according to to, specific, logger; Los to Angeles, -artan, Santos."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="24.0" y="24.0" text-anchor="start">attn-only-1l: the top 3 rows of W<tspan baseline-shift="sub" style="font-size:72%">U</tspan> W<tspan baseline-shift="sub" style="font-size:72%">E</tspan> for a few tokens</text><text class="t-tick" x="24.0" y="44.0" text-anchor="start">no context at all: just "this token, then which token?"  (a leading dash marks a piece glued to the word before)</text><rect class="box-1" x="24.0" y="66.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="85.5" text-anchor="middle">Barack</text><line class="edge" x1="150.0" y1="81.0" x2="196.0" y2="81.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="66.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="85.5" text-anchor="middle">Obama</text><rect class="box" x="334.0" y="66.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="85.5" text-anchor="middle">Hussein</text><rect class="box" x="464.0" y="66.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="85.5" text-anchor="middle">-lung</text><text class="t-muted" x="604.0" y="86.0" text-anchor="start">itself: rank 33</text><rect class="box-1" x="24.0" y="106.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="125.5" text-anchor="middle">United</text><line class="edge" x1="150.0" y1="121.0" x2="196.0" y2="121.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="106.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="125.5" text-anchor="middle">States</text><rect class="box" x="334.0" y="106.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="125.5" text-anchor="middle">Nations</text><rect class="box" x="464.0" y="106.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="125.5" text-anchor="middle">Kingdom</text><text class="t-muted" x="604.0" y="126.0" text-anchor="start">itself: rank 2,623</text><rect class="box-1" x="24.0" y="146.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="165.5" text-anchor="middle">New</text><line class="edge" x1="150.0" y1="161.0" x2="196.0" y2="161.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="146.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="165.5" text-anchor="middle">Zealand</text><rect class="box" x="334.0" y="146.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="165.5" text-anchor="middle">York</text><rect class="box" x="464.0" y="146.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="165.5" text-anchor="middle">Yorker</text><text class="t-muted" x="604.0" y="166.0" text-anchor="start">itself: rank 33,197</text><rect class="box-1" x="24.0" y="186.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="205.5" text-anchor="middle">Hong</text><line class="edge" x1="150.0" y1="201.0" x2="196.0" y2="201.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="186.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="205.5" text-anchor="middle">Kong</text><rect class="box" x="334.0" y="186.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="205.5" text-anchor="middle">-wei</text><rect class="box" x="464.0" y="186.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="205.5" text-anchor="middle">-chen</text><text class="t-muted" x="604.0" y="206.0" text-anchor="start">itself: rank 2,794</text><rect class="box-1" x="24.0" y="226.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="245.5" text-anchor="middle">Mr</text><line class="edge" x1="150.0" y1="241.0" x2="196.0" y2="241.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="226.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="245.5" text-anchor="middle">Corbyn</text><rect class="box" x="334.0" y="226.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="245.5" text-anchor="middle">-unal</text><rect class="box" x="464.0" y="226.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="245.5" text-anchor="middle">Modi</text><text class="t-muted" x="604.0" y="246.0" text-anchor="start">itself: rank 9,452</text><rect class="box-1" x="24.0" y="266.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="285.5" text-anchor="middle">according</text><line class="edge" x1="150.0" y1="281.0" x2="196.0" y2="281.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="266.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="285.5" text-anchor="middle">to</text><rect class="box" x="334.0" y="266.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="285.5" text-anchor="middle">specific</text><rect class="box" x="464.0" y="266.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="285.5" text-anchor="middle">logger</text><text class="t-muted" x="604.0" y="286.0" text-anchor="start">itself: rank 620</text><rect class="box-1" x="24.0" y="306.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="84.0" y="325.5" text-anchor="middle">Los</text><line class="edge" x1="150.0" y1="321.0" x2="196.0" y2="321.0" marker-end="url(#ah)"/><rect class="box-3" x="204.0" y="306.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="264.0" y="325.5" text-anchor="middle">Angeles</text><rect class="box" x="334.0" y="306.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="394.0" y="325.5" text-anchor="middle">-artan</text><rect class="box" x="464.0" y="306.0" width="120.0" height="30.0" rx="6"/><text class="t-tick" x="524.0" y="325.5" text-anchor="middle">Santos</text><text class="t-muted" x="604.0" y="326.0" text-anchor="start">itself: rank 15,214</text></svg><figcaption>The direct path of attn-only-1l, read from two weight matrices. After "Barack" it predicts "Obama", after "United" it predicts "States", after "Hong" "Kong", after "according" "to", after "Los" "Angeles". These are bigrams. The token itself is never the top choice.</figcaption></figure>

This is the paper's own example, found in a different model trained by different people: "Barack" → "Obama". The other rows are bigrams too: "United States", "New Zealand", "Hong Kong", "according to", "Los Angeles". After "Mr" come surnames from the news.

### The unembedding is not the inverse of the embedding

A footnote adds: although $$W_{\mathrm{U}}$$ is called the "un-embedding", it should not be the **inverse** of $$W_{\mathrm{E}}$$. If it were, $$W_{\mathrm{U}} W_{\mathrm{E}}$$ would be the identity matrix, and the direct path would predict "the next token repeats this one", which is rarely true of text. The output above agrees: "Barack" ranks only 33rd after itself, "New" 33,197th. Over 2,000 ordinary tokens, we compared GPT-2, which uses the **same** matrix to embed and unembed ("tied" weights):

```text
  attn-only-1l: tokens 1000..2999, share whose top direct-path prediction is the token itself: 0.000
  gpt2: tokens 1000..2999, share whose top direct-path prediction is the token itself: 0.988
```

The untied toy never picks the token itself; tied GPT-2 does so 98.8% of the time, and must use its layers to undo it. The direct path is a clean bigram table only when $$W_{\mathrm{U}}$$ is free to differ from $$W_{\mathrm{E}}$$, as in the paper's models.

**What it is useful for.** This is the paper's method in miniature: a behaviour ("after Barack, say Obama") found by multiplying two weight matrices, with no input text, and read by a person. Part 3 shows every model in the paper splits into terms like this; Parts 4 to 6 read the others.

> [!TAKEAWAYS] Key takeaways
> - **Mechanistic interpretability** reverse-engineers the exact computation inside a network, like turning a binary back into source code. The paper starts with attention-only transformers of zero, one and two layers.
> - **Three simplifications:** the paper drops MLPs, which is a real loss. It drops biases, which is free because a bias is an always-1 input. It drops layer norm, which is almost free: folding it changed no log-probability by more than 0.00008.
> - **Four equations:** the model is $$x_0=W_{\mathrm{E}} t$$ (a column lookup), $$x_{i+1}=x_i+\sum_h h(x_i)$$ (every head adds), the same for MLPs, and $$T(t)=W_{\mathrm{U}} x_{-1}$$. In `attn-only-2l`, $$W_{\mathrm{E}}$$ (512 × 48,262) and $$W_{\mathrm{U}}$$ hold 95% of the 52.1 million weights.
> - **The residual stream is a sum.** Our 20 pieces added up to the real vector (error 0.0000024), and the logit of "ley" (20.645) split into the same pieces; head 1.6 gave 9.63 of it by looking back at the earlier "ley".
> - **No privileged basis:** rotating the whole stream of `attn-only-1l` moved no logit by more than 0.0001. Study the matrices, not the dimensions.
> - **Virtual weights** $$W_{\mathrm{I}}^2W_{\mathrm{O}}^1$$ connect any two layers directly (64 × 64 between two heads); different subspaces mean a virtual weight of 0.
> - **Bandwidth is scarce:** GPT-2 small computes 60 times more numbers than its stream has dimensions.
> - **A zero-layer model is one table**, $$W_{\mathrm{U}} W_{\mathrm{E}}$$, of rank at most 512; its best values are log bigram probabilities. Inside every model it is the **direct path**: Barack → Obama, Hong → Kong, Los → Angeles.

**Next, in Part 2:** we open up one attention head. Part 2 shows why heads are independent and simply add, how a head moves information between tokens, and how its four matrices collapse into two: the **QK circuit** $${W_{\mathrm{Q}}}^{\top} W_{\mathrm{K}}$$ (where to look) and the **OV circuit** $$W_{\mathrm{O}} W_{\mathrm{V}}$$ (what to copy).

<details>
<summary>Run it yourself</summary>

The two scripts behind this part are [`circuits_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/circuits/circuits_part1.py) (every real-model number: shapes, layer-norm folding, the residual stream split into parts, the direct path, virtual weights, bandwidth and the rotation test) and [`circuits_part1_toy.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/circuits/circuits_part1_toy.py) (the hand-sized examples). The first one downloads `attn-only-1l`, `attn-only-2l` and `gpt2` from Hugging Face the first time, then runs on the CPU in about 15 seconds.

TransformerLens 4 replaced the `HookedTransformer` class used here, so install a 3.x version:

```bash
pip install torch "transformer_lens<4"
python circuits_part1.py        # writes results/part1.json and results/part1_stdout.txt
python circuits_part1_toy.py    # writes results/part1_toy.json
```

<figure><img src="/img/papers/circuits/part1-run.png" alt="Terminal output of circuits_part1.py: the model shapes, the layer-norm folding check, the residual stream split into 20 parts with the share of each in the logit of ley, the direct-path top five tokens for three models, the virtual weight shapes, the bandwidth and PCA numbers, and the rotation test" loading="lazy" /><figcaption>The real output of circuits_part1.py.</figcaption></figure>

<figure><img src="/img/papers/circuits/part1_toy-run.png" alt="Terminal output of circuits_part1_toy.py: the bigram counts of a nine-word corpus, the toy zero-layer model with its 4 by 4 table, the residual stream as a sum, the virtual weights 2 and 0, the bias folding and the rotation" loading="lazy" /><figcaption>The real output of circuits_part1_toy.py.</figcaption></figure>

</details>

## References

**The paper**

1. N. Elhage, N. Nanda, C. Olsson, T. Henighan, N. Joseph, B. Mann, A. Askell, Y. Bai, A. Chen, T. Conerly, N. DasSarma, D. Drain, D. Ganguli, Z. Hatfield-Dodds, D. Hernandez, A. Jones, J. Kernion, L. Lovitt, K. Ndousse, D. Amodei, T. Brown, J. Clark, J. Kaplan, S. McCandlish, C. Olah. [*A Mathematical Framework for Transformer Circuits*](https://transformer-circuits.pub/2021/framework/index.html). Transformer Circuits Thread, Anthropic, 22 December 2021. The screenshots in this part come from this page.
2. [Transformer Circuits Thread](https://transformer-circuits.pub/), the series of articles this paper opened.

**Related papers**

3. C. Olah, N. Cammarata, L. Schubert, G. Goh, M. Petrov, S. Carter. [*Zoom In: An Introduction to Circuits*](https://distill.pub/2020/circuits/zoom-in/). Distill, 2020. Part of the [Distill Circuits thread](https://distill.pub/2020/circuits/) the paper builds on.
4. A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, I. Polosukhin. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
5. K. He, X. Zhang, S. Ren, J. Sun. [*Deep Residual Learning for Image Recognition*](https://arxiv.org/abs/1512.03385) (ResNet). CVPR 2016.
6. R. K. Srivastava, K. Greff, J. Schmidhuber. [*Highway Networks*](https://arxiv.org/abs/1505.00387). 2015. The early residual-style network the paper mentions.
7. J. L. Ba, J. R. Kiros, G. E. Hinton. [*Layer Normalization*](https://arxiv.org/abs/1607.06450). 2016.
8. T. B. Brown et al. [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165) (GPT-3). NeurIPS 2020.
9. A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever. [*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (GPT-2). OpenAI, 2019.
10. O. Levy, Y. Goldberg. [*Neural Word Embedding as Implicit Matrix Factorization*](https://proceedings.neurips.cc/paper/2014/hash/b78666971ceae55a8e87efb7cbfd9ad4-Abstract.html). NeurIPS 2014. The paper's footnote on embeddings as factorised log-likelihood tables.
11. C. Olsson, N. Elhage, N. Nanda, et al. [*In-context Learning and Induction Heads*](https://arxiv.org/abs/2209.11895). 2022 ([web version](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html)). The follow-up on large models.
12. N. Elhage, T. Hume, C. Olsson, et al. [*Toy Models of Superposition*](https://arxiv.org/abs/2209.10652). 2022 ([web version](https://transformer-circuits.pub/2022/toy_model/index.html)).

**Tools and models**

13. N. Nanda, J. Bloom, and contributors. [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens), the library used for every experiment; version [3.9.0](https://pypi.org/project/transformer-lens/3.9.0/).
14. Model cards: [`NeelNanda/Attn_Only_1L512W_C4_Code`](https://huggingface.co/NeelNanda/Attn_Only_1L512W_C4_Code) (`attn-only-1l`), [`NeelNanda/Attn_Only_2L512W_C4_Code`](https://huggingface.co/NeelNanda/Attn_Only_2L512W_C4_Code) (`attn-only-2l`) and [`openai-community/gpt2`](https://huggingface.co/openai-community/gpt2).
15. Code for this part: [`circuits_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/circuits/circuits_part1.py) and [`circuits_part1_toy.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/circuits/circuits_part1_toy.py).
