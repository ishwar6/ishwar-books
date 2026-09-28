---
title: "Speculative Decoding: Several Tokens per Step, Not One Word Changed"
description: "How a small model can guess ahead and a big model can check all the guesses at once, why the output stays exactly the same, and what it really buys on real hardware: speculative decoding built from scratch and measured, from 1.16x on prose to 1.89x on code and 3.49x when the answer copies the prompt."
date: 2026-09-28
tags: [inference, speculative-decoding, llm]
series: "LLM Inference from the Ground Up"
series_part: 4
motif: waves
accent: "#f5b942"
---

Part 1 left us with a strange fact. On our test machine, pushing **16 tokens** through a model in one pass took **10.8 ms**. Pushing **one** token through took **10.0 ms**. Sixteen times the work, for almost the same time.

That is decode's big inefficiency in one line. Every step, the GPU hauls all of the model's weights out of memory (the expensive part) to do a tiny amount of math with them (the cheap part). The compute sits mostly idle.

This part is about a clever way to put that idle compute to work. It lets a model produce **several tokens per step instead of one**, and it does it without changing a single word of the output. It is called **speculative decoding**, and by the end you will have built it from scratch and seen exactly when it pays off and when it does not.

> [!TIP] The idea in one minute
> A **small, fast model** guesses the next few tokens. The **big model** then checks all of those guesses **in one pass**, which costs about the same as producing one token. It keeps every guess up to the first one it disagrees with, and replaces that one with its own choice. Good guesses mean several tokens per step; bad guesses cost a little time, never correctness.

## First: checking is cheap, writing is not

Why would checking guesses be cheaper than writing? Go back to the France example from Part 1. Writing "The capital is Paris." took four decode steps, one token each, because each word depended on the one before.

But suppose someone **handed** you the finished sentence and asked, "is this what you would have written?" Now every word is known. The model can read the whole sentence at once, exactly like a prefill, and at every position ask: "given everything before this word, is this the word I would have picked?" One pass answers the question for all of them.

That asymmetry is the whole trick. **Producing** tokens is sequential. **Checking** tokens is parallel.

Here it is measured on the model we will use as our "big" model in this part, Qwen2.5-3B-Instruct:

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Time for one pass of the 3B target model as it checks more tokens at once."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="axis" x1="64" y1="244" x2="740" y2="244"/><g class="mark"><title>1: 30.9 ms</title><rect class="hit-area" x="64.0" y="30" width="112.66666666666667" height="214"/><path class="s1 bar-mark" d="M108.33333333333334,244.0V114.7945471949339Q108.33333333333334,110.7945471949339 112.33333333333334,110.7945471949339H128.33333333333334Q132.33333333333334,110.7945471949339 132.33333333333334,114.7945471949339V244.0Z"/></g><text class="t-val" x="120.33333333333334" y="102.7945471949339" text-anchor="middle">30.9 ms</text><text class="t-tick" x="120.33333333333334" y="264" text-anchor="middle">1</text><g class="mark"><title>2: 32.7 ms</title><rect class="hit-area" x="176.66666666666669" y="30" width="112.66666666666667" height="214"/><path class="s1 bar-mark" d="M221.00000000000003,244.0V107.24768839112355Q221.00000000000003,103.24768839112355 225.00000000000003,103.24768839112355H241.00000000000003Q245.00000000000003,103.24768839112355 245.00000000000003,107.24768839112355V244.0Z"/></g><text class="t-val" x="233.00000000000003" y="95.24768839112355" text-anchor="middle">32.7 ms</text><text class="t-tick" x="233.00000000000003" y="264" text-anchor="middle">2</text><g class="mark"><title>3: 33.5 ms</title><rect class="hit-area" x="289.33333333333337" y="30" width="112.66666666666667" height="214"/><path class="s1 bar-mark" d="M333.6666666666667,244.0V103.42347591949485Q333.6666666666667,99.42347591949485 337.6666666666667,99.42347591949485H353.6666666666667Q357.6666666666667,99.42347591949485 357.6666666666667,103.42347591949485V244.0Z"/></g><text class="t-val" x="345.6666666666667" y="91.42347591949485" text-anchor="middle">33.5 ms</text><text class="t-tick" x="345.6666666666667" y="264" text-anchor="middle">3</text><g class="mark"><title>5: 33.7 ms</title><rect class="hit-area" x="402.0" y="30" width="112.66666666666667" height="214"/><path class="s1 bar-mark" d="M446.3333333333333,244.0V102.80050927230101Q446.3333333333333,98.80050927230101 450.3333333333333,98.80050927230101H466.3333333333333Q470.3333333333333,98.80050927230101 470.3333333333333,102.80050927230101V244.0Z"/></g><text class="t-val" x="458.3333333333333" y="90.80050927230101" text-anchor="middle">33.7 ms</text><text class="t-tick" x="458.3333333333333" y="264" text-anchor="middle">5</text><g class="mark"><title>9: 34.4 ms</title><rect class="hit-area" x="514.6666666666667" y="30" width="112.66666666666667" height="214"/><path class="s1 bar-mark" d="M559.0000000000001,244.0V99.69274729200134Q559.0000000000001,95.69274729200134 563.0000000000001,95.69274729200134H579.0000000000001Q583.0000000000001,95.69274729200134 583.0000000000001,99.69274729200134V244.0Z"/></g><text class="t-val" x="571.0000000000001" y="87.69274729200134" text-anchor="middle">34.4 ms</text><text class="t-tick" x="571.0000000000001" y="264" text-anchor="middle">9</text><g class="mark"><title>17: 38.2 ms</title><rect class="hit-area" x="627.3333333333334" y="30" width="112.66666666666667" height="214"/><path class="s1 bar-mark" d="M671.6666666666667,244.0V83.38461538461539Q671.6666666666667,79.38461538461539 675.6666666666667,79.38461538461539H691.6666666666667Q695.6666666666667,79.38461538461539 695.6666666666667,83.38461538461539V244.0Z"/></g><text class="t-val" x="683.6666666666667" y="71.38461538461539" text-anchor="middle">38.2 ms</text><text class="t-tick" x="683.6666666666667" y="264" text-anchor="middle">17</text><text class="t-tick" x="402.0" y="290" text-anchor="middle">tokens checked in one target pass</text><text class="t-tick" x="14" y="18" text-anchor="start">milliseconds</text></svg><figcaption>Time for one pass of the 3B model as it checks more tokens at once. Checking 17 tokens costs barely more than producing one.</figcaption></figure>

| Tokens checked in one pass | Time |
|---|---|
| 1 | 30.9 ms |
| 5 | 33.7 ms |
| 9 | 34.4 ms |
| 17 | 38.2 ms |

Checking 17 tokens takes only about 1.2 times as long as producing one. The weights get read once either way.

## Draft, then verify

So we need someone to make guesses. The classic choice is a **draft model**: a much smaller model from the same family that uses exactly the same vocabulary of tokens (it has to: the two models' tokens are compared one by one). Here that is Qwen2.5-0.5B-Instruct, six times smaller than the 3B target. It is not as smart, but on easy stretches of text it usually guesses what the big model would say.

Each round of speculative decoding has three steps:

<figure class="fig"><svg viewBox="0 0 760 296" role="img" aria-label="Speculative decoding: a small model drafts several tokens, the big model checks them all in one pass, and keeps the longest correct prefix plus its own next token."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="20" y="24" text-anchor="start">the text so far</text><rect class="box" x="20" y="34" width="76" height="30" rx="6"/><text class="t-note" x="58" y="54" text-anchor="middle">A</text><rect class="box" x="104" y="34" width="76" height="30" rx="6"/><text class="t-note" x="142" y="54" text-anchor="middle">fridge</text><rect class="box" x="188" y="34" width="76" height="30" rx="6"/><text class="t-note" x="226" y="54" text-anchor="middle">moves</text><text class="t-tick" x="300" y="24" text-anchor="start">1. the small draft model guesses 4 tokens, one at a time (cheap)</text><rect class="box-2" x="300" y="34" width="76" height="30" rx="6"/><text class="t-note" x="338" y="54" text-anchor="middle">heat</text><rect class="box-2" x="384" y="34" width="76" height="30" rx="6"/><text class="t-note" x="422" y="54" text-anchor="middle">out</text><rect class="box-2" x="468" y="34" width="76" height="30" rx="6"/><text class="t-note" x="506" y="54" text-anchor="middle">of</text><rect class="box-2" x="552" y="34" width="76" height="30" rx="6"/><text class="t-note" x="590" y="54" text-anchor="middle">the</text><text class="t-tick" x="20" y="108" text-anchor="start">2. the big target model checks all 4 guesses in ONE pass</text><rect class="box-1" x="300" y="118" width="330" height="40" rx="10"/><text class="t-strong" x="465" y="143" text-anchor="middle">target model: one forward pass</text><line class="edge" x1="338" y1="66" x2="338" y2="114" marker-end="url(#ah)"/><line class="edge" x1="422" y1="66" x2="422" y2="114" marker-end="url(#ah)"/><line class="edge" x1="506" y1="66" x2="506" y2="114" marker-end="url(#ah)"/><line class="edge" x1="590" y1="66" x2="590" y2="114" marker-end="url(#ah)"/><text class="t-tick" x="20" y="196" text-anchor="start">3. keep guesses up to the first disagreement, then take the target's own token</text><rect class="box-3" x="300" y="206" width="76" height="30" rx="6"/><text class="t-note" x="338" y="226" text-anchor="middle">heat</text><text class="t-tick" x="338" y="254" text-anchor="middle">kept</text><rect class="box-3" x="384" y="206" width="76" height="30" rx="6"/><text class="t-note" x="422" y="226" text-anchor="middle">out</text><text class="t-tick" x="422" y="254" text-anchor="middle">kept</text><rect class="box-1" x="468" y="206" width="76" height="30" rx="6"/><text class="t-note" x="506" y="226" text-anchor="middle">from</text><text class="t-tick" x="506" y="254" text-anchor="middle">target's fix</text><rect class="waste" x="552" y="206" width="76" height="30" rx="6"/><text class="t-note" x="590" y="226" text-anchor="middle">the</text><text class="t-tick" x="590" y="254" text-anchor="middle">thrown away</text><text class="t-note" x="300" y="280" text-anchor="start">result: 3 new tokens for the price of one target pass (plus 4 cheap draft steps)</text></svg><figcaption>One round of speculative decoding: the draft model guesses four tokens, the target model checks them in one pass, and the round keeps the correct prefix plus the target's own next token.</figcaption></figure>

1. **Draft.** The small model writes the next K tokens the normal way, one at a time. It is small, so each of those steps is quick.
2. **Verify.** The big model takes all K guesses and runs **one** pass over them, computing at every position which token it would have chosen.
3. **Accept.** Walk along the guesses. Keep each one that matches what the big model would have picked. At the first mismatch, stop, throw away that guess and everything after it, and use the big model's choice instead.

Notice step 3 always produces at least one token. Even if the very first guess is wrong, the big model has already computed what the right token is, so the round still moves forward by one. **Speculative decoding can never produce fewer tokens per big-model pass than plain decoding.** It can only waste the draft model's time.

### A real round, from our run

Here are the first few rounds of an actual run, with K = 4 guesses per round, on the prompt "Explain how a refrigerator keeps food cold":

| Round | Draft guessed | Kept | Big model adds | Tokens this round |
|---|---|---|---|---|
| 1 | `refrigerator` `works` `by` `utilizing` | 1 of 4 | `keeps` | 2 |
| 2 | `food` `cold` `by` `utilizing` | 2 of 4 | `through` | 3 |
| 3 | `a` `process` `called` `refriger` | 4 of 4 | `ation` | 5 |
| 4 | `.` `Here` `'s` `a` | 1 of 4 | `It` | 2 |
| 5 | `works` `by` `using` `a` | 4 of 4 | `cycle` | 5 |
| 6 | `of` `heat` `and` `cold` | 1 of 4 | `heating` | 2 |
| 7 | `and` `cooling` `to` `transfer` | 2 of 4 | `.` | 3 |
| 8 | `When` `food` `is` `placed` | 0 of 4 | `The` | 1 |

Round 3 is the dream case: the draft guessed `a process called refriger`, the big model agreed with all four, and added `ation` itself. Five tokens for one pass. Round 8 is the worst case: the first guess was wrong, so the round produced just the big model's own token, exactly what plain decoding would have produced, and only the draft's time was lost.

Every round that keeps even one guess is a round where the big model did more than one token's worth of progress for the price of one pass.

## Why the output does not change at all

This is the part that makes speculative decoding special. It is not an approximation. Done right, the text that comes out is **exactly** what the big model would have produced on its own.

**With greedy decoding** (always pick the most likely token) it is easy to see why. We only keep a guess if it is *the same token* the big model would have picked at that position. And when a guess is wrong, we take the big model's own pick. Every token that reaches the output is a token the big model chose. The draft model only decides how many of them we get per pass.

**With sampling** (picking randomly according to the model's probabilities, which is how most chatbots run) it needs one more idea, from the two papers that introduced the method, Leviathan et al. and Chen et al. (both 2023). Let $$p(x)$$ be the big model's probability for token $$x$$, and $$q(x)$$ the draft model's. The draft proposes $$x$$. Then:

- keep it with probability $$\min\left(1, \frac{p(x)}{q(x)}\right)$$;
- if it is rejected, draw a replacement from what the draft **under-weighted**: the distribution proportional to $$\max(0,\ p(x) - q(x))$$.

If the big model likes a token at least as much as the draft did, it is always kept. If the draft was over-eager about a token, it is kept only some of the time, and the replacement step adds back exactly the probability the draft missed. The two steps together rebuild $$p$$ exactly.

You do not have to take that on faith. Here is a simulation of one million tokens with a draft that is badly miscalibrated: it thinks "a" is the most likely word, when the big model prefers "the".

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="The draft model guesses a very different distribution, but after the accept-or-resample rule the output matches the big model exactly."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="56" y1="250.0" x2="740" y2="250.0"/><text class="t-tick" x="48" y="254.0" text-anchor="end">0.0</text><line class="grid" x1="56" y1="208.8" x2="740" y2="208.8"/><text class="t-tick" x="48" y="212.8" text-anchor="end">0.1</text><line class="grid" x1="56" y1="167.6" x2="740" y2="167.6"/><text class="t-tick" x="48" y="171.6" text-anchor="end">0.2</text><line class="grid" x1="56" y1="126.4" x2="740" y2="126.4"/><text class="t-tick" x="48" y="130.4" text-anchor="end">0.3</text><line class="grid" x1="56" y1="85.2" x2="740" y2="85.2"/><text class="t-tick" x="48" y="89.19999999999999" text-anchor="end">0.4</text><line class="grid" x1="56" y1="44.0" x2="740" y2="44.0"/><text class="t-tick" x="48" y="48.0" text-anchor="end">0.5</text><g class="mark"><title>"the": big model wants (p) 0.450</title><path class="s1 bar-mark" d="M75.0,250.0V68.6Q75.0,64.6 79.0,64.6H93.0Q97.0,64.6 97.0,68.6V250.0Z"/></g><g class="mark"><title>"the": draft guesses (q) 0.200</title><path class="s2 bar-mark" d="M101.0,250.0V171.6Q101.0,167.6 105.0,167.6H119.0Q123.0,167.6 123.0,171.6V250.0Z"/></g><g class="mark"><title>"the": speculative output 0.450</title><path class="s3 bar-mark" d="M127.0,250.0V68.50400400000001Q127.0,64.50400400000001 131.0,64.50400400000001H145.0Q149.0,64.50400400000001 149.0,68.50400400000001V250.0Z"/></g><text class="t-tick" x="113.0" y="270" text-anchor="middle">"the"</text><g class="mark"><title>"a": big model wants (p) 0.250</title><path class="s1 bar-mark" d="M189.0,250.0V151.0Q189.0,147.0 193.0,147.0H207.0Q211.0,147.0 211.0,151.0V250.0Z"/></g><g class="mark"><title>"a": draft guesses (q) 0.400</title><path class="s2 bar-mark" d="M215.0,250.0V89.19999999999999Q215.0,85.19999999999999 219.0,85.19999999999999H233.0Q237.0,85.19999999999999 237.0,89.19999999999999V250.0Z"/></g><g class="mark"><title>"a": speculative output 0.249</title><path class="s3 bar-mark" d="M241.0,250.0V151.23072000000002Q241.0,147.23072000000002 245.0,147.23072000000002H259.0Q263.0,147.23072000000002 263.0,151.23072000000002V250.0Z"/></g><text class="t-tick" x="227.0" y="270" text-anchor="middle">"a"</text><g class="mark"><title>"this": big model wants (p) 0.120</title><path class="s1 bar-mark" d="M303.0,250.0V204.56Q303.0,200.56 307.0,200.56H321.0Q325.0,200.56 325.0,204.56V250.0Z"/></g><g class="mark"><title>"this": draft guesses (q) 0.100</title><path class="s2 bar-mark" d="M329.0,250.0V212.8Q329.0,208.8 333.0,208.8H347.0Q351.0,208.8 351.0,212.8V250.0Z"/></g><g class="mark"><title>"this": speculative output 0.120</title><path class="s3 bar-mark" d="M355.0,250.0V204.71244000000002Q355.0,200.71244000000002 359.0,200.71244000000002H373.0Q377.0,200.71244000000002 377.0,204.71244000000002V250.0Z"/></g><text class="t-tick" x="341.0" y="270" text-anchor="middle">"this"</text><g class="mark"><title>"one": big model wants (p) 0.080</title><path class="s1 bar-mark" d="M417.0,250.0V221.04Q417.0,217.04 421.0,217.04H435.0Q439.0,217.04 439.0,221.04V250.0Z"/></g><g class="mark"><title>"one": draft guesses (q) 0.200</title><path class="s2 bar-mark" d="M443.0,250.0V171.6Q443.0,167.6 447.0,167.6H461.0Q465.0,167.6 465.0,171.6V250.0Z"/></g><g class="mark"><title>"one": speculative output 0.080</title><path class="s3 bar-mark" d="M469.0,250.0V220.864076Q469.0,216.864076 473.0,216.864076H487.0Q491.0,216.864076 491.0,220.864076V250.0Z"/></g><text class="t-tick" x="455.0" y="270" text-anchor="middle">"one"</text><g class="mark"><title>"my": big model wants (p) 0.060</title><path class="s1 bar-mark" d="M531.0,250.0V229.28Q531.0,225.28 535.0,225.28H549.0Q553.0,225.28 553.0,229.28V250.0Z"/></g><g class="mark"><title>"my": draft guesses (q) 0.050</title><path class="s2 bar-mark" d="M557.0,250.0V233.4Q557.0,229.4 561.0,229.4H575.0Q579.0,229.4 579.0,233.4V250.0Z"/></g><g class="mark"><title>"my": speculative output 0.060</title><path class="s3 bar-mark" d="M583.0,250.0V229.280412Q583.0,225.280412 587.0,225.280412H601.0Q605.0,225.280412 605.0,229.280412V250.0Z"/></g><text class="t-tick" x="569.0" y="270" text-anchor="middle">"my"</text><g class="mark"><title>"our": big model wants (p) 0.040</title><path class="s1 bar-mark" d="M645.0,250.0V237.52Q645.0,233.52 649.0,233.52H663.0Q667.0,233.52 667.0,237.52V250.0Z"/></g><g class="mark"><title>"our": draft guesses (q) 0.050</title><path class="s2 bar-mark" d="M671.0,250.0V233.4Q671.0,229.4 675.0,229.4H689.0Q693.0,229.4 693.0,233.4V250.0Z"/></g><g class="mark"><title>"our": speculative output 0.040</title><path class="s3 bar-mark" d="M697.0,250.0V237.408348Q697.0,233.408348 701.0,233.408348H715.0Q719.0,233.408348 719.0,237.408348V250.0Z"/></g><text class="t-tick" x="683.0" y="270" text-anchor="middle">"our"</text><line class="axis" x1="56" y1="250" x2="740" y2="250"/><rect class="s1" x="56" y="12" width="12" height="12" rx="3"/><text class="t-note" x="74" y="23" text-anchor="start">big model wants (p)</text><rect class="s2" x="252.8" y="12" width="12" height="12" rx="3"/><text class="t-note" x="270.8" y="23" text-anchor="start">draft guesses (q)</text><rect class="s3" x="435.20000000000005" y="12" width="12" height="12" rx="3"/><text class="t-note" x="453.20000000000005" y="23" text-anchor="start">speculative output</text><text class="t-tick" x="398.0" y="290" text-anchor="middle">probability of each next word (1,000,000 simulated tokens)</text></svg><figcaption>The draft's guesses (orange) look nothing like what the big model wants (blue). After the accept-or-resample rule, the output (green) matches the big model to within 0.0006 on every word.</figcaption></figure>

The output matches the big model's distribution to within **0.0006** on every word. And the share of guesses that were kept came out at **72.0%**, exactly the theory's prediction of $$\sum_x \min(p(x), q(x)) = 0.72$$. A worse draft does not make the output wrong. It only makes the process slower.

> [!NOTE] One honest footnote
> "Exactly the same" holds up to the limits of computer arithmetic. Checking five tokens in one pass uses slightly different floating-point operations than producing them one at a time, so in 16-bit precision two almost-tied words can occasionally swap. We saw exactly this. In 16-bit precision, a few of our speculative outputs drifted from plain decoding after more than a hundred tokens, each time at a near-tie (the first one at token 142, where the two top choices were separated by 0.125 in score, about one rounding step). Rerun in full 32-bit precision, all 8 speculative runs (four tasks, both drafters) were identical to plain decoding, token for token. vLLM's documentation describes the same thing: lossless "up to the precision limits of hardware numerics."

## The economics: how many tokens per pass?

How much faster this is depends on one number above all: the **acceptance rate** $$\alpha$$, the chance that a given guess is right. The trouble is that acceptance compounds. Guess 3 only counts if guesses 1 and 2 were right too.

If each guess is right with probability $$\alpha$$ and you make $$K$$ guesses, the expected number of tokens per big-model pass is

$$
\frac{1 - \alpha^{K+1}}{1 - \alpha}
$$

(Leviathan et al., 2023). Here is what that looks like:

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Expected tokens per target pass as the number of guesses grows, for three acceptance rates. Each extra guess helps less, because it only counts if every guess before it was right."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="56" y1="250.0" x2="610" y2="250.0"/><text class="t-tick" x="48" y="254.0" text-anchor="end">0</text><line class="grid" x1="56" y1="193.5" x2="610" y2="193.5"/><text class="t-tick" x="48" y="197.5" text-anchor="end">2</text><line class="grid" x1="56" y1="137.0" x2="610" y2="137.0"/><text class="t-tick" x="48" y="141.0" text-anchor="end">4</text><line class="grid" x1="56" y1="80.5" x2="610" y2="80.5"/><text class="t-tick" x="48" y="84.5" text-anchor="end">6</text><line class="grid" x1="56" y1="24.0" x2="610" y2="24.0"/><text class="t-tick" x="48" y="28.0" text-anchor="end">8</text><text class="t-tick" x="56.0" y="270" text-anchor="middle">0</text><text class="t-tick" x="194.5" y="270" text-anchor="middle">2</text><text class="t-tick" x="333.0" y="270" text-anchor="middle">4</text><text class="t-tick" x="471.5" y="270" text-anchor="middle">6</text><text class="t-tick" x="610.0" y="270" text-anchor="middle">8</text><line class="axis" x1="56" y1="250" x2="610" y2="250"/><polyline class="l2" points="56.0,221.8 125.2,207.6 194.5,200.6 263.8,197.0 333.0,195.3 402.2,194.4 471.5,193.9 540.8,193.7 610.0,193.6"/><g class="mark"><title>acceptance 50%, 0 guesses: 1.00 tokens per target pass</title><circle class="s2 ring" cx="56.0" cy="221.8" r="4.5"/></g><g class="mark"><title>acceptance 50%, 1 guesses: 1.50 tokens per target pass</title><circle class="s2 ring" cx="125.2" cy="207.6" r="4.5"/></g><g class="mark"><title>acceptance 50%, 2 guesses: 1.75 tokens per target pass</title><circle class="s2 ring" cx="194.5" cy="200.6" r="4.5"/></g><g class="mark"><title>acceptance 50%, 3 guesses: 1.88 tokens per target pass</title><circle class="s2 ring" cx="263.8" cy="197.0" r="4.5"/></g><g class="mark"><title>acceptance 50%, 4 guesses: 1.94 tokens per target pass</title><circle class="s2 ring" cx="333.0" cy="195.3" r="4.5"/></g><g class="mark"><title>acceptance 50%, 5 guesses: 1.97 tokens per target pass</title><circle class="s2 ring" cx="402.2" cy="194.4" r="4.5"/></g><g class="mark"><title>acceptance 50%, 6 guesses: 1.98 tokens per target pass</title><circle class="s2 ring" cx="471.5" cy="193.9" r="4.5"/></g><g class="mark"><title>acceptance 50%, 7 guesses: 1.99 tokens per target pass</title><circle class="s2 ring" cx="540.8" cy="193.7" r="4.5"/></g><g class="mark"><title>acceptance 50%, 8 guesses: 2.00 tokens per target pass</title><circle class="s2 ring" cx="610.0" cy="193.6" r="4.5"/></g><text class="t-note" x="622.0" y="197.6103515625" text-anchor="start">50% acceptance: 2.0</text><polyline class="l1" points="56.0,221.8 125.2,202.0 194.5,188.1 263.8,178.4 333.0,171.7 402.2,166.9 471.5,163.6 540.8,161.3 610.0,159.6"/><g class="mark"><title>acceptance 70%, 0 guesses: 1.00 tokens per target pass</title><circle class="s1 ring" cx="56.0" cy="221.8" r="4.5"/></g><g class="mark"><title>acceptance 70%, 1 guesses: 1.70 tokens per target pass</title><circle class="s1 ring" cx="125.2" cy="202.0" r="4.5"/></g><g class="mark"><title>acceptance 70%, 2 guesses: 2.19 tokens per target pass</title><circle class="s1 ring" cx="194.5" cy="188.1" r="4.5"/></g><g class="mark"><title>acceptance 70%, 3 guesses: 2.53 tokens per target pass</title><circle class="s1 ring" cx="263.8" cy="178.4" r="4.5"/></g><g class="mark"><title>acceptance 70%, 4 guesses: 2.77 tokens per target pass</title><circle class="s1 ring" cx="333.0" cy="171.7" r="4.5"/></g><g class="mark"><title>acceptance 70%, 5 guesses: 2.94 tokens per target pass</title><circle class="s1 ring" cx="402.2" cy="166.9" r="4.5"/></g><g class="mark"><title>acceptance 70%, 6 guesses: 3.06 tokens per target pass</title><circle class="s1 ring" cx="471.5" cy="163.6" r="4.5"/></g><g class="mark"><title>acceptance 70%, 7 guesses: 3.14 tokens per target pass</title><circle class="s1 ring" cx="540.8" cy="161.3" r="4.5"/></g><g class="mark"><title>acceptance 70%, 8 guesses: 3.20 tokens per target pass</title><circle class="s1 ring" cx="610.0" cy="159.6" r="4.5"/></g><text class="t-note" x="622.0" y="163.63329799250002" text-anchor="start">70% acceptance: 3.2</text><polyline class="l3" points="56.0,221.8 125.2,196.3 194.5,173.4 263.8,152.8 333.0,134.3 402.2,117.6 471.5,102.6 540.8,89.1 610.0,76.9"/><g class="mark"><title>acceptance 90%, 0 guesses: 1.00 tokens per target pass</title><circle class="s3 ring" cx="56.0" cy="221.8" r="4.5"/></g><g class="mark"><title>acceptance 90%, 1 guesses: 1.90 tokens per target pass</title><circle class="s3 ring" cx="125.2" cy="196.3" r="4.5"/></g><g class="mark"><title>acceptance 90%, 2 guesses: 2.71 tokens per target pass</title><circle class="s3 ring" cx="194.5" cy="173.4" r="4.5"/></g><g class="mark"><title>acceptance 90%, 3 guesses: 3.44 tokens per target pass</title><circle class="s3 ring" cx="263.8" cy="152.8" r="4.5"/></g><g class="mark"><title>acceptance 90%, 4 guesses: 4.10 tokens per target pass</title><circle class="s3 ring" cx="333.0" cy="134.3" r="4.5"/></g><g class="mark"><title>acceptance 90%, 5 guesses: 4.69 tokens per target pass</title><circle class="s3 ring" cx="402.2" cy="117.6" r="4.5"/></g><g class="mark"><title>acceptance 90%, 6 guesses: 5.22 tokens per target pass</title><circle class="s3 ring" cx="471.5" cy="102.6" r="4.5"/></g><g class="mark"><title>acceptance 90%, 7 guesses: 5.70 tokens per target pass</title><circle class="s3 ring" cx="540.8" cy="89.1" r="4.5"/></g><g class="mark"><title>acceptance 90%, 8 guesses: 6.13 tokens per target pass</title><circle class="s3 ring" cx="610.0" cy="76.9" r="4.5"/></g><text class="t-note" x="622.0" y="80.94628814249998" text-anchor="start">90% acceptance: 6.1</text><text class="t-tick" x="333.0" y="290" text-anchor="middle">guesses per step (K)</text><text class="t-tick" x="12" y="16" text-anchor="start">tokens per target pass</text></svg><figcaption>Expected tokens per big-model pass as the number of guesses grows. With 70% acceptance, going past 4 or 5 guesses adds almost nothing.</figcaption></figure>

Two lessons fall out of this curve:

- **Acceptance matters more than anything.** At 90%, eight guesses give about 6 tokens per pass. At 50%, you never get past 2, no matter how many guesses you make.
- **Guessing further has diminishing returns**, and every extra guess still costs a draft step. So there is a best K, and it depends on how predictable the text is.

The full cost of a round is K draft steps plus one verify pass. Speculative decoding wins only if the tokens you gain outweigh the draft steps you spend.

## Measured: what it really buys

I implemented speculative decoding from scratch (the code is at the end), with Qwen2.5-3B-Instruct as the big model and Qwen2.5-0.5B-Instruct as the draft, greedy decoding, 200 tokens per answer, on an Apple M5 Pro GPU. Plain decoding with the 3B model ran at about 31 tokens per second. One draft step cost 9.9 ms and one target pass 30.9 ms.

<figure class="fig"><svg viewBox="0 0 760 320" role="img" aria-label="Speedup against the number of guesses per round, for four kinds of prompt. Predictable text keeps improving; open-ended text gets slower past one or two guesses."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="56" y1="270.0" x2="570" y2="270.0"/><text class="t-tick" x="48" y="274.0" text-anchor="end">0.6x</text><line class="grid" x1="56" y1="201.4" x2="570" y2="201.4"/><text class="t-tick" x="48" y="205.42857142857142" text-anchor="end">1.0x</text><line class="grid" x1="56" y1="132.9" x2="570" y2="132.9"/><text class="t-tick" x="48" y="136.85714285714286" text-anchor="end">1.4x</text><line class="grid" x1="56" y1="64.3" x2="570" y2="64.3"/><text class="t-tick" x="48" y="68.28571428571425" text-anchor="end">1.8x</text><line class="base-line" x1="56" y1="201.4" x2="570" y2="201.4"/><text class="t-tick" x="566.0" y="194.42857142857142" text-anchor="end">plain decoding = 1.0x</text><text class="t-tick" x="56.0" y="290" text-anchor="middle">1</text><text class="t-tick" x="129.42857142857144" y="290" text-anchor="middle">2</text><text class="t-tick" x="202.85714285714286" y="290" text-anchor="middle">3</text><text class="t-tick" x="276.2857142857143" y="290" text-anchor="middle">4</text><text class="t-tick" x="423.14285714285717" y="290" text-anchor="middle">6</text><text class="t-tick" x="570.0" y="290" text-anchor="middle">8</text><polyline class="l3" points="56.0,155.3 129.4,111.2 202.9,82.4 276.3,57.3 423.1,66.3 570.0,54.2"/><g class="mark"><title>rewrite a paragraph, K=1: 1.27x, 98% of guesses kept</title><circle class="s3 ring" cx="56.0" cy="155.3" r="4.5"/></g><g class="mark"><title>rewrite a paragraph, K=2: 1.53x, 94% of guesses kept</title><circle class="s3 ring" cx="129.4" cy="111.2" r="4.5"/></g><g class="mark"><title>rewrite a paragraph, K=3: 1.69x, 91% of guesses kept</title><circle class="s3 ring" cx="202.9" cy="82.4" r="4.5"/></g><g class="mark"><title>rewrite a paragraph, K=4: 1.84x, 95% of guesses kept</title><circle class="s3 ring" cx="276.3" cy="57.3" r="4.5"/></g><g class="mark"><title>rewrite a paragraph, K=6: 1.79x, 83% of guesses kept</title><circle class="s3 ring" cx="423.1" cy="66.3" r="4.5"/></g><g class="mark"><title>rewrite a paragraph, K=8: 1.86x, 79% of guesses kept</title><circle class="s3 ring" cx="570.0" cy="54.2" r="4.5"/></g><polyline class="l1" points="56.0,150.1 129.4,106.3 202.9,85.7 276.3,68.7 423.1,62.2 570.0,48.4"/><g class="mark"><title>write code, K=1: 1.30x, 94% of guesses kept</title><circle class="s1 ring" cx="56.0" cy="150.1" r="4.5"/></g><g class="mark"><title>write code, K=2: 1.56x, 96% of guesses kept</title><circle class="s1 ring" cx="129.4" cy="106.3" r="4.5"/></g><g class="mark"><title>write code, K=3: 1.67x, 90% of guesses kept</title><circle class="s1 ring" cx="202.9" cy="85.7" r="4.5"/></g><g class="mark"><title>write code, K=4: 1.77x, 89% of guesses kept</title><circle class="s1 ring" cx="276.3" cy="68.7" r="4.5"/></g><g class="mark"><title>write code, K=6: 1.81x, 83% of guesses kept</title><circle class="s1 ring" cx="423.1" cy="62.2" r="4.5"/></g><g class="mark"><title>write code, K=8: 1.89x, 81% of guesses kept</title><circle class="s1 ring" cx="570.0" cy="48.4" r="4.5"/></g><polyline class="l2" points="56.0,193.3 129.4,174.2 202.9,185.6 276.3,188.0 423.1,211.7 570.0,243.1"/><g class="mark"><title>explain a concept, K=1: 1.05x, 63% of guesses kept</title><circle class="s2 ring" cx="56.0" cy="193.3" r="4.5"/></g><g class="mark"><title>explain a concept, K=2: 1.16x, 58% of guesses kept</title><circle class="s2 ring" cx="129.4" cy="174.2" r="4.5"/></g><g class="mark"><title>explain a concept, K=3: 1.09x, 49% of guesses kept</title><circle class="s2 ring" cx="202.9" cy="185.6" r="4.5"/></g><g class="mark"><title>explain a concept, K=4: 1.08x, 43% of guesses kept</title><circle class="s2 ring" cx="276.3" cy="188.0" r="4.5"/></g><g class="mark"><title>explain a concept, K=6: 0.94x, 34% of guesses kept</title><circle class="s2 ring" cx="423.1" cy="211.7" r="4.5"/></g><g class="mark"><title>explain a concept, K=8: 0.76x, 26% of guesses kept</title><circle class="s2 ring" cx="570.0" cy="243.1" r="4.5"/></g><polyline class="l4" points="56.0,177.7 129.4,187.3 202.9,198.2 276.3,220.6 423.1,244.8 570.0,249.0"/><g class="mark"><title>write a story, K=1: 1.14x, 67% of guesses kept</title><circle class="s4 ring" cx="56.0" cy="177.7" r="4.5"/></g><g class="mark"><title>write a story, K=2: 1.08x, 48% of guesses kept</title><circle class="s4 ring" cx="129.4" cy="187.3" r="4.5"/></g><g class="mark"><title>write a story, K=3: 1.02x, 41% of guesses kept</title><circle class="s4 ring" cx="202.9" cy="198.2" r="4.5"/></g><g class="mark"><title>write a story, K=4: 0.89x, 33% of guesses kept</title><circle class="s4 ring" cx="276.3" cy="220.6" r="4.5"/></g><g class="mark"><title>write a story, K=6: 0.75x, 24% of guesses kept</title><circle class="s4 ring" cx="423.1" cy="244.8" r="4.5"/></g><g class="mark"><title>write a story, K=8: 0.72x, 23% of guesses kept</title><circle class="s4 ring" cx="570.0" cy="249.0" r="4.5"/></g><line class="grid" x1="575.0" y1="48.4" x2="586.0" y2="48.4"/><rect class="s1" x="590.0" y="43.4" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="52.44431431890706" text-anchor="start">write code: 1.89x</text><line class="grid" x1="575.0" y1="54.2" x2="586.0" y2="65.4"/><rect class="s3" x="590.0" y="60.4" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="69.44431431890706" text-anchor="start">rewrite a paragraph: 1.86x</text><line class="grid" x1="575.0" y1="243.1" x2="586.0" y2="243.1"/><rect class="s2" x="590.0" y="238.1" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="247.07917530769748" text-anchor="start">explain a concept: 0.76x</text><line class="grid" x1="575.0" y1="249.0" x2="586.0" y2="260.1"/><rect class="s4" x="590.0" y="255.1" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="264.0791753076975" text-anchor="start">write a story: 0.72x</text><text class="t-tick" x="313.0" y="310" text-anchor="middle">guesses per round (K), draft model</text><text class="t-tick" x="12" y="18" text-anchor="start">speedup vs plain decoding</text></svg><figcaption>Speedup against plain decoding as the number of guesses per round grows, for four kinds of task. Hover a point for the share of guesses kept.</figcaption></figure>

| Task | Best K | Speedup | Guesses kept | Tokens per big-model pass |
|---|---|---|---|---|
| rewrite a paragraph (copy with small edits) | 8 | **1.86x** | 79% | 6.7 |
| write a Python function | 8 | **1.89x** | 81% | 7.1 |
| explain how a fridge works | 2 | **1.16x** | 58% | 2.1 |
| write an original story | 1 | **1.14x** | 67% | 1.7 |

The pattern is exactly what the economics predicted:

- **Predictable text wins big.** Code and the rewrite task kept **80 to 98%** of the draft's guesses. Every extra guess kept paying off, all the way to K = 8, where the big model produced **7.1 tokens per pass** on the code task, for a **1.89x** speedup.
- **Open-ended text barely wins.** The explanation and the story kept only 25 to 65% of guesses. The best result was one or two guesses per round, for about **1.15x**. With more guesses it got **slower than plain decoding**: at K = 8 the story ran at **0.72x**, because most guesses were thrown away and each one still cost a draft step.
- **The drafter is expensive here.** One step of the 0.5B draft took 9.9 ms, about a third of a 3B pass (30.9 ms), because on this machine a small model's step is dominated by fixed overhead rather than by its size. A drafter that cheap relative to the target is the main reason the ceiling is under 2x. On a data-centre GPU serving a 70B model, where the big model's step is far more expensive than the drafter's, the same machinery has much more room: that is where the published 2x to 3x results come from.

Every run was also checked against plain decoding, token by token. More on that in a moment.

## A drafter that costs nothing: prompt lookup

A draft model is not the only way to guess. In a lot of real work the answer **copies from the prompt**: summarising a document, answering questions about it, editing code, rewriting text. There, the best guess for "what comes next" is often "whatever came next the last time this phrase appeared".

**Prompt lookup decoding** (Saxena, 2023) does exactly that. Take the last few tokens written, search the prompt and the answer so far for the same sequence, and propose whatever followed it. No second model, no extra memory, and the search takes microseconds.

On the rewrite task, where the answer is the prompt's paragraph with a few words changed, prompt lookup was the **fastest thing in this whole article**: **3.49x** with 8 guesses per round, and 2.70x with 4. It beat the draft model (1.86x) because its guesses cost nothing.

On everything else it did little or nothing: 1.23x on code, and slightly **slower** than plain decoding on the explanation and the story (0.93x and 0.97x), where there is almost nothing to copy and each failed lookup still means an extra check. It is the right tool when the output copies the input, and the wrong one otherwise.

## Better drafters

Most of the research since 2023 has been about getting better guesses more cheaply:

- **Medusa** (Cai et al., 2024) skips the separate draft model. It adds a few extra prediction "heads" to the big model itself, each guessing a different position ahead, and checks several candidate continuations at once with a tree-shaped attention pattern. The paper reports over 2.2x speedup with the original model frozen.
- **EAGLE** (Li et al., 2024) trains a very small network that works on the big model's own internal features rather than on raw tokens, which makes its guesses far more accurate. The paper reports 2.7x to 3.5x lower latency on LLaMA2-Chat 70B. **EAGLE-3** (2025) reports up to 6.5x.
- **Multi-token prediction** trains the model to predict several future tokens from the start, which gives it a built-in drafter.

vLLM supports all of these. A draft model looks like this:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --speculative-config '{"method": "draft_model", "model": "Qwen/Qwen2.5-0.5B-Instruct", "num_speculative_tokens": 4}'
```

and prompt lookup like this:

```bash
vllm serve Qwen/Qwen2.5-3B-Instruct \
  --speculative-config '{"method": "ngram", "num_speculative_tokens": 4, "prompt_lookup_min": 2, "prompt_lookup_max": 3}'
```

## When it helps, and when it hurts

Everything above comes back to Part 1. Speculative decoding spends **idle compute** to save **sequential steps**. So it helps exactly when there is idle compute, and stops helping when there is not.

| It helps when | It hurts or does nothing when |
|---|---|
| the text is predictable (code, structured output, copying from the input) | the text is open-ended and creative, so acceptance is low |
| the GPU is lightly loaded, so each step is memory-bound with compute to spare | the GPU is already busy with a large batch (Part 1): the "spare" compute is no longer spare |
| the big model is slow relative to the drafter | the drafter is expensive relative to the big model |
| you care about latency for one user | you care only about total throughput at high load |

That is why vLLM's documentation lists the biggest gains at **low load** ("latency focused"), and why serving systems adjust speculation, or switch it off, as load grows (vLLM's documentation lists a dynamic speculative decoding option for exactly this). It is a tool for making one user's answer arrive faster, not a free throughput boost.

## Summary

- Decode leaves the GPU's compute mostly idle. **Checking** tokens is parallel and costs about the same as producing one: 38.2 ms to check 17 tokens against 30.9 ms for one, on our 3B model.
- **Speculative decoding**: a cheap drafter guesses K tokens, the big model checks them in one pass, keeps the correct prefix and adds its own next token. Never fewer than one token per pass.
- **It is exact.** With greedy decoding, every kept token is the big model's own choice. With sampling, the accept-with-probability $$\min(1, p/q)$$ rule plus resampling from $$\max(0, p - q)$$ reproduces the big model's distribution, confirmed here to within 0.0006.
- The payoff depends on the **acceptance rate**: expected tokens per pass $$= (1 - \alpha^{K+1}) / (1 - \alpha)$$.
- Measured: a from-scratch implementation reached **1.89x** on code and up to **3.49x** with prompt lookup on a rewrite task, but only about **1.15x** on open-ended prose, and it got slower than plain decoding with too many guesses.
- It is a **latency** tool: best at low load, on predictable text, with a cheap drafter.

<details>
<summary>Speculative decoding from scratch (greedy)</summary>

```python
"""Greedy speculative decoding with a draft model, from scratch.
target: the big model, draft: a small model with the same tokenizer (e.g. Qwen2.5-3B and 0.5B)."""
import torch
from transformers import DynamicCache

@torch.inference_mode()
def run(model, ids, cache):
    return model(input_ids=torch.tensor([ids], device=model.device), past_key_values=cache).logits[0]

@torch.inference_mode()
def speculative(target, draft, prompt, k=4, max_new=200, eos=None):
    tcache, dcache = DynamicCache(), DynamicCache()
    out = [int(run(target, prompt, tcache)[-1].argmax())]    # prefill: the first token is the target's
    run(draft, prompt, dcache)
    while len(out) < max_new and out[-1] != eos:
        committed = prompt + out                              # out[-1] is not in the target's cache yet
        # 1. the draft guesses k tokens, one at a time
        guesses, x = [], committed[dcache.get_seq_length():]
        for _ in range(k):
            g = int(run(draft, x, dcache)[-1].argmax())
            guesses.append(g)
            x = [g]
        # 2. the target checks every guess in ONE pass: preds[i] is its choice after position i
        preds = run(target, [out[-1]] + guesses, tcache).argmax(-1).tolist()
        # 3. keep guesses while they match, then take the target's own token
        n = 0
        while n < k and guesses[n] == preds[n]:
            n += 1
        keep = len(prompt) + len(out) + n                     # roll both caches back past rejected guesses
        tcache.crop(keep)
        if dcache.get_seq_length() > keep:
            dcache.crop(keep)
        out += guesses[:n] + [preds[n]]
    return out[:max_new]
```

Plain greedy decoding with the target alone produces exactly the same tokens (up to floating-point ties), one per pass. The only difference is how many target passes it takes.

</details>

## References

1. Y. Leviathan, M. Kalman, Y. Matias. [*Fast Inference from Transformers via Speculative Decoding*](https://arxiv.org/abs/2211.17192). ICML 2023. (2x to 3x on T5-XXL "with identical outputs.")
2. C. Chen et al. [*Accelerating Large Language Model Decoding with Speculative Sampling*](https://arxiv.org/abs/2302.01318). 2023. (2x to 2.5x on Chinchilla 70B, preserving the target distribution "within hardware numerics.")
3. T. Cai et al. [*Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads*](https://arxiv.org/abs/2401.10774). 2024.
4. Y. Li et al. [*EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty*](https://arxiv.org/abs/2401.15077). 2024.
5. Y. Li et al. [*EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test*](https://arxiv.org/abs/2503.01840). 2025.
6. A. Saxena. [*Prompt Lookup Decoding*](https://github.com/apoorvumang/prompt-lookup-decoding). 2023.
7. vLLM documentation: [Speculative decoding](https://docs.vllm.ai/en/latest/features/speculative_decoding/).
