---
title: "The KV Cache: An LLM's Short-Term Memory, and Its Bill"
description: "Why an LLM keeps notes on every token, how big they get (16 GiB for one 128K-token Llama 3.1 8B chat), and why the old way of storing them wasted most of the memory."
date: 2026-09-28
tags: [inference, kv-cache, llm]
series: "LLM Inference from the Ground Up"
series_part: 2
motif: kv
accent: "#5fd4b0"
---

[Part 1](llm-inference-1-prefill-and-decode.md) ended with a rule. A decode step reads all the model's weights from memory, whether it writes one token or a hundred. So a GPU is only productive when it decodes many conversations at once. The catch was in the last line: every conversation in that batch needs its own memory for its notes. Those notes are called the **KV cache**.

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="One decode step reads the weights once for four conversations; each conversation keeps its own notes in GPU memory."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="26" text-anchor="start">one decode step, four conversations in the batch</text><rect class="box-2" x="250" y="44" width="260" height="46" rx="10"/><text class="t-title" x="380" y="66" text-anchor="middle">model weights</text><text class="t-tick" x="380" y="82" text-anchor="middle">read from memory once per step</text><line class="edge" x1="380" y1="92" x2="110" y2="138" marker-end="url(#ah)"/><rect class="box-1" x="40" y="140" width="140" height="36" rx="8"/><text class="t-title" x="110" y="163" text-anchor="middle">conversation A</text><rect class="box-3" x="50" y="186" width="120" height="26" rx="6"/><text class="t-tick" x="110" y="204" text-anchor="middle">notes on A's past</text><line class="edge" x1="380" y1="92" x2="290" y2="138" marker-end="url(#ah)"/><rect class="box-1" x="220" y="140" width="140" height="36" rx="8"/><text class="t-title" x="290" y="163" text-anchor="middle">conversation B</text><rect class="box-3" x="230" y="186" width="120" height="26" rx="6"/><text class="t-tick" x="290" y="204" text-anchor="middle">notes on B's past</text><line class="edge" x1="380" y1="92" x2="470" y2="138" marker-end="url(#ah)"/><rect class="box-1" x="400" y="140" width="140" height="36" rx="8"/><text class="t-title" x="470" y="163" text-anchor="middle">conversation C</text><rect class="box-3" x="410" y="186" width="120" height="26" rx="6"/><text class="t-tick" x="470" y="204" text-anchor="middle">notes on C's past</text><line class="edge" x1="380" y1="92" x2="650" y2="138" marker-end="url(#ah)"/><rect class="box-1" x="580" y="140" width="140" height="36" rx="8"/><text class="t-title" x="650" y="163" text-anchor="middle">conversation D</text><rect class="box-3" x="590" y="186" width="120" height="26" rx="6"/><text class="t-tick" x="650" y="204" text-anchor="middle">notes on D's past</text><text class="t-note" x="20" y="244" text-anchor="start">GPU memory</text><rect class="box-2" x="120" y="228" width="250" height="26" rx="4"/><text class="t-tick" x="245.0" y="246" text-anchor="middle">weights</text><rect class="box-3" x="373" y="228" width="70" height="26" rx="4"/><text class="t-tick" x="408.0" y="246" text-anchor="middle">A</text><rect class="box-3" x="446" y="228" width="110" height="26" rx="4"/><text class="t-tick" x="501.0" y="246" text-anchor="middle">B</text><rect class="box-3" x="559" y="228" width="50" height="26" rx="4"/><text class="t-tick" x="584.0" y="246" text-anchor="middle">C</text><rect class="box-3" x="612" y="228" width="90" height="26" rx="4"/><text class="t-tick" x="657.0" y="246" text-anchor="middle">D</text><rect class="box-ghost" x="705" y="228" width="60" height="26" rx="4"/><text class="t-tick" x="735.0" y="246" text-anchor="middle">free</text><text class="t-tick" x="20" y="284" text-anchor="start">Every conversation in the batch keeps its own notes in GPU memory, next to the weights.</text></svg><figcaption>One decode step reads the weights once and serves every conversation in the batch. Each conversation also keeps its own notes in GPU memory. This part is about those notes.</figcaption></figure>

This part answers four questions about the notes. What are they? Why does the model keep them? How big do they get? And why did the first serving systems waste most of the memory they set aside for them? The last answer is the problem that [Part 3](llm-inference-3-vllm.md) solves.

## What the model needs from the past

Start with the loop from Part 1. To write the next token, the model takes everything so far, runs it through its stack of layers and gets one new token. Then it appends that token and runs again.

Here is the question that Part 1 skipped. When the model writes token 500, how does it use tokens 1 to 499? A word like "it" means nothing on its own. To write the next word well, the model must find out what "it" refers to, and that answer sits somewhere earlier in the text.

The part of each layer that looks back at earlier tokens is called **attention**. To understand the KV cache, you need to understand attention first. Attention is a lookup, and a small example shows all of it.

> [!DEFINITION] Vector
> A list of numbers, such as [1, 0] or [0.81, -1.2, 0.05]. Inside a model, every token is a vector at every layer: a few hundred or a few thousand numbers that describe it.

## Attention: a lookup with a question, labels and contents

Think of a library with a card for every book. Each card has a **label** on the front (what the book is about) and the **contents** inside. You walk in with a **question**. You compare your question with every label, pick the cards that match best, and read mostly their contents.

Attention does the same thing with vectors. At each layer, every token makes three vectors from its own vector:

- a **query**: the question this token asks ("which earlier word do I refer to?");
- a **key**: the label it shows to later tokens ("I am a noun, an animal");
- a **value**: the content it hands over when a later token picks it.

Each of the three comes from multiplying the token's vector by a fixed matrix of weights. The model learned these three matrices during training, and they stay the same at inference.

<figure class="fig"><svg viewBox="0 0 760 286" role="img" aria-label="Each token vector is multiplied by three weight matrices to give its query, key and value. The query is used once; the key and value are kept."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">one token at one layer</text><rect class="box" x="30" y="110" width="110" height="40" rx="8"/><text class="t-title" x="85" y="135" text-anchor="middle">vector of "cat"</text><line class="edge" x1="142" y1="130" x2="218" y2="68" marker-end="url(#ah)"/><rect class="box-ghost" x="220" y="50" width="90" height="36" rx="8"/><text class="t-mono" x="265" y="73" text-anchor="middle">× W_q</text><line class="edge" x1="312" y1="68" x2="368" y2="68" marker-end="url(#ah)"/><rect class="box-2" x="370" y="50" width="110" height="36" rx="8"/><text class="t-title" x="425" y="73" text-anchor="middle">query q</text><text class="t-tick" x="496" y="73" text-anchor="start">used now, then thrown away</text><line class="edge" x1="142" y1="130" x2="218" y2="138" marker-end="url(#ah)"/><rect class="box-ghost" x="220" y="120" width="90" height="36" rx="8"/><text class="t-mono" x="265" y="143" text-anchor="middle">× W_k</text><line class="edge" x1="312" y1="138" x2="368" y2="138" marker-end="url(#ah)"/><rect class="box-1" x="370" y="120" width="110" height="36" rx="8"/><text class="t-title" x="425" y="143" text-anchor="middle">key k</text><text class="t-tick" x="496" y="143" text-anchor="start">kept: later tokens compare with it</text><line class="edge" x1="142" y1="130" x2="218" y2="208" marker-end="url(#ah)"/><rect class="box-ghost" x="220" y="190" width="90" height="36" rx="8"/><text class="t-mono" x="265" y="213" text-anchor="middle">× W_v</text><line class="edge" x1="312" y1="208" x2="368" y2="208" marker-end="url(#ah)"/><rect class="box-3" x="370" y="190" width="110" height="36" rx="8"/><text class="t-title" x="425" y="213" text-anchor="middle">value v</text><text class="t-tick" x="496" y="213" text-anchor="start">kept: later tokens mix it in</text><text class="t-tick" x="20" y="270" text-anchor="start">Three fixed weight matrices turn the same token vector into three different vectors.</text></svg><figcaption>One token at one layer. The same vector, multiplied by three different weight matrices, gives a query, a key and a value. Only the key and the value are needed again later.</figcaption></figure>

> [!DEFINITION] Query, key and value
> Three vectors that each token makes at each attention layer. The **query** asks what the token looks for. The **key** says what the token offers. The **value** is the information it passes on. Keys and queries are compared; values are mixed.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §2.1 · page 3
> [![The paragraph on self-attention with Equation 2, with self-attention layers and query, key, and value vectors highlighted](/img/papers/vllm/p2-qkv.png)](/img/papers/vllm/p2-qkv.png)
>
> **What it says:** the most important part of a transformer language model is its "self-attention layers". For each position $$i$$, a layer applies linear transformations to the hidden state $$x_i$$ to get the "query, key, and value vectors": $$q_i = W_q x_i$$, $$k_i = W_k x_i$$ and $$v_i = W_v x_i$$.
>
> **What it means:**
>
> This is the picture above, written as maths. The hidden state $$x_i$$ is the vector of token $$i$$ at this layer. $$W_q$$, $$W_k$$ and $$W_v$$ are the three weight matrices. A "linear transformation" is a matrix multiplication, nothing more.
>
> Note one detail. Each key and value depends only on its own token's vector at that layer. Once a token's vector is known, its key and value are fixed. That fact makes the whole KV cache possible, as the next sections show.

### A lookup with tiny numbers

Take the text "The cat sat" and a new token, "it". To keep the arithmetic small, every key, value and query here has only 2 numbers. Real models use 64 or 128 numbers per head, but the steps are the same.

The three earlier tokens have these keys and values:

| Token | Key | Value |
|---|---|---|
| The | [0, 1] | [1, 0] |
| cat | [3, 0] | [0, 4] |
| sat | [1, 1] | [2, 2] |

The new token "it" asks with the query [1, 0]. Attention now takes three steps.

**Step 1: compare the query with every key.** The comparison is a **dot product**: multiply the numbers in matching positions and add the results. For "cat", that is 1 × 3 + 0 × 0 = 3. For "The" it is 0, and for "sat" it is 1. A higher score means a better match.

**Step 2: turn the scores into weights.** First divide each score by the square root of the vector length, here √2. This keeps scores from growing too large in long vectors. Then apply **softmax**: raise e to the power of each score and divide by the total. The weights come out as 0.088, 0.734 and 0.178, and they add up to 1.

**Step 3: mix the values.** Multiply each value by its weight and add them up. The answer is [0.44, 3.29]. It is mostly the value of "cat", because the key of "cat" matched the query best.

<figure class="fig"><svg viewBox="0 0 760 340" role="img" aria-label="A tiny attention lookup: the query is compared with three keys, the scores become weights 0.088, 0.734 and 0.178, and the answer is the weighted mix of the values."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">the new token "it" asks with its query q = [1, 0]</text><text class="t-note" x="20" y="54" text-anchor="start">step 1: compare q with every key</text><text class="t-note" x="270" y="54" text-anchor="start">step 2: turn scores into weights</text><text class="t-note" x="520" y="54" text-anchor="start">step 3: mix the values</text><rect class="box" x="20" y="78" width="52" height="30" rx="6"/><text class="t-title" x="46" y="98" text-anchor="middle">The</text><rect class="box-1" x="80" y="78" width="76" height="30" rx="6"/><text class="t-tick" x="118" y="98" text-anchor="middle">k = [0, 1]</text><text class="t-val" x="166" y="98" text-anchor="start">q·k = 0</text><text class="t-tick" x="270" y="98" text-anchor="start">0 / √2 = 0.00</text><rect class="box-ghost" x="370" y="82" width="120" height="22" rx="4"/><rect class="box-on" x="370" y="82" width="10.56" height="22" rx="4"/><text class="t-val" x="386.56" y="98" text-anchor="start">0.088</text><rect class="box-3" x="520" y="78" width="76" height="30" rx="6"/><text class="t-tick" x="558" y="98" text-anchor="middle">v = [1, 0]</text><text class="t-tick" x="606" y="98" text-anchor="start">× 0.088</text><rect class="box" x="20" y="140" width="52" height="30" rx="6"/><text class="t-title" x="46" y="160" text-anchor="middle">cat</text><rect class="box-1" x="80" y="140" width="76" height="30" rx="6"/><text class="t-tick" x="118" y="160" text-anchor="middle">k = [3, 0]</text><text class="t-val" x="166" y="160" text-anchor="start">q·k = 3</text><text class="t-tick" x="270" y="160" text-anchor="start">3 / √2 = 2.12</text><rect class="box-ghost" x="370" y="144" width="120" height="22" rx="4"/><rect class="box-on" x="370" y="144" width="88.08" height="22" rx="4"/><text class="t-val" x="464.08" y="160" text-anchor="start">0.734</text><rect class="box-3" x="520" y="140" width="76" height="30" rx="6"/><text class="t-tick" x="558" y="160" text-anchor="middle">v = [0, 4]</text><text class="t-tick" x="606" y="160" text-anchor="start">× 0.734</text><rect class="box" x="20" y="202" width="52" height="30" rx="6"/><text class="t-title" x="46" y="222" text-anchor="middle">sat</text><rect class="box-1" x="80" y="202" width="76" height="30" rx="6"/><text class="t-tick" x="118" y="222" text-anchor="middle">k = [1, 1]</text><text class="t-val" x="166" y="222" text-anchor="start">q·k = 1</text><text class="t-tick" x="270" y="222" text-anchor="start">1 / √2 = 0.71</text><rect class="box-ghost" x="370" y="206" width="120" height="22" rx="4"/><rect class="box-on" x="370" y="206" width="21.36" height="22" rx="4"/><text class="t-val" x="397.36" y="222" text-anchor="start">0.178</text><rect class="box-3" x="520" y="202" width="76" height="30" rx="6"/><text class="t-tick" x="558" y="222" text-anchor="middle">v = [2, 2]</text><text class="t-tick" x="606" y="222" text-anchor="start">× 0.178</text><text class="t-tick" x="270" y="278" text-anchor="start">softmax: the weights sum to 1</text><line class="edge" x1="640" y1="270" x2="640" y2="262"/><rect class="box-2" x="520" y="268" width="220" height="36" rx="8"/><text class="t-title" x="630" y="291" text-anchor="middle">answer = [0.44, 3.29]</text><text class="t-tick" x="20" y="326" text-anchor="start">The answer is mostly the value of "cat", because its key matched the query best.</text></svg><figcaption>The whole lookup with tiny numbers. The query of "it" is compared with three keys, the scores become weights that add up to 1, and the answer is a weighted mix of the three values. A PyTorch attention function gives the same answer, [0.44, 3.29].</figcaption></figure>

> [!DEFINITION] Dot product and softmax
> The **dot product** of two vectors multiplies matching numbers and adds the results: [1, 0] · [3, 0] = 3. **Softmax** turns a list of scores into weights between 0 and 1 that add up to 1. A bigger score always gets a bigger weight.

The three steps fit in one line. Here $$Q$$ is the query, $$K$$ holds the keys of all earlier tokens, $$V$$ holds their values and $$d$$ is the length of one key:

$$
\text{attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Read it from the inside out. $$Q K^\top$$ is step 1, one dot product for every key. Dividing by $$\sqrt{d}$$ and the softmax are step 2. Multiplying by $$V$$ is step 3. Our script checked the hand result against PyTorch's own attention function, and both give [0.44, 3.29].

Now look at what the lookup needed. It needed one query, from the new token only. It needed the key and the value of **every earlier token**. The next section is about where those keys and values come from on each step.

## The waste hiding in the loop

A model like Llama writes from left to right. Each token may look only at the tokens before it, never at the tokens after it. So adding a new token at the end changes nothing about the tokens already there.

That has a strong consequence. The key and value of "cat" at layer 5 depend only on "The cat" and on the fixed weights. When "sat", "on" and "the" arrive later, the key of "cat" stays exactly the same. It was the same on step 3 and it will be the same on step 300.

Now picture a model without notes. On every step it would run the whole text through every layer again. It would compute every key and value again, only to get the same numbers as last time. It is like a writer who rereads the whole book from page one before each new sentence.

The fix is to keep notes. After a token's key and value are computed at a layer, store them. On the next step, compute the key and value of the new token only, add them to the store, and read the old ones back. That store is the **KV cache**.

<figure class="fig"><svg viewBox="0 0 760 270" role="img" aria-label="Without a KV cache each step recomputes keys and values for all tokens; with a cache each step computes only the new token and reads the rest."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="30" y="26" text-anchor="start">no cache: recompute K and V for all tokens</text><text class="t-note" x="420" y="26" text-anchor="start">KV cache: compute only the new token</text><rect class="box-2" x="30" y="44" width="38" height="24" rx="5"/><rect class="box-2" x="30" y="74" width="38" height="24" rx="5"/><rect class="box-2" x="74" y="74" width="38" height="24" rx="5"/><rect class="box-2" x="30" y="104" width="38" height="24" rx="5"/><rect class="box-2" x="74" y="104" width="38" height="24" rx="5"/><rect class="box-2" x="118" y="104" width="38" height="24" rx="5"/><rect class="box-2" x="30" y="134" width="38" height="24" rx="5"/><rect class="box-2" x="74" y="134" width="38" height="24" rx="5"/><rect class="box-2" x="118" y="134" width="38" height="24" rx="5"/><rect class="box-2" x="162" y="134" width="38" height="24" rx="5"/><rect class="box-2" x="30" y="164" width="38" height="24" rx="5"/><rect class="box-2" x="74" y="164" width="38" height="24" rx="5"/><rect class="box-2" x="118" y="164" width="38" height="24" rx="5"/><rect class="box-2" x="162" y="164" width="38" height="24" rx="5"/><rect class="box-2" x="206" y="164" width="38" height="24" rx="5"/><rect class="box-2" x="30" y="194" width="38" height="24" rx="5"/><rect class="box-2" x="74" y="194" width="38" height="24" rx="5"/><rect class="box-2" x="118" y="194" width="38" height="24" rx="5"/><rect class="box-2" x="162" y="194" width="38" height="24" rx="5"/><rect class="box-2" x="206" y="194" width="38" height="24" rx="5"/><rect class="box-2" x="250" y="194" width="38" height="24" rx="5"/><text class="t-tick" x="30" y="236" text-anchor="start">steps 1 to 6 (rows), tokens (columns)</text><rect class="box-2" x="420" y="44" width="38" height="24" rx="5"/><rect class="box-3" x="420" y="74" width="38" height="24" rx="5"/><rect class="box-2" x="464" y="74" width="38" height="24" rx="5"/><rect class="box-3" x="420" y="104" width="38" height="24" rx="5"/><rect class="box-3" x="464" y="104" width="38" height="24" rx="5"/><rect class="box-2" x="508" y="104" width="38" height="24" rx="5"/><rect class="box-3" x="420" y="134" width="38" height="24" rx="5"/><rect class="box-3" x="464" y="134" width="38" height="24" rx="5"/><rect class="box-3" x="508" y="134" width="38" height="24" rx="5"/><rect class="box-2" x="552" y="134" width="38" height="24" rx="5"/><rect class="box-3" x="420" y="164" width="38" height="24" rx="5"/><rect class="box-3" x="464" y="164" width="38" height="24" rx="5"/><rect class="box-3" x="508" y="164" width="38" height="24" rx="5"/><rect class="box-3" x="552" y="164" width="38" height="24" rx="5"/><rect class="box-2" x="596" y="164" width="38" height="24" rx="5"/><rect class="box-3" x="420" y="194" width="38" height="24" rx="5"/><rect class="box-3" x="464" y="194" width="38" height="24" rx="5"/><rect class="box-3" x="508" y="194" width="38" height="24" rx="5"/><rect class="box-3" x="552" y="194" width="38" height="24" rx="5"/><rect class="box-3" x="596" y="194" width="38" height="24" rx="5"/><rect class="box-2" x="640" y="194" width="38" height="24" rx="5"/><text class="t-tick" x="420" y="236" text-anchor="start">steps 1 to 6 (rows), tokens (columns)</text><text class="t-tick" x="30" y="258" text-anchor="start">orange = computed this step</text><text class="t-tick" x="420" y="258" text-anchor="start">aqua = read from the cache</text></svg><figcaption>Without a cache, every step computes keys and values for every token so far (orange). With a cache, every step computes one new row and reads the rest from memory (green).</figcaption></figure>

> [!DEFINITION] KV cache
> The stored keys and values of every token a sequence has seen so far, for every layer. The model writes one new entry per token and reads all the old entries on every decode step. Each sequence has its own cache.

> [!PAPER] Kwon et al. (2023) · §2.2 · page 3
> [![The paragraph about the KV cache, with one by one, cached for generating future tokens, KV cache of one token depends on all its previous tokens, and will be different highlighted](/img/papers/vllm/p2-kvcache.png)](/img/papers/vllm/p2-kvcache.png)
>
> **What it says:** the model generates new tokens "one by one", and each new token depends on the key and value vectors of all previous tokens. So these vectors are "cached for generating future tokens", which is the KV cache. The paper adds that the "KV cache of one token depends on all its previous tokens", so the same word at two positions "will be different".
>
> **What it means:**
>
> The first half is the idea of this section: compute once, keep, reuse. The second half is a warning that matters later. At layer 1 the vector of "the" is about the word alone. At deeper layers, the vector of each token has already mixed in the tokens before it, through earlier attention steps.
>
> So the cache entry for "the" in "The cat" differs from the entry for "the" in "on the mat". You cannot keep one shared entry per word. Each sequence needs its own cache, built in its own order.

### What the cache saves, measured

How much time do the notes save? We measured the time to produce one more token on Qwen2.5-0.5B, the small model from Part 1, on the GPU of an Apple M5 Pro. Without a cache, the model runs the whole text again. With a cache, it runs only the new token.

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Time for the next token with and without a KV cache."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="244.0" x2="740" y2="244.0"/><text class="t-tick" x="56" y="248.0" text-anchor="end">0</text><line class="grid" x1="64" y1="203.2" x2="740" y2="203.2"/><text class="t-tick" x="56" y="207.2" text-anchor="end">100</text><line class="grid" x1="64" y1="162.4" x2="740" y2="162.4"/><text class="t-tick" x="56" y="166.39999999999998" text-anchor="end">200</text><line class="grid" x1="64" y1="121.6" x2="740" y2="121.6"/><text class="t-tick" x="56" y="125.60000000000001" text-anchor="end">300</text><line class="grid" x1="64" y1="80.8" x2="740" y2="80.8"/><text class="t-tick" x="56" y="84.79999999999998" text-anchor="end">400</text><line class="grid" x1="64" y1="40.0" x2="740" y2="40.0"/><text class="t-tick" x="56" y="44.0" text-anchor="end">500</text><g class="mark"><title>128 tokens, no cache: 13.6 ms</title><path class="s2 bar-mark" d="M105.6,244.0V242.4512Q105.6,238.4512 109.6,238.4512H125.6Q129.6,238.4512 129.6,242.4512V244.0Z"/></g><text class="t-val" x="117.6" y="230.4512" text-anchor="middle">13.6</text><g class="mark"><title>128 tokens, KV cache: 9.9 ms</title><path class="s3 bar-mark" d="M133.6,244.0V243.97304Q133.6,239.97304 137.6,239.97304H153.6Q157.6,239.97304 157.6,243.97304V244.0Z"/></g><text class="t-val" x="145.6" y="231.97304" text-anchor="middle">9.9</text><text class="t-tick" x="131.6" y="264" text-anchor="middle">128 tokens so far</text><g class="mark"><title>512 tokens, no cache: 27.3 ms</title><path class="s2 bar-mark" d="M240.79999999999995,244.0V236.8616Q240.79999999999995,232.8616 244.79999999999995,232.8616H260.79999999999995Q264.79999999999995,232.8616 264.79999999999995,236.8616V244.0Z"/></g><text class="t-val" x="252.79999999999995" y="224.8616" text-anchor="middle">27.3</text><g class="mark"><title>512 tokens, KV cache: 10.3 ms</title><path class="s3 bar-mark" d="M268.79999999999995,244.0V243.818Q268.79999999999995,239.818 272.79999999999995,239.818H288.79999999999995Q292.79999999999995,239.818 292.79999999999995,243.818V244.0Z"/></g><text class="t-val" x="280.79999999999995" y="231.818" text-anchor="middle">10.3</text><text class="t-tick" x="266.79999999999995" y="264" text-anchor="middle">512 tokens so far</text><g class="mark"><title>2048 tokens, no cache: 94.9 ms</title><path class="s2 bar-mark" d="M376.0,244.0V209.2808Q376.0,205.2808 380.0,205.2808H396.0Q400.0,205.2808 400.0,209.2808V244.0Z"/></g><text class="t-val" x="388.0" y="197.2808" text-anchor="middle">94.9</text><g class="mark"><title>2048 tokens, KV cache: 10.9 ms</title><path class="s3 bar-mark" d="M404.0,244.0V243.57319999999999Q404.0,239.57319999999999 408.0,239.57319999999999H424.0Q428.0,239.57319999999999 428.0,243.57319999999999V244.0Z"/></g><text class="t-val" x="416.0" y="231.57319999999999" text-anchor="middle">10.9</text><text class="t-tick" x="402.0" y="264" text-anchor="middle">2,048 tokens so far</text><g class="mark"><title>4096 tokens, no cache: 200.7 ms</title><path class="s2 bar-mark" d="M511.19999999999993,244.0V166.1144Q511.19999999999993,162.1144 515.1999999999999,162.1144H531.1999999999999Q535.1999999999999,162.1144 535.1999999999999,166.1144V244.0Z"/></g><text class="t-val" x="523.1999999999999" y="154.1144" text-anchor="middle">201</text><g class="mark"><title>4096 tokens, KV cache: 11.0 ms</title><path class="s3 bar-mark" d="M539.1999999999999,244.0V243.52016Q539.1999999999999,239.52016 543.1999999999999,239.52016H559.1999999999999Q563.1999999999999,239.52016 563.1999999999999,243.52016V244.0Z"/></g><text class="t-val" x="551.1999999999999" y="231.52016" text-anchor="middle">11.0</text><text class="t-tick" x="537.1999999999999" y="264" text-anchor="middle">4,096 tokens so far</text><g class="mark"><title>8192 tokens, no cache: 449.1 ms</title><path class="s2 bar-mark" d="M646.4,244.0V64.7672Q646.4,60.7672 650.4,60.7672H666.4Q670.4,60.7672 670.4,64.7672V244.0Z"/></g><text class="t-val" x="658.4" y="52.7672" text-anchor="middle">449</text><g class="mark"><title>8192 tokens, KV cache: 11.5 ms</title><path class="s3 bar-mark" d="M674.4,244.0V243.308Q674.4,239.308 678.4,239.308H694.4Q698.4,239.308 698.4,243.308V244.0Z"/></g><text class="t-val" x="686.4" y="231.308" text-anchor="middle">11.5</text><text class="t-tick" x="672.4" y="264" text-anchor="middle">8,192 tokens so far</text><line class="axis" x1="64" y1="244" x2="740" y2="244"/><rect class="s2" x="64" y="10" width="12" height="12" rx="3"/><text class="t-note" x="82" y="21" text-anchor="start">no cache</text><rect class="s3" x="174" y="10" width="12" height="12" rx="3"/><text class="t-note" x="192" y="21" text-anchor="start">KV cache</text><text class="t-tick" x="402.0" y="290" text-anchor="middle">time to produce the next token, milliseconds</text></svg><figcaption>Time to produce the next token on Qwen2.5-0.5B (Apple M5 Pro GPU). Without a cache the cost climbs with the length of the text; with a cache it stays almost flat.</figcaption></figure>

| Tokens so far | No cache | KV cache | Slower without |
|---|---|---|---|
| 128 | 13.6 ms | 9.9 ms | 1.4x |
| 512 | 27.3 ms | 10.3 ms | 2.7x |
| 2,048 | 94.9 ms | 10.9 ms | 8.7x |
| 4,096 | 200.7 ms | 11.0 ms | 18x |
| 8,192 | 449.1 ms | 11.5 ms | 39x |

Without a cache, every new token costs a full prefill of the whole text. Compare the no-cache column with Part 1's prefill table: the numbers match. So a long answer gets slower with every token it writes, and at 8,192 tokens each step is 39 times slower. With a cache, each step costs about the same, whatever the length. Every serious inference engine uses one.

The price is in the word "store". The cache swaps compute for **memory**, and the next sections show how fast that memory adds up.

## What exactly is stored

A real model does attention many times for each token, not once. It has a stack of **layers**, and each layer has its own attention. Inside each layer, attention is split into several **heads**: smaller lookups that run side by side, each with its own keys and values.

> [!DEFINITION] Layer and attention head
> A **layer** is one block of the model's stack; Llama 3.1 8B has 32. An **attention head** is one of several small attention lookups inside a layer. Each head has its own query, key and value vectors, so it can look for a different kind of match.

So for every token, the cache keeps one key and one value per KV head, in every layer. Each key and each value is a vector of `head_dim` numbers, and each number takes a fixed number of bytes.

<figure class="fig"><svg viewBox="0 0 760 278" role="img" aria-label="For one token, Llama 3.1 8B stores a key and a value for each of 8 heads in each of 32 layers, each 128 numbers of 2 bytes: 128 KiB."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">what the cache keeps for ONE token of Llama 3.1 8B</text><rect class="box" x="20" y="60" width="90" height="40" rx="8"/><text class="t-title" x="65" y="85" text-anchor="middle">"cat"</text><line class="edge" x1="112" y1="80" x2="146" y2="60"/><rect class="box-ghost" x="150" y="44" width="100" height="32" rx="6"/><text class="t-tick" x="200" y="65" text-anchor="middle">layer 1</text><line class="edge" x1="112" y1="80" x2="146" y2="100"/><rect class="box-ghost" x="150" y="84" width="100" height="32" rx="6"/><text class="t-tick" x="200" y="105" text-anchor="middle">layer 2</text><line class="edge" x1="112" y1="80" x2="146" y2="140"/><rect class="box-ghost" x="150" y="124" width="100" height="32" rx="6"/><text class="t-tick" x="200" y="145" text-anchor="middle">layer 3</text><text class="t-title" x="200" y="186" text-anchor="middle">⋮</text><line class="edge" x1="112" y1="80" x2="146" y2="220"/><rect class="box-ghost" x="150" y="204" width="100" height="32" rx="6"/><text class="t-tick" x="200" y="225" text-anchor="middle">layer 32</text><line class="edge" x1="252" y1="60" x2="300" y2="60" marker-end="url(#ah)"/><text class="t-note" x="310" y="50" text-anchor="start">inside one layer: 8 KV heads</text><rect class="box-1" x="310" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="322" y="79" text-anchor="middle">K</text><rect class="box-3" x="336" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="348" y="79" text-anchor="middle">V</text><rect class="box-1" x="364" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="376" y="79" text-anchor="middle">K</text><rect class="box-3" x="390" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="402" y="79" text-anchor="middle">V</text><rect class="box-1" x="418" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="430" y="79" text-anchor="middle">K</text><rect class="box-3" x="444" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="456" y="79" text-anchor="middle">V</text><rect class="box-1" x="472" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="484" y="79" text-anchor="middle">K</text><rect class="box-3" x="498" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="510" y="79" text-anchor="middle">V</text><rect class="box-1" x="526" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="538" y="79" text-anchor="middle">K</text><rect class="box-3" x="552" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="564" y="79" text-anchor="middle">V</text><rect class="box-1" x="580" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="592" y="79" text-anchor="middle">K</text><rect class="box-3" x="606" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="618" y="79" text-anchor="middle">V</text><rect class="box-1" x="634" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="646" y="79" text-anchor="middle">K</text><rect class="box-3" x="660" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="672" y="79" text-anchor="middle">V</text><rect class="box-1" x="688" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="700" y="79" text-anchor="middle">K</text><rect class="box-3" x="714" y="62" width="24" height="24" rx="4"/><text class="t-tick" x="726" y="79" text-anchor="middle">V</text><text class="t-tick" x="310" y="112" text-anchor="start">each K and each V is a list of 128 numbers (head_dim)</text><text class="t-tick" x="310" y="132" text-anchor="start">each number takes 2 bytes (bfloat16)</text><rect class="box-2" x="310" y="156" width="430" height="70" rx="10"/><text class="t-note" x="525" y="180" text-anchor="middle">2 (K and V) × 32 layers × 8 heads × 128 numbers × 2 bytes</text><text class="t-title" x="525" y="206" text-anchor="middle">= 131,072 bytes = 128 KiB per token</text><text class="t-tick" x="20" y="262" text-anchor="start">One token leaves 512 small vectors behind: a key and a value for every head of every layer.</text></svg><figcaption>What the cache keeps for one token of Llama 3.1 8B. In each of the 32 layers, each of the 8 KV heads stores a key and a value of 128 numbers, at 2 bytes per number.</figcaption></figure>

Is that the real layout? We can look inside a running model. We ran a 100-token prompt through Qwen2.5-0.5B and opened its cache. Layer 0 holds a key tensor of shape (1, 2, 100, 64): 1 sequence, 2 KV heads, 100 tokens and 64 numbers per key. There are 24 layers, each with keys and values.

```text
== 2. The KV cache of a real model: Qwen2.5-0.5B, bfloat16
  config: layers 24, query heads 14, KV heads 2, head_dim 64, device mps
  100 tokens: 24 layers, key tensor of layer 0 (1, 2, 100, 64) torch.bfloat16, cache 1,228,800 bytes
  200 tokens: 24 layers, key tensor of layer 0 (1, 2, 200, 64) torch.bfloat16, cache 2,457,600 bytes
  measured: 12,288 bytes per token; formula 2 x 24 x 2 x 64 x 2 = 12,288 bytes
```

The cache for 200 tokens is exactly twice the cache for 100 tokens. Every token adds the same 12,288 bytes. That gives a formula you can apply to any model.

### The size formula

Every number in the formula comes from a model's `config.json` file. Here is the file for Llama 3.1 8B, the main example of this part:

<figure class="fig"><svg viewBox="0 0 760 290" role="img" aria-label="The four numbers in a model config that decide its KV cache size: layers, KV heads, head dimension and data type."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><rect class="code-bg" x="20" y="20" width="360" height="250" rx="12"/><text class="t-tick" x="36" y="44" text-anchor="start">Llama-3.1-8B  config.json (excerpt)</text><text class="t-code" x="36" y="72" text-anchor="start">{</text><text x="36" y="94" xml:space="preserve"><tspan class="t-code-key">  "hidden_size":</tspan><tspan class="t-code"> 4096,</tspan></text><text x="36" y="116" xml:space="preserve"><tspan class="t-code-key">  "num_attention_heads":</tspan><tspan class="t-code"> 32,</tspan></text><rect x="30" y="123" width="340" height="21" rx="4" style="fill:color-mix(in srgb, var(--h3) 14%, transparent)"/><text x="36" y="138" xml:space="preserve"><tspan class="t-code-key">  "num_hidden_layers":</tspan><tspan class="t-code-num"> 32,</tspan></text><rect x="30" y="145" width="340" height="21" rx="4" style="fill:color-mix(in srgb, var(--h3) 14%, transparent)"/><text x="36" y="160" xml:space="preserve"><tspan class="t-code-key">  "num_key_value_heads":</tspan><tspan class="t-code-num"> 8,</tspan></text><rect x="30" y="167" width="340" height="21" rx="4" style="fill:color-mix(in srgb, var(--h3) 14%, transparent)"/><text x="36" y="182" xml:space="preserve"><tspan class="t-code-key">  "head_dim":</tspan><tspan class="t-code-num"> 128,</tspan></text><rect x="30" y="189" width="340" height="21" rx="4" style="fill:color-mix(in srgb, var(--h3) 14%, transparent)"/><text x="36" y="204" xml:space="preserve"><tspan class="t-code-key">  "torch_dtype":</tspan><tspan class="t-code-num"> "bfloat16",</tspan></text><text x="36" y="226" xml:space="preserve"><tspan class="t-code-key">  "max_position_embeddings":</tspan><tspan class="t-code"> 131072</tspan></text><text class="t-code" x="36" y="248" text-anchor="start">}</text><text class="t-val" x="470" y="88" text-anchor="start">2</text><text class="t-tick" x="566" y="88" text-anchor="start">K and V</text><text class="t-val" x="470" y="122" text-anchor="start">× 32</text><text class="t-tick" x="566" y="122" text-anchor="start">layers</text><line class="edge" x1="372" y1="133" x2="460" y2="117" marker-end="url(#ah)"/><text class="t-val" x="470" y="156" text-anchor="start">× 8</text><text class="t-tick" x="566" y="156" text-anchor="start">KV heads</text><line class="edge" x1="372" y1="155" x2="460" y2="151" marker-end="url(#ah)"/><text class="t-val" x="470" y="190" text-anchor="start">× 128</text><text class="t-tick" x="566" y="190" text-anchor="start">head_dim</text><line class="edge" x1="372" y1="177" x2="460" y2="185" marker-end="url(#ah)"/><text class="t-val" x="470" y="224" text-anchor="start">× 2 bytes</text><text class="t-tick" x="566" y="224" text-anchor="start">bfloat16</text><line class="edge" x1="372" y1="199" x2="460" y2="219" marker-end="url(#ah)"/><line class="axis" x1="470" y1="240" x2="720" y2="240"/><text class="t-val" x="470" y="264" text-anchor="start">= 131,072 bytes</text><text class="t-note" x="470" y="284" text-anchor="start">= 128 KiB for every token</text></svg><figcaption>The four fields in a model config that decide the size of its KV cache: the number of layers, the number of key/value heads, the head dimension and the data type.</figcaption></figure>

$$
\text{KV bytes per token} = 2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes per number}
$$

Read it from left to right. The 2 counts the key and the value. Layers and KV heads count how many keys and values each token leaves behind. Head dim is the length of each one. Bytes per number is 2 for the 16-bit formats that models usually run in.

Check it on Qwen2.5-0.5B first: $$2 \times 24 \times 2 \times 64 \times 2 = 12{,}288$$ bytes, which is exactly what we measured. Now Llama 3.1 8B: $$2 \times 32 \times 8 \times 128 \times 2 = 131{,}072$$ bytes. That is **128 KiB for every token** of every conversation the GPU serves.

> [!DEFINITION] KiB and GiB
> Units of memory size. 1 KiB is 1,024 bytes and 1 GiB is 1,024 × 1,024 × 1,024 bytes, about 1.07 billion bytes. A GB, as in "80 GB GPU", is exactly 1 billion bytes.

> [!DEFINITION] bfloat16
> A way to store a number in 16 bits, which is 2 bytes. Most models today run their weights and their KV cache in bfloat16 or a similar 16-bit format.

## How big it gets

The cache grows by one entry per token, so its size is the formula times the number of tokens. Llama 3.1 8B accepts up to 131,072 tokens (128K). One conversation at that length needs $$131{,}072 \times 128$$ KiB, which is **16 GiB**.

<figure class="fig"><svg viewBox="0 0 760 230" role="img" aria-label="KV cache per token for four models."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><g class="mark"><title>Qwen2.5-0.5B: 12.0 KiB</title><rect class="hit-area" x="0" y="16.0" width="760" height="49.5"/><path class="s1 bar-mark" d="M190,28.75H193.43534482758622Q197.43534482758622,28.75 197.43534482758622,32.75V48.75Q197.43534482758622,52.75 193.43534482758622,52.75H190Z"/></g><text class="t-note" x="178" y="45.75" text-anchor="end">Qwen2.5-0.5B</text><text class="t-val" x="205.43534482758622" y="45.75" text-anchor="start">12.0 KiB</text><text class="t-tick" x="205.43534482758622" y="62.75" text-anchor="start">1.5 GiB for a 128K-token context</text><g class="mark"><title>Llama-2-7B (MHA): 512 KiB</title><rect class="hit-area" x="0" y="65.5" width="760" height="49.5"/><path class="s1 bar-mark" d="M190,78.25H503.2413793103448Q507.2413793103448,78.25 507.2413793103448,82.25V98.25Q507.2413793103448,102.25 503.2413793103448,102.25H190Z"/></g><text class="t-note" x="178" y="95.25" text-anchor="end">Llama-2-7B (MHA)</text><text class="t-val" x="515.2413793103449" y="95.25" text-anchor="start">512 KiB</text><text class="t-tick" x="515.2413793103449" y="112.25" text-anchor="start">64.0 GiB for a 128K-token context</text><g class="mark"><title>Llama-3.1-8B (GQA): 128 KiB</title><rect class="hit-area" x="0" y="115.0" width="760" height="49.5"/><path class="s1 bar-mark" d="M190,127.75H265.3103448275862Q269.3103448275862,127.75 269.3103448275862,131.75V147.75Q269.3103448275862,151.75 265.3103448275862,151.75H190Z"/></g><text class="t-note" x="178" y="144.75" text-anchor="end">Llama-3.1-8B (GQA)</text><text class="t-val" x="277.3103448275862" y="144.75" text-anchor="start">128 KiB</text><text class="t-tick" x="277.3103448275862" y="161.75" text-anchor="start">16.0 GiB for a 128K-token context</text><g class="mark"><title>Llama-3.1-70B (GQA): 320 KiB</title><rect class="hit-area" x="0" y="164.5" width="760" height="49.5"/><path class="s1 bar-mark" d="M190,177.25H384.2758620689655Q388.2758620689655,177.25 388.2758620689655,181.25V197.25Q388.2758620689655,201.25 384.2758620689655,201.25H190Z"/></g><text class="t-note" x="178" y="194.25" text-anchor="end">Llama-3.1-70B (GQA)</text><text class="t-val" x="396.2758620689655" y="194.25" text-anchor="start">320 KiB</text><text class="t-tick" x="396.2758620689655" y="211.25" text-anchor="start">40.0 GiB for a 128K-token context</text><line class="axis" x1="190" y1="16" x2="190" y2="214"/></svg><figcaption>KV cache per token, and for one 128K-token conversation, for four models, from each model's config.</figcaption></figure>

| Model | Layers | KV heads | Head dim | Per token | One 128K-token conversation |
|---|---|---|---|---|---|
| Qwen2.5-0.5B | 24 | 2 | 64 | 12 KiB | 1.5 GiB |
| Llama 3.1 8B | 32 | 8 | 128 | 128 KiB | 16 GiB |
| Llama 3.1 70B | 80 | 8 | 128 | 320 KiB | 40 GiB |
| Llama 2 7B | 32 | 32 | 128 | 512 KiB | 64 GiB |

Look at the Llama 3.1 8B row again. The weights of this model take about 16 GB. One conversation at full length needs a cache of the same size. One long chat costs as much memory as the whole model.

Now put that on a real GPU. An NVIDIA H100 has 80 GB of memory. The weights of Llama 3.1 8B take 16.1 GB (8.03 billion numbers at 2 bytes each). One 128K-token cache is 17.2 GB. After the weights, the card has room for only **three** such conversations.

<figure class="fig"><svg viewBox="0 0 760 180" role="img" aria-label="An 80 GB H100 holds the 16.1 GB of Llama 3.1 8B weights and only three 128K-token conversations of 17.2 GB each."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">one 80 GB H100 serving Llama 3.1 8B, conversations at the full 128K tokens</text><rect class="box-2" x="40" y="50" width="134.85000000000002" height="44" rx="6"/><text class="t-tick" x="108.42500000000001" y="77" text-anchor="middle">weights 16.1 GB</text><rect class="box-3" x="176.85000000000002" y="50" width="144.2" height="44" rx="6"/><text class="t-tick" x="249.95000000000002" y="77" text-anchor="middle">conv. 1: 17.2 GB</text><rect class="box-3" x="323.05" y="50" width="144.2" height="44" rx="6"/><text class="t-tick" x="396.15" y="77" text-anchor="middle">conv. 2: 17.2 GB</text><rect class="box-3" x="469.25" y="50" width="144.2" height="44" rx="6"/><text class="t-tick" x="542.35" y="77" text-anchor="middle">conv. 3: 17.2 GB</text><rect class="box-ghost" x="615.45" y="50" width="102.55000000000004" height="44" rx="6"/><text class="t-tick" x="667.725" y="77" text-anchor="middle">12.3 GB</text><text class="t-tick" x="40" y="116" text-anchor="start">0 GB</text><text class="t-tick" x="720" y="116" text-anchor="end">80 GB</text><text class="t-tick" x="40" y="146" text-anchor="start">A fourth full conversation needs 17.2 GB and does not fit in the 12.3 GB left over.</text><text class="t-tick" x="40" y="166" text-anchor="start">(The model also needs some working memory, so the real room is a little smaller.)</text></svg><figcaption>An 80 GB H100 serving Llama 3.1 8B. After the 16.1 GB of weights, three full-length conversations of 17.2 GB each fill most of the card, and a fourth does not fit.</figcaption></figure>

Part 1 showed that throughput grows with the batch. Here the batch is three. Most conversations are much shorter than 128K tokens, so a real server fits more. But the point stands: the cache, not compute, sets how many conversations a GPU can hold.

> [!PAPER] Kwon et al. (2023) · §3 · page 4
> [![The paragraph on the large KV cache, with 800 KB of space, as much as 1.6 GB and only a few tens of requests could highlighted](/img/papers/vllm/p2-large-kv.png)](/img/papers/vllm/p2-large-kv.png)
>
> **What it says:** for the 13B-parameter OPT model, the KV cache of one token needs "800 KB of space": 2 × 5,120 × 40 × 2 bytes. A request can be 2,048 tokens long, so its cache can be "as much as 1.6 GB". Even if all GPU memory went to the cache, "only a few tens of requests could" fit.
>
> **What it means:**
>
> This is our formula on an older model. OPT-13B has 40 layers and stores keys and values of 5,120 numbers per layer (all its heads together). It has no shared heads, so its cost per token is more than six times that of Llama 3.1 8B.
>
> The last sentence is the key. The paper was written for a GPU with tens of gigabytes. With 1.6 GB per request, a few tens of requests fill it, and the batch can never grow past that. The paper's Figure 1 shows the same budget on a real card.

> [!PAPER] Kwon et al. (2023) · Figure 1 · page 1
> [![Figure 1 of the paper: on a 40 GB A100 serving a 13B model, the parameters take 26 GB (65%) and the KV cache more than 30%, with a small part for other data](/img/papers/vllm/p1-figure1.png)](/img/papers/vllm/p1-figure1.png)
>
> **What it says:** when a 13B model is served on an NVIDIA A100 with 40 GB, the parameters take 26 GB, about 65% of memory, and "persist in GPU memory throughout serving". The KV cache takes more than 30%, and it is "(de)allocated per serving request".
>
> **What it means:**
>
> Look at the left box. The grey part, the weights, never changes. The red part, the cache, is the only part that changes while the server runs. Requests come and go, and each one takes cache memory when it starts and returns it when it ends.
>
> So the whole serving problem lives in that red box. Ignore the right half of the figure for now. It shows what the paper's own system does with that box, which is the subject of Part 3.

## Shrinking the notes

The table above holds a surprise. Llama 2 7B and Llama 3.1 8B are about the same size, but the older model needs **four times** more cache per token. The difference is one number in the config: 32 KV heads against 8.

### Fewer key and value heads: MHA, GQA and MQA

In the original transformer, every query head has its own key head and value head. That is **multi-head attention (MHA)**. Llama 2 7B works this way: 32 query heads and 32 KV heads.

The idea behind the newer designs is that query heads can share keys and values. Each query head still asks its own question. But a group of query heads compares against the same keys and mixes the same values. Only the shared keys and values go into the cache.

<figure class="fig"><svg viewBox="0 0 760 220" role="img" aria-label="Multi-head attention caches one K and V per query head; grouped-query attention shares each KV head across several query heads; multi-query shares one."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-strong" x="125" y="28" text-anchor="middle">multi-head (MHA)</text><circle class="s2" cx="38" cy="60" r="8"/><circle class="s2" cx="63" cy="60" r="8"/><circle class="s2" cx="88" cy="60" r="8"/><circle class="s2" cx="113" cy="60" r="8"/><circle class="s2" cx="138" cy="60" r="8"/><circle class="s2" cx="163" cy="60" r="8"/><circle class="s2" cx="188" cy="60" r="8"/><circle class="s2" cx="213" cy="60" r="8"/><rect class="box-3" x="27.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="38.0" y="137" text-anchor="middle"></text><line class="edge" x1="38" y1="68" x2="38.0" y2="118"/><rect class="box-3" x="52.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="63.0" y="137" text-anchor="middle"></text><line class="edge" x1="63" y1="68" x2="63.0" y2="118"/><rect class="box-3" x="77.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="88.0" y="137" text-anchor="middle"></text><line class="edge" x1="88" y1="68" x2="88.0" y2="118"/><rect class="box-3" x="102.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="113.0" y="137" text-anchor="middle"></text><line class="edge" x1="113" y1="68" x2="113.0" y2="118"/><rect class="box-3" x="127.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="138.0" y="137" text-anchor="middle"></text><line class="edge" x1="138" y1="68" x2="138.0" y2="118"/><rect class="box-3" x="152.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="163.0" y="137" text-anchor="middle"></text><line class="edge" x1="163" y1="68" x2="163.0" y2="118"/><rect class="box-3" x="177.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="188.0" y="137" text-anchor="middle"></text><line class="edge" x1="188" y1="68" x2="188.0" y2="118"/><rect class="box-3" x="202.5" y="120" width="21" height="24" rx="5"/><text class="t-tick" x="213.0" y="137" text-anchor="middle"></text><line class="edge" x1="213" y1="68" x2="213.0" y2="118"/><text class="t-note" x="125" y="176" text-anchor="middle">8 KV heads to cache</text><text class="t-strong" x="375" y="28" text-anchor="middle">grouped-query (GQA)</text><circle class="s2" cx="288" cy="60" r="8"/><circle class="s2" cx="313" cy="60" r="8"/><circle class="s2" cx="338" cy="60" r="8"/><circle class="s2" cx="363" cy="60" r="8"/><circle class="s2" cx="388" cy="60" r="8"/><circle class="s2" cx="413" cy="60" r="8"/><circle class="s2" cx="438" cy="60" r="8"/><circle class="s2" cx="463" cy="60" r="8"/><rect class="box-3" x="305.5" y="120" width="40" height="24" rx="5"/><text class="t-tick" x="325.5" y="137" text-anchor="middle">KV</text><line class="edge" x1="288" y1="68" x2="325.5" y2="118"/><line class="edge" x1="313" y1="68" x2="325.5" y2="118"/><line class="edge" x1="338" y1="68" x2="325.5" y2="118"/><line class="edge" x1="363" y1="68" x2="325.5" y2="118"/><rect class="box-3" x="405.5" y="120" width="40" height="24" rx="5"/><text class="t-tick" x="425.5" y="137" text-anchor="middle">KV</text><line class="edge" x1="388" y1="68" x2="425.5" y2="118"/><line class="edge" x1="413" y1="68" x2="425.5" y2="118"/><line class="edge" x1="438" y1="68" x2="425.5" y2="118"/><line class="edge" x1="463" y1="68" x2="425.5" y2="118"/><text class="t-note" x="375" y="176" text-anchor="middle">2 KV heads to cache</text><text class="t-strong" x="625" y="28" text-anchor="middle">multi-query (MQA)</text><circle class="s2" cx="538" cy="60" r="8"/><circle class="s2" cx="563" cy="60" r="8"/><circle class="s2" cx="588" cy="60" r="8"/><circle class="s2" cx="613" cy="60" r="8"/><circle class="s2" cx="638" cy="60" r="8"/><circle class="s2" cx="663" cy="60" r="8"/><circle class="s2" cx="688" cy="60" r="8"/><circle class="s2" cx="713" cy="60" r="8"/><rect class="box-3" x="605.5" y="120" width="40" height="24" rx="5"/><text class="t-tick" x="625.5" y="137" text-anchor="middle">KV</text><line class="edge" x1="538" y1="68" x2="625.5" y2="118"/><line class="edge" x1="563" y1="68" x2="625.5" y2="118"/><line class="edge" x1="588" y1="68" x2="625.5" y2="118"/><line class="edge" x1="613" y1="68" x2="625.5" y2="118"/><line class="edge" x1="638" y1="68" x2="625.5" y2="118"/><line class="edge" x1="663" y1="68" x2="625.5" y2="118"/><line class="edge" x1="688" y1="68" x2="625.5" y2="118"/><line class="edge" x1="713" y1="68" x2="625.5" y2="118"/><text class="t-note" x="625" y="176" text-anchor="middle">1 KV head to cache</text><text class="t-tick" x="20" y="206" text-anchor="start">orange = query heads (not cached). aqua = key/value heads (cached for every token).</text></svg><figcaption>In multi-head attention every query head has its own key and value head. Grouped-query attention lets several query heads share one; multi-query attention shares a single one across all.</figcaption></figure>

- **Multi-query attention (MQA)** shares one key head and one value head across all query heads.
- **Grouped-query attention (GQA)** sits between the two: each group of query heads shares one KV head. Llama 3.1 8B has 32 query heads in 8 groups of 4.

What would Llama 3.1 8B cost under each design? Keep 32 layers, a head dim of 128 and 2 bytes, and change only the number of KV heads:

| Design | KV heads | Per token | 128K-token conversation |
|---|---|---|---|
| MHA | 32 | 512 KiB | 64 GiB |
| GQA (the real model) | 8 | 128 KiB | 16 GiB |
| MQA | 1 | 16 KiB | 2 GiB |

> [!PAPER] Shazeer (2019), Fast Transformer Decoding: One Write-Head is All You Need · Abstract · page 1
> [![The abstract of the multi-query attention paper, with the memory-bandwidth cost of repeatedly loading the large, the keys and values are shared across all of the different attention and only minor quality degradation highlighted](/img/writings/llm-inference/p2-mqa-abstract.png)](/img/writings/llm-inference/p2-mqa-abstract.png)
>
> **What it says:** writing one token at a time is often slow because of "the memory-bandwidth cost of repeatedly loading the large" keys and values. In multi-query attention, "the keys and values are shared across all of the different attention" heads. The models decode much faster with "only minor quality degradation".
>
> **What it means:**
>
> This 2019 paper saw the problem of this whole part early. Its first concern was not memory size but memory reading: every decode step loads all the keys and values again. A smaller cache is faster to read, as well as cheaper to keep. We come back to the reading cost below.
>
> The price is quality. With one KV head, all query heads must look through the same keys, so the model has fewer ways to find matches.

> [!PAPER] Ainslie et al. (2023), GQA · Abstract · page 1
> [![The end of the GQA abstract, with grouped-query attention (GQA), an intermediate (more than one, less than and achieves quality close to multi-head attention with comparable speed to MQA highlighted](/img/writings/llm-inference/p2-gqa-abstract.png)](/img/writings/llm-inference/p2-gqa-abstract.png)
>
> **What it says:** grouped-query attention uses "an intermediate (more than one, less than" the number of query heads) number of key-value heads. It "achieves quality close to multi-head attention with comparable speed to MQA".
>
> **What it means:**
>
> GQA is the middle setting of a dial. One KV head is fast but loses some quality. One KV head per query head keeps quality but makes a big cache. A few KV heads give almost the quality of the first with almost the speed of the second.
>
> That trade is why many open models since 2023, Llama 3 among them, use GQA. For Llama 3.1 8B it cuts the cache to a quarter of the MHA size.

### Compressed keys and values: MLA

A third design stores something smaller than the keys and values. **Multi-head latent attention (MLA)**, from DeepSeek, squeezes each token's keys and values into one short vector, called a latent vector. At attention time, the model expands it back. The cache holds only the short vector.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V2 · Abstract · page 1
> [![The DeepSeek-V2 abstract, with Multi-head Latent Attention (MLA) and reduces the KV cache by 93.3% highlighted](/img/writings/llm-inference/p2-deepseek-kv.png)](/img/writings/llm-inference/p2-deepseek-kv.png)
>
> **What it says:** DeepSeek-V2 uses "Multi-head Latent Attention (MLA)", which compresses the KV cache into a latent vector. Compared with the earlier DeepSeek 67B, it "reduces the KV cache by 93.3%".
>
> **What it means:**
>
> A cut of 93.3% means the new model keeps about one fifteenth of the cache per token of the old one. The two models differ in more than attention, so not all of that gap comes from MLA alone. But MLA is the part built for it.
>
> The [attention series](/writings/attention-2-mqa-gqa-mla/) builds MQA, GQA and MLA from scratch, with the shapes of every tensor.

### Fewer bytes per number: a quantised cache

The other lever is precision. The cache normally uses 2 bytes per number. An 8-bit format, such as FP8, stores each number in 1 byte. The numbers keep fewer digits, and the cache takes half the memory.

<figure class="fig"><svg viewBox="0 0 760 184" role="img" aria-label="The same key stored with 2 bytes per number takes 16 bytes; with 1 byte per number (FP8) it takes 8 bytes."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">one key of 8 numbers, stored two ways</text><text class="t-note" x="20" y="66" text-anchor="start">bfloat16</text><rect class="box-1" x="110" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="141" y="66" text-anchor="middle">0.81</text><rect class="box-1" x="176" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="207" y="66" text-anchor="middle">-1.2</text><rect class="box-1" x="242" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="273" y="66" text-anchor="middle">0.05</text><rect class="box-1" x="308" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="339" y="66" text-anchor="middle">2.3</text><rect class="box-1" x="374" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="405" y="66" text-anchor="middle">-0.4</text><rect class="box-1" x="440" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="471" y="66" text-anchor="middle">1.1</text><rect class="box-1" x="506" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="537" y="66" text-anchor="middle">0.66</text><rect class="box-1" x="572" y="46" width="62" height="30" rx="4"/><text class="t-tick" x="603" y="66" text-anchor="middle">-2.0</text><text class="t-val" x="650" y="66" text-anchor="start">16 bytes</text><text class="t-note" x="20" y="124" text-anchor="start">FP8</text><rect class="box-3" x="110" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="146" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="182" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="218" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="254" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="290" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="326" y="104" width="32" height="30" rx="4"/><rect class="box-3" x="362" y="104" width="32" height="30" rx="4"/><text class="t-tick" x="410" y="124" text-anchor="start">8 bytes: half the memory, slightly rounder numbers</text><text class="t-tick" x="20" y="168" text-anchor="start">Each number keeps fewer digits. The cache halves; quality usually changes very little, but measure it.</text></svg><figcaption>The same key stored at 2 bytes per number and at 1 byte per number. The 8-bit version takes half the memory, and each number is a little less exact.</figcaption></figure>

> [!DEFINITION] Quantisation and FP8
> **Quantisation** stores numbers with fewer bits than the model was trained with. **FP8** is an 8-bit floating-point format: 1 byte per number instead of 2. It halves the memory and loses a little precision.

For Llama 3.1 8B, an FP8 cache needs 64 KiB per token, so 8 GiB for a 128K-token conversation instead of 16 GiB. The vLLM engine offers this with the flag `--kv-cache-dtype fp8`. The cost is a small risk to accuracy, so measure it on your own tasks before you use it.

All of these levers make each token cheaper. None of them changes the shape of the problem. The cache still grows by one entry per token, for every conversation, as long as it runs.

## How many conversations fit on a real GPU

Now we can answer the question an inference engineer asks first about any deployment: how many conversations can this GPU hold at once? The size formula is all you need. Take the GPU's memory, take away the weights and some working memory, and divide what is left by the cache of one conversation.

Think of a car park. The building has a fixed number of spaces. Some are taken by staff cars all day, which are the weights. A few are kept free for deliveries, which is the working memory. The rest is for visitors. How many visitors fit depends on the size of each car, and a long conversation is a long car.

### A small example first

Take the H100 from above, with Llama 3.1 8B and conversations of 4,096 tokens.

1. **Memory for the cache.** Keep 10% of the 80 GB as working memory, which leaves 72 GB. The weights take 16.1 GB, so 55.9 GB is left for the cache.
2. **Cache per conversation.** At 128 KiB per token, 4,096 tokens need 512 MiB, or 0.54 GB.
3. **Divide.** 55.9 GB ÷ 0.54 GB = **104 conversations**.

> [!DEFINITION] Working memory
> GPU memory that the engine needs besides the weights and the KV cache. It holds the numbers a step creates on its way through the layers (activations), the CUDA graphs and the GPU's own software. We keep 10% of the GPU for it. This is an assumption: the real amount depends on the engine, the model and the batch.

With an FP8 cache, each token needs 64 KiB, so the same memory holds twice as many: **208**. Conversations of 32,768 tokens are 8 times larger, so only 13 fit, or 26 with FP8.

### Three GPUs, two models

A newer GPU has more memory, and the weights stay the same size. So most of the extra memory goes to the cache. These numbers come from the script `p2_gpu_capacity.py`, with the same 10% kept for working memory.

| Model and GPUs | Memory for the cache | 4,096 tokens, BF16 cache | 4,096 tokens, FP8 cache | 32,768 tokens, BF16 | 32,768 tokens, FP8 |
|---|---|---|---|---|---|
| 8B on 1 H100 (80 GB) | 55.9 GB | 104 | 208 | 13 | 26 |
| 8B on 1 H200 (141 GB) | 110.8 GB | 206 | 412 | 25 | 51 |
| 8B on 1 B200 (180 GB) | 145.9 GB | 271 | 543 | 33 | 67 |
| 70B on 4 H100 | 146.9 GB | 109 | 218 | 13 | 27 |
| 70B on 2 H200 | 112.7 GB | 83 | 167 | 10 | 20 |
| 70B on 2 B200 | 182.9 GB | 136 | 272 | 17 | 34 |

<figure class="fig"><svg viewBox="0 0 760 476" role="img" aria-label="How many 4,096-token conversations fit after the weights, on real GPUs, for Llama 3.1 8B and 70B, with a BF16 or an FP8 cache."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">Llama 3.1 8B (BF16 weights) on one GPU, conversations of 4,096 tokens</text><text class="t-note" x="120" y="64" text-anchor="end">H100 80 GB</text><rect class="box-2" x="130" y="44" width="53.535074986666665" height="30" rx="3"/><rect class="box-3" x="183.53507498666667" y="44" width="186.46492501333333" height="30" rx="3"/><rect class="box-ghost" x="370.0" y="44" width="26.666666666666668" height="30" rx="3"/><text class="t-tick" x="156.76753749333332" y="64" text-anchor="middle">weights</text><text class="t-note" x="276.76753749333335" y="64" text-anchor="middle">KV cache 56 GB</text><text class="t-tick" x="130" y="92" text-anchor="start">104 conversations with a BF16 cache, 208 with FP8</text><text class="t-note" x="120" y="128" text-anchor="end">H200 141 GB</text><rect class="box-2" x="130" y="108" width="53.535074986666665" height="30" rx="3"/><rect class="box-3" x="183.53507498666667" y="108" width="369.4649250133333" height="30" rx="3"/><rect class="box-ghost" x="553.0" y="108" width="47.0" height="30" rx="3"/><text class="t-tick" x="156.76753749333332" y="128" text-anchor="middle">weights</text><text class="t-note" x="368.2675374933333" y="128" text-anchor="middle">KV cache 111 GB</text><text class="t-tick" x="130" y="156" text-anchor="start">206 conversations with a BF16 cache, 412 with FP8</text><text class="t-note" x="120" y="192" text-anchor="end">B200 180 GB</text><rect class="box-2" x="130" y="172" width="53.535074986666665" height="30" rx="3"/><rect class="box-3" x="183.53507498666667" y="172" width="486.46492501333336" height="30" rx="3"/><rect class="box-ghost" x="670.0" y="172" width="60.0" height="30" rx="3"/><text class="t-tick" x="156.76753749333332" y="192" text-anchor="middle">weights</text><text class="t-note" x="426.76753749333335" y="192" text-anchor="middle">KV cache 146 GB</text><text class="t-tick" x="130" y="220" text-anchor="start">271 conversations with a BF16 cache, 543 with FP8</text><text class="t-title" x="20" y="242" text-anchor="start">Llama 3.1 70B (141.1 GB of weights) needs several GPUs; bars show the total over all of them</text><text class="t-note" x="120" y="280" text-anchor="end">4 x H100 80 GB</text><rect class="box-2" x="130" y="260" width="235.17902165333334" height="30" rx="3"/><rect class="box-3" x="365.17902165333334" y="260" width="244.82097834666666" height="30" rx="3"/><rect class="box-ghost" x="610.0" y="260" width="53.333333333333336" height="30" rx="3"/><text class="t-tick" x="247.58951082666667" y="280" text-anchor="middle">weights</text><text class="t-note" x="487.5895108266667" y="280" text-anchor="middle">KV 147 GB</text><text class="t-tick" x="130" y="308" text-anchor="start">109 conversations with a BF16 cache, 218 with FP8</text><text class="t-note" x="120" y="344" text-anchor="end">2 x H200 141 GB</text><rect class="box-2" x="130" y="324" width="235.17902165333334" height="30" rx="3"/><rect class="box-3" x="365.17902165333334" y="324" width="187.8209783466667" height="30" rx="3"/><rect class="box-ghost" x="553.0" y="324" width="47.0" height="30" rx="3"/><text class="t-tick" x="247.58951082666667" y="344" text-anchor="middle">weights</text><text class="t-note" x="459.0895108266667" y="344" text-anchor="middle">KV 113 GB</text><text class="t-tick" x="130" y="372" text-anchor="start">83 conversations with a BF16 cache, 167 with FP8</text><text class="t-note" x="120" y="408" text-anchor="end">2 x B200 180 GB</text><rect class="box-2" x="130" y="388" width="235.17902165333334" height="30" rx="3"/><rect class="box-3" x="365.17902165333334" y="388" width="304.82097834666666" height="30" rx="3"/><rect class="box-ghost" x="670.0" y="388" width="60.0" height="30" rx="3"/><text class="t-tick" x="247.58951082666667" y="408" text-anchor="middle">weights</text><text class="t-note" x="517.5895108266667" y="408" text-anchor="middle">KV 183 GB</text><text class="t-tick" x="130" y="436" text-anchor="start">136 conversations with a BF16 cache, 272 with FP8</text><rect class="box-2" x="20" y="449" width="14" height="14" rx="3"/><text class="t-tick" x="40" y="460" text-anchor="start">weights</text><rect class="box-3" x="170" y="449" width="14" height="14" rx="3"/><text class="t-tick" x="190" y="460" text-anchor="start">KV cache</text><rect class="box-ghost" x="320" y="449" width="14" height="14" rx="3"/><text class="t-tick" x="340" y="460" text-anchor="start">10% kept for working memory</text></svg><figcaption>GPU memory after the weights, and how many 4,096-token conversations fit in it. For Llama 3.1 8B a bigger GPU gives almost all of its extra memory to the cache. Llama 3.1 70B needs several GPUs before any memory is left for the cache.</figcaption></figure>

### Why 70B needs several GPUs

Llama 3.1 70B has 70.6 billion weights, which is 141.1 GB in BF16. With 10% kept for working memory, an H100 offers 72 GB and an H200 offers 127 GB. Neither can hold the weights. A B200 offers 162 GB, so the weights fit, but only 20.9 GB is left for the cache: 15 conversations of 4,096 tokens.

So the model is split across several GPUs, and each GPU holds a share of every layer. On H100s, two GPUs would hold the weights with only 2.9 GB to spare, which is about 2 conversations. Four H100s leave 146.9 GB for the cache. The split must divide the model's 8 KV heads evenly, so the usual counts are 1, 2, 4 or 8 GPUs. The weights and the cache are then shared out in the same way. [Part 3](llm-inference-3-vllm.md) shows how the cache works on split GPUs, and [Part 7](llm-inference-7-beyond-one-gpu.md) covers the split itself.

Note one more effect in the table. Per token, the 70B model's cache is 320 KiB, 2.5 times the 8B model's. So four H100s with 146.9 GB of cache hold only about as many 70B conversations as one H100 holds for the 8B model.

### What this number means for the batch

This number caps the batch from Part 1. A decode step can only include conversations whose notes are in GPU memory. On one H100 with 4,096-token conversations, the batch can never pass 104. But Part 1 showed that the ridge of an H100 is near a batch of 295. So memory, not compute, stops the batch first, as Part 1 said.

The table is also the best case. It assumes every byte of cache memory holds real tokens. The next sections show that the first serving systems got nowhere near that.

## The cache is read on every step

Size is the first cost of the cache. The second cost takes us back to Part 1.

Part 1 showed that a decode step is slow because it must read all the weights from GPU memory. Memory delivers only so many bytes per second. That rate is the **memory bandwidth**. Now look again at the attention lookup. To write one token, the new query is compared with the key of every earlier token, and every value is mixed in. So every decode step must also read the **whole cache** of every sequence in the batch.

On an H100, with a bandwidth of 3.35 TB per second, reading the 16.1 GB of Llama 3.1 8B weights takes at least 4.8 ms. Reading the 17.2 GB cache of one 128K-token conversation takes at least 5.1 ms more. For a long conversation, the cache costs as much reading time as the whole model, on every step.

<figure class="fig"><svg viewBox="0 0 760 200" role="img" aria-label="On an H100 a decode step must read the 16 GB of weights (4.8 ms) and, for one 128K-token chat, also its 17.2 GB cache (5.1 ms more)."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">the least time one decode step can take on an H100 (3.35 TB/s), Llama 3.1 8B</text><text class="t-note" x="200" y="76" text-anchor="end">short chat (cache tiny)</text><rect class="box-2" x="214" y="52" width="285.6417910447761" height="36" rx="6"/><text class="t-tick" x="357.820895522388" y="75" text-anchor="middle">weights 4.8 ms</text><text class="t-note" x="200" y="132" text-anchor="end">one 128K-token chat</text><rect class="box-2" x="214" y="108" width="285.6417910447761" height="36" rx="6"/><text class="t-tick" x="357.820895522388" y="131" text-anchor="middle">weights 4.8 ms</text><rect class="box-3" x="501.6417910447761" y="108" width="305.69914956417904" height="36" rx="6"/><text class="t-tick" x="655.4913658268656" y="131" text-anchor="middle">its cache 5.1 ms</text><text class="t-tick" x="20" y="186" text-anchor="start">Reading a long cache costs as much time as reading the whole model again, on every step.</text></svg><figcaption>The least time one decode step can take on an H100, from bandwidth alone. A short chat reads the weights. A 128K-token chat also reads its cache, which roughly doubles the time.</figcaption></figure>

Our small model shows the same effect in a real run. Here is the decode step time as the context grows, for 1 sequence and for 16 sequences at once:

<figure class="fig"><svg viewBox="0 0 760 320" role="img" aria-label="Decode step time as the context grows, for 1 and 16 sequences."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="264.0" x2="610" y2="264.0"/><text class="t-tick" x="56" y="268.0" text-anchor="end">0</text><line class="grid" x1="64" y1="205.5" x2="610" y2="205.5"/><text class="t-tick" x="56" y="209.5" text-anchor="end">10</text><line class="grid" x1="64" y1="147.0" x2="610" y2="147.0"/><text class="t-tick" x="56" y="151.0" text-anchor="end">20</text><line class="grid" x1="64" y1="88.5" x2="610" y2="88.5"/><text class="t-tick" x="56" y="92.5" text-anchor="end">30</text><line class="grid" x1="64" y1="30.0" x2="610" y2="30.0"/><text class="t-tick" x="56" y="34.0" text-anchor="end">40</text><text class="t-tick" x="64.0" y="284" text-anchor="middle">0</text><text class="t-tick" x="200.5" y="284" text-anchor="middle">2,048</text><text class="t-tick" x="337.0" y="284" text-anchor="middle">4,096</text><text class="t-tick" x="473.5" y="284" text-anchor="middle">6,144</text><text class="t-tick" x="610.0" y="284" text-anchor="middle">8,192</text><line class="axis" x1="64" y1="264" x2="610" y2="264"/><polyline class="l1" points="72.5,204.6 132.2,197.0 337.0,200.1 610.0,197.0"/><g class="mark"><title>batch 1, 128 tokens of context: 10.2 ms per step</title><circle class="s1 ring" cx="72.5" cy="204.6" r="5"/></g><g class="mark"><title>batch 1, 1,024 tokens of context: 11.5 ms per step</title><circle class="s1 ring" cx="132.2" cy="197.0" r="5"/></g><g class="mark"><title>batch 1, 4,096 tokens of context: 10.9 ms per step</title><circle class="s1 ring" cx="337.0" cy="200.1" r="5"/></g><g class="mark"><title>batch 1, 8,192 tokens of context: 11.5 ms per step</title><circle class="s1 ring" cx="610.0" cy="197.0" r="5"/></g><text class="t-note" x="622.0" y="200.98386263777502" text-anchor="start">batch 1: 11.5 ms</text><polyline class="l2" points="72.5,198.0 132.2,174.0 337.0,126.9 610.0,53.8"/><g class="mark"><title>batch 16, 128 tokens of context: 11.3 ms per step</title><circle class="s2 ring" cx="72.5" cy="198.0" r="5"/></g><g class="mark"><title>batch 16, 1,024 tokens of context: 15.4 ms per step</title><circle class="s2 ring" cx="132.2" cy="174.0" r="5"/></g><g class="mark"><title>batch 16, 4,096 tokens of context: 23.4 ms per step</title><circle class="s2 ring" cx="337.0" cy="126.9" r="5"/></g><g class="mark"><title>batch 16, 8,192 tokens of context: 35.9 ms per step</title><circle class="s2 ring" cx="610.0" cy="53.8" r="5"/></g><text class="t-note" x="622.0" y="57.82095995161217" text-anchor="start">batch 16: 35.9 ms</text><text class="t-tick" x="337.0" y="310" text-anchor="middle">tokens already in the context (each sequence)</text><text class="t-tick" x="14" y="18" text-anchor="start">ms per decode step</text></svg><figcaption>Time per decode step as the context grows, on Qwen2.5-0.5B (Apple M5 Pro GPU). One sequence barely notices; sixteen long sequences more than triple the step time.</figcaption></figure>

| Context per sequence | 1 sequence | 16 sequences |
|---|---|---|
| 128 tokens | 10.2 ms | 11.3 ms |
| 1,024 | 11.5 ms | 15.4 ms |
| 4,096 | 10.9 ms | 23.4 ms |
| 8,192 | 11.5 ms | 35.9 ms |

For **one** sequence the line is flat. This model's cache is small, 12 KiB per token, so even 8,192 tokens is only 96 MiB. That is tiny next to the 0.99 GB of weights. For **16** sequences at 8,192 tokens each, the cache is **1.5 GiB**, bigger than the weights. Every step must read all of it, and the step time more than triples.

So decode is limited by memory reads twice over. Each step reads the weights once, and it reads the cache of every sequence once. Long contexts and big batches make the second read the larger one. This is also why the shrinking methods above make decode faster, not only cheaper: fewer bytes in the cache means fewer bytes to read.

## Nobody knows how long the answer will be

So far the cache sounds like a sizing problem: work out the bytes and buy enough memory. The real difficulty is that a request's cache **grows while the request runs**, and nobody knows in advance how far it will grow.

When a request arrives, the server knows the length of its prompt. It does not know the length of the answer. The model decides that on its own, token by token, when it writes its end token. The answer could be 3 tokens or 3,000.

How much does the length vary? The vLLM series on this site asked a small model, OPT-125M, the question from Part 1 twenty times. It used random sampling and a limit of 200 tokens:

<figure class="fig"><svg viewBox="0 0 760 366" role="img" aria-label="Twenty sampled answers to one question range from 2 to 200 tokens; three hit the 200-token limit."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">20 answers to "What is the capital of France?" (OPT-125M, random sampling, limit 200 tokens)</text><rect class="box-3" x="60" y="40" width="6.0" height="11" rx="2"/><text class="t-tick" x="72.0" y="50" text-anchor="start">2</text><rect class="box-3" x="60" y="55" width="9.0" height="11" rx="2"/><text class="t-tick" x="75.0" y="65" text-anchor="start">3</text><rect class="box-3" x="60" y="70" width="9.0" height="11" rx="2"/><text class="t-tick" x="75.0" y="80" text-anchor="start">3</text><rect class="box-3" x="60" y="85" width="12.0" height="11" rx="2"/><text class="t-tick" x="78.0" y="95" text-anchor="start">4</text><rect class="box-3" x="60" y="100" width="15.0" height="11" rx="2"/><text class="t-tick" x="81.0" y="110" text-anchor="start">5</text><rect class="box-3" x="60" y="115" width="18.0" height="11" rx="2"/><text class="t-tick" x="84.0" y="125" text-anchor="start">6</text><rect class="box-3" x="60" y="130" width="45.0" height="11" rx="2"/><text class="t-tick" x="111.0" y="140" text-anchor="start">15</text><rect class="box-3" x="60" y="145" width="57.0" height="11" rx="2"/><text class="t-tick" x="123.0" y="155" text-anchor="start">19</text><rect class="box-3" x="60" y="160" width="69.0" height="11" rx="2"/><text class="t-tick" x="135.0" y="170" text-anchor="start">23</text><rect class="box-3" x="60" y="175" width="81.0" height="11" rx="2"/><text class="t-tick" x="147.0" y="185" text-anchor="start">27</text><rect class="box-3" x="60" y="190" width="84.0" height="11" rx="2"/><text class="t-tick" x="150.0" y="200" text-anchor="start">28</text><rect class="box-3" x="60" y="205" width="87.0" height="11" rx="2"/><text class="t-tick" x="153.0" y="215" text-anchor="start">29</text><rect class="box-3" x="60" y="220" width="105.0" height="11" rx="2"/><text class="t-tick" x="171.0" y="230" text-anchor="start">35</text><rect class="box-3" x="60" y="235" width="144.0" height="11" rx="2"/><text class="t-tick" x="210.0" y="245" text-anchor="start">48</text><rect class="box-3" x="60" y="250" width="219.0" height="11" rx="2"/><text class="t-tick" x="285.0" y="260" text-anchor="start">73</text><rect class="box-3" x="60" y="265" width="264.0" height="11" rx="2"/><text class="t-tick" x="330.0" y="275" text-anchor="start">88</text><rect class="box-3" x="60" y="280" width="453.0" height="11" rx="2"/><text class="t-tick" x="519.0" y="290" text-anchor="start">151</text><rect class="box-2" x="60" y="295" width="600.0" height="11" rx="2"/><text class="t-tick" x="666.0" y="305" text-anchor="start">200</text><rect class="box-2" x="60" y="310" width="600.0" height="11" rx="2"/><text class="t-tick" x="666.0" y="320" text-anchor="start">200</text><rect class="box-2" x="60" y="325" width="600.0" height="11" rx="2"/><text class="t-tick" x="666.0" y="335" text-anchor="start">200</text><text class="t-tick" x="696.0" y="320" text-anchor="start">hit the limit</text><text class="t-tick" x="60" y="352" text-anchor="start">answer length in tokens (one bar per answer, sorted from short to long)</text></svg><figcaption>Twenty answers to the same question from OPT-125M with random sampling. The lengths run from 2 to 200 tokens, and three answers hit the 200-token limit. Nothing in the prompt tells the server which length it will get.</figcaption></figure>

The same question gave answers from 2 to 200 tokens. Across 2,000 real answers in the Alpaca dataset, the median answer has 43 tokens. But one answer in ten has 127 tokens or more, and the longest has 499. ([The vLLM Part 2 page](/papers/vllm/part-2-why-memory-is-the-bottleneck/) has the script and its output.)

> [!PAPER] Kwon et al. (2023) · §3 · page 4
> [![The paragraph on unknown input and output lengths, with variability in their input and output lengths, exhaust available memory and deleting or swapping out the highlighted](/img/papers/vllm/p2-lengths.png)](/img/papers/vllm/p2-lengths.png)
>
> **What it says:** requests show "variability in their input and output lengths". As an output grows, its KV cache grows too, and it may "exhaust available memory". The system must then make hard choices, such as "deleting or swapping out the" KV cache of some requests.
>
> **What it means:**
>
> A server must accept a short question and a long document, side by side. Worse, a request that looks small can keep writing until memory runs out. Then some request must stop and give its memory back.
>
> So the server faces a planning problem. It must decide how much memory to give each request before it knows how much the request will use.

Every server sets an upper limit. The model has a **maximum length**, such as 2,048 tokens for OPT or 131,072 for Llama 3.1 8B. A request can also set its own limit on new tokens, often called `max_tokens`. These limits say how big the cache **could** get. They say nothing about how big it **will** get.

## How older systems stored the cache: one slab per request

To see why the size matters so much, we need one fact about how GPU programs keep data in memory.

> [!DEFINITION] Tensor and contiguous memory
> A **tensor** is an array of numbers with any number of dimensions. An example is the keys of one layer, with shape (sequences, heads, tokens, head dim). **Contiguous** memory means its numbers fill one unbroken run of addresses, with no gaps.

Deep learning libraries such as PyTorch expect each tensor to be contiguous. That makes reading fast and simple. To find token 5 of a cache that starts at address 1,000, the code computes 1,000 + 5 × (bytes per token) and jumps there. No lookup table is needed.

The catch comes when the cache grows. Picture a group of friends at a theatre who want seats side by side. If a stranger already sits in the next seat, a late friend cannot join the row. In memory, if another request's data sits right after request A's cache, A has no room to add its next token.

<figure class="fig"><svg viewBox="0 0 760 276" role="img" aria-label="A contiguous cache finds token i by simple arithmetic, but cannot grow when another request sits right behind it, so systems book the maximum room up front."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">a contiguous cache: token i sits at start + i × (bytes per token)</text><rect class="box-1" x="40" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="53.0" y="63.0" text-anchor="middle">0</text><rect class="box-1" x="69" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="82.0" y="63.0" text-anchor="middle">1</text><rect class="box-1" x="98" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="111.0" y="63.0" text-anchor="middle">2</text><rect class="box-1" x="127" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="140.0" y="63.0" text-anchor="middle">3</text><rect class="box-1" x="156" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="169.0" y="63.0" text-anchor="middle">4</text><rect class="box-3" x="185" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="198.0" y="63.0" text-anchor="middle">5</text><rect class="box-3" x="214" y="44" width="26" height="30" rx="3"/><text class="t-tick" x="227.0" y="63.0" text-anchor="middle">6</text><text class="t-tick" x="40" y="96" text-anchor="start">request A, start address 1000</text><text class="t-tick" x="40" y="114" text-anchor="start">token 5 is at 1000 + 5 × 128 KiB: one multiply, one add, no lookup</text><text class="t-title" x="20" y="156" text-anchor="start">the catch: A cannot grow if someone sits right behind it</text><rect class="box-1" x="40" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="53.0" y="191.0" text-anchor="middle">A</text><rect class="box-1" x="69" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="82.0" y="191.0" text-anchor="middle">A</text><rect class="box-1" x="98" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="111.0" y="191.0" text-anchor="middle">A</text><rect class="box-1" x="127" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="140.0" y="191.0" text-anchor="middle">A</text><rect class="box-1" x="156" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="169.0" y="191.0" text-anchor="middle">A</text><rect class="box-3" x="185" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="198.0" y="191.0" text-anchor="middle">A</text><rect class="box-3" x="214" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="227.0" y="191.0" text-anchor="middle">A</text><rect class="box-2" x="243" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="256.0" y="191.0" text-anchor="middle">B</text><rect class="box-2" x="272" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="285.0" y="191.0" text-anchor="middle">B</text><rect class="box-2" x="301" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="314.0" y="191.0" text-anchor="middle">B</text><rect class="box-2" x="330" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="343.0" y="191.0" text-anchor="middle">B</text><rect class="box-2" x="359" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="372.0" y="191.0" text-anchor="middle">B</text><rect class="box-2" x="388" y="172" width="26" height="30" rx="3"/><text class="t-tick" x="401.0" y="191.0" text-anchor="middle">B</text><rect class="box-ghost" x="417" y="172" width="26" height="30" rx="3"/><line class="path bad" x1="239" y1="228" x2="239" y2="206"/><text class="t-tick" x="40" y="244" text-anchor="start">A's next token must go in slot 7, but B already owns it.</text><text class="t-tick" x="40" y="262" text-anchor="start">So a contiguous system books all the room A might ever need on day one.</text></svg><figcaption>Top: in a contiguous cache, the address of any token is one multiply and one add. Bottom: request A cannot grow, because request B owns the slot right behind it.</figcaption></figure>

The systems before vLLM, such as FasterTransformer and Orca, chose the safe answer. When a request arrives, they book one contiguous **slab** of memory big enough for the longest output the request could ever produce. Then the cache can grow into its own slab, and it never runs into a neighbour.

> [!PAPER] Kwon et al. (2023) · §3.1 · page 4
> [![The opening of §3.1, with contiguous memory, contiguous tensor, statically allocate a chunk of memory and maximum possible sequence highlighted](/img/papers/vllm/p2-existing.png)](/img/papers/vllm/p2-existing.png)
>
> **What it says:** most deep learning operators need tensors stored in "contiguous memory", so earlier serving systems stored each request's KV cache as one "contiguous tensor". Because output lengths are unpredictable, they "statically allocate a chunk of memory" based on the request's "maximum possible sequence" length.
>
> **What it means:**
>
> "Statically allocate" means the size is fixed on arrival and never changes. The system books the worst case, the maximum length, because a smaller slab might run out in the middle of an answer.
>
> It is like a restaurant that seats every guest at a table for twelve, because any guest might bring eleven friends. Nobody is ever turned away halfway through dinner. But the room fills up with empty chairs.

That last line is the whole problem. A slab sized for the maximum is safe, but most of it sits empty. The vLLM paper splits that empty space into three kinds of waste. Each has a different cause, so we take them one at a time, each on a small memory of slots.

In the pictures below, GPU memory is a row of **slots**. One slot holds the keys and values of one token, for all layers. For Llama 3.1 8B, one slot would be 128 KiB.

## Waste 1: reserved slots

Picture a friend who saves a row of seats at a cinema for friends who will arrive during the trailers. The seats will be used. But until each friend arrives, the seat is empty, and nobody else may sit in it.

Request A shows the same thing. Its slab has 8 slots. Its prompt has 3 tokens, and over its life it will write 5 more, so in the end all 8 slots will hold a token.

<figure class="fig"><svg viewBox="0 0 760 270" role="img" aria-label="Reserved slots: request A books 8 slots; after its prompt 5 of them wait for future tokens, after 2 new tokens 3 still wait."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">request A books 8 slots; its prompt has 3 tokens; it will write 5</text><text class="t-note" x="170" y="70" text-anchor="end">after the prompt</text><rect class="box-1" x="184" y="50" width="26" height="30" rx="3"/><rect class="box-1" x="213" y="50" width="26" height="30" rx="3"/><rect class="box-1" x="242" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="271" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="300" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="329" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="358" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="387" y="50" width="26" height="30" rx="3"/><text class="t-tick" x="428" y="70" text-anchor="start">3 hold tokens, 5 wait</text><text class="t-note" x="170" y="120" text-anchor="end">after 2 new tokens</text><rect class="box-1" x="184" y="100" width="26" height="30" rx="3"/><rect class="box-1" x="213" y="100" width="26" height="30" rx="3"/><rect class="box-1" x="242" y="100" width="26" height="30" rx="3"/><rect class="box-3" x="271" y="100" width="26" height="30" rx="3"/><rect class="box-3" x="300" y="100" width="26" height="30" rx="3"/><rect class="box-4" x="329" y="100" width="26" height="30" rx="3"/><rect class="box-4" x="358" y="100" width="26" height="30" rx="3"/><rect class="box-4" x="387" y="100" width="26" height="30" rx="3"/><text class="t-tick" x="428" y="120" text-anchor="start">5 hold tokens, 3 wait</text><text class="t-note" x="170" y="170" text-anchor="end">finished</text><rect class="box-1" x="184" y="150" width="26" height="30" rx="3"/><rect class="box-1" x="213" y="150" width="26" height="30" rx="3"/><rect class="box-1" x="242" y="150" width="26" height="30" rx="3"/><rect class="box-3" x="271" y="150" width="26" height="30" rx="3"/><rect class="box-3" x="300" y="150" width="26" height="30" rx="3"/><rect class="box-3" x="329" y="150" width="26" height="30" rx="3"/><rect class="box-3" x="358" y="150" width="26" height="30" rx="3"/><rect class="box-3" x="387" y="150" width="26" height="30" rx="3"/><text class="t-tick" x="428" y="170" text-anchor="start">all 8 hold tokens</text><rect class="box-1" x="184" y="209" width="14" height="14" rx="3"/><text class="t-tick" x="204" y="220" text-anchor="start">prompt</text><rect class="box-3" x="260" y="209" width="14" height="14" rx="3"/><text class="t-tick" x="280" y="220" text-anchor="start">written</text><rect class="box-4" x="343" y="209" width="14" height="14" rx="3"/><text class="t-tick" x="363" y="220" text-anchor="start">reserved</text><text class="t-tick" x="20" y="256" text-anchor="start">The reserved slots are empty for now. They will be used, but until then no other request can use them.</text></svg><figcaption>Reserved slots. Request A books 8 slots. After its prompt, 5 slots wait for future tokens. After 2 new tokens, 3 still wait. Only at the very end are all 8 in use.</figcaption></figure>

Follow it step by step. After the prompt, 3 slots hold tokens and 5 are **reserved**: booked for tokens that do not exist yet. After 2 new tokens, 5 slots hold tokens and 3 are still reserved. Only when the answer ends are all 8 slots full.

> [!DEFINITION] Reserved slots
> Slots in a request's booking that will hold a token later, but are empty now. They are not lost forever, but while they wait, no other request can use them.

Now scale it up. Suppose a request will write a 500-token answer, and its slab already covers that length. On its first decode step, 499 of its answer slots are reserved and empty. One slot fills per step. So over the life of the answer, about half of its answer slots are empty at any moment.

The cost is not lost data. The cost is timing. The memory sits idle exactly while other requests in the queue could have used it.

## Waste 2: internal fragmentation

Now picture a booking at a restaurant: a table for eight. Only four guests come. The other four chairs stay empty all evening, and the restaurant cannot give them to anyone else, because they belong to your table.

Request B shows the same thing. Its slab has 8 slots, because 8 is its maximum length. Its prompt has 2 tokens. Its answer ends after 2 tokens, when the model writes its end token.

<figure class="fig"><svg viewBox="0 0 760 220" role="img" aria-label="Internal fragmentation: request B books 8 slots but its answer ends early, so 4 slots inside its booking are never used."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">request B books 8 slots; its prompt has 2 tokens; its answer stops after 2</text><text class="t-note" x="170" y="70" text-anchor="end">after the prompt</text><rect class="box-1" x="184" y="50" width="26" height="30" rx="3"/><rect class="box-1" x="213" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="242" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="271" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="300" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="329" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="358" y="50" width="26" height="30" rx="3"/><rect class="box-4" x="387" y="50" width="26" height="30" rx="3"/><text class="t-tick" x="428" y="70" text-anchor="start">nobody knows yet</text><text class="t-note" x="170" y="120" text-anchor="end">answer done</text><rect class="box-1" x="184" y="100" width="26" height="30" rx="3"/><rect class="box-1" x="213" y="100" width="26" height="30" rx="3"/><rect class="box-3" x="242" y="100" width="26" height="30" rx="3"/><rect class="box-3" x="271" y="100" width="26" height="30" rx="3"/><rect class="waste" x="300" y="100" width="26" height="30" rx="3"/><rect class="waste" x="329" y="100" width="26" height="30" rx="3"/><rect class="waste" x="358" y="100" width="26" height="30" rx="3"/><rect class="waste" x="387" y="100" width="26" height="30" rx="3"/><text class="t-tick" x="428" y="120" text-anchor="start">4 slots never used</text><rect class="box-1" x="184" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="204" y="170" text-anchor="start">prompt</text><rect class="box-3" x="260" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="280" y="170" text-anchor="start">written</text><rect class="box-4" x="343" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="363" y="170" text-anchor="start">reserved</text><rect class="waste" x="433" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="453" y="170" text-anchor="start">never used</text><text class="t-tick" x="20" y="206" text-anchor="start">The red slots sat empty for the whole life of B. That memory did nothing at all.</text></svg><figcaption>Internal fragmentation. Request B books 8 slots, but its answer ends after 2 tokens. The last 4 slots of its booking are never used, for the whole life of the request.</figcaption></figure>

Request B used 4 of its 8 slots. The other 4 were empty from start to finish. This is **internal fragmentation**: memory inside a booking that the owner never uses. "Internal" means the waste sits inside a slab that someone owns.

> [!DEFINITION] Internal fragmentation
> Memory inside a booked slab that its owner never fills, because the real need was smaller than the booking. Nobody else can use it while the owner holds the slab.

Note the difference from reserved slots. A reserved slot will be used later. An internally fragmented slot will never be used. But at the moment of booking, the server cannot tell the two apart. Both are just empty slots in a slab, and only the end of the answer shows which was which.

At real sizes this waste is large. Take a request with a 100-token prompt, a slab for OPT's maximum of 2,048 tokens, and an answer of 43 tokens, the Alpaca median. It fills 143 of its 2,048 slots, about 7%. The other 1,905 slots are internal fragmentation.

## Waste 3: external fragmentation

The third waste sits between slabs, not inside them. Picture a street with no marked parking bays. Cars of different lengths park along it, and some of them leave. Now the free spaces add up to enough room for a bus, but no single gap is long enough, so the bus cannot park.

Here is the same thing in a memory of 24 slots. Four requests have booked slabs of different sizes: A has 8, B has 4, C has 6 and D has 6. Then B and D finish and give their slabs back.

<figure class="fig"><svg viewBox="0 0 760 264" role="img" aria-label="External fragmentation: after B and D finish, 10 slots are free in gaps of 4 and 6, so a request for 7 slots in one piece is refused."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">a GPU memory of 24 slots, booked by four requests of different sizes</text><text class="t-note" x="170" y="70" text-anchor="end">all four running</text><rect class="box-1" x="184" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="194.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="206" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="216.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="228" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="238.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="250" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="260.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="272" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="282.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="294" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="304.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="316" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="326.0" y="69.0" text-anchor="middle">A</text><rect class="box-1" x="338" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="348.0" y="69.0" text-anchor="middle">A</text><rect class="box-3" x="360" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="370.0" y="69.0" text-anchor="middle">B</text><rect class="box-3" x="382" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="392.0" y="69.0" text-anchor="middle">B</text><rect class="box-3" x="404" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="414.0" y="69.0" text-anchor="middle">B</text><rect class="box-3" x="426" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="436.0" y="69.0" text-anchor="middle">B</text><rect class="box-1" x="448" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="458.0" y="69.0" text-anchor="middle">C</text><rect class="box-1" x="470" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="480.0" y="69.0" text-anchor="middle">C</text><rect class="box-1" x="492" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="502.0" y="69.0" text-anchor="middle">C</text><rect class="box-1" x="514" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="524.0" y="69.0" text-anchor="middle">C</text><rect class="box-1" x="536" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="546.0" y="69.0" text-anchor="middle">C</text><rect class="box-1" x="558" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="568.0" y="69.0" text-anchor="middle">C</text><rect class="box-3" x="580" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="590.0" y="69.0" text-anchor="middle">D</text><rect class="box-3" x="602" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="612.0" y="69.0" text-anchor="middle">D</text><rect class="box-3" x="624" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="634.0" y="69.0" text-anchor="middle">D</text><rect class="box-3" x="646" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="656.0" y="69.0" text-anchor="middle">D</text><rect class="box-3" x="668" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="678.0" y="69.0" text-anchor="middle">D</text><rect class="box-3" x="690" y="50" width="20" height="30" rx="3"/><text class="t-tick" x="700.0" y="69.0" text-anchor="middle">D</text><text class="t-note" x="170" y="130" text-anchor="end">B and D finish</text><rect class="box-1" x="184" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="194.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="206" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="216.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="228" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="238.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="250" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="260.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="272" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="282.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="294" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="304.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="316" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="326.0" y="129.0" text-anchor="middle">A</text><rect class="box-1" x="338" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="348.0" y="129.0" text-anchor="middle">A</text><rect class="box-ghost" x="360" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="382" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="404" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="426" y="110" width="20" height="30" rx="3"/><rect class="box-1" x="448" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="458.0" y="129.0" text-anchor="middle">C</text><rect class="box-1" x="470" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="480.0" y="129.0" text-anchor="middle">C</text><rect class="box-1" x="492" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="502.0" y="129.0" text-anchor="middle">C</text><rect class="box-1" x="514" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="524.0" y="129.0" text-anchor="middle">C</text><rect class="box-1" x="536" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="546.0" y="129.0" text-anchor="middle">C</text><rect class="box-1" x="558" y="110" width="20" height="30" rx="3"/><text class="t-tick" x="568.0" y="129.0" text-anchor="middle">C</text><rect class="box-ghost" x="580" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="602" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="624" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="646" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="668" y="110" width="20" height="30" rx="3"/><rect class="box-ghost" x="690" y="110" width="20" height="30" rx="3"/><path class="edge" d="M360,146 v6 H446 v-6" style="stroke-width:1.3"/><text class="t-tick" x="403.0" y="166" text-anchor="middle">gap of 4</text><path class="edge" d="M580,146 v6 H710 v-6" style="stroke-width:1.3"/><text class="t-tick" x="645.0" y="166" text-anchor="middle">gap of 6</text><text class="t-note" x="170" y="210" text-anchor="end">E needs 7</text><rect class="waste" x="184" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="194.0" y="209.0" text-anchor="middle">E</text><rect class="waste" x="206" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="216.0" y="209.0" text-anchor="middle">E</text><rect class="waste" x="228" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="238.0" y="209.0" text-anchor="middle">E</text><rect class="waste" x="250" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="260.0" y="209.0" text-anchor="middle">E</text><rect class="waste" x="272" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="282.0" y="209.0" text-anchor="middle">E</text><rect class="waste" x="294" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="304.0" y="209.0" text-anchor="middle">E</text><rect class="waste" x="316" y="190" width="20" height="30" rx="3"/><text class="t-tick" x="326.0" y="209.0" text-anchor="middle">E</text><text class="t-tick" x="350" y="210" text-anchor="start">refused: 10 slots are free, but no single gap holds 7</text><text class="t-tick" x="20" y="250" text-anchor="start">Free memory exists, yet it is cut into pieces that are each too small for the next request.</text></svg><figcaption>External fragmentation. After B and D finish, 10 slots are free, in a gap of 4 and a gap of 6. Request E needs 7 slots in one piece, so it is refused, although enough memory is free in total.</figcaption></figure>

Now 10 slots are free. Request E arrives and needs a slab of 7. Ten is more than seven, but the free slots sit in two gaps, of 4 and 6. Neither gap holds 7 in one piece, so E must wait. This is **external fragmentation**: free memory cut into gaps that are each too small to use.

> [!DEFINITION] External fragmentation
> Free memory between booked slabs, cut into gaps that are each too small for the next request. The memory is free, but it cannot be used.

Slabs of different sizes cause these gaps. They come in different sizes because each request has its own prompt length and its own limit. Many memory allocators also round each booking up, for example to a power of two, which makes the sizes even more uneven. The vLLM paper names one such allocator, the buddy allocator.

> [!DEFINITION] Buddy allocator
> A classic memory allocator that hands out blocks whose sizes are powers of two (512, 1,024, 2,048...). It is fast and simple, but a request for 513 slots gets 1,024, and the leftover sizes rarely fit the next request.

Why not slide the slabs together to close the gaps? That is called **compaction**, and it means copying the cache of every request that moves. A cache can be many gigabytes, and no request can write a token while its cache is in motion. The paper calls compaction "impractical" for a fast serving system, and it does nothing for the first two wastes anyway.

## All three wastes at once

In a real server, all three wastes happen at the same time. Here is one moment in a 24-slot memory, with two requests and the gaps around them:

<figure class="fig"><svg viewBox="0 0 760 278" role="img" aria-label="All three wastes in one 24-slot memory: 8 slots hold tokens, 3 are reserved, 5 are never used and 8 are free but in gaps too small for the next request."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20" y="24" text-anchor="start">one moment in a 24-slot memory: request A, a gap, request B, free slots</text><rect class="box-1" x="40" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="52.0" y="93.0" text-anchor="middle">A</text><rect class="box-1" x="68" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="80.0" y="93.0" text-anchor="middle">A</text><rect class="box-1" x="96" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="108.0" y="93.0" text-anchor="middle">A</text><rect class="box-3" x="124" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="136.0" y="93.0" text-anchor="middle">A</text><rect class="box-3" x="152" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="164.0" y="93.0" text-anchor="middle">A</text><rect class="box-4" x="180" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="192.0" y="93.0" text-anchor="middle">A</text><rect class="box-4" x="208" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="220.0" y="93.0" text-anchor="middle">A</text><rect class="waste" x="236" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="248.0" y="93.0" text-anchor="middle">A</text><rect class="box-ghost" x="264" y="74" width="24" height="30" rx="3"/><rect class="box-ghost" x="292" y="74" width="24" height="30" rx="3"/><rect class="box-ghost" x="320" y="74" width="24" height="30" rx="3"/><rect class="box-1" x="348" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="360.0" y="93.0" text-anchor="middle">B</text><rect class="box-1" x="376" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="388.0" y="93.0" text-anchor="middle">B</text><rect class="box-3" x="404" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="416.0" y="93.0" text-anchor="middle">B</text><rect class="box-4" x="432" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="444.0" y="93.0" text-anchor="middle">B</text><rect class="waste" x="460" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="472.0" y="93.0" text-anchor="middle">B</text><rect class="waste" x="488" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="500.0" y="93.0" text-anchor="middle">B</text><rect class="waste" x="516" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="528.0" y="93.0" text-anchor="middle">B</text><rect class="waste" x="544" y="74" width="24" height="30" rx="3"/><text class="t-tick" x="556.0" y="93.0" text-anchor="middle">B</text><rect class="box-ghost" x="572" y="74" width="24" height="30" rx="3"/><rect class="box-ghost" x="600" y="74" width="24" height="30" rx="3"/><rect class="box-ghost" x="628" y="74" width="24" height="30" rx="3"/><rect class="box-ghost" x="656" y="74" width="24" height="30" rx="3"/><rect class="box-ghost" x="684" y="74" width="24" height="30" rx="3"/><path class="edge" d="M40,112 v6 H260 v-6" style="stroke-width:1.3"/><text class="t-tick" x="150.0" y="132" text-anchor="middle">A books 8</text><path class="edge" d="M348,112 v6 H568 v-6" style="stroke-width:1.3"/><text class="t-tick" x="458.0" y="132" text-anchor="middle">B books 8</text><path class="edge" d="M264,68 v-6 H344 v6" style="stroke-width:1.3"/><text class="t-tick" x="304.0" y="50" text-anchor="middle">gap 3</text><path class="edge" d="M572,68 v-6 H708 v6" style="stroke-width:1.3"/><text class="t-tick" x="640.0" y="50" text-anchor="middle">free 5</text><rect class="box-1" x="40" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="60" y="170" text-anchor="start">prompt</text><rect class="box-3" x="170" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="190" y="170" text-anchor="start">written</text><rect class="box-4" x="300" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="320" y="170" text-anchor="start">reserved</text><rect class="waste" x="430" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="450" y="170" text-anchor="start">never used</text><rect class="box-ghost" x="560" y="159" width="14" height="14" rx="3"/><text class="t-tick" x="580" y="170" text-anchor="start">free</text><text class="t-tick" x="40" y="206" text-anchor="start">holding tokens: 8 slots;  reserved for later tokens: 3;  never used (internal fragmentation): 5</text><text class="t-tick" x="40" y="226" text-anchor="start">free: 8, but in gaps of 3 and 5, and the next request needs 6 in one piece (external fragmentation)</text><text class="t-title" x="40" y="262" text-anchor="start">Only 8 of 24 slots (33%) hold real keys and values.</text></svg><figcaption>All three wastes at one moment. Of 24 slots, 8 hold tokens, 3 are reserved for later tokens, 5 will never be used, and 8 are free but cut into gaps of 3 and 5, too small for the next request of 6.</figcaption></figure>

Count the slots. Request A holds 5 tokens, has 2 reserved slots and 1 slot it will never use. Request B holds 3 tokens, has 1 reserved slot and 4 it will never use. Between and after them, 8 slots are free, in gaps of 3 and 5, and the next request needs 6. So only 8 of the 24 slots, **33%**, hold real keys and values.

The vLLM paper draws the same picture with real words and real slab sizes:

> [!PAPER] Kwon et al. (2023) · Figure 3 · page 4
> [![Figure 3 of the paper: the memory of two requests. Request A has a 7-token prompt, 2 generated tokens, 2 reserved slots and 2038 slots never used. Request B has a 3-token prompt, 1 generated token, 1 reserved slot and 507 slots never used. A gap between them is external fragmentation](/img/papers/vllm/p2-figure3.png)](/img/papers/vllm/p2-figure3.png)
>
> **What it says:** "Three types of memory wastes", reserved, internal fragmentation and external fragmentation, "prevent other requests from fitting into the memory". Each slot holds the KV cache of one token.
>
> **What it means:**
>
> Read it from the left. Request A has a slab of 2,048 slots. Its prompt "Four score and seven years ago our" fills 7. It has written "fathers" and is writing "brought" now. It will still write "forth" and the end token: those 2 slots are **reserved**. About 2,038 more slots will never hold anything: **internal fragmentation**.
>
> Request B has a slab of 512 slots. Its prompt "You only live" fills 3, it is writing "once", and its end token is reserved. Its last 507 slots will stay empty. The grey slots between the two slabs are a gap too small for anyone: **external fragmentation**.
>
> This is our 24-slot picture with real sizes. The difference is scale. When both requests finish, only 16 of their 2,560 booked slots will ever have held a token.

> [!PAPER] Kwon et al. (2023) · §3.1 · page 4
> [![The paragraph that names the three sources of waste, with three primary sources of memory wastes, reserved slots for future tokens, internal fragmentation due to and external fragmentation from the memory allocator highlighted](/img/papers/vllm/p2-wastes.png)](/img/papers/vllm/p2-wastes.png)
> [![The rest of the paragraph, with pure memory waste, eventually used, could otherwise be used to and as low as 20.4% highlighted](/img/papers/vllm/p2-wastes-2.png)](/img/papers/vllm/p2-wastes-2.png)
>
> **What it says:** slab booking has "three primary sources of memory wastes": "reserved slots for future tokens", "internal fragmentation due to" booking for the maximum length, and "external fragmentation from the memory allocator", such as the buddy allocator. The two fragmentations are "pure memory waste". Reserved memory is "eventually used", but it takes space that "could otherwise be used to" serve other requests. The memory really used can be "as low as 20.4%".
>
> **What it means:**
>
> The paper sorts the three wastes by when you can see them. External fragmentation is visible at once: the gap exists before any token is written. Internal fragmentation shows only at the end, when the answer stops and its empty slots stay empty. Reserved slots are wasted only for a while, but that while can be most of the request's life.
>
> The last number, 20.4%, comes from a measurement in the paper's Figure 2, which we look at next.

## How much memory was really wasted

The authors measured the three wastes in real serving runs. Orca, the leading system they compared with, is not public, so they built their own version of it in three variants. The variants differ only in how much space each request books for its answer:

- **Orca (Max)** books the model's maximum length, 2,048 tokens, for every request.
- **Orca (Pow2)** books at most twice the real answer length, rounded up to a power of two. An answer of 25 tokens gets 32 slots.
- **Orca (Oracle)** books exactly the real answer length. No real system can do this, because it needs to know the future. It shows the best that slab booking could ever do.

> [!PAPER] Kwon et al. (2023) · Figure 2 · page 2
> [![Figure 2 of the paper: stacked bars of KV cache usage. Token states take 20.4% for Orca (Max), 26.8% for Orca (Pow2), 38.2% for Orca (Oracle) and 96.3% for vLLM; the rest is reservation, internal fragmentation and external fragmentation](/img/papers/vllm/p1-figure2.png)](/img/papers/vllm/p1-figure2.png)
>
> **What it says:** the figure shows the "average percentage of memory wastes" in each system during the paper's serving experiments. Each bar splits the KV cache memory into token states (green), reservation (orange), internal fragmentation (red) and external fragmentation and others (grey).
>
> **What it means:**
>
> Read only the green part first. It is the share of cache memory that held real keys and values. Orca (Max) used 20.4% of it. Orca (Pow2) used 26.8%. Even Orca (Oracle), which knew every answer length in advance, used only 38.2%.
>
> Now read the other colours with the three wastes in mind. In Orca (Max) the red part, internal fragmentation, is the largest at 57.3%: slabs of 2,048 for answers far shorter. In Orca (Oracle) the red part is gone, because it books the exact length. But reservation (25.2%) and external fragmentation (36.6%) are still there. Knowing the future removes only one of the three wastes.
>
> The fourth bar, vLLM at 96.3%, is the answer this series reaches in Part 3.

> [!PAPER] Kwon et al. (2023) · §1 · page 2
> [![The paragraph from the introduction on fragmentation, with pre-allocate, request's maximum length, other shorter requests cannot utilize any and only 20.4% - 38.2% of the KV cache memory is used to store the actual token states highlighted](/img/papers/vllm/p1-intro-frag.png)](/img/papers/vllm/p1-intro-frag.png)
>
> **What it says:** existing systems "pre-allocate a contiguous chunk of memory with the request's maximum length". Even if the length were known, while a chunk is held, "other shorter requests cannot utilize any" part of it. The authors' profiling shows "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states".
>
> **What it means:**
>
> The middle sentence is about reserved slots. Even a perfect guess of the length does not help. The slab is booked for the whole life of the request, while its later slots are still empty.
>
> So the headline number holds for all three variants. Between about one fifth and two fifths of the most valuable memory on the card held real data. The rest held nothing.

### What the waste costs: a smaller batch

Why does wasted memory matter so much? Go back to the rule from Part 1. Throughput grows with the number of sequences in each decode step. Each sequence needs its cache in memory. So memory that holds nothing is memory that could have held another sequence.

> [!PAPER] Kwon et al. (2023) · Figure 13 · page 10
> [![Figure 13 of the paper: the average number of batched requests when serving OPT-13B. On ShareGPT, Orca (Max) 7.00, Orca (Pow2) 9.81, Orca (Oracle) 13.62 and vLLM 30.42. On Alpaca, 7.00, 43.24, 72.75 and 132.44](/img/writings/llm-inference/p2-figure13.png)](/img/writings/llm-inference/p2-figure13.png)
>
> **What it says:** the figure shows the "Average number of batched requests" when the paper serves OPT-13B on two sets of real requests, ShareGPT and Alpaca.
>
> **What it means:**
>
> On ShareGPT, which has long conversations, Orca (Max) ran on average 7.00 requests at a time. Orca (Oracle), with the same GPU and the same model, ran 13.62. Nothing changed but the booking rule, and the batch almost doubled.
>
> So the waste is not only a memory problem. It is a throughput problem. Every slot that holds nothing is a sequence that is not in the batch. The last bar shows what vLLM does with the same memory, and Part 3 explains how.

## The question for Part 3

Put the three wastes side by side, and each one points to what a better design must do:

1. **Reserved slots** come from booking memory before the tokens exist. A better design gives memory to a request only when a new token actually needs it.
2. **Internal fragmentation** comes from booking for the maximum length. A better design never books more than a small amount beyond what the request has already written.
3. **External fragmentation** comes from slabs of different sizes that must each be one unbroken piece. A better design lets a request use free memory wherever it is, in pieces that all fit.

But the old design had a reason for its slabs. Attention reads keys and values fast because they sit in one contiguous run. If a request's cache is spread across memory, how does attention find its tokens? And how can memory grow in small pieces without a slab?

Operating systems faced the same problem with programs in the 1960s, and they solved it with an idea called **paging**. Part 3 shows how vLLM applied that idea to the KV cache, and what happens to each of the three wastes.

> [!TAKEAWAYS]
> - Attention is a lookup: the new token's **query** is compared with the **key** of every earlier token, and the **values** are mixed by those weights.
> - In a left-to-right model, old keys and values never change, so they are computed once and kept in the **KV cache**. At 8,192 tokens the next token took **449 ms** without it and **11.5 ms** with it.
> - Size per token $$= 2 \times \text{layers} \times \text{KV heads} \times \text{head dim} \times \text{bytes}$$. Llama 3.1 8B: **128 KiB** per token, **16 GiB** for one 128K-token conversation, about as much as its weights.
> - **GQA**, **MQA**, **MLA** and an **FP8** cache shrink it. GQA alone gives Llama 3.1 8B a quarter of the cache that multi-head attention would need.
> - On one H100 with 10% kept for working memory, Llama 3.1 8B fits **104** conversations of 4,096 tokens (**208** with FP8). That number caps the batch. Llama 3.1 70B needs at least two GPUs, and four H100s in practice.
> - Every decode step reads the whole cache of every sequence, so long contexts and big batches slow decode down.
> - Answer lengths are unknown, so older systems booked one **contiguous slab** per request for the maximum length.
> - Slabs waste memory three ways: **reserved slots** (empty until later tokens arrive), **internal fragmentation** (never used) and **external fragmentation** (free gaps too small to use).
> - The vLLM paper measured that only **20.4% to 38.2%** of KV memory held real tokens in such systems. The booking rule alone moved the average batch from 7.00 to 13.62 requests.

<details>
<summary>The scripts behind the numbers</summary>

The cache and no-cache timings and the decode step times come from the measurement below (Qwen2.5-0.5B, bfloat16, Apple M5 Pro GPU, median of 5 runs):

```python
import statistics, time
import torch
from transformers import AutoModelForCausalLM, DynamicCache

DEV, sync = "mps", torch.mps.synchronize          # "cuda", torch.cuda.synchronize on NVIDIA
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16).to(DEV).eval()

def median_ms(fn, reps=5):
    fn(); sync()
    ts = []
    for _ in range(reps):
        sync(); t = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1000

with torch.inference_mode():
    for n in (128, 512, 2048, 4096, 8192):
        x = torch.randint(0, model.config.vocab_size, (1, n), device=DEV)
        # no cache: producing the next token means running the whole sequence again
        no_cache = median_ms(lambda: model(input_ids=x, use_cache=False, logits_to_keep=1))
        # cache: run the prompt once, then each new token is a single-token step
        cache = DynamicCache()
        out = model(input_ids=x, past_key_values=cache, logits_to_keep=1)
        tok = out.logits[:, -1:].argmax(-1)
        def step():
            global tok
            o = model(input_ids=tok, past_key_values=cache, logits_to_keep=1)
            tok = o.logits[:, -1:].argmax(-1)
        print(f"{n:5d} tokens: no cache {no_cache:6.1f} ms, cache {median_ms(step):5.2f} ms")
```

The other new numbers come from `p2_kv_numbers.py`: the attention example, the measured cache, the sizes, the H100 times and the slot counts. Its full output:

```text
== 1. Attention as a lookup: a query, three keys, three values
  The  key [0.0, 1.0]  q.k = 0  /sqrt(2) = 0.00  weight = 0.088
  cat  key [3.0, 0.0]  q.k = 3  /sqrt(2) = 2.12  weight = 0.734
  sat  key [1.0, 1.0]  q.k = 1  /sqrt(2) = 0.71  weight = 0.178
  answer = 0.088*[1.0, 0.0] + 0.734*[0.0, 4.0] + 0.178*[2.0, 2.0] = [0.44, 3.29]
  PyTorch scaled_dot_product_attention gives [0.44, 3.29]

== 2. The KV cache of a real model: Qwen2.5-0.5B, bfloat16
  config: layers 24, query heads 14, KV heads 2, head_dim 64, device mps
  100 tokens: 24 layers, key tensor of layer 0 (1, 2, 100, 64) torch.bfloat16, cache 1,228,800 bytes
  200 tokens: 24 layers, key tensor of layer 0 (1, 2, 200, 64) torch.bfloat16, cache 2,457,600 bytes
  measured: 12,288 bytes per token; formula 2 x 24 x 2 x 64 x 2 = 12,288 bytes

== 3. KV bytes per token = 2 x layers x KV heads x head_dim x bytes
  Qwen2.5-0.5B    24 layers   2 KV heads   64 dims:     12 KiB/token,   1.5 GiB for 131,072 tokens
  Llama 3.1 8B    32 layers   8 KV heads  128 dims:    128 KiB/token,  16.0 GiB for 131,072 tokens
  Llama 3.1 70B   80 layers   8 KV heads  128 dims:    320 KiB/token,  40.0 GiB for 131,072 tokens
  Llama 2 7B      32 layers  32 KV heads  128 dims:    512 KiB/token,  64.0 GiB for 131,072 tokens
  Llama 3.1 8B with other attention layouts (32 layers, head_dim 128, 2 bytes):
    MHA, 32 KV heads   512 KiB/token,  64.0 GiB for 128K tokens
    GQA, 8 KV heads    128 KiB/token,  16.0 GiB for 128K tokens
    MQA, 1 KV head      16 KiB/token,   2.0 GiB for 128K tokens
    GQA + 8-bit cache    64 KiB/token,   8.0 GiB for 128K tokens
  Qwen2.5-0.5B, 16 sequences x 8,192 tokens: 1.50 GiB of cache

== 4. One decode step of Llama 3.1 8B on an H100 must read (3.35 TB/s)
  weights 16.1 GB -> at least 4.8 ms
  one 128K-token cache 17.2 GB (16 GiB) -> at least 5.1 ms more
  on an 80 GB H100, after the weights: room for 3 such conversations at full length (63.9 GB / 17.2 GB)

== 5. Toy GPU memory: a row of 24 slots, one slot = the K and V of one token (all layers)
  reserved: A books 8, prompt 3, wrote 2 -> holds 5, 3 reserved for tokens still to come
  internal: B books 8, uses 4, 4 never used
  external: AAAAAAAABBBBCCCCCCDDDDDD -> B, D finish -> AAAAAAAA....CCCCCC......; free 10 in gaps [4, 6]; E needs 7 in one piece: refused
  all three: PPPWWRRI___PPWRIIII_____ (24 slots): tokens 8, reserved 3, never used 5, free 8 in gaps of 3 and 5 (next request needs 6: refused); holding tokens 33%
```

The GPU capacity table comes from `p2_gpu_capacity.py`, which only does arithmetic on the configs and the vendor specs. Its full output:

```text
Llama 3.1 8B: 8,030,261,248 weights -> 16.1 GB in BF16; KV per token BF16 128 KiB, FP8 64 KiB
Llama 3.1 70B: 70,553,706,496 weights -> 141.1 GB in BF16; KV per token BF16 320 KiB, FP8 160 KiB
assumption: 10% of GPU memory kept for working memory

Llama 3.1 8B   on 1 x H100 80GB : weights  16.1 GB per GPU, KV memory   55.9 GB in total =   426,784 tokens BF16 /   853,568 FP8 | conversations that fit: 4,096 tok BF16  104, FP8  208; 32,768 tok BF16  13, FP8  26
Llama 3.1 8B   on 1 x H200 141GB: weights  16.1 GB per GPU, KV memory  110.8 GB in total =   845,638 tokens BF16 / 1,691,276 FP8 | conversations that fit: 4,096 tok BF16  206, FP8  412; 32,768 tok BF16  25, FP8  51
Llama 3.1 8B   on 1 x B200 180GB: weights  16.1 GB per GPU, KV memory  145.9 GB in total = 1,113,429 tokens BF16 / 2,226,859 FP8 | conversations that fit: 4,096 tok BF16  271, FP8  543; 32,768 tok BF16  33, FP8  67

Llama 3.1 70B  on 2 x H100 80GB : weights  70.6 GB per GPU, KV memory    2.9 GB in total =     8,827 tokens BF16 /    17,654 FP8 | conversations that fit: 4,096 tok BF16    2, FP8    4; 32,768 tok BF16   0, FP8   0
Llama 3.1 70B  on 4 x H100 80GB : weights  35.3 GB per GPU, KV memory  146.9 GB in total =   448,280 tokens BF16 /   896,561 FP8 | conversations that fit: 4,096 tok BF16  109, FP8  218; 32,768 tok BF16  13, FP8  27
Llama 3.1 70B  on 2 x H200 141GB: weights  70.6 GB per GPU, KV memory  112.7 GB in total =   343,910 tokens BF16 /   687,820 FP8 | conversations that fit: 4,096 tok BF16   83, FP8  167; 32,768 tok BF16  10, FP8  20
Llama 3.1 70B  on 4 x H200 141GB: weights  35.3 GB per GPU, KV memory  366.5 GB in total = 1,118,446 tokens BF16 / 2,236,893 FP8 | conversations that fit: 4,096 tok BF16  273, FP8  546; 32,768 tok BF16  34, FP8  68
Llama 3.1 70B  on 1 x B200 180GB: weights 141.1 GB per GPU, KV memory   20.9 GB in total =    63,759 tokens BF16 /   127,518 FP8 | conversations that fit: 4,096 tok BF16   15, FP8   31; 32,768 tok BF16   1, FP8   3
Llama 3.1 70B  on 2 x B200 180GB: weights  70.6 GB per GPU, KV memory  182.9 GB in total =   558,143 tokens BF16 / 1,116,287 FP8 | conversations that fit: 4,096 tok BF16  136, FP8  272; 32,768 tok BF16  17, FP8  34

70B on one H100 80GB: usable 72 GB vs weights 141.1 GB -> does not fit
70B on one H200 141GB: usable 127 GB vs weights 141.1 GB -> does not fit
70B on one B200 180GB: usable 162 GB vs weights 141.1 GB -> fits, 20.9 GB left for KV
```

</details>

## References

1. W. Kwon et al. [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180). SOSP 2023. (Figures 1, 2, 3 and 13; 800 KB per token for OPT-13B; 20.4% to 38.2% of KV memory used in existing systems.)
2. A. Vaswani et al. [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
3. N. Shazeer. [*Fast Transformer Decoding: One Write-Head is All You Need*](https://arxiv.org/abs/1911.02150). 2019.
4. J. Ainslie et al. [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). EMNLP 2023.
5. DeepSeek-AI. [*DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model*](https://arxiv.org/abs/2405.04434). 2024.
6. G.-I. Yu et al. [*Orca: A Distributed Serving System for Transformer-Based Generative Models*](https://www.usenix.org/conference/osdi22/presentation/yu). OSDI 2022.
7. NVIDIA, [H100 Tensor Core GPU](https://www.nvidia.com/en-us/data-center/h100/) (3.35 TB/s memory bandwidth, 80 GB), [H200](https://www.nvidia.com/en-us/data-center/h200/) (141 GB) and the [Blackwell datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf) (HGX B200: 180 GB per GPU).
8. vLLM documentation: [Quantized KV cache](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache.html).
9. Model configs: [Llama 3.1 8B](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct), [Llama 3.1 70B](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).

**Next:** [Part 3: How vLLM Serves Thousands](llm-inference-3-vllm.md), where paging removes the three wastes.
