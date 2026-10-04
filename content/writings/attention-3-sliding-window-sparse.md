---
title: "Attention, Part 3: Sliding-Window and Sparse Attention"
description: "Why looking at every earlier token gets too expensive, and the three ways today's models avoid it: sliding windows (Mistral), mixing local and global layers (Gemma 2, Gemma 3, gpt-oss), and letting the model pick its own tokens (DeepSeek Sparse Attention). Explained from zero, with equations, quotes from the papers, and real experiments: a lightning indexer trained on Qwen2.5-0.5B keeps quality close to full attention while each token reads only 64 of up to 2,048 earlier tokens."
date: 2026-10-04
tags: [attention, sparse-attention, long-context, llm]
series: "Attention, From the Ground Up"
series_part: 3
motif: graph
accent: "#f0a35e"
---

[Part 2](attention-2-mqa-gqa-mla.md) made each token **store less**: fewer key/value heads (MQA, GQA) or a compressed latent (MLA). But every token still looked at **every** earlier token.

This part attacks that second problem: **how many tokens each token looks at**. It is the biggest cost of long conversations, long documents and long chains of reasoning, and it is why every model that handles 128,000 tokens or more uses at least one of the ideas below.

This is a long part, so here is the map.

> [!TIP] What you will learn
> 1. **Why full attention gets expensive:** the cost grows with the *square* of the length.
> 2. **Sliding-window attention:** each token looks only at the last W tokens (Mistral 7B).
> 3. **How stacked layers still see far**, and the **rolling buffer cache** that keeps memory fixed.
> 4. **Local and global layers mixed together** (Gemma 2, Gemma 3, gpt-oss), with real memory numbers.
> 5. **Attention sinks again:** a real experiment where cutting off the first token breaks a model.
> 6. **DeepSeek Sparse Attention:** a tiny "lightning indexer" picks which tokens to read. I train one on a real model and test it.

As in the earlier parts, every technical word gets a yellow box the first time it appears, every number comes from code I ran, and every quote links to its paper.

> [!TIP] Quick recap
> Each token makes a **query**, a **key** and a **value**. Its query is compared with the keys of earlier tokens (dot products), softmax turns the scores into weights that add up to 1, and the weights mix the values. While a model writes, it keeps all earlier keys and values in the **KV cache** ([Part 2](attention-2-mqa-gqa-mla.md)).

## 1. The problem: everyone looks at everyone

In full attention, token number $$i$$ compares its query with the keys of tokens $$1, 2, \dots, i$$. So for a text of $$T$$ tokens, the number of query-key comparisons in **one head of one layer** is:

$$
1 + 2 + 3 + \dots + T = \frac{T\,(T+1)}{2}
$$

> [!DEFINITION] Quadratic growth
> When a cost grows with the **square** of the size ($$T^2$$), doubling the length makes it about **four times** bigger, and ten times the length makes it about **a hundred times** bigger. Attention's comparisons grow this way.

Here is what that means in numbers:

| Text length $$T$$ | Comparisons, full attention | Comparisons, window of 1,024 | Full ÷ window |
|---|---|---|---|
| 1,024 tokens | 524,800 | 524,800 | 1.0× |
| 8,192 tokens | 33,558,528 | 7,864,832 | 4.3× |
| 131,072 tokens (128K) | 8,590,000,128 | 133,693,952 | 64.3× |

At 128K tokens, full attention makes **8.6 billion** comparisons, in every head of every layer. A window of 1,024 tokens makes 64 times fewer.

In arithmetic operations, each comparison is a dot product of length $$d$$ (about $$2d$$ multiply-and-add steps), and the weighted mix of values costs about the same again. So per head and layer:

$$
\text{work}_{\text{full}} \approx 2 \cdot 2d \cdot \frac{T(T+1)}{2} \;\approx\; 2\,d\,T^2
\qquad\qquad
\text{work}_{\text{window}} \approx 2 \cdot 2d \cdot T\,W \;=\; 4\,d\,T\,W
$$

where $$T$$ is the text length, $$W$$ the window and $$d$$ the head size. The first grows with $$T^2$$, the second only with $$T$$: doubling a document doubles the windowed cost but quadruples the full cost.

There is also a memory cost while the model writes. The KV cache grows by one token at every step, and **every step reads all of it**. Twice the conversation means twice the memory and twice the reading per step.

> [!NOTE] The hopeful observation
> Most attention is **local**. In [Part 1](attention-1-self-attention.md) we saw a head that put 84% of its attention on the *previous token*, and many heads that dump attention on the *first token*. In the Qwen experiment later in this part, just the last 64 tokens plus the first 4 already catch **72%** of all attention, averaged over every layer (all heads together). So most of those 8.6 billion comparisons produce weights close to zero.

The rest of this part is about skipping the comparisons that do not matter, without skipping the ones that do.

<figure class="fig"><svg viewBox="0 0 760 284" role="img" aria-label="Three attention patterns on 12 tokens: full causal attention, a sliding window of 4, and a sparse top-k selection."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="122.0" y="32.0" text-anchor="middle">full (causal)</text><rect class="cell" x="21" y="45" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="38" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="55" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="72" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="89" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="106" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="123" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="140" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="45" width="15" height="15" rx="2"/><rect class="cell" x="21" y="62" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="62" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="55" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="72" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="89" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="106" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="123" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="140" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="62" width="15" height="15" rx="2"/><rect class="cell" x="21" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="72" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="89" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="106" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="123" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="140" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="79" width="15" height="15" rx="2"/><rect class="cell" x="21" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="89" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="106" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="123" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="140" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="96" width="15" height="15" rx="2"/><rect class="cell" x="21" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="106" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="123" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="140" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="113" width="15" height="15" rx="2"/><rect class="cell" x="21" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="123" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="140" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="130" width="15" height="15" rx="2"/><rect class="cell" x="21" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="123" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="140" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="157" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="147" width="15" height="15" rx="2"/><rect class="cell" x="21" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="123" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="140" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="157" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="174" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="164" width="15" height="15" rx="2"/><rect class="cell" x="21" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="123" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="140" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="157" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="174" y="181" width="15" height="15" rx="2"/><rect class="cell-masked" x="191" y="181" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="181" width="15" height="15" rx="2"/><rect class="cell" x="21" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="123" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="140" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="157" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="174" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="191" y="198" width="15" height="15" rx="2"/><rect class="cell-masked" x="208" y="198" width="15" height="15" rx="2"/><rect class="cell" x="21" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="123" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="140" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="157" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="174" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="191" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="208" y="215" width="15" height="15" rx="2"/><rect class="cell" x="21" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="38" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="55" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="72" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="89" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="106" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="123" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="140" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="157" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="174" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="191" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="208" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><text class="t-title" x="374.0" y="32.0" text-anchor="middle">sliding window, W = 4</text><rect class="cell" x="273" y="45" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="290" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="307" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="324" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="341" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="358" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="375" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="392" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="45" width="15" height="15" rx="2"/><rect class="cell" x="273" y="62" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="290" y="62" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="307" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="324" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="341" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="358" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="375" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="392" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="62" width="15" height="15" rx="2"/><rect class="cell" x="273" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="290" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="307" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="324" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="341" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="358" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="375" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="392" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="79" width="15" height="15" rx="2"/><rect class="cell" x="273" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="290" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="307" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="324" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="341" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="358" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="375" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="392" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="96" width="15" height="15" rx="2"/><rect class="cell" x="273" y="113" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="307" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="324" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="341" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="358" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="375" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="392" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="113" width="15" height="15" rx="2"/><rect class="cell" x="273" y="130" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="130" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="324" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="341" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="358" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="375" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="392" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="130" width="15" height="15" rx="2"/><rect class="cell" x="273" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="324" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="341" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="358" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="375" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="392" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="409" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="147" width="15" height="15" rx="2"/><rect class="cell" x="273" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="324" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="341" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="358" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="375" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="392" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="409" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="426" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="164" width="15" height="15" rx="2"/><rect class="cell" x="273" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="324" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="341" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="358" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="375" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="392" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="409" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="426" y="181" width="15" height="15" rx="2"/><rect class="cell-masked" x="443" y="181" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="181" width="15" height="15" rx="2"/><rect class="cell" x="273" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="324" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="341" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="358" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="375" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="392" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="409" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="426" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="443" y="198" width="15" height="15" rx="2"/><rect class="cell-masked" x="460" y="198" width="15" height="15" rx="2"/><rect class="cell" x="273" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="324" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="341" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="358" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="375" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="392" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="409" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="426" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="443" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="460" y="215" width="15" height="15" rx="2"/><rect class="cell" x="273" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="290" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="307" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="324" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="341" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="358" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="375" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="392" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="409" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="426" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="443" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="460" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><text class="t-title" x="626.0" y="32.0" text-anchor="middle">sparse: top-k picked</text><rect class="cell" x="525" y="45" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="542" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="559" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="576" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="593" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="610" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="627" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="644" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="45" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="45" width="15" height="15" rx="2"/><rect class="cell" x="525" y="62" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="62" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="559" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="576" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="593" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="610" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="627" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="644" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="62" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="62" width="15" height="15" rx="2"/><rect class="cell" x="525" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="559" y="79" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="576" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="593" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="610" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="627" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="644" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="79" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="79" width="15" height="15" rx="2"/><rect class="cell" x="525" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="96" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="576" y="96" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="593" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="610" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="627" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="644" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="96" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="96" width="15" height="15" rx="2"/><rect class="cell" x="525" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="113" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="113" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="576" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="593" y="113" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="610" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="627" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="644" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="113" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="113" width="15" height="15" rx="2"/><rect class="cell" x="525" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="130" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="576" y="130" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="593" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="610" y="130" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="627" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="644" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="130" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="130" width="15" height="15" rx="2"/><rect class="cell" x="525" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="559" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="576" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="593" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="610" y="147" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="627" y="147" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="644" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="661" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="147" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="147" width="15" height="15" rx="2"/><rect class="cell" x="525" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="576" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="593" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="610" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="627" y="164" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="644" y="164" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="661" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="678" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="164" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="164" width="15" height="15" rx="2"/><rect class="cell" x="525" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="576" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="593" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="610" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="627" y="181" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="644" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="661" y="181" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="678" y="181" width="15" height="15" rx="2"/><rect class="cell-masked" x="695" y="181" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="181" width="15" height="15" rx="2"/><rect class="cell" x="525" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="576" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="593" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="610" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="627" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="644" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="661" y="198" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="678" y="198" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="695" y="198" width="15" height="15" rx="2"/><rect class="cell-masked" x="712" y="198" width="15" height="15" rx="2"/><rect class="cell" x="525" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="576" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="593" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="610" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="627" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="644" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="661" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="678" y="215" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="695" y="215" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell-masked" x="712" y="215" width="15" height="15" rx="2"/><rect class="cell" x="525" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="542" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="559" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="576" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="593" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="610" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="627" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="644" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="661" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><rect class="cell" x="678" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="695" y="232" width="15" height="15" rx="2" style="fill-opacity:0.07"/><rect class="cell" x="712" y="232" width="15" height="15" rx="2" style="fill-opacity:0.9"/><text class="t-tick" x="20.0" y="270.0" text-anchor="start">Each row is one token; each bright square is a token it may look at. Faint squares are skipped. Dark empty squares are the future.</text></svg><figcaption>Three patterns on 12 tokens. Full attention: every token sees all earlier tokens. Sliding window: each token sees only itself and the 3 before it. Sparse top-k: each token sees a small, chosen set (here an illustrative random pick).</figcaption></figure>

## 2. Sliding-window attention (SWA)

The simplest fix: **each token looks only at the last W tokens**, itself included. Nothing older.

> [!DEFINITION] Sliding window and window size W
> A fixed-size "view" that moves along with the text. With window size $$W$$, token $$i$$ may attend to tokens $$i-W+1, \dots, i$$. As the text moves forward, the window slides with it. Typical sizes: 128 (gpt-oss), 1,024 (Gemma 3), 4,096 (Mistral 7B, Gemma 2).

In Part 1 we wrote attention with a **mask** $$M$$ that blocks the future. Sliding-window attention is the same formula with a stricter mask:

$$
\text{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d}} + M\right) V,
\qquad
M_{ij} = \begin{cases} 0 & \text{if } 0 \le i - j < W \\ -\infty & \text{otherwise} \end{cases}
$$

where:

- $$i$$ is the position of the query token and $$j$$ the position of a key token;
- $$i - j$$ is how far back token $$j$$ is from token $$i$$;
- $$i - j < 0$$ would be the future (blocked, as before);
- $$i - j \ge W$$ is too far back (newly blocked);
- $$-\infty$$ becomes a weight of exactly 0 after softmax.

That is the whole idea. In code it is one extra condition on the mask:

```python
def window_mask(T, W, device=None):
    """True where query i may look at key j: j <= i (causal) and i - j < W (inside the window)."""
    i = torch.arange(T, device=device)[:, None]
    j = torch.arange(T, device=device)[None, :]
    return (j <= i) & (i - j < W)


def attend(q, k, v, allowed):
    """Plain attention with a boolean 'allowed' mask. q, k, v: (..., T, d)."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    scores = scores.masked_fill(~allowed, float('-inf'))
    return torch.softmax(scores, -1) @ v
```

For 10 tokens and $$W = 4$$, the mask looks like this (1 = may look, 0 = may not). Each row is one token; it sees itself and the three before it:

```text
1 0 0 0 0 0 0 0 0 0
1 1 0 0 0 0 0 0 0 0
1 1 1 0 0 0 0 0 0 0
1 1 1 1 0 0 0 0 0 0
0 1 1 1 1 0 0 0 0 0
0 0 1 1 1 1 0 0 0 0
0 0 0 1 1 1 1 0 0 0
0 0 0 0 1 1 1 1 0 0
0 0 0 0 0 1 1 1 1 0
0 0 0 0 0 0 1 1 1 1
```

> [!NOTE] Two ways of counting the window
> The Mistral paper says position $$i$$ attends to positions "between $$i - W$$ and $$i$$", which is $$W + 1$$ tokens if both ends are included. The Hugging Face library, which most people use to run these models, uses $$i - j < W$$: exactly $$W$$ tokens, the current one included. I use the Hugging Face rule everywhere in this part. The difference of one token never matters in practice, but it explains why formulas in different sources differ by one.

### Proof that it works

I checked three things in `part3_sparse.py` (24 tokens, $$W = 6$$, 64-bit numbers):

```text
1. SWA vs PyTorch SDPA with the same mask:  max |diff| = 4.4e-16
   window wider than the text vs causal:   max |diff| = 0.0e+00
   change all 18 tokens outside the window of the last token:
      last output with sliding window moves by 0.0e+00
      last output with full attention moves by 0.65
```

1. My function agrees with PyTorch's built-in attention to $$4.4 \times 10^{-16}$$ (rounding noise).
2. A window wider than the text is exactly ordinary causal attention (difference 0).
3. The real test of "only the last W tokens matter": I replaced **all 18 tokens outside the last token's window** with random new ones. With the sliding window, the last token's output did not move at all (difference exactly 0). With full attention, it moved by 0.65.

## 3. "But then the model forgets everything older than W?"

Not quite, and this is the clever part. A model has many layers stacked on top of each other. Layer 2 reads the *outputs* of layer 1, and each of those outputs already mixed information from its own window.

> [!DEFINITION] Receptive field
> All the input tokens that can affect one output, directly or indirectly. In a single sliding-window layer it is just the window. Across stacked layers it grows, like a rumour passed from person to person.

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="With a sliding window, each layer can only look W tokens back, but stacking layers lets information travel further: the reach grows by W minus 1 per layer."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-tick" x="20.0" y="235.0" text-anchor="start">input tokens</text><circle class="node" cx="150" cy="230" r="6"/><circle class="node" cx="200" cy="230" r="6"/><circle class="node" cx="250" cy="230" r="6"/><circle class="node" cx="300" cy="230" r="6"/><circle class="node" cx="350" cy="230" r="6"/><circle class="s1" cx="400" cy="230" r="9"/><circle class="s1" cx="450" cy="230" r="9"/><circle class="s1" cx="500" cy="230" r="9"/><circle class="s1" cx="550" cy="230" r="9"/><circle class="s1" cx="600" cy="230" r="9"/><circle class="s1" cx="650" cy="230" r="9"/><circle class="s1" cx="700" cy="230" r="9"/><text class="t-tick" x="20.0" y="173.0" text-anchor="start">after layer 1</text><circle class="node" cx="150" cy="168" r="6"/><circle class="node" cx="200" cy="168" r="6"/><circle class="node" cx="250" cy="168" r="6"/><circle class="node" cx="300" cy="168" r="6"/><circle class="node" cx="350" cy="168" r="6"/><circle class="node" cx="400" cy="168" r="6"/><circle class="node" cx="450" cy="168" r="6"/><circle class="s1" cx="500" cy="168" r="9"/><line class="edge" x1="500" y1="177" x2="400" y2="221"/><line class="edge" x1="500" y1="177" x2="450" y2="221"/><line class="edge" x1="500" y1="177" x2="500" y2="221"/><circle class="s1" cx="550" cy="168" r="9"/><line class="edge" x1="550" y1="177" x2="450" y2="221"/><line class="edge" x1="550" y1="177" x2="500" y2="221"/><line class="edge" x1="550" y1="177" x2="550" y2="221"/><circle class="s1" cx="600" cy="168" r="9"/><line class="edge" x1="600" y1="177" x2="500" y2="221"/><line class="edge" x1="600" y1="177" x2="550" y2="221"/><line class="edge" x1="600" y1="177" x2="600" y2="221"/><circle class="s1" cx="650" cy="168" r="9"/><line class="edge" x1="650" y1="177" x2="550" y2="221"/><line class="edge" x1="650" y1="177" x2="600" y2="221"/><line class="edge" x1="650" y1="177" x2="650" y2="221"/><circle class="s1" cx="700" cy="168" r="9"/><line class="edge" x1="700" y1="177" x2="600" y2="221"/><line class="edge" x1="700" y1="177" x2="650" y2="221"/><line class="edge" x1="700" y1="177" x2="700" y2="221"/><text class="t-tick" x="20.0" y="111.0" text-anchor="start">after layer 2</text><circle class="node" cx="150" cy="106" r="6"/><circle class="node" cx="200" cy="106" r="6"/><circle class="node" cx="250" cy="106" r="6"/><circle class="node" cx="300" cy="106" r="6"/><circle class="node" cx="350" cy="106" r="6"/><circle class="node" cx="400" cy="106" r="6"/><circle class="node" cx="450" cy="106" r="6"/><circle class="node" cx="500" cy="106" r="6"/><circle class="node" cx="550" cy="106" r="6"/><circle class="s1" cx="600" cy="106" r="9"/><line class="edge" x1="600" y1="115" x2="500" y2="159"/><line class="edge" x1="600" y1="115" x2="550" y2="159"/><line class="edge" x1="600" y1="115" x2="600" y2="159"/><circle class="s1" cx="650" cy="106" r="9"/><line class="edge" x1="650" y1="115" x2="550" y2="159"/><line class="edge" x1="650" y1="115" x2="600" y2="159"/><line class="edge" x1="650" y1="115" x2="650" y2="159"/><circle class="s1" cx="700" cy="106" r="9"/><line class="edge" x1="700" y1="115" x2="600" y2="159"/><line class="edge" x1="700" y1="115" x2="650" y2="159"/><line class="edge" x1="700" y1="115" x2="700" y2="159"/><text class="t-tick" x="20.0" y="49.0" text-anchor="start">after layer 3</text><circle class="node" cx="150" cy="44" r="6"/><circle class="node" cx="200" cy="44" r="6"/><circle class="node" cx="250" cy="44" r="6"/><circle class="node" cx="300" cy="44" r="6"/><circle class="node" cx="350" cy="44" r="6"/><circle class="node" cx="400" cy="44" r="6"/><circle class="node" cx="450" cy="44" r="6"/><circle class="node" cx="500" cy="44" r="6"/><circle class="node" cx="550" cy="44" r="6"/><circle class="node" cx="600" cy="44" r="6"/><circle class="node" cx="650" cy="44" r="6"/><circle class="s1" cx="700" cy="44" r="9"/><line class="edge" x1="700" y1="53" x2="600" y2="97"/><line class="edge" x1="700" y1="53" x2="650" y2="97"/><line class="edge" x1="700" y1="53" x2="700" y2="97"/><text class="t-tick" x="150.0" y="268.0" text-anchor="start">Window W = 3. Each layer reaches W − 1 = 2 more tokens back: 1 → 3 → 5 → 7 tokens.</text><text class="t-tick" x="150.0" y="286.0" text-anchor="start">So after L layers, a token can be influenced by L × (W − 1) + 1 tokens.</text></svg><figcaption>How the reach grows. Window W = 3. The last token after layer 3 reads 3 tokens of layer 2, each of which read 3 tokens of layer 1, and so on. By the input, 7 tokens can affect it.</figcaption></figure>

Each layer lets information travel $$W - 1$$ more tokens back. After $$L$$ layers:

$$
\text{reach} = L \times (W - 1) + 1 \ \text{tokens}
$$

The Mistral 7B paper says the same thing (counting the window with its $$W+1$$ convention):

> [!PAPER] Jiang et al. (2023), Mistral 7B · Section 2 · page 2, Figure 1
> [![Figure 1 of the Mistral 7B paper: vanilla attention, sliding window attention and the effective context length growing across layers](/img/attention/papers/mistral-figure1.png)](/img/attention/papers/mistral-figure1.png)
>
> **Context:** The first figure of the Mistral 7B paper. Left: ordinary causal attention on "The cat sat on the" (1 = may look). Middle: a sliding window of $$W = 3$$. Right: how the reach grows through the layers.
>
> **What it says:** "each token can attend to at most W tokens from the previous layer", yet "tokens outside the sliding window still influence next word prediction", because "after k attention layers, information can move forward by up to k × W tokens".
>
> **Why it matters:** The right-hand panel is the same idea as the receptive-field figure above, drawn by the people who shipped it in a popular open model.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2310.06825)

> [!PAPER] Mistral 7B · Section 2, Architectural details · page 2
> [![The Mistral 7B paragraph on sliding window attention, with the theoretical attention span of about 131K tokens highlighted, next to the model architecture table](/img/attention/papers/mistral-span.png)](/img/attention/papers/mistral-span.png)
>
> **Context:** The paragraph that defines SWA in Mistral 7B, next to Table 1 with the model's settings (window_size 4096, 32 layers, 8 key/value heads).
>
> **What it says:** "At the last layer, using a window size of W = 4096, we have a theoretical attention span of approximately 131K tokens." In practice, at 16K tokens, their kernels gave "a 2x speed improvement over a vanilla attention baseline".
>
> **Why it matters:** 32 layers × 4,096 = 131,072: the reach formula with real numbers. And the table shows Mistral also used GQA (n_kv_heads 8 for 32 heads), combining Part 2 and Part 3 in one model.

Longformer (2020) wrote the same rule in its own notation, with $$\ell$$ layers and window $$w$$:

> [!PAPER] Beltagy, Peters, Cohan (2020), Longformer · Section 3.1 · page 4
> [![The Longformer sentence stating that in a transformer with l layers the receptive field size at the top layer is l times w](/img/attention/papers/longformer-receptive.png)](/img/attention/papers/longformer-receptive.png)
>
> **Context:** Longformer's description of its sliding window.
>
> **What it says:** "In a transformer with $$\ell$$ layers, the receptive field size at the top layer is $$\ell \times w$$ (assuming $$w$$ is fixed for all layers)."
>
> **Why it matters:** Two papers, three years apart, the same formula; and my gradient measurement above confirms it exactly.

**Proof.** I stacked 1 to 6 sliding-window layers ($$W = 4$$, with the usual "add the input back" connection that real models use), and asked PyTorch which input tokens the last output depends on. It can tell exactly, by computing the **gradient**.

> [!DEFINITION] Gradient (as a dependency detector)
> The gradient of an output with respect to an input says how much the output would change if that input changed slightly. If it is exactly zero for an input, the output does not depend on that input at all.

```text
2. Receptive field of the last token, window W = 4
   1 layer(s): depends on  4 tokens (formula L*(W-1)+1 =  4), reaches  3 tokens back
   2 layer(s): depends on  7 tokens (formula L*(W-1)+1 =  7), reaches  6 tokens back
   3 layer(s): depends on 10 tokens (formula L*(W-1)+1 = 10), reaches  9 tokens back
   4 layer(s): depends on 13 tokens (formula L*(W-1)+1 = 13), reaches 12 tokens back
   5 layer(s): depends on 16 tokens (formula L*(W-1)+1 = 16), reaches 15 tokens back
   6 layer(s): depends on 19 tokens (formula L*(W-1)+1 = 19), reaches 18 tokens back
```

The measured reach matches the formula exactly, at every depth.

> [!WARNING] "Can reach" is not "remembers well"
> The formula gives the *theoretical* reach. Information that travels through many layers gets blended with everything else on the way, like a message whispered down a long line. A sliding-window-only model can technically be influenced by a token 100,000 positions back, but it cannot *look it up* directly. That is why most recent models add some global layers (section 5).

## 4. The rolling buffer cache: memory that never grows

With a window, a token never needs keys and values older than $$W$$ steps. So the KV cache can be a fixed-size **ring**.

> [!DEFINITION] Rolling (ring) buffer
> A fixed number of slots used in a circle. When the last slot is filled, writing goes back to slot 0 and overwrites the oldest item. Token $$i$$ goes into slot $$i \bmod W$$ ("the remainder when $$i$$ is divided by $$W$$").

> [!PAPER] Mistral 7B · Section 2 · page 3, Figure 2
> [![Figure 2 of the Mistral 7B paper: a rolling buffer cache of 4 slots for three sentences over three time steps](/img/attention/papers/mistral-rolling.png)](/img/attention/papers/mistral-rolling.png)
>
> **Context:** Three sentences being written at once, over three time steps, each with a 4-slot cache. Orange marks the slot written at that step.
>
> **What it says:** "Keys and values for position $$i$$ are stored in position $$i \bmod W$$ of the cache. When the position $$i$$ is larger than $$W$$, past values in the cache are overwritten." Look at the first row: at step $$i+2$$, "of" overwrites "This" in slot 0.
>
> **Why it matters:** This is exactly the loop in the code below, and the out-of-order slots ("of is an example") are why the next paragraph has to explain that order does not matter.

<figure class="fig"><svg viewBox="0 0 760 254" role="img" aria-label="A rolling buffer cache: token i is written into slot i mod W, so the cache always holds only the last W tokens."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Writing token 9 with a cache of W = 4 slots</text><rect class="box" x="30.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="58.0" y="65.0" text-anchor="middle">tok 0</text><line class="edge" x1="38" y1="60" x2="78" y2="60"/><rect class="box" x="100.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="128.0" y="65.0" text-anchor="middle">tok 1</text><line class="edge" x1="108" y1="60" x2="148" y2="60"/><rect class="box" x="170.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="198.0" y="65.0" text-anchor="middle">tok 2</text><line class="edge" x1="178" y1="60" x2="218" y2="60"/><rect class="box" x="240.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="268.0" y="65.0" text-anchor="middle">tok 3</text><line class="edge" x1="248" y1="60" x2="288" y2="60"/><rect class="box" x="310.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="338.0" y="65.0" text-anchor="middle">tok 4</text><line class="edge" x1="318" y1="60" x2="358" y2="60"/><rect class="box" x="380.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="408.0" y="65.0" text-anchor="middle">tok 5</text><line class="edge" x1="388" y1="60" x2="428" y2="60"/><rect class="box-3" x="450.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="478.0" y="65.0" text-anchor="middle">tok 6</text><rect class="box-3" x="520.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="548.0" y="65.0" text-anchor="middle">tok 7</text><rect class="box-3" x="590.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="618.0" y="65.0" text-anchor="middle">tok 8</text><rect class="box-1" x="660.0" y="44.0" width="56.0" height="32.0" rx="6"/><text class="t-tick" x="688.0" y="65.0" text-anchor="middle">tok 9</text><text class="t-tick" x="30.0" y="100.0" text-anchor="start">tokens 0 to 5: overwritten (crossed out); tokens 6 to 9: still in the cache</text><rect class="box-3" x="150.0" y="140.0" width="100.0" height="52.0" rx="8"/><text class="t-note" x="200.0" y="162.0" text-anchor="middle">slot 0</text><text class="t-tick" x="200.0" y="182.0" text-anchor="middle">holds tok 8</text><line class="edge" x1="618" y1="78" x2="200" y2="138" marker-end="url(#ah)"/><rect class="box-1" x="270.0" y="140.0" width="100.0" height="52.0" rx="8"/><text class="t-note" x="320.0" y="162.0" text-anchor="middle">slot 1</text><text class="t-tick" x="320.0" y="182.0" text-anchor="middle">holds tok 9</text><line class="edge" x1="688" y1="78" x2="320" y2="138" marker-end="url(#ah)"/><rect class="box-3" x="390.0" y="140.0" width="100.0" height="52.0" rx="8"/><text class="t-note" x="440.0" y="162.0" text-anchor="middle">slot 2</text><text class="t-tick" x="440.0" y="182.0" text-anchor="middle">holds tok 6</text><line class="edge" x1="478" y1="78" x2="440" y2="138" marker-end="url(#ah)"/><rect class="box-3" x="510.0" y="140.0" width="100.0" height="52.0" rx="8"/><text class="t-note" x="560.0" y="162.0" text-anchor="middle">slot 3</text><text class="t-tick" x="560.0" y="182.0" text-anchor="middle">holds tok 7</text><line class="edge" x1="548" y1="78" x2="560" y2="138" marker-end="url(#ah)"/><text class="t-tick" x="150.0" y="222.0" text-anchor="start">Rule: token i goes into slot i mod 4. Token 9 → slot 1, overwriting token 5.</text><text class="t-tick" x="150.0" y="240.0" text-anchor="start">The cache never grows past W slots, no matter how long the text gets.</text></svg><figcaption>A rolling buffer with W = 4 slots while writing token 9. Token i goes into slot i mod 4. Tokens 0 to 5 have been overwritten; the cache always holds just the last 4 tokens.</figcaption></figure>

Here is the decoding loop from `part3_sparse.py`. It writes one token at a time into a cache of only $$W$$ slots:

```python
k_cache = torch.zeros(H, W, d, dtype=torch.float64)                     # the whole cache: W slots, never more
v_cache = torch.zeros(H, W, d, dtype=torch.float64)
step_out = []
for i in range(T):
    xi = x[i:i + 1]
    qi = rope((xi @ Wq).view(1, H, d).transpose(0, 1), pos[i:i + 1])
    ki = rope((xi @ Wk).view(1, H, d).transpose(0, 1), pos[i:i + 1])
    vi = (xi @ Wv).view(1, H, d).transpose(0, 1)
    slot = i % W                                                        # position i goes to slot i mod W
    k_cache[:, slot], v_cache[:, slot] = ki[:, 0], vi[:, 0]
    n = min(i + 1, W)                                                   # how many slots are filled
    step_out.append(attend(qi, k_cache[:, :n], v_cache[:, :n], torch.ones(1, n, dtype=torch.bool)))
```

Notice that the slots are **out of order**: in the picture, slot 0 holds token 8 and slot 2 holds token 6. Does that break anything? No, for two reasons:

1. **Softmax and the weighted sum do not care about order.** Adding up "weight × value" over a set of tokens gives the same answer in any order.
2. **Position is already baked into each key.** RoPE (explained in [Part 2](attention-2-mqa-gqa-mla.md)) rotates each key by its position *before* it is stored, so the key remembers where it came from, whatever slot it sits in.

**Proof.** I compared this token-by-token ring cache with computing all 50 tokens at once using the sliding-window mask (2 heads, RoPE on):

```text
3. Rolling buffer cache (W = 8 slots) vs computing all 50 tokens at once: max |diff| = 1.1e-15
```

Identical, while the cache held 8 slots instead of 50.

### How much memory does it save?

The KV cache formula from Part 2, with the number of stored tokens capped at $$W$$:

$$
\text{KV cache} = 2 \times L \times H_{kv} \times d_h \times \min(n, W) \times b
$$

where $$n$$ is the number of tokens so far, and the other symbols are as in Part 2 (layers, key/value heads, head size, bytes per number).

For Mistral 7B (32 layers, 8 key/value heads, head size 128, window 4,096) at 32,768 tokens:

```text
   Mistral 7B at 32K: full 4.00 GiB, window 4096 0.50 GiB (8x smaller)
```

> [!PAPER] Mistral 7B · Section 2, Rolling Buffer Cache · page 2
> [![The Mistral 7B rolling buffer cache paragraph, highlighting that on a 32k-token sequence it reduces cache memory usage by 8x without impacting model quality](/img/attention/papers/mistral-8x.png)](/img/attention/papers/mistral-8x.png)
>
> **Context:** The paragraph that introduces the rolling buffer cache.
>
> **What it says:** "On a sequence length of 32k tokens, this reduces the cache memory usage by 8x, without impacting the model quality."
>
> **Why it matters:** 32,768 ÷ 4,096 = 8. My calculation from Mistral's published configuration gives the same 8×.

### And speed?

Every writing step reads the whole cache. With full attention that read keeps growing; with a window it stays the same size. I timed one attention step (16 heads, head size 128, 16-bit numbers) on an Apple M5 Pro GPU:

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Time for one decode step of attention as the conversation grows: full attention versus a 1,024-token sliding window."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="570" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="195.0" x2="570" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">1.00</text><line class="grid" x1="64" y1="140.0" x2="570" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">2.00</text><line class="grid" x1="64" y1="85.0" x2="570" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">3.00</text><line class="grid" x1="64" y1="30.0" x2="570" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">4.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1,024</text><text class="t-tick" x="208.6" y="270.0" text-anchor="middle">4,096</text><text class="t-tick" x="353.1" y="270.0" text-anchor="middle">16,384</text><text class="t-tick" x="497.7" y="270.0" text-anchor="middle">65,536</text><text class="t-tick" x="570.0" y="270.0" text-anchor="middle">131,072</text><line class="axis" x1="64" y1="250" x2="570" y2="250"/><polyline class="l2" points="64.0,246.9 208.6,242.5 353.1,221.3 497.7,138.9 570.0,35.9"/><g class="mark"><title>full attention, 1,024: 0.06</title><circle class="s2 ring" cx="64.0" cy="246.9" r="5"/></g><g class="mark"><title>full attention, 4,096: 0.14</title><circle class="s2 ring" cx="208.6" cy="242.5" r="5"/></g><g class="mark"><title>full attention, 16,384: 0.52</title><circle class="s2 ring" cx="353.1" cy="221.3" r="5"/></g><g class="mark"><title>full attention, 65,536: 2.02</title><circle class="s2 ring" cx="497.7" cy="138.9" r="5"/></g><g class="mark"><title>full attention, 131,072: 3.89</title><circle class="s2 ring" cx="570.0" cy="35.9" r="5"/></g><polyline class="l1" points="64.0,247.7 208.6,248.2 353.1,248.3 497.7,248.2 570.0,248.3"/><g class="mark"><title>window 1,024, 1,024: 0.04</title><circle class="s1 ring" cx="64.0" cy="247.7" r="5"/></g><g class="mark"><title>window 1,024, 4,096: 0.03</title><circle class="s1 ring" cx="208.6" cy="248.2" r="5"/></g><g class="mark"><title>window 1,024, 16,384: 0.03</title><circle class="s1 ring" cx="353.1" cy="248.3" r="5"/></g><g class="mark"><title>window 1,024, 65,536: 0.03</title><circle class="s1 ring" cx="497.7" cy="248.2" r="5"/></g><g class="mark"><title>window 1,024, 131,072: 0.03</title><circle class="s1 ring" cx="570.0" cy="248.3" r="5"/></g><line class="grid" x1="576.0" y1="35.9" x2="586.0" y2="35.9"/><rect class="s2" x="590.0" y="30.9" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="39.9" text-anchor="start">full attention: 3.89</text><line class="grid" x1="576.0" y1="248.3" x2="586.0" y2="248.3"/><rect class="s1" x="590.0" y="243.3" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="252.3" text-anchor="start">window 1,024: 0.03</text><text class="t-tick" x="317.0" y="290.0" text-anchor="middle">tokens so far (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">milliseconds per step</text></svg><figcaption>Time for one decode step of attention. Full attention gets slower as the conversation grows, because it reads the whole cache. With a 1,024-token window, the step time stays flat.</figcaption></figure>

| Tokens so far | Full attention | Window of 1,024 |
|---|---|---|
| 1,024 | 0.057 ms | 0.041 ms |
| 4,096 | 0.137 ms | 0.033 ms |
| 16,384 | 0.521 ms | 0.030 ms |
| 65,536 | 2.019 ms | 0.034 ms |
| 131,072 | 3.893 ms | 0.031 ms |

At 128K tokens, the windowed step is about **125 times faster**. (At 1,024 tokens both read the same amount; the small gap there is timing noise.)

## 5. Mixing local and global layers

A pure sliding-window model has the weakness from the warning above: it can never look up an old token directly. The fix that most recent models use is to **mix** two kinds of layers:

> [!DEFINITION] Local layer and global layer
> A **local** layer uses sliding-window attention: cheap, short memory. A **global** layer uses ordinary full attention: expensive, but it can look directly at any earlier token. A model can stack them in a fixed pattern, for example 5 local then 1 global.

This idea is older than chatbots. Longformer (2020) already combined the two inside one layer:

> [!PAPER] Longformer · Section 3.1, Attention Pattern · page 3, Figure 2
> [![Figure 2 of the Longformer paper: full n squared attention, sliding window attention, dilated sliding window and global plus sliding window patterns](/img/attention/papers/longformer-figure2.png)](/img/attention/papers/longformer-figure2.png)
>
> **Context:** Four attention patterns drawn as grids, like this article's own mask figure (a dark cell = may look).
>
> **What it says:** (a) full attention; (b) a sliding window; (c) a **dilated** window that skips every other token to reach further at the same cost; (d) a window plus a few **global** tokens (the full rows and columns) that see and are seen by everything.
>
> **Why it matters:** Pattern (d) is the ancestor of today's local and global layers: Longformer mixed them inside one layer, Gemma and gpt-oss mix them across layers. And the global tokens play the same role as the attention sinks of section 6.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2004.05150)

> [!DEFINITION] Dilated window
> A window with gaps: instead of the last $$W$$ tokens, look at every 2nd (or every 4th) of the last $$2W$$ (or $$4W$$) tokens. Same cost, longer reach, but some nearby tokens are skipped.

Today's models do it **layer by layer**:

- **Gemma 2** (2024) alternates **1 local : 1 global**, with a local window of 4,096 tokens.
- **Gemma 3** (2025) goes further: **5 local : 1 global**, and shrinks the window to **1,024**.
- **gpt-oss** (OpenAI, 2025) alternates **1 : 1** with a tiny window of just **128** tokens.

> [!PAPER] Gemma Team (2024), Gemma 2 · Section 2 · page 2
> [![The Gemma 2 paragraph stating that local sliding window and global attention alternate in every other layer, with a 4096-token window and an 8192-token global span](/img/attention/papers/gemma2-alternate.png)](/img/attention/papers/gemma2-alternate.png)
>
> **Context:** Gemma 2's list of architecture changes.
>
> **What it says:** Local and global attention alternate "in every other layer. The sliding window size of local attention layers is set to 4096 tokens, while the span of the global attention layers is set to 8192 tokens."
>
> **Why it matters:** With a maximum length of 8,192, the window covers half of it, so Gemma 2's saving is modest. Gemma 3 pushed the same idea much further.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2408.00118)

> [!PAPER] Gemma Team (2025), Gemma 3 Technical Report · Section 1 · page 1
> [![The Gemma 3 introduction, highlighting that a challenge with long context is the memory explosion of the KV cache during inference](/img/attention/papers/gemma3-memory.png)](/img/attention/papers/gemma3-memory.png)
>
> **Context:** The introduction, explaining the second main change from Gemma 2: a context of 128K tokens.
>
> **What it says:** "A challenge with long context is the memory explosion of the KV cache during inference. To reduce this issue, we interleave multiple local layers between each global layer", with a pattern of 5 local layers for every global layer and "a smaller span of only 1024 tokens" for the local ones.
>
> **Why it matters:** The reason is not speed but **memory**: at 128K tokens, the cache of an all-global model would be larger than the model itself (see the memory math below).
>
> [Read the paper on arXiv](https://arxiv.org/abs/2503.19786)

Do the local layers hurt quality? Gemma 3 tested it:

> [!PAPER] Gemma 3 · Section 5, Ablations · page 6, Figure 3
> [![Figure 3 of the Gemma 3 report: change in perplexity as the local to global ratio goes from 1:1 to 7:1 for 2B and 9B models, staying almost flat](/img/attention/papers/gemma3-figure3.png)](/img/attention/papers/gemma3-figure3.png)
>
> **Context:** The change in perplexity (lower is better) relative to Gemma 2's 1:1 pattern, as more local layers are used per global layer, for 2B and 9B models.
>
> **What it says:** "The impact is minimal, even with 7-to-1 local to global." All the lines stay within about ±0.03 perplexity.
>
> **Why it matters:** This is the evidence behind 5:1. Most layers simply do not need to see far back, as long as a few layers still can.

<figure class="fig"><svg viewBox="0 0 760 162" role="img" aria-label="The layer pattern of Gemma 3 27B: five sliding-window layers, then one global layer, repeated through 62 layers."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="24.0" text-anchor="start">Gemma 3 27B: 62 layers, 5 local (sliding window 1,024) then 1 global, repeated</text><g class="mark"><title>layer 1: local (last 1,024 tokens)</title><rect class="s1" x="20.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 2: local (last 1,024 tokens)</title><rect class="s1" x="31.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 3: local (last 1,024 tokens)</title><rect class="s1" x="43.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 4: local (last 1,024 tokens)</title><rect class="s1" x="54.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 5: local (last 1,024 tokens)</title><rect class="s1" x="66.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 6: global (sees every token)</title><rect class="s2" x="78.0" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 7: local (last 1,024 tokens)</title><rect class="s1" x="89.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 8: local (last 1,024 tokens)</title><rect class="s1" x="101.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 9: local (last 1,024 tokens)</title><rect class="s1" x="112.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 10: local (last 1,024 tokens)</title><rect class="s1" x="124.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 11: local (last 1,024 tokens)</title><rect class="s1" x="136.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 12: global (sees every token)</title><rect class="s2" x="147.6" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 13: local (last 1,024 tokens)</title><rect class="s1" x="159.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 14: local (last 1,024 tokens)</title><rect class="s1" x="170.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 15: local (last 1,024 tokens)</title><rect class="s1" x="182.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 16: local (last 1,024 tokens)</title><rect class="s1" x="194.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 17: local (last 1,024 tokens)</title><rect class="s1" x="205.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 18: global (sees every token)</title><rect class="s2" x="217.2" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 19: local (last 1,024 tokens)</title><rect class="s1" x="228.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 20: local (last 1,024 tokens)</title><rect class="s1" x="240.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 21: local (last 1,024 tokens)</title><rect class="s1" x="252.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 22: local (last 1,024 tokens)</title><rect class="s1" x="263.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 23: local (last 1,024 tokens)</title><rect class="s1" x="275.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 24: global (sees every token)</title><rect class="s2" x="286.8" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 25: local (last 1,024 tokens)</title><rect class="s1" x="298.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 26: local (last 1,024 tokens)</title><rect class="s1" x="310.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 27: local (last 1,024 tokens)</title><rect class="s1" x="321.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 28: local (last 1,024 tokens)</title><rect class="s1" x="333.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 29: local (last 1,024 tokens)</title><rect class="s1" x="344.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 30: global (sees every token)</title><rect class="s2" x="356.4" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 31: local (last 1,024 tokens)</title><rect class="s1" x="368.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 32: local (last 1,024 tokens)</title><rect class="s1" x="379.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 33: local (last 1,024 tokens)</title><rect class="s1" x="391.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 34: local (last 1,024 tokens)</title><rect class="s1" x="402.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 35: local (last 1,024 tokens)</title><rect class="s1" x="414.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 36: global (sees every token)</title><rect class="s2" x="426.0" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 37: local (last 1,024 tokens)</title><rect class="s1" x="437.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 38: local (last 1,024 tokens)</title><rect class="s1" x="449.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 39: local (last 1,024 tokens)</title><rect class="s1" x="460.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 40: local (last 1,024 tokens)</title><rect class="s1" x="472.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 41: local (last 1,024 tokens)</title><rect class="s1" x="484.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 42: global (sees every token)</title><rect class="s2" x="495.6" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 43: local (last 1,024 tokens)</title><rect class="s1" x="507.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 44: local (last 1,024 tokens)</title><rect class="s1" x="518.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 45: local (last 1,024 tokens)</title><rect class="s1" x="530.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 46: local (last 1,024 tokens)</title><rect class="s1" x="542.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 47: local (last 1,024 tokens)</title><rect class="s1" x="553.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 48: global (sees every token)</title><rect class="s2" x="565.2" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 49: local (last 1,024 tokens)</title><rect class="s1" x="576.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 50: local (last 1,024 tokens)</title><rect class="s1" x="588.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 51: local (last 1,024 tokens)</title><rect class="s1" x="600.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 52: local (last 1,024 tokens)</title><rect class="s1" x="611.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 53: local (last 1,024 tokens)</title><rect class="s1" x="623.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 54: global (sees every token)</title><rect class="s2" x="634.8" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 55: local (last 1,024 tokens)</title><rect class="s1" x="646.4" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 56: local (last 1,024 tokens)</title><rect class="s1" x="658.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 57: local (last 1,024 tokens)</title><rect class="s1" x="669.6" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 58: local (last 1,024 tokens)</title><rect class="s1" x="681.2" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 59: local (last 1,024 tokens)</title><rect class="s1" x="692.8" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 60: global (sees every token)</title><rect class="s2" x="704.4" y="40" width="9" height="48" rx="2"/></g><g class="mark"><title>layer 61: local (last 1,024 tokens)</title><rect class="s1" x="716.0" y="52" width="9" height="24" rx="2"/></g><g class="mark"><title>layer 62: local (last 1,024 tokens)</title><rect class="s1" x="727.6" y="52" width="9" height="24" rx="2"/></g><rect class="s1" x="20" y="112" width="12" height="12" rx="2"/><text class="t-tick" x="38.0" y="122.0" text-anchor="start">52 local layers: keep only the last 1,024 tokens</text><rect class="s2" x="380" y="112" width="12" height="12" rx="2"/><text class="t-tick" x="398.0" y="122.0" text-anchor="start">10 global layers: keep every token</text><text class="t-tick" x="20.0" y="150.0" text-anchor="start">layer 1</text><text class="t-tick" x="740.0" y="150.0" text-anchor="end">layer 62</text></svg><figcaption>The real layer pattern of Gemma 3 27B: five local layers (blue, window 1,024), then one global layer (orange), repeated. 52 local and 10 global layers in total. Hover a bar to see the layer.</figcaption></figure>

Here is the published configuration (from Hugging Face), which is where those numbers come from:

```json
{
  "num_hidden_layers": 62,
  "num_attention_heads": 32,
  "num_key_value_heads": 16,
  "head_dim": 128,
  "sliding_window": 1024,
  "sliding_window_pattern": 6,
  "rope_theta": 1000000.0,
  "rope_local_base_freq": 10000.0
}
```

`sliding_window_pattern: 6` means every 6th layer is global: layers 6, 12, 18, ..., 60. That gives 10 global and 52 local layers.

### The memory math

Global layers store every token; local layers store at most $$W$$:

$$
\text{KV cache} = 2 \, H_{kv} \, d_h \, b \, \Big( L_{\text{global}} \cdot n \;+\; L_{\text{local}} \cdot \min(n, W) \Big)
$$

where $$L_{\text{global}}$$ and $$L_{\text{local}}$$ are the numbers of global and local layers. For Gemma 3 27B ($$H_{kv} = 16$$, $$d_h = 128$$, 2 bytes per number):

```text
   Gemma 3 27B (52 local + 10 global layers):
         1024 tokens: all global   0.48 GiB   5 local : 1 global  0.48 GiB   (1.0x smaller)
         2048 tokens: all global   0.97 GiB   5 local : 1 global  0.56 GiB   (1.7x smaller)
         4096 tokens: all global   1.94 GiB   5 local : 1 global  0.72 GiB   (2.7x smaller)
         8192 tokens: all global   3.88 GiB   5 local : 1 global  1.03 GiB   (3.8x smaller)
        16384 tokens: all global   7.75 GiB   5 local : 1 global  1.66 GiB   (4.7x smaller)
        32768 tokens: all global  15.50 GiB   5 local : 1 global  2.91 GiB   (5.3x smaller)
        65536 tokens: all global  31.00 GiB   5 local : 1 global  5.41 GiB   (5.7x smaller)
       131072 tokens: all global  62.00 GiB   5 local : 1 global 10.41 GiB   (6.0x smaller)
```

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="KV cache memory of Gemma 3 27B for one sequence, with every layer global versus the real 5 local to 1 global pattern."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="570" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.0</text><line class="grid" x1="64" y1="195.0" x2="570" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">16</text><line class="grid" x1="64" y1="140.0" x2="570" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">32</text><line class="grid" x1="64" y1="85.0" x2="570" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">48</text><line class="grid" x1="64" y1="30.0" x2="570" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">64</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1,024</text><text class="t-tick" x="208.6" y="270.0" text-anchor="middle">4,096</text><text class="t-tick" x="353.1" y="270.0" text-anchor="middle">16,384</text><text class="t-tick" x="497.7" y="270.0" text-anchor="middle">65,536</text><text class="t-tick" x="570.0" y="270.0" text-anchor="middle">131,072</text><line class="axis" x1="64" y1="250" x2="570" y2="250"/><polyline class="l2" points="64.0,248.3 136.3,246.7 208.6,243.3 280.9,236.7 353.1,223.4 425.4,196.7 497.7,143.4 570.0,36.9"/><g class="mark"><title>all layers global, 1,024: 0.5</title><circle class="s2 ring" cx="64.0" cy="248.3" r="5"/></g><g class="mark"><title>all layers global, 2,048: 1.0</title><circle class="s2 ring" cx="136.3" cy="246.7" r="5"/></g><g class="mark"><title>all layers global, 4,096: 1.9</title><circle class="s2 ring" cx="208.6" cy="243.3" r="5"/></g><g class="mark"><title>all layers global, 8,192: 3.9</title><circle class="s2 ring" cx="280.9" cy="236.7" r="5"/></g><g class="mark"><title>all layers global, 16,384: 7.8</title><circle class="s2 ring" cx="353.1" cy="223.4" r="5"/></g><g class="mark"><title>all layers global, 32,768: 16</title><circle class="s2 ring" cx="425.4" cy="196.7" r="5"/></g><g class="mark"><title>all layers global, 65,536: 31</title><circle class="s2 ring" cx="497.7" cy="143.4" r="5"/></g><g class="mark"><title>all layers global, 131,072: 62</title><circle class="s2 ring" cx="570.0" cy="36.9" r="5"/></g><polyline class="l1" points="64.0,248.3 136.3,248.1 208.6,247.5 280.9,246.5 353.1,244.3 425.4,240.0 497.7,231.4 570.0,214.2"/><g class="mark"><title>5 local : 1 global, 1,024: 0.5</title><circle class="s1 ring" cx="64.0" cy="248.3" r="5"/></g><g class="mark"><title>5 local : 1 global, 2,048: 0.6</title><circle class="s1 ring" cx="136.3" cy="248.1" r="5"/></g><g class="mark"><title>5 local : 1 global, 4,096: 0.7</title><circle class="s1 ring" cx="208.6" cy="247.5" r="5"/></g><g class="mark"><title>5 local : 1 global, 8,192: 1.0</title><circle class="s1 ring" cx="280.9" cy="246.5" r="5"/></g><g class="mark"><title>5 local : 1 global, 16,384: 1.7</title><circle class="s1 ring" cx="353.1" cy="244.3" r="5"/></g><g class="mark"><title>5 local : 1 global, 32,768: 2.9</title><circle class="s1 ring" cx="425.4" cy="240.0" r="5"/></g><g class="mark"><title>5 local : 1 global, 65,536: 5.4</title><circle class="s1 ring" cx="497.7" cy="231.4" r="5"/></g><g class="mark"><title>5 local : 1 global, 131,072: 10</title><circle class="s1 ring" cx="570.0" cy="214.2" r="5"/></g><line class="grid" x1="576.0" y1="36.9" x2="586.0" y2="36.9"/><rect class="s2" x="590.0" y="31.9" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="40.9" text-anchor="start">all layers global: 62</text><line class="grid" x1="576.0" y1="214.2" x2="586.0" y2="214.2"/><rect class="s1" x="590.0" y="209.2" width="10" height="10" rx="2"/><text class="t-note" x="606.0" y="218.2" text-anchor="start">5 local : 1 global: 10</text><text class="t-tick" x="317.0" y="290.0" text-anchor="middle">tokens in the conversation (log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">KV cache, GiB</text></svg><figcaption>KV cache of Gemma 3 27B for one conversation. If every layer were global, 128K tokens would need 62 GiB. With the real 5:1 pattern it needs 10.4 GiB.</figcaption></figure>

Gemma 3's own measurement, for a 2B model, shows the same shape:

> [!PAPER] Gemma 3 · Section 5 · page 7, Figure 6
> [![Figure 6 of the Gemma 3 report: KV cache memory versus context length for a 2B model, global only against the 5:1 pattern with a 1024 window](/img/attention/papers/gemma3-figure6.png)](/img/attention/papers/gemma3-figure6.png)
>
> **Context:** KV cache memory (in MB) against context length, for a 2B model with only global layers (yellow) and with Gemma 3's design (red).
>
> **What it says:** Global-only grows to about 7,000 MB at 128K tokens; the 5:1 design with 1,024-token windows stays near 1,100 MB.
>
> **Why it matters:** Same curve as my Gemma 3 27B calculation above, from the authors. The red line still grows (the global layers), just much more slowly.

At the full 128K context: **62 GiB → 10.4 GiB**. The savings can never pass 6.2× (62 layers ÷ 10 global layers), because the global layers still grow with the text. They become the main cost.

For gpt-oss-20b (24 layers alternating, 8 key/value heads, head size 64, window 128), at 128K tokens the cache drops from 6.00 GiB to 3.00 GiB: half the layers keep almost nothing.

> [!PAPER] Gemma 3 · Section 2, Long context · page 2
> [![The Gemma 3 paragraph on long context, highlighting the increase of the RoPE base frequency from 10k to 1M on global layers while local layers keep 10k](/img/attention/papers/gemma3-rope.png)](/img/attention/papers/gemma3-rope.png)
>
> **Context:** How Gemma 3 handles positions at 128K tokens.
>
> **What it says:** "We increase RoPE base frequency from 10k to 1M on global self-attention layers, and keep the frequency of the local layers at 10k."
>
> **Why it matters:** RoPE (Part 2) turns each pair of numbers by an angle $$m\theta$$. A larger base makes $$\theta$$ smaller, so the angles turn more slowly and positions 100,000 tokens apart still look different. Local layers never see more than 1,024 tokens, so they keep the usual setting.

The RoPE angle for pair $$i$$ of a head of size $$d$$ is

$$
\theta_i = \text{base}^{-2i/d}, \qquad i = 0, 1, \dots, \tfrac{d}{2} - 1
$$

where "base" is 10,000 normally and 1,000,000 in Gemma 3's global layers. The slowest pair ($$i = d/2 - 1$$) has a period, in tokens, of about $$2\pi \cdot \text{base}$$: roughly 63,000 tokens for base 10k, roughly 6.3 million for base 1M. That is why the global layers need the larger base to tell positions apart across 128K tokens.

| Model | Pattern | Window | Why it matters |
|---|---|---|---|
| Mistral 7B | every layer local | 4,096 | made SWA and the rolling cache popular in open models |
| Gemma 2 | 1 local : 1 global | 4,096 | half the layers stay small |
| Gemma 3 | 5 local : 1 global | 1,024 | 6× smaller cache at 128K |
| gpt-oss | 1 local : 1 global | 128 | tiny windows, half the layers almost free |

## 6. Attention sinks strike again: a real experiment

All the models above were **trained** with their windows, so they learned to live with them. What happens if you take a normal model, trained with full attention, and simply force a window on it? This is what "StreamingLLM" studied, and the answer surprised people.

> [!PAPER] Xiao et al. (2023), StreamingLLM · Section 1 · page 2, Figure 1
> [![Figure 1 of the StreamingLLM paper comparing dense attention, window attention, sliding window with re-computation and StreamingLLM, with their complexity and perplexity](/img/attention/papers/streaming-figure1.png)](/img/attention/papers/streaming-figure1.png)
>
> **Context:** Four ways to run a model on a text much longer than it was trained on, with the cost and the perplexity (PPL, lower is better) of Llama-2-13B on a 65K-token book.
>
> **What it says:** (a) Dense attention: $$O(T^2)$$ and PPL 5,641 (broken, because the text is longer than training). (b) Window attention: cheap, but PPL 5,158: it "breaks when initial tokens are evicted". (c) Recomputing a window for each token works (PPL 5.43) but is very slow. (d) StreamingLLM, window plus the first few tokens (yellow, the attention sink): cheap **and** PPL 5.40.
>
> **Why it matters:** Panel (b) versus (d) is the whole discovery: the only difference is keeping a handful of first tokens, and perplexity goes from 5,158 to 5.40.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2309.17453)

> [!PAPER] StreamingLLM · Section 3.1 · page 4, Figure 3
> [![Figure 3 of the StreamingLLM paper: log perplexity over 20K tokens for Llama-2-7B, Pythia-12B, Falcon-7B and MPT-7B under dense attention, window attention, sliding window with recomputation and StreamingLLM](/img/attention/papers/streaming-figure3.png)](/img/attention/papers/streaming-figure3.png)
>
> **Context:** Log perplexity over 20,000 tokens for four different model families. The dashed lines mark the cache size and the training length.
>
> **What it says:** Dense attention (blue) fails once the text passes the training length. Window attention (orange) jumps up as soon as the first tokens leave the cache. StreamingLLM (red) stays flat, matching the slow recomputation baseline (green).
>
> **Why it matters:** It is not one model's quirk: Llama, Pythia, Falcon and MPT all behave the same. My Qwen2.5 experiment below adds a fifth family.

Remember the **attention sink** from [Part 1](attention-1-self-attention.md): in Qwen2.5-0.5B, 68% of heads put most of their attention on the very first token, because softmax forces the weights to add up to 1 and the heads need somewhere harmless to "park" them. A sliding window cuts that first token off.

I tested this myself.

> [!DEFINITION] Perplexity
> The standard score for how well a language model predicts text. Roughly: "on average, how many tokens was the model choosing between?" **Lower is better.** It is $$e^{\text{average loss}}$$, where the loss for each token is $$-\log(\text{probability the model gave the correct next token})$$.

**Setup** (`part3_qwen.py`):

- Model: **Qwen2.5-0.5B**, trained with full attention.
- Text: four chunks of 2,048 tokens from *Pride and Prejudice* (public domain, from Project Gutenberg).
- I score only tokens 1,024 to 2,047, where every window really cuts something off.
- Each rule gets the **same budget**: each token may look at exactly 64, 256 or 1,024 tokens.
- "4 sinks + window" means: the first 4 tokens of the text, plus the most recent (budget − 4) tokens.
- First, a sanity check: passing my own full-attention mask gives exactly the model's normal output (max difference 0.0).

```text
full attention: perplexity 17.39
budget   64 tokens: window only   132.70   4 sinks + window 60:  22.45
budget  256 tokens: window only    67.90   4 sinks + window 252:  18.83
budget 1024 tokens: window only   508.88   4 sinks + window 1020:  17.80
```

What this shows:

- **A plain window destroys the model.** Perplexity jumps from 17.4 to between 68 and 509.
- **Keeping just 4 sink tokens fixes almost all of it.** With 1,024 tokens of budget, 4 sinks plus a window give 17.80, very close to full attention's 17.39.
- **A bigger window does not save you if the sink is gone.** The 1,024-token window without sinks was the *worst* of all (508.9), even though it sees 16 times more text than the 64-token window. I did not expect this. It shows the problem is not missing information: the model is thrown off by losing its "parking spot".

> [!PAPER] StreamingLLM · Section 4.3, Ablation · page 9
> [![The StreamingLLM ablation paragraph stating that one or two initial tokens are not enough, while a threshold of four initial tokens appears enough](/img/attention/papers/streaming-four.png)](/img/attention/papers/streaming-four.png)
>
> **Context:** How many first tokens must be kept?
>
> **What it says:** "merely one or two initial tokens" are not sufficient, while "a threshold of four initial tokens appears enough, with subsequent additions contributing marginal effects."
>
> **Why it matters:** This is why my experiment kept exactly 4 sink tokens.

With $$S$$ sink tokens and a window of $$W$$ recent tokens, the cache holds a fixed

$$
\text{cache size} = (S + W) \times \text{bytes per token}
$$

whatever the length of the conversation: 4 + 1,020 = 1,024 tokens in my 1,024-budget test.

> [!NOTE] How this differs from StreamingLLM exactly
> StreamingLLM also renumbers the positions inside the cache when it evicts tokens. I kept the original positions and only changed which tokens are visible. The lesson is the same: a model trained with full attention depends on its first tokens, and **"just use a window" is not safe for a model that was not trained with one**. Models like Mistral and Gemma learn to cope with windows during training.

## 7. Sparse attention: let the model choose

Windows are **fixed** patterns: they always keep the most recent tokens. But sometimes the important token is far away. A character's name introduced in chapter 1, a function defined at the top of a file, the original question at the start of a long chat.

> [!DEFINITION] Sparse attention
> Attention where each token looks at only a small **subset** of earlier tokens instead of all of them. "Sparse" means mostly empty: in the attention grid, most squares are skipped. The subset can be fixed (a window, a stride) or chosen by the model.

Early sparse transformers ([Child et al., 2019](https://arxiv.org/abs/1904.10509)) used fixed patterns, like "the last few tokens plus every 64th token". The newer idea is to let the model **pick** the tokens that matter for each query. The problem: to know which tokens score highest, you seem to need the scores, which is the very thing you were trying to avoid computing.

DeepSeek's answer: compute the scores with a **much smaller, much cheaper** model first, then do the expensive attention only on the winners.

> [!DEFINITION] Top-k selection
> From a list of scores, keep the $$k$$ largest and drop the rest. If $$k = 2{,}048$$, each token reads at most 2,048 earlier tokens, no matter whether the text is 10,000 or 128,000 tokens long.

## 8. DeepSeek Sparse Attention (DSA)

DSA arrived with DeepSeek-V3.2 (2025). It has two pieces.

> [!PAPER] DeepSeek-AI (2025), DeepSeek-V3.2 · Section 2.1 · page 4, Figure 2
> [![Figure 2 of the DeepSeek-V3.2 paper: the MLA attention architecture with the lightning indexer and top-k selector added in green](/img/attention/papers/dsv32-figure2.png)](/img/attention/papers/dsv32-figure2.png)
>
> **Context:** The full attention block of DeepSeek-V3.2. The black parts are MLA from Part 2 (compressed latent $$c_t^{KV}$$, separate RoPE key $$k_t^R$$). The green parts are new.
>
> **What it says:** The lightning indexer (green box) reads the input $$h_t$$ through its own small queries $$q^I_{t,j}$$, keys $$k^I_t$$ and weights $$w^I_{t,j}$$, scores all earlier tokens (the little bar chart), and the "Top-k Selector" passes only the best entries $$[c_t^{KV}; k_t^R]$$ to the main "Multi-Query Attention (Core Attention)".
>
> **Why it matters:** DSA does not replace MLA, it sits in front of it like a filter. Everything from Part 2 is still there; it just runs on 2,048 chosen tokens instead of all of them.
>
> [Read the paper on arXiv](https://arxiv.org/abs/2512.02556)

<figure class="fig"><svg viewBox="0 0 760 230" role="img" aria-label="DeepSeek Sparse Attention: a small lightning indexer scores every earlier token, the top k are kept, and the main attention runs only over those."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box" x="20.0" y="112.0" width="90.0" height="44.0" rx="10"/><text class="t-note" x="65.0" y="132.0" text-anchor="middle">token t</text><text class="t-tick" x="65.0" y="148.0" text-anchor="middle">query</text><line class="edge" x1="112.0" y1="134.0" x2="160.0" y2="134.0" marker-end="url(#ah)"/><rect class="box-2" x="160.0" y="96.0" width="150.0" height="76.0" rx="10"/><text class="t-note" x="235.0" y="120.0" text-anchor="middle">lightning indexer</text><text class="t-tick" x="235.0" y="140.0" text-anchor="middle">few heads, small, FP8</text><text class="t-tick" x="235.0" y="158.0" text-anchor="middle">scores every earlier token</text><line class="edge" x1="312.0" y1="134.0" x2="360.0" y2="134.0" marker-end="url(#ah)"/><rect class="box" x="360.0" y="108.0" width="110.0" height="52.0" rx="10"/><text class="t-note" x="415.0" y="130.0" text-anchor="middle">top-k</text><text class="t-tick" x="415.0" y="148.0" text-anchor="middle">keep the best k</text><line class="edge" x1="472.0" y1="134.0" x2="520.0" y2="134.0" marker-end="url(#ah)"/><rect class="box-1" x="520.0" y="96.0" width="130.0" height="76.0" rx="10"/><text class="t-note" x="585.0" y="120.0" text-anchor="middle">main attention</text><text class="t-tick" x="585.0" y="140.0" text-anchor="middle">only over the</text><text class="t-tick" x="585.0" y="158.0" text-anchor="middle">k chosen tokens</text><line class="edge" x1="652.0" y1="134.0" x2="690.0" y2="134.0" marker-end="url(#ah)"/><text class="t-note" x="700.0" y="139.0" text-anchor="start">output</text><text class="t-tick" x="235.0" y="76.0" text-anchor="middle">cheap, but looks at all L tokens</text><text class="t-tick" x="585.0" y="76.0" text-anchor="middle">expensive, but looks at only k</text><text class="t-tick" x="20.0" y="214.0" text-anchor="start">DeepSeek-V3.2: k = 2,048. The main attention cost drops from L² to L × k; the indexer is still L², but tiny.</text></svg><figcaption>DeepSeek Sparse Attention. The lightning indexer is small and cheap, but it scores every earlier token. Top-k keeps the best k. The big main attention then runs only over those k tokens.</figcaption></figure>

### Piece 1: the lightning indexer

> [!DEFINITION] Lightning indexer
> A tiny attention-like scorer that runs **before** the real attention. For the current token $$t$$ and each earlier token $$s$$, it produces one number, the **index score** $$I_{t,s}$$: "how useful will token $$s$$ be for token $$t$$?" It has only a few heads and can run in low precision, so it is very fast.

The index score is:

$$
I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \cdot \operatorname{ReLU}\!\left(q^I_{t,j} \cdot k^I_s\right)
$$

where:

- $$t$$ is the current (query) token and $$s$$ an earlier token;
- $$H^I$$ is the number of **indexer heads** (a small number);
- $$q^I_{t,j}$$ is the indexer query of token $$t$$ for indexer head $$j$$;
- $$k^I_s$$ is the indexer key of token $$s$$ (one key per token, shared by all indexer heads);
- $$w^I_{t,j}$$ is a weight that token $$t$$ gives to indexer head $$j$$ (how much to trust that head for this token);
- $$\operatorname{ReLU}(x) = \max(0, x)$$.

> [!DEFINITION] ReLU
> A very simple function: negative numbers become 0, positive numbers stay as they are. $$\operatorname{ReLU}(-3) = 0$$, $$\operatorname{ReLU}(2) = 2$$. It is cheaper than softmax because it needs no exponentials and no sum over all tokens.

> [!DEFINITION] FP8
> Numbers stored in 8 bits (1 byte) instead of the usual 16. Less precise, but twice as small and much faster on modern GPUs. Good enough for *ranking* tokens, which is all the indexer must do.

The paper explains both choices: ReLU was chosen "for throughput consideration", and the indexer "can be implemented in FP8".

> [!PAPER] DeepSeek-V3.2 · Section 2.1 · page 3, Equation (1)
> [![Equation 1 of the DeepSeek-V3.2 paper: the lightning indexer score as a weighted sum over indexer heads of ReLU of query-key dot products](/img/attention/papers/dsv32-indexer.png)](/img/attention/papers/dsv32-indexer.png)
>
> **Context:** The definition of the lightning indexer, the heart of DSA.
>
> **What it says:** Exactly the equation above, with each symbol explained: $$H^I$$ indexer heads, $$q^I_{t,j}$$ and $$w^I_{t,j}$$ from the query token, $$k^I_s$$ from the earlier token. "Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable."
>
> **Why it matters:** Compare it with softmax attention: there is no softmax and no value mix, just a score. The indexer only has to **rank** tokens, which is a much easier job than attending to them.

### Piece 2: top-k token selection

For each query token $$t$$, keep only the $$k$$ earlier tokens with the highest index scores, and run the real attention over just those:

$$
S_t = \operatorname{Top\text{-}k}\big(I_{t,:}\big), \qquad u_t = \operatorname{Attn}\big(h_t, \{\, c_s : s \in S_t \,\}\big)
$$

where:

- $$I_{t,:}$$ means all of token $$t$$'s index scores;
- $$S_t$$ is the set of selected positions;
- $$h_t$$ is the current token's hidden vector;
- $$c_s$$ is the cached key-value entry of token $$s$$ (in DeepSeek, the MLA latent from [Part 2](attention-2-mqa-gqa-mla.md));
- $$u_t$$ is the attention output.

DeepSeek-V3.2 uses $$k = 2{,}048$$.

### What it saves

> [!PAPER] DeepSeek-V3.2 · Section 2.3, Inference Costs · page 5
> [![The DeepSeek-V3.2 inference cost paragraph: DSA reduces core attention complexity from O(L squared) to O(Lk), while the lightning indexer still has O(L squared) complexity but much less computation](/img/attention/papers/dsv32-complexity.png)](/img/attention/papers/dsv32-complexity.png)
>
> **Context:** The paper's own cost argument.
>
> **What it says:** "DSA reduces the core attention complexity of the main model from $$O(L^2)$$ to $$O(Lk)$$". The indexer "still has a complexity of $$O(L^2)$$", but "requires much less computation compared with MLA".
>
> **Why it matters:** "Much less" can be put in numbers; see just below.

> [!DEFINITION] Big-O notation
> A shorthand for how cost grows with size. $$O(L^2)$$: grows with the square of the length $$L$$. $$O(Lk)$$: grows with $$L$$ times a fixed number $$k$$, so only linearly in $$L$$.

So the expensive part (the big attention with 128 heads) becomes linear in length. The cheap part (the indexer) is still quadratic, but it is so small and so fast that it costs much less.

**How much less, exactly?** DeepSeek-V3.2's published configuration gives the sizes: the indexer has $$H^I = 64$$ heads of size $$d^I = 128$$; the main attention has 128 heads, each scoring against a cached entry of $$512 + 64 = 576$$ numbers. Counting multiply-adds for the scoring step only:

$$
\text{dense} \approx P \times 128 \times 576,
\qquad
\text{DSA} \approx \underbrace{P \times 64 \times 128}_{\text{indexer, FP8}} + \underbrace{L\,k \times 128 \times 576}_{\text{main, top-}k}
$$

where $$P = L(L+1)/2$$ is the number of (query, earlier token) pairs and $$k = 2{,}048$$. At $$L = 131{,}072$$ tokens:

| | Pairs scored | Multiply-adds per pair | Total |
|---|---|---|---|
| Dense MLA | 8.59 billion | 73,728 | $$6.3 \times 10^{14}$$ |
| DSA indexer | 8.59 billion | 8,192 (in FP8) | $$7.0 \times 10^{13}$$ |
| DSA main | 0.27 billion (at most) | 73,728 | $$2.0 \times 10^{13}$$ |
| **DSA total** | | | $$9.0 \times 10^{13}$$, about **7× less** |

The indexer still touches every pair, but each touch is 9 times cheaper and runs in 8-bit numbers. The main attention touches 32 times fewer pairs. (This counts only the scoring arithmetic. Real speed also depends on memory reads and GPU code; the paper's Figure 3 below shows the measured result.)

### How the indexer learns: two training stages

The indexer starts out random. How does it learn which tokens matter? It copies the model's own attention. DeepSeek trains it in two stages:

**Stage 1, dense warm-up.** Keep normal full attention, **freeze the whole model**, and train only the indexer to predict where the model's attention goes.

> [!QUOTE] DeepSeek-V3.2 paper, Section 2.1.1
> "for the t-th query token, we first aggregate the main attention scores by summing across all attention heads. This sum is then L1-normalized along the sequence dimension to produce a target distribution $$p_{t,:}$$"

As an equation, with $$A^{(h)}_{t,s}$$ the main attention weight of head $$h$$ from token $$t$$ to token $$s$$:

$$
p_{t,s} = \frac{\sum_{h} A^{(h)}_{t,s}}{\sum_{s'} \sum_{h} A^{(h)}_{t,s'}}
$$

The top sums over heads; the bottom divides by the row total so the numbers add up to 1.

The training loss compares the indexer's scores (turned into a distribution by softmax) with that target:

$$
\mathcal{L}^I = \sum_t D_{\mathrm{KL}}\Big(p_{t,:} \,\Big\|\, \operatorname{Softmax}(I_{t,:})\Big)
$$

> [!DEFINITION] L1-normalised and KL divergence
> **L1-normalised:** divided by its total, so the numbers add up to 1, like percentages. **KL divergence** $$D_{\mathrm{KL}}(p \,\|\, q)$$: a measure of how different two such distributions are. It is 0 when they are identical and grows as they differ. Training makes it as small as possible, so the indexer's ranking matches the real attention.

This stage is short: 1,000 steps, 2.1 billion tokens, learning rate $$10^{-3}$$.

**Stage 2, sparse training.** Switch on top-k selection and train the **whole** model to work with it: 15,000 steps, 943.7 billion tokens. The indexer keeps learning from the KL loss, but it is trained separately: the paper says "we detach the indexer input from the computational graph for separate optimization", meaning the indexer's learning signal does not flow back into the main model.

> [!NOTE] Built on MLA
> DSA is added on top of the MLA attention from Part 2, in "MQA mode": each cached latent is shared by all query heads. That way, choosing a token means fetching **one** small latent, which all 128 heads then use. DeepSeek reports no substantial quality drop against the dense model it started from (DeepSeek-V3.1-Terminus), with large speedups at long context.

## 9. I trained a lightning indexer on a real model

To see whether this really works, and not just trust the paper, I built a small DSA for **Qwen2.5-0.5B** and ran **stage 1** (the dense warm-up) exactly as described: model frozen, indexer trained with the KL loss against the model's own head-summed, L1-normalised attention.

**The indexer**, one per layer, 24 in total:

- 4 indexer heads of size 32, one shared key per token, ReLU, and the weights $$w$$;
- RoPE on its queries and keys, so it knows positions;
- a LayerNorm on its input (more on this below);
- **148,736 parameters per layer**, about 8% of the size of the attention layer it serves (1,836,160).

```python
class LightningIndexer(torch.nn.Module):
    """I[t, s] = sum_j w[t, j] * ReLU(q[t, j] . k[s])   (DeepSeek-V3.2, eq. 1). One shared key per token."""
    def __init__(self, d_model):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d_model)                        # Qwen's hidden states have a few huge values; tame them
        self.q = torch.nn.Linear(d_model, HI * DI, bias=False)
        self.k = torch.nn.Linear(d_model, DI, bias=False)
        self.w = torch.nn.Linear(d_model, HI, bias=False)

    def forward(self, h):                                              # h: (T, d_model)
        n, h = h.shape[0], self.norm(h)
        q = rope(self.q(h).view(n, HI, DI).transpose(0, 1))            # (HI, T, DI)
        k = rope(self.k(h))                                            # (T, DI)
        return torch.einsum('jt,jts->ts', self.w(h).T, F.relu(q @ k.T))   # (T, T)
```

The target and the loss, as in the paper (summing the 14 heads, then dividing by the total; the KL loss written as cross-entropy, which differs from KL only by a constant):

```python
        A = output[1][0]                                               # (heads, T, T)
        captured[layer]['p'] = (A.sum(0) / A.sum(0).sum(-1, keepdim=True)).detach()   # sum over heads, L1-normalise
```

```python
def kl_loss(ix, h, p):
    I = ix(h).masked_fill(~CAUSAL, float('-inf'))
    logq = torch.log_softmax(I, -1)
    return -(p * logq.masked_fill(~CAUSAL, 0)).sum(-1).mean()          # KL(p || softmax(I)) up to a constant
```

**Training:** 400 steps per layer, learning rate $$10^{-3}$$ (the paper's warm-up rate), on six 2,048-token chunks of *The Adventures of Sherlock Holmes*. **Testing** used a different book, *Pride and Prejudice*, so the indexer never saw the test text.

> [!WARNING] A real problem I hit: "dead" indexers
> My first run had no LayerNorm. In layers 22 and 23, the training loss got stuck at exactly 6.627. That number is the loss of a completely flat guess over 2,048 tokens: the indexer had "died". The ReLU outputs were all zero, so every token got the same score and nothing could be learned. The cause: Qwen's hidden vectors contain a few enormous values, especially in late layers, which pushed the first updates too far. Normalising the input (LayerNorm) and limiting the size of each update (gradient clipping) fixed it, and all 24 layers then trained normally. ReLU's "nothing below zero" makes this failure possible, so it is worth knowing about.

### Result 1: does it pick the tokens the model really attends to?

For each layer and each query token (positions 1,024 to 2,047), I measured **what share of the model's real attention lands on the chosen tokens**. 1.0 would mean the chosen tokens hold all of the attention.

```text
budget 64: attention captured  window 0.423  sinks+window 0.721  indexer 0.776  (untrained, layer 0: 0.042)
budget 256: attention captured  window 0.520  sinks+window 0.820  indexer 0.887  (untrained, layer 0: 0.177)
```

| Budget per token | Last k tokens | 4 sinks + window | Trained indexer | Untrained indexer (layer 1) |
|---|---|---|---|---|
| 64 | 42.3% | 72.1% | **77.6%** | 4.2% |
| 256 | 52.0% | 82.0% | **88.7%** | 17.7% |

- An **untrained** indexer catches almost nothing (4.2%), so the training clearly did the work.
- The **trained** indexer beats the strong "sinks + window" rule at both budgets. It found the sinks and the recent tokens by itself, *plus* some important far-away tokens.

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="Share of the model's real attention that lands on the 64 chosen tokens, layer by layer, for the trained indexer and for 4 sinks plus a window."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="560" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0.00</text><line class="grid" x1="64" y1="195.0" x2="560" y2="195.0"/><text class="t-tick" x="56.0" y="199.0" text-anchor="end">0.25</text><line class="grid" x1="64" y1="140.0" x2="560" y2="140.0"/><text class="t-tick" x="56.0" y="144.0" text-anchor="end">0.50</text><line class="grid" x1="64" y1="85.0" x2="560" y2="85.0"/><text class="t-tick" x="56.0" y="89.0" text-anchor="end">0.75</text><line class="grid" x1="64" y1="30.0" x2="560" y2="30.0"/><text class="t-tick" x="56.0" y="34.0" text-anchor="end">1.00</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="171.8" y="270.0" text-anchor="middle">6</text><text class="t-tick" x="301.2" y="270.0" text-anchor="middle">12</text><text class="t-tick" x="430.6" y="270.0" text-anchor="middle">18</text><text class="t-tick" x="560.0" y="270.0" text-anchor="middle">24</text><line class="axis" x1="64" y1="250" x2="560" y2="250"/><polyline class="l1" points="64.0,111.1 85.6,113.5 107.1,77.9 128.7,66.6 150.3,85.2 171.8,86.7 193.4,71.1 215.0,40.5 236.5,43.8 258.1,49.6 279.7,63.4 301.2,61.3 322.8,69.8 344.3,70.2 365.9,61.2 387.5,66.3 409.0,62.5 430.6,95.1 452.2,54.9 473.7,53.9 495.3,86.9 516.9,93.2 538.4,167.7 560.0,150.9"/><g class="mark"><title>indexer top-64, 1: 0.63</title><circle class="s1 ring" cx="64.0" cy="111.1" r="5"/></g><g class="mark"><title>indexer top-64, 2: 0.62</title><circle class="s1 ring" cx="85.6" cy="113.5" r="5"/></g><g class="mark"><title>indexer top-64, 3: 0.78</title><circle class="s1 ring" cx="107.1" cy="77.9" r="5"/></g><g class="mark"><title>indexer top-64, 4: 0.83</title><circle class="s1 ring" cx="128.7" cy="66.6" r="5"/></g><g class="mark"><title>indexer top-64, 5: 0.75</title><circle class="s1 ring" cx="150.3" cy="85.2" r="5"/></g><g class="mark"><title>indexer top-64, 6: 0.74</title><circle class="s1 ring" cx="171.8" cy="86.7" r="5"/></g><g class="mark"><title>indexer top-64, 7: 0.81</title><circle class="s1 ring" cx="193.4" cy="71.1" r="5"/></g><g class="mark"><title>indexer top-64, 8: 0.95</title><circle class="s1 ring" cx="215.0" cy="40.5" r="5"/></g><g class="mark"><title>indexer top-64, 9: 0.94</title><circle class="s1 ring" cx="236.5" cy="43.8" r="5"/></g><g class="mark"><title>indexer top-64, 10: 0.91</title><circle class="s1 ring" cx="258.1" cy="49.6" r="5"/></g><g class="mark"><title>indexer top-64, 11: 0.85</title><circle class="s1 ring" cx="279.7" cy="63.4" r="5"/></g><g class="mark"><title>indexer top-64, 12: 0.86</title><circle class="s1 ring" cx="301.2" cy="61.3" r="5"/></g><g class="mark"><title>indexer top-64, 13: 0.82</title><circle class="s1 ring" cx="322.8" cy="69.8" r="5"/></g><g class="mark"><title>indexer top-64, 14: 0.82</title><circle class="s1 ring" cx="344.3" cy="70.2" r="5"/></g><g class="mark"><title>indexer top-64, 15: 0.86</title><circle class="s1 ring" cx="365.9" cy="61.2" r="5"/></g><g class="mark"><title>indexer top-64, 16: 0.84</title><circle class="s1 ring" cx="387.5" cy="66.3" r="5"/></g><g class="mark"><title>indexer top-64, 17: 0.85</title><circle class="s1 ring" cx="409.0" cy="62.5" r="5"/></g><g class="mark"><title>indexer top-64, 18: 0.70</title><circle class="s1 ring" cx="430.6" cy="95.1" r="5"/></g><g class="mark"><title>indexer top-64, 19: 0.89</title><circle class="s1 ring" cx="452.2" cy="54.9" r="5"/></g><g class="mark"><title>indexer top-64, 20: 0.89</title><circle class="s1 ring" cx="473.7" cy="53.9" r="5"/></g><g class="mark"><title>indexer top-64, 21: 0.74</title><circle class="s1 ring" cx="495.3" cy="86.9" r="5"/></g><g class="mark"><title>indexer top-64, 22: 0.71</title><circle class="s1 ring" cx="516.9" cy="93.2" r="5"/></g><g class="mark"><title>indexer top-64, 23: 0.37</title><circle class="s1 ring" cx="538.4" cy="167.7" r="5"/></g><g class="mark"><title>indexer top-64, 24: 0.45</title><circle class="s1 ring" cx="560.0" cy="150.9" r="5"/></g><polyline class="l2" points="64.0,130.1 85.6,122.1 107.1,98.6 128.7,74.6 150.3,91.5 171.8,94.7 193.4,77.6 215.0,44.8 236.5,49.6 258.1,56.3 279.7,73.3 301.2,73.5 322.8,77.2 344.3,83.6 365.9,70.6 387.5,77.2 409.0,77.0 430.6,102.2 452.2,59.3 473.7,57.7 495.3,103.4 516.9,105.8 538.4,211.6 560.0,183.0"/><g class="mark"><title>4 sinks + window 60, 1: 0.54</title><circle class="s2 ring" cx="64.0" cy="130.1" r="5"/></g><g class="mark"><title>4 sinks + window 60, 2: 0.58</title><circle class="s2 ring" cx="85.6" cy="122.1" r="5"/></g><g class="mark"><title>4 sinks + window 60, 3: 0.69</title><circle class="s2 ring" cx="107.1" cy="98.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 4: 0.80</title><circle class="s2 ring" cx="128.7" cy="74.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 5: 0.72</title><circle class="s2 ring" cx="150.3" cy="91.5" r="5"/></g><g class="mark"><title>4 sinks + window 60, 6: 0.71</title><circle class="s2 ring" cx="171.8" cy="94.7" r="5"/></g><g class="mark"><title>4 sinks + window 60, 7: 0.78</title><circle class="s2 ring" cx="193.4" cy="77.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 8: 0.93</title><circle class="s2 ring" cx="215.0" cy="44.8" r="5"/></g><g class="mark"><title>4 sinks + window 60, 9: 0.91</title><circle class="s2 ring" cx="236.5" cy="49.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 10: 0.88</title><circle class="s2 ring" cx="258.1" cy="56.3" r="5"/></g><g class="mark"><title>4 sinks + window 60, 11: 0.80</title><circle class="s2 ring" cx="279.7" cy="73.3" r="5"/></g><g class="mark"><title>4 sinks + window 60, 12: 0.80</title><circle class="s2 ring" cx="301.2" cy="73.5" r="5"/></g><g class="mark"><title>4 sinks + window 60, 13: 0.79</title><circle class="s2 ring" cx="322.8" cy="77.2" r="5"/></g><g class="mark"><title>4 sinks + window 60, 14: 0.76</title><circle class="s2 ring" cx="344.3" cy="83.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 15: 0.82</title><circle class="s2 ring" cx="365.9" cy="70.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 16: 0.79</title><circle class="s2 ring" cx="387.5" cy="77.2" r="5"/></g><g class="mark"><title>4 sinks + window 60, 17: 0.79</title><circle class="s2 ring" cx="409.0" cy="77.0" r="5"/></g><g class="mark"><title>4 sinks + window 60, 18: 0.67</title><circle class="s2 ring" cx="430.6" cy="102.2" r="5"/></g><g class="mark"><title>4 sinks + window 60, 19: 0.87</title><circle class="s2 ring" cx="452.2" cy="59.3" r="5"/></g><g class="mark"><title>4 sinks + window 60, 20: 0.87</title><circle class="s2 ring" cx="473.7" cy="57.7" r="5"/></g><g class="mark"><title>4 sinks + window 60, 21: 0.67</title><circle class="s2 ring" cx="495.3" cy="103.4" r="5"/></g><g class="mark"><title>4 sinks + window 60, 22: 0.66</title><circle class="s2 ring" cx="516.9" cy="105.8" r="5"/></g><g class="mark"><title>4 sinks + window 60, 23: 0.17</title><circle class="s2 ring" cx="538.4" cy="211.6" r="5"/></g><g class="mark"><title>4 sinks + window 60, 24: 0.30</title><circle class="s2 ring" cx="560.0" cy="183.0" r="5"/></g><line class="grid" x1="566.0" y1="150.9" x2="576.0" y2="150.9"/><rect class="s1" x="580.0" y="145.9" width="10" height="10" rx="2"/><text class="t-note" x="596.0" y="154.9" text-anchor="start">indexer top-64: 0.45</text><line class="grid" x1="566.0" y1="183.0" x2="576.0" y2="183.0"/><rect class="s2" x="580.0" y="178.0" width="10" height="10" rx="2"/><text class="t-note" x="596.0" y="187.0" text-anchor="start">4 sinks + window 60: 0.30</text><text class="t-tick" x="312.0" y="290.0" text-anchor="middle">layer</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">share of attention captured</text></svg><figcaption>Layer by layer, the share of real attention caught by 64 tokens. The trained indexer (blue) is above "4 sinks + window" (orange) in every layer. The last two layers spread their attention widely, so 64 tokens catch less there.</figcaption></figure>

Here is one real example: the very last token of a test chunk, in layer 13.

<figure class="fig"><svg viewBox="0 0 760 236" role="img" aria-label="For one real query, the tokens the model attends to most compared with the tokens the trained lightning indexer selected."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="30.0" y="24.0" text-anchor="start">Layer 13, the last token (position 2,047): which earlier tokens matter?</text><text class="t-tick" x="30.0" y="48.0" text-anchor="start">top 64 tokens by the model's real attention</text><line class="axis" x1="30" y1="70" x2="730" y2="70"/><rect class="s2" x="28.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="227.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="672.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="684.0" y="56" width="3" height="28" rx="1"/><rect class="s2" x="685.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="686.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="686.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="689.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="689.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="691.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="694.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="701.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="702.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="703.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="703.9" y="56" width="3" height="28" rx="1"/><rect class="s2" x="704.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="704.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="705.9" y="56" width="3" height="28" rx="1"/><rect class="s2" x="706.3" y="56" width="3" height="28" rx="1"/><rect class="s2" x="707.3" y="56" width="3" height="28" rx="1"/><rect class="s2" x="707.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="708.3" y="56" width="3" height="28" rx="1"/><rect class="s2" x="708.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="709.4" y="56" width="3" height="28" rx="1"/><rect class="s2" x="709.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="710.0" y="56" width="3" height="28" rx="1"/><rect class="s2" x="710.4" y="56" width="3" height="28" rx="1"/><rect class="s2" x="710.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="711.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="711.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="712.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="713.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="713.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="714.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="714.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="714.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="715.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="715.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="716.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="717.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="717.9" y="56" width="3" height="28" rx="1"/><rect class="s2" x="718.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="718.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="718.9" y="56" width="3" height="28" rx="1"/><rect class="s2" x="719.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="720.0" y="56" width="3" height="28" rx="1"/><rect class="s2" x="720.3" y="56" width="3" height="28" rx="1"/><rect class="s2" x="720.6" y="56" width="3" height="28" rx="1"/><rect class="s2" x="721.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="722.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="723.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="724.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="724.7" y="56" width="3" height="28" rx="1"/><rect class="s2" x="725.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="725.4" y="56" width="3" height="28" rx="1"/><rect class="s2" x="725.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="726.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="726.4" y="56" width="3" height="28" rx="1"/><rect class="s2" x="726.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="727.1" y="56" width="3" height="28" rx="1"/><rect class="s2" x="727.5" y="56" width="3" height="28" rx="1"/><rect class="s2" x="727.8" y="56" width="3" height="28" rx="1"/><rect class="s2" x="728.2" y="56" width="3" height="28" rx="1"/><rect class="s2" x="728.5" y="56" width="3" height="28" rx="1"/><text class="t-tick" x="30.0" y="112.0" text-anchor="start">64 tokens picked by the trained indexer</text><line class="axis" x1="30" y1="134" x2="730" y2="134"/><rect class="s1" x="28.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="465.2" y="120" width="3" height="28" rx="1"/><rect class="s1" x="662.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="667.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="672.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="683.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="684.0" y="120" width="3" height="28" rx="1"/><rect class="s1" x="685.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="686.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="689.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="701.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="702.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="703.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="703.9" y="120" width="3" height="28" rx="1"/><rect class="s1" x="704.2" y="120" width="3" height="28" rx="1"/><rect class="s1" x="704.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="705.9" y="120" width="3" height="28" rx="1"/><rect class="s1" x="706.3" y="120" width="3" height="28" rx="1"/><rect class="s1" x="707.3" y="120" width="3" height="28" rx="1"/><rect class="s1" x="707.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="708.3" y="120" width="3" height="28" rx="1"/><rect class="s1" x="708.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="709.4" y="120" width="3" height="28" rx="1"/><rect class="s1" x="709.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="710.0" y="120" width="3" height="28" rx="1"/><rect class="s1" x="710.4" y="120" width="3" height="28" rx="1"/><rect class="s1" x="711.1" y="120" width="3" height="28" rx="1"/><rect class="s1" x="711.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="712.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="713.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="713.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="714.1" y="120" width="3" height="28" rx="1"/><rect class="s1" x="714.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="714.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="715.2" y="120" width="3" height="28" rx="1"/><rect class="s1" x="715.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="716.2" y="120" width="3" height="28" rx="1"/><rect class="s1" x="717.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="717.9" y="120" width="3" height="28" rx="1"/><rect class="s1" x="718.2" y="120" width="3" height="28" rx="1"/><rect class="s1" x="718.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="718.9" y="120" width="3" height="28" rx="1"/><rect class="s1" x="719.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="720.0" y="120" width="3" height="28" rx="1"/><rect class="s1" x="720.3" y="120" width="3" height="28" rx="1"/><rect class="s1" x="720.6" y="120" width="3" height="28" rx="1"/><rect class="s1" x="721.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="722.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="723.4" y="120" width="3" height="28" rx="1"/><rect class="s1" x="723.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="724.1" y="120" width="3" height="28" rx="1"/><rect class="s1" x="724.4" y="120" width="3" height="28" rx="1"/><rect class="s1" x="724.7" y="120" width="3" height="28" rx="1"/><rect class="s1" x="725.1" y="120" width="3" height="28" rx="1"/><rect class="s1" x="725.4" y="120" width="3" height="28" rx="1"/><rect class="s1" x="725.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="726.1" y="120" width="3" height="28" rx="1"/><rect class="s1" x="726.4" y="120" width="3" height="28" rx="1"/><rect class="s1" x="726.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="727.1" y="120" width="3" height="28" rx="1"/><rect class="s1" x="727.5" y="120" width="3" height="28" rx="1"/><rect class="s1" x="727.8" y="120" width="3" height="28" rx="1"/><rect class="s1" x="728.2" y="120" width="3" height="28" rx="1"/><rect class="s1" x="728.5" y="120" width="3" height="28" rx="1"/><text class="t-tick" x="30.0" y="196.0" text-anchor="middle">0</text><text class="t-tick" x="205.1" y="196.0" text-anchor="middle">512</text><text class="t-tick" x="380.2" y="196.0" text-anchor="middle">1,024</text><text class="t-tick" x="555.3" y="196.0" text-anchor="middle">1,536</text><text class="t-tick" x="730.0" y="196.0" text-anchor="middle">2,047</text><text class="t-tick" x="30.0" y="222.0" text-anchor="start">Both rows agree on 58 of 64 tokens. Many picks are recent tokens (right edge) or the first token (left edge).</text></svg><figcaption>For one real query: the 64 tokens the model actually attends to most (top) and the 64 tokens the trained indexer picked (bottom), across positions 0 to 2,047. They agree on 58 of 64. Both include the first token (the sink) and a cluster of recent tokens, plus a few far-away ones.</figcaption></figure>

### Result 2: does the model still work with sparse attention?

The real test: run Qwen with **every layer** using top-k attention chosen by its indexer, and measure perplexity on *Pride and Prejudice*.

<figure class="fig"><svg viewBox="0 0 780 286" role="img" aria-label="Perplexity of Qwen2.5-0.5B when each token may look at only 64 or 256 earlier tokens, chosen in three different ways, against full attention."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><g class="mark"><title>64: last 64 tokens: perplexity 132.70</title><path class="s4 bar-mark" d="M230,14H576.0Q580.0,14 580.0,18V30Q580.0,34 576.0,34H230Z"/></g><text class="t-tick" x="220.0" y="29.0" text-anchor="end">64: last 64 tokens</text><text class="t-val" x="588.0" y="29.0" text-anchor="start">132.7, bar cut off</text><g class="mark"><title>64: 4 sinks + last 60: perplexity 22.45</title><path class="s1 bar-mark" d="M230,48H422.3975113647957Q426.3975113647957,48 426.3975113647957,52V64Q426.3975113647957,68 422.3975113647957,68H230Z"/></g><text class="t-tick" x="220.0" y="63.0" text-anchor="end">64: 4 sinks + last 60</text><text class="t-val" x="434.4" y="63.0" text-anchor="start">22.4</text><g class="mark"><title>64: indexer top-64: perplexity 18.93</title><path class="s3 bar-mark" d="M230,82H391.6787553819154Q395.6787553819154,82 395.6787553819154,86V98Q395.6787553819154,102 391.6787553819154,102H230Z"/></g><text class="t-tick" x="220.0" y="97.0" text-anchor="end">64: indexer top-64</text><text class="t-val" x="403.7" y="97.0" text-anchor="start">18.9</text><g class="mark"><title>256: last 256 tokens: perplexity 67.90</title><path class="s4 bar-mark" d="M230,116H576.0Q580.0,116 580.0,120V132Q580.0,136 576.0,136H230Z"/></g><text class="t-tick" x="220.0" y="131.0" text-anchor="end">256: last 256 tokens</text><text class="t-val" x="588.0" y="131.0" text-anchor="start">67.9, bar cut off</text><g class="mark"><title>256: 4 sinks + last 252: perplexity 18.83</title><path class="s1 bar-mark" d="M230,150H390.7732712689401Q394.7732712689401,150 394.7732712689401,154V166Q394.7732712689401,170 390.7732712689401,170H230Z"/></g><text class="t-tick" x="220.0" y="165.0" text-anchor="end">256: 4 sinks + last 252</text><text class="t-val" x="402.8" y="165.0" text-anchor="start">18.8</text><g class="mark"><title>256: indexer top-256: perplexity 17.62</title><path class="s3 bar-mark" d="M230,184H380.1706943925657Q384.1706943925657,184 384.1706943925657,188V200Q384.1706943925657,204 380.1706943925657,204H230Z"/></g><text class="t-tick" x="220.0" y="199.0" text-anchor="end">256: indexer top-256</text><text class="t-val" x="392.2" y="199.0" text-anchor="start">17.6</text><g class="mark"><title>full attention (2,048): perplexity 17.39</title><path class="s2 bar-mark" d="M230,218H378.1619576630461Q382.1619576630461,218 382.1619576630461,222V234Q382.1619576630461,238 378.1619576630461,238H230Z"/></g><text class="t-tick" x="220.0" y="233.0" text-anchor="end">full attention (2,048)</text><text class="t-val" x="390.2" y="233.0" text-anchor="start">17.4</text><line class="axis" x1="230" y1="10" x2="230" y2="252"/><text class="t-tick" x="230.0" y="274.0" text-anchor="start">perplexity on Pride and Prejudice, tokens 1,024 to 2,047 (lower is better)</text></svg><figcaption>Perplexity when each token may look at only 64 or 256 earlier tokens, chosen three ways, against full attention (lower is better). The trained indexer comes closest to full attention at both budgets.</figcaption></figure>

| Budget per token | Last k tokens | 4 sinks + window | Trained indexer, top-k | Full attention |
|---|---|---|---|---|
| 64 | 132.7 | 22.45 | **18.93** | 17.39 |
| 256 | 67.9 | 18.83 | **17.62** | 17.39 |

With 256 tokens per query (full attention reads 1,025 to 2,048 at these positions), the indexer version reaches **17.62**, against 17.39 for full attention, **without retraining the model at all**. With 64 tokens it stays at 18.93, clearly better than the best fixed pattern (22.45).

> [!NOTE] What this experiment does and does not show
> - It shows the **quality** side: a small trained indexer can pick the right tokens, even in a model never trained for sparse attention.
> - It is only **stage 1**. DeepSeek's stage 2 retrains the whole model with sparse attention, which should close the remaining gap.
> - It does **not** show speed. My version still computes the full score grid and then hides most of it with a mask, so it is not faster. Real speedups need special GPU code that fetches only the selected tokens.
> - It is one small model, one test book and 2,048-token texts, so read the exact numbers as an illustration, not a benchmark.

## 10. Proof of the runs

Both scripts, exactly as they ran (the same outputs quoted above):

<figure class="fig"><img src="/img/attention/part3-sparse-run.png" alt="Terminal output of part3_sparse.py: sliding-window checks, receptive field, rolling cache, KV memory and decode timing" loading="lazy" /><figcaption>Output of part3_sparse.py on an Apple M5 Pro.</figcaption></figure>

<figure class="fig"><img src="/img/attention/part3-qwen-run.png" alt="Terminal output of part3_qwen.py: window and sink perplexities, indexer warm-up losses per layer, attention captured and sparse perplexities" loading="lazy" /><figcaption>Output of part3_qwen.py: Qwen2.5-0.5B with windows, sinks and trained lightning indexers. The whole run took 106 seconds.</figcaption></figure>

## The impact, and where you meet it

Long context went from a research problem to a standard feature between 2023 and 2025, and the ideas in this part are a big reason why:

- **Mistral 7B** (2023) made sliding windows and the rolling cache standard in open models.
- **Gemma 2 and 3, gpt-oss** made local and global layers the default way to reach 128K tokens with a manageable cache.
- **StreamingLLM** showed how to keep a model running on a never-ending stream; keeping attention sinks is now a standard trick in serving systems.
- **DeepSeek-V3.2** showed that a learned top-k selection can keep quality while cutting long-context cost sharply.

> [!PAPER] DeepSeek-V3.2 · Section 2.3 · page 6, Figure 3
> [![Figure 3 of the DeepSeek-V3.2 paper: cost per million tokens against token position for prefilling and decoding, DeepSeek-V3.1-Terminus rising steeply and DeepSeek-V3.2 staying low](/img/attention/papers/dsv32-figure3.png)](/img/attention/papers/dsv32-figure3.png)
>
> **Context:** The measured serving cost (in dollars per million tokens, on H800 GPUs) of the same model family with dense attention (V3.1-Terminus, blue) and with DSA (V3.2, orange), against the position of the token in the text.
>
> **What it says:** With dense attention, cost grows steadily with position: a token at position 128K costs about \$2.1 per million to decode. With DSA it stays near \$0.25. For short texts (the far left) DSA is slightly more expensive, because the indexer is extra work.
>
> **Why it matters:** This is the use case in one picture: long documents, long chats and long reasoning traces become several times cheaper to serve, which in turn makes them practical to offer at all.

> [!PAPER] StreamingLLM · Section 4.5, Efficiency · page 9, Figure 10
> [![Figure 10 of the StreamingLLM paper: per-token latency and memory of sliding window with recomputation against StreamingLLM for Llama-2-7B and Llama-2-13B](/img/attention/papers/streaming-figure10.png)](/img/attention/papers/streaming-figure10.png)
>
> **Context:** Time per token (latency) and memory, for the slow-but-correct recomputation baseline (grey) and StreamingLLM (red), at cache sizes from 256 to 4,096 tokens.
>
> **What it says:** With a 4,096-token cache on Llama-2-13B, recomputation takes 2,355 ms per token; StreamingLLM takes 106 ms, "a remarkable speedup of up to 22.2× per token", with about the same memory.
>
> **Why it matters:** The practical use case: chat assistants and agents that run for hours, or read a live stream (logs, transcripts), without their memory or their speed getting worse over time.

**Use cases in one line each:**

- **Long documents** (contracts, books, codebases): local and global layers keep the cache affordable at 128K tokens.
- **Endless chats and live streams:** a rolling window plus attention sinks keeps memory fixed for as long as the stream runs.
- **Reasoning models** that write tens of thousands of tokens: sparse attention (DSA) keeps each new token cheap, however long the reasoning gets.
- **Retrieval inside long context** ("find the clause that mentions X"): this needs global layers or a learned selector, because a pure window cannot look that far directly.

## Which one to use?

| | Each token looks at | KV cache | Can look far back directly? | Used by |
|---|---|---|---|---|
| **Full attention** | all earlier tokens | grows with length | yes | most models, in at least some layers |
| **Sliding window** | the last W | fixed at W | no (only indirectly) | Mistral 7B |
| **Local + global layers** | last W in local layers, all in global | local layers fixed, global grow | yes, in global layers | Gemma 2, Gemma 3, gpt-oss |
| **Window + sinks** | first few + last W | fixed | no | StreamingLLM (inference trick) |
| **DSA (top-k)** | k tokens chosen by the indexer | all kept, but only k read per step | yes | DeepSeek-V3.2 |

Notice that DSA still **keeps** every token's latent in memory (any of them might be chosen), but reads only $$k$$ of them per step. Sliding windows save memory too; DSA saves mainly reading and computing.

All of these still use softmax attention over *some* set of tokens. Part 4 goes one step further: layers that replace softmax attention with a fixed-size **memory** that never grows at all (linear attention, Gated DeltaNet), and the hybrid models that mix them with ordinary attention.

## Summary

- Full attention compares every token with every earlier token: $$T(T+1)/2$$ comparisons per head per layer, **8.6 billion** at 128K tokens.
- **Sliding-window attention** adds one condition to the mask, $$0 \le i - j < W$$. I checked it against PyTorch ($$4.4 \times 10^{-16}$$) and showed that tokens outside the window have exactly zero effect.
- Stacked layers still see far: the reach is $$L(W-1)+1$$, measured exactly with gradients for 1 to 6 layers.
- A **rolling buffer** stores token $$i$$ in slot $$i \bmod W$$. It matched full computation to $$1.1 \times 10^{-15}$$, cut Mistral 7B's cache **8×** at 32K, and kept a decode step flat (0.031 ms vs 3.893 ms at 128K).
- **Local + global layers** keep some full-attention layers for direct long-range lookups. Gemma 3 27B: **62 GiB → 10.4 GiB** at 128K.
- A model trained with full attention **breaks** under a plain window (perplexity 17.4 → up to 509) because it loses its **attention sink**. Keeping 4 first tokens fixes most of it (17.80).
- **DeepSeek Sparse Attention** uses a tiny **lightning indexer**, $$I_{t,s} = \sum_j w^I_{t,j}\,\operatorname{ReLU}(q^I_{t,j} \cdot k^I_s)$$, to pick the top-k tokens, so the main attention costs $$O(Lk)$$ instead of $$O(L^2)$$.
- My indexer for Qwen2.5-0.5B, trained only with the warm-up KL loss, caught **77.6%** of the real attention with 64 tokens and brought perplexity to **17.62** with 256 tokens (full: 17.39).
- With DeepSeek-V3.2's real sizes, DSA scores a 128K-token text with about **7× fewer** multiply-adds than dense attention, and the paper's measured decode cost at 128K drops from about \$2.1 to \$0.25 per million tokens.

<details>
<summary>Run it yourself</summary>

- [`code/attention/part3_sparse.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part3_sparse.py): sliding-window checks, receptive field, rolling cache, KV memory, decode timing. Runs in seconds; the timing part uses a GPU if one is available.
- [`code/attention/part3_qwen.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/attention/part3_qwen.py): Qwen2.5-0.5B with windows and sinks, plus training and testing the 24 lightning indexers. Downloads the model (about 1 GB) and two public-domain books from Project Gutenberg. About 2 minutes on an Apple M5 Pro.

```bash
pip install torch transformers
python part3_sparse.py     # writes results/part3.json
python part3_qwen.py       # writes results/part3_qwen.json
```

</details>

## References

1. A. Q. Jiang et al. [*Mistral 7B*](https://arxiv.org/abs/2310.06825). 2023.
2. I. Beltagy, M. E. Peters, A. Cohan. [*Longformer: The Long-Document Transformer*](https://arxiv.org/abs/2004.05150). 2020.
3. R. Child, S. Gray, A. Radford, I. Sutskever. [*Generating Long Sequences with Sparse Transformers*](https://arxiv.org/abs/1904.10509). 2019.
4. G. Xiao, Y. Tian, B. Chen, S. Han, M. Lewis. [*Efficient Streaming Language Models with Attention Sinks*](https://arxiv.org/abs/2309.17453). ICLR 2024.
5. Gemma Team. [*Gemma 2: Improving Open Language Models at a Practical Size*](https://arxiv.org/abs/2408.00118). 2024.
6. Gemma Team. [*Gemma 3 Technical Report*](https://arxiv.org/abs/2503.19786). 2025.
7. DeepSeek-AI. [*DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models*](https://arxiv.org/abs/2512.02556). 2025.
8. Model configurations: [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1), [Gemma 3 27B](https://huggingface.co/google/gemma-3-27b-it), [Gemma 2 9B](https://huggingface.co/google/gemma-2-9b), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B).
9. Texts: [*Pride and Prejudice*](https://www.gutenberg.org/ebooks/1342) and [*The Adventures of Sherlock Holmes*](https://www.gutenberg.org/ebooks/1661), Project Gutenberg.
