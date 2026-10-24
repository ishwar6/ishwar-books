---
title: "The Big Idea: Memory, Not Arithmetic, Limits LLM Serving"
description: "How a chat server answers many users at once, why its GPU runs out of memory before it runs out of speed, and the paper's fix: store each request's memory in small pages. We read the title, the abstract, Figures 1 and 2 and §1, and we measure the KV cache of a real model."
part: 1
covers: "Title, Abstract, Figures 1 and 2, §1"
date: 2026-10-19
tags: [llm-serving, inference, kv-cache, gpu-memory, systems]
---

When many people use a chatbot at once, one GPU answers many of them together. How many it can answer at once depends on memory, and this paper shows that most of that memory was wasted. Its fix comes from the way an operating system hands out memory to programs.

Each piece below shows the exact lines of the paper, then explains them.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

## The title {§Title}

> [!PAPER] · Title
> [![The title of the paper: Efficient Memory Management for Large Language Model Serving with PagedAttention, with the author names and their universities](/img/papers/vllm/p1-title.png)](/img/papers/vllm/p1-title.png)
>
> **What it says:** "Efficient Memory Management for Large Language Model Serving with PagedAttention."
>
> **What it means:**
>
> Most papers about language models are about the model: a new layer, more data, a better score. This title has no new model in it. It is about the computer that runs a model after training, and about one resource of that computer: memory. The paper does not change a single weight. Every model gives exactly the same answers before and after.
>
> "Serving" is the job of answering requests from users, all day, for many users at once. "Memory management" is the job of deciding which piece of memory each request gets, and when it gives the memory back. "PagedAttention" is the paper's new idea. The word "paged" comes from operating systems, which have cut memory into small pages since the 1960s. The rest of the paper shows why that old idea fits a new problem so well.

> [!DEFINITION] Large language model (LLM)
> A neural network that reads text and writes the next words. Its knowledge sits in billions of numbers, its **weights**. In this paper the weights never change; only the way the computer stores the model's working memory changes. The [Background page](background-the-kv-cache-and-paging.md) shows how such a model writes, one token at a time.

> [!DEFINITION] LLM serving
> Running a trained model as a service: users send **requests** (a prompt), and the system sends back the model's answer. A serving system decides which requests run together, where their data sits in memory, and when each one finishes.

## How LLM serving works

The abstract assumes that you know how a chat server works. So we first follow one request from start to end, then add many users, then look at the parts of vLLM.

### One chat request

You type "What is the capital of France?" and press Enter. The text goes to a server with a GPU, and the server cuts it into tokens. Say the prompt is 7 tokens. The model reads all 7 in one pass and writes the first token of the answer, "The". This first pass is the **prompt phase**.

Then the model writes one token per pass: "capital", then "is", then "Paris", then ".". This is the **generation phase**. The server sends each new token back at once, which is why a chatbot's answer appears word by word. When the model writes a special end token, the request is done.

During all of this, the model keeps a record of every token it has seen: the **KV cache**. After the prompt phase it holds 7 slots, one per prompt token. Each step of the generation phase adds one slot. When the request ends, the server frees the memory for the next request.

<figure class="fig"><svg viewBox="0 0 720 330" role="img" aria-label="A timeline of one request. It arrives, waits, then the prompt phase reads all 7 prompt tokens in one pass and writes the first answer token. Each later step writes one more token. The KV cache grows from 7 to 12 slots, one per step, and is freed when the end token appears."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The life of one chat request</text><text class="t-tick" x="151.0" y="58.0" text-anchor="middle">arrives</text><text class="t-tick" x="197.0" y="58.0" text-anchor="middle">waits</text><text class="t-tick" x="243.0" y="58.0" text-anchor="middle">prompt</text><text class="t-tick" x="289.0" y="58.0" text-anchor="middle">step 1</text><text class="t-tick" x="335.0" y="58.0" text-anchor="middle">step 2</text><text class="t-tick" x="381.0" y="58.0" text-anchor="middle">step 3</text><text class="t-tick" x="427.0" y="58.0" text-anchor="middle">step 4</text><text class="t-tick" x="473.0" y="58.0" text-anchor="middle">step 5</text><text class="t-tick" x="519.0" y="58.0" text-anchor="middle">done</text><text class="t-tick" x="118.0" y="90.0" text-anchor="end">model output</text><rect class="box-2" x="222.0" y="74.0" width="42.0" height="26.0" rx="6"/><text class="t-tick" x="243.0" y="91.5" text-anchor="middle">The</text><rect class="box-2" x="268.0" y="74.0" width="42.0" height="26.0" rx="6"/><text class="t-tick" x="289.0" y="91.5" text-anchor="middle">capital</text><rect class="box-2" x="314.0" y="74.0" width="42.0" height="26.0" rx="6"/><text class="t-tick" x="335.0" y="91.5" text-anchor="middle">is</text><rect class="box-2" x="360.0" y="74.0" width="42.0" height="26.0" rx="6"/><text class="t-tick" x="381.0" y="91.5" text-anchor="middle">Paris</text><rect class="box-2" x="406.0" y="74.0" width="42.0" height="26.0" rx="6"/><text class="t-tick" x="427.0" y="91.5" text-anchor="middle">.</text><rect class="box-2" x="452.0" y="74.0" width="42.0" height="26.0" rx="6"/><text class="t-tick" x="473.0" y="91.5" text-anchor="middle">&lt;end&gt;</text><text class="t-tick" x="118.0" y="122.0" text-anchor="end">GPU passes</text><rect class="box" x="222.0" y="106.0" width="42.0" height="24.0" rx="6"/><text class="t-tick" x="243.0" y="122.5" text-anchor="middle">1 pass</text><rect class="box" x="268.0" y="106.0" width="42.0" height="24.0" rx="6"/><text class="t-tick" x="289.0" y="122.5" text-anchor="middle">1 pass</text><rect class="box" x="314.0" y="106.0" width="42.0" height="24.0" rx="6"/><text class="t-tick" x="335.0" y="122.5" text-anchor="middle">1 pass</text><rect class="box" x="360.0" y="106.0" width="42.0" height="24.0" rx="6"/><text class="t-tick" x="381.0" y="122.5" text-anchor="middle">1 pass</text><rect class="box" x="406.0" y="106.0" width="42.0" height="24.0" rx="6"/><text class="t-tick" x="427.0" y="122.5" text-anchor="middle">1 pass</text><rect class="box" x="452.0" y="106.0" width="42.0" height="24.0" rx="6"/><text class="t-tick" x="473.0" y="122.5" text-anchor="middle">1 pass</text><text class="t-tick" x="118.0" y="170.0" text-anchor="end">KV cache</text><text class="t-tick" x="118.0" y="186.0" text-anchor="end">held (slots)</text><text class="t-tick" x="151.0" y="296.0" text-anchor="middle">0</text><text class="t-tick" x="197.0" y="296.0" text-anchor="middle">0</text><rect class="box-1" x="232.0" y="270.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="232.0" y="260.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="232.0" y="250.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="232.0" y="240.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="232.0" y="230.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="232.0" y="220.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="232.0" y="210.0" width="22.0" height="9.0" rx="2"/><text class="t-tick" x="243.0" y="296.0" text-anchor="middle">7</text><rect class="box-1" x="278.0" y="270.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="278.0" y="260.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="278.0" y="250.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="278.0" y="240.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="278.0" y="230.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="278.0" y="220.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="278.0" y="210.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="278.0" y="200.0" width="22.0" height="9.0" rx="2"/><text class="t-tick" x="289.0" y="296.0" text-anchor="middle">8</text><rect class="box-1" x="324.0" y="270.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="324.0" y="260.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="324.0" y="250.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="324.0" y="240.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="324.0" y="230.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="324.0" y="220.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="324.0" y="210.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="324.0" y="200.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="324.0" y="190.0" width="22.0" height="9.0" rx="2"/><text class="t-tick" x="335.0" y="296.0" text-anchor="middle">9</text><rect class="box-1" x="370.0" y="270.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="370.0" y="260.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="370.0" y="250.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="370.0" y="240.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="370.0" y="230.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="370.0" y="220.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="370.0" y="210.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="370.0" y="200.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="370.0" y="190.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="370.0" y="180.0" width="22.0" height="9.0" rx="2"/><text class="t-tick" x="381.0" y="296.0" text-anchor="middle">10</text><rect class="box-1" x="416.0" y="270.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="416.0" y="260.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="416.0" y="250.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="416.0" y="240.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="416.0" y="230.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="416.0" y="220.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="416.0" y="210.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="416.0" y="200.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="416.0" y="190.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="416.0" y="180.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="416.0" y="170.0" width="22.0" height="9.0" rx="2"/><text class="t-tick" x="427.0" y="296.0" text-anchor="middle">11</text><rect class="box-1" x="462.0" y="270.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="462.0" y="260.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="462.0" y="250.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="462.0" y="240.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="462.0" y="230.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="462.0" y="220.0" width="22.0" height="9.0" rx="2"/><rect class="box-1" x="462.0" y="210.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="462.0" y="200.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="462.0" y="190.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="462.0" y="180.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="462.0" y="170.0" width="22.0" height="9.0" rx="2"/><rect class="box-2" x="462.0" y="160.0" width="22.0" height="9.0" rx="2"/><text class="t-tick" x="473.0" y="296.0" text-anchor="middle">12</text><text class="t-tick" x="519.0" y="296.0" text-anchor="middle">freed</text><line class="axis" x1="128" y1="281" x2="542" y2="281"/><rect class="box-1" x="560" y="150" width="12" height="12" rx="2"/><text class="t-tick" x="578.0" y="160.0" text-anchor="start">prompt tokens</text><rect class="box-2" x="560" y="172" width="12" height="12" rx="2"/><text class="t-tick" x="578.0" y="182.0" text-anchor="start">answer tokens</text><text class="t-tick" x="560.0" y="214.0" text-anchor="start">The cache grows by</text><text class="t-tick" x="560.0" y="230.0" text-anchor="start">one slot per step.</text><text class="t-tick" x="560.0" y="246.0" text-anchor="start">Nobody knows the</text><text class="t-tick" x="560.0" y="262.0" text-anchor="start">final size in advance.</text><text class="t-tick" x="20.0" y="316.0" text-anchor="start">Prompt: "What is the capital of France?" (7 tokens in this sketch). The answer streams back one token per step.</text></svg><figcaption>One chat request over time. The prompt phase reads 7 prompt tokens in one pass and writes the first answer token. Each later pass writes one more token, and the KV cache grows by one slot per step until the end token frees it.</figcaption></figure>

> [!DEFINITION] KV cache
> The keys and values that the model computed for every earlier token of a request. They stay in GPU memory, so the model does not compute them again for each new token. "KV" stands for key and value, two vectors from the attention layers. The sections below show what they are and how large they get.

> [!DEFINITION] Prompt phase and generation phase
> The two phases of one request. In the **prompt phase**, the model reads all prompt tokens in one pass, because they are all known. In the **generation phase**, it writes one new token per pass, because each token depends on the one before it.

Look at the bottom row of the picture. The cache grows by one slot at every step, and nobody knows when it will stop. The answer could be 5 tokens or 500. The whole paper is about this growing, unpredictable piece of memory.

### Many users at once

A server does not answer one user and then the next. Many requests run at the same time, and the server moves them forward together. At every step, it takes the requests that are running and runs one pass of the model for all of them. That group is a **batch**.

Take four requests, A to D, that arrive at different times. In step 1 only A runs. In step 2, B arrives and joins, so the batch holds 2 requests. A ends after step 5 and leaves. D arrives later and joins in step 6. So the batch changes at every step:

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="A grid of four requests over eight steps. Request A runs in steps 1 to 5, B in steps 2 to 7, C in steps 4 and 5, D in steps 6 to 8. The first cell of each is its prompt phase. The batch size per step is 1, 2, 2, 3, 3, 2, 2, 1."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Many users at once: each column is one pass of the model</text><text class="t-tick" x="142.0" y="54.0" text-anchor="middle">step 1</text><text class="t-tick" x="206.0" y="54.0" text-anchor="middle">step 2</text><text class="t-tick" x="270.0" y="54.0" text-anchor="middle">step 3</text><text class="t-tick" x="334.0" y="54.0" text-anchor="middle">step 4</text><text class="t-tick" x="398.0" y="54.0" text-anchor="middle">step 5</text><text class="t-tick" x="462.0" y="54.0" text-anchor="middle">step 6</text><text class="t-tick" x="526.0" y="54.0" text-anchor="middle">step 7</text><text class="t-tick" x="590.0" y="54.0" text-anchor="middle">step 8</text><text class="t-tick t-strong" x="98.0" y="86.0" text-anchor="end">request A</text><rect class="box-1" x="113.0" y="68.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="142.0" y="86.5" text-anchor="middle">prompt</text><rect class="box-2" x="177.0" y="68.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="206.0" y="86.5" text-anchor="middle">token</text><rect class="box-2" x="241.0" y="68.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="270.0" y="86.5" text-anchor="middle">token</text><rect class="box-2" x="305.0" y="68.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="334.0" y="86.5" text-anchor="middle">token</text><rect class="box-2" x="369.0" y="68.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="398.0" y="86.5" text-anchor="middle">token</text><text class="t-muted" x="438.0" y="86.0" text-anchor="start">done</text><text class="t-tick t-strong" x="98.0" y="122.0" text-anchor="end">request B</text><rect class="box-1" x="177.0" y="104.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="206.0" y="122.5" text-anchor="middle">prompt</text><rect class="box-2" x="241.0" y="104.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="270.0" y="122.5" text-anchor="middle">token</text><rect class="box-2" x="305.0" y="104.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="334.0" y="122.5" text-anchor="middle">token</text><rect class="box-2" x="369.0" y="104.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="398.0" y="122.5" text-anchor="middle">token</text><rect class="box-2" x="433.0" y="104.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="462.0" y="122.5" text-anchor="middle">token</text><rect class="box-2" x="497.0" y="104.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="526.0" y="122.5" text-anchor="middle">token</text><text class="t-muted" x="566.0" y="122.0" text-anchor="start">done</text><text class="t-tick t-strong" x="98.0" y="158.0" text-anchor="end">request C</text><rect class="box-1" x="305.0" y="140.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="334.0" y="158.5" text-anchor="middle">prompt</text><rect class="box-2" x="369.0" y="140.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="398.0" y="158.5" text-anchor="middle">token</text><text class="t-muted" x="438.0" y="158.0" text-anchor="start">done</text><text class="t-tick t-strong" x="98.0" y="194.0" text-anchor="end">request D</text><rect class="box-1" x="433.0" y="176.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="462.0" y="194.5" text-anchor="middle">prompt</text><rect class="box-2" x="497.0" y="176.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="526.0" y="194.5" text-anchor="middle">token</text><rect class="box-2" x="561.0" y="176.0" width="58.0" height="28.0" rx="6"/><text class="t-tick" x="590.0" y="194.5" text-anchor="middle">token</text><text class="t-muted" x="630.0" y="194.0" text-anchor="start">done</text><text class="t-tick t-strong" x="98.0" y="234.0" text-anchor="end">batch size</text><text class="t-val" x="142.0" y="234.0" text-anchor="middle">1</text><text class="t-val" x="206.0" y="234.0" text-anchor="middle">2</text><text class="t-val" x="270.0" y="234.0" text-anchor="middle">2</text><text class="t-val" x="334.0" y="234.0" text-anchor="middle">3</text><text class="t-val" x="398.0" y="234.0" text-anchor="middle">3</text><text class="t-val" x="462.0" y="234.0" text-anchor="middle">2</text><text class="t-val" x="526.0" y="234.0" text-anchor="middle">2</text><text class="t-val" x="590.0" y="234.0" text-anchor="middle">1</text><text class="t-tick" x="20.0" y="270.0" text-anchor="start">All requests in one column share one read of the weights. A request joins when it arrives</text><text class="t-tick" x="20.0" y="288.0" text-anchor="start">and leaves when it writes its end token, so the batch changes at every step.</text></svg><figcaption>Four requests over eight passes of the model. Each request starts with its prompt phase, then writes one token per pass. Requests join when they arrive and leave when they finish, so the batch size changes from step to step.</figcaption></figure>

> [!NOTE] Where this idea comes from: batching at every step
> Batching is older than language models. In training, a GPU always processes many images or sentences in one pass. Text generation adds a problem: requests end at different times. Earlier servers formed a batch and kept it until every request in it was done. A short request finished early and its place stayed idle, while new requests waited outside.
>
> Orca (OSDI 2022) changed this. It schedules the batch again after every pass of the model: finished requests leave, and waiting requests join. The paper calls this **iteration-level scheduling**, and vLLM uses it too. Think of a lift that stops at every floor, against a bus that only opens its doors at the last stop.

> [!DEFINITION] Batch and batch size
> A **batch** is a group of requests that the GPU processes in the same pass. The **batch size** is how many requests are in it. A larger batch uses the GPU better, until memory runs out.

> [!DEFINITION] Throughput
> How much work a system finishes per second. Here it is requests or tokens per second. Higher throughput means each request costs less.

Now connect the two pictures. Every request in the batch holds its own KV cache in GPU memory, for as long as it runs. A request can join the batch only if its cache fits. So memory sets the batch size, and the batch size sets the throughput. The rest of this part makes that chain exact.

### The map of vLLM

Here is the whole system that the paper builds, drawn from its Figure 4. Nothing on it is new yet. It is a map for the next parts, so you can see where each piece fits.

<figure class="fig"><svg viewBox="0 0 720 380" role="img" aria-label="The architecture of vLLM. On the left, requests from users. In the centre, the scheduler, which contains the KV cache manager with block tables, a GPU block allocator and a CPU block allocator. On the right, workers 0 to N-1, each with a cache engine and a shard of the model. New tokens flow back."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The parts of vLLM (a redraw of the paper's Figure 4)</text><text class="t-tick t-strong" x="20.0" y="62.0" text-anchor="start">Requests</text><rect class="box" x="20.0" y="76.0" width="120.0" height="34.0" rx="8"/><text class="t-tick" x="80.0" y="97.5" text-anchor="middle">user 1: prompt</text><rect class="box" x="20.0" y="126.0" width="120.0" height="34.0" rx="8"/><text class="t-tick" x="80.0" y="147.5" text-anchor="middle">user 2: prompt</text><rect class="box" x="20.0" y="176.0" width="120.0" height="34.0" rx="8"/><text class="t-tick" x="80.0" y="197.5" text-anchor="middle">user 3: prompt</text><text class="t-tick" x="80.0" y="238.0" text-anchor="middle">...</text><rect class="box" x="170.0" y="48.0" width="250.0" height="236.0" rx="10"/><text class="t-title" x="184.0" y="72.0" text-anchor="start">Scheduler</text><text class="t-tick" x="184.0" y="90.0" text-anchor="start">picks the requests of the next step</text><rect class="box-ghost" x="184.0" y="104.0" width="222.0" height="166.0" rx="8"/><text class="t-tick t-strong" x="196.0" y="124.0" text-anchor="start">KV cache manager</text><rect class="box" x="196.0" y="136.0" width="198.0" height="34.0" rx="8"/><text class="t-tick" x="295.0" y="157.5" text-anchor="middle">Block tables</text><rect class="box" x="196.0" y="180.0" width="198.0" height="34.0" rx="8"/><text class="t-tick" x="295.0" y="201.5" text-anchor="middle">GPU block allocator</text><rect class="box" x="196.0" y="224.0" width="198.0" height="34.0" rx="8"/><text class="t-tick" x="295.0" y="245.5" text-anchor="middle">CPU block allocator</text><rect class="box" x="504.0" y="48.0" width="196.0" height="100.0" rx="10"/><text class="t-tick t-strong" x="516.0" y="70.0" text-anchor="start">Worker 0</text><rect class="box" x="516.0" y="80.0" width="172.0" height="26.0" rx="8"/><text class="t-tick" x="602.0" y="97.5" text-anchor="middle">Cache engine</text><rect class="box" x="516.0" y="112.0" width="172.0" height="26.0" rx="8"/><text class="t-tick" x="602.0" y="129.5" text-anchor="middle">Model shard</text><rect class="box" x="504.0" y="184.0" width="196.0" height="100.0" rx="10"/><text class="t-tick t-strong" x="516.0" y="206.0" text-anchor="start">Worker N-1</text><rect class="box" x="516.0" y="216.0" width="172.0" height="26.0" rx="8"/><text class="t-tick" x="602.0" y="233.5" text-anchor="middle">Cache engine</text><rect class="box" x="516.0" y="248.0" width="172.0" height="26.0" rx="8"/><text class="t-tick" x="602.0" y="265.5" text-anchor="middle">Model shard</text><text class="t-tick" x="602.0" y="172.0" text-anchor="middle">...</text><line class="edge" x1="144.0" y1="166.0" x2="164.0" y2="166.0" marker-end="url(#ah)"/><line class="edge" x1="422.0" y1="120.0" x2="498.0" y2="98.0" marker-end="url(#ah)"/><line class="edge" x1="422.0" y1="212.0" x2="498.0" y2="234.0" marker-end="url(#ah)"/><text class="t-tick" x="462.0" y="150.0" text-anchor="middle">one step:</text><text class="t-tick" x="462.0" y="166.0" text-anchor="middle">batch and</text><text class="t-tick" x="462.0" y="182.0" text-anchor="middle">block tables</text><path class="edge-dim" d="M602.0,290 V306 H80 V258" marker-end="url(#ah)"/><text class="t-tick" x="300.0" y="322.0" text-anchor="middle">new tokens go back to the users, and to the scheduler for the next step</text><text class="t-tick" x="20.0" y="352.0" text-anchor="start">Part 3: the block tables, the allocators and the attention kernel.</text><text class="t-tick" x="20.0" y="370.0" text-anchor="start">Part 4: sharing, the scheduler, the CPU swap and many GPUs.</text></svg><figcaption>The parts of vLLM. Requests enter the scheduler. Inside it, the KV cache manager keeps a block table per request and two allocators. Workers run the model on the GPUs. New tokens go back to the users and to the next step.</figcaption></figure>

Requests from users arrive on the left. In the centre, the **scheduler** decides which requests run in the next step, as in the grid above. Inside it, the **KV cache manager** decides where in memory each request's cache lives. It keeps a **block table** for each request. It also has two allocators: one hands out pieces of GPU memory, and one hands out CPU memory as an overflow area.

On the right are the **workers**. Each worker drives one GPU. Its **cache engine** holds the KV cache memory on that GPU, and its **model shard** holds its part of the weights. After each step, the new tokens go back to the users and to the scheduler.

This part explains why a KV cache manager is needed at all. Part 3 opens the block tables, the allocators and the attention kernel. Part 4 explains sharing, the scheduler's choices when memory runs out, and how the workers split a model across GPUs.

> [!DEFINITION] Worker and model shard
> A **worker** is the program that controls one GPU and runs the model on it. When a model is too large for one GPU, its weights are split into pieces, and each GPU holds one piece: a **shard**. With one GPU, there is one worker and the shard is the whole model.

## The abstract {§Abstract}

> [!PAPER] · Abstract
> [![The first sentences of the abstract, with requires batching sufficiently many requests at a time, is huge and grows and shrinks dynamically, and fragmentation and redundant duplication, limiting the batch size highlighted](/img/papers/vllm/p1-abstract-1.png)](/img/papers/vllm/p1-abstract-1.png)
>
> **What it says:** high throughput "requires batching sufficiently many requests at a time". But the key-value cache of each request "is huge and grows and shrinks dynamically", and when it is managed badly, memory is wasted by "fragmentation and redundant duplication, limiting the batch size".
>
> **What it means:**
>
> Think of a bus. Driving the bus costs about the same with one passenger or with forty. A GPU that serves a language model is the same. One step of the model costs almost the same for one request or for many. So the system wants to run many requests together, in one batch.
>
> The seats on this bus are memory. Each request needs space for its KV cache. The cache is large, and it grows with every token. Nobody knows in advance how long the answer will be. If the system books seats badly, the bus leaves half empty, and fewer requests run together.
>
> The abstract names two ways to book seats badly. **Fragmentation** means free memory that is cut into pieces nobody can use. **Redundant duplication** means two copies of the same data, where one copy shared by both would do.

> [!PAPER] · Abstract
> [![The middle of the abstract, with inspired by the classical virtual memory and paging techniques in operating systems, near-zero waste in KV cache memory, and flexible sharing of KV cache within and across requests highlighted](/img/papers/vllm/p1-abstract-2.png)](/img/papers/vllm/p1-abstract-2.png)
>
> **What it says:** the paper proposes PagedAttention, an attention algorithm "inspired by the classical virtual memory and paging techniques in operating systems". On top of it, vLLM achieves "near-zero waste in KV cache memory" and "flexible sharing of KV cache within and across requests".
>
> **What it means:**
>
> Your laptop runs a browser, an editor and a music player at the same time. Each program thinks it owns one long, unbroken stretch of memory. In fact, the operating system has cut the real memory into small pages of equal size and scattered each program's pages wherever there was room. A table per program records where each page really is. This trick is **virtual memory with paging**, and it is the reason a program almost never wastes more than part of one page.
>
> PagedAttention does the same with the KV cache. The cache of a request is cut into small **blocks** of a fixed number of tokens. The blocks can sit anywhere in GPU memory, and a table per request records where. A request gets a new block only when it needs one. So the waste per request is at most the unused part of its last block: this is "near-zero waste".
>
> Blocks also make sharing easy. If two answers start from the same prompt, both tables can point to the same blocks of that prompt. One copy serves both. That is the "sharing within and across requests".

> [!DEFINITION] Virtual memory and paging
> A method of operating systems. Memory is cut into **pages** of a fixed size, often 4 KB. Each program sees its own unbroken address space, and a **page table** maps each of its pages to a real place in memory. Pages that sit next to each other for the program can be far apart in reality. The [Background page](background-the-kv-cache-and-paging.md) builds it step by step.

> [!PAPER] · Abstract
> [![The end of the abstract, with by 2-4× with the same level of latency, and longer sequences, larger models, and more complex decoding algorithms highlighted](/img/papers/vllm/p1-abstract-3.png)](/img/papers/vllm/p1-abstract-3.png)
>
> **What it says:** vLLM improves the throughput of popular LLMs "by 2-4× with the same level of latency" compared with FasterTransformer and Orca. The gain is larger with "longer sequences, larger models, and more complex decoding algorithms".
>
> **What it means:**
>
> Throughput alone is easy to raise: make every user wait longer, and the GPU can collect huge batches. The claim is fair only because of the second half, "with the same level of latency". Each user waits about as long as before, and the same GPU serves two to four times as many of them. For a company that pays for GPUs by the hour, that is a cost cut of half or more.
>
> The three conditions tell you where the gain comes from. Longer sequences mean larger caches, so wasted memory hurts more. Larger models leave less room for the cache. Complex decoding, such as several answers to one prompt, gives more chances to share. In each case memory is the tighter limit, so better memory management helps more.

> [!DEFINITION] Latency
> How long one user waits for an answer. The paper measures **normalized latency**: the time of a request divided by the number of tokens in its answer, in seconds per token. Part 5 reads those curves.

> [!DEFINITION] FasterTransformer and Orca
> The two strongest serving systems of 2022. **FasterTransformer** is NVIDIA's fast inference library. **Orca** (OSDI 2022) added a better way to form batches. Orca's code was not public, so the authors built their own version, in three variants that reserve memory in different ways. Part 5 describes them.

> [!NOTE] How serving systems are compared
> A fair test works like a shop on a busy day. Requests arrive at a fixed rate, say a few per second, and the test measures the normalized latency of each one. At a low rate, latency stays flat, because the server keeps up. At some rate the server falls behind, a queue forms, and latency shoots up.
>
> The highest rate a system can serve before that jump is its throughput. So "2-4× with the same level of latency" means that vLLM's jump comes at a rate two to four times higher. Part 5 reads these curves for every model and dataset of the paper.

## Why serving is expensive {§1}

> [!PAPER] · §1
> [![The first paragraph of the introduction, with very expensive, 10× more expensive than a traditional keyword query, and increasing the throughput highlighted](/img/papers/vllm/p1-intro-cost.png)](/img/papers/vllm/p1-intro-cost.png)
>
> **What it says:** running LLM applications "is very expensive", and one LLM request can be "10× more expensive than a traditional keyword query". So "increasing the throughput", and with it reducing the cost per request, matters more and more.
>
> **What it means:**
>
> A web search looks up an index that was built in advance. An LLM request runs billions of multiplications for every word of the answer, on hardware that costs thousands of dollars per card. The 10× estimate comes from a news report that the paper cites, not from the authors' own measurement.
>
> The paper's goal follows from this. You cannot make the model smaller without changing its answers. You can serve more requests per GPU. Every gain in throughput is a direct cut in the cost of each request.

## Figure 1: where the memory goes {§Figure 1}

> [!PAPER] · Figure 1
> [![Figure 1 of the paper: left, the memory of a 40 GB A100 split into parameters (26 GB, 65%), KV cache (more than 30%) and others; right, memory usage and throughput against batch size for existing systems and vLLM](/img/papers/vllm/p1-figure1.png)](/img/papers/vllm/p1-figure1.png)
>
> **What it says:** the parameters of a 13B model "persist in GPU memory throughout serving". The memory for the KV cache "is (de)allocated per serving request". On the right, vLLM "smooths out the rapid growth" of KV cache memory, "leading to a notable boost in serving throughput".
>
> **What it means:**
>
> The left picture is a budget. A GPU with 40 GB holds a model of 13 billion parameters. The parameters take 26 GB, 65% of the card, and they never move. A small part, "Others", holds short-lived data of the computation. Everything else, a bit over 30%, is for the KV cache of the requests. That slice is the only part a serving system can manage, so it decides how many requests run together.
>
> The right picture has two panels that share one x-axis, the batch size. The top panel shows memory. With existing systems (orange), memory climbs steeply and hits the 40 GB ceiling at a batch of about 8 requests. With vLLM (blue), each request takes much less memory, so the line climbs slowly and reaches the ceiling at about 40. The bottom panel shows what that buys. Throughput rises with the batch size, so the system that fits 40 requests gets far more tokens per second than the one that fits 8.
>
> The figure states the whole paper in one picture. The model and the GPU are the same. Only the memory per request changes, and throughput follows it.

> [!DEFINITION] GPU memory
> The memory on the graphics card, separate from the computer's main memory. The NVIDIA A100 in the paper has 40 GB or 80 GB. The model weights and the KV cache must both sit in it during serving, because reading from main memory is far too slow.

> [!DEFINITION] FP16
> A way to store each number in 16 bits, which is 2 bytes. Models are usually served in FP16, so a model of 13 billion parameters needs 13 billion times 2 bytes, which is 26 GB.

## Why one token at a time {§1}

> [!PAPER] · §1
> [![The paragraph about autoregressive generation, with one at a time, repeated until the model outputs a termination token, and memory-bound, underutilizing the computation power of GPUs highlighted](/img/papers/vllm/p1-intro-autoregressive.png)](/img/papers/vllm/p1-intro-autoregressive.png)
>
> **What it says:** the model "generates words (tokens), one at a time", based on the prompt and the tokens it generated so far. The process is "repeated until the model outputs a termination token". This makes the workload "memory-bound, underutilizing the computation power of GPUs".
>
> **What it means:**
>
> A language model cannot write a whole answer at once. It writes one token, adds it to the text, and runs again to write the next one. An answer of 300 tokens needs 300 runs of the model, one after another. Each run needs the one before it, so they cannot happen at the same time.
>
> Each run is small work for a GPU: one new token per request. But to do it, the GPU must read all 26 GB of weights from its memory. A GPU can multiply much faster than it can read memory. So most of the time, its arithmetic units sit idle and wait for data. That is what **memory-bound** means. The GPU is not too slow at maths; it is starved of data.
>
> Batching is the cure. If 40 requests share one run, the weights are read once and used 40 times. The cost of the read is split 40 ways.

> [!DEFINITION] Autoregressive generation
> Writing text one token at a time, where each new token depends on all tokens before it. "Auto" (self) and "regressive" (predicting from earlier values): the model feeds its own output back in as input.

> [!DEFINITION] Memory-bound
> A computation whose speed is set by how fast data moves from memory, not by how fast the processor computes. The opposite is **compute-bound**. Writing one token for one request is memory-bound on a GPU.

<figure class="fig"><svg viewBox="0 0 720 270" role="img" aria-label="A diagram: the weights in GPU memory are read once per step by the arithmetic units. A batch of 1 takes 4.51 ms and writes 1 token; a batch of 64 takes 10.01 ms and writes 64 tokens."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">One step reads all the weights, whatever the batch size</text><rect class="box" x="20.0" y="60.0" width="150.0" height="120.0" rx="8"/><text class="t-tick" x="95.0" y="124.5" text-anchor="middle"></text><text class="t-tick t-strong" x="95.0" y="100.0" text-anchor="middle">GPU memory</text><text class="t-tick" x="95.0" y="122.0" text-anchor="middle">weights:</text><text class="t-tick" x="95.0" y="140.0" text-anchor="middle">every number is</text><text class="t-tick" x="95.0" y="158.0" text-anchor="middle">read once per step</text><rect class="box" x="240.0" y="60.0" width="150.0" height="120.0" rx="8"/><text class="t-tick" x="315.0" y="124.5" text-anchor="middle"></text><text class="t-tick t-strong" x="315.0" y="110.0" text-anchor="middle">arithmetic</text><text class="t-tick t-strong" x="315.0" y="130.0" text-anchor="middle">units</text><line class="edge" x1="172.0" y1="120.0" x2="236.0" y2="120.0" marker-end="url(#ah)"/><text class="t-tick" x="204.0" y="110.0" text-anchor="middle">read</text><line class="edge" x1="392.0" y1="90.0" x2="440.0" y2="80.0" marker-end="url(#ah)"/><line class="edge" x1="392.0" y1="150.0" x2="440.0" y2="160.0" marker-end="url(#ah)"/><text class="t-tick t-strong" x="450.0" y="66.0" text-anchor="start">batch of 1: 4.51 ms</text><rect class="box-2" x="450.0" y="74.0" width="20.0" height="16.0" rx="6"/><text class="t-tick" x="460.0" y="86.5" text-anchor="middle"></text><text class="t-tick" x="478.0" y="87.0" text-anchor="start">1 token per step</text><text class="t-tick t-strong" x="450.0" y="132.0" text-anchor="start">batch of 64: 10.01 ms</text><rect class="box-2" x="450.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="463.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="476.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="489.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="502.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="515.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="528.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="541.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="554.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="567.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="580.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="593.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="606.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="619.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="632.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="645.0" y="140.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="450.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="463.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="476.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="489.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="502.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="515.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="528.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="541.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="554.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="567.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="580.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="593.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="606.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="619.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="632.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="645.0" y="153.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="450.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="463.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="476.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="489.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="502.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="515.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="528.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="541.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="554.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="567.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="580.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="593.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="606.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="619.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="632.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="645.0" y="166.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="450.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="463.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="476.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="489.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="502.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="515.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="528.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="541.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="554.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="567.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="580.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="593.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="606.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="619.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="632.0" y="179.0" width="10.0" height="10.0" rx="2"/><rect class="box-2" x="645.0" y="179.0" width="10.0" height="10.0" rx="2"/><text class="t-tick" x="450.0" y="208.0" text-anchor="start">64 tokens per step</text><text class="t-tick" x="20.0" y="236.0" text-anchor="start">The read costs the same for 1 request or 64, so a larger batch spreads it over more tokens.</text><text class="t-tick" x="20.0" y="254.0" text-anchor="start">Times are our run of OPT-125M on a laptop GPU (one decode step, 128 tokens of context).</text></svg><figcaption>Why serving is memory-bound. Each step reads every weight once, whatever the batch size. In our run, a batch of 1 takes 4.51 ms for 1 token, and a batch of 64 takes 10.01 ms for 64 tokens.</figcaption></figure>

> [!NOTE] Where this idea comes from: the memory wall
> For decades, processors got faster much more quickly than memory did. In 1995, the computer architects William Wulf and Sally McKee called this gap the "memory wall". A GPU makes the gap extreme: thousands of arithmetic units wait on one memory.
>
> Think of a chef who can chop a carrot every second, with a helper who brings one carrot every ten seconds. The chef's speed does not matter; the helper's does. When a model writes one token for one request, it does about one multiply and one add for each weight it reads. With a batch of 64, it does 64 of them for the same read. That is why batching helps so much, and why the next limit is memory space.

### Why the KV cache exists, on a laptop

Before we measure the cache, we check why it exists. The attention layers of the model compare the newest token with every earlier token. For that, each earlier token needs two vectors, its **key** and its **value**. Without a cache, the model would compute the keys and values of the whole text again at every step. With a cache, it computes them once and keeps them.

We use OPT-125M, the smallest model of the OPT family that the paper serves, in FP16 on an Apple laptop GPU. We write greedy generation three ways: with no cache, with our own loop that keeps a cache, and with the library's `model.generate`. All three must give the same tokens. In the output, look for two things: the word True on both checks, and the times.

```python
@torch.no_grad()
def generate_no_cache(ids, n_new):
    """Recompute the keys and values of every earlier token at every step."""
    for _ in range(n_new):
        logits = model(ids, use_cache=False).logits
        ids = torch.cat([ids, logits[:, -1:].argmax(-1)], dim=1)
    return ids


@torch.no_grad()
def generate_with_cache(ids, n_new):
    """The prompt phase once, then one new token per step; the cache keeps every earlier key and value."""
    out = model(ids, use_cache=True)                       # prompt phase: all prompt tokens in one pass
    cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
    toks = [nxt]
    for _ in range(n_new - 1):                             # generation phase: one token in, one token out
        out = model(nxt, past_key_values=cache, use_cache=True)
        cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        toks.append(nxt)
    return torch.cat([ids] + toks, dim=1), cache
```

The first function feeds the whole text into the model at every step. The second feeds the prompt once, then only the newest token, and passes the cache back in. The model reads the old keys and values from the cache instead of computing them again:

```text
== 3. Greedy generation three ways: no cache, our own KV cache loop, model.generate
64 new tokens; same tokens, no cache vs our cache: True; our cache vs generate: True
text: ', a new nation, a new nation, a new nation, a new nation, a new nation, a new nation, a ne' ...
cache after generation: 80 tokens = 2,949,120 bytes (the last token gets its key and value at the next step)
time: no cache 0.308 s, with cache 0.260 s (1.2x)
with a 400-token prompt: no cache 0.633 s, with cache 0.270 s (2.3x)
```

The three methods give the same 64 tokens, so the cache changes nothing about the answer. (A model of 125M parameters repeats itself; the text is not the point.) The time is what changes. With a short prompt the gain is small, 1.2 times, because each step of a tiny model is dominated by fixed costs. With a prompt of 400 tokens, the cache cuts the time by 2.3 times. The longer the context, the more work the cache saves.

So the cache trades memory for time. Every request keeps its keys and values in GPU memory for as long as it runs. The rest of the paper is about the memory side of that trade.

<figure class="fig"><svg viewBox="0 0 720 250" role="img" aria-label="Seven tokens of a request in a row. Under each token sit its key and value. The first four, the prompt, are written in one pass. The next three are added one per step during generation."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The KV cache of one request, as it grows</text><rect class="box" x="110.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="147.0" y="62.5" text-anchor="middle">Four</text><rect class="box-1" x="110.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="147.0" y="102.5" text-anchor="middle">k1</text><rect class="box-1" x="110.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="147.0" y="136.5" text-anchor="middle">v1</text><rect class="box" x="190.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="227.0" y="62.5" text-anchor="middle">score</text><rect class="box-1" x="190.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="227.0" y="102.5" text-anchor="middle">k2</text><rect class="box-1" x="190.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="227.0" y="136.5" text-anchor="middle">v2</text><rect class="box" x="270.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="307.0" y="62.5" text-anchor="middle">and</text><rect class="box-1" x="270.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="307.0" y="102.5" text-anchor="middle">k3</text><rect class="box-1" x="270.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="307.0" y="136.5" text-anchor="middle">v3</text><rect class="box" x="350.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="387.0" y="62.5" text-anchor="middle">seven</text><rect class="box-1" x="350.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="387.0" y="102.5" text-anchor="middle">k4</text><rect class="box-1" x="350.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="387.0" y="136.5" text-anchor="middle">v4</text><rect class="box" x="430.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="467.0" y="62.5" text-anchor="middle">years</text><rect class="box-2" x="430.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="467.0" y="102.5" text-anchor="middle">k5</text><rect class="box-2" x="430.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="467.0" y="136.5" text-anchor="middle">v5</text><rect class="box" x="510.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="547.0" y="62.5" text-anchor="middle">ago</text><rect class="box-2" x="510.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="547.0" y="102.5" text-anchor="middle">k6</text><rect class="box-2" x="510.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="547.0" y="136.5" text-anchor="middle">v6</text><rect class="box" x="590.0" y="44.0" width="74.0" height="28.0" rx="6"/><text class="t-tick" x="627.0" y="62.5" text-anchor="middle">our</text><rect class="box-2" x="590.0" y="84.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="627.0" y="102.5" text-anchor="middle">k7</text><rect class="box-2" x="590.0" y="118.0" width="74.0" height="28.0" rx="6"/><text class="t-mono" x="627.0" y="136.5" text-anchor="middle">v7</text><text class="t-tick" x="100.0" y="63.0" text-anchor="end">token</text><text class="t-tick" x="100.0" y="103.0" text-anchor="end">key</text><text class="t-tick" x="100.0" y="137.0" text-anchor="end">value</text><line class="edge-dim" x1="427.0" y1="40" x2="427.0" y2="160"/><text class="t-s1" x="110.0" y="176.0" text-anchor="start">prompt phase: 4 tokens, one pass</text><text class="t-s2" x="430.0" y="176.0" text-anchor="start">generation phase: one new column per step</text><text class="t-tick" x="20.0" y="210.0" text-anchor="start">Each column holds one key and one value vector for every layer. A new token reads all earlier</text><text class="t-tick" x="20.0" y="230.0" text-anchor="start">columns and adds its own. The cache grows by one column per step, until the request ends.</text></svg><figcaption>The KV cache of one request. The prompt phase writes the keys and values of all prompt tokens in one pass. Each step of the generation phase adds one more column.</figcaption></figure>

## How large the KV cache is {§1}

> [!PAPER] · §1
> [![The paragraph about batching and Figure 1, with batching multiple requests, Approximately 65% of the memory, remain static, Close to 30% of the memory, dynamic states of the requests, and commonly referred to as KV cache highlighted](/img/papers/vllm/p1-intro-batching.png)](/img/papers/vllm/p1-intro-batching.png)
>
> **What it says:** throughput improves by "batching multiple requests together". For a 13B model on a 40 GB A100, "Approximately 65% of the memory" holds the weights, which "remain static". "Close to 30% of the memory" stores the "dynamic states of the requests": the key and value tensors, "commonly referred to as KV cache".
>
> **What it means:**
>
> The weights are the same for every request, so they are loaded once and stay. The KV cache is different: each request has its own, it starts small, it grows with every token, and it vanishes when the request ends. "Dynamic states" is a good name. It is the working memory of each conversation.
>
> The paper's §3 gives the size. For OPT-13B, one token needs 800 KB of cache. That is 2 (a key and a value) times 5,120 (the vector size) times 40 layers times 2 bytes. A full request of 2,048 tokens needs about 1.6 GB. So 12 GB holds only a few requests of full length.

We check this formula twice. Method 1 applies it to the real configuration file of OPT-13B. Method 2 runs OPT-125M on a prompt and adds up the bytes of the cache tensors that the library really keeps. If the formula is right, it predicts those bytes exactly:

```python
def kv_bytes_per_token(cfg, bytes_per_value=2):
    """The paper's formula (§3): 2 (key and value) x hidden size x number of layers x bytes per FP16 number."""
    return 2 * cfg['hidden_size'] * cfg['num_hidden_layers'] * bytes_per_value


def cache_bytes(cache):
    """Method 2: add up the bytes of every key and value tensor that the library really keeps."""
    return sum(t.numel() * t.element_size() for layer in cache.layers for t in (layer.keys, layer.values))
```

```text
== 1. The KV cache of one token
OPT-13B config: 40 layers, hidden size 5120, max length 2048 tokens
Method 1, formula: 2 x 5120 x 40 x 2 bytes = 819,200 bytes = 800 KB per token
one request of 2048 tokens: 1,677,721,600 bytes = 1.68 GB
OPT-125M: 12 layers, hidden size 768; prompt of 17 tokens
shape of the keys of layer 0: (1, 12, 17, 64)  (batch, heads, tokens, head size), torch.float16
Method 1, formula: 36,864 bytes per token x 17 tokens = 626,688 bytes
Method 2, real tensors: 626,688 bytes; agree: True
```

The formula matches the real tensors to the byte. The shape shows where the "hidden size" goes: 12 attention heads of 64 numbers each make 768. The 800 KB of the paper is 819,200 bytes, which is 800 times 1,024. A full request of OPT-13B is 1.68 GB in decimal units, or 1.56 GiB; the paper rounds to "1.6 GB".

> [!DEFINITION] Key and value vectors
> In each attention layer, every token gets a **query**, a **key** and a **value** vector. The new token's query is compared with the keys of all earlier tokens to decide how much attention each one gets. The answer is a weighted average of their values. Keys and values of earlier tokens never change, which is why they can be cached.

Now Figure 1's budget, as arithmetic. The 13B model takes 26 GB. Suppose the cache gets 30% of the card, 12 GB:

```text
== 2. Figure 1 (left): a 13B model on a 40 GB A100
weights: 13B parameters x 2 bytes = 26 GB = 65% of 40 GB
KV cache room at 30% of the GPU: 12 GB = 14,648 tokens of OPT-13B
requests of 2048 tokens that fit: 7.2
```

Twelve gigabytes sounds large, but it holds only 14,648 tokens of OPT-13B. If every request books room for 2,048 tokens, only 7 requests fit. That matches the orange line of Figure 1, which hits the ceiling at a batch of about 8.

> [!PAPER] · §1
> [![The next lines of the introduction, with the way the KV cache is managed is critical in determining the maximum batch size highlighted](/img/papers/vllm/p1-intro-batch-limit.png)](/img/papers/vllm/p1-intro-batch-limit.png)
>
> **What it says:** because the weights are constant and the activations are small, "the way the KV cache is managed is critical in determining the maximum batch size".
>
> **What it means:**
>
> This is the key sentence of §1. Of the three slices of memory, two are fixed. The weights cannot shrink without changing the model. The activations are small and short-lived. Only the KV cache slice depends on decisions that the serving system makes. So the batch size, and with it the throughput, is decided by how well the system packs KV caches into that slice.

> [!DEFINITION] Activations
> The temporary results inside the model while it computes one step, such as the output of each layer. They are thrown away after the step. The paper's Figure 1 calls them "Others".

### The two phases, and what batching buys

The paper's §2 names the two phases that we met in the chat request above. The prompt phase reads all prompt tokens in one pass. The generation phase writes one token per pass. We time both on the same 400 tokens. If the GPU is starved of data, the second way should be far slower:

```text
== 4. The prompt phase against the generation phase (OPT-125M, 400 tokens)
prompt phase, 400 tokens in one pass:       9.6 ms =    41,536 tokens/s
the same 400 tokens one at a time:       1629.9 ms =       245 tokens/s
one pass is 169x faster for the same work
```

The work is the same, but one pass is 169 times faster. In one pass, the GPU reads the weights once and uses them for 400 tokens. One at a time, it reads them 400 times. On our laptop, part of each step is also the fixed cost of a call from Python. So the exact factor is larger than on a server GPU. The direction is the paper's point: the generation phase wastes the GPU.

Batching gives back what the generation phase loses. We run one decode step for 1 to 64 requests at once, each with 128 tokens of cache. The function below is one such step; after it, we cut the cache back, so every timed run does the same work. Watch the step time grow much more slowly than the batch:

```python
def step():
    with torch.no_grad():
        model(step_in, past_key_values=cache, use_cache=True)
        for layer in cache.layers:                      # drop the token just added, so every run is the same
            layer.keys, layer.values = layer.keys[:, :, :CTX], layer.values[:, :, :CTX]
```

```text
== 5. One decode step for a batch of requests (each with a 128-token context)
batch  step ms  tokens/s
    1     4.51       221
    2     5.45       367
    4     5.07       788
    8     6.18     1,294
   16     7.22     2,217
   32     7.93     4,033
   64    10.01     6,393
from batch 1 to 64: step time x2.2, throughput x28.9
```

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="Tokens per second of one decode step against the batch size, from 221, 367, 788, 1,294, 2,217, 4,033, 6,393 tokens per second for batches of 1, 2, 4, 8, 16, 32, 64."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="64" y1="250.0" x2="550" y2="250.0"/><text class="t-tick" x="56.0" y="254.0" text-anchor="end">0</text><line class="grid" x1="64" y1="187.1" x2="550" y2="187.1"/><text class="t-tick" x="56.0" y="191.1" text-anchor="end">2,000</text><line class="grid" x1="64" y1="124.3" x2="550" y2="124.3"/><text class="t-tick" x="56.0" y="128.3" text-anchor="end">4,000</text><line class="grid" x1="64" y1="61.4" x2="550" y2="61.4"/><text class="t-tick" x="56.0" y="65.4" text-anchor="end">6,000</text><text class="t-tick" x="64.0" y="270.0" text-anchor="middle">1</text><text class="t-tick" x="145.0" y="270.0" text-anchor="middle">2</text><text class="t-tick" x="226.0" y="270.0" text-anchor="middle">4</text><text class="t-tick" x="307.0" y="270.0" text-anchor="middle">8</text><text class="t-tick" x="388.0" y="270.0" text-anchor="middle">16</text><text class="t-tick" x="469.0" y="270.0" text-anchor="middle">32</text><text class="t-tick" x="550.0" y="270.0" text-anchor="middle">64</text><line class="axis" x1="64" y1="250" x2="550" y2="250"/><polyline class="l1" points="64.0,243.0 145.0,238.5 226.0,225.2 307.0,209.3 388.0,180.3 469.0,123.2 550.0,49.1"/><g class="mark"><title>tokens/s, 1: 221</title><circle class="s1 ring" cx="64.0" cy="243.0" r="5"/></g><g class="mark"><title>tokens/s, 2: 367</title><circle class="s1 ring" cx="145.0" cy="238.5" r="5"/></g><g class="mark"><title>tokens/s, 4: 788</title><circle class="s1 ring" cx="226.0" cy="225.2" r="5"/></g><g class="mark"><title>tokens/s, 8: 1,294</title><circle class="s1 ring" cx="307.0" cy="209.3" r="5"/></g><g class="mark"><title>tokens/s, 16: 2,217</title><circle class="s1 ring" cx="388.0" cy="180.3" r="5"/></g><g class="mark"><title>tokens/s, 32: 4,033</title><circle class="s1 ring" cx="469.0" cy="123.2" r="5"/></g><g class="mark"><title>tokens/s, 64: 6,393</title><circle class="s1 ring" cx="550.0" cy="49.1" r="5"/></g><line class="grid" x1="556.0" y1="49.1" x2="566.0" y2="49.1"/><rect class="s1" x="570.0" y="44.1" width="10" height="10" rx="2"/><text class="t-note" x="586.0" y="53.1" text-anchor="start">tokens/s: 6,393</text><text class="t-tick" x="307.0" y="290.0" text-anchor="middle">batch size (requests decoded together, log scale)</text><text class="t-tick" x="14.0" y="18.0" text-anchor="start">tokens per second</text></svg><figcaption>Our run: one decode step of OPT-125M on a laptop GPU. A batch of 64 takes 2.2 times as long as a batch of 1, but it writes 64 tokens instead of 1, so throughput grows 29 times.</figcaption></figure>

Sixty-four requests take 2.2 times as long as one request, and they produce 64 tokens. So throughput grows 29 times. This is the bottom panel of Figure 1 on a small scale: throughput climbs with the batch size. On a real server, what stops the climb is the memory for all those caches. That is the subject of the rest of the paper. Our timings are the fastest of 7 runs on a laptop, so treat them as a shape, not as server numbers.

## Why the existing systems waste memory {§1}

> [!PAPER] · §1
> [![The paragraph about contiguous memory, with contiguous memory space, tensors to be stored in contiguous memory, dynamically grows and shrinks over time, and lifetime and length are not known a priori highlighted](/img/papers/vllm/p1-intro-contiguous.png)](/img/papers/vllm/p1-intro-contiguous.png)
>
> **What it says:** existing systems store the KV cache of a request "in contiguous memory space", because most deep learning frameworks require "tensors to be stored in contiguous memory". But the KV cache "dynamically grows and shrinks over time", and its "lifetime and length are not known a priori".
>
> **What it means:**
>
> A tensor in PyTorch is one unbroken block of memory. The fast GPU code that computes attention expects the keys of a request to sit side by side in that block. So the simple design is one block per request, and the block must be big enough for the whole cache.
>
> The trouble is the size. When a request arrives, nobody knows how long the answer will be. It may be 10 tokens or 2,000. A block that is too small cannot grow in place, because other data may sit right after it. So the system books the largest size up front, just in case.
>
> Ordinary deep learning never hits this problem. When you train a model or classify an image, the tensor sizes are known before the computation starts. The KV cache is the unusual case: its size is decided by the model's own output, one token at a time.

> [!DEFINITION] Contiguous memory
> Memory addresses that follow each other without a gap, like seats in one row. A contiguous block cannot grow if the next seats are taken; it must move to a larger free row.

> [!DEFINITION] Tensor
> An array of numbers with any number of dimensions, the basic data type of PyTorch. A vector is a 1-D tensor and a matrix is a 2-D tensor. The KV cache of a request is stored as tensors.

> [!NOTE] Where this comes from: why tensors are contiguous
> A tensor library stores a matrix as one long row of numbers. To find the number in row i and column j, it computes one address: the start, plus i times the row length, plus j. No lookup is needed. GPU code is fast partly because of this. Threads that sit side by side read numbers that sit side by side, in one wide read.
>
> Think of a street where the house numbers run in order. You find number 40 without asking anyone. If the houses were scattered across town, you would need a directory first. PagedAttention accepts that small cost: Part 3 shows how its kernel reads a table first, then the keys and values.

> [!PAPER] · §1
> [![The paragraph about fragmentation, with pre-allocate a contiguous chunk, request's maximum length, internal fragmentation, other shorter requests cannot utilize any, external, and only 20.4% - 38.2% of the KV cache memory is used to store the actual token states highlighted](/img/papers/vllm/p1-intro-frag.png)](/img/papers/vllm/p1-intro-frag.png)
>
> **What it says:** existing systems "pre-allocate a contiguous chunk of memory with the request's maximum length (e.g., 2048 tokens)". This causes internal fragmentation, because the real length is often much shorter. Even with a known length, "other shorter requests cannot utilize any part of the chunk" that is still unused. External fragmentation adds more waste. In the authors' profiling, "only 20.4% - 38.2% of the KV cache memory is used to store the actual token states".
>
> **What it means:**
>
> Picture a restaurant that books tables by the longest possible dinner. A couple that stays one hour holds a table for five hours. That is the first waste, **internal fragmentation**: memory inside a request's chunk that its tokens never fill.
>
> Now suppose the restaurant knows exactly how long each guest will stay. It still books the table from the start of the meal to the end. During the first hour the table is half used, but nobody else can sit there. That is the second waste, which the paper calls **reservation**: memory that this request will fill later, but which is blocked now.
>
> The third waste is **external fragmentation**. Chunks of different sizes come and go, and they leave gaps between them. A gap of 300 tokens is useless to a request that needs a chunk of 2,048. This is true even if many such gaps add up to more. All three together leave only 20% to 38% of the cache memory holding real data.

> [!DEFINITION] Internal and external fragmentation
> **Internal fragmentation** is unused memory inside a block that has been given to someone. **External fragmentation** is free memory between blocks, split into gaps that are each too small to use. Both are old terms from operating systems.

<figure class="fig"><svg viewBox="0 0 720 300" role="img" aria-label="Two panels. Left: requests A and B each own one contiguous chunk with reserved and never-used slots, and a small gap between chunks. Right: memory is cut into small blocks, and the blocks of A and B sit at any free place, mapped by a table per request."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Contiguous chunks against blocks</text><text class="t-tick t-strong" x="24.0" y="58.0" text-anchor="start">Existing systems: one chunk per request, sized for its maximum</text><rect class="box-1" x="24.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-1" x="37.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-1" x="50.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-2" x="63.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-2" x="76.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="89.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="102.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="115.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="128.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="141.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="154.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="167.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="180.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="193.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-ghost" x="206.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-ghost" x="219.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-3" x="232.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-3" x="245.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-3" x="258.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="box-2" x="271.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="284.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="297.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="310.0" y="70.0" width="11.0" height="30.0" rx="2"/><rect class="waste" x="323.0" y="70.0" width="11.0" height="30.0" rx="2"/><text class="t-tick" x="24.0" y="120.0" text-anchor="start">A: 3 tokens, 2 reserved, 9 never used</text><text class="t-tick" x="24.0" y="138.0" text-anchor="start">B: 3 tokens, 1 reserved, 4 never used</text><text class="t-tick" x="24.0" y="156.0" text-anchor="start">gap of 2 slots: too small for a new chunk</text><text class="t-tick t-strong" x="400.0" y="58.0" text-anchor="start">PagedAttention: blocks of fixed size, anywhere</text><rect class="box-1" x="400.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-mono" x="417.0" y="90.0" text-anchor="middle">A0</text><rect class="box-3" x="438.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-mono" x="455.0" y="90.0" text-anchor="middle">B0</text><rect class="box-ghost" x="476.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-muted" x="493.0" y="90.0" text-anchor="middle">2</text><rect class="box-1" x="514.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-mono" x="531.0" y="90.0" text-anchor="middle">A1</text><rect class="box-ghost" x="552.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-muted" x="569.0" y="90.0" text-anchor="middle">4</text><rect class="box-3" x="590.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-mono" x="607.0" y="90.0" text-anchor="middle">B1</text><rect class="box-ghost" x="628.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-muted" x="645.0" y="90.0" text-anchor="middle">6</text><rect class="box-ghost" x="666.0" y="70.0" width="34.0" height="30.0" rx="3"/><text class="t-muted" x="683.0" y="90.0" text-anchor="middle">7</text><text class="t-tick" x="400.0" y="120.0" text-anchor="start">A: blocks 0 and 3 (3 tokens, 1 free slot)</text><text class="t-tick" x="400.0" y="138.0" text-anchor="start">B: blocks 1 and 5 (3 tokens, 1 free slot)</text><text class="t-tick" x="400.0" y="156.0" text-anchor="start">a table per request maps its blocks</text><text class="t-tick" x="20.0" y="220.0" text-anchor="start">Left: memory is reserved for the longest possible answer, and gaps between chunks stay empty.</text><text class="t-tick" x="20.0" y="240.0" text-anchor="start">Right: a request takes one small block at a time, wherever one is free, so at most the last</text><text class="t-tick" x="20.0" y="260.0" text-anchor="start">block of each request has empty slots. Part 3 builds this block table step by step.</text></svg><figcaption>Contiguous chunks against blocks. Each request books one chunk for its longest possible answer, and gaps between chunks stay empty. With blocks, a request takes one small block at a time, wherever one is free.</figcaption></figure>

> [!NOTE] Where these words come from: memory allocators
> Operating systems met this problem in the 1960s. Programs ask for pieces of memory of many sizes, and give them back in any order. The code that hands out the pieces is an **allocator**, and both kinds of fragmentation are its classic enemies.
>
> One old answer is to round every request up to a power of two. The **buddy allocator**, described by Kenneth Knowlton in 1965, keeps free memory in pieces of 2, 4, 8, 16 units and so on. Splitting and merging such pieces is fast. The price is internal fragmentation: a request for 17 units gets 32. Orca (Pow2) below books memory in the same way. The [Background page](background-the-kv-cache-and-paging.md) works through both kinds of waste with small numbers.

> [!PAPER] · Figure 2
> [![Figure 2 of the paper: stacked bars of KV cache usage for Orca (Max), Orca (Pow2), Orca (Oracle) and vLLM, split into token states, reservation, internal fragmentation, and external fragmentation and others; token states are 20.4, 26.8, 38.2 and 96.3 percent](/img/papers/vllm/p1-figure2.png)](/img/papers/vllm/p1-figure2.png)
>
> **What it says:** the average percentage of "memory wastes" in different serving systems, measured during the experiment of §6.2.
>
> **What it means:**
>
> Each bar is all of the KV cache memory of one system, split by use. Only the green part holds real tokens. Orca (Max) books 2,048 tokens per request: 20.4% green, and 57.3% internal fragmentation. Orca (Pow2) rounds the answer's real length up to a power of two, so it books at most twice too much. It gets 26.8% green, but now 41.6% is lost to external fragmentation. Chunks of many sizes leave odd gaps.
>
> Orca (Oracle) cheats on purpose: it knows each answer's length in advance and books exactly that. Even this impossible best case keeps only 38.2% green, because of reservation (25.2%) and external fragmentation (36.6%).
>
> vLLM is 96.3% green. The bars say that the problem was never the size of the GPU. In the earlier systems, 62% to 80% of the cache memory was empty or blocked. Fill it, and the same card runs two to four times as many requests.

> [!DEFINITION] Power of two
> Any of the numbers 2, 4, 8, 16, ..., 1024, 2048. Booking "the next power of two" for an answer of 25 tokens books 32. Memory allocators often round to powers of two because it makes free space easier to track.

### Figure 2 in miniature

We cannot run Orca, and we have no A100. But we can count where the memory goes under each booking rule, using the lengths of real requests. We take 2,000 random requests of Alpaca, one of the paper's two datasets. We measure their prompt and answer lengths with the OPT tokenizer. Then we walk each request through its life, token by token, and count three kinds of slots at every step:

```python
SCHEMES = {                                                # slots held, given current length, prompt and final length
    'max (2048)': lambda cur, p, final: MAXLEN,
    'pow2': lambda cur, p, final: p + 2 ** math.ceil(math.log2(final - p)),   # output booked up to 2x (25 -> 32)
    'oracle': lambda cur, p, final: final,
    f'paged ({BLOCK})': lambda cur, p, final: BLOCK * math.ceil(cur / BLOCK),
}


def simulate(alloc):
    """Method 1: walk every request step by step. At each step, count the slots that hold a token (used),
    the slots kept for tokens that will come later (reserved) and the slots that no token will ever fill (waste)."""
    used = reserved = waste = 0
    for p, o in REQS:
        final = p + o
        for cur in range(p, final + 1):                    # after the prompt phase, then after each new token
            a = alloc(cur, p, final)
            used += cur
            fut = min(a, final) - cur                       # slots that a later token of this request will fill
            reserved += fut
            waste += a - cur - fut
    return used, reserved, waste
```

Each rule takes the current length, the prompt length and the final length of a request, and returns how many slots it holds. The first three book once, at the start, as the paper's three versions of Orca do. The power-of-two rule rounds the answer's length up, so an answer of 25 tokens books 32 slots. The paged rule books blocks of 16 slots, one more each time the last block is full.

Method 2 computes the same totals with a closed formula instead of the loop. The sum of the current lengths over the steps $$p, p+1, \dots, p+o$$ is $$(o+1)(2p+o)/2$$, so no walk is needed. The two methods must agree to the slot:

```text
== 6. How much reserved KV cache memory holds real tokens (Alpaca lengths, OPT tokenizer)
2000 requests; mean prompt 18.9 tokens, mean output 58.4 tokens, longest 514 tokens
scheme       token states  reserved  never used  Method 2 agrees
max (2048)           3.6%      2.7%       93.7%             True
pow2                41.2%     31.3%       27.4%             True
oracle              56.8%     43.2%        0.0%             True
paged (16)          90.7%      8.4%        0.8%             True
external fragmentation is not simulated: we place no requests in a real address space
requests of OPT-13B that fit in 12 GB of KV cache, on average over their lifetime:
  max (2048)    2048.0 slots per request ->    7.2 requests
  pow2           177.5 slots per request ->   82.5 requests
  oracle         128.9 slots per request ->  113.7 requests
  paged (16)      80.7 slots per request ->  181.5 requests
```

<figure class="fig"><svg viewBox="0 0 720 240" role="img" aria-label="Four stacked bars. The share of memory that holds a token: max (2048) 3.6 percent, pow2 41.2 percent, oracle 56.8 percent, paged (16) 90.7 percent."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Where the KV cache memory goes (our simulation, 2,000 Alpaca requests)</text><rect class="s1" x="20" y="40" width="12" height="12" rx="2"/><text class="t-tick" x="38.0" y="50.0" text-anchor="start">holds a token now</text><rect class="s2" x="180" y="40" width="12" height="12" rx="2"/><text class="t-tick" x="198.0" y="50.0" text-anchor="start">reserved for a later token</text><rect class="waste" x="412" y="40" width="12" height="12" rx="2"/><text class="t-tick" x="430.0" y="50.0" text-anchor="start">never used</text><text class="t-tick t-strong" x="132.0" y="92.0" text-anchor="end">max (2048)</text><g class="mark"><title>max (2048), tokens: 3.6%</title><rect class="s1" x="140.0" y="76" width="18.6" height="26"/></g><g class="mark"><title>max (2048), reserved: 2.7%</title><rect class="s2" x="158.6" y="76" width="14.1" height="26"/></g><g class="mark"><title>max (2048), waste: 93.7%</title><rect class="waste" x="172.7" y="76" width="487.3" height="26"/></g><text class="t-val" x="668.0" y="92.0" text-anchor="start">3.6%</text><text class="t-tick t-strong" x="132.0" y="132.0" text-anchor="end">pow2</text><g class="mark"><title>pow2, tokens: 41.2%</title><rect class="s1" x="140.0" y="116" width="214.5" height="26"/></g><g class="mark"><title>pow2, reserved: 31.3%</title><rect class="s2" x="354.5" y="116" width="163.0" height="26"/></g><g class="mark"><title>pow2, waste: 27.4%</title><rect class="waste" x="517.4" y="116" width="142.6" height="26"/></g><text class="t-val" x="668.0" y="132.0" text-anchor="start">41.2%</text><text class="t-tick t-strong" x="132.0" y="172.0" text-anchor="end">oracle</text><g class="mark"><title>oracle, tokens: 56.8%</title><rect class="s1" x="140.0" y="156" width="295.5" height="26"/></g><g class="mark"><title>oracle, reserved: 43.2%</title><rect class="s2" x="435.5" y="156" width="224.5" height="26"/></g><text class="t-val" x="668.0" y="172.0" text-anchor="start">56.8%</text><text class="t-tick t-strong" x="132.0" y="212.0" text-anchor="end">paged (16)</text><g class="mark"><title>paged (16), tokens: 90.7%</title><rect class="s1" x="140.0" y="196" width="471.8" height="26"/></g><g class="mark"><title>paged (16), reserved: 8.4%</title><rect class="s2" x="611.8" y="196" width="43.8" height="26"/></g><g class="mark"><title>paged (16), waste: 0.8%</title><rect class="waste" x="655.6" y="196" width="4.4" height="26"/></g><text class="t-val" x="668.0" y="212.0" text-anchor="start">90.7%</text></svg><figcaption>Our simulation on 2,000 Alpaca requests: the share of booked KV cache memory that holds a token, that waits for a later token, and that no token ever fills. Blocks of 16 slots keep 90.7% of the memory full.</figcaption></figure>

The order is the paper's order: max, then power of two, then oracle, then paged. Alpaca requests are short, 77 tokens on average, so booking 2,048 slots is extreme here: only 3.6% of the memory holds tokens. Even the oracle keeps 43.2% blocked as reservation, because a request books its final length on its first step. Paged blocks keep 90.7% of the memory full. On average a request holds 80.7 slots instead of 2,048, so 181 requests fit where 7 did.

Our numbers differ from Figure 2, and the reasons are worth knowing. We place no chunks in a real address space, so we count no external fragmentation; that is why our oracle shows 56.8% instead of 38.2%. The paper's bars come from a running system with ShareGPT or Alpaca traffic, a scheduler and requests that arrive over time. Our paged share is below the paper's 96.3% because our requests are short. The half-empty last block of 16 slots is a larger part of each one. The lesson is the same: booking ahead is what wastes memory.

Here is what those counts mean for one real card. We take Figure 1's budget, 12 GB of KV cache space, and fill it with requests under the first rule and under the paged rule:

<figure class="fig"><svg viewBox="0 0 720 330" role="img" aria-label="Two bars for one 40 GB GPU. Both give 65 percent to the weights. Before vLLM, the 12 GB of KV cache space holds 7.2 chunks of 2,048 slots, mostly empty. With vLLM it holds small blocks, 90.7 percent full, for 181.5 requests on average."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The same 40 GB GPU, before vLLM and with vLLM</text><text class="t-tick t-strong" x="20.0" y="56.0" text-anchor="start">Before: one chunk of 2,048 slots per request</text><rect class="box" x="20.0" y="64.0" width="440.0" height="54.0" rx="4"/><text class="t-tick" x="241.0" y="96.0" text-anchor="middle">weights: 26 GB (65%), never change</text><rect class="box-ghost" x="462.0" y="64.0" width="32.0" height="54.0" rx="4"/><text class="t-tick" x="20.0" y="138.0" text-anchor="start">7.2 requests fit in the KV cache space; only 3.6% of it holds a token (our Alpaca simulation)</text><rect class="waste" x="496.0" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="496.0" y="115.1" width="27.1" height="2.9" rx="1"/><rect class="waste" x="525.1" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="525.1" y="115.1" width="27.1" height="2.9" rx="1"/><rect class="waste" x="554.3" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="554.3" y="115.1" width="27.1" height="2.9" rx="1"/><rect class="waste" x="583.4" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="583.4" y="115.1" width="27.1" height="2.9" rx="1"/><rect class="waste" x="612.6" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="612.6" y="115.1" width="27.1" height="2.9" rx="1"/><rect class="waste" x="641.7" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="641.7" y="115.1" width="27.1" height="2.9" rx="1"/><rect class="waste" x="670.9" y="64.0" width="27.1" height="54.0" rx="2"/><rect class="box-2" x="670.9" y="115.1" width="27.1" height="2.9" rx="1"/><text class="t-tick t-strong" x="20.0" y="182.0" text-anchor="start">With vLLM: blocks of 16 slots, given on demand</text><rect class="box" x="20.0" y="190.0" width="440.0" height="54.0" rx="4"/><text class="t-tick" x="241.0" y="222.0" text-anchor="middle">weights: 26 GB (65%), never change</text><rect class="box-ghost" x="462.0" y="190.0" width="32.0" height="54.0" rx="4"/><text class="t-tick" x="20.0" y="264.0" text-anchor="start">181.5 requests fit on average; 90.7% of the space holds a token</text><rect class="box-2" x="496.0" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="509.6" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="523.2" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="536.8" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="550.4" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="564.0" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="577.6" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="591.2" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="604.8" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="618.4" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="632.0" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="645.6" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="659.2" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="672.8" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="686.4" y="190.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="496.0" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="509.6" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="523.2" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="536.8" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="550.4" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="564.0" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="577.6" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="591.2" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="604.8" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="618.4" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="632.0" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="645.6" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="659.2" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="672.8" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="686.4" y="199.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="496.0" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="509.6" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="523.2" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="536.8" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="550.4" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="564.0" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="577.6" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="591.2" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="604.8" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="618.4" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="632.0" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="645.6" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="659.2" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="672.8" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="686.4" y="208.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="496.0" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="509.6" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="523.2" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="536.8" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="550.4" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="564.0" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="577.6" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="591.2" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="604.8" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="618.4" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="632.0" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="645.6" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="659.2" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="672.8" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="686.4" y="217.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="496.0" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="509.6" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="523.2" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="536.8" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="550.4" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="564.0" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="577.6" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="591.2" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="604.8" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="618.4" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="632.0" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="645.6" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="659.2" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="672.8" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="686.4" y="226.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="496.0" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="509.6" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="523.2" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="536.8" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="550.4" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="564.0" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="box-2" x="577.6" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="591.2" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="604.8" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="618.4" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="632.0" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="645.6" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="659.2" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="672.8" y="235.0" width="12.1" height="7.5" rx="1"/><rect class="waste" x="686.4" y="235.0" width="12.1" height="7.5" rx="1"/><text class="t-tick" x="598.0" y="54.0" text-anchor="middle">KV cache: 12 GB</text><text class="t-tick" x="20.0" y="314.0" text-anchor="start">Orange holds a token. Red dashes are booked but empty. The thin gap is other data. Same model, same card.</text></svg><figcaption>The same 40 GB GPU in our simulation. Before vLLM, 12 GB of KV cache space holds 7.2 chunks of 2,048 slots, and only 3.6% of it holds a token. With blocks of 16 slots, it holds 181.5 requests on average, and 90.7% of it holds a token.</figcaption></figure>

The weights do not change, and neither does the card. Only the way the cache is booked changes, and the number of requests that fit grows from 7 to about 181. For these short Alpaca requests the gap is extreme. The paper's own measurement, with real traffic, is the 2 to 4 times of the abstract.

## The second problem: no sharing {§1}

> [!PAPER] · §1
> [![The start of the paragraph about memory sharing, with cannot exploit the opportunities for memory sharing highlighted](/img/papers/vllm/p1-intro-sharing.png)](/img/papers/vllm/p1-intro-sharing.png)
> [![The rest of the paragraph, with parallel sampling and beam search, partially share their KV cache, and separate contiguous spaces highlighted](/img/papers/vllm/p1-intro-sharing-2.png)](/img/papers/vllm/p1-intro-sharing-2.png)
>
> **What it says:** existing systems "cannot exploit the opportunities for memory sharing". Decoding algorithms such as "parallel sampling and beam search" produce several outputs per request, which could "partially share their KV cache". But each sequence is stored in "separate contiguous spaces", so sharing is impossible.
>
> **What it means:**
>
> A coding assistant often shows you three suggestions for the same line. All three start from the same prompt, so the keys and values of the prompt are identical in all three. Only the new tokens differ. In a contiguous design, each suggestion gets its own chunk, and the prompt's cache is copied three times. That is the "redundant duplication" of the abstract.
>
> Beam search shares even more. It keeps several candidate answers alive, and they share long common beginnings that change as the search goes on. With one chunk per candidate, the shared part cannot be stored once. The paper's §6.3 measures how much memory sharing saves: up to 55% with beam search.

<figure class="fig"><svg viewBox="0 0 720 290" role="img" aria-label="Left: three answers, each with its own copy of the prompt cache. Right: one copy of the prompt blocks, and three answers whose tables all point to it."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Three suggestions for one prompt</text><text class="t-tick t-strong" x="20.0" y="58.0" text-anchor="start">Before: each answer has its own chunk</text><rect class="box-1" x="20.0" y="70.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="95.0" y="89.5" text-anchor="middle">prompt (copy)</text><rect class="box-2" x="176.0" y="70.0" width="90.0" height="30.0" rx="6"/><text class="t-tick" x="221.0" y="89.5" text-anchor="middle">answer 1</text><rect class="box-1" x="20.0" y="110.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="95.0" y="129.5" text-anchor="middle">prompt (copy)</text><rect class="box-2" x="176.0" y="110.0" width="90.0" height="30.0" rx="6"/><text class="t-tick" x="221.0" y="129.5" text-anchor="middle">answer 2</text><rect class="box-1" x="20.0" y="150.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="95.0" y="169.5" text-anchor="middle">prompt (copy)</text><rect class="box-2" x="176.0" y="150.0" width="90.0" height="30.0" rx="6"/><text class="t-tick" x="221.0" y="169.5" text-anchor="middle">answer 3</text><text class="t-tick" x="20.0" y="206.0" text-anchor="start">the prompt's keys and values</text><text class="t-tick" x="20.0" y="222.0" text-anchor="start">are stored 3 times</text><text class="t-tick t-strong" x="380.0" y="58.0" text-anchor="start">With vLLM: tables point at one shared copy</text><rect class="box-1" x="380.0" y="140.0" width="150.0" height="30.0" rx="6"/><text class="t-tick" x="455.0" y="159.5" text-anchor="middle">prompt blocks (1 copy)</text><rect class="box-2" x="600.0" y="70.0" width="100.0" height="30.0" rx="6"/><text class="t-tick" x="650.0" y="89.5" text-anchor="middle">answer 1</text><line class="edge" x1="596" y1="85" x2="534" y2="155" marker-end="url(#ah)"/><rect class="box-2" x="600.0" y="140.0" width="100.0" height="30.0" rx="6"/><text class="t-tick" x="650.0" y="159.5" text-anchor="middle">answer 2</text><line class="edge" x1="596" y1="155" x2="534" y2="155" marker-end="url(#ah)"/><rect class="box-2" x="600.0" y="210.0" width="100.0" height="30.0" rx="6"/><text class="t-tick" x="650.0" y="229.5" text-anchor="middle">answer 3</text><line class="edge" x1="596" y1="225" x2="534" y2="155" marker-end="url(#ah)"/><text class="t-tick" x="380.0" y="206.0" text-anchor="start">each answer's table lists the</text><text class="t-tick" x="380.0" y="222.0" text-anchor="start">same prompt blocks, then its own</text><text class="t-tick" x="20.0" y="270.0" text-anchor="start">Part 4 shows how vLLM copies a shared block only when one answer must change it.</text></svg><figcaption>Three suggestions for one prompt. Before, each answer has its own chunk with its own copy of the prompt's cache. With vLLM, the three block tables all point to one shared copy of the prompt blocks.</figcaption></figure>

> [!DEFINITION] Parallel sampling and beam search
> Two **decoding algorithms**, ways to choose the output tokens. **Parallel sampling** draws several different answers to the same prompt. **Beam search** keeps the few most likely partial answers at every step and extends each of them, then returns the best complete one.

## The fix: pages for the KV cache {§1}

> [!PAPER] · §1
> [![The paragraph that introduces PagedAttention, with virtual memory with paging, divides the request's KV cache into blocks, fixed number of tokens, not necessarily stored in contiguous space, and blocks as pages, tokens as bytes, and requests as processes highlighted](/img/papers/vllm/p1-intro-paged.png)](/img/papers/vllm/p1-intro-paged.png)
>
> **What it says:** PagedAttention borrows "virtual memory with paging" from operating systems. It "divides the request's KV cache into blocks", each holding the keys and values of a "fixed number of tokens". The blocks are "not necessarily stored in contiguous space". Think of "blocks as pages, tokens as bytes, and requests as processes".
>
> **What it means:**
>
> The analogy is exact, and it is worth reading slowly. In an operating system, a **process** (a running program) sees its memory as one long row of bytes. Behind the scenes the row is cut into pages, and the pages sit anywhere. In vLLM, a request sees its cache as one long row of tokens. Behind the scenes the row is cut into blocks, and the blocks sit anywhere.
>
> The hard part is in the name: Paged**Attention**. The attention computation must now read keys and values from blocks scattered across memory, not from one tidy tensor. The authors wrote a new GPU kernel that follows each request's block table while it computes attention. Part 3 opens that kernel.

> [!NOTE] Where this idea comes from: paging
> Around 1962, the Atlas computer at the University of Manchester had a small, fast core memory and a large, slow magnetic drum. Programs were larger than the fast memory. Atlas cut memory into pages of a fixed size and moved them between core and drum when needed. A table recorded where each page was, so a programmer could write as if all memory were one large, fast store.
>
> This is **virtual memory with paging**, and every laptop and phone uses it today. Think of a binder of loose pages. The pages of one chapter can sit anywhere, because the table of contents lists where each one is. The paper makes the same move with the KV cache:

<figure class="fig"><svg viewBox="0 0 720 330" role="img" aria-label="Two rows. Top: a program sees pages 0, 1 and 2 in order; its page table maps them to frames 5, 1 and 6 of physical memory. Bottom: a request sees blocks 0, 1 and 2; its block table maps them to GPU blocks 2, 7 and 4."><defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker><marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker><marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">The same idea twice: pages in an operating system, blocks in vLLM</text><text class="t-tick t-strong" x="20.0" y="64.0" text-anchor="start">Operating system</text><rect class="box-1" x="20.0" y="76.0" width="50.0" height="28.0" rx="6"/><text class="t-tick" x="45.0" y="94.5" text-anchor="middle">page 0</text><rect class="box-1" x="76.0" y="76.0" width="50.0" height="28.0" rx="6"/><text class="t-tick" x="101.0" y="94.5" text-anchor="middle">page 1</text><rect class="box-1" x="132.0" y="76.0" width="50.0" height="28.0" rx="6"/><text class="t-tick" x="157.0" y="94.5" text-anchor="middle">page 2</text><text class="t-muted" x="20.0" y="124.0" text-anchor="start">one program (a process)</text><text class="t-tick t-strong" x="214.0" y="64.0" text-anchor="start">page table</text><rect class="box" x="214.0" y="76.0" width="56.0" height="20.0" rx="6"/><text class="t-tick" x="242.0" y="90.5" text-anchor="middle">page 0</text><rect class="box" x="274.0" y="76.0" width="40.0" height="20.0" rx="6"/><text class="t-tick" x="294.0" y="90.5" text-anchor="middle">-&gt; 5</text><rect class="box" x="214.0" y="98.0" width="56.0" height="20.0" rx="6"/><text class="t-tick" x="242.0" y="112.5" text-anchor="middle">page 1</text><rect class="box" x="274.0" y="98.0" width="40.0" height="20.0" rx="6"/><text class="t-tick" x="294.0" y="112.5" text-anchor="middle">-&gt; 1</text><rect class="box" x="214.0" y="120.0" width="56.0" height="20.0" rx="6"/><text class="t-tick" x="242.0" y="134.5" text-anchor="middle">page 2</text><rect class="box" x="274.0" y="120.0" width="40.0" height="20.0" rx="6"/><text class="t-tick" x="294.0" y="134.5" text-anchor="middle">-&gt; 6</text><text class="t-tick t-strong" x="400.0" y="64.0" text-anchor="start">physical memory (frames 0 to 7)</text><rect class="box-ghost" x="400.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="417.0" y="94.5" text-anchor="middle">0</text><rect class="box-1" x="438.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-tick" x="455.0" y="94.5" text-anchor="middle">p1</text><rect class="box-ghost" x="476.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="493.0" y="94.5" text-anchor="middle">2</text><rect class="box-ghost" x="514.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="531.0" y="94.5" text-anchor="middle">3</text><rect class="box-ghost" x="552.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="569.0" y="94.5" text-anchor="middle">4</text><rect class="box-1" x="590.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-tick" x="607.0" y="94.5" text-anchor="middle">p0</text><rect class="box-1" x="628.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-tick" x="645.0" y="94.5" text-anchor="middle">p2</text><rect class="box-ghost" x="666.0" y="76.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="683.0" y="94.5" text-anchor="middle">7</text><text class="t-muted" x="400.0" y="124.0" text-anchor="start">the real places, scattered</text><text class="t-tick t-strong" x="20.0" y="184.0" text-anchor="start">vLLM</text><rect class="box-2" x="20.0" y="196.0" width="50.0" height="28.0" rx="6"/><text class="t-tick" x="45.0" y="214.5" text-anchor="middle">block 0</text><rect class="box-2" x="76.0" y="196.0" width="50.0" height="28.0" rx="6"/><text class="t-tick" x="101.0" y="214.5" text-anchor="middle">block 1</text><rect class="box-2" x="132.0" y="196.0" width="50.0" height="28.0" rx="6"/><text class="t-tick" x="157.0" y="214.5" text-anchor="middle">block 2</text><text class="t-muted" x="20.0" y="244.0" text-anchor="start">one request</text><text class="t-tick t-strong" x="214.0" y="184.0" text-anchor="start">block table</text><rect class="box" x="214.0" y="196.0" width="56.0" height="20.0" rx="6"/><text class="t-tick" x="242.0" y="210.5" text-anchor="middle">block 0</text><rect class="box" x="274.0" y="196.0" width="40.0" height="20.0" rx="6"/><text class="t-tick" x="294.0" y="210.5" text-anchor="middle">-&gt; 2</text><rect class="box" x="214.0" y="218.0" width="56.0" height="20.0" rx="6"/><text class="t-tick" x="242.0" y="232.5" text-anchor="middle">block 1</text><rect class="box" x="274.0" y="218.0" width="40.0" height="20.0" rx="6"/><text class="t-tick" x="294.0" y="232.5" text-anchor="middle">-&gt; 7</text><rect class="box" x="214.0" y="240.0" width="56.0" height="20.0" rx="6"/><text class="t-tick" x="242.0" y="254.5" text-anchor="middle">block 2</text><rect class="box" x="274.0" y="240.0" width="40.0" height="20.0" rx="6"/><text class="t-tick" x="294.0" y="254.5" text-anchor="middle">-&gt; 4</text><text class="t-tick t-strong" x="400.0" y="184.0" text-anchor="start">GPU KV cache (blocks 0 to 7)</text><rect class="box-ghost" x="400.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="417.0" y="214.5" text-anchor="middle">0</text><rect class="box-ghost" x="438.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="455.0" y="214.5" text-anchor="middle">1</text><rect class="box-2" x="476.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-tick" x="493.0" y="214.5" text-anchor="middle">b0</text><rect class="box-ghost" x="514.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="531.0" y="214.5" text-anchor="middle">3</text><rect class="box-2" x="552.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-tick" x="569.0" y="214.5" text-anchor="middle">b2</text><rect class="box-ghost" x="590.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="607.0" y="214.5" text-anchor="middle">5</text><rect class="box-ghost" x="628.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-muted" x="645.0" y="214.5" text-anchor="middle">6</text><rect class="box-2" x="666.0" y="196.0" width="34.0" height="28.0" rx="6"/><text class="t-tick" x="683.0" y="214.5" text-anchor="middle">b1</text><text class="t-muted" x="400.0" y="244.0" text-anchor="start">the real places, scattered</text><text class="t-tick" x="20.0" y="300.0" text-anchor="start">Pages are blocks, bytes are tokens, programs are requests. Only the table knows where each piece lives.</text></svg><figcaption>Paging in an operating system and in vLLM. A program sees pages 0, 1 and 2 in order, and its page table maps them to scattered frames. A request sees blocks 0, 1 and 2, and its block table maps them to scattered GPU blocks.</figcaption></figure>

> [!DEFINITION] Process
> A running program, as the operating system sees it. Each process has its own address space and its own page table.

> [!DEFINITION] GPU kernel
> A function that runs on the GPU, on thousands of threads at once. Attention in a serving system is one kernel. PagedAttention is a kernel that reads keys and values through a block table.

> [!PAPER] · §1
> [![The next lines, with relatively small blocks and allocating them on demand, eliminates external fragmentation, all blocks have the same size, and at the granularity of a block highlighted](/img/papers/vllm/p1-intro-paged-2.png)](/img/papers/vllm/p1-intro-paged-2.png)
>
> **What it says:** the design reduces internal fragmentation "by using relatively small blocks and allocating them on demand". It "eliminates external fragmentation as all blocks have the same size". And it enables memory sharing "at the granularity of a block".
>
> **What it means:**
>
> Each of the three wastes falls to one property of blocks. Blocks are given **on demand**, so nothing is booked for tokens that do not exist yet: no reservation. Blocks are **small**, so the empty part of the last block is small: almost no internal fragmentation. Blocks are **all the same size**, so any free block fits any request: no external fragmentation.
>
> Sharing follows from the block table. Two tables can point to the same block. When one of the two sequences must write into a shared block, vLLM copies that one block first, a trick from operating systems called **copy-on-write**. Part 4 shows it at work.

> [!NOTE] Where this idea comes from: copy-on-write
> In Unix, the call `fork()` makes a copy of a running program. Copying all of its memory would be slow, and the copy often replaces itself with another program right away. So modern Unix systems let the two programs share every page, marked read-only. When one program writes to a page, the system copies that one page first.
>
> Think of two students who share one textbook. When one of them wants to write notes in a chapter, they photocopy that chapter only. The paper does the same with blocks. It counts how many tables point to each block, and copies a block only when a shared block must change.

## vLLM, and the claim {§1}

> [!PAPER] · §1
> [![The paragraph about vLLM, with near-zero waste in KV cache memory, preemptive request scheduling, by 2-4×, and without affecting the model accuracy at all highlighted](/img/papers/vllm/p1-intro-vllm.png)](/img/papers/vllm/p1-intro-vllm.png)
>
> **What it says:** vLLM is "a high-throughput distributed LLM serving engine" built on PagedAttention that achieves "near-zero waste in KV cache memory". It uses block-level memory management and "preemptive request scheduling". It improves throughput "by 2-4×" over the state of the art "without affecting the model accuracy at all".
>
> **What it means:**
>
> PagedAttention is a kernel. vLLM is the whole engine around it. It has a scheduler that decides which requests run in each step. A manager hands out blocks, and workers run the model on one or more GPUs. "Preemptive" means that when memory runs out, the scheduler can pause a request, free or move its blocks, and resume it later.
>
> "Without affecting the model accuracy at all" is an important promise. Many speed-ups for language models change the numbers: smaller number formats, pruned weights, approximate attention. PagedAttention changes only where the keys and values are stored. The attention result is mathematically the same, so the answers are the same. Our own check above showed the same idea on a small scale: a cache changes the time, never the tokens.

> [!DEFINITION] Preemption
> Stopping a running job before it finishes, to give its resources to another job, and resuming it later. Operating systems preempt programs many times per second. When memory is short, they can also move a paused program's pages to disk and bring them back later; this is **swapping**. The paper can move blocks to CPU memory in the same way, with the CPU block allocator of the map above.

### The paper's numbers, checked

The last section of our script turns Figure 2's percentages into ratios, so we can compare them with the claim of "2-4×". Look at the line that starts with "vLLM keeps":

```text
== 7. The paper's headline numbers, checked
Figure 2, token states in Orca (Max)      20.4%   -> waste  79.6%
Figure 2, token states in Orca (Pow2)     26.8%   -> waste  73.2%
Figure 2, token states in Orca (Oracle)   38.2%   -> waste  61.8%
Figure 2, token states in vLLM            96.3%   -> waste   3.7%
Figure 2: vLLM keeps 4.7x to 2.5x more of the memory for real tokens
§3: 819,200 bytes x 2048 tokens = 1.56 GiB = 1.68 GB ("1.6 GB")
```

These ratios explain the size of the headline claim. vLLM keeps 2.5 to 4.7 times more of the cache memory full of real tokens. More full memory means larger batches, and larger batches mean more throughput. The measured gain of "2-4×" sits inside that range. It cannot be the full ratio, because other limits, such as compute, start to matter when batches get large.

## What our small run cannot show

Our run measured the KV cache of a real model, checked the paper's formula to the byte, and showed that batching raises throughput. It did not run vLLM or Orca. The paper's PagedAttention kernel is CUDA code for NVIDIA GPUs, and our laptop has an Apple GPU. Our timings come from a model of 125 million parameters, where fixed costs per step are large. On an A100 with a 13B model, the numbers would differ, though the direction would not.

Our Figure 2 is a count, not a measurement of a running server. It has no external fragmentation, no arrival times and no scheduler. Part 6 builds a small paged cache and a serving loop that do allocate real blocks, so we can see the block table at work.

> [!TAKEAWAYS] Key takeaways
> - **A serving step** runs one pass of the model for every running request. Requests join and leave the batch after every step, as in Orca's iteration-level scheduling.
> - **Serving is memory-bound.** Each new token needs a pass over all the weights, so one request at a time leaves the GPU idle. Batching many requests fixes this, until memory runs out.
> - **The KV cache** holds the keys and values of every earlier token. It costs 2 × hidden size × layers × 2 bytes per token: 800 KB for OPT-13B. In our run, the formula matched the real tensors to the byte.
> - **On a 40 GB A100** with a 13B model, the weights take 26 GB (65%). About 12 GB is left for the cache, which is 14,648 tokens, or 7 requests if each books 2,048.
> - **Existing systems book one contiguous chunk per request,** sized for the longest answer. Reservation, internal fragmentation and external fragmentation leave only 20.4% to 38.2% of the cache memory holding real tokens.
> - **They also cannot share.** Several answers to one prompt each get a full copy of the prompt's cache.
> - **PagedAttention** cuts the cache into small blocks of a fixed size. Blocks are given on demand and placed anywhere, with a block table per request. vLLM keeps 96.3% of the cache memory holding real tokens.
> - **In our simulation** on 2,000 Alpaca requests, blocks of 16 slots kept 90.7% of the memory full, against 3.6% when booking 2,048.
> - **The claim:** 2 to 4 times the throughput of FasterTransformer and Orca at the same latency, with exactly the same model outputs.

**Next, in Part 2:** §2 and §3 of the paper. Part 2 reads the attention equations behind the KV cache, the prompt and generation phases, and the batching methods of earlier systems. It then takes Figure 3 apart, slot by slot.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/vllm/vllm_part1.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part1.py). It loads OPT-125M and the OPT-13B configuration file, and measures the KV cache two ways. It times generation with and without a cache, the two phases and a batched decode step. Then it runs the memory simulation on 2,000 Alpaca requests. Every number is saved in `results/part1.json`. On an Apple M-series laptop it takes about two minutes.

```bash
pip install torch transformers datasets
python vllm_part1.py   # downloads facebook/opt-125m (about 250 MB), the OPT-13B config, and tatsu-lab/alpaca
```

<figure><img src="/img/papers/vllm/part1-run.png" alt="Terminal output of vllm_part1.py: the KV cache sizes, the Figure 1 arithmetic, generation with and without a cache, the two phases, the batch timings, the memory simulation and the paper's numbers" loading="lazy" /><figcaption>The full output of vllm_part1.py, as it ran for this article.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
