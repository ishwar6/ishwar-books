---
title: "vLLM on a Free GPU: Run It, Load It, Break It"
description: "A real vLLM server on a free Colab T4, measured like a production service: KV cache, batching, the latency knee, prefix caching, preemption, failures."
date: 2026-11-10
tags: [inference, vllm, serving, colab, llm]
series: "LLM Inference from the Ground Up"
series_part: 8
motif: code
accent: "#8bd17c"
---

Seven parts of this series took an inference engine apart. Most of their numbers came from small models on an Apple GPU and from simulators, because the subject was the ideas inside the engine. This part does the opposite. It runs the real thing: vLLM 0.33.0, serving a real model on a real NVIDIA GPU, the way a production team would run it.

The GPU is a free one. Google Colab gives anyone with a Google account an NVIDIA **T4** for a few hours at a time. It is an old card, from 2018. That turns out to be useful: its limits make every effect in this part easy to see.

Here is the result that matters most. With one user, the server wrote about **61 tokens per second**. With 8 users at the same time, it wrote about **307**, about 5 times as much, and each user's stream slowed only a little. With 128 users it wrote about **668**, about 11 times as much. But now each user got only about 6 tokens per second. Where between those two points you run your server is most of what this part is about.

> [!TIP] The whole idea in one minute
> You start one vLLM server on a free GPU and treat it like a production service. You read its startup log to learn how much memory went to the KV cache, and you check that number by hand. You add users until throughput stops growing. You raise the arrival rate until latency breaks. You switch prefix caching off and on, shrink the KV cache until requests get preempted, and kill the server under load. Every result comes from one notebook that you can run yourself, for free, in about an hour.

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb)
&nbsp; or [download the notebook](/notebooks/vllm-on-colab.ipynb).

Every number on this page comes from one full run of that notebook, on 10 November 2026. The run served Qwen/Qwen2.5-1.5B-Instruct in float16 on a Colab **Tesla T4** (15.0 GiB), with vLLM 0.33.0. Your numbers will differ by a few percent. The shapes will not.

The page follows the notebook's 11 steps. Each step asks one question, shows what we saw, and ends with what it means for a real server.

## 1. What Colab gives you, and what it does not

Colab is a notebook service. You get a Linux machine with a GPU for a few hours, and you run code in cells. It is meant for experiments, not for serving users. That is fine here. We want to **learn** how a serving engine behaves, and for that one GPU and one model are enough.

Three facts about the GPU decide most of what follows.

| | T4 (free) | L4 (paid) | A100 (paid) |
|---|---|---|---|
| Memory | 16 GB (15 GiB usable) | 24 GB | 40 or 80 GB |
| Memory bandwidth | 320 GB/s | 300 GB/s | 1,555 to 2,039 GB/s |
| Compute capability | 7.5 (Turing) | 8.9 (Ada) | 8.0 (Ampere) |
| bfloat16 | no | yes | yes |
| FP8 | no | yes | weights only, slower path |
| Attention kernel in vLLM | Triton | FlashAttention | FlashAttention |

> [!DEFINITION] Compute capability
> NVIDIA's version number for a GPU generation. Software checks it to decide which features and kernels it may use. Many of vLLM's fastest kernels need 8.0 or newer.

> [!DEFINITION] Kernel
> One program that runs on the GPU, such as "multiply these two matrices" or "compute attention for this batch". An inference engine is mostly a choice of kernels and an order to run them in.

The T4 has no hardware for **bfloat16**, the 16-bit number format most models are trained in. So we run the model in **float16**, the older 16-bit format, and we tell the server so with `--dtype half`.

> [!DEFINITION] float16 and bfloat16
> Two ways to store a number in 16 bits. The float16 format keeps more digits of precision but has a smaller range. The bfloat16 format keeps the range of a 32-bit number, with fewer digits. Most new models are trained in bfloat16.

What Colab cannot give you matters as much. There is no Docker, so you cannot run vLLM's official container image. There is one GPU, so there is no tensor parallelism. The server is reachable only from inside the notebook machine. And the session ends after some hours, or sooner if you leave it idle. A later section lists what that leaves out.

### The model

The notebook serves **Qwen/Qwen2.5-1.5B-Instruct**. It has 28 layers and 1.5 billion weights, about 3 GB in float16. It is small enough to leave most of the T4's memory for the KV cache, and good enough to give sensible answers. It needs no Hugging Face login.

A larger model would make each step slower and leave a smaller KV cache. The effects in this part would look the same. On an L4 or an A100, set `MODEL = "Qwen/Qwen2.5-7B-Instruct"` in the notebook and compare.

## 2. Start the server, and read what it tells you

**The question:** how long does a server take to start, and how much room does it have for conversations?

One command starts the server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --api-key colab-demo-key --port 8000
```

Three settings on that line matter in production:

- `--max-model-len 4096` caps one request at 4,096 tokens, prompt and answer together.
- `--gpu-memory-utilization 0.85` lets vLLM take 85% of the GPU's memory. The default is 0.92. We leave a little more room for Colab.
- `--api-key` makes every request to `/v1` carry a key. Without it, anyone who can reach the port can use your GPU.

The notebook runs this command in the background, writes its log to a file and waits until `GET /health` answers.

> [!DEFINITION] Health check
> A URL that answers "200 OK" when the server is ready to serve. Load balancers call it every few seconds and stop sending traffic to a server that does not answer.

### Where the startup time goes

The first start took **295 s**, about five minutes, from the command to the first healthy answer. That is a **cold start**.

> [!DEFINITION] Cold start and warm start
> A **cold start** begins on a machine that has never seen the model: it must download the weights and build every cache. A **warm start** finds the weights on disk and the compiled kernels in a cache.

Later in the notebook, the server restarts four more times on the same machine. Those warm starts took between 68 s and 108 s. The log shows where the difference comes from:

| Phase | What happens | Cold start | First warm start |
|---|---|---|---|
| Load the weights | Download about 3 GB (cold start only), then copy them into GPU memory | 160 s | 14 s |
| `torch.compile` | Fuse the model's operations into faster kernels; the result is cached on disk | 20 s | 0.3 s |
| Engine start | Compile, measure memory, make the KV cache, record CUDA graphs | 73 s | 22 s |
| From command to healthy | All of the above, plus starting Python and the HTTP server | **295 s** | **79 s** |

> [!DEFINITION] CUDA graph
> A recording of a fixed sequence of GPU work. Replaying it costs one launch from the CPU instead of hundreds. For small decode steps, where each kernel runs for microseconds, the launch cost would otherwise be a large share of the step.

Most of the cold start was the download. The same "Model loading" line said 160 s on the cold start and 14 s on the warm one. The compile step fell from 20 s to under a second, because its result came from the cache.

> [!TIP] In production
> Cold start time decides how fast you can add capacity when traffic jumps. Here the cold start was about 4 times slower than the warm one. The cheapest cuts are the ones you just saw. Keep the weights on a fast local disk, not in a remote bucket. Keep vLLM's compile cache (`~/.cache/vllm`) between restarts. [Part 6](llm-inference-6-serving-in-production.md) shows what slow starts do to autoscaling.

### The lines that matter

Most of the startup log is noise. These lines are not:

```text
Using TRITON_ATTN attention backend
Model loading took 2.98 GiB memory and 160.272162 seconds
[backbone] torch.compile took 20.50 s in total
Graph capturing finished in 11 secs, took 0.29 GiB
Available KV cache memory: 8.38 GiB
GPU KV cache size: 313,920 tokens, Maximum concurrency for 4,096 tokens per request: 76.64x
Graph capturing finished in 7 secs, took 0.18 GiB
init engine (profile, create kv cache, warmup model) took 73.35 s (compilation: 20.50 s)
```

Read them from the top.

**The attention backend** is `TRITON_ATTN`. vLLM picks the fastest attention kernel your GPU supports. FlashAttention and FlashInfer need compute capability 8.0. So on a T4, vLLM falls back to a kernel written in **Triton**, a language for writing GPU kernels in Python. The Triton kernel works. But on this older GPU it is slower at prefill than a native kernel. That is one reason the TTFT numbers below are higher than on an L4.

**The weights** took **2.98 GiB**.

**The KV cache** got **8.38 GiB**, which holds **313,920 tokens**. The same line turns that into a capacity: 76.64 requests of the full 4,096 tokens could keep their caches in memory at once.

That last number is the one to remember. It is the server's memory budget for conversations. Sections 4 to 8 are about how it gets spent.

<figure class="fig"><svg viewBox="0 0 760 180" role="img" aria-label="GPU memory split: weights 2.98 GiB, working memory 1.39 GiB, KV cache 8.38 GiB, and 2.25 GiB left outside vLLM."><defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="26.0" text-anchor="start">Tesla T4: 15.0 GiB of memory, 85% given to vLLM</text><rect class="box-2" x="20.0" y="50.0" width="143.0" height="46.0" rx="4"/><text class="t-note" x="91.5" y="70.0" text-anchor="middle">weights</text><text class="t-tick" x="91.5" y="87.0" text-anchor="middle">2.98 GiB</text><rect class="box" x="163.0" y="50.0" width="66.7" height="46.0" rx="4"/><line class="axis" x1="196.4" y1="97.0" x2="196.4" y2="104.0"/><text class="t-tick" x="196.4" y="117.0" text-anchor="middle">working memory: 1.39 GiB</text><rect class="box-1" x="229.8" y="50.0" width="402.2" height="46.0" rx="4"/><text class="t-note" x="430.9" y="70.0" text-anchor="middle">KV cache</text><text class="t-tick" x="430.9" y="87.0" text-anchor="middle">8.38 GiB</text><rect class="box-ghost" x="632.0" y="50.0" width="108.0" height="46.0" rx="4"/><line class="axis" x1="686.0" y1="97.0" x2="686.0" y2="104.0"/><text class="t-tick" x="686.0" y="117.0" text-anchor="middle">outside vLLM&#x27;s share: 2.25 GiB</text><text class="t-tick" x="20.0" y="146.0" text-anchor="start">KV cache: 313,920 tokens of 28 KiB each, enough for 76.6 requests of the full 4,096 tokens at once.</text><text class="t-tick" x="20.0" y="166.0" text-anchor="start">Working memory: activations for the largest batch, measured by a profile run at startup, plus the CUDA graphs.</text></svg><figcaption>Where the GPU's memory went. vLLM takes the share set by --gpu-memory-utilization, loads the weights, keeps working memory for the largest batch, and gives everything left to the KV cache.</figcaption></figure>

## 3. The first requests

**The question:** what does one user feel?

The server speaks the **OpenAI API**. Any program written for OpenAI's service works with it if you change two settings, the base URL and the key:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="colab-demo-key")
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct", temperature=0, max_tokens=120,
    messages=[{"role": "user", "content": "Explain in three sentences why an LLM server batches requests."}])
print(r.choices[0].message.content)
```

The model answered:

> An LLM (Large Language Model) server batches requests to improve efficiency and scalability. By processing multiple requests simultaneously, the server can handle more queries per unit of time, reducing wait times for users and optimizing resource utilization. This approach also helps in managing memory usage effectively, as it avoids unnecessary storage overhead associated with individual request handling. Additionally, batching allows for better load balancing across servers, ensuring that no single machine becomes overwhelmed while others remain underutilized.

That compatibility is a big part of why vLLM is easy to put into production. Your application, your SDKs and your tools stay the same. Only the address changes.

### One user, timed

With `stream=True`, the server sends each token as soon as it exists. The notebook times two things on the client side.

> [!DEFINITION] Time to first token (TTFT)
> From sending the request to receiving the first token. It covers the time in the queue and the **prefill**, the step that reads the whole prompt and writes the first token.

> [!DEFINITION] Time per output token (TPOT)
> The average gap between later tokens. Each one is a **decode** step, which writes one token per request.

For a short prompt and a 256-token answer, three runs gave:

| Run | TTFT | TPOT | Tokens per second |
|---|---|---|---|
| 1 | 32 ms | 14.7 ms | 68 |
| 2 | 36 ms | 14.7 ms | 68 |
| 3 | 29 ms | 14.8 ms | 68 |

So one conversation gets its first token in about 32 ms and then streams at about **68 tokens per second**. That is far faster than anyone reads. It is also a waste of the GPU, as Section 5 shows.

### Why decode is slow: a quick check

[Part 1](llm-inference-1-prefill-and-decode.md) explained why the two numbers differ so much. Prefill handles all the prompt tokens in one pass. Decode writes one token per step, and every step must still read all 2.98 GiB of weights from memory.

The T4 reads memory at most at 320 GB/s. Reading the weights once at that speed takes:

$$t_{\text{read}} = \frac{\text{weight bytes}}{\text{bandwidth}} = \frac{3.20 \text{ GB}}{320 \text{ GB/s}} \approx 10.0 \text{ ms}$$

Read it left to right: one decode step can never be shorter than the time to read every weight once. The measured TPOT was **14.7 ms**. The rest goes to reading the KV cache, launching kernels and work on the CPU, and no kernel reaches the full bandwidth. So the weight read explains most of each step.

## 4. The KV cache budget, checked by hand

**The question:** can we predict the KV cache size from the model's shape?

The log said the KV cache holds 313,920 tokens. Predicting that number yourself is the best way to make sure you know what it means.

> [!DEFINITION] KV cache
> For every token it has seen, each layer of the model keeps a **key** vector and a **value** vector. Later tokens look back at them without computing them again. That store is the KV cache. [Part 2](llm-inference-2-kv-cache.md) builds it from zero.

Think of the cache as a notebook that each layer keeps, with one line per token. The size of a line depends on how many key and value heads the layer has, and how long each head's vector is.

$$\text{bytes per token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times b$$

The 2 counts one key and one value. $$L$$ is the number of layers. $$H_{kv}$$ is the number of key/value heads per layer. $$d_{\text{head}}$$ is the length of one head's vector. $$b$$ is the bytes per number, 2 for float16.

For Qwen/Qwen2.5-1.5B-Instruct, the model's `config.json` gives $$L = 28$$, $$H_{kv} = 2$$ and $$d_{\text{head}} = 128$$:

$$2 \times 28 \times 2 \times 128 \times 2 = 28,672 \text{ bytes} = 28 \text{ KiB per token}$$

Divide the cache memory by that:

$$\frac{8.38 \text{ GiB} \times 2^{30}}{28,672 \text{ bytes}} \approx 313,824 \text{ tokens}$$

The log said 313,920. The two differ by only 96 tokens. The small gap is rounding: the log prints the memory with only two decimals. Turn it around and 313,920 tokens at 28,672 bytes each is 8.3826 GiB, which the log rounds to 8.38. The engine stores the cache in fixed-size **blocks** of 16 tokens, the pages of [Part 3](llm-inference-3-vllm.md). So the token count is always a whole number of blocks.

> [!DEFINITION] Grouped-query attention (GQA)
> A model design where several query heads share one key/value head. This model has 12 query heads and only 2 KV heads. So its cache is 6 times smaller than with one KV head per query head.

### Sizing the KV cache for your traffic

The same arithmetic tells you whether a server has enough room. Multiply the requests you expect to run at once by the tokens each one holds at its longest, prompt plus answer:

$$\text{tokens needed} = \text{requests at once} \times (\text{prompt tokens} + \text{output tokens})$$

The requests in Sections 5 and 6 have 512 prompt tokens and 128 output tokens, 640 in all. The cache holds about **490** of them at once. That is more than vLLM will ever run together. The scheduler also caps the number of requests in one batch, and Section 6 shows that cap at work. So on this GPU, with this model, memory is not the first limit. Section 8 makes it the limit on purpose.

> [!TIP] In production
> Do this sum before you choose a GPU. Use the longest prompts and answers you allow, not the average. If the answer is close to the log's KV cache size, expect preemption (Section 8) at busy times.

## 5. Batching: the free lunch, and its bill

**The question:** what happens when many users share the GPU?

This is the most important experiment in the notebook. We send identical requests: 512 random prompt tokens and exactly 128 output tokens. We keep a fixed number of them **in flight** at the same time: 1, then 2, 4, 8 and so on up to 128. As soon as one finishes, the next one starts. The tool is `vllm bench serve`, the load generator that ships with vLLM:

```bash
vllm bench serve --backend openai --base-url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --max-concurrency 64 --num-prompts 256 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result
```

> [!DEFINITION] Closed loop
> A load test where a fixed number of simulated users each wait for their answer before they send the next request. The load adapts to the server's speed: a slow server gets fewer requests.

> [!DEFINITION] p50 and p99
> Percentiles. The **p50** (median) is the value that half of the requests beat. The **p99** is the value that 99 requests in 100 beat, so it shows the slowest 1%. Users remember the slow ones, so production targets are usually set on p99.

`--ignore-eos` makes every answer exactly 128 tokens long, so every run does the same work. The random prompts are random token IDs. Their text is noise, but their cost to the GPU is real.

<figure class="fig"><svg viewBox="0 0 760 560" role="img" aria-label="Output throughput rises from 61 to 668 tokens per second as requests in flight go from 1 to 128, while time per token per user rises from 15.6 to 178.7 ms."><defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-title" x="20.0" y="22.0" text-anchor="start">Whole server: output tokens per second (higher is better)</text><line class="grid" x1="80.0" y1="214.0" x2="720.0" y2="214.0"/><text class="t-tick" x="72.0" y="218.0" text-anchor="end">0</text><line class="grid" x1="80.0" y1="171.5" x2="720.0" y2="171.5"/><text class="t-tick" x="72.0" y="175.5" text-anchor="end">200</text><line class="grid" x1="80.0" y1="129.0" x2="720.0" y2="129.0"/><text class="t-tick" x="72.0" y="133.0" text-anchor="end">400</text><line class="grid" x1="80.0" y1="86.5" x2="720.0" y2="86.5"/><text class="t-tick" x="72.0" y="90.5" text-anchor="end">600</text><line class="grid" x1="80.0" y1="44.0" x2="720.0" y2="44.0"/><text class="t-tick" x="72.0" y="48.0" text-anchor="end">800</text><text class="t-tick" x="80.0" y="232.0" text-anchor="middle">1</text><text class="t-tick" x="171.4" y="232.0" text-anchor="middle">2</text><text class="t-tick" x="262.9" y="232.0" text-anchor="middle">4</text><text class="t-tick" x="354.3" y="232.0" text-anchor="middle">8</text><text class="t-tick" x="445.7" y="232.0" text-anchor="middle">16</text><text class="t-tick" x="537.1" y="232.0" text-anchor="middle">32</text><text class="t-tick" x="628.6" y="232.0" text-anchor="middle">64</text><text class="t-tick" x="720.0" y="232.0" text-anchor="middle">128</text><line class="axis" x1="80.0" y1="214.0" x2="720.0" y2="214.0"/><text class="t-tick" x="400.0" y="252.0" text-anchor="middle">requests in flight</text><polyline class="l1" points="80.0,201.1 171.4,190.3 262.9,172.6 354.3,148.7 445.7,120.4 537.1,100.5 628.6,81.2 720.0,72.1"/><g class="mark"><title>1 in flight: 61 tok/s</title><circle class="s1" cx="80.0" cy="201.1" r="4"/></g><g class="mark"><title>2 in flight: 111 tok/s</title><circle class="s1" cx="171.4" cy="190.3" r="4"/></g><g class="mark"><title>4 in flight: 195 tok/s</title><circle class="s1" cx="262.9" cy="172.6" r="4"/></g><g class="mark"><title>8 in flight: 307 tok/s</title><circle class="s1" cx="354.3" cy="148.7" r="4"/></g><g class="mark"><title>16 in flight: 441 tok/s</title><circle class="s1" cx="445.7" cy="120.4" r="4"/></g><g class="mark"><title>32 in flight: 534 tok/s</title><circle class="s1" cx="537.1" cy="100.5" r="4"/></g><g class="mark"><title>64 in flight: 625 tok/s</title><circle class="s1" cx="628.6" cy="81.2" r="4"/></g><g class="mark"><title>128 in flight: 668 tok/s</title><circle class="s1" cx="720.0" cy="72.1" r="4"/></g><text class="t-tick" x="88.0" y="193.1" text-anchor="start">61</text><text class="t-tick" x="712.0" y="62.1" text-anchor="end">668</text><text class="t-title" x="20.0" y="302.0" text-anchor="start">Each user: milliseconds per token (lower is better)</text><line class="grid" x1="80.0" y1="494.0" x2="720.0" y2="494.0"/><text class="t-tick" x="72.0" y="498.0" text-anchor="end">0</text><line class="grid" x1="80.0" y1="451.5" x2="720.0" y2="451.5"/><text class="t-tick" x="72.0" y="455.5" text-anchor="end">50</text><line class="grid" x1="80.0" y1="409.0" x2="720.0" y2="409.0"/><text class="t-tick" x="72.0" y="413.0" text-anchor="end">100</text><line class="grid" x1="80.0" y1="366.5" x2="720.0" y2="366.5"/><text class="t-tick" x="72.0" y="370.5" text-anchor="end">150</text><line class="grid" x1="80.0" y1="324.0" x2="720.0" y2="324.0"/><text class="t-tick" x="72.0" y="328.0" text-anchor="end">200</text><text class="t-tick" x="80.0" y="512.0" text-anchor="middle">1</text><text class="t-tick" x="171.4" y="512.0" text-anchor="middle">2</text><text class="t-tick" x="262.9" y="512.0" text-anchor="middle">4</text><text class="t-tick" x="354.3" y="512.0" text-anchor="middle">8</text><text class="t-tick" x="445.7" y="512.0" text-anchor="middle">16</text><text class="t-tick" x="537.1" y="512.0" text-anchor="middle">32</text><text class="t-tick" x="628.6" y="512.0" text-anchor="middle">64</text><text class="t-tick" x="720.0" y="512.0" text-anchor="middle">128</text><line class="axis" x1="80.0" y1="494.0" x2="720.0" y2="494.0"/><text class="t-tick" x="400.0" y="532.0" text-anchor="middle">requests in flight</text><polyline class="l2" points="80.0,480.8 171.4,479.9 262.9,479.4 354.3,476.3 445.7,469.7 537.1,452.1 628.6,416.5 720.0,342.1"/><g class="mark"><title>1 in flight: 15.6 ms per token</title><circle class="s2" cx="80.0" cy="480.8" r="4"/></g><g class="mark"><title>2 in flight: 16.6 ms per token</title><circle class="s2" cx="171.4" cy="479.9" r="4"/></g><g class="mark"><title>4 in flight: 17.2 ms per token</title><circle class="s2" cx="262.9" cy="479.4" r="4"/></g><g class="mark"><title>8 in flight: 20.9 ms per token</title><circle class="s2" cx="354.3" cy="476.3" r="4"/></g><g class="mark"><title>16 in flight: 28.6 ms per token</title><circle class="s2" cx="445.7" cy="469.7" r="4"/></g><g class="mark"><title>32 in flight: 49.3 ms per token</title><circle class="s2" cx="537.1" cy="452.1" r="4"/></g><g class="mark"><title>64 in flight: 91.2 ms per token</title><circle class="s2" cx="628.6" cy="416.5" r="4"/></g><g class="mark"><title>128 in flight: 178.7 ms per token</title><circle class="s2" cx="720.0" cy="342.1" r="4"/></g><text class="t-tick" x="88.0" y="472.8" text-anchor="start">15.6 ms</text><text class="t-tick" x="712.0" y="346.1" text-anchor="end">178.7 ms</text></svg><figcaption>Top: output tokens per second for the whole server. Bottom: milliseconds per token for each user. Both against the number of requests in flight, on a log scale.</figcaption></figure>

| In flight | Server output, tok/s | Speed-up vs 1 | Per-user TPOT, p50 | Per-user tok/s | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|---|
| 1 | 61 | 1.0x | 15.6 ms | 64 | 118 ms | 211 ms |
| 2 | 111 | 1.8x | 16.6 ms | 60 | 219 ms | 336 ms |
| 4 | 195 | 3.2x | 17.2 ms | 58 | 427 ms | 571 ms |
| 8 | 307 | 5.1x | 20.9 ms | 48 | 667 ms | 1.00 s |
| 16 | 441 | 7.3x | 28.6 ms | 35 | 966 ms | 1.80 s |
| 32 | 534 | 8.8x | 49.3 ms | 20 | 1.27 s | 3.51 s |
| 64 | 625 | 10.3x | 91.2 ms | 11 | 1.34 s | 7.08 s |
| 128 | 668 | 11.0x | 178.7 ms | 6 | 1.88 s | 14.40 s |

Read the table in two halves.

**The top half is the free lunch.** From 1 to 8 requests in flight, the server's output grew about 5 times. Each user's time per token grew only from 15.6 to 20.9 ms.

[Part 1](llm-inference-1-prefill-and-decode.md) promised this. A decode step reads all the weights once, whether it writes a token for one request or for eight. The extra arithmetic for eight tokens is small next to that read. So eight requests per step cost little more than one.

> [!DEFINITION] Continuous batching
> The engine decides which requests are in the batch at every step. A finished request leaves at once, and a waiting one joins at the next step. [Part 3](llm-inference-3-vllm.md) explains how vLLM does it.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §2.3 · page 3
> [![The paragraph on fine-grained batching, with the iteration level, completed requests are removed from the batch and new ones are added, waiting for a single iteration and eliminate the need to pad highlighted](/img/papers/vllm/p2-fine-batching.png)](/img/papers/vllm/p2-fine-batching.png)
>
> **Context:** the vLLM paper's background section, on how serving systems batch requests.
>
> **What it says:** batching "at the iteration level" removes finished requests and adds new ones after every step, so a new request waits "a single iteration".
>
> **Why it matters:** this is the mechanism behind the table. Because the batch is rebuilt at every step, the next request fills a freed slot at once, and the GPU stays close to the batch size in the first column.

**The bottom half is the bill.** From 8 to 128 in flight, throughput grew only about 2.2 times. Each user's time per token grew from 20.9 to 178.7 ms, so each stream fell to about 6 tokens per second. That is slower than many people read. The TTFT p99 grew from 1.00 s to 14.40 s.

Two things end the lunch. As the batch grows, the arithmetic and the KV cache reads grow with it, until they, not the weight read, set the step time. And every new request must be prefilled: 512 prompt tokens, in the same steps as everyone else's decodes. With more requests in flight, more prefills crowd into those steps. New requests wait longer for their first token, and every running stream stutters while a prefill runs.

> [!DEFINITION] Chunked prefill
> Splitting a long prompt's prefill into pieces, so that each step mixes a bounded amount of prefill work with the decodes of running requests. It keeps streams smooth while new requests start. vLLM does this by default.

> [!TIP] In production
> Choose the operating point from your targets, not from the biggest throughput number. A chat product needs a per-user speed (say 20 tokens per second, faster than reading) and a TTFT limit. The right concurrency is the highest one that still meets both. Here the targets are 20 tokens per second per user and a TTFT p99 under 1 second. The best row is then 4 requests in flight, at 195 output tokens per second. At 8 in flight, the TTFT p99 was 1,003 ms, just over the line.

## 6. Past capacity: where latency breaks

**The question:** how much traffic can the server take before users notice?

Real users do not wait for each other. They arrive at random times, whether the server keeps up or not.

> [!DEFINITION] Open loop
> A load test where requests arrive on a schedule, no matter how fast the server answers. Here they arrive at random times with a fixed average rate, a **Poisson process**. This is how real traffic behaves. [Part 6](llm-inference-6-serving-in-production.md) shows why a closed loop can hide a slow server.

The notebook takes the best request rate from Section 5, **5.22 requests per second**, as the server's capacity. Then it sends about a minute of traffic at 25%, 50%, 75%, 90%, 100% and 120% of that rate.

It also measures **goodput** against a latency target. We chose one that suits a chat product: TTFT under 1 second and TPOT under 100 ms, for each request.

> [!DEFINITION] SLO and goodput
> A **service level objective (SLO)** is a latency target, such as "first token within 1 second". **Goodput** is the number of requests per second that meet it. A server can have high throughput and low goodput, if it finishes everything but late.

> [!PAPER] Zhong et al., DistServe · Section 1 · page 2
> [![DistServe: an effective LLM serving system should balance these needs and maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned; higher per-GPU goodput directly translates into lower cost per query](/img/serving/paper-distserve-goodput.png)](/img/serving/paper-distserve-goodput.png)
>
> **Context:** the introduction of DistServe, a paper on serving prefill and decode on separate GPUs.
>
> **What it says:** a serving system should maximize per-GPU **goodput**, "the maximum request rate that can be served adhering to the SLO attainment goal".
>
> **Why it matters:** throughput counts every finished request. Goodput counts only the ones that were fast enough to be useful. The `--goodput` option of `vllm bench serve` measures exactly this, so the notebook can report it directly.

| Offered load | Arrival rate, req/s | Served, req/s | Goodput, req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|---|---|
| 25% | 1.30 | 1.25 | 1.25 | 164 ms | 337 ms | 20.4 ms | 27.3 ms |
| 50% | 2.61 | 2.48 | 2.48 | 182 ms | 434 ms | 28.7 ms | 46.6 ms |
| 75% | 3.91 | 3.62 | 3.62 | 268 ms | 586 ms | 56.1 ms | 72.0 ms |
| 90% | 4.70 | 4.23 | 3.02 | 383 ms | 1.03 s | 83.4 ms | 114.5 ms |
| 100% | 5.22 | 4.64 | 0.90 | 505 ms | 1.38 s | 145.7 ms | 175.2 ms |
| 120% | 6.26 | 5.01 | 0.00 | 1.41 s | 6.46 s | 277.3 ms | 357.3 ms |

<figure class="fig"><svg viewBox="0 0 760 396" role="img" aria-label="Arrival rate, served rate and goodput at six loads. Goodput peaks at 3.62 requests per second at 75% load and falls to 0.00 at 120%."><defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><rect class="box-3" x="110.0" y="12.0" width="18.0" height="11.0" rx="2"/><text class="t-tick" x="134.0" y="22.0" text-anchor="start">arrived</text><rect class="box-1" x="205.0" y="12.0" width="18.0" height="11.0" rx="2"/><text class="t-tick" x="229.0" y="22.0" text-anchor="start">served</text><rect class="box-2" x="293.0" y="12.0" width="18.0" height="11.0" rx="2"/><text class="t-tick" x="317.0" y="22.0" text-anchor="start">met the SLO (goodput)</text><text class="t-note" x="98.0" y="68.0" text-anchor="end">25% load</text><rect class="box-3" x="110.0" y="44.0" width="91.0" height="11.0" rx="2"/><rect class="box-1" x="110.0" y="57.0" width="87.5" height="11.0" rx="2"/><rect class="box-2" x="110.0" y="70.0" width="87.5" height="11.0" rx="2"/><text class="t-tick" x="209.0" y="66.0" text-anchor="start">goodput 1.25 req/s</text><text class="t-note" x="98.0" y="118.0" text-anchor="end">50% load</text><rect class="box-3" x="110.0" y="94.0" width="182.7" height="11.0" rx="2"/><rect class="box-1" x="110.0" y="107.0" width="173.9" height="11.0" rx="2"/><rect class="box-2" x="110.0" y="120.0" width="173.9" height="11.0" rx="2"/><text class="t-tick" x="300.7" y="116.0" text-anchor="start">goodput 2.48 req/s</text><text class="t-note" x="98.0" y="168.0" text-anchor="end">75% load</text><rect class="box-3" x="110.0" y="144.0" width="273.7" height="11.0" rx="2"/><rect class="box-1" x="110.0" y="157.0" width="253.7" height="11.0" rx="2"/><rect class="box-2" x="110.0" y="170.0" width="253.7" height="11.0" rx="2"/><text class="t-tick" x="391.7" y="166.0" text-anchor="start">goodput 3.62 req/s</text><text class="t-note" x="98.0" y="218.0" text-anchor="end">90% load</text><rect class="box-3" x="110.0" y="194.0" width="329.0" height="11.0" rx="2"/><rect class="box-1" x="110.0" y="207.0" width="296.4" height="11.0" rx="2"/><rect class="box-2" x="110.0" y="220.0" width="211.3" height="11.0" rx="2"/><text class="t-tick" x="447.0" y="216.0" text-anchor="start">goodput 3.02 req/s</text><text class="t-note" x="98.0" y="268.0" text-anchor="end">100% load</text><rect class="box-3" x="110.0" y="244.0" width="365.4" height="11.0" rx="2"/><rect class="box-1" x="110.0" y="257.0" width="324.6" height="11.0" rx="2"/><rect class="box-2" x="110.0" y="270.0" width="63.3" height="11.0" rx="2"/><text class="t-tick" x="483.4" y="266.0" text-anchor="start">goodput 0.90 req/s</text><text class="t-note" x="98.0" y="318.0" text-anchor="end">120% load</text><rect class="box-3" x="110.0" y="294.0" width="438.2" height="11.0" rx="2"/><rect class="box-1" x="110.0" y="307.0" width="350.9" height="11.0" rx="2"/><rect class="box-2" x="110.0" y="320.0" width="0.0" height="11.0" rx="2"/><text class="t-tick" x="556.2" y="316.0" text-anchor="start">goodput 0.00 req/s</text><line class="axis" x1="110.0" y1="344.0" x2="670.0" y2="344.0"/><text class="t-tick" x="110.0" y="362.0" text-anchor="middle">0</text><text class="t-tick" x="250.0" y="362.0" text-anchor="middle">2</text><text class="t-tick" x="390.0" y="362.0" text-anchor="middle">4</text><text class="t-tick" x="530.0" y="362.0" text-anchor="middle">6</text><text class="t-tick" x="670.0" y="362.0" text-anchor="middle">8</text><text class="t-tick" x="390.0" y="382.0" text-anchor="middle">requests per second</text></svg><figcaption>For each load: how many requests per second arrived, how many the server finished, and how many met the SLO (TTFT under 1 s and TPOT under 100 ms).</figcaption></figure>

Read the goodput column from the top. Up to 75% load, every request met the SLO. At 90%, about 29% of them missed it. At 100%, goodput fell to 0.90 requests per second. At 120%, it was 0.00: the server still finished 5.01 requests per second, and not one of them was fast enough.

So the useful capacity of this server is not 5.22 requests per second. It is about 3.62, the goodput at 75% load.

Now look at which target broke. At 100% load, the TTFT p50 was still 505 ms, well under 1 second. The TPOT p50 was 146 ms, far over 100. **The streams broke before the first tokens did.**

<figure class="fig"><svg viewBox="0 0 760 300" role="img" aria-label="TTFT against offered load: p50 from 164 ms at 25% to 1.41 s at 120%; p99 from 337 ms to 6.46 s."><defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="90.0" y1="250.0" x2="710.0" y2="250.0"/><text class="t-tick" x="82.0" y="254.0" text-anchor="end">100 ms</text><line class="grid" x1="90.0" y1="199.9" x2="710.0" y2="199.9"/><text class="t-tick" x="82.0" y="203.9" text-anchor="end">300 ms</text><line class="grid" x1="90.0" y1="145.0" x2="710.0" y2="145.0"/><text class="t-tick" x="82.0" y="149.0" text-anchor="end">1 s</text><line class="grid" x1="90.0" y1="94.9" x2="710.0" y2="94.9"/><text class="t-tick" x="82.0" y="98.9" text-anchor="end">3 s</text><line class="grid" x1="90.0" y1="40.0" x2="710.0" y2="40.0"/><text class="t-tick" x="82.0" y="44.0" text-anchor="end">10 s</text><text class="t-tick" x="119.5" y="268.0" text-anchor="middle">25%</text><text class="t-tick" x="267.1" y="268.0" text-anchor="middle">50%</text><text class="t-tick" x="414.8" y="268.0" text-anchor="middle">75%</text><text class="t-tick" x="503.3" y="268.0" text-anchor="middle">90%</text><text class="t-tick" x="562.4" y="268.0" text-anchor="middle">100%</text><text class="t-tick" x="680.5" y="268.0" text-anchor="middle">120%</text><line class="axis" x1="90.0" y1="250.0" x2="710.0" y2="250.0"/><text class="t-tick" x="400.0" y="288.0" text-anchor="middle">offered load, as a share of measured capacity</text><text class="t-tick" x="82.0" y="28.0" text-anchor="start">TTFT (log scale)</text><line class="drop" x1="562.4" y1="40.0" x2="562.4" y2="250.0"/><text class="t-tick" x="568.4" y="52.0" text-anchor="start">capacity</text><polyline class="l1" points="119.5,227.5 267.1,222.7 414.8,205.1 503.3,188.8 562.4,176.1 680.5,129.4"/><polyline class="l2" points="119.5,194.6 267.1,183.1 414.8,169.3 503.3,143.6 562.4,130.2 680.5,59.9"/><g class="mark"><title>25% load: TTFT p50 164 ms</title><circle class="s1" cx="119.5" cy="227.5" r="4"/></g><g class="mark"><title>50% load: TTFT p50 182 ms</title><circle class="s1" cx="267.1" cy="222.7" r="4"/></g><g class="mark"><title>75% load: TTFT p50 268 ms</title><circle class="s1" cx="414.8" cy="205.1" r="4"/></g><g class="mark"><title>90% load: TTFT p50 383 ms</title><circle class="s1" cx="503.3" cy="188.8" r="4"/></g><g class="mark"><title>100% load: TTFT p50 505 ms</title><circle class="s1" cx="562.4" cy="176.1" r="4"/></g><g class="mark"><title>120% load: TTFT p50 1.41 s</title><circle class="s1" cx="680.5" cy="129.4" r="4"/></g><g class="mark"><title>25% load: TTFT p99 337 ms</title><circle class="s2" cx="119.5" cy="194.6" r="4"/></g><g class="mark"><title>50% load: TTFT p99 434 ms</title><circle class="s2" cx="267.1" cy="183.1" r="4"/></g><g class="mark"><title>75% load: TTFT p99 586 ms</title><circle class="s2" cx="414.8" cy="169.3" r="4"/></g><g class="mark"><title>90% load: TTFT p99 1.03 s</title><circle class="s2" cx="503.3" cy="143.6" r="4"/></g><g class="mark"><title>100% load: TTFT p99 1.38 s</title><circle class="s2" cx="562.4" cy="130.2" r="4"/></g><g class="mark"><title>120% load: TTFT p99 6.46 s</title><circle class="s2" cx="680.5" cy="59.9" r="4"/></g><line class="l1" x1="420.0" y1="16.0" x2="442.0" y2="16.0"/><text class="t-tick" x="448.0" y="20.0" text-anchor="start">TTFT p50</text><line class="l2" x1="516.0" y1="16.0" x2="538.0" y2="16.0"/><text class="t-tick" x="544.0" y="20.0" text-anchor="start">TTFT p99</text></svg><figcaption>TTFT p50 and p99 against the offered load, as a share of the capacity measured in Section 5. The vertical axis is logarithmic.</figcaption></figure>

TTFT did break too, one step later. At 120%, the TTFT p99 was 6.46 s. Above capacity, more requests arrive each second than leave, so the backlog grows for as long as the overload lasts.

### Inside the server during the overload

During the 120% run, the notebook read `/metrics` every half second. It recorded three numbers that a production dashboard shows: requests running, requests waiting, and the share of the KV cache in use.

<figure class="fig"><svg viewBox="0 0 760 290" role="img" aria-label="During the overload, up to 256 requests ran and up to 25 waited; KV cache use peaked at 48%."><defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><line class="grid" x1="70.0" y1="240.0" x2="690.0" y2="240.0"/><text class="t-tick" x="62.0" y="244.0" text-anchor="end">0</text><line class="grid" x1="70.0" y1="190.0" x2="690.0" y2="190.0"/><text class="t-tick" x="62.0" y="194.0" text-anchor="end">75</text><line class="grid" x1="70.0" y1="140.0" x2="690.0" y2="140.0"/><text class="t-tick" x="62.0" y="144.0" text-anchor="end">150</text><line class="grid" x1="70.0" y1="90.0" x2="690.0" y2="90.0"/><text class="t-tick" x="62.0" y="94.0" text-anchor="end">225</text><line class="grid" x1="70.0" y1="40.0" x2="690.0" y2="40.0"/><text class="t-tick" x="62.0" y="44.0" text-anchor="end">300</text><text class="t-tick" x="70.0" y="258.0" text-anchor="middle">0</text><text class="t-tick" x="173.3" y="258.0" text-anchor="middle">16</text><text class="t-tick" x="276.7" y="258.0" text-anchor="middle">31</text><text class="t-tick" x="380.0" y="258.0" text-anchor="middle">47</text><text class="t-tick" x="483.3" y="258.0" text-anchor="middle">62</text><text class="t-tick" x="586.7" y="258.0" text-anchor="middle">78</text><text class="t-tick" x="690.0" y="258.0" text-anchor="middle">93</text><line class="axis" x1="70.0" y1="240.0" x2="690.0" y2="240.0"/><text class="t-tick" x="380.0" y="278.0" text-anchor="middle">seconds since the run started</text><text class="t-tick" x="62.0" y="28.0" text-anchor="start">requests</text><line class="drop" x1="70.0" y1="69.3" x2="690.0" y2="69.3"/><text class="t-tick" x="80.0" y="63.3" text-anchor="start">most running at once: 256</text><polyline class="l1" points="70.1,240.0 73.5,240.0 77.0,240.0 80.4,240.0 83.8,240.0 87.2,240.0 90.6,240.0 94.1,240.0 97.4,240.0 100.9,240.0 104.3,240.0 107.7,240.0 111.1,240.0 114.6,240.0 118.0,240.0 121.4,240.0 124.8,240.0 128.3,240.0 131.7,240.0 135.3,240.0 138.9,240.0 142.4,240.0 145.9,240.0 149.4,240.0 152.9,240.0 156.5,240.0 159.9,240.0 163.3,240.0 166.7,240.0 170.2,240.0 173.5,240.0 176.9,240.0 180.4,240.0 183.8,240.0 187.3,240.0 190.7,240.0 194.1,240.0 197.5,237.3 201.0,236.0 204.4,233.3 207.8,230.0 211.2,228.0 214.6,226.7 218.0,224.0 221.5,220.7 224.9,218.7 228.4,216.0 231.9,215.3 235.4,213.3 238.7,210.0 242.2,210.0 245.6,208.0 249.0,206.0 252.4,204.0 255.8,202.7 259.1,200.7 262.5,196.0 265.9,194.7 269.3,192.0 272.7,189.3 276.1,186.7 279.5,181.3 282.9,180.0 286.5,180.0 289.8,178.0 293.2,176.0 296.6,174.0 300.0,172.0 303.4,168.0 306.8,165.3 310.3,164.0 313.7,162.0 317.5,162.0 320.9,158.7 324.4,156.0 328.2,154.0 331.7,150.7 335.1,149.3 338.6,146.7 342.0,144.7 345.3,142.7 348.8,142.7 352.3,142.7 355.6,142.7 359.3,141.3 362.7,140.0 366.1,138.0 369.5,136.7 373.3,134.0 376.8,130.7 380.2,130.0 383.6,127.3 386.9,126.0 390.3,123.3 393.8,121.3 397.2,119.3 400.6,117.3 404.0,116.7 407.4,114.0 410.8,112.7 414.3,111.3 417.6,110.0 421.1,108.0 424.5,107.3 428.0,104.7 431.6,104.7 435.0,102.0 438.4,100.0 441.8,98.7 445.4,96.0 448.8,96.0 452.1,94.0 455.5,91.3 458.9,88.7 462.3,86.7 465.7,83.3 469.1,81.3 472.5,78.7 475.9,76.7 479.3,74.0 483.0,74.0 486.8,72.7 490.6,70.7 494.0,70.0 497.4,70.0 500.9,69.3 504.3,69.3 507.8,69.3 511.6,69.3 515.0,70.0 518.4,70.0 521.9,69.3 525.7,70.7 529.2,70.7 532.6,69.3 536.1,69.3 539.5,70.0 542.9,70.0 546.3,70.7 549.7,71.3 553.0,71.3 556.4,71.3 559.8,72.0 563.2,72.0 566.6,72.0 570.0,72.7 573.4,72.0 576.8,70.7 580.1,69.3 583.6,70.0 587.0,70.0 590.4,70.7 593.8,69.3 597.2,70.7 600.5,70.0 604.0,71.3 607.4,70.0 610.9,71.3 614.6,71.3 618.1,69.3 621.5,71.3 624.9,76.7 628.4,83.3 631.9,89.3 635.3,94.0 638.7,97.3 642.1,102.0 645.7,110.7 649.0,116.0 652.4,124.0 656.0,136.0 659.4,148.0 662.8,160.0 666.1,177.3 669.5,190.7 673.0,190.7 676.4,193.3 679.8,196.7 683.2,201.3 686.5,222.0 690.0,240.0"/><polyline class="l2" points="70.1,240.0 73.5,240.0 77.0,240.0 80.4,240.0 83.8,240.0 87.2,240.0 90.6,240.0 94.1,240.0 97.4,240.0 100.9,240.0 104.3,240.0 107.7,240.0 111.1,240.0 114.6,240.0 118.0,240.0 121.4,240.0 124.8,240.0 128.3,240.0 131.7,240.0 135.3,240.0 138.9,240.0 142.4,240.0 145.9,240.0 149.4,240.0 152.9,240.0 156.5,240.0 159.9,240.0 163.3,240.0 166.7,240.0 170.2,240.0 173.5,240.0 176.9,240.0 180.4,240.0 183.8,240.0 187.3,240.0 190.7,240.0 194.1,240.0 197.5,240.0 201.0,240.0 204.4,240.0 207.8,240.0 211.2,240.0 214.6,240.0 218.0,240.0 221.5,240.0 224.9,240.0 228.4,240.0 231.9,240.0 235.4,240.0 238.7,240.0 242.2,240.0 245.6,240.0 249.0,240.0 252.4,240.0 255.8,240.0 259.1,240.0 262.5,240.0 265.9,240.0 269.3,239.3 272.7,240.0 276.1,238.7 279.5,239.3 282.9,240.0 286.5,240.0 289.8,240.0 293.2,240.0 296.6,240.0 300.0,240.0 303.4,240.0 306.8,240.0 310.3,240.0 313.7,240.0 317.5,240.0 320.9,240.0 324.4,240.0 328.2,240.0 331.7,240.0 335.1,240.0 338.6,240.0 342.0,240.0 345.3,240.0 348.8,240.0 352.3,240.0 355.6,240.0 359.3,240.0 362.7,240.0 366.1,240.0 369.5,240.0 373.3,239.3 376.8,240.0 380.2,240.0 383.6,240.0 386.9,240.0 390.3,240.0 393.8,240.0 397.2,236.0 400.6,236.7 404.0,236.0 407.4,234.7 410.8,234.0 414.3,234.0 417.6,233.3 421.1,234.7 424.5,231.3 428.0,232.0 431.6,232.0 435.0,231.3 438.4,234.0 441.8,234.0 445.4,234.7 448.8,234.7 452.1,235.3 455.5,238.0 458.9,239.3 462.3,240.0 465.7,240.0 469.1,238.0 472.5,237.3 475.9,237.3 479.3,238.0 483.0,238.0 486.8,239.3 490.6,238.0 494.0,236.7 497.4,236.7 500.9,237.3 504.3,234.0 507.8,230.0 511.6,228.7 515.0,228.0 518.4,229.3 521.9,228.7 525.7,227.3 529.2,228.0 532.6,226.7 536.1,224.0 539.5,223.3 542.9,225.3 546.3,225.3 549.7,225.3 553.0,225.3 556.4,227.3 559.8,226.0 563.2,228.0 566.6,228.0 570.0,228.7 573.4,230.7 576.8,231.3 580.1,229.3 583.6,226.0 587.0,226.0 590.4,225.3 593.8,226.7 597.2,228.0 600.5,229.3 604.0,232.0 607.4,234.0 610.9,234.7 614.6,237.3 618.1,239.3 621.5,240.0 624.9,240.0 628.4,240.0 631.9,240.0 635.3,240.0 638.7,240.0 642.1,240.0 645.7,240.0 649.0,240.0 652.4,240.0 656.0,240.0 659.4,240.0 662.8,240.0 666.1,240.0 669.5,240.0 673.0,240.0 676.4,240.0 679.8,240.0 683.2,240.0 686.5,240.0 690.0,240.0"/><polyline class="l3" points="70.1,240.0 73.5,240.0 77.0,240.0 80.4,240.0 83.8,240.0 87.2,240.0 90.6,240.0 94.1,240.0 97.4,240.0 100.9,240.0 104.3,240.0 107.7,240.0 111.1,240.0 114.6,240.0 118.0,240.0 121.4,240.0 124.8,240.0 128.3,240.0 131.7,240.0 135.3,240.0 138.9,240.0 142.4,240.0 145.9,240.0 149.4,240.0 152.9,240.0 156.5,240.0 159.9,240.0 163.3,240.0 166.7,240.0 170.2,240.0 173.5,240.0 176.9,240.0 180.4,240.0 183.8,240.0 187.3,240.0 190.7,240.0 194.1,240.0 197.5,238.7 201.0,238.0 204.4,236.6 207.8,234.9 211.2,233.9 214.6,233.1 218.0,231.7 221.5,230.0 224.9,229.0 228.4,227.6 231.9,227.0 235.4,225.9 238.7,224.1 242.2,223.9 245.6,222.7 249.0,221.5 252.4,220.5 255.8,219.5 259.1,218.4 262.5,216.1 265.9,215.3 269.3,214.0 272.7,212.7 276.1,211.4 279.5,208.7 282.9,207.8 286.5,207.7 289.8,206.6 293.2,205.6 296.6,204.4 300.0,203.2 303.4,201.1 306.8,199.6 310.3,199.0 313.7,198.0 317.5,197.8 320.9,196.1 324.4,194.8 328.2,193.6 331.7,191.9 335.1,190.9 338.6,189.5 342.0,188.6 345.3,187.7 348.8,187.5 352.3,187.5 355.6,187.6 359.3,186.8 362.7,186.2 366.1,185.0 369.5,184.3 373.3,183.0 376.8,181.1 380.2,180.6 383.6,179.1 386.9,178.3 390.3,177.0 393.8,175.8 397.2,174.9 400.6,174.1 404.0,173.6 407.4,172.3 410.8,171.8 414.3,170.8 417.6,170.3 421.1,169.4 424.5,168.8 428.0,167.5 431.6,167.5 435.0,166.1 438.4,165.2 441.8,164.3 445.4,163.0 448.8,163.0 452.1,161.7 455.5,160.4 458.9,159.1 462.3,157.8 465.7,156.5 469.1,155.1 472.5,153.8 475.9,152.4 479.3,151.1 483.0,151.1 486.8,150.1 490.6,148.9 494.0,148.5 497.4,148.6 500.9,148.1 504.3,147.7 507.8,147.2 511.6,146.4 515.0,146.4 518.4,146.3 521.9,145.7 525.7,145.9 529.2,145.8 532.6,144.9 536.1,144.1 539.5,144.3 542.9,144.2 546.3,144.6 549.7,144.9 553.0,145.0 556.4,145.1 559.8,145.6 563.2,145.9 566.6,146.2 570.0,146.4 573.4,146.3 576.8,145.4 580.1,144.5 583.6,144.7 587.0,144.6 590.4,145.0 593.8,144.1 597.2,144.8 600.5,144.4 604.0,145.1 607.4,144.4 610.9,145.1 614.6,145.0 618.1,143.9 621.5,145.0 624.9,147.9 628.4,151.4 631.9,154.4 635.3,156.8 638.7,158.1 642.1,160.5 645.7,165.3 649.0,168.1 652.4,172.6 656.0,179.6 659.4,186.4 662.8,193.4 666.1,203.8 669.5,211.5 673.0,211.1 676.4,212.4 679.8,214.0 683.2,216.6 686.5,229.0 690.0,240.0"/><text class="t-tick" x="700.0" y="244.0" text-anchor="start">0%</text><text class="t-tick" x="700.0" y="144.0" text-anchor="start">50%</text><text class="t-tick" x="700.0" y="44.0" text-anchor="start">100%</text><text class="t-tick" x="745.0" y="32.0" text-anchor="end">KV used</text><line class="l1" x1="170.0" y1="16.0" x2="192.0" y2="16.0"/><text class="t-tick" x="198.0" y="20.0" text-anchor="start">running</text><line class="l2" x1="259.0" y1="16.0" x2="281.0" y2="16.0"/><text class="t-tick" x="287.0" y="20.0" text-anchor="start">waiting</text><line class="l3" x1="348.0" y1="16.0" x2="370.0" y2="16.0"/><text class="t-tick" x="376.0" y="20.0" text-anchor="start">KV cache used (right scale)</text></svg><figcaption>Requests running and waiting inside the server, and the share of the KV cache in use, during the run at 120% of capacity. The first 19 seconds are the load tool starting up.</figcaption></figure>

The picture is not the one most people expect. The `running` line climbed for about 45 seconds, then flattened at **256**. Until then, the `waiting` line stayed short: about 2 on average. Its peak, **25**, came only after the batch was full. The KV cache never passed **48.0%** full.

Here is why. vLLM admits a new request whenever there is memory for it. So in an overload, the backlog does not sit in the queue at first. It sits inside the batch.

The batch grows, every step gets longer, and every user's stream slows down. That is the TPOT column above. Only when the batch hits the scheduler's cap does the queue really grow. In this run that cap was 256 requests, which matches vLLM's `--max-num-seqs` limit.

That also checks the sizing sum from Section 4. 256 requests of at most 640 tokens need at most 163,840 tokens, about half of the 313,920 in the cache. The gauge showed 48.0%.

> [!DEFINITION] `--max-num-seqs`
> The most requests vLLM will run in one batch. When it is reached, new requests wait in the queue even if the KV cache has room.

> [!WARNING] On vLLM, overload shows up as slow streams first
> A waiting queue near zero does not mean the server is fine. Here the queue stayed small while TPOT went past the SLO. Watch three things together: `vllm:num_requests_running` against your `--max-num-seqs`, the time between tokens (`vllm:inter_token_latency_seconds`), and `vllm:num_requests_waiting`. Alert on TPOT p99 and TTFT p99, not on GPU use.

> [!TIP] In production
> Keep each server below the knee. Here goodput held up to 75% of the closed-loop capacity and fell apart above 90%. Add replicas before you reach that point, not after. Some products would rather make users wait for the first token than give everyone a slow stream. For them, lower `--max-num-seqs`: the backlog then moves from the batch to the queue. [Part 6](llm-inference-6-serving-in-production.md) explains the queueing maths behind the knee.

## 7. Prefix caching: pay for the shared prompt once

**The question:** how much does reusing a shared start of the prompt save?

Many products send the same long start with every request: a system prompt, tool definitions, a document. If the server keeps the KV cache of that start, later requests can skip its prefill and compute only what is new.

> [!DEFINITION] Prefix caching
> Keeping the KV cache of earlier prompts and reusing it when a new prompt starts with the same tokens. vLLM finds matches in whole blocks. It names each block by a hash of its tokens and of every token before them. [Part 5](llm-inference-5-sglang-vs-vllm.md) covers how vLLM and SGLang do it.

The workload: every request starts with the same 2,048 random tokens and adds 128 tokens of its own. It asks for 64 output tokens, with 8 requests in flight. We run it on a normal server, where prefix caching is on by default. Then we restart with `--no-enable-prefix-caching` and run it again.

<figure class="fig"><svg viewBox="0 0 760 210" role="img" aria-label="Prefix caching off: TTFT p50 3.71 s, p99 6.81 s. On: p50 735 ms, p99 1.85 s."><defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs><text class="t-note" x="138.0" y="70.0" text-anchor="end">cache off</text><rect class="box-1" x="150.0" y="40.0" width="250.5" height="22.0" rx="4"/><text class="t-tick" x="408.5" y="56.0" text-anchor="start">p50 3.71 s</text><rect class="box-2" x="150.0" y="68.0" width="459.4" height="22.0" rx="4"/><text class="t-tick" x="617.4" y="84.0" text-anchor="start">p99 6.81 s</text><text class="t-note" x="138.0" y="150.0" text-anchor="end">cache on</text><rect class="box-1" x="150.0" y="120.0" width="49.6" height="22.0" rx="4"/><text class="t-tick" x="207.6" y="136.0" text-anchor="start">p50 735 ms</text><rect class="box-2" x="150.0" y="148.0" width="125.1" height="22.0" rx="4"/><text class="t-tick" x="283.1" y="164.0" text-anchor="start">p99 1.85 s</text><text class="t-title" x="20.0" y="22.0" text-anchor="start">Time to first token, shared 2,048-token start + 128 own tokens</text></svg><figcaption>TTFT p50 and p99 with prefix caching off and on, for prompts with a shared 2,048-token start.</figcaption></figure>

| Prefix caching | Cache hit rate | TTFT p50 | TTFT p99 | TPOT p50 | Requests per second |
|---|---|---|---|---|---|
| off | (off) | 3.71 s | 6.81 s | 99.0 ms | 0.81 |
| on | 93.5% | 735 ms | 1.85 s | 29.2 ms | 3.05 |

The hit rate comes from the server's own counters: `vllm:prefix_cache_hits_total` divided by `vllm:prefix_cache_queries_total`, read before and after the run. With the cache on, the server found **93.5%** of the prompt tokens it looked up already in the cache.

The median TTFT fell from about 3.7 s to 735 ms, **about 5 times** faster. The server finished about 4 times as many requests per second. Even the streams got faster, from 99 to 29 ms per token, because fewer long prefills crowded into the decode steps.

Why so large? Without the cache, every request prefills about 2,176 tokens. With it, the first request pays for the shared 2,048, and every later one prefills only its own 128. On a T4, where prefill runs on the slower Triton attention kernel, every prefill token saved counts even more.

> [!TIP] In production: when prefix caching pays
> It pays when the shared start is long next to the part that changes, as here (2,048 shared tokens, 128 new). It pays nothing when every prompt is different. Put the parts of a prompt that never change first, and the parts that do change last, such as the user's question or the date. One changed token early in the prompt makes everything after it a miss.
>
> With many replicas, each server has its own cache, so a router that sends similar prompts to the same server keeps the hit rate high. Watch the hit rate on `/metrics`. If it is near zero when you expected reuse, look for a changing token near the start.

## 8. A KV cache too small: preemption

**The question:** what happens when the KV cache runs out?

Section 4 showed that this server has room for hundreds of requests. Production servers often do not. Bigger models, longer prompts and longer answers fill the cache quickly.

vLLM starts a request when there is room for its prompt. As the answer grows, the request needs new blocks. When a running request needs a block and none is free, the scheduler must take memory from someone.

> [!DEFINITION] Preemption
> Stopping a running request to free its memory, and resuming it later. In vLLM's current engine, the preempted request drops its whole KV cache. Later it rebuilds the cache with one prefill over its prompt and the tokens it has written so far.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §4.5 · page 8
> [![The start of section 4.5, with output lengths are not known a priori and can run out of the GPU's physical blocks highlighted](/img/writings/llm-inference/p3-preempt.png)](/img/writings/llm-inference/p3-preempt.png)
>
> **Context:** the vLLM paper's section on scheduling when memory runs out.
>
> **What it says:** because "output lengths are not known a priori", a server that admits requests without reserving room for their whole answer "can run out of the GPU's physical blocks".
>
> **Why it matters:** admitting requests without a full reservation is what lets paged memory fit so many of them. It is a bet that they will not all grow long at the same time. The experiment below makes the bet lose on purpose, to see what losing costs.

The notebook restarts the server with `--kv-cache-memory-bytes` set to fit only **6,144 tokens**. Then it sends 32 requests at a time, each with 512 prompt tokens and 512 output tokens. Together they would need 32,768 tokens of cache, more than five times what there is. The same job also runs on a normal server, for comparison.

| Server | KV cache, tokens | Preemptions | Output tok/s | TTFT p50 | TTFT p99 | TPOT p50 | End-to-end p99 |
|---|---|---|---|---|---|---|---|
| normal | 320,768 | 0 | 714 | 1.27 s | 4.43 s | 41.5 ms | 25.44 s |
| small KV | 6,144 | 70 | 320 | 35.82 s | 43.33 s | 22.6 ms | 57.58 s |

The small server ran out of room 70 times. Each time, a request lost its cache and had to rebuild it, so part of the GPU's work was done twice. Throughput fell to 44.8% of the normal server's, a bit less than half.

The bigger effect is on waiting. The cache has room for only about 6 requests of 1,024 tokens at once. So most of the 32 requests in flight sit in the queue. The median TTFT went from about 1.3 s to about 36 s.

Now look at the TPOT column. The small server's streams were *faster*: 22.6 ms per token against 41.5. Only a few requests ran at a time, so each step was short. A dashboard that showed only TPOT would say this server was healthier. Its users waited more than half a minute for a first word.

> [!WARNING] A rising preemption counter is an alarm
> `vllm:num_preemptions_total` should stay flat. If it rises, the KV cache is too small for the traffic. The fixes, cheapest first:
>
> 1. Raise `--gpu-memory-utilization`.
> 2. Lower `--max-model-len` if no request needs it.
> 3. Lower `--max-num-seqs`, so fewer requests compete for blocks.
> 4. Use quantized weights or an FP8 KV cache, so more fits.
> 5. Add replicas.

## 9. Break it on purpose

**The question:** what does failure look like from the client's side?

Production skill is knowing what a failure looks like before it happens at 3 a.m. The notebook tries four.

### Clean failures: bad requests

| What we sent | Status | What the server said |
|---|---|---|
| The word `hello` 4,596 times, over the 4,096-token limit | 400 | This model's maximum context length is 4096 tokens. However, your prompt contains at least 4097 input tokens, which exceeds this limit even without any output t |
| The wrong API key | 401 | Unauthorized |
| `max_tokens` of 4,106 | 400 | max_tokens=4106 cannot be greater than max_model_len=4096. Please request fewer output tokens. (parameter=max_tokens, value=4106) |

These are good failures. The server rejects the bad request at once, says why, and keeps serving everyone else. A client should treat any 4xx status as "my request is wrong": fix it, do not retry it.

> [!DEFINITION] Status code
> The three-digit number at the start of every HTTP answer. 200 means success. 4xx means the request was wrong (400 a bad request, 401 a missing or wrong key). 5xx means the server failed.

Two of these protect the server. The 4,096-token limit from `--max-model-len` stops one huge prompt from taking the KV cache from everyone. The API key stops strangers from using your GPU. Note that `/health` and `/metrics` stay open without the key, so keep the port itself private.

### The ugly failure: the server is gone

The notebook started a load test with 32 requests in flight in a background thread. After 15 seconds it killed the server with `SIGKILL`, the way an out-of-memory killer or a failed machine would.

| | Result |
|---|---|
| Requests completed | 0 |
| Requests failed | 200 |
| Time until all of them had failed | 0.7 s |
| `/health` after the kill | ConnectionError (no answer at all) |

One honest note about this run. The load tool spent more than 15 seconds starting up, so the kill came before its first request. The server log confirms that no request from the test ever reached it. So this drill measured a server that was already gone, not one that died halfway through an answer. Every one of the 200 requests failed within 0.7 s, with a connection error.

That fast failure is the better case. When a server dies in the middle of an answer, the requests in flight are lost too. Their KV caches lived in the GPU memory of a process that no longer exists, so nothing inside vLLM can save them. The client sees a stream that stops partway.

That is why production setups never rely on one server. They run several **replicas** behind a load balancer that calls `/health` and stops routing to a dead one. Clients retry failed requests with **exponential backoff** and **jitter**: they wait a little, then longer, with a random part added. Then a thousand clients do not all retry at the same moment.

> [!DEFINITION] Replica
> One copy of the server, with its own GPU and its own copy of the model. Several replicas behind a load balancer let one fail without taking the service down.

> [!DEFINITION] Exponential backoff with jitter
> A retry rule: wait about 1 second before the first retry, 2 before the second, 4 before the third, and so on. Add a random amount each time. It gives a failing service room to recover instead of hitting it with every retry at once.

> [!WARNING] Retries can keep a server down
> A server that comes back after a crash meets every client's retry at once, plus the new traffic. Without backoff, that burst can push it straight past capacity again. And remember the cold start from Section 2: a replacement server here needed more than a minute even when warm. [Part 6](llm-inference-6-serving-in-production.md) simulates this retry storm.

## 10. FP8 weights: skipped on the T4

**The question:** what do 8-bit weights buy?

**FP8** stores each weight in 8 bits instead of 16. vLLM can convert the weights while it loads them (`--quantization fp8`). The weights then take half the memory, so the KV cache grows, and each decode step reads half the bytes.

The notebook skips this step on the T4. FP8 needs a GPU with compute capability 8.0 or newer, and the T4 is 7.5. On an L4 the step runs, and on an A100 it runs on a slower path that saves memory but may not save time. We have no measured FP8 numbers for this page, so it shows none. [Part 6](llm-inference-6-serving-in-production.md) measures quantization on a real model.

## 11. Save everything, and run it yourself

The last step writes `vllm_colab_results.json`, with every number above plus the GPU and the versions. It also writes a zip with every server log and raw benchmark file. This page was built from exactly those two files.

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb)

1. Open the notebook in Colab and choose **Runtime > Change runtime type > T4 GPU**.
2. Run every cell from the top with **Shift + Enter**. The whole run takes about an hour on a T4. Set `QUICK = True` in the settings cell to cut it to about 25 minutes, with noisier numbers.
3. The last cell downloads `vllm_colab_results.json` and `vllm_colab_run.zip`. Run it after any experiment you care about: if Colab disconnects, everything in memory is gone.

Then change one thing and run it again. Try `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-3B-Instruct`, and see how the KV cache budget and the knee move. Try an L4: the attention backend changes, and step 10 runs.

## Running vLLM in production: what this run teaches

Every lesson above, in one place.

**The settings that matter.**

| Setting | What it controls | What we saw |
|---|---|---|
| `--gpu-memory-utilization` | The share of GPU memory vLLM may take; all of it beyond weights and working memory becomes KV cache | 0.85 gave 8.38 GiB of KV cache |
| `--max-model-len` | The longest prompt plus answer for one request | Longer requests were rejected with a 400 |
| `--max-num-seqs` | The most requests in one batch | The batch stopped at 256, and only then did a queue form |
| `--dtype` | The number format of weights and cache | `half` on the T4, which has no bfloat16 |
| `--api-key` | A key on every `/v1` request | A wrong key got a 401 |
| `--no-enable-prefix-caching` | Turns off prefix caching, which is on by default | TTFT p50 about 5 times slower with a shared prompt |
| `--kv-cache-memory-bytes` | A fixed KV cache size, in place of the share above | A small cache gave 70 preemptions |

**What to watch on `/metrics`.**

| Metric | What it tells you | The warning sign in this run |
|---|---|---|
| `vllm:num_requests_running` | Requests in the batch now | Stuck at the `--max-num-seqs` cap (256) |
| `vllm:num_requests_waiting` | Requests in the queue | Above zero once the batch was full |
| `vllm:kv_cache_usage_perc` | Share of KV blocks in use | Near 100% together with preemptions |
| `vllm:num_preemptions_total` | Requests stopped to free memory | Any steady rise (70 in one small-cache run) |
| `vllm:prefix_cache_hits_total` ÷ `vllm:prefix_cache_queries_total` | Share of prompt tokens reused | Near zero when you expected reuse |
| `vllm:time_to_first_token_seconds` | TTFT, as a histogram | p99 over your SLO |
| `vllm:inter_token_latency_seconds` | The gaps between tokens | p99 over your SLO: on vLLM this breaks first |

**How to size the KV cache.** Multiply the requests you expect at once by their longest prompt plus answer, and compare with the log's "GPU KV cache size" line. Check the result on the running server: the KV gauge should match the sum.

**Where latency breaks.** Measure capacity with a closed loop. Then find the knee with an open loop and an SLO, and plan with goodput, not throughput. Here the closed loop said 5.22 requests per second, and the SLO said about 3.62.

**When prefix caching pays.** When many requests share a long start. Measure the hit rate, and put the fixed parts of the prompt first.

**What preemption looks like.** A rising preemption counter, a full KV cache, long waits for the first token, and streams that look fine.

**How it fails.** Bad requests fail cleanly with a 4xx. A dead server fails everything, fast. Run replicas behind health checks, and retry with backoff.

## All the results in one table

| What | Measured value |
|---|---|
| Setup | Tesla T4 (15.0 GiB), vLLM 0.33.0, Qwen/Qwen2.5-1.5B-Instruct, float16, max_model_len 4,096, gpu_memory_utilization 0.85 |
| Attention backend | TRITON_ATTN |
| Cold start / warm start | 294.7 s / 79.2 s |
| Weights in GPU memory | 2.98 GiB |
| KV cache (log) | 8.38 GiB = 313,920 tokens = 76.64 requests of 4,096 tokens |
| KV cache (by hand) | 28,672 bytes per token, predicts 313,824 tokens |
| One user, streaming | TTFT about 32 ms, 14.7 ms per token |
| 1 to 128 requests in flight | 61 to 668 output tok/s (11.0 times); TPOT p50 15.6 to 178.7 ms |
| Capacity used for the load test | 5.22 req/s (best closed-loop rate) |
| Best goodput (TTFT under 1 s, TPOT under 100 ms) | 3.62 req/s, at 75% load |
| Goodput at 100% and 120% load | 0.90 and 0.00 req/s |
| Prefix caching off / on (2,048 shared tokens) | TTFT p50 3.71 s / 735 ms (5.0 times); hit rate 93.5% |
| Small KV cache vs normal | 6,144 tokens: 70 preemptions, TTFT p50 35.82 s vs 1.27 s |
| Bad requests | too long: 400; wrong key: 401; max_tokens too big: 400 |
| Server killed | 0 completed, 200 failed; /health then gave ConnectionError |
| FP8 weights | skipped: needs compute capability 8.0 or newer, the T4 is 7.5 |

## What Colab cannot teach

Everything above ran on one GPU, in one notebook, for free. That covered the engine: its memory budget, batching, the latency knee, caching, preemption and failure. It did not cover the system around the engine. Those parts need a rented GPU machine or a cluster:

| Topic | Why Colab cannot do it | What to try |
|---|---|---|
| Containers | No Docker on Colab | Run the official `vllm/vllm-openai` image on a GPU machine |
| Dashboards and alerts | No long-running services | Scrape `/metrics` with Prometheus, chart it in Grafana, alert on TPOT p99 and TTFT p99 |
| Several GPUs | One GPU per session | `--tensor-parallel-size 2` with a model that does not fit on one GPU ([Part 7](llm-inference-7-beyond-one-gpu.md)) |
| Several replicas | One server at a time | Two servers behind a router: the vLLM production stack or llm-d on Kubernetes |
| Autoscaling and cold starts | No orchestration | Scale replicas on load, and watch the cold start from Section 2 decide how fast it reacts |
| Real traffic | Random tokens with fixed lengths | Replay real prompts with their real lengths and arrival times |

The workflow from this part carries over unchanged:

1. Start the server and read the startup log.
2. Check the KV cache size against your traffic.
3. Measure capacity with a closed loop.
4. Find the knee with an open loop, and measure goodput against an SLO.
5. Watch the batch, the queue, the KV cache and the preemption counter.
6. Break it before your users do.

<details>
<summary>Research notes: how far to trust these numbers</summary>

- **One run, one machine.** Each number is one benchmark run on one Colab session. Colab machines are shared, and GPU clocks, CPU speed and disk speed differ between sessions. Expect a few percent of difference between runs, more on TTFT p99, which depends on a handful of requests.
- **The client shares the machine.** `vllm bench serve` runs on the same 2-CPU machine as the server and takes some CPU time. At high request rates, the client can add a little latency of its own.
- **Random prompts.** Random token IDs have the right lengths but not the right content. Content does not change the cost of a step, but real lengths vary, and varied lengths make batching less even than here.
- **Fixed output lengths.** `--ignore-eos` makes every answer the same length. Real answers stop at different points, which frees KV cache earlier and changes the batch over time.
- **The capacity estimate.** Section 6 takes the best closed-loop request rate from Section 5 (at 128 in flight) as capacity. That row already had a TTFT p99 of 14.40 s, so it is an upper bound, not a safe operating point.
- **One failed request.** The 50% load run finished 155 of 156 requests; one failed. The saved files do not record why.
- **The killed-server drill.** As Section 9 says, the kill landed before the load tool sent its first request. A run that waits until requests are running before the kill would show a server dying mid-answer.
- **Different servers, slightly different caches.** The first server got 8.38 GiB of KV cache; the restarted ones got 8.57 GiB. The prefix caching "off" run and the preemption baseline ran on restarted servers. The difference is about 2% and does not change any comparison.
- **The `--max-num-seqs` cap.** The startup log does not print it. The value 256 is what the `running` gauge never went above.
- **Versions.** vLLM 0.33.0. Defaults change between versions: in this version prefix caching and chunked prefill are on by default, and `--gpu-memory-utilization` defaults to 0.92. Flag names were checked against the source of this version.

</details>

## Exercises

1. Your prompts average 1,500 tokens and answers 500, and you want 100 requests running at once. With this model's 28 KiB per token, how much KV cache memory do you need? Does the T4 have it?
2. Your SLO is TTFT under 2 seconds and 15 tokens per second per user. Using the table in Section 5, which concurrency would you run at, and what throughput would you get?
3. In Section 6, the waiting queue stayed small while TPOT broke the SLO. Which setting would you change to trade slow streams for a longer wait before the first token, and what would you alert on afterwards?
4. A product sends a 3,000-token system prompt, then the date, then the user's question. Why might its prefix cache hit rate be low, and how would you fix it?
5. Design a version of the killed-server drill that is sure to kill the server while requests are running. Which metric would you watch before you send the kill?

## Summary

> [!TAKEAWAYS]
> - One command, `vllm serve`, gives an OpenAI-compatible server. Its startup log tells you the attention kernel, the weight memory and the KV cache size in tokens. Read it every time.
> - A cold start took 295 s, a warm one 79 s. Keep weights on local disk and keep the compile cache.
> - The KV cache size follows from the model's shape: 28,672 bytes per token here, and the arithmetic matched the log to within its rounding.
> - Batching is the free lunch at first: about 5 times the output at 8 requests in flight for slightly slower streams. At 128 in flight, about 11 times the output cost each user about 11 times slower tokens.
> - Plan with goodput against an SLO. Every request met the SLO up to 75% of the closed-loop capacity. Goodput then fell to 0.90 requests per second at 100% and 0.00 at 120%.
> - On vLLM, overload first grows the batch and slows every stream; the queue grows only when the batch reaches `--max-num-seqs`.
> - Prefix caching cut the median TTFT about 5 times for prompts with a shared 2,048-token start, with a 93.5% hit rate.
> - A KV cache that is too small causes preemption. Here it caused 70 preemptions, less than half the throughput, and a median first token after 36 s. TPOT looked fine.
> - Bad requests fail cleanly with a 4xx. A dead server fails every request. Replicas, health checks and retries with backoff are not optional.

## The series so far

| Part | Question | Main idea |
|---|---|---|
| [1](llm-inference-1-prefill-and-decode.md) | Why is generating slow? | Prefill is compute-bound, decode is memory-bound; batching shares each weight read |
| [2](llm-inference-2-kv-cache.md) | What does the model remember? | The KV cache, its size per token, and why it limits the batch |
| [3](llm-inference-3-vllm.md) | How does an engine fit many users? | Paged KV memory, continuous batching, chunked prefill, preemption |
| [4](llm-inference-4-speculative-decoding.md) | Can we write several tokens per step? | A cheap draft, checked in one pass of the big model |
| [5](llm-inference-5-sglang-vs-vllm.md) | Can requests share work? | Prefix caching with a radix tree or hashed blocks |
| [6](llm-inference-6-serving-in-production.md) | How do you run it for real? | Latency percentiles, goodput, queueing, capacity, cost, failures |
| [7](llm-inference-7-beyond-one-gpu.md) | What if one GPU is not enough? | Tensor, pipeline and expert parallelism, split prefill and decode |
| 8 (this part) | What does it look like on a real GPU? | A real vLLM server, measured, loaded and broken, in one free notebook |

**Next.** Run the notebook yourself, then read [Part 6](llm-inference-6-serving-in-production.md) again. Its queueing maths, capacity planning and failure stories will now have numbers from your own server behind them.

## References

**Papers**

1. W. Kwon et al. [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180). SOSP 2023.
2. Y. Zhong et al. [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*](https://arxiv.org/abs/2401.09670). OSDI 2024.
3. A. Agrawal et al. [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310). OSDI 2024.
4. J. Ainslie et al. [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). EMNLP 2023.

**Documentation and code**

5. vLLM. [Source code at v0.33.0](https://github.com/vllm-project/vllm/tree/v0.33.0), [`vllm serve`](https://docs.vllm.ai/en/latest/cli/serve/), [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/), [Metrics](https://docs.vllm.ai/en/latest/usage/metrics/), [Automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).
6. vLLM issue [#60821](https://github.com/vllm-project/vllm/issues/60821): Triton attention on Turing GPUs (T4) and its prefill speed.
7. Qwen. [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
8. NVIDIA. [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [L4](https://www.nvidia.com/en-us/data-center/l4/) and [A100](https://www.nvidia.com/en-us/data-center/a100/) specifications; [CUDA GPU compute capability](https://developer.nvidia.com/cuda-gpus).
9. Google. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

**Companion files**

10. [The notebook](https://github.com/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb), its [generator](https://github.com/ishwar6/ishwar-books/blob/main/code/vllm-colab/make_notebook.py), the [saved results of the run on this page](https://github.com/ishwar6/ishwar-books/blob/main/code/vllm-colab/results/colab_run.json) and its [server logs and raw benchmark files](https://github.com/ishwar6/ishwar-books/tree/main/code/vllm-colab/results/run).
