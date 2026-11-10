---
title: "vLLM on a Free GPU: Run It, Load It, Break It"
description: "A real vLLM server on a free Colab T4, run like a production service: cold starts, the KV cache budget, batching, the latency knee, prefix caching, preemption and failures, all measured."
date: 2026-11-10
tags: [inference, vllm, serving, colab, llm]
series: "LLM Inference from the Ground Up"
series_part: 8
motif: code
accent: "#8bd17c"
---

Seven parts of this series took an inference engine apart. Most of the numbers came from small models on an Apple GPU and from simulators, because the subject was the ideas inside the engine. This part does the opposite. It runs the real thing: vLLM {{R:R['vllm_version']}}, serving a real model on a real NVIDIA GPU, the way a production team runs it.

The GPU is a free one. Google Colab gives anyone with a Google account an NVIDIA **T4** for a few hours at a time. It is an old card, from 2018, and that turns out to be useful: its limits make the effects in this part easy to see.

Here is one result to start with. With one user, the server wrote **{{R:n0(B('conc_1','output_throughput'))}} tokens per second**. With 64 users at the same time, it wrote **{{R:n0(B('conc_64','output_throughput'))}} tokens per second**, {{R:x(B('conc_64','output_throughput')/B('conc_1','output_throughput'))}} as much, on the same GPU. Each user's stream became only {{R:x(B('conc_64','median_tpot_ms')/B('conc_1','median_tpot_ms'))}} slower. Why that happens, and where it stops, is most of what you need to know to run an LLM server.

> [!TIP] The whole idea in one minute
> You start one vLLM server on a free GPU, then treat it like a production service. You read its startup log to learn how much memory went to the KV cache. You raise the number of users until throughput stops growing. You raise the arrival rate until latency breaks. You switch prefix caching off and on, shrink the KV cache until requests get preempted, and kill the server under load. Every result comes from one notebook that you can run yourself, for free, in about an hour.

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb)
&nbsp; or [download the notebook](/notebooks/vllm-on-colab.ipynb).

Every number on this page comes from one run of that notebook. The run served {{R:R['model']}} in float16 on a Colab **{{R:R['gpu']['name']}}** ({{R:R['gpu']['memory_gib']}} GiB), with vLLM {{R:R['vllm_version']}} and driver {{R:R['gpu']['driver']}}. Your numbers will differ by a few percent. The shapes will not.

## 1. What Colab gives you, and what it does not

Colab is a notebook service. You get a Linux machine with a GPU for a few hours, and you run code in cells. It is meant for experiments, not for serving users. That is fine here: we want to **learn** how a serving engine behaves, and for that one GPU and one model are enough.

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

The T4 has no hardware for **bfloat16**, the 16-bit number format most models are trained in. So we run the model in **float16**, the older 16-bit format. For a model this size the difference in output quality is negligible, but the server must be told: `--dtype half`.

> [!DEFINITION] float16 and bfloat16
> Two ways to store a number in 16 bits. The float16 format keeps more digits of precision but has a smaller range. The bfloat16 format keeps the range of a 32-bit number, with fewer digits. Most new models are trained in bfloat16.

What Colab cannot give you matters as much. There is no Docker, so you cannot run vLLM's official container image. There is one GPU, so there is no tensor parallelism. The server is reachable only from inside the notebook machine. And the session ends after some hours or when you stop using it. Section 10 lists what that leaves out, and where to learn it next.

### The model

The notebook serves **{{R:R['model']}}**: {{R:R['kv_math']['layers']}} layers, 1.5 billion weights, about 3 GB in float16. It is small enough to leave most of the T4's memory for the KV cache, and good enough to give sensible answers. It needs no Hugging Face login.

A larger model would make each step slower and leave a smaller KV cache, but the effects in this part would look the same. On an L4 or an A100, set `MODEL = "Qwen/Qwen2.5-7B-Instruct"` in the notebook and compare.

## 2. Start the server, and read what it tells you

One command starts a production-grade server:

```bash
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --dtype half --max-model-len 4096 --gpu-memory-utilization 0.85 \
  --api-key colab-demo-key --port 8000
```

`--max-model-len` caps one request at 4,096 tokens, prompt and answer together. `--gpu-memory-utilization 0.85` lets vLLM take 85% of the GPU's memory. The notebook runs this command in the background, writes its log to a file and waits until `GET /health` answers.

> [!DEFINITION] Health check
> A URL that answers "200 OK" when the server is ready to serve. Load balancers call it every few seconds and stop sending traffic to a server that does not answer.

### Where the startup time goes

The first start took **{{R:s(S('default','ready_s'))}}** from the command to the first healthy answer. That is a **cold start**: the machine had never seen the model. The log shows where the time went:

| Phase | What happens | Time |
|---|---|---|
| Download and load weights | 3 GB from Hugging Face to disk, then into GPU memory | {{R:s(S('default','weights_gib_and_load_s')[1])}} for the load into the GPU, plus the download |
| `torch.compile` | Fuse the model's operations into faster kernels; the result is cached on disk | {{R:s(S('default','torch_compile_s'))}} |
| Memory profile and KV cache | Run one large dummy batch, measure the peak, give the rest to the KV cache | part of the engine start |
| CUDA graph capture | Record a decode step for many batch sizes, so each later step starts with one call | {{R:s(S('default','cuda_graphs_s_and_gib')[0])}} |
| Whole engine start | Everything after the weights are loaded | {{R:s(S('default','init_engine_s'))}} |

> [!DEFINITION] CUDA graph
> A recording of a fixed sequence of GPU work. Replaying it costs one launch from the CPU instead of hundreds. For small decode steps, where each kernel runs for microseconds, the launch cost would otherwise be a large share of the step.

Later in the notebook, the server restarts several times. The second start took **{{R:s(S('no_prefix_cache','ready_s'))}}**. The weights were on disk now, and the compiled kernels came from the cache.

> [!TIP] In production
> Cold start time decides how fast you can add capacity when traffic jumps. The cheapest cuts are the ones you just saw. Keep the weights on a fast local disk, not in a remote bucket. Keep vLLM's compile cache (`~/.cache/vllm`) between restarts. [Part 6](llm-inference-6-serving-in-production.md) measured each phase of a cold start and showed what it does to autoscaling.

### The lines that matter

Most of the startup log is noise. These lines are not:

```text
{{R:loglines()}}
```

Read them from the top.

**The attention backend** is `{{R:S('default','attention_backend')}}`. vLLM picks the fastest attention kernel your GPU supports. FlashAttention and FlashInfer need compute capability 8.0. So on a T4, vLLM falls back to a kernel written in **Triton**, a language for GPU kernels in Python. The Triton kernel works. But on Turing GPUs it does not use the tensor cores for attention, so prefill is slower than with a native kernel. That is a property of this GPU, and the reason later TTFT numbers are higher than on an L4.

> [!DEFINITION] Tensor cores
> Special units in NVIDIA GPUs that multiply small matrices very fast. Most of a GPU's peak arithmetic speed comes from them, but only when the kernel is written to use them.

**The weights** took **{{R:R['starts'][0]['weights_gib_and_load_s'][0]}} GiB**.

**The KV cache** got **{{R:R['starts'][0]['kv_cache_memory_gib']}} GiB**, which holds **{{R:n0(S('default','kv_cache_tokens'))}} tokens**. The last line turns that into a capacity. {{R:n2(S('default','max_concurrency')[1])}} requests of the full 4,096 tokens could keep their caches in memory at once.

That second number is the one to remember. It is the server's memory budget for conversations. Everything in Sections 5 to 8 is about how it is spent.

{{FIG:memory|Where the GPU's memory went. vLLM takes the share set by --gpu-memory-utilization, loads the weights, reserves working memory for the largest batch, and gives everything left to the KV cache.}}

## 3. The first requests

The server speaks the **OpenAI API**. Any program written for OpenAI works with it if you change two settings, the base URL and the key:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="colab-demo-key")
r = client.chat.completions.create(
    model="Qwen/Qwen2.5-1.5B-Instruct", temperature=0, max_tokens=120,
    messages=[{"role": "user", "content": "Explain in three sentences why an LLM server batches requests."}])
print(r.choices[0].message.content)
```

The model answered:

> {{R:R.get('first_answer','(see the notebook output)')}}

That compatibility is a big part of why vLLM is easy to put into production. Your application, your SDKs and your tools stay the same; only the address changes.

### What one user feels

With `stream=True`, the server sends each token as soon as it exists. The notebook times two things on the client side.

> [!DEFINITION] Time to first token (TTFT)
> From sending the request to receiving the first token. It covers the time in the queue and the **prefill**, the step that reads the whole prompt and writes the first token.

> [!DEFINITION] Time per output token (TPOT)
> The average gap between later tokens. Each one is a **decode** step, which writes one token per request.

For a short prompt and a 256-token answer, three runs gave:

| Run | TTFT | TPOT | Tokens per second |
|---|---|---|---|
{{R:single_rows()}}

So one conversation streams at about **{{R:n0(sum(x['tokens_per_s'] for x in R['single_user_stream'])/3)}} tokens per second**. That is faster than anyone reads. It is also a waste of the GPU, as Section 5 shows.

[Part 1](llm-inference-1-prefill-and-decode.md) explained why the two numbers are so different. Prefill handles all the prompt tokens in one pass, so the GPU's arithmetic units are busy. Decode handles one token per request per step, and every step must still read all {{R:R['starts'][0]['weights_gib_and_load_s'][0]}} GiB of weights from memory. The read, not the arithmetic, sets the step time.

### A quick check of that claim

The T4 reads memory at most at 320 GB/s. Reading {{R:R['starts'][0]['weights_gib_and_load_s'][0]}} GiB at that speed takes:

$$t_{\text{read}} = \frac{\text{weight bytes}}{\text{bandwidth}} = \frac{{{R:n2(W_GB())}} \text{ GB}}{320 \text{ GB/s}} \approx {{R:n1(W_GB()/320*1000)}} \text{ ms}$$

Read it left to right: the time for one decode step can never be shorter than the time to read every weight once. The measured TPOT was **{{R:n1(sum(x['tpot_ms'] for x in R['single_user_stream'])/3)}} ms**. The gap comes from the KV cache reads, the kernel launches and the work on the CPU side. Also, no kernel reaches the full bandwidth. The floor explains most of the step.

## 4. The KV cache budget, checked by hand

The log said the KV cache holds {{R:n0(S('default','kv_cache_tokens'))}} tokens. We can predict that from the model's shape, and doing so is the best way to make sure you understand what the number means.

> [!DEFINITION] KV cache
> For every token it has seen, each layer of the model keeps a **key** vector and a **value** vector. Later tokens look back at them without computing them again. That store is the KV cache. [Part 2](llm-inference-2-kv-cache.md) builds it from zero.

The intuition first. Think of the cache as a notebook that each layer keeps, with one line per token. The size of a line depends on how many key and value heads the layer has and how long each one is.

$$\text{bytes per token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times b$$

Read it left to right. The 2 counts one key and one value. $$L$$ is the number of layers. $$H_{kv}$$ is the number of key/value heads per layer. $$d_{\text{head}}$$ is the length of one head's vector. $$b$$ is the bytes per number, 2 for float16.

For {{R:R['model']}} the model's `config.json` gives $$L = {{R:R['kv_math']['layers']}}$$, $$H_{kv} = {{R:R['kv_math']['kv_heads']}}$$ and $$d_{\text{head}} = {{R:R['kv_math']['head_dim']}}$$:

$$2 \times {{R:R['kv_math']['layers']}} \times {{R:R['kv_math']['kv_heads']}} \times {{R:R['kv_math']['head_dim']}} \times 2 = {{R:n0(R['kv_math']['bytes_per_token'])}} \text{ bytes} = {{R:n0(R['kv_math']['bytes_per_token']/1024)}} \text{ KiB per token}$$

Divide the cache memory by that:

$$\frac{{{R:R['kv_math']['kv_cache_memory_gib']}} \text{ GiB} \times 2^{30}}{{{R:n0(R['kv_math']['bytes_per_token'])}} \text{ bytes}} \approx {{R:n0(R['kv_math']['predicted_tokens'])}} \text{ tokens}$$

The log said {{R:n0(R['kv_math']['log_tokens'])}}. The two agree to {{R:pct(abs(R['kv_math']['predicted_tokens']-R['kv_math']['log_tokens'])/R['kv_math']['log_tokens'])}}. The small gap comes from rounding. The engine stores the cache in fixed-size **blocks** of tokens (the pages of [Part 3](llm-inference-3-vllm.md)), and it hands out memory in whole blocks only.

> [!DEFINITION] Grouped-query attention (GQA)
> A model design where several query heads share one key/value head. This model has {{R:R['kv_math']['heads']}} query heads and only {{R:R['kv_math']['kv_heads']}} KV heads. So its cache is {{R:R['kv_math']['heads']//R['kv_math']['kv_heads']}} times smaller than with one KV head per query head.

What does {{R:n0(R['kv_math']['log_tokens'])}} tokens buy? A request in Section 5 uses 512 prompt tokens and 128 answer tokens, 640 in all. The cache holds about **{{R:n0(R['kv_math']['log_tokens']/640)}}** of them at once, far more than the server's limit on running requests. On this GPU, with this model, memory is not the first limit. Section 8 makes it one on purpose.

## 5. Batching: the free lunch, and its bill

This is the most important experiment in the notebook.

We send identical requests: 512 random prompt tokens and exactly 128 output tokens. We keep a fixed number of them **in flight**: 1, then 2, 4, 8, and so on up to 128. As soon as one finishes, the next one starts. The tool is `vllm bench serve`, the load generator that ships with vLLM:

```bash
vllm bench serve --backend openai --base-url http://localhost:8000 --model Qwen/Qwen2.5-1.5B-Instruct \
  --dataset-name random --random-input-len 512 --random-output-len 128 --ignore-eos \
  --max-concurrency 64 --num-prompts 256 \
  --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 --save-result
```

> [!DEFINITION] Closed loop
> A load test where a fixed number of simulated users each wait for their answer before sending the next request. The load adapts to the server's speed: a slow server gets fewer requests.

`--ignore-eos` makes every answer exactly 128 tokens long, so every run does the same work. The random prompts are random token IDs. Their text is noise, but their cost to the GPU is real.

{{FIG:batching|Left: output tokens per second for the whole server. Right: milliseconds per token for each user. Both against the number of requests in flight, on a log scale.}}

| In flight | Server output, tok/s | Speed-up vs 1 | Per-user TPOT, p50 | Per-user tok/s | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|---|
{{R:conc_rows()}}

Read the second and fourth columns together. From 1 to 64 requests in flight, the server's output grew **{{R:x(B('conc_64','output_throughput')/B('conc_1','output_throughput'))}}**. The time per token for each user grew only **{{R:x(B('conc_64','median_tpot_ms')/B('conc_1','median_tpot_ms'))}}**.

That is the free lunch [Part 1](llm-inference-1-prefill-and-decode.md) promised. A decode step reads all the weights once, whether it writes a token for one request or for 64. The arithmetic for 64 tokens is small next to that read. So 64 requests per step cost little more than one, and each gets its token.

> [!DEFINITION] Continuous batching
> The engine decides which requests are in the batch at every step. A finished request leaves at once, and a waiting one joins at the next step. [Part 3](llm-inference-3-vllm.md) explains how vLLM does it.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §2.3 · page 3
> [![The paragraph on fine-grained batching, with the iteration level, completed requests are removed from the batch and new ones are added, waiting for a single iteration and eliminate the need to pad highlighted](/img/papers/vllm/p2-fine-batching.png)](/img/papers/vllm/p2-fine-batching.png)
>
> **What it says:** batching "at the iteration level" removes finished requests and adds new ones after every step, so a new request waits "a single iteration".
>
> **What it means:**
>
> This is the mechanism behind the table. With 64 requests in flight and answers of different lengths, the batch would empty out unevenly if it were fixed at the start. Because the batch is rebuilt at every step, the next request fills a freed slot at once. So the GPU stays close to the batch size the table shows.

### Where the lunch ends

The curve flattens at the top. From 64 to 128 in flight, throughput grew only **{{R:x(B('conc_128','output_throughput')/B('conc_64','output_throughput'))}}**, while TTFT p50 went from {{R:ms(B('conc_64','median_ttft_ms'))}} to {{R:ms(B('conc_128','median_ttft_ms'))}}.

Two things end the lunch. As the batch grows, the arithmetic and the KV cache reads grow with it, until they, not the weight read, set the step time. And every new request must be prefilled: 512 prompt tokens at once, in the same steps as the decodes of everyone else. With more requests in flight, more prefills crowd into those steps. New requests wait longer for their first token, and every stream stutters while a prefill runs.

> [!DEFINITION] Chunked prefill
> Splitting a long prompt's prefill into pieces, so that each step mixes a bounded amount of prefill work with the decodes of running requests. It keeps streams smooth while new requests start. vLLM does this by default.

> [!TIP] In production
> Pick the operating point from the right half of this table, not the left. A product needs a per-user speed (say 20 tokens per second, faster than reading) and a TTFT limit. The best concurrency is the highest one that still meets both. {{R:per_user_claim()}}

## 6. Past capacity: where latency breaks

Real users do not wait for each other. They arrive at random times, whether the server keeps up or not.

> [!DEFINITION] Open loop
> A load test where requests arrive on a schedule, regardless of how fast the server answers. Here they arrive at random times with a fixed average rate, a **Poisson process**. This is how real traffic behaves. [Part 6](llm-inference-6-serving-in-production.md) shows why a closed loop can hide a slow server.

The notebook takes the best request rate from Section 5, **{{R:n2(R['capacity_req_s'])}} requests per second**, as the server's capacity. Then it sends about a minute of traffic at 25%, 50%, 75%, 90%, 100% and 120% of that rate.

It also measures **goodput** against a latency target. We chose a target that suits a chat product: TTFT under 1 second and TPOT under 100 ms for every request.

> [!DEFINITION] SLO and goodput
> A **service level objective (SLO)** is a latency target, such as "first token within 1 second". **Goodput** is the number of requests per second that meet it. A server can have high throughput and low goodput, if it finishes everything but late.

> [!PAPER] Zhong et al., DistServe · Section 1 · page 2
> [![DistServe: an effective LLM serving system should balance these needs and maximize per-GPU goodput, defined as the maximum request rate that can be served adhering to the SLO attainment goal (say, 90%) for each GPU provisioned; higher per-GPU goodput directly translates into lower cost per query](/img/serving/paper-distserve-goodput.png)](/img/serving/paper-distserve-goodput.png)
>
> **What it says:** a serving system should maximize per-GPU **goodput**, "the maximum request rate that can be served adhering to the SLO attainment goal".
>
> **What it means:**
>
> Throughput counts every finished request. Goodput counts only the ones that arrived in time to be useful. The `--goodput` option of `vllm bench serve` uses this definition, which is why the notebook can report it directly.

{{FIG:knee|TTFT p50 and p99 against the offered load, as a fraction of the capacity measured in Section 5. The vertical axis is logarithmic.}}

| Offered load | Arrival rate, req/s | Served, req/s | Goodput, req/s | TTFT p50 | TTFT p99 | TPOT p50 | TPOT p99 |
|---|---|---|---|---|---|---|---|
{{R:rate_rows()}}

Read the TTFT columns from the top. While the load stays well below capacity, TTFT barely moves: a request arrives, finds room in the next step and starts. Near capacity, the p99 climbs first. Random arrivals come in bursts, and a burst has to wait. Above capacity, more requests arrive each second than leave, so the queue grows for as long as the overload lasts, and TTFT grows with it.

Notice the third column at 120%. The server serves no more than at 100%. Pushing more traffic at a full server does not buy throughput; it only builds a queue.

### Inside the server during the overload

During the 120% run, the notebook read `/metrics` every half second. Three of the numbers a production dashboard shows:

{{FIG:timeline|Requests running and waiting inside the server, and the share of the KV cache in use, during one minute at 120% of capacity.}}

The `running` line stays near the engine's limit. The `waiting` line is the queue. It climbs through the whole run and falls only after the arrivals stop. Every request in it adds its wait to its TTFT.

> [!WARNING] Watch the queue, not the GPU
> Served throughput was about the same at 100% load and at 120%, so a throughput chart cannot tell the two apart. `vllm:num_requests_waiting` can. A queue that stays above zero means the server is past its capacity, and latency will only get worse. Alert on it, and on TTFT p99.

> [!TIP] In production
> Keep each server below the knee: often 60% to 80% of measured capacity, so that bursts have room. Add replicas before the queue forms, not after. [Part 6](llm-inference-6-serving-in-production.md) explains the queueing maths behind the knee.

## 7. Prefix caching: pay for the shared prompt once

Many products send the same long start with every request: a system prompt, tool definitions, a document. If the server keeps the KV cache of that start, later requests can skip its prefill and compute only what is new.

> [!DEFINITION] Prefix caching
> Keeping the KV cache of earlier prompts and reusing it when a new prompt starts with the same tokens. vLLM finds matches in whole blocks. It hashes each block's tokens together with everything before them. [Part 5](llm-inference-5-sglang-vs-vllm.md) covers how vLLM and SGLang do it.

The workload: every request starts with the same 2,048 random tokens and adds 128 tokens of its own. It asks for 64 output tokens, with 8 requests in flight. We run it on a normal server, where prefix caching is on by default, then restart with `--no-enable-prefix-caching` and run it again.

{{FIG:prefix|TTFT p50 and p99 with prefix caching off and on, for prompts with a shared 2,048-token start.}}

| Prefix caching | Cache hit rate | TTFT p50 | TTFT p99 | Requests per second |
|---|---|---|---|---|
| off | {{R:pct(B('prefix_off','prefix_cache_hit_rate') or 0)}} | {{R:ms(B('prefix_off','median_ttft_ms'))}} | {{R:ms(B('prefix_off','p99_ttft_ms'))}} | {{R:n2(B('prefix_off','request_throughput'))}} |
| on | {{R:pct(B('prefix_on','prefix_cache_hit_rate'))}} | {{R:ms(B('prefix_on','median_ttft_ms'))}} | {{R:ms(B('prefix_on','p99_ttft_ms'))}} | {{R:n2(B('prefix_on','request_throughput'))}} |

The hit rate comes from the server's own counters, `vllm:prefix_cache_hits_total` divided by `vllm:prefix_cache_queries_total`, read before and after the run. With the cache on, the server found **{{R:pct(B('prefix_on','prefix_cache_hit_rate'))}}** of the prompt tokens it looked up already in the cache. The median TTFT fell **{{R:x(B('prefix_off','median_ttft_ms')/B('prefix_on','median_ttft_ms'))}}**, and the server finished **{{R:x(B('prefix_on','request_throughput')/B('prefix_off','request_throughput'))}}** as many requests per second.

Why so large? Without the cache, every request prefills about 2,176 tokens. With it, the first request pays for the shared 2,048, and every later one prefills only its own 128 tokens. On a T4, where prefill runs on the slower Triton attention kernel, saving prefill matters even more.

> [!TIP] In production
> Put the parts of a prompt that never change first, and the parts that do (the user's question, the date) last. One changed token early in the prompt makes everything after it a cache miss. With many replicas, the cache is per server, so a router that sends similar prompts to the same server keeps the hit rate high. [Part 6](llm-inference-6-serving-in-production.md) compares routing policies.

## 8. A KV cache too small: preemption

Section 4 showed that this server has room for hundreds of requests' caches. Production servers often do not: bigger models, longer prompts and longer answers fill the cache quickly. What happens then?

vLLM starts a request when there is room for its prompt. As the answer grows, the request needs new blocks. When a running request needs a block and none is free, the scheduler must take memory from someone.

> [!DEFINITION] Preemption
> Stopping a running request to free its memory, and resuming it later. In vLLM V1 the preempted request drops its whole KV cache. Later it rebuilds the cache with one prefill over its prompt and the tokens it has written so far.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §4.5 · page 8
> [![The start of section 4.5, with output lengths are not known a priori and can run out of the GPU's physical blocks highlighted](/img/writings/llm-inference/p3-preempt.png)](/img/writings/llm-inference/p3-preempt.png)
>
> **What it says:** because "output lengths are not known a priori", a server that admits requests without reserving room for their whole answer "can run out of the GPU's physical blocks".
>
> **What it means:**
>
> Admitting requests without a full reservation is what lets paged memory fit so many of them. It is a bet that they will not all grow long at the same time. The experiment below makes the bet lose on purpose, to see what losing costs.

The notebook restarts the server with `--kv-cache-memory-bytes` set to fit only **{{R:n0(S('small_kv','kv_cache_tokens'))}} tokens**. Then it sends 32 requests at a time, each with 512 prompt tokens and 512 output tokens. Together they would need 32,768 tokens of cache. The same job also runs on a normal server, for comparison.

| Server | KV cache, tokens | Preemptions | Output tok/s | TTFT p50 | TTFT p99 | End-to-end p99 |
|---|---|---|---|---|---|---|
| normal | {{R:n0(S('default_2','kv_cache_tokens'))}} | {{R:n0(B('preempt_baseline','preemptions'))}} | {{R:n0(B('preempt_baseline','output_throughput'))}} | {{R:ms(B('preempt_baseline','median_ttft_ms'))}} | {{R:ms(B('preempt_baseline','p99_ttft_ms'))}} | {{R:ms(B('preempt_baseline','p99_e2el_ms'))}} |
| small KV | {{R:n0(S('small_kv','kv_cache_tokens'))}} | {{R:n0(B('preempt_small_kv','preemptions'))}} | {{R:n0(B('preempt_small_kv','output_throughput'))}} | {{R:ms(B('preempt_small_kv','median_ttft_ms'))}} | {{R:ms(B('preempt_small_kv','p99_ttft_ms'))}} | {{R:ms(B('preempt_small_kv','p99_e2el_ms'))}} |

The small server ran out of room {{R:n0(B('preempt_small_kv','preemptions'))}} times. Each time, a request lost its cache and had to rebuild it, so part of the GPU's work was done twice. Throughput fell to **{{R:pct(B('preempt_small_kv','output_throughput')/B('preempt_baseline','output_throughput'))}}** of the normal server's.

The bigger effect is on waiting. With room for only about {{R:n0(S('small_kv','kv_cache_tokens')/1024)}} requests of 1,024 tokens at once, most of the 32 requests in flight sit in the queue. TTFT p99 went from {{R:ms(B('preempt_baseline','p99_ttft_ms'))}} to {{R:ms(B('preempt_small_kv','p99_ttft_ms'))}}.

> [!WARNING] A rising preemption counter is an alarm
> `vllm:num_preemptions_total` should stay flat. If it rises, the KV cache is too small for the traffic. The fixes, from the cheapest:
>
> 1. Raise `--gpu-memory-utilization`.
> 2. Lower `--max-model-len` if no request needs it.
> 3. Lower `--max-num-seqs`, so fewer requests compete.
> 4. Use quantized weights or an FP8 KV cache, so more fits.
> 5. Add replicas.

## 9. Break it on purpose

Production skill is knowing what a failure looks like before it happens at 3 a.m. The notebook tries four.

### Clean failures: bad requests

| What we sent | Status | What the server said |
|---|---|---|
{{R:fail_rows()}}

These are good failures. The server rejects the bad request at once, says why, and keeps serving everyone else. A client should treat any 4xx status as "my request is wrong": fix it, do not retry it.

### The ugly failure: the server dies under load

The notebook starts a benchmark with 32 requests in flight. After 15 seconds it kills the server with `SIGKILL`, the way an out-of-memory killer or a failed machine would.

| | Result |
|---|---|
| Requests completed before the kill | {{R:n0(R['failures']['killed_mid_run'].get('completed', 0))}} |
| Requests failed | {{R:n0(R['failures']['killed_mid_run'].get('failed', 0))}} |
| `/health` after the kill | {{R:R['failures']['killed_mid_run']['health_after']}} |

Every request in flight was lost, and so was every request sent after the kill. Nothing inside vLLM can save them: their KV caches lived in the GPU memory of a process that no longer exists.

That is why production setups never rely on one server. They run several **replicas** behind a load balancer that calls `/health` and stops routing to a dead one. Clients retry failed requests with **exponential backoff** and **jitter**. They wait a little, then longer, with a random part added. Then a thousand clients do not retry at the same moment.

> [!DEFINITION] Exponential backoff with jitter
> A retry rule: wait about 1 second before the first retry, 2 before the second, 4 before the third, and so on. Add a random amount each time. It gives a failing service room to recover instead of hitting it with every retry at once.

> [!WARNING] Retries can keep a server down
> A server that comes back after a crash meets every client's retry at once, plus the new traffic. Without backoff, that burst can push it straight past capacity again. [Part 6](llm-inference-6-serving-in-production.md) simulates this retry storm.

{{R:fp8_section()}}

## 10. What Colab cannot teach, and where to go next

Everything above ran on one GPU, in one notebook, for free. That covered the engine: its memory budget, batching, the latency knee, caching, preemption and failure. It did not cover the system around the engine. Those parts need a rented GPU machine or a cluster, for a few dollars an hour:

| Topic | Why Colab cannot do it | What to try |
|---|---|---|
| Containers | No Docker on Colab | Run the official `vllm/vllm-openai` image on a GPU machine |
| Dashboards and alerts | No long-running services | Scrape `/metrics` with Prometheus, chart it in Grafana, alert on queue and TTFT p99 |
| Several GPUs | One GPU per session | `--tensor-parallel-size 2` with a model that does not fit on one GPU ([Part 7](llm-inference-7-beyond-one-gpu.md)) |
| Several replicas | One server at a time | Two servers behind a router: the vLLM production stack or llm-d on Kubernetes |
| Autoscaling and cold starts | No orchestration | Scale replicas on queue length and watch the cold start from Section 2 decide how fast it reacts |
| Real traffic | Random tokens with fixed lengths | Replay real prompts with their real lengths and arrival times |

The workflow from this part carries over unchanged:

1. Start the server and read the startup log.
2. Measure capacity with a closed loop.
3. Find the knee with an open loop.
4. Set an SLO and measure goodput.
5. Watch the queue, the KV cache use and the preemption counter.
6. Break it before your users do.

<details>
<summary>Research notes: how far to trust these numbers</summary>

- **One run, one machine.** Each number is one benchmark run on one Colab session. Colab machines are shared, and GPU clocks, CPU speed and disk speed differ between sessions. Expect a few percent of difference between runs, more on TTFT p99, which depends on a handful of requests.
- **The client shares the machine.** `vllm bench serve` runs on the same machine as the server and takes some CPU time. At high request rates, the client can add a little latency of its own.
- **Random prompts.** Random token IDs have the right lengths but not the right content. Real text does not change the cost of a step, but real lengths vary, and varied lengths make batching less even than here.
- **Fixed output lengths.** `--ignore-eos` makes every answer the same length. Real answers stop at different points, which frees KV cache earlier and changes the batch over time.
- **The capacity estimate.** Section 6 takes the best closed-loop request rate from Section 5 as capacity. That is a good estimate, not an exact one; the true knee depends on the SLO.
- **Versions.** vLLM {{R:R['vllm_version']}}. Defaults change between versions: in this version prefix caching and chunked prefill are on by default, and `--gpu-memory-utilization` defaults to 0.92. Flag names were checked against the source of this version.

</details>

## Summary

> [!TAKEAWAYS]
> - One command, `vllm serve`, gives an OpenAI-compatible server. The startup log tells you the attention kernel, the weight memory, and the KV cache size in tokens. Read it every time.
> - The KV cache size follows from the model's shape: {{R:n0(R['kv_math']['bytes_per_token']/1024)}} KiB per token here, and the arithmetic matched the log to {{R:pct(abs(R['kv_math']['predicted_tokens']-R['kv_math']['log_tokens'])/R['kv_math']['log_tokens'])}}.
> - Batching is the free lunch: {{R:x(B('conc_64','output_throughput')/B('conc_1','output_throughput'))}} the output at 64 requests in flight, for {{R:x(B('conc_64','median_tpot_ms')/B('conc_1','median_tpot_ms'))}} slower streams.
> - Past capacity, throughput stops rising and only the queue grows. Plan with goodput against an SLO, and alert on the waiting queue.
> - Prefix caching cut the median TTFT {{R:x(B('prefix_off','median_ttft_ms')/B('prefix_on','median_ttft_ms'))}} for prompts with a shared 2,048-token start.
> - A KV cache that is too small causes preemption: {{R:n0(B('preempt_small_kv','preemptions'))}} preemptions and {{R:pct(B('preempt_small_kv','output_throughput')/B('preempt_baseline','output_throughput'))}} of the throughput in our run.
> - A crashed server loses every request in flight. Replicas, health checks and retries with backoff are not optional.

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

## Try it yourself

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb)

1. Open the notebook in Colab and choose **Runtime > Change runtime type > T4 GPU**.
2. Run every cell from the top. The whole run takes about an hour on a T4; `QUICK = True` cuts it to about 25 minutes.
3. The last cell downloads `vllm_colab_results.json` and a zip with every log and raw benchmark file.

Then change one thing and run it again. Try `Qwen/Qwen2.5-0.5B-Instruct` and `Qwen/Qwen2.5-3B-Instruct` and see how the KV cache budget and the knee move. Try an L4: the attention backend changes, and step 10 runs FP8.

## References

**Papers**

1. W. Kwon et al. [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180). SOSP 2023.
2. Y. Zhong et al. [*DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving*](https://arxiv.org/abs/2401.09670). OSDI 2024.
3. A. Agrawal et al. [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310). OSDI 2024.
4. J. Ainslie et al. [*GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints*](https://arxiv.org/abs/2305.13245). EMNLP 2023.

**Documentation and code**

5. vLLM. [Source code at v{{R:R['vllm_version']}}](https://github.com/vllm-project/vllm/tree/v{{R:R['vllm_version']}}), [`vllm serve`](https://docs.vllm.ai/en/latest/cli/serve/), [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/), [Metrics](https://docs.vllm.ai/en/latest/usage/metrics/), [Automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).
6. vLLM issue [#60821](https://github.com/vllm-project/vllm/issues/60821): Triton attention on Turing GPUs (T4) and its prefill speed.
7. Qwen. [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
8. NVIDIA. [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [L4](https://www.nvidia.com/en-us/data-center/l4/) and [A100](https://www.nvidia.com/en-us/data-center/a100/) specifications; [CUDA GPU compute capability](https://developer.nvidia.com/cuda-gpus).
9. Google. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

**Companion files**

10. [The notebook](https://github.com/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb), its [generator](https://github.com/ishwar6/ishwar-books/blob/main/code/vllm-colab/make_notebook.py) and the [saved results of the run on this page](https://github.com/ishwar6/ishwar-books/blob/main/code/vllm-colab/results/colab_run.json).
