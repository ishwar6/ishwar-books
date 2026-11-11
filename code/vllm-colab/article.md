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

Seven parts of this series took an inference engine apart. Most of their numbers came from small models on an Apple GPU and from simulators, because the subject was the ideas inside the engine. This part does the opposite. It runs the real thing: vLLM {{R:R['vllm_version']}}, serving a real model on a real NVIDIA GPU, the way a production team would run it.

The GPU is a free one. Google Colab gives anyone with a Google account an NVIDIA **T4** for a few hours at a time. It is an old card, from 2018. That turns out to be useful: its limits make every effect in this part easy to see.

Here is the result that matters most. With one user, the server wrote about **{{R:n0(B('conc_1','output_throughput'))}} tokens per second**. With 8 users at the same time, it wrote about **{{R:n0(B('conc_8','output_throughput'))}}**, {{R:about(B('conc_8','output_throughput')/B('conc_1','output_throughput'))}} as much, and each user's stream slowed only a little. With 128 users it wrote about **{{R:n0(B('conc_128','output_throughput'))}}**, {{R:about(B('conc_128','output_throughput')/B('conc_1','output_throughput'))}} as much. But now each user got only about {{R:n0(1000/B('conc_128','median_tpot_ms'))}} tokens per second. Where between those two points you run your server is most of what this part is about.

> [!TIP] The whole idea in one minute
> You start one vLLM server on a free GPU and treat it like a production service. You read its startup log to learn how much memory went to the KV cache, and you check that number by hand. You add users until throughput stops growing. You raise the arrival rate until latency breaks. You switch prefix caching off and on, shrink the KV cache until requests get preempted, and kill the server under load. Every result comes from one notebook that you can run yourself, for free, in about an hour.

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb)
&nbsp; or [download the notebook](/notebooks/vllm-on-colab.ipynb).

Every number on this page comes from one full run of that notebook, on 10 November 2026. The run served {{R:R['model']}} in float16 on a Colab **{{R:R['gpu']['name']}}** ({{R:R['gpu']['memory_gib']}} GiB), with vLLM {{R:R['vllm_version']}}. Your numbers will differ by a few percent. The shapes will not.

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

The notebook serves **{{R:R['model']}}**. It has {{R:R['kv_math']['layers']}} layers and 1.5 billion weights, about 3 GB in float16. It is small enough to leave most of the T4's memory for the KV cache, and good enough to give sensible answers. It needs no Hugging Face login.

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

The first start took **{{R:s(S('default','ready_s'))}}**, about five minutes, from the command to the first healthy answer. That is a **cold start**.

> [!DEFINITION] Cold start and warm start
> A **cold start** begins on a machine that has never seen the model: it must download the weights and build every cache. A **warm start** finds the weights on disk and the compiled kernels in a cache.

Later in the notebook, the server restarts four more times on the same machine. Those warm starts took between {{R:s(min(x_['ready_s'] for x_ in R['starts'][1:]))}} and {{R:s(max(x_['ready_s'] for x_ in R['starts'][1:]))}}. The log shows where the difference comes from:

| Phase | What happens | Cold start | First warm start |
|---|---|---|---|
| Load the weights | Download about 3 GB (cold start only), then copy them into GPU memory | {{R:s(S('default','weights_gib_and_load_s')[1])}} | {{R:s(S('no_prefix_cache','weights_gib_and_load_s')[1])}} |
| `torch.compile` | Fuse the model's operations into faster kernels; the result is cached on disk | {{R:s(S('default','torch_compile_s'))}} | {{R:s(S('no_prefix_cache','torch_compile_s'))}} |
| Engine start | Compile, measure memory, make the KV cache, record CUDA graphs | {{R:s(S('default','init_engine_s'))}} | {{R:s(S('no_prefix_cache','init_engine_s'))}} |
| From command to healthy | All of the above, plus starting Python and the HTTP server | **{{R:s(S('default','ready_s'))}}** | **{{R:s(S('no_prefix_cache','ready_s'))}}** |

> [!DEFINITION] CUDA graph
> A recording of a fixed sequence of GPU work. Replaying it costs one launch from the CPU instead of hundreds. For small decode steps, where each kernel runs for microseconds, the launch cost would otherwise be a large share of the step.

Most of the cold start was the download. The same "Model loading" line said {{R:s(S('default','weights_gib_and_load_s')[1])}} on the cold start and {{R:s(S('no_prefix_cache','weights_gib_and_load_s')[1])}} on the warm one. The compile step fell from {{R:s(S('default','torch_compile_s'))}} to under a second, because its result came from the cache.

> [!TIP] In production
> Cold start time decides how fast you can add capacity when traffic jumps. Here the cold start was {{R:about(S('default','ready_s')/S('no_prefix_cache','ready_s'))}} slower than the warm one. The cheapest cuts are the ones you just saw. Keep the weights on a fast local disk, not in a remote bucket. Keep vLLM's compile cache (`~/.cache/vllm`) between restarts. [Part 6](llm-inference-6-serving-in-production.md) shows what slow starts do to autoscaling.

### The lines that matter

Most of the startup log is noise. These lines are not:

```text
{{R:loglines()}}
```

Read them from the top.

**The attention backend** is `{{R:S('default','attention_backend')}}`. vLLM picks the fastest attention kernel your GPU supports. FlashAttention and FlashInfer need compute capability 8.0. So on a T4, vLLM falls back to a kernel written in **Triton**, a language for writing GPU kernels in Python. The Triton kernel works. But on this older GPU it is slower at prefill than a native kernel. That is one reason the TTFT numbers below are higher than on an L4.

**The weights** took **{{R:R['starts'][0]['weights_gib_and_load_s'][0]}} GiB**.

**The KV cache** got **{{R:R['starts'][0]['kv_cache_memory_gib']}} GiB**, which holds **{{R:n0(S('default','kv_cache_tokens'))}} tokens**. The same line turns that into a capacity: {{R:n2(S('default','max_concurrency')[1])}} requests of the full 4,096 tokens could keep their caches in memory at once.

That last number is the one to remember. It is the server's memory budget for conversations. Sections 4 to 8 are about how it gets spent.

{{FIG:memory|Where the GPU's memory went. vLLM takes the share set by --gpu-memory-utilization, loads the weights, keeps working memory for the largest batch, and gives everything left to the KV cache.}}

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

> {{R:R.get('first_answer','(see the notebook output)')}}

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
{{R:single_rows()}}

So one conversation gets its first token in about {{R:n0(sum(x_['ttft_ms'] for x_ in R['single_user_stream'])/3)}} ms and then streams at about **{{R:n0(sum(x_['tokens_per_s'] for x_ in R['single_user_stream'])/3)}} tokens per second**. That is far faster than anyone reads. It is also a waste of the GPU, as Section 5 shows.

### Why decode is slow: a quick check

[Part 1](llm-inference-1-prefill-and-decode.md) explained why the two numbers differ so much. Prefill handles all the prompt tokens in one pass. Decode writes one token per step, and every step must still read all {{R:R['starts'][0]['weights_gib_and_load_s'][0]}} GiB of weights from memory.

The T4 reads memory at most at 320 GB/s. Reading the weights once at that speed takes:

$$t_{\text{read}} = \frac{\text{weight bytes}}{\text{bandwidth}} = \frac{{{R:n2(W_GB())}} \text{ GB}}{320 \text{ GB/s}} \approx {{R:n1(W_GB()/320*1000)}} \text{ ms}$$

Read it left to right: one decode step can never be shorter than the time to read every weight once. The measured TPOT was **{{R:n1(sum(x_['tpot_ms'] for x_ in R['single_user_stream'])/3)}} ms**. The rest goes to reading the KV cache, launching kernels and work on the CPU, and no kernel reaches the full bandwidth. So the weight read explains most of each step.

## 4. The KV cache budget, checked by hand

**The question:** can we predict the KV cache size from the model's shape?

The log said the KV cache holds {{R:n0(S('default','kv_cache_tokens'))}} tokens. Predicting that number yourself is the best way to make sure you know what it means.

> [!DEFINITION] KV cache
> For every token it has seen, each layer of the model keeps a **key** vector and a **value** vector. Later tokens look back at them without computing them again. That store is the KV cache. [Part 2](llm-inference-2-kv-cache.md) builds it from zero.

Think of the cache as a notebook that each layer keeps, with one line per token. The size of a line depends on how many key and value heads the layer has, and how long each head's vector is.

$$\text{bytes per token} = 2 \times L \times H_{kv} \times d_{\text{head}} \times b$$

The 2 counts one key and one value. $$L$$ is the number of layers. $$H_{kv}$$ is the number of key/value heads per layer. $$d_{\text{head}}$$ is the length of one head's vector. $$b$$ is the bytes per number, 2 for float16.

For {{R:R['model']}}, the model's `config.json` gives $$L = {{R:R['kv_math']['layers']}}$$, $$H_{kv} = {{R:R['kv_math']['kv_heads']}}$$ and $$d_{\text{head}} = {{R:R['kv_math']['head_dim']}}$$:

$$2 \times {{R:R['kv_math']['layers']}} \times {{R:R['kv_math']['kv_heads']}} \times {{R:R['kv_math']['head_dim']}} \times 2 = {{R:n0(R['kv_math']['bytes_per_token'])}} \text{ bytes} = {{R:n0(R['kv_math']['bytes_per_token']/1024)}} \text{ KiB per token}$$

Divide the cache memory by that:

$$\frac{{{R:R['kv_math']['kv_cache_memory_gib']}} \text{ GiB} \times 2^{30}}{{{R:n0(R['kv_math']['bytes_per_token'])}} \text{ bytes}} \approx {{R:n0(kv_pred())}} \text{ tokens}$$

The log said {{R:n0(R['kv_math']['log_tokens'])}}. The two differ by only {{R:n0(abs(kv_pred()-R['kv_math']['log_tokens']))}} tokens. The small gap is rounding: the log prints the memory with only two decimals. Turn it around and {{R:n0(R['kv_math']['log_tokens'])}} tokens at {{R:n0(R['kv_math']['bytes_per_token'])}} bytes each is {{R:f"{R['kv_math']['log_tokens']*R['kv_math']['bytes_per_token']/2**30:.4f}"}} GiB, which the log rounds to {{R:R['kv_math']['kv_cache_memory_gib']}}. The engine stores the cache in fixed-size **blocks** of 16 tokens, the pages of [Part 3](llm-inference-3-vllm.md). So the token count is always a whole number of blocks.

> [!DEFINITION] Grouped-query attention (GQA)
> A model design where several query heads share one key/value head. This model has {{R:R['kv_math']['heads']}} query heads and only {{R:R['kv_math']['kv_heads']}} KV heads. So its cache is {{R:R['kv_math']['heads']//R['kv_math']['kv_heads']}} times smaller than with one KV head per query head.

### Sizing the KV cache for your traffic

The same arithmetic tells you whether a server has enough room. Multiply the requests you expect to run at once by the tokens each one holds at its longest, prompt plus answer:

$$\text{tokens needed} = \text{requests at once} \times (\text{prompt tokens} + \text{output tokens})$$

The requests in Sections 5 and 6 have 512 prompt tokens and 128 output tokens, 640 in all. The cache holds about **{{R:n0(R['kv_math']['log_tokens']/640)}}** of them at once. That is more than vLLM will ever run together. The scheduler also caps the number of requests in one batch, and Section 6 shows that cap at work. So on this GPU, with this model, memory is not the first limit. Section 8 makes it the limit on purpose.

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

{{FIG:batching|Top: output tokens per second for the whole server. Bottom: milliseconds per token for each user. Both against the number of requests in flight, on a log scale.}}

| In flight | Server output, tok/s | Speed-up vs 1 | Per-user TPOT, p50 | Per-user tok/s | TTFT p50 | TTFT p99 |
|---|---|---|---|---|---|---|
{{R:conc_rows()}}

Read the table in two halves.

**The top half is the free lunch.** From 1 to 8 requests in flight, the server's output grew {{R:about(B('conc_8','output_throughput')/B('conc_1','output_throughput'))}}. Each user's time per token grew only from {{R:n1(B('conc_1','median_tpot_ms'))}} to {{R:n1(B('conc_8','median_tpot_ms'))}} ms.

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

**The bottom half is the bill.** From 8 to 128 in flight, throughput grew only {{R:about(B('conc_128','output_throughput')/B('conc_8','output_throughput'))}}. Each user's time per token grew from {{R:n1(B('conc_8','median_tpot_ms'))}} to {{R:n1(B('conc_128','median_tpot_ms'))}} ms, so each stream fell to about {{R:n0(1000/B('conc_128','median_tpot_ms'))}} tokens per second. That is slower than many people read. The TTFT p99 grew from {{R:ms(B('conc_8','p99_ttft_ms'))}} to {{R:ms(B('conc_128','p99_ttft_ms'))}}.

Two things end the lunch. As the batch grows, the arithmetic and the KV cache reads grow with it, until they, not the weight read, set the step time. And every new request must be prefilled: 512 prompt tokens, in the same steps as everyone else's decodes. With more requests in flight, more prefills crowd into those steps. New requests wait longer for their first token, and every running stream stutters while a prefill runs.

> [!DEFINITION] Chunked prefill
> Splitting a long prompt's prefill into pieces, so that each step mixes a bounded amount of prefill work with the decodes of running requests. It keeps streams smooth while new requests start. vLLM does this by default.

> [!TIP] In production
> Choose the operating point from your targets, not from the biggest throughput number. A chat product needs a per-user speed (say 20 tokens per second, faster than reading) and a TTFT limit. The right concurrency is the highest one that still meets both. {{R:per_user_claim()}} At 8 in flight, the TTFT p99 was {{R:n0(B('conc_8','p99_ttft_ms'))}} ms, just over the line.

## 6. Past capacity: where latency breaks

**The question:** how much traffic can the server take before users notice?

Real users do not wait for each other. They arrive at random times, whether the server keeps up or not.

> [!DEFINITION] Open loop
> A load test where requests arrive on a schedule, no matter how fast the server answers. Here they arrive at random times with a fixed average rate, a **Poisson process**. This is how real traffic behaves. [Part 6](llm-inference-6-serving-in-production.md) shows why a closed loop can hide a slow server.

The notebook takes the best request rate from Section 5, **{{R:n2(R['capacity_req_s'])}} requests per second**, as the server's capacity. Then it sends about a minute of traffic at 25%, 50%, 75%, 90%, 100% and 120% of that rate.

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
{{R:rate_rows()}}

{{FIG:goodput|For each load: how many requests per second arrived, how many the server finished, and how many met the SLO (TTFT under 1 s and TPOT under 100 ms).}}

Read the goodput column from the top. Up to 75% load, every request met the SLO. At 90%, about {{R:n0(100*(1-R['bench']['rate_0.9']['request_goodput']/R['bench']['rate_0.9']['request_throughput']))}}% of them missed it. At 100%, goodput fell to {{R:n2(R['bench']['rate_1.0']['request_goodput'])}} requests per second. At 120%, it was {{R:n2(R['bench']['rate_1.2']['request_goodput'])}}: the server still finished {{R:n2(R['bench']['rate_1.2']['request_throughput'])}} requests per second, and not one of them was fast enough.

So the useful capacity of this server is not {{R:n2(R['capacity_req_s'])}} requests per second. It is about {{R:n2(R['bench']['rate_0.75']['request_goodput'])}}, the goodput at 75% load.

Now look at which target broke. At 100% load, the TTFT p50 was still {{R:ms(R['bench']['rate_1.0']['median_ttft_ms'])}}, well under 1 second. The TPOT p50 was {{R:n0(R['bench']['rate_1.0']['median_tpot_ms'])}} ms, far over 100. **The streams broke before the first tokens did.**

{{FIG:knee|TTFT p50 and p99 against the offered load, as a share of the capacity measured in Section 5. The vertical axis is logarithmic.}}

TTFT did break too, one step later. At 120%, the TTFT p99 was {{R:ms(R['bench']['rate_1.2']['p99_ttft_ms'])}}. Above capacity, more requests arrive each second than leave, so the backlog grows for as long as the overload lasts.

### Inside the server during the overload

During the 120% run, the notebook read `/metrics` every half second. It recorded three numbers that a production dashboard shows: requests running, requests waiting, and the share of the KV cache in use.

{{FIG:timeline|Requests running and waiting inside the server, and the share of the KV cache in use, during the run at 120% of capacity. The first {{R:n0(next(p['t'] for p in R['overload_timeline'] if p['running']>0))}} seconds are the load tool starting up.}}

The picture is not the one most people expect. The `running` line climbed for about 45 seconds, then flattened at **{{R:n0(max(p['running'] for p in R['overload_timeline']))}}**. Until then, the `waiting` line stayed short: about {{R:n0(__import__('statistics').mean(p['waiting'] for p in R['overload_timeline'] if p['running'] > 0 and p['running'] < max(q['running'] for q in R['overload_timeline']) - 6 and p['t'] < 65))}} on average. Its peak, **{{R:n0(max(p['waiting'] for p in R['overload_timeline']))}}**, came only after the batch was full. The KV cache never passed **{{R:pct(max(p['kv_usage'] for p in R['overload_timeline']))}}** full.

Here is why. vLLM admits a new request whenever there is memory for it. So in an overload, the backlog does not sit in the queue at first. It sits inside the batch.

The batch grows, every step gets longer, and every user's stream slows down. That is the TPOT column above. Only when the batch hits the scheduler's cap does the queue really grow. In this run that cap was {{R:n0(max(p['running'] for p in R['overload_timeline']))}} requests, which matches vLLM's `--max-num-seqs` limit.

That also checks the sizing sum from Section 4. {{R:n0(max(p['running'] for p in R['overload_timeline']))}} requests of at most 640 tokens need at most {{R:n0(max(p['running'] for p in R['overload_timeline'])*640)}} tokens, about half of the {{R:n0(S('default','kv_cache_tokens'))}} in the cache. The gauge showed {{R:pct(max(p['kv_usage'] for p in R['overload_timeline']))}}.

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

{{FIG:prefix|TTFT p50 and p99 with prefix caching off and on, for prompts with a shared 2,048-token start.}}

| Prefix caching | Cache hit rate | TTFT p50 | TTFT p99 | TPOT p50 | Requests per second |
|---|---|---|---|---|---|
| off | (off) | {{R:ms(B('prefix_off','median_ttft_ms'))}} | {{R:ms(B('prefix_off','p99_ttft_ms'))}} | {{R:n1(B('prefix_off','median_tpot_ms'))}} ms | {{R:n2(B('prefix_off','request_throughput'))}} |
| on | {{R:pct(B('prefix_on','prefix_cache_hit_rate'))}} | {{R:ms(B('prefix_on','median_ttft_ms'))}} | {{R:ms(B('prefix_on','p99_ttft_ms'))}} | {{R:n1(B('prefix_on','median_tpot_ms'))}} ms | {{R:n2(B('prefix_on','request_throughput'))}} |

The hit rate comes from the server's own counters: `vllm:prefix_cache_hits_total` divided by `vllm:prefix_cache_queries_total`, read before and after the run. With the cache on, the server found **{{R:pct(B('prefix_on','prefix_cache_hit_rate'))}}** of the prompt tokens it looked up already in the cache.

The median TTFT fell from about {{R:s(B('prefix_off','median_ttft_ms')/1000)}} to {{R:ms(B('prefix_on','median_ttft_ms'))}}, **{{R:about(B('prefix_off','median_ttft_ms')/B('prefix_on','median_ttft_ms'))}}** faster. The server finished {{R:about(B('prefix_on','request_throughput')/B('prefix_off','request_throughput'))}} as many requests per second. Even the streams got faster, from {{R:n0(B('prefix_off','median_tpot_ms'))}} to {{R:n0(B('prefix_on','median_tpot_ms'))}} ms per token, because fewer long prefills crowded into the decode steps.

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

The notebook restarts the server with `--kv-cache-memory-bytes` set to fit only **{{R:n0(S('small_kv','kv_cache_tokens'))}} tokens**. Then it sends 32 requests at a time, each with 512 prompt tokens and 512 output tokens. Together they would need 32,768 tokens of cache, more than five times what there is. The same job also runs on a normal server, for comparison.

| Server | KV cache, tokens | Preemptions | Output tok/s | TTFT p50 | TTFT p99 | TPOT p50 | End-to-end p99 |
|---|---|---|---|---|---|---|---|
| normal | {{R:n0(S('default_2','kv_cache_tokens'))}} | {{R:n0(B('preempt_baseline','preemptions'))}} | {{R:n0(B('preempt_baseline','output_throughput'))}} | {{R:ms(B('preempt_baseline','median_ttft_ms'))}} | {{R:ms(B('preempt_baseline','p99_ttft_ms'))}} | {{R:n1(B('preempt_baseline','median_tpot_ms'))}} ms | {{R:ms(B('preempt_baseline','p99_e2el_ms'))}} |
| small KV | {{R:n0(S('small_kv','kv_cache_tokens'))}} | {{R:n0(B('preempt_small_kv','preemptions'))}} | {{R:n0(B('preempt_small_kv','output_throughput'))}} | {{R:ms(B('preempt_small_kv','median_ttft_ms'))}} | {{R:ms(B('preempt_small_kv','p99_ttft_ms'))}} | {{R:n1(B('preempt_small_kv','median_tpot_ms'))}} ms | {{R:ms(B('preempt_small_kv','p99_e2el_ms'))}} |

The small server ran out of room {{R:n0(B('preempt_small_kv','preemptions'))}} times. Each time, a request lost its cache and had to rebuild it, so part of the GPU's work was done twice. Throughput fell to {{R:pct(B('preempt_small_kv','output_throughput')/B('preempt_baseline','output_throughput'))}} of the normal server's, a bit less than half.

The bigger effect is on waiting. The cache has room for only about {{R:n0(S('small_kv','kv_cache_tokens')/1024)}} requests of 1,024 tokens at once. So most of the 32 requests in flight sit in the queue. The median TTFT went from about {{R:s(B('preempt_baseline','median_ttft_ms')/1000)}} to about {{R:s(B('preempt_small_kv','median_ttft_ms')/1000)}}.

Now look at the TPOT column. The small server's streams were *faster*: {{R:n1(B('preempt_small_kv','median_tpot_ms'))}} ms per token against {{R:n1(B('preempt_baseline','median_tpot_ms'))}}. Only a few requests ran at a time, so each step was short. A dashboard that showed only TPOT would say this server was healthier. Its users waited more than half a minute for a first word.

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
{{R:fail_rows()}}

These are good failures. The server rejects the bad request at once, says why, and keeps serving everyone else. A client should treat any 4xx status as "my request is wrong": fix it, do not retry it.

> [!DEFINITION] Status code
> The three-digit number at the start of every HTTP answer. 200 means success. 4xx means the request was wrong (400 a bad request, 401 a missing or wrong key). 5xx means the server failed.

Two of these protect the server. The 4,096-token limit from `--max-model-len` stops one huge prompt from taking the KV cache from everyone. The API key stops strangers from using your GPU. Note that `/health` and `/metrics` stay open without the key, so keep the port itself private.

### The ugly failure: the server is gone

The notebook started a load test with 32 requests in flight in a background thread. After 15 seconds it killed the server with `SIGKILL`, the way an out-of-memory killer or a failed machine would.

| | Result |
|---|---|
| Requests completed | {{R:n0(R['failures']['killed_mid_run'].get('completed', 0))}} |
| Requests failed | {{R:n0(R['failures']['killed_mid_run'].get('failed', 0))}} |
| Time until all of them had failed | {{R:s(B('killed_mid_run','duration'))}} |
| `/health` after the kill | {{R:R['failures']['killed_mid_run']['health_after']}} (no answer at all) |

One honest note about this run. The load tool spent more than 15 seconds starting up, so the kill came before its first request. The server log confirms that no request from the test ever reached it. So this drill measured a server that was already gone, not one that died halfway through an answer. Every one of the {{R:n0(R['failures']['killed_mid_run'].get('failed', 0))}} requests failed within {{R:s(B('killed_mid_run','duration'))}}, with a connection error.

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

The notebook skips this step on the T4. FP8 needs a GPU with compute capability 8.0 or newer, and the T4 is {{R:R['gpu']['compute_capability']}}. On an L4 the step runs, and on an A100 it runs on a slower path that saves memory but may not save time. We have no measured FP8 numbers for this page, so it shows none. [Part 6](llm-inference-6-serving-in-production.md) measures quantization on a real model.

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
| `--gpu-memory-utilization` | The share of GPU memory vLLM may take; all of it beyond weights and working memory becomes KV cache | 0.85 gave {{R:R['starts'][0]['kv_cache_memory_gib']}} GiB of KV cache |
| `--max-model-len` | The longest prompt plus answer for one request | Longer requests were rejected with a 400 |
| `--max-num-seqs` | The most requests in one batch | The batch stopped at {{R:n0(max(p['running'] for p in R['overload_timeline']))}}, and only then did a queue form |
| `--dtype` | The number format of weights and cache | `half` on the T4, which has no bfloat16 |
| `--api-key` | A key on every `/v1` request | A wrong key got a 401 |
| `--no-enable-prefix-caching` | Turns off prefix caching, which is on by default | TTFT p50 {{R:about(B('prefix_off','median_ttft_ms')/B('prefix_on','median_ttft_ms'))}} slower with a shared prompt |
| `--kv-cache-memory-bytes` | A fixed KV cache size, in place of the share above | A small cache gave {{R:n0(B('preempt_small_kv','preemptions'))}} preemptions |

**What to watch on `/metrics`.**

| Metric | What it tells you | The warning sign in this run |
|---|---|---|
| `vllm:num_requests_running` | Requests in the batch now | Stuck at the `--max-num-seqs` cap ({{R:n0(max(p['running'] for p in R['overload_timeline']))}}) |
| `vllm:num_requests_waiting` | Requests in the queue | Above zero once the batch was full |
| `vllm:kv_cache_usage_perc` | Share of KV blocks in use | Near 100% together with preemptions |
| `vllm:num_preemptions_total` | Requests stopped to free memory | Any steady rise ({{R:n0(B('preempt_small_kv','preemptions'))}} in one small-cache run) |
| `vllm:prefix_cache_hits_total` ÷ `vllm:prefix_cache_queries_total` | Share of prompt tokens reused | Near zero when you expected reuse |
| `vllm:time_to_first_token_seconds` | TTFT, as a histogram | p99 over your SLO |
| `vllm:inter_token_latency_seconds` | The gaps between tokens | p99 over your SLO: on vLLM this breaks first |

**How to size the KV cache.** Multiply the requests you expect at once by their longest prompt plus answer, and compare with the log's "GPU KV cache size" line. Check the result on the running server: the KV gauge should match the sum.

**Where latency breaks.** Measure capacity with a closed loop. Then find the knee with an open loop and an SLO, and plan with goodput, not throughput. Here the closed loop said {{R:n2(R['capacity_req_s'])}} requests per second, and the SLO said about {{R:n2(R['bench']['rate_0.75']['request_goodput'])}}.

**When prefix caching pays.** When many requests share a long start. Measure the hit rate, and put the fixed parts of the prompt first.

**What preemption looks like.** A rising preemption counter, a full KV cache, long waits for the first token, and streams that look fine.

**How it fails.** Bad requests fail cleanly with a 4xx. A dead server fails everything, fast. Run replicas behind health checks, and retry with backoff.

## All the results in one table

| What | Measured value |
|---|---|
{{R:results_rows()}}

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
- **The capacity estimate.** Section 6 takes the best closed-loop request rate from Section 5 (at 128 in flight) as capacity. That row already had a TTFT p99 of {{R:ms(B('conc_128','p99_ttft_ms'))}}, so it is an upper bound, not a safe operating point.
- **One failed request.** The 50% load run finished {{R:R['bench']['rate_0.5']['completed']}} of {{R:R['bench']['rate_0.5']['num_prompts']}} requests; one failed. The saved files do not record why.
- **The killed-server drill.** As Section 9 says, the kill landed before the load tool sent its first request. A run that waits until requests are running before the kill would show a server dying mid-answer.
- **Different servers, slightly different caches.** The first server got {{R:R['starts'][0]['kv_cache_memory_gib']}} GiB of KV cache; the restarted ones got {{R:R['starts'][1]['kv_cache_memory_gib']}} GiB. The prefix caching "off" run and the preemption baseline ran on restarted servers. The difference is about 2% and does not change any comparison.
- **The `--max-num-seqs` cap.** The startup log does not print it. The value {{R:n0(max(p['running'] for p in R['overload_timeline']))}} is what the `running` gauge never went above.
- **Versions.** vLLM {{R:R['vllm_version']}}. Defaults change between versions: in this version prefix caching and chunked prefill are on by default, and `--gpu-memory-utilization` defaults to 0.92. Flag names were checked against the source of this version.

</details>

## Exercises

1. Your prompts average 1,500 tokens and answers 500, and you want 100 requests running at once. With this model's {{R:n0(R['kv_math']['bytes_per_token']/1024)}} KiB per token, how much KV cache memory do you need? Does the T4 have it?
2. Your SLO is TTFT under 2 seconds and 15 tokens per second per user. Using the table in Section 5, which concurrency would you run at, and what throughput would you get?
3. In Section 6, the waiting queue stayed small while TPOT broke the SLO. Which setting would you change to trade slow streams for a longer wait before the first token, and what would you alert on afterwards?
4. A product sends a 3,000-token system prompt, then the date, then the user's question. Why might its prefix cache hit rate be low, and how would you fix it?
5. Design a version of the killed-server drill that is sure to kill the server while requests are running. Which metric would you watch before you send the kill?

## Summary

> [!TAKEAWAYS]
> - One command, `vllm serve`, gives an OpenAI-compatible server. Its startup log tells you the attention kernel, the weight memory and the KV cache size in tokens. Read it every time.
> - A cold start took {{R:s(S('default','ready_s'))}}, a warm one {{R:s(S('no_prefix_cache','ready_s'))}}. Keep weights on local disk and keep the compile cache.
> - The KV cache size follows from the model's shape: {{R:n0(R['kv_math']['bytes_per_token'])}} bytes per token here, and the arithmetic matched the log to within its rounding.
> - Batching is the free lunch at first: {{R:about(B('conc_8','output_throughput')/B('conc_1','output_throughput'))}} the output at 8 requests in flight for slightly slower streams. At 128 in flight, {{R:about(B('conc_128','output_throughput')/B('conc_1','output_throughput'))}} the output cost each user {{R:about(B('conc_128','median_tpot_ms')/B('conc_1','median_tpot_ms'))}} slower tokens.
> - Plan with goodput against an SLO. Every request met the SLO up to 75% of the closed-loop capacity. Goodput then fell to {{R:n2(R['bench']['rate_1.0']['request_goodput'])}} requests per second at 100% and {{R:n2(R['bench']['rate_1.2']['request_goodput'])}} at 120%.
> - On vLLM, overload first grows the batch and slows every stream; the queue grows only when the batch reaches `--max-num-seqs`.
> - Prefix caching cut the median TTFT {{R:about(B('prefix_off','median_ttft_ms')/B('prefix_on','median_ttft_ms'))}} for prompts with a shared 2,048-token start, with a {{R:pct(B('prefix_on','prefix_cache_hit_rate'))}} hit rate.
> - A KV cache that is too small causes preemption. Here it caused {{R:n0(B('preempt_small_kv','preemptions'))}} preemptions, less than half the throughput, and a median first token after {{R:s(B('preempt_small_kv','median_ttft_ms')/1000)}}. TPOT looked fine.
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

5. vLLM. [Source code at v{{R:R['vllm_version']}}](https://github.com/vllm-project/vllm/tree/v{{R:R['vllm_version']}}), [`vllm serve`](https://docs.vllm.ai/en/latest/cli/serve/), [`vllm bench serve`](https://docs.vllm.ai/en/latest/cli/bench/serve/), [Metrics](https://docs.vllm.ai/en/latest/usage/metrics/), [Automatic prefix caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [Optimization and tuning](https://docs.vllm.ai/en/latest/configuration/optimization/).
6. vLLM issue [#60821](https://github.com/vllm-project/vllm/issues/60821): Triton attention on Turing GPUs (T4) and its prefill speed.
7. Qwen. [Qwen2.5-1.5B-Instruct model card](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct).
8. NVIDIA. [T4](https://www.nvidia.com/en-us/data-center/tesla-t4/), [L4](https://www.nvidia.com/en-us/data-center/l4/) and [A100](https://www.nvidia.com/en-us/data-center/a100/) specifications; [CUDA GPU compute capability](https://developer.nvidia.com/cuda-gpus).
9. Google. [Colab FAQ](https://research.google.com/colaboratory/faq.html).

**Companion files**

10. [The notebook](https://github.com/ishwar6/ishwar-books/blob/main/public/notebooks/vllm-on-colab.ipynb), its [generator](https://github.com/ishwar6/ishwar-books/blob/main/code/vllm-colab/make_notebook.py), the [saved results of the run on this page](https://github.com/ishwar6/ishwar-books/blob/main/code/vllm-colab/results/colab_run.json) and its [server logs and raw benchmark files](https://github.com/ishwar6/ishwar-books/tree/main/code/vllm-colab/results/run).
