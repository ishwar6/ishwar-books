---
title: "The Experiments"
description: "How the paper tests vLLM against FasterTransformer and three versions of Orca, how to read its latency plots, and what the ablations show. We rebuild the main experiments as a small serving simulation on real request lengths."
part: 5
covers: "§6 to §10, Figures 11 to 19"
date: 2026-10-24
tags: [llm-serving, inference, kv-cache, benchmarks, simulation]
---

Parts 1 to 4 built the idea. Store each request's KV cache in small blocks, hand them out on demand, and share them when requests can. This part asks the question that decides whether any of it matters. On a real GPU with real requests, does vLLM serve more people at the same speed?

Each piece below shows the exact lines of the paper, then explains them. Next to the paper's numbers we put our own: a small serving simulator fed with real request lengths. Those numbers come from a laptop, not from the paper's A100 GPUs, and the page says so each time.

> [!DEFINITION] Definition boxes
> A yellow box like this explains one technical word, the first time that the word occurs. If you know the word, skip the box.

Parts 3 and 4 opened the boxes of vLLM one by one. This part closes them again. It treats the whole system as one machine: requests go in on the left, answers come out, and a stopwatch measures each request. The map below is the paper's Figure 4, redrawn. The highlighted scheduler, with its KV cache manager, is the piece that changes from one system to the next.

{{FIG:p5_arch|The vLLM architecture, redrawn from the paper's Figure 4 as a map. Part 5 measures it from the outside. Each baseline keeps the workers and the model, and replaces the highlighted scheduler and memory manager with its own.}}

## How a serving system is tested

Before we read the paper's tests, we need the shape of any serving test. It is the same for a web server, a database or a language model.

> [!NOTE] Where this idea comes from: load testing
> Engineers tested web servers and databases this way long before language models. A program called a **load generator** plays thousands of users. It sends requests at a set pace and records when each answer comes back. Then it raises the pace and repeats.
>
> Think of testing a restaurant kitchen. You send in diners at one table a minute, then two, then three. At each pace you note how long a diner waits for the food. At some pace the kitchen cannot keep up, and the waits explode. That pace is the kitchen's capacity.

A serving test has three inputs. The first is the **requests**: for us, a prompt length and an answer length each. The second is the **arrival times**: when each request reaches the server. The third is the **server** itself, the system under test. The test keeps the first two fixed and swaps the third.

The test then reads two kinds of numbers. One is how much work the server finishes per second. The other is how long each user waits. The next two boxes name them.

> [!DEFINITION] Throughput
> The work a server finishes per second: requests per second, or tokens per second. Say a server finishes 3 requests of 100 tokens each in 2 seconds. Its throughput is 1.5 requests per second, or 150 tokens per second.

> [!DEFINITION] Latency
> The time from the moment a request arrives to the moment its last token is written. It includes the time the request waits in the queue before the server starts on it.

A good server has both: high throughput and low latency. The two pull against each other. A server can raise its throughput by packing more requests into each step, but each request then waits longer. So the paper does not report one number. It reports a curve: latency at each request rate.

## The test bench {§6.1}

> [!PAPER] · §6.1 · page 10
> [![The paragraph on model and server configurations, with 13B, 66B, and 175B parameters, LLaMA with 13B parameters and NVIDIA A100 highlighted](/img/papers/vllm/p5-setup.png)](/img/papers/vllm/p5-setup.png)
> [![Table 1 of the paper: for 13B, 66B and 175B models, the GPUs, total GPU memory, parameter size, memory for KV cache and maximum number of KV cache slots, with the last two rows highlighted](/img/papers/vllm/p5-table1.png)](/img/papers/vllm/p5-table1.png)
>
> **What it says:** the tests use OPT models with "13B, 66B, and 175B parameters" and LLaMA with "13B parameters", on "NVIDIA A100" GPUs in the cloud. Table 1 gives each server's memory for the KV cache (12 GB, 21 GB and 264 GB) and the number of KV cache slots (15.7K, 9.7K and 60.1K).
>
> **What it means:**
>
> Three servers, from one GPU to eight, as Part 4 explained. The last row of the table matters most for this part. A slot holds the keys and values of one token. The 13B server has room for 15.7 thousand tokens of cache, all requests together, and every scheduler in this part fights over those slots.

> [!DEFINITION] KV cache slot
> Room for the keys and values of one token, in every layer of the model. For OPT-13B one slot is 800 KB. A server with 15.7K slots can keep 15.7 thousand tokens in memory at once, shared among all its running requests.

Table 1 is a division: the memory for the KV cache over the size of one slot, from Part 2. A tiny case first. If one slot were 1 MB and the cache had 10 MB, there would be 10 slots. We check the real table with the real model configurations. The function gives the bytes of one slot; look at the last two columns of the output, ours and the paper's:

```python
def kv_bytes_per_token(layers, hidden, bytes_per_value=2):
    """A key and a value (2) of `hidden` numbers in every layer, 2 bytes each in FP16."""
    return 2 * layers * hidden * bytes_per_value
```

```text
== 1. Table 1: KV cache slots = memory for the KV cache / KV cache of one token
model     layers hidden  KB/token KV memory slots (GiB)  Table 1
OPT-13B       40   5120       800     12 GB      15,729    15.7K
OPT-66B       64   9216     2,304     21 GB       9,557     9.7K
OPT-175B      96  12288     4,608    264 GB      60,075    60.1K
our simulated server: 15,728 slots (OPT-13B on one A100); with 2048-token requests that is 7.68 requests
the same 15,728 slots of OPT-125M take 580 MB, so a laptop can hold them
```

Two rows match exactly once we read "GB" as binary gigabytes (GiB, $$2^{30}$$ bytes). For OPT-66B we get 9,557 slots, a little under the table's 9.7K; the table probably rounds the memory. The first row is our server for the rest of this part: 15,728 slots. Only 7.68 requests of the full 2,048 tokens fit in it. Remember that number.

## The workloads {§6.1}

> [!PAPER] · §6.1 · page 10
> [![The workloads paragraph, with input and output lengths to synthesize client requests, 8.4x longer input prompts and 5.8x longer outputs, and Poisson distribution highlighted](/img/papers/vllm/p5-workloads.png)](/img/papers/vllm/p5-workloads.png)
> [![Figure 11 of the paper: histograms of input and output lengths for ShareGPT, with means 161.31 and 337.99, and for Alpaca, with means 19.31 and 58.45](/img/papers/vllm/p5-figure11.png)](/img/papers/vllm/p5-figure11.png)
>
> **What it says:** the workloads come from two datasets, ShareGPT and Alpaca. The authors use only their "input and output lengths to synthesize client requests". ShareGPT has "8.4× longer input prompts and 5.8× longer outputs" than Alpaca. Arrival times follow a "Poisson distribution" at different request rates.
>
> **What it means:**
>
> A serving test needs two things: how long each request is, and when it arrives. The words of a request do not matter to memory, only its lengths do.
>
> ShareGPT is a set of real conversations that users shared from a chat service, with long prompts and long answers. Alpaca is a set of short instructions with short answers. The two datasets are the two ends of the scale. Short requests leave less room for memory waste than long ones.

> [!NOTE] Where the two datasets come from
> **ShareGPT** comes from a website of the same name. From December 2022, users of a popular chat assistant could share their conversations there with one click. People then collected tens of thousands of these shared chats into a public dataset. Its prompts are what real users typed, and its answers are what a strong model wrote back. So it is a fair picture of chat traffic: long questions, pasted documents, long answers.
>
> **Alpaca** comes from a Stanford project of 2023. It started from 175 example tasks that people had written by hand. Then a GPT-3.5 model wrote 52,000 more instructions with answers in the same style, a method called self-instruct. The first entry is "Give three tips for staying healthy." Most entries are that short, and so are their answers. The team used the data to fine-tune a 7B LLaMA model.
>
> Neither dataset has timestamps. Nobody recorded when each chat happened. That is why the paper invents the arrival times, as the next box explains.

> [!DEFINITION] Request rate
> How many requests arrive per second, on average. A rate of 2 means 7,200 requests in an hour, but not one every half second exactly.

> [!DEFINITION] Poisson arrivals
> Each request comes at a random moment, independent of all the others, like customers walking into a shop. The gap between two arrivals follows an exponential distribution: many short gaps, a few long ones. Only the average rate is fixed.

The picture shows the difference. Both rows have the same average rate of 1 request per second. The top row is a metronome. The bottom row is one random draw of Poisson arrivals, from the same generator our simulator uses.

{{FIG:p5_poisson|Regular arrivals against Poisson arrivals at the same average rate. The random row has clumps, three requests in less than a tenth of a second, and long quiet gaps. A server must absorb the clumps, so random arrivals are harder to serve than regular ones.}}

> [!NOTE] Where this idea comes from: Poisson and the telephone exchange
> The distribution carries the name of the French mathematician Siméon Denis Poisson, who described it in 1837. Its use for arrivals came later. Around 1909, the Danish engineer Agner Krarup Erlang studied calls at the Copenhagen telephone exchange. He found that calls arrive like this: many people, each deciding on their own, at random moments.
>
> The same model fits shoppers at a till, cars at a toll gate and users of a chat service. Each user acts alone, so the sum of their requests looks random. That is why load tests use it when real timestamps are missing.

We repeat the counting on 4,000 requests of each dataset, with the OPT tokenizer. For ShareGPT we take the first question and its answer. The paper does not say exactly how it cut the conversations, so our ShareGPT numbers cannot match to the token:

```text
== 2. Figure 11: prompt and output lengths (OPT tokenizer)
dataset   requests mean prompt mean output paper prompt paper output
ShareGPT      4000       148.0       294.3       161.31       337.99
Alpaca        4000        18.2        59.1        19.31        58.45
ShareGPT / Alpaca: prompts 8.1x longer, outputs 5.0x longer (paper: 8.4x and 5.8x)
```

{{FIG:p5_fig11|Our requests, counted with the OPT tokenizer. ShareGPT prompts and answers spread out to 2,000 tokens; Alpaca is packed near zero. Compare with the paper's Figure 11 above.}}

Alpaca matches closely: 18.2 and 59.1 tokens against the paper's 19.31 and 58.45. ShareGPT is near too, with 8.1 times longer prompts against the paper's 8.4. Note the long tail of ShareGPT answers: the unknown length of Part 2, now with real data.

The arrival times come from a few lines of code. Each gap is drawn from an exponential distribution with a mean of one over the rate. At 2 requests per second, the mean gap is half a second. The function adds the gaps one by one to get the arrival times:

```python
def poisson(n, rate, seed):
    """Arrival times of n requests: gaps drawn from an exponential distribution with mean 1 / rate."""
    r, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        t += r.expovariate(rate); out.append(t)
    return out
```

## The baselines {§6.1}

A **baseline** is the system you compare against. A new system is only as good as the strongest baseline it beats. The paper picks two families: the fastest engine of the day, and the best scheduler of the day.

> [!NOTE] Where FasterTransformer comes from
> FasterTransformer is an open-source library from NVIDIA, the company that makes the A100. It holds hand-tuned GPU code for the layers of a transformer: attention, the matrix products, the layer normalization. Several small steps are merged into one GPU program, so data makes fewer trips to memory. Its later work went into NVIDIA's TensorRT-LLM.
>
> It is an engine, not a server. It runs one batch as fast as it can, but it does not decide which requests go into the batch. Serving tools such as NVIDIA's Triton Inference Server do that part, with dynamic batching.

> [!DEFINITION] Dynamic batching
> A scheduler collects the requests that have arrived, up to a maximum batch size, and sends them to the GPU as one batch. The batch then runs until every request in it is done. New requests wait for the next batch. Part 2 called this the minibus.

> [!PAPER] · §6.1 · page 10
> [![The first lines on FasterTransformer, with highly optimized for latency highlighted](/img/papers/vllm/p5-ft.png)](/img/papers/vllm/p5-ft.png)
> [![The rest of the paragraph, with dynamic batching, maximum batch size, large as possible and earliest arrived requests highlighted](/img/papers/vllm/p5-ft-2.png)](/img/papers/vllm/p5-ft-2.png)
>
> **What it says:** FasterTransformer is an inference engine "highly optimized for latency". It has no scheduler of its own, so the authors add one with "dynamic batching". They set the "maximum batch size" as "large as possible" for the GPU memory, and the scheduler sends the "earliest arrived requests" as one batch.
>
> **What it means:**
>
> FasterTransformer has very fast GPU code, but it is the minibus of Part 2. A batch runs until its longest request ends, and new requests wait for the next trip. It also books the full maximum length for every request. If vLLM beats it, the reason cannot be faster arithmetic; it must be scheduling and memory.

> [!NOTE] Where Orca comes from
> Orca is a serving system from a team at Seoul National University and FriendliAI, published at the OSDI conference in 2022. Its main idea is **iteration-level scheduling**, the city bus of Part 2. The scheduler looks at the batch after every single token. A finished request leaves at once, and a waiting request joins at once.
>
> Orca also gave this paper its yardstick. The normalized latency of the next section is the metric that the Orca paper used. So vLLM is tested against Orca on Orca's own measure.
>
> One part of Orca stayed old-fashioned: memory. Like FasterTransformer, it gives each request one contiguous chunk of KV cache when the request joins. The chunk must be large enough for the whole answer, so Orca must guess the answer length in advance.

> [!PAPER] · §6.1 · page 10
> [![The Orca baseline paragraph with its three versions, with implement our own version of Orca, buddy allocation algorithm, infeasible to achieve in practice, reserves 32 positions for outputs and i.e., 2048 tokens highlighted](/img/papers/vllm/p5-orca.png)](/img/papers/vllm/p5-orca.png)
>
> **What it says:** Orca is not public, so the authors "implement our own version of Orca". They assume it uses the "buddy allocation algorithm". There are three versions. **Orca (Oracle)** knows the real output length in advance, which is "infeasible to achieve in practice". **Orca (Pow2)** books up to twice the output length: for 25 tokens it "reserves 32 positions for outputs". **Orca (Max)** books up to the maximum, "i.e., 2048 tokens".
>
> **What it means:**
>
> Orca is the strong baseline. It has iteration-level batching, the city bus of Part 2, and only its memory works the old way: one contiguous chunk per request.
>
> Orca must guess how long each answer will be. Max guesses the worst case, Pow2 guesses within a factor of two, and Oracle never guesses wrong, which no real system can do. If vLLM beats even Oracle, then better guessing would never close the gap.

> [!DEFINITION] Buddy allocator
> A classic way to hand out memory. Every chunk is a power of two in size: 16, 32, 64 slots and so on. A request for 25 slots gets a chunk of 32. When a chunk is freed, it merges with its "buddy", the neighbour of the same size, if that one is free too. It is fast, but it wastes the rounding.

> [!NOTE] Where the buddy allocator comes from
> Kenneth Knowlton described the buddy system in 1965, for the memory of early computers. It is still in use: Linux hands out the physical pages of a computer's memory with a buddy allocator. The rule is easy to check. A chunk of 64 slots splits into two buddies of 32, and two free buddies of 32 join back into 64.
>
> The paper does not know what allocator Orca used. It assumes a buddy allocator because it is the standard choice for chunks of mixed sizes.

The three Orca versions differ only in how much answer they book. Max books for the worst case, Pow2 books up to twice the real answer, and Oracle books the real answer. In our simulator, this function gives the slots each version books for a prompt of $$p$$ tokens and an answer of $$o$$ tokens. Then our buddy allocator rounds that up to a power of two:

```python
def reserve(system, p, o):
    """Slots that an Orca version books for a request at admission (before the buddy allocator rounds it up)."""
    if system == 'Orca (Max)':
        return 2048                                            # the model's maximum sequence length
    if system == 'Orca (Pow2)':
        return p + (1 << (o - 1).bit_length())                 # output booked up to 2x: 25 -> 32
    return p + o                                               # Oracle: knows the real output length
```

A worked example. A request has a prompt of 19 tokens and an answer of 25. Oracle books $$19 + 25 = 44$$ slots, and the buddy allocator gives it 64. Pow2 books $$19 + 32 = 51$$, which also becomes 64. Max books 2,048.

With 16-token blocks, vLLM holds 2 blocks (32 slots) right after the prompt. At the end, after all 44 tokens, it holds 3 blocks (48 slots).

{{FIG:p5_reserve|One request, five systems. Blue is the prompt, orange is the answer that the model really writes, and the dashed red part is memory set aside but never used. Only vLLM keeps the waste below one block.}}

The picture is the whole experiment in small. Every system writes the same 44 tokens. FasterTransformer and Orca (Max) block 2,048 slots to do it, about 46 times more than needed. Even the perfect guess of Oracle loses 20 slots to the buddy rounding. The vLLM bar loses 4, and it takes its blocks one at a time, as the answer grows.

## Normalized latency {§6.1}

> [!PAPER] · §6.1 · page 10
> [![The key metrics paragraph, with normalized latency, end-to-end latency divided by its output length, retain low normalized latency against high request rates and 1-hour traces highlighted](/img/papers/vllm/p5-metrics.png)](/img/papers/vllm/p5-metrics.png)
>
> **What it says:** the main measure is "normalized latency": each request's "end-to-end latency divided by its output length", averaged over all requests. A high-throughput system should "retain low normalized latency against high request rates". Most tests use "1-hour traces".
>
> **What it means:**
>
> A 500-token answer naturally takes longer than a 5-token answer, so plain latency would be unfair. Dividing by the output length gives seconds per token for long and short requests alike. It includes the time spent in the queue, so a crowded server shows up as a high number.
>
> The test turns one knob, the request rate, and watches this one number. It does not ask "how many tokens per second at full load", but "how much load before the users notice".

> [!DEFINITION] Normalized latency
> For one request: (time it finished minus time it arrived) divided by the number of tokens in its answer, in seconds per token. The paper reports the mean over all requests. A request that waited 4 seconds and then wrote 100 tokens in 2 seconds has a normalized latency of 6 / 100 = 0.06 s/token.

In symbols, with $$N$$ requests, arrival times $$a_i$$, finish times $$f_i$$ and output lengths $$o_i$$:

$$
\text{normalized latency} = \frac{1}{N} \sum_{i=1}^{N} \frac{f_i - a_i}{o_i}
$$

Read it from left to right. It is an average over the $$N$$ requests. Each term is $$f_i - a_i$$, the time request $$i$$ spent waiting plus running, divided by its $$o_i$$ answer tokens.

A tiny example with three requests, small enough to check by hand:

| request | arrives | finishes | answer tokens | (finish - arrive) / tokens |
|---|---|---|---|---|
| A | 0 s | 2 s | 100 | 2 / 100 = 0.02 |
| B | 1 s | 5 s | 50 | 4 / 50 = 0.08 |
| C | 1.5 s | 3.5 s | 10 | 2 / 10 = 0.20 |

The mean is $$(0.02 + 0.08 + 0.20) / 3 = 0.10$$ s/token. Request B waited in a queue, and its number is four times that of A. Request C wrote only 10 tokens, so each of its seconds weighs heavily. The short requests count as much as the long ones in the mean.

The paper says it focuses on throughput, but it plots latency. The two are linked. The throughput of a server, in the paper's sense, is the highest request rate it can take while the normalized latency stays low. A user never sees throughput. A user sees the wait for each token.

## The simulator

We cannot run OPT-13B on an A100 here. We can rebuild the experiment's logic and keep the part that the paper is about: memory. Our simulator has five pieces.

1. **Requests**: 1,000 real (prompt, answer) lengths from ShareGPT or Alpaca, with Poisson arrival times.
2. **Memory**: 15,728 slots, the Table 1 budget of OPT-13B on one A100.
3. **Five ways to store the cache**: FasterTransformer, three Orca versions with a real buddy allocator, and vLLM. Its 16-token blocks come with preemption by recomputation (Part 4, and the box below).
4. **A scheduler**: first come, first served, at most 256 running requests, as in vLLM's default setting.
5. **A clock**: each iteration costs a time from a step-time model measured on our laptop.

> [!DEFINITION] Preemption by recomputation
> When memory runs out, the scheduler stops a running request and frees all its blocks. The request goes back to the queue. When there is room again, one prompt pass over its prompt and the tokens written so far rebuilds the KV cache.

> [!DEFINITION] Iteration
> One step of the model for the whole batch. Each running request gets one new token, and each new request gets its prompt processed. A 100-token answer needs 100 iterations.

A step with more requests, or with longer caches, takes longer. We time real decode steps of OPT-125M on the laptop GPU for many batch sizes and cache lengths, and fit a straight line:

```python
def step_time(n_decode, cached, prompt_tokens):
    """One iteration: the fixed cost, plus each decoding request, plus each cached token it reads, plus prompt tokens."""
    return A_STEP + B_REQ * n_decode + C_TOK * cached + E_PRE * prompt_tokens
```

```text
== 3. Step-time model: OPT-125M on the laptop GPU (FP16, fastest of 15 runs)
batch context decode step ms
    1      64          10.47
    1     512          11.88
    1    1024          13.14
    8      64          12.40
    8     512          14.01
    8    1024          21.82
   32      64          16.91
   32     512          32.51
   32    1024          60.79
   64      64          25.13
   64     256          31.63
   64     512          73.90
  128      64          25.64
  128     256          81.49
fit: decode step = 8.62 ms + 78.0 us per request + 1667.8 ns per cached token (largest error of the fit 29%)
prompt pass ms for 64, 128, 256, 512, 1024 tokens: 12.3, 14.3, 16.1, 32.2, 49.8
fit: prompt pass = 8.83 ms + 40.6 us per prompt token
```

One decode step costs 8.62 ms even for a single request: the fixed cost of a pass through the model, which batching shares. Each cached token adds about 1.7 microseconds, because attention must read it. The fit is off by up to 29%, and the laptop was busy with other work. So treat our seconds as a shape, not as a server measurement.

The heart of the simulator is the vLLM path. Before each iteration, every running request needs room for one more token. If a new block is needed and none is free, the latest arrival gives up all its blocks. It goes back to the front of the queue.

A small case shows the rule. Say 2 blocks are free, and three running requests each need a new block. The first two get them. The third finds none, so the request that arrived last is preempted, and its blocks pay for the third. In the code, `need` is the number of new blocks a request needs, `free` the free blocks, and `hold` the blocks each request holds:

```python
        if paged:                                              # room for one more token in each running request
            k = 0
            while k < len(running):
                i = running[k]
                need = -(-(P[i] + gen[i] + 1) // block) - hold[i]
                while need > free and running[-1] != i:        # preempt the latest arrival: free all its blocks
                    v = running.pop(); free += hold.pop(v); waiting.appendleft(v); preempt += 1; preempted = True
                if need > free:                                # i itself is the latest arrival
                    running.pop(); free += hold.pop(i); waiting.appendleft(i); preempt += 1; preempted = True
                    break
                free -= need; hold[i] += need; k += 1
```

The expression `-(-x // block)` is the number of blocks that $$x$$ tokens need, rounded up. For 17 tokens and blocks of 16, it gives 2. In the Orca path there is no such loop. A request joins only if the buddy allocator finds a chunk for its whole reservation, and it never needs more.

## Figure 12: latency against request rate {§6.2}

### How to read a latency plot

The paper's main figures all have the same shape, so it pays to learn to read one first. The picture below is drawn on our own ShareGPT run, with only two systems. Read the four numbered marks in order.

{{FIG:p5_howto|Anatomy of a latency plot. (1) At low load every system sits on the same flat floor. (2) At the knee the batch is full and a queue starts. (3) Past the knee the line becomes a wall. (4) The system whose knee is further right serves more users on the same GPU.}}

The horizontal axis is the load: requests per second. The vertical axis is the normalized latency: how slow each token feels to a user. A point on a line answers one question. If users arrive at this rate, how slow does this system feel?

On the floor, the server has spare room. Each request is served at once, and the latency is just the time to compute a token. At the knee, the batch is as large as memory allows, and newcomers start to wait. The wall follows: the queue grows, and so does every wait. So the plot is not about the height of the lines. It is about where each line leaves the floor.

### The paper's plot

> [!PAPER] · Figure 12 · page 10
> [![Figure 12 of the paper: six panels of normalized latency against request rate for OPT-13B, 66B and 175B on ShareGPT and Alpaca. In each panel FasterTransformer rises first, then Orca Max, Pow2 and Oracle, and vLLM last](/img/papers/vllm/p5-figure12.png)](/img/papers/vllm/p5-figure12.png)
>
> **What it says:** "Single sequence generation with OPT models on the ShareGPT and Alpaca dataset." Each panel plots normalized latency in s/token against the request rate, for the three models (columns) and two datasets (rows).
>
> **What it means:**
>
> This is the main result of the paper, so learn to read one panel. The horizontal axis is the load in requests per second. The vertical axis is how slow each token feels to a user, cut off at 1 second per token. Each line is one system.
>
> Every line is flat at first and then turns up almost vertically. The point where it turns is the system's capacity. In panel (a), FasterTransformer turns before 0.5 requests per second. Orca (Max) turns near 0.5, Orca (Oracle) near 1.1, and vLLM near 1.9. Further right means more users served on the same GPU.

> [!PAPER] · §6.2 · page 11
> [![The paragraph on basic sampling, with initially increases at a gradual pace, suddenly explodes and queue length continues to grow infinitely highlighted](/img/papers/vllm/p5-explode.png)](/img/papers/vllm/p5-explode.png)
>
> **What it says:** as the request rate grows, the latency "initially increases at a gradual pace" but then "suddenly explodes". When the rate passes the capacity of the system, the "queue length continues to grow infinitely", and so does the latency.
>
> **What it means:**
>
> Think of a supermarket checkout. While customers arrive more slowly than the cashier serves them, the queue stays short. If they arrive even slightly faster, the queue grows without end, and every customer waits a little longer than the one before.
>
> That is why the curves bend so sharply. Below capacity, extra load only fills idle slots in the batch. Above capacity, the waiting time grows with the length of the test, so the exact height above the bend means little. The position of the bend is the result.

> [!DEFINITION] Capacity in a latency plot
> The request rate where a system's curve bends up. Two systems are compared by how far apart their bends are, at the same latency. "2 times higher request rate" means the second system bends at twice the rate of the first.

### Why the bend is so sharp

> [!NOTE] Where this idea comes from: queueing theory
> Erlang's telephone studies grew into a whole field, **queueing theory**: the mathematics of waiting lines. It explains why a road near its capacity jams so suddenly. On a road at half its capacity, one extra car changes nothing. On a road at 95% of capacity, one car that brakes starts a jam that lasts long after it has gone. A serving system is a road for requests, and the curves in Figure 12 are its traffic jams.

The simplest queue has one cashier and random arrivals. Customers arrive at a rate $$\lambda$$ (lambda) per minute, and the cashier serves $$\mu$$ (mu) per minute on average, also at random. Queueing theory gives the mean time a customer spends in the shop, queue and checkout together:

$$
W = \frac{1}{\mu - \lambda}
$$

Read it from left to right. $$W$$ is the mean time in the shop. The bottom, $$\mu - \lambda$$, is the spare speed of the cashier: how many more customers a minute it could serve. The formula holds only while $$\lambda < \mu$$.

A worked example with a cashier who serves $$\mu = 10$$ customers a minute. At $$\lambda = 5$$, $$W = 1/5$$ of a minute, or 12 seconds. At $$\lambda = 8$$, $$W = 1/2$$ minute, 30 seconds. At $$\lambda = 9$$, $$W = 1$$ minute. At $$\lambda = 9.9$$, $$W = 10$$ minutes. The last 10% of load costs far more than the first 90%.

{{FIG:p5_queue|The cashier's queue from the formula above. The mean time in the shop stays low up to about half the cashier's speed, then climbs ever faster, and has no limit at 10 customers a minute. It is the same shape as each line in Figure 12.}}

A language model server is not a single cashier. It serves a whole batch at once, and its speed changes with the batch. But the shape carries over. Its "cashier speed" is the rate at which its memory lets requests through. vLLM raises that speed, because more requests fit in memory. So its wall moves to the right.

Here is our simulator on the same experiment. Each point is a run of 1,000 requests. We use OPT-125M step times and the 15,728 slots of OPT-13B. Each row is one request rate, and each column one system. Read down a column: the number stays near 0.01, then jumps. Look at which column jumps first:

```text
== 4. Figure 12 rebuilt: normalized latency (s/token) against request rate, 1000 requests per point, 15,728 slots
normalized latency = mean over requests of (finish time - arrival time) / output tokens
ShareGPT:
 req/s FasterTransformer        Orca (Max)       Orca (Pow2)     Orca (Oracle)              vLLM
  0.25             0.089             0.010             0.010             0.010             0.010
   0.5             2.759             0.011             0.011             0.011             0.011
     1            13.667             0.014             0.013             0.012             0.012
   1.5            17.891             0.086             0.027             0.015             0.015
     2            20.039             1.445             0.488             0.033             0.018
   2.5            21.222             2.587             1.581             0.363             0.024
     3            22.011             3.355             2.358             1.081             0.076
   3.5            22.575             3.904             2.887             1.640             0.401
     4            22.998             4.321             3.267             2.000             0.809
     5            23.589             4.907             3.846             2.573             1.345
     6            23.984             5.298             4.261             2.968             1.727
     8            24.477             5.788             4.741             3.415             2.197
Alpaca:
 req/s FasterTransformer        Orca (Max)       Orca (Pow2)     Orca (Oracle)              vLLM
     5             1.959             0.010             0.010             0.010             0.010
    10             5.701             0.031             0.010             0.010             0.010
    20             7.607             1.391             0.012             0.012             0.012
    30             8.242             2.020             0.015             0.015             0.015
    40             8.559             2.336             0.055             0.019             0.019
    60             8.742             2.652             0.307             0.095             0.029
    80             8.964             2.810             0.448             0.213             0.053
   100             9.059             2.905             0.523             0.292             0.096
   120             9.115             2.968             0.573             0.357             0.131
   140             9.117             3.013             0.624             0.395             0.164
   160             9.152             3.047             0.650             0.426             0.188
   200             9.200             3.095             0.692             0.468             0.225
```

{{FIG:p5_fig12|Our simulation of Figure 12, cut at 1.0 s/token as in the paper. The order of the bends is the paper's order: FasterTransformer, then Orca (Max), Orca (Pow2), Orca (Oracle) and vLLM. FasterTransformer on Alpaca is above 1.0 s/token even at 5 requests per second, so its line is off the chart.}}

At low load, all four iteration-level systems give the same 0.010 s/token on ShareGPT: when nobody waits, memory management does not matter. Then the bends appear in the paper's order. To turn bends into "times higher", we draw a line at 0.1 s/token, about ten times the unloaded latency. Then we read where each curve crosses it:

```python
def crossing(rates, vals, limit=LIMIT):
    """The rate at which the curve first rises above the limit, read between the two points around it."""
    if vals[0] > limit:
        return 0.0
    for (r0, v0), (r1, v1) in zip(zip(rates, vals), zip(rates[1:], vals[1:])):
        if v1 > limit:
            return r0 + (r1 - r0) * (limit - v0) / (v1 - v0)
    return float('inf')
```

```text
request rate where normalized latency crosses 0.1 s/token (our "same latency" line, linear between points):
ShareGPT  FasterTransformer 0.3, Orca (Max) 1.5, Orca (Pow2) 1.6, Orca (Oracle) 2.1, vLLM 3.0
          vLLM / Orca (Oracle) = 1.4x, vLLM / Orca (Pow2) = 1.9x, vLLM / Orca (Max) = 2.0x
Alpaca    FasterTransformer < 5, Orca (Max) 10.5, Orca (Pow2) 43.6, Orca (Oracle) 60.8, vLLM 102.3
          vLLM / Orca (Oracle) = 1.7x, vLLM / Orca (Pow2) = 2.3x, vLLM / Orca (Max) = 9.7x
paper, ShareGPT: vLLM sustains 1.7x to 2.7x the rate of Orca (Oracle), 2.7x to 8x that of Orca (Max), up to 22x that of FasterTransformer
```

{{FIG:p5_read|How to read "times higher". Fix a latency (the dashed line), find where each curve crosses it, and divide the two rates. On our ShareGPT run, Orca (Oracle) crosses at 2.1 requests per second and vLLM at 3.0.}}

The choice of 0.1 s/token is ours; the paper does not give the latency level it used. Another line would give other ratios, but in our table the order of the systems is the same at every rate.

## How much faster, and why {§6.2}

> [!PAPER] · §6.2 · page 11
> [![The ShareGPT results paragraph, with 1.7x to 2.7x higher request rates compared to Orca (Oracle), 2.7x to 8x, 2.2x more requests, 4.3x more requests and up to 22x higher request rates highlighted](/img/papers/vllm/p5-sharegpt.png)](/img/papers/vllm/p5-sharegpt.png)
>
> **What it says:** on ShareGPT, vLLM sustains "1.7×–2.7×" higher request rates than Orca (Oracle) and "2.7×–8×" higher than Orca (Max), at similar latency. The reason is the batch: for OPT-13B, vLLM runs "2.2× more requests" at once than Orca (Oracle) and "4.3× more requests" than Orca (Max). Against FasterTransformer it reaches "up to 22× higher request rates".
>
> **What it means:**
>
> Each range is the smallest and the largest gain over the three models. Our simulated 13B server gives 1.4 times Orca (Oracle) and 2.0 times Orca (Max) on ShareGPT, smaller gains for a reason we come to below.
>
> The second half of the paragraph names the cause. More requests fit in memory at once, so each step of the model does more useful work. The cost of reading the weights is shared by more requests. Nothing about the model or its GPU code got faster. FasterTransformer loses by the most because it adds request-level batching to Orca (Max)'s memory waste.

> [!PAPER] · Figure 13 · page 10
> [![Figure 13 of the paper: bar charts of the average number of batched requests. ShareGPT: Orca Max 7.00, Pow2 9.81, Oracle 13.62, vLLM 30.42. Alpaca: 7.00, 43.24, 72.75, 132.44](/img/papers/vllm/p5-figure13.png)](/img/papers/vllm/p5-figure13.png)
>
> **What it says:** "Average number of batched requests" when serving OPT-13B on ShareGPT at 2 requests per second and Alpaca at 30. ShareGPT: 7.00, 9.81, 13.62 and 30.42 for Orca (Max), Orca (Pow2), Orca (Oracle) and vLLM. Alpaca: 7.00, 43.24, 72.75 and 132.44.
>
> **What it means:**
>
> This figure is the mechanism behind Figure 12. Look at Orca (Max): exactly 7.00 on both datasets. That is not a measurement of the data at all. It is 15.7K slots divided by 2,048 slots per request, rounded down. Orca (Max) books the same chunk for a short question and a long chat, so the data cannot change it.
>
> The other bars depend on the data. On ShareGPT, vLLM runs 30 requests at once against 14 for the best possible Orca.

Our simulator gives the same bars. We let all 1,000 requests arrive at time 0, so the queue is never empty, and we count the running requests at each iteration. We also check the count with a second method. When memory is full, the slots held by the running requests add up to the budget. A request stays in the batch for $$o$$ steps, so a running request is a long request more often than a short one. That gives a closed form:

$$
\text{batch} \approx S \cdot \frac{\sum_i o_i}{\sum_i \sum_{g=1}^{o_i} m_i(g)}
$$

Read it from left to right. $$S$$ is the budget, 15,728 slots. The top sum adds up the answer lengths, so it counts request-steps. The bottom sum adds up $$m_i(g)$$, the slots that request $$i$$ holds after writing $$g$$ tokens, over its whole life.

The fraction is one over the slots of an average running request. For Orca, $$m_i(g)$$ is its buddy chunk at every step; for vLLM, the blocks it holds so far.

A tiny example. Two requests, with answers of 1 and 3 tokens. Each holds 4 slots at every step. The sums are $$1 + 3 = 4$$ request-steps and $$4 \cdot 1 + 4 \cdot 3 = 16$$ slot-steps, so a running request holds 4 slots on average. With $$S = 40$$, the batch is $$40 \cdot 4 / 16 = 10$$.

```python
def closed_form_batch(system, reqs, block=16, slots=S):
    """Method 2: when memory is full, the slots held by the running requests add up to the budget. A request is
    running for o steps, so a random running request is picked in proportion to o. Fragmentation is ignored."""
    held = 0
    for p, o in reqs:
        if system == 'vLLM':
            held += sum(block * -(-(p + g) // block) for g in range(1, o + 1))   # blocks grow as it writes
        else:
            held += o * (1 << (reserve(system, p, o) - 1).bit_length())         # one chunk for its whole life
    return slots * sum(o for _, o in reqs) / held
```

```text
== 5. Figure 13 rebuilt: requests running at once while the queue is never empty (all arrive at time 0)
ShareGPT: system             simulated closed form tokens/step  tokens/s preemptions
          Orca (Max)              7.00        7.68         6.9       495           0
          Orca (Pow2)             9.30       10.55         9.1       583           0
          Orca (Oracle)          14.91       17.08        14.5       731           0
          vLLM                   37.54       38.02        34.8       952        1443
          vLLM against Orca (Oracle): 2.4x the tokens per step, 1.3x the tokens per second (bigger batches make each step slower)
          vLLM runs 2.5x the requests of Orca (Oracle) and 5.4x those of Orca (Max)
Alpaca: system             simulated closed form tokens/step  tokens/s preemptions
          Orca (Max)              7.00        7.68         6.9       678           0
          Orca (Pow2)            43.98       46.55        36.6     2,211           0
          Orca (Oracle)          75.91       82.34        57.3     2,723           0
          vLLM                  204.01      183.95       103.9     3,296         274
          vLLM against Orca (Oracle): 1.8x the tokens per step, 1.2x the tokens per second (bigger batches make each step slower)
          vLLM runs 2.7x the requests of Orca (Oracle) and 29.1x those of Orca (Max)
paper, Figure 13 (OPT-13B, A100): ShareGPT 7.00, 9.81, 13.62, 30.42; Alpaca 7.00, 43.24, 72.75, 132.44
note: we cap a batch at 256 requests, as vLLM did by default
```

{{FIG:p5_fig13|Requests in the batch, our simulation next to the paper's Figure 13. The Orca bars are close to the paper's on both datasets. Our vLLM bars are higher than the paper's, most of all on Alpaca.}}

Orca (Max) gives exactly 7.00, as in the paper. The closed form says 7.68, and the gap is the buddy allocator. The budget of 15,728 is not a power of two, so it is split into pieces of 8,192, 4,096 and fewer slots. Only seven chunks of 2,048 fit in them. That lost 0.68 of a request is external fragmentation: free memory cut into pieces too small to use. It appears on its own.

For Orca (Pow2) and Orca (Oracle), the simulation sits 5% to 13% below the closed form: chunks of mixed sizes leave holes. For vLLM on ShareGPT, the two methods agree within 1.3%, because equal blocks leave no holes. On Alpaca, the 256-request cap and the 274 preemptions break the formula's assumption of a steady, full memory.

Our Orca bars are close to the paper's: 9.30 against 9.81, 14.91 against 13.62, 43.98 against 43.24, 75.91 against 72.75. Memory counts depend only on lengths and budget, so a laptop can get them right. Our vLLM bars are higher, 37.54 against 30.42 and 204 against 132, because a real server has limits that we leave out.

The picture below puts the ShareGPT bars into memory. Each bar is the same 15,728 slots. It is cut into one piece per running request, and the blue part of each piece is what an average running request really holds.

{{FIG:p5_memory|Before vLLM and with vLLM, on the same memory. Orca (Max) cuts it into 7 chunks of 2,048 slots, mostly empty, and 1,392 slots are left over, too small for an eighth chunk. Orca (Oracle) fits about 15 requests, with empty room and holes in each. vLLM fills the memory with about 37 requests.}}

The blue part is the same size in all three bars: about 419 slots, or 15,728 divided by vLLM's 37.54 requests. That is fair, because a running request holds the same tokens whatever system serves it. Only the empty part around it changes. Remove the empty part, and the same memory holds 5.4 times as many requests as Orca (Max).

Now the reason our speed gains are smaller. On ShareGPT, vLLM writes 2.4 times as many tokens per step as Orca (Oracle), but only 1.3 times as many per second. On our laptop, a step with 38 requests and their caches takes much longer than a step with 15. On an A100, each step must read 26 GB of weights, a fixed cost that a bigger batch shares. The next box shows what happens when that stops being true.

> [!PAPER] · §6.2 · page 11
> [![The Alpaca results paragraph, with less pronounced, large GPU memory space, short sequences and compute-bound rather than memory-bound highlighted](/img/papers/vllm/p5-alpaca.png)](/img/papers/vllm/p5-alpaca.png)
>
> **What it says:** Alpaca shows the same trend, with one exception. In Fig. 12 (f), vLLM's advantage over Orca (Oracle) and Orca (Pow2) is "less pronounced". The 175B server has "large GPU memory space" for the cache, while Alpaca has "short sequences". So the Orca versions also batch many requests, and the system becomes "compute-bound rather than memory-bound".
>
> **What it means:**
>
> This is the paper being honest about its limits. vLLM's gain comes from fitting more requests into memory. The 175B server has 60.1K slots, and Alpaca requests use only about 80 tokens each. Hundreds of them fit even with waste, so there is little left to gain.
>
> Then the limit moves to arithmetic. Each step must do the work of every request in the batch, and saving memory cannot speed that up. Our laptop shows the same effect at a smaller scale. Our step time grew with our batches, and the speed gain shrank below the memory gain.

> [!DEFINITION] Compute-bound and memory-bound
> A system is **compute-bound** when its arithmetic sets the speed, and **memory-bound** when memory sets it. In this paper, "memory-bound" means the amount of KV cache memory limits how many requests run at once. Freeing memory helps only a memory-bound system.

## Parallel sampling and beam search {§6.3}

> [!PAPER] · Figure 14 · page 11
> [![Figure 14 of the paper: normalized latency against request rate for parallel generation with 2, 4 and 6 samples, and beam search with width 2, 4 and 6, OPT-13B on Alpaca. vLLM bends last in every panel](/img/papers/vllm/p5-figure14.png)](/img/papers/vllm/p5-figure14.png)
> [![The paragraph on parallel sampling and beam search results, with share the KV cache for the prompt, vLLM brings more improvement, allows for more sharing and from 1.3x in basic sampling to 2.3x in beam search with a width of 6 highlighted](/img/papers/vllm/p5-parallel.png)](/img/papers/vllm/p5-parallel.png)
>
> **What it says:** in parallel sampling, all samples of one request "share the KV cache for the prompt". With more samples, "vLLM brings more improvement" over Orca. Beam search "allows for more sharing", so the gain is larger still. Over Orca (Oracle), it goes "from 1.3× in basic sampling to 2.3× in beam search with a width of 6".
>
> **What it means:**
>
> Read Figure 14 like Figure 12. From left to right, each request asks for more outputs: 2, 4 or 6 samples on top, 2, 4 or 6 beams below. Every system bends earlier, but Orca bends much earlier than vLLM. Orca stores the prompt once per output, and vLLM once per request.
>
> Beam search gains the most because its beams share more than the prompt. As Part 4 showed, beams often agree for many tokens before they split, and vLLM keeps one copy of each shared block.

> [!PAPER] · Figure 15 · page 11
> [![Figure 15 of the paper: memory saving from sharing KV blocks for OPT-13B on Alpaca. Parallel sampling with 2, 4, 6 outputs: 6.09, 8.53, 9.79 percent. Beam search with width 2, 4, 6: 37.56, 53.13, 55.16 percent](/img/papers/vllm/p5-figure15.png)](/img/papers/vllm/p5-figure15.png)
> [![The paragraph on memory saving, with number of blocks we saved by sharing divided by the number of total blocks without sharing, 6.1% - 9.8%, 37.6% - 55.2%, 16.2% - 30.5% and 44.3% - 66.3% highlighted](/img/papers/vllm/p5-saving.png)](/img/papers/vllm/p5-saving.png)
>
> **What it says:** the saving is "the number of blocks we saved by sharing divided by the number of total blocks without sharing". On Alpaca it is "6.1% - 9.8%" for parallel sampling and "37.6% - 55.2%" for beam search. On ShareGPT it is "16.2% - 30.5%" and "44.3% - 66.3%".
>
> **What it means:**
>
> Parallel sampling saves little on Alpaca because Alpaca prompts are short. A prompt of 18 tokens is one full block of 16 plus a partial one. Only the full block stays shared; everything after it belongs to one sample. ShareGPT prompts are about eight times longer, so far more blocks are shared, and the saving roughly triples.
>
> Beam search saves much more, because beams share parts of their answers too, not only the prompt.

We rebuild the parallel sampling bars two ways. Method 1 is a small block manager with reference counts and copy-on-write, as in Part 4. Method 2 is a closed form. After the first step, only the $$\lfloor p / 16 \rfloor$$ full prompt blocks stay shared, so with $$n$$ samples and $$\ell = p + g$$ tokens each:

$$
\text{blocks without sharing} = n \left\lceil \frac{\ell}{16} \right\rceil, \qquad \text{blocks with sharing} = \left\lfloor \frac{p}{16} \right\rfloor + n \left( \left\lceil \frac{\ell}{16} \right\rceil - \left\lfloor \frac{p}{16} \right\rfloor \right)
$$

Read it this way. $$\lceil \ell / 16 \rceil$$ is the blocks one sample needs for $$\ell$$ tokens, rounded up. Without sharing, each of the $$n$$ samples has all of them. With sharing, the full prompt blocks are stored once, and each sample keeps only its own blocks after them.

A tiny example: $$p = 18$$, two samples, each with 6 new tokens, so $$\ell = 24$$. Without sharing, $$2 \cdot 2 = 4$$ blocks. With sharing, $$1 + 2 \cdot (2 - 1) = 3$$ blocks. That saves 1 block of 4, or 25%, at this step.

```python
def sharing_refcount(p, o, n, block=16):
    """Method 1: a block manager. The prompt blocks are mapped by all n samples (reference count n). When a sample
    writes into a block that others still map, it gets its own copy first (copy-on-write). Each step we count the
    physical blocks in use and the blocks that n separate copies would need."""
    ref, nxt = {}, 0

    def new_block():
        nonlocal nxt
        ref[nxt] = 1; nxt += 1
        return nxt - 1
    prompt_blocks = [new_block() for _ in range(-(-p // block))]
    for b in prompt_blocks:
        ref[b] = n
    tables, length = [list(prompt_blocks) for _ in range(n)], [p] * n
    with_sharing = without = 0
    for _ in range(o):                                         # each step, every sample writes one token
        for s in range(n):
            if length[s] % block == 0:
                tables[s].append(new_block())                 # the last block is full: take a fresh one
            elif ref[tables[s][-1]] > 1:                       # shared and about to change: copy-on-write
                ref[tables[s][-1]] -= 1; tables[s][-1] = new_block()
            length[s] += 1
        with_sharing += sum(1 for c in ref.values() if c > 0)
        without += sum(len(t) for t in tables)
    return with_sharing, without
```

```text
== 6. Figure 15a rebuilt: memory saved by sharing prompt blocks in parallel sampling (block size 16)
ShareGPT  2 samples: saved 20.74% (block manager), 20.74% (closed form)
ShareGPT  4 samples: saved 31.10% (block manager), 31.10% (closed form)
ShareGPT  6 samples: saved 34.56% (block manager), 34.56% (closed form)
Alpaca    2 samples: saved  4.51% (block manager),  4.51% (closed form)
Alpaca    4 samples: saved  6.76% (block manager),  6.76% (closed form)
Alpaca    6 samples: saved  7.52% (block manager),  7.52% (closed form)
paper, Figure 15a (Alpaca): 6.09%, 8.53%, 9.79%; §6.3 (ShareGPT): 16.2% to 30.5%
assumption: all samples of a request write as many tokens as the dataset answer (real samples differ)
```

The block manager and the formula agree to every digit. Our Alpaca savings, 4.51% to 7.52%, are a little under the paper's 6.09% to 9.79%. Our ShareGPT savings, 20.74% to 34.56%, are a little over its 16.2% to 30.5%. The pattern is the same: more samples save more, with diminishing returns, and long prompts save far more. One assumption shapes our numbers: every sample writes as many tokens as the dataset's answer, while real samples stop at different times. We did not rebuild beam search, whose sharing depends on the model's choices.

## Shared prefixes and chatbots {§6.4}

> [!PAPER] · §6.4 · page 12
> [![The shared prefix paragraph, with LLaMA-13B, English-to-German translation, single example, 5 examples, 1.67x higher throughput and 3.58x higher throughput highlighted](/img/papers/vllm/p5-prefix.png)](/img/papers/vllm/p5-prefix.png)
> [![Figure 16 of the paper: normalized latency against request rate for Orca Oracle and vLLM with a 1-shot prefix of 80 tokens and a 5-shot prefix of 341 tokens](/img/papers/vllm/p5-figure16.png)](/img/papers/vllm/p5-figure16.png)
>
> **What it says:** a translation test with "LLaMA-13B" on "English-to-German translation". Every prompt starts with the same prefix: an instruction and a "single example" (80 tokens) or "5 examples" (341 tokens). vLLM reaches "1.67× higher throughput" than Orca (Oracle) with the one-example prefix and "3.58× higher throughput" with five.
>
> **What it means:**
>
> Many services put the same long text in front of every user's input: instructions, rules, a few worked examples. Part 4 showed that vLLM can compute the cache of that prefix once and map its blocks into every request. Orca must store a fresh copy in each request's chunk.
>
> The longer the shared part, the bigger the win. With 80 shared tokens the gain is 1.67×. With 341 shared tokens it is 3.58×, the largest gain over Orca (Oracle) in the paper. Here vLLM saves not only memory but also compute, because the prefix is never processed again.

> [!PAPER] · §6.5 · page 12
> [![The chatbot paragraph, with concatenating the chatting history, last 1024 tokens, at most 1024 tokens, We do not store the KV cache between, 2x higher request, 1024 tokens for the request outputs and behave similarly highlighted](/img/papers/vllm/p5-chatbot.png)](/img/papers/vllm/p5-chatbot.png)
> [![The end of the paragraph, with resolves the problem of memory fragmentation and reservation highlighted](/img/papers/vllm/p5-chatbot-2.png)](/img/papers/vllm/p5-chatbot-2.png)
> [![Figure 17 of the paper: normalized latency against request rate on the chatbot workload; the three Orca versions bend together near 0.5 requests per second and vLLM near 0.85](/img/papers/vllm/p5-figure17.png)](/img/papers/vllm/p5-figure17.png)
>
> **What it says:** for a chatbot, the prompt is the chat history plus the new question, cut to the "last 1024 tokens", and the model writes "at most 1024 tokens". The KV cache is not kept between rounds of a conversation. vLLM sustains "2× higher request" rates than all three Orca versions, which "behave similarly", because the buddy allocator makes them reserve "1024 tokens for the request outputs" whatever they predict.
>
> **What it means:**
>
> Long chats fill the prompt to its 1,024-token limit, and any answer at all pushes the total past 1,024. The next power of two is 2,048. So Max, Pow2 and Oracle all get the same 2,048-slot chunk, and their curves fall on top of each other.
>
> This is the buddy allocator's rounding at its worst. It shows why even a perfect length prediction cannot save a contiguous design. The blocks of vLLM grow 16 tokens at a time, so the same answer costs only the blocks it uses.

## The cost of the indirection {§7.1}

An **ablation** test removes or changes one part of a system to see what that part contributes. Section 7 asks what vLLM's design costs and which settings are right.

> [!DEFINITION] Kernel
> One program that runs on the GPU, on thousands of threads at once. The attention of one layer is one kernel. The block table adds a step to that kernel: before it reads a block, it looks up where the block lives.

> [!PAPER] · §7.1 · page 12
> [![The kernel microbenchmark paragraph, with accessing the block table, handling variable sequence lengths, 20 to 26% higher attention kernel latency and only affects the attention operator highlighted](/img/papers/vllm/p5-kernel.png)](/img/papers/vllm/p5-kernel.png)
> [![Figure 18a of the paper: attention kernel latency in microseconds against context length for vLLM and FasterTransformer at batch sizes 8 and 32; vLLM's lines sit a little above](/img/papers/vllm/p5-figure18a.png)](/img/papers/vllm/p5-figure18a.png)
>
> **What it says:** PagedAttention's kernel has extra work: "accessing the block table", extra branches and "handling variable sequence lengths". This gives "20–26% higher attention kernel latency" than FasterTransformer's kernel. The authors call this small, because it "only affects the attention operator", not the other layers.
>
> **What it means:**
>
> Nothing is free. A contiguous kernel computes one address and reads straight through. The paged kernel must first look up each block in the block table. Figure 18a measures the cost: the vLLM lines sit a little above the FasterTransformer lines at every context length.
>
> The authors' argument is about proportion. The large matrix products of the other layers are not touched, so a slower attention costs only part of a step. In return, the batch is two or more times larger.

## Block size {§7.2}

> [!PAPER] · §7.2 · page 12
> [![The block size paragraph, with too small, fully utilize the GPU's parallelism, too large, internal fragmentation increases, block sizes from 16 to 128, sequences become shorter than the block sizes, block size 16 is large enough and default block size as 16 highlighted](/img/papers/vllm/p5-blocksize.png)](/img/papers/vllm/p5-blocksize.png)
> [![Figure 18b of the paper: normalized latency against block size from 1 to 256 for ShareGPT and Alpaca; both are high at sizes 1 to 4, low at 16 and 32, and Alpaca rises steeply above 64](/img/papers/vllm/p5-figure18b.png)](/img/papers/vllm/p5-figure18b.png)
>
> **What it says:** if the block size is "too small", vLLM may not "fully utilize the GPU's parallelism". If it is "too large", internal fragmentation increases and sharing becomes less likely. On ShareGPT, "block sizes from 16 to 128" work best. On Alpaca, larger blocks hurt because the "sequences become shorter than the block sizes". The authors find "block size 16 is large enough" and set the "default block size as 16".
>
> **What it means:**
>
> Block size is a tug of war between two costs. Small blocks waste no memory, but the GPU reads them in tiny pieces, and many tiny reads are slow. Large blocks are read efficiently, but the last block of each request is mostly empty. An Alpaca request of 80 tokens in a 256-token block wastes more than two thirds of it.
>
> Figure 18b shows both sides. At block sizes 1 to 4, latency is high on both datasets: that is the slow-read side. Above 64, Alpaca's latency climbs steeply: that is the fragmentation side. In between, at 16 and 32, both datasets do well. The paper picks 16.

Our simulator can show only one side of this tug of war. It counts memory exactly, but its step time does not depend on the block size, so tiny blocks cost nothing in it. We run vLLM with block sizes 1 to 256, at the rate where 16-token blocks reach our 0.1 s/token line:

```text
== 7. Figure 18b rebuilt: vLLM in the simulator with block sizes 1 to 256 (memory only, no kernel cost)
ShareGPT at 3.0 req/s: 1: 0.072, 2: 0.073, 4: 0.072, 8: 0.074, 16: 0.076, 32: 0.080, 64: 0.085, 128: 0.107, 256: 0.150
Alpaca at 102.3 req/s: 1: 0.084, 2: 0.085, 4: 0.086, 8: 0.092, 16: 0.100, 32: 0.114, 64: 0.148, 128: 0.221, 256: 0.392
```

{{FIG:p5_fig18b|Our simulator with block sizes 1 to 256. Without a kernel cost, small blocks are free, so only the right side of the paper's Figure 18b appears: large blocks hurt, and they hurt short Alpaca requests far more.}}

The right side of the paper's figure appears clearly. From 16 to 256, Alpaca's latency grows almost four times, from 0.100 to 0.392 s/token. ShareGPT's grows by about half, from 0.076 to 0.150. Short requests suffer more from large blocks, as the paper says.

The left side does not appear, and that is the lesson. In our simulator, block size 1 is the best. The paper's high latency at block sizes 1 to 4 comes from the GPU kernel, which we did not model. So memory alone pushes toward small blocks, and the GPU kernel sets the lower limit.

## Recompute or swap {§7.3}

> [!NOTE] Where this idea comes from: swapping
> Operating systems have done this since the early 1960s. The Atlas computer at Manchester kept the pages that a program was not using on a magnetic drum, and brought them back on demand. Today the drum is a disk, and the move is called swapping. The vLLM engine does the same with GPU memory and CPU memory: CPU memory plays the disk.

> [!PAPER] · Figure 19 · page 13
> [![Figure 19 of the paper: (a) time in milliseconds of recompute, swap in, swap out and swap in plus out against block size; (b) normalized latency of recompute and swap against block size for OPT-13B on ShareGPT](/img/papers/vllm/p5-figure19.png)](/img/papers/vllm/p5-figure19.png)
> [![The §7.3 paragraph, with excessive overhead with small block sizes, limits the effective PCIe bandwidth, remains constant across, never higher than 20% of swapping's latency and comparable end-to-end highlighted](/img/papers/vllm/p5-swap.png)](/img/papers/vllm/p5-swap.png)
>
> **What it says:** swapping has "excessive overhead with small block sizes", because many small copies between CPU and GPU limit "the effective PCIe bandwidth". The cost of recomputation "remains constant across" block sizes. Swapping wins at large block sizes, but recomputation's cost is "never higher than 20% of swapping's latency". From 16 to 64, the two give "comparable end-to-end" performance.
>
> **What it means:**
>
> Part 4 described the two ways to bring back a preempted request. Swapping copies its blocks to CPU memory and back. Recomputing throws the blocks away and runs the prompt phase again over the prompt and the tokens written so far.
>
> Each copy between CPU and GPU has a fixed start-up cost, whatever its size. With 1-token blocks, a request's cache becomes thousands of tiny copies, and the start-up costs dominate. Recomputation does not care about blocks at all: it is one prompt pass, one big matrix product of the kind GPUs do well.

> [!DEFINITION] PCIe bandwidth
> PCIe is the link between the CPU's memory and the GPU. Its bandwidth is how many bytes per second it moves. A big copy reaches the full bandwidth; a tiny copy spends most of its time starting up, so its effective bandwidth is low.

We time both on the laptop. We take the real KV cache of OPT-125M after a 1,024-token prompt. We copy it to CPU memory and back, one block of each layer at a time, for block sizes 1 to 256. Then we time one prompt pass that rebuilds the same cache:

```python
for b in BS:
    def out():
        for g, c in zip(gpu, cpu):
            for j in range(0, CTX, b):
                c[:, :, j:j + b].copy_(g[:, :, j:j + b])

    def back():
        for g, c in zip(gpu, cpu):
            for j in range(0, CTX, b):
                g[:, :, j:j + b].copy_(c[:, :, j:j + b])
    t_out, t_in = fastest(out, 3), fastest(back, 3)
```

```text
== 8. Figure 19a rebuilt: move a 1,024-token KV cache (OPT-125M, FP16) GPU -> CPU -> GPU, block by block,
       against recomputing it with one prompt pass (laptop, fastest of 3 runs)
KV cache: 37.7 MB in 24 tensors; recompute (one prompt pass): 21.1 ms
block  copies swap out ms swap in ms
    1  24,576      7350.2     5298.7
    2  12,288      2826.4     2787.9
    4   6,144      1278.6     1217.5
    8   3,072       716.2      577.4
   16   1,536       351.9      325.7
   32     768       131.6      165.3
   64     384       126.7      151.3
  128     192        56.5       56.4
  256      96        29.2       29.4
swap in + out against recompute: 599x slower at block size 1, 2.8x at block size 256; never faster on this laptop
```

{{FIG:p5_fig19|Our laptop version of Figure 19a, on a log scale. Swapping gets cheaper as blocks grow, because there are fewer copies. Recomputing does not depend on the block size at all.}}

The shape is the paper's. With 1-token blocks, the cache becomes 24,576 copies: 7,350 ms to move it out and 5,299 ms to move it back. With 256-token blocks it is 96 copies, 29.2 ms out and 29.4 ms back. The start-up cost of each copy dominates. Recomputation takes 21.1 ms at every block size.

The ending differs. On the A100, swapping wins at large blocks. On our laptop, recomputation wins at every size, by 2.8 times even at 256. An Apple laptop has no PCIe link, each small copy is a separate GPU command, and a 125M model recomputes 1,024 tokens very quickly. So the winner depends on the hardware, which is why vLLM supports both.

## What the idea is good for {§8}

> [!PAPER] · §8 · page 13
> [![The discussion paragraph on other GPU workloads, with dynamic memory allocation, bound by the GPU memory capacity, tensor shapes are typically static, primarily compute-bound and may rather degrade the performance highlighted](/img/papers/vllm/p5-disc-other.png)](/img/papers/vllm/p5-disc-other.png)
>
> **What it says:** paging works for LLM serving because the workload needs "dynamic memory allocation" and its performance is "bound by the GPU memory capacity". That is not true of every GPU workload. In training, "tensor shapes are typically static". Serving smaller models is "primarily compute-bound". There, vLLM's techniques "may rather degrade the performance".
>
> **What it means:**
>
> This paragraph says when not to use the idea. Paging pays off only when two things are true at once. Memory needs must change at run time in ways nobody can predict, and memory, not arithmetic, must limit the speed. LLM serving has both.
>
> Training has neither: every batch has the same shapes, so a framework can plan all memory before the first step. Smaller models on a server are limited by arithmetic. There, the block table is just an extra lookup, the 20% to 26% of §7.1 with nothing to buy.

> [!PAPER] · §8 · page 13
> [![The start of the paragraph on LLM-specific optimizations, with application-specific semantics and all-or-nothing highlighted](/img/papers/vllm/p5-disc-llm.png)](/img/papers/vllm/p5-disc-llm.png)
> [![The rest of the paragraph, with swap-out policy, recomputation method to recover the evicted blocks, not feasible in OS and fusing the GPU kernels highlighted](/img/papers/vllm/p5-disc-llm-2.png)](/img/papers/vllm/p5-disc-llm-2.png)
>
> **What it says:** vLLM adapts paging using "application-specific semantics". Its "all-or-nothing" "swap-out policy" uses the fact that a request needs all its tokens in GPU memory. The "recomputation method to recover the evicted blocks" is "not feasible in OS". And vLLM hides the cost of the indirection by "fusing the GPU kernels" for memory access with attention.
>
> **What it means:**
>
> vLLM did not copy an operating system; it borrowed the idea and bent it to fit. An operating system knows little about the programs it serves. It evicts single pages and guesses which ones will be needed next. The vLLM scheduler knows that a request touches every one of its blocks at every step. Evicting half a request would be pointless, so it evicts all of it or none.
>
> Recomputation is the clearest difference. An operating system cannot rebuild a lost page of a program's memory. The vLLM engine can, because the KV cache is a function of the tokens, and the tokens are kept.

## Related work {§9}

> [!PAPER] · §9 · page 13
> [![The comparison to Orca paragraph, with complementary, scheduling and interleaving the requests, increasing memory utilization, working sets of more requests fit into memory, 2-4x and speedup compared to Orca highlighted](/img/papers/vllm/p5-orca-compare.png)](/img/papers/vllm/p5-orca-compare.png)
>
> **What it says:** Orca's iteration-level scheduling and vLLM's PagedAttention are "complementary". Orca raises GPU use by "scheduling and interleaving the requests". vLLM raises it by "increasing memory utilization" so that the "working sets of more requests fit into memory". The result is a "2-4×" "speedup compared to Orca".
>
> **What it means:**
>
> The two ideas solve two different problems, and vLLM uses both. Iteration-level scheduling makes sure no slot in the batch sits idle. Paging makes sure more requests fit in memory in the first place. The paper adds a sharp point: fine-grained scheduling makes memory harder to manage, because requests start and stop at every step. So the better the scheduler, the more paging matters.
>
> The rest of §9 places vLLM among general serving systems and memory techniques. Its claim is a new idea: block-level memory management for online serving, where nobody knows who arrives next or how long they will talk.

## The conclusion {§10}

> [!PAPER] · §10 · page 14
> [![The conclusion, with non-contiguous paged memory, virtual memory and copy-on-write, and 2-4x throughput improvements highlighted](/img/papers/vllm/p5-conclusion.png)](/img/papers/vllm/p5-conclusion.png)
>
> **What it says:** PagedAttention lets keys and values be stored in "non-contiguous paged memory". vLLM shows how "virtual memory and copy-on-write" from operating systems can manage the KV cache. The experiments show "2-4× throughput improvements" over the best earlier systems.
>
> **What it means:**
>
> The whole paper fits in three steps. The KV cache is large, it grows, and nobody knows how far. Contiguous storage wastes most of it, so batches stay small and the GPU sits idle. Blocks on demand waste almost nothing, so batches grow and the same GPU serves two to four times more users.

## The limits

Some limits are the paper's. Orca was not public, so the authors rebuilt it with an assumed buddy allocator. The largest numbers, 8× and 22×, compare against Orca (Max) and FasterTransformer; against the best possible Orca the ShareGPT gain is 1.7× to 2.7×. Arrival times are random, while real traffic comes in bursts. And the gain shrinks when memory is not the limit, as Figure 12 (f) and §8 say.

Some limits are ours. The simulator is a model, not vLLM. Its clock comes from OPT-125M on a busy laptop, with a fit that is off by up to 29%. It also leaves out some limits of a real server. On our laptop a larger batch makes each step much slower than on an A100, so our speed gains are smaller than the paper's. What our runs show reliably is the memory side: how many requests fit, and why.

> [!TAKEAWAYS] Key takeaways
> - **Table 1** is a division: KV cache memory over the cache of one token. OPT-13B on one A100 gets 15.7K slots, enough for only 7 requests that book 2,048 tokens.
> - **Normalized latency** is time in the system divided by answer length. Its curve stays flat, then bends up where the queue starts to grow without end. The bend is the capacity.
> - **"2 to 4 times"** compares the request rates at which two systems reach the same latency. On ShareGPT, vLLM sustains 1.7× to 2.7× the rate of Orca (Oracle); our laptop simulation gives 1.4×.
> - **The cause is batch size.** On ShareGPT, OPT-13B runs 30.42 requests at once with vLLM against 13.62 with Orca (Oracle). Our simulator gives 37.54 against 14.91, and exactly 7.00 for Orca (Max).
> - **Sharing helps most with long prompts and beam search:** up to 66.3% memory saved in the paper. Our block manager and closed form agree exactly.
> - **The paged kernel is 20% to 26% slower in attention**, a price the larger batches repay.
> - **Block size 16** balances slow tiny reads against wasted large blocks; our memory-only simulator shows only the waste side.
> - **Swap or recompute** depends on the hardware. On our laptop, recomputing beat swapping at every block size.
> - **Paging is not for every workload.** It helps when memory needs change unpredictably and memory limits the speed.

**Next, in Part 6:** build it from scratch. A small paged KV cache and serving loop in Python, checked against a library model.

<details>
<summary>Run it yourself</summary>

The script behind this part is [`code/papers/vllm/vllm_part5.py`](https://github.com/ishwar6/ishwar-books/blob/main/code/papers/vllm/vllm_part5.py). It checks Table 1 and counts the lengths of ShareGPT and Alpaca requests. It times OPT-125M steps on the laptop GPU and fits a step-time model. Then it runs the serving simulator for five systems and checks its batch sizes with a closed form.

Last, it counts the blocks of parallel sampling two ways, sweeps the block size, and times swapping against recomputing. Every number is saved in `results/part5.json`. On an Apple M-series laptop it takes a few minutes after the downloads. The timings change from run to run.

```bash
pip install torch transformers datasets numpy
python vllm_part5.py   # downloads facebook/opt-125m, two OPT configs, tatsu-lab/alpaca and the ShareGPT file (about 670 MB)
```

<figure><img src="/img/papers/vllm/part5-run.png" alt="Terminal output of vllm_part5.py: Table 1, the dataset lengths, the step-time model, the latency table of five systems, the batch sizes, the sharing savings, the block sizes and the swap timings" loading="lazy" /><figcaption>The first 60 lines of the output of vllm_part5.py, as it ran for this article. The full log is results/part5_stdout.txt.</figcaption></figure>

</details>

## The paper and the code

[vLLM on arXiv](https://arxiv.org/abs/2309.06180) · [vLLM on GitHub](https://github.com/vllm-project/vllm) · [our scripts for this part](https://github.com/ishwar6/ishwar-books/tree/main/code/papers/vllm)
