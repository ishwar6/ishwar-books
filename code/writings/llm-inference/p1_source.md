---
title: "How an LLM Writes: Prefill and Decode"
description: "Why the first word takes a moment and the rest stream steadily. Prefill, decode, the GPU's two speeds, the roofline and batching, measured on a real model: reading 512 tokens takes 27 ms, writing 512 takes 5.2 s."
date: 2026-09-28
tags: [inference, llm, vllm]
series: "LLM Inference from the Ground Up"
series_part: 1
motif: timeline
accent: "#7c9cff"
---

You type a question into a chatbot and press enter. There is a short pause. Then the first word appears, and the rest of the answer streams out at a steady pace, word after word.

The pause and the stream are two different jobs. They use the same model, but they hit opposite limits inside the computer. Once you see why, most of what LLM serving systems do, from vLLM's design to the size of a GPU bill, starts to make sense.

> [!TIP] The whole idea in one minute
> An LLM writes one small piece of text per step, in a loop. It reads your whole question in one fast step, then writes the answer one piece per step. Every step must move the entire model from the GPU's memory to its calculators, and that move, not the arithmetic, takes most of the time. So the most useful trick is to write for many conversations in the same step: one move of the model then serves all of them. This page builds each of those ideas from zero, with measurements.

Every number here comes from a real model, Qwen2.5-0.5B, on the GPU of an Apple M5 Pro (64 GB) with PyTorch. It is a small model, so the absolute times are small too. The shapes are the same on a data-centre GPU, and where it matters I show the numbers for an NVIDIA H100.

## How a model writes one word

A language model does not see words. It sees **tokens**.

> [!DEFINITION] Token
> A small piece of text that the model treats as one unit: often a whole word, sometimes part of a word or a punctuation mark. The model has a fixed list of all the tokens it knows, called its vocabulary.

Here is how our model's tokenizer cuts two sentences. Common words are one token each. A long or rare word is cut into pieces the model has seen often.

{{FIG:tokens|Two sentences cut into tokens by Qwen2.5-0.5B's tokenizer. "The cat sat on the" is five tokens; "Tokenization" and "unbelievably" are each cut into two pieces.}}

The model's job is narrow. You give it a list of tokens. It runs them through a stack of layers (24 in our model). Out comes one thing: a score for every token in its vocabulary, saying how likely it is to come **next**. You pick one of the likely tokens, and you have the next piece of text.

{{FIG:loop|The generation loop. Everything so far goes in, one new token comes out, it is added to the end, and the whole model runs again.}}

To write a whole answer, you repeat that. Add the new token to the end, run the model again, and get the next one. A 500-token answer needs 500 runs of the model, one after another. Each run needs the token from the run before it, so the runs cannot happen at the same time.

That loop is the whole of LLM inference. Every trick in this series makes the loop faster or makes one GPU run more loops at once.

> [!DEFINITION] Inference
> Using a trained model to produce output, as opposed to training it. For a language model, inference is the generation loop above.

## Two phases: reading and writing

The clearest way to see the two phases is to follow one request from start to finish. Say you ask:

> What is the capital of France?

The tokenizer cuts that question into 7 tokens: `What`, `is`, `the`, `capital`, `of`, `France`, `?`. Here is what the model does with them, step by step.

{{FIG:request_steps|One request, step by step. Step 0 takes in all seven prompt tokens together and produces the first word. Every later step takes in one word and produces the next.}}

**Step 0 is special.** The whole question already exists, so nothing about it is unknown. The model can take in **all 7 tokens at the same time**, the way you read a short sentence at a glance. Inside the GPU, the 7 tokens travel through every layer side by side, as one block of numbers. Each layer handles all 7 in one operation.

At the end of step 0 the model has done two things. It has picked the first word of the answer, `The`. And while it read, it wrote down **notes** about each of the 7 tokens, so that it never has to read them again. Those notes matter a lot for memory, and they come back at the end of this page.

**Every step after that is different.** To produce the second word, the model needs the first. To produce the third, it needs the second. So each step takes in exactly **one** new token, the one it just wrote, and produces exactly one more.

| Step | Goes in | Comes out | Phase |
|---|---|---|---|
| 0 | `What is the capital of France?` (7 tokens, together) | `The` | prefill |
| 1 | `The` | `capital` | decode |
| 2 | `capital` | `is` | decode |
| 3 | `is` | `Paris` | decode |
| 4 | `Paris` | `.` | decode |

> [!DEFINITION] Prefill
> The first step of a request. The model reads the whole prompt in one pass, writes its notes on every prompt token, and produces the first token of the answer.

> [!DEFINITION] Decode
> Every later step: the model takes in the one token it wrote last and produces one more. A 500-token answer is 500 decode steps.

### Why can't the answer be written all at once too?

Because the answer does not exist yet. The prompt is complete the moment you press enter, so all of it can be handled together. The answer is invented one token at a time, and the model cannot know word 3 before it has chosen word 2. Reading a finished page is fast because you can take it in at a glance. Writing a sentence is slower because each word waits for the one before it.

### What the two phases look like in time

Here are the two phases drawn to scale, with real timings for a 512-token prompt.

{{FIG:timeline|The two phases drawn to scale from real measurements: one wide prefill step, then a long run of narrow decode steps, one for each new token.}}

The two phases match the two numbers that every serving dashboard shows, and the two things you feel as a user.

> [!DEFINITION] TTFT (time to first token)
> The time from sending a request to seeing the first token of the answer. It is mostly the prefill step, plus any time the request waits in a queue.

> [!DEFINITION] TPOT (time per output token)
> The average time between two tokens of the answer after the first one, also called inter-token latency. It is the length of one decode step.

| Metric | What you feel | Set by |
|---|---|---|
| TTFT | the pause before anything appears | prefill (step 0) |
| TPOT | how fast the words stream after that | decode (every later step) |

In the France example, TTFT is the length of step 0. TPOT is the average length of steps 1 to 4. In the 512-token run, TTFT is about 27 ms and TPOT is about 10 ms.

## The measurement that surprises everyone

Here is a simple experiment. Take a 512-token prompt and measure how long the model takes to **read** it (prefill). Then measure how long it takes to **write** 512 new tokens (decode). It is the same model and the same number of tokens.

{{FIG:read_vs_write|Reading 512 tokens versus writing 512 tokens, measured on Qwen2.5-0.5B. The prefill bar is so short it is barely visible.}}

Reading took **27 milliseconds**. Writing took **5.2 seconds**. That is about 190 times longer for the same number of tokens.

Nothing is broken. Every LLM on every GPU behaves like this. To see why, we need to look at what a GPU is made of, at the level that matters here.

## What a GPU is made of

First, what is the model, physically? It is a very large set of numbers called **weights**.

> [!DEFINITION] Weights (parameters)
> The numbers a model learned during training. Running the model means multiplying the input by these numbers, layer by layer. Our model has 494 million of them.

Each weight is stored in 16-bit format, which takes 2 bytes. So our small model is 494 million × 2 bytes = **0.99 GB** of numbers. Llama 3.1 8B is about 16 GB. The largest models are hundreds of gigabytes.

For this page, a GPU has two parts that matter: a place that **stores** numbers and a place that **calculates** with them.

> [!DEFINITION] HBM (GPU memory)
> High-bandwidth memory: the GPU's main memory, built from memory chips beside the processor. It holds the weights and everything else the model needs. An H100 has 80 GB of it.

> [!DEFINITION] Compute cores
> The parts of the GPU that do arithmetic. There are thousands of them, and they work in parallel. They can only calculate with numbers that have already arrived from memory.

Between the two runs a link with a fixed speed. So a GPU has two speed limits, and each one has its own unit.

> [!DEFINITION] FLOP and FLOP/s
> One floating-point operation: one multiplication or one addition on decimal numbers. FLOP/s counts how many of them the compute cores can do per second. An H100 can do about 989 trillion 16-bit FLOP/s (989 TFLOP/s).

> [!DEFINITION] Memory bandwidth
> How many bytes per second can move from memory to the compute cores. An H100 moves about 3.35 trillion bytes per second (3.35 TB/s). I measured our M5 Pro at 262 GB/s.

{{FIG:gpu_parts|A GPU reduced to the two parts that matter here: memory that holds the weights, compute cores that do the arithmetic, and a link between them with a fixed bandwidth.}}

Now compare the two speeds of the H100. In one second it can do 989 trillion operations, but it can move only 3.35 trillion bytes. Divide one by the other: in the time it takes to bring in **one byte**, the cores could do about **295 operations**. The calculators are far faster than the delivery.

### A kitchen with a narrow door

Picture a chef who cooks extremely fast. The chef works at a small counter with room for only a few jars. All the ingredients sit in a big pantry down the hall, behind a narrow door. To cook, the chef must bring jars through the door, use them, and send them back to make room for the next ones.

{{FIG:kitchen|The kitchen picture. A fast chef works at a small counter. The ingredient jars wait in a pantry, and everything must pass through a narrow door.}}

Now the strange rule of this kitchen. Each round of cooking needs **every jar in the pantry**, once, in order.

With one dish per round, every jar comes through the door for a single dish. The chef waits at the door most of the time. If the chef cooks 512 dishes per round, each jar is used 512 times before it goes back. Now the chef is the busy one.

The mapping is exact. The pantry is HBM, and the jars are the weights. The door is the memory bandwidth, and the chef's speed is FLOP/s. The small counter is the small, fast memory beside the compute cores, which is far too small to keep a whole model. That is why every step must bring the weights in again.

## Why decode waits for memory

A layer of the model is mostly a **matrix multiplication**: a grid of weights times the numbers that describe each token.

> [!DEFINITION] Matrix multiplication
> Multiplying a grid of numbers (a matrix) by a list of numbers (a vector) or by another grid. Each output number is a sum of products: each weight is multiplied by an input number and added to a running total. This "multiply-add" is 2 FLOPs.

A tiny example shows the whole effect. Take a 3 × 3 weight matrix, which is 9 weights. At 2 bytes each, reading it costs 18 bytes. Multiply it by one token's 3 numbers: each weight gets one multiply-add, so the work is 9 × 2 = 18 FLOPs. That is **1 FLOP for every byte** read.

Now multiply the same matrix by four tokens at once. You still read the 9 weights once (18 bytes), but each weight now does four multiply-adds: 72 FLOPs, or **4 FLOPs per byte**.

{{FIG:matmul|A 3 × 3 weight matrix times one token, and the same matrix times four tokens. The weights are read once in both cases; four tokens get four times the work out of the same read.}}

Decode is panel A, at a huge scale. It multiplies each weight matrix by **one** token, which engineers call a matrix-vector multiplication. Prefill is panel B with 512 columns instead of 4: a matrix-matrix multiplication. Both read every weight once per pass. Prefill gets 512 tokens of work out of that read. Decode gets one.

{{FIG:weights_trip|Both phases pay for the same trip through memory. Prefill gets 512 tokens of work out of it. Decode gets one.}}

We can check this with our measurements. Our GPU moves 262 GB/s, and the weights are 0.99 GB. So reading the weights once takes at least 0.99 ÷ 262 = **3.8 ms**. That is a floor: no decode step can be faster than one read of the model. The arithmetic for one token is about 0.99 billion FLOPs, which the GPU's cores could finish in a small fraction of that time.

{{FIG:step_budget|Where the 10 ms of one decode step go at batch 1: at least 3.8 ms reading weights, most of the rest fixed overhead, and almost no time on arithmetic.}}

We measured a decode step at **10 ms**. The weight read is over a third of it. Most of the rest is a fixed cost per step. A small model like this one runs as hundreds of small GPU jobs, called kernels, one after another. Each one has a start-up cost, and the cores sit idle between them.

> [!DEFINITION] Kernel
> One function that runs on the GPU, such as one matrix multiplication or one normalisation. A forward pass of the model is a long sequence of kernels.

Neither part of the 10 ms is arithmetic. That is the central fact of decode.

> [!DEFINITION] Memory-bound and compute-bound
> A job is memory-bound when its speed is set by how fast data arrives from memory, so the compute cores wait. It is compute-bound when its speed is set by how fast the cores can calculate, so memory waits.

> [!NOTE] The one idea
> **Decode is limited by how fast the GPU can move the weights, not by how fast it can multiply.** Decode is memory-bound. Prefill, which uses each weight for many tokens, is compute-bound.

The vLLM paper puts the same fact at the centre of its introduction.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §1 · page 1
> [![The paragraph about autoregressive generation, with one at a time, repeated until the model outputs a termination token, and memory-bound, underutilizing the computation power of GPUs highlighted](/img/papers/vllm/p1-intro-autoregressive.png)](/img/papers/vllm/p1-intro-autoregressive.png)
>
> **What it says:** the model generates tokens "one at a time", and this is "repeated until the model outputs a termination token". The result is a workload that is "memory-bound, underutilizing the computation power of GPUs".
>
> **What it means:**
>
> "Autoregressive" is the generation loop: each token is produced from all the tokens before it. A termination token is a special token that means "the answer is finished", so the loop stops there. The last sentence is the chef at the door. The GPU's calculators sit mostly idle because the loop feeds them one token at a time.
>
> This is the problem the whole vLLM paper sets out to fix. It cannot make one decode step do more arithmetic per byte for a single user. It can make each step serve many users, which is where this page ends.

Two lines on page 3 of the same paper name the two kinds of multiplication from the 3 × 3 example.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §2.2 · page 3
> [![The paragraph on the prompt phase, with The prompt phase and parallelized using matrix-matrix multiplication operations highlighted](/img/papers/vllm/p2-prompt-phase.png)](/img/papers/vllm/p2-prompt-phase.png)
>
> **What it says:** in "the prompt phase", the work "can be parallelized using matrix-matrix multiplication operations", so it "can efficiently use the parallelism inherent in GPUs".
>
> **What it means:**
>
> The paper's "prompt phase" is our prefill. The symbols $$x_1, \ldots, x_n$$ are the $$n$$ prompt tokens, and $$P(x_{n+1} \mid x_1, \ldots, x_n)$$ means "the chance of each possible next token, given the prompt". The key vectors $$k$$ and value vectors $$v$$ are the notes the model writes on each prompt token.
>
> The point of the highlight is panel B of the 3 × 3 example. All prompt tokens are known, so they form a matrix with one column per token, and each weight is used once for every column.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §2.2 · page 3
> [![The paragraph on the autoregressive generation phase, with only the new key and value vector, maximum length, end-of-sequence token, matrix-vector multiplication and becomes memory-bound highlighted](/img/papers/vllm/p2-gen-phase.png)](/img/papers/vllm/p2-gen-phase.png)
>
> **What it says:** the generation phase "often uses matrix-vector multiplication, which is less efficient", so it "becomes memory-bound" and is responsible for most of the time a single request takes.
>
> **What it means:**
>
> This is decode, panel A. At each step $$t$$ the model takes in one token and computes the notes for that one token only. The older notes were saved in earlier steps. The step multiplies every weight matrix by a single vector, so each weight gives only one multiply-add for the bytes it cost to read.
>
> The paper also says when decode stops: at a maximum length, or when the model writes the end-of-sequence token. Until then, every step reads the whole model again. That is why decode, and not prefill, fills most of the 5.2 seconds in our measurement.

## From a Python line to the GPU: CUDA and kernels

The 10 ms step had a large part that was neither reading weights nor arithmetic: the fixed cost of starting hundreds of kernels. To see where that cost comes from, follow one line of Python all the way into the GPU.

On NVIDIA GPUs, the software layer between your program and the chip is called **CUDA**.

> [!DEFINITION] CUDA
> NVIDIA's platform for running programs on its GPUs. It has a language based on C++, a driver, and libraries of ready-made kernels, such as matrix multiplication. PyTorch calls these libraries for you.

### An analogy: a foreman and a crew

Think of a building site. A foreman (the CPU) reads the plan, one line at a time. For each line, the foreman shouts one order to a huge crew (the GPU), such as "lay all these bricks". Then the crew splits into teams, and the teams work at the same time. The foreman does no building at all. The foreman only reads the plan and gives orders.

Two facts follow from this picture. Each order takes the foreman a moment to shout, however small the job. And if every job is small, the crew finishes each one before the next order arrives, so the crew stands and waits.

### What happens for `y = x @ W`

Here is the real sequence for one matrix multiplication in PyTorch.

{{FIG:cuda_path|One line of PyTorch on an NVIDIA GPU. The CPU launches a kernel. Its blocks of threads are spread over the streaming multiprocessors (SMs). The threads read W and x from HBM through the L2 cache into their own fast on-chip memory, do the multiply-adds, and write y back to HBM.}}

1. **The CPU picks a kernel.** PyTorch sees a matrix multiplication of two GPU tensors. It chooses a matrix-multiply kernel from NVIDIA's libraries, sized for these shapes.
2. **The CPU launches it.** It sends the GPU the kernel's code, the memory addresses of `x`, `W` and `y`, and how many **threads** to start. Then the CPU moves on at once. It does not wait for the answer.
3. **The GPU spreads the work.** The threads come in groups called **thread blocks**. The GPU hands the blocks out to its **streaming multiprocessors**. An H100 has 132 of them. Each block computes one tile of `y`.
4. **Each block reads, computes and writes.** Its threads copy their part of `W` and `x` from HBM into fast memory on the chip. They do the multiply-adds there, and they write their tile of `y` back to HBM.

> [!DEFINITION] Thread, warp and thread block
> A **thread** is one small sequence of instructions that runs on the GPU, usually working on a few numbers. The GPU runs threads in groups of 32 that move in step, called **warps**. A **thread block** is a group of threads, often a few hundred, that run on the same SM and can share data.

> [!DEFINITION] Streaming multiprocessor (SM)
> One of the GPU's processor units. Each SM has its own arithmetic units, including Tensor Cores for matrix multiplication, its own registers and its own small, fast shared memory. An H100 SXM has 132 SMs. The "compute cores" of this page live inside the SMs.

The CUDA programming guide says it in two sentences. A kernel, "when called", is "executed N times in parallel by N different CUDA threads". And when the CPU launches it, "the blocks of the grid are enumerated and distributed to multiprocessors with available execution capacity".

### The memory ladder

Step 4 hides an important detail. A GPU has several kinds of memory, and they differ in size by a factor of thousands. The kitchen's "small counter" from earlier is really a ladder of memories.

{{FIG:memory_ladder|The memories of an H100 on a log scale. Registers and shared memory sit inside each SM and add up to tens of megabytes. The L2 cache is 50 MB. HBM, 80 GB, is the only level big enough to hold a model.}}

| Memory | Where it is | Size on an H100 SXM |
|---|---|---|
| registers | inside each SM | 256 KB per SM |
| shared memory and L1 cache | inside each SM | 256 KB per SM, together |
| L2 cache | on the chip, shared by all SMs | 50 MB |
| HBM | memory chips beside the GPU | 80 GB, at 3.35 TB/s |

> [!DEFINITION] SRAM and cache
> **SRAM** is the fast memory built into the chip itself, such as registers and shared memory. A **cache** is SRAM that keeps copies of recently used data from HBM, so a second read of the same data is faster.

The on-chip memories are much faster than HBM, but they are tiny. All 132 SMs together have about 34 MB of registers and about 34 MB of shared memory. Llama 3.1 8B needs 16.1 GB. So the weights cannot stay on the chip between steps. Every decode step must pull them from HBM again, at 3.35 TB/s, which is the floor this page measured.

### Launch overhead: why small steps are slow

Now back to the foreman. A launch takes the CPU a few microseconds. The work of a kernel at batch 1 is often smaller than that. Then the GPU finishes a kernel and waits for the next order.

> [!DEFINITION] Launch overhead
> The fixed time it takes to start one kernel: the CPU work in PyTorch and the driver, plus the GPU's start-up of the kernel. It is paid once per kernel, whatever the size of the work.

We counted the operations in one decode step of our model. A step at batch 1 runs **880 GPU operations**, about 37 per layer. Then we timed a tiny operation, an add on 16 numbers, repeated 2,000 times. Each one took **3.4 microseconds**, with almost no arithmetic inside.

So the launches alone cost about 880 × 3.4 µs = **3.0 ms** per step. Add the 3.8 ms floor for reading the weights, and about 6.8 ms of the 10 ms step is explained. The rest is time that these two numbers do not cover. Some kernels cost more than a tiny add to start, and each one does some work too.

{{FIG:launch_gaps|Two timelines of one decode step. Without a CUDA graph, the CPU launches each kernel in turn, and the GPU waits in the gaps. With a CUDA graph, one launch replays the whole recorded step, and the kernels run back to back.}}

This cost does not grow with the batch. A batch of 64 runs the same 880 operations, each on more numbers. That is part of the "fixed part" of the step that you will meet in the batching section below.

> [!NOTE] Measured on Apple's GPU, not on CUDA
> Our measurements ran on the GPU of an Apple M5 Pro, through Apple's Metal framework (PyTorch's `mps` device), not through CUDA. Apple's GPU has its own names: GPU cores instead of SMs, for example. But the ideas are the same on both. There is one launch per operation and a fixed cost per launch. There is small, fast memory on the chip and big, slow memory beside it. The scripts run on an NVIDIA GPU if you change `mps` to `cuda`.

### Two fixes: CUDA graphs and kernel fusion

**CUDA graphs** remove most of the launch cost. The engine runs one decode step once while CUDA records every kernel and its inputs. That record is a **graph**.

From then on, the CPU launches the whole graph with one call, and the GPU runs the kernels back to back. A graph works only for fixed shapes, so engines record one graph for each of many batch sizes. By default, vLLM records full CUDA graphs for decode steps. For prefill and mixed steps, it uses graphs for the parts between the attention calls.

> [!DEFINITION] CUDA graph
> A recording of a fixed sequence of kernels and their inputs that the CPU can launch again with one call. It removes the launch cost of every kernel after the first, but it only fits the shapes it was recorded with.

**Kernel fusion** removes trips to HBM. A layer has many small steps, such as a multiply, an add and a normalisation. Each one reads its input from HBM and writes its output back. A fused kernel does all of them in one pass. The numbers in between stay in registers and never go back to HBM. It also means fewer kernels, so fewer launches.

{{FIG:fusion|Kernel fusion. Three small kernels make three reads and three writes to HBM. One fused kernel does the same three steps with one read and one write, because the in-between numbers stay in registers.}}

> [!DEFINITION] Kernel fusion
> Joining several GPU operations into one kernel, so the data is read from memory once and the results in between stay on the chip.

Neither fix changes the floor. The weights must still come from HBM once per step. The fixes only cut the time on top of the floor. The rest of this page is about the floor itself, and about the one trick that spreads it over many users.

## Counting work per byte

The 3 × 3 example gives us a number to put on each job. Engineers call it **arithmetic intensity**.

> [!DEFINITION] Arithmetic intensity
> The number of FLOPs a job does for each byte it reads from memory. Low intensity means the job waits for memory. High intensity means it keeps the compute cores busy.

$$
\text{arithmetic intensity} = \frac{\text{FLOPs done}}{\text{bytes read from memory}}
$$

Read it left to right: count the arithmetic, count the bytes that had to come from memory, and divide. For a weight matrix the rule is short. Each weight costs 2 bytes and gives 2 FLOPs (one multiply-add) per token. So the intensity is about **1 FLOP per byte for each token** that uses the weights in the same pass.

Now a real matrix from our model. Each layer has matrices of 896 × 896 weights, which is 802,816 weights or 1.61 MB. One token through it is 1.61 million FLOPs: 1 FLOP per byte. Four tokens give 6.42 million FLOPs for the same 1.61 MB: 4 FLOPs per byte. A 512-token prefill gives 512 FLOPs per byte.

| Job | FLOPs per byte (weights) |
|---|---|
| decode, one sequence | about 1 |
| prefill, 512 tokens | about 512 |

Compare those with the H100's balance point of about 295, from the GPU section. Below 295 FLOPs per byte, memory cannot deliver bytes fast enough to keep the cores busy. Above it, memory is fast enough and the cores become the limit. Decode at 1 is far below the line. Prefill at 512 is above it.

## The roofline: one picture for both limits

A **roofline** turns those two limits into one picture. It answers a single question: for a job with a given work per byte, what is the most speed this GPU can give it?

> [!DEFINITION] Roofline
> A chart of the highest FLOP/s a processor can reach, plotted against arithmetic intensity. It has a sloped part set by memory bandwidth and a flat part set by peak compute.

Build it in three steps.

1. **The memory limit.** If memory delivers 3.35 trillion bytes per second, and each byte gives you $$I$$ FLOPs, you cannot do more than 3.35 × $$I$$ trillion FLOP/s. More work per byte means more speed, so this limit is a line that rises with $$I$$.
2. **The compute limit.** However many FLOPs per byte you have, the cores cannot go faster than 989 TFLOP/s. That is a flat line, the roof.
3. **The roofline.** The real limit is the lower of the two. It rises along the slope, then flattens at the roof. The bend, where the two lines meet, is the **ridge point**: 989 ÷ 3.35 ≈ 295 FLOPs per byte.

{{FIG:roofline_build|The roofline built in three steps for an H100: a sloped memory limit, a flat compute limit, and the lower of the two, which bends at the ridge point.}}

$$
\text{highest speed} = \min\left(\text{peak FLOP/s},\ \text{bandwidth} \times I\right)
$$

Read it left to right. The speed a job can reach is the smaller of two numbers: the roof, and the bandwidth times the work per byte. A worked example for decode at $$I = 1$$ on an H100: 3.35 TB/s × 1 = 3.35 TFLOP/s, and the roof is 989. The smaller one is 3.35 TFLOP/s, which is **0.3%** of what the GPU can compute. For prefill at $$I = 512$$: 3.35 × 512 = 1,715, which is above the roof, so the limit is the full 989 TFLOP/s.

{{FIG:roofline_phases|A roofline for an H100. On the left slope, speed is capped by memory. On the flat roof, it is capped by compute. Decode at batch 1 sits at the bottom left; prefill sits on the roof.}}

The roofline gives only a ceiling. Real jobs sit below it, because of the fixed costs per step that we saw above. Our M5 Pro shows the same pattern in what it really did:

| Job (measured on the M5 Pro) | FLOPs per byte | Real speed |
|---|---|---|
| decode, one sequence, 10.0 ms per step | 1 | 0.10 TFLOP/s |
| prefill, 512 tokens in 27.4 ms | about 371 | 13.4 TFLOP/s |

Prefill did over 100 times more arithmetic per second on the same chip. (The prefill intensity here is 371, not 512, because prefill only needs the large output layer for the last token. The script in the research notes counts both.) That gap is the 190x gap from the measurement above, seen as a picture.

<details>
<summary>Research notes: how the roofline numbers were counted</summary>

The FLOP counts use the usual rule of 2 FLOPs per weight per token. Our model has 494,032,768 weights. Of these, 136,134,656 are the token embedding, which the model also uses as its output layer. A decode step uses every weight once per token, so it does 0.988 GFLOP per token. Prefill keeps the output only for the last prompt token, so its other tokens skip the output layer: 0.716 GFLOP per token. The counts leave out attention, which is small at these lengths.

The H100 numbers come from NVIDIA's specification: 3.35 TB/s of memory bandwidth and 1,979 TFLOP/s of BF16 with sparsity. For the dense matrices that LLMs use, that is about 989 TFLOP/s. The M5 Pro's bandwidth, 262 GB/s, was measured for this series. The script `p1_roofline.py` prints every number in this section.

</details>

Prompt length changes the prefill side too. Here is prefill on our model for prompts from 16 to 8,192 tokens.

| Prompt length | Prefill time | Tokens per second |
|---|---|---|
| 16 | 10.8 ms | 1,484 |
| 256 | 17.7 ms | 14,483 |
| 1,024 | 47.9 ms | 21,371 |
| 2,048 | 95.0 ms | 21,551 |
| 8,192 | 450.5 ms | 18,185 |

Short prompts are slow per token. A 16-token prefill takes about as long as one decode step, because the fixed cost per step is larger than the work. As the prompt grows, the GPU gets enough work per weight to stay busy, and the rate climbs to about 21,500 tokens per second. At 8,192 tokens it dips a little.

Most of that dip comes from attention, the step where every token looks at every earlier token. That part of the work grows with the square of the length. Part 2 explains attention.

A paper on scheduling these two phases, Sarathi-Serve, states the same split in one sentence and draws it from its own measurements.

> [!PAPER] Agrawal et al. (2024), Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve · §1 · page 2
> [![The introduction's paragraph on the two phases, with The prefill phase is compute-bound, the decode phase is memory-bound and decodes benefit significantly from batching highlighted](/img/writings/llm-inference/p1-sarathi-phases.png)](/img/writings/llm-inference/p1-sarathi-phases.png)
>
> **What it says:** "The prefill phase is compute-bound" because it handles all prompt tokens in parallel, while "the decode phase is memory-bound" because it handles one token per request at a time. "Therefore, decodes benefit significantly from batching".
>
> **What it means:**
>
> This is the roofline in words. Prefill is already on the roof, so giving it more work in the same pass gains little. Decode sits low on the slope, so anything that adds work per byte moves it up the slope.
>
> The last highlight names that thing: batching. The rest of this page explains what it is, why it works, and where it stops.

> [!PAPER] Agrawal et al. (2024), Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve · Figure 3 · page 5
> [![Figure 3 of Sarathi-Serve: prefill and decode throughput against batch size for Mistral-7B on one A100, with prefills are much more efficient than decode and batching boosts decode throughput almost linearly highlighted](/img/writings/llm-inference/p1-sarathi-fig3.png)](/img/writings/llm-inference/p1-sarathi-fig3.png)
>
> **What it says:** for Mistral-7B on one A100 GPU, "prefills are much more efficient than decode", and "batching boosts decode throughput almost linearly but has a marginal effect on prefill throughput".
>
> **What it means:**
>
> Look at the two y-axes first. The prefill panel counts in thousands of tokens per second; the decode panel counts in hundreds. That is our 190x gap again, on a different model and a different GPU.
>
> Then look at the shapes. Doubling the decode batch roughly doubles decode's tokens per second. Adding prefills to a batch does almost nothing, because prefill is already on the roof. Our own measurements below show the same two shapes.

## The fix: write for many conversations at once

Decode wastes each trip to memory on a single token. The fix follows from the 3 × 3 example: put more tokens into each trip. A real server has many users at the same time, and each of them needs a next token. So instead of one decode step per user, run **one step for all of them together**.

> [!DEFINITION] Batch
> A group of sequences that the GPU processes in the same step. In decode, a batch of 64 means one forward pass of the model produces the next token for 64 different conversations.

> [!DEFINITION] Throughput
> How much useful output the whole system produces per second, here counted in tokens per second across all users.

### The bus

Think of a bus between two towns. The trip costs the same fuel and the same 10 minutes whether one passenger rides or forty. With one passenger, almost all of that cost is wasted. With forty, the same trip carries forty people.

{{FIG:bus|A bus trip costs about the same with one passenger or four. One read of the weights is the trip; each sequence that needs a next token is a passenger.}}

In decode, the trip is one read of all the weights from memory. The passengers are the sequences in the batch. The read costs about the same whether it serves one sequence or four, because, as we saw, the arithmetic is the cheap part.

### One decode step, batch 1 and batch 4

Here is what happens inside one step when four users share it. Each user is at a different point in a different conversation.

{{FIG:batch_step|One decode step for one sequence and for four. With four, the newest token of each sequence is stacked into a small matrix, every weight is read once, and four next tokens come out.}}

1. **Stack the newest tokens.** The step takes the latest token of each of the four sequences and puts their numbers side by side, one column per sequence. This is panel B of the 3 × 3 example.
2. **Read each weight once and use it four times.** Each weight matrix comes from memory once and is multiplied by all four columns. The arithmetic is four times larger, but it was the cheap part.
3. **Each sequence gets its own next token.** The output has one column per sequence. Each column becomes that sequence's next token, which is added to the end of its own text.

The sequences do not share any text. User A's answer about Paris never mixes with user B's answer about the sky. Each column is computed only from its own sequence. They share only the trip to memory.

Measured on our model, one step for one sequence takes **10.0 ms**. One step for four sequences takes **11.1 ms**. The GPU makes 4 tokens in 11.1 ms instead of 1 token in 10.0 ms. That is 361 tokens per second instead of 100, for 1.1 ms more per step.

The vLLM paper describes the same effect in its background section.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §2.3 · page 3
> [![The paragraph on batching, with amortized across the requests in a batch, arrive at different times, queueing delays, pad the inputs and wasting highlighted](/img/papers/vllm/p2-batching.png)](/img/papers/vllm/p2-batching.png)
>
> **What it says:** "Because the requests share the same model weights, the overhead of moving weights is amortized across the requests in a batch". Then two problems: requests "arrive at different times", and a simple batch would "pad the inputs" to the same length, "wasting" compute and memory.
>
> **What it means:**
>
> "Amortized" means the cost of one weight read is split across all requests in the batch. With 4 requests, each pays a quarter of the trip. With 64, each pays a sixty-fourth. This is the bus.
>
> The two problems are real, and this page leaves them for later. Users do not arrive together, and their answers have different lengths. A naive server would make early users wait for late ones, or fill short answers with empty padding. Part 3 shows how vLLM refills the batch at every step so that neither happens.

### The batch, measured from 1 to 128

Here is what batching does on our model, from 1 sequence to 128. Every sequence has a 512-token context.

{{FIG:batch_bars|Tokens per second as more sequences are decoded together. Hover a bar to see the time per step.}}

| Sequences at once | Time per step | Tokens per second (all users) | Tokens per second (each user) |
|---|---|---|---|
| 1 | 10.0 ms | 100 | 100 |
| 2 | 10.6 ms | 190 | 94 |
| 4 | 11.1 ms | 361 | 90 |
| 8 | 11.9 ms | 670 | 84 |
| 16 | 12.8 ms | 1,247 | 78 |
| 32 | 15.1 ms | 2,120 | 66 |
| 64 | 19.7 ms | 3,242 | 51 |
| 128 | 29.3 ms | 4,367 | 34 |

Read the table one row at a time.

- **1 to 2.** The step gets 0.6 ms longer, and the GPU makes almost twice as many tokens. The second passenger rides almost free.
- **2 to 8.** Each doubling adds less than a millisecond. Eight sequences make 670 tokens per second, 6.7 times one sequence, while each step is only 19% longer.
- **8 to 32.** The step grows from 11.9 to 15.1 ms. Tokens per second still more than triple, from 670 to 2,120.
- **32 to 64.** The step grows by 4.6 ms. Tokens per second rise by about half, to 3,242. Each new sequence now costs a little time.
- **64 to 128.** The step grows by 9.6 ms, to 29.3 ms. Doubling the batch adds only 35% more tokens per second, to 4,367.

From 1 to 128 sequences, each step became about 3 times slower but produced about 44 times more tokens. Each user's answer streams more slowly, from 100 to 34 tokens per second, but the GPU does far more useful work. Batching is the single biggest lever in LLM serving.

### Where the step time goes

The step times follow a simple pattern. Each step costs a **fixed part** of about 10.6 ms, plus about **0.146 ms for each sequence** in the batch. This line, fitted to the measurements from batch 8 to 128, matches every measured step from batch 4 to 128 within 0.2 ms.

{{FIG:step_parts|Decode step time split into a fixed part and a part that grows with each sequence. The dots are the measured step times; the bars are the fitted line, 10.6 ms plus 0.146 ms per sequence.}}

The fixed part is the trip: reading the weights plus the start-up costs of the kernels. Every step pays it once, whatever the batch size. The part per sequence is the work that belongs to one sequence: its arithmetic, and reading its own notes from memory. At batch 4 the fixed part is 95% of the step, so the extra sequences are nearly free. At batch 128 the part per sequence is 18.6 ms, which is larger than the fixed part.

The same line shows the end of the road. When the batch grows without limit, the fixed part stops mattering, and each token costs 0.146 ms. So this GPU can never make more than about **6,865 tokens per second** on this model, however large the batch. At 128 sequences it is already at 4,367.

### Batching on the roofline

On the roofline, batching moves decode to the right. Every sequence in the batch adds about one FLOP of work for each byte of weights read. So a batch of 64 has about 64 FLOPs per byte, and the highest speed rises with it.

{{FIG:roofline_batch|Decode climbs the memory slope of the H100 roofline as the batch grows. Each sequence in the batch adds about one FLOP per byte of weights.}}

On an H100, batch 1 can use at most 0.3% of the compute roof. Batch 64 can reach 21.7%, and batch 128 can reach 43.4%. Decode only reaches the roof near batch 295. Real points sit lower than these ceilings, because each sequence also reads its own notes, which adds bytes without adding weight reuse.

## Real data-centre GPUs

So far the only big GPU on this page was the H100. A new inference engineer meets several others: the older A100, the H200, the newer B200 and AMD's MI300X. Their spec sheets list dozens of numbers. For decode, three of them matter, and you already know what each one means.

1. **Memory capacity (GB).** How big a model fits, and how much room is left for the notes of each conversation.
2. **Memory bandwidth (TB/s).** How fast one decode step can read the weights. This sets the floor of every step.
3. **Peak compute (TFLOP/s).** The roof of the roofline. Decode reaches it only with a very large batch.

> [!DEFINITION] Dense and sparse TFLOP/s
> Spec sheets often show two compute numbers. The "with sparsity" number counts a special mode that skips half of the weights, and it is twice the dense number. LLM serving uses dense matrices, so use the dense number. For the NVIDIA GPUs below, it is half of the sparsity figure.

Here are the five GPUs, from each vendor's own spec page or datasheet. BF16 (bfloat16) is the 16-bit number format that the weights on this page use. The ridge point is the compute divided by the bandwidth, as in the roofline section.

| GPU | Memory | Bandwidth | Dense BF16 compute | Ridge point (FLOPs per byte) |
|---|---|---|---|---|
| NVIDIA A100 80GB SXM | 80 GB | 2.04 TB/s | 312 TFLOP/s | 153 |
| NVIDIA H100 SXM | 80 GB | 3.35 TB/s | 989 TFLOP/s | 295 |
| NVIDIA H200 SXM | 141 GB | 4.8 TB/s | 989 TFLOP/s | 206 |
| NVIDIA B200, HGX | 180 GB | 7.7 TB/s | 2,250 TFLOP/s | 292 |
| AMD Instinct MI300X | 192 GB | 5.3 TB/s | 1,307 TFLOP/s | 247 |

### A worked example: one user on each GPU

Take Llama 3.1 8B in BF16, 16.1 GB of weights. At batch 1, every decode step must read all 16.1 GB. So the least time one step can take is the weight size divided by the bandwidth. On an H100: 16.1 GB ÷ 3.35 TB/s = **4.8 ms**. One step gives one token, so one user can get at most 1,000 ÷ 4.8 = **209 tokens per second**.

$$
\text{least time per step} = \frac{\text{weight bytes}}{\text{bandwidth}}, \qquad \text{most tokens/s for one user} = \frac{1}{\text{least time per step}}
$$

Read it left to right: bytes to move, divided by bytes per second, gives seconds. The same arithmetic on every GPU:

| GPU | Least time per decode step | Most tokens/s for one user |
|---|---|---|
| A100 80GB | 7.9 ms | 127 |
| H100 SXM | 4.8 ms | 209 |
| H200 SXM | 3.4 ms | 299 |
| B200 | 2.1 ms | 479 |
| MI300X | 3.0 ms | 330 |

{{FIG:gpu_bars|Left: the most tokens per second one user can get from Llama 3.1 8B at batch 1, which is set by bandwidth alone. Right: each GPU's ridge point, about the batch size where decode stops being limited by memory.}}

These are upper limits. Real steps are slower, because of the launch costs and the other work from the CUDA section. But the order holds. For one user, the GPU with more bandwidth answers faster, and its compute hardly matters. The H200 has exactly the H100's compute, and it is 43% faster for one user only because its memory is faster.

### Why the ridge stays high

Look at the last column of the first table. From the A100 to the B200, compute grew 7.2 times, while bandwidth grew 3.8 times. So the ridge point almost doubled, from 153 to 292 FLOPs per byte. It does not rise with every chip. The H200 is an H100 with faster memory, so its ridge fell to 206. But over the generations, arithmetic has become cheaper faster than memory traffic.

Each sequence in the batch adds about one FLOP per byte of weights. So the ridge point is about the batch size at which decode stops waiting for memory. On an A100, that is a batch of about 150. On an H100 or a B200, it is about 300. A newer GPU leaves even more of its compute idle at a small batch, so batching matters more with each generation.

### Many GPUs in one server

Data-centre GPUs often come eight to a server, as in NVIDIA's DGX systems. The eight GPUs connect through **NVLink**, NVIDIA's fast link between GPUs, so that one GPU can read another's memory quickly.

{{FIG:server8|An 8-GPU H100 server. Every GPU connects to NVLink switches, so any GPU can exchange data with any other at high speed.}}

> [!DEFINITION] NVLink
> NVIDIA's direct link between GPUs. An H100 SXM has 900 GB/s of NVLink bandwidth, and a B200 has 1.8 TB/s. That is far faster than PCIe, the standard link between a GPU and the CPU, but still slower than the GPU's own HBM.

A model that does not fit on one GPU is split across several of them. The GPUs then exchange partial results over NVLink in every layer. Llama 3.1 70B, for example, has 141 GB of weights in BF16, more than an 80 GB H100 can hold. [Part 7](llm-inference-7-beyond-one-gpu.md) shows how the split works and what it costs.

## Why the batch cannot grow forever

If one more passenger is almost free, why not run a thousand sequences in every step? Two limits stop you.

### Limit 1: every step gets slower

Every user in the batch waits for the whole step. At batch 1, a user gets a new token every 10.0 ms. At batch 128, every user waits 29.3 ms per token. That is still fast enough to read, but there is a cliff. Once the part per sequence is larger than the fixed part, each new sequence slows down everyone already in the batch.

On a large GPU the same thing happens at the roof. Past the ridge point, a step has more arithmetic than the memory read can hide. Then the step time grows in step with the batch, and tokens per second stop growing. The bus is full: one more passenger needs one more trip.

### Limit 2: every sequence needs its own notes

Remember the notes from prefill. Each sequence keeps notes on every token it has read or written, so that it never has to read those tokens again. Those notes are different for every conversation, so the batch cannot share them the way it shares the weights. They must all sit in GPU memory, next to the weights, for as long as the conversation runs.

> [!DEFINITION] KV cache
> The notes a model keeps for every token of a sequence: one pair of vectors, a key and a value, per token in every layer. It grows by one entry per token as the answer is written. Part 2 explains what the keys and values are and why they exist.

For our small model, each token's notes take 12 KiB. A batch of 128 sequences with 512 tokens each needs 128 × 512 × 12 KiB = 768 MiB of notes, which is **0.81 GB**. The weights take 0.99 GB. So at batch 128, the notes are already almost as large as the whole model.

For a larger model the notes are far larger. Llama 3.1 8B needs 128 KiB of notes per token. On an 80 GB H100, its weights take 16.1 GB, which leaves 63.9 GB. A 4,096-token conversation needs 512 MiB of notes, so at most 119 such conversations fit, even before any working memory. At the model's full 128K-token context, one conversation needs 16 GiB, and only 3 fit.

{{FIG:memory_fill|GPU memory split between the weights and the notes of every sequence in the batch. The notes grow with the size of the batch and with the length of each conversation.}}

So on a real server the batch is usually limited by memory for notes, long before the compute roof. The vLLM paper says this in one line.

> [!PAPER] Kwon et al. (2023), Efficient Memory Management for Large Language Model Serving with PagedAttention · §1 · page 2
> [![The next lines of the introduction, with the way the KV cache is managed is critical in determining the maximum batch size highlighted](/img/papers/vllm/p1-intro-batch-limit.png)](/img/papers/vllm/p1-intro-batch-limit.png)
>
> **What it says:** "the way the KV cache is managed is critical in determining the maximum batch size". Managed badly, the cache "can significantly limit the batch size and consequently the throughput".
>
> **What it means:**
>
> The weights are fixed, and the short-lived working numbers of a step are small. So the one part of memory that changes with the batch is the KV cache. Every byte wasted there is a sequence that cannot join the batch, and every missing sequence is lost throughput.
>
> That sentence links this page to the next two. Part 2 measures how big the notes get. Part 3 shows how vLLM stores them so that almost no memory is wasted.

> [!IMPORTANT] The rule of Part 1
> **To make a GPU productive, decode many sequences in the same step.** One read of the weights then serves the whole batch. The batch has two limits. Each step gets slower as the batch grows. And every sequence needs its own memory for its notes, the KV cache. On real servers, memory for the notes is usually the tighter limit.

## Latency or throughput: choose your trade

Batching exposes the main trade-off in serving. **Latency** is how fast one user gets their tokens. **Throughput** is how many tokens the whole GPU makes per second. The batch size moves one against the other.

{{FIG:trade|As the batch grows from 1 to 128, each user receives tokens about three times more slowly, while the GPU as a whole makes 44 times more.}}

> [!DEFINITION] Latency
> How long one user waits: for the first token (TTFT) or between tokens (TPOT).

| You care about | You want | What it costs |
|---|---|---|
| one user, fast | small batches | low TPOT, but an expensive GPU sits mostly idle |
| many users, cheap | large batches | high throughput per GPU, but each token arrives later |
| a quick first word | prefills that are not stuck behind other work | a scheduler that stops long prompts from blocking short ones (Part 3) |

Every serving system, vLLM included, manages this trade. It packs as many sequences into each step as memory allows, and it keeps every user's TTFT and TPOT within limits.

## A hint of a very useful trick

One measurement from earlier deserves a second look. A prefill of **16 tokens took 10.8 ms**. A decode step for **1 token took 10.0 ms**. Sixteen tokens in one pass cost almost the same as one.

{{FIG:spec_hint|One decode step for 1 token takes 10.0 ms; one pass over 16 tokens takes 10.8 ms. Sixteen times the tokens cost 8% more time.}}

That suggests a clever move. A cheap helper model could guess the next few tokens. Then the big model checks all the guesses in a single pass, for about the cost of one decode step. If most guesses are right, you get several tokens for the price of one step.

This is **speculative decoding**, and it can be done so that the output is exactly what the big model would have written anyway. It works for the reason this page is about: decode leaves the compute cores idle. Part 4 builds it from scratch.

## Summary

> [!TAKEAWAYS]
> - An LLM writes by running the whole model once per token, in a loop.
> - **Prefill** reads the whole prompt in one pass and sets the **time to first token**. **Decode** writes one token per step and sets the **time per output token**.
> - A GPU has two speed limits: FLOP/s for arithmetic and bandwidth for moving bytes. An H100 can do about 295 FLOPs in the time it moves one byte.
> - Every step reads all the weights. Decode uses that read for 1 token (about 1 FLOP per byte), so it is **memory-bound**. Prefill uses it for 512 tokens, so it is **compute-bound**.
> - Measured: reading 512 tokens took **27 ms**; writing 512 tokens took **5.2 s**.
> - **Batching** shares one weight read across many sequences: 128 sequences gave **44x** the tokens per second for about **3x** the step time.
> - A line of PyTorch becomes a **kernel launch**: the CPU gives the order, and thousands of GPU threads in blocks run it on the **SMs**. At batch 1 our model ran **880** GPU operations per step, at about **3.4 µs** of launch cost each. **CUDA graphs** and **kernel fusion** cut that cost, but not the weight-read floor.
> - On real GPUs, one user of Llama 3.1 8B can get at most about **127** (A100) to **479** (B200) tokens per second, set by bandwidth. The ridge point is **150 to 300** FLOPs per byte, so a large batch is needed to use the compute.
> - The batch stops at two limits: steps get slower, and every sequence needs memory for its notes, the **KV cache**. On real servers, memory is usually the tighter one.

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

The roofline, batching and memory numbers on this page come from `code/writings/llm-inference/p1_roofline.py`. It counts the model's weights and works out each number from the measured step times.

The kernel count and the launch cost come from `p1_kernels.py`, run on the same Apple GPU. The GPU table and the decode floors come from `p1_gpus.py`, which only does arithmetic on the vendor specs. Their outputs:

```text
device: mps, torch 2.14.1
one decode step (batch 1, 512-token context): 1328 PyTorch operations, 448 of them views or no-op type checks, 880 that do work on the GPU
most common working ops: mul 220, linear 169, add 146, cat 97, pow 49, mean 49, rsqrt 49, neg 48, scaled_dot_product_attention 24, silu 24
per layer (24 layers): about 37 working ops
tiny op 'add on 16 numbers': 3.4 microseconds per op (median of 7 runs of 2,000)
tiny op '16x16 matrix times a vector': 27.7 microseconds per op (median of 7 runs of 2,000)
880 working ops x 3.4 us = 3.0 ms of launch cost per decode step, if each op costs what a tiny op costs
decode step measured now: 15.0 ms (published Part 1 value: 10.0 ms)
```

```text
Llama 3.1 8B weights in BF16: 16.1 GB
GPU              HBM GB   TB/s  TFLOP/s  ridge  floor ms  max tok/s  weights % of HBM
A100 80GB SXM        80   2.04      312    153      7.88        127              20%
H100 SXM             80   3.35      989    295      4.79        209              20%
H200 SXM            141   4.80      989    206      3.35        299              11%
B200 (HGX)          180   7.70     2250    292      2.09        479               9%
MI300X              192   5.30     1307    247      3.03        330               8%
A100 -> B200: compute x7.2, bandwidth x3.8, ridge 153 -> 292
A100 -> H100: compute x3.2, bandwidth x1.6
H100 -> H200: same compute, bandwidth x1.43, ridge 295 -> 206
batch needed to reach the ridge (weights only, about 1 FLOP per byte per sequence): A100 80GB SXM 153, H100 SXM 295, H200 SXM 206, B200 (HGX) 292, MI300X 247
source A100 80GB SXM: nvidia.com/en-us/data-center/a100: 80GB HBM2e, 2,039 GB/s, BF16 312 TFLOPS (624 with sparsity)
source H100 SXM: nvidia.com/en-us/data-center/h100: 80GB, 3.35TB/s, BF16 1,979 TFLOPS with sparsity -> 989 dense
source H200 SXM: nvidia.com/en-us/data-center/h200: 141GB, 4.8TB/s, BF16 1,979 TFLOPS with sparsity -> 989 dense
source B200 (HGX): NVIDIA Blackwell datasheet, HGX B200 per GPU: 180 GB HBM3E, 7.7 TB/s, FP16/BF16 4.5 PFLOPS sparse -> 2,250 dense
source MI300X: amd.com MI300X page and data sheet: 192 GB HBM3, 5.3 TB/s, BF16 1307.4 TFLOPS dense
```

The kernel run measured a decode step of 15.0 ms, not 10.0 ms. The machine was busier during that run than during the earlier measurements, and the tables on this page keep the earlier values.

</details>

## References

1. NVIDIA, [H100 Tensor Core GPU specifications](https://www.nvidia.com/en-us/data-center/h100/) (3.35 TB/s memory bandwidth; 1,979 TFLOPS BF16 with sparsity, so about 989 dense).
2. S. Williams, A. Waterman, D. Patterson. *Roofline: an insightful visual performance model for multicore architectures.* Communications of the ACM, 2009.
3. W. Kwon et al. [*Efficient Memory Management for Large Language Model Serving with PagedAttention*](https://arxiv.org/abs/2309.06180). SOSP 2023.
4. A. Agrawal et al. [*Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve*](https://arxiv.org/abs/2403.02310). OSDI 2024.
5. NVIDIA, [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/archive/12.6.0/cuda-c-programming-guide/index.html) (kernels, thread blocks, warps of 32 threads).
6. NVIDIA, [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/) (132 SMs in the H100 SXM5, 256 KB register file and 256 KB of shared memory and L1 per SM, 50 MB L2).
7. vLLM documentation: [CUDA Graphs](https://docs.vllm.ai/en/stable/design/cuda_graphs/) (the default mode, `FULL_AND_PIECEWISE`, uses full graphs for decode batches).
8. Specs used in the GPU table: NVIDIA [A100](https://www.nvidia.com/en-us/data-center/a100/) (80 GB, 2,039 GB/s, BF16 312 TFLOPS dense), [H200](https://www.nvidia.com/en-us/data-center/h200/) (141 GB, 4.8 TB/s, BF16 1,979 TFLOPS with sparsity), the [Blackwell datasheet](https://dam-cdn.nvd.orangelogic.com/AssetLink/y441155802qub41q118b2852i557jem5.pdf) (HGX B200 per GPU: 180 GB, 7.7 TB/s, BF16 4.5 PFLOPS with sparsity, dense is half), the [HGX platform page](https://www.nvidia.com/en-us/data-center/hgx/) (B200 NVLink 1.8 TB/s per GPU) and AMD [Instinct MI300X](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) (192 GB, 5.3 TB/s, BF16 1,307.4 TFLOPS dense). The H100's 900 GB/s of NVLink is on its spec page, reference 1.

**Next: [Part 2, The KV Cache](llm-inference-2-kv-cache.md).** Every sequence in the batch carries notes on every token it has seen. Part 2 shows what those notes are, how big they get, and why they decide how many people one GPU can serve.
