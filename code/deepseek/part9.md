---
title: "DeepInfer: How DeepSeek Serves Its Models, and How It Differs from vLLM and SGLang"
description: "A code-level tour of DeepSeek's open-sourced production inference engine: its scheduler, 890-byte KV blocks, expert balancing and DSpark, next to vLLM and SGLang."
date: 2026-11-10
tags: [deepseek, inference, serving, vllm, sglang, moe, kv-cache, speculative-decoding]
series: "DeepSeek, From the Ground Up"
series_part: 9
motif: blocks
accent: "#3987e5"
---

In April 2025, DeepSeek published a short note called "The Path to Open-Sourcing the DeepSeek Inference Engine". It explained why the company would not release its own serving engine. The engine was "an early fork of vLLM from over a year ago", it was "tightly coupled with our internal infrastructure", and the team was small. So DeepSeek chose to give pieces to vLLM and SGLang instead.

Eighteen months later, in October 2026, a repository called DeepInfer appeared under the same GitHub account. Its first line says it is "the high-performance inference framework that powers DeepSeek's large-scale production services". The document that announces it makes a large claim: one DeepSeek cluster "currently serves 155 trillion tokens per day". That is DeepSeek's own number, with no method given. Still, it tells you the scale of the system whose engine you can now read.

This part reads that engine. We follow one request through the code and look at each design choice. Then we put each choice next to the matching code in vLLM v0.33.0 and SGLang v0.5.23. We did not run DeepInfer: it needs NVIDIA Blackwell-class GPUs or Huawei Ascend 950 chips, and we have neither. Every claim below comes from the source at a fixed commit, from DeepSeek's documents, or from small analysis scripts we wrote and show.

> [!TIP] The whole idea in one minute
> DeepInfer is DeepSeek's production engine, released as an offline Python library for one model, DeepSeek-V4.1-Flash. It runs one process per GPU group. Every process runs the same kind of step, all prefill or all decode, decided by a vote.
>
> Its KV cache is very small, 890 bytes per token. Only 4 of the 40 layers store keys and values, and they store them in 4-bit numbers. Mixture-of-experts layers run with expert parallelism and a built-in expert load balancer. Speculative decoding (DSpark) checks only as many guessed tokens as pay for themselves at the current load. Four service parts are not released: the HTTP front end, the shared KV store, the load balancers and the prefill-to-decode transfer.

{{FIG:timeline|DeepSeek first released kernels and descriptions, said in April 2025 that the full engine would not be released, and then released the engine itself in October 2026.}}

## 1. The engine's job

An inference engine turns a weights file into a service. It prefills prompts, decodes answers, manages the KV cache, schedules requests and splits the model across GPUs.

The [LLM Inference series](llm-inference-1-prefill-and-decode.md) builds each of these ideas from zero. [Part 2 of that series](llm-inference-2-kv-cache.md) covers the KV cache. [Part 3](llm-inference-3-vllm.md) shows how vLLM pages it and schedules steps. [Part 5](llm-inference-5-sglang-vs-vllm.md) explains SGLang's radix tree, and [Part 7](llm-inference-7-beyond-one-gpu.md) explains expert parallelism and split prefill and decode. This part assumes those ideas and spends its words on what DeepInfer does differently.

## 2. What was released, and what was not

DeepInfer is published under the Apache-2.0 licence, version 0.1.0, marked "Beta". We read commit `a679bb8` of 3 November 2026. The README is short and exact about scope:

```text
DeepInfer/README.md @ a679bb8, lines 3-7 and 33-42

This open-source release provides offline inference with the core features needed for serving at scale.

Supported models:
- DeepSeek-V4.1-Flash

Supported hardware:
- NVIDIA SM100 GPUs
- Huawei Ascend 950 NPUs

Required for a complete inference service (not included in this release):
- A KV cache system
- VE/Prefill/Decode disaggregation (most required engine functionality is already implemented)
- Load balancing across data-parallel replicas for both prefill and decode
```

The code agrees with every line. The model registry has one entry, `DeepseekV41ForCausalLM`. The config loader raises an error unless the checkpoint's `model_type` is `deepseek_v41`. The KV code rejects the compression ratios that V4 and V4-Pro use. So V3, R1, V3.2 and V4 checkpoints do not load.

> [!DEFINITION] SM100
> NVIDIA's name for the compute capability of its Blackwell data-centre GPUs (B200, GB200 and relatives). Hopper GPUs such as the H100 and H800 are SM90.

The hardware limit is also in the code, not only in the README. When the engine starts, it asks for the GPU's peak speed to compute utilisation statistics:

```python
# deep_cluster/hardware/device.py @ a679bb8, lines 151-156
def get_bf16_tflops_limit() -> int:
    if accelerator_kind() == "ascend":
        return 432
    if is_sm_major_10():
        return 2500 if platform.machine() in {"aarch64", "arm64"} else 2250
    raise RuntimeError("unknown GPU type")
```

`LLMEngine.__init__` calls this function on every start. On a Hopper GPU it would raise "unknown GPU type" before any model code runs. That is our reading of the code; we could not run it.

There is also no web server. The engine's docstring says it plainly:

```python
# deepinfer/engine/llm_engine.py @ a679bb8, lines 140-141
    The `LLM` class wraps this class for offline batched inference. This release does not include an
    online-serving wrapper.
```

So the public API is a Python call, `LLM.generate(prompts, sampling_params)`, much like vLLM's offline `LLM` class. Sampling options are few: temperature, top-p, top-k, maximum tokens, log-probabilities, a seed, and structured output (a regex, JSON schema or grammar, compiled with XGrammar). There are no stop strings, no `n` samples per prompt and no priority field.

{{FIG:repo|DeepInfer has about 53,000 non-blank lines of its own code. The largest part is 21,000 lines of TileLang kernels; the scheduler core is 4,600 lines of C++.}}

We counted the repository with a small script that reads files as text (`part9_repo_stats.py`). It finds 313 source files and 53,260 non-blank lines, without the six kernel libraries that come in as git submodules. Python is 89.7% of the lines and C++ is 9.7%. The biggest single part is 21,000 lines of TileLang kernels. The hot loop of the engine (scheduler, block manager, output processing) is C++.

> [!DEFINITION] TileLang
> A Python-embedded language for writing GPU kernels in terms of tiles of data. DeepInfer keeps one copy of most kernels for CUDA and one for Ascend, under `tilelang_ops/cuda` and `tilelang_ops/asc`.

The history shows too. Twenty-three files start with the header "Derived from vLLM; modified by DeepSeek.", and the README thanks vLLM, SGLang and TensorRT-LLM. The class names will feel familiar to a vLLM reader: `LLM`, `LLMEngine`, `SamplingParams`, `CacheEngine`, `Worker`. Under those names, most of the logic is new.

## 3. The system around the engine

DeepSeek published the rest of the design as a long question-and-answer document in the repository (`doc/inference_deployment_qa.md`). It includes a picture of the full V4.1-Flash serving system. We redraw it here and mark the part you can download.

{{FIG:youarehere|The production system. A request enters through DeepSeek Recipe and the Entrypoint, goes through load balancers to prefill and decode services, and uses a KV cache store on 3FS. Only the engine inside the prefill and decode services is open.}}

A request's life in production goes like this. DeepSeek Recipe (a separate open repository) turns the API's JSON into token IDs. The Entrypoint looks up how much of the prompt is already in the KV store, using a chain of hashes over the token blocks. A prefill load balancer hands the request to a prefill engine, which computes the missing KV cache. A decode load balancer then picks a decode engine, which writes the answer.

Prefill and decode run on different machines, as in [Part 7](deepseek-7-running-deepseek.md). DeepSeek now splits three ways: vision encoder, prefill and decode (EPD).

The sizes are very different on the two sides. The Q&A says prefill runs on "ordinary 8-GPU machines" with expert parallelism 2 to 8. Decode runs on supernodes of 64 to 128 GPUs, with expert parallelism 64 to 128. It says "TP and CP are not used for now". The smallest efficient unit is 64 GPUs of prefill plus 64 of decode.

> [!PAPER] DeepSeek-AI (2026), DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression · §3.2.1 · page 19
> [![Page 19 of the V4.1 report: EPD disaggregation, and the persistent KV cache that no longer stores sliding-window KV, which instead goes to a memory pool made from 10% of each machine's DRAM.](/img/deepseek/papers/p9-v41-swa.png)](/img/deepseek/papers/p9-v41-swa.png)
>
> **What it says:** V4.1 uses "Encoder–Prefill–Decode (EPD) disaggregation". Its persistent cache "no longer stores SWA KV"; that part goes to "a distributed memory pool provisioned from 10% of the host DRAM", and global KV stays for at least 72 hours.
>
> **What it means:**
>
> V4.1 keeps two kinds of memory about the past. The global KV is compressed and small, and it is worth keeping for days because users come back to the same long documents. The sliding-window KV covers only the last 128 tokens of each layer, and it is useless once a conversation moves on.
>
> So DeepSeek stores them in different places with different lifetimes. This choice shapes the open engine directly. Inside DeepInfer the two kinds live in two separate GPU pools. The window part even has a fixed slot per sequence, as Section 7 shows.

None of the boxes around the engine are in the repository. The Q&A says the KV cache management and load balancing code is "tightly coupled to our internal infrastructure". It does describe them in detail, and we use those descriptions below, labelled as DeepSeek's description.

## 4. One request through the engine

Now zoom into one prefill or decode service. DeepInfer starts one operating-system process per data-parallel rank. Each process owns one GPU, or a few GPUs if context parallelism is on. Each process builds its own `LLM`, its own `LLMEngine`, its own C++ scheduler and its own block manager.

{{FIG:steploop|One engine's loop. The C++ scheduler picks a batch, the worker runs the model and the sampler, and C++ code commits the accepted tokens before the next step.}}

The path of one prompt through one process has six stops:

1. `LLM.generate()` tokenizes the prompt and calls `add_request` for it.
2. `LLMEngine.add_request()` builds a `Sequence` and puts it in the scheduler's waiting queue.
3. `Scheduler.schedule()`, in C++, picks this step's batch and the KV blocks it needs.
4. `Worker.execute_model_step()` runs the forward pass on the GPU, through CUDA graphs for decode.
5. The sampler picks tokens, and the DSpark draft head guesses the next few.
6. `Scheduler.process_outputs()`, again in C++, appends the accepted tokens and checks for the end.

The processes do not share a scheduler. They stay in step through collective operations: every MoE layer exchanges tokens between all GPUs, and the scheduler takes a vote at fixed points. That vote is the most unusual idea in the engine.

## 5. The scheduler: prefill or decode, never both

In vLLM, one step can mix work. A batch may hold the next token for 200 running requests and a 2,000-token slice of a new prompt. The vLLM project calls this chunked prefill and turns it on by default. Its scheduler fills one token budget per step. SGLang builds separate prefill and decode batches and alternates between them, with an option to mix them.

DeepInfer goes further than SGLang in the same direction: it never mixes them. A step on a normal engine is all prefill or all decode. Every MoE layer needs all GPUs in the group to take part. So every rank must run the same kind of step at the same time. So the ranks vote.

{{FIG:vote|Ranks run decode steps until a vote point. If the vote says prefill, every rank runs a prefill step; a rank with nothing to prefill runs an empty one so the group stays in step.}}

Here is the vote, from the C++ scheduler. The Chinese comments explain the two kinds of vote point.

```cpp
// csrc/engine/scheduler.hpp @ a679bb8, lines 308-321 (comments shortened)
bool is_any_point = (decode_step != 0) && (decode_step % prefill_any_attempt_interval == 0);
bool is_all_point = (decode_step % prefill_attempt_interval == 0);
if (!is_any_point && !is_all_point) {
    return _schedule_decode(acquire);
}
bool want = (get_gpu_cache_usage() <= GPU_CACHE_PREFILL_WATERMARK) && _has_prefill_waiting();
bool do_prefill;
if (is_any_point) {
    do_prefill = (_dp_allreduce(want ? 0 : 1, c10d::ReduceOp::MIN) == 0); // ANY
} else {
    do_prefill = (_dp_allreduce(want ? 1 : 0, c10d::ReduceOp::MIN) == 1); // ALL
}
```

Read it as a rule for a group of cooks who share one oven. Most of the time everyone cooks (decode). Every 70 decode steps there is an "ALL" point: the group bakes (prefill) only if every cook has something to bake. Every 100 decode steps there is an "ANY" point: the group bakes if anyone has something, so no request waits forever. A cook with nothing to bake still stands at the oven, running an empty prefill step.

The numbers 70 and 100 are defaults (`DEEPINFER_PREFILL_ATTEMPT_INTERVAL` and `DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL`). A rank also says "no" when its KV cache is more than 90% full. The trade is clear. Decode steps never slow down because a long prompt shares their batch. But a new prompt may wait up to 100 decode steps for its turn.

In production this matters less than it looks. DeepSeek runs separate prefill-only and decode-only engines, and the code has engine types for them (`PROMPT` and `COMPLETION`). The vote is for the combined `NORMAL` engine, which the open examples use.

### Prefill: first in, first out

Inside a prefill step, the scheduler takes requests from the front of the waiting queue, in arrival order. It stops when the token budget, the sequence limit or free blocks run out. A prompt longer than the remaining budget is cut into chunks, but only if chunked prefill is on. It is off in `EngineArgs` and on in the examples. One more detail shows care for the vision model: a chunk boundary never falls inside an image's tokens.

DeepSeek's production prefill scheduler is smarter than this, but it is not in the release. The Q&A describes a time budget per step and "a fitted quadratic cost model" that predicts how long a chunk will take. We searched the code for it and found nothing; the open engine has only a FLOPs estimator used for statistics.

### Decode: who waits when memory runs out

A decode step gives every running sequence a slot for its next tokens. When free blocks run out, someone must leave. The code sorts the running list by output length, longest first, then serves from the short end and evicts from the long end:

```cpp
// csrc/engine/scheduler.hpp @ a679bb8, lines 622-643
// LCFS
std::sort(running.begin(), running.end(),
    [](const SequencePtr &a, const SequencePtr &b) { return a->get_output_len() > b->get_output_len(); });
...
while (l < r && scheduled.size() < max_seqs) {
    auto &seq = running[r - 1];
    while (!block_manager->can_allocate_single_nolock() && r - l > 1) {
        auto victim_seq = running[l];
        _preempt(victim_seq, ignored_seqs);
        l += 1;
    }
```

By our reading, the victims are the sequences with the longest outputs so far. The comment at the top of the function says the opposite ("preempt the seq with the shortest output"), so treat this as our reading of the code, not DeepSeek's statement. What happens to a victim is clearer. If `stop_preempted_request` is set (the default in `EngineArgs`), the request simply ends with status `FINISHED_PREEMPTED`. Otherwise it goes back to the front of the waiting queue and recomputes later. There is no swap to CPU memory.

vLLM's default policy is first come, first served, with an optional priority policy (`vllm/v1/core/sched/request_queue.py`). On memory pressure it preempts and recomputes. SGLang offers several queue policies, including longest-prefix-match, which favours requests that hit its cache (`python/sglang/srt/managers/schedule_policy.py`).

## 6. Hiding the CPU: asynchronous scheduling

A decode step on a big MoE model can take only a few tens of milliseconds. The CPU work around it (choose the batch, prepare inputs, read back tokens, check stop conditions) is not free. If the GPU waits for the CPU at every step, those milliseconds add up.

{{FIG:zos|Without overlap the GPU idles while the CPU post-processes one step and schedules the next. With overlap the CPU does that work for neighbouring steps while the GPU runs.}}

DeepInfer calls its fix "zero-overhead scheduling", and it is on by default for decode. The worker's main function lays the order out in its docstring:

```python
# deepinfer/worker/worker.py @ a679bb8, lines 96-102
"""The unified entry for model execution: every step (prompt/decode, ZOS on/off,
placeholder) goes through this Forward-Post_process-Sample round.
  1. Forward current step (main model)
  2. Post-process previous step
  3. Sample current step
  4. Post-process current step: in place for prompt / non-ZOS steps;
     ZOS decode steps defer it to the next step to overlap with forward
```

The catch is that the CPU schedules step N+1 before it knows how many tokens step N accepted. With speculative decoding that number varies per request. So the CPU makes an optimistic plan, and a small GPU kernel corrects the sequence lengths once the real counts exist. The Q&A credits the idea to "existing implementations (SGLang, NanoFlow)". vLLM has the same idea as `AsyncScheduler` (`vllm/v1/core/sched/async_scheduler.py`), and SGLang calls it the overlap scheduler.

## 7. The KV cache: 890 bytes per token

This is where DeepInfer differs most from a general engine, because it only has to serve one model. DeepSeek-V4.1-Flash has 40 layers. Its attention mixes three ideas from [Part 6](deepseek-6-v4-and-v4-1.md). There is a sliding window over the last 128 tokens and a compressed global KV. A sparse indexer picks which compressed entries to read.

DeepInfer splits GPU cache memory into two pools.

The **block pool** is paged. One block covers 128 tokens of context for all 40 layers at once. The size is fixed: `resolved_tokens_per_block` simply returns 128.

The **state pool** has one fixed slot per running sequence. It holds the sliding-window rings and small accumulators for compression. Its size does not depend on the context length.

```python
# deepinfer/worker/cache_engine.py @ a679bb8, lines 205-218
@dataclass
class KVCache:
    ica_cache: torch.Tensor | None = None
    """index attn cache: [num_blocks, block_size // ratio, 1, index_bytes_per_token], uint8."""
    ica_state: torch.Tensor | None = None
    """index attn state: [num_state_seqs, ring_size, 2, index_dim], fp32."""
    ca_cache: torch.Tensor | None = None
    """compressed attn cache: [num_blocks, block_size // ratio, 1, bytes_per_token], uint8.
    V41 stores [data | scale] per token: 288 bytes for per-16 FP4 or 528 for MXFP8."""
    ca_state: torch.Tensor | None = None
    """compressed attn state: [num_state_seqs, ring_size, 2, dim], fp32."""
    swa_cache: torch.Tensor | None = None
    """sliding window attn cache: [num_state_seqs, swa_block_size, 1, 528], uint8.
    Each V41 token stores [512B FP8 data | 16B UE8M0 scale]."""
```

Three tricks make the block small. First, most layers do not own a cache at all. The V4.1 config lists `kv_source_layer_ids` = [2, 8, 14, 20], and DeepInfer maps every later layer to the nearest earlier owner. Second, layers 2 to 19 compress two tokens into one entry. Third, each main entry is 512 numbers in 4-bit floating point with a shared scale for every 16 numbers, so 288 bytes.

> [!DEFINITION] FP4 with block scales
> A 4-bit floating-point number can only hold 16 different values. To keep accuracy, each small group of numbers (16 or 32) shares one scale factor stored next to them. 512 four-bit values take 256 bytes; 32 one-byte scales add 32 more.

We wrote `part9_kv_calc.py` to redo the engine's sizing by hand. It copies the rules from `cache_engine.py` and `hugging_face/config.py` and applies them to the real V4.1-Flash `config.json`. This is the core of its output:

```text
$ python part9_kv_calc.py   (results/part9_kv_calc_stdout.txt)
encoder (prefill) layers: 0..19; decoder (YOCO consumer) layers: 20..39

Layers that own a slice of the paged block (all others reuse an earlier layer's cache):
  layer  2  ratio 2  main  64 entries x 288 B = 18,432 B   indexer  64 x 68 B = 4,352 B
  layer  8  ratio 2  main  64 entries x 288 B = 18,432 B   indexer  64 x 68 B = 4,352 B
  layer 14  ratio 2  main  64 entries x 288 B = 18,432 B   indexer  64 x 68 B = 4,352 B
  layer 20  ratio 1  main 128 entries x 288 B = 36,864 B   indexer 128 x 68 B = 8,704 B

bytes per 128-token block: 113,920 (aligned to 1152: 114,048)
=> 890.0 bytes per token of context (891.0 after alignment)
```

The formula behind it is a sum over the owner layers. Each owner stores one main entry and one indexer key for every $$r$$ tokens, where $$r$$ is its compression ratio:

$$
\text{bytes per token} = \sum_{\ell \in \text{owners}} \frac{b_{\text{main}} + b_{\text{index}}}{r_\ell}
$$

Read it left to right. The sum runs over the four owner layers $$\ell$$. For each, $$b_{\text{main}}$$ is the size of one main KV entry (288 bytes in FP4). The term $$b_{\text{index}}$$ is one indexer key: 128 four-bit values plus 4 scales, 68 bytes. Dividing by $$r_\ell$$ spreads the entry over the tokens it covers.

Check it by hand. Layers 2, 8 and 14 have $$r = 2$$, so each costs $$(288 + 68) / 2 = 178$$ bytes per token. Layer 20 has $$r = 1$$ and costs $$356$$ bytes. The total is $$3 \times 178 + 356 = 890$$ bytes per token. That is exactly the "890 bytes per token" global KV figure in DeepSeek's V4.1 technical report, which we cover in [Part 6](deepseek-6-v4-and-v4-1.md). The engine's layout and the paper's number agree to the byte.

{{FIG:kvblock|One 128-token block is 113,920 bytes, split among the four owner layers. Each running sequence also has a fixed slot of about 3 MiB, mostly sliding-window rings.}}

The state slot is the other half of the bill. Every one of the 40 layers, plus the 3 DSpark draft layers, keeps a ring of the last 134 tokens. That is the 128-token window plus 6 tokens so that rejected speculative tokens can be rolled back. Each ring entry is 528 bytes in FP8. With 96,768 bytes of compression state, our script gets 3,139,488 bytes, or 2.99 MiB, per running sequence.

Only the first 20 rings belong to the "encoder" layers whose window is worth saving. The script gets 20 × 134 × 528 = 1,415,040 bytes for that stored part. DeepSeek's Q&A gives the same number for `state_size_to_disk`.

{{FIG:kvbytes|If every layer kept its own 528-byte FP8 entry per token, the cache would cost 21,120 bytes per token; DeepInfer's layout for V4.1 costs 890.}}

What does that buy? Take a round, made-up budget of 100 GiB of free GPU memory for cache. With 256 sequences allowed, the engine reserves 308 state slots and keeps two copies of each, 1.80 GiB in total. The rest holds about 118 million tokens of context, or about 462,000 tokens per sequence if all 256 slots are full. One full 1M-token context needs only 891 MiB of blocks. These are our calculations from the code's rules, not measurements.

### How vLLM and SGLang store the same model

Both vLLM v0.33.0 and SGLang v0.5.23 already support V4.1-Flash. The vLLM tree has a whole package for it, `vllm/models/deepseek_v41/`, with FlashMLA, DSpark and Engram code under `nvidia/` and `common/`. Its KV manager handles mixed attention types through one manager per type (`vllm/v1/core/single_type_kv_cache_manager.py`), including `SlidingWindowManager`. SGLang has V4 memory pools (`mem_cache/deepseek_v4_memory_pool.py`), a V4.1 request window (`mem_cache/dsv41_request_window.py`) and a V4.1 config (`configs/deepseek_v41.py`).

The difference is generality. vLLM and SGLang carry managers for dozens of attention types and a configurable block or page size. DeepInfer hard-codes one layout and lets the state slot and the paged block be exactly what V4.1 needs.

## 8. Prefix caching lives outside the engine

Many requests start with the same tokens: a system prompt, a shared document, the earlier turns of a chat. Reusing the KV cache for that shared start is called prefix caching.

The vLLM engine hashes each full block of tokens together with the hash of the block before it. It then looks blocks up in a table, and this is on by default (`enable_prefix_caching: bool = True` in `vllm/config/cache.py`). SGLang keeps a radix tree of token sequences (`RadixCache` in `mem_cache/radix_cache.py`). Both do this inside the engine, in GPU memory, with optional host and disk tiers.

DeepInfer's engine has no prefix cache of its own. We searched for hash tables and trees and found none. What it has is a `fork_seq` call: given a sequence whose prefix is already computed, it creates a new sequence that shares those blocks. It pins the shared blocks, copies the partly filled last block, and clones the state slot. The benchmark script uses it to build many requests over one shared context.

In production, the prefix cache is a separate service. The Q&A gives its shape. The Entrypoint matches prefixes by "a hash-chain prefix match over metadata only". Block KV lives on a dedicated 3FS SSD cluster, so any prefill machine can hit any prefix. Sliding-window KV stays in the host memory of the machine that made it. The store keeps about 100 PiB, with a time-to-live of 3 days for agent traffic and 7 days for chat.

The reason for this split is load balancing. If the cache sits on the prefill GPU, a request should go to the machine that holds its prefix, but that machine may be busy. With a shared store, any machine can serve any request.

> [!TIP] In production
> Both open engines can also use 3FS for KV storage: vLLM has a connector under `vllm/distributed/kv_transfer/kv_connector/v1/hf3fs/`, and SGLang has a storage backend under `mem_cache/storage/hf3fs/`. If you want DeepSeek's design of "KV on a shared SSD cluster", those are the open paths today.

### Rebuilding the window after a hit

One problem is special to V4.1. A prefix hit restores the compressed block KV, but the sliding-window state may be missing. Recomputing it exactly would mean running the window's tokens through every layer. DeepInfer's answer is "SWA bounded replay": recompute only a short tail of the hit, with attention cut off at the start of that replayed piece. The tail is 140 tokens by default, a little more than the 128-token window.

The result is an approximate window, at a fixed cost. The Q&A says the decoder-side version alone is "nearly halving total prefill computation". It is controlled by `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact` and `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140`, and it is off by default. SGLang has a matching flag, `--enable-encoder-swa-bounded-replay`, with a list of features it does not yet support. The vLLM V4.1 package has a `decoder_replay_layers.py` file. We did not trace how close it is.

## 9. Split prefill and decode: hooks, not wires

The engine is ready for disaggregation, but the wiring is missing. `EngineType` has three values: `NORMAL` (both), `PROMPT` (prefill only) and `COMPLETION` (decode only). The scheduler has a `prompted` queue for sequences that arrive from a prefill engine with their KV already filled.

The transfer itself is a set of empty methods. In `llm_engine.py`, functions such as `init_pd_split`, `send_processed_prompts`, `do_save_kvcache` and `pe_do_save_kvcache` have bodies that are only a docstring, "虚函数" (virtual function). DeepSeek's internal subclass fills them in. You would have to write your own.

vLLM and SGLang both ship working transfer paths. vLLM has KV connectors for NIXL, Mooncake, LMCache, 3FS and others under `vllm/distributed/kv_transfer/kv_connector/v1/`. SGLang has a `disaggregation/` module with `prefill.py`, `decode.py` and back ends for Mooncake, NIXL, Mori and Ascend. For anyone who needs split serving today, this is the clearest gap.

## 10. Parallelism: wide experts, no tensor parallelism

DeepInfer has no tensor parallelism at all. In `ParallelConfig`, the "model parallel size" is simply the context-parallel size. Attention is data-parallel: each GPU runs whole attention layers for its own requests. MoE layers are expert-parallel across the group.

> [!PAPER] DeepSeek-AI (2024), DeepSeek-V3 Technical Report · §3.4.1 · page 19
> [![Page 19 of the V3 report: prefill used 4-way tensor parallelism for attention, and redundant copies of high-load experts adjusted about every 10 minutes.](/img/deepseek/papers/p9-v3-redundant.png)](/img/deepseek/papers/p9-v3-redundant.png)
>
> **What it says:** for V3, "the attention part employs 4-way Tensor Parallelism (TP4)", and a strategy of "redundant experts ... duplicates high-load experts and deploys them redundantly", adjusted "every 10 minutes".
>
> **What it means:**
>
> Two ideas from 2024 meet different fates in DeepInfer. Tensor parallelism for attention is gone. V4.1's cache is so small that one GPU can hold the attention state of its own requests. The Q&A adds that TP "has high communication overhead".
>
> Redundant experts stay and grow. The engine copies hot experts to more GPUs and moves them while serving. What was a periodic job in V3 is now a built-in loop with its own statistics window, as the code below shows.

The Q&A says why decode wants very wide expert parallelism. With 64 or more GPUs, each GPU holds few experts, so it reads less weight memory per step and can run a bigger batch. A chart in the Q&A shows EP64 beating EP8 on per-user speed at every batch size (DeepSeek's own measurement).

When the expert group fits inside one NVLink domain, DeepInfer runs each MoE layer through DeepGEMM's fused "Mega MoE" kernel. That kernel works on a buffer shared by all GPUs in the domain. Outside one domain it uses separate DeepEP dispatch and combine calls around DeepGEMM grouped GEMMs. The Q&A says Mega MoE is "significantly faster than unfused MoE" inside a scale-up domain.

```python
# deepinfer/model_executor/models/v4_modules/mlp.py @ a679bb8, lines 184-188
        self.use_mega_moe = (
            (PLATFORM.is_gpu or self.mma_type == "fp8xfp4")
            and self.mma_type in ("fp8xfp4", "fp8xfp8", "bf16xbf16")
            and PG.is_within_nvl_domain(PG.expert_parallel_group())
        )
```

> [!DEFINITION] NVLink domain (scale-up domain)
> The set of GPUs joined by NVIDIA's fast GPU-to-GPU links, without going through the network. A GB200 NVL72 rack is one such domain of 72 GPUs.

### Expert load balancing, built in

Some experts are popular. If GPU 0 holds the most popular expert, GPU 0 sets the speed for everyone, because the layer waits for the slowest GPU. The expert-parallel load balancer (EPLB) fixes this in two greedy steps.

> [!DEFINITION] EPLB (expert-parallel load balancer)
> A routine that counts how many tokens each expert received recently. It gives spare slots to copies of the busiest experts, and places all copies so that every GPU gets about the same load.

{{FIG:eplb|A toy layer with 8 experts and 4 GPUs. Expert 0 is hot. Giving it 3 extra copies and placing copies greedily brings the slowest GPU from 2.0 times the average load to 1.07 times.}}

Step one gives each spare slot to the expert whose per-copy load is highest. Step two places copies one by one, heaviest first. Each copy goes on the GPU where it raises the peak load least, summed over the recent steps. Here is the heart of step two:

```python
# deepinfer/model_executor/eplb/policy.py @ a679bb8, lines 58-66
        torch.add(loads, vector.unsqueeze(1), out=candidates)
        torch.maximum(candidates, peaks.unsqueeze(1), out=candidates)
        torch.sum(candidates, dim=-1, out=scores)
        # 排除已满的 rank 后选择最低分，argmin 在同分时返回最小 rank ID
        scores.masked_fill_(occupancy == capacity, float("inf"))
        rank = scores.argmin(dim=-1)
```

The comment says: skip GPUs that are full, then take the lowest score; ties go to the lowest GPU number. We rewrote both steps in plain Python for one layer in `part9_toys.py`, on invented loads:

```text
$ python part9_toys.py   (results/part9_toys_stdout.txt, EPLB part)
  no replicas, experts in order: per-step GPU loads [[100, 30, 30, 40], [95, 35, 30, 40], [105, 35, 20, 40]], slowest/mean = 2.00
  replicas per expert after step 1: [4, 1, 1, 1, 1, 1, 2, 1]
  placement (logical expert ids per GPU): [[0, 3, 7], [0, 6, 4], [0, 6, 1], [0, 5, 2]]
  per-step GPU loads [[52, 52, 48, 48], [50, 48, 52, 50], [55, 50, 45, 50]], slowest/mean = 1.07
```

The hot expert gets three extra copies, one on every GPU, and the slowest GPU drops from twice the average to 7% above it. DeepSeek reports the same measure in production as `moe_lb`, the slowest rank's load over the mean. The Q&A gives 1.074 and 1.045 for two prefill layouts and 1.090 for decode at EP64. Decode keeps one spare slot per GPU, which at EP64 rounds up to 7 experts per GPU, and re-plans every 10,000 steps.

vLLM's balancer lives in `vllm/distributed/eplb/`, and its policy file says it is "adapted from DeepSeek EPLB", the library DeepSeek released in 2025. It is off by default (`enable_eplb: bool = False`). SGLang has its own under `python/sglang/srt/eplb/`. The difference is not the idea but the default: in DeepInfer, balancing is part of the normal path, and the examples turn it on.

## 11. DSpark: verify only what pays

Speculative decoding lets a small draft head guess several tokens, and the big model checks them all in one pass. [LLM Inference Part 4](llm-inference-4-speculative-decoding.md) explains why the output does not change. DeepSeek's drafter is DSpark, which [Part 6](deepseek-6-v4-and-v4-1.md) introduced. In DeepInfer the draft head guesses 5 tokens by default (`DEEPINFER_USE_NEXTN_DECODE=5`).

> [!DEFINITION] Acceptance rate
> The chance that a guessed token matches what the big model would have produced. Guesses are checked in order, so a guess counts only if every guess before it was also accepted.

Checking a guess is not free. Under light load, an extra token in the batch costs almost nothing. Under heavy load, it takes a place that another request could use. So the right number of tokens to check depends on both the guess's chances and the current load. DSpark adds a confidence head that predicts each guess's chance, and a scheduler that chooses the lengths.

> [!PAPER] DeepSeek-AI (2026), DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation · §3.2.2 · page 8
> [![Algorithm 1 of the DSpark paper: compute prefix survival probabilities, sort all draft positions by them, and add positions while expected throughput rises.](/img/deepseek/papers/p9-dspark-alg.png)](/img/deepseek/papers/p9-dspark-alg.png)
>
> **What it says:** the scheduler builds "prefix survival probabilities", sorts all candidate draft positions in the batch "descending" by them, and adds them one by one, tracking "Current throughput" as expected accepts times steps per second.
>
> **What it means:**
>
> Picture a cashier choosing which trolleys to scan at a busy till. Each extra item takes a moment, and some items will be returned anyway. The cashier scans the items most likely to be kept first, and stops when the next one costs more time than it is worth.
>
> Because a guess only counts if all earlier guesses survive, its chance can only fall along the block. Sorting by that chance never picks a later guess before an earlier one in the same request. So one greedy pass over a sorted list finds the best set of lengths.

The paper's objective fits in one line. With $$B$$ tokens in the verification pass and $$\tau$$ tokens expected to be accepted, the engine wants to maximise expected tokens per second:

$$
\Theta(B) = \tau(B) \cdot \mathrm{SPS}(B), \qquad \tau(B) = R + \sum_{\text{chosen } (r, j)} a_{r,j}
$$

Here $$R$$ is the number of requests, and each request always gets its one real token. The term $$a_{r,j}$$ is the chance that guess $$j$$ of request $$r$$ survives. Last, $$\mathrm{SPS}(B)$$ is the measured number of steps per second for a pass of $$B$$ tokens. Adding a guess raises $$\tau$$ by its chance but can lower $$\mathrm{SPS}$$.

DeepInfer's C++ code is a close copy of Algorithm 1. It walks the sorted guesses and a measured step-time curve:

```cpp
// csrc/engine/mtp_varlen.hpp @ a679bb8, lines 242-266
        const auto &[graph_size, baseline_ms] = _decode_perf_curve[cur_bsz_idx];
        const double time_ms = baseline_ms + slope_ms * graph_size + sampler_ms;
        const double cur_throughput = cur_expected / time_ms;
        if (cur_throughput > best_throughput) {
            best_throughput = cur_throughput;
            best_bsz = cur_bsz;
        }
    ...
    for (auto &acc : acc_queue_for_bsz) {
        cur_expected += acc.first;
        cur_bsz += 1;
        update_best();
    }
    ...
    seqlens_q.assign(num_seqs, 1);
    for (int i = 0; i < best_bsz - num_seqs; ++i) {
        seqlens_q[acc_queue_for_seqs_q[i].second] += 1;
    }
```

Two engineering details are not in the paper. The step-time curve is indexed by CUDA-graph size and corrected online with a fitted slope and the median sampler time. Also, unlike Algorithm 1, the code does not stop at the first drop. It walks the whole list and keeps the best point, which helps because step time jumps in steps.

{{FIG:varlen|In our toy, expected tokens per millisecond peaks when 16 tokens are verified; checking all 24 tokens is 15% slower than the peak. The step-time curve is invented.}}

Our toy in `part9_toys.py` runs the same greedy on four invented requests and an invented step-time curve. Two requests look like code or JSON with high confidence; two look like chat. The best plan checks all 5 guesses for the two confident requests and 1 guess for each chat request, 16 tokens in total. On this made-up curve it gives 1.141 expected tokens per millisecond, against 0.967 for checking everything.

DeepSeek's own measurement is more modest. On the Open-PerfectBlend dataset with EP64 benchmark data, the Q&A reports these estimated numbers:

| Requests per GPU | Tokens checked, fixed / varlen | Tokens accepted, fixed / varlen | Estimated tokens/s per GPU, fixed / varlen | Gain |
|---:|---:|---:|---:|---:|
| 16 | 5 / 4.44 | 3.32 / 3.24 | 4,891 / 4,816 | −1.5% |
| 32 | 5 / 3.65 | 3.25 / 3.17 | 8,964 / 9,111 | +1.6% |
| 60 | 5 / 3.58 | 3.30 / 3.21 | 13,730 / 14,466 | +5.4% |
| 96 | 5 / 3.24 | 3.30 / 3.15 | 16,817 / 18,660 | +11.0% |

The pattern matches the theory. At light load, checking everything is nearly free and the variable plan even loses a little. At heavy load, skipping weak guesses frees room for other requests. Note that the feature is off by default; the README tells you to set `DEEPINFER_ENABLE_MTP_VARLEN=1`.

vLLM supports DSpark drafts (`vllm/v1/worker/gpu/spec_decode/dspark/` and `vllm/models/deepseek_v41/nvidia/dspark.py`). SGLang has `SpeculativeAlgorithm.DSPARK` and a `deepseek_v4_dspark.py` model. Both also offer EAGLE, n-gram and other drafters. We did not check, line by line, whether either one schedules verification lengths by confidence the way DeepInfer does.

## 12. Numbers, kernels and the rest

### Precision

The Q&A lists where 4-bit numbers are used. The main KV cache uses blocks of 16 values with an E4M3 scale. The indexer keys and the routed-expert weights use MXFP4, blocks of 32 with a power-of-two scale. Activations entering the MoE layers are FP8, so the expert products run as FP8 × FP4. The sliding-window cache stays in FP8.

### Kernels

DeepInfer pins six kernel libraries as git submodules: FlashMLA, DeepGEMM, DeepEP and DeepSelect for NVIDIA, with DeepEP-Ascend and DeepGEMM-Ascend replacing the middle two on Ascend. A grep of the imports shows where each is used.

| Library | What DeepInfer uses it for |
|---|---|
| FlashMLA | `flash_mla_sparse_fwd`, `flash_mla_with_kvcache`, and a fused norm-RoPE-attention kernel |
| DeepGEMM | Mega MoE, FP8 × FP4 grouped GEMMs, the indexer's paged MQA logits, the mHC pre-norm GEMM |
| DeepEP | expert dispatch and combine, Engram buffers, context-parallel all-gather |
| DeepSelect | the indexer's top-k selection |
| TileKernels | the MoE gate, quantization, mHC kernels |
| TileLang (in-tree) | 21,000 lines: sampler, cache stores, metadata preparation, DSpark scheduling on GPU |
| XGrammar | structured output |

DeepJIT, the JIT runtime DeepSeek released in September 2026, is not referenced anywhere in the repository. The Q&A adds that the public FlashMLA lacks "one hand-tuned SASS optimization" of the internal version. That makes the public decoding kernel 5 to 10% slower, while end-to-end speed is within 5%.

### CUDA graphs

Decode steps run through captured CUDA graphs; prefill runs eagerly. Graph sizes are 1, 2, 4, then every 8 tokens up to the maximum. The DSpark draft head has its own graph runner.

### Determinism

Two features are rare in serving engines. With `DEEPINFER_BATCH_INVARIANT=1`, a request's result does not depend on which other requests share its batch. With that flag, varlen off and a fixed `seq_seed`, sampling becomes deterministic. Both work on NVIDIA only. SGLang also has batch-invariant operations (`python/sglang/srt/batch_invariant_ops/`).

## 13. What DeepSeek reports

All numbers in this section are DeepSeek's own. We could not reproduce any of them.

The README shows two charts, "Measured on 4 SM100 GPUs for prefill and 64 for decode", each inside one scale-up domain, with V4.1-Flash at a 131,072-token context. We read the points off the charts, so the values below are approximate:

| Chart | Setup | Low batch | High batch |
|---|---|---|---|
| Prefill | EP4 DP4, 384 + 32 experts, decoder bounded replay of 140 tokens | 4,096 tokens per rank: about 60,000 tokens/s per GPU, about 70 ms per step | 65,536 tokens per rank: about 89,000 tokens/s per GPU, about 730 ms per step |
| Decode | EP64 DP64, 384 + 64 experts, 3 draft tokens, 75% assumed acceptance | 8 requests per rank: about 2,600 tokens/s per GPU, about 325 tokens/s per user | 256 requests per rank: about 26,000 tokens/s per GPU, about 100 tokens/s per user |

> [!WARNING]
> The decode throughput is computed, not counted. The plotting script multiplies the measured steps per second by $$\tau = 1 + 3 \times 0.75 = 3.25$$, an acceptance rate given as an input (`examples/sweep/plot_pareto.py`, lines 58-67). The examples README calls this "full-acceptance accounting". Real traffic with lower acceptance would give lower numbers.

The Q&A adds system-level figures: 155 trillion tokens per day on one cluster, and revenue of US$38 per GPU-hour at the off-peak price. It says load balancing reaches "roughly 80% of a perfectly balanced offline benchmark". For scale, the February 2025 write-up for V3 and R1 reported 608 billion input tokens and 168 billion output tokens in one day, on H800s. The two are not measured the same way, so do not divide one by the other.

## 14. Side by side

| | DeepInfer (a679bb8) | vLLM (v0.33.0) | SGLang (v0.5.23) |
|---|---|---|---|
| Models | DeepSeek-V4.1-Flash only | many architectures, including DeepSeek V2 to V4.1 | many architectures, including DeepSeek V2 to V4.1 |
| Hardware | NVIDIA SM100, Huawei Ascend 950 | in-tree: CUDA, ROCm, TPU, XPU, CPU | in-tree: CUDA, ROCm, Ascend NPU, XPU, MUSA, MPS, CPU |
| API | offline Python `LLM.generate` | OpenAI-compatible server and offline `LLM` | OpenAI-compatible server, offline `Engine`, more front ends |
| Scheduler | C++; each step all prefill or all decode, chosen by a DP-wide vote; FIFO | Python; one token budget, mixed prefill and decode, chunked prefill on by default; FCFS or priority | Python; separate prefill and decode batches, optional mixing; several cache-aware policies |
| Async CPU/GPU overlap | yes ("zero-overhead"), on for decode | yes (`AsyncScheduler`) | yes (overlap scheduler) |
| KV layout | fixed 128-token blocks plus a fixed per-sequence state slot, written for V4.1 | configurable blocks, one manager per attention type | configurable pages, pools per model family |
| Prefix cache | none in the engine; production uses an external 3FS store | hashed blocks, on by default | radix tree |
| Split prefill/decode | engine roles and empty hooks; transfer not released | KV connectors (NIXL, Mooncake, LMCache, 3FS, ...) | `disaggregation/` (Mooncake, NIXL, Mori, Ascend) |
| Parallelism | DP attention + EP, optional context parallelism, no TP | TP, PP, DP, EP, context parallelism | TP, PP, DP attention, EP, context parallelism |
| Expert balancing | built in, used in the examples | EPLB, off by default | EPLB |
| Speculative decoding | DSpark with confidence-scheduled lengths | DSpark, EAGLE, n-gram, others | DSpark, EAGLE, n-gram, others |
| Licence | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Maturity | v0.1.0 "Beta", released October 2026, used in DeepSeek's production per its README | many releases, wide community | many releases, wide community |

## 15. When to choose which

**Choose vLLM or SGLang** if you want to serve DeepSeek models today as an online service. You get an HTTP API, a prefix cache and split prefill and decode. You also get V3, R1, V3.2, V4 and V4.1 on Hopper, Blackwell and other hardware. [Part 7](deepseek-7-running-deepseek.md) shows how. This is the right answer for almost everyone.

**Read DeepInfer** if you build or tune an engine. It shows what a production team keeps when it serves one model at large scale. There is a C++ hot path, and steps of one kind voted across ranks. The cache is shaped exactly for its attention, expert balancing is always on, and a cost curve chooses speculative lengths. Each of these is a few hundred lines you can study.

**Run DeepInfer** if you have SM100 GPUs or Ascend 950 NPUs, you serve V4.1-Flash in large offline batches, and you want DeepSeek's exact numerics. Batch-invariant inference and deterministic sampling are real reasons to choose it for evaluation or data generation. The repository even ships evaluation scripts for AIME 2026, HLE and GPQA Diamond.

**Build on DeepInfer** only if you can write the missing pieces: an online server, a KV store, load balancers and the prefill-to-decode transfer. The README is honest that these are needed for "a complete inference service".

## 16. What is not open, or not supported

- No online server and no OpenAI-compatible API. API conversion is in the separate deepseek-recipe repository.
- One model: DeepSeek-V4.1-Flash (default checkpoint `DeepSeek-V4.1-Flash-1017`). No V3, R1, V3.2, V4 or V4-Pro.
- NVIDIA SM100 and Ascend 950 only. No Hopper, no AMD, no consumer GPUs.
- No KV cache store and no prefix cache inside the engine.
- No prefill-to-decode KV transfer; the hooks are empty.
- No data-parallel load balancers; the production prefill scheduler with its cost model is described but not released.
- No published numbers for Ascend.
- DeepSeek's internal FlashMLA has an extra hand-tuned optimisation that the public one lacks.

> [!TAKEAWAYS]
> - DeepInfer is DeepSeek's production engine, released in October 2026 under Apache-2.0, eighteen months after DeepSeek said it would not release its full engine.
> - The release is an offline Python library for one model, DeepSeek-V4.1-Flash, on NVIDIA SM100 or Huawei Ascend 950.
> - About 53,000 lines: 90% Python (including 21,000 lines of TileLang kernels) and a C++ scheduler core; 23 files still say "Derived from vLLM".
> - One process per data-parallel rank; every step is all prefill or all decode, chosen by a vote across ranks every 70 or 100 decode steps.
> - The KV cache is 890 bytes per token: four owner layers, 2x compression in layers 2 to 19 and FP4 entries. Each sequence also has a fixed state slot of about 3 MiB.
> - Prefix caching, the 3FS KV store, load balancers and prefill-to-decode transfer exist in production but are not in the repository.
> - Expert load balancing is built in; DeepSeek reports slowest-to-mean expert load of 1.045 to 1.090.
> - DSpark checks only the draft tokens that raise expected throughput at the current load; DeepSeek estimates −1.5% to +11% against checking all 5.
> - The README's decode throughput assumes 75% acceptance rather than counting accepted tokens.
> - For serving DeepSeek models online today, vLLM and SGLang remain the practical choice.

## Next

This is the last part of the series for now. An earlier part, [Part 7](deepseek-7-running-deepseek.md), shows how to run DeepSeek models yourself with llama.cpp, KTransformers, vLLM and SGLang. It also shows how much memory each model needs. The two parts fit together: Part 7 is what you can run, and this part is how DeepSeek runs it.

<details><summary>Research notes: commits, method and open questions</summary>

Repositories read, without building or running them: DeepInfer `a679bb86ce36e883a98a97733492b686e5b9a08e` (3 November 2026), vLLM `v0.33.0` (`4ba787ec`), SGLang `v0.5.23` (`297e54fd`). DeepInfer's vendor submodules are pinned at DeepEP `f6aa38f`, DeepGEMM `1fc9c66`, FlashMLA `2e5429f`, DeepSelect `bfa4507`, DeepEP-Ascend `3b25377` and DeepGEMM-Ascend `3a35326`; we did not check them out.

The GitHub API gives 16 October 2026 as the repository's creation date. Press reports of the release appeared a few days later. DeepSeek's linked Zhihu article was not readable by our tools, so we rely on the Q&A document in the repository.

Our scripts are `part9_repo_stats.py` (line counts and environment defaults, reading files as text), `part9_kv_calc.py` (the cache sizing rules applied to the V4.1-Flash config) and `part9_toys.py` (the EPLB and varlen greedy steps on invented data). Their outputs are saved in `code/deepseek/results/part9_*`.

Open questions: the default values of SGLang's server flags live in separate argument-group modules that we did not trace. We did not compare DeepInfer's variable-length verification with the DSpark code in vLLM and SGLang line by line. The preemption victim choice in DeepInfer's decode scheduler disagrees with its own comment; we report the code.

</details>

## References

- DeepSeek-AI. DeepInfer, commit a679bb8. https://github.com/deepseek-ai/DeepInfer
- DeepSeek-AI. DeepSeek V4.1 Flash Inference Deployment Q&A (in the DeepInfer repository). https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md
- DeepSeek-AI (2025). The Path to Open-Sourcing the DeepSeek Inference Engine. https://github.com/deepseek-ai/open-infra-index/blob/main/OpenSourcing_DeepSeek_Inference_Engine/README.md
- DeepSeek-AI (2025). Day 6: DeepSeek-V3/R1 Inference System Overview. https://github.com/deepseek-ai/open-infra-index/blob/main/202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md
- DeepSeek-AI (2024). DeepSeek-V3 Technical Report. arXiv:2412.19437. https://arxiv.org/abs/2412.19437
- DeepSeek-AI (2026). DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation. arXiv:2607.05147. https://arxiv.org/abs/2607.05147
- DeepSeek-AI (2026). DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression. arXiv:2609.19969. https://arxiv.org/abs/2609.19969
- DeepSeek-AI. deepseek-recipe. https://github.com/deepseek-ai/deepseek-recipe
- DeepSeek-V4.1-Flash-1017 model card. https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017
- vLLM v0.33.0. https://github.com/vllm-project/vllm/tree/v0.33.0
- SGLang v0.5.23. https://github.com/sgl-project/sglang/tree/v0.5.23
