---
title: "SGLang Explained: RadixAttention, Scheduling, and How It Differs from vLLM"
description: "Build prefix caching from scratch, check the attention maths, and compare SGLang with modern vLLM using runnable experiments and primary sources."
date: 2026-10-08
tags: [inference, sglang, vllm, kv-cache, llm]
series: "LLM Inference from the Ground Up"
series_part: 5
motif: graph
accent: "#4fc3d9"
---

An assistant reads an 8,000-token company handbook, then answers a question. A second user asks a different question about the same handbook. A third request asks for a summary. The questions differ, but most of what the model has to read is identical.

[Part 3](llm-inference-3-vllm.md) explained how vLLM packs requests into GPU memory and keeps a batch busy. Here we follow another inference engine, **SGLang**, starting with a different question: **how much of the work has already been done?**

We will build two small prefix-cache indexes, run them on exactly the same requests, check reuse with actual attention calculations, and work through where the time and memory go. That gives us a way to compare the engines without relying on a headline speedup.

> [!IMPORTANT] What was actually run
> The companion code ran on an Apple Silicon CPU with Python 3.12.13 and NumPy 2.5.3. It includes a two-layer causal-attention calculation, a compressed radix tree, a block-hash cache, randomized correctness checks, and explicit scheduling simulations. **These are teaching experiments, not measurements of SGLang or vLLM serving a model on a GPU.** The GPU launch commands later are a reproduction recipe; they were not executed for this article. Source behavior was checked on 8 October 2026, with upstream commit IDs recorded in the [research manifest](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/sources.json).

## 1. What SGLang is

SGLang is a system for running language models. You load a model into a server, send prompts, and receive generated tokens. It manages the batch, the KV cache, device execution, and request lifecycle. The project began with a Python language for expressing connected model calls and a runtime designed to execute those calls efficiently. The runtime can also serve ordinary API requests; using the original language frontend is not a requirement. [Sources: the SGLang paper](https://arxiv.org/abs/2312.07104), [current project documentation](https://docs.sglang.io/).

Consider an application that reads a document and launches three tasks:

```text
company policy + document + "Summarise this."
company policy + document + "Extract the risks."
company policy + document + "Draft a reply."
```

The application still chooses the tasks. The inference engine decides how to execute their model calls. If the beginning of each prompt is identical, it should be possible to compute that beginning once, then branch.

{{FIG:tree|A shared prefix can serve several different continuations. The tree indexes cached work; it does not store a ready-made answer to each question.}}

SGLang's best-known mechanism for finding that shared work is **RadixAttention**. Despite the name, it does not replace softmax attention with a new mathematical definition. It organises reusable KV state around token prefixes.

And vLLM? Modern vLLM also caches shared prefixes. A useful comparison therefore asks how each system finds, stores, schedules, and reuses the work. It does not start by giving one engine a cache and switching the other's cache off.

## 2. Why the same prefix can reuse KV

A quick reminder from [Part 2](llm-inference-2-kv-cache.md): at each attention layer, a token produces a **query**, a **key**, and a **value**. The query describes what the current position is looking for. Keys are compared with it; the resulting weights mix the values.

For position $$i$$, causal attention is:

$$
o_i = \sum_{j=1}^{i} \alpha_{ij}v_j,
\qquad
\alpha_{ij}=\frac{\exp(q_i^T k_j/\sqrt{d_k})}
{\sum_{u=1}^{i}\exp(q_i^T k_u/\sqrt{d_k})}.
$$

Here $$d_k$$ is the width of a query/key vector. The dot product scores a match; dividing by its square root keeps the scores from growing simply because the vectors get wider. Softmax turns the scores into positive weights that sum to one.

The crucial detail is the upper limit: **position $$i$$ can read only positions 1 through $$i$$**. A future question cannot change the earlier handbook tokens' states. With the same model, positional setup, and preceding tokens, their keys and values can be reused at every layer.

> [!PAPER] The condition in the original research
> [![Two separately labelled crops from page 4 of the SGLang paper: the RadixAttention section heading and the sentence explaining that KV computation depends only on prefix tokens.](/img/sglang/paper-prefix.png)](/img/sglang/paper-prefix.png)
>
> Source: Zheng et al., [SGLang, arXiv version 2, page 4](https://arxiv.org/pdf/2312.07104v2#page=4). These are actual PDF pixels, with a highlight added. The two excerpts are shown separately so their surrounding layout is not misrepresented.

### A calculation you can run

Our `attention_check()` constructs two causal-attention layers with fixed random weights. It computes a nine-token sequence in two ways:

1. Process all nine tokens together.
2. Process the first six, save each layer's K and V, then process only the last three against that saved state.

The largest absolute difference in the final three output vectors is **4.44 × 10⁻¹⁶**, ordinary floating-point rounding in this float64 calculation.

Then we change the first token but deliberately reuse the old cache. The largest error becomes **0.952**. The code asserts both outcomes, so a broken implementation cannot quietly produce the article's result.

This is an attention-equivalence test with random weights, not a trained language model or a quality evaluation. Its purpose is to isolate the mathematical condition that makes caching possible.

### The same paragraph in a different place is not enough

Suppose two prompts contain the same handbook, but one starts with a different timestamp. The handbook now has a different preceding context. Its later-layer states can differ even when the handbook's own token IDs match.

That is why a prefix cache matches **the sequence from the beginning**, not arbitrary repeated text in the middle. Put stable instructions and documents early when that preserves the intended prompt semantics. Put changing questions later. Match token IDs, not visual similarity: chat templates, whitespace, tool definitions, and tokenization can change the actual sequence. Adapters and multimodal inputs also belong in the cache identity where applicable.

## 3. RadixAttention, one insertion at a time

A **trie** is a tree of shared prefixes. A **radix tree** compresses stretches with no branches into one edge. Instead of creating a separate node for every token, an edge can hold a whole sequence.

Use letters as token IDs for a moment. Our first completed prompt is:

```text
A B C D E
```

We store one edge, `ABCDE`. The next prompt is `ABCXY`. Matching stops after `ABC`, so the edge splits:

```text
root
  └── ABC              shared KV
       ├── DE          first continuation
       └── XY          second continuation
```

A third prompt, `ABCDZ`, follows `ABC`, matches `D` on the first branch, and splits that edge again. Its first four token positions are reusable. Only `Z` needs new prefix computation in this simplified example.

The tree is an index: token sequences lead to locations containing the corresponding KV tensors. Splitting an index edge need not recompute the underlying tensors. The original design maintains the tree on the CPU while the KV tensors occupy device memory. [Source: SGLang paper, Section 3](https://arxiv.org/html/2312.07104v2#S3).

The companion `Radix` class implements matching and edge splitting. Its tests compare the matched length against a deliberately slow reference: scan every previously inserted sequence and find the longest common prefix. Six hundred randomized request checks cover three page-size comparisons, followed by explicit edge-splitting and context-identity checks.

### A cache must also forget

Finished requests leave potentially useful KV behind, but a running request needs guaranteed access to its own state. Reclaiming a node that an active request still uses would be a correctness bug.

The original RadixAttention policy protects in-use nodes and evicts unused leaves by recency. Dropping a leaf preserves a shared ancestor until that ancestor itself becomes an eligible leaf. Current SGLang has configurable eviction policies; its source tracks references and evictable leaves. Our small index deliberately leaves out eviction, references, and real tensor allocation. The later scheduling experiment models a capacity limit separately. [Sources: paper, Section 3](https://arxiv.org/html/2312.07104v2#S3), [pinned SGLang cache implementation](https://github.com/sgl-project/sglang/blob/943621c3364de6948fc02ee13c55b4f1cdf97f3a/python/sglang/srt/mem_cache/radix_cache.py).

## 4. How vLLM finds the same reusable work

vLLM's documented prefix cache uses **hashes of complete blocks**. A block's identity includes its parent prefix hash, its own token IDs, and relevant extra identity such as adapters or multimodal inputs. [Source: vLLM prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/).

An illustrative recurrence is:

$$
h_j = H(h_{j-1},\; t_{jB:(j+1)B},\; e_j).
$$

$$B$$ is the block size, $$H$$ a hash function, and $$e_j$$ the extra identity. The parent hash carries the preceding context forward. Two equal blocks after different prefixes should not produce the same cache identity.

Our `BlockHash` class expresses that idea with SHA-256:

```python
parent = b""
for block in full_blocks(tokens):
    parent = sha256(parent + serialize(block)).digest()
    # This digest identifies the block together with its prefix.
```

This is a teaching sketch. Production serialization, namespaces, allocation, collision handling, and model-specific cache groups need more machinery.

**PagedAttention and RadixAttention answer different questions.** Paging arranges physical KV storage in reusable allocation units. A radix tree finds previously computed prefixes. A system can use a radix index and paged storage together. Comparing the two as mutually exclusive attention algorithms mixes the storage layer with the lookup layer.

### Run both indexes on the same workload

We create 64 prompts. Every prompt contains:

- A shared 1,024-token introduction.
- One of four 256-token group contexts.
- A unique 64-token question.

Requests arrive in alternating group order. There is enough cache space to retain everything. Each request finishes inserting its prompt before the next lookup. The token IDs are synthetic, so tokenization is not a hidden variable.

Without reuse, the number of prompt tokens computed is:

$$
64(1024+256+64)=86{,}016.
$$

With perfect reuse in this workload:

$$
1024+4(256)+64(64)=6{,}144.
$$

The shared introduction is computed once, each group context once, and every question once. Both our radix index and our 16-token block-hash index reach that result.

{{FIG:work|Executed CPU experiment: both indexes eliminate the same repeated token work for aligned prefixes. This is not an engine throughput comparison.}}

| Synthetic workload | No cache: tokens computed | Radix index | 16-token block-hash index |
|---|---:|---:|---:|
| Aligned prefixes | 86,016 | 6,144 | 6,144 |
| Unaligned prefixes | 86,080 | 5,959 | 6,208 |

For the second row, we change the three segment lengths to 1,027, 257, and 61. The exact-prefix index can reuse the last few matching tokens; the block index rounds a reusable prefix down to complete blocks.

For matched length $$m$$, the teaching block model reuses:

$$
m_B=B\left\lfloor\frac{m}{B}\right\rfloor,
\qquad 0\le m-m_B < B.
$$

The difference here is **249 additional token computations across 64 requests**, not an order-of-magnitude engine advantage.

Do not turn this into “SGLang always caches individual tokens.” Current SGLang's radix implementation also rounds matches to the configured page size. A backend or model can impose additional constraints. Our exact-prefix tree illustrates page size 1; our block model illustrates size 16. Neither is a universal default claim. [Source: `RadixKey.match_at` and `page_aligned` at the pinned revision](https://github.com/sgl-project/sglang/blob/943621c3364de6948fc02ee13c55b4f1cdf97f3a/python/sglang/srt/mem_cache/radix_cache.py).

## 5. Token savings are not latency savings

The aligned experiment avoids 92.9% of prompt-token computation, a 14-fold reduction in that token count. It does **not** demonstrate 14 times faster inference.

A useful decomposition of time to first token is:

$$
T_{\mathrm{first}} = T_{\mathrm{queue}} + T_{\mathrm{input}}
+ T_{\mathrm{lookup/transfer}} + T_{\mathrm{remaining\ prefill}}
+ T_{\mathrm{first\ output}}.
$$

Queueing can dominate a busy service. An offloaded cache must be transferred. The uncached suffix still needs attention over the retained prefix. The first output also requires model execution and transport to the client.

You can see the remaining attention work directly. If a prompt has $$n$$ tokens and $$p$$ are cached, the new query rows still attend to earlier keys. The number of causal query-key pairs for those new rows is:

$$
\sum_{i=p+1}^{n} i
=\frac{n(n+1)-p(p+1)}{2}.
$$

For $$n=10$$ and $$p=8$$, we compute 19 pairs rather than 55: the ninth token attends to nine positions and the tenth to ten. We do not attend only within a two-token suffix. Other layer operations save different amounts, and GPU kernels have their own efficiency curves.

There is also a whole-request limit. If prefill takes 100 ms and generation takes 900 ms, making prefill five times faster gives:

$$
\frac{100+900}{100/5+900}=1.087.
$$

That is about **1.09 times faster end to end**, even though the prefill improvement was fivefold. These times are assumed arithmetic inputs, not measurements.

Prefix reuse matters most when repeated input work is a substantial part of the workload. It cannot eliminate the sequential cost of generating a long answer. [Source for the prefill/decode distinction: vLLM APC documentation](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/).

## 6. How much memory can sharing save?

For a conventional transformer with grouped-query attention, unsharded KV payload per token is:

$$
M_{\mathrm{token}}=2L H_{\mathrm{KV}}d_h b.
$$

The factor 2 counts keys and values. $$L$$ is the layer count, $$H_{\mathrm{KV}}$$ the number of KV heads, $$d_h$$ the head width, and $$b$$ bytes per element. Query-head count is not the right quantity when several query heads share KV heads.

Take a hypothetical configuration with 32 layers, eight KV heads, width 128, and two-byte values:

$$
2\times32\times8\times128\times2
=131{,}072\;\text{bytes}=128\;\text{KiB/token}.
$$

An 8,192-token prefix therefore occupies **1 GiB** of KV payload. Thirty-two independent copies occupy 32 GiB; one shared copy occupies 1 GiB, before private suffixes and metadata. This arithmetic appears in the executed results file.

This is not total device memory. We have excluded model weights, activations, graph buffers, allocator overhead, and private generated tokens. Sharding changes per-device placement. MLA, sliding-window attention, hybrid state, and cache quantization require their own accounting.

Sharing is valuable in two ways: it avoids producing the same states repeatedly, and it can keep multiple requests from storing duplicate states simultaneously.

## 7. Why scheduling and routing change the result

Imagine four tenants, each with a different 1,024-token prefix. The cache can hold only two whole tenant prefixes. Requests arrive as:

```text
A B C D A B C D ...       eight rounds, 32 requests
```

Under a simple least-recently-used whole-prefix policy, every prefix has been evicted when its next request arrives. Our simulation gets **32 misses and zero hits**.

Now suppose all requests are already waiting and we group them:

```text
A A A A A A A A B B ... C C ... D D ...
```

The same simulation gets **four misses and 28 hits**. Including a fresh 64-token question per request, prompt work falls from **34,816 to 6,144 tokens**.

This is a model of locality, not SGLang's actual scheduler. Real arrivals have deadlines; postponing a tenant to improve cache reuse can make that tenant's wait unacceptable. The useful objective is completed work within latency targets, not the largest cache-hit number.

At several replicas, routing matters too. A warm prefix on worker A is no help if the next request goes to worker B, unless the system can obtain that state there. SGLang's v0.4 release introduced a cache-aware router alongside CPU/GPU overlap. Its reported gains belong to its stated shared-prefix workloads, not every deployment. [Source: the team's December 2024 release report](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/).

A practical routing decision has to balance the expected saving from a cache hit against extra queueing or transfer time. A warm but overloaded worker can lose to a cold idle worker.

## 8. Keeping the CPU out of the GPU's way

For each batch, the host prepares request metadata and memory mappings, then the device runs the model. If those phases run serially, the device waits during preparation. SGLang's overlap scheduler prepares a following batch while current device work is in flight. Dependencies still need explicit handling; “zero overhead” describes work being hidden, not work disappearing. [Source: SGLang v0.4 scheduler explanation](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/).

Assume preparation takes $$C=2$$ ms and GPU work $$G=8$$ ms. For $$N=100$$ equal batches:

$$
T_{\mathrm{serial}}=N(C+G)=1000\;\text{ms},
$$

$$
T_{\mathrm{pipeline}}\approx C+G+(N-1)\max(C,G)=802\;\text{ms}.
$$

The first batch fills the pipeline. Each later batch can complete every 8 ms if preparation is fully hidden. The ideal improvement is about 1.25 times for this finite example. Synchronization, variable batch sizes, and data dependencies can reduce it.

{{FIG:overlap|An illustrative timeline generated by the companion code. These are assumed durations, not profiler traces.}}

Modern vLLM also has asynchronous scheduling support. Its pinned scheduler configuration explicitly describes avoiding GPU utilization gaps. This is another shared optimization whose exact implementation and compatibility constraints matter more than a yes/no feature label. [Source: vLLM scheduler configuration](https://github.com/vllm-project/vllm/blob/458ba2edf85bc9b7ebc0d5141f34ea658520faf5/vllm/config/scheduler.py).

## 9. Beyond device memory: HiCache

SGLang's **HiCache** extends prefix storage through GPU memory, host memory, and an optional storage backend. Host caches belong to individual instances; a storage tier is shared only when its backend is configured that way. A cache hit therefore needs a location as well as a length. [Source: HiCache design documentation](https://docs.sglang.io/docs/advanced_features/hicache_design).

{{FIG:tiers|More cache capacity introduces a transfer decision. A host or storage hit is not equivalent to an already-resident device hit.}}

For payload size $$S$$ and effective transfer bandwidth $$B$$, a first approximation is:

$$
T_{\mathrm{restore}} \approx T_{\mathrm{fixed}}+\frac{S}{B}.
$$

At an assumed effective 25 GB/s, transferring our 1 GiB prefix alone takes **42.95 ms**. This uses binary GiB for the payload and decimal GB/s for bandwidth. Allocation, lookup, staging, and contention add time; pipelining can hide part of it.

If recomputing the prefix takes 20 ms, a blocking 43 ms copy loses. If recomputation takes 200 ms, the same copy may help. Measure both on the actual path. These are break-even examples, not measured link or model performance.

vLLM also has KV transfer and offload integrations, including LMCache. “SGLang has hierarchical caching; vLLM cannot move KV” would be an inaccurate comparison. The setup, backend, model coverage, and operational behavior must be compared. [Source: vLLM LMCache integration](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/).

## 10. Structured output is a separate question

A cache can avoid repeated input work. A grammar can constrain which output tokens are allowed. These solve different problems.

For a schema that requires a JSON object, a grammar-aware decoder masks tokens that cannot continue a valid output. That can reduce formatting failures, but it does not prove that an extracted amount or date is correct. Measure schema validity and semantic accuracy separately.

Both SGLang and vLLM document structured outputs, including grammar backends such as XGrammar. Supported schema features and interactions with reasoning or tool parsers vary by release and model. Do not interpret “JSON supported” as identical behavior across all schemas. [Sources: SGLang structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs), [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/).

## 11. SGLang versus vLLM: the comparison to keep

| Question | SGLang | vLLM | What to inspect |
|---|---|---|---|
| How is prefix reuse indexed? | Radix-tree family of caches | Hash-linked full-block prefix cache | Actual hit length, page/block alignment, model-specific cache behavior |
| Does it batch and manage KV memory? | Yes | Yes | Allocation pressure, preemption, batch limits |
| Can host preparation overlap device work? | Overlap scheduler | Async scheduling support | Compatibility and gaps in a real profiler trace |
| Can it constrain output? | Structured-output grammar backends | Structured-output grammar backends | Schema coverage, valid output, semantic correctness |
| Can KV live beyond one device? | HiCache and distributed serving features | KV connector/offload integrations | Transfer cost, capacity, supported configurations |
| Is it universally faster? | No demonstrated universal winner | No demonstrated universal winner | The same workload, hardware, output quality, and latency target |

The source trail above explains the rows; this table is not a fresh GPU benchmark. The original SGLang paper's **up to 6.4 times** throughput result is a historical result against its tested baselines. It is not a measured advantage over the October 2026 vLLM source. [Source: paper abstract and evaluation](https://arxiv.org/abs/2312.07104).

If your application already works well on one engine, test a specific hypothesis before switching. For example: “shared long documents are driving our first-token latency; does the alternative retain and route those prefixes better under our memory limit?” That is a measurable question. “Which name is faster?” is not.

## 12. Run the code, then test a real server

The complete [companion directory](https://github.com/ishwar6/ishwar-books/tree/main/code/sglang) includes the experiment, saved JSON, source manifest, figure generator, and research-screenshot recipe.

From the repository root:

```bash
python3 -m venv /tmp/sglang-article-env
/tmp/sglang-article-env/bin/pip install -r code/sglang/requirements.txt
/tmp/sglang-article-env/bin/python code/sglang/experiments.py
/tmp/sglang-article-env/bin/python code/sglang/figures.py
/tmp/sglang-article-env/bin/python code/sglang/assemble.py
```

The CPU experiment needs only NumPy. Pillow and PyMuPDF in the requirements support screenshot reproduction. The assembler inserts saved figures into this article and checks local image paths and unresolved placeholders.

### Launch either engine on a compatible GPU machine

Use separate environments with a recorded engine release, matching driver/toolchain, and the **same downloaded model snapshot**. Point both commands at that snapshot so a moving model revision does not become a hidden variable. Install each engine using its hardware-specific guide: [SGLang](https://docs.sglang.io/docs/get-started/install), [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/).

Run one server at a time on the same device:

```bash
# SGLang environment; replace the snapshot path.
python -m sglang.launch_server \
  --model-path /models/qwen2.5-0.5b-instruct-snapshot \
  --served-model-name comparison-model \
  --host 127.0.0.1 --port 30000
```

```bash
# vLLM environment; the exact same model snapshot.
vllm serve /models/qwen2.5-0.5b-instruct-snapshot \
  --served-model-name comparison-model \
  --enable-prefix-caching \
  --host 127.0.0.1 --port 8000
```

These are baseline examples, not performance-tuned configurations. Check flags against the installed version's `--help`. SGLang's serving entry point and request format are documented in its [request tutorial](https://docs.sglang.io/docs/basic_usage/send_request); vLLM's server exposes its [compatible API](https://docs.vllm.ai/en/latest/serving/online_serving/).

The companion client uses raw completions so differing chat templates do not obscure a first smoke test:

```bash
python3 code/sglang/probe_server.py \
  --url http://127.0.0.1:30000 --model comparison-model \
  --output /tmp/sglang-probe.json

python3 code/sglang/probe_server.py \
  --url http://127.0.0.1:8000 --model comparison-model \
  --output /tmp/vllm-probe.json
```

It sends two different questions after an identical long text prefix and records the response, observed first-text latency, total latency, and server-reported token usage when available. **Two requests are a connectivity/reuse smoke test, not a throughput benchmark.** The second may benefit from the first; the first may already find cached state if the server was used earlier. Inspect engine cache metrics to establish whether reuse actually occurred.

For a defensible performance comparison, expand beyond that probe:

1. Record model and tokenizer revisions, engine versions, GPU model/count, quantization, attention backend, memory limits, and every launch argument.
2. Replay identical request traces: unique prompts, repeated system prompts, shared long documents, multi-turn sessions, and any structured outputs your application needs.
3. Separate cold-start, deliberately warmed, and steady-state runs. Include traces whose working set exceeds cache capacity. Use both arrival-rate sweeps and concurrency sweeps; fixed-concurrency tests hide some overload behavior.
4. Keep generation settings and intended output lengths comparable. Record actual generated tokens and validate output quality; an engine that stops early has done less work.
5. Report errors, throughput, p50/p95/p99 first-token and whole-request latency, decoding latency, cache-hit tokens, and preemptions. Repeat runs and report variability.

For a response of $$n_{\mathrm{out}}>1$$ tokens, average time per output token after the first is often summarized as:

$$
\mathrm{TPOT}=\frac{T_{\mathrm{last}}-T_{\mathrm{first}}}{n_{\mathrm{out}}-1}.
$$

Streaming chunks can contain several tokens, so packet timestamps are not exact token-generation timestamps. Use engine instrumentation when the distinction matters. The probe intentionally does not label chunk intervals as inter-token latency.

Finally, count the requests that satisfy your service target:

$$
\mathrm{goodput}=\frac{\#\{\text{successful requests meeting the latency target}\}}
{\text{measurement duration}}.
$$

High throughput with long tail delays may be the wrong trade for an interactive assistant. A batch job may prefer it. The engine choice follows the workload and the target.

## What to remember

SGLang makes repeated prompt structure an explicit part of execution through its radix-cache design. Modern vLLM can reuse the same prefixes through a different index. Our aligned experiment shows why the representation alone does not determine the saving: both recover the same repeated work.

The harder questions come next: whether the prefix is still resident, which worker holds it, how much new work remains, whether transfers beat recomputation, and whether the request finishes on time. Those are the questions to bring to a real SGLang-versus-vLLM benchmark.

## Sources and reproduction

- Zheng et al., [SGLang: Efficient Execution of Structured Language Model Programs](https://arxiv.org/abs/2312.07104), with [version 2 PDF](https://arxiv.org/pdf/2312.07104v2) used for the screenshot.
- SGLang team, [v0.4 scheduler and router explanation](https://www.lmsys.org/blog/2024-12-04-sglang-v0-4/), December 2024. Historical results, explicitly dated.
- SGLang, [HiCache design](https://docs.sglang.io/docs/advanced_features/hicache_design), [structured outputs](https://docs.sglang.io/docs/advanced_features/structured_outputs), and [request tutorial](https://docs.sglang.io/docs/basic_usage/send_request).
- vLLM, [prefix-cache design](https://docs.vllm.ai/en/latest/design/prefix_caching/), [APC behavior](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/), [structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/), and [LMCache integration](https://docs.vllm.ai/en/latest/examples/disaggregated/lmcache/).
- [Pinned source revisions and checksums](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/sources.json), [executed experiment results](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/results/experiments.json), and [reproduction instructions](https://github.com/ishwar6/ishwar-books/blob/main/code/sglang/README.md).
