# Research dossier 06: DeepInfer, compared with vLLM and SGLang

Written 2026-11-10 for Part 9. Every claim maps to a file and line range at the commits below. Repositories were
cloned read-only into the session scratchpad; no code from them was built, installed or executed. Our only scripts
(`part9_repo_stats.py`, `part9_kv_calc.py`) read DeepInfer files as text or re-implement its sizing rules by hand.

## 0. Sources and commits

| Source | Commit / version | Date |
|---|---|---|
| github.com/deepseek-ai/DeepInfer (`git clone --depth 1`) | `a679bb86ce36e883a98a97733492b686e5b9a08e` ("docs: update V4.1 Flash Q&A to v0.2") | commit 2026-11-03T15:31:40+08:00 |
| DeepInfer vendor submodules (gitlinks at that commit, not checked out) | DeepEP `f6aa38f`, DeepEP-Ascend `3b25377`, DeepGEMM `1fc9c66`, DeepGEMM-Ascend `3a35326`, DeepSelect `bfa4507`, FlashMLA `2e5429f` | |
| github.com/vllm-project/vllm (already cloned) | tag `v0.33.0` = `4ba787ec9959e566d3b587a796dff07673324da5` | 2026-10-31 |
| github.com/sgl-project/sglang, latest release tag via GitHub API | `v0.5.23` = `297e54fd105e974cdf401e5b4d20253a22c16f90` (release published 2026-11-02) | commit 2026-10-28 |
| open-infra-index `OpenSourcing_DeepSeek_Inference_Engine/README.md` (April 2025) | raw from `main` | |
| open-infra-index `202502OpenSourceWeek/day_6_one_more_thing_deepseekV3R1_inference_system_overview.md` | raw from `main` | 2025-03-01 |
| DeepInfer announcement text | `doc/inference_deployment_qa.md` in the repo ("Today we are open-sourcing DeepInfer ...", v0.2) | |
| GitHub API for DeepInfer | `created_at` 2026-10-16T07:47:23Z, license `Apache-2.0`, 25,779 stars on 2026-11-10 | |

Release date: our earlier notes and the GitHub API say 16 October 2026 (repo creation). Press coverage (technode.com,
2026-10-21; a search summary said "October 20") dates the public announcement a few days later. Use "October 2026" in
prose; the exact public-announcement day is UNVERIFIED. `api-docs.deepseek.com/news/news261016` redirects to the
generic first-call page; no DeepSeek news post about DeepInfer was found. The Zhihu post linked from the Q&A
(zhuanlan.zhihu.com/p/2095560295930508953) returns 403 to our tools: not read.

## 1. The April 2025 position and the reversal

- open-infra-index, "The Path to Open-Sourcing the DeepSeek Inference Engine": "our inference engine is built upon
  vLLM"; "we initially considered open-sourcing our full internal inference engine, we identified several challenges";
  three reasons: "Codebase Divergence" ("an early fork of vLLM from over a year ago"), "Infrastructure Dependencies"
  ("tightly coupled with our internal infrastructure"), "Limited Maintenance Bandwidth". Decision: "we've decided to
  collaborate with existing open-source projects as more sustainable alternatives", by "Extract Standalone Features"
  and "Share Optimizations". Note: the text does not literally say "we will not open-source"; it says they decided
  against releasing the full engine and chose upstream collaboration instead.
- Day 6 overview (Feb 2025, V3/R1 on H800): prefill "EP32, DP32", 4 nodes, 32 redundant experts; decode "EP144,
  DP144", 18 nodes; dual-batch overlap; three load balancers (prefill, decode, expert-parallel); 608B input tokens
  in 24 h, 56.3% hit the on-disk KV cache; ~73.7k input tok/s per H800 node in prefill, ~14.8k output tok/s per
  node in decode; 545% theoretical cost profit margin (with the caveat that real revenue is lower).
- October 2026: DeepInfer released. README line 3: "DeepInfer is the high-performance inference framework that
  powers DeepSeek's large-scale production services. This open-source release provides offline inference with the
  core features needed for serving at scale." README line 95: "originally built primarily on vLLM, with additional
  inspiration from SGLang and TensorRT-LLM". 23 files carry the header "Derived from vLLM; modified by DeepSeek."
  (part9_repo_stats).
- Q&A lines 17-21: "A single cluster currently serves 155 trillion tokens per day, with overall revenue of US$38 per
  GPU-hour (at the off-peak price)." (DeepSeek claim; no method given.) "The KV cache management and load balancing
  components ... are tightly coupled to our internal infrastructure, so that code is not yet fully open-sourced."
  Also announces deepseek-recipe (Rust tokenization/API conversion, MIT per dossier 01).

## 2. What the release contains, and what it does not

README lines 5-42:
- Supported models: only DeepSeek-V4.1-Flash. Code agrees: `deepinfer/model_executor/models/__init__.py:15-17` registers
  only `"DeepseekV41ForCausalLM"`; `deepinfer/hugging_face/config.py:241-249` raises unless `model_type ==
  "deepseek_v41"`; `model_metadata_utils.py:97` asserts "只支持 MoE 模型（V4.1）"; `:52-53` rejects C4/C128 compression
  ("C4 and C128 have been removed"), so V4 / V4-Pro checkpoints are not accepted. Default checkpoint
  `deepseek-ai/DeepSeek-V4.1-Flash-1017` (`examples/_config.py:32`).
- Hardware: "NVIDIA SM100 GPUs", "Huawei Ascend 950 NPUs" (README 35-36). Code: `deep_cluster/hardware/device.py:151-164`
  `get_bf16_tflops_limit()` returns 432 for Ascend, 2500/2250 for SM major 10, else `raise RuntimeError("unknown GPU
  type")`; it is called unconditionally in `LLMEngine.__init__` (`llm_engine.py:304`). So Hopper (SM90) is not
  supported: the engine would raise at start-up (reading of the code; not executed).
  Prerequisites (README 45-52): Linux, GCC >= 13, Python 3.12, CUDA 13 (NVIDIA) or CANN 9.2 (Ascend).
- Not included (README 38-42): "A KV cache system"; "VE/Prefill/Decode disaggregation (most required engine
  functionality is already implemented)"; "Load balancing across data-parallel replicas for both prefill and decode".
- No server: `llm_engine.py:140-141` "The `LLM` class wraps this class for offline batched inference. This release
  does not include an online-serving wrapper." The only HTTP code is a profiling endpoint
  (`engine/profile_server.py`, FastAPI, off unless `DEEPINFER_PROFILE_HTTP_ENABLE`). No OpenAI-compatible API in the
  repo; API conversion lives in the separate deepseek-recipe repo (not an HTTP server per the Q&A description; not
  inspected in depth).
- API surface: `LLM(model, ..., expert_parallel_size, gpu_memory_utilization, ...)` and `LLM.generate(prompts |
  prompt_token_ids, sampling_params, multi_modal_data)` (`entrypoints/llm.py:54-142`). `SamplingParams`
  (`sampling_params.py:49-61`): temperature, top_p, top_k, ignore_eos, max_tokens, logprobs, prompt_logprobs,
  top_logprobs, first_n_token_no_eos, so_args, golden_token_ids, seq_seed. No stop strings, no n>1, no penalties, no
  per-request priority field.
- Licence: Apache-2.0 (`pyproject.toml` `license = "Apache-2.0"`, LICENSE file, GitHub API). Version 0.1.0,
  classifier "Development Status :: 4 - Beta".

## 3. Language split and size (part9_repo_stats, vendor/ excluded)

313 source files, 53,260 non-blank lines: Python 47,761 (89.7%), C++ 5,153 (9.7%), shell 346. Largest component:
TileLang kernels in `deepinfer/model_executor/layers/tilelang_ops/` (20,997 lines, Python-embedded TileLang, with
`cuda/` and `asc/` backends). C++ engine core `csrc/engine/` 4,598 lines (scheduler.hpp 1,329; sequence, block
manager, MTP varlen, input preparation, ZOS). No Rust in DeepInfer (Rust is in deepseek-recipe per the Q&A). 58
`DEEPINFER_*` environment knobs in `deepinfer/environ.py`.

## 4. Process and component architecture

- One OS process per data-parallel (DP) rank. `examples/_launch.py:26-50` spawns `main_fn(dp_size, dp_rank)` per rank;
  `examples/_config.py:55-61` sets `CUDA_VISIBLE_DEVICES` to `cp_size` devices per engine. Each process builds its own
  `LLM` -> `LLMEngine` with its own C++ `Scheduler` and `BlockManager` (`llm_engine.py:265-267`). Context-parallel
  ranks > 0 are extra worker processes (`llm_engine.py:377-395`).
- Ranks coordinate through collectives: `has_unfinished_requests` does an all-reduce MAX over the DP group
  (`llm_engine.py:665-679`); the scheduler votes on step type with all-reduce MIN (`scheduler.hpp:295-341`); MoE
  layers exchange tokens through DeepEP or Mega MoE.
- `ParallelConfig` (`config.py:150-182`): `model_parallel_size = context_parallel_size`; there is no tensor
  parallelism parameter at all. DP defaults to `ep // cp`. Engram has its own group size (`engram_parallel_size`).
- Engine types (`engine/engine_type.py`, `scheduler.hpp:37-43`): NORMAL (prefill + decode), PROMPT (prefill only),
  COMPLETION (decode only). PD hooks are empty virtual methods in the released `LLMEngine` (`llm_engine.py:800-858`:
  `init_pd_split`, `send_processed_prompts`, `do_save_kvcache`, `pe_do_save_kvcache` ... with docstrings "虚函数").
  The scheduler has a `prompted` queue for sequences arriving from a prefill engine (`scheduler.hpp:595-620`). The
  transfer itself is not in the release.

## 5. Scheduler (C++, `csrc/engine/scheduler.hpp`)

- Steps are pure prefill or pure decode, never mixed. NORMAL engine (`:295-341`): at "ALL" points (every
  `prefill_attempt_interval` = 70 decode steps, `environ.py:181`) it prefills only if every DP rank wants to; at "ANY"
  points (every 100 decode steps, `environ.py:184`) it prefills if any rank wants to; ranks with nothing to prefill
  run an empty prompt step "陪跑" to keep step types aligned (`:339-340`). `want` also requires GPU cache usage <= 0.9
  (`GPU_CACHE_PREFILL_WATERMARK`, `:80`, `:313`).
- Prefill (`:351-554`): FIFO from `waiting` deque; stops at the first unschedulable head; limits: `max_num_prompts`
  (= max_num_seqs), `max_num_batched_tokens`, `prefill_max_sum_seqlen_k`; chunking when the prompt exceeds the
  remaining token budget, only if `chunked_prefill` (`:504-507`); NORMAL engine allows at most one chunked sequence
  per step (`:543-545`); chunks never split an image (`_adjust_chunk_size`, `:232-260`); CP alignment to `2*cp`.
  `EngineArgs.chunked_prefill` default False (`arg_utils.py:42`) but `examples/_config.py:120` sets True.
- Decode (`:559-660`): admits `prompted` sequences, then sorts `running` by output length descending
  (`:624-625`), allocates one block slot per sequence from the shortest-output end, and preempts from the
  longest-output end when blocks run out (`:627-643`). The code comment says "LCFS" and "抢占 output 最短的 seq"; the
  code as written picks victims at index `l` of a list sorted by output length descending, i.e. the longest outputs
  (my reading; flag in text as "my reading of the code").
- Preemption (`:766-790`): if `stop_preempted_request` (EngineArgs default True, LLM() sets False) or structured
  output: finish the request as FINISHED_PREEMPTED; otherwise reset and push to the front of `waiting` for
  recomputation (no swap to CPU).
- Production prefill policy described in Q&A lines 135-143 (compute-quota budget, fitted quadratic cost model, 4 length
  tiers, pull model): not in the release (`grep quota` finds nothing). `FlopsEstimator` (`csrc/engine/flops_estimator.hpp`)
  is used for stats (`scheduler.hpp:712-719`).
- Priorities: none (no priority field; FIFO waiting queue).

## 6. Asynchronous ("zero-overhead") scheduling

- `DEEPINFER_ZERO_OVERHEAD_SCHEDULING`, set to 1 by default for NORMAL and COMPLETION engines (`arg_utils.py:51-55`,
  `engine_type.py:15-17`). README: "Enabled by default."
- `worker/worker.py:89-189` `execute_model_step`: 1. forward current step, 2. post-process previous step, 3. sample
  current step, 4. post-process now for prompt/non-ZOS steps, defer for ZOS decode steps. Seqlens on CPU lag one step
  and are corrected on GPU (`_correct_seqlens_qk`, `:191-245`; `layers/tilelang_ops/compute_seqlens_k_zos.py`).
- Q&A lines 326-334: "Following existing implementations (SGLang, NanoFlow)". 
- vLLM: `vllm/v1/core/sched/async_scheduler.py` (`AsyncScheduler(Scheduler)`), `SchedulerConfig.async_scheduling`
  (`vllm/config/scheduler.py:211`). SGLang: overlap scheduler, `python/sglang/srt/managers/overlap_utils.py`, flag
  `disable_overlap_schedule` (`managers/scheduler_components/dp_attn.py:754`).

## 7. KV cache management

- Block = 128 tokens, fixed: `model_metadata_utils.py:133-135` `resolved_tokens_per_block -> 128`. Block 0 is
  reserved; free list is a doubly linked list with ref counts (`csrc/engine/block_manager.hpp:27-117`), `pin_block`
  for sharing (fork).
- Two pools (`worker/cache_engine.py:205-218`, `:221-306`): (a) the paged block pool, one 128-token block holds every
  layer's compressed main KV (`ca_cache`) and indexer K (`ica_cache`) slices; (b) a state pool with one fixed-size
  slot per sequence (`StateBlock`) holding SWA rings and fp32 compression accumulators. `num_state_slots =
  max_num_seqs * 1.2 + 1` (`:201-202`, `environ.py:169`); state reserved x2 (store + copy buffer,
  `worker/base.py:241-244`).
- Formats (`layers/tilelang_ops/cache_common.py:109-125`): V41_FORMAT 512 e4m3 + UE8M0 per 32 = 528 B/entry (SWA);
  V41_FP4_FORMAT 512 e2m1 + scale per 16 = 288 B/entry (main compressed KV, `use_fp4_sparse_kv` default True,
  `hugging_face/config.py:303`); indexer FP4 128/2 + 128/32 = 68 B (`cache_engine.py:73-80`).
- Layer sharing (`hugging_face/config.py:62-85`): `kv_source_layer_ids [2, 8, 14, 20]` -> only 4 layers own main KV and
  indexer K; others reuse via `layer_kv_map`. Compression ratios from config: layers 0-1 SWA only, 2-19 ratio 2,
  20-39 ratio 1. `n_prefill_layer` = first ratio-1 layer = 20 (`config.py:158-161`): layers 20-39 are YOCO consumer
  (decoder) layers.
- part9_kv_calc result: block = 3 x (64 x 288 + 64 x 68) + (128 x 288 + 128 x 68) = 113,920 B per 128 tokens =
  890.0 B/token (891.0 after 1,152-byte alignment). This equals the V4.1 tech report's "890 bytes per token" (dossier
  04). State slot: 43 SWA rings (40 layers + 3 DSpark layers) x 134 tokens x 528 B + 96,768 B compression state =
  3,139,488 B (2.99 MiB) per sequence; stored part 20 x 134 x 528 = 1,415,040 B, equal to Q&A line 223.
- Prefix caching: no hash or radix prefix cache in the engine. `cache.hit_tokens` and `fork_seq` (`scheduler.hpp:1061-1121`)
  let a caller fork a computed prefix, pinning blocks and cloning the state slot and the partial last block (used by
  `examples/benchmark.py:337`). Production prefix reuse is the external kvcache-store on 3FS (Q&A 163-245): hash-chain
  prefix match over metadata by the Entrypoint; block KV on 3FS with 10+2 erasure coding; SWA KV kept in host memory
  of the producing machine; TTL 3 days agent / 7 days chatbot; ~100 PiB SSD; 14.4 TB logical per GPU.
- SWA bounded replay (Q&A 235-245, README 23): replay only the last 128 tokens; hit length capped at L - 134.
  Decoder variant: `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, `KEEP_TOKENS=140` (README; env default "off").
  Q&A: "nearly halving total prefill computation" (claim).
- vLLM v0.33: block hashing (`vllm/v1/core/kv_cache_utils.py`, xxhash/sha256 options, `NONE_HASH`), prefix caching on by
  default (`vllm/config/cache.py:257` `enable_prefix_caching: bool = True`), hybrid managers per attention type
  (`vllm/v1/core/single_type_kv_cache_manager.py`: `FullAttentionManager` :739, `SlidingWindowManager` :946,
  `HiSparse*` :2223-2472), V4.1 model in `vllm/models/deepseek_v41/` (nvidia/flashmla.py, nvidia/dspark.py,
  common/engram.py, sparse_mla.py, `decoder_replay_layers.py`), sparse SWA backend
  `vllm/v1/attention/backends/mla/sparse_swa.py`. Dossier 04 recipe uses `--block-size 256` for V4-Pro.
- SGLang v0.5.23: `RadixCache` (`python/sglang/srt/mem_cache/radix_cache.py:281`), `unified_radix_cache.py`,
  `pure_swa_radix_cache.py`, V4 pools `deepseek_v4_memory_pool.py`, `deepseek_v4_compress_state.py`,
  `dsv41_request_window.py`; V4.1 config `configs/deepseek_v41.py`; flag `--enable-encoder-swa-bounded-replay`
  (`arg_groups/deepseek_v4_hook.py:300-330`, with its incompatibility list including "mixed prefill/decode" and "PD
  decode"). Both vLLM (`distributed/kv_transfer/kv_connector/v1/hf3fs/`) and SGLang (`mem_cache/storage/hf3fs/`) ship
  3FS-backed KV storage.

## 8. PD disaggregation

- DeepInfer: engine roles exist (PROMPT/COMPLETION), hooks empty; README says not included. Q&A: E/P/D with
  prefill EP2-8/DP2-8 on 8-GPU machines, decode EP64-128/DP64-128 on supernodes; "1P + 1D is deployable"; smallest
  efficient unit 64 + 64 GPUs; ~400 Gbps per 8 GPUs across data centres; SWA KV and block KV share the P-to-D path.
- vLLM: `vllm/distributed/kv_transfer/kv_connector/v1/` with `nixl/`, `mooncake/`, `lmcache_connector.py`,
  `hf3fs/`, `multi_connector.py`, `offloading_connector.py`, etc.
- SGLang: `python/sglang/srt/disaggregation/` with `prefill.py`, `decode.py`, backends `mooncake/`, `nixl/`, `mori/`,
  `ascend/`, plus `encoder/`.

## 9. Parallelism and MoE

- DP attention + EP for MoE, no TP (Q&A 53: "TP and CP are not used for now" in production; code supports CP).
- CP "mirror" balancing: sequences split into `2*cp` pieces (`scheduler.hpp:90` `cp_align`,
  `csrc/engine/input_metadata/prepare_tensor_input_base.cpp:145-155`, `layers/tilelang_ops/prepare_cp_mirror.py`),
  all-gather via DeepEP `BucketBuffer` (`parallel_utils/cp.py`).
- MoE (`models/v4_modules/mlp.py`): gate via TileKernels `moe_topk_gate`; Mega MoE (`deep_gemm.fp8_fp4_mega_moe` /
  `bf16_mega_moe`) whenever the EP group is inside one NVLink domain (`:184-188`, `:528-563`); otherwise DeepEP
  `EPBuffer.dispatch/combine` + DeepGEMM grouped GEMM (`:565-616`). Weights: MXFP4 experts with FP8 activations
  (`mma_type` "fp8xfp4", `:177-182`).
- EPLB (`model_executor/eplb/policy.py:73-140`): greedy replica allocation by per-replica load, then greedy packing that
  minimises the sum over steps of the max rank load (`_batch_greedy_indices`, `:12-70`); weights shuffled in place over
  symmetric memory / IPC intranode or P2P (`eplb/shuffle.py`). Knobs: `DEEPINFER_OVERRIDE_NUM_EXTRA_EXPERTS` (examples
  default 32, rounded up to divide EP), `EXPERT_COUNT_WINDOW_SIZE` default 100, `EXPERT_REARRANGEMENT_STEP_INTERVAL`
  default 0 (examples set 30). Q&A production: prefill 4 redundant/GPU, window 100, every 1,000 steps, moe_lb 1.074
  (EP4DP4) / 1.045 (EP2DP2); decode 1 redundant/GPU (7 experts/GPU at EP64), window 1,000, every 10,000 steps, moe_lb
  1.090.
- vLLM: EPLB `vllm/distributed/eplb/` (`policy/default.py` docstring: "adapted from DeepSeek EPLB"),
  `ParallelConfig.enable_eplb` False, `num_redundant_experts` 0 (`vllm/config/parallel.py:77,206`); DeepEP / DeepGEMM
  references in `vllm/model_executor/layers/fused_moe/`. SGLang: `python/sglang/srt/eplb/` (`eplb_manager.py`,
  `eplb_algorithms/`, `expert_location.py`), `elastic_ep/`.

## 10. Speculative decoding (DSpark)

- Draft model: `models/mtp/dflash.py` (internal name DFlash; config keys `dspark_*`, `hugging_face/config.py:177-192`),
  `ConfidenceHead` "Predict the acceptance probability of each draft token" (`dflash.py:46-70`). V4.1 config:
  `num_nextn_predict_layers 3`, `dspark_block_size 5`, `dspark_markov_rank 256`.
- `DEEPINFER_USE_NEXTN_DECODE` default 5 (`environ.py:136`); `DEEPINFER_ENABLE_MTP_VARLEN` default off (README says set
  it to 1).
- Variable-length verification: `csrc/engine/mtp_varlen.hpp:150-267` `MTPVarlen::schedule`: sort all draft tokens of
  the batch by estimated accept rate, walk a measured decode performance curve (`_decode_perf_curve`, batch size ->
  ms) plus a fitted slope and median sampler time, pick the total query count with the best expected tokens per ms,
  then give the extra verify slots to the highest-rate drafts. Under ZOS the per-request split is recomputed on GPU
  (`layers/tilelang_ops/schedule_mtp_varlen.py`).
- Q&A table (lines 307-316, Open-PerfectBlend, EP64/DP64 data, estimated throughput): at 60 seqs/rank, verified 5 ->
  3.58, accepted 3.30 -> 3.21, 13,730 -> 14,466 tok/s/rank (+5.4%); at 16 seqs/rank -1.5%; at 96 +11.0%.
- vLLM: `vllm/v1/worker/gpu/spec_decode/dspark/` (speculator.py), `vllm/models/deepseek_v41/nvidia/dspark.py`, registry
  `"DSparkDraftModel"`. SGLang: `SpeculativeAlgorithm.DSPARK` (`speculative/spec_info.py:41`),
  `models/deepseek_v4_dspark.py`, `speculative/dspark_components/`, `dspark_disaggregation.py`. Whether vLLM/SGLang
  implement confidence-scheduled variable-length verification: not checked line by line (gap; say "I did not verify").

## 11. Kernels and libraries actually imported

`flash_mla` (attention_backend.py:2-3: `flash_mla_sparse_fwd`, `flash_mla_with_kvcache`,
`fused_norm_rope_attn_rope_cast`), `deep_gemm` (Mega MoE, grouped FP8/FP4 GEMM, paged MQA logits for the indexer,
`tf32_hc_prenorm_gemm` for mHC), `deep_ep` (`EPBuffer`, `EngramBuffer`, `BucketBuffer`), `deep_select.interface.topk`
(index_attn.py, indexer top-k), `tile_kernels` (moe gate, quant, mHC kernels, ViT ops), `tilelang` 0.1.15 for the
in-tree kernels, `flash_attn.cute` for the ViT, `xgrammar` 0.2.2 for structured output. DeepJIT: not referenced
anywhere in the repo (grep returns nothing). Build (`scripts/build_vendor.sh`): NVIDIA builds DeepSelect, FlashMLA,
DeepEP, DeepGEMM; Ascend builds DeepSelect, FlashMLA, DeepEP-Ascend, DeepGEMM-Ascend.

## 12. CUDA graphs

Decode only (`worker/model_runner.py:73` disables for PROMPT engines; `:182` `run_cuda_graph = not is_prompt and ...`);
batch sizes 1, 2, 4, then every 8 up to the max (`worker/cuda_graph_runner.py:41-51`, `DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL`
8). Separate DFlash graph runner. `DEEPINFER_ENFORCE_EAGER` to disable.

## 13. Published numbers (all DeepSeek's own; conditions)

- README charts (`doc/images/*.png`), "Measured on 4 SM100 GPUs for prefill and 64 for decode" within one scale-up
  domain. Prefill chart subtitle: "EP4DP4 | 384+32 experts | context 131,072 | Decoder Bounded Replay (140 tokens)";
  points labelled 4,096 to 65,536 tokens per DP rank; read from the chart: about 60k tok/s/GPU at about 70 ms per step
  (4,096) up to about 89k tok/s/GPU at about 730 ms (65,536). Decode chart: "EP64DP64 | 384+64 experts | context
  131,072 | nextn=3 fixed | per-token acceptance 75% -> tau=3.25"; 8 to 256 requests per rank; read from the chart:
  about 26k tok/s/GPU at about 100 tok/s/user (256) to about 2.6k tok/s/GPU at about 325 tok/s/user (8). Values
  read off PNG charts are approximate.
- Important caveat: decode throughput is computed, not counted: `plot_pareto.py:58-67` multiplies measured steps/s by
  tau = 1 + nextn x acceptance, with acceptance an input (0.75); `examples/README.md:55` "under full-acceptance
  accounting".
- Wide-EP chart: EP8DP8 (384+32) vs EP64DP64 (384+64); at 256 req/rank about 25k vs 26k tok/s/GPU; EP64 gives higher
  per-user speed at every batch size.
- Q&A: 155 trillion tokens/day per cluster, US$38/GPU-hour revenue at off-peak price; load balancing reaches "roughly
  80% of a perfectly balanced offline benchmark"; CP up to 5% slower than DP end to end; internal FlashMLA 5-10%
  faster decode kernel, end-to-end within 5%.
- No GPU was available to us; we ran nothing.

## 14. Comparison summary (for the big table)

| Point | DeepInfer a679bb8 | vLLM v0.33.0 | SGLang v0.5.23 |
|---|---|---|---|
| Models | V4.1-Flash only | hundreds of architectures incl. DeepSeek V2-V4.1 (`vllm/model_executor/models/registry.py`, `vllm/models/deepseek_v41`) | many incl. DeepSeek V2-V4.1 (`srt/models/`, `configs/deepseek_v41.py`) |
| Hardware | SM100 + Ascend 950 | in-tree platforms cuda, rocm, tpu, xpu, cpu (`vllm/platforms/`) | cuda, rocm, npu, xpu, musa, mps, cpu (`srt/platforms/`) |
| API | offline Python `LLM.generate` | OpenAI-compatible server (`vllm/entrypoints/openai/`) + offline | OpenAI-compatible server (`srt/entrypoints/openai/`), also anthropic, ollama, gRPC dirs + offline Engine |
| Scheduler | C++; pure prefill or pure decode steps, DP vote; FIFO | Python; token budget, mixed prefill+decode, chunked prefill on by default; fcfs/priority | Python; separate prefill (extend) and decode batches, optional mixed chunk; lpm, dfs-weight, fcfs, lof policies |
| KV block | 128 tokens + per-seq state slot | configurable block, hybrid managers | page size configurable, token-level radix |
| Prefix cache | none in engine (external 3FS store in production) | hashed blocks, on by default | radix tree |
| PD | roles + hooks, transfer not released | KV connectors (NIXL, Mooncake, LMCache, 3FS...) | disaggregation module (Mooncake, NIXL, Mori, Ascend) |
| EP/EPLB | DP attention + EP, Mega MoE, EPLB built in | EP + EPLB (off by default) | EP, DP attention, EPLB |
| Spec decode | DSpark with confidence-scheduled varlen verify | DSpark, EAGLE, n-gram, DFlash ... | DSPARK, EAGLE, DFlash, n-gram ... |
| Licence | Apache-2.0 | Apache-2.0 | Apache-2.0 |
| Maturity | v0.1.0, Beta, released Oct 2026, production-proven inside DeepSeek per README | v0.33 | v0.5.23 |

UNVERIFIED / not checked: default values of SGLang's server args (defaults live in arg-group modules; not traced);
whether vLLM/SGLang V4.1 paths implement DeepInfer's confidence-scheduled varlen verification; performance of any
engine on V4.1 (no shared benchmark exists); Ascend performance of DeepInfer (no published numbers).
