# 02. DeepSeek LLM architecture, from 67B to V4.1 (research dossier)

Research date: 2026-11-10. Scope: the architecture of every major DeepSeek language model. Every number has a source. Paper page numbers are 1-based PDF page numbers (the page index in the PDF file, which for these papers equals the printed page number unless noted). Quotes were extracted from the PDFs with PyMuPDF and are copied verbatim, except that math symbols that PyMuPDF renders as Unicode italics are written here in plain notation inside `code` or LaTeX-style text.

Status markers:
- **VERIFIED**: read in the primary source (config.json, paper PDF, official README or official code).
- **DERIVED**: my own arithmetic from verified inputs. The arithmetic is shown.
- **UNVERIFIED**: could not be confirmed in a primary source.

## 0. Sources used

### Papers (all downloaded to `~/.cache/papers/<id>.pdf`)

| arXiv id | Title | Pages | arXiv submission date |
|---|---|---|---|
| 2401.02954 | DeepSeek LLM: Scaling Open-Source Language Models with Longtermism | 48 | Jan 2024 |
| 2401.06066 | DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models | 33 | Jan 2024 |
| 2405.04434 | DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model | 52 | May 2024 |
| 2412.19437 | DeepSeek-V3 Technical Report | 53 | Dec 2024 |
| 2502.11089 | Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention | 25 | Feb 2025 |
| 2505.09343 | Insights into DeepSeek-V3: Scaling Challenges and Reflections on Hardware for AI Architectures (ISCA 2025) | | May 2025 |
| 2512.02556 | DeepSeek-V3.2: Pushing the Frontier of Open Large Language Models | 23 | 2 Dec 2025 |
| 2512.24880 | mHC: Manifold-Constrained Hyper-Connections | 19 | 31 Dec 2025 (v2 5 Jan 2026) |
| 2601.07372 | Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models (Engram) | 35 | 12 Jan 2026 |
| 2606.19348 | DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | 58 | 26 Apr 2026 (see note) |
| 2607.05147 | DSpark: Confidence-Scheduled Speculative Decoding with Semi-Autoregressive Generation | | 6 Jul 2026 |
| 2609.19969 | DeepSeek-V4.1-Flash: Pushing the Limits of KV Cache Compression | 51 | 17 Sep 2026 |

Note on 2606.19348: the arXiv id prefix suggests June 2026, but both the abs page (https://arxiv.org/abs/2606.19348, "Submitted on 26 Apr 2026") and the PDF margin stamp ("arXiv:2606.19348v1 [cs.CL] 26 Apr 2026", PDF p.1) say 26 April 2026. The HF repos for V4 were created 2026-04-22. Writers should cite the date as April 2026 and not try to explain the id.

### Configs (raw `config.json`, fetched 2026-11-10)

All URLs have the form `https://huggingface.co/deepseek-ai/<repo>/raw/main/config.json`:

- deepseek-llm-67b-base, deepseek-moe-16b-base, DeepSeek-V2, DeepSeek-V2-Lite, DeepSeek-V3, DeepSeek-V3-Base, DeepSeek-R1, DeepSeek-V3.1, DeepSeek-V3.1-Terminus, DeepSeek-V3.2-Exp, DeepSeek-V3.2, DeepSeek-V3.2-Speciale, DeepSeek-V4-Flash, DeepSeek-V4-Flash-Base, DeepSeek-V4-Pro, DeepSeek-V4-Pro-Base, DeepSeek-V4-Flash-0731, DeepSeek-V4-Pro-0813, DeepSeek-V4.1-Flash, DeepSeek-V4.1-Flash-1017.

The full list of deepseek-ai repos (sorted by creation date) came from `https://huggingface.co/api/models?author=deepseek-ai&sort=createdAt&direction=-1`.

### Code

- DeepSeek-V3 reference inference: https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py
- DeepSeek-V3 weight doc: https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/README_WEIGHTS.md
- DeepSeek-V3 safetensors index: https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/model.safetensors.index.json
- DeepSeek-V3.2-Exp inference: https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/inference/model.py
- DeepSeek-V4-Pro inference: https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/inference/model.py
- FlashMLA README, KV cache byte layouts (pre-2026.09.30 commit): https://github.com/deepseek-ai/FlashMLA/tree/ba89a3466e9470ad08ab39738d4e7bb66989e1e7

---

## 1. Configurations

### 1.1 Comparison table (raw values from config.json)

`-` means the field is absent from that config. R1 has the same config as V3 (a `diff` of the two files differs only in `transformers_version`). V3.2 and V3.2-Exp configs are identical. V3.2-Speciale is identical to V3.2.

| Field | 67B | MoE 16B | V2 | V2-Lite | V3 / R1 | V3.1 | V3.2-Exp / V3.2 | V4-Flash | V4-Pro | V4.1-Flash (`text_config`) |
|---|---|---|---|---|---|---|---|---|---|---|
| architectures | LlamaForCausalLM | DeepseekForCausalLM | DeepseekV2ForCausalLM | DeepseekV2ForCausalLM | DeepseekV3ForCausalLM | DeepseekV3ForCausalLM | DeepseekV32ForCausalLM | DeepseekV4ForCausalLM | DeepseekV4ForCausalLM | DeepseekV41ForCausalLM |
| num_hidden_layers | 95 | 28 | 60 | 27 | 61 | 61 | 61 | 43 | 61 | 40 |
| hidden_size | 8192 | 2048 | 5120 | 2048 | 7168 | 7168 | 7168 | 4096 | 7168 | 5120 |
| num_attention_heads | 64 | 16 | 128 | 16 | 128 | 128 | 128 | 64 | 128 | 64 |
| num_key_value_heads | 8 | 16 | 128 | 16 | 128 | 128 | 128 | 1 | 1 | 1 |
| head_dim | - | - | - | - | - | - | - | 512 | 512 | 512 |
| q_lora_rank | - | - | 1536 | null | 1536 | 1536 | 1536 | 1024 | 1536 | 1280 |
| kv_lora_rank | - | - | 512 | 512 | 512 | 512 | 512 | - | - | - |
| qk_rope_head_dim | - | - | 64 | 64 | 64 | 64 | 64 | 64 | 64 | 64 |
| qk_nope_head_dim | - | - | 128 | 128 | 128 | 128 | 128 | - | - | - |
| v_head_dim | - | - | 128 | 128 | 128 | 128 | 128 | - | - | - |
| o_lora_rank / o_groups | - | - | - | - | - | - | - | 1024 / 8 | 1024 / 16 | 1024 / 8 |
| vocab_size | 102400 | 102400 | 102400 | 102400 | 129280 | 129280 | 129280 | 129280 | 129280 | 129280 |
| intermediate_size (dense FFN) | 22016 | 10944 | 12288 | 10944 | 18432 | 18432 | 18432 | - | - | - |
| moe_intermediate_size | - | 1408 | 1536 | 1408 | 2048 | 2048 | 2048 | 2048 | 3072 | 2304 |
| n_routed_experts | - | 64 | 160 | 64 | 256 | 256 | 256 | 256 | 384 | 384 |
| n_shared_experts | - | 2 | 2 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| num_experts_per_tok | - | 6 | 6 | 6 | 8 | 8 | 8 | 6 | 6 | 6 |
| first_k_dense_replace | - | 1 | 1 | 1 | 3 | 3 | 3 | - | - | - |
| num_hash_layers | - | - | - | - | - | - | - | 3 | 3 | - |
| n_group / topk_group | - | - | 8 / 3 | 1 / 1 | 8 / 4 | 8 / 4 | 8 / 4 | - | - | - |
| topk_method | - | - | group_limited_greedy | greedy | noaux_tc | noaux_tc | noaux_tc | noaux_tc | noaux_tc | noaux_tc |
| scoring_func | - | softmax | softmax | softmax | sigmoid | sigmoid | sigmoid | sqrtsoftplus | sqrtsoftplus | sqrtsoftplus |
| routed_scaling_factor | - | - | 16.0 | 1.0 | 2.5 | 2.5 | 2.5 | 1.5 | 2.5 | 1.5 |
| norm_topk_prob | - | false | false | false | true | true | true | true | true | true |
| aux_loss_alpha / seq_aux | - | - | 0.001 / true | 0.001 / true | - | - | - | - | - | - |
| max_position_embeddings | 4096 | 4096 | 163840 | 163840 | 163840 | 163840 | 163840 | 1048576 | 1048576 | 1048576 |
| rope_theta | 10000.0 | 10000 | 10000 | 10000 | 10000 | 10000 | 10000 | 10000 (compress_rope_theta 160000) | 10000 (compress_rope_theta 160000) | 10000 (compress_rope_theta 160000) |
| rope_scaling | null | null | yarn, factor 40, orig 4096, mscale 0.707 | yarn, factor 40, orig 4096, mscale 0.707 | yarn, factor 40, orig 4096, mscale 1.0 | same as V3 | same as V3 | yarn, factor 16, orig 65536 | yarn, factor 16, orig 65536 | yarn, factor 16, orig 65536 |
| torch_dtype | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 | bfloat16 (`dtype`) |
| quantization_config | - | - | - | - | fp8 e4m3, block [128,128], dynamic | fp8 e4m3, block [128,128], scale_fmt ue8m0 | fp8 e4m3, block [128,128], ue8m0 | fp8 e4m3 [128,128] ue8m0, expert_dtype fp4 | fp8 e4m3 [128,128] ue8m0, expert_dtype fp4 | fp8, block [32,32], ue8m0, expert_dtype fp4 |
| num_nextn_predict_layers (MTP) | - | - | - | - | 1 | 1 | 1 | 1 | 1 | 3 (DSpark drafter blocks; see 6.6) |
| index_n_heads / index_head_dim / index_topk | - | - | - | - | - | - | 64 / 128 / 2048 | 64 / 128 / 512 | 64 / 128 / 1024 | 32 / 128 / 512 |
| sliding_window | - | - | - | - | - | - | - | 128 | 128 | 128 |
| hc_mult / hc_sinkhorn_iters (mHC) | - | - | - | - | - | - | - | 4 / 20 | 4 / 20 | 4 / 20 |
| swiglu_limit | - | - | - | - | - | - | - | 10.0 | 10.0 | 10.0 |

Published totals (from papers and READMEs, not configs):

| Model | Total params | Active params | Context | Source |
|---|---|---|---|---|
| DeepSeek LLM 67B | 67B (dense) | 67B | 4K | 2401.02954 p.5 Table 2 |
| DeepSeekMoE 16B | 16.4B | 2.8B | 4K | 2401.06066 p.15 |
| DeepSeek-V2 | 236B | 21B | 128K | 2405.04434 p.1, p.12 |
| DeepSeek-V2-Lite | 15.7B | 2.4B | 32K | 2405.04434 p.29 |
| DeepSeek-V3 / R1 / V3.1 / V3.2 | 671B (+14B MTP on HF = 685B) | 37B | 128K | 2412.19437 p.22; V3 README; V3.1 README |
| DeepSeek-V4-Flash | 284B | 13B | 1M | 2606.19348 p.1, p.24 |
| DeepSeek-V4-Pro | 1.6T | 49B | 1M | 2606.19348 p.1, p.25 |
| DeepSeek-V4.1-Flash | 552B backbone + 196B Engram | 8B prefill / 16B decode | 1M | 2609.19969 p.1, p.7, p.22 |

### 1.2 Per-model notes

**DeepSeek LLM 67B** (https://huggingface.co/deepseek-ai/deepseek-llm-67b-base/raw/main/config.json). LLaMA-style dense model. `"num_hidden_layers": 95`, `"hidden_size": 8192`, `"num_attention_heads": 64`, `"num_key_value_heads": 8` (GQA with 8 KV heads), `"intermediate_size": 22016`, `"max_position_embeddings": 4096`, `"vocab_size": 102400`. Paper quote (2401.02954, PDF p.5, Section 2.2):
> "To optimize inference cost, the 67B model uses Grouped-Query Attention (GQA) (Ainslie et al., 2023) instead of the traditional Multi-Head Attention (MHA)."
> "Unlike most works using Grouped-Query Attention (GQA), we expanded the 67B model's parameters in network depth rather than the common practice of widening the intermediate width of FFN layers, aiming for better performance."

The V4.1 paper calls this generation "DeepSeek-V1" in its KV cache chart (2609.19969 p.1, Figure 1(b), 389,120 bytes per token, dated 2023.11).

**DeepSeekMoE 16B** (https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/raw/main/config.json). Uses standard MHA (16 heads = 16 KV heads). `"first_k_dense_replace": 1`, `"n_routed_experts": 64`, `"n_shared_experts": 2`, `"num_experts_per_tok": 6`, `"moe_intermediate_size": 1408`, `"scoring_func": "softmax"`. Paper (2401.06066 p.15): "Each MoE layer consists of 2 shared experts and 64 routed experts, where each expert is 0.25 times the size of a standard FFN. Each token will be routed to these 2 shared experts and 6 out of 64 routed experts." and "DeepSeekMoE 16B has approximately 16.4B total parameters, with the number of activated parameters around 2.8B."

**DeepSeek-V2** (https://huggingface.co/deepseek-ai/DeepSeek-V2/raw/main/config.json). First MLA model. `"kv_lora_rank": 512`, `"q_lora_rank": 1536`, `"qk_rope_head_dim": 64`, `"qk_nope_head_dim": 128`, `"v_head_dim": 128`, `"n_routed_experts": 160`, `"n_shared_experts": 2`, `"num_experts_per_tok": 6`, `"n_group": 8`, `"topk_group": 3`, `"topk_method": "group_limited_greedy"` (this is the device-limited routing with D = 8 devices, M = 3), `"routed_scaling_factor": 16.0`. Context: `"max_position_embeddings": 163840` = YaRN factor 40 x original 4096; the paper states a 128K context (p.1). The config field `num_key_value_heads: 128` is a leftover from the HF template; MLA has no KV heads in the GQA sense.

**DeepSeek-V2-Lite** (https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/raw/main/config.json). `"q_lora_rank": null`: no query compression. Paper (2405.04434 p.29): "Its KV compression dimension is 512, but slightly different from DeepSeek-V2, it does not compress the queries." `"n_group": 1`, `"topk_method": "greedy"` (no device limit). Paper context: 32K (p.29 table), although the config says 163840.

**DeepSeek-V3 / R1** (https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json, https://huggingface.co/deepseek-ai/DeepSeek-R1/raw/main/config.json). Same MLA ranks as V2, but 61 layers, d = 7168, `"first_k_dense_replace": 3`, `"n_routed_experts": 256`, `"n_shared_experts": 1`, `"num_experts_per_tok": 8`, `"n_group": 8`, `"topk_group": 4` (node-limited routing, M = 4 nodes), `"scoring_func": "sigmoid"`, `"topk_method": "noaux_tc"` (auxiliary-loss-free, bias term), `"routed_scaling_factor": 2.5`, `"num_nextn_predict_layers": 1`, `"vocab_size": 129280`, FP8 block quantization `"weight_block_size": [128, 128]`. R1 is a post-trained V3-Base; the architecture is identical.

**DeepSeek-V3.1 and V3.1-Terminus** (https://huggingface.co/deepseek-ai/DeepSeek-V3.1/raw/main/config.json). Architecture identical to V3. The only config change is `"scale_fmt": "ue8m0"`. README (https://huggingface.co/deepseek-ai/DeepSeek-V3.1/raw/main/README.md): "DeepSeek-V3.1 is trained using the **UE8M0 FP8 scale data format on both model weights and activations**" and "The 32K extension phase has been increased 10-fold to 630B tokens, while the 128K extension phase has been extended by 3.3x to 209B tokens." The Terminus config differs from V3.1 only by an empty `"attn_module_list_cfg": []` and no `scale_fmt` key.

**DeepSeek-V3.2-Exp and V3.2** (https://huggingface.co/deepseek-ai/DeepSeek-V3.2-Exp/raw/main/config.json, https://huggingface.co/deepseek-ai/DeepSeek-V3.2/raw/main/config.json). V3 plus the DSA lightning indexer: `"index_n_heads": 64`, `"index_head_dim": 128`, `"index_topk": 2048`, `"model_type": "deepseek_v32"`. Paper (2512.02556 p.3): "DeepSeek-V3.2 uses exactly the same architecture as DeepSeek-V3.2-Exp. Compared with DeepSeek-V3.1-Terminus, the last version of DeepSeek-V3.1, the only architectural modification of DeepSeek-V3.2 is the introduction of DeepSeek Sparse Attention (DSA) through continued training."

**DeepSeek-V4-Flash and V4-Pro** (https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash/raw/main/config.json, https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro/raw/main/config.json). MLA is gone. `"num_key_value_heads": 1` and `"head_dim": 512`: a single shared KV entry (MQA-style) of width 512 per compressed position, with partial RoPE on the last 64 dims. New fields: `"compress_ratios"` per layer (0 = sliding window only, 4 = CSA, 128 = HCA; the list has one more entry than the layer count, for the MTP layer), `"sliding_window": 128`, `"o_lora_rank": 1024` and `"o_groups"` (grouped output projection), `"hc_mult": 4`, `"hc_sinkhorn_iters": 20`, `"hc_eps": 1e-06` (mHC), `"num_hash_layers": 3` (hash-routed first MoE layers), `"scoring_func": "sqrtsoftplus"`, `"swiglu_limit": 10.0`, `"expert_dtype": "fp4"`. There is no `n_group`/`topk_group`: V4 removes node-limited routing (see 6.3). Flash `compress_ratios` = `[0, 0, 4, 128, 4, 128, ..., 4, 0]` (44 entries); Pro = `[128, 128, 4, 128, ..., 4, 0]` (62 entries).

**DeepSeek-V4-Flash-0731 and V4-Pro-0813** (official releases that supersede the April preview). Same backbone config plus DSpark fields: Pro-0813 adds `"dspark_block_size": 5`, `"dspark_noise_token_id": 128799`, `"dspark_target_layer_ids": [58, 59, 60]`, `"dspark_markov_rank": 512`, and `compress_ratios` gains three trailing zeros. README (https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro-0813/raw/main/README.md): "It is built on the DeepSeek-V4-Pro (Preview) model structure, with a DSpark speculative decoding module attached."

**DeepSeek-V4.1-Flash** (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/raw/main/config.json; the -1017 config is byte-identical). Multimodal: top-level `text_config` plus `vision_config` (32 layers, hidden 1024, 16 heads, patch 14). Text backbone: 40 layers, d = 5120, 384 routed experts of width 2304, top-6, `"q_lora_rank": 1280`. New fields: `"compress_ratios"` (2 for encoder CSA2 layers, 1 for decoder layers), `"kv_source_layer_ids": [2, 8, 14, 20]`, `"index_source_layer_ids": [2, 8, 14, 20, 24, 28, 32, 36]`, `"candidate_source_layer_id": 20`, `"candidate_topk_blocks": 2048`, `"candidate_block_size": 8`, `"engram_layer_ids": [1, 14]`, `"engram_max_ngram_size": 4`, `"engram_n_heads": 8`, `"engram_head_dim": 256`, `"engram_num_embeddings": [384006168, 384016682]`, `"dspark_n_routed_experts": 128`, `"dspark_num_experts_per_tok": 3`, `"rms_norm_eps": 1e-20`, and FP8 `"weight_block_size": [32, 32]`.

---

## 2. DeepSeekMoE (arXiv 2401.06066)

### 2.1 Fine-grained expert segmentation (PDF p.5 to p.6, Section 3.1)

Key quote (p.5):
> "we segment each expert FFN into m smaller experts by reducing the FFN intermediate hidden dimension to 1/m times its original size. Since each expert becomes smaller, in response, we also increase the number of activated experts to m times to keep the same computation cost, as illustrated in Figure 2(b)."

Equations (p.6, Eq. 6 to 8):

```
h_t^l = sum_{i=1}^{mN} g_{i,t} FFN_i(u_t^l) + u_t^l                         (6)
g_{i,t} = s_{i,t} if s_{i,t} in Topk({s_{j,t} | 1 <= j <= mN}, mK), else 0  (7)
s_{i,t} = Softmax_i(u_t^l^T e_i^l)                                         (8)
```

The combinatorics argument (p.6): "we consider the case where N = 16. A typical top-2 routing strategy can yield C(16,2) = 120 possible combinations. By contrast, if each expert is split into 4 smaller experts, the fine-grained routing strategy can yield C(64,8) = 4,426,165,368 potential combinations."

Figure 2 (p.5) is the standard screenshot: (a) conventional top-2, (b) fine-grained segmentation, (c) plus shared expert isolation. Caption: "It is noteworthy that across these three architectures, the number of expert parameters and computational costs remain constant."

### 2.2 Shared expert isolation (PDF p.6, Section 3.2)

Quote (p.6):
> "Towards this objective, in addition to the fine-grained expert segmentation strategy, we further isolate K_s experts to serve as shared experts. Regardless of the router module, each token will be deterministically assigned to these shared experts. In order to maintain a constant computational cost, the number of activated experts among the other routed experts will be decreased by K_s, as depicted in Figure 2(c)."

Equations (p.6, Eq. 9 to 11):

```
h_t^l = sum_{i=1}^{K_s} FFN_i(u_t^l) + sum_{i=K_s+1}^{mN} g_{i,t} FFN_i(u_t^l) + u_t^l     (9)
g_{i,t} = s_{i,t} if s_{i,t} in Topk({s_{j,t} | K_s+1 <= j <= mN}, mK - K_s), else 0       (10)
s_{i,t} = Softmax_i(u_t^l^T e_i^l)                                                        (11)
```

Credit note (p.6): "the prototype of shared expert isolation can be credited to Rajbhandari et al. (2022). The key distinction lies in the fact that they derive this strategy from an engineering perspective, while we approach it from an algorithmic standpoint."

### 2.3 Load-balance losses (PDF p.7, Section 3.3)

Expert-level balance loss (Eq. 12 to 14):

```
L_ExpBal = alpha_1 * sum_{i=1}^{N'} f_i P_i                       (12)
f_i = N' / (K' T) * sum_{t=1}^{T} 1(Token t selects Expert i)     (13)
P_i = 1/T * sum_{t=1}^{T} s_{i,t}                                 (14)
```
with N' = mN - K_s and K' = mK - K_s.

Device-level balance loss (Eq. 15 to 17), experts partitioned into D groups E_1..E_D, one per device:

```
L_DevBal = alpha_2 * sum_{i=1}^{D} f'_i P'_i     (15)
f'_i = 1/|E_i| * sum_{j in E_i} f_j              (16)
P'_i = sum_{j in E_i} P_j                        (17)
```

Quote (p.7): "In practice, we set a small expert-level balance factor to mitigate the risk of routing collapse, and meanwhile set a larger device-level balance factor to promote balanced computation across the devices."

Validation setting (p.9): "In order to prevent routing collapse, we set an expert-level balance factor of 0.01."

### 2.4 Headline results

- 2B validation model (p.9): "DeepSeekMoE has 1 shared expert and 63 routed experts, where each expert is 0.25 times the size of a standard FFN."
- 16B (p.17, Table 3): DeepSeek 7B dense: 6.9B params, 183.5T FLOPs per 4K tokens; DeepSeekMoE 16B: 16.4B total, 2.8B active, 74.4T FLOPs per 4K tokens. Abstract (p.1): "achieves comparable performance with LLaMA2 7B, with only about 40% of computations."
- 145B (p.1): "comparable with DeepSeek 67B, using only 28.5% (maybe even 18.2%) of computations."
- First layer stays dense (p.15): "...substitute all FFNs except for the first layer with MoE layers, since we observe that the load balance status converges especially slower for the first layer." This is the origin of `first_k_dense_replace`.

---

## 3. Multi-head Latent Attention (DeepSeek-V2, arXiv 2405.04434)

### 3.1 Low-rank KV joint compression (PDF p.7, Section 2.1.2)

Equations (p.7, Eq. 9 to 11):

```
c_t^KV = W^DKV h_t          (9)     c_t^KV in R^{d_c}, d_c << d_h n_h
k_t^C  = W^UK c_t^KV        (10)
v_t^C  = W^UV c_t^KV        (11)
```

Quote (p.7), the key screenshot:
> "During inference, MLA only needs to cache c_t^KV, so its KV cache has only d_c l elements, where l denotes the number of layers. In addition, during inference, since W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to compute keys and values out for attention."

Query compression (p.8, Eq. 12 to 13): "we also perform low-rank compression for the queries, even if it cannot reduce the KV cache". The reason given is activation memory during training.

### 3.2 Decoupled RoPE (PDF p.8, Section 2.1.3)

The problem, quoted (p.8):
> "However, RoPE is incompatible with low-rank KV compression. To be specific, RoPE is position-sensitive for both keys and queries. If we apply RoPE for the keys k_t^C, W^UK in Equation 10 will be coupled with a position-sensitive RoPE matrix. In this way, W^UK cannot be absorbed into W^Q any more during inference, since a RoPE matrix related to the currently generating token will lie between W^Q and W^UK and matrix multiplication does not obey a commutative law. As a result, we must recompute the keys for all the prefix tokens during inference, which will significantly hinder the inference efficiency."

The fix (p.8): "we propose the decoupled RoPE strategy that uses additional multi-head queries q_{t,i}^R in R^{d_h^R} and a shared key k_t^R in R^{d_h^R} to carry RoPE".

Equations (p.8, Eq. 14 to 19):

```
[q^R_{t,1}; ...; q^R_{t,n_h}] = q^R_t = RoPE(W^QR c^Q_t)          (14)
k^R_t = RoPE(W^KR h_t)                                            (15)
q_{t,i} = [q^C_{t,i}; q^R_{t,i}]                                  (16)
k_{t,i} = [k^C_{t,i}; k^R_t]                                      (17)
o_{t,i} = sum_j Softmax_j( q_{t,i}^T k_{j,i} / sqrt(d_h + d_h^R) ) v^C_{j,i}   (18)
u_t = W^O [o_{t,1}; ...; o_{t,n_h}]                               (19)
```

Quote (p.8): "During inference, the decoupled key should also be cached. Therefore, DeepSeek-V2 requires a total KV cache containing (d_c + d_h^R) l elements."

Full MLA formulas are in Appendix C (p.31, Eq. 37 onward).

### 3.3 Weight absorption in practice

What "absorb" means, in the V3 reference code (https://raw.githubusercontent.com/deepseek-ai/DeepSeek-V3/main/inference/model.py, lines 17 and 481 to 495; VERIFIED):

```python
attn_impl: Literal["naive", "absorb"] = "absorb"
...
wkv_b = wkv_b.view(self.n_local_heads, -1, self.kv_lora_rank)
q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b[:, :self.qk_nope_head_dim])   # fold W^UK into the query
self.kv_cache[:bsz, start_pos:end_pos] = self.kv_norm(kv)                         # cache only the 512-d latent
self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)                          # and the 64-d RoPE key
scores = (torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos]) +
          torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])) * self.softmax_scale
...
x = torch.einsum("bsht,btc->bshc", scores, self.kv_cache[:bsz, :end_pos])         # attend in latent space
x = torch.einsum("bshc,hdc->bshd", x, wkv_b[:, -self.v_head_dim:])                # apply W^UV after attention
```

So at decode time every head's query is projected into the 512-d latent space, scores are taken against the cached latents directly (plus the 64-d RoPE part), the weighted sum is taken in latent space, and W^UV is applied once at the end. The V3.2 paper names the two forms (2512.02556 p.20, Appendix A, Figure 7 caption): "For DeepSeek-V3.1-Terminus, the MHA mode is used for training and prefilling, while the MQA mode is used for decoding." In MQA mode, the "single KV head" is the 576-d vector [c^KV; k^R], shared by all 128 query heads.

### 3.4 The KV cache comparison table (PDF p.9, Table 1)

| Attention Mechanism | KV Cache per Token (# Element) | Capability |
|---|---|---|
| Multi-Head Attention (MHA) | 2 n_h d_h l | Strong |
| Grouped-Query Attention (GQA) | 2 n_g d_h l | Moderate |
| Multi-Query Attention (MQA) | 2 d_h l | Weak |
| MLA (Ours) | (d_c + d_h^R) l ≈ (9/2) d_h l | Stronger |

Caption (p.9): "For DeepSeek-V2, d_c is set to 4 d_h and d_h^R is set to d_h / 2. So, its KV cache is equal to GQA with only 2.25 groups, but its performance is stronger than MHA."

Ablation (p.32, Table 9, MLA vs MHA, KV cache per token in elements): Small MoE w/ MHA 110.6K, w/ MLA 15.6K; Large MoE w/ MHA 860.2K, w/ MLA 34.6K. Table 8 on the same page shows MHA beats GQA and MQA at 7B dense.

Headline (p.1, abstract): "DeepSeek-V2 achieves significantly stronger performance, and meanwhile saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times."

**DERIVED: where 93.3% comes from.** The paper deploys V2 with KV cache quantized to "6 bits on average" (p.16). V2 cache = (512 + 64) x 60 layers = 34,560 elements x 0.75 bytes = 25,920 bytes. DeepSeek 67B cache = 2 x 8 KV heads x 128 x 95 layers = 194,560 elements x 2 bytes (BF16) = 389,120 bytes. 1 - 25,920 / 389,120 = 0.9334 = 93.3%. The match is exact, so the 93.3% compares a 6-bit V2 cache with a BF16 67B cache, not like for like. At equal BF16 precision the saving is 1 - 69,120 / 389,120 = 82.2%. (DERIVED; the paper does not spell this out.)

### 3.5 DERIVED: KV bytes per token for DeepSeek-V3

Inputs (VERIFIED, V3 config and 2412.19437 p.22): kv_lora_rank d_c = 512, qk_rope_head_dim d_h^R = 64, layers l = 61, n_h = 128, qk_nope_head_dim = 128, v_head_dim = 128.

MLA cache per token per layer: d_c + d_h^R = 512 + 64 = 576 elements.

| Format | Arithmetic | Bytes per token |
|---|---|---|
| MLA, BF16 | 576 x 61 x 2 B | **70,272 B (68.6 KiB)** |
| MLA, FP8 serving layout (FlashMLA "FP8 with scale": 512 B FP8 NoPE + 16 B scales + 128 B BF16 RoPE = 656 B per layer) | 656 x 61 | 40,016 B |
| Equivalent MHA (true V3 head shapes: K 192 dims, V 128 dims, 128 heads), BF16 | 128 x (192 + 128) x 61 x 2 B | 4,997,120 B (4.77 MiB) |
| Textbook MHA with d_h = 128 (Table 1 formula 2 n_h d_h l), BF16 | 2 x 128 x 128 x 61 x 2 B | 3,997,696 B |
| GQA with 8 groups, d_h = 128, BF16 | 2 x 8 x 128 x 61 x 2 B | 249,856 B |
| MQA, d_h = 128, BF16 | 2 x 128 x 61 x 2 B | 31,232 B |

Ratios: MLA BF16 is 70,272 / 4,997,120 = 1.41% of the true MHA cache, 28% of GQA-8, and 2.25x MQA (576 / 256 = 2.25, the "2.25 groups" of Table 1).

Cross-check (VERIFIED): the ISCA paper 2505.09343 (PDF p.4, Table 1, "KV cache size comparison (BF16 precision)") lists "DeepSeek-V3 (MLA) 70.272 KB 1x; Qwen-2.5 72B (GQA) 327.680 KB 4.66x; LLaMA-3.1 405B (GQA) 516.096 KB 7.28x". 70.272 KB matches the derivation exactly.

FP8 layout source (VERIFIED): FlashMLA README at commit ba89a34: "In the "FP8 with scale" format for DeepSeek V3.2 (head_dim = 576, sparse attention), a page block is page_block_size token-major rows of 656 Bytes each: First 512 bytes: The "quantized NoPE" part ... Next 16 bytes: Scale factors, containing 4 float32 values ... Last 128 bytes: The "RoPE" part, containing 64 bfloat16 values. This part is not quantized for accuracy."

---

## 4. DeepSeek-V3 (arXiv 2412.19437)

### 4.1 Sigmoid gating with normalization (PDF p.8 to p.9, Section 2.1.2)

Equations (p.8, Eq. 12 to 15):

```
h'_t = u_t + sum_{i=1}^{N_s} FFN^(s)_i(u_t) + sum_{i=1}^{N_r} g_{i,t} FFN^(r)_i(u_t)   (12)
g_{i,t} = g'_{i,t} / sum_{j=1}^{N_r} g'_{j,t}                                            (13)
g'_{i,t} = s_{i,t} if s_{i,t} in Topk({s_{j,t} | 1 <= j <= N_r}, K_r), else 0             (14)
s_{i,t} = Sigmoid(u_t^T e_i)                                                             (15)
```

Quote (p.9): "Slightly different from DeepSeek-V2, DeepSeek-V3 uses the sigmoid function to compute the affinity scores, and applies a normalization among all selected affinity scores to produce the gating values."

In code, the normalized weights are then multiplied by `route_scale` = `routed_scaling_factor` = 2.5 (V3 model.py lines 595 to 597: `weights /= weights.sum(dim=-1, keepdim=True)`; `weights *= self.route_scale`).

### 4.2 Auxiliary-loss-free load balancing (PDF p.9)

Quote (p.9):
> "To be specific, we introduce a bias term b_i for each expert and add it to the corresponding affinity scores s_{i,t} to determine the top-K routing"

```
g'_{i,t} = s_{i,t} if s_{i,t} + b_i in Topk({s_{j,t} + b_j | 1 <= j <= N_r}, K_r), else 0   (16)
```

Quote (p.9), the key sentences:
> "Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score s_{i,t}. During training, we keep monitoring the expert load on the whole batch of each training step. At the end of each step, we will decrease the bias term by γ if its corresponding expert is overloaded, and increase it by γ if its corresponding expert is underloaded, where γ is a hyper-parameter called bias update speed."

Values (p.23): "we set the bias update speed γ to 0.001 for the first 14.3T tokens, and to 0.0 for the remaining 500B tokens."

In the weights, the bias is the tensor `model.layers.N.mlp.gate.e_score_correction_bias` (V3 safetensors index). In code: `self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.float32))`, added to scores before top-k, while `weights = original_scores.gather(1, indices)` uses the unbiased scores (V3 model.py lines 564 to 594).

The method comes from Wang et al. 2024a (arXiv 2408.15664, "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts"). I did not download that paper.

### 4.3 Complementary sequence-wise auxiliary loss (PDF p.9)

```
L_Bal = alpha * sum_{i=1}^{N_r} f_i P_i                                          (17)
f_i = N_r / (K_r T) * sum_{t=1}^{T} 1(s_{i,t} in Topk({s_{j,t}}, K_r))           (18)
s'_{i,t} = s_{i,t} / sum_{j=1}^{N_r} s_{j,t}                                    (19)
P_i = 1/T * sum_{t=1}^{T} s'_{i,t}                                              (20)
```

Quote (p.9): "where the balance factor α is a hyper-parameter, which will be assigned an extremely small value for DeepSeek-V3; 1(·) denotes the indicator function; and T denotes the number of tokens in a sequence." Value (p.23): "For the balance loss, we set α to 0.0001, just to avoid extreme imbalance within any single sequence."

Note the difference from V2: T is now the tokens of one sequence, and the scores are renormalized (Eq. 19) because sigmoid scores do not sum to 1.

### 4.4 Node-limited routing (PDF p.10)

Quote (p.10): "Like the device-limited routing used by DeepSeek-V2, DeepSeek-V3 also uses a restricted routing mechanism to limit communication costs during training. In short, we ensure that each token will be sent to at most M nodes, which are selected according to the sum of the highest K_r/M affinity scores of the experts distributed on each node. Under this constraint, our MoE training framework can nearly achieve full computation-communication overlap."

Values (p.23): "for each layer, the routed experts will be uniformly deployed on 64 GPUs belonging to 8 nodes. As for the node-limited routing, each token will be sent to at most 4 nodes (i.e., M = 4)."

Code mapping (VERIFIED): `n_group = 8` (one group per node), `topk_group = 4` (M), K_r / M = 8 / 4 = 2, which is exactly `group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)` in V3 model.py line 589. V2's equivalent used the max score per group (`scores.amax(dim=-1)`, line 587, the branch taken when there is no bias).

No token dropping (p.10): "DeepSeek-V3 does not drop any tokens during training. In addition, we also implement specific deployment strategies to ensure inference load balance, so DeepSeek-V3 also does not drop tokens during inference."

### 4.5 Multi-Token Prediction (PDF p.10 to p.11, Section 2.2)

Quote (p.10): "Different from Gloeckle et al. (2024), which parallelly predicts D additional tokens using independent output heads, we sequentially predict additional tokens and keep the complete causal chain at each prediction depth."

Module definition (p.10): "The k-th MTP module consists of a shared embedding layer Emb(·), a shared output head OutHead(·), a Transformer block TRM_k(·), and a projection matrix M_k in R^{d x 2d}."

```
h'^k_i = M_k [RMSNorm(h^{k-1}_i); RMSNorm(Emb(t_{i+k}))]        (21)
h^k_{1:T-k} = TRM_k(h'^k_{1:T-k})                               (22)
P^k_{i+k+1} = OutHead(h^k_i)                                    (23)
L^k_MTP = CrossEntropy(P^k_{2+k:T+1}, t_{2+k:T+1}) = -1/T sum_{i=2+k}^{T+1} log P^k_i[t_i]   (24)
L_MTP = λ/D * sum_{k=1}^{D} L^k_MTP                             (25)
```

Values (p.22 to p.23): "The multi-token prediction depth D is set to 1, i.e., besides the exact next token, each token will predict one additional token." and "The MTP loss weight λ is set to 0.3 for the first 10T tokens, and to 0.1 for the remaining 4.8T tokens."

Inference use (p.11): "Our MTP strategy mainly aims to improve the performance of the main model, so during inference, we can directly discard the MTP modules and the main model can function independently and normally. Additionally, we can also repurpose these MTP modules for speculative decoding to further improve the generation latency."

Measured speculative decoding gain (p.35, Section 5.4.3): "Based on our evaluation, the acceptance rate of the second token prediction ranges between 85% and 90% across various generation topics, demonstrating consistent reliability. This high acceptance rate enables DeepSeek-V3 to achieve a significantly improved decoding speed, delivering 1.8 times TPS (Tokens Per Second)."

Weight layout (README_WEIGHTS.md, VERIFIED): the MTP module is `model.layers.61` with `enorm`, `hnorm`, `eh_proj` (the M_k matrix), a full attention plus MoE block, `embed_tokens` and `shared_head`. Later, DSpark (Section 6.6) replaced MTP-1 as the production drafter for V4.

### 4.6 Other V3 architecture facts (PDF p.22)

"We substitute all FFNs except for the first three layers with MoE layers. Each MoE layer consists of 1 shared expert and 256 routed experts, where the intermediate hidden dimension of each expert is 2048. Among the routed experts, 8 experts will be activated for each token, and each token will be ensured to be sent to at most 4 nodes." and "As DeepSeek-V2, DeepSeek-V3 also employs additional RMSNorm layers after the compressed latent vectors, and multiplies additional scaling factors at the width bottlenecks. Under this configuration, DeepSeek-V3 comprises 671B total parameters, of which 37B are activated for each token."

---

## 5. Sparse attention: NSA and DSA

### 5.1 Native Sparse Attention (arXiv 2502.11089)

Authors include Jingyang Yuan, Huazuo Gao, Damai Dai, Wangding Zeng, Wenfeng Liang (DeepSeek-AI and Peking University). NSA is a research paper; no released DeepSeek LLM used NSA as published. Its ideas (blockwise selection with a learned scorer, sliding-window branch, compression) reappear in DSA and in V4's CSA.

Framework (p.6, Eq. 5): `o*_t = sum_{c in C} g^c_t · Attn(q_t, K~^c_t, V~^c_t)`, with C = {cmp, slc, win}. Quote (p.6): "NSA have three mapping strategies C = {cmp, slc, win}, representing compression, selection, and sliding window for keys and values. g^c_t in [0, 1] is the gate score for corresponding strategy c, derived from input features via an MLP and sigmoid activation."

Compression (p.6, Eq. 7): `K~^cmp_t = { φ(k_{id+1 : id+l}) | 0 <= i <= floor((t - l)/d) }`, "where l is the block length, d is the sliding stride between adjacent blocks, and φ is a learnable MLP with intra-block position encoding to map keys in a block to a single compressed key."

Selection (p.7, Eq. 8): block importance comes for free from the compression branch: `p^cmp_t = Softmax(q_t^T K~^cmp_t)`. For GQA (p.7, Eq. 10), scores are summed across the heads of a group so all heads pick the same blocks: "This aggregation ensures consistent block selection across heads within the same group." Top-n blocks (p.8, Eq. 11 to 12).

Sliding window (p.8): "we introduce a dedicated sliding window branch that explicitly handles local context, allowing other branches (compression and selection) to focus on learning their respective features without being shortcutted by local patterns."

Hyper-parameters (p.10): "we set compression block size l = 32, sliding stride d = 16, selected block size l' = 64, selected block count n = 16 (including fixed activating the 1 initial block and 2 local blocks), and sliding window size w = 512." Backbone (p.9): "27B total parameters with 3B active parameters. The model consists of 30 layers with a hidden dimension of 2560. For GQA, we set the number of groups to 4, with a total of 64 attention heads."

**Cost.** Per query, NSA reads N_t = (t - l)/d + n·l' + w keys instead of t. **DERIVED** at t = 65,536: (65,536 - 32)/16 + 16 x 64 + 512 = 4,094 + 1,024 + 512 = 5,630, which matches the paper's Table 4 (p.14) "NSA ... 5632" tokens of memory access at 64K. Table 4 values: context 8192 / 16384 / 32768 / 65536, NSA 2048 / 2560 / 3584 / 5632, "Expected Speedup 4x / 6.4x / 9.1x / 11.6x".

Measured speedups (VERIFIED):
- Training, Triton kernel vs Triton FlashAttention-2 (p.13, Figure 6): forward 2.1x / 3.8x / 6.3x / 9.0x at 8k / 16k / 32k / 64k; backward 1.1x / 2.0x / 3.4x / 6.0x. Quote: "our NSA achieves progressively greater speedups as context length increases, up to 9.0x forward and 6.0x backward speedup at 64k context-length."
- Decoding (p.14): "achieving up to 11.6x speedup at 64k context-length." Note: the 11.6x figure is the memory-access ratio of Table 4 ("expected speedup"), not a separately timed kernel.

### 5.2 DeepSeek Sparse Attention (V3.2-Exp and V3.2, arXiv 2512.02556)

V3.2-Exp has its own tech report PDF in the GitHub repo (cited in its HF README as "DeepSeek-V3.2-Exp: Boosting Long-Context Efficiency with DeepSeek Sparse Attention"). The V3.2 arXiv paper repeats and supersedes it, so I quote the arXiv paper.

**Lightning indexer** (p.3, Eq. 1):

```
I_{t,s} = sum_{j=1}^{H^I} w^I_{t,j} · ReLU(q^I_{t,j} · k^I_s)        (1)
```

Quote (p.3): "where H^I denotes the number of indexer heads; q^I_{t,j} in R^{d^I} and w^I_{t,j} in R are derived from the query token h_t; and k^I_s in R^{d^I} is derived from the preceding token h_s. We choose ReLU as the activation function for throughput consideration. Given that the lightning indexer has a small number of heads and can be implemented in FP8, its computational efficiency is remarkable."

**Top-k selection** (p.3, Eq. 2): `u_t = Attn(h_t, { c_s | I_{t,s} in Top-k(I_{t,:}) })`. Quote: "our fine-grained token selection mechanism retrieves only the key-value entries {c_s} corresponding to the top-k index scores."

**Instantiated on MLA's MQA mode** (p.3): "At the kernel level, each key-value entry must be shared across multiple queries for computational efficiency (Yuan et al., 2025). Therefore, we implement DSA based on the MQA (Shazeer, 2019) mode of MLA, where each latent vector (the key-value entry of MLA) will be shared across all query heads of the query token." Figure 2 (p.4) is the architecture screenshot.

**Settings** (config + code, VERIFIED): 64 indexer heads, indexer head dim 128, top-k 2048. Code (V3.2-Exp model.py, `class Indexer`, lines 435 to 487): the indexer query reuses MLA's query latent (`self.wq_b = Linear(self.q_lora_rank, self.n_heads * self.head_dim)`), the indexer key is a single 128-d head (`self.wk = Linear(self.dim, self.head_dim)`) cached in FP8 (`k_cache ... dtype=torch.float8_e4m3fn`), per-head weights come from `weights_proj = Linear(self.dim, self.n_heads)`, q and k pass through a Hadamard rotation (`rotate_activation`) before FP8 quantization, and selection is `index_score.topk(min(self.index_topk, end_pos), dim=-1)`.

**Training** (p.4):
- Dense warm-up: only the indexer trains, with a KL loss to the head-summed main attention distribution: `L^I = sum_t D_KL(p_{t,:} || Softmax(I_{t,:}))` (Eq. 3). "We train the indexer for only 1000 steps, with each step consisting of 16 sequences of 128K tokens, resulting in a total of 2.1B tokens."
- Sparse stage (Eq. 4, KL only over the selected set S_t): "we use a learning rate of 7.3 x 10^-6, and select 2048 key-value tokens for each query token. We train both the main model and the indexer for 15000 steps, with each step consisting of 480 sequences of 128K tokens, resulting in a total of 943.7B tokens." Also: "we detach the indexer input from the computational graph for separate optimization."

**Complexity** (p.5, Section 2.3), the key quote:
> "DSA reduces the core attention complexity of the main model from O(L^2) to O(Lk), where k (≪ L) is the number of selected tokens. Although the lightning indexer still has a complexity of O(L^2), it requires much less computation compared with MLA in DeepSeek-V3.1-Terminus."

**DERIVED: why the O(L^2) indexer is cheap.** Per (query, key) pair, the indexer does 64 heads x 128 dims = 8,192 multiply-adds in FP8. Dense MLA in MQA mode does 128 heads x (576 for QK + 512 for PV) = 139,264 multiply-adds in BF16. Ratio about 17x fewer operations, and FP8 tensor cores run about 2x faster than BF16 on H800, so the quadratic term is roughly 30x to 35x cheaper per pair. Then core attention only touches 2,048 keys. This is my estimate, not a paper number.

**Measured cost reductions** (p.5 to p.6, Figure 3): costs "estimated from benchmarking the actual service deployed on H800 GPUs, at a rental price of 2 USD per GPU hour." Reading the plot (approximate, DERIVED from the figure, not stated in text): at token position 128K, prefill cost per million tokens falls from about $0.65 (V3.1-Terminus) to about $0.19 (V3.2); decode cost falls from about $2.15 to about $0.25. Note (p.5): "for short-sequence prefilling, we specially implement a masked MHA mode to simulate DSA, which can achieve higher efficiency under short-context conditions." The paper gives no single headline percentage. The widely reported "API price cut of over 50%" at the V3.2-Exp launch is pricing, not a measured cost, and I did not verify it in a primary source (UNVERIFIED).

**Quality parity** (p.5): "we do not observe substantial performance degradation compared with DeepSeek-V3.1-Terminus, on both short- and long-context tasks."

**DERIVED: V3.2 KV bytes per token.** Main MLA cache in FlashMLA FP8 layout 656 B/layer + indexer key 128 B FP8 + 4 B float32 scale = 132 B/layer. (656 + 132) x 61 = **48,068 B**, which matches exactly the "DeepSeek-V3.2 48,068" bar in 2609.19969 p.1 Figure 1(b). So DSA does not shrink the cache; it adds 20% to it. DSA saves compute and memory bandwidth (only 2,048 entries are read per query), not capacity.

Kernels: FlashMLA README: the DSA sparse kernels "achieve up to 640 TFlops during prefilling and 410 TFlops during decoding" (2025.09.29 entry, VERIFIED).

---

## 6. New architecture research, late 2025 to 2026

All items below are verified to exist in primary sources.

### 6.1 mHC: Manifold-Constrained Hyper-Connections (arXiv 2512.24880)

Authors: Zhenda Xie (corresponding), Yixuan Wei, Huanqi Cao, and others incl. Wenfeng Liang. Submitted 31 Dec 2025.

Background: Hyper-Connections (HC, Zhu et al., ByteDance) widen the residual stream into n parallel streams. HC single layer (p.3, Eq. 3): `x_{l+1} = H^res_l x_l + H^post_l^T F(H^pre_l x_l, W_l)`, "the feature dimension of x_l and x_{l+1} is expanded from C to n x C, where n is the expansion rate."

The problem (p.1, abstract): "this diversification fundamentally compromises the identity mapping property intrinsic to the residual connection, which causes severe training instability and restricted scalability, and additionally incurs notable memory access overhead." Evidence (p.6): "the Amax Gain Magnitude yields extreme values with peaks of 3000, a stark divergence from 1 that confirms the presence of exploding residual streams."

The fix (p.8, Eq. 6): constrain H^res_l to the doubly stochastic matrices (the Birkhoff polytope): `H^res_l 1_n = 1_n, 1_n^T H^res_l = 1_n^T, H^res_l >= 0`. Quoted properties (p.8): "Norm Preservation: The spectral norm of a doubly stochastic matrix is bounded by 1 ... Compositional Closure: The set of doubly stochastic matrices is closed under matrix multiplication." and "the residual mapping acts as a convex combination of permutations."

Projection (p.9, Eq. 8 to 9): `H^pre = σ(H~^pre)`, `H^post = 2σ(H~^post)`, `H^res = Sinkhorn-Knopp(H~^res)`; Sinkhorn-Knopp exponentiates then alternates row and column normalization; "We choose t_max = 20 as a practical value in our experiments."

Cost (p.4): "In-house large-scale training indicates that mHC supports training at scale and introduces only a 6.7% additional time overhead when expansion rate n = 4."

Results (p.13, Table 4, 27B models): BBH 43.8 (baseline) / 48.9 (HC) / 51.0 (mHC); DROP 47.0 / 51.6 / 53.9; GSM8K 46.7 / 53.2 / 53.8; MMLU 59.0 / 63.0 / 63.4; MATH 22.0 / 26.4 / 26.0. Scaling runs at 3B, 9B and 27B (p.13, Figure 6).

Production use: V4-Flash and V4-Pro (`hc_mult: 4`, `hc_sinkhorn_iters: 20`), and V4.1-Flash as "Single-Pass mHC" (6.5).

### 6.2 Engram: conditional memory (arXiv 2601.07372)

Authors: Xin Cheng, Rui Tian, Wangding Zeng, Damai Dai, and others (Peking University and DeepSeek-AI). Code: https://github.com/deepseek-ai/Engram.

Idea (p.1, abstract): "While Mixture-of-Experts (MoE) scales capacity via conditional computation, Transformers lack a native primitive for knowledge lookup, forcing them to inefficiently simulate retrieval through computation. To address this, we introduce conditional memory as a complementary sparsity axis, instantiated via Engram, a module that modernizes classic N-gram embedding for O(1) lookup."

Mechanism (p.4):
- Tokenizer compression: "this process achieves a 23% reduction in the effective vocabulary size for a 128k tokenizer".
- Multi-head hashing (Eq. 1): `z_{t,n,k} = φ_{n,k}(g_{t,n})`, `e_{t,n,k} = E_{n,k}[z_{t,n,k}]`, tables of prime size, "φ_{n,k} is implemented as a lightweight multiplicative-XOR hash".
- Context-aware gating (Eq. 3 to 4): `k_t = W_K e_t`, `v_t = W_V e_t`, `α_t = σ(RMSNorm(h_t)^T RMSNorm(k_t) / sqrt(d))`, output `α_t · v_t`. Quote: "if the retrieved memory e_t contradicts the current context h_t, the gate α_t tends toward zero, effectively suppressing the noise."
- Short depthwise causal conv (Eq. 5), then residual add: `H^(l) <- H^(l) + Y` (p.5).

Scaling law (p.8): "Figure 3 (left) reveals a consistent U-shaped relationship between validation loss and the allocation ratio ρ ... reallocating roughly 20%-25% of the sparse parameter budget to Engram yields the best performance."

Main result (p.9 to p.10): MoE-27B vs Engram-27B, both 26.7B total, 3.8B active, 262B tokens: "We reduce the number of routed experts from 72 to 55 and reallocate the freed parameters to a 5.7B-parameter embedding module (ρ = 74.3%)". Abstract gains: "MMLU +3.4; CMMLU +4.0 ... BBH +5.0; ARC-Challenge +3.7 ... HumanEval +3.0; MATH +2.4", "Multi-Query NIAH: 84.2 -> 97.0".

Systems (p.3): "offloading a 100B-parameter table to host memory incurs negligible overhead (< 3%)."

Production use: V4.1-Flash (6.5), 196B Engram parameters at layers 1 and 14. V4-Flash and V4-Pro do **not** use Engram (no engram fields in their configs).

### 6.3 DeepSeek-V4 (arXiv 2606.19348): architecture

Abstract (p.1): "DeepSeek-V4-Pro with 1.6T parameters (49B activated) and DeepSeek-V4-Flash with 284B parameters (13B activated) [DASH] both supporting a context length of one million tokens." ([DASH] marks a long dash in the PDF, removed here under the site rule.) Upgrades: "(1) a hybrid attention architecture that combines Compressed Sparse Attention (CSA) and Heavily Compressed Attention (HCA) to improve long-context efficiency; (2) Manifold-Constrained Hyper-Connections (mHC) that enhance conventional residual connections; (3) and the Muon optimizer for faster convergence and greater training stability." Efficiency (p.1): "In the one-million-token context setting, DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2." Figure 1 right (p.1): single-token FLOPs "3.7x lower" (Pro) and "9.8x lower" (Flash); accumulated KV cache "9.5x smaller" (Pro) and "13.7x smaller" (Flash), vs V3.2 at 1M.

**MoE changes** (p.7, Section 2.1):
- "we change the activation function that computes the affinity scores from Sigmoid(·) into Sqrt(Softplus(·))." Code: `scores = F.softplus(scores).sqrt()` (V4 model.py line 571).
- Aux-loss-free bias kept, "augmented by a slight sequence-wise balance loss".
- "we remove the constraint on the number of routing target nodes" (no `n_group` in config).
- Hash routing replaces the dense FFN layers: "we replace the dense FFN layers in the initial several Transformer blocks with MoE layers that employ Hash routing (Roller et al., 2021). The Hash routing strategy determines the target experts of each token according to a predefined hash function with regard to the input token ID." Code: a fixed table `self.tid2eid = nn.Parameter(torch.empty(args.vocab_size, args.n_activated_experts, dtype=torch.int32), requires_grad=False)` and `indices = self.tid2eid[input_ids]` (lines 559, 577). Config: `num_hash_layers: 3`.
- MTP: "we adopt the same strategy for DeepSeek-V4 series without modification" (p.7).

**mHC in V4** (p.7 to p.8, Eq. 1 to 8): `X_{l+1} = B_l X_l + C_l F_l(A_l X_l)`; B_l projected to the Birkhoff polytope by Sinkhorn-Knopp, A_l = σ(A~), C_l = 2σ(C~), t_max = 20; parameters are input-dependent: `B~_l = α^res_l · Mat(X^_l W^res_l) + S^res_l` (Eq. 4).

**CSA, Compressed Sparse Attention** (p.9 to p.11). Quote (p.9): "CSA integrates both compression and sparse attention strategies: it first compresses the Key-Value (KV) cache of every m tokens into one entry, and then applies DeepSeek Sparse Attention (DSA) (DeepSeek-AI, 2025) where each query token attends to only k compressed KV entries. HCA aims for extreme compression by consolidating the KV cache of every m' (≫ m) tokens into a single entry."
- Compression (p.10, Eq. 9 to 12): two KV series C^a, C^b and weights Z^a, Z^b with learned positional biases; a softmax over 2m slots produces each compressed entry from 2m overlapping tokens, so the sequence is still compressed m-fold.
- Indexer (p.10, Eq. 13 to 17): same DSA form, `I_{t,s} = sum_h w^I_{t,h} · ReLU(q^I_{t,h} · K^IComp_s)`, run over compressed blocks; the indexer query shares the query latent c^Q_t with the main attention.
- Core attention is MQA over the selected entries; each entry is both key and value (p.11, Eq. 19).
- Grouped output projection (p.11): n_h outputs split into g groups, each projected to d_g, then to d. This is `o_groups` and `o_lora_rank` in the config.

**HCA, Heavily Compressed Attention** (p.11 to p.12, Eq. 20 to 26): non-overlapping compression by m' = 128, dense MQA over all compressed entries, no indexer.

**Other details** (p.12 to p.13):
- RMSNorm on each query head and on the KV entry ("avoids exploding attention logits").
- Partial RoPE on the last 64 dims, plus RoPE at position -i on the output: "we also apply RoPE with position −i on the last 64 dimensions of each o_{t,i}. In this way, the output of the core attention will also carry relative position embeddings".
- Sliding window branch of n_win = 128 uncompressed tokens in every CSA and HCA layer.
- Attention sink (Eq. 27): learnable per-head logit added to the softmax denominator.
- Storage (p.13): "BF16 precision is used for the rotary positional embedding (RoPE) dimensions, while FP8 precision is applied to the remaining dimensions"; "attention computation within the lightning indexer is performed in FP4 precision"; vs a BF16 GQA8 baseline (head dim 128) the KV cache is "approximately 2% times of that baseline in the 1M-context setting."

**Layer plans** (p.24 to p.25, VERIFIED against configs):
- Flash: 43 layers, d = 4096; "For the first two layers, we use pure sliding window attention. For the subsequent layers, CSA and HCA are used in an interleaved manner." CSA m = 4, indexer 64 heads x 128, top-k 512; HCA m' = 128; 64 query heads, head dim c = 512, d_c = 1024; g = 8, d_g = 1024; n_win = 128; "Hash routing strategy for the first 3 MoE layers"; 1 shared + 256 routed experts of width 2048, top-6; MTP depth 1; n_hc = 4, t_max = 20.
- Pro: 61 layers, d = 7168; "For the first two layers, we use HCA"; top-k 1024; 128 query heads, d_c = 1536; g = 16; 1 shared + 384 routed experts of width 3072, top-6.

**Training stability** (p.26): "Anticipatory Routing" (routing indices computed with parameters from step t - Δt, triggered only after a loss spike) and "SwiGLU Clamping": "we clamped the linear component of SwiGLU to the range of [−10, 10], while capping the upper bound of the gate component at 10." This is `swiglu_limit: 10.0`.

**Optimizer** (p.14): Muon for most matrices, AdamW for embedding, head, mHC static biases and gates, and RMSNorm weights; hybrid Newton-Schulz, 8 steps with (3.4445, −4.7750, 2.0315) then 2 steps with (2, −1.5, 0.5); no QK-Clip because of the query and KV RMSNorm.

**DERIVED: V4 KV bytes per token.** FlashMLA README (commit ba89a34): "V4: 584 Bytes per token. The data row is 448 Bytes of quantized NoPE (float8_e4m3) followed by 128 Bytes, i.e. the 64 bfloat16 RoPE values (not quantized). The scale row is 8 Bytes". I assume the FP4 indexer key is 128 x 0.5 B = 64 B plus 4 B of scales = 68 B (this indexer layout is my assumption; it is UNVERIFIED on its own, but the totals below match the paper's chart).
- V4-Flash: 21 CSA layers x (584 + 68) / 4 + 20 HCA layers x 584 / 128 = 3,423 + 91.25 = **3,514 B per token**. The V4.1 paper's Figure 1(b) bar says "DeepSeek-V4-Flash 3,514". Exact match. 48,068 / 3,514 = 13.7x, the "13.7x smaller" of V4 Figure 1.
- V4-Pro: 30 CSA x 163 + 31 HCA x 4.5625 = 4,890 + 141 = about 5,031 B per token; 48,068 / 5,031 = 9.55x, the "9.5x smaller" of V4 Figure 1.
- Sliding-window KV (128 tokens per layer) is a fixed per-request state, not per-token, so it is excluded, as in the paper's chart.

**DERIVED: V4 parameter sanity check** (my script; counts wq_a, wq_b, wkv, grouped wo_a and wo_b, compressors, indexer, mHC mixers, experts, routers, embedding and head; excludes MTP): Flash about 284.3B total, about 13.8B active (published 284B / 13B); Pro about 1,573B total, about 49.8B active (published 1.6T / 49B). Experts dominate: Flash 43 x 257 x 25.17M = 278.1B; Pro 61 x 385 x 66.06M = 1,551B. Small differences come from items I could not pin down exactly (norm weights, MTP, whether the embedding counts as "active").

### 6.4 DeepSeek-V4 releases after the preview

- DeepSeek-V4-Flash-0731 and DeepSeek-V4-Pro-0813: same backbone plus a DSpark drafter (config fields in 1.2). HF repo creation dates 2026-07-31 and 2026-08-13.
- DeepSeek-V4-Flash-Vision-Exp (repo created 2026-08-31): exists on HF; I did not read its config or a paper for it (UNVERIFIED details).
- DeepSeek-V4-Pro-DSpark and V4-Flash-DSpark (2026-06-27): DSpark checkpoints.

### 6.5 DeepSeek-V4.1-Flash (arXiv 2609.19969)

Abstract (p.1): "a multimodal Mixture-of-Experts (MoE) model with 552B backbone parameters and support for contexts of up to one million tokens. With its Causal Encoder-Decoder (CED) architecture, the model activates 16B parameters per token during decode but only 8B parameters during prefill" and "These designs reduce its global KV cache footprint (always in HBM) to 890 bytes per token, roughly 1/4 of the corresponding footprint of DeepSeek-V4-Flash." Pretraining: 45T multimodal tokens. Total incl. memory (p.7): "552B backbone parameters and 196B Engram parameters".

The most screenshot-worthy figure in the whole series is **2609.19969 p.1, Figure 1(b), "Global KV Cache Per Token (Bytes)"**: DeepSeek-V1 (2023.11) 389,120; DeepSeek-V3.2 (2025.12) 48,068 ("8.1x smaller"); DeepSeek-V4-Flash (2026.04) 3,514 ("13.7x smaller"); DeepSeek-V4.1-Flash (2026.09) 890 ("3.9x smaller"). Caption: "approximately 4-fold and 437-fold reductions in per-token global KV cache size relative to DeepSeek-V4-Flash and DeepSeek-V1, respectively." All four bars are reproduced exactly by the derivations in this dossier (3.4, 5.2, 6.3 and below).

Components:
- **Causal Encoder-Decoder** (p.9, Eq. 1): inspired by YOCO; decoder layers (l > L/2) take their global KV from the last encoder hidden state: `C_l = H_{L/2} W^KV_l, Z_l = H_{L/2} W^Z_l`. Sliding-window KV stays layer-local. "CED reduces the prefill complexity from O(NL) to O(NL/2 + n_win x L/2) ≈ O(NL/2), effectively halving the overall computation."
- **CSA2** (p.9 to p.11): KV and indexer keys shared across layers; three per-layer modes, Full, Reindex, Reuse (Figure 4, p.10); compressor simplified (no overlap, no absolute position embedding); indexer K projected from main KV.
- **Hierarchical Sparse Indexer** (p.11 to p.12): the first decoder Full layer selects 2,048 blocks of 8 positions = 16,384 candidates; later Reindex layers score only these, so "the per-query cost of deeper indexers from linear in context length to constant."
- **Single-Pass mHC** (p.12 to p.13, Eq. 6): `X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)`; input mixing uses the previous block's coefficients, so one fused "Mega-mHC" kernel reads and writes the residual once, "halving the activation memory traffic of our original implementation."
- **Engram** (p.13): "We allocate 196B Engram parameters evenly across two modules. Each module uses N-gram orders {2, 3, 4}, with 8 hash heads and a total embedding dimension of 2048 per order. Each head indexes a table of approximately 16M entries ... The modules are placed at layers 1 and 14 (zero-indexed)". The short convolution is dropped.
- **DSpark** (p.13 to p.14): "The drafter comprises three Transformer blocks with a sliding attention window of 128 tokens. A single forward pass through these blocks computes base logits for five draft positions in parallel, while a lightweight Markov head models dependencies among the draft tokens." MTP is omitted during backbone pretraining (p.8).
- **FP4 main KV cache** (p.14): E2M1 with one E4M3 scale per 16 channels; SWA cache stays FP8.
- Layer plan (p.21 to p.22): encoder = 2 SWA layers + 18 CSA2 layers at m = 2 in three groups of six (1 Full + 5 Reuse); decoder = 20 CSA2 layers at m = 1 in five groups of four (first group Full + 3 Reuse; the rest Reindex + 3 Reuse); indexer 32 heads x 128, top-k 512; 64 query heads, head dim 512, d_c = 1280; 1 shared + 384 routed experts of width 2304, top-6.

**DERIVED: the 890 bytes.** FP4 main entry = 288 B (FlashMLA: "256 Bytes containing 512 e2m1 values ... The scale row is 32 Bytes"); indexer key assumed 68 B as above. Only the 4 KV-producing layers store global KV (`kv_source_layer_ids: [2, 8, 14, 20]`): 3 encoder layers at m = 2: 3 x (288 + 68) / 2 = 534; 1 decoder layer at m = 1: 288 + 68 = 356. Total = **890 B**. Exact match with the paper. And 389,120 / 890 = 437.2, the paper's "437-fold".

### 6.6 DSpark (arXiv 2607.05147)

Abstract (p.1): "DSpark utilizes a semi-autoregressive architecture [DASH] coupling a parallel backbone with a lightweight sequential module [DASH] to introduce intra-block dependency modeling and mitigate suffix decay." and "Compared to the established production baseline (MTP-1), DSpark accelerates per-user generation speeds by 60%–85% at matched throughput levels." ([DASH] marks a long dash in the PDF, removed here under the site rule.) Relevance for the series: it is the successor to V3's MTP module as the speculative decoder in production (V4-*-0731/0813, V4.1-Flash).

---

## 7. DeepSeek-V3 parameter count breakdown (DERIVED)

Inputs (all VERIFIED from https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json): d = 7168, n_h = 128, qk_nope = 128, qk_rope = 64, v_head = 128, q_lora = 1536, kv_lora = 512, vocab V = 129,280, dense FFN width 18,432, expert width 2,048, 256 routed + 1 shared experts, top-8, 61 layers, first 3 dense. Weight shapes follow the V3 reference model.py (wq_a, q_norm, wq_b, wkv_a, kv_norm, wkv_b, wo; gate weight plus fp32 bias; SwiGLU experts with 3 matrices each).

### 7.1 Per-component sizes

| Component | Formula | Parameters |
|---|---|---|
| MLA per layer | 7168x1536 + 1536 + 1536x(128x192) + 7168x(512+64) + 512 + 512x(128x256) + (128x128)x7168 | 187,107,328 |
| Layer norms per layer | 2 x 7168 | 14,336 |
| Dense FFN per layer (SwiGLU) | 3 x 7168 x 18432 | 396,361,728 |
| One expert (SwiGLU) | 3 x 7168 x 2048 | 44,040,192 |
| Router per MoE layer | 256 x 7168 + 256 (bias) | 1,835,264 |
| Routed experts per MoE layer | 256 x 44,040,192 | 11,274,289,152 |
| Embedding | 129280 x 7168 | 926,679,040 |
| Output head (not tied) | 129280 x 7168 | 926,679,040 |

### 7.2 Totals for the main model (61 layers)

| Bucket | Arithmetic | Parameters | Share |
|---|---|---|---|
| Embedding | | 0.927B | 0.14% |
| Output head | | 0.927B | 0.14% |
| Attention (MLA), all 61 layers | 61 x 187,107,328 | 11.414B | 1.70% |
| Dense FFN, layers 0 to 2 | 3 x 396,361,728 | 1.189B | 0.18% |
| Routed experts, layers 3 to 60 | 58 x 11,274,289,152 | 653.909B | 97.45% |
| Shared experts, layers 3 to 60 | 58 x 44,040,192 | 2.554B | 0.38% |
| Routers incl. bias | 58 x 1,835,264 | 0.106B | 0.02% |
| RMSNorm weights | 61 x 14,336 + 7,168 | 0.0009B | |
| **Total main model** | | **671,026,419,200 = 671.03B** | |

Published: "671B total parameters" (2412.19437 p.22). Match.

### 7.3 Active parameters per token

Per MoE layer active = attention + norms + 8 routed experts + 1 shared expert + router = 187.1M + 0.014M + 8 x 44.04M + 44.04M + 1.84M = 629.3M.

| Bucket | Parameters |
|---|---|
| Embedding + head | 1.853B |
| 3 dense layers (attention + FFN + norms) | 3 x 583,483,392 = 1.750B |
| 58 MoE layers at 629.3M | 36.497B |
| **Active total** | **37,552,297,472 = 37.55B** |
| Active total without the input embedding (a lookup, not a matmul) | 36.63B |

Published: "37B are activated for each token" (2412.19437 p.22). The weight doc (README_WEIGHTS.md) says "Activation parameters: **36.7B** (including 0.9B for Embedding and 0.9B for the output Head)". My count with both embedding and head is 37.55B; my count without the input embedding is 36.63B. I could not reproduce 36.7B exactly while also counting the embedding; the 0.85B gap is unexplained (UNVERIFIED which convention the README used). Use "37B" as the paper does.

### 7.4 The MTP module

MTP block = enorm + hnorm (2 x 7168) + eh_proj (7168 x 14336 = 102.8M) + one full MoE layer (11,507,286,272) + shared_head norm (7168) = **11,610,068,224 = 11.61B unique**. README_WEIGHTS.md: "Parameters: **11.5B unique parameters**, excluding the shared 0.9B Embedding and 0.9B output Head)". Active in MTP: 2.54B including embedding and head (README: "2.4B").

On Hugging Face the MTP layer stores its own copies of `model.layers.61.embed_tokens.weight` and `model.layers.61.shared_head.head.weight` (VERIFIED in model.safetensors.index.json). So checkpoint total = 671.03 + 11.61 + 0.93 + 0.93 = **684.5B**, which matches the README: "The total size of DeepSeek-V3 models on Hugging Face is 685B, which includes 671B of the Main Model weights and 14B of the Multi-Token Prediction (MTP) Module weights." (14B = 11.6B unique + 1.85B duplicated embedding and head, rounded.)

### 7.5 One-line takeaway for writers

97.5% of V3's parameters are routed experts; attention is only 1.7%. Each token uses 8 of 256 routed experts (3.1% of the routed pool), plus everything else, which is how 671B total becomes about 37B active.

### 7.6 V3.2 addition

The DSA indexer adds per layer: wq_b 1536 x 8192 = 12.58M, wk 7168 x 128 = 0.92M, weights_proj 7168 x 64 = 0.46M, k_norm 256 (LayerNorm weight + bias). About 13.96M per layer, about 0.85B over 61 layers (DERIVED from V3.2-Exp model.py shapes). V3.2 is still described as a 671B model in its README (`config_671B_v3.2.json`).

---

## 8. Key passages for screenshots (index)

| # | arXiv id | PDF page | Section | Exact text (first words) |
|---|---|---|---|---|
| 1 | 2401.06066 | 5 | Fig. 2 + 3.1 | "we segment each expert FFN into m smaller experts by reducing the FFN intermediate hidden dimension to 1/m times its original size." |
| 2 | 2401.06066 | 6 | 3.2 | "Regardless of the router module, each token will be deterministically assigned to these shared experts." |
| 3 | 2401.06066 | 7 | 3.3 | "In practice, we set a small expert-level balance factor to mitigate the risk of routing collapse, and meanwhile set a larger device-level balance factor" |
| 4 | 2405.04434 | 7 | 2.1.2 | "During inference, MLA only needs to cache c_t^KV ... since W^UK can be absorbed into W^Q, and W^UV can be absorbed into W^O, we even do not need to compute keys and values out for attention." |
| 5 | 2405.04434 | 8 | 2.1.3 | "However, RoPE is incompatible with low-rank KV compression." |
| 6 | 2405.04434 | 9 | Table 1 | KV cache per token: MHA 2 n_h d_h l, GQA 2 n_g d_h l, MQA 2 d_h l, MLA (d_c + d_h^R) l ≈ 9/2 d_h l |
| 7 | 2405.04434 | 1 | Abstract | "saves 42.5% of training costs, reduces the KV cache by 93.3%, and boosts the maximum generation throughput to 5.76 times." |
| 8 | 2412.19437 | 9 | 2.1.2 | "Note that the bias term is only used for routing. The gating value, which will be multiplied with the FFN output, is still derived from the original affinity score" |
| 9 | 2412.19437 | 10 | 2.1.2 | "each token will be sent to at most M nodes, which are selected according to the sum of the highest K_r/M affinity scores" |
| 10 | 2412.19437 | 10 | Fig. 3 | "Illustration of our Multi-Token Prediction (MTP) implementation. We keep the complete causal chain for the prediction of each token at each depth." |
| 11 | 2412.19437 | 35 | 5.4.3 | "the acceptance rate of the second token prediction ranges between 85% and 90% ... delivering 1.8 times TPS (Tokens Per Second)." |
| 12 | 2502.11089 | 3 | Fig. 2 | "three parallel attention branches: For a given query, preceding keys and values are processed into compressed attention for coarse-grained patterns, selected attention for important token blocks, and sliding attention for local context." (caption, partially quoted) |
| 13 | 2502.11089 | 13 | 5.1 | "up to 9.0x forward and 6.0x backward speedup at 64k context-length." |
| 14 | 2512.02556 | 3 | 2.1 | "We choose ReLU as the activation function for throughput consideration." |
| 15 | 2512.02556 | 5 | 2.3 | "DSA reduces the core attention complexity of the main model from O(L^2) to O(Lk)" |
| 16 | 2512.02556 | 6 | Fig. 3 | Inference cost curves, V3.1-Terminus vs V3.2, prefill and decode |
| 17 | 2512.24880 | 8 | 4.1 | "Norm Preservation: The spectral norm of a doubly stochastic matrix is bounded by 1" |
| 18 | 2512.24880 | 4 | 1 | "introduces only a 6.7% additional time overhead when expansion rate n = 4." |
| 19 | 2601.07372 | 8 | 3 | "reallocating roughly 20%-25% of the sparse parameter budget to Engram yields the best performance." |
| 20 | 2606.19348 | 1 | Abstract | "DeepSeek-V4-Pro requires only 27% of single-token inference FLOPs and 10% of KV cache compared with DeepSeek-V3.2." |
| 21 | 2606.19348 | 9 | 2.3 | "CSA integrates both compression and sparse attention strategies" |
| 22 | 2606.19348 | 7 | 2.1 | "we change the activation function that computes the affinity scores from Sigmoid(·) into Sqrt(Softplus(·))." |
| 23 | 2609.19969 | 1 | Fig. 1(b) | Global KV Cache Per Token: 389,120 / 48,068 / 3,514 / 890 bytes |

(The "x" in rows 13 and 18 stands for the multiplication sign used in the PDFs.)

---

## 9. What I could not verify

- The README's "36.7B activated" for V3 could not be reproduced exactly; my count is 37.55B with embedding and head, 36.63B without the input embedding (7.3).
- The exact byte layout of the FP4 indexer key cache in V4 and V4.1 (I assumed 64 B + 4 B scales). The assumption reproduces the published 3,514 B and 890 B totals exactly, which is strong but indirect evidence.
- The "API price cut of over 50%" at the V3.2-Exp launch: not checked against a primary DeepSeek page.
- Exact dollar values in V3.2 Figure 3 are read off the plot by eye.
- DeepSeek-V4-Flash-Vision-Exp: repo exists; config and design not examined.
- Third-party claims seen in search results (for example a vLLM recipe saying V4.1-Flash is "522B total") conflict with the paper and README (552B backbone). Trust the paper.
