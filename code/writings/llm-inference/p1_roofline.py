"""Part 1 numbers for the roofline and batching sections.
1. Counts Qwen2.5-0.5B's weights from the model itself (bytes, FLOPs per token).
2. Works out where the H100 roofline puts each batch size (NVIDIA spec numbers).
3. Places the published Part 1 measurements (prefill and decode step times) on the roofline:
   FLOPs per byte, achieved TFLOP/s, achieved bandwidth.
Device: mps (Apple M5 Pro), then cpu. Output: results/p1_roofline_stdout.txt (run with | tee)."""
import torch
from transformers import AutoModelForCausalLM

DEV = "mps" if torch.backends.mps.is_available() else "cpu"
sync = torch.mps.synchronize if DEV == "mps" else (lambda: None)
print(f"device: {DEV}, torch {torch.__version__}")

# ---- 1. the weights
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.bfloat16)
cfg = model.config
n_all = sum(p.numel() for p in model.parameters())          # tied embedding counted once
n_emb = model.get_input_embeddings().weight.numel()
n_body = n_all - n_emb                                       # the 24 layers (+ norms)
print(f"layers {cfg.num_hidden_layers}, hidden {cfg.hidden_size}, mlp {cfg.intermediate_size}, vocab {cfg.vocab_size}, tied {cfg.tie_word_embeddings}")
print(f"weights: {n_all:,} total, {n_emb:,} in the embedding (= output head), {n_body:,} in the layers")
print(f"weight bytes (bf16): {n_all * 2 / 1e9:.3f} GB")
# one forward pass reads every weight once; each weight used for one token costs a multiply and an add
flop_decode_tok = 2 * n_all          # layers + output head for every new token
flop_prefill_tok = 2 * n_body        # prefill keeps logits for the last token only (logits_to_keep=1)
print(f"FLOPs per decoded token (2 x all weights): {flop_decode_tok / 1e9:.3f} GFLOP")
print(f"FLOPs per prefill token (2 x layer weights, attention not counted): {flop_prefill_tok / 1e9:.3f} GFLOP")

# tiny example: one 896 x 896 matrix of the model
d = cfg.hidden_size
print(f"one {d}x{d} matrix: {d*d:,} weights, {d*d*2/1e6:.2f} MB, one token = {2*d*d/1e6:.2f} MFLOP -> {2*d*d/(d*d*2):.1f} FLOP/byte")
for b in (1, 4, 128):
    print(f"  batch {b:3d}: {2*d*d*b/1e6:8.2f} MFLOP for the same {d*d*2/1e6:.2f} MB -> {b:.0f} FLOP/byte")
del model

# ---- 2. the two roofs
BW = 262e9   # memory bandwidth of this GPU, measured for the published Part 1 (GB/s)
print(f"memory roof here: {BW/1e9:.0f} GB/s (published Part 1 measurement)")
print(f"H100 (NVIDIA spec): 989 TFLOP/s dense BF16, 3.35 TB/s -> ridge {989/3.35:.0f} FLOP/byte")
for b in (1, 4, 64, 128, 512):
    print(f"  H100 roofline at {b:3d} FLOP/byte: at most {min(989, 3.35*b):6.1f} TFLOP/s ({min(989, 3.35*b)/989:6.1%} of the roof)")

# ---- 3. the published measurements on the roofline
W = n_all * 2
points = [  # (label, tokens of work, ms) from the published Part 1 runs
    ("prefill 512 tokens", 512, 27.4, "prefill"),
    ("decode batch 1", 1, 10.0, "decode"), ("decode batch 4", 4, 11.1, "decode"),
    ("decode batch 8", 8, 11.9, "decode"), ("decode batch 32", 32, 15.1, "decode"),
    ("decode batch 64", 64, 19.7, "decode"), ("decode batch 128", 128, 29.3, "decode"),
]
print("point, FLOP/byte (weights only), achieved TFLOP/s, weight bytes per second")
for label, n, ms, kind in points:
    flops = n * (flop_prefill_tok if kind == "prefill" else flop_decode_tok)
    print(f"  {label:20s} {flops / W:6.1f} FLOP/byte  {flops / (ms/1e3) / 1e12:6.2f} TFLOP/s  {W / (ms/1e3) / 1e9:5.0f} GB/s"
          )
print(f"weight read floor: {W / BW * 1e3:.2f} ms per step")

# ---- 4. why the batch cannot grow forever: the notes (KV cache) of every sequence
cfg_head = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
kv_tok = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * cfg_head * 2
print(f"Qwen2.5-0.5B notes (KV cache) per token: {kv_tok:,} bytes = {kv_tok/1024:.0f} KiB")
for b, ctx in ((1, 512), (128, 512)):
    print(f"  batch {b:3d} x {ctx} tokens each: {b*ctx*kv_tok/2**20:,.0f} MiB of notes (weights: {W/2**20:,.0f} MiB)")
llama_kv = 2 * 32 * 8 * 128 * 2                  # Llama 3.1 8B config (Part 2)
llama_w = 8.03e9 * 2                              # 8.03B weights in bf16
free = 80e9 - llama_w                             # one 80 GB H100, before working memory
print(f"Llama 3.1 8B on an 80 GB H100: weights {llama_w/1e9:.1f} GB, {free/1e9:.1f} GB left (before working memory)")
for ctx in (4096, 131072):
    print(f"  conversations of {ctx:,} tokens that fit: at most {free / (llama_kv * ctx):.1f}"
          f"  ({llama_kv*ctx/2**20:,.0f} MiB each)")

# ---- 5. what one user feels at each batch size (published step times)
for b, ms in ((1, 10.0), (4, 11.1), (8, 11.9), (32, 15.1), (64, 19.7), (128, 29.3)):
    print(f"  batch {b:3d}: step {ms:4.1f} ms -> each user gets {1000/ms:5.1f} tokens/s, GPU total {b*1000/ms:6.0f} tokens/s")

# ---- 6. the step time as a fixed part plus a part per sequence (least-squares line, batch 8 to 128)
import numpy as np
bs = np.array([8, 32, 64, 128]); ms = np.array([11.9, 15.1, 19.7, 29.3])
slope, icpt = np.polyfit(bs, ms, 1)
print(f"step time ~ {icpt:.1f} ms + {slope:.3f} ms per sequence (fit on batch 8..128)")
for b in (1, 4, 8, 32, 64, 128):
    print(f"  batch {b:3d}: fixed {icpt:.1f} ms, per-sequence part {slope*b:5.1f} ms -> {icpt + slope*b:5.1f} ms predicted")
print(f"ceiling as batch grows without limit: {1000/slope:,.0f} tokens/s")
notes_read = 128 * 512 * kv_tok / BW * 1e3
print(f"reading the notes of 128 sequences x 512 tokens at 262 GB/s: {notes_read:.1f} ms of the 29.3 ms step")
worst = max(abs(icpt + slope * b - m) for b, m in ((4, 11.1), (8, 11.9), (32, 15.1), (64, 19.7), (128, 29.3)))
print(f"largest gap between the line and the measured step, batch 4..128: {worst:.2f} ms")

# ---- 7. small derived facts used in the text
print(f"arithmetic of one decode token on an H100 at full speed: {flop_decode_tok / 989e12 * 1e6:.1f} microseconds")
print(f"prefill 16 tokens 10.8 ms vs decode 1 token 10.0 ms: {10.8 / 10.0 - 1:.0%} more time for 16x the tokens")
print(f"read 512 vs write 512: 5158 / 27.4 = {5158 / 27.4:.0f}x")
print(f"batch 128 vs 1: step {29.3 / 10.0:.1f}x slower, tokens/s {4367 / 100:.0f}x more")
print(f"notes at batch 128 x 512: {128*512*kv_tok/1e9:.2f} GB; weights {W/1e9:.2f} GB")
print(f"batch 4: fixed part is {icpt / (icpt + slope*4):.0%} of the predicted step")
for b, ms, tps in ((1, 10.0, 100), (2, 10.6, 190), (4, 11.1, 361), (8, 11.9, 670), (16, 12.8, 1247), (32, 15.1, 2120), (64, 19.7, 3242), (128, 29.3, 4367)):
    print(f"  batch {b:3d}: {ms} ms, {tps:,} tokens/s total (published), {1000/ms:.0f} tokens/s per user")
print("row by row: 1->8 step +{:.0%}, 8 seqs = {:.1f}x tokens; 32->64 tokens x{:.2f}; 64->128 tokens x{:.2f}".format(11.9/10-1, 670/100, 3242/2120, 4367/3242))
print(f"prefill vs decode-batch-1 arithmetic rate: {13.38/0.0988:.0f}x")
