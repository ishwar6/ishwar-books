"""Every new number in Part 2 of "LLM Inference from the Ground Up" (writings/llm-inference-2-kv-cache).

1. A tiny attention lookup by hand (3 earlier tokens, 2-number keys and values), checked against PyTorch.
2. The KV cache of a real model, Qwen2.5-0.5B: bytes per token measured from the cache tensors,
   compared with the config formula.
3. The size formula for the models in the article, and what GQA, MQA and an 8-bit cache would give for Llama 3.1 8B.
4. The bytes one decode step must read on an H100 (3.35 TB/s, NVIDIA spec): weights and a 128K-token cache.
5. The toy memory examples for the three kinds of waste (a row of 24 slots).

Run: ~/.venvs/attn/bin/python p2_kv_numbers.py   (saves results/p2_kv_numbers_stdout.txt and .json)"""
import json, math, os, sys, io

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
os.makedirs(RES, exist_ok=True)
OUT = {}
buf = io.StringIO()


def say(*a):
    s = ' '.join(str(x) for x in a)
    print(s)
    buf.write(s + '\n')


# ---------------------------------------------------------------- 1. attention by hand
say('== 1. Attention as a lookup: a query, three keys, three values')
tokens = ['The', 'cat', 'sat']
K = [[0.0, 1.0], [3.0, 0.0], [1.0, 1.0]]          # one key per earlier token (2 numbers each)
V = [[1.0, 0.0], [0.0, 4.0], [2.0, 2.0]]          # one value per earlier token
q = [1.0, 0.0]                                    # the query of the new token "it"
d = len(q)
scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in K]          # dot products
scaled = [s / math.sqrt(d) for s in scores]                          # divide by sqrt(d)
ex = [math.exp(s) for s in scaled]
w = [e / sum(ex) for e in ex]                                         # softmax: weights that sum to 1
out = [sum(w[j] * V[j][i] for j in range(3)) for i in range(2)]      # weighted mix of values
for t, k, s, sc, wj in zip(tokens, K, scores, scaled, w):
    say(f'  {t:4s} key {k}  q.k = {s:.0f}  /sqrt(2) = {sc:.2f}  weight = {wj:.3f}')
say(f'  answer = {w[0]:.3f}*{V[0]} + {w[1]:.3f}*{V[1]} + {w[2]:.3f}*{V[2]} = [{out[0]:.2f}, {out[1]:.2f}]')
OUT['toy'] = dict(scores=scores, scaled=[round(x, 2) for x in scaled], weights=[round(x, 3) for x in w],
                  out=[round(x, 2) for x in out])
try:
    import torch
    t = torch.nn.functional.scaled_dot_product_attention(torch.tensor([[q]]), torch.tensor([K]), torch.tensor([V]))
    say('  PyTorch scaled_dot_product_attention gives', [round(x, 2) for x in t[0, 0].tolist()])
except Exception as e:  # torch not installed
    say('  (PyTorch check skipped:', e, ')')

# ---------------------------------------------------------------- 2. the real cache of Qwen2.5-0.5B
say('\n== 2. The KV cache of a real model: Qwen2.5-0.5B, bfloat16')
try:
    import torch
    from transformers import AutoModelForCausalLM, DynamicCache
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-0.5B', dtype=torch.bfloat16).to(dev).eval()
    cfg = model.config
    head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
    say(f'  config: layers {cfg.num_hidden_layers}, query heads {cfg.num_attention_heads}, '
        f'KV heads {cfg.num_key_value_heads}, head_dim {head_dim}, device {dev}')
    with torch.inference_mode():
        sizes = {}
        for n in (100, 200):
            cache = DynamicCache()
            model(input_ids=torch.randint(0, cfg.vocab_size, (1, n), device=dev), past_key_values=cache)
            layers = cache.layers if hasattr(cache, 'layers') else None
            if layers is not None:
                ks = [l.keys for l in layers]; vs = [l.values for l in layers]
            else:
                ks, vs = cache.key_cache, cache.value_cache
            total = sum(x.numel() * x.element_size() for x in ks + vs)
            sizes[n] = total
            say(f'  {n} tokens: {len(ks)} layers, key tensor of layer 0 {tuple(ks[0].shape)} {ks[0].dtype}, '
                f'cache {total:,} bytes')
    per_tok = (sizes[200] - sizes[100]) / 100
    formula = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2
    say(f'  measured: {per_tok:,.0f} bytes per token; formula 2 x {cfg.num_hidden_layers} x '
        f'{cfg.num_key_value_heads} x {head_dim} x 2 = {formula:,} bytes')
    OUT['qwen'] = dict(per_token=per_tok, formula=formula, shape=list(ks[0].shape))
except Exception as e:
    say('  (model run skipped:', e, ')')

# ---------------------------------------------------------------- 3. the formula for real configs
say('\n== 3. KV bytes per token = 2 x layers x KV heads x head_dim x bytes')
KiB, GiB = 1024, 1024 ** 3
CTX = 131072
MODELS = {  # from each model's config.json
    'Qwen2.5-0.5B': (24, 2, 64), 'Llama 3.1 8B': (32, 8, 128),
    'Llama 3.1 70B': (80, 8, 128), 'Llama 2 7B': (32, 32, 128),
}
OUT['models'] = {}
for name, (L, H, D) in MODELS.items():
    b = 2 * L * H * D * 2
    say(f'  {name:14s} {L:3d} layers {H:3d} KV heads {D:4d} dims: {b / KiB:6.0f} KiB/token, '
        f'{b * CTX / GiB:5.1f} GiB for {CTX:,} tokens')
    OUT['models'][name] = dict(kib=b / KiB, gib_128k=b * CTX / GiB)
say('  Llama 3.1 8B with other attention layouts (32 layers, head_dim 128, 2 bytes):')
for name, H in (('MHA, 32 KV heads', 32), ('GQA, 8 KV heads', 8), ('MQA, 1 KV head', 1)):
    b = 2 * 32 * H * 128 * 2
    say(f'    {name:17s} {b / KiB:4.0f} KiB/token, {b * CTX / GiB:5.1f} GiB for 128K tokens')
b8 = 2 * 32 * 8 * 128 * 1
say(f'    GQA + 8-bit cache  {b8 / KiB:4.0f} KiB/token, {b8 * CTX / GiB:5.1f} GiB for 128K tokens')
qwen16 = 16 * 8192 * 2 * 24 * 2 * 64 * 2
say(f'  Qwen2.5-0.5B, 16 sequences x 8,192 tokens: {qwen16 / GiB:.2f} GiB of cache')

# ---------------------------------------------------------------- 4. bytes read per decode step on an H100
say('\n== 4. One decode step of Llama 3.1 8B on an H100 must read (3.35 TB/s)')
BW = 3.35e12
weights = 8.03e9 * 2                      # 8.03 billion parameters x 2 bytes
cache = 2 * 32 * 8 * 128 * 2 * CTX
say(f'  weights {weights / 1e9:.1f} GB -> at least {weights / BW * 1000:.1f} ms')
say(f'  one 128K-token cache {cache / 1e9:.1f} GB ({cache / GiB:.0f} GiB) -> at least {cache / BW * 1000:.1f} ms more')
fit = int((80e9 - weights) // cache)
say(f'  on an 80 GB H100, after the weights: room for {fit} such conversations at full length '
    f'({(80e9 - weights) / 1e9:.1f} GB / {cache / 1e9:.1f} GB)')
OUT['h100'] = dict(weights_ms=weights / BW * 1000, cache_ms=cache / BW * 1000)

# ---------------------------------------------------------------- 5. the toy memory for the three wastes
say('\n== 5. Toy GPU memory: a row of 24 slots, one slot = the K and V of one token (all layers)')
# (1) reserved: request A books 8 slots; prompt 3, it will write 5 tokens in total; look after 2 written
A_book, A_prompt, A_final, A_written = 8, 3, 8, 2
held = A_prompt + A_written
say(f'  reserved: A books {A_book}, prompt {A_prompt}, wrote {A_written} -> holds {held}, '
    f'{A_final - held} reserved for tokens still to come')
# (2) internal fragmentation: request B books 8; prompt 2, answer ends after 2 tokens
B_book, B_prompt, B_answer = 8, 2, 2
say(f'  internal: B books {B_book}, uses {B_prompt + B_answer}, {B_book - B_prompt - B_answer} never used')
# (3) external fragmentation: A=8, B=4, C=6, D=6 fill 24; B and D finish; E needs 7
row = 'A' * 8 + 'B' * 4 + 'C' * 6 + 'D' * 6
after = row.replace('B', '.').replace('D', '.')
gaps = [len(g) for g in after.replace('A', ' ').replace('C', ' ').split() if g]
say(f'  external: {row} -> B, D finish -> {after}; free {after.count(".")} in gaps {gaps}; '
    f'E needs 7 in one piece: {"fits" if max(gaps) >= 7 else "refused"}')
OUT['toy_mem'] = dict(gaps=gaps)

# (4) all three at one moment: A books 8 (3 prompt, 2 written, 2 still to come, 1 never used),
#     a 3-slot gap, B books 8 (2 prompt, 1 written, 1 still to come, 4 never used), 5 free slots at the end;
#     the next request needs 6 slots in one piece
cells = 'PPPWWRRI' + '___' + 'PPWRIIII' + '_____'
tok = cells.count('P') + cells.count('W'); res = cells.count('R'); internal = cells.count('I')
free = cells.count('_')
say(f'  all three: {cells} ({len(cells)} slots): tokens {tok}, reserved {res}, never used {internal}, '
    f'free {free} in gaps of 3 and 5 (next request needs 6: refused); holding tokens {tok / len(cells):.0%}')
OUT['toy_all'] = dict(tokens=tok, reserved=res, internal=internal, free=free, share=tok / len(cells))

open(os.path.join(RES, 'p2_kv_numbers_stdout.txt'), 'w').write(buf.getvalue())
json.dump(OUT, open(os.path.join(RES, 'p2_kv_numbers.json'), 'w'), indent=1)
