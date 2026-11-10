"""Part 6 experiment 1: the global KV cache per token of five DeepSeek generations, rebuilt from config.json.

The V4.1-Flash paper (arXiv 2609.19969, page 1, Figure 1b) plots the "global KV cache per token" of DeepSeek models.
This script reproduces every bar from the real config.json files plus the cache byte layouts that DeepSeek publishes:

  * DeepSeek LLM 67B (2023): grouped-query attention, BF16. Every layer stores K and V for 8 KV heads.
  * DeepSeek-V3 (2024): MLA, one 512-wide latent + one 64-wide RoPE key per layer, BF16.
  * DeepSeek-V3.2 (2025): the same MLA latent in FlashMLA's FP8 layout (656 B per layer)
    plus the DSA indexer key (128 FP8 values + one FP32 scale = 132 B per layer).
  * DeepSeek-V4-Flash / V4-Pro (2026): one 512-wide entry per COMPRESSED position. FlashMLA's V4 layout is 584 B
    (448 FP8 + 64 BF16 RoPE values + 8 B of scales). CSA layers (ratio 4) also keep an indexer key in MXFP4
    (128 values x 4 bits = 64 B + one 1-byte scale per 32 values = 4 B, so 68 B). HCA layers (ratio 128) keep no indexer.
  * DeepSeek-V4.1-Flash (2026): FP4 main entry 288 B (512 E2M1 values = 256 B + one E4M3 scale per 16 = 32 B) and the
    same 68 B indexer key, stored ONLY in the four layers that produce KV (kv_source_layer_ids); the other layers reuse it.

The sliding-window cache (the last 128 tokens of each layer) is a fixed amount per request, not per token, so it is
left out, exactly as in the paper's chart. The configs are in results/part6_configs/ (downloaded 2026-11-10).
Run:  ~/.venvs/attn/bin/python part6_kv_per_token.py
"""
import json, os
from common import Log, save, RESULTS

CFG = os.path.join(RESULTS, 'part6_configs')
log = Log('part6_kv_per_token')


def cfg(name):
    c = json.load(open(os.path.join(CFG, name + '.json')))
    return c.get('text_config', c)


# Bytes of one cached entry, from the published layouts (FlashMLA README, commit ba89a34; V4/V4.1 papers)
V32_MAIN = 512 * 1 + 4 * 4 + 64 * 2          # 512 FP8 + 4 FP32 scales + 64 BF16 RoPE = 656
V32_INDEX = 128 * 1 + 4                      # 128 FP8 indexer key + 1 FP32 scale = 132
V4_MAIN = 448 * 1 + 64 * 2 + 8               # 448 FP8 + 64 BF16 RoPE + 8-byte scale row = 584
FP4_INDEX = 128 // 2 + 128 // 32             # 128 MXFP4 values (64 B) + 4 one-byte scales = 68
V41_MAIN_FP4 = 512 // 2 + 512 // 16          # 512 E2M1 values (256 B) + 32 E4M3 scales = 288


def llm67b():
    c = cfg('deepseek-llm-67b-base')
    L, g, dh = c['num_hidden_layers'], c['num_key_value_heads'], c['hidden_size'] // c['num_attention_heads']
    per_layer = 2 * g * dh * 2                # K and V, g heads, dh numbers, 2 bytes (BF16)
    log(f'  67B: {L} layers x (2 x {g} KV heads x {dh} dims x 2 B = {per_layer:,} B)')
    return L * per_layer


def v3():
    c = cfg('DeepSeek-V3')
    L, dc, dr = c['num_hidden_layers'], c['kv_lora_rank'], c['qk_rope_head_dim']
    per_layer = (dc + dr) * 2
    log(f'  V3: {L} layers x (({dc} latent + {dr} RoPE) x 2 B = {per_layer:,} B)')
    return L * per_layer


def v32():
    c = cfg('DeepSeek-V3.2')
    L = c['num_hidden_layers']
    log(f'  V3.2: {L} layers x ({V32_MAIN} B MLA in FP8 + {V32_INDEX} B indexer key = {V32_MAIN + V32_INDEX} B)'
        f'  [index_topk {c["index_topk"]}]')
    return L * (V32_MAIN + V32_INDEX)


def v4(name):
    c = cfg(name)
    L = c['num_hidden_layers']
    ratios = c['compress_ratios'][:L]          # the last entry belongs to the MTP layer
    csa, hca, swa = ratios.count(4), ratios.count(128), ratios.count(0)
    b_csa = csa * (V4_MAIN + FP4_INDEX) / 4
    b_hca = hca * V4_MAIN / 128
    log(f'  {name}: {L} layers = {swa} sliding-window only + {csa} CSA (ratio 4) + {hca} HCA (ratio 128)')
    log(f'     CSA: {csa} x ({V4_MAIN} + {FP4_INDEX}) B / 4  = {b_csa:,.2f} B')
    log(f'     HCA: {hca} x {V4_MAIN} B / 128       = {b_hca:,.2f} B')
    return b_csa + b_hca, dict(csa=csa, hca=hca, swa=swa, csa_bytes=b_csa, hca_bytes=b_hca)


def v41():
    c = cfg('DeepSeek-V4.1-Flash')
    L = c['num_hidden_layers']
    ratios = c['compress_ratios'][:L]
    src = c['kv_source_layer_ids']
    total = 0
    log(f'  V4.1-Flash: {L} layers, compress ratios {sorted(set(ratios))}, KV stored only in layers {src}')
    for l in src:
        m = ratios[l]
        b = (V41_MAIN_FP4 + FP4_INDEX) / m
        total += b
        log(f'     layer {l:2d} ({"encoder" if l < L // 2 else "decoder"}, ratio {m}): ({V41_MAIN_FP4} + {FP4_INDEX}) B / {m} = {b:,.1f} B')
    return total


log('Global KV cache per token, rebuilt from config.json\n')
rows = []
b = llm67b(); rows.append(('DeepSeek LLM 67B', '2023-11', b))
b = v3(); rows.append(('DeepSeek-V3', '2024-12', b))
b = v32(); rows.append(('DeepSeek-V3.2', '2025-12', b))
bf, flash = v4('DeepSeek-V4-Flash'); rows.append(('DeepSeek-V4-Flash', '2026-04', bf))
bp, pro = v4('DeepSeek-V4-Pro')
b = v41(); rows.append(('DeepSeek-V4.1-Flash', '2026-09', b))

paper = {'DeepSeek LLM 67B': 389120, 'DeepSeek-V3.2': 48068, 'DeepSeek-V4-Flash': 3514, 'DeepSeek-V4.1-Flash': 890}
log('\nmodel                  date      bytes/token   paper Fig 1b   x smaller   GB at 128K   GB at 1M')
prev = None
for name, date, b in rows:
    ref = paper.get(name)
    ratio = f'{prev / b:6.1f}x' if prev else '      '
    log(f'{name:22s} {date}  {b:11,.0f}   {ref and f"{ref:,}" or "-":>12s}   {ratio:>9s}   '
        f'{b * 131072 / 1e9:9.2f}   {b * 1048576 / 1e9:8.2f}')
    prev = b
log(f'\nV4-Pro (not in the chart): {bp:,.0f} B per token; V3.2 / V4-Pro = {rows[2][2] / bp:.2f}x, '
    f'V3.2 / V4-Flash = {rows[2][2] / bf:.2f}x  (V4 paper Figure 1: "9.5x smaller", "13.7x smaller")')
log(f'67B / V3.2 = {rows[0][2] / rows[2][2]:.1f}x  (V4.1 paper Figure 1b: "8.1x smaller")')
log(f'67B / V4.1-Flash = {rows[0][2] / rows[-1][2]:.1f}x  (V4.1 paper: "437-fold");  '
    f'V4-Flash / V4.1-Flash = {bf / rows[-1][2]:.2f}x  ("approximately 4-fold")')
log(f'V2-style check: 67B / V3 at the same BF16 precision = {rows[0][2] / rows[1][2]:.2f}x')

# The V4 paper's other yardstick: a BF16 GQA model with 8 KV heads of 128 dims and the same number of layers
for nm, b, L in (('V4-Flash', bf, 43), ('V4-Pro', bp, 61)):
    gqa8 = 2 * 8 * 128 * 2 * L
    log(f'{nm}: BF16 GQA8 baseline with {L} layers = {gqa8:,} B per token; V4 / baseline = {b / gqa8 * 100:.1f}%  (paper: "approximately 2%")')
# The sliding-window cache is per request, not per token: 128 recent tokens in every layer
swa = 128 * 43 * V4_MAIN
log(f'V4-Flash sliding-window cache: 128 tokens x 43 layers x {V4_MAIN} B = {swa:,} B = {swa / 1e6:.1f} MB per request, whatever the length')

# Engram: how many parameters sit in the tables, and how many are read per token
c = cfg('DeepSeek-V4.1-Flash')
rows_tab = sum(c['engram_num_embeddings'])
params = rows_tab * c['engram_head_dim']
orders = c['engram_max_ngram_size'] - 1            # n-gram orders 2, 3, 4
tables = len(c['engram_layer_ids']) * orders * c['engram_n_heads']
read = tables * c['engram_head_dim']
log(f'\nEngram (V4.1-Flash): layers {c["engram_layer_ids"]}, {orders} n-gram orders x {c["engram_n_heads"]} hash heads per module')
log(f'  table rows  = {c["engram_num_embeddings"][0]:,} + {c["engram_num_embeddings"][1]:,} = {rows_tab:,}')
log(f'  parameters  = {rows_tab:,} rows x {c["engram_head_dim"]} dims = {params / 1e9:.1f}B   (paper: "196B Engram parameters")')
log(f'  per table   = {rows_tab / tables:,.0f} rows on average   (paper: "approximately 16M entries")')
log(f'  read per token = {tables} rows x {c["engram_head_dim"]} = {read:,} numbers = {read:,} bytes in FP8 '
    f'({read / params * 100:.7f}% of the tables)')

# FP4 experts: a rough size check of the V4-Flash checkpoint
c = cfg('DeepSeek-V4-Flash')
routed = c['num_hidden_layers'] * c['n_routed_experts'] * 3 * c['hidden_size'] * c['moe_intermediate_size']
fp4_gb = routed * (4 + 8 / 32) / 8 / 1e9
rest = 284e9 - routed
log(f'\nV4-Flash routed experts: {c["num_hidden_layers"]} layers x {c["n_routed_experts"]} x 3 x {c["hidden_size"]} x '
    f'{c["moe_intermediate_size"]} = {routed / 1e9:.1f}B parameters')
log(f'  in FP4 with one 1-byte scale per 32 values: {fp4_gb:.1f} GB;  the other ~{rest / 1e9:.0f}B in FP8: ~{rest / 1e9:.0f} GB;'
    f'  estimate {fp4_gb + rest / 1e9:.0f} GB  (published files: 159.6 GB)')
log(f'  the same 284B in BF16 would be {284e9 * 2 / 1e9:.0f} GB')

save('part6_kv_per_token', dict(rows=[dict(name=n, date=d, bytes=b) for n, d, b in rows], v4_pro=bp,
                                flash=flash, pro=pro, engram_params=params, engram_read=read,
                                layouts=dict(v32_main=V32_MAIN, v32_index=V32_INDEX, v4_main=V4_MAIN,
                                             fp4_index=FP4_INDEX, v41_main_fp4=V41_MAIN_FP4)))
