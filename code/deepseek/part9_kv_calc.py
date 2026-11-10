"""Part 9: DeepInfer's KV-cache arithmetic for DeepSeek-V4.1-Flash, recomputed by hand.

    ~/.venvs/attn/bin/python part9_kv_calc.py

This is analysis, not a run of DeepInfer. It re-implements, in a few lines, the sizing rules of
deepinfer/worker/cache_engine.py (KVCacheManager.process, StateBlock) and
deepinfer/hugging_face/config.py (_source_map, _build_attn_module_list, _n_prefill_layer) at commit
a679bb8, and applies them to the real V4.1-Flash config.json saved in results/part6_configs/.
Output: results/part9_kv_calc.json and results/part9_kv_calc_stdout.txt.
"""
import json, math, os
from common import Log, save, RESULTS

log = Log('part9_kv_calc')
cfg = json.load(open(os.path.join(RESULTS, 'part6_configs/DeepSeek-V4.1-Flash.json')))['text_config']

BLOCK = 128                     # model_metadata_utils.py: resolved_tokens_per_block -> 128
NEXTN = 5                       # environ.py: DEEPINFER_USE_NEXTN_DECODE default 5
BACKTRACK = max(0, NEXTN + 1)   # environ.py: DEEPINFER_MAX_BACKTRACKING_TOKENS = max(env, nextn + 1)
MAIN_FP4 = 256 + 32             # cache_common.py V41_FP4_FORMAT: 512 e2m1 values (256 B) + one 1-byte scale per 16
MAIN_FP8 = 512 + 16             # cache_common.py V41_FORMAT: 512 e4m3 values + one UE8M0 scale per 32 = 528 B
IDX = cfg['index_head_dim'] // 2 + cfg['index_head_dim'] // 32   # cache_engine.py _get_qk_topk_bytes: FP4 + UE8M0


def pad(x, a):
    return (x + a - 1) // a * a


def source_map(ids, n):  # hugging_face/config.py _source_map
    ids = sorted(ids)
    return {l: max(s for s in ids if s <= l) for l in range(ids[0], n)}


n = cfg['num_hidden_layers']
ratios = cfg['compress_ratios']
nextn_layers = cfg['num_nextn_predict_layers']
kv_map = source_map(cfg['kv_source_layer_ids'], n)
idx_map = source_map(cfg.get('index_k_source_layer_ids', cfg['kv_source_layer_ids']), n)
n_prefill = next(i for i, r in enumerate(ratios) if r == 1)          # first ratio-1 layer: encoder/decoder split
window = cfg['sliding_window']
swa_tokens = window + BACKTRACK                                       # get_inference_swa_block_size

log(f'V4.1-Flash: {n} layers + {nextn_layers} draft (DSpark) layers; sliding window {window}; block = {BLOCK} tokens')
log(f'compress ratio per layer: {ratios[:n]}')
log(f'encoder (prefill) layers: 0..{n_prefill - 1}; decoder (YOCO consumer) layers: {n_prefill}..{n - 1}')

# ---- paged block pool: one block covers 128 tokens, all layers stacked ----
rows, block_bytes = [], 0
for l in range(n):
    r = ratios[l]
    entry = {'layer': l, 'ratio': r, 'main': 0, 'index': 0, 'owner_main': kv_map.get(l, l), 'owner_index': idx_map.get(l, l)}
    if r > 0:  # compressed + sparse layers carry index_topk > 0 in _compress_spec
        if idx_map.get(l, l) == l:
            entry['index'] = pad(BLOCK // r * IDX, 32)
        if kv_map.get(l, l) == l:
            entry['main'] = pad(BLOCK // r * MAIN_FP4, 32)
    block_bytes += entry['main'] + entry['index']
    rows.append(entry)
owners = [e for e in rows if e['main'] or e['index']]
log('\nLayers that own a slice of the paged block (all others reuse an earlier layer\'s cache):')
for e in owners:
    log(f"  layer {e['layer']:2d}  ratio {e['ratio']}  main {BLOCK // e['ratio']:3d} entries x {MAIN_FP4} B = {e['main']:6,d} B"
        f"   indexer {BLOCK // e['ratio']:3d} x {IDX} B = {e['index']:5,d} B")
align = math.lcm(128, MAIN_FP4)                                        # KVCacheManager.finalize
block_aligned = pad(block_bytes, align)
log(f'\nbytes per 128-token block: {block_bytes:,} (aligned to {align}: {block_aligned:,})')
log(f'=> {block_bytes / BLOCK:.1f} bytes per token of context ({block_aligned / BLOCK:.1f} after alignment)')

naive = n * MAIN_FP8
log(f'\nFor scale: one 528-byte FP8 entry per token in every one of the {n} layers would be {naive:,} B/token, '
    f'{naive / (block_bytes / BLOCK):.0f}x more.')

# ---- per-sequence state slot (fixed size, independent of context length) ----
swa_layers = n + nextn_layers                                         # every runtime layer has sink_mode="swa"
swa_store = sum(1 for l in range(n) if l < n_prefill)                 # _is_dropped_swa_layer: layer_id >= n_prefill
swa_drop = swa_layers - swa_store
swa_per_layer = pad(swa_tokens * MAIN_FP8, 32)
ring = {r: r + max(0, BACKTRACK - 1) for r in set(ratios[:n]) if r > 1}  # get_inference_compress_block_size
comp_state = 0
for l in range(n):
    r = ratios[l]
    if r > 1 and kv_map.get(l, l) == l:                               # owns main compression state; use_kproj=True -> no index state
        comp_state += ring[r] * 2 * (cfg['head_dim'] + cfg['qk_rope_head_dim']) * 4
state = pad(swa_layers * swa_per_layer + comp_state, math.lcm(32, MAIN_FP8))
to_disk = swa_store * swa_per_layer
log(f'\nPer-sequence state slot: {swa_layers} SWA rings x {swa_tokens} tokens x {MAIN_FP8} B = {swa_layers * swa_per_layer:,} B, '
    f'+ compression state {comp_state:,} B  => {state:,} B ({state / 2**20:.2f} MiB)')
log(f'Part of the slot written to the KV store: {swa_store} layers x {swa_tokens} x {MAIN_FP8} = {to_disk:,} B '
    f'(the Q&A says state_size_to_disk = 1,415,040 B)')

# ---- what fits: a worked example with a round, hypothetical budget ----
budget = 100 * 2**30
for max_seqs in (64, 256):
    slots = int(max_seqs * 1.2) + 1                                   # get_num_state_slots, ratio 1.2
    reserve = 2 * slots * state                                       # base.py: state size x2 (store + copy buffer)
    blocks = (budget - reserve) // block_aligned
    log(f'\nHypothetical 100 GiB left for cache, max_num_seqs={max_seqs}: state reserve {reserve / 2**30:.2f} GiB '
        f'({slots} slots x2), {blocks:,} blocks = {blocks * BLOCK / 1e6:.1f} M tokens of context')
    log(f'  = {blocks * BLOCK // max_seqs:,} tokens per sequence if all {max_seqs} slots are full')

ctx = [8192, 32768, 131072, 1048576]
per_seq = [{'context': c, 'blocks_MiB': math.ceil(c / BLOCK) * block_aligned / 2**20, 'state_MiB': state / 2**20} for c in ctx]
log('\nOne sequence, cache in HBM (blocks + its state slot):')
for p in per_seq:
    log(f"  {p['context']:>9,} tokens: blocks {p['blocks_MiB']:8.1f} MiB + state {p['state_MiB']:.2f} MiB")

save('part9_kv_calc', dict(block_tokens=BLOCK, block_bytes=block_bytes, block_bytes_aligned=block_aligned,
                           bytes_per_token=block_bytes / BLOCK, owners=owners, naive_bytes_per_token=naive,
                           state_bytes=state, state_to_disk=to_disk, swa_layers=swa_layers, swa_store=swa_store,
                           swa_tokens=swa_tokens, comp_state=comp_state, per_seq=per_seq, n_prefill=n_prefill,
                           ratios=ratios[:n]))
