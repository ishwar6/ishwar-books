"""Part 1 experiment: read the real config.json of every main DeepSeek language model and print one table.

For each repo it downloads https://huggingface.co/deepseek-ai/<repo>/raw/main/config.json (a few KB, cached in
results/configs/), then counts the parameters from the config fields alone:

  * dense models (DeepSeek LLM, Coder): embeddings + attention + SwiGLU FFN + norms
  * DeepSeekMoE / V2 / V3 family: embeddings + attention (MHA or MLA) + dense FFN layers + experts + routers + norms
  * V4 and V4.1: the attention changed too much for a short formula, so the script counts only the experts
    (routed + shared) and prints the published totals next to them.

"Active" counts what one token touches: every attention block, the dense layers, top-k routed experts plus the
shared experts, the routers, the embedding row and the output head. The MTP module of V3 is left out, as in the paper.
Run:  ~/.venvs/attn/bin/python part1_configs.py
"""
import json, os, urllib.request
from common import Log, save, RESULTS

CACHE = os.path.join(RESULTS, 'configs')
log = Log('part1_configs')

# repo, release date, published total / active (as the paper or model card states them), published context
MODELS = [
    ('deepseek-coder-33b-base', '2023-11', '33B', 'dense', '16K'),
    ('deepseek-llm-67b-base', '2023-11', '67B', 'dense', '4K'),
    ('deepseek-moe-16b-base', '2024-01', '16.4B', '2.8B', '4K'),
    ('DeepSeek-V2', '2024-05', '236B', '21B', '128K'),
    ('DeepSeek-V2-Lite', '2024-05', '15.7B', '2.4B', '32K'),
    ('DeepSeek-Coder-V2-Instruct', '2024-06', '236B', '21B', '128K'),
    ('DeepSeek-V2.5', '2024-09', '236B', '21B', '128K'),
    ('DeepSeek-V3', '2024-12', '671B', '37B', '128K'),
    ('DeepSeek-R1', '2025-01', '671B', '37B', '128K'),
    ('DeepSeek-V3.1', '2025-08', '671B', '37B', '128K'),
    ('DeepSeek-V3.2', '2025-12', '671B', '37B', '128K'),
    ('DeepSeek-V4-Flash', '2026-04', '284B', '13B', '1M'),
    ('DeepSeek-V4-Pro', '2026-04', '1.6T', '49B', '1M'),
    ('DeepSeek-V4.1-Flash', '2026-09', '552B + 196B Engram', '8B / 16B', '1M'),
]


def config(repo):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, repo + '.json')
    if not os.path.exists(path):
        url = f'https://huggingface.co/deepseek-ai/{repo}/raw/main/config.json'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        open(path, 'wb').write(urllib.request.urlopen(req, timeout=30).read())
    c = json.load(open(path))
    return c.get('text_config', c), os.path.getsize(path)


def config_top(repo):
    return json.load(open(os.path.join(CACHE, repo + '.json')))


def attention(c):
    """Parameters of one attention block, read from the config."""
    d, H = c['hidden_size'], c['num_attention_heads']
    if c.get('kv_lora_rank'):                                   # Multi-head Latent Attention (V2, V3)
        nope, rope, vd, r_kv, r_q = c['qk_nope_head_dim'], c['qk_rope_head_dim'], c['v_head_dim'], c['kv_lora_rank'], c.get('q_lora_rank')
        q = d * r_q + r_q + r_q * H * (nope + rope) if r_q else d * H * (nope + rope)
        kv = d * (r_kv + rope) + r_kv + r_kv * H * (nope + vd)
        return q + kv + H * vd * d
    hd = d // H                                                 # standard multi-head or grouped-query attention
    kvh = c.get('num_key_value_heads') or H
    return d * H * hd + 2 * d * kvh * hd + H * hd * d


def count(c):
    """Total and active parameters for the dense, DeepSeekMoE, V2 and V3 configs."""
    d, L, V = c['hidden_size'], c['num_hidden_layers'], c['vocab_size']
    emb = V * d * (1 if c.get('tie_word_embeddings') else 2)
    attn, norms = attention(c), 2 * d
    if not c.get('n_routed_experts'):                           # dense
        ffn = 3 * d * c['intermediate_size']
        total = emb + L * (attn + ffn + norms) + d
        return total, total
    k_dense = c.get('first_k_dense_replace', 0)
    n_moe = L - k_dense
    dense_ffn = 3 * d * c['intermediate_size']
    expert = 3 * d * c['moe_intermediate_size']
    E, S, k = c['n_routed_experts'], c['n_shared_experts'], c['num_experts_per_tok']
    router = E * d + (E if c.get('topk_method') == 'noaux_tc' else 0)
    common = emb + L * (attn + norms) + k_dense * dense_ffn + n_moe * router + d
    total = common + n_moe * (E + S) * expert
    active = common + n_moe * (k + S) * expert
    return total, active


def experts_only(c):
    """V4 / V4.1: every layer is a MoE layer; count only the expert weights."""
    expert = 3 * c['hidden_size'] * c['moe_intermediate_size']
    L, E, S, k = c['num_hidden_layers'], c['n_routed_experts'], c['n_shared_experts'], c['num_experts_per_tok']
    return L * (E + S) * expert, L * (k + S) * expert


def B(n):
    return f'{n / 1e12:.2f}T' if n >= 1e12 else f'{n / 1e9:.1f}B'


def ctx(n):
    return f'{n // 1048576}M' if n >= 1048576 else f'{n // 1024}K'


rows = []
log('Real config.json files from huggingface.co/deepseek-ai, counted from the config fields alone')
log('')
hdr = f'{"model":<27}{"date":<9}{"layers":>6}{"d":>6}{"experts":>13}{"ctx(config)":>12}{"total(count)":>14}{"active(count)":>14}   published'
log(hdr)
log('-' * len(hdr))
for repo, date, pub_t, pub_a, pub_ctx in MODELS:
    c, size = config(repo)
    E = c.get('n_routed_experts')
    experts = f'{E}+{c["n_shared_experts"]} top-{c["num_experts_per_tok"]}' if E else 'dense'
    arch = (c.get('architectures') or config_top(repo).get('architectures') or ['?'])[0]
    if 'compress_ratios' in c:                                 # V4 and V4.1 configs
        t, a = experts_only(c)
        tc, ac = f'>{B(t)}*', f'>{B(a)}*'
    else:
        t, a = count(c)
        tc, ac = B(t), B(a)
    mpe = c['max_position_embeddings']
    log(f'{repo:<27}{date:<9}{c["num_hidden_layers"]:>6}{c["hidden_size"]:>6}{experts:>13}{ctx(mpe):>12}{tc:>14}{ac:>14}   {pub_t} / {pub_a}, {pub_ctx}')
    rows.append(dict(repo=repo, date=date, arch=arch, layers=c['num_hidden_layers'], d=c['hidden_size'],
                     routed=E, shared=c.get('n_shared_experts'), topk=c.get('num_experts_per_tok'),
                     max_pos=mpe, total=t, active=a, experts_only='compress_ratios' in c, config_bytes=size,
                     published_total=pub_t, published_active=pub_a, published_context=pub_ctx))
log('')
log('* V4 and V4.1 rows count the expert weights only (a lower bound); attention, embeddings and the rest are not counted.')
log('  V4.1-Flash: the 9.9B is all 40 layers; DeepSeek says prefill uses 8B because most prompt tokens skip the decoder half.')
log('ctx(config) is max_position_embeddings; the model cards state the context the model was trained and tested for.')
log('')
v3 = next(r for r in rows if r['repo'] == 'DeepSeek-V3')
log(f'DeepSeek-V3 share of weights one token uses: {v3["active"] / v3["total"] * 100:.1f}%  '
    f'({B(v3["active"])} of {B(v3["total"])})')
f = next(r for r in rows if r['repo'] == 'DeepSeek-V4-Flash')
log(f'DeepSeek-V4-Flash experts alone: {B(f["total"])}, which is {f["total"] / 284e9 * 100:.0f}% of the published 284B')
log(f'Config files read: {len(rows)}, total size {sum(r["config_bytes"] for r in rows) / 1024:.1f} KB')
save('part1_configs', rows)
