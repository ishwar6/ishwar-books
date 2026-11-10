"""Part 2: count DeepSeek-V3's parameters from its real config.json, two ways.

Method 1: our own formula, one line per weight matrix (names from the V3 reference model.py).
Method 2: build DeepseekV3ForCausalLM from the same config on PyTorch's "meta" device (shapes only, no memory)
and count what transformers creates. Both must agree.
Then: active parameters per token, the MTP module, the 685B checkpoint on Hugging Face, and the same check for
the real DeepSeek-V2-Lite config (published: 15.7B total, 2.4B active).
Input: results/part2_v3_config.json, fetched from https://huggingface.co/deepseek-ai/DeepSeek-V3/raw/main/config.json
"""
import glob, json, os
import torch
from transformers import DeepseekV2Config, DeepseekV2ForCausalLM, DeepseekV3Config, DeepseekV3ForCausalLM
from common import Log, save

log = Log('part2_params')
HERE = os.path.dirname(os.path.abspath(__file__))
cfg = json.load(open(os.path.join(HERE, 'results/part2_v3_config.json')))


def count(c):
    """Method 1: parameters of a DeepSeek-V2/V3 model, bucket by bucket, straight from the config."""
    d, V, L = c['hidden_size'], c['vocab_size'], c['num_hidden_layers']
    nh, nope, rope, vd = c['num_attention_heads'], c['qk_nope_head_dim'], c['qk_rope_head_dim'], c['v_head_dim']
    qr, kvr = c['q_lora_rank'], c['kv_lora_rank']
    # attention (MLA): query down + norm + up, KV down (latent + RoPE key) + norm + up, output
    q = d * nh * (nope + rope) if qr is None else d * qr + qr + qr * nh * (nope + rope)
    attn = q + d * (kvr + rope) + kvr + kvr * nh * (nope + vd) + nh * vd * d
    norms = 2 * d                                       # RMSNorm before attention and before the FFN
    dense_ffn = 3 * d * c['intermediate_size']          # SwiGLU: gate, up and down matrices
    expert = 3 * d * c['moe_intermediate_size']         # one small expert, also SwiGLU
    E, S, K = c['n_routed_experts'], c['n_shared_experts'], c['num_experts_per_tok']
    bias = E if c.get('topk_method') == 'noaux_tc' else 0
    router = E * d + bias                               # one centroid per expert (+ V3's balancing bias)
    nd = c['first_k_dense_replace']
    nm = L - nd
    b = {
        'embedding': V * d,
        'output head': V * d,
        'attention (MLA), all layers': L * attn,
        'dense FFN, first layers': nd * dense_ffn,
        'routed experts': nm * E * expert,
        'shared experts': nm * S * expert,
        'routers (incl. bias)': nm * router,
        'RMSNorm weights': L * norms + d,
    }
    active = {
        'embedding': V * d,
        'output head': V * d,
        'attention (MLA), all layers': L * attn,
        'dense FFN, first layers': nd * dense_ffn,
        'routed experts': nm * K * expert,
        'shared experts': nm * S * expert,
        'routers (incl. bias)': nm * router,
        'RMSNorm weights': L * norms + d,
    }
    return b, active, dict(attn=attn, dense_ffn=dense_ffn, expert=expert, router=router, nd=nd, nm=nm)


def library_count(cls_cfg, cls_model, c):
    """Method 2: what transformers builds from the same config (meta device: shapes only)."""
    c = {k: v for k, v in c.items() if k != 'quantization_config'}
    with torch.device('meta'):
        m = cls_model(cls_cfg(**c))
    params = sum(p.numel() for p in m.parameters())
    bias = sum(b.numel() for n, b in m.named_buffers() if 'e_score_correction_bias' in n)
    return params + bias, m


B = lambda n: f'{n / 1e9:8.3f}B'

log('== DeepSeek-V3 config.json (the fields that decide the size) ==')
for k in ['hidden_size', 'num_hidden_layers', 'first_k_dense_replace', 'num_attention_heads', 'q_lora_rank',
          'kv_lora_rank', 'qk_nope_head_dim', 'qk_rope_head_dim', 'v_head_dim', 'intermediate_size',
          'moe_intermediate_size', 'n_routed_experts', 'n_shared_experts', 'num_experts_per_tok', 'n_group',
          'topk_group', 'vocab_size', 'num_nextn_predict_layers']:
    log(f'  {k:26s} {cfg[k]}')

b, act, per = count(cfg)
total = sum(b.values())
log('\n== Per-piece sizes ==')
log(f'  MLA attention, one layer     {per["attn"]:>15,}')
log(f'  dense SwiGLU FFN, one layer  {per["dense_ffn"]:>15,}')
log(f'  one expert (SwiGLU)          {per["expert"]:>15,}')
log(f'  router, one MoE layer        {per["router"]:>15,}')
log(f'  dense layers: {per["nd"]}, MoE layers: {per["nm"]}')

log('\n== Method 1: our formula, total parameters ==')
for k, v in b.items():
    log(f'  {k:30s} {B(v)}  {100 * v / total:6.2f}%')
log(f'  {"TOTAL main model":30s} {B(total)}   ({total:,})')

lib, model = library_count(DeepseekV3Config, DeepseekV3ForCausalLM, cfg)
log('\n== Method 2: transformers DeepseekV3ForCausalLM on the meta device ==')
log(f'  parameters + balancing bias   {B(lib)}   ({lib:,})')
log(f'  equal to Method 1: {lib == total}')

atot = sum(act.values())
log('\n== Active parameters for one token ==')
for k, v in act.items():
    log(f'  {k:30s} {B(v)}')
log(f'  {"TOTAL active":30s} {B(atot)}')
log(f'  without the input embedding (a table lookup, not a matrix multiply): {B(atot - act["embedding"])}')
log(f'  share of all parameters used per token: {100 * atot / total:.1f}%')

d = cfg['hidden_size']
mtp_unique = 2 * d + d * 2 * d + (per['attn'] + 2 * d + per['router'] + (cfg['n_routed_experts'] + cfg['n_shared_experts']) * per['expert']) + d
emb = cfg['vocab_size'] * d
log('\n== The MTP module (one extra layer, num_nextn_predict_layers = 1) ==')
log(f'  enorm + hnorm + eh_proj (7168 x 14336) + one full MoE layer + final norm = {B(mtp_unique)}')
log(f'  plus its stored copies of the embedding and output head            = {B(2 * emb)}')
log(f'  checkpoint on Hugging Face: {B(total)} + {B(mtp_unique)} + {B(2 * emb)} = {B(total + mtp_unique + 2 * emb)}')

log('\n== Check on a second real config: DeepSeek-V2-Lite-Chat (published 15.7B total, 2.4B active) ==')
lite_path = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/*/config.json'))[0]
lite = json.load(open(lite_path))
lb, la, _ = count(lite)
lt = sum(lb.values())
llib, _ = library_count(DeepseekV2Config, DeepseekV2ForCausalLM, {k: v for k, v in lite.items() if k not in ('auto_map',)})
log(f'  Method 1 total {B(lt)}, Method 2 total {B(llib)}, equal: {lt == llib}')
log(f'  Method 1 active {B(sum(la.values()))}, without the input embedding {B(sum(la.values()) - la["embedding"])}')

save('part2_params', dict(total=total, buckets=b, active=act, active_total=atot, per=per, library_total=lib,
                          mtp_unique=mtp_unique, embedding=emb, checkpoint=total + mtp_unique + 2 * emb,
                          v2_lite_total=lt, v2_lite_active=sum(la.values()), v2_lite_library=llib))
