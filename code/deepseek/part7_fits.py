"""Part 7: what fits where. Memory arithmetic for every major DeepSeek model, from real configs and real file sizes.

1. Bytes per weight: the rule, then the real files (official checkpoints, GGUF quants) and their effective bits.
2. The GB versus GiB trap ("131 GB" is really 140.2 GB).
3. Where the weights of DeepSeek-V3 live: routed experts versus everything else (from config.json).
4. KV cache per token for every model, computed from each config.json, at 32K, 128K and 1M tokens.
5. The "what fits where" table: weights + one 32K-token KV cache against common machines.

Configs are fetched once from Hugging Face (small JSON files) and cached in results/part7_configs/.
File sizes are the sums of Hugging Face file listings recorded in research/04-hardware-and-serving.md
(checked 2026-11-10); the source repo is printed next to each one."""
import json, os, glob, urllib.request
from common import Log, save, RESULTS

log = Log('part7_fits')
CFG_DIR = os.path.join(RESULTS, 'part7_configs')
os.makedirs(CFG_DIR, exist_ok=True)


def config(repo):
    """config.json of a Hugging Face repo (text part for multimodal V4.1)."""
    path = os.path.join(CFG_DIR, repo.replace('/', '__') + '.json')
    if not os.path.exists(path):
        local = glob.glob(os.path.expanduser(f'~/.cache/huggingface/hub/models--{repo.replace("/", "--")}/snapshots/*/config.json'))
        if local:
            data = open(local[0]).read()
        else:
            req = urllib.request.Request(f'https://huggingface.co/{repo}/raw/main/config.json', headers={'User-Agent': 'Mozilla/5.0'})
            data = urllib.request.urlopen(req).read().decode()
        open(path, 'w').write(data)
    c = json.load(open(path))
    return c.get('text_config', c)


GB, GiB = 1e9, 2 ** 30

# ---------------------------------------------------------------- 1. bytes per weight
log('== 1. Bytes per weight: the rule ==')
log('memory = parameters x bits per weight / 8')
for fmt, bits in [('BF16', 16), ('FP8', 8), ('INT4 / FP4', 4)]:
    log(f'  671B in {fmt:10s}: 671e9 x {bits:2d} / 8 = {671e9 * bits / 8 / GB:7,.1f} GB')

# (model, format, real size in GB, parameter count used for bits/weight, source repo)
FILES = [
    ('V3 / R1 (671B)', 'BF16 GGUF', 1342.3, 671e9, 'unsloth/DeepSeek-R1-GGUF'),
    ('V3 / R1 (671B)', 'GGUF Q8_0', 713.3, 671e9, 'unsloth/DeepSeek-R1-GGUF'),
    ('V3 / R1 (671B)', 'official FP8', 688.6, 684.49e9, 'deepseek-ai/DeepSeek-V3'),
    ('V3 / R1 (671B)', 'GGUF Q4_K_M', 404.4, 671e9, 'unsloth/DeepSeek-R1-GGUF'),
    ('V3 / R1 (671B)', 'UD-Q2_K_XL', 226.6, 671e9, 'unsloth/DeepSeek-R1-GGUF'),
    ('V3 / R1 (671B)', 'UD-IQ1_S', 140.2, 671e9, 'unsloth/DeepSeek-R1-GGUF'),
    ('V3.2 (671B)', 'official FP8', 689.5, 684.49e9, 'deepseek-ai/DeepSeek-V3.2'),
    ('V3.2 (671B)', 'UD-IQ1_S', 184.1, 671e9, 'unsloth/DeepSeek-V3.2-GGUF'),
    ('V2 (236B)', 'official BF16', 471.5, 235.7e9, 'deepseek-ai/DeepSeek-V2'),
    ('V2-Lite (16B)', 'official BF16', 31.4, 15.7e9, 'deepseek-ai/DeepSeek-V2-Lite'),
    ('V4-Flash (284B)', 'official FP4+FP8', 159.6, 284e9, 'deepseek-ai/DeepSeek-V4-Flash'),
    ('V4-Flash (284B)', 'UD-Q2_K_XL', 96.8, 284e9, 'unsloth/DeepSeek-V4-Flash-GGUF'),
    ('V4-Pro (1.6T)', 'official FP4+FP8', 864.7, 1.6e12, 'deepseek-ai/DeepSeek-V4-Pro'),
    ('V4.1-Flash (552B+196B)', 'official FP4+FP8', 510.3, 748e9, 'deepseek-ai/DeepSeek-V4.1-Flash-1017'),
    ('R1-Distill-Llama-70B', 'official BF16', 141.1, 70.55e9, 'deepseek-ai/DeepSeek-R1-Distill-Llama-70B'),
    ('R1-Distill-Llama-70B', 'GGUF Q4_K_M', 42.5, 70.55e9, 'unsloth/DeepSeek-R1-Distill-Llama-70B-GGUF'),
]
log('')
log('== Real files: size and effective bits per weight ==')
log(f'{"model":24s} {"format":17s} {"GB":>8s} {"GiB":>8s} {"bits/weight":>11s}')
files_out = []
for m, f, gb, n, src in FILES:
    bits = gb * GB * 8 / n
    files_out.append(dict(model=m, fmt=f, gb=gb, gib=round(gb * GB / GiB, 1), bits=round(bits, 2), source=src))
    log(f'{m:24s} {f:17s} {gb:8,.1f} {gb * GB / GiB:8,.1f} {bits:11.2f}')

# ---------------------------------------------------------------- 2. GB versus GiB
log('')
log('== 2. The GB versus GiB trap ==')
for name, gb in [('R1 UD-IQ1_S ("131 GB" in the blog)', 140.2), ('R1 UD-Q2_K_XL ("212 GB" in the blog)', 226.6)]:
    log(f'{name}: {gb} GB = {gb * GB:,.0f} bytes = {gb * GB / GiB:.1f} GiB')
log(f'average bits per weight of the "1.58-bit" file: 140.2e9 x 8 / 671e9 = {140.2e9 * 8 / 671e9:.2f}')

# ---------------------------------------------------------------- 3. where V3's weights live
c = config('deepseek-ai/DeepSeek-V3')
d, L, nd = c['hidden_size'], c['num_hidden_layers'], c['first_k_dense_replace']
n_moe = L - nd
expert = 3 * d * c['moe_intermediate_size']                       # SwiGLU: gate, up, down
mla = (d * c['q_lora_rank'] + c['q_lora_rank']
       + c['q_lora_rank'] * c['num_attention_heads'] * (c['qk_nope_head_dim'] + c['qk_rope_head_dim'])
       + d * (c['kv_lora_rank'] + c['qk_rope_head_dim']) + c['kv_lora_rank']
       + c['kv_lora_rank'] * c['num_attention_heads'] * (c['qk_nope_head_dim'] + c['v_head_dim'])
       + c['num_attention_heads'] * c['v_head_dim'] * d)
routed = n_moe * c['n_routed_experts'] * expert
rest = (2 * c['vocab_size'] * d + L * mla + nd * 3 * d * c['intermediate_size']
        + n_moe * (c['n_shared_experts'] * expert + c['n_routed_experts'] * d + c['n_routed_experts'])
        + L * 2 * d + d)
total = routed + rest
log('')
log('== 3. DeepSeek-V3: where the 671B weights live (from config.json) ==')
log(f'one expert = 3 x {d} x {c["moe_intermediate_size"]} = {expert:,} parameters')
log(f'routed experts: {n_moe} layers x {c["n_routed_experts"]} x {expert:,} = {routed / 1e9:.2f}B ({routed / total:.1%})')
log(f'everything else (attention, dense layers, shared experts, routers, embedding, head) = {rest / 1e9:.2f}B ({rest / total:.1%})')
log(f'total = {total / 1e9:.2f}B')
q4 = 404.4e9 / 671e9                                              # bytes per weight of the Q4_K_M file
per_tok_experts = n_moe * c['num_experts_per_tok'] * expert
log(f'per token: {c["num_experts_per_tok"]} of {c["n_routed_experts"]} routed experts in each of {n_moe} layers = {per_tok_experts / 1e9:.2f}B parameters')
log(f'at Q4_K_M ({q4 * 8:.2f} bits): one expert = {expert * q4 / 1e6:.1f} MB; routed experts read per token = {per_tok_experts * q4 / GB:.1f} GB '
    f'out of {routed * q4 / GB:.0f} GB; the rest = {rest * q4 / GB:.1f} GB')

# ---------------------------------------------------------------- 4. KV cache per token
log('')
log('== 4. KV cache per token, from each config.json ==')


def mla_bf16(c):
    return (c['kv_lora_rank'] + c['qk_rope_head_dim']) * c['num_hidden_layers'] * 2


def gqa_bf16(c):
    hd = c.get('head_dim') or c['hidden_size'] // c['num_attention_heads']
    return 2 * c['num_hidden_layers'] * c['num_key_value_heads'] * hd * 2


def v32(c):                       # FlashMLA FP8 row: 512 B latent + 16 B scales + 128 B BF16 RoPE = 656 B,
    return c['num_hidden_layers'] * (656 + c['index_head_dim'] + 4)   # + FP8 indexer key (128 B) + 4 B scale


def v4(c):                        # FlashMLA V4 row 584 B per compressed entry; FP4 indexer key 64 B + 4 B scale on CSA
    r = c['compress_ratios'][:c['num_hidden_layers']]             # the last entries belong to the MTP / DSpark layers
    csa = sum(1 for x in r if x == 4)
    hca = sum(1 for x in r if x == 128)
    return csa * (584 + 68) / 4 + hca * 584 / 128, csa, hca


MODELS = {}                       # name -> (bytes per token, context limit, how)
for repo, name, ctx in [('deepseek-ai/DeepSeek-V2-Lite-Chat', 'V2-Lite', 32768), ('deepseek-ai/DeepSeek-V2', 'V2', 131072),
                        ('deepseek-ai/DeepSeek-V3', 'V3 / R1 / V3.1', 131072)]:
    cc = config(repo)
    b = mla_bf16(cc)
    MODELS[name] = (b, ctx, f'MLA BF16: ({cc["kv_lora_rank"]} + {cc["qk_rope_head_dim"]}) x {cc["num_hidden_layers"]} layers x 2 B')
naive = c['num_attention_heads'] * (c['qk_nope_head_dim'] + c['qk_rope_head_dim'] + c['v_head_dim']) * L * 2
cc = config('deepseek-ai/DeepSeek-V3.2')
MODELS['V3.2'] = (v32(cc), 131072, f'FP8 MLA row 656 B + indexer 132 B, x {cc["num_hidden_layers"]} layers')
for repo, name in [('deepseek-ai/DeepSeek-V4-Flash', 'V4-Flash'), ('deepseek-ai/DeepSeek-V4-Pro', 'V4-Pro')]:
    cc = config(repo)
    b, csa, hca = v4(cc)
    MODELS[name] = (b, 1048576, f'{csa} CSA layers x 652 B / 4 + {hca} HCA layers x 584 B / 128')
MODELS['V4.1-Flash'] = (890, 1048576, 'published: 890 B per token (V4.1 report, abstract)')
for repo, name in [('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B', 'Distill-Qwen-1.5B'), ('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', 'Distill-Qwen-7B'),
                   ('deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'Distill-Llama-8B'), ('deepseek-ai/DeepSeek-R1-Distill-Qwen-14B', 'Distill-Qwen-14B'),
                   ('deepseek-ai/DeepSeek-R1-Distill-Qwen-32B', 'Distill-Qwen-32B'), ('deepseek-ai/DeepSeek-R1-Distill-Llama-70B', 'Distill-Llama-70B')]:
    cc = config(repo)
    hd = cc.get('head_dim') or cc['hidden_size'] // cc['num_attention_heads']
    MODELS[name] = (gqa_bf16(cc), 131072, f'GQA BF16: 2 x {cc["num_hidden_layers"]} layers x {cc["num_key_value_heads"]} KV heads x {hd} x 2 B')

log(f'naive multi-head cache for V3 (for contrast): 128 heads x (128 + 64 + 128) x 61 x 2 B = {naive:,} B per token')
log(f'{"model":18s} {"B/token":>9s} {"32K":>9s} {"128K":>9s} {"1M":>9s}   how')
kv_out = {}


def gb_or_na(b, n, ctx):
    return f'{b * n / GB:8.2f}G' if n <= ctx else '      n/a'


for name, (b, ctx, how) in MODELS.items():
    kv_out[name] = dict(bytes_per_token=round(b, 1), context=ctx, how=how,
                        gb={str(n): round(b * n / GB, 3) for n in (32768, 131072, 1048576)})
    log(f'{name:18s} {b:9,.0f} {gb_or_na(b, 32768, ctx)} {gb_or_na(b, 131072, ctx)} {gb_or_na(b, 1048576, ctx)}   {how}')
log('checks against published numbers:')
log(f'  V3 {MODELS["V3 / R1 / V3.1"][0]:,} B = ISCA 2025 Table 1 "70.272 KB"')
log(f'  V3.2 {MODELS["V3.2"][0]:,.0f} B, V4-Flash {MODELS["V4-Flash"][0]:,.0f} B: V4.1 report Figure 1(b) gives 48,068 and 3,514')
log(f'  V4-Pro / V3.2 = {MODELS["V4-Pro"][0] / MODELS["V3.2"][0]:.1%} (V4 report: "10% of KV cache"); '
    f'V4-Flash / V3.2 = {MODELS["V4-Flash"][0] / MODELS["V3.2"][0]:.1%} (report: "7%")')
log(f'  V3 MLA versus naive multi-head: {naive / MODELS["V3 / R1 / V3.1"][0]:.0f}x smaller; '
    f'Distill-Llama-8B needs {MODELS["Distill-Llama-8B"][0] / MODELS["V3 / R1 / V3.1"][0]:.2f}x more per token than the 671B V3')

# ---------------------------------------------------------------- 5. what fits where
# (label, weights GB, KV model, note). Weights are real file sizes; for V4.1 with Engram on SSD, the "Rest" part.
ROWS = [
    ('V3/R1 FP8 official', 688.6, 'V3 / R1 / V3.1'),
    ('V3/R1 Q4_K_M', 404.4, 'V3 / R1 / V3.1'),
    ('R1 UD-Q2_K_XL', 226.6, 'V3 / R1 / V3.1'),
    ('R1 UD-IQ1_S', 140.2, 'V3 / R1 / V3.1'),
    ('V3.2 FP8 official', 689.5, 'V3.2'),
    ('V2 BF16 official', 471.5, 'V2'),
    ('V2 4-bit (Ollama)', 133.0, 'V2'),
    ('V2-Lite 4-bit (Ollama)', 8.9, 'V2-Lite'),
    ('V4-Pro-0813 official', 892.7, 'V4-Pro'),
    ('V4-Flash official', 159.6, 'V4-Flash'),
    ('V4-Flash UD-Q2_K_XL', 96.8, 'V4-Flash'),
    ('V4.1-Flash-1017 official', 510.3, 'V4.1-Flash'),
    ('V4.1 UD-IQ1_S, Engram on SSD', 122.4, 'V4.1-Flash'),
    ('Distill-Llama-70B Q4_K_M', 42.5, 'Distill-Llama-70B'),
    ('Distill-Qwen-32B Q4_K_M', 19.9, 'Distill-Qwen-32B'),
    ('Distill-Qwen-14B Q4_K_M', 9.0, 'Distill-Qwen-14B'),
    ('Distill-Qwen-7B Q4_K_M', 4.7, 'Distill-Qwen-7B'),
]
MACHINES = [('16GB lap', 16), ('24GB GPU', 24), ('32GB GPU', 32), ('128GB box', 128), ('512GB Mac', 512),
            ('8x H100', 640), ('8x H200', 1128), ('2x8 H100', 1280), ('8x MI300X', 1536)]
USABLE = 0.90                     # our rule: keep 10% for activations, engine buffers and the OS (an estimate)
CTX = 32768
log('')
log(f'== 5. What fits where: weights + KV for one {CTX:,}-token sequence <= {USABLE:.0%} of memory ==')
log('(capacities taken as decimal GB; RAM sold as "512 GB" is really 512 GiB, so this errs on the safe side)')
head = f'{"setup":30s} {"need GB":>8s} ' + ' '.join(f'{m[:9]:>9s}' for m, _ in MACHINES)
log(head)
fits = []
for label, w, kvm in ROWS:
    need = w + MODELS[kvm][0] * CTX / GB
    row = [cap * USABLE >= need for _, cap in MACHINES]
    fits.append(dict(setup=label, weights=w, need=round(need, 1), fits=row))
    log(f'{label:30s} {need:8,.1f} ' + ' '.join(f'{"yes" if ok else "-":>9s}' for ok in row))

save('part7_fits', dict(files=files_out, kv=kv_out, naive_v3=naive, machines=MACHINES, fits=fits,
                        v3=dict(expert=expert, routed=routed, rest=rest, total=total, per_token_experts=per_tok_experts)))
