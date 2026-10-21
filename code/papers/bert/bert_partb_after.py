"""Background page of BERT, explained: after ModernBERT (2024 to 2026), run on a laptop.

What it shows (every number is printed and saved to results/partb_after_stdout.txt and results/partb_after.json):
  7. BERT against ModernBERT: sizes and settings read from the released configs, the same blank filled by both,
     and the longest input each one accepts (BERT stops at 512 tokens; ModernBERT reads 8,192).
  8. ModernBERT's alternating attention: how many layers look at the whole text, and how much work local
     attention saves at 8,192 tokens.
  9. The masked diffusion loss of LLaDA (2025), L = -E[(1/t) sum over masked tokens of log p(x_i | x_t)]:
     computed by hand (Method 1) and with PyTorch's cross-entropy (Method 2) on BERT's own predictions.
 10. BERT writes text: start from a row of [MASK] tokens and fill the most confident blank, step by step
     (the reverse process of a masked diffusion model, with plain bert-base-uncased, no extra training).
Runs on the CPU. Run: python bert_partb_after.py   (about one minute after the downloads)"""
import math
import time
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from common import Log, save

log = Log('partb_after')
torch.manual_seed(0)
OUT = {}
NAMES = {'BERT-base': 'bert-base-uncased', 'ModernBERT-base': 'answerdotai/ModernBERT-base'}
tok, mlm = {}, {}
for k, n in NAMES.items():
    tok[k] = AutoTokenizer.from_pretrained(n)
    mlm[k] = AutoModelForMaskedLM.from_pretrained(n).eval()

# ---------------------------------------------------------------- 7. BERT against ModernBERT
log('== 7. BERT (2018) against ModernBERT (2024), read from the released models')
OUT['compare'] = {}
log(f'{"":26}{"BERT-base":>14}{"ModernBERT-base":>18}')
rows = {}
for k, n in NAMES.items():
    c = AutoConfig.from_pretrained(n)
    rows[k] = dict(layers=c.num_hidden_layers, hidden=c.hidden_size, ffn=c.intermediate_size,
                   heads=c.num_attention_heads, vocab=c.vocab_size, max_len=c.max_position_embeddings,
                   params=sum(p.numel() for p in mlm[k].parameters()) / 1e6)
for f, label in [('layers', 'layers'), ('hidden', 'hidden size'), ('ffn', 'feed-forward size'),
                 ('heads', 'attention heads'), ('vocab', 'vocabulary'), ('max_len', 'longest input (tokens)'),
                 ('params', 'parameters (millions)')]:
    a, b = rows['BERT-base'][f], rows['ModernBERT-base'][f]
    fmt = (lambda v: f'{v:,.1f}') if f == 'params' else (lambda v: f'{v:,}')
    log(f'{label:26}{fmt(a):>14}{fmt(b):>18}')
OUT['compare'] = rows
log('(ModernBERT uses GeGLU: its first feed-forward layer makes 2 x 1,152 = 2,304 numbers, one half a value, the other a gate)')

sent = 'The capital of France is [MASK].'
OUT['fill'] = {}
for k in NAMES:
    s = sent.replace('[MASK]', tok[k].mask_token)
    x = tok[k](s, return_tensors='pt')
    with torch.no_grad():
        logits = mlm[k](**x).logits[0]
    pos = (x['input_ids'][0] == tok[k].mask_token_id).nonzero()[0, 0]
    p = logits[pos].softmax(-1)
    top = p.topk(5)
    words = [(tok[k].decode([int(i)]).strip(), float(v)) for v, i in zip(top.values, top.indices)]
    OUT['fill'][k] = words
    log(f'{k:16} "{sent}" -> ' + ', '.join(f'{w} ({v:.2f})' for w, v in words))

long_text = ' '.join(['The river bank was quiet in the morning, and the boats stayed at the shore.'] * 400)
OUT['long'] = {}
for k in NAMES:
    ids = tok[k](long_text, return_tensors='pt', truncation=False)['input_ids'][:, :2000]
    try:
        with torch.no_grad():
            mlm[k](input_ids=ids)
        msg = 'runs'
    except Exception as e:
        msg = 'fails: ' + type(e).__name__
    OUT['long'][k] = msg
    log(f'{k:16} on a {ids.shape[1]:,}-token input: {msg}')

# ---------------------------------------------------------------- 8. alternating attention
log('\n== 8. ModernBERT: every third layer looks at the whole text, the others at a 128-token window')
c = AutoConfig.from_pretrained(NAMES['ModernBERT-base'])
every, window, L = c.global_attn_every_n_layers, c.local_attention, c.num_hidden_layers
glob = [i for i in range(L) if i % every == 0]
log(f'layers: {L}; global layers (0-based): {glob} -> {len(glob)} global, {L - len(glob)} local; window {window} tokens')
n = 8192
full = L * n * n                                           # query-key pairs if every layer were global
alt = len(glob) * n * n + (L - len(glob)) * n * window
log(f'query-key pairs at {n:,} tokens: all global {full:.3e}, alternating {alt:.3e} -> {alt / full:.1%} of the work')
OUT['alternating'] = dict(layers=L, global_layers=len(glob), window=window, share=alt / full)

# ---------------------------------------------------------------- 9. the masked diffusion loss
log('\n== 9. LLaDA (2025): L = -E_t [ (1/t) * sum over masked tokens of log p(x_i | x_t) ], t ~ U(0, 1)')
k = 'BERT-base'
x0 = tok[k]('the cat sat on the mat because it was warm and the sun was out .', return_tensors='pt')['input_ids'][0]
inner = torch.arange(1, len(x0) - 1)                        # never mask [CLS] or [SEP]
g = torch.Generator().manual_seed(1)
rows9 = []
for t in [0.15, 0.5, 0.9]:
    m = inner[torch.rand(len(inner), generator=g) < t]     # forward process: each token masked with probability t
    xt = x0.clone()
    xt[m] = tok[k].mask_token_id
    with torch.no_grad():
        logits = mlm[k](input_ids=xt[None]).logits[0].double()   # float64, so the two methods can agree closely
    # Method 1: by hand. log-softmax of the true token at every masked place, summed, times 1/t
    logp = logits[m] - logits[m].logsumexp(-1, keepdim=True)
    hand = -(1 / t) * logp.gather(1, x0[m][:, None]).sum()
    # Method 2: the library. cross-entropy summed over masked places, then the same 1/t weight
    lib = (1 / t) * F.cross_entropy(logits[m], x0[m], reduction='sum')
    rows9.append(dict(t=t, masked=len(m), of=len(inner), hand=float(hand), lib=float(lib),
                      per_token=float(lib * t / len(m))))
    log(f't = {t:.2f}: {len(m):2} of {len(inner)} tokens masked; loss by hand {hand:.4f}, library {lib:.4f}; '
        f'per masked token {lib * t / len(m):.3f} nats')
OUT['diffusion_loss'] = rows9
log('BERT itself is the t = 0.15 case without the 1/t weight: one fixed masking ratio')

# ---------------------------------------------------------------- 10. BERT writes text
log('\n== 10. Writing with BERT: start from [MASK] x 8, unmask the most confident blank each step')
prompt = 'the best thing about a quiet morning is'
pi = tok[k](prompt, add_special_tokens=False)['input_ids']
N = 8
ids = torch.tensor([tok[k].cls_token_id] + pi + [tok[k].mask_token_id] * N + [tok[k].sep_token_id])
steps = []
for step in range(N):
    with torch.no_grad():
        p = mlm[k](input_ids=ids[None]).logits[0].softmax(-1)
    masked = (ids == tok[k].mask_token_id).nonzero()[:, 0]
    conf, best = p[masked].max(-1)
    j = int(conf.argmax())                                  # the blank BERT is surest about
    ids[masked[j]] = best[j]
    shown = tok[k].convert_ids_to_tokens(ids[1 + len(pi):-1].tolist())
    line = ' '.join('_' if w == '[MASK]' else w for w in shown)
    steps.append(dict(step=step + 1, filled=tok[k].decode([int(best[j])]), conf=float(conf[j]), text=line))
    log(f'step {step + 1}: fills "{tok[k].decode([int(best[j])])}" (p = {conf[j]:.2f})   {prompt} {line}')
OUT['write'] = dict(prompt=prompt, steps=steps)
log('(eight steps for eight words: one model call per word, the same count as a left-to-right model)')

# ---------------------------------------------------------------- 11. the masking ratio over time
log('\n== 11. How much of the text each model hides during training (from the papers named)')
MASK = [('2018', 'BERT', '15%', 'BERT paper, §3.1'),
        ('2019', 'RoBERTa', '15%', 'RoBERTa paper, §2.4; NeoBERT paper, Table 1'),
        ('2024', 'ModernBERT', '30%', 'ModernBERT paper, §2.2.2'),
        ('2025', 'NeoBERT', '20%', 'NeoBERT paper, Table 1'),
        ('2025', 'mmBERT', '30%, then 15%, then 5%', 'mmBERT paper, §3.3'),
        ('2025', 'LLaDA', 'any share from 0 to 100%', 'LLaDA paper, §2.1'),
        ('2026', 'UltraBERT', '5% and 50% in every batch', 'UltraBERT paper, §4')]
for y, m, r, src in MASK:
    log(f'  {y}  {m:11} {r:27} {src}')
OUT['mask_ratio'] = MASK

save('partb_after', OUT)
log('\nsaved results/partb_after.json')
