"""Attention from the ground up, Part 1: self-attention and multi-head attention.

Every number in the article comes from this script:
  1. a tiny self-attention, step by step
  2. why the scores are divided by sqrt(d)
  3. our multi-head attention vs PyTorch's two built-in implementations
  4. real attention maps from Qwen2.5-0.5B
Run:  python part1_attention.py   (writes results/part1.json)
"""
import json, math, os
import torch
import torch.nn.functional as F

torch.manual_seed(0)
OUT = {}
os.makedirs('results', exist_ok=True)


# ---------------------------------------------------------------- 1. one head, by hand
def self_attention(x, Wq, Wk, Wv, causal=True):
    """x: (T, d_model). Returns the attention weights A (T, T) and the output Z (T, d_v)."""
    Q, K, V = x @ Wq, x @ Wk, x @ Wv
    scores = Q @ K.T / math.sqrt(K.shape[-1])           # how much each token matches each other token
    if causal:                                          # a token may not look at the future
        T = x.shape[0]
        future = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(future, float('-inf'))
    A = torch.softmax(scores, dim=-1)                   # each row sums to 1
    return A, A @ V


tokens = ['The', 'cat', 'sat', 'down']
T, d_model, d_k = len(tokens), 8, 4
x = torch.randn(T, d_model)
Wq, Wk, Wv = (torch.randn(d_model, d_k) / math.sqrt(d_model) for _ in range(3))
A, Z = self_attention(x, Wq, Wk, Wv)
OUT['toy'] = {'tokens': tokens, 'A': A.tolist(), 'row_sums': A.sum(-1).tolist(), 'Z_shape': list(Z.shape)}
print('1. attention weights (rows = query token, columns = key token):')
for t, row in zip(tokens, A):
    print(f'   {t:>5}', ' '.join(f'{v:5.2f}' for v in row))
print('   row sums:', [round(s, 6) for s in A.sum(-1).tolist()])


# ---------------------------------------------------------------- 2. why divide by sqrt(d)?
scale = {}
for d in (16, 64, 256, 1024):
    q, k = torch.randn(10_000, d), torch.randn(10_000, 64, d)
    dots = torch.einsum('nd,nkd->nk', q, k)              # 64 keys per query
    raw = torch.softmax(dots, -1).max(-1).values.mean().item()
    scaled = torch.softmax(dots / math.sqrt(d), -1).max(-1).values.mean().item()
    scale[d] = {'dot_std': dots.std().item(), 'max_weight_raw': raw, 'max_weight_scaled': scaled}
    print(f'2. d={d:5d}: std of q.k = {dots.std():6.1f} (sqrt(d) = {math.sqrt(d):5.1f})   '
          f'top weight without scaling {raw:.3f}, with scaling {scaled:.3f}')
OUT['scaling'] = scale


# ---------------------------------------------------------------- 3. multi-head attention, checked
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d = n_heads, d_model // n_heads
        self.Wq = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wk = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wv = torch.nn.Linear(d_model, d_model, bias=False)
        self.Wo = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):                                # x: (B, T, d_model)
        B, T, _ = x.shape
        split = lambda t: t.view(B, T, self.h, self.d).transpose(1, 2)    # (B, h, T, d)
        q, k, v = split(self.Wq(x)), split(self.Wk(x)), split(self.Wv(x))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d)              # (B, h, T, T): one matrix per head
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), 1)
        A = torch.softmax(scores.masked_fill(mask, float('-inf')), -1)
        heads = A @ v                                                     # (B, h, T, d)
        return self.Wo(heads.transpose(1, 2).reshape(B, T, self.h * self.d))


B, T, D, H = 2, 10, 64, 8
mha = MultiHeadAttention(D, H).double()
x = torch.randn(B, T, D, dtype=torch.float64)
ours = mha(x)

# (a) PyTorch's fused kernel, same projections
split = lambda t: t.view(B, T, H, D // H).transpose(1, 2)
sdpa = F.scaled_dot_product_attention(split(mha.Wq(x)), split(mha.Wk(x)), split(mha.Wv(x)), is_causal=True)
ref_a = mha.Wo(sdpa.transpose(1, 2).reshape(B, T, D))

# (b) torch.nn.MultiheadAttention with our weights copied in
ref = torch.nn.MultiheadAttention(D, H, bias=False, batch_first=True).double()
with torch.no_grad():
    ref.in_proj_weight.copy_(torch.cat([mha.Wq.weight, mha.Wk.weight, mha.Wv.weight]))
    ref.out_proj.weight.copy_(mha.Wo.weight)
causal = torch.triu(torch.ones(T, T, dtype=torch.bool), 1)
ref_b, _ = ref(x, x, x, attn_mask=causal, need_weights=False)

OUT['equivalence'] = {'vs_sdpa': (ours - ref_a).abs().max().item(), 'vs_nn_mha': (ours - ref_b).abs().max().item()}
print(f"3. our MHA vs F.scaled_dot_product_attention: max |diff| = {OUT['equivalence']['vs_sdpa']:.1e}")
print(f"   our MHA vs torch.nn.MultiheadAttention:     max |diff| = {OUT['equivalence']['vs_nn_mha']:.1e}")


# ---------------------------------------------------------------- 4. a real model's attention
from transformers import AutoModelForCausalLM, AutoTokenizer
name = 'Qwen/Qwen2.5-0.5B'
tok = AutoTokenizer.from_pretrained(name)
model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, attn_implementation='eager').eval()
cfg = model.config
text = 'The cat sat on the mat because it was tired.'
ids = tok(text, return_tensors='pt').input_ids
words = [tok.decode([i]) for i in ids[0]]
with torch.no_grad():
    att = torch.stack(model(ids, output_attentions=True).attentions)[:, 0]   # (layers, heads, T, T)
L, Hq, T, _ = att.shape

first = att[..., 1:, 0].mean(-1)                         # mean weight each head puts on token 0 (rows 1..T-1)
prev = torch.stack([att[..., i, i - 1] for i in range(1, T)], -1).mean(-1)   # weight on the previous token
diag = torch.stack([att[..., i, i] for i in range(1, T)], -1).mean(-1)
it, cat = words.index(' it'), words.index(' cat')
coref = att[..., it, cat]                                # how much "it" looks at "cat"

def pick(score):
    l, h = divmod(int(score.flatten().argmax()), Hq)
    return {'layer': l, 'head': h, 'score': float(score[l, h]), 'matrix': att[l, h].tolist()}

OUT['real'] = {
    'model': name, 'layers': L, 'heads': Hq, 'kv_heads': cfg.num_key_value_heads, 'tokens': words,
    'sink_share': float((first > 0.5).float().mean()), 'mean_first_token_weight': float(first.mean()),
    'sink': pick(first), 'previous': pick(prev), 'coref': pick(coref),
    'coref_rank_of_cat': None,
}
c = OUT['real']['coref']
row = att[c['layer'], c['head'], it]
OUT['real']['coref_top3'] = [(words[j], float(row[j])) for j in row.argsort(descending=True)[:3]]
print(f"4. {name}: {L} layers x {Hq} query heads ({cfg.num_key_value_heads} key/value heads) on {T} tokens")
print(f"   heads putting >50% of their attention on the first token: {OUT['real']['sink_share']:.0%}")
for kind in ('sink', 'previous', 'coref'):
    r = OUT['real'][kind]
    print(f"   strongest {kind:8s} head: layer {r['layer']:2d}, head {r['head']:2d}, score {r['score']:.2f}")
print(f"   that head, from ' it': top keys {[(w, round(p, 2)) for w, p in OUT['real']['coref_top3']]}")

json.dump(OUT, open('results/part1.json', 'w'), indent=1)
print('saved results/part1.json')
