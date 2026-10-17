"""Part 7: check every block of our from-scratch Transformer (scratch/model.py) against PyTorch's own layers.

1. Token embedding: the sqrt(d_model) scale.
2. Positional encoding: the paper's interleaved layout and Marian's sines-then-cosines layout hold the same numbers.
3. Scaled dot-product attention (Eq. 1): a tiny case by hand, then against torch.nn.functional.scaled_dot_product_attention.
4. Multi-head attention: against torch.nn.MultiheadAttention with the same weights.
5. The masks.
6. Encoder layer: against nn.TransformerEncoderLayer (post-norm, ReLU) with the same weights.
7. Decoder layer: against nn.TransformerDecoderLayer with the same weights.
8. Parameter count of the paper's base and big models, from our code.
Writes results/part7_stdout.txt and results/part7.json. CPU, a few seconds."""
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'scratch'))
from common import Log, save
from model import (Config, Transformer, TokenEmbedding, sinusoid_table, scaled_dot_product_attention,
                   MultiHeadAttention, EncoderLayer, DecoderLayer, causal_mask, padding_mask, count_parameters)

log = Log('part7')
R = {}
torch.manual_seed(0)


def maxdiff(a, b):
    return (a - b).abs().max().item()


# ---------------------------------------------------------------------------------------------------- 1. embedding
log('== 1. Token embedding (§3.4)')
emb = TokenEmbedding(vocab_size=1000, d_model=512)
ids = torch.tensor([[5, 17, 42]])                        # (b=1, n=3)
x = emb(ids)                                             # (1, 3, 512)
log('ids', tuple(ids.shape), '-> vectors', tuple(x.shape))
log(f'std of a table row: {emb.table.weight.std().item():.4f}   after x sqrt(512) = x {emb.scale:.2f}: {x.std().item():.4f}')
R['emb_std'] = [emb.table.weight.std().item(), x.std().item()]

# ---------------------------------------------------------------------------------------------------- 2. positions
log('')
log('== 2. Positional encoding (§3.5)')
pe_paper = sinusoid_table(50, 8, 'interleaved')          # small d_model = 8 so we can print a row
pe_marian = sinusoid_table(50, 8, 'sin_then_cos')
torch.set_printoptions(precision=4, sci_mode=False)
log('position 1, interleaved  (sin, cos, sin, cos, ...):', pe_paper[1])
log('position 1, sin_then_cos (sin, sin, ..., cos, cos):', pe_marian[1])
perm = [0, 2, 4, 6, 1, 3, 5, 7]                          # the column order that turns one layout into the other
log('same numbers, other column order:', torch.equal(pe_paper[:, perm], pe_marian))

# ---------------------------------------------------------------------------------------------------- 3. Eq. 1
log('')
log('== 3. Scaled dot-product attention (Eq. 1)')
Q = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])  # 3 queries, d_k = 2
K = torch.tensor([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])  # 3 keys (the example of Part 3)
V = torch.tensor([[10.0, 0.0], [0.0, 10.0], [5.0, 5.0]])
out, w = scaled_dot_product_attention(Q, K, V)
log('weights (each row sums to 1):')
log(w)
log('output:')
log(out)
R['tiny_weights'] = w.tolist()
# Method 2: PyTorch's fused function, on random inputs of the real size (b=2, h=8, n=10, d_k=64)
Q, K, V = (torch.randn(2, 8, 10, 64) for _ in range(3))
ours, _ = scaled_dot_product_attention(Q, K, V)
lib = F.scaled_dot_product_attention(Q, K, V)
log('ours vs torch.nn.functional.scaled_dot_product_attention, max abs difference:', f'{maxdiff(ours, lib):.2e}')
cm = causal_mask(10)                                     # (1, 1, 10, 10)
ours_c, _ = scaled_dot_product_attention(Q, K, V, cm)
lib_c = F.scaled_dot_product_attention(Q, K, V, is_causal=True)
log('with the causal mask, max abs difference:', f'{maxdiff(ours_c, lib_c):.2e}')
R['sdpa_diff'] = [maxdiff(ours, lib), maxdiff(ours_c, lib_c)]

# ---------------------------------------------------------------------------------------------------- 4. MHA
log('')
log('== 4. Multi-head attention (§3.2.2) vs torch.nn.MultiheadAttention')
d_model, h = 512, 8
ours = MultiHeadAttention(d_model, h).eval()
lib = nn.MultiheadAttention(d_model, h, batch_first=True).eval()
with torch.no_grad():
    # PyTorch keeps W_q, W_k, W_v stacked in one (3 * d_model, d_model) matrix called in_proj_weight.
    lib.in_proj_weight.copy_(torch.cat([ours.W_q.weight, ours.W_k.weight, ours.W_v.weight], dim=0))
    lib.in_proj_bias.copy_(torch.cat([ours.W_q.bias, ours.W_k.bias, ours.W_v.bias], dim=0))
    lib.out_proj.weight.copy_(ours.W_o.weight)
    lib.out_proj.bias.copy_(ours.W_o.bias)
x = torch.randn(2, 10, d_model)                          # (b=2, n=10, d_model)
ids = torch.tensor([[3] * 10, [3] * 7 + [0] * 3])        # the second sentence has 3 pad tokens at the end
with torch.no_grad():
    a = ours(x, x, x, padding_mask(ids, 0))              # our mask: True = may look
    b, wb = lib(x, x, x, key_padding_mask=(ids == 0),    # PyTorch's mask: True = pad, do NOT look
                average_attn_weights=False)
log('output', tuple(a.shape), 'weights', tuple(ours.last_weights.shape))
log('self-attention with padding, max abs difference of outputs:', f'{maxdiff(a, b):.2e}')
log('max abs difference of the 8 heads\' weights:', f'{maxdiff(ours.last_weights, wb):.2e}')
y = torch.randn(2, 6, d_model)                           # decoder side for cross-attention
with torch.no_grad():
    a = ours(y, x, x, padding_mask(ids, 0))
    b, _ = lib(y, x, x, key_padding_mask=(ids == 0))
log('cross-attention (queries from another sequence), max abs difference:', f'{maxdiff(a, b):.2e}')
R['mha_diff'] = maxdiff(a, b)

# ---------------------------------------------------------------------------------------------------- 5. masks
log('')
log('== 5. Masks (True = may look)')
log('causal_mask(4):')
log(causal_mask(4)[0, 0].int())
log('padding_mask for ids [7, 8, 9, 0, 0] with pad id 0:', padding_mask(torch.tensor([[7, 8, 9, 0, 0]]), 0)[0, 0, 0].int())

# ---------------------------------------------------------------------------------------------------- 6. encoder layer
log('')
log('== 6. Encoder layer vs nn.TransformerEncoderLayer (post-norm, ReLU)')
cfg = Config(d_model=512, n_heads=8, d_ff=2048, dropout=0.0)
ours = EncoderLayer(cfg).eval()
lib = nn.TransformerEncoderLayer(512, 8, dim_feedforward=2048, dropout=0.0, activation='relu',
                                 batch_first=True, norm_first=False)


def copy_attn(lib_attn, our_attn):
    lib_attn.in_proj_weight.copy_(torch.cat([our_attn.W_q.weight, our_attn.W_k.weight, our_attn.W_v.weight]))
    lib_attn.in_proj_bias.copy_(torch.cat([our_attn.W_q.bias, our_attn.W_k.bias, our_attn.W_v.bias]))
    lib_attn.out_proj.weight.copy_(our_attn.W_o.weight)
    lib_attn.out_proj.bias.copy_(our_attn.W_o.bias)


def copy_linear_norm(dst, src):
    dst.weight.copy_(src.weight)
    dst.bias.copy_(src.bias)


with torch.no_grad():
    for layer in (ours,):                                # make the norms non-trivial so the check means something
        for m in layer.modules():
            if isinstance(m, nn.LayerNorm):
                m.weight.uniform_(0.5, 1.5); m.bias.uniform_(-0.2, 0.2)
    copy_attn(lib.self_attn, ours.self_attn)
    copy_linear_norm(lib.linear1, ours.ffn.W_1)          # FFN first matrix W_1, b_1
    copy_linear_norm(lib.linear2, ours.ffn.W_2)          # FFN second matrix W_2, b_2
    copy_linear_norm(lib.norm1, ours.add_norm_1.norm)    # norm after self-attention
    copy_linear_norm(lib.norm2, ours.add_norm_2.norm)    # norm after the FFN
lib.train()                                              # train mode with dropout 0: PyTorch's slow, exact path
with torch.no_grad():
    a = ours(x, padding_mask(ids, 0))
    b = lib(x, src_key_padding_mask=(ids == 0))
log('output', tuple(a.shape), ' max abs difference:', f'{maxdiff(a, b):.2e}')
R['enc_diff'] = maxdiff(a, b)

# ---------------------------------------------------------------------------------------------------- 7. decoder layer
log('')
log('== 7. Decoder layer vs nn.TransformerDecoderLayer (post-norm, ReLU)')
ours = DecoderLayer(cfg).eval()
lib = nn.TransformerDecoderLayer(512, 8, dim_feedforward=2048, dropout=0.0, activation='relu',
                                 batch_first=True, norm_first=False)
with torch.no_grad():
    copy_attn(lib.self_attn, ours.self_attn)
    copy_attn(lib.multihead_attn, ours.cross_attn)       # PyTorch calls cross-attention "multihead_attn"
    copy_linear_norm(lib.linear1, ours.ffn.W_1)
    copy_linear_norm(lib.linear2, ours.ffn.W_2)
    copy_linear_norm(lib.norm1, ours.add_norm_1.norm)
    copy_linear_norm(lib.norm2, ours.add_norm_2.norm)
    copy_linear_norm(lib.norm3, ours.add_norm_3.norm)
lib.train()
memory = x
with torch.no_grad():
    a = ours(y, memory, causal_mask(6), padding_mask(ids, 0))
    b = lib(y, memory, tgt_mask=nn.Transformer.generate_square_subsequent_mask(6), tgt_is_causal=True,
            memory_key_padding_mask=(ids == 0))
log('output', tuple(a.shape), ' max abs difference:', f'{maxdiff(a, b):.2e}')
R['dec_diff'] = maxdiff(a, b)

# ---------------------------------------------------------------------------------------------------- 8. parameters
log('')
log('== 8. Parameter count from our code (paper, Table 3: base 65M, big 213M)')
for name, c in [('base', Config(vocab_size=37000)),
                ('big', Config(vocab_size=37000, d_model=1024, n_heads=16, d_ff=4096))]:
    m = Transformer(c)
    total = count_parameters(m)
    emb_n = m.embedding.table.weight.numel()
    enc = sum(p.numel() for p in m.encoder.layers.parameters())
    dec = sum(p.numel() for p in m.decoder.layers.parameters())
    log(f'{name}: d_model {c.d_model}, h {c.n_heads}, d_ff {c.d_ff}, vocabulary {c.vocab_size:,}')
    log(f'  shared embedding {emb_n:,} | one encoder layer {enc // 6:,} | one decoder layer {dec // 6:,}')
    log(f'  total {total:,} = {total / 1e6:.1f}M')
    if name == 'base':
        attn, ffn, norms = 4 * (512 * 512 + 512), 512 * 2048 + 2048 + 2048 * 512 + 512, 2 * 1024
        log(f'  by hand, one encoder layer: attention {attn:,} + FFN {ffn:,} + 2 norms {norms:,} = {attn + ffn + norms:,}')
        log(f'  share of the shared embedding table: {emb_n / total * 100:.1f}%')
    R[f'params_{name}'] = {'total': total, 'embedding': emb_n, 'enc_layer': enc // 6, 'dec_layer': dec // 6}
    del m

save('part7', R)
