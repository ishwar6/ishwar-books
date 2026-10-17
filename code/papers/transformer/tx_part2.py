"""Part 2 (the architecture): open a released Transformer translation model and check each piece of Section 3.

Model: Helsinki-NLP/opus-mt-en-de (MarianMT), English to German, with the paper's base shapes
(6 + 6 layers, d_model 512, 8 heads, d_ff 2048, sinusoidal positions). It is not the paper's own model.

It shows:
  1. the configuration and the parameter count of each piece (embeddings, attention, feed-forward);
  2. one real translation and the tensor shape after each step;
  3. encoder layer 1 rebuilt by hand: LayerNorm(x + Sublayer(x)) twice, compared with the library output;
  4. the feed-forward network is position-wise: one token alone gives the same output as in the sentence;
  5. one weight matrix shared by the two embeddings and the output layer; the sqrt(d_model) scaling;
  6. the sinusoidal position encoding by hand, compared with the model's table; wavelengths;
     PE(pos + k) as a fixed rotation of PE(pos); the dot product depends only on the offset.
Run: ~/.venvs/attn/bin/python tx_part2.py   (CPU, under a minute after the download)."""
import math
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer
from common import Log, save

torch.manual_seed(0)
log = Log('part2')
R = {}
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME).eval()
cfg = model.config

# ------------------------------------------------------------------ 1. configuration and parameters
log('== 1. configuration of', NAME)
for k in ['d_model', 'encoder_layers', 'decoder_layers', 'encoder_attention_heads', 'encoder_ffn_dim',
          'vocab_size', 'activation_function', 'scale_embedding', 'max_position_embeddings']:
    log(f'  {k:26s} {getattr(cfg, k)}')
D = cfg.d_model
enc, dec = model.model.encoder, model.model.decoder
L0 = enc.layers[0]
log('  encoder layer 1 modules:', ', '.join(n for n, _ in L0.named_children()))


def n_params(m):
    return sum(p.numel() for p in m.parameters())


shared = model.model.shared.weight
attn_p = n_params(L0.self_attn)
ffn_p = n_params(L0.fc1) + n_params(L0.fc2)
ln_p = n_params(L0.self_attn_layer_norm) + n_params(L0.final_layer_norm)
enc_layer = n_params(L0)
dec_layer = n_params(dec.layers[0])
total = n_params(model)          # counts the shared matrix once
log('== parameters')
log(f'  shared embedding matrix   {shared.shape[0]:,} x {shared.shape[1]} = {shared.numel():,}')
log(f'  encoder layer: attention {attn_p:,}  feed-forward {ffn_p:,}  layer norms {ln_p:,}  total {enc_layer:,}')
log(f'  feed-forward share of an encoder layer: {ffn_p / enc_layer:.1%}')
log(f'  decoder layer total {dec_layer:,} (one more attention block: {dec_layer - enc_layer:,})')
log(f'  6 encoder layers {6 * enc_layer:,}; 6 decoder layers {6 * dec_layer:,}; whole model {total:,}')
R['params'] = dict(shared=shared.numel(), attn=attn_p, ffn=ffn_p, ln=ln_p, enc_layer=enc_layer, dec_layer=dec_layer,
                   total=total, vocab=shared.shape[0])

# ------------------------------------------------------------------ 2. one translation and the shapes
SRC = 'The cat sat on the mat because it was tired.'
batch = tok([SRC], return_tensors='pt')
ids = batch['input_ids']
pieces = tok.convert_ids_to_tokens(ids[0])
log('== 2. one sentence through the model')
log('  input  :', SRC)
log('  tokens :', ' '.join(pieces), f'({len(pieces)} tokens)')
log('  ids    :', ids[0].tolist())
with torch.no_grad():
    out = model.generate(**batch, num_beams=4, max_new_tokens=40)
log('  output :', tok.decode(out[0], skip_special_tokens=True))
log('  output tokens:', ' '.join(tok.convert_ids_to_tokens(out[0])))

with torch.no_grad():
    emb = enc.embed_tokens(ids)                          # learned embedding (the shared matrix)
    scale = enc.embed_scale if hasattr(enc, 'embed_scale') else math.sqrt(D)
    pos = enc.embed_positions(ids.shape)                 # sinusoidal table rows 0..n-1
    x = emb * scale + pos
    eo = enc(**batch, output_hidden_states=True)
log(f'  embedding lookup           {tuple(emb.shape)}')
log(f'  times sqrt(d_model) = {float(scale):.3f}, plus positions {tuple(pos.shape)}  -> {tuple(x.shape)}')
for i, h in enumerate(eo.hidden_states[1:], 1):
    log(f'  after encoder layer {i}      {tuple(h.shape)}')
log(f'  encoder output z           {tuple(eo.last_hidden_state.shape)}  (one vector of {D} numbers per input token)')
log('  max |our input - library input| = %.2e' % (x - eo.hidden_states[0]).abs().max())
R['sentence'] = dict(src=SRC, pieces=pieces, n=len(pieces), out=tok.decode(out[0], skip_special_tokens=True))

# ------------------------------------------------------------------ 3. encoder layer 1 by hand
log('== 3. encoder layer 1 rebuilt by hand: LayerNorm(x + Sublayer(x))')
act = L0.activation_fn
with torch.no_grad():
    h0 = eo.hidden_states[0]
    a = L0.self_attn(hidden_states=h0)[0]
    s1 = L0.self_attn_layer_norm(h0 + a)
    f = L0.fc2(act(L0.fc1(s1)))
    s2 = L0.final_layer_norm(s1 + f)
    lib = eo.hidden_states[1]
    # layer norm by hand on token 2 ("cat"?) after sub-layer 1
    t = 2
    v = (h0 + a)[0, t]
    mu, var = v.mean(), v.var(unbiased=False)
    ln_hand = (v - mu) / torch.sqrt(var + L0.self_attn_layer_norm.eps) * L0.self_attn_layer_norm.weight + L0.self_attn_layer_norm.bias
log(f'  x (layer input) norm per token: ' + ' '.join(f'{float(n):.1f}' for n in h0[0].norm(dim=-1)))
log(f'  attention output norm per token: ' + ' '.join(f'{float(n):.1f}' for n in a[0].norm(dim=-1)))
log(f'  token {t} ({pieces[t]}): x + attn has mean {float(mu):.3f} and std {float(var.sqrt()):.3f}')
log('  max |layer norm by hand - library layer norm| = %.2e' % (ln_hand - s1[0, t]).abs().max())
log('  max |our layer 1 output - library layer 1 output| = %.2e' % (s2 - lib).abs().max())
# how much the residual keeps: cosine between x and the sub-layer 1 output
cos = torch.nn.functional.cosine_similarity(h0[0], s1[0], dim=-1)
log('  cosine(x, LayerNorm(x + attn)) per token: ' + ' '.join(f'{float(c):.2f}' for c in cos))
R['layer1'] = dict(err=float((s2 - lib).abs().max()), mean=float(mu), std=float(var.sqrt()))

# ------------------------------------------------------------------ 4. the FFN is position-wise
log('== 4. the feed-forward network is position-wise')
with torch.no_grad():
    full = L0.fc2(act(L0.fc1(s1)))[0]
    alone = torch.stack([L0.fc2(act(L0.fc1(s1[0, i:i + 1])))[0] for i in range(s1.shape[1])])
    hid = act(L0.fc1(s1))[0]
log(f'  W1 {tuple(L0.fc1.weight.T.shape)}  b1 {tuple(L0.fc1.bias.shape)}  W2 {tuple(L0.fc2.weight.T.shape)}  b2 {tuple(L0.fc2.bias.shape)}')
log(f'  one token: 512 -> {hid.shape[-1]} -> 512')
log('  max |each token alone - whole sentence at once| = %.2e' % (full - alone).abs().max())
zero_share = float((hid.abs() < 1e-3).float().mean())
log(f'  hidden units near zero (|h| < 0.001) for this sentence: {zero_share:.1%} (Marian uses {cfg.activation_function}, the paper uses ReLU)')

# ------------------------------------------------------------------ 5. shared weights and the sqrt(d_model) scale
log('== 5. one matrix for three jobs')
same_enc = enc.embed_tokens.weight.data_ptr() == shared.data_ptr()
same_dec = dec.embed_tokens.weight.data_ptr() == shared.data_ptr()
same_out = model.lm_head.weight.data_ptr() == shared.data_ptr()
log(f'  encoder embedding is the shared matrix: {same_enc}; decoder embedding: {same_dec}; output layer: {same_out}')
log(f'  without sharing: 3 x {shared.numel():,} = {3 * shared.numel():,} numbers; with sharing: {shared.numel():,}')
with torch.no_grad():
    rown = shared.norm(dim=-1)
    pos_all = enc.embed_positions(torch.Size([1, 64]))
log(f'  embedding row norm: median {float(rown.median()):.2f}  (min {float(rown.min()):.2f}, max {float(rown.max()):.2f})')
log(f'  times sqrt(512) = {math.sqrt(D):.2f}: median {float(rown.median()) * math.sqrt(D):.1f}')
log(f'  position vector norm: {float(pos_all[0].norm()):.2f} at pos 0, {float(pos_all[10].norm()):.2f} at pos 10, sqrt(256) = 16')
R['shared'] = dict(row_median=float(rown.median()), scaled=float(rown.median()) * math.sqrt(D))

# ------------------------------------------------------------------ 6. sinusoidal positions
log('== 6. sinusoidal position encoding')


def pe_paper(n, d):
    P = np.zeros((n, d))
    for p in range(n):
        for i in range(d // 2):
            P[p, 2 * i] = math.sin(p / 10000 ** (2 * i / d))
            P[p, 2 * i + 1] = math.cos(p / 10000 ** (2 * i / d))
    return P


P = pe_paper(64, D)
M = pos_all.numpy() if pos_all.dim() == 2 else pos_all[0].numpy()
M = M.reshape(-1, D)[:64]
perm = np.concatenate([np.arange(0, D, 2), np.arange(1, D, 2)])   # Marian stores all sines first, then all cosines
log('  max |paper formula (reordered) - model table| = %.2e' % np.abs(P[:, perm] - M).max())
log('  pos  dim0 sin   dim1 cos   dim2 sin   dim3 cos  ...  dim510 sin  dim511 cos')
for p in [0, 1, 2, 3, 10, 50]:
    log(f'  {p:3d}  {P[p, 0]:8.4f}  {P[p, 1]:8.4f}  {P[p, 2]:8.4f}  {P[p, 3]:8.4f}  ...  {P[p, 510]:9.6f}  {P[p, 511]:9.6f}')
wl = [2 * math.pi * 10000 ** (2 * i / D) for i in [0, 1, 2, 32, 64, 128, 192, 255]]
log('  wavelengths (positions per full turn) for i = 0, 1, 2, 32, 64, 128, 192, 255: ' + ', '.join(f'{w:,.1f}' for w in wl))
log(f'  shortest 2*pi = {2 * math.pi:.2f}; longest limit 10000*2*pi = {10000 * 2 * math.pi:,.0f}')
# relative positions: a fixed rotation for each frequency
k = 5
err = 0
for i in range(D // 2):
    w = 1 / 10000 ** (2 * i / D)
    Rm = np.array([[math.cos(k * w), math.sin(k * w)], [-math.sin(k * w), math.cos(k * w)]])
    for p in range(0, 50):
        err = max(err, np.abs(Rm @ P[p, 2 * i:2 * i + 2] - P[p + k, 2 * i:2 * i + 2]).max())
log(f'  PE(pos + {k}) = rotation(k={k}) applied to PE(pos), max error over pos 0..49 and all 256 pairs: {err:.1e}')
dots = {kk: [float(P[p] @ P[p + kk]) for p in (0, 7, 20, 40)] for kk in (0, 1, 2, 5, 10, 20)}
log('  dot product PE(pos) . PE(pos + k) for pos = 0, 7, 20, 40:')
for kk, v in dots.items():
    log(f'    k = {kk:2d}: ' + '  '.join(f'{x:7.2f}' for x in v))
R['pe'] = dict(P=P[:64, :].tolist(), dots=dots, wl=wl)

# order matters: shuffle the words, the encoder output changes only through positions
log('== 7. without positions, word order is invisible')
with torch.no_grad():
    lay = enc.layers[0]
    e = emb * scale
    perm_t = torch.tensor([3, 1, 2, 0] + list(range(4, ids.shape[1])))       # swap tokens 0 and 3
    out_np = lay(e, attention_mask=None)[0] if isinstance(lay(e, attention_mask=None), tuple) else lay(e, attention_mask=None)
    out_sw = lay(e[:, perm_t], attention_mask=None)
    out_sw = out_sw[0] if isinstance(out_sw, tuple) else out_sw
    d_nopos = float((out_np[:, perm_t] - out_sw).abs().max())
    e2 = e + pos
    o1 = lay(e2, attention_mask=None); o1 = o1[0] if isinstance(o1, tuple) else o1
    o2 = lay(e[:, perm_t] + pos, attention_mask=None); o2 = o2[0] if isinstance(o2, tuple) else o2
    d_pos = float((o1[:, perm_t] - o2).abs().max())
log(f'  swap tokens 0 and 3 ({pieces[0]} <-> {pieces[3]}), no positions: outputs are the same vectors, swapped; max difference {d_nopos:.1e}')
log(f'  same swap with positions added: max difference {d_pos:.2f}')
save('part2', R)
