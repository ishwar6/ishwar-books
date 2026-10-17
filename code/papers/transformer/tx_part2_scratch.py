"""Part 2, built by hand: every block of §3.1, §3.3, §3.4 and §3.5, written from scratch and compared with the
released model (Helsinki-NLP/opus-mt-en-de, MarianMT) on our 12-token sentence.

Method 1 is our own code: plain tensor maths, names from the paper, a comment on each step.
Method 2 is the library: the model's own modules (nn.Embedding, nn.LayerNorm, nn.Linear, the Marian layers).
Each block prints the shapes and the largest difference between the two methods.

  1. the embedding: look up rows of the shared table and multiply by sqrt(d_model)
  2. the sinusoidal position encoding, from the formula, against the model's table
  3. layer normalization by hand against nn.LayerNorm (and the common mistake: the unbiased variance)
  4. the position-wise feed-forward network by hand against fc1 / fc2 (and swish against ReLU)
  5. multi-head self-attention by hand (Part 3 explains it; here we only need it to build a layer)
  6. the residual wrapper "Add & Norm", and one full encoder layer from our pieces against the Marian layer
  7. the whole encoder (6 layers) from our pieces against the library encoder
  8. the tied output layer: decoder vector times the embedding table, against the model's logits
  9. what goes wrong when a learner forgets one detail
Run: ~/.venvs/attn/bin/python tx_part2_scratch.py   (CPU, a few seconds after the download)"""
import math
import torch
import torch.nn.functional as F
from transformers import MarianMTModel, MarianTokenizer
from common import Log, save

torch.manual_seed(0)
log = Log('part2_scratch')
R = {}
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME).eval()
enc = model.model.encoder
d_model = model.config.d_model                      # 512
h = model.config.encoder_attention_heads            # 8
d_k = d_model // h                                  # 64

SRC = 'The cat sat on the mat because it was tired.'
ids = tok([SRC], return_tensors='pt')['input_ids']  # (1, 12): one sentence of 12 token ids
n = ids.shape[1]
log(f'sentence: {SRC}')
log(f'ids {tuple(ids.shape)}: {ids[0].tolist()}')


def diff(a, b):
    return (a - b).abs().max().item()


# ------------------------------------------------------------------ 1. embedding times sqrt(d_model)
def embed(ids, E):
    """Method 1: pick one row of the table E for each token id, then scale by sqrt(d_model)."""
    rows = E[ids]                                   # (1, n) ids -> (1, n, d_model) vectors
    return rows * math.sqrt(d_model)                # sqrt(512) = 22.627


E = model.model.shared.weight                       # (58101, 512): one row per vocabulary token
with torch.no_grad():
    x_ours = embed(ids, E)
    x_lib = enc.embed_tokens(ids) * enc.embed_scale  # Method 2: nn.Embedding, then Marian's scale
log('')
log('== 1. embedding: row lookup times sqrt(d_model)')
log(f'  table E {tuple(E.shape)}, ids {tuple(ids.shape)} -> vectors {tuple(x_ours.shape)}')
log(f'  row 12064 ("▁cat") first 4 numbers: {[round(v, 4) for v in E[12064, :4].tolist()]}')
log(f'  after the scale x 22.627:          {[round(v, 4) for v in x_ours[0, 1, :4].tolist()]}')
log(f'  max |ours - library| = {diff(x_ours, x_lib):.2e}')


# ------------------------------------------------------------------ 2. sinusoidal position encoding
def positional_encoding(n_pos, d_model):
    """Method 1: the formula of §3.5, in the paper's order (sin in even dimensions, cos in odd ones)."""
    pos = torch.arange(n_pos, dtype=torch.float64).unsqueeze(1)          # (n_pos, 1): 0, 1, 2, ...
    i = torch.arange(d_model // 2, dtype=torch.float64).unsqueeze(0)     # (1, 256): pair number i
    angle = pos / 10000 ** (2 * i / d_model)                             # (n_pos, 256): one angle per pair
    PE = torch.zeros(n_pos, d_model, dtype=torch.float64)
    PE[:, 0::2] = torch.sin(angle)                                       # dimensions 0, 2, 4, ...
    PE[:, 1::2] = torch.cos(angle)                                       # dimensions 1, 3, 5, ...
    return PE


with torch.no_grad():
    PE = positional_encoding(n, d_model)                                 # (12, 512)
    PE_marian = torch.cat([PE[:, 0::2], PE[:, 1::2]], dim=1)             # Marian: all 256 sines, then all 256 cosines
    PE_lib = enc.embed_positions(ids.shape)                              # Method 2: the model's frozen table
log('')
log('== 2. sinusoidal position encoding')
log(f'  PE {tuple(PE.shape)}; position 1, dims 0-3: {[round(v, 4) for v in PE[1, :4].tolist()]}')
log(f'  max |our formula (paper order) - model table| = {diff(PE.float(), PE_lib):.2f}  (different order)')
log(f'  max |our formula (sines first) - model table| = {diff(PE_marian.float(), PE_lib):.2e}')
with torch.no_grad():
    x0 = x_ours + PE_marian.float()                                      # (1, 12, 512): the input of layer 1
    eo = enc(input_ids=ids, output_hidden_states=True)
log(f'  input of layer 1 = embedding + PE {tuple(x0.shape)}; max |ours - library| = {diff(x0, eo.hidden_states[0]):.2e}')


# ------------------------------------------------------------------ 3. layer normalization
def layer_norm(x, gamma, beta, eps=1e-5):
    """Method 1: normalise each token vector over its own d_model numbers."""
    mu = x.mean(dim=-1, keepdim=True)                       # (1, n, 1): one mean per token
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)        # (1, n, 1): divide by d, not d - 1
    x_hat = (x - mu) / torch.sqrt(var + eps)                # mean 0, standard deviation 1
    return gamma * x_hat + beta                             # learned scale and shift, (d_model,) each


L0 = enc.layers[0]
ln = L0.self_attn_layer_norm                                 # an nn.LayerNorm(512) with eps = 1e-5
with torch.no_grad():
    v = x0 + 0.0                                             # any (1, 12, 512) tensor works; we use layer 1's input
    ln_ours = layer_norm(v, ln.weight, ln.bias, ln.eps)
    ln_lib = ln(v)                                           # Method 2
    x_hat = layer_norm(v, torch.ones(d_model), torch.zeros(d_model))
log('')
log('== 3. layer normalization')
log(f'  input {tuple(v.shape)}: token 1 ("▁cat") mean {v[0, 1].mean():.3f}, std {v[0, 1].std(unbiased=False):.3f}')
log(f'  after normalising (before gamma, beta): mean {x_hat[0, 1].mean():.1e}, std {x_hat[0, 1].std(unbiased=False):.4f}')
log(f'  gamma: mean {ln.weight.mean():.3f}, range {ln.weight.min():.3f} to {ln.weight.max():.3f}; '
    f'beta: mean {ln.bias.mean():.3f}')
log(f'  max |ours - nn.LayerNorm| = {diff(ln_ours, ln_lib):.2e}')
with torch.no_grad():
    mu = v.mean(-1, keepdim=True)
    wrong = (v - mu) / torch.sqrt(v.var(-1, unbiased=True, keepdim=True) + ln.eps) * ln.weight + ln.bias
log(f'  mistake, dividing by d - 1 (torch.var default): max |wrong - nn.LayerNorm| = {diff(wrong, ln_lib):.2e}')


# ------------------------------------------------------------------ 4. position-wise feed-forward network
def swish(z):
    return z * torch.sigmoid(z)                              # Marian's activation; the paper uses max(0, z)


def ffn(x, W1, b1, W2, b2, act):
    """Method 1: FFN(x) = act(x W1 + b1) W2 + b2, applied to every token with the same weights."""
    hidden = act(x @ W1 + b1)                                # (1, n, 512) @ (512, 2048) -> (1, n, 2048)
    return hidden @ W2 + b2                                  # (1, n, 2048) @ (2048, 512) -> (1, n, 512)


W1, b1 = L0.fc1.weight.T, L0.fc1.bias                        # nn.Linear stores (out, in); the paper's W1 is (in, out)
W2, b2 = L0.fc2.weight.T, L0.fc2.bias
with torch.no_grad():
    s = L0.self_attn_layer_norm(x0 + L0.self_attn(x0)[0])    # the FFN input in layer 1: (1, 12, 512)
    f_ours = ffn(s, W1, b1, W2, b2, swish)
    f_lib = L0.fc2(L0.activation_fn(L0.fc1(s)))              # Method 2: two nn.Linear layers and SiLU
    f_relu = ffn(s, W1, b1, W2, b2, torch.relu)
    hid_relu = torch.relu(s @ W1 + b1)
log('')
log('== 4. position-wise feed-forward network')
log(f'  W1 {tuple(W1.shape)}  b1 {tuple(b1.shape)}  W2 {tuple(W2.shape)}  b2 {tuple(b2.shape)}')
log(f'  x {tuple(s.shape)} -> x W1 + b1 {tuple((s @ W1 + b1).shape)} -> act -> times W2 + b2 {tuple(f_ours.shape)}')
log(f'  max |ours (swish) - library fc1, SiLU, fc2| = {diff(f_ours, f_lib):.2e}')
log(f'  swish(z) = z * sigmoid(z): swish(-2) = {swish(torch.tensor(-2.0)):.3f}, swish(0) = 0, swish(2) = {swish(torch.tensor(2.0)):.3f}')
log(f'  with ReLU in place of swish: {float((hid_relu == 0).float().mean()):.1%} of the 2048 hidden numbers are exactly 0; '
    f'output changes by up to {diff(f_relu, f_lib):.2f}')


# ------------------------------------------------------------------ 5. multi-head self-attention (Part 3 explains it)
def self_attention(x, attn):
    """Method 1: softmax(Q K^T / sqrt(d_k)) V in h = 8 heads, then the output projection W_O."""
    B, n, _ = x.shape
    Q = x @ attn.q_proj.weight.T + attn.q_proj.bias              # (1, n, 512)
    K = x @ attn.k_proj.weight.T + attn.k_proj.bias              # (1, n, 512)
    V = x @ attn.v_proj.weight.T + attn.v_proj.bias              # (1, n, 512)
    Q = Q.view(B, n, h, d_k).transpose(1, 2)                     # (1, 8, n, 64): 8 heads of 64 numbers
    K = K.view(B, n, h, d_k).transpose(1, 2)
    V = V.view(B, n, h, d_k).transpose(1, 2)
    scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)            # (1, 8, n, n): every token against every token
    weights = torch.softmax(scores, dim=-1)                      # each row adds up to 1
    heads = weights @ V                                          # (1, 8, n, 64)
    concat = heads.transpose(1, 2).reshape(B, n, d_model)        # (1, n, 512): the 8 heads side by side
    return concat @ attn.out_proj.weight.T + attn.out_proj.bias  # (1, n, 512)


with torch.no_grad():
    a_ours = self_attention(x0, L0.self_attn)
    a_lib = L0.self_attn(x0)[0]
log('')
log('== 5. multi-head self-attention (shapes; Part 3 explains the formula)')
log(f'  x {tuple(x0.shape)} -> Q, K, V {tuple(x0.shape)} -> 8 heads {(1, h, n, d_k)} -> scores {(1, h, n, n)} '
    f'-> output {tuple(a_ours.shape)}')
log(f'  max |ours - library attention| = {diff(a_ours, a_lib):.2e}')


# ------------------------------------------------------------------ 6. Add & Norm, and one full encoder layer
def add_and_norm(x, sublayer, norm):
    """The residual wrapper of §3.1: LayerNorm(x + Sublayer(x))."""
    return norm(x + sublayer(x))


def encoder_layer(x, L):
    """One encoder layer from our own pieces, with the weights of the Marian layer L."""
    ln1 = lambda t: layer_norm(t, L.self_attn_layer_norm.weight, L.self_attn_layer_norm.bias)
    ln2 = lambda t: layer_norm(t, L.final_layer_norm.weight, L.final_layer_norm.bias)
    attn = lambda t: self_attention(t, L.self_attn)
    ff = lambda t: ffn(t, L.fc1.weight.T, L.fc1.bias, L.fc2.weight.T, L.fc2.bias, swish)
    x = add_and_norm(x, attn, ln1)        # sub-layer 1: tokens exchange information
    x = add_and_norm(x, ff, ln2)          # sub-layer 2: each token alone
    return x


with torch.no_grad():
    y_ours = encoder_layer(x0, L0)
    y_lib = L0(x0, attention_mask=None)
    y_lib = y_lib[0] if isinstance(y_lib, tuple) else y_lib
    sub1 = x0 + a_ours
log('')
log('== 6. Add & Norm, and encoder layer 1 from our pieces')
log(f'  x {tuple(x0.shape)}: norm of token 1 ("▁cat") {x0[0, 1].norm():.1f}; attention adds a change of norm {a_ours[0, 1].norm():.1f}')
log(f'  x + Sublayer(x) {tuple(sub1.shape)}; after the layer norm, norm {layer_norm(sub1, 1, 0)[0, 1].norm():.1f} '
    f'= sqrt(512) before gamma and beta')
log(f'  our encoder layer {tuple(y_ours.shape)}; max |ours - Marian layer 1| = {diff(y_ours, y_lib):.2e}')

# ------------------------------------------------------------------ 7. the whole encoder from our pieces
log('')
log('== 7. the whole encoder: our layer six times, with the weights of each Marian layer')
with torch.no_grad():
    z = embed(ids, E) + PE_marian.float()            # (1, 12, 512)
    for i, L in enumerate(enc.layers, 1):
        z = encoder_layer(z, L)                       # six layers, each with its own weights
        log(f'  ours, after layer {i}: {tuple(z.shape)}; max |ours - library| = {diff(z, eo.hidden_states[i]):.2e}')
log(f'  encoder output z {tuple(z.shape)}; max |ours - library encoder| = {diff(z, eo.last_hidden_state):.2e}')
R['encoder_err'] = diff(z, eo.last_hidden_state)

# ------------------------------------------------------------------ 8. the tied output layer
log('')
log('== 8. the output layer reuses the embedding table')
de_ids = tok(text_target=['Die Katze saß auf der Matte, weil sie müde war.'], return_tensors='pt')['input_ids']
with torch.no_grad():
    out = model(input_ids=ids, labels=de_ids, output_hidden_states=True)
    top = out.decoder_hidden_states[-1]                       # (1, m, 512): the last decoder layer
    logits_ours = top @ E.T + model.final_logits_bias         # (1, m, 512) @ (512, 58101) -> (1, m, 58101)
log(f'  decoder vectors {tuple(top.shape)} @ E^T {tuple(E.T.shape)} -> logits {tuple(logits_ours.shape)}')
log(f'  max |ours - model logits| = {diff(logits_ours, out.logits):.2e}')
p0 = torch.softmax(logits_ours[0, 0], -1)
best = p0.topk(3)
log('  position 0 (after the start token), top 3: ' + ', '.join(
    f'{tok.convert_ids_to_tokens([int(i)])[0]} {float(p):.3f}' for p, i in zip(best.values, best.indices)))
rank = int((logits_ours[0, 1] > logits_ours[0, 1, de_ids[0, 1]]).sum()) + 1
log(f'  position 1: the row of {tok.convert_ids_to_tokens([int(de_ids[0, 1])])[0]} gets score rank {rank} of 58,101')

# ------------------------------------------------------------------ 9. the usual mistakes
log('')
log('== 9. the usual mistakes, measured on the encoder output z')


def encode(ids, scale=math.sqrt(d_model), pe=PE_marian.float(), use_pe=True):
    z = E[ids] * scale + (pe if use_pe else 0)
    for L in enc.layers:
        z = encoder_layer(z, L)
    return z


with torch.no_grad():
    ref = eo.last_hidden_state
    cases = {
        'forget the sqrt(d_model) scale': encode(ids, scale=1.0),
        'positions in the paper order, not sines first': encode(ids, pe=PE.float()),
        'no position encoding at all': encode(ids, use_pe=False),
    }
for k, zz in cases.items():
    c = F.cosine_similarity(zz[0], ref[0], dim=-1).mean()
    log(f'  {k:46s} max difference {diff(zz, ref):6.2f}, mean cosine with the right z {c:.3f}')
with torch.no_grad():
    right_out = model.generate(encoder_outputs=(z,), attention_mask=torch.ones_like(ids),
                               num_beams=1, do_sample=False, max_new_tokens=30)
    wrong_out = model.generate(encoder_outputs=(cases['forget the sqrt(d_model) scale'],),
                               attention_mask=torch.ones_like(ids), num_beams=1, do_sample=False, max_new_tokens=30)
log(f'  the decoder reads our correct z and writes:  {tok.decode(right_out[0], skip_special_tokens=True)}')
log(f'  the decoder reads the unscaled z and writes: {tok.decode(wrong_out[0], skip_special_tokens=True)}')
save('part2_scratch', R)
