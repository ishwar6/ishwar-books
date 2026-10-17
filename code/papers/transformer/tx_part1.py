"""Part 1 (the big idea): what the abstract, §1 and §2 claim, checked with small real runs.

1. A released Transformer translation model (Helsinki-NLP/opus-mt-en-de, MarianMT) translates one sentence,
   and we print its shapes: 6 + 6 layers, d_model 512, 8 heads, d_ff 2048.
2. The recurrence of §1, h_t = tanh(w * h_{t-1} + u * x_t), worked by hand with one number per state.
3. The sequential loop against one attention step: a GRU layer (one step per word) and one self-attention layer
   (all words at once), both with width 512, timed on the CPU for growing sentence lengths.
4. Training is parallel, generation is not: Marian scores a whole German sentence in one decoder call
   (teacher forcing), but writes it one token per call.
5. The path length of §2: how many layers a signal crosses between two words d positions apart.
Run: ~/.venvs/attn/bin/python tx_part1.py   (CPU, about one minute after the model download)"""
import math, time
import torch
from common import Log, save

log = Log('part1')
torch.manual_seed(0)
torch.set_num_threads(4)
out = {}

# ------------------------------------------------------------------ 1. the released model
from transformers import MarianMTModel, MarianTokenizer
NAME = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(NAME)
model = MarianMTModel.from_pretrained(NAME).eval()
c = model.config
log('== 1. A released Transformer: ' + NAME)
log(f'encoder layers {c.encoder_layers}, decoder layers {c.decoder_layers}, d_model {c.d_model}, '
    f'heads {c.encoder_attention_heads}, d_ff {c.encoder_ffn_dim}, vocabulary {c.vocab_size:,}')
n_params = sum(p.numel() for p in model.parameters())
log(f'parameters: {n_params:,}')
src = 'The dog sat on the mat because it was tired.'
enc = tok(src, return_tensors='pt')
with torch.no_grad():
    gen = model.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=40)
de = tok.decode(gen[0], skip_special_tokens=True)
log(f'English: {src}')
log(f'source tokens ({enc.input_ids.shape[1]}): {tok.convert_ids_to_tokens(enc.input_ids[0])}')
log(f'German : {de}')
log(f'output tokens ({gen.shape[1]}): {tok.convert_ids_to_tokens(gen[0])}')
out['model'] = dict(layers=[c.encoder_layers, c.decoder_layers], d_model=c.d_model, heads=c.encoder_attention_heads,
                    d_ff=c.encoder_ffn_dim, vocab=c.vocab_size, params=n_params, src=src, de=de)

# ------------------------------------------------------------------ 2. the recurrence by hand
log('')
log('== 2. The recurrence h_t = tanh(w * h_(t-1) + u * x_t), with w = 0.5, u = 1.0, h_0 = 0')
w, u, h = 0.5, 1.0, 0.0
xs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
hs = []
for t, x in enumerate(xs, 1):
    pre = w * h + u * x
    h = math.tanh(pre)
    hs.append(round(h, 4))
    log(f'  t={t}: x={x:.0f}  h = tanh({w} x {hs[-2] if t > 1 else 0.0:.4f} + {u} x {x:.0f}) = tanh({pre:.4f}) = {h:.4f}')
log('  the trace of word 1 shrinks by about half at each step: it must pass through every later state')
out['recurrence'] = hs

# ------------------------------------------------------------------ 3. a loop against one attention step
log('')
log('== 3. One GRU layer (a loop over the words) against one self-attention layer (all words at once)')
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
log(f'   width 512, batch 32, device {DEV} (Apple M5 Pro laptop GPU), forward pass only, fastest of 20 runs')
d, B = 512, 32
gru = torch.nn.GRU(d, d, batch_first=True)
att = torch.nn.MultiheadAttention(d, 8, batch_first=True).to(DEV)


def sync():
    if DEV == 'mps':
        torch.mps.synchronize()


def fastest(f, k=20):
    f(); sync()
    best = 1e9
    for _ in range(k):
        t0 = time.perf_counter(); f(); sync(); best = min(best, time.perf_counter() - t0)
    return best


def gru_loop(x):
    """The GRU of §1, written as the loop that it is: step t needs h_(t-1)."""
    cell = torch.nn.GRUCell(d, d)
    cell.weight_ih, cell.weight_hh, cell.bias_ih, cell.bias_hh = gru.weight_ih_l0, gru.weight_hh_l0, gru.bias_ih_l0, gru.bias_hh_l0
    h = torch.zeros(x.shape[0], d)
    for t in range(x.shape[1]):
        h = cell(x[:, t], h)
    return h


rows = []
log(f'{"words n":>8} {"GRU steps":>10} {"GRU ms":>8} {"attention steps":>16} {"attention ms":>13} {"GRU / attention":>16}')
with torch.no_grad():
    x0 = torch.randn(B, 8, d)
    diff = (gru(x0)[1][0] - gru_loop(x0)).abs().max().item()
    gru = gru.to(DEV)
    for n in [16, 64, 256, 1024]:
        x = torch.randn(B, n, d, device=DEV)
        tg = fastest(lambda: gru(x))
        ta = fastest(lambda: att(x, x, x, need_weights=False))
        rows.append(dict(n=n, gru_ms=round(tg * 1e3, 1), att_ms=round(ta * 1e3, 1), ratio=round(tg / ta, 1)))
        log(f'{n:>8} {n:>10} {tg * 1e3:>8.1f} {1:>16} {ta * 1e3:>13.1f} {tg / ta:>15.1f}x')
log(f'check: the library GRU equals our step-by-step loop, max difference {diff:.1e}')
out['timing'] = rows

# ------------------------------------------------------------------ 4. training is parallel, generation is not
log('')
log('== 4. Marian: score a known German sentence in one call, or write it one token per call')
labels = gen[:, 1:]                        # the German tokens after the start token
with torch.no_grad():
    t0 = time.perf_counter()
    o = model(**enc, labels=labels)
    t_one = time.perf_counter() - t0
    t0 = time.perf_counter()
    g2 = model.generate(**enc, num_beams=1, do_sample=False, max_new_tokens=40)
    t_gen = time.perf_counter() - t0
m = labels.shape[1]
pred = o.logits.argmax(-1)
log(f'German tokens m = {m}')
log(f'teacher forcing: 1 decoder call gives logits of shape {tuple(o.logits.shape)} (one row per position), '
    f'{t_one * 1e3:.0f} ms')
log(f'  the {m} predicted tokens equal the generated ones: {bool((pred == labels).all())}; mean loss {o.loss.item():.3f}')
log(f'generation: {g2.shape[1] - 1} decoder calls, one per token, {t_gen * 1e3:.0f} ms')
out['teacher_forcing'] = dict(m=m, logits=list(o.logits.shape), loss=round(o.loss.item(), 3), one_ms=round(t_one * 1e3),
                              gen_ms=round(t_gen * 1e3), same=bool((pred == labels).all()))

# ------------------------------------------------------------------ 5. path length (§2)
log('')
log('== 5. Layers a signal crosses between two words d positions apart')
log('   ConvS2S: kernel width 3, each layer reaches 2 more words back; ByteNet: kernel width 2, dilations 1, 2, 4, ...')
log(f'{"d":>5} {"recurrent":>10} {"ConvS2S":>8} {"ByteNet":>8} {"self-attention":>15}')
paths = []
for dist in [2, 10, 50, 200]:
    rnn = dist                                   # d steps of the loop
    conv = math.ceil(dist / 2)                   # a causal kernel of width 3 reaches 2 more words back per layer
    byte = math.ceil(math.log2(dist + 1))        # kernel 2, dilations 1, 2, 4, ...: reach 1, 3, 7, 15 = 2^L - 1
    paths.append(dict(d=dist, rnn=rnn, conv=conv, bytenet=byte, attention=1))
    log(f'{dist:>5} {rnn:>10} {conv:>8} {byte:>8} {1:>15}')
out['paths'] = paths
save('part1', out)
