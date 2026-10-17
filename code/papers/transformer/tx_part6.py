"""Part 6 experiments (results, variations, impact).
1. Table 2 arithmetic: training FLOPs = hours x GPUs x sustained TFLOPS (footnote 5), for the base and big models.
2. Table 3 arithmetic: parameter counts of the base, big and some variation rows from the shapes alone.
3. A released Transformer translator (Helsinki-NLP/opus-mt-en-de, MarianMT, base shapes): greedy versus beam 4
   with length penalty 0.6 (the paper's decoding settings), and the input length + 50 limit.
4. The appendix sentences (Figures 3 and 4): where the word 'making' and the word 'its' attend in encoder layer 5 of 6.
Marian is not the paper's model: other data, other vocabulary, other training. The heads will not match the figures."""
import torch
from transformers import MarianMTModel, MarianTokenizer
from common import Log, save

log = Log('part6')
out = {}

# ---------------------------------------------------------------- 1. Table 2: training cost
log('== 1. Training cost in FLOPs (Table 2, footnote 5) ==')
P100 = 9.5e12
for name, hours in [('base', 12), ('big', 3.5 * 24)]:
    flops = hours * 3600 * 8 * P100
    log(f'{name:5s}: {hours:5.1f} h x 3600 s x 8 GPUs x 9.5e12 FLOP/s = {flops:.2e} FLOPs')
    out[f'flops_{name}'] = flops

# ---------------------------------------------------------------- 2. Table 3: parameters from shapes
log('')
log('== 2. Parameters from the shapes (Table 3), shared vocabulary of 37,000 tokens ==')
V = 37000


def params(N=6, d=512, dff=2048, h=8, dk=None, dv=None):
    dk = dk or d // h
    dv = dv or d // h
    attn = d * h * dk * 2 + h * dk * 2 + d * h * dv + h * dv + h * dv * d + d   # W_Q, W_K, W_V, W_O with biases
    ffn = d * dff + dff + dff * d + d
    ln = 2 * d
    enc = N * (attn + ffn + 2 * ln)
    dec = N * (2 * attn + ffn + 3 * ln)
    emb = V * d                         # one matrix shared by both embeddings and the output layer
    return enc + dec + emb, enc, dec, emb


rows = [('base', {}), ('(A) h=1, dk=dv=512', dict(h=1, dk=512, dv=512)), ('(A) h=16, dk=dv=32', dict(h=16, dk=32, dv=32)),
        ('(B) dk=16', dict(dk=16)), ('(C) N=2', dict(N=2)), ('(C) N=8', dict(N=8)),
        ('(C) d_model=256, dk=dv=32', dict(d=256, dk=32, dv=32)), ('(C) d_model=1024, dk=dv=128', dict(d=1024, dk=128, dv=128)),
        ('(C) d_ff=4096', dict(dff=4096)), ('big', dict(d=1024, dff=4096, h=16))]
paper = {'base': 65, '(B) dk=16': 58, '(C) N=2': 36, '(C) N=8': 80, '(C) d_model=256, dk=dv=32': 28,
         '(C) d_model=1024, dk=dv=128': 168, '(C) d_ff=4096': 90, 'big': 213}
log(f'{"row":30s} {"ours (M)":>9s} {"paper (M)":>9s}')
for name, kw in rows:
    tot, enc, dec, emb = params(**kw)
    log(f'{name:30s} {tot / 1e6:9.1f} {str(paper.get(name, "same")):>9s}')
    out[f'params_{name}'] = tot
tot, enc, dec, emb = params()
log(f'base split: encoder {enc / 1e6:.1f} M, decoder {dec / 1e6:.1f} M, shared embedding {emb / 1e6:.1f} M')

# ---------------------------------------------------------------- 3. Translation with the paper's decoding settings
log('')
log('== 3. A released base-size Transformer: Helsinki-NLP/opus-mt-en-de ==')
name = 'Helsinki-NLP/opus-mt-en-de'
tok = MarianTokenizer.from_pretrained(name)
model = MarianMTModel.from_pretrained(name, attn_implementation='eager').eval()
c = model.config
n_par = sum(p.numel() for p in model.parameters() if p.requires_grad) - 0
log(f'layers {c.encoder_layers}+{c.decoder_layers}, d_model {c.d_model}, heads {c.encoder_attention_heads}, '
    f'd_ff {c.encoder_ffn_dim}, vocabulary {c.vocab_size}, parameters {n_par / 1e6:.1f} M')
out['marian_params'] = n_par

src = 'The law will never be perfect, but its application should be just.'
x = tok([src], return_tensors='pt')
n_in = x['input_ids'].shape[1]
with torch.no_grad():
    g = model.generate(**x, num_beams=1, do_sample=False, max_new_tokens=n_in + 50)
    b = model.generate(**x, num_beams=4, length_penalty=0.6, max_new_tokens=n_in + 50)
log(f'source ({n_in} tokens): {src}')
log(f'greedy          : {tok.decode(g[0], skip_special_tokens=True)}')
log(f'beam 4, alpha .6: {tok.decode(b[0], skip_special_tokens=True)}')
log(f'output limit = input length + 50 = {n_in + 50} tokens; used {b.shape[1] - 1}')

# ---------------------------------------------------------------- 4. The appendix sentences
log('')
log('== 4. Encoder self-attention, layer 5 of 6, the sentences of Figures 3 and 4 ==')


def attend(sentence, word, layer=4, top=3):
    x = tok([sentence], return_tensors='pt')
    pieces = tok.convert_ids_to_tokens(x['input_ids'][0])
    with torch.no_grad():
        enc = model.get_encoder()(**x, output_attentions=True)
    A = enc.attentions[layer][0]                 # heads x n x n
    i = next(k for k, p in enumerate(pieces) if p.lstrip('▁') == word)
    log(f'sentence: {sentence}')
    log(f'{len(pieces)} tokens; query token "{pieces[i]}" at position {i}')
    res = []
    for hd in range(A.shape[0]):
        row = A[hd, i]
        v, idx = row.topk(top)
        far = sum(row[j].item() for j in range(len(pieces)) if abs(j - i) >= 5)
        s = ', '.join(f'{pieces[j]} {w:.2f}' for w, j in zip(v.tolist(), idx.tolist()))
        log(f'  head {hd + 1}: {s}   | weight on tokens 5 or more away: {far:.2f}')
        res.append(dict(head=hd + 1, top=[(pieces[j], round(w, 3)) for w, j in zip(v.tolist(), idx.tolist())], far=round(far, 3)))
    return pieces, A[:, i].tolist(), res


s3 = ('It is in this spirit that a majority of American governments have passed new laws since 2009 '
      'making the registration or voting process more difficult.')
p3, a3, r3 = attend(s3, 'making')
log('')
s4 = 'The Law will never be perfect, but its application should be just - this is what we are missing, in my opinion.'
p4, a4, r4 = attend(s4, 'its')
out.update(fig3=dict(pieces=p3, attn=a3, heads=r3), fig4=dict(pieces=p4, attn=a4, heads=r4))
save('part6', out)
