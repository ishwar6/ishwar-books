"""Precompute the real-translation part of the Transformer demo (/papers/transformer/demo).

The released model Helsinki-NLP/opus-mt-en-de (MarianMT, the paper's base shapes: 6 + 6 layers, d_model 512, 8 heads,
74.4 million weights) is too heavy to download into a web page, so we run it here, in Python, and save its numbers.

For each English sentence this script
  1. translates it greedily (num_beams=1, as in Parts 1 and 3 of the series), one token per decoder call;
  2. runs one teacher-forced forward pass on the source and the translation with output_attentions=True, and keeps
     the encoder self-attention, the decoder (masked) self-attention and the cross-attention of every layer and head;
  3. keeps the top-5 next-token probabilities of every decoder step.

File format (public/demos/transformer/marian/):
  sentences.json  list of {slug, en, de, src: [tokens], tgt: [decoder input tokens, starting with <pad>],
                  out: [the token each decoder position chose], top5: [[[token, p], ...] per step], note,
                  enc: [L, H, n, n], cross: [L, H, m, n], dec: [L, H, m, m], bin: file name}
  <slug>.bin      the three attention blocks, one after the other (enc, cross, dec), each as
                    rows float16 scales (the largest weight of each row), then rows x cols uint8 levels;
                  weight = level / 255 * row scale.  A weight of 0 (the masked future) stays exactly 0.
Writes results/marian.json with the check of the stored weights against the model's float32 weights.
"""
import json, os, sys
import numpy as np
import torch
from transformers import MarianMTModel, MarianTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'transformer', 'marian')
RES = os.path.join(HERE, 'results')
os.makedirs(PUB, exist_ok=True)
os.makedirs(RES, exist_ok=True)
NAME = 'Helsinki-NLP/opus-mt-en-de'

SENTENCES = [
    ('cat', 'The cat sat on the mat because it was tired.',
     'The sentence of Parts 2, 3 and 5. German needs "sie" for "it", because "Katze" is feminine.'),
    ('animal-tired', "The animal didn't cross the street because it was too tired.",
     'Here "it" is the animal (das Tier), so German needs "es". The model writes "es".'),
    ('animal-wide', "The animal didn't cross the street because it was too wide.",
     'One word changes, and "it" is now the street (die Straße), which needs "sie". The model still writes "es", as for the animal: it gets this one wrong.'),
    ('trophy', "The trophy didn't fit into the suitcase because it was too big.",
     'Here "it" is the trophy. The model writes "Trophäe", a feminine word, so "sie" is right.'),
    ('suitcase', "The trophy didn't fit into the suitcase because it was too small.",
     'One word changes, and "it" is now the suitcase (der Koffer), which needs "er". The model still writes "sie", the pronoun of the trophy: it gets this one wrong.'),
    ('making', 'It is in this spirit that a majority of American governments have passed new laws since 2009 '
               'making the registration or voting process more difficult.',
     'The sentence of the paper\'s Figure 3: "making" belongs with "more difficult", four words away.'),
    ('law', 'The Law will never be perfect, but its application should be just - this is what we are missing, '
            'in my opinion.',
     'The sentence of the paper\'s Figures 4 and 5: "its" refers back to "Law".'),
    ('attention', 'Attention is all you need.', 'The title of the paper.'),
    ('agreement', 'The agreement on the European Economic Area was signed in August 1992.',
     'A sentence where the German word order changes: the verb moves to the end.'),
]


REFERENCE = {'cat': ('cross', 2), 'making': ('enc', 4)}     # (kind, layer index) kept in float32 for the browser check


def pack(a):
    """(rows, cols) float32 weights -> float16 row scales + uint8 levels, and the weights they decode to."""
    scale = a.max(axis=1).astype(np.float16)
    s32 = scale.astype(np.float32)
    lev = np.where(s32[:, None] > 0, np.rint(a / np.maximum(s32[:, None], 1e-30) * 255), 0)
    lev = np.clip(lev, 0, 255).astype(np.uint8)
    back = lev.astype(np.float32) / 255 * s32[:, None]
    return scale.tobytes() + lev.tobytes(), back


def main():
    torch.manual_seed(0)
    tok = MarianTokenizer.from_pretrained(NAME)
    model = MarianMTModel.from_pretrained(NAME, attn_implementation='eager').eval()
    cfg = model.config
    L, H = cfg.encoder_layers, cfg.encoder_attention_heads
    out, checks, refs = [], [], []
    for slug, en, note in SENTENCES:
        x = tok([en], return_tensors='pt')
        with torch.no_grad():
            gen = model.generate(**x, num_beams=1, do_sample=False, max_new_tokens=80)
            dec_in = gen[:, :-1]                              # <pad> t1 ... t_(m-1); the last generated token is </s>
            o = model(**x, decoder_input_ids=dec_in, output_attentions=True)
        target = gen[0, 1:]
        logits = o.logits[0]
        assert torch.equal(logits.argmax(-1), target), 'teacher forcing must give the greedy tokens'
        probs = logits.softmax(-1)
        top = probs.topk(5, dim=-1)
        top5 = [[[tok.convert_ids_to_tokens(int(i)), round(float(p), 5)] for i, p in zip(ti, tp)]
                for ti, tp in zip(top.indices, top.values)]
        enc = torch.stack(o.encoder_attentions)[:, 0].numpy()      # (L, H, n, n)
        cross = torch.stack(o.cross_attentions)[:, 0].numpy()      # (L, H, m, n)
        dec = torch.stack(o.decoder_attentions)[:, 0].numpy()      # (L, H, m, m)
        blob, worst = b'', {}
        for name, a in (('enc', enc), ('cross', cross), ('dec', dec)):
            rows = a.reshape(-1, a.shape[-1])
            b, back = pack(rows)
            blob += b
            d = np.abs(back - rows)
            worst[name] = {'max_abs_diff': float(d.max()),
                           'argmax_equal_rows': int((back.argmax(1) == rows.argmax(1)).sum()), 'rows': rows.shape[0],
                           'future_weight_max': float(np.triu(a, 1).max()) if name == 'dec' else None}
        with open(os.path.join(PUB, f'{slug}.bin'), 'wb') as f:
            f.write(blob)
        src = tok.convert_ids_to_tokens(x['input_ids'][0])
        tgt = tok.convert_ids_to_tokens(dec_in[0])
        outt = tok.convert_ids_to_tokens(target)
        de = tok.decode(gen[0], skip_special_tokens=True)
        out.append({'slug': slug, 'en': en, 'de': de, 'note': note, 'src': src, 'tgt': tgt, 'out': outt, 'top5': top5,
                    'enc': list(enc.shape), 'cross': list(cross.shape), 'dec': list(dec.shape), 'bin': f'{slug}.bin',
                    'bytes': len(blob)})
        checks.append({'slug': slug, 'de': de, **worst})
        if slug in REFERENCE:                      # float32 rows for check_browser.mjs
            kind, layer = REFERENCE[slug]
            a = {'enc': enc, 'cross': cross, 'dec': dec}[kind][layer]
            refs.append({'slug': slug, 'kind': kind, 'layer': layer, 'heads': a.round(6).tolist()})
        print(f'{slug:13s} n={len(src):2d} m={len(tgt):2d} {len(blob)/1024:6.1f} KB  {de}')
        for k, v in worst.items():
            print(f'   {k:5s} max |stored - model| {v["max_abs_diff"]:.4f}  same argmax {v["argmax_equal_rows"]}/{v["rows"]}'
                  + (f'  future weight max {v["future_weight_max"]:.1e}' if v['future_weight_max'] is not None else ''))
    meta = {'model': NAME, 'layers': L, 'heads': H, 'd_model': cfg.d_model,
            'params': sum(p.numel() for p in model.parameters()), 'sentences': out,
            'max_abs_diff': max(c[k]['max_abs_diff'] for c in checks for k in ('enc', 'cross', 'dec'))}
    with open(os.path.join(PUB, 'sentences.json'), 'w') as f:
        json.dump(meta, f, ensure_ascii=False, separators=(',', ':'))
    with open(os.path.join(RES, 'marian.json'), 'w') as f:
        json.dump({'model': NAME, 'torch': torch.__version__, 'checks': checks,
                   'overall_max_abs_diff': max(c[k]['max_abs_diff'] for c in checks for k in ('enc', 'cross', 'dec'))},
                  f, indent=1, ensure_ascii=False)
    with open(os.path.join(RES, 'marian_reference.json'), 'w') as f:
        json.dump(refs, f)
    print('params', meta['params'])


if __name__ == '__main__':
    main()
