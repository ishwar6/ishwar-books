"""Figures for Part 2 (architecture and input), drawn from results/part2.json. Writes results/figs_part2.json."""
import json
from figlib import svg, text, box, arrow, heatmap
from bertfig import token, row, line, hbars

R = json.load(open('results/part2.json'))
F = {}


def bank():
    b = R['bank']
    labels = ['[1] river bank', '[2] bank of river', '[3] savings bank', '[4] bank loan']
    parts = heatmap(150, 40, b['cosine'], labels, labels, 58, title='cosine similarity of the output vectors of "bank"', show_values=True, label_w=120)
    parts += [text(530, 70, 'river vs river: 0.747', 't-tick'), text(530, 92, 'money vs money: 0.786', 't-tick'),
              text(530, 114, 'river vs money: 0.432 to 0.479', 't-tick'),
              text(530, 160, 'The input (static) vector', 't-note'), text(530, 180, 'of "bank" is one fixed row', 't-note'),
              text(530, 200, 'of the table: similarity', 't-note'), text(530, 220, '1.000 for every pair.', 't-note')]
    return svg(760, 380, 'Cosine similarity between BERT output vectors for the word bank in four sentences. The two river sentences are close to each other, the two money sentences are close to each other, and river versus money pairs are much less similar.', parts)


F['p2_bank'] = bank()


def stack():
    c = R['bert-base-uncased']['config']
    b = []
    words = ['[CLS]', 'my', 'dog', 'is', 'cute', '[SEP]']
    parts, cx = row(150, 360, words, 70, 10, classes=['box-2', 'box', 'box', 'box', 'box', 'box-2'])
    b += parts + [text(140, 380, 'tokens', 't-muted', 'end')]
    b += [box(140, 300, 490, 34, 'box-4', 8), text(385, 322, 'input embeddings: token + segment + position (768 numbers each)', 't-tick', 'middle')]
    for x in cx:
        b.append(arrow(x, 358, x, 336))
    # layers
    ys = [236, 176, 116]
    names = ['layer 1', 'layer 2', 'layer 12']
    for y, n in zip(ys, names):
        b += [box(140, y, 490, 44, 'box-1', 10), text(150, y + 18, n, 't-note'),
              box(240, y + 8, 180, 28, 'box', 6), text(330, y + 26, 'self-attention, 12 heads', 't-tick', 'middle'),
              box(432, y + 8, 188, 28, 'box', 6), text(526, y + 26, 'feed-forward 768→3072→768', 't-tick', 'middle')]
    b += [arrow(385, 298, 385, 282), arrow(385, 234, 385, 222), text(150, 172.5, '(9 more layers in between)', 't-muted')]
    outs = ['C', 'T₁', 'T₂', 'T₃', 'T₄', 'T₅']
    parts, cx2 = row(150, 66, outs, 70, 10, classes=['box-3'] + ['box-3'] * 5)
    b += parts + [text(140, 86, 'outputs', 't-muted', 'end')]
    for x in cx2:
        b.append(arrow(x, 114, x, 98))
    b = [f'<g transform="translate(-60,0)">'] + b + ['</g>']
    b += [text(590, 84, f'H = {c["H"]}', 't-note'), text(590, 102, 'numbers per token', 't-tick'), text(590, 120, 'at every level', 't-tick'),
          text(590, 186, f'L = {c["L"]} layers', 't-note'), text(590, 204, 'stacked', 't-tick'),
          text(590, 250, f'A = {c["A"]} heads', 't-note'), text(590, 268, 'in every layer', 't-tick'),
          text(20, 30, 'BERT-base: L = 12, H = 768, A = 12', 't-title')]
    return svg(760, 410, 'The BERT-base encoder. Tokens become input embeddings, pass up through 12 identical layers of self-attention and feed-forward, and come out as one 768-number vector per token: C for [CLS] and T for the others.', b)


F['p2_stack'] = stack()


def params():
    f = R['bert-base-uncased']['formula']
    L = R['bert-base-uncased']['config']['L']
    items = [('embeddings', f['embeddings'] / 1e6), ('attention, 12 layers', L * f['attention_per_layer'] / 1e6),
             ('feed-forward, 12 layers', L * f['ffn_per_layer'] / 1e6), ('layer norms, 12 layers', L * f['layernorm_per_layer'] / 1e6),
             ('pooler', f['pooler'] / 1e6)]
    b = hbars(20, 50, items, 380, row_h=34, cls='s1', label_w=190, fmt=lambda v: f'{v:.2f} M', vmax=60,
              title=f'Where the {R["bert-base-uncased"]["counted"]:,} parameters of BERT-base live')
    return svg(760, 240, 'Parameters of BERT-base by part: embeddings 23.84 million, attention 28.35 million, feed-forward 56.67 million, layer norms 0.04 million, pooler 0.59 million.', b)


F['p2_params'] = params()


def embed_sum():
    t = R['figure2']
    toks = t['tokens']
    W, G, x0 = 52, 6, 150
    b = [text(20, 30, 'The real input for the pair ("my dog is cute", "he likes playing")', 't-title')]
    rows = [('input', toks, ['box'] * len(toks), 't-tick'),
            ('token', [f'E{i}' for i in range(len(toks))], ['box-4'] * len(toks), 't-mono'),
            ('segment', ['A' if s == 0 else 'B' for s in t['token_type_ids']], ['box-3' if s == 0 else 'box-1' for s in t['token_type_ids']], 't-mono'),
            ('position', [str(p) for p in t['position_ids']], ['box'] * len(toks), 't-mono')]
    y = 52
    for name, labels, cls, tc in rows:
        if name == 'token':
            labels = [str(i) for i in t['input_ids']]
        parts, cx = row(x0, y, labels, W, G, classes=cls, tcls='t-tick' if name in ('input', 'token') else tc)
        b += parts + [text(x0 - 10, y + 20, name if name != 'token' else 'token id', 't-note', 'end')]
        if name != 'input':
            for x in cx:
                b.append(text(x, y - 6, '+', 't-muted', 'middle') if name != 'token' else '')
        y += 52
    b += [text(x0 - 10, y + 20, 'sum', 't-note', 'end'), box(x0, y, len(toks) * (W + G) - G, 30, 'box-on', 8),
          text(x0 + (len(toks) * (W + G) - G) / 2, y + 20, 'add the three 768-number vectors, then LayerNorm (and dropout in training)', 't-tick', 'middle')]
    return svg(760, y + 50, 'BERT input representation for a real sentence pair. Each token adds three learned vectors: its token embedding, a segment embedding (A or B) and a position embedding (0 to 9). The sum is normalised and sent to layer 1.', b)


F['p2_embed_sum'] = embed_sum()


def masks():
    toks = ['[CLS]', 'my', 'dog', 'is', 'cute', '[SEP]']
    n, cell = len(toks), 40
    b = []
    for k, (title, x0, causal) in enumerate([('BERT: every token sees every token', 70, False), ('GPT: each token sees only its left', 440, True)]):
        b.append(text(x0 + 55 + n * cell / 2, 26, title, 't-title', 'middle'))
        for i in range(n):
            b.append(text(x0 + 50, 60 + i * cell + cell / 2 + 4, toks[i], 't-tick', 'end'))
            for j in range(n):
                ok = (j <= i) or not causal
                b.append(box(x0 + 56 + j * cell, 60 + i * cell, cell - 4, cell - 4, 'box-1' if ok else 'box-ghost', 4))
        b.append(text(x0 + 56 + n * cell / 2, 60 + n * cell + 22, 'columns: the token being looked at', 't-muted', 'middle'))
    b.append(text(20, 60 + n * cell + 48, 'Rows: the token doing the looking. Filled square: allowed. Dashed square: blocked by the mask.', 't-tick'))
    return svg(760, 60 + n * cell + 62, 'Attention masks. In BERT every row is full: each token may attend to all tokens. In GPT the upper triangle is blocked, so each token attends only to itself and tokens on its left.', b)


F['p2_masks'] = masks()

json.dump(F, open('results/figs_part2.json', 'w'))
print('figures:', ', '.join(F))
