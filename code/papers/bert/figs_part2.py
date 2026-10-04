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


# ================================================================ v2 figures (illustrated, Alammar style)
from alammar import vec, vec_len, matrix, attn_grid, frame, brace

M = json.load(open('results/part2_math.json'))


def signed(x, y, vals, cell=30, gap=4, show=True, fmt='{:.3f}', tcls='t-cell'):
    """A strip of numbers: blue for positive, orange for negative, stronger colour = larger size; value written below."""
    m = max(1e-9, max(abs(v) for v in vals))
    b = []
    for i, v in enumerate(vals):
        xx = x + i * (cell + gap)
        b.append(f'<g class="mark"><title>{v:.4f}</title><rect class="{"s1" if v >= 0 else "s2"}" x="{xx:.1f}" y="{y:.1f}" width="{cell}" height="{cell}" rx="3" style="fill-opacity:{0.15 + 0.8 * abs(v) / m:.2f}"/></g>')
        if show:
            b.append(text(xx + cell / 2, y + cell + 14, fmt.format(v), 't-tick', 'middle'))
    return b


def related_map():
    """Section 2 as three lanes on a time line, all flowing into BERT."""
    x0, x1, y0 = 225, 640, 70
    X = lambda yr: x0 + (yr - 2008) / (2018.5 - 2008) * (x1 - x0)
    lanes = [('§2.1 feature-based', 'box-1', [(2010, 'Turian'), (2013, 'word2vec'), (2014, 'GloVe'), (2015, 'skip-thought'), (2018.1, 'ELMo')]),
             ('§2.2 fine-tuning', 'box-3', [(2008, 'Collobert & Weston'), (2015.25, 'Dai & Le'), (2018, 'ULMFiT'), (2018.45, 'OpenAI GPT')]),
             ('§2.3 labelled transfer', 'box-2', [(2009, 'ImageNet'), (2014, 'Yosinski'), (2017.05, 'InferSent'), (2017.6, 'CoVe')])]
    b = [text(20, 26, 'Related work (Section 2): three families of pre-training, all feeding into BERT', 't-title')]
    for yr in range(2008, 2019, 2):
        b += [line(X(yr), y0 - 14, X(yr), y0 + 3 * 74 - 18, 'grid'), text(X(yr), y0 - 20, str(yr), 't-tick', 'middle')]
    for k, (name, cls, pts) in enumerate(lanes):
        y = y0 + k * 74 + 14
        b += [text(20, y + 5, name, 't-note'), line(x0 - 6, y, x1, y, 'edge-dim')]
        for i, (yr, lab) in enumerate(pts):
            up = (i % 2 == 0)
            b += [f'<circle class="node on" cx="{X(yr):.1f}" cy="{y}" r="6"/>', text(X(yr), y + (-12 if up else 22), lab, 't-tick', 'middle')]
        b.append(arrow(x1 + 2, y, 668, 124 + 0 * k + (k - 1) * 22))
    b += [box(670, 96, 80, 56, 'box-on', 12), text(710, 121, 'BERT', 't-big', 'middle'), text(710, 139, '2018', 't-tick', 'middle')]
    b += [text(20, 300, 'Feature-based: frozen vectors feed a separate task model. Fine-tuning: the whole pre-trained network keeps training.', 't-muted'),
          text(20, 318, 'Labelled transfer: pre-train on a big labelled task. BERT takes the fine-tuning recipe and the ImageNet habit, with unlabelled text.', 't-muted')]
    return svg(760, 330, 'Section 2 of the BERT paper as a map. Three lanes over the years 2008 to 2018: feature-based approaches (word embeddings, skip-thought, ELMo), fine-tuning approaches (Collobert and Weston, Dai and Le, ULMFiT, OpenAI GPT) and transfer from labelled data (ImageNet, InferSent, CoVe). All three lead to BERT.', b)


F['p2_related_map'] = related_map()


def word_reprs():
    """Non-neural vs neural word representations, and static vs contextual (an illustration, not real numbers)."""
    b = [text(20, 26, 'Three ways to turn the word "bank" into numbers', 't-title')]
    # 1. Brown cluster: a path in a binary tree
    b += [box(20, 44, 230, 236, 'box', 12), text(135, 68, '1. Brown clusters (1992)', 't-note', 'middle'), text(135, 86, 'non-neural: count word neighbours', 't-muted', 'middle')]
    nodes = {(0, 0): (135, 110)}
    for d in range(1, 4):
        for k in range(2 ** d):
            nodes[(d, k)] = (35 + (k + 0.5) * 200 / 2 ** d, 110 + d * 38)
    path = [0, 1, 2, 5]          # 0 -> 1 -> 10 -> 101
    for (d, k), (x, y) in nodes.items():
        if d:
            px, py = nodes[(d - 1, k // 2)]
            on = path[d] == k and path[d - 1] == k // 2
            b.append(line(px, py, x, y, 'edge-on' if on else 'edge'))
            b.append(text((px + x) / 2 + (-8 if k % 2 == 0 else 8), (py + y) / 2, str(k % 2), 't-muted', 'middle'))
    for (d, k), (x, y) in nodes.items():
        b.append(f'<circle class="node{" on" if path[d] == k else ""}" cx="{x:.1f}" cy="{y:.1f}" r="7"/>')
    b += [text(135, 262, '"bank" = cluster 101', 't-mono', 'middle'), text(135, 276, '(with "shore", "branch", ...)', 't-muted', 'middle')]
    # 2. static dense vector
    b += [box(266, 44, 230, 236, 'box', 12), text(381, 68, '2. word2vec, GloVe (2013, 2014)', 't-note', 'middle'), text(381, 86, 'neural: one learned vector per word', 't-muted', 'middle')]
    v = [0.4, -0.8, 0.2, 0.9, -0.3, 0.6, 0.1, -0.5]
    for k, s_ in enumerate(['river bank', 'savings bank']):
        y = 122 + k * 70
        b += [text(381, y, f'"{s_}"', 't-tick', 'middle')] + vec(381 - vec_len(8, 16, 3) / 2, y + 10, v, 's1', 16, 3)
    b += [text(381, 262, 'the same row every time', 't-tick', 'middle'), text(381, 276, '(a static embedding)', 't-muted', 'middle')]
    # 3. contextual vector
    b += [box(512, 44, 230, 236, 'box', 12), text(627, 68, '3. ELMo, BERT (2018)', 't-note', 'middle'), text(627, 86, 'contextual: computed per sentence', 't-muted', 'middle')]
    v1 = [0.7, -0.2, 0.5, 0.9, -0.6, 0.1, 0.3, -0.4]
    v2 = [-0.3, 0.8, -0.6, 0.2, 0.5, -0.9, 0.4, 0.6]
    for k, (s_, vv, c) in enumerate([('river bank', v1, 's3'), ('savings bank', v2, 's2')]):
        y = 122 + k * 70
        b += [text(627, y, f'"{s_}"', 't-tick', 'middle')] + vec(627 - vec_len(8, 16, 3) / 2, y + 10, vv, c, 16, 3)
    b += [text(627, 262, 'a different vector per sentence', 't-tick', 'middle'), text(627, 276, '(real BERT: river vs money cosine 0.43 to 0.48)', 't-muted', 'middle')]
    return svg(760, 292, 'Three ways to represent a word. Brown clusters, a non-neural method, give each word a path in a binary tree of word groups. word2vec and GloVe give each word one learned vector, the same in every sentence. ELMo and BERT compute a new vector for each sentence, so river bank and savings bank differ. The colours are an illustration, not real values.', b)


F['p2_word_reprs'] = word_reprs()


def window():
    """word2vec: the left and right context window, and negative sampling."""
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    cls = ['box-1', 'box-1', 'box-on', 'box-1', 'box-1', 'box']
    b = [text(20, 26, 'word2vec (skip-gram): learn a vector by predicting the words around it', 't-title')]
    parts, cx = row(150, 50, words, 70, 10, classes=cls)
    b += parts + [text(140, 70, 'sentence', 't-muted', 'end')]
    b += brace(cx[0] - 33, cx[1] + 33, 92, 'left context (2 words)', 'edge-1', tcls='t-s1') + brace(cx[3] - 33, cx[4] + 33, 92, 'right context (2 words)', 'edge-1', tcls='t-s1')
    b += [text(cx[2], 112, 'centre word', 't-on', 'middle'), text(cx[5], 112, 'outside window', 't-muted', 'middle')]
    # negative sampling
    y = 150
    b += [text(20, y, 'Training signal: tell real neighbours from random words ("discriminate correct from incorrect words")', 't-note')]
    pairs = [('sat', 'cat', 'real neighbour', '1', 'box-3'), ('sat', 'on', 'real neighbour', '1', 'box-3'),
             ('sat', 'banana', 'random word', '0', 'box-2'), ('sat', 'quickly', 'random word', '0', 'box-2')]
    for k, (a, c, lab, t, bc) in enumerate(pairs):
        yy = y + 18 + k * 38
        b += token(150, yy, 70, a, 'box-on') + [text(232, yy + 20, '+', 't-muted', 'middle')] + token(244, yy, 84, c, bc)
        b += [arrow(334, yy + 15, 380, yy + 15), box(384, yy, 170, 30, 'box', 6), text(469, yy + 20, f'σ( v′{c} · v{a} )', 't-tick', 'middle'),
              arrow(558, yy + 15, 590, yy + 15), text(600, yy + 20, f'target {t}: {lab}', 't-tick')]
    return svg(760, y + 18 + 4 * 38 + 6, 'word2vec skip-gram. The centre word sat is trained to predict the two words on its left and the two on its right. With negative sampling, the model learns to give a high score to real neighbour pairs (sat with cat, sat with on) and a low score to random pairs (sat with banana).', b)


F['p2_window'] = window()


def sentence_objectives():
    """Two ways to learn a sentence vector: generate the neighbours (skip-thought) or pick the true next sentence."""
    b = [text(20, 26, 'Learning sentence vectors from neighbouring sentences', 't-title')]
    # skip-thought
    b += [text(20, 56, 'Skip-thought (Kiros et al., 2015): encode a sentence, then write its neighbours', 't-note')]
    b += token(20, 96, 170, 'I could see the cat', 'box-1', 32) + [arrow(194, 112, 226, 112)]
    b += [box(230, 92, 92, 40, 'box-1', 10), text(276, 117, 'encoder', 't-note', 'middle'), arrow(326, 112, 352, 112)]
    b += vec(358, 106, [0.5, 0.9, 0.3, 0.7, 0.4], 's1', 12, 2) + [text(392, 96, 'sentence vector', 't-muted', 'middle')]
    b += [arrow(432, 106, 476, 84), arrow(432, 118, 476, 140)]
    b += [box(480, 64, 92, 36, 'box-2', 10), text(526, 87, 'decoder', 't-note', 'middle'), arrow(576, 82, 596, 82), text(602, 87, '"I got back home"', 't-tick'),
          box(480, 122, 92, 36, 'box-3', 10), text(526, 145, 'decoder', 't-note', 'middle'), arrow(576, 140, 596, 140), text(602, 145, '"This was strange"', 't-tick')]
    b += [text(602, 70, 'previous sentence', 't-muted'), text(602, 128, 'next sentence', 't-muted')]
    # ranking
    y = 196
    b += [text(20, y, 'Ranking (Jernite et al., 2017; Logeswaran and Lee, 2018): pick the true next sentence', 't-note')]
    b += token(20, y + 24, 170, 'Spring had come.', 'box-1', 32) + [text(105, y + 72, 'the sentence', 't-muted', 'middle')]
    cands = [('They were so black.', 0.08), ('And yet his crops didn\'t grow.', 0.81), ('He had blue eyes.', 0.11)]
    for k, (c, p) in enumerate(cands):
        yy = y + 24 + k * 40
        b += [arrow(194, y + 40, 232, yy + 15)] + token(236, yy, 230, c, 'box-3' if p > 0.5 else 'box', 30)
        b += [arrow(470, yy + 15, 496, yy + 15), f'<rect class="{"s3" if p > 0.5 else "s1"}" x="500" y="{yy + 7}" width="{p * 140:.1f}" height="16" rx="3" style="fill-opacity:0.8"/>',
              text(506 + p * 140, yy + 20, f'{p:.2f}', 't-val')]
    b += [text(500, y + 24 + 3 * 40 + 12, 'classifier scores (illustration)', 't-muted')]
    return svg(760, y + 24 + 3 * 40 + 24, 'Two older ways to learn sentence vectors. Skip-thought encodes a sentence into one vector and trains two decoders to write the previous and the next sentence from it. Ranking methods replace the decoder with a classifier that picks the true next sentence from a few candidates. The scores shown are an illustration.', b)


F['p2_sentence_objectives'] = sentence_objectives()


def denoise():
    """Denoising auto-encoder (Hill et al., 2016) vs BERT's masked LM."""
    words = ['the', 'dog', 'chased', 'a', 'red', 'ball']
    b = [text(20, 26, 'Denoising: damage the input, learn to repair it', 't-title')]
    b += frame(20, 40, 720, 62, 1, 'clean sentence S') + row(300, 56, words, 66, 6, h=28)[0]
    corrupt = ['the', 'chased', 'dog', 'red', 'ball']
    b += frame(20, 110, 720, 92, 2, 'noise: delete "a" (pₒ), swap "dog chased" (pₓ)')
    b += row(300, 158, corrupt, 66, 6, classes=['box', 'box-2', 'box-2', 'box', 'box'], h=28)[0]
    b += [text(290, 177, 'damaged input', 't-muted', 'end')]
    b += frame(20, 210, 720, 104, 3, 'SDAE: an LSTM encoder-decoder rebuilds ALL of S')
    b += [box(50, 252, 110, 34, 'box-1', 8), text(105, 274, 'encoder', 't-note', 'middle'), arrow(164, 269, 186, 269),
          box(190, 252, 110, 34, 'box-1', 8), text(245, 274, 'decoder', 't-note', 'middle'), arrow(304, 269, 326, 269)]
    b += row(330, 256, words, 62, 6, cls='box-3', h=26)[0] + [text(534, 302, 'every word of S is a target', 't-muted', 'middle')]
    b += frame(20, 322, 720, 92, 4, 'BERT masked LM: predict ONLY the hidden word')
    mw = ['the', 'dog', 'chased', 'a', '[MASK]', 'ball']
    parts, cx = row(50, 366, mw, 60, 6, classes=['box', 'box', 'box', 'box', 'box-mask', 'box'], h=28)
    b += parts + [arrow(450, 380, 476, 380), box(480, 364, 90, 32, 'box-1', 8), text(525, 385, 'BERT', 't-note', 'middle'),
                  arrow(574, 380, 596, 380)] + token(600, 366, 56, 'red', 'box-3', 28) + [text(664, 385, 'one target', 't-muted')]
    return svg(760, 424, 'Four frames. 1: a clean sentence. 2: the noise function of Hill et al. deletes a word and swaps two neighbouring words. 3: their sequential denoising auto-encoder rebuilds the whole clean sentence. 4: BERT also repairs damaged input, but only the hidden word is a target.', b)


F['p2_denoise'] = denoise()


def elmo_concat():
    """ELMo's token vector = forward LSTM state glued to backward LSTM state."""
    words = ['he', 'sat', 'on', 'the', 'bank']
    b = [text(20, 26, 'ELMo: two one-way readers, glued together at the end', 't-title')]
    parts, cx = row(150, 220, words, 76, 12)
    b += parts
    yF, yB = 150, 80
    for i, x in enumerate(cx):
        b += [box(x - 30, yF, 60, 30, 'box-1', 6), text(x, yF + 20, 'LSTM →', 't-tick', 'middle'),
              box(x - 30, yB, 60, 30, 'box-2', 6), text(x, yB + 20, '← LSTM', 't-tick', 'middle'), arrow(x, 218, x, yF + 32)]
        if i < len(cx) - 1:
            b.append(arrow(x + 30, yF + 15, cx[i + 1] - 32, yF + 15))
            b.append(arrow(cx[i + 1] - 30, yB + 15, x + 32, yB + 15))
    b += [line(cx[-1], yF, cx[-1], yB + 32, 'edge-dim'), text(140, yF + 20, 'left to right', 't-s1', 'end'), text(140, yB + 20, 'right to left', 't-s2', 'end')]
    x = cx[-1] + 50
    b += [text(x, 60, 'vector for "bank"', 't-note')] + vec(x, 76, [0.5, 0.8, 0.3, 0.9], 's2', 14, 2) + vec(x + 66, 76, [0.7, 0.2, 0.6, 0.4], 's1', 14, 2)
    b += [text(x, 112, '← half  +  → half', 't-muted'), text(x, 200, 'Concatenated, not mixed:', 't-muted'), text(x, 216, 'inside each reader', 't-muted'), text(x, 232, 'information flows one way.', 't-muted')]
    b += [line(cx[-1] + 30, yB + 15, x - 4, 84, 'edge-2'), line(cx[-1] + 30, yF + 15, x + 96, 92, 'edge-1')]
    return svg(760, 264, 'ELMo runs a left-to-right LSTM and a right-to-left LSTM over the sentence. The vector for bank is the forward state glued to the backward state. Inside each reader, information flows only one way; the two directions meet only in this final concatenation.', b)


F['p2_elmo_concat'] = elmo_concat()


def transfer():
    """The ImageNet recipe, and BERT's version of it."""
    b = [text(20, 26, 'The transfer-learning recipe, in vision and in language', 't-title')]
    rows_ = [('vision', 'millions of labelled photos', 'ImageNet classes', 'CNN', 'box-2', ['spot tumours', 'find cars', 'sort plants']),
             ('language (BERT)', '3.3B words, no labels', 'masked LM + NSP', 'BERT', 'box-1', ['sentiment', 'answer questions', 'find names'])]
    for k, (name, data, task, model_, cls, tasks) in enumerate(rows_):
        y = 52 + k * 120
        b += [text(20, y + 8, name, 't-note')]
        b += [box(20, y + 22, 150, 64, 'box', 10), text(95, y + 48, data, 't-tick', 'middle'), text(95, y + 66, task, 't-muted', 'middle'),
              arrow(174, y + 54, 214, y + 54), box(218, y + 30, 110, 48, cls, 12), text(273, y + 52, model_, 't-big', 'middle'),
              text(273, y + 70, 'pre-trained', 't-muted', 'middle')]
        for j, t in enumerate(tasks):
            yy = y + 14 + j * 30
            b += [arrow(330, y + 54, 398, yy + 12), box(402, yy, 88, 24, cls, 6), text(446, yy + 16, 'copy', 't-tick', 'middle'),
                  arrow(492, yy + 12, 512, yy + 12), text(518, yy + 16, f'fine-tune: {t}', 't-tick')]
    b += [text(20, 300, 'Step 1 is done once on a huge dataset; step 2 starts every new task from those weights instead of from random numbers.', 't-muted')]
    return svg(760, 312, 'Transfer learning. In vision, a network is pre-trained once on ImageNet labels and then fine-tuned for many other image tasks. BERT does the same for language, but its pre-training data is unlabelled text.', b)


F['p2_transfer'] = transfer()


def figure1_redraw():
    """Figure 1, redrawn: the same encoder in pre-training and in fine-tuning, with different output layers."""
    b = []
    for side, (x0, title) in enumerate([(10, 'Pre-training'), (390, 'Fine-tuning (SQuAD)')]):
        b.append(text(x0 + 180, 24, title, 't-title', 'middle'))
        if side == 0:
            toks = ['[CLS]', 'my', '[MASK]', '[SEP]', 'he', '[MASK]', '[SEP]']
            cls = ['box-on', 'box-1', 'box-mask', 'box', 'box-2', 'box-mask', 'box']
            note = 'sentence A, sentence B (masked)'
        else:
            toks = ['[CLS]', 'who', 'won', '[SEP]', 'Ada', 'won', '[SEP]']
            cls = ['box-on', 'box-1', 'box-1', 'box', 'box-2', 'box-2', 'box']
            note = 'question, paragraph'
        W, G = 46, 6
        parts, cx = row(x0 + 2, 300, toks, W, G, classes=cls, h=28, tcls='t-muted')
        b += parts + [text(x0 + 180, 346, note, 't-muted', 'middle')]
        outs = ['C', 'T₁', 'T₂', 'T[SEP]', "T₁′", "T₂′", 'T[SEP]']
        for x, o in zip(cx, toks):
            b += [arrow(x, 298, x, 280)]
        b += [box(x0 + 2, 250, 360, 28, 'box-4', 6), text(x0 + 182, 269, 'E: input embeddings', 't-tick', 'middle')]
        b += [box(x0 + 2, 166, 360, 72, 'box-1', 12), text(x0 + 182, 198, 'BERT', 't-big', 'middle'),
              text(x0 + 182, 218, 'same encoder, same starting weights', 't-muted', 'middle'), arrow(x0 + 182, 248, x0 + 182, 240)]
        parts, cx2 = row(x0 + 2, 112, ['C', 'T₁', 'T₂', 'T', 'T₁′', 'T₂′', 'T'], W, G, classes=['box-on'] + ['box-3'] * 6, h=28, tcls='t-tick')
        b += parts
        for x in cx2:
            b.append(arrow(x, 164, x, 142))
        if side == 0:
            b += [line(cx2[0], 110, cx2[0], 74, 'edge-on'), box(cx2[0] - 23, 44, 46, 28, 'box-on', 6), text(cx2[0], 63, 'NSP', 't-tick', 'middle'),
                  text(cx2[0], 38, 'IsNext?', 't-muted', 'middle')]
            for j in (2, 5):
                b += [line(cx2[j], 110, cx2[j], 74, 'edge-1'), box(cx2[j] - 30, 44, 60, 28, 'box-mask', 6), text(cx2[j], 63, 'Mask LM', 't-tick', 'middle')]
            b += [text(cx2[2], 38, '"dog"', 't-muted', 'middle'), text(cx2[5], 38, '"likes"', 't-muted', 'middle')]
        else:
            b += [line(cx2[4], 110, cx2[4], 74, 'edge-1'), box(cx2[4] - 26, 44, 52, 28, 'box-3', 6), text(cx2[4], 63, 'start', 't-tick', 'middle'),
                  line(cx2[4], 110, cx2[5], 74, 'edge-1'), box(cx2[5] - 2, 44, 52, 28, 'box-3', 6), text(cx2[5] + 24, 63, 'end', 't-tick', 'middle'),
                  text(cx2[4] + 12, 38, 'answer span: "Ada"', 't-muted', 'middle'), text(cx2[0], 98, 'C unused', 't-muted', 'middle')]
    b += [line(378, 40, 378, 340, 'edge-dim')]
    return svg(760, 360, 'Figure 1 redrawn. Left, pre-training: a masked sentence pair goes in; C feeds next sentence prediction and the T vectors at masked positions feed the masked language model. Right, fine-tuning for SQuAD: the same encoder with the same starting weights reads a question and a paragraph, and only the output layer changes: start and end scores over the paragraph tokens.', b)


F['p2_figure1'] = figure1_redraw()


def attn_shapes():
    """Shapes of one attention head in BERT-base, for n tokens."""
    b = [text(20, 26, 'One attention head of BERT-base, shape by shape (n tokens)', 't-title')]
    # X Wq = Q
    X, xw, xh = matrix(40, 70, 5, 12, 's4', 8, 2, label='X', shape='n × 768')
    b += X + [text(40 + xw + 14, 98, '×', 't-big', 'middle')]
    Wq, ww, wh = matrix(40 + xw + 30, 60, 12, 4, 's1', 8, 2, label='Wq', shape='768 × 64')
    b += Wq + [text(40 + xw + 30 + ww + 16, 98, '=', 't-big', 'middle')]
    Qm, qw, qh = matrix(40 + xw + 30 + ww + 34, 70, 5, 4, 's1', 8, 2, label='Q', shape='n × 64')
    b += Qm
    b += [text(380, 76, 'the same for K = X Wk and V = X Wv', 't-tick'), text(380, 96, '(each W is 768 × 64: one slice of', 't-muted'),
          text(380, 112, 'the 768 × 768 query, key, value matrices)', 't-muted')]
    # Q K^T
    y = 200
    Qm, qw, qh = matrix(40, y, 5, 4, 's1', 8, 2, label='Q', shape='n × 64')
    b += Qm + [text(40 + qw + 14, y + 26, '×', 't-big', 'middle')]
    Kt, kw, kh = matrix(40 + qw + 28, y + 8, 4, 5, 's3', 8, 2, label='Kᵀ', shape='64 × n')
    b += Kt + [text(40 + qw + 28 + kw + 16, y + 26, '=', 't-big', 'middle')]
    S_, sw, sh = matrix(40 + qw + 28 + kw + 32, y, 5, 5, 's4', 8, 2, label='scores', shape='n × n', label_cls='t-tick')
    b += S_
    x2 = 40 + qw + 28 + kw + 32 + sw + 14
    b += [arrow(x2, y + 22, x2 + 26, y + 22), text(x2 + 14, y + 10, '÷ 8', 't-tick', 'middle'), text(x2 + 14, y + 44, 'softmax', 't-muted', 'middle')]
    A_, aw, ah = matrix(x2 + 34, y, 5, 5, 'cell', 8, 2, label='weights', shape='n × n, rows sum to 1', label_cls='t-tick')
    b += A_ + [text(x2 + 34 + aw + 14, y + 26, '×', 't-big', 'middle')]
    V_, vw, vh = matrix(x2 + 34 + aw + 28, y, 5, 4, 's2', 8, 2, label='V', shape='n × 64')
    b += V_ + [text(x2 + 34 + aw + 28 + vw + 16, y + 26, '=', 't-big', 'middle')]
    Z_, zw, zh = matrix(x2 + 34 + aw + 28 + vw + 32, y, 5, 4, 's3', 8, 2, label='Z', shape='n × 64')
    b += Z_
    b += [text(20, 320, 'Each row of the weights says how much one token looks at every token. Z, the head\'s output, mixes the value rows with those weights.', 't-muted')]
    return svg(760, 334, 'The shapes inside one attention head. The input X (n by 768) times Wq (768 by 64) gives Q (n by 64); K and V are made the same way. Q times K transposed gives an n by n grid of scores, divided by 8 and passed through softmax so each row sums to 1. The weights times V give the head output Z, n by 64.', b)


F['p2_attn_shapes'] = attn_shapes()


def attn_tiny():
    t = M['tiny']
    w = t['words']
    b = [text(20, 26, 'The tiny example: 3 tokens, dₖ = 4', 't-title')]
    g, gw, gh = attn_grid(20, 96, w, w, lambda i, j: True, 44, title='scores Q Kᵀ / 2', values=[[v / 1.5 for v in r] for r in t['scaled']], cls_fn=lambda i, j: 's4')
    b += g
    for i in range(3):
        for j in range(3):
            b.append(text(20 + 64 + j * 44 + 22, 96 + i * 44 + 26, f'{t["scaled"][i][j]:.1f}', 't-cell', 'middle'))
    b += [arrow(240, 162, 284, 162), text(262, 152, 'softmax', 't-muted', 'middle'), text(262, 182, 'per row', 't-muted', 'middle')]
    g, gw2, gh = attn_grid(290, 96, w, w, lambda i, j: True, 44, title='weights (rows sum to 1)', values=t['weights'], show_values=True)
    b += g
    b += [arrow(510, 162, 548, 162), text(529, 152, '× V', 't-muted', 'middle')]
    b += [text(640, 70, 'output (2 numbers each)', 't-tick', 'middle')]
    for i, r in enumerate(t['out']):
        y = 96 + i * 44
        b += [text(570, y + 26, w[i], 't-tick', 'end')] + signed(580, y + 6, r, 32, 4, show=False)
        b += [text(580 + 16, y + 26, f'{r[0]:.2f}', 't-cell', 'middle'), text(580 + 52, y + 26, f'{r[1]:.2f}', 't-cell', 'middle')]
    return svg(760, 260, 'The tiny attention example computed in code. Left: the scaled scores. Middle: after softmax each row sums to 1; "kid" puts 0.547 on itself. Right: each output row is a weighted average of the value rows.', b)


F['p2_attn_tiny'] = attn_tiny()


def scale_fig():
    s_ = M['scale']
    b = [text(20, 26, 'Why divide by √64 = 8: the same 8 scores, softmax with and without the division', 't-title')]
    for k, (name, p, cls) in enumerate([('without ÷ 8 (spread 8)', s_['p_raw'], 's2'), ('with ÷ 8 (spread 1)', s_['p_scaled'], 's1')]):
        x0 = 30 + k * 370
        b.append(text(x0, 56, name, 't-note'))
        base = 200
        for i, v in enumerate(p):
            h = v * 260
            b += [f'<g class="mark"><title>key {i + 1}: {v:.3f}</title><rect class="{cls}" x="{x0 + i * 40:.1f}" y="{base - h:.1f}" width="30" height="{max(1, h):.1f}" rx="3"/></g>',
                  text(x0 + i * 40 + 15, base - h - 6, f'{v:.2f}', 't-tick', 'middle'), text(x0 + i * 40 + 15, base + 16, str(i + 1), 't-muted', 'middle')]
        b.append(line(x0 - 4, base, x0 + 8 * 40 - 6, base, 'axis'))
    b += [text(30, 238, 'Unscaled, one key takes 0.53 and three get almost 0: softmax is close to "pick one", and its gradients are tiny.', 't-muted'),
          text(30, 256, 'Scaled, the weights stay spread out (largest 0.21), so the model can still learn which keys matter.', 't-muted')]
    return svg(760, 268, 'Softmax of the same eight scores. Without dividing by 8, the spread of dot products of 64-number vectors is about 8, and softmax puts 0.53 on one key and almost nothing on several. With the division the largest weight is 0.21.', b)


F['p2_scale'] = scale_fig()


def attn_real():
    r = M['real_attention']
    toks = r['tokens']
    b = []
    g, gw, gh = attn_grid(150, 76, toks, toks, lambda i, j: True, 52, title='real bert-base-uncased, layer 1, head 1', values=r['weights'], show_values=True)
    b += g
    b += [text(560, 100, 'Every cell is allowed:', 't-note'), text(560, 120, 'no mask, both directions.', 't-tick'),
          text(560, 150, 'Rows sum to 1.', 't-note'), text(560, 170, '"kid" looks most at', 't-tick'), text(560, 188, '[SEP] (0.340) and', 't-tick'),
          text(560, 206, '"smiles" (0.281).', 't-tick'), text(560, 236, 'Weights near 0.2 =', 't-tick'), text(560, 254, 'nearly uniform.', 't-tick')]
    return svg(760, 360, 'Attention weights of head 1 in layer 1 of the real bert-base-uncased, for the input [CLS] the kid smiles [SEP]. Every token may look at every token. The row for kid puts 0.340 on [SEP] and 0.281 on smiles.', b)


F['p2_attn_real'] = attn_real()


def multihead():
    b = [text(20, 26, 'Multi-head attention: 12 heads of 64 numbers, joined back into 768', 't-title')]
    cls = ['s1', 's2', 's3', 's4']
    for h in range(12):
        x = 20 + h * 46
        b += [box(x, 46, 40, 30, 'box', 6), text(x + 20, 66, f'h{h + 1}', 't-tick', 'middle')]
        b += vec(x + 13, 90, 5, cls[h % 4], 12, 2, vertical=True, outline=False)
    b += [text(580, 64, 'each head: its own', 't-tick'), text(580, 80, 'Wq, Wk, Wv (768 × 64)', 't-tick'), text(580, 112, 'output: n × 64', 't-tick')]
    b += [line(20, 176, 568, 176, 'edge'), arrow(294, 176, 294, 196)]
    y = 202
    for h in range(12):
        b += vec(20 + h * 46, y, [0.6, 0.8, 0.5], cls[h % 4], 12, 2, outline=False)
    b += [text(580, y + 10, 'Concat: n × 768', 't-tick'), text(580, y + 26, '(12 × 64 = 768)', 't-muted'), arrow(294, y + 18, 294, y + 40)]
    b += [box(194, y + 44, 200, 30, 'box-1', 8), text(294, y + 64, '× Wo (768 × 768)', 't-note', 'middle'), arrow(294, y + 76, 294, y + 96),
          box(144, y + 100, 300, 26, 'box-3', 6), text(294, y + 118, 'MultiHead output: n × 768', 't-tick', 'middle')]
    return svg(760, 340, 'Multi-head attention in BERT-base. Twelve heads run side by side, each with its own query, key and value matrices of 768 by 64, and each returns 64 numbers per token. Their outputs are concatenated into 768 numbers per token and multiplied by Wo, a 768 by 768 matrix.', b)


F['p2_multihead'] = multihead()


def ffn():
    f = M['ffn']
    b = [text(20, 26, 'The feed-forward network: wider, GELU, narrower (one token at a time)', 't-title')]
    b += [text(20, 64, '"kid" after attention', 't-tick')] + vec(20, 74, 24, 's1', 9, 1)
    b += [text(20 + vec_len(24, 9, 1) / 2, 102, '768', 't-val', 'middle'), arrow(260, 80, 290, 80)]
    b += [box(294, 62, 80, 36, 'box', 8), text(334, 85, '× W₁ + b₁', 't-tick', 'middle'), arrow(378, 80, 400, 80)]
    b += [text(404, 64, 'wide: 3,072 numbers (4H)', 't-tick')]
    vals = [abs(v) for v in f['first6_wide']] * 9
    b += vec(404, 74, [v if i % 7 else -v for i, v in enumerate(vals[:44])], 's4', 6, 1)
    b += [arrow(560, 102, 560, 130), text(570, 120, 'GELU', 't-note')]
    b += [text(404, 152, f'after GELU: only {100 * f["share_positive"]:.1f}% were positive before it', 't-tick')]
    b += vec(404, 162, [0.9 if i % 19 == 0 else 0.05 for i in range(44)], 's4', 6, 1)
    b += [arrow(400, 168, 378, 168), box(294, 150, 80, 36, 'box', 8), text(334, 173, '× W₂ + b₂', 't-tick', 'middle'), arrow(290, 168, 262, 168)]
    b += vec(20, 162, 24, 's3', 9, 1) + [text(20 + vec_len(24, 9, 1) / 2, 190, 'back to 768', 't-val', 'middle')]
    b += [text(20, 226, f'W₁: {f["W1"][0]} × {f["W1"][1]}    W₂: {f["W2"][0]} × {f["W2"][1]}    the same weights for every position in a layer', 't-muted'),
          text(20, 244, f'Real numbers (layer 1, "kid"): first 6 of the 3,072 before GELU {", ".join(f"{v:.2f}" for v in f["first6_wide"])}', 't-muted')]
    return svg(760, 256, 'The feed-forward network of one BERT-base layer, applied to each token on its own. The 768 numbers are multiplied by W1 into 3,072, passed through GELU, and multiplied by W2 back to 768. In layer 1, for the token kid, only 5.2 percent of the 3,072 numbers were positive before GELU.', b)


F['p2_ffn'] = ffn()


def embed_numbers():
    e = M['embed']
    b = [text(20, 26, 'The input embedding of "dog", number by number (first 6 of 768)', 't-title')]
    rows_ = [('Tok[3899]  "dog"', e['tok'], 'token table, row 3899'), ('Seg[A]', e['seg'], 'segment table, row 0'),
             (f'Pos[{e["position"]}]', e['pos'], 'position table, row 2'), ('sum', e['sum'], 'add the three'),
             ('LayerNorm(sum)', e['ln'], f'subtract mean {e["mean"]:.4f}, divide by {e["std"]:.4f}, scale, shift')]
    y = 46
    for k, (name, vals, note) in enumerate(rows_):
        if k in (1, 2):
            b.append(text(170, y + 18, '+', 't-big', 'middle'))
        if k == 3:
            b += [line(185, y - 6, 395, y - 6, 'axis'), text(170, y + 18, '=', 't-big', 'middle')]
        if k == 4:
            b += [arrow(290, y - 18, 290, y + 2)]
        b += [text(160, y + 20, name, 't-note' if k >= 3 else 't-tick', 'end')] + signed(185, y + 2, vals, 30, 6, fmt='{:.3f}')
        b += [text(410, y + 20, note, 't-muted'), text(410 if False else 405, y + 36, '', 't-muted')]
        y += 56 if k != 3 else 62
    b += [text(20, y + 8, 'Blue: positive. Orange: negative. Darker: larger in size (each row has its own scale).', 't-muted'),
          text(20, y + 26, f'Lengths of the full 768-number vectors: Tok {e["norms"]["tok"]}, Seg {e["norms"]["seg"]}, Pos {e["norms"]["pos"]}, sum {e["norms"]["sum"]}, after LayerNorm {e["norms"]["ln"]}.', 't-muted')]
    return svg(760, y + 40, 'The input embedding of the token dog in the Figure 2 pair, using the real tables of bert-base-uncased. The first six numbers of the token row, the segment A row and the position 2 row are added, then LayerNorm subtracts the mean, divides by the spread and applies a learned scale and shift.', b)


F['p2_embed_numbers'] = embed_numbers()


def wordpiece_frames():
    t = M['wp_trace']
    trace = t['trace']
    # group the lookups by the piece they end in
    groups, cur = [], []
    for p, h in trace:
        cur.append((p, h))
        if h:
            groups.append(cur)
            cur = []
    b = [text(20, 26, f'Greedy longest-match-first on "{t["word"]}": {len(trace)} lookups, {len(groups)} pieces', 't-title')]
    y = 44
    done = []
    for k, g in enumerate(groups):
        tries = [p for p, h in g]
        b += frame(20, y, 720, 62, k + 1, f'try {len(tries)} piece{"s" if len(tries) > 1 else ""}, longest first')
        shown = tries if len(tries) <= 4 else tries[:2] + ['…'] + tries[-1:]
        x = 300
        for p in shown:
            hit = p == tries[-1]
            w = 14 + 7.6 * len(p)
            b += token(x, y + 12, w, p, 'box-3' if hit else ('box-ghost' if p == '…' else 'box'), 28, 't-mono' if hit else 't-muted')
            if not hit and p != '…':
                b.append(line(x + 4, y + 26, x + w - 4, y + 26, 'bad-line'))
            x += w + 8
        done.append(tries[-1])
        b += [text(36, y + 50, 'kept so far: ' + ' '.join(done), 't-tick')]
        y += 70
    return svg(760, y + 4, 'WordPiece on the word embeddings, step by step. Each frame tries the longest remaining piece first and shortens it until it is in the vocabulary. It takes 18 lookups to find the four pieces em, ##bed, ##ding and ##s.', b)


F['p2_wordpiece_frames'] = wordpiece_frames()


def params_terms():
    P = M['params']
    g = M['gpt']
    b = [text(20, 26, 'Parameters term by term, in millions: BERT-base, BERT-large and OpenAI GPT', 't-title')]
    groups = []
    for name in ['bert-base-uncased', 'bert-large-uncased']:
        d = P[name]
        L = d['L']
        t = dict(d['terms'])
        emb = sum(v for k, v in d['terms'] if 'per layer' not in k and 'pooler' not in k)
        attn = L * (t['Q, K, V, O matrices 4 H x H (per layer)'] + t['their biases 4H (per layer)'])
        ffn_ = L * (t['FFN W1 H x 4H + W2 4H x H (per layer)'] + t['FFN biases 4H + H (per layer)'])
        ln = L * t['two LayerNorms 4H (per layer)']
        groups.append((name.replace('-uncased', ''), d['total'], [('embeddings', emb, 's4'), ('attention', attn, 's1'), ('feed-forward', ffn_, 's2'),
                                                                     ('LayerNorms', ln, 's3'), ('pooler', t['pooler H x H + H'], 's3')]))
    scale = 560 / 340e6
    for k, (name, total, parts) in enumerate(groups):
        y = 60 + k * 70
        b.append(text(20, y + 20, name, 't-note'))
        x = 140
        for lab, v, c in parts:
            w = v * scale
            b.append(f'<g class="mark"><title>{name}, {lab}: {v:,}</title><rect class="{c}" x="{x:.1f}" y="{y}" width="{max(w, 1):.1f}" height="30" rx="2" style="fill-opacity:0.85"/></g>')
            if w > 60:
                b.append(text(x + w / 2, y + 20, f'{lab} {v / 1e6:.1f}', 't-cell on', 'middle'))
            x += w
        b.append(text(x + 8, y + 20, f'{total / 1e6:.1f} M', 't-val'))
    y = 60 + 2 * 70
    b += [text(20, y + 20, 'OpenAI GPT', 't-note'), f'<rect class="s4" x="140" y="{y}" width="{g["params"] * scale:.1f}" height="30" rx="2" style="fill-opacity:0.5"/>',
          text(140 + g['params'] * scale + 8, y + 20, f'{g["params"] / 1e6:.1f} M', 't-val'), text(150, y + 20, f'L={g["L"]}, H={g["H"]}, {g["A"]} heads, vocabulary {g["vocab"]:,}', 't-tick')]
    b += [text(20, y + 60, 'BERT-base copies GPT\'s L, H and A. The totals differ by 7 million mainly because GPT\'s vocabulary is larger (40,478 vs 30,522 tokens).', 't-muted')]
    return svg(760, y + 74, 'Parameter counts split by part. BERT-base 109.5 million: embeddings 23.8, attention 28.3, feed-forward 56.7. BERT-large 335.1 million: embeddings 31.8, attention 100.8, feed-forward 201.4. OpenAI GPT, which has the same number of layers, hidden size and heads as BERT-base, has 116.5 million, mostly because of its larger vocabulary.', b)


F['p2_params_terms'] = params_terms()

json.dump(F, open('results/figs_part2.json', 'w'))
print('figures:', ', '.join(F))
