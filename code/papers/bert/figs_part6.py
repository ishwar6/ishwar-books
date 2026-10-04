"""Figures for Part 6 (impact, limits, summary), drawn from results/part6.json. Writes results/figs_part6.json."""
import json
from figlib import svg, text, box, arrow, esc
from bertfig import token, row, line, hbars
from alammar import vec, matrix, frame, brace

R = json.load(open('results/part6.json'))
M = json.load(open('results/part6_math.json'))
F = {}


def timeline():
    # dates: first arXiv version (the month in the arXiv id); the Search post 25 Oct 2019; the NAACL award June 2019
    ev = [
        ('Jun 2017', 'Transformer', 'the architecture BERT is built from', 'box'),
        ('Oct 2018', 'BERT', 'this paper: deep bidirectional pre-training', 'box-on'),
        ('Jun 2019', 'BERT wins NAACL best long paper', 'the conference where it was published', 'box-3'),
        ('Jun 2019', 'XLNet', 'bidirectional context without [MASK]', 'box-1'),
        ('Jul 2019', 'RoBERTa', 'same model, trained better; drops NSP', 'box-1'),
        ('Jul 2019', 'SpanBERT', 'masks whole spans of words', 'box-1'),
        ('Aug 2019', 'Sentence-BERT', 'turns BERT into a sentence-embedding model', 'box-1'),
        ('Sep 2019', 'ALBERT', 'far fewer parameters; sentence order instead of NSP', 'box-1'),
        ('Oct 2019', 'DistilBERT', '40% smaller, 60% faster, keeps 97%', 'box-1'),
        ('Oct 2019', 'BERT in Google Search', '"one in 10 searches in the U.S. in English"', 'box-3'),
        ('Mar 2020', 'ELECTRA', 'learns from every token, not just the masked 15%', 'box-1'),
        ('Jun 2020', 'DeBERTa', 'separate vectors for content and position', 'box-1'),
        ('Dec 2024', 'ModernBERT', '8192-token inputs, 2 trillion training tokens', 'box-1'),
    ]
    b, x = [], 130
    b.append(line(x, 20, x, 20 + 30 * len(ev) - 14, 'edge-dim'))
    for k, (d, name, note, c) in enumerate(ev):
        y = 26 + k * 30
        b += [text(x - 18, y + 4, d, 't-tick', 'end'), box(x - 6, y - 6, 12, 12, c, 3),
              text(x + 20, y + 5, name, 't-note'), text(410, y + 5, note, 't-muted')]
    b.append(text(20, 30 * len(ev) + 28, 'Dates are first arXiv versions, except the award (NAACL, June 2019) and the Search post (25 October 2019).', 't-muted'))
    return svg(760, 30 * len(ev) + 40, 'A timeline from the Transformer in 2017 to ModernBERT in 2024: BERT in October 2018, then a wave of follow-up models in 2019 and 2020.', b)


F['p6_timeline'] = timeline()


def similarity():
    rows = R['similarity']
    methods = [('cls', 'raw BERT [CLS]'), ('mean', 'raw BERT, mean of tokens'), ('st', 'all-MiniLM-L6-v2 (fine-tuned)')]
    L, Rr, T = 220, 720, 50
    X = lambda v: L + (v + 0.2) / 1.2 * (Rr - L)
    b = [text(20, 24, 'Cosine similarity of six sentence pairs', 't-title')]
    for v in [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        b += [f'<line class="grid" x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{T + 3 * 56}"/>', text(X(v), T + 3 * 56 + 18, f'{v:.1f}', 't-tick', 'middle')]
    for k, (key, name) in enumerate(methods):
        yy = T + 28 + k * 56
        b.append(text(L - 14, yy + 4, name, 't-tick', 'end'))
        for r in rows:
            c = 's3' if r['kind'] == 'related' else 's2'
            dy = -7 if r['kind'] == 'related' else 7
            b.append(f'<g class="mark"><title>{esc(r["a"])} | {esc(r["b"])}: {r[key]:.3f}</title><circle class="{c} ring" cx="{X(r[key]):.1f}" cy="{yy + dy}" r="6"/></g>')
    ly = T + 3 * 56 + 40
    b += [f'<circle class="s3" cx="230" cy="{ly}" r="6"/>', text(242, ly + 4, 'related pair', 't-tick'),
          f'<circle class="s2" cx="350" cy="{ly}" r="6"/>', text(362, ly + 4, 'unrelated pair', 't-tick'),
          text(480, ly + 4, 'cosine similarity (1 = same direction)', 't-muted')]
    return svg(760, ly + 16, 'Cosine similarities for three related and three unrelated sentence pairs. With raw BERT the two groups overlap. The model fine-tuned for similarity separates them cleanly.', b)


F['p6_similarity'] = similarity()


def signal():
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'looked', 'out', 'of', 'the', 'window', 'at', 'the', 'birds']
    W, G = 42, 5
    b = [text(20, 24, 'BERT (masked LM): only the hidden tokens give a training signal', 't-title'),
         text(20, 134, 'ELECTRA (replaced token detection): every token gives a signal', 't-title')]
    masked = {2, 9}
    cls = ['box-mask' if i in masked else 'box' for i in range(len(words))]
    shown = ['[M]' if i in masked else w for i, w in enumerate(words)]
    parts, cx = row(20, 40, shown, W, G, classes=cls, tcls='t-muted')
    b += parts
    for i, x in enumerate(cx):
        if i in masked:
            b += [arrow(x, 72, x, 90), text(x, 106, 'loss', 't-on', 'middle')]
    rep = {5: 'rug'}
    shown2 = [rep.get(i, w) for i, w in enumerate(words)]
    cls2 = ['box-2' if i in rep else 'box' for i in range(len(words))]
    parts, cx = row(20, 150, shown2, W, G, classes=cls2, tcls='t-muted')
    b += parts
    for i, x in enumerate(cx):
        b += [arrow(x, 182, x, 198), text(x, 214, 'fake' if i in rep else 'real', 't-bad' if i in rep else 't-s3', 'middle')]
    b.append(text(20, 240, 'The paper masks 15% of tokens (about 2 of these 15). ELECTRA asks "real or replaced?" at every position.', 't-muted'))
    return svg(760, 252, 'Where the training signal comes from. In BERT only the masked positions are scored. In ELECTRA every position is scored as real or replaced.', b)


F['p6_signal'] = signal()


def napkin():
    b = [text(20, 24, '1. Input (Part 2)', 't-title'), text(20, 150, '2. Pre-training, two tasks at once (Part 3)', 't-title'),
         text(20, 330, '3. Fine-tuning: same body, one new layer per task (Part 4)', 't-title')]
    toks = ['[CLS]', 'my', 'dog', 'is', '[MASK]', '[SEP]', 'he', 'likes', 'play', '##ing', '[SEP]']
    W, G = 58, 6
    cls = ['box-on', 'box-1', 'box-1', 'box-1', 'box-mask', 'box', 'box-2', 'box-2', 'box-2', 'box-2', 'box']
    parts, cx = row(20, 40, toks, W, G, classes=cls, tcls='t-muted')
    b += parts
    b += [text(20, 92, 'each input vector = token embedding + segment embedding (A or B) + position embedding', 't-tick'),
          text(20, 110, 'blue = sentence A, orange = sentence B; WordPiece splits "playing" into play + ##ing', 't-muted')]
    # encoder
    b += [box(20, 166, 700, 50, 'box-1', 10), text(370, 188, 'Transformer encoder: 12 layers, 768 numbers per token, 12 heads (BERT-base)', 't-note', 'middle'),
          text(370, 206, 'every token looks at every token, left and right, in every layer', 't-tick', 'middle')]
    for x in (cx[0], cx[4]):
        b.append(arrow(x, 218, x, 244))
    b += [box(cx[0] - 29, 246, 92, 30, 'box-on', 6), text(cx[0] + 17, 266, 'C → NSP', 't-tick', 'middle'),
          text(cx[0] + 70, 266, 'IsNext or NotNext?', 't-muted'),
          box(cx[4] - 46, 246, 92, 30, 'box-mask', 6), text(cx[4], 266, 'T → MLM', 't-tick', 'middle'),
          text(cx[4] + 52, 266, 'which word was hidden? ("cute")', 't-muted'),
          text(20, 298, 'loss = mean masked-LM loss + mean NSP loss; 15% of tokens are chosen for prediction (80% [MASK], 10% random, 10% kept)', 't-muted')]
    heads = [('classify', 'C · Wᵀ → label', 'GLUE, sentiment', 'box-3'),
             ('find a span', 'S · Tᵢ and E · Tⱼ', 'SQuAD', 'box-2'),
             ('tag tokens', 'Tᵢ → label', 'NER', 'box-4'),
             ('pick a choice', 'v · C, softmax', 'SWAG', 'box-on')]
    for i, (name, f, ex, c) in enumerate(heads):
        x = 20 + i * 178
        b += [box(x, 346, 166, 76, c, 10), text(x + 83, 368, name, 't-note', 'middle'), text(x + 83, 388, f, 't-mono', 'middle'),
              text(x + 83, 408, ex, 't-muted', 'middle')]
    return svg(760, 436, 'BERT on a napkin. Text goes in as WordPiece tokens with [CLS] and [SEP]. A 12-layer bidirectional encoder is pre-trained with masked LM and next sentence prediction. For each task, one small layer is added on top and everything is fine-tuned.', b)


F['p6_napkin'] = napkin()


def quadrant():
    """The conclusion as a 2 x 2: how deep the mixing is, and which directions it uses."""
    b = [text(20, 24, 'Where the conclusion puts BERT', 't-title'),
         text(300, 52, 'one direction (left to right)', 't-note', 'middle'), text(580, 52, 'both directions', 't-note', 'middle'),
         text(118, 132, 'shallow:', 't-note', 'end'), text(118, 150, 'directions meet', 't-muted', 'end'), text(118, 166, 'only at the end', 't-muted', 'end'),
         text(118, 262, 'deep:', 't-note', 'end'), text(118, 280, 'every layer', 't-muted', 'end'), text(118, 296, 'mixes context', 't-muted', 'end')]
    cells = [(160, 70, 'box', 'a single one-way LSTM LM', 'one direction, nothing to join', ''),
             (440, 70, 'box-2', 'ELMo (Peters et al., 2018a)', 'a left LSTM + a right LSTM,', 'outputs concatenated at the top'),
             (160, 200, 'box-1', 'OpenAI GPT (Radford et al., 2018)', 'deep Transformer, left context only:', '"deep unidirectional architectures"'),
             (440, 200, 'box-on', 'BERT (this paper)', 'deep Transformer, both sides in all', 'layers: "deep bidirectional"')]
    for x, y, c, t1, t2, t3 in cells:
        b += [box(x, y, 270, 112, c, 12), text(x + 135, y + 40, t1, 't-note', 'middle'), text(x + 135, y + 64, t2, 't-tick', 'middle'),
              text(x + 135, y + 82, t3, 't-tick', 'middle')]
    b += [arrow(432, 256, 446, 256, True), text(440, 330, 'the step the conclusion calls "our major contribution"', 't-muted', 'middle')]
    return svg(760, 342, 'A two by two grid. Columns: one direction or both directions. Rows: shallow (directions meet only at the end) or deep (every layer mixes context). ELMo is shallow and two-way, OpenAI GPT is deep and one-way, BERT is deep and two-way.', b)


F['p6_quadrant'] = quadrant()


def pipeline():
    """Alammar-style: one sentence pair all the way through BERT-base."""
    toks = ['[CLS]', 'my', 'dog', 'is', '[MASK]', '[SEP]', 'he', 'likes', 'play', '##ing', '[SEP]']
    seg = ['A'] * 6 + ['B'] * 5
    W, G, x0 = 54, 6, 24
    b = [text(20, 22, 'One input, all the way through BERT-base', 't-title')]
    b += [text(20, 50, '1  WordPiece tokens', 't-note')]
    cls = ['box-on', 'box-1', 'box-1', 'box-1', 'box-mask', 'box', 'box-2', 'box-2', 'box-2', 'box-2', 'box']
    parts, cx = row(x0, 60, toks, W, G, classes=cls, tcls='t-muted')
    b += parts
    b += [text(20, 122, '2  three 768-number vectors per token, added: token + segment + position', 't-note')]
    import math as _m
    for i, x in enumerate(cx):
        for k, (c, yy) in enumerate([('s1', 134), ('s2' if seg[i] == 'B' else 's3', 150), ('s4', 166)]):
            vals = [0.3 + 0.7 * abs(_m.sin(1.7 * i + 2.3 * k + 0.9 * j)) for j in range(5)]
            b += vec(x - 21, yy, vals, c, cell=7, gap=1.5, outline=False)
    b += [text(690, 141, 'token', 't-s1'), text(690, 157, 'segment A', 't-s3'), text(690, 173, 'position', 't-s4'), text(690, 189, '(B: orange)', 't-muted')]
    b += [text(20, 208, '3  twelve identical encoder layers (12 attention heads + a feed-forward network each)', 't-note')]
    for k in range(3):
        y = 218 + k * 20
        b += [box(x0 - 6, y, 11 * (W + G) + 4 - G, 18, 'box-1', 6)]
    b += [text(x0 + 5.5 * (W + G) - G / 2, 231, 'layer 1   ...   layer 12: every token attends to every token', 't-tick', 'middle')]
    b += [text(20, 300, '4  one output vector per token (768 numbers each)', 't-note')]
    for i, x in enumerate(cx):
        b += [arrow(x, 306, x, 318)]
        vals = [0.3 + 0.7 * abs(_m.cos(1.3 * i + 0.7 * j)) for j in range(5)]
        b += vec(x - 21, 324, vals, 's3' if i else 's2', cell=7, gap=1.5)
        b += [text(x, 356, 'C' if i == 0 else f'T{chr(0x2080 + i) if i < 10 else chr(0x2081) + chr(0x2080)}', 't-math', 'middle')]
    b += [text(20, 392, '5  a small head reads what the task needs', 't-note')]
    heads = [('C', 'NSP / classification', 'softmax(C Wᵀ)', 'box-3'), ('T at [MASK]', 'masked LM', 'softmax over 30,522 words', 'box-mask'),
             ('every Tᵢ', 'tagging (NER)', 'one label per token', 'box-4'), ('Tᵢ · S, Tⱼ · E', 'answer span', 'start and end scores', 'box-2')]
    for i, (src, name, f, c) in enumerate(heads):
        x = 20 + i * 182
        b += [box(x, 404, 172, 70, c, 10), text(x + 86, 424, name, 't-note', 'middle'), text(x + 86, 444, f, 't-tick', 'middle'),
              text(x + 86, 463, 'reads ' + src, 't-muted', 'middle')]
    return svg(760, 486, 'One sentence pair through BERT-base. WordPiece tokens; three vectors per token (token, segment, position) are added; twelve encoder layers; one output vector per token, C for [CLS] and T for the others; a small head per task reads C or the T vectors.', b)


F['p6_pipeline'] = pipeline()


def windows():
    w = M['windows']
    N, step, L = w['N'], w['step'], 510
    X = lambda t: 40 + t / N * 680
    b = [text(20, 22, f'A {N}-token text, read as {w["n_windows"]} overlapping windows of 512', 't-title')]
    b += [box(X(0), 46, X(N) - X(0), 26, 'box', 5), text(X(N / 2), 64, f'the whole text: {N} tokens (BERT has position vectors for only 512)', 't-tick', 'middle')]
    for k in range(w['n_windows']):
        s0 = k * step
        s1 = min(N, s0 + L)
        y = 92 + k * 40
        b += [box(X(s0), y, X(s1) - X(s0), 26, 'box-1' if k == 0 else 'box-2', 5),
              text(X(s0) + 8, y + 18, f'window {k + 1}: tokens {s0}-{s1 - 1} ({s1 - s0} + [CLS] + [SEP])', 't-tick')]
    b += [box(X(step), 160, X(L) - X(step), 14, 'box-on', 3), text((X(step) + X(L)) / 2, 192, f'overlap o = {w["overlap"]} tokens, seen by both windows', 't-on', 'middle'),
          text(X(0), 192, f'step s = {step}', 't-muted')]
    # cost bars, log-free: one long input vs windows, scaled to the largest
    b += [text(20, 236, 'Attention entries per head per layer: one long input vs windows of 512', 't-title')]
    rows_ = w['table']
    vmax = max(max(r['one_long'], r['windowed']) for r in rows_)
    for i, r in enumerate(rows_):
        y = 256 + i * 58
        b += [text(130, y + 14, f'N = {r["N"]:,}', 't-note', 'end'), text(130, y + 32, f'{r["windows"]} windows', 't-muted', 'end')]
        for j, (v, c, lab) in enumerate([(r['one_long'], 's2', 'one long input'), (r['windowed'], 's1', 'windows')]):
            ww = max(3, 440 * v / vmax)
            b += [f'<g class="mark"><title>{lab}: {v:,}</title><rect class="{c}" x="140" y="{y + j * 22}" width="{ww:.1f}" height="18" rx="3"/></g>',
                  text(146 + ww, y + j * 22 + 14, f'{v:,} ({lab})', 't-tick')]
    return svg(760, 440, 'A 720-token text is cut into two 512-token windows that overlap by 128 tokens. Below, the number of attention entries for one long input grows with the square of the length, while windows grow in a straight line.', b)


F['p6_windows'] = windows()


def twomask():
    t = M['two_blanks']
    pr = {(r['a'], r['b']): r for r in t['pairs']}
    nt, ny = pr[('new', 'town')], pr[('new', 'york')]
    b = []
    fw, fh = 362, 170
    pos = [(10, 10), (388, 10), (10, 196), (388, 196)]
    titles = ['both blanks hidden', 'multiply the two guesses', 'fill in the first blank', 'the chain rule']
    for k, ((x, y), ttl) in enumerate(zip(pos, titles)):
        b += frame(x, y, fw, fh, k + 1, ttl)
    # frame 1
    x, y = 10, 10
    b += token(x + 16, y + 44, 70, 'from', 'box', 26, 't-muted') + token(x + 92, y + 44, 70, '[MASK]', 'box-mask', 26, 't-muted') + \
        token(x + 168, y + 44, 70, '[MASK]', 'box-mask', 26, 't-muted') + token(x + 244, y + 44, 60, 'to', 'box', 26, 't-muted')
    b += [text(x + 127, y + 92, 'blank 1', 't-tick', 'middle'), text(x + 203, y + 92, 'blank 2', 't-tick', 'middle')]
    for i, (w_, p) in enumerate(R['two_masks'][0]['top5'][0][:3]):
        b += [text(x + 127, y + 112 + i * 16, f'{w_} {p:.3f}', 't-tick', 'middle')]
    for i, (w_, p) in enumerate(R['two_masks'][0]['top5'][1][:3]):
        b += [text(x + 203, y + 112 + i * 16, f'{w_} {p:.3f}', 't-tick', 'middle')]
    # frame 2
    x, y = 388, 10
    b += [text(x + 18, y + 60, 'independent score p(a) · p(b):', 't-tick'),
          text(x + 18, y + 88, f'new town  {nt["p_a"]:.4f} × {nt["p_b"]:.4f} = {nt["independent"]:.4f}', 't-mono'),
          text(x + 18, y + 112, f'new york  {ny["p_a"]:.4f} × {ny["p_b"]:.4f} = {ny["independent"]:.4f}', 't-mono'),
          text(x + 18, y + 144, 'the winner is "new town", a place nobody flew from', 't-bad')]
    # frame 3
    x, y = 10, 196
    b += token(x + 16, y + 44, 70, 'from', 'box', 26, 't-muted') + token(x + 92, y + 44, 70, 'new', 'box-on', 26, 't-tick') + \
        token(x + 168, y + 44, 70, '[MASK]', 'box-mask', 26, 't-muted') + token(x + 244, y + 44, 60, 'to', 'box', 26, 't-muted')
    for i, (w_, p) in enumerate(R['conditional'][0]['top3']):
        b += [text(x + 203, y + 98 + i * 16, f'{w_} {p:.3f}', 't-tick' if i else 't-tick t-strong', 'middle')]
    b += [text(x + 18, y + 150, 'knowing blank 1 makes blank 2 almost certain', 't-muted')]
    # frame 4
    x, y = 388, 196
    b += [text(x + 18, y + 60, 'chain rule p(a) · p(b | a):', 't-tick'),
          text(x + 18, y + 88, f'new york  {ny["p_a"]:.4f} × {ny["p_b_given_a"]:.4f} = {ny["chain"]:.4f}', 't-mono'),
          text(x + 18, y + 112, f'new town  {nt["p_a"]:.4f} × {nt["p_b_given_a"]:.6f} ≈ 0', 't-mono'),
          text(x + 18, y + 144, '"new york" wins by far', 't-s3')]
    return svg(760, 378, 'Two blanks in four steps. With both blanks hidden, BERT guesses each one on its own, and multiplying the two guesses favours new town. Filling the first blank with new makes york almost certain, and the chain rule picks new york.', b)


F['p6_twomask'] = twomask()


def electra():
    words = ['the', 'chef', 'cooked', 'the', 'meal']
    b = [text(20, 22, 'ELECTRA: replaced token detection, in three steps', 't-title')]
    W, G = 70, 8
    # step 1: mask
    b += frame(10, 36, 740, 76, 1, 'mask a few tokens (like BERT)')
    parts, cx = row(250, 64, ['the', '[MASK]', 'cooked', 'the', '[MASK]'], W, G, classes=['box', 'box-mask', 'box', 'box', 'box-mask'], tcls='t-muted')
    b += parts
    # step 2: generator fills
    b += frame(10, 122, 740, 76, 2, 'a small generator (a small masked LM) fills them')
    parts, cx = row(250, 150, ['the', 'chef', 'cooked', 'the', 'soup'], W, G, classes=['box', 'box-3', 'box', 'box', 'box-2'], tcls='t-tick')
    b += parts
    b += [text(cx[1], 196, 'guessed right', 't-muted', 'middle'), text(cx[4], 196, 'plausible but wrong', 't-muted', 'middle')]
    # step 3: discriminator labels all
    b += frame(10, 208, 740, 100, 3, 'the main model labels EVERY token: original or replaced?')
    parts, cx = row(250, 236, ['the', 'chef', 'cooked', 'the', 'soup'], W, G, tcls='t-tick')
    b += parts
    labels = ['original', 'original', 'original', 'original', 'replaced']
    for x, l in zip(cx, labels):
        b += [arrow(x, 268, x, 280), text(x, 296, l, 't-bad' if l == 'replaced' else 't-s3', 'middle')]
    b += [text(20, 330, 'BERT learns from 2 of these 5 positions; ELECTRA\'s main model gets a loss at all 5.', 't-muted'),
          text(20, 348, '"chef" counts as original, because the generator happened to produce the real word.', 't-muted')]
    return svg(760, 360, 'ELECTRA in three steps: mask a few tokens, let a small generator fill them with plausible words, then train the main model to label every token as original or replaced.', b)


F['p6_electra'] = electra()


def cosine():
    c = M['cosine']
    b = [text(20, 22, 'Cosine similarity: the angle between two vectors', 't-title')]
    ox, oy, s = 70, 220, 36
    b += [line(ox, oy, ox + 6 * s, oy, 'edge-dim'), line(ox, oy, ox, oy - 5.5 * s, 'edge-dim')]
    b += [line(ox, oy, ox + 3 * s, oy - 4 * s, 'edge-1', 'ah'), line(ox, oy, ox + 4 * s, oy - 3 * s, 'edge-2', 'ah'),
          text(ox + 3 * s - 4, oy - 4 * s - 10, 'a = (3, 4)', 't-s1', 'middle'), text(ox + 4 * s + 12, oy - 3 * s, 'b = (4, 3)', 't-s2'),
          f'<path class="edge-on" d="M{ox + 1.2 * s * 0.6:.1f},{oy - 1.2 * s * 0.8:.1f} A{1.2 * s:.1f},{1.2 * s:.1f} 0 0 1 {ox + 1.2 * s * 0.8:.1f},{oy - 1.2 * s * 0.6:.1f}"/>',
          text(ox + 56, oy - 46, 'θ', 't-on'),
          text(ox, oy + 26, 'toy, in 2-D: a · b = 3·4 + 4·3 = 24, |a| = |b| = 5, cos θ = 24 / 25 = 0.96', 't-tick')]
    x = 380
    b += [text(x, 60, 'Real, in 768-D: [CLS] vectors of bert-base-uncased', 't-note'),
          text(x, 84, 'a = C("A man is playing a guitar.")', 't-tick'), text(x, 102, 'b = C("A person is making music.")', 't-tick')]
    b += vec(x, 116, c['first4_a'] + [0.2, 0.5, 0.1, 0.4], 's1', cell=12, gap=2) + [text(x + 120, 126, '... 768 numbers', 't-muted')]
    b += vec(x, 136, c['first4_b'] + [0.3, 0.4, 0.2, 0.1], 's2', cell=12, gap=2) + [text(x + 120, 146, '... 768 numbers', 't-muted')]
    b += [text(x, 180, f'a · b = {c["dot"]:.3f}', 't-mono'), text(x, 200, f'|a| = {c["norm_a"]:.3f},  |b| = {c["norm_b"]:.3f}', 't-mono'),
          text(x, 226, f'cos = {c["dot"]:.3f} / ({c["norm_a"]:.3f} × {c["norm_b"]:.3f}) = {c["cos"]:.3f}', 't-mono')]
    return svg(760, 262, 'Cosine similarity measures the angle between two vectors. In two dimensions, (3, 4) and (4, 3) have cosine 0.96. For two real 768-number [CLS] vectors of BERT, the dot product 199.086 divided by the two lengths gives 0.851.', b)


F['p6_cosine'] = cosine()


def distill():
    d = M['distill']
    b = [text(20, 22, 'Distillation: the student learns from the teacher\'s whole probability list', 't-title'),
         text(20, 44, 'i went to the [MASK] to deposit my paycheck.', 't-mono')]
    b += hbars(20, 92, [(w, p) for w, p in d['teacher_top5']], 210, title=f'teacher: BERT-base ({d["params_teacher"] / 1e6:.1f}M encoder weights)', cls='s1', label_w=80, hi='bank')
    b += hbars(400, 92, [(w, p) for w, p in d['student_top5']], 210, title=f'student: DistilBERT ({d["params_student"] / 1e6:.1f}M)', cls='s2', label_w=90, hi='bank')
    b += [text(20, 236, f'hard label only:  −log s(bank) = {d["ce_hard"]:.3f}', 't-mono'),
          text(20, 258, f'soft targets:     −Σ t(w) log s(w) = {d["ce_soft"]:.3f}   (smallest possible: the teacher\'s entropy, {d["teacher_entropy"]:.3f})', 't-mono')]
    return svg(760, 276, 'The same blank through a teacher (BERT-base) and a student (DistilBERT). The teacher puts 0.901 on bank, the student 0.365. Distillation trains the student to match the teacher\'s whole list of probabilities, not only the one right word.', b)


F['p6_distill'] = distill()


def family():
    """Which follow-up model attacks which limit."""
    lim = [('512-token wall (Limit 1)', ['ModernBERT: 8192 tokens']),
           ('[MASK] mismatch (Limit 2)', ['XLNet: no [MASK], permuted order', 'ELECTRA: real tokens in, real/replaced out']),
           ('only 15% teach (Limit 3)', ['ELECTRA: a loss at every token']),
           ('blanks independent (Limit 4)', ['XLNet: one at a time, later guesses see earlier']),
           ('no sentence vector (Limit 5)', ['Sentence-BERT: fine-tuned for cosine']),
           ('too big, too slow (Limit 7)', ['ALBERT: shared layers', 'DistilBERT: 6 layers, distilled']),
           ('undertrained recipe', ['RoBERTa: more data, no NSP, dynamic masks', 'SpanBERT: mask whole spans'])]
    b = [text(20, 22, 'Which follow-up attacks which weakness', 't-title')]
    y = 40
    for name, fixes in lim:
        h = 26 * len(fixes) + 8
        b += [box(20, y, 250, h, 'box-mask', 8), text(32, y + h / 2 + 4, name, 't-note')]
        for k, f in enumerate(fixes):
            yy = y + 4 + k * 26
            b += [box(330, yy, 410, 22, 'box-1', 6), text(342, yy + 15, f, 't-tick')]
        b += [arrow(272, y + h / 2, 326, y + h / 2)]
        y += h + 10
    return svg(760, y + 6, 'Each weakness of BERT on the left, and the follow-up model that attacks it on the right: ModernBERT for length, XLNet and ELECTRA for the mask, ELECTRA for the 15% signal, Sentence-BERT for sentence vectors, ALBERT and DistilBERT for size, RoBERTa and SpanBERT for the training recipe.', b)


F['p6_family'] = family()


def crossbi():
    b = [text(190, 22, 'Cross-encoder (BERT as in the paper)', 't-title', 'middle'), text(570, 22, 'Bi-encoder (Sentence-BERT)', 't-title', 'middle')]
    b += token(40, 200, 140, 'query', 'box-1', 30, 't-tick') + token(200, 200, 140, 'passage', 'box-2', 30, 't-tick')
    b += [arrow(190, 198, 190, 172), box(40, 110, 300, 60, 'box-on', 10), text(190, 136, 'one BERT reads both together', 't-note', 'middle'),
          text(190, 154, 'every query token attends to every passage token', 't-muted', 'middle'),
          arrow(190, 108, 190, 84), box(130, 52, 120, 30, 'box-3', 8), text(190, 72, 'one score', 't-tick', 'middle'),
          text(190, 252, 'accurate; must run once per (query, passage) pair', 't-muted', 'middle')]
    for x, lab, c in [(430, 'query', 'box-1'), (600, 'passage', 'box-2')]:
        b += token(x, 200, 130, lab, c, 30, 't-tick') + [arrow(x + 65, 198, x + 65, 172), box(x, 116, 130, 54, 'box-on', 10),
                                                        text(x + 65, 140, 'BERT', 't-note', 'middle'), text(x + 65, 158, '(same weights)', 't-muted', 'middle'),
                                                        arrow(x + 65, 114, x + 65, 98)]
        b += vec(x + 20, 84, [0.4, 0.9, 0.3, 0.7, 0.5, 0.8], 's1' if lab == 'query' else 's2', cell=12, gap=2)
    b += [line(544, 90, 616, 90, 'edge-on'), text(580, 74, 'cosine', 't-on', 'middle'),
          text(570, 252, 'fast; passage vectors are computed once, ahead of time', 't-muted', 'middle'),
          line(380, 40, 380, 262, 'edge-dim')]
    return svg(760, 270, 'Two ways to compare a query with a passage. A cross-encoder reads both in one input and outputs one score. A bi-encoder turns each into its own vector with the same BERT, then compares the vectors with cosine similarity.', b)


F['p6_crossbi'] = crossbi()

json.dump(F, open('results/figs_part6.json', 'w'))
print('figures:', ', '.join(F))
