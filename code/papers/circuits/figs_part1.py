"""Figures for Part 1 (the big idea and the residual stream), drawn from results/part1.json and results/part1_toy.json.
Writes results/figs_part1.json (key -> svg string). Colours come only from CSS classes, so every figure follows the
site's dark, dim and light themes."""
import json
from figlib import svg, text, box, arrow, esc
from cfig import token, line

R = json.load(open('results/part1.json'))
T = json.load(open('results/part1_toy.json'))
F = {}
SER = ['s1', 's2', 's3', 's4']
BOX = ['box-1', 'box-2', 'box-3', 'box-4']


def dashed_arrow(d, label=None, lx=0, ly=0, anchor='middle'):
    b = [f'<path class="drop" d="{d}" marker-end="url(#ah)"/>']
    if label:
        b.append(text(lx, ly, label, 't-note', anchor))
    return b


def column(x, y, vals, cls='s1', cell=30, fmt=lambda v: f'{v:g}', hi=None, w=46):
    """A vector drawn as a vertical column of cells with its numbers."""
    b = []
    m = max(1e-9, max(abs(v) for v in vals))
    for i, v in enumerate(vals):
        op = 0.10 + 0.55 * abs(v) / m
        b.append(f'<rect class="{cls}" x="{x:.1f}" y="{y + i * cell:.1f}" width="{w}" height="{cell - 4}" rx="4" style="fill-opacity:{op:.2f}"/>')
        if hi is not None and i == hi:
            b.append(f'<rect class="box-on" x="{x - 2:.1f}" y="{y + i * cell - 2:.1f}" width="{w + 4}" height="{cell}" rx="5" style="fill:none"/>')
        b.append(text(x + w / 2, y + i * cell + cell / 2 + 2, fmt(v), 't-val', 'middle'))
    return b


# ---------------------------------------------------------------- 1. reverse engineering: program vs network
def reverse():
    b = []
    rows = [('Ordinary software', 'source code', ['if word == "Barack":', '    say("Obama")'], 'compiler', 'binary file',
             [' '.join(T['binary_bytes'].split()[:8]), ' '.join(T['binary_bytes'].split()[8:16]) + ' ...'], 'reverse engineering'),
            ('A trained transformer', 'the algorithm', ['(unknown: this is what', ' we want to find)'], 'training', 'weights',
             [f'{R["attn-only-2l"]["params"]:,} numbers:', ', '.join(f'{v:g}' for v in R['we_sample']) + ' ...'], 'mechanistic interpretability')]
    for k, (title, a, alines, verb, c, clines, back) in enumerate(rows):
        y = 34 + k * 170
        b.append(text(24, y, title, 't-title'))
        b += [box(24, y + 14, 230, 84, 'box-1' if k == 0 else 'box-3', 10), text(139, y + 36, a, 't-note', 'middle')]
        for i, l in enumerate(alines):
            b.append(text(40, y + 60 + 18 * i, l, 't-code' if k == 0 else 't-tick'))
        b += [arrow(262, y + 56, 466, y + 56), text(364, y + 48, verb, 't-tick', 'middle')]
        b += [box(474, y + 14, 262, 84, 'box', 10), text(605, y + 36, c, 't-note', 'middle')]
        for i, l in enumerate(clines):
            b.append(text(490, y + 60 + 18 * i, l, 't-code' if k == 0 else 't-tick'))
        b += dashed_arrow(f'M605,{y + 102} C605,{y + 132} 139,{y + 132} 139,{y + 104}', back, 372, y + 146)
    return svg(760, 372, 'Two rows. Top: source code is compiled into an unreadable binary file, and reverse engineering goes back from the binary to readable code. Bottom: training turns an unknown algorithm into 52 million weights, and mechanistic interpretability tries to go back from the weights to the algorithm.', b)


F['p1_reverse'] = reverse()


# ---------------------------------------------------------------- 2. the ladder of toy models
def ladder():
    cols = [('zero layers', 0, ['bigram statistics', '"Barack" → "Obama"'], 'Part 1 (this part)'),
            ('one layer', 1, ['bigrams + skip-trigrams', '"A … B C", simple copying'], 'Parts 2 to 4'),
            ('two layers', 2, ['heads compose:', 'induction heads'], 'Parts 5 and 6')]
    b = [text(380, 22, 'The paper climbs a ladder of attention-only models', 't-title', 'middle')]
    for k, (name, L, learns, part) in enumerate(cols):
        x = 40 + k * 245
        cx = x + 95
        b += [box(x, 36, 190, 334, 'box-ghost', 12), text(cx, 62, name, 't-title', 'middle')]
        ys = 326
        b += token(cx - 55, ys, 110, 'tokens', 'box', 26)
        b += [arrow(cx, ys - 2, cx, ys - 16)]
        b += token(cx - 55, ys - 46, 110, 'embed W_E', 'box-1', 26)
        top = ys - 46
        for l in range(L):
            yy = top - 56
            b += [arrow(cx, top, cx, yy + 28)]
            b += token(cx - 62, yy, 124, f'layer {l}: 8 heads', 'box-2', 26)
            b += [f'<circle class="node" cx="{cx + 76}" cy="{yy + 13}" r="9"/>', text(cx + 76, yy + 17.5, '+', 't-tick', 'middle')]
            top = yy
        yu = 108
        b += [arrow(cx, top, cx, yu + 28)]
        b += token(cx - 55, yu, 110, 'unembed W_U', 'box-3', 26)
        b += [arrow(cx, yu, cx, yu - 12), text(cx, yu - 18, 'logits', 't-tick', 'middle')]
        for i, l in enumerate(learns):
            b.append(text(cx, 398 + 18 * i, l, 't-note' if i == 0 else 't-tick', 'middle'))
        b.append(text(cx, 446, part, 't-muted', 'middle'))
    return svg(760, 460, 'Three attention-only models side by side: zero layers (embed then unembed: bigram statistics, covered in Part 1), one layer of eight heads (bigrams plus skip-trigrams, Parts 2 to 4), and two layers of eight heads (heads compose into induction heads, Parts 5 and 6).', b)


F['p1_ladder'] = ladder()


# ---------------------------------------------------------------- 3. what the toy model drops
def simplify():
    b = [text(190, 24, 'A standard block (GPT-2 style)', 't-title', 'middle'),
         text(570, 24, 'The paper\'s toy block', 't-title', 'middle'), line(380, 14, 380, 384, 'edge-dim')]
    # left stack, bottom to top
    L = [('layer norm', 'waste'), ('attention heads + biases', 'box-2'), ('add to stream', None), ('layer norm', 'waste'),
         ('MLP (+ biases)', 'waste'), ('add to stream', None)]
    y = 340
    b += [line(60, 372, 60, 44, 'edge', 'ah')]
    b.append(text(52, 372, 'x', 't-math', 'end'))
    for name, cls in L:
        if cls is None:
            b += [f'<circle class="node" cx="60" cy="{y + 14}" r="10"/>', text(60, 18.5 + y, '+', 't-tick', 'middle'), text(80, y + 19, name, 't-muted')]
        else:
            b += token(100, y, 210, name, cls, 28)
            b += [line(60, y + 14, 100, y + 14, 'edge')]
        y -= 52
    # right stack
    b += [line(440, 372, 440, 44, 'edge', 'ah')]
    b += token(480, 236, 200, 'attention heads', 'box-2', 28) + [line(440, 250, 480, 250, 'edge')]
    b += [f'<circle class="node" cx="440" cy="196" r="10"/>', text(440, 200.5, '+', 't-tick', 'middle'), text(460, 201, 'add to stream', 't-muted')]
    notes = [('no MLP: the big change', 'MLP layers were too hard to read'), ('no layer norm', 'folded into nearby weights'),
             ('no biases', 'folded in with an always-1 input')]
    for i, (a, c) in enumerate(notes):
        yy = 300 + 40 * i
        b += [box(470, yy - 13, 12, 12, 'waste', 2), text(490, yy - 2, a, 't-note'), text(490, yy + 15, c, 't-muted')]
    b.append(text(560, 100, 'one equation per layer:', 't-tick', 'middle'))
    b.append(text(560, 126, 'x  ←  x  +  Σ h(x)', 't-math', 'middle'))
    return svg(760, 410, 'Left: a standard transformer block with layer norm, attention heads with biases, an addition, another layer norm, an MLP with biases and another addition. Right: the paper\'s toy block keeps only the attention heads and the addition. The MLP is dropped because it was too hard to read; layer norm and biases are dropped because they can be folded into the weights.', b)


F['p1_simplify'] = simplify()


# ---------------------------------------------------------------- 4. one-hot times W_E picks a column
def onehot():
    W = T['B']['W_E']
    V = ['the', 'cat', 'dog', 'sat']
    b = [text(380, 22, 'Multiplying by a one-hot vector picks one column', 't-title', 'middle')]
    x0, y0, c = 80, 70, 46
    for j, w in enumerate(V):
        b.append(text(x0 + j * c + c / 2, y0 - 10, w, 't-note' if j == 0 else 't-tick', 'middle'))
    b.append(f'<rect class="box-1" x="{x0 - 3}" y="{y0 - 3}" width="{c + 2}" height="{3 * c + 2}" rx="6"/>')
    for i in range(3):
        b.append(text(x0 - 12, y0 + i * c + c / 2 + 4, f'dim {i + 1}', 't-muted', 'end'))
        for j in range(4):
            b += [box(x0 + j * c + 2, y0 + i * c + 2, c - 6, c - 6, 'box', 6), text(x0 + j * c + c / 2 - 1, y0 + i * c + c / 2 + 4, f'{W[i][j]:g}', 't-val', 'middle')]
    b += [text(x0 + 2 * c, y0 + 3 * c + 24, 'W_E   (3 × 4)', 't-math', 'middle'),
          text(x0 + 2 * c, y0 + 3 * c + 44, 'one column per token', 't-muted', 'middle')]
    # t
    tx = 330
    b.append(text(tx - 22, y0 + 2 * c - 20, '×', 't-title', 'middle'))
    for i, (w, v) in enumerate(zip(V, [1, 0, 0, 0])):
        yy = y0 - 22 + i * 40
        b += [box(tx, yy, 40, 34, 'box-1' if v else 'box', 6), text(tx + 20, yy + 22, str(v), 't-val', 'middle'), text(tx + 50, yy + 22, w, 't-tick')]
    b += [text(tx + 20, y0 + 3 * c + 24, 't  (4)', 't-math', 'middle'), text(tx + 20, y0 + 3 * c + 44, '"the" is token 1', 't-muted', 'middle')]
    # x0
    rx = 520
    b.append(text(rx - 28, y0 + 2 * c - 20, '=', 't-title', 'middle'))
    for i, v in enumerate(T['B']['x0']):
        b += [box(rx, y0 + i * c + 2, 44, c - 6, 'box-1', 6), text(rx + 22, y0 + i * c + c / 2 + 4, f'{v:g}', 't-val', 'middle')]
    b += [text(rx + 22, y0 + 3 * c + 24, 'x₀  (3)', 't-math', 'middle'), text(rx + 22, y0 + 3 * c + 44, 'the vector of "the"', 't-muted', 'middle')]
    b += [text(600, y0 + 30, 'each output number is', 't-tick'), text(600, y0 + 48, 'row · t = 1 × (column 1)', 't-tick'),
          text(600, y0 + 66, '+ 0 × (everything else)', 't-tick')]
    return svg(760, 270, 'A 3 by 4 embedding matrix with one column per token (the, cat, dog, sat), multiplied by the one-hot vector 1, 0, 0, 0 for "the", gives the first column, 1, 0, 0: the vector of "the".', b)


F['p1_onehot'] = onehot()


# ---------------------------------------------------------------- 5. the whole attention-only model with real shapes
def pipeline():
    b = [text(380, 22, 'attn-only-2l, with the paper\'s shapes', 't-title', 'middle')]
    sx = 300
    b += [line(sx, 392, sx, 144, 'path'), arrow(sx, 160, sx, 142)]
    # embed and unembed sit on the stream
    b += token(sx - 120, 392, 240, 'embed: x₀ = W_E t', 'box-1', 30)
    b += token(sx - 120, 110, 240, 'unembed: logits = W_U x₂', 'box-3', 30)
    b += token(sx - 120, 40, 240, 'softmax → next token', 'box', 30)
    b += [arrow(sx, 110, sx, 72)]
    # two layers of heads: read from the stream below, write back at a + node above
    for L, (yb, yr, yp, xin, xout) in enumerate([(296, 366, 270, 'x₀', 'x₁'), (200, 254, 174, 'x₁', 'x₂')]):
        b += token(40, yb, 190, f'layer {L}: 8 heads', 'box-2', 30)
        b += [line(sx, yr, 135, yr, 'edge'), arrow(135, yr, 135, yb + 32)]
        b += [f'<path class="edge" d="M230,{yb + 15} L{sx - 13},{yp + 6}" marker-end="url(#ah)"/>']
        b += [f'<circle class="node on" cx="{sx}" cy="{yp}" r="11"/>', text(sx, yp + 4.5, '+', 't-tick', 'middle')]
        b.append(text(sx + 16, yp - 8, xout, 't-math'))
        b.append(text(150, yr - 6, f'reads {xin}', 't-muted'))
    b.append(text(sx + 16, 384, 'x₀', 't-math'))
    notes = [(407, 'W_E: 512 × 48,262.  t: one-hot, 48,262 long.'), (318, 'each head adds a 512-vector to x₀'),
             (222, 'these heads read x₁, which already holds layer 0'), (125, 'W_U: 48,262 × 512.  48,262 logits out.'),
             (55, f'on our prompt: "{R["pred_token"]}" with probability {R["top5"][0][1]:.3f}')]
    for y, n in notes:
        b.append(text(440, y + 4, n, 't-tick'))
    b.append(text(440, 270, 'the stream: 512 numbers per token,', 't-muted'))
    b.append(text(440, 288, 'never transformed, only added to', 't-muted'))
    return svg(760, 436, 'The two-layer attention-only model as a vertical residual stream. The embedding writes x0 = W_E t (W_E is 512 by 48,262). Layer 0\'s eight heads read x0 and add their outputs to make x1. Layer 1\'s eight heads read x1 and add to make x2. The unembedding W_U (48,262 by 512) turns x2 into 48,262 logits, and softmax picks the next token.', b)


F['p1_pipeline'] = pipeline()


# ---------------------------------------------------------------- 6. the residual stream is a sum (toy numbers)
def sum_fig():
    C = T['C']
    x0 = T['B']['x0']
    S = C['split']
    b = [text(24, 26, '1. The stream at the end is the sum of what everyone wrote', 't-title')]
    cols = [('x₀ (embedding)', x0, 's1'), ('head 1 wrote', C['h1'], 's2'), ('head 2 wrote', C['h2'], 's3'), ('x (final)', C['x1'], 's4')]
    xs = [60, 230, 400, 600]
    for k, ((name, v, cls), x) in enumerate(zip(cols, xs)):
        b += column(x, 60, v, cls, 30)
        b.append(text(x + 23, 160, name, 't-tick', 'middle'))
        if k < 3:
            b.append(text(x + 23 + (170 if k < 2 else 185) / 2 + 2, 110, '+' if k < 2 else '=', 't-title', 'middle'))
    b.append(text(24, 206, '2. Multiply by W_U: the logits split into the same three pieces', 't-title'))
    V = ['the', 'cat', 'dog', 'sat']
    cols2 = [('W_U x₀', S['embedding x0'], 's1'), ('W_U h₁', S['head 1'], 's2'), ('W_U h₂', S['head 2'], 's3'), ('W_U x  (logits)', C['logits'], 's4')]
    for i, w in enumerate(V):
        b.append(text(50, 248 + i * 30, w, 't-tick', 'end'))
    for k, ((name, v, cls), x) in enumerate(zip(cols2, xs)):
        b += column(x, 232, v, cls, 30, fmt=lambda q: f'{q:g}')
        b.append(text(x + 23, 364, name, 't-math' if k < 3 else 't-tick', 'middle'))
        if k < 3:
            b.append(text(x + 23 + (170 if k < 2 else 185) / 2 + 2, 296, '+' if k < 2 else '=', 't-title', 'middle'))
    return svg(760, 380, 'Top: the embedding of "the" (1, 0, 0) plus what head 1 wrote (0, 0.5, 0) plus what head 2 wrote (0.2, 0, -0.3) equals the final vector (1.2, 0.5, -0.3). Bottom: multiplying each piece by the unembedding gives logits (0, 2, 1, 0), (0, 0, 0, 1) and (-0.3, 0.4, 0.2, 0), which add up to the final logits (-0.3, 2.4, 1.2, 1.0) for the, cat, dog and sat.', b)


F['p1_sum'] = sum_fig()


# ---------------------------------------------------------------- 7. direct logit attribution on the real model
def dla():
    P = R['parts'] + [dict(name='unembedding bias b_U', logit=R['b_U_pred'], norm=0)]
    tok = R['pred_token']
    row_h, x_zero, scale = 19, 330, 30
    top = 64
    b = [text(24, 24, f'Who pushed the logit of "{tok}"? Each part of the stream, one bar', 't-title'),
         text(24, 44, f'attn-only-2l, last position of the prompt ("… Mrs D|urs"); the bars add up to the real logit {R["logit_real"]:.2f}', 't-tick')]
    lo, hi = -2, 10
    for v in range(lo, hi + 1, 2):
        xx = x_zero + v * scale
        b += [line(xx, top - 6, xx, top + len(P) * row_h, 'grid'), text(xx, top + len(P) * row_h + 16, f'{v:+d}' if v else '0', 't-muted', 'middle')]
    b.append(line(x_zero, top - 6, x_zero, top + len(P) * row_h, 'axis'))
    for i, p in enumerate(P):
        y = top + i * row_h
        v = p['logit']
        cls = 's1' if v >= 0 else 's2'
        x = x_zero if v >= 0 else x_zero + v * scale
        w = max(1.5, abs(v) * scale)
        strong = abs(v) > 2
        b.append(text(x_zero - 2 * scale - 14, y + row_h / 2 + 4, p['name'].replace('embedding W_E t', 'token embedding').replace('position W_pos', 'position embedding'), 't-tick t-strong' if strong else 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(p["name"])}: {v:+.3f}</title><rect class="{cls}" x="{x:.1f}" y="{y + 3:.1f}" width="{w:.1f}" height="{row_h - 6}" rx="2"/></g>')
        b.append(text((x + w + 6) if v >= 0 else (x_zero + 6), y + row_h / 2 + 4, f'{v:+.2f}', 't-val' if strong else 't-tick'))
    yb = top + len(P) * row_h + 34
    b.append(text(24, yb, 'Blue pushes "' + tok + '" up, orange pushes it down. One head of the second layer (1.6) does almost half the work;', 't-tick'))
    a = R['top_head']['attends'][0]
    b.append(text(24, yb + 18, f'It looks back at "{a[1]}", the token that followed "urs" earlier in the prompt (attention {a[2]:.2f}), and copies it.', 't-tick'))
    return svg(760, yb + 30, f'Horizontal bars for the 21 parts of the residual stream at the last position, showing how much each adds to the logit of "{tok}". Head 1.6 adds 9.63, the layer 1 output bias 2.78, the unembedding bias 2.73, the token embedding 2.22; most other heads add less than 1.3 either way.', b)


F['p1_dla'] = dla()


# ---------------------------------------------------------------- 8. virtual weights on a 3-lane stream
def virtual():
    D = T['D']
    b = [text(24, 24, 'A writer and two readers, connected only through the stream', 't-title')]
    lanes = [130, 170, 210]
    for i, y in enumerate(lanes):
        b += [f'<rect class="{"box-3" if i == 2 else "box"}" x="150" y="{y - 13}" width="460" height="26" rx="13"/>', text(140, y + 4, f'dimension {i + 1}', 't-tick', 'end')]
    b.append(text(150, 100, 'residual stream (3 numbers) →', 't-muted'))
    b += token(450, 50, 220, 'reader B: W_I = [1, 0, 0]', 'box', 30)
    b += [arrow(560, 117, 560, 82), text(548, 106, 'reads dim 1: gets 0', 't-tick', 'end')]
    b += token(200, 270, 200, 'writer: W_O = [0, 0, 1]ᵀ', 'box-2', 30)
    b += [arrow(300, 268, 300, 225), text(310, 252, f'adds {D["a"]:.1f} to dim 3', 't-tick'), text(300, 320, f'sends a = {D["a"]:.1f}', 't-tick', 'middle')]
    b += token(450, 270, 220, 'reader A: W_I = [0, 0, 2]', 'box-1', 30)
    b += [arrow(560, 225, 560, 268), text(570, 252, f'reads 2 × {D["a"]:.1f} = {D["read_a"]:.1f}', 't-note')]
    b += [text(24, 362, f'virtual weight, writer → A:  W_I W_O = [0, 0, 2] · [0, 0, 1]ᵀ = {D["va"]:.0f}', 't-tick'),
          text(24, 386, f'virtual weight, writer → B:  W_I W_O = [1, 0, 0] · [0, 0, 1]ᵀ = {D["vb"]:.0f}   (they never talk)', 't-tick')]
    return svg(760, 400, 'A residual stream with three dimensions drawn as three lanes. A writer with W_O = (0, 0, 1) adds 0.7 into dimension 3. Reader A with W_I = (0, 0, 2) reads dimension 3 and gets 1.4: their virtual weight is 2. Reader B with W_I = (1, 0, 0) reads dimension 1 and gets 0: their virtual weight is 0.', b)


F['p1_virtual'] = virtual()


# ---------------------------------------------------------------- 9. bandwidth
def bandwidth():
    B = R['bandwidth']
    items = [('attn-only-2l', 'residual stream', B['attn-only-2l']['d_model'], 's1'),
             ('attn-only-2l', 'all head outputs', B['attn-only-2l']['head_dims'], 's2'),
             ('gpt2 (small)', 'residual stream', B['gpt2']['d_model'], 's1'),
             ('gpt2 (small)', 'all head outputs', B['gpt2']['head_dims'], 's2'),
             ('gpt2 (small)', 'all MLP neurons', B['gpt2']['neurons'], 's3')]
    vmax = 37000
    x0, bw, rh, top = 250, 420, 34, 56
    b = [text(24, 26, 'Many writers, one narrow stream (dimensions)', 't-title')]
    for i, (m, lab, v, cls) in enumerate(items):
        y = top + i * rh + (16 if i >= 2 else 0)
        if i in (0, 2):
            b.append(text(24, y + 15, m, 't-note'))
        b.append(text(x0 - 10, y + 17, lab, 't-tick', 'end'))
        w = max(3, bw * v / vmax)
        b.append(f'<g class="mark"><title>{esc(m)}, {esc(lab)}: {v:,}</title><rect class="{cls}" x="{x0}" y="{y + 4}" width="{w:.1f}" height="{rh - 12}" rx="3"/></g>')
        b.append(text(x0 + w + 8, y + 19, f'{v:,}', 't-val'))
    y = top + 5 * rh + 30
    b.append(text(24, y, f'gpt2: {B["gpt2"]["head_dims"] + B["gpt2"]["neurons"]:,} computed dimensions share {B["gpt2"]["d_model"]} stream dimensions, '
                  f'{(B["gpt2"]["head_dims"] + B["gpt2"]["neurons"]) / B["gpt2"]["d_model"]:.0f} times more.', 't-tick'))
    return svg(760, y + 14, 'Bar chart. attn-only-2l: a residual stream of 512 dimensions and 1,024 head output dimensions. gpt2 small: a residual stream of 768 dimensions, 9,216 head output dimensions and 36,864 MLP neurons, 60 times more than the stream.', b)


F['p1_bandwidth'] = bandwidth()


# ---------------------------------------------------------------- 10. the toy bigram table W_U W_E
def bigram():
    Tb = T['B']['table']
    V = ['the', 'cat', 'dog', 'sat']
    P = T['B']['probs']
    b = [text(24, 24, 'The zero-layer model is one table: W_U W_E', 't-title')]
    x0, y0, c = 120, 76, 46
    b.append(text(x0 + 2 * c, y0 - 34, 'current token (column)', 't-muted', 'middle'))
    for j, w in enumerate(V):
        b.append(text(x0 + j * c + c / 2, y0 - 10, w, 't-note' if j == 0 else 't-tick', 'middle'))
    b.append(text(x0 - 64, y0 + 2 * c + 4, 'next', 't-muted', 'end'))
    b.append(text(x0 - 64, y0 + 2 * c + 20, 'token', 't-muted', 'end'))
    b.append(text(x0 - 64, y0 + 2 * c + 36, '(row)', 't-muted', 'end'))
    for i, w in enumerate(V):
        b.append(text(x0 - 10, y0 + i * c + c / 2 + 4, w, 't-tick', 'end'))
        for j in range(4):
            v = Tb[i][j]
            b.append(f'<g class="mark"><title>after {V[j]}, logit of {w}: {v:g}</title><rect class="cell" x="{x0 + j * c + 2}" y="{y0 + i * c + 2}" width="{c - 4}" height="{c - 4}" rx="5" style="fill-opacity:{0.08 + 0.4 * v / 2:.2f}"/></g>')
            b.append(text(x0 + j * c + c / 2, y0 + i * c + c / 2 + 4, f'{v:g}', 't-val', 'middle'))
    b.append(f'<rect class="box-on" x="{x0}" y="{y0}" width="{c}" height="{4 * c}" rx="6" style="fill:none"/>')
    b.append(text(x0 + 2 * c, y0 + 4 * c + 26, 'the "the" column = logits after "the"', 't-tick', 'middle'))
    # softmax bars
    bx = 470
    b.append(text(bx, y0 - 10, 'softmax of that column', 't-note'))
    for i, (w, p) in enumerate(zip(V, P)):
        y = y0 + i * c + 10
        b.append(text(bx + 30, y + 17, w, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>P({w} | the) = {p:.3f}</title><rect class="s1" x="{bx + 40}" y="{y + 4}" width="{200 * p:.1f}" height="20" rx="3"/></g>')
        b.append(text(bx + 48 + 200 * p, y + 19, f'{p:.3f}', 't-val'))
    b.append(text(bx, y0 + 4 * c + 26, 'probabilities of the next token', 't-tick'))
    return svg(760, 310, 'A 4 by 4 table of logits W_U W_E for the toy model, rows are the next token and columns the current token. The column for "the" is 0, 2, 1, 0; its softmax gives the 0.083, cat 0.610, dog 0.225, sat 0.083.', b)


F['p1_bigram'] = bigram()


# ---------------------------------------------------------------- 11. the direct path of the real one-layer model
def direct():
    D = R['direct']['attn-only-1l']
    words = [' Barack', ' United', ' New', ' Hong', ' Mr', ' according', ' Los']
    b = [text(24, 24, 'attn-only-1l: the top 3 rows of W_U W_E for a few tokens', 't-title'),
         text(24, 44, 'no context at all: just "this token, then which token?"  (a leading dash marks a piece glued to the word before)', 't-tick')]
    for i, w in enumerate(words):
        y = 66 + i * 40
        b += token(24, y, 120, w.strip(), 'box-1', 30)
        b += [arrow(150, y + 15, 196, y + 15)]
        for k, (t, _) in enumerate(D[w]['top'][:3]):
            lab = t.strip() if t.startswith(' ') else '-' + t
            b += token(204 + k * 130, y, 120, lab, 'box-3' if k == 0 else 'box', 30)
        b.append(text(604, y + 20, f'itself: rank {D[w]["self_rank"]:,}', 't-muted'))
    return svg(760, 66 + len(words) * 40 + 6, 'For each of seven tokens, the three next tokens the direct path of attn-only-1l scores highest: Barack to Obama, Hussein, -lung; United to States, Nations, Kingdom; New to Zealand, York, Yorker; Hong to Kong, -wei, -chen; Mr to Corbyn, -unal, Modi; according to to, specific, logger; Los to Angeles, -artan, Santos.', b)


F['p1_direct'] = direct()

json.dump(F, open('results/figs_part1.json', 'w'))
print('figures:', len(F), ', '.join(F))
