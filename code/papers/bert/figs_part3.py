"""Figures for Part 3 (pre-training), drawn from the results/part3_*.json files. Writes results/figs_part3.json."""
import json, math
from figlib import svg, text, box, arrow, esc
from bertfig import token, row, line, hbars

M = json.load(open('results/part3_mlm.json'))
N = json.load(open('results/part3_nsp.json'))
S = json.load(open('results/part3_schedule.json'))
E = json.load(open('results/part3_seeitself.json'))
F = {}


def polyline(pts, cls):
    return f'<polyline class="{cls}" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts) + '"/>'


def axes(L, T, W, H, xt, yt, X, Y, xfmt, yfmt):
    b = []
    for v in yt:
        b += [f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + W}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, yfmt(v), 't-tick', 'end')]
    for v in xt:
        b.append(text(X(v), T + H + 18, xfmt(v), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + H}" x2="{L + W}" y2="{T + H}"/>')
    return b


# ---------------------------------------------------------------- why not just look both ways
def see_itself():
    words = ['the', 'cat', 'sat', 'down']
    xs = [200 + i * 130 for i in range(4)]
    b = [text(20, 24, 'Predict "cat" (position 2) from every other word, with two layers', 't-title')]
    ys = {'in': 250, 'l1': 160, 'l2': 70}
    b += [text(20, ys['in'] + 20, 'input tokens', 't-tick'), text(20, ys['l1'] + 20, 'layer 1', 't-tick'), text(20, ys['l2'] + 20, 'layer 2', 't-tick')]
    for i, (x, w) in enumerate(zip(xs, words)):
        b += token(x - 40, ys['in'], 80, w, 'box-mask' if i == 1 else 'box', 32, 't-note')
        b += [box(x - 40, ys['l1'], 80, 32, 'box', 8), box(x - 40, ys['l2'], 80, 32, 'box-on' if i == 1 else 'box', 8)]
    b.append(text(xs[1], ys['l2'] + 21, 'predict ?', 't-tick', 'middle'))
    # layer 1: position 3 ("sat") reads "cat" from the input: allowed, it is not position 2
    b.append(line(xs[1] + 10, ys['in'], xs[2] - 10, ys['l1'] + 32, 'bad-line'))
    b.append(text(xs[2] + 46, ys['l1'] + 56, 'step 1: "sat" reads "cat"', 't-bad'))
    b.append(text(xs[2] + 46, ys['l1'] + 72, '(allowed: it is not its own word)', 't-muted'))
    # layer 2: position 2 reads position 3, which now contains "cat"
    b.append(line(xs[2] - 10, ys['l1'], xs[1] + 10, ys['l2'] + 32, 'bad-line'))
    b.append(text(xs[2] + 46, ys['l2'] + 46, 'step 2: position 2 reads "sat",', 't-bad'))
    b.append(text(xs[2] + 46, ys['l2'] + 62, 'which already contains "cat"', 't-bad'))
    # honest links (grey)
    for j in (0, 3):
        b.append(line(xs[j], ys['in'], xs[1], ys['l1'] + 32, 'edge-dim'))
    b.append(text(xs[1], ys['in'] + 50, 'the answer leaks back in through a neighbour', 't-muted', 'middle'))
    return svg(760, 310, 'Why a deep model cannot simply condition on both sides. Even if position 2 never looks at its own word directly, in layer 1 a neighbour reads the word, and in layer 2 position 2 reads the neighbour. With two or more layers, every word can indirectly see itself.', b)


F['p3_see_itself'] = see_itself()


def see_chart():
    runs = E['runs']
    names = list(runs)
    W, H, L, T = 470, 230, 64, 30
    smax = runs[names[0]]['curve'][-1][0]
    X = lambda s: L + s / smax * W
    Y = lambda v: T + H - v / 10 * H
    b = axes(L, T, W, H, [0, smax // 4, smax // 2, 3 * smax // 4, smax], [0, 2, 4, 6, 8, 10], X, Y, lambda v: f'{v:,}', lambda v: f'{v}')
    cls = [('l2', 's2'), ('l3', 's3'), ('l1', 's1')]
    ends = []
    for (lc, dc), n in zip(cls, names):
        pts = [(X(s), Y(v)) for s, v in runs[n]['curve']]
        b.append(polyline(pts, lc))
        ends.append([pts[-1][1], n, dc, runs[n]])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 34)
    for y, n, dc, r in ends:
        b += [f'<rect class="{dc}" x="{L + W + 14}" y="{y - 6:.1f}" width="10" height="10" rx="2"/>',
              text(L + W + 30, y + 3, n.split(': ')[1], 't-note'),
              text(L + W + 30, y + 19, f'train {r["train_last"]:.2f}, unseen text {r["valid"]:.2f}', 't-tick')]
    b += [text(L + W / 2, T + H + 40, 'training step', 't-tick', 'middle'), text(L - 50, T - 12, 'loss (lower is better)', 't-tick')]
    return svg(760, 310, 'Training loss of the three tiny models. The left-to-right model and the one-layer both-sides model learn slowly and honestly. The two-layer both-sides model drops far lower, on unseen text too, because it reads its own answer back.', b)


F['p3_see_chart'] = see_chart()


# ---------------------------------------------------------------- the masked LM
def mlm():
    words = ['[CLS]', 'my', 'dog', 'is', '[MASK]', '.', '[SEP]']
    cls = ['box-4', 'box', 'box', 'box', 'box-mask', 'box', 'box-4']
    parts, cx = row(40, 250, words, 84, 10, classes=cls)
    b = [text(20, 24, 'Masked LM: predict only the chosen positions', 't-title')] + parts
    b += [box(40, 160, 6 * 94 + 84, 50, 'box-1', 12), text(40 + (6 * 94 + 84) / 2, 190, 'BERT: 12 layers, every token sees every token', 't-note', 'middle')]
    for x in cx:
        b.append(arrow(x, 248, x, 212))
    b += [arrow(cx[4], 158, cx[4], 124), box(cx[4] - 40, 92, 80, 30, 'box-on', 8), text(cx[4], 112, 'T₄', 't-math', 'middle'),
          arrow(cx[4] + 42, 107, 530, 86), box(534, 40, 200, 92, 'box', 10), text(634, 62, 'softmax over all', 't-tick', 'middle'),
          text(634, 80, '30,522 vocabulary ids', 't-tick', 'middle'), text(634, 106, 'cross-entropy against', 't-muted', 'middle'),
          text(634, 122, 'the original word', 't-muted', 'middle')]
    for x in (cx[1], cx[2], cx[3], cx[5]):
        b.append(text(x, 150, 'no loss', 't-muted', 'middle'))
    return svg(760, 300, 'The masked language model. The whole sentence goes in, with the chosen token hidden. Only the output vectors of the chosen positions (here T4) go through a softmax over the vocabulary, and only they count in the loss.', b)


F['p3_mlm'] = mlm()


def eighty():
    b = [text(20, 24, 'For each chosen position (15% of tokens), roll the dice once', 't-title')]
    b += token(30, 120, 150, 'hairy (chosen)', 'box-on', 36, 't-note')
    rows = [('80%', 'replace with [MASK]', 'my dog is [MASK]', 'box-mask', 'learn to fill blanks'),
            ('10%', 'replace with a random word', 'my dog is apple', 'box-2', 'never fully trust the input'),
            ('10%', 'keep the word unchanged', 'my dog is hairy', 'box-3', 'learn about real, unmasked words')]
    for i, (pct, what, ex, cls, why) in enumerate(rows):
        y = 50 + i * 74
        b += [arrow(182, 138, 236, y + 22), text(244, y + 14, pct, 't-val'), text(286, y + 14, what, 't-note'),
              box(244, y + 22, 190, 30, cls, 6), text(339, y + 42, ex, 't-mono', 'middle'), text(452, y + 42, why, 't-tick')]
    b.append(text(30, 270, 'In all three cases the target is the same: predict "hairy" at position 4.', 't-tick'))
    return svg(760, 290, 'The 80/10/10 rule of Section 3.1 and Appendix A.1. A chosen token becomes [MASK] 80% of the time, a random word 10% of the time, and stays itself 10% of the time. The model must predict the original word in every case.', b)


F['p3_8010'] = eighty()


def stats():
    s = M['stats']
    items = [('chosen / all tokens', s['chosen_share'] * 100, 15), ('[MASK] / chosen', s['mask_share'] * 100, 80),
             ('random / chosen', s['random_share'] * 100, 10), ('unchanged / chosen', s['same_share'] * 100, 10),
             ('random / all tokens', s['random_of_all'] * 100, 1.5)]
    b = [text(20, 24, f'Measured over {M["counts"]["tokens"]:,} word pieces of real Wikipedia text (seed 0)', 't-title')]
    L, W = 170, 480
    for i, (lab, v, target) in enumerate(items):
        y = 46 + i * 40
        w = W * v / 100
        b += [text(L - 10, y + 17, lab, 't-tick', 'end'),
              f'<g class="mark"><title>{esc(lab)}: {v:.2f}% (paper: {target}%)</title><rect class="s1" x="{L}" y="{y + 4}" width="{max(w, 2):.1f}" height="20" rx="3"/></g>',
              f'<line class="base-line" x1="{L + W * target / 100:.1f}" y1="{y}" x2="{L + W * target / 100:.1f}" y2="{y + 28}"/>',
              text(L + max(w, 2) + 8, y + 18, f'{v:.2f}%', 't-val'), text(L + max(w, 2) + 70, y + 18, f'paper: {target}%', 't-tick')]
    return svg(760, 260, 'The masking procedure, measured. Every share lands within 0.1 points of the paper: 15% of tokens are chosen; of those, 80% become [MASK], 10% a random token and 10% stay the same; so 1.5% of all tokens are random replacements. The dashed lines mark the paper\'s numbers.', b)


F['p3_stats'] = stats()


def accuracy():
    a = M['accuracy']
    items = [('[MASK]', a['by_case']['mask']), ('random token', a['by_case']['random']), ('unchanged', a['by_case']['same']), ('all chosen', a['all'])]
    b = [text(20, 24, f'bert-base-uncased, top-1 accuracy on {a["n"]:,} chosen positions', 't-title')]
    b += hbars(20, 50, [(k, v * 100) for k, v in items], 440, 34, 's1', 140, lambda v: f'{v:.1f}%', 100)
    return svg(760, 200, 'How often the real model\'s top guess is the original word, split by what the chosen position showed. It recovers words hidden by [MASK] most of the time, nearly half of the words replaced by a random token, and almost all unchanged words.', b)


F['p3_accuracy'] = accuracy()


# ---------------------------------------------------------------- next sentence prediction
def nsp():
    b = [text(20, 24, 'Building one next-sentence example', 't-title')]
    b += [box(20, 44, 200, 120, 'box', 10), text(120, 64, 'document 1', 't-tick', 'middle'),
          box(34, 76, 172, 26, 'box-1', 5), text(120, 94, 'sentence A', 't-tick', 'middle'),
          box(34, 108, 172, 26, 'box-3', 5), text(120, 126, 'the real next sentence', 't-tick', 'middle'),
          box(20, 186, 200, 70, 'box', 10), text(120, 206, 'document 2 (random)', 't-tick', 'middle'),
          box(34, 218, 172, 26, 'box-2', 5), text(120, 236, 'some other sentence', 't-tick', 'middle'),
          line(208, 121, 236, 146, 'edge-3'), text(226, 112, '50%', 't-val', 'middle'), line(208, 231, 236, 166, 'edge-2'), text(226, 226, '50%', 't-val', 'middle'),
          box(236, 140, 56, 32, 'box-on', 6), text(264, 160, 'pick B', 't-tick', 'middle'), arrow(294, 140, 304, 134)]
    parts, cx = row(306, 116, ['[CLS]', 'A', '[SEP]', 'B', '[SEP]'], 56, 6, classes=['box-4', 'box-1', 'box-4', 'box', 'box-4'])
    b += parts + [box(306, 186, 302, 40, 'box-1', 10), text(457, 211, 'BERT', 't-note', 'middle')]
    for x in cx:
        b.append(arrow(x, 148, x, 184))
    b += [arrow(cx[0], 228, cx[0], 262), box(cx[0] - 22, 264, 44, 26, 'box-on', 6), text(cx[0], 282, 'C', 't-math', 'middle'),
          arrow(cx[0] + 24, 277, 420, 277), box(424, 262, 300, 30, 'box', 8), text(574, 282, 'IsNext or NotNext? (2 classes)', 't-tick', 'middle')]
    return svg(760, 310, 'How next sentence prediction examples are made. Sentence A comes from a document; half of the time B is the real next sentence (IsNext), half of the time a sentence from a random document (NotNext). The final vector C of the [CLS] token makes the two-way decision.', b)


F['p3_nsp'] = nsp()


def cls_cos():
    short = ['related: guitar on stage / musician', 'related: stocks fell / prices dropped', 'unrelated: guitar / stock market',
             'unrelated: penguins / invoice', 'opposite: loved it / hated it']
    items = [(lab, d['cos']) for lab, d in zip(short, N['cls_cosine'])]
    b = [text(20, 24, 'Cosine similarity of raw [CLS] vectors (1 = same direction)', 't-title')]
    b += hbars(20, 48, items, 330, 34, 's2', 300, lambda v: f'{v:.3f}', 1.0)
    return svg(760, 240, 'Footnote 6, measured. Without fine-tuning, the [CLS] vectors of any two sentences point in nearly the same direction. "I loved this movie" and "I hated this movie" score the highest similarity of all.', b)


F['p3_cls_cos'] = cls_cos()


# ---------------------------------------------------------------- the training recipe
def lr():
    pts = S['lr_curve']
    b = [text(20, 24, 'Learning rate over pre-training', 't-title'), text(420, 24, 'Zoom: the first 20,000 steps', 't-title')]
    for (L, W, smax, ticks, fmt) in ((60, 300, 1_000_000, [0, 250_000, 500_000, 750_000, 1_000_000], lambda v: f'{v // 1000:,}k'),
                                     (460, 260, 20_000, [0, 5_000, 10_000, 15_000, 20_000], lambda v: f'{v // 1000:,}k')):
        T, H = 44, 170
        X = lambda s, L=L, W=W, smax=smax: L + s / smax * W
        Y = lambda v, T=T, H=H: T + H - v / 1e-4 * H
        b += axes(L, T, W, H, ticks, [0, 2.5e-5, 5e-5, 7.5e-5, 1e-4], X, Y, fmt, lambda v: f'{v:.1e}'.replace('e-05', 'e-5').replace('e-04', 'e-4').replace('0.0e+00', '0'))
        b.append(polyline([(X(s), Y(v)) for s, v in pts if s <= smax], 'l1'))
    b += [text(610, 150, 'warmup: 0 → 1e-4', 't-tick'), text(610, 166, 'over 10,000 steps', 't-tick'), text(200, 270, 'step', 't-tick', 'middle'), text(590, 270, 'step', 't-tick', 'middle')]
    return svg(760, 285, 'The learning-rate schedule of Appendix A.2, as the released code computes it: it climbs linearly from 0 over the first 10,000 steps, then falls linearly to 0 at step 1,000,000.', b)


F['p3_lr'] = lr()


def gelu():
    g = S['gelu']
    L, T, W, H = 70, 40, 420, 220
    X = lambda v: L + (v + 4) / 8 * W
    Y = lambda v: T + H - (v + 1) / 5 * H
    b = [text(20, 24, 'Two activation functions', 't-title')]
    b += axes(L, T, W, H, [-4, -2, 0, 2, 4], [-1, 0, 1, 2, 3, 4], X, Y, lambda v: f'{v}', lambda v: f'{v}')
    b.append(polyline([(X(x), Y(y)) for x, y in zip(g['x'], g['relu'])], 'l2'))
    b.append(polyline([(X(x), Y(y)) for x, y in zip(g['x'], g['gelu'])], 'l1'))
    b += [f'<rect class="s2" x="{L + W + 24}" y="70" width="10" height="10" rx="2"/>', text(L + W + 40, 79, 'ReLU: max(0, x)', 't-note'),
          text(L + W + 40, 96, 'a sharp corner at 0', 't-tick'),
          f'<rect class="s1" x="{L + W + 24}" y="126" width="10" height="10" rx="2"/>', text(L + W + 40, 135, 'GELU: x · Φ(x)', 't-note'),
          text(L + W + 40, 152, 'smooth; slightly negative', 't-tick'), text(L + W + 40, 168, f'for x < 0 (lowest {g["min"]:.2f})', 't-tick'),
          text(L + W / 2, T + H + 40, 'input x', 't-tick', 'middle')]
    return svg(760, 310, 'ReLU and GELU. For large positive inputs they agree. GELU bends smoothly through zero and lets small negative inputs pass as small negative outputs, instead of cutting them to exactly 0.', b)


F['p3_gelu'] = gelu()

json.dump(F, open('results/figs_part3.json', 'w'))
print('figures:', ', '.join(F))
