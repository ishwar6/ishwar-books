"""Figures for Part 3 (pre-training), drawn from the results/part3_*.json files. Writes results/figs_part3.json."""
import json, math
from figlib import svg, text, box, arrow, esc
from bertfig import token, row, line, hbars
from alammar import vec, matrix, frame, brace

M = json.load(open('results/part3_mlm.json'))
N = json.load(open('results/part3_nsp.json'))
S = json.load(open('results/part3_schedule.json'))
E = json.load(open('results/part3_seeitself.json'))
Q = json.load(open('results/part3_math.json'))
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

# ================================================================ illustrated figures (v2)

def dae_vs_mlm():
    """Denoising auto-encoder (rebuild everything) next to the masked LM (predict only the hidden positions)."""
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat']
    W, G, x0 = 78, 10, 190
    cx = [x0 + i * (W + G) + W / 2 for i in range(6)]
    b = []
    # panel 1: denoising auto-encoder
    y = 20
    b += [text(20, y + 16, 'Denoising auto-encoder', 't-title'), text(20, y + 34, '(Vincent et al., 2008)', 't-muted')]
    noisy = ['the', '·', 'sat', 'on', '·', 'mat']
    for i, w in enumerate(noisy):
        b += token(cx[i] - W / 2, y + 112, W, 'dropped' if w == '·' else w, 'box-ghost' if w == '·' else 'box', 30, 't-muted' if w == '·' else 't-tick')
    b += [text(20, y + 132, 'damaged input', 't-tick'),
          box(x0, y + 62, 6 * (W + G) - G, 36, 'box-1', 10), text(x0 + (6 * (W + G) - G) / 2, y + 85, 'encoder, then decoder', 't-note', 'middle')]
    for i, w in enumerate(words):
        b += [arrow(cx[i], y + 110, cx[i], y + 100), arrow(cx[i], y + 60, cx[i], y + 46)]
        b += token(cx[i] - W / 2, y + 14, W, w, 'box-3', 30, 't-tick')
    b += [text(20, y + 70, 'rebuild', 't-tick'), text(20, y + 86, 'ALL 6 tokens', 't-tick t-strong')]
    for i in range(6):
        b.append(text(cx[i], y + 8, 'loss', 't-s3', 'middle'))
    # panel 2: masked LM
    y = 200
    b += [line(20, y - 14, 740, y - 14, 'edge-dim'), text(20, y + 16, 'Masked LM (BERT)', 't-title'), text(20, y + 34, '(Devlin et al., 2018)', 't-muted')]
    inp = ['the', '[MASK]', 'sat', 'on', 'the', 'mat']
    for i, w in enumerate(inp):
        b += token(cx[i] - W / 2, y + 112, W, w, 'box-mask' if w == '[MASK]' else 'box', 30, 't-tick')
    b += [text(20, y + 132, 'masked input', 't-tick'),
          box(x0, y + 62, 6 * (W + G) - G, 36, 'box-1', 10), text(x0 + (6 * (W + G) - G) / 2, y + 85, 'encoder only (12 layers)', 't-note', 'middle')]
    for i, w in enumerate(words):
        b.append(arrow(cx[i], y + 110, cx[i], y + 100))
        if i == 1:
            b += [arrow(cx[i], y + 60, cx[i], y + 46)] + token(cx[i] - W / 2, y + 14, W, 'cat', 'box-on', 30, 't-note')
            b.append(text(cx[i], y + 8, 'loss', 't-on', 'middle'))
        else:
            b.append(text(cx[i], y + 38, 'no loss', 't-muted', 'middle'))
    b += [text(20, y + 70, 'predict only', 't-tick'), text(20, y + 86, 'the hidden one', 't-tick t-strong')]
    return svg(760, 360, 'Two ways to learn from damaged text. A denoising auto-encoder damages the input and is trained to rebuild every token, so all six positions have a loss. The masked LM hides some tokens and is trained to predict only those, so only the hidden position has a loss.', b)


F['p3_dae_vs_mlm'] = dae_vs_mlm()


def pipeline():
    """The masking procedure as four frames on one 20-token sequence (one possible roll of the dice)."""
    words = ['the', 'man', 'went', 'to', 'the', 'store', 'to', 'buy', 'milk', 'and',
             'he', 'came', 'home', 'with', 'a', 'big', 'bag', 'of', 'bread', '.']
    ch = {5: ('[MASK]', 'box-mask'), 8: ('piano', 'box-2'), 16: ('bag', 'box-3')}
    W, G, x0 = 62, 8, 34
    titles = ['Choose 15% of the positions: 3 of these 20 tokens',
              '80% of the chosen: replace with [MASK]   (here "store")',
              '10% of the chosen: replace with a random token   (here "milk" → "piano")',
              '10% of the chosen: keep it unchanged   (here "bag"); predict the 3 originals']
    b = []
    FH, RS = 140, 48
    for f in range(4):
        y0 = 10 + f * (FH + 12)
        b += frame(20, y0, 720, FH, f + 1, titles[f])
        for i, w in enumerate(words):
            r, c = divmod(i, 10)
            x, y = x0 + c * (W + G), y0 + 40 + r * RS
            cls, s, tc = 'box', w, 't-tick'
            if i in ch:
                if f == 0:
                    cls = 'box-on'
                else:
                    stage = {5: 1, 8: 2, 16: 3}[i]
                    if f >= stage:
                        s, cls = ch[i]
                        tc = 't-note'
                    else:
                        cls = 'box-on'
            b += token(x, y, W, s, cls, 30, tc)
        if f == 3:
            for i, tgt in ((5, 'store'), (8, 'milk'), (16, 'bag')):
                r, c = divmod(i, 10)
                x = x0 + c * (W + G) + W / 2
                yy = y0 + 40 + r * RS
                b.append(text(x, yy + 44, f'→ {tgt}', 't-on', 'middle'))
    b.append(text(380, 4 * (FH + 12) + 18, 'Each chosen token rolls its own dice, so a real sequence can have any mix. The other 17 tokens have no loss.', 't-muted', 'middle'))
    return svg(760, 4 * (FH + 12) + 30, 'The masking procedure in four frames. Frame 1: choose 15% of positions, here 3 of 20 tokens. Frame 2: 80% of chosen tokens become [MASK]. Frame 3: 10% become a random token. Frame 4: 10% stay unchanged. The model must then predict the original word at all three chosen positions, and only there.', b)


F['p3_pipeline'] = pipeline()


def budget():
    """200 tokens as a unit chart: 170 untouched, 24 [MASK], 3 random, 3 unchanged but predicted."""
    kinds = [('untouched', 170, 'box'), ('[MASK]', 24, 's-mask'), ('random', 3, 's2'), ('same', 3, 's3')]
    cells = [k for k, n, c in kinds for _ in range(n)]
    cls = {k: c for k, n, c in kinds}
    C, x0, y0 = 30, 80, 40
    b = [text(20, 24, 'Every 200 tokens of training text, on average', 't-title')]
    for idx, k in enumerate(cells):
        r, c = divmod(idx, 20)
        x, y = x0 + c * C, y0 + r * C
        if cls[k] == 'box':
            b.append(box(x + 2, y + 2, C - 4, C - 4, 'box', 4))
        else:
            b.append(f'<rect class="{cls[k]}" x="{x + 2}" y="{y + 2}" width="{C - 4}" height="{C - 4}" rx="4"/>')
    gy = y0 + 10 * C + 26
    leg = [('box', '85% untouched', '170 of 200: no loss, look original'),
           ('s-mask', '12% [MASK]', '15% × 80% = 24 of 200'),
           ('s2', '1.5% random token', '15% × 10% = 3 of 200'),
           ('s3', '1.5% unchanged, predicted', '15% × 10% = 3 of 200')]
    for i, (c, a, n) in enumerate(leg):
        x = 30 + (i % 2) * 370
        y = gy + (i // 2) * 44
        b.append(box(x, y, 18, 18, 'box', 4) if c == 'box' else f'<rect class="{c}" x="{x}" y="{y}" width="18" height="18" rx="4"/>')
        b += [text(x + 28, y + 13, a, 't-note'), text(x + 28, y + 30, n, 't-tick')]
    yb = gy + 100
    b += [text(30, yb, 'carry a loss: 12 + 1.5 + 1.5 = 15%', 't-on'),
          text(400, yb, 'look like the real text: 85 + 1.5 = 86.5%', 't-on')]
    return svg(760, yb + 16, 'The 80/10/10 rule as a share of all tokens, drawn as 200 squares. 170 are never touched and have no loss. 24 become [MASK], 3 become a random token and 3 stay unchanged but must still be predicted. So 15% of tokens carry a loss, and 86.5% look like the original text.', b)


F['p3_budget'] = budget()


def mlm_head():
    """Shapes of the masked-LM head, with the real numbers of one prediction."""
    m = Q['mlm']
    b = [text(20, 24, 'From Tᵢ to a probability for every token of the vocabulary', 't-title')]
    top = 62
    b += vec(30, top, m['T4'], 's1', cell=12, gap=2, vertical=True, title='T')
    b += [text(36, top - 14, 'Tᵢ', 't-math', 'middle'), text(36, top + 186, '768', 't-tick', 'middle'),
          arrow(50, top + 82, 72, top + 82),
          box(74, top + 40, 112, 84, 'box-1', 10), text(130, top + 64, 'dense 768 × 768', 't-tick', 'middle'),
          text(130, top + 82, '+ GELU', 't-tick', 'middle'), text(130, top + 100, '+ LayerNorm', 't-tick', 'middle'),
          arrow(188, top + 82, 206, top + 82)]
    b += vec(212, top, [0.3, 0.8, 0.5, 0.9, 0.2, 0.6, 0.7, 0.4, 0.9, 0.3, 0.5, 0.8], 's1', cell=12, gap=2, vertical=True)
    b += [text(218, top - 14, 'h', 't-math', 'middle'), text(218, top + 186, '768', 't-tick', 'middle'),
          text(244, top + 87, '×', 't-big', 'middle')]
    mat, MW, MH = matrix(262, top + 30, 9, 16, 's4', cell=9, gap=2, label='Eᵀ', shape='768 × 30,522')
    b += mat
    b += [text(262 + MW / 2, top + 30 + MH + 38, 'the input token-embedding', 't-muted', 'middle'), text(262 + MW / 2, top + 30 + MH + 54, 'matrix, reused (tied)', 't-muted', 'middle'),
          text(262 + MW + 20, top + 87, '+ b', 't-math', 'middle')]
    zx = 262 + MW + 42
    zv = [0.2, 0.1, 1.0, 0.3, 0.86, 0.2, 0.85, 0.1, 0.83, 0.3, 0.8, 0.2]
    b += vec(zx, top, zv, 's2', cell=12, gap=2, vertical=True)
    b += [text(zx + 6, top - 14, 'z', 't-math', 'middle'), text(zx + 6, top + 186, '30,522', 't-tick', 'middle'),
          text(zx + 6, top + 202, 'scores', 't-muted', 'middle'), arrow(zx + 20, top + 82, zx + 62, top + 82), text(zx + 41, top + 72, 'softmax', 't-muted', 'middle')]
    tx = zx + 72
    b += [text(tx, top + 4, 'word', 't-tick t-strong'), text(tx + 128, top + 4, 'z', 't-tick t-strong', 'end'), text(tx + 184, top + 4, 'p', 't-tick t-strong', 'end')]
    for k, t in enumerate(m['top5']):
        y = top + 28 + k * 22
        on = t['word'] == m['target']
        b += [text(tx, y, t['word'], 't-on' if on else 't-tick'), text(tx + 128, y, f'{t["z"]:.2f}', 't-val', 'end'),
              text(tx + 184, y, f'{t["p"]:.3f}', 't-val', 'end')]
    b += [text(tx, top + 28 + 5 * 22, '30,517 more ...', 't-muted'),
          text(tx, top + 176, f'loss = −log {m["p_target"]:.3f}', 't-on'), text(tx, top + 194, f'= {m["loss"]:.3f}', 't-on')]
    return svg(760, 290, f'The masked-LM head on a real prediction. The 768 numbers of T at the masked position go through one dense layer with GELU and LayerNorm, then are multiplied by the transposed token-embedding matrix (768 by 30,522, shared with the input) and a bias is added, giving 30,522 scores. Softmax turns them into probabilities. For "the man went to the [MASK] to buy a gallon of milk", "store" gets {m["p_target"]:.3f}, so the loss is {m["loss"]:.3f}.', b)


F['p3_mlm_head'] = mlm_head()


def ce_curve():
    """-log p as a curve, with the real predictions of this part marked on it."""
    m, bt = Q['mlm'], Q['batch']
    L, T, W, H = 70, 60, 520, 210
    lp = lambda p: math.log10(p)
    X = lambda p: L + (lp(p) + 5) / 5 * W
    Y = lambda v: T + H - v / 12 * H
    b = [text(20, 24, 'Cross-entropy: the loss is −log of the probability given to the right word', 't-title')]
    b += axes(L, T, W, H, [], [0, 3, 6, 9, 12], X, Y, str, lambda v: f'{v}')
    for p, lab in ((1e-5, '0.00001'), (1e-4, '0.0001'), (1e-3, '0.001'), (1e-2, '0.01'), (1e-1, '0.1'), (1, '1')):
        b.append(text(X(p), T + H + 18, lab, 't-tick', 'middle'))
    pts = [(X(10 ** (-5 + 5 * k / 200)), Y(-math.log(10 ** (-5 + 5 * k / 200)))) for k in range(201)]
    b.append(polyline(pts, 'l1'))
    marks = [(1 / 30522, 'uniform guess', 's2'), (bt['per'][3]['p'], '"##s" (penguins)', 's2'), (bt['per'][2]['p'], '"went"', 's2'),
             (m['p_target'], '"store"', 's3'), (bt['per'][1]['p'], '"of"', 's3')]
    for p, lab, c in marks:
        v = -math.log(p)
        b.append(f'<g class="mark"><title>{esc(lab)}: p = {p:.4g}, loss {v:.3f}</title><circle class="{c} ring" cx="{X(p):.1f}" cy="{Y(v):.1f}" r="6"/></g>')
        if lab in ('"of"', '"store"'):
            ly = Y(v) - 46 if lab == '"store"' else Y(v) - 16
            b.append(line(X(p) + 4, Y(v) - 4, 604, ly - 4, 'edge-dim'))
            b.append(text(608, ly, f'{lab}: p {p:.4f}' if lab == '"of"' else f'{lab}: p {p:.3f}', 't-tick'))
            b.append(text(608, ly + 16, f'loss {v:.4f}' if lab == '"of"' else f'loss {v:.2f}', 't-tick'))
        else:
            b.append(text(X(p) + 10, Y(v) - 8, f'{lab}: p {p:.3g}, loss {v:.2f}', 't-tick'))
    b += [text(L + W / 2, T + H + 40, 'probability the model gave to the right word (log scale)', 't-tick', 'middle'), text(L - 50, T - 14, 'loss', 't-tick')]
    return svg(760, 320, 'The cross-entropy loss, minus the log of the probability given to the right word, with real predictions from this part. A sure, right answer costs almost nothing ("of": 0.0014). "store" at 0.474 costs 0.75. A uniform guess over the whole vocabulary costs 10.33.', b)


F['p3_ce_curve'] = ce_curve()


def nsp_head():
    """C -> pooler -> W (2 x 768) -> softmax, with the real numbers of the two A.1 examples."""
    ex = Q['nsp']
    b = [text(20, 24, 'Next sentence prediction reads one vector, C, and makes 2 scores', 't-title')]
    top = 56
    b += vec(30, top, ex[0]['C'], 's1', cell=12, gap=2, vertical=True)
    b += [text(36, top - 12, 'C', 't-math', 'middle'), text(36, top + 184, '768', 't-tick', 'middle'),
          arrow(52, top + 82, 80, top + 82)]
    mat, MW, MH = matrix(96, top + 70, 2, 13, 's3', cell=10, gap=3, label='W', shape='2 × 768')
    b += mat
    b += [text(96 + MW / 2, top + 140, 'C first goes through the', 't-muted', 'middle'), text(96 + MW / 2, top + 156, 'pooler: 768 × 768, tanh', 't-muted', 'middle'), text(96 + MW + 22, top + 88, '+ b', 't-math', 'middle'),
          arrow(96 + MW + 40, top + 82, 96 + MW + 64, top + 82)]
    x = 96 + MW + 72
    b += [text(x, top + 2, 'two scores', 't-tick t-strong'), text(x + 170, top + 2, 'softmax', 't-tick t-strong')]
    for k, e in enumerate(ex):
        y = top + 34 + k * 82
        lab = 'A.1 example 1: "the man went to [MASK] store" + "he bought a gallon [MASK] milk"' if k == 0 else 'A.1 example 2: "the man [MASK] to the store" + "penguin [MASK] are flightless birds"'
        b.append(text(x, y - 8, ('A.1 example 1, label IsNext' if k == 0 else 'A.1 example 2 (penguins), label NotNext'), 't-muted'))
        for j, name in enumerate(('IsNext', 'NotNext')):
            yy = y + j * 22
            pv = e['p'][j]
            b += [text(x, yy + 12, f'{name}  {e["scores"][j]:+.3f}', 't-tick'),
                  f'<g class="mark"><title>P({name}) = {pv:.6f}</title><rect class="{"s3" if j == 0 else "s2"}" x="{x + 170}" y="{yy + 2}" width="{max(2, 150 * pv):.1f}" height="14" rx="3"/></g>',
                  text(x + 170 + max(2, 150 * pv) + 6, yy + 13, f'{pv:.5f}', 't-val')]
    b.append(text(20, 280, 'Both examples: the scores come out exactly as the library computes them; the loss is −log of the right class (0.00009 and 0.0011).', 't-muted'))
    return svg(760, 296, f'The next-sentence head on the paper\'s two examples. The 768 numbers of C (after the pooler layer) are multiplied by W, which has 2 rows, and a bias is added: two scores. Softmax gives P(IsNext) 0.99991 for the first example and P(NotNext) {ex[1]["p"][1]:.5f} for the second.', b)


F['p3_nsp_head'] = nsp_head()


def loss_sum():
    """Mean MLM loss + mean NSP loss on the two A.1 examples as one batch."""
    bt = Q['batch']
    b = [text(20, 24, 'The training loss for one tiny batch (the two examples of Appendix A.1)', 't-title')]
    items = [(f'pair {p["pair"]}: "{p["word"]}"', p['loss']) for p in bt['per']]
    b += hbars(20, 60, items, 330, 30, 's2', 170, lambda v: f'{v:.4f}', 6.0, title='masked LM: one loss per hidden token')
    y = 60 + 4 * 30 + 8
    b += [line(190, y, 560, y, 'edge-dim'), text(180, y + 22, 'mean of 4', 't-tick t-strong', 'end'), text(190, y + 22, f'{bt["mlm_mean"]:.4f}', 't-val')]
    items2 = [('pair 1: IsNext', bt['nsp'][0]), ('pair 2: NotNext', bt['nsp'][1])]
    y2 = y + 66
    b += hbars(20, y2, items2, 330, 30, 's3', 170, lambda v: f'{v:.6f}', 6.0, title='next sentence: one loss per pair')
    y3 = y2 + 2 * 30 + 8
    b += [line(190, y3, 560, y3, 'edge-dim'), text(180, y3 + 22, 'mean of 2', 't-tick t-strong', 'end'), text(190, y3 + 22, f'{bt["nsp_mean"]:.6f}', 't-val'),
          box(20, y3 + 40, 720, 40, 'box-on', 10),
          text(380, y3 + 66, f'total = {bt["mlm_mean"]:.4f} + {bt["nsp_mean"]:.6f} = {bt["total"]:.4f}   (BertForPreTraining: {bt["lib"]:.4f})', 't-note', 'middle')]
    return svg(760, y3 + 92, f'The pre-training loss is the mean masked-LM loss plus the mean next-sentence loss. On the two Appendix A.1 examples: four hidden tokens with losses from 0.0014 to 5.7557 average to {bt["mlm_mean"]:.4f}; the two next-sentence losses average to {bt["nsp_mean"]:.6f}; the sum {bt["total"]:.4f} is exactly what the library computes.', b)


F['p3_loss_sum'] = loss_sum()


def doc_vs_shuffled():
    """Why a document-level corpus: next sentences and long spans only exist in whole documents."""
    b = [text(20, 24, 'Document-level corpus (BooksCorpus, Wikipedia)', 't-title'), text(400, 24, 'Shuffled sentences (Billion Word)', 't-title')]
    docs = [('box-1', 'document 1'), ('box-3', 'document 2')]
    y = 44
    for d, (c, name) in enumerate(docs):
        b.append(text(30, y + 14, name, 't-muted'))
        for k in range(4):
            b += token(30, y + 22 + k * 30, 300, f'sentence {k + 1} of {name}', c, 24, 't-tick')
        y += 22 + 4 * 30 + 12
    b += [box(22, 62, 316, 54, 'box-ghost', 8), text(345, 92, 'A, B: IsNext', 't-on')]
    b += [text(30, y + 12, 'real next sentences exist, and long spans', 't-tick'), text(30, y + 28, 'of up to 512 tokens stay on one topic', 't-tick')]
    order = [('box-1', 'doc 1', 2), ('box-3', 'doc 2', 4), ('box-4', 'doc 7', 1), ('box-1', 'doc 1', 4), ('box-2', 'doc 5', 3),
             ('box-3', 'doc 2', 1), ('box-4', 'doc 7', 3), ('box-2', 'doc 5', 1)]
    y2 = 66
    for k, (c, d, s) in enumerate(order):
        b += token(410, y2 + k * 30, 300, f'sentence {s} of {d}', c, 24, 't-tick')
    b += [box(402, y2 - 4, 316, 58, 'box-ghost', 8), text(722, y2 + 32, '?', 't-bad')]
    b += [text(410, y2 + 8 * 30 + 22, 'the "next" sentence is from another place:', 't-tick'),
          text(410, y2 + 8 * 30 + 38, 'no IsNext pairs, no long coherent spans', 't-tick')]
    return svg(760, 360, 'Why the corpus must keep whole documents. In a document-level corpus, sentence B can really be the next sentence of A, and long spans stay on one topic. In a shuffled sentence-level corpus such as the Billion Word Benchmark, neighbouring sentences come from different documents, so next sentence prediction and long contiguous sequences are impossible.', b)


F['p3_doc_vs_shuffled'] = doc_vs_shuffled()


def schedule():
    """128 tokens for 90% of the steps, then 512 for the last 10%."""
    L, W = 60, 640
    X = lambda s: L + s / 1_000_000 * W
    b = [text(20, 24, 'Pre-training schedule (Appendix A.2)', 't-title')]
    y = 60
    b += [box(X(0), y, X(900_000) - X(0), 46, 'box-1', 8), text((X(0) + X(900_000)) / 2, y + 20, 'sequences of 128 tokens', 't-note', 'middle'),
          text((X(0) + X(900_000)) / 2, y + 37, '900,000 steps (90%)', 't-tick', 'middle'),
          box(X(900_000), y, X(1_000_000) - X(900_000), 46, 'box-2', 8), text((X(900_000) + X(1_000_000)) / 2, y + 20, '512', 't-note', 'middle'),
          text((X(900_000) + X(1_000_000)) / 2, y + 37, '100,000', 't-tick', 'middle')]
    for s in (0, 250_000, 500_000, 750_000, 1_000_000):
        b.append(text(X(s), y + 66, f'{s // 1000:,}k', 't-tick', 'middle'))
    b.append(text(X(500_000), y + 84, 'training step', 't-muted', 'middle'))
    # which position embeddings learn when
    y = 190
    b += [text(20, y, 'Which position embeddings get trained', 't-title')]
    b += [text(L - 6, y + 32, '0-127', 't-tick', 'end'), box(X(0), y + 18, W, 22, 'box-3', 5), text(X(500_000), y + 34, 'trained the whole time', 't-tick', 'middle'),
          text(L - 6, y + 64, '128-511', 't-tick', 'end'), box(X(0), y + 50, X(900_000) - X(0), 22, 'box-ghost', 5),
          text(X(450_000), y + 66, 'never used yet: no training signal', 't-muted', 'middle'),
          box(X(900_000), y + 50, X(1_000_000) - X(900_000), 22, 'box-3', 5)]
    b.append(text(20, y + 100, 'The paper\'s reason for the last phase: "to learn the positional embeddings" of positions 128 to 511.', 't-muted'))
    return svg(760, 310, 'The pre-training schedule of Appendix A.2. The first 900,000 steps use sequences of 128 tokens; the last 100,000 steps use 512 tokens. Position embeddings 0 to 127 are trained all the time; positions 128 to 511 only appear, and so only learn, in the final 10% of steps.', b)


F['p3_schedule'] = schedule()


def attn_cost():
    """128 x 128 versus 512 x 512 attention scores, and the share of a layer's work."""
    c = Q['cost']
    b = [text(20, 24, 'One head, one layer: an n × n table of attention scores', 't-title')]
    s1, s4 = 44, 176
    x1, y1 = 40, 60 + s4 - s1
    b += [f'<rect class="cell" x="{x1}" y="{y1}" width="{s1}" height="{s1}" rx="3" style="fill-opacity:0.7"/>',
          text(x1 + s1 / 2, y1 + s1 + 20, '128 × 128', 't-note', 'middle'), text(x1 + s1 / 2, y1 + s1 + 38, f'{c["128"]["scores_per_head"]:,}', 't-tick', 'middle')]
    x4 = 140
    for i in range(4):
        for j in range(4):
            b.append(f'<rect class="cell" x="{x4 + j * s1}" y="{60 + i * s1}" width="{s1 - 2}" height="{s1 - 2}" rx="3" style="fill-opacity:{0.7 if i == 0 and j == 0 else 0.3}"/>')
    b += [text(x4 + s4 / 2, 60 + s4 + 20, '512 × 512', 't-note', 'middle'), text(x4 + s4 / 2, 60 + s4 + 38, f'{c["512"]["scores_per_head"]:,} = 16 ×', 't-tick', 'middle')]
    # bars: share of the layer's work
    bx = 380
    b += [text(bx, 66, 'Share of one layer\'s multiply-adds', 't-title'), text(bx, 84, 'spent on the n × n attention part (BERT-base)', 't-muted')]
    for k, n in enumerate(('128', '512')):
        y = 104 + k * 44
        sh = c[n]['share']
        b += [text(bx + 60, y + 16, f'n = {n}', 't-tick', 'end'),
              box(bx + 70, y + 2, 280, 20, 'box', 4),
              f'<rect class="s2" x="{bx + 70}" y="{y + 2}" width="{280 * sh:.1f}" height="20" rx="4"/>',
              text(bx + 70 + 280 * sh + 8, y + 17, f'{sh:.1%}', 't-val')]
    r = c['512']; q = c['128']
    seqx = (r['linear'] + r['attention']) / (q['linear'] + q['attention'])
    b += [text(bx, 214, f'whole sequence, 512 vs 128: {seqx:.2f}× the work for 4× the tokens', 't-tick'),
          text(bx, 232, f'per token: {r["per_token"] / q["per_token"]:.2f}× the work', 't-tick'),
          text(bx, 250, 'attention weights to store per layer: 16× as many', 't-tick')]
    return svg(760, 290, f'Attention is quadratic. One head in one layer fills a 128 by 128 table of scores (16,384 numbers) at length 128, but a 512 by 512 table (262,144 numbers, 16 times more) at length 512. In BERT-base the n-squared part is {q["share"]:.1%} of a layer\'s multiply-adds at length 128 and {r["share"]:.1%} at length 512.', b)


F['p3_attn_cost'] = attn_cost()

json.dump(F, open('results/figs_part3.json', 'w'))
print('figures:', ', '.join(F))
