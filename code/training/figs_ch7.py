"""Figures for Chapter 7 (reward models). Writes results/figs_ch7.json (key -> svg).
Colours come only from CSS classes, so every figure follows the site's themes. Every number is read from a results/ch7_*.json
file written by a script (ch7_bt.py, ch7_data.py, ch7_train.py, ch7_eval.py, ch7_bon.py)."""
import json, math, os
import numpy as np
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars

J = lambda n: json.load(open(f'results/{n}.json')) if os.path.exists(f'results/{n}.json') else None
BT, DATA, TRAIN, EV, BON = J('ch7_bt'), J('ch7_data'), J('ch7_train_main'), J('ch7_eval'), J('ch7_bon')
TLR1 = J('ch7_train_lr1')
# helpers copied from figs_ch4.py (importing it would rebuild the Chapter 4 figures)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick'):
    b = [box(x, y, w, h, cls, 10)]
    b.append(text(x + w / 2, y + (h / 2 + 5 if not lines else 22), title, tcls, 'middle'))
    b += para(x + w / 2, y + 41, lines, lcls, 'middle')
    return b


def chart(x0, y0, w, h, series, xr, yr, xticks, yticks, xlog=False, ylog=False, fx=str, fy=str, xlabel='', ylabel='',
          end_labels=True, dots=True):
    """A small line chart. series: [(name, xs, ys, line_cls, dot_cls)]. Returns (parts, X, Y)."""
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X = lambda v: x0 + (tx(v) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0])) * w
    Y = lambda v: y0 + h - (ty(v) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0])) * h
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, fy(t), 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), y0 + h + 18, fx(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    ends = []
    for name, xs, ys, lc, dc in series:
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(xs, ys)) + '"/>')
        if dots:
            for a, c in zip(xs, ys):
                b.append(f'<g class="mark"><title>{esc(name)}: {fx(a)}, {fy(c)}</title><circle class="{dc} ring" cx="{X(a):.1f}" cy="{Y(c):.1f}" r="4"/></g>')
        ends.append([Y(ys[-1]), name, dc, X(xs[-1]), Y(ys[-1])])
    if end_labels:
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
        for ly, name, dc, ex, ey in ends:
            b.append(f'<rect class="{dc}" x="{ex + 12:.1f}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
            b.append(text(ex + 27, ly + 4, name, 't-tick'))
    if xlabel:
        b.append(text(x0 + w / 2, y0 + h + 38, xlabel, 't-muted', 'middle'))
    if ylabel:
        b.append(text(x0 - 8, y0 - 16, ylabel, 't-muted', 'start'))
    return b, X, Y


F = {}
DASH = ' stroke-dasharray="6 4"'


def dline(xs, ys, X, Y, cls, dash=False):
    return f'<polyline class="{cls}"{DASH if dash else ""} points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(xs, ys)) + '"/>'


def legend(x, y, items):
    """items: [(label, cls, kind)] kind 'line', 'dash' or 'box'. Returns parts and the end x."""
    b = []
    for lab, cls, kind in items:
        if kind == 'box':
            b.append(f'<rect class="{cls}" x="{x}" y="{y - 9}" width="12" height="12" rx="2"/>')
        else:
            b.append(f'<line class="{cls}" x1="{x}" y1="{y - 3}" x2="{x + 22}" y2="{y - 3}"{DASH if kind == "dash" else ""}/>')
        b.append(text(x + 28, y + 1, lab, 't-tick'))
        x += 40 + 7.0 * len(lab)
    return b, x


# ---------------------------------------------------------------- 1. where the reward model sits
def where():
    b = [text(20, 24, 'Where the reward model sits in RLHF', 't-title')]
    steps = [('SFT model', ['writes several', 'answers per prompt'], 'box-1'),
             ('people compare', ['"A is better', 'than B"'], 'box-2'),
             ('reward model', ['learns a score', 'r(x, y) from pairs'], 'box-on'),
             ('optimise', ['RL (Chapter 8) or', 'best-of-n'], 'box-3')]
    x = 20
    for i, (t, ls, c) in enumerate(steps):
        b += block(x, 46, 170, 84, t, ls, c)
        if i < 3:
            b.append(arrow(x + 172, 88, x + 196, 88))
        x += 198
    b += [line(105, 132, 105, 168, 'edge-dim'), line(105, 168, 698, 168, 'edge-dim'), line(698, 168, 698, 132, 'edge-dim'),
          text(400, 186, 'the optimised model writes new answers; the loop can repeat with fresh comparisons', 't-muted', 'middle')]
    b += [box(216, 206, 368, 52, 'box-ghost', 8),
          text(400, 228, 'People label a few thousand pairs once.', 't-tick', 'middle'),
          text(400, 246, 'The reward model then scores millions of answers for free.', 't-tick', 'middle')]
    return svg(800, 270, 'The RLHF loop: the SFT model writes answers, people compare pairs, a reward model learns a score from the pairs, and an optimiser uses that score.', b)


F['ch7_where'] = where()


# ---------------------------------------------------------------- 2. ratings vs comparisons simulation
def ratings_sim():
    if not BT:
        return None
    rv = BT['ratings_vs_comparisons']
    b = [text(20, 24, 'Ratings or comparisons? 200 answers, 20 simulated raters, 50 repeats (ch7_bt.py)', 't-title')]
    x0, y0, w, h = 90, 66, 600, 200
    Y = lambda v: y0 + h - (v - 0.5) / 0.5 * h
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, f'{t:.1f}', 't-tick', 'end')]
    b.append(text(x0 - 60, y0 - 12, 'rank correlation with the true quality', 't-muted'))
    labels = {'0.0': 'stable raters', '0.3': 'some drift', '0.6': 'strong drift'}
    for i, k in enumerate(['0.0', '0.3', '0.6']):
        cx = x0 + 100 + i * 200
        for j, (key, cls, nm) in enumerate([('rating', 's1', 'mean rating'), ('bt', 's2', 'Bradley-Terry')]):
            v = float(np.mean(rv[k][key]))
            bx = cx - 58 + j * 62
            b.append(f'<g class="mark"><title>{nm}: {v:.3f}</title><rect class="{cls}" x="{bx}" y="{Y(v):.1f}" width="54" height="{y0 + h - Y(v):.1f}" rx="3"/></g>')
            b.append(text(bx + 27, Y(v) - 6, f'{v:.2f}', 't-val', 'middle'))
        b.append(text(cx - 2, y0 + h + 20, f'{labels[k]} (drift {k})', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    lg, _ = legend(x0 + 150, y0 + h + 50, [('mean of 1,200 ratings (1 to 10)', 's1', 'box'), ('Bradley-Terry on 600 comparisons', 's2', 'box')])
    b += lg
    return svg(800, y0 + h + 66, 'Grouped bars: with stable raters, averaging ratings recovers the true order better; as raters drift, ratings get worse while comparisons stay at about 0.76.', b)


F['ch7_ratings_sim'] = ratings_sim()


# ---------------------------------------------------------------- 3. Bradley-Terry step by step
def bt_frames():
    if not BT:
        return None
    p = BT['pair']
    b = [text(20, 24, 'From two rewards to a loss, with the numbers from ch7_bt.py', 't-title')]
    fr = [('1. score each answer', [f'r(x, chosen) = {p["rc"]}', f'r(x, rejected) = {p["rr"]}'], 'box-1'),
          ('2. take the gap', [f'{p["rc"]} - {p["rr"]} = {p["rc"] - p["rr"]:.1f}', 'only the gap matters'], 'box-2'),
          ('3. squash with sigmoid', [f'sigmoid({p["rc"] - p["rr"]:.1f}) = {p["p"]:.3f}', 'P(chosen wins)'], 'box-3'),
          ('4. loss = -log P', [f'-log {p["p"]:.3f} = {p["loss"]:.3f}', f'wrong way round: {p["loss_wrong"]:.3f}'], 'box-on')]
    x = 20
    for i, (t, ls, c) in enumerate(fr):
        b += block(x, 46, 178, 88, t, ls, c)
        if i < 3:
            b.append(arrow(x + 180, 90, x + 196, 90))
        x += 196
    b.append(text(400, 162, f'The gradient pushes r(chosen) up and r(rejected) down by the same amount: 1 - P = {1 - p["p"]:.3f}.', 't-muted', 'middle'))
    return svg(800, 178, 'Four steps: score both answers, take the gap, turn it into a probability with the sigmoid, and take minus the log.', b)


F['ch7_bt_frames'] = bt_frames()


# ---------------------------------------------------------------- 4. loss and gradient against the gap (plus label noise)
def loss_curve():
    if not BT:
        return None
    c = BT['curve']
    g = c['gap']
    noisy = [-math.log(0.05 + 0.9 / (1 + math.exp(-x))) for x in g]
    b = [text(20, 24, 'The reward model loss as a function of the reward gap', 't-title')]
    parts, X, Y = chart(70, 56, 540, 230, [('loss, plain', g, c['loss'], 'l1', 's1'), ('loss, 10% random labels', g, noisy, 'l2', 's2'),
                                         ('-gradient on r(chosen)', g, [-v for v in c['grad']], 'l3', 's3')],
                        (-4, 4), (0, 4.2), [-4, -2, 0, 2, 4], [0, 1, 2, 3, 4], fx=lambda v: f'{v:+d}' if v else '0', fy=str,
                        xlabel='reward gap r(chosen) - r(rejected)', ylabel='', dots=False, end_labels=False)
    b += parts
    lg = []
    for i, (lab, cls) in enumerate([('loss, plain', 'l1'), ('loss, 10% random labels', 'l2'), ('size of the gradient on r(chosen)', 'l3')]):
        lg += [f'<line class="{cls}" x1="650" y1="{80 + i * 22}" x2="672" y2="{80 + i * 22}"/>', text(678, 84 + i * 22, lab, 't-tick')]
    b += lg
    p = BT['pair']
    b += [f'<circle class="s1 ring" cx="{X(p["rc"] - p["rr"]):.1f}" cy="{Y(p["loss"]):.1f}" r="5"/>',
          text(X(0.9) + 8, Y(p['loss']) - 8, f'gap 0.9: loss {p["loss"]:.3f}', 't-tick')]
    b.append(text(X(-3.9), Y(4.05), 'wrong order: big loss', 't-muted'))
    b.append(text(X(1.6), Y(1.2), 'right order: loss goes to 0', 't-muted'))
    return svg(900, 330, 'Line chart over the reward gap from -4 to 4: the plain loss falls from 4 to near 0; with 10% random labels the loss is capped near 3 for wrong orderings; the gradient shrinks as the gap grows.', b)


F['ch7_loss_curve'] = loss_curve()


# ---------------------------------------------------------------- 5. a ranking of four answers becomes six pairs
def ranking():
    if not BT:
        return None
    rk = BT['rank']
    rw = rk['rewards']
    b = [text(20, 24, 'A labeler ranks four answers: A > B > C > D. That is 6 comparisons.', 't-title')]
    for i, (k, v) in enumerate(rw.items()):
        b += [box(20 + i * 74, 46, 64, 50, ['box-3', 'box-1', 'box-2', 'box-4'][i], 8), text(52 + i * 74, 68, k, 't-note', 'middle'),
              text(52 + i * 74, 86, f'r = {v}', 't-tick', 'middle')]
    b.append(text(20, 118, "the model's current rewards", 't-muted'))
    y = 46
    for nm, l in zip(rk['pairs'], rk['losses']):
        w, lo = nm.split('>')
        gap = rw[w] - rw[lo]
        b += [text(360, y + 15, f'{w} > {lo}', 't-tick t-strong'), text(420, y + 15, f'gap {gap:+.1f}', 't-tick')]
        bw = l / 0.6 * 200
        b.append(f'<rect class="s1" x="500" y="{y + 3}" width="{bw:.1f}" height="16" rx="3"/>')
        b.append(text(506 + bw, y + 15, f'loss {l:.3f}', 't-val'))
        y += 24
    b.append(text(360, y + 18, f'mean over the 6 pairs = {rk["mean"]:.4f}: one batch element, one forward pass per answer', 't-muted'))
    return svg(800, y + 32, 'Four answers with rewards 2.1, 1.2, 0.9 and -0.5; the six pairs, their gaps and losses; the smallest gap (B over C) has the largest loss.', b)


F['ch7_ranking'] = ranking()


# ---------------------------------------------------------------- 6. architecture
def arch():
    b = [text(20, 24, 'A reward model is a language model with its last layer swapped for one number', 't-title')]
    toks = ['<|im_start|>', 'user', '...', '<|im_end|>', '<|im_start|>', 'assistant', 'The', 'word', '"red"', '...', '<|im_end|>']
    x = 20
    xs = []
    for t in toks:
        w = max(34, 7.2 * len(t) + 10)
        last = t == '<|im_end|>' and len(xs) == len(toks) - 1
        b += [box(x, 196, w, 26, 'box-on' if last else 'box', 5), text(x + w / 2, 213, t, 't-tick', 'middle')]
        xs.append((x, w))
        x += w + 4
    W = x - 24
    b += [box(20, 120, W, 52, 'box-1', 10), text(20 + W / 2, 142, 'Qwen2.5-0.5B-Instruct transformer, 24 layers (frozen, plus LoRA adapters)', 't-note', 'middle'),
          text(20 + W / 2, 160, 'every token gets a 896-number hidden state; causal attention, so the last one has seen everything', 't-tick', 'middle')]
    lx, lw = xs[-1]
    cx = lx + lw / 2
    b += [line(cx, 194, cx, 174, 'edge-on'), box(cx - 60, 62, 120, 40, 'box-3', 8), text(cx, 80, 'score head', 't-note', 'middle'),
          text(cx, 96, '896 x 1, no bias', 't-tick', 'middle'), line(cx, 118, cx, 104, 'edge-on'),
          box(cx - 46, 14, 92, 34, 'box-on', 8), text(cx, 36, f'r = {EV["example"]["rc"]:.2f}' if EV else 'r', 't-note', 'middle'), line(cx, 60, cx, 50, 'edge-on')]
    b += para(20, 66, ['The language model head (896 x 151,936) is gone.', 'In its place: one weight per hidden unit.',
                       'Hidden states of the other tokens are ignored.'], 't-muted')
    b.append(text(20, 246, 'The reward is read at the token that closes the reply, so it can depend on the whole prompt and the whole answer.', 't-muted'))
    return svg(800, 260, 'The chat goes into the transformer; the hidden state at the final <|im_end|> token goes through a 896-by-1 linear layer that outputs the reward.', b)


F['ch7_arch'] = arch()


# ---------------------------------------------------------------- 7. one training example
def example():
    if not DATA:
        return None
    e = DATA['example']
    b = [text(20, 24, 'One UltraFeedback pair (test set), rated by GPT-4 on a 1 to 10 scale', 't-title')]
    b += [box(20, 40, 760, 34, 'box-ghost', 8), text(32, 62, 'prompt: ' + e['prompt'].replace('\n', ' '), 't-tick')]
    b += [box(20, 86, 372, 112, 'box-3', 8), text(32, 106, f'chosen: rating {e["sc"]}', 't-note')]
    b += para(32, 128, ['The word "red" would be an appropriate word', 'to fill in the blank in the sentence', '"She was wearing a [___] dress."'], 't-tick')
    b += [box(408, 86, 372, 112, 'box-4', 8), text(420, 106, f'rejected: rating {e["sr"]}', 't-note')]
    b += para(420, 128, ['Is the dress navy blue, black, or any other color?'], 't-tick')
    if EV:
        b += [text(32, 186, f'our reward model: {EV["example"]["rc"]:+.2f}', 't-val'), text(420, 186, f'our reward model: {EV["example"]["rr"]:+.2f}', 't-val')]
    b.append(text(400, 220, 'The reward model never sees the ratings. It only learns that the left answer should score higher.', 't-muted', 'middle'))
    return svg(800, 234, 'A preference pair: the prompt asks to fill a blank; the chosen answer suggests red (rating 7), the rejected answer asks a question back (rating 3).', b)


F['ch7_example'] = example()


# ---------------------------------------------------------------- 8. training curve
def train_curve():
    if not TRAIN:
        return None
    h = TRAIN['hist']
    st = h['step']
    b = [text(20, 24, f'Training the reward model: {TRAIN["args"]["n_train"]:,} pairs, one epoch, {TRAIN["steps"]} steps of 8 pairs (ch7_train.py)', 't-title')]
    s1 = [(s, v) for s, v in zip(st, h['train_loss']) if v is not None]
    parts, X, Y = chart(70, 60, 250, 200, [('train loss', [a for a, _ in s1], [v for _, v in s1], 'l1', 's1'),
                                         ('test loss', st, h['test_loss'], 'l2', 's2')],
                        (0, st[-1]), (0.5, 0.8), [0, st[-1] // 2, st[-1]], [0.5, 0.6, 0.7, 0.8], fy=lambda v: f'{v:.1f}',
                        xlabel='optimizer step', ylabel='loss (chance = 0.693)')
    b += parts
    h1 = TLR1['hist'] if TLR1 else None
    ser = [('lr 3e-4 (main)', st, h['test_acc'], 'l3', 's3')] + ([('lr 1e-4', h1['step'], h1['test_acc'], 'l4', 's4')] if h1 else [])
    parts, X, Y = chart(450, 60, 220, 200, ser,
                        (0, st[-1]), (0.4, 0.8), [0, st[-1] // 2, st[-1]], [0.4, 0.5, 0.6, 0.7, 0.8], fy=lambda v: f'{v:.0%}',
                        xlabel='optimizer step', ylabel='held-out accuracy (600 pairs)', end_labels=True)
    b += parts
    b.append(text(X(0) + 6, Y(h['test_acc'][0]) + 18, f'{h["test_acc"][0]:.1%} before training', 't-tick'))
    return svg(820, 310, 'Left: training and test loss of the main run fall from about 0.79 to below 0.6. Right: held-out accuracy of the main run and of the same run with a three times smaller learning rate.', b)


F['ch7_train_curve'] = train_curve()


# ---------------------------------------------------------------- 9. accuracy by rating gap
def by_gap():
    if not EV:
        return None
    bg = EV['by_gap']
    names = list(bg)
    b = [text(20, 24, 'Held-out accuracy by how far apart the two GPT-4 ratings are (600 test pairs)', 't-title')]
    x0, y0, w, h = 80, 54, 640, 210
    Y = lambda v: y0 + h - (v - 0.3) / 0.7 * h
    for t in [0.3, 0.5, 0.7, 0.9, 1.0]:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, f'{t:.0%}', 't-tick', 'end')]
    b.append(f'<line class="edge-dim" x1="{x0}" y1="{Y(0.5):.1f}" x2="{x0 + w}" y2="{Y(0.5):.1f}"/>')
    series = [('lr1', 's4', 'plain, lr 1e-4'), ('margin', 's2', 'margin, lr 1e-4'), ('main', 's1', 'plain, lr 3e-4 (main)'), ('gold', 's3', 'gold RM (Skywork V2)')]
    for i, nm in enumerate(names):
        cx = x0 + 80 + i * 160
        for j, (k, cls, lab) in enumerate(series):
            v = bg[nm].get(k) if k != 'gold' else None
            if k == 'gold':
                v = EV['by_gap_gold'][nm] if 'by_gap_gold' in EV else None
            if v is None:
                continue
            bx = cx - 68 + j * 34
            b.append(f'<g class="mark"><title>{lab}, {nm}: {v:.3f}</title><rect class="{cls}" x="{bx}" y="{Y(v):.1f}" width="30" height="{y0 + h - Y(v):.1f}" rx="3"/></g>')
            b.append(text(bx + 15, Y(v) - 5, f'{v * 100:.0f}', 't-val', 'middle'))
        b += [text(cx, y0 + h + 18, nm, 't-tick', 'middle'), text(cx, y0 + h + 34, f'n = {bg[nm]["n"]}', 't-muted', 'middle')]
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    lg, _ = legend(x0 - 40, y0 + h + 62, [(lab, cls, 'box') for _, cls, lab in series])
    b += lg
    return svg(800, y0 + h + 78, 'Grouped bars of accuracy for four rating-gap buckets: all reward models are near chance when the ratings are close and much better when one answer is clearly better.', b)


F['ch7_by_gap'] = by_gap()


# ---------------------------------------------------------------- 10. calibration
def calib():
    if not EV:
        return None
    c = EV['calib']
    b = [text(20, 24, f'Calibration: predicted confidence against actual accuracy (ECE {EV["ece"]:.3f})', 't-title')]
    x0, y0, s = 90, 64, 240
    X = lambda v: x0 + (v - 0.5) / 0.5 * s
    Y = lambda v: y0 + s - (v - 0.3) / 0.7 * s
    for t in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        b.append(text(X(t), y0 + s + 18, f'{t:.1f}', 't-tick', 'middle'))
    for t in [0.3, 0.5, 0.7, 0.9]:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + s}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, f'{t:.1f}', 't-tick', 'end')]
    b.append(f'<line class="edge-dim" x1="{X(0.5):.1f}" y1="{Y(0.5):.1f}" x2="{X(1):.1f}" y2="{Y(1):.1f}"/>')
    b.append(text(X(0.93), Y(0.98) - 4, 'perfect', 't-muted', 'end'))
    pts = [(X(r['conf']), Y(r['acc'])) for r in c]
    b.append('<polyline class="l1" points="' + ' '.join(f'{a:.1f},{bb:.1f}' for a, bb in pts) + '"/>')
    for r, (a, bb) in zip(c, pts):
        rad = 3 + 6 * math.sqrt(r['n'] / 600)
        b.append(f'<g class="mark"><title>confidence {r["conf"]:.2f}, accuracy {r["acc"]:.2f}, n={r["n"]}</title><circle class="s1 ring" cx="{a:.1f}" cy="{bb:.1f}" r="{rad:.1f}"/></g>')
    b += [text(x0 + s / 2, y0 + s + 38, 'mean confidence in the bin, max(P, 1 - P)', 't-muted', 'middle'), text(x0 - 60, y0 - 12, 'accuracy in the bin', 't-muted')]
    tx = x0 + s + 70
    cols = [(tx, 'bin', 'start'), (tx + 120, 'pairs', 'end'), (tx + 220, 'confidence', 'end'), (tx + 310, 'accuracy', 'end')]
    for x, h, a in cols:
        b.append(text(x, y0 + 10, h, 't-tick t-strong', a))
    for i, r in enumerate(c):
        vals = [f'{r["lo"]:.1f} to {r["hi"]:.1f}', str(r['n']), f'{r["conf"]:.3f}', f'{r["acc"]:.3f}']
        for (x, _, a), v in zip(cols, vals):
            b.append(text(x, y0 + 34 + i * 22, v, 't-tick', a))
    b += para(tx, y0 + 34 + len(c) * 22 + 20, ['Dot size: number of pairs in the bin.', 'Points below the diagonal: overconfident;', 'above: underconfident.'], 't-muted')
    return svg(800, y0 + s + 52, 'Reliability diagram: the reward model confidence on each pair against how often it is right, in five bins, with a table of the values.', b)


F['ch7_calib'] = calib()


def hist(x0, y0, w, h, data, lo, hi, nb, cls_list, labels, xlabel):
    """Overlaid histograms as step outlines (filled lightly via classes)."""
    b = []
    edges = np.linspace(lo, hi, nb + 1)
    cnts = [np.histogram(np.clip(d, lo, hi - 1e-9), edges)[0] for d in data]
    top = max(c.max() for c in cnts)
    X = lambda v: x0 + (v - lo) / (hi - lo) * w
    Y = lambda v: y0 + h - v / top * h
    for c, cls in zip(cnts, cls_list):
        pts = [(X(edges[0]), Y(0))]
        for i, v in enumerate(c):
            pts += [(X(edges[i]), Y(v)), (X(edges[i + 1]), Y(v))]
        pts.append((X(edges[-1]), Y(0)))
        b.append(f'<polyline class="{cls}" points="' + ' '.join(f'{a:.1f},{bb:.1f}' for a, bb in pts) + '"/>')
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    step = 1 if hi - lo <= 8 else 2
    for t in np.arange(math.ceil(lo), hi + 0.01, step):
        b.append(text(X(t), y0 + h + 16, f'{t:g}', 't-tick', 'middle'))
    b.append(text(x0 + w / 2, y0 + h + 34, xlabel, 't-muted', 'middle'))
    return b, X, Y


# ---------------------------------------------------------------- 11. reward distributions
def dist():
    if not EV:
        return None
    rc, rr = np.array(EV['dist']['rc']), np.array(EV['dist']['rr'])
    lo, hi = math.floor(min(rc.min(), rr.min())), math.ceil(max(rc.max(), rr.max()))
    b = [text(20, 24, 'Rewards of the 600 chosen and 600 rejected test answers (our reward model)', 't-title')]
    parts, X, Y = hist(60, 50, 660, 200, [rc, rr], lo, hi, 30, ['l1', 'l2'], ['chosen', 'rejected'], 'reward')
    b += parts
    for v, cls, lab, dy in [(rc.mean(), 'l1', f'chosen mean {rc.mean():+.2f}', 0), (rr.mean(), 'l2', f'rejected mean {rr.mean():+.2f}', 16)]:
        b += [f'<line class="{cls}"{DASH} x1="{X(v):.1f}" y1="50" x2="{X(v):.1f}" y2="250"/>']
    lg, _ = legend(80, 300, [(f'chosen (mean {rc.mean():+.2f})', 'l1', 'line'), (f'rejected (mean {rr.mean():+.2f})', 'l2', 'line'), ('dashed: means', 'edge-dim', 'dash')])
    b += lg
    return svg(800, 316, 'Two overlapping histograms of rewards: chosen answers are shifted to the right of rejected ones, but the two distributions overlap heavily.', b)


F['ch7_dist'] = dist()


# ---------------------------------------------------------------- 12. reward against length
def length():
    if not EV:
        return None
    L = EV['length']
    r, n = np.array(L['r']), np.array(L['len'])
    b = [text(20, 24, f'Reward against answer length, 1,200 test answers (Spearman {L["spearman"]:.2f})', 't-title')]
    x0, y0, w, h = 70, 50, 420, 230
    lo, hi = math.floor(r.min()), math.ceil(r.max())
    X = lambda v: x0 + min(v, 512) / 512 * w
    Y = lambda v: y0 + h - (v - lo) / (hi - lo) * h
    for t in range(lo, hi + 1, 2 if hi - lo > 8 else 1):
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, str(t), 't-tick', 'end')]
    for t in [0, 128, 256, 384, 512]:
        b.append(text(X(t), y0 + h + 16, str(t), 't-tick', 'middle'))
    for a, c in zip(n, r):
        b.append(f'<circle class="s1" cx="{X(a):.1f}" cy="{Y(c):.1f}" r="2" opacity="0.45"/>')
    # binned means
    edges = [0, 32, 64, 128, 192, 256, 384, 512]
    mids, means = [], []
    for a, c in zip(edges[:-1], edges[1:]):
        sel = (n >= a) & (n < c)
        if sel.sum() > 5:
            mids.append(n[sel].mean()); means.append(r[sel].mean())
    b.append('<polyline class="l2" points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(mids, means)) + '"/>')
    for a, c in zip(mids, means):
        b.append(f'<circle class="s2 ring" cx="{X(a):.1f}" cy="{Y(c):.1f}" r="4"/>')
    b += [text(x0 + w / 2, y0 + h + 34, 'answer length (tokens)', 't-muted', 'middle'), text(x0 - 50, y0 - 12, 'reward', 't-muted')]
    tx = 530
    rows = [('reward vs length', L['spearman']), ('GPT-4 rating vs length', L['rating_spearman']),
            ('reward gap vs length gap', L['pair_spearman'])]
    b.append(text(tx, 70, 'Spearman correlations', 't-tick t-strong'))
    for i, (k, v) in enumerate(rows):
        b.append(text(tx, 94 + i * 22, f'{k}: {v:+.2f}', 't-tick'))
    b.append(text(tx, 180, 'accuracy when chosen is', 't-tick t-strong'))
    b.append(text(tx, 204, f'longer: {L["acc_longer"]:.1%}', 't-tick'))
    b.append(text(tx, 226, f'shorter: {L["acc_shorter"]:.1%}', 't-tick'))
    b.append(text(tx, 256, 'orange: mean reward per length bin', 't-muted'))
    return svg(800, y0 + h + 50, 'Scatter plot of reward against answer length with binned means, and a table of correlations and of accuracy split by whether the chosen answer is longer.', b)


F['ch7_length'] = length()


# ---------------------------------------------------------------- 13. probes
def probes():
    if not EV:
        return None
    P = EV['probes']
    b = [text(20, 24, 'Edit an answer, re-score it: mean change in reward, in units of each model\'s reward sd (150 answers)', 't-title')]
    x0, y0 = 20, 56
    zero, sc = 500, 110
    b.append(f'<line class="axis" x1="{zero}" y1="{y0 - 8}" x2="{zero}" y2="{y0 + 3 * 64}"/>')
    for i, (nm, v) in enumerate(P.items()):
        y = y0 + i * 64
        b.append(text(x0, y + 26, nm, 't-tick t-strong'))
        for j, (k, cls, lab) in enumerate([('main', 's1', 'our RM'), ('gold', 's3', 'gold RM')]):
            m = float(np.mean(v[k])) / v[k + '_sd']
            up = float(np.mean(np.array(v[k]) > 0))
            yy = y + j * 24
            x1, x2 = sorted([zero, zero + m * sc])
            b.append(f'<g class="mark"><title>{lab}: {m:+.2f} sd, up in {up:.0%}</title><rect class="{cls}" x="{x1:.1f}" y="{yy + 4}" width="{max(2, x2 - x1):.1f}" height="18" rx="3"/></g>')
            tx = (x2 + 6) if m >= 0 else (zero + 8)
            b.append(text(tx, yy + 17, f'{lab} {m:+.2f} (up in {up:.0%})', 't-val'))
    b.append(text(zero, y0 + 3 * 64 + 14, 'no change', 't-muted', 'middle'))
    return svg(800, y0 + 3 * 64 + 28, 'Bars for three edits: adding a polite closing line, repeating the answer and cutting it in half, showing how much our reward model and the gold reward model change their scores.', b)


F['ch7_probes'] = probes()


# ---------------------------------------------------------------- 14. best-of-n: proxy vs gold
def bon_curves():
    if not BON:
        return None
    C = BON['curves']
    d = [math.sqrt(k) for k in BON['kl']]
    b = [text(20, 24, 'Best-of-n on 100 prompts x 32 samples: what the proxy promises and what the gold RM sees', 't-title')]
    panels = [('main', 'proxy = our RM (3,200 pairs)', 30), ('small', 'proxy = small RM (400 pairs)', 420)]
    allv = [v for k, _, _ in panels for v in C[k]['proxy'] + C[k]['gold']]
    ymax = math.ceil(max(allv) * 2) / 2
    ymin = min(0, math.floor(min(allv) * 2) / 2)
    for k, title, x0 in panels:
        X = lambda v, x0=x0: x0 + 50 + v / d[-1] * 280
        Y = lambda v: 70 + 200 - (v - ymin) / (ymax - ymin) * 200
        b.append(text(x0 + 50, 54, title, 't-tick t-strong'))
        t = ymin
        while t <= ymax + 1e-9:
            b += [f'<line class="grid" x1="{x0 + 50}" y1="{Y(t):.1f}" x2="{x0 + 330}" y2="{Y(t):.1f}"/>', text(x0 + 42, Y(t) + 4, f'{t:+.1f}' if t else '0', 't-tick', 'end')]
            t += 0.5
        for n, dv in zip(BON['ns'], d):
            b.append(text(X(dv), 290, str(n), 't-tick', 'middle'))
        b.append(text(x0 + 190, 308, 'n (x axis: square root of KL)', 't-muted', 'middle'))
        b.append(dline(d, C[k]['proxy'], X, Y, 'l1', True))
        b.append(dline(d, C[k]['gold'], X, Y, 'l3'))
        for dv, pv, gv in zip(d, C[k]['proxy'], C[k]['gold']):
            b += [f'<circle class="s1 ring" cx="{X(dv):.1f}" cy="{Y(pv):.1f}" r="3.5"/>', f'<circle class="s3 ring" cx="{X(dv):.1f}" cy="{Y(gv):.1f}" r="3.5"/>']
        b.append(text(X(d[-1]) + 6, Y(C[k]['proxy'][-1]) + 4, f'{C[k]["proxy"][-1]:+.2f}', 't-val'))
        b.append(text(X(d[-1]) + 6, Y(C[k]['gold'][-1]) + 4, f'{C[k]["gold"][-1]:+.2f}', 't-val'))
    lg, _ = legend(160, 340, [('proxy reward of the kept answer (dashed)', 'l1', 'dash'), ('gold reward of the same answer', 'l3', 'line')])
    b += lg
    b.append(text(20, 362, 'Rewards in units of each model\'s standard deviation over all samples, measured from n = 1.', 't-muted'))
    return svg(800, 374, 'Two panels. For each proxy reward model, the proxy score of the best-of-n answer rises steeply with n, while the gold score rises much less.', b)


F['ch7_bon_curves'] = bon_curves()


# ---------------------------------------------------------------- 15. best-of-n: length of the kept answer
def bon_words():
    if not BON:
        return None
    C = BON['curves']
    ns = BON['ns']
    series = [('our RM', 'main', 'l1', 's1'), ('small RM', 'small', 'l2', 's2'), ('ensemble', 'ens_mean', 'l4', 's4'), ('gold RM', 'gold', 'l3', 's3')]
    b = [text(20, 24, 'What does best-of-n select? Share of kept answers that finished, and their length', 't-title')]
    parts, X, Y = chart(60, 60, 250, 210, [(nm, ns, C[k]['finished'], lc, dc) for nm, k, lc, dc in series], (1, 32), (0.2, 0.7),
                        ns, [0.2, 0.3, 0.4, 0.5, 0.6, 0.7], xlog=True, fx=str, fy=lambda v: f'{v:.0%}', xlabel='n (log scale)',
                        ylabel='kept answer ended within 256 tokens')
    b += parts
    parts, X, Y = chart(470, 60, 220, 210, [(nm, ns, C[k]['words'], lc, dc) for nm, k, lc, dc in series], (1, 32), (120, 170),
                        ns, [120, 130, 140, 150, 160, 170], xlog=True, fx=str, fy=str, xlabel='n (log scale)', ylabel='words in the kept answer')
    b += parts
    return svg(820, 320, 'Two line charts against n. Left: the share of kept answers that finished rises for every selector. Right: the length of the kept answer stays flat for our reward model and falls for the small and gold models.', b)


F['ch7_bon_words'] = bon_words()


# ---------------------------------------------------------------- 16. margin: distribution of reward gaps
def margin():
    if not EV:
        return None
    O = EV['others']
    gm = np.array(O['lr1']['rc']) - np.array(O['lr1']['rr'])
    gg = np.array(O['margin']['rc']) - np.array(O['margin']['rr'])
    lo, hi = -12, 16
    b = [text(20, 24, 'Reward gaps on the 600 test pairs: plain loss against the Llama 2 margin loss (same data, seed and lr)', 't-title')]
    parts, X, Y = hist(60, 50, 660, 200, [gm, gg], lo, hi, 56, ['l1', 'l2'], [], 'reward gap r(chosen) - r(rejected)')
    b += parts
    b.append(f'<line class="edge-dim" x1="{X(0):.1f}" y1="50" x2="{X(0):.1f}" y2="250"/>')
    lg, _ = legend(40, 300, [(f'plain, lr 1e-4 (sd of rewards {O["lr1"]["sd"]:.2f}, accuracy {O["lr1"]["acc"]:.1%})', 'l1', 'line'),
                             (f'margin (sd {O["margin"]["sd"]:.2f}, accuracy {O["margin"]["acc"]:.1%})', 'l2', 'line')])
    b += lg
    return svg(800, 316, 'Two histograms of reward gaps: the margin-trained model spreads its gaps more widely than the plain one.', b)


F['ch7_margin'] = margin()


# ---------------------------------------------------------------- 17. all reward models on the test set
def models():
    if not EV:
        return None
    O, E = EV['others'], EV['ensemble']
    items = [('small: 400, 1e-4', O['small']['acc']), ('seed1: 1,600, 1e-4', O['seed1']['acc']), ('seed2: 1,600, 1e-4', O['seed2']['acc']),
             ('lr1: 3,200, 1e-4', O['lr1']['acc']), ('margin: 3,200, 1e-4', O['margin']['acc']), ('main: 3,200, 3e-4', O['main']['acc']),
             ('ensemble: main+seed1+seed2', E['acc']), ('gold: Skywork V2', O['gold']['acc'])]
    b = hbars(20, 50, items, 400, row_h=28, cls='s1', label_w=210, fmt=lambda v: f'{v:.1%}', vmax=1.0,
              title='Accuracy on the same 600 held-out pairs (name: training pairs, learning rate)', hi='main: 3,200, 3e-4')
    y = 50 + len(items) * 28 + 22
    b.append(text(20, y, f'The three ensemble members agree on {E["agree"]:.0%} of pairs: accuracy {E["acc_agree"]:.1%} there, '
                         f'{E["acc_disagree"]:.1%} where they disagree.', 't-muted'))
    return svg(800, y + 16, 'Horizontal bars of accuracy for each reward model: more training data helps, the ensemble and margin are close to the main model, the gold model is best.', b)


F['ch7_models'] = models()


# ---------------------------------------------------------------- 18. process vs outcome
def prm():
    if not BT:
        return None
    P = BT['prm']
    b = [text(20, 24, 'Outcome reward vs process reward on a five-step solution (illustrative numbers, ch7_bt.py)', 't-title')]
    y = 50
    for nm, cls_bad in [('right answer', None), ('one wrong step', 3)]:
        ps = P[nm]['steps']
        b.append(text(20, y + 20, nm, 't-tick t-strong'))
        x = 140
        for k, p in enumerate(ps):
            cls = 'box-4' if p < 0.5 else 'box-3'
            b += [box(x, y, 84, 32, cls, 6), text(x + 42, y + 14, f'step {k + 1}', 't-tick', 'middle'), text(x + 42, y + 27, f'p = {p:.2f}', 't-tick', 'middle')]
            if k < len(ps) - 1:
                b.append(text(x + 92, y + 21, 'x', 't-muted', 'middle'))
            x += 100
        b.append(text(x + 4, y + 21, f'= {P[nm]["product"]:.3f}', 't-note'))
        y += 52
    b += [box(140, y + 6, 480, 40, 'box-ghost', 8),
          text(380, y + 24, 'An ORM sees only the end: one label per solution, no idea where it broke.', 't-tick', 'middle'),
          text(380, y + 40, 'A PRM scores each step; the product collapses at the first wrong one.', 't-tick', 'middle')]
    return svg(800, y + 60, 'Two solutions as rows of five steps with step probabilities; the product is 0.815 for the correct one and 0.244 for the one with a wrong fourth step.', b)


F['ch7_prm'] = prm()

json.dump({k: v for k, v in F.items() if v}, open('results/figs_ch7.json', 'w'))
print('figures:', ', '.join(k for k, v in F.items() if v), '| missing:', ', '.join(k for k, v in F.items() if not v))
