"""Figures for Part 4 (fine-tuning and results), drawn from results/part4_*.json. Writes results/figs_part4.json."""
import json, math
from figlib import svg, text, box, arrow, esc
from bertfig import token, row, line, hbars

F = {}
J = lambda n: json.load(open(f'results/{n}.json'))


def inputs():
    """The four ways a task fills the two slots, sentence A and sentence B."""
    rows_ = [('paraphrase (MRPC, QQP)', 'sentence 1', 'sentence 2', 'same meaning?'),
             ('entailment (MNLI, RTE)', 'premise', 'hypothesis', 'entail / contradict / neutral'),
             ('question answering (SQuAD)', 'question', 'passage', 'answer span'),
             ('one text (SST-2, CoLA, NER)', 'the text', '(nothing)', 'label or tags')]
    b = [text(20, 24, 'Pre-training saw pairs: [CLS] sentence A [SEP] sentence B [SEP]. Fine-tuning fills the same two slots.', 't-title')]
    for k, (name, a, bb, out) in enumerate(rows_):
        y = 48 + k * 52
        b += [text(20, y + 21, name, 't-note')]
        x = 222
        b += token(x, y, 48, '[CLS]', 'box', 32, 't-muted')
        b += token(x + 52, y, 96, a, 'box-1', 32, 't-tick')
        b += token(x + 152, y, 48, '[SEP]', 'box', 32, 't-muted')
        if bb == '(nothing)':
            b += token(x + 204, y, 96, 'empty', 'box-ghost', 32, 't-muted')
        else:
            b += token(x + 204, y, 96, bb, 'box-2', 32, 't-tick')
            b += token(x + 304, y, 48, '[SEP]', 'box', 32, 't-muted')
        b += [arrow(x + 358, y + 16, x + 376, y + 16), text(x + 380, y + 21, out, 't-tick')]
    b += [text(222 + 100, 268, 'segment A', 't-s1', 'middle'), text(222 + 252, 268, 'segment B', 't-s2', 'middle')]
    return svg(760, 282, 'Every task is written as a pair in the same input format BERT saw during pre-training: sentence A in segment A, sentence B in segment B. A single-text task leaves B empty.', b)


F['p4_inputs'] = inputs()


def cls_head():
    words = ['[CLS]', 'a', 'man', 'plays', '[SEP]', 'a', 'person', '...', '[SEP]']
    b = [text(20, 24, 'Sentence-level tasks read one vector: the final vector of [CLS]', 't-title')]
    parts, cx = row(40, 230, words, 64, 8, classes=['box-on'] + ['box'] * 8)
    b += parts
    b += [box(30, 140, 650, 56, 'box-1', 12), text(355, 166, 'BERT (12 layers, all fine-tuned)', 't-big', 'middle'),
          text(355, 184, 'every token looks at every other token', 't-tick', 'middle')]
    for x in cx:
        b += [arrow(x, 228, x, 198)]
    b += [arrow(cx[0], 138, cx[0], 112), box(cx[0] - 32, 80, 64, 30, 'box-on', 6), text(cx[0], 100, 'C', 't-math', 'middle'),
          text(cx[0] + 40, 92, '768 numbers', 't-muted'), text(cx[0] + 40, 106, '(pooler: dense + tanh)', 't-muted'),
          arrow(cx[0] + 34, 70, 300, 56),
          box(300, 36, 130, 40, 'box-3', 8), text(365, 54, 'W: K x 768', 't-note', 'middle'), text(365, 69, 'the only new weights', 't-muted', 'middle'),
          arrow(432, 56, 470, 56), box(470, 36, 100, 40, 'box', 8), text(520, 61, 'softmax', 't-note', 'middle'),
          arrow(572, 56, 600, 56)]
    for k, (lab, p) in enumerate([('entailment', 0.91), ('neutral', 0.07), ('contradiction', 0.02)]):
        y = 34 + k * 17
        b += [f'<rect class="s3" x="604" y="{y}" width="{p * 60:.1f}" height="11" rx="2"/>', text(668, y + 10, lab, 't-tick')]
    b += [text(740, 92, 'example probabilities', 't-muted', 'end')]
    return svg(760, 280, 'Sentence-level fine-tuning. The final vector of the [CLS] token goes through one new layer W (K rows, one per label) and a softmax. The probabilities on the right are an illustration.', b)


F['p4_cls_head'] = cls_head()


def swag():
    ctx = 'She opened the fridge and'
    ends = ['took out a carton of milk.', 'flew to the moon.', 'the fridge sang a song.', 'painted the ocean blue.']
    b = [text(20, 24, 'SWAG: one sequence per choice, one score per sequence, softmax across the four', 't-title')]
    for k, e in enumerate(ends):
        y = 46 + k * 50
        b += token(20, y, 46, '[CLS]', 'box', 32, 't-muted') + token(70, y, 190, ctx, 'box-1', 32, 't-tick') + token(264, y, 46, '[SEP]', 'box', 32, 't-muted')
        b += token(314, y, 180, e, 'box-2', 32, 't-tick')
        b += [arrow(498, y + 16, 528, y + 16), box(530, y + 2, 64, 28, 'box-1', 6), text(562, y + 21, 'BERT', 't-note', 'middle'),
              arrow(596, y + 16, 616, y + 16), text(620, y + 21, f'C{k + 1} . w = s{k + 1}', 't-tick')]
    b += [line(708, 60, 708, 216, 'edge'), arrow(708, 138, 724, 138), text(728, 134, 'soft-', 't-note'), text(728, 150, 'max', 't-note')]
    b += [text(20, 258, 'The same BERT runs four times. The only new weights: one vector w (768 numbers) and one bias.', 't-tick')]
    return svg(760, 272, 'SWAG fine-tuning. Each of the four endings is paired with the sentence and read by BERT separately. The [CLS] vector of each pair is multiplied by one learned vector w to give a score, and a softmax over the four scores picks the ending.', b)


F['p4_swag'] = swag()


def span_strip():
    d = J('part4_squad')['demo'][0]
    toks, ps, pe = d['tokens'], d['p_start'], d['p_end']
    i0 = 0
    for k, t in enumerate(toks):
        if t == 'called':
            i0 = k
    win = range(i0 + 1, i0 + 16)
    cw = 44
    b = [text(20, 22, f'Q: {d["question"]}', 't-title')]
    labels = [('P(start)', ps, 'cell'), ('P(end)', pe, 'cell')]
    for r, (lab, P, cls) in enumerate(labels):
        y = 40 + r * 46
        b.append(text(88, y + 26, lab, 't-tick', 'end'))
        for c, k in enumerate(win):
            x = 96 + c * cw
            v = P[k]
            b.append(f'<g class="mark"><title>{esc(toks[k])}: {v:.3f}</title><rect class="{cls}" x="{x + 1}" y="{y + 1}" width="{cw - 2}" height="38" rx="3" style="fill-opacity:{max(0.05, v):.3f}"/></g>')
            if v >= 0.01:
                b.append(text(x + cw / 2, y + 25, f'{v:.2f}', 't-cell' + (' on' if v > 0.55 else ''), 'middle'))
    for c, k in enumerate(win):
        x = 96 + c * cw + cw / 2
        b.append(text(x, 150, toks[k], 't-tick', 'end', f' transform="rotate(-45 {x:.1f} 150)"'))
    si, sj = d['i'], d['j']
    return svg(780, 236, 'Start and end probabilities over part of the passage, from the public SQuAD v1.1 checkpoint. The start of "Bidirectional" and the end of "transformers" get almost all the probability, so the best span is the expansion of BERT.', b)


def null_bars():
    d = J('part4_squad')['demo2']
    b = [text(20, 22, 'SQuAD 2.0 model: best span score minus the no-answer score', 't-title')]
    x0, mid, scale = 330, 540, 9
    b.append(line(mid, 36, mid, 36 + 44 * len(d), 'base-line'))
    for k, r in enumerate(d):
        y = 40 + k * 44
        v = r['diff']
        w = abs(v) * scale
        cls = 's3' if v > 0 else 's2'
        x = mid if v > 0 else mid - w
        q = r['question'] if len(r['question']) < 44 else r['question'][:42] + '...'
        b += [text(20, y + 19, q, 't-tick'), f'<g class="mark"><title>{esc(r["question"])}: {v:+.2f}</title><rect class="{cls}" x="{x:.1f}" y="{y + 4}" width="{w:.1f}" height="22" rx="3"/></g>',
              text(mid + (w + 6 if v > 0 else -w - 6), y + 19, f'{v:+.2f}', 't-val', 'start' if v > 0 else 'end')]
    y = 46 + 44 * len(d)
    b += [text(mid + 10, y, 'answer: a span wins', 't-s3'), text(mid - 10, y, 'abstain: [CLS] wins', 't-s2', 'end')]
    return svg(760, y + 14, 'For each question, the best span score minus the no-answer score. Positive: the model answers with a span. Negative: the no-answer option at [CLS] wins, and the model says there is no answer. Here the threshold tau is 0.', b)


def tau_curve():
    v2 = J('part4_squad')['v2_dev']
    sw = v2['sweep']
    xs = [r['tau'] for r in sw]
    W_, H_, L, T, B = 760, 300, 64, 30, 50
    pw, ph = W_ - L - 170, H_ - T - B
    lo = math.floor(min(r['f1'] for r in sw) / 5) * 5
    hi = math.ceil(max(r['f1'] for r in sw) / 5) * 5
    X = lambda v: L + (v - xs[0]) / (xs[-1] - xs[0]) * pw
    Y = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    b = []
    for t in range(lo, hi + 1, 1 if hi - lo <= 10 else 5):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, f'{t}', 't-tick', 'end')]
    for t in range(-6, 7, 2):
        b.append(text(X(t), T + ph + 20, f'{t:+d}' if t else '0', 't-tick', 'middle'))
    b.append(f'<polyline class="l1" points="' + ' '.join(f'{X(r["tau"]):.1f},{Y(r["f1"]):.1f}' for r in sw) + '"/>')
    best, at0 = v2['best'], v2['at0']
    b += [f'<circle class="s1 ring" cx="{X(best["tau"]):.1f}" cy="{Y(best["f1"]):.1f}" r="6"/>',
          text(X(best['tau']) + 10, Y(best['f1']) - 10, f'best τ {best["tau"]:+.2f}: F1 {best["f1"]:.2f}', 't-note'),
          f'<circle class="s2 ring" cx="{X(0):.1f}" cy="{Y(at0["f1"]):.1f}" r="5"/>',
          text(X(0) + 10, Y(at0['f1']) + 18, f'τ = 0: F1 {at0["f1"]:.2f}', 't-tick'),
          text(L + pw / 2, H_ - 10, 'threshold τ (answer only if the best span beats the no-answer score by more than τ)', 't-tick', 'middle'),
          text(L - 50, T - 12, 'SQuAD 2.0 dev F1', 't-tick')]
    return svg(W_, H_, 'SQuAD 2.0 dev F1 of the public checkpoint as the threshold tau changes. The paper picks tau on the dev set to maximise F1; this is that search.', b)


def curve():
    s = J('part4_sst2')
    c = [r for r in s['curve'] if 'train_loss' in r]
    W_, H_, L, T, B = 760, 300, 64, 30, 50
    pw, ph = W_ - L - 190, H_ - T - B
    X = lambda v: L + v / s['steps'] * pw
    Ya = lambda v: T + ph - (v - 0.80) / 0.15 * ph
    Yl = lambda v: T + ph - v / 0.6 * ph
    b = []
    for t in [0.80, 0.85, 0.90, 0.95]:
        b += [f'<line class="grid" x1="{L}" y1="{Ya(t):.1f}" x2="{L + pw}" y2="{Ya(t):.1f}"/>', text(L - 8, Ya(t) + 4, f'{t:.2f}', 't-tick', 'end')]
    for t in [0, 0.2, 0.4, 0.6]:
        b.append(text(L + pw + 8, Yl(t) + 4, f'{t:.1f}', 't-tick'))
    per = s['steps'] // s['epochs']
    for e in range(1, s['epochs']):
        b.append(f'<line class="base-line" x1="{X(e * per):.1f}" y1="{T}" x2="{X(e * per):.1f}" y2="{T + ph}"/>')
    for e in range(s['epochs']):
        b.append(text(X(e * per + per / 2), T + ph + 20, f'epoch {e + 1}', 't-tick', 'middle'))
    b.append(f'<polyline class="l2" points="' + ' '.join(f'{X(r["step"]):.1f},{Yl(r["train_loss"]):.1f}' for r in c) + '"/>')
    b.append(f'<polyline class="l1" points="' + ' '.join(f'{X(r["step"]):.1f},{Ya(max(0.80, r["dev_acc"])):.1f}' for r in c) + '"/>')
    for r in c:
        b.append(f'<g class="mark"><title>step {r["step"]}: dev accuracy {r["dev_acc"]:.4f}, train loss {r["train_loss"]:.4f}</title><circle class="s1 ring" cx="{X(r["step"]):.1f}" cy="{Ya(max(0.80, r["dev_acc"])):.1f}" r="3.5"/></g>')
    last = c[-1]
    b += [text(L + pw + 40, Ya(last['dev_acc']) + 4, f'dev accuracy {last["dev_acc"]:.4f}', 't-s1'),
          text(L + pw + 40, Yl(last['train_loss']) + 4, f'train loss {last["train_loss"]:.3f}', 't-s2'),
          text(L - 50, T - 12, 'dev accuracy', 't-tick'), text(L + pw + 8, T - 12, 'loss', 't-tick')]
    return svg(W_, H_, f'Our SST-2 fine-tuning run: dev accuracy (blue, left axis) and training loss (orange, right axis, averaged between evaluations) over {s["steps"]:,} steps.', b)


def mrpc():
    rs = []
    for lr in ['5e-5', '4e-5', '3e-5', '2e-5']:
        try:
            rs.append((lr, J(f'part4_mrpc_lr{lr}')['dev_acc'] * 100))
        except FileNotFoundError:
            pass
    best = max(rs, key=lambda r: r[1])[0]
    b = hbars(20, 50, rs, 380, row_h=30, cls='s1', label_w=150, fmt=lambda v: f'{v:.1f}%', vmax=100, title='MRPC dev accuracy, one run per learning rate', hi=best)
    b = [x.replace('>5e-5<', '>lr 5e-5<').replace('>4e-5<', '>lr 4e-5<').replace('>3e-5<', '>lr 3e-5<').replace('>2e-5<', '>lr 2e-5<') for x in b]
    b += [text(20, 50 + 30 * len(rs) + 22, 'the paper reports 86.7 for BERT-base (Table 5, dev); bars start at 0', 't-muted')]
    return svg(760, 50 + 30 * len(rs) + 36, 'Our MRPC grid: the same fine-tuning run with each of the paper\'s four learning rates. Like the paper, we would keep the best one on the dev set.', b)


for key, fn in [('p4_span', span_strip), ('p4_null', null_bars), ('p4_tau', tau_curve), ('p4_curve', curve), ('p4_mrpc', mrpc)]:
    try:
        F[key] = fn()
    except (FileNotFoundError, KeyError, ValueError) as e:
        print('skipped', key, '(missing results:', e, ')')


# ---------------------------------------------------------------- second pass: illustrated maths figures
from alammar import vec, matrix, attn_grid, frame, brace
M4 = json.load(open('results/part4_math.json'))


def glue_shapes():
    t = M4['glue_toy']
    b = [text(20, 24, 'The classification head, shape by shape (toy numbers: H = 4, K = 3)', 't-title')]
    b += [text(40, 64, 'C', 't-math')] + vec(60, 52, t['C'], 's1', 18, 3) + [text(60, 92, '1 × H', 't-tick')]
    b += [text(170, 70, '×', 't-note', 'middle')]
    m, W, H = matrix(195, 52, 4, 3, 's3', 18, 3, [list(r) for r in zip(*t['W'])], 'Wᵀ', 'H × K')
    b += m + [text(195 + W + 25, 70, '=', 't-note', 'middle')]
    x = 195 + W + 50
    b += vec(x, 52, t['logits'], 's2', 18, 3) + [text(x, 40, 'logits', 't-math'), text(x, 92, '1 × K', 't-tick')]
    x2 = x + 90
    b += [arrow(x2, 62, x2 + 60, 62, on=True), text(x2 + 30, 50, 'softmax', 't-tick', 'middle')]
    x3 = x2 + 75
    for k, (lg, p) in enumerate(zip(t['logits'], t['p'])):
        y = 46 + k * 22
        b += [f'<rect class="s1" x="{x3}" y="{y}" width="{p * 140:.1f}" height="16" rx="3"/>', text(x3 + p * 140 + 6, y + 13, f'class {k}: {p:.3f}', 't-val')]
    b += [text(20, 168, f'logits = C Wᵀ = [{t["logits"][0]:.2f}, {t["logits"][1]:.2f}, {t["logits"][2]:.2f}]   →   softmax = [{t["p"][0]:.3f}, {t["p"][1]:.3f}, {t["p"][2]:.3f}]', 't-tick'),
          text(20, 188, f'if the right class is 0: loss = −log {t["p"][0]:.3f} = {t["loss"]:.3f}.   In BERT-base: C has H = 768 numbers, W is K × 768, so K × 768 + K new weights.', 't-tick')]
    return svg(760, 204, 'The GLUE classification head with toy numbers: the 4-number vector C times the transpose of the 3 by 4 matrix W gives three logits, and softmax turns them into probabilities 0.867, 0.032 and 0.101.', b)
F['p4_glue_shapes'] = glue_shapes()


def pack_vs_cross():
    q = ['who', 'sat', '?']
    p_ = ['a', 'cat', 'sat', '.']
    b = [text(20, 24, 'Two ways to let a question and a passage look at each other', 't-title')]
    toks = ['[CLS]'] + q + ['[SEP]'] + p_ + ['[SEP]']
    nq = 1 + len(q) + 1
    seg = lambda i: 'A' if i < nq else 'B'
    cls_fn = lambda i, j: 'cell' if seg(i) == seg(j) else 's2'
    b += attn_grid(20, 100, toks, toks, lambda i, j: True, cell=36, title='BERT: one square self-attention over the packed pair', cls_fn=cls_fn, label_w=58)[0]
    b += attn_grid(470, 100, q, p_, lambda i, j: True, cell=34, title='Older models: a separate cross-attention', label_w=46)[0]
    b += [text(470, 250, 'rows: question words; columns: passage words.', 't-tick'), text(470, 268, 'A separate block, built on top of', 't-tick'),
          text(470, 286, 'two independently encoded texts.', 't-tick'),
          f'<rect class="cell" x="20" y="490" width="12" height="12" rx="2"/>', text(38, 500, 'within one text', 't-tick'),
          f'<rect class="s2" x="160" y="490" width="12" height="12" rx="2"/>', text(178, 500, 'across the two texts (question ↔ passage): cross attention, for free', 't-tick')]
    return svg(760, 514, 'Left: BERT packs the question and passage into one sequence, so its square self-attention already contains question-to-passage and passage-to-question attention. Right: older models encoded the two texts separately and added a rectangular cross-attention between them.', b)
F['p4_pack_vs_cross'] = pack_vs_cross()


def span_grid():
    t = M4['span_toy']
    n = 4
    b = [text(20, 24, 'Span scores score(i, j) = S·Tᵢ + E·Tⱼ, toy passage of 4 tokens', 't-title')]
    cell = 46
    gx, gy = 150, 80
    b += [text(gx + cell * n / 2, gy - 30, 'end j', 't-muted', 'middle'), text(gx - 70, gy + cell * n / 2, 'start i', 't-muted', 'middle')]
    for j in range(n):
        b.append(text(gx + j * cell + cell / 2, gy - 10, f'j={j}  E·T={t["end"][j]}', 't-tick', 'middle') if False else text(gx + j * cell + cell / 2, gy - 10, f'{j}', 't-tick', 'middle'))
    best = tuple(t['best']); free = tuple(t['free'])
    for i in range(n):
        b.append(text(gx - 10, gy + i * cell + cell / 2 + 4, f'{i}', 't-tick', 'end'))
        for j in range(n):
            v = t['M'][i][j]; x, y = gx + j * cell, gy + i * cell
            ok = j >= i
            cls = 'cell' if ok else 'cell-masked'
            op = 0.15 + 0.8 * (v / 6) if ok else 1
            b.append(f'<rect class="{cls}" x="{x + 2}" y="{y + 2}" width="{cell - 4}" height="{cell - 4}" rx="4" style="fill-opacity:{op:.2f}"/>')
            if (i, j) == best:
                b.append(f'<rect class="box-on" x="{x}" y="{y}" width="{cell}" height="{cell}" rx="5" style="fill:none"/>')
            b.append(text(x + cell / 2, y + cell / 2 + 5, f'{v:.1f}', 't-cell' + (' on' if ok and v > 3 else ''), 'middle'))
    x0 = gx + n * cell + 40
    b += [text(x0, 90, f'S·T = {t["start"]}', 't-tick'), text(x0, 110, f'E·T = {t["end"]}', 't-tick'),
          text(x0, 140, f'highest of all: i={free[0]}, j={free[1]}, score 5.8', 't-tick'), text(x0, 158, 'but it ends before it starts: crossed out', 't-tick'),
          text(x0, 188, f'best with j ≥ i: i={best[0]}, j={best[1]}, score 4.5', 't-val'),
          text(x0, 218, f'training loss if (2, 3) is right:', 't-tick'), text(x0, 236, f'−log {t["p_start"][2]:.3f} − log {t["p_end"][3]:.3f} = {t["loss"]:.3f}', 't-tick')]
    return svg(760, 280, 'A 4 by 4 table of span scores. The highest score sits below the diagonal, where the end comes before the start, so it is not allowed. The best allowed span starts at 2 and ends at 3 with score 4.5.', b)
F['p4_span_grid'] = span_grid()


def f1_fig():
    rows = M4['emf1'][1:3]
    b = [text(20, 24, 'Exact match and F1: word overlap after normalising (lowercase, no punctuation, no a/an/the)', 't-title')]
    y = 60
    for r in rows:
        pw, gw = r['pn'].split(), r['gn'].split()
        b.append(text(20, y, 'prediction', 't-muted'))
        x = 110
        for w in pw:
            W = 9 * len(w) + 18; b += [box(x, y - 16, W, 24, 'box-1' if w in gw else 'box', 5), text(x + W / 2, y, w, 't-tick', 'middle')]; x += W + 6
        b.append(text(20, y + 32, 'gold', 't-muted'))
        x = 110
        for w in gw:
            W = 9 * len(w) + 18; b += [box(x, y + 16, W, 24, 'box-1' if w in pw else 'box', 5), text(x + W / 2, y + 32, w, 't-tick', 'middle')]; x += W + 6
        b.append(text(110, y + 64, f'shared {r["same"]}:  precision {r["same"]}/{len(pw)} = {r["P"]:.3f},  recall {r["same"]}/{len(gw)} = {r["R"]:.3f},  F1 = {r["F1"]:.3f},  EM = {int(r["EM"])}', 't-val'))
        y += 104
    return svg(760, y - 16, 'Two predictions scored against the gold answer. Shared words are highlighted. A prediction missing two of five gold words gets precision 1, recall 0.6, F1 0.75 and exact match 0.', b)
try:
    F['p4_f1'] = f1_fig()
except KeyError as e:
    print('p4_f1 skipped', e)


def new_weights():
    P = M4['params']
    items = [(k, v) for k, v in P['heads'].items()]
    b = [text(20, 24, 'New weights added for each task, next to the 109.5 million that are fine-tuned', 't-title')]
    y = 54
    for k, v in items:
        b += [text(20, y + 14, k, 't-tick'), f'<rect class="s2" x="250" y="{y}" width="{max(3, v / 2307 * 60):.1f}" height="20" rx="3"/>', text(320, y + 14, f'{v:,}', 't-val')]
        y += 30
    b += [text(20, y + 14, 'BERT-base itself (all trained)', 't-tick'), f'<rect class="s1" x="250" y="{y}" width="480" height="20" rx="3"/>', text(260, y + 14, f'{P["trained"] - P["new"]:,}', 't-cell on')]
    b.append(text(20, y + 50, 'The new head is about 0.001% to 0.002% of the model. Fine-tuning still updates every one of the 109.5 million weights.', 't-muted'))
    return svg(760, y + 62, 'Number of new weights per task head (769 to 2,307) compared with the 109.5 million weights of BERT-base, which are all updated during fine-tuning.', b)
F['p4_new_weights'] = new_weights()

json.dump(F, open('results/figs_part4.json', 'w'))
print('figures:', ', '.join(F))
