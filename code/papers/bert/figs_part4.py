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

json.dump(F, open('results/figs_part4.json', 'w'))
print('figures:', ', '.join(F))
