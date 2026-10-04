"""Figures for Part 5 (ablations). Table numbers are typed in from the paper (Tables 5, 6, 7, 8);
measured numbers come from results/part5.json and results/part5_ner.json. Writes results/figs_part5.json."""
import json
from figlib import svg, text, box, arrow, esc
from bertfig import token, line, hbars

R = json.load(open('results/part5.json'))
import os
NER = json.load(open('results/part5_ner.json')) if os.path.exists('results/part5_ner.json') else None
F = {}


def table5():
    tasks = ['MNLI-m', 'QNLI', 'MRPC', 'SST-2', 'SQuAD F1']
    rows = [('BERT-base', [84.4, 88.4, 86.7, 92.7, 88.5], 's1'), ('No NSP', [83.9, 84.9, 86.5, 92.6, 87.9], 's3'),
            ('LTR & No NSP', [82.1, 84.3, 77.5, 92.1, 77.8], 's2'), ('+ BiLSTM', [82.1, 84.1, 75.7, 91.6, 84.9], 's4')]
    L, T, B, W, Hh = 56, 50, 40, 760, 330
    ph = Hh - T - B
    lo, hi = 70, 95
    Y = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    b = []
    for t in range(lo, hi + 1, 5):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{W - 10}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, str(t), 't-tick', 'end')]
    gw = (W - 10 - L) / len(tasks)
    bw = 26
    for j, task in enumerate(tasks):
        x0 = L + j * gw + (gw - 4 * bw - 3 * 4) / 2
        for i, (name, vals, cls) in enumerate(rows):
            v = vals[j]
            x = x0 + i * (bw + 4)
            b.append(f'<g class="mark"><title>{esc(name)}, {task}: {v}</title><rect class="{cls}" x="{x:.1f}" y="{Y(v):.1f}" width="{bw}" height="{Y(lo) - Y(v):.1f}" rx="3"/></g>')
            b.append(text(x + bw / 2, Y(v) - 5, f'{v:g}', 't-cell', 'middle'))
        b.append(text(L + j * gw + gw / 2, Hh - 14, task, 't-tick', 'middle'))
    for i, (name, _, cls) in enumerate(rows):
        x = L + i * 170
        b += [f'<rect class="{cls}" x="{x}" y="14" width="12" height="12" rx="2"/>', text(x + 18, 24, name, 't-note')]
    b.append(f'<line class="axis" x1="{L}" y1="{Y(lo):.1f}" x2="{W - 10}" y2="{Y(lo):.1f}"/>')
    return svg(W, Hh, 'Table 5 of the BERT paper as a bar chart: Dev set scores of BERT-base, No NSP, LTR and No NSP, and LTR plus BiLSTM on five tasks. The axis starts at 70.', b)


F['p5_table5'] = table5()


def ladder():
    steps = [('BERT-base', 'MLM + NSP', 'box-1'), ('No NSP', 'MLM only', 'box-3'), ('LTR & No NSP', 'left-to-right only', 'box-2'),
             ('+ BiLSTM', 'LTR, BiLSTM on top', 'box-4')]
    changes = [('drop', 'NSP'), ('MLM to', 'LTR'), ('add a', 'BiLSTM')]
    b = [text(14, 18, 'Each row of Table 5 changes one thing from the row before', 't-title')]
    for i, (name, what, cls) in enumerate(steps):
        x = 14 + i * 186
        b += [box(x, 34, 132, 58, cls, 10), text(x + 66, 59, name, 't-note', 'middle'), text(x + 66, 78, what, 't-tick', 'middle')]
        if i < 3:
            cx = x + 159
            b += [arrow(x + 136, 63, x + 182, 63), text(cx, 52, changes[i][0], 't-muted', 'middle'), text(cx, 84, changes[i][1], 't-muted', 'middle')]
    b.append(text(14, 122, 'Same pre-training data, same fine-tuning recipe, same hyperparameters: only the named change differs.', 't-tick'))
    return svg(760, 134, 'The four models of Table 5 in order. Each one changes a single thing from the one before: drop NSP, then switch to a left-to-right model, then add a BiLSTM.', b)


F['p5_ladder'] = ladder()


def size():
    rows = R['table6']
    W, Hh, L, T, B, right = 760, 350, 56, 40, 90, 120
    pw, ph = W - L - right, Hh - T - B
    lo, hi = 76, 96
    X = lambda i: L + 30 + i * (pw - 60) / (len(rows) - 1)
    Y = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    b = []
    for t in range(lo, hi + 1, 4):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, str(t), 't-tick', 'end')]
    series = [('SST-2', 'sst2', 'l3', 's3'), ('MRPC', 'mrpc', 'l2', 's2'), ('MNLI-m', 'mnli', 'l1', 's1')]
    for name, k, lc, dc in series:
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(i):.1f},{Y(r[k]):.1f}' for i, r in enumerate(rows)) + '"/>')
        for i, r in enumerate(rows):
            b.append(f'<g class="mark"><title>{name}, L={r["L"]} H={r["H"]} A={r["A"]}: {r[k]}</title><circle class="{dc} ring" cx="{X(i):.1f}" cy="{Y(r[k]):.1f}" r="5"/></g>')
        last = rows[-1][k]
        b += [f'<rect class="{dc}" x="{X(5) + 16}" y="{Y(last) - 5:.1f}" width="10" height="10" rx="2"/>', text(X(5) + 32, Y(last) + 4, f'{name} {last}', 't-note')]
    for i, r in enumerate(rows):
        b += [text(X(i), T + ph + 20, f'L={r["L"]}, H={r["H"]}', 't-tick', 'middle'), text(X(i), T + ph + 36, f'A={r["A"]}', 't-tick', 'middle'),
              text(X(i), T + ph + 54, f'{r["params"] / 1e6:.0f}M params', 't-muted', 'middle'),
              text(X(i), T + ph + 70, f'LM ppl {r["ppl"]}', 't-muted', 'middle')]
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    b.append(text(L - 44, T - 16, 'Dev accuracy (Table 6, average of 5 fine-tuning runs)', 't-tick'))
    return svg(W, Hh, 'Table 6 as a chart: Dev accuracy on MNLI-m, MRPC and SST-2 rises with every bigger model, from 3 layers (46M parameters) to 24 layers (335M). Parameter counts are measured; accuracies are from the paper.', b)


F['p5_size'] = size()


def layers():
    choices = [('Embeddings', [0], 'one'), ('Second-to-last hidden', [11], 'one'), ('Last hidden', [12], 'one'),
               ('Weighted sum last four', [9, 10, 11, 12], 'sum'), ('Concat last four', [9, 10, 11, 12], 'cat'),
               ('Weighted sum all 12 layers', list(range(1, 13)), 'sum')]
    paper = [91.0, 95.6, 94.9, 95.9, 96.1, 95.5]
    x0, y0, c, rh = 214, 52, 30, 34
    b = [text(x0 + 6.5 * c, 22, 'which of the 13 hidden states each feature choice uses', 't-title', 'middle')]
    for j in range(13):
        b.append(text(x0 + j * c + c / 2, y0 - 8, 'E' if j == 0 else str(j), 't-tick', 'middle'))
    for i, (name, use, how) in enumerate(choices):
        y = y0 + i * rh
        b.append(text(x0 - 10, y + 20, name, 't-tick', 'end'))
        for j in range(13):
            cls = {'one': 'box-1', 'sum': 'box-3', 'cat': 'box-2'}[how] if j in use else 'box-ghost'
            b.append(box(x0 + j * c + 2, y + 4, c - 4, rh - 8, cls, 4))
        how_txt = {'one': 'one layer', 'sum': f'weighted sum of {len(use)}', 'cat': '4 x 768 = 3,072 numbers'}[how]
        b.append(text(x0 + 13 * c + 12, y + 16, how_txt, 't-tick'))
        b.append(text(x0 + 13 * c + 12, y + 30, f'paper Dev F1 {paper[i]}', 't-muted'))
    b.append(text(x0, y0 + 6 * rh + 22, 'E = embedding output (before any layer); 1 to 12 = outputs of the 12 layers', 't-muted'))
    return svg(760, y0 + 6 * rh + 34, 'The six feature choices of Table 7, drawn as which hidden states they read: the embeddings alone, one layer, a weighted sum, or the last four layers glued together.', b)


F['p5_layers'] = layers()


def ner():
    names = list(NER['results'])
    paper = {'Embeddings': 91.0, 'Second-to-last hidden': 95.6, 'Last hidden': 94.9, 'Weighted sum last four': 95.9,
             'Concat last four': 96.1, 'Weighted sum all 12 layers': 95.5}
    ours = {n: NER['results'][n]['mean'] for n in names}
    lo, hi = 75, 100
    x0, bw, rh = 210, 440, 40
    Xv = lambda v: x0 + (v - lo) / (hi - lo) * bw
    b = [f'<rect class="s1" x="{x0}" y="10" width="12" height="12" rx="2"/>', text(x0 + 18, 20, 'paper, Table 7 (BiLSTM, document context, 5 runs)', 't-note'),
         f'<rect class="s2" x="{x0}" y="30" width="12" height="12" rx="2"/>', text(x0 + 18, 40, 'our smaller re-run (5,000 sentences, 3 runs)', 't-note')]
    y0 = 58
    for t in range(lo, hi + 1, 5):
        b += [f'<line class="grid" x1="{Xv(t):.1f}" y1="{y0}" x2="{Xv(t):.1f}" y2="{y0 + len(names) * rh}"/>',
              text(Xv(t), y0 + len(names) * rh + 16, str(t), 't-tick', 'middle')]
    for i, n in enumerate(names):
        y = y0 + i * rh
        b.append(text(x0 - 10, y + 22, n, 't-tick', 'end'))
        for k, (v, cls) in enumerate([(paper[n], 's1'), (ours[n], 's2')]):
            yy = y + 5 + k * 15
            b.append(f'<g class="mark"><title>{esc(n)}: {v}</title><rect class="{cls}" x="{x0}" y="{yy}" width="{Xv(v) - x0:.1f}" height="13" rx="3"/></g>')
            b.append(text(Xv(v) + 6, yy + 11, f'{v:.1f}' if cls == 's1' else f'{v:.2f}', 't-val'))
    b.append(text(x0 + bw / 2, y0 + len(names) * rh + 34, 'entity-level F1 on the CoNLL-2003 Dev set (axis starts at 75)', 't-tick', 'middle'))
    return svg(760, y0 + len(names) * rh + 44, 'Dev F1 for the six feature choices: the paper (Table 7) and our smaller re-run. Absolute numbers differ, but in both the embeddings alone are clearly worst and the deep layers are far better.', b)


if NER:
    F['p5_ner'] = ner()


def table8():
    rows = [(80, 10, 10, 84.2, 95.4, 94.9), (100, 0, 0, 84.3, 94.9, 94.0), (80, 0, 20, 84.1, 95.2, 94.6),
            (80, 20, 0, 84.4, 95.2, 94.7), (0, 20, 80, 83.7, 94.8, 94.6), (0, 0, 100, 83.6, 94.9, 94.6)]
    x0, bw, rh, y0 = 20, 300, 38, 66
    b = [text(x0, 20, 'what happens to a chosen token', 't-title'), text(450, 20, 'Dev results (Table 8)', 't-title'),
         f'<rect class="s1" x="{x0}" y="32" width="12" height="12" rx="2"/>', text(x0 + 18, 42, 'MASK', 't-note'),
         f'<rect class="s3" x="{x0 + 80}" y="32" width="12" height="12" rx="2"/>', text(x0 + 98, 42, 'SAME', 't-note'),
         f'<rect class="s2" x="{x0 + 160}" y="32" width="12" height="12" rx="2"/>', text(x0 + 178, 42, 'RND', 't-note')]
    for k, h in enumerate(['MNLI', 'NER fine-tune', 'NER feature-based']):
        b.append(text(470 + k * 100, 52, h, 't-tick', 'middle'))
    for i, (m, s, r, mnli, nf, nfb) in enumerate(rows):
        y = y0 + i * rh
        x = x0
        for v, cls in [(m, 's1'), (s, 's3'), (r, 's2')]:
            if v:
                w = bw * v / 100
                b.append(f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="24" rx="3"/>')
                if v >= 10:
                    b.append(text(x + w / 2, y + 16, f'{v}%', 't-cell on', 'middle'))
                x += w
        if i == 0:
            b.append(text(x0 + bw + 10, y + 16, 'BERT', 't-tick t-strong'))
        for k, v in enumerate([mnli, nf, nfb]):
            best = {0: 84.4, 1: 95.4, 2: 94.9}[k]
            worst = {0: 83.6, 1: 94.8, 2: 94.0}[k]
            cls = 't-val' if v not in (best, worst) else ('t-val t-on' if v == best else 't-val t-bad')
            b.append(text(470 + k * 100, y + 16, f'{v}', cls, 'middle'))
    b.append(text(x0, y0 + 6 * rh + 14, 'highlighted: best and worst value in each column', 't-muted'))
    return svg(760, y0 + 6 * rh + 26, 'Table 8 as a picture: six masking recipes (share of chosen tokens replaced by [MASK], kept the same, or replaced by a random token) and their Dev scores.', b)


F['p5_table8'] = table8()

json.dump(F, open('results/figs_part5.json', 'w'))
print('figures:', ', '.join(F))
