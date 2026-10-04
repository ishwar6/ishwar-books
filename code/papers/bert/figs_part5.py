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

# ---------------------------------------------------------------------------------------------------------------
# Second pass: illustrated figures (token blocks, vector strips, attention grids, shapes), from results/part5_math.json
from alammar import vec, vec_len, matrix, attn_grid, frame, brace

M = json.load(open('results/part5_math.json'))


def four_models():
    """The four models of Table 5: which positions each token may attend to, and which heads are trained."""
    toks = ['[CLS]', 'my', 'dog', 'is', '[SEP]']
    panels = [('BERT-base', 'masked LM + NSP', True, ['MLM', 'NSP on C']),
              ('No NSP', 'masked LM only', True, ['MLM']),
              ('LTR & No NSP', 'left-to-right LM, like GPT', False, ['next word']),
              ('+ BiLSTM', 'the LTR model + a BiLSTM', False, ['next word'])]
    b = []
    for k, (name, sub, bi, heads) in enumerate(panels):
        px, py = 10 + (k % 2) * 380, 10 + (k // 2) * 276
        b += [box(px, py, 366, 262, 'box-1' if bi else 'box-2', 14), text(px + 16, py + 26, name, 't-title'), text(px + 16, py + 44, sub, 't-tick')]
        g, gw, gh = attn_grid(px + 8, py + 100, toks, toks, (lambda i, j: True) if bi else (lambda i, j: j <= i), cell=30,
                              row_title='', col_title='', label_w=46)
        b += g
        b.append(text(px + 16, py + 68, 'who may look at whom', 't-muted'))
        hx = px + 230
        b.append(text(hx, py + 68, 'pre-training heads', 't-muted'))
        for i, h in enumerate(heads):
            b += [box(hx, py + 86 + i * 34, 120, 26, 'box-on', 7), text(hx + 60, py + 104 + i * 34, h, 't-note', 'middle')]
        if name == 'No NSP':
            b += [box(hx, py + 120, 120, 26, 'box-ghost', 7), text(hx + 60, py + 138, 'NSP removed', 't-muted', 'middle'),
                  line(hx + 6, py + 133, hx + 114, py + 133, 'bad-line')]
        if name == '+ BiLSTM':
            b += [box(hx, py + 128, 120, 44, 'box-4', 7), text(hx + 60, py + 147, 'BiLSTM', 't-note', 'middle'),
                  text(hx + 60, py + 163, 'new, random start', 't-muted', 'middle')]
        if not bi:
            b += [text(hx, py + 210, 'left-only mask kept', 't-muted'), text(hx, py + 226, 'during fine-tuning too', 't-muted')]
    return svg(760, 562, 'The four models of Table 5. BERT-base and No NSP let every token attend to every token; No NSP drops the next-sentence head. LTR and No NSP lets each token see only itself and the tokens to its left, like GPT. Plus BiLSTM adds a randomly initialised BiLSTM on top of that model during fine-tuning.', b)


F['p5_four_models'] = four_models()


def ltr_squad():
    """What the vector of the answer's first word can see, left-to-right versus bidirectional."""
    q = ['who', 'painted', 'it', '?']
    p_ = ['the', 'mona', 'lisa', 'was', 'painted', 'by', 'leonardo', 'da', 'vinci']
    toks = ['[CLS]'] + q + ['[SEP]'] + p_ + ['[SEP]']
    G, x0 = 3, 14
    Ws = [max(30, 7 * len(t) + 10) for t in toks]
    X = [x0 + sum(Ws[:i]) + G * i for i in range(len(toks))]
    focus = toks.index('leonardo')
    b = [text(14, 22, 'Question: who painted it?   Passage: the mona lisa was painted by leonardo da vinci', 't-title')]
    for r, (name, see_right) in enumerate([('Left-to-right model', False), ('Bidirectional model (BERT)', True)]):
        y = 60 + r * 120
        b.append(text(14, y, name, 't-note'))
        for i, t in enumerate(toks):
            x, W = X[i], Ws[i]
            vis = i <= focus or see_right
            ans = 'leonardo' == t or t in ('da', 'vinci')
            cls = 'box-on' if i == focus else ('box-3' if ans and vis else ('box' if vis else 'box-ghost'))
            b += [box(x, y + 12, W, 30, cls, 5), text(x + W / 2, y + 31, t, 't-tick' if vis else 't-muted', 'middle')]
            if i != focus and vis:
                fx = X[focus] + Ws[focus] / 2
                b.append(f'<path class="{"edge-on" if i > focus else "edge"}" d="M{x + W / 2:.1f},{y + 44:.1f} Q{(x + W / 2 + fx) / 2:.1f},{y + 84:.1f} {fx:.1f},{y + 46:.1f}" style="stroke-opacity:0.7"/>')
        if not see_right:
            b += brace(X[focus + 1], X[-1] + Ws[-1], y + 56, 'hidden from "leonardo"', 'edge')
        else:
            b.append(text(X[focus + 1], y + 96, 'teal arrows: right-side words, only BERT can use them', 't-muted'))
    b.append(text(14, 268, 'Without "da vinci", the vector of "leonardo" cannot tell whether a longer name follows, or where the answer should end.', 't-muted'))
    b.append(text(14, 290, 'The highlighted box is the token whose vector must say "the answer starts here". Green: the true answer span.', 't-muted'))
    return svg(760, 302, 'What the vector of the answer word leonardo can see. In a left-to-right model it sees only the tokens up to itself, so it cannot know that da vinci follows. In BERT it sees the whole question and passage, in every layer.', b)


F['p5_ltr_squad'] = ltr_squad()


def rtl_qa():
    """Can a passage token see the question? LTR yes, RTL no, BERT yes."""
    rows = ['painted', 'by', 'leonardo']
    cols = ['who', 'painted', 'it', '?']
    b = [text(14, 20, 'Rows: passage tokens. Columns: question tokens. The question comes first in the input.', 't-title')]
    for k, (name, ok, note) in enumerate([('left-to-right', True, 'question is on the left: visible'),
                                          ('right-to-left', False, 'question is on the left: never visible'),
                                          ('BERT', True, 'every token sees every token')]):
        x = 14 + k * 250
        g, gw, gh = attn_grid(x, 100, rows, cols, lambda i, j: ok, cell=34, title=name, row_title='', col_title='key', label_w=70)
        b += g
        b.append(text(x, 100 + gh + 24, note, 't-bad' if not ok else 't-tick'))
    return svg(760, 236, 'Attention from passage tokens to question tokens. A left-to-right model reads the question before the passage, so passage tokens can use it. A right-to-left model reads the passage first and never reaches the question to its left. BERT sees both.', b)


F['p5_rtl_qa'] = rtl_qa()


def elmo_concat():
    c = M['concat_cost']
    b = [text(14, 22, 'ELMo-style: two one-way models, glued at the top', 't-title'), text(420, 22, 'BERT: one model, both sides in every layer', 't-title')]
    toks = ['the', 'kid', 'smiles']
    # left: an LTR stack and an RTL stack
    for s, (lab, cls, edge, x0) in enumerate([('LTR', 'box-1', 'edge-1', 22), ('RTL', 'box-2', 'edge-2', 202)]):
        for l in range(3):
            y = 186 - l * 44
            for i in range(3):
                b.append(box(x0 + i * 54, y, 44, 26, cls, 5))
                if l < 2:
                    for j in range(3):
                        if (lab == 'LTR' and j <= i) or (lab == 'RTL' and j >= i):
                            b.append(line(x0 + j * 54 + 22, y, x0 + i * 54 + 22, y - 18, edge, extra=' style="stroke-opacity:0.55"'))
        for i, t in enumerate(toks):
            b.append(text(x0 + i * 54 + 22, 236, t, 't-tick', 'middle'))
        b.append(text(x0 + 76, 84, f'{lab} stack', 't-note', 'middle'))
    b += vec(40, 52, 6, 's1', 11) + vec(40 + vec_len(6, 11) + 4, 52, 6, 's2', 11)
    b += [text(40, 44, 'smiles = [LTR ; RTL] = 1,536 numbers', 't-muted'), line(76, 98, 76, 70, 'edge', 'ah'), line(256, 98, 160, 70, 'edge', 'ah')]
    # right: one bidirectional stack
    x0 = 470
    for l in range(3):
        y = 186 - l * 44
        for i in range(3):
            b.append(box(x0 + i * 70, y, 56, 26, 'box-3', 5))
            if l < 2:
                for j in range(3):
                    b.append(line(x0 + j * 70 + 28, y, x0 + i * 70 + 28, y - 18, 'edge-3', extra=' style="stroke-opacity:0.55"'))
    for i, t in enumerate(toks):
        b.append(text(x0 + i * 70 + 28, 236, t, 't-tick', 'middle'))
    b += vec(x0 + 30, 52, 6, 's3', 11) + [text(x0 + 30, 44, 'smiles = 768 numbers', 't-muted'), line(x0 + 168, 98, x0 + 70, 70, 'edge', 'ah')]
    # measured cost
    y = 270
    b.append(text(14, y, f'Measured, BERT-base size, batch of 8 x 512 tokens on an Apple GPU (bert_part5_math.py):', 't-tick'))
    mx = max(c['ms_two'], 1)
    for k, (lab, ms, prm, cls) in enumerate([('LTR + RTL', c['ms_two'], c['params_two'], 's2'), ('one bidirectional', c['ms_one'], c['params_one'], 's3')]):
        yy = y + 14 + k * 30
        b += [text(150, yy + 15, lab, 't-tick', 'end'), f'<rect class="{cls}" x="160" y="{yy + 3}" width="{300 * ms / mx:.1f}" height="16" rx="3"/>',
              text(166 + 300 * ms / mx, yy + 16, f'{ms} ms, {prm / 1e6:.1f}M parameters', 't-val')]
    return svg(760, 346, 'Left: ELMo-style, a left-to-right stack and a right-to-left stack whose top vectors are concatenated; inside each stack information flows one way only. Right: BERT, one stack where every layer mixes both sides. Measured: two stacks need twice the parameters and twice the time.', b)


F['p5_elmo_concat'] = elmo_concat()


def gpt_vs_bert():
    rows = [('pre-training text', 'BooksCorpus, 800M words', 'BooksCorpus + Wikipedia, 3,300M words', 800, 3300),
            ('batch size', '32,000 words per step, 1M steps', '128,000 words per step, 1M steps', 32, 128)]
    b = [text(14, 22, 'Appendix A.4: the four non-architecture differences between OpenAI GPT and BERT', 't-title'),
         f'<rect class="s2" x="14" y="36" width="12" height="12" rx="2"/>', text(32, 46, 'OpenAI GPT', 't-note'),
         f'<rect class="s1" x="130" y="36" width="12" height="12" rx="2"/>', text(148, 46, 'BERT', 't-note')]
    y = 64
    for name, g, bb, gv, bv in rows:
        mx = max(gv, bv)
        b += [text(14, y + 14, name, 't-note'),
              f'<rect class="s2" x="190" y="{y}" width="{330 * gv / mx:.1f}" height="18" rx="3"/>', text(196 + 330 * gv / mx, y + 14, g, 't-tick'),
              f'<rect class="s1" x="190" y="{y + 24}" width="{330 * bv / mx:.1f}" height="18" rx="3"/>', text(196 + 330 * bv / mx if bv < mx else 190 + 330 + 6, y + 38, bb, 't-tick')]
        y += 64
    b.append(text(14, y + 14, '[CLS], [SEP], A/B', 't-note'))
    b += [box(190, y, 250, 22, 'box-2', 5), text(200, y + 15, 'GPT: added only at fine-tuning', 't-tick'),
          box(190, y + 28, 250, 22, 'box-1', 5), text(200, y + 43, 'BERT: learned during pre-training', 't-tick')]
    y += 64
    b.append(text(14, y + 14, 'fine-tuning rate', 't-note'))
    b += [box(190, y, 250, 22, 'box-2', 5), text(200, y + 15, 'GPT: always 5e-5', 't-tick'),
          box(190, y + 28, 250, 22, 'box-1', 5), text(200, y + 43, 'BERT: best of 5e-5, 3e-5, 2e-5 on Dev', 't-tick')]
    y += 70
    b += [box(14, y, 732, 46, 'box-on', 10),
          text(26, y + 19, 'The "LTR & No NSP" row of Table 5 = GPT\'s objective + all four BERT advantages above.', 't-note'),
          text(26, y + 37, 'So its gap to BERT-base (MRPC 77.5 vs 86.7, SQuAD 77.8 vs 88.5) comes from the masked LM and NSP.', 't-tick')]
    return svg(760, y + 58, 'The four differences between GPT and BERT besides the architecture: BERT used about four times more text, a four times larger batch, learned its special tokens during pre-training, and tuned the fine-tuning learning rate per task. The LTR and No NSP ablation keeps all four and still loses.', b)


F['p5_gpt_vs_bert'] = gpt_vs_bert()


def params_terms():
    rows = M['params']
    parts = [('embeddings', 'embeddings', 's1'), ('attention', 'attention', 's2'), ('feed-forward', 'feed_forward', 's3'),
             ('LayerNorm + pooler', None, 's4')]
    mx = max(r['total'] for r in rows)
    x0, bw, rh, y0 = 150, 470, 34, 58
    b = [text(14, 22, 'Where the parameters of each Table 6 model live (counted, term by term)', 't-title')]
    for k, (lab, _, cls) in enumerate(parts):
        b += [f'<rect class="{cls}" x="{14 + k * 160}" y="34" width="12" height="12" rx="2"/>', text(32 + k * 160, 44, lab, 't-note')]
    for i, r in enumerate(rows):
        y = y0 + i * rh
        b.append(text(x0 - 10, y + 17, f'L={r["L"]}, H={r["H"]}, A={r["A"]}', 't-tick', 'end'))
        x = x0
        for lab, key, cls in parts:
            v = r[key] if key else r['layer_norms'] + r['pooler']
            w = bw * v / mx
            b.append(f'<g class="mark"><title>{lab}: {v:,}</title><rect class="{cls}" x="{x:.1f}" y="{y + 4}" width="{max(w, 1.5):.1f}" height="20"/></g>')
            x += w
        b.append(text(x + 8, y + 18, f'{r["total"] / 1e6:.1f}M', 't-val'))
    y = y0 + len(rows) * rh + 14
    b.append(text(14, y, 'Each layer adds 12H² + 13H numbers: 7.09M at H = 768, 12.60M at H = 1024. The embedding tables do not grow with L.', 't-muted'))
    return svg(760, y + 12, 'Stacked bars for the six model sizes of Table 6: embeddings, attention, feed-forward, and the small LayerNorm and pooler terms. The feed-forward networks are always two thirds of the layer parameters; the embeddings are a large share only in small models.', b)


F['p5_params_terms'] = params_terms()


def ppl():
    r = M['ppl'][0]
    rows = r['bert']
    b = [text(14, 22, 'she opened the [MASK] with her [MASK] and walked into the [MASK] .', 't-title')]
    cols = [(14, '1. probability of the true word'), (276, '2. surprise = minus log p'), (520, '3. average, then e to that')]
    for x, t in cols:
        b.append(text(x, 52, t, 't-note'))
    mxn = max(q['nll'] for q in rows)
    for i, q in enumerate(rows):
        y = 70 + i * 40
        b += [text(14, y + 17, q['word'], 't-tick'),
              f'<rect class="s1" x="80" y="{y + 4}" width="{150 * q["p"]:.1f}" height="18" rx="3"/>', text(86 + 150 * q['p'], y + 18, f'{q["p"]:.4f}', 't-val'),
              f'<rect class="s2" x="276" y="{y + 4}" width="{150 * q["nll"] / mxn:.1f}" height="18" rx="3"/>', text(282 + 150 * q['nll'] / mxn, y + 18, f'{q["nll"]:.4f}', 't-val')]
    b += [box(520, 74, 226, 104, 'box-on', 10),
          text(633, 100, f'mean = {r["bert_mean_nll"]:.4f} nats', 't-note', 'middle'),
          text(633, 128, f'ppl = e^{r["bert_mean_nll"]:.4f} = {r["bert_ppl"]:.2f}', 't-big', 'middle'),
          text(633, 152, 'about as unsure as a pick', 't-muted', 'middle'), text(633, 168, f'among {r["bert_ppl"]:.1f} equal words', 't-muted', 'middle')]
    b.append(text(14, 210, f'Same three words for GPT-2, which sees only the left side: perplexity {r["gpt2_ppl"]:.2f}. "key" alone gets p = {r["gpt2"][1]["p"]:.4f}.', 't-tick'))
    return svg(760, 224, 'Perplexity of bert-base-uncased on one sentence with three masked words. The model gives the true words probabilities 0.9586, 0.1324 and 0.1823; their minus logs average 1.2554; e to that power is a perplexity of 3.51.', b)


F['p5_ppl'] = ppl()


def bio():
    d = M['bio']
    b = [text(14, 22, f'A real CoNLL-2003 dev sentence (#{d["index"]}): one tag per word, one vector per word', 't-title')]
    x = 14
    tagcls = {'O': 'box', 'B-PER': 'box-2', 'I-PER': 'box-2', 'B-LOC': 'box-3', 'I-LOC': 'box-3'}
    for w, t, pcs in zip(d['words'], d['tags'], d['pieces']):
        pw = [max(34, 9 * len(p) + 12) for p in pcs]
        width = max(sum(pw) + 4 * (len(pw) - 1), 9 * len(w) + 14, 62)
        b += [box(x, 40, width, 28, tagcls.get(t, 'box-4'), 6), text(x + width / 2, 59, t, 't-note' if t != 'O' else 't-muted', 'middle'),
              text(x + width / 2, 92, w, 't-mono', 'middle')]
        px = x + (width - (sum(pw) + 4 * (len(pw) - 1))) / 2
        for k, (p, ww) in enumerate(zip(pcs, pw)):
            b += [box(px, 112, ww, 28, 'box-on' if k == 0 else 'box-ghost', 5), text(px + ww / 2, 131, p, 't-tick' if k == 0 else 't-muted', 'middle')]
            if k == 0:
                b += [line(px + ww / 2, 142, px + ww / 2, 168, 'edge-on', 'ah-on')]
                b += vec(px + ww / 2 - 18, 174, 3, 's1', 10)
            px += ww + 4
        x += width + 8
    b += [text(14, 214, 'top: BIO tags (B = beginning of a name, I = inside it, O = outside). middle: WordPiece pieces (bert-base-cased).', 't-muted'),
          text(14, 232, 'bottom: the tagger reads only the vector of the first piece of each word (highlighted); the other pieces are ignored.', 't-muted')]
    return svg(760, 244, 'A real CoNLL-2003 sentence: Lithuania - Danius Gleveckas ( 13rd ). Each word has a BIO tag (B-LOC, O, B-PER, I-PER, O, O, O). WordPiece splits Danius into two pieces and Gleveckas into four; the tagger uses only the first piece of each word.', b)


F['p5_bio'] = bio()


def crf():
    c = M['crf']
    T = c['tags']
    b = [text(14, 22, 'Which tag may follow which (BIO rules)', 't-title'), text(330, 22, 'Hand-made tag scores for three words', 't-title')]
    g, gw, gh = attn_grid(14, 94, T, T, lambda i, j: not (j == 2 and i == 0), cell=44, row_title='previous tag (row)', col_title='next tag (column)', label_w=56)
    b += g
    sc = c['scores']
    for j, t in enumerate(T):
        b.append(text(440 + j * 90, 56, t, 't-tick', 'middle'))
    for i, w in enumerate(c['words']):
        y = 66 + i * 40
        b.append(text(390, y + 22, w, 't-mono', 'end'))
        for j in range(3):
            g_on = c['greedy'][i] == T[j]
            v_on = c['best'][i] == T[j]
            cls = 'box-on' if v_on else ('box-2' if g_on else 'box')
            b += [box(400 + j * 90, y + 4, 80, 28, cls, 6), text(440 + j * 90, y + 23, f'{sc[i][j]:.1f}', 't-val', 'middle')]
    b += [box(400, 196, 12, 12, 'box-2', 2), text(418, 206, f'each word on its own: {" ".join(c["greedy"])} (sum {c["greedy_sum"]}): invalid', 't-tick'),
          box(400, 216, 12, 12, 'box-on', 2), text(418, 226, f'best valid sequence: {" ".join(c["best"])} (sum {c["best_sum"]})', 't-tick')]
    return svg(760, 244, 'Left: BIO transition rules; O followed by I-PER is not allowed. Right: made-up scores for the words met, Ada, Lovelace. Choosing each word\'s best tag alone gives O, O, I-PER, which breaks the rule; the best valid sequence is O, B-PER, I-PER.', b)


F['p5_crf'] = crf()


def tagger():
    t = M['tagger']
    b = [text(14, 22, 'The frozen-feature tagger of Section 5.3, shape by shape (n = number of words)', 't-title')]
    steps = [('BERT hidden states', 'n x 13 x 768', 'frozen', 'box'), ('concat last four', 'n x 3,072', 'no weights', 'box'),
             ('BiLSTM, 2 layers', 'n x 768', f'{(t["lstm_layer1"] + t["lstm_layer2"]) / 1e6:.2f}M weights', 'box-4'),
             ('linear W, 9 x 768', 'n x 9', f'{t["classifier"]:,} weights', 'box-3'), ('softmax', '9 tag probabilities', 'per word', 'box')]
    for k, (name, shape, note, cls) in enumerate(steps):
        x = 14 + k * 150
        b += [box(x, 44, 132, 50, cls, 10), text(x + 66, 66, name, 't-note', 'middle'), text(x + 66, 84, note, 't-muted', 'middle'),
              text(x + 66, 116, shape, 't-mono', 'middle')]
        if k < 4:
            b.append(arrow(x + 134, 69, x + 148, 69))
    # the concat drawn as four strips glued
    for k in range(4):
        b += vec(164 + k * 31, 140, 2, ['s1', 's2', 's3', 's4'][k], 12, 2, outline=False)
    b.append(text(164, 172, '[h9 ; h10 ; h11 ; h12]', 't-tick'))
    b += vec(316, 140, [0.4, 0.8, 0.3, 0.9, 0.5], 's4', 12) + [text(316, 172, '384 forward + 384 backward', 't-tick')]
    m, mw, mh = matrix(470, 138, 3, 9, 's3', cell=9, gap=2, shape='9 rows, one per tag')
    b += m
    b += [text(14, 214, f'Trainable: {t["tagger_total"]:,} numbers (BiLSTM + classifier). BERT-base-cased: {t["bert_frozen"]:,} numbers, never updated.', 't-tick')]
    return svg(760, 228, 'The feature-based NER tagger. The frozen BERT gives 13 hidden states per word; the last four are concatenated into 3,072 numbers, a two-layer BiLSTM with 384 units per direction turns them into 768, and a 9 by 768 matrix gives one score per tag.', b)


F['p5_tagger'] = tagger()


def signal():
    s = M['signal']
    b = [text(14, 22, 'Appendix C.1: how many predictions one batch gives', 't-title')]
    for k, (name, n_on, cls, note) in enumerate([('masked LM', 15, 's1', f'15% of tokens: {s["mlm"]:,} predictions per batch'),
                                                  ('left-to-right LM', 100, 's2', f'every token: about {s["tokens"]:,} per batch')]):
        x0 = 14 + k * 380
        b.append(text(x0, 50, name, 't-note'))
        for i in range(100):
            r_, c_ = divmod(i, 20)
            on = (i * 7) % 100 < n_on if n_on < 100 else True
            b.append(f'<rect class="{cls if on else "cell-masked"}" x="{x0 + c_ * 17:.1f}" y="{62 + r_ * 17:.1f}" width="14" height="14" rx="2"/>')
        b.append(text(x0, 164, note, 't-tick'))
    b.append(text(14, 192, f'Same batch of {s["tokens"]:,} tokens. The masked LM gets {s["ratio"]}x fewer training signals, yet overtakes LTR almost at once (Figure 5).', 't-muted'))
    return svg(760, 204, 'Each square is one token of a batch. The masked LM is trained to predict only the 15 percent that were chosen; a left-to-right model predicts every token.', b)


F['p5_signal'] = signal()

json.dump(F, open('results/figs_part5.json', 'w'))
print('figures:', ', '.join(F))
