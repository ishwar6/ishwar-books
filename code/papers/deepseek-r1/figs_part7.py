"""Figures for Part 7, drawn from Table 4 of the paper and from results/r1_part7_passk.json and results/r1_part7_gen.json.
Writes results/figs_part7.json (key -> svg string). Colours come only from CSS classes."""
import json, math, os
from figlib import svg, text, box, arrow, esc, lines_chart

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
PK = json.load(open(os.path.join(RES, 'r1_part7_passk.json')))
GEN_PATH = os.path.join(RES, 'r1_part7_gen.json')
GEN = json.load(open(GEN_PATH)) if os.path.exists(GEN_PATH) else None
F = {}


def wrap_tok(x, y, w, h, s, cls='box', tcls='t-tick'):
    return [box(x, y, w, h, cls, 8), text(x + w / 2, y + h / 2 + 4.5, s, tcls, 'middle')]


# ------------------------------------------------------------------ 1. you are here + how one question is scored
def eval_fig():
    W, H = 760, 320
    b = []
    stages = ['V3-Base', 'cold start SFT', 'reasoning RL', 'reject + SFT', 'RL, all tasks', 'R1', 'evaluate']
    x, y, w, h, gap = 10, 30, 92, 34, 14
    b.append(text(10, 18, 'Where this part sits in the R1 pipeline', 't-title'))
    for i, s in enumerate(stages):
        cls = 'box-on' if s == 'evaluate' else ('box-2' if s == 'R1' else 'box')
        b += wrap_tok(x + i * (w + gap), y, w, h, s, cls, 't-tick')
        if i:
            b.append(arrow(x + i * (w + gap) - gap + 1, y + h / 2, x + i * (w + gap) - 2, y + h / 2))
    b.append(text(x + 6 * (w + gap) + w / 2, y + h + 16, 'you are here', 't-on', 'middle'))
    # one question, k sampled answers, graded
    y0 = 120
    b.append(text(10, y0 - 8, 'One question, scored the way §3 scores it', 't-title'))
    b += wrap_tok(10, y0 + 70, 110, 40, 'question', 'box-1', 't-tick')
    marks = [('answer 1: 12', True), ('answer 2: 7', False), ('answer 3: 12', True), ('answer 4: 12', True)]
    for i, (s, ok) in enumerate(marks):
        yy = y0 + 4 + i * 44
        b.append(arrow(122, y0 + 90, 196, yy + 17))
        b += wrap_tok(200, yy, 150, 34, s, 'box-3' if ok else 'box', 't-tick')
        b.append(text(362, yy + 22, 'right' if ok else 'wrong', 't-s3' if ok else 't-bad'))
    b.append(text(10, y0 + 132, 'k samples (4 to 64)', 't-muted'))
    b.append(text(10, y0 + 148, 'T = 0.6, top-p = 0.95', 't-muted'))
    # two scores
    b += wrap_tok(450, y0 + 20, 290, 40, 'pass@1 = mean of p_i = 3/4 = 0.75', 'box-2', 't-tick')
    b += wrap_tok(450, y0 + 110, 290, 40, 'cons@k: majority answer 12, right = 1', 'box-2', 't-tick')
    b.append(arrow(418, y0 + 60, 446, y0 + 42))
    b.append(arrow(418, y0 + 110, 446, y0 + 128))
    b.append(text(595, y0 + 76, 'average over all answers', 't-muted', 'middle'))
    b.append(text(595, y0 + 166, 'vote first, then grade once', 't-muted', 'middle'))
    return svg(W, H, 'A pipeline row from DeepSeek-V3-Base through cold start, reasoning RL, rejection sampling, '
               'a second RL stage and R1 to evaluation, then one question with four sampled answers graded right or '
               'wrong, giving pass@1 of 0.75 and a cons@k score of 1.', b)


F['p7_eval'] = eval_fig()

# ------------------------------------------------------------------ 2. Table 4 as a dot chart by group
T4 = [  # (group, benchmark, R1, o1-1217, V3), all from Table 4 of the paper (page 13); None = not reported
    ('English', 'MMLU', 90.8, 91.8, 88.5), ('English', 'MMLU-Redux', 92.9, None, 89.1),
    ('English', 'MMLU-Pro', 84.0, None, 75.9), ('English', 'DROP', 92.2, 90.2, 91.6),
    ('English', 'IF-Eval', 83.3, None, 86.1), ('English', 'GPQA Diamond', 71.5, 75.7, 59.1),
    ('English', 'SimpleQA', 30.1, 47.0, 24.9), ('English', 'FRAMES', 82.5, None, 73.3),
    ('English', 'AlpacaEval 2.0', 87.6, None, 70.0), ('English', 'ArenaHard', 92.3, None, 85.5),
    ('Code', 'LiveCodeBench', 65.9, 63.4, 36.2), ('Code', 'Codeforces pct', 96.3, 96.6, 58.7),
    ('Code', 'SWE Verified', 49.2, 48.9, 42.0), ('Code', 'Aider-Polyglot', 53.3, 61.7, 49.6),
    ('Math', 'AIME 2024', 79.8, 79.2, 39.2), ('Math', 'MATH-500', 97.3, 96.4, 90.2),
    ('Math', 'CNMO 2024', 78.8, None, 43.2),
    ('Chinese', 'CLUEWSC', 92.8, None, 90.9), ('Chinese', 'C-Eval', 91.8, None, 86.5),
    ('Chinese', 'C-SimpleQA', 63.7, None, 68.0),
]


def table4_fig():
    L, R_, top, row = 190, 30, 50, 22
    W = 760
    pw = W - L - R_
    X = lambda v: L + v / 100 * pw
    groups = []
    for g, *_ in T4:
        if g not in groups:
            groups.append(g)
    H = top + len(T4) * row + len(groups) * 14 + 40
    b = [text(10, 18, 'Table 4 by group: R1, o1-1217 and V3 (score out of 100)', 't-title')]
    lx = L
    for cls, name in (('s1', 'DeepSeek-R1'), ('s2', 'OpenAI-o1-1217'), ('s3', 'DeepSeek-V3')):
        b.append(f'<circle class="{cls}" cx="{lx + 6}" cy="34" r="5"/>')
        b.append(text(lx + 16, 38, name, 't-note'))
        lx += 150
    for v in range(0, 101, 20):
        b.append(f'<line class="grid" x1="{X(v):.1f}" y1="{top - 6}" x2="{X(v):.1f}" y2="{H - 34}"/>')
        b.append(text(X(v), H - 18, str(v), 't-tick', 'middle'))
    y = top
    cur = None
    for g, name, r1, o1, v3 in T4:
        if g != cur:
            cur = g
            b.append(text(10, y + 12, g, 't-strong'))
            y += 14
        cy = y + row / 2
        win = (r1 < v3) or (o1 is not None and r1 < o1)
        b.append(text(L - 10, cy + 4, name, 't-tick t-strong' if win else 't-tick', 'end'))
        vals = [v for v in (r1, o1, v3) if v is not None]
        b.append(f'<line class="axis" x1="{X(min(vals)):.1f}" y1="{cy:.1f}" x2="{X(max(vals)):.1f}" y2="{cy:.1f}"/>')
        for v, cls, who in ((v3, 's3', 'DeepSeek-V3'), (o1, 's2', 'OpenAI-o1-1217'), (r1, 's1', 'DeepSeek-R1')):
            if v is None:
                continue
            b.append(f'<g class="mark"><title>{esc(name)}, {who}: {v}</title>'
                     f'<circle class="{cls} ring" cx="{X(v):.1f}" cy="{cy:.1f}" r="5.5"/></g>')
        y += row
    b.append(text(10, H - 4, 'bold names: R1 is behind V3 or o1-1217', 't-muted'))
    return svg(W, H, 'Dot chart of Table 4: for each benchmark, grouped into English, code, maths and Chinese, the '
               'scores of DeepSeek-R1, OpenAI-o1-1217 and DeepSeek-V3. R1 is far ahead of V3 on maths and '
               'competitive code, and behind on IF-Eval, C-SimpleQA, SimpleQA, GPQA and Aider.', b)


F['p7_table4'] = table4_fig()

# ------------------------------------------------------------------ 3. pass@k curves
cur = PK['curve_n16']
ks = list(range(1, 17))
F['p7_passk'] = lines_chart(
    760, 340, 'pass@k for k from 1 to 16, computed with the unbiased estimator from 16 samples of which c are '
    'correct, for c = 1, 2, 4 and 8. Each curve rises fast at small k and reaches 1 at k = 16.',
    ks, [(f'c = {c} of 16', cur[str(c)], f'l{i + 1}', f's{i + 1}') for i, c in enumerate([1, 2, 4, 8])],
    'k (the number of tries allowed)', 'pass@k', 0, 1, [0, 0.25, 0.5, 0.75, 1.0], xlog=True, xticks=[1, 2, 4, 8, 16],
    fmt=lambda v: f'{v:.2f}')


# ------------------------------------------------------------------ 4. greedy vs sampling vs few-shot (our run)
def gen_fig():
    if GEN is None:
        return None
    modes = [('greedy', 'greedy, 1 answer'), ('sample', 'sampled, zero-shot'), ('fewshot', 'sampled, few-shot')]
    metrics = [('accuracy', 'answers correct'), ('hit', 'hit the 2,048-token limit'), ('rep50', 'repeat a 50+ token block')]
    W, L, top, gh, bh = 760, 200, 56, 86, 20
    pw = W - L - 90
    H = top + len(metrics) * gh + 20
    b = [text(10, 18, 'Our run: R1-Distill-Qwen-1.5B on 8 MATH-500 problems', 't-title')]
    lx = L
    for i, (_, name) in enumerate(modes):
        b.append(f'<rect class="s{i + 1}" x="{lx}" y="28" width="10" height="10" rx="2"/>')
        b.append(text(lx + 16, 37, name, 't-note'))
        lx += 170
    for j, (key, lab) in enumerate(metrics):
        y = top + j * gh
        b.append(text(L - 12, y + gh / 2 - 4, lab, 't-tick', 'end'))
        for i, (m, name) in enumerate(modes):
            s = GEN[m]
            n = s['answers']
            cnt = round(s['accuracy'] * n) if key == 'accuracy' else (GEN['repeat'][m]['over50'] if key == 'rep50' else s[key])
            v = cnt / n
            w = max(2, pw * v)
            yy = y + i * (bh + 4)
            b.append(f'<g class="mark"><title>{esc(name)}, {esc(lab)}: {cnt} of {n}</title>'
                     f'<rect class="s{i + 1}" x="{L}" y="{yy:.1f}" width="{w:.1f}" height="{bh}" rx="3"/></g>')
            b.append(text(L + w + 6, yy + bh - 5, f'{cnt}/{n}', 't-val'))
    return svg(W, H, 'Grouped bars from our run: for greedy decoding, zero-shot sampling and few-shot sampling, the '
               'share of answers that were correct, that hit the 2,048-token limit, and that wrote a block of 50 or more tokens twice.', b)


g = gen_fig()
if g:
    F['p7_greedy'] = g


# ------------------------------------------------------------------ 5. nucleus sizes and the size of the tree
def nucleus_fig():
    if GEN is None:
        return None
    t = GEN['tree']
    W, L, top, H = 760, 70, 50, 300
    pw, ph = W - L - 30, H - top - 50
    labels = [str(i) for i in range(1, 11)] + ['11+']
    tot6, tot1 = sum(t['hist']), sum(t['hist_t1'])
    a6 = [v / tot6 for v in t['hist']]
    a1 = [v / tot1 for v in t['hist_t1']]
    vmax = max(a6 + a1)
    ymax = math.ceil(vmax * 10) / 10
    cw = pw / len(labels)
    Y = lambda v: top + ph - v / ymax * ph
    b = [text(10, 18, 'How many tokens are in the top-p 0.95 nucleus at each step?', 't-title')]
    for i, (cls, name) in enumerate((('s1', 'T = 0.6 (the paper)'), ('s2', 'T = 1.0'))):
        b.append(f'<rect class="{cls}" x="{L + i * 180}" y="28" width="10" height="10" rx="2"/>')
        b.append(text(L + i * 180 + 16, 37, name, 't-note'))
    for v in [x / 10 for x in range(0, int(ymax * 10) + 1, 2)]:
        b.append(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}"/>')
        b.append(text(L - 8, Y(v) + 4, f'{v:.0%}', 't-tick', 'end'))
    for i, lab in enumerate(labels):
        x = L + i * cw
        for j, (arr, cls, name) in enumerate(((a6, 's1', 'T = 0.6'), (a1, 's2', 'T = 1.0'))):
            v = arr[i]
            bw = cw * 0.38
            bx = x + cw * 0.1 + j * (bw + 2)
            b.append(f'<g class="mark"><title>nucleus of {lab} tokens, {name}: {v:.1%} of steps</title>'
                     f'<rect class="{cls}" x="{bx:.1f}" y="{Y(v):.1f}" width="{bw:.1f}" height="{max(0.5, Y(0) - Y(v)):.1f}" rx="2"/></g>')
        b.append(text(x + cw / 2, top + ph + 18, lab, 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{Y(0):.1f}" x2="{L + pw}" y2="{Y(0):.1f}"/>')
    b.append(text(L + pw / 2, H - 8, 'tokens in the nucleus (the choices sampling can make at that step)', 't-tick', 'middle'))
    return svg(W, H, f'Histogram of nucleus sizes over every step of our sampled answers. At temperature 0.6, '
               f'{t["one"]:.0%} of steps have a single token; a few steps have many.', b)


def tree_fig():
    if GEN is None:
        return None
    t = GEN['tree']
    items = [('chess, 35^80', t['chess']), ('Go, 250^150', t['go']),
             (f'one answer\'s schedule ({t["L"]} tok)', t['m1']),
             (f'all 151,665 tokens ({t["L"]} tok)', t['vocab'])]
    W, L, top, rh = 760, 230, 40, 34
    vmax = max(v for _, v in items)
    pw = W - L - 110
    H = top + len(items) * rh + 40
    b = [text(10, 18, 'Size of a full search tree, as a power of ten (log scale)', 't-title')]
    for i, (lab, v) in enumerate(items):
        y = top + i * rh
        w = max(2, pw * v / vmax)
        b.append(text(L - 10, y + rh / 2 + 4, lab, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(lab)}: 10^{v:.0f}</title>'
                 f'<rect class="{"s2" if i < 2 else "s1"}" x="{L}" y="{y + 6}" width="{w:.1f}" height="{rh - 12}" rx="3"/></g>')
        b.append(text(L + w + 6, y + rh / 2 + 4, f'10^{v:,.0f}', 't-val'))
    b.append(text(L, H - 10, 'bar length = number of digits of the count', 't-muted'))
    return svg(W, H, 'Bars of the number of digits of four search-tree sizes: chess about 10 to the 124, Go about '
               '10 to the 360, a hypothetical tree that repeats the nucleus sizes of one of our answers at every step, and the paths '
               'through the whole vocabulary for the same length.', b)


for k, f in (('p7_nucleus', nucleus_fig()), ('p7_tree', tree_fig())):
    if f:
        F[k] = f


# ------------------------------------------------------------------ 6. the four steps of MCTS
def mcts_fig():
    W, H = 760, 260
    b = [text(10, 18, 'One round of Monte Carlo Tree Search', 't-title')]
    titles = ['1. select', '2. expand', '3. evaluate', '4. back up']
    notes = ['follow the best-looking child', 'add a new child', 'value model scores it', 'update every parent']
    for k in range(4):
        ox = 10 + k * 188
        b.append(text(ox + 85, 48, titles[k], 't-title', 'middle'))
        b.append(text(ox + 85, 245, notes[k], 't-muted', 'middle'))
        root, l1, l2 = (ox + 85, 76), [(ox + 40, 130), (ox + 130, 130)], [(ox + 15, 186), (ox + 65, 186)]
        nodes = [root] + l1 + l2
        edges = [(root, l1[0]), (root, l1[1]), (l1[0], l2[0]), (l1[0], l2[1])]
        new = (ox + 130, 186)
        if k >= 1:
            nodes.append(new); edges.append((l1[1], new))
        on_path = {root, l1[1]} | ({new} if k >= 1 else set())
        for a, c in edges:
            hot = (k in (0, 3)) and a in on_path and c in on_path
            b.append(f'<line class="{"path" if hot else "edge"}" x1="{a[0]}" y1="{a[1]}" x2="{c[0]}" y2="{c[1]}"/>')
        for n in nodes:
            cls = 'node on' if (n == new and k >= 1) or (k in (0, 3) and n in on_path) else 'node'
            b.append(f'<circle class="{cls}" cx="{n[0]}" cy="{n[1]}" r="13"/>')
        if k == 2:
            b.append(text(new[0], new[1] + 32, 'v = 0.7', 't-on', 'middle'))
        if k == 3:
            for n in (root, l1[1]):
                b.append(text(n[0] + 18, n[1] - 10, '+0.7', 't-on'))
    return svg(W, H, 'Four small trees showing one round of Monte Carlo Tree Search: select a path of promising '
               'children, expand a new child, let a value model score it, and back the score up to every parent.', b)


F['p7_mcts'] = mcts_fig()

json.dump(F, open(os.path.join(RES, 'figs_part7.json'), 'w'), indent=1)
print('wrote results/figs_part7.json:', ', '.join(F))
