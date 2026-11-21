"""Figures for Part 1 (the big idea). Writes results/figs_part1.json (key -> svg string). CSS classes only, no colours.
Numbers: Figure 1 and Table 2 of the paper (typed in below, with the page), and results/r1_part1.json from r1_part1.py."""
import json, os
from figlib import svg, text, box, arrow, bar_path, esc
from samfig import token, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'r1_part1.json')))
F = {}


def node(x, y, w, h, lines, cls='box', tcls='t-node'):
    b = [box(x, y, w, h, cls, 8)]
    y0 = y + h / 2 - (len(lines) - 1) * 8 + 4
    for i, s in enumerate(lines):
        b.append(text(x + w / 2, y0 + i * 16, s, tcls if i == 0 else 't-muted', 'middle'))
    return b


def paper_map():
    """The whole paper on one page: three roads, with the part of this series that explains each step."""
    W, H = 760, 440
    b = [text(20, 26, 'The whole paper in one picture', 't-title')]
    h = 50
    # road 1: R1-Zero
    y1 = 70
    b.append(text(20, y1 - 10, '1. DeepSeek-R1-Zero (§2.2): reinforcement learning only', 't-strong'))
    b += node(20, y1, 150, h, ['DeepSeek-V3-Base', 'pre-trained, 671B'], 'box')
    b += node(225, y1, 230, h, ['RL with GRPO', 'reward: right answer + format'], 'box-1')
    b += node(510, y1, 170, h, ['DeepSeek-R1-Zero', 'AIME 71.0%'], 'box-on')
    b.append(arrow(170, y1 + h / 2, 223, y1 + h / 2, on=True))
    b.append(arrow(455, y1 + h / 2, 508, y1 + h / 2, on=True))
    b.append(text(340, y1 + h + 16, 'Parts 2, 3 and 4', 't-muted', 'middle'))
    # road 2: R1
    y2 = 190
    b.append(text(20, y2 - 10, '2. DeepSeek-R1 (§2.3): four stages', 't-strong'))
    xs = [20, 165, 310, 455, 600]
    labels = [['V3-Base', ''], ['1. cold start', 'SFT, thousands'], ['2. reasoning RL', '+ language reward'],
              ['3. rejection', 'sampling + SFT'], ['4. RL for all', 'scenarios']]
    for i, (x, lab) in enumerate(zip(xs, labels)):
        cls = 'box' if i == 0 else ('box-1' if i in (2, 4) else 'box-2')
        b += node(x, y2, 125, h, [s for s in lab if s], cls)
        if i:
            b.append(arrow(xs[i - 1] + 125, y2 + h / 2, x - 2, y2 + h / 2, on=True))
    b += node(600, y2 + 82, 125, h, ['DeepSeek-R1', 'AIME 79.8%'], 'box-on')
    b.append(arrow(662, y2 + h, 662, y2 + 80, on=True))
    b.append(text(20, y2 + h + 16, 'SFT = supervised fine-tuning on examples; RL = reinforcement learning on rewards', 't-muted'))
    b.append(text(400, y2 + h + 34, 'Part 5', 't-muted', 'middle'))
    # road 3: distillation
    y3 = 360
    b.append(text(20, y3 - 10, '3. Distillation (§2.4): small models copy R1', 't-strong'))
    b += node(555, y3, 170, h, ['800k samples', 'written by R1 + V3'], 'box-2')
    b += node(290, y3, 210, h, ['SFT on Qwen2.5 and Llama', 'no RL'], 'box-2')
    b += node(20, y3, 210, h, ['6 distilled models', '1.5B to 70B'], 'box-on')
    b.append(arrow(662, y2 + 82 + h, 662, y3 - 2, on=True))
    b.append(arrow(555, y3 + h / 2, 502, y3 + h / 2, on=True))
    b.append(arrow(290, y3 + h / 2, 232, y3 + h / 2, on=True))
    b.append(text(20, y3 + h + 18, 'Part 6. We run the 1.5B model on this page.', 't-muted'))
    return svg(W, H, 'A map of the paper with three roads. Road 1: DeepSeek-V3-Base goes through reinforcement learning with '
               'GRPO and becomes DeepSeek-R1-Zero. Road 2: V3-Base goes through cold-start fine-tuning, reasoning RL, rejection '
               'sampling with fine-tuning, and RL for all scenarios, and becomes DeepSeek-R1. Road 3: 800 thousand samples '
               'from R1 fine-tune six small Qwen and Llama models.', b)


# Figure 1 of the paper (page 1): benchmark -> values for R1, o1-1217, R1-32B, o1-mini, V3
FIG1 = [('AIME 2024', 'pass@1', [79.8, 79.2, 72.6, 63.6, 39.2]),
        ('Codeforces', 'percentile', [96.3, 96.6, 90.6, 93.4, 58.7]),
        ('GPQA Diamond', 'pass@1', [71.5, 75.7, 62.1, 60.0, 59.1]),
        ('MATH-500', 'pass@1', [97.3, 96.4, 94.3, 90.0, 90.2]),
        ('MMLU', 'pass@1', [90.8, 91.8, 87.4, 85.2, 88.5]),
        ('SWE-bench Verified', 'resolved', [49.2, 48.9, 36.8, 41.6, 42.0])]
MODELS = ['DeepSeek-R1', 'OpenAI-o1-1217', 'DeepSeek-R1-32B (distilled)', 'OpenAI-o1-mini', 'DeepSeek-V3']
CLS = ['s1', 's2', 's3', 's4', 'box']


def figure1():
    W, H = 760, 400
    L, T, B = 46, 70, 64
    pw, ph = W - L - 10, H - T - B
    gw = pw / len(FIG1)
    bw = 20
    Y = lambda v: T + ph - v / 100 * ph
    b = [text(20, 22, "Figure 1, redrawn from the paper's numbers", 't-title')]
    lx = 20
    for m, c in zip(MODELS, CLS):                               # legend
        b.append(f'<rect class="{c}" x="{lx}" y="36" width="12" height="12" rx="2"/>')
        b.append(text(lx + 17, 46, m, 't-tick'))
        lx += 17 + 7.2 * len(m) + 18
    for t in (0, 20, 40, 60, 80, 100):
        b.append(f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>')
        b.append(text(L - 6, Y(t) + 4, str(t), 't-tick', 'end'))
    for g, (name, metric, vals) in enumerate(FIG1):
        x0 = L + g * gw + (gw - 5 * bw) / 2
        for i, v in enumerate(vals):
            x = x0 + i * bw
            b.append(f'<g class="mark"><title>{esc(name)}, {esc(MODELS[i])}: {v}</title>'
                     f'<path class="{CLS[i]}" d="{bar_path(x + 1, Y(v), bw - 2, Y(0) - Y(v))}"/></g>')
        b.append(text(x0 + bw / 2, Y(vals[0]) - 5, f'{vals[0]}', 't-val', 'middle'))
        b.append(text(L + g * gw + gw / 2, Y(0) + 18, name, 't-tick', 'middle'))
        b.append(text(L + g * gw + gw / 2, Y(0) + 34, f'({metric})', 't-muted', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{Y(0):.1f}" x2="{L + pw}" y2="{Y(0):.1f}"/>')
    b.append(text(L, H - 6, 'Score in %. The number on each group is DeepSeek-R1. Hover a bar for its value.', 't-muted'))
    return svg(W, H, 'Grouped bars for six benchmarks and five models, with the numbers of Figure 1 of the paper. DeepSeek-R1 '
               'and OpenAI-o1-1217 are close on every benchmark; DeepSeek-V3 is far lower on AIME 2024 and Codeforces.', b)


def aime():
    """AIME 2024 numbers from §1 and Table 2 (page 7) and Figure 1."""
    items = [('R1-Zero, start of RL', 15.6), ('R1-Zero, end of RL', 71.0), ('R1-Zero, majority of 64', 86.7),
             ('o1-0912', 74.4), ('o1-0912, majority of 64', 83.3), ('DeepSeek-R1', 79.8), ('o1-1217', 79.2)]
    b = hbars(20, 50, items, 420, row_h=30, cls='s1', label_w=190, fmt=lambda v: f'{v}%', vmax=100,
              title='AIME 2024: the share of problems solved', hi='R1-Zero, majority of 64')
    b.append(text(20, 50 + 7 * 30 + 22, 'pass@1 unless the row says majority (cons@64). Sources: §1, Table 2, Figure 1.', 't-muted'))
    return svg(720, 50 + 7 * 30 + 34, 'Horizontal bars of AIME 2024 scores: DeepSeek-R1-Zero rises from 15.6 to 71.0 percent '
               'with RL and to 86.7 percent with majority voting; OpenAI o1-0912 has 74.4 and 83.3; DeepSeek-R1 has 79.8 and o1-1217 79.2.', b)


def sampling():
    """Four toy logits -> probabilities at T = 1 -> at T = 0.6 -> after top-p 0.95 (values from r1_part1.py)."""
    t = R['toy']
    cols = [('logits', t['logits'], lambda v: f'{v:.1f}'), ('softmax, T = 1', t['p1'], lambda v: f'{v:.3f}'),
            ('softmax, T = 0.6', t['p06'], lambda v: f'{v:.3f}'), ('top-p 0.95', t['top_p'], lambda v: f'{v:.3f}')]
    W, H = 760, 250
    b = [text(20, 24, 'One decoding step, on four toy tokens', 't-title')]
    cw = 180
    for c, (title, vals, fmt) in enumerate(cols):
        x0 = 20 + c * (cw + 6)
        b.append(text(x0, 54, f'{c + 1}. {title}', 't-strong'))
        vmax = 2.0 if c == 0 else 1.0
        for i, (nm, v) in enumerate(zip(t['names'], vals)):
            y = 70 + i * 36
            b.append(text(x0, y + 18, nm.strip('"'), 't-tick'))
            w = max(0, 110 * v / vmax)
            cls = 's4' if (c == 3 and v == 0) else ('s2' if c == 0 else 's1')
            if v > 0:
                b.append(f'<path class="{cls}" d="{bar_path(x0 + 26, y + 4, w, 20, horizontal=True)}"/>')
            b.append(text(x0 + 30 + max(w, 0), y + 19, fmt(v) if not (c == 3 and v == 0) else '0 (cut)', 't-val'))
    b.append(text(20, H - 14, 'Lower temperature sharpens the distribution; top-p then drops the tail and samples from the rest.', 't-muted'))
    return svg(W, H, 'Four panels for four toy tokens. Logits 2.0, 1.0, 0.5 and -1.0 become probabilities with a softmax; '
               'temperature 0.6 makes the top token more likely; top-p 0.95 removes the least likely token and renormalises.', b)


def anatomy():
    """The distilled model's answer to the GSM8K question: prompt, thinking, answer, in tokens (our run)."""
    q = R['q1_ours']
    parts = [('thinking, inside <think> ... </think>', q['think'], 'box-1'), ('answer', q['answer'], 'box-on')]
    total = q['think'] + q['answer']
    W, H = 760, 150
    b = [text(20, 24, f'One answer of the distilled 1.5B model: {q["total"]:,} new tokens, cut at the limit', 't-title')]
    x, pw = 20, 720
    for lab, n, cls in parts:
        w = pw * n / total
        b.append(box(x, 50, w, 40, cls, 6))
        b.append(text(x + w / 2, 75, f'{n}', 't-val', 'middle'))
        b.append(text(x + w / 2 if w > 200 else x + w, 112, f'{lab}: {n} tokens', 't-tick', 'middle' if w > 200 else 'end'))
        x += w
    b.append(text(20, 138, 'GSM8K test question 0, temperature 0.6, top-p 0.95, seed 0, our own sampling loop. The tag </think> is one more token.', 't-muted'))
    return svg(W, H, f'A bar split in two: {q["think"]} thinking tokens, then {q["answer"]} answer tokens.', b)


def samples():
    """8 samples per model on MATH-500 problem 10: length of each sample, marked right or wrong."""
    W = 760
    b = [text(20, 24, 'Eight samples per model on one MATH-500 problem', 't-title')]
    y = 50
    for label, key, mx in (('distilled 1.5B', 'q2_distilled', 6144), ('base 1.5B', 'q2_base', 1024)):
        d = R[key]
        b.append(text(20, y + 4, f'{label}: pass@1 = {d["pass1"]:.3f}, majority answer {d["majority"]}', 't-strong'))
        y += 14
        for i, r in enumerate(d['rows']):
            w = 420 * r['tokens'] / 6144
            cls = 's1' if r['correct'] else 's4'
            b.append(text(20, y + 15, f'sample {i}', 't-tick'))
            b.append(f'<g class="mark"><title>{label} sample {i}: {r["tokens"]} tokens, answer {esc(r["pred"])}</title>'
                     f'<path class="{cls}" d="{bar_path(90, y + 3, max(w, 2), 16, horizontal=True)}"/></g>')
            b.append(text(96 + w, y + 16, (f'{r["tokens"]:,} tokens, answer {r["pred"]}, right' if r['correct'] else (f'{r["tokens"]:,} tokens, still thinking at the limit' if r['pred'] is None else f'{r["tokens"]:,} tokens, answer {r["pred"]}, wrong')), 't-tick'))
            y += 22
        y += 22
    b.append(text(20, y, 'Bar length = new tokens (the scale ends at 6,144). Right answers use the first colour.', 't-muted'))
    return svg(W, y + 12, 'Two groups of eight bars, the lengths of eight sampled answers from the distilled model and from '
               'the base model on the same MATH-500 problem, each marked right or wrong.', b)


F['p1_map'] = paper_map()
F['p1_fig1'] = figure1()
F['p1_aime'] = aime()
F['p1_sampling'] = sampling()
F['p1_anatomy'] = anatomy()
F['p1_samples'] = samples()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
