"""Figures for Part 2 (the method). Writes results/figs_part2.json (key -> svg string). CSS classes only.
p2_table1: Table 1's context per evaluation on a log scale.   p2_loop: Algorithm 1 with our folder layout.
p2_pareto: Pareto frontier of Table 2's numbers.               p2_run: our run, search accuracy vs test accuracy."""
import json, math, os
from figlib import svg, text, box, arrow, esc
from samfig import token, line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part2.json')))
F = {}


def table1():
    rows = [('OPRO', 0.002), ('GEPA', 0.008), ('Feedback Descent', 0.012), ('TextGrad', 0.015),
            ('AlphaEvolve', 0.022), ('TTT-Discover', 0.026), ('Meta-Harness', 10.0)]
    W, L, top, rh = 720, 150, 50, 34
    lo, hi = 3, 7                                     # 10^3 to 10^7 tokens
    pw = W - L - 110
    X = lambda tok: L + (math.log10(tok) - lo) / (hi - lo) * pw
    b = [text(20, 26, 'Context the proposer can get from one evaluation (Table 1, log scale)', 't-title')]
    H = top + rh * len(rows) + 40
    for e in range(lo, hi + 1):
        x = L + (e - lo) / (hi - lo) * pw
        b.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 6}" x2="{x:.1f}" y2="{top + rh * len(rows)}"/>')
        b.append(text(x, top + rh * len(rows) + 18, f'{10 ** e:,}', 't-tick', 'middle'))
    for i, (name, mtok) in enumerate(rows):
        y, tok = top + i * rh, mtok * 1e6
        on = name == 'Meta-Harness'
        b.append(text(L - 10, y + rh / 2 + 4, name, 't-tick t-strong' if on else 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(name)}: {mtok} MTok = {tok:,.0f} tokens</title>'
                 f'<rect class="{"s2" if on else "s1"}" x="{L}" y="{y + 6}" width="{X(tok) - L:.1f}" height="{rh - 12}" rx="3"/></g>')
        b.append(text(X(tok) + 8, y + rh / 2 + 4, f'{tok:,.0f}', 't-val'))
    b.append(text(L + pw / 2, H - 4, 'tokens per evaluation (each grid line is 10 times the one before)', 't-tick', 'middle'))
    return svg(W, H + 6, 'Horizontal bars on a log scale. Six earlier optimizers get between 2,000 and 26,000 tokens of '
               'context per evaluation. Meta-Harness gets up to 10,000,000, about 385 times the largest of them.', b)


def loop():
    W, H = 720, 420
    b = [text(20, 26, 'Algorithm 1, with the folders of our run', 't-title')]
    # the filesystem D
    b.append(box(20, 50, 250, 250, 'box-1'))
    b.append(text(34, 74, 'filesystem D (part2_fs/)', 't-label'))
    files = ['000_zero_shot/', '001_few_shot_4/', '002_one_word_answer/', '...', 'rejected.jsonl', 'frontier.json']
    for i, f in enumerate(files):
        b.append(text(44, 102 + i * 22, f, 't-mono'))
    b.append(text(44, 250, 'each folder: harness.py, scores.json,', 't-note'))
    b.append(text(44, 268, 'traces.jsonl, proposal.md', 't-note'))
    # proposer
    b += token(330, 60, 170, 'proposer P', 'box-on', h=44, tcls='t-label')
    b.append(text(415, 122, 'reads files (grep, cat)', 't-note', 'middle'))
    b.append(text(415, 140, 'writes k new harness files', 't-note', 'middle'))
    b.append(arrow(272, 82, 328, 82, on=True))
    # validation
    b += token(330, 170, 170, 'interface check', 'box', h=40, tcls='t-label')
    b.append(arrow(415, 146, 415, 168))
    b.append(text(510, 194, 'fails: one line in rejected.jsonl', 't-note'))
    # evaluate
    b += token(330, 250, 170, 'Evaluate(H, M, search)', 'box', h=40, tcls='t-label')
    b.append(arrow(415, 212, 415, 248))
    b.append(text(510, 266, 'frozen model M', 't-note'))
    b.append(text(510, 284, '24 search reviews', 't-note'))
    b.append(arrow(330, 270, 272, 230, on=True))
    b.append(text(282, 312, 'store: new folder', 't-note'))
    # after the loop
    b.append(box(20, 334, 680, 70, 'box-ghost'))
    b.append(text(34, 358, 'After N iterations (line 13): the Pareto frontier of D.', 't-label'))
    b.append(text(34, 382, 'Only then is the frontier run once on the 48 test reviews. Those results are written outside D.', 't-note'))
    return svg(W, H, 'A diagram of the loop. On the left, the filesystem D holds one folder per evaluated harness. '
               'The proposer reads it and writes new harness files. Each file goes through an interface check, '
               'then is evaluated on the search set with the frozen model, and the result is stored as a new folder. '
               'After the loop, the Pareto frontier is run once on the test set, and those results stay outside D.', b)


def scatter(W, H, label, pts, front, xlab, ylab, xmax, ymin, ymax, xticks, yticks, title, fmt_x=lambda v: f'{v:g}'):
    L, T, B, Rt = 64, 44, 48, 30
    pw, ph = W - L - Rt, H - T - B
    X = lambda v: L + v / xmax * pw
    Y = lambda v: T + ph - (v - ymin) / (ymax - ymin) * ph
    b = [text(20, 24, title, 't-title')]
    for t in yticks:
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, f'{t:g}', 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), T + ph + 18, fmt_x(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    fp = sorted((pts[k][1], pts[k][0]) for k in front)
    b.append('<polyline class="l2" points="' + ' '.join(f'{X(c):.1f},{Y(a):.1f}' for c, a in fp) + '"/>')
    for k, (acc, ctx) in pts.items():
        on = k in front
        b.append(f'<g class="mark"><title>{esc(k)}: accuracy {acc:g}, context {ctx:g}</title>'
                 f'<circle class="{"s2" if on else "s1"} ring" cx="{X(ctx):.1f}" cy="{Y(acc):.1f}" r="{6 if on else 5}"/></g>')
        dy = 17 if k == 'Few-Shot (8)' else -7                     # keep two close labels apart
        b.append(text(X(ctx) + 9, Y(acc) + dy, k, 't-tick t-strong' if on else 't-tick'))
    b.append(text(L + pw / 2, H - 8, xlab, 't-tick', 'middle'))
    b.append(text(L - 50, T - 10, ylab, 't-tick'))
    return svg(W, H, label, b)


def pareto():
    T2 = {'Zero-Shot': (27.4, 0.0), 'Few-Shot (8)': (34.3, 2.0), 'Few-Shot (32)': (35.4, 7.9),
          'Few-Shot (all)': (40.8, 12.3), 'MCE': (40.0, 28.5), 'ACE': (40.9, 50.8), 'Meta-Harness': (48.6, 11.4)}
    return scatter(720, 380, 'A scatter plot of Table 2: accuracy against extra context in thousands of tokens. '
                   'The frontier joins Zero-Shot, Few-Shot (8), Few-Shot (32) and Meta-Harness. Few-Shot (all), MCE and '
                   'ACE lie below and to the right of it, so each is dominated.',
                   T2, R['pareto_table2'], 'extra context (thousands of tokens, lower is better)', 'accuracy % (higher is better)',
                   56, 25, 52, [0, 10, 20, 30, 40, 50], [25, 30, 35, 40, 45, 50],
                   'Table 2 as points: the line joins the Pareto frontier')


def run_fig():
    T = R['test']
    names = list(T)
    W, L, top, rh = 720, 190, 56, 46
    pw = W - L - 70
    b = [text(20, 26, 'Our run: the frontier on the search set and on the test set', 't-title')]
    for i, n in enumerate(names):
        y = top + i * rh
        b.append(text(L - 10, y + 18, n.split('_', 1)[1], 't-tick', 'end'))
        for j, (key, cls) in enumerate((('search', 's1'), ('test', 's2'))):
            v = T[n][key]
            b.append(f'<g class="mark"><title>{esc(n)} {key}: {v:.3f}</title><rect class="{cls}" x="{L}" y="{y + j * 17:.1f}" '
                     f'width="{max(2, v * pw):.1f}" height="14" rx="3"/></g>')
            b.append(text(L + max(2, v * pw) + 6, y + j * 17 + 11, f'{key} {v:.3f}', 't-val'))
    H = top + rh * len(names) + 10
    return svg(W, H, 'Bars for each harness on the Pareto frontier of our run: accuracy on the 24 search reviews and on '
               'the 48 test reviews. ' + '; '.join(f'{n}: search {T[n]["search"]:.3f}, test {T[n]["test"]:.3f}' for n in names), b)


F['p2_table1'] = table1()
F['p2_loop'] = loop()
F['p2_pareto'] = pareto()
F['p2_run'] = run_fig()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part2.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
