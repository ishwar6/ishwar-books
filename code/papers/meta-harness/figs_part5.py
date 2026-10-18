"""Figures for Part 5, drawn from results/part5.json (written by mh_part5.py).
p5_scatter: our 12 prompt harnesses, search-set vs test accuracy, with the line y = x (our small Figure 7).
p5_curse:   how much the best of N equal candidates is inflated on a search set of n examples (exact values).
Run: python figs_part5.py  ->  results/figs_part5.json"""
import json, os
from figlib import svg, text, esc, lines_chart

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part5.json')))
F = {}

# ---- p5_scatter
res, chosen, best = R['real']['results'], R['real']['chosen'], R['real']['best_test']
W, H, L, T, B, RT = 640, 430, 70, 30, 56, 190
lo, hi = 60, 100
pw, ph = W - L - RT, H - T - B
X = lambda v: L + (v - lo) / (hi - lo) * pw
Y = lambda v: T + ph - (v - lo) / (hi - lo) * ph
b = []
for t in range(lo, hi + 1, 10):
    b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>',
          f'<line class="grid" x1="{X(t):.1f}" y1="{T}" x2="{X(t):.1f}" y2="{T + ph}"/>',
          text(L - 8, Y(t) + 4, f'{t}%', 't-tick', 'end'), text(X(t), T + ph + 20, f'{t}%', 't-tick', 'middle')]
b.append(f'<line class="axis" x1="{X(lo):.1f}" y1="{Y(lo):.1f}" x2="{X(hi):.1f}" y2="{Y(hi):.1f}" style="stroke-dasharray:5 5"/>')
b.append(text(X(hi) - 4, Y(hi) + 16, 'y = x', 't-muted', 'end'))
for name, r in res.items():
    dc = 's1' if name.startswith('A') else 's2'
    b.append(f'<g class="mark"><title>{esc(name)}: search {r["search_acc"]:.1f}%, test {r["test_acc"]:.1f}%</title>'
             f'<circle class="{dc} ring" cx="{X(r["search_acc"]):.1f}" cy="{Y(r["test_acc"]):.1f}" r="6"/></g>')
for name, lab, dy in [(chosen, 'chosen by search score', -14), (best, 'best on test', 22)]:
    r = res[name]
    b.append(f'<circle class="axis" cx="{X(r["search_acc"]):.1f}" cy="{Y(r["test_acc"]):.1f}" r="11" style="fill:none;stroke-width:2"/>')
    b.append(text(X(r['search_acc']) - 14, Y(r['test_acc']) + dy, f'{lab}: {name}', 't-note', 'end'))
lx = L + pw + 22
b += [f'<rect class="s1" x="{lx}" y="{T + 10}" width="10" height="10" rx="2"/>', text(lx + 16, T + 19, 'template A (Review: ...)', 't-note'),
      f'<rect class="s2" x="{lx}" y="{T + 32}" width="10" height="10" rx="2"/>', text(lx + 16, T + 41, 'template B (Text: ...)', 't-note'),
      text(lx, T + 70, 'Each dot is one prompt', 't-muted'), text(lx, T + 86, 'harness around', 't-muted'),
      text(lx, T + 102, 'Qwen2.5-0.5B-Instruct.', 't-muted'), text(lx, T + 126, 'Dots above the line did', 't-muted'),
      text(lx, T + 142, 'better on the test set', 't-muted'), text(lx, T + 158, 'than on the search set.', 't-muted')]
b.append(text(L + pw / 2, H - 12, 'search-set accuracy (60 examples)', 't-tick', 'middle'))
b.append(text(L - 56, T - 12, 'test accuracy (300 examples)', 't-tick'))
F['p5_scatter'] = svg(W, H, 'Search-set accuracy against test accuracy for 12 prompt harnesses on SST-2, with the line y = x', b)

# ---- p5_curse
tab = R['winners_curse']['table']
xs = [1, 4, 10, 40, 109]
ns = [50, 88, 100, 250]
series = [(f'search set {n}', [tab[str(x)][i] for x in xs], f'l{i + 1}', f's{i + 1}') for i, n in enumerate(ns)]
F['p5_curse'] = lines_chart(640, 340, 'Inflation of the best search score against the number of candidates, for four search-set sizes',
                            xs, series, 'number of candidates (log scale)', 'points above the true 40%', 0, 20,
                            [0, 5, 10, 15, 20], xlog=True, right=190, fmt=lambda v: f'{v:.1f}' if isinstance(v, float) else f'{v}')

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part5.json'), 'w'), indent=1)
print('wrote', ', '.join(F))
