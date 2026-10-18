"""SVG figures for Part 4, drawn from results/part4.json and results/part4_llm.json (run mh_part4.py and
mh_part4_llm.py first) and from numbers printed in Tables 6 and 7 of the paper. Writes results/figs_part4.json."""
import json, os
from figlib import svg, text, heatmap, lines_chart
from samfig import hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part4.json')))
L = json.load(open(os.path.join(HERE, 'results', 'part4_llm.json')))
F = {}

# ---------------------------------------------------------------- BM25 saturation and length norm (our numbers)
sat = R['tiny']['sat']
F['p4_sat'] = lines_chart(
    760, 300, 'The term-frequency part of BM25 with k1 = 1.5 and b = 0.75, for 1 to 10 hits of one word. '
    'It grows fast at first and then flattens toward 2.5. A short document gets more credit per hit than a long one.',
    list(range(1, 11)),
    [('short doc (half the average length)', sat['0.5'], 'l1', 's1'),
     ('average-length doc', sat['1.0'], 'l2', 's2'),
     ('long doc (twice the average)', sat['2.0'], 'l3', 's3')],
    'times the word occurs in the document (tf)', 'tf part of the score', 0, 2.5, [0, 0.5, 1.0, 1.5, 2.0, 2.5],
    fmt=lambda v: f'{v:.2f}', right=260)

# ---------------------------------------------------------------- Table 6, average column (the paper's numbers)
T6 = [('No retriever', 34.1), ('Dense, k=1', 34.4), ('Dense, k=5', 38.1), ('Random few-shot', 32.2),
      ('BM25', 37.5), ('Meta-Harness', 38.8)]
b = hbars(20, 40, T6, 420, row_h=30, label_w=150, fmt=lambda v: f'{v:.1f}%', vmax=45, hi='Meta-Harness',
          title='Table 6: average pass@1 over five held-out models')
F['p4_table6'] = svg(680, 40 + 30 * len(T6) + 20, 'Bar chart of the average column of Table 6: no retriever 34.1, '
                     'dense k=1 34.4, dense k=5 38.1, random few-shot 32.2, BM25 37.5, Meta-Harness 38.8 percent.', b)

# ---------------------------------------------------------------- Table 7, Haiku 4.5 (the paper's numbers)
T7 = [('OpenHands', 13.9), ('Claude Code', 27.5), ('Terminus 2', 28.3), ('Mini-SWE-Agent', 29.8),
      ('Terminus-KIRA', 33.7), ('Goose', 35.5), ('Meta-Harness', 37.6)]
T7 = [(("Anthropic's agent" if n == 'Claude Code' else n), v) for n, v in T7]   # the site does not print that name
b = hbars(20, 40, T7, 420, row_h=30, label_w=150, fmt=lambda v: f'{v:.1f}%', vmax=45, hi='Meta-Harness',
          title='Table 7: TerminalBench-2 pass rate with Haiku 4.5')
F['p4_table7'] = svg(680, 40 + 30 * len(T7) + 20, 'Bar chart of the Haiku 4.5 rows of Table 7: the hand-written harnesses '
                     'score 13.9 to 35.5 percent, Meta-Harness 37.6 percent.', b)

# ---------------------------------------------------------------- our router: confusion table, rows add up to 1
routes = ['combinatorics', 'geometry', 'number_theory', 'default']
names = ['combinatorics', 'geometry', 'number theory', 'default']
conf = R['router']['conf']
M = []
for a in routes:
    row = [conf[f'{a}|{b}'] for b in routes]
    M.append([v / sum(row) for v in row])
body = heatmap(20, 50, M, names, names, 70, show_values=True, label_w=120)
body.append(text(20 + 120 + 140, 30, 'route chosen by the router', 't-muted', 'middle'))
body.append(text(20 + 112, 40, 'true route', 't-muted', 'end'))
F['p4_router'] = svg(460, 50 + 70 * 4 + 110, 'Confusion table of our lexical router on 4,509 MATH test problems. '
                     'Each row is a true route and adds up to 1. The diagonal holds 0.61, 0.83, 0.81 and 0.79.', body)

# ---------------------------------------------------------------- our tiny Table 6 with Qwen2.5-1.5B-Instruct
items = [(k, 100 * v['acc']) for k, v in L.items()]
n = len(next(iter(L.values()))['correct'])
b = hbars(20, 40, items, 400, row_h=30, label_w=150, fmt=lambda v: f'{v:.1f}%', vmax=100,
          title=f'Our run: Qwen2.5-1.5B-Instruct, {n} MATH test problems')
F['p4_llm'] = svg(680, 40 + 30 * len(items) + 20, 'Bar chart of our accuracy with four harnesses: '
                  + ', '.join(f'{k} {v:.1f} percent' for k, v in items) + '.', b)

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'))
print('figures:', ', '.join(F))
