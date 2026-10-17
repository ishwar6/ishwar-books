"""Figures for Part 6, drawn from results/part6.json. Writes results/figs_part6.json (key -> svg string).
Colours come only from CSS classes."""
import json, os
from figlib import svg, heatmap

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part6.json')))
F = {}


def heads(key, word, label):
    d = R[key]
    cols = [p.replace('▁', '') for p in d['pieces']]
    rows = [f'head {i + 1}' for i in range(len(d['attn']))]
    cell = 20
    b = heatmap(10, 34, d['attn'], rows, cols, cell, title=f'Where "{word}" looks, encoder layer 5 of 6', label_w=70)
    return svg(10 + 70 + cell * len(cols) + 20, 34 + cell * len(rows) + 110, label, b)


F['p6_making'] = heads('fig3', 'making', 'Attention weights from the word making to every token of the sentence, '
                       'for the 8 heads of encoder layer 5 of the released Marian English-to-German model.')
F['p6_its'] = heads('fig4', 'its', 'Attention weights from the word its to every token of the sentence, '
                    'for the 8 heads of encoder layer 5 of the released Marian English-to-German model.')

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part6.json'), 'w'), indent=1)
print('wrote results/figs_part6.json:', ', '.join(F))
