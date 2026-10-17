"""Figures for Part 1 (the big idea). Writes results/figs_part1.json (key -> svg string). CSS classes only."""
import json, os
from figlib import svg, text, box, arrow
from samfig import token, line

HERE = os.path.dirname(os.path.abspath(__file__))
F = {}
WORDS = ['The', 'cat', 'sat', 'on', 'the', 'mat']


def loop_vs_attention():
    W, w, gap, x0 = 720, 74, 26, 60
    cx = [x0 + i * (w + gap) + w / 2 for i in range(6)]
    b = [text(20, 28, 'A recurrent network: one state after the other', 't-title')]
    y_word, y_h = 120, 62
    for i, s in enumerate(WORDS):
        b += token(cx[i] - w / 2, y_word, w, s, 'box')
        b += token(cx[i] - w / 2, y_h, w, f'h{i + 1}', 'box-on' if i == 5 else 'box-1')
        b.append(arrow(cx[i], y_word, cx[i], y_h + 32))
        if i:
            b.append(arrow(cx[i - 1] + w / 2, y_h + 15, cx[i] - w / 2 - 2, y_h + 15, on=True))
    b.append(text(20, 178, 'Step t must wait for step t - 1. The word "The" reaches h6 through 5 later states.', 't-note'))
    y2 = 230
    b.append(text(20, y2, 'Self-attention: every word looks at every word in one step', 't-title'))
    y_out, y_in = y2 + 34, y2 + 150
    for i, s in enumerate(WORDS):
        b += token(cx[i] - w / 2, y_in, w, s, 'box')
        b += token(cx[i] - w / 2, y_out, w, s + "'", 'box-on' if i == 5 else 'box-1')
    for j in range(6):                       # the new "mat" reads all six words at once
        b.append(line(cx[j], y_in, cx[5], y_out + 32, 'path', 'ah-on'))
    for i in range(5):                       # every other word does the same (drawn faint)
        for j in range(6):
            b.append(line(cx[j], y_in, cx[i], y_out + 32, 'edge'))
    b.append(text(20, y_in + 58, 'All six outputs are computed together. "The" reaches the new "mat" in one step.', 't-note'))
    return svg(W, y_in + 72, 'Two diagrams. Top: a recurrent network reads six words one at a time, and each state h1 to h6 '
               'waits for the one before it. Bottom: self-attention computes six new word vectors at once, each one reading all six words.', b)


F['p1_loop'] = loop_vs_attention()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
