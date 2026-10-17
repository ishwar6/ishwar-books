"""Figures for Part 4 (Why Self-Attention), drawn from results/part4.json. Writes results/figs_part4.json.
Colours come only from CSS classes."""
import json, math, os
from figlib import svg, text, box, lines_chart
from samfig import line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part4.json')))
F = {}


# ---------------------------------------------------------------- 1. the path from the first word to the last output
def paths():
    n, pw, gap = 8, 240, 20
    Y_IN, Y_MID, Y_OUT = 250, 165, 80
    b = [text(20, 24, 'How many steps from word 1 to the output at word 8?', 't-title')]

    def xs(p):
        x0 = 20 + p * (pw + gap)
        return [x0 + 22 + i * (pw - 44) / (n - 1) for i in range(n)]

    def node(x, y, cls='box'):
        return box(x - 11, y - 11, 22, 22, cls, 11)

    # recurrent: a chain
    X = xs(0)
    b += [text(X[0] - 10, 52, 'Recurrent: 7 steps, one after another', 't-note')]
    for i in range(n - 1):
        b.append(line(X[i] + 11, Y_MID, X[i + 1] - 11, Y_MID, 'edge-on'))
    for i in range(n):
        b.append(line(X[i], Y_IN - 11, X[i], Y_MID + 11, 'edge-on' if i == 0 else 'edge-dim'))
        b += [node(X[i], Y_IN), node(X[i], Y_MID, 'box-1')]
    b.append(line(X[7], Y_MID - 11, X[7], Y_OUT + 11, 'edge-on'))
    b.append(node(X[7], Y_OUT, 'box-3'))
    # dilated convolution, k = 3, dilations 1 and 3
    X = xs(1)
    b += [text(X[0] - 10, 52, 'Dilated convolution (k = 3): 2 layers', 't-note')]
    for c in (1, 4, 7):
        for j in (c - 1, c, c + 1):
            if 0 <= j < n:
                b.append(line(X[j], Y_IN - 11, X[c], Y_MID + 11, 'edge-on' if (j == 0 and c == 1) else 'edge-dim'))
        b.append(line(X[c], Y_MID - 11, X[7], Y_OUT + 11, 'edge-on' if c == 1 else 'edge-dim'))
    for i in range(n):
        b += [node(X[i], Y_IN)]
    for c in (1, 4, 7):
        b.append(node(X[c], Y_MID, 'box-1'))
    b.append(node(X[7], Y_OUT, 'box-3'))
    # self-attention: one step from every word
    X = xs(2)
    b += [text(X[0] - 10, 52, 'Self-attention: 1 step from every word', 't-note')]
    for i in range(n):
        b.append(line(X[i], Y_IN - 11, X[7], Y_OUT + 11, 'edge-on' if i == 0 else 'edge-dim'))
        b.append(node(X[i], Y_IN))
    b.append(node(X[7], Y_OUT, 'box-3'))
    for p in range(3):
        X = xs(p)
        b += [text(X[0], Y_IN + 30, 'word 1', 't-tick', 'middle'), text(X[7], Y_IN + 30, 'word 8', 't-tick', 'middle')]
    b.append(text(20, 312, 'Bold lines: the shortest path from word 1 to the output at word 8. The longer the path, the harder the signal is to learn.', 't-muted'))
    return svg(780, 326, 'Three ways to connect 8 words. A recurrent layer passes the signal from word 1 through a chain of 7 steps. '
               'A dilated convolution with kernel 3 needs 2 layers. A self-attention layer connects every word to the output in 1 step.', b)


F['p4_paths'] = paths()


# ---------------------------------------------------------------- 2. gradient against distance
def grad():
    G = R['grad']
    n = G['n']
    dists = [0, 1, 2, 5, 10, 15, 20, 25, 30, 35, 40, 45, 49]
    rn = [math.log10(G['rnn'][n - 1 - t]) for t in dists]
    at = [math.log10(G['attn'][n - 1 - t]) for t in dists]
    return lines_chart(780, 320, 'Size of the gradient of the last output with respect to each input, against the distance between them, '
                       'mean of 20 random networks with 64 numbers per vector and 50 positions. For the recurrent network it falls from about 5 '
                       'at distance 0 to about 1.6 times 10 to the minus 9 at distance 49. For one self-attention layer it stays between about 0.18 and 0.31 '
                       'at every distance above 0.',
                       dists, [('recurrent (tanh)', rn, 'l2', 's2'), ('self-attention', at, 'l3', 's3')],
                       'distance between the input and the last position (in positions)', 'gradient size (log scale)', -9, 1,
                       [-9, -7, -5, -3, -1, 1], xticks=[0, 10, 20, 30, 40, 49],
                       fmt=lambda v: f'10^{v:.0f}' if float(v).is_integer() else f'10^{v:.1f}', right=230)


F['p4_grad'] = grad()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'), indent=1)
print('wrote', list(F))
