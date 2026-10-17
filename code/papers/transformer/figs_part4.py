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


# ---------------------------------------------------------------- 3. restricted attention: the window mask and its band
B = json.load(open(os.path.join(HERE, 'results', 'part4b.json')))


def local():
    M = B['local_mask']
    n, cell = len(M), 24
    r, half = 5, 2
    x0, y0 = 50, 60
    b = [text(20, 24, 'Restricted self-attention, n = 10, window r = 5', 't-title')]
    b.append(text(x0, y0 - 12, 'mask: all n x n = 100 pairs', 't-note'))
    for i in range(n):
        b.append(text(x0 - 8, y0 + i * cell + 16, f'q{i + 1}', 't-tick', 'end'))
        for j in range(n):
            cls = 'box-1' if M[i][j] else 'box-ghost'
            b.append(box(x0 + j * cell + 1, y0 + i * cell + 1, cell - 2, cell - 2, cls, 3))
    for j in range(n):
        b.append(text(x0 + j * cell + cell / 2, y0 + n * cell + 16, f'k{j + 1}', 't-tick', 'middle'))
    ax = x0 + n * cell + 24
    b.append(line(ax, y0 + n * cell / 2, ax + 60, y0 + n * cell / 2, 'edge-on', 'ah-on'))
    b.append(text(ax + 30, y0 + n * cell / 2 - 10, 'keep the band', 't-tick', 'middle'))
    x1 = ax + 100
    b.append(text(x1, y0 - 12, 'band: n x r = 50 slots', 't-note'))
    for i in range(n):
        b.append(text(x1 - 8, y0 + i * cell + 16, f'q{i + 1}', 't-tick', 'end'))
        for t in range(r):
            j = i + t - half
            ok = 0 <= j < n
            b.append(box(x1 + t * cell + 1, y0 + i * cell + 1, cell - 2, cell - 2, 'box-1' if ok else 'box-ghost', 3))
            if ok:
                b.append(text(x1 + t * cell + cell / 2, y0 + i * cell + 16, f'{j + 1}', 't-tick', 'middle'))
    for t in range(r):
        b.append(text(x1 + t * cell + cell / 2, y0 + n * cell + 16, f'{t - half:+d}' if t != half else '0', 't-tick', 'middle'))
    b.append(text(20, y0 + n * cell + 44, 'Left: a mask still computes all 100 scores and hides 56. Right: the band computes only the 44 '
                  'real ones (number = key position);', 't-muted'))
    b.append(text(20, y0 + n * cell + 60, 'the 6 empty slots at the ends are padding. The work falls from n x n to n x r.', 't-muted'))
    return svg(x1 + r * cell + 330, y0 + n * cell + 72, 'Restricted self-attention for 10 tokens and a window of 5. On the left, the 10 by 10 mask, '
               'with a diagonal band of 44 allowed pairs. On the right, the same band stored as 10 rows of 5 slots, so only the scores inside the window '
               'are computed.', b)


F['p4_local'] = local()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'), indent=1)
print('wrote', list(F))
