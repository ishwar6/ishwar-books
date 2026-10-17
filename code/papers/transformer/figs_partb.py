"""Figures for the Background page (the road to the Transformer), partly from results/partb.json.
Writes results/figs_partb.json (key -> svg string). Colours come only from CSS classes."""
import json, math, os
from figlib import svg, text, box, arrow, lines_chart
from samfig import token, line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'partb.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


# ---------------------------------------------------------------- 1. timeline
def timeline():
    ev = [('1997', 'LSTM', ['a memory that', 'lasts longer'], 's1'),
          ('2014', 'Seq2seq', ['sentence in,', 'one vector, out'], 's1'),
          ('2015', 'Attention', ['look back at', 'every word'], 's2'),
          ('2016', 'GNMT', ['16 LSTM layers,', '6 days, 96 GPUs'], 's1'),
          ('2017', 'ConvS2S', ['no recurrence,', 'convolutions'], 's2'),
          ('2017', 'Transformer', ['attention only,', 'all at once'], 's3')]
    xs = [75, 197, 319, 441, 563, 685]
    b = [line(40, 96, 735, 96, 'axis', 'ah')]
    for (yr, name, ls, cls), x in zip(ev, xs):
        b += [f'<circle class="{cls}" cx="{x}" cy="96" r="8"/>', text(x, 40, yr, 't-title', 'middle'), text(x, 64, name, 't-note', 'middle')]
        for k, l in enumerate(ls):
            b.append(text(x, 124 + 17 * k, l, 't-muted', 'middle'))
    b += legend(40, 190, [('s1', 'recurrent models (one step at a time)'), ('s2', 'steps toward parallel'), ('s3', 'the Transformer')])
    return svg(760, 204, 'Timeline: LSTM 1997, sequence to sequence 2014, attention 2015, GNMT 2016, ConvS2S 2017 and the Transformer 2017.', b)


F['pb_timeline'] = timeline()


# ---------------------------------------------------------------- 2. one vector versus attention
def bottleneck():
    words, out = ['the', 'black', 'cat'], ['le', 'chat', 'noir']
    b = [text(20, 24, 'Seq2seq (2014): everything goes through one vector', 't-title')]
    hx = [40, 130, 220]
    for i, w in enumerate(words):
        b += token(hx[i], 80, 70, f'h{i + 1}', 'box-1', 30)
        b += [text(hx[i] + 35, 132, w, 't-muted', 'middle')]
        if i:
            b.append(arrow(hx[i - 1] + 72, 95, hx[i] - 2, 95))
    b += token(330, 76, 60, 'c', 'box-on', 38, 't-note')
    b.append(arrow(292, 95, 328, 95))
    for i, w in enumerate(out):
        b += token(440 + 90 * i, 80, 70, f's{i + 1}', 'box-2', 30)
        b += [text(475 + 90 * i, 132, w, 't-muted', 'middle')]
        if i:
            b.append(arrow(440 + 90 * (i - 1) + 72, 95, 440 + 90 * i - 2, 95))
    b.append(arrow(392, 95, 438, 95))
    b.append(text(360, 60, 'fixed size', 't-tick', 'middle'))
    # attention row
    y0 = 200
    b.append(text(20, y0 - 14, 'Attention (2015): each output word looks back at every input word', 't-title'))
    for i, w in enumerate(words):
        b += token(hx[i], y0 + 70, 70, f'h{i + 1}', 'box-1', 30)
        b += [text(hx[i] + 35, y0 + 122, w, 't-muted', 'middle')]
    b += token(470, y0 + 10, 70, 's3', 'box-2', 30)
    b.append(text(560, y0 + 30, 'about to write "noir"', 't-muted'))
    alphas = R['attn']['alpha']
    for i in range(3):
        cls = 'path' if i == 1 else 'edge'
        b.append(f'<line class="{cls}" x1="{hx[i] + 35:.1f}" y1="{y0 + 68}" x2="468" y2="{y0 + 25}"/>')
        b.append(text(hx[i] + 60, y0 + 56 - 4 * i, f'α = {alphas[i]:.2f}', 't-tick'))
    b.append(text(330, y0 + 122, 'c = 0.04·h1 + 0.70·h2 + 0.26·h3: a new mix for each output word', 't-note'))
    return svg(760, 340, 'Top: a sequence to sequence model squeezes the whole input into one fixed vector c. Bottom: with attention, the decoder takes a weighted mix of all input vectors, with a new set of weights for each output word.', b)


F['pb_bottleneck'] = bottleneck()


# ---------------------------------------------------------------- 3. how far the gradient travels
def grad():
    g = R['grad']
    xs = g['dists']
    lg = lambda vs: [math.log10(v) for v in vs]
    return lines_chart(780, 320, 'Size of the gradient that reaches step 0 from step T, on a log scale. In an untrained tanh RNN it falls from about 4 to 2 times 10 to the minus 16 after 50 steps. In a default LSTM it falls to about 5 times 10 to the minus 11. In an LSTM whose forget gates start open, it stays near 2.',
                       xs, [('tanh RNN', lg(g['rnn']), 'l1', 's1'), ('LSTM, default start', lg(g['lstm']), 'l2', 's2'),
                            ('LSTM, forget gate open', lg(g['lstm_f3']), 'l3', 's3')],
                       'distance T in steps', 'size of the gradient (log scale)', -16, 2, [-16, -12, -8, -4, 0],
                       fmt=lambda v: f'1e{v:.0f}' if v != 0 else '1', right=230)


F['pb_grad'] = grad()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_partb.json'), 'w'))
print('figures:', ', '.join(F))
