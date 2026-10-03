"""Figures for Part 1, drawn from results/part1.json."""
import json
from figlib import svg, text, box, arrow, heatmap, lines_chart

R = json.load(open('results/part1.json'))
F = {}


def bottleneck():
    src = ['Le', 'chat', 'est', 'noir']
    b = [text(20, 24, 'Before attention: the decoder sees one summary vector', 't-note'),
         text(400, 24, 'With attention: the decoder looks back at every input', 't-note')]
    # left: encoder states squeezed into one vector
    for i, w in enumerate(src):
        x = 30 + i * 80
        b += [box(x, 44, 64, 30, 'box', 6), text(x + 32, 64, w, 't-tick', 'middle')]
        b += [box(x + 12, 96, 40, 22, 'box-2', 5)]
        b.append(arrow(x + 32, 76, x + 32, 94))
        if i < 3:
            b.append(arrow(x + 54, 107, x + 90, 107))
    b += [box(150, 150, 70, 30, 'box-2', 6), text(185, 170, 'one vector', 't-tick', 'middle'),
          arrow(304, 120, 222, 150), box(130, 214, 110, 34, 'box-3', 6), text(185, 236, 'The cat is …', 't-tick', 'middle'),
          arrow(185, 182, 185, 210), text(30, 274, 'Everything must squeeze through one fixed-size state.', 't-tick')]
    # right: attention weights to every encoder state
    weights = [0.06, 0.81, 0.08, 0.05]
    for i, (w, a) in enumerate(zip(src, weights)):
        x = 410 + i * 82
        b += [box(x, 44, 64, 30, 'box', 6), text(x + 32, 64, w, 't-tick', 'middle'), box(x + 12, 96, 40, 22, 'box-2', 5),
              arrow(x + 32, 76, x + 32, 94),
              f'<line class="{"path" if a > 0.5 else "edge"}" x1="{x + 32}" y1="120" x2="565" y2="212" style="stroke-width:{1 + a * 5:.1f}"/>',
              text(x + 32, 140, f'{a:.2f}', 't-val', 'middle')]
    b += [box(500, 214, 130, 34, 'box-3', 6), text(565, 236, 'The cat …', 't-tick', 'middle'),
          text(410, 274, 'To write "cat", it looks mostly at "chat".', 't-tick')]
    return svg(760, 290, 'Before attention, a decoder saw a single compressed vector. With attention, it looks back at every input word with learned weights.', b)
F['p1_bottleneck'] = bottleneck()


def pipeline():
    b = []
    b += [box(20, 120, 70, 44, 'box'), text(55, 147, 'X', 't-math', 'middle'), text(55, 182, 'tokens', 't-tick', 'middle')]
    for i, (n, cls) in enumerate((('Q', 'box-2'), ('K', 'box-1'), ('V', 'box-3'))):
        y = 40 + i * 80
        b += [arrow(92, 142, 150, y + 22), box(150, y, 64, 44, cls), text(182, y + 28, n, 't-math', 'middle'),
              text(120, y + 14 if i == 0 else y + 50, f'× W{n.lower()}', 't-tick', 'middle')]
    b += [arrow(216, 62, 270, 108), arrow(216, 142, 270, 120), box(270, 92, 120, 44, 'box'), text(330, 119, 'Q Kᵀ / √d', 't-math', 'middle'),
          text(330, 82, 'how well each pair matches', 't-tick', 'middle'),
          arrow(392, 114, 420, 114), box(420, 92, 86, 44, 'box'), text(463, 119, 'mask', 't-note', 'middle'), text(463, 82, 'hide the future', 't-tick', 'middle'),
          arrow(508, 114, 536, 114), box(536, 92, 86, 44, 'box'), text(579, 119, 'softmax', 't-note', 'middle'), text(579, 82, 'rows sum to 1', 't-tick', 'middle'),
          arrow(579, 138, 579, 176), box(536, 180, 86, 44, 'box-1'), text(579, 207, 'A', 't-math', 'middle'),
          arrow(216, 222, 532, 202), text(380, 236, 'A × V: a weighted mix of values', 't-tick', 'middle'),
          arrow(624, 202, 664, 202), box(664, 180, 76, 44, 'box-3'), text(702, 207, 'Z', 't-math', 'middle'),
          text(702, 240, 'new tokens', 't-tick', 'middle')]
    return svg(760, 260, 'The self-attention pipeline: project tokens into queries, keys and values, score every pair, mask the future, softmax into weights, and mix the values.', b)
F['p1_pipeline'] = pipeline()

t = R['toy']
F['p1_toy'] = svg(760, 290, 'The attention matrix of the four-token example: each row is one token, masked future positions are empty, and every row sums to 1.',
                  heatmap(230, 40, t['A'], t['tokens'], t['tokens'], 54, show_values=True, mask_upper=True, label_w=70)
                  + [text(560, 70, 'each row: where one token looks', 't-tick'), text(560, 90, 'empty cells: the future (masked)', 't-tick'),
                     text(560, 110, 'every row sums to 1', 't-tick')])

s = R['scaling']
ds = [int(k) for k in s]
F['p1_scaling'] = lines_chart(760, 300, 'The largest softmax weight as the vector size grows, with and without dividing the scores by the square root of d.',
                              ds, [('without ÷√d', [s[str(d)]['max_weight_raw'] for d in ds], 'l2', 's2'),
                                   ('with ÷√d', [s[str(d)]['max_weight_scaled'] for d in ds], 'l1', 's1')],
                              'vector size d (log scale)', 'largest weight in a 64-key softmax', 0, 1, [0, 0.25, 0.5, 0.75, 1.0], xlog=True,
                              fmt=lambda v: f'{v:.2f}')


def heads():
    b = [box(20, 112, 80, 40, 'box'), text(60, 137, 'x', 't-math', 'middle')]
    for i in range(4):
        y = 22 + i * 58
        b += [arrow(102, 132, 150, y + 20), box(150, y, 150, 40, 'box-1' if i == 0 else 'box', 8),
              text(225, y + 25, f'head {i + 1}: own Wq, Wk, Wv', 't-tick', 'middle'),
              arrow(302, y + 20, 340, y + 20), box(340, y, 70, 40, 'box', 8), text(375, y + 25, f'A{i + 1}·V{i + 1}', 't-tick', 'middle'),
              arrow(412, y + 20, 470, 132)]
    b += [box(470, 104, 110, 56, 'box-2', 10), text(525, 129, 'concatenate', 't-note', 'middle'), text(525, 148, 'all heads', 't-tick', 'middle'),
          arrow(582, 132, 620, 132), box(620, 112, 70, 40, 'box-3', 8), text(655, 137, '× Wo', 't-math', 'middle'),
          text(400, 270, 'Each head learns its own pattern; the output projection blends them back into one vector per token.', 't-tick', 'middle')]
    return svg(760, 284, 'Multi-head attention runs several attention heads in parallel, each with its own projections, concatenates their outputs and mixes them with an output projection.', b)
F['p1_heads'] = heads()

r = R['real']
words = r['tokens']
b = []
for k, (key, title) in enumerate((('sink', 'attention sink'), ('previous', 'previous token'), ('coref', 'pronoun → noun'))):
    h = r[key]
    b += heatmap(10 + k * 252, 44, h['matrix'], words, words, 15.5, title=f"{title} (L{h['layer']} H{h['head']})", mask_upper=True, label_w=62)
F['p1_real'] = svg(770, 300, 'Three real attention heads from Qwen2.5-0.5B on the sentence "The cat sat on the mat because it was tired": one dumps attention on the first token, one looks at the previous token, one links "it" to "cat".', b)

json.dump(F, open('results/figs_part1.json', 'w'))
print(len(F), 'figures:', list(F))
