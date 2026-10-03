"""Figures for Part 4, drawn from results/part4.json and results/part4_train.json."""
import json, os
from figlib import svg, text, box, arrow, bar_path, lines_chart

R = json.load(open('results/part4.json'))
TR = json.load(open('results/part4_train.json')) if os.path.exists('results/part4_train.json') else None
F = {}


def memory_kinds():
    b = [text(20, 24, 'Softmax attention: a list that keeps growing', 't-title'),
         text(20, 168, 'Linear attention / Gated DeltaNet: one fixed-size memory', 't-title')]
    for i in range(9):
        x = 20 + i * 62
        b += [box(x, 40, 52, 26, 'box-3', 5), text(x + 26, 58, f'k{i + 1} v{i + 1}', 't-tick', 'middle')]
    b += [text(590, 58, '… + one more per token', 't-tick'),
          text(20, 92, 'Each new token reads ALL stored keys and values. Memory and work grow with the text.', 't-tick'),
          text(20, 112, 'Nothing is ever forgotten, so recall is exact, but at 1M tokens the list is huge.', 't-tick')]
    b += [box(20, 186, 120, 76, 'box-1', 8), text(80, 222, 'memory S', 't-note', 'middle'), text(80, 242, 'd × d numbers', 't-tick', 'middle'),
          arrow(150, 224, 210, 224, on=True), box(210, 196, 150, 56, 'box', 8), text(285, 220, 'write: add or', 't-tick', 'middle'),
          text(285, 238, 'correct one fact', 't-tick', 'middle'), arrow(362, 224, 420, 224, on=True),
          box(420, 186, 120, 76, 'box-1', 8), text(480, 222, 'memory S', 't-note', 'middle'), text(480, 242, 'same size', 't-tick', 'middle'),
          text(560, 214, 'Every token does one small', 't-tick'), text(560, 232, 'update. The size never changes,', 't-tick'),
          text(560, 250, 'so old facts must share space.', 't-tick')]
    return svg(780, 280, 'Softmax attention keeps every key and value in a growing list; linear attention and its successors keep one fixed-size memory matrix and update it once per token.', b)
F['p4_kinds'] = memory_kinds()


def assoc():
    b = [text(190, 24, 'softmax: (Q Kᵀ) first', 't-title', 'middle'), text(580, 24, 'linear: (Kᵀ V) first', 't-title', 'middle')]
    b += [box(40, 50, 40, 120, 'box-2', 4), text(60, 190, 'Q', 't-math', 'middle'), text(60, 206, 'T × d', 't-tick', 'middle'),
          text(95, 115, '×', 't-note', 'middle'), box(110, 90, 120, 40, 'box-1', 4), text(170, 150, 'Kᵀ', 't-math', 'middle'),
          text(170, 166, 'd × T', 't-tick', 'middle'), text(245, 115, '=', 't-note', 'middle'),
          box(260, 50, 120, 120, 'box', 4), text(320, 115, 'T × T', 't-note', 'middle'), text(320, 190, 'grows with T²', 't-tick', 'middle')]
    b += [box(440, 90, 120, 40, 'box-1', 4), text(500, 150, 'Kᵀ', 't-math', 'middle'), text(500, 166, 'd × T', 't-tick', 'middle'),
          text(575, 115, '×', 't-note', 'middle'), box(590, 50, 40, 120, 'box-3', 4), text(610, 190, 'V', 't-math', 'middle'),
          text(610, 206, 'T × d', 't-tick', 'middle'), text(645, 115, '=', 't-note', 'middle'),
          box(660, 90, 40, 40, 'box', 4), text(680, 150, 'd × d', 't-note', 'middle'), text(680, 166, 'fixed size', 't-tick', 'middle'),
          text(20, 240, 'Same numbers, different order. With softmax in the middle you must build the T × T grid;', 't-tick'),
          text(20, 258, 'without softmax you can multiply Kᵀ V first and only ever keep a small d × d matrix.', 't-tick')]
    return svg(760, 272, 'Matrix multiplication is associative: (Q K transposed) V equals Q (K transposed V). The second order never builds the T by T grid.', b)
F['p4_assoc'] = assoc()


def overwrite():
    o = R['overwrite']
    rows = [('read x, linear attention', o['linear_x'], 's2'), ('read x, delta rule', o['delta_x'], 's1'),
            ('read y, linear attention', o['linear_y'], 's2'), ('read y, delta rule', o['delta_y'], 's1')]
    L, W, T0, band, vmax = 230, 760, 46, 36, 8
    b = [text(20, 24, 'Write "x = 1", "y = 7", then "x = 5". What comes back?', 't-title')]
    for i, (name, v, cls) in enumerate(rows):
        y = T0 + i * band
        bw = v / vmax * (W - L - 150)
        b += [f'<g class="mark"><title>{name}: {v:.2f}</title><path class="{cls} bar-mark" d="{bar_path(L, y, bw, 22, horizontal=True)}"/></g>',
              text(L - 10, y + 16, name, 't-tick', 'end'), text(L + bw + 8, y + 16, f'{v:.2f}', 't-val')]
    xr = L + 5 / vmax * (W - L - 150)
    b += [f'<line class="axis" x1="{xr:.1f}" y1="{T0 - 6}" x2="{xr:.1f}" y2="{T0 + 2 * band - 8}" stroke-dasharray="4 3"/>',
          text(xr + 4, T0 - 10, 'correct x = 5', 't-tick'),
          text(20, T0 + 4 * band + 18, 'Linear attention can only ADD, so x becomes 1 + 5 = 6. The delta rule replaces the old value: x = 5.', 't-tick')]
    return svg(W, T0 + 4 * band + 32, 'Overwriting a fact: linear attention returns the sum of the old and new value, the delta rule returns the new value.', b)
F['p4_overwrite'] = overwrite()

c = R['capacity']
F['p4_capacity'] = lines_chart(760, 300, 'Error when reading back n random facts from a 64 by 64 memory, for linear attention and the delta rule.',
                               c['n'], [('linear attention', c['linear'], 'l2', 's2'), ('delta rule', c['delta'], 'l1', 's1')],
                               'facts stored in a 64 × 64 memory (log scale)', 'read-back error (0 = perfect)', 0, 2.2, [0, 0.5, 1.0, 1.5, 2.0],
                               xlog=True, xticks=[4, 16, 64, 256], fmt=lambda v: f'{v:.2f}', right=190)

fd = R['fade']
F['p4_fade'] = lines_chart(760, 290, 'How much of a stored fact remains after t more tokens, for forget gates alpha of 0.9, 0.99 and 0.999.',
                           fd['steps'][1:], [(f'α = {a}', fd['alphas'][a][1:], f'l{i + 1}', f's{i + 1}') for i, a in enumerate(fd['alphas'])],
                           'tokens since the fact was written', 'strength left (α^t)', 0, 1, [0, 0.25, 0.5, 0.75, 1.0],
                           xticks=[10, 100, 250, 500], fmt=lambda v: f'{v:.2f}', right=160)


def gdn_step():
    b = [text(20, 24, 'One Gated DeltaNet step for token t', 't-title')]
    steps = [('1. forget', 'S ← αₜ · S', 'shrink every old fact', 'box-2'),
             ('2. look up', 'old = S kₜ', 'what is stored at kₜ now?', 'box'),
             ('3. correct', 'S ← S + βₜ (vₜ − old) kₜᵀ', 'move it toward vₜ', 'box-1'),
             ('4. read', 'oₜ = S qₜ', 'answer the query', 'box-3')]
    for i, (h, eq, sub, cls) in enumerate(steps):
        x = 20 + i * 186
        b += [box(x, 44, 166, 86, cls, 10), text(x + 83, 68, h, 't-note', 'middle'), text(x + 83, 94, eq, 't-tick', 'middle'),
              text(x + 83, 116, sub, 't-tick', 'middle')]
        if i < 3:
            b.append(arrow(x + 168, 87, x + 184, 87))
    b += [text(20, 160, 'αₜ (between 0 and 1): how much of the old memory to keep. Near 1 = remember, near 0 = wipe.', 't-tick'),
          text(20, 180, 'βₜ (between 0 and 1): how strongly to write. 1 = fully replace what was stored at kₜ.', 't-tick'),
          text(20, 200, 'Both are computed from the token itself, so the model decides, token by token, what to forget and what to write.', 't-tick')]
    return svg(780, 214, 'One step of Gated DeltaNet: decay the memory by alpha, look up the value stored under the key, correct it towards the new value by beta, then read with the query.', b)
F['p4_gdn'] = gdn_step()


def gate_fig():
    b = [box(20, 70, 90, 44, 'box'), text(65, 97, 'x', 't-math', 'middle'),
         arrow(112, 92, 160, 92), box(160, 70, 140, 44, 'box-1'), text(230, 90, 'attention', 't-note', 'middle'), text(230, 106, '(softmax)', 't-tick', 'middle'),
         arrow(302, 92, 380, 92), text(340, 84, 'Y', 't-math', 'middle'),
         f'<circle class="node" cx="400" cy="92" r="18"/>', text(400, 98, '×', 't-note', 'middle'),
         f'<path class="edge" d="M65,116 V170 H400 V112" fill="none" marker-end="url(#ah)"/>',
         box(250, 150, 120, 40, 'box-2', 8), text(310, 175, 'σ(x Wθ)', 't-note', 'middle'),
         arrow(420, 92, 470, 92), box(470, 70, 110, 44, 'box-3'), text(525, 97, '× Wo', 't-note', 'middle'),
         arrow(582, 92, 620, 92), text(630, 97, 'output', 't-note'),
         text(20, 30, 'Gated attention: each output number is multiplied by a gate between 0 and 1, computed from the token x.', 't-tick'),
         text(20, 222, 'A gate near 0 lets a head say "nothing useful here" without dumping attention on the first token.', 't-tick')]
    return svg(760, 236, 'Gated attention: the attention output Y is multiplied elementwise by a sigmoid gate computed from the input, before the output projection.', b)
F['p4_gate'] = gate_fig()


def strips():
    rows = [('Qwen3-Next-80B (48 layers)', ['g' if (i + 1) % 4 else 'a' for i in range(48)]),
            ('Kimi Linear 48B (27 layers)', ['a' if i + 1 in (4, 8, 12, 16, 20, 24, 27) else 'g' for i in range(27)]),
            ('Nemotron-H-8B (52 layers)', [{'M': 'm', '-': 'f', '*': 'a'}[ch] for ch in 'M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-'])]
    cls = {'g': 's1', 'a': 's2', 'm': 's3', 'f': 'box'}
    tip = {'g': 'linear (Gated DeltaNet / KDA)', 'a': 'full attention', 'm': 'Mamba-2', 'f': 'MLP only (no token mixing)'}
    b = []
    for r, (name, pat) in enumerate(rows):
        y = 30 + r * 64
        b.append(text(20, y, name, 't-note'))
        w = min(12, 720 / len(pat))
        for i, k in enumerate(pat):
            b.append(f'<g class="mark"><title>layer {i + 1}: {tip[k]}</title><rect class="{cls[k]}" x="{20 + i * w:.1f}" y="{y + 10}" width="{w - 2:.1f}" height="26" rx="2"/></g>')
    ly = 30 + 3 * 64 + 4
    for j, (k, lab) in enumerate((('g', 'linear: Gated DeltaNet or KDA'), ('m', 'Mamba-2'), ('a', 'full attention'), ('f', 'MLP only'))):
        x = 20 + j * 190
        b += [f'<rect class="{cls[k]}" x="{x}" y="{ly}" width="12" height="12" rx="2"/>', text(x + 18, ly + 11, lab, 't-tick')]
    return svg(760, ly + 26, 'Layer patterns of three hybrid models: Qwen3-Next, Kimi Linear and Nemotron-H. Most layers are linear or Mamba-2; only a few are full attention.', b)
F['p4_hybrids'] = strips()

m = R['memory']['qwen3_next']['curve']
F['p4_memory'] = lines_chart(760, 300, 'Memory for one conversation in Qwen3-Next-80B: if all 48 layers were attention, versus the real hybrid with 12 attention layers plus fixed Gated DeltaNet state.',
                             m['context'], [('all 48 layers attention', m['all_attention_gib'], 'l2', 's2'), ('real hybrid (12 attention)', m['hybrid_gib'], 'l1', 's1')],
                             'tokens in the conversation (log scale)', 'memory, GiB', 0, 24, [0, 6, 12, 18, 24], xlog=True,
                             xticks=[4096, 16384, 65536, 262144], fmt=lambda v: f'{v:.2f}', right=220)

t = R['decode_timing']['results']
ns = [int(n) for n in t]
F['p4_decode'] = lines_chart(760, 300, 'Time for one decode step of one layer: gated softmax attention over a growing cache versus one Gated DeltaNet state update.',
                             ns, [('gated attention', [t[str(n)]['attention_ms'] for n in ns], 'l2', 's2'),
                                  ('Gated DeltaNet', [t[str(n)]['gdn_ms'] for n in ns], 'l1', 's1')],
                             'tokens so far (log scale)', 'milliseconds per step', 0, 6, [0, 2, 4, 6], xlog=True,
                             xticks=[4096, 16384, 65536, 262144], fmt=lambda v: f'{v:.2f}', right=190)

if TR:
    order = [k for k in ('softmax', 'gated', 'linear', 'gdn', 'hybrid') if k in TR]
    nice = {'softmax': 'softmax attention', 'gated': 'gated attention', 'linear': 'linear attention', 'gdn': 'Gated DeltaNet', 'hybrid': 'hybrid 3 GDN : 1 gated'}
    colour = {'softmax': 's2', 'gated': 's4', 'linear': 's3', 'gdn': 's1', 'hybrid': 's1'}
    L, W, T0, band = 210, 760, 14, 36
    lo, hi = 1.6, max(TR[k]['val_bits_per_byte'] for k in order) + 0.05
    b = []
    for i, k in enumerate(order):
        v = TR[k]['val_bits_per_byte']; y = T0 + i * band
        bw = (v - lo) / (hi - lo) * (W - L - 90)
        b += [f'<g class="mark"><title>{nice[k]}: {v:.3f} bits per byte</title><path class="{colour[k]} bar-mark" d="{bar_path(L, y, bw, 22, horizontal=True)}"/></g>',
              text(L - 10, y + 16, nice[k], 't-tick', 'end'), text(L + bw + 8, y + 16, f'{v:.3f}', 't-val')]
    b.append(text(L, T0 + len(order) * band + 18, f'bits per byte on a held-out book (lower is better); bars start at {lo}', 't-tick'))
    F['p4_lm'] = svg(W, T0 + len(order) * band + 30, 'Next-byte prediction on a held-out book for three small models that differ only in their token-mixing layers.', b)
    ns2 = [32, 64, 128, 256]
    F['p4_recall'] = lines_chart(760, 320, 'Recall accuracy as the number of stored facts grows, for the three small models (seed 0, 3,000 training steps).',
                                 ns2, [(nice[k], [TR[k]['recall_accuracy'][str(n)] for n in ns2], f'l{c}', f's{c}') for k, c in (('softmax', 2), ('gated', 4), ('linear', 3)) if k in TR],
                                 'facts to remember (log scale)', 'share of lookups answered correctly', 0, 1, [0, 0.25, 0.5, 0.75, 1.0],
                                 xlog=True, xticks=ns2, fmt=lambda v: f'{v:.2f}', right=230)

json.dump(F, open('results/figs_part4.json', 'w'))
print(len(F), 'figures:', list(F))
