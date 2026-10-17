"""Figures for Part 5 (two-layer models, how heads compose), drawn from results/part5.json and results/part5_toy.json.
Writes results/figs_part5.json (key -> svg string). Colours come only from CSS classes, so every figure follows the
site's dark, dim and light themes."""
import json
from figlib import svg, text, box, arrow, esc, heatmap
from cfig import hbars, line

R = json.load(open('results/part5.json'))
T = json.load(open('results/part5_toy.json'))
F = {}
NH = 8


def num(v):
    return f'{v:g}'


def cells(x, y, M, cls, cw=44, ch=28, fmt=num):
    b = []
    m = max(1e-9, max(abs(v) for r in M for v in r))
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            op = 0.10 + 0.55 * abs(v) / m
            b.append(f'<rect class="{cls}" x="{x + j * cw:.1f}" y="{y + i * ch:.1f}" width="{cw - 4}" height="{ch - 4}" rx="4" style="fill-opacity:{op:.2f}"/>')
            b.append(text(x + j * cw + (cw - 4) / 2, y + i * ch + ch / 2 + 2, fmt(v), 't-val', 'middle'))
    return b


# ---------------------------------------------------------------- 1. the three kinds of composition
def three_kinds():
    b = [text(24, 26, 'A layer-0 head writes into the stream. A layer-1 head can read that write in three places.', 't-title')]
    # the stream as a vertical band
    sx = 70
    b.append(box(sx, 50, 34, 300, 'box', 8))
    b.append(text(sx + 17, 370, 'stream', 't-tick', 'middle'))
    # layer-0 head
    b.append(box(150, 270, 170, 44, 'box-1', 8))
    b.append(text(235, 297, 'layer-0 head h₁', 't-note', 'middle'))
    b.append(arrow(sx + 34, 300, 148, 300))
    b.append(line(320, 292, 340, 292, 'path'))
    b.append(line(340, 292, 340, 236, 'path'))
    b.append(line(340, 236, sx + 40, 236, 'path', 'ah-on'))
    b.append(text(350, 268, 'writes W_OV of h₁', 't-tick'))
    # layer-1 head with three inputs
    hx, hy = 470, 60
    b.append(box(hx, hy, 260, 160, 'box', 10))
    b.append(text(hx + 130, hy + 22, 'layer-1 head h₂', 't-note', 'middle'))
    rows = [('W_Q: query', 'Q-composition: where to look, from the reading side', 's1'),
            ('W_K: key', 'K-composition: what each token offers to be found', 's2'),
            ('W_V: value', 'V-composition: what is carried once it looks', 's3')]
    for i, (lab, note, cls) in enumerate(rows):
        y = hy + 44 + i * 38
        b.append(f'<rect class="{cls}" x="{hx + 14}" y="{y}" width="96" height="28" rx="5" style="fill-opacity:0.35"/>')
        b.append(text(hx + 62, y + 19, lab, 't-tick', 'middle'))
        b.append(line(sx + 34, y + 14, hx + 12, y + 14, 'edge', 'ah'))
        b.append(text(hx + 118, y + 19, note.split(':')[0], 't-tick'))
    b.append(text(140, 392, 'Q and K change the attention pattern of h₂. V changes what h₂ moves: h₂ then acts like a new head.', 't-muted'))
    return svg(760, 410, 'A diagram of the residual stream as a vertical band. A layer-0 head reads the stream and writes its output back into it. A layer-1 head reads the stream three times: through W_Q for its queries, W_K for its keys and W_V for its values. If W_Q reads what the layer-0 head wrote, that is Q-composition; if W_K does, K-composition; if W_V does, V-composition.', b)


F['p5_three_kinds'] = three_kinds()


# ---------------------------------------------------------------- 2. the toy path expansion
def toy_expansion():
    A = T['A']
    b = [text(24, 26, 'Two layers, one head each, a 1-number stream: the output is a sum of four paths', 't-title')]
    rows = [('direct path: x₀', A['direct'], 's1'), ('head 1: w₁ A₁ x₀', A['h1'], 's2'), ('head 2: w₂ A₂ x₀', A['h2'], 's3'),
            ('virtual head: w₂ w₁ A₂A₁ x₀', A['virt'], 's2'), ('sum of the four', A['sum'], 's1'), ('the model: x₂', A['x2'], 's1')]
    for j, t in enumerate(['token 1', 'token 2', 'token 3']):
        b.append(text(300 + j * 70 + 30, 60, t, 't-tick', 'middle'))
    for i, (lab, vals, cls) in enumerate(rows):
        y = 74 + i * 38 + (12 if i >= 4 else 0)
        b.append(text(280, y + 19, lab, 't-tick t-strong' if i >= 4 else 't-tick', 'end'))
        b += cells(300, y, [vals], cls, 70, 34)
    b.append(line(110, 74 + 4 * 38 + 4, 520, 74 + 4 * 38 + 4, 'axis'))
    b += [text(540, 92, 'x₀ = (1, 2, 3), w₁ = 0.5, w₂ = 2', 't-muted'),
          text(540, 112, 'A₁: each token reads the one before', 't-muted'),
          text(540, 132, '(token 1 reads itself)', 't-muted'),
          text(540, 152, 'A₂: token 2 reads tokens 1 and 2', 't-muted'),
          text(540, 172, 'half each; others read themselves', 't-muted')]
    return svg(760, 330, 'A table with three columns, one per token. Direct path x0 = (1, 2, 3). Head 1 alone: (0.5, 0.5, 1). Head 2 alone, reading x0: (2, 3, 6). Virtual head, head 2 reading what head 1 wrote: (1, 1, 2). Their sum is (4.5, 6.5, 12), exactly the two-layer output x2.', b)


F['p5_toy_expansion'] = toy_expansion()


# ---------------------------------------------------------------- 3. term importance: paper vs our run
def term_importance():
    L = R['loss']
    ours = [('order 0: direct path', L['order0'] - L['uniform']), ('order 1: 16 head terms', L['order1'] - L['order0']),
            ('order 2: 64 virtual heads', L['order2'] - L['order1'])]
    paper = [('order 0: direct path', -1.8), ('order 1: 24 head terms', -5.2), ('order 2: 144 virtual heads', -0.3)]
    b = [text(24, 26, 'How much each order of terms lowers the loss (nats per token)', 't-title')]
    vmax = 5.2
    b += hbars(24, 70, [(n, -v) for n, v in paper], 300, 30, 's1', 190, lambda v: f'-{v:.1f}', vmax,
               title='the paper (12 heads per layer, its own text)')
    b += hbars(24, 210, [(n, -v) for n, v in ours], 300, 30, 's2', 190, lambda v: f'-{v:.2f}', vmax,
               title='our run (attn-only-2l, 8 heads per layer, 3,072 WikiText tokens)')
    b.append(text(24, 330, f'Our losses: uniform {L["uniform"]:.2f}, direct path {L["order0"]:.2f}, + heads {L["order1"]:.2f}, '
                           f'+ virtual heads {L["order2"]:.2f} (= the model, {L["model"]:.2f}).', 't-muted'))
    return svg(760, 346, f'Two bar charts of the loss drop from each order of terms. The paper: direct path 1.8 nats, single heads 5.2, virtual heads 0.3. Our run on attn-only-2l: direct path {-(L["order0"] - L["uniform"]):.2f}, single heads {-(L["order1"] - L["order0"]):.2f}, virtual heads {-(L["order2"] - L["order1"]):.2f}. In both, the single-head terms do most of the work.', b)


F['p5_term_importance'] = term_importance()


# ---------------------------------------------------------------- 4. composition scores, three heatmaps
def comp_maps():
    base = R['comp_baseline']
    b = [text(24, 26, 'Composition scores minus the random baseline (darker = stronger), attn-only-2l', 't-title')]
    mx = max(max(v - base for r in R['comp'][k] for v in r) for k in 'QKV')
    rows = [f'1.{h}' for h in range(NH)]
    cols = [f'0.{h}' for h in range(NH)]
    for i, k in enumerate('QKV'):
        M = [[max(0.0, (v - base) / mx) for v in r] for r in R['comp'][k]]
        b += heatmap(10 + i * 248, 74, M, rows, cols, 22, title=f'{k}-composition', label_w=44)
    b.append(text(24, 300, 'rows: layer-1 head that reads; columns: layer-0 head that wrote. Hover a cell for its value.', 't-muted'))
    k = R['comp']['K']
    pt = R['prev_head']
    i = max(range(NH), key=lambda h: k[h][pt])
    return svg(760, 316, f'Three 8 by 8 heatmaps: Q-, K- and V-composition between every layer-0 head and every layer-1 head, after subtracting the random baseline {base:.3f}. Almost every cell is pale. The brightest cell is in the K-composition map: head 1.{i} reading the previous-token head 0.{pt}, with score {k[i][pt]:.3f}. The column of head 0.4 is slightly above the baseline in the Q and K maps.', b)


F['p5_comp_maps'] = comp_maps()


# ---------------------------------------------------------------- 5. QK expansion: what happens without the K or the Q terms
def qk_terms():
    b = [text(24, 26, 'Layer-1 heads: how much the attention pattern changes without one kind of term', 't-title'),
         text(24, 46, '0 = the same pattern, 1 = completely different (total variation, mean over positions)', 't-tick')]
    y0, rh = 70, 34
    for h in range(NH):
        d = R['qk'][str(h)]
        y = y0 + h * rh
        b.append(text(70, y + 19, f'head 1.{h}', 't-tick t-strong' if h == 6 else 't-tick', 'end'))
        for j, (key, cls) in enumerate([('no_K', 's2'), ('no_Q', 's1')]):
            v = d[key]
            w = 300 * v
            b.append(f'<g class="mark"><title>head 1.{h}, without {key[3:]} terms: {v:.2f}</title><rect class="{cls}" x="80" y="{y + j * 14 + 2}" width="{w:.1f}" height="12" rx="2"/></g>')
            b.append(text(86 + w, y + j * 14 + 12, f'{v:.2f}', 't-tick'))
    b += [f'<rect class="s2" x="480" y="80" width="12" height="12" rx="2"/>', text(498, 91, 'without the K-composition terms', 't-tick'),
          f'<rect class="s1" x="480" y="102" width="12" height="12" rx="2"/>', text(498, 113, 'without the Q-composition terms', 't-tick'),
          text(480, 150, 'Head 1.6 (the induction head of', 't-muted'), text(480, 168, 'Part 6) depends on its K terms', 't-muted'),
          text(480, 186, 'and hardly on its Q terms.', 't-muted')]
    d6 = R['qk']['6']
    return svg(760, y0 + NH * rh + 10, f'Paired bars for the eight layer-1 heads. For each, how much its attention pattern changes when the K-composition terms are removed, and when the Q-composition terms are removed. Head 1.6 changes by {d6["no_K"]:.2f} without its K terms and only {d6["no_Q"]:.2f} without its Q terms. Heads 1.0 to 1.5 change by 0.44 to 0.81 either way.', b)


F['p5_qk_terms'] = qk_terms()


# ---------------------------------------------------------------- 6. a virtual head from two previous-token patterns (toy B)
def virtual_toy():
    B = T['B']
    n = 4
    toks = ['t1', 't2', 't3', 't4']
    b = []
    b += heatmap(10, 60, B['P'], toks, toks, 40, title='A₁: previous token', show_values=True, mask_upper=True, label_w=40)
    b.append(text(240, 150, '×', 't-title', 'middle'))
    b += heatmap(250, 60, B['P'], toks, toks, 40, title='A₂: previous token', show_values=True, mask_upper=True, label_w=40)
    b.append(text(480, 150, '=', 't-title', 'middle'))
    b += heatmap(490, 60, B['PP'], toks, toks, 40, title='A₂A₁: two tokens back', show_values=True, mask_upper=True, label_w=40)
    return svg(760, 280, 'Three 4 by 4 attention patterns. The first two are previous-token patterns: each token reads the one before it (token 1 reads itself). Their product is a new pattern in which token 3 reads token 1 and token 4 reads token 2: two tokens back. Every row still adds up to 1.', b)


F['p5_virtual_toy'] = virtual_toy()


# ---------------------------------------------------------------- 7. virtual heads: where their effect comes from
def virtual_split():
    L = R['loss']
    vs = R['virt_split']
    items = [('all 64 virtual heads', -(L['order2'] - L['order1'])), ('via the start token', -vs['bos']), ('via real tokens', -vs['tok'])]
    b = [text(24, 26, 'Loss drop from adding the virtual heads to the direct path + head terms (nats)', 't-title')]
    b += hbars(24, 50, items, 380, 32, 's2', 190, lambda v: f'-{v:.2f}', max(v for _, v in items))
    tv = R['top_virtual'][:3]
    b.append(text(24, 170, 'Largest single virtual heads (loss rise when removed): ' +
                  ', '.join(f'{n.replace(" o ", " ∘ ")} +{v:.2f}' for n, v in tv), 't-muted'))
    return svg(760, 186, f'Bars: adding all 64 virtual heads lowers the loss by {-(L["order2"] - L["order1"]):.2f} nats. The part that a layer-0 head copied from the start token gives {-vs["bos"]:.2f}; the part copied from real tokens gives only {-vs["tok"]:.2f}.', b)


F['p5_virtual_split'] = virtual_split()

json.dump(F, open('results/figs_part5.json', 'w'), indent=1)
print('figures:', ', '.join(F))
