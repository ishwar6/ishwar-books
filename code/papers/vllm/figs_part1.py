"""Figures for Part 1 (the big idea). Writes results/figs_part1.json (key -> svg string). CSS classes only.
Numbers come from results/part1.json (our run) and from the paper's text (§1, §3, Figure 2)."""
import json, os
from figlib import svg, text, box, arrow, lines_chart
from samfig import token

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part1.json')))
F = {}


def kv_cache_grows():
    """The prompt phase writes the keys and values of all prompt tokens at once; each later step adds one column."""
    words = ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our']
    W, H, x0, w, gap = 720, 250, 110, 74, 6
    b = [text(20, 26, 'The KV cache of one request, as it grows', 't-title')]
    for i, s in enumerate(words):
        x = x0 + i * (w + gap)
        cls = 'box-1' if i < 4 else 'box-2'
        b += token(x, 44, w, s, 'box', 28)
        b += token(x, 84, w, f'k{i + 1}', cls, 28, 't-mono')
        b += token(x, 118, w, f'v{i + 1}', cls, 28, 't-mono')
    b += [text(x0 - 10, 63, 'token', 't-tick', 'end'), text(x0 - 10, 103, 'key', 't-tick', 'end'),
          text(x0 - 10, 137, 'value', 't-tick', 'end')]
    xp = x0 + 4 * (w + gap)
    b += [f'<line class="edge-dim" x1="{xp - gap / 2:.1f}" y1="40" x2="{xp - gap / 2:.1f}" y2="160"/>',
          text(x0, 176, 'prompt phase: 4 tokens, one pass', 't-s1'),
          text(xp, 176, 'generation phase: one new column per step', 't-s2')]
    b.append(text(20, 210, 'Each column holds one key and one value vector for every layer. A new token reads all earlier', 't-tick'))
    b.append(text(20, 230, 'columns and adds its own. The cache grows by one column per step, until the request ends.', 't-tick'))
    return svg(W, H, 'Seven tokens of a request in a row. Under each token sit its key and value. The first four, '
               'the prompt, are written in one pass. The next three are added one per step during generation.', b)


def batching():
    bs = sorted(int(k) for k in R['batch'])
    tps = [R['batch'][str(k)]['tokens_per_s'] for k in bs]
    return lines_chart(720, 300, 'Tokens per second of one decode step against the batch size, from '
                       + ', '.join(f'{v:,.0f}' for v in tps) + ' tokens per second for batches of '
                       + ', '.join(str(k) for k in bs) + '.',
                       bs, [('tokens/s', tps, 'l1', 's1')], 'batch size (requests decoded together, log scale)',
                       'tokens per second', 0, 7000, [0, 2000, 4000, 6000], xlog=True,
                       fmt=lambda v: f'{v:,.0f}')


def waste():
    """Stacked bars: the share of reserved KV memory that holds tokens, waits for later tokens, or is never used."""
    rows = list(R['fig2'].items())
    x0, lw, bw, rh, y0 = 20, 120, 520, 40, 70
    b = [text(x0, 26, 'Where the KV cache memory goes (our simulation, 2,000 Alpaca requests)', 't-title')]
    leg = [('s1', 'holds a token now'), ('s2', 'reserved for a later token'), ('waste', 'never used')]
    lx = x0
    for cls, lab in leg:
        b += [f'<rect class="{cls}" x="{lx}" y="40" width="12" height="12" rx="2"/>', text(lx + 18, 50, lab, 't-tick')]
        lx += 24 + 8 * len(lab)
    for i, (name, f) in enumerate(rows):
        y, x = y0 + i * rh, x0 + lw
        b.append(text(x - 8, y + rh / 2 + 2, name, 't-tick t-strong', 'end'))
        for cls, key in [('s1', 'tokens'), ('s2', 'reserved'), ('waste', 'waste')]:
            ww = bw * f[key] / 100
            if ww > 0.5:
                b.append(f'<g class="mark"><title>{name}, {key}: {f[key]:.1f}%</title>'
                         f'<rect class="{cls}" x="{x:.1f}" y="{y + 6}" width="{ww:.1f}" height="{rh - 14}"/></g>')
            x += ww
        b.append(text(x0 + lw + bw + 8, y + rh / 2 + 2, f"{f['tokens']:.1f}%", 't-val'))
    return svg(720, y0 + rh * len(rows) + 10, 'Four stacked bars. The share of memory that holds a token: '
               + ', '.join(f"{n} {f['tokens']:.1f} percent" for n, f in rows) + '.', b)


def contiguous_vs_paged():
    """Left: one contiguous chunk per request, sized for the maximum. Right: small blocks anywhere, a table per request."""
    W, H = 720, 300
    b = [text(20, 26, 'Contiguous chunks against blocks', 't-title')]
    # left panel: a strip of 24 slots; request A reserves 14, request B reserves 8, 2 slots too small to use
    sx, sy, sw = 24, 70, 13
    b.append(text(sx, 58, 'Existing systems: one chunk per request, sized for its maximum', 't-tick t-strong'))
    layout = ['A'] * 3 + ['a'] * 2 + ['x'] * 9 + ['g'] * 2 + ['B'] * 3 + ['b'] * 1 + ['y'] * 4
    cls = {'A': 'box-1', 'B': 'box-3', 'a': 'box-2', 'b': 'box-2', 'x': 'waste', 'y': 'waste', 'g': 'box-ghost'}
    for i, c in enumerate(layout):
        b.append(box(sx + i * sw, sy, sw - 2, 30, cls[c], 2))
    b += [text(sx, sy + 50, 'A: 3 tokens, 2 reserved, 9 never used', 't-tick'),
          text(sx, sy + 68, 'B: 3 tokens, 1 reserved, 4 never used', 't-tick'),
          text(sx, sy + 86, 'gap of 2 slots: too small for a new chunk', 't-tick')]
    # right panel: blocks of 2 slots (drawn as one box each); requests A and B take blocks wherever they are free
    px, py = 400, 70
    b.append(text(px, 58, 'PagedAttention: blocks of fixed size, anywhere', 't-tick t-strong'))
    phys = ['A0', 'B0', '', 'A1', '', 'B1', '', '']
    for j, p in enumerate(phys):
        x = px + j * 38
        c = 'box-1' if p.startswith('A') else 'box-3' if p.startswith('B') else 'box-ghost'
        b.append(box(x, py, 34, 30, c, 3))
        b.append(text(x + 17, py + 20, p or str(j), 't-mono' if p else 't-muted', 'middle'))
    b += [text(px, py + 50, 'A: blocks 0 and 3 (3 tokens, 1 free slot)', 't-tick'),
          text(px, py + 68, 'B: blocks 1 and 5 (3 tokens, 1 free slot)', 't-tick'),
          text(px, py + 86, 'a table per request maps its blocks', 't-tick')]
    b.append(text(20, 220, 'Left: memory is reserved for the longest possible answer, and gaps between chunks stay empty.', 't-tick'))
    b.append(text(20, 240, 'Right: a request takes one small block at a time, wherever one is free, so at most the last', 't-tick'))
    b.append(text(20, 260, 'block of each request has empty slots. Part 3 builds this block table step by step.', 't-tick'))
    return svg(W, H, 'Two panels. Left: requests A and B each own one contiguous chunk with reserved and never-used slots, '
               'and a small gap between chunks. Right: memory is cut into small blocks, and the blocks of A and B sit '
               'at any free place, mapped by a table per request.', b)


F['p1_cache'] = kv_cache_grows()
F['p1_batch'] = batching()
F['p1_waste'] = waste()
F['p1_paged'] = contiguous_vs_paged()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
