"""Figures for Part 3 (PagedAttention, the block table, the KV cache manager). Writes results/figs_part3.json.
CSS classes only, so the figures follow the dark, dim and light themes. Numbers come from results/part3.json."""
import json, os
from figlib import svg, text, box, arrow, esc
from samfig import token, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part3.json')))
F = {}
WORDS_A = ['Four', 'score', 'and', 'seven', 'years', 'ago', 'our', 'fathers', 'brought', 'forth']


def eq4():
    """Figure 5 with real numbers: the share of attention of the query 'forth' on each slot of each block."""
    W, H, cw, ch = 720, 300, 76, 34
    b = [text(20, 24, 'Eq. 4 on OPT-125M (layer 6, mean of 12 heads): where "forth" looks', 't-title')]
    b += token(20, 128, 90, 'forth', 'box-on', 34)
    b.append(text(65, 182, 'query', 't-tick', 'middle'))
    for r, blk in enumerate(R['eq4']):
        y = 56 + r * 76
        b.append(text(176, y + ch / 2 - 2, f'block {blk["block"]}', 't-tick t-strong'))
        b.append(text(176, y + ch / 2 + 14, f'physical {blk["phys"]}', 't-tick'))
        b.append(arrow(114, 145, 168, y + ch / 2))
        for k in range(4):
            x = 262 + k * (cw + 4)
            if k < len(blk['words']):
                wgt = blk['weights'][k]
                b.append(f'<g class="mark"><title>{esc(blk["words"][k])}: {wgt:.3f}</title>'
                         f'<rect class="cell" x="{x}" y="{y}" width="{cw}" height="{ch}" rx="4" '
                         f'style="fill-opacity:{max(0.06, min(1, wgt / 0.75)):.3f}"/></g>')
                b.append(text(x + cw / 2, y + 14, blk['words'][k], 't-cell' + (' on' if wgt > 0.4 else ''), 'middle'))
                b.append(text(x + cw / 2, y + 28, f'{wgt:.3f}', 't-cell' + (' on' if wgt > 0.4 else ''), 'middle'))
            else:
                b.append(box(x, y, cw, ch, 'box-ghost', 4))
        b.append(text(262 + 4 * (cw + 4) + 8, y + ch / 2 + 4, f'total {sum(blk["weights"]):.3f}', 't-mono'))
    b.append(text(20, 286, 'One softmax over all ten keys: the three block totals add up to 1.', 't-tick'))
    return svg(W, H, 'The query forth points to three KV blocks stored in physical blocks 7, 1 and 3. Each slot is '
               f'shaded by its attention weight. Four gets {R["eq4"][0]["weights"][0]:.3f}; the block totals are '
               + ', '.join(f'{sum(k["weights"]):.3f}' for k in R['eq4']) + '.', b)


def tradeoff():
    """Block size: share of held memory that holds tokens (Alpaca lengths) and time of our block-by-block loop."""
    Wd = R['waste']
    Bs = ['1', '4', '16', '64', '128']
    W, H = 720, 310
    b = []
    b += hbars(20, 50, [(f'B = {B}', Wd[B]['use']) for B in Bs] + [('2,048 chunk', R['contig']['use'])], 200, 26, 's1', 100,
               fmt=lambda v: f'{v:.1f}%', vmax=100, title='memory holding tokens (Alpaca)', hi='B = 16')
    Sp = R['speed']
    b += hbars(380, 50, [(f'B = {B}', Sp[B]) for B in ('1', '4', '16', '64', '256')], 190, 26, 's2', 70,
               fmt=lambda v: f'{v:.1f} ms', vmax=max(Sp.values()), title='time of our block loop (2,048 keys)', hi='B = 16')
    b.append(text(20, 250, 'Small blocks waste little memory but cost many separate fetches.', 't-tick'))
    b.append(text(20, 270, 'Large blocks fetch fast but leave up to B - 1 empty slots per request.', 't-tick'))
    b.append(text(20, 296, f'The 2,048-slot chunks: {R["lens"]["total"]:,} tokens in {R["contig"]["held"]:,} slots.', 't-tick'))
    return svg(W, H, f'Left: the share of held memory that holds tokens falls from {Wd["1"]["use"]:.1f}% at block size 1 '
               f'to {Wd["128"]["use"]:.1f}% at 128, against {R["contig"]["use"]:.1f}% for chunks of 2,048. Right: our block loop takes '
               f'{Sp["1"]:.0f} ms at block size 1 and {Sp["256"]:.1f} ms at 256.', b)




def vmap():
    """The vLLM architecture (paper Figure 4) as a 'you are here' map. The parts this page explains use box-on."""
    W, H = 720, 384
    b = [text(20, 24, 'vLLM, the whole engine: what this part explains is highlighted', 't-title')]
    # left: requests
    b.append(text(70, 62, 'Requests', 't-label', 'middle'))
    for i, r in enumerate(['request A', 'request B', 'request C']):
        b += token(20, 78 + i * 44, 100, r, 'box', 32)
    b.append(arrow(124, 160, 168, 160))
    # centre: the scheduler with the KV cache manager inside
    b.append(box(172, 44, 300, 290, 'plane', 12))
    b.append(text(322, 70, 'Scheduler', 't-node'))
    b.append(text(322, 92, 'picks the requests of each step', 't-tick', 'middle'))
    b.append(box(188, 108, 268, 212, 'box-on', 10))
    b.append(text(322, 132, 'KV cache manager', 't-node'))
    b += token(204, 150, 236, 'Block tables: one per request', 'box-on', 34, 't-tick t-strong')
    b += token(204, 196, 236, 'GPU block allocator', 'box-on', 34, 't-tick t-strong')
    b += token(204, 242, 236, 'CPU block allocator', 'box', 34)
    b.append(text(322, 302, 'the CPU side: Part 4', 't-muted', 'middle'))
    b.append(arrow(476, 160, 516, 160))
    b.append(text(496, 148, 'tables', 't-muted', 'middle'))
    # right: the workers
    for i, w in enumerate(['Worker 0', 'Worker 1', 'Worker N-1']):
        y = 44 + i * 100
        b.append(box(520, y, 180, 88, 'box', 10))
        b.append(text(610, y + 20, w, 't-tick t-strong', 'middle'))
        b += token(530, y + 30, 160, 'Cache engine', 'box-on', 24, 't-tick t-strong')
        b += token(530, y + 58, 160, 'Model shard', 'box', 24)
    b.append(text(610, 356, 'PagedAttention runs inside each shard', 't-tick', 'middle'))
    b.append(text(610, 374, 'and reads the blocks of its cache engine', 't-tick', 'middle'))
    return svg(W, H, 'The vLLM architecture. Requests on the left go to the scheduler in the centre. Inside the '
               'scheduler, the KV cache manager holds the block tables, a GPU block allocator and a CPU block '
               'allocator. On the right, workers 0 to N minus 1 each hold a cache engine and a model shard. The block '
               'tables, the GPU block allocator and the cache engines are highlighted.', b)


def before_after():
    """A tiny made-up example: 24 slots of memory, four requests, chunks of 8 against blocks of 4."""
    W, H, cw, gap = 720, 300, 24, 2
    cls = {'A': 'box-1', 'B': 'box-2', 'C': 'box-3', 'D': 'box-4'}
    b = [text(20, 24, 'Before vLLM: one chunk of 8 slots per request, booked at the start', 't-title')]
    x0 = 20
    chunks = [('A', 5), ('B', 2), ('C', 7)]
    for c, (r, n) in enumerate(chunks):
        for k in range(8):
            x = x0 + (c * 8 + k) * (cw + gap) + c * 8
            if k < n:
                b += token(x, 40, cw, r, cls[r], 30)
            else:
                b.append(box(x, 40, cw, 30, 'waste', 4))
        b.append(text(x0 + c * (8 * (cw + gap) + 8) + 4 * (cw + gap), 88, f'{r}: {n} tokens, {8 - n} empty', 't-tick', 'middle'))
    b += token(672, 40, 34, 'D', 'box-4', 30)
    b.append(text(689, 88, 'D waits', 't-tick t-strong', 'middle'))
    b.append(text(20, 114, '24 slots, 14 tokens: 10 slots sit empty and request D (3 tokens) cannot start.', 't-tick'))
    b.append(text(20, 160, 'With vLLM: blocks of 4 slots, one more block only when the last one is full', 't-title'))
    blocks = [('A', 4), ('C', 4), ('B', 2), ('A', 1), ('C', 3), ('D', 3)]
    for j, (r, n) in enumerate(blocks):
        for k in range(4):
            x = x0 + (j * 4 + k) * (cw + gap) + j * 8
            if k < n:
                b += token(x, 176, cw, r, cls[r], 30)
            else:
                b.append(box(x, 176, cw, 30, 'box-ghost', 4))
        b.append(text(x0 + j * (4 * (cw + gap) + 8) + 2 * (cw + gap), 224, f'block {j}', 't-tick', 'middle'))
    b.append(text(20, 252, '24 slots, 17 tokens: all four requests run. Their blocks sit in any order.', 't-tick'))
    b.append(text(20, 272, 'Each request leaves at most 3 empty slots, the end of its last block.', 't-tick'))
    return svg(W, H, 'Two rows of 24 memory slots. Before vLLM, requests A, B and C each book a chunk of 8 slots; '
               '10 slots stay empty and request D waits. With vLLM, the same memory holds six blocks of 4: A, C, B, A, C '
               'and D. All four requests run, with 7 empty slots in all.', b)


def kernel():
    """Why ordinary attention wants one contiguous tensor, and what PagedAttention changes."""
    W, H, cw = 720, 300, 56
    words = WORDS_A
    b = [text(20, 24, 'Ordinary attention kernel: all keys of a request in one row of memory', 't-title')]
    for t, w in enumerate(words):
        b += token(20 + t * (cw + 4), 40, cw, w, 'box-1', 28)
    b.append(text(20, 88, 'key t sits at  start + t × (size of one key).  One multiplication finds any key.', 't-tick'))
    b.append(text(20, 106, 'The row must be booked for the longest answer, because it cannot grow into memory that is in use.', 't-tick'))
    b.append(text(20, 148, 'PagedAttention kernel: keys in blocks of 4, anywhere in a pool of 9 blocks', 't-title'))
    tab = [p for p, _ in R['fig6']['steps'][-1]['table']]
    fill = {p: f for p, f in R['fig6']['steps'][-1]['table']}
    bw = 4 * 15 + 4
    for k in range(9):
        x = 20 + k * (bw + 12)
        for s in range(4):
            on = k in fill and s < fill[k]
            b.append(box(x + s * 15, 164, 13, 28, 'box-1' if on else 'box-ghost', 3))
        b.append(text(x + bw / 2 - 2, 210, f'block {k}', 't-tick' + (' t-strong' if k in fill else ''), 'middle'))
    b.append(text(20, 240, f'block table of the request: {tab}', 't-mono'))
    b.append(text(20, 264, 'key t sits in physical block  table[t // 4],  slot  t % 4.', 't-tick'))
    b.append(text(20, 282, 'One look into the table, then the same multiplication. The kernel needs the table as an input.', 't-tick'))
    return svg(W, H, 'Top: the ten keys of a request in one contiguous row, so key t is at the start plus t times the '
               f'size of a key. Bottom: the same keys in blocks {tab[0]}, {tab[1]} and {tab[2]} of a pool of nine; the kernel '
               'first reads the block table to find the block, then the slot inside it.', b)


def lookup():
    """The page-table lookup of an operating system next to the block-table lookup of vLLM, from our run."""
    W, H = 720, 380
    tok9 = R['lookup']['tokens'][9]                              # [t, word, logical, offset, phys, slot]
    b = [text(180, 24, 'Operating system: a byte', 't-title', 'middle'),
         text(540, 24, 'vLLM: a token', 't-title', 'middle')]
    cols = [
        dict(x=20, top='address 10,000', split=('page 2', 'offset 1,808'), rule='10,000 = 2 × 4,096 + 1,808',
             table=[('page 0', 'frame 5'), ('page 1', 'frame 2'), ('page 2', 'frame 7'), ('page 3', 'frame 0')], hit=2,
             tname='page table', out='frame 7', final='7 × 4,096 + 1,808 = 30,480'),
        dict(x=380, top=f'token {tok9[0]}, "{tok9[1]}"', split=(f'logical block {tok9[2]}', f'offset {tok9[3]}'),
             rule=f'{tok9[0]} = {tok9[2]} × 4 + {tok9[3]}',
             table=[(f'logical {j}', f'physical {p}') for j, (p, _) in enumerate(R['fig6']['steps'][-1]['table'])],
             hit=tok9[2], tname='block table', out=f'physical block {tok9[4]}',
             final=f'{tok9[4]} × 4 + {tok9[3]} = slot {tok9[5]} of the pool'),
    ]
    for c in cols:
        x = c['x']
        b += token(x + 60, 40, 200, c['top'], 'box', 30, 't-tick t-strong')
        b.append(arrow(x + 120, 72, x + 90, 94))
        b.append(arrow(x + 200, 72, x + 230, 94))
        b += token(x, 98, 150, c['split'][0], 'box-on', 30, 't-tick t-strong')
        b += token(x + 170, 98, 150, c['split'][1], 'box-2', 30, 't-tick t-strong')
        b.append(text(x + 160, 146, c['rule'], 't-tick', 'middle'))
        b.append(text(x + 150, 172, c['tname'], 't-label', 'end'))
        for i, (k, v) in enumerate(c['table']):
            y = 180 + i * 28
            hit = i == c['hit']
            b += token(x, y, 72, k, 'box-on' if hit else 'box', 24)
            b += token(x + 78, y, 72, v.replace('physical ', 'phys. '), 'box-on' if hit else 'box', 24)
        yh = 180 + c['hit'] * 28 + 12
        b.append(arrow(x + 30, 130, x + 30, 176))
        b.append(arrow(x + 154, yh, x + 186, yh))
        b += token(x + 190, yh - 15, 130, c['out'], 'box-on', 30, 't-tick t-strong')
        b.append(arrow(x + 245, 130, x + 245, yh - 18))
        b.append(text(x + 160, 318, c['final'], 't-mono', 'middle'))
    b.append(text(360, 352, 'The same three moves: split the number, look the first part up, add the second part back.',
                  't-tick', 'middle'))
    return svg(W, H, 'Left: the address 10,000 splits into page 2 and offset 1,808; the page table sends page 2 to frame 7; '
               f'the physical address is 30,480. Right: token {tok9[0]}, forth, splits into logical block {tok9[2]} and offset '
               f'{tok9[3]}; the block table sends logical block {tok9[2]} to physical block {tok9[4]}; the token sits in slot '
               f'{tok9[5]} of the pool.', b)


def softmax_split():
    """Eq. 4 per block, then the sum: the Background example in blocks of 2, numbers from our run."""
    T = R['tiny']
    sc, ex, part, den = T['scores'], T['exps'], T['block_sums'], T['den']
    W, H = 720, 340
    b = [text(20, 24, 'Eq. 4 on the Background example: query (1, 0), three tokens, blocks of 2', 't-title'),
         text(150, 52, '1. each block alone', 't-label', 'middle'),
         text(400, 52, '2. one addition', 't-label', 'middle'),
         text(600, 52, '3. each block again', 't-label', 'middle')]
    rows = [(0, [0, 1]), (1, [2])]
    for r, toks in rows:
        y = 70 + r * 110
        b.append(text(20, y + 18, f'block {r + 1}', 't-tick t-strong'))
        for i, t in enumerate(toks):
            yy = y + i * 34
            b += token(80, yy, 56, f'token {t + 1}', 'box-1', 28)
            b.append(text(146, yy + 18, f'score {sc[t]:.3f}', 't-tick'))
            b.append(text(232, yy + 18, f'exp {ex[t]:.3f}', 't-tick'))
        b += token(80, y + 34 * len(toks) + 4, 220, f'block sum {part[r]:.3f}', 'box-2', 26, 't-tick t-strong')
        b.append(arrow(304, y + 34 * len(toks) + 17, 340, 160))
        wts = T['blocks'][r]['weights']
        ctb = T['blocks'][r]['contrib']
        b.append(text(510, y + 18, 'weights ' + ', '.join(f'{w:.3f}' for w in wts), 't-tick'))
        b += token(510, y + 30, 190, f'adds ({ctb[0]:.3f}, {ctb[1]:.3f})', 'box-1', 26, 't-tick t-strong')
    b += token(344, 140, 120, f'{part[0]:.3f} + {part[1]:.3f}', 'box-on', 30, 't-tick t-strong')
    b.append(text(404, 190, f'= {den:.3f}', 't-mono', 'middle'))
    b.append(text(404, 208, 'the shared', 't-tick', 'middle'))
    b.append(text(404, 224, 'denominator', 't-tick', 'middle'))
    b.append(arrow(468, 150, 504, 88))
    b.append(arrow(468, 160, 504, 196))
    o = T['out']
    b += token(470, 280, 230, f'output ({o[0]:.3f}, {o[1]:.3f})', 'box-on', 32, 't-tick t-strong')
    b.append(text(20, 300, 'Weight = exp / shared denominator. The two block', 't-tick'))
    b.append(text(20, 318, 'outputs add up to the answer of ordinary attention.', 't-tick'))
    return svg(W, H, f'Block 1 holds tokens 1 and 2, with exponentials {ex[0]:.3f} and {ex[1]:.3f}, sum {part[0]:.3f}. '
               f'Block 2 holds token 3, exponential {ex[2]:.3f}. The block sums add up to the shared denominator {den:.3f}. '
               f'Each block then turns its exponentials into weights and adds its values; the output is ({o[0]:.3f}, {o[1]:.3f}).', b)


def mem_grid(x, y, cells, label=True):
    """Nine physical blocks of 4 slots in a 3 x 3 grid. cells: {block: [class or None] * 4}."""
    b = []
    for k in range(9):
        bx, by = x + (k % 3) * 92, y + (k // 3) * 26
        if label:
            b.append(text(bx + 10, by + 15, str(k), 't-tick' + (' t-strong' if k in cells else ''), 'end'))
        for s in range(4):
            c = cells.get(k, [None] * 4)[s]
            b.append(box(bx + 16 + s * 17, by + 2, 15, 18, c or 'box-ghost', 3))
    return b


def fig6_steps():
    """Figure 6 from our run: block table and physical blocks after the prefill and each decoding step."""
    steps = R['fig6']['steps']
    W, rowh = 720, 108
    H = 74 + len(steps) * rowh
    b = [text(20, 24, 'Request A, block size 4: the block table and the 9 physical blocks after each step', 't-title'),
         text(20, 50, 'step', 't-label'), text(196, 50, 'block table', 't-label'),
         text(440, 50, 'physical blocks 0 to 8', 't-label')]
    for s, st in enumerate(steps):
        y0 = 62 + s * rowh
        n = sum(f for _, f in st['table'])                       # tokens so far
        label = 'prefill' if st['step'] == 'prefill' else f'+ "{st["word"]}"'
        b.append(text(20, y0 + 20, label, 't-tick t-strong'))
        b.append(text(20, y0 + 38, f'{n} tokens', 't-tick'))
        b.append(text(20, y0 + 56, f'{len(st["table"])} blocks', 't-tick'))
        prev = steps[s - 1]['table'] if s else []
        for j, (p, f) in enumerate(st['table']):
            changed = s > 0 and (j >= len(prev) or prev[j][1] != f)
            b += token(196, y0 + 4 + j * 26, 220, f'logical {j} -> physical {p}  ({f} filled)',
                       'box-on' if changed else 'box', 22)
        cells = {}
        for j, (p, f) in enumerate(st['table']):
            cells[p] = ['box-1' if k < f else None for k in range(4)]
            if s > 0 and (j >= len(prev) or prev[j][1] != f):
                cells[p][f - 1] = 'box-4'
        b += mem_grid(440, y0 + 2, cells)
        if s < len(steps) - 1:
            b.append(f'<line class="grid" x1="20" y1="{y0 + rowh - 14}" x2="700" y2="{y0 + rowh - 14}"/>')
    b += [box(440, H - 22, 12, 12, 'box-1', 3), text(458, H - 12, 'filled slot', 't-tick'),
          box(530, H - 22, 12, 12, 'box-4', 3), text(548, H - 12, 'slot written in this step', 't-tick')]
    return svg(W, H, 'Four snapshots of request A. After the prefill of 7 tokens, logical blocks 0 and 1 map to physical '
               'blocks 7 and 1, with 4 and 3 slots filled. Fathers fills the last slot of physical block 1. Brought needs '
               'logical block 2, which maps to a new physical block 3. Forth goes into the second slot of block 3.', b)


def fig7_steps():
    """Figure 7 from our run: two requests in one pool, iteration by iteration; then A ends and C arrives."""
    it = R['fig7']['iters']
    cls = {'A': 'box-1', 'B': 'box-2', 'C': 'box-3'}
    free_now = None
    for line in open(os.path.join(HERE, 'results', 'part3_stdout.txt')):
        if line.startswith('A ends; its blocks go back. free list now:'):
            free_now = line.split(':', 1)[1].strip()
    snaps = []
    for i, st in enumerate(it):
        label = 'iteration 1' if i == 0 else f'iteration {i + 1}'
        sub = ['both prompts'] if i == 0 else [f'A + "{st["wA"]}",', f'B + "{st["wB"]}"']
        snaps.append((label, sub, {'A': st['A'], 'B': st['B']}, it[i - 1] if i else None))
    last = it[-1]
    snaps.append(('A ends', ['free list now', free_now], {'B': last['B']}, None))
    snaps.append(('C arrives', [f'{sum(f for _, f in R["fig7"]["C"])} tokens'], {'B': last['B'], 'C': R['fig7']['C']}, None))
    W, rowh = 720, 96
    H = 74 + len(snaps) * rowh
    b = [text(20, 24, 'Requests A and B share one pool of 9 physical blocks (block size 4)', 't-title'),
         text(20, 50, 'step', 't-label'), text(196, 50, 'block tables (physical blocks)', 't-label'),
         text(440, 50, 'physical blocks 0 to 8', 't-label')]
    for s, (label, sub, tables, prev) in enumerate(snaps):
        y0 = 62 + s * rowh
        b.append(text(20, y0 + 20, label, 't-tick t-strong'))
        for li, ln in enumerate(sub):
            b.append(text(20, y0 + 38 + li * 16, ln, 't-tick'))
        cells = {}
        for r_i, (r, tab) in enumerate(tables.items()):
            ptab = prev[r] if prev else None
            b += token(196, y0 + 4 + r_i * 28, 220, f'{r}: ' + ', '.join(f'{p} ({f})' for p, f in tab), cls[r], 24,
                       't-tick t-strong')
            for j, (p, f) in enumerate(tab):
                cells[p] = [cls[r] if k < f else None for k in range(4)]
                if ptab is not None and (j >= len(ptab) or ptab[j][1] != f):
                    cells[p][f - 1] = 'box-4'
        b += mem_grid(440, y0 + 2, cells)
        if s < len(snaps) - 1:
            b.append(f'<line class="grid" x1="20" y1="{y0 + rowh - 14}" x2="700" y2="{y0 + rowh - 14}"/>')
    b += [box(196, H - 22, 12, 12, 'box-1', 3), text(214, H - 12, 'A', 't-tick'),
          box(240, H - 22, 12, 12, 'box-2', 3), text(258, H - 12, 'B', 't-tick'),
          box(284, H - 22, 12, 12, 'box-3', 3), text(302, H - 12, 'C', 't-tick'),
          box(328, H - 22, 12, 12, 'box-4', 3), text(346, H - 12, 'slot written in this step', 't-tick')]
    return svg(W, H, 'Five snapshots of one pool of nine physical blocks. After iteration 1, request A holds blocks 7 and 1 '
               'and request B holds block 5. Iteration 2 gives B block 2; iteration 3 gives A block 3. When A ends, '
               'blocks 7, 1 and 3 go back to the free list. Request C then takes blocks 0, 4 and 6.', b)


F['p3_map'] = vmap()
F['p3_before_after'] = before_after()
F['p3_kernel'] = kernel()
F['p3_lookup'] = lookup()
F['p3_softmax'] = softmax_split()
F['p3_eq4'] = eq4()
F['p3_fig6_steps'] = fig6_steps()
F['p3_fig7_steps'] = fig7_steps()
F['p3_tradeoff'] = tradeoff()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
