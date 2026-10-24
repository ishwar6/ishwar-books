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


def label(x, y, w, h, s, cls='box', tcls='t-tick t-strong', rx=8):
    """A box with one centred line of text."""
    return [box(x, y, w, h, cls, rx), text(x + w / 2, y + h / 2 + 4.5, s, tcls, 'middle')]


def architecture():
    """The 'you are here' map: requests -> scheduler with the KV cache manager -> workers. Nothing highlighted."""
    W, H = 720, 380
    b = [text(20, 26, 'The parts of vLLM (a redraw of the paper\'s Figure 4)', 't-title')]
    # requests
    b.append(text(20, 62, 'Requests', 't-tick t-strong'))
    for i, s in enumerate(['user 1: prompt', 'user 2: prompt', 'user 3: prompt']):
        b += label(20, 76 + i * 50, 120, 34, s, 'box', 't-tick')
    b.append(text(80, 238, '...', 't-tick', 'middle'))
    # scheduler with the KV cache manager
    sx, sy, sw, sh = 170, 48, 250, 236
    b.append(box(sx, sy, sw, sh, 'box', 10))
    b.append(text(sx + 14, sy + 24, 'Scheduler', 't-title'))
    b.append(text(sx + 14, sy + 42, 'picks the requests of the next step', 't-tick'))
    kx, ky, kw, kh = sx + 14, sy + 56, sw - 28, 166
    b.append(box(kx, ky, kw, kh, 'box-ghost', 8))
    b.append(text(kx + 12, ky + 20, 'KV cache manager', 't-tick t-strong'))
    for i, s in enumerate(['Block tables', 'GPU block allocator', 'CPU block allocator']):
        b += label(kx + 12, ky + 32 + i * 44, kw - 24, 34, s, 'box', 't-tick')
    # workers
    wx, ww = 504, 196
    for j, (name, y) in enumerate([('Worker 0', 48), ('Worker N-1', 184)]):
        b.append(box(wx, y, ww, 100, 'box', 10))
        b.append(text(wx + 12, y + 22, name, 't-tick t-strong'))
        b += label(wx + 12, y + 32, ww - 24, 26, 'Cache engine', 'box', 't-tick')
        b += label(wx + 12, y + 64, ww - 24, 26, 'Model shard', 'box', 't-tick')
    b.append(text(wx + ww / 2, 172, '...', 't-tick', 'middle'))
    # arrows
    b.append(arrow(144, 166, sx - 6, 166))
    b.append(arrow(sx + sw + 2, 120, wx - 6, 98))
    b.append(arrow(sx + sw + 2, 212, wx - 6, 234))
    b.append(text(462, 150, 'one step:', 't-tick', 'middle'))
    b.append(text(462, 166, 'batch and', 't-tick', 'middle'))
    b.append(text(462, 182, 'block tables', 't-tick', 'middle'))
    b.append(f'<path class="edge-dim" d="M{wx + ww / 2},{290} V{306} H{80} V{258}" marker-end="url(#ah)"/>')
    b.append(text(300, 322, 'new tokens go back to the users, and to the scheduler for the next step', 't-tick', 'middle'))
    b.append(text(20, 352, 'Part 3: the block tables, the allocators and the attention kernel.', 't-tick'))
    b.append(text(20, 370, 'Part 4: sharing, the scheduler, the CPU swap and many GPUs.', 't-tick'))
    return svg(W, H, 'The architecture of vLLM. On the left, requests from users. In the centre, the scheduler, which '
               'contains the KV cache manager with block tables, a GPU block allocator and a CPU block allocator. On '
               'the right, workers 0 to N-1, each with a cache engine and a shard of the model. New tokens flow back.', b)


def request_life():
    """One chat request over time: queue, prompt phase, one token per step, then the memory is freed."""
    W, H = 720, 330
    b = [text(20, 26, 'The life of one chat request', 't-title')]
    x0, cw = 128, 46
    cols = ['arrives', 'waits', 'prompt', 'step 1', 'step 2', 'step 3', 'step 4', 'step 5', 'done']
    for i, c in enumerate(cols):
        b.append(text(x0 + i * cw + cw / 2, 58, c, 't-tick', 'middle'))
    b.append(text(x0 - 10, 90, 'model output', 't-tick', 'end'))
    outs = ['', '', 'The', 'capital', 'is', 'Paris', '.', '<end>', '']
    for i, o in enumerate(outs):
        if o:
            b += token(x0 + i * cw + 2, 74, cw - 4, o, 'box-2', 26, 't-tick')
    b.append(text(x0 - 10, 122, 'GPU passes', 't-tick', 'end'))
    for i in range(2, 8):
        b += token(x0 + i * cw + 2, 106, cw - 4, '1 pass', 'box', 24, 't-tick')
    # memory: slots of KV cache held, as stacked squares (prompt of 7 tokens, then +1 per step)
    b.append(text(x0 - 10, 170, 'KV cache', 't-tick', 'end'))
    b.append(text(x0 - 10, 186, 'held (slots)', 't-tick', 'end'))
    base, sq = 280, 9
    held = [0, 0, 7, 8, 9, 10, 11, 12, 0]
    for i, n in enumerate(held):
        for k in range(n):
            cls = 'box-1' if k < 7 else 'box-2'
            b.append(box(x0 + i * cw + 12, base - (k + 1) * (sq + 1), cw - 24, sq, cls, 2))
        b.append(text(x0 + i * cw + cw / 2, base + 16, str(n) if n else ('freed' if i == 8 else '0'), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{base + 1}" x2="{x0 + 9 * cw}" y2="{base + 1}"/>')
    lx = x0 + 9 * cw + 18
    b += [f'<rect class="box-1" x="{lx}" y="150" width="12" height="12" rx="2"/>', text(lx + 18, 160, 'prompt tokens', 't-tick'),
          f'<rect class="box-2" x="{lx}" y="172" width="12" height="12" rx="2"/>', text(lx + 18, 182, 'answer tokens', 't-tick')]
    b.append(text(lx, 214, 'The cache grows by', 't-tick'))
    b.append(text(lx, 230, 'one slot per step.', 't-tick'))
    b.append(text(lx, 246, 'Nobody knows the', 't-tick'))
    b.append(text(lx, 262, 'final size in advance.', 't-tick'))
    b.append(text(20, 316, 'Prompt: "What is the capital of France?" (7 tokens in this sketch). The answer streams back one token per step.', 't-tick'))
    return svg(W, H, 'A timeline of one request. It arrives, waits, then the prompt phase reads all 7 prompt tokens in one '
               'pass and writes the first answer token. Each later step writes one more token. The KV cache grows from '
               '7 to 12 slots, one per step, and is freed when the end token appears.', b)


def many_users():
    """Iteration-level batching: four requests join and leave the batch at different steps."""
    W, H = 720, 300
    b = [text(20, 26, 'Many users at once: each column is one pass of the model', 't-title')]
    x0, cw, y0, rh = 110, 64, 64, 36
    steps = 8
    for t in range(steps):
        b.append(text(x0 + t * cw + cw / 2, y0 - 10, f'step {t + 1}', 't-tick', 'middle'))
    # (name, first step, prompt step count 1, number of answer tokens)
    reqs = [('A', 0, 5), ('B', 1, 6), ('C', 3, 2), ('D', 5, 3)]
    count = [0] * steps
    for r, (n, start, out) in enumerate(reqs):
        y = y0 + r * rh
        b.append(text(x0 - 12, y + rh / 2 + 4, f'request {n}', 't-tick t-strong', 'end'))
        for t in range(start, min(start + out, steps)):
            first = t == start
            b += token(x0 + t * cw + 3, y + 4, cw - 6, 'prompt' if first else 'token', 'box-1' if first else 'box-2', rh - 8, 't-tick')
            count[t] += 1
        if start + out <= steps:
            b.append(text(x0 + (start + out) * cw + 8, y + rh / 2 + 4, 'done', 't-muted'))
    yb = y0 + len(reqs) * rh + 22
    b.append(text(x0 - 12, yb + 4, 'batch size', 't-tick t-strong', 'end'))
    for t, c in enumerate(count):
        b.append(text(x0 + t * cw + cw / 2, yb + 4, str(c), 't-val', 'middle'))
    b.append(text(20, yb + 40, 'All requests in one column share one read of the weights. A request joins when it arrives', 't-tick'))
    b.append(text(20, yb + 58, 'and leaves when it writes its end token, so the batch changes at every step.', 't-tick'))
    return svg(W, H, 'A grid of four requests over eight steps. Request A runs in steps 1 to 5, B in steps 2 to 7, C in '
               'steps 4 and 5, D in steps 6 to 8. The first cell of each is its prompt phase. The batch size per step is '
               + ', '.join(str(c) for c in count) + '.', b)


def memory_bound():
    """Why serving is memory-bound: one read of the weights per step, shared by every request in the batch."""
    W, H = 720, 270
    bt = R['batch']
    t1, t64 = bt['1']['step_ms'], bt['64']['step_ms']
    b = [text(20, 26, 'One step reads all the weights, whatever the batch size', 't-title')]
    b += label(20, 60, 150, 120, '', 'box', 't-tick')
    b += [text(95, 100, 'GPU memory', 't-tick t-strong', 'middle'), text(95, 122, 'weights:', 't-tick', 'middle'),
          text(95, 140, 'every number is', 't-tick', 'middle'), text(95, 158, 'read once per step', 't-tick', 'middle')]
    b += label(240, 60, 150, 120, '', 'box', 't-tick')
    b += [text(315, 110, 'arithmetic', 't-tick t-strong', 'middle'), text(315, 130, 'units', 't-tick t-strong', 'middle')]
    b.append(arrow(172, 120, 236, 120))
    b.append(text(204, 110, 'read', 't-tick', 'middle'))
    # outputs: batch 1 and batch 64
    b.append(arrow(392, 90, 440, 80))
    b.append(arrow(392, 150, 440, 160))
    b.append(text(450, 66, f'batch of 1: {t1:.2f} ms', 't-tick t-strong'))
    b += token(450, 74, 20, '', 'box-2', 16)
    b.append(text(478, 87, '1 token per step', 't-tick'))
    b.append(text(450, 132, f'batch of 64: {t64:.2f} ms', 't-tick t-strong'))
    for k in range(64):
        b.append(box(450 + (k % 16) * 13, 140 + (k // 16) * 13, 10, 10, 'box-2', 2))
    b.append(text(450, 208, '64 tokens per step', 't-tick'))
    b.append(text(20, 236, 'The read costs the same for 1 request or 64, so a larger batch spreads it over more tokens.', 't-tick'))
    b.append(text(20, 254, 'Times are our run of OPT-125M on a laptop GPU (one decode step, 128 tokens of context).', 't-tick'))
    return svg(W, H, f'A diagram: the weights in GPU memory are read once per step by the arithmetic units. A batch of 1 '
               f'takes {t1:.2f} ms and writes 1 token; a batch of 64 takes {t64:.2f} ms and writes 64 tokens.', b)


def paging_origin():
    """Operating-system paging next to vLLM's blocks: the same three pieces."""
    W, H = 720, 330
    b = [text(20, 26, 'The same idea twice: pages in an operating system, blocks in vLLM', 't-title')]

    def panel(y, who, sees, unit, short, table, mem, order, cls):
        b.append(text(20, y, who, 't-tick t-strong'))
        for i in range(3):
            b.extend(token(20 + i * 56, y + 12, 50, f'{unit} {i}', cls, 28, 't-tick'))
        b.append(text(20, y + 60, sees, 't-muted'))
        tx = 214
        b.append(text(tx, y, table, 't-tick t-strong'))
        for i, f in enumerate(order):
            b.extend(token(tx, y + 12 + i * 22, 56, f'{unit} {i}', 'box', 20, 't-tick'))
            b.extend(token(tx + 60, y + 12 + i * 22, 40, f'-> {f}', 'box', 20, 't-tick'))
        mx = 400
        b.append(text(mx, y, mem, 't-tick t-strong'))
        for j in range(8):
            hit = j in order
            k = order.index(j) if hit else None
            b.extend(token(mx + j * 38, y + 12, 34, f'{short}{k}' if hit else str(j), cls if hit else 'box-ghost', 28,
                           't-tick' if hit else 't-muted'))
        b.append(text(mx, y + 60, 'the real places, scattered', 't-muted'))

    panel(64, 'Operating system', 'one program (a process)', 'page', 'p', 'page table', 'physical memory (frames 0 to 7)', [5, 1, 6], 'box-1')
    panel(184, 'vLLM', 'one request', 'block', 'b', 'block table', 'GPU KV cache (blocks 0 to 7)', [2, 7, 4], 'box-2')
    b.append(text(20, 300, 'Pages are blocks, bytes are tokens, programs are requests. Only the table knows where each piece lives.', 't-tick'))
    return svg(W, H, 'Two rows. Top: a program sees pages 0, 1 and 2 in order; its page table maps them to frames 5, 1 '
               'and 6 of physical memory. Bottom: a request sees blocks 0, 1 and 2; its block table maps them to GPU '
               'blocks 2, 7 and 4.', b)


def before_after():
    """The same 40 GB card, before vLLM and with vLLM, drawn from our simulation's per-request slot counts."""
    W, H = 720, 330
    f = R['fig2']
    mx, pg = f['max (2048)'], f['paged (16)']
    b = [text(20, 26, 'The same 40 GB GPU, before vLLM and with vLLM', 't-title')]
    x0, bw = 20, 680
    wpx = bw * 0.65
    kx, kw = x0 + wpx + bw * 0.05, bw * 0.30

    def card(y, title, note):
        b.append(text(x0, y - 8, title, 't-tick t-strong'))
        b.append(box(x0, y, wpx - 2, 54, 'box', 4))
        b.append(text(x0 + wpx / 2, y + 32, 'weights: 26 GB (65%), never change', 't-tick', 'middle'))
        b.append(box(x0 + wpx, y, bw * 0.05 - 2, 54, 'box-ghost', 4))
        b.append(text(x0, y + 74, note, 't-tick'))

    # before: 7 chunks of 2048 slots; each holds 3.6% tokens on average
    y = 64
    card(y, 'Before: one chunk of 2,048 slots per request', f"{mx['fit_12gb']:.1f} requests fit in the KV cache space; "
         f"only {mx['tokens']:.1f}% of it holds a token (our Alpaca simulation)")
    n = 7
    cw = kw / n
    for i in range(n):
        b.append(box(kx + i * cw, y, cw - 2, 54, 'waste', 2))
        b.append(box(kx + i * cw, y + 54 - 54 * mx['tokens'] / 100 - 1, cw - 2, 54 * mx['tokens'] / 100 + 1, 'box-2', 1))
    # after: blocks of 16 slots; 90.7% hold a token
    y = 190
    card(y, 'With vLLM: blocks of 16 slots, given on demand', f"{pg['fit_12gb']:.1f} requests fit on average; "
         f"{pg['tokens']:.1f}% of the space holds a token")
    cols, rows = 15, 6
    cw, rh = kw / cols, 54 / rows
    full = round(cols * rows * pg['tokens'] / 100)
    for k in range(cols * rows):
        i, j = k % cols, k // cols
        b.append(box(kx + i * cw, y + j * rh, cw - 1.5, rh - 1.5, 'box-2' if k < full else 'waste', 1))
    b.append(text(kx + kw / 2, 54, 'KV cache: 12 GB', 't-tick', 'middle'))
    b.append(text(x0, 314, 'Orange holds a token. Red dashes are booked but empty. The thin gap is other data. Same model, same card.', 't-tick'))
    return svg(W, H, f"Two bars for one 40 GB GPU. Both give 65 percent to the weights. Before vLLM, the 12 GB of KV cache "
               f"space holds {mx['fit_12gb']:.1f} chunks of 2,048 slots, mostly empty. With vLLM it holds small blocks, "
               f"{pg['tokens']:.1f} percent full, for {pg['fit_12gb']:.1f} requests on average.", b)


def sharing():
    """Three answers to one prompt: three copies of the prompt's cache, or one copy shared through three tables."""
    W, H = 720, 290
    b = [text(20, 26, 'Three suggestions for one prompt', 't-title')]
    b.append(text(20, 58, 'Before: each answer has its own chunk', 't-tick t-strong'))
    for i in range(3):
        y = 70 + i * 40
        b += token(20, y, 150, 'prompt (copy)', 'box-1', 30, 't-tick')
        b += token(176, y, 90, f'answer {i + 1}', 'box-2', 30, 't-tick')
    b.append(text(20, 206, 'the prompt\'s keys and values', 't-tick'))
    b.append(text(20, 222, 'are stored 3 times', 't-tick'))
    px = 380
    b.append(text(px, 58, 'With vLLM: tables point at one shared copy', 't-tick t-strong'))
    b += token(px, 140, 150, 'prompt blocks (1 copy)', 'box-1', 30, 't-tick')
    for i in range(3):
        y = 70 + i * 70
        b += token(px + 220, y, 100, f'answer {i + 1}', 'box-2', 30, 't-tick')
        b.append(f'<line class="edge" x1="{px + 216}" y1="{y + 15}" x2="{px + 154}" y2="155" marker-end="url(#ah)"/>')
    b.append(text(px, 206, 'each answer\'s table lists the', 't-tick'))
    b.append(text(px, 222, 'same prompt blocks, then its own', 't-tick'))
    b.append(text(20, 270, 'Part 4 shows how vLLM copies a shared block only when one answer must change it.', 't-tick'))
    return svg(W, H, 'Left: three answers, each with its own copy of the prompt cache. Right: one copy of the prompt '
               'blocks, and three answers whose tables all point to it.', b)


F['p1_cache'] = kv_cache_grows()
F['p1_batch'] = batching()
F['p1_waste'] = waste()
F['p1_paged'] = contiguous_vs_paged()
F['p1_arch'] = architecture()
F['p1_life'] = request_life()
F['p1_users'] = many_users()
F['p1_bound'] = memory_bound()
F['p1_os'] = paging_origin()
F['p1_before'] = before_after()
F['p1_share'] = sharing()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
