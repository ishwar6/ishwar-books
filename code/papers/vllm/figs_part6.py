"""Figures for Part 6 (build it from scratch). Writes results/figs_part6.json. CSS classes only.
Every number and token comes from results/part6.json (our run)."""
import json, os
from figlib import svg, text, box, arrow
from samfig import token, hbars, line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part6.json')))
F = {}


def arch():
    """The paper's Figure 4, redrawn as a map, with the minivllm file that builds each box."""
    W, H = 720, 430
    b = [text(20, 24, 'Figure 4 of the paper, with the file of minivllm that builds each box', 't-title'),
         box(10, 38, 700, 336, 'box-ghost', 12),
         text(24, 58, 'engine.py: the loop that drives every step (LLMEngine)', 't-label')]
    # Requests
    b += [box(24, 170, 130, 96, 'box-on', 10), text(89, 196, 'Requests', 't-note', 'middle'),
          text(89, 218, 'sequence.py', 't-mono', 'middle'), text(89, 238, 'add_request()', 't-muted', 'middle'),
          text(89, 254, 'prompt + settings', 't-muted', 'middle')]
    # Scheduler with the KV cache manager inside
    b += [box(184, 72, 290, 290, 'box-1', 12), text(198, 96, 'Scheduler', 't-note'),
          text(460, 96, 'scheduler.py', 't-mono', 'end'),
          box(198, 110, 262, 238, 'box', 10), text(212, 132, 'KV cache manager', 't-note'),
          text(212, 150, 'block_manager.py', 't-mono')]
    for i, (name, cls_name, cls) in enumerate((('Block tables', 'BlockTable', 'box-on'),
                                               ('GPU block allocator', 'BlockAllocator', 'box-on'),
                                               ('CPU block allocator', 'not built (no swapping)', 'box-ghost'))):
        y = 162 + i * 60
        b += [box(212, y, 234, 50, cls, 8), text(224, y + 21, name, 't-tick t-strong'),
              text(224, y + 39, cls_name, 't-muted')]
    # Workers
    b += [box(508, 72, 190, 190, 'box-1', 12), text(522, 96, 'Worker 0', 't-note'),
          box(522, 108, 162, 62, 'box-on', 8), text(534, 130, 'Cache engine', 't-tick t-strong'),
          text(534, 150, 'kv_cache.py', 't-mono'),
          box(522, 180, 162, 70, 'box-on', 8), text(534, 202, 'Model shard', 't-tick t-strong'),
          text(534, 222, 'model.py', 't-mono'), text(534, 240, 'attention.py', 't-mono'),
          box(508, 276, 190, 86, 'box-ghost', 12), text(522, 300, 'Workers 1 to N-1', 't-note'),
          text(522, 320, 'not built: one process,', 't-muted'), text(522, 338, 'one device, no all-reduce', 't-muted')]
    b += [arrow(156, 218, 182, 218), arrow(476, 160, 506, 160), arrow(506, 228, 476, 228)]
    b += [text(491, 150, 'slots', 't-muted', 'middle'), text(491, 246, 'logits', 't-muted', 'middle')]
    # legend
    b += [box(20, 392, 22, 16, 'box-on', 4), text(50, 405, 'built in minivllm', 't-tick'),
          box(190, 392, 22, 16, 'box-ghost', 4), text(220, 405, 'in the paper, not built here', 't-tick')]
    return svg(W, H, 'The vLLM architecture of Figure 4 as a map. Requests enter on the left (sequence.py). The '
               'scheduler (scheduler.py) holds the KV cache manager (block_manager.py) with block tables and the GPU '
               'block allocator; the CPU block allocator is not built. Worker 0 holds the cache engine (kv_cache.py) '
               'and the model (model.py, attention.py); workers 1 to N-1 are not built. engine.py runs the loop.', b)


def flow():
    """Who calls whom in one step, in time order: a sequence diagram of the seven files."""
    cols = ['engine.py', 'scheduler.py', 'block_manager.py', 'model.py', 'attention.py', 'kv_cache.py']
    W, x0, dx, top, rh = 760, 70, 124, 76, 30
    X = {c: x0 + i * dx for i, c in enumerate(cols)}
    msgs = [  # (from, to, label, back)
        ('engine.py', 'scheduler.py', 'schedule()', False),
        ('scheduler.py', 'block_manager.py', 'allocate / append', False),
        ('block_manager.py', 'scheduler.py', 'slots, copies', True),
        ('scheduler.py', 'engine.py', 'SchedulerOutput', True),
        ('engine.py', 'kv_cache.py', 'copy_blocks(copies): copy-on-write first', False),
        ('engine.py', 'model.py', 'forward(Batch)', False),
        ('model.py', 'kv_cache.py', 'write(slots, k, v)', False),
        ('model.py', 'attention.py', 'paged_attention', False),
        ('attention.py', 'kv_cache.py', 'read blocks', False),
        ('model.py', 'engine.py', 'logits, one row per sequence', True),
        ('engine.py', 'block_manager.py', 'fork() and free()', False),
    ]
    H = top + len(msgs) * rh + 40
    b = [text(20, 24, 'One step, in time order: which file calls which (read from top to bottom)', 't-title')]
    for c in cols:
        b += [box(X[c] - 58, 38, 116, 28, 'box-on' if c == 'engine.py' else 'box-1', 6),
              text(X[c], 57, c, 't-tick t-strong', 'middle'),
              f'<line class="edge-dim" x1="{X[c]}" y1="68" x2="{X[c]}" y2="{H - 30}"/>']
    for i, (f, t, lab, back) in enumerate(msgs):
        y = top + 18 + i * rh
        x1, x2 = X[f], X[t]
        s = 1 if x2 > x1 else -1
        b.append(f'<line class="{"edge" if back else "edge-on"}" x1="{x1 + 4 * s}" y1="{y}" x2="{x2 - 4 * s}" y2="{y}" '
                 f'marker-end="url(#{"ah" if back else "ah-on"})"{" stroke-dasharray=\"5 4\"" if back else ""}/>')
        b.append(text((x1 + x2) / 2 + (8 if back else 0), y - 7, f'{i + 1}. {lab}', 't-tick', 'middle'))
    yb = top + 18 + 6 * rh - 14
    b += [f'<rect class="box-ghost" x="{X["model.py"] - 50}" y="{yb}" width="{X["kv_cache.py"] - X["model.py"] + 100}" '
          f'height="{3 * rh + 4}" rx="6"/>',
          text(X['kv_cache.py'] + 50, yb + 3 * rh + 20, 'steps 7 to 9 repeat in each of the 12 layers', 't-muted', 'end')]
    return svg(W, H, 'A sequence diagram of one step. The engine asks the scheduler for work; the scheduler gets slots '
               'and copies from the block manager. The engine copies blocks, then runs the model. In each layer the '
               'model writes keys and values to the cache and PagedAttention reads blocks. The logits go back to the '
               'engine, which forks and frees through the block manager.', b)


def step():
    """One engine step with real values: decoding step 1 of the two-request run."""
    st0, st1 = R['fig6'][0], R['fig6'][1]
    a1, b1 = st0['A'][1]['physical'], st0['B'][1]['physical']
    slot_a, slot_b = a1 * 4 + 3, b1 * 4 + 3
    tok_a, tok_b = st1['A'][1]['tokens'][3], st1['B'][1]['tokens'][3]
    nxt_a, nxt_b = R['fig6'][2]['A'][2]['tokens'][0], R['fig6'][2]['B'][2]['tokens'][0]
    tab_a = [r['physical'] for r in st0['A']]
    tab_b = [r['physical'] for r in st0['B']]
    stages = [
        ('1. schedule', 'scheduler.py, block_manager.py',
         [f'A and B are running; each needs one slot for the token it feeds.',
          f'A: last block {a1} has 1 free place -> slot {slot_a}.  B: block {b1} -> slot {slot_b}.',
          'No block taken, no copy, no preemption.']),
        ('2. build the batch', 'engine.py',
         [f'tokens ["{tok_a}", "{tok_b}"], positions [7, 7], slots [{slot_a}, {slot_b}]',
          f'block tables {tab_a} and {tab_b}, context lengths [8, 8]']),
        ('3. run the model', 'model.py, attention.py, kv_cache.py',
         [f'In each of 12 layers: write the 2 new keys and values at slots {slot_a} and {slot_b},',
          f'then A attends over blocks {tab_a} (8 tokens) and B over blocks {tab_b}.',
          'Out: logits, 2 rows of 50,272 scores.']),
        ('4. sample', 'engine.py',
         [f'Greedy: the largest score of each row. A writes "{nxt_a}", B writes "{nxt_b}".']),
        ('5. tidy up', 'engine.py, block_manager.py',
         ['Nobody finished, so nothing is freed. Both last blocks are now full,',
          'so in the next step append_slot takes a new block for each request.']),
    ]
    W, y, b = 720, 46, [text(20, 24, 'One engine step with real values: decoding step 1 of requests A and B (block size 4)', 't-title')]
    for i, (name, files, lines) in enumerate(stages):
        h = 34 + 18 * len(lines)
        b += [box(20, y, 680, h, 'box-on' if i == 2 else 'box-1', 8), text(34, y + 22, name, 't-note'),
              text(686, y + 22, files, 't-mono', 'end')]
        for k, ln in enumerate(lines):
            b.append(text(34, y + 42 + 18 * k, ln, 't-tick'))
        if i < len(stages) - 1:
            b.append(arrow(360, y + h + 1, 360, y + h + 15))
        y += h + 16
    return svg(W, y + 4, f'Five stages of one engine step. Schedule: A gets slot {slot_a}, B slot {slot_b}. Build the '
               f'batch of 2 tokens. Run the model: write keys and values, read blocks through the block tables. Sample: '
               f'both write "{nxt_a}". Tidy up: nothing finished.', b)


def batch():
    """The prompt step of the A/B run: two prompts joined into one list of 14 tokens."""
    st = R['fig6'][0]
    toks, pos, slots, owner = [], [], [], []
    for name in ('A', 'B'):
        rows = st[name]
        for r in rows:
            for k, t in enumerate(r['tokens']):
                toks.append(t)
                pos.append(r['logical'] * 4 + k)
                slots.append(r['physical'] * 4 + k)
                owner.append(name)
    n = len(toks)
    W, L, cw = 720, 92, (720 - 92 - 16) / n
    b = [text(20, 24, f'The prompt step: two prompts joined into one list of {n} tokens', 't-title')]
    labels = [('token', 50), ('position', 86), ('slot', 116)]
    for lab, yy in labels:
        b.append(text(L - 10, yy + 4, lab, 't-label', 'end'))
    for i in range(n):
        x = L + i * cw
        cls = 'box-1' if owner[i] == 'A' else 'box-2'
        b += token(x, 34, cw - 4, toks[i], cls, 30, 't-tick', 4)
        b.append(text(x + cw / 2 - 2, 90, str(pos[i]), 't-tick', 'middle'))
        b.append(text(x + cw / 2 - 2, 120, str(slots[i]), 't-mono', 'middle'))
    na = owner.count('A')
    for name, i0, i1 in (('A', 0, na), ('B', na, n)):
        x1, x2 = L + i0 * cw, L + i1 * cw - 4
        tab = [r['physical'] for r in st[name]]
        b += [f'<path class="edge" d="M{x1:.1f},136 v8 H{x2:.1f} v-8"/>',
              text((x1 + x2) / 2, 162, f'request {name}: prefill ({i0}, {i1}), block table {tab}', 't-tick', 'middle'),
              text((x1 + x2) / 2, 180, f'its logits come from token {i1 - 1} only', 't-muted', 'middle')]
    b.append(text(20, 210, 'Layer norms, projections and the feed-forward run once over all 14 tokens. '
                  'Only attention keeps A and B apart.', 't-note'))
    return svg(W, 222, f'The {n} tokens of the prompt step in one list. Request A fills positions 0 to 6 and slots 0 to 6; '
               'request B fills positions 0 to 6 and slots 8 to 14. Each request is a prefill range with its own block table.', b)


def slot():
    """Address translation: a token's position to its slot, through the block table of request B."""
    st = R['fig6'][-1]['B']
    tab = [r['physical'] for r in st]
    toks = [t for r in st for t in r['tokens']]
    ex = [(2, toks[2]), (8, toks[8])]
    W, b = 720, [text(20, 24, f'Where does a token live? Request B, block size 4, block table {tab}', 't-title')]
    heads = ['token', 'position p', 'p // 4, p % 4', 'block table', 'physical block', 'slot']
    xs = [20, 120, 230, 360, 480, 590]
    ws = [90, 100, 120, 110, 100, 120]
    for x, w, h in zip(xs, ws, heads):
        b.append(text(x + w / 2, 56, h, 't-label', 'middle'))
    for r, (p, t) in enumerate(ex):
        y = 70 + r * 66
        j, k = p // 4, p % 4
        phys = tab[j]
        vals = [f'"{t}"', str(p), f'logical {j}, place {k}', f'entry {j} = {phys}', f'block {phys}', f'{phys} x 4 + {k} = {phys * 4 + k}']
        for i, (x, w, v) in enumerate(zip(xs, ws, vals)):
            b += [box(x, y, w, 40, 'box-on' if i == 5 else 'box-2', 8), text(x + w / 2, y + 25, v, 't-tick', 'middle')]
            if i < 5:
                b.append(arrow(x + w + 1, y + 20, xs[i + 1] - 2, y + 20))
    b += [text(20, 212, 'slot = physical block x block size + place', 't-note'),
          text(20, 232, 'The slot is the token\'s address in the cache tensor. Every layer uses the same slot.', 't-tick')]
    return svg(W, 244, f'Two tokens of request B. "{ex[0][1]}" at position 2 is in logical block 0, place 2, which the '
               f'block table maps to physical block {tab[0]}, so its slot is {tab[0] * 4 + 2}. "{ex[1][1]}" at position 8 '
               f'is in logical block 2, place 0, physical block {tab[2]}, slot {tab[2] * 4}.', b)


def first_step():
    """Before vLLM and with vLLM: the same 2,048-slot cache after the first step of the serving run."""
    s, B, nb = R['serve'], R['serve_setup']['block_size'], R['serve_setup']['blocks']
    total = nb * B
    W, L, bw = 720, 220, 480
    b = [text(20, 24, f'The same cache of {total:,} slots after the first step of the serving run', 't-title')]
    for r, (name, sub, k) in enumerate((('one chunk each', 'before vLLM (512 slots booked)', 'contiguous'),
                                        ('paged', 'with vLLM (blocks of 16)', 'paged'))):
        y = 50 + r * 90
        tokens, held, nseq = s[k]['tokens'][0], s[k]['held'][0] * B, s[k]['curve'][0]
        segs = [('s1', tokens), ('waste', held - tokens), ('box', total - held)]
        x = L
        b += [text(L - 12, y + 18, name, 't-note', 'end'), text(L - 12, y + 36, sub, 't-muted', 'end')]
        for cls, v in segs:
            w = bw * v / total
            if w > 0:
                b.append(f'<rect class="{cls}" x="{x:.1f}" y="{y}" width="{w:.1f}" height="28" rx="2"/>')
            x += w
        b.append(text(L, y + 50, f'{nseq} requests running; {tokens:,} slots hold a token; '
                      f'{held - tokens:,} booked but empty; {total - held:,} free', 't-tick'))
    b += [f'<rect class="s1" x="{L}" y="226" width="12" height="12" rx="2"/>', text(L + 18, 236, 'holds a token', 't-tick'),
          f'<rect class="waste" x="{L + 130}" y="226" width="12" height="12" rx="2"/>', text(L + 148, 236, 'booked, empty', 't-tick'),
          f'<rect class="box" x="{L + 270}" y="226" width="12" height="12" rx="2"/>', text(L + 288, 236, 'free', 't-tick')]
    c, p = s['contiguous'], s['paged']
    return svg(W, 250, f'After the first step, the contiguous run has {c["curve"][0]} requests and {c["tokens"][0]} slots '
               f'holding a token, with the rest booked but empty. The paged run has {p["curve"][0]} requests and '
               f'{p["tokens"][0]:,} slots holding a token, {p["held"][0] * B - p["tokens"][0]} booked but empty and '
               f'{total - p["held"][0] * B} free.', b)


def tables():
    """Our Figures 6 and 7: two requests after two decoding steps, logical blocks and the physical blocks."""
    st = R['fig6'][-1]
    W, H, cw, ch = 720, 400, 58, 26
    b = [text(20, 26, 'Our run: two requests, block size 4, 9 physical blocks (after decoding step 2)', 't-title')]
    phys = {}
    for name in ('A', 'B'):
        for r in st[name]:
            phys[r['physical']] = (name, r['tokens'])
    px, py = 470, 56
    b.append(text(px + 2 * cw, py - 8, 'physical KV blocks', 't-tick', 'middle'))
    for k in range(9):
        y = py + k * (ch + 6)
        b.append(text(px - 8, y + 17, f'block {k}', 't-tick', 'end'))
        owner, toks = phys.get(k, (None, []))
        for i in range(4):
            cls = ('box-1' if owner == 'A' else 'box-2') if i < len(toks) else 'box'
            b += token(px + i * cw, y, cw - 4, toks[i] if i < len(toks) else '', cls, ch, 't-tick', 4)
    for n, (name, x0, y0) in enumerate((('A', 40, 70), ('B', 40, 250))):
        b.append(text(x0, y0 - 10, f'request {name}: logical blocks', 't-label'))
        for r in st[name]:
            y = y0 + r['logical'] * (ch + 6)
            for i in range(4):
                tk = r['tokens'][i] if i < len(r['tokens']) else ''
                b += token(x0 + 40 + i * cw, y, cw - 4, tk, ('box-1' if name == 'A' else 'box-2') if tk else 'box', ch, 't-tick', 4)
            b.append(text(x0 + 30, y + 17, str(r['logical']), 't-tick', 'end'))
            b.append(text(x0 + 40 + 4 * cw + 6, y + 17, f'-> {r["physical"]}', 't-mono'))
            b.append(arrow(x0 + 40 + 4 * cw + 40, y + ch / 2, px - 60, py + r['physical'] * (ch + 6) + ch / 2))
    return svg(W, H, 'Two requests with their logical blocks on the left and the nine physical blocks on the right. '
               'Each logical block points to a physical block; neighbouring logical blocks sit in physical blocks '
               'that are not neighbours.', b)


def cow():
    """Copy-on-write in our run: the two samples before and after the first write."""
    tr = R['cow']['trace']
    W, H, cw, ch = 720, 300, 46, 26
    b = [text(20, 26, 'Parallel sampling in our run: two samples, one prompt (block size 4)', 't-title')]
    for col, (k, title) in enumerate(((0, 'after the prompt step'), (1, 'after decoding step 1'))):
        st = tr[k]
        x0 = 20 + col * 360
        b.append(text(x0, 60, title, 't-label'))
        for r, s in enumerate(('A1', 'A2')):
            y = 80 + r * 96
            b.append(text(x0, y + 16, f'sample {s}', 't-tick'))
            toks = st[s + '_tokens']
            for j, blk in enumerate(st[s]):
                yy = y + 22 + j * (ch + 4)
                cells = toks[j * 4:(j + 1) * 4]
                shared = st['ref'][str(blk)] > 1
                for i in range(4):
                    tk = cells[i] if i < len(cells) else ''
                    b += token(x0 + i * cw, yy, cw - 4, tk, ('box-on' if shared else 'box-1') if tk else 'box', ch, 't-tick', 4)
                b.append(text(x0 + 4 * cw + 6, yy + 17, f'block {blk}, ref {st["ref"][str(blk)]}', 't-mono'))
        cp = st['copies']
        b.append(text(x0, 290, ('copy-on-write: block %d copied to block %d' % tuple(cp[0])) if cp else 'no copy yet: both samples point at the same 2 blocks', 't-note'))
    return svg(W, H, 'Left: after the prompt step, both samples point at the same two physical blocks with a reference '
               'count of 2. Right: after the first decoding step, sample A1 got a copy of the shared last block and '
               'wrote its token there; sample A2 wrote into the original block, whose count is now 1.', b)


def preempt():
    """Which request runs at each step of the preemption run."""
    tl, n = R['preempt']['timeline'], 6
    W, x0, y0, rh = 720, 110, 60, 26
    sx = (W - x0 - 20) / len(tl)
    b = [text(20, 26, 'Six requests, a cache of 12 blocks: who runs at each step', 't-title'),
         f'<rect class="s1" x="20" y="36" width="12" height="12" rx="2"/>', text(38, 46, 'running', 't-tick'),
         f'<rect class="waste" x="110" y="36" width="12" height="12" rx="2"/>', text(128, 46, 'waiting after a preemption', 't-tick')]
    started = set()
    for r in range(n):
        y = y0 + r * rh
        b.append(text(x0 - 10, y + 15, f'request {r}', 't-tick', 'end'))
        for s, st in enumerate(tl):
            if r in st['running']:
                started.add(r)
                b.append(f'<rect class="s1" x="{x0 + s * sx:.1f}" y="{y}" width="{sx + 0.3:.2f}" height="{rh - 6}"/>')
            elif r in st['waiting'] and r in started:
                b.append(f'<rect class="waste" x="{x0 + s * sx:.1f}" y="{y}" width="{sx + 0.3:.2f}" height="{rh - 6}"/>')
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + n * rh}" x2="{W - 20}" y2="{y0 + n * rh}"/>')
    for s in range(0, len(tl) + 1, 25):
        b.append(text(x0 + s * sx, y0 + n * rh + 16, str(s), 't-tick', 'middle'))
    b.append(text(x0 + (W - x0 - 20) / 2, y0 + n * rh + 36, 'step', 't-tick', 'middle'))
    return svg(W, y0 + n * rh + 46, 'A row per request and a column per step. Filled cells are steps in which the '
               'request runs; hatched cells are steps in which it waits after being preempted.', b)


def serving():
    """Sequences in the batch at each step, for both runs, drawn as two step lines."""
    s = R['serve']
    c, p = s['contiguous']['curve'], s['paged']['curve']
    W, H, L, T, B, Rr = 720, 300, 60, 40, 46, 20
    m = max(len(c), len(p))
    top = 10 * (max(max(c), max(p)) // 10 + 1)
    X = lambda i: L + i / m * (W - L - Rr)
    Y = lambda v: T + (1 - v / top) * (H - T - B)
    b = [text(20, 24, 'Sequences in the batch at each step (64 Alpaca requests, same 2,048-slot cache)', 't-title')]
    for t in range(0, top + 1, top // 5):
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{W - Rr}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, str(t), 't-tick', 'end')]
    for i in range(0, m + 1, 100):
        b.append(text(X(i), H - B + 18, str(i), 't-tick', 'middle'))
    b.append(text(L + (W - L - Rr) / 2, H - 8, 'step', 't-tick', 'middle'))
    for name, a, cls in (('paged', p, 'l1'), ('contiguous', c, 'l2')):
        pts = ' '.join(f'{X(i):.1f},{Y(v):.1f} {X(i + 1):.1f},{Y(v):.1f}' for i, v in enumerate(a))
        b.append(f'<polyline class="{cls}" points="{pts}"/>')
    b += [f'<rect class="s1" x="{W - 330}" y="{T + 10}" width="12" height="12" rx="2"/>',
          text(W - 312, T + 20, f'paged: mean {s["paged"]["mean_batch"]:.1f}, done after {len(p)} steps', 't-note'),
          f'<rect class="s2" x="{W - 330}" y="{T + 32}" width="12" height="12" rx="2"/>',
          text(W - 312, T + 42, f'contiguous: mean {s["contiguous"]["mean_batch"]:.1f}, done after {len(c)} steps', 't-note')]
    return svg(W, H, f'Sequences in the batch at each step. Paged: mean {s["paged"]["mean_batch"]:.1f}, '
               f'done after {len(p)} steps. Contiguous: mean {s["contiguous"]["mean_batch"]:.1f}, done after {len(c)} steps.', b)


def memory():
    s = R['serve']
    W, H = 720, 170
    b = [text(20, 26, 'Of the KV slots booked at each step, the share that holds a token (mean over steps)', 't-title')]
    b += hbars(20, 50, [('paged', s['paged']['used_share'] * 100), ('contiguous', s['contiguous']['used_share'] * 100)],
               480, row_h=34, label_w=110, fmt=lambda v: f'{v:.1f}%', vmax=100)
    b.append(text(20, 150, 'Same 2,048-slot cache in both runs. The contiguous run books 512 slots per request.', 't-tick'))
    return svg(W, H, f'Paged: {s["paged"]["used_share"] * 100:.1f} percent of booked slots hold a token. '
               f'Contiguous: {s["contiguous"]["used_share"] * 100:.1f} percent.', b)


F['p6_arch'] = arch()
F['p6_flow'] = flow()
F['p6_step'] = step()
F['p6_batch'] = batch()
F['p6_slot'] = slot()
F['p6_tables'] = tables()
F['p6_cow'] = cow()
F['p6_preempt'] = preempt()
F['p6_serving'] = serving()
F['p6_memory'] = memory()
F['p6_first'] = first_step()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part6.json'), 'w'))
print('figures:', ', '.join(F))
