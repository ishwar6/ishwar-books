"""SVG figures for Part 3 of "LLM Inference from the Ground Up" (paging, blocks, the three wastes, sharing, scheduling).
CSS classes only (no colours), so dark, dim and light themes all work. Numbers come from results/p3_paging_stdout.txt,
from the paper (Kwon et al. 2023) or from the figures already published (results/p3_figs_kept.json).
Run: python p3_figs.py  ->  results/p3_figs.json (key -> SVG string)"""
import json, os

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
F = {}
_mid = ['m']


def svg(key, w, h, label, body):
    m = _mid[0]
    defs = (f'<defs><marker id="{m}-a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker>'
            f'<marker id="{m}-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" '
            f'orient="auto-start-reverse"><path class="arrow-on" d="M0,0L10,5L0,10z"/></marker></defs>')
    F[key] = f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">{defs}{"".join(body)}</svg>'


def start(key):
    _mid[0] = 'p3' + key.replace('_', '')


def R(x, y, w, h, c='box', rx=5):
    return f'<rect class="{c}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"/>'


def T(x, y, s, c='t-note', a='start'):
    return f'<text class="{c}" x="{x:.1f}" y="{y:.1f}" text-anchor="{a}">{s}</text>'


def L(x1, y1, x2, y2, c='edge', arrow=True, on=False):
    m = _mid[0] + ('-on' if on else '-a')
    end = f' marker-end="url(#{m})"' if arrow else ''
    return f'<line class="{c}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"{end}/>'


def P(d, c='path', on=True):
    m = _mid[0] + ('-on' if on else '-a')
    return f'<path class="{c}" d="{d}" fill="none" marker-end="url(#{m})"/>'


def block(x, y, filled, c='box-1', cw=9, h=22, new=False):
    """One KV block of 16 slots: filled slots coloured, empty slots dashed."""
    out = []
    for i in range(16):
        out.append(R(x + i * cw, y, cw - 1.5, h, c if i < filled else 'box', 2))
    if new:
        out.append(R(x - 3, y - 3, 16 * cw + 4.5, h + 6, 'box-ghost', 4))
    return ''.join(out)


# ------------------------------------------------------------------ 1. an OS page table
start('os_pages')
b = [T(20, 22, 'what program A sees', 't-tick'), T(250, 22, 'A\'s page table', 't-tick'),
     T(470, 22, 'physical memory (RAM), cut into frames', 't-tick')]
amap = {0: 5, 1: 2, 2: 8, 3: 0}
bmap = {0: 1, 1: 4, 2: 6}
fy = lambda j: 36 + j * 29
for i in range(4):
    y = 40 + i * 50
    b += [R(20, y, 150, 34, 'box-1'), T(95, y + 22, f'page {i}', 't-note', 'middle')]
    b += [R(250, y, 160, 34, 'box'), T(330, y + 22, f'page {i} → frame {amap[i]}', 't-note', 'middle')]
    b += [L(172, y + 17, 247, y + 17)]
    b += [L(412, y + 17, 467, fy(amap[i]) + 12)]
for j in range(10):
    y = fy(j)
    who = [k for k, v in amap.items() if v == j]
    wb = [k for k, v in bmap.items() if v == j]
    if who:
        b += [R(470, y, 200, 24, 'box-1'), T(570, y + 17, f'A, page {who[0]}', 't-note', 'middle')]
    elif wb:
        b += [R(470, y, 200, 24, 'box-2'), T(570, y + 17, f'B, page {wb[0]}', 't-note', 'middle')]
    else:
        b += [R(470, y, 200, 24, 'box'), T(570, y + 17, 'free', 't-muted', 'middle')]
    b += [T(684, y + 17, f'frame {j}', 't-tick')]
b += [T(20, 345, 'A sees pages 0, 1, 2, 3 in order. In RAM they sit in frames 5, 2, 8 and 0, between the pages of program B.', 't-tick')]
svg('os_pages', 760, 356, 'A program sees its pages in order; a page table maps each page to a frame anywhere in physical memory.', b)

# ------------------------------------------------------------------ 2. a program grows: one more page
start('os_grow')
b = []
names = {0: 'A3', 1: 'B0', 2: 'A1', 3: None, 4: 'B1', 5: 'A0', 6: 'B2', 7: None, 8: 'A2', 9: None}
for row, (title, extra) in enumerate([('before: A holds 4 pages; frames 3, 7 and 9 are free', None),
                                      ('A needs more memory: the OS hands it one free frame, frame 3, as page 4', 3)]):
    y0 = 30 + row * 110
    b += [T(20, y0, f'step {row + 1}', 't-title'), T(80, y0, title, 't-note')]
    for j in range(10):
        x = 20 + j * 72
        n = names[j]
        if extra == j:
            b += [R(x, y0 + 14, 66, 36, 'box-on'), T(x + 33, y0 + 37, 'A4', 't-on', 'middle')]
        elif n:
            b += [R(x, y0 + 14, 66, 36, 'box-1' if n[0] == 'A' else 'box-2'), T(x + 33, y0 + 37, n, 't-note', 'middle')]
        else:
            b += [R(x, y0 + 14, 66, 36, 'box'), T(x + 33, y0 + 37, 'free', 't-muted', 'middle')]
        b += [T(x + 33, y0 + 66, f'frame {j}', 't-tick', 'middle')]
b += [T(20, 236, 'A\'s page table after step 2: page 0 → 5, 1 → 2, 2 → 8, 3 → 0, and the new row 4 → 3.', 't-note')]
b += [T(20, 258, 'Nothing was reserved in advance. A gets a frame only at the moment it needs one, and any free frame will do.', 't-tick')]
svg('os_grow', 760, 270, 'Two steps: program A holds four pages, then it needs more memory and the operating system gives it one free frame, anywhere in RAM.', b)

# ------------------------------------------------------------------ 3. KV blocks and block tables
start('kv_blocks')
b = [T(20, 22, 'logical blocks: each request sees its cache in order (one box = one token slot)', 't-tick')]
A = [16, 16, 5]
Bq = [16, 4]
tabA, tabB = [7, 1, 3], [2, 6]
CW = 8.5
for k, f in enumerate(A):
    x = 120 + k * 150
    b += [block(x, 52, f, 'box-1', CW), T(x + 68, 46, f'logical {k}', 't-tick', 'middle')]
for k, f in enumerate(Bq):
    x = 120 + k * 150
    b += [block(x, 118, f, 'box-2', CW), T(x + 68, 112, f'logical {k}', 't-tick', 'middle')]
b += [T(20, 64, 'request A', 't-title'), T(20, 82, '37 tokens', 't-tick'),
      T(20, 130, 'request B', 't-title'), T(20, 148, '20 tokens', 't-tick')]
b += [R(570, 44, 170, 40, 'box'), T(655, 60, 'block table of A', 't-tick', 'middle'), T(655, 77, '[7, 1, 3]', 't-mono', 'middle'),
      R(570, 110, 170, 40, 'box'), T(655, 126, 'block table of B', 't-tick', 'middle'), T(655, 143, '[2, 6]', 't-mono', 'middle')]
b += [T(20, 196, 'physical blocks: where the keys and values really are in GPU memory', 't-tick')]
phys = {7: ('A', 0, 16), 1: ('A', 1, 16), 3: ('A', 2, 5), 2: ('B', 0, 16), 6: ('B', 1, 4)}
px = lambda k: 20 + k * 92
for k in range(8):
    x = px(k)
    if k in phys:
        r, lb, f = phys[k]
        b += [block(x, 214, f, 'box-1' if r == 'A' else 'box-2', 5.3, 26)]
        b += [T(x + 42, 258, f'#{k}: {r}, logical {lb}', 't-tick', 'middle')]
    else:
        b += [R(x, 214, 84, 26, 'box', 3), T(x + 42, 232, 'free', 't-muted', 'middle'), T(x + 42, 258, f'#{k}', 't-tick', 'middle')]
for k, p in enumerate(tabA):
    b += [L(120 + k * 150 + 68, 76, px(p) + 42, 210)]
for k, p in enumerate(tabB):
    b += [L(120 + k * 150 + 68, 142, px(p) + 42, 210)]
b += [T(20, 290, 'A block holds the keys and values of 16 tokens, in every layer. For Llama 3.1 8B that is 16 × 128 KiB = 2 MiB.', 't-tick')]
svg('kv_blocks', 760, 302, 'Two requests: their logical blocks map through each request\'s block table to physical blocks scattered in GPU memory.', b)

# ------------------------------------------------------------------ 4. one request grows, step by step
start('grow_steps')
b = []
steps = [(30, [7, 1], [16, 14], None, 'prompt of 30 tokens: 2 blocks; the last one has 2 empty slots'),
         (32, [7, 1], [16, 16], None, 'two decode steps later: the last block is full; still 2 blocks'),
         (33, [7, 1, 3], [16, 16, 1], 2, 'token 33 has no room, so the request gets one new block: 3'),
         (49, [7, 1, 3, 12], [16, 16, 16, 1], 3, 'token 49: block 3 is full, so it gets block 12')]
for s, (n, tab, fill, new, note) in enumerate(steps):
    y = 20 + s * 96
    b += [T(20, y + 18, f'step {s + 1}', 't-title'), T(20, y + 38, f'{n} tokens', 't-tick')]
    for k, f in enumerate(fill):
        x = 110 + k * 158
        b += [block(x, y + 4, f, 'box-1', 9, 24, new == k), T(x + 72, y + 46, f'logical {k} → #{tab[k]}', 't-tick', 'middle')]
    b += [T(110, y + 70, f'block table [{", ".join(map(str, tab))}]  ·  {note}', 't-note')]
b += [T(20, 404, 'A new block arrives only when the last one is full, so at most 15 slots of a request are ever empty.', 't-tick')]
svg('grow_steps', 760, 414, 'Four steps of one request growing: 30 tokens in two blocks, 32 tokens filling them, token 33 getting a new block, token 49 getting another.', b)

# ------------------------------------------------------------------ 5. attention over blocks
start('paged_attn')
b = [R(20, 118, 120, 46, 'box-3'), T(80, 138, 'query of the', 't-note', 'middle'), T(80, 155, 'new token 38', 't-note', 'middle')]
b += [R(190, 92, 160, 98, 'box'), T(270, 110, 'block table', 't-tick', 'middle')]
rows = [('logical 0 → #7', 7, 'tokens 1 to 16'), ('logical 1 → #1', 1, 'tokens 17 to 32'), ('logical 2 → #3', 3, 'tokens 33 to 37')]
for i, (r, p, lab) in enumerate(rows):
    b += [T(270, 132 + i * 20, r, 't-mono', 'middle')]
b += [L(142, 141, 186, 141)]
by = lambda k: 30 + k * 30
for k in range(8):
    used = {7: 16, 1: 16, 3: 5}.get(k)
    if used:
        b += [block(420, by(k), used, 'box-1', 9, 22)]
    else:
        b += [R(420, by(k), 142, 22, 'box', 3)]
    b += [T(570, by(k) + 16, f'#{k}', 't-tick')]
for i, (r, p, lab) in enumerate(rows):
    b += [L(352, 128 + i * 20, 416, by(p) + 11, 'edge-on', True, True)]
    b += [T(600, by(p) + 16, f'{i + 1}. {lab}' , 't-tick')]
b += [T(20, 284, '4. Softmax over all 37 scores, then a weighted sum of the 37 values. The answer is the same as reading one array.', 't-note')]
svg('paged_attn', 760, 296, 'The attention kernel reads the block table, visits physical blocks 7, 1 and 3 in logical order, and combines the scores of all 37 tokens.', b)

# ------------------------------------------------------------------ 6. waste 1: reserved slots
start('waste_reserved')
sc = 680 / 2048
b = [T(20, 24, 'before: one slab of 2,048 slots, reserved when the request arrives (now: 37 tokens; the answer will end at 300)', 't-note')]
b += [R(40, 36, 37 * sc, 30, 's1', 2), R(40 + 37 * sc, 36, 263 * sc, 30, 'box-2', 2), R(40 + 300 * sc, 36, 1748 * sc, 30, 'waste', 2)]
b += [T(40, 84, '37 used', 't-tick'), T(40 + 37 * sc + 4, 100, '263 reserved slots: empty now, filled later', 't-s2'),
      T(40 + 300 * sc + 10, 56, '1,748 slots that the answer will never reach', 't-tick')]
b += [T(20, 136, 'after: paged, 16-token blocks, drawn on the same scale', 't-note')]
b += [R(40, 148, 37 * sc, 30, 's1', 2), R(40 + 37 * sc, 148, 11 * sc, 30, 'box-2', 2)]
b += [T(64, 168, '← 48 slots: almost nothing', 't-tick')]
b += [T(20, 214, 'the same 48 slots, zoomed in: 3 blocks', 't-tick')]
for k, f in enumerate([16, 16, 5]):
    b += [block(40 + k * 220, 226, f, 'box-1', 12.5, 26)]
b += [T(40 + 2 * 220 + 64 + 40, 276, '11 empty slots, filled by the next 11 tokens', 't-s2', 'middle')]
b += [T(20, 306, 'Right now the slab holds 2,011 empty slots. The blocks hold 11, and a 4th block comes only at token 49.', 't-tick')]
svg('waste_reserved', 760, 318, 'Reserved slots: a 2,048-slot slab holds 37 tokens and 263 slots kept for later; paged storage holds 3 blocks with 11 empty slots.', b)

# ------------------------------------------------------------------ 7. waste 2: internal fragmentation
start('waste_internal')
b = [T(20, 24, 'before: the answer has ended at 300 tokens; the slab was 2,048 slots', 't-note')]
b += [R(40, 36, 300 * sc, 30, 's1', 2), R(40 + 300 * sc, 36, 1748 * sc, 30, 'waste', 2)]
b += [T(44, 84, '300 used', 't-tick'), T(40 + 300 * sc + 10, 56, '1,748 slots never used (14.6% of the slab used)', 't-tick')]
b += [T(20, 128, 'after: 19 blocks of 16 = 304 slots (zoomed in)', 't-note')]
for k in range(19):
    x = 40 + k * 36
    if k < 18:
        b += [R(x, 142, 32, 30, 's1', 3)]
    else:
        b += [R(x, 142, 32 * 12 / 16, 30, 's1', 2), R(x + 24, 142, 8, 30, 'waste', 2)]
b += [T(40 + 18 * 36 + 32, 194, 'last block: 12 of 16 used, 4 empty', 't-tick', 'end')]
b += [T(20, 228, 'Paged: 4 slots never used (98.7% used). The worst case is 15 empty slots, in the last block only.', 't-tick')]
svg('waste_internal', 760, 240, 'Internal fragmentation: a finished 300-token answer leaves 1,748 unused slots in a 2,048 slab, but only 4 in its last 16-token block.', b)

# ------------------------------------------------------------------ 8. waste 3: external fragmentation
start('waste_external')
mem = 'AAAA..BBB...CC..'
cls = {'A': 'box-1', 'B': 'box-2', 'C': 'box-3'}
b = [T(20, 24, 'before: 16 slots, 7 of them free, but in gaps of 2, 3 and 2', 't-note')]
for i, c in enumerate(mem):
    x = 40 + i * 42
    b += [R(x, 36, 38, 32, cls.get(c, 'box'), 3), T(x + 19, 57, c if c != '.' else 'free', 't-note' if c != '.' else 't-muted', 'middle')]
b += [T(20, 96, 'request D needs 6 slots in one piece:', 't-tick')]
for i in range(6):
    b += [R(260 + i * 42, 82, 38, 22, 'box-on', 3)]
b += [T(530, 98, 'no gap is big enough: D must wait', 't-bad')]
b += [T(20, 146, 'after: memory cut into blocks of 2 slots; D needs 3 blocks, and any 3 free blocks will do', 't-note')]
pm = ['AA', 'AA', 'DD', 'BB', 'B.', 'DD', 'CC', 'DD']
dn = {2: 0, 5: 1, 7: 2}
for k, pair in enumerate(pm):
    x0 = 40 + k * 84
    for j, c in enumerate(pair):
        x = x0 + j * 42
        c2 = cls.get(c, 'box-on' if c == 'D' else 'box')
        lab = c if c in 'ABC' else ('D' if c == 'D' else 'empty')
        b += [R(x, 158, 38, 32, c2, 3), T(x + 19, 179, lab, 't-note' if c != '.' else 't-muted', 'middle')]
    b += [R(x0 - 2, 155, 82, 38, 'box-ghost', 4), T(x0 + 40, 210, f'block {k}' + (f' = D{dn[k]}' if k in dn else ''), 't-tick', 'middle')]
b += [T(20, 242, 'D\'s block table: [2, 5, 7]. Every block is the same size, so no gap is ever too small for a block.', 't-tick')]
svg('waste_external', 760, 254, 'External fragmentation: seven free slots in three small gaps cannot hold a six-slot request; with two-slot blocks the request uses blocks 2, 5 and 7.', b)

# ------------------------------------------------------------------ 9. the cost: contiguous read vs gather through a table
start('cost_gather')
b = [T(20, 24, 'contiguous: the keys and values of one request are one array', 't-note')]
for i in range(37):
    b += [R(40 + i * 18, 36, 16, 24, 'box-1', 2)]
b += [P('M40,74 L700,74', 'path', True), T(40, 94, 'the kernel knows the start and the length, and reads straight through: one address, one sweep', 't-tick')]
b += [T(20, 134, 'paged: the kernel must first find each block', 't-note')]
b += [R(40, 148, 220, 96, 'box'), T(150, 168, 'block table, length 37', 't-tick', 'middle')]
order = [7, 1, 3]
bx = lambda k: 300 + k * 56
for i, p in enumerate(order):
    b += [T(56, 192 + i * 20, f'{i + 1}. logical {i} → #{p}', 't-mono')]
for k in range(8):
    used = k in order
    b += [R(bx(k), 214, 50, 40, 'box-1' if used else 'box', 4), T(bx(k) + 25, 239, f'#{k}', 't-note' if used else 't-muted', 'middle')]
for i, p in enumerate(order):
    b += [L(262, 188 + i * 20, bx(p) + 25, 210, 'edge-on', True, True)]
    b += [T(bx(p) + 25, 270, f'visit {i + 1}', 't-on', 'middle')]
b += [T(20, 306, 'For every block, extra work: read the table entry, work out where the block starts, and check how full it is', 't-tick'),
      T(20, 324, '(the last block is partial, and requests in one batch have different lengths). That is the price of paging.', 't-tick')]
svg('cost_gather', 760, 336, 'A contiguous array is read in one sweep; paged storage makes the kernel read the block table first and then jump to blocks 7, 1 and 3.', b)

# ------------------------------------------------------------------ 10. what paging costs and what it buys
start('cost_gain')
b = [T(20, 24, 'what it costs: attention kernel time (paper, Fig. 18a)', 't-note'),
     T(410, 24, 'what it buys: requests that fit (our simulation)', 't-note')]
b += [T(20, 56, 'contiguous', 't-tick'), R(100, 42, 160, 24, 's2', 3), T(266, 59, '100', 't-val'),
      T(20, 92, 'paged', 't-tick'), R(100, 78, 192, 24, 's1', 3), R(292, 78, 9.6, 24, 'box-1', 0), T(308, 95, '120 to 126', 't-val')]
b += [T(100, 124, 'contiguous kernel = 100', 't-tick')]
mx = 230 / 415
b += [T(410, 56, 'reserve max', 't-tick'), R(500, 42, 48 * mx, 24, 's2', 3), T(500 + 48 * mx + 6, 59, '48', 't-val'),
      T(410, 92, 'paged', 't-tick'), R(500, 78, 415 * mx, 24, 's1', 3), T(500 + 415 * mx - 6, 95, '415', 't-cell on', 'end')]
b += [T(500, 124, 'Llama 3.1 8B, one 80 GB GPU', 't-tick')]
b += [T(20, 162, 'Attention gets 20 to 26% slower. The batch can get more than 8 times larger. Part 1 showed that larger batches', 't-tick'),
      T(20, 180, 'are where throughput comes from, so the trade is lopsided in favour of paging.', 't-tick')]
svg('cost_gain', 760, 192, 'Left: the paged attention kernel takes 20 to 26 percent longer than a contiguous one. Right: paged storage fits 415 requests where reserving the maximum fits 48.', b)

# ------------------------------------------------------------------ 11. copy-on-write
start('cow')
panels = [('1. after prefill', '[7, 1]', '[7, 1]', [('#7', 'ref 2', 'box-3'), ('#1', 'ref 2', 'box-3'), ('#3', 'free', 'box')],
           'both samples point at the same 2 blocks'),
          ('2. sample 1 writes token 21', '[7, 3]', '[7, 1]', [('#7', 'ref 2', 'box-3'), ('#1', 'ref 1', 'box-2'), ('#3', 'copy of #1, ref 1', 'box-on')],
           '#1 is shared: copy it, then write'),
          ('3. sample 2 writes token 21', '[7, 3]', '[7, 1]', [('#7', 'ref 2', 'box-3'), ('#1', 'ref 1, written', 'box-on'), ('#3', 'ref 1', 'box-1')],
           '#1 is no longer shared: write in place')]
b = []
for i, (title, t1, t2, blocks_, note) in enumerate(panels):
    x = 20 + i * 250
    b += [T(x, 24, title, 't-title')]
    b += [T(x, 50, 'sample 1 table', 't-tick'), T(x + 150, 50, t1, 't-mono'), T(x, 72, 'sample 2 table', 't-tick'), T(x + 150, 72, t2, 't-mono')]
    for j, (n, lab, c) in enumerate(blocks_):
        y = 92 + j * 44
        b += [R(x, y, 220, 34, c), T(x + 12, y + 22, n, 't-strong'), T(x + 50, y + 22, lab, 't-note')]
    b += [T(x, 240, note, 't-tick')]
b += [T(20, 270, 'The prompt (20 tokens) is stored once. Only the block that both samples want to change is copied, and only once.', 't-tick')]
svg('cow', 760, 282, 'Copy-on-write in three steps: two samples share blocks 7 and 1; the first to write into block 1 gets a copy, block 3; the second writes in place.', b)

# ------------------------------------------------------------------ 12. prefix caching: chained hashes
start('prefix_hash')
rowsp = [('request 1', ['9266d336', 'e9d16e46', 'df7fc873'], ['system 1-16', 'system 17-32', 'question'], [False] * 3, 'new: prefill all 50 tokens'),
         ('request 2', ['9266d336', 'e9d16e46', '3953b78c'], ['system 1-16', 'system 17-32', 'question'], [True, True, False], 'hit: reuse 32 tokens, prefill 17'),
         ('request 3', ['5b33b966', '255d9ee6', '9925b8e7'], ['other first token', 'system 17-32', 'question'], [False] * 3, 'first token differs: prefill all 50')]
b = [T(20, 22, 'name of each full block = hash(name of the block before it, the 16 tokens of this block)', 't-tick')]
for r, (name, hs, labs, hit, note) in enumerate(rowsp):
    y = 44 + r * 80
    b += [T(20, y + 27, name, 't-title')]
    for k in range(3):
        x = 110 + k * 160
        b += [R(x, y, 140, 44, 'box-on' if hit[k] else ('box-3' if r == 0 else 'box'))]
        b += [T(x + 70, y + 18, labs[k], 't-tick', 'middle'), T(x + 70, y + 36, hs[k], 't-mono', 'middle')]
        if k:
            b += [L(x - 20, y + 22, x - 3, y + 22)]
    n1, n2 = note.split(': ')
    b += [T(580, y + 18, n1 + ':', 't-note'), T(580, y + 36, n2, 't-tick')]
b += [T(20, 296, 'Highlighted blocks were found in the cache by their name and reused. A changed token changes the name of every later block.', 't-tick')]
svg('prefix_hash', 760, 308, 'Three requests and their block names: request 2 shares the first two names with request 1 and reuses them; request 3 differs in its first token and reuses nothing.', b)

# ------------------------------------------------------------------ 13. preemption: swap or recompute
start('preempt')
b = [R(20, 70, 170, 110, 'box'), T(105, 92, 'GPU memory is full', 't-note', 'middle'), T(105, 116, 'request C is the', 't-tick', 'middle'),
     T(105, 133, 'newest arrival:', 't-tick', 'middle'), T(105, 150, 'it gives way', 't-tick', 'middle')]
b += [T(240, 40, 'swap', 't-title'), R(240, 52, 220, 50, 'box-2'), T(350, 72, 'copy all of C\'s blocks', 't-note', 'middle'), T(350, 90, 'to CPU memory', 't-tick', 'middle'),
      R(510, 52, 230, 50, 'box-2'), T(625, 72, 'later: copy them back', 't-note', 'middle'), T(625, 90, 'over the CPU-GPU link', 't-tick', 'middle')]
b += [T(240, 150, 'recompute (the V1 default)', 't-title'), R(240, 162, 220, 50, 'box-1'), T(350, 182, 'free all of C\'s blocks', 't-note', 'middle'), T(350, 200, 'and keep only its tokens', 't-tick', 'middle'),
      R(510, 162, 230, 50, 'box-1'), T(625, 182, 'later: one prefill over', 't-note', 'middle'), T(625, 200, 'prompt + tokens so far', 't-tick', 'middle')]
b += [L(192, 110, 236, 78), L(192, 140, 236, 186), L(462, 77, 506, 77), L(462, 187, 506, 187)]
b += [T(20, 244, 'Either way, C loses all its blocks at once and comes back later with the same text. Its answer does not change.', 't-tick')]
svg('preempt', 760, 256, 'When memory runs out, the newest request gives way: either its blocks are copied to CPU memory and back, or they are freed and rebuilt later with one prefill.', b)

# ------------------------------------------------------------------ 14. two GPUs, one block table
start('tp_blocks')
b = [R(20, 70, 200, 90, 'box-3'), T(120, 94, 'scheduler', 't-strong', 'middle'), T(120, 116, 'one block table per request', 't-tick', 'middle'),
     T(120, 140, 'A: [7, 1, 3]', 't-mono', 'middle')]
for g in range(2):
    y = 30 + g * 110
    b += [R(300, y, 440, 90, 'box'), T(316, y + 22, f'GPU {g}', 't-title'), T(370, y + 22, f'stores KV heads {4 * g} to {4 * g + 3} of every layer', 't-tick')]
    for k, p in enumerate([7, 1, 3]):
        b += [R(316 + k * 140, y + 38, 126, 36, 'box-1'), T(316 + k * 140 + 63, y + 61, f'#{p}: heads {4 * g}-{4 * g + 3}', 't-note', 'middle')]
    b += [L(222, 115, 296, y + 45)]
b += [T(20, 256, 'Both GPUs use the same block numbers. Each one keeps only its own share of the heads (Llama 3.1 8B has 8 KV heads).', 't-tick')]
svg('tp_blocks', 760, 268, 'Tensor parallelism on two GPUs: the scheduler keeps one block table; both GPUs use the same block numbers, and each stores the keys and values of its own heads.', b)

# ------------------------------------------------------------------ 15. the startup arithmetic (computed, not measured)
start('startup_math')
steps = [('80 GB', 'H100 memory'), ('× 0.92', 'utilization'), ('73.60 GB', 'vLLM may use'),
         ('− 16.06 GB', 'weights'), ('− 4 GB', 'working, assumed'), ('53.54 GB', '= 49.86 GiB')]
b = [T(20, 22, '1. how much memory the KV cache gets', 't-title')]
for i, (v, lab) in enumerate(steps):
    x = 20 + i * 122
    b += [R(x, 34, 110, 52, 'box-3' if i in (2, 5) else 'box'), T(x + 55, 56, v, 't-strong', 'middle'), T(x + 55, 76, lab, 't-tick', 'middle')]
b += [T(20, 122, '2. how many blocks and tokens that is', 't-title')]
steps2 = [('2 MiB', 'one block: 16 tokens × 128 KiB'), ('25,529 blocks', '53.54 GB ÷ 2 MiB'), ('408,464 tokens', 'blocks × 16')]
for i, (v, lab) in enumerate(steps2):
    x = 20 + i * 245
    b += [R(x, 134, 225, 52, 'box-1'), T(x + 112, 156, v, 't-strong', 'middle'), T(x + 112, 176, lab, 't-tick', 'middle')]
b += [T(20, 222, '3. how many requests of the longest allowed length fit', 't-title')]
steps3 = [('--max-model-len 131,072', '8,192 blocks each', '3.12x'), ('--max-model-len 8,192', '512 blocks each', '49.86x')]
for i, (a, c, v) in enumerate(steps3):
    x = 20 + i * 370
    b += [R(x, 234, 350, 52, 'box-2'), T(x + 14, 256, a, 't-mono'), T(x + 14, 276, c, 't-tick'), T(x + 336, 268, v, 't-big', 'end')]
b += [T(20, 312, 'Computed with vLLM v0.32.0\'s formulas, not measured. The real log differs a little: the profiling run measures the working memory.', 't-tick')]
svg('startup_math', 760, 326, 'The startup arithmetic for Llama 3.1 8B on an H100: memory for the cache, then blocks and tokens, then how many full-length requests fit.', b)

# ------------------------------------------------------------------ kept figures from the earlier version of Part 3
kept = json.load(open(os.path.join(RES, 'p3_figs_kept.json')))
kept['chunked'] = kept['chunked'].replace('class="t-val" x="370.66324891522527" y="56" text-anchor="middle" style="fill:#fff"',
                                          'class="t-cell on" x="370.66324891522527" y="56" text-anchor="middle"')
assert 'style=' not in kept['chunked']
F.update(kept)

json.dump(F, open(os.path.join(RES, 'p3_figs.json'), 'w'), indent=0)
print(f'{len(F)} figures:', ', '.join(F))
