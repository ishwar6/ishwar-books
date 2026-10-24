"""Figures for Part 4 (sharing, scheduling and many GPUs). Writes results/figs_part4.json. CSS classes only.
Every number comes from results/part4.json (our run)."""
import json, os
from figlib import svg, text, box, arrow, lines_chart, esc
from samfig import token, hbars, line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part4.json')))
F = {}


def fig8():
    """Our Figure 8: the block tables of 4 samples right after fork, and after each wrote its first token."""
    f = R['fig8']
    W, H = 720, 350
    b = [text(20, 26, 'Our Figure 8: four samples, block size 4 (OPT-125M run)', 't-title')]
    words = [' '.join(w.strip() or '</s>' for w in blk) for blk in f['blocks_text']]
    b.append(text(20, 50, f'prompt blocks: logical 0 = "{words[0]}", logical 1 = "{words[1]}" (1 free slot)', 't-tick'))
    cols = (('after fork', f['before'], 30), ('after the first new token', f['after'], 390))
    shared = set(f['before']['s0'])
    for title, tables, x0 in cols:
        b.append(text(x0, 84, title, 't-tick t-strong'))
        b.append(text(x0 + 110, 104, 'logical 0', 't-tick', 'middle'))
        b.append(text(x0 + 220, 104, 'logical 1', 't-tick', 'middle'))
        for i, (s, t) in enumerate(tables.items()):
            y = 116 + i * 42
            b.append(text(x0, y + 21, f'sample {i + 1}', 't-tick'))
            for j, p in enumerate(t):
                cls = 'box-1' if p in shared else 'box-on'
                b += token(x0 + 60 + j * 110, y, 100, f'block {p}', cls, 32)
    b.append(arrow(330, 200, 380, 200))
    b += [box(30, 294, 12, 12, 'box-1', 2), text(48, 304, 'shared physical block (reference count above 1 before the write)', 't-tick'),
          box(30, 314, 12, 12, 'box-on', 2), text(48, 324, f'new block made by copy-on-write ({f["cow_after_first"]} copies)', 't-tick')]
    ref = ', '.join(f'block {k}: {v}' for k, v in f['ref_before'].items())
    b.append(text(30, 344, f'reference counts after fork: {ref}', 't-tick'))
    return svg(W, H, 'Left: four samples whose block tables all point to the same two physical blocks. Right: after the '
               'first new token, three samples got their own copy of the second block and the last one writes in place.', b)


def fig9():
    """Our Figure 9: the block tables of the 4 beams at the end, coloured by how many beams map each block."""
    f = R['fig9']
    W, H = 720, 280
    cls_of = {4: 'box-1', 3: 'box-2', 2: 'box-3', 1: 'box-4'}
    b = [text(20, 26, 'Our Figure 9: block tables of 4 beams after 24 steps, block size 4', 't-title')]
    for i, t in enumerate(f['tables']):
        y = 50 + i * 42
        b.append(text(20, y + 21, f'beam {i + 1}', 't-tick'))
        for j, p in enumerate(t):
            b += token(80 + j * 78, y, 72, str(p), cls_of[f['refs'][str(p)]], 32)
    lx = 80
    for n, lab in ((4, 'mapped by all 4 beams'), (3, 'by 3 beams'), (1, 'by 1 beam')):
        b += [box(lx, 230, 12, 12, cls_of[n], 2), text(lx + 18, 240, lab, 't-tick')]
        lx += 40 + 7 * len(lab)
    u, s = f['steps'][-1]
    b.append(text(80, 268, f'{u} physical blocks hold what would need {s} without sharing', 't-tick'))
    return svg(W, H, f'Four rows of eight blocks. The first four blocks are the same in every row; three beams share all '
               f'eight blocks; one beam has its own last four. {u} physical blocks in use instead of {s}.', b)


def blocks_over_steps():
    """Physical blocks in use, with and without sharing, at every step of the beam search and parallel sampling."""
    bs, ss = R['fig9']['steps'], R['fig8']['steps']
    xs = list(range(1, len(bs) + 1))
    return lines_chart(720, 300, 'Physical blocks in use at each step. Beam search with sharing stays near '
                       f'{bs[-1][0]} blocks while it would need {bs[-1][1]} without sharing.', xs,
                       [('beams, no sharing', [s for _, s in bs], 'l3', 's3'),
                        ('beams, shared', [u for u, _ in bs], 'l1', 's1'),
                        ('samples, shared', [u for u, _ in ss], 'l2', 's2')],
                       'decoding step (block size 4)', 'physical blocks', 0, 35, [0, 10, 20, 30],
                       xticks=[1, 5, 10, 15, 20, 23], fmt=lambda v: f'{v:.0f}', right=190)


def saving():
    s = R['sweep']
    items = [(f'{n} samples', v) for n, v in s['sampling'].items()] + [(f'beam width {k}', v) for k, v in s['beam'].items()]
    W, H = 720, 230
    b = hbars(20, 50, items, 470, row_h=26, cls='s1', label_w=130, fmt=lambda v: f'{v:.1f}%', vmax=60,
              title='Memory saved by sharing blocks (block size 16, 41-token prompt, 48 new tokens)')
    b.append(text(150, 222, 'saving = blocks saved by sharing / blocks without sharing, mean over steps', 't-tick'))
    return svg(W, H, 'Bars: parallel sampling saves ' + ', '.join(f'{v:.1f}%' for v in s['sampling'].values())
               + ' for 2, 4 and 6 samples; beam search saves ' + ', '.join(f'{v:.1f}%' for v in s['beam'].values())
               + ' for widths 2, 4 and 6.', b)


def schedule():
    """The scheduler simulation: running requests and free blocks per step, with a tick for each preemption."""
    s = R['sched']['swap']
    batch, free, pre = s['batch'], s['free'], s['preempt_steps']
    W, H, L, T, pw, ph = 720, 360, 60, 70, 620, 220
    n = len(batch)
    X = lambda t: L + t / (n - 1) * pw
    Y = lambda v: T + ph - v / 64 * ph
    b = [text(20, 24, f'Our scheduler on Alpaca lengths: 64 blocks of 16 tokens, {s["preemptions"]} preemptions', 't-title')]
    for v in (0, 16, 32, 48, 64):
        b += [f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + pw}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, str(v), 't-tick', 'end')]
    for t in pre:
        b.append(f'<line class="edge-dim" x1="{X(t):.1f}" y1="{T + ph}" x2="{X(t):.1f}" y2="{T + ph + 8}"/>')
    b.append('<polyline class="l1" points="' + ' '.join(f'{X(t):.1f},{Y(v):.1f}' for t, v in enumerate(batch)) + '"/>')
    b.append('<polyline class="l3" points="' + ' '.join(f'{X(t):.1f},{Y(v):.1f}' for t, v in enumerate(free)) + '"/>')
    for t in range(0, n, 200):
        b.append(text(X(t), T + ph + 24, f'{t:,}', 't-tick', 'middle'))
    b += [text(L + pw / 2, H - 14, 'step (ticks under the axis: a preemption)', 't-tick', 'middle'),
          box(L + 10, T - 26, 12, 12, 's1', 2), text(L + 28, T - 16, 'running requests', 't-tick'),
          box(L + 170, T - 26, 12, 12, 's3', 2), text(L + 188, T - 16, 'free blocks', 't-tick')]
    return svg(W, H, f'Over {n} steps, the number of running requests stays near 12 and the free blocks stay near zero; '
               f'{s["preemptions"]} preemptions happen while the queue is long.', b)


def swap_chart():
    s = R['swap']
    sizes = [int(k) for k in s['sweep']]
    out = [s['sweep'][str(k)][0] for k in sizes]
    inn = [s['sweep'][str(k)][1] for k in sizes]
    rec = [s['rec_ms']] * len(sizes)
    top = max(out + inn + rec)
    ymax = 10 * (int(top // 10) + 1)
    return lines_chart(720, 300, f'Time to swap the {s["tokens"]}-token cache out and in against block size, with the '
                       'flat cost of recomputation. Small blocks make swapping slow.', sizes,
                       [('swap out', out, 'l1', 's1'), ('swap in', inn, 'l2', 's2'), ('recompute', rec, 'l3', 's3')],
                       'block size in tokens (log scale)', 'milliseconds', 0, ymax,
                       list(range(0, ymax + 1, 10 if ymax <= 60 else 20)), xlog=True, fmt=lambda v: f'{v:.1f}')


def tp():
    """Megatron-style tensor parallelism of one attention layer over 2 GPUs."""
    W, H = 720, 340
    b = [text(20, 26, 'One attention layer on 2 GPUs (our OPT-125M run: 12 heads, 6 per GPU)', 't-title')]
    b += token(250, 44, 220, 'input x (same on both GPUs)', 'box', 32)
    for g, x0 in ((0, 40), (1, 380)):
        heads = '0-5' if g == 0 else '6-11'
        b.append(box(x0, 104, 300, 150, 'box-1' if g == 0 else 'box-2', 10))
        b.append(text(x0 + 150, 126, f'GPU {g}: heads {heads}', 't-tick t-strong', 'middle'))
        b.append(text(x0 + 150, 152, 'Q, K, V rows of its own heads', 't-tick', 'middle'))
        b.append(text(x0 + 150, 176, 'attention for 6 heads', 't-tick', 'middle'))
        b.append(text(x0 + 150, 200, 'its 384 columns of the output matrix', 't-tick', 'middle'))
        b.append(text(x0 + 150, 232, f'KV cache: heads {heads}, same block IDs', 't-tick', 'middle'))
        b.append(arrow(360, 76, x0 + 150, 102))
        b.append(arrow(x0 + 150, 256, 360, 286))
    b += token(230, 288, 260, 'all-reduce: sum the 2 partial outputs', 'box-on', 32)
    b.append(text(360, 334, f'sum vs. unsplit layer: max difference {R["tp"]["attn_diff"]:.1e}', 't-tick', 'middle'))
    return svg(W, H, 'The input goes to both GPUs. Each computes attention for six heads and a partial output; '
               'an all-reduce adds the two partial outputs into the layer output.', b)


def arch_map():
    """'You are here': a redraw of the paper's Figure 4. The pieces this part explains use the box-on class."""
    W, H = 720, 330
    b = [text(20, 24, 'Where this part sits in vLLM (a redraw of the paper\'s Figure 4)', 't-title')]
    # requests
    b.append(box(20, 120, 110, 96, 'box', 10))
    b.append(text(75, 142, 'Requests', 't-tick t-strong', 'middle'))
    for i in range(3):
        b += token(34, 152 + i * 20, 82, f'request {i + 1}', 'box-ghost', 16, 't-tick', 4)
    b.append(arrow(132, 168, 160, 168))
    # scheduler with the KV cache manager
    b.append(box(162, 40, 290, 270, 'box', 12))
    b.append(text(307, 62, 'Scheduler', 't-tick t-strong', 'middle'))
    b.append(text(307, 80, 'who runs, who waits, who is preempted (§4.5)', 't-tick', 'middle'))
    b.append(box(178, 94, 258, 202, 'box-ghost', 10))
    b.append(text(307, 114, 'KV cache manager', 't-tick t-strong', 'middle'))
    for i, (lab, cls) in enumerate((('Block tables, ref counts (§4.4)', 'box-on'),
                                    ('GPU block allocator', 'box-1'),
                                    ('CPU block allocator (§4.5)', 'box-on'))):
        b += token(192, 126 + i * 56, 230, lab, cls, 40)
    b.append(arrow(454, 168, 486, 168))
    # workers
    for k, (lab, y) in enumerate((('Worker 0', 40), ('Worker N-1', 194))):
        b.append(box(488, y, 214, 116, 'box', 10))
        b.append(text(595, y + 22, lab, 't-tick t-strong', 'middle'))
        b += token(502, y + 34, 186, 'Cache engine', 'box-1', 30)
        b += token(502, y + 72, 186, 'Model shard (§4.6)', 'box-on', 30)
    b.append(text(595, 178, '. . .', 't-tick t-strong', 'middle'))
    b += [box(20, 300, 12, 12, 'box-on', 2), text(38, 310, 'explained in this part', 't-tick')]
    return svg(W, H, 'Requests enter the scheduler. Inside it, the KV cache manager holds the block tables with reference '
               'counts, a GPU block allocator and a CPU block allocator. The scheduler sends work to workers 0 to N-1, '
               'each with a cache engine and a model shard. The block tables, the CPU block allocator and the model '
               'shards are highlighted as the subject of this part.', b)


def beam_tree():
    """The tiny beam search example of the text as a tree: k = 2, three tokens a, b, c."""
    W, H = 720, 368
    b = [text(20, 24, 'Beam search with k = 2 and a vocabulary of three tokens (a, b, c)', 't-title')]
    b += token(20, 173, 90, 'prompt', 'box', 34)
    b.append(text(170, 50, 'step 1: 2 beams', 't-tick t-strong', 'middle'))
    b.append(text(420, 50, 'step 2: 6 candidates, keep the best 2', 't-tick t-strong', 'middle'))
    parents = (('a', -0.5, 110, ((-0.2, True), (-2.0, False), (-3.0, False))),
               ('b', -1.0, 236, ((-0.1, True), (-0.3, False), (-4.0, False))))
    for p, ps, py, kids in parents:
        b.append(arrow(110, 190, 128, py + 17))
        b += token(130, py, 100, f'"{p}"  {ps:+.1f}'.replace('-', '−'), 'box-on', 34)
        for j, (lp, keep) in enumerate(kids):
            ky = py - 46 + j * 46
            w = p + 'abc'[j]
            tot = ps + lp
            b.append(f'<line class="{"edge" if keep else "edge-dim"}" x1="230" y1="{py + 17}" x2="298" y2="{ky + 15}"/>')
            lab = f'"{w}"   {ps:+.1f} {lp:+.1f} = {tot:+.1f}'.replace('-', '−')
            b += token(300, ky, 240, lab, 'box-on' if keep else 'box-ghost', 30, 't-tick' if keep else 't-tick t-muted')
            b.append(text(552, ky + 20, 'kept' if keep else 'dropped', 't-tick t-strong' if keep else 't-tick t-muted'))
    b.append(text(20, 338, 'score of a beam = sum of the log-probabilities of its tokens; "ab" (−2.5) lost to "ba" (−1.1)', 't-tick'))
    b.append(text(20, 358, 'each beam keeps one child here; if "ab" had beaten "ba", beam "b" would die', 't-tick'))
    return svg(W, H, 'A tree. The prompt has two beams, a with score minus 0.5 and b with minus 1.0. Each has three '
               'children. The two best of the six, aa at minus 0.7 and ba at minus 1.1, are kept; the other four are dropped.', b)


def cow_before_after():
    """Reference counts and the copy of the paper's Figure 8, before and after sample A1 writes."""
    W, H = 720, 330
    b = [text(20, 24, 'Copy-on-write in the paper\'s Figure 8, block size 4', 't-title')]
    panels = (('1. after the prompt: both samples share', 0,
               (('A1', ('7', '1')), ('A2', ('7', '1'))),
               (('7', 2, 'Four score and seven', 'box-1'), ('1', 2, 'years ago our  _', 'box-1'))),
              ('2. A1 writes "fathers" into a copy', 370,
               (('A1', ('7', '3')), ('A2', ('7', '1'))),
               (('7', 2, 'Four score and seven', 'box-1'), ('1', 1, 'years ago our mothers', 'box-2'),
                ('3', 1, 'years ago our fathers', 'box-on'))))
    for title, x0, tables, blocks in panels:
        b.append(text(x0 + 10, 56, title, 't-tick t-strong'))
        b.append(text(x0 + 10, 82, 'block tables', 't-tick'))
        b.append(text(x0 + 170, 82, 'physical blocks (GPU memory)', 't-tick'))
        ypos = {}
        for i, (num, ref, words, cls) in enumerate(blocks):
            y = 94 + i * 64
            ypos[num] = y + 26
            b.append(box(x0 + 170, y, 170, 52, cls, 6))
            b.append(text(x0 + 180, y + 20, f'block {num}   ref count {ref}', 't-tick t-strong'))
            b.append(text(x0 + 180, y + 40, words, 't-tick'))
        for i, (name, ent) in enumerate(tables):
            y = 100 + i * 90
            b.append(text(x0 + 10, y + 20, name, 't-tick t-strong'))
            for j, p in enumerate(ent):
                cy = y + j * 34
                b += token(x0 + 40, cy, 80, f'{j} → {p}', 'box', 28)
                b.append(f'<line class="edge-dim" x1="{x0 + 120}" y1="{cy + 14}" x2="{x0 + 168}" y2="{ypos[p]:.0f}"/>')
    b.append(arrow(352, 170, 366, 170, True))
    b.append(text(20, 312, 'A1 got a copy (block 3) and block 1 fell from 2 to 1; A2 then writes "mothers" into block 1 with no copy.', 't-tick'))
    return svg(W, H, 'Left: block tables of A1 and A2 both map to physical blocks 7 and 1, each with reference count 2. '
               'Right: after A1 writes, A1 maps to blocks 7 and 3, a new copy; block 1 has reference count 1 and holds '
               'mothers, block 3 holds fathers; block 7 is still shared with count 2.', b)


def prefix_share():
    """A shared system prompt, before vLLM (each request stores it) and with vLLM (stored once, mapped by all)."""
    p = R['prefix']
    r1, r2 = p['requests']
    W, H = 720, 320
    b = [text(20, 24, f'A {p["len"]}-token shared prefix (the paper\'s Figure 10), two user requests', 't-title')]
    b.append(text(20, 56, 'Without sharing: every request computes and stores the prefix again', 't-tick t-strong'))
    for i, r in enumerate((r1, r2)):
        y = 68 + i * 40
        b += token(20, y, 360, 'prefix: instruction + 3 examples', 'box-ghost', 30)
        b += token(388, y, 150, r['task'], 'box-2', 30)
        b.append(text(548, y + 20, f'request {i + 1}', 't-tick'))
    b.append(text(20, 176, 'With sharing: the prefix is computed once and every request maps its blocks', 't-tick t-strong'))
    b += token(20, 190, 360, f'prefix, stored once ({p["blocks"]} blocks of 4)', 'box-on', 30)
    for i, r in enumerate((r1, r2)):
        y = 236 + i * 40
        b += token(388, y, 150, r['task'], 'box-2', 30)
        b.append(f'<line class="edge" x1="388" y1="{y + 15}" x2="300" y2="222" marker-end="url(#ah)"/>')
        b.append(text(548, y + 20, f'{r["n"]} tokens computed, not {r["n"] + p["len"]}', 't-tick'))
    return svg(W, H, 'Top: two requests, each with its own copy of the same long prefix followed by its task. '
               f'Bottom: one stored copy of the {p["len"]}-token prefix, and two short task inputs that point to it. '
               f'The requests compute {r1["n"]} and {r2["n"]} tokens instead of {r1["n"] + p["len"]} and {r2["n"] + p["len"]}.', b)


def preempt_timeline():
    """A schematic timeline: the request that came last is preempted, then swapped back or recomputed."""
    W, H = 720, 442
    L, cw = 110, 46
    b = [text(20, 24, 'One preemption, two ways to recover (a sketch, 12 steps)', 't-title')]
    for s in range(12):
        b.append(text(L + s * cw + cw / 2, 50, str(s + 1), 't-tick', 'middle'))
    b.append(text(20, 50, 'step', 't-tick'))

    def lane(y, name, segs):
        out = [text(20, y + 19, name, 't-tick')]
        for s0, s1, cls, lab in segs:
            out += token(L + s0 * cw + 1, y, (s1 - s0) * cw - 2, lab, cls, 28)
        return out
    for title, y0, back in (('Swapping', 70, ('box-2', 'blocks in CPU memory', 'box-3', 'swap in')),
                            ('Recomputation', 236, ('box-ghost', 'blocks thrown away', 'box-3', 'prompt pass'))):
        b.append(text(20, y0 + 6, title, 't-tick t-strong'))
        b += lane(y0 + 16, 'request 1', [(0, 7, 'box-1', 'running'), (7, 8, 'box', 'done')])
        b += lane(y0 + 52, 'request 2', [(0, 12, 'box-1', 'running')])
        b += lane(y0 + 88, 'request 3', [(0, 4, 'box-1', 'running'), (4, 8, back[0], back[1]),
                                         (8, 10, back[2], back[3]), (10, 12, 'box-1', 'running')])
        b.append(f'<line class="bad-line" x1="{L + 4 * cw}" y1="{y0 + 98}" x2="{L + 4 * cw}" y2="{y0 + 140}"/>')
        b.append(text(L + 4 * cw - 4, y0 + 152, 'preempted', 't-tick t-strong', 'end'))
    b.append(text(20, 414, 'Step 5: no block is free, so the request that came last loses all its blocks at once.', 't-tick'))
    b.append(text(20, 432, 'Step 8: request 1 ends and frees its blocks, so request 3 can come back.', 't-tick'))
    return svg(W, H, 'Two timelines of 12 steps with three requests. At step 5 memory is full and request 3, the last to '
               'arrive, is preempted. With swapping its blocks wait in CPU memory and are copied back in steps 9 and 10. With '
               'recomputation its blocks are thrown away and one prompt pass rebuilds them in steps 9 and 10. Request 1 ends at '
               'step 8 and frees the room.', b)


def model_sizes():
    """Why one model needs several GPUs: the weights of the paper's three models against one GPU (Table 1)."""
    items = [('OPT-13B', 26), ('OPT-66B', 132), ('OPT-175B', 346)]
    W, H = 720, 186
    bw, vmax, lw = 470, 360, 110
    b = hbars(20, 44, items, bw, row_h=34, cls='s1', label_w=lw, fmt=lambda v: f'{v} GB', vmax=vmax,
              title='Weights alone, in GB (the paper\'s Table 1), against the memory of one A100')
    x80 = 20 + lw + bw * 80 / vmax
    b.append(f'<line class="bad-line" x1="{x80:.1f}" y1="40" x2="{x80:.1f}" y2="150"/>')
    b.append(text(x80 + 4, 172, 'one A100: 80 GB at most', 't-tick'))
    b.append(text(20, 172, 'the paper uses 1, 4 and 8 GPUs', 't-tick'))
    return svg(W, H, 'Bars: OPT-13B has 26 GB of weights, OPT-66B 132 GB and OPT-175B 346 GB. A line at 80 GB marks one '
               'A100. Only the smallest model fits on one GPU.', b)


def heads_split():
    """The 12 heads of one attention layer split over 2 GPUs, and the all-reduce with the tiny example of the text."""
    W, H = 720, 360
    b = [text(20, 24, 'Splitting attention by heads, and the all-reduce that joins the halves', 't-title')]
    for g, x0 in ((0, 20), (1, 370)):
        b.append(box(x0, 40, 330, 190, 'box-1' if g == 0 else 'box-2', 10))
        b.append(text(x0 + 165, 62, f'GPU {g}', 't-tick t-strong', 'middle'))
        for h in range(6):
            b += token(x0 + 15 + h * 51, 74, 45, f'head {h + 6 * g}', 'box', 30)
        b.append(text(x0 + 165, 130, f'outputs of its 6 heads: h({g})', 't-tick', 'middle'))
        b.append(text(x0 + 165, 152, f'times its rows of the output matrix: Wo({g})', 't-tick', 'middle'))
        b += token(x0 + 65, 168, 200, f'partial output h({g}) Wo({g})', 'box', 30)
        b.append(text(x0 + 165, 220, 'tiny example: ' + ('1 × 3 = 3' if g == 0 else '2 × 4 = 8'), 't-tick t-strong', 'middle'))
        b.append(arrow(x0 + 165, 232, 360, 262))
    b += token(220, 264, 280, 'all-reduce: 3 + 8 = 11', 'box-on', 34)
    b.append(text(360, 320, 'afterwards both GPUs hold 11, the same as the unsplit layer (1 × 3 + 2 × 4)', 't-tick', 'middle'))
    b.append(text(360, 342, 'in our OPT-125M run: max difference 4.2e-07 against the library', 't-tick', 'middle'))
    return svg(W, H, 'Two GPU frames. GPU 0 holds heads 0 to 5, GPU 1 holds heads 6 to 11. Each multiplies its head '
               'outputs by its rows of the output matrix and gets a partial output: 3 and 8 in the tiny example. An '
               'all-reduce adds them to 11 and gives the sum to both GPUs.', b)


def fusion():
    """Kernel fusion: many small launches against one launch, with our timings of the block write."""
    k = R['kernels']
    W, H = 720, 250
    b = [text(20, 24, 'Kernel fusion: 256 small writes against one batched write (our laptop GPU)', 't-title')]
    b.append(text(20, 60, 'one launch per token', 't-tick t-strong'))
    for i in range(32):
        x = 180 + i * 16
        b.append(box(x, 46, 9, 22, 'box', 1))
        b.append(box(x + 9, 46, 4, 22, 'box-on', 1))
    b.append(text(180, 88, f'(32 of the 256 shown) each = fixed launch cost (grey) + tiny work (blue): {k["write_loop"]:.2f} ms', 't-tick'))
    b.append(text(20, 132, 'one fused launch', 't-tick t-strong'))
    b.append(box(180, 118, 9, 22, 'box', 1))
    b.append(box(189, 118, 128, 22, 'box-on', 1))
    b.append(text(180, 160, f'one fixed cost, then all 256 writes together: {k["write_fused"]:.3f} ms', 't-tick'))
    b.append(text(20, 206, 'The work is the same; the fused version pays the start-up cost once.', 't-tick'))
    b.append(text(20, 228, 'The bars are a sketch, not to scale; the times are from our run.', 't-tick t-muted'))
    return svg(W, H, f'Top: a row of many small launches, each with a fixed start cost and a little work, '
               f'{k["write_loop"]:.2f} ms in all. Bottom: one launch with one start cost and all the work, '
               f'{k["write_fused"]:.3f} ms.', b)



F['p4_map'] = arch_map()
F['p4_beamtree'] = beam_tree()
F['p4_cow'] = cow_before_after()
F['p4_prefix'] = prefix_share()
F['p4_preempt'] = preempt_timeline()
F['p4_gpus'] = model_sizes()
F['p4_heads'] = heads_split()
F['p4_fusion'] = fusion()
F['p4_fig8'] = fig8()
F['p4_fig9'] = fig9()
F['p4_blocks'] = blocks_over_steps()
F['p4_saving'] = saving()
F['p4_sched'] = schedule()
F['p4_swap'] = swap_chart()
F['p4_tp'] = tp()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part4.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
