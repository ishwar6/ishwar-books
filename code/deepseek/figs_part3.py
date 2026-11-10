"""Figures for Part 3 (training DeepSeek-V3). Writes results/figs_part3.json (key -> SVG).
CSS classes only, so every figure follows the dark, dim and light themes. Numbers come from
results/part3_numbers.json and results/part3_fp8.json (our scripts) or from the V3 report as cited."""
import json, math, os
from figlib import svg, text, box, arrow, esc

HERE = os.path.dirname(os.path.abspath(__file__))
N = json.load(open(os.path.join(HERE, 'results', 'part3_numbers.json')))
Q = json.load(open(os.path.join(HERE, 'results', 'part3_fp8.json')))
F = {}


def tok(x, y, w, s, cls='box', h=30, tcls='t-tick'):
    return [box(x, y, w, h, cls, 6), text(x + w / 2, y + h / 2 + 4.5, s, tcls, 'middle')]


def line(x1, y1, x2, y2, cls='edge'):
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'


def poly(pts, cls):
    return f'<polyline class="{cls}" points="' + ' '.join(f'{x:.1f},{y:.1f}' for x, y in pts) + '"/>'


# 1. the whole run ---------------------------------------------------------------------------
def timeline():
    W, H = 720, 300
    b = [text(20, 24, 'DeepSeek-V3, from random weights to a chat model', 't-title')]
    stages = [('Pre-training', '14.8T tokens', '4K-token sequences', '2,664K GPU hours', 'box-1'),
              ('Context, phase 1', f'{N["ctx"]["p1"] / 1e9:.1f}B tokens', '32K-token sequences', '1,000 steps', 'box-3'),
              ('Context, phase 2', f'{N["ctx"]["p2"] / 1e9:.1f}B tokens', '128K-token sequences', '1,000 steps', 'box-3'),
              ('Post-training', 'SFT + RL', 'Part 4', '5K GPU hours', 'box-2')]
    x = 20
    for i, (t, a, c, h, cls) in enumerate(stages):
        w = 160
        b.append(box(x, 44, w, 104, cls, 10))
        b.append(text(x + w / 2, 68, t, 't-note', 'middle'))
        b.append(text(x + w / 2, 92, a, 't-tick', 'middle'))
        b.append(text(x + w / 2, 112, c, 't-tick', 'middle'))
        b.append(text(x + w / 2, 134, h, 't-strong t-tick', 'middle'))
        if i < 3:
            b.append(arrow(x + w + 2, 96, x + w + 18, 96))
        x += w + 20
    b.append(f'<path class="drop" d="M200,154 V162 H540 V154"/>')
    b.append(text(370, 176, 'both phases together: 119K GPU hours', 't-tick t-strong', 'middle'))
    # share of GPU hours
    tot = 2788
    b.append(text(20, 194, 'Share of the 2,788K GPU hours', 't-label'))
    x0, ww = 20, 680
    parts = [(2664, 's1', 'pre-training 95.6%'), (119, 's3', 'context 4.3%'), (5, 's2', 'post 0.2%')]
    xx = x0
    for v, cls, lab in parts:
        w = ww * v / tot
        b.append(f'<g class="mark"><title>{esc(lab)}</title><rect class="{cls}" x="{xx:.1f}" y="204" width="{max(w, 2):.1f}" height="24" rx="3"/></g>')
        xx += w
    b.append(text(24, 246, 'pre-training: 2,664K (95.6%)', 't-tick'))
    b.append(text(700, 246, 'context 119K (4.3%), post 5K (0.2%)', 't-tick', 'end'))
    b.append(text(20, 276, 'GPU hours from V3 report Table 1 (p.5). Token counts of the two phases: our arithmetic from p.23.', 't-muted'))
    return svg(W, H, 'The four stages of training DeepSeek-V3 and their share of the 2,788K H800 GPU hours.', b)


# 2. fill-in-the-middle ----------------------------------------------------------------------
def fim():
    W, H = 720, 250
    b = [text(20, 24, 'Fill-in-the-middle: one document, cut in three and reordered (PSM)', 't-title')]
    b.append(text(20, 56, 'Original file', 't-label'))
    b += tok(20, 66, 230, 'def area(r):', 'box-1', 32, 't-mono')
    b += tok(254, 66, 230, 'return 3.14159 * r * r', 'box-2', 32, 't-mono')
    b += tok(488, 66, 212, '# end of file', 'box-3', 32, 't-mono')
    b.append(text(135, 116, 'prefix', 't-tick', 'middle'))
    b.append(text(369, 116, 'middle', 't-tick', 'middle'))
    b.append(text(594, 116, 'suffix', 't-tick', 'middle'))
    b.append(text(20, 150, 'What the model reads in training (10% of documents)', 't-label'))
    y = 160
    seq = [('<|fim_begin|>', 'box', 104), ('def area(r):', 'box-1', 104), ('<|fim_hole|>', 'box', 96),
           ('# end of file', 'box-3', 104), ('<|fim_end|>', 'box', 92), ('return 3.14...', 'box-2', 112),
           ('<|eos|>', 'box', 60)]
    x = 20
    for s, cls, w in seq:
        b += tok(x, y, w, s, cls, 32, 't-tick')
        x += w + 4
    b.append(text(20, 222, 'The middle comes last, so plain next-token prediction learns to write it from both sides.', 't-tick'))
    b.append(text(20, 242, 'Format and rate from V3 report §4.1 (p.22); our own example file.', 't-muted'))
    return svg(W, H, 'A small code file split into prefix, middle and suffix, then reordered as prefix, suffix, middle with special tokens.', b)


# 3. schedules -------------------------------------------------------------------------------
def schedule():
    W, H = 720, 340
    b = [text(20, 24, 'Learning rate and batch size during pre-training', 't-title')]
    # LR panel
    L, T, pw, ph = 70, 66, 360, 220
    X = lambda t: L + t / 14.8 * pw
    Y = lambda v: T + ph - v / 2.4e-4 * ph
    for v in [0, 0.5e-4, 1e-4, 1.5e-4, 2e-4]:
        b += [line(L, Y(v), L + pw, Y(v), 'grid'), text(L - 6, Y(v) + 4, f'{v * 1e4:.1f}', 't-tick', 'end')]
    for t in [0, 5, 10, 14.8]:
        b.append(text(X(t), T + ph + 18, f'{t:g}T', 't-tick', 'middle'))
    b.append(line(L, T + ph, L + pw, T + ph, 'axis'))
    b.append(poly([(X(t), Y(v)) for t, v in N['lr']], 'l1'))
    b.append(text(L - 50, T - 10, 'learning rate (x 10⁻⁴)', 't-tick'))
    b.append(text(L + pw / 2, T + ph + 38, 'tokens seen', 't-tick', 'middle'))
    b.append(text(X(5), Y(2.2e-4) - 8, 'constant 2.2 x 10⁻⁴', 't-tick', 'middle'))
    b.append(text(X(12.2), Y(1.4e-4), 'cosine', 't-tick', 'start'))
    b.append(text(X(14.8) - 30, Y(0.05e-4) - 14, '0.22, then 0.073', 't-tick', 'end'))
    # batch panel
    L2, pw2 = 500, 200
    X2 = lambda t: L2 + t / 600 * pw2
    Y2 = lambda v: T + ph - v / 16000 * ph
    for v in [0, 5000, 10000, 15360]:
        b += [line(L2, Y2(v), L2 + pw2, Y2(v), 'grid'), text(L2 - 6, Y2(v) + 4, f'{v:,}', 't-tick', 'end')]
    for t in [0, 200, 469, 600]:
        b.append(text(X2(t), T + ph + 18, f'{t}B', 't-tick', 'middle'))
    b.append(line(L2, T + ph, L2 + pw2, T + ph, 'axis'))
    b.append(poly([(X2(t), Y2(v)) for t, v in N['batch']], 'l2'))
    b.append(text(L2 - 50, T - 10, 'sequences per step', 't-tick'))
    b.append(text(L2 + pw2 / 2, T + ph + 38, 'tokens seen (first 600B)', 't-tick', 'middle'))
    b.append(text(X2(280), Y2(9000), 'ramp shape not given;', 't-muted', 'start'))
    b.append(text(X2(280), Y2(9000) + 16, 'drawn straight', 't-muted', 'start'))
    return svg(W, H, 'Left: the learning rate holds at 2.2e-4 until 10T tokens, falls along a cosine to 2.2e-5 by 14.3T, then steps to 7.3e-6. '
               'Right: the batch grows from 3,072 to 15,360 sequences over the first 469B tokens.', b)


# 4. one H800 node ---------------------------------------------------------------------------
def node():
    W, H = 720, 330
    b = [text(20, 24, 'One node of the cluster (256 such nodes = 2,048 GPUs)', 't-title')]
    b.append(box(20, 44, 560, 236, 'plane', 12))
    b.append(text(36, 66, 'Node: 8 x H800', 't-label'))
    b.append(box(60, 150, 480, 34, 'box-on', 8))
    b.append(text(300, 172, 'NVSwitch: any GPU to any GPU in the node', 't-tick', 'middle'))
    for i in range(8):
        x = 46 + i * 66
        b += tok(x, 84, 58, f'GPU {i}', 'box-1', 36, 't-tick t-strong')
        b.append(line(x + 29, 120, x + 29, 150))
        b += tok(x, 220, 58, f'NIC {i}', 'box-3', 30, 't-tick')
        b.append(line(x + 29, 120, x + 29, 120))
        b.append(f'<path class="drop" d="M{x + 29},{120} C{x + 2},{160} {x + 2},{190} {x + 29},{220}"/>')
    b.append(box(600, 120, 100, 110, 'box-3', 10))
    b.append(text(650, 158, 'InfiniBand', 't-note', 'middle'))
    b.append(text(650, 178, 'to the other', 't-tick', 'middle'))
    b.append(text(650, 196, '255 nodes', 't-tick', 'middle'))
    b.append(arrow(582, 235, 600, 210))
    b.append(text(20, 300, 'NVLink in the node: 400 GB/s per GPU on the H800 (900 GB/s on the H100); about 160 GB/s usable one way.', 't-tick'))
    b.append(text(20, 320, 'InfiniBand out of the node: one 400 Gb/s NIC per GPU, about 50 GB/s.  Sources: ISCA paper p.7, V3 report p.13.', 't-tick'))
    return svg(W, H, 'An H800 node: eight GPUs joined by NVSwitch, and eight InfiniBand cards, one per GPU, that link the node to the rest of the cluster.', b)


# 5. the parallelism plan ----------------------------------------------------------------------
def parallel():
    W, H = 720, 400
    b = [text(20, 24, 'How 2,048 GPUs share one model', 't-title')]
    b.append(text(20, 52, '16-way pipeline: the 61 layers are cut into 16 stages, in order', 't-label'))
    for i in range(16):
        x = 20 + i * 42
        b += tok(x, 62, 38, f'S{i}', 'box-on' if i == 5 else 'box-1', 30, 't-tick')
    b.append(text(20, 112, 'each stage: about 4 layers, run by 128 GPUs = 16 nodes', 't-tick'))
    b.append(f'<path class="drop" d="M{20 + 5 * 42},{94} L{20},{132} M{20 + 5 * 42 + 38},{94} L{700},{132}"/>')
    b.append(box(20, 132, 680, 200, 'plane', 12))
    for g in range(2):
        gx = 36 + g * 334
        b.append(box(gx, 146, 318, 172, 'box-ghost', 10))
        b.append(text(gx + 159, 166, f'expert-parallel group {g + 1}: 8 nodes, 64 GPUs', 't-tick t-strong', 'middle'))
        for n in range(8):
            nx, ny = gx + 12 + (n % 4) * 76, 180 + (n // 4) * 66
            b.append(box(nx, ny, 70, 58, 'box', 6))
            b.append(text(nx + 35, ny + 14, f'node {g * 8 + n + 1}', 't-muted', 'middle'))
            for k in range(8):
                b.append(box(nx + 5 + (k % 4) * 15.5, ny + 22 + (k // 4) * 16, 13, 13, 'box-1', 2))
    b.append(text(20, 352, 'Inside a group, the 256 routed experts of each layer are split 4 per GPU (64 x 4 = 256).', 't-tick'))
    b.append(text(20, 372, 'The two groups hold the same experts and see different data; attention weights sit on all 128 GPUs.', 't-tick'))
    b.append(text(20, 392, 'Data parallelism with ZeRO-1 splits the optimizer state over the copies. (Stage layout: our reading.)', 't-muted'))
    return svg(W, H, 'The 2,048 GPUs as 16 pipeline stages; one stage of 128 GPUs is shown as two expert-parallel groups of 8 nodes and 64 GPUs each.', b)


# 6. pipeline bubbles (toy simulation) ---------------------------------------------------------
def bubbles():
    W, H = 720, 350
    sim = N['sim']
    S, span = 4, sim['GPipe']['span']
    L, u = 90, 600 / span
    b = [text(20, 24, f'Toy pipeline: 4 stages, 8 micro-batches, forward = 1, backward = 2 time units', 't-title')]
    for p, name in enumerate(['GPipe', '1F1B']):
        y0 = 50 + p * 135
        b.append(text(20, y0 + 6, f'{name}: all forwards, then all backwards' if name == 'GPipe' else
                      '1F1B: one forward, one backward, in turn', 't-label'))
        for d in range(S):
            y = y0 + 16 + d * 26
            b.append(text(L - 8, y + 16, f'GPU {d}', 't-tick', 'end'))
            b.append(box(L, y, span * u, 22, 'box-ghost', 2))
            for kind, m, s, e in sim[name]['busy'][d]:
                cls = 'box-1' if kind == 'F' else 'box-2'
                b.append(f'<g class="mark"><title>{kind}{m}: {s:g} to {e:g}</title>{box(L + s * u, y, (e - s) * u, 22, cls, 2)}</g>')
                b.append(text(L + (s + e) / 2 * u, y + 15, f'{kind}{m}', 't-cell', 'middle'))
    b.append(text(20, 338, f'Dashed gaps = bubbles: each GPU waits {sim["GPipe"]["idle"]:g} of {span:g} units ({sim["GPipe"]["idle"] / span:.0%}) in both schedules. '
                  'Blue: forward. Orange: backward.', 't-tick'))
    return svg(W, H, 'Two toy pipeline schedules from our simulation. Both leave each GPU idle for 9 of 33 time units; 1F1B starts backward passes earlier, so it holds fewer activations.', b)


# 7. overlap -------------------------------------------------------------------------------------
def overlap():
    W, H = 752, 310
    u = 48
    b = [text(20, 24, 'Two forward/backward pairs when sending takes as long as computing (toy times)', 't-title')]

    def row(y, label, items):
        out = [text(104, y + 19, label, 't-tick', 'end')]
        for s0, d, lab, cls in items:
            out += tok(112 + s0 * u, y, d * u - 2, lab, cls, 28, 't-tick')
        return out
    b.append(text(20, 56, 'One after the other', 't-label'))
    b += row(66, 'compute', [(0, 1, 'F1', 'box-1'), (2, 2, 'B1', 'box-2'), (6, 1, 'F2', 'box-1'), (8, 2, 'B2', 'box-2')])
    b += row(98, 'network', [(1, 1, 'F1', 'box-3'), (4, 2, 'B1', 'box-3'), (7, 1, 'F2', 'box-3'), (10, 2, 'B2', 'box-3')])
    b.append(text(112 + 12 * u + 8, 104, '12 units', 't-val'))
    b.append(text(20, 156, 'Overlapped: the network sends one chunk while the GPU computes the next', 't-label'))
    b += row(166, 'compute', [(0, 1, 'F1', 'box-1'), (1, 2, 'B1', 'box-2'), (3, 1, 'F2', 'box-1'), (4, 2, 'B2', 'box-2')])
    b += row(198, 'network', [(1, 1, 'F1', 'box-3'), (3, 2, 'B1', 'box-3'), (5, 1, 'F2', 'box-3'), (6, 2, 'B2', 'box-3')])
    b.append(text(112 + 8 * u + 8, 204, '8 units', 't-val'))
    b.append(text(20, 256, 'Blue and orange: computing a forward (F) or backward (B) chunk. Green: sending its results to other GPUs.', 't-tick'))
    b.append(text(20, 276, 'With many pairs in a row, each pair costs 3 units instead of 6: the network hides under the compute.', 't-tick'))
    b.append(text(20, 298, 'Toy numbers, chosen to match the 1:1 ratio the V3 report gives. Not a measurement.', 't-muted'))
    return svg(W, H, 'Toy timing of two forward and backward pairs: run one after the other they take 12 units; overlapped, with sending hidden under computing, they take 8.', b)


# 8. two directions -----------------------------------------------------------------------------
def dualdir():
    W, H = 720, 280
    b = [text(20, 24, 'DualPipe feeds micro-batches from both ends (sketch with 4 GPUs)', 't-title')]
    for d in range(4):
        x = 150 + d * 120
        b.append(box(x, 70, 104, 120, 'box', 10))
        b.append(text(x + 52, 92, f'GPU {d}', 't-note', 'middle'))
        b += tok(x + 8, 104, 88, f'stage {d}', 'box-1', 30, 't-tick')
        b += tok(x + 8, 146, 88, f'stage {3 - d}', 'box-2', 30, 't-tick')
    b.append(arrow(40, 119, 146, 119))
    b.append(text(40, 110, 'batches in →', 't-tick'))
    b.append(arrow(690, 161, 634, 161))
    b.append(text(640, 186, '← batches in', 't-tick'))
    for d in range(3):
        b.append(arrow(150 + d * 120 + 104, 119, 150 + (d + 1) * 120, 119))
        b.append(arrow(150 + (d + 1) * 120, 161, 150 + d * 120 + 104, 161))
    b.append(text(20, 222, 'Blue: the model run left to right. Orange: a second copy run right to left.', 't-tick'))
    b.append(text(20, 242, 'Each GPU holds two stages (2 x parameters), but always has work from one side or the other.', 't-tick'))
    b.append(text(20, 264, 'GPU 0 holds the first and the last stage, so embedding and output head can live together.', 't-muted'))
    return svg(W, H, 'A sketch of the two directions of DualPipe on four GPUs: GPU 0 holds stage 0 of one copy and stage 3 of the other.', b)


# 9. route of one token ------------------------------------------------------------------------
def route():
    W, H = 720, 320
    b = [text(20, 24, 'How one token reaches experts on another node', 't-title')]
    for n, (x, lab) in enumerate([(20, 'node A (token is here)'), (380, 'node B (3 of its experts)')]):
        b.append(box(x, 64, 320, 190, 'plane', 12))
        b.append(text(x + 16, 86, lab, 't-label'))
        for i in range(8):
            gx, gy = x + 16 + (i % 4) * 76, 102 + (i // 4) * 74
            hot = (n == 0 and i == 3) or (n == 1 and i in (3, 5, 6, 7))
            cls = 'box-on' if (n == 0 and i == 3) or (n == 1 and i == 3) else ('box-2' if hot else 'box')
            b += tok(gx, gy, 66, f'GPU {i}', cls, 52, 't-tick')
    # IB hop: A3 -> B3
    b.append('<path class="path" d="M287,102 C287,44 647,44 647,102"/>')
    b.append(text(467, 44, 'InfiniBand, once', 't-tick t-strong', 'middle'))
    # NVLink hops inside B
    bx = lambda i: 380 + 16 + (i % 4) * 76 + 33
    by = lambda i: 102 + (i // 4) * 74 + 26
    for t in (5, 6, 7):
        b.append(arrow(bx(3), by(3) + 26, bx(t), by(t) - 26 if t >= 4 else by(t)))
    b.append(text(560, 248, 'NVLink inside the node', 't-tick', 'middle'))
    b.append(text(20, 280, 'The token crosses InfiniBand once per node, to the GPU with the same number (GPU 3 → GPU 3),', 't-tick'))
    b.append(text(20, 300, 'then NVLink copies it to the GPUs that hold its experts. At most 4 nodes per token. (V3 report p.13)', 't-tick'))
    return svg(W, H, 'A token on GPU 3 of node A is sent once over InfiniBand to GPU 3 of node B, then forwarded over NVLink to GPUs 5, 6 and 7, which hold its experts.', b)


# 10. number formats --------------------------------------------------------------------------
def formats():
    W, H = 720, 360
    b = [text(20, 24, 'Bits of each number format: sign, exponent (range), mantissa (precision)', 't-title')]
    rows = [('FP32', 1, 8, 23, 'about 3.4 x 10³⁸'), ('BF16', 1, 8, 7, 'about 3.4 x 10³⁸'),
            ('FP8 E5M2', 1, 5, 2, '57,344'), ('FP8 E4M3', 1, 4, 3, '448')]
    cw = 15
    for r, (name, s, e, m, mx) in enumerate(rows):
        y = 44 + r * 64
        b.append(text(20, y + 20, name, 't-note'))
        x = 110
        for k in range(s + e + m):
            cls = 'box-2' if k < s else ('box-1' if k < s + e else 'box-3')
            b.append(box(x + k * cw, y + 4, cw - 2, 24, cls, 2))
        b.append(text(110, y + 46, f'{s + e + m} bits ({s} sign, {e} exponent, {m} mantissa); largest number {mx}', 't-tick'))
    b.append(box(110, 312, 13, 13, 'box-2', 2)); b.append(text(128, 323, 'sign', 't-tick'))
    b.append(box(180, 312, 13, 13, 'box-1', 2)); b.append(text(198, 323, 'exponent: how big', 't-tick'))
    b.append(box(330, 312, 13, 13, 'box-3', 2)); b.append(text(348, 323, 'mantissa: how exact', 't-tick'))
    b.append(text(20, 350, 'FP8 E4M3 has only 8 steps between 1 and 2; FP32 has about 8 million.', 't-muted'))
    return svg(W, H, 'Bit layouts of FP32, BF16, FP8 E5M2 and FP8 E4M3, with the largest number each can hold.', b)


# 11. scaling groups ---------------------------------------------------------------------------
def scaling():
    W, H = 720, 290
    b = [text(20, 24, 'Who shares a scale? One outlier (red) in an 8 x 8 corner of a tensor', 't-title')]
    titles = ['one scale for the tensor', '1 x 128 tiles (activations)', '128 x 128 blocks (weights)']
    cell = 22
    for p in range(3):
        x0, y0 = 20 + p * 240, 50
        b.append(text(x0, y0 + 4, titles[p], 't-label'))
        for i in range(8):
            for j in range(8):
                if p == 0:
                    cls = 'box-1'
                elif p == 1:
                    cls = 'box-1' if (i, j // 4) != (2, 1) else 'box-2'
                else:
                    cls = 'box-1' if (i // 4, j // 4) != (0, 1) else 'box-2'
                if p == 0:
                    cls = 'box-2'
                b.append(box(x0 + j * cell, y0 + 14 + i * cell, cell - 2, cell - 2, cls, 2))
        b.append(f'<rect class="waste" x="{x0 + 5 * cell:.1f}" y="{y0 + 14 + 2 * cell:.1f}" width="{cell - 2}" height="{cell - 2}" rx="2"/>')
        if p == 1:
            for i in range(9):
                b.append(line(x0 - 1, y0 + 13 + i * cell, x0 + 8 * cell - 1, y0 + 13 + i * cell, 'axis'))
            b.append(line(x0 + 4 * cell - 1, y0 + 13, x0 + 4 * cell - 1, y0 + 13 + 8 * cell, 'axis'))
        if p == 2:
            b.append(line(x0 - 1, y0 + 13 + 4 * cell, x0 + 8 * cell - 1, y0 + 13 + 4 * cell, 'axis'))
            b.append(line(x0 + 4 * cell - 1, y0 + 13, x0 + 4 * cell - 1, y0 + 13 + 8 * cell, 'axis'))
    b.append(text(20, 276, 'Orange = numbers whose scale the outlier sets. A group here is 4 cells wide; in V3 it is 128 numbers.', 't-tick'))
    return svg(W, H, 'Three ways to group numbers for FP8 scaling. With one scale, the outlier sets the scale of every number; with tiles or blocks it sets the scale of only its own group.', b)


# 12. our FP8 experiment ------------------------------------------------------------------------
def fp8err():
    W, H = 720, 360
    b = [text(20, 24, 'Our FP8 test on random data (CPU, PyTorch float8_e4m3fn)', 't-title')]
    series = [('tensor', 'l2', 's2', 'one scale'), ('tile', 'l1', 's1', '1x128 tiles'), ('block', 'l3', 's3', '128x128 blocks')]
    # left: outlier sweep
    L, T, pw, ph = 70, 84, 240, 190
    xs = [1, 2, 3, 4, 5]   # positions for outlier 0, 1e2, 1e3, 1e4, 1e5
    labels = ['none', '100', '1K', '10K', '100K']
    X = lambda i: L + (i - 1) / 4 * pw
    Y = lambda v: T + ph - v / 0.35 * ph
    b.append(text(L - 50, T - 18, 'A: one outlier in 128 x 1,024', 't-label'))
    b.append(text(L - 50, T - 4, 'error of the other numbers', 't-tick'))
    for v in [0, 0.1, 0.2, 0.3]:
        b += [line(L, Y(v), L + pw, Y(v), 'grid'), text(L - 6, Y(v) + 4, f'{v:.0%}', 't-tick', 'end')]
    for i, lab in zip(xs, labels):
        b.append(text(X(i), T + ph + 18, lab, 't-tick', 'middle'))
    b.append(text(L + pw / 2, T + ph + 38, 'size of the outlier', 't-tick', 'middle'))
    for key, lc, dc, name in series:
        vals = [r[key]['err'] for r in Q['outlier_sweep']]
        b.append(poly([(X(i), Y(v)) for i, v in zip(xs, vals)], lc))
        for i, v in zip(xs, vals):
            b.append(f'<g class="mark"><title>{name}, outlier {labels[i - 1]}: {v:.2%}</title><circle class="{dc} ring" cx="{X(i):.1f}" cy="{Y(v):.1f}" r="4.5"/></g>')
    # right: imbalanced rows
    L2 = 440
    X2 = lambda s: L2 + s / 8 * pw
    b.append(text(L2 - 50, T - 18, 'B: rows differ by up to 10^S', 't-label'))
    b.append(text(L2 - 50, T - 4, 'mean error per row', 't-tick'))
    for v in [0, 0.1, 0.2, 0.3]:
        b += [line(L2, Y(v), L2 + pw, Y(v), 'grid'), text(L2 - 6, Y(v) + 4, f'{v:.0%}', 't-tick', 'end')]
    for s in [0, 2, 4, 6, 8]:
        b.append(text(X2(s), T + ph + 18, f'S = {s}', 't-tick', 'middle'))
    for key, lc, dc, name in series:
        vals = [(r['S'], r[key]['err']) for r in Q['imbalanced']]
        b.append(poly([(X2(s), Y(v)) for s, v in vals], lc))
        for s, v in vals:
            b.append(f'<g class="mark"><title>{name}, S = {s}: {v:.2%}</title><circle class="{dc} ring" cx="{X2(s):.1f}" cy="{Y(v):.1f}" r="4.5"/></g>')
    lx = 70
    for key, lc, dc, name in series:
        b.append(f'<rect class="{dc}" x="{lx}" y="{H - 22}" width="12" height="12" rx="2"/>')
        b.append(text(lx + 18, H - 12, name, 't-tick'))
        lx += 160
    return svg(W, H, 'Error of FP8 quantization in our test. A: with one scale, error jumps only when the outlier reaches 100,000. '
               'B: when rows differ by a million times or more, one scale and 128x128 blocks fail, 1x128 tiles stay at 2.6%.', b)


# 13. accumulation ----------------------------------------------------------------------------
def accum():
    W, H = 720, 210
    a = Q['accumulation']
    keys = list(a)
    b = [text(20, 24, f'SIMULATION: adding up {Q["K"]:,} products with a 14-bit accumulator', 't-title')]
    vmax = max(v['max'] for v in a.values())
    for i, (k, lab, cls) in enumerate([(keys[0], 'narrow sum of all 4,096', 's2'), (keys[1], 'narrow for 128, then FP32', 's1')]):
        y = 50 + i * 60
        b.append(text(20, y + 16, lab, 't-tick t-strong'))
        w = max(3, 260 * a[k]['max'] / vmax)
        b.append(f'<g class="mark"><title>{esc(lab)}: max {a[k]["max"]:.4%}</title><rect class="{cls}" x="220" y="{y}" width="{w:.1f}" height="22" rx="3"/></g>')
        b.append(text(220 + w + 8, y + 16, f'worst {a[k]["max"]:.3%}   mean {a[k]["mean"]:.4%}', 't-val'))
    b.append(text(20, 176, 'Error = |result - exact sum| / (sum of |terms|), over 4,096 dot products. A model of the idea,', 't-tick'))
    b.append(text(20, 196, 'not the H800 circuit: the V3 report measured "nearly 2%" worst-case on the real Tensor Cores.', 't-tick'))
    return svg(W, H, f'Simulated accumulation error: {a[keys[0]]["max"]:.2%} worst case when all 4,096 products go through the narrow accumulator, '
               f'{a[keys[1]]["max"]:.4%} when the partial sum moves to FP32 every 128 products.', b)


# 14. what the cost includes -------------------------------------------------------------------
def cost():
    W, H = 720, 300
    b = [text(20, 24, 'What the $5.576M covers, and what it does not', 't-title')]
    b.append(box(20, 44, 330, 230, 'box-1', 12))
    b.append(text(185, 68, 'Counted (V3 report Table 1)', 't-note', 'middle'))
    for i, s in enumerate(['the final pre-training run: 2,664K h', 'context extension: 119K h', 'post-training of V3: 5K h',
                           'priced at an assumed $2 per', 'H800 GPU hour (a rental rate)']):
        b.append(text(36, 98 + i * 26, s, 't-tick'))
    b.append(text(185, 258, '= 2,788K GPU hours', 't-strong t-tick', 'middle'))
    b.append(box(370, 44, 330, 230, 'box-ghost', 12))
    b.append(text(535, 68, 'Not counted', 't-note', 'middle'))
    for i, s in enumerate(['research and ablation runs (stated)', 'buying the GPUs and the buildings',
                           'power, network, storage, staff', 'collecting and cleaning the data',
                           'the internal R1 model that made', "V3's reasoning training data",
                           'failed or restarted runs, if any']):
        b.append(text(386, 98 + i * 24, s, 't-tick'))
    b.append(text(20, 294, 'Left: what the paper states. Right: the paper names the first item; the rest follow from what a rental-price total is.', 't-muted'))
    return svg(W, H, 'The training cost counts only GPU hours of the final run at a rental price; research, hardware, staff and data work are outside it.', b)


for k, f in [('timeline', timeline), ('fim', fim), ('schedule', schedule), ('node', node), ('parallel', parallel),
             ('bubbles', bubbles), ('overlap', overlap), ('dualdir', dualdir), ('route', route), ('formats', formats),
             ('scaling', scaling), ('fp8err', fp8err), ('accum', accum), ('cost', cost)]:
    F[k] = f()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part3.json'), 'w'))
print(f'figs_part3: {len(F)} figures')
