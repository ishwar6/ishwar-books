"""Figures for Part 9 (DeepInfer). Writes results/figs_part9.json (key -> SVG). CSS classes only."""
import json, os
from figlib import svg, text, box, arrow, esc
from common import RESULTS

KV = json.load(open(os.path.join(RESULTS, 'part9_kv_calc.json')))
ST = json.load(open(os.path.join(RESULTS, 'part9_repo_stats.json')))
TY = json.load(open(os.path.join(RESULTS, 'part9_toys.json')))
FIGS = {}
W = 760


def label_box(x, y, w, h, title, sub=None, cls='box', tcls='t-node'):
    b = [box(x, y, w, h, cls)]
    if sub:
        b.append(text(x + w / 2, y + h / 2 - 3, title, tcls, 'middle'))
        b.append(text(x + w / 2, y + h / 2 + 14, sub, 't-muted', 'middle'))
    else:
        b.append(text(x + w / 2, y + h / 2 + 5, title, tcls, 'middle'))
    return b


# 1. timeline -----------------------------------------------------------------------------------------------
events = [('Feb 2025', 'Open Source Week', 'kernel libraries'),
          ('Mar 2025', 'Day 6 write-up', 'V3/R1 serving'),
          ('Apr 2025', 'Path to Open-Sourcing', 'no full engine'),
          ('Sep 2026', 'V4.1-Flash', 'new kernels'),
          ('Oct 2026', 'DeepInfer', 'the engine itself')]
b = [text(20, 26, 'From "we will upstream pieces" to "here is the engine"', 't-title')]
y0 = 84
b.append(f'<line class="axis" x1="40" y1="{y0}" x2="{W - 40}" y2="{y0}"/>')
for i, (d, l1, l2) in enumerate(events):
    x = 90 + i * (W - 180) / (len(events) - 1)
    cls = 'box-on' if i == len(events) - 1 else ('box-2' if i == 2 else 'box-1')
    b.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y0}" r="8"/>')
    b.append(text(x, y0 - 16, d, 't-title', 'middle'))
    b.append(text(x, y0 + 30, l1, 't-note', 'middle'))
    b.append(text(x, y0 + 48, l2, 't-muted', 'middle'))
FIGS['timeline'] = svg(W, 156, 'Timeline from February 2025 to October 2026: kernel libraries, the serving write-up, the April 2025 decision, V4.1-Flash, and the DeepInfer release.', b)

# 2. you are here: production system, with what DeepInfer contains --------------------------------------------
b = [text(20, 26, 'DeepSeek\'s V4.1-Flash serving system, and the part that is open', 't-title')]
b += label_box(270, 44, 220, 40, 'DeepSeek Recipe', 'JSON <-> tokens (separate repo)', 'box-2')
b += label_box(270, 110, 220, 40, 'Entrypoint', 'prefix-hash lookup (not open)', 'box-ghost')
b += label_box(20, 180, 210, 40, 'Vision Encoder LB', None, 'box-ghost')
b += label_box(275, 180, 210, 40, 'Prefill LB', None, 'box-ghost')
b += label_box(530, 180, 210, 40, 'Decode LB', None, 'box-ghost')
b += label_box(20, 250, 210, 84, 'Vision Encoder', 'service (1 GPU each)', 'box-1')
for x0, t, sub in ((275, 'Prefill service', 'EP2-8 / DP2-8'), (530, 'Decode service', 'EP64-128 / DP64-128')):
    b.append(box(x0, 250, 210, 84, 'box-on'))
    b.append(text(x0 + 105, 272, t, 't-node', 'middle'))
    b.append(text(x0 + 105, 290, sub, 't-muted', 'middle'))
    b.append(box(x0 + 12, 302, 186, 22, 'box-2', 4))
    b.append(text(x0 + 105, 318, 'DeepInfer engine (open)', 't-tick', 'middle'))
b += label_box(140, 372, 480, 40, 'KV cache store on 3FS (SSD cluster)', 'not open', 'box-ghost')
for x1, y1, x2, y2 in [(380, 84, 380, 108), (380, 150, 380, 178), (380, 220, 380, 248), (125, 220, 125, 248),
                       (635, 220, 635, 248), (230, 292, 273, 292), (485, 292, 528, 292), (380, 370, 380, 336),
                       (635, 334, 560, 370)]:
    b.append(arrow(x1, y1, x2, y2))
b.append(f'<path class="edge" d="M270,130 H125 V178" marker-end="url(#ah)"/>')
b.append(f'<path class="edge" d="M490,130 H635 V178" marker-end="url(#ah)"/>')
b.append(text(20, 436, 'Bright boxes run the open engine. Grey boxes are described in DeepSeek\'s Q&A but not released.', 't-muted'))
FIGS['youarehere'] = svg(W, 448, 'Architecture of DeepSeek\'s serving system: Recipe, Entrypoint, three load balancers, vision encoder, prefill and decode services and a 3FS KV store; only the engine inside prefill and decode is open.', b)

# 3. repository size by component ----------------------------------------------------------------------------
rows = [(k.replace('deepinfer/model_executor/', 'model_executor/').replace('deepinfer/', '').replace('(C++ scheduler, block manager, input prep)', '(C++ engine core)'), sum(v.values()))
        for k, v in ST['by_component'].items()][:10]
mx = max(v for _, v in rows)
b = [text(20, 24, f'DeepInfer source by component ({ST["total_lines"]:,} non-blank lines, vendor/ excluded)', 't-title')]
for i, (name, v) in enumerate(rows):
    y = 44 + i * 30
    cls = 's1' if 'tilelang' in name else ('s2' if name.startswith('csrc') else 's3')
    b.append(text(330, y + 16, name[:52], 't-tick', 'end'))
    b.append(f'<g class="mark"><title>{esc(name)}: {v:,} lines</title><rect class="{cls}" x="340" y="{y}" width="{max(2, v / mx * 330):.1f}" height="20" rx="4"/></g>')
    b.append(text(340 + v / mx * 330 + 8, y + 15, f'{v:,}', 't-val'))
b.append(text(20, 44 + len(rows) * 30 + 16, 'Python 89.7%, C++ 9.7%. Blue: TileLang kernels. Orange: the C++ engine core.', 't-muted'))
FIGS['repo'] = svg(W, 44 + len(rows) * 30 + 30, 'Bar chart of DeepInfer lines of code per component; TileLang kernels are the largest part at about 21,000 lines.', b)

# 4. one request through the engine ---------------------------------------------------------------------------
steps = [('LLM.generate()', 'entrypoints/llm.py', 'box-1'),
         ('LLMEngine.add_request()', 'tokens -> Sequence', 'box-1'),
         ('Scheduler.schedule()', 'C++: prefill OR decode', 'box-on'),
         ('Worker.execute_model_step()', 'forward on GPU', 'box-2'),
         ('sampler + DSpark draft', 'TileLang fused sampler', 'box-2'),
         ('process_outputs()', 'C++: accept tokens, stop?', 'box-on')]
b = [text(20, 24, 'One request through one DeepInfer engine (one process per data-parallel rank)', 't-title')]
bw, bh = 220, 52
pos = [(30, 50), (270, 50), (510, 50), (510, 150), (270, 150), (30, 150)]
for (t, s, c), (x, y) in zip(steps, pos):
    b += label_box(x, y, bw, bh, t, s, c)
b.append(arrow(250, 76, 268, 76)); b.append(arrow(490, 76, 508, 76)); b.append(arrow(620, 102, 620, 148))
b.append(arrow(510, 176, 492, 176)); b.append(arrow(270, 176, 252, 176))
b.append(f'<path class="path" d="M140,202 V236 H748 V76 H734" marker-end="url(#ah-on)"/>')
b.append(text(380, 254, 'loop: one step per call to LLMEngine.step(), until every rank is idle', 't-muted', 'middle'))
FIGS['steploop'] = svg(W, 266, 'Flow of one request: LLM.generate, add_request, the C++ scheduler, the worker forward pass, sampling with DSpark drafts, and C++ output processing, then back to the scheduler.', b)

# 5. the step vote across DP ranks ----------------------------------------------------------------------------
b = [text(20, 24, 'Every rank runs the same step type: decode, or (after a vote) prefill', 't-title')]
ranks = ['rank 0', 'rank 1', 'rank 2']
pattern = ['D'] * 6 + ['P'] + ['D'] * 5
real = {0: True, 1: False, 2: True}
cw = 50
for r, name in enumerate(ranks):
    y = 50 + r * 44
    b.append(text(70, y + 22, name, 't-tick', 'end'))
    for i, p in enumerate(pattern):
        x = 80 + i * cw
        if p == 'D':
            b.append(box(x, y, cw - 6, 32, 'box-1', 4)); b.append(text(x + (cw - 6) / 2, y + 21, 'D', 't-note', 'middle'))
        else:
            cls = 'box-on' if real[r] else 'box-ghost'
            b.append(box(x, y, cw - 6, 32, cls, 4)); b.append(text(x + (cw - 6) / 2, y + 21, 'P' if real[r] else 'P0', 't-note', 'middle'))
x = 80 + 6 * cw - 3
b.append(f'<line class="edge-on" x1="{x:.1f}" y1="40" x2="{x:.1f}" y2="186"/>')
b.append(text(x, 204, 'vote point (every 70 or 100 decode steps by default)', 't-muted', 'middle'))
b.append(text(20, 228, 'D = decode step. P = prefill step. P0 = an empty prefill step: rank 1 had nothing to prefill but must keep in step.', 't-muted'))
FIGS['vote'] = svg(W, 240, 'Three data-parallel ranks run decode steps; at a vote point all switch to prefill together, and a rank with no waiting prompt runs an empty prefill step.', b)

# 6. zero-overhead scheduling ---------------------------------------------------------------------------------
b = [text(20, 24, 'Asynchronous ("zero-overhead") scheduling hides the CPU work', 't-title')]


def lane(y, label, segs):
    out = [text(110, y + 20, label, 't-tick', 'end')]
    for x0, w, cls, t in segs:
        out.append(box(120 + x0, y, w - 3, 30, cls, 4))
        if t:
            out.append(text(120 + x0 + (w - 3) / 2, y + 20, t, 't-tick', 'middle'))
    return out


b.append(text(20, 52, 'Without overlap', 't-title'))
b += lane(62, 'CPU', [(0, 60, 'box-2', 'sched N'), (200, 60, 'box-2', 'post N'), (260, 60, 'box-2', 'sched N+1'), (460, 60, 'box-2', 'post N+1')])
b += lane(98, 'GPU', [(60, 140, 'box-1', 'forward + sample N'), (320, 140, 'box-1', 'forward + sample N+1')])
b.append(text(120 + 200 + 60, 146, 'GPU idle here', 't-muted', 'middle'))
b.append(text(20, 178, 'With overlap (DeepInfer default for decode)', 't-title'))
b += lane(188, 'CPU', [(0, 60, 'box-2', 'sched N'), (80, 60, 'box-2', 'sched N+1'), (140, 60, 'box-2', 'post N-1'), (220, 60, 'box-2', 'sched N+2'), (280, 60, 'box-2', 'post N')])
b += lane(224, 'GPU', [(60, 140, 'box-1', 'forward + sample N'), (200, 140, 'box-1', 'forward + sample N+1'), (340, 140, 'box-1', 'N+2')])
b.append(text(20, 280, 'The CPU prepares step N+1 and finishes step N-1 while the GPU runs step N.', 't-muted'))
FIGS['zos'] = svg(W, 292, 'Two timelines: without overlap the GPU waits while the CPU schedules and post-processes; with overlap the CPU work for neighbouring steps runs while the GPU computes.', b)

# 7. one 128-token block and one state slot ---------------------------------------------------------------------
b = [text(20, 24, 'What one 128-token block holds for V4.1-Flash (all 40 layers)', 't-title')]
owners = KV['owners']
total = KV['block_bytes']
x, y, wtot = 30, 50, W - 60
for o in owners:
    for part, cls, lab in (('main', 's1', 'main KV'), ('index', 's2', 'indexer K')):
        v = o[part]
        w = v / total * wtot
        b.append(f'<g class="mark"><title>layer {o["layer"]} {lab}: {v:,} B</title><rect class="{cls}" x="{x:.1f}" y="{y}" width="{w - 2:.1f}" height="36" rx="3"/></g>')
        if part == 'main':
            b.append(text(x + w / 2, y + 23, f'L{o["layer"]} main {v / 1024:.0f} KiB', 't-tick', 'middle'))
        x += w
b.append(text(30, y + 58, f'{total:,} bytes per 128 tokens = {KV["bytes_per_token"]:.0f} bytes per token. Only layers 2, 8, 14, 20 own a slice;', 't-note'))
b.append(text(30, y + 76, 'other layers reuse the nearest earlier owner. Blue: FP4 main KV, 288 B per entry.', 't-note'))
b.append(text(30, y + 94, 'Orange: FP4 indexer keys, 68 B per entry. Layers 2-19 keep one entry per 2 tokens.', 't-note'))
y2 = 192
b.append(text(20, y2, 'Plus one fixed slot per running sequence (does not grow with context)', 't-title'))
sw = KV['swa_layers'] * KV['swa_tokens'] * 528
tot2 = KV['state_bytes']
store = KV['state_to_disk']
segs = [(store, 's2', f'20 encoder SWA rings: {store / 2**20:.2f} MiB (saved to the store)'),
        (sw - store, 's3', '23 decoder + draft rings (rebuilt, never saved)'),
        (KV['comp_state'], 's1', '')]
x = 30
for v, cls, lab in segs:
    w = v / tot2 * wtot
    b.append(f'<g class="mark"><title>{esc(lab or "compression state")}: {v:,} B</title><rect class="{cls}" x="{x:.1f}" y="{y2 + 14}" width="{max(3, w - 2):.1f}" height="36" rx="3"/></g>')
    if lab:
        b.append(text(x + w / 2, y2 + 37, lab, 't-tick', 'middle'))
    x += w
b.append(text(30, y2 + 72, f'{tot2:,} bytes = {tot2 / 2**20:.2f} MiB per sequence: 43 sliding-window rings of 134 tokens x 528 B,', 't-note'))
b.append(text(30, y2 + 90, 'plus a thin blue sliver of FP32 compression state.', 't-note'))
FIGS['kvblock'] = svg(W, 300, 'Two stacked bars: a 128-token paged block of 113,920 bytes split among four owner layers, and a per-sequence state slot of about 3 MiB made of sliding-window rings.', b)

# 8. bytes per token, two ways ---------------------------------------------------------------------------------
rows = [('every layer keeps its own FP8 entry (40 x 528 B)', KV['naive_bytes_per_token'], 's3'),
        ('DeepInfer paged block, V4.1-Flash', KV['bytes_per_token'], 's1')]
b = [text(20, 24, 'KV bytes per token of context', 't-title')]
mx = rows[0][1]
for i, (n, v, c) in enumerate(rows):
    y = 44 + i * 40
    b.append(text(20, y + 15, n, 't-note'))
    b.append(f'<rect class="{c}" x="20" y="{y + 20}" width="{max(3, v / mx * 560):.1f}" height="14" rx="4"/>')
    b.append(text(20 + max(3, v / mx * 560) + 8, y + 32, f'{v:,.0f} B', 't-val'))
b.append(text(20, 148, 'Sharing across layers, 2x compression in layers 2-19, and FP4 entries give the 24x gap.', 't-muted'))
FIGS['kvbytes'] = svg(W, 160, 'Two bars: 21,120 bytes per token if all 40 layers stored an FP8 entry, versus 890 bytes per token in DeepInfer\'s block layout.', b)

# 9. variable-length verification toy -------------------------------------------------------------------------
vt = TY['varlen']
tr = vt['trace']
b = [text(20, 24, 'Toy: expected tokens per millisecond as more draft tokens are verified', 't-title')]
L, T, Wp, Hp = 70, 44, 600, 170
xs = [t[0] for t in tr]
ys = [t[3] for t in tr]
X = lambda v: L + (v - 4) / (24 - 4) * Wp
Y = lambda v: T + Hp - (v - 0.3) / (1.2 - 0.3) * Hp
for tv in (0.4, 0.6, 0.8, 1.0, 1.2):
    b.append(f'<line class="grid" x1="{L}" y1="{Y(tv):.1f}" x2="{L + Wp}" y2="{Y(tv):.1f}"/>')
    b.append(text(L - 8, Y(tv) + 4, f'{tv:.1f}', 't-tick', 'end'))
for xv in (4, 8, 12, 16, 20, 24):
    b.append(text(X(xv), T + Hp + 18, str(xv), 't-tick', 'middle'))
b.append(f'<polyline class="l1" points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(xs, ys)) + '"/>')
bx = vt['best_total']
by = vt['best_tp']
b.append(f'<circle class="box-on" cx="{X(bx):.1f}" cy="{Y(by):.1f}" r="6"/>')
b.append(text(X(bx), Y(by) - 12, f'best: {bx} tokens, {by:.2f}', 't-title', 'middle'))
b.append(f'<circle class="box-2" cx="{X(24):.1f}" cy="{Y(vt["fixed_tp"]):.1f}" r="6"/>')
b.append(text(X(24) - 8, Y(vt['fixed_tp']) + 20, f'verify all 5: {vt["fixed_tp"]:.2f}', 't-note', 'end'))
b.append(text(L + Wp / 2, T + Hp + 38, 'tokens in the verification pass (4 requests x 1 real token + chosen draft tokens)', 't-muted', 'middle'))
b.append(text(L + Wp / 2, T + Hp + 56, 'Invented step-time curve and confidences; the steps in the curve are the CUDA-graph sizes.', 't-muted', 'middle'))
FIGS['varlen'] = svg(W, 290, 'Line chart from the toy: expected accepted tokens per millisecond rises, peaks at 16 verified tokens, and falls when the step time jumps.', b)

# 10. EPLB toy ---------------------------------------------------------------------------------------------------
ep = TY['eplb']
b = [text(20, 24, 'Toy EPLB: 8 experts, 4 GPUs, 4 spare slots (step 1 of 3 shown)', 't-title')]
for k, (title, rl, ratio, assign) in enumerate([('one copy each, in order', ep['rank_loads_before'], ep['naive_ratio'], ep['naive']),
                                                ('after replication + greedy placement', ep['rank_loads_after'], ep['ratio'], ep['placement'])]):
    x0 = 30 + k * 370
    b.append(text(x0, 52, title, 't-title'))
    loads = rl[0]
    for g, v in enumerate(loads):
        y = 66 + g * 34
        b.append(text(x0 + 48, y + 17, f'GPU {g}', 't-tick', 'end'))
        b.append(f'<rect class="{"s3" if k == 0 else "s1"}" x="{x0 + 56}" y="{y}" width="{v * 1.5:.1f}" height="22" rx="4"/>')
        b.append(text(x0 + 60 + v * 1.5, y + 16, f'{v:.0f} tokens; experts {assign[g]}', 't-tick'))
    b.append(text(x0, 220, f'slowest GPU / average, over 3 steps: {ratio:.2f}', 't-note'))
b.append(text(20, 250, 'Expert 0 is hot. It gets 3 extra copies; each copy then carries a quarter of its tokens.', 't-muted'))
FIGS['eplb'] = svg(W, 262, 'Two bar groups: before balancing GPU 0 carries 100 tokens and the others 20 to 40; after replicating the hot expert, all four GPUs carry about 50.', b)

json.dump(FIGS, open(os.path.join(RESULTS, 'figs_part9.json'), 'w'))
print(f'{len(FIGS)} figures -> results/figs_part9.json')
