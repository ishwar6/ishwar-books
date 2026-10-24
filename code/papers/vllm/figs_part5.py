"""Figures for Part 5 (the experiments). Writes results/figs_part5.json. CSS classes only.
Every number comes from results/part5.json (our simulation and laptop timings) or, where marked, from the paper."""
import json, math, os
from figlib import svg, text, box, arrow, esc

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part5.json')))
F = {}
SYS = ['FasterTransformer', 'Orca (Max)', 'Orca (Pow2)', 'Orca (Oracle)', 'vLLM']
STYLE = {'FasterTransformer': ('long', 'box'), 'Orca (Max)': ('l2', 's2'), 'Orca (Pow2)': ('l4', 's4'),
         'Orca (Oracle)': ('l3', 's3'), 'vLLM': ('l1', 's1')}


def chart(x0, y0, w, h, xs, series, xmin, xmax, ymin, ymax, xticks, yticks, xlabel, ylabel, title=None,
          xlog=False, ylog=False, fmt_x=lambda v: f'{v:g}', fmt_y=lambda v: f'{v:g}', clip=True, labels=True):
    """A small line chart inside a bigger SVG. A line that leaves the top of the box stops at the edge (as in the
    paper's plots, which cut the y axis at 1.0)."""
    fx = (lambda v: math.log2(v)) if xlog else (lambda v: v)
    fy = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X = lambda v: x0 + (fx(v) - fx(xmin)) / (fx(xmax) - fx(xmin)) * w
    Y = lambda v: y0 + h - (fy(v) - fy(ymin)) / (fy(ymax) - fy(ymin)) * h
    b = []
    if title:
        b.append(text(x0, y0 - 14, title, 't-title'))
    for t in yticks:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>',
              text(x0 - 6, Y(t) + 4, fmt_y(t), 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), y0 + h + 18, fmt_x(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    b.append(text(x0 + w / 2, y0 + h + 38, xlabel, 't-tick', 'middle'))
    if ylabel:
        b.append(text(x0 - 6, y0 - 10, ylabel, 't-tick', 'end'))
    for name, vals, lc, dc in series:
        pts = []
        for x, v in zip(xs, vals):
            if clip and v > ymax:
                if pts:                                   # draw up to the top edge, then stop
                    (xa, va), xb = pts[-1], x
                    xt = X(xa) + (X(xb) - X(xa)) * (ymax - va) / (v - va)
                    pts.append((None, (xt, Y(ymax))))
                break
            pts.append((x, v))
        coords = [p[1] if p[0] is None else (X(p[0]), Y(p[1])) for p in pts]
        if len(coords) > 1:
            b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{a:.1f},{c:.1f}' for a, c in coords) + '"/>')
        for p in pts:
            if p[0] is None:
                continue
            cx, cy = X(p[0]), Y(p[1])
            mark = (f'<rect class="{dc}" x="{cx - 3.5:.1f}" y="{cy - 3.5:.1f}" width="7" height="7" rx="1"/>' if dc == 'box'
                    else f'<circle class="{dc} ring" cx="{cx:.1f}" cy="{cy:.1f}" r="4"/>')
            b.append(f'<g class="mark"><title>{esc(name)}, {fmt_x(p[0])}: {p[1]:.3f}</title>{mark}</g>')
    return b


def legend(x, y, items, gap=24):
    b = []
    for name, (lc, dc) in items:
        b.append(f'<line class="{lc}" x1="{x}" y1="{y - 4}" x2="{x + 18}" y2="{y - 4}"/>')
        b.append(f'<rect class="{dc}" x="{x + 5.5}" y="{y - 7.5}" width="7" height="7" rx="1"/>')
        b.append(text(x + 24, y, name, 't-tick'))
        x += 30 + 7.2 * len(name) + gap
    return b


def fig11():
    """Histograms of our prompt and output lengths, as in the paper's Figure 11."""
    W, H = 720, 300
    b = [text(20, 24, 'Prompt and output lengths of our requests (OPT tokenizer)', 't-title')]
    b += legend(540, 24, [('prompt', ('l1', 's1')), ('output', ('l2', 's2'))])
    for k, ds in enumerate(['ShareGPT', 'Alpaca']):
        L = R['lengths'][ds]
        x0, y0, w, h, bin_w, top = 60 + k * 340, 60, 290, 170, 50, 2000
        nb = top // bin_w
        hp = [0] * nb; ho = [0] * nb
        for v in L['prompts']:
            hp[min(v // bin_w, nb - 1)] += 1
        for v in L['outputs']:
            ho[min(v // bin_w, nb - 1)] += 1
        m = max(hp + ho)
        b.append(text(x0, y0 - 8, f'{ds}: mean prompt {L["prompt_mean"]:.0f}, mean output {L["output_mean"]:.0f} tokens', 't-tick t-strong'))
        bw = w / nb
        for i in range(nb):
            for vals, cls, off in ((hp, 's1', 0), (ho, 's2', bw / 2)):
                hh = h * vals[i] / m
                if hh > 0:
                    b.append(f'<rect class="{cls}" x="{x0 + i * bw + off:.1f}" y="{y0 + h - hh:.1f}" width="{bw / 2 - 0.5:.1f}" height="{hh:.1f}"/>')
        b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
        for t in (0, 500, 1000, 1500, 2000):
            b.append(text(x0 + w * t / top, y0 + h + 18, str(t), 't-tick', 'middle'))
        b.append(text(x0 + w / 2, y0 + h + 38, 'tokens (bins of 50)', 't-tick', 'middle'))
    return svg(W, H, 'Two histograms. ShareGPT has long prompts and long answers that spread out to 2,000 tokens. '
               'Alpaca prompts and answers are short and packed near zero.', b)


def fig12():
    """Our Figure 12: normalized latency against request rate, both datasets, y axis cut at 1.0 as in the paper."""
    W, H = 720, 360
    b = legend(20, 24, [(s, STYLE[s]) for s in SYS], gap=6)
    for k, ds in enumerate(['ShareGPT', 'Alpaca']):
        rates = R['rates'][ds]
        x0 = 70 + k * 345
        xmax = rates[-1]
        ticks = [0, 2, 4, 6, 8] if ds == 'ShareGPT' else [0, 50, 100, 150, 200]
        series = [(s, R['curves'][ds][s], *STYLE[s]) for s in SYS]
        b += chart(x0, 70, 270, 220, rates, series, 0, xmax, 0, 1.0, ticks, [0, 0.25, 0.5, 0.75, 1.0],
                   'request rate (requests per second)', '', title=f'{ds}', fmt_y=lambda v: f'{v:.2f}')
    b.append(text(20, H - 8, 'normalized latency (s/token), simulated: OPT-125M step times from our laptop, '
                  '15,728 KV slots', 't-tick'))
    return svg(W, H, 'Two panels of normalized latency against request rate. In each, the curves stay flat and then '
               'shoot up. FasterTransformer shoots up first, then Orca (Max), Orca (Pow2), Orca (Oracle), and vLLM last.', b)


def fig_read():
    """How to read one panel: the latency line, and where two systems cross it."""
    ds, lim = 'ShareGPT', 0.1
    rates = R['rates'][ds]
    W, H = 720, 330
    x0, y0, w, h = 80, 50, 440, 220
    series = [(s, R['curves'][ds][s], *STYLE[s]) for s in ('Orca (Oracle)', 'vLLM')]
    b = chart(x0, y0, w, h, rates, series, 0, 4, 0, 0.5, [0, 1, 2, 3, 4], [0, 0.1, 0.2, 0.3, 0.4, 0.5],
              'request rate (requests per second)', 's/token', fmt_y=lambda v: f'{v:.1f}')
    X = lambda v: x0 + v / 4 * w
    Y = lambda v: y0 + h - v / 0.5 * h
    b.append(f'<line class="base-line" x1="{x0}" y1="{Y(lim):.1f}" x2="{x0 + w}" y2="{Y(lim):.1f}"/>')
    b.append(text(x0 + 8, Y(lim) - 6, f'our line: {lim} s/token', 't-tick'))
    for s, cls in (('Orca (Oracle)', 't-tick'), ('vLLM', 't-tick')):
        c = R['capacity'][ds][s]
        b.append(f'<line class="drop" x1="{X(c):.1f}" y1="{Y(lim):.1f}" x2="{X(c):.1f}" y2="{y0 + h}"/>')
        b.append(text(X(c), y0 + h - 6, f'{c:.1f}', 't-tick t-strong', 'middle'))
    o, v = R['capacity'][ds]['Orca (Oracle)'], R['capacity'][ds]['vLLM']
    b.append(text(x0 + w + 8, Y(0.35), 'Orca (Oracle)', 't-note'))
    b.append(text(x0 + w + 8, Y(0.35) + 18, f'crosses at {o:.1f} req/s', 't-tick'))
    b.append(text(x0 + w + 8, Y(0.25), 'vLLM', 't-note'))
    b.append(text(x0 + w + 8, Y(0.25) + 18, f'crosses at {v:.1f} req/s', 't-tick'))
    b.append(text(x0 + w + 8, Y(0.15), f'{v:.1f} / {o:.1f} = {v / o:.1f}x the rate', 't-tick t-strong'))
    b.append(text(x0, 24, 'Reading a panel: pick a latency, see which rate each system reaches (ShareGPT, our run)', 't-title'))
    return svg(W, H, 'A zoom on two curves. A dashed line at 0.1 seconds per token crosses the Orca Oracle curve at a '
               'lower rate than the vLLM curve. The ratio of the two rates is what the paper reports as times higher.', b)


def fig13():
    """Requests running at once: our simulation next to the paper's Figure 13."""
    paper = {'ShareGPT': [7.00, 9.81, 13.62, 30.42], 'Alpaca': [7.00, 43.24, 72.75, 132.44]}
    sys4 = SYS[1:]
    W, H = 720, 340
    b = [text(20, 24, 'Requests in the batch when the queue is never empty', 't-title')]
    b += legend(440, 50, [('our simulation', ('l1', 's1')), ('paper (A100)', ('l3', 's3'))])
    for k, ds in enumerate(['ShareGPT', 'Alpaca']):
        x0, y0, w, h = 50 + k * 350, 90, 290, 180
        vmax = 40 if ds == 'ShareGPT' else 220
        b.append(text(x0, y0 - 10, ds, 't-tick t-strong'))
        gw = w / 4
        for i, s in enumerate(sys4):
            for j, (v, cls) in enumerate(((R['batch'][ds][s]['sim'], 's1'), (paper[ds][i], 's3'))):
                bh = h * v / vmax
                x = x0 + i * gw + 8 + j * (gw / 2 - 6)
                b.append(f'<g class="mark"><title>{esc(s)}: {v:.2f}</title><rect class="{cls}" x="{x:.1f}" '
                         f'y="{y0 + h - bh:.1f}" width="{gw / 2 - 10:.1f}" height="{bh:.1f}" rx="2"/></g>')
                b.append(text(x + (gw / 2 - 10) / 2, y0 + h - bh - 4, f'{v:.0f}' if v >= 10 else f'{v:.1f}', 't-tick', 'middle'))
            b.append(text(x0 + i * gw + gw / 2, y0 + h + 16, s.replace('Orca ', ''), 't-tick', 'middle'))
        b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    b.append(text(20, H - 10, 'Orca labels: (Max), (Pow2), (Oracle). Our batch is capped at 256 requests.', 't-tick'))
    return svg(W, H, 'Grouped bars. For each Orca version and vLLM, our simulated batch size stands next to the paper\'s. '
               'They follow the same order, and vLLM runs the most requests at once on both datasets.', b)


def fig18b():
    bs = R['blocks']['sizes']
    W, H = 720, 300
    series = [('ShareGPT', R['blocks']['latency']['ShareGPT'], 'l1', 's1'),
              ('Alpaca', R['blocks']['latency']['Alpaca'], 'l2', 's2')]
    b = [text(20, 24, 'vLLM with different block sizes (our simulator, memory effects only)', 't-title')]
    b += legend(20, 50, [('ShareGPT', ('l1', 's1')), ('Alpaca', ('l2', 's2'))])
    b += chart(80, 80, 560, 160, bs, series, 1, 256, 0, 0.4, bs, [0, 0.1, 0.2, 0.3, 0.4], 'block size (tokens)',
               's/token', xlog=True, fmt_y=lambda v: f'{v:.1f}')
    r = R['blocks']['rate']
    b.append(text(20, H - 6, f'fixed rates: ShareGPT {r["ShareGPT"]} req/s, Alpaca {r["Alpaca"]} req/s', 't-tick'))
    return svg(W, H, 'Two lines of normalized latency against block size. Both stay flat for small blocks and rise for '
               'large blocks; the Alpaca line rises earlier and higher.', b)


def fig19():
    bs = sorted(int(k) for k in R['swap']['blocks'])
    sw = [R['swap']['blocks'][str(b)]['out_ms'] + R['swap']['blocks'][str(b)]['in_ms'] for b in bs]
    re = [R['swap']['recompute_ms']] * len(bs)
    W, H = 720, 300
    b = [text(20, 24, f'Bring back a 1,024-token KV cache on our laptop ({R["swap"]["kv_mb"]:.1f} MB, OPT-125M)', 't-title')]
    b += legend(20, 50, [('swap out + swap in', ('l2', 's2')), ('recompute (one prompt pass)', ('l1', 's1'))])
    b += chart(80, 80, 560, 160, bs, [('swap', sw, 'l2', 's2'), ('recompute', re, 'l1', 's1')], 1, 256, 10, 20000,
               bs, [10, 100, 1000, 10000], 'block size (tokens)', 'ms (log)', xlog=True, ylog=True,
               fmt_y=lambda v: f'{v:,.0f}', clip=False)
    return svg(W, H, f'Two lines on a log scale. Swapping falls from {sw[0]:,.0f} ms at block size 1 to {sw[-1]:,.0f} ms at '
               f'block size 256. Recompute is flat at {re[0]:.1f} ms.', b)



# ---------------------------------------------------------------- new figures: background and how to read the tests
def fig_arch():
    """The vLLM architecture (paper Figure 4), redrawn as a "you are here" map. Part 5 tests the scheduler and its KV
    cache manager from the outside: requests go in at a set rate, and a stopwatch measures each one."""
    W, H = 720, 380
    b = [text(20, 24, 'You are here: Part 5 tests the whole system from the outside', 't-title')]
    # left: requests
    b += [box(20, 120, 130, 120, 'box', 10), text(85, 146, 'Requests', 't-note', 'middle'),
          text(85, 170, 'prompt lengths', 't-tick', 'middle'), text(85, 188, 'from ShareGPT', 't-tick', 'middle'),
          text(85, 206, 'or Alpaca, at', 't-tick', 'middle'), text(85, 224, 'random times', 't-tick', 'middle')]
    # centre: scheduler with the KV cache manager
    b += [box(190, 60, 270, 250, 'box-on', 12), text(325, 84, 'Scheduler', 't-note', 'middle'),
          text(325, 102, 'what the baselines replace', 't-tick', 'middle'),
          box(210, 116, 230, 178, 'box', 10), text(325, 138, 'KV cache manager', 't-note', 'middle')]
    for k, s in enumerate(['Block tables', 'GPU block allocator', 'CPU block allocator']):
        b += [box(226, 152 + k * 46, 198, 34, 'box-1', 6), text(325, 174 + k * 46, s, 't-tick', 'middle')]
    # right: workers
    for k, s in enumerate(['Worker 0', 'Worker 1', 'Worker N-1']):
        y = 60 + k * 86
        b += [box(510, y, 190, 74, 'box', 10), text(522, y + 20, s, 't-note'),
              box(522, y + 32, 80, 30, 'box-2', 6), text(562, y + 52, 'Cache engine', 't-tick', 'middle'),
              box(610, y + 32, 80, 30, 'box-3', 6), text(650, y + 52, 'Model shard', 't-tick', 'middle')]
    b += [arrow(150, 180, 188, 180, on=True), arrow(460, 150, 508, 97), arrow(460, 185, 508, 183), arrow(460, 220, 508, 269)]
    # the measurement band
    b += [box(20, 328, 680, 40, 'box-ghost', 8),
          text(360, 353, 'Measured outside: request rate in (requests per second), and for each request: '
               'time in the system / tokens written', 't-tick', 'middle')]
    return svg(W, H, 'The vLLM architecture as a map. Requests on the left go to a scheduler in the centre, which holds '
               'the KV cache manager with block tables and GPU and CPU block allocators. The scheduler is highlighted. '
               'On the right are workers 0 to N-1, each with a cache engine and a model shard. A band at the bottom '
               'says what the tests measure from outside.', b)


def poisson_draw(n, rate, seed):
    """The same generator as vllm_part5.py: gaps drawn from an exponential distribution with mean 1 / rate."""
    import random
    r, t, out = random.Random(seed), 0.0, []
    for _ in range(n):
        t += r.expovariate(rate); out.append(t)
    return out


def fig_poisson():
    """Regular arrivals against Poisson arrivals at the same average rate of 1 request per second."""
    T = 30
    reg = list(range(1, T + 1))
    poi = [t for t in poisson_draw(60, 1.0, 2) if t <= T]
    W, H = 720, 230
    x0, w = 120, 570
    X = lambda t: x0 + t / T * w
    b = [text(20, 24, 'Thirty seconds of arrivals at an average of 1 request per second', 't-title')]
    for k, (name, ts, cls) in enumerate((('every second', reg, 's3'), ('Poisson', poi, 's1'))):
        y = 72 + k * 70
        b.append(text(x0 - 12, y + 4, name, 't-note', 'end'))
        b.append(f'<line class="axis" x1="{x0}" y1="{y}" x2="{x0 + w}" y2="{y}"/>')
        for t in ts:
            b.append(f'<g class="mark"><title>{name}: {t:.2f} s</title><circle class="{cls} ring" cx="{X(t):.1f}" cy="{y:.1f}" r="5"/></g>')
        b.append(text(x0 - 12, y + 20, f'{len(ts)} arrivals', 't-tick', 'end'))
    gaps = [(b2 - a2, a2, b2) for a2, b2 in zip(poi, poi[1:])]
    g, ga, gb = max(gaps)
    y = 142
    b.append(f'<line class="drop" x1="{X(ga):.1f}" y1="{y - 16}" x2="{X(gb):.1f}" y2="{y - 16}"/>')
    b.append(text((X(ga) + X(gb)) / 2, y - 22, f'a gap of {g:.1f} s', 't-tick', 'middle'))
    # the tightest group of three
    trip = min((poi[i + 2] - poi[i], i) for i in range(len(poi) - 2))
    i = trip[1]
    b.append(text(X(poi[i + 1]), y + 26, f'3 within {trip[0]:.2f} s', 't-tick', 'middle'))
    for t in (0, 5, 10, 15, 20, 25, 30):
        b.append(text(X(t), 196, f'{t}', 't-tick', 'middle'))
    b.append(text(x0 + w / 2, 218, 'time (seconds)', 't-tick', 'middle'))
    return svg(W, H, 'Two timelines of arrivals. On the top line a request comes exactly every second. On the bottom '
               'line, drawn at random with the same average, requests come in clumps with long gaps between them.', b)


def fig_queue():
    """The simplest queue: one cashier who serves 10 customers a minute on average. Mean time in the shop
    1 / (mu - lambda) against the arrival rate lambda. A formula, not a measurement."""
    mu = 10.0
    W, H = 720, 320
    x0, y0, w, h = 330, 60, 340, 200
    lam = [i / 10 for i in range(0, 98)]
    X = lambda v: x0 + v / mu * w
    Y = lambda v: y0 + h - min(v, 5) / 5 * h
    b = [text(20, 24, 'One cashier, 10 customers a minute: why the curve bends so sharply', 't-title')]
    # left: a little shop
    b += [box(40, 120, 70, 60, 'box-on', 8), text(75, 155, 'cashier', 't-tick', 'middle')]
    for k in range(5):
        b.append(f'<circle class="s2" cx="{140 + k * 30}" cy="150" r="10"/>')
    b += [text(200, 200, 'the queue', 't-tick', 'middle'), arrow(290, 150, 268, 150)]
    b.append(text(40, 236, 'Customers arrive at random.', 't-tick'))
    b.append(text(40, 254, 'Below 10 a minute, the queue empties now and then.', 't-tick'))
    b.append(text(40, 272, 'At 10 or more, it never empties again.', 't-tick'))
    # right: the curve
    for t in (0, 1, 2, 3, 4, 5):
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>',
              text(x0 - 6, Y(t) + 4, f'{t}', 't-tick', 'end')]
    for t in (0, 2, 4, 6, 8, 10):
        b.append(text(X(t), y0 + h + 18, f'{t}', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    pts = [(X(l), Y(1 / (mu - l))) for l in lam if 1 / (mu - l) <= 5]
    b.append('<polyline class="l1" points="' + ' '.join(f'{a:.1f},{c:.1f}' for a, c in pts) + '"/>')
    b.append(f'<line class="base-line" x1="{X(mu):.1f}" y1="{y0}" x2="{X(mu):.1f}" y2="{y0 + h}"/>')
    b.append(text(X(mu) - 4, y0 - 6, 'capacity: 10', 't-tick', 'end'))
    for l in (5, 8, 9):
        v = 1 / (mu - l)
        b.append(f'<g class="mark"><title>{l:g} a minute: {v:.2f} min</title><circle class="s1 ring" cx="{X(l):.1f}" cy="{Y(v):.1f}" r="5"/></g>')
        b.append(text(X(l) - 8, Y(v) - 8, f'{l:g}/min: {v * 60:.0f} s', 't-tick', 'end'))
    b.append(text(x0 + w / 2, y0 + h + 38, 'customers arriving per minute', 't-tick', 'middle'))
    b.append(text(x0 - 6, y0 - 12, 'minutes in the shop', 't-tick', 'end'))
    return svg(W, H, 'Left: a cashier with a queue of five customers. Right: the mean time a customer spends in the shop '
               'against the arrival rate. It stays low up to about 5 a minute, then climbs faster and faster, and goes '
               'to infinity at 10 a minute, the cashier\'s speed.', b)


def fig_reserve():
    """What each system sets aside for one request with a 19-token prompt and a 25-token answer."""
    p, o, blk = 19, 25, 16
    W, H = 720, 340
    x0, unit = 170, 5.4                      # pixels per slot; 76 slots = 410 px
    b = [text(20, 24, 'Memory set aside for one request: prompt 19 tokens, answer 25 tokens', 't-title')]
    b += [box(20, 40, 14, 12, 's1', 2), text(40, 51, 'prompt', 't-tick'),
          box(110, 40, 14, 12, 's2', 2), text(130, 51, 'answer, written', 't-tick'),
          box(250, 40, 14, 12, 'waste', 2), text(270, 51, 'set aside but never used', 't-tick')]

    def seg(x, y, n, cls):
        return box(x, y, n * unit, 26, cls, 2)

    rows = [('FasterTransformer', 2048, 2048), ('Orca (Max)', 2048, 2048), ('Orca (Pow2)', p + 32, 64),
            ('Orca (Oracle)', p + o, 64)]
    for k, (name, booked, chunk) in enumerate(rows):
        y = 74 + k * 50
        b.append(text(x0 - 10, y + 18, name, 't-note', 'end'))
        b += [seg(x0, y, p, 's1'), seg(x0 + p * unit, y, o, 's2')]
        if chunk == 2048:
            b.append(seg(x0 + (p + o) * unit, y, 76 - p - o, 'waste'))
            zx = x0 + 76 * unit
            b.append(f'<path class="edge" d="M{zx + 4},{y - 2} l6,10 l-6,10 l6,10"/>')
            b.append(text(zx + 18, y + 18, '... 2,048 slots', 't-tick t-strong'))
        else:
            b.append(seg(x0 + (p + o) * unit, y, chunk - p - o, 'waste'))
            b.append(text(x0 + chunk * unit + 8, y + 12, f'books {booked}', 't-tick'))
            b.append(text(x0 + chunk * unit + 8, y + 27, f'buddy gives {chunk}', 't-tick t-strong'))
    y = 74 + 4 * 50
    b.append(text(x0 - 10, y + 18, 'vLLM', 't-note', 'end'))
    b += [seg(x0, y, p, 's1'), seg(x0 + p * unit, y, o, 's2')]
    n_blocks = -(-(p + o) // blk)
    b.append(seg(x0 + (p + o) * unit, y, n_blocks * blk - p - o, 'waste'))
    for j in range(n_blocks):
        b.append(box(x0 + j * blk * unit, y - 3, blk * unit, 32, 'box-ghost', 3))
    b.append(text(x0 + n_blocks * blk * unit + 8, y + 12, f'{n_blocks} blocks of 16 = {n_blocks * blk}', 't-tick t-strong'))
    b.append(text(x0 + n_blocks * blk * unit + 8, y + 27, 'taken one at a time', 't-tick'))
    b.append(text(20, H - 6, 'One slot holds the keys and values of one token. Bars are drawn to scale up to 76 slots.', 't-tick'))
    return svg(W, H, 'Five bars, one per system, for a request with a 19-token prompt and a 25-token answer. '
               'FasterTransformer and Orca Max set aside 2,048 slots. Orca Pow2 books 51 and Orca Oracle books 44, '
               'and the buddy allocator rounds both up to 64. vLLM holds 3 blocks of 16, or 48 slots, '
               'and wastes only 4.', b)


def fig_howto():
    """An annotated reading guide for one latency-vs-rate panel, drawn on our ShareGPT run."""
    ds = 'ShareGPT'
    rates = R['rates'][ds]
    W, H = 720, 380
    x0, y0, w, h = 80, 60, 420, 250
    xmax, ymax = 5, 1.0
    series = [(s, R['curves'][ds][s], *STYLE[s]) for s in ('Orca (Oracle)', 'vLLM')]
    b = [text(20, 24, 'How to read a latency plot (our ShareGPT run, two systems)', 't-title')]
    b += chart(x0, y0, w, h, rates, series, 0, xmax, 0, ymax, [0, 1, 2, 3, 4, 5], [0, 0.25, 0.5, 0.75, 1.0],
               'request rate: the load (requests per second)', 's/token', fmt_y=lambda v: f'{v:.2f}')
    X = lambda v: x0 + v / xmax * w
    Y = lambda v: y0 + h - v / ymax * h
    floor = R['curves'][ds]['vLLM'][0]
    o, v = R['capacity'][ds]['Orca (Oracle)'], R['capacity'][ds]['vLLM']

    def badge(x, y, n):
        return [f'<circle class="box-on" cx="{x:.1f}" cy="{y:.1f}" r="10"/>', text(x, y + 4.5, str(n), 't-note', 'middle')]
    for c, cls in ((o, 's3'), (v, 's1')):
        b.append(f'<circle class="{cls} ring" cx="{X(c):.1f}" cy="{Y(0.1):.1f}" r="7"/>')
    b.append(f'<line class="edge-on" x1="{X(o) + 10:.1f}" y1="{Y(0.1):.1f}" x2="{X(v) - 10:.1f}" y2="{Y(0.1):.1f}" marker-end="url(#ah-on)"/>')
    b += badge(X(1.0), Y(0.09), 1)
    b += badge(X(v) + 18, Y(0.03), 2)
    b += badge(X(4.35), Y(0.6), 3)
    b += badge((X(o) + X(v)) / 2 + 6, Y(0.1) - 18, 4)
    lx = x0 + w + 30
    b += legend(lx, 70, [('Orca (Oracle)', STYLE['Orca (Oracle)'])])
    b += legend(lx, 92, [('vLLM', STYLE['vLLM'])])
    notes = [(1, 'Flat floor: nobody waits.', f'{floor:.3f} s to write a token.'),
             (2, 'Knee: the batch is full,', 'and a queue starts.'),
             (3, 'Wall: requests come in', 'faster than they leave.'),
             (4, 'Further right is better:', 'more users on one GPU.')]
    for k, (n, l1, l2) in enumerate(notes):
        yy = 130 + k * 48
        b += badge(lx + 10, yy - 4, n)
        b += [text(lx + 26, yy, l1, 't-tick t-strong'), text(lx + 26, yy + 17, l2, 't-tick')]
    b.append(text(lx, 330, 'Lines that leave the top', 't-tick'))
    b.append(text(lx, 347, 'are cut, as in the paper.', 't-tick'))
    return svg(W, H, 'An annotated latency plot with two curves. Both start on a flat floor at low load. Each has a knee, '
               'marked with a circle, where its batch is full and a queue starts, and then a nearly vertical wall. '
               'An arrow from the Orca knee to the vLLM knee says further right is better.', b)


def fig_memory():
    """Before vLLM and with vLLM: the same 15,728 slots, cut into chunks for the requests that run at once.
    Batch sizes from results/part5.json (ShareGPT, queue never empty)."""
    S = 15728
    W, H = 720, 300
    x0, w = 150, 550
    U = w / S
    per = S / R['batch']['ShareGPT']['vLLM']['sim']        # slots an average running request really holds
    b = [text(20, 24, 'The same 15,728 slots of KV cache, ShareGPT, the queue never empty (our simulation)', 't-title')]
    rows = [('Orca (Max)', 2048, int(R['batch']['ShareGPT']['Orca (Max)']['sim'])),
            ('Orca (Oracle)', None, R['batch']['ShareGPT']['Orca (Oracle)']['sim']),
            ('vLLM', None, R['batch']['ShareGPT']['vLLM']['sim'])]
    for k, (name, chunk, n) in enumerate(rows):
        y = 60 + k * 66
        b.append(text(x0 - 10, y + 20, name, 't-note', 'end'))
        size = chunk or S / n
        full = int(n)
        for j in range(full):
            x = x0 + j * size * U
            b.append(box(x, y, size * U - 1, 30, 'waste' if name != 'vLLM' else 'box-1', 1))
            b.append(box(x, y, min(per, size) * U - 1, 30, 's1', 1))
        used = full * size
        if name == 'Orca (Max)':
            b.append(box(x0 + used * U, y, (S - used) * U, 30, 'box-ghost', 1))
            b.append(text(x0 + w, y + 48, f'{S - used:,} slots left, too small for a chunk', 't-tick', 'end'))
        b.append(text(x0, y + 48 if name != 'Orca (Max)' else y + 48, f'{n:.2f} requests at once' if n != int(n) else f'{full} requests at once', 't-tick t-strong'))
    b += [box(20, H - 26, 14, 12, 's1', 2), text(40, H - 15, f'held by an average running request ({per:.0f} slots = 15,728 / {R["batch"]["ShareGPT"]["vLLM"]["sim"]:.2f})', 't-tick'),
          box(440, H - 26, 14, 12, 'waste', 2), text(460, H - 15, 'booked but empty, or holes', 't-tick')]
    return svg(W, H, 'Three bars for the same memory. Orca Max cuts it into 7 chunks of 2,048 slots, each mostly empty, '
               'with a leftover piece too small to use. Orca Oracle fits about 15 requests. vLLM fits about 37, '
               'with almost nothing empty.', b)

F['p5_arch'] = fig_arch()
F['p5_poisson'] = fig_poisson()
F['p5_queue'] = fig_queue()
F['p5_reserve'] = fig_reserve()
F['p5_howto'] = fig_howto()
F['p5_memory'] = fig_memory()
F['p5_fig11'] = fig11()
F['p5_fig12'] = fig12()
F['p5_read'] = fig_read()
F['p5_fig13'] = fig13()
F['p5_fig18b'] = fig18b()
F['p5_fig19'] = fig19()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part5.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
