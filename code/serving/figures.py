"""Theme-aware SVG figures for Part 6, drawn from the saved results. Classes (box-1, s1, l1, t-note, ...) are
styled by the site's CSS, so every figure follows the dark and light themes. No colours are hard-coded.
Run: python code/serving/figures.py  -> results/figures.json (assemble.py inlines them)."""
import json, math
from pathlib import Path
from html import escape

R = Path(__file__).resolve().parent / 'results'
J = lambda n: json.loads((R / f'{n}.json').read_text())

DEFS = ('<defs><marker id="ah6" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs>')
def text(x, y, s, cls='t-note', anchor='middle', extra=''):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{escape(str(s))}</text>'
def rect(x, y, w, h, cls='box-1', rx=6):
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}"/>'
def line(x, y, a, b, cls='edge', arrow=False):
    tip = ' marker-end="url(#ah6)"' if arrow else ''
    return f'<line class="{cls}" x1="{x:.1f}" y1="{y:.1f}" x2="{a:.1f}" y2="{b:.1f}"{tip}/>'
def poly(pts, cls='l1'):
    return f'<polyline class="{cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'
def dot(x, y, cls='s1', r=4):
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'
def mark(title, body):
    return f'<g class="mark"><title>{escape(title)}</title>{body}</g>'
def svg(h, body, label, w=760):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}">{DEFS}{body}</svg>'
def legend(items, x, y, gap=24):
    """items: (label, cls, kind) with kind 'line' or 'box'. Laid out left to right from x."""
    out = ''
    for lab, cls, kind in items:
        if kind == 'line':
            out += line(x, y - 4, x + 22, y - 4, cls)
            x0 = x + 28
        else:
            out += rect(x, y - 11, 12, 12, cls, 2); x0 = x + 18
        out += text(x0, y, lab, 't-tick', 'start')
        x = x0 + 7.0 * len(lab) + gap
    return out


class Axes:
    """A plot area with linear or log scales, ticks, grid and axis titles."""
    def __init__(self, x0, y0, w, h, xr, yr, xlog=False, ylog=False):
        self.x0, self.y0, self.w, self.h, self.xr, self.yr, self.xlog, self.ylog = x0, y0, w, h, xr, yr, xlog, ylog
    def _t(self, v, r, log):
        if log:
            return (math.log10(v) - math.log10(r[0])) / (math.log10(r[1]) - math.log10(r[0]))
        return (v - r[0]) / (r[1] - r[0])
    def X(self, v): return self.x0 + self.w * self._t(v, self.xr, self.xlog)
    def Y(self, v): return self.y0 + self.h - self.h * self._t(min(max(v, self.yr[0]), self.yr[1]), self.yr, self.ylog)
    def frame(self, xt, yt, xfmt=str, yfmt=str, xtitle='', ytitle=''):
        b = ''
        for v in yt:
            b += line(self.x0, self.Y(v), self.x0 + self.w, self.Y(v), 'grid') + text(self.x0 - 8, self.Y(v) + 4, yfmt(v), 't-tick', 'end')
        for v in xt:
            b += text(self.X(v), self.y0 + self.h + 18, xfmt(v), 't-tick')
        b += line(self.x0, self.y0 + self.h, self.x0 + self.w, self.y0 + self.h, 'axis')
        if xtitle: b += text(self.x0 + self.w / 2, self.y0 + self.h + 40, xtitle, 't-tick')
        if ytitle: b += text(self.x0 - 8, self.y0 - 12, ytitle, 't-tick', 'start')
        return b


f = {}
ms = lambda v: f'{v:,.0f} ms' if v < 1000 else (f'{v / 1000:.2f} s' if v < 10000 else f'{v / 1000:,.0f} s')

# ---------------------------------------------------------------- 1. the life of one request
def fig_timeline():
    b = text(20, 22, 'One streamed request, as the user sees it', 't-title', 'start')
    y = 70; x0 = 40
    segs = [(x0, 110, 'box', 'waiting'), (150, 120, 'box-2', 'prefill')]
    for x, w, cls, lab in segs:
        b += rect(x, y, w, 34, cls) + text(x + w / 2, y + 22, lab)
    toks = [270, 310, 352, 392, 470, 512, 552, 596, 640, 690]
    for i, x in enumerate(toks):
        b += rect(x, y, 30, 34, 'box-1', 4) + text(x + 15, y + 22, f't{i + 1}', 't-tick')
    b += text(441, y + 22, '...', 't-note')
    b += line(x0, y - 12, x0, y + 110, 'drop') + text(x0, y - 18, 'request sent', 't-tick', 'start')
    # TTFT
    b += line(x0, y + 52, 300, y + 52, 'edge', True) + line(300, y + 52, x0, y + 52, 'edge', True)
    b += text(175, y + 74, 'TTFT: time to first token (waiting + prefill)', 't-tick')
    b += line(300, y + 40, 300, y + 110, 'drop')
    # ITL
    b += line(300, y - 8, 340, y - 8, 'edge') + line(300, y - 13, 300, y - 3, 'edge') + line(340, y - 13, 340, y - 3, 'edge')
    b += text(320, y - 18, 'ITL', 't-val')
    b += text(560, y - 18, 'ITL: the gap between two streamed tokens', 't-tick')
    # E2E
    b += line(x0, y + 104, 720, y + 104, 'edge', True) + line(720, y + 104, x0, y + 104, 'edge', True)
    b += text(380, y + 122, 'E2E latency: sent to last token', 't-tick')
    b += line(720, y - 6, 720, y + 110, 'drop')
    b += text(380, y + 158, 'TPOT = (E2E - TTFT) / (output tokens - 1): the average gap, one number per request', 't-note')
    return svg(y + 175, b, 'Timeline of one request: waiting, prefill, then tokens t1 to t10. TTFT runs from sending to the first token; ITL is one gap between tokens; end-to-end latency runs to the last token; TPOT is the average gap.')
f['timeline'] = fig_timeline()


# ---------------------------------------------------------------- loops: closed vs open
def fig_loops():
    b = text(190, 22, 'Closed loop: wait for the reply, then send', 't-title') + text(575, 22, 'Open loop: send on a schedule', 't-title')
    for k in range(3):
        y = 54 + k * 46
        b += text(24, y + 18, f'user {k + 1}', 't-tick', 'start')
        x = 80
        for w in ((60, 30), (90, 30), (50, 30))[k:] + ((60, 30), (90, 30), (50, 30))[:k]:
            b += rect(x, y, w[0], 26, 'box-1', 4); x += w[0] + 6
    b += text(190, 212, 'one request in flight per user: a slow server slows the arrivals', 't-tick')
    arr = [(410, 70), (432, 110), (470, 60), (505, 120), (522, 80), (580, 100), (612, 70), (640, 90), (690, 60)]
    lanes = [0, 0, 0, 0]
    b += line(400, 56, 740, 56, 'axis') + text(745, 60, 't', 't-tick', 'start')
    for t, w in arr:
        lane = next(k for k in range(4) if lanes[k] <= t)
        lanes[lane] = t + w + 4
        y = 70 + lane * 34
        b += dot(t, 56, 's2', 3.5) + line(t, 56, t, y, 'drop') + rect(t, y, min(w, 738 - t), 24, 'box-2', 4)
    b += text(575, 212, 'requests arrive whether or not earlier ones have finished', 't-tick')
    b += line(385, 36, 385, 220, 'axis')
    return svg(228, b, 'Closed loop: each simulated user sends the next request only after the previous reply. Open loop: requests arrive at scheduled times and overlap when the server is slow.')
f['loops'] = fig_loops()


# ---------------------------------------------------------------- quantization grid diagram
def fig_quant_grid():
    b = text(20, 22, 'Where the scales live', 't-title', 'start')
    cw, ch = 17, 18
    for panel, (x0, title, sub) in enumerate(((30, 'per output channel', 'one scale per row'),
                                              (400, 'per group (g128)', 'one scale per row per 128 inputs'))):
        b += text(x0 + 8 * cw, 48, title, 't-note') + text(x0 + 8 * cw, 66, sub, 't-tick')
        for r in range(6):
            for c in range(16):
                cls = 'box-1' if panel == 0 or (c // 4) % 2 == 0 else 'box-3'
                b += rect(x0 + c * cw, 80 + r * ch, cw - 2, ch - 2, cls, 2)
            n = 1 if panel == 0 else 4
            for g in range(n):
                cls = 'box-2'
                b += rect(x0 + 16 * cw + 10 + g * 14, 80 + r * ch, 12, ch - 2, cls, 2)
        b += text(x0 + 16 * cw + 10 + (7 if panel == 0 else 28), 74, 'scales', 't-tick')
        b += text(x0 + 8 * cw, 80 + 6 * ch + 22, 'inputs (columns) ->', 't-tick')
    b += text(380, 232, 'A scale maps the biggest weight it covers to the largest code (127 for INT8, 7 for INT4).', 't-tick')
    b += text(380, 252, 'One large weight squeezes every other weight under the same scale onto a few codes; smaller groups contain the damage.', 't-tick')
    return svg(264, b, 'Per-channel quantization keeps one scale per output row. Group quantization keeps one scale per row for each group of 128 inputs (here drawn as groups of 4), so an outlier only affects its own group.')
f['quant_grid'] = fig_quant_grid()


# ---------------------------------------------------------------- routing diagram
def fig_router():
    b = rect(300, 20, 160, 44, 'box-2') + text(380, 47, 'router')
    b += text(472, 47, 'which replica gets this request?', 't-tick', 'start')
    for k in range(4):
        x = 30 + k * 182
        b += line(380, 64, x + 80, 110, 'edge', True)
        b += rect(x, 110, 160, 92, 'box') + text(x + 80, 132, f'replica {k + 1}', 't-note')
        b += text(x + 80, 152, 'queue + running batch', 't-tick')
        b += rect(x + 12, 164, 136, 26, 'box-1', 4) + text(x + 80, 182, 'prefix cache (LRU)', 't-tick')
    b += text(380, 230, 'load says: pick the shortest queue.  cache says: pick the replica that already holds the prefix.', 't-tick')
    return svg(244, b, 'A router in front of four replicas, each with its own queue and its own prefix cache.')
f['router'] = fig_router()


# ---------------------------------------------------------------- capacity (H100 arithmetic and simulation)
def fig_capacity():
    c = J('capacity')
    m = c['memory']
    out = {}
    # memory budget bars
    b = ''
    rows = [('BF16 weights + BF16 KV', m['weights_bf16'], m['kv_space'], m['kv_tokens']),
            ('FP8 weights + FP8 KV', m['weights_fp8'], m['kv_space_fp8'], m['kv_tokens_fp8'])]
    sc = 600 / 80e9
    for i, (lab, wb, kv, toks) in enumerate(rows):
        y = 40 + i * 92
        b += text(30, y - 8, lab, 't-note', 'start')
        x = 30
        for cls, v, name in (('s2', wb, 'weights'), ('s3', m['reserve'], 'reserve'), ('s1', kv, 'KV cache'),
                             ('waste', 80e9 * (1 - m['util']), '')):
            w = v * sc
            b += mark(f'{name} {v / 1e9:.1f} GB', rect(x, y, w, 30, cls, 3))
            lab_ = f'{name} {v / 1e9:.1f} GB'
            if name and name != 'reserve' and w > 7 * len(lab_) + 8:
                b += text(x + w / 2, y + 20, lab_, 't-tick')
            x += w
        b += text(30, y + 50, f'KV space holds {toks:,.0f} tokens ({m["kv_bytes_per_token"] // (1 if i == 0 else 2) // 1024} KiB each)', 't-tick', 'start')
    b += text(30 + 600 * m['util'] - 6, 200, f'{m["util"]:.0%} of 80 GB (vLLM default)', 't-tick', 'end') + line(30 + 600 * m['util'], 30, 30 + 600 * m['util'], 206, 'drop')
    b += legend([('weights', 's2', 'box'), (f'reserve {m["reserve"] / 1e9:.0f} GB (assumed)', 's3', 'box'), ('KV cache', 's1', 'box'), ('left unused', 'waste', 'box')], 30, 230)
    out['memory_budget'] = svg(244, b, f'H100 80 GB budget for Llama 3.1 8B: BF16 leaves {m["kv_space"] / 1e9:.1f} GB for KV ({m["kv_tokens"]:,.0f} tokens); FP8 weights and KV leave {m["kv_space_fp8"] / 1e9:.1f} GB ({m["kv_tokens_fp8"]:,.0f} tokens).')

    # roofline decode cost per token vs batch
    d = c['roofline']['decode']
    ax = Axes(90, 40, 600, 200, (1, 512), (10, 6000), xlog=True, ylog=True)
    b = ax.frame([1, 8, 32, 64, 128, 256, 512], [10, 30, 100, 300, 1000, 3000], str, lambda v: f'{v:,} us',
                 'sequences decoded together (batch)', 'time per generated token (roofline, H100)')
    pts = [(ax.X(r['B']), ax.Y(r['us_per_token'])) for r in d]
    b += poly(pts, 'l1') + ''.join(mark(f'batch {r["B"]}: {r["us_per_token"]:.0f} us per token, step {r["step_ms"]:.1f} ms', dot(x, y)) for (x, y), r in zip(pts, d))
    pf = c['roofline']['prefill_us_per_token']
    b += line(ax.x0, ax.Y(pf), ax.x0 + ax.w, ax.Y(pf), 'l2') + text(ax.x0 + ax.w, ax.Y(pf) - 8, f'prefill: {pf:.1f} us per prompt token (compute bound)', 't-tick', 'end')
    for r in d:
        if r['B'] in (1, 64, 512):
            dy = 18 if r['B'] == 1 else -8
            b += text(ax.X(r['B']) + 8, ax.Y(r['us_per_token']) + dy, f'{r["us_per_token"]:,.0f} us', 't-val', 'start')
    out['roofline'] = svg(300, b, 'Roofline time per generated token for Llama 3.1 8B on an H100 falls from 4,838 microseconds at batch 1 to 53 at batch 512, but never reaches the 15 microseconds a prompt token costs in prefill.')

    # simulated SLO attainment vs rate
    ax = Axes(90, 80, 600, 190, (10, 100), (0, 100))
    b = ax.frame([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], [0, 25, 50, 75, 100], str, lambda v: f'{v}%',
                 'requests per second offered to one H100', 'requests inside the SLO (TTFT <= 500 ms and TPOT <= 50 ms)')
    for name, cls in (('BF16', 'l2'), ('FP8', 'l1')):
        rows = [r for r in c['sim'][name]['rows'] if r['rate'] <= 100]
        pts = [(ax.X(r['rate']), ax.Y(r['slo_attainment'] * 100)) for r in rows]
        b += poly(pts, cls) + ''.join(mark(f'{name} {r["rate"]} req/s: {r["slo_attainment"] * 100:.1f}% inside SLO, TTFT p99 {r["ttft"]["p99"]:.0f} ms',
                                           dot(x, y, 's' + cls[1], 3.5)) for (x, y), r in zip(pts, rows))
        cap = c['sim'][name]['max_rate_99']
        b += text(ax.X(cap) + 8, ax.Y(99) + 22, f'{name}: {cap} req/s', 't-val', 'start')
    b += line(ax.x0, ax.Y(99), ax.x0 + ax.w, ax.Y(99), 'base-line') + text(ax.x0 + ax.w + 4, ax.Y(99) + 4, '99%', 't-tick', 'start')
    b += legend([('BF16 weights + KV', 'l2', 'line'), ('FP8 weights + KV', 'l1', 'line')], 90, 24)
    out['capacity_sim'] = svg(330, b, f'Simulated share of requests inside the SLO against offered load on one H100 at the roofline: BF16 holds 99% up to {c["sim"]["BF16"]["max_rate_99"]} requests per second, FP8 up to {c["sim"]["FP8"]["max_rate_99"]}.')

    # cost split
    k = c['cost']
    b = text(20, 24, 'Where one GPU-second goes at the BF16 operating point (roofline)', 't-title', 'start')
    tot = k['busy_s']
    w1, w2 = 600 * k['prefill_s'] / tot, 600 * k['decode_s'] / tot
    b += rect(40, 44, w1, 34, 's2', 3) + rect(40 + w1, 44, w2, 34, 's1', 3)
    b += text(40 + w1 / 2, 66, f'prompt tokens {k["prefill_s"] / tot:.0%}', 't-tick') + text(40 + w1 + w2 / 2, 66, f'generated tokens {k["decode_s"] / tot:.0%}', 't-tick')
    b += text(40, 106, f'Per token: prompt {k["in_us"]:.1f} us, generated {k["out_us"]:.1f} us ({k["out_us"] / k["in_us"]:.1f}x)', 't-note', 'start')
    for i, r in enumerate(k['rows']):
        y = 140 + i * 26
        b += text(40, y, f'${r["price"]:.2f} per GPU-hour:', 't-tick', 'start')
        b += text(220, y, f'${r["input_per_M"]:.3f} per 1M prompt tokens', 't-tick', 'start')
        b += text(450, y, f'${r["output_per_M"]:.3f} per 1M generated tokens', 't-tick', 'start')
    b += text(40, 140 + 3 * 26 + 6, 'Floor prices: 100% busy and at the roofline. Divide by your real utilisation and efficiency.', 't-tick', 'start')
    out['cost_split'] = svg(140 + 3 * 26 + 20, b, f'At the operating point the GPU spends {k["prefill_s"] / tot:.0%} of its time on prompt tokens and {k["decode_s"] / tot:.0%} on generated tokens; a generated token costs {k["out_us"] / k["in_us"]:.1f} times a prompt token.')
    return out
f.update(fig_capacity())


# ---------------------------------------------------------------- quantization results
def fig_quant():
    q = J('quant')
    rows = q['rows']
    base = rows[0]['ppl']
    b = text(20, 22, f'Qwen2.5-0.5B on WikiText-2 ({q["tokens"]:,} tokens): perplexity, lower is better', 't-title', 'start')
    x0, W = 250, 330
    mx = 25.0
    y = 44
    for r in rows:
        v = r['ppl']; w = W * min(v, mx) / mx
        cls = 's1' if v - base < 0.25 else ('s3' if v - base < 3 else 's2')
        b += text(x0 - 10, y + 14, r['scheme'], 't-tick', 'end')
        b += mark(f'{r["scheme"]}: perplexity {v:.3f} ({v - base:+.3f}), {r["weight_MiB"]:.0f} MiB, top-1 agreement {r["top1"] * 100:.1f}%',
                  rect(x0, y + 2, w, 16, cls, 3))
        lab = f'{v:.2f} ({v - base:+.2f})' + (', off scale' if v > mx else '')
        b += text(x0 + w + 6, y + 15, lab, 't-tick', 'start')
        y += 24
    b += line(x0 + W * base / mx, 38, x0 + W * base / mx, y, 'drop')
    b += text(x0 + W * base / mx, y + 16, f'BF16 {base:.2f}', 't-tick')
    b += legend([('within 0.25 of BF16', 's1', 'box'), ('within 3', 's3', 'box'), ('worse', 's2', 'box')], x0 - 200, y + 40)
    return svg(y + 52, b, 'Measured WikiText-2 perplexity of Qwen2.5-0.5B under each quantization scheme. 8-bit weights cost almost nothing; INT4 per channel and INT3 are much worse; group-128 and AWQ scales recover most of the INT4 loss; INT8 activations with one scale per tensor collapse.')
f['quant_ppl'] = fig_quant()


def fig_quant_bytes():
    sizes = [('BF16', 16), ('FP8 / INT8', 8), ('INT4 g128', 4.25)]
    p = 8.03e9
    b = text(20, 22, 'Llama 3.1 8B weights, and the decode step they set (H100, batch 1, roofline)', 't-title', 'start')
    for i, (n, bits) in enumerate(sizes):
        y = 50 + i * 44
        gb = p * bits / 8 / 1e9
        w = 330 * gb / 16.06
        b += text(140, y + 18, n, 't-note', 'end') + rect(150, y, w, 26, 's1' if i else 's2', 3)
        b += text(158 + w, y + 18, f'{gb:.1f} GB: {gb / 3.35:.1f} ms per step at 3.35 TB/s', 't-tick', 'start')
    b += text(20, 200, 'Weight-only formats shrink what decode must read; INT4 also needs a scale (and zero) per group of 128, about 4.25 bits per weight.', 't-tick', 'start')
    return svg(214, b, 'Llama 3.1 8B weighs 16.1 GB in BF16, 8.0 GB in FP8 or INT8 and about 4.3 GB in INT4 with group scales; reading them at 3.35 TB/s takes 4.8, 2.4 and 1.3 ms.')
f['quant_bytes'] = fig_quant_bytes()


# ---------------------------------------------------------------- queueing (laptop-calibrated simulation)
def fig_queueing():
    q = J('queueing'); out = {}
    # averages: TTFT histogram with markers
    a = q['averages']; xs = a['ttft']
    mx = max(xs); nb = 40; bw = mx / nb
    counts = [0] * nb
    for x in xs: counts[min(nb - 1, int(x / bw))] += 1
    ax = Axes(70, 50, 620, 170, (0, mx), (0, max(counts) * 1.1))
    b = ax.frame([0, 200, 400, 600], [], ms, str, f'time to first token (simulated, {a["rate"]} requests/s)')
    for k, c in enumerate(counts):
        b += rect(ax.X(k * bw) + 1, ax.Y(c), ax.X(bw) - ax.X(0) - 2, ax.y0 + ax.h - ax.Y(c), 's1', 1)
    for key, lab, dy in (('p50', 'p50', 0), ('mean', 'mean', 16), ('p90', 'p90', 0), ('p99', 'p99', 0)):
        v = a[key]; x = ax.X(v)
        b += line(x, 40, x, ax.y0 + ax.h, 'base-line') + text(x + 4, 36 + dy, f'{lab} {v:.0f} ms', 't-val', 'start')
    out['averages'] = svg(270, b, f'Histogram of simulated TTFT at {a["rate"]} requests per second: p50 {a["p50"]:.0f} ms, mean {a["mean"]:.0f} ms, p90 {a["p90"]:.0f} ms, p99 {a["p99"]:.0f} ms, maximum {a["max"]:.0f} ms. The tail is long.')

    # M/M/1
    rows = q['mm1']
    ax = Axes(90, 66, 600, 200, (0, 1), (100, 50000), ylog=True)
    b = ax.frame([0, 0.2, 0.4, 0.6, 0.8, 1.0], [100, 300, 1000, 3000, 10000, 30000], lambda v: f'{v:.0%}', ms,
                 'utilisation rho = arrival rate / service rate', 'time in the system (service time 100 ms)')
    for key, fkey, cls, lab in (('mean_ms', 'mean_formula_ms', 'l1', 'mean'), ('p99_ms', 'p99_formula_ms', 'l2', 'p99')):
        pts = [(ax.X(r_), ax.Y(1000 / (10 * (1 - r_)) * (1 if key == 'mean_ms' else 4.60517))) for r_ in [k / 200 for k in range(0, 199)]]
        b += poly(pts, cls)
        for r in rows:
            b += mark(f'rho {r["rho"]}: {lab} simulated {r[key]:.0f} ms, formula {r[fkey]:.0f} ms', dot(ax.X(r['rho']), ax.Y(r[key]), 's' + cls[1], 4))
    b += legend([('mean: 1/(mu - lambda)', 'l1', 'line'), ('p99: ln(100)/(mu - lambda)', 'l2', 'line')], 90, 24) + text(690, 24, 'lines: formula   dots: simulated', 't-tick', 'end')
    out['mm1'] = svg(326, b, 'M/M/1 queue: mean and p99 time in the system against utilisation. Simulated points sit on the formulas. At 90% utilisation the mean is 10 times the service time and the p99 is 46 times.')

    # sweep: TTFT p99 and TPOT p99 vs rate (two panels)
    sw = q['sweep']; slo = (1000, 50)
    b = ''
    for panel, (key, lim, ytl, title) in enumerate((('ttft', slo[0], [30, 100, 300, 1000, 3000, 10000, 100000], 'TTFT'),
                                                     ('tpot', slo[1], [10, 20, 50, 100, 200], 'TPOT'))):
        x0 = 70 + panel * 370
        ax = Axes(x0, 50, 290, 190, (0, 24), ((30, 200000) if panel == 0 else (10, 200)), ylog=True)
        b += ax.frame([0, 6, 12, 18, 24], ytl, str, ms, 'requests per second', f'{title} (simulated)')
        for p_, cls in (('p50', 'l1'), ('p99', 'l2')):
            pts = [(ax.X(r['rate']), ax.Y(r[key][p_])) for r in sw]
            b += poly(pts, cls) + ''.join(mark(f'{r["rate"]}/s: {title} {p_} {r[key][p_]:.0f} ms', dot(x, y, 's' + cls[1], 3)) for (x, y), r in zip(pts, sw))
        b += line(ax.x0, ax.Y(lim), ax.x0 + ax.w, ax.Y(lim), 'base-line') + text(ax.x0 + ax.w, ax.Y(lim) - 6, f'SLO {lim} ms', 't-tick', 'end')
    b += legend([('p50', 'l1', 'line'), ('p99', 'l2', 'line')], 300, 22)
    out['sweep'] = svg(300, b, 'Simulated TTFT and TPOT percentiles against request rate on the laptop-calibrated engine. TPOT rises steadily as the batch grows; TTFT stays low until about 8 requests per second, then the queue explodes.')

    # goodput vs throughput
    ax = Axes(80, 70, 610, 190, (0, 24), (0, 12))
    b = ax.frame([0, 4, 8, 12, 16, 20, 24], [0, 2, 4, 6, 8, 10, 12], str, str, 'requests per second offered', 'requests per second')
    b += poly([(ax.X(0), ax.Y(0)), (ax.X(12), ax.Y(12))], 'base-line')
    for key, cls, lab in (('throughput_rps', 'l1', 'throughput: finished'), ('goodput_rps', 'l2', 'goodput: finished inside the SLO')):
        pts = [(ax.X(r['rate']), ax.Y(r[key])) for r in sw]
        b += poly(pts, cls) + ''.join(mark(f'{r["rate"]}/s offered: {key} {r[key]:.2f}', dot(x, y, 's' + cls[1], 3.5)) for (x, y), r in zip(pts, sw))
    b += legend([('throughput: requests finished', 'l1', 'line'), ('goodput: finished inside the SLO', 'l2', 'line')], 80, 24)
    out['goodput'] = svg(320, b, 'Simulated throughput keeps rising to about 8.4 requests per second and then stays flat, while goodput peaks near 6 and falls to zero: past that point the server is busy finishing requests that already missed the SLO.')

    # coordinated omission: p99 and p99.9 bars per method
    co = q['coordinated_omission']
    names = list(co)
    b = ''
    mxv = max(v['p999'] for v in co.values())
    for i, n_ in enumerate(names):
        y = 30 + i * 52
        b += text(250, y + 14, n_, 't-tick', 'end')
        for j, (key, cls) in enumerate((('p99', 's1'), ('p999', 's2'))):
            w = 400 * co[n_][key] / mxv
            b += mark(f'{n_}: {key} {co[n_][key] / 1000:.2f} s', rect(260, y + j * 18, w, 15, cls, 3))
            b += text(266 + w, y + j * 18 + 12, f'{co[n_][key] / 1000:.2f} s', 't-tick', 'start')
    b += legend([('p99', 's1', 'box'), ('p99.9', 's2', 'box')], 260, 30 + len(names) * 52 + 6)
    out['omission'] = svg(30 + len(names) * 52 + 20, b, 'Simulated end-to-end latency percentiles reported by four ways of measuring the same server with a 6-second stall. The closed-loop client that records from its actual send time reports a p99 as if nothing happened.')
    return out
f.update(fig_queueing())


# ---------------------------------------------------------------- routing
def fig_routing():
    r = J('route'); out = {}
    pols = ['random', 'round_robin', 'least_outstanding', 'power_of_two', 'prefix_hash', 'cache_aware', 'cache_aware_abs8']
    labels = {'cache_aware_abs8': 'cache_aware (abs 8)'}
    b = ''
    for panel, scen in enumerate(('zipf', 'hot')):
        y0 = 40 + panel * 272
        b += text(20, y0 - 14, 'Zipf popularity over 200 prefixes' if scen == 'zipf' else 'one hot prefix takes 50% of traffic', 't-title', 'start')
        mx = max(v['ttft_p99'] for v in r[scen].values())
        for i, p_ in enumerate(pols):
            v = r[scen][p_]; y = y0 + i * 30
            b += text(170, y + 14, labels.get(p_, p_), 't-tick', 'end')
            w = 200 * v['hit_rate']
            b += mark(f'{p_}: hit rate {v["hit_rate"] * 100:.1f}%', rect(180, y + 2, w, 16, 's1', 3)) + text(186 + w, y + 15, f'{v["hit_rate"] * 100:.0f}%', 't-tick', 'start')
            lw = 210 * math.log10(max(v['ttft_p99'], 100) / 100) / math.log10(max(mx, 1000) / 100)
            b += mark(f'{p_}: TTFT p99 {v["ttft_p99"]:.0f} ms', rect(440, y + 2, lw, 16, 's2', 3)) + text(446 + lw, y + 15, ms(v['ttft_p99']), 't-tick', 'start')
        b += text(180, y0 + 7 * 30 + 12, 'prefix cache hit rate', 't-tick', 'start') + text(440, y0 + 7 * 30 + 12, 'TTFT p99 (log scale from 100 ms)', 't-tick', 'start')
    out['routing'] = svg(552, b, 'Simulated hit rate and TTFT p99 for seven routing policies across four replicas, for Zipf traffic and for one hot prefix. Pure prefix hashing gets the most hits but overloads one replica; cache-aware routing with a tight balance threshold gets most of the hits without the tail.')
    return out
f.update(fig_routing())


# ---------------------------------------------------------------- LoRA measured
def fig_lora():
    m = J('lora')['measured']
    rows = [('base model only', m['base']), ('1 adapter shared by all 32', m['same']), ('32 adapters, gathered + bmm', m['gather']), ('32 adapters, Python loop', m['loop'])]
    cap = 3 * rows[2][1]['min_ms']
    b = ''
    for i, (lab, v) in enumerate(rows):
        y = 24 + i * 40
        x = v['min_ms']; w = 330 * min(x, cap) / cap
        b += text(230, y + 15, lab, 't-tick', 'end') + mark(f'{lab}: fastest {x:.1f} ms, median {v["median_ms"]:.1f} ms', rect(240, y + 2, w, 20, 's2' if x > cap else 's1', 3))
        b += text(246 + w, y + 17, f'{x:,.1f} ms' + ('  (off the scale)' if x > cap else f'  ({x / rows[0][1]["min_ms"]:.2f}x)'), 't-tick', 'start')
    out = svg(24 + 4 * 40 + 6, b, 'Measured decode step of Qwen2.5-0.5B with 32 requests and rank-16 adapters on all linear layers: base only, one shared adapter, 32 gathered adapters, and a loop over 32 adapters.')
    return out
f['lora_measured'] = fig_lora()


# ---------------------------------------------------------------- cold start
def fig_cold():
    c = J('coldstart'); m = c['median']
    phases = [('import', 'import', 's3'), ('read_build', 'read + build', 's2'), ('to_gpu', 'copy to GPU', 's1'), ('first_pass', 'first pass', 's4')]
    tot = sum(m[k] for k, _, _ in phases)
    b = text(20, 22, f'Measured: a fresh process until its first token, Qwen2.5-0.5B (median of 5): {tot:.1f} s', 't-title', 'start')
    x = 30
    for k, lab, cls in phases:
        w = 700 * m[k] / tot
        b += mark(f'{lab}: {m[k]:.2f} s', rect(x, 40, w, 30, cls, 2))
        x += w
    y = 92
    for i, (k, lab, cls) in enumerate(phases):
        xx = 30 + i * 180
        b += rect(xx, y - 11, 12, 12, cls, 2) + text(xx + 18, y, f'{lab}: {m[k]:.1f} s', 't-tick', 'start')
    return svg(110, b, f'Measured cold start of a new process serving Qwen2.5-0.5B on the laptop, median of five: import {m["import"]:.1f} s, read and build {m["read_build"]:.1f} s, copy to GPU {m["to_gpu"]:.1f} s, first pass {m["first_pass"]:.1f} s.')
f['coldstart'] = fig_cold()


# ---------------------------------------------------------------- autoscaling timeline
def fig_autoscale():
    a = J('autoscale'); out = {}
    keys = [('static', 'l3'), ('load, cold 180s', 'l2'), ('load, cold 20s', 'l1')]
    ax = Axes(80, 80, 610, 140, (0, 600), (30, 100000), ylog=True)
    b = ax.frame([0, 120, 240, 360, 480, 600], [100, 1000, 10000, 100000], lambda v: f'{v} s', ms, 'time (traffic jumps from 6 to 24 requests/s at 120 s)', 'TTFT p99 per 20 s window (simulated)')
    for k, cls in keys:
        w = [x for x in a[k]['windows'] if x['p99'] == x['p99']]
        pts = [(ax.X(x['t'] + 10), ax.Y(x['p99'])) for x in w]
        b += poly(pts, cls)
    b += line(ax.X(120), 40, ax.X(120), ax.y0 + ax.h, 'drop')
    ax2 = Axes(80, 300, 610, 90, (0, 600), (0, 8))
    b += ax2.frame([0, 120, 240, 360, 480, 600], [0, 2, 4, 6, 8], lambda v: f'{v} s', str, '', 'replicas serving')
    for k, cls in keys + [('busy, cold 20s', 'l4')]:
        tl = a[k]['timeline']
        pts = []
        for x in tl:
            pts += [(ax2.X(x['t']), ax2.Y(x['ready']))]
        b += poly(pts, cls)
    b += legend([('never scale', 'l3', 'line'), ('scale on load, 180 s cold start', 'l2', 'line')], 80, 20, gap=40)
    b += legend([('scale on load, 20 s cold start', 'l1', 'line'), ('scale on GPU busy % (bottom panel only)', 'l4', 'line')], 80, 42, gap=40)
    out['autoscale'] = svg(430, b, 'Simulated autoscaling after a traffic jump: TTFT p99 per 20-second window and the number of serving replicas, for no scaling, load-based scaling with 180-second and 20-second cold starts, and scaling on GPU busy percentage.')
    return out
f.update(fig_autoscale())


# ---------------------------------------------------------------- failures
def fig_failures():
    d = J('failures'); out = {}
    rt = d['retry']
    ax = Axes(80, 70, 610, 170, (0, 300), (0, 100))
    b = ax.frame([0, 60, 120, 180, 240, 300], [0, 25, 50, 75, 100], lambda v: f'{v} s', lambda v: f'{v}%', 'arrival time (the engine freezes for 10 s at 60 s)', 'users answered within 10 s (simulated)')
    for k, cls in (('no retries', 'l3'), ('retry at once, no cancel', 'l2'), ('retry at once, cancel', 'l1')):
        pts = [(ax.X(w['t'] + 5), ax.Y(w['success'] * 100)) for w in rt[k]['windows']]
        b += poly(pts, cls)
    b += legend([('no retries', 'l3', 'line'), ('retry at once, server keeps working', 'l2', 'line'), ('retry, server cancels abandoned', 'l1', 'line')], 80, 24, gap=14)
    out['retry'] = svg(300, b, 'Simulated share of users answered within the timeout, by arrival time, after a 10-second freeze. Without retries the server recovers about 10 seconds after the freeze; with immediate retries and no cancellation it never recovers; with cancellation it recovers at once.')

    pr = d['preemption']
    ax = Axes(80, 50, 270, 170, (2, 6), (0, max(r['preemptions'] for r in pr) * 1.15))
    b = ax.frame([2, 3, 4, 5, 6], [0, 500, 1000], str, lambda v: f'{v:,.0f}', 'requests per second', 'preemptions (simulated)')
    pts = [(ax.X(r['rate']), ax.Y(r['preemptions'])) for r in pr]
    b += poly(pts, 'l2') + ''.join(dot(x, y, 's2', 3.5) for x, y in pts)
    ax2 = Axes(450, 50, 270, 170, (2, 6), (0, max(r['itl_max'] for r in pr) * 1.1))
    b += ax2.frame([2, 3, 4, 5, 6], [0, 1000, 2000, 3000, 4000, 5000], str, ms, 'requests per second', 'longest pause in a stream (ITL max)')
    pts = [(ax2.X(r['rate']), ax2.Y(r['itl_max'])) for r in pr]
    b += poly(pts, 'l1') + ''.join(dot(x, y, 's1', 3.5) for x, y in pts)
    out['preempt'] = svg(280, b, 'Simulated with a deliberately small KV cache: no preemptions up to 3 requests per second, then about a thousand, and the longest pause inside a streamed answer jumps from 0.2 seconds to several seconds.')
    return out
f.update(fig_failures())

(R / 'figures.json').write_text(json.dumps(f) + '\n')
print('figures:', len(f), ', '.join(f))
