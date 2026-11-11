"""Assemble LLM Inference Part 8 from article.md and the saved Colab run.

    python3 code/vllm-colab/assemble.py                 # reads results/colab_run.json (+ results/run/ logs)
    python3 code/vllm-colab/assemble.py --results X.json --out /tmp/p8.md   # test against another file

Every {{R:expr}} is replaced by a value computed from the results; every {{FIG:key|caption}} by an SVG drawn
from the same results with the site's theme classes. Refuses em dashes, banned words, missing images,
unresolved placeholders, and (with --strict) any ASD-STE100 problem in our prose.
"""
import argparse, importlib.util, json, math, re, sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'content/writings/llm-inference-8-vllm-on-colab.md'
BANNED = ['Alammar', 'ChatGPT', 'Miro ', 'lecture', 'YouTube', 'Vizuara', 'Claude', 'plain English', 'simple words']

ap = argparse.ArgumentParser()
ap.add_argument('--results', default=str(HERE / 'results/colab_run.json'))
ap.add_argument('--logs', default=str(HERE / 'results/run/vllm_logs'))
ap.add_argument('--out', default=str(OUT))
ap.add_argument('--strict', action='store_true')
args = ap.parse_args()
R = json.loads(Path(args.results).read_text())
assert 'T4' in R['gpu']['name'], 'The prose of this part is written for a T4 run.'

# ---------------------------------------------------------------- value helpers used inside {{R:...}}
def B(label, key):
    return R['bench'][label][key]
def S(name, key):
    return next(s for s in R['starts'] if s['name'] == name)[key]
def n0(v): return f'{v:,.0f}'
def n1(v): return f'{v:,.1f}'
def n2(v): return f'{v:,.2f}'
def x(v): return f'{v:.1f} times' if v >= 1.95 else f'{v:.2f} times'
def about(v): return f'about {v:.0f} times' if v >= 2.95 else f'about {v:.1f} times'   # the way people say it
def kv_pred(): return R['kv_math']['kv_cache_memory_gib'] * 2**30 / R['kv_math']['bytes_per_token']
def pct(v): return f'{100 * v:.1f}%' if v < 0.995 or v > 1 else f'{100 * v:.2f}%'
def ms(v): return f'{v:,.0f} ms' if v < 1000 else f'{v / 1000:,.2f} s'
def s(v): return f'{v:,.0f} s' if v >= 10 else f'{v:.1f} s'
def W_GB(): return R['starts'][0]['weights_gib_and_load_s'][0] * 2**30 / 1e9
CONC = [1, 2, 4, 8, 16, 32, 64, 128]
RATES = [0.25, 0.5, 0.75, 0.9, 1.0, 1.2]

def loglines():
    keep = ('attention backend', 'Model loading took', 'torch.compile took', 'Available KV cache memory',
            'KV cache size', 'Graph capturing finished', 'init engine')
    out = []
    for line in (Path(args.logs) / 'default.log').read_text(errors='replace').splitlines():
        if any(k in line for k in keep):
            line = re.sub(r'^.*?(INFO|WARNING)\s+\S+\s+\S+\s+(\[[^\]]*\]\s*)?', '', line)   # drop prefix and timestamp
            line = re.sub(r' out of potential backends: .*', '', line)
            if line.startswith('CUDA graph memory profiling'):
                continue
            out.append(line.strip())
    return '\n'.join(dict.fromkeys(out))

def single_rows():
    return '\n'.join(f"| {i + 1} | {ms(r['ttft_ms'])} | {r['tpot_ms']:.1f} ms | {r['tokens_per_s']:.0f} |"
                     for i, r in enumerate(R['single_user_stream']))

def conc_rows():
    base = B('conc_1', 'output_throughput')
    return '\n'.join(f"| {c} | {n0(B(f'conc_{c}', 'output_throughput'))} | {B(f'conc_{c}', 'output_throughput') / base:.1f}x | "
                     f"{B(f'conc_{c}', 'median_tpot_ms'):.1f} ms | {1000 / B(f'conc_{c}', 'median_tpot_ms'):.0f} | "
                     f"{ms(B(f'conc_{c}', 'median_ttft_ms'))} | {ms(B(f'conc_{c}', 'p99_ttft_ms'))} |" for c in CONC)

def per_user_claim():
    ok = [c for c in CONC if 1000 / B(f'conc_{c}', 'median_tpot_ms') >= 20 and B(f'conc_{c}', 'p99_ttft_ms') <= 1000]
    if not ok:
        return 'Here, no row met both a 20 tokens per second stream and a 1 second TTFT p99.'
    return (f'Here the targets are 20 tokens per second per user and a TTFT p99 under 1 second. The best row is then '
            f'{max(ok)} requests in flight, at {n0(B(f"conc_{max(ok)}", "output_throughput"))} output tokens per second.')

def rate_rows():
    rows = []
    for f in RATES:
        b = R['bench'][f'rate_{f}']
        rows.append(f"| {f:.0%} | {n2(float(b['request_rate']))} | {n2(b['request_throughput'])} | {n2(b['request_goodput'] or 0)} | "
                    f"{ms(b['median_ttft_ms'])} | {ms(b['p99_ttft_ms'])} | {b['median_tpot_ms']:.1f} ms | {b['p99_tpot_ms']:.1f} ms |")
    return '\n'.join(rows)

def fail_rows():
    F = R['failures']
    def body(k):
        t = F[k]['body']
        try:
            j = json.loads(t); t = j.get('error', {}).get('message', t) if isinstance(j.get('error'), dict) else (j.get('error') or t)
        except Exception:
            pass
        return str(t).replace('|', '/').replace('\n', ' ')[:160]
    names = {'too_long': f"The word `hello` {R['max_model_len'] + 500:,} times, over the {R['max_model_len']:,}-token limit",
             'wrong_key': 'The wrong API key', 'max_tokens_too_big': f"`max_tokens` of {R['max_model_len'] + 10:,}"}
    return '\n'.join(f"| {names[k]} | {F[k]['status']} | {body(k)} |" for k in names)

def results_rows():
    st = {x['name']: x for x in R['starts']}; k = R['kv_math']; F = R['failures']
    rows = [
        ('Setup', f"{R['gpu']['name']} ({R['gpu']['memory_gib']} GiB), vLLM {R['vllm_version']}, {R['model']}, float16, max_model_len {R['max_model_len']:,}, gpu_memory_utilization {R['gpu_memory_utilization']}"),
        ('Attention backend', st['default']['attention_backend']),
        ('Cold start / warm start', f"{st['default']['ready_s']} s / {st['no_prefix_cache']['ready_s']} s"),
        ('Weights in GPU memory', f"{st['default']['weights_gib_and_load_s'][0]} GiB"),
        ('KV cache (log)', f"{st['default']['kv_cache_memory_gib']} GiB = {n0(st['default']['kv_cache_tokens'])} tokens = {st['default']['max_concurrency'][1]} requests of {R['max_model_len']:,} tokens"),
        ('KV cache (by hand)', f"{n0(k['bytes_per_token'])} bytes per token, predicts {n0(kv_pred())} tokens"),
        ('One user, streaming', f"TTFT about {n0(sum(r['ttft_ms'] for r in R['single_user_stream']) / 3)} ms, {n1(sum(r['tpot_ms'] for r in R['single_user_stream']) / 3)} ms per token"),
        ('1 to 128 requests in flight', f"{n0(B('conc_1', 'output_throughput'))} to {n0(B('conc_128', 'output_throughput'))} output tok/s ({x(B('conc_128', 'output_throughput') / B('conc_1', 'output_throughput'))}); TPOT p50 {B('conc_1', 'median_tpot_ms'):.1f} to {B('conc_128', 'median_tpot_ms'):.1f} ms"),
        ('Capacity used for the load test', f"{n2(R['capacity_req_s'])} req/s (best closed-loop rate)"),
        ('Best goodput (TTFT under 1 s, TPOT under 100 ms)', f"{n2(max(R['bench'][f'rate_{f}']['request_goodput'] or 0 for f in RATES))} req/s, at {max(RATES, key=lambda f: R['bench'][f'rate_{f}']['request_goodput'] or 0):.0%} load"),
        ('Goodput at 100% and 120% load', f"{n2(R['bench']['rate_1.0']['request_goodput'] or 0)} and {n2(R['bench']['rate_1.2']['request_goodput'] or 0)} req/s"),
        ('Prefix caching off / on (2,048 shared tokens)', f"TTFT p50 {ms(B('prefix_off', 'median_ttft_ms'))} / {ms(B('prefix_on', 'median_ttft_ms'))} ({x(B('prefix_off', 'median_ttft_ms') / B('prefix_on', 'median_ttft_ms'))}); hit rate {pct(B('prefix_on', 'prefix_cache_hit_rate'))}"),
        ('Small KV cache vs normal', f"{n0(st['small_kv']['kv_cache_tokens'])} tokens: {n0(B('preempt_small_kv', 'preemptions'))} preemptions, TTFT p50 {ms(B('preempt_small_kv', 'median_ttft_ms'))} vs {ms(B('preempt_baseline', 'median_ttft_ms'))}"),
        ('Bad requests', f"too long: {F['too_long']['status']}; wrong key: {F['wrong_key']['status']}; max_tokens too big: {F['max_tokens_too_big']['status']}"),
        ('Server killed', f"{F['killed_mid_run']['completed']} completed, {F['killed_mid_run']['failed']} failed; /health then gave {F['killed_mid_run']['health_after']}"),
        ('FP8 weights', f"skipped: needs compute capability 8.0 or newer, the T4 is {R['gpu']['compute_capability']}"),
    ]
    return '\n'.join(f'| {a} | {b_} |' for a, b_ in rows)

def fp8_section():
    return ''          # T4 runs skip FP8 (compute capability 7.5); the L4/A100 text is written when such a run exists

# ---------------------------------------------------------------- SVG helpers (theme classes only)
DEFS = ('<defs><marker id="ah8" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
        'orient="auto-start-reverse"><path class="arrow" d="M0,0L10,5L0,10z"/></marker></defs>')
def text(x_, y, t, cls='t-note', anchor='middle'):
    return f'<text class="{cls}" x="{x_:.1f}" y="{y:.1f}" text-anchor="{anchor}">{escape(str(t))}</text>'
def rect(x_, y, w, h, cls='box-1', rx=4):
    return f'<rect class="{cls}" x="{x_:.1f}" y="{y:.1f}" width="{max(w, 0):.1f}" height="{max(h, 0):.1f}" rx="{rx}"/>'
def line(x1, y1, x2, y2, cls='edge'):
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"/>'
def poly(pts, cls='l1'):
    return f'<polyline class="{cls}" points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}"/>'
def dot(x_, y, cls='s1', r=4, title=''):
    c = f'<circle class="{cls}" cx="{x_:.1f}" cy="{y:.1f}" r="{r}"/>'
    return f'<g class="mark"><title>{escape(title)}</title>{c}</g>' if title else c
def svg(h, body, label, w=760):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{escape(label)}">{DEFS}{body}</svg>'
def legend(items, x_, y):
    out = ''
    for lab, cls in items:
        out += line(x_, y - 4, x_ + 22, y - 4, cls) + text(x_ + 28, y, lab, 't-tick', 'start')
        x_ += 40 + 7 * len(lab)
    return out

class Axes:
    def __init__(s_, x0, y0, w, h, xr, yr, xlog=False, ylog=False):
        s_.x0, s_.y0, s_.w, s_.h, s_.xr, s_.yr, s_.xlog, s_.ylog = x0, y0, w, h, xr, yr, xlog, ylog
    def _t(s_, v, r, log):
        return (math.log(v) - math.log(r[0])) / (math.log(r[1]) - math.log(r[0])) if log else (v - r[0]) / (r[1] - r[0])
    def X(s_, v): return s_.x0 + s_.w * s_._t(v, s_.xr, s_.xlog)
    def Y(s_, v): return s_.y0 + s_.h - s_.h * s_._t(min(max(v, s_.yr[0]), s_.yr[1]), s_.yr, s_.ylog)
    def frame(s_, xt, yt, xf=str, yf=str, xtitle='', ytitle=''):
        b = ''
        for v in yt:
            b += line(s_.x0, s_.Y(v), s_.x0 + s_.w, s_.Y(v), 'grid') + text(s_.x0 - 8, s_.Y(v) + 4, yf(v), 't-tick', 'end')
        for v in xt:
            b += text(s_.X(v), s_.y0 + s_.h + 18, xf(v), 't-tick')
        b += line(s_.x0, s_.y0 + s_.h, s_.x0 + s_.w, s_.y0 + s_.h, 'axis')
        if xtitle: b += text(s_.x0 + s_.w / 2, s_.y0 + s_.h + 38, xtitle, 't-tick')
        if ytitle: b += text(s_.x0 - 8, s_.y0 - 12, ytitle, 't-tick', 'start')
        return b

def nice_max(v):
    e = 10 ** math.floor(math.log10(v))
    return next(m * e for m in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10) if m * e >= v)
def ticks(top, n=4): return [top * i / n for i in range(n + 1)]
def logticks(lo, hi): return [v for k in range(-1, 7) for v in (10 ** k, 3 * 10 ** k) if lo <= v <= hi]
def fmt_k(v): return f'{v / 1000:g}k' if v >= 1000 else f'{v:g}'
def fmt_ms(v): return f'{v / 1000:g} s' if v >= 1000 else f'{v:g} ms'

FIG = {}

def fig_memory():
    st = R['starts'][0]; total = R['gpu']['memory_gib']; budget = total * R['gpu_memory_utilization']
    w_gib = st['weights_gib_and_load_s'][0]; kv = st['kv_cache_memory_gib']
    graphs = st.get('cuda_graphs_s_and_gib', [0, 0])[1]
    work = max(budget - w_gib - kv - graphs, 0)
    segs = [('weights', w_gib, 'box-2'), ('working memory', work + graphs, 'box'), ('KV cache', kv, 'box-1'),
            ('outside vLLM\'s share', total - budget, 'box-ghost')]
    x0, W = 20, 720; sc = W / total; b = text(20, 26, f"{R['gpu']['name']}: {total} GiB of memory, {R['gpu_memory_utilization']:.0%} given to vLLM", 't-title', 'start')
    x_ = x0
    for lab, v, cls in segs:
        w = v * sc; b += rect(x_, 50, w, 46, cls)
        if w > 8 * len(lab) + 12:
            b += text(x_ + w / 2, 70, lab, 't-note') + text(x_ + w / 2, 87, f'{v:.2f} GiB', 't-tick')
        else:                                   # too narrow: label under the bar
            b += line(x_ + w / 2, 97, x_ + w / 2, 104, 'axis') + text(x_ + w / 2, 117, f'{lab}: {v:.2f} GiB', 't-tick')
        x_ += w
    b += text(20, 146, f"KV cache: {n0(st['kv_cache_tokens'])} tokens of {n0(R['kv_math']['bytes_per_token'] / 1024)} KiB each, "
              f"enough for {st['max_concurrency'][1]:.1f} requests of the full {R['max_model_len']:,} tokens at once.", 't-tick', 'start')
    b += text(20, 166, 'Working memory: activations for the largest batch, measured by a profile run at startup, plus the CUDA graphs.', 't-tick', 'start')
    return svg(180, b, f'GPU memory split: weights {w_gib:.2f} GiB, working memory {work + graphs:.2f} GiB, KV cache {kv:.2f} GiB, and {total - budget:.2f} GiB left outside vLLM.')
FIG['memory'] = fig_memory()

def fig_batching():
    thr = [B(f'conc_{c}', 'output_throughput') for c in CONC]; tp = [B(f'conc_{c}', 'median_tpot_ms') for c in CONC]
    b = ''
    a = Axes(80, 44, 640, 170, (1, 128), (0, nice_max(max(thr))), xlog=True)
    b += text(20, 22, 'Whole server: output tokens per second (higher is better)', 't-title', 'start')
    b += a.frame(CONC, ticks(a.yr[1]), str, fmt_k, 'requests in flight', '')
    b += poly([(a.X(c), a.Y(v)) for c, v in zip(CONC, thr)], 'l1')
    b += ''.join(dot(a.X(c), a.Y(v), 's1', 4, f'{c} in flight: {v:,.0f} tok/s') for c, v in zip(CONC, thr))
    b += text(a.X(CONC[0]) + 8, a.Y(thr[0]) - 8, f'{thr[0]:,.0f}', 't-tick', 'start')
    b += text(a.X(CONC[-1]) - 8, a.Y(thr[-1]) - 10, f'{thr[-1]:,.0f}', 't-tick', 'end')
    a2 = Axes(80, 324, 640, 170, (1, 128), (0, nice_max(max(tp))), xlog=True)
    b += text(20, 302, 'Each user: milliseconds per token (lower is better)', 't-title', 'start')
    b += a2.frame(CONC, ticks(a2.yr[1]), str, lambda v: f'{v:g}', 'requests in flight', '')
    b += poly([(a2.X(c), a2.Y(v)) for c, v in zip(CONC, tp)], 'l2')
    b += ''.join(dot(a2.X(c), a2.Y(v), 's2', 4, f'{c} in flight: {v:.1f} ms per token') for c, v in zip(CONC, tp))
    b += text(a2.X(CONC[0]) + 8, a2.Y(tp[0]) - 8, f'{tp[0]:.1f} ms', 't-tick', 'start')
    b += text(a2.X(CONC[-1]) - 8, a2.Y(tp[-1]) + 4, f'{tp[-1]:.1f} ms', 't-tick', 'end')
    return svg(560, b, f'Output throughput rises from {thr[0]:,.0f} to {thr[-1]:,.0f} tokens per second as requests in flight go from 1 to 128, while time per token per user rises from {tp[0]:.1f} to {tp[-1]:.1f} ms.')
FIG['batching'] = fig_batching()

def fig_knee():
    p50 = [R['bench'][f'rate_{f}']['median_ttft_ms'] for f in RATES]; p99 = [R['bench'][f'rate_{f}']['p99_ttft_ms'] for f in RATES]
    lo = 10 ** math.floor(math.log10(min(p50))); hi = 10 ** math.ceil(math.log10(max(p99)))
    a = Axes(90, 40, 620, 210, (0.2, 1.25), (lo, hi), ylog=True)
    b = a.frame(RATES, logticks(lo, hi), lambda v: f'{v:.0%}', fmt_ms, 'offered load, as a share of measured capacity', 'TTFT (log scale)')
    b += line(a.X(1.0), 40, a.X(1.0), 250, 'drop') + text(a.X(1.0) + 6, 52, 'capacity', 't-tick', 'start')
    b += poly([(a.X(f), a.Y(v)) for f, v in zip(RATES, p50)], 'l1') + poly([(a.X(f), a.Y(v)) for f, v in zip(RATES, p99)], 'l2')
    b += ''.join(dot(a.X(f), a.Y(v), 's1', 4, f'{f:.0%} load: TTFT p50 {ms(v)}') for f, v in zip(RATES, p50))
    b += ''.join(dot(a.X(f), a.Y(v), 's2', 4, f'{f:.0%} load: TTFT p99 {ms(v)}') for f, v in zip(RATES, p99))
    b += legend([('TTFT p50', 'l1'), ('TTFT p99', 'l2')], 420, 20)
    return svg(300, b, f'TTFT against offered load: p50 from {ms(p50[0])} at 25% to {ms(p50[-1])} at 120%; p99 from {ms(p99[0])} to {ms(p99[-1])}.')
FIG['knee'] = fig_knee()

def fig_goodput():
    rows = [(f, float(R['bench'][f'rate_{f}']['request_rate']), R['bench'][f'rate_{f}']['request_throughput'],
             R['bench'][f'rate_{f}']['request_goodput'] or 0) for f in RATES]
    top = nice_max(max(r[1] for r in rows)); x0, W, rowh = 110, 560, 50; b = ''
    lx = 110
    for lab, cls in [('arrived', 'box-3'), ('served', 'box-1'), ('met the SLO (goodput)', 'box-2')]:
        b += rect(lx, 12, 18, 11, cls, 2) + text(lx + 24, 22, lab, 't-tick', 'start'); lx += 46 + 7 * len(lab)
    for i, (f, arr, srv, good) in enumerate(rows):
        y = 44 + i * rowh
        b += text(x0 - 12, y + 24, f'{f:.0%} load', 't-note', 'end')
        for j, (v, cls) in enumerate([(arr, 'box-3'), (srv, 'box-1'), (good, 'box-2')]):
            b += rect(x0, y + j * 13, W * v / top, 11, cls, 2)
        b += text(x0 + W * max(arr, srv) / top + 8, y + 22, f'goodput {good:.2f} req/s', 't-tick', 'start')
    b += line(x0, 44 + len(rows) * rowh, x0 + W, 44 + len(rows) * rowh, 'axis')
    for v in ticks(top, 4):
        b += text(x0 + W * v / top, 44 + len(rows) * rowh + 18, f'{v:g}', 't-tick')
    b += text(x0 + W / 2, 44 + len(rows) * rowh + 38, 'requests per second', 't-tick')
    best = max(rows, key=lambda r: r[3])
    return svg(44 + len(rows) * rowh + 52, b, f'Arrival rate, served rate and goodput at six loads. Goodput peaks at {best[3]:.2f} requests per second at {best[0]:.0%} load and falls to {rows[-1][3]:.2f} at {rows[-1][0]:.0%}.')
FIG['goodput'] = fig_goodput()

def fig_timeline():
    tl = R['overload_timeline']; T = max(p['t'] for p in tl)
    top = nice_max(max(max(p['running'], p['waiting']) for p in tl) or 1)
    a = Axes(70, 40, 620, 200, (0, T), (0, top))
    b = a.frame(ticks(T, 6), ticks(top), lambda v: f'{v:.0f}', lambda v: f'{v:.0f}', 'seconds since the run started', 'requests')
    cap = max(p['running'] for p in tl)
    b += line(70, a.Y(cap), 690, a.Y(cap), 'drop') + text(80, a.Y(cap) - 6, f'most running at once: {cap:.0f}', 't-tick', 'start')
    b += poly([(a.X(p['t']), a.Y(p['running'])) for p in tl], 'l1') + poly([(a.X(p['t']), a.Y(p['waiting'])) for p in tl], 'l2')
    kv = Axes(70, 40, 620, 200, (0, T), (0, 1))
    b += poly([(kv.X(p['t']), kv.Y(p['kv_usage'])) for p in tl], 'l3')
    for v in (0, 0.5, 1):
        b += text(700, kv.Y(v) + 4, f'{v:.0%}', 't-tick', 'start')
    b += text(745, 32, 'KV used', 't-tick', 'end')
    b += legend([('running', 'l1'), ('waiting', 'l2'), ('KV cache used (right scale)', 'l3')], 170, 20)
    return svg(290, b, f"During the overload, up to {max(p['running'] for p in tl):.0f} requests ran and up to {max(p['waiting'] for p in tl):.0f} waited; KV cache use peaked at {max(p['kv_usage'] for p in tl):.0%}.")
FIG['timeline'] = fig_timeline()

def fig_prefix():
    rows = [('off', B('prefix_off', 'median_ttft_ms'), B('prefix_off', 'p99_ttft_ms')),
            ('on', B('prefix_on', 'median_ttft_ms'), B('prefix_on', 'p99_ttft_ms'))]
    top = nice_max(max(r[2] for r in rows)); x0, W = 150, 540; b = ''
    for i, (lab, p50, p99) in enumerate(rows):
        y = 40 + i * 80
        b += text(x0 - 12, y + 30, f'cache {lab}', 't-note', 'end')
        b += rect(x0, y, W * p50 / top, 22, 'box-1') + text(x0 + W * p50 / top + 8, y + 16, f'p50 {ms(p50)}', 't-tick', 'start')
        b += rect(x0, y + 28, W * p99 / top, 22, 'box-2') + text(x0 + W * p99 / top + 8, y + 44, f'p99 {ms(p99)}', 't-tick', 'start')
    b += text(20, 22, 'Time to first token, shared 2,048-token start + 128 own tokens', 't-title', 'start')
    return svg(210, b, f'Prefix caching off: TTFT p50 {ms(rows[0][1])}, p99 {ms(rows[0][2])}. On: p50 {ms(rows[1][1])}, p99 {ms(rows[1][2])}.')
FIG['prefix'] = fig_prefix()

# ---------------------------------------------------------------- assemble
src = (HERE / 'article.md').read_text()
env = {k: v for k, v in globals().items() if not k.startswith('_')}
def fill(m):
    try:
        return str(eval(m[1], env))
    except Exception as e:
        sys.exit(f'Could not fill {{{{R:{m[1]}}}}}: {e!r}')
out = re.sub(r'\{\{R:(.+?)\}\}(?!\})', fill, src)
out = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', lambda m: f'<figure class="fig">{FIG[m[1]]}<figcaption>{m[2]}</figcaption></figure>', out)
out = re.sub(r'\n{3,}', '\n\n', out)
assert '{{' not in out.replace('{{{', ''), 'Unresolved placeholder: ' + re.search(r'.{30}\{\{.{40}', out)[0]
assert chr(0x2014) not in out, 'Em dash found'
for w in BANNED:
    assert w not in out, f'Banned word: {w}'
for img in re.findall(r'\]\((/img/[^)]+)\)', out):
    assert (ROOT / 'public' / img.lstrip('/')).is_file(), f'missing image {img}'
assert (ROOT / 'public/notebooks/vllm-on-colab.ipynb').is_file(), 'notebook missing: run make_notebook.py'

spec = importlib.util.spec_from_file_location('vllm_paper_assemble', ROOT / 'code/papers/vllm/assemble.py')
ste = importlib.util.module_from_spec(spec); spec.loader.exec_module(ste)
body = re.sub(r'<figure.*?</figure>', '', out.split('\n---\n', 1)[1], flags=re.S)
body = re.sub(r'<details>.*?</details>', '', body, flags=re.S)
problems = ste.ste_report(body, 8)
if args.strict and problems:
    sys.exit('STE problems: not written')
Path(args.out).write_text(out)
print(f'wrote {args.out}: {len(body.split())} words of prose, {out.count("<figure")} figures')
