"""Figures for Chapter 8 (RLHF with PPO). Writes results/figs_ch8.json (key -> svg).
Colours come only from CSS classes. Every number is read from a results/ch8_*.json file written by a script
(ch8_toy.py, ch8_ppo.py, ch8_eval.py)."""
import json, math, os
import numpy as np
from figlib import svg, text, box, arrow, esc
from agentfig import line

J = lambda n: json.load(open(f'results/{n}.json')) if os.path.exists(f'results/{n}.json') else None
TOY = J('ch8_toy')
RUNS = {'beta005': J('ch8_ppo_beta005'), 'nokl': J('ch8_ppo_nokl')}
EV = J('ch8_eval')
LAB = {'beta005': 'beta = 0.05', 'nokl': 'no KL penalty'}
LC = {'beta005': ('l1', 's1'), 'nokl': ('l2', 's2')}
F = {}
DASH = ' stroke-dasharray="6 4"'


def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def chart(x0, y0, w, h, series, xr, yr, xticks, yticks, fx=str, fy=str, xlabel='', ylabel='', end_labels=True, dots=True):
    """A small line chart (copied from figs_ch7.py). series: [(name, xs, ys, line_cls, dot_cls[, dashed])]."""
    X = lambda v: x0 + (v - xr[0]) / (xr[1] - xr[0]) * w
    Y = lambda v: y0 + h - (v - yr[0]) / (yr[1] - yr[0]) * h
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, fy(t), 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), y0 + h + 18, fx(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    ends = []
    for s in series:
        name, xs, ys, lc, dc = s[:5]
        dashed = len(s) > 5 and s[5]
        b.append(f'<polyline class="{lc}"{DASH if dashed else ""} points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(xs, ys)) + '"/>')
        if dots:
            for a, c in zip(xs, ys):
                b.append(f'<g class="mark"><title>{esc(name)}: {fx(a)}, {fy(c)}</title><circle class="{dc} ring" cx="{X(a):.1f}" cy="{Y(c):.1f}" r="3.5"/></g>')
        ends.append([Y(ys[-1]), name, dc, X(xs[-1])])
    if end_labels:
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
        for ly, name, dc, ex in ends:
            b.append(f'<rect class="{dc}" x="{ex + 10:.1f}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
            b.append(text(ex + 25, ly + 4, name, 't-tick'))
    if xlabel:
        b.append(text(x0 + w / 2, y0 + h + 38, xlabel, 't-muted', 'middle'))
    if ylabel:
        b.append(text(x0 - 8, y0 - 14, ylabel, 't-muted', 'start'))
    return b, X, Y


def nice(lo, hi, n=4):
    """Round axis limits and ticks."""
    span = hi - lo
    step = next(s for s in [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.25, 0.5, 1, 2, 5, 10, 20, 50] if span / s <= n)
    lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
    ticks = [round(lo + i * step, 6) for i in range(int(round((hi - lo) / step)) + 1)]
    return lo, hi, ticks


def smooth(v, k=5):
    v = np.array([np.nan if x is None else x for x in v], dtype=float)
    return [float(np.nanmean(v[max(0, i - k + 1):i + 1])) for i in range(len(v))]


# ------------------------------------------------------------------ 1. REINFORCE vs PPO use of a batch
def reuse():
    b = [text(20, 26, 'What each method does with one batch of 32 sampled replies', 't-title')]
    y = 50
    for name, n, note in [('REINFORCE (Chapter 6)', 1, 'one gradient step, then the batch is thrown away'),
                          ('PPO (this chapter)', 4, '2 epochs x 2 minibatches = 4 gradient steps on the same batch')]:
        b.append(text(20, y + 26, name, 't-note'))
        b += [box(200, y + 6, 120, 34, 'box-2', 8), text(260, y + 28, 'sample 32', 't-tick', 'middle'), arrow(322, y + 23, 344, y + 23)]
        for k in range(n):
            x = 348 + k * 46
            b += [box(x, y + 10, 40, 26, 'box-on' if n > 1 else 'box-1', 6), text(x + 20, y + 28, f'g{k + 1}', 't-tick', 'middle')]
        b.append(text(200, y + 58, note, 't-muted'))
        y += 84
    b.append(text(20, y + 14, 'Sampling 32 replies of up to 128 tokens takes most of a step; reusing it is where PPO saves time.', 't-tick'))
    return svg(760, y + 30, 'Two rows: REINFORCE samples a batch and takes one gradient step; PPO samples a batch and takes four gradient steps on it.', b)


F['ch8_reuse'] = reuse()


# ------------------------------------------------------------------ 2. the clipped objective as a function of the ratio
def clip_shape():
    if not TOY:
        return None
    cases = TOY['cases']
    b = []
    for pi, (sgn, A, title) in enumerate([(1, 1.5, 'advantage A = +1.5'), (-1, -0.8, 'advantage A = -0.8')]):
        x0, y0, w, h = 70 + pi * 380, 70, 290, 190
        rs = np.linspace(0.5, 1.6, 111)
        un = rs * A
        cl = np.minimum(rs * A, np.clip(rs, 0.8, 1.2) * A)
        yr = (-1.4, 2.6) if A > 0 else (-1.4, 0.0)
        yt = [-1, 0, 1, 2] if A > 0 else [-1.2, -0.8, -0.4, 0]
        p, X, Y = chart(x0, y0, w, h, [('r x A', rs, un, 'l4', 's4', True), ('PPO: min(...)', rs, cl, 'l1', 's1')], (0.5, 1.6), yr,
                        [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], yt, fx=lambda v: f'{v:.1f}', fy=lambda v: f'{v:g}', xlabel='ratio r = pi_new / pi_old',
                        ylabel='objective for this token', end_labels=False, dots=False)
        b += p
        b.append(text(x0 - 50, 22, title, 't-title'))
        for e in (0.8, 1.2):
            b.append(f'<line class="edge-dim" x1="{X(e):.1f}" y1="{y0}" x2="{X(e):.1f}" y2="{y0 + h}"{DASH}/>')
        for c in cases:
            if (c['A'] > 0) == (A > 0):
                b.append(f'<circle class="s2" cx="{X(c["ratio"]):.1f}" cy="{Y(c["objective"]):.1f}" r="5"/>')
                b.append(text(X(c['ratio']), Y(c['objective']) - 10, f'r={c["ratio"]:.2f}', 't-tick', 'middle'))
    b += [f'<rect class="s1" x="80" y="310" width="10" height="10" rx="2"/>', text(96, 319, 'clipped objective (what PPO maximises)', 't-tick'),
          f'<rect class="s4" x="370" y="310" width="10" height="10" rx="2"/>', text(386, 319, 'unclipped r x A', 't-tick'),
          f'<circle class="s2" cx="545" cy="315" r="5"/>', text(556, 319, 'the four worked cases', 't-tick')]
    return svg(780, 335, 'Two panels plotting the clipped objective against the ratio: for positive advantage it rises then goes flat above 1.2; for negative advantage it goes flat below 0.8.', b)


F['ch8_clip_shape'] = clip_shape()


# ------------------------------------------------------------------ 3. one batch, many steps (bandit)
def bandit_one():
    if not TOY:
        return None
    R = TOY['bandit']['runs']
    ks = list(range(TOY['bandit']['K'] + 1))
    b = []
    names = [('reinforce', 'REINFORCE, reused', 'l2', 's2'), ('ratio', 'ratio x A', 'l4', 's4'), ('clip', 'PPO clipped', 'l1', 's1')]
    p, X, Y = chart(60, 66, 240, 180, [(n, ks, R[k]['kl'], lc, dc) for k, n, lc, dc in names], (0, 60), (0, 6), [0, 20, 40, 60], [0, 2, 4, 6],
                    fx=str, fy=lambda v: f'{v:g}', xlabel='gradient steps on the same batch', ylabel='KL(old || new), nats', dots=False)
    b += p + [text(20, 22, 'How far the policy moves', 't-title')]
    p, X, Y = chart(480, 66, 240, 180, [(n, ks, R[k]['reward_p10'], lc, dc) for k, n, lc, dc in names], (0, 60), (-0.12, 0.08),
                    [0, 20, 40, 60], [-0.1, -0.05, 0, 0.05], fx=str, fy=lambda v: f'{v:+.2f}', xlabel='gradient steps on the same batch',
                    ylabel='true reward, worst 10% of seeds', dots=False, end_labels=False)
    b += p + [text(440, 22, 'The unlucky batches', 't-title')]
    st = TOY['bandit']['start']
    b.append(f'<line class="edge-dim" x1="480" y1="{Y(st):.1f}" x2="720" y2="{Y(st):.1f}"{DASH}/>')
    b.append(text(716, Y(st) - 6, f'start {st:+.3f}', 't-tick', 'end'))
    return svg(780, 310, 'Left: KL from the old policy grows without limit for the two unclipped objectives and stays near 0.05 for PPO. Right: the worst 10% of seeds end below the starting reward without clipping.', b)


F['ch8_bandit_one'] = bandit_one()


def bandit_loop():
    if not TOY:
        return None
    R = TOY['loop']['runs']
    it = list(range(TOY['loop']['iters'] + 1))
    names = [('reinforce_K1', 'REINFORCE, 1 step per batch', 'l2', 's2'), ('ratio_K10', 'ratio x A, 10 steps', 'l4', 's4'),
             ('clip_K10', 'PPO clipped, 10 steps', 'l1', 's1')]
    b = []
    p, X, Y = chart(60, 40, 330, 210, [(n, it, R[k]['reward'], lc, dc) for k, n, lc, dc in names], (0, 40), (0.0, 0.3), [0, 10, 20, 30, 40],
                    [0, 0.1, 0.2, 0.3], fx=str, fy=lambda v: f'{v:.1f}', xlabel='iterations (a fresh batch of 32 each)',
                    ylabel='true expected reward (mean of 500 seeds)', dots=False, end_labels=False)
    b += p
    for i, (k, n, lc, dc) in enumerate(names):
        b += [f'<rect class="{dc}" x="80" y="{310 + i * 18}" width="10" height="10" rx="2"/>', text(96, 319 + i * 18, n, 't-tick')]
    bars = [('REINFORCE, 1 step', R['reinforce_K1']['stuck']), ('REINFORCE, 10 steps', R['reinforce_K10']['stuck']),
            ('ratio x A, 10 steps', R['ratio_K10']['stuck']), ('PPO clipped, 10 steps', R['clip_K10']['stuck'])]
    b.append(text(470, 30, 'Seeds stuck on a worse token at the end', 't-title'))
    for i, (n, v) in enumerate(bars):
        y = 58 + i * 40
        b += [text(470, y, n, 't-tick'), box(470, y + 6, max(2, 260 * v / 0.5), 16, 's1' if 'PPO' in n else 's2', 3),
              text(476 + 260 * v / 0.5, y + 19, f'{v:.0%}', 't-val')]
    b += para(470, 238, ['"Stuck": the policy puts over 95% of its', 'probability on one token that is not the', 'best one, so new batches teach it nothing.'], 't-tick')
    return svg(780, 370, 'Left: expected reward over 40 iterations; PPO ends highest. Right: share of seeds stuck on a worse token: 0% for one-step REINFORCE, about 40% for unclipped reuse, 11% for PPO.', b)


F['ch8_bandit_loop'] = bandit_loop()

# ------------------------------------------------------------------ 4. the loop
def loop_fig():
    steps = [('1  rollout', 'policy samples one reply per prompt', '32 prompts, T = 0.7, <= 128 tokens', 'box-2'),
             ('2  score', 'reward model scores each reply;', 'policy, reference, value read every token', 'box-1'),
             ('3  rewards', '-beta x (log pi_old - log pi_ref) per token,', 'plus the normalised score at the last token', 'box-3'),
             ('4  advantages', 'GAE (gamma 1, lambda 0.95), then whiten;', 'returns = advantages + values', 'box-4'),
             ('5  update', 'clipped policy loss + clipped value loss,', '2 epochs x 2 minibatches on this batch', 'box-on')]
    b = []
    y = 20
    for i, (t, l1, l2, cls) in enumerate(steps):
        b += [box(40, y, 160, 52, cls, 10), text(120, y + 31, t, 't-note', 'middle'),
              text(220, y + 22, l1, 't-tick'), text(220, y + 40, l2, 't-tick')]
        if i < len(steps) - 1:
            b.append(arrow(120, y + 54, 120, y + 70))
        y += 72
    b += [f'<path class="edge" d="M40,{y - 46} H18 V46 H36" fill="none" marker-end="url(#ah)"/>',
          text(14, (y + 20) / 2, 'repeat', 't-tick', 'middle', ' transform="rotate(-90 14 %.1f)"' % ((y + 20) / 2))]
    b += [box(560, 30, 190, 120, 'box-ghost', 8), text(655, 52, 'no gradient', 't-note', 'middle'),
          text(655, 72, 'steps 1 to 4', 't-tick', 'middle'), text(655, 92, 'store for every token:', 't-tick', 'middle'),
          text(655, 110, 'old log-prob, ref log-prob,', 't-tick', 'middle'), text(655, 128, 'value, reward, advantage', 't-tick', 'middle'),
          box(560, 300, 190, 60, 'box-ghost', 8), text(655, 324, 'gradient', 't-note', 'middle'),
          text(655, 344, 'policy LoRA + value model', 't-tick', 'middle')]
    return svg(770, y + 10, 'Five boxes from top to bottom: rollout, score, rewards, advantages, update, with an arrow from update back to rollout.', b)


F['ch8_loop'] = loop_fig()


def four_models():
    b = []
    mods = [(30, 'Policy', 'Qwen2.5-0.5B-Instruct', '+ LoRA adapter (trains)', 'box-on', 'writes replies; gets the PPO gradient'),
            (215, 'Reference', 'same weights, adapter off', '(frozen)', 'box-ghost', 'log pi_ref for the KL penalty'),
            (400, 'Reward model', 'Chapter 7 RM', '(frozen)', 'box-3', 'one score per finished reply'),
            (585, 'Value model', 'copy of the Chapter 7 RM', 'adapter + head train', 'box-4', 'a value at every token, for GAE')]
    for x, t, l1, l2, cls, job in mods:
        b += [box(x, 30, 165, 96, cls, 10), text(x + 82, 56, t, 't-note', 'middle'), text(x + 82, 80, l1, 't-tick', 'middle'),
              text(x + 82, 98, l2, 't-tick', 'middle'), text(x + 82, 214, job, 't-tick', 'middle')]
    b += [f'<path class="edge" d="M112,128 V150 H297 V128" fill="none"/>', text(205, 170, 'one copy in memory', 't-muted', 'middle'),
          f'<path class="edge" d="M482,128 V150 H667 V128" fill="none"/>', text(575, 170, 'value starts as a copy of the reward model', 't-muted', 'middle'),
          text(30, 194, 'What each one does:', 't-muted')]
    return svg(770, 230, 'Four boxes: the policy (trains), the reference (frozen, shares the policy weights), the reward model (frozen) and the value model (trains, initialised from the reward model).', b)


F['ch8_four_models'] = four_models()


def memory():
    if not TOY:
        return None
    M = TOY['memory']
    b = [text(20, 24, 'GB of weights + optimizer state, full fine-tuning, mixed precision', 't-title')]
    parts = [('policy', 's1'), ('value', 's2'), ('ref', 's3'), ('rm', 's4')]
    for i, m in enumerate(M):
        y = 50 + i * 56
        tot = m['total']
        b.append(text(150, y + 20, m['name'], 't-tick', 'end'))
        x, W = 160, 470
        scale = W / tot
        for k, cls in parts:
            w = m[k] * scale
            b.append(f'<g class="mark"><title>{k}: {m[k] / 1e9:.1f} GB</title><rect class="{cls}" x="{x:.1f}" y="{y + 6}" width="{w:.1f}" height="22" rx="3"/></g>')
            x += w
        b.append(text(x + 8, y + 22, f'{tot / 1e9:,.1f} GB', 't-val'))
        b.append(text(160, y + 44, f'policy {m["policy"] / 1e9:,.1f} + value {m["value"] / 1e9:,.1f} + reference {m["ref"] / 1e9:,.1f} + reward {m["rm"] / 1e9:,.1f}', 't-tick'))
    y = 50 + len(M) * 56
    for j, (k, cls) in enumerate(parts):
        nm = {'policy': 'policy', 'value': 'value model', 'ref': 'reference', 'rm': 'reward model'}[k]
        b += [f'<rect class="{cls}" x="{160 + j * 130}" y="{y}" width="10" height="10" rx="2"/>', text(176 + j * 130, y + 9, nm, 't-tick')]
    b.append(text(160, y + 30, 'Each bar is scaled to its own total; the totals grow 10x per row.', 't-muted'))
    return svg(780, y + 44, 'Stacked bars for 0.5B, 7B and 70B models: 17.8 GB, 252 GB and 2,520 GB, dominated by the trained policy and value model.', b)


F['ch8_memory'] = memory()

# ------------------------------------------------------------------ 5. training curves
def curves():
    if not all(RUNS.values()):
        return None
    panels = [('score', 'normalised proxy score (RM, EOS rule)', None), ('kl', 'KL to the reference, nats per reply', None),
              ('length', 'reply length, tokens', None), ('finished', 'share of replies that ended', None)]
    b = []
    for pi, (key, title, _) in enumerate(panels):
        col, row = pi % 2, pi // 2
        x0, y0, w, h = 60 + col * 380, 46 + row * 250, 250, 160
        ser, allv = [], []
        for rn, R in RUNS.items():
            H = R['hist']
            xs = [d['step'] for d in H]
            ys = smooth([d[key] for d in H], 5)
            allv += ys
            ser.append((LAB[rn], xs, ys, *LC[rn]))
        lo, hi, ticks = nice(min(allv), max(allv))
        p, X, Y = chart(x0, y0, w, h, ser, (0, max(xs)), (lo, hi), [0, 5, 10, 15, 20], ticks, fx=str,
                        fy=lambda v: f'{v:g}', xlabel='PPO iteration' if row else '', dots=False,
                        end_labels=False)
        b += p + [text(x0 - 40, y0 - 22, title, 't-title')]
    for j, rn in enumerate(RUNS):
        b += [f'<rect class="{LC[rn][1]}" x="{80 + j * 200}" y="530" width="10" height="10" rx="2"/>', text(96 + j * 200, 539, LAB[rn], 't-tick')]
    b.append(text(500, 539, '5-iteration running mean of each batch', 't-muted'))
    return svg(780, 552, 'Four panels over 20 PPO iterations for the two runs: proxy score, KL to the reference, reply length, and share of finished replies.', b)


F['ch8_curves'] = curves()


def ppo_stats():
    if not all(RUNS.values()):
        return None
    panels = [('clipfrac', 'clip fraction'), ('approx_kl', 'approx KL (old to new), per token'),
              ('explained_var', 'value model: explained variance'), ('entropy', 'policy entropy, nats per token')]
    b = []
    for pi, (key, title) in enumerate(panels):
        col, row = pi % 2, pi // 2
        x0, y0, w, h = 60 + col * 380, 46 + row * 250, 250, 160
        ser, allv = [], []
        for rn, R in RUNS.items():
            H = [d for d in R['hist'] if key in d]
            xs = [d['step'] for d in H]
            ys = smooth([d[key] for d in H], 3)
            allv += ys
            ser.append((LAB[rn], xs, ys, *LC[rn]))
        lo, hi, ticks = nice(min(allv), max(allv))
        p, X, Y = chart(x0, y0, w, h, ser, (0, 20), (lo, hi), [0, 5, 10, 15, 20], ticks, fx=str,
                        fy=lambda v: f'{v:g}', xlabel='PPO iteration' if row else '', dots=False, end_labels=False)
        b += p + [text(x0 - 40, y0 - 22, title, 't-title')]
    for j, rn in enumerate(RUNS):
        b += [f'<rect class="{LC[rn][1]}" x="{80 + j * 200}" y="530" width="10" height="10" rx="2"/>', text(96 + j * 200, 539, LAB[rn], 't-tick')]
    b.append(text(500, 539, '3-iteration running mean', 't-muted'))
    return svg(780, 552, 'Four panels of PPO diagnostics for the two runs: clip fraction, approximate KL between old and new policy, explained variance of the value model, and entropy.', b)


F['ch8_ppo_stats'] = ppo_stats()


# ------------------------------------------------------------------ 6. proxy vs gold on held-out prompts
def ev_series(rn, key):
    gm, gs = EV['gold_norm']['mean'], EV['gold_norm']['sd']
    pts = [(0, EV['start'])] + sorted((int(k), v) for k, v in EV[rn].items())
    xs = [p for p, _ in pts]
    if key == 'gold':
        ys = [float(np.mean((np.array(v['gold']) - gm) / gs)) for _, v in pts]
    elif key == 'proxy_eos':
        ys = [float(np.mean(np.where(v['finished'], v['proxy'], -1.0))) for _, v in pts]
    else:
        ys = [float(np.mean(v[key])) for _, v in pts]
    return xs, ys


def gold_fig():
    if not EV or not all(r in EV for r in RUNS):
        return None
    b = []
    for pi, rn in enumerate(RUNS):
        x0, y0, w, h = 60 + pi * 380, 50, 250, 200
        xs, pr = ev_series(rn, 'proxy')
        _, go = ev_series(rn, 'gold')
        allv = pr + go
        lo, hi, tk = nice(min(allv + [0]) - 0.05, max(allv) + 0.05)
        p, X, Y = chart(x0, y0, w, h, [('proxy (our RM)', xs, pr, 'l2', 's2', True), ('gold (Skywork)', xs, go, 'l1', 's1')],
                        (0, 20), (lo, hi), [0, 10, 20], tk, fx=str, fy=lambda v: f'{v:+.2f}',
                        xlabel='PPO iteration (checkpoint)', end_labels=False)
        b += p + [text(x0 + w / 2, 28, LAB[rn], 't-title', 'middle')]
    b += [f'<rect class="s2" x="80" y="300" width="10" height="10" rx="2"/>', text(96, 309, 'proxy: our Chapter 7 reward model, raw score (no EOS rule)', 't-tick'),
          f'<rect class="s1" x="80" y="320" width="10" height="10" rx="2"/>', text(96, 329, 'gold: Skywork-Reward-V2-Qwen3-0.6B (never seen in training)', 't-tick'),
          text(500, 309, 'both in sd units of the starting model', 't-muted')]
    return svg(780, 342, 'Two panels, one per run: proxy reward and gold reward of held-out replies at each checkpoint.', b)


F['ch8_gold'] = gold_fig()


def gold_kl():
    if not EV or not all(r in EV for r in RUNS):
        return None
    ser, allx, ally = [], [], []
    for rn in RUNS:
        _, kl = ev_series(rn, 'kl')
        _, go = ev_series(rn, 'gold')
        _, pr = ev_series(rn, 'proxy')
        ser += [(LAB[rn] + ', gold', kl, go, *LC[rn]), (LAB[rn] + ', proxy', kl, pr, LC[rn][0], LC[rn][1], True)]
        allx += kl; ally += go + pr
    xr = (0, max(allx) * 1.08)
    lo, hi, tk = nice(min(ally + [0]) - 0.05, max(ally) + 0.05)
    xt = [0, 1, 2, 3, 4, 5, 6]
    p, X, Y = chart(70, 40, 480, 240, ser, (0, 6.2), (lo, hi), xt, tk, fx=lambda v: f'{v:.0f}',
                    fy=lambda v: f'{v:+.2f}', xlabel='KL to the starting model on held-out replies (nats per reply)', ylabel='reward (sd units)')
    return svg(780, 330, 'Proxy (dashed) and gold (solid) reward against KL for both runs.', p)


F['ch8_gold_kl'] = gold_kl()

# ------------------------------------------------------------------ 7. one reply, token by token
def token_gae():
    R = RUNS['beta005']
    if not R or 'example' not in R['samples']:
        return None
    E = R['samples']['example']
    toks = E['tokens']
    n = len(toks)
    cw = min(62, 680 / n)
    x0 = 120
    rows = [('token', None), ('V(s_t)', E['values']), ('r_t', E['rewards']), ('A_t (GAE)', E['adv_raw']), ('return', E['ret'])]
    b = [text(20, 24, 'One reply from the first batch, every token (beta = 0.05, gamma = 1, lambda = 0.95)', 't-title')]
    for ri, (name, vals) in enumerate(rows):
        y = 44 + ri * 40
        b.append(text(x0 - 10, y + 22, name, 't-note', 'end'))
        for i in range(n):
            x = x0 + i * cw
            if vals is None:
                t = {'<|im_end|>': 'im_end'}.get(toks[i], toks[i].replace('\n', '\\n').strip() or '_')
                b += [box(x + 1, y + 4, cw - 2, 28, 'box-1' if i < n - 1 else 'box-on', 4),
                      text(x + cw / 2, y + 23, t[:8], 't-tick', 'middle')]
            else:
                v = 0.0 if abs(vals[i]) < 0.005 else vals[i]
                cls = 's1' if v >= 0 else 's2'
                op = min(1.0, 0.25 + abs(v) / 1.5)
                b += [f'<rect class="{cls}" x="{x + 1:.1f}" y="{y + 4}" width="{cw - 2:.1f}" height="28" rx="4" opacity="{op:.2f}"/>',
                      text(x + cw / 2, y + 23, f'{v:+.2f}', 't-tick', 'middle')]
    y = 44 + len(rows) * 40 + 10
    b.append(text(x0, y + 4, f'Reward model score (normalised) at the last token: {E["score"]:+.3f}; the other rewards are -0.05 x (log pi_old - log pi_ref).', 't-muted'))
    return svg(max(780, x0 + n * cw + 20), y + 20, 'A table of one short reply: tokens, values, per-token rewards, GAE advantages and returns.', b)


F['ch8_token_gae'] = token_gae()

json.dump({k: v for k, v in F.items() if v}, open('results/figs_ch8.json', 'w'))
print('figures:', ', '.join(k for k, v in F.items() if v), '| missing:', ', '.join(k for k, v in F.items() if not v))
