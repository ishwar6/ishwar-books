"""Figures for Chapter 6 (reinforcement learning, from scratch). Writes results/figs_ch6.json (key -> svg).
Colours come only from CSS classes, so every figure follows the site's themes. Every number is read from a results/ch6_*.json
file written by a script (ch6_bandit.py, ch6_gae.py, ch6_lm_reinforce.py)."""
import json, math, os
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars, token
from alammar import frame, brace

J = lambda n: json.load(open(f'results/{n}.json')) if os.path.exists(f'results/{n}.json') else None
BAN, GAE = J('ch6_bandit'), J('ch6_gae')
RUNS = [('nokl', 'no KL penalty', 'l2', 's2'), ('beta005', 'KL penalty, beta 0.05', 'l1', 's1'), ('beta02', 'KL penalty, beta 0.2', 'l3', 's3')]
LM = {k: J(f'ch6_lm_{k}') for k, *_ in RUNS}
F = {}


# para, block and chart are copied from figs_ch4.py (importing it would rebuild the Chapter 4 figures)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick'):
    b = [box(x, y, w, h, cls, 10)]
    b.append(text(x + w / 2, y + (h / 2 + 5 if not lines else 22), title, tcls, 'middle'))
    b += para(x + w / 2, y + 41, lines, lcls, 'middle')
    return b


def chart(x0, y0, w, h, series, xr, yr, xticks, yticks, xlog=False, ylog=False, fx=str, fy=str, xlabel='', ylabel='',
          end_labels=True, dots=True):
    """A small line chart. series: [(name, xs, ys, line_cls, dot_cls)]. Returns (parts, X, Y)."""
    tx = (lambda v: math.log10(v)) if xlog else (lambda v: v)
    ty = (lambda v: math.log10(v)) if ylog else (lambda v: v)
    X = lambda v: x0 + (tx(v) - tx(xr[0])) / (tx(xr[1]) - tx(xr[0])) * w
    Y = lambda v: y0 + h - (ty(v) - ty(yr[0])) / (ty(yr[1]) - ty(yr[0])) * h
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + w}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, fy(t), 't-tick', 'end')]
    for t in xticks:
        b.append(text(X(t), y0 + h + 18, fx(t), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{x0}" y1="{y0 + h}" x2="{x0 + w}" y2="{y0 + h}"/>')
    ends = []
    for name, xs, ys, lc, dc in series:
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in zip(xs, ys)) + '"/>')
        if dots:
            for a, c in zip(xs, ys):
                b.append(f'<g class="mark"><title>{esc(name)}: {fx(a)}, {fy(c)}</title><circle class="{dc} ring" cx="{X(a):.1f}" cy="{Y(c):.1f}" r="4"/></g>')
        ends.append([Y(ys[-1]), name, dc, X(xs[-1]), Y(ys[-1])])
    if end_labels:
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 16)
        for ly, name, dc, ex, ey in ends:
            b.append(f'<rect class="{dc}" x="{ex + 12:.1f}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
            b.append(text(ex + 27, ly + 4, name, 't-tick'))
    if xlabel:
        b.append(text(x0 + w / 2, y0 + h + 38, xlabel, 't-muted', 'middle'))
    if ylabel:
        b.append(text(x0 - 8, y0 - 16, ylabel, 't-muted', 'start'))
    return b, X, Y


def band(X, Y, xs, lo, hi, cls):
    pts = [f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(xs, hi)] + [f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(xs[::-1], lo[::-1])]
    return f'<polygon class="{cls}" fill-opacity="0.16" points="{" ".join(pts)}"/>'


# ---------------------------------------------------------------- 1. the loop, mapped onto text
def loop():
    b = [text(390, 24, 'The reinforcement learning loop, and the same loop for a language model', 't-title', 'middle')]
    b += block(40, 60, 200, 92, 'agent (policy)', ['looks at the state,', 'picks an action'], 'box-1')
    b += block(540, 60, 200, 92, 'environment', ['moves to a new state,', 'sometimes gives a reward'], 'box-2')
    b += [arrow(240, 88, 538, 88, True), text(390, 80, 'action', 't-note', 'middle')]
    b += [arrow(540, 128, 242, 128), text(390, 146, 'new state, reward', 't-note', 'middle')]
    y = 190
    b.append(line(20, y - 14, 760, y - 14, 'edge-dim'))
    b += block(40, y, 200, 92, 'the language model', ['reads prompt + tokens so far,', 'samples the next token'], 'box-1')
    b += block(540, y, 200, 92, 'the "world"', ['appends the token (no surprise);', 'a judge scores the full reply'], 'box-2')
    b += [arrow(240, y + 28, 538, y + 28, True), text(390, y + 20, 'next token, e.g. " wonderful"', 't-note', 'middle')]
    b += [arrow(540, y + 68, 242, y + 68), text(390, y + 86, 'longer text; reward 0 until the end', 't-note', 'middle')]
    b += para(390, y + 124, ['The state transition is deterministic: the new state is the old text plus the chosen token.',
                             'All the randomness is in the policy, and the reward usually arrives only once, after the last token.'], 't-muted', 'middle')
    return svg(780, y + 150, 'The agent-environment loop of reinforcement learning, and its language-model version: the model is the agent, '
               'the next token is the action, the text so far is the state, and a reward comes at the end.', b)


F['ch6_loop'] = loop()


# ---------------------------------------------------------------- 2. one episode, token by token
def episode():
    toks = ['The', ' film', ' was', ' a', ' joy', '<end>']
    b = [text(20, 24, 'One episode = one reply. Each token is one action; the reward arrives after the last one', 't-title')]
    x0, w, gap = 130, 96, 10
    b.append(text(20, 74, 'prompt', 't-muted'))
    b += token(20, 82, 96, 'Review:', 'box-ghost', 30)
    for i, t in enumerate(toks):
        x = x0 + i * (w + gap)
        b += [text(x + w / 2, 58, f'state s{i}', 't-tick', 'middle'), text(x + w / 2, 74, f'prompt + {i}' if i else 'prompt', 't-muted', 'middle')]
        b += token(x, 82, w, t, 'box-1' if t != '<end>' else 'box-3', 30)
        b += [text(x + w / 2, 132, f'action a{i}', 't-tick', 'middle'), text(x + w / 2, 152, 'r = 0' if i < len(toks) - 1 else 'r = score', 't-val', 'middle')]
        b.append(text(x + w / 2, 172, f'p = pi(a{i} | s{i})', 't-muted', 'middle'))
    b += brace(x0, x0 + len(toks) * (w + gap) - gap, 190, 'trajectory tau = (s0, a0, s1, a1, ..., s5, a5); its probability is the product of the six p values', 'edge', False, 't-tick')
    b += para(20, 236, ['The judge (a reward model, a classifier, a unit test) sees only the finished reply. Every token before the last gets',
                        'reward 0, so the learner must work out which of the earlier choices deserve the credit: the credit assignment problem.'], 't-muted')
    return svg(780, 270, 'A reply drawn as an episode: states are the prompt plus the tokens so far, actions are tokens, and the only reward comes at the end.', b)


F['ch6_episode'] = episode()


# ---------------------------------------------------------------- 3. RL words and LM words
def vocab():
    rows = [('state s_t', 'prompt + the reply tokens written so far'), ('action a_t', 'the next token (one of about 50,000 to 150,000)'),
            ('policy pi_theta(a | s)', 'the language model: a softmax over the vocabulary'), ('episode, trajectory', 'one full reply, from prompt to end token'),
            ('reward r_t', 'usually 0 for every token, a score after the last one'), ('return G_t', 'the sum of rewards from token t to the end'),
            ('value V(s_t)', 'expected final score from this partial reply'), ('advantage A(s_t, a_t)', 'how much better this token was than average here'),
            ('transition P(s\' | s, a)', 'deterministic: append the token'), ('discount gamma', 'almost always 1 for language models')]
    b = [text(20, 24, 'A dictionary: reinforcement learning words and what they mean for a language model', 't-title')]
    y = 42
    for k, v in rows:
        b += [box(20, y, 230, 30, 'box-1', 6), text(32, y + 20, k, 't-tick t-strong'), box(260, y, 500, 30, 'box', 6), text(272, y + 20, v, 't-tick')]
        y += 36
    return svg(780, y + 8, 'A table mapping reinforcement learning terms to their meaning for language models.', b)


F['ch6_vocab'] = vocab()


# ---------------------------------------------------------------- 4. discount weights
def discount():
    b = [text(20, 24, 'How much a reward k steps ahead counts today: the weight gamma^k', 't-title')]
    x0, y0, h, bw = 90, 50, 120, 22
    for gi, g in enumerate([1.0, 0.9, 0.5]):
        yy = y0 + gi * (h + 40)
        b.append(text(20, yy + h / 2 + 4, f'gamma = {g}', 't-note'))
        for k in range(24):
            v = g ** k
            x = x0 + 30 + k * (bw + 4)
            b.append(f'<g class="mark"><title>gamma^{k} = {v:.3f}</title><rect class="s{gi + 1}" x="{x}" y="{yy + h - v * h:.1f}" width="{bw}" height="{max(1, v * h):.1f}" rx="2"/></g>')
            if k in (0, 5, 10, 23):
                b.append(text(x + bw / 2, yy + h + 14, f'k={k}', 't-tick', 'middle'))
                b.append(text(x + bw / 2, yy + h - v * h - 5, f'{v:.2f}' if v >= 0.005 else '0.00', 't-tick', 'middle'))
    b += para(20, y0 + 3 * (h + 40) + 4, ['With gamma = 1 (what RLHF uses) a score after the 24th token counts in full for the first token.',
                                          'A smaller gamma makes far-away rewards matter less: lower variance, but it changes what is optimised.'], 't-muted')
    return svg(780, y0 + 3 * (h + 40) + 40, 'Bars of gamma to the power k for gamma 1, 0.9 and 0.5, over 24 steps.', b)


F['ch6_discount'] = discount()


# ---------------------------------------------------------------- 5. the log-derivative trick in four frames
def logtrick():
    b = []
    W, H = 370, 128
    steps = [('Start: the objective', ['J(theta) = sum over replies y of', 'p_theta(y) R(y)', 'R does not depend on theta']),
             ('Move the gradient inside', ['grad J = sum over y of', 'grad p_theta(y) R(y)', 'but this is not an average: no p in front']),
             ('The trick: grad p = p grad log p', ['because grad log p = grad p / p', 'grad J = sum over y of p_theta(y)', 'grad log p_theta(y) R(y)']),
             ('Now it is an average: sample it', ['grad J = E over y ~ p_theta of', '[ R(y) grad log p_theta(y) ]', 'estimate: sample y, compute, average'])]
    for i, (t, ls) in enumerate(steps):
        x, y = 20 + (i % 2) * (W + 20), 16 + (i // 2) * (H + 18)
        b += frame(x, y, W, H, i + 1, t, 'box-on' if i == 3 else 'box')
        b += para(x + 22, y + 56, ls, 't-tick', 'start', 22)
    return svg(780, 16 + 2 * (H + 18), 'Four steps of the log-derivative trick that turns the gradient of an expected reward into an average that can be sampled.', b)


F['ch6_logtrick'] = logtrick()


# ---------------------------------------------------------------- 6. one bandit step by hand
def bandit_step():
    if not BAN:
        return None
    s = BAN['step']
    b = [text(20, 24, f'One REINFORCE step on the four-reply bandit: reply 3 was sampled and rated r = {s["r"]}', 't-title')]
    cols = [('policy before', s['pi'], 'box-ghost'), ('after, no baseline', s['pi_new0'], 'box-2'), (f'after, baseline b = {s["b"]}', s['pi_newb'], 'box-1')]
    x0, h = 40, 130
    for ci, (name, p, cls) in enumerate(cols):
        x = x0 + ci * 250
        b.append(text(x + 100, 56, name, 't-note', 'middle'))
        for k in range(4):
            v = p[k]
            bx = x + 10 + k * 46
            b.append(f'<rect class="{cls}" x="{bx}" y="{70 + h - v * h / 0.5:.1f}" width="36" height="{v * h / 0.5:.1f}" rx="3"/>')
            b.append(text(bx + 18, 70 + h - v * h / 0.5 - 6, f'{v:.3f}', 't-tick', 'middle'))
            b.append(text(bx + 18, 70 + h + 16, f'reply {k + 1}', 't-tick', 'middle'))
        b.append(line(x + 4, 70 + h, x + 196, 70 + h, 'axis'))
    g0 = ', '.join(f'{v:+.3f}' for v in s['g0'])
    gb = ', '.join(f'{v:+.3f}' for v in s['gb'])
    b += para(20, 250, [f'score vector  onehot(3) - pi = ({", ".join(f"{v:+.2f}" for v in s["score"])})',
                        f'no baseline:   r x score = ({g0})   pushes reply 3 up hard, although 7.3 is only a bit above average',
                        f'baseline 6.5:  (r - b) x score = ({gb})   a small push, the size the evidence deserves'], 't-tick', 'start', 20)
    return svg(780, 318, 'Bars of the four reply probabilities before and after one update, without and with a baseline.', b)


F['ch6_bandit_step'] = bandit_step()


# ---------------------------------------------------------------- 7. variance against the baseline value
def var_sweep():
    if not BAN:
        return None
    xs = [p[0] for p in BAN['sweep']]
    ys = [p[1] for p in BAN['sweep']]
    b, X, Y = chart(90, 56, 520, 220, [('total variance', xs, ys, 'l1', 's1')], (0, 12), (0, 35), [0, 2, 4, 6, 8, 10, 12], [0, 5, 10, 15, 20, 25, 30, 35],
                    fx=lambda v: f'{v:g}', fy=lambda v: f'{v:g}', xlabel='constant baseline b subtracted from every reward', ylabel='variance of the gradient estimate', end_labels=False)
    bs, vs = BAN['b_star']
    b.insert(0, text(20, 24, 'Same average gradient, very different noise: variance against the baseline value', 't-title'))
    b += [f'<circle class="s2" cx="{X(bs):.1f}" cy="{Y(vs):.1f}" r="6"/>', text(X(bs), Y(vs) - 40, f'best b = {bs:.2f}: variance {vs:.2f}', 't-val', 'middle'),
          f'<circle class="s3" cx="{X(0):.1f}" cy="{Y(ys[0]):.1f}" r="6"/>', text(X(0) + 12, Y(ys[0]) + 4, f'no baseline: {ys[0]:.1f}', 't-val')]
    b += para(630, 110, ['Every point is', 'measured on 200,000', 'samples at the', 'starting policy.', '', 'The mean gradient', 'is the same at every b', '(it is unbiased);', 'only the noise moves.'], 't-muted')
    return svg(780, 330, 'A U-shaped curve: the variance of the REINFORCE gradient estimate as a function of the constant baseline, smallest near the average reward.', b)


F['ch6_var_sweep'] = var_sweep()


# ---------------------------------------------------------------- 8. bandit learning curves
def bandit_curves():
    if not BAN:
        return None
    T = BAN['train']
    names = [('none', 'no baseline', 'l2', 's2'), ('running', 'running-average baseline', 'l1', 's1'), ('value', 'exact value baseline', 'l3', 's3')]
    b, X, Y = chart(80, 56, 470, 240, [(lab, T['runs'][k]['steps'], T['runs'][k]['mean'], lc, dc) for k, lab, lc, dc in names],
                    (0, T['steps']), (6.4, 8.05), [0, 100, 200, 300, 400], [6.5, 7.0, 7.5, 8.0], fx=lambda v: f'{int(v)}', fy=lambda v: f'{v:.1f}',
                    xlabel='update step (one sampled reply per step)', ylabel='expected rating of the policy (best possible 8.0)', dots=False)
    for k, lab, lc, dc in names[:2]:
        r = T['runs'][k]
        b.insert(0, band(X, Y, r['steps'], r['p10'], r['p90'], dc))
    b.insert(0, text(20, 24, f'REINFORCE on the bandit, {T["seeds"]} seeds: mean and 10th to 90th percentile band', 't-title'))
    w = T['runs']['none']['winners']
    b += para(20, 368, [f'At step 400, without a baseline the most likely reply was reply 4 (the best) in {w[3]} seeds, reply 3 in {w[2]} and reply 2 in {w[1]}.',
                        'With either baseline, all 200 seeds ended on reply 4. The two baselined bands are so narrow they almost vanish.'], 't-muted')
    return svg(780, 410, 'Learning curves of expected rating for REINFORCE with no baseline, a running-average baseline and the exact value baseline.', b)


F['ch6_bandit_curves'] = bandit_curves()


# ---------------------------------------------------------------- 9. value grid of the token world
def value_grid():
    if not GAE:
        return None
    V = GAE['V']
    T = GAE['T']
    b = [text(20, 24, f'The token world: V(t, k) = chance of ending with at least {GAE["need"]} A\'s, for P(A) = {GAE["pA"]}', 't-title')]
    x0, y0, c = 150, 66, 62
    for t in range(T + 1):
        b.append(text(x0 + t * c + c / 2, y0 - 8, f't = {t}', 't-tick', 'middle'))
    for k in range(T + 1):
        b.append(text(x0 - 10, y0 + k * 40 + 25, f'k = {k} A\'s', 't-tick', 'end'))
        for t in range(T + 1):
            if k > t:
                b.append(f'<rect class="cell-masked" x="{x0 + t * c + 2}" y="{y0 + k * 40 + 2}" width="{c - 4}" height="36" rx="4"/>')
                continue
            v = V[t][k]
            b.append(f'<g class="mark"><title>V({t}, {k}) = {v:.3f}</title><rect class="cell" x="{x0 + t * c + 2}" y="{y0 + k * 40 + 2}" width="{c - 4}" height="36" rx="4" fill-opacity="{0.08 + 0.8 * v:.2f}"/></g>')
            b.append(text(x0 + t * c + c / 2, y0 + k * 40 + 25, f'{v:.2f}', 't-cell on' if v > 0.55 else 't-cell', 'middle'))
    b += para(20, y0 + 7 * 40 + 24, ['Start at the top left (no tokens yet): the chance of success is 0.54. Writing A moves one cell right and one down;',
                                     'writing B moves one cell right. After three B\'s (k = 0 at t = 3) the reply can no longer succeed: V = 0.'], 't-muted')
    return svg(780, y0 + 7 * 40 + 62, 'A grid of exact state values for the token world, by position t and number of A tokens k.', b)


F['ch6_value_grid'] = value_grid()


# ---------------------------------------------------------------- 10. TD error on the worked episode
def td_episode():
    if not GAE:
        return None
    e = GAE['episode']
    rows = e['rows']
    b = [text(20, 24, f'One episode, "{e["tokens"]}", with a rough value estimate: TD errors and GAE (lambda {e["lam"]}, gamma 1)', 't-title')]
    x0, w = 120, 104
    labels = [('token', None), ('V(s_t)', 4), ('V(s_t+1)', 5), ('reward r_t', 3), ('TD error delta_t', 6), ('GAE A_t', None), ('exact A', 7)]
    for i, r in enumerate(rows):
        b.append(text(x0 + i * w + w / 2, 56, f't = {i}', 't-tick', 'middle'))
    for j, (lab, idx) in enumerate(labels):
        y = 66 + j * 36
        b.append(text(x0 - 10, y + 21, lab, 't-tick t-strong', 'end'))
        for i, r in enumerate(rows):
            if lab == 'token':
                b += token(x0 + i * w + 8, y, w - 16, r[2], 'box-1', 30)
                continue
            v = e['gae'][i] if lab == 'GAE A_t' else r[idx]
            cls = 'box-on' if lab == 'GAE A_t' else ('box-3' if lab == 'TD error delta_t' else 'box')
            b += [box(x0 + i * w + 8, y, w - 16, 30, cls, 6), text(x0 + i * w + w / 2, y + 20, f'{v:+.3f}' if lab not in ('V(s_t)', 'V(s_t+1)', 'reward r_t') else f'{v:.3f}', 't-tick', 'middle')]
    y = 66 + len(labels) * 36 + 10
    b += para(20, y, ['delta_t = r_t + V(s_t+1) - V(s_t): did things go better or worse than the critic expected, one step later?',
                      'GAE is computed backwards: A_5 = delta_5, then A_t = delta_t + lambda x A_t+1. With lambda = 0.95 each A_t is close to the',
                      'whole remaining sum of deltas (= final reward - V(s_t)), so it is noisy; one episode cannot recover the exact advantages.'], 't-muted')
    return svg(780, y + 62, 'A table for one six-token episode: value estimates, rewards, TD errors, GAE advantages and the exact advantages.', b)


F['ch6_td_episode'] = td_episode()


# ---------------------------------------------------------------- 11. the GAE weights on future TD errors
def gae_weights():
    b = [text(20, 24, 'GAE adds up future TD errors with weights (gamma x lambda)^l  (here gamma = 1)', 't-title')]
    lams = [0.0, 0.5, 0.9, 0.95, 1.0]
    x0, bw = 170, 46
    for li, lam in enumerate(lams):
        y = 50 + li * 66
        b.append(text(20, y + 26, f'lambda = {lam}', 't-note'))
        for l in range(12):
            v = lam ** l if not (lam == 0 and l > 0) else 0.0
            if lam == 0 and l == 0:
                v = 1.0
            hh = 38 * v
            b.append(f'<g class="mark"><title>weight on delta_t+{l}: {v:.3f}</title><rect class="s{1 + li % 4}" x="{x0 + l * bw}" y="{y + 40 - hh:.1f}" width="{bw - 8}" height="{max(1, hh):.1f}" rx="2"/></g>')
            if l in (0, 1, 5, 11):
                b.append(text(x0 + l * bw + (bw - 8) / 2, y + 40 - hh - 3, f'{v:.2f}', 't-tick', 'middle'))
    for l in range(12):
        b.append(text(x0 + l * bw + (bw - 8) / 2, 50 + 5 * 66 + 4, f'd(t+{l})' if l else 'd(t)', 't-tick', 'middle'))
    b += para(20, 50 + 5 * 66 + 34, ['lambda = 0: only the next TD error (trusts the critic completely: low variance, biased if the critic is wrong).',
                                     'lambda = 1: all of them, which telescopes to "actual return - V(s_t)" (no trust in the critic beyond a baseline: unbiased, noisy).'], 't-muted')
    return svg(780, 50 + 5 * 66 + 72, 'Bar rows of the weight GAE puts on each future TD error for lambda 0, 0.5, 0.9, 0.95 and 1.', b)


F['ch6_gae_weights'] = gae_weights()


# ---------------------------------------------------------------- 12. bias and variance against lambda
def gae_biasvar():
    if not GAE:
        return None
    b = [text(20, 24, 'GAE against the exact advantage (100,000 episodes): squared bias and variance as lambda changes', 't-title')]
    names = list(GAE['gae'])
    for i, n in enumerate(names):
        rows = GAE['gae'][n]
        lam = [r[0] for r in rows]
        x0 = 60 + i * 245
        ser = [('bias^2', lam, [r[1] for r in rows], 'l2', 's2'), ('variance', lam, [r[2] for r in rows], 'l1', 's1'), ('total', lam, [r[3] for r in rows], 'l3', 's3')]
        bb, X, Y = chart(x0, 70, 180, 190, ser, (0, 1), (0, 0.17), [0, 0.5, 1], [0, 0.05, 0.10, 0.15], fx=lambda v: f'{v:g}', fy=lambda v: f'{v:.2f}',
                         xlabel='lambda', end_labels=False)
        b += bb
        b.append(text(x0 + 90, 58, n, 't-note', 'middle'))
        best = min(rows, key=lambda r: r[3])
        b.append(text(x0 + 90, 316, f'lowest total error at lambda {best[0]:g}', 't-tick', 'middle'))
    b += [f'<rect class="s2" x="60" y="332" width="12" height="12" rx="2"/>', text(78, 342, 'squared bias', 't-tick'),
          f'<rect class="s1" x="200" y="332" width="12" height="12" rx="2"/>', text(218, 342, 'variance', 't-tick'),
          f'<rect class="s3" x="320" y="332" width="12" height="12" rx="2"/>', text(338, 342, 'total error = bias^2 + variance', 't-tick')]
    return svg(780, 356, 'Three small charts of squared bias, variance and total error of GAE versus lambda, for an exact, a slightly noisy and a noisy value function.', b)


F['ch6_gae_biasvar'] = gae_biasvar()


# ---------------------------------------------------------------- 13. reward offset: what a baseline buys during training
def offset_train():
    if not GAE:
        return None
    T = GAE['train_offset']
    names = [('REINFORCE, no baseline', 'l2', 's2'), ('REINFORCE, leave-one-out baseline', 'l1', 's1'), ('GAE lambda 0.95', 'l3', 's3')]
    xs = list(range(len(T[names[0][0]]['mean'])))
    b, X, Y = chart(80, 56, 440, 230, [(n, xs, T[n]['mean'], lc, dc) for n, lc, dc in names], (0, xs[-1]), (0.3, 1.0), [0, 50, 100, 150],
                    [0.4, 0.6, 0.8, 1.0], fx=lambda v: f'{int(v)}', fy=lambda v: f'{v:.1f}', xlabel='update (16 episodes each)',
                    ylabel='P(success) of the policy', dots=False)
    for n, lc, dc in names[:2]:
        b.insert(0, band(X, Y, xs, T[n]['p10'], T[n]['p90'], dc))
    b.insert(0, text(20, 24, 'Token world with every reward shifted by +5: 100 seeds, mean and 10th to 90th percentile', 't-title'))
    b += para(20, 360, [f'Spread across seeds after 25 updates (standard deviation of P(success)): no baseline {T[names[0][0]]["sd25"]:.3f}, '
                        f'leave-one-out {T[names[1][0]]["sd25"]:.3f}, GAE {T[names[2][0]]["sd25"]:.3f}.',
                        'Adding 5 to every reward changes nothing about which replies are better, but without a baseline it multiplies the noise.'], 't-muted')
    return svg(780, 400, 'Training curves in the token world with rewards shifted by 5, for REINFORCE without a baseline, with a leave-one-out baseline, and with a GAE critic.', b)


F['ch6_offset_train'] = offset_train()


# ---------------------------------------------------------------- 14. per-token KL shaping on a real reply
def kl_shaping():
    run = LM.get('beta005')
    if not run or 'worked' not in run:
        return None
    w = run['worked']
    n = len(w['toks'])
    lr = [a - c for a, c in zip(w['lp'], w['lq'])]
    b = [text(20, 24, f'Per-token rewards for one real reply (beta = {run["beta"]}): -beta x log-ratio at every token, plus the score at the end', 't-title')]
    x0, cw, mid = 30, 30, 170
    m = max(abs(v) for v in lr) or 1
    sc = 70 / m
    for i in range(n):
        x = x0 + i * cw
        v = lr[i]
        hh = abs(v) * sc
        b.append(f'<g class="mark"><title>{esc(w["toks"][i])}: log-ratio {v:+.3f}</title><rect class="{"s2" if v > 0 else "s1"}" x="{x}" y="{mid - hh if v > 0 else mid:.1f}" width="{cw - 6}" height="{max(1, hh):.1f}" rx="2"/></g>')
        t = w['toks'][i].replace('\n', '\\n').strip() or '_'
        b.append(f'<text class="t-tick" x="{x + (cw - 6) / 2:.1f}" y="{mid + 92}" text-anchor="end" transform="rotate(-60 {x + (cw - 6) / 2:.1f} {mid + 92})">{esc(t[:10])}</text>')
    b.append(line(x0 - 4, mid, x0 + n * cw, mid, 'axis'))
    b.append(text(x0, 66, 'log pi(token) - log pi_ref(token): above the line the tuned model likes the token more than GPT-2 does', 't-muted'))
    s = sum(lr)
    b += para(20, mid + 166, [f'Sum of the {n} log-ratios = {s:.2f} nats. Penalty = beta x sum = {run["beta"]} x {s:.2f} = {run["beta"] * s:.2f}.',
                              f'Classifier score R = {w["R"]:+.2f}. Total reward of the reply = R - penalty = {w["G"][0]:+.2f} (the return G_0).'], 't-tick', 'start', 20)
    return svg(780, mid + 214, 'Bars of the per-token log-probability ratio between the tuned model and the reference for one real reply.', b)


F['ch6_kl_shaping'] = kl_shaping()


# ---------------------------------------------------------------- 15/16. the language-model runs
def lm_curves(key, title, ylabel, yr, yt, fy, foot):
    runs = [(k, lab, lc, dc) for k, lab, lc, dc in RUNS if LM.get(k)]
    if not runs:
        return None
    ser = []
    for k, lab, lc, dc in runs:
        c = LM[k]['curve']
        xs = [r['step'] for r in c]
        ys = [r[key] for r in c]
        sm = [sum(ys[max(0, i - 4):i + 1]) / len(ys[max(0, i - 4):i + 1]) for i in range(len(ys))]   # 5-step running mean
        ser.append((lab, xs[::5], sm[::5], lc, dc))
    S = max(s[1][-1] for s in ser)
    b, X, Y = chart(80, 56, 470, 240, ser, (0, S), yr, [0, S // 3, 2 * S // 3, S], yt, fx=lambda v: f'{int(v)}', fy=fy,
                    xlabel='update step (64 replies per step)', ylabel=ylabel, dots=False)
    b.insert(0, text(20, 24, title, 't-title'))
    b += para(20, 370, foot, 't-muted')
    return svg(780, 370 + 18 * len(foot) + 10, title, b)


def lm_reward():
    return lm_curves('R', 'GPT-2 trained with REINFORCE: classifier reward (log-odds of "positive")', 'reward per reply (5-step mean)',
                     (-0.5, 6.5), [0, 2, 4, 6], lambda v: f'{v:g}', ['Without a penalty the reward climbs fastest and highest. The KL penalty holds it back on purpose:',
                                                                   'the model is only allowed to gain reward it can get while staying close to GPT-2.'])


def lm_kl():
    if not LM.get('nokl'):
        return None
    top = max(r['k1'] for r in LM['nokl']['curve'])
    hi = 20 * math.ceil(top / 20)
    return lm_curves('k1', 'How far each model moved from GPT-2: KL per reply (sum over 24 tokens)', 'KL to the reference, nats per reply',
                     (0, hi), list(range(0, hi + 1, 20)), lambda v: f'{v:g}', ['Measured on the sampled tokens as the sum of log pi - log pi_ref. With no penalty nothing stops the drift.'])


def lm_div():
    return lm_curves('d2', 'Diversity of the replies: distinct-2 (share of word pairs in a batch that are unique)', 'distinct-2 (5-step mean)',
                     (0, 1), [0, 0.25, 0.5, 0.75, 1.0], lambda v: f'{v:.2f}', ['When distinct-2 falls, the batch is repeating the same word pairs: the model has found a few phrases the',
                                                                              'classifier loves and says them again and again.'])


F['ch6_lm_reward'] = lm_reward()
F['ch6_lm_kl'] = lm_kl()
F['ch6_lm_div'] = lm_div()


# ---------------------------------------------------------------- 17. held-out evaluation summary
def lm_eval():
    runs = [(k, lab, lc, dc) for k, lab, lc, dc in RUNS if LM.get(k) and 'eval_end' in LM[k]]
    if not runs:
        return None
    start = LM[runs[0][0]]['eval_start']
    rows = [('GPT-2 (start)', start, 'box-ghost')] + [(lab, LM[k]['eval_end'], f'box-{ {"s1": 1, "s2": 2, "s3": 3}[dc] }') for k, lab, lc, dc in runs]
    cols = [('P(positive)', 'p_pos', '{:.3f}'), ('KL, nats', 'kl', '{:.1f}'), ('ref-perplexity', 'ref_ppl', '{:.1f}'), ('distinct-2', 'd2', '{:.3f}'), ('entropy', 'ent', '{:.2f}')]
    b = [text(20, 24, 'Held-out check: 64 new prompts x 4 replies each, after training', 't-title')]
    x0, cw = 200, 112
    for j, (h, _, _) in enumerate(cols):
        b.append(text(x0 + j * cw + cw / 2, 56, h, 't-tick t-strong', 'middle'))
    for i, (lab, e, cls) in enumerate(rows):
        y = 68 + i * 40
        b += [box(20, y, 172, 32, cls, 6), text(30, y + 21, lab, 't-tick t-strong')]
        for j, (_, k, f) in enumerate(cols):
            b += [box(x0 + j * cw + 4, y, cw - 8, 32, 'box', 6), text(x0 + j * cw + cw / 2, y + 21, f.format(e[k]), 't-val', 'middle')]
    y = 68 + len(rows) * 40 + 12
    b += para(20, y, ['KL: exact KL to GPT-2 summed over the 24 reply tokens. Ref-perplexity: how surprised GPT-2 is by the reply (lower = more',
                      'predictable to GPT-2). Entropy: average uncertainty of the tuned model per token, in nats.'], 't-muted')
    return svg(780, y + 44, 'A table of held-out reward, KL, reference perplexity, diversity and entropy for GPT-2 and the trained models.', b)


F['ch6_lm_eval'] = lm_eval()


# ---------------------------------------------------------------- 18. timeline
def timeline():
    items = [('1983', 'Barto, Sutton, Anderson', 'actor-critic: a learned critic judges the actor', 1),
             ('1988', 'Sutton', 'temporal-difference learning, TD(lambda)', 1),
             ('1992', 'Williams', 'REINFORCE: unbiased gradient from samples, with a baseline', 2),
             ('2000', 'Sutton, McAllester, Singh, Mansour', 'the policy gradient theorem with function approximation', 2),
             ('2015', 'Schulman et al.', 'TRPO, and GAE: the lambda trade-off for advantages', 1),
             ('2015', 'Ranzato et al. (MIXER)', 'REINFORCE to train text generators on BLEU', 3),
             ('2016', 'Mnih et al. (A3C)', 'advantage actor-critic with deep networks', 1),
             ('2017', 'Schulman et al.', 'PPO (Chapter 8)', 1),
             ('2017', 'Jaques et al.', 'KL control: stay close to a pretrained sequence model', 3),
             ('2019', 'Ziegler et al.', 'RL from human preferences on GPT-2 with a KL penalty', 3),
             ('2024', 'Ahmadian et al.; Shao et al.', 'back to REINFORCE: RLOO and GRPO drop the critic', 2)]
    cls = {1: 'box-1', 2: 'box-2', 3: 'box-3'}
    b = [text(20, 24, 'Policy gradients from 1983 to 2024', 't-title')]
    y = 40
    for yr, who, d, k in items:
        b += [text(56, y + 17, yr, 't-tick', 'end'), f'<circle class="node on" cx="76" cy="{y + 12}" r="5"/>',
              box(92, y, 250, 24, cls[k], 6), text(102, y + 16.5, who, 't-tick t-strong'), text(354, y + 16.5, d, 't-tick')]
        y += 30
    b.append(line(76, 46, 76, y - 18, 'edge-dim'))
    b += [f'<rect class="box-1" x="92" y="{y + 4}" width="12" height="12" rx="2"/>', text(110, y + 14, 'critics and advantages', 't-tick'),
          f'<rect class="box-2" x="290" y="{y + 4}" width="12" height="12" rx="2"/>', text(308, y + 14, 'the policy gradient itself', 't-tick'),
          f'<rect class="box-3" x="480" y="{y + 4}" width="12" height="12" rx="2"/>', text(498, y + 14, 'applied to text', 't-tick')]
    return svg(780, y + 28, 'A timeline of policy-gradient methods from actor-critic in 1983 to RLOO and GRPO in 2024.', b)


F['ch6_timeline'] = timeline()

F = {k: v for k, v in F.items() if v}
json.dump(F, open('results/figs_ch6.json', 'w'))
print(f'{len(F)} figures:', ', '.join(F))
