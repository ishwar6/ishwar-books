"""Figures for Chapter 4 (workflow patterns). Writes results/figs_ch4.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-note / t-muted, 9 px for t-title, 14 px for t-big; 17 px between baselines.
The drawing helpers are the ones of figs_ch3.py. At the end, check() estimates every text's box and reports overlaps and clipping,
so no figure ships with colliding labels. The result charts read results/ch4_*.json written by the chapter's scripts."""
import json, math, re, textwrap
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars
from alammar import frame

CH = json.load(open('results/ch4_chain.json'))
RO = json.load(open('results/ch4_router.json'))
PA = json.load(open('results/ch4_parallel.json'))
EV = json.load(open('results/ch4_evaluator.json'))
CO = json.load(open('results/ch4_cost.json'))
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def notes(x, y, lines, cls='t-muted', right=740):
    """Wrap a list of note sentences to the figure width; returns (parts, number of lines)."""
    out = textwrap.wrap(' '.join(lines), int((right - x) / 7.2))
    return para(x, y, out, cls), len(out)


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick', rx=10):
    b = [box(x, y, w, h, cls, rx)]
    ty = y + (h / 2 + 5 if not lines else 22)
    b.append(text(x + w / 2, ty, title, tcls, 'middle'))
    b += para(x + w / 2, y + 40, lines, lcls, 'middle')
    return b


def hpath(points, cls='edge', marker='ah'):
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return f'<path class="{cls}" d="{d}" marker-end="url(#{marker})"/>'


def circ(x, y, r, cls='node on'):
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def diamond(cx, cy, hw, hh, label, cls='box-4'):
    pts = f'{cx},{cy - hh} {cx + hw},{cy} {cx},{cy + hh} {cx - hw},{cy}'
    return [f'<polygon class="{cls}" points="{pts}"/>', text(cx, cy + 4, label, 't-tick', 'middle')]


def chart(x0, y0, W, H, xs, series, xlabel, ylabel, ymax, yticks, fmt=lambda v: f'{v:.0f}', right=170, dashed=()):
    """Line chart as parts (so it can sit inside a larger figure). series: [(name, values)], direct end labels."""
    L, T, B = 56, 26, 40
    pw, ph = W - L - right, H - T - B
    X = lambda v: x0 + L + (v - xs[0]) / (xs[-1] - xs[0]) * pw
    Y = lambda v: y0 + T + ph - v / ymax * ph
    b = []
    for t in yticks:
        b += [line(x0 + L, Y(t), x0 + L + pw, Y(t), 'grid'), text(x0 + L - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
    for v in xs:
        b.append(text(X(v), y0 + T + ph + 18, f'{v}', 't-tick', 'middle'))
    b.append(line(x0 + L, y0 + T + ph, x0 + L + pw, y0 + T + ph, 'axis'))
    ends = []
    for i, (name, vals) in enumerate(series):
        dc = f's{i + 1}'
        extra = ' style="stroke-dasharray:5 4"' if name in dashed else ''
        b.append(f'<polyline class="edge-{i + 1}"{extra} points="' + ' '.join(f'{X(x):.1f},{Y(v):.1f}' for x, v in zip(xs, vals)) + '"/>')
        for x, v in zip(xs, vals):
            b.append(f'<g class="mark"><title>{esc(name)}, n={x}: {fmt(v)}</title><circle class="{dc} ring" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="4.5"/></g>')
        ends.append([Y(vals[-1]), name, dc, Y(vals[-1])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
    for ly, lab, dc, ty in ends:
        b.append(line(x0 + L + pw + 6, ty, x0 + L + pw + 16, ly, 'grid'))
        b.append(f'<rect class="{dc}" x="{x0 + L + pw + 20}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
        b.append(text(x0 + L + pw + 36, ly + 4, lab, 't-tick'))
    b.append(text(x0 + L + pw / 2, y0 + H - 6, xlabel, 't-tick', 'middle'))
    b.append(text(x0 + L - 46, y0 + T - 12, ylabel, 't-tick'))
    return b


# ---------------------------------------------------------------- 1. who holds the control flow
def spectrum():
    b = [text(380, 22, 'Who holds the control flow: from a fixed chain to a free loop', 't-title', 'middle')]
    cols = [('chain', 'fixed steps', 'in code', 'box-3'), ('router', 'LLM picks', 'one branch', 'box-3'), ('parallel', 'fixed', 'fan-out', 'box-3'),
            ('orchestrator', 'LLM splits', 'the work', 'box-2'), ('evaluator', 'LLM judges', 'when to stop', 'box-2'), ('agent', 'LLM picks', 'every step', 'box-1')]
    x0, cw, gap, y0 = 130, 92, 10, 44
    for i, (name, a, c, cls) in enumerate(cols):
        x = x0 + i * (cw + gap)
        b += [box(x, y0, cw, 70, cls, 10), text(x + cw / 2, y0 + 22, name, 't-note', 'middle'),
              text(x + cw / 2, y0 + 42, a, 't-tick', 'middle'), text(x + cw / 2, y0 + 58, c, 't-tick', 'middle')]
    rows = [('predictable', [5, 4, 4, 3, 3, 1], 's3'), ('easy to test', [5, 4, 4, 2, 3, 1], 's3'),
            ('cost, latency', [1, 1, 2, 3, 3, 5], 's1'), ('novel inputs', [1, 2, 2, 4, 3, 5], 's2')]
    y = y0 + 96
    for lab, vals, cls in rows:
        b.append(text(20, y + 9, lab, 't-tick'))
        for i, v in enumerate(vals):
            x = x0 + i * (cw + gap) + 6
            b.append(f'<g class="mark"><title>{esc(lab)}, {esc(cols[i][0])}: {v} of 5</title><rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{v / 5 * 80:.1f}" height="11" rx="3"/></g>')
        y += 26
    y += 10
    b.append(hpath([(x0, y), (x0 + 6 * (cw + gap) - gap, y)], 'path', 'ah-on'))
    b += [text(x0, y + 20, 'workflows: the engineer writes the path', 't-muted'),
          text(x0 + 6 * (cw + gap) - gap, y + 20, 'agents: the model writes the path', 't-muted', 'end')]
    _p, _n = notes(20, y + 50, ['Bars are qualitative (0 to 5), a summary of this chapter\'s measurements and the papers it reads, not a benchmark.'])
    b += _p
    return svg(760, y + 50 + _n * 17 + 6, 'Six arrangements of model calls ordered by who decides the next step, from a fixed chain written in code to an agent that decides every step, with how predictable, testable, costly and adaptable each one is.', b)


F['ch4_spectrum'] = spectrum()


# ---------------------------------------------------------------- 2. graph vocabulary
def graph_vocab():
    b = [text(380, 22, 'A workflow as a graph: nodes, edges, a gate, a conditional edge, checkpoints', 't-title', 'middle')]
    cy = 120
    b += [circ(36, cy, 14, 'node on'), text(36, cy + 34, 'START', 't-tick', 'middle'), arrow(50, cy, 78, cy)]
    b += block(78, cy - 20, 92, 40, 'extract', cls='box-3')
    b.append(arrow(170, cy, 194, cy))
    b += diamond(226, cy, 32, 24, 'gate')
    b.append(text(226, cy - 34, 'schema ok?', 't-muted', 'middle'))
    b += [arrow(258, cy, 284, cy), text(270, cy - 8, 'yes', 't-muted', 'middle')]
    b += block(284, cy - 20, 92, 40, 'classify', cls='box-3')
    b.append(hpath([(226, cy + 24), (226, cy + 56), (124, cy + 56), (124, cy + 22)], 'path', 'ah-on'))
    b.append(text(175, cy + 72, 'no: retry once with the error', 't-muted', 'middle'))
    b += [hpath([(376, cy - 6), (404, cy - 6), (404, cy - 50), (430, cy - 50)], 'edge'), hpath([(376, cy + 6), (404, cy + 6), (404, cy + 50), (430, cy + 50)], 'edge')]
    b += [text(398, cy - 30, 'refund', 't-muted', 'end'), text(398, cy + 40, 'other', 't-muted', 'end')]
    b += block(430, cy - 70, 110, 40, 'refund flow', cls='box-2')
    b += block(430, cy + 30, 110, 40, 'reply flow', cls='box-2')
    b += [arrow(540, cy - 50, 566, cy - 50)]
    b += block(566, cy - 70, 110, 40, 'human approves', cls='box-1')
    b += [hpath([(676, cy - 50), (720, cy - 50), (720, cy - 16)], 'edge'), hpath([(540, cy + 50), (720, cy + 50), (720, cy + 16)], 'edge')]
    b += [circ(720, cy, 14, 'node on'), text(690, cy + 4, 'END', 't-tick', 'end')]
    for x in [124, 330, 485]:                        # checkpoint markers: state saved after the node
        yy = cy - 34 if x != 485 else cy - 22
        b.append(f'<rect class="s4" x="{x - 5}" y="{yy:.1f}" width="10" height="10" rx="2"/>')
    b += [f'<rect class="s4" x="540" y="{cy + 92}" width="10" height="10" rx="2"/>', text(556, cy + 101, '= checkpoint (state saved)', 't-muted')]
    y = cy + 132
    defs = [('node', 'a step: an LLM call, a tool call or plain code; reads state, returns an update'),
            ('edge', 'a fixed transition: after this node, always run that one'),
            ('conditional edge', 'a function of the state picks the next node (here: refund or other)'),
            ('gate', 'a programmatic check between steps; failing it retries or stops the chain'),
            ('checkpoint', 'the state written to storage after a node, so a run can pause and resume')]
    for i, (t, d) in enumerate(defs):
        b += [text(30, y + i * 20, t, 't-note'), text(160, y + i * 20, d, 't-tick')]
    return svg(760, y + 5 * 20 + 6, 'A small workflow drawn as a graph: start, an extract node, a gate that retries on a schema failure, a classify node, a conditional edge to a refund flow with human approval or a reply flow, and checkpoints after each node; with one-line definitions of node, edge, conditional edge, gate and checkpoint.', b)


F['ch4_graph_vocab'] = graph_vocab()


# ---------------------------------------------------------------- 3. prompt chain with gates
def chain():
    b = [text(380, 22, 'Prompt chaining: fixed steps, a programmatic gate after each one', 't-title', 'middle')]
    steps = [('extract', ['ticket -> JSON:', 'type, order id,', 'urgency'], 'box-3'), ('draft', ['facts -> category', 'and a reply', '(JSON)'], 'box-2'),
             ('check', ['reply -> policy', 'verdict', '(JSON)'], 'box-4')]
    x0, fw, gap, y0 = 20, 168, 70, 44
    for i, (t, ls, cls) in enumerate(steps):
        x = x0 + i * (fw + gap)
        b += frame(x, y0, fw, 112, i + 1, t, cls)
        b += para(x + 18, y0 + 52, ls, 't-tick')
        gx = x + fw + gap / 2
        if i < 2:
            b += diamond(gx, y0 + 56, 24, 18, 'gate')
            b += [arrow(x + fw, y0 + 56, gx - 24, y0 + 56), arrow(gx + 24, y0 + 56, gx + gap / 2, y0 + 56)]
    xe = x0 + 2 * (fw + gap) + fw
    b += [arrow(xe, y0 + 56, xe + 30, y0 + 56), text(xe + 34, y0 + 52, 'send', 't-tick'), text(xe + 34, y0 + 68, 'reply', 't-tick')]
    gx = x0 + fw + gap / 2
    b.append(hpath([(gx, y0 + 74), (gx, y0 + 140), (x0 + fw / 2, y0 + 140), (x0 + fw / 2, y0 + 114)], 'path', 'ah-on'))
    b.append(text(x0 + 4, y0 + 158, 'fail: retry the step once, with the error in the prompt', 't-muted'))
    b.append(hpath([(gx + gap + fw, y0 + 74), (gx + gap + fw, y0 + 186)], 'edge'))
    b.append(text(gx + gap + fw + 8, y0 + 196, 'fails twice: stop, hand to a person', 't-muted'))
    y = y0 + 226
    b += [line(20, y - 10, 740, y - 10, 'edge-dim'), text(20, y + 8, 'what a gate can check, for free and in milliseconds', 't-note')]
    checks = ['the output parses as JSON', 'every required key is present', 'enum values in the allowed list',
              'ids match a pattern (ORD-NNNNN)', 'word limits, banned phrases', 'a value exists in the database']
    for i, c in enumerate(checks):
        b.append(text(20 + (i % 3) * 240, y + 32 + (i // 3) * 20, '- ' + c, 't-tick'))
    _p, _n = notes(20, y + 82, ['what a gate cannot check: a plausible value that is wrong (a valid "delivery" label on a question about shipping', 'zones). Those need a labelled set and an eval, not a schema.'])
    b += _p
    return svg(760, y + 82 + _n * 17 + 6, 'A three-step prompt chain (extract, draft, check) with a programmatic gate after each step: a failed check retries the step once with the error in the prompt, and a second failure stops the chain and hands it to a person. Below, the checks a gate can run and the one kind of error it cannot see.', b)


F['ch4_chain'] = chain()


# ---------------------------------------------------------------- 4. chain reliability curve
def chain_reliability():
    rows = CO['chain']
    ns = list(range(1, 11))
    pick = lambda r, d: [next(c['p_success'] for c in rows if c['r'] == r and c['d'] == d and c['n'] == n) * 100 for n in ns]
    series = [('99% per step', pick(0.99, 0.0)), ('90% per step', pick(0.90, 0.0)), ('90% + gate (d=0.8)', pick(0.90, 0.8))]
    b = [text(380, 20, 'End-to-end success of an n-step chain: r to the power n', 't-title', 'middle')]
    b += chart(0, 28, 760, 300, ns, series, 'steps in the chain (n)', 'P(chain succeeds), %', 100, [0, 25, 50, 75, 100], right=210,
               dashed=('90% + gate (d=0.8)',))
    _p, _n = notes(30, 350, ['At 95% per step, ten steps succeed 60% of the time. A gate that detects 80% of bad outputs and retries once turns a',
                        '90%-per-step step into a 97% one: five such steps go from 59% to 87% end to end, for about 8% more calls (ch4_cost.py part 1).'])
    b += _p
    return svg(760, 350 + _n * 17 + 6, 'Probability that a chain of n steps succeeds when each step succeeds with probability 99, 95 or 90 percent, and for 90 percent per step with a gate that catches 80 percent of bad outputs and retries once.', b)


F['ch4_chain_reliability'] = chain_reliability()


# ---------------------------------------------------------------- 5. our chain results
def chain_results():
    S = CH['summary']
    names = [('ungated', 'strict prompt, no gates'), ('gated', 'strict prompt, gates'), ('loose-ungated', 'loose prompt, no gates'), ('loose-gated', 'loose prompt, gates')]
    b = [text(380, 22, f'Our chain on {S["gated"]["n"]} tickets: where the gates earn their place', 't-title', 'middle'),
         text(30, 50, 'end-to-end success', 't-note'), text(470, 50, 'calls per ticket', 't-note')]
    succ = [(lab, S[k]['success']) for k, lab in names]
    calls = [('', S[k]['calls'] / S[k]['n']) for k, _ in names]
    n = S['gated']['n']
    b += hbars(30, 62, succ, 170, row_h=30, cls='s3', label_w=190, fmt=lambda v: f'{v:.0f}/{n}', vmax=n)
    b += hbars(470, 62, calls, 180, row_h=30, cls='s1', label_w=16, fmt=lambda v: f'{v:.1f}', vmax=4.5)
    y = 62 + 4 * 30 + 24
    agree = S['gated']['checker_agreement']
    flags = CH.get('checker_flags', {})
    lu, lg = S['loose-ungated'], S['loose-gated']
    _p, _n = notes(30, y, [f'Strict prompt: the gates had nothing to catch, and the one failure (a shipping question labelled "delivery") was a valid label, so',
                           f'no gate could see it. Loose prompt: {lu["n"] - lu["success"]} of {lu["n"]} chains broke on free-form labels and bare order numbers; with gates',
                           f'{lg["success"]} of {lg["n"]} succeeded, for {lg["calls"] - lu["calls"]} extra calls. Two retries "fixed" the order id by setting it to null, which the schema allows.',
                           f'The step-3 model checker flagged "over 70 words" {flags.get("P1", 0)} times across the four runs on replies a word count passed.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Results of ch4_chain.py: end-to-end success and calls per ticket for four conditions, a strict or loose first-step prompt, each with and without gates.', b)


F['ch4_chain_results'] = chain_results()


# ---------------------------------------------------------------- 6. routing: classifier and cascade
def router():
    b = [text(190, 22, 'classifier router', 't-title', 'middle'), text(570, 22, 'cheap-first cascade', 't-title', 'middle'),
         line(380, 10, 380, 300, 'edge-dim')]
    b += block(20, 120, 70, 40, 'input', cls='box')
    b += [arrow(90, 140, 112, 140)]
    b += block(112, 116, 86, 48, 'router', cls='box-4')
    b.append(text(155, 182, 'small model', 't-muted', 'middle'))
    b.append(text(155, 198, 'or classifier', 't-muted', 'middle'))
    dests = [('billing prompt', 'box-3', 52), ('tech prompt', 'box-3', 104), ('strong model', 'box-2', 156), ('unknown: human', 'box-1', 208)]
    for lab, cls, y in dests:
        b += [hpath([(198, 140), (220, 140), (220, y + 16), (240, y + 16)], 'edge')]
        b += block(240, y, 124, 32, lab, cls=cls)
    b += para(20, 270, ['one decision up front; each branch has its own', 'prompt, model and eval set'], 't-muted')
    b += block(400, 120, 64, 40, 'input', cls='box')
    b += [arrow(464, 140, 484, 140)]
    b += block(484, 116, 92, 48, 'cheap model', cls='box-3')
    b += [arrow(576, 140, 596, 140)]
    b += diamond(640, 140, 44, 30, 'confident?')
    b += [arrow(640, 110, 640, 74), text(648, 92, 'yes', 't-muted')]
    b += block(600, 40, 80, 34, 'answer', cls='box-on')
    b += [arrow(640, 170, 640, 206), text(648, 192, 'no', 't-muted')]
    b += block(592, 206, 96, 40, 'strong model', cls='box-2')
    b += para(400, 270, ['"confident" = two cheap samples agree, a', 'scorer passes, or a check on the output'], 't-muted')
    return svg(760, 306, 'Two routing designs. A classifier router makes one decision up front and sends the input to a specialised prompt, a strong model or a human. A cheap-first cascade answers with the cheap model and escalates to the strong one only when a confidence check fails.', b)


F['ch4_router'] = router()


# ---------------------------------------------------------------- 7. router results and confusion matrix
def router_results():
    S = {s['condition']: s for s in RO['summary']}
    order = ['always-cheap', 'always-strong', 'cascade', 'classifier router', 'oracle router']
    n = S['always-cheap']['n']
    b = [text(380, 22, f'Our router experiment: {n} questions, cheap = minimal effort, strong = medium effort', 't-title', 'middle'),
         text(30, 50, 'accuracy', 't-note'), text(420, 50, '$ per 1,000 correct answers', 't-note')]
    b += hbars(30, 62, [(k, S[k]['acc'] * 100) for k in order], 150, row_h=28, cls='s3', label_w=140, fmt=lambda v: f'{v:.0f}%', vmax=100)
    b += hbars(420, 62, [('', S[k]['cost_per_correct'] * 1000) for k in order], 220, row_h=28, cls='s1', label_w=16, fmt=lambda v: f'{v:.3f}',
               vmax=max(S[k]['cost_per_correct'] for k in order) * 1000 * 1.15)
    y = 62 + 5 * 28 + 30
    o = RO['oracle']
    b.append(text(30, y, 'the classifier router against the truth', 't-note'))
    cx, cw, ch = 200, 110, 34
    b += [text(cx + cw / 2, y + 22, 'sent to cheap', 't-tick', 'middle'), text(cx + cw * 1.5, y + 22, 'sent to strong', 't-tick', 'middle')]
    for i, (lab, key) in enumerate([('cheap suffices', 'cheap suffices'), ('needs strong', 'needs strong')]):
        yy = y + 30 + i * ch
        b.append(text(cx - 10, yy + ch / 2 + 4, lab, 't-tick', 'end'))
        for j, col in enumerate(['easy', 'hard']):
            v = o[key][col]
            good = (i == 0 and j == 0) or (i == 1 and j == 1)
            b += [box(cx + j * cw, yy, cw - 4, ch - 4, 'box-3' if good else 'box-2', 4), text(cx + j * cw + cw / 2 - 2, yy + ch / 2 + 2, str(v), 't-note', 'middle')]
    sent_strong = o['cheap suffices']['hard'] + o['needs strong']['hard']
    _p, _n = notes(440, y + 34, ['green: right decision; orange: wrong. Bottom-left costs accuracy (a hard question answered cheaply);',
                                 f'top-right costs money. Ours sent {sent_strong} of {n} questions to the strong model and caught',
                                 f'{o["needs strong"]["hard"]} of the {o["needs strong"]["easy"] + o["needs strong"]["hard"]} that needed it.'])
    b += _p
    return svg(760, y + 34 + _n * 17 + 6, 'Results of ch4_router.py: accuracy and cost per thousand correct answers for always-cheap, always-strong, a cascade, a classifier router and an oracle router, and the classifier router confusion matrix against the truth.', b)


F['ch4_router_results'] = router_results()


# ---------------------------------------------------------------- 8. sectioning and voting
def parallel():
    b = [text(190, 22, 'sectioning: split the work', 't-title', 'middle'), text(570, 22, 'voting: same work, n times', 't-title', 'middle'),
         line(380, 10, 380, 280, 'edge-dim')]
    for side, x0 in [('s', 20), ('v', 400)]:
        b += block(x0, 116, 64, 40, 'input', cls='box')
        labs = ['section A', 'section B', 'section C'] if side == 's' else ['sample 1', 'sample 2', 'sample 3']
        cls = ['box-3', 'box-2', 'box-4'] if side == 's' else ['box-3', 'box-3', 'box-3']
        for i, (lab, c) in enumerate(zip(labs, cls)):
            y = 52 + i * 64
            b.append(hpath([(x0 + 64, 136), (x0 + 84, 136), (x0 + 84, y + 20), (x0 + 104, y + 20)], 'edge'))
            b += block(x0 + 104, y, 104, 40, lab, cls=c)
            b.append(hpath([(x0 + 208, y + 20), (x0 + 228, y + 20), (x0 + 228, 136), (x0 + 246, 136)], 'edge'))
        b += block(x0 + 246, 112, 100, 48, 'merge' if side == 's' else 'vote', cls='box-on')
    b += para(20, 260, ['different prompts on different parts; merge', 'by code (union) or by one model call'], 't-muted')
    b += para(400, 260, ['the same prompt n times; aggregate by majority,', 'a verifier, or a judge choosing the best'], 't-muted')
    b += [line(20, 300, 740, 300, 'edge-dim')]
    _p, _n = notes(20, 322, ['latency = the slowest branch + the merge;  cost = the sum of the branches + the merge.',
                             'Parallel calls buy time, never tokens. Voting only helps when samples are right more often than wrong and do not share mistakes.'])
    b += _p
    return svg(760, 322 + _n * 17 + 6, 'Two forms of parallelisation. Sectioning splits a task into different independent parts handled by different prompts and merges the results; voting runs the same prompt several times and aggregates by majority, a verifier or a judge.', b)


F['ch4_parallel'] = parallel()


# ---------------------------------------------------------------- 9. our parallel results
def parallel_results():
    ns = [1, 3, 5]
    vs, bs = PA['vote_summary'], PA['p24_summary']
    p = PA['per_sample_pass']
    series = [('vote, word problems', [vs[str(n)]['acc'] * 100 for n in ns]), ('best-of-n, Game of 24', [bs[str(n)]['acc'] * 100 for n in ns]),
              ('if independent', [(1 - (1 - p) ** n) * 100 for n in ns])]
    b = [text(380, 20, 'Our parallel experiment: accuracy against samples, and wall-clock time', 't-title', 'middle')]
    b += chart(0, 26, 480, 280, ns, series, 'samples per task (n)', 'solved, %', 100, [0, 25, 50, 75, 100], right=170, dashed=('if independent',))
    L = PA['latency_mean']
    b += [text(510, 60, 'five samples, mean of 3 runs', 't-note')]
    b += hbars(510, 74, [('one by one', L['sequential_s']), ('5 threads', L['concurrent_s'])], 120, row_h=30, cls='s2', label_w=90,
               fmt=lambda v: f'{v:.1f} s', vmax=L['sequential_s'] * 1.15)
    b += para(510, 160, ['same tokens, same bill;', f'{L["sequential_s"] / L["concurrent_s"]:.1f}x faster when fanned out'], 't-muted')
    _p, _n = notes(30, 330, [f'Voting stayed at {vs["1"]["acc"]:.0%}: on the problems the model missed, its samples scattered over wrong answers and no majority was right.',
                        f'Best-of-n with a verifier rose from {bs["1"]["acc"]:.0%} to {bs["5"]["acc"]:.0%}, far below the {1 - (1 - p) ** 5:.0%} that independent samples would give: the samples fail on',
                        'the same puzzles. Cost grew exactly n times in both.'])
    b += _p
    return svg(760, 330 + _n * 17 + 6, 'Results of ch4_parallel.py: accuracy of majority voting and of best-of-n with a programmatic verifier for one, three and five samples, against the curve independent samples would follow, and the wall-clock time of five samples run one by one or on five threads.', b)


F['ch4_parallel_results'] = parallel_results()


# ---------------------------------------------------------------- 10. orchestrator-workers
def orchestrator():
    b = [text(380, 22, 'Orchestrator-workers: the split is decided per input, at run time', 't-title', 'middle')]
    b += block(20, 120, 70, 40, 'task', cls='box')
    b.append(arrow(90, 140, 112, 140))
    b += block(112, 100, 130, 80, 'orchestrator', ['plans, writes one', 'brief per worker'], cls='box-4')
    briefs = ['brief 1: objective,', 'brief 2: format,', 'brief k: tools,']
    for i in range(3):
        y = 48 + i * 70
        b.append(hpath([(242, 140), (262, 140), (262, y + 26), (300, y + 26)], 'edge'))
        b += [box(300, y + 8, 190, 44, 'box-ghost', 8), text(312, y + 26, f'worker {i + 1 if i < 2 else "k"}', 't-note'),
              text(312, y + 43, 'own context, own tools', 't-tick')]
        b.append(hpath([(490, y + 30), (520, y + 30), (520, 140), (540, 140)], 'edge'))
    b += block(540, 112, 100, 56, 'synthesise', cls='box-2')
    b += [arrow(640, 140, 662, 140)]
    b += block(662, 120, 78, 40, 'answer', cls='box-on')
    b.append(hpath([(590, 168), (590, 252), (177, 252), (177, 182)], 'path', 'ah-on'))
    b.append(text(380, 272, 'not enough yet: plan again, spawn more workers', 't-muted', 'middle'))
    y = 300
    b += [line(20, y - 10, 740, y - 10, 'edge-dim')]
    _p, _n = notes(20, y + 10, ['the difference from sectioning: there the sections are written in code; here the model decides how many and which.',
                           'what goes wrong: vague briefs (workers duplicate each other), too many workers for an easy task, a synthesis that',
                           'drops or contradicts findings. What it costs: the sum of every worker\'s whole loop; latency is the slowest worker.'])
    b += _p
    return svg(760, y + 10 + _n * 17 + 6, 'Orchestrator-workers: an orchestrator plans and writes a brief for each worker; workers run with their own contexts and tools; a synthesis step merges their results, and the orchestrator can plan again and spawn more workers.', b)


F['ch4_orchestrator'] = orchestrator()


# ---------------------------------------------------------------- 11. orchestrator cost arithmetic
def orch_cost():
    O = CO['orchestrator']
    b = [text(380, 22, 'One agent doing k subtasks in turn, against an orchestrator with k workers', 't-title', 'middle'),
         text(30, 50, 'input tokens per task', 't-note'), text(420, 50, 'latency per task (s)', 't-note')]
    toks, lats = [], []
    for o in O:
        toks += [(f'k={o["k"]}, one agent', o['single_inp']), (f'k={o["k"]}, orchestrator', o['orch_inp'])]
        lats += [('', o['single_lat']), ('', o['orch_lat'])]
    b += hbars(30, 62, toks, 170, row_h=24, cls='s1', label_w=170, fmt=lambda v: f'{v:,.0f}', vmax=max(v for _, v in toks) * 1.25)
    b += hbars(420, 62, lats, 230, row_h=24, cls='s2', label_w=16, fmt=lambda v: f'{v:.0f}', vmax=max(v for _, v in lats) * 1.2)
    y = 62 + 6 * 24 + 20
    _p, _n = notes(30, y, ['Planning numbers of ch4_cost.py (a worker gets a 400-token brief plus the 1,500-token prompt). With fresh worker contexts the',
                      'tokens are similar to a loop that re-reads its history, and the time is the slowest worker. Anthropic\'s 15x multiplier comes',
                      'from each worker running a full tool-using loop of its own, which this arithmetic leaves out.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Input tokens and latency per task for one agent doing k subtasks in sequence and for an orchestrator with k parallel workers, for k equal to 2, 4 and 8, from the planning numbers in ch4_cost.py.', b)


F['ch4_orch_cost'] = orch_cost()


# ---------------------------------------------------------------- 12. evaluator-optimiser loop
def evaluator():
    b = [text(380, 22, 'Evaluator-optimiser: generate, evaluate, feed back, stop by a rule', 't-title', 'middle')]
    b += block(20, 92, 70, 40, 'task', cls='box')
    b.append(arrow(90, 112, 112, 112))
    b += block(112, 88, 120, 48, 'generator', cls='box-3')
    b.append(arrow(232, 112, 256, 112))
    b += block(256, 92, 80, 40, 'draft', cls='box')
    b.append(arrow(336, 112, 360, 112))
    b += block(360, 76, 150, 72, 'evaluator', ['1. code checks first', '2. a judge model'], cls='box-4')
    b.append(arrow(510, 112, 536, 112))
    b += diamond(588, 112, 52, 30, 'stop?')
    b += [arrow(640, 112, 666, 112), text(652, 104, 'pass', 't-muted', 'middle')]
    b += block(666, 92, 74, 40, 'ship', cls='box-on')
    b.append(hpath([(588, 142), (588, 190), (172, 190), (172, 138)], 'path', 'ah-on'))
    b.append(text(380, 208, 'fail: the exact failures go back to the generator ("C3: a sentence of 23 words, limit 20")', 't-muted', 'middle'))
    y = 240
    b += [line(20, y - 6, 740, y - 6, 'edge-dim'), text(20, y + 14, 'stop rules, in the order to apply them', 't-note')]
    rules = [('passes', 'every check passes, or the score clears a threshold'), ('max rounds', 'two to four; most of the gain comes by the third'),
             ('no progress', 'the same failure twice in a row: stop, do not loop'), ('then', 'hand the best draft so far to a person, with its failures')]
    for i, (k, d) in enumerate(rules):
        b += [text(20, y + 38 + i * 19, k, 't-tick'), text(120, y + 38 + i * 19, d, 't-tick')]
    return svg(760, y + 38 + 4 * 19 + 6, 'The evaluator-optimiser loop: a generator writes a draft, an evaluator runs code checks and then a judge model, a stop rule decides, and on failure the exact failures go back to the generator; below, the stop rules in the order to apply them.', b)


F['ch4_evaluator'] = evaluator()


# ---------------------------------------------------------------- 13. our evaluator results
def evaluator_results():
    pc = EV['judge_vs_checker']['per_constraint']
    names = {'C1': 'C1 55-75 words', 'C2': 'C2 four sentences', 'C3': 'C3 sentence <= 20', 'C4': 'C4 keywords', 'C5': 'C5 banned words'}
    S = EV['summary']
    n = S['judge']['n']
    b = [text(380, 22, 'Our evaluator experiment: a judge model against a programmatic checker', 't-title', 'middle'),
         text(30, 50, 'judge agrees with the checker, per constraint', 't-note'), text(470, 50, f'final drafts ({n} paragraphs)', 't-note')]
    b += hbars(30, 62, [(names[c], pc[c]['agree'] * 100) for c in ['C1', 'C2', 'C3', 'C4', 'C5']], 170, row_h=28, cls='s3', label_w=150,
               fmt=lambda v: f'{v:.0f}%', vmax=100)
    items = [('code: loop stopped', S['programmatic']['stopped_on_pass']), ('code: truly pass', S['programmatic']['truly_passes']),
             ('judge: loop stopped', S['judge']['stopped_on_pass']), ('judge: truly pass', S['judge']['truly_passes'])]
    b += hbars(470, 62, items, 110, row_h=28, cls='s2', label_w=140, fmt=lambda v: f'{v:.0f}/{n}', vmax=n)
    J = EV['judge_vs_checker']
    y = 62 + 5 * 28 + 20
    _p, _n = notes(30, y, [f'Over {J["drafts"]} drafts the judge agreed with the checker {J["agreement"]:.0%} of the time: {J["false_pass"]} false passes and {J["false_fail"]} false fails.',
                           f'It was worst at counting words (C1, {pc["C1"]["agree"]:.0%}). In this run its errors were mostly false fails, so the judge loop ran',
                           f'{S["judge"]["mean_rounds"]:.1f} rounds on average against {S["programmatic"]["mean_rounds"]:.1f} and stopped on only {S["judge"]["stopped_on_pass"]} of {n} paragraphs, although {S["judge"]["truly_passes"]} of its',
                           f'final drafts met every constraint. A held-out fact check, never fed back, passed {S["programmatic"]["heldout_ok"]}/{n} and {S["judge"]["heldout_ok"]}/{n}.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Results of ch4_evaluator.py: how often a judge model agreed with a programmatic checker on each of five constraints, and how many final drafts each loop stopped on against how many truly passed.', b)


F['ch4_evaluator_results'] = evaluator_results()


# ---------------------------------------------------------------- 14. composing patterns, with durable execution
def compose():
    b = [text(380, 22, 'Patterns compose: router in front, chain or orchestrator behind, evaluator last', 't-title', 'middle')]
    b += block(20, 104, 60, 40, 'input', cls='box')
    b.append(arrow(80, 124, 100, 124))
    b += block(100, 100, 80, 48, 'router', cls='box-4')
    b += [hpath([(180, 116), (200, 116), (200, 74), (214, 74)], 'edge'), hpath([(180, 132), (200, 132), (200, 176), (214, 176)], 'edge')]
    for i, lab in enumerate(['step 1', 'step 2', 'step 3']):
        x = 214 + i * 82
        b += block(x, 56, 70, 36, lab, cls='box-3')
        if i < 2:
            b.append(arrow(x + 70, 74, x + 82, 74))
    b.append(text(214, 46, 'known request: a gated chain', 't-muted'))
    b += block(214, 156, 92, 40, 'orchestrator', cls='box-4')
    for i in range(3):
        b += [arrow(306, 176, 330, 160 + i * 16), box(330, 152 + i * 16, 70, 14, 'box-ghost', 4)]
    b.append(text(336, 212, 'workers in parallel', 't-muted'))
    b.append(text(214, 230, 'open request: orchestrator-workers', 't-muted'))
    b += [hpath([(460, 74), (490, 74), (490, 116), (510, 116)], 'edge'), hpath([(400, 176), (490, 176), (490, 132), (510, 132)], 'edge')]
    b += block(510, 100, 90, 48, 'evaluator', cls='box-2')
    b += [arrow(600, 124, 620, 124)]
    b += block(620, 100, 120, 48, 'human approves', cls='box-1')
    b.append(text(680, 166, 'interrupt: pause', 't-muted', 'middle'))
    b.append(text(680, 182, 'and wait', 't-muted', 'middle'))
    y = 262
    b += [line(20, y, 740, y, 'edge-dim'), text(20, y + 20, 'durable execution: what happens when the process dies at step 3', 't-note')]
    ty = y + 52
    xs = [60, 166, 272, 378, 484, 590, 696]
    labs = ['router', 'step 1', 'step 2', 'step 3', 'resume', 'step 3', 'evaluate']
    for i, (x, lab) in enumerate(zip(xs, labs)):
        cls = 'box-1' if i == 3 else ('box-on' if i == 4 else 'box-3')
        b += [box(x - 40, ty, 80, 30, cls, 6), text(x, ty + 19, lab, 't-tick', 'middle')]
        if i < 6:
            b.append(arrow(x + 40, ty + 15, xs[i + 1] - 40, ty + 15))
        if i in (0, 1, 2, 5):
            b.append(f'<rect class="s4" x="{x - 5}" y="{ty + 38}" width="10" height="10" rx="2"/>')
    b += [text(378, ty + 62, 'crash', 't-muted', 'middle'), text(510, ty + 62, 'load the last checkpoint', 't-muted', 'middle')]
    _p, _n = notes(20, ty + 92, ['Saved after each node (squares): the state and a trace id. Resuming re-runs only step 3, so step 3 must be idempotent: running',
                            'it twice must not send two emails or charge twice. Give each side effect an idempotency key; log one span per node.'])
    b += _p
    return svg(760, ty + 92 + _n * 17 + 6, 'A composed workflow: a router sends known requests to a gated chain and open requests to an orchestrator with parallel workers; an evaluator checks the result and a human approval interrupt pauses the run. Below, durable execution: state is checkpointed after each node, a crash at step 3 resumes from the last checkpoint, and the re-run step must be idempotent.', b)


F['ch4_compose'] = compose()


# ---------------------------------------------------------------- 15. companies and patterns
def companies():
    pats = ['chain', 'routing', 'sectioning', 'voting/multi-run', 'orchestrator', 'evaluator', 'human gate']
    rows = [('Uber uReview', [1, 0, 1, 1, 0, 1, 0]), ('Uber QueryGPT', [1, 1, 0, 0, 0, 0, 1]), ('Uber Genie (EAg-RAG)', [1, 1, 0, 0, 0, 1, 0]),
            ('LinkedIn assistant', [1, 1, 0, 0, 0, 0, 0]), ('LinkedIn SQL Bot', [1, 0, 0, 0, 0, 1, 0]), ('DoorDash Dasher support', [1, 0, 0, 0, 0, 1, 1]),
            ('Spotify Honk', [0, 0, 0, 0, 0, 1, 1]), ('Airbnb test migration', [1, 0, 0, 0, 0, 1, 1]), ('Stripe Minions', [1, 0, 0, 0, 0, 1, 0]), ('Anthropic Research', [0, 0, 1, 0, 1, 0, 0]),
            ('Microsoft Magentic-One', [0, 0, 0, 0, 1, 0, 0])]
    b = [text(380, 22, 'Workflow patterns in the write-ups of Section 4.8 (filled = stated on the page)', 't-title', 'middle')]
    x0, lw, cw, y0, rh = 20, 200, 72, 140, 26
    for j, p in enumerate(pats):
        cx = x0 + lw + j * cw + cw / 2
        b.append(text(cx - 4, y0 - 10, p, 't-tick', 'start', f' transform="rotate(-50 {cx - 4:.1f} {y0 - 10})"'))
    for i, (name, vals) in enumerate(rows):
        y = y0 + i * rh
        b.append(text(x0 + lw - 10, y + rh / 2 + 4, name, 't-tick', 'end'))
        for j, v in enumerate(vals):
            cx = x0 + lw + j * cw + cw / 2
            b.append(f'<g class="mark"><title>{esc(name)}: {esc(pats[j])}</title><rect class="{"cell" if v else "cell-masked"}" x="{cx - 11:.1f}" y="{y + 3:.1f}" width="22" height="{rh - 6}" rx="4" style="fill-opacity:{0.85 if v else 1}"/></g>')
    y = y0 + len(rows) * rh + 16
    _p, _n = notes(20, y, ['Chains dominate; most add routing in front or an evaluator behind; orchestrators appear where the decomposition',
                      'cannot be written in advance (open-ended research). Voting/multi-run at Uber is re-running review five times to evaluate.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Workflow patterns stated in the production write-ups of Section 4.8, one row per system: prompt chains are the most common, usually with a router in front or an evaluator behind, and orchestrator-workers appear for open-ended research.', b)


F['ch4_companies'] = companies()


# ---------------------------------------------------------------- 16. choosing and evolving
def decision():
    b = [text(380, 22, 'Evolving a workflow: start with a chain, add one pattern per measured problem', 't-title', 'middle')]
    steps = [('start', 'box', 'a single call, then a chain of fixed steps', 'gates between steps; an eval set for each step'),
             ('inputs diverge?', 'box-4', 'add a router in front', 'measure its confusion matrix; keep an unknown bucket'),
             ('latency too high?', 'box-3', 'run independent sections in parallel', 'cost stays the sum; time becomes the slowest'),
             ('split not knowable?', 'box-2', 'add an orchestrator over workers', 'cost: every worker\'s loop; write precise briefs'),
             ('quality measurable?', 'box-1', 'add an evaluator loop with a stop rule', 'code checks first; measure the judge first'),
             ('steps unknowable?', 'box-2', 'only then an agent loop (Chapter 3)', 'a budget, a trace and a verifier still apply')]
    y = 44
    for i, (q, cls, act, note) in enumerate(steps):
        b += [box(30, y, 220, 44, cls, 8), text(140, y + 27, q, 't-tick', 'middle')]
        b += [arrow(250, y + 22, 300, y + 22), text(275, y + 16, 'yes', 't-muted', 'middle')]
        b += [text(310, y + 19, act, 't-tick'), text(310, y + 36, note, 't-muted')]
        if i < len(steps) - 1:
            b += [arrow(140, y + 44, 140, y + 58), text(150, y + 56, 'no', 't-muted')]
        y += 58
    b += para(380, y + 8, ['re-measure accuracy, cost per correct answer and p95 latency after every addition'], 't-muted', 'middle')
    return svg(760, y + 26, 'A migration path for workflows: start with a gated chain; add a router when inputs diverge, parallel sections when latency hurts, an orchestrator when the split cannot be written in advance, an evaluator where quality can be measured, and an agent loop only when the steps themselves cannot be known.', b)


F['ch4_decision'] = decision()


# ---------------------------------------------------------------- overlap and clipping check (as in figs_ch3.py)
def check(F):
    """Estimate every <text> box from its class and anchor; report texts outside the viewBox and pairs that overlap."""
    PX = {'t-title': 9.0, 't-big': 14.0, 't-strong': 7.6, 't-val': 7.2}
    problems = 0
    for key, s in F.items():
        W, H = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s).groups())
        boxes = []
        for m in re.finditer(r'<text class="([^"]+)" x="([\d.-]+)" y="([\d.-]+)" text-anchor="(\w+)"([^>]*)>([^<]*)</text>', s):
            cls, x, y, anchor, extra, t = m.groups()
            if not t.strip() or 'rotate' in extra:
                continue
            pw = next((PX[c] for c in cls.split() if c in PX), 7.2)
            t = t.replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
            w = len(t) * pw
            x, y = float(x), float(y)
            x0 = x - w if anchor == 'end' else x - w / 2 if anchor == 'middle' else x
            bb = (x0, y - 11, x0 + w, y + 3, t)
            if bb[0] < -2 or bb[2] > W + 2 or bb[1] < 0 or bb[3] > H:
                print(f'{key}: CLIPPED {t!r} box=({bb[0]:.0f},{bb[1]:.0f},{bb[2]:.0f},{bb[3]:.0f}) viewBox={W:.0f}x{H:.0f}'); problems += 1
            boxes.append(bb)
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, c = boxes[i], boxes[j]
                if a[0] < c[2] - 2 and c[0] < a[2] - 2 and a[1] < c[3] - 2 and c[1] < a[3] - 2:
                    print(f'{key}: OVERLAP {a[4]!r} / {c[4]!r}'); problems += 1
    print(f'check: {problems} problems')
    return problems


json.dump(F, open('results/figs_ch4.json', 'w'))
print(f'{len(F)} figures -> results/figs_ch4.json:', ', '.join(F))
check(F)
