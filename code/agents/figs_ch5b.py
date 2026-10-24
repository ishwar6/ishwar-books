"""Figures for Chapter 5, second half (Sections 5.4 to 5.6). Writes results/figs_ch5b.json (key -> svg); keys start with b_.
Colours come only from CSS classes (box, box-1..4, box-on, s1..s4, edge, path, grid, axis ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-note / t-muted, 9 px for t-title; 17 px between baselines. The helpers are the
ones of figs_ch4.py, copied here because figs_ch4.py builds its own figures on import. At the end, check() estimates every text's box
and reports overlaps and clipping. The result charts read results/ch5b_*.json written by this half's scripts."""
import json, re, textwrap
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars

DB = json.load(open('results/ch5b_debate.json'))
DS = json.load(open('results/ch5b_debate_stats.json'))
HO = json.load(open('results/ch5b_handoff.json'))
MA = json.load(open('results/ch5b_math.json'))
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def notes(x, y, lines, cls='t-muted', right=740):
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


def chart(x0, y0, W, H, xs, series, xlabel, ylabel, ymax, yticks, fmt=lambda v: f'{v:.0f}', right=170, dashed=(), xlabels=None):
    """Line chart as parts. series: [(name, values)], direct end labels."""
    L, T, B = 56, 26, 40
    pw, ph = W - L - right, H - T - B
    X = lambda v: x0 + L + (v - xs[0]) / (xs[-1] - xs[0]) * pw
    Y = lambda v: y0 + T + ph - v / ymax * ph
    b = []
    for t in yticks:
        b += [line(x0 + L, Y(t), x0 + L + pw, Y(t), 'grid'), text(x0 + L - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
    for i, v in enumerate(xs):
        b.append(text(X(v), y0 + T + ph + 18, xlabels[i] if xlabels else f'{v}', 't-tick', 'middle'))
    b.append(line(x0 + L, y0 + T + ph, x0 + L + pw, y0 + T + ph, 'axis'))
    ends = []
    for i, (name, vals) in enumerate(series):
        dc = f's{i + 1}'
        extra = ' style="fill:none;stroke-dasharray:5 4"' if name in dashed else ' style="fill:none"'
        b.append(f'<polyline class="edge-{i + 1}"{extra} points="' + ' '.join(f'{X(x):.1f},{Y(v):.1f}' for x, v in zip(xs, vals)) + '"/>')
        for x, v in zip(xs, vals):
            b.append(f'<g class="mark"><title>{esc(name)}, {x}: {fmt(v)}</title><circle class="{dc} ring" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="4.5"/></g>')
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


def agent(cx, cy, label, cls='box-3', w=64, h=30):
    return [box(cx - w / 2, cy - h / 2, w, h, cls, 8), text(cx, cy + 4, label, 't-tick', 'middle')]


# ---------------------------------------------------------------- 1. vote, debate and the self-review control at equal calls
def debate_shapes():
    b = [text(380, 22, 'Three ways to spend six model calls on one question', 't-title', 'middle')]
    cols = [(130, 'vote of 6'), (380, 'debate: 3 agents, 1 round'), (630, 'control: self-review')]
    for cx, t in cols:
        b.append(text(cx, 52, t, 't-note', 'middle'))
    # vote: six independent calls -> plurality
    y1 = 92
    xs = [45, 79, 113, 147, 181, 215]
    for x in xs:
        b += agent(x, y1, 'call', 'box-3', w=30, h=26)
        b.append(arrow(x, y1 + 13, 130, y1 + 92))
    b += [box(80, y1 + 92, 100, 30, 'box-4', 8), text(130, y1 + 112, 'plurality', 't-tick', 'middle')]
    b += para(130, y1 + 150, ['no call sees another', '6 calls'], 't-muted', 'middle')
    # debate
    xs = [320, 380, 440]
    for i, x in enumerate(xs):
        b += agent(x, y1, f'A{i + 1}', 'box-3', w=48, h=26)
        b += agent(x, y1 + 66, f'A{i + 1}', 'box-2', w=48, h=26)
        for x2 in xs:
            b.append(line(x2, y1 + 13, x, y1 + 51, 'edge' if x2 == x else 'grid', 'ah' if x2 == x else None))
        b.append(arrow(x, y1 + 79, 380, y1 + 112))
    b += [text(470, y1 + 4, 'round 0', 't-muted'), text(470, y1 + 70, 'round 1', 't-muted')]
    b += [box(330, y1 + 112, 100, 30, 'box-4', 8), text(380, y1 + 132, 'plurality', 't-tick', 'middle')]
    b += para(380, y1 + 170, ['each agent reads the other two', '6 calls'], 't-muted', 'middle')
    # self-review
    xs = [570, 630, 690]
    for i, x in enumerate(xs):
        b += agent(x, y1, f'A{i + 1}', 'box-3', w=48, h=26)
        b += agent(x, y1 + 66, f'A{i + 1}', 'box-1', w=48, h=26)
        b.append(arrow(x, y1 + 13, x, y1 + 51))
        b.append(arrow(x, y1 + 79, 630, y1 + 112))
    b += [box(580, y1 + 112, 100, 30, 'box-4', 8), text(630, y1 + 132, 'plurality', 't-tick', 'middle')]
    b += para(630, y1 + 170, ['each agent re-checks only itself', '6 calls'], 't-muted', 'middle')
    y = y1 + 210
    _p, _n = notes(20, y, ['Same model, same prompt and the same word budget per call in all three. Round 0 of the debate and of the control is the',
                           'first three samples of the vote, so the comparison is paired question by question. The 9-call versions add one more round.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Three ways to spend six model calls on one question: a plurality vote over six independent samples; a debate in which three agents answer, read each other\'s answers and answer again; and a control in which three agents answer and re-check only their own answer. Each ends in a plurality vote.', b)


F['b_debate_shapes'] = debate_shapes()


# ---------------------------------------------------------------- 2. debate results
def debate_results():
    S = DB['summary']
    keys = [('single', 'one call'), ('vote3', 'vote of 3'), ('vote6', 'vote of 6'), ('debate1', 'debate, 6 calls'), ('review1', 'self-review, 6 calls'),
            ('vote9', 'vote of 9'), ('debate2', 'debate, 9 calls'), ('review2', 'self-review, 9 calls')]
    b = [text(380, 22, 'Debate against voting at equal calls: 30 problems, two working budgets', 't-title', 'middle'),
         text(30, 52, 'terse: one sentence of working', 't-note'), text(420, 52, 'brief: 40 words of working', 't-note')]
    hi = {'debate1', 'debate2'}
    for col, mode in [(30, 'terse'), (420, 'brief')]:
        items = [(lab if col == 30 else '', S[mode][k] * 100) for k, lab in keys]
        b += hbars(col, 64, items, 170 if col == 30 else 240, row_h=26, cls='s3', label_w=150 if col == 30 else 12,
                   fmt=lambda v: f'{v:.0f}%', vmax=100)
        for i, (k, _) in enumerate(keys):
            if k in hi:
                b.append(f'<rect class="s1" x="{col + (150 if col == 30 else 12) - 6}" y="{64 + i * 26 + 4:.1f}" width="3" height="16" rx="1"/>')
    y = 64 + len(keys) * 26 + 22
    st = DS['terse']
    _p, _n = notes(20, y, [f'Terse: debate beat the vote of the same size on {st["debate1_vs_vote6"]["wins_a"]} problems and lost on {st["debate1_vs_vote6"]["wins_b"]} (6 calls; sign test p = {st["debate1_vs_vote6"]["p"]:.3f}),',
                           f'and {st["debate2_vs_vote9"]["wins_a"]} to {st["debate2_vs_vote9"]["wins_b"]} at 9 calls (p = {st["debate2_vs_vote9"]["p"]:.3f}). Brief: no difference. One run, one model (gpt-5-mini, minimal reasoning), 30 problems;',
                           'a marker shows the debate rows.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Accuracy of one call, plurality votes of 3, 6 and 9 samples, a three-agent debate with one or two rounds, and a self-review control, for a terse and a brief working budget. With terse answers, debate beat voting at equal calls; with brief working, all multi-call methods were about equal.', b)


F['b_debate_results'] = debate_results()


# ---------------------------------------------------------------- 3. how answers changed: debate against self-review
def flips():
    b = [text(380, 22, 'What a second round did to single answers (terse budget, 180 updates each)', 't-title', 'middle')]
    rows = [('debate', DS['terse']['flips_debate']), ('self-review', DS['terse']['flips_self-review'])]
    parts = [('wrong_to_right', 'wrong to right', 's3'), ('right_to_wrong', 'right to wrong', 's1'), ('kept_right', 'kept right', 's2'), ('kept_wrong', 'kept wrong', 's4')]
    x0, W, y0 = 130, 600, 56
    for i, (name, fl) in enumerate(rows):
        y = y0 + i * 46
        b.append(text(x0 - 10, y + 18, name, 't-tick', 'end'))
        x = x0
        tot = sum(fl.values())
        for k, lab, cls in parts:
            w = W * fl.get(k, 0) / tot
            b.append(f'<g class="mark"><title>{esc(name)}: {esc(lab)} {fl.get(k, 0)}</title><rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="26" rx="2"/></g>')
            if w > 30:
                b.append(text(x + w / 2, y + 18, str(fl.get(k, 0)), 't-val', 'middle'))
            x += w
    y = y0 + 2 * 46 + 8
    for j, (k, lab, cls) in enumerate(parts):
        b += [f'<rect class="{cls}" x="{x0 + j * 150}" y="{y}" width="10" height="10" rx="2"/>', text(x0 + j * 150 + 16, y + 9, lab, 't-tick')]
    d, s = rows[0][1], rows[1][1]
    _p, _n = notes(20, y + 40, [f'Reading two other answers moved {d["wrong_to_right"]} answers from wrong to right and {d["right_to_wrong"]} the other way. Re-checking alone was a coin',
                                f'toss: {s["wrong_to_right"]} fixed, {s["right_to_wrong"]} broken. The gain came from seeing other answers, not from a second look.'])
    b += _p
    return svg(760, y + 40 + _n * 17 + 6, 'Stacked bars of how agent answers changed between rounds in the terse setting: in debate, many more answers went from wrong to right than from right to wrong; in the self-review control, about as many went each way.', b)


F['b_flips'] = flips()


# ---------------------------------------------------------------- 4. the telephone chain and its two fixes
def telephone():
    roles = ['researcher', 'analyst', 'risk reviewer', 'writer', 'editor']
    b = [text(380, 22, 'Five agents in a chain: only the first one reads the supplier pages', 't-title', 'middle')]
    x0, w, gap, y = 104, 116, 14, 70
    b += [box(10, y - 18, 72, 36, 'box', 8), text(46, y + 4, '6 pages', 't-tick', 'middle'), arrow(82, y, x0, y)]
    for i, r in enumerate(roles):
        x = x0 + i * (w + gap)
        b += [box(x, y - 18, w, 36, 'box-3', 8), text(x + w / 2, y + 4, r, 't-tick', 'middle')]
        if i < 4:
            b.append(arrow(x + w, y, x + w + gap, y))
    b.append(text(380, y + 40, 'free: each handoff is prose, rewritten by every agent; facts can drop out at any hop', 't-muted', 'middle'))
    y2 = y + 92
    b.append(text(20, y2 - 26, 'schema: facts travel as a table that agents copy; their own work goes in a separate field', 't-note'))
    for i, r in enumerate(roles):
        x = x0 + i * (w + gap)
        b += [box(x, y2 - 18, w, 36, 'box-3', 8), text(x + w / 2, y2 + 4, r, 't-tick', 'middle')]
    b += [box(x0, y2 + 32, 5 * w + 4 * gap, 26, 'box-2', 6), text(x0 + (5 * w + 4 * gap) / 2, y2 + 49, 'supplier table: price, UK delivery days, minimum order, risks (copied unchanged)', 't-tick', 'middle')]
    for i in range(5):
        x = x0 + i * (w + gap) + w / 2
        b.append(line(x, y2 + 18, x, y2 + 32, 'edge'))
    y3 = y2 + 104
    b.append(text(20, y3 - 26, '+ verifier: after every hop, a checker reads the source pages and the handoff; the hop is redone once', 't-note'))
    for i, r in enumerate(roles):
        x = x0 + i * (w + gap)
        b += [box(x, y3 - 18, w, 36, 'box-3', 8), text(x + w / 2, y3 + 4, r, 't-tick', 'middle'),
              box(x + 18, y3 + 32, w - 36, 26, 'box-4', 6), text(x + w / 2, y3 + 49, 'verify', 't-tick', 'middle')]
        b += [arrow(x + w / 2 - 10, y3 + 18, x + w / 2 - 10, y3 + 32), arrow(x + w / 2 + 10, y3 + 32, x + w / 2 + 10, y3 + 18)]
    y = y3 + 84
    _p, _n = notes(20, y, ['After each hop a separate probe call reads only that hop\'s output and extracts the 30 gold facts (6 suppliers x price, UK delivery',
                           'days, minimum order and two risks). The probe found all 30 when it read the source pages themselves.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'The handoff experiment: six supplier pages feed a chain of five agents (researcher, analyst, risk reviewer, writer, editor). In the free condition each handoff is prose; in the schema condition the facts travel as a copied table beside each agent\'s own work; with a verifier, a checker compares each hop with the source pages and the hop is redone once.', b)


F['b_telephone'] = telephone()


# ---------------------------------------------------------------- 5. handoff results
def handoff_results():
    S = HO['summary']
    names = [('free', 'free text'), ('free+ver', 'free text + verifier'), ('schema', 'schema'), ('schema+ver', 'schema + verifier')]
    xs = [1, 2, 3, 4, 5]
    series = [(lab, S[k]['kept']) for k, lab in names]
    b = [text(380, 20, 'Facts still stated after each hop (of 30; mean of 3 runs)', 't-title', 'middle')]
    b += chart(0, 28, 760, 300, xs, series, 'hop', 'facts kept', 30, [0, 10, 20, 30], right=220,
               xlabels=['researcher', 'analyst', 'risk review', 'writer', 'editor'], dashed=('free text + verifier', 'schema + verifier'))
    c = {k: S[k]['cost_per_run_ex_probe'] for k, _ in names}
    _p, _n = notes(30, 350, [f'Free text lost most facts at the risk reviewer, which wrote about risks and dropped prices. Cost per chain, without the probes:',
                             f'free ${c["free"]:.3f}, free + verifier ${c["free+ver"]:.3f}, schema ${c["schema"]:.3f}, schema + verifier ${c["schema+ver"]:.3f}. The schema was both the most faithful and the cheapest,',
                             'because agents wrote about 400 words instead of about 1,800.'])
    b += _p
    return svg(760, 350 + _n * 17 + 6, 'Line chart of how many of 30 gold facts were still stated after each of five hops, for free-text handoffs, free text with a verifier, a structured schema, and the schema with a verifier. Free text fell to about half by the editor; the other three stayed near 30.', b)


F['b_handoff_results'] = handoff_results()


# ---------------------------------------------------------------- 6. compounding across hops
def compound():
    A = MA['A']
    ns = [1, 2, 3, 5, 8, 10]
    series = [(f'q = {q}', [v * 100 for v in A[q]]) for q in ['0.99', '0.95', '0.9']] + [('q = 0.867 (dots)', [v * 100 for v in A['0.867']])]
    b = [text(380, 20, 'Share of facts that survive n handoffs if each hop keeps a fact with probability q', 't-title', 'middle')]
    b += chart(0, 28, 760, 300, ns, series, 'number of handoffs (n)', 'facts kept, %', 100, [0, 25, 50, 75, 100], right=170)
    _p, _n = notes(30, 350, ['q to the power n, assuming independent losses that are never recovered. q = 0.867 is the per-hop rate implied by our free-text',
                             'chain; its losses were not independent (most fell at one hop), so the curve shows the scale, not the shape.'])
    b += _p
    return svg(760, 350 + _n * 17 + 6, 'Curves of q to the power n for q of 0.99, 0.95, 0.90 and 0.867, the share of facts that survive n independent handoffs.', b)


F['b_compound'] = compound()


# ---------------------------------------------------------------- 7. group chat tokens
def groupchat():
    rows = [r for r in MA['C'] if r[1] == 6]
    b = [text(380, 22, 'Input tokens over 6 rounds: re-read the whole chat, or read only a brief', 't-title', 'middle'),
         text(30, 50, 'group chat', 't-note'), text(420, 50, 'briefs only', 't-note')]
    vmax = max(r[3] for r in rows) * 1.2
    b += hbars(30, 62, [(f'{r[0]} agents', r[3]) for r in rows], 200, row_h=28, cls='s1', label_w=80, fmt=lambda v: f'{v:,.0f}', vmax=vmax)
    b += hbars(420, 62, [('', r[4]) for r in rows], 200, row_h=28, cls='s3', label_w=12, fmt=lambda v: f'{v:,.0f}', vmax=vmax)
    y = 62 + 4 * 28 + 22
    _p, _n = notes(20, y, ['300 tokens per message and 500 of instructions per turn (ch5b_math.py, part C). The chat grows with the square of the number of turns;',
                           f'with 8 agents it reads {rows[-1][3] / rows[-1][4]:.1f} times the tokens of the brief-only design.'])
    b += _p
    return svg(760, y + _n * 17 + 6, 'Bars of input tokens over six rounds for 2, 3, 5 and 8 agents: a group chat in which every turn re-reads the whole conversation grows quadratically, while turns that read only a brief grow linearly.', b)


F['b_groupchat'] = groupchat()


# ---------------------------------------------------------------- 8. MAST mapped to controls
def mast_map():
    b = [text(380, 22, 'The MAST failure categories and the control that targets each one', 't-title', 'middle')]
    cats = [('system design (44.2%)', 'box-1', ['1.1 disobey task spec 11.8%', '1.3 step repetition 15.7%', '1.5 unaware of termination 12.4%', '1.2, 1.4: role, lost history'],
             ['a written spec and role per agent', 'a turn limit and a stop rule in code', 'budgets; dedupe repeated steps']),
            ('inter-agent misalignment (32.3%)', 'box-2', ['2.6 reasoning-action mismatch 13.2%', '2.3 task derailment 7.4%', '2.2 no clarification 6.8%', '2.1, 2.4, 2.5: reset, withhold, ignore'],
             ['handoff schemas, not prose', 'one owner per decision', 'ask-back allowed in the brief']),
            ('task verification (23.5%)', 'box-4', ['3.3 incorrect verification 9.1%', '3.2 no or incomplete verification 8.2%', '3.1 premature termination 6.2%'],
             ['checks against the sources or tests', 'verify the task objective, not only format', 'a final gate before the answer'])]
    y = 48
    for name, cls, modes, ctrl in cats:
        h = 22 + 17 * max(len(modes), len(ctrl)) + 8
        b += [box(20, y, 720, h, cls, 10), text(36, y + 20, name, 't-note')]
        b += para(36, y + 40, modes, 't-tick')
        b += para(420, y + 40, ['-> ' + c for c in ctrl], 't-tick')
        y += h + 10
    _p, _n = notes(20, y + 8, ['Percentages are shares of all failures in 1,642 annotated traces of seven open-source systems (Cemri et al. 2025, Figure 1). The',
                               'right-hand column is our mapping to the controls of Sections 5.5 and 5.6, not the paper\'s.'])
    b += _p
    return svg(760, y + 8 + _n * 17 + 6, 'The three MAST categories with their most common failure modes and shares, each paired with the engineering control that targets it: specs, turn limits and budgets for system design; handoff schemas and single owners for misalignment; checks against sources and objective-level verification for verification failures.', b)


F['b_mast_map'] = mast_map()


# ---------------------------------------------------------------- 9. decision checklist
def checklist():
    b = [text(380, 22, 'Before adding a second agent: a checklist that starts from one agent', 't-title', 'middle')]
    steps = [('one agent with good tools fails?', 'box', 'measure first: a fixed eval set, traces, cost per task'),
             ('failure is context, not skill?', 'box-4', 'too much text for one context, or tool overload'),
             ('work splits into independent parts?', 'box-3', 'parallel subagents with written briefs (Section 5.2)'),
             ('parts share decisions?', 'box-2', 'if yes: keep them in one agent, or one writer'),
             ('value pays for ~15x tokens?', 'box-1', 'budget per run, turn limits, one trace per agent'),
             ('ship with checks', 'box-on', 'verifier, human checkpoints, evals per agent')]
    y = 44
    for i, (q, cls, note) in enumerate(steps):
        b += [box(30, y, 250, 40, cls, 8), text(155, y + 25, q, 't-tick', 'middle')]
        b += [arrow(280, y + 20, 320, y + 20), text(330, y + 24, note, 't-tick')]
        if i < len(steps) - 1:
            b += [arrow(155, y + 40, 155, y + 54)]
        y += 54
    _p, _n = notes(20, y + 10, ['Read down: each "yes" lets you go one step further. A "no" anywhere means one agent (Chapters 3 and 4) is the better design.',
                                'The 15x is Anthropic\'s reported token ratio of its multi-agent system to chat (Chapter 1).'])
    b += _p
    return svg(760, y + 10 + _n * 17 + 6, 'A checklist from one agent to several: confirm one agent fails on a measured eval, that the failure is about context rather than skill, that the work splits into independent parts that do not share decisions, and that the value pays for the tokens; then ship with verifiers, human checkpoints and per-agent evals.', b)


F['b_checklist'] = checklist()


# ---------------------------------------------------------------- 10. running a multi-agent system: where the controls sit
def ops():
    b = [text(380, 22, 'Where the controls sit in a running multi-agent system', 't-title', 'middle')]
    b += [box(290, 50, 180, 40, 'box-1', 8), text(380, 75, 'lead agent', 't-note', 'middle')]
    xs = [150, 380, 610]
    for i, x in enumerate(xs):
        b += [box(x - 70, 140, 140, 40, 'box-3', 8), text(x, 165, f'subagent {i + 1}', 't-tick', 'middle'), arrow(380, 90, x, 140)]
        b += [box(x - 60, 206, 120, 26, 'box-2', 6), text(x, 223, 'schema check', 't-tick', 'middle'), arrow(x, 180, x, 206)]
    b += [box(290, 262, 180, 34, 'box-4', 8), text(380, 284, 'verifier vs sources', 't-tick', 'middle')]
    for x in xs:
        b.append(arrow(x, 232, 380, 262))
    b += [box(290, 322, 180, 34, 'box-on', 8), text(380, 344, 'human checkpoint', 't-tick', 'middle'), arrow(380, 296, 380, 322)]
    side = [('budget', 'tokens and money per run; stop at the cap'), ('turn limit', 'per agent and per run (max_turns)'),
            ('trace', 'one span per agent, call, tool and handoff'), ('evals', 'per agent and end to end (Chapters 6 to 8)')]
    y = 390
    for i, (k, v) in enumerate(side):
        b += [text(150, y + i * 20, k + ':', 't-note', 'end'), text(160, y + i * 20, v, 't-tick')]
    return svg(760, y + 4 * 20, 'A lead agent fans out to three subagents; each result passes a schema check, then a verifier compares the merged result with the sources, then a human checkpoint approves; budgets, turn limits, traces and evals apply to every part.', b)


F['b_ops'] = ops()


# ---------------------------------------------------------------- overlap and clipping check (as in figs_ch4.py)
def check(F):
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


json.dump(F, open('results/figs_ch5b.json', 'w'))
print(f'{len(F)} figures -> results/figs_ch5b.json:', ', '.join(F))
check(F)
