"""Figures for Chapter 3 (single-agent reasoning patterns). Writes results/figs_ch3.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-note / t-muted, 9 px for t-title, 14 px for t-big; 17 px between baselines.
The drawing helpers (para, block, hpath, circ, two_lines) are the ones of figs_ch1.py and figs_ch2.py, so the chapters look the same.
At the end, check() estimates every text's box and reports overlaps and clipping, so no figure ships with colliding labels."""
import json, math, re
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars
from alammar import frame

R = json.load(open('results/ch3_react.json'))
C = json.load(open('results/ch3_cost.json'))
try:
    X = json.load(open('results/ch3_reflexion.json'))
except FileNotFoundError:
    X = None
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick', rx=10):
    b = [box(x, y, w, h, cls, rx)]
    ty = y + (h / 2 + 5 if not lines else 22)
    b.append(text(x + w / 2, ty, title, tcls, 'middle'))
    b += para(x + w / 2, y + 42, lines, lcls, 'middle')
    return b


def hpath(points, cls='edge', marker='ah'):
    d = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in points)
    return f'<path class="{cls}" d="{d}" marker-end="url(#{marker})"/>'


def circ(x, y, r, cls='node on'):
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def two_lines(W, H, label, xs, series, xlabel, ylabel, ymax, yticks, xticks, notes=(), right=210, fmt=lambda v: f'{v:,.0f}', notes_at='top'):
    """Two or more series on a shared x axis, direct end labels, note lines at the top right."""
    L, T, B = 70, 36, 50
    pw, ph = W - L - right, H - T - B
    X = lambda v: L + (v - xs[0]) / (xs[-1] - xs[0]) * pw
    Y = lambda v: T + ph - v / ymax * ph
    b = []
    for t in yticks:
        b += [line(L, Y(t), L + pw, Y(t), 'grid'), text(L - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
    for v in xticks:
        b.append(text(X(v), T + ph + 20, f'{v}', 't-tick', 'middle'))
    b.append(line(L, T + ph, L + pw, T + ph, 'axis'))
    ends = []
    for i, (name, vals) in enumerate(series):
        dc = f's{i + 1}'
        b.append(f'<polyline class="edge-{i + 1}" points="' + ' '.join(f'{X(x):.1f},{Y(v):.1f}' for x, v in zip(xs, vals)) + '"/>')
        for x, v in zip(xs, vals):
            b.append(f'<g class="mark"><title>{esc(name)}, {x}: {fmt(v)}</title><circle class="{dc} ring" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="4.5"/></g>')
        ends.append([Y(vals[-1]), name, dc, Y(vals[-1])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
    for ly, lab, dc, ty in ends:
        b.append(line(L + pw + 6, ty, L + pw + 16, ly, 'grid'))
        b.append(f'<rect class="{dc}" x="{L + pw + 20}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
        b.append(text(L + pw + 36, ly + 4, lab, 't-tick'))
    b.append(text(L + pw / 2, H - 10, xlabel, 't-tick', 'middle'))
    b.append(text(L - 56, T - 14, ylabel, 't-tick'))
    ny = {'top': T + 12, 'middle': T + ph / 2 - 8, 'bottom': T + ph - 17 * len(notes) + 6}[notes_at]
    for i, n in enumerate(notes):
        b.append(text(L + pw + 20, ny + 17 * i, n, 't-muted'))
    return svg(W, H, label, b)


# ---------------------------------------------------------------- 1. the space of patterns
def pattern_space():
    b = [text(380, 22, 'Six shapes of the loop, same model and same tools', 't-title', 'middle')]
    cells = [('think only', 'box-2', 'chain of thought'), ('act only', 'box-3', 'tool calls, no thoughts'), ('interleave', 'box-2', 'ReAct: think, act, observe'),
             ('plan first', 'box-4', 'plan, execute, solve'), ('reflect', 'box-1', 'try, check, critique, retry'), ('search', 'box-4', 'branch, score, prune')]
    cw, x0, y0, fh = 243, 24, 44, 152
    for i, (name, cls, sub) in enumerate(cells):
        col, row = i % 3, i // 3
        x, y = x0 + col * cw, y0 + row * (fh + 14)
        b += [box(x, y, cw - 14, fh, 'box-ghost', 12), text(x + 14, y + 20, name, 't-note'), text(x + 14, y + 37, sub, 't-muted')]
        cx, cy = x + (cw - 14) / 2, y + 92
        if name == 'think only':
            b += [box(cx - 96, cy - 14, 40, 28, 'box', 6), text(cx - 76, cy + 4, 'q', 't-tick', 'middle'), arrow(cx - 56, cy, cx - 40, cy),
                  box(cx - 40, cy - 14, 76, 28, cls, 6), text(cx - 2, cy + 4, 'thoughts', 't-tick', 'middle'), arrow(cx + 36, cy, cx + 52, cy),
                  box(cx + 52, cy - 14, 44, 28, 'box-on', 6), text(cx + 74, cy + 4, 'ans', 't-tick', 'middle')]
        elif name == 'act only':
            for k in range(3):
                xx = cx - 100 + k * 60
                b += [box(xx, cy - 14, 44, 28, cls, 6), text(xx + 22, cy + 4, 'call', 't-tick', 'middle')]
                if k < 2:
                    b.append(arrow(xx + 44, cy, xx + 60, cy))
            b += [arrow(cx + 64, cy, cx + 78, cy), text(cx + 82, cy + 4, 'ans', 't-tick')]
        elif name == 'interleave':
            for k, (lab, c) in enumerate([('think', 'box-2'), ('act', 'box-3'), ('observe', 'box')]):
                xx = cx - 104 + k * 72
                b += [box(xx, cy - 14, 60, 28, c, 6), text(xx + 30, cy + 4, lab, 't-tick', 'middle')]
                if k < 2:
                    b.append(arrow(xx + 60, cy, xx + 72, cy))
            b.append(hpath([(cx + 70, cy + 14), (cx + 70, cy + 34), (cx - 74, cy + 34), (cx - 74, cy + 16)], 'path', 'ah-on'))
            b.append(text(cx, cy + 50, 'repeat until done', 't-muted', 'middle'))
        elif name == 'plan first':
            b += [box(cx - 108, cy - 14, 48, 28, 'box-4', 6), text(cx - 84, cy + 4, 'plan', 't-tick', 'middle')]
            for k in range(3):
                yy = cy - 32 + k * 26
                b += [arrow(cx - 60, cy, cx - 36, yy + 9), box(cx - 36, yy - 2, 54, 22, 'box-3', 5), text(cx - 9, yy + 13, f'step {k + 1}', 't-tick', 'middle'),
                      arrow(cx + 18, yy + 9, cx + 42, cy)]
            b += [box(cx + 42, cy - 14, 56, 28, 'box-on', 6), text(cx + 70, cy + 4, 'solve', 't-tick', 'middle')]
        elif name == 'reflect':
            b += [box(cx - 108, cy - 14, 60, 28, 'box-2', 6), text(cx - 78, cy + 4, 'attempt', 't-tick', 'middle'), arrow(cx - 48, cy, cx - 34, cy),
                  box(cx - 34, cy - 14, 52, 28, 'box-3', 6), text(cx - 8, cy + 4, 'check', 't-tick', 'middle'), arrow(cx + 18, cy, cx + 32, cy),
                  box(cx + 32, cy - 14, 66, 28, cls, 6), text(cx + 65, cy + 4, 'critique', 't-tick', 'middle')]
            b.append(hpath([(cx + 65, cy + 14), (cx + 65, cy + 34), (cx - 78, cy + 34), (cx - 78, cy + 16)], 'path', 'ah-on'))
            b.append(text(cx, cy + 50, 'retry with the critique', 't-muted', 'middle'))
        else:
            b += [circ(cx - 20, cy - 36, 7, 'node on')]
            for k, xx in enumerate([cx - 80, cx - 20, cx + 40]):
                b += [line(cx - 20, cy - 29, xx, cy - 6, 'edge'), circ(xx, cy, 7, 'node on' if k == 1 else 'node')]
                for j, x2 in enumerate([xx - 18, xx + 18]):
                    on = k == 1 and j == 1
                    b += [line(xx, cy + 7, x2, cy + 30, 'edge' if on else 'edge-dim'), circ(x2, cy + 36, 6, 'node on' if on else 'node')]
            b += [text(cx + 108, cy - 2, 'score,', 't-muted', 'end'), text(cx + 108, cy + 15, 'keep best', 't-muted', 'end')]
    y = y0 + 2 * (fh + 14) + 8
    b += para(380, y, ['each shape adds something to the one before (a scratchpad, tools, a loop, a plan, a critic, a tree),',
                       'and each addition costs calls, tokens and seconds. This chapter asks when each one pays for itself.'], 't-muted', 'middle')
    return svg(760, y + 28, 'Six shapes of the agent loop with the same model and tools: think only (chain of thought), act only, interleave (ReAct), plan first, reflect (try, check, critique, retry) and search (branch, score, prune).', b)


F['ch3_pattern_space'] = pattern_space()


# ---------------------------------------------------------------- 2. chain of thought and self-consistency
def cot():
    b = [text(190, 22, 'standard prompting', 't-title', 'middle'), text(560, 22, 'chain of thought', 't-title', 'middle'), line(380, 10, 380, 190, 'edge-dim')]
    b += block(40, 50, 120, 50, 'question', cls='box')
    b += [arrow(160, 75, 220, 75)]
    b += block(220, 50, 130, 50, 'answer', ['"The answer is 27."'], 'box-on')
    b += para(190, 130, ['one call, about 30 output tokens;', 'the arithmetic has to happen inside', 'the model, with nothing written down'], 't-muted', 'middle')
    b += block(410, 50, 90, 50, 'question', cls='box')
    b += [arrow(500, 75, 526, 75)]
    b += block(526, 50, 130, 50, 'reasoning', ['"23 - 20 = 3; 3 + 6 = 9"'], 'box-2')
    b += [arrow(656, 75, 682, 75)]
    b += block(682, 50, 58, 50, 'answer', cls='box-on')
    b += para(575, 130, ['one call, hundreds of output tokens;', 'each intermediate result is written', 'and read back before the next step'], 't-muted', 'middle')
    b += [line(30, 200, 730, 200, 'edge-dim'), text(380, 224, 'self-consistency: sample several chains, take the majority answer', 't-title', 'middle')]
    y = 248
    b += block(40, y, 100, 40, 'question', cls='box')
    answers = ['9', '9', '27', '9', '12']
    for k, a in enumerate(answers):
        xx = 196 + k * 90
        b += [arrow(140, y + 20, xx, y + 20), box(xx, y + 4, 66, 32, 'box-2', 6), text(xx + 33, y + 24, f'chain {k + 1}', 't-tick', 'middle')]
        b += [arrow(xx + 33, y + 36, xx + 33, y + 56), text(xx + 33, y + 72, a, 't-strong' if a == '9' else 't-tick', 'middle')]
    b += [box(656, y - 2, 84, 44, 'box-on', 8), text(698, y + 16, 'vote', 't-tick', 'middle'), text(698, y + 32, '9 (3 of 5)', 't-tick', 'middle')]
    b += para(380, y + 102, ['five chains cost five times the tokens of one. The paper reports +17.9 points on GSM8K for',
                             'PaLM-540B with 40 samples. It helps when answers can be compared exactly (a number, a label);',
                             'it cannot help when every chain makes the same mistake.'], 't-muted', 'middle')
    return svg(760, y + 142, 'Standard prompting against chain of thought (the model writes its intermediate steps before the answer, at the price of more output tokens), and self-consistency (several sampled chains and a majority vote over their answers).', b)


F['ch3_cot'] = cot()


# ---------------------------------------------------------------- 3. what reasoning models changed
def reasoning_models():
    left = [('who writes the reasoning', 'the prompt asks for it ("think step by step")'), ('where it appears', 'in the visible output, as ordinary tokens'),
            ('how it was learned', 'imitation of human-written steps, or none'), ('what you pay', 'output tokens you can see and count'),
            ('what you can inspect', 'the whole chain (faithful or not)'), ('what you control', 'format, length, examples, when to think')]
    right = [('who writes the reasoning', 'the model, trained by reinforcement learning'), ('where it appears', 'hidden or summarised; the answer is shown'),
             ('how it was learned', 'rewarded for verifiable answers (maths, code)'), ('what you pay', 'reasoning tokens billed as output, unseen'),
             ('what you can inspect', 'a summary, if any; not the raw chain'), ('what you control', 'an effort setting; little else')]
    b = [text(190, 22, 'prompted chain of thought (2022 to 2024)', 't-title', 'middle'), text(570, 22, 'reasoning models (late 2024 onwards)', 't-title', 'middle'),
         line(380, 10, 380, 300, 'edge-dim')]
    for x0, items, cls in [(30, left, 'box-2'), (410, right, 'box-4')]:
        b.append(box(x0, 40, 320, 26 + 38 * len(items), cls, 12))
        for i, (k, v) in enumerate(items):
            y = 62 + 38 * i
            b += [text(x0 + 14, y, k, 't-muted'), text(x0 + 14, y + 17, v, 't-tick')]
    y = 40 + 26 + 38 * len(left) + 24
    b += para(380, y, ['what did not change: a loop still needs tools, a stop condition, a budget and a trace;',
                       'a wrong answer is still wrong; and the counter-evidence later in this section (unfaithful',
                       'explanations, weak self-correction) was measured on the visible kind of reasoning.'], 't-muted', 'middle')
    return svg(760, y + 46, 'What reasoning models changed about chain of thought (who writes it, where it appears, how it was learned, what you pay, what you can inspect and control) and what did not change for an agent built around one.', b)


F['ch3_reasoning_models'] = reasoning_models()


# ---------------------------------------------------------------- 4. the ReAct loop and its two formats
def react_loop():
    b = [text(380, 22, 'ReAct: thought, action, observation, repeated', 't-title', 'middle')]
    pos = {'Thought': (190, 72), 'Action': (280, 170), 'Observation': (100, 170)}
    for name, cls in [('Thought', 'box-2'), ('Action', 'box-3'), ('Observation', 'box')]:
        x, y = pos[name]
        b += [box(x - 50, y - 15, 100, 30, cls, 8), text(x, y + 4.5, name, 't-tick', 'middle')]
    b += [arrow(226, 88, 258, 154, True), arrow(230, 170, 150, 170, True), arrow(122, 154, 154, 88, True)]
    b += [text(258, 118, 'write', 't-muted'), text(190, 188, 'run the tool', 't-muted', 'middle'), text(122, 118, 'read', 't-muted', 'end')]
    b += para(190, 226, ['the model writes a thought and an action;', 'the loop runs the tool and appends the', 'observation; exit with Action: finish[answer]'], 't-muted', 'middle')
    b += [box(390, 44, 350, 112, 'box-ghost', 10), text(565, 64, 'the paper\'s text format (2022)', 't-note', 'middle')]
    b += para(404, 86, ['Thought: I need the depot\'s opening year.', 'Action: wiki[Porto depot]', 'Observation: depot opened 2017. Trucks: 64.',
                        'Thought: ... Action: finish[3]'], 't-tick')
    b += [box(390, 170, 350, 112, 'box-ghost', 10), text(565, 190, 'the same loop with function calling (now)', 't-note', 'middle')]
    b += para(404, 212, ['assistant: tool_use wiki {name: "Porto depot"}', 'user: tool_result "depot opened 2017 ..."',
                         'the thought is optional text before the call,', 'or hidden reasoning tokens, or absent'], 't-tick')
    b += para(380, 308, ['the loop shape is identical; what changed is who parses the action (a regular expression then,',
                         'the API now) and whether the thought is shown at all.'], 't-muted', 'middle')
    return svg(760, 340, 'The ReAct loop (thought, action, observation, repeated until a finish action) and its two surface forms: the paper\'s plain-text format parsed by a regular expression, and the same loop with function calling, where the thought is optional, hidden or absent.', b)


F['ch3_react_loop'] = react_loop()


# ---------------------------------------------------------------- 5. our own results: direct, CoT, ReAct, ReAct+
def react_results():
    S = R['summary']
    conds = [('direct answer', 'direct'), ('chain of thought', 'cot'), ('ReAct, plain loop', 'react'), ('ReAct + guard + better tool', 'react+')]
    b = [text(380, 22, f'Twelve two-hop questions, {R["model"]} (ch3_react.py)', 't-title', 'middle')]
    b.append(text(160, 52, 'accuracy, all 12 questions', 't-note', 'middle'))
    b.append(text(560, 52, 'cost per correct answer, US cents', 't-note', 'middle'))
    items = [(n, S[k]['acc'] * 100) for n, k in conds]
    b += hbars(30, 66, items, 150, row_h=30, cls='s1', label_w=210, fmt=lambda v: f'{v:.0f}%', vmax=100)
    cpc = [('', (S[k]['cost_per_correct'] or 0) * 100) for n, k in conds]
    b += hbars(420, 66, cpc, 190, row_h=30, cls='s3', label_w=16, fmt=lambda v: (f'{v:.2f}c' if v else 'none correct'), vmax=max(v for _, v in cpc) * 1.15)
    y = 66 + 4 * 30 + 16
    rows = [f'{n}: {S[k]["calls"]} calls, {S[k]["input_tokens"]:,} in + {S[k]["output_tokens"]:,} out tokens, ${S[k]["cost_usd"]:.4f}' for n, k in conds]
    b += para(30, y, rows, 't-muted')
    y += 17 * len(rows) + 8
    b += para(30, y, ['public questions (world facts plus arithmetic): a scratchpad helps; private questions (a fictional',
                      'company): only acting helps. The plain loop lost half the questions to one failure, repeating an',
                      'action it had already run; a loop guard and a tool that lists related entries fixed it. Twelve',
                      'sampled questions: the ordering is stable across runs; the exact percentages move a little.'], 't-muted')
    return svg(760, y + 17 * 4 + 10, 'Our own run of four conditions on twelve two-hop questions with gpt-5-mini: accuracy on the left, cost per correct answer on the right, and the token and call counts underneath. Direct answering got none right; chain of thought helped only where the facts were in the model; the plain ReAct loop was undone by repeated actions; a loop guard and a tool that lists related entries fixed it.', b)


F['ch3_react_results'] = react_results()


# ---------------------------------------------------------------- 6. the repetition failure and the guard
def react_failure():
    b = [text(380, 22, 'The most common ReAct failure, and where the fix lives', 't-title', 'middle')]
    frames_ = [('step 1', ['Thought: need both years', 'Action: wiki[Sputnik 1]', 'Obs: launched 1957']),
               ('step 2', ['Thought: 1969 - 1957', 'Action: calc[1969-1957]', 'Obs: 12']),
               ('step 3', ['Thought: 1969 - 1957', 'Action: calc[1969-1957]', 'Obs: 12']),
               ('steps 4 to 8', ['the same thought', 'the same action', 'budget exhausted: wrong'])]
    for i, (t, lines) in enumerate(frames_):
        x = 30 + i * 182
        b += frame(x, 44, 170, 112, i + 1, t, 'box' if i < 2 else 'box-4')
        b += para(x + 14, 92, lines, 't-tick')
        if i < 3:
            b.append(arrow(x + 170, 100, x + 182, 100))
    b += para(380, 178, ['the answer was in the context after step 2; the model kept regenerating its last thought',
                         'instead of finishing (the "reasoning error" row of the ReAct paper\'s Table 2)'], 't-muted', 'middle')
    b += [line(30, 212, 730, 212, 'edge-dim'), text(380, 236, 'two fixes that are not "a bigger model"', 't-title', 'middle')]
    b += block(30, 252, 340, 96, 'the loop: a repeated-action guard', ['the loop sees the same (tool, args) twice and', 'replaces the observation with "you have been',
                                                                       'round this loop; using what you have, finish now"'], 'box-4')
    b += block(390, 252, 340, 96, 'the tool: observations that name next steps', ['a company entry lists its depots as look-up names;', 'several matches come back as entries, not errors;',
                                                                                 'a miss explains how entries are named'], 'box-3')
    b.append(text(380, 372, f'on the twelve questions: {R["summary"]["react"]["acc"]:.0%} to {R["summary"]["react+"]["acc"]:.0%}, with fewer calls and lower cost (Section 3.3)', 't-muted', 'middle'))
    return svg(760, 388, 'The repetition failure of a plain ReAct loop (the answer is available after step 2, the model repeats the action until the step budget runs out) and the two fixes that live in the loop and the tool rather than in the model: a repeated-action guard that forces the finish, and tool observations that name the next things to look up.', b)


F['ch3_react_failure'] = react_failure()


# ---------------------------------------------------------------- 7. plan-and-execute against ReAct
def plan_execute():
    b = [text(190, 22, 'ReAct: one growing context', 't-title', 'middle'), text(562, 22, 'plan-and-execute: plan, then small contexts', 't-title', 'middle'),
         line(380, 10, 380, 300, 'edge-dim')]
    for k in range(4):
        w = 90 + k * 60
        y = 48 + k * 46
        lab = f'call {k + 1}' + (f': prompt + {k} thoughts + {k} obs' if k >= 2 else (f' + {k} obs' if k else ''))
        b += [box(40, y, w, 34, 'box-2', 6), text(40 + w / 2, y + 21, lab, 't-tick', 'middle')]
        if k < 3:
            b += [arrow(40 + w + 6, y + 17, 40 + w + 30, y + 17), text(40 + w + 34, y + 21, 'tool', 't-muted')]
    b += para(190, 246, ['every call re-reads everything before it:', f'{C["growth"][1]["react_input"]:,} input tokens for 5 steps,', f'{C["growth"][3]["react_input"]:,} for 20 (ch3_cost.py)'], 't-muted', 'middle')
    b += block(410, 48, 110, 56, 'planner', ['one big call'], 'box-4')
    b += para(530, 60, ['#E1 = wiki[Porto depot]', '#E2 = wiki[Leeds depot]', '#E3 = calc[#E2.year - #E1.year]'], 't-tick')
    for k in range(3):
        y = 130 + k * 40
        b += [arrow(465, 104, 440 + k * 100 + 40, y), box(440 + k * 100, y, 80, 30, 'box-3', 6), text(480 + k * 100, y + 19, f'worker {k + 1}', 't-tick', 'middle')]
        b.append(arrow(480 + k * 100, y + 30, 570, 262))
    b += block(510, 262, 120, 40, 'solver', cls='box-on')
    b += para(570, 322, ['plan once, run the steps with small contexts', '(in parallel where the plan allows), answer once:',
                         f'{C["growth"][1]["plan_exec_input"]:,} input tokens for 5 steps, {C["growth"][3]["plan_exec_input"]:,} for 20.',
                         'the price: a wrong plan is noticed only at the end'], 't-muted', 'middle')
    return svg(760, 392, 'ReAct re-reads a growing context on every call; plan-and-execute (ReWOO, LLMCompiler) writes the whole plan in one call, runs the steps with small contexts and in parallel where their dependencies allow, and answers once. The token counts come from the planning numbers in ch3_cost.py.', b)


F['ch3_plan_execute'] = plan_execute()


# ---------------------------------------------------------------- 8. input tokens against step count
def cost_growth():
    g = C['growth']
    xs = [r['steps'] for r in g]
    return two_lines(760, 320, 'Input tokens a task costs against the number of tool steps, for a ReAct loop that re-reads its history and a plan-and-execute design with small per-step contexts (planning numbers of ch3_cost.py).',
                     xs, [('ReAct, re-reads history', [r['react_input'] for r in g]), ('plan-and-execute', [r['plan_exec_input'] for r in g])],
                     'tool steps in the task', 'input tokens per task', 140000, [0, 25000, 50000, 75000, 100000, 125000], xs,
                     notes=['planning numbers: 1,500 prompt,', '150 thought, 300 observation', f'ratio {g[-1]["react_input"] / g[-1]["plan_exec_input"]:.1f}x at 20 steps'], right=250, notes_at='middle')


F['ch3_cost_growth'] = cost_growth()


# ---------------------------------------------------------------- 9. reflection: weak and strong critics
def reflection():
    b = [text(380, 22, 'Reflection: try, check, critique, retry. The check decides whether it works.', 't-title', 'middle')]
    b += block(30, 50, 120, 56, 'actor', ['writes an attempt'], 'box-2')
    b += [arrow(150, 78, 200, 78)]
    b += block(200, 50, 150, 56, 'evaluator', ['pass or fail'], 'box-3')
    b += [arrow(350, 78, 400, 78)]
    b += block(400, 50, 150, 56, 'self-reflection', ['why it failed, in words'], 'box-1')
    b += [arrow(550, 78, 600, 78)]
    b += block(600, 50, 130, 56, 'memory', ['reflections so far'], 'box-1')
    b.append(hpath([(665, 106), (665, 130), (90, 130), (90, 106)], 'path', 'ah-on'))
    b.append(text(380, 146, 'next attempt, with the reflections in context (Reflexion, Figure 2)', 't-muted', 'middle'))
    b += [line(30, 164, 730, 164, 'edge-dim')]
    b += [text(200, 188, 'weak check: the model judges itself', 't-title', 'middle'), text(560, 188, 'strong check: an external signal', 't-title', 'middle')]
    b.append(box(30, 200, 340, 118, 'box-ghost', 10))
    b += para(44, 222, ['"review your answer and improve it"', 'no new information enters the loop', 'Huang et al.: accuracy fell on every benchmark',
                        'Self-Refine: no gain on maths', 'useful for style, format, checklist coverage'], 't-tick')
    b.append(box(390, 200, 340, 118, 'box-3', 10))
    b += para(404, 222, ['unit tests, a compiler, a linter, a type checker', 'a search result, a database row, a calculator', 'a human review, a user\'s reply',
                         'Reflexion: HumanEval 80.1 to 91.0 with tests', 'CRITIC: tools as critic; without them, no gain'], 't-tick')
    b += para(380, 342, ['the loop is the same; the signal is what you pay for. If there is no external signal,',
                         'build one before you add a retry.'], 't-muted', 'middle')
    return svg(760, 372, 'The reflection loop (actor, evaluator, verbal self-reflection, memory, retry) and the distinction that decides whether it helps: a weak check where the model judges itself against a strong check where an external signal such as tests, a compiler, a search result or a person tells it what is wrong.', b)


F['ch3_reflection'] = reflection()


# ---------------------------------------------------------------- 10. our reflexion results
def reflexion_results():
    if not X or X.get('stub'):
        return None
    S = X['summary']
    n = S['n']
    b = [text(380, 22, f'{len(X["tasks"])} coding tasks x {X["samples"]} samples, {X["model"]}, hidden tests (ch3_reflexion.py)', 't-title', 'middle')]
    items = [('passed on attempt 1', S['pass1'] / n * 100), ('after self-critique only', S['self_only'] / n * 100), ('after tests fed back', S['reflexion'] / n * 100)]
    b += hbars(30, 56, items, 300, row_h=32, cls='s2', label_w=190, fmt=lambda v: f'{v:.0f}%', vmax=100)
    y = 56 + 3 * 32 + 14
    b += para(30, y, [f'{n} first attempts; {S["pass1"]} passed. Of the {S["failures"]} failures, self-critique alone (no test output shown)',
                      f'fixed {S["self_only_fixed"]}; feeding the failing test back fixed {S["reflexion_fixed"]}. Both branches started from the same failed',
                      f'attempt with the same budget of {X["max_attempts"]} attempts. Whole experiment: {S["calls"]} API calls, ${S["cost_usd"]:.3f}.'], 't-muted')
    return svg(760, y + 17 * 3 + 8, 'Our own reflection experiment: ten small coding tasks with hidden unit tests, three samples each; every failed first attempt was continued both by self-critique alone and by feeding back the failing test, from the same starting point.', b)


r = reflexion_results()
if r:
    F['ch3_reflexion_results'] = r


# ---------------------------------------------------------------- 11. tree of thoughts
def tot():
    b = [text(380, 22, 'Search over reasoning: propose, score, keep the best, repeat (b = 3, depth = 3)', 't-title', 'middle')]
    root = (200, 60)
    b += [circ(*root, 9, 'node on'), text(200, 44, 'problem', 't-muted', 'middle')]
    lvl1 = [(90, 130), (200, 130), (310, 130)]
    scores1 = ['sure', 'maybe', 'impossible']
    for (x, y), s in zip(lvl1, scores1):
        on = s != 'impossible'
        b += [line(root[0], root[1] + 9, x, y - 9, 'edge' if on else 'edge-dim'), circ(x, y, 9, 'node on' if on else 'node'), text(x, y + 26, s, 't-tick' if on else 't-muted', 'middle')]
    lvl2 = {0: [(50, 210), (90, 210), (130, 210)], 1: [(170, 210), (200, 210), (230, 210)]}
    s2 = {0: ['maybe', 'sure', 'impossible'], 1: ['impossible', 'maybe', 'maybe']}
    for i, kids in lvl2.items():
        px, py = lvl1[i]
        for (x, y), s in zip(kids, s2[i]):
            on = s != 'impossible'
            b += [line(px, py + 9, x, y - 8, 'edge' if on else 'edge-dim'), circ(x, y, 8, 'node on' if on else 'node')]
    for (x, y) in [(70, 284), (90, 284), (110, 284)]:
        b += [line(90, 218, x, y - 7, 'edge'), circ(x, y, 7, 'node on' if x == 90 else 'node')]
    b += [text(90, 308, 'answer', 't-tick', 'middle'), text(210, 330, 'every node scored sure, maybe or impossible by the model', 't-muted', 'middle')]
    b += [box(400, 44, 330, 290, 'box-ghost', 12), text(565, 66, 'what the tree costs (ch3_cost.py, breadth-first)', 't-note', 'middle')]
    rows_ = ['per level: 3 propose calls + 9 evaluation calls', f'three levels: {3 * 3 + 9 * 3} model calls for one task',
             f'planning numbers: {C["rows"][4]["inp"]:,} in + {C["rows"][4]["out"]:,} out tokens', f'about {C["rows"][4]["cost_frontier"] / C["rows"][1]["cost_frontier"]:.0f}x the cost of one chain of thought',
             f'and {C["rows"][4]["lat"] / C["rows"][1]["lat"]:.1f}x its latency (levels are sequential)', '',
             'the paper: Game of 24, GPT-4', 'chain of thought 4%, best of 100 chains 49%,', 'tree of thoughts b=5: 74% (Table 2);',
             'about 5.5k completion tokens per problem,', 'close to 100 chain-of-thought trials (Table 7)']
    b += para(414, 90, rows_, 't-tick')
    b += para(380, 356, ['worth it when steps can be scored before the end (a puzzle state, a test, a type check) and',
                         'single chains fail early; not for open-ended writing or routine tool use'], 't-muted', 'middle')
    return svg(760, 386, 'Tree of thoughts: at each level the model proposes several next steps, scores each as sure, maybe or impossible, and only the promising ones are expanded; on the right, the call and token arithmetic for breadth 3 and depth 3 and the paper\'s own Game of 24 numbers.', b)


F['ch3_tot'] = tot()


# ---------------------------------------------------------------- 12. cost and latency of the patterns, one task
def pattern_cost():
    rows = C['rows']
    base = rows[0]
    b = [text(380, 22, 'One task, seven patterns: cost and latency relative to a direct answer (ch3_cost.py)', 't-title', 'middle')]
    names = [r['pattern'].replace(' with a model verifier', ', model verifier').replace(' with a free checker (tests)', ', free checker').replace(', 5 steps (parallel)', ', 5 parallel steps') for r in rows]
    b.append(text(200, 52, 'cost per task (x direct)', 't-note', 'middle'))
    b.append(text(560, 52, 'latency per task (x direct)', 't-note', 'middle'))
    cost_items = [(n, r['cost_frontier'] / base['cost_frontier']) for n, r in zip(names, rows)]
    lat_items = [('', r['lat'] / base['lat']) for r in rows]
    b += hbars(30, 66, cost_items, 90, row_h=28, cls='s3', label_w=262, fmt=lambda v: f'{v:.1f}x', vmax=max(v for _, v in cost_items) * 1.1)
    b += hbars(420, 66, lat_items, 240, row_h=28, cls='s1', label_w=16, fmt=lambda v: f'{v:.1f}x', vmax=max(v for _, v in lat_items) * 1.1)
    y = 66 + len(rows) * 28 + 14
    b += para(30, y, ['the tree is the outlier on both axes; best-of-n is cheap in time (parallel) and expensive in tokens;',
                      'plan-and-execute buys back latency by running steps in parallel. None of these numbers says which',
                      'pattern is right: divide each by its accuracy first (Section 3.9).'], 't-muted')
    return svg(760, y + 60, 'Cost and latency of seven patterns for one task, relative to a direct answer, from the planning numbers in ch3_cost.py: chain of thought, ReAct with five tool steps, plan-and-execute with five parallel steps, tree of thoughts with breadth 3 and depth 3, and best-of-five with a model verifier or a free checker.', b)


F['ch3_pattern_cost'] = pattern_cost()


# ---------------------------------------------------------------- 13. verifier and retries
def retry_math():
    ret = C['retry']
    ks = [1, 2, 3, 5]
    series = []
    for p in [0.3, 0.5, 0.7]:
        series.append((f'p = {p:.0%} per attempt', [next(r['p_success'] for r in ret if r['p'] == p and r['k'] == k) * 100 for k in ks]))
    return two_lines(760, 300, 'Probability of success within k attempts when a verifier can tell right from wrong, for three per-attempt pass rates.',
                     ks, series, 'attempts allowed (k)', 'P(success within k), %', 100, [0, 25, 50, 75, 100], ks,
                     notes=['P = 1 - (1 - p)^k', 'cost per correct stays c / p', 'for any k (ch3_cost.py part 3)'], right=240, fmt=lambda v: f'{v:.0f}', notes_at='bottom')


F['ch3_retry_math'] = retry_math()


# ---------------------------------------------------------------- 14. code as action and the agentless counterpoint
def code_as_action():
    b = [text(190, 22, 'JSON tool calls: three round trips', 't-title', 'middle'), text(570, 22, 'one code action in a sandbox', 't-title', 'middle'),
         line(380, 10, 380, 226, 'edge-dim')]
    for k, lab in enumerate(['lookup_rate("USD", "EUR")', 'lookup_price("phone", "DE")', 'convert(price, rate)']):
        y = 46 + k * 52
        b += [box(40, y, 180, 34, 'box-3', 6), text(130, y + 21, lab, 't-tick', 'middle'), arrow(220, y + 17, 262, y + 17),
              box(262, y, 90, 34, 'box', 6), text(307, y + 21, 'result', 't-tick', 'middle')]
        if k < 2:
            b.append(hpath([(307, y + 34), (307, y + 44), (130, y + 44), (130, y + 52)], 'edge', 'ah'))
    b += para(190, 212, ['each result passes through the model', 'before the next call can be made'], 't-muted', 'middle')
    b += [box(410, 46, 320, 112, 'box-3', 8)]
    b += para(424, 68, ['rate = lookup_rate("USD", "EUR")', 'for country in ["DE", "FR", "IT"]:', '    p = lookup_price("phone", country)',
                        '    prices[country] = convert(p, rate)', 'print(min(prices.items(), key=...))'], 't-tick')
    b += para(570, 180, ['one round trip; loops and conditions are free;', 'the model reads one line back.', 'CodeAct: up to 20 points higher success, fewer turns'], 't-muted', 'middle')
    b += [line(30, 240, 730, 240, 'edge-dim'), text(380, 264, 'the counterpoint: Agentless, a fixed three-phase pipeline, no agent at all', 't-title', 'middle')]
    for k, (lab, sub, cls) in enumerate([('localise', 'files > functions > lines', 'box-4'), ('repair', 'sample many patches', 'box-2'),
                                          ('validate', 'reproduce, test, rank', 'box-3')]):
        x = 40 + k * 236
        b += [box(x, 280, 200, 56, cls, 8), text(x + 100, 302, lab, 't-note', 'middle'), text(x + 100, 322, sub, 't-tick', 'middle')]
        if k < 2:
            b.append(arrow(x + 200, 308, x + 236, 308))
    b += para(380, 360, ['SWE-bench Lite, mid-2024: 32.00% of issues resolved at $0.70 each,', 'above every open-source agent of the time (Table 1)'], 't-muted', 'middle')
    return svg(760, 392, 'Code as action: three JSON tool calls take three round trips through the model, while one code action composes the same tools with a loop in a sandbox and returns one line. Below, the Agentless counterpoint: a fixed pipeline of localisation, repair and validation that beat the open-source agents of its time on SWE-bench Lite.', b)


F['ch3_code_as_action'] = code_as_action()


# ---------------------------------------------------------------- 15. what companies chose
def companies():
    pats = ['fixed pipeline', 'ReAct loop', 'plan+execute', 'reflect on tests', 'search/best-of-n', 'judge/human gate']
    rows = [('Uber Genie, uReview, FixrLeak', [1, 0, 0, 1, 0, 1]), ('Amazon Q code transformation', [0, 0, 1, 1, 0, 1]), ('Spotify Honk', [0, 1, 0, 1, 0, 1]),
            ('DoorDash support, Flux', [1, 1, 0, 0, 0, 1]), ('LinkedIn SQL Bot, Hiring Asst.', [0, 1, 1, 1, 0, 1]), ('Stripe Minions', [0, 1, 0, 1, 0, 1]),
            ('Airbnb test migration', [1, 0, 0, 1, 0, 0]), ('Anthropic SWE-bench scaffold', [0, 1, 0, 0, 0, 0]), ('Google Deep Research', [0, 1, 1, 0, 0, 1]),
            ('Klarna assistant (as reported)', [0, 1, 0, 0, 0, 1])]
    b = [text(380, 22, 'Patterns described in the production write-ups of Section 3.8 (filled = on the page)', 't-title', 'middle')]
    x0, lw, cw, y0, rh = 30, 220, 78, 134, 26
    for j, p in enumerate(pats):
        cx = x0 + lw + j * cw + cw / 2
        b.append(text(cx - 4, y0 - 10, p, 't-tick', 'start', f' transform="rotate(-50 {cx - 4:.1f} {y0 - 10})"'))
    for i, (name, vals) in enumerate(rows):
        y = y0 + i * rh
        b.append(text(x0 + lw - 10, y + rh / 2 + 4, name, 't-tick', 'end'))
        for j, v in enumerate(vals):
            cx = x0 + lw + j * cw + cw / 2
            b.append(f'<g class="mark"><title>{esc(name)}: {esc(pats[j])}</title><rect class="{"cell" if v else "cell-masked"}" x="{cx - 11:.1f}" y="{y + 3:.1f}" width="22" height="{rh - 6}" rx="4" style="fill-opacity:{0.85 if v else 1}"/></g>')
    y = y0 + len(rows) * rh + 14
    b += para(30, y, ['nearly every row is filled in one of the last two columns: a verifier (tests, CI, a judge model) or a',
                      'human gate. The shape of the loop varied from company to company; the presence of a check did not.'], 't-muted')
    return svg(760, y + 42, 'The patterns described in the production write-ups of Section 3.8, one row per system: fixed pipelines and ReAct loops are both common; almost every system adds an external check (tests, CI, a judge model) or a human gate.', b)


F['ch3_companies'] = companies()


# ---------------------------------------------------------------- 16. choosing a pattern
def decision():
    b = [text(380, 22, 'Choosing a pattern: start simple, add one thing per measured failure', 't-title', 'middle')]
    steps = [('start', 'box', 'one call, or a fixed pipeline if the steps are known', 'measure accuracy, cost and latency on a labelled set'),
             ('wrong on multi-step reasoning?', 'box-2', 'add chain of thought, or a reasoning model', 'cost: output tokens; do not assume faithfulness'),
             ('facts or state not in the model?', 'box-3', 'add tools in a ReAct loop: step budget, repeat guard', 'cost: n calls on a growing context'),
             ('many steps; latency or tokens too high?', 'box-4', 'plan first, execute with small parallel contexts', 'cost: a wrong plan is found late; replan'),
             ('an external check exists?', 'box-1', 'add reflection on tests, compiler, search, human', 'cost: up to k attempts; never self-critique alone'),
             ('early choices decide; states scorable?', 'box-4', 'search: best-of-n with a verifier, or a tree', 'cost: 10x to 100x tokens; puzzles, code with tests')]
    y = 44
    for i, (q, cls, act, note) in enumerate(steps):
        b += [box(30, y, 300, 44, cls, 8), text(180, y + 27, q, 't-tick', 'middle')]
        b += [arrow(330, y + 22, 372, y + 22), text(351, y + 16, 'yes', 't-muted', 'middle')]
        b += [text(380, y + 19, act, 't-tick'), text(380, y + 36, note, 't-muted')]
        if i < len(steps) - 1:
            b += [arrow(180, y + 44, 180, y + 58), text(190, y + 56, 'no, or still failing', 't-muted')]
        y += 58
    b += para(380, y + 6, ['at every arrow the same three numbers are measured again:', 'accuracy on the labelled set, cost per correct answer, p95 latency'], 't-muted', 'middle')
    return svg(760, y + 36, 'A decision ladder for choosing a reasoning pattern: start with one call or a fixed pipeline, and add chain of thought, a tool loop, a plan, an external check or search only when a measured failure demands it, re-measuring accuracy, cost per correct answer and latency at every step.', b)


F['ch3_decision'] = decision()


# ---------------------------------------------------------------- overlap and clipping check
def check(F):
    """Estimate every <text> box from its class and anchor; report texts outside the viewBox and pairs that overlap."""
    PX = {'t-title': 9.0, 't-big': 14.0, 't-strong': 7.6, 't-val': 7.2}
    problems = 0
    for key, s in F.items():
        W, H = (float(v) for v in re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s).groups())
        boxes = []
        for m in re.finditer(r'<text class="([^"]+)" x="([\d.]+)" y="([\d.]+)" text-anchor="(\w+)"([^>]*)>([^<]*)</text>', s):
            cls, x, y, anchor, extra, t = m.groups()
            if not t.strip() or 'rotate' in extra:
                continue
            pw = next((PX[c] for c in cls.split() if c in PX), 7.2)
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


json.dump(F, open('results/figs_ch3.json', 'w'))
print(f'{len(F)} figures -> results/figs_ch3.json:', ', '.join(F))
check(F)
