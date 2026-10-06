"""Figures for Chapter 1 (what an agent is, and when not to build one). Writes results/figs_ch1.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-note / t-muted, 9 px for t-title; 17 px between baselines."""
import json, math
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars

M = json.load(open('results/ch1_math.json'))
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick', rx=10):
    """A rounded block with a centred title and optional centred lines under it."""
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


# ---------------------------------------------------------------- 1. the agent loop, with the on-call example
def loop_fig():
    b = []
    y, h = 95, 60
    b += block(30, y, 150, h, 'context', ['what the model sees'], 'box-1')
    b += block(220, y, 150, h, 'model', ['reads, decides next step'], 'box-2')
    b += block(410, y, 150, h, 'tool call', ['the action'], 'box-3')
    b += [box(600, 75, 140, 100, 'box-ghost', 10), text(670, 95, 'environment', 't-note', 'middle')]
    for i, t in enumerate(['search_logs', 'open_ticket', 'on-call engineer']):
        b.append(text(670, 118 + 17 * i, t, 't-tick', 'middle'))
    b += [arrow(180, y + 30, 220, y + 30), arrow(370, y + 30, 410, y + 30), arrow(560, y + 30, 600, y + 30)]
    b += [text(200, y + 92, 'perceive', 't-muted', 'middle'), text(390, y + 92, 'reason', 't-muted', 'middle'), text(580, y + 92, 'act', 't-muted', 'middle')]
    # the return path: observation appended to the context
    b.append(hpath([(670, 175), (670, 212), (105, 212), (105, 158)], 'path', 'ah-on'))
    b.append(text(390, 232, 'observe: the tool result is appended to the context, and the loop runs again', 't-note', 'middle'))
    # the stop branch
    b += [box(200, 18, 190, 32, 'box', 8), text(295, 39, 'final answer → user, stop', 't-tick', 'middle'), arrow(295, 95, 295, 50)]
    b.append(text(400, 39, 'when the model decides it is done (or the step budget runs out)', 't-muted'))
    # the example, one line under each block
    ex = [(105, 'ALERT: error rate 12%'), (295, '"check the logs first"'), (485, 'search_logs(checkout-api)'), (670, '212 errors; deploy 13:58')]
    for x, s in ex:
        b.append(text(x, 76, s, 't-muted', 'middle'))
    b.append(text(380, 275, 'example, step 1 of the on-call assistant in §1.1: the grey line over each block is what that block held', 't-muted', 'middle'))
    return svg(760, 290, 'The agent loop: the context is given to the model, the model picks a tool call, the tool acts on the environment, and the result is appended to the context for the next round, until the model gives a final answer.', b)


F['ch1_loop'] = loop_fig()


# ---------------------------------------------------------------- 2. model versus agent
def model_vs_agent():
    b = [text(185, 30, 'A model: one call', 't-title', 'middle'), text(570, 30, 'An agent: a goal pursued over many steps', 't-title', 'middle'),
         line(375, 14, 375, 300, 'edge-dim')]
    b += block(30, 110, 90, 50, 'prompt', cls='box')
    b += block(145, 100, 90, 70, 'model', cls='box-2')
    b += block(260, 110, 80, 50, 'text', cls='box')
    b += [arrow(120, 135, 145, 135), arrow(235, 135, 260, 135)]
    b += para(185, 205, ['prompt in, text out', 'no memory of earlier calls', 'no actions, no goal of its own'], 't-muted', 'middle')
    # the agent: a frame with the model inside
    b.append(box(400, 50, 340, 250, 'box-ghost', 12))
    b += [box(420, 64, 300, 30, 'box', 8), text(570, 84, 'goal: investigate the alert, propose a fix', 't-tick', 'middle')]
    b += block(560, 118, 110, 60, 'model', ['reasons'], 'box-2')
    b += block(420, 118, 100, 60, 'tools', ['act, observe'], 'box-3')
    b += block(560, 206, 110, 50, 'memory', ['state so far'], 'box-1')
    b += [arrow(560, 138, 520, 138), arrow(520, 160, 560, 160)]
    b += [text(540, 128, 'act', 't-muted', 'middle'), text(540, 176, 'observe', 't-muted', 'middle')]
    b += [arrow(605, 178, 605, 206), arrow(625, 206, 625, 178), text(650, 196, 'read, write', 't-muted')]
    b += [arrow(570, 94, 570, 118)]
    b += para(475, 205, ['the model is one part;', 'the loop, tools and', 'memory are the rest'], 't-muted', 'middle')
    b += para(570, 272, ['many steps, state carried between them,', 'and the model picks each next step'], 't-muted', 'middle')
    return svg(760, 312, 'A model computes one output from one prompt. An agent wraps a model in a loop with a goal, tools and memory, and pursues the goal over many steps.', b)


F['ch1_model_vs_agent'] = model_vs_agent()


# ---------------------------------------------------------------- 3. the spectrum: script, workflow, agent, multi-agent
def spectrum():
    b = []
    xs = [30, 215, 400, 585]
    items = [('script', 'box', ['code decides', 'every step'], 'cron job, a rules engine'),
             ('workflow + LLM steps', 'box-1', ['code decides the path;', 'the LLM fills in steps'], 'summarise, then translate'),
             ('agent', 'box-2', ['the LLM decides', 'each next step', 'and when to stop'], 'on-call assistant'),
             ('multi-agent', 'box-3', ['several LLMs decide;', 'one coordinates'], 'research system')]
    for x, (name, cls, lines, ex) in zip(xs, items):
        b += block(x, 50, 165, 100, name, lines, cls)
        b.append(text(x + 82, 172, ex, 't-muted', 'middle'))
    # small pictograms inside the blocks: boxes for code, a circle for a model
    b.append(arrow(40, 205, 740, 205))
    b.append(text(390, 228, 'more autonomy, more cost, more variance between runs, harder to test  →', 't-note', 'middle'))
    b.append(text(30, 256, 'who decides the next step:', 't-label'))
    b.append(text(230, 256, 'code', 't-tick'))
    b.append(text(400, 256, 'code, with LLM help', 't-tick'))
    b.append(text(585, 256, 'the LLM(s)', 't-tick'))
    return svg(760, 272, 'The spectrum from a plain script to a multi-agent system. Moving right, the model takes over more of the decisions about which step comes next, and cost, variance and testing effort all rise.', b)


F['ch1_spectrum'] = spectrum()


# ---------------------------------------------------------------- 4. workflow versus agent paths
def workflow_vs_agent():
    b = [text(190, 30, 'Workflow: the path is written in code', 't-title', 'middle'),
         text(570, 30, 'Agent: the model chooses the path', 't-title', 'middle'), line(375, 14, 375, 352, 'edge-dim')]
    steps = [('input: a support email', 'box'), ('LLM: classify the topic', 'box-2'), ('code: look up the account', 'box-1'),
             ('LLM: draft a reply', 'box-2'), ('code: queue for review', 'box-1')]
    for i, (s, cls) in enumerate(steps):
        y = 56 + i * 54
        b += [box(80, y, 220, 34, cls, 8), text(190, y + 22, s, 't-tick', 'middle')]
        if i < len(steps) - 1:
            b.append(arrow(190, y + 34, 190, y + 54))
    b.append(text(190, 322, 'same five steps on every run; easy to test step by step', 't-muted', 'middle'))
    # the agent: model in the middle, tools around, a chosen order
    mx, my = 570, 185
    b += block(mx - 60, my - 28, 120, 56, 'model', ['picks a tool'], 'box-2')
    tools = [('read_email', 420, 70, 1), ('lookup_account', 660, 70, 2), ('search_kb', 410, 300, 3), ('draft_reply', 670, 300, 4)]
    for name, x, y, k in tools:
        b += [box(x - 50, y - 17, 100, 34, 'box-3', 8), text(x, y + 5, name, 't-tick', 'middle')]
        dx, dy = x - mx, y - my
        L = math.hypot(dx, dy)
        ux, uy = dx / L, dy / L
        b.append(f'<line class="path" x1="{mx + ux * 62:.1f}" y1="{my + uy * 34:.1f}" x2="{x - ux * 54:.1f}" y2="{y - uy * 20:.1f}" marker-end="url(#ah-on)"/>')
        b += [circ(mx + ux * 95, my + uy * 60, 10), text(mx + ux * 95, my + uy * 60 + 4, str(k), 't-tick', 'middle')]
    b += [box(522, 262, 96, 30, 'box', 8), text(570, 282, 'stop: answer', 't-tick', 'middle'), arrow(570, 213, 570, 262)]
    b.append(text(570, 345, 'order, count and stop chosen at run time; test the whole trajectory', 't-muted', 'middle'))
    return svg(760, 358, 'Left: a workflow runs the same code path on every input, calling the model inside fixed steps. Right: an agent gives the model a set of tools and lets it choose the order, the number of steps and when to stop.', b)


F['ch1_workflow_vs_agent'] = workflow_vs_agent()


# ---------------------------------------------------------------- 5. the timeline, 2022 to 2025
def timeline():
    ax0, ax1, ay = 60, 730, 120
    def X(m):                      # months after October 2022
        return ax0 + m * (ax1 - 20 - ax0) / 32
    b = [line(ax0 - 20, ay, ax1, ay, 'axis', 'ah')]
    for m, yr in [(3, '2023'), (15, '2024'), (27, '2025')]:
        b += [line(X(m), ay - 5, X(m), ay + 5, 'axis'), text(X(m), ay + 20, yr, 't-tick', 'middle')]
    ev = [  # (months, name, date, gist, side, label x, anchor, class)
        (0, 'ReAct', 'Oct 2022', 'reason, act, observe', 'up', 40, 'start', 's1'),
        (4, 'Toolformer', 'Feb 2023', 'a model taught to call APIs', 'down', 70, 'start', 's1'),
        (5, 'Reflexion', 'Mar 2023', 'verbal self-reflection, retry', 'up', 290, 'middle', 's1'),
        (6, 'Generative Agents', 'Apr 2023', '25 agents in a small town', 'down', 380, 'middle', 's1'),
        (26, 'Building effective agents', 'Dec 2024', 'workflows versus agents', 'up', 465, 'middle', 's2'),
        (29, 'MAST', 'Mar 2025', 'why multi-agent systems fail', 'down', 600, 'middle', 's1'),
        (32, 'Multi-agent research system', 'Jun 2025', 'orchestrator and subagents', 'up', 745, 'end', 's2'),
    ]
    for m, name, date, gist, side, lx, anc, cls in ev:
        x = X(m)
        b.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{ay}" r="7"/>')
        tx = lx if anc != 'middle' else lx
        if side == 'up':
            b.append(line(x, ay - 8, tx if anc == 'middle' else (lx + 60 if anc == 'start' else lx - 80), 72, 'edge-dim'))
            yy = 30
        else:
            b.append(line(x, ay + 8, tx if anc == 'middle' else lx + 60, 168, 'edge-dim'))
            yy = 186
        b += [text(tx, yy, name, 't-note', anc), text(tx, yy + 17, date, 't-muted', anc), text(tx, yy + 34, gist, 't-muted', anc)]
    b += [f'<circle class="s1" cx="60" cy="258" r="6"/>', text(72, 262, 'research paper', 't-tick'),
          f'<circle class="s2" cx="200" cy="258" r="6"/>', text(212, 262, 'engineering blog post', 't-tick')]
    return svg(760, 272, 'Timeline: ReAct (October 2022) made the reason-act-observe loop explicit; Toolformer, Reflexion and Generative Agents followed within six months in 2023; the production write-ups (workflows versus agents, the failure taxonomy, the multi-agent research system) arrived in 2024 and 2025.', b)


F['ch1_timeline'] = timeline()


# ---------------------------------------------------------------- 6. the autonomy ladder
def autonomy():
    b = []
    rungs = [('1  suggest', ['drafts a reply;', 'a person sends it'], ['a wrong draft is read,', 'not acted on'], 'box'),
             ('2  act with approval', ['proposes a rollback;', 'a person clicks yes'], ['a person is the', 'last check'], 'box-1'),
             ('3  act and report', ['rolls back, then', 'posts what it did'], ['wrong actions happen', 'and are seen fast'], 'box-2'),
             ('4  fully autonomous', ['acts; nobody looks', 'unless alerted'], ['wrong actions', 'compound unseen'], 'box-3')]
    for i, (name, ex, risk, cls) in enumerate(rungs):
        x, y = 40 + i * 180, 190 - i * 50
        b += [box(x, y, 165, 48, cls, 10), text(x + 82, y + 30, name, 't-note', 'middle')]
        b += para(x + 82, y + 68, ex, 't-tick', 'middle')
        b += para(x + 82, y + 106, risk, 't-muted', 'middle')
        if i < 3:
            b.append(arrow(x + 165, y + 24, x + 180, y - 26 + 24))
    b.append(arrow(40, 345, 740, 345))
    b.append(text(390, 366, 'more autonomy: less human time per task, bigger blast radius when a step goes wrong  →', 't-note', 'middle'))
    return svg(760, 380, 'Four levels of autonomy, from suggesting to acting without review. Each level up saves human time and widens the damage a single wrong step can do.', b)


F['ch1_autonomy'] = autonomy()


# ---------------------------------------------------------------- 7. reliability: p^n against n
def reliability():
    W, H = 760, 330
    L, T, B, R = 64, 36, 50, 232
    pw, ph = W - L - R, H - T - B
    ns = list(range(1, 51))
    X = lambda n: L + (n - 1) / 49 * pw
    Y = lambda v: T + ph - v / 100 * ph
    b = []
    for t in [0, 25, 50, 75, 100]:
        b += [line(L, Y(t), L + pw, Y(t), 'grid'), text(L - 8, Y(t) + 4, f'{t}%', 't-tick', 'end')]
    b.append(line(L, Y(90), L + pw, Y(90), 'base-line'))
    b.append(text(L + pw - 6, Y(90) + 14, '90% target', 't-muted', 'end'))
    for n in [1, 5, 10, 20, 30, 40, 50]:
        b.append(text(X(n), T + ph + 20, str(n), 't-tick', 'middle'))
    b.append(line(L, T + ph, L + pw, T + ph, 'axis'))
    ends = []
    for k, p in enumerate(M['ps']):
        vals = [100 * p ** n for n in ns]
        b.append(f'<polyline class="l{k + 1}" points="' + ' '.join(f'{X(n):.1f},{Y(v):.1f}' for n, v in zip(ns, vals)) + '"/>')
        for n in M['ns']:
            v = 100 * p ** n
            b.append(f'<g class="mark"><title>p = {p}, {n} steps: {v:.1f}%</title><circle class="s{k + 1} ring" cx="{X(n):.1f}" cy="{Y(v):.1f}" r="4.5"/></g>')
        ends.append([Y(vals[-1]), f'p = {p}: {vals[-1]:.1f}% at 50 steps', f's{k + 1}', Y(vals[-1])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
    for ly, lab, dc, ty in ends:
        b.append(line(L + pw + 6, ty, L + pw + 16, ly, 'grid'))
        b.append(f'<rect class="{dc}" x="{L + pw + 20}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
        b.append(text(L + pw + 36, ly + 4, lab, 't-tick'))
    b.append(text(L + pw / 2, H - 10, 'number of steps n that must all succeed', 't-tick', 'middle'))
    b.append(text(L - 50, T - 14, 'P(task succeeds) = pⁿ', 't-tick'))
    return svg(W, H, 'Probability that a task of n steps succeeds when each step succeeds with probability p. At p equal to 0.9, ten steps succeed about a third of the time; only p equal to 0.999 stays above 90 percent for a hundred steps.', b)


F['ch1_reliability'] = reliability()


# ---------------------------------------------------------------- 8. the cost comparison
def cost():
    c = M['cost']
    items = [(k, 10000 * v['usd']) for k, v in c.items()]
    b = [text(30, 30, f'Cost of 10,000 tasks at the EXAMPLE price of ${M["price_in"]:.0f} / M input and ${M["price_out"]:.0f} / M output tokens', 't-title')]
    b += hbars(30, 56, items, 400, row_h=34, cls='s1', label_w=160, fmt=lambda v: f'${v:,.0f}', vmax=max(v for _, v in items), hi='agent, 8 steps')
    y = 56 + 3 * 34 + 46
    b.append(text(30, y, 'model calls', 't-label')); b.append(text(30, y + 17, 'input tokens', 't-label')); b.append(text(30, y + 34, 'latency at 2 s per call', 't-label'))
    for i, (k, v) in enumerate(c.items()):
        x = 230 + i * 170
        b += [text(x, y - 20, k, 't-tick'), text(x, y, f'{v["calls"]}', 't-val'), text(x, y + 17, f'{v["in"]:,}', 't-val')]
        lat = v['calls'] * M['latency_per_call_s'] if i < 2 else 8 * M['latency_per_call_s'] + 8 * M['latency_per_call_s']
        b.append(text(x, y + 34, f'{lat:.0f} s' + (' (subagents in parallel)' if i == 2 else ''), 't-val'))
    return svg(760, y + 50, 'The same task as one model call, as an eight-step agent, and as a lead agent with four subagents of eight steps each, at an example price. The agent costs about seven times the single call and the multi-agent system about thirty-six times.', b)


F['ch1_cost'] = cost()


# ---------------------------------------------------------------- 9. Uber's Genie, as described by the blog
def genie():
    b = [text(380, 24, "Genie (Uber, 2024), as the engineering blog describes it", 't-title', 'middle')]
    # ingestion row
    b += block(30, 44, 150, 50, 'internal docs', ['the knowledge sources'], 'box')
    b += block(220, 44, 150, 50, 'chunk + embed', ['offline'], 'box-1')
    b += block(410, 44, 150, 50, 'vector database', ['chunks by embedding'], 'box-1')
    b += [arrow(180, 69, 220, 69), arrow(370, 69, 410, 69)]
    # serving row
    b += block(30, 134, 150, 70, 'Slack question', ['a user in a', 'support channel'], 'box')
    b += block(220, 134, 150, 70, 'Knowledge Service', ['embed the question,', 'fetch relevant chunks'], 'box-1')
    b += block(410, 134, 150, 70, 'LLM', ['via the Michelangelo', 'gateway; prompt = chunks'], 'box-2')
    b += block(580, 134, 160, 70, 'answer in Slack', ['with citations'], 'box')
    b += [arrow(180, 169, 220, 169), arrow(370, 169, 410, 169), arrow(560, 169, 580, 169), arrow(485, 94, 485, 134)]
    b.append(text(300, 125, 'query', 't-muted', 'middle'))
    # feedback row
    b += block(580, 244, 160, 70, 'user feedback', ['Resolved / Helpful /', 'Not Helpful / Not Relevant'], 'box-3')
    b += block(220, 244, 340, 70, 'evaluation', ['LLM as a judge on custom metrics (hallucination, relevancy);', 'channel owners use the results to improve their docs'], 'box-3')
    b += [arrow(660, 204, 660, 244), arrow(580, 279, 560, 279)]
    b.append(hpath([(220, 279), (105, 279), (105, 94)], 'edge', 'ah'))
    b.append(text(60, 269, 'better docs', 't-muted'))
    return svg(760, 330, "Uber's Genie as the blog describes it: documents are chunked and embedded into a vector database; a Slack question is embedded, matched to chunks, and answered by an LLM with citations; user feedback and LLM-as-a-judge metrics feed back into the documents.", b)


F['ch1_genie'] = genie()


# ---------------------------------------------------------------- 10. Anthropic's multi-agent research system
def research():
    b = [text(380, 24, "Anthropic's research system (2025): an orchestrator and parallel subagents", 't-title', 'middle')]
    b += block(30, 50, 130, 60, 'user query', ['a research question'], 'box')
    b += block(250, 50, 260, 70, 'lead agent (Opus 4)', ['plans, splits the question into', 'subtasks, later synthesises'], 'box-2')
    b += block(30, 150, 120, 50, 'memory', ['the saved plan'], 'box-1')
    b += [arrow(160, 80, 250, 80), line(250, 100, 150, 165, 'edge')]
    for i in range(3):
        x = 170 + i * 200
        b += block(x, 180, 170, 60, f'subagent {i + 1} (Sonnet 4)', ['searches, reads, reports'], 'box-3')
        b += block(x + 25, 270, 120, 36, 'web search', cls='box')
        b += [arrow(x + 75, 240, x + 75, 270), arrow(x + 95, 270, x + 95, 240)]
        b.append(f'<line class="edge" x1="{300 + i * 80:.1f}" y1="120" x2="{x + 70:.1f}" y2="180" marker-end="url(#ah)"/>')
        b.append(f'<line class="path" x1="{x + 95:.1f}" y1="180" x2="{330 + i * 80:.1f}" y2="120" marker-end="url(#ah-on)"/>')
    b += para(30, 230, ['grey: subtasks out,', 'in parallel;', 'blue: findings back'], 't-muted')
    b += block(568, 320, 180, 40, 'citation agent → report', cls='box-1')
    b.append(hpath([(510, 100), (748, 100), (748, 320)], 'edge', 'ah'))
    b.append(text(740, 160, 'after synthesis', 't-muted', 'end'))
    return svg(760, 375, 'The orchestrator-worker design of the multi-agent research system: a lead agent plans and delegates, several subagents search in parallel with their own tools, the lead synthesises their findings, and a citation agent attaches sources.', b)


F['ch1_research'] = research()


# ---------------------------------------------------------------- 11. LinkedIn's pipeline, as described by the blog
def linkedin():
    b = [text(380, 24, "LinkedIn's generative AI product (2024): route, retrieve, generate, then measure", 't-title', 'middle')]
    b += block(30, 50, 130, 60, 'member question', ['in the feed or on a job'], 'box')
    b += block(200, 50, 170, 60, 'routing', ['in scope? which agent?'], 'box-2')
    b += [arrow(160, 80, 200, 80)]
    agents = ['job assessment', 'company understanding', 'takeaways for posts']
    for i, a in enumerate(agents):
        y = 140 + i * 50
        b += [box(190, y, 210, 36, 'box-3', 8), text(295, y + 23, a + ' agent', 't-tick', 'middle')]
        b.append(arrow(240, 110, 240, y) if i == 0 else line(240, y - 14, 240, y, 'edge-dim'))
        if i > 0:
            b.append(line(400, y + 18, 420, 175, 'edge-dim'))
    b.append(text(295, 300, 'one such path per agent', 't-muted', 'middle'))
    b += block(420, 140, 150, 70, 'retrieval', ['internal APIs and', 'search, recall first'], 'box-1')
    b += block(600, 140, 140, 70, 'generation', ['precision: write', 'from what was found'], 'box-2')
    b += [arrow(400, 158, 420, 158), arrow(570, 175, 600, 175)]
    b += block(410, 250, 330, 92, 'quality measurement', ['linguists score up to 500 conversations a day:', 'overall quality, hallucination rate,', 'Responsible AI violations, coherence, style'], 'box-3')
    b.append(arrow(670, 210, 670, 250))
    b += para(380, 368, ['the blog: routing and retrieval were tuned on dev sets, like classifiers;', 'generation reached 80% fast, and the last 20% took most of the work'], 't-muted', 'middle')
    return svg(760, 395, "LinkedIn's design as the blog describes it: a router decides whether a question is in scope and which topic agent handles it; the agent retrieves from internal APIs and search, then generates an answer; a linguist team scores hundreds of conversations a day.", b)


F['ch1_linkedin'] = linkedin()


# ---------------------------------------------------------------- 12. the decision flow
def decision():
    b = []
    qs = [('Can plain code do it reliably?', 'yes', 'Write the code. No model needed.', 'box-1'),
          ('Are the steps the same on every run?', 'yes', 'A workflow: a fixed path, the model inside fuzzy steps', 'box-1'),
          ('Can you measure success (an eval set)?', 'no', 'Not yet. Collect examples and build the eval first.', 'box-3'),
          ('Does the step budget fit cost and latency?', 'no', 'Fewer steps, a cheaper model, or a workflow.', 'box-3'),
          ('Would a wrong action be hard to undo?', 'yes', 'An agent that acts only with approval (level 2)', 'box-2')]
    for i, (q, ans, out, cls) in enumerate(qs):
        y = 30 + i * 74
        b += [box(40, y, 320, 44, 'box', 10), text(200, y + 27, q, 't-note', 'middle')]
        b += [arrow(360, y + 22, 420, y + 22), text(390, y + 14, ans, 't-muted', 'middle')]
        b += [box(420, y, 320, 44, cls, 10), text(580, y + 27, out, 't-tick', 'middle')]
        nxt = 'no' if ans == 'yes' else 'yes'
        b += [arrow(200, y + 44, 200, y + 74), text(214, y + 64, nxt, 't-muted')]
    y = 30 + 5 * 74
    b += [box(40, y, 320, 44, 'box-2', 10), text(200, y + 27, 'An agent that acts and reports (level 3)', 't-note', 'middle')]
    b.append(text(580, y + 27, 'raise autonomy only as the evals and traces earn it', 't-muted', 'middle'))
    return svg(760, y + 60, 'A decision flow for whether to build an agent. Most tasks leave at the first two questions; an agent is the answer only when the steps vary, success can be measured, the budget fits, and wrong actions can be undone or approved.', b)


F['ch1_decision'] = decision()


# ---------------------------------------------------------------- 13. the map of the book
def book_map():
    cx, cy = 380, 215
    b = []
    b.append(f'<circle class="box-3" cx="{cx}" cy="{cy}" r="200"/>')
    b.append(f'<circle class="box-1" cx="{cx}" cy="{cy}" r="138"/>')
    b.append(f'<circle class="box-2" cx="{cx}" cy="{cy}" r="72"/>')
    b += para(cx, cy - 12, ['Ch 1 and 2', 'the loop, tools,', 'memory, context'], 't-note', 'middle')
    b += para(cx, cy - 115, ['Ch 3', 'reasoning patterns'], 't-tick', 'middle')
    b += para(cx - 88, cy + 88, ['Ch 4', 'workflow patterns'], 't-tick', 'middle')
    b += para(cx + 88, cy + 88, ['Ch 5', 'multi-agent'], 't-tick', 'middle')
    b += para(cx, cy - 178, ['Ch 6 and 7', 'evals'], 't-tick', 'middle')
    b += para(cx - 168, cy, ['Ch 8', 'tracing'], 't-tick', 'middle')
    b += para(cx + 168, cy, ['Ch 9', 'guardrails'], 't-tick', 'middle')
    b.append(text(cx, cy + 180, 'the outer ring is most of this book', 't-muted', 'middle'))
    b += [box(600, 300, 150, 70, 'box', 10)] + para(675, 324, ['Ch 10: capstone', 'one agent through', 'every layer'], 't-tick', 'middle')
    b += para(14, 300, ['centre: what an agent is made of', 'middle: how its steps are arranged', 'outer: how it is measured, watched', 'and kept safe'], 't-muted')
    b.append(text(14, 30, 'The ten chapters as layers', 't-title'))
    return svg(760, 430, 'The map of the book: the loop, tools and memory at the centre (Chapters 1 and 2), the reasoning, workflow and multi-agent patterns around them (3 to 5), and evals, tracing and guardrails as the outer ring (6 to 9), with the capstone (10) cutting through every layer.', b)


F['ch1_book_map'] = book_map()

json.dump(F, open('results/figs_ch1.json', 'w'))
print(f'{len(F)} figures -> results/figs_ch1.json:', ', '.join(F))
