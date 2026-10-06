"""Figures for Chapter 2 (the building blocks: model, tools, instructions, memory and state). Writes results/figs_ch2.json.
Colours come only from CSS classes (box, box-1..4, s1..s4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-note / t-muted, 9 px for t-title; 17 px between baselines.
The drawing helpers (para, block, hpath, circ) are copied from figs_ch1.py so that the two chapters look the same."""
import json
from figlib import svg, text, box, arrow, esc
from agentfig import line
from alammar import frame

M = json.load(open('results/ch2_math.json'))
MEM = json.load(open('results/ch2_memory.json'))
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


def two_lines(W, H, label, xs, series, xlabel, ylabel, ymax, yticks, xticks, notes=(), right=210, fmt=lambda v: f'{v:,.0f}'):
    """Two (or more) series against a shared x axis, with direct end labels, a title row and optional note lines."""
    L, T, B = 70, 36, 50
    pw, ph = W - L - right, H - T - B
    X = lambda v: L + (v - xs[0]) / (xs[-1] - xs[0]) * pw
    Y = lambda v: T + ph - v / ymax * ph
    b = []
    for t in yticks:
        b += [line(L, Y(t), L + pw, Y(t), 'grid'), text(L - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
    for v in xticks:
        b.append(text(X(v), T + ph + 20, str(v), 't-tick', 'middle'))
    b.append(line(L, T + ph, L + pw, T + ph, 'axis'))
    ends = []
    for k, (name, vals) in enumerate(series):
        b.append(f'<polyline class="l{k + 1}" points="' + ' '.join(f'{X(x):.1f},{Y(v):.1f}' for x, v in zip(xs, vals)) + '"/>')
        for x, v in zip(xs, vals):
            b.append(f'<g class="mark"><title>{esc(name)}, step {x}: {fmt(v)}</title><circle class="s{k + 1} ring" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="3.5"/></g>')
        ends.append([Y(vals[-1]), f'{name}: {fmt(vals[-1])}', f's{k + 1}', Y(vals[-1])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
    for ly, lab, dc, ty in ends:
        b.append(line(L + pw + 6, ty, L + pw + 16, ly, 'grid'))
        b.append(f'<rect class="{dc}" x="{L + pw + 20}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
        b.append(text(L + pw + 36, ly + 4, lab, 't-tick'))
    b.append(text(L + pw / 2, H - 10, xlabel, 't-tick', 'middle'))
    b.append(text(L - 56, T - 14, ylabel, 't-tick'))
    for i, n in enumerate(notes):
        b.append(text(L + pw + 20, T + ph - 40 + 17 * i, n, 't-muted'))
    return svg(W, H, label, b)


# ---------------------------------------------------------------- 1. the five blocks and how data flows between them
def blocks():
    b = [text(380, 22, 'The five blocks of an agent and what flows between them', 't-title', 'middle')]
    b += block(30, 50, 180, 64, 'instructions', ['shape: role, rules, format'], 'box-1')
    b += block(30, 250, 180, 64, 'memory', ['remember: facts, notes, store'], 'box-1')
    b += block(290, 150, 180, 76, 'model', ['reason: read the context,', 'pick the next action'], 'box-2')
    b += block(550, 150, 180, 76, 'tools', ['act: run the call, return', 'an observation'], 'box-3')
    b += block(290, 36, 180, 70, 'state (the loop)', ['control: step, plan, pending', 'call, budget, errors'], 'box-4')
    # context assembled from instructions + memory + history, then given to the model
    b += [arrow(210, 82, 290, 170), text(238, 118, 'every turn', 't-muted')]
    b += [arrow(210, 282, 290, 206), text(214, 262, 'retrieve', 't-muted')]
    b.append(f'<line class="edge-dim" x1="296" y1="222" x2="212" y2="298" marker-end="url(#ah)"/>')
    b.append(text(222, 326, 'write notes, facts', 't-muted'))
    # model <-> tools
    b += [arrow(470, 172, 550, 172), text(510, 164, 'tool call', 't-muted', 'middle')]
    b += [arrow(550, 204, 470, 204), text(510, 220, 'observation', 't-muted', 'middle')]
    # state <-> model: the loop decides whether to call again, counts the budget
    b += [arrow(366, 106, 366, 150), arrow(394, 150, 394, 106)]
    b.append(text(404, 132, 'next turn?  budget left?', 't-muted'))
    # tools -> state (errors, retries) and environment
    b.append(hpath([(640, 150), (640, 72), (470, 72)], 'edge', 'ah'))
    b.append(text(560, 64, 'errors, retries', 't-muted', 'middle'))
    b += [box(560, 250, 170, 64, 'box-ghost', 10), text(645, 274, 'environment', 't-note', 'middle'),
          text(645, 294, 'calendar, mail, CRM, files', 't-tick', 'middle'), arrow(640, 226, 640, 250), arrow(660, 250, 660, 226)]
    b += para(380, 345, ['what each block is for: the model decides, the tools do, the instructions constrain,',
                         'the memory carries knowledge forward, the state says where in the task the loop is'], 't-muted', 'middle')
    return svg(760, 372, 'The five building blocks of an agent: instructions and retrieved memory are assembled into the context the model reads each turn; the model emits a tool call; the tool acts on the environment and returns an observation; the loop keeps the state (step, plan, pending call, budget, errors).', b)


F['ch2_blocks'] = blocks()


# ---------------------------------------------------------------- 2. model-choice factors as a dot table
def model_factors():
    factors = [('capability on the task', [3, 2, 1]), ('tool-call reliability', [3, 2, 1]), ('speed (low latency)', [1, 2, 3]),
               ('low cost per task', [1, 2, 3]), ('context window', [3, 2, 2]), ('structured-output support', [3, 3, 2]),
               ('control: weights, data, hosting', [1, 1, 3])]
    cols = ['large API model', 'mid-size model', 'small open-weight model']
    b = [text(380, 24, 'The usual shape of the trade-off (a sketch, not a measurement: the table in Section 2.2 says what to measure)', 't-title', 'middle')]
    x0, lw, cw, rh, y0 = 30, 230, 160, 32, 64
    for j, c in enumerate(cols):
        b.append(text(x0 + lw + j * cw + cw / 2, y0 - 14, c, 't-note', 'middle'))
    for i, (f, vals) in enumerate(factors):
        y = y0 + i * rh
        b.append(text(x0 + lw - 10, y + rh / 2 + 4, f, 't-tick', 'end'))
        for j, v in enumerate(vals):
            cx = x0 + lw + j * cw + cw / 2
            b.append(f'<g class="mark"><title>{esc(f)}, {esc(cols[j])}: {["low", "medium", "high"][v - 1]}</title>'
                     f'<circle class="s{j + 1}" cx="{cx - 28:.1f}" cy="{y + rh / 2:.1f}" r="9" style="fill-opacity:{[0.25, 0.55, 0.95][v - 1]}"/></g>')
            b.append(text(cx - 12, y + rh / 2 + 4, ['low', 'medium', 'high'][v - 1], 't-tick'))
    y = y0 + len(factors) * rh + 14
    b += para(30, y, ['Read the rows as questions for your eval set, not as a ranking: an agent of ten routine steps and one hard step',
                      'may want the third column for the routine steps and the first for the hard one (the router, below).'], 't-muted')
    return svg(760, y + 44, 'The factors that decide which model should sit inside an agent, sketched for three classes of model. Capability and tool-call reliability tend to rise with size; speed, cost and control fall. The sketch is qualitative; the numbers must come from a per-step evaluation on your own tasks.', b)


F['ch2_model_factors'] = model_factors()


# ---------------------------------------------------------------- 3. the router: a small model for routine steps, a big one for hard steps
def router():
    b = [text(380, 22, 'A model router inside the loop', 't-title', 'middle')]
    b += block(30, 120, 130, 60, 'next step', ['from the loop'], 'box')
    b += block(210, 110, 170, 80, 'router', ['a rule, a classifier', 'or the small model itself'], 'box-4')
    b += [arrow(160, 150, 210, 150)]
    b += block(450, 50, 180, 64, 'small model', ['routine: format, extract,', 'pick an obvious tool'], 'box-2')
    b += block(450, 190, 180, 64, 'large model', ['hard: plan, recover,', 'ambiguous tool choice'], 'box-2')
    b += [arrow(380, 135, 450, 82), text(410, 92, 'routine', 't-muted', 'middle'),
          arrow(380, 165, 450, 222), text(410, 212, 'hard', 't-muted', 'middle')]
    b += block(670, 120, 70, 60, 'tool', cls='box-3')
    b += [arrow(630, 82, 670, 140), arrow(630, 222, 670, 160)]
    # escalation path
    b.append(hpath([(540, 114), (540, 150), (560, 150), (560, 190)], 'path', 'ah-on'))
    b.append(text(534, 146, 'escalate on a failed', 't-muted', 'end'))
    b.append(text(534, 163, 'check or low confidence', 't-muted', 'end'))
    b += para(30, 285, ['pros: most steps are cheap and fast; the large model is paid for only where it changes the outcome',
                        'cons: two models to evaluate and version; a wrong route is a silent quality loss; the router can be the weak step',
                        'to decide: label a per-step eval set routine or hard, run both models on it, route to the small one',
                        'wherever it matches the large one'], 't-muted')
    return svg(760, 362, 'A router sends each step of the loop to a small model (routine steps: formatting, extraction, an obvious tool choice) or a large one (planning, recovery, ambiguous choices), with an escalation path when the small model fails a check. A per-step evaluation set decides where the line is.', b)


F['ch2_router'] = router()


# ---------------------------------------------------------------- 4. the function-calling round trip
def roundtrip():
    b = [text(190, 22, 'your code (the loop)', 't-title', 'middle'), text(570, 22, 'the model (API)', 't-title', 'middle'),
         line(380, 34, 380, 392, 'edge-dim')]
    steps = [
        (1, 'left', 'box-1', 'request: system prompt + tool definitions + user turn',
         ['tools = [{name, description, input_schema}]', 'user: "Find 45 min for Priya and Tom..."']),
        (2, 'right', 'box-2', 'reply: a structured call, stop_reason = tool_use',
         ['{type: tool_use, id: call_01,', ' name: calendar_find_free_slots,', ' input: {attendee_emails: [...], ...}}']),
        (3, 'left', 'box-3', 'your code validates the arguments and runs the function',
         ['result = find_free_slots(**input)', 'the model never executes anything']),
        (4, 'left', 'box-1', 'request: the result goes back as a tool_result turn',
         ['{type: tool_result, tool_use_id: call_01,', ' content: "{slots: [...], time_zone: ...}"}']),
        (5, 'right', 'box-2', 'reply: text (or another tool call), stop_reason = end_turn',
         ['"Three slots fit (none before 10:00) ..."']),
    ]
    y = 44
    for n, side, cls, title, lines in steps:
        h = 36 + 17 * len(lines)
        x = 30 if side == 'left' else 410
        b += [box(x, y, 320, h, cls, 10), circ(x + 18, y + 18, 11), text(x + 18, y + 22.5, str(n), 't-tick', 'middle'),
              text(x + 36, y + 22, title, 't-tick')]
        b += para(x + 36, y + 42, lines, 't-tick')
        if n in (1, 4):
            b.append(arrow(350, y + h / 2, 410, y + h / 2))
        if n in (2, 5):
            b.append(arrow(410, y + h / 2, 350, y + h / 2))
        y += h + 12
    b.append(text(380, y + 8, 'steps 1 to 4 repeat while the model keeps asking for tools; the whole exchange is the context the next call sees', 't-muted', 'middle'))
    return svg(760, y + 24, 'One function-calling round trip: the request carries the tool definitions and the user turn; the model replies with a structured tool call; your code runs the function and sends the result back as a tool_result turn; the model replies with text or another call. The model never executes anything itself.', b)


F['ch2_roundtrip'] = roundtrip()


# ---------------------------------------------------------------- 5. a bad tool and a good tool, side by side
def tool_good_bad():
    b = [text(200, 22, 'a tool the model will misuse', 't-title', 'middle'), text(560, 22, 'a tool the model can use', 't-title', 'middle'),
         line(380, 10, 380, 400, 'edge-dim')]
    bad = [('name', 'get_data'), ('description', 'Gets data.'), ('arguments', 'q: string  (what goes in q?)'),
           ('returns', 'the whole table, 40,000 tokens'), ('on error', '"Error 500"'), ('on empty', 'raises an exception'),
           ('side effects', 'unknown; sometimes writes'), ('idempotent', 'no: a retry creates duplicates')]
    good = [('name', 'calendar_find_free_slots'), ('description', 'Find slots when ALL attendees are free.'), ('', 'Use before booking; [] if none are free.'),
            ('arguments', 'attendee_emails[], duration_minutes,'), ('', 'window_start, window_end, earliest_hour'),
            ('returns', 'at most 10 slots + the time zone'), ('on error', '"window_end is before window_start:'), ('', 'swap them or widen the window"'),
            ('side effects', 'none (read-only, annotated)'), ('idempotent', 'yes: safe to retry')]
    for x0, items, cls in [(24, bad, 'box-ghost'), (404, good, 'box-3')]:
        h = 30 + 19 * len(items)
        b.append(box(x0, 40, 336, h, cls, 12))
        for i, (k, v) in enumerate(items):
            y = 64 + 19 * i
            if k:
                b.append(text(x0 + 14, y, k, 't-muted'))
            b.append(text(x0 + 104, y, v, 't-tick'))
    b += para(200, 255, ['the model has to guess what q is, reads 40k tokens', 'to find one row, and cannot tell a failure from', 'an empty answer; a retry books twice'], 't-muted', 'middle')
    b += para(560, 300, ['name says the service and the verb; arguments are', 'named so they cannot be confused; the result fits in', 'the context; the error says what to do next'], 't-muted', 'middle')
    return svg(760, 360, 'A tool the model will misuse and a tool it can use. The good tool has a namespaced name, a description that says when to use it and what it returns, unambiguous argument names, a bounded return size, an error message that says what to do, no side effects and safe retries.', b)


F['ch2_tool_good_bad'] = tool_good_bad()


# ---------------------------------------------------------------- 6. MCP: host, clients, servers
def mcp():
    b = [text(380, 22, 'The Model Context Protocol: one host, one client per server, three primitives per server', 't-title', 'middle')]
    b.append(box(30, 44, 330, 250, 'box-ghost', 12))
    b.append(text(195, 66, 'MCP host (the AI application: an IDE, a desktop app, your agent)', 't-tick', 'middle'))
    b += block(60, 84, 130, 60, 'model', ['the agent loop'], 'box-2')
    b += block(60, 180, 130, 50, 'MCP client A', cls='box-4')
    b += block(220, 180, 130, 50, 'MCP client B', cls='box-4')
    b += [arrow(110, 144, 110, 180), arrow(150, 144, 270, 180)]
    b.append(text(195, 262, 'the host lists tools from every client and', 't-muted', 'middle'))
    b.append(text(195, 279, 'puts their definitions into the model\'s context', 't-muted', 'middle'))
    # servers
    b += block(420, 70, 310, 90, 'MCP server: filesystem (local, stdio)', ['tools: read_file, search_files', 'resources: file contents', 'prompts: "summarise this folder"'], 'box-3')
    b += block(420, 190, 310, 90, 'MCP server: issue tracker (remote, HTTP)', ['tools: issues_search, issue_create', 'resources: an issue, a project', 'prompts: "triage this issue"'], 'box-3')
    b += [arrow(190, 196, 420, 118), arrow(350, 205, 420, 228)]
    b.append(text(575, 300, 'both links speak JSON-RPC: tools/list to discover, tools/call to run', 't-muted', 'middle'))
    b += para(380, 318, ['host: coordinates the clients and owns the model  ·  client: one connection to one server',
                         'server: provides tools (actions), resources (data) and prompts (templates)'], 't-muted', 'middle')
    return svg(760, 352, 'The participants of the Model Context Protocol as its documentation describes them: an MCP host (the AI application) creates one MCP client per MCP server; each server exposes tools (actions), resources (data) and prompts (templates) over JSON-RPC, locally over stdio or remotely over HTTP.', b)


F['ch2_mcp'] = mcp()


# ---------------------------------------------------------------- 7. the anatomy of a system prompt
def instructions():
    b = [text(380, 22, 'The anatomy of an agent\'s instructions, and the smell when a section is missing', 't-title', 'middle')]
    secs = [('role and goal', 'box-1', '"You schedule meetings for the sales team. Your job ends when the invite is sent."', 'missing: the agent does adjacent tasks nobody asked for'),
            ('hard constraints', 'box-4', '"Never book before 10:00. Never email outside the company. Ask before deleting."', 'missing: the guardrail lives only in people\'s heads'),
            ('tool guidance', 'box-3', '"Use find_free_slots before create_event. Prefer one search with filters."', 'missing: the right tool, called in the wrong order'),
            ('output format', 'box-2', '"Reply with a one-line summary and the event id. No markdown."', 'missing: a parser that breaks on every third run'),
            ('stop conditions', 'box-4', '"Stop when the invite is sent, after 8 steps, or if a tool fails twice."', 'missing: loops that never end, budgets that run out'),
            ('canonical examples', 'box-1', 'two or three short, diverse transcripts of a good run', 'missing: edge cases listed as rules instead of shown')]
    y = 44
    for name, cls, ex, smell in secs:
        b += [box(30, y, 150, 44, cls, 8), text(105, y + 27, name, 't-tick', 'middle')]
        b.append(text(192, y + 19, ex, 't-tick'))
        b.append(text(192, y + 36, smell, 't-muted'))
        y += 54
    b.append(text(380, y + 6, 'order matters less than clarity; keep every section short enough that a new colleague would read it', 't-muted', 'middle'))
    return svg(760, y + 22, 'The six sections of an agent\'s system prompt (role and goal, hard constraints, tool guidance, output format, stop conditions, canonical examples), each with an example line and the production smell that appears when it is missing.', b)


F['ch2_instructions'] = instructions()


# ---------------------------------------------------------------- 8. the memory ladder and the three kinds of memory
def memory_ladder():
    b = [text(200, 22, 'where memory lives', 't-title', 'middle'), text(575, 22, 'what kind of thing is remembered', 't-title', 'middle'),
         line(390, 10, 390, 330, 'edge-dim')]
    tiers = [('short-term: the context window', ['this turn; the model sees it directly;', 'costs tokens on every call; gone at the end'], 'box-2'),
             ('working: scratchpad and plan', ['this run; written by the agent as notes;', 'survives compaction; small by design'], 'box-4'),
             ('long-term: a store outside', ['across runs; written and retrieved by key,', 'search or embedding; cheap to hold, costly to get wrong'], 'box-1')]
    for i, (t, lines, cls) in enumerate(tiers):
        y = 50 + i * 90
        b += block(30, y, 340, 76, t, lines, cls)
        if i < 2:
            b += [arrow(120, y + 76, 120, y + 90), arrow(280, y + 90, 280, y + 76)]
    b += [text(132, 221, 'write', 't-muted'), text(292, 221, 'retrieve', 't-muted'), text(132, 131, 'summarise', 't-muted'), text(292, 131, 'pull in', 't-muted')]
    kinds = [('episodic', 'what happened', ['"on Monday the invite bounced, Tom was out";', 'past runs, traces, reflections (Reflexion)'], 'box-1'),
             ('semantic', 'facts about the world and the user', ['"Priya is in Dublin"; "the user never meets', 'before 10:00"; a profile, a knowledge base'], 'box-1'),
             ('procedural', 'how to do things', ['"always check the room before booking"; the', 'instructions, learned rules, tool recipes'], 'box-1')]
    for i, (k, sub, lines, cls) in enumerate(kinds):
        y = 50 + i * 90
        b += [box(410, y, 320, 76, cls, 10), text(424, y + 22, k, 't-note'), text(500, y + 22, sub, 't-muted')]
        b += para(424, y + 42, lines, 't-tick')
    return svg(760, 335, 'Left: the three places memory lives, from the context window (seen directly, paid for every turn) through the agent\'s own scratchpad to a store outside the context. Right: the three kinds of memory, episodic (what happened), semantic (facts) and procedural (how to), each with an example.', b)


F['ch2_memory_ladder'] = memory_ladder()


# ---------------------------------------------------------------- 9. Generative Agents' retrieval score
def retrieval_score():
    mems = [('Isabella is planning a party on the 14th', 0.55, 0.85, 0.95), ('met Klaus at Hobbs Cafe, talked about research', 0.70, 0.50, 0.60),
            ('the refrigerator is empty', 0.95, 0.20, 0.10), ('ate breakfast at 8 in the room', 0.90, 0.10, 0.15)]
    b = [text(380, 22, 'Generative Agents: score = recency + importance + relevance, then keep the top ones that fit', 't-title', 'middle')]
    b.append(text(30, 50, 'query: "what should Klaus do this afternoon?"   (illustrative values, chosen to show the mechanism)', 't-muted'))
    cols = ['recency', 'importance', 'relevance', 'score']
    x0, lw, cw, y0, rh = 30, 300, 100, 86, 34
    for j, c in enumerate(cols):
        b.append(text(x0 + lw + j * cw + cw / 2, y0 - 12, c, 't-note', 'middle'))
    rows = sorted(mems, key=lambda m: -(m[1] + m[2] + m[3]))
    for i, (t, r, im, rel) in enumerate(rows):
        y = y0 + i * rh
        sc = r + im + rel
        b.append(text(x0 + lw - 10, y + rh / 2 + 4, t, 't-tick', 'end'))
        for j, v in enumerate([r, im, rel]):
            cx = x0 + lw + j * cw
            b.append(f'<g class="mark"><title>{esc(t)}: {cols[j]} {v:.2f}</title><rect class="s{j + 1}" x="{cx + 10:.1f}" y="{y + 8:.1f}" width="{(cw - 30) * v:.1f}" height="{rh - 16:.1f}" rx="3" style="fill-opacity:0.85"/></g>')
            b.append(text(cx + 10 + (cw - 30) * v + 4, y + rh / 2 + 4, f'{v:.2f}', 't-val'))
        b.append(text(x0 + lw + 3 * cw + cw / 2, y + rh / 2 + 4, f'{sc:.2f}', 't-strong' if i < 2 else 't-tick', 'middle'))
        if i < 2:
            b.append(box(x0 + lw + 3 * cw + 8, y + 5, cw - 16, rh - 10, 'box-on', 6))
    b += para(30, y0 + 4 * rh + 16, ['recency decays with the time since the memory was last used (0.995 per game hour in the paper);',
                                     'importance is a score the model gave the memory when it was stored; relevance is the embedding',
                                     'similarity to the query. All three weights are 1 in the paper. The two highlighted memories go into the prompt.'], 't-muted')
    return svg(760, y0 + 4 * rh + 76, 'The retrieval function of Generative Agents: each memory gets a recency score, an importance score (assigned when stored) and a relevance score (similarity to the current situation); the paper sums them with equal weights and puts the top-ranked memories that fit into the prompt. Values here are illustrative.', b)


F['ch2_retrieval_score'] = retrieval_score()


# ---------------------------------------------------------------- 10. MemGPT's paging analogy
def memgpt():
    b = [text(380, 22, 'MemGPT: the context window as main memory, external stores as disk, the model as its own operating system', 't-title', 'middle')]
    b.append(box(30, 44, 330, 230, 'box-2', 12))
    b.append(text(195, 66, 'main context (fixed size: the prompt tokens)', 't-note', 'middle'))
    b += [box(50, 80, 290, 40, 'box', 8), text(195, 104, 'system instructions (how to use the memory functions)', 't-tick', 'middle')]
    b += [box(50, 130, 290, 50, 'box-4', 8), text(195, 150, 'working context: facts the model chose to keep', 't-tick', 'middle'), text(195, 168, '"birthday is February 7", "boyfriend named James"', 't-muted', 'middle')]
    b += [box(50, 190, 290, 50, 'box', 8), text(195, 210, 'FIFO queue: the most recent messages', 't-tick', 'middle'), text(195, 228, 'oldest ones are evicted (and summarised) first', 't-muted', 'middle')]
    b.append(text(195, 262, 'alert when the queue nears the limit: "memory pressure"', 't-muted', 'middle'))
    b.append(box(420, 44, 310, 230, 'box-1', 12))
    b.append(text(575, 66, 'external context (unbounded: a database)', 't-note', 'middle'))
    b += [box(440, 80, 270, 70, 'box', 8), text(575, 102, 'recall storage', 't-tick', 'middle'), text(575, 120, 'the full message history, searchable', 't-muted', 'middle'), text(575, 137, 'recall_storage.search("six flags")', 't-muted', 'middle')]
    b += [box(440, 164, 270, 70, 'box', 8), text(575, 186, 'archival storage', 't-tick', 'middle'), text(575, 204, 'documents and notes the model filed away', 't-muted', 'middle'), text(575, 221, 'archival_memory.insert(...)', 't-muted', 'middle')]
    b.append(text(575, 262, 'only what is paged in is ever seen by the model', 't-muted', 'middle'))
    b += [arrow(360, 150, 420, 110), text(378, 168, 'page out', 't-muted', 'start'), arrow(420, 200, 360, 215), text(372, 232, 'page in', 't-muted', 'start')]
    b.append(text(380, 300, 'the paging is done by function calls the model itself makes when it sees the alert; the loop only runs them', 't-muted', 'middle'))
    return svg(760, 316, 'The operating-system analogy of MemGPT: the fixed-size context window is main memory (system instructions, a working context of kept facts, a queue of recent messages); recall and archival storage are disk; the model pages information in and out by calling memory functions when it receives a memory-pressure alert.', b)


F['ch2_memgpt'] = memgpt()


# ---------------------------------------------------------------- 11. the state object over four steps of a run
def state():
    b = [text(380, 22, 'The state object of one run, checkpointed after every step', 't-title', 'middle')]
    frames = [('after step 1', ['step: 1 of 8', 'plan: slots → book → invite', 'pending: find_free_slots', 'budget: 7 steps, 38k tok', 'errors: 0', 'status: running']),
              ('after step 2', ['step: 2 of 8', 'plan: slots ✓ → book', 'pending: none', 'last: 3 slots found', 'budget: 6 steps, 31k tok', 'status: running']),
              ('after step 3', ['step: 3 of 8', 'plan: book ✓ → invite', 'pending: send_email', 'event: EVT-1187 (Mon)', 'budget: 5 steps, 24k tok', 'status: running']),
              ('after step 4', ['step: 4 of 8', 'plan: invite ✗ → move', 'pending: none', 'errors: 1 (Tom out Mon)', 'budget: 4 steps, 17k tok', 'status: needs_retry'])]
    for i, (t, lines) in enumerate(frames):
        x = 30 + i * 182
        b += frame(x, 40, 170, 160, i + 1, t, 'box' if i < 3 else 'box-4')
        b += para(x + 14, 92, lines, 't-tick')
        b += [box(x + 30, 214, 110, 26, 'box-1', 6), text(x + 85, 231, 'checkpoint', 't-tick', 'middle'), arrow(x + 85, 200, x + 85, 214)]
        if i < 3:
            b.append(arrow(x + 170, 120, x + 182, 120))
    b += para(380, 268, ['memory says what the agent knows; state says where it is. A crash after step 3 resumes from checkpoint 3',
                         'with the pending call still recorded, so the retry is deliberate; the email tool must be idempotent',
                         'or the second attempt invites everyone twice.'], 't-muted', 'middle')
    return svg(760, 318, 'The state object of a meeting-scheduling run over four steps: the current step, the plan with its progress marks, the pending tool call, the budget left, the errors so far and the status. Each step ends with a checkpoint, so a crashed run resumes from the last one instead of starting again.', b)


F['ch2_state'] = state()


# ---------------------------------------------------------------- 12. the context window as a budget bar
def context_budget():
    budget, total = M['budget'], M['budget_total']
    b = [text(380, 22, f'What one turn sends to the model at step 10 of a run: {total:,} tokens (planning numbers from ch2_math.py)', 't-title', 'middle')]
    x, W, y, h = 30, 700, 50, 44
    classes = ['box-1', 'box-3', 'box-4', 'box-2', 'box']
    labels = ['instructions', 'tool definitions', 'memory', 'history', 'observation']
    cx = x
    for (k, v), cls, lab in zip(budget.items(), classes, labels):
        w = W * v / total
        b.append(f'<g class="mark"><title>{esc(k)}: {v:,} tokens</title>{box(cx, y, w, h, cls, 4)}</g>')
        if w > 140:
            b.append(text(cx + w / 2, y + h / 2 + 4, f'{lab} {v:,}', 't-tick', 'middle'))
        cx += w
    # callouts for the thin segments
    cx, k_ = x, 0
    for (k, v), lab in zip(budget.items(), labels):
        w = W * v / total
        if w <= 140:
            yy = y + h + (20 if k_ % 2 == 0 else 40)
            b += [line(cx + w / 2, y + h, cx + w / 2, yy, 'edge-dim'), text(cx + w / 2, yy + 14, f'{lab} {v:,}', 't-tick', 'middle')]
            k_ += 1
        cx += w
    b += para(30, 170, ['the history is already 63% of the turn and grows by about 1,200 tokens per step; by step 30 the same',
                        'run sends 40,300 tokens per turn without compaction (Section 2.7). The window\'s hard limit is far above',
                        'this; the practical limit is attention (Section 2.7).'], 't-muted')
    return svg(760, 232, 'The context of one turn as a budget bar at step 10 of a run: instructions, tool definitions, retrieved memory, the growing history and the current observation. The history already dominates and grows every step; the fixed parts are paid for on every call.', b)


F['ch2_context_budget'] = context_budget()


# ---------------------------------------------------------------- 13. the 30-step run with and without compaction
def compaction_cost():
    xs = list(range(1, M['steps'] + 1))
    notes = [f'input tokens over the run:', f'  {M["in_plain"]:,} without', f'  {M["in_compact"]:,} with',
             f'cost at the example price:', f'  ${M["cost_plain"]:.2f} vs ${M["cost_compact"]:.2f}']
    return two_lines(760, 340, 'Input tokens sent on each step of a thirty-step run, without compaction (growing linearly to 40,300) and with compaction every ten steps (sawtooth, never above 19,300).',
                     xs, [('no compaction', M['per_step_plain']), (f'compaction every {M["compact_every"]} steps', M['per_step_compact'])],
                     'step of the run', 'input tokens on this step', 45000, [0, 10000, 20000, 30000, 40000], [1, 5, 10, 15, 20, 25, 30], notes, right=230)


F['ch2_compaction_cost'] = compaction_cost()


# ---------------------------------------------------------------- 14. the toy memory from ch2_memory.py: context size per step
def memory_sizes():
    rows = {r['label']: r for r in MEM['rows']}
    xs = list(range(1, 13))
    notes = ['toy run, 4 chars per token:', f'  {rows["without compaction"]["total"]:,} tokens total without',
             f'  {rows["with compaction"]["total"]:,} with (every 4 steps)']
    return two_lines(760, 300, 'Approximate context size per step of the twelve-step toy run in ch2_memory.py, with and without compaction every four steps.',
                     xs, [('without compaction', rows['without compaction']['sizes']), ('with compaction', rows['with compaction']['sizes'])],
                     'step of the toy run', 'approx. tokens in context', 550, [0, 100, 200, 300, 400, 500], xs, notes, right=230, fmt=lambda v: f'{v:,.0f}')


F['ch2_memory_sizes'] = memory_sizes()


# ---------------------------------------------------------------- 15. the meeting example, block by block
def meeting_trace():
    cols = [('instructions', 'box-1'), ('memory', 'box-1'), ('model', 'box-2'), ('tools', 'box-3'), ('state', 'box-4')]
    rows = [('1', ['role: scheduler;', 'rule: not before 10'], ['recall: user prefers', 'mornings'], ['plan: slots, room,', 'book, invite'], ['(none yet)'], ['step 1/8, plan set,', 'budget 8']),
            ('2', ['tool guidance: find', 'slots before booking'], ['note: candidates', 'Mon/Wed/Thu/Fri 10'], ['picks find_free_slots', 'earliest_hour=10'], ['find_free_slots ->', '14 slots'], ['pending call,', 'then cleared']),
            ('3', ['output: event id', 'in the summary'], ['note: room 4B'], ['picks find_rooms,', 'then create_event'], ['create_event ->', 'EVT-1187 Mon 10:00'], ['step 3/8, event id', 'stored']),
            ('4', ['constraint: only', 'company addresses'], ['store: Tom out', 'on Monday'], ['reads the bounce;', 'decides to move'], ['send_email ->', 'ok + auto-reply'], ['errors: 1, status', 'needs_retry']),
            ('5', ['stop: when the', 'invite is sent'], ['recall: the 10:00', 'rule (compacted!)'], ['picks update_event', 'Wed 10:00'], ['update_event ->', 'moved, invites sent'], ['step 5/8, done;', 'checkpoint final'])]
    b = [text(380, 22, 'The meeting-scheduling run, block by block', 't-title', 'middle')]
    x0, cw, y0, rh = 92, 130, 44, 58
    for j, (c, cls) in enumerate(cols):
        b += [box(x0 + j * cw + 3, y0, cw - 6, 28, cls, 6), text(x0 + j * cw + cw / 2, y0 + 19, c, 't-tick', 'middle')]
    for i, (s, *cells) in enumerate(rows):
        y = y0 + 36 + i * rh
        b += [circ(50, y + 20, 11), text(50, y + 24.5, s, 't-tick', 'middle'), text(50, y + 44, 'step', 't-muted', 'middle')]
        b.append(line(x0, y + rh - 6, x0 + 5 * cw, y + rh - 6, 'grid'))
        for j, lines in enumerate(cells):
            b += para(x0 + j * cw + cw / 2, y + 18, lines, 't-tick', 'middle')
    y = y0 + 36 + len(rows) * rh + 6
    b.append(text(380, y, 'every step touches all five blocks; the trace in Chapter 8 records exactly these columns', 't-muted', 'middle'))
    return svg(760, y + 16, 'The meeting-scheduling example traced block by block over five steps: what the instructions contributed, what memory recalled or stored, what the model decided, which tool ran and what the state recorded. The compacted 10:00 rule is recovered from long-term memory at step 5.', b)


F['ch2_meeting_trace'] = meeting_trace()

json.dump(F, open('results/figs_ch2.json', 'w'))
print(f'{len(F)} figures -> results/figs_ch2.json:', ', '.join(F))
