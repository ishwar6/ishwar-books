"""Figures for Chapter 5, first half (multi-agent systems: when, architectures, coordination). Writes results/figs_ch5a.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-note / t-muted, 9 px for t-title; 17 px between baselines.
At the end, check() estimates every text's box and reports overlaps and clipping. The result charts read results/ch5a_*.json."""
import json, os, re, textwrap
from figlib import svg, text, box, arrow, esc
from agentfig import line, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, 'results', n)))
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def notes(x, y, sentences, cls='t-muted', right=740):
    out = textwrap.wrap(' '.join(sentences), int((right - x) / 7.2))
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


def twoway(x1, y1, x2, y2):
    return f'<line class="edge" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" marker-start="url(#ah)" marker-end="url(#ah)"/>'


def finish(title, label, b, y, sentences):
    p, n = notes(20, y, sentences)
    return svg(760, y + n * 17 + 6, label, [text(380, 22, title, 't-title', 'middle')] + b + p)


# ---------------------------------------------------------------- 1. why several agents can help, and what they cost
def why():
    b = []
    gains = [('context isolation', ['each agent reads only its part;', 'the lead keeps a short context']),
             ('parallel breadth', ['independent parts run at once;', 'time = the slowest part']),
             ('specialisation', ['own prompt, own tools, own', 'model per role'])]
    costs = [('tokens', ['every agent pays its own prompt,', 'brief, reading and report']),
             ('latency', ['planning, waiting for the', 'slowest, then merging']),
             ('coordination', ['what the brief leaves out is lost;', 'parallel choices can conflict'])]
    b += [text(200, 54, 'what you can gain', 't-note', 'middle'), text(560, 54, 'what you pay', 't-note', 'middle')]
    for i, (t, l) in enumerate(gains):
        b += block(40, 68 + i * 74, 320, 62, t, l, 'box-3')
    for i, (t, l) in enumerate(costs):
        b += block(400, 68 + i * 74, 320, 62, t, l, 'box-1')
    return finish('Several agents: three things they can buy, three things they always cost',
                  'Two columns: what a multi-agent system can gain (context isolation, parallel breadth, specialisation) and what it costs (tokens, latency, coordination).',
                  b, 310, ['A multi-agent design is worth it only when the left column is large for your task: the work is broad, splits into parts that do not '
                           'depend on each other, and is too much for one context. The right column is paid on every task.'])


F['a_why'] = why()


# ---------------------------------------------------------------- 2. the two systems of the experiment
def setup():
    b = [text(190, 52, 'one agent', 't-note', 'middle'), text(570, 52, 'lead agent with subagents', 't-note', 'middle')]
    b += block(60, 66, 260, 54, 'single agent', ['one context, all five tools'], 'box-2')
    tools = ['read_requirements', 'list_suppliers', 'read_supplier', 'search_news', 'shipping_lookup']
    for i, t in enumerate(tools):
        b += [box(80, 140 + i * 30, 220, 24, 'box-ghost', 6), text(190, 156 + i * 30, t, 't-tick', 'middle')]
        b.append(line(190, 120, 190, 140, 'edge') if i == 0 else '')
    b += [text(190, 312, '24 pages, news and ports all land', 't-muted', 'middle'), text(190, 329, 'in the same growing context', 't-muted', 'middle')]
    b += block(450, 66, 240, 54, 'lead agent', ['rules, supplier list, run_subagents'], 'box-4')
    for i in range(4):
        x = 420 + i * 78
        b.append(hpath([(570, 120), (570, 140), (x + 34, 140), (x + 34, 162)], 'edge'))
        b += [box(x, 162, 68, 70, 'box-2', 8), text(x + 34, 182, f'sub {i + 1}', 't-note', 'middle'),
              text(x + 34, 200, 'brief only', 't-tick', 'middle'), text(x + 34, 216, '6 pages', 't-tick', 'middle')]
        b.append(hpath([(x + 34, 232), (x + 34, 262)], 'edge'))
    b += [box(420, 262, 302, 30, 'box-ghost', 6), text(571, 282, 'read_supplier, search_news, shipping_lookup', 't-tick', 'middle')]
    b += [text(571, 312, 'each subagent has its own context; the lead', 't-muted', 'middle'),
          text(571, 329, 'sees only the briefs it wrote and the reports', 't-muted', 'middle')]
    return finish('The experiment: the same question, tools and model, arranged two ways',
                  'Left: one agent with five tools and one context. Right: a lead agent that can read the rules and the supplier list and starts subagents; each subagent has three research tools and sees only its brief.',
                  b, 356, ['Same model (gpt-5-mini, reasoning effort low), same task text, same local files. The only differences are who reads what, '
                           'and what passes between the agents. The lead cannot read pages itself, so everything it knows about a supplier comes back in a report.'])


F['a_setup'] = setup()


# ---------------------------------------------------------------- 3. cost arithmetic
def cost():
    C = R('ch5a_cost.json')
    rows = C['rows']
    b = [text(30, 50, 'input tokens per task', 't-note'), text(470, 50, 'time per task (s)', 't-note')]
    toks = [(r['design'], r['input']) for r in rows]
    secs = [('', r['seconds']) for r in rows]
    b += hbars(30, 62, toks, 150, row_h=26, cls='s1', label_w=250, fmt=lambda v: f'{v:,.0f}', vmax=max(v for _, v in toks) * 1.3)
    b += hbars(470, 62, secs, 210, row_h=26, cls='s2', label_w=10, fmt=lambda v: f'{v:.0f}', vmax=max(v for _, v in secs) * 1.25)
    y = 62 + len(rows) * 26 + 22
    return finish('The arithmetic: re-reading a growing context against splitting it',
                  'Input tokens and time per task for one agent with one, six or twenty-four tool calls per turn, and for a lead agent with two to eight subagents, from ch5a_cost.py.',
                  b, y, [f'From ch5a_cost.py: 24 pages of about {C["page_tokens"]:,} tokens (measured from our trace), a {C["P"]:,}-token prompt, '
                         f'{C["t_call"]:.0f} s per model call. One agent that reads one page per turn re-reads everything it has read at every step, '
                         'so its input grows with the square of the pages. Splitting the reading over subagents or batching tool calls in one agent both cut that term.'])


# ---------------------------------------------------------------- 4. results of the single-versus-multi experiment
def exp1():
    S = R('ch5a_single_vs_multi.json')['summary']
    names = {'single': 'one agent', 'seq': 'one agent, 1 call/turn', 'multi': 'lead + subagents'}
    arms = [a for a in ('single', 'seq', 'multi') if a in S]
    b = []
    panels = [('pages read (of 24)', 'pages', 24, lambda v: f'{v:.1f}', 's3'), ('gold picks (of 3)', 'gold', 3, lambda v: f'{v:.1f}', 's3'),
              ('total tokens', None, None, lambda v: f'{v / 1000:,.0f}k', 's1'), ('wall-clock time (s)', 'seconds', None, lambda v: f'{v:.0f}', 's2')]
    for k, (title, key, vmax, fmt, cls) in enumerate(panels):
        x0, y0 = 30 + (k % 2) * 370, 50 + (k // 2) * 130
        b.append(text(x0, y0, title, 't-note'))
        items = []
        for a in arms:
            v = S[a]['input']['mean'] + S[a]['output']['mean'] if key is None else S[a][key]['mean']
            items.append((names[a], v))
        vm = vmax or max(v for _, v in items) * 1.25
        b += hbars(x0, y0 + 12, items, 150, row_h=26, cls=cls, label_w=170, fmt=fmt, vmax=vm)
    n = len(R('ch5a_single_vs_multi.json')['runs']) // len(arms)
    return finish('One agent against a lead with subagents, on Northwind\'s supplier question',
                  'Four panels comparing one agent, one agent limited to one tool call per turn, and a lead agent with subagents: supplier pages read, gold picks found, total tokens and wall-clock time, means over runs.',
                  b, 50 + 2 * 130 + 10, [f'Means over {n} runs per arm of ch5a_single_vs_multi.py (gpt-5-mini, reasoning effort low). '
                                         'Gold picks are the three suppliers that the buying rules select, computed in code from the fake catalogue.'])


# ---------------------------------------------------------------- 5 to 10. the six architectures
def arch_single():
    b = block(40, 60, 120, 44, 'user', [], 'box')
    b.append(arrow(160, 82, 260, 82))
    b += block(260, 50, 200, 64, 'one agent', ['one loop, one context'], 'box-2')
    for i, t in enumerate(['search', 'read', 'compute', 'write']):
        x = 222 + i * 72
        b += [box(x, 150, 60, 28, 'box-ghost', 6), text(x + 30, 168, t, 't-tick', 'middle'), line(360, 114, x + 30, 150, 'edge')]
    b.append(arrow(460, 82, 560, 82))
    b += block(560, 60, 150, 44, 'answer', [], 'box-on')
    return finish('Baseline: one agent with tools',
                  'One agent: a single loop with one context and several tools, from the user to the answer.',
                  b, 210, ['Every decision and every tool result stays in one context, so nothing is lost between agents and nothing can conflict. '
                           'It breaks when the work outgrows the context or must be done faster than one loop can do it in sequence.'])


def arch_orch():
    b = block(30, 92, 90, 40, 'task', [], 'box')
    b.append(arrow(120, 112, 160, 112))
    b += block(160, 80, 150, 64, 'lead agent', ['plans, briefs, merges'], 'box-4')
    for i in range(3):
        y = 46 + i * 52
        b.append(hpath([(310, 112), (340, 112), (340, y + 20), (380, y + 20)], 'edge'))
        b += [box(380, y, 170, 40, 'box-2', 8), text(465, y + 25, f'subagent {i + 1}: own context', 't-tick', 'middle')]
        b.append(hpath([(550, y + 20), (580, y + 20), (580, 112), (600, 112)], 'edge'))
    b += block(600, 92, 130, 40, 'answer', [], 'box-on')
    b.append(hpath([(235, 144), (235, 214), (465, 214), (465, 198)], 'path', 'ah-on'))
    b.append(text(350, 232, 'brief down, structured result up; the lead may run another round', 't-muted', 'middle'))
    return finish('Orchestrator (supervisor) with subagents',
                  'A lead agent plans, writes a brief for each subagent, runs them in parallel with their own contexts, and merges their results.',
                  b, 262, ['Anthropic\'s research system has this shape. The lead owns the answer; subagents never talk to each other. '
                           'Chapter 4 showed the same shape as a workflow with single-call workers; here each subagent runs a full tool loop.'])


def arch_hier():
    b = block(310, 44, 140, 40, 'top lead', [], 'box-4')
    for i, nm in enumerate(['team lead: price', 'team lead: risk']):
        x = 120 + i * 360
        b.append(hpath([(380, 84), (380, 100), (x + 80, 100), (x + 80, 116)], 'edge'))
        b += [box(x, 116, 160, 36, 'box-3', 8), text(x + 80, 139, nm, 't-tick', 'middle')]
        for j in range(3):
            xx = x - 40 + j * 82
            b.append(hpath([(x + 80, 152), (x + 80, 166), (xx + 36, 166), (xx + 36, 180)], 'edge'))
            b += [box(xx, 180, 72, 32, 'box-2', 6), text(xx + 36, 200, f'worker', 't-tick', 'middle')]
    return finish('Hierarchical teams: leads of leads',
                  'A tree of agents: a top lead delegates to team leads, and each team lead runs its own workers.',
                  b, 240, ['Each level summarises for the level above, so the top lead\'s context stays small even with many workers. '
                           'Each level is also one more handoff where detail can be lost, and one more set of calls to pay for.'])


def arch_handoff():
    b = block(30, 80, 90, 40, 'user', [], 'box')
    b.append(arrow(120, 100, 160, 100))
    b += block(160, 74, 130, 52, 'triage agent', [], 'box-4')
    for i, (nm, y) in enumerate([('orders agent', 46), ('refunds agent', 112)]):
        b.append(hpath([(290, 100), (330, 100), (330, y + 22), (420, y + 22)], 'path' if i == 1 else 'edge', 'ah-on' if i == 1 else 'ah'))
        b += [box(420, y, 150, 44, 'box-2', 8), text(495, y + 27, nm, 't-note', 'middle')]
    b.append(text(495, 176, 'via the tool transfer_to_refunds_agent()', 't-tick', 'middle'))
    b.append(hpath([(570, 134), (620, 134)], 'path', 'ah-on'))
    b += block(620, 114, 110, 40, 'reply', [], 'box-on')
    return finish('Peer-to-peer handoffs: the conversation moves to another agent',
                  'A triage agent hands the whole conversation to a specialist agent through a handoff tool; the specialist then answers the user directly.',
                  b, 196, ['In OpenAI\'s Swarm and its successor, the Agents SDK, a handoff is a tool call. Control moves with it: the new agent '
                           'becomes the active one and, by default, sees the whole conversation so far. No agent sits above the others.'])


def arch_board():
    b = [box(250, 92, 260, 80, 'box-4', 10), text(380, 116, 'shared state (blackboard)', 't-note', 'middle'),
         text(380, 138, 'facts, plan, partial results,', 't-tick', 'middle'), text(380, 155, 'who has claimed which task', 't-tick', 'middle')]
    pos = [(60, 50, 'price agent'), (60, 170, 'risk agent'), (580, 50, 'delivery agent'), (580, 170, 'writer agent')]
    for x, y, nm in pos:
        b += [box(x, y, 130, 40, 'box-2', 8), text(x + 65, y + 25, nm, 't-tick', 'middle')]
        x1 = x + 130 if x < 300 else x
        x2 = 250 if x < 300 else 510
        b.append(twoway(x1, y + 20, x2, 132 + (y - 110) / 4))
    b.append(text(380, 238, 'agents read what they need and write what they find; no agent calls another', 't-muted', 'middle'))
    return finish('Shared state: a blackboard that every agent reads and writes',
                  'Four agents around a shared store; each reads from it and writes to it, and none sends messages to another directly.',
                  b, 266, ['The idea is old: speech-understanding systems of the 1970s used a blackboard. MetaGPT\'s shared message pool is a modern form, '
                           'with a subscription filter so that each role reads only what concerns it. The store needs rules for who may write what, or agents overwrite each other.'])


def arch_roles():
    roles = [('CEO', 'CTO', '{task} to {ideas}'), ('CTO', 'programmer', '{ideas} to {code}'), ('reviewer', 'programmer', '{code} to {code}'), ('tester', 'programmer', '{code} to {code}')]
    b = []
    for i, (a, c, io) in enumerate(roles):
        x = 30 + i * 182
        b += [box(x, 54, 150, 34, 'box-4', 8), text(x + 75, 76, a, 't-tick', 'middle'), twoway(x + 75, 88, x + 75, 116),
              box(x, 116, 150, 34, 'box-2', 8), text(x + 75, 138, c, 't-tick', 'middle'), text(x + 75, 172, io, 't-muted', 'middle')]
        if i < 3:
            b.append(arrow(x + 150, 102, x + 182, 102))
    b.append(text(380, 202, 'instructor (top) and assistant (bottom) chat in turns; only the result goes on', 't-muted', 'middle'))
    return finish('Role-play teams: agents that act out a company',
                  'Four pairs of role-playing agents in sequence, as in ChatDev: CEO and CTO, CTO and programmer, reviewer and programmer, tester and programmer; each pair chats and passes its result on.',
                  b, 232, ['CAMEL paired an AI user with an AI assistant; ChatDev chains such pairs through design, coding and testing; MetaGPT gives each role '
                           'a standard operating procedure and a required document format. The roles are prompts on the same model, not different skills.'])


F['a_arch_single'] = arch_single()
F['a_arch_orch'] = arch_orch()
F['a_arch_hier'] = arch_hier()
F['a_arch_handoff'] = arch_handoff()
F['a_arch_board'] = arch_board()
F['a_arch_roles'] = arch_roles()


# ---------------------------------------------------------------- 11. what passes between agents
def passes():
    b = []
    rows = [('full history', 'everything the lead saw and said', ['nothing is lost', 'large, and the receiver may', 'take the sender\'s role'], 'box-1'),
            ('task brief', 'objective, rules, format, tools, limits', ['small and focused', 'loses whatever the writer', 'forgot to put in it'], 'box-3'),
            ('structured result', 'fields the receiver asked for', ['easy to check and merge', 'loses the reasoning', 'behind each value'], 'box-2')]
    for i, (nm, what, notes_, cls) in enumerate(rows):
        y = 50 + i * 76
        b += [box(30, y, 100, 60, 'box-4', 8), text(80, y + 35, 'sender', 't-note', 'middle')]
        b.append(arrow(130, y + 30, 158, y + 30))
        b += [box(158, y + 6, 270, 48, cls, 8), text(293, y + 26, nm, 't-note', 'middle'), text(293, y + 44, what, 't-tick', 'middle')]
        b.append(arrow(428, y + 30, 456, y + 30))
        b += [box(456, y, 80, 60, 'box-2', 8), text(496, y + 35, 'receiver', 't-tick', 'middle')]
        b += para(548, y + 18, notes_, 't-tick')
    return finish('Three things one agent can hand another',
                  'Three handoff payloads between a sender and a receiver: the full history, a task brief, and a structured result, with what each keeps and loses.',
                  b, 50 + 3 * 76 + 6, ['Downward (lead to subagent) the choice is history or brief; upward (subagent to lead) it is usually a structured result. '
                                       'Section 5.3\'s experiment measures the first choice.'])


F['a_passes'] = passes()


# ---------------------------------------------------------------- 12. anatomy of a brief
def brief():
    parts = [('objective', 'what to find out and why it matters to the whole task'),
             ('rules and facts', 'every rule, number and definition the work depends on, copied in full'),
             ('output format', 'the exact fields to return, so the lead can check and merge them in code'),
             ('tools and sources', 'which tools to use for which part, and which sources to trust'),
             ('limits', 'scope (these suppliers only), a tool budget, what not to do')]
    b = []
    for i, (nm, d) in enumerate(parts):
        y = 48 + i * 42
        b += [box(30, y, 160, 34, 'box-4' if i else 'box-3', 8), text(110, y + 22, nm, 't-note', 'middle'), text(206, y + 22, d, 't-tick')]
    return finish('A subagent brief, part by part',
                  'Five parts of a good subagent brief: objective, rules and facts, output format, tools and sources, and limits.',
                  b, 48 + 5 * 42 + 10, ['Anthropic lists objective, output format, tools and sources, and task boundaries. We add the rules and facts the '
                                        'lead learned before delegating, because a subagent cannot apply a rule it was never given.'])


F['a_brief'] = brief()


# ---------------------------------------------------------------- 13. results of the handoff experiment
def exp2():
    S = R('ch5a_handoff.json')['summary']
    names = {'thin': 'one-line brief', 'brief': 'full brief', 'history': 'full history'}
    conds = ['thin', 'brief', 'history']
    b = []
    panels = [('facts right (of 12)', lambda s: s['facts'], 12, lambda v: f'{v:.1f}', 's3'),
              ('input tokens', lambda s: s['input'], None, lambda v: f'{v / 1000:,.1f}k', 's1'),
              ('report in the lead\'s format (runs)', lambda s: s['json_format'], 3, lambda v: f'{v:.0f}', 's2'),
              ('answered the lead\'s question itself (runs)', lambda s: s['took_over'], 3, lambda v: f'{v:.0f}', 's4')]
    for k, (title, fn, vmax, fmt, cls) in enumerate(panels):
        x0, y0 = 30 + (k % 2) * 370, 50 + (k // 2) * 116
        b.append(text(x0, y0, title, 't-note'))
        items = [(names[c], fn(S[c])) for c in conds]
        vm = vmax or max(v for _, v in items) * 1.25
        b += hbars(x0, y0 + 12, items, 160, row_h=26, cls=cls, label_w=120, fmt=fmt, vmax=vm)
    n = len(R('ch5a_handoff.json')['runs']) // 3
    return finish('The same subagent, handed the same six suppliers three ways',
                  'Four panels for a one-line brief, a full brief and the full history: facts right out of 12, input tokens, reports in the lead\'s format, and runs where the subagent answered the lead\'s question itself.',
                  b, 50 + 2 * 116 + 10, [f'Means over {n} runs per condition of ch5a_handoff.py (gpt-5-mini, reasoning effort low). The 12 facts are a verdict for '
                                         'each of the six suppliers and the landed cost and delivery days of the three that should pass.'])


# ---------------------------------------------------------------- 14. MCP and A2A
def protocols():
    b = [box(260, 70, 240, 70, 'box-4', 10), text(380, 98, 'your agent', 't-note', 'middle'), text(380, 118, '(MCP host and client, A2A client)', 't-tick', 'middle')]
    for i, nm in enumerate(['tool server: news', 'tool server: shipping', 'data: catalogue']):
        y = 50 + i * 44
        b += [box(30, y, 170, 34, 'box-2', 8), text(115, y + 22, nm, 't-tick', 'middle'), twoway(200, y + 17, 260, 105)]
    b.append(text(115, 200, 'MCP: tools, resources, prompts', 't-note', 'middle'))
    for i, nm in enumerate(['remote agent: freight broker', 'remote agent: compliance']):
        y = 62 + i * 60
        b += [box(560, y, 180, 40, 'box-1', 8), text(650, y + 25, nm, 't-tick', 'middle'), twoway(500, 105, 560, y + 20)]
    b.append(text(650, 200, 'A2A: tasks, messages, artifacts', 't-note', 'middle'))
    b.append(text(380, 170, 'JSON-RPC 2.0 underneath both', 't-muted', 'middle'))
    return finish('Two protocols for two different edges: MCP to tools, A2A to other agents',
                  'An agent in the middle connects through MCP to tool and data servers on the left, and through A2A to remote agents on the right.',
                  b, 232, ['MCP standardises how an application gives a model tools and context. A2A standardises how one agent asks another, '
                           'possibly opaque, agent to do a task and gets results back. An agent behind an A2A endpoint may itself use MCP for its tools.'])


F['a_protocols'] = protocols()


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


if __name__ == '__main__':
    for key, fn, src in [('a_cost', cost, 'ch5a_cost.json'), ('a_exp1', exp1, 'ch5a_single_vs_multi.json'), ('a_exp2', exp2, 'ch5a_handoff.json')]:
        try:
            F[key] = fn()
        except (FileNotFoundError, KeyError) as e:
            print(f'skipped {key}: results/{src} missing or old ({e})')
    json.dump(F, open(os.path.join(HERE, 'results', 'figs_ch5a.json'), 'w'))
    print(f'{len(F)} figures -> results/figs_ch5a.json:', ', '.join(F))
    check(F)
