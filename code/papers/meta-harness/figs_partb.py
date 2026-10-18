"""Figures for the Background page (what a harness is). Reads results/partb.json for our own run; the Table 1 numbers
are the paper's (Meta-Harness, Table 1, MTok per iteration). Writes results/figs_partb.json (key -> SVG string).
Colours come only from CSS classes, so the figures follow the dark, dim and light themes."""
import json, math, os
from figlib import svg, text, box, arrow, esc
from samfig import token, line, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'partb.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


# ---------------------------------------------------------------- 1. timeline of the ideas on this page
def timeline():
    ev = [('2022', 'ReAct', ['think, act,', 'observe, repeat'], 's1'),
          ('2023', 'OPRO, DSPy', ['an LLM rewrites', 'the prompt'], 's2'),
          ('2024', 'TextGrad', ['feedback as a', '"gradient" in words'], 's2'),
          ('2025', 'GEPA, AlphaEvolve', ['reflect on traces;', 'evolve code'], 's2'),
          ('2025', 'ACE', ['a context that', 'grows and is curated'], 's1'),
          ('2026', 'TTT-Discover', ['train the model', 'during the search'], 's2'),
          ('2026', 'Meta-Harness', ['search over the', 'whole harness code'], 's3')]
    xs = [62, 172, 282, 392, 502, 612, 722]
    b = [line(30, 96, 770, 96, 'axis', 'ah')]
    for (yr, name, ls, cls), x in zip(ev, xs):
        b += [f'<circle class="{cls}" cx="{x}" cy="96" r="8"/>', text(x, 40, yr, 't-title', 'middle'),
              text(x, 64, name, 't-note', 'middle')]
        for k, l in enumerate(ls):
            b.append(text(x, 124 + 17 * k, l, 't-muted', 'middle'))
    b += legend(30, 190, [('s1', 'harnesses built by hand'), ('s2', 'optimizers that search'), ('s3', 'the paper')])
    return svg(790, 204, 'Timeline: ReAct 2022, OPRO and DSPy 2023, TextGrad 2024, GEPA, AlphaEvolve and ACE 2025, '
               'TTT-Discover and Meta-Harness 2026.', b)


F['pb_timeline'] = timeline()


# ---------------------------------------------------------------- 2. a model call versus a harness
def harness():
    b = [text(20, 26, 'One model call', 't-title')]
    b += token(20, 60, 110, 'prompt', 'box-1', 36, 't-note')
    b += token(170, 60, 110, 'model', 'box-2', 36, 't-note')
    b += token(320, 60, 110, 'answer', 'box-1', 36, 't-note')
    b += [arrow(132, 78, 168, 78), arrow(282, 78, 318, 78)]
    b.append(text(20, 124, 'You write the prompt. The model writes the answer. Nothing is kept.', 't-muted'))
    y = 160
    b.append(text(20, y + 6, 'A harness: the program around the call', 't-title'))
    b.append(box(14, y + 22, 760, 210, 'box', 14))
    b += token(36, y + 48, 120, 'task input x', 'box-1', 34, 't-note')
    b += token(36, y + 120, 120, 'memory', 'box-1', 34, 't-note')
    b += token(36, y + 176, 120, 'tools', 'box-1', 34, 't-note')
    b += token(206, y + 84, 150, 'retrieve + choose', 'box-on', 34, 't-note')
    b += token(396, y + 84, 130, 'build prompt', 'box-on', 34, 't-note')
    b += token(566, y + 84, 90, 'model', 'box-2', 34, 't-note')
    b += token(680, y + 84, 80, 'output', 'box-1', 34, 't-note')
    b += [arrow(158, y + 65, 204, y + 95), arrow(158, y + 137, 204, y + 106), arrow(358, y + 101, 394, y + 101),
          arrow(528, y + 101, 564, y + 101), arrow(658, y + 101, 678, y + 101)]
    b.append(f'<path class="edge" d="M720,{y + 120} V{y + 205} H158" marker-end="url(#ah)" fill="none"/>')
    b.append(text(440, y + 198, 'update memory, or call a tool and loop again', 't-muted', 'middle'))
    b.append(text(206, y + 58, 'the harness decides what the model sees', 't-tick'))
    return svg(790, 400, 'Top: one model call, a prompt in and an answer out. Bottom: a harness wraps the same model. '
               'It keeps memory, retrieves examples, builds the prompt, calls the model, and updates its state or calls a tool.', b)


F['pb_harness'] = harness()


# ---------------------------------------------------------------- 3. the ReAct loop
def react():
    b = []
    b += token(20, 40, 150, 'question', 'box-1', 36, 't-note')
    b += token(240, 40, 200, 'model: Thought + Action', 'box-2', 36, 't-note')
    b += token(520, 40, 190, 'harness runs the tool', 'box-on', 36, 't-note')
    b += token(240, 150, 200, 'context grows', 'box-1', 36, 't-note')
    b += [arrow(172, 58, 238, 58), arrow(442, 58, 518, 58)]
    b.append(f'<path class="edge" d="M615,78 V168 H442" marker-end="url(#ah)" fill="none"/>')
    b.append(arrow(340, 148, 340, 80))
    b.append(text(625, 124, 'Observation: result', 't-tick'))
    b.append(text(350, 118, 'c(t+1) = c(t) + action + observation', 't-tick'))
    b += token(520, 220, 190, 'Final Answer', 'box-1', 36, 't-note')
    b.append(f'<path class="edge" d="M240,168 H200 V238 H518" marker-end="url(#ah)" fill="none"/>')
    b.append(text(210, 258, 'stop when the model writes "Final Answer:"', 't-muted'))
    return svg(760, 280, 'The ReAct loop. The model writes a thought and an action. The harness runs the tool and adds the '
               'observation to the context. The loop repeats until the model writes a final answer.', b)


F['pb_react'] = react()


# ---------------------------------------------------------------- 4. our three harnesses
def ours():
    H = R['harness']
    b = []
    y0 = 40
    for k, m in enumerate(['Qwen2.5-0.5B-Instruct', 'Qwen2.5-1.5B-Instruct']):
        items = [(h, H[f'{m}/{h}']['acc']) for h in ['zero-shot', 'few-shot', 'retrieval']]
        b += hbars(20, y0 + k * 120, items, 420, row_h=28, cls='s1' if k == 0 else 's2', label_w=110,
                   fmt=lambda v: f'{v:.1f}%', vmax=100, title=m.replace('-Instruct', ''))
        for i, h in enumerate(['zero-shot', 'few-shot', 'retrieval']):
            b.append(text(650, y0 + k * 120 + i * 28 + 18, f'{H[f"{m}/{h}"]["tokens"]:.0f} tokens', 't-muted'))
    b.append(text(650, 22, 'prompt length', 't-tick'))
    return svg(760, 290, 'Accuracy of three harnesses around the same frozen model, on 60 Banking77 questions of six card '
               'intents. Qwen2.5-0.5B goes from 41.7 to 83.3 percent; Qwen2.5-1.5B from 73.3 to 78.3 percent.', b)


F['pb_ours'] = ours()


# ---------------------------------------------------------------- 5. Table 1: how much each optimizer sees per step
def feedback():
    rows = [('OPRO', 0.002), ('GEPA', 0.008), ('Feedback Descent', 0.012), ('TextGrad', 0.015),
            ('AlphaEvolve', 0.022), ('TTT-Discover', 0.026), ('Meta-Harness', 10.0)]
    x0, lw, bw, rh = 20, 150, 460, 30
    lo, hi = math.log10(0.001), math.log10(10)                     # axis from 1,000 to 10,000,000 tokens
    b = [text(x0, 22, 'Context from one evaluation, in tokens (log scale)', 't-title')]
    for i, (name, v) in enumerate(rows):
        y = 40 + i * rh
        w = bw * (math.log10(v) - lo) / (hi - lo)
        cls = 's3' if name == 'Meta-Harness' else 's2'
        b.append(text(x0 + lw - 8, y + rh / 2 + 4, name, 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(name)}: {v} MTok</title><rect class="{cls}" x="{x0 + lw}" y="{y + 4}" '
                 f'width="{w:.1f}" height="{rh - 9}" rx="3"/></g>')
        b.append(text(x0 + lw + w + 6, y + rh / 2 + 4, f'{v * 1e6:,.0f}', 't-val'))
    ya = 40 + len(rows) * rh + 6
    b.append(line(x0 + lw, ya, x0 + lw + bw, ya, 'axis'))
    for e, lab in [(-3, '1K'), (-2, '10K'), (-1, '100K'), (0, '1M'), (1, '10M')]:
        x = x0 + lw + bw * (e - lo) / (hi - lo)
        b += [line(x, ya, x, ya + 5, 'axis'), text(x, ya + 18, lab, 't-tick', 'middle')]
    return svg(760, ya + 30, "Table 1 of the paper as bars on a log scale: the context one evaluation produces for each "
               "optimizer, from 2,000 tokens for OPRO to 26,000 for TTT-Discover, and 10,000,000 for Meta-Harness.", b)


F['pb_feedback'] = feedback()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_partb.json'), 'w'))
print('figures:', ', '.join(F))
