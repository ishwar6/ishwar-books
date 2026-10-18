"""Figures for Part 1 (the big idea). Writes results/figs_part1.json (key -> svg string). CSS classes only.
Numbers come from results/part1.json (our run) and from the paper's text (Figure 1, Table 1 range, §1)."""
import json, math, os
from figlib import svg, text, box, arrow
from samfig import token, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part1.json')))
F = {}


def harness_around_model():
    """The model in one box; the harness code in the four boxes around it, as a cycle."""
    W, H = 720, 300
    b = [box(10, 10, 700, 280, 'box', 14), text(30, 38, 'The harness: the code around the model', 't-title')]
    bw, bh = 180, 58
    pos = {'1. Store': (40, 62, 'keep past cases'), '2. Retrieve': (270, 62, 'find related ones'),
           '3. Present': (500, 62, 'build the prompt'), '4. The model': (500, 178, 'frozen weights'),
           '5. Read': (270, 178, 'text to an answer')}
    for t, (x, y, n) in pos.items():
        b += [box(x, y, bw, bh, 'box-on' if t == '4. The model' else 'box-1', 8),
              text(x + bw / 2, y + 24, t, 't-tick t-strong', 'middle'), text(x + bw / 2, y + 44, n, 't-tick', 'middle')]
    b.append(arrow(220, 91, 266, 91))
    b.append(arrow(450, 91, 496, 91))
    b.append(arrow(590, 120, 590, 174, on=True))
    b.append(arrow(500, 207, 454, 207, on=True))
    b.append(arrow(270, 207, 132, 207))
    b.append(arrow(130, 207, 130, 124))
    b.append(text(40, 268, 'Only box 4 is the model. Boxes 1, 2, 3 and 5 are code: change them, and the score changes.', 't-tick'))
    return svg(W, H, 'A cycle of five boxes. Store, retrieve and present are code that builds the prompt. The prompt goes '
               'into box 4, the model with frozen weights. Its text goes to the read step, which turns it into an answer, '
               'and the answer can be stored for later cases.', b)


def our_gap():
    h = R['harnesses']
    names = {'A_bare': 'A: bare question', 'B_labels': 'B: + label list', 'C_labels_lenient': 'C: + lenient reader',
             'D_labels_retrieval': 'D: + 5 retrieved cases'}
    items = [(names[k], 100 * v['accuracy']) for k, v in h.items()]
    b = hbars(20, 50, items, 380, row_h=34, cls='s1', label_w=200, fmt=lambda v: f'{v:.1f}%', vmax=100,
              title='Same model, same 60 test cases, four harnesses', hi='D: + 5 retrieved cases')
    b.append(text(20, 50 + 4 * 34 + 24, f"Qwen2.5-1.5B-Instruct on Symptom2Disease (22 diseases). Accuracy on {R['n_eval']} fixed test cases.", 't-note'))
    return svg(720, 50 + 4 * 34 + 40, 'Horizontal bars of accuracy for four harnesses around the same frozen model: '
               + ', '.join(f'{n} {v:.1f} percent' for n, v in items) + '.', b)


def feedback_sizes():
    """Feedback sizes on a log scale: ours (counted) and the paper's (from §1)."""
    fb = R['feedback_tokens']
    rows = [('our run: a score', fb['score'], 's2'), ('our run: a summary', fb['summary'], 's2'),
            ('our run: full traces', fb['traces'], 's2'),
            ('paper: prior optimizers, top of range', 30000, 's3'),
            ('paper: one evaluation, up to', 10_000_000, 's1')]
    x0, lw, bw, rh, y0 = 20, 270, 360, 34, 56
    lo, hi = 0, 7                       # 10^0 .. 10^7 tokens
    b = [text(x0, 26, 'How much feedback the optimizer can read (tokens, log scale)', 't-title')]
    for p in range(lo, hi + 1):
        x = x0 + lw + bw * (p - lo) / (hi - lo)
        b.append(f'<line class="axis" x1="{x:.1f}" y1="{y0 - 6}" x2="{x:.1f}" y2="{y0 + rh * len(rows)}"/>')
        b.append(text(x, y0 + rh * len(rows) + 16, f'10^{p}', 't-tick', 'middle'))
    for i, (lab, v, cls) in enumerate(rows):
        y = y0 + i * rh
        w = max(3, bw * (math.log10(v) - lo) / (hi - lo))
        b.append(text(x0 + lw - 8, y + rh / 2 + 4, lab, 't-tick', 'end'))
        b.append(f'<rect class="{cls}" x="{x0 + lw:.1f}" y="{y + 5:.1f}" width="{w:.1f}" height="{rh - 12:.1f}" rx="3"/>')
        b.append(text(x0 + lw + w + 6, y + rh / 2 + 4, f'{v:,}', 't-val'))
    return svg(720, y0 + rh * len(rows) + 30, 'Bars on a log scale. Our run: a score is ' + f"{fb['score']}" + ' tokens, a summary '
               + f"{fb['summary']}" + ' tokens, the full traces ' + f"{fb['traces']:,}" + ' tokens. The paper: earlier optimizers '
               'see at most about 30,000 tokens per step; one Meta-Harness evaluation can produce up to 10,000,000 tokens.', b)


def the_loop():
    W, H = 720, 210
    b = [text(20, 26, 'The Meta-Harness loop, in one picture', 't-title')]
    nodes = [(30, 'Proposer agent', 'reads any files it wants'), (270, 'New harness', 'one Python file'),
             (510, 'Evaluate', 'run it on search tasks')]
    for x, t, n in nodes:
        b += [box(x, 50, 180, 58, 'box-1', 8), text(x + 90, 74, t, 't-tick t-strong', 'middle'), text(x + 90, 94, n, 't-tick', 'middle')]
    b.append(arrow(210, 79, 266, 79, on=True))
    b.append(arrow(450, 79, 506, 79, on=True))
    b += [box(150, 140, 420, 50, 'box-on', 8), text(360, 162, 'Filesystem: one folder per harness', 't-tick t-strong', 'middle'),
          text(360, 181, 'its code, its scores, its full execution traces', 't-tick', 'middle')]
    b.append(arrow(600, 108, 520, 138, on=True))
    b.append(arrow(200, 138, 120, 110, on=True))
    return svg(W, H, 'A loop of three boxes. The proposer agent reads the filesystem and writes a new harness. The harness is '
               'evaluated on search tasks, and its code, scores and traces go into a new folder of the filesystem. Then the loop repeats.', b)


F['p1_harness'] = harness_around_model()
F['p1_gap'] = our_gap()
F['p1_feedback'] = feedback_sizes()
F['p1_loop'] = the_loop()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
