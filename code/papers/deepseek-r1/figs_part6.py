"""SVG figures for Part 6 (distillation). Paper numbers come from Tables 5 and 6 of arXiv v1; our numbers come from
results/part6_loss.json, results/part6_regrade.json and results/part6_tiny.json (run r1_part6_run.py first).
Writes results/figs_part6.json. CSS classes only, so every theme works."""
import json, os
from figlib import svg, text, box, esc, bar_path
from samfig import line, token

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
F = {}


def load(name):
    p = os.path.join(RES, name + '.json')
    return json.load(open(p)) if os.path.exists(p) else None


# ---------------------------------------------------------------- Table 5 and Table 6 of arXiv v1 (page 14 and 15)
T5 = [  # model, AIME pass@1, AIME cons@64, MATH-500, GPQA, LiveCodeBench, Codeforces rating
    ('GPT-4o-0513', 9.3, 13.4, 74.6, 49.9, 32.9, 759),
    ('Sonnet 3.5 (Anthropic)', 16.0, 26.7, 78.3, 65.0, 38.9, 717),
    ('OpenAI-o1-mini', 63.6, 80.0, 90.0, 60.0, 53.8, 1820),
    ('QwQ-32B-Preview', 50.0, 60.0, 90.6, 54.5, 41.9, 1316),
    ('R1-Distill-Qwen-1.5B', 28.9, 52.7, 83.9, 33.8, 16.9, 954),
    ('R1-Distill-Qwen-7B', 55.5, 83.3, 92.8, 49.1, 37.6, 1189),
    ('R1-Distill-Qwen-14B', 69.7, 80.0, 93.9, 59.1, 53.1, 1481),
    ('R1-Distill-Qwen-32B', 72.6, 83.3, 94.3, 62.1, 57.2, 1691),
    ('R1-Distill-Llama-8B', 50.4, 80.0, 89.1, 49.0, 39.6, 1205),
    ('R1-Distill-Llama-70B', 70.0, 86.7, 94.5, 65.2, 57.5, 1633),
]
T6 = [
    ('QwQ-32B-Preview', 50.0, 60.0, 90.6, 54.5, 41.9),
    ('R1-Zero-Qwen-32B (RL)', 47.0, 60.0, 91.6, 55.0, 40.2),
    ('R1-Distill-Qwen-32B (SFT)', 72.6, 83.3, 94.3, 62.1, 57.2),
]


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b.append(f'<rect class="{cls}" x="{x}" y="{y - 10}" width="12" height="12" rx="2"/>')
        b.append(text(x + 18, y, lab, 't-note'))
        x += 30 + 7.2 * len(lab)
    return b


def hbar(x, y, w, h, cls, title):
    return f'<g class="mark"><title>{esc(title)}</title><path class="{cls}" d="{bar_path(x, y, max(w, 2), h, horizontal=True)}"/></g>'


# ---------------------------------------------------------------- 1. the "you are here" pipeline
def pipeline():
    W, H = 720, 440
    b = [text(20, 24, 'Where distillation sits: R1 writes the data, six small models copy it', 't-title')]
    # the R1 pipeline (Part 5), compressed into one column
    stages = [('DeepSeek-V3-Base', 'box'), ('cold start SFT', 'box-1'), ('reasoning RL', 'box-1'),
              ('rejection sampling', 'box-1'), ('SFT + RL for all', 'box-1'), ('DeepSeek-R1', 'box-on')]
    y = 46
    for i, (s, c) in enumerate(stages):
        b += token(20, y, 170, s, c, h=34, tcls='t-note')
        if i < len(stages) - 1:
            b.append(line(105, y + 34, 105, y + 52, 'edge', 'ah'))
        y += 54
    b.append(text(105, y + 14, 'Part 5 (§2.3)', 't-muted', 'middle'))
    # the data
    b += token(250, 156, 170, '800k samples', 'box-on', h=34, tcls='t-note')
    b.append(text(335, 210, '600k reasoning traces', 't-muted', 'middle'))
    b.append(text(335, 226, '(rejection sampled)', 't-muted', 'middle'))
    b.append(text(335, 246, '200k other: writing,', 't-muted', 'middle'))
    b.append(text(335, 262, 'factual QA, translation', 't-muted', 'middle'))
    b.append(line(190, 173, 246, 173, 'edge-on', 'ah-on'))
    b.append(text(218, 165, '§2.3.3', 't-tick', 'middle'))
    # SFT only
    b += token(460, 156, 90, 'SFT only', 'box-1', h=34, tcls='t-note')
    b.append(line(420, 173, 456, 173, 'edge-on', 'ah-on'))
    b.append(text(505, 210, 'no RL stage', 't-muted', 'middle'))
    b.append(text(505, 226, '(§2.4)', 't-muted', 'middle'))
    # six students
    studs = [('Qwen2.5-Math-1.5B', 'R1-Distill-Qwen-1.5B'), ('Qwen2.5-Math-7B', 'R1-Distill-Qwen-7B'),
             ('Qwen2.5-14B', 'R1-Distill-Qwen-14B'), ('Qwen2.5-32B', 'R1-Distill-Qwen-32B'),
             ('Llama-3.1-8B', 'R1-Distill-Llama-8B'), ('Llama-3.3-70B-Instruct', 'R1-Distill-Llama-70B')]
    y0 = 52
    for i, (base, out) in enumerate(studs):
        yy = y0 + i * 52
        b.append(f'<g class="mark"><title>{esc(base)} fine-tuned on the 800k samples gives {esc(out)}</title>')
        b += token(575, yy, 130, base.replace('-Instruct', '-Inst.'), 'box', h=22, tcls='t-tick')
        b.append(text(640, yy + 36, out, 't-tick', 'middle'))
        b.append('</g>')
        b.append(line(550, 173, 573, yy + 11, 'edge', 'ah'))
    b.append(text(575, 382, 'student base model, and the', 't-muted'))
    b.append(text(575, 398, 'released model it becomes', 't-muted'))
    return svg(W, H, 'Pipeline diagram. DeepSeek-V3-Base goes through the four R1 stages to become DeepSeek-R1. '
               'The 800k samples of the rejection sampling stage then train six smaller open models with SFT only: '
               'four Qwen2.5 models and two Llama models.', b)


F['p6_pipeline'] = pipeline()


# ---------------------------------------------------------------- 2. word-level vs sequence-level, step by step
def kd_kinds():
    top = None                      # parsed from the saved log of r1_part6_tiny.py (step 0 finished; see the page)
    import ast, re
    lp = os.path.join(RES, 'part6_tiny_stdout.txt')
    if os.path.exists(lp):
        lines = open(lp).read().split('\n')
        for i, l in enumerate(lines):
            m = re.match(r'== 0\. teacher distribution at response position \d+ \(after (.*)\):$', l)
            if m:
                ctx = ast.literal_eval(m.group(1))
                pairs = re.findall(r"'([^']*)' ([0-9.]+)", lines[i + 1])
                pk = re.search(r"the trace has '([^']*)' \(p = ([0-9.]+)\)", lines[i + 1])
                top = dict(context=ctx, top=[(t, float(p)) for t, p in pairs[:5]],
                           picked=pk.group(1), picked_p=float(pk.group(2)))
    W, H = 720, 300
    b = [text(20, 24, 'Two ways to copy a teacher', 't-title')]
    # left: word-level (Hinton): the full distribution at each position
    b.append(text(20, 52, 'Word-level (Hinton et al., 2015)', 't-note'))
    b.append(text(20, 70, 'target = the teacher\'s whole next-token distribution', 't-muted'))
    if top:
        ctx = top['context'].replace('\n', ' ').strip()[-26:]
        b.append(text(20, 94, f'after "...{ctx}" (blank line)', 't-tick'))
        for i, (tokn, p) in enumerate(top['top'][:5]):
            y = 108 + i * 26
            b.append(text(110, y + 15, repr(tokn), 't-mono', 'end'))
            b.append(hbar(118, y + 3, 200 * p, 16, 's1', f'{tokn!r}: p = {p:.3f}'))
            b.append(text(124 + 200 * p, y + 16, f'{p:.3f}', 't-val'))
        b.append(text(20, 250, 'the student matches every bar,', 't-muted'))
        b.append(text(20, 266, 'at every position of the text', 't-muted'))
    # right: sequence-level (Kim and Rush; R1)
    x = 380
    b.append(text(x, 52, 'Sequence-level (Kim and Rush, 2016; R1)', 't-note'))
    b.append(text(x, 70, 'target = one text the teacher wrote', 't-muted'))
    if top:
        b.append(text(x, 94, 'the same position in R1-style training data:', 't-tick'))
        tok0, p0 = top['picked'], top['picked_p']
        b.append(text(x + 90, 123, repr(tok0), 't-mono', 'end'))
        b.append(hbar(x + 98, 111, 200, 16, 's2', f'{tok0!r}: target probability 1'))
        b.append(text(x + 304, 124, '1', 't-val'))
        b.append(text(x, 160, f'The teacher gave {tok0!r} p = {p0:.3f}; the sampled', 't-muted'))
        b.append(text(x, 176, 'text keeps only that token, as a hard label.', 't-muted'))
    b.append(text(x, 214, 'needs: the teacher\'s text only', 't-note'))
    b.append(text(x, 232, '(no logits, any tokenizer, any API)', 't-muted'))
    b.append(text(20, 292, 'Left: what a soft-label student sees. Right: what an R1 student sees. Numbers from our run.', 't-muted'))
    return svg(W, H, 'Word-level distillation trains the student on the teacher\'s whole probability distribution '
               'for the next token. Sequence-level distillation, used by R1, trains it on one text the teacher wrote, '
               'so each position has a single target token.', b)


# ---------------------------------------------------------------- 3. the mask, on our real tokens
def mask_fig():
    L = load('part6_loss')
    if not L:
        return None
    from transformers import AutoTokenizer          # the tokenizer only, no model
    from r1_part6 import build_example, DISTILL
    tok = AutoTokenizer.from_pretrained(DISTILL)
    ids, labels, P = build_example(tok, L['trace']['question'], L['trace']['text'])
    toks = [dict(tok=tok.decode([ids[t]]), label=labels[t]) for t in range(P - 4, P + 5)]
    W, H = 720, 170
    b = [text(20, 24, 'One SFT example: the prompt is masked, the response is trained on', 't-title')]
    x = 20
    for t in toks:
        s = t['tok'].replace('\n', '\\n')
        w = max(46, 8.2 * len(s) + 16)
        if x + w > W - 10:
            break
        cls = 'box' if t['label'] == -100 else 'box-on'
        b += token(x, 50, w, s, cls, h=30, tcls='t-mono')
        b.append(text(x + w / 2, 100, '-100' if t['label'] == -100 else str(t['label']), 't-tick', 'middle'))
        x += w + 6
    b.append(text(20, 122, 'label', 't-muted'))
    b.append(text(20, 148, 'Plain boxes: prompt tokens, label -100, no loss.  Outlined boxes: response tokens, label = the token id, '
                  'trained on.', 't-muted'))
    return svg(W, H, 'The tokens around the boundary between prompt and response in our real example, with their labels. '
               'Prompt tokens get the label minus 100 and add nothing to the loss; response tokens keep their own id.', b)


# ---------------------------------------------------------------- 4. Table 5 as grouped bars: AIME pass@1 and MATH-500
def table5():
    W = 720
    rowh, top, L = 40, 64, 190
    H = top + rowh * len(T5) + 60
    pw = W - L - 70
    b = [text(20, 24, 'Table 5: the six distilled models against four other models (pass@1, %)', 't-title')]
    b += legend(L, 48, [('s1', 'AIME 2024'), ('s2', 'MATH-500')])
    for t in (0, 25, 50, 75, 100):
        x = L + pw * t / 100
        b.append(f'<line class="grid" x1="{x:.1f}" y1="{top - 4}" x2="{x:.1f}" y2="{top + rowh * len(T5)}"/>')
        b.append(text(x, top + rowh * len(T5) + 16, str(t), 't-tick', 'middle'))
    for i, r in enumerate(T5):
        y = top + i * rowh
        if i == 4:
            b.append(f'<line class="axis" x1="20" y1="{y - 4}" x2="{W - 20}" y2="{y - 4}"/>')
        dist = r[0].startswith('R1-')
        b.append(text(L - 10, y + 20, r[0], 't-tick t-strong' if dist else 't-tick', 'end'))
        for j, (v, cls) in enumerate([(r[1], 's1'), (r[3], 's2')]):
            yy = y + 4 + j * 16
            b.append(hbar(L, yy, pw * v / 100, 13, cls, f'{r[0]}, {"AIME 2024" if j == 0 else "MATH-500"}: {v}'))
            b.append(text(L + pw * v / 100 + 5, yy + 11, f'{v}', 't-val'))
    b.append(text(20, H - 18, 'Above the line: models that were not distilled from R1. Below: the six students. '
                  'Source: arXiv v1, Table 5.', 't-muted'))
    return svg(W, H, 'Grouped bar chart of Table 5. For each of ten models, a bar for AIME 2024 pass@1 and a bar for '
               'MATH-500 pass@1. Every distilled model except the 1.5B beats o1-mini on MATH-500; the 1.5B still beats '
               'GPT-4o and Sonnet 3.5 on both maths tests.', b)


F['p6_table5'] = table5()


# ---------------------------------------------------------------- 5. the 1.5B against GPT-4o and Sonnet 3.5: maths vs the rest
def small_vs_big():
    rows = [T5[0], T5[1], T5[4]]
    cols = [('AIME 2024', 1), ('MATH-500', 3), ('GPQA Diamond', 4), ('LiveCodeBench', 5)]
    cls = ['s3', 's4', 's1']
    W, H = 720, 320
    b = [text(20, 24, 'The 1.5B student wins on maths and loses on science and code', 't-title')]
    b += legend(20, 50, [(c, r[0]) for c, r in zip(cls, rows)])
    pw, L, top, gh = 140, 30, 80, 170
    for k, (name, idx) in enumerate(cols):
        x0 = L + k * 172
        b.append(text(x0 + pw / 2, top + gh + 22, name, 't-tick', 'middle'))
        b.append(f'<line class="axis" x1="{x0}" y1="{top + gh}" x2="{x0 + pw}" y2="{top + gh}"/>')
        for j, r in enumerate(rows):
            v = r[idx]
            h = gh * v / 100
            x = x0 + 10 + j * 42
            b.append(f'<g class="mark"><title>{esc(r[0])}, {name}: {v}</title>'
                     f'<path class="{cls[j]}" d="{bar_path(x, top + gh - h, 36, h)}"/></g>')
            b.append(text(x + 18, top + gh - h - 6, f'{v}', 't-val', 'middle'))
        if k == 1:
            b.append(f'<line class="grid" x1="{x0 + pw + 16}" y1="{top - 10}" x2="{x0 + pw + 16}" y2="{top + gh + 30}"/>')
    b.append(text(L, H - 12, 'Left of the line: maths. Right: science questions and code. Pass@1 in %, arXiv v1 Table 5.',
                  't-muted'))
    return svg(W, H, 'Bars for GPT-4o, Sonnet 3.5 and R1-Distill-Qwen-1.5B on four benchmarks. The 1.5B model is best on '
               'AIME 2024 and MATH-500, and worst on GPQA Diamond and LiveCodeBench.', b)


F['p6_small_vs_big'] = small_vs_big()


# ---------------------------------------------------------------- 6. Table 6: RL vs distillation on the same 32B base
def table6():
    cols = [('AIME pass@1', 1), ('AIME cons@64', 2), ('MATH-500', 3), ('GPQA Diamond', 4), ('LiveCodeBench', 5)]
    cls = ['s3', 's4', 's1']
    W, H = 720, 320
    b = [text(20, 24, 'Table 6: the same 32B base, RL alone against distillation (%)', 't-title')]
    b += legend(20, 50, [(c, r[0]) for c, r in zip(cls, T6)])
    pw, L, top, gh = 128, 20, 80, 170
    for k, (name, idx) in enumerate(cols):
        x0 = L + k * 138
        b.append(text(x0 + pw / 2, top + gh + 22, name, 't-tick', 'middle'))
        b.append(f'<line class="axis" x1="{x0}" y1="{top + gh}" x2="{x0 + pw}" y2="{top + gh}"/>')
        for j, r in enumerate(T6):
            v = r[idx]
            h = gh * v / 100
            x = x0 + 6 + j * 40
            b.append(f'<g class="mark"><title>{esc(r[0])}, {name}: {v}</title>'
                     f'<path class="{cls[j]}" d="{bar_path(x, top + gh - h, 34, h)}"/></g>')
            b.append(text(x + 17, top + gh - h - 6, f'{v}', 't-val', 'middle'))
    b.append(text(L, H - 12, 'R1-Zero-Qwen-32B: over 10K steps of RL on Qwen2.5-32B. R1-Distill-Qwen-32B: SFT on the 800k '
                  'samples. arXiv v1 Table 6.', 't-muted'))
    return svg(W, H, 'Bars for QwQ-32B-Preview, R1-Zero-Qwen-32B and R1-Distill-Qwen-32B on five columns. The RL model '
               'is about level with QwQ-32B-Preview; the distilled model is clearly higher on every column.', b)


F['p6_table6'] = table6()


# ---------------------------------------------------------------- 7. our run: loss of R1 text under base and distilled
def our_loss():
    L = load('part6_loss')
    if not L:
        return None
    rows = L['texts']
    W = 720
    rowh, top, Lx = 46, 70, 230
    H = top + rowh * len(rows) + 50
    vmax = max(max(r['base'], r['distill']) for r in rows) * 1.15
    pw = W - Lx - 80
    b = [text(20, 24, 'Mean loss per response token (nats, lower = the model finds the text more likely)', 't-title')]
    b += legend(Lx, 50, [('s3', 'Qwen2.5-Math-1.5B (base)'), ('s1', 'R1-Distill-Qwen-1.5B')])
    for i, r in enumerate(rows):
        y = top + i * rowh
        b.append(text(Lx - 10, y + 24, r['name'], 't-tick', 'end'))
        for j, (v, cls, nm) in enumerate([(r['base'], 's3', 'base'), (r['distill'], 's1', 'distilled')]):
            yy = y + 6 + j * 17
            b.append(hbar(Lx, yy, pw * v / vmax, 14, cls, f'{r["name"]}, {nm}: loss {v:.3f}, perplexity {r["ppl_" + nm.replace("distilled", "distill")]:.2f}'))
            b.append(text(Lx + pw * v / vmax + 5, yy + 12, f'{v:.3f}', 't-val'))
    b.append(text(20, H - 14, 'Each text is at most 1,024 tokens. The last row is an answer the base model wrote itself.',
                  't-muted'))
    return svg(W, H, 'Bars of the mean per-token loss of five texts under the base model and the distilled model. On the '
               'four R1-style texts the distilled model has a much lower loss; on the base model\'s own answer the base '
               'model is lower.', b)


def loss_curve():
    L = load('part6_loss')
    if not L:
        return None
    r = next((t for t in L['texts'] if 'curve_base' in t), None)
    if not r:
        return None
    win = 32

    def smooth(v):
        out, s = [], 0.0
        for i, x in enumerate(v):
            s += x
            if i >= win:
                s -= v[i - win]
            out.append(s / min(i + 1, win))
        return out
    cb, cd = smooth(r['curve_base']), smooth(r['curve_distill'])
    n = len(cb)
    W, H, Lx, T, B = 720, 280, 50, 40, 46
    pw, ph = W - Lx - 150, H - T - B
    ymax = max(1.0, round(max(cb[win:] + cd[win:]) * 1.2 + 0.5))   # the first tokens are left off the scale
    cb = [min(v, ymax) for v in cb]; cd = [min(v, ymax) for v in cd]
    X = lambda i: Lx + i / (n - 1) * pw
    Y = lambda v: T + ph - v / ymax * ph
    b = [text(20, 22, f'Loss along a real R1 trace (running mean over {win} tokens)', 't-title')]
    for t in [x * 0.5 for x in range(0, int(ymax * 2) + 1)]:
        b.append(f'<line class="grid" x1="{Lx}" y1="{Y(t):.1f}" x2="{Lx + pw}" y2="{Y(t):.1f}"/>')
        b.append(text(Lx - 8, Y(t) + 4, str(t), 't-tick', 'end'))
    for k in range(0, n, 256):
        b.append(text(X(k), T + ph + 18, str(k), 't-tick', 'middle'))
    b.append(text(Lx + pw / 2, H - 8, 'token position in the trace', 't-tick', 'middle'))
    for v, cls, nm in [(cb, 'l3', 'base'), (cd, 'l1', 'distilled')]:
        step = max(1, n // 300)
        pts = ' '.join(f'{X(i):.1f},{Y(v[i]):.1f}' for i in range(0, n, step))
        b.append(f'<polyline class="{cls}" points="{pts}"/>')
        yl = T + 20 if nm == 'base' else T + 40
        b.append(f'<rect class="{cls.replace("l", "s")}" x="{X(n - 1) - 4:.1f}" y="{yl - 9}" width="10" height="10" rx="2"/>')
        b.append(text(X(n - 1) + 10, yl, f'{nm}: mean {sum(r["curve_base" if nm == "base" else "curve_distill"]) / n:.3f}', 't-note'))
    return svg(W, H, 'Two lines of per-token loss along the first real R1 trace. The base model line stays well above the '
               'distilled model line from start to end.', b)


# ---------------------------------------------------------------- 8. our run: accuracy and length, base vs distilled
def our_eval():
    E = load('part6_regrade')            # the saved answers of r1_part6_eval.py, regraded with the Part 4 policy
    if not E:
        return None
    S = E['summary']
    W, H = 720, 250
    b = [text(20, 24, f'Our run: 16 MATH-500 problems (levels 1 to 3), {E["k"]} samples each, max {E["max_new"]} new tokens', 't-title')]
    items = [('pass@1 (%)', lambda s: 100 * s['pass1'], 100, lambda v: f'{v:.1f}'),
             ('mean answer length (tokens)', lambda s: s['mean_tokens'], E['max_new'], lambda v: f'{v:.0f}'),
             ('answers that hit the limit', lambda s: s['hit_limit'], S['base']['samples'], lambda v: f'{v:.0f}')]
    b += legend(20, 50, [('s3', 'Qwen2.5-Math-1.5B (base)'), ('s1', 'R1-Distill-Qwen-1.5B')])
    Lx, pw = 220, 400
    for i, (nm, f, vmax, fmt) in enumerate(items):
        y = 76 + i * 56
        b.append(text(Lx - 10, y + 22, nm, 't-tick', 'end'))
        for j, (tag, cls) in enumerate([('base', 's3'), ('distilled', 's1')]):
            v = f(S[tag])
            yy = y + 4 + j * 18
            b.append(hbar(Lx, yy, pw * v / vmax, 14, cls, f'{tag}, {nm}: {fmt(v)}'))
            b.append(text(Lx + pw * v / vmax + 5, yy + 12, fmt(v), 't-val'))
    b.append(text(20, H - 12, 'A cut-off answer counts as wrong; the distilled model is graded after </think>. MacBook (mps), bf16.',
                  't-muted'))
    return svg(W, H, 'Bars comparing the base model and the distilled model in our run: accuracy, mean answer length, '
               'and how many answers ran into the token limit.', b)


def tiny_fig():
    Lt = load('part6_tiny')
    if not Lt or 'after' not in Lt:
        return None
    W, H = 720, 210
    b = [text(20, 24, 'Tiny distillation: Qwen2.5-0.5B-Instruct before and after SFT on teacher traces', 't-title')]
    b += legend(20, 50, [('s3', 'before'), ('s1', 'after')])
    n = Lt['before']['n']
    items = [('accuracy (%)', 'acc', 100, 100), ('mean answer length (tokens)', 'tokens', 1, 1024),
             ('answers with "Wait"', 'wait', 1, n)]
    Lx, pw = 220, 400
    for i, (nm, key, mul, vmax) in enumerate(items):
        y = 70 + i * 44
        b.append(text(Lx - 10, y + 20, nm, 't-tick', 'end'))
        for j, (tag, cls) in enumerate([('before', 's3'), ('after', 's1')]):
            v = Lt[tag][key] * mul
            yy = y + 2 + j * 16
            b.append(hbar(Lx, yy, pw * v / vmax, 13, cls, f'{tag}, {nm}: {v:.1f}'))
            b.append(text(Lx + pw * v / vmax + 5, yy + 11, f'{v:.1f}' if key == 'acc' else f'{v:.0f}', 't-val'))
    return svg(W, H, 'Bars for the 0.5B student before and after a few hundred steps of SFT on traces from the 1.5B '
               'distilled teacher: accuracy, answer length and use of the word Wait.', b)


for key, fn in [('p6_kd_kinds', kd_kinds), ('p6_mask', mask_fig), ('p6_our_loss', our_loss),
                ('p6_loss_curve', loss_curve), ('p6_our_eval', our_eval), ('p6_tiny', tiny_fig)]:
    s = fn()
    if s:
        F[key] = s
json.dump(F, open(os.path.join(RES, 'figs_part6.json'), 'w'))
print('figures:', ', '.join(F))
