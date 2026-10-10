"""Figures for Chapter 3 (how we measure a model). Writes results/figs_ch3.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, l1..l4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7 px per character for t-tick / t-muted, 7.5 for t-note, 8.5 for t-title; 17 px between baselines.
Numbers come from results/ch3_*.json, written by the experiment scripts."""
import json, math, re
import numpy as np
from figlib import svg, text, box, arrow, esc
from agentfig import line

R = lambda n: json.load(open(f'results/{n}.json'))
F = {}


# ---------------------------------------------------------------- helpers (no colours, only classes)
def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def block(x, y, w, h, title, lines=(), cls='box', tcls='t-note', lcls='t-tick', rx=10):
    b = [box(x, y, w, h, cls, rx)]
    ty = y + (h / 2 + 5 if not lines else 22)
    b.append(text(x + w / 2, ty, title, tcls, 'middle'))
    b += para(x + w / 2, y + 40, lines, lcls, 'middle')
    return b


def dot(x, y, r, cls):
    return f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


def rect(x, y, w, h, cls, rx=3, title=None):
    r = f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.5):.1f}" height="{max(h, 0.5):.1f}" rx="{rx}"/>'
    return f'<g class="mark"><title>{esc(title)}</title>{r}</g>' if title else r


def hbar_panel(x0, y0, items, scale, label_w, row_h=30, fmt=lambda v: f'{v:.2f}', bar_h=18):
    """items: (label, value, class). Bars start at x0 + label_w."""
    b = []
    for i, (lab, v, c) in enumerate(items):
        y = y0 + i * row_h
        b.append(text(x0 + label_w - 10, y + bar_h / 2 + 5, lab, 't-tick', 'end'))
        b.append(rect(x0 + label_w, y, v * scale, bar_h, c, 3, f'{lab}: {fmt(v)}'))
        b.append(text(x0 + label_w + v * scale + 6, y + bar_h / 2 + 5, fmt(v), 't-val'))
    return b


def interval(x_of, y, lo, hi, mid, cls, r=5, title=None):
    """A horizontal confidence interval with a dot at the estimate."""
    b = [line(x_of(lo), y, x_of(hi), y, 'axis'), line(x_of(lo), y - 5, x_of(lo), y + 5, 'axis'),
         line(x_of(hi), y - 5, x_of(hi), y + 5, 'axis')]
    d = dot(x_of(mid), y, r, cls)
    b.append(f'<g class="mark"><title>{esc(title)}</title>{d}</g>' if title else d)
    return b


def xaxis(x_of, y, ticks, fmt, label=None, top=None):
    b = [line(x_of(ticks[0]), y, x_of(ticks[-1]), y, 'axis')]
    for t in ticks:
        b.append(text(x_of(t), y + 18, fmt(t), 't-tick', 'middle'))
        if top is not None:
            b.append(f'<line class="grid" x1="{x_of(t):.1f}" y1="{top:.1f}" x2="{x_of(t):.1f}" y2="{y:.1f}"/>')
    if label:
        b.append(text((x_of(ticks[0]) + x_of(ticks[-1])) / 2, y + 38, label, 't-tick', 'middle'))
    return b


# ---------------------------------------------------------------- 1. the landscape of evaluation
def landscape():
    W, bw, bh, gx, gy = 760, 232, 150, 14, 16
    cells = [
        ('box-1', 'Held-out loss, perplexity', 'How well does it predict', 'text it has not seen?', 'WikiText-2, a validation split', 'cheap and smooth, but says', 'nothing about answers'),
        ('box-2', 'Benchmarks with a key', 'Does it reach the known', 'right answer?', 'MMLU, GSM8K, HumanEval', 'leaks into training data,', 'saturates, format-sensitive'),
        ('box-3', 'Human preference', 'Which of two answers do', 'people like more?', 'Chatbot Arena, InstructGPT', 'slow, costly, noisy;', 'style can beat substance'),
        ('box-4', 'A model as the judge', 'Which answer does a strong', 'model prefer?', 'MT-Bench, AlpacaEval', 'position, length and', 'self-preference biases'),
        ('box-1', 'Reward-model tests', 'Does the reward model', 'rank answer pairs right?', 'RewardBench', 'a high score does not mean', 'safe to optimise against'),
        ('box-2', 'Safety and behaviour', 'Does it refuse, invent or', 'comply when it should not?', 'refusal sets, red-teaming', 'endless cases; passing a', 'test is not proof of safety'),
    ]
    b = [text(20, 26, 'Six families of evaluation, from cheapest to closest to what users feel', 't-title')]
    for i, (c, title, q1, q2, ex, w1, w2) in enumerate(cells):
        x, y = 20 + (i % 3) * (bw + gx), 44 + (i // 3) * (bh + gy)
        b.append(box(x, y, bw, bh, c, 10))
        b.append(text(x + 12, y + 24, title, 't-note'))
        b += para(x + 12, y + 46, [q1, q2], 't-tick')
        b.append(text(x + 12, y + 90, 'e.g. ' + ex, 't-muted'))
        b += para(x + 12, y + 114, [w1, w2], 't-muted')
    return svg(W, 44 + 2 * bh + gy + 14, 'Six boxes in two rows: held-out loss, benchmarks with an answer key, human preference, a model as the judge, reward-model tests, and safety and behaviour tests, each with the question it asks, an example and its main weakness.', b)


F['ch3_landscape'] = landscape()


# ---------------------------------------------------------------- 2. Goodhart's law in three places
def goodhart():
    W = 760
    rows = [('knowledge of the subject', 'benchmark score', 'test questions leak into the training data;', 'the score rises, the knowledge does not'),
            ('a better answer', 'a judge model prefers it', 'the judge likes long answers and the first', 'position; padding wins votes'),
            ('what people actually want', 'reward model score', 'the policy finds outputs the reward model', 'over-rates: reward hacking')]
    b = [text(20, 24, "Goodhart's law: once a measure becomes the target, it stops tracking the goal", 't-title')]
    hx = [20, 230, 440]
    for x, h in zip(hx, ['what we want', 'what we can count', 'how optimising the count breaks the link']):
        b.append(text(x, 52, h, 't-muted'))
    for i, (want, count, h1, h2) in enumerate(rows):
        y = 64 + i * 70
        b += block(20, y, 180, 52, want, cls='box-3')
        b.append(arrow(204, y + 26, 226, y + 26))
        b += block(230, y, 180, 52, count, cls='box-2')
        b.append(arrow(414, y + 26, 436, y + 26))
        b.append(box(440, y, 300, 52, 'box-ghost', 10))
        b += para(452, y + 22, [h1, h2], 't-tick')
    return svg(W, 64 + 3 * 70, 'Three rows. Knowledge is measured by a benchmark score, which breaks when test questions leak into training. Answer quality is measured by a judge model, which breaks because judges favour long answers and the first position. Human wishes are measured by a reward model, which breaks through reward hacking.', b)


F['ch3_goodhart'] = goodhart()


# ---------------------------------------------------------------- 3. the sliding window for perplexity
def window():
    P = R('ch3_perplexity')
    ntok = P['A']['Qwen2.5-0.5B base']['tokens']
    W, x0, unit = 760, 40, 64          # one unit = 256 tokens
    b = [text(20, 24, 'Scoring a long text with a fixed context: window 1024 tokens, stride 512', 't-title'),
         text(20, 44, f'WikiText-2 test is {ntok:,} Qwen tokens long, far more than one forward pass can take', 't-muted')]
    b.append(box(x0, 64, unit * 10, 26, 'box-ghost', 4))
    for k in range(11):
        b.append(text(x0 + k * unit, 108, f'{k * 256:,}', 't-tick', 'middle'))
    b.append(text(x0 + unit * 10 + 10, 82, 'tokens ...', 't-tick'))
    for j in range(4):
        y = 128 + j * 44
        start = j * 2
        b.append(text(20, y + 17, f'{j + 1}', 't-note', 'middle'))
        b.append(rect(x0 + start * unit, y, 4 * unit, 26, 'box', 4))
        s0 = start if j == 0 else start + 2
        b.append(rect(x0 + s0 * unit, y, (4 if j == 0 else 2) * unit, 26, 's2', 4))
        b.append(text(x0 + start * unit + 8, y + 18, 'context only' if j else '', 't-tick'))
        b.append(text(x0 + s0 * unit + (4 if j == 0 else 2) * unit / 2, y + 18, 'scored', 't-tick', 'middle'))
    b += para(20, 128 + 4 * 44 + 10, ['Each window re-reads the previous 512 tokens as context but only scores its new 512, so every token is scored once',
                                       'and (after the first window) always with at least 512 tokens of history. Smaller stride = more context, more compute.'], 't-muted')
    return svg(W, 128 + 4 * 44 + 50, 'A long strip of tokens with four overlapping windows of 1024 tokens, each shifted by 512. In the first window all tokens are scored; in the later windows only the last 512 tokens are scored and the first 512 serve as context.', b)


# ---------------------------------------------------------------- 4. perplexity versus bits per byte
def ppl_bars():
    P = R('ch3_perplexity')['A']
    names = ['GPT-2 (124M)', 'Qwen2.5-0.5B base', 'Qwen2.5-0.5B-Instruct']
    cls = ['s4', 's1', 's3']
    W = 760
    b = [text(20, 24, 'WikiText-2 test set: the same text, three models', 't-title')]
    b.append(text(20, 52, 'perplexity (per token: depends on the tokenizer)', 't-muted'))
    b += hbar_panel(20, 64, [(n, P[n]['ppl'], c) for n, c in zip(names, cls)], 6.0, 170, fmt=lambda v: f'{v:.2f}')
    b.append(text(400, 52, 'bits per byte (comparable across tokenizers)', 't-muted'))
    b += hbar_panel(400, 64, [('', P[n]['bpb'], c) for n, c in zip(names, cls)], 260, 10, fmt=lambda v: f'{v:.3f}')
    y = 64 + 3 * 30 + 16
    b += para(20, y, [f'tokens: GPT-2 {P[names[0]]["tokens"]:,} ({P[names[0]]["bytes_per_token"]:.2f} bytes each), '
                      f'Qwen {P[names[1]]["tokens"]:,} ({P[names[1]]["bytes_per_token"]:.2f} bytes each).',
                      'Lower is better on both. Perplexity divides by a different number of tokens for each tokenizer,',
                      'so only bits per byte can say which model predicts this text better.'], 't-muted')
    return svg(W, y + 56, 'Two bar panels. Left: perplexity of GPT-2, Qwen2.5-0.5B base and Qwen2.5-0.5B-Instruct on WikiText-2. Right: bits per byte for the same three.', b)


# ---------------------------------------------------------------- 5. perplexity is not usefulness
def ppl_vs_use():
    P = R('ch3_perplexity')
    MM, G = R('ch3_mmlu'), R('ch3_gsm8k')
    try:
        GL = R('ch3_gsm8k_long')
    except FileNotFoundError:
        GL = None
    acc = lambda c: sum(c) / len(c)
    rows = [('WikiText-2 perplexity', P['A']['Qwen2.5-0.5B base']['ppl'], P['A']['Qwen2.5-0.5B-Instruct']['ppl'], 'lower', lambda v: f'{v:.2f}'),
            ('perplexity of chat answers', P['C']['Qwen2.5-0.5B base']['ppl'], P['C']['Qwen2.5-0.5B-Instruct']['ppl'], 'lower', lambda v: f'{v:.2f}'),
            ('MMLU, 5-shot letter', acc(MM['models']['base']['letter-5shot']['correct']), acc(MM['models']['instruct']['letter-5shot']['correct']), 'higher', lambda v: f'{100 * v:.1f}%'),
            ('GSM8K, as each is meant to be used', acc(G['models']['base']['flexible_correct']), acc(GL['strict_correct'] if GL else G['models']['instruct']['strict_correct']), 'higher', lambda v: f'{100 * v:.1f}%')]
    W = 760
    b = [text(20, 24, 'Base or Instruct: which is "better"? It depends on the measurement', 't-title'),
         text(330, 52, 'base', 't-note', 'middle'), text(470, 52, 'Instruct', 't-note', 'middle'), text(620, 52, 'better is', 't-muted', 'middle')]
    for i, (lab, vb, vi, better, f) in enumerate(rows):
        y = 66 + i * 40
        wins_i = (vi < vb) if better == 'lower' else (vi > vb)
        b.append(text(20, y + 19, lab, 't-tick'))
        b.append(box(270, y, 120, 28, 'box-on' if not wins_i else 'box', 6))
        b.append(text(330, y + 19, f(vb), 't-val', 'middle'))
        b.append(box(410, y, 120, 28, 'box-on' if wins_i else 'box', 6))
        b.append(text(470, y + 19, f(vi), 't-val', 'middle'))
        b.append(text(620, y + 19, better, 't-muted', 'middle'))
    y = 66 + 4 * 40 + 8
    b += para(20, y, ['Highlighted: the winner on that row. The base model predicts Wikipedia text better; the Instruct model predicts',
                      'assistant answers better. Neither number says which one is more useful: that needs task metrics.'], 't-muted')
    return svg(W, y + 40, 'A table of four measurements for the base and Instruct models: WikiText-2 perplexity, perplexity of chat answers, MMLU accuracy and GSM8K accuracy, with the winner of each row highlighted.', b)


# ---------------------------------------------------------------- 6. three ways to score one multiple-choice question
def mc_scoring():
    D = R('ch3_mc_demo')
    m = D['models']['base']
    L = 'ABCD'
    W = 760
    b = [text(20, 24, 'Three ways to score the same multiple-choice question (Qwen2.5-0.5B base)', 't-title')]
    q = D['question']
    qs = [q[i:i + 100] for i in range(0, len(q), 100)][:2]
    b += para(20, 48, qs, 't-tick')
    y0 = 48 + 17 * len(qs) + 4
    for i, c in enumerate(D['choices']):
        b.append(text(20 + (i % 2) * 360, y0 + (i // 2) * 17, f'{L[i]}. {c}' + ('   (correct)' if i == D['answer'] else ''),
                      't-strong t-tick' if i == D['answer'] else 't-tick'))
    top = y0 + 46
    cw = 236
    titles = ['1. probability of the letter', '2. log-probability of the text', '3. let it write, read the letter']
    for k in range(3):
        x = 20 + k * (cw + 10)
        b.append(box(x, top, cw, 196, 'box-ghost', 10))
        b.append(text(x + 12, top + 22, titles[k], 't-note'))
    # panel 1
    p4 = m['p_letters']
    mx = max(p4)
    for i in range(4):
        y = top + 40 + i * 30
        b.append(text(32, y + 14, f'" {L[i]}"', 't-tick'))
        b.append(rect(70, y, 130 * p4[i] / mx, 18, 's3' if i == D['answer'] else 's1', 3, f'P(" {L[i]}") = {p4[i]:.3f}'))
        b.append(text(70 + 130 * p4[i] / mx + 4, y + 14, f'{p4[i]:.3f}', 't-val'))
    b.append(text(32, top + 178, f'pick: {L[int(np.argmax(p4))]}  (letters hold {100 * sum(p4):.1f}%)', 't-muted'))
    # panel 2
    cl = m['cloze']
    lo = min(cl)
    x = 20 + cw + 10
    for i in range(4):
        y = top + 40 + i * 30
        w = 120 * (cl[i] - lo + 1) / (max(cl) - lo + 1)
        b.append(text(x + 12, y + 14, L[i], 't-tick'))
        b.append(rect(x + 30, y, w, 18, 's3' if i == D['answer'] else 's1', 3, f'log P(text {L[i]}) = {cl[i]:.2f}'))
        b.append(text(x + 34 + w, y + 14, f'{cl[i]:.2f}', 't-val'))
    b.append(text(x + 12, top + 178, f'pick: {L[int(np.argmax(cl))]}  (sum of log-probs; higher wins)', 't-muted'))
    # panel 3
    x = 20 + 2 * (cw + 10)
    g = m['generated'].replace('\n', ' / ')
    lines = [g[i:i + 26] for i in range(0, min(len(g), 104), 26)]
    b += para(x + 12, top + 50, [f'"{l}' if i == 0 else l for i, l in enumerate(lines)], 't-mono t-tick')
    found = re.search(r'\b([ABCD])\b', m['generated'])
    b.append(text(x + 12, top + 178, f'pick: {found.group(1) if found else "none"}  (first letter A to D)', 't-muted'))
    return svg(W, top + 210, 'One MMLU question and three ways to score the base model on it: the probabilities of the letter tokens after "Answer:", the log-probabilities of each option text, and the letter found in a generated answer.', b)


# ---------------------------------------------------------------- 7. MMLU results with confidence intervals
def mmlu_results():
    S = R('ch3_stats')['mmlu']
    rows = [('letter-5shot', '5-shot, letter probability'), ('letter-0shot', '0-shot, letter probability'),
            ('cloze', '0-shot, option text'), ('cloze-norm', '0-shot, option text per character'),
            ('chat-letter', 'chat template, letter probability'), ('generate', 'generate 48 tokens, read the letter'),
            ('generate-256', 'generate 256 tokens, read the letter')]
    W, L0, X0, X1 = 760, 250, 0.15, 0.60
    x_of = lambda v: L0 + (v - X0) / (X1 - X0) * (W - L0 - 40)
    b = [text(20, 24, f'MMLU, {S["n"]} questions: accuracy with 95% bootstrap intervals', 't-title'),
         dot(26, 46, 5, 's1'), text(36, 50, 'base', 't-tick'), dot(96, 46, 5, 's3'), text(106, 50, 'Instruct', 't-tick')]
    top = 70
    for i, (k, lab) in enumerate(rows):
        y = top + i * 40
        b.append(text(20, y + 16, lab, 't-tick'))
        for j, (mname, c) in enumerate([('base', 's1'), ('instruct', 's3')]):
            r = S[mname].get(k)
            if r:
                b += interval(x_of, y + 6 + j * 14, r['lo'], r['hi'], r['acc'], c, 5, f'{mname}, {lab}: {100 * r["acc"]:.1f}% ({100 * r["lo"]:.1f} to {100 * r["hi"]:.1f})')
                b.append(text(x_of(r['hi']) + 8, y + 10 + j * 14, f'{100 * r["acc"]:.1f}%', 't-val'))
    bot = top + len(rows) * 40 + 4
    for v, lab in [(0.25, 'chance 25%'), (0.475, 'reported 47.5% (base, 5-shot)')]:
        b.append(f'<line class="grid" x1="{x_of(v):.1f}" y1="{top - 8}" x2="{x_of(v):.1f}" y2="{bot}"/>')
        b.append(text(x_of(v), top - 12, lab, 't-muted', 'middle'))
    b += xaxis(x_of, bot, [0.15, 0.2, 0.3, 0.4, 0.5, 0.6], lambda t: f'{100 * t:.0f}%')
    return svg(W, bot + 30, 'Accuracy of the base and Instruct models on 400 MMLU questions under six scoring methods, each with a 95% bootstrap interval, with chance (25%) and the reported 47.5% marked.', b)


# ---------------------------------------------------------------- 8. moving the right answer
def position():
    M = R('ch3_mmlu')
    L = 'ABCD'
    W = 760
    b = [text(20, 24, 'Move the right answer to A, B, C or D: does the score change?', 't-title'),
         text(20, 44, 'same 400 questions, 0-shot letter probability; left: accuracy, right: how often each letter is predicted', 't-muted')]
    for j, (mname, c, lab) in enumerate([('base', 's1', 'base'), ('instruct', 's3', 'Instruct')]):
        y0 = 70 + j * 150
        b.append(text(20, y0 + 10, lab, 't-note'))
        for i in range(4):
            acc = sum(M['models'][mname][f'answer-at-{L[i]}']['correct']) / M['n']
            y = y0 + 22 + i * 28
            b.append(text(40, y + 14, f'right answer at {L[i]}', 't-tick'))
            b.append(rect(160, y, 220 * acc, 18, c, 3, f'{lab}, answer at {L[i]}: {acc:.1%}'))
            b.append(text(166 + 220 * acc, y + 14, f'{100 * acc:.1f}%', 't-val'))
        preds = M['models'][mname]['letter-0shot']['pred']
        for i in range(4):
            f = preds.count(i) / len(preds)
            y = y0 + 22 + i * 28
            b.append(text(470, y + 14, f'predicts {L[i]}', 't-tick'))
            b.append(rect(550, y, 300 * f, 18, c, 3, f'{lab} predicts {L[i]}: {f:.1%}'))
            b.append(text(556 + 300 * f, y + 14, f'{100 * f:.0f}%', 't-val'))
    truth = [M['answer'].count(i) / M['n'] for i in range(4)]
    b.append(text(20, 70 + 2 * 150 + 4, 'for comparison, in the original order the right letter is ' + ', '.join(f'{L[i]} {100 * truth[i]:.0f}%' for i in range(4)), 't-muted'))
    return svg(W, 70 + 2 * 150 + 16, 'For the base and the Instruct model: accuracy when the right answer is always placed at A, B, C or D, and how often each letter is predicted in the original order.', b)


# ---------------------------------------------------------------- 9. the bootstrap, step by step
def bootstrap():
    S = R('ch3_stats')
    bs = S['bootstrap_example']
    vals = np.array(bs['samples'])
    W = 760
    b = [text(20, 24, f'The bootstrap: how sure are we of {100 * bs["acc"]:.1f}%?', 't-title')]
    steps = [('1', f'{bs["n"]} results (1 = right)', 'the real per-question scores'),
             ('2', f'resample {bs["n"]}', 'with replacement: some twice'),
             ('3', 'take the mean', 'one "could have been" score'),
             ('4', f'repeat {bs["B"]:,} times', 'middle 95% = the interval')]
    for i, (n, t1, t2) in enumerate(steps):
        x = 20 + i * 184
        b.append(box(x, 40, 172, 58, 'box-1' if i < 3 else 'box-2', 8))
        b.append(f'<circle class="node on" cx="{x + 16:.1f}" cy="{57:.1f}" r="10"/>')
        b.append(text(x + 16, 61, n, 't-tick', 'middle'))
        b.append(text(x + 32, 61, t1, 't-tick'))
        b.append(text(x + 12, 86, t2, 't-muted'))
        if i < 3:
            b.append(arrow(x + 172, 69, x + 184, 69))
    lo, hi = bs['lo'], bs['hi']
    edges = np.linspace(vals.min() - 0.0025, vals.max() + 0.0025, 32)
    h, _ = np.histogram(vals, edges)
    X0, X1, top, base = edges[0], edges[-1], 130, 300
    x_of = lambda v: 60 + (v - X0) / (X1 - X0) * 640
    for k in range(len(h)):
        mid = (edges[k] + edges[k + 1]) / 2
        c = 's3' if lo <= mid <= hi else 's4'
        hh = (base - top) * h[k] / h.max()
        b.append(rect(x_of(edges[k]) + 1, base - hh, x_of(edges[k + 1]) - x_of(edges[k]) - 2, hh, c, 2, f'{100 * edges[k]:.1f} to {100 * edges[k + 1]:.1f}%: {h[k]} resamples'))
    for v, lab in [(lo, f'2.5%: {100 * lo:.1f}%'), (hi, f'97.5%: {100 * hi:.1f}%')]:
        b.append(f'<line class="path" x1="{x_of(v):.1f}" y1="{top - 6}" x2="{x_of(v):.1f}" y2="{base}"/>')
        b.append(text(x_of(v), top - 10, lab, 't-tick', 'middle'))
    ticks = [t for t in np.arange(0.30, 0.70, 0.02) if X0 <= t <= X1]
    b += xaxis(x_of, base, ticks, lambda t: f'{100 * t:.0f}%', f'accuracy of each resample ({bs["label"]})')
    return svg(W, base + 46, f'Four steps of the bootstrap, then a histogram of {len(vals)} resampled accuracies with the middle 95% shaded and its two ends marked.', b)


# ---------------------------------------------------------------- 10. GSM8K results
def gsm8k():
    S = R('ch3_stats')['gsm8k']
    W, L0 = 760, 270
    x_of = lambda v: L0 + v / 0.7 * (W - L0 - 60)
    rows = [('base', 'flexible', 'base, 4-shot, last number', 's1'), ('base', 'strict', 'base, 4-shot, "#### N" only', 's1'),
            ('instruct', 'flexible', 'Instruct, chat, last number', 's3'), ('instruct', 'strict', 'Instruct, chat, \\boxed{N} only', 's3'),
            ('instruct', 'strict-1024', 'same, up to 1,024 tokens', 's3')]
    rows = [r for r in rows if r[1] in S[r[0]]]
    b = [text(20, 24, f'GSM8K, {S["n"]} questions: exact-match accuracy with 95% bootstrap intervals', 't-title')]
    top = 80
    for i, (m, k, lab, c) in enumerate(rows):
        r = S[m][k]
        y = top + i * 34
        b.append(text(20, y + 5, lab, 't-tick'))
        b += interval(x_of, y, r['lo'], r['hi'], r['acc'], c, 5, f'{lab}: {100 * r["acc"]:.1f}%')
        b.append(text(x_of(r['hi']) + 8, y + 5, f'{100 * r["acc"]:.1f}%', 't-val'))
    bot = top + len(rows) * 34
    for v, lab in [(0.416, 'reported base 41.6%'), (0.496, 'reported Instruct 49.6%')]:
        b.append(f'<line class="grid" x1="{x_of(v):.1f}" y1="{top - 14}" x2="{x_of(v):.1f}" y2="{bot - 10}"/>')
    b.append(text(x_of(0.416) - 4, top - 18 + 0, 'reported: base 41.6%', 't-muted', 'end'))
    b.append(text(x_of(0.496) + 4, top - 18 + 0, 'Instruct 49.6%', 't-muted'))
    b += xaxis(x_of, bot - 10, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7], lambda t: f'{100 * t:.0f}%')
    return svg(W, bot + 26, 'GSM8K accuracy for the base and Instruct models under strict and flexible answer extraction, with 95% bootstrap intervals and the reported numbers marked.', b)


# ---------------------------------------------------------------- 11. pass@k
def passk():
    H = R('ch3_humaneval')
    n = H['n']
    ks = list(range(1, n + 1))
    counts = H['counts']

    def pk(c, k):
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    unb = [np.mean([pk(c, k) for c in counts]) for k in ks]
    naive = [np.mean([1 - (1 - c / n) ** k for c in counts]) for k in ks]
    W, L0, T, PW, PH = 760, 60, 50, 380, 220
    X = lambda k: L0 + (k - 1) / (n - 1) * PW
    Y = lambda v: T + PH - v * PH / 0.8
    b = [text(20, 24, f'HumanEval, Qwen2.5-0.5B base, {n} samples per problem', 't-title')]
    for t in [0, 0.2, 0.4, 0.6, 0.8]:
        b += [f'<line class="grid" x1="{L0}" y1="{Y(t):.1f}" x2="{L0 + PW}" y2="{Y(t):.1f}"/>', text(L0 - 8, Y(t) + 4, f'{100 * t:.0f}%', 't-tick', 'end')]
    for k in ks:
        b.append(text(X(k), T + PH + 18, str(k), 't-tick', 'middle'))
    b.append(text(L0 + PW / 2, T + PH + 38, 'k (attempts allowed)', 't-tick', 'middle'))
    for vals, lc, dc, lab in [(unb, 'l3', 's3', 'unbiased estimator'), (naive, 'l2', 's2', '1 - (1 - c/n)^k (biased)')]:
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(k):.1f},{Y(v):.1f}' for k, v in zip(ks, vals)) + '"/>')
        for k, v in zip(ks, vals):
            b.append(f'<g class="mark"><title>{lab}, k={k}: {100 * v:.1f}%</title>{dot(X(k), Y(v), 4, dc)}</g>')
    b.append(text(X(n) + 8, Y(unb[-1]) + 4, f'{100 * unb[-1]:.1f}%', 't-val'))
    b.append(text(X(1) + 8, Y(unb[0]) + 14, f'pass@1 {100 * unb[0]:.1f}%', 't-val'))
    b.append(rect(L0 + 10, T + 6, 10, 10, 's3', 2)); b.append(text(L0 + 26, T + 15, 'unbiased estimator', 't-tick'))
    b.append(rect(L0 + 10, T + 24, 10, 10, 's2', 2)); b.append(text(L0 + 26, T + 33, '1 - (1 - c/n)^k, biased low', 't-tick'))
    # histogram of c
    hx, hw = 500, 230
    b.append(text(hx, T + 2, 'problems by number of passing samples c', 't-muted'))
    hist = [counts.count(c) for c in range(n + 1)]
    mx = max(hist)
    for c in range(n + 1):
        x = hx + c * hw / (n + 1)
        hh = (PH - 44) * hist[c] / mx
        b.append(rect(x + 1, T + PH - hh, hw / (n + 1) - 2, hh, 's1' if c else 's4', 2, f'c = {c}: {hist[c]} problems'))
        b.append(text(x + hw / (n + 1) / 2, T + PH + 18, str(c), 't-tick', 'middle'))
        b.append(text(x + hw / (n + 1) / 2, T + PH - hh - 4, str(hist[c]), 't-tick', 'middle'))
    b.append(text(hx + hw / 2, T + PH + 38, 'c (out of 10)', 't-tick', 'middle'))
    return svg(W, T + PH + 50, 'Left: pass@k for k from 1 to 10, computed with the unbiased estimator and with the biased shortcut. Right: how many problems had 0 to 10 passing samples.', b)


# ---------------------------------------------------------------- 12. the 13-gram check
def ngram():
    C = R('ch3_contamination')['A']
    W = 760
    norm = lambda s: [w for w in re.sub(r'[^0-9a-z ]+', ' ', s.lower()).split() if w]
    grams = lambda ws: {' '.join(ws[i:i + 13]) for i in range(len(ws) - 12)}

    def words_row(ws, y, hi, cls_hi):
        x, out, row = 20, [], 0
        for i, w in enumerate(ws):
            ww = 7.2 * len(w) + 12
            if x + ww > W - 20:
                x, row = 20, row + 1
            out.append(rect(x, y + row * 28, ww, 22, cls_hi if i in hi else 'box', 4))
            out.append(text(x + ww / 2, y + row * 28 + 15, w, 't-tick', 'middle'))
            x += ww + 4
        return out, row + 1
    b = [text(20, 24, 'The 13-gram check: does any 13-word window of the test item appear in the training data?', 't-title')]
    orig, rew = C['reword_example']
    orig2, para = C['para_example']
    y = 46
    for lab, o, v, cls in [('lightly reworded (changed words shaded)', orig, rew, 's4'), ('paraphrased by a model (words not in the original shaded)', orig2, para, 's2')]:
        wo, wv = norm(o), norm(v)
        kept = len(grams(wo) & grams(wv))
        b.append(text(20, y, f'original test question: {len(grams(wo))} windows of 13 words', 't-muted'))
        parts, nr = words_row(wo, y + 10, set(), 's1')
        b += parts
        y += 10 + nr * 28 + 14
        so = set(wo)
        b.append(text(20, y, f'{lab}: {kept} of its windows still match the original', 't-muted'))
        parts, nr = words_row(wv, y + 10, {i for i, w in enumerate(wv) if w not in so}, cls)
        b += parts
        y += 10 + nr * 28 + 26
    return svg(W, y - 8, 'Two GSM8K test questions, each followed by a changed version: a light rewording with a few words swapped, which still shares many 13-word windows with the original, and a full paraphrase by a model, which shares none or almost none.', b)


# ---------------------------------------------------------------- 13. one Elo update
def elo_update():
    W = 760
    e = 1 / (1 + 10 ** (-100 / 400))
    b = [text(20, 24, 'One Elo update (K = 32): the surprise decides how far the ratings move', 't-title')]
    fr = [('before the game', ['A: 1200', 'B: 1100'], ['expected score of A', f'1 / (1 + 10^(-100/400)) = {e:.3f}']),
          ('A wins (expected)', [f'A: 1200 + 32 x {1 - e:.3f} = {1200 + 32 * (1 - e):.1f}', f'B: 1100 - 32 x {1 - e:.3f} = {1100 - 32 * (1 - e):.1f}'], ['small surprise,', 'small move (11.5 points)']),
          ('B wins (an upset)', [f'A: 1200 - 32 x {e:.3f} = {1200 - 32 * e:.1f}', f'B: 1100 + 32 x {e:.3f} = {1100 + 32 * e:.1f}'], ['big surprise,', 'big move (20.5 points)'])]
    for i, (t, l1, l2) in enumerate(fr):
        x = 20 + i * 248
        b.append(box(x, 40, 236, 150, 'box' if i == 0 else ('box-3' if i == 1 else 'box-2'), 10))
        b.append(f'<circle class="node on" cx="{x + 18:.1f}" cy="{58:.1f}" r="11"/>')
        b.append(text(x + 18, 62.5, str(i + 1), 't-note', 'middle'))
        b.append(text(x + 36, 63, t, 't-note'))
        b += para(x + 14, 96, l1, 't-tick')
        b += para(x + 14, 150, l2, 't-muted')
    b.append(text(20, 214, 'The two ratings always move by the same amount in opposite directions, so the total stays the same.', 't-muted'))
    return svg(W, 226, 'Three frames: A rated 1200 and B rated 1100, expected score 0.64 for A; if A wins, A gains 11.5 and B loses 11.5; if B wins, A loses 20.5 and B gains 20.5.', b)


# ---------------------------------------------------------------- 14. ratings recovered from simulated votes
def arena_fit():
    A = R('ch3_arena')
    W, L0 = 760, 110
    lo_all = min(min(A['elo_order_lo']), min(A['bt_lo'])) - 20
    hi_all = max(max(A['elo_order_hi']), max(A['bt_hi'])) + 20
    X0, X1 = math.floor(lo_all / 50) * 50, math.ceil(hi_all / 50) * 50
    x_of = lambda v: L0 + (v - X0) / (X1 - X0) * (W - L0 - 30)
    b = [text(20, 24, f'{A["votes"]:,} simulated votes: how well do the ratings come back?', 't-title'),
         line(L0, 38, L0, 54, 'path'), text(L0 + 10, 50, 'true rating', 't-tick'),
         dot(L0 + 110, 46, 5, 's3'), text(L0 + 122, 50, 'Bradley-Terry fit, 95% bootstrap CI', 't-tick'),
         dot(L0 + 380, 46, 5, 's2'), text(L0 + 392, 50, 'online Elo: range over 100 vote orders', 't-tick')]
    top = 76
    for i, nm in enumerate(A['names']):
        y = top + i * 44
        b.append(text(20, y + 10, nm, 't-tick'))
        b.append(line(x_of(A['true'][i]), y - 10, x_of(A['true'][i]), y + 26, 'path'))
        b += interval(x_of, y, A['bt_lo'][i], A['bt_hi'][i], A['bt'][i], 's3', 5, f'{nm}: BT {A["bt"][i]:.0f} ({A["bt_lo"][i]:.0f} to {A["bt_hi"][i]:.0f})')
        b += interval(x_of, y + 16, A['elo_order_lo'][i], A['elo_order_hi'][i], A['elo'][i], 's2', 4, f'{nm}: Elo {A["elo"][i]:.0f} (orders: {A["elo_order_lo"][i]:.0f} to {A["elo_order_hi"][i]:.0f})')
    bot = top + 6 * 44
    b += xaxis(x_of, bot - 10, list(range(X0, X1 + 1, 50)), lambda t: str(t), 'rating (model-F anchored at 1000)')
    return svg(W, bot + 40, 'For six simulated models: the true rating, the Bradley-Terry estimate with its 95% bootstrap interval, and the spread of online Elo ratings over 100 random orders of the same votes.', b)


# ---------------------------------------------------------------- 15. more votes, tighter intervals
def votes_ci():
    A = R('ch3_arena')
    W = 760
    b = [text(20, 24, 'How many votes do you need? Width of the 95% interval', 't-title')]
    b.append(text(20, 52, "model-A's Bradley-Terry rating (true 1250)", 't-muted'))
    L0, X0, X1 = 110, 1080, 1340
    x_of = lambda v: L0 + (v - X0) / (X1 - X0) * 250
    for i, r in enumerate(A['by_n']):
        y = 76 + i * 34
        b.append(text(20, y + 5, f'{r["n"]:,} votes', 't-tick'))
        b += interval(x_of, y, r['lo'], r['hi'], r['fit'], 's3', 5, f'{r["n"]} votes: {r["fit"]:.0f} ({r["lo"]:.0f} to {r["hi"]:.0f})')
    b.append(f'<line class="grid" x1="{x_of(1250):.1f}" y1="64" x2="{x_of(1250):.1f}" y2="{76 + 4 * 34 - 16}"/>')
    b += xaxis(x_of, 76 + 4 * 34 - 10, [1100, 1150, 1200, 1250, 1300], str)
    b.append(text(420, 52, f'win rate of model-A over model-B (true {A["p_true_AB"]:.3f})', 't-muted'))
    L1 = 510
    x2 = lambda v: L1 + (v - 0.3) / 0.5 * 230
    for i, r in enumerate(A['winrate']):
        y = 76 + i * 34
        b.append(text(420, y + 5, f'{r["n"]:,} votes', 't-tick'))
        b += interval(x2, y, r['blo'], r['bhi'], r['p'], 's1', 5, f'{r["n"]} votes: {r["p"]:.3f} ({r["blo"]:.3f} to {r["bhi"]:.3f})')
    b.append(f'<line class="grid" x1="{x2(0.5):.1f}" y1="64" x2="{x2(0.5):.1f}" y2="{76 + 4 * 34 - 16}"/>')
    b += xaxis(x2, 76 + 4 * 34 - 10, [0.3, 0.4, 0.5, 0.6, 0.7, 0.8], lambda t: f'{t:.1f}')
    b.append(text(20, 76 + 4 * 34 + 40, 'Four times the votes halves the width. Intervals that cross 0.5 cannot tell you which model is better.', 't-muted'))
    return svg(W, 76 + 4 * 34 + 52, 'Left: the 95% interval of a Bradley-Terry rating for 300, 1,000, 3,000 and 10,000 simulated votes. Right: the 95% interval of a win rate for 20, 100, 400 and 1,600 votes.', b)


# ---------------------------------------------------------------- 16. the judge's position bias
def judge():
    J = R('ch3_judge')
    W = 760
    sets = [('easy', 'clear quality gap (AlpacaEval easy)'), ('maths', 'correct vs wrong maths (PRM)'), ('length', 'same answer, padded vs plain')]
    b = [text(20, 24, 'Each pair judged twice, answers swapped: what does the judge do?', 't-title')]
    cats = [('consistent_right', 's3', 'right both times'), ('always_first', 's1', 'says "A" both times'),
            ('always_second', 's2', 'says "B" both times'), ('wrong', 's4', 'wrong both times')]
    lx = 20
    for k, c, lab in cats:
        b.append(rect(lx, 38, 10, 10, c, 2)); b.append(text(lx + 16, 47, lab, 't-tick'))
        lx += 30 + 7 * len(lab)
    y = 70
    for jname, jr in J['judges'].items():
        b.append(text(20, y + 4, jname, 't-note'))
        y += 14
        for key, lab in sets:
            r = jr[key]
            wrong = max(0.0, 1 - r['consistent_right'] - r['always_first'] - r['always_second'])
            vals = [r['consistent_right'], r['always_first'], r['always_second'], wrong]
            if key == 'length':
                lab2 = 'padded vs plain (right = prefers padded)'
            else:
                lab2 = lab
            b.append(text(20, y + 14, lab2, 't-tick'))
            x = 300
            for (k, c, nm), v in zip(cats, vals):
                w = 420 * v
                b.append(rect(x, y, w, 20, c, 0, f'{jname}, {lab}: {nm} {100 * v:.0f}%'))
                if w > 26:
                    b.append(text(x + w / 2, y + 14, f'{100 * v:.0f}', 't-cell on', 'middle'))
                x += w
            y += 26
        y += 18
    b.append(text(20, y, 'Swapping and keeping only consistent verdicts (or averaging the two orders) removes the position effect, not the weak judgment.', 't-muted'))
    return svg(W, y + 12, 'Stacked bars for two small judges on three sets of pairs: the share of pairs judged right in both orders, "A" both times, "B" both times, and wrong both times.', b)


# ---------------------------------------------------------------- 17. reward models on RewardBench
def rewardbench():
    B = R('ch3_rewardbench')
    W = 760
    secs = ['Chat', 'Chat Hard', 'Safety', 'Reasoning']
    rms = list(B['rms'])
    cls = ['s4', 's3', 's1']
    b = [text(20, 24, 'RewardBench accuracy, 100 pairs per section (dashed line: coin flip)', 't-title')]
    lx = 20
    for nm, c in zip(rms, cls):
        b.append(rect(lx, 38, 10, 10, c, 2)); b.append(text(lx + 16, 47, nm, 't-tick'))
        lx += 40 + 7 * len(nm)
    T, H, gw = 70, 200, 135
    Y = lambda v: T + H - v * H
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        b += [f'<line class="grid" x1="60" y1="{Y(t):.1f}" x2="{60 + 5 * gw:.1f}" y2="{Y(t):.1f}"/>', text(52, Y(t) + 4, f'{100 * t:.0f}%', 't-tick', 'end')]
    groups = secs + ['overall']
    for g, sec in enumerate(groups):
        x0 = 70 + g * gw
        for k, (nm, c) in enumerate(zip(rms, cls)):
            v = B['rms'][nm]['overall'] if sec == 'overall' else B['rms'][nm]['by_section'][sec]
            b.append(rect(x0 + k * 36, Y(v), 31, H * v, c, 3, f'{nm}, {sec}: {100 * v:.0f}%'))
            b.append(text(x0 + k * 36 + 15.5, Y(v) - 5, f'{100 * v:.0f}', 't-tick', 'middle'))
        b.append(text(x0 + 51, T + H + 18, sec, 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="60" y1="{Y(0.5):.1f}" x2="{60 + 5 * gw:.1f}" y2="{Y(0.5):.1f}" stroke-dasharray="5 4"/>')
    return svg(W, T + H + 30, 'Grouped bars: accuracy of three reward models on the Chat, Chat Hard, Safety and Reasoning sections of RewardBench and overall, with 50% marked as a coin flip.', b)


# ---------------------------------------------------------------- 18. best-of-n: proxy versus gold
def bestofn():
    Bn = R('ch3_bestofn')
    rows = Bn['rows']
    ns = [r['n'] for r in rows]
    W, T, PH, PW = 760, 60, 200, 230
    b = [text(20, 24, f'Best-of-n against a reward model: {Bn["nq"]} GSM8K questions, Qwen2.5-0.5B-Instruct', 't-title')]

    def panel(x0, title, series, ymin, ymax, ticks, fmt):
        X = lambda n: x0 + math.log2(n) / 4 * PW
        Y = lambda v: T + PH - (v - ymin) / (ymax - ymin) * PH
        o = [text(x0, T - 14, title, 't-muted')]
        for t in ticks:
            o += [f'<line class="grid" x1="{x0}" y1="{Y(t):.1f}" x2="{x0 + PW}" y2="{Y(t):.1f}"/>', text(x0 - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
        for n in ns:
            o.append(text(X(n), T + PH + 18, str(n), 't-tick', 'middle'))
        o.append(text(x0 + PW / 2, T + PH + 38, 'n (samples to choose from)', 't-tick', 'middle'))
        ends = []
        for key, lc, dc, lab in series:
            vals = [r[key] for r in rows]
            o.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(n):.1f},{Y(v):.1f}' for n, v in zip(ns, vals)) + '"/>')
            for n, v in zip(ns, vals):
                o.append(f'<g class="mark"><title>{lab}, n={n}: {fmt(v)}</title>{dot(X(n), Y(v), 4, dc)}</g>')
            ends.append([Y(vals[-1]) + 4, lab])
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 15)
        for y, lab in ends:
            o.append(text(X(ns[-1]) + 8, y, lab, 't-tick'))
        return o
    pr = [r['proxy'] for r in rows]
    lo, hi = math.floor(min(pr)), math.ceil(max(pr))
    b += panel(70, 'proxy: reward-model score of the kept answer', [('proxy', 'l2', 's2', 'RM score')], lo, hi,
               list(np.linspace(lo, hi, 5)), lambda v: f'{v:.1f}')
    b += panel(430, 'gold: share of kept answers that are right', [('oracle', 'l4', 's4', 'oracle'), ('gold', 'l3', 's3', 'best-of-n'),
                                                                  ('majority', 'l1', 's1', 'majority')], 0, 1.0,
               [0, 0.25, 0.5, 0.75, 1.0], lambda v: f'{100 * v:.0f}%')
    return svg(W, T + PH + 50, 'Left: the reward-model score of the answer picked by best-of-n, rising with n. Right: the accuracy of the picked answer, of a majority vote, and of an oracle that picks any right answer, for n from 1 to 16.', b)


# ---------------------------------------------------------------- 19. evaluation inside a training run
def pipeline():
    W = 760
    b = [text(20, 24, 'Where evaluation sits in a training run', 't-title')]
    stages = [('pretraining', 'box-1'), ('SFT', 'box-2'), ('preference tuning / RL', 'box-3'), ('release', 'box-4')]
    xs = [20, 230, 400, 610]
    ws = [180, 140, 180, 130]
    for (s, c), x, w in zip(stages, xs, ws):
        b.append(box(x, 44, w, 34, c, 8))
        b.append(text(x + w / 2, 66, s, 't-note', 'middle'))
    for x, w, nx in zip(xs[:-1], ws[:-1], xs[1:]):
        b.append(arrow(x + w + 4, 61, nx - 4, 61))
    rows = [('every few hundred steps', 'held-out loss on a validation split', 'catches divergence, overfitting, data bugs'),
            ('every checkpoint', 'a fast benchmark suite (few-shot, small subsets)', 'is the model still learning skills?'),
            ('after each stage', 'full benchmarks + a regression suite', 'did this stage break something that worked?'),
            ('before release', 'human or judge comparisons, safety tests', 'is it better for users, and safe enough?')]
    y = 100
    for i, (when, what, why) in enumerate(rows):
        b.append(box(20, y, 720, 44, 'box-ghost', 8))
        b.append(text(34, y + 19, when, 't-note'))
        b.append(text(220, y + 19, what, 't-tick'))
        b.append(text(220, y + 36, why, 't-muted'))
        y += 52
    b.append(text(20, y + 16, 'Cheap checks run often; expensive ones run rarely. The test sets used for decisions must never be trained on.', 't-muted'))
    return svg(W, y + 28, 'A training pipeline of pretraining, SFT, preference tuning and release, with four kinds of evaluation: held-out loss every few hundred steps, a fast benchmark suite at every checkpoint, full benchmarks and a regression suite after each stage, and human or judge comparisons plus safety tests before release.', b)


F['ch3_pipeline'] = pipeline()
F['ch3_elo_update'] = elo_update()

for key, fn in [('ch3_window', window), ('ch3_ppl_bars', ppl_bars), ('ch3_ppl_vs_use', ppl_vs_use), ('ch3_mc_scoring', mc_scoring),
                ('ch3_mmlu_results', mmlu_results), ('ch3_position', position), ('ch3_bootstrap', bootstrap), ('ch3_gsm8k', gsm8k),
                ('ch3_passk', passk), ('ch3_ngram', ngram), ('ch3_arena_fit', arena_fit), ('ch3_votes_ci', votes_ci),
                ('ch3_judge', judge), ('ch3_rewardbench', rewardbench), ('ch3_bestofn', bestofn)]:
    try:
        F[key] = fn()
    except (FileNotFoundError, KeyError) as e:
        print(f'skip {key}: {e!r}')
json.dump(F, open('results/figs_ch3.json', 'w'))
print(f'{len(F)} figures:', ', '.join(F))
