"""Figures for Chapter 1 (from next word to assistant). Writes results/figs_ch1.json (key -> svg).
Colours come only from CSS classes (box, box-1..4, s1..s4, edge, path ...), so every figure follows the site's themes.
Text widths: about 7.2 px per character for t-tick / t-muted (12 px), 7.8 for t-note (13 px), 9 px for t-title; 17 px between lines.
Every number drawn here is read from a results/ch1_*.json file written by a script, or quoted from a cited paper."""
import json, math
from figlib import svg, box, arrow, esc
import figlib


def text(x, y, s, cls='t-note', anchor='start', extra=''):
    """figlib.text, with font ligatures off for monospace token text (so <| is not drawn as a triangle)."""
    if 't-mono' in cls:
        extra += ' style="font-variant-ligatures:none"'
    return figlib.text(x, y, s, cls, anchor, extra)
from agentfig import line, hbars
from alammar import frame, brace

R = lambda n: json.load(open(f'results/{n}.json'))
import os
NT, BVI, TS, KL, SFT = (R('ch1_nexttoken'), R('ch1_base_vs_instruct'), R('ch1_token_shift'), R('ch1_kl_example'), R('ch1_sft_tiny'))
EV = R('ch1_eval_preview') if os.path.exists('results/ch1_eval_preview.json') else None
F = {}


def para(x, y, lines, cls='t-muted', anchor='start', lh=17):
    return [text(x, y + i * lh, l, cls, anchor) for i, l in enumerate(lines)]


def show(s):
    """Make a token visible: spaces as a dot, newlines as an arrow."""
    return s.replace('\n', '↵').replace(' ', '·')


def tokbox(x, y, s, cls='box', h=28, tcls='t-mono', pad=12, cw=7.8):
    w = max(26, len(s) * cw + pad)
    return [box(x, y, w, h, cls, 6), text(x + w / 2, y + h / 2 + 4.5, s, tcls, 'middle')], w


def bar(x, y, w, h, cls, tip=None, op=None):
    st = f' style="fill-opacity:{op:.2f}"' if op is not None else ''
    t = f'<title>{esc(tip)}</title>' if tip else ''
    return f'<g class="mark">{t}<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{max(1.5, w):.1f}" height="{h:.1f}" rx="3"{st}/></g>'


# ---------------------------------------------------------------- 1. the training pipeline map
def pipeline():
    W, cw, gap, x0 = 900, 166, 10, 15
    stages = [
        ('box-1', 'Pretraining', ['web pages, books,', 'code, papers'], ['predict the next', 'token, everywhere'],
         ['Qwen2.5: 18T tokens', 'Llama 3: 15.6T tokens'], ['knowledge, grammar,', 'skills']),
        ('box-1', 'Mid-training', ['curated high-quality', 'maths, code, long text'], ['same objective,', 'learning rate to 0'],
         ['OLMo 2 7B: 3 runs ×', '50B (3.7% of 4.05T)'], ['fills gaps, longer', 'context']),
        ('box-2', 'SFT', ['prompt + one good', 'answer, chat format'], ['next token, on the', 'answer tokens only'],
         ['Qwen2.5: 1M+ examples', 'InstructGPT: 13k prompts'], ['format, turn-taking,', 'when to stop']),
        ('box-3', 'Preference tuning', ['prompt + preferred', '+ rejected answer'], ['RLHF (reward model', '+ PPO), or DPO'],
         ['Qwen2.5 DPO: ~150k pairs', 'Llama 3: 6 rounds'], ['which answer people', 'like more']),
        ('box-4', 'RL, verifiable reward', ['problems with a', 'checkable answer'], ['reward 1 if the final', 'answer is right'],
         ['DeepSeek-R1-Zero: AIME', '15.6% → 77.9%'], ['longer, more careful', 'reasoning']),
    ]
    b = [text(W / 2, 22, 'One model, five training stages (each starts from the weights the previous stage left)', 't-title', 'middle')]
    labels = ['data', 'objective', 'scale (published)', 'what it changes']
    top = 46
    for i, (cls, name, *rows) in enumerate(stages):
        x = x0 + i * (cw + gap)
        b += [box(x, top, cw, 40, cls, 10), text(x + cw / 2, top + 25, name, 't-note', 'middle')]
        if i < 4:
            b.append(arrow(x + cw + 1, top + 20, x + cw + gap - 1, top + 20))
        for k, lines in enumerate(rows):
            yy = top + 66 + k * 70
            b.append(box(x, yy - 16, cw, 60, 'box-ghost', 8))
            b.append(text(x + 8, yy, labels[k], 't-label'))
            b += para(x + 8, yy + 17, lines, 't-tick')
    yb = top + 66 + 4 * 70 - 4
    b += brace(x0, x0 + 2 * cw + gap, yb, 'pre-training → the "base" model (Qwen2.5-0.5B)', 'edge', False, 't-note')
    b += brace(x0 + 2 * (cw + gap), x0 + 5 * cw + 4 * gap, yb, 'post-training → the "instruct" model (Qwen2.5-0.5B-Instruct)', 'edge', False, 't-note')
    b.append(text(W / 2, yb + 48, 'compute: InstructGPT SFT 4.9 and PPO 60 petaflop/s-days, against 3,640 for pretraining GPT-3', 't-muted', 'middle'))
    return svg(W, yb + 62, 'The training pipeline: pretraining and mid-training produce a base model; SFT, preference tuning and RL with verifiable rewards turn it into an assistant. For each stage: data, objective, published scale, and what it changes.', b)


F['ch1_pipeline'] = pipeline()


# ---------------------------------------------------------------- 2. text -> tokens -> ids
def tokens_fig():
    b = [text(20, 24, 'text', 't-label'), text(110, 24, NT['sentence'], 't-mono')]
    b.append(text(20, 74, 'tokens', 't-label'))
    x = 110
    xs = []
    for i, p in enumerate(NT['pieces']):
        bx, w = tokbox(x, 54, show(p), f'box-{1 + i % 4}')
        b += bx
        xs.append(x + w / 2)
        b.append(text(x + w / 2, 110, str(NT['ids'][i]), 't-val', 'middle'))
        b.append(line(x + w / 2, 84, x + w / 2, 96, 'edge-dim'))
        x += w + 8
    b.append(text(20, 114, 'ids', 't-label'))
    b.append(text(110, 146, f'7 tokens. The model never sees letters, only these ids: rows of a table of {NT["vocab_rows"]:,} entries.', 't-muted'))
    b.append(text(110, 164, '"·" marks a space: most tokens carry their leading space, so " Paris" and "Paris" are different tokens.', 't-muted'))
    return svg(760, 178, 'The sentence The capital of France is Paris split into 7 tokens by the Qwen2.5 tokenizer, with their ids.', b)


F['ch1_tokens'] = tokens_fig()


# ---------------------------------------------------------------- 3. what a model is: a function from ids to a distribution
def model_fig():
    b = []
    b += [box(20, 60, 170, 70, 'box'), text(105, 88, 'context (token ids)', 't-note', 'middle'),
          text(105, 108, '785 6722 315 9625 374', 't-mono', 'middle')]
    b += [box(240, 40, 220, 110, 'box-2'), text(350, 72, 'Qwen2.5-0.5B', 't-title', 'middle'),
          text(350, 94, f'{NT["params"]:,} weights', 't-note', 'middle'),
          text(350, 113, f'{NT["layers"]} transformer layers', 't-tick', 'middle'),
          text(350, 130, f'hidden size {NT["hidden"]}', 't-tick', 'middle')]
    b += [arrow(190, 95, 240, 95), arrow(460, 95, 505, 95)]
    b.append(text(612, 30, 'one probability per vocabulary entry', 't-muted', 'middle'))
    for k, c in enumerate(NT['top5']):
        y = 44 + k * 22
        b.append(text(580, y + 13, show(c['tok']), 't-tick', 'end'))
        b.append(bar(586, y + 2, 180 * c['p'] / 0.35, 14, 's1', f'{c["tok"]}: {c["p"]:.4f}'))
        b.append(text(590 + 180 * c['p'] / 0.35, y + 13, f'{c["p"]:.3f}', 't-val'))
    b.append(text(580, 167, f'... {NT["vocab_rows"] - 5:,} more', 't-muted', 'end'))
    b.append(text(240, 180, 'Training = changing the weights so that the right next token gets a higher probability.', 't-muted'))
    return svg(800, 196, 'A language model as a function: token ids go in, the weights compute, and one probability for every possible next token comes out.', b)


F['ch1_model'] = model_fig()


# ---------------------------------------------------------------- 4. next-token prediction, frame by frame
def frames_fig():
    rows = NT['rows']
    W, fw, fh = 860, 410, 112
    b = []
    for k, r in enumerate(rows):
        col, rw = k % 2, k // 2
        x, y = 15 + col * (fw + 20), 12 + rw * (fh + 14)
        b += frame(x, y, fw, fh, k + 1, f'context: "{r["context"]}"', 'box-ghost')
        b.append(text(x + 18, y + 54, 'actual next', 't-muted'))
        b.append(text(x + 110, y + 54, show(r['next']), 't-mono'))
        b.append(bar(x + 200, y + 42, 190 * r['p'], 15, 's2', f'P({r["next"]}) = {r["p"]:.4f}'))
        b.append(text(x + 204 + 190 * r['p'], y + 54, f'{r["p"]:.4f}', 't-val'))
        b.append(text(x + 18, y + 82, "model's top-1", 't-muted'))
        b.append(text(x + 110, y + 82, show(r['top1']), 't-mono'))
        b.append(bar(x + 200, y + 70, 190 * r['top1_p'], 15, 's1', f'top-1 {r["top1"]}: {r["top1_p"]:.3f}'))
        b.append(text(x + 204 + 190 * r['top1_p'], y + 82, f'{r["top1_p"]:.3f}', 't-val'))
        b.append(text(x + 18, y + 102, f'rank of the actual token: {r["rank"]}   ·   −log P = {r["nll"]:.3f}', 't-tick'))
    H = 12 + 3 * (fh + 14)
    return svg(W, H, 'Six frames: at each position Qwen2.5-0.5B sees the context and gives a probability to the actual next token; its own top guess is shown below.', b)


F['ch1_frames'] = frames_fig()


# ---------------------------------------------------------------- 5. chain rule: multiply the probabilities, add the logs
def chain_fig():
    rows = NT['rows']
    b = [text(20, 24, 'P(capital of France is Paris . | The) = product of six next-token probabilities', 't-title')]
    x = 20
    for k, r in enumerate(rows):
        b += [box(x, 44, 100, 44, f'box-{1 + k % 4}', 8), text(x + 50, 62, show(r['next']), 't-mono', 'middle'),
              text(x + 50, 80, f'{r["p"]:.4f}', 't-val', 'middle')]
        b.append(text(x + 50, 108, f'log = {math.log(r["p"]):.3f}', 't-tick', 'middle'))
        if k < 5:
            b.append(text(x + 110, 72, '×', 't-title', 'middle'))
        x += 120
    b.append(text(20, 140, f'product = {math.exp(NT["logp_sum"]):.3e}', 't-note'))
    b.append(text(300, 140, f'sum of logs = {NT["logp_sum"]:.3f}', 't-note'))
    b.append(text(520, 140, f'loss = {-NT["logp_sum"]:.3f} / 6 = {NT["loss"]:.3f}', 't-note'))
    b.append(text(20, 164, 'Tiny numbers multiply into tinier ones; adding logs gives the same information without underflow.', 't-muted'))
    b.append(text(20, 182, f'The single unlikely token " capital" (P = {rows[0]["p"]:.4f}) contributes {rows[0]["nll"]:.2f} of the {-NT["logp_sum"]:.2f} total.', 't-muted'))
    return svg(740, 196, 'The chain rule: the probability of the whole sentence is the product of the six next-token probabilities; its log is the sum of their logs.', b)


F['ch1_chain'] = chain_fig()


# ---------------------------------------------------------------- 6. softmax by hand
def softmax_fig():
    c = NT['top5']
    b = [text(20, 22, 'After "The capital of France is": logits → softmax', 't-title')]
    hdr = [(110, 'logit z'), (250, 'e^(z − max)'), (400, 'softmax (all 151,936)'), (640, 'softmax (just these 5)')]
    for x, h in hdr:
        b.append(text(x, 50, h, 't-label'))
    for k, r in enumerate(c):
        y = 62 + k * 28
        e = math.exp(r['logit'] - c[0]['logit'])
        b.append(text(100, y + 14, show(r['tok']), 't-mono', 'end'))
        b.append(bar(110, y + 2, 85 * (r['logit'] - 15.5) / 2.5, 16, 's4', f'logit {r["logit"]:.3f}'))
        b.append(text(110 + 85 * (r['logit'] - 15.5) / 2.5 + 5, y + 15, f'{r["logit"]:.2f}', 't-val'))
        b.append(bar(250, y + 2, 90 * e, 16, 's3', f'exp {e:.4f}'))
        b.append(text(255 + 90 * e, y + 15, f'{e:.3f}', 't-val'))
        b.append(bar(400, y + 2, 300 * r['p'], 16, 's1', f'P {r["p"]:.4f}'))
        b.append(text(405 + 300 * r['p'], y + 15, f'{r["p"]:.3f}', 't-val'))
        b.append(bar(640, y + 2, 160 * r['p5'], 16, 's2', f'P among 5: {r["p5"]:.4f}'))
        b.append(text(645 + 160 * r['p5'], y + 15, f'{r["p5"]:.3f}', 't-val'))
    b.append(text(20, 222, 'Logit bars start at 15.5 to show the gaps. A logit 1.0 higher means e ≈ 2.72 times more probable.', 't-muted'))
    b.append(text(20, 240, f'These five take {sum(r["p"] for r in c):.0%} of the probability; the other 151,931 tokens share the remaining {1 - sum(r["p"] for r in c):.0%}.', 't-muted'))
    return svg(870, 254, 'Softmax worked by hand on the five most likely next tokens after The capital of France is: raw logits, their exponentials, and the normalised probabilities.', b)


F['ch1_softmax'] = softmax_fig()


# ---------------------------------------------------------------- 7. the -log p curve with the six real points
def nll_fig():
    W, H, L, T, PW, PH = 720, 300, 70, 30, 560, 210
    X = lambda p: L + p * PW
    Y = lambda v: T + PH - v / 10 * PH
    b = []
    for v in [0, 2, 4, 6, 8, 10]:
        b += [f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + PW}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, str(v), 't-tick', 'end')]
    for p in [0, 0.25, 0.5, 0.75, 1.0]:
        b.append(text(X(p), T + PH + 20, f'{p:g}', 't-tick', 'middle'))
    pts = [(math.exp(-v), v) for v in [10 - i * 0.02 for i in range(501)]]           # from P = e^-10 up to P = 1
    b.append('<polyline class="l1" points="' + ' '.join(f'{X(p):.1f},{Y(v):.1f}' for p, v in pts) + '"/>')
    for k, r in enumerate(NT['rows']):
        b.append(f'<g class="mark"><title>{esc(r["next"])}: P={r["p"]:.4f}, -log P={r["nll"]:.3f}</title>'
                 f'<circle class="s2 ring" cx="{X(r["p"]):.1f}" cy="{Y(r["nll"]):.1f}" r="6"/></g>')
        dx, dy = (10, -8) if k != 1 else (10, 16)
        b.append(text(X(r['p']) + dx, Y(r['nll']) + dy, f'{show(r["next"])} ({r["nll"]:.2f})', 't-tick'))
    b.append(text(L + PW / 2, H - 12, 'probability the model gave to the actual next token, P', 't-tick', 'middle'))
    b.append(text(L - 50, T - 12, 'loss = −log P (nats)', 't-tick'))
    return svg(W, H, 'The cross-entropy loss minus log P as a curve, with the six real tokens of the example sentence marked on it.', b)


F['ch1_nll'] = nll_fig()


# ---------------------------------------------------------------- 8. compute per stage (InstructGPT, Section 5.1)
def compute_fig():
    items = [('GPT-3 pretraining', 3640.0), ('InstructGPT PPO-ptx', 60.0), ('InstructGPT SFT', 4.9)]
    b = [text(20, 24, 'Training compute for the 175B models, petaflop/s-days (log scale)', 't-title')]
    L, PW = 190, 470
    X = lambda v: L + math.log10(v) / math.log10(5000) * PW
    for v in [1, 10, 100, 1000]:
        b += [f'<line class="grid" x1="{X(v):.1f}" y1="40" x2="{X(v):.1f}" y2="150"/>', text(X(v), 166, f'{v:,}', 't-tick', 'middle')]
    for k, (lab, v) in enumerate(items):
        y = 48 + k * 34
        b.append(text(L - 8, y + 16, lab, 't-tick', 'end'))
        b.append(bar(L, y + 3, X(v) - L, 20, ['s1', 's3', 's2'][k], f'{lab}: {v:,}'))
        b.append(text(X(v) + 6, y + 18, f'{v:,.0f}' if v >= 10 else f'{v}', 't-val'))
    b.append(text(20, 192, 'Source: Ouyang et al. (2022), Section 5.1. SFT used about 0.13% of the pretraining compute; RLHF about 1.6%.', 't-muted'))
    return svg(720, 206, 'Compute bars on a log scale: GPT-3 pretraining 3,640 petaflop/s-days, InstructGPT PPO-ptx 60, InstructGPT SFT 4.9.', b)


F['ch1_compute'] = compute_fig()


# ---------------------------------------------------------------- 9. the chat template, token by token
def template_fig():
    toks = BVI['template_tokens']
    b = [text(20, 22, f'"What is the capital of France?" as the instruct model reads it: {len(toks)} tokens', 't-title')]
    x, y = 20, 40
    for t in toks:
        s = show(t['tok'])
        cls = 'box-on' if t['special'] else ('box-ghost' if t['tok'] in ('system', 'user', 'assistant', '\n') else 'box')
        w = max(22, len(s) * 7.8 + 10)
        if t['tok'] == '\n':
            w = 22
        if x + w > 800:
            x, y = 20, y + 38
        b += [box(x, y, w, 28, cls, 6), text(x + w / 2, y + 18.5, s, 't-mono', 'middle')]
        x += w + 5
        if t['tok'] == '\n':
            x, y = 20, y + 38
    y += 6
    b += [box(20, y, 18, 14, 'box-on', 3), text(46, y + 12, 'special token: one id, never produced by ordinary text (<|im_start|> = 151644, <|im_end|> = 151645)', 't-tick')]
    b += [box(20, y + 22, 18, 14, 'box-ghost', 3), text(46, y + 34, 'role name and line break', 't-tick'),
          box(250, y + 22, 18, 14, 'box', 3), text(276, y + 34, 'ordinary text (the default system prompt, then the question)', 't-tick')]
    b.append(text(20, y + 62, 'The template ends with "<|im_start|>assistant↵": the model\'s job is to write the next turn and close it with <|im_end|>.', 't-muted'))
    return svg(820, y + 76, 'The Qwen2.5 chat template laid out token by token: special tokens im_start and im_end mark each turn, followed by the role name system, user or assistant.', b)


F['ch1_template'] = template_fig()


# ---------------------------------------------------------------- 10. base vs instruct, side by side
def side_fig():
    W = 900
    picks = [(0, 'What is the capital of France?'), (4, 'Who are you?')]
    cols = [('base, raw text', 'box'), ('base, chat template', 'box-ghost'), ('instruct, chat template', 'box-2')]
    cw = 286
    b = []
    y = 10
    for pi, q in picks:
        r = BVI['results'][pi]
        b.append(text(15, y + 18, f'prompt: "{q}"', 't-title'))
        y += 30
        for c, (lab, cls) in enumerate(cols):
            x = 15 + c * (cw + 8)
            o = r[lab]
            txt = o['text'].replace('<|endoftext|>', ' [END]').replace('<|im_end|>', ' [END]')
            words, lines, cur = txt.replace('\n', ' ↵ ').split(' '), [], ''
            for w_ in words:
                if len(cur) + len(w_) + 1 > 36:
                    lines.append(cur)
                    cur = w_
                else:
                    cur = (cur + ' ' + w_).strip()
            lines.append(cur)
            lines = lines[:6] + (['...'] if len(lines) > 6 else [])
            h = 72 + 17 * len(lines)
            b.append(box(x, y, cw, h, cls, 10))
            b.append(text(x + 12, y + 22, lab, 't-note'))
            badge = f'{o["n_tokens"]} tokens, ' + ('stopped by itself' if o['stopped'] else f'cut at {BVI["max_new_tokens"]}')
            b.append(text(x + 12, y + 40, badge, 't-muted' if o['stopped'] else 't-bad'))
            b += para(x + 12, y + 64, lines, 't-tick')
        y += 72 + 17 * 7 + 20
    return svg(W, y, 'The same two prompts given to the base model as raw text, to the base model inside the chat template, and to the instruct model. Greedy decoding.', b)


F['ch1_side'] = side_fig()


# ---------------------------------------------------------------- 11. the token-shift method (URIAL, Figure 2), drawn with our numbers
def shift_method():
    ex = next(p for p in TS['top_kl'] if p['tok'] != '<|im_end|>')
    b = [text(20, 24, 'Measuring the shift at one position t', 't-title')]
    b += [box(20, 44, 240, 64, 'box'), text(140, 68, 'context = prompt + o₁ … o₍ₜ₋₁₎', 't-note', 'middle'),
          text(140, 90, '(the instruct model\'s own answer)', 't-muted', 'middle')]
    b += [box(320, 34, 200, 40, 'box-2'), text(420, 59, 'instruct: greedy oₜ', 't-note', 'middle'),
          box(320, 84, 200, 40, 'box-1'), text(420, 109, 'base: rank of oₜ', 't-note', 'middle')]
    b += [arrow(260, 70, 320, 54), arrow(260, 82, 320, 104)]
    cls = [('unshifted', 'rank 1: both models agree', 's1'), ('marginal', 'rank 2 or 3', 's3'), ('shifted', 'rank > 3', 's4')]
    for k, (n, d, c) in enumerate(cls):
        y = 40 + k * 30
        b += [bar(580, y, 20, 18, c), text(608, y + 14, f'{n}: {d}', 't-tick')]
    b += [arrow(520, 104, 570, 74)]
    b.append(text(20, 150, 'Also at every position: KL(P_instruct ‖ P_base), how different the two whole distributions are.', 't-muted'))
    return svg(820, 164, 'The token-shift method: the instruct model writes an answer greedily; at every position the base model sees the same context, and the rank it gives to the instruct token classifies the position as unshifted, marginal or shifted.', b)


F['ch1_shift_method'] = shift_method()


# ---------------------------------------------------------------- 12. stacked bars: our result vs URIAL's three pairs
def shift_bars():
    f, fb = TS['frac'], TS['fracB']
    rows = [(f'Qwen2.5-0.5B → Instruct (ours, {TS["n_prompts"]} prompts, {TS["n_tokens"]:,} tokens)', f['unshifted'], f['marginal'], f['shifted']),
            ('Qwen2.5-0.5B → Instruct (ours, "Question:/Answer:" format)', fb['unshifted'], fb['top3'] - fb['unshifted'], 1 - fb['top3']),
            ('Llama-2-7b → Llama-2-7b-chat (URIAL)', .777, .145, .078),
            ('Llama-2-7b → Vicuna-7b-v1.5 (URIAL)', .824, .128, .048),
            ('Mistral-7b → Mistral-7b-instruct (URIAL)', .822, .125, .052)]
    L, PW = 20, 560
    b = []
    for k, (lab, u, m, s) in enumerate(rows):
        y = 18 + k * 56
        b.append(text(L, y + 12, lab, 't-tick t-strong' if k < 2 else 't-tick'))
        x = L
        for v, c, n in [(u, 's1', 'unshifted'), (m, 's3', 'marginal'), (s, 's4', 'shifted')]:
            b.append(bar(x, y + 20, PW * v - 1, 20, c, f'{n}: {v:.1%}'))
            x += PW * v
        b.append(text(L + PW + 10, y + 35, f'{u:.1%} · {m:.1%} · {s:.1%}', 't-val'))
    y = 18 + len(rows) * 56
    for k, (n, c) in enumerate([('unshifted (base top-1)', 's1'), ('marginal (base rank 2-3)', 's3'), ('shifted (base rank > 3)', 's4')]):
        b += [bar(L + k * 210, y, 14, 14, c), text(L + k * 210 + 20, y + 12, n, 't-tick')]
    return svg(820, y + 24, 'Stacked bars of unshifted, marginal and shifted token fractions: our Qwen2.5-0.5B measurement next to the three 7B pairs from the URIAL paper.', b)


F['ch1_shift_bars'] = shift_bars()


# ---------------------------------------------------------------- 13. one answer, coloured token by token
def strip_fig():
    pp = TS['per_prompt']
    # choose the answer used in the text: "Give me three tips for sleeping better."
    target = 'Write a two-sentence story about a lost key.'
    toks = [p for p in TS['strip_all'] if p['prompt_text'] == target][:64] if 'strip_all' in TS else TS['strip']
    b = [text(15, 22, f'Instruct answer to "{target}", coloured by how the base model ranked each token', 't-title')]
    cls = {'unshifted': 'box', 'marginal': 'box-3', 'shifted': 'box-4'}
    x, y = 15, 38
    for t in toks:
        s = show(t['tok']).replace('↵', '↵')
        w = max(14, len(s) * 7.8 + 8)
        if x + w > 845:
            x, y = 15, y + 34
        b.append(f'<g class="mark"><title>{esc(t["tok"])}: base rank {t["rank"]}, KL {t["kl"]:.2f}, base top-1 {esc(t["base_top"])}</title>'
                 f'{box(x, y, w, 26, cls[t["cls"]], 5)}</g>')
        b.append(text(x + w / 2, y + 17.5, s, 't-mono', 'middle'))
        x += w + 4
    y += 46
    for k, (n, c) in enumerate([('unshifted', 'box'), ('marginal', 'box-3'), ('shifted', 'box-4')]):
        b += [box(15 + k * 140, y, 18, 14, c, 3), text(41 + k * 140, y + 12, n, 't-tick')]
    b.append(text(440, y + 12, 'first 64 tokens; hover a token for its base rank and KL', 't-muted'))
    return svg(860, y + 26, 'The first 64 tokens of one instruct answer, each coloured as unshifted, marginal or shifted according to the base model rank.', b)


# ---------------------------------------------------------------- 14. shift and KL by position in the answer
def bypos_fig():
    rows = TS['by_pos']
    b = [text(20, 22, 'Shift is largest at the start of the answer and fades', 't-title')]
    L = 150
    b.append(text(L, 48, 'shifted tokens (%)', 't-label'))
    b.append(text(L + 330, 48, 'mean KL per token (nats)', 't-label'))
    for k, r in enumerate(rows):
        y = 60 + k * 30
        b.append(text(L - 10, y + 14, f'tokens {r["range"]} (n={r["n"]:,})', 't-tick', 'end'))
        b.append(bar(L, y + 2, 2400 * r['shifted'], 16, 's4', f'{r["shifted"]:.1%}'))
        b.append(text(L + 2400 * r['shifted'] + 5, y + 15, f'{r["shifted"]:.1%}', 't-val'))
        b.append(bar(L + 330, y + 2, 300 * r['kl'], 16, 's2', f'{r["kl"]:.3f}'))
        b.append(text(L + 330 + 300 * r['kl'] + 5, y + 15, f'{r["kl"]:.3f}', 't-val'))
    y = 60 + len(rows) * 30 + 12
    b.append(text(20, y, 'Includes the end-of-turn token <|im_end|>, which is always shifted and sits late in long answers.', 't-muted'))
    return svg(720, y + 14, 'Two bar charts by position in the answer: the share of shifted tokens and the mean KL divergence both fall from the first tokens to later ones.', b)


F['ch1_bypos'] = bypos_fig()


# ---------------------------------------------------------------- 15. KL worked example: style vs fact
def kl_fig():
    W = 880
    b = []
    for c, case in enumerate(KL):
        x0 = 15 + c * 440
        title = 'style: first token of the tips answer' if case['name'] == 'style' else 'fact: after "The capital of Australia is"'
        b.append(text(x0, 22, title, 't-title'))
        b.append(text(x0 + 110, 44, 'P_instruct', 't-label'))
        b.append(text(x0 + 275, 44, 'P_base', 't-label'))
        toks = case['rows'][:3]
        for k, r in enumerate(toks):
            y = 54 + k * 46
            b.append(text(x0 + 100, y + 14, show(r['tok']), 't-mono', 'end'))
            b.append(bar(x0 + 110, y + 2, 100 * r['p'], 15, 's2', f'P_instruct {r["p"]:.4f}'))
            b.append(text(x0 + 114 + 100 * r['p'], y + 14, f'{r["p"]:.3f}', 't-val'))
            b.append(bar(x0 + 275, y + 2, 100 * r['q'], 15, 's1', f'P_base {r["q"]:.4f}'))
            b.append(text(x0 + 279 + 100 * r['q'], y + 14, f'{r["q"]:.3f}', 't-val'))
            b.append(text(x0 + 110, y + 34, f'term p·log(p/q) = {r["term"]:+.3f}', 't-tick'))
        b.append(text(x0, 54 + 3 * 46 + 16, f'KL = {case["kl"]:.3f} nats', 't-note'))
    return svg(W, 54 + 3 * 46 + 30, 'KL divergence worked at two real positions: a style position, where the two models disagree strongly, and a fact position, where they agree.', b)


F['ch1_kl'] = kl_fig()


# ---------------------------------------------------------------- 16. SFT label masking
def mask_fig():
    lines = open('results/ch1_sft_tiny_stdout.txt').read().split('\n')
    toks = []
    for l in lines[1:]:
        if not l.startswith('  '):
            break
        s = l.strip()
        lab = s.rsplit('label', 1)[1].strip()
        toks.append((eval(s.rsplit('label', 1)[0].strip()), lab != '-100'))
    b = [text(15, 22, 'One SFT example: which tokens count in the loss', 't-title')]
    x, y = 15, 38
    for s, on in toks:
        s2 = show(s)
        w = max(22, len(s2) * 7.8 + 10)
        if s == '\n':
            w = 22
        if x + w > 840:
            x, y = 15, y + 50
        b += [box(x, y, w, 26, 'box-2' if on else 'box-ghost', 5), text(x + w / 2, y + 17.5, s2, 't-mono', 'middle')]
        b.append(text(x + w / 2, y + 40, 'id' if on else '−100', 't-tick' if on else 't-muted', 'middle'))
        x += w + 4
    y += 66
    n_on = sum(on for _, on in toks)
    b += [box(15, y, 18, 14, 'box-ghost', 3), text(41, y + 12, f'prompt: label −100, ignored by the loss ({len(toks) - n_on} tokens)', 't-tick'),
          box(430, y, 18, 14, 'box-2', 3), text(456, y + 12, f'answer + <|im_end|>: trained on ({n_on} tokens)', 't-tick')]
    return svg(860, y + 26, 'An SFT training example token by token: the system prompt, user turn and assistant header have label minus 100 and are ignored; only the answer and the end-of-turn token are trained.', b)


F['ch1_mask'] = mask_fig()


# ---------------------------------------------------------------- 17. one training step
def step_fig():
    b = []
    steps = [('box-1', '1 · forward', ['run the batch through', 'the model → logits']),
             ('box-2', '2 · loss', ['cross-entropy on the', 'unmasked labels']),
             ('box-3', '3 · backward', ['a gradient for each', f'of the {NT["params"] / 1e6:.0f}M weights']),
             ('box-4', '4 · update', ['AdamW moves each', 'weight a little'])]
    for k, (c, t, ls) in enumerate(steps):
        x = 15 + k * 200
        b += [box(x, 30, 170, 76, c, 10), text(x + 85, 54, t, 't-note', 'middle')] + para(x + 85, 76, ls, 't-tick', 'middle')
        if k < 3:
            b.append(arrow(x + 170, 68, x + 200, 68))
    b.append(f'<path class="path" d="M700,106 L700,136 L100,136 L100,110" marker-end="url(#ah-on)"/>')
    b.append(text(400, 156, 'zero the gradients, take the next batch, repeat', 't-muted', 'middle'))
    b.append(text(15, 18, 'out = model(input_ids, labels=labels)   →   out.loss.backward()   →   opt.step(); opt.zero_grad()', 't-code'))
    return svg(820, 170, 'One training step in four parts: forward pass, loss, backward pass for gradients, and the optimizer update; then repeat.', b)


F['ch1_step'] = step_fig()


# ---------------------------------------------------------------- 18. the tiny SFT runs: losses and P(<|im_end|>), two learning rates
SFT4 = R('ch1_sft_tiny_lr1e-4')


def sft_fig():
    W, H, T, PW, PH = 900, 300, 40, 200, 190
    b = []

    def panel(L, title, ymax, yt, fmt):
        X = lambda st: L + st / 30 * PW
        Y = lambda v: T + PH - v / ymax * PH
        b.append(text(L, 20, title, 't-label'))
        for v in yt:
            b.extend([f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{L + PW}" y2="{Y(v):.1f}"/>', text(L - 8, Y(v) + 4, fmt(v), 't-tick', 'end')])
        for st in [0, 10, 20, 30]:
            b.append(text(X(st), T + PH + 18, str(st), 't-tick', 'middle'))
        b.append(text(L + PW / 2, T + PH + 40, 'training step', 't-tick', 'middle'))
        return X, Y

    for k, (run, lab) in enumerate([(SFT, 'lr 1e-5'), (SFT4, 'lr 1e-4')]):
        L = 50 + k * 310
        X, Y = panel(L, f'loss, {lab}', 3, [0, 1, 2, 3], lambda v: str(v))
        h = run['hist']
        tr = [(x['step'], x['train']) for x in h if x['train'] is not None]
        b.append('<polyline class="l1" points="' + ' '.join(f'{X(a):.1f},{Y(v):.1f}' for a, v in tr) + '"/>')
        b.append('<polyline class="l2" points="' + ' '.join(f'{X(x["step"]):.1f},{Y(x["held"]):.1f}' for x in h) + '"/>')
        b.append(text(X(30) + 4, Y(tr[-1][1]) + 4, f'train {tr[-1][1]:.2f}', 't-tick'))
        b.append(text(X(30) + 4, Y(h[-1]['held']) - 6, f'held-out {h[-1]["held"]:.2f}', 't-tick'))
    L = 680
    X, Y = panel(L, 'P(<|im_end|>), held-out', 1, [0, 0.5, 1], lambda v: f'{v:g}')
    for run, cls, lab in [(SFT, 'l3', '1e-5'), (SFT4, 'l2', '1e-4')]:
        h = run['hist']
        b.append(f'<polyline class="{cls}" points="' + ' '.join(f'{X(x["step"]):.1f},{Y(x["p_end"]):.1f}' for x in h) + '"/>')
        b.append(text(X(14), Y(h[-1]['p_end']) + (-10 if h[-1]['p_end'] < 0.5 else 20), f'lr {lab}: {h[-1]["p_end"]:.4f}', 't-tick'))
    return svg(W, H, 'Two tiny SFT runs on 8 examples. With learning rate 1e-5 the training loss falls slowly and the end-of-turn probability stays near zero; with 1e-4 the training loss reaches zero, the held-out loss rises, and the end-of-turn probability reaches almost 1.', b)


F['ch1_sft'] = sft_fig()


# ---------------------------------------------------------------- 19. evaluation preview
def eval_fig():
    s = EV['summary']
    b = [text(20, 22, f'{s["instruct, chat template"]["n"]} short questions, greedy, at most {EV["max_new_tokens"]} new tokens', 't-title')]
    b.append(text(250, 46, 'expected answer appears', 't-label'))
    b.append(text(520, 46, 'ended its turn by itself', 't-label'))
    for k, (name, v) in enumerate(s.items()):
        y = 56 + k * 32
        b.append(text(240, y + 15, name, 't-tick', 'end'))
        b.append(bar(250, y + 2, 220 * v['correct'] / v['n'], 18, 's1', f'{v["correct"]}/{v["n"]}'))
        b.append(text(255 + 220 * v['correct'] / v['n'], y + 16, f'{v["correct"]}/{v["n"]}', 't-val'))
        b.append(bar(520, y + 2, 220 * v['stopped'] / v['n'], 18, 's2', f'{v["stopped"]}/{v["n"]}'))
        b.append(text(525 + 220 * v['stopped'] / v['n'], y + 16, f'{v["stopped"]}/{v["n"]}', 't-val'))
    return svg(820, 56 + 3 * 32 + 10, 'A first evaluation: how often each model setting contains the expected answer, and how often it ends its turn by itself.', b)


if EV:
    F['ch1_eval'] = eval_fig()

if 'strip_all' in TS:
    F['ch1_strip'] = strip_fig()

json.dump(F, open('results/figs_ch1.json', 'w'))
print(f'{len(F)} figures:', ', '.join(F))
