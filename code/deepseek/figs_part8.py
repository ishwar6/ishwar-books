"""Figures for Part 8 (the specialised models). Writes results/figs_part8.json (key -> SVG string).
CSS classes only, so every figure follows the site's dark, dim and light themes."""
import json, os
from figlib import svg, text, box, arrow, esc, lines_chart
from common import RESULTS, save

R = lambda n: json.load(open(os.path.join(RESULTS, f'{n}.json')))
F = {}
W = 780


def label_box(x, y, w, h, lines, cls='box', tcls='t-tick', first='t-note'):
    b = [box(x, y, w, h, cls)]
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * 9 + 4
    for i, s in enumerate(lines):
        b.append(text(x + w / 2, y0 + i * 18, s, first if i == 0 and first else tcls, 'middle'))
    return b


def line(x1, y1, x2, y2, cls='edge', dash=False):
    d = ' stroke-dasharray="5 4"' if dash else ''
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"{d}/>'


def dot(x, y, cls='s1', r=6):
    return f'<circle class="{cls} ring" cx="{x:.1f}" cy="{y:.1f}" r="{r}"/>'


# 1. Timeline of the specialised families -------------------------------------------------------------------
def timeline():
    lanes = [('Code', [(2023.83, 'Coder 1.3B-33B', -1), (2024.07, 'Coder-v1.5', 1), (2024.45, 'Coder-V2', -1),
                       (2024.68, 'merged into V2.5', 1)]),
             ('Maths', [(2024.10, 'DeepSeekMath 7B + GRPO', -1), (2025.91, 'Math-V2', -1)]),
             ('Proofs', [(2024.39, 'Prover-V1', -1), (2024.62, 'V1.5', 1), (2025.33, 'Prover-V2', -1)]),
             ('Vision', [(2024.18, 'VL', -1), (2024.80, 'Janus', -1), (2024.87, 'JanusFlow', 1), (2024.95, 'VL2', -1),
                         (2025.08, 'Janus-Pro', 2), (2025.80, 'OCR', -1), (2026.08, 'OCR 2', 1),
                         (2026.66, 'V4-Flash-Vision-Exp', -1)]),
             ('Other', [(2024.51, 'ESFT', -1), (2025.26, 'GRM / SPCT', 1), (2026.49, 'DSpark', -1)])]
    x0, x1, t0, t1 = 96, 760, 2023.75, 2026.85
    X = lambda t: x0 + (t - t0) / (t1 - t0) * (x1 - x0)
    b = [text(20, 24, 'Specialised releases by family (dates: Hugging Face repo creation or arXiv)', 't-title')]
    top, lh = 84, 66
    for yr in (2024, 2025, 2026):
        b += [line(X(yr), top - 20, X(yr), top + lh * len(lanes) - 20, 'grid'), text(X(yr), top - 26, str(yr), 't-tick', 'middle')]
    for i, (name, items) in enumerate(lanes):
        y = top + i * lh + 12
        b += [text(20, y + 4, name, 't-note'), line(x0, y, x1, y, 'axis')]
        for t, lab, side in items:
            b.append(dot(X(t), y, f's{(i % 3) + 1}', 5))
            b.append(text(X(t), y + {-1: -11, 1: 21, 2: 36}[side], lab, 't-tick', 'middle'))
    return svg(W, top + lh * len(lanes) - 6, 'A timeline of DeepSeek specialised releases from 2023 to 2026 in five lanes: code, maths, proofs, vision and other.', b)


F['timeline'] = timeline()


# 2. Repository-level ordering -------------------------------------------------------------------------------
def repo_order():
    b = [text(20, 24, 'One repository, ordered so each file comes after the files it imports', 't-title')]
    files = [('utils.py', 60, 70), ('model.py', 60, 160), ('data.py', 250, 70), ('train.py', 250, 160)]
    pos = {}
    for n, x, y in files:
        b += label_box(x, y, 140, 46, [n], 'box')
        pos[n] = (x, y)
    b += [arrow(130, 116, 130, 160), arrow(200, 93, 250, 93), arrow(320, 116, 320, 160), arrow(200, 183, 250, 183)]
    b.append(text(60, 236, 'Arrows: "is imported by". Found with regular expressions', 't-tick'))
    b.append(text(60, 254, 'for import (Python), using (C#), include (C).', 't-tick'))
    b.append(arrow(410, 140, 460, 140, True))
    order = ['# utils.py', '...code...', '# data.py', '...code...', '# model.py', '...code...', '# train.py', '...code...']
    b.append(box(470, 50, 290, 200, 'code-bg', 8))
    b.append(text(484, 72, 'one training sample', 't-label'))
    for i, s in enumerate(order):
        b.append(text(490, 96 + i * 19, s, 't-code-key' if s.startswith('#') else 't-code'))
    return svg(W, 270, 'Four files with import arrows are sorted so that every file comes after the files it uses, then joined into one training sample with a path comment on top of each file.', b)


F['repo_order'] = repo_order()


# 3. Fill-in-the-middle --------------------------------------------------------------------------------------
def fim():
    b = [text(20, 24, 'Fill-in-the-middle: cut a file in three, then move the middle to the end', 't-title')]
    parts = [('prefix', 'def area(r):', 'box-1'), ('middle', '    return 3.14159 * r * r', 'box-2'), ('suffix', 'print(area(2))', 'box-3')]
    b.append(text(20, 56, 'The file as written', 't-label'))
    for i, (lab, code, cls) in enumerate(parts):
        y = 66 + i * 32
        b += [box(20, y, 330, 28, cls, 6), text(30, y + 19, code, 't-code'), text(360, y + 19, lab, 't-tick')]
    b.append(arrow(420, 112, 460, 112, True))
    b.append(text(470, 56, 'The training sample (PSM order)', 't-label'))
    seq = [('FIM start', 'box'), ('prefix', 'box-1'), ('hole', 'box'), ('suffix', 'box-3'), ('FIM end', 'box'), ('middle', 'box-2')]
    for i, (lab, cls) in enumerate(seq):
        y = 66 + i * 26
        b += [box(470, y, 150, 22, cls, 5), text(545, y + 16, lab, 't-tick', 'middle')]
    b.append(text(636, 145, 'the model learns', 't-tick'))
    b.append(text(636, 163, 'to write the middle', 't-tick'))
    b.append(text(636, 181, 'after seeing both sides', 't-tick'))
    b.append(text(20, 240, 'In an editor: the code above and below the cursor become prefix and suffix; the model writes the middle.', 't-tick'))
    return svg(W, 256, 'A small Python file is cut into a prefix, a middle and a suffix; the training sample puts the prefix, then the suffix, then the middle, separated by special tokens.', b)


F['fim'] = fim()


# 4. The DeepSeekMath corpus loop ----------------------------------------------------------------------------
def math_loop():
    b = [text(20, 24, 'Building a 120B-token maths corpus from Common Crawl: four rounds of the same loop', 't-title')]
    steps = [(30, 50, ['Seed: 500K pages', 'from OpenWebMath']), (290, 50, ['Train fastText', 'math vs not math']),
             (550, 50, ['Score 40B pages', 'keep the top ones']), (550, 170, ['Find maths domains', '>10% pages collected']),
             (290, 170, ['People mark maths', 'URL paths']), (30, 170, ['Add missed pages', 'to the seed'])]
    for x, y, ls in steps:
        b += label_box(x, y, 200, 56, ls, 'box')
    b += [arrow(230, 78, 290, 78), arrow(490, 78, 550, 78), arrow(650, 106, 650, 170), arrow(550, 198, 490, 198),
          arrow(290, 198, 230, 198), arrow(130, 170, 130, 106)]
    b.append(box(200, 252, 380, 40, 'box-on', 8))
    b.append(text(390, 277, 'After 4 rounds: 35.5M pages, 120B tokens', 't-strong', 'middle'))
    b.append(text(390, 312, 'Round 4 found almost nothing new: nearly 98% was already in round 3, so they stopped.', 't-tick', 'middle'))
    return svg(W, 326, 'The DeepSeekMath data loop: a seed set trains a fastText classifier, which scores Common Crawl; maths-heavy domains are found, people mark maths URL paths, missed pages join the seed, and the loop repeats four times to reach 35.5 million pages and 120 billion tokens.', b)


F['math_loop'] = math_loop()


# 5. Sharpening two answer distributions ---------------------------------------------------------------------
def sharpen_fig():
    import numpy as np
    qs = [('Question A: the right answer is already the favourite', [0.5, 0.2, 0.15, 0.1, 0.05]),
          ('Question B: the right answer is second', [0.25, 0.45, 0.15, 0.1, 0.05])]
    b = [text(20, 24, 'RL as sharpening: squaring each probability and renormalising (alpha = 2)', 't-title')]
    out = {}
    for qi, (title, p) in enumerate(qs):
        p = np.array(p); s = p ** 2 / (p ** 2).sum()
        out[title] = [p.round(3).tolist(), s.round(3).tolist()]
        x0 = 20 + qi * 390
        b.append(text(x0, 54, title, 't-note'))
        for k, (lab, vals, cls) in enumerate([('before RL', p, 'box'), ('after RL', s, 'box-1')]):
            yb = 70 + k * 120
            b.append(text(x0, yb + 12, lab, 't-label'))
            for j, v in enumerate(vals):
                bx = x0 + 70 + j * 58
                h = v * 80
                b.append(box(bx, yb + 96 - h, 40, h, 'box-on' if j == 0 else cls, 3))
                b.append(text(bx + 20, yb + 90 - h, f'{v:.2f}', 't-tick', 'middle'))
                b.append(text(bx + 20, yb + 112, 'right' if j == 0 else f'w{j}', 't-tick', 'middle'))
    b.append(text(20, 330, 'Highlighted bar: the right answer. Sharpening helps A (0.50 to 0.77) and hurts B (0.25 to 0.21).', 't-tick'))
    save('part8_sharpen_example', out)
    return svg(W, 344, 'Two answer distributions before and after squaring and renormalising: the right answer grows from 0.50 to 0.77 when it is already the favourite, and shrinks from 0.25 to 0.21 when it is second.', b)


F['sharpen'] = sharpen_fig()


# 6. Maj@K vs Pass@K from the toy simulation -----------------------------------------------------------------
def majk():
    r = R('part8_majk_passk')
    pct = lambda v: [100 * x for x in v]
    return lines_chart(W, 330, 'Toy simulation: Pass@K and Maj@K for an instruct model and its sharpened RL version, K from 1 to 64.',
                       r['K'], [('Pass@K instruct', pct(r['instruct']['pass']), 'l1', 's1'),
                                ('Pass@K RL', pct(r['rl']['pass']), 'l2', 's2'),
                                ('Maj@K instruct', pct(r['instruct']['maj']), 'l3', 's3'),
                                ('Maj@K RL', pct(r['rl']['maj']), 'l4', 's4')],
                       'K: number of samples per question (log scale)', 'accuracy (%)', 40, 100, [40, 50, 60, 70, 80, 90, 100],
                       xlog=True, fmt=lambda v: f'{v:.0f}', right=190)


F['majk'] = majk()


# 7. DeepSeekMath-V2: generator, verifier, meta-verifier -----------------------------------------------------
def mathv2():
    b = [text(20, 24, 'DeepSeekMath-V2: one model learns to write proofs and to grade them', 't-title')]
    b += label_box(300, 40, 180, 50, ['Hard new proofs', 'labelled by many checks'], 'box')
    b.append(arrow(390, 90, 390, 120))
    b.append(text(500, 70, 'retrain the verifier, so it stays ahead', 't-tick'))
    b += label_box(20, 120, 180, 64, ['Generator', 'writes a proof +', 'its own critique'], 'box-1')
    b += label_box(300, 120, 180, 64, ['Verifier', 'lists issues,', 'scores 0, 0.5 or 1'], 'box-2')
    b += label_box(580, 120, 180, 64, ['Meta-verifier', 'are those issues', 'real and fair?'], 'box-3')
    b += [arrow(200, 152, 300, 152), arrow(480, 152, 580, 152)]
    b += label_box(300, 240, 180, 56, ['Reward', 'proof score x honesty'], 'box-on')
    b += [arrow(390, 184, 390, 240), arrow(670, 184, 480, 260)]
    b.append(arrow(300, 268, 110, 184, True))
    b.append(text(150, 260, 'RL update', 't-tick', 'middle'))
    b.append(text(20, 326, 'The generator is paid for a good proof and for an honest self-check:', 't-tick'))
    b.append(text(20, 344, 'claiming "no issues" on a flawed proof costs reward.', 't-tick'))
    return svg(W, 358, 'DeepSeekMath-V2 training loop: a generator writes a proof and a self-critique, a verifier lists issues and scores it, a meta-verifier checks that the listed issues are real, and the resulting reward updates the generator; heavy verification labels hard new proofs to retrain the verifier.', b)


F['mathv2'] = mathv2()


# 8. Prover-V2 subgoal pipeline ------------------------------------------------------------------------------
def prover_v2():
    b = [text(20, 24, 'Prover-V2 cold start: a big model plans, a small model fills the holes, Lean checks', 't-title')]
    b += label_box(20, 56, 160, 60, ['Hard theorem', 'in Lean 4'], 'box')
    b += label_box(220, 56, 200, 60, ['DeepSeek-V3', 'sketch + subgoals', 'each "by sorry"'], 'box-1')
    b += [arrow(180, 86, 220, 86), arrow(420, 86, 460, 86)]
    for i in range(3):
        b += label_box(460, 40 + i * 44, 140, 36, [f'subgoal {i + 1}'], 'box-2', first='t-note')
    b += label_box(630, 56, 130, 60, ['Prover 7B', 'proves each', 'subgoal'], 'box-2')
    b += [arrow(600, 86, 630, 86)]
    b += label_box(460, 200, 300, 56, ['Lean 4 checks the stitched proof', 'accepted or rejected, nothing in between'], 'box-on')
    b += [arrow(695, 116, 695, 200)]
    b += label_box(20, 200, 360, 56, ['Training data', "V3's reasoning + the checked formal proof"], 'box-3')
    b += [arrow(460, 228, 380, 228)]
    b.append(text(20, 296, 'Then RL (GRPO) with the Lean verdict as reward. The released 671B model starts from DeepSeek-V3-Base.', 't-tick'))
    return svg(W, 312, 'Prover-V2 cold-start pipeline: DeepSeek-V3 writes a proof sketch with subgoals left as sorry, a 7B prover proves each subgoal, Lean checks the stitched proof, and the checked proof plus V3 reasoning becomes training data for reinforcement learning.', b)


F['prover_v2'] = prover_v2()


# 9. miniF2F progress bars ------------------------------------------------------------------------------------
def prover_bars():
    rows = [('Prover-V1, 64 samples', 46.3, 'box'), ('Prover-V1, cumulative', 52.0, 'box'),
            ('Prover-V1.5, whole proof', 60.2, 'box-1'), ('Prover-V1.5 + RMaxTS', 63.5, 'box-1'),
            ('Prover-V2 7B, Pass@8192', 82.0, 'box-2'), ('Prover-V2 671B, Pass@8192', 88.9, 'box-on')]
    b = [text(20, 24, 'miniF2F-test, share of 244 competition problems proved and checked by Lean (%)', 't-title')]
    x0, wmax = 230, 470
    for i, (lab, v, cls) in enumerate(rows):
        y = 48 + i * 40
        b += [text(x0 - 10, y + 20, lab, 't-tick', 'end'), box(x0, y + 4, wmax * v / 100, 26, cls, 4),
              text(x0 + wmax * v / 100 + 8, y + 22, f'{v:.1f}', 't-val')]
    b.append(text(20, 304, 'Settings differ between rows (samples, search), so read it as a trend, not a fair race. Sources: each paper.', 't-tick'))
    return svg(W, 318, 'Bar chart of miniF2F-test pass rates: Prover-V1 46.3 and 52.0, Prover-V1.5 60.2 and 63.5 with tree search, Prover-V2 7B 82.0 and 671B 88.9 at 8192 samples.', b)


F['prover_bars'] = prover_bars()


# 10. VL2 dynamic tiling ---------------------------------------------------------------------------------------
def vl2():
    r = R('part8_vl2_tiles')
    shot = next(x for x in r if x['image'].startswith('screenshot'))
    b = [text(20, 24, 'DeepSeek-VL2 dynamic tiling: a 1920 x 1080 screenshot', 't-title')]
    b.append(box(20, 50, 192, 108, 'box', 4))
    b.append(text(116, 108, '1920 x 1080', 't-tick', 'middle'))
    b.append(arrow(220, 104, 260, 104, True))
    cols, rows = shot['cols'], shot['rows']
    s = 52
    for i in range(rows):
        for j in range(cols):
            b.append(box(270 + j * s, 52 + i * s, s - 4, s - 4, 'box-1', 3))
    b.append(text(270 + cols * s / 2, 52 + rows * s + 18, f'{cols} x {rows} local tiles of 384 x 384', 't-tick', 'middle'))
    b.append(box(510, 66, 70, 70, 'box-2', 3))
    b.append(text(545, 156, '+ 1 thumbnail', 't-tick', 'middle'))
    b.append(arrow(590, 104, 620, 104, True))
    b += label_box(626, 60, 140, 90, ['SigLIP per tile', '27 x 27 = 729', '2x2 squeeze: 196', f'total {shot["tokens"]:,}'], 'box')
    b.append(text(20, 206, f'Total vision tokens = 14 x 15 (thumbnail) + 1 + ({rows} x 14) x ({cols} x 14 + 1) = {shot["tokens"]:,}.', 't-note'))
    b.append(text(20, 228, 'The grid is chosen from all cols x rows <= 9 to keep the most detail; the language side is a DeepSeekMoE model.', 't-tick'))
    return svg(W, 244, f'DeepSeek-VL2 cuts a 1920 by 1080 screenshot into {cols} by {rows} tiles of 384 pixels plus one thumbnail; each tile becomes 196 tokens, for {shot["tokens"]:,} vision tokens in total.', b)


F['vl2_tiles'] = vl2()


# 11. Janus: two encoders, one transformer -------------------------------------------------------------------
def janus():
    b = [text(20, 24, 'Janus: one transformer, two ways in for images', 't-title')]
    b += label_box(20, 220, 170, 56, ['Understanding', 'SigLIP: meaning'], 'box-1')
    b += label_box(210, 220, 150, 56, ['Text tokenizer', 'the question'], 'box')
    b += label_box(420, 220, 150, 56, ['Text tokenizer', 'the prompt'], 'box')
    b += label_box(590, 220, 170, 56, ['Generation', 'VQ: image codes'], 'box-2')
    b += label_box(20, 120, 740, 50, ['One autoregressive transformer (DeepSeek-LLM 1.3B to 7B)'], 'box-on')
    b += [arrow(105, 220, 105, 170), arrow(285, 220, 285, 170), arrow(495, 220, 495, 170), arrow(675, 220, 675, 170)]
    b += label_box(110, 46, 200, 48, ['Text head', 'writes the answer'], 'box-1')
    b += label_box(470, 46, 200, 48, ['Image head', 'next image code'], 'box-2')
    b += [arrow(210, 120, 210, 94), arrow(570, 120, 570, 94)]
    b.append(text(20, 304, 'Understanding needs "what is in the picture"; drawing needs "which pixels go where". One encoder for both made each worse.', 't-tick'))
    return svg(W, 318, 'Janus architecture: images for understanding enter through a SigLIP encoder and images for generation through a VQ tokenizer, both feeding one autoregressive transformer with a text head and an image head.', b)


F['janus'] = janus()


# 12. DeepEncoder pipeline ------------------------------------------------------------------------------------
def deepencoder():
    o = R('part8_ocr_calc')
    b = [text(20, 24, 'DeepSeek-OCR at Base mode: a 1024 x 1024 page becomes 256 vision tokens', 't-title')]
    stages = [('page image', '1024 x 1024', 'box'), ('16 x 16 patches', '4,096 tokens', 'box'),
              ('SAM-base 80M', 'window attention', 'box-1'), ('conv squeeze', '16x: 256 tokens', 'box-on'),
              ('CLIP-large 300M', 'global attention', 'box-2'), ('MoE decoder', '3B, 570M active', 'box-3')]
    for i, (a, c, cls) in enumerate(stages):
        x = 20 + i * 126
        b += label_box(x, 56, 112, 64, [a, c], cls)
        if i:
            b.append(arrow(x - 14, 88, x, 88))
    pages = o['pages']
    vals = sorted(v['text_tokens'] for v in pages.values())
    b.append(text(20, 160, f'The same page as text: {vals[0]:,} to {vals[-1]:,} text tokens (three pages of the OCR paper, our count).', 't-note'))
    b.append(text(20, 182, f'Compression = text tokens / vision tokens: about {vals[0] / 256:.1f}x to {vals[-1] / 256:.1f}x at Base, '
                  f'{vals[0] / 64:.1f}x to {vals[-1] / 64:.1f}x at Tiny (64 tokens).', 't-tick'))
    b.append(text(20, 202, 'Window attention keeps memory low on 4,096 tokens; global attention runs only after the squeeze.', 't-tick'))
    return svg(W, 218, 'DeepSeek-OCR pipeline: a 1024 by 1024 page is cut into 4,096 patches, passed through SAM with window attention, squeezed 16 times to 256 tokens by convolutions, passed through CLIP with global attention, and read by a 3B mixture-of-experts decoder.', b)


F['deepencoder'] = deepencoder()


# 13. Precision vs compression (Table 2) ---------------------------------------------------------------------
def ocr_curve():
    t = R('part8_ocr_calc')['table2']
    L, T, Wd, Hd = 70, 40, 560, 230
    X = lambda v: L + (v - 5) / (21 - 5) * Wd
    Y = lambda v: T + Hd - (v - 50) / 50 * Hd
    b = [text(20, 22, 'DeepSeek-OCR: decoding precision against compression (Fox benchmark, paper Table 2)', 't-title')]
    for v in (50, 60, 70, 80, 90, 100):
        b += [line(L, Y(v), L + Wd, Y(v), 'grid'), text(L - 8, Y(v) + 4, f'{v}%', 't-tick', 'end')]
    for v in (5, 10, 15, 20):
        b.append(text(X(v), T + Hd + 20, f'{v}x', 't-tick', 'middle'))
    b.append(line(X(10), T, X(10), T + Hd, 'base-line'))
    b.append(text(X(10) + 6, T + Hd - 8, '10x line', 't-tick'))
    for key, rk, cls, lc, lab in [('p100', 'r100', 's2', 'l2', '100 vision tokens'), ('p64', 'r64', 's1', 'l1', '64 vision tokens')]:
        pts = [(row[rk], row[key]) for row in t]
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(a):.1f},{Y(c):.1f}' for a, c in pts) + '"/>')
        for (a, c), row in zip(pts, t):
            b.append(f'<g class="mark"><title>{esc(lab)}, {row["bucket"]} text tokens: {c}% at {a}x</title>{dot(X(a), Y(c), cls, 5)}</g>')
        a, c = pts[-1]
        ly = T + Hd + 64 + (0 if key == 'p100' else 20)
        b += [dot(L + 6, ly - 4, cls, 5), text(L + 18, ly, f'{lab}: {pts[0][1]}% at {pts[0][0]}x, falling to {c}% at {a}x', 't-note')]
    b.append(text(L + Wd / 2, T + Hd + 40, 'compression = text tokens on the page / vision tokens', 't-tick', 'middle'))
    return svg(W, 364, 'Line chart of DeepSeek-OCR precision against compression ratio: with 100 vision tokens precision stays between 87.1 and 98.5 percent up to 12.6 times; with 64 tokens it falls from 96.5 percent at 10.5 times to 59.1 percent at 19.7 times.', b)


F['ocr_curve'] = ocr_curve()


# 14. What flowed back into the main models -------------------------------------------------------------------
def flow_map():
    left = [('Coder-v1.5 7B', 'box-1'), ('DeepSeekMath 7B (GRPO, data loop)', 'box-1'), ('Coder-V2', 'box-1'),
            ('mHC (residual research)', 'box'), ('DeepSeekMath-V2 (self-verification)', 'box-2'), ('', None),
            ('GRM / SPCT reward models', 'box-3'), ('VL, VL2, Janus, OCR', 'box-3'), ('DSpark (speculative decoding)', 'box-3'),
            ('Engram (conditional memory)', 'box')]
    right = [('DeepSeekMath 7B start', 'box'), ('V2-Chat, Coder-V2, R1, V4 (GRPO)', 'box'), ('V2.5 (merge, Sept 2024)', 'box'),
             ('V4 residual stream (mHC)', 'box'), ('V3.2-Speciale', 'box'), ('V4 formal-maths pipeline', 'box'),
             ('V4 post-training GRM', 'box'), ('V4-Flash-Vision-Exp', 'box'), ('V4 serving, V4-*-DSpark', 'box'),
             ('V4.1 (Engram memory)', 'box')]
    b = [text(20, 24, 'What flowed from the side models into the main line', 't-title'),
         text(20, 46, 'Specialised work', 't-label'), text(470, 46, 'Main models', 't-label')]
    ly = lambda i: 58 + i * 40
    for i, (s, cls) in enumerate(left):
        if cls:
            b += [box(20, ly(i), 300, 32, cls, 6), text(32, ly(i) + 21, s, 't-note')]
    for i, (s, cls) in enumerate(right):
        b += [box(470, ly(i), 290, 32, cls, 6), text(482, ly(i) + 21, s, 't-note')]
    edges = [(0, 0, False), (1, 1, False), (2, 2, False), (3, 3, False), (4, 4, False), (4, 5, False), (6, 6, True),
             (7, 7, True), (8, 8, False), (9, 9, False)]
    for a, c, dash in edges:
        b.append(f'<line class="edge" x1="320" y1="{ly(a) + 16}" x2="466" y2="{ly(c) + 16}"{" stroke-dasharray=\"5 4\"" if dash else ""} marker-end="url(#ah)"/>')
    b.append(text(20, ly(10) + 14, 'Solid: stated in a DeepSeek paper or model card. Dashed: our inference, or reuse not confirmed.', 't-tick'))
    b.append(text(20, ly(10) + 32, 'Not drawn: the DeepSeekMath data loop also built the Coder-V2 corpus; V3 and V3.2 were the bases of Prover-V2 and Math-V2.', 't-tick'))
    return svg(W, ly(10) + 44, 'Map of influence: Coder-v1.5 started DeepSeekMath; DeepSeekMath gave GRPO to V2, R1, V3.2 and V4 and its data loop to Coder-V2; Coder-V2 merged into V2.5; DeepSeekMath-V2 fed V3.2-Speciale and the V4 formal-maths pipeline; reward-model work, the vision line and DSpark connect to V4.', b)


F['flow_map'] = flow_map()

json.dump(F, open(os.path.join(RESULTS, 'figs_part8.json'), 'w'))
print(f'wrote {len(F)} figures:', ', '.join(F))
