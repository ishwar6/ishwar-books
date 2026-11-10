"""Figures for Part 1 (every DeepSeek model, and what is open). Writes results/figs_part1.json (key -> SVG).
Sizes come from results/part1_configs.json (run part1_configs.py first); dates, licenses and prices come from the
research dossier 01-catalog-and-openness.md (each with its primary source there)."""
import json, math, os
from datetime import date
from figlib import svg, text, box, arrow, esc

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = {r['repo']: r for r in json.load(open(os.path.join(HERE, 'results', 'part1_configs.json')))}
F = {}


def X(d, x0, x1, start=date(2023, 10, 1), end=date(2026, 11, 30)):
    return x0 + (d - start).days / (end - start).days * (x1 - x0)


def year_axis(b, x0, x1, y, start=date(2023, 10, 1), end=date(2026, 11, 30), top=None):
    b.append(f'<line class="axis" x1="{x0}" y1="{y}" x2="{x1}" y2="{y}"/>')
    for yr in range(start.year + (start.month > 1), end.year + 1):
        x = X(date(yr, 1, 1), x0, x1, start, end)
        if top is not None:
            b.append(f'<line class="grid" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{y}"/>')
        b.append(f'<line class="axis" x1="{x:.1f}" y1="{y}" x2="{x:.1f}" y2="{y + 6}"/>')
        b.append(text(x, y + 22, str(yr), 't-tick', 'middle'))


# 1. Timeline of the main line ------------------------------------------------------------------------------------
def timeline():
    W, H, x0, x1, ay = 860, 400, 30, 830, 300
    main = [  # date, label, size, level (0 = lowest label row)
        (date(2023, 11, 29), 'LLM 67B', 'dense 67B', 0),
        (date(2024, 5, 7), 'V2', '236B / 21B', 0),
        (date(2024, 9, 5), 'V2.5', '236B / 21B', 1),
        (date(2024, 12, 26), 'V3', '671B / 37B', 0),
        (date(2025, 1, 20), 'R1', '671B / 37B', 2),
        (date(2025, 3, 24), 'V3-0324', 'MIT now', 1),
        (date(2025, 5, 28), 'R1-0528', '671B / 37B', 3),
        (date(2025, 8, 21), 'V3.1', 'think + non-think', 0),
        (date(2025, 9, 29), 'V3.2-Exp', 'sparse attention', 2),
        (date(2025, 12, 1), 'V3.2', '671B / 37B', 1),
        (date(2026, 4, 24), 'V4 preview', 'Pro 1.6T / 49B', 0),
        (date(2026, 4, 24), '', 'Flash 284B / 13B', -1),
        (date(2026, 7, 31), 'V4-Flash-0731', '', 2),
        (date(2026, 8, 13), 'V4-Pro-0813', '', 3),
        (date(2026, 9, 10), 'V4.1-Flash', '552B + 196B / 16B', 1),
    ]
    b = []
    year_axis(b, x0, x1, ay)
    lic = [(date(2023, 10, 20), date(2025, 1, 20), 'box-2', 'DeepSeek License (use restrictions)'),
           (date(2025, 1, 20), date(2026, 11, 10), 'box-1', 'MIT for R1 at once; for the V3 line from V3-0324')]
    for d0, d1, cls, lab in lic:
        xa, xb = X(d0, x0, x1), X(d1, x0, x1)
        b.append(box(xa, ay + 34, xb - xa, 26, cls, rx=4))
        b.append(text((xa + xb) / 2, ay + 52, lab, 't-note', 'middle'))
    b.append(text(x0, ay + 84, 'Weight license of the flagship models. The first V3 repo still carries the DeepSeek License.', 't-muted'))
    for d, lab, size, lvl in main:
        x = X(d, x0, x1)
        if lvl < 0:
            continue
        ty = ay - 28 - lvl * 58
        b.append(f'<line class="grid" x1="{x:.1f}" y1="{ty + 14:.1f}" x2="{x:.1f}" y2="{ay}"/>')
        b.append(f'<circle class="s1 ring" cx="{x:.1f}" cy="{ay}" r="6"/>')
        anchor = 'end' if x > 760 else 'start'
        dx = -4 if anchor == 'end' else -4
        b.append(text(x + (4 if anchor == 'end' else dx), ty - 16, lab, 't-title', anchor))
        extra = size
        if lab == 'V4 preview':
            extra = 'Pro 1.6T/49B, Flash 284B/13B'
        if extra:
            b.append(text(x + (4 if anchor == 'end' else dx), ty, extra, 't-muted', anchor))
    b.append(text(x0, 18, 'Main DeepSeek language models (total / active parameters)', 't-tick'))
    F['timeline'] = svg(W, H, 'A timeline of the main DeepSeek language models from DeepSeek LLM 67B in November 2023 to V4.1-Flash in September 2026, with a license band under it', b)


# 2. Family map ---------------------------------------------------------------------------------------------------
def families():
    W, x0, x1, lane_h, top = 900, 170, 880, 66, 34
    lanes = [
        ('Main line', 'box-on', [(date(2023, 11, 29), 'LLM 67B'), (date(2024, 5, 7), 'V2'), (date(2024, 9, 5), 'V2.5'),
                                 (date(2024, 12, 26), 'V3'), (date(2025, 8, 21), 'V3.1'), (date(2025, 12, 1), 'V3.2'),
                                 (date(2026, 4, 24), 'V4'), (date(2026, 9, 10), 'V4.1')]),
        ('Reasoning', 'box-1', [(date(2025, 1, 20), 'R1 + distills'), (date(2025, 5, 28), 'R1-0528')]),
        ('Code', 'box-1', [(date(2023, 11, 1), 'Coder'), (date(2024, 1, 25), 'Coder 1.5'), (date(2024, 6, 17), 'Coder-V2')]),
        ('Math and proofs', 'box-1', [(date(2024, 2, 5), 'Math 7B'), (date(2024, 5, 23), 'Prover'), (date(2024, 8, 15), 'Prover-1.5'),
                                      (date(2025, 4, 30), 'Prover-V2'), (date(2025, 11, 27), 'Math-V2')]),
        ('Vision and documents', 'box-2', [(date(2024, 3, 7), 'VL'), (date(2024, 10, 18), 'Janus'), (date(2024, 12, 13), 'VL2'),
                                           (date(2025, 1, 27), 'Janus-Pro'), (date(2025, 10, 17), 'OCR'), (date(2026, 1, 27), 'OCR 2'),
                                           (date(2026, 8, 21), 'V4-Vision-Exp')]),
        ('Architecture research', 'box', [(date(2024, 1, 9), 'MoE 16B'), (date(2024, 7, 4), 'ESFT'), (date(2025, 2, 16), 'NSA'),
                                          (date(2025, 12, 31), 'mHC'), (date(2026, 1, 12), 'Engram'), (date(2026, 7, 6), 'DSpark')]),
    ]
    H = top + len(lanes) * lane_h + 46
    b, pos = [], {}
    year_axis(b, x0, x1, top + len(lanes) * lane_h + 4, top=top - 6)
    for i, (name, cls, items) in enumerate(lanes):
        y = top + i * lane_h
        b.append(f'<line class="grid" x1="10" y1="{y + lane_h - 2}" x2="{x1}" y2="{y + lane_h - 2}"/>')
        b.append(text(12, y + lane_h / 2 + 4, name, 't-title' if i == 0 else 't-note'))
        last_end = [-1e9, -1e9]
        for d, lab in items:
            w = max(40, 7.3 * len(lab) + 14)
            cx = X(d, x0, x1)
            bx = min(max(cx - w / 2, x0 - 10), x1 - w)
            row = 0 if bx > last_end[0] + 3 else 1
            if row == 1 and bx <= last_end[1] + 3:
                row = 0
            last_end[row] = bx + w
            by = y + 6 + row * 28
            b.append(box(bx, by, w, 24, cls, rx=6))
            b.append(text(bx + w / 2, by + 16.5, lab, 't-note', 'middle'))
            pos[lab] = (bx, by, w)
    def link(a, z):
        ax, ay_, aw = pos[a]; zx, zy, zw = pos[z]
        down = zy > ay_
        y1, y2 = (ay_ + 24, zy - 2) if down else (ay_, zy + 26)
        b.append(arrow(ax + aw / 2, y1, zx + zw / 2, y2))
    link('V3', 'R1 + distills'); link('V2', 'Coder-V2'); link('Coder-V2', 'V2.5');
    link('V3', 'Prover-V2'); link('V3.2', 'Math-V2'); link('V4', 'V4-Vision-Exp'); link('Engram', 'V4.1')
    b.append(text(x0 - 160, H - 6, 'Arrows: "built on" or "fed into". Dates are first public release.', 't-muted'))
    F['families'] = svg(W, H, 'A map of DeepSeek model families in six lanes against time, with arrows where one model was built on another', b)


# 3. Total and active parameters (log scale) ----------------------------------------------------------------------
def sizes():
    rows = [('LLM 67B', 67.4e9, 67.4e9, 'deepseek-llm-67b-base'), ('MoE 16B', 16.4e9, 2.8e9, None),
            ('V2-Lite', 15.7e9, 2.4e9, None), ('V2', 236e9, 21e9, None), ('V3 / R1 / V3.x', 671e9, 37e9, None),
            ('V4-Flash', 284e9, 13e9, None), ('V4.1-Flash', 552e9, 16e9, None), ('V4-Pro', 1.6e12, 49e9, None)]
    W, L, R, T, rh = 860, 130, 760, 40, 40
    H = T + len(rows) * rh + 76
    lo, hi = math.log10(1e9), math.log10(2e12)
    Xv = lambda v: L + (math.log10(v) - lo) / (hi - lo) * (R - L)
    b = []
    for t, lab in [(1e9, '1B'), (1e10, '10B'), (1e11, '100B'), (1e12, '1T')]:
        b.append(f'<line class="grid" x1="{Xv(t):.1f}" y1="{T - 8}" x2="{Xv(t):.1f}" y2="{T + len(rows) * rh}"/>')
        b.append(text(Xv(t), T + len(rows) * rh + 20, lab, 't-tick', 'middle'))
    for i, (lab, tot, act, _) in enumerate(rows):
        y = T + i * rh
        b.append(text(L - 10, y + 22, lab, 't-note', 'end'))
        b.append(f'<g class="mark"><title>{esc(lab)}: total {tot / 1e9:,.1f}B</title>' + box(L, y + 4, Xv(tot) - L, 14, 'box-1', rx=3) + '</g>')
        b.append(f'<g class="mark"><title>{esc(lab)}: active {act / 1e9:,.1f}B</title>' + box(L, y + 20, Xv(act) - L, 14, 'box-on', rx=3) + '</g>')
        fmt = lambda v: f'{v / 1e12:.1f}T' if v >= 1e12 else f'{v / 1e9:.0f}B' if v >= 1e11 or round(v / 1e9, 1) == round(v / 1e9) else f'{v / 1e9:.1f}B'
        b.append(text(Xv(tot) + 6, y + 15, fmt(tot), 't-tick'))
        b.append(text(Xv(act) + 6, y + 31, fmt(act) + ('' if act < tot else ' (all of it)'), 't-tick'))
    b.append(box(L, 10, 14, 12, 'box-1', rx=2)); b.append(text(L + 20, 21, 'total parameters', 't-tick'))
    b.append(box(L + 160, 10, 14, 12, 'box-on', rx=2)); b.append(text(L + 180, 21, 'active per token', 't-tick'))
    b.append(text(L, H - 30, 'Log scale: each grid line is 10 times the one before.', 't-muted'))
    b.append(text(L, H - 12, 'V4.1-Flash: the 552B backbone (its 196B Engram memory is not shown); active is the decode figure.', 't-muted'))
    F['sizes'] = svg(W, H, 'Bars of total and active parameters for eight DeepSeek models on a log scale', b)


# 4. Dense vs mixture of experts -----------------------------------------------------------------------------------
def dense_vs_moe():
    W, H = 860, 364
    b = [text(20, 24, 'Dense model: every weight works on every token', 't-title'),
         text(450, 24, 'DeepSeek-V3 MoE layer: 1 shared + 8 of 256 routed experts', 't-title')]
    b.append(box(20, 50, 90, 34, 'box', rx=6)); b.append(text(65, 72, 'token', 't-note', 'middle'))
    b.append(arrow(112, 67, 150, 67))
    b.append(box(152, 40, 240, 230, 'box-on', rx=8))
    b.append(text(272, 160, 'one big feed-forward block', 't-note', 'middle'))
    b.append(text(272, 180, '(all of it is used)', 't-muted', 'middle'))
    b.append(text(20, 332, 'DeepSeek LLM 67B: all 67B weights work on each token.', 't-tick'))
    b.append(box(450, 50, 70, 34, 'box', rx=6)); b.append(text(485, 72, 'token', 't-note', 'middle'))
    b.append(box(450, 110, 70, 50, 'box-on', rx=6)); b.append(text(485, 132, 'shared', 't-note', 'middle')); b.append(text(485, 148, 'expert', 't-note', 'middle'))
    b.append(box(450, 180, 70, 40, 'box', rx=6)); b.append(text(485, 205, 'router', 't-note', 'middle'))
    b.append(arrow(485, 86, 485, 108)); b.append(arrow(522, 200, 566, 200))
    b.append('<path class="edge" d="M450,67 H438 V200 H446" marker-end="url(#ah)"/>')
    on = {7, 30, 61, 99, 140, 171, 203, 250}
    c, gx, gy = 15, 570, 40
    for k in range(256):
        r, q = divmod(k, 16)
        cls = 'box-on' if k in on else 'box'
        b.append(f'<rect class="{cls}" x="{gx + q * c:.1f}" y="{gy + r * c:.1f}" width="{c - 3}" height="{c - 3}" rx="2"/>')
    b.append(text(gx + 8 * c, gy + 16 * c + 16, '256 routed experts (16 x 16)', 't-tick', 'middle'))
    b.append(text(450, 332, f'Counted from the V3 config: {CFG["DeepSeek-V3"]["active"] / 1e9:.1f}B of', 't-tick'))
    b.append(text(450, 350, f'{CFG["DeepSeek-V3"]["total"] / 1e9:.0f}B weights work on each token.', 't-tick'))
    F['dense_vs_moe'] = svg(W, H, 'Left: a dense model sends each token through one large block. Right: a DeepSeek-V3 MoE layer sends it to one shared expert and 8 of 256 routed experts', b)


# 5. What a model release can contain -------------------------------------------------------------------------------
def release_pipeline():
    W, H = 860, 300
    steps = [('Training data', 'box', 'described only'), ('Training code', 'box', 'not released*'),
             ('Training run', 'box', 'GPU hours, in papers'), ('Weights', 'box-1', 'released'),
             ('Inference code', 'box-1', 'released'), ('Answers', 'box-1', 'API, web, app')]
    b = [text(20, 24, 'From data to answers: which pieces DeepSeek publishes for its main models', 't-title')]
    w, gap, y = 122, 18, 70
    for i, (lab, cls, note) in enumerate(steps):
        x = 20 + i * (w + gap)
        b.append(box(x, y, w, 60, cls, rx=8))
        b.append(text(x + w / 2, y + 35, lab, 't-note', 'middle'))
        b.append(text(x + w / 2, y + 84, note, 't-muted', 'middle'))
        if i < len(steps) - 1:
            b.append(arrow(x + w + 2, y + 30, x + w + gap - 2, y + 30))
    b.append(f'<path class="edge" d="M20,{y + 104} H{20 + 3 * (w + gap) - gap}"/>')
    b.append(text(20 + (3 * (w + gap) - gap) / 2, y + 124, 'the recipe: kept in-house', 't-note', 'middle'))
    xs = 20 + 3 * (w + gap)
    b.append(f'<path class="path" d="M{xs},{y + 104} H{20 + 6 * (w + gap) - gap}"/>')
    b.append(text(xs + (3 * (w + gap) - gap) / 2, y + 124, 'the result: published', 't-note', 'middle'))
    b.append(text(20, y + 170, 'Papers describe all six steps. Kernels and systems for training and serving are also open (see the infrastructure map).', 't-tick'))
    b.append(text(20, y + 192, '* Narrow exceptions: the Coder fine-tuning script, ESFT, and DeepSpec (training code for draft models).', 't-muted'))
    F['release_pipeline'] = svg(W, H, 'Six boxes from training data to answers. Data, training code and the training run are kept in-house; weights, inference code and answers are published', b)


# 6. Openness matrix -----------------------------------------------------------------------------------------------
def matrix():
    cols = ['Paper', 'Weights', 'License', 'Inference code', 'Training code', 'Training data']
    # 2 = open, 1 = partial, 0 = not released; text for the license column
    rows = [('LLM 67B (2023)', [2, 2, 'DeepSeek', 1, 0, 0]),
            ('Coder (2023)', [2, 2, 'DeepSeek', 1, 1, 0]),
            ('V2 / V2.5 (2024)', [2, 2, 'DeepSeek', 2, 0, 0]),
            ('V3 (Dec 2024)', [2, 2, 'DeepSeek', 2, 0, 0]),
            ('R1 (Jan 2025)', [2, 2, 'MIT', 2, 0, 0]),
            ('V3-0324, V3.1', [1, 2, 'MIT', 2, 0, 0]),
            ('V3.2 (Dec 2025)', [2, 2, 'MIT', 2, 0, 0]),
            ('V4 (Apr 2026)', [2, 2, 'MIT', 2, 1, 0]),
            ('V4.1-Flash (Sep 2026)', [2, 2, 'MIT', 2, 0, 0])]
    W, L, cw, rh, T = 860, 190, 110, 32, 70
    H = T + len(rows) * rh + 70
    b = []
    for j, c in enumerate(cols):
        b.append(text(L + j * cw + cw / 2, T - 16, c, 't-title', 'middle'))
    for i, (name, vals) in enumerate(rows):
        y = T + i * rh
        b.append(f'<line class="grid" x1="20" y1="{y + rh}" x2="{L + len(cols) * cw}" y2="{y + rh}"/>')
        b.append(text(L - 12, y + 21, name, 't-note', 'end'))
        for j, v in enumerate(vals):
            cx, cy = L + j * cw + cw / 2, y + rh / 2
            if isinstance(v, str):
                b.append(box(cx - 42, cy - 11, 84, 22, 'box-1' if v == 'MIT' else 'box-2', rx=11))
                b.append(text(cx, cy + 4.5, v, 't-tick', 'middle'))
            elif v == 2:
                b.append(f'<circle class="s1" cx="{cx}" cy="{cy}" r="8"/>')
            elif v == 1:
                b.append(f'<circle class="box" cx="{cx}" cy="{cy}" r="8"/><path class="s1" d="M{cx},{cy - 8} A8,8 0 0,1 {cx},{cy + 8} Z"/>')
            else:
                b.append(f'<circle class="box" cx="{cx}" cy="{cy}" r="8"/>')
    ly = T + len(rows) * rh + 30
    b.append(f'<circle class="s1" cx="30" cy="{ly - 4}" r="7"/>'); b.append(text(44, ly, 'released', 't-tick'))
    b.append(f'<circle class="box" cx="140" cy="{ly - 4}" r="7"/><path class="s1" d="M140,{ly - 11} A7,7 0 0,1 140,{ly + 3} Z"/>'); b.append(text(154, ly, 'partial (see the table)', 't-tick'))
    b.append(f'<circle class="box" cx="330" cy="{ly - 4}" r="7"/>'); b.append(text(344, ly, 'not released', 't-tick'))
    b.append(text(20, ly + 26, 'Paper partial: V3.1 and V3-0324 have model cards and change logs, but no new paper.', 't-muted'))
    F['matrix'] = svg(W, H, 'A matrix of nine DeepSeek releases against six things that can be open: paper, weights, license, inference code, training code and training data', b)


# 7. License history ------------------------------------------------------------------------------------------------
def licenses():
    W, H, x0, x1 = 860, 250, 30, 830
    start, end = date(2023, 10, 1), date(2026, 11, 30)
    b = []
    year_axis(b, x0, x1, 200, start, end)
    ev = [(date(2023, 10, 23), 'DeepSeek License v1.0', 'commercial use allowed, use list in Attachment A', 0),
          (date(2025, 1, 20), 'R1: MIT', '"including ... distillation"', 1),
          (date(2025, 3, 24), 'V3-0324: MIT', 'the V3 line follows', 2),
          (date(2025, 11, 27), 'Math-V2: Apache-2.0', 'then OCR 2, Engram code, DeepInfer', 0),
          (date(2026, 4, 24), 'V4: MIT', 'base models too', 2)]
    for d, a, c, lvl in ev:
        x = X(d, x0, x1, start, end)
        y = 60 + lvl * 44
        b.append(f'<line class="grid" x1="{x:.1f}" y1="{y + 8}" x2="{x:.1f}" y2="200"/>')
        b.append(f'<circle class="s1 ring" cx="{x:.1f}" cy="200" r="6"/>')
        anc = 'end' if x > 700 else 'start'
        b.append(text(x + (6 if anc == 'start' else -6), y, a, 't-title', anc))
        b.append(text(x + (6 if anc == 'start' else -6), y + 17, c, 't-muted', anc))
    b.append(text(x0, 22, 'Weight licenses: the dates when each kind first appeared on a DeepSeek model', 't-tick'))
    F['licenses'] = svg(W, H, 'A timeline of DeepSeek weight licenses: DeepSeek License from October 2023, MIT from R1 in January 2025, Apache-2.0 from late 2025', b)


# 8. Open infrastructure map --------------------------------------------------------------------------------------
def infra():
    layers = [
        ('Serving', [('DeepInfer', 'box-2', 'Oct 2026'), ('DeepSpec', 'box-2', 'Jun 2026'), ('deepseek-recipe', 'box-2', 'Sep 2026')]),
        ('Parallelism and balance', [('DualPipe', 'box-1', 'Feb 2025'), ('EPLB', 'box-1', 'Feb 2025'), ('LPLB', 'box-2', 'Nov 2025')]),
        ('Communication', [('DeepEP', 'box-1', 'Feb 2025'), ('DeepEP-Ascend', 'box-3', 'Sep 2026')]),
        ('GPU kernels', [('FlashMLA', 'box-1', 'Feb 2025'), ('DeepGEMM', 'box-1', 'Feb 2025'), ('TileKernels', 'box-2', 'Apr 2026'),
                         ('DeepSelect', 'box-2', 'Sep 2026'), ('DeepGEMM-Ascend', 'box-3', 'Sep 2026')]),
        ('Storage and data', [('3FS', 'box-1', 'Feb 2025'), ('smallpond', 'box-1', 'Feb 2025')]),
    ]
    W, L, rh, T = 860, 190, 62, 50
    H = T + len(layers) * rh + 50
    b = [text(20, 24, 'Open infrastructure repos, from the disk up to the server', 't-title')]
    for i, (name, items) in enumerate(layers):
        y = T + i * rh
        b.append(box(14, y, W - 28, rh - 8, 'box', rx=8))
        b.append(text(28, y + rh / 2, name, 't-note'))
        x = L
        for lab, cls, when in items:
            w = max(96, 8 * len(lab) + 20)
            b.append(box(x, y + 7, w, rh - 22, cls, rx=6))
            b.append(text(x + w / 2, y + 24, lab, 't-note', 'middle'))
            b.append(text(x + w / 2, y + 39, when, 't-muted', 'middle'))
            x += w + 10
    ly = H - 18
    for k, (cls, lab) in enumerate([('box-1', 'Open Source Week, Feb 2025'), ('box-2', 'later releases'), ('box-3', 'ports to Huawei Ascend')]):
        b.append(box(20 + k * 260, ly - 11, 16, 14, cls, rx=3)); b.append(text(42 + k * 260, ly, lab, 't-tick'))
    F['infra'] = svg(W, H, 'Five layers of open DeepSeek infrastructure: storage, kernels, communication, parallelism and serving, with the repos in each', b)


# 9. API price history ------------------------------------------------------------------------------------------
def prices():
    W, H, L, R, T, B = 884, 380, 60, 650, 40, 320
    start, end = date(2024, 11, 1), date(2026, 11, 10)
    Xd = lambda d: L + (d - start).days / (end - start).days * (R - L)
    Y = lambda v: B - v / 4.2 * (B - T)
    default_out = [(date(2024, 11, 1), 0.28), (date(2025, 2, 8), 1.10), (date(2025, 9, 5), 1.68), (date(2025, 9, 29), 0.42),
                   (date(2026, 4, 24), 0.28), (date(2026, 8, 16), 1.32), (date(2026, 9, 10), 1.20), (end, 1.20)]
    default_in = [(date(2024, 11, 1), 0.14), (date(2025, 2, 8), 0.27), (date(2025, 9, 5), 0.56), (date(2025, 9, 29), 0.28),
                  (date(2026, 4, 24), 0.14), (date(2026, 8, 16), 0.44), (date(2026, 9, 10), 0.30), (end, 0.30)]
    premium_out = [(date(2025, 1, 20), 2.19), (date(2025, 9, 5), 1.68), (date(2025, 9, 29), 0.42), (date(2026, 4, 24), 3.48),
                   (date(2026, 5, 31), 3.48)]
    premium_out2 = [(date(2026, 8, 16), 3.96), (end, 3.96)]

    def step(pts, cls, dash=False):
        d = f'M{Xd(pts[0][0]):.1f},{Y(pts[0][1]):.1f}'
        for (d0, v0), (d1, v1) in zip(pts, pts[1:]):
            d += f' H{Xd(d1):.1f} V{Y(v1):.1f}'
        extra = ' stroke-dasharray="5 4"' if dash else ''
        return f'<path class="{cls}" d="{d}"{extra}/>'
    b = []
    for v in [0, 1, 2, 3, 4]:
        b.append(f'<line class="grid" x1="{L}" y1="{Y(v):.1f}" x2="{R}" y2="{Y(v):.1f}"/>')
        b.append(text(L - 8, Y(v) + 4, f'${v}', 't-tick', 'end'))
    for yr, m in [(2025, 1), (2026, 1)]:
        x = Xd(date(yr, m, 1)); b.append(text(x, B + 20, str(yr), 't-tick', 'middle'))
        b.append(f'<line class="axis" x1="{x:.1f}" y1="{B}" x2="{x:.1f}" y2="{B + 6}"/>')
    b.append(f'<line class="axis" x1="{L}" y1="{B}" x2="{R}" y2="{B}"/>')
    b.append(step(default_out, 'l1')); b.append(step(default_in, 'l2')); b.append(step(premium_out, 'l3'))
    b.append(f'<path class="l3" stroke-dasharray="4 5" d="M{Xd(date(2026, 5, 31)):.1f},{Y(3.48):.1f} H{Xd(date(2026, 8, 16)):.1f} V{Y(3.96):.1f}"/>')
    b.append(step(premium_out2, 'l3'))
    notes = [(date(2025, 2, 8), 1.10, 'V3 list price'), (date(2025, 9, 29), 0.42, 'V3.2-Exp'),
             (date(2026, 4, 24), 0.28, 'V4-Flash'), (date(2026, 9, 10), 1.20, 'V4.1-Flash'), (date(2026, 4, 24), 3.48, 'V4-Pro list'),
             (date(2025, 1, 20), 2.19, 'R1')]
    for d, v, lab in notes:
        b.append(f'<circle class="s1 ring" cx="{Xd(d):.1f}" cy="{Y(v):.1f}" r="4"/>')
        left = lab in ('V4.1-Flash', 'V3.2-Exp')
        b.append(text(Xd(d) + (-6 if left else 6), Y(v) + (20 if lab == 'V3.2-Exp' else -22 if lab == 'V4.1-Flash' else -8), lab, 't-muted', 'end' if left else 'start'))
    lx = R + 18
    for k, (cls, lab) in enumerate([('l1', 'default model, output'), ('l2', 'default model, input (cache miss)'), ('l3', 'reasoner, then V4-Pro, output')]):
        y = 70 + k * 26
        b.append(f'<line class="{cls}" x1="{lx}" y1="{y - 4}" x2="{lx + 22}" y2="{y - 4}"/>')
        b.append(text(lx + 28, y, lab, 't-tick'))
    b.append(text(lx, 170, 'Dashed: no archived', 't-muted')); b.append(text(lx, 186, 'price page found.', 't-muted'))
    b.append(text(lx, 214, 'From Aug 2026: peak', 't-muted')); b.append(text(lx, 230, 'prices; off-peak is half.', 't-muted'))
    b.append(text(L, 22, 'US dollars per 1 million tokens, DeepSeek API', 't-tick'))
    F['prices'] = svg(W, H, 'Step lines of DeepSeek API prices per million tokens from November 2024 to November 2026 for the default model and the premium model', b)


for fn in (timeline, families, sizes, dense_vs_moe, release_pipeline, matrix, licenses, infra, prices):
    fn()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_part1.json'), 'w'))
print(f'figs_part1: {len(F)} figures: {", ".join(F)}')
