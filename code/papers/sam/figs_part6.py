"""Figures for Part 6 (impact, limits and summary), drawn from results/part6.json.
Writes results/figs_part6.json (key -> svg string). Colours come only from CSS classes."""
import json, os
from figlib import svg, text, box, arrow, esc, lines_chart
from samfig import hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'part6.json')))
F = {}
pct = lambda v: f'{v:.2f}' if isinstance(v, float) else f'{v}'


# ---------------------------------------------------------------- 1. the drawn wheel: recall and gap fill against spoke width
def wheel_chart():
    H, B = R['wheel']['ViT-H'], R['wheel']['ViT-B']
    xs = [r['width'] for r in H]
    series = [('ViT-H spoke recall', [r['recall'] for r in H], 'l1', 's1'),
              ('ViT-H gap fill', [r['gap_fill'] for r in H], 'l2', 's2'),
              ('ViT-B spoke recall', [r['recall'] for r in B], 'l3', 's3'),
              ('ViT-B gap fill', [r['gap_fill'] for r in B], 'l4', 's4')]
    return lines_chart(780, 320, 'The drawn wheel. Spoke recall (the share of spoke pixels inside the mask) and gap fill (the share of the empty '
                       'pixels between the spokes inside the mask) against the spoke width. For thin spokes both are near 0; from 4 to 6 pixels '
                       'both jump to near 1. The correct answer, recall 1 and gap fill 0, never appears.',
                       xs, series, 'spoke width (pixels, in a 1024 × 1024 picture)', 'share of pixels', 0, 1.0, [0, 0.25, 0.5, 0.75, 1.0],
                       xlog=True, xticks=xs, fmt=pct, right=200)


F['p6_wheel'] = wheel_chart()


# ---------------------------------------------------------------- 2. the teeth: recall and gap fill against tooth width
def teeth_chart():
    H = [r for r in R['comb']['ViT-H'] if r['t']]
    B = [r for r in R['comb']['ViT-B'] if r['t']]
    xs = [r['t'] for r in H]
    series = [('ViT-H tooth recall', [r['recall'] for r in H], 'l1', 's1'),
              ('ViT-H gap fill', [r['gap_fill'] for r in H], 'l2', 's2'),
              ('ViT-B tooth recall', [r['recall'] for r in B], 'l3', 's3'),
              ('ViT-B gap fill', [r['gap_fill'] for r in B], 'l4', 's4')]
    return lines_chart(780, 320, 'The edge with teeth. Tooth recall and gap fill against the tooth width. Up to 4 pixels the mask is a straight '
                       'line over the tips of the teeth (both near 1). At 8 pixels it starts to follow them. At 32 pixels it follows them, '
                       'but with rounded corners.',
                       xs, series, 'tooth width = gap width (pixels, in a 1024 × 1024 picture)', 'share of pixels', 0, 1.0,
                       [0, 0.25, 0.5, 0.75, 1.0], xlog=True, xticks=xs, fmt=pct, right=200)


F['p6_teeth'] = teeth_chart()


# ---------------------------------------------------------------- 3. stray pieces before and after the filters of Appendix B
def specks():
    C = R['components']
    b = [text(20, 24, 'Masks with stray pieces under 100 pixels (16 × 16 clicks per photo)', 't-title')]
    y = 52
    for iname, title in [('bike', 'bicycle photo'), ('cats', 'cats photo')]:
        for k in ('ViT-H', 'ViT-B'):
            c = C[iname][k]
            items = [('all masks', c['pct_small']),
                     ('IoU > 0.88', 100 * c['good_small'] / max(1, c['good'])),
                     ('+ stable', 100 * c['stable_small'] / max(1, c['stable']))]
            b.append(text(20, y + 10, f'{title}, {k}', 't-note'))
            b += hbars(20, y + 18, items, 420, row_h=24, cls='s1' if k == 'ViT-H' else 's3', label_w=110,
                       fmt=lambda v: f'{v:.0f}%', vmax=100)
            y += 18 + 3 * 24 + 18
    b.append(text(20, y + 4, '"+ stable": predicted IoU > 0.88 and IoU of the masks at logit +1 and −1 of at least 0.95 (Appendix B)', 't-tick'))
    return svg(620, y + 20, 'The share of masks with stray pieces under 100 pixels, for the bicycle and the cats photos, with ViT-H and ViT-B. '
               'Between 34 and 82 percent of the raw masks have them. The two filters of Appendix B lower the share, most for ViT-B, '
               'but they never remove it.', b)


F['p6_specks'] = specks()


# ---------------------------------------------------------------- 4. the limits and the papers that answered them
def family():
    rows = [('one picture at a time, no video', 'SAM 2 (2024)', 'a memory of past frames'),
            ('heavy image encoder, not real-time', 'MobileSAM (2023)', 'a small encoder copies ViT-H'),
            ('coarse edges, missed thin parts', 'HQ-SAM (2023)', 'one extra output token'),
            ('domain tools do better', 'MedSAM (2024)', 'fine-tuned on medical images'),
            ('text prompts exploratory, no semantic', 'SAM 3 (2025)', 'noun phrases find all instances')]
    b = [text(20, 24, 'A limit of Section 8', 't-title'), text(450, 24, 'A later paper, and its main idea', 't-title')]
    y = 40
    for lim, paper, idea in rows:
        b += [box(20, y, 360, 46, 'box', 8), text(36, y + 28, lim, 't-note')]
        b += [arrow(384, y + 23, 444, y + 23, on=True)]
        b += [box(448, y, 312, 46, 'box-on', 8), text(464, y + 20, paper, 't-note t-strong'), text(464, y + 37, idea, 't-tick')]
        y += 58
    return svg(780, y + 6, 'Five limits from Section 8 and five later papers. One picture at a time: SAM 2 adds a memory for video. '
               'Heavy encoder: MobileSAM trains a small encoder to copy ViT-H. Coarse edges: HQ-SAM adds one output token. '
               'Domain tools: MedSAM fine-tunes SAM on medical images. Text prompts: SAM 3 finds all instances of a noun phrase.', b)


F['p6_family'] = family()


# ---------------------------------------------------------------- 5. the whole paper in one picture
def summary():
    M = R['models']['ViT-H']
    cols = [('The task', 'Part 2', ['a prompt in,', 'a valid mask out', '(3 masks if ambiguous)']),
            ('The model', 'Part 3', [f'encoder {M["encoder"] / 1e6:.0f} M weights,', 'once per picture;', f'decoder {M["decoder"] / 1e6:.2f} M, per click']),
            ('The data engine', 'Part 4', ['assisted, semi-automatic,', 'then fully automatic:', '11 M pictures, 1.1 B masks']),
            ('The experiments', 'Part 5', ['zero-shot, 23 datasets', 'and 4 other tasks,', 'no extra training'])]
    b = []
    x = 14
    for i, (t, part, lines) in enumerate(cols):
        b += [box(x, 20, 176, 130, 'box-on' if i == 1 else 'box', 10), text(x + 88, 46, t, 't-title', 'middle'),
              text(x + 88, 66, part, 't-tick', 'middle')]
        for k, l in enumerate(lines):
            b.append(text(x + 88, 92 + 17 * k, l, 't-note', 'middle'))
        if i < 3:
            b.append(arrow(x + 178, 85, x + 196, 85))
        x += 196
    b.append(text(394, 176, 'The model makes the data; the data trains the model; the prompt lets one model do many tasks.', 't-tick', 'middle'))
    return svg(800, 190, 'The paper in one picture: the task (Part 2), the model (Part 3), the data engine and SA-1B (Part 4) and the '
               'zero-shot experiments (Part 5). The model makes the data, the data trains the model, and the prompt lets one model do many tasks.', b)


F['p6_summary'] = summary()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_part6.json'), 'w'), indent=1)
print('wrote results/figs_part6.json:', ', '.join(F))
