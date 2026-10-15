"""Figures for the background page (the road to SAM), partly from results/road.json.
Writes results/figs_partb.json (key -> svg string). Colours come only from CSS classes."""
import json, os
from figlib import svg, text, box, arrow
from samfig import line

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'road.json')))
F = {}


def legend(x, y, items):
    b = []
    for cls, lab in items:
        b += [f'<rect class="{cls}" x="{x}" y="{y - 9}" width="10" height="10" rx="2"/>', text(x + 16, y, lab, 't-tick')]
        x += 16 + 7.2 * len(lab) + 22
    return b


# ---------------------------------------------------------------- 1. timeline with YOLO
def timeline():
    ev = [('2015', 'FCN', ['a class for', 'every pixel'], 's1'),
          ('2016', 'YOLO', ['boxes and names,', 'in one pass'], 's1'),
          ('2017', 'Mask R-CNN', ['a mask for', 'every object'], 's1'),
          ('2020', 'GPT-3', ['one model, many', 'tasks, by prompt'], 's2'),
          ('2021', 'CLIP', ['pictures and text,', 'any name'], 's2'),
          ('2023', 'SAM', ['any object,', 'by prompt'], 's3')]
    xs = [75, 197, 319, 441, 563, 685]
    b = [line(40, 96, 735, 96, 'axis', 'ah')]
    for (yr, name, ls, cls), x in zip(ev, xs):
        b += [f'<circle class="{cls}" cx="{x}" cy="96" r="8"/>', text(x, 40, yr, 't-title', 'middle'), text(x, 64, name, 't-note', 'middle')]
        for k, l in enumerate(ls):
            b.append(text(x, 124 + 17 * k, l, 't-muted', 'middle'))
    b += legend(40, 190, [('s1', 'segmentation and detection'), ('s2', 'models that take a prompt'), ('s3', 'SAM: the two roads meet')])
    return svg(760, 204, 'Timeline: FCN 2015, YOLO 2016 and Mask R-CNN 2017 find and segment objects from a fixed list. GPT-3 2020 and CLIP 2021 take prompts. SAM 2023 joins the two.', b)


F['road_timeline'] = timeline()


# ---------------------------------------------------------------- 2. the question each model answers
def questions():
    rows = [('Classification', 'What is in this picture?', '"cat"'),
            ('Detection (YOLO)', 'Where are the objects of my list?', 'boxes + names'),
            ('Segmentation (FCN, Mask R-CNN)', 'Which pixels belong to each object of my list?', 'masks + names'),
            ('Promptable segmentation (SAM)', 'Which pixels belong to the thing that I show you?', 'masks, no names')]
    b = [text(20, 24, 'Model type', 't-tick'), text(255, 24, 'The question it answers', 't-tick'), text(630, 24, 'Output', 't-tick')]
    for k, (a, q, o) in enumerate(rows):
        y = 40 + k * 46
        cls = 'box-on' if k == 3 else 'box-1'
        b += [box(14, y, 762, 36, cls, 8), text(26, y + 23, a, 't-note'), text(255, y + 23, q, 't-note'), text(630, y + 23, o, 't-note')]
    b.append(text(20, 238, 'From top to bottom, the answer becomes more exact. Only the last row has no fixed list of object types.', 't-muted'))
    return svg(790, 252, 'Four types of model and the question that each one answers. Classification: what is in the picture. Detection: where are the objects of my list. Segmentation: which pixels belong to each object of my list. SAM: which pixels belong to the thing that I show you.', b)


F['road_questions'] = questions()


# ---------------------------------------------------------------- 3. YOLO and SAM side by side, with the real numbers
def yolo_sam():
    y, s = R['yolo'], R['sam']
    b = [text(20, 26, 'YOLO11n-seg', 't-title'), text(400, 26, 'SAM ViT-H', 't-title')]
    # YOLO row
    b += [box(20, 44, 100, 54, 'box-2', 8), text(70, 76, 'picture', 't-note', 'middle'), arrow(124, 71, 150, 71),
          box(154, 44, 90, 54, 'box-1', 8), text(199, 68, 'one', 't-note', 'middle'), text(199, 86, 'network', 't-note', 'middle'),
          arrow(248, 71, 274, 71), box(278, 36, 100, 70, 'box-on', 8),
          text(328, 60, 'boxes', 't-note', 'middle'), text(328, 77, 'names', 't-note', 'middle'), text(328, 94, 'masks', 't-note', 'middle')]
    # SAM row
    b += [box(400, 36, 100, 30, 'box-2', 8), text(450, 56, 'picture', 't-note', 'middle'),
          box(400, 74, 100, 30, 'box-2', 8), text(450, 94, 'your prompt', 't-note', 'middle'),
          arrow(504, 51, 530, 66), arrow(504, 89, 530, 76), box(534, 44, 90, 54, 'box-1', 8),
          text(579, 68, 'encoder +', 't-note', 'middle'), text(579, 86, 'decoder', 't-note', 'middle'),
          arrow(628, 71, 654, 71), box(658, 44, 90, 54, 'box-on', 8),
          text(703, 68, 'a mask +', 't-note', 'middle'), text(703, 86, 'a score', 't-note', 'middle')]
    facts = [('object types', f'{y["classes"]}, fixed', 'no list'),
             ('weights', f'{y["params"] / 1e6:.1f} million', f'{s["params"] / 1e6:.0f} million'),
             ('time on our laptop', f'{y["ms"]["mps"]:.0f} ms for all objects', f'{s["enc_ms"] / 1000:.1f} s, then 6 ms a prompt')]
    for k, (lab, a, c) in enumerate(facts):
        yy = 140 + k * 26
        b += [text(20, yy, lab, 't-tick'), text(160, yy, a, 't-note'), text(400, yy, lab, 't-tick'), text(540, yy, c, 't-note')]
    return svg(760, 214, 'YOLO takes only the picture and gives boxes, names and masks for 80 fixed object types in about 10 milliseconds. SAM takes the picture and a prompt and gives a mask and a score, with no list of types; its encoder takes about 3 seconds on our laptop.', b)


F['road_yolo_sam'] = yolo_sam()

json.dump(F, open(os.path.join(HERE, 'results', 'figs_partb.json'), 'w'))
print(f'{len(F)} figures:', ', '.join(F))
