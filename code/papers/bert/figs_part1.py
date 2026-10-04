"""Figures for Part 1 (the big idea), drawn from results/part1.json. Writes results/figs_part1.json."""
import json
from figlib import svg, text, box, arrow
from bertfig import token, row, line, hbars

R = json.load(open('results/part1.json'))
F = {}


def two_steps():
    b = [text(20, 26, 'Step 1: pre-training (once, very expensive)', 't-title'),
         text(430, 26, 'Step 2: fine-tuning (per task, cheap)', 't-title')]
    # a pile of unlabeled text
    for i in range(4):
        b.append(box(30 + i * 6, 52 + i * 6, 130, 90, 'box', 6))
    b += [text(101, 96, 'Wikipedia', 't-note', 'middle'), text(101, 116, '+ books', 't-note', 'middle'),
          text(101, 136, 'no labels', 't-muted', 'middle'),
          text(101, 178, '3.3 billion words', 't-tick', 'middle'),
          arrow(178, 100, 228, 100),
          box(232, 70, 130, 64, 'box-1', 12), text(297, 98, 'BERT', 't-big', 'middle'), text(297, 118, 'learns language', 't-tick', 'middle'),
          text(297, 160, 'all the weights', 't-tick', 'middle'), text(297, 178, 'are learned here', 't-tick', 'middle')]
    tasks = [('sentiment', 'positive / negative', 'box-3'), ('question answering', 'find the answer span', 'box-2'), ('named entities', 'person / place / ...', 'box-4')]
    for i, (name, out, cls) in enumerate(tasks):
        y = 48 + i * 74
        b += [arrow(364, 102, 444, y + 26), box(448, y, 104, 52, 'box-1', 10), text(500, y + 23, 'BERT copy', 't-note', 'middle'),
              text(500, y + 41, 'starts pre-trained', 't-muted', 'middle'),
              box(552, y + 8, 26, 36, cls, 6), arrow(580, y + 26, 606, y + 26),
              text(612, y + 21, name, 't-note'), text(612, y + 39, out, 't-tick')]
    b += [text(565, 278, '+ one small output layer per task', 't-tick', 'middle')]
    return svg(760, 292, 'BERT is pre-trained once on unlabeled text, then a copy is fine-tuned for each task with one small extra output layer.', b)


F['p1_two_steps'] = two_steps()


def strategies():
    b = [text(190, 24, 'Feature-based (ELMo)', 't-title', 'middle'), text(570, 24, 'Fine-tuning (OpenAI GPT, BERT)', 't-title', 'middle')]
    # feature-based: frozen model gives features to a separate task model
    b += [box(60, 190, 260, 30, 'box', 6), text(190, 210, 'your sentence', 't-tick', 'middle'), arrow(190, 188, 190, 164),
          box(60, 116, 260, 46, 'box-ghost', 10), box(60, 116, 260, 46, 'box-1', 10), text(190, 137, 'pre-trained model', 't-note', 'middle'),
          text(190, 154, 'frozen: its weights never change', 't-muted', 'middle'),
          arrow(190, 114, 190, 92), text(200, 104, 'features (vectors)', 't-muted'),
          box(60, 44, 260, 46, 'box-2', 10), text(190, 65, 'task-specific model', 't-note', 'middle'), text(190, 82, 'designed by hand, trained from zero', 't-muted', 'middle')]
    # fine-tuning: the whole model is trained with a tiny layer on top
    b += [box(440, 190, 260, 30, 'box', 6), text(570, 210, 'your sentence', 't-tick', 'middle'), arrow(570, 188, 570, 164),
          box(440, 98, 260, 64, 'box-1', 10), text(570, 124, 'pre-trained model', 't-note', 'middle'),
          text(570, 143, 'every weight is updated a little', 't-muted', 'middle'),
          arrow(570, 96, 570, 80), box(520, 50, 100, 28, 'box-3', 8), text(570, 69, 'tiny layer', 't-tick', 'middle'),
          text(570, 40, 'the only new part', 't-muted', 'middle')]
    b += [line(380, 40, 380, 222, 'edge-dim')]
    return svg(760, 236, 'Two ways to reuse a pre-trained model. Feature-based: the model is frozen and only produces input features for a separate task model. Fine-tuning: the whole model keeps training on the task, with one tiny new layer.', b)


F['p1_strategies'] = strategies()


def context():
    words = ['went', 'to', 'the', '?', 'to', 'deposit', 'my']
    W, G, x0 = 62, 8, 230
    panels = [('Left-to-right (OpenAI GPT)', 'the blank sees only the words before it', 'ltr'),
              ('Two one-way readers, joined (ELMo)', 'each reader sees one side; they meet only at the end', 'elmo'),
              ('Deeply bidirectional (BERT)', 'the blank sees both sides, in every layer', 'bi')]
    b = []
    for k, (title, note, kind) in enumerate(panels):
        y0 = 20 + k * 150
        b += [text(20, y0 + 6, title, 't-title'), text(20, y0 + 24, note, 't-tick')]
        classes = ['box'] * len(words)
        classes[3] = 'box-mask'
        parts, cx = row(x0, y0 + 96, words, W, G, classes=classes)
        t = cx[3]
        top = y0 + 96
        if kind == 'elmo':
            lx, rx = (cx[0] + t) / 2, (t + cx[6]) / 2
            b += [box(lx - 60, y0 + 50, 120, 24, 'box-1', 6), text(lx, y0 + 66, 'left reader', 't-tick', 'middle'),
                  box(rx - 60, y0 + 50, 120, 24, 'box-2', 6), text(rx, y0 + 66, 'right reader', 't-tick', 'middle'),
                  box(t - 31, y0 + 12, 62, 24, 'box-on', 6), text(t, y0 + 28, 'join', 't-muted', 'middle'),
                  line(lx + 40, y0 + 50, t - 12, y0 + 36, 'edge-1'), line(rx - 40, y0 + 50, t + 12, y0 + 36, 'edge-2')]
            for i, x in enumerate(cx):
                if i <= 3:
                    b.append(line(x, top - 2, lx + (x - lx) * 0.5, y0 + 74, 'edge-1'))
                if i >= 3:
                    b.append(line(x, top - 2, rx + (x - rx) * 0.5, y0 + 74, 'edge-2'))
        else:
            ty = y0 + 46
            b += [box(t - 31, ty, 62, 24, 'box-on', 6), text(t, ty + 16, 'layer', 't-muted', 'middle')]
            if kind == 'bi':
                b.append(text(t + 44, ty + 16, 'the same in all 12 layers', 't-muted'))
            for i, x in enumerate(cx):
                if kind == 'ltr' and i > 3:
                    b.append(text(x, top + 46, 'hidden', 't-muted', 'middle'))
                    continue
                b.append(line(x, top - 2, t + (x - t) * 0.15, ty + 24, 'edge-1' if kind == 'ltr' else 'edge-on'))
        b += parts
    return svg(760, 470, 'What the blank can see. A left-to-right model sees only the earlier words. ELMo runs a left reader and a right reader separately and joins their outputs at the end. BERT lets every layer look at both sides.', b)


F['p1_context'] = context()


def fill():
    r = R['fill'][0]
    g = [(w, p) for w, p in r['gpt2']]
    bb = [(w, p) for w, p in r['bert']]
    b = [text(20, 24, 'I went to the ____ to deposit my paycheck.', 't-title')]
    b += hbars(20, 66, g, 230, title='GPT-2 (sees "I went to the")', cls='s2', label_w=80)
    b += hbars(400, 66, bb, 230, title='BERT (sees both sides)', cls='s1', label_w=70, hi='bank')
    return svg(760, 200, 'Top five guesses for the blank. GPT-2, reading left to right, has no way to know about the paycheck and spreads its guesses. BERT sees "to deposit my paycheck" and puts 0.901 on "bank".', b)


F['p1_fill'] = fill()


def tasks():
    b = [text(20, 24, 'Sentence-level task: one answer for the whole input', 't-title'),
         text(20, 168, 'Token-level task: one answer for every token', 't-title')]
    b += [box(20, 40, 330, 30, 'box', 6), text(185, 60, 'A man is playing a guitar.', 't-tick', 'middle'),
          box(20, 78, 330, 30, 'box', 6), text(185, 98, 'A person is making music.', 't-tick', 'middle'),
          arrow(354, 74, 420, 74), box(424, 56, 150, 36, 'box-3', 8), text(499, 79, 'entailment', 't-note', 'middle'),
          text(590, 70, 'the second sentence', 't-muted'), text(590, 86, 'follows from the first', 't-muted')]
    words = ['Ada', 'Lovelace', 'was', 'born', 'in', 'London']
    labels = ['PERSON', 'PERSON', 'none', 'none', 'none', 'PLACE']
    cls = ['box-2', 'box-2', 'box', 'box', 'box', 'box-4']
    parts, cx = row(20, 186, words, 96, 10)
    b += parts
    for x, l, c in zip(cx, labels, cls):
        b += [arrow(x, 218, x, 236)] + token(x - 44, 238, 88, l, c, 28, 't-muted' if l == 'none' else 't-note')
    return svg(760, 280, 'Two kinds of tasks. Sentence-level tasks give one answer per input, like whether one sentence follows from another. Token-level tasks give an answer for every token, like marking which words are names of people or places.', b)


F['p1_tasks'] = tasks()

json.dump(F, open('results/figs_part1.json', 'w'))
print('figures:', ', '.join(F))
