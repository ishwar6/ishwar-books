# ---------------------------------------------------------------- figures added in the second pass (illustrated style)
from alammar import vec, matrix, attn_grid, frame, brace
from figlib import esc
import math as _m

M1 = json.load(open('results/part1_math.json'))


def understand_vs_generate():
    b = [text(190, 24, 'Understanding: read, then decide', 't-title', 'middle'),
         text(570, 24, 'Generation: write something new', 't-title', 'middle'),
         line(380, 40, 380, 300, 'edge-dim')]
    # language, understanding
    b += [text(20, 58, 'text', 't-muted')]
    b += token(20, 66, 340, 'The battery died after two days.', 'box', 32, 't-tick')
    b += [arrow(190, 100, 190, 122), box(110, 124, 160, 36, 'box-1', 10), text(190, 147, 'BERT (encoder)', 't-note', 'middle'),
          arrow(190, 162, 190, 184)]
    b += token(60, 186, 120, 'negative', 'box-2', 28, 't-note') + token(200, 186, 120, 'product: battery', 'box-3', 28, 't-tick')
    b += [text(190, 232, 'a label, a span, a tag per word', 't-muted', 'middle')]
    # language, generation
    b += [text(400, 58, 'prompt', 't-muted')]
    b += token(400, 66, 340, 'Write a review of this phone:', 'box', 32, 't-tick')
    b += [arrow(570, 100, 570, 122), box(490, 124, 160, 36, 'box-2', 10), text(570, 147, 'GPT (decoder)', 't-note', 'middle'),
          arrow(570, 162, 570, 184)]
    words = ['The', 'battery', 'is', 'weak', '...']
    x = 418
    for k, w in enumerate(words):
        ww = 18 + 8 * len(w)
        b += token(x, 186, ww, w, 'box-4' if k < 4 else 'box-ghost', 28, 't-tick')
        x += ww + 6
    b += [text(570, 232, 'new text, one word at a time', 't-muted', 'middle')]
    # the image analogy
    b += [text(20, 270, 'Same split for images:', 't-note'),
          text(190, 292, 'recognise it: photo → "cat"', 't-tick', 'middle'),
          text(570, 292, 'draw it: "a cat" → new photo', 't-tick', 'middle')]
    return svg(760, 306, 'Understanding tasks read a text and output a decision about it, such as a label or a span; BERT is built for these. Generation tasks write new text one word at a time; GPT is built for these. Images have the same split: recognising a cat in a photo versus drawing a new picture of a cat.', b)


F['p1_understand_gen'] = understand_vs_generate()


def enc_dec():
    b = [text(190, 22, 'Left half: the encoder', 't-title', 'middle'), text(570, 22, 'Right half: the decoder', 't-title', 'middle')]

    def stack(x, y, w, parts, n_label):
        out = [box(x, y, w, 30 + 46 * len(parts), 'box-ghost', 14)]
        for k, (name, cls, note) in enumerate(parts):
            yy = y + 16 + (len(parts) - 1 - k) * 46
            out += [box(x + 16, yy, w - 32, 34, cls, 8), text(x + w / 2, yy + 15, name, 't-note', 'middle'),
                    text(x + w / 2, yy + 29, note, 't-muted', 'middle')]
        out.append(text(x + w + 8, y + 20, n_label, 't-tick'))
        return out
    b += stack(80, 50, 220, [('self-attention', 'box-1', 'every word sees every word'), ('feed-forward', 'box', 'each word on its own')], '× N')
    b += stack(450, 50, 240, [('masked self-attention', 'box-2', 'a word sees only its left'), ('cross-attention', 'box-ghost', 'looks at the encoder output'),
                              ('feed-forward', 'box', 'each word on its own')], '× N')
    b += [arrow(190, 192, 190, 172), text(190, 208, 'input words', 't-tick', 'middle'),
          arrow(570, 238, 570, 218), text(570, 254, 'words written so far', 't-tick', 'middle'),
          line(300, 120, 450, 120, 'edge-dim', 'ah')]
    b += [box(60, 270, 260, 52, 'box-1', 12), text(190, 292, 'BERT keeps only this half', 't-note', 'middle'),
          text(190, 310, 'no mask: both directions', 't-muted', 'middle'),
          box(430, 270, 280, 52, 'box-2', 12), text(570, 292, 'GPT keeps only this half', 't-note', 'middle'),
          text(570, 310, 'without the cross-attention; mask stays', 't-muted', 'middle')]
    return svg(760, 336, 'The original Transformer has two halves. The encoder (left) uses self-attention where every word sees every word. The decoder (right) uses masked self-attention where a word sees only the words on its left, plus cross-attention to the encoder. BERT keeps only the encoder half; GPT keeps only the decoder half, without the cross-attention.', b)


F['p1_enc_dec'] = enc_dec()


def kid_masks():
    w = ['the', 'kid', 'smiles']
    b = []
    g, gw, gh = attn_grid(40, 80, w, w, lambda i, j: j <= i, cell=52, title='GPT: causal (masked)', cls_fn=lambda i, j: 's2')
    b += g
    g, gw2, gh = attn_grid(420, 80, w, w, lambda i, j: True, cell=52, title='BERT: bidirectional', cls_fn=lambda i, j: 's1')
    b += g
    b += [text(40, 280, 'Row "kid": GPT lets it read "the" and itself; "smiles" is blocked.', 't-tick'),
          text(40, 300, 'Row "kid" in BERT: it reads "the", itself and "smiles".', 't-tick'),
          text(40, 320, 'Blocked squares (×) are the upper triangle: keys to the right of the query.', 't-muted')]
    return svg(760, 336, 'Attention masks for the sentence "the kid smiles". Rows are queries (the word doing the looking), columns are keys (the word being looked at). In GPT the upper triangle is blocked, so each word sees only itself and the words before it. In BERT all nine squares are allowed.', b)


F['p1_kid_masks'] = kid_masks()


def kid_weights():
    h = M1['head']
    a = M1['avg']
    b = []
    g, gw, gh = attn_grid(20, 84, h['toks'], h['toks'], lambda i, j: j <= i, cell=56, values=h['weights'], show_values=True,
                          title=f'GPT-2, layer 1, head {h["head"]}', cls_fn=lambda i, j: 's2')
    b += g
    bt = a['bert_toks']
    g, gw2, gh2 = attn_grid(330, 84, bt, bt, lambda i, j: True, cell=56, values=a['bert'], show_values=True,
                            title='BERT, layer 1, average of 12 heads')
    b += g
    b += [text(20, 300, 'every row adds up to 1', 't-muted'), text(20, 318, 'grey × = weight forced to 0 by the mask', 't-muted')]
    return svg(760, 400, 'Real attention weights for "the kid smiles". Left: one head of GPT-2 layer 1, recomputed by hand; the upper triangle is exactly zero. Right: BERT layer 1, averaged over its 12 heads, including the [CLS] and [SEP] tokens; every square has some weight.', b)


F['p1_kid_weights'] = kid_weights()


def layers():
    """Three panels, two layers each: who feeds the middle word's top vector."""
    xs = [0, 1, 2]
    b = []
    panels = [(20, 'BERT', 'every layer mixes both sides'), (275, 'OpenAI GPT', 'every layer sees only the left'), (530, 'ELMo', 'two separate towers, joined at the top')]
    for px, name, note in panels:
        b += [text(px + 105, 24, name, 't-title', 'middle'), text(px + 105, 42, note, 't-muted', 'middle')]
        cx = [px + 35 + 70 * i for i in xs]
        ys = [272, 196, 120]          # input, layer 1, layer 2
        words = ['the', 'kid', 'smiles']
        if name != 'ELMo':
            for L in range(2):
                for i in xs:
                    for j in xs:
                        if name == 'OpenAI GPT' and j > i:
                            continue
                        on = (L == 1 and i == 1) or (L == 0 and (name == 'BERT' or i <= 1))
                        b.append(line(cx[j], ys[L] - 14, cx[i], ys[L + 1] + 14, 'edge-on' if on else 'edge'))
            for L in range(1, 3):
                for i in xs:
                    b += token(cx[i] - 26, ys[L] - 14, 52, f'L{L}', 'box-1' if name == 'BERT' else 'box-2', 28, 't-tick')
        else:
            for i in xs:
                cxf, cxb = cx[i] - 14, cx[i] + 14
                for L in range(1, 3):
                    b += token(cxf - 12, ys[L] - 12, 24, '→', 'box-1', 24, 't-tick') + token(cxb - 12, ys[L] - 12, 24, '←', 'box-2', 24, 't-tick')
                b.append(line(cx[i], ys[0] - 14, cxf, ys[1] + 12, 'edge'))
                b.append(line(cx[i], ys[0] - 14, cxb, ys[1] + 12, 'edge'))
                b.append(line(cxf, ys[1] - 12, cxf, ys[2] + 12, 'edge-1'))
                b.append(line(cxb, ys[1] - 12, cxb, ys[2] + 12, 'edge-2'))
                for L in (1, 2):
                    if i < 2:      # forward: arc over the top of the row, left to right
                        nx = cx[i + 1] - 14
                        b.append(f'<path class="edge-1" d="M{cxf:.1f},{ys[L] - 12:.1f} Q{(cxf + nx) / 2:.1f},{ys[L] - 40:.1f} {nx:.1f},{ys[L] - 13:.1f}" marker-end="url(#ah)"/>')
                    if i > 0:      # backward: arc under the row, right to left
                        nx = cx[i - 1] + 14
                        b.append(f'<path class="edge-2" d="M{cxb:.1f},{ys[L] + 12:.1f} Q{(cxb + nx) / 2:.1f},{ys[L] + 40:.1f} {nx:.1f},{ys[L] + 13:.1f}" marker-end="url(#ah)"/>')
            b += token(cx[1] - 40, 54, 80, 'concat', 'box-on', 26, 't-tick')
            b += [line(cx[1] - 14, ys[2] - 12, cx[1] - 14, 78, 'edge-1'), line(cx[1] + 14, ys[2] - 12, cx[1] + 14, 78, 'edge-2')]
        for i in xs:
            b += token(cx[i] - 30, ys[0] - 14, 60, words[i], 'box-mask' if i == 1 else 'box', 28, 't-tick')
    b += [text(380, 322, 'Highlighted lines: what feeds "kid" at the top. Only BERT mixes the left and the right inside the layers.', 't-tick', 'middle')]
    return svg(760, 336, 'Three ways to build the top vector of the word "kid" with two layers. BERT: each layer reads every position of the layer below, so information from "smiles" reaches "kid" inside every layer. GPT: each layer reads only positions on the left. ELMo: a left-to-right tower and a right-to-left tower run separately and their top vectors are only concatenated at the end.', b)


F['p1_layers'] = layers()


def token_tasks():
    b = [text(20, 22, 'Named entity recognition: one tag per token', 't-title')]
    words = ['Barack', 'Obama', 'was', 'born', 'in', 'Hawaii', ',', 'United', 'States']
    tags = ['B-PER', 'I-PER', 'O', 'O', 'O', 'B-LOC', 'O', 'B-LOC', 'I-LOC']
    W = [72, 70, 46, 52, 36, 70, 26, 70, 70]
    x = 20
    for w, t, ww in zip(words, tags, W):
        c = 'box-2' if 'PER' in t else 'box-3' if 'LOC' in t else 'box'
        b += token(x, 36, ww, w, c, 30, 't-tick') + [arrow(x + ww / 2, 68, x + ww / 2, 84)] + token(x, 86, ww, t, c if t != 'O' else 'box-ghost', 24, 't-muted' if t == 'O' else 't-tick')
        x += ww + 6
    b += brace(20, 162, 124, 'a person', 'edge-2') + brace(368, 440, 124, 'a place', 'edge-3') + brace(476, 622, 124, 'a place', 'edge-3')
    b += [text(20, 182, 'Question answering: point at the answer inside the passage', 't-title')]
    b += token(20, 196, 330, 'Q: Where was Barack Obama born?', 'box-1', 30, 't-tick')
    pw = ['Barack', 'Obama', 'was', 'born', 'in', 'Honolulu', ',', 'Hawaii', '.']
    PW = [64, 62, 40, 46, 30, 80, 22, 64, 22]
    x = 20
    xs = []
    for k, (w, ww) in enumerate(zip(pw, PW)):
        b += token(x, 240, ww, w, 'box-on' if 5 <= k <= 7 else 'box', 30, 't-tick')
        xs.append((x, x + ww))
        x += ww + 6
    b += [text(xs[5][0] + 2, 290, '▲ start', 't-on'), text(xs[7][1] - 2, 290, 'end ▲', 't-on', 'end'),
          text(500, 260, 'answer = the span', 't-tick'), text(500, 276, '"Honolulu, Hawaii"', 't-note')]
    return svg(760, 304, 'Two token-level tasks. Named entity recognition gives every token a tag: B-PER and I-PER mark the beginning and inside of a person name, B-LOC and I-LOC a place, O means outside any name. Question answering marks a start token and an end token in the passage; the answer is the span between them.', b)


F['p1_token_tasks'] = token_tasks()


def selfsup():
    b = []
    sent = ['the', 'kid', 'smiles', 'at', 'the', 'dog']
    b += frame(10, 10, 740, 64, 1, 'Start with plain text. Nobody labels anything.')
    parts, _ = row(372, 34, sent, 54, 6)
    b += parts
    b += frame(10, 86, 740, 106, 2, 'Language model (GPT): input = the left side, label = the next word')
    for k, n in enumerate([2, 3]):
        y = 118 + k * 36
        parts, cx = row(40, y, sent[:n], 56, 6, cls='box-2')
        b += parts + [arrow(40 + n * 62, y + 15, 40 + n * 62 + 30, y + 15)] + token(40 + n * 62 + 34, y, 70, sent[n], 'box-on', 30, 't-note')
        b += [text(372, y + 20, f'P({sent[n]} | {" ".join(sent[:n])})', 't-tick')]
    b += frame(10, 204, 740, 106, 3, 'Masked LM (BERT): input = text with a hole, label = the hidden word')
    for k, hide in enumerate([2, 5]):
        y = 236 + k * 36
        ws = [('[MASK]' if i == hide else w) for i, w in enumerate(sent)]
        parts, cx = row(40, y, ws, 56, 6, classes=['box-mask' if i == hide else 'box-1' for i in range(6)])
        b += parts + [arrow(412, y + 15, 442, y + 15)] + token(446, y, 70, sent[hide], 'box-on', 30, 't-note')
        b += [text(530, y + 20, 'both sides are input', 't-muted')]
    b += [text(380, 330, 'The labels come from the text itself, so every sentence ever written is training data.', 't-tick', 'middle')]
    return svg(760, 342, 'Self-supervised learning in three frames. Start from plain text. A language model turns it into (left side, next word) pairs. A masked language model hides a word and asks for it back, using both sides. In both cases the label was already in the text.', b)


F['p1_selfsup'] = selfsup()


def what_updates():
    b = [text(130, 22, 'Feature-based', 't-title', 'middle'), text(380, 22, 'Not BERT: only the top', 't-title', 'middle'),
         text(630, 22, 'BERT fine-tuning', 't-title', 'middle')]
    names = ['embeddings', 'layer 1', 'layer 2', '...', 'layer 12']
    cols = [(30, 'frozen', 'frozen', 'box-ghost'), (280, 'frozen', 'top', 'box-ghost'), (530, 'all', 'all', 'box-1')]
    for x, mode, _, _ in cols:
        for k, n in enumerate(names):
            y = 262 - k * 38
            upd = mode == 'all' or (mode == 'top' and k == 4)
            if mode == 'frozen':
                upd = False
            if x == 280:
                upd = k == 4
            b += token(x, y, 200, n, 'box-1' if upd else 'box-ghost', 30, 't-tick' if upd else 't-muted')
            b.append(text(x + 196, y + 20, 'updated' if upd else 'frozen', 't-s3' if upd else 't-muted', 'end'))
        top = 262 - 4 * 38 - 44
        if x == 30:
            b += token(x, top, 200, 'task model (BiLSTM ...)', 'box-3', 34, 't-note') + [text(x + 100, top - 8, 'trained from zero', 't-s3', 'middle')]
        else:
            b += token(x + 50, top, 100, 'new layer', 'box-3', 34, 't-note') + [text(x + 100, top - 8, 'trained from zero', 't-s3', 'middle')]
    b += [line(380 - 120, 86, 380 + 120, 286, 'bad-line'), line(380 + 120, 86, 380 - 120, 286, 'bad-line')]
    b += [text(130, 316, 'ELMo-style', 't-muted', 'middle'), text(380, 316, 'a common misreading', 't-bad', 'middle'),
          text(630, 316, 'all 110M weights + the new layer', 't-s3', 'middle')]
    return svg(760, 330, 'Which weights change. Feature-based: the pre-trained model is frozen and a separate task model is trained. Updating only the top layer is a common misreading and is not what BERT does. BERT fine-tuning updates every pre-trained weight, from the embeddings to layer 12, together with the new output layer.', b)


F['p1_what_updates'] = what_updates()


def transfer():
    b = [text(20, 22, 'Transfer learning: learn once on a big general dataset, reuse for a small specific one', 't-title')]
    rows_ = [('Vision', 'ImageNet, 1.2M labelled photos', 'image network', 'chest X-ray: normal or not?'),
             ('Language (BERT, 2018)', 'Wikipedia + books, 3.3B words', 'BERT', 'support email: billing or bug?')]
    for k, (name, data, model, task) in enumerate(rows_):
        y = 50 + k * 104
        b += [text(20, y + 14, name, 't-note')]
        b += token(20, y + 26, 220, data, 'box', 44, 't-tick')
        b += [arrow(244, y + 48, 276, y + 48)]
        b += token(280, y + 26, 130, model, 'box-1', 44, 't-note')
        b += [text(345, y + 86, 'pre-train', 't-muted', 'middle'), arrow(414, y + 48, 446, y + 48)]
        b += token(450, y + 26, 80, model if k else 'network', 'box-1', 44, 't-tick') + token(532, y + 26, 30, '+', 'box-3', 44, 't-note')
        b += [text(505, y + 86, 'fine-tune on a few', 't-muted', 'middle'), text(505, y + 100, 'thousand examples', 't-muted', 'middle'),
              arrow(566, y + 48, 590, y + 48)]
        b += [text(596, y + 44, task.split(':')[0] + ':', 't-tick'), text(596, y + 62, task.split(': ')[1], 't-tick')]
    return svg(760, 264, 'Transfer learning in vision and in language. An image network pre-trained on ImageNet photos is reused, with a new last layer, for a small task like reading chest X-rays. BERT does the same for text: pre-train on Wikipedia and books, then fine-tune with a new layer on a small labelled task.', b)


F['p1_transfer'] = transfer()


def chain():
    rows_ = M1['chain']['rows']
    b = [text(20, 22, 'P("The kid smiles at the dog.") built one word at a time, left side only (GPT-2)', 't-title')]
    lo = -11.0
    for k, r in enumerate(rows_):
        y = 44 + k * 34
        b += [text(20, y + 20, r['left'] if r['left'] != '(start)' else '(start)', 't-muted')]
        b += [arrow(196, y + 15, 216, y + 15)]
        b += token(220, y, 74, r['word'], 'box-2', 28, 't-note')
        wbar = 300 * (r['logp'] - lo) / (0 - lo)
        b.append(f'<g class="mark"><title>P({esc(r["word"])} | {esc(r["left"])}) = {r["p"]}</title><rect class="s2" x="310" y="{y + 6}" width="{wbar:.1f}" height="16" rx="3" style="fill-opacity:0.8"/></g>')
        p = r['p']
        b.append(text(316 + wbar, y + 19, f'{p:.6f}'.rstrip('0') if p < 0.001 else f'{p:.4f}', 't-val'))
    y = 44 + len(rows_) * 34 + 8
    c = M1['chain']
    b += [line(20, y, 740, y, 'edge-dim'),
          text(20, y + 22, f'multiply the seven:  P(sentence) = {float(c["p_sentence"]):.2e}', 't-note'),
          text(20, y + 42, f'add the logs:  sum of log P = {c["sum_logp"]}   (the same number, safe from underflow)', 't-tick'),
          text(310, 36, 'bar length = log P (longer = more likely)', 't-muted')]
    return svg(760, y + 56, 'The chain rule on a real model. GPT-2 gives each token a probability given only the tokens on its left. Multiplying the seven probabilities gives the probability of the whole sentence, about 2.2 in a million billion; adding their logarithms gives the same information as -33.752.', b)


F['p1_chain'] = chain()


def smiles_two_ways():
    m = M1['masked']
    b = [text(20, 22, 'Predicting "smiles" with one side versus with both sides', 't-title')]
    b += token(20, 40, 340, 'GPT-2 sees:  The kid ____', 'box-2', 30, 't-tick')
    b += token(400, 40, 340, 'BERT sees:  the kid [MASK] at the dog.', 'box-1', 30, 't-tick')
    b += hbars(20, 100, [(w, p) for w, p in m['top_gpt']], 220, cls='s2', label_w=70, fmt=lambda v: f'{v:.3f}', vmax=0.35)
    b += hbars(400, 100, [(w, p) for w, p in m['top_bert']], 220, cls='s1', label_w=70, fmt=lambda v: f'{v:.3f}', vmax=0.35)
    b += [text(20, 92, 'top 5 guesses', 't-muted'), text(400, 92, 'top 5 guesses', 't-muted'),
          text(20, 238, f'P(smiles) = {m["p_gpt"]:.4f}', 't-tick'), text(400, 238, f'P(smiles) = {m["p_bert"]:.4f}', 't-tick'),
          text(20, 260, 'guesses fit "The kid ..." (who, is, was)', 't-muted'),
          text(400, 260, 'every guess fits "___ at the dog" (a verb + at)', 't-muted')]
    return svg(760, 276, 'The same position predicted two ways. GPT-2 sees only "The kid" and guesses words like who, is and was. BERT sees "at the dog" on the right and guesses verbs that take "at": looked, stared, glanced, pointed, glared.', b)


F['p1_smiles'] = smiles_two_ways()
