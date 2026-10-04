"""Figures for Part 6 (impact, limits, summary), drawn from results/part6.json. Writes results/figs_part6.json."""
import json
from figlib import svg, text, box, arrow, esc
from bertfig import token, row, line

R = json.load(open('results/part6.json'))
F = {}


def timeline():
    # dates: first arXiv version (the month in the arXiv id); the Search post 25 Oct 2019; the NAACL award June 2019
    ev = [
        ('Jun 2017', 'Transformer', 'the architecture BERT is built from', 'box'),
        ('Oct 2018', 'BERT', 'this paper: deep bidirectional pre-training', 'box-on'),
        ('Jun 2019', 'BERT wins NAACL best long paper', 'the conference where it was published', 'box-3'),
        ('Jun 2019', 'XLNet', 'bidirectional context without [MASK]', 'box-1'),
        ('Jul 2019', 'RoBERTa', 'same model, trained better; drops NSP', 'box-1'),
        ('Jul 2019', 'SpanBERT', 'masks whole spans of words', 'box-1'),
        ('Aug 2019', 'Sentence-BERT', 'turns BERT into a sentence-embedding model', 'box-1'),
        ('Sep 2019', 'ALBERT', 'far fewer parameters; sentence order instead of NSP', 'box-1'),
        ('Oct 2019', 'DistilBERT', '40% smaller, 60% faster, keeps 97%', 'box-1'),
        ('Oct 2019', 'BERT in Google Search', '"one in 10 searches in the U.S. in English"', 'box-3'),
        ('Mar 2020', 'ELECTRA', 'learns from every token, not just the masked 15%', 'box-1'),
        ('Jun 2020', 'DeBERTa', 'separate vectors for content and position', 'box-1'),
        ('Dec 2024', 'ModernBERT', '8192-token inputs, 2 trillion training tokens', 'box-1'),
    ]
    b, x = [], 130
    b.append(line(x, 20, x, 20 + 30 * len(ev) - 14, 'edge-dim'))
    for k, (d, name, note, c) in enumerate(ev):
        y = 26 + k * 30
        b += [text(x - 18, y + 4, d, 't-tick', 'end'), box(x - 6, y - 6, 12, 12, c, 3),
              text(x + 20, y + 5, name, 't-note'), text(410, y + 5, note, 't-muted')]
    b.append(text(20, 30 * len(ev) + 28, 'Dates are first arXiv versions, except the award (NAACL, June 2019) and the Search post (25 October 2019).', 't-muted'))
    return svg(760, 30 * len(ev) + 40, 'A timeline from the Transformer in 2017 to ModernBERT in 2024: BERT in October 2018, then a wave of follow-up models in 2019 and 2020.', b)


F['p6_timeline'] = timeline()


def similarity():
    rows = R['similarity']
    methods = [('cls', 'raw BERT [CLS]'), ('mean', 'raw BERT, mean of tokens'), ('st', 'all-MiniLM-L6-v2 (fine-tuned)')]
    L, Rr, T = 220, 720, 50
    X = lambda v: L + (v + 0.2) / 1.2 * (Rr - L)
    b = [text(20, 24, 'Cosine similarity of six sentence pairs', 't-title')]
    for v in [-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        b += [f'<line class="grid" x1="{X(v):.1f}" y1="{T}" x2="{X(v):.1f}" y2="{T + 3 * 56}"/>', text(X(v), T + 3 * 56 + 18, f'{v:.1f}', 't-tick', 'middle')]
    for k, (key, name) in enumerate(methods):
        yy = T + 28 + k * 56
        b.append(text(L - 14, yy + 4, name, 't-tick', 'end'))
        for r in rows:
            c = 's3' if r['kind'] == 'related' else 's2'
            dy = -7 if r['kind'] == 'related' else 7
            b.append(f'<g class="mark"><title>{esc(r["a"])} | {esc(r["b"])}: {r[key]:.3f}</title><circle class="{c} ring" cx="{X(r[key]):.1f}" cy="{yy + dy}" r="6"/></g>')
    ly = T + 3 * 56 + 40
    b += [f'<circle class="s3" cx="230" cy="{ly}" r="6"/>', text(242, ly + 4, 'related pair', 't-tick'),
          f'<circle class="s2" cx="350" cy="{ly}" r="6"/>', text(362, ly + 4, 'unrelated pair', 't-tick'),
          text(480, ly + 4, 'cosine similarity (1 = same direction)', 't-muted')]
    return svg(760, ly + 16, 'Cosine similarities for three related and three unrelated sentence pairs. With raw BERT the two groups overlap. The model fine-tuned for similarity separates them cleanly.', b)


F['p6_similarity'] = similarity()


def signal():
    words = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'looked', 'out', 'of', 'the', 'window', 'at', 'the', 'birds']
    W, G = 42, 5
    b = [text(20, 24, 'BERT (masked LM): only the hidden tokens give a training signal', 't-title'),
         text(20, 134, 'ELECTRA (replaced token detection): every token gives a signal', 't-title')]
    masked = {2, 9}
    cls = ['box-mask' if i in masked else 'box' for i in range(len(words))]
    shown = ['[M]' if i in masked else w for i, w in enumerate(words)]
    parts, cx = row(20, 40, shown, W, G, classes=cls, tcls='t-muted')
    b += parts
    for i, x in enumerate(cx):
        if i in masked:
            b += [arrow(x, 72, x, 90), text(x, 106, 'loss', 't-on', 'middle')]
    rep = {5: 'rug'}
    shown2 = [rep.get(i, w) for i, w in enumerate(words)]
    cls2 = ['box-2' if i in rep else 'box' for i in range(len(words))]
    parts, cx = row(20, 150, shown2, W, G, classes=cls2, tcls='t-muted')
    b += parts
    for i, x in enumerate(cx):
        b += [arrow(x, 182, x, 198), text(x, 214, 'fake' if i in rep else 'real', 't-bad' if i in rep else 't-s3', 'middle')]
    b.append(text(20, 240, 'The paper masks 15% of tokens (about 2 of these 15). ELECTRA asks "real or replaced?" at every position.', 't-muted'))
    return svg(760, 252, 'Where the training signal comes from. In BERT only the masked positions are scored. In ELECTRA every position is scored as real or replaced.', b)


F['p6_signal'] = signal()


def napkin():
    b = [text(20, 24, '1. Input (Part 2)', 't-title'), text(20, 150, '2. Pre-training, two tasks at once (Part 3)', 't-title'),
         text(20, 330, '3. Fine-tuning: same body, one new layer per task (Part 4)', 't-title')]
    toks = ['[CLS]', 'my', 'dog', 'is', '[MASK]', '[SEP]', 'he', 'likes', 'play', '##ing', '[SEP]']
    W, G = 58, 6
    cls = ['box-on', 'box-1', 'box-1', 'box-1', 'box-mask', 'box', 'box-2', 'box-2', 'box-2', 'box-2', 'box']
    parts, cx = row(20, 40, toks, W, G, classes=cls, tcls='t-muted')
    b += parts
    b += [text(20, 92, 'each input vector = token embedding + segment embedding (A or B) + position embedding', 't-tick'),
          text(20, 110, 'blue = sentence A, orange = sentence B; WordPiece splits "playing" into play + ##ing', 't-muted')]
    # encoder
    b += [box(20, 166, 700, 50, 'box-1', 10), text(370, 188, 'Transformer encoder: 12 layers, 768 numbers per token, 12 heads (BERT-base)', 't-note', 'middle'),
          text(370, 206, 'every token looks at every token, left and right, in every layer', 't-tick', 'middle')]
    for x in (cx[0], cx[4]):
        b.append(arrow(x, 218, x, 244))
    b += [box(cx[0] - 29, 246, 92, 30, 'box-on', 6), text(cx[0] + 17, 266, 'C → NSP', 't-tick', 'middle'),
          text(cx[0] + 70, 266, 'IsNext or NotNext?', 't-muted'),
          box(cx[4] - 46, 246, 92, 30, 'box-mask', 6), text(cx[4], 266, 'T → MLM', 't-tick', 'middle'),
          text(cx[4] + 52, 266, 'which word was hidden? ("cute")', 't-muted'),
          text(20, 298, 'loss = mean masked-LM loss + mean NSP loss; 15% of tokens are chosen for prediction (80% [MASK], 10% random, 10% kept)', 't-muted')]
    heads = [('classify', 'C · Wᵀ → label', 'GLUE, sentiment', 'box-3'),
             ('find a span', 'S · Tᵢ and E · Tⱼ', 'SQuAD', 'box-2'),
             ('tag tokens', 'Tᵢ → label', 'NER', 'box-4'),
             ('pick a choice', 'v · C, softmax', 'SWAG', 'box-on')]
    for i, (name, f, ex, c) in enumerate(heads):
        x = 20 + i * 178
        b += [box(x, 346, 166, 76, c, 10), text(x + 83, 368, name, 't-note', 'middle'), text(x + 83, 388, f, 't-mono', 'middle'),
              text(x + 83, 408, ex, 't-muted', 'middle')]
    return svg(760, 436, 'BERT on a napkin. Text goes in as WordPiece tokens with [CLS] and [SEP]. A 12-layer bidirectional encoder is pre-trained with masked LM and next sentence prediction. For each task, one small layer is added on top and everything is fine-tuned.', b)


F['p6_napkin'] = napkin()

json.dump(F, open('results/figs_part6.json', 'w'))
print('figures:', ', '.join(F))
