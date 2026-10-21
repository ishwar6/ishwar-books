"""Figures for the Background page (before and after BERT). Writes results/figs_partb.json. CSS classes only.
Numbers come from results/partb.json (our run, and the scores and sizes copied there from the papers)."""
import json, math, os
from figlib import svg, text, box, esc, heatmap
from bertfig import token, hbars

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'partb.json')))
RA = json.load(open(os.path.join(HERE, 'results', 'partb_after.json')))
F = {}


def timeline():
    """From counting words to ModernBERT. Years are the years of first publication."""
    ev = [
        ('1948', 'n-grams (Shannon)', 'guess the next letter or word from counts', 'box-1'),
        ('1997', 'LSTM', 'a recurrent network with gates that keep a memory', 'box-1'),
        ('2003', 'Neural language model', 'learn a vector for each word while predicting', 'box-1'),
        ('2013', 'word2vec', 'cheap word vectors from 1.6 billion words', 'box-1'),
        ('2014', 'seq2seq', 'read a sentence into one vector, write another', 'box-1'),
        ('2014', 'Attention', 'look back at every input word, not one vector', 'box-1'),
        ('2017', 'Transformer', 'attention only, no recurrence', 'box-1'),
        ('2018', 'ULMFiT, ELMo, GPT', 'pre-train a language model, then reuse it', 'box-1'),
        ('2018', 'GLUE', 'one score over nine understanding tasks', 'box-1'),
        ('2018', 'BERT', 'deep bidirectional pre-training: fill in the blanks', 'box-on'),
        ('2019', 'GPT-2, XLNet, RoBERTa, ALBERT', 'bigger, trained longer, new objectives', 'box-2'),
        ('2019', 'SuperGLUE, T5', 'harder tests; every task as text-to-text', 'box-2'),
        ('2020', 'Scaling laws, GPT-3', 'loss falls smoothly with size; tasks from prompts', 'box-2'),
        ('2021', 'DeBERTa', 'first single model above humans on SuperGLUE', 'box-2'),
        ('2024', 'ModernBERT', 'an encoder for search: 8192 tokens, 2 trillion tokens', 'box-2'),
        ('2025', 'NeoBERT, mmBERT', 'deeper encoders; 1800 languages in one model', 'box-2'),
        ('2025', 'LLaDA', 'fill-in-the-blanks becomes a way to write text', 'box-2'),
        ('2025', 'Ettin', 'encoders and decoders trained the same way, compared', 'box-2'),
        ('2026', 'UltraBERT', 'two masking ratios in one encoder', 'box-2'),
    ]
    W, rh, x = 720, 34, 70
    H = 60 + rh * len(ev) + 20
    b = [text(20, 26, 'From counting words to BERT, and after', 't-title'),
         f'<line class="edge-dim" x1="{x}" y1="48" x2="{x}" y2="{48 + rh * len(ev) - 10}"/>']
    for i, (yr, name, what, cls) in enumerate(ev):
        y = 48 + i * rh
        b.append(text(x - 12, y + 16, yr, 't-tick', 'end'))
        b.append(f'<circle class="{"box-on" if cls == "box-on" else "s1" if cls == "box-1" else "s2"}" cx="{x}" cy="{y + 12}" r="5"/>')
        b += token(x + 16, y, 250, name, cls, 24, 't-tick t-strong')
        b.append(text(x + 280, y + 16, what, 't-tick'))
    y = 48 + rh * len(ev) + 14
    b += [text(x + 16, y, 'before BERT', 't-s1'), text(x + 140, y, 'BERT', 't-on'), text(x + 210, y, 'after BERT', 't-s2')]
    return svg(W, H, 'A timeline from n-gram models in 1948 through LSTM, word2vec, seq2seq, attention, the Transformer, '
               'ELMo, ULMFiT and GPT to BERT in 2018, then GPT-2, RoBERTa, T5, GPT-3, DeBERTa, ModernBERT, NeoBERT, mmBERT, '
               'LLaDA, Ettin and UltraBERT in 2026.', b)


def progress():
    """GLUE (top) and SuperGLUE (bottom) scores over time, with the human baselines. Shared time axis."""
    W, L, Rr, PH, GAP = 720, 56, 20, 170, 46
    pw = W - L - Rr
    x0, x1 = 2018.2, 2021.25
    X = lambda y, m: L + ((y + (m - 1) / 12) - x0) / (x1 - x0) * pw
    H = 56 + 2 * PH + GAP + 60
    b = [text(20, 24, 'Benchmark scores over time (numbers from the papers)', 't-title')]
    # label placement per point: (dx, dy, anchor); None = listed at the side instead
    place = {('glue', 'BiLSTM + ELMo + Attn'): (8, 16, 'start'), ('glue', 'OpenAI GPT'): (8, 14, 'start'),
             ('glue', 'BERT-large'): (8, 14, 'start'), ('superglue', 'BERT-large'): (8, 4, 'start'),
             ('superglue', 'T5-11B'): (-8, -8, 'end'), ('superglue', 'DeBERTa-1.5B'): (-8, 18, 'end')}
    for k, (key, title, lc, dc, tc, hum, lo, hi) in enumerate((
            ('glue', 'GLUE: nine tasks (2018)', 'l1', 's1', 't-s1', 87.1, 66, 93),
            ('superglue', 'SuperGLUE: eight harder tasks (2019)', 'l2', 's2', 't-s2', 89.8, 66, 93))):
        T = 62 + k * (PH + GAP)
        Y = lambda v: T + PH - (v - lo) / (hi - lo) * PH
        b.append(text(L, T - 8, title, tc + ' t-strong'))
        for t in (70, 80, 90):
            b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, str(t), 't-tick', 'end')]
        for yr in (2019, 2020, 2021):
            b.append(f'<line class="grid" x1="{X(yr, 1):.1f}" y1="{T}" x2="{X(yr, 1):.1f}" y2="{T + PH}"/>')
        b.append(f'<line class="axis" x1="{L}" y1="{T + PH}" x2="{L + pw}" y2="{T + PH}"/>')
        b.append(f'<line class="{lc}" stroke-dasharray="5 4" x1="{L}" y1="{Y(hum):.1f}" x2="{L + pw}" y2="{Y(hum):.1f}"/>')
        b.append(text(L + 6, Y(hum) - 6, f'human baseline {hum}', tc))
        rows = R[key]
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(y, m):.1f},{Y(v):.1f}' for y, m, _, v, _ in rows) + '"/>')
        side = []
        for y, m, name, v, src in rows:
            b.append(f'<g class="mark"><title>{esc(name)}: {v} ({esc(src)})</title>'
                     f'<circle class="{dc} ring" cx="{X(y, m):.1f}" cy="{Y(v):.1f}" r="5"/></g>')
            p = place.get((key, name))
            if p:
                b.append(text(X(y, m) + p[0], Y(v) + p[1], f'{name} {v}', 't-tick', p[2]))
            else:
                side.append(f'{name} {v} ({["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"][m - 1]} {y})')
        for j, lab in enumerate(side):
            b.append(text(X(2020, 2), Y(80) + j * 16, lab, 't-tick'))
        if side:
            b.append(text(X(2020, 2), Y(80) - 18, 'the four points at the top, left to right:', 't-muted'))
    ya = 62 + 2 * PH + GAP + 18
    for yr in (2019, 2020, 2021):
        b.append(text(X(yr, 1), ya, str(yr), 't-tick', 'middle'))
    return svg(W, H, 'Two panels over time. GLUE rises from 70.0 in April 2018 through GPT 72.8 and BERT 80.5 to T5 90.3, '
               'passing the human baseline of 87.1 in 2019. SuperGLUE rises from BERT 69.0 to DeBERTa 89.9, just above '
               'its human baseline of 89.8.', b)


def sizes():
    """The sizes the GPT-3 paper lists, plus GPT-3 itself, on a log scale."""
    pts = [('GPT', 2018, 1e8), ('BERT', 2018, 3e8), ('GPT-2', 2019, 1.5e9), ('Megatron', 2019, 8e9),
           ('T5', 2019, 1.1e10), ('Turing-NLG', 2020, 1.7e10), ('GPT-3', 2020, 1.75e11)]
    W, H, L, T, B = 720, 320, 70, 40, 56
    pw, ph = W - L - 40, H - T - B
    Y = lambda n: T + ph - (math.log10(n) - 8) / (11.5 - 8) * ph
    X = lambda i: L + 30 + i * (pw - 60) / (len(pts) - 1)
    b = [text(20, 24, 'Model size, as the GPT-3 paper lists it (parameters, log scale)', 't-title')]
    for e, lab in ((8, '100M'), (9, '1B'), (10, '10B'), (11, '100B')):
        b += [f'<line class="grid" x1="{L}" y1="{Y(10 ** e):.1f}" x2="{L + pw}" y2="{Y(10 ** e):.1f}"/>',
              text(L - 8, Y(10 ** e) + 4, lab, 't-tick', 'end')]
    b.append(f'<polyline class="l1" points="' + ' '.join(f'{X(i):.1f},{Y(n):.1f}' for i, (_, _, n) in enumerate(pts)) + '"/>')
    for i, (name, yr, n) in enumerate(pts):
        lab = f'{n / 1e9:g}B' if n >= 1e9 else f'{n / 1e6:g}M'
        b.append(f'<circle class="{"s2" if name == "BERT" else "s1"} ring" cx="{X(i):.1f}" cy="{Y(n):.1f}" r="5"/>')
        b.append(text(X(i), Y(n) - 12, lab, 't-tick t-strong', 'middle'))
        b.append(text(X(i), T + ph + 20, name, 't-tick', 'middle'))
        b.append(text(X(i), T + ph + 38, str(yr), 't-muted', 'middle'))
    return svg(W, H, 'Seven models from 2018 to 2020 on a log scale: GPT 100 million, BERT 300 million, GPT-2 1.5 billion, '
               'Megatron 8 billion, T5 11 billion, Turing-NLG 17 billion and GPT-3 175 billion parameters.', b)


def rnn():
    """Gradient size against distance from the end, log scale, from our run."""
    g = R['rnn_all'][::-1]                                 # index = distance from the end
    W, H, L, T, B = 720, 300, 70, 40, 50
    pw, ph = W - L - 30, H - T - B
    lo, hi = -28, 1
    X = lambda d: L + d / (len(g) - 1) * pw
    Y = lambda v: T + ph - (math.log10(max(v, 1e-30)) - lo) / (hi - lo) * ph
    b = [text(20, 24, 'How much of the gradient reaches each earlier word (plain RNN, our run)', 't-title')]
    for e in (0, -5, -10, -15, -20, -25):
        b += [f'<line class="grid" x1="{L}" y1="{Y(10 ** e):.1f}" x2="{L + pw}" y2="{Y(10 ** e):.1f}"/>',
              text(L - 8, Y(10 ** e) + 4, f'1e{e}' if e else '1', 't-tick', 'end')]
    for d in (0, 10, 20, 30, 40, 50, 59):
        b.append(text(X(d), T + ph + 20, str(d), 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    b.append(f'<polyline class="l1" points="' + ' '.join(f'{X(d):.1f},{Y(v):.1f}' for d, v in enumerate(g)) + '"/>')
    b.append(text(L + pw / 2, H - 8, 'distance from the last word (steps back)', 't-tick', 'middle'))
    return svg(W, H, 'On a log scale, the gradient falls in a nearly straight line from about 3 at the last word to about '
               '1e-28 at 59 steps back.', b)


def bigram():
    bg = R['bigram']
    items = [('unigram', bg['ppl_uni']), ('bigram, add-one', bg['ppl_bi']), ('bigram, mixed', bg['ppl_int'])]
    W = 720
    b = hbars(20, 60, items, 440, 34, 's1', 170, fmt=lambda v: f'{v:,.0f}', vmax=9000,
              title='Perplexity on WikiText-2 test (lower is better), our run', hi='bigram, mixed')
    b.append(text(20, 180, f'{100 * bg["unseen"]:.1f}% of the word pairs in the test text never appear in training.', 't-tick'))
    return svg(W, 196, f'Three bars: unigram perplexity {bg["ppl_uni"]:,.0f}, bigram with add-one smoothing '
               f'{bg["ppl_bi"]:,.0f}, bigram mixed with unigram {bg["ppl_int"]:,.0f}.', b)


def bank():
    names = ['river', 'money', 'loan', 'stream']
    M = [[1.0 if i == j else R['bank'].get(f'{names[min(i, j)]}-{names[max(i, j)]}') for j in range(4)] for i in range(4)]
    rows = ['bank of the river', 'money in the bank', 'bank approved my loan', 'bank of the stream']
    b = heatmap(20, 70, M, rows, ['river', 'money', 'loan', 'stream'], 56, title='"bank" in BERT\'s last layer: cosine similarity',
                show_values=True, label_w=190)
    b.append(text(20, 372, 'In the input table (and in word2vec) all four cells are 1.000: one vector for every use.', 't-tick'))
    return svg(720, 388, 'A four by four grid of cosine similarities. River and stream: 0.899. Money and loan: 0.787. '
               'Across the two meanings: 0.400 to 0.487.', b)



def mask():
    """How much of the text each model hides in training, 2018 to 2026 (from the papers, via partb_after.json)."""
    rows = [('BERT', '2018', [15]), ('RoBERTa', '2019', [15]), ('ModernBERT', '2024', [30]), ('NeoBERT', '2025', [20]),
            ('mmBERT', '2025', 'down'), ('LLaDA', '2025', 'range'), ('UltraBERT', '2026', [5, 50])]
    W, L, T, rh = 720, 150, 50, 34
    pw = W - L - 40
    X = lambda v: L + v / 100 * pw
    H = T + rh * len(rows) + 50
    b = [text(20, 26, 'How much of the text is hidden during training', 't-title')]
    for v in (0, 15, 30, 50, 100):
        b += [f'<line class="grid" x1="{X(v):.1f}" y1="{T - 6}" x2="{X(v):.1f}" y2="{T + rh * len(rows)}"/>',
              text(X(v), T + rh * len(rows) + 18, f'{v}%', 't-tick', 'middle')]
    for i, (name, yr, r) in enumerate(rows):
        y = T + i * rh + 14
        b += [text(L - 60, y + 4, name, 't-tick t-strong', 'end'), text(L - 12, y + 4, yr, 't-muted', 'end')]
        if r == 'range':
            b += [f'<line class="l2" x1="{X(0):.1f}" y1="{y}" x2="{X(100):.1f}" y2="{y}" stroke-width="6"/>',
                  text(X(50), y - 8, 'a new random share for every sequence', 't-tick', 'middle')]
        elif r == 'down':
            b.append(f'<line class="l1" x1="{X(30):.1f}" y1="{y}" x2="{X(5):.1f}" y2="{y}"/>')
            for v in (30, 15, 5):
                b.append(f'<circle class="s1" cx="{X(v):.1f}" cy="{y}" r="6"/>')
            b.append(text(X(30) + 12, y + 4, '30%, then 15%, then 5% (one stage each)', 't-tick'))
        else:
            cls = 'box-on' if name == 'BERT' else 's2' if name == 'UltraBERT' else 's1'
            for v in r:
                b += [f'<circle class="{cls}" cx="{X(v):.1f}" cy="{y}" r="6"/>', text(X(v) + 12, y + 4, f'{v}%', 't-tick')]
            if name == 'UltraBERT':
                b.append(text(X(50) + 50, y + 4, 'both in every batch, one output layer each', 't-tick'))
    return svg(W, H, 'Masking ratio by model: BERT 15%, RoBERTa 15%, ModernBERT 30%, NeoBERT 20%, mmBERT 30% then 15% then 5%, '
               'LLaDA any share from 0 to 100%, UltraBERT both 5% and 50%.', b)


def write():
    """BERT writes eight words by filling the most confident blank each step (our run)."""
    st = RA['write']['steps']
    W, T, rh = 720, 50, 30
    H = T + rh * len(st) + 20
    b = [text(20, 26, f'BERT fills 8 blanks after "{RA["write"]["prompt"]}" (our run)', 't-title')]
    for i, s in enumerate(st):
        y = T + i * rh
        b.append(text(20, y + 17, f'step {s["step"]}', 't-muted'))
        words = s['text'].split()
        new = s['filled']
        x = 90
        prev = st[i - 1]['text'].split() if i else ['_'] * len(words)
        for j, w in enumerate(words):
            cls = 'box-on' if prev[j] == '_' and w != '_' else 'box-1' if w != '_' else 'box-2'
            b += token(x, y, 64, '' if w == '_' else w, cls, 22, 't-tick t-strong')
            x += 70
        b.append(text(x + 6, y + 17, f'p = {s["conf"]:.2f}', 't-tick'))
    return svg(W, H, 'Eight rows. Each row fills one more blank: first the full stop, then not, to, a, for, while, be and '
               'alone, ending with: not to be alone for a while.', b)


F['pb_timeline'] = timeline()
F['pb_progress'] = progress()
F['pb_sizes'] = sizes()
F['pb_rnn'] = rnn()
F['pb_bigram'] = bigram()
F['pb_bank'] = bank()
F['pb_mask'] = mask()
F['pb_write'] = write()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_partb.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
