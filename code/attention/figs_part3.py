"""Figures for Part 3, drawn from results/part3.json and results/part3_qwen.json."""
import json, math, os
from figlib import svg, text, box, arrow, bar_path, lines_chart

R = json.load(open('results/part3.json'))
Q = json.load(open('results/part3_qwen.json')) if os.path.exists('results/part3_qwen.json') else None
F = {}


def grid(x0, y0, n, cell, allowed, title):
    b = [text(x0 + n * cell / 2, y0 - 12, title, 't-title', 'middle')]
    for i in range(n):
        for j in range(n):
            x, y = x0 + j * cell, y0 + i * cell
            if j > i:
                b.append(f'<rect class="cell-masked" x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="2"/>')
            else:
                op = 0.9 if allowed(i, j) else 0.07
                b.append(f'<rect class="cell" x="{x + 1}" y="{y + 1}" width="{cell - 2}" height="{cell - 2}" rx="2" style="fill-opacity:{op}"/>')
    return b


def masks():
    n, c = 12, 17
    import random
    rnd = random.Random(3)                                    # an illustrative top-k choice: first token, itself, 2 others
    picks = {i: {0, i} | set(rnd.sample(range(i + 1), min(2, i + 1))) for i in range(n)}
    b = grid(20, 44, n, c, lambda i, j: True, 'full (causal)')
    b += grid(272, 44, n, c, lambda i, j: i - j < 4, 'sliding window, W = 4')
    b += grid(524, 44, n, c, lambda i, j: j in picks[i], 'sparse: top-k picked')
    b += [text(20, 270, 'Each row is one token; each bright square is a token it may look at. Faint squares are skipped. Dark empty squares are the future.', 't-tick')]
    return svg(760, 284, 'Three attention patterns on 12 tokens: full causal attention, a sliding window of 4, and a sparse top-k selection.', b)
F['p3_masks'] = masks()


def receptive():
    n, W, levels = 12, 3, 4
    names = ['input tokens', 'after layer 1', 'after layer 2', 'after layer 3']
    xs = lambda j: 150 + j * 50
    ys = lambda lv: 230 - lv * 62
    b = []
    for lv in range(levels):
        reach = 1 + (levels - 1 - lv) * (W - 1)        # how many positions at this level feed the last token at the top
        b.append(text(20, ys(lv) + 5, names[lv], 't-tick'))
        for j in range(n):
            on = j >= n - reach if lv < levels - 1 else j == n - 1
            b.append(f'<circle class="{"s1" if on else "node"}" cx="{xs(j)}" cy="{ys(lv)}" r="{9 if on else 6}"/>')
            if lv > 0 and on:
                for jj in range(max(0, j - W + 1), j + 1):
                    b.append(f'<line class="edge" x1="{xs(j)}" y1="{ys(lv) + 9}" x2="{xs(jj)}" y2="{ys(lv - 1) - 9}"/>')
    b += [text(150, 268, 'Window W = 3. Each layer reaches W − 1 = 2 more tokens back: 1 → 3 → 5 → 7 tokens.', 't-tick'),
          text(150, 286, 'So after L layers, a token can be influenced by L × (W − 1) + 1 tokens.', 't-tick')]
    return svg(760, 300, 'With a sliding window, each layer can only look W tokens back, but stacking layers lets information travel further: the reach grows by W minus 1 per layer.', b)
F['p3_receptive'] = receptive()


def ring():
    W, now = 4, 9
    b = [text(20, 26, f'Writing token {now} with a cache of W = {W} slots', 't-title')]
    for i in range(now + 1):
        x = 30 + i * 70
        kept = i > now - W
        b += [box(x, 44, 56, 32, 'box-1' if i == now else ('box-3' if kept else 'box'), 6),
              text(x + 28, 65, f'tok {i}', 't-tick', 'middle')]
        if not kept:
            b.append(f'<line class="edge" x1="{x + 8}" y1="60" x2="{x + 48}" y2="60"/>')
    b.append(text(30, 100, 'tokens 0 to 5: overwritten (crossed out); tokens 6 to 9: still in the cache', 't-tick'))
    for s in range(W):
        x = 150 + s * 120
        holder = max(i for i in range(now + 1) if i % W == s)
        b += [box(x, 140, 100, 52, 'box-1' if holder == now else 'box-3', 8),
              text(x + 50, 162, f'slot {s}', 't-note', 'middle'), text(x + 50, 182, f'holds tok {holder}', 't-tick', 'middle')]
        b.append(f'<line class="edge" x1="{30 + holder * 70 + 28}" y1="78" x2="{x + 50}" y2="138" marker-end="url(#ah)"/>')
    b += [text(150, 222, f'Rule: token i goes into slot i mod {W}. Token 9 → slot 1, overwriting token 5.', 't-tick'),
          text(150, 240, 'The cache never grows past W slots, no matter how long the text gets.', 't-tick')]
    return svg(760, 254, 'A rolling buffer cache: token i is written into slot i mod W, so the cache always holds only the last W tokens.', b)
F['p3_ring'] = ring()


def gemma_layers():
    b = [text(20, 24, 'Gemma 3 27B: 62 layers, 5 local (sliding window 1,024) then 1 global, repeated', 't-title')]
    for i in range(62):
        glob = (i + 1) % 6 == 0
        x = 20 + i * 11.6
        b.append(f'<g class="mark"><title>layer {i + 1}: {"global (sees every token)" if glob else "local (last 1,024 tokens)"}</title>'
                 f'<rect class="{"s2" if glob else "s1"}" x="{x:.1f}" y="{40 if glob else 52}" width="9" height="{48 if glob else 24}" rx="2"/></g>')
    b += [f'<rect class="s1" x="20" y="112" width="12" height="12" rx="2"/>', text(38, 122, '52 local layers: keep only the last 1,024 tokens', 't-tick'),
          f'<rect class="s2" x="380" y="112" width="12" height="12" rx="2"/>', text(398, 122, '10 global layers: keep every token', 't-tick'),
          text(20, 150, 'layer 1', 't-tick'), text(740, 150, 'layer 62', 't-tick', 'end')]
    return svg(760, 162, 'The layer pattern of Gemma 3 27B: five sliding-window layers, then one global layer, repeated through 62 layers.', b)
F['p3_gemma'] = gemma_layers()

g = R['kv_memory']['gemma3_27b']
F['p3_kv'] = lines_chart(760, 300, 'KV cache memory of Gemma 3 27B for one sequence, with every layer global versus the real 5 local to 1 global pattern.',
                         g['context'], [('all layers global', g['all_global_gib'], 'l2', 's2'),
                                        ('5 local : 1 global', g['five_to_one_gib'], 'l1', 's1')],
                         'tokens in the conversation (log scale)', 'KV cache, GiB', 0, 64, [0, 16, 32, 48, 64], xlog=True,
                         xticks=[1024, 4096, 16384, 65536, 131072], fmt=lambda v: f'{v:.1f}' if v < 10 else f'{v:.0f}', right=190)

t = R['decode_timing']['results']
ns = [int(n) for n in t]
F['p3_decode'] = lines_chart(760, 300, 'Time for one decode step of attention as the conversation grows: full attention versus a 1,024-token sliding window.',
                             ns, [('full attention', [t[str(n)]['full_ms'] for n in ns], 'l2', 's2'),
                                  ('window 1,024', [t[str(n)]['window_ms'] for n in ns], 'l1', 's1')],
                             'tokens so far (log scale)', 'milliseconds per step', 0, 4, [0, 1, 2, 3, 4], xlog=True,
                             xticks=[1024, 4096, 16384, 65536, 131072], fmt=lambda v: f'{v:.2f}', right=190)


def dsa_diagram():
    b = [box(20, 112, 90, 44, 'box'), text(65, 132, 'token t', 't-note', 'middle'), text(65, 148, 'query', 't-tick', 'middle'),
         arrow(112, 134, 160, 134), box(160, 96, 150, 76, 'box-2', 10), text(235, 120, 'lightning indexer', 't-note', 'middle'),
         text(235, 140, 'few heads, small, FP8', 't-tick', 'middle'), text(235, 158, 'scores every earlier token', 't-tick', 'middle'),
         arrow(312, 134, 360, 134), box(360, 108, 110, 52, 'box', 10), text(415, 130, 'top-k', 't-note', 'middle'),
         text(415, 148, 'keep the best k', 't-tick', 'middle'),
         arrow(472, 134, 520, 134), box(520, 96, 130, 76, 'box-1', 10), text(585, 120, 'main attention', 't-note', 'middle'),
         text(585, 140, 'only over the', 't-tick', 'middle'), text(585, 158, 'k chosen tokens', 't-tick', 'middle'),
         arrow(652, 134, 690, 134), text(700, 139, 'output', 't-note'),
         text(235, 76, 'cheap, but looks at all L tokens', 't-tick', 'middle'), text(585, 76, 'expensive, but looks at only k', 't-tick', 'middle'),
         text(20, 214, 'DeepSeek-V3.2: k = 2,048. The main attention cost drops from L² to L × k; the indexer is still L², but tiny.', 't-tick')]
    return svg(760, 230, 'DeepSeek Sparse Attention: a small lightning indexer scores every earlier token, the top k are kept, and the main attention runs only over those.', b)
F['p3_dsa'] = dsa_diagram()


if Q:
    # perplexity bars: budget 64 and 256, four ways of choosing the tokens
    rows = []
    for k in ('64', '256'):
        rows += [(f'{k}: last {k} tokens', Q['window'][k]['window_only'], 's4'),
                 (f'{k}: 4 sinks + last {int(k) - 4}', Q['window'][k]['window_plus_4_sinks'], 's1'),
                 (f'{k}: indexer top-{k}', Q['dsa_ppl'][k], 's3')]
    rows.append(('full attention (2,048)', Q['full_ppl'], 's2'))
    W, L, Rm, T0, band = 780, 230, 200, 14, 34
    cap = 40.0
    b = []
    for i, (name, v, cls) in enumerate(rows):
        y = T0 + i * band
        bw = max(3, min(v, cap) / cap * (W - L - Rm))
        b.append(f'<g class="mark"><title>{name}: perplexity {v:.2f}</title><path class="{cls} bar-mark" d="{bar_path(L, y, bw, 20, horizontal=True)}"/></g>')
        b += [text(L - 10, y + 15, name, 't-tick', 'end'), text(L + bw + 8, y + 15, f'{v:,.1f}' + (', bar cut off' if v > cap else ''), 't-val')]
    b.append(f'<line class="axis" x1="{L}" y1="{T0 - 4}" x2="{L}" y2="{T0 + len(rows) * band}"/>')
    b.append(text(L, T0 + len(rows) * band + 22, 'perplexity on Pride and Prejudice, tokens 1,024 to 2,047 (lower is better)', 't-tick'))
    F['p3_ppl'] = svg(W, T0 + len(rows) * band + 34, 'Perplexity of Qwen2.5-0.5B when each token may look at only 64 or 256 earlier tokens, chosen in three different ways, against full attention.', b)

    c = Q['coverage_by_layer_64']
    F['p3_cov'] = lines_chart(760, 300, 'Share of the model\'s real attention that lands on the 64 chosen tokens, layer by layer, for the trained indexer and for 4 sinks plus a window.',
                              list(range(1, len(c['indexer']) + 1)),
                              [('indexer top-64', c['indexer'], 'l1', 's1'), ('4 sinks + window 60', c['sinks_window'], 'l2', 's2')],
                              'layer', 'share of attention captured', 0, 1, [0, 0.25, 0.5, 0.75, 1.0], xticks=[1, 6, 12, 18, 24],
                              fmt=lambda v: f'{v:.2f}', right=200)

    ex = Q['example']
    sel, top = set(ex['indexer_selected']), set(ex['attention_last_row_top'])
    Wd, x0, x1 = 760, 30, 730
    X = lambda p: x0 + p / (Q['T'] - 1) * (x1 - x0)
    b = [text(x0, 24, f'Layer {ex["layer"] + 1}, the last token (position 2,047): which earlier tokens matter?', 't-title')]
    for row, (label, s, cls) in enumerate([('top 64 tokens by the model\'s real attention', top, 's2'), ('64 tokens picked by the trained indexer', sel, 's1')]):
        y = 56 + row * 64
        b.append(text(x0, y - 8, label, 't-tick'))
        b.append(f'<line class="axis" x1="{x0}" y1="{y + 14}" x2="{x1}" y2="{y + 14}"/>')
        for p in sorted(s):
            b.append(f'<rect class="{cls}" x="{X(p) - 1.5:.1f}" y="{y}" width="3" height="28" rx="1"/>')
    for p in (0, 512, 1024, 1536, 2047):
        b.append(text(X(p), 196, f'{p:,}', 't-tick', 'middle'))
    b.append(text(x0, 222, f'Both rows agree on {len(sel & top)} of 64 tokens. Many picks are recent tokens (right edge) or the first token (left edge).', 't-tick'))
    F['p3_pick'] = svg(Wd, 236, 'For one real query, the tokens the model attends to most compared with the tokens the trained lightning indexer selected.', b)

json.dump(F, open('results/figs_part3.json', 'w'))
print(len(F), 'figures:', list(F))
