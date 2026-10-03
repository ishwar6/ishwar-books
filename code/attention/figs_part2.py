"""Figures for Part 2, drawn from results/part2.json."""
import json
from figlib import svg, text, box, arrow, bar_path

R = json.load(open('results/part2.json'))
F = {}


def sharing():
    b = []
    panels = [('MHA', 8, 8, 'own K, V per head'), ('GQA', 8, 2, 'groups share K, V'),
              ('MQA', 8, 1, 'all share one K, V')]
    for p, (name, q, kv, sub) in enumerate(panels):
        x0 = 12 + p * 192
        b.append(text(x0 + 88, 24, name, 't-title', 'middle'))
        for i in range(q):
            b.append(f'<circle class="s2" cx="{x0 + 18 + i * 21}" cy="56" r="7"/>')
        for j in range(kv):
            span = q // kv
            cx = x0 + 18 + (j * span + (span - 1) / 2) * 21
            wdt = max(14, span * 21 - 6)
            b.append(box(cx - wdt / 2, 112, wdt, 24, 'box-3', 5))
            if wdt >= 30:
                b.append(text(cx, 129, 'K V', 't-tick', 'middle'))
            for i in range(span):
                qi = j * span + i
                b.append(f'<line class="edge" x1="{x0 + 18 + qi * 21}" y1="63" x2="{cx}" y2="110"/>')
        b.append(text(x0 + 88, 160, sub, 't-tick', 'middle'))
        b.append(text(x0 + 88, 178, f'{kv} K/V head{"s" if kv > 1 else ""} cached', 't-note', 'middle'))
    # MLA panel
    x0 = 12 + 3 * 192
    b.append(text(x0 + 88, 24, 'MLA', 't-title', 'middle'))
    for i in range(8):
        b.append(f'<circle class="s2" cx="{x0 + 18 + i * 21}" cy="56" r="7"/>')
        b.append(f'<line class="edge" x1="{x0 + 18 + i * 21}" y1="63" x2="{x0 + 88}" y2="110"/>')
    b += [box(x0 + 40, 112, 96, 24, 'box-1', 5), text(x0 + 88, 129, 'latent c', 't-tick', 'middle'),
          text(x0 + 88, 160, 'K, V rebuilt from c', 't-tick', 'middle'),
          text(x0 + 88, 178, 'c + RoPE key cached', 't-note', 'middle')]
    b.append(text(12, 222, 'orange: query heads (computed fresh, never cached). Aqua and blue: what each token leaves in the KV cache.', 't-tick'))
    return svg(780, 236, 'MHA, GQA, MQA and MLA side by side: how many key/value heads each one stores per token.', b)
F['p2_sharing'] = sharing()


def kv_bars():
    s = R['kv_per_token_bytes']
    rows = [('Llama 2 7B', 'MHA, 32 KV heads', s['Llama 2 7B (MHA)'], 's2'),
            ('Llama 3.1 8B', 'GQA, 8 KV heads', s['Llama 3.1 8B (GQA, 8 KV heads)'], 's1'),
            ('DeepSeek-V3 (671B)', 'MLA, latent 512 + RoPE 64', s['DeepSeek-V3 (MLA)'], 's3'),
            ('Falcon 7B', 'MQA, 1 KV head', s['Falcon 7B (MQA)'], 's4')]
    W, L, Rm, T = 780, 220, 150, 16
    band = 50
    vmax = max(r[2] for r in rows) / 1024
    b = []
    for i, (name, sub, v, cls) in enumerate(rows):
        y = T + i * band + 12
        bw = max(3, v / 1024 / vmax * (W - L - Rm))
        b.append(f'<g class="mark"><title>{name} ({sub}): {v / 1024:.1f} KiB per token</title><path class="{cls} bar-mark" d="{bar_path(L, y, bw, 22, horizontal=True)}"/></g>')
        b += [text(L - 12, y + 10, name, 't-note', 'end'), text(L - 12, y + 26, sub, 't-tick', 'end'),
              text(L + bw + 8, y + 16, f'{v / 1024:,.1f} KiB', 't-val')]
    b.append(f'<line class="axis" x1="{L}" y1="{T}" x2="{L}" y2="{T + len(rows) * band + 6}"/>')
    return svg(W, T + len(rows) * band + 16, 'KV cache per token, in BF16, for four real models using MHA, GQA, MLA and MQA.', b)
F['p2_kv'] = kv_bars()


def speed():
    d = R['decode_speed']['results']
    keys = ['32', '8', '1']
    labels = {'32': '32 KV heads (MHA)', '8': '8 KV heads (GQA)', '1': '1 KV head (MQA)'}
    W, H, L, T, B = 780, 300, 64, 30, 60
    pw, ph = W - L - 30, H - T - B
    ymax = 3.5
    b = []
    for t in (0, 1, 2, 3):
        y = T + ph - t / ymax * ph
        b += [f'<line class="grid" x1="{L}" y1="{y:.1f}" x2="{L + pw}" y2="{y:.1f}"/>', text(L - 8, y + 4, f'{t} ms', 't-tick', 'end')]
    band = pw / 3
    for i, k in enumerate(keys):
        v = d[k]['ms']; mb = d[k]['kv_mb']
        bh = v / ymax * ph; x = L + band * i + band / 2 - 12
        b.append(f'<g class="mark"><title>{labels[k]}: {v:.2f} ms, reads {mb:.0f} MB of cache</title><path class="s1 bar-mark" d="{bar_path(x, T + ph - bh, 24, bh)}"/></g>')
        b += [text(x + 12, T + ph - bh - 8, f'{v:.2f} ms', 't-val', 'middle'),
              text(x + 12, T + ph + 20, labels[k], 't-tick', 'middle'), text(x + 12, T + ph + 38, f'reads {mb:,.0f} MB', 't-tick', 'middle')]
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    return svg(W, H, 'Time for one decode-step attention over a 4,096-token cache, with 32, 8 and 1 key/value heads.', b)
F['p2_speed'] = speed()


def mla_diagram():
    b = [box(20, 120, 70, 44, 'box'), text(55, 147, 'x', 't-math', 'middle'), text(55, 182, 'token', 't-tick', 'middle'),
         arrow(92, 132, 160, 74), box(160, 52, 120, 44, 'box-1'), text(220, 72, 'latent c', 't-note', 'middle'),
         text(220, 89, 'via W_dkv', 't-tick', 'middle'),
         arrow(92, 152, 160, 204), box(160, 182, 120, 44, 'box-1'), text(220, 202, 'RoPE key', 't-note', 'middle'),
         text(220, 219, 'via W_kr + RoPE', 't-tick', 'middle'),
         f'<rect x="150" y="36" width="140" height="206" rx="12" fill="none" class="drop"/>',
         text(220, 266, 'the whole KV cache', 't-note', 'middle')]
    for i, (lab, cls) in enumerate((('keys, head 1..h', 'box-2'), ('values, head 1..h', 'box-3'))):
        y = 40 + i * 64
        b += [arrow(282, 74, 360, y + 20), box(360, y, 150, 40, cls, 8), text(435, y + 25, lab, 't-tick', 'middle')]
    b += [text(436, 158, 'rebuilt by W_uk and W_uv', 't-tick', 'middle'),
          text(560, 60, 'Training: rebuild every', 't-tick'), text(560, 78, "head's K and V from c.", 't-tick'),
          text(560, 120, 'Inference: never rebuild.', 't-note'), text(560, 140, 'Fold W_uk into the query and', 't-tick'),
          text(560, 158, 'W_uv into the output, and', 't-tick'), text(560, 176, 'attend directly over c.', 't-tick'),
          text(560, 222, 'Same answer, measured:', 't-tick'), text(560, 240, 'max |diff| = 3.9e-16', 't-val')]
    return svg(780, 290, 'Multi-head latent attention: each token is compressed to a small latent vector plus a shared RoPE key, and only those are cached. Per-head keys and values are rebuilt from the latent, or never built at all at inference.', b)
F['p2_mla'] = mla_diagram()


import math


def cache_fig():
    words = ['The', 'cat', 'sat', 'on', 'the']
    b = [f'<rect x="20" y="74" width="510" height="90" rx="12" fill="none" class="drop"/>',
         text(548, 112, 'KV cache:', 't-note'), text(548, 132, 'kept in memory,', 't-tick'), text(548, 150, 'one K and V per token', 't-tick')]
    for i, w in enumerate(words):
        x = 30 + i * 100
        new = i == len(words) - 1
        b += [box(x, 30, 80, 32, 'box-1' if new else 'box', 8), text(x + 40, 51, w, 't-strong', 'middle'),
              box(x + 10, 84, 60, 30, 'box-1' if new else 'box-3', 6), text(x + 40, 104, 'V', 't-math', 'middle'),
              box(x + 10, 124, 60, 30, 'box-1' if new else 'box-3', 6), text(x + 40, 144, 'K', 't-math', 'middle')]
    qx = 30 + 4 * 100
    b += [box(qx + 10, 236, 60, 30, 'box-2', 6), text(qx + 40, 256, 'Q', 't-math', 'middle'),
          text(qx + 84, 256, 'the new query', 't-tick')]
    for i in range(5):
        b.append(f'<line class="edge" x1="{qx + 40}" y1="236" x2="{30 + i * 100 + 40}" y2="157" marker-end="url(#ah)"/>')
    b += [text(20, 22, 'Writing the next word after "The cat sat on the"', 't-title'),
          text(20, 298, 'Blue: the newest token. Its K and V are added to the cache. Its query is compared with every stored key, then thrown away.', 't-tick')]
    return svg(780, 312, 'How the KV cache works: every token leaves a key and a value in memory; the newest token adds its own and compares its query with all the stored keys.', b)
F['p2_cache'] = cache_fig()


def rope_fig():
    b = []
    th = 30
    for m in range(4):
        cx, cy, r = 100 + m * 190, 118, 66
        a0, a = math.radians(15), math.radians(15 + m * th)
        b += [f'<circle class="grid" cx="{cx}" cy="{cy}" r="{r}" fill="none"/>',
              f'<line class="edge" x1="{cx}" y1="{cy}" x2="{cx + r * math.cos(a0):.1f}" y2="{cy - r * math.sin(a0):.1f}" stroke-dasharray="4 4"/>',
              f'<line class="path" x1="{cx}" y1="{cy}" x2="{cx + (r - 4) * math.cos(a):.1f}" y2="{cy - (r - 4) * math.sin(a):.1f}" marker-end="url(#ah-on)"/>',
              f'<circle class="node" cx="{cx}" cy="{cy}" r="4"/>',
              text(cx, 30, f'position {m}', 't-title', 'middle'),
              text(cx, 214, f'turned by {m * th}°', 't-val', 'middle')]
    b.append(text(20, 250, 'Dashed: the vector before RoPE. Blue: after RoPE. Each position turns it a little more (here 30° per position).', 't-tick'))
    return svg(780, 264, 'RoPE turns a query or key vector by an angle that grows with its position in the text.', b)
F['p2_rope'] = rope_fig()

json.dump(F, open('results/figs_part2.json', 'w'))
print(len(F), 'figures:', list(F))
