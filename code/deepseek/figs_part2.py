"""Figures for Part 2 (the V3 architecture), drawn from results/part2_*.json. Writes results/figs_part2.json.
CSS classes only (box, box-1, box-on, s1..s3, l1..l3, t-title, t-note, t-tick, ...), so every theme works."""
import json, os
from figlib import svg, text, box, arrow, bar_path, lines_chart, esc

HERE = os.path.dirname(os.path.abspath(__file__))
R = lambda n: json.load(open(os.path.join(HERE, 'results', f'{n}.json')))
PAR, ROU, BAL, KV = R('part2_params'), R('part2_router'), R('part2_balance'), R('part2_kv')
F = {}


def label_box(x, y, w, h, cls, title, sub=None, tcls='t-note'):
    b = [box(x, y, w, h, cls, 8), text(x + w / 2, y + (h / 2 + 5 if not sub else h / 2 - 3), title, tcls, 'middle')]
    if sub:
        b.append(text(x + w / 2, y + h / 2 + 13, sub, 't-tick', 'middle'))
    return b


# ------------------------------------------------------------------ 1. the V3 map ("you are here")
def block():
    b = [text(20, 24, 'The whole model (61 layers)', 't-title'), text(420, 24, 'Inside one MoE layer (layers 3 to 60)', 't-title')]
    col = [('token ids', None, 'box'), ('embedding', '129,280 x 7,168', 'box'),
           ('layers 0 to 2', 'MLA + dense FFN', 'box-2'), ('layers 3 to 60', 'MLA + DeepSeekMoE', 'box-1'),
           ('RMSNorm + output head', 'scores for 129,280 tokens', 'box'), ('next token', None, 'box')]
    y = 40
    for i, (t, s, c) in enumerate(col):
        h = 40 if s else 30
        b += label_box(40, y, 250, h, c, t, s)
        if i < len(col) - 1:
            b.append(arrow(165, y + h, 165, y + h + 14))
        y += h + 14
    b += label_box(40, y + 8, 250, 40, 'box-3', 'MTP module (layer 61)', 'drafts the token after next')
    b.append(text(165, y + 66, 'used in training; at inference, optional', 't-tick', 'middle'))
    moe_y = 40 + 30 + 14 + 40 + 14 + 40 + 14 + 20
    # right panel: one MoE layer
    X, W = 420, 340
    rows = [('input from the layer below', 'box', None), ('RMSNorm', 'box', None),
            ('Multi-head Latent Attention', 'box-on', 'caches 576 numbers per token'),
            ('+ residual', 'box', None), ('RMSNorm', 'box', None),
            ('DeepSeekMoE', 'box-on', 'router picks 8 of 256 + 1 shared'), ('+ residual', 'box', None)]
    y = 40
    ys = []
    for i, (t, c, s) in enumerate(rows):
        h = 42 if s else 28
        b += label_box(X, y, W, h, c, t, s)
        ys.append((y, h))
        if i < len(rows) - 1:
            b.append(arrow(X + W / 2, y + h, X + W / 2, y + h + 12))
        y += h + 12
    b.append(text(X + W / 2, y + 14, 'output to the layer above', 't-tick', 'middle'))
    b.append(f'<path class="edge-dim" d="M290,{moe_y} C350,{moe_y} 360,54 {X - 4},54"/>')
    b.append(f'<path class="edge-dim" d="M290,{moe_y} C350,{moe_y} 360,{y} {X - 4},{y}"/>')
    return svg(780, 420, 'Map of DeepSeek-V3: an embedding, 3 dense layers, 58 MoE layers, the output head and one MTP module; inside each MoE layer, MLA attention and the DeepSeekMoE feed-forward block.', b)


F['block'] = block()


# ------------------------------------------------------------------ 2. dense FFN vs MoE: same cost, 28x the parameters
def dense_vs_moe():
    p = PAR['per']
    b = [text(20, 24, 'Dense FFN (layers 0 to 2)', 't-title'), text(400, 24, 'DeepSeekMoE layer (layers 3 to 60)', 't-title')]
    b += [box(20, 40, 330, 120, 'box-on', 10), text(185, 95, 'one SwiGLU FFN, width 18,432', 't-note', 'middle'),
          text(185, 115, f'{p["dense_ffn"] / 1e6:.1f}M parameters, all used', 't-tick', 'middle')]
    # 257 small boxes: 1 shared + 256 routed, 9 lit
    lit = {0, 17, 40, 77, 123, 150, 199, 230, 251}
    cols, cw, ch, gx, gy = 22, 14, 9, 2, 2
    for k in range(257):
        r, c = divmod(k, cols)
        x, y = 400 + c * (cw + gx), 40 + r * (ch + gy)
        cls = 'box-3' if k == 0 else ('box-on' if k in lit else 'box')
        b.append(f'<rect class="{cls}" x="{x}" y="{y}" width="{cw}" height="{ch}" rx="2"/>')
    b += [text(400, 190, '1 shared expert (green) + 256 routed experts, width 2,048 each', 't-tick'),
          text(400, 208, f'{257 * p["expert"] / 1e9:.2f}B parameters stored, 9 experts used per token', 't-tick')]
    used = 9 * p['expert']
    b += [text(20, 190, 'Work per token (multiply-adds, same in both):', 't-tick'),
          text(20, 208, f'dense: {p["dense_ffn"] / 1e6:.2f}M   MoE: 9 x {p["expert"] / 1e6:.2f}M = {used / 1e6:.2f}M', 't-note'),
          text(20, 236, f'Same work for each token, {257 / 9:.1f} times the parameters to learn with.', 't-s1')]
    return svg(780, 250, 'A dense feed-forward layer next to a DeepSeekMoE layer: the same work per token, 28.6 times more parameters.', b)


F['dense_vs_moe'] = dense_vs_moe()


# ------------------------------------------------------------------ 2b. where the 256 experts of one layer live during training
def cluster():
    b = [text(20, 22, 'Training layout of one MoE layer: 256 experts on 64 GPUs in 8 nodes (4 experts per GPU)', 't-title')]
    kept = {0, 3, 4, 7}
    for n in range(8):
        r, c = divmod(n, 4)
        x, y = 20 + c * 190, 44 + r * 120
        on = n in kept
        b.append(box(x, y, 176, 100, 'box-1' if on else 'box', 10))
        b.append(text(x + 88, y + 18, f'node {n}: experts {n * 32} to {n * 32 + 31}', 't-tick', 'middle'))
        for g in range(8):
            gx, gy = x + 10 + (g % 4) * 40, y + 28 + (g // 4) * 34
            b.append(box(gx, gy, 34, 28, 'box', 4))
            b.append(text(gx + 17, gy + 18, f'GPU{g}', 't-tick', 'middle'))
    b.append(text(20, 296, 'Inside a node, GPUs talk over NVLink (fast). Between nodes, over InfiniBand: 160 GB/s against 50 GB/s on DeepSeek\'s H800 cluster.', 't-tick'))
    b.append(text(20, 316, 'Node-limited routing: the 8 experts of a token must sit on at most 4 nodes (blue: the 4 nodes of the token in the next figure).', 't-tick'))
    return svg(780, 330, 'Training layout of one DeepSeek-V3 MoE layer: 8 nodes of 8 GPUs, 4 experts per GPU; a token may reach at most 4 nodes.', b)


F['cluster'] = cluster()


# ------------------------------------------------------------------ 3. the hand-sized routing example
def hand_route():
    h = ROU['hand']
    s, ns, chosen = h['s'], h['node_score'], set(h['chosen'])
    kept = sorted(range(4), key=lambda i: -ns[i])[:2]
    b = [text(20, 22, 'Step 1: score each node by its 2 best experts.  Step 2: keep 2 nodes.  Step 3: top-4 inside them.', 't-note')]
    for n in range(4):
        x = 20 + n * 190
        on = n in kept
        b.append(box(x, 36, 176, 150, 'box-1' if on else 'box-ghost', 10))
        b.append(text(x + 88, 56, f'node {n}' + ('  (kept)' if on else '  (skipped)'), 't-note', 'middle'))
        for j in range(3):
            e = n * 3 + j
            cls = 'box-on' if e in chosen else 'box'
            b.append(box(x + 10 + j * 54, 70, 48, 50, cls, 6))
            b.append(text(x + 34 + j * 54, 90, f'e{e}', 't-tick', 'middle'))
            b.append(text(x + 34 + j * 54, 110, f'{s[e]:.2f}', 't-note', 'middle'))
        b.append(text(x + 88, 146, f'node score {ns[n]:.2f}', 't-tick', 'middle'))
        if n == 3:
            b.append(text(x + 88, 170, 'e9 (0.85) is lost', 't-bad', 'middle'))
    b.append(text(20, 212, 'Chosen: e0, e3, e4, e2 (highlighted). Gates = 0.90, 0.60, 0.55, 0.30 divided by their sum 2.35.', 't-tick'))
    return svg(780, 226, 'Hand example of node-limited routing: 12 experts on 4 nodes, the 2 best nodes are kept, the top 4 experts are chosen inside them.', b)


F['hand_route'] = hand_route()


# ------------------------------------------------------------------ 4. one token through the real-shape router
def route_token():
    t = ROU['token']
    s, nodes, chosen = t['s'], set(t['nodes']), set(t['experts'])
    lo, hi = min(s), max(s)
    b = [text(20, 22, 'One token, 256 experts in 8 nodes of 32. Shade = affinity s (sigmoid). Boxed = chosen.', 't-note')]
    cw, ch, X0, Y0 = 16, 22, 90, 40
    for n in range(8):
        y = Y0 + n * (ch + 8)
        b.append(text(X0 - 10, y + 15, f'node {n}', 't-s1' if n in nodes else 't-tick', 'end'))
        for j in range(32):
            e = n * 32 + j
            op = 0.06 + 0.9 * (s[e] - lo) / (hi - lo)
            b.append(f'<g class="mark"><title>expert {e}: s = {s[e]:.3f}</title><rect class="cell" x="{X0 + j * cw + 1}" y="{y}" width="{cw - 2}" height="{ch}" rx="2" style="fill-opacity:{op:.3f}"/></g>')
            if e in chosen:
                b.append(f'<rect class="box-on" x="{X0 + j * cw}" y="{y - 2}" width="{cw}" height="{ch + 4}" rx="3" style="fill-opacity:0"/>')
        b.append(text(X0 + 32 * cw + 14, y + 15, f'{t["node_score"][n]:.3f}' + ('  kept' if n in nodes else ''), 't-s1' if n in nodes else 't-tick'))
    b.append(text(X0 + 32 * cw + 14, Y0 - 6, 'node score', 't-tick'))
    y = Y0 + 8 * (ch + 8) + 14
    b.append(text(20, y, 'Gates of the 8 chosen experts (sum 2.5): ' + ', '.join(f'e{e} {g:.2f}' for e, g in zip(t['experts'], t['gates'])), 't-tick'))
    return svg(780, y + 14, 'One token routed through a V3-shaped router with random weights: 4 of 8 nodes kept, 8 of 256 experts chosen.', b)


F['route_token'] = route_token()


# ------------------------------------------------------------------ 5. expert load before and after the bias rule
def load_bars():
    A, B = BAL['runs']['A no balancing'], BAL['runs']['B bias rule']
    fair = BAL['T'] * BAL['K'] / BAL['E']
    panels = [('No balancing (last step)', sorted(A['load_last'], reverse=True)),
              (f'With the bias rule, after {BAL["steps"]} steps', sorted(B['load_last'], reverse=True))]
    W, L, ph, top = 780, 50, 130, 30
    vmax = 700
    b = []
    for p, (title, load) in enumerate(panels):
        y0 = top + p * (ph + 50)
        b.append(text(L, y0 - 8, title, 't-title'))
        bw = (W - L - 20) / len(load)
        for i, v in enumerate(load):
            h = v / vmax * ph
            b.append(f'<rect class="s1" x="{L + i * bw:.1f}" y="{y0 + ph - h:.1f}" width="{max(bw - 0.4, 0.8):.2f}" height="{h:.1f}"/>')
        fy = y0 + ph - fair / vmax * ph
        b.append(f'<line class="base-line" x1="{L}" y1="{fy:.1f}" x2="{W - 20}" y2="{fy:.1f}"/>')
        b.append(text(W - 22, fy - 5, f'fair share {fair:.0f}', 't-tick', 'end'))
        for t in (0, 350, 700):
            b.append(text(L - 6, y0 + ph - t / vmax * ph + 4, str(t), 't-tick', 'end'))
        b.append(f'<line class="axis" x1="{L}" y1="{y0 + ph}" x2="{W - 20}" y2="{y0 + ph}"/>')
        b.append(text(W - 22, y0 + 14, f'busiest {load[0]}, idlest {load[-1]}', 't-note', 'end'))
    b.append(text(L, top + 2 * (ph + 50) - 22, 'Each bar is one of 256 experts, sorted from busiest to idlest. Height = tokens it got in one batch of 4,096 tokens x 8 picks.', 't-tick'))
    return svg(780, top + 2 * (ph + 50) - 10, 'Tokens per expert in one batch, sorted: very uneven without balancing, close to the fair share of 128 with the bias rule.', b)


F['load_bars'] = load_bars()


# ------------------------------------------------------------------ 6. MaxVio over training steps
def maxvio():
    A, B = BAL['runs']['A no balancing']['maxvio'], BAL['runs']['B bias rule']['maxvio']
    xs = list(range(0, BAL['steps'] + 1, 20))
    xs[0] = 1
    pick = lambda h: [h[x - 1] for x in xs]
    return lines_chart(780, 300, 'MaxVio over 400 steps: stays near 4 without balancing, falls to about 0.3 with the bias rule.',
                       xs, [('no balancing', pick(A), 'l2', 's2'), ('bias rule', pick(B), 'l1', 's1')],
                       'training step (one batch of 4,096 tokens each)', 'MaxVio = (busiest - mean) / mean', 0, 5,
                       [0, 1, 2, 3, 4, 5], xticks=[1, 100, 200, 300, 400], fmt=lambda v: f'{v:.2f}' if v % 1 else f'{v:.0f}')


F['maxvio'] = maxvio()


# ------------------------------------------------------------------ 7. MLA data flow
def mla_flow():
    b = [text(20, 22, 'MLA in DeepSeek-V3, one layer, one token (sizes from config.json)', 't-title')]
    b += label_box(300, 40, 180, 40, 'box', 'hidden state h', '7,168 numbers')
    # query side
    b += label_box(30, 120, 200, 40, 'box', 'query latent', '1,536 (q_lora_rank)')
    b += label_box(30, 200, 200, 52, 'box-2', 'queries, 128 heads', '128 content + 64 RoPE each')
    # kv side
    b += label_box(300, 120, 180, 40, 'box-on', 'KV latent c', '512 (kv_lora_rank)')
    b += label_box(520, 120, 200, 40, 'box-on', 'RoPE key k_R', '64, shared by all heads')
    b += label_box(300, 200, 180, 52, 'box', 'keys, 128 heads', '128 content (+ k_R)')
    b += label_box(520, 200, 200, 52, 'box', 'values, 128 heads', '128 each')
    b += [arrow(340, 80, 140, 118), arrow(390, 80, 390, 118), arrow(440, 80, 610, 118),
          arrow(130, 160, 130, 198), arrow(380, 160, 380, 198), arrow(430, 160, 600, 198)]
    b += [text(150, 100, 'W_DQ', 't-tick', 'middle'), text(400, 104, 'W_DKV', 't-tick'), text(540, 100, 'W_KR + RoPE', 't-tick'),
          text(140, 184, 'W_UQ, W_QR', 't-tick'), text(390, 184, 'W_UK', 't-tick'), text(520, 184, 'W_UV', 't-tick')]
    b += [box(290, 112, 440, 56, 'box-ghost', 10), text(735, 145, '', 't-tick')]
    b.append(text(510, 290, 'Cached: 512 + 64 = 576 numbers per token per layer (the two highlighted boxes).', 't-s1', 'middle'))
    b.append(text(510, 310, 'Keys and values are never stored; with the absorb trick they are never even built.', 't-tick', 'middle'))
    return svg(780, 324, 'MLA data flow in DeepSeek-V3: the hidden state is compressed to a 512-number latent plus a 64-number RoPE key, which are cached; keys and values for 128 heads come from the latent.', b)


F['mla_flow'] = mla_flow()


# ------------------------------------------------------------------ 8. naive vs absorbed order
def absorb():
    b = [text(20, 22, 'Two orders for the same attention score (one head, one cached token)', 't-title')]
    y1, y2 = 50, 140
    b.append(text(20, y1 - 6, 'Naive: rebuild the key, then compare', 't-note'))
    b += label_box(20, y1, 150, 40, 'box-on', 'cached c', '512')
    b += label_box(220, y1, 150, 40, 'box', 'key = W_UK c', '128, rebuilt')
    b += label_box(420, y1, 150, 40, 'box-2', 'query q', '128')
    b += label_box(620, y1, 140, 40, 'box', 'score q . key', '1 number')
    b += [arrow(170, y1 + 20, 218, y1 + 20), arrow(370, y1 + 20, 618, y1 + 20), arrow(570, y1 + 20, 618, y1 + 20)]
    b.append(text(20, y2 - 6, 'Absorbed: move W_UK to the query side once, compare with the cache directly', 't-note'))
    b += label_box(20, y2, 150, 40, 'box-2', 'query q', '128')
    b += label_box(220, y2, 150, 40, 'box', 'q_lat = W_UK^T q', '512, once per step')
    b += label_box(420, y2, 150, 40, 'box-on', 'cached c', '512')
    b += label_box(620, y2, 140, 40, 'box', 'score q_lat . c', 'same number')
    b += [arrow(170, y2 + 20, 218, y2 + 20), arrow(370, y2 + 20, 618, y2 + 20), arrow(570, y2 + 20, 618, y2 + 20)]
    b.append(text(20, y2 + 72, 'q . (W_UK c) = (W_UK^T q) . c : the same product, grouped differently. The value side works the same way with W_UV.', 't-tick'))
    return svg(780, y2 + 86, 'Weight absorption: instead of rebuilding a key from the cached latent for every token, the query is moved into the latent space once.', b)


F['absorb'] = absorb()


# ------------------------------------------------------------------ 9. KV bytes per token
def kv_bars():
    rows = KV['rows']
    names = ['MHA (V3 head shapes)', 'MHA (textbook, 128 x 128)', 'GQA, 8 groups', 'MLA, BF16', 'MLA, FP8 layout', 'MQA, 1 head']
    order = [0, 1, 2, 4, 5, 3]
    cls = ['s2', 's2', 's3', 's1', 's1', 's3']
    W, L, Rm, T, band = 780, 200, 160, 20, 40
    vmax = rows[0]['bytes'] / 1024
    b = []
    for i, (k, nm, c) in enumerate(zip(order, names, cls)):
        r = rows[k]
        y = T + i * band
        kib = r['bytes'] / 1024
        bw = max(3, kib / vmax * (W - L - Rm))
        b.append(f'<g class="mark"><title>{esc(nm)}: {r["bytes"]:,} bytes per token</title><path class="{c} bar-mark" d="{bar_path(L, y, bw, 24, horizontal=True)}"/></g>')
        b.append(text(L - 10, y + 17, nm, 't-note', 'end'))
        b.append(text(L + bw + 8, y + 17, f'{kib:,.1f} KiB, {r["gib_128k"]:.1f} GiB', 't-val'))
    b.append(f'<line class="axis" x1="{L}" y1="{T - 4}" x2="{L}" y2="{T + len(order) * band - 10}"/>')
    b.append(text(20, T + len(order) * band + 12, 'Per token in KiB, then the cache for one 128K-token (131,072) conversation in GiB. All with 61 layers.', 't-tick'))
    return svg(780, T + len(order) * band + 26, 'KV cache per token for DeepSeek-V3 shapes: MHA 4,880 KiB, GQA-8 244 KiB, MLA 68.6 KiB in BF16 and 39.1 KiB in the FP8 layout, MQA 30.5 KiB.', b)


F['kv_bars'] = kv_bars()


# ------------------------------------------------------------------ 10. MTP with the toy sentence
def mtp():
    toks = ['The', 'cat', 'sat', 'on']
    b = [text(20, 22, 'Training: at each position, the main model predicts the next token, the MTP module the one after', 't-title')]
    for i, t in enumerate(toks):
        x = 40 + i * 180
        b += label_box(x, 210, 130, 30, 'box', f'input t{i + 1} = {t}')
        b += label_box(x, 150, 130, 34, 'box-1', 'main model', f'h{i + 1}')
        b += label_box(x, 90, 130, 34, 'box-3', 'MTP module', f'+ Emb(t{i + 2})')
        b.append(arrow(x + 65, 210, x + 65, 186))
        b.append(arrow(x + 65, 150, x + 65, 126))
        nxt = ['cat', 'sat', 'on', 'the', 'mat'][i]
        nn = ['sat', 'on', 'the', 'mat', ''][i]
        b.append(text(x + 135, 172, f'-> {nxt}', 't-s1'))
        b.append(text(x + 65, 70, f'predicts t{i + 3} = {nn}', 't-s3', 'middle'))
    b.append(text(20, 270, 'Inference: keep the module as a drafter. The main model checks the draft in its next pass; DeepSeek reports 85% to 90% accepted.', 't-tick'))
    return svg(780, 284, 'Multi-token prediction on a short sentence: the main model predicts the next token, the MTP module uses that state and the next token embedding to predict the token after it.', b)


F['mtp'] = mtp()


# ------------------------------------------------------------------ 11. parameter breakdown
def params():
    bk, ac = PAR['buckets'], PAR['active']
    groups = [('routed experts', 'routed experts', 's1'), ('attention (MLA)', 'attention (MLA), all layers', 's2'),
              ('shared experts', 'shared experts', 's3'), ('dense FFN', 'dense FFN, first layers', 's4'),
              ('embedding, head, routers, norms', None, 'box')]
    val = lambda d, k: d['embedding'] + d['output head'] + d['routers (incl. bias)'] + d['RMSNorm weights'] if k is None else d[k]
    W, L, Rr = 780, 20, 20
    b = []
    for p, (title, d) in enumerate([(f'All {PAR["total"] / 1e9:.1f}B parameters', bk), (f'Used for one token: {PAR["active_total"] / 1e9:.1f}B', ac)]):
        y = 44 + p * 120
        tot = sum(d.values())
        b.append(text(L, y - 10, title, 't-title'))
        x = L
        for name, k, c in groups:
            v = val(d, k)
            w = v / tot * (W - L - Rr)
            cls = c if c != 'box' else 'box'
            b.append(f'<g class="mark"><title>{name}: {v / 1e9:.2f}B ({100 * v / tot:.1f}%)</title><rect class="{cls}" x="{x:.1f}" y="{y}" width="{max(w, 1):.1f}" height="34"/></g>')
            x += w
        x = L
        lab = []
        for name, k, c in groups:
            v = val(d, k)
            lab.append(f'{name} {v / 1e9:.1f}B ({100 * v / tot:.1f}%)')
        b.append(text(L, y + 54, '  ·  '.join(lab[:3]), 't-tick'))
        b.append(text(L, y + 72, '  ·  '.join(lab[3:]), 't-tick'))
    b.append(text(L, 290, 'Colours: blue routed experts, orange attention, green shared experts, yellow dense FFN, grey the rest.', 't-tick'))
    return svg(780, 304, 'Where DeepSeek-V3 keeps its parameters: 97.4 percent in routed experts; for one token, routed experts are 54 percent and attention 30 percent.', b)


F['params'] = params()

json.dump(F, open(os.path.join(HERE, 'results/figs_part2.json'), 'w'))
print(f'{len(F)} figures:', ', '.join(F))
