"""Figures for Part 4 (one-layer models: skip-trigrams, copying and their bugs), drawn from results/part4.json and
results/part4_toy.json. Writes results/figs_part4.json (key -> svg string). Colours come only from CSS classes."""
import json, math
from figlib import svg, text, box, arrow, esc
from cfig import token, line

R = json.load(open('results/part4.json'))
T = json.load(open('results/part4_toy.json'))
F = {}


def tok(s):
    """A token for display: drop the leading space; a piece glued to the word before gets a leading dash."""
    if s == '<|BOS|>':
        return 'BOS'
    s = s.replace('\n', '\\n')
    return s[1:] if s.startswith(' ') else '-' + s


# ---------------------------------------------------------------- 1. one skip-trigram on a real prompt
def skip():
    toks = R['ralph']['tokens'][1:]
    a, push = R['ralph']['attn_00'], R['ralph']['push'][0]
    b = [text(24, 26, 'One skip-trigram at work: [source] ... [destination] → [out]', 't-title')]
    x0, w, g, y = 24, 60, 6, 120
    cx = []
    for i, s in enumerate(toks):
        cls = 'box-1' if s == ' Ralph' else ('box-2' if i == len(toks) - 1 else 'box')
        b += token(x0 + i * (w + g), y, w, tok(s), cls, 30)
        cx.append(x0 + i * (w + g) + w / 2)
    xs, xd = cx[toks.index(' Ralph')], cx[-1]
    b.append(f'<path class="path" d="M{xd:.1f},{y - 4} C{xd:.1f},{y - 70} {xs:.1f},{y - 70} {xs:.1f},{y - 6}" fill="none" marker-end="url(#ah-on)"/>')
    b.append(text((xs + xd) / 2, y - 62, f'QK: the destination "R" reads the source " Ralph" (head 0.0 weight {a:.2f})', 't-note', 'middle'))
    xo = x0 + len(toks) * (w + g) + 30
    b.append(arrow(cx[-1] + w / 2 + 4, y + 15, xo - 6, y + 15, True))
    b += token(xo, y, 56, '-alph', 'box-3', 30)
    b.append(text(xo + 28, y + 52, 'out', 't-tick', 'middle'))
    b.append(text(xs, y + 52, 'source', 't-tick', 'middle'))
    b.append(text(xd, y + 52, 'destination', 't-tick', 'middle'))
    b.append(text(24, y + 84, f'OV: reading " Ralph" raises "alph"; head 0.0 adds {push:+.2f} to its logit at "R"', 't-note'))
    return svg(800, y + 100, f'The prompt My friend Ralph called me today, new line, quote, R. Head 0.0 at the last token R reads back to Ralph with weight {a:.2f} (the QK part) and raises the logit of the piece alph by {push:.2f} (the OV part). Ralph is the source, R the destination, alph the out token.', b)


F['p4_skip'] = skip()


# ---------------------------------------------------------------- 2. skip-trigram tables from the weights
def tables():
    rows = [r for r in R['tables'] if r['src'] in (' perfect', ' Ralph', ' couldn', ' Paris')]
    b = [text(24, 24, 'Large entries of the QK and OV circuits of attn-only-1l (from the weights only)', 't-title')]
    cols = [(24, 'head'), (84, 'source'), (184, 'top destinations (QK)'), (480, 'top out tokens (OV)')]
    for x, s in cols:
        b.append(text(x, 52, s, 't-tick t-strong'))
    b.append(f'<line class="axis" x1="24" y1="60" x2="776" y2="60"/>')
    for i, r in enumerate(rows):
        y = 84 + i * 34
        b.append(text(24, y, r['head'], 't-tick'))
        b.append(text(84, y, tok(r['src']), 't-mono'))
        b.append(text(184, y, ', '.join(tok(t) for t in r['dest'][:3]), 't-mono'))
        b.append(text(480, y, ', '.join(tok(t) for t in r['out'][:3]), 't-mono'))
    alt = '; '.join(f'head {r["head"]}, source {tok(r["src"])}: destinations {", ".join(tok(t) for t in r["dest"][:3])}; out {", ".join(tok(t) for t in r["out"][:3])}' for r in rows)
    return svg(800, 84 + len(rows) * 34, 'A table of skip-trigrams read from the weights. ' + alt, b)


F['p4_tables'] = tables()


# ---------------------------------------------------------------- 3. factored form (toy A)
def factored():
    A = T['A']
    b = [text(24, 26, 'One head = (attention on " keep") × (OV column of " keep")', 't-title')]
    outs, dests = ['mind', 'bay'], ['in', 'at']
    want = {'in': {'mind': 1, 'bay': 0}, 'at': {'mind': 0, 'bay': 1}}

    def grid(x, y, title, val, fmt):
        g = [text(x, y - 14, title, 't-note')]
        for j, o in enumerate(outs):
            g.append(text(x + 90 + j * 90 + 40, y + 6, o, 't-tick', 'middle'))
        for i, d in enumerate(dests):
            yy = y + 16 + i * 44
            g.append(text(x + 70, yy + 26, f'keep … {d}', 't-tick', 'end'))
            for j, o in enumerate(outs):
                v = val(d, o)
                g.append(f'<rect class="s{1 if j == 0 else 2}" x="{x + 90 + j * 90}" y="{yy}" width="80" height="38" rx="5" style="fill-opacity:{0.12 + 0.13 * v:.2f}"/>')
                g.append(text(x + 90 + j * 90 + 40, yy + 24, fmt(v), 't-val', 'middle'))
        return g
    b += grid(24, 70, 'What the text needs', lambda d, o: want[d][o] * 4, lambda v: 'high' if v else 'low')
    b += grid(420, 70, 'What one head can give (toy numbers)', lambda d, o: A['table'][d][o], lambda v: f'{v:.1f}')
    b.append(text(420, 186, f'attention on keep: in {A["att"]["in"]}, at {A["att"]["at"]};  OV: mind {A["ov"]["mind"]:g}, bay {A["ov"]["bay"]:g}', 't-muted'))
    b.append(text(420, 206, 'bay / mind = 3.0 in both rows: bay wins after "in" too', 't-muted'))
    return svg(800, 220, 'Two 2 by 2 tables. On the left, what the text needs: keep in should raise mind, keep at should raise bay. On the right, what one head can give with toy numbers: after in, mind 1.0 and bay 3.0; after at, mind 1.6 and bay 4.8. The ratio of bay to mind is 3.0 in both rows, so the head raises bay after in as well.', b)


F['p4_factored'] = factored()


# ---------------------------------------------------------------- 4. the real bug in head 0.2
def keep():
    P = R['bug']['prompts']
    b = [text(24, 26, 'Head 0.2, through the source " keep": push on the logits of " mind" and " bay"', 't-title')]
    vmax = max(max(p['via_keep'].values()) for p in P)
    for i, p in enumerate(P):
        y = 60 + i * 96
        b.append(text(24, y, f'"{p["prompt"]}"  (head 0.2 puts {p["attn"]:.2f} of its attention on " keep")', 't-note'))
        for j, w in enumerate([' mind', ' bay']):
            v = p['via_keep'][w]
            yy = y + 12 + j * 30
            b.append(text(100, yy + 17, w.strip(), 't-tick', 'end'))
            b.append(f'<g class="mark"><title>{esc(w)}: {v:+.2f}</title><rect class="s{j + 1}" x="110" y="{yy + 3}" width="{500 * v / vmax:.1f}" height="20" rx="3"/></g>')
            b.append(text(116 + 500 * v / vmax, yy + 18, f'{v:+.2f}', 't-val'))
        r = p['via_keep'][' bay'] / p['via_keep'][' mind']
        b.append(text(700, y + 42, f'bay / mind = {r:.2f}', 't-note', 'middle'))
    return svg(800, 60 + 2 * 96, 'Bars for two prompts. In You need to keep this in, head 0.2 reads keep with weight {:.2f} and pushes mind by {:.2f} and bay by {:.2f}. In You need to keep the dogs at, it reads keep with weight {:.2f} and pushes mind by {:.2f} and bay by {:.2f}. The ratio of bay to mind is 2.96 in both.'.format(
        P[0]['attn'], P[0]['via_keep'][' mind'], P[0]['via_keep'][' bay'], P[1]['attn'], P[1]['via_keep'][' mind'], P[1]['via_keep'][' bay']), b)


F['p4_keep'] = keep()


# ---------------------------------------------------------------- 5. positional heads
def positional():
    b = [text(24, 26, 'Where each head looks, by distance (0 = the token itself, 1 = the token before)', 't-title')]
    cell = 46

    def grid(x, y, title, M, n):
        g = [text(x, y - 18, title, 't-note')]
        for j in range(n):
            g.append(text(x + 70 + j * cell + cell / 2, y - 2, str(j), 't-tick', 'middle'))
        for h in range(8):
            yy = y + 4 + h * (cell - 14)
            g.append(text(x + 60, yy + 21, f'head 0.{h}', 't-tick', 'end'))
            for j in range(n):
                v = M[str(h)][j]
                g.append(f'<g class="mark"><title>head 0.{h}, {j} back: {v:.3f}</title><rect class="cell" x="{x + 70 + j * cell + 1}" y="{yy + 1}" width="{cell - 2}" height="{cell - 16}" rx="3" style="fill-opacity:{max(0.05, v):.3f}"/></g>')
                g.append(text(x + 70 + j * cell + cell / 2, yy + 20, f'{v:.2f}', 't-cell' + (' on' if v > 0.55 else ''), 'middle'))
        return g
    b += grid(24, 78, 'From positions alone (query at position 200)', R['pos_only'], 3)
    b += grid(420, 78, 'Real text, 1,000 tokens (mean weight)', R['pos_text'], 4)
    return svg(800, 78 + 8 * (cell - 14) + 20, 'Two tables of attention weight by distance for the eight heads. From the position embeddings alone, heads 0.1, 0.5 and 0.7 put 0.91, 0.94 and 0.88 on the token itself and the rest on the token before; head 0.2 puts 0.56, 0.33 and 0.07 on distances 0, 1 and 2; the other heads spread almost evenly. On real text the same four heads put the most weight on distances 0 and 1, for example head 0.5 0.16 and 0.26.', b)


F['p4_positional'] = positional()


# ---------------------------------------------------------------- 6. eigenvalues of every head
def eigs():
    panels = [(f'head 0.{h}', R['eig'][str(h)]['ev'], R['eig'][str(h)]['score']) for h in range(8)]
    rs = R['random_ev']
    panels.append(('random', rs, sum(e[0] for e in rs) / sum(math.hypot(*e) for e in rs)))
    b = [text(24, 26, 'The 64 non-zero eigenvalues of each OV circuit (angle = direction, radius = log size)', 't-title')]
    allmag = [math.hypot(*e) for _, ev, _ in panels for e in ev if math.hypot(*e) > 0]
    r = 62
    for k, (name, ev, score) in enumerate(panels):
        cx, cy = 24 + 84 + (k % 5) * 154, 72 + 70 + (k // 5) * 176
        mags = [math.hypot(*e) for e in ev]
        lo, hi = math.log10(max(min(mags), 1e-9)), math.log10(max(mags))
        b.append(f'<circle class="grid" cx="{cx}" cy="{cy}" r="{r}" fill="none"/>')
        b.append(f'<circle class="grid" cx="{cx}" cy="{cy}" r="{r / 2}" fill="none"/>')
        b.append(f'<line class="axis" x1="{cx - r}" y1="{cy}" x2="{cx + r}" y2="{cy}"/>')
        b.append(f'<line class="grid" x1="{cx}" y1="{cy - r}" x2="{cx}" y2="{cy + r}"/>')
        for re_, im in ev:
            mg = math.hypot(re_, im)
            rad = 6 + (r - 8) * ((math.log10(mg) - lo) / (hi - lo + 1e-9))
            ang = math.atan2(im, re_)
            b.append(f'<circle class="s1" cx="{cx + rad * math.cos(ang):.1f}" cy="{cy - rad * math.sin(ang):.1f}" r="2.6"/>')
        b.append(text(cx, cy + r + 20, f'{name}: {score:+.2f}', 't-tick t-strong' if score > 0.5 else 't-tick', 'middle'))
    b.append(text(24 + 84 + 4 * 154, 72 + 70 + 176 - 30, 'right half = positive', 't-muted', 'middle'))
    b.append(text(24 + 84 + 4 * 154, 72 + 70 + 176 - 12, 'left half = negative', 't-muted', 'middle'))
    return svg(800, 72 + 2 * 176 + 10, 'Nine small polar plots, one per head of attn-only-1l and one random head. Each dot is one eigenvalue; dots on the right are positive. Heads 0.0, 0.3, 0.4, 0.6 and 0.7 have nearly all dots on the right (copy scores 1.00, 1.00, 0.97, 0.91, 0.97), head 0.2 mostly (0.71), heads 0.1 and 0.5 mostly on the left (-0.42 and -0.37), and the random head spreads evenly (about 0).', b)


F['p4_eigs'] = eigs()


# ---------------------------------------------------------------- 7. copy score per head, against random heads
def copyscore():
    b = [text(24, 26, 'Copy score Σλ / Σ|λ| of every head (1 = all eigenvalues positive, -1 = all negative)', 't-title')]
    x0, w, y0 = 120, 560, 50
    xz = x0 + w / 2
    sc = [r['score'] for r in R['random']]
    lo, hi = min(sc), max(sc)
    b.append(f'<rect class="box-ghost" x="{xz + lo * w / 2:.1f}" y="{y0 - 6}" width="{(hi - lo) * w / 2 + 2:.1f}" height="{8 * 28 + 4}" rx="2"/>')
    for i in range(8):
        e = R['eig'][str(i)]
        v = e['score']
        y = y0 + i * 28
        b.append(text(x0 - 10, y + 15, f'head 0.{i}', 't-tick', 'end'))
        x, ww = (xz, v * w / 2) if v >= 0 else (xz + v * w / 2, -v * w / 2)
        b.append(f'<g class="mark"><title>head 0.{i}: {v:+.3f}</title><rect class="{"s1" if v > 0.5 else "s2"}" x="{x:.1f}" y="{y + 3}" width="{ww:.1f}" height="18" rx="3"/></g>')
        tx = xz + v * w / 2 + (6 if v >= 0 else -6)
        b.append(text(tx, y + 17, f'{v:+.2f}', 't-val', 'start' if v >= 0 else 'end'))
    b.append(f'<line class="axis" x1="{xz}" y1="{y0 - 8}" x2="{xz}" y2="{y0 + 8 * 28}"/>')
    for t in (-1, -0.5, 0, 0.5, 1):
        b.append(text(xz + t * w / 2, y0 + 8 * 28 + 18, f'{t:g}', 't-tick', 'middle'))
    b.append(text(xz, y0 + 8 * 28 + 38, f'dashed band: 20 random heads of the same shapes, {lo:+.3f} to {hi:+.3f}', 't-muted', 'middle'))
    return svg(800, y0 + 8 * 28 + 50, 'Horizontal bars of the copy score for the eight heads: 0.0 +1.00, 0.1 -0.42, 0.2 +0.71, 0.3 +1.00, 0.4 +0.97, 0.5 -0.37, 0.6 +0.91, 0.7 +0.97. A thin grey band around zero shows 20 random heads, from -0.035 to +0.070.', b)


F['p4_copyscore'] = copyscore()

json.dump(F, open('results/figs_part4.json', 'w'))
print('figures:', len(F), ', '.join(F))
