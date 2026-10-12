"""Figures for Part 2 (one attention head, taken apart), drawn from results/part2.json and results/part2_toy.json.
Writes results/figs_part2.json (key -> svg string). Colours come only from CSS classes, so every figure follows the
site's dark, dim and light themes."""
import json
from figlib import svg, text, box, arrow, esc
from cfig import token, line, hbars
from alammar import attn_grid, matrix, frame

R = json.load(open('results/part2.json'))
T = json.load(open('results/part2_toy.json'))
F = {}


def num(v):
    return f'{v:g}'


def cells(x, y, M, classes, cw=40, ch=28, fmt=num):
    """A matrix (list of rows) as cells with their numbers. classes: one class per column. Returns (parts, w, h)."""
    b = []
    m = max(1e-9, max(abs(v) for r in M for v in r))
    for i, r in enumerate(M):
        for j, v in enumerate(r):
            op = 0.10 + 0.55 * abs(v) / m
            b.append(f'<rect class="{classes[j]}" x="{x + j * cw:.1f}" y="{y + i * ch:.1f}" width="{cw - 4}" height="{ch - 4}" rx="4" style="fill-opacity:{op:.2f}"/>')
            b.append(text(x + j * cw + (cw - 4) / 2, y + i * ch + ch / 2 + 2, fmt(v), 't-val', 'middle'))
    return b, len(M[0]) * cw - 4, len(M) * ch - 4


def col(x, y, vals, classes, cw=40, ch=28, fmt=num):
    if isinstance(classes, str):
        classes = [classes] * len(vals)
    b = []
    for i, v in enumerate(vals):
        bb, _, _ = cells(x, y + i * ch, [[v]], [classes[i]], cw, ch, fmt)
        b += bb
    return b, cw - 4, len(vals) * ch - 4


def tok(s):
    """A token for display: drop the leading space; a piece glued to the word before gets a leading dash."""
    if s == '<|BOS|>':
        return 'BOS'
    return s[1:] if s.startswith(' ') else '-' + s


# ---------------------------------------------------------------- 1. heads are independent and additive (toy A)
def concat():
    A = T['A']
    b = [text(24, 26, '1. The usual way: stack the result vectors, then one big multiply', 't-title')]
    y = 50
    bb, w, h = cells(40, y, A['WOH'], ['s1', 's1', 's2', 's2'])
    b += bb + [text(40 + w / 2, y + h + 20, 'W_Oᴴ (3 × 4)', 't-tick', 'middle')]
    b.append(text(40 + w + 18, y + h / 2 + 5, '×', 't-title', 'middle'))
    bb, _, h2 = col(232, y - 14, A['concat'], ['s1', 's1', 's2', 's2'])
    b += bb + [text(250, y - 14 + h2 + 20, '[r¹; r²]', 't-tick', 'middle')]
    b.append(text(300, y + h / 2 + 5, '=', 't-title', 'middle'))
    bb, _, _ = col(324, y, A['big'], 's3')
    b += bb + [text(342, y + h + 20, 'result', 't-tick', 'middle')]
    b += [text(420, y + 20, 'columns 1-2 of W_Oᴴ belong to head 1,', 't-muted'), text(420, y + 38, 'columns 3-4 to head 2', 't-muted'),
          text(420, y + 64, '1×1 + 0×2 + 2×3 + 0×(-1) = 7', 't-muted')]
    y = 214
    b.append(text(24, y - 24, '2. The paper\'s way: each head uses its own slice of W_O and the results are added', 't-title'))
    x = 40
    for k, (W, r, out, cls, name) in enumerate([(A['WO1'], A['r1'], A['h1'], 's1', '¹'), (A['WO2'], A['r2'], A['h2'], 's2', '²')]):
        bb, w, h = cells(x, y, W, [cls, cls])
        b += bb + [text(x + w / 2, y + h + 20, f'W_O{name}', 't-tick', 'middle')]
        b.append(text(x + w + 14, y + h / 2 + 5, '×', 't-title', 'middle'))
        bb, _, _ = col(x + w + 28, y + 14, r, cls)
        b += bb + [text(x + w + 46, y + h + 20, f'r{name}', 't-tick', 'middle')]
        b.append(text(x + w + 82, y + h / 2 + 5, '=', 't-title', 'middle'))
        bb, _, _ = col(x + w + 98, y, out, cls)
        b += bb
        if k == 0:
            b.append(text(x + w + 160, y + h / 2 + 5, '+', 't-title', 'middle'))
        x = 340
    b.append(text(590, y + 37, '=', 't-title', 'middle'))
    bb, _, _ = col(610, y, A['sum'], 's3')
    b += bb + [text(628, y + 104, 'same result', 't-tick', 'middle')]
    return svg(760, 340, 'Two heads writing into a 3-number stream. Top: the stacked result vector (1, 2, 3, -1) times the 3 by 4 matrix W_O^H gives (7, 2, 2). Bottom: head 1 alone gives W_O^1 r^1 = (1, 2, 3), head 2 alone gives W_O^2 r^2 = (6, 0, -1), and their sum is (7, 2, 2), the same.', b)


F['p2_concat'] = concat()


# ---------------------------------------------------------------- 2. one head moves information (toy B)
def moving():
    B = T['B']
    b = []
    fw, fh, fy = 240, 300, 10
    names = ['token 1', 'token 2', 'token 3']
    # frame 1: read
    b += frame(10, fy, fw, fh, 1, 'Read: v = W_V x')
    for i in range(3):
        y = fy + 62 + i * 62
        b.append(text(26, y + 18, names[i], 't-tick'))
        bb, _, _ = cells(80, y, [B['X'][i]], ['s1'] * 3, 30, 28)
        b += bb + [arrow(172, y + 12, 196, y + 12)]
        bb, _, _ = cells(202, y, [[B['v'][i]]], ['s2'], 34, 28)
        b += bb
    b += [text(80, fy + 54, 'x (3 numbers)', 't-muted'), text(202, fy + 54, 'v', 't-muted'),
          text(26, fy + 266, 'W_V = (1, 1, 0) adds', 't-muted'), text(26, fy + 284, 'dimensions 1 and 2', 't-muted')]
    # frame 2: mix
    x2 = 260
    b += frame(x2, fy, fw, fh, 2, 'Mix: r = A v')
    gx, gy, c = x2 + 60, fy + 82, 40
    for j in range(3):
        b.append(text(gx + j * c + c / 2, gy - 8, f'v={num(B["v"][j])}', 't-tick', 'middle'))
    for i in range(3):
        b.append(text(gx - 8, gy + i * c + c / 2 + 4, names[i].replace('token ', 'to '), 't-tick', 'end'))
        for j in range(3):
            v = B['A'][i][j]
            if j > i:
                b.append(f'<rect class="cell-masked" x="{gx + j * c + 2}" y="{gy + i * c + 2}" width="{c - 4}" height="{c - 4}" rx="3"/>')
                continue
            b.append(f'<rect class="cell" x="{gx + j * c + 2}" y="{gy + i * c + 2}" width="{c - 4}" height="{c - 4}" rx="3" style="fill-opacity:{max(0.08, v * 0.8):.2f}"/>')
            b.append(text(gx + j * c + c / 2, gy + i * c + c / 2 + 4, num(v), 't-val', 'middle'))
        bb, _, _ = cells(gx + 3 * c + 14, gy + i * c + 6, [[B['r'][i]]], ['s2'], 40, 28)
        b += bb
    b += [text(gx + 3 * c + 32, gy - 8, 'r', 't-muted', 'middle'), text(gx + 60, fy + 56, 'A: from token (column)', 't-muted', 'middle'),
          text(x2 + 16, fy + 226, 'row 3: 0.8 × 2 + 0.2 × 1 = 1.8', 't-muted'),
          text(x2 + 16, fy + 248, 'each row adds up to 1, and', 't-muted'), text(x2 + 16, fy + 266, 'no token looks at a later one', 't-muted')]
    # frame 3: write
    x3 = 510
    b += frame(x3, fy, fw, fh, 3, 'Write: h(x) = W_O r')
    for i in range(3):
        y = fy + 62 + i * 62
        b.append(text(x3 + 16, y + 18, names[i], 't-tick'))
        bb, _, _ = cells(x3 + 80, y, [B['h'][i]], ['s3'] * 3, 40, 28)
        b += bb
    b += [text(x3 + 80, fy + 54, 'added to the stream', 't-muted'),
          text(x3 + 16, fy + 266, 'W_O = (0, 0, 1)ᵀ writes', 't-muted'), text(x3 + 16, fy + 284, 'into dimension 3 only', 't-muted')]
    return svg(760, 320, 'One toy head in three steps. Read: each token\'s value is W_V x, giving 2, 1 and 0. Mix: the attention pattern averages the values, so token 3 gets 0.8 times 2 plus 0.2 times 1, which is 1.8; the results are 2, 1.5 and 1.8. Write: W_O puts each result into dimension 3 of that token\'s stream, giving (0, 0, 2), (0, 0, 1.5) and (0, 0, 1.8).', b)


F['p2_moving'] = moving()


# ---------------------------------------------------------------- 3. the two sides of x
def two_sides():
    P = R['pattern']
    n = len(P)
    b = [text(24, 26, 'x is a table (tokens × dimensions). A and W_OV act on different sides.', 't-title')]
    y = 110
    bb, wA, hA = matrix(40, y, n, n, 's2', 14, 2, vals=[[P[i][j] for j in range(n)] for i in range(n)], label='A', shape='10 × 10')
    b += bb
    b.append(text(40 + wA + 22, y + hA / 2 + 5, '·', 't-big', 'middle'))
    bb, wx, hx = matrix(40 + wA + 44, y, n, 16, 's1', 12, 2, label='x', shape='10 tokens × 512 dims')
    b += bb
    b.append(text(40 + wA + 44 + wx + 22, y + hA / 2 + 5, '·', 't-big', 'middle'))
    xw = 40 + wA + 44 + wx + 44
    bb, ww, hw = matrix(xw, y - 48, 16, 16, 's3', 12, 2, label='', shape='512 × 512')
    b += bb + [text(xw + ww / 2, y - 64, 'W_OVᵀ', 't-math', 'middle')]
    b += [text(40 + wA / 2, y + hA + 46, 'mixes rows:', 't-tick t-strong', 'middle'), text(40 + wA / 2, y + hA + 64, 'which token reads', 't-tick', 'middle'),
          text(40 + wA / 2, y + hA + 82, 'from which', 't-tick', 'middle'),
          text(xw + ww / 2, y - 48 + hw + 46, 'acts inside each row:', 't-tick t-strong', 'middle'),
          text(xw + ww / 2, y - 48 + hw + 64, 'what is read and', 't-tick', 'middle'),
          text(xw + ww / 2, y - 48 + hw + 82, 'where it is written', 't-tick', 'middle')]
    return svg(760, 390, 'The stream x drawn as a table with 10 rows (tokens) and 512 columns (dimensions). On its left, the 10 by 10 attention pattern A of head 0.3 multiplies across rows: which token reads from which. On its right, the 512 by 512 matrix W_OV transposed multiplies inside each row: what information is read and where it is written.', b)


F['p2_two_sides'] = two_sides()


# ---------------------------------------------------------------- 4. the tensor product in both orders (toy C)
def tensor():
    C = T['C']
    b = [text(24, 26, '(A ⊗ W) x: W changes each token, A mixes tokens. Either order works.', 't-title')]
    cl = ['s1', 's1']

    def grid(x, y, M, cls, label):
        bb, w, h = cells(x, y, M, [cls, cls], 44, 30)
        return bb + [text(x - 8, y + 19, 't1', 't-tick', 'end'), text(x - 8, y + 49, 't2', 't-tick', 'end'),
                     text(x + w / 2, y + h + 20, label, 't-tick', 'middle')], w
    xs = [60, 300, 560]
    for row, (mid, midlab, step1, step2) in enumerate([(C['w_first'], 'x Wᵀ', 'W: swap the 2 numbers', 'A: token 2 takes half of each'),
                                                       (C['a_first'], 'A x', 'A: token 2 takes half of each', 'W: swap the 2 numbers')]):
        y = 74 + row * 150
        bb, w = grid(xs[0], y, C['x'], 's1', 'x')
        b += bb
        b += [arrow(xs[0] + w + 12, y + 28, xs[1] - 24, y + 28), text((xs[0] + w + xs[1]) / 2 - 6, y - 8, step1, 't-muted', 'middle')]
        bb, w = grid(xs[1], y, mid, 's2', midlab)
        b += bb
        b += [arrow(xs[1] + w + 12, y + 28, xs[2] - 24, y + 28), text((xs[1] + w + xs[2]) / 2 - 6, y - 8, step2, 't-muted', 'middle')]
        bb, w = grid(xs[2], y, C['res'] if row == 0 else C['res_other'], 's3', 'result')
        b += bb
    b.append(text(xs[0], 338, 'A = [[1, 0], [0.5, 0.5]] and W = [[0, 1], [1, 0]]. Both orders give the same result.', 't-tick'))
    return svg(760, 352, 'A toy x with two tokens, (1, 2) and (3, 4). Top: swap the two numbers of each token first, giving (2, 1) and (4, 3), then let token 2 take half of each token, giving (2, 1) and (3, 2). Bottom: mix first, giving (1, 2) and (2, 3), then swap, giving (2, 1) and (3, 2). Both orders give the same result.', b)


F['p2_tensor'] = tensor()


# ---------------------------------------------------------------- 5. a real attention pattern
def pattern():
    P = R['pattern']
    toks = ['BOS' if t == '<|BOS|>' else t.strip() for t in R['tokens']]
    n = len(toks)
    bb, w, h = attn_grid(95, 80, toks, toks, lambda i, j: j <= i, cell=50, title='attn-only-1l, head 0.3: the attention pattern A',
                         row_title='destination (the token that reads), one row each', col_title='source (the token read from)',
                         values=P, show_values=True, hi_rows=(n - 1,), label_w=70)
    return svg(760, 80 + h + 40, 'The 10 by 10 attention pattern of head 0.3 on the prompt "I like green tea, so I drink green". Each row adds up to 1 and the upper triangle is empty. In the last row, the final "green" puts 0.37 on "tea", 0.37 on the first token BOS and 0.15 on "drink".', bb)


F['p2_pattern'] = pattern()


# ---------------------------------------------------------------- 6. what the head moved
def moved():
    items = [(tok(t), v) for t, v in R['moved_top']]
    b = [text(24, 26, 'What head 0.3 writes at the last token "green", read through W_U', 't-title'),
         text(24, 46, 'the five tokens whose logits it raises most (out of 48,262)', 't-tick')]
    b += hbars(24, 66, items, 420, 30, 's2', 110, lambda v: f'+{v:.2f}', max(v for _, v in items))
    return svg(760, 66 + 5 * 30 + 14, 'Head 0.3\'s output at the last position, multiplied by the unembedding, raises tea by 2.62, teac by 2.44, Tea by 2.30, cups by 2.25 and brew by 2.16.', b)


F['p2_moved'] = moved()


# ---------------------------------------------------------------- 7. a score in one step (toy D)
def qk_score():
    D = T['D']
    b = [text(24, 26, 'Two ways to get the same attention score', 't-title'),
         text(24, 60, '1. With a query and a key', 't-note')]
    b += token(24, 76, 150, 'x_dst = (2, 0)', 'box-1', 30) + [arrow(180, 91, 214, 91)]
    b += token(220, 76, 140, 'W_Q = (1, 0)', 'box', 30) + [arrow(366, 91, 400, 91)]
    b += token(406, 76, 80, f'q = {D["q"]:g}', 'box-1', 30)
    b += token(24, 122, 150, 'x_src = (0, 3)', 'box-2', 30) + [arrow(180, 137, 214, 137)]
    b += token(220, 122, 140, 'W_K = (0, 1)', 'box', 30) + [arrow(366, 137, 400, 137)]
    b += token(406, 122, 80, f'k = {D["k"]:g}', 'box-2', 30)
    b += [line(490, 91, 530, 112, 'edge'), line(490, 137, 530, 116, 'edge')]
    b += token(536, 99, 190, f'q · k = 2 × 3 = {D["score"]:g}', 'box-3', 30)
    b.append(text(24, 200, '2. With one matrix, W_QK = W_Qᵀ W_K', 't-note'))
    bb, w, h = cells(60, 216, D['WQK'], ['s1', 's1'], 40, 28)
    b += bb + [text(60 + w / 2, 216 + h + 18, 'W_QK (2 × 2)', 't-tick', 'middle')]
    b += [text(170, 245, 'x_dstᵀ W_QK x_src = 2 × 1 × 3', 't-note'), text(170, 268, 'no query or key vector is ever made', 't-muted')]
    b += token(536, 228, 190, f'score = {D["score1"]:g}', 'box-3', 30)
    return svg(760, 300, 'Top: the query of the destination token (2, 0) is W_Q x = 2, the key of the source token (0, 3) is W_K x = 3, and the score is their product, 6. Bottom: the single matrix W_QK = W_Q transposed times W_K is [[0, 1], [0, 0]], and x_dst transposed times W_QK times x_src is also 6.', b)


F['p2_qk_score'] = qk_score()


# ---------------------------------------------------------------- 8. singular values: rank 64
def svals():
    b = [text(24, 26, 'Head 0.3: the first 80 of 512 singular values. Only 64 are not zero.', 't-title')]
    for k, (name, cls, x0) in enumerate([('W_OV', 's2', 40), ('W_QK', 's1', 400)]):
        S = R[name]['sv'][:80]
        top = max(S)
        y0, ph, bw = 60, 150, 4
        b.append(text(x0, y0 - 8, f'{name}: 512 × 512, rank {R[name]["rank"]}', 't-note'))
        b.append(f'<line class="axis" x1="{x0}" y1="{y0 + ph}" x2="{x0 + 80 * bw}" y2="{y0 + ph}"/>')
        for i, s in enumerate(S):
            h = max(0.5, ph * s / top)
            b.append(f'<g class="mark"><title>{name} singular value {i + 1}: {s:.3g}</title><rect class="{cls}" x="{x0 + i * bw:.1f}" y="{y0 + ph - h:.1f}" width="{bw - 1}" height="{h:.1f}"/></g>')
        xm = x0 + 64 * bw
        b += [f'<line class="base-line" x1="{xm}" y1="{y0 - 2}" x2="{xm}" y2="{y0 + ph + 6}"/>',
              text(xm, y0 + ph + 22, '64', 't-tick', 'middle'), text(x0, y0 + ph + 22, '1', 't-tick', 'middle'),
              text(x0 + 80 * bw, y0 + ph + 22, '80', 't-tick', 'middle'),
              text(x0, y0 + ph + 44, f'1st {S[0]:.2f}, 64th {S[63]:.2f}, 65th {S[64]:.1e}', 't-muted')]
    return svg(760, 270, 'Bar charts of the first 80 singular values of head 0.3\'s W_OV and W_QK. In both, the first 64 bars are clearly above zero (W_OV from 2.87 down to 1.06, W_QK from 1.31 down to 0.19) and bars 65 to 80 are zero, so both 512 by 512 matrices have rank 64.', b)


F['p2_svals'] = svals()


# ---------------------------------------------------------------- 9. the QK and OV circuits as paths
def circuits():
    b = [text(24, 26, 'The two circuits of one head, as paths from tokens to tokens', 't-title')]
    W = 92

    def chain(y, items, cls):
        x = 24
        out = []
        for k, (lab, shape) in enumerate(items):
            w = 116 if k == 0 else W
            out += token(x, y, w, lab, cls if k == 0 or k == len(items) - 1 else 'box', 32)
            if shape:
                out.append(text(x + w / 2, y + 50, shape, 't-muted', 'middle'))
            if k < len(items) - 1:
                out.append(arrow(x + w + 4, y + 16, x + w + 24, y + 16))
            x += w + 28
        return out
    b.append(text(24, 58, 'QK circuit  W_Eᵀ W_Qᵀ W_K W_E: how much a destination token wants to read a source token', 't-note'))
    b += chain(72, [('destination', None), ('W_E', '512 × 48,262'), ('W_Q', '64 × 512'), ('query q', '64')], 'box-1')
    b += chain(140, [('source', None), ('W_E', '512 × 48,262'), ('W_K', '64 × 512'), ('key k', '64')], 'box-1')
    b += [line(504, 88, 530, 118, 'edge'), line(504, 156, 530, 126, 'edge')]
    b += token(536, 106, 200, 'score q · k (one number)', 'box-3', 32)
    b.append(text(24, 232, 'OV circuit  W_U W_O W_V W_E: what reading a source token does to the output logits', 't-note'))
    b += chain(248, [('source', None), ('W_E', '512 × 48,262'), ('W_V', '64 × 512'), ('W_O', '512 × 64'), ('W_U', '48,262 × 512'), ('logits', '48,262')], 'box-2')
    b.append(text(24, 336, 'Each circuit is one 48,262 × 48,262 table: one row or column per token.', 't-tick'))
    return svg(760, 350, 'Two path diagrams. QK circuit: a destination token goes through W_E and W_Q to a 64-number query, a source token goes through W_E and W_K to a 64-number key, and their dot product is one attention score. OV circuit: a source token goes through W_E, W_V, W_O and W_U to 48,262 logits. Each circuit is a 48,262 by 48,262 table.', b)


F['p2_circuits'] = circuits()


# ---------------------------------------------------------------- 10. the OV circuit of head 0.3, read from the weights
def ov_real():
    O = R['ov']
    words = [' tea', ' green', ' Paris', ' Obama', ' coffee', ' London']
    b = [text(24, 24, 'OV circuit of head 0.3: "if I read this token, I raise these logits"', 't-title'),
         text(24, 44, 'top 4 entries of a column of W_U W_OV W_E, computed from the weights only (no text is run)', 't-tick')]
    for i, w in enumerate(words):
        y = 64 + i * 40
        b += token(24, y, 110, tok(w), 'box-2', 30) + [arrow(140, y + 15, 176, y + 15)]
        for k, (t, _) in enumerate(O[w]['top'][:4]):
            b += token(184 + k * 140, y, 130, tok(t), 'box-3' if k == 0 else 'box', 30)
    return svg(760, 64 + len(words) * 40 + 6, 'For six source tokens, the four output tokens whose logits head 0.3 raises most when it attends to them: tea to tea, Tea, teac, brew; green to green, -Green, greens, -green; Paris to -Paris, Paris, -France, rue; Obama to -Obama, Obama, McCain, Romney; coffee to Coffee, coffee, caffeine, Starbucks; London to -London, London, -inburgh, Thames.', b)


F['p2_ov_real'] = ov_real()


# ---------------------------------------------------------------- 11. the QK circuit of head 0.3, read from the weights
def qk_real():
    Q = R['qk']
    words = [' in', ' see', ' of', ' the', ' drink']
    b = [text(24, 24, 'QK circuit of head 0.3: "standing on this token, which earlier tokens do I read?"', 't-title'),
         text(24, 44, 'top 4 entries of a row of W_Eᵀ W_QK W_E (no positions, no biases), from the weights only', 't-tick')]
    for i, w in enumerate(words):
        y = 64 + i * 40
        b += token(24, y, 90, tok(w), 'box-1', 30) + [arrow(120, y + 15, 152, y + 15)]
        for k, (t, _) in enumerate(Q[w]['top'][:3 if w == ' drink' else 4]):
            b += token(160 + k * 128, y, 120, tok(t), 'box-3' if k == 0 and w != ' drink' else 'box', 30)
        if w == ' drink':
            b.append(text(160 + 3 * 128 + 60, y + 20, '(no clear pattern)', 't-muted', 'middle'))
    return svg(760, 64 + len(words) * 40 + 6, 'For five destination tokens, the four source tokens head 0.3 scores highest: in to -imbabwe (from Zimbabwe), Bulgaria, Cambodia, -agua; see to Derek, Trudeau, Shawn, Stephanie; of to HBV, schizophrenia, cirrhosis, hepatocellular; the to cytosol, -arynx, myocardium, landlord; drink to code punctuation with no clear pattern.', b)


F['p2_qk_real'] = qk_real()

json.dump(F, open('results/figs_part2.json', 'w'))
print('figures:', len(F), ', '.join(F))
