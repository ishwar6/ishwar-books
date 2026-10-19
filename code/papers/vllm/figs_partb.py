"""Figures for the Background page (the KV cache and paging). Writes results/figs_partb.json. CSS classes only.
Numbers come from results/partb.json (our run)."""
import json, os
from figlib import svg, text, box, arrow
from samfig import token

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, 'results', 'partb.json')))
F = {}


def attention_tiny():
    """One query, three keys and values; the weights and the weighted average from our run."""
    w, out = R['tiny']['weights'], R['tiny']['out']
    W, H = 720, 250
    b = [text(20, 26, 'One query looks at three tokens', 't-title')]
    b += token(30, 110, 120, 'q = [1, 0]', 'box-on', 34, 't-mono')
    keys = ['[1, 0]', '[0, 1]', '[1, 1]']
    for i, (k, wi) in enumerate(zip(keys, w)):
        y = 50 + i * 60
        b += token(250, y, 130, f'k{i + 1} = {k}', 'box-1', 34, 't-mono')
        b += token(400, y, 130, f'v{i + 1} = {k}', 'box-2', 34, 't-mono')
        b.append(arrow(152, 127, 246, y + 17))
        b.append(text(560, y + 22, f'weight {wi:.3f}', 't-tick t-strong'))
    b.append(text(250, 236, f'output = 0.401 v1 + 0.198 v2 + 0.401 v3 = [{out[0]:.3f}, {out[1]:.3f}]', 't-mono'))
    return svg(W, H, 'A query vector [1, 0] compared with three keys. Keys 1 and 3 match it equally well and get weight 0.401 '
               'each; key 2 gets 0.198. The output is the weighted average of the three values, [0.802, 0.599].', b)


def causal_mask():
    """Who can see whom: token i reads tokens 1..i only. Adding token 6 adds a row, never changes old rows."""
    words = ['Four', 'score', 'and', 'seven', 'years', 'ago']
    W, H, x0, y0, c = 720, 300, 140, 60, 40
    b = [text(20, 26, 'The causal mask: each token reads only itself and the tokens before it', 't-title')]
    for j, s in enumerate(words):
        b.append(text(x0 + j * c + c / 2, y0 - 8, s, 't-tick', 'middle'))
    for i, s in enumerate(words):
        b.append(text(x0 - 10, y0 + i * c + c / 2 + 4, s, 't-tick', 'end'))
        for j in range(len(words)):
            cls = ('box-2' if i == len(words) - 1 else 'box-1') if j <= i else 'box-ghost'
            b.append(box(x0 + j * c + 2, y0 + i * c + 2, c - 4, c - 4, cls, 4))
    tx = x0 + len(words) * c + 30
    b += [text(tx, y0 + 20, 'Row = the token that reads.', 't-tick'),
          text(tx, y0 + 40, 'Column = the token it may read.', 't-tick'),
          text(tx, y0 + 80, 'The last row (orange) is the newest', 't-tick'),
          text(tx, y0 + 100, 'token. It adds one row and reads', 't-tick'),
          text(tx, y0 + 120, 'all earlier columns.', 't-tick'),
          text(tx, y0 + 160, 'No earlier row ever reads a later', 't-tick'),
          text(tx, y0 + 180, 'token, so earlier keys and values', 't-tick'),
          text(tx, y0 + 200, 'never change: they can be cached.', 't-tick t-strong')]
    return svg(W, H, 'A six by six grid. Each row is a token; it may read the columns up to and including itself. The upper '
               'right triangle is empty. The last row, the newest token, reads all earlier tokens.', b)


def fragmentation():
    """The 32-slot memory of our run: after B and D leave, 16 slots are free but no gap holds 12."""
    W, H, x0, sw = 720, 190, 30, 20
    b = [text(20, 26, 'External fragmentation in 32 slots', 't-title')]
    rows = [('A=8, B=6, C=8, D=6', 'AAAAAAAABBBBBBCCCCCCCCDDDDDD....'), ('B and D finish', 'AAAAAAAA......CCCCCCCC..........')]
    cls = {'A': 'box-1', 'B': 'box-3', 'C': 'box-1', 'D': 'box-3', '.': 'box-ghost'}
    for r, (lab, pic) in enumerate(rows):
        y = 60 + r * 64
        b.append(text(x0, y - 8, lab, 't-tick t-strong'))
        for i, ch in enumerate(pic):
            b.append(box(x0 + i * sw, y, sw - 3, 30, cls[ch], 3))
            if ch != '.':
                b.append(text(x0 + i * sw + (sw - 3) / 2, y + 20, ch, 't-tick', 'middle'))
    b.append(text(x0, 178, 'Free: 6 + 10 = 16 slots. Request E needs 12 in one piece: refused. In blocks of 2 slots it fits.', 't-tick'))
    return svg(W, H, 'Two rows of 32 slots. First, A, B, C and D fill 28 slots. Then B and D leave, which leaves a gap of 6 '
               'and a gap of 10. A request for 12 contiguous slots is refused, although 16 slots are free.', b)


def paging():
    """Virtual pages of one program, the page table, physical frames."""
    W, H = 720, 260
    b = [text(20, 26, 'Paging: the program sees one row; the pages sit anywhere', 't-title')]
    table = {0: 5, 1: 2, 2: 7, 3: 0}
    b.append(text(30, 56, 'virtual pages (the program)', 't-tick t-strong'))
    for v in range(4):
        b += token(30, 66 + v * 42, 120, f'page {v}', 'box-1', 34, 't-mono')
    b.append(text(240, 56, 'page table', 't-tick t-strong'))
    for v, f in table.items():
        b += token(240, 66 + v * 42, 110, f'{v} -> {f}', 'box', 34, 't-mono')
        b.append(arrow(152, 83 + v * 42, 236, 83 + v * 42))
    b.append(text(450, 56, 'physical frames (real memory)', 't-tick t-strong'))
    used = {f: v for v, f in table.items()}
    for f in range(8):
        x, y = 450 + (f % 4) * 62, 66 + (f // 4) * 50
        if f in used:
            b += token(x, y, 56, f'p{used[f]}', 'box-1', 34, 't-mono')
        else:
            b.append(box(x, y, 56, 34, 'box-ghost', 6))
            b.append(text(x + 28, y + 22, str(f), 't-muted', 'middle'))
    b.append(text(450, 190, 'frame numbers 0 to 7; empty frames', 't-tick'))
    b.append(text(450, 208, 'belong to other programs or are free', 't-tick'))
    b.append(text(30, 248, 'address 10,000: page 2, offset 1,808 -> frame 7 -> 7 x 4,096 + 1,808 = 30,480', 't-mono'))
    return svg(W, H, 'Four virtual pages of one program map through a page table to frames 5, 2, 7 and 0 of physical memory. '
               'Address 10,000 is in page 2 at offset 1,808, which maps to physical address 30,480.', b)


F['pb_attention'] = attention_tiny()
F['pb_mask'] = causal_mask()
F['pb_frag'] = fragmentation()
F['pb_paging'] = paging()
json.dump(F, open(os.path.join(HERE, 'results', 'figs_partb.json'), 'w'), indent=1)
print('figures:', ', '.join(F))
