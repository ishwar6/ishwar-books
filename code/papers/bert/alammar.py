"""Drawing helpers for "illustrated" figures (coloured token blocks, vector strips, matrices with their shapes,
attention grids, numbered step frames). Shared by figs_part*.py. Colours come only from CSS classes
(s1..s4, box-1..box-4, cell, cell-masked ...), so every figure follows the dark, dim and light themes.
Opacity is used to show size, never a hard-coded colour."""
from figlib import text, box, esc
from bertfig import line

SERIES = ['s1', 's2', 's3', 's4']


def vec(x, y, vals, cls='s1', cell=13, gap=2, vertical=False, outline=True, title=None):
    """A vector as a strip of small squares. Larger |value| = stronger colour. Returns parts.
    vals: list of numbers (any scale) or an int n for a neutral strip of n cells."""
    if isinstance(vals, int):
        vals = [0.55] * vals
    m = max(1e-9, max(abs(v) for v in vals))
    b = []
    n = len(vals)
    L = n * cell + (n - 1) * gap
    if outline:
        w, h = (cell + 6, L + 6) if vertical else (L + 6, cell + 6)
        b.append(box(x - 3, y - 3, w, h, 'box-ghost', 4))
    for i, v in enumerate(vals):
        xx, yy = (x, y + i * (cell + gap)) if vertical else (x + i * (cell + gap), y)
        op = 0.18 + 0.77 * abs(v) / m
        tip = f'<title>{esc(title)}[{i}] = {v:.3f}</title>' if title else ''
        b.append(f'<g class="mark">{tip}<rect class="{cls}" x="{xx:.1f}" y="{yy:.1f}" width="{cell}" height="{cell}" rx="2" style="fill-opacity:{op:.2f}"/></g>')
    return b


def vec_len(n, cell=13, gap=2):
    return n * cell + (n - 1) * gap


def matrix(x, y, rows, cols, cls='s1', cell=12, gap=2, vals=None, label=None, shape=None, label_cls='t-math'):
    """A matrix as a grid of squares, with an optional name above and its shape below (e.g. 'K × H')."""
    b = []
    W, H = vec_len(cols, cell, gap), vec_len(rows, cell, gap)
    b.append(box(x - 4, y - 4, W + 8, H + 8, 'box-ghost', 5))
    flat = [abs(v) for r in vals for v in r] if vals else [1]
    m = max(1e-9, max(flat))
    for i in range(rows):
        for j in range(cols):
            v = abs(vals[i][j]) / m if vals else 0.5 + 0.35 * (((i * 7 + j * 3) % 5) / 4 - 0.5)
            b.append(f'<rect class="{cls}" x="{x + j * (cell + gap):.1f}" y="{y + i * (cell + gap):.1f}" width="{cell}" height="{cell}" rx="2" style="fill-opacity:{0.18 + 0.75 * v:.2f}"/>')
    if label:
        b.append(text(x + W / 2, y - 12, label, label_cls, 'middle'))
    if shape:
        b.append(text(x + W / 2, y + H + 20, shape, 't-tick', 'middle'))
    return b, W, H


def attn_grid(x, y, row_toks, col_toks, allowed, cell=34, title=None, row_title='query (row)', col_title='key (column)',
              values=None, show_values=False, hi_rows=(), cls_fn=None, label_w=None):
    """An attention grid: rows are queries, columns are keys. allowed(i, j) -> bool.
    values: optional matrix of weights (0..1) for the opacity of allowed cells. cls_fn(i, j) -> fill class for a cell."""
    lw = label_w if label_w is not None else 8 + 7.2 * max(len(t) for t in row_toks)
    gx = x + lw
    b = []
    if title:
        b.append(text(gx + cell * len(col_toks) / 2, y - 44, title, 't-title', 'middle'))
    b.append(text(gx + cell * len(col_toks) / 2, y - 26, col_title, 't-muted', 'middle'))
    for j, c in enumerate(col_toks):
        b.append(text(gx + j * cell + cell / 2, y - 8, c, 't-tick', 'middle'))
    for i, r in enumerate(row_toks):
        b.append(text(gx - 8, y + i * cell + cell / 2 + 4, r, 't-tick t-strong' if i in hi_rows else 't-tick', 'end'))
        for j in range(len(col_toks)):
            xx, yy = gx + j * cell, y + i * cell
            if allowed(i, j):
                v = values[i][j] if values else 0.62
                c = cls_fn(i, j) if cls_fn else 'cell'
                tip = f'<title>{esc(r)} attends to {esc(col_toks[j])}' + (f': {v:.2f}' if values else '') + '</title>'
                b.append(f'<g class="mark">{tip}<rect class="{c}" x="{xx + 1.5:.1f}" y="{yy + 1.5:.1f}" width="{cell - 3:.1f}" height="{cell - 3:.1f}" rx="3" style="fill-opacity:{max(0.12, v):.2f}"/></g>')
                if show_values and values:
                    b.append(text(xx + cell / 2, yy + cell / 2 + 4, f'{v:.2f}', 't-cell' + (' on' if v > 0.55 else ''), 'middle'))
            else:
                b.append(f'<rect class="cell-masked" x="{xx + 1.5:.1f}" y="{yy + 1.5:.1f}" width="{cell - 3:.1f}" height="{cell - 3:.1f}" rx="3"/>')
                b.append(line(xx + cell * 0.32, yy + cell * 0.32, xx + cell * 0.68, yy + cell * 0.68, 'edge'))
                b.append(line(xx + cell * 0.68, yy + cell * 0.32, xx + cell * 0.32, yy + cell * 0.68, 'edge'))
    b.append(text(x, y + len(row_toks) * cell + 20, row_title, 't-muted'))
    return b, lw + cell * len(col_toks), cell * len(row_toks)


def frame(x, y, w, h, n, title, cls='box'):
    """One numbered stage of a step-by-step sequence ("frame 1 of 4")."""
    return [box(x, y, w, h, cls, 12),
            f'<circle class="node on" cx="{x + 18:.1f}" cy="{y + 18:.1f}" r="11"/>',
            text(x + 18, y + 22.5, str(n), 't-note', 'middle'),
            text(x + 36, y + 23, title, 't-title')]


def brace(x1, x2, y, label, cls='edge', up=False, tcls='t-tick'):
    """A flat bracket under (or over) a range, with a label."""
    d = -8 if up else 8
    return [f'<path class="{cls}" d="M{x1:.1f},{y - d:.1f} L{x1:.1f},{y:.1f} L{x2:.1f},{y:.1f} L{x2:.1f},{y - d:.1f}"/>',
            text((x1 + x2) / 2, y + (-8 if up else 18), label, tcls, 'middle')]
