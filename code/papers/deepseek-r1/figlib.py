"""Tiny SVG helpers for the site's figures. Classes (box-1, s1, t-note, ...) are styled by the site's CSS,
so every figure follows the dark, dim and light themes."""
import math

DEFS = ('<defs><marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path class="arrow" d="M0,0L10,5L0,10z"/></marker>'
        '<marker id="ah-on" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path class="arrow-on" d="M0,0L10,5L0,10z"/></marker>'
        '<marker id="ah-2" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path class="arrow-2" d="M0,0L10,5L0,10z"/></marker></defs>')


def svg(w, h, label, body):
    return f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">{DEFS}{"".join(body)}</svg>'


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, cls='t-note', anchor='start', extra=''):
    return f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}"{extra}>{esc(s)}</text>'


def box(x, y, w, h, cls='box', rx=10):
    return f'<rect class="{cls}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}"/>'


def arrow(x1, y1, x2, y2, on=False):
    return f'<line class="{"path" if on else "edge"}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" marker-end="url(#{"ah-on" if on else "ah"})"/>'


def bar_path(x, y, w, h, horizontal=False, r=4):
    r = min(r, (w if not horizontal else h) / 2, (h if not horizontal else w))
    if horizontal:
        return f'M{x},{y}H{x + w - r}Q{x + w},{y} {x + w},{y + r}V{y + h - r}Q{x + w},{y + h} {x + w - r},{y + h}H{x}Z'
    return f'M{x},{y + h}V{y + r}Q{x},{y} {x + r},{y}H{x + w - r}Q{x + w},{y} {x + w},{y + r}V{y + h}Z'


def heatmap(x0, y0, M, rows, cols, cell, title=None, show_values=False, mask_upper=False, label_w=70):
    """Sequential single-hue heatmap: cell opacity = value (0..1). Masked cells are drawn as empty."""
    b = []
    if title:
        b.append(text(x0 + label_w + cell * len(cols) / 2, y0 - 12, title, 't-title', 'middle'))
    for i, r in enumerate(rows):
        b.append(text(x0 + label_w - 8, y0 + i * cell + cell / 2 + 4, r.strip() or '·', 't-tick', 'end'))
        for j in range(len(cols)):
            x, y = x0 + label_w + j * cell, y0 + i * cell
            if mask_upper and j > i:
                b.append(f'<rect class="cell-masked" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2"/>')
                continue
            v = M[i][j]
            b.append(f'<g class="mark"><title>{esc(r.strip())} → {esc(cols[j].strip())}: {v:.2f}</title>'
                     f'<rect class="cell" x="{x + 1:.1f}" y="{y + 1:.1f}" width="{cell - 2:.1f}" height="{cell - 2:.1f}" rx="2" style="fill-opacity:{max(0.04, v):.3f}"/></g>')
            if show_values:
                b.append(text(x + cell / 2, y + cell / 2 + 4, f'{v:.2f}', 't-cell' + (' on' if v > 0.55 else ''), 'middle'))
    for j, c in enumerate(cols):
        cx, cy = x0 + label_w + j * cell + cell / 2, y0 + len(rows) * cell + 10
        b.append(text(cx, cy, c.strip() or '·', 't-tick', 'end', f' transform="rotate(-50 {cx:.1f} {cy:.1f})"'))
    return b


def lines_chart(w, h, label, xs, series, xlabel, ylabel, ymin, ymax, yticks, xlog=False, xticks=None, fmt=lambda v: f'{v}', right=170):
    """series: list of (name, values, line_class, dot_class). Direct end labels + hover titles."""
    L, T, B = 64, 30, 50
    pw, ph = w - L - right, h - T - B
    lx = (lambda v: math.log2(v)) if xlog else (lambda v: v)
    x0, x1 = lx(xs[0]), lx(xs[-1])
    X = lambda v: L + (lx(v) - x0) / (x1 - x0) * pw
    Y = lambda v: T + ph - (v - ymin) / (ymax - ymin) * ph
    b = []
    for t in yticks:
        b += [f'<line class="grid" x1="{L}" y1="{Y(t):.1f}" x2="{L + pw}" y2="{Y(t):.1f}"/>', text(L - 8, Y(t) + 4, fmt(t), 't-tick', 'end')]
    for v in (xticks or xs):
        b.append(text(X(v), T + ph + 20, f'{v:,}', 't-tick', 'middle'))
    b.append(f'<line class="axis" x1="{L}" y1="{T + ph}" x2="{L + pw}" y2="{T + ph}"/>')
    ends = []
    for name, vals, lc, dc in series:
        b.append(f'<polyline class="{lc}" points="' + ' '.join(f'{X(x):.1f},{Y(v):.1f}' for x, v in zip(xs, vals)) + '"/>')
        for x, v in zip(xs, vals):
            b.append(f'<g class="mark"><title>{esc(name)}, {x:,}: {fmt(v)}</title><circle class="{dc} ring" cx="{X(x):.1f}" cy="{Y(v):.1f}" r="5"/></g>')
        ends.append([Y(vals[-1]), f'{name}: {fmt(vals[-1])}', dc, Y(vals[-1])])
    ends.sort()
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 17)
    for ly, lab, dc, ty in ends:
        b.append(f'<line class="grid" x1="{X(xs[-1]) + 6}" y1="{ty:.1f}" x2="{X(xs[-1]) + 16}" y2="{ly:.1f}"/>')
        b.append(f'<rect class="{dc}" x="{X(xs[-1]) + 20}" y="{ly - 5:.1f}" width="10" height="10" rx="2"/>')
        b.append(text(X(xs[-1]) + 36, ly + 4, lab, 't-note'))
    b.append(text(L + pw / 2, h - 10, xlabel, 't-tick', 'middle'))
    b.append(text(L - 50, T - 12, ylabel, 't-tick'))
    return svg(w, h, label, b)
