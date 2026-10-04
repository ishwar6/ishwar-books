"""Extra drawing helpers for the BERT figures (on top of figlib). All colours come from CSS classes,
so the figures follow the dark, dim and light themes. Never hard-code a colour here."""
from figlib import svg, text, box, arrow, esc


def token(x, y, w, s, cls='box', h=30, tcls='t-tick', rx=6):
    """One token as a rounded block with its text centred."""
    return [box(x, y, w, h, cls, rx), text(x + w / 2, y + h / 2 + 4.5, s, tcls, 'middle')]


def row(x, y, words, w=64, gap=8, cls='box', h=30, tcls='t-tick', classes=None):
    """A row of tokens. classes: optional per-token box classes. Returns (parts, centres)."""
    b, cx = [], []
    for i, s in enumerate(words):
        xx = x + i * (w + gap)
        b += token(xx, y, w, s, (classes[i] if classes else cls), h, tcls)
        cx.append(xx + w / 2)
    return b, cx


def line(x1, y1, x2, y2, cls='edge', marker=None, extra=''):
    m = f' marker-end="url(#{marker})"' if marker else ''
    return f'<line class="{cls}" x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}"{m}{extra}/>'


def hbars(x0, y0, items, bar_w, row_h=24, cls='s1', label_w=90, fmt=lambda v: f'{v:.3f}', vmax=1.0, title=None, hi=None):
    """Horizontal bars: items = [(label, value)]. hi: label to draw in the 'on' class."""
    b = []
    if title:
        b.append(text(x0, y0 - 10, title, 't-title'))
    for i, (lab, v) in enumerate(items):
        y = y0 + i * row_h
        w = max(2, bar_w * v / vmax)
        on = hi is not None and lab == hi
        b.append(text(x0 + label_w - 8, y + row_h / 2 + 4, lab, 't-tick t-strong' if on else 't-tick', 'end'))
        b.append(f'<g class="mark"><title>{esc(lab)}: {fmt(v)}</title><rect class="{cls}" x="{x0 + label_w:.1f}" y="{y + 3:.1f}" width="{w:.1f}" height="{row_h - 8:.1f}" rx="3"/></g>')
        b.append(text(x0 + label_w + w + 6, y + row_h / 2 + 4, fmt(v), 't-val'))
    return b
