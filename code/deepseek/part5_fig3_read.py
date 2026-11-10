"""Part 5: read the exact numbers behind Figure 3 of the DeepSeek-V3.2 paper (arXiv 2512.02556, PDF page 6).

The figure is vector graphics, so the curves are stored as line segments in the PDF. We find the grid
lines (for the axis scale) and the blue (V3.1-Terminus) and orange (V3.2) paths, and convert their
points into (token position, USD per million tokens). No reading by eye.
"""
import pymupdf as fitz
from paper_shots import pdf
from common import Log, save

log = Log('part5_fig3')
pg = pdf('2512.02556')[5]
BLUE, ORANGE, GREY = (0.12, 0.47, 0.71), (1.0, 0.5, 0.05), (0.69, 0.69, 0.69)


def near(c, ref):
    return c is not None and all(abs(a - b) < 0.03 for a, b in zip(c, ref))


draws = pg.get_drawings()
panels = {'prefill': (90, 290, 0.7), 'decode': (320, 520, 2.4)}    # x range of the panel, top y-tick value
out = {}
for name, (xa, xb, ytop) in panels.items():
    grid = [d for d in draws if near(d.get('color'), GREY) and xa < d['rect'].x0 < xb]
    hz = sorted({round(d['items'][0][1].y, 2) for d in grid if d['rect'].height < 0.1})
    vt = sorted({round(d['items'][0][1].x, 2) for d in grid if d['rect'].width < 0.1})
    y0, y1 = hz[-1], hz[0]                       # $0 line (bottom) and top tick line
    x0, x1 = vt[0], vt[-1]                       # 0K and 128K
    X = lambda x: min(131072, max(0, (x - x0) / (x1 - x0) * 131072))   # clamp the half-pixel overhang of the stroke
    Y = lambda y: (y0 - y) / (y0 - y1) * ytop
    out[name] = {}
    for label, col in [('V3.1-Terminus', BLUE), ('V3.2', ORANGE)]:
        path = max((d for d in draws if near(d.get('color'), col) and xa < d['rect'].x0 < xb),
                   key=lambda d: d['rect'].width)  # the curve, not the short legend sample
        pts = [path['items'][0][1]] + [it[2] for it in path['items']]
        vals = [(round(X(p.x)), round(Y(p.y), 4)) for p in pts]
        out[name][label] = vals
        log(f'{name:8s} {label:14s} corner points (token position, $ per 1M tokens): {vals}')

    # value at 128K and the crossing point of the two curves
    t, s = out[name]['V3.1-Terminus'], out[name]['V3.2']
    def at(curve, x):
        x = min(max(x, curve[0][0]), curve[-1][0])    # the strokes stop 2 tokens short of the axis ends
        for (xa_, ya), (xb_, yb) in zip(curve, curve[1:]):
            if xa_ <= x <= xb_:
                return ya + (yb - ya) * (x - xa_) / (xb_ - xa_)
    lo = next(x for x in range(0, 131073, 64) if at(t, x) >= at(s, x))
    log(f'{name:8s} at 128K: Terminus ${at(t, 131072):.3f}, V3.2 ${at(s, 131072):.3f}, '
        f'ratio {at(t, 131072) / at(s, 131072):.1f}x; curves cross near {lo:,} tokens')
    log(f'{name:8s} at 32K:  Terminus ${at(t, 32768):.3f}, V3.2 ${at(s, 32768):.3f}; '
        f'at 64K: Terminus ${at(t, 65536):.3f}, V3.2 ${at(s, 65536):.3f}')
    out[name]['cross'] = lo
    out[name]['at'] = {str(x): [at(t, x), at(s, x)] for x in [0, 8192, 16384, 32768, 65536, 98304, 131072]}

save('part5_fig3', out)
