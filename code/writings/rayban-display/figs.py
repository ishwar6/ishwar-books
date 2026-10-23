"""SVG figures for "How Ray-Ban Display Glasses Put a Screen in a Lens".

CSS classes only (box, box-1..4, box-on, box-ghost, edge, edge-1..3, edge-dim, path, l1..l4, s1..s4, arrow, arrow-on,
t-title, t-note, t-tick, t-muted, t-s1..t-s4, t-on, t-bad, axis, grid ...), never colours, so every figure follows the
site theme (dark, dim, light). Numbers come from results/numbers.json (numbers.py) and the cited specs.
Run: python figs.py  ->  results/figs.json  (used by assemble.py)"""
import json, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [x for x in sys.path if os.path.abspath(x or '.') != HERE]
N = json.load(open(os.path.join(HERE, 'results', 'numbers.json')))
F = {}


def esc(s):
    return str(s).replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')


def text(x, y, s, cls='t-note', anchor=None, rot=None):
    a = f' text-anchor="{anchor}"' if anchor else ''
    r = f' transform="rotate({rot} {x:g} {y:g})"' if rot is not None else ''
    return f'<text class="{cls}" x="{x:g}" y="{y:g}"{a}{r}>{esc(s)}</text>'


def lines(x, y, rows, cls='t-note', anchor=None, gap=17):
    return ''.join(text(x, y + i * gap, s, cls if i == 0 or cls != 't-note' else 't-muted', anchor)
                   for i, s in enumerate(rows))


def rect(x, y, w, h, cls='box', rx=4):
    return f'<rect class="{cls}" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="{rx}"/>'


def box(x, y, w, h, cls='box', rows=(), rx=6, sub='t-muted', gap=17):
    out = [rect(x, y, w, h, cls, rx)]
    n = len(rows)
    y0 = y + h / 2 - (n - 1) * gap / 2 + 4.5
    for i, s in enumerate(rows):
        out.append(text(x + w / 2, y0 + i * gap, s, 't-note' if i == 0 else sub, 'middle'))
    return ''.join(out)


def line(x1, y1, x2, y2, cls='edge'):
    return f'<line class="{cls}" x1="{x1:g}" y1="{y1:g}" x2="{x2:g}" y2="{y2:g}"/>'


def poly(pts, cls='edge'):
    return f'<polyline class="{cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'


def polygon(pts, cls='box'):
    return f'<polygon class="{cls}" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>'


def head(x, y, dx, dy, cls='arrow', size=8):
    L = math.hypot(dx, dy) or 1
    ux, uy = dx / L, dy / L
    px, py = -uy, ux
    p2 = (x - ux * size + px * size * 0.5, y - uy * size + py * size * 0.5)
    p3 = (x - ux * size - px * size * 0.5, y - uy * size - py * size * 0.5)
    return f'<polygon class="{cls}" points="{x:.1f},{y:.1f} {p2[0]:.1f},{p2[1]:.1f} {p3[0]:.1f},{p3[1]:.1f}"/>'


def arrow(pts, cls='edge', hcls='arrow', size=8):
    (x1, y1), (x2, y2) = pts[-2], pts[-1]
    return poly(pts, cls) + head(x2, y2, x2 - x1, y2 - y1, hcls, size)


def ray(pts, cls='path', hcls='arrow-on', size=9):
    return arrow(pts, cls, hcls, size)


def circle(cx, cy, r, cls='box'):
    return f'<circle class="{cls}" cx="{cx:g}" cy="{cy:g}" r="{r:g}"/>'


def path(d, cls='edge'):
    return f'<path class="{cls}" d="{d}"/>'


def badge(x, y, n):
    return circle(x, y, 10, 'box-on') + text(x, y + 4.5, str(n), 't-note', 'middle')


def eye_up(cx, cy, r=26):
    """An eye seen from above, looking up the page: light enters at the top."""
    return (circle(cx, cy, r, 'box') +
            f'<ellipse class="box-1" cx="{cx:g}" cy="{cy - r + 7:g}" rx="{r * 0.45:g}" ry="5"/>' +
            path(f'M{cx - r * 0.75:.1f},{cy + r * 0.6:.1f} A{r},{r} 0 0 0 {cx + r * 0.75:.1f},{cy + r * 0.6:.1f}', 'edge-2'))


def eye_left(cx, cy, r=40):
    """An eye seen from the side, looking left: light enters from the left."""
    return (circle(cx, cy, r, 'box') +
            f'<ellipse class="box-1" cx="{cx - r + 9:g}" cy="{cy:g}" rx="7" ry="{r * 0.42:g}"/>' +
            path(f'M{cx + r * 0.55:.1f},{cy - r * 0.8:.1f} A{r},{r} 0 0 1 {cx + r * 0.55:.1f},{cy + r * 0.8:.1f}', 'edge-2'))


def svg(key, w, h, label, body, caption):
    F[key] = dict(svg=(f'<figure class="fig"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{esc(label)}">'
                       + ''.join(body) + f'</svg><figcaption>{caption}</figcaption></figure>'))


def hbars(key, title, rows, vmax, label, caption, unit='', x0=250, x1=640, fmt='{:g}', note=None):
    H = 46 + 34 * len(rows) + (30 if note else 14)
    b = [text(10, 22, title, 't-title')]
    for i, (name, v, cls) in enumerate(rows):
        y = 40 + 34 * i
        b.append(text(x0 - 10, y + 16, name, 't-note', 'end'))
        w = max(2, (x1 - x0) * v / vmax)
        b.append(rect(x0, y, w, 22, cls, 3))
        b.append(text(x0 + w + 8, y + 16, fmt.format(v) + unit, 't-val'))
    if note:
        b.append(text(10, H - 10, note, 't-muted'))
    svg(key, 720, H, label, b, caption)


# =================================================================================================
# 1. The whole path
# =================================================================================================
b = [text(10, 22, 'From a tiny chip to your eye (seen from above, right side)', 't-title')]
b.append(rect(170, 40, 210, 52, 'box-ghost'))
b.append(text(275, 62, 'the picture seems to', 't-muted', 'middle'))
b.append(text(275, 79, 'float here, far ahead', 't-muted', 'middle'))
b.append(rect(110, 178, 370, 18, 'box-1', 3))
b.append(text(110, 170, 'right lens = waveguide (a thin glass plate)', 't-note'))
b.append(rect(480, 172, 22, 30, 'box', 3))
b.append(rect(484, 202, 18, 140, 'box', 3))
b.append(text(512, 336, 'temple arm', 't-muted'))
b.append(rect(462, 222, 58, 44, 'box-on', 5))
b.append(text(528, 242, 'projector:', 't-note'))
b.append(text(528, 259, 'LEDs + LCoS chip', 't-muted'))
b.append(text(528, 276, '+ lenses', 't-muted'))
# in-coupling mirror and zigzag
b.append(line(452, 196, 470, 178, 'edge-on'))
zig = [(491, 222), (461, 187)]
x, up = 461, False
while x > 280:
    x -= 26
    zig.append((x, 180 if up else 194))
    up = not up
b.append(poly(zig, 'path'))
for fx in (230, 262, 294, 326):
    b.append(line(fx - 6, 194, fx + 6, 180, 'edge-on'))
    b.append(ray([(fx, 196), (fx, 262)], 'path', 'arrow-on', 8))
b.append(eye_up(278, 300, 28))
b.append(text(320, 318, 'your right eye', 't-muted'))
b.append(line(278, 272, 262, 92, 'edge-dim'))
b.append(line(278, 272, 300, 92, 'edge-dim'))
for n, (x, y) in enumerate([(446, 244), (436, 208), (410, 162), (214, 222), (330, 128)], 1):
    b.append(badge(x, y, n))
b.append(text(530, 58, 'Five steps', 't-note'))
for i, s in enumerate(['1  the projector makes the picture', '2  it enters the edge of the lens', '3  it zigzags, trapped inside',
                       '4  mirrors send copies out', '5  the eye sees a far screen']):
    b.append(text(530, 80 + 19 * i, s, 't-muted'))
svg('overview', 720, 360, 'Top view of the right side of the glasses. A projector in the temple arm sends light into the edge of the '
    'right lens. The light zigzags inside the lens, partial mirrors send copies of it to the eye, and the eye sees a picture '
    'that seems to float far ahead.',
    b, 'The whole journey in one picture. The rest of this piece takes each numbered step in turn. Sizes are not to scale.')

# =================================================================================================
# 2. The eye focuses near and far
# =================================================================================================
b = [text(10, 22, 'The eye can bend light only so much', 't-title')]
for k, (ox, title, near) in enumerate([(10, 'A far object: parallel rays', False), (370, 'A screen 2 cm away: spreading rays', True)]):
    b.append(text(ox, 52, title, 't-note'))
    ex, ey = ox + 250, 150
    b.append(eye_left(ex, ey, 46))
    if not near:
        for dy in (-26, 0, 26):
            b.append(line(ox + 20, ey + dy, ex - 37, ey + dy, 'path'))
            b.append(line(ex - 37, ey + dy, ex + 44, ey, 'path'))
        b.append(circle(ex + 44, ey, 4, 's3'))
        b.append(text(ox, 232, 'Rays meet on the retina: sharp.', 't-note'))
        b.append(text(ox, 250, 'Extra focusing power needed: 0 dioptres', 't-muted'))
    else:
        sx = ox + 150
        b.append(rect(sx - 4, ey - 30, 8, 60, 'box-on', 2))
        b.append(text(sx, ey - 40, 'screen', 't-muted', 'middle'))
        for dy in (-26, 0, 26):
            b.append(line(sx + 4, ey, ex - 37, ey + dy, 'path'))
            b.append(line(ex - 37, ey + dy, ex + 75, ey - dy * 0.5, 'path'))
        b.append(text(716, ey - 70, 'focus would fall', 't-bad', 'end'))
        b.append(text(716, ey - 54, 'behind the eye', 't-bad', 'end'))
        b.append(text(ox, 232, 'Rays still spread at the retina: a blur.', 't-note'))
        b.append(text(ox, 250, f'Extra focusing power needed: {N["dioptre_2cm"]:.0f} dioptres', 't-muted'))
b.append(text(10, 282, f'A young eye can add about {N["dioptre_25cm"]:.0f} dioptres (it focuses down to about 25 cm). '
                        'Nobody can add 50.', 't-muted'))
svg('eye-focus', 720, 296, 'Left: parallel rays from a far object meet on the retina. Right: rays from a screen 2 cm away spread too much; '
    'the eye cannot bend them enough, so the focus falls behind the retina and the picture blurs.', b,
    'Why you cannot read a screen glued to your glasses. The eye is drawn from the side; the green dot is a sharp focus on the retina.')

# =================================================================================================
# 3. A lens makes a far, virtual image
# =================================================================================================
b = [text(10, 22, 'Put the display at the focal point of a lens', 't-title')]
lx, cy = 250, 150
b.append(rect(96, cy - 50, 8, 100, 'box-on', 2))
b.append(text(100, cy - 60, 'tiny display', 't-muted', 'middle'))
b.append(f'<ellipse class="box-1" cx="{lx}" cy="{cy}" rx="10" ry="70"/>')
b.append(text(lx, cy - 80, 'lens', 't-muted', 'middle'))
b.append(line(104, cy + 70, lx, cy + 70, 'edge-dim'))
b.append(text((104 + lx) / 2, cy + 86, 'focal length f', 't-muted', 'middle'))
for py, cls, hc in ((-36, 'l1', 's1'), (0, 'l3', 's3'), (36, 'l2', 's2')):
    for dy in (-40, 0, 40):
        b.append(line(104, cy + py, lx, cy + dy, cls))
        # after the lens all rays from one point run parallel, along the direction of the chief ray
        slope = -py / (lx - 100)
        b.append(line(lx, cy + dy, 400, cy + dy + slope * (400 - lx), cls))
b.append(eye_left(452, cy, 46))
b.append(text(360, 276, 'Each point of the display leaves as one bundle of parallel rays.', 't-note', 'middle'))
b.append(text(360, 294, 'Parallel rays look as if they come from very far away, so the relaxed eye sees them sharp.', 't-muted', 'middle'))
b.append(text(360, 312, 'Different points leave at different angles: angle on the eye = position on the display.', 't-muted', 'middle'))
svg('virtual-image', 720, 326, 'A small display sits at the focal point of a lens. Three points of the display each become a bundle of '
    'parallel rays after the lens, at three different angles, and all three bundles enter the eye.', b,
    'Collimation, the first trick of every display in glasses. The blue, green and orange bundles start from the top, middle and bottom of the display.')

# =================================================================================================
# 4. Refraction at a surface
# =================================================================================================
b = [text(10, 22, "Light bends when it enters glass (Snell's law)", 't-title')]
cx, cy = 360, 150
b.append(rect(120, cy, 480, 110, 'box-1', 0))
b.append(text(130, cy - 10, 'air, n = 1.0', 't-note'))
b.append(text(130, cy + 24, 'glass, n = 1.5', 't-note'))
b.append(line(cx, 40, cx, 260, 'edge-dim'))
b.append(text(cx + 6, 52, 'normal', 't-muted'))
L = 120
a1, a2 = math.radians(30), math.radians(N['snell_30_out'])
b.append(ray([(cx - L * math.sin(a1), cy - L * math.cos(a1)), (cx, cy)]))
b.append(ray([(cx, cy), (cx + L * math.sin(a2), cy + L * math.cos(a2))]))
b.append(arrow([(cx, cy), (cx + 70 * math.sin(a1), cy - 70 * math.cos(a1))], 'edge-dim', 'arrow', 7))
b.append(path(f'M{cx},{cy - 50} A50,50 0 0 0 {cx - 50 * math.sin(a1):.1f},{cy - 50 * math.cos(a1):.1f}', 'edge-1'))
b.append(text(cx - 48, cy - 52, '30°', 't-s1', 'middle'))
b.append(path(f'M{cx},{cy + 60} A60,60 0 0 0 {cx + 60 * math.sin(a2):.1f},{cy + 60 * math.cos(a2):.1f}', 'edge-1'))
b.append(text(cx + 22, cy + 82, f'{N["snell_30_out"]}°', 't-s1', 'middle'))
b.append(text(cx + 46, cy - 46, 'a little reflects', 't-muted'))
b.append(text(cx + 70, cy + 104, 'bent towards the normal', 't-muted'))
svg('refraction', 720, 270, 'A ray in air hits glass at 30 degrees from the normal and continues inside the glass at 19.5 degrees. '
    'A weak reflected ray goes back up.', b,
    f'A ray from air meets glass at 30° and continues at {N["snell_30_out"]}°: glass slows light, so the ray turns towards the normal. '
    'Angles are always measured from the normal, the dashed line at right angles to the surface.')

# =================================================================================================
# 5. Total internal reflection, three cases
# =================================================================================================
b = [text(10, 22, 'Going the other way: from glass out into air', 't-title')]
crit = N['critical']['common glass / acrylic']
cases = [(30, f'30°: leaves at {N["snell_glass_air_30"]}°', 'escapes, bent away'),
         (crit, f'{crit}°: the critical angle', 'skims along the surface'),
         (55, '55°: past the critical angle', 'all of it reflects')]
for k, (ang, t1, t2) in enumerate(cases):
    ox = 15 + 236 * k
    cx, cy = ox + 110, 110
    b.append(rect(ox, cy, 220, 100, 'box-1', 0))
    b.append(text(ox + 6, cy - 8, 'air', 't-muted'))
    b.append(text(ox + 6, cy + 92, 'glass, n = 1.5', 't-muted'))
    b.append(line(cx, 50, cx, 205, 'edge-dim'))
    a = math.radians(ang)
    L = 80
    b.append(ray([(cx - L * math.sin(a), cy + L * math.cos(a)), (cx, cy)]))
    if k == 0:
        a2 = math.radians(N['snell_glass_air_30'])
        b.append(ray([(cx, cy), (cx + 70 * math.sin(a2), cy - 70 * math.cos(a2))]))
        b.append(arrow([(cx, cy), (cx + 40 * math.sin(a), cy + 40 * math.cos(a))], 'edge-dim', 'arrow', 6))
    elif k == 1:
        b.append(ray([(cx, cy), (cx + 95, cy)]))
    else:
        b.append(ray([(cx, cy), (cx + L * math.sin(a), cy + L * math.cos(a))]))
    b.append(path(f'M{cx},{cy + 36} A36,36 0 0 1 {cx - 36 * math.sin(a):.1f},{cy + 36 * math.cos(a):.1f}', 'edge-1'))
    b.append(text(ox + 110, 236, t1, 't-note', 'middle'))
    b.append(text(ox + 110, 254, t2, 't-s3' if k == 2 else 't-muted', 'middle'))
svg('tir', 720, 268, 'Three rays inside glass hit the top surface. At 30 degrees the ray escapes into the air at 48.6 degrees. '
    'At 41.8 degrees it skims along the surface. At 55 degrees all of it reflects back into the glass.', b,
    'Total internal reflection. Inside glass, a ray that meets the surface at more than the critical angle cannot leave at all, and every bit of it reflects.')

# =================================================================================================
# 6. Critical angle by material
# =================================================================================================
c = N['critical']
hbars('critical', 'Critical angle against air (smaller = more angles stay trapped)',
      [('water, n = 1.333', c['water'], 'box-1'), ('glass or acrylic, n = 1.5', c['common glass / acrylic'], 'box-1'),
       ('high-index glass, n = 1.8', c['best glass Meta had in 2016'], 'box-1'),
       ('lithium niobate, n = 2.3', c['lithium niobate'], 'box-1'),
       ('silicon carbide, n = 2.7', c['silicon carbide (Orion)'], 'box-on')], 60,
      'Bar chart of critical angles: water 48.6, glass 41.8, high-index glass 33.7, lithium niobate 25.8, silicon carbide 21.7 degrees.',
      'The higher the refractive index, the smaller the critical angle, and the wider the range of angles that stay trapped. '
      "Meta's Orion prototype uses silicon carbide for this reason. Computed in numbers.py.", unit='°', fmt='{:.1f}')

# =================================================================================================
# 7. Zigzag in a plate
# =================================================================================================
b = [text(10, 22, 'A ray trapped in a plate walks along it in zigzags', 't-title')]
t, y0, y1 = 70, 80, 150
b.append(rect(60, y0, 600, t, 'box-1', 0))
ang = math.radians(60)
step = t * math.tan(ang)
x, pts, top = 60, [(60, y1)], True
while x + step < 660:
    x += step
    pts.append((x, y0 if top else y1))
    top = not top
b.append(poly(pts, 'path'))
b.append(head(pts[-1][0], pts[-1][1], pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1], 'arrow-on', 9))
b.append(line(40, y0, 40, y1, 'edge'))
b.append(text(30, (y0 + y1) / 2 + 4, 't', 't-note', 'end'))
b.append(line(pts[1][0], 175, pts[3][0], 175, 'edge-1'))
b.append(line(pts[1][0], 168, pts[1][0], 182, 'edge-1'))
b.append(line(pts[3][0], 168, pts[3][0], 182, 'edge-1'))
b.append(text((pts[1][0] + pts[3][0]) / 2, 198, '2 t tan θ', 't-s1', 'middle'))
b.append(line(pts[2][0], y1, pts[2][0], y1 - 46, 'edge-dim'))
b.append(path(f'M{pts[2][0]},{y1 - 34} A34,34 0 0 0 {pts[2][0] - 34 * math.sin(ang):.1f},{y1 - 34 * math.cos(ang):.1f}', 'edge-1'))
b.append(text(pts[2][0] - 22, y1 - 40, 'θ', 't-s1', 'middle'))
b.append(text(60, 230, f'Example: a 1 mm plate and θ = 60° give a bounce on the same face every {N["bounce_60"]} mm,', 't-note'))
b.append(text(60, 248, f'so over 30 mm of lens the light makes about {N["trips_30mm"]:g} round trips. (Example values, not specs.)', 't-muted'))
svg('zigzag', 720, 262, 'A ray zigzags along a glass plate of thickness t at angle theta from the normal. It returns to the same face every '
    '2 t tan theta.', b, 'The light walks along the lens by bouncing between its two faces. Steeper rays (larger θ) take longer strides.')

# =================================================================================================
# 8. Four kinds of microdisplay
# =================================================================================================
b = [text(10, 22, 'Four ways to make a picture smaller than a fingernail', 't-title')]
kinds = [('LCoS', 'box-on', ['a mirror chip with liquid', 'crystal; needs a lamp', '(LEDs) to light it'], 'Ray-Ban Display'),
         ('micro-OLED', 'box-1', ['each pixel is an organic', 'light emitter on silicon;', 'true black'], ''),
         ('microLED', 'box-3', ['each pixel is a tiny LED;', 'very bright, hard to make', 'small in full colour'], 'Orion prototype'),
         ('laser scanning', 'box-4', ['one beam, steered by a', 'moving mirror, draws', 'the picture line by line'], '')]
for k, (name, cls, rows, used) in enumerate(kinds):
    x = 12 + 177 * k
    b.append(rect(x, 40, 165, 200, cls, 8))
    b.append(text(x + 82, 62, name, 't-note', 'middle'))
    cx, cy = x + 82, 112
    if k == 0:
        for i in range(4):
            b.append(rect(cx - 36 + 18 * i, cy, 16, 16, 'box', 1))
        b.append(arrow([(cx - 52, cy - 34), (cx - 22, cy - 2)], 'l4', 's4', 7))
        b.append(arrow([(cx + 14, cy - 2), (cx + 46, cy - 34)], 'path', 'arrow-on', 7))
        b.append(text(cx - 58, cy - 38, 'LED', 't-muted', 'middle'))
    elif k in (1, 2):
        for i in range(4):
            b.append(rect(cx - 36 + 18 * i, cy, 16, 16, 'box', 1))
            b.append(arrow([(cx - 28 + 18 * i, cy - 2), (cx - 28 + 18 * i, cy - 26)], 'path', 'arrow-on', 6))
    else:
        b.append(rect(cx - 50, cy + 2, 18, 14, 'box', 2))
        b.append(line(cx - 32, cy + 9, cx, cy + 9, 'path'))
        b.append(line(cx - 8, cy + 18, cx + 8, cy, 'edge'))
        for dy in (-34, -20, -6):
            b.append(line(cx, cy + 9, cx + 46, cy + dy, 'path'))
    for i, s in enumerate(rows):
        b.append(text(x + 82, 160 + 17 * i, s, 't-muted', 'middle'))
    if used:
        b.append(text(x + 82, 228, used, 't-on', 'middle'))
svg('microdisplays', 720, 252, 'Four boxes: LCoS reflects light from LEDs; micro-OLED and microLED pixels make their own light; '
    'laser scanning draws the picture with one steered beam.', b,
    'The four main families of tiny displays used in head-worn devices. Only LCoS needs a separate light source; '
    'the other three emit light themselves. LCoS in Ray-Ban Display is reported by the iFixit teardown; microLED in Orion is from Meta.')

# =================================================================================================
# 9. An LCoS pixel
# =================================================================================================
b = [text(10, 22, 'Inside an LCoS chip: two pixels, side by side', 't-title')]
layers = [('cover glass', 'box', 24), ('clear electrode', 'box-ghost', 18), ('liquid crystal', 'box-1', 34),
          ('mirror pixels', 'box-on', 18), ('silicon circuits', 'box-4', 30)]
y = 140
X0, X1 = 140, 560
for name, cls, h in layers:
    b.append(rect(X0, y, X1 - X0, h, cls, 0))
    b.append(text(X1 + 10, y + h / 2 + 4, name, 't-note'))
    y += h
b.append(line((X0 + X1) / 2, 182, (X0 + X1) / 2, 264, 'edge'))
for k, (cx, state, out, res) in enumerate([(245, 'pixel OFF', 'polarisation unchanged', 'beam splitter blocks it: dark'),
                                            (455, 'pixel ON', 'polarisation turned', 'beam splitter passes it: bright')]):
    b.append(arrow([(cx - 40, 66), (cx - 8, 216)], 'l4', 's4', 8))
    b.append(arrow([(cx + 8, 216), (cx + 40, 66)], 'path' if k else 'edge-dim', 'arrow-on' if k else 'arrow', 8))
    b.append(text(cx, 52, state, 't-on' if k else 't-note', 'middle'))
    b.append(text(cx - 44, 92, 'in', 't-muted', 'end'))
    b.append(text(cx + 46, 92, 'out', 't-muted'))
    b.append(text(cx, 290, out, 't-note', 'middle'))
    b.append(text(cx, 308, res, 't-muted', 'middle'))
svg('lcos', 720, 322, 'Cross-section of an LCoS chip: cover glass, a clear electrode, liquid crystal, mirror pixels and silicon circuits. '
    'Light enters, passes the liquid crystal twice and reflects off the mirror. An OFF pixel keeps its polarisation and is '
    'blocked; an ON pixel turns it and passes.', b,
    'LCoS means liquid crystal on silicon. The chip does not make light. Each pixel decides how much of the incoming light '
    'it turns, and a polarising beam splitter turns that into bright or dark.')

# =================================================================================================
# 10. The light engine
# =================================================================================================
b = [text(10, 22, 'The projector in the temple arm (reported layout)', 't-title')]
b.append(box(10, 60, 92, 56, 'box-4', ['LEDs', 'red, green, blue']))
b.append(box(122, 60, 92, 56, 'box', ['mix', 'dichroic mirror']))
b.append(box(234, 60, 92, 56, 'box', ['even out', 'fly-eye lenses']))
b.append(box(346, 52, 92, 72, 'box-on', ['beam', 'splitter', '(PBS)']))
b.append(box(346, 160, 92, 50, 'box-1', ['LCoS chip', '600 x 600']))
b.append(box(458, 60, 112, 56, 'box', ['collimate', 'lenses + mirror']))
b.append(box(590, 60, 120, 56, 'box-3', ['waveguide', 'entrance']))
for x1, x2 in ((102, 122), (214, 234), (326, 346), (438, 458), (570, 590)):
    b.append(arrow([(x1, 88), (x2 - 1, 88)], 'path', 'arrow-on', 7))
b.append(arrow([(380, 124), (380, 159)], 'l4', 's4', 7))
b.append(arrow([(404, 159), (404, 125)], 'path', 'arrow-on', 7))
b.append(text(372, 146, 'in', 't-muted', 'end'))
b.append(text(412, 146, 'out', 't-muted'))
steps = ['1. LEDs flash one colour at a time.', '2. A colour mirror and tiny lens arrays make the light even.',
         '3. The beam splitter sends it to the LCoS chip, which reflects the picture back.',
         '4. Lenses make every pixel a parallel bundle at its own angle.', '5. The bundles enter the waveguide.']
for i, s in enumerate(steps):
    b.append(text(10, 238 + 18 * i, s, 't-muted'))
svg('engine', 720, 334, 'Block diagram: LEDs, a dichroic mirror, fly-eye lenses, a polarising beam splitter that sends light to the '
    'LCoS chip and back, collimating optics, then the waveguide entrance.', b,
    'The light engine, as described in a teardown analysis by Karl Guttag. Meta has not published this layout; '
    'the parts and order here are reported, not official.')

# =================================================================================================
# 11. Field-sequential colour
# =================================================================================================
b = [text(10, 22, 'One chip, three colours: colour by time', 't-title')]
x0, x1 = 110, 690
fw = (x1 - x0) / 9
for i in range(9):
    cls = ['box-2', 'box-3', 'box-1'][i % 3]
    b.append(rect(x0 + i * fw, 60, fw - 2, 34, cls, 2))
    b.append(text(x0 + i * fw + fw / 2, 82, 'RGB'[i % 3], 't-note', 'middle'))
b.append(text(100, 82, 'LED on', 't-note', 'end'))
for j in range(3):
    xa, xb = x0 + 3 * j * fw, x0 + 3 * (j + 1) * fw - 2
    b.append(line(xa, 108, xb, 108, 'edge-1'))
    b.append(text((xa + xb) / 2, 126, f'one display frame, {N["frame_ms"]} ms', 't-tick', 'middle'))
b.append(line(x0, 150, x1 - 2, 150, 'edge-2'))
b.append(text((x0 + x1) / 2, 168, f'one content update (30 Hz), {1000 / 30:.1f} ms', 't-s2', 'middle'))
b.append(text(10, 200, f'Each colour field lasts at most {N["field_ms"]} ms. The chip shows the red part of the picture while only the', 't-muted'))
b.append(text(10, 218, 'red LED is on, then green, then blue. The eye blends them into one full-colour picture.', 't-muted'))
svg('fsc', 720, 232, 'A timeline of nine coloured blocks, red, green and blue repeating. Every three blocks make one 11.1 millisecond '
    'display frame, and three frames make one 33.3 millisecond content update.', b,
    f'Field-sequential colour at the reported 90 Hz display rate: at least {N["fields"]} colour fields per second. '
    'If the eye moves fast, the colours can separate for an instant ("colour break-up").')

# =================================================================================================
# 12. The waveguide cross-section
# =================================================================================================
b = [text(10, 22, 'Inside the lens: in, along, and out (geometric waveguide)', 't-title')]
gy0, gy1 = 100, 150
b.append(rect(40, gy0, 640, gy1 - gy0, 'box-1', 0))
b.append(text(44, gy0 - 8, 'glass plate, about the thickness of a coin (not to scale)', 't-muted'))
# in-coupling mirror on the right
b.append(line(630, gy1, 670, gy0, 'edge-on'))
b.append(text(676, gy0 - 8, 'in', 't-on', 'end'))
b.append(ray([(650, 230), (650, 118)], 'path', 'arrow-on', 8))
b.append(text(658, 220, 'from the', 't-muted'))
b.append(text(658, 236, 'projector', 't-muted'))
pts = [(650, 117)]
x, top = 650, True
step = 34
while x > 190:
    x -= step
    pts.append((x, gy0 if top else gy1))
    top = not top
b.append(poly(pts, 'path'))
b.append(text(480, 175, 'trapped by total internal reflection', 't-note', 'middle'))
facets = [120, 175, 230, 285, 340]
for i, fx in enumerate(facets):
    b.append(line(fx - 16, gy1, fx + 16, gy0, 'edge-on'))
    b.append(ray([(fx, gy1), (fx, 236)], 'path', 'arrow-on', 7))
b.append(text(40, 56, 'partial mirrors: each sends a little light out, passes the rest on', 't-note'))
b.append(eye_up(230, 284, 26))
b.append(line(100, 252, 360, 252, 'edge-dim'))
b.append(line(100, 246, 100, 258, 'edge-dim'))
b.append(line(360, 246, 360, 258, 'edge-dim'))
b.append(text(370, 256, 'eyebox: anywhere here, the eye sees the picture', 't-muted'))
svg('waveguide', 720, 322, 'Cross-section of a geometric waveguide. Light from the projector enters at the right through an angled mirror, '
    'zigzags left by total internal reflection, and five slanted partial mirrors each send part of it down to the eye.', b,
    'The three jobs of the waveguide: couple the light in, carry it by total internal reflection, and couple it out in many copies. '
    'The copies fill a region called the eyebox.')

# =================================================================================================
# 13. The reflectance ladder
# =================================================================================================
b = [text(10, 22, 'Sharing the light fairly between mirrors', 't-title')]
for r_, (rs, title) in enumerate([([1 / 3, 1 / 2, 1.0], 'Patent example A: 33%, 50%, 100%'),
                                   ([0.20, 0.25, 1 / 3], 'Patent example B: 20%, 25%, 33%')]):
    y = 50 + 160 * r_
    b.append(text(10, y, title, 't-note'))
    left = 1.0
    for i, r in enumerate(rs):
        x = 40 + 210 * i
        b.append(rect(x, y + 26, 120 * left, 22, 'box-1', 3))
        b.append(text(x, y + 66, f'arrives: {100 * left:.0f}%', 't-tick'))
        out = left * r
        b.append(line(x + 140, y + 60, x + 160, y + 16, 'edge-on'))
        b.append(text(x + 166, y + 30, f'{100 * r:.0f}%', 't-on'))
        b.append(ray([(x + 150, y + 64), (x + 150, y + 104)], 'path', 'arrow-on', 7))
        b.append(text(x + 150, y + 124, f'to eye {100 * out:.0f}%', 't-note', 'middle'))
        left -= out
    if left > 0.01:
        b.append(text(712, y + 66, f'{100 * left:.0f}% left', 't-muted', 'end'))
b.append(text(10, 358, 'B is closer to real see-through optics: a 100% mirror would block your view of the world.', 't-muted'))
svg('ladder', 720, 372, 'Two rows of three mirrors. With reflectances 33, 50 and 100 percent each mirror sends one third of the light to the eye. '
    'With 20, 25 and 33 percent each sends 20 percent and 40 percent is left over.', b,
    'Each mirror sees less light than the one before, so it must reflect a larger share. Both examples are from Lumus patent '
    'US 6,829,095; the arithmetic is in numbers.py.')

# =================================================================================================
# 14. Why many copies: the eyebox
# =================================================================================================
b = [text(10, 22, 'Why the picture must be copied: your eye moves', 't-title')]
for k, (title, beams) in enumerate([('One exit beam', [190]), ('Many copies of the beam', [110, 150, 190, 230, 270])]):
    ox = 360 * k
    b.append(text(ox + 20, 50, title, 't-note'))
    b.append(rect(ox + 60, 70, 260, 18, 'box-1', 0))
    for bx in beams:
        b.append(f'<rect class="box-on" x="{ox + bx - 8}" y="88" width="16" height="120" rx="0" opacity="0.6"/>')
    for i, (ex, ok) in enumerate([(ox + 120, None), (ox + 190, None), (ox + 270, None)]):
        hit = any(abs(ex - (ox + bx)) < 14 for bx in beams)
        b.append(eye_up(ex, 230, 16))
        b.append(text(ex, 270, 'sees it' if hit else 'nothing', 't-s3' if hit else 't-bad', 'middle'))
b.append(text(10, 300, 'The beam that leaves the projector is only a few millimetres wide. Copying it side by side', 't-muted'))
b.append(text(10, 318, 'makes a wider box of light, so the picture stays visible when the eye or the frame shifts.', 't-muted'))
svg('eyebox', 720, 330, 'Left: a single narrow exit beam reaches the eye only in one position. Right: five copies side by side reach the eye '
    'in all three positions.', b,
    'Pupil expansion. Each copy carries the whole picture, because each pixel is a direction, not a place.')

# =================================================================================================
# 15. Two-dimensional expansion
# =================================================================================================
b = [text(10, 22, 'Copying in two directions (front view of the lens)', 't-title')]
b.append(f'<ellipse class="box-1" cx="300" cy="170" rx="240" ry="120"/>')
b.append(rect(560, 64, 34, 34, 'box-on', 3))
b.append(text(600, 116, 'beam from', 't-muted'))
b.append(text(600, 132, 'the projector', 't-muted'))
b.append(rect(170, 66, 380, 30, 'box-3', 3))
for i in range(8):
    x = 520 - 44 * i
    b.append(line(x - 8, 92, x + 8, 70, 'edge-on'))
b.append(ray([(560, 81), (190, 81)], 'path', 'arrow-on', 8))
b.append(text(170, 44, 'first set of mirrors: copies the beam sideways', 't-note'))
for i in range(8):
    x = 520 - 44 * i
    for j in range(4):
        b.append(circle(x, 150 + 32 * j, 6, 'box-on'))
    b.append(line(x, 96, x, 242, 'edge-dim'))
b.append(text(300, 312, 'second set: copies each one downwards and sends it to the eye', 't-note', 'middle'))
b.append(text(640, 180, '1 beam in', 't-note'))
b.append(text(640, 198, f'8 x 4 = 32', 't-on'))
b.append(text(640, 216, 'copies out', 't-note'))
svg('expand2d', 720, 326, 'Front view of a lens. A small beam enters at the top right, a strip of mirrors copies it sideways into a row, '
    'and a second set of mirrors copies each of those downwards into a grid of exit points.', b,
    'Two-dimensional pupil expansion, drawn as a sketch. The numbers of mirrors and copies are made up for the drawing; '
    'the real counts in the Ray-Ban Display waveguide are not public.')

# =================================================================================================
# 16. Geometric versus diffractive
# =================================================================================================
b = [text(10, 22, 'Two ways to get light out of the lens', 't-title')]
for k, title in enumerate(['Geometric (reflective): tiny mirrors', 'Diffractive: nano-scale gratings']):
    ox = 10 + 360 * k
    b.append(text(ox, 52, title, 't-note'))
    b.append(rect(ox + 10, 90, 320, 40, 'box-1', 0))
    b.append(text(ox + 10, 80, 'world side', 't-muted'))
    pts, x, top = [(ox + 320, 110)], ox + 320, True
    while x > ox + 30:
        x -= 30
        pts.append((x, 90 if top else 130))
        top = not top
    b.append(poly(pts, 'path'))
    for i in range(4):
        fx = ox + 70 + 60 * i
        if k == 0:
            b.append(line(fx - 12, 130, fx + 12, 90, 'edge-on'))
            b.append(ray([(fx, 130), (fx, 180)], 'path', 'arrow-on', 7))
        else:
            for j in range(5):
                b.append(line(fx - 10 + 5 * j, 126, fx - 10 + 5 * j, 134, 'edge-on'))
            b.append(ray([(fx, 134), (fx - 6, 180)], 'l1', 's1', 7))
            b.append(ray([(fx, 134), (fx, 180)], 'l3', 's3', 7))
            b.append(ray([(fx, 134), (fx + 6, 180)], 'l2', 's2', 7))
            b.append(ray([(fx, 90), (fx, 60)], 'edge-dim', 'arrow', 6))
    b.append(text(ox + 170, 200, 'to the eye', 't-muted', 'middle'))
    rows = (['Mirrors reflect every colour the same way.', 'Little light leaks forward:',
             'about 1.5% "eye glow" (measured).', 'Harder to make: many glued slices.'] if k == 0 else
            ['The grating bends each colour differently.', 'Much light leaks forward: often',
             '50 to 100% of the eye-side brightness.', 'Easier to mass-produce, like a stamp.'])
    for i, s in enumerate(rows):
        b.append(text(ox, 228 + 18 * i, s, 't-muted'))
svg('geo-diff', 720, 306, 'Left: a geometric waveguide with slanted partial mirrors sends white light down to the eye. Right: a diffractive '
    'waveguide with fine gratings splits the light into colours and also sends some of it forward, out of the front.', b,
    'The two main families of waveguide. The leak figures are from Karl Guttag\'s measurements and survey, not from Meta '
    '(Meta says 2% light leakage).')

# =================================================================================================
# 17. A grating splits colours
# =================================================================================================
g = N['grating']
b = [text(10, 22, 'A grating sends each colour into the glass at its own angle', 't-title')]
cx, cy = 300, 230
b.append(rect(60, 80, 600, 150, 'box-1', 0))
for j in range(30):
    b.append(line(225 + 5 * j, 226, 225 + 5 * j, 234, 'edge-on'))
b.append(text(320, 252, 'grating (example period 400 nm) on the bottom face', 't-muted'))
b.append(ray([(cx, 300), (cx, 236)], 'l4', 's4', 8))
b.append(text(cx + 8, 290, 'white light', 't-muted'))
b.append(line(cx, cy, cx, 70, 'edge-dim'))
L = 150
for colour, cls, hc in (('blue', 'l1', 's1'), ('green', 'l3', 's3')):
    a = math.radians(g[colour])
    b.append(ray([(cx, cy), (cx + L * math.sin(a), cy - L * math.cos(a))], cls, hc, 8))
    b.append(text(cx + L * math.sin(a) + 8, cy - L * math.cos(a) + 4, f'{colour} 450 nm: {g[colour]}°' if colour == 'blue'
                  else f'green 530 nm: {g[colour]}°', 't-s1' if colour == 'blue' else 't-s3'))
cr = math.radians(N['critical']['common glass / acrylic'])
b.append(line(cx, cy, cx - 160 * math.sin(cr), cy - 160 * math.cos(cr), 'edge-dim'))
b.append(text(cx - 160 * math.sin(cr) - 6, cy - 160 * math.cos(cr) + 4, f'critical angle {N["critical"]["common glass / acrylic"]}°', 't-muted', 'end'))
b.append(text(470, 214, 'red 630 nm: no ray inside at all', 't-s2'))
svg('grating', 720, 310, 'White light enters a glass plate through a grating. Blue light bends to 48.6 degrees and green to 62.0 degrees, '
    'both past the critical angle of 41.8 degrees. Red light has no first-order ray inside the glass.', b,
    'The grating equation at work (computed in numbers.py). One grating cannot treat three colours alike, which is why many '
    'diffractive designs stack two or three plates, and why they can show rainbow tints.')

# =================================================================================================
# 18. Luminance scale
# =================================================================================================
b = [text(10, 22, 'How bright is bright? Luminance in nits (log scale)', 't-title')]
x0, x1 = 60, 680
lo, hi = 10, 20000
sx = lambda v: x0 + (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (x1 - x0)
AY = 180
b.append(line(x0, AY, x1, AY, 'axis'))
for v in (10, 100, 1000, 10000):
    b.append(line(sx(v), AY - 5, sx(v), AY + 5, 'axis'))
    b.append(text(sx(v), AY + 22, f'{v:,}', 't-tick', 'middle'))
items = [(30, 'display, lowest', 'box-on', 1), (80, 'office monitor (sRGB)', 'box', -1),
         (250, 'typical LCD monitor, peak', 'box', 1), (2000, 'average cloudy sky', 'box-4', -1),
         (5000, 'display, highest', 'box-on', 1), (8000, 'average clear sky', 'box-4', 2)]
for v, name, cls, d in items:
    x = sx(v)
    b.append(circle(x, AY, 6, cls))
    anc = 'end' if v == 8000 else 'middle'
    if d > 0:
        top = AY - 46 if d == 1 else AY - 104
        b.append(line(x, AY - 6, x, top, 'edge-dim'))
        b.append(text(x, top - 8, f'{v:,}', 't-val', anc))
        b.append(text(x, top - 24, name, 't-note', anc))
    else:
        b.append(line(x, AY + 30, x, AY + 46, 'edge-dim'))
        b.append(text(x, AY + 62, f'{v:,}', 't-val', 'middle'))
        b.append(text(x, AY + 78, name, 't-note', 'middle'))
svg('nits', 720, 272, 'A log scale from 10 to 20,000 nits. Display lowest 30; office monitor 80; typical LCD peak 250; average cloudy sky 2,000; '
    'display highest 5,000; average clear sky 8,000.', b,
    'A nit is one candela per square metre. The display range is from Road to VR\'s launch spec list; the sky and monitor '
    'values are typical figures from a standard table of luminances. Each step on the axis is ten times brighter.')

# =================================================================================================
# 19. Contrast against the world
# =================================================================================================
cr = N['contrast']
rows = [('5,000 nits, clear sky, clear lens', cr['5,000 nits on clear sky, clear lens'], 'box-1'),
        ('5,000 nits, cloudy sky, clear lens', cr['5,000 nits on cloudy sky, clear lens'], 'box-1'),
        ('5,000 nits, clear sky, lens at 18%', cr['5,000 nits on clear sky, lens at 18%'], 'box-on'),
        ('5,000 nits, clear sky, lens at 8%', cr['5,000 nits on clear sky, lens at 8%'], 'box-on'),
        ('250 nits, clear sky, clear lens', cr['250 nits (monitor level) on clear sky, clear lens'], 'box-2')]
hbars('contrast', 'See-through contrast: (display + world) / world', rows, 10,
      'Bar chart of contrast ratios: 1.62 on a clear sky with a clear lens, 3.50 on a cloudy sky, 4.47 and 8.81 with the lens '
      'darkened to 18 and 8 percent, and 1.03 for a 250-nit display.',
      'Contrast of a see-through display. A darkening lens (sunglass category 3 lets 8 to 18% through) dims the world but, '
      'as the next sections show, not the display. Computed in numbers.py.', unit=' : 1', x0=290, x1=600, fmt='{:.2f}')

# =================================================================================================
# 20. The lens stack
# =================================================================================================
b = [text(10, 22, 'The right lens is a sandwich (side view)', 't-title')]
layers = [(150, 26, 'box-4', 'outer lens', 'darkens in sunlight'), (285, 30, 'box-1', 'waveguide', 'carries the picture'),
          (420, 26, 'box-3', 'inner lens', 'your prescription')]
for x, w, cls, a, c2 in layers:
    b.append(rect(x, 60, w, 190, cls, 4))
    b.append(text(x + w / 2, 272, a, 't-note', 'middle'))
    b.append(text(x + w / 2, 290, c2, 't-muted', 'middle'))
b.append(text(40, 52, 'world', 't-note'))
b.append(text(560, 52, 'eye', 't-note'))
b.append(eye_left(600, 150, 34))
for y in (90, 120):
    b.append(line(30, y, 150, y, 'l4'))
    b.append(f'<line class="l4" x1="176" y1="{y}" x2="562" y2="{y + (150 - y) * 0.5:.0f}" stroke-dasharray="3 4"/>')
b.append(text(30, 150, 'sunlight: dimmed', 't-s4'))
b.append(text(30, 166, 'by the outer lens', 't-s4'))
b.append(ray([(300, 214), (300, 196), (566, 166)], 'path', 'arrow-on', 8))
b.append(text(452, 216, 'display light starts inside,', 't-on'))
b.append(text(452, 234, 'so the darkening never', 't-on'))
b.append(text(452, 252, 'touches it', 't-on'))
svg('stack', 720, 304, 'A side view of the right lens: an outer darkening lens, the waveguide, and an inner prescription lens. Sunlight passes '
    'all three and is dimmed by the outer lens. Display light starts in the waveguide and passes only the inner lens.', b,
    'Why darkening lenses help the display. iFixit reports that the lenses are glued on both sides of the waveguide and that one '
    'of them darkens; the order drawn here (darkening layer outside) is the arrangement that makes the trick work.')

# =================================================================================================
# 21. Field of view and pixels per degree
# =================================================================================================
b = [text(10, 22, 'Field of view and pixels per degree', 't-title')]
S = 200
x0, y0 = 60, 50
b.append(rect(x0, y0, S, S, 'box-1', 0))
for i in range(1, 10):
    b.append(line(x0 + S * i / 10, y0, x0 + S * i / 10, y0 + S, 'grid'))
    b.append(line(x0, y0 + S * i / 10, x0 + S, y0 + S * i / 10, 'grid'))
b.append(line(x0, y0 + S, x0 + S, y0, 'edge-on'))
b.append(text(x0 + S / 2 + 18, y0 + S / 2 - 8, '20° diagonal', 't-on'))
b.append(text(x0 + S / 2, y0 + S + 20, f'{N["side_deg"]}° wide, 600 pixels', 't-note', 'middle'))
b.append(text(x0 - 10, y0 + S / 2, f'{N["side_deg"]}°', 't-note', 'middle', -90))
rx = 330
eqs = [f'diagonal pixels = 600 x √2 = {N["diag_px"]}', f'pixels per degree = {N["diag_px"]} / 20 = {N["ppd"]}',
       f'one pixel = 60 / {N["ppd"]} = {N["arcmin_per_px"]} arcminutes', '20/20 vision sees detail of about 1 arcminute']
for i, s in enumerate(eqs):
    b.append(text(rx, 66 + 22 * i, s, 't-note' if i < 3 else 't-muted'))
bars = [('20/20 eye', 60, 'box-3'), ('Meta spec', N['ppd'], 'box-on'), ('effective (estimate)', N['ppd_eff_400'], 'box-2')]
for i, (name, v, cls) in enumerate(bars):
    y = 170 + 30 * i
    b.append(text(rx + 136, y + 15, name, 't-note', 'end'))
    b.append(rect(rx + 146, y, 3.0 * v, 20, cls, 3))
    b.append(text(rx + 152 + 3.0 * v, y + 15, f'{v:g} ppd', 't-val'))
svg('ppd', 720, 290, 'A square grid of 600 by 600 pixels spans 14.14 degrees on each side and 20 degrees on the diagonal. Pixels per degree: '
    '60 for 20/20 vision, 42.4 for the Meta spec, and 28.3 effective by an outside estimate.', b,
    'Pixels per degree measures sharpness the way the eye does. The "effective" bar uses Karl Guttag\'s estimate that the '
    'picture resolves about 400 x 400 pixels; it is an outside estimate, not a Meta figure.')

# =================================================================================================
# 22. Field of view to scale
# =================================================================================================
b = [text(10, 22, 'How big is 20 degrees? Fields of view drawn to scale', 't-title')]
cx, cy, k = 190, 180, 5.2
for i, (diag, name, cls) in enumerate([(70, '70°: Orion prototype, Lumus ZOE demo', 'box-ghost'),
                                       (30, '30° diagonal: Lumus Z-30 module', 'box-4'),
                                       (20, '20° diagonal: Meta Ray-Ban Display', 'box-on')]):
    s = diag / math.sqrt(2) * k
    b.append(rect(cx - s / 2, cy - s / 2, s, s, cls, 2))
    b.append(rect(390, 110 + 40 * i, 22, 22, cls, 2))
    b.append(text(420, 126 + 40 * i, name, 't-note' if diag == 20 else 't-muted'))
b.append(text(10, 340, f'At 42 pixels per degree, a 70° diagonal square would need about {N["px_70_at42"]:,} x {N["px_70_at42"]:,} pixels.', 't-muted'))
svg('fov', 720, 352, 'Three nested squares to scale: a small 20 degree square for Ray-Ban Display, a 30 degree square for the Lumus Z-30, '
    'and a large 70 degree square for the Orion prototype and the Lumus ZOE demo.', b,
    'The Ray-Ban Display picture is a small window. Orion is a prototype and ZOE is a waveguide demo, not products you can buy.')

# =================================================================================================
# 23. Light budget: where the projector light goes
# =================================================================================================
b = [text(10, 22, 'Where 1 lumen of projector light ends up (3 mm pupil)', 't-title')]
side = N['ideal_box_side']
k = 4
b.append(rect(60, 50, side * k, side * k, 'box-1', 0))
b.append(text(60 + side * k / 2, 50 + side * k + 18, f'{side} x {side} mm: the area 1 lm could fill at 5,000 nits', 't-muted', 'middle'))
b.append(circle(60 + side * k / 2, 50 + side * k / 2, 1.5 * k, 'box-on'))
b.append(arrow([(60 + side * k / 2 + 40, 50 + side * k / 2 - 30), (60 + side * k / 2 + 8, 50 + side * k / 2 - 4)], 'edge', 'arrow', 7))
b.append(text(60 + side * k / 2 + 44, 50 + side * k / 2 - 34, 'a 3 mm pupil', 't-note'))
rx = 330
rows = ['Light the eye needs at 5,000 nits:', f'  luminance x pupil area x solid angle',
        f'  = 5,000 x 7.07 mm² x {N["omega"]} sr', f'  = {N["pupil3_mlm"]} millilumens', '',
        f'That is {N["pupil3_pct"]}% of the 1 lumen out of the projector.',
        'The rest lights up the eyebox around the pupil,', 'is lost at the mirrors and edges, or leaks out.']
for i, s in enumerate(rows):
    b.append(text(rx, 70 + 20 * i, s, 't-note' if i in (0, 3, 5) else 't-muted'))
svg('budget', 720, 300, 'A large square stands for the area that one lumen could light at 5,000 nits; a tiny circle in its middle is a 3 millimetre '
    'pupil. Text: the pupil receives 2.14 millilumens, 0.21 percent of the projector output.', b,
    'A light budget from two measured numbers in a teardown (about 1 lumen out of the projector, about 5,000 nits at the eye) '
    'and the geometry of the picture. Computed in numbers.py; the pupil size is an example within the normal daylight range.')

# =================================================================================================
# 24. From thought to wristband signal
# =================================================================================================
b = [text(10, 22, 'From a decision to move to a signal on the skin', 't-title')]
steps = [('brain', 'decides to pinch'), ('spinal cord', 'neurons fire'), ('nerve', 'carries spikes'),
         ('forearm', 'muscles contract'), ('skin at wrist', 'tiny voltages'), ('wristband', '16 channels')]
for i, (a, c2) in enumerate(steps):
    x = 10 + 118 * i
    b.append(box(x, 50, 104, 54, 'box-on' if i == 5 else ('box-3' if i == 3 else 'box'), [a, c2]))
    if i:
        b.append(arrow([(x - 14, 77), (x - 1, 77)], 'edge', 'arrow', 6))
wy = 170
pts = []
for i in range(300):
    t = i / 299
    v = 0
    for c0, amp in ((0.25, 1.0), (0.32, -0.7), (0.62, 0.8), (0.70, -0.5)):
        v += amp * math.exp(-((t - c0) / 0.012) ** 2)
    v += 0.06 * math.sin(i * 1.7) * math.sin(i * 0.31)
    pts.append((60 + 600 * t, wy + 28 - 38 * v))
b.append(poly(pts, 'l3'))
b.append(text(60, 150, 'one wristband channel (a sketch): motor-unit spikes riding on noise', 't-note'))
b.append(text(60, 240, f'The research band records 16 channels at 2,000 samples a second: {N["emg_rate"]:,} numbers every second,', 't-muted'))
b.append(text(60, 258, 'with noise of only 2.46 microvolts (Nature, 2025).', 't-muted'))
svg('emg-chain', 720, 272, 'A chain of six boxes: brain, spinal cord, nerve, forearm muscles, skin at the wrist, wristband. Below, a sketch of one '
    'signal channel with sharp spikes over low noise.', b,
    'Surface electromyography (sEMG) listens to muscles, not to the brain. The spikes are motor-unit action potentials: the '
    'electrical sign of a group of muscle fibres firing. The waveform is a drawing, not recorded data.')

# =================================================================================================
# 25. The decoder
# =================================================================================================
b = [text(10, 22, 'How a network turns 16 signals into a gesture (research models)', 't-title')]
row = [('16 sEMG channels', '2 kHz', 'box'), ('1D convolution', 'local patterns', 'box-1'),
       ('3 LSTM layers', 'memory over time', 'box-1'), ('classifier', 'which gesture?', 'box-on')]
for i, (a, c2, cls) in enumerate(row):
    x = 10 + 178 * i
    b.append(box(x, 62, 150, 56, cls, [a, c2]))
    if i:
        b.append(arrow([(x - 26, 90), (x - 2, 90)], 'edge', 'arrow', 7))
b.append(text(10, 52, 'Gestures (index pinch, thumb tap, thumb swipes...)', 't-note'))
row2 = [('16 sEMG channels', '2 kHz', 'box'), ('power features', 'per frequency band', 'box-1'),
        ('conformer', 'attention + convolution', 'box-1'), ('letters', 'handwriting', 'box-on')]
for i, (a, c2, cls) in enumerate(row2):
    x = 10 + 178 * i
    b.append(box(x, 172, 150, 56, cls, [a, c2]))
    if i:
        b.append(arrow([(x - 26, 200), (x - 2, 200)], 'edge', 'arrow', 7))
b.append(text(10, 162, 'Handwriting (letters drawn with a finger on a surface)', 't-note'))
b.append(text(10, 262, 'Trained on data from thousands of people, so a new wearer needs no calibration.', 't-muted'))
svg('decoder', 720, 276, 'Two pipelines. Gestures: 16 channels, a 1D convolution, three LSTM layers, a classifier. Handwriting: 16 channels, '
    'power features per frequency band, a conformer, letters.', b,
    'The two decoders described in the Nature paper. The product\'s own models are not public; Meta says they were trained on data '
    'from nearly 200,000 consenting participants.')

# =================================================================================================
# 26. The whole system
# =================================================================================================
b = [text(10, 22, 'The whole system', 't-title')]
b.append(rect(10, 40, 470, 270, 'box-ghost', 10))
b.append(text(20, 60, 'glasses (69 g)', 't-note'))
b.append(box(24, 74, 140, 62, 'box', ['left arm', 'battery 960 mWh', 'speakers'], gap=16))
b.append(box(176, 74, 140, 62, 'box', ['frame', '12 MP camera', '5 microphones'], gap=16))
b.append(box(328, 74, 140, 62, 'box-on', ['right arm', 'Snapdragon AR1', '2 GB RAM, 32 GB'], gap=16))
b.append(box(176, 168, 140, 56, 'box-1', ['right lens', 'waveguide display']))
b.append(box(328, 168, 140, 56, 'box-on', ['projector', 'LEDs + LCoS']))
b.append(arrow([(398, 137), (398, 167)], 'edge', 'arrow', 7))
b.append(arrow([(328, 196), (317, 196)], 'path', 'arrow-on', 7))
b.append(text(24, 260, 'camera, microphones, speakers and display', 't-muted'))
b.append(text(24, 278, 'all connect to the chip in the right arm', 't-muted'))
b.append(box(530, 50, 180, 62, 'box-3', ['Meta Neural Band', 'sEMG wristband', '18 h battery'], gap=16))
b.append(box(530, 150, 180, 56, 'box', ['phone', 'Meta AI app']))
b.append(box(530, 244, 180, 56, 'box', ['internet', 'Meta AI, messages']))
b.append(arrow([(529, 82), (469, 100)], 'edge-3', 's3', 7))
b.append(text(500, 76, 'gestures', 't-muted', 'middle'))
b.append(poly([(481, 178), (529, 178)], 'edge'))
b.append(head(529, 178, 1, 0, 'arrow', 7))
b.append(head(481, 178, -1, 0, 'arrow', 7))
b.append(text(505, 170, 'wireless', 't-muted', 'middle'))
b.append(poly([(620, 207), (620, 243)], 'edge'))
b.append(head(620, 243, 0, 1, 'arrow', 7))
b.append(head(620, 207, 0, -1, 'arrow', 7))
svg('system', 720, 322, 'Block diagram. The glasses hold a left arm with battery and speakers, a front with camera and microphones, a right arm '
    'with the Snapdragon AR1 chip, and a projector feeding the waveguide in the right lens. A Neural Band sends gestures to the '
    'glasses; the glasses talk wirelessly to a phone, which talks to the internet.', b,
    'Sources: Meta (69 g, Neural Band 18 h), Road to VR launch specs (12 MP camera, five microphones), iFixit teardown (battery, '
    'chip, memory, which arm holds what). Where each AI task runs (glasses, phone or cloud) is not fully public.')

json.dump(F, open(os.path.join(HERE, 'results', 'figs.json'), 'w'), indent=0)
print(len(F), 'figures:', ', '.join(F))
