"""Figures for the writing "NVIDIA GPUs: every architecture and every product".
SVG with CSS classes only (src/styles/prose.css .fig rules), so dark, dim and light all work.
Every number drawn here is in prices.json or is a public spec."""
import json, math, pathlib
HERE = pathlib.Path(__file__).parent
P = json.loads((HERE / 'prices.json').read_text())

def svg(w, h, label, body):
    return (f'<figure class="fig"><svg viewBox="0 0 {w} {h}" role="img" aria-label="{label}">' + body + '</svg>')
def t(x, y, s, cls='t-note', anchor='start', size=None):
    st = f' style="font-size:{size}px"' if size else ''
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{st}>{s}</text>'
def box(x, y, w, h, cls='box', r=8):
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" class="{cls}"/>'
def cap(c): return f'<figcaption>{c}</figcaption></figure>'

def timeline():
    gens = [('Tesla', 2006, 'GeForce 8'), ('Fermi', 2010, 'GTX 400'), ('Kepler', 2012, 'GTX 600, K80'), ('Maxwell', 2014, 'GTX 900'),
            ('Pascal', 2016, 'GTX 10, P100'), ('Volta', 2017, 'V100'), ('Turing', 2018, 'RTX 20, T4'), ('Ampere', 2020, 'RTX 30, A100'),
            ('Ada Lovelace', 2022, 'RTX 40, L4'), ('Hopper', 2022, 'H100, H200'), ('Blackwell', 2024, 'RTX 50, B200'), ('Rubin', 2026, 'Vera Rubin')]
    W, rowh = 760, 34
    b = t(20, 24, 'ARCHITECTURE', 't-label') + t(190, 24, 'FIRST YEAR', 't-label') + t(300, 24, 'WELL-KNOWN PRODUCTS', 't-label') + t(530, 24, 'WHAT WAS NEW', 't-label')
    new = ['CUDA: GPUs run general code', 'Memory that corrects errors', 'Lower power', 'Much lower power', 'Big jump for deep learning',
           'Tensor Cores (AI maths)', 'RT Cores (ray tracing), DLSS', 'Faster Tensor Cores, sparsity', 'Frame generation, AV1',
           'Data center only, 8-bit floats', '4-bit floats, more memory', 'HBM4 memory, Vera CPU']
    for i, (n, y, prod) in enumerate(gens):
        yy = 40 + i * rowh
        cls = 'box-1' if n in ('Ampere', 'Ada Lovelace', 'Blackwell') else ('box-2' if n in ('Hopper', 'Rubin') else 'box')
        b += box(14, yy, W - 28, rowh - 6, cls, 6)
        b += t(24, yy + 19, n, 't-title') + t(190, yy + 19, str(y), 't-val') + t(300, yy + 19, prod) + t(530, yy + 19, new[i], 't-tick')
    return svg(W, 40 + len(gens) * rowh + 6, 'A table of NVIDIA GPU architectures from Tesla in 2006 to Rubin in 2026, with the first year, well-known products and what was new in each.', b) + \
        cap('Every NVIDIA architecture since 2006. Blue rows are the three you meet most in shops and clouds today. Orange rows are made for data centers only.')

def one_to_many():
    W, H = 760, 330
    b = box(290, 20, 180, 56, 'box-1') + t(380, 45, 'Blackwell', 't-title', 'middle') + t(380, 64, 'one chip design', 't-tick', 'middle')
    cols = [('GeForce RTX 50', 'games and home AI', 'RTX 5090 ... 5050'), ('RTX PRO', 'design, film, AI', 'RTX PRO 6000 ... 2000'),
            ('Data center', 'train and serve AI', 'B200, B300, GB200, GB300'), ('DGX Spark', 'a desk AI computer', 'GB10 chip, 64 or 128 GB'), ('Jetson', 'robots and cameras', 'Jetson AGX Thor')]
    w = 138
    for i, (a, bb, c) in enumerate(cols):
        x = 14 + i * (w + 10.5)
        b += f'<line class="edge" x1="380" y1="76" x2="{x + w / 2}" y2="150"/>'
        b += box(x, 150, w, 150, 'box')
        b += t(x + w / 2, 176, a, 't-title', 'middle')
        b += t(x + w / 2, 200, bb, 't-tick', 'middle')
        for k, part in enumerate(c.replace(' ... ', ' to ').split(', ')):
            b += t(x + w / 2, 240 + k * 18, part, 't-note', 'middle', 12.5)
    return svg(W, H, 'One architecture, Blackwell, is sold as five product families: GeForce RTX 50, RTX PRO, data center GPUs, DGX Spark and Jetson.', b) + \
        cap('One architecture, many products. The same Blackwell design is cut to different sizes and sold to gamers, professionals, data centers and robot builders.')

def name_anatomy():
    W, H = 760, 300
    b = ''
    parts = [('GeForce', 'the family:\nfor games', 60), ('RTX', 'has ray tracing\nand AI cores', 215), ('50', 'the generation:\nBlackwell', 340), ('70', 'the tier:\nhigher is faster', 455), ('Ti', 'a small\nstep up', 575)]
    for name, note, x in parts:
        b += box(x, 30, 110, 54, 'box-1') + t(x + 55, 64, name, 't-title', 'middle', 20)
        for k, line in enumerate(note.split('\n')):
            b += t(x + 55, 108 + k * 17, line, 't-tick', 'middle')
    b += '<line class="axis" x1="20" y1="158" x2="740" y2="158"/>'
    b += t(20, 186, 'Generation number', 't-label') + t(20, 268, 'Tier number', 't-label')
    for i, (g, a) in enumerate([('20', 'Turing'), ('30', 'Ampere'), ('40', 'Ada Lovelace'), ('50', 'Blackwell')]):
        x = 200 + i * 138
        b += box(x, 168, 128, 46, 'box') + t(x + 64, 187, 'RTX ' + g, 't-title', 'middle') + t(x + 64, 205, a, 't-tick', 'middle')
    for i, (g, a) in enumerate([('50', 'entry'), ('60', 'mainstream'), ('70', 'upper middle'), ('80', 'high end'), ('90', 'the largest')]):
        x = 200 + i * 110
        b += box(x, 232, 100, 46, 'box-3' if g == '90' else 'box') + t(x + 50, 251, g, 't-title', 'middle') + t(x + 50, 269, a, 't-tick', 'middle')
    return svg(W, H, 'How to read the name GeForce RTX 5070 Ti: GeForce is the family, RTX means ray tracing and AI cores, 50 is the generation, 70 is the tier and Ti is a small step up.', b) + \
        cap('How to read a card name. The first digits give the generation, and so the architecture. The last two digits give the tier inside that generation.')

def where_map():
    W, H = 760, 318
    rows = [('In a laptop', 'GeForce RTX 50 Laptop', 'up to 24 GB', 'box'), ('In a home PC', 'GeForce RTX 5050 to 5090', '8 to 32 GB', 'box-1'),
            ('In a workstation', 'RTX PRO 2000 to 6000', '16 to 96 GB', 'box-1'), ('On a desk, as a box', 'DGX Spark, DGX Station', '64 GB shared and up', 'box-3'),
            ('In a server', 'L4, L40S, H100, H200, B200, B300', '24 to 288 GB each', 'box-2'), ('In a full rack', 'GB300 NVL72, Vera Rubin NVL72', '72 GPUs together', 'box-2'),
            ('In a robot or car', 'Jetson Orin, Jetson Thor, DRIVE', '8 to 128 GB shared', 'box-3')]
    b = t(24, 24, 'WHERE IT LIVES', 't-label') + t(250, 24, 'PRODUCTS', 't-label') + t(570, 24, 'MEMORY FOR THE GPU', 't-label')
    for i, (a, p, m, cls) in enumerate(rows):
        y = 36 + i * 40
        b += box(14, y, W - 28, 33, cls, 6) + t(24, y + 21, a, 't-title') + t(250, y + 21, p) + t(570, y + 21, m, 't-val')
    return svg(W, H, 'A map of NVIDIA products by where they live: laptop, home PC, workstation, desk box, server, full rack, and robot or car.', b) + \
        cap('The whole product range by where the GPU lives. Memory grows as you go down from a laptop to a server.')

def hbar(title, items, unit, label, caption, fmt, note=None):
    W = 760; rowh = 30; x0 = 250; x1 = 640
    lo = min(v for _, v, _ in items); hi = max(v for _, v, _ in items)
    l0, l1 = math.log10(lo * 0.6), math.log10(hi)
    b = t(14, 22, title, 't-title')
    for i, (n, v, cls) in enumerate(items):
        y = 38 + i * rowh
        w = (math.log10(v) - l0) / (l1 - l0) * (x1 - x0)
        b += t(x0 - 10, y + 16, n, 't-note', 'end') + f'<rect x="{x0}" y="{y + 3}" width="{w:.1f}" height="{rowh - 12}" rx="3" class="{cls}"/>' + t(x0 + w + 8, y + 16, fmt(v), 't-val')
    h = 44 + len(items) * rowh
    if note: b += t(14, h + 8, note, 't-tick'); h += 18
    return svg(W, h, label, b) + cap(caption)

def price_ladder():
    g = P['buy_ladder']
    items = [(r['name'], r['usd'], {'edge': 's3', 'home': 's1', 'pro': 's2', 'dc': 's2'}[r['kind']]) for r in g]
    return hbar('Price to buy one, US dollars, November 2026 (log scale)', items, '$',
        'A bar chart on a log scale of the street price to buy one NVIDIA product in November 2026, from a Jetson Orin Nano Super kit at 399 dollars to an H100 at about 30,000 dollars.',
        'The street price ladder in November 2026: what one product costs in a US shop today, not its list price. The scale is logarithmic: each step to the right is a multiplication, not an addition.',
        lambda v: f'${v:,.0f}', 'Blue: GeForce. Orange: workstation and data center. Green: small computers.')

def rent_ladder():
    items = [(r['name'], r['usd'], 's2') for r in P['rent_ladder']]
    return hbar('Price to rent one GPU for one hour, US dollars, late 2026 (log scale)', items, '$',
        'A bar chart on a log scale of cloud rental prices per GPU-hour, from a T4 at about 0.55 dollars to a B300 at about 8 dollars.',
        'Renting one GPU for one hour in a cloud. Old architectures stay cheap; each new one costs more per hour and does much more work in that hour.',
        lambda v: f'${v:.2f}')

def vram():
    items = [('RTX 5060 (8 GB)', 8, 's1'), ('RTX 5070 (12 GB)', 12, 's1'), ('RTX 5080 (16 GB)', 16, 's1'), ('L4 (24 GB)', 24, 's2'), ('RTX 5090 (32 GB)', 32, 's1'),
             ('L40S (48 GB)', 48, 's2'), ('H100 (80 GB)', 80, 's2'), ('RTX PRO 6000 (96 GB)', 96, 's2'), ('H200 (141 GB)', 141, 's2'), ('B300 (288 GB)', 288, 's2')]
    return hbar('Memory on one GPU, gigabytes (log scale)', items, 'GB',
        'A bar chart of the memory on one GPU, from 8 gigabytes on an RTX 5060 to 288 gigabytes on a B300.',
        'Memory on one GPU. For AI work this number matters more than any other, because a model must fit in it.', lambda v: f'{v} GB')

def bandwidth():
    items = [('DGX Spark (LPDDR5X)', 273, 's3'), ('RTX 5060 Ti (GDDR7)', 448, 's1'), ('RTX 5070 (GDDR7)', 672, 's1'), ('L40S (GDDR6)', 864, 's2'),
             ('RTX 5070 Ti (GDDR7)', 896, 's1'), ('RTX 5080 (GDDR7)', 960, 's1'), ('RTX 5090 (GDDR7)', 1792, 's1'), ('A100 80 GB (HBM2e)', 2000, 's2'),
             ('H100 (HBM3)', 3350, 's2'), ('H200 (HBM3e)', 4800, 's2'), ('B200 (HBM3e)', 7700, 's2')]
    return hbar('Memory bandwidth of one GPU, gigabytes per second (log scale)', items, 'GB/s',
        'A bar chart of memory bandwidth, from 273 gigabytes per second on a DGX Spark to about 7,700 on a B200.',
        'Memory bandwidth, with the memory type in brackets. Every card with HBM is above every card with GDDR. Data center values are for the server versions; the B200 is given as 7,700 or 8,000 GB/s, depending on the source.',
        lambda v: f'{v:,} GB/s', 'Blue: GeForce. Orange: workstation and data center. Green: small computers.')

FIGS = dict(bandwidth=bandwidth, timeline=timeline, one_to_many=one_to_many, name=name_anatomy, where=where_map, price=price_ladder, rent=rent_ladder, vram=vram)
