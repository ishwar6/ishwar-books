"""Cut highlighted excerpts out of research papers (arXiv PDFs). Shared by the shots_partN.py job lists.

For each job: download the PDF, find the given phrases on the page, highlight them in yellow,
and render a crop around them (a paragraph, an equation, a figure or a table) as a PNG.
Usage: from paper_shots import run; run(JOBS)   ->  public/img/agents/<name>.png"""
import os, sys, urllib.request
import re, unicodedata
import pymupdf as fitz

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'public', 'img', 'agents')
VIT = '2210.03629'   # default paper: ReAct; pass arxiv_id= for others
CACHE = os.path.expanduser('~/.cache/papers')
YELLOW = (1.0, 0.86, 0.2)


def pdf(arxiv_id):
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, arxiv_id.replace('/', '_') + '.pdf')
    if not os.path.exists(path):
        req = urllib.request.Request(f'https://arxiv.org/pdf/{arxiv_id}', headers={'User-Agent': 'Mozilla/5.0'})
        open(path, 'wb').write(urllib.request.urlopen(req).read())
    return fitz.open(path)


def norm(w):
    w = unicodedata.normalize('NFKC', w).lower()
    return re.sub(r'[^0-9a-z%]+', '', w)


def find(pg, phrase, nth=0):
    """Find a phrase in the page's word list, across line breaks. Returns one rectangle per line it covers."""
    words = [(*w[:4], part, *w[5:]) for w in pg.get_text('words')     # x0, y0, x1, y1, word, block, line, word_no
             for part in re.split(r'[-\u2013\u2014/]', w[4]) if part]  # split "encoder-decoder" into two words
    toks = [norm(w[4]) for w in words]
    want = [t for t in (norm(x) for x in re.split(r'[\s\u2013\u2014/-]+', phrase)) if t]
    for i in range(len(toks)):
        j, k, used = i, 0, []
        while j < len(toks) and k < len(want):
            if not toks[j]:
                j += 1; continue
            if toks[j] == want[k]:
                used.append(j); j += 1; k += 1
            elif want[k].startswith(toks[j]) and j + 1 < len(toks) and want[k] == toks[j] + toks[j + 1]:
                used += [j, j + 1]; j += 2; k += 1     # a word hyphenated across lines
            else:
                break
        if k == len(want) and nth > 0:
            nth -= 1
            continue
        if k == len(want):
            lines = {}
            for u in used:
                key = (words[u][5], words[u][6])
                r = fitz.Rect(words[u][:4])
                lines[key] = lines[key] | r if key in lines else r
            return list(lines.values())
    return []


def shot(name, arxiv_id=VIT, highlight=(), anchor=None, page=None, above=40, below=40, dpi=190, snap=True, band=None,
         figure=False, column=None, nth=0, hl_nth=None, fig_top=None, end=None):
    """Render one highlighted excerpt.
    anchor: phrase that positions the crop (defaults to the first highlight); nth picks a later match on the page.
    above/below: points of context kept around the anchor (text mode).
    figure=True: the anchor is a caption; the crop is the figure graphics above it plus the whole caption.
    column: None = automatic (one column of a two-column paper if the anchor sits in one), or 'full'.
    band: (dy0, dy1) points below the anchor's bottom to mark as a highlighted band (equations, table rows).
    end: a phrase where the excerpt stops (the crop ends at the line holding its last word; `below` is then ignored)."""
    doc = pdf(arxiv_id)
    anchor = anchor or highlight[0]
    pages = [page] if page is not None else range(len(doc))
    for pno in pages:
        pg = doc[pno]
        hits = find(pg, anchor, nth)
        if hits:
            break
    else:
        raise SystemExit(f'{name}: anchor not found: {anchor!r}')
    box = fitz.Rect(hits[0])
    for h in hits[1:]:
        box |= h
    W, Hh = pg.rect.width, pg.rect.height
    blocks = [fitz.Rect(b[:4]) for b in pg.get_text('blocks')]
    blk = next((b for b in blocks if b.intersects(fitz.Rect(hits[0]))), box)
    left = [b for b in blocks if b.x1 < 0.53 * W and b.width > 0.3 * W]
    right = [b for b in blocks if b.x0 > 0.47 * W and b.width > 0.3 * W]
    two_col = len(left) >= 2 and len(right) >= 2
    one_col = column != 'full' and two_col and blk.width < 0.6 * W
    if one_col:
        x0, x1 = blk.x0 - 10, blk.x1 + 10
    else:                                             # the page's text width, without the empty side margins
        ws = [w for w in pg.get_text('words') if (w[3] - w[1]) < 3 * (w[2] - w[0])]   # skip arXiv's rotated side stamp
        x0, x1 = max(20, min(w[0] for w in ws) - 10), min(W - 20, max(w[2] for w in ws) + 10)

    found = []
    for phrase in highlight:
        n = (hl_nth or {}).get(phrase, 0)
        rects = find(pg, phrase, n)
        if not rects:
            raise SystemExit(f'{name}: highlight not found on page {pno + 1}: {phrase!r}')
        for r in rects:
            pg.add_highlight_annot(r).set_colors(stroke=YELLOW); found.append(r)
            pg.annots
    if band:
        y0b, y1b = box.y1 + band[0], box.y1 + band[1]
        xs = [w for w in pg.get_text('words') if w[2] > x0 and w[0] < x1 and w[3] > y0b and w[1] < y1b]
        xs = xs or [w for w in pg.get_text('words') if w[2] > x0 and w[0] < x1]
        bx0, bx1 = min(w[0] for w in xs) - 8, max(w[2] for w in xs) + 8
        r = fitz.Rect(bx0, box.y1 + band[0], bx1, box.y1 + band[1])
        pg.draw_rect(r, color=(0.95, 0.7, 0.0), fill=YELLOW, fill_opacity=0.28, width=1.2, overlay=False)

    if figure:
        cap = blk
        art = [fitz.Rect(d['rect']) for d in pg.get_drawings()] + [fitz.Rect(i['bbox']) for i in pg.get_image_info()]
        art = sorted((r for r in art if r.y1 <= cap.y0 + 3 and r.y0 > 60 and r.width > 2 and r.x1 > x0 and r.x0 < x1),
                     key=lambda r: -r.y1)
        top = cap.y0
        for r in (art if fig_top is None else []):
            if r.y1 >= top - 40:
                top = min(top, r.y0)
        # figure labels are text: include text blocks that sit inside the figure area
        for b in (blocks if fig_top is None else []):
            if b.y0 >= top - 2 and b.y1 <= cap.y0 + 1 and b.x1 > x0 and b.x0 < x1:
                top = min(top, b.y0)
        if fig_top is not None:
            top = cap.y0 - fig_top
        changed = fig_top is None
        while changed:                                # short text lines just above the art are figure titles
            changed = False
            for b in blocks:
                if b.y0 < top and b.y1 >= top - 14 and b.height < 30 and b.y1 <= cap.y0:
                    top, changed = b.y0, True
        clip = fitz.Rect(x0, max(0, top - 6), x1, min(Hh, cap.y1 + 4 + below))
    else:
        top, bot = max(0, box.y0 - above), min(Hh, box.y1 + below)
        if end:
            er = find(pg, end)
            if not er:
                raise SystemExit(f'{name}: end phrase not found: {end!r}')
            bot = max(r.y1 for r in er) + 1
        if snap:
            lines = sorted({(round(r[1], 1), round(r[3], 1)) for r in pg.get_text('words') if r[2] > x0 and r[0] < x1})
            tops = [a for a, b in lines if a >= top - 0.5]
            bots = [b for a, b in lines if b <= bot + 0.5]
            if tops:                                   # pad to halfway between lines, so no neighbour line peeks in
                t0 = min(tops)
                prev = [b for a, b in lines if b <= t0 + 0.5 and a < t0 - 0.5]
                top = max(t0 - 4, (max(prev) + t0) / 2 if prev else t0 - 4)
            if bots:
                b0 = max(bots)
                nxt = [a for a, b in lines if a >= b0 - 0.5 and b > b0 + 0.5]
                bot = min(b0 + 4, (min(nxt) + b0) / 2 if nxt else b0 + 4)
        clip = fitz.Rect(x0, top, x1, bot)
    os.makedirs(OUT, exist_ok=True)
    pix = pg.get_pixmap(clip=clip, dpi=dpi, annots=True)
    pix.save(os.path.join(OUT, name + '.png'))
    print(f'{name}: page {pno + 1}, {pix.width}x{pix.height}, highlights: {len(found)}')



def run(jobs):
    """Render every job, or only the names listed in the ONLY environment variable (comma separated)."""
    only = os.environ.get('ONLY')
    for job in jobs:
        if not only or job['name'] in only.split(','):
            shot(**job)
