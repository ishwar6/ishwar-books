"""Build the files of the side-by-side reader (/papers/vit/read): our own annotated copy of the paper, and a map
that ties each "From the paper" box of the articles to its place in the PDF.

For each box of the numbered parts (in the order of the articles), the box's screenshot name is looked up in the
shots_partN.py job lists, and paper_shots.shot(dry=True) gives the page, the crop and the highlighted phrases.
The PDF then gets, drawn into the page itself:
  - the same yellow highlights as the screenshots,
  - a number in the margin and a thin bar along the excerpt, so the reader can match the paper to the box.
A box with two screenshots (a passage that runs across two pages or columns) gets one region per screenshot.
Writes public/papers/pdf/vit.pdf (images down-sampled, so it loads fast) and public/papers/pdf/vit.json.
Usage: python reader.py"""
import glob, importlib, json, os, re
import pymupdf as fitz
from paper_shots import VIT, pdf, shot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
OUT = os.path.join(ROOT, 'public', 'papers', 'pdf')
YELLOW = (1.0, 0.86, 0.2)
TEAL = (0.2, 0.6, 0.53)
STITCHED = {}   # one screenshot made of two crops (ViT has none; two-screenshot boxes are handled below)


def jobs():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'shots_part*.py'))):
        for j in importlib.import_module(os.path.basename(f)[:-3]).JOBS:
            out[j['name']] = j
    return out


def boxes_in_order():
    """(part slug, [screenshot names]) for every box of this paper, in the order the articles show them."""
    parts = []
    for f in glob.glob(os.path.join(ROOT, 'content', 'papers', 'vit', 'part-*.md')):
        n = int(re.search(r'^part:\s*(\d+)', open(f).read(), re.M).group(1))
        parts.append((n, f))
    out = []
    for _, f in sorted(parts):
        slug = os.path.basename(f)[:-3]
        in_box = False
        for line in open(f):
            if line.startswith('> [!PAPER]'):
                in_box = line.startswith('> [!PAPER] ·')        # an unnamed box shows this paper
                if in_box:
                    out.append((slug, []))
            elif not line.startswith('>'):
                in_box = False
            elif in_box:
                m = re.search(r'/img/papers/vit/([\w-]+)\.png', line)
                if m and line.startswith('> [!['):
                    out[-1][1].append(m.group(1))
    return out


def text_width(clip, j):
    """Page-1 crops with column='full' are cut back to the text column in shots_part1.trim_stamp (the rotated arXiv
    stamp fools the margin logic); do the same here, so the marker sits beside the text and not on the stamp."""
    if j.get('column') != 'full' or j.get('page') != 0:
        return clip
    pg = pdf(VIT)[0]
    W = pg.rect.width
    ws = [w for w in pg.get_text('words') if (w[3] - w[1]) < 3 * (w[2] - w[0])]
    body = [w for w in ws if w[0] > 60 and w[2] < W - 40]
    x0, x1 = min(w[0] for w in body) - 10, max(w[2] for w in body) + 10
    return [max(clip[0], round(x0, 2)), clip[1], min(clip[2], round(x1, 2)), clip[3]]


USED = {}   # page number -> centres of the dots already drawn


def mark(pg, clip, n):
    """A thin bar along the excerpt and a numbered dot, in the margin nearest to the excerpt.
    Two boxes can show the same excerpt (Figure 1 is in Part 1 and in Part 2): the later dot moves down."""
    W = pg.rect.width
    left = (clip.x0 + clip.x1) / 2 < W / 2
    x = clip.x0 - 9 if left else clip.x1 + 9
    x = min(max(x, 12), W - 12)
    sh = pg.new_shape()
    sh.draw_line((x, clip.y0 + 2), (x, clip.y1 - 2))
    sh.finish(color=TEAL, width=1.6, lineCap=1)
    cx = x - 9 if left else x + 9
    cy = clip.y0 + 7
    while any(abs(cx - x2) < 16 and abs(cy - y2) < 16 for x2, y2 in USED.get(pg.number, [])):
        cy += 17
    USED.setdefault(pg.number, []).append((cx, cy))
    sh.draw_circle((cx, cy), 7.5)
    sh.finish(color=TEAL, fill=TEAL)
    sh.commit()
    s = str(n)
    fs = 8 if len(s) < 3 else 6.5
    w = fitz.get_text_length(s, fontname='hebo', fontsize=fs)
    pg.insert_text((cx - w / 2, cy + fs * 0.36), s, fontname='hebo', fontsize=fs, color=(1, 1, 1))


def main():
    J = jobs()
    doc = pdf(VIT)
    boxes, missing = [], []
    for slug, names in boxes_in_order():
        regions = []
        for name in names:
            for part in STITCHED.get(name, [name]):
                j = J.get(part)
                if not j or j.get('arxiv_id', VIT) != VIT:
                    continue
                r = shot(**j, dry=True)
                regions.append({'page': r['page'], 'clip': text_width(r['clip'], j), 'hl': r['hl']})
        if not regions:
            missing.append(names[0] if names else slug)
            continue
        boxes.append({'n': len(boxes) + 1, 'id': names[0], 'part': slug, 'regions': regions})

    for b in boxes:
        for r in b['regions']:
            pg = doc[r['page']]
            for h in r['hl']:
                pg.draw_rect(fitz.Rect(h), color=None, fill=YELLOW, fill_opacity=0.38, overlay=False)
            mark(pg, fitz.Rect(r['clip']), b['n'])

    os.makedirs(OUT, exist_ok=True)
    doc.rewrite_images(dpi_threshold=160, dpi_target=130, quality=72)   # shrink the paper's photos; the text stays vector
    dest = os.path.join(OUT, 'vit.pdf')
    doc.save(dest, garbage=4, deflate=True, clean=True)
    meta = {'pdf': '/papers/pdf/vit.pdf', 'arxiv': VIT, 'pages': [[round(p.rect.width, 2), round(p.rect.height, 2)] for p in doc],
            'boxes': boxes}
    json.dump(meta, open(os.path.join(OUT, 'vit.json'), 'w'), separators=(',', ':'))
    print(f'{len(boxes)} boxes on {len({r["page"] for b in boxes for r in b["regions"]})} pages, '
          f'{os.path.getsize(dest) / 1e6:.1f} MB; not found in the shot lists: {missing or "none"}')


if __name__ == '__main__':
    main()
