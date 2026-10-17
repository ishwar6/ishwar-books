"""Build the files of the side-by-side reader (/papers/transformer-circuits/read): our own annotated copy of the
paper, and a map that ties each "From the paper" box of the articles to its place in the PDF.

The paper is a web page, not a PDF, so paper_pdf.mjs prints it to ~/.cache/papers/circuits-framework.pdf first
(run here if that file is missing, or with --fresh). While it prints, it tags each excerpt of shots_part*.json with
invisible links (https://reader.invalid/box/<name> over the excerpt, .../hl/<name> over each yellow phrase), so this
script reads the exact places from the PDF instead of guessing them.

For each unnamed box of the numbered parts (in the order of the articles), its screenshot name gives:
  - the excerpt's regions (one per page it covers), trimmed to the text and figures inside,
  - the yellow phrases (painted by paper_pdf.mjs, the same phrases as the screenshots).
Then each excerpt is checked by text: the words of its `find` and `upto` phrases must be inside its regions.
The PDF gets a number in the left margin and a thin bar along each excerpt, the tag links are removed, the images
are down-sampled, and it is written with the map to public/papers/pdf/transformer-circuits.pdf and .json
(the same format as sam.json).
Usage: python reader.py [--fresh]"""
import glob, json, os, re, subprocess, sys, unicodedata
from collections import defaultdict
import pymupdf as fitz

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
OUT = os.path.join(ROOT, 'public', 'papers', 'pdf')
SLUG = 'transformer-circuits'
URL = 'https://transformer-circuits.pub/2021/framework/index.html'
RAW = os.path.expanduser('~/.cache/papers/circuits-framework.pdf')
TAG = 'https://reader.invalid/'
TEAL = (0.2, 0.6, 0.53)


def shots():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'shots_part*.json'))):
        for s in json.load(open(f)):
            out[s['out'][:-4]] = s
    return out


def boxes_in_order():
    """(part slug, screenshot name) for every box of this paper, in the order the articles show them."""
    parts = []
    for f in glob.glob(os.path.join(ROOT, 'content', 'papers', SLUG, 'part-*.md')):
        m = re.search(r'^part:\s*(\d+)', open(f).read(), re.M)
        if m:
            parts.append((int(m.group(1)), f))
    out = []
    for _, f in sorted(parts):
        slug = os.path.basename(f)[:-3]
        in_box = False
        for line in open(f):
            if line.startswith('> [!PAPER]'):
                in_box = line.startswith('> [!PAPER] ·')        # an unnamed box shows this paper
            elif not line.startswith('>'):
                in_box = False
            elif in_box:
                m = re.search(r'/img/papers/circuits/([\w-]+)\.png', line)
                if m:
                    out.append((slug, m.group(1)))
                    in_box = False
    return out


def norm(t):
    return [w for w in (re.sub(r'[^0-9a-z]+', '', x) for x in unicodedata.normalize('NFKC', t).lower().split()) if w]


def has_words(text, phrase):
    """True if the words of `phrase` (maths dropped) appear in order in `text`."""
    got = ' '.join(norm(text))
    runs, run = [], []
    for w in phrase.split():
        if re.search(r'[_^\\]', w):          # a TeX symbol such as W_O: printed as maths, not as these letters
            runs.append(run); run = []
        else:
            run.append(w)
    runs.append(run)
    return all(' '.join(norm(' '.join(r))) in got for r in runs if r)


def tighten(pg, r):
    """Trim a rectangle to the text, pictures and drawings whose centre lies inside it."""
    stuff = [fitz.Rect(b[:4]) for b in pg.get_text('words')] + [fitz.Rect(i['bbox']) for i in pg.get_image_info()]
    stuff += [d['rect'] for d in pg.get_drawings() if d['rect'].width < 0.95 * pg.rect.width]
    inside = [s for s in stuff if r.contains(fitz.Point((s.x0 + s.x1) / 2, (s.y0 + s.y1) / 2)) and not s.is_empty]
    if not inside:
        return r
    t = fitz.Rect(inside[0])
    for s in inside[1:]:
        t |= s
    return (t + (-4, -3, 4, 3)) & r & pg.rect


def mark(pg, clip, n):
    """A thin bar along the excerpt and a numbered dot, in the left margin (the paper has one column)."""
    x = max(clip.x0 - 7, 20)
    sh = pg.new_shape()
    sh.draw_line((x, clip.y0 + 2), (x, clip.y1 - 2))
    sh.finish(color=TEAL, width=1.6, lineCap=1)
    cx, cy = x - 9, clip.y0 + 7
    sh.draw_circle((cx, cy), 7.5)
    sh.finish(color=TEAL, fill=TEAL)
    sh.commit()
    s = str(n)
    fs = 8 if len(s) < 3 else 6.5
    w = fitz.get_text_length(s, fontname='hebo', fontsize=fs)
    pg.insert_text((cx - w / 2, cy + fs * 0.36), s, fontname='hebo', fontsize=fs, color=(1, 1, 1))


def main():
    if '--fresh' in sys.argv or not os.path.exists(RAW):
        subprocess.run(['node', os.path.join(HERE, 'paper_pdf.mjs'), RAW], check=True)
    S = shots()
    doc = fitz.open(RAW)

    # the tag links: (kind, name) -> block k -> [(page, Rect)]
    tags = defaultdict(lambda: defaultdict(list))
    for pno, pg in enumerate(doc):
        for ln in pg.get_links():
            u = ln.get('uri') or ''
            if u.startswith(TAG):
                kind, name, *k = u[len(TAG):].split('/')
                tags[(kind, name)][int(k[0]) if k else 0].append((pno, fitz.Rect(ln['from'])))

    def spans(name):
        """page -> the rectangle the excerpt covers on that page, from the line where it starts to where it ends."""
        over = [x for v in tags[('box', name)].values() for x in v]
        x0 = min((r.x0 for _, r in over), default=40)
        x1 = max((r.x1 for _, r in over), default=doc[0].rect.width - 40)
        st, en = tags[('start', name)].get(0), tags[('end', name)].get(0)
        out = {}
        if st and en:
            (ps, rs), (pe, re_) = st[0], en[-1]
            for pno in range(ps, pe + 1):
                out[pno] = fitz.Rect(x0, rs.y0 - 3 if pno == ps else 0, x1, re_.y1 + 4 if pno == pe else doc[pno].rect.height)
        else:                                               # no text tags (a display formula): the block links
            for pno, r in over:
                out[pno] = out[pno] | r if pno in out else r
        return out

    boxes, missing, problems = [], [], []
    for slug, name in boxes_in_order():
        pages = spans(name)
        if not pages:
            missing.append(name)
            continue
        hls = [x for v in tags[('hl', name)].values() for x in v]
        regions, text = [], ''
        for pno in sorted(pages):
            pg = doc[pno]
            body = fitz.Rect(30, 30, pg.rect.width - 30, pg.rect.height - 34)
            r = tighten(pg, pages[pno] & body)
            if r.is_empty or r.height < 6:
                continue
            hl = [h for p, h in hls if p == pno and h.intersects(r)]
            regions.append({'page': pno, 'clip': [round(v, 2) for v in r], 'hl': [[round(v, 2) for v in h] for h in hl]})
            text += pg.get_text(clip=r) + ' '
        if not regions:
            missing.append(name)
            continue
        s = S.get(name, {})
        for key in ('find', 'upto'):
            if s.get(key) and not has_words(text, s[key]):
                problems.append(f'{name}: {key} phrase not in its region: {s[key]!r}')
        nh = sum(len(r['hl']) for r in regions)
        if len(s.get('mark', [])) > nh:
            problems.append(f'{name}: {len(s["mark"])} phrases to highlight, {nh} highlight rectangles')
        boxes.append({'n': len(boxes) + 1, 'id': name, 'part': slug, 'regions': regions})

    for pg in doc:                                          # the tags have done their job
        for ln in pg.get_links():
            if (ln.get('uri') or '').startswith(TAG):
                pg.delete_link(ln)
    for b in boxes:
        for r in b['regions']:
            mark(doc[r['page']], fitz.Rect(r['clip']), b['n'])

    doc.set_metadata({'title': 'A Mathematical Framework for Transformer Circuits (annotated copy)',
                      'author': 'Elhage et al., Anthropic (2021)',
                      'subject': f'Printed from {URL}; highlights and numbers added for ishwarj.com/papers/{SLUG}/',
                      'creator': 'ishwarj.com', 'producer': 'ishwarj.com'})
    os.makedirs(OUT, exist_ok=True)
    doc.rewrite_images(dpi_threshold=200, dpi_target=150, quality=80)   # the figures are large PNGs; text stays vector
    dest = os.path.join(OUT, f'{SLUG}.pdf')
    doc.save(dest, garbage=4, deflate=True, clean=True)
    meta = {'pdf': f'/papers/pdf/{SLUG}.pdf', 'url': URL,
            'pages': [[round(p.rect.width, 2), round(p.rect.height, 2)] for p in doc], 'boxes': boxes}
    json.dump(meta, open(os.path.join(OUT, f'{SLUG}.json'), 'w'), separators=(',', ':'))
    print(f'{len(boxes)} boxes on {len({r["page"] for b in boxes for r in b["regions"]})} pages, {len(doc)} pages, '
          f'{os.path.getsize(RAW) / 1e6:.1f} MB -> {os.path.getsize(dest) / 1e6:.1f} MB')
    print('not found in the PDF:', missing or 'none')
    print('checks:', problems or 'all excerpts hold their phrases')


if __name__ == '__main__':
    main()
