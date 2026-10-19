"""Build the files of the side-by-side reader (/papers/vllm/read): our own annotated copy of the paper, and a map
that ties each "From the paper" box of the articles to its place in the PDF.

For each box of the numbered parts (in the order of the articles), the box's screenshot names are looked up in the
shots_partN.py job lists, and paper_shots.shot(dry=True) gives the page, the crop and the highlighted phrases.
Many boxes show two or three screenshots (a passage that runs over a page or column break, or a passage and its
footnote): each screenshot becomes one region of the box. The box's id is its first screenshot, which is how the
reader page finds the box in the article.
The PDF then gets, drawn into the page itself:
  - the same yellow highlights as the screenshots,
  - a number in the margin and a thin bar along the excerpt, so the reader can match the paper to the box.
Writes public/papers/pdf/vllm.pdf (images down-sampled, so it loads fast) and public/papers/pdf/vllm.json.
Usage: python reader.py"""
import glob, importlib, json, os, re
import pymupdf as fitz
from paper_shots import SAM as PAPER, pdf, shot

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
OUT = os.path.join(ROOT, 'public', 'papers', 'pdf')
YELLOW = (1.0, 0.86, 0.2)
TEAL = (0.2, 0.6, 0.53)
STITCHED = {}   # one screenshot made of several crops: {'name': ['name-a', 'name-b']}


def jobs():
    out = {}
    for f in sorted(glob.glob(os.path.join(HERE, 'shots_part*.py'))):
        for j in importlib.import_module(os.path.basename(f)[:-3]).JOBS:
            out[j['name']] = j
    return out


def boxes_in_order():
    """(part slug, [screenshot names]) for every box of this paper, in the order the articles show them."""
    parts = []
    for f in glob.glob(os.path.join(ROOT, 'content', 'papers', 'vllm', 'part-*.md')):
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
                m = re.search(r'/img/papers/vllm/([\w-]+)\.png', line)   # a linked image names its file twice
                if m and m.group(1) not in out[-1][1]:
                    out[-1][1].append(m.group(1))
    for slug, names in out:
        if not names:
            print(f'{slug}: a box of this paper has no screenshot')
    return [(slug, names) for slug, names in out if names]


def mark(pg, clip, n):
    """A thin bar along the excerpt and a numbered dot, in the margin nearest to the excerpt."""
    W = pg.rect.width
    left = (clip.x0 + clip.x1) / 2 < W / 2
    x = clip.x0 - 9 if left else clip.x1 + 9
    x = min(max(x, 12), W - 12)
    sh = pg.new_shape()
    sh.draw_line((x, clip.y0 + 2), (x, clip.y1 - 2))
    sh.finish(color=TEAL, width=1.6, lineCap=1)
    cx = min(max(x - 9 if left else x + 9, 9), W - 9)   # keep the dot inside the page (the title spans the full width)
    cy = clip.y0 + 7
    sh.draw_circle((cx, cy), 7.5)
    sh.finish(color=TEAL, fill=TEAL)
    sh.commit()
    s = str(n)
    fs = 8 if len(s) < 3 else 6.5
    w = fitz.get_text_length(s, fontname='hebo', fontsize=fs)
    pg.insert_text((cx - w / 2, cy + fs * 0.36), s, fontname='hebo', fontsize=fs, color=(1, 1, 1))


def main():
    J = jobs()
    doc = pdf(PAPER)
    boxes, missing = [], []
    for slug, names in boxes_in_order():
        name, regions = names[0], []
        for part in (p for n in names for p in STITCHED.get(n, [n])):
            j = J.get(part)
            if not j or j.get('arxiv_id', PAPER) != PAPER:
                missing += [] if j else [part]
                continue
            r = shot(**j, dry=True)
            regions.append({'page': r['page'], 'clip': r['clip'], 'hl': r['hl']})
        if not regions:
            missing += [] if name in missing else [name]
            continue
        boxes.append({'n': len(boxes) + 1, 'id': name, 'part': slug, 'regions': regions})

    for b in boxes:
        for r in b['regions']:
            pg = doc[r['page']]
            for h in r['hl']:
                pg.draw_rect(fitz.Rect(h), color=None, fill=YELLOW, fill_opacity=0.38, overlay=False)
            mark(pg, fitz.Rect(r['clip']), b['n'])

    os.makedirs(OUT, exist_ok=True)
    doc.rewrite_images(dpi_threshold=160, dpi_target=130, quality=72)   # the paper's photos are huge; the text stays vector
    dest = os.path.join(OUT, 'vllm.pdf')
    doc.save(dest, garbage=4, deflate=True, clean=True)
    meta = {'pdf': '/papers/pdf/vllm.pdf', 'arxiv': PAPER, 'pages': [[round(p.rect.width, 2), round(p.rect.height, 2)] for p in doc],
            'boxes': boxes}
    json.dump(meta, open(os.path.join(OUT, 'vllm.json'), 'w'), separators=(',', ':'))
    print(f'{len(boxes)} boxes on {len({r["page"] for b in boxes for r in b["regions"]})} pages, '
          f'{os.path.getsize(dest) / 1e6:.1f} MB; not found in the shot lists: {missing or "none"}')


if __name__ == '__main__':
    main()
