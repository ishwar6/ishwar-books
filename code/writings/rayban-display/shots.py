"""Highlighted excerpts and figure crops for "How Ray-Ban Display Glasses Put a Screen in a Lens".

1. The Nature 2025 paper on the wristband (Kaifosh, Reardon et al., open access, CC BY-NC-ND 4.0), cut with the shared
   tool in code/papers/bert/paper_shots.py. PDF: ~/.cache/papers/nature_emg2025.pdf, from the open-access copy at
   https://repository.cshl.edu/id/eprint/42049/
2. Drawings from two Lumus patents (US patents are in the public domain), cropped by page, trimmed, unchanged otherwise:
   US 6,829,095 B2 (Amitai, 2004) and US 10,551,544 B2 (Danziger and Sharlin, 2020). PDFs from patentimages.storage.googleapis.com,
   cached as ~/.cache/papers/US6829095.pdf and US10551544.pdf.
Run: python shots.py   (or ONLY=name1,name2 python shots.py)"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
# numbers.py in this folder shadows the standard-library module of the same name: keep this folder off the import path
sys.path = [x for x in sys.path if os.path.abspath(x or '.') != HERE]
from PIL import Image, ImageChops
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'papers', 'bert'))
import paper_shots
from paper_shots import shot, pdf
import pymupdf as fitz

OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'writings', 'rayban-display')
paper_shots.OUT = OUT
NAT = 'nature_emg2025'

JOBS = [
    dict(name='p-emg-abstract', arxiv_id=NAT, page=0, anchor='Here, we describe the development of a generic non-',
         highlight=['generic sEMG', 'decoding models that generalize across people',
                    'handwriting at 20.9 words per minute'],
         above=2, end='interface with performant out-of-the-box generalization across people.'),
    dict(name='p-emg-device', arxiv_id=NAT, page=1, anchor='Consenting participants (Methods) were seated in front of a computer',
         highlight=['multichannel recording device with a high sample rate', 'low-noise'],
         above=2, end='protocols and provided a'),
    dict(name='p-emg-fig1', arxiv_id=NAT, page=2, anchor='Fig. 1 | A hardware and software platform for high-throughput recording',
         figure=True, column='full', below=-150, highlight=[]),
    dict(name='p-emg-baseline', arxiv_id=NAT, page=5, anchor='For each of these interactions, we also provide performance metrics',
         highlight=['at 25.1 WPM, higher than the 20.9 WPM achieved with', 'the sEMG handwriting decoder'],
         above=2, end='of hand-encumbering devices or external instrumentation.'),
    dict(name='p-patent-ladder', arxiv_id='US6829095', page=29,
         highlight=['33%, 50% and 100%'], anchor='These reflecting', above=50, below=70),
]

# (name, pdf id, 0-based page, crop as fractions of the page: x0, y0, x1, y1, clockwise rotation in degrees)
PATENT_FIGS = [
    ('pat-fig2', 'US6829095', 2, (0.05, 0.10, 0.95, 0.97), 90),    # FIG. 2: substrate, TIR, array of partial mirrors
    ('pat-fig9', 'US6829095', 6, (0.05, 0.10, 0.95, 0.90), 0),     # FIG. 9: many exit beams reach one eye
    ('pat-fig13', 'US6829095', 8, (0.05, 0.495, 0.95, 0.97), 0),    # FIG. 13: mirrors along a second axis (2D expansion)
    ('pat-2d', 'US10551544', 3, (0.05, 0.10, 0.95, 0.425), 0),      # FIG. 1: two-dimensional aperture expansion
]


def trim(path, pad=14):
    im = Image.open(path).convert('RGB')
    bg = Image.new('RGB', im.size, (255, 255, 255))
    diff = ImageChops.difference(im, bg).convert('L').point(lambda v: 255 if v > 40 else 0)
    box = diff.getbbox()
    if box:
        x0, y0, x1, y1 = box
        im = im.crop((max(0, x0 - pad), max(0, y0 - pad), min(im.width, x1 + pad), min(im.height, y1 + pad)))
    im.save(path, optimize=True)
    return im.size


def patent_figs():
    only = os.environ.get('ONLY')
    for name, pid, pno, (a, b, c, d), rot in PATENT_FIGS:
        if only and name not in only.split(','):
            continue
        pg = pdf(pid)[pno]
        W, H = pg.rect.width, pg.rect.height
        clip = fitz.Rect(a * W, b * H, c * W, d * H)
        p = os.path.join(OUT, name + '.png')
        pg.get_pixmap(clip=clip, dpi=150, colorspace=fitz.csGRAY).save(p)
        if rot:
            Image.open(p).rotate(-rot, expand=True, fillcolor=255).save(p)
        print(name, 'page', pno + 1, trim(p))


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    paper_shots.run(JOBS)
    f1 = os.path.join(OUT, 'p-emg-fig1.png')
    if not os.environ.get('ONLY') or 'p-emg-fig1' in os.environ['ONLY']:
        im = Image.open(f1)                       # keep the drawing only: drop the running head and the caption
        im.crop((22, 78, im.width, 1170)).save(f1, optimize=True)
    f2 = os.path.join(OUT, 'p-patent-ladder.png')
    if not os.environ.get('ONLY') or 'p-patent-ladder' in os.environ['ONLY']:
        im = Image.open(f2)                       # the scanned text layer is uneven: drop the half lines at both ends
        im.crop((0, 40, im.width, im.height - 12)).save(f2, optimize=True)
    patent_figs()
