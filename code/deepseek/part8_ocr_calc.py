"""Part 8 experiment 1: an optical-compression calculator for DeepSeek-OCR.

How many vision tokens does DeepEncoder make for a page, and how many text tokens does the same page hold?
The vision-token rule comes from the paper (arXiv 2510.18234, Section 3.2): 16 x 16 patches, then a 16x
convolutional compressor. Precision numbers are NOT predicted here: they are looked up in the paper's Table 2.
Text tokens are counted with the DeepSeek-V2-Lite tokenizer (about 100K vocabulary), which is a stand-in:
DeepSeek-OCR's own tokenizer has about 129K entries, so its counts differ a little. CPU only."""
import glob, math, os
import pymupdf as fitz
from transformers import AutoTokenizer
from common import Log, save

log = Log('part8_ocr_calc')
PATCH, SQUEEZE = 16, 16            # SAM patch size; the 2-layer conv compressor cuts the token count 16x

# Paper Table 1 (p6): native modes and their vision tokens
MODES = {'Tiny': 512, 'Small': 640, 'Base': 1024, 'Large': 1280}
PAPER_TOKENS = {'Tiny': 64, 'Small': 100, 'Base': 256, 'Large': 400}

# Paper Table 2 (p10): Fox English pages, precision and compression with 64 and 100 vision tokens
TABLE2 = [  # (text-token bucket, precision@64, ratio@64, precision@100, ratio@100, pages)
    ((600, 700), 96.5, 10.5, 98.5, 6.7, 7), ((700, 800), 93.8, 11.8, 97.3, 7.5, 28),
    ((800, 900), 83.8, 13.2, 96.8, 8.5, 28), ((900, 1000), 85.9, 15.1, 96.8, 9.7, 14),
    ((1000, 1100), 79.3, 16.5, 91.5, 10.6, 11), ((1100, 1200), 76.4, 17.7, 89.8, 11.3, 8),
    ((1200, 1300), 59.1, 19.7, 87.1, 12.6, 4)]


def vision_tokens(side):
    patches = (side // PATCH) ** 2
    return patches, patches // SQUEEZE


def valid_tokens(n_actual, w, h):
    """Paper Eq. 1 (p6): padding a non-square page wastes tokens; only this many carry the page."""
    return math.ceil(n_actual * (1 - (max(w, h) - min(w, h)) / max(w, h)))


log('== 1. Vision tokens per mode: 16x16 patches, then a 16x squeeze ==')
log(f'{"mode":6} {"side":>5} {"patches":>8} {"tokens":>7} {"paper":>6}')
modes = {}
for m, side in MODES.items():
    p, t = vision_tokens(side)
    modes[m] = {'side': side, 'patches': p, 'tokens': t}
    log(f'{m:6} {side:5d} {p:8d} {t:7d} {PAPER_TOKENS[m]:6d}')
    assert t == PAPER_TOKENS[m]

log('\n== 2. Valid tokens for an A4-shaped page (210 x 297 mm) padded to a square (paper Eq. 1) ==')
valid = {}
for m in ('Base', 'Large'):
    v = valid_tokens(modes[m]['tokens'], 210, 297)
    valid[m] = v
    log(f'{m:6} {modes[m]["tokens"]} tokens, of which {v} cover the page')

log('\n== 3. Text tokens on real pages (DeepSeek-V2-Lite tokenizer, a stand-in for the OCR tokenizer) ==')
snap = glob.glob(os.path.expanduser('~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite-Chat/snapshots/*/'))[0]
tok = AutoTokenizer.from_pretrained(snap)
doc = fitz.open(os.path.expanduser('~/.cache/papers/2510.18234.pdf'))
pages = {}
for label, pno in [('OCR paper, page 5 (text and a figure)', 4), ('OCR paper, page 10 (text and Table 2)', 9),
                   ('OCR paper, page 12 (text and Table 4)', 11)]:
    text = doc[pno].get_text()
    n = len(tok(text, add_special_tokens=False)['input_ids'])
    pages[label] = {'chars': len(text), 'words': len(text.split()), 'text_tokens': n}
    log(f'{label}: {len(text):,} characters, {len(text.split()):,} words, {n:,} text tokens')


def bucket(n):
    for (lo, hi), p64, r64, p100, r100, k in TABLE2:
        if lo <= n < hi:
            return {'bucket': f'{lo}-{hi}', 'p64': p64, 'p100': p100, 'pages': k}
    return None


log('\n== 4. Compression ratio = text tokens / vision tokens, for each page and mode ==')
log(f'{"page":40} {"Tiny 64":>8} {"Small 100":>10} {"Base 256":>9} {"Large 400":>10}')
rows = {}
for label, d in pages.items():
    r = {m: d['text_tokens'] / modes[m]['tokens'] for m in modes}
    rows[label] = r
    log(f'{label[:40]:40} {r["Tiny"]:7.1f}x {r["Small"]:9.1f}x {r["Base"]:8.1f}x {r["Large"]:9.1f}x')

log('\n== 5. What the paper measured near those ratios (Table 2, Fox English pages, 600-1300 text tokens) ==')
for label, d in pages.items():
    b = bucket(d['text_tokens'])
    if b:
        log(f'{label}: bucket {b["bucket"]} -> {b["p64"]}% with 64 tokens, {b["p100"]}% with 100 tokens '
            f'({b["pages"]} test pages)')
    else:
        log(f'{label}: {d["text_tokens"]} text tokens is outside the 600-1300 range of Table 2, no measured precision')

log('\n== 6. A newspaper page: the paper says 4,000 to 5,000 text tokens (p12) ==')
news = {}
for n in (4000, 5000):
    news[n] = {m: n / modes[m]['tokens'] for m in modes}
    log(f'{n} text tokens: ' + ', '.join(f'{m} {news[n][m]:.1f}x' for m in modes))
log('Under 10x needs at least 400 to 500 vision tokens, which is why the paper uses Gundam modes for newspapers.')

save('part8_ocr_calc', {'modes': modes, 'valid_a4': valid, 'pages': pages, 'ratios': rows, 'newspaper': news,
                        'table2': [{'bucket': f'{a}-{b}', 'p64': p64, 'r64': r64, 'p100': p100, 'r100': r100, 'pages': k}
                                   for (a, b), p64, r64, p100, r100, k in TABLE2]})
