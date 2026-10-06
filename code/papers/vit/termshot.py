"""Render a saved terminal log (results/<name>_stdout.txt) as a dark terminal screenshot PNG.
Usage: python termshot.py <name> [<name> ...]   ->  public/img/papers/vit/<name>-run.png
Only the first MAX_LINES lines are drawn (with a "... N more lines" footer), so the picture stays readable."""
import os, sys
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'papers', 'vit')
MAX_LINES = 60
FONT_CANDIDATES = ['/System/Library/Fonts/Menlo.ttc', '/System/Library/Fonts/Monaco.ttf', '/Library/Fonts/Courier New.ttf']


def font(size):
    for f in FONT_CANDIDATES:
        if os.path.exists(f):
            return ImageFont.truetype(f, size)
    return ImageFont.load_default()


def render(name, title=None, max_lines=MAX_LINES, scale=2):
    lines = open(os.path.join(HERE, 'results', f'{name}_stdout.txt')).read().rstrip('\n').split('\n')
    more = max(0, len(lines) - max_lines)
    lines = lines[:max_lines] + ([f'... {more} more lines'] if more else [])
    fs = 13 * scale
    f = font(fs)
    pad, lh = 18 * scale, int(fs * 1.45)
    width = min(1100 * scale, max(f.getlength(l) for l in lines + ['x' * 40]) + 2 * pad)
    width = int(width)
    height = pad * 2 + 30 * scale + lh * len(lines)
    im = Image.new('RGB', (width, height), (24, 25, 33))
    d = ImageDraw.Draw(im)
    # title bar with three dots
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse([pad + i * 22 * scale, pad - 2 * scale, pad + i * 22 * scale + 12 * scale, pad + 10 * scale], fill=c)
    d.text((pad + 80 * scale, pad - 3 * scale), title or f'python {name}.py', font=font(12 * scale), fill=(140, 142, 160))
    y = pad + 30 * scale
    for l in lines:
        col = (205, 207, 220)
        if l.startswith('$ ') or l.startswith('>>> '):
            col = (143, 180, 255)
        elif l.startswith('==') or l.startswith('--') or l.isupper():
            col = (79, 195, 217)
        elif l.startswith('...'):
            col = (120, 122, 140)
        d.text((pad, y), l[:170], font=f, fill=col)
        y += lh
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f'{name}-run.png')
    im.save(path, optimize=True)
    print(f'{name}: {im.width}x{im.height} -> {path}')


if __name__ == '__main__':
    for n in sys.argv[1:]:
        render(n)
