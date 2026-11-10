"""Part 8 experiment 4: DeepSeek-VL2 dynamic tiling, as a calculator.

Candidate grids: (m x 384) by (n x 384) with m * n <= 9 (paper, arXiv 2412.10302, p4).
Pick: the released code (deepseek_vl2/models/processing_deepseek_vl_v2.py, select_best_resolution) keeps the
grid with the largest effective resolution, then the least wasted area. The paper's text says "minimizes the
padding area"; for most images both rules agree.
Tokens: each 384 x 384 tile gives 27 x 27 SigLIP outputs, squeezed by a 2 x 2 pixel shuffle to 14 x 14 = 196.
Sequence = global thumbnail 14 x (14 + 1) + 1 separator + (rows * 14) x (cols * 14 + 1)  (paper p5, same in code)."""
from common import Log, save

log = Log('part8_vl2_tiles')
T, G = 384, 14
CANDS = [(w * T, h * T) for w in range(1, 10) for h in range(1, 10) if w * h <= 9]


def best(W, H):
    best_fit, max_eff, min_waste = None, 0, float('inf')
    for cw, ch in CANDS:
        s = min(cw / W, ch / H)
        eff = min(int(W * s) * int(H * s), W * H)
        waste = cw * ch - eff
        if eff > max_eff or (eff == max_eff and waste < min_waste):
            best_fit, max_eff, min_waste = (cw, ch), eff, waste
    return best_fit


def tokens(cols, rows):
    return G * (G + 1) + 1 + (rows * G) * (cols * G + 1)


images = [('small square photo', 384, 384), ('phone photo 3024 x 4032', 3024, 4032),
          ('screenshot 1920 x 1080', 1920, 1080), ('A4 scan at 150 dpi', 1240, 1754), ('panorama 3000 x 1000', 3000, 1000)]
log(f'{len(CANDS)} candidate grids (cols x rows with cols * rows <= 9)')
log(f'{"image":26} {"grid":>7} {"tiles":>6} {"vision tokens":>14}')
out = []
for name, W, H in images:
    cw, ch = best(W, H)
    cols, rows = cw // T, ch // T
    n = tokens(cols, rows)
    out.append({'image': name, 'W': W, 'H': H, 'cols': cols, 'rows': rows, 'tiles': 1 + cols * rows, 'tokens': n})
    log(f'{name:26} {cols}x{rows:<5} {1 + cols * rows:6d} {n:14,d}')
log('\nEvery tile is one 384 x 384 SigLIP pass: 729 outputs before the 2 x 2 squeeze, 196 after.')
save('part8_vl2_tiles', out)
