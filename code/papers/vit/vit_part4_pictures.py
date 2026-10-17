"""Part 4: a picture of what "1-shot" means: the 37 Oxford-IIIT Pets training pictures of the first 1-shot draw in
vit_part4_fewshot.py (same shuffles, same seeds), one per breed, with the breed name under each.
Writes public/img/papers/vit/p4-pets-oneshot.png."""
import glob, io, os
import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

HUB = os.path.expanduser('~/.cache/huggingface/hub')
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'public', 'img', 'papers', 'vit')
POOL = 20
NAMES = ['abyssinian', 'american bulldog', 'american pit bull terrier', 'basset hound', 'beagle', 'bengal', 'birman', 'bombay',
         'boxer', 'british shorthair', 'chihuahua', 'egyptian mau', 'english cocker spaniel', 'english setter', 'german shorthaired',
         'great pyrenees', 'havanese', 'japanese chin', 'keeshond', 'leonberger', 'maine coon', 'miniature pinscher', 'newfoundland',
         'persian', 'pomeranian', 'pug', 'ragdoll', 'russian blue', 'saint bernard', 'samoyed', 'scottish terrier', 'shiba inu',
         'siamese', 'sphynx', 'staffordshire bull terrier', 'wheaten terrier', 'yorkshire terrier']

df = pq.read_table(glob.glob(os.path.join(HUB, 'datasets--timm--oxford-iiit-pet/snapshots/*/data/train-*.parquet'))[0]).to_pandas()
y = df['label'].to_numpy()
rng = np.random.default_rng(0)
pool = []
for c in range(37):                                  # the same shuffle as load() in vit_part4_fewshot.py
    where = np.flatnonzero(y == c)
    rng.shuffle(where)
    pool.append(where[:POOL])
draw = np.random.default_rng(0)                      # the same draw as fewshot(..., k=1, rng=default_rng(0))
picks = [pool[c][draw.permutation(POOL)[:1]][0] for c in range(37)]

cols, cell, cap, pad = 8, 120, 30, 6
rows = (37 + cols - 1) // cols
im = Image.new('RGB', (cols * (cell + pad) + pad, rows * (cell + cap + pad) + pad), (255, 255, 255))
d = ImageDraw.Draw(im)
font = ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 12)
for n, i in enumerate(picks):
    r, c = divmod(n, cols)
    x, yy = pad + c * (cell + pad), pad + r * (cell + cap + pad)
    ph = Image.open(io.BytesIO(df['image'].iloc[i]['bytes'])).convert('RGB')
    s = min(ph.size)
    ph = ph.crop(((ph.width - s) // 2, (ph.height - s) // 2, (ph.width - s) // 2 + s, (ph.height - s) // 2 + s)).resize((cell, cell))
    im.paste(ph, (x, yy))
    name = NAMES[n] if d.textlength(NAMES[n], font=font) <= cell else NAMES[n][:16] + '.'
    d.text((x + cell / 2, yy + cell + 8), name, fill=(40, 40, 40), font=font, anchor='mt')
im.save(os.path.join(OUT, 'p4-pets-oneshot.png'), optimize=True)
print('p4-pets-oneshot.png', im.size, 'pictures:', len(picks))
