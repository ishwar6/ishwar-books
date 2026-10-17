"""The curated pictures of the demo: COCO val2017 photos (each from Flickr, under the Creative Commons licence COCO
lists for it) and the two-cat photo that Part 1 of the series uses (also COCO val2017).
Downloads each photo, resizes it to 224 x 224 the way the model's preprocessing does (PIL bilinear, no crop),
and writes public/demos/vit/img/<slug>.png. That PNG is exactly what the model reads, in Python and in the browser."""
import io, os, urllib.request
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'vit', 'img')
LICENCES = {4: ('CC BY 2.0', 'https://creativecommons.org/licenses/by/2.0/'),
            5: ('CC BY-SA 2.0', 'https://creativecommons.org/licenses/by-sa/2.0/')}

# slug, COCO image id, COCO licence id, Flickr photo page (from the photo id in COCO's flickr_url), short name
PICTURES = [
    ('cats', 39769, 5, 'https://www.flickr.com/photo.gne?id=210383891', 'Two cats'),
    ('zebra', 552902, 4, 'https://www.flickr.com/photo.gne?id=8603855827', 'Zebra'),
    ('dog', 309938, 5, 'https://www.flickr.com/photo.gne?id=3742502573', 'Small dog'),
    ('bird', 56545, 4, 'https://www.flickr.com/photo.gne?id=9470927378', 'Bird on a branch'),
    ('polar-bear', 572517, 4, 'https://www.flickr.com/photo.gne?id=8754578923', 'Polar bear and heron'),
    ('teddy', 73118, 4, 'https://www.flickr.com/photo.gne?id=8017027419', 'Two teddy bears'),
    ('bananas', 28809, 5, 'https://www.flickr.com/photo.gne?id=4444464874', 'Bananas'),
    ('giraffes', 205282, 5, 'https://www.flickr.com/photo.gne?id=8892272617', 'Giraffes'),
]


def coco_url(i):
    return f'http://images.cocodataset.org/val2017/{i:012d}.jpg'


def load(i):
    return Image.open(io.BytesIO(urllib.request.urlopen(coco_url(i)).read())).convert('RGB')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for slug, i, lic, _, name in PICTURES:
        im = load(i)
        small = im.resize((224, 224), Image.BILINEAR)
        small.save(os.path.join(OUT, f'{slug}.png'), optimize=True)
        print(f'{slug:11s} COCO {i:6d}  {im.size[0]} x {im.size[1]} -> 224 x 224  {LICENCES[lic][0]}')
