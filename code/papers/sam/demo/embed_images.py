"""Precompute the image embedding of each demo photo with the released SAM ViT-H (facebook/sam-vit-huge).

The image encoder (637M weights, 5.96 TFLOPs per picture, see results/part3_stdout.txt) is too heavy for a browser,
so the demo downloads its output: 256 x 64 x 64 numbers per photo, stored as float16 (2 MB). The fp32 embeddings are
kept in ~/.cache/papers/sam_demo/ for the checks (reference.py, export_decoder.py).

Writes public/demos/sam/img/<key>.jpg, public/demos/sam/emb/<key>.bin (float16, little endian, (256, 64, 64)),
public/demos/sam/photos.json and code/papers/sam/demo/results/embed_stdout.txt."""
import io, os, json, time, shutil, urllib.request
import numpy as np
import torch
from PIL import Image
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import SamModel, SamProcessor

HERE = os.path.dirname(os.path.abspath(__file__))
PUB = os.path.join(HERE, '..', '..', '..', '..', 'public', 'demos', 'sam')
RES = os.path.join(HERE, 'results')
CACHE = os.path.expanduser('~/.cache/papers/sam_demo')
for d in (os.path.join(PUB, 'img'), os.path.join(PUB, 'emb'), RES, CACHE):
    os.makedirs(d, exist_ok=True)
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'

LIC = {
    'by': ('CC BY 2.0', 'https://creativecommons.org/licenses/by/2.0/'),
    'by-sa': ('CC BY-SA 2.0', 'https://creativecommons.org/licenses/by-sa/2.0/'),
    'by-nd': ('CC BY-ND 2.0', 'https://creativecommons.org/licenses/by-nd/2.0/'),
}

# key, title, source, licence, COCO id or None, Flickr URL from the COCO annotations, the first click, why this photo
PHOTOS = [
    ('cats', 'Two cats', 'COCO val2017 39769 (the huggingface/cats-image test photo of Parts 1 to 4)', 'by-sa', 39769,
     'http://farm1.staticflickr.com/60/210383891_f91a89fd5e_z.jpg', [275, 62],
     'Parts inside wholes: an ear, a head, a cat.'),
    # The car (no stated licence) and the COCO bicycle (CC BY-ND: no changed versions, and a mask overlay is one)
    # are left out of the demo on purpose.
    ('shirt', 'A shirt and a kite', 'COCO val2017 357888', 'by-sa', 357888, None, [450, 330],
     'The paper\'s example: a point on a shirt can mean the shirt or the person.'),
    ('fruit', 'A fruit stall', 'COCO val2017 498919', 'by', 498919, None, [100, 250], 'Many small objects of one kind.'),
    ('sheep', 'Sheep', 'COCO val2017 364322', 'by-sa', 364322, None, [575, 165], 'A head, a sheep, a flock.'),
    ('elephants', 'Elephants', 'COCO val2017 414795', 'by', 414795, None, [330, 180], 'Many objects that touch each other.'),
    ('carrots', 'Carrots', 'COCO val2017 530052', 'by-sa', 530052, None, [420, 420], 'Long thin objects and a price sign.'),
]


def flickr_urls():
    """The Flickr URL of each COCO photo, from instances_val2017.json (fetched once into the cache)."""
    path = os.path.join(CACHE, 'coco_val2017_images.json')
    if os.path.exists(path):
        return json.load(open(path))
    src = os.environ.get('COCO_INSTANCES_VAL2017')   # path to annotations/instances_val2017.json
    d = json.load(open(src))
    out = {str(i['id']): dict(flickr=i['flickr_url'], license=i['license'], w=i['width'], h=i['height'])
           for i in d['images']}
    json.dump(out, open(path, 'w'))
    return out


def load(key, coco_id):
    if key == 'cats':
        return load_dataset('huggingface/cats-image')['test'][0]['image'].convert('RGB')
    if key == 'car':
        im = Image.open(hf_hub_download('ybelkada/segment-anything', 'assets/car.png')).convert('RGB')   # 2646 x 1764
        return im.resize((1024, round(im.size[1] * 1024 / im.size[0])), Image.BICUBIC)
    path = os.path.join(CACHE, f'{coco_id}.jpg')
    if not os.path.exists(path):
        urllib.request.urlretrieve(f'http://images.cocodataset.org/val2017/{coco_id:012d}.jpg', path)
    return Image.open(path).convert('RGB')


def main():
    lines = []
    log = lambda s: (print(s), lines.append(s))
    proc = SamProcessor.from_pretrained('facebook/sam-vit-huge')
    sam = SamModel.from_pretrained('facebook/sam-vit-huge').to(DEV).eval()
    coco = flickr_urls()
    meta = []
    log(f'model facebook/sam-vit-huge on {DEV}; embedding (256, 64, 64) stored as float16')
    for key, title, source, lic, coco_id, _, click, why in PHOTOS:
        im = load(key, coco_id)
        w, h = im.size
        jpg = os.path.join(PUB, 'img', f'{key}.jpg')
        if coco_id and key != 'cats':
            shutil.copy(os.path.join(CACHE, f'{coco_id}.jpg'), jpg)      # the original COCO file, unchanged
        else:
            im.save(jpg, quality=90)
        inp = proc(images=im, return_tensors='pt')
        t = time.time()
        with torch.no_grad():
            emb = sam.get_image_embeddings(inp['pixel_values'].to(DEV)).float().cpu().numpy()[0]   # (256, 64, 64)
        dt = time.time() - t
        np.save(os.path.join(CACHE, f'emb_{key}_fp32.npy'), emb[None])
        e16 = emb.astype('<f2')
        e16.tofile(os.path.join(PUB, 'emb', f'{key}.bin'))
        err = np.abs(e16.astype(np.float32) - emb)
        rh, rw = inp['reshaped_input_sizes'][0].tolist()
        log(f'{key:<10} {w} x {h} -> {rw} x {rh} in the 1024 frame; embedding range [{emb.min():.3f}, {emb.max():.3f}], '
            f'float16 max abs error {err.max():.2e} (mean {err.mean():.2e}); {dt:.1f} s')
        entry = dict(key=key, title=title, file=f'img/{key}.jpg', emb=f'emb/{key}.bin', width=w, height=h,
                     source=source, why=why)
        if coco_id:
            c = coco[str(coco_id)]
            entry['url'] = c['flickr']
            entry['licence'], entry['licence_url'] = LIC[lic]
            assert {4: 'by', 5: 'by-sa', 6: 'by-nd'}[c['license']] == lic, (key, c['license'])
        else:
            entry['url'] = 'https://huggingface.co/ybelkada/segment-anything/blob/main/assets/car.png'
        if click:
            entry['click'] = click
        meta.append(entry)
    json.dump(dict(model='facebook/sam-vit-huge', embedding=[256, 64, 64], dtype='float16', photos=meta),
              open(os.path.join(PUB, 'photos.json'), 'w'), indent=1)
    open(os.path.join(RES, 'embed_stdout.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
