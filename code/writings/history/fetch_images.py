"""Download the Wikimedia Commons images used in the series, check their licence, resize, and save to public/img/writings/history/.
Each entry: (output name, Commons file title, target width). The licence and author are printed from the file page metadata."""
import io, json, os, time, urllib.parse, urllib.request
from PIL import Image
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'public', 'img', 'writings', 'history')
UA = {'User-Agent': 'ishwarj-history-article/1.0 (iisudrj11@gmail.com)'}
FILES = [
    ('img-cajal', "File:Cajal cortex drawings.png", 700),
    ('img-mark1', "File:Mark I Perceptron, Figure 2 of operator's manual.png", 900),
    ('img-turing', "File:Alan Turing Aged 16.jpg", 500),
    ('img-eliza', "File:ELIZA conversation.png", 751),
    ('img-hopfield', "File:Hopfield-net-vector.svg", 520),
    ('img-deepblue', "File:IBM Deep Blue at Computer History Museum (9361685537).jpg", 700),
    ('img-gtx580', "File:NVIDIA@40nm@Fermi@GF110@GeForce GTX 580@UA10B338 1041A1 N2Y540.000 GF110-375-A1 DSC04152 (27576804994).jpg", 1000),
    ('img-alphago', "File:Lee Sedol (W) vs AlphaGo (B) - Game 2.svg", 600),
    ('img-hinton', "File:Geoffrey E. Hinton, 2024 Nobel Prize Laureate in Physics (cropped1).jpg", 500),
    ('img-kernel', "File:Kernel trick idea.svg", 1100),
    ('img-iris', "File:Anderson's Iris data set.png", 900),
    ('img-galton', "File:Francis Galton 1850s.jpg", 500),
    # Part 3: reinforcement learning
    ('img-puzzlebox', 'File:Original "Puzzle Box" Apparatus Design.png', 463),
    ('img-menace', "File:Mscroggs-MENACE-cropped.jpg", 1000),
    ('img-ibm701', "File:IBM 701console.jpg", 600),
    ('img-atari', "File:Atari-2600-Wood-4Sw-Set.jpg", 900),
    ('img-sutton', "File:Richard Sutton, October 27, 2016.jpg", 371),
    ('img-shadowhand', "File:Shadow Hand Bulb large.jpg", 600),
    # Part 5: inside a modern language model
    ('img-tpu3', "File:Tensor Processing Unit 3.0.jpg", 900),
    ('img-tpu4', "File:TPU v4 (cropped).png", 800),
]
ONLY = os.environ.get('ONLY')
AS_JPG = {'img-tpu4'}   # a large PNG photo: save it as JPEG to keep the page light
def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()
        except Exception as e:
            print('  retry', e); time.sleep(4 + 4 * i)
    raise SystemExit('failed ' + url)
for name, title, w in FILES:
    if ONLY and name not in ONLY.split(','):
        continue
    q = urllib.parse.urlencode(dict(action='query', titles=title, prop='imageinfo', iiprop='url|extmetadata', iiurlwidth=w, format='json'))
    d = json.loads(get('https://commons.wikimedia.org/w/api.php?' + q))
    p = next(iter(d['query']['pages'].values()))
    i = p['imageinfo'][0]; m = i['extmetadata']
    g = lambda k: (m.get(k, {}).get('value', '') or '')
    import re
    strip = lambda s: re.sub('<[^>]+>', '', s).strip()
    im = Image.open(io.BytesIO(get(i.get('thumburl') or i['url'])))
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    jpg = im.mode in ('RGB', 'L') and title.lower().endswith(('.jpg', '.jpeg'))
    if name in AS_JPG:
        im, jpg = im.convert('RGB'), True
    if not jpg and im.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', im.size, 'white'); im = im.convert('RGBA'); bg.paste(im, mask=im.split()[3]); im = bg
    path = os.path.join(OUT, name + ('.jpg' if jpg else '.png'))
    if jpg:
        im.convert('RGB').save(path, quality=84)
    else:
        im.save(path, optimize=True)
    print(f"{name}: {os.path.basename(path)} {im.size} {os.path.getsize(path)//1024} KB | {g('LicenseShortName')} | {strip(g('Artist'))[:80]} | {i['descriptionurl']}")
    time.sleep(1.5)
