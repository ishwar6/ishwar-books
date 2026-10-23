"""Download the Wikimedia Commons images used in the article, print their licence and author, resize, and save to
public/img/writings/rayban-display/. Each entry: (output name, Commons file title, target width).
The two photochromic photos (CC0) are also joined side by side into one picture.
Run: python fetch_images.py   (or ONLY=img-a,img-b python fetch_images.py)"""
import io, json, os, re, sys, time, urllib.parse, urllib.request
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [x for x in sys.path if os.path.abspath(x or '.') != HERE]   # numbers.py here shadows the stdlib module
from PIL import Image

OUT = os.path.join(HERE, '..', '..', '..', 'public', 'img', 'writings', 'rayban-display')
UA = {'User-Agent': 'ishwarj-article/1.0 (iisudrj11@gmail.com)'}
FILES = [
    ('img-tir-plate', 'File:Acrylglas Totalreflexion01 2009-05-07.jpg', 640),
    ('img-tir-block', 'File:Total internal reflection.jpg', 640),
    ('img-lcos-coin', 'File:WXGA-R5 compared to 1 Euro.jpg', 600),
    ('img-microled', 'File:First active-driven microLED microdisplay.jpg', 770),
    ('img-photo-dark', 'File:ROCKBROS photochromic lens darkened in sunlight.jpg', 800),
    ('img-photo-clear', 'File:Photochromic cycling lens in low light conditions.jpg', 800),
    ('img-emg-lab', 'File:EMG - SIMI.jpg', 760),
    ('img-glass-evolution', 'File:Google Glasses Evolution.png', 1000),
    ('img-hololens', 'File:ISS-46 Scott Kelly with HoloLens in the Destiny lab.jpg', 760),
]
ONLY = os.environ.get('ONLY')


def get(url):
    for i in range(5):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA)).read()
        except Exception as e:
            print('  retry', e)
            time.sleep(4 + 4 * i)
    raise SystemExit('failed ' + url)


os.makedirs(OUT, exist_ok=True)
for name, title, w in FILES:
    if ONLY and name not in ONLY.split(','):
        continue
    q = urllib.parse.urlencode(dict(action='query', titles=title, prop='imageinfo', iiprop='url|extmetadata',
                                    iiurlwidth=w, format='json'))
    d = json.loads(get('https://commons.wikimedia.org/w/api.php?' + q))
    i = next(iter(d['query']['pages'].values()))['imageinfo'][0]
    m = i['extmetadata']
    g = lambda k: re.sub('<[^>]+>', '', m.get(k, {}).get('value', '') or '').strip()
    im = Image.open(io.BytesIO(get(i.get('thumburl') or i['url'])))
    if im.width > w:
        im = im.resize((w, int(im.height * w / im.width)), Image.LANCZOS)
    if im.mode in ('RGBA', 'LA', 'P'):
        bg = Image.new('RGB', im.size, 'white')
        im = im.convert('RGBA')
        bg.paste(im, mask=im.split()[3])
        im = bg
    png = title.lower().endswith('.png')
    path = os.path.join(OUT, name + ('.png' if png else '.jpg'))
    if png:
        im.save(path, optimize=True)
    else:
        im.convert('RGB').save(path, quality=84)
    print(f"{name}: {im.size} {os.path.getsize(path) // 1024} KB | {g('LicenseShortName')} | {g('Artist')[:70]} | {i['descriptionurl']}")
    time.sleep(1.5)

# one picture of the same photochromic lens, clear (dusk) and dark (sun); both CC0 by JaredMcKenzie
a, b = os.path.join(OUT, 'img-photo-clear.jpg'), os.path.join(OUT, 'img-photo-dark.jpg')
if os.path.exists(a) and os.path.exists(b):
    A, B = Image.open(a), Image.open(b)
    h = min(A.height, B.height)
    A = A.resize((int(A.width * h / A.height), h))
    B = B.resize((int(B.width * h / B.height), h))
    c = Image.new('RGB', (A.width + B.width + 12, h), 'white')
    c.paste(A, (0, 0))
    c.paste(B, (A.width + 12, 0))
    c.save(os.path.join(OUT, 'img-photochromic.jpg'), quality=84)
    os.remove(a)
    os.remove(b)
    print('img-photochromic.jpg', c.size)
