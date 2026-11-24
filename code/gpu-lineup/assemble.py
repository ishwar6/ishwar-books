"""article.md -> content/writings/nvidia-gpus-every-architecture-and-product.md ({{FIG:key}} replaced by SVG).
Refuses an em dash and banned words; reports long sentences (over 25 words) in prose."""
import re, sys, json, pathlib
from figs import FIGS
HERE = pathlib.Path(__file__).parent
src = (HERE / 'article.md').read_text()
out = re.sub(r'\{\{FIG:(\w+)\}\}', lambda m: FIGS[m.group(1)](), src)
PHOTOS = json.loads((HERE / 'photos.json').read_text())
def photo(m):
    k = m.group(1); p = PHOTOS[k]
    tall = ' tall' if p['h'] > p['w'] else ''
    return (f'<figure class="fig photo{tall}"><img src="/img/writings/gpu/{k}.jpg" width="{p["w"]}" height="{p["h"]}" alt="{p["alt"]}" loading="lazy" />'
            f'<figcaption>{p["caption"]} <span class="credit">Photo: <a href="{p["page"]}">{p["author"]}</a>, {p["license"]}, Wikimedia Commons.</span></figcaption></figure>')
out = re.sub(r'\{\{PHOTO:(\w+)\}\}', photo, out)
bad = [w for w in [chr(0x2014), 'Claude', 'ChatGPT', 'plain English', 'simple words'] if w in out]
if bad: sys.exit(f'banned: {bad}')
body = re.sub(r'<figure.*?</figure>', '', out.split('---', 2)[2], flags=re.S)
long = []
for line in body.splitlines():
    if line.startswith('|') or line.startswith('#') or 'http' in line: continue
    line = re.sub(r'^>\s*(\[![A-Z]+\][^\n]*)?', '', line)
    for s in re.split(r'(?<=[.?!])\s+', line):
        n = len(s.split())
        if n > 25: long.append((n, s[:90]))
for n, s in long: print(n, s)
print('long sentences:', len(long), '| words:', len(body.split()))
(HERE.parent.parent / 'content/writings/nvidia-gpus-every-architecture-and-product.md').write_text(out)
