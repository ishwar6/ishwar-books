"""Turn code/papers/vit/partN.md into a site page: replace {{FIG:key|caption}} with the generated SVG figures
and write content/papers/vit/<slug>.md.  Usage: python assemble.py 1 [2 ...]   (no argument: every part that exists)
Also refuses em dashes, unreplaced placeholders, missing images and a few words that must never appear."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
OUT = os.path.join(ROOT, 'content', 'papers', 'vit')
SLUGS = {
    '1': 'part-1-the-big-idea',
    '2': 'part-2-the-model',
    '3': 'part-3-fine-tuning-and-setup',
    '4': 'part-4-results',
    '5': 'part-5-scaling-and-inside-vit',
    '6': 'part-6-impact-and-summary',
}
EM_DASH = chr(0x2014)   # never allowed in the site's text
BANNED = ['Alammar', 'ChatGPT', 'Miro ', 'lecture', 'YouTube', 'Vizuara']


def build(part):
    src = os.path.join(HERE, f'part{part}.md')
    figs = os.path.join(HERE, 'results', f'figs_part{part}.json')
    F = json.load(open(figs)) if os.path.exists(figs) else {}
    s = open(src).read()
    used = set()
    def rep(m):
        used.add(m.group(1))
        return f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>'
    s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', rep, s)
    assert '{{' not in s, 'unreplaced figure placeholder'
    assert EM_DASH not in s, 'em dash found'
    for w in BANNED:
        assert w not in s, f'banned word: {w!r}'
    unused = sorted(set(F) - used)
    if unused:
        print(f'part {part}: WARNING unused figures: {", ".join(unused)}')
    for img in re.findall(r'\]\((/img/[^)]+)\)|src="(/img/[^"]+)"', s):
        p = next(x for x in img if x)
        assert os.path.exists(os.path.join(ROOT, 'public', p.lstrip('/'))), f'missing image {p}'
    os.makedirs(OUT, exist_ok=True)
    open(os.path.join(OUT, SLUGS[part] + '.md'), 'w').write(s)
    print(f'part {part}: wrote {SLUGS[part]}.md, {len(s.split())} words, {len(used)} figures')


if __name__ == '__main__':
    parts = sys.argv[1:] or [p for p in SLUGS if os.path.exists(os.path.join(HERE, f'part{p}.md'))]
    for p in parts:
        build(p)
