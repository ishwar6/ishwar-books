"""Turn code/papers/circuits/partN.md into a site page: replace {{FIG:key|caption}} with the generated SVG figures
and write content/papers/transformer-circuits/<slug>.md.  Usage: python assemble.py 1 [2 ...]   (no argument: every part that exists)
Also refuses em dashes, unreplaced placeholders, missing images and a few words that must never appear."""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
OUT = os.path.join(ROOT, 'content', 'papers', 'transformer-circuits')
SLUGS = {
    '1': 'part-1-the-residual-stream',
    '2': 'part-2-one-attention-head',
    '3': 'part-3-paths-through-the-model',
    '4': 'part-4-one-layer-models',
    '5': 'part-5-two-layer-models',
    '6': 'part-6-induction-heads',
    '7': 'part-7-after-the-paper',
}
EM_DASH = chr(0x2014)   # never allowed in the site's text
BANNED = ['Alammar', 'ChatGPT', 'Miro ', 'lecture', 'YouTube', 'Vizuara', 'Claude', 'Influur', 'Pulse', 'laptop']


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
