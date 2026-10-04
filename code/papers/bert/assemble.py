"""Turn code/papers/bert/partN.md into a site page: replace {{FIG:key|caption}} with the generated SVG figures
and write content/papers/bert/<slug>.md.  Usage: python assemble.py 1 [2 ...]   (no argument: every part that exists)"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, '..', '..', '..', 'content', 'papers', 'bert')
SLUGS = {
    '1': 'part-1-the-big-idea',
    '2': 'part-2-architecture-and-input',
    '3': 'part-3-pre-training',
    '4': 'part-4-fine-tuning-and-results',
    '5': 'part-5-ablations',
    '6': 'part-6-impact-and-summary',
}
EM_DASH = chr(0x2014)   # never allowed in the site's text


def build(part):
    src = os.path.join(HERE, f'part{part}.md')
    figs = os.path.join(HERE, 'results', f'figs_part{part}.json')
    F = json.load(open(figs)) if os.path.exists(figs) else {}
    s = open(src).read()
    s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', lambda m: f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>', s)
    assert '{{' not in s, 'unreplaced figure placeholder'
    assert EM_DASH not in s, 'em dash found'
    for img in re.findall(r'\]\((/img/[^)]+)\)|src="(/img/[^"]+)"', s):
        p = next(x for x in img if x)
        assert os.path.exists(os.path.join(HERE, '..', '..', '..', 'public', p.lstrip('/'))), f'missing image {p}'
    open(os.path.join(OUT, SLUGS[part] + '.md'), 'w').write(s)
    print(f'part {part}: wrote {SLUGS[part]}.md, {len(s.split())} words, {len(F)} figures')


if __name__ == '__main__':
    parts = sys.argv[1:] or [p for p in SLUGS if os.path.exists(os.path.join(HERE, f'part{p}.md'))]
    for p in parts:
        build(p)
