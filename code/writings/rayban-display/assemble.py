"""Turn code/writings/rayban-display/article.md into content/writings/how-ray-ban-display-glasses-work.md:
replace {{FIG:key}} with the SVG figures from results/figs.json (run numbers.py and figs.py first).
Refuses em dashes, unreplaced placeholders, missing images and words that must never appear, and runs the same
ASD-STE100 check as the BERT series (code/papers/bert/assemble.py: ste_report) on our own prose.
Usage: python assemble.py [--strict]"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path = [x for x in sys.path if os.path.abspath(x or '.') != HERE]   # numbers.py here shadows the stdlib module
ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'code', 'papers', 'bert'))
from assemble import ste_report, EM_DASH   # noqa: E402

SLUG = 'how-ray-ban-display-glasses-work'
BANNED = ['Alammar', 'ChatGPT', 'Claude', 'Miro ', 'lecture', 'YouTube', 'Vizuara', 'plain English', 'simple words',
          'in simple terms', "let's dive in", 'Let’s dive in', 'simply put', 'STE100', 'Simplified Technical English',
          'for beginners']


def build(strict=False):
    F = json.load(open(os.path.join(HERE, 'results', 'figs.json')))
    s = open(os.path.join(HERE, 'article.md')).read()
    used = []

    def rep(m):
        used.append(m.group(1))
        return F[m.group(1)]['svg']
    s = re.sub(r'\{\{FIG:([\w-]+)\}\}', rep, s)
    assert '{{' not in s, 'unreplaced placeholder'
    assert EM_DASH not in s, 'em dash found'
    for w in BANNED:
        assert w.lower() not in s.lower(), f'banned word: {w!r}'
    unused = sorted(set(F) - set(used))
    if unused:
        print('WARNING unused figures:', ', '.join(unused))
    imgs = re.findall(r'\]\((/img/[^)]+)\)|src="(/img/[^"]+)"', s)
    for img in imgs:
        p = next(x for x in img if x)
        assert os.path.exists(os.path.join(ROOT, 'public', p.lstrip('/'))), f'missing image {p}'
    body = s.split('---', 2)[2]
    problems = ste_report(body, 'rayban')
    if strict and problems:
        raise SystemExit('fix the STE problems first')
    open(os.path.join(ROOT, 'content', 'writings', SLUG + '.md'), 'w').write(s)
    prose = re.sub(r'<figure.*?</figure>', '', body, flags=re.S)
    prose = re.sub(r'\$\$.*?\$\$', '', prose, flags=re.S)
    prose = re.sub(r'\(/img/[^)]+\)|\(https?://[^)]+\)', '', prose)
    print(f'wrote {SLUG}.md: {len(prose.split())} words (figures and maths excluded), {len(used)} SVG figures, '
          f'{len(set(next(x for x in i if x) for i in imgs))} images')


if __name__ == '__main__':
    build(strict='--strict' in sys.argv)
