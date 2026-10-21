"""Turn code/writings/history/part3.md, part4.md and part5.md into site pages: replace {{FIG:key}} with the SVG figures from
results/figs_rl.json (run figs_rl.py first) and write content/writings/<slug>.md.
Refuses em dashes, unreplaced placeholders, missing images and words that must never appear, and runs the same
ASD-STE100 check as the BERT series (code/papers/bert/assemble.py: ste_report) on our own prose.
Usage: python assemble_rl.py [3] [4] [5] [--strict]"""
import json, os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'code', 'papers', 'bert'))
from assemble import ste_report, EM_DASH   # noqa: E402

SLUGS = {'3': 'reinforcement-learning-history', '4': 'how-models-are-trained-today', '5': 'inside-a-modern-language-model'}
# "Claude" is allowed here only as a historical subject (Anthropic's models), so it is checked by hand, not banned.
BANNED = ['Alammar', 'ChatGPT', 'Miro ', 'lecture', 'YouTube', 'Vizuara', 'plain English', 'simple words',
          'in simple terms', "let's dive in", 'Let’s dive in', 'STE100', 'Simplified Technical English']


def build(part, strict=False):
    F = json.load(open(os.path.join(HERE, 'results', 'figs_rl.json')))
    s = open(os.path.join(HERE, f'part{part}.md')).read()
    used = set()

    def rep(m):
        used.add(m.group(1))
        return F[m.group(1)]
    s = re.sub(r'\{\{FIG:([\w-]+)\}\}', rep, s)
    assert '{{' not in s, 'unreplaced placeholder'
    assert EM_DASH not in s, 'em dash found'
    for w in BANNED:
        assert w.lower() not in s.lower(), f'banned word: {w!r}'
    for img in re.findall(r'\]\((/img/[^)]+)\)|src="(/img/[^"]+)"', s):
        p = next(x for x in img if x)
        assert os.path.exists(os.path.join(ROOT, 'public', p.lstrip('/'))), f'missing image {p}'
    body = s.split('---', 2)[2]
    problems = ste_report(body, part)
    if strict and problems:
        raise SystemExit(f'part {part}: fix the STE problems first')
    open(os.path.join(ROOT, 'content', 'writings', SLUGS[part] + '.md'), 'w').write(s)
    prose = re.sub(r'<figure.*?</figure>', '', body, flags=re.S)
    print(f'part {part}: wrote {SLUGS[part]}.md, {len(prose.split())} words (figures excluded), {len(used)} figures')


if __name__ == '__main__':
    args = [a for a in sys.argv[1:] if not a.startswith('-')]
    for p in args or [p for p in SLUGS if os.path.exists(os.path.join(HERE, f'part{p}.md'))]:
        build(p, strict='--strict' in sys.argv)
