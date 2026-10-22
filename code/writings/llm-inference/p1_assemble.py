"""Build content/writings/llm-inference-1-prefill-and-decode.md from p1_source.md and results/p1_figs.json.
Replaces {{FIG:key|caption}} with the SVG figures, refuses em dashes, banned words and missing images,
and runs the site's sentence-length and word check (ste_report from code/papers/bert/assemble.py)."""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'code', 'papers', 'bert'))
from assemble import ste_report, EM_DASH, BANNED

src = open(os.path.join(HERE, 'p1_source.md')).read()
F = json.load(open(os.path.join(HERE, 'results', 'p1_figs.json')))
body = src.split('---', 2)[2]
problems = ste_report(body, 'llm-inference 1')
used = []

def rep(m):
    used.append(m.group(1))
    return f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>'

out = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', rep, src)
assert '{{' not in out, 'unreplaced figure placeholder'
assert EM_DASH not in out, 'em dash found'
for w in BANNED + ['plain English', 'simple words', 'in simple terms', 'simply put', "let's dive"]:
    assert w.lower() not in out.lower(), f'banned phrase: {w!r}'
for img in re.findall(r'\]\((/img/[^)]+)\)', out):
    assert os.path.exists(os.path.join(ROOT, 'public', img.lstrip('/'))), f'missing image {img}'
print(f'{len(used)} figures: {", ".join(used)}')
print(f'unused figures: {sorted(set(F) - set(used))}')
if problems and '--strict' in sys.argv:
    raise SystemExit('STE problems: not written')
open(os.path.join(ROOT, 'content', 'writings', 'llm-inference-1-prefill-and-decode.md'), 'w').write(out)
print('written')
