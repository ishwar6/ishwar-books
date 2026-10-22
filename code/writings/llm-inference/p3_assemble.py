"""Build content/writings/llm-inference-3-vllm.md from p3_article.md and results/p3_figs.json.
Replaces {{FIG:key|caption}} with the SVG, refuses em dashes, banned words, missing images and unused figures,
and runs the STE check from code/papers/bert/assemble.py. Run: python p3_assemble.py [--strict]"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'code', 'papers', 'bert'))
from assemble import ste_report, EM_DASH, BANNED

BANNED_PHRASES = ['plain English', 'simple words', 'in simple terms', "let's dive", 'simply put', 'In this part we will']
F = json.load(open(os.path.join(HERE, 'results', 'p3_figs.json')))
s = open(os.path.join(HERE, 'p3_article.md')).read()
problems = ste_report(s, 'p3')
used = []


def rep(m):
    used.append(m.group(1))
    return f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>'


s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', rep, s)
assert '{{' not in s, 'unreplaced figure placeholder'
assert EM_DASH not in s, 'em dash found'
for w in BANNED + BANNED_PHRASES:
    assert w.lower() not in s.lower(), f'banned: {w!r}'
unused = sorted(set(F) - set(used))
assert not unused, f'unused figures: {unused}'
for img in re.findall(r'\]\((/img/[^)]+)\)', s):
    assert os.path.exists(os.path.join(ROOT, 'public', img.lstrip('/'))), f'missing image {img}'
if '--strict' in sys.argv and problems:
    raise SystemExit('fix the STE problems first')
out = os.path.join(ROOT, 'content', 'writings', 'llm-inference-3-vllm.md')
open(out, 'w').write(s)
body = re.sub(r'<figure.*?</figure>', '', s, flags=re.S)
body = re.sub(r'```.*?```', '', body, flags=re.S)
print(f'wrote {out}: {len(used)} figures, {len(re.findall(r"!PAPER", s))} paper boxes, '
      f'{len(re.findall(r"!DEFINITION", s))} definitions, about {len(body.split())} words of prose (no code, no SVG)')
