"""Build content/writings/llm-inference-2-kv-cache.md from p2_source.md.
Replaces {{FIG:key|caption}} (results/p2_figs.json, from p2_figs.py), {{KEEP:key|caption}} (the published charts
kept from the first version, results/p2_kept.json) and {{STDOUT}} (results/p2_kv_numbers_stdout.txt).
Refuses em dashes, banned words, unreplaced placeholders and missing images, and runs the STE check.
Run: python p2_assemble.py"""
import json, os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..', '..', '..')
sys.path.insert(0, os.path.join(ROOT, 'code', 'papers', 'bert'))
from assemble import ste_report, EM_DASH, BANNED

F = json.load(open(os.path.join(HERE, 'results', 'p2_figs.json')))
F.update(json.load(open(os.path.join(HERE, 'results', 'p2_kept.json'))))
s = open(os.path.join(HERE, 'p2_source.md')).read()
problems = ste_report(s, 2)
s = re.sub(r'\{\{(?:FIG|KEEP):(\w+)\|([^}]*)\}\}',
           lambda m: f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>', s)
s = s.replace('{{STDOUT}}', open(os.path.join(HERE, 'results', 'p2_kv_numbers_stdout.txt')).read().rstrip())
assert '{{' not in s, 'unreplaced placeholder'
assert EM_DASH not in s, 'em dash'
for w in BANNED + ['plain English', 'simple words', 'in simple terms', 'simply put', "let's dive"]:
    assert w.lower() not in s.lower(), w
for img in re.findall(r'\]\((/img/[^)]+)\)', s):
    assert os.path.exists(os.path.join(ROOT, 'public', img.lstrip('/'))), img
open(os.path.join(ROOT, 'content', 'writings', 'llm-inference-2-kv-cache.md'), 'w').write(s)
print('figures:', len(re.findall('<figure class="fig">', s)), 'words (prose, rough):',
      len(re.sub(r'<svg.*?</svg>', '', s).split()), 'STE problems:', problems)
