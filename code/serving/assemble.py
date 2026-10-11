"""Assemble Part 6 from article.md with the saved theme-aware SVG figures. Refuses em dashes, missing images,
unresolved figure placeholders and unfilled XX markers."""
from pathlib import Path
import json, re
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
s = (HERE / 'article.md').read_text()
figures = json.loads((HERE / 'results/figures.json').read_text())
s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', lambda m: f'<figure class="fig">{figures[m[1]]}<figcaption>{m[2]}</figcaption></figure>', s)
assert '{{' not in s, 'Unresolved placeholder'
assert '\u2014' not in s, 'Em dash found'
assert not re.search(r'XX[A-Z]', s), 'Unfilled XX marker: ' + re.search(r'.{40}XX[A-Z].{20}', s)[0]
for src in re.findall(r'\]\((/img/[^)]+)\)', s):
    assert (ROOT / 'public' / src.lstrip('/')).is_file(), src
path = ROOT / 'content/writings/llm-inference-6-serving-in-production.md'
path.write_text(s)
prose = re.sub(r'<figure.*?</figure>', '', s, flags=re.S)
print('Assembled', path.name, len(s.split()), 'words including SVG,', len(prose.split()), 'without;',
      s.count('<figure class="fig">'), 'figures')
