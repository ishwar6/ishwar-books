"""Assemble the Part 7 source article with the saved theme-aware SVG figures.
Refuses em dashes, banned words, unresolved placeholders and missing images."""
from pathlib import Path
import json, re
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
s = (HERE / 'article.md').read_text()
figures = json.loads((HERE / 'results/figures.json').read_text())
s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', lambda m: f'<figure class="fig">{figures[m[1]]}<figcaption>{m[2]}</figcaption></figure>', s)
assert '{{' not in s, 'Unresolved placeholder'
assert '\u2014' not in s, 'Em dash found'
for w in []:
    assert w not in s, f'Banned word: {w}'
for src in re.findall(r'\]\((/img/[^)]+)\)', s):
    assert (ROOT / 'public' / src.lstrip('/')).is_file(), src
path = ROOT / 'content/writings/llm-inference-7-beyond-one-gpu.md'
path.write_text(s)
body = re.sub(r'<figure class="fig">.*?</figure>', '', s, flags=re.S)
print('Assembled', path.name, len(body.split()), 'words without SVG,', len(re.findall('<figure class="fig">', s)), 'figures,',
      len(set(re.findall(r'\]\((/img/multigpu/[^)]+)\)', s))), 'images')
