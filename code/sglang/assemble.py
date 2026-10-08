"""Assemble the SGLang source article with saved theme-aware SVG figures."""
from pathlib import Path
import json,re
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
s=(HERE/'article.md').read_text()
figures=json.loads((HERE/'results/figures.json').read_text())
s=re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}',lambda m:f'<figure class="fig">{figures[m[1]]}<figcaption>{m[2]}</figcaption></figure>',s)
assert '{{' not in s,'Unresolved placeholder'
assert '\u2014' not in s,'Em dash found'
for src in re.findall(r'\]\((/img/[^)]+)\)',s):
    assert (ROOT/'public'/src.lstrip('/')).is_file(),src
path=ROOT/'content/writings/llm-inference-5-sglang-vs-vllm.md'
path.write_text(s)
print('Assembled',path.name,len(s.split()),'words including SVG')
