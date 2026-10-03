"""Turn code/attention/partN.md into a site page: replace {{FIG:key|caption}} with the generated SVG figures."""
import json, re, sys
part = sys.argv[1]
slug = {'1': 'attention-1-self-attention', '2': 'attention-2-mqa-gqa-mla', '3': 'attention-3-sliding-window-sparse', '4': 'attention-4-linear-hybrid'}[part]
F = json.load(open(f'results/figs_part{part}.json'))
s = open(f'part{part}.md').read()
s = re.sub(r'\{\{FIG:(\w+)\|([^}]*)\}\}', lambda m: f'<figure class="fig">{F[m.group(1)]}<figcaption>{m.group(2)}</figcaption></figure>', s)
assert '{{' not in s, 'unreplaced figure'
assert '—' not in s, 'em dash found'
open(f'../../content/writings/{slug}.md', 'w').write(s)
print('wrote', slug, len(s.split()), 'words')
