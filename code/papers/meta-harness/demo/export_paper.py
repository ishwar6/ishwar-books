"""Check the paper's numbers we typed by hand, then write them for the demo.

Reads paper_numbers.json (typed from the paper), extracts the text of the paper's PDF page by page, and checks that
every row of every table appears on the right page, in order. Then writes public/demos/meta-harness/paper.json.

    ~/.venvs/attn/bin/python export_paper.py [path/to/2603.28052.pdf]
"""
import json
import re
import sys
from pathlib import Path

import pymupdf

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
PDF = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / '.cache/papers/2603.28052.pdf'
OUT = ROOT / 'public/demos/meta-harness/paper.json'

data = json.loads((HERE / 'paper_numbers.json').read_text())
pages = [p.get_text() for p in pymupdf.open(PDF)]


def tokens(text):
    """The page as a list of words and numbers, so a row can be found as a run of tokens in order."""
    return re.findall(r'[A-Za-z0-9.+\-%=]+|[()]', text.replace('−', '-'))


def num(x):
    return f'{x:.1f}' if isinstance(x, float) else str(x)


def find_in_order(page, items, label):
    """Every item must appear on the page after the item before it."""
    toks = tokens(pages[page])
    pos = 0
    for it in items:
        want = tokens(it)
        for start in range(pos, len(toks) - len(want) + 1):
            if toks[start:start + len(want)] == want:
                pos = start + len(want)
                break
        else:
            raise SystemExit(f'NOT FOUND on page {page}: {label}: {it!r}')
    return len(items)


checked = 0
t = data['table1']
for r in t['rows']:
    checked += find_in_order(t['page'], [r['method'], r['history'], r['log'], str(r['mtok'])], 'Table 1')

for key, name in [('table3', 'method'), ('table4', 'method')]:
    t = data[key]
    for r in t['rows']:
        vals = [num(r['median']), num(r['best'])] + ([str(r['above_zs'])] if 'above_zs' in r else [])
        checked += find_in_order(t['page'], [r[name].split(' (')[0]] + vals, key)

t = data['table2']
for r in t['rows']:
    checked += find_in_order(t['page'], [r['harness'].split(' (')[0]] + [num(r[k]) for k in ['uspto', 's2d', 'law', 'avg']]
                             + [num(r['ctx_ktok']) if r['ctx_ktok'] else '0'], 'Table 2')

t = data['table9']
for r in t['rows']:
    checked += find_in_order(t['page'], [r['variant'].split(' ')[-1]] + [num(r[k]) for k in ['uspto', 's2d', 'law', 'avg', 'ctx_kchar']], 'Table 9')

t = data['table6']
find_in_order(t['page'], [m['short'] for m in t['models']], 'Table 6 header')
for r in t['rows']:
    if 'delta' in r:
        cells = [f'{a:.1f} ({d:+.1f})' for a, d in zip(r['acc'], r['delta'])] + [f'{r["avg"]:.1f} ({r["avg_delta"]:+.1f})']
    else:
        cells = [num(a) for a in r['acc']] + [num(r['avg'])]
    checked += find_in_order(t['page'], [r['method'].split(' (')[0]] + cells, 'Table 6')
    # the improvements must equal the differences to the no-retriever row
    if 'delta' in r:
        base = t['rows'][0]
        for a, d, b in zip(r['acc'], r['delta'], base['acc']):
            assert abs((a - b) - d) < 0.051, (r['method'], a, b, d)

t = data['table7']
for g in t['groups']:
    items = [g['model'].split()[0]]
    for r in g['rows']:
        items += ([r['harness']] if 'Anthropic' not in r['harness'] else []) + [num(r['pass'])]
    checked += find_in_order(t['page'], items, 'Table 7 ' + g['model'])

t = data['table8']
checked += find_in_order(t['page'], [str(t['files_median']), '69', '99'] + [f'{b["pct"]}%' for b in t['breakdown']], 'Table 8')

t = data['appendixA']
checked += find_in_order(t['page'], [f'{t["baseline"]}%'] + sum([[f'{x["pass"]}%', f'{x["change"]}pp'] for x in t['tries']], []), 'Appendix A')

# Figure 3: the baselines' accuracy comes from Table 2, the stars from Table 9; the 115k and 200k ticks are on the page.
find_in_order(data['figure3']['page'], ['115k', '200k', 'Additional context (chars)'], 'Figure 3 axis')

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(data, ensure_ascii=False, separators=(',', ':')))
print(f'checked {checked} cells and labels against the PDF text: all found in order')
print(f'wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size:,} bytes)')
