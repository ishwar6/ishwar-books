"""Part 2: read the traces of our Algorithm 1 run (results/part2_fs, written by mh_part2.py) the way a proposer
would: count unclear answers and wrong labels per harness, and print the first unclear answer of each.
No model is called; this only reads files."""
import json
from pathlib import Path

from common import Log, HERE

log = Log('part2_traces')
D = Path(HERE) / 'results' / 'part2_fs'
log('== what the traces of each harness say')
for folder in sorted(p for p in D.iterdir() if p.is_dir()):
    traces = [json.loads(l) for l in (folder / 'traces.jsonl').read_text().splitlines()]
    unclear = [t for t in traces if t['pred'] is None]
    wrong = [t for t in traces if t['pred'] is not None and not t['correct']]
    log(f'{folder.name:22s} unclear {len(unclear):2d}   wrong {len(wrong):2d}   right {len(traces) - len(unclear) - len(wrong):2d}   of {len(traces)}')
    if unclear:
        out = unclear[0]['steps'][0]['output'].replace('\n', ' ')
        log(f'   first unclear answer: {out[:90]!r}')
