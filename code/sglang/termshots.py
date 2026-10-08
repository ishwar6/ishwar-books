"""Render the saved terminal logs of Part 5 as terminal screenshots: public/img/sglang/<name>-run.png.
The long simulate.py log is split by section so each picture stays readable.
Run after walkthrough.py, measure_prefix.py and simulate.py."""
import sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents'))
import termshot

termshot.OUT = str(HERE.parents[1] / 'public' / 'img' / 'sglang')
log = lambda f: (HERE / 'results' / f).read_text().rstrip('\n').split('\n')
sim = log('simulate_stdout.txt')


def section(*titles):
    out, keep = [], False
    for line in sim:
        if line and line[0] not in ' $' and not line.startswith('cost model'):
            keep = any(line.startswith(t) for t in titles)
        if keep:
            out.append(line)
    return out


RUN = '$ python code/sglang/simulate.py'
parts = {
    'walkthrough': ('python walkthrough.py', ['$ python code/sglang/walkthrough.py'] + log('walkthrough_stdout.txt')),
    'measure_prefix': ('python measure_prefix.py', log('measure_prefix_stdout.txt')),
    'sim_math': ('python simulate.py', [RUN + '   # sections 1-2'] + section('KV CACHE', 'PREFILL FLOPs', 'WHOLE REQUEST')),
    'sim_memory': ('python simulate.py', [RUN + '   # sections 3-4'] + section('MEMORY SHARING', 'LRU EVICTION')),
    'sim_capacity': ('python simulate.py', [RUN + '   # sections 5-6'] + section('BLOCK ALIGNMENT', 'CAPACITY SWEEP')),
    'sim_schedule': ('python simulate.py', [RUN + '   # sections 7-8', sim[1]] + section('SCHEDULING ORDER', 'STARVATION')),
    'sim_jump': ('python simulate.py', [RUN + '   # section 9'] + section('JUMP-FORWARD')),
}
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / 'results').mkdir()
    termshot.HERE = tmp                      # termshot reads HERE/results/<name>_stdout.txt
    for name, (title, lines) in parts.items():
        (Path(tmp) / 'results' / f'{name}_stdout.txt').write_text('\n'.join(lines) + '\n')
        termshot.render(name, title=title)
