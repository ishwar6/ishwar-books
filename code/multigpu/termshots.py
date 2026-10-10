"""Render the saved terminal logs of Part 7 as terminal screenshots: public/img/multigpu/<name>-run.png.
Long logs are split by section so each picture stays readable. Run after the experiment scripts."""
import sys, tempfile, textwrap
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents'))
import termshot

termshot.OUT = str(HERE.parents[1] / 'public' / 'img' / 'multigpu')
log = lambda f: (HERE / 'results' / f).read_text().rstrip('\n').split('\n')


def between(lines, start, stop=None):
    """Lines from the first one starting with `start` up to (not including) the first starting with `stop`."""
    i = next(k for k, l in enumerate(lines) if l.startswith(start))
    j = next((k for k, l in enumerate(lines) if stop and k > i and l.startswith(stop)), len(lines))
    return lines[i:j]


tp, col, dis = log('tp_demo_stdout.txt'), log('collectives_stdout.txt'), log('disagg_sim_stdout.txt')
dis_short = [l for l in dis if not l.startswith('   ') or any(f' {r} req/s' in l for r in (' 8', '16', '24', '32', ' 2', ' 4'))]
parts = {
    'memory_math': ('python memory_math.py', ['$ python code/multigpu/memory_math.py'] + log('memory_math_stdout.txt')),
    'tp_demo_b': ('python tp_demo.py', ['$ python code/multigpu/tp_demo.py   # parts A and B'] + between(tp, 'PART A', 'PART C')),
    'tp_demo_c': ('python tp_demo.py', ['$ python code/multigpu/tp_demo.py   # part C'] + between(tp, 'PART C')),
    'collectives': ('python collectives.py', ['$ python code/multigpu/collectives.py   # sections 1-2'] + between(col, '1. RING', '3. COST')),
    'collectives_cost': ('python collectives.py', ['$ python code/multigpu/collectives.py   # sections 3-4'] + between(col, '3. COST')),
    'layout_model': ('python layout_model.py', ['$ python code/multigpu/layout_model.py'] + between(log('layout_model_stdout.txt'), '1. TENSOR', '4. A PIPELINE')),
    'measure_mps': ('python measure_mps.py', ['$ python code/multigpu/measure_mps.py'] + log('measure_mps_stdout.txt')),
    'moe_routing': ('python moe_routing.py', ['$ python code/multigpu/moe_routing.py'] + log('moe_routing_stdout.txt')),
    'kv_transfer': ('python kv_transfer.py', ['$ python code/multigpu/kv_transfer.py'] + log('kv_transfer_stdout.txt')),
    'disagg_sim': ('python disagg_sim.py', ['$ python code/multigpu/disagg_sim.py   # selected rates'] + dis_short),
}
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / 'results').mkdir()
    termshot.HERE = tmp                      # termshot reads HERE/results/<name>_stdout.txt
    for name, (title, lines) in parts.items():
        wrapped = []                         # wrap long lines so nothing is cut off at the picture's edge
        for l in lines:
            ind = len(l) - len(l.lstrip())
            wrapped += textwrap.wrap(l, 118, subsequent_indent=' ' * (ind + 4), drop_whitespace=False) or ['']
        (Path(tmp) / 'results' / f'{name}_stdout.txt').write_text('\n'.join(wrapped) + '\n')
        termshot.render(name, title=title, max_lines=70)
