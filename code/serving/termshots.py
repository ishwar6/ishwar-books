"""Render the saved terminal logs of Part 6 as terminal screenshots: public/img/serving/<name>-run.png.
Long logs are split by numbered section so each picture stays readable. Run after the experiment scripts."""
import re, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'agents'))
import termshot

termshot.OUT = str(HERE.parents[1] / 'public' / 'img' / 'serving')
log = lambda f: (HERE / 'results' / f'{f}_stdout.txt').read_text().rstrip('\n').split('\n')


def sections(name, nums, note=''):
    """Lines of the numbered sections `nums` (e.g. '12') from a log whose sections start with 'N. '."""
    lines, out, keep = log(name), [f'$ python code/serving/{name}.py' + (f'   # {note}' if note else '')], False
    for l in lines[1:]:
        m = re.match(r'^(\d+)\. ', l)
        if m:
            keep = m.group(1) in nums
        if keep:
            out.append(l)
    return out


parts = {
    'measure': ('python measure.py', log('measure')),
    'walkthrough': ('python walkthrough.py', log('walkthrough')),
    'queueing_a': ('python queueing.py', sections('queueing', '12', 'sections 1-2')),
    'queueing_b': ('python queueing.py', sections('queueing', '3', 'section 3')),
    'queueing_c': ('python queueing.py', sections('queueing', '45', 'sections 4-5')),
    'queueing_d': ('python queueing.py', sections('queueing', '6', 'section 6')),
    'capacity_a': ('python capacity.py', sections('capacity', '123', 'sections 1-3')),
    'capacity_b': ('python capacity.py', sections('capacity', '4', 'section 4')),
    'capacity_c': ('python capacity.py', sections('capacity', '5', 'section 5')),
    'capacity_d': ('python capacity.py', sections('capacity', '6', 'section 6')),
    'quant': ('python quant.py', log('quant')),
    'route': ('python route.py', log('route')),
    'lora': ('python lora.py', log('lora')),
    'coldstart': ('python coldstart.py', log('coldstart')),
    'autoscale': ('python autoscale.py', log('autoscale')),
    'failures_a': ('python failures.py', sections('failures', '12', 'sections 1-2')),
    'failures_b': ('python failures.py', sections('failures', '34', 'sections 3-4')),
}
only = sys.argv[1:]
with tempfile.TemporaryDirectory() as tmp:
    (Path(tmp) / 'results').mkdir()
    termshot.HERE = tmp                      # termshot reads HERE/results/<name>_stdout.txt
    for name, (title, lines) in parts.items():
        if only and name not in only:
            continue
        (Path(tmp) / 'results' / f'{name}_stdout.txt').write_text('\n'.join(lines) + '\n')
        termshot.render(name, title=title)
