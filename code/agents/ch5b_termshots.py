"""Chapter 5, second half: terminal screenshots of the saved runs. The long per-question logs are cut to their summary parts first
(results/<name>_summary_stdout.txt, copied verbatim from the end of the full log), then termshot.py renders each as a PNG.
Usage: python ch5b_termshots.py"""
import os
from termshot import render, HERE

R = os.path.join(HERE, 'results')


def tail_from(name, marker, out, head=None):
    lines = open(os.path.join(R, f'{name}_stdout.txt')).read().split('\n')
    i = next(k for k, l in enumerate(lines) if l.startswith(marker))
    keep = (lines[:head] + ['...'] if head else []) + lines[i:]
    open(os.path.join(R, f'{out}_stdout.txt'), 'w').write('\n'.join(keep).rstrip('\n') + '\n')
    return out


def blocks(name, starts, out, head=2):
    lines = open(os.path.join(R, f'{name}_stdout.txt')).read().split('\n')
    keep = lines[:head]
    for st in starts:
        i = next(k for k, l in enumerate(lines) if l.startswith(st))
        j = next(k for k in range(i + 1, len(lines)) if not lines[k].strip())
        keep += [''] + lines[i:j]
    open(os.path.join(R, f'{out}_stdout.txt'), 'w').write('\n'.join(keep) + '\n')
    return out


def wrap_file(n, width=124):
    p = os.path.join(R, f'{n}_stdout.txt')
    out = []
    for l in open(p).read().rstrip('\n').split('\n'):
        while len(l) > width:
            cut = l.rfind(' ', 0, width)
            cut = cut if cut > 40 else width
            out.append(l[:cut]); l = '      ' + l[cut:].lstrip()
        out.append(l)
    open(p, 'w').write('\n'.join(out) + '\n')
    return n


for n in [tail_from('ch5b_debate', '== Summary', 'ch5b_debate_summary', head=2), 'ch5b_debate_stats',
          blocks('ch5b_handoff', ['-- free, run 1', '-- free+ver, run 1', '-- schema, run 1'], 'ch5b_handoff_run1'), tail_from('ch5b_handoff', '== Facts kept', 'ch5b_handoff_summary', head=2),
          'ch5b_handoff_stats', 'ch5b_math']:
    wrap_file(n) if n.endswith(('_summary', '_run1')) else None
    render(n, title=f'python {n.replace("_summary", "").replace("_run1", "")}.py')
