"""Export the saved search runs of our small Meta-Harness (../scratch/runs/<run>/) for the demo page.

For each finished run (it has run.json and summary.json; runs named smoke* are skipped) it writes, under
public/demos/meta-harness/run/:

  index.json                    the list of runs
  <run>.json                    every candidate: code, parent, the proposal text, search scores, test score,
                                per-message right/wrong on the search set, rejected candidates, the log
  <run>/<candidate>.json        the full search-set traces of one candidate (every prompt and reply), loaded on click

Nothing is computed here that the run did not save, except two things the page needs: the parent folder (parsed from
proposal.md) and the iteration (parsed from proposal.md; seeds are iteration 0). The script checks that every
number it writes equals the run's own files (scores.json, summary.json, heldout/*.json).

    ~/.venvs/attn/bin/python export_run.py [run names...]
"""
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
RUNS = Path(os.environ.get('MH_RUNS', HERE.parent / 'scratch' / 'runs'))
OUT = ROOT / 'public' / 'demos' / 'meta-harness' / 'run'


def dump(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    text = text.replace(str(ROOT) + '/', '').replace(str(Path.home()), '~')     # no local paths in published files
    path.write_text(text)
    return path.stat().st_size


def proposal_info(text):
    # the scripted proposer writes "Parent: <folder>"; the LLM proposer is shown the best harness so far to start from
    parent = (re.search(r'^Parents?:\s*`?([0-9]{3}_[A-Za-z0-9_]+)', text, re.M)
              or re.search(r'also shown:\s*([0-9]{3}_[A-Za-z0-9_]+)/harness\.py', text))
    it = re.search(r'iteration\s+(\d+)', text[:400], re.I)
    return (parent.group(1) if parent else None), (int(it.group(1)) if it else 0)


def export(run_dir):
    name = run_dir.name
    meta = json.loads((run_dir / 'run.json').read_text())
    summary = {r['folder']: r for r in json.loads((run_dir / 'summary.json').read_text())}
    cands, total = [], 0
    for d in sorted((run_dir / 'candidates').iterdir()):
        if not (d / 'scores.json').exists():
            continue
        scores = json.loads((d / 'scores.json').read_text())
        traces = [json.loads(l) for l in (d / 'traces.jsonl').read_text().splitlines() if l.strip()]
        proposal = (d / 'proposal.md').read_text() if (d / 'proposal.md').exists() else ''
        parent, it = proposal_info(proposal)
        row = summary[d.name]
        assert row['search_acc'] == scores['accuracy']
        assert round(100 * sum(t['correct'] for t in traces) / len(traces), 1) == scores['accuracy']
        held = run_dir / 'heldout' / f'{d.name}.json'
        test = None
        if held.exists():
            h = json.loads(held.read_text())
            test = {'acc': h['scores']['accuracy'], 'correct': h['scores']['correct'], 'n': h['scores']['n'],
                    'ctx_chars': h['scores']['ctx_chars'], 'ok': [1 if t['correct'] else 0 for t in h['traces']]}
            assert test['acc'] == row.get('test_acc')
        files = [{'path': f.name, 'bytes': f.stat().st_size} for f in sorted(d.iterdir()) if f.is_file() and not f.name.startswith('.')]
        cands.append({
            'folder': d.name, 'iter': it, 'seed': row.get('seed', False), 'frontier': row.get('frontier', False),
            'parent': parent, 'proposal': proposal, 'code': (d / 'harness.py').read_text(), 'files': files,
            'search': {k: scores[k] for k in ['accuracy', 'correct', 'n', 'ctx_chars', 'calls_per_message', 'invalid',
                                               'crashes', 'accuracy_by_label', 'top_confusions'] if k in scores},
            'test': test,
            # per search message: right (1) or wrong (0), and the predicted label
            'ok': [1 if t['correct'] else 0 for t in traces], 'pred': [t['pred'] for t in traces],
        })
        total += dump(OUT / name / f'{d.name}.json', [
            {'id': t['id'], 'text': t['text'], 'gold': t['gold'], 'pred': t['pred'], 'ok': t['correct'],
             'calls': t['calls'], 'chars': t['prompt_chars'], **({'error': t['error']} if 'error' in t else {})}
            for t in traces])
    rejected = []
    if (run_dir / 'rejected').exists():
        for d in sorted((run_dir / 'rejected').iterdir()):
            proposal = (d / 'proposal.md').read_text() if (d / 'proposal.md').exists() else ''
            parent, it = proposal_info(proposal)
            rejected.append({'folder': d.name, 'iter': it, 'parent': parent, 'proposal': proposal,
                             'error': (d / 'error.txt').read_text().strip() if (d / 'error.txt').exists() else '',
                             'code': (d / 'harness.py').read_text() if (d / 'harness.py').exists() else ''})
    # the search messages themselves (the same for every candidate), so the page can show which message is which
    first = sorted((run_dir / 'candidates').iterdir())[0]
    msgs = [{'text': t['text'], 'gold': t['gold']} for t in map(json.loads, (first / 'traces.jsonl').read_text().splitlines())]
    out = {'name': name, 'meta': meta, 'candidates': cands, 'rejected': rejected, 'messages': msgs,
           'log': (run_dir / 'log.txt').read_text() if (run_dir / 'log.txt').exists() else ''}
    total += dump(OUT / f'{name}.json', out)
    sel = next(c for c in cands if c['folder'] == meta['selected'])
    print(f"{name}: {len(cands)} candidates, {len(rejected)} rejected; selected {sel['folder']} "
          f"search {sel['search']['accuracy']}% test {sel['test'] and sel['test']['acc']}%; {total / 1e6:.2f} MB")
    return {'file': f'{name}.json', 'name': name, 'proposer': meta.get('proposer'), 'candidates': len(cands)}


if __name__ == '__main__':
    names = sys.argv[1:] or sorted(d.name for d in RUNS.iterdir()
                                   if (d / 'run.json').exists() and (d / 'summary.json').exists() and not d.name.startswith('smoke'))
    names.sort(key=lambda n: (not n.startswith('scripted'), n))      # the scripted run first: the page opens on it
    index = [export(RUNS / n) for n in names]
    dump(OUT / 'index.json', {'runs': index})
    print('wrote', OUT.relative_to(ROOT), [r['name'] for r in index])
