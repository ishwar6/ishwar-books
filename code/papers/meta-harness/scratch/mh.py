"""A small CLI over a run folder (Appendix D: "Make logs queryable through a small CLI").

  python mh.py runs/scripted-1 list              every candidate: search accuracy, prompt size, calls
  python mh.py runs/scripted-1 frontier          the Pareto frontier on the search set
  python mh.py runs/scripted-1 errors 004        the wrong answers of candidate 004, with the model's reply
  python mh.py runs/scripted-1 diff 001 004      the code difference between two candidates
  python mh.py runs/scripted-1 test              search vs test accuracy (only after the search is over)
"""
import difflib, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from store import Store


def find(store, prefix):
    return next(c for c in store.candidates() if c['folder'].startswith(prefix))


def main(argv):
    run, cmd, rest = argv[0], argv[1], argv[2:]
    store = Store(run)
    if cmd == 'list':
        for c in store.candidates():
            s = c['scores']
            print(f"{c['folder']:44s} {s['accuracy']:5.1f}%  {s['ctx_chars']:5d} chars  {s['calls_per_message']} calls")
    elif cmd == 'frontier':
        for c in sorted(store.frontier(), key=lambda c: c['scores']['ctx_chars']):
            print(f"{c['folder']:44s} {c['scores']['accuracy']:5.1f}%  {c['scores']['ctx_chars']:5d} chars")
    elif cmd == 'errors':
        c = find(store, rest[0])
        for t in store.traces(c['folder']):
            if not t['correct']:
                print(f"gold {t['gold']:28s} pred {str(t['pred']):28s} reply {t['calls'][-1]['reply']!r:30s} {t['text'][:70]}")
    elif cmd == 'diff':
        a, b = find(store, rest[0]), find(store, rest[1])
        sys.stdout.writelines(difflib.unified_diff(a['source'].splitlines(True), b['source'].splitlines(True),
                                                   a['folder'], b['folder'], n=1))
    elif cmd == 'test':
        for row in json.load(open(os.path.join(run, 'summary.json'))):
            if 'test_acc' in row:
                print(f"{row['folder']:44s} search {row['search_acc']:5.1f}%  test {row['test_acc']:5.1f}%  "
                      f"gap {row['gap']:+5.1f}{'  frontier' if row['frontier'] else ''}")
    else:
        print(__doc__)


if __name__ == '__main__':
    main(sys.argv[1:]) if len(sys.argv) > 2 else print(__doc__)
