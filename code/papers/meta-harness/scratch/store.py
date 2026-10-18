"""The filesystem D of Algorithm 1: one folder per evaluated harness, as in the paper (§3, Figure 2).

runs/<run>/
  run.json                    the settings of the run
  candidates/<NNN_name>/      one folder per candidate the proposer can read
      harness.py              the source code
      scores.json             search-set accuracy, context size, calls, errors by label
      traces.jsonl            one line per search message: the prompts, the model's replies, the answer, the gold label
      proposal.md             who proposed it and why (the proposer's own words)
  rejected/<NNN_name>/        candidates that failed interface validation: source and the error
  heldout/<NNN_name>.json     test-set results. Written only after the search. Proposers never read this folder.
  summary.json                every candidate: search score, context, test score, on the Pareto frontier or not
"""
import json, os, re


def pareto_front(points):
    """Method 1, from the definition. points: list of (accuracy, context). Higher accuracy and lower context are better.
    A point is on the frontier if no other point is at least as good on both and strictly better on one."""
    front = []
    for i, (a, c) in enumerate(points):
        dominated = any((a2 >= a and c2 <= c) and (a2 > a or c2 < c) for j, (a2, c2) in enumerate(points) if j != i)
        if not dominated:
            front.append(i)
    return front


def pareto_front_sweep(points):
    """Method 2, the usual fast way: sort by context (cheapest first), keep each point that beats the best accuracy so far."""
    order = sorted(range(len(points)), key=lambda i: (points[i][1], -points[i][0]))
    front, best = [], float('-inf')
    for i in order:
        if points[i][0] > best:
            front.append(i)
            best = points[i][0]
        elif front and points[i] == points[front[-1]]:      # an exact tie with a kept point: neither dominates
            front.append(i)
    return sorted(front)


class Store:
    def __init__(self, run_dir):
        self.root = run_dir
        self.cand_dir = os.path.join(run_dir, 'candidates')
        for d in ('candidates', 'rejected', 'heldout'):
            os.makedirs(os.path.join(run_dir, d), exist_ok=True)

    # ---- writing ---------------------------------------------------------------------------------
    def next_id(self):
        n = [int(x[:3]) for d in ('candidates', 'rejected') for x in os.listdir(os.path.join(self.root, d))
             if x[:3].isdigit()]
        return max(n, default=-1) + 1

    def folder_name(self, name):
        safe = re.sub(r'[^a-z0-9_]+', '_', name.lower()).strip('_')[:40] or 'harness'
        return f'{self.next_id():03d}_{safe}'

    def add(self, folder, source, scores, traces, proposal):
        d = os.path.join(self.cand_dir, folder)
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'harness.py'), 'w').write(source)
        json.dump(scores, open(os.path.join(d, 'scores.json'), 'w'), indent=1)
        with open(os.path.join(d, 'traces.jsonl'), 'w') as f:
            for t in traces:
                f.write(json.dumps(t) + '\n')
        open(os.path.join(d, 'proposal.md'), 'w').write(proposal)
        return d

    def reject(self, name, source, error, proposal):
        d = os.path.join(self.root, 'rejected', self.folder_name(name))
        os.makedirs(d, exist_ok=True)
        open(os.path.join(d, 'harness.py'), 'w').write(source)
        open(os.path.join(d, 'error.txt'), 'w').write(error + '\n')
        open(os.path.join(d, 'proposal.md'), 'w').write(proposal)
        return d

    # ---- reading (what a proposer may use) -------------------------------------------------------
    def candidates(self):
        """All evaluated candidates, oldest first: dicts with folder, source and scores."""
        out = []
        for folder in sorted(os.listdir(self.cand_dir)):
            d = os.path.join(self.cand_dir, folder)
            if not os.path.exists(os.path.join(d, 'scores.json')):
                continue
            out.append({'folder': folder, 'dir': d, 'scores': json.load(open(os.path.join(d, 'scores.json'))),
                        'source': open(os.path.join(d, 'harness.py')).read()})
        return out

    def traces(self, folder):
        return [json.loads(l) for l in open(os.path.join(self.cand_dir, folder, 'traces.jsonl'))]

    def frontier(self):
        cands = self.candidates()
        pts = [(c['scores']['accuracy'], c['scores']['ctx_chars']) for c in cands]
        return [cands[i] for i in pareto_front(pts)]

    def best(self):
        cands = self.candidates()
        return max(cands, key=lambda c: (c['scores']['accuracy'], -c['scores']['ctx_chars']))
