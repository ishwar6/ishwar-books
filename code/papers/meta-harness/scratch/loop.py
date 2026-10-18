"""Algorithm 1 of the paper, the Meta-Harness outer loop, line by line.

 1: Input: tasks X, LLM M, proposer P, iterations N
 2: Initialize: population H                         (the seed harnesses)
 3: Initialize: filesystem D <- empty                (stores code, scores, traces)
 4: for H in H do
 5:     E_H <- Evaluate(H, M, X)
 6:     D <- D + {(H, E_H)}
 7: for t = 1 ... N do
 8:     Proposer P queries filesystem D              (inspects prior harnesses and scores)
 9:     Proposer P proposes k new harnesses {H_1, ..., H_k}
10:     for H in {H_1, ..., H_k} do
11:         if H passes interface validation then
12:             D <- D + {(H, Evaluate(H, M, X))}
13: return Pareto frontier of harnesses stored in D
"""
import os, re, tempfile, time
from evaluate import evaluate
from harness_api import validate
from store import Store


def normalise(src):
    return re.sub(r'\s+', ' ', re.sub(r'#.*', '', src)).strip()


def evaluate_into(store, name, source, proposal, llm, labels, memory, search, log):
    """Evaluate one harness on the search set (lines 5 and 12) and add its folder to the filesystem (lines 6, 12)."""
    folder = store.folder_name(name)
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f:
        f.write(source)
    try:
        scores, traces = evaluate(f.name, llm, labels, memory, search)
    finally:
        os.unlink(f.name)
    store.add(folder, source, scores, traces, proposal)
    log(f"  + {folder:44s} search acc {scores['accuracy']:5.1f}%   ctx {scores['ctx_chars']:5d} chars   "
        f"calls {scores['calls_per_message']}   ({scores['seconds']}s)")
    return folder, scores


def meta_harness(run_dir, seeds, proposer, llm, labels, memory, search, iterations, k, log=print):
    store = Store(run_dir)                                                   # line 3: D <- empty
    log(f'== seeds ({len(seeds)})')
    for name, source, proposal in seeds:                                     # lines 4 to 6
        evaluate_into(store, name, source, proposal, llm, labels, memory, search, log)
    stats = {'proposed': 0, 'rejected': 0, 'duplicates': 0, 'evaluated': 0}
    for t in range(1, iterations + 1):                                       # line 7
        t0 = time.time()
        proposals = proposer.propose(store, t, k)                            # lines 8 and 9
        log(f'== iteration {t}: {len(proposals)} proposals ({time.time() - t0:.0f}s to propose)')
        seen = {normalise(c['source']): c['folder'] for c in store.candidates()}
        for p in proposals:                                                  # line 10
            stats['proposed'] += 1
            ok, err = validate(p['source'], labels, memory)                  # line 11
            if ok and normalise(p['source']) in seen:
                ok, err = False, f"duplicate: same code as {seen[normalise(p['source'])]}"
                stats['duplicates'] += 1
            if not ok:
                stats['rejected'] += 1
                d = store.reject(p['name'], p['source'], err, p['proposal'])
                log(f'  x {os.path.basename(d):44s} rejected: {err[:90]}')
                continue
            folder, _ = evaluate_into(store, p['name'], p['source'], p['proposal'], llm, labels, memory, search, log)
            seen[normalise(p['source'])] = folder
            stats['evaluated'] += 1
    return store, store.frontier(), stats                                    # line 13
