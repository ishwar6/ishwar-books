"""Part 8: print real answers from both training runs (saved by train.py every 25 steps) and from the held-out
evaluation before and after, so the page can quote them. No model is loaded."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

log = common.Log('r1_part8_samples')
R = common.RESULTS
for tag in ['base', 'instruct']:
    p = os.path.join(R, f'r1_part8_samples_{tag}.json')
    if not os.path.exists(p):
        continue
    for s in json.load(open(p)):
        log(f'== {tag} model, training step {s["step"]}, gold answer {s["gold"]}')
        log('Q:', s['question'][:150] + ('...' if len(s['question']) > 150 else ''))
        for a in s['answers'][:3]:
            log(f'  [accuracy {a["acc"]:.0f}, format {a["fmt"]:.0f}] {a["text"].strip()[:420]!r}')
p = os.path.join(R, 'r1_part8_eval_instruct.json')
if os.path.exists(p):
    E = json.load(open(p))
    for k in (0, 1, 2, 3):
        b, a = E['out_before'][k], E['out_after'][k]
        log(f'== held-out question {k}, gold answer {b["gold"]}')
        log('Q:', b['question'][:150] + ('...' if len(b['question']) > 150 else ''))
        log(f'  before [accuracy {b["acc"]:.0f}, format {b["fmt"]:.0f}, {b["length"]} tokens] {b["text"].strip()[:500]!r}')
        log(f'  after  [accuracy {a["acc"]:.0f}, format {a["fmt"]:.0f}, {a["length"]} tokens] {a["text"].strip()[:500]!r}')
