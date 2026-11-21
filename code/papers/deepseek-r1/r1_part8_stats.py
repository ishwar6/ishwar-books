"""Part 8: summarise the two training logs in windows of 10 steps (means), with the time each step took.
No model is loaded."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

log = common.Log('r1_part8_stats')
for tag in ['base', 'instruct']:
    rows = [json.loads(l) for l in open(os.path.join(common.RESULTS, f'r1_part8_log_{tag}.jsonl'))]
    log(f'== {tag} model: {len(rows)} steps in {rows[-1]["seconds"] / 60:.0f} minutes')
    log('steps    accuracy  format  length  KL      clipped  equal-reward groups  seconds per step  psutil RSS GB')
    prev = 0.0
    for a in range(0, len(rows), 10):
        r = rows[a:a + 10]
        m = lambda k: sum(x[k] for x in r) / len(r)
        log(f'{r[0]["step"]:3d}-{r[-1]["step"]:3d}  {m("acc"):8.3f}  {m("fmt"):6.3f}  {m("length"):6.1f}  {m("kl"):6.3f}  '
            f'{m("clipped"):7.4f}  {m("zero_std_groups"):19.2f}  {(r[-1]["seconds"] - prev) / len(r):16.1f}  {max(x.get("rss_gb", 0) for x in r):13.1f}')
        prev = r[-1]['seconds']
    big = max(rows, key=lambda x: x['kl'])
    log(f'{tag} model, largest KL of one step: {big["kl"]:.3f} at step {big["step"]}; '
        f'median KL {sorted(x["kl"] for x in rows)[len(rows) // 2]:.3f}')
