"""Chapter 3, experiment 4: HumanEval and pass@k with Qwen2.5-0.5B base.
For each of the 164 problems: one greedy completion, and n = 10 sampled completions (temperature 0.8, top-p 0.95).
Each completion is run against the problem's unit tests in a separate Python process (5 s time limit, no network,
no writing outside a temporary folder). pass@k is then computed with the unbiased estimator of Chen et al. (2021)."""
import json, os, subprocess, sys, tempfile, time
import numpy as np
from common import Log, save
from ch3_common import BASE, data, load, generate

log = Log('ch3_humaneval')
N_SAMPLES = 10
STOPS = ['\ndef ', '\nclass ', '\nif __name__', '\nprint(', '\n#', '\nassert ', '\n```']
SANDBOX = '(version 1)(allow default)(deny network*)(deny file-write*)(allow file-write* (subpath "{d}") (subpath "/dev") (subpath "/private/var/folders"))'


def pass_at_k(n, c, k):
    """Unbiased estimate of pass@k from n samples of which c passed (Chen et al. 2021, Figure 3)."""
    if n - c < k:
        return 1.0
    return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))


def truncate(completion):
    cut = min([completion.find(s) for s in STOPS if s in completion] + [len(completion)])
    return completion[:cut]


def run_tests(problem, completion):
    prog = problem['prompt'] + completion + '\n\n' + problem['test'] + f'\ncheck({problem["entry_point"]})\n'
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, 'prog.py')
        open(f, 'w').write(prog)
        cmd = [sys.executable, '-I', f]
        if os.path.exists('/usr/bin/sandbox-exec'):
            cmd = ['/usr/bin/sandbox-exec', '-p', SANDBOX.format(d=os.path.realpath(d))] + cmd
        try:
            r = subprocess.run(cmd, cwd=d, capture_output=True, timeout=5)
            return r.returncode == 0
        except subprocess.TimeoutExpired:
            return False


# ---------------------------------------------------------------- worked example of the estimator
log('== the unbiased pass@k estimator, by hand ==')
for n, c, k in [(10, 3, 1), (10, 3, 5), (10, 3, 8), (10, 0, 5), (200, 10, 100)]:
    naive = 1 - (1 - c / n) ** k
    log(f'n={n:3d} c={c:2d} k={k:3d}:  unbiased pass@k = {pass_at_k(n, c, k):.4f}   (naive 1-(1-c/n)^k = {naive:.4f})')
log('')

probs = data('humaneval').to_dict('records')
tok, model = load(BASE)
t0 = time.time()
prompts = [p['prompt'] for p in probs]
greedy = [truncate(o) for o in generate(tok, model, prompts, max_new_tokens=384, batch=24)]
g_ok = [run_tests(p, c) for p, c in zip(probs, greedy)]
log(f'greedy: {sum(g_ok)}/{len(probs)} problems pass  ->  pass@1 (greedy) = {sum(g_ok) / len(probs):.3f}   ({time.time() - t0:.0f}s)')

samples = [[] for _ in probs]
rep = [p for p in prompts for _ in range(N_SAMPLES)]
outs = generate(tok, model, rep, max_new_tokens=384, batch=40, do_sample=True, temperature=0.8, top_p=0.95, seed=0)
for i, o in enumerate(outs):
    samples[i // N_SAMPLES].append(truncate(o))
counts = []
for p, ss in zip(probs, samples):
    counts.append(sum(run_tests(p, s) for s in ss))
log(f'sampled: {N_SAMPLES} completions per problem, {sum(counts)} of {N_SAMPLES * len(probs)} pass   ({time.time() - t0:.0f}s)')
hist = {c: counts.count(c) for c in range(N_SAMPLES + 1)}
log('problems by number of passing samples c: ' + ', '.join(f'c={c}: {v}' for c, v in hist.items()))
res = {'greedy_pass': g_ok, 'counts': counts, 'n': N_SAMPLES, 'passk': {}, 'hist': hist}
for k in [1, 2, 5, 10]:
    est = float(np.mean([pass_at_k(N_SAMPLES, c, k) for c in counts]))
    naive = float(np.mean([1 - (1 - c / N_SAMPLES) ** k for c in counts]))
    res['passk'][k] = dict(unbiased=est, naive=naive)
    log(f'pass@{k:<2d} = {est:.3f}   (the biased shortcut 1-(1-c/n)^k would say {naive:.3f})')
i = next(j for j, c in enumerate(counts) if 0 < c < N_SAMPLES)
res['example'] = dict(task=probs[i]['task_id'], prompt=probs[i]['prompt'], c=counts[i],
                      passing=[s for s in samples[i] if run_tests(probs[i], s)][:1],
                      failing=[s for s in samples[i] if not run_tests(probs[i], s)][:1])
log(f'\nexample {probs[i]["task_id"]}: {counts[i]} of {N_SAMPLES} samples pass')
save('ch3_humaneval', res)
