"""Part 2 (the method) of the Meta-Harness series, as code that runs on a laptop.
  1. Table 1 in numbers: how much more context per step Meta-Harness gives its proposer.
  2. The objective H* = argmax_H E_{x~X, tau~p_M(H,x)} r(tau, x): exact sum (Method 1) vs Monte Carlo (Method 2),
     then a real estimate with Qwen2.5-0.5B-Instruct, sampled and greedy.
  3. Pareto dominance: brute force (Method 1) vs sort-and-sweep (Method 2), on Table 2's numbers.
  4. Algorithm 1 as runnable Python: a filesystem D with one folder per harness (code, scores, traces),
     a stub proposer that reads D and writes new harness files, interface validation, the Pareto frontier,
     and a final test-set evaluation that the proposer never sees.
  5. The filesystem as memory: list it with os.walk (Method 1) and pathlib (Method 2); grep it in Python
     (Method 1) and with the grep program (Method 2); count its tokens with a rule of thumb and the real tokenizer.
The task model M is Qwen/Qwen2.5-0.5B-Instruct (frozen, local).  The proposer is a stub, not a language model."""
import importlib.util, json, os, random, re, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np
sys.dont_write_bytecode = True                         # no __pycache__ folders inside D
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import Log, save, HERE
from mh_part2_harnesses import SEEDS, EDITS

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_DATASETS_OFFLINE', '1')
from datasets import load_dataset

log = Log('part2')
R = {}
D_ROOT = Path(HERE) / 'results' / 'part2_fs'           # the filesystem D of our run (the proposer may read it)
TEST_OUT = Path(HERE) / 'results' / 'part2_test.json'  # test results live outside D: the proposer never sees them

# ---------------------------------------------------------------------------------------------------- 1. Table 1
log('== 1. Table 1: context per evaluation (MTok/iter, the paper\'s estimates)')
TABLE1 = {'OPRO': 0.002, 'TextGrad': 0.015, 'AlphaEvolve': 0.022, 'GEPA': 0.008,
          'Feedback Descent': 0.012, 'TTT-Discover': 0.026, 'Meta-Harness': 10.0}
largest_prior = max(v for k, v in TABLE1.items() if k != 'Meta-Harness')
for k, v in TABLE1.items():
    log(f'  {k:17s} {v:>7.3f} MTok = {v * 1e6:>12,.0f} tokens   x{v / largest_prior:8.1f} of the largest prior')
log(f'  Meta-Harness / largest prior (TTT-Discover): {TABLE1["Meta-Harness"] / largest_prior:.0f}x')
log(f'  log10 of that ratio: {np.log10(TABLE1["Meta-Harness"] / largest_prior):.2f} orders of magnitude')
R['table1_ratio'] = TABLE1['Meta-Harness'] / largest_prior

# ---------------------------------------------------------------------------------------------------- 2. objective
log('')
log('== 2. The objective: E over tasks x and rollouts tau of r(tau, x)')
# A toy world: two tasks, equally likely; the model is right with these probabilities under each harness.
P_TASK = {'x1': 0.5, 'x2': 0.5}
P_RIGHT = {'H_A': {'x1': 0.9, 'x2': 0.5}, 'H_B': {'x1': 0.7, 'x2': 0.8}}


def expected_reward_exact(h):
    """Method 1: the definition. Sum over tasks of p(x) times the sum over rollouts of p(tau) r(tau, x)."""
    total = 0.0
    for x, px in P_TASK.items():
        p = P_RIGHT[h][x]
        total += px * (p * 1.0 + (1 - p) * 0.0)        # two rollouts: right (reward 1) or wrong (reward 0)
    return total


def expected_reward_sampled(h, n=100_000, seed=0):
    """Method 2: Monte Carlo. Draw a task, draw a rollout, average the rewards (what evaluation really does)."""
    rng = np.random.default_rng(seed)
    xs = rng.choice(list(P_TASK), size=n, p=list(P_TASK.values()))           # x ~ X
    p = np.array([P_RIGHT[h][x] for x in xs])
    r = (rng.random(n) < p).astype(float)                                      # tau ~ p_M(H, x), r(tau, x) in {0, 1}
    return r.mean(), r.std() / np.sqrt(n)


for h in P_RIGHT:
    ex = expected_reward_exact(h)
    mc, se = expected_reward_sampled(h)
    log(f'  {h}: exact {ex:.4f}   Monte Carlo {mc:.4f} (+/- {se:.4f})   difference {abs(ex - mc):.4f}')
best = max(P_RIGHT, key=expected_reward_exact)
log(f'  argmax over the two harnesses: {best}')
R['toy'] = {h: expected_reward_exact(h) for h in P_RIGHT}

# ---------------------------------------------------------------------------------------------------- the model M
DEV = 'mps' if torch.backends.mps.is_available() else 'cpu'
NAME = 'Qwen/Qwen2.5-0.5B-Instruct'
tok = AutoTokenizer.from_pretrained(NAME)
lm = AutoModelForCausalLM.from_pretrained(NAME, dtype=torch.float16 if DEV == 'mps' else torch.float32).to(DEV).eval()
CALLS = {'n': 0}


def make_model(temperature=0.0, seed=0):
    """The frozen model M as a function prompt -> text. temperature 0 = greedy (the same answer every time)."""
    rng = random.Random(seed)

    def model(prompt):
        CALLS['n'] += 1
        msgs = [{'role': 'user', 'content': prompt}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors='pt', return_dict=True)['input_ids'].to(DEV)
        kw = dict(do_sample=False) if temperature == 0 else dict(do_sample=True, temperature=temperature, top_p=1.0)
        if temperature:
            torch.manual_seed(rng.randrange(1 << 30))
        with torch.no_grad():
            out = lm.generate(ids, max_new_tokens=20, pad_token_id=tok.eos_token_id, **kw)
        return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
    return model


# ---------------------------------------------------------------------------------------------------- the data
ds = load_dataset('nyu-mll/glue', 'sst2')
LAB = ['negative', 'positive']
val = [(r['sentence'].strip(), LAB[r['label']]) for r in ds['validation']]
rng = random.Random(0)
idx = list(range(len(val)))
rng.shuffle(idx)
pool = [val[i] for i in idx[:96]]                      # candidates for the search set
test = [val[i] for i in idx[96:144]]                   # 48 test reviews, never used during search
train = [(r['sentence'].strip(), LAB[r['label']]) for r in ds['train'].select(range(2000))]
random.Random(1).shuffle(train)
memory = train[:64]                                    # labelled examples a harness may show the model


def load_harness(path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def evaluate(path, tasks, model):
    """Run one harness on a list of (x, gold) tasks. Return the scores and one trace record per task."""
    h = load_harness(path)
    traces, right, chars = [], 0, 0
    for i, (x, gold) in enumerate(tasks):
        pred, steps = h.run(x, model, memory)
        right += pred == gold
        chars += sum(len(s.get('prompt', '')) for s in steps)
        traces.append(dict(id=i, x=x, gold=gold, pred=pred, correct=pred == gold, steps=steps))
    return dict(accuracy=right / len(tasks), avg_prompt_chars=chars / len(tasks), n=len(tasks)), traces


# The search set (Appendix D): hard for the baseline. Run zero-shot on the pool; keep its misses plus some hits.
log('')
log('== search set: 96 candidate reviews, zero-shot greedy')
tmp = D_ROOT.parent / 'part2_tmp'
tmp.mkdir(parents=True, exist_ok=True)
(tmp / 'zero_shot.py').write_text(SEEDS['zero_shot'])
greedy = make_model(0.0)
t = time.time()
s0, tr0 = evaluate(tmp / 'zero_shot.py', pool, greedy)
miss = [pool[r['id']] for r in tr0 if not r['correct']]
hit = [pool[r['id']] for r in tr0 if r['correct']]
search = miss[:12] + hit[:12]
shutil.rmtree(tmp)
log(f'  zero-shot on the pool: {s0["accuracy"]:.3f} ({len(miss)} misses, {sum(r["pred"] is None for r in tr0)} unclear answers), {time.time() - t:.0f} s')
log(f'  search set: {len(miss[:12])} misses + {len(hit[:12])} hits = {len(search)} reviews;  test set: {len(test)} other reviews')
R['pool'] = dict(acc=s0['accuracy'], misses=len(miss), search=len(search), test=len(test))

# The objective with the real model: sample tau several times per task, then average.
log('')
log('== 2b. E[r] with Qwen2.5-0.5B-Instruct on the 24 search reviews')
est = {}
for name in ('zero_shot', 'few_shot_4'):
    p = D_ROOT.parent / f'part2_{name}.py'
    p.write_text(SEEDS[name])
    runs = [evaluate(p, search, make_model(1.0, seed=s))[0]['accuracy'] for s in range(3)]
    g = evaluate(p, search, greedy)[0]['accuracy']
    p.unlink()
    est[name] = float(np.mean(runs))
    log(f'  {name:11s} sampled (T=1.0) runs {[round(v, 3) for v in runs]} -> mean {np.mean(runs):.3f};  greedy {g:.3f}')
log(f'  argmax of the sampled estimate: {max(est, key=est.get)}')
R['objective_real'] = est

# ---------------------------------------------------------------------------------------------------- 3. Pareto
log('')
log('== 3. Pareto dominance: higher accuracy and lower context are both better')
TABLE2 = {'Zero-Shot': (27.4, 0.0), 'Few-Shot (8)': (34.3, 2.0), 'Few-Shot (32)': (35.4, 7.9),
          'Few-Shot (all)': (40.8, 12.3), 'MCE': (40.0, 28.5), 'ACE': (40.9, 50.8), 'Meta-Harness': (48.6, 11.4)}


def dominates(a, b):
    """a = (accuracy, context). a dominates b if it is no worse on both and strictly better on one."""
    return a[0] >= b[0] and a[1] <= b[1] and (a[0] > b[0] or a[1] < b[1])


def pareto_brute(points):
    """Method 1: keep every point that no other point dominates (compare all pairs)."""
    return sorted(k for k, p in points.items() if not any(dominates(q, p) for j, q in points.items() if j != k))


def pareto_sweep(points):
    """Method 2: sort by context (then by accuracy, high first); keep a point if it beats the best accuracy so far."""
    out, best = [], -float('inf')
    for k, (acc, ctx) in sorted(points.items(), key=lambda kv: (kv[1][1], -kv[1][0])):
        if acc > best:
            out.append(k)
            best = acc
    return sorted(out)


f1, f2 = pareto_brute(TABLE2), pareto_sweep(TABLE2)
log(f'  brute force: {f1}')
log(f'  sweep      : {f2}')
log(f'  same frontier: {f1 == f2}')
for k, p in TABLE2.items():
    if k not in f1:
        log(f'  {k} is dominated by {[j for j, q in TABLE2.items() if dominates(q, p)]}')
R['pareto_table2'] = f1

# ---------------------------------------------------------------------------------------------------- 4. Algorithm 1
log('')
log('== 4. Algorithm 1: the outer loop')
READS = []                                              # every file the proposer opens: (iteration, kind, path)


def read(path, it):
    """The proposer's `cat`: read a file of D and record the access (as the paper's Table 8 does)."""
    p = Path(path)
    kind = ('source' if p.name == 'harness.py' else 'trace' if p.name == 'traces.jsonl'
            else 'score' if p.name == 'scores.json' else 'other')
    READS.append((it, kind, str(p.relative_to(D_ROOT))))
    return p.read_text()


def validate(path):
    """Interface validation (line 11): the file imports, has run(), and run() answers one tiny task."""
    try:
        h = load_harness(path)
        label, trace = h.run('a fine film .', lambda prompt: 'positive', memory[:4])
        return label in ('positive', 'negative', None) and isinstance(trace, list)
    except Exception as e:                              # a syntax error, a missing run(), a crash...
        return False


def store(name, source, proposal, model):
    """D <- D + {(H, Evaluate(H, M, X))}: one new folder with the code, the scores and the traces."""
    d = D_ROOT / f'{len([p for p in D_ROOT.iterdir() if p.is_dir()]):03d}_{name}'
    d.mkdir()
    (d / 'harness.py').write_text(source)
    (d / 'proposal.md').write_text(proposal)
    scores, traces = evaluate(d / 'harness.py', search, model)
    (d / 'scores.json').write_text(json.dumps(scores, indent=1))
    (d / 'traces.jsonl').write_text(''.join(json.dumps(t) + '\n' for t in traces))
    log(f'  stored {d.name:24s} search acc {scores["accuracy"]:.3f}   prompt chars {scores["avg_prompt_chars"]:7.1f}')
    return d


def stub_proposer(it, k=2):
    """Stand-in for the paper's coding agent. It reads D like an agent would (scores, code, traces), writes a
    short diagnosis, and returns k new harness files. Unlike the real proposer, it picks the edits from a fixed list."""
    folders = sorted(p for p in D_ROOT.iterdir() if p.is_dir())
    scores = {f.name: json.loads(read(f / 'scores.json', it)) for f in folders}       # every score
    ranked = sorted(scores, key=lambda n: -scores[n]['accuracy'])
    for n in ranked[:2]:
        read(D_ROOT / n / 'harness.py', it)                                             # the code of the top two
    for f in folders:
        read(f / 'proposal.md', it)                                                     # what earlier steps tried
    best = ranked[0]
    traces = [json.loads(l) for l in read(D_ROOT / best / 'traces.jsonl', it).splitlines()]
    unclear = sum(t['pred'] is None for t in traces)
    wrong = sum(t['pred'] is not None and not t['correct'] for t in traces)
    tried = {f.name.split('_', 1)[1] for f in folders}
    rejected = D_ROOT / 'rejected.jsonl'
    if rejected.exists():                                                               # candidates that failed line 11
        tried |= {json.loads(l)['name'] for l in read(rejected, it).splitlines()}
    new = [(n, why, src) for n, why, src in EDITS if n not in tried][:k]
    out = []
    for n, why, src in new:
        note = (f'# iteration {it}: {n}\nRead {len(folders)} score files. Best so far: {best} '
                f'(search acc {scores[best]["accuracy"]:.3f}). In its traces: {unclear} unclear answers, '
                f'{wrong} wrong labels out of {len(traces)}.\nEdit: answers "{why}".\n')
        out.append((n, src, note))
    return out


if D_ROOT.exists():
    shutil.rmtree(D_ROOT)
D_ROOT.mkdir(parents=True)                                                  # line 3: D <- empty
CALLS['n'] = 0
t = time.time()
log('-- lines 4-6: evaluate the initial population')
for name, src in SEEDS.items():
    store(name, src, f'# seed: {name}\nA baseline harness from the initial population.\n', greedy)
N = 2
for it in range(1, N + 1):                                                  # line 7
    log(f'-- iteration {it}: the proposer reads D and proposes')
    for name, src, note in stub_proposer(it):                               # lines 8-9
        cand = D_ROOT.parent / 'part2_candidate.py'
        cand.write_text(src)
        ok = validate(cand)                                                 # line 11
        cand.unlink()
        if ok:
            store(name, src, note, greedy)                                  # line 12
        else:
            log(f'  rejected {name}: failed interface validation (not evaluated; one line in rejected.jsonl)')
            with open(D_ROOT / 'rejected.jsonl', 'a') as fh:
                fh.write(json.dumps(dict(iteration=it, name=name, reason='interface validation failed')) + '\n')
log(f'  search done: {CALLS["n"]} model calls, {time.time() - t:.0f} s')

# line 13: the Pareto frontier of D (search accuracy up, prompt characters down)
pts = {f.name: (json.loads((f / 'scores.json').read_text())['accuracy'],
                json.loads((f / 'scores.json').read_text())['avg_prompt_chars']) for f in sorted(D_ROOT.iterdir()) if f.is_dir()}
front = pareto_sweep(pts)
assert front == pareto_brute(pts)
log(f'-- line 13: Pareto frontier of D: {front}')
(D_ROOT / 'frontier.json').write_text(json.dumps(front, indent=1))

# After the search: the test set, once, on the frontier only. Written outside D.
log('')
log('== after the search: the held-out test set (48 reviews), frontier only')
T = {}
for n in front:
    s, _ = evaluate(D_ROOT / n / 'harness.py', test, greedy)
    T[n] = dict(search=pts[n][0], test=s['accuracy'], prompt_chars=s['avg_prompt_chars'])
    log(f'  {n:24s} search {pts[n][0]:.3f}   test {s["accuracy"]:.3f}   gap {pts[n][0] - s["accuracy"]:+.3f}')
TEST_OUT.write_text(json.dumps(T, indent=1))
R['search_points'] = pts
R['test'] = T

# ---------------------------------------------------------------------------------------------------- Table 8 style
log('')
log('== the proposer\'s file reads (Table 8 style; a scripted proposer, so only a picture of the bookkeeping)')
for it in range(1, N + 1):
    log(f'  iteration {it}: {sum(1 for r in READS if r[0] == it)} files read')
kinds = [r[1] for r in READS]
for kind in ('source', 'trace', 'score', 'other'):
    log(f'  {kind:6s} {kinds.count(kind):3d}  {100 * kinds.count(kind) / len(kinds):5.1f}%')
R['reads'] = {k: kinds.count(k) for k in ('source', 'trace', 'score', 'other')}

# ---------------------------------------------------------------------------------------------------- 5. filesystem
log('')
log('== 5. The filesystem D')


def tree(root):
    """Method 1: walk the folders ourselves with os.walk; return every file path relative to root."""
    out = []
    for folder, dirs, files in os.walk(root):
        dirs.sort()
        for f in sorted(files):
            out.append(os.path.relpath(os.path.join(folder, f), root))
    return sorted(out)


def tree_lib(root):
    """Method 2: the standard library's pathlib does the walking."""
    return sorted(str(p.relative_to(root)) for p in Path(root).rglob('*') if p.is_file())


a, b = tree(D_ROOT), tree_lib(D_ROOT)
for p in a:
    log(f'  {p:40s} {os.path.getsize(D_ROOT / p):>7,} bytes')
log(f'  os.walk: {len(a)} files, pathlib: {len(b)} files, same list: {a == b}')


def grep(root, pattern):
    """Method 1: grep -rl in Python. Return the files that contain the pattern."""
    rx = re.compile(pattern)
    return sorted(p for p in tree(root) if rx.search((Path(root) / p).read_text()))


def grep_tool(root, pattern):
    """Method 2: the real grep program, the tool the paper's proposer calls."""
    out = subprocess.run(['grep', '-rlE', pattern, '.'], cwd=root, capture_output=True, text=True).stdout
    return sorted(l[2:] for l in out.split())


pat = '"pred": null'
g1, g2 = grep(D_ROOT, pat), grep_tool(D_ROOT, pat)
log(f'  files with an unclear answer ({pat}): {g1}')
log(f'  Python grep and grep -rlE agree: {g1 == g2}')

# How big is D, in tokens?  Method 1: the rule of thumb (4 characters per token). Method 2: the real tokenizer.
text = ''.join((D_ROOT / p).read_text() for p in a)
rough = len(text) / 4
real = len(tok(text)['input_ids'])
log(f'  D holds {len(text):,} characters: about {rough:,.0f} tokens by the rule of thumb, {real:,} by the Qwen tokenizer')
log(f'  Table 1 says up to 10,000,000 tokens per evaluation in the paper: {10_000_000 / real:,.0f}x our whole D')
R['fs'] = dict(files=len(a), chars=len(text), tokens=real)
save('part2', R)
