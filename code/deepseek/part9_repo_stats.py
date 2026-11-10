"""Part 9: read-only statistics of the DeepInfer source tree (no code from it is imported or run).

    ~/.venvs/attn/bin/python -I part9_repo_stats.py /path/to/DeepInfer

It counts non-blank lines per component and per language, counts the files that carry the
"Derived from vLLM" header, and pulls the default value of every DEEPINFER_* knob out of
deepinfer/environ.py with a regular expression. Output: results/part9_repo_stats.json and _stdout.txt.
"""
import os, re, subprocess, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import Log, save

root = os.path.abspath(sys.argv[1])
log = Log('part9_repo_stats')
commit = subprocess.run(['git', '-C', root, 'log', '-1', '--format=%H %cI'], capture_output=True, text=True).stdout.strip()
log(f'DeepInfer commit: {commit}')

LANG = {'.py': 'Python', '.cpp': 'C++', '.hpp': 'C++', '.h': 'C++', '.cu': 'CUDA', '.sh': 'shell'}


def component(rel):
    p = rel.split('/')
    if p[0] == 'csrc':
        return 'csrc/engine (C++ scheduler, block manager, input prep)' if len(p) > 1 and p[1] == 'engine' else 'csrc (other C++)'
    if p[0] == 'deepinfer':
        if 'tilelang_ops' in p:
            return 'deepinfer/.../tilelang_ops (TileLang kernels)'
        if p[1] == 'model_executor':
            if len(p) > 2 and p[2] == 'models':
                return 'deepinfer/model_executor/models (V4.1 model)'
            if len(p) > 2 and p[2] == 'eplb':
                return 'deepinfer/model_executor/eplb'
            return 'deepinfer/model_executor (other)'
        if p[1] in ('engine', 'entrypoints'):
            return 'deepinfer/engine + entrypoints'
        if p[1] == 'worker':
            return 'deepinfer/worker (cache, CUDA graphs, runner)'
        return 'deepinfer (other Python)'
    return p[0]


by_comp = defaultdict(lambda: defaultdict(int))
by_lang = defaultdict(int)
derived = []
nfiles = 0
for dp, dns, fns in os.walk(root):
    dns[:] = [d for d in dns if d not in ('.git', 'vendor')]
    for fn in fns:
        ext = os.path.splitext(fn)[1]
        if ext not in LANG:
            continue
        path = os.path.join(dp, fn)
        rel = os.path.relpath(path, root)
        text = open(path, encoding='utf-8', errors='replace').read()
        n = sum(1 for line in text.splitlines() if line.strip())
        nfiles += 1
        by_comp[component(rel)][LANG[ext]] += n
        by_lang[LANG[ext]] += n
        if 'Derived from vLLM' in text[:400]:
            derived.append(rel)

total = sum(by_lang.values())
log(f'\n{nfiles} source files, {total:,} non-blank lines (vendor/ submodules excluded)')
log('\nBy language:')
for k, v in sorted(by_lang.items(), key=lambda kv: -kv[1]):
    log(f'  {k:8s} {v:7,d}  ({100 * v / total:4.1f}%)')
log('\nBy component:')
rows = sorted(by_comp.items(), key=lambda kv: -sum(kv[1].values()))
for k, d in rows:
    log(f'  {sum(d.values()):7,d}  {k}')
log(f'\nFiles with the "Derived from vLLM; modified by DeepSeek." header: {len(derived)}')
for r in sorted(derived):
    log(f'  {r}')

env_src = open(os.path.join(root, 'deepinfer/environ.py')).read()
knobs = re.findall(r'^\s+(DEEPINFER_\w+)\s*=\s*EnvVar(?:\.(\w+)\((?:default=)?([^)]*)\)|\((.*)\))', env_src, re.M)
env = {}
for name, kind, default, raw in knobs:
    env[name] = (kind or 'custom') + (f' default={default}' if default else (' default=False/0' if kind in ('bool', 'int', 'float') else ' (computed)'))
log(f'\n{len(env)} DEEPINFER_* environment knobs in deepinfer/environ.py; selected defaults:')
for k in ['DEEPINFER_USE_NEXTN_DECODE', 'DEEPINFER_ENABLE_MTP_VARLEN', 'DEEPINFER_ZERO_OVERHEAD_SCHEDULING',
          'DEEPINFER_PREFILL_ATTEMPT_INTERVAL', 'DEEPINFER_PREFILL_ANY_ATTEMPT_INTERVAL', 'DEEPINFER_EXPERT_COUNT_WINDOW_SIZE',
          'DEEPINFER_EXPERT_REARRANGEMENT_STEP_INTERVAL', 'DEEPINFER_CUDAGRAPH_BATCH_SIZE_INTERVAL', 'DEEPINFER_NUM_STATE_SLOTS_RATIO',
          'DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE', 'DEEPINFER_BATCH_INVARIANT', 'DEEPINFER_NCCL_NUM_SMS']:
    log(f'  {k:46s} {env.get(k, "?")}')

save('part9_repo_stats', dict(commit=commit, files=nfiles, total_lines=total, by_language=dict(by_lang),
                              by_component={k: dict(d) for k, d in rows}, derived_from_vllm=sorted(derived), env=env))
