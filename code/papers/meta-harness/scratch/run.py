"""Run one Meta-Harness search, then score the held-out test set once.

  python run.py --proposer scripted --iters 6 --k 2 --name scripted-1
  python run.py --proposer local-llm --iters 6 --k 2 --name local-llm-1
  python run.py --proposer api ...        (needs MH_API_BASE, MH_API_KEY, MH_API_MODEL; see proposers.py)
  python run.py --proposer agent-cli ...  (needs MH_AGENT_CMD; see proposers.py)

Writes runs/<name>/ (see store.py). --test chooses which candidates get a test score after the search:
'frontier' (the paper's choice: the Pareto frontier, plus the seeds and the best search candidate), or 'all' (every candidate, to measure the search-test gap)."""
import argparse, json, os, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import design_space as ds
import task
from evaluate import evaluate
from llm import LocalLLM, TASK_MODEL
from loop import meta_harness
from proposers import ScriptedProposer, LocalLLMProposer, ChatAPIProposer, AgentCLIProposer


def make_logger(path, also=None):
    lines = []

    def log(*a):
        line = ' '.join(str(x) for x in a)
        (also or print)(line)
        lines.append(line)
        open(path, 'w').write('\n'.join(lines) + '\n')
    return log


def main(argv=None, echo=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--proposer', default='scripted', choices=['scripted', 'local-llm', 'api', 'agent-cli'])
    ap.add_argument('--iters', type=int, default=6)
    ap.add_argument('--k', type=int, default=2)
    ap.add_argument('--name', default=None)
    ap.add_argument('--test', default='all', choices=['frontier', 'all'])
    ap.add_argument('--proposer-model', default='Qwen/Qwen2.5-1.5B-Instruct')
    args = ap.parse_args(argv)
    name = args.name or f'{args.proposer}-{time.strftime("%Y%m%d-%H%M%S")}'
    run_dir = os.path.join(HERE, 'runs', name)
    os.makedirs(run_dir, exist_ok=True)
    log = make_logger(os.path.join(run_dir, 'log.txt'), echo)
    t_start = time.time()

    labels, memory, search, test = task.load()
    llm = LocalLLM()                                              # the frozen task model M
    log(f'run {name}: proposer {args.proposer}, {args.iters} iterations x {args.k} proposals')
    log(f'task model {llm.name} on {llm.device}; {len(labels)} labels; memory {len(memory)}, '
        f'search {len(search)}, test {len(test)} (test is never shown to the proposer)')

    if args.proposer == 'scripted':
        proposer = ScriptedProposer(seed=0)
    elif args.proposer == 'local-llm':
        proposer = LocalLLMProposer(model_name=args.proposer_model, labels=labels, examples=memory)
    elif args.proposer == 'api':
        proposer = ChatAPIProposer(labels=labels, examples=memory)
    else:
        proposer = AgentCLIProposer()
    seeds = [(n, ds.render(n, cfg, about=f'seed, {ds.describe(cfg)}'), f'# Seed\n\nSeed: {ds.describe(cfg)}\n')
             for n, cfg in ds.SEEDS.items()]

    store, frontier, stats = meta_harness(run_dir, seeds, proposer, llm, labels, memory, search,
                                          args.iters, args.k, log)
    search_secs = time.time() - t_start
    front = {c['folder'] for c in frontier}
    log(f'== search done in {search_secs / 60:.1f} min: {stats}')
    log('Pareto frontier on the search set (accuracy up, prompt characters down):')
    for c in sorted(frontier, key=lambda c: c['scores']['ctx_chars']):
        log(f"  {c['folder']:44s} {c['scores']['accuracy']:5.1f}%  {c['scores']['ctx_chars']:5d} chars")

    # ---- after the search: the held-out test set, scored once -------------------------------------
    cands = store.candidates()
    best_search = max(cands, key=lambda c: (c['scores']['accuracy'], -c['scores']['ctx_chars']))['folder']
    to_test = cands if args.test == 'all' else [c for c in cands if c['folder'] in front
                                                 or c['folder'][4:] in ds.SEEDS or c['folder'] == best_search]
    log(f'== test set ({len(test)} messages) for {len(to_test)} candidates')
    summary = []
    for c in cands:
        row = {'folder': c['folder'], 'seed': c['folder'][4:] in ds.SEEDS, 'frontier': c['folder'] in front,
               'search_acc': c['scores']['accuracy'], 'ctx_chars': c['scores']['ctx_chars'],
               'calls': c['scores']['calls_per_message']}
        if c in to_test:
            scores, traces = evaluate(os.path.join(c['dir'], 'harness.py'), llm, labels, memory, test)
            for t in traces:                                     # keep the replies; drop the long prompts (size)
                t['calls'] = [{'prompt_chars': len(x['prompt']), 'reply': x['reply']} for x in t['calls']]
            json.dump({'scores': scores, 'traces': traces}, open(os.path.join(run_dir, 'heldout', c['folder'] + '.json'), 'w'))
            row['test_acc'] = scores['accuracy']
            row['gap'] = round(row['search_acc'] - row['test_acc'], 1)
            log(f"  {c['folder']:44s} search {row['search_acc']:5.1f}%  test {row['test_acc']:5.1f}%  "
                f"gap {row['gap']:+5.1f}{'   (frontier)' if row['frontier'] else ''}")
        summary.append(row)
    best = max(summary, key=lambda r: (r['search_acc'], -r['ctx_chars']))
    log(f"selected by search accuracy: {best['folder']} search {best['search_acc']}% test {best.get('test_acc')}%")
    json.dump(summary, open(os.path.join(run_dir, 'summary.json'), 'w'), indent=1)
    meta = {'name': name, 'proposer': args.proposer, 'iterations': args.iters, 'k': args.k, 'task_model': llm.name,
            'device': llm.device, 'labels': labels, 'sizes': {'memory': len(memory), 'search': len(search),
            'test': len(test)}, 'stats': stats, 'selected': best['folder'],
            'minutes_search': round(search_secs / 60, 1), 'minutes_total': round((time.time() - t_start) / 60, 1),
            'task_model_calls': llm.calls}
    if args.proposer == 'local-llm':
        meta['proposer_model'] = args.proposer_model
        meta['proposer_reads'] = proposer.reads
    json.dump(meta, open(os.path.join(run_dir, 'run.json'), 'w'), indent=1)
    log(f"total {meta['minutes_total']} min, {llm.calls} task-model calls")
    return run_dir


if __name__ == '__main__':
    main()
