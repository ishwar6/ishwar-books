"""Part 7: the evaluation setup of §3 and the limits of §5, tried on the released DeepSeek-R1-Distill-Qwen-1.5B.

Phase A (heavy): greedy decoding vs sampling (T = 0.6, top-p = 0.95, 4 answers) on 8 MATH-500 problems.
Phase B (heavy): the same sampling with a few-shot prompt (two worked examples) instead of the zero-shot prompt.
Phase C (heavy): one forward pass over every zero-shot sampled answer; at each step, count the tokens in the
                 top-p = 0.95 nucleus (the branching factor a tree search over tokens would face).
Then (light): grade with math-verify, measure repetition and the token limit, pass@1, pass@k, cons@4, and the
size of a search tree over tokens against Go, two ways.

Limits: max_new_tokens = 2048 (the paper allows 32,768), 8 problems, 4 samples, one Mac (mps, bfloat16).
Raw outputs are saved in results/r1_part7_gen_raw.json, so a re-run only re-does the analysis.
"""
import json, math, os, sys
import numpy as np
from common import Log, save, heavy, RESULTS

MODEL = 'deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B'
N_PROBLEMS, N_SAMPLES, MAX_NEW = 8, 4, 2048
TEMP, TOP_P = 0.6, 0.95
RAW = os.path.join(RESULTS, 'r1_part7_gen_raw.json')
SUFFIX = 'Please reason step by step, and put your final answer within \\boxed{}.'   # the model card's advice
FEW_SHOT = ('Here are two solved examples.\n\n'
            'Problem: What is the sum of the first 10 positive integers?\n'
            'Solution: The sum is 10 * 11 / 2 = 55. The answer is \\boxed{55}.\n\n'
            'Problem: Solve for x: 2x + 3 = 11.\n'
            'Solution: Subtract 3 from both sides: 2x = 8. Divide by 2: x = 4. The answer is \\boxed{4}.\n\n'
            'Now solve this problem in the same way.\n\nProblem: ')

log = Log('r1_part7_gen')
raw = json.load(open(RAW)) if os.path.exists(RAW) else {}


def dump():
    json.dump(raw, open(RAW, 'w'))


def problems():
    from datasets import load_dataset
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test')
    keep = [r for r in ds if '[asy]' not in r['problem']][:N_PROBLEMS]   # the first 8 without a drawing
    return [dict(id=r['unique_id'], level=r['level'], problem=r['problem'], answer=r['answer']) for r in keep]


def load():
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev).eval()
    return tok, model, dev


def prompt_ids(tok, text, dev):
    import torch
    ids = tok.apply_chat_template([{'role': 'user', 'content': text}], add_generation_prompt=True, return_tensors='pt')
    ids = ids if torch.is_tensor(ids) else ids['input_ids']
    return ids.to(dev)                                            # (1, prompt_len); ends with "<think>\n"


def run(tok, model, dev, text, sample, seed):
    """Generate 1 greedy answer or N_SAMPLES sampled answers. Returns a list of dicts (token ids, finished?)."""
    import torch, time
    torch.manual_seed(seed)
    ids = prompt_ids(tok, text, dev)
    n = N_SAMPLES if sample else 1
    kw = dict(do_sample=True, temperature=TEMP, top_p=TOP_P) if sample else dict(do_sample=False, temperature=None, top_p=None)
    t = time.time()
    with torch.no_grad():
        out = model.generate(ids.repeat(n, 1), attention_mask=torch.ones_like(ids).repeat(n, 1),
                             max_new_tokens=MAX_NEW, pad_token_id=tok.eos_token_id, **kw)
    gen = out[:, ids.shape[1]:].tolist()                           # (n, <= MAX_NEW)
    res = []
    for g in gen:
        done = tok.eos_token_id in g
        g = g[:g.index(tok.eos_token_id)] if done else g           # cut at the end-of-sentence token
        res.append(dict(ids=g, finished=done))
    return res, time.time() - t, ids.shape[1]


def phase(name, build_text, sample, tok, model, dev):
    if name in raw:
        return
    P = problems()
    if True:
        rows = []
        for i, p in enumerate(P):
            res, dt, plen = run(tok, model, dev, build_text(p), sample, seed=i)
            for r in res:
                r['text'] = tok.decode(r['ids'])
            rows.append(dict(problem=p, prompt_len=plen, outputs=res, seconds=dt))
            print(f'[{name}] problem {i + 1}/{len(P)}: {[len(r["ids"]) for r in res]} tokens, {dt:.0f} s', file=sys.stderr)
        raw[name] = dict(rows=rows, device=dev)
        dump()


def nucleus_phase(tok, model, dev):
    """Phase C: for every zero-shot sampled answer, the size of the top-p nucleus at each step."""
    if 'nucleus' in raw:
        return
    import torch
    if True:
        sizes, sizes_t1, ent = [], [], []
        per_answer = []
        for row in raw['sample']['rows']:
            ids = prompt_ids(tok, row['problem']['problem'] + ' ' + SUFFIX, dev)
            for o in row['outputs']:
                full = torch.cat([ids, torch.tensor([o['ids']], device=dev)], 1)       # (1, prompt + answer)
                with torch.no_grad():
                    h = model.model(full).last_hidden_state[0, ids.shape[1] - 1:-1]     # states that predict the answer
                s_a, s1_a = [], []
                for c in range(0, h.shape[0], 256):                                    # 256 steps at a time
                    logits = model.lm_head(h[c:c + 256]).float()                        # (<=256, vocab)
                    for T, store in ((TEMP, s_a), (1.0, s1_a)):
                        pr = torch.softmax(logits / T, -1)
                        srt = pr.sort(-1, descending=True).values.cumsum(-1)
                        store += ((srt < TOP_P).sum(-1) + 1).clamp(max=pr.shape[-1]).tolist()   # tokens to reach 0.95
                    p6 = torch.softmax(logits / TEMP, -1)
                    ent += (-(p6 * torch.log(p6.clamp_min(1e-30))).sum(-1)).tolist()
                sizes += s_a; sizes_t1 += s1_a
                per_answer.append(dict(problem=row['problem']['id'], n=len(s_a), sizes=s_a))
        raw['nucleus'] = dict(sizes=sizes, sizes_t1=sizes_t1, entropy=ent, per_answer=per_answer,
                              vocab=len(tok), vocab_cfg=model.config.vocab_size)
        dump()


if not all(k in raw for k in ('greedy', 'sample', 'fewshot', 'nucleus')):
    with heavy('part7'):                       # one lock for all model work (about 30 minutes)
        M = load()
        phase('greedy', lambda p: p['problem'] + ' ' + SUFFIX, False, *M)
        phase('sample', lambda p: p['problem'] + ' ' + SUFFIX, True, *M)
        phase('fewshot', lambda p: FEW_SHOT + p['problem'] + '\n\n' + SUFFIX, True, *M)
        nucleus_phase(*M)
        del M

# ====================================================================== analysis (no model)
from math_verify import parse, verify


def pass_at_k_product(n, c, k):
    """The unbiased pass@k estimator, product form (same function as in r1_part7_passk.py)."""
    if n - c < k:
        return 1.0
    return 1 - math.prod(1 - k / i for i in range(n - c + 1, n + 1))


log(f'model {MODEL}, device {raw["greedy"]["device"]}, bfloat16')
log(f'{N_PROBLEMS} MATH-500 problems (the first {N_PROBLEMS} without a drawing), max_new_tokens {MAX_NEW}')
log(f'greedy: 1 answer each; sampling: {N_SAMPLES} answers each at T={TEMP}, top-p={TOP_P}')


def answer_part(text):
    """The text after the last </think>; None if the model never closed its thinking (cut by the limit)."""
    return text.split('</think>')[-1] if '</think>' in text else None


def grade(text, gold):
    a = answer_part(text)
    if a is None:
        return False, None
    pred = parse(a)
    return (bool(pred) and verify(parse('$' + gold + '$'), pred)), (pred[-1] if pred else None)


def rep_rate(ids, n=4):
    """Repeated n-gram rate: share of the n-token windows that already appeared earlier in the same answer."""
    grams = [tuple(ids[i:i + n]) for i in range(len(ids) - n + 1)]
    return 1 - len(set(grams)) / len(grams) if grams else 0.0


def loop_period(ids, tail=512, need=0.95):
    """Smallest p <= 256 such that the last `tail` tokens repeat with period p (95% of positions agree)."""
    t = ids[-tail:]
    if len(t) < tail:
        return None
    for p in range(1, 257):
        same = sum(t[i] == t[i - p] for i in range(p, len(t))) / (len(t) - p)
        if same >= need:
            return p
    return None


summary = {}
for mode in ('greedy', 'sample', 'fewshot'):
    log()
    log(f'== {mode}')
    log('  problem              lvl  gold            tokens  hit-limit  rep4   tail-rep4  loop  correct')
    tot, rows_out = [], []
    for row in raw[mode]['rows']:
        p = row['problem']
        outs = []
        for o in row['outputs']:
            ok, pred = grade(o['text'], p['answer'])
            per = loop_period(o['ids'])
            d = dict(tokens=len(o['ids']), hit=not o['finished'], rep4=rep_rate(o['ids']),
                     tail=rep_rate(o['ids'][-512:]), loop=per, ok=ok, pred=str(pred) if pred is not None else None)
            outs.append(d); tot.append(d)
        rows_out.append(dict(id=p['id'], level=p['level'], outs=outs))
        g = p['answer'] if len(p['answer']) < 15 else p['answer'][:12] + '...'
        log(f'  {p["id"][:20]:20s} {p["level"]:3d}  {g:15s} '
            + ' '.join(f'{o["tokens"]:5d}' for o in outs[:1]) + f'{"":4s}'
            + f'{sum(o["hit"] for o in outs)}/{len(outs)}{"":6s}'
            + f'{np.mean([o["rep4"] for o in outs]):.2f}   {np.mean([o["tail"] for o in outs]):.2f}       '
            + f'{sum(o["loop"] is not None for o in outs)}/{len(outs)}   {sum(o["ok"] for o in outs)}/{len(outs)}')
    n = len(tot)
    acc = sum(o['ok'] for o in tot) / n
    s = dict(answers=n, accuracy=acc, hit=sum(o['hit'] for o in tot), loops=sum(o['loop'] is not None for o in tot),
             rep4=float(np.mean([o['rep4'] for o in tot])), tail=float(np.mean([o['tail'] for o in tot])),
             tokens=float(np.mean([o['tokens'] for o in tot])),
             ok_finished=sum(o['ok'] for o in tot if not o['hit']), finished=sum(not o['hit'] for o in tot),
             rows=rows_out)
    log(f'  total: {n} answers, correct {sum(o["ok"] for o in tot)}/{n} = {acc:.3f}, hit the {MAX_NEW}-token limit '
        f'{s["hit"]}/{n}, loop at the end {s["loops"]}/{n}, mean tokens {s["tokens"]:.0f}')
    log(f'  mean repeated 4-gram rate {s["rep4"]:.3f}, in the last 512 tokens {s["tail"]:.3f}; '
        f'correct among finished answers {s["ok_finished"]}/{s["finished"]}')
    summary[mode] = s

# pass@1, pass@k and cons@4 of the sampled runs, per problem
log()
log('== sampled answers: pass@1, pass@k (unbiased, n = 4) and cons@4 (fraction credit on ties)')
for mode in ('sample', 'fewshot'):
    pk = {k: [] for k in (1, 2, 4)}
    cons = []
    for r in summary[mode]['rows']:
        c = sum(o['ok'] for o in r['outs'])
        for k in pk:
            pk[k].append(pass_at_k_product(N_SAMPLES, c, k))
        # majority vote: group answers that math-verify calls equal, then vote
        groups = []                                      # [representative, votes, is_correct]
        for o in r['outs']:
            if o['pred'] is None:
                continue
            for g in groups:
                if o['pred'] == g[0] or verify(parse('$' + g[0] + '$'), parse('$' + o['pred'] + '$')):
                    g[1] += 1; break
            else:
                groups.append([o['pred'], 1, o['ok']])
        if groups:
            top = max(g[1] for g in groups)
            win = [g for g in groups if g[1] == top]
            cons.append(sum(g[2] for g in win) / len(win))
        else:
            cons.append(0.0)
    summary[mode]['passk'] = {k: float(np.mean(v)) for k, v in pk.items()}
    summary[mode]['cons4'] = float(np.mean(cons))
    log(f'  {mode:8s} pass@1 {np.mean(pk[1]):.3f}  pass@2 {np.mean(pk[2]):.3f}  pass@4 {np.mean(pk[4]):.3f}  '
        f'cons@4 {np.mean(cons):.3f}')
log(f'  greedy   accuracy {summary["greedy"]["accuracy"]:.3f} (one answer per problem)')

# the worst greedy loop, as an excerpt
loops = [(o, r) for r in raw['greedy']['rows'] for o in r['outputs'] if loop_period(o['ids']) is not None]
if loops:
    import transformers  # noqa: F401
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    o, r = loops[0]
    per = loop_period(o['ids'])
    log()
    log(f'== a greedy loop: problem {r["problem"]["id"]}, period {per} tokens; the last {2 * per} tokens:')
    for line in tok.decode(o['ids'][-2 * per:]).strip().split('\n'):
        if line.strip():
            log('  | ' + line.strip()[:110])
    summary['loop_example'] = dict(id=r['problem']['id'], period=per, text=tok.decode(o['ids'][-2 * per:]))
else:
    log()
    log('== no greedy answer ended in a loop under this definition')

# the longest block of tokens that the answer writes twice (a loop in the making)
def longest_repeat(ids):
    """Length and start of the longest token block that occurs at least twice (binary search on the length)."""
    def has(L):
        seen = {}
        for i in range(len(ids) - L + 1):
            key = tuple(ids[i:i + L])
            if key in seen:
                return seen[key], i
            seen[key] = i
        return None
    lo, hi, best = 1, len(ids) // 2, (0, None)
    while lo <= hi:
        mid = (lo + hi) // 2
        hit = has(mid)
        if hit:
            best, lo = (mid, hit), mid + 1
        else:
            hi = mid - 1
    return best


log()
log('== longest block of tokens written twice in one answer')
rep = {}
for mode in ('greedy', 'sample', 'fewshot'):
    lens = [longest_repeat(o['ids']) for r in raw[mode]['rows'] for o in r['outputs']]
    L = [x[0] for x in lens]
    rep[mode] = dict(mean=float(np.mean(L)), max=int(max(L)), over50=sum(x >= 50 for x in L), n=len(L))
    log(f'  {mode:8s} mean {np.mean(L):6.1f} tokens, max {max(L):4d}, answers with a block of 50+ tokens repeated: '
        f'{rep[mode]["over50"]}/{len(L)}')
summary['repeat'] = rep
best = max(((longest_repeat(o['ids']), r, o) for r in raw['greedy']['rows'] for o in r['outputs']), key=lambda t: t[0][0])
(L, (i, j)), r, o = best
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(MODEL)
log(f'  greedy, problem {r["problem"]["id"]}: a block of {L} tokens at token {i} comes again at token {j}:')
for line in tok.decode(o['ids'][i:i + L]).strip().split('\n'):
    if line.strip():
        log('  | ' + line.strip()[:110])
summary['repeat_example'] = dict(id=r['problem']['id'], L=L, first=i, second=j, text=tok.decode(o['ids'][i:i + L]))

# ====================================================================== the size of a token search tree
nu = raw['nucleus']
S, S1, H = np.array(nu['sizes']), np.array(nu['sizes_t1']), np.array(nu['entropy'])
log()
log(f'== branching factor of a search over tokens ({len(S):,} steps of {len(nu["per_answer"])} sampled answers)')
log(f'  vocabulary: {nu["vocab"]:,} tokens in the tokenizer ({nu["vocab_cfg"]:,} rows in the output layer)')
for name, A in ((f'T={TEMP}', S), ('T=1.0', S1)):
    log(f'  top-p {TOP_P} nucleus at {name}: median {np.median(A):.0f}, mean {A.mean():.2f}, 90th pct {np.percentile(A, 90):.0f}, '
        f'99th pct {np.percentile(A, 99):.0f}, max {A.max()}, size 1 at {np.mean(A == 1):.1%} of steps')
log(f'  mean entropy at T={TEMP}: {H.mean():.3f} nats')

# Method 1: add logarithms. Method 2: exact integers (Python big ints), count the digits.
one = max(nu['per_answer'], key=lambda a: a['n'] if a['n'] < MAX_NEW else 0)   # the longest finished answer
L = one['n']
m1 = sum(math.log10(s) for s in one['sizes'])
prod = 1
for s in one['sizes']:
    prod *= s
digits = str(prod)
m2 = len(digits) - 1 + math.log10(int(digits[:15]) / 10 ** (min(15, len(digits)) - 1))
log()
log(f'== how many different answers could top-p sampling write? one answer ({one["problem"]}), {L} tokens')
log(f'  Method 1 (sum of log10 of nucleus sizes):  10^{m1:.1f}')
log(f'  Method 2 (exact product, count the digits): {len(str(prod))} digits = 10^{m2:.1f}')
go = 250 ** 150
chess = 35 ** 80
log(f'  Go, b = 250, d = 150:   250^150 = 10^{150 * math.log10(250):.1f}  (exact: {len(str(go))} digits)')
log(f'  chess, b = 35, d = 80:  35^80 = 10^{80 * math.log10(35):.1f}  (exact: {len(str(chess))} digits)')
log(f'  whole vocabulary, {L} tokens: {nu["vocab"]:,}^{L} = 10^{L * math.log10(nu["vocab"]):.0f}')
log(f'  a tree of depth 150 with our median nucleus at T={TEMP}: {np.median(S):.0f}^150 = 10^{150 * math.log10(max(np.median(S), 1)):.1f}')
summary['tree'] = dict(L=L, m1=m1, m2=m2, go=150 * math.log10(250), chess=80 * math.log10(35),
                       vocab=L * math.log10(nu['vocab']), median=float(np.median(S)), mean=float(S.mean()),
                       p90=float(np.percentile(S, 90)), p99=float(np.percentile(S, 99)), max=int(S.max()),
                       one=float(np.mean(S == 1)), median_t1=float(np.median(S1)), mean_t1=float(S1.mean()),
                       one_t1=float(np.mean(S1 == 1)))
hist = np.bincount(np.minimum(S, 11))[1:]          # sizes 1..10 and "11 or more"
summary['tree']['hist'] = [int(x) for x in hist]
summary['tree']['hist_t1'] = [int(x) for x in np.bincount(np.minimum(S1, 11), minlength=12)[1:]]
save('r1_part7_gen', summary)
log()
log('saved results/r1_part7_gen.json')
