"""Part 4: does retrieval help a small local model solve math problems? (a tiny version of Table 6)

The frozen task model is Qwen2.5-1.5B-Instruct (local, greedy decoding). Four harnesses build the prompt:
  no retrieval | random few-shot (3 corpus examples) | BM25 top-3 | the routed harness of mh_part4.py
The queries are decontaminated MATH test problems (levels 1 to 3), the same number from each of the four routes.
The answer is the last \\boxed{...} of the output, compared with the reference after light normalisation.
Every printed line is saved to results/part4_llm_stdout.txt. N and the number of new tokens are kept small
so the run takes minutes; the paper used 200 IMO-level problems, 3 samples each, and five much larger models."""
import os, random, re, sys, time
from collections import Counter, defaultdict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import Log, save
from mh_part4 import (load_math, bm25_index, bm25_scores, top_k, tokenize_math, retrieve, words, contaminated)

N_PER_ROUTE = int(os.environ.get('N_PER_ROUTE', 6))
MAX_NEW = int(os.environ.get('MAX_NEW', 512))
BATCH = 12
SOL_CHARS = 1200          # each retrieved solution is cut to this many characters (the paper cuts at 3,000)
MODEL = 'Qwen/Qwen2.5-1.5B-Instruct'
SYSTEM = 'Please reason step by step, and put your final answer within \\boxed{}.'


def last_boxed(text):
    """The content of the last \\boxed{...}, with nested braces handled."""
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    depth, j = 0, i + len('\\boxed{')
    for k in range(j, len(text)):
        depth += {'{': 1, '}': -1}.get(text[k], 0)
        if depth < 0:
            return text[j:k]
    return None


def normalize(a):
    if a is None:
        return None
    a = a.replace('\\dfrac', '\\frac').replace('\\tfrac', '\\frac').replace('\\left', '').replace('\\right', '')
    a = re.sub(r'\\text\{([^}]*)\}', r'\1', a)
    a = re.sub(r'\^\{?\\circ\}?|\\!|\\,|\\ |\$|\s', '', a)
    return a.rstrip('.').replace('{,}', '')


def build_prompt(problem, examples):
    """The harness's only job here: decide which solved examples go above the problem."""
    parts = []
    if examples:
        parts.append('Here are some solved problems that may help.\n')
        for ex in examples:
            parts.append(f'Problem: {ex["problem"]}\nSolution: {ex["solution"][:SOL_CHARS]}\n')
        parts.append('Now solve this problem.\n')
    parts.append(f'Problem: {problem}')
    return '\n'.join(parts)


def main():
    log = Log('part4_llm')
    train, test = load_math()
    ix = bm25_index([tokenize_math(r['problem']) for r in train])
    hard_geo = [i for i, r in enumerate(train) if r['type'] == 'Geometry' and r['difficulty'] == 5]
    cw = [words(r['problem']) for r in train]
    cp = {r['problem'][:60].strip().lower() for r in train}
    wp = defaultdict(list)
    for i, ws in enumerate(cw):
        for w in ws:
            wp[w].append(i)
    random.seed(4)
    pool = [q for q in test if 1 <= q["difficulty"] <= 3]
    random.shuffle(pool)
    sample, per = [], Counter()
    for q in pool:
        if per[q['route_true']] < N_PER_ROUTE and not contaminated(q['problem'], cw, wp, cp):
            sample.append(q)
            per[q['route_true']] += 1
    log(f'{len(sample)} decontaminated MATH test problems, levels 1-3: {dict(per)}')

    harnesses = {
        'no retrieval': lambda q: [],
        'random few-shot': lambda q: random.Random(q['problem']).sample(train, 3),
        'BM25 top-3': lambda q: [train[i] for i in top_k(bm25_scores(ix, tokenize_math(q['problem'])), 3)],
        'routed harness': lambda q: [train[i] for i in retrieve(q['problem'], ix, train, hard_geo)[1]],
    }
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    tok = AutoTokenizer.from_pretrained(MODEL, padding_side='left')
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).to(dev).eval()
    log(f'model {MODEL} on {dev}, greedy, max {MAX_NEW} new tokens, solutions cut at {SOL_CHARS} chars')

    R = {}
    for name, pick in harnesses.items():
        t = time.time()
        prompts = [tok.apply_chat_template([{'role': 'system', 'content': SYSTEM},
                                            {'role': 'user', 'content': build_prompt(q['problem'], pick(q))}],
                                           tokenize=False, add_generation_prompt=True) for q in sample]
        outs = []
        for b in range(0, len(prompts), BATCH):
            enc = tok(prompts[b:b + BATCH], return_tensors='pt', padding=True).to(dev)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False)
            outs += tok.batch_decode(gen[:, enc['input_ids'].shape[1]:], skip_special_tokens=True)
        ok = [normalize(last_boxed(o)) == normalize(last_boxed(q['solution'])) for o, q in zip(outs, sample)]
        by_route = defaultdict(list)
        for q, c in zip(sample, ok):
            by_route[q['route_true']].append(c)
        ctx = sum(len(tok(p)['input_ids']) for p in prompts) / len(prompts)
        R[name] = dict(acc=sum(ok) / len(ok), by_route={r: sum(v) / len(v) for r, v in by_route.items()},
                       ctx=ctx, correct=ok)
        log(f'{name:>16}: {sum(ok):2d}/{len(ok)} = {100 * sum(ok) / len(ok):5.1f}%   '
            + '  '.join(f'{r[:6]} {sum(v)}/{len(v)}' for r, v in sorted(by_route.items()))
            + f'   prompt {ctx:5.0f} tokens   {time.time() - t:4.0f} s')
        if name == 'no retrieval':
            log(f'   (no \\boxed answer in {sum(last_boxed(o) is None for o in outs)} outputs)')
    base = R['no retrieval']['correct']
    for name in list(harnesses)[1:]:
        c = R[name]['correct']
        log(f'{name} vs no retrieval: fixes {sum(a and not b for a, b in zip(c, base))}, '
            f'breaks {sum(b and not a for a, b in zip(c, base))}')
    save('part4_llm', R)


if __name__ == '__main__':
    main()
