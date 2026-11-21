"""Part 4 experiment, step 3 (light, no model): score the answers saved by r1_part4.py.

Every number is computed two ways and the two are checked against each other:
- grading:      Method 1 our own answer extraction and normalisation; Method 2 the math-verify library.
- pass@1:       Method 1 loops (the paper's formula, page 12); Method 2 numpy.
- cons@k:       Method 1 majority vote over our normalised answer strings; Method 2 vote over math-verify classes.
- length:       Method 1 our own bins; Method 2 numpy.histogram.
- reflection:   Method 1 regular expressions; Method 2 word tokens counted with collections.Counter.
Writes results/r1_part4_stdout.txt, results/r1_part4.json and results/r1_part4_passages.txt."""
import json, os, re
from collections import Counter
import numpy as np
from math_verify import parse, verify
import common

log = common.Log('r1_part4')
R = common.RESULTS
PROBLEMS = {p['idx']: p for p in json.load(open(os.path.join(R, 'r1_part4_problems.json')))}
TAGS = ['base', 'distill']
NAMES = {'base': 'Qwen2.5-Math-1.5B (base, R1-Zero template)', 'distill': 'DeepSeek-R1-Distill-Qwen-1.5B'}
REFLECT = {                                   # the four phrases we count; lower case, word boundaries
    'wait': r'\bwait\b',
    'let me check': r'\blet me check\b',
    'alternatively': r'\balternatively\b',
    're-evaluate': r'\bre-?evaluat\w*',
}


def load(name):
    return [json.loads(l) for l in open(os.path.join(R, name))]


# ---------- grading, Method 1: our own extraction and normalisation ----------
def last_boxed(text):
    """The content of the last \\boxed{...}, with nested braces matched by hand. None if there is none."""
    i = text.rfind('\\boxed')
    if i < 0:
        return None
    j = text.find('{', i)
    if j < 0:
        return None
    depth, k = 0, j
    while k < len(text):
        depth += {'{': 1, '}': -1}.get(text[k], 0)
        if depth == 0:
            return text[j + 1:k]
        k += 1
    return None                                   # the box was never closed (a truncated answer)


def final_part(text, tag):
    """The part of a finished answer that we grade (the same policy as Part 6): for the distill, only the text after
    </think>; a distill answer that never closes </think> has no final answer. The base model writes no </think>
    of its own in most answers, so for it we take the text after the last </think> if there is one, else all of it."""
    if '</think>' in text:
        return text.split('</think>')[-1]
    return None if tag == 'distill' else text


def extract(text):
    """The final answer: the last box; else the text inside <answer> </answer> (the R1-Zero template)."""
    if text is None:
        return None
    b = last_boxed(text)
    if b is not None:
        return b
    m = re.findall(r'<answer>(.*?)</answer>', text, flags=re.S)
    return m[-1].strip() if m else None


def normalise(s):
    """Make two ways of writing the same answer look the same: spaces, \\left, \\dfrac, degrees, $ and so on."""
    if s is None:
        return None
    s = s.strip().strip('$').strip()
    s = re.sub(r'\\text\{([^}]*)\}', r'\1', s)
    for a, b in [('\\left', ''), ('\\right', ''), ('\\dfrac', '\\frac'), ('\\tfrac', '\\frac'), ('^\\circ', ''),
                 ('^{\\circ}', ''), ('\\!', ''), ('\\,', ''), ('\\ ', ''), ('°', '')]:
        s = s.replace(a, b)
    s = re.sub(r'\\frac(\d)(\d)', r'\\frac{\1}{\2}', s)   # \frac14 -> \frac{1}{4}
    s = re.sub(r'\s+', '', s).rstrip('.')
    if re.fullmatch(r'-?\d+\.0+', s):
        s = s.split('.')[0]
    return s


def correct_m1(pred, gold):
    return pred is not None and normalise(pred) == normalise(gold)


# ---------- grading, Method 2: math-verify ----------
def correct_m2(pred, gold):
    if pred is None:
        return False
    try:
        return bool(verify(parse('$' + gold + '$'), parse('$' + pred + '$')))
    except Exception:
        return False


def same_m2(p, q):
    try:
        return bool(verify(parse('$' + p + '$'), parse('$' + q + '$')))
    except Exception:
        return False


# ---------- pass@1 and cons@k ----------
def pass_at_1_m1(groups):
    """The paper's formula, page 12: pass@1 = (1/k) * sum_i p_i for each question, then the mean over questions."""
    per_q = []
    for g in groups:                              # g: the k correctness values (0 or 1) of one question
        per_q.append(sum(g) / len(g))
    return sum(per_q) / len(per_q)


def pass_at_1_m2(groups):
    return float(np.mean(np.array(groups, dtype=float)))      # [questions, k] -> one number


def cons_m1(answers, gold):
    """Majority vote: the most common normalised answer wins (ties: the one seen first). None votes do not count."""
    votes = [normalise(a) for a in answers if a is not None]
    if not votes:
        return False, None
    counts = {}
    for v in votes:
        counts[v] = counts.get(v, 0) + 1
    best = max(counts, key=lambda v: (counts[v], -votes.index(v)))
    return best == normalise(gold), best


def cons_m2(answers, gold):
    """Majority vote where two answers are one class if math-verify says they are equal."""
    classes = []                                  # [representative, count]
    for a in answers:
        if a is None:
            continue
        for c in classes:
            if same_m2(c[0], a):
                c[1] += 1
                break
        else:
            classes.append([a, 1])
    if not classes:
        return False, None
    best = max(classes, key=lambda c: c[1])[0]     # max keeps the first class on a tie
    return correct_m2(best, gold), best


# ---------- reflection words ----------
def reflect_m1(text):
    t = text.lower()
    return {k: len(re.findall(p, t)) for k, p in REFLECT.items()}


def reflect_m2(text):
    """Split into lower-case word tokens and count with Counter; two-word phrases from pairs of tokens."""
    toks = re.findall(r"[a-z]+(?:-[a-z]+)?", text.lower())
    c1 = Counter(toks)
    c2 = Counter(zip(toks, toks[1:], toks[2:]))
    return {'wait': c1['wait'], 'let me check': c2[('let', 'me', 'check')], 'alternatively': c1['alternatively'],
            're-evaluate': sum(n for w, n in c1.items() if w.startswith(('reevaluat', 're-evaluat')))}


def hist_m1(values, edges):
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[-1]):
                counts[i] += 1
                break
    return counts


CJK = re.compile(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]')

if __name__ == '__main__':
    out = {}
    # Why Figure 2 averages 16 samples: the noise of a score on 30 AIME questions at a true rate of 50%
    # The 30 questions are fixed; the samples of one question are independent draws. Each question's
    # Bernoulli variance p(1 - p) is at most 0.25, so these are upper bounds on the decoding (sampling) noise.
    # They say nothing about how the score would move on a different set of 30 questions.
    se1, se16 = (0.25 / 30) ** 0.5, (0.25 / (30 * 16)) ** 0.5
    log(f'AIME decoding noise on 30 fixed questions (each question p(1-p) <= 0.25): 1 sample per question, '
        f'standard error at most {100 * se1:.1f} points; 16 independent samples per question, at most {100 * se16:.1f} points '
        f'(an upper bound; it leaves out the noise of which 30 questions were chosen)')
    gen ={t: load(f'r1_part4_gen_{t}.jsonl') for t in TAGS}
    k = max(r['sample'] for r in gen['distill']) + 1
    log(f'{len(PROBLEMS)} MATH-500 problems x {k} samples, temperature 0.6, top-p 0.95, max_new_tokens 2048')
    for t in TAGS:
        rows = gen[t]
        for r in rows:
            gold = PROBLEMS[r['idx']]['answer']
            r['pred'] = None if r['truncated'] else extract(final_part(r['text'], t))   # a cut-off answer has no final answer
            r['m1'] = correct_m1(r['pred'], gold)
            r['m2'] = correct_m2(r['pred'], gold)
            r['refl'] = reflect_m1(r['text'])
            r['refl2'] = reflect_m2(r['text'])
        agree = sum(r['m1'] == r['m2'] for r in rows)
        log(f'\n== {NAMES[t]}: {len(rows)} answers')
        log(f'grading: ours and math-verify agree on {agree}/{len(rows)}; '
            f'ours says {sum(r["m1"] for r in rows)} correct, math-verify {sum(r["m2"] for r in rows)} (we use math-verify)')
        for r in rows:
            if r['m1'] != r['m2']:
                log(f'   differ: problem {r["idx"]} gold {PROBLEMS[r["idx"]]["answer"]!r} pred {r["pred"]!r} '
                    f'ours {r["m1"]} math-verify {r["m2"]}')
        trunc = sum(r['truncated'] for r in rows)
        nobox = sum(r['pred'] is None for r in rows)
        log(f'truncated at 2048 tokens: {trunc}/{len(rows)} (scored as wrong); finished but no final answer found: {nobox - trunc}/{len(rows)}')
        fin = [r for r in rows if not r['truncated']]
        log(f'answers that finished inside 2048 tokens: {len(fin)}/{len(rows)}, of which correct: '
            f'{sum(r["m2"] for r in fin)}/{len(fin)} = {sum(r["m2"] for r in fin) / max(1, len(fin)):.3f}')
        groups, cons1, cons2, anyc = [], [], [], []
        for idx in PROBLEMS:
            g = sorted((r for r in rows if r['idx'] == idx), key=lambda r: r['sample'])
            groups.append([int(r['m2']) for r in g])
            c1, b1 = cons_m1([r['pred'] for r in g], PROBLEMS[idx]['answer'])
            c2, b2 = cons_m2([r['pred'] for r in g], PROBLEMS[idx]['answer'])
            cons1.append(c1); cons2.append(c2); anyc.append(any(r['m2'] for r in g))
        p1, p2 = pass_at_1_m1(groups), pass_at_1_m2(groups)
        log(f'pass@1 (mean of {k} samples per problem): Method 1 {p1:.4f}, Method 2 numpy {p2:.4f}, '
            f'agree: {abs(p1 - p2) < 1e-12}')
        log(f'cons@{k} (majority vote): Method 1 {np.mean(cons1):.4f} ({sum(cons1)}/{len(cons1)}), '
            f'Method 2 {np.mean(cons2):.4f} ({sum(cons2)}/{len(cons2)})')
        log(f'pass@{k} (at least one of {k} correct): {np.mean(anyc):.4f} ({sum(anyc)}/{len(anyc)})')
        per_q = ' '.join(f'{sum(g)}' for g in groups)
        log(f'correct answers out of {k}, per problem: {per_q}')
        L = np.array([r['n_tokens'] for r in rows])
        Lc = np.array([r['n_tokens'] for r in rows if r['m2']])
        Lw = np.array([r['n_tokens'] for r in rows if not r['m2']])
        Lw_fin = np.array([r['n_tokens'] for r in rows if not r['m2'] and not r['truncated']])
        log(f'length (new tokens): mean {L.mean():.0f}, median {np.median(L):.0f}; correct answers mean '
            f'{Lc.mean() if len(Lc) else 0:.0f} (n={len(Lc)}), wrong answers mean {Lw.mean() if len(Lw) else 0:.0f} '
            f'(n={len(Lw)}), wrong and finished mean {Lw_fin.mean() if len(Lw_fin) else 0:.0f} (n={len(Lw_fin)})')
        edges = list(range(0, 2049, 256))
        hc1, hw1 = hist_m1(Lc, edges), hist_m1(Lw, edges)
        hc2, hw2 = np.histogram(Lc, bins=edges)[0].tolist(), np.histogram(Lw, bins=edges)[0].tolist()
        log(f'length histogram, bins of 256 tokens: correct {hc1}, wrong {hw1}; numpy agrees: {hc1 == hc2 and hw1 == hw2}')
        tot1 = {w: sum(r['refl'][w] for r in rows) for w in REFLECT}
        tot2 = {w: sum(r['refl2'][w] for r in rows) for w in REFLECT}
        log(f'reflection words, total over {len(rows)} answers: Method 1 {tot1}')
        log(f'                                              Method 2 {tot2}, agree: {tot1 == tot2}')
        per_ans = {w: tot1[w] / len(rows) for w in REFLECT}
        per_k = {w: 1000 * tot1[w] / L.sum() for w in REFLECT}
        has_wait = sum(r['refl']['wait'] > 0 for r in rows)
        log('per answer: ' + ', '.join(f'{w} {v:.2f}' for w, v in per_ans.items()))
        log('per 1,000 tokens: ' + ', '.join(f'{w} {v:.2f}' for w, v in per_k.items()))
        log(f'answers with at least one "wait": {has_wait}/{len(rows)}')
        wc = [r['refl']['wait'] for r in rows if r['m2']]
        ww = [r['refl']['wait'] for r in rows if not r['m2']]
        log(f'"wait" per answer: correct answers {np.mean(wc) if wc else 0:.2f}, wrong answers {np.mean(ww) if ww else 0:.2f}')
        cjk = sum(bool(CJK.search(r['text'])) for r in rows)
        log(f'answers with any Chinese, Japanese or Korean character: {cjk}/{len(rows)}')
        # GRPO's view (equation 3): inside each group, which answers would get a positive advantage?
        pos, neg = [], []
        for idx in PROBLEMS:
            g = [r for r in rows if r['idx'] == idx]
            rew = np.array([float(r['m2']) for r in g])
            if rew.std() == 0:
                continue                               # all right or all wrong: every advantage is 0
            A = (rew - rew.mean()) / rew.std()
            pos += [r['n_tokens'] for r, a in zip(g, A) if a > 0]
            neg += [r['n_tokens'] for r, a in zip(g, A) if a < 0]
        log(f'GRPO view, groups with mixed results: positive-advantage answers mean length {np.mean(pos) if pos else 0:.0f} '
            f'(n={len(pos)}), negative-advantage answers {np.mean(neg) if neg else 0:.0f} (n={len(neg)})')
        out[t] = dict(pass1=p2, cons=float(np.mean(cons2)), cons_m1=float(np.mean(cons1)), passk=float(np.mean(anyc)),
                      k=k, n=len(rows), truncated=trunc, nobox=nobox, len_mean=float(L.mean()),
                      len_correct=float(Lc.mean()) if len(Lc) else 0, len_wrong=float(Lw.mean()) if len(Lw) else 0,
                      hist_edges=edges, hist_correct=hc2, hist_wrong=hw2, reflect_total=tot1, reflect_per_answer=per_ans,
                      reflect_per_1k=per_k, has_wait=has_wait, cjk=cjk, per_problem=[sum(g) for g in groups],
                      grpo_pos_len=float(np.mean(pos)) if pos else 0, grpo_neg_len=float(np.mean(neg)) if neg else 0)

    # ---------- the Table 3 question ----------
    gold = r'\frac{\sqrt{4a-3}-1}{2}'
    for t in TAGS:
        p = os.path.join(R, f'r1_part4_aha_{t}.jsonl')
        if not os.path.exists(p):
            continue
        rows = load(os.path.basename(p))
        log(f'\n== Table 3 question, {NAMES[t]}: {len(rows)} answers')
        for r in rows:
            r['pred'] = None if r['truncated'] else extract(final_part(r['text'], t))
            ok = correct_m2(r['pred'], gold)
            log(f'sample {r["sample"]}: {r["n_tokens"]} tokens{" (truncated)" if r["truncated"] else ""}, '
                f'"wait" x{reflect_m1(r["text"])["wait"]}, answer {r["pred"]!r}, correct {ok}')
        out[f'aha_{t}'] = [dict(sample=r['sample'], n_tokens=r['n_tokens'], truncated=r['truncated'],
                                pred=r['pred'], wait=reflect_m1(r['text'])['wait']) for r in rows]
    common.save('r1_part4', out)

    # candidate "Wait" passages for the page: each "Wait" in a correct distill answer, with context
    with open(os.path.join(R, 'r1_part4_passages.txt'), 'w') as f:
        for r in gen['distill']:
            if not r['m2']:
                continue
            for m in re.finditer(r'\bWait\b', r['text']):
                f.write(f'### problem {r["idx"]} sample {r["sample"]} offset {m.start()}\n')
                f.write(r['text'][max(0, m.start() - 500):m.start() + 700] + '\n\n')

    # the passage quoted on the page: problem 20, sample 2 of the distill, copied verbatim
    r = next(r for r in gen['distill'] if r['idx'] == 20 and r['sample'] == 2)
    i, j = r['text'].index('Another thought:'), r['text'].index('Wait, but if I try')
    log(f'\n== a real "Wait" (distill, problem 20: {PROBLEMS[20]["problem"]!r}, sample 2, '
        f'{r["n_tokens"]} tokens, final answer {r["pred"]!r}, correct {r["m2"]})')
    log(r['text'][i:j].strip())
