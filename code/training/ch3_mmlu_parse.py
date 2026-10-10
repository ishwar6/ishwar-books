"""Chapter 3: how the answer-extraction rule changes a generative MMLU score (no model needed; reads ch3_mmlu.json).
loose   the first standalone capital A, B, C or D anywhere in the text (what ch3_mmlu.py used)
strict  only an explicit answer: a letter at the very start of the reply, or "answer is X", or "Answer: X"
Also counts replies that were cut off by the 48-token budget before giving any letter."""
import json, re
from common import Log, save

log = Log('ch3_mmlu_parse')
M = json.load(open('results/ch3_mmlu.json'))
L = 'ABCD'
STRICT = [r'\A\s*\(?\**([ABCD])\b', r'answer is[:\s]*\(?\**([ABCD])\b', r'Answer:\s*\(?\**([ABCD])\b']   # start of reply first


def strict(t):
    for p in STRICT:
        m = re.search(p, t)
        if m:
            return L.index(m.group(1))
    return -1


def loose(t):
    m = re.search(r'\b([ABCD])\b', t)
    return L.index(m.group(1)) if m else -1


res = {}
for name in ['base', 'instruct']:
    texts = M['models'][name]['generate']['text']
    ans = M['answer']
    lo = [loose(t) for t in texts]
    st = [strict(t) for t in texts]
    acc_l = sum(p == a for p, a in zip(lo, ans)) / len(ans)
    acc_s = sum(p == a for p, a in zip(st, ans)) / len(ans)
    article = sum(1 for t, p in zip(texts, lo) if p == 0 and re.search(r'\bA [a-z]', t) and re.search(r'\bA\b', t).start() == re.search(r'\bA [a-z]', t).start())
    res[name] = dict(loose=acc_l, strict=acc_s, loose_none=lo.count(-1), strict_none=st.count(-1), article_A=article)
    log(f'{name:8s} loose rule {100 * acc_l:5.1f}% (no letter {lo.count(-1):3d})   strict rule {100 * acc_s:5.1f}% (no letter {st.count(-1):3d})'
        f'   loose rule took the word "A" (as in "A cyclic group") as the answer: {article}')
ex = next(t for t, p in zip(M['models']['instruct']['generate']['text'], [loose(t) for t in M['models']['instruct']['generate']['text']]) if p >= 0 and strict(t) < 0)
log('\nan Instruct reply where the loose rule finds a letter but the strict rule finds no answer:')
log('  ' + repr(ex[:200]))
res['example'] = ex
save('ch3_mmlu_parse', res)
